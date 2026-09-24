"""Stage-5 QA validator (evaluate_script / verdict aggregation) tests."""

from __future__ import annotations

import re
from dataclasses import replace

import pytest

from agents.qa.script.models import ClaimCoverage
from agents.qa.script.validator import (
    VERDICT_PASS,
    VERDICT_REJECT,
    VERDICT_REVISE,
    aggregate_verdict,
    evaluate_script,
)
from agents.script.models import ScriptLine
from script_qa_test_utils import (
    FIXED_AT,
    octopus_ok_lines,
    make_octopus_pair,
    make_script,
    run_qa_eval,
)


def _errors_of(qa_result):
    return qa_result.issues


def test_generate_then_eval_passes():
    result, package = make_octopus_pair()
    from script_qa_test_utils import generate_and_eval

    qa = generate_and_eval(result, package)
    assert qa.overall_verdict == VERDICT_PASS
    assert not qa.issues


def test_valid_script_passes_and_records_checks():
    result, package = make_octopus_pair()
    qa = run_qa_eval(
        make_script(result, lines=octopus_ok_lines(result)),
        result,
        package,
    )
    assert qa.overall_verdict == VERDICT_PASS
    assert qa.score == 100.0
    assert len(qa.checks) == 13
    assert all(record.errors == 0 and record.warnings == 0 for record in qa.checks)
    assert qa.approved_claim_coverage.coverage_ratio == 1.0


def test_issue_ids_assigned_in_stable_order():
    result, package = make_octopus_pair()
    script = make_script(result, lines=[
        {"text": "Octopuses have nine legs and glow purple plasma.",
         "purpose": "fact", "claim_refs": (result.approved_claims[0],)},
        {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
    ])
    qa = run_qa_eval(script, result, package)
    ids = [issue.issue_id for issue in qa.all_issues()]
    assert ids
    assert all(re.fullmatch(r"qa_\d{2}", issue_id) for issue_id in ids)
    assert len(set(ids)) == len(ids)
    assert min(ids) >= "qa_00"
    assert qa.overall_verdict == VERDICT_REJECT  # invented-detail fixture


def test_verdict_aggregation_table():
    def issue(severity, gate):
        from agents.qa.script.models import QAIssue

        return QAIssue(category="structure", severity=severity, gate=gate, message=f"{severity}-{gate}")

    assert aggregate_verdict([issue("INFO", "soft")]) == VERDICT_PASS
    assert aggregate_verdict([issue("WARNING", "soft")]) == VERDICT_PASS
    assert aggregate_verdict([issue("ERROR", "soft")]) == VERDICT_REVISE
    assert aggregate_verdict([issue("ERROR", "hard")]) == VERDICT_REJECT
    assert aggregate_verdict([issue("ERROR", "hard"), issue("ERROR", "soft")]) == VERDICT_REJECT


def test_soft_error_gives_revise():
    result, package = make_octopus_pair()
    # Enough lines to stay in the usable duration window while still missing an
    # ending and a hook - only soft errors, so the verdict must be REVISE.
    lines = octopus_ok_lines(result)[:4][:-1] + [
        {"text": "There is one more thing worth noting.", "purpose": "transition",
         "claim_refs": ()},
        {"text": "And that detail matters a lot in the end.", "purpose": "transition",
         "claim_refs": ()},
    ]
    qa = run_qa_eval(
        make_script(result, lines=lines, estimated_duration=40.0),
        result,
        package,
    )
    assert qa.overall_verdict == VERDICT_REVISE
    assert qa.issues
    assert all(issue.gate == "soft" for issue in qa.issues)


@pytest.mark.parametrize(
    "story",
    [
        "no_traceability",
        "rejected_ref",
        "review_ref",
        "unknown_ref",
        "invented",
        "certainty",
        "hedge_drop",
        "contradiction",
        "empty",
        "package_reject",
    ],
)
def test_hard_gates_reject(story):
    result, package = make_octopus_pair()
    approved = result.approved_claims
    smartest = package.claims[-2].claim_id
    tank = package.claims[-1].claim_id
    if story == "no_traceability":
        script = make_script(result, lines=[
            {"text": octopus_ok_lines(result)[2]["text"], "purpose": "fact", "claim_refs": ()},
            {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
        ])
    elif story == "rejected_ref":
        script = make_script(result, lines=[
            {"text": "Octopuses can escape tanks.", "purpose": "fact", "claim_refs": (tank,)},
            {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
        ])
    elif story == "review_ref":
        script = make_script(result, lines=[
            {"text": "Octopuses are smart.", "purpose": "fact", "claim_refs": (smartest,)},
            {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
        ])
    elif story == "unknown_ref":
        script = make_script(result, lines=[
            {"text": "Octopuses have three hearts.", "purpose": "fact",
             "claim_refs": ("cl_missing",)},
            {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
        ])
    elif story == "invented":
        script = make_script(result, lines=[
            {"text": "Octopuses have nine hearts that glow purple.",
             "purpose": "fact", "claim_refs": (approved[0],)},
            {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
        ])
    elif story == "certainty":
        script = make_script(result, lines=[
            {"text": "It is definitely the case that " +
             octopus_ok_lines(result)[2]["text"],
             "purpose": "fact", "claim_refs": (approved[0],)},
            {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
        ])
    elif story == "hedge_drop":
        hedged = "Scientists suggest that octopuses may use tools to open jars."
        from script_qa_test_utils import make_simple_pair
        from agents.research.models import claim_id_for

        result, package = make_simple_pair(claims=[(hedged, "inference")], verdicts={hedged: "PASS"})
        script = make_script(result, lines=[
            {"text": "Scientists suggest that octopuses use tools to open jars.",
             "purpose": "fact", "claim_refs": (claim_id_for(hedged),)},
            {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
        ])
    elif story == "contradiction":
        script = make_script(result, lines=[
            {"text": octopus_ok_lines(result)[2]["text"].replace("have three", "do not have three"),
             "purpose": "fact", "claim_refs": (approved[0],)},
            {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
        ])
    elif story == "empty":
        script = make_script(result, lines=[], used=(), unused=())
    else:  # package_reject
        result, package = make_octopus_pair(package_verdict="REJECT")
        script = make_script(result, lines=octopus_ok_lines(result))

    qa = evaluate_script(
        script,
        result,
        {c.claim_id: c for c in package.claims},
        run_id="script-test",
        created_at=FIXED_AT,
    )
    assert qa.overall_verdict == VERDICT_REJECT, story
    assert any(issue.gate == "hard" for issue in qa.issues), story


@pytest.mark.parametrize(
    "story",
    [
        "package_mismatch",
        "missing_package_binding",
        "accounting_incomplete",
        "accounting_foreign_unused",
        "duplicate_line_ids",
        "unknown_purpose",
    ],
)
def test_binding_accounting_contract_gates_reject(story):
    """Fail-closed gates: binding, accounting, and malformed script never PASS."""
    result, package = make_octopus_pair()
    approved = result.approved_claims
    lines = octopus_ok_lines(result)
    script = make_script(result, lines=lines)
    if story == "package_mismatch":
        script = replace(script, package_id="rp_other_package")
    elif story == "missing_package_binding":
        script = replace(script, package_id="")
    elif story == "accounting_incomplete":
        base = make_script(
            result,
            lines=[
                {"text": lines[2]["text"], "purpose": "fact", "claim_refs": (approved[0],)},
                {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
            ],
            used=(approved[0],),
        )
        script = replace(base, unused_approved_claim_ids=(approved[2],))
    elif story == "accounting_foreign_unused":
        base = make_script(
            result,
            lines=[
                {"text": lines[2]["text"], "purpose": "fact", "claim_refs": (approved[0],)},
                {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
            ],
            used=(approved[0],),
        )
        script = replace(base, unused_approved_claim_ids=("cl_ghost",))
    elif story == "duplicate_line_ids":
        base = make_script(
            result,
            lines=[
                {"text": lines[2]["text"], "purpose": "fact", "claim_refs": (approved[0],)},
                {"text": lines[6]["text"], "purpose": "conclusion", "claim_refs": (approved[0],)},
            ],
            used=(approved[0],),
        )
        script = replace(
            base,
            lines=(
                ScriptLine("ln_01", lines[2]["text"], (approved[0],), "fact"),
                ScriptLine("ln_01", lines[6]["text"], (approved[0],), "conclusion"),
            ),
        )
    else:  # unknown_purpose
        base = make_script(
            result,
            lines=[
                {"text": lines[2]["text"], "purpose": "fact", "claim_refs": (approved[0],)},
                {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
            ],
            used=(approved[0],),
        )
        script = replace(
            base,
            lines=(
                ScriptLine("ln_01", lines[2]["text"], (approved[0],), "fact"),
                ScriptLine("ln_02", "Thanks for watching.", (), "bogus_purpose"),
            ),
        )

    qa = evaluate_script(
        script,
        result,
        {c.claim_id: c for c in package.claims},
        run_id="script-test",
        created_at=FIXED_AT,
    )
    assert qa.overall_verdict == VERDICT_REJECT, story
    assert any(issue.gate == "hard" for issue in qa.issues), story


def test_coverage_reported_in_result():
    result, package = make_octopus_pair()
    qa = run_qa_eval(
        make_script(result, lines=[
            {"text": octopus_ok_lines(result)[2]["text"], "purpose": "fact",
             "claim_refs": (result.approved_claims[0],)},
            {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
        ]),
        result,
        package,
    )
    coverage: ClaimCoverage = qa.approved_claim_coverage
    assert coverage is not None
    assert coverage.coverage_ratio == pytest.approx(1 / 3, abs=0.001)
    assert len(coverage.unused_approved_claim_ids) == 2


def test_no_network_access(capfd):
    """QA must never open sockets: monkeypatch the live provider to raise."""
    result, package = make_octopus_pair()

    def boom(*args, **kwargs):
        raise AssertionError("QA opened the network")

    import socket

    original = socket.socket
    socket.socket = boom
    try:
        qa = run_qa_eval(
            make_script(result, lines=octopus_ok_lines(result)), result, package
        )
    finally:
        socket.socket = original
    assert qa.overall_verdict == VERDICT_PASS


def test_deterministic_across_runs():
    result, package = make_octopus_pair()
    script = make_script(result, lines=octopus_ok_lines(result))
    first = run_qa_eval(script, result, package)
    second = run_qa_eval(script, result, package)
    assert first.to_json() == second.to_json()
    assert first.overall_verdict == second.overall_verdict


def test_artifact_provenance_and_field_values():
    result, package = make_octopus_pair()
    script = make_script(result, lines=octopus_ok_lines(result))
    qa = run_qa_eval(script, result, package)
    assert qa.produced_by["model"] == "ScriptQAValidator"
    assert set(qa.produced_by) == {"prompt_version", "provider", "model"}
    assert qa.provenance is not None
    assert qa.provenance.model == "ScriptQAValidator"
    assert qa.provenance.source == "DeterministicScriptQA"
    assert qa.provenance.prompt_version == "script_qa.v1"
    assert qa.schema == "ScriptQAResult"
    assert qa.schema_version == 1
    assert qa.stage == "script_qa"
    assert qa.script_id == script.script_id
    assert qa.result_id == result.result_id