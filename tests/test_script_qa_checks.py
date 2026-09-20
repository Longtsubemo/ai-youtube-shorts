"""Stage-5 QA checks1 (13 check functions) unit tests."""

from __future__ import annotations

from agents.qa.script.checks import CHECKS
from agents.qa.script.models import ScriptQAConfig
from agents.research.models import claim_id_for
from script_qa_test_utils import (
    OCTOPUS_BLUE,
    OCTOPUS_SMARTEST,
    OCTOPUS_SWIM,
    OCTOPUS_THREE_HEARTS,
    FIXED_AT,
    octopus_ok_lines,
    make_critical_pair,
    make_octopus_pair,
    make_script,
    make_simple_pair,
    run_checks_only,
)


def _issues_by_category(issues):
    result: dict[str, list] = {}
    for issue in issues:
        result.setdefault(issue.category, []).append(issue)
    return result


def _octopus_script(result, package, **overrides):
    return make_script(
        result,
        lines=octopus_ok_lines(result),
        used=result.approved_claims,
        unused=(),
        **overrides,
    )


def test_registry_has_13_unique_deterministic_checks():
    names = [check.name for check in CHECKS]
    assert len(names) == 13
    assert len(set(names)) == 13
    assert CHECKS[0].name == "factual_fidelity"
    assert CHECKS[1].name == "claim_traceability"
    hard = {check.name for check in CHECKS if check.hard_gate}
    assert hard == {
        "factual_fidelity", "claim_traceability", "length_duration",
        "structure", "uncertainty_preservation",
    }
    for check in CHECKS:
        assert check.dimension in {
            "factual_fidelity", "traceability", "coverage", "uncertainty",
            "clarity", "structure", "pacing", "hook", "repetition", "format",
        }


# --------------------------------------------------------------------------- #
# 1. Factual fidelity
# --------------------------------------------------------------------------- #


def test_fidelity_accepts_claims_wording():
    result, package = make_octopus_pair()
    script = _octopus_script(result, package)
    issues = run_checks_only(script, result, package)
    assert not _issues_by_category(issues).get("factual_fidelity")


def test_fidelity_rejects_invented_detail():
    result, package = make_octopus_pair()
    script = make_script(
        result,
        lines=[
            {"text": "Octopuses have nine legs and glow purple plasma.",
             "purpose": "fact", "claim_refs": (result.approved_claims[0],)},
            {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
        ],
    )
    issues = _issues_by_category(run_checks_only(script, result, package))
    errors = [i for i in issues.get("factual_fidelity", []) if i.gate == "hard"]
    assert errors and "invents wording" in errors[0].message


def test_fidelity_rejects_material_contradiction():
    result, package = make_octopus_pair()
    script = make_script(
        result,
        lines=[
            {"text": "Octopuses do not have three hearts.",
             "purpose": "fact", "claim_refs": (result.approved_claims[0],)},
            {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
        ],
    )
    issues = _issues_by_category(run_checks_only(script, result, package))
    errors = [i for i in issues.get("factual_fidelity", []) if i.severity == "ERROR"]
    assert errors and "materially contradicts" in errors[0].message


# --------------------------------------------------------------------------- #
# 2. Claim traceability
# --------------------------------------------------------------------------- #


def test_traceability_missing_refs_on_factual_line():
    result, package = make_octopus_pair()
    script = make_script(
        result,
        lines=[
            {"text": OCTOPUS_THREE_HEARTS, "purpose": "fact", "claim_refs": ()},
            {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
        ],
    )
    issues = _issues_by_category(run_checks_only(script, result, package))
    errors = [i for i in issues.get("claim_traceability", []) if i.severity == "ERROR"]
    assert errors and any("no claim traceability" in i.message for i in errors)


def test_traceability_unknown_claim_ref():
    result, package = make_octopus_pair()
    script = make_script(
        result,
        lines=[
            {"text": "Octopuses have three hearts.", "purpose": "fact",
             "claim_refs": ("cl_does_not_exist",)},
            {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
        ],
    )
    issues = _issues_by_category(run_checks_only(script, result, package))
    assert any(
        i.severity == "ERROR" and "unknown claim" in i.message
        for i in issues.get("claim_traceability", [])
    )


def test_traceability_rejected_claim_ref():
    result, package = make_octopus_pair()
    rejected = package.claims[-1].claim_id  # OCTOPUS_TANK (REJECT)
    script = make_script(result, lines=[
        {"text": "Octopuses can escape tanks.", "purpose": "fact", "claim_refs": (rejected,)},
        {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
    ])
    issues = _issues_by_category(run_checks_only(script, result, package))
    assert any(
        i.severity == "ERROR" and "REJECTED claim" in i.message
        for i in issues.get("claim_traceability", [])
    )


def test_traceability_review_claim_ref():
    result, package = make_octopus_pair()
    review = package.claims[-2].claim_id  # OCTOPUS_SMARTEST (REVIEW)
    script = make_script(result, lines=[
        {"text": "Octopuses are smart.", "purpose": "fact", "claim_refs": (review,)},
        {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
    ])
    issues = _issues_by_category(run_checks_only(script, result, package))
    assert any(
        i.severity == "ERROR" and "REVIEW (unapproved)" in i.message
        for i in issues.get("claim_traceability", [])
    )


def test_traceability_corrupt_used_unused():
    result, package = make_octopus_pair()
    script = make_script(
        result,
        lines=[
            {"text": OCTOPUS_THREE_HEARTS, "purpose": "fact",
             "claim_refs": (result.approved_claims[0],)},
            {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
        ],
        used=(result.approved_claims[0], "cl_ghost"),
    )
    issues = _issues_by_category(run_checks_only(script, result, package))
    assert any(
        i.severity == "ERROR" and "used_claim_ids must equal" in i.message
        for i in issues.get("claim_traceability", [])
    )


def test_traceability_package_reject_is_fatal():
    result, package = make_octopus_pair(package_verdict="REJECT")
    script = _octopus_script(result, package)
    issues = _issues_by_category(run_checks_only(script, result, package))
    assert any(
        i.severity == "ERROR" and "package verdict REJECT" in i.message
        for i in issues.get("claim_traceability", [])
    )


def test_traceability_missing_provenance_is_fatal():
    result, package = make_octopus_pair()
    script = _octopus_script(result, package, provenance=None)
    issues = _issues_by_category(run_checks_only(script, result, package))
    assert any(
        i.severity == "ERROR" and "no provenance" in i.message
        for i in issues.get("claim_traceability", [])
    )


# --------------------------------------------------------------------------- #
# 3. Claim coverage
# --------------------------------------------------------------------------- #


def test_coverage_full_use_has_no_issue():
    result, package = make_octopus_pair()
    script = _octopus_script(result, package)
    issues = run_checks_only(script, result, package)
    assert not _issues_by_category(issues).get("claim_coverage")


def test_coverage_noncritical_unused_is_info():
    result, package = make_octopus_pair()
    script = make_script(
        result,
        lines=[
            {"text": OCTOPUS_THREE_HEARTS, "purpose": "fact",
             "claim_refs": (result.approved_claims[0],)},
            {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
        ],
    )
    issues = _issues_by_category(run_checks_only(script, result, package))
    infos = [i for i in issues.get("claim_coverage", []) if i.severity == "INFO"]
    assert infos and "is not used" in infos[0].message


def test_coverage_critical_unused_is_error_by_default():
    result, package = make_critical_pair()
    script = make_script(
        result,
        lines=[
            # Uses first + third approved claims; drops critical OCTOPUS_SWIM.
            {"text": OCTOPUS_THREE_HEARTS, "purpose": "fact",
             "claim_refs": (result.approved_claims[0],)},
            {"text": OCTOPUS_BLUE, "purpose": "explanation",
             "claim_refs": (result.approved_claims[2],)},
            {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
        ],
    )
    issues = _issues_by_category(run_checks_only(script, result, package))
    errors = [i for i in issues.get("claim_coverage", []) if i.severity == "ERROR"]
    assert errors and "critical approved claim" in errors[0].message


def test_coverage_critical_unused_downgrades_to_warning_when_configured():
    result, package = make_critical_pair()
    script = make_script(
        result,
        lines=[
            {"text": OCTOPUS_THREE_HEARTS, "purpose": "fact",
             "claim_refs": (result.approved_claims[0],)},
            {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
        ],
    )
    config = ScriptQAConfig(critical_unused_is_error=False)
    issues = _issues_by_category(run_checks_only(script, result, package, config=config))
    warnings = [i for i in issues.get("claim_coverage", []) if i.severity == "WARNING"]
    assert warnings and "critical approved claim" in warnings[0].message


def test_approved_only_passes_traceability_fidelity():
    """The whitelist-only rule: referencing a review/rejected claim anywhere
    fails QA even though the claim text is 'about' the topic."""
    result, package = make_octopus_pair()
    script = _octopus_script(result, package)
    issues = run_checks_only(script, result, package)
    assert not _issues_by_category(issues).get("factual_fidelity")
    assert not [i for i in issues if i.category == "claim_traceability" and i.severity == "ERROR"]


# --------------------------------------------------------------------------- #
# 4. Length / duration
# --------------------------------------------------------------------------- #


def test_duration_in_target_window_ok():
    result, package = make_octopus_pair()
    script = _octopus_script(result, package, estimated_duration=45.0)
    issues = _issues_by_category(run_checks_only(script, result, package))
    assert not issues.get("length_duration")


def test_duration_preferred_range_edges_warn():
    result, package = make_octopus_pair()
    for duration, marker in ((25.0, "below the preferred"), (75.0, "above the preferred")):
        script = _octopus_script(result, package, estimated_duration=duration)
        issues = _issues_by_category(run_checks_only(script, result, package))
        warnings = [i for i in issues.get("length_duration", []) if i.severity == "WARNING"]
        assert warnings and marker in warnings[0].message


def test_duration_hard_bounds_error():
    result, package = make_octopus_pair()
    for duration in (8.0, 240.0):
        script = _octopus_script(result, package, estimated_duration=duration)
        issues = _issues_by_category(run_checks_only(script, result, package))
        errors = [i for i in issues.get("length_duration", []) if i.gate == "hard"]
        assert errors


# --------------------------------------------------------------------------- #
# 5. Pacing
# --------------------------------------------------------------------------- #


def test_pacing_long_line_error_then_warning():
    result, package = make_octopus_pair()
    long_words = "extra " * 46  # well past the 45-word hard pacing limit
    script = make_script(result, lines=[
        {"text": "Octopuses " + long_words + "hearts.", "purpose": "fact",
         "claim_refs": (result.approved_claims[0],)},
        {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
    ])
    issues = _issues_by_category(run_checks_only(script, result, package))
    pacing = [i for i in issues.get("pacing", []) if i.severity == "ERROR"]
    assert pacing and "excessively long" in pacing[0].message


def test_pacing_longish_line_warns():
    result, package = make_octopus_pair()
    script = make_script(result, lines=[
        {"text": (OCTOPUS_THREE_HEARTS + " " + OCTOPUS_SWIM).replace(":", " "), "purpose": "fact",
         "claim_refs": (result.approved_claims[0],)},
        {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
    ])
    issues = _issues_by_category(run_checks_only(script, result, package))
    warnings = [i for i in issues.get("pacing", []) if i.severity == "WARNING" and "long" in i.message]
    assert warnings


def test_pacing_multi_sentence_line_warns():
    result, package = make_octopus_pair()
    text = ("Octopuses are smart. They solve puzzles. They remember the people "
            "who feed them and even learn to open tricky latch mechanisms.")
    assert len(text.split()) > 20
    script = make_script(result, lines=[
        {"text": text, "purpose": "explanation", "claim_refs": (result.approved_claims[0],)},
        {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
    ])
    issues = _issues_by_category(run_checks_only(script, result, package))
    assert any(
        i.severity == "WARNING" and "packs multiple sentences" in i.message
        for i in issues.get("pacing", [])
    )


def test_pacing_excessive_setup_warns():
    result, package = make_octopus_pair()
    opening = "Before we begin, " + "imagine " * 36 + "now."
    script = make_script(result, lines=[
        {"text": opening, "purpose": "context", "claim_refs": ()},
        {"text": OCTOPUS_THREE_HEARTS, "purpose": "fact",
         "claim_refs": (result.approved_claims[0],)},
        {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
    ])
    issues = _issues_by_category(run_checks_only(script, result, package))
    assert any(
        i.severity == "WARNING" and "excessive setup" in i.message
        for i in issues.get("pacing", [])
    )


# --------------------------------------------------------------------------- #
# 6. Clarity
# --------------------------------------------------------------------------- #


def test_clarity_long_sentence_warns():
    result, package = make_octopus_pair()
    text = ("Octopuses have a truly remarkable nervous system that allows them "
            "to solve complex puzzles quickly, remember the people they meet, "
            "and coordinate every single one of their flexible arms independently.")
    assert len(text.split()) > 25
    script = make_script(result, lines=[
        {"text": text, "purpose": "fact", "claim_refs": (result.approved_claims[0],)},
        {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
    ])
    issues = _issues_by_category(run_checks_only(script, result, package))
    assert any(
        i.severity == "WARNING" and "long sentence" in i.message
        for i in issues.get("clarity", [])
    )


def test_clarity_acronym_warns():
    result, package = make_octopus_pair()
    script = make_script(result, lines=[
        {"text": "NASA studies deep ocean life forms.", "purpose": "fact",
         "claim_refs": (result.approved_claims[0],)},
        {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
    ])
    issues = _issues_by_category(run_checks_only(script, result, package))
    assert any(
        i.severity == "WARNING" and "acronym" in i.message
        for i in issues.get("clarity", [])
    )


# --------------------------------------------------------------------------- #
# 7. Repetition
# --------------------------------------------------------------------------- #


def test_repetition_identical_factual_lines_warn():
    result, package = make_octopus_pair()
    script = make_script(result, lines=[
        {"text": OCTOPUS_THREE_HEARTS, "purpose": "fact",
         "claim_refs": (result.approved_claims[0],)},
        {"text": OCTOPUS_THREE_HEARTS, "purpose": "explanation",
         "claim_refs": (result.approved_claims[0],)},
        {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
    ])
    issues = _issues_by_category(run_checks_only(script, result, package))
    assert any(
        i.severity == "WARNING" and "identical factual statement" in i.message
        for i in issues.get("repetition", [])
    )


def test_repetition_near_duplicate_factual_lines_warn():
    result, package = make_octopus_pair()
    # Same token set, reordered so the normalized strings differ (not exact dup).
    reordered = ("Octopuses have three hearts that push blood through the gills "
                 "and a systemic heart that powers the body - two branchial hearts.")
    script = make_script(result, lines=[
        {"text": OCTOPUS_THREE_HEARTS, "purpose": "fact",
         "claim_refs": (result.approved_claims[0],)},
        {"text": "Let's take a closer look.", "purpose": "context", "claim_refs": ()},
        {"text": reordered, "purpose": "explanation",
         "claim_refs": (result.approved_claims[0],)},
        {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
    ])
    issues = _issues_by_category(run_checks_only(script, result, package))
    assert any(
        i.severity == "WARNING" and "near-duplicate factual statement" in i.message
        for i in issues.get("repetition", [])
    )


def test_repetition_hook_conclusion_mirror_is_info_not_failure():
    result, package = make_octopus_pair()
    rebuttal = result.approved_claims[0]
    script = make_script(result, lines=[
        {"text": "Octopuses have three hearts.", "purpose": "hook", "claim_refs": (rebuttal,)},
        {"text": "Remember: octopuses have three hearts.", "purpose": "conclusion",
         "claim_refs": (rebuttal,)},
    ])
    issues = _issues_by_category(run_checks_only(script, result, package))
    repetition = issues.get("repetition", [])
    assert repetition and repetition[0].severity == "INFO"
    assert all(i.severity != "ERROR" for i in repetition)
    assert not [i for i in repetition if i.severity == "WARNING"]


# --------------------------------------------------------------------------- #
# 8. Structure
# --------------------------------------------------------------------------- #


def test_structure_missing_ending_errors():
    result, package = make_octopus_pair()
    script = make_script(result, lines=[
        {"text": OCTOPUS_THREE_HEARTS, "purpose": "fact",
         "claim_refs": (result.approved_claims[0],)},
        {"text": OCTOPUS_BLUE, "purpose": "explanation",
         "claim_refs": (result.approved_claims[2],)},
    ])
    issues = _issues_by_category(run_checks_only(script, result, package))
    assert any(
        i.severity == "ERROR" and "no ending" in i.message
        for i in issues.get("structure", [])
    )


def test_structure_empty_fails_hard():
    result, package = make_octopus_pair()
    script = make_script(result, lines=[], used=(), unused=())
    issues = _issues_by_category(run_checks_only(script, result, package))
    assert any(
        i.severity == "ERROR" and i.gate == "hard" and "empty" in i.message
        for i in issues.get("structure", [])
    )


# --------------------------------------------------------------------------- #
# 9. Hook quality
# --------------------------------------------------------------------------- #


def test_hook_missing_is_error_when_required():
    result, package = make_octopus_pair()
    script = make_script(result, lines=[
        {"text": OCTOPUS_THREE_HEARTS, "purpose": "fact",
         "claim_refs": (result.approved_claims[0],)},
        {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
    ])
    issues = _issues_by_category(run_checks_only(script, result, package))
    assert any(
        i.severity == "ERROR" and "no hook line" in i.message
        for i in issues.get("hook_quality", [])
    )


def test_hook_identical_to_first_fact_warns():
    result, package = make_octopus_pair()
    script = make_script(result, lines=[
        {"text": OCTOPUS_THREE_HEARTS, "purpose": "hook",
         "claim_refs": (result.approved_claims[0],)},
        {"text": OCTOPUS_THREE_HEARTS, "purpose": "fact",
         "claim_refs": (result.approved_claims[0],)},
        {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
    ])
    issues = _issues_by_category(run_checks_only(script, result, package))
    assert any(
        "identical to the first fact line" in i.message
        for i in issues.get("hook_quality", [])
    )


# --------------------------------------------------------------------------- #
# 10. Uncertainty preservation
# --------------------------------------------------------------------------- #


def test_hedge_drop_is_hard_error():
    hedged = "Scientists suggest that octopuses may use tools to open jars."
    result, package = make_simple_pair(
        claims=[(hedged, "inference")],
        verdicts={hedged: "PASS"},
    )
    ref = claim_id_for(hedged)
    script = make_script(result, lines=[
        {"text": "Scientists suggest that octopuses use tools to open jars.",
         "purpose": "fact", "claim_refs": (ref,)},
        {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
    ])
    issues = _issues_by_category(run_checks_only(script, result, package))
    assert any(
        i.severity == "ERROR" and i.gate == "hard" and "removes the hedge" in i.message
        for i in issues.get("uncertainty_preservation", [])
    )


def test_certainty_strengthening_is_hard_error():
    claim = "Octopuses use tools to open jars."
    result, package = make_simple_pair(claims=[(claim, "fact")], verdicts={claim: "PASS"})
    ref = claim_id_for(claim)
    script = make_script(result, lines=[
        {"text": "Scientists proved octopuses use tools to open jars.",
         "purpose": "fact", "claim_refs": (ref,)},
        {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
    ])
    issues = _issues_by_category(run_checks_only(script, result, package))
    assert any(
        i.severity == "ERROR" and i.gate == "hard" and "strengthens certainty" in i.message
        for i in issues.get("uncertainty_preservation", [])
    )


def test_hedge_preserved_is_not_an_error():
    hedged = "Scientists suggest that octopuses may use tools."
    result, package = make_simple_pair(
        claims=[(hedged, "inference")],
        verdicts={hedged: "PASS"},
    )
    ref = claim_id_for(hedged)
    script = make_script(result, lines=[
        {"text": hedged, "purpose": "fact", "claim_refs": (ref,)},
        {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
    ])
    issues = _issues_by_category(run_checks_only(script, result, package))
    uncertainty = issues.get("uncertainty_preservation", [])
    assert not [i for i in uncertainty if i.severity == "ERROR"]
    assert any(i.severity == "INFO" and "preserves the hedge" in i.message for i in uncertainty)


# --------------------------------------------------------------------------- #
# 11. Misleading language
# --------------------------------------------------------------------------- #


def test_misleading_phrase_warns():
    result, package = make_octopus_pair()
    script = make_script(result, lines=[
        {"text": "The truth is octopuses have three hearts.",
         "purpose": "fact", "claim_refs": (result.approved_claims[0],)},
        {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
    ])
    issues = _issues_by_category(run_checks_only(script, result, package))
    assert any(
        i.severity == "WARNING" and "misleading phrasing" in i.message
        for i in issues.get("misleading_language", [])
    )


def test_supported_phrase_is_not_flagged():
    result, package = make_octopus_pair()
    script = _octopus_script(result, package)  # clean script, no misleading phrase
    issues = run_checks_only(script, result, package)
    assert not _issues_by_category(issues).get("misleading_language")


# --------------------------------------------------------------------------- #
# 12. CTA quality
# --------------------------------------------------------------------------- #


def test_missing_cta_is_info_never_failure():
    result, package = make_octopus_pair()
    script = make_script(result, lines=[
        {"text": OCTOPUS_THREE_HEARTS, "purpose": "fact",
         "claim_refs": (result.approved_claims[0],)},
        {"text": "So that's the answer.", "purpose": "conclusion", "claim_refs": ()},
    ])
    issues_by_category = _issues_by_category(run_checks_only(script, result, package))
    infos = [i for i in issues_by_category.get("cta_quality", []) if i.severity == "INFO"]
    assert infos and "no call-to-action" in infos[0].message
    all_issues = [issue for group in issues_by_category.values() for issue in group]
    assert "cta_quality" not in {issue.category for issue in all_issues
                                 if issue.severity in {"ERROR", "WARNING"}}


def test_long_cta_warns():
    result, package = make_octopus_pair()
    cta = ("Please consider subscribing to our channel today and following "
           "along for more science content like this one.")
    assert len(cta.split()) > 14
    script = make_script(result, lines=[
        {"text": OCTOPUS_THREE_HEARTS, "purpose": "fact",
         "claim_refs": (result.approved_claims[0],)},
        {"text": cta, "purpose": "call_to_action", "claim_refs": ()},
    ])
    issues = _issues_by_category(run_checks_only(script, result, package))
    assert any(
        i.severity == "WARNING" and "CTA" in i.message and "too long" in i.message
        for i in issues.get("cta_quality", [])
    )


# --------------------------------------------------------------------------- #
# 13. Format / Shorts suitability
# --------------------------------------------------------------------------- #


def test_few_lines_warns():
    result, package = make_octopus_pair()
    script = make_script(result, lines=[
        {"text": OCTOPUS_THREE_HEARTS, "purpose": "fact",
         "claim_refs": (result.approved_claims[0],)},
        {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
    ])
    issues = _issues_by_category(run_checks_only(script, result, package))
    assert any(
        i.severity == "WARNING" and "only 2 line" in i.message
        for i in issues.get("format_shorts", [])
    )


def test_giant_block_warns():
    result, package = make_octopus_pair()
    block = ("Wow, that is remarkable. " * 14) + "Thanks for watching!"
    assert len(block) > 160
    script = make_script(result, lines=[
        {"text": OCTOPUS_THREE_HEARTS, "purpose": "fact",
         "claim_refs": (result.approved_claims[0],)},
        {"text": block, "purpose": "call_to_action", "claim_refs": ()},
    ])
    issues = _issues_by_category(run_checks_only(script, result, package))
    assert any(
        i.severity == "WARNING" and "giant block" in i.message
        for i in issues.get("format_shorts", [])
    )