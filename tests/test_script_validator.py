"""Validator tests: scripts may not invent, strengthen, or drop hedges.

The "no hallucination" guarantees (docs/DATA_CONTRACTS.md §8): refs must be
approved claims, factual lines must carry refs, used/unused tracking must be
exact, and factual wording must stay inside the approved claims' token
coverage while preserving their certainty level.
"""

from __future__ import annotations

import pytest

from agents.script.script_generator import ScriptConfig
from agents.script.validator import validate_script

from script_test_utils import (
    FIXED_AT,
    OCTOPUS_BLUE,
    OCTOPUS_SMARTEST,
    OCTOPUS_SWIM,
    OCTOPUS_TANK,
    OCTOPUS_THREE_HEARTS,
    build_octopus_pair,
    build_pair,
    make_script,
)

DEFAULT_CFG = ScriptConfig()


def _hook(primary_id):
    return {
        "text": "You probably didn't know this, but here is why it matters.",
        "purpose": "hook",
        "claim_refs": (primary_id,),
    }


def _pkg(result):
    _, package = build_octopus_pair()
    return package


def _validate(script, result, claims=None):
    from agents.script.stage import approved_claims

    if claims is None:
        claims = approved_claims(result, _pkg(result))
    return validate_script(script, result, claims, DEFAULT_CFG)


def _approved_lines(result, texts=None):
    lines = []
    primary = result.approved_claims[0]
    lines.append(
        {"text": "You probably didn't know this, but the answer explains it.", "purpose": "hook", "claim_refs": (primary,)}
    )
    for no, claim_id in enumerate(result.approved_claims):
        if texts is not None and no >= len(texts):
            break
        text = texts[no] if texts is not None else (OCTOPUS_THREE_HEARTS, OCTOPUS_SWIM, OCTOPUS_BLUE)[no]
        lines.append(
            {"text": text, "purpose": ("fact", "explanation", "explanation")[no], "claim_refs": (claim_id,)}
        )
    return lines


def test_valid_script_passes():
    result, _ = build_octopus_pair(approved=("fact", "explanation", "explanation"))
    script = make_script(result, lines=_approved_lines(result))
    validation = _validate(script, result)
    assert validation.valid, validation.errors
    assert not validation.errors


def test_factual_line_requires_claim_refs():
    result, _ = build_octopus_pair(approved=("fact",))
    script = make_script(
        result,
        lines=[
            _hook(result.approved_claims[0]),
            {"text": OCTOPUS_THREE_HEARTS, "purpose": "fact", "claim_refs": ()},
        ],
    )
    validation = _validate(script, result)
    assert not validation.valid
    assert any("no claim_refs" in error for error in validation.errors)


def test_unknown_claim_ref_rejected():
    result, _ = build_octopus_pair(approved=("fact",))
    script = make_script(
        result,
        lines=[
            _hook(result.approved_claims[0]),
            {"text": OCTOPUS_THREE_HEARTS, "purpose": "fact", "claim_refs": ("cl_does_not_exist",)},
        ],
    )
    assert not _validate(script, result).valid


def test_ref_to_rejected_claim_rejected():
    result, _ = build_octopus_pair(approved=("fact",))
    rejected_id = result.rejected_claims[0]
    script = make_script(
        result,
        lines=[
            _hook(result.approved_claims[0]),
            {"text": OCTOPUS_TANK, "purpose": "fact", "claim_refs": (rejected_id,)},
        ],
    )
    validation = _validate(script, result)
    assert not validation.valid
    assert any("REJECTED" in error for error in validation.errors)


def test_ref_to_review_claim_rejected():
    result, _ = build_octopus_pair(approved=("fact", "explanation", "explanation"))
    review_id = result.review_items[0]
    script = make_script(
        result,
        lines=[
            _hook(result.approved_claims[0]),
            {"text": OCTOPUS_SMARTEST, "purpose": "fact", "claim_refs": (review_id,)},
        ],
    )
    validation = _validate(script, result)
    assert not validation.valid
    assert any("REVIEW" in error for error in validation.errors)


def test_claim_passed_but_not_approved_is_unapproved():
    # Both claims verify PASS, but the approved_claims list admits only the
    # first; citing the second is an unapproved-claim violation.
    result, _ = build_pair(
        claims=[(OCTOPUS_THREE_HEARTS, "fact"), (OCTOPUS_BLUE, "explanation")],
        verdicts={OCTOPUS_THREE_HEARTS: "PASS", OCTOPUS_BLUE: "PASS"},
    )
    second_id = result.approved_claims[1]
    result.approved_claims = (result.approved_claims[0],)
    script = make_script(
        result,
        lines=[
            _hook(result.approved_claims[0]),
            {"text": OCTOPUS_BLUE, "purpose": "explanation", "claim_refs": (second_id,)},
        ],
    )
    validation = _validate(script, result)
    assert not validation.valid
    assert any("unapproved" in error for error in validation.errors)


def test_used_ids_must_match_refs():
    result, _ = build_octopus_pair(approved=("fact",))
    used = (result.approved_claims[0],)
    script = make_script(
        result,
        lines=[{"text": OCTOPUS_THREE_HEARTS, "purpose": "fact", "claim_refs": used}],
        used=("cl_wrong",),
    )
    validation = _validate(script, result)
    assert not validation.valid
    assert any("used_claim_ids" in error for error in validation.errors)


def test_used_unused_must_cover_approved():
    result, _ = build_octopus_pair(approved=("fact", "explanation", "explanation"))
    used = tuple(result.approved_claims[:2])
    script = make_script(
        result,
        lines=_approved_lines(result),
        used=used,
        unused=(),
    )
    validation = _validate(script, result)
    assert not validation.valid
    assert any("cover the approved_claims" in error for error in validation.errors)


def test_used_unused_must_be_disjoint():
    result, _ = build_octopus_pair(approved=("fact", "explanation", "explanation"))
    script = make_script(
        result,
        lines=_approved_lines(result),
        used=tuple(result.approved_claims[:1]),
        unused=tuple(result.approved_claims[:1]),
    )
    validation = _validate(script, result)
    assert not validation.valid
    assert any("overlaps" in error for error in validation.errors)


def test_missing_fact_or_explanation_line():
    result, _ = build_octopus_pair(approved=("fact",))
    claim_id = result.approved_claims[0]
    script = make_script(
        result,
        lines=[
            {"text": "You probably didn't know this.", "purpose": "hook", "claim_refs": ()},
            {"text": "Follow me for more.", "purpose": "call_to_action", "claim_refs": ()},
        ],
    )
    validation = _validate(script, result)
    assert not validation.valid
    assert any("no fact/explanation line" in error for error in validation.errors)


def test_empty_title_hook_lines_fail():
    result, _ = build_octopus_pair(approved=("fact",))
    claim_id = result.approved_claims[0]
    script = make_script(
        result,
        title="",
        hook="",
        lines=[{"text": OCTOPUS_THREE_HEARTS, "purpose": "fact", "claim_refs": (claim_id,)}],
    )
    validation = _validate(script, result)
    assert not validation.valid


def test_missing_provenance_fails():
    result, _ = build_octopus_pair(approved=("fact",))
    script = make_script(result, lines=_approved_lines(result), provenance=None)
    validation = _validate(script, result)
    assert not validation.valid
    assert any("provenance" in error for error in validation.errors)


def test_result_id_mismatch_fails():
    result, _ = build_octopus_pair(approved=("fact",))
    script = make_script(result, lines=_approved_lines(result), result_id="fc_wrong")
    validation = _validate(script, result)
    assert not validation.valid
    assert any("result_id" in error for error in validation.errors)


def test_duration_below_min_is_warning_not_error():
    result, _ = build_octopus_pair(approved=("fact",))
    script = make_script(result, lines=_approved_lines(result))
    cfg = ScriptConfig(min_duration_seconds=999.0)
    from agents.script.stage import approved_claims

    claims = approved_claims(result, _pkg(result))
    validation = validate_script(script, result, claims, cfg)
    assert validation.valid
    assert any("below the" in warning for warning in validation.warnings)


def test_duration_above_max_is_warning_not_error():
    result, _ = build_octopus_pair(approved=("fact", "explanation", "explanation"))
    script = make_script(result, lines=_approved_lines(result))
    cfg = ScriptConfig(max_duration_seconds=1.0)
    from agents.script.stage import approved_claims

    claims = approved_claims(result, _pkg(result))
    validation = validate_script(script, result, claims, cfg)
    assert validation.valid
    assert any("exceeds the" in warning for warning in validation.warnings)


def test_declared_duration_must_match_derived():
    result, _ = build_octopus_pair(approved=("fact",))
    script = make_script(
        result,
        lines=_approved_lines(result),
        estimated_duration=9999.0,
    )
    validation = _validate(script, result)
    assert not validation.valid
    assert any("does not match" in error for error in validation.errors)


def test_out_of_coverage_wording_blocked():
    result, _ = build_octopus_pair(approved=("fact",))
    claim_id = result.approved_claims[0]
    script = make_script(
        result,
        lines=[
            _hook(result.approved_claims[0]),
            {"text": "Researchers discovered that octopuses evolved hearts millions of years ago.", "purpose": "fact", "claim_refs": (claim_id,)}
        ],
    )
    validation = _validate(script, result)
    assert not validation.valid
    assert any("not in its approved claims" in error for error in validation.errors)


def test_strengthened_certainty_blocked():
    result, _ = build_octopus_pair(approved=("fact",))
    claim_id = result.approved_claims[0]
    script = make_script(
        result,
        lines=[
            _hook(claim_id),
            {"text": "Researchers proved octopuses evolved three hearts.", "purpose": "fact", "claim_refs": (claim_id,)}
        ],
    )
    validation = _validate(script, result)
    assert not validation.valid
    assert any("unsupported" in error or "strengthens" in error for error in validation.errors)


def test_dropped_hedge_blocked():
    result, _ = build_pair(
        claims=[(OCTOPUS_SMARTEST, "inference")],
        verdicts={OCTOPUS_SMARTEST: "PASS"},
    )
    claim_id = result.approved_claims[0]
    script = make_script(
        result,
        lines=[
            _hook(claim_id),
            {"text": "Octopuses are the smartest invertebrates in the sea.", "purpose": "fact", "claim_refs": (claim_id,)}
        ],
    )
    validation = _validate(script, result)
    assert not validation.valid
    assert any("hedge" in error for error in validation.errors)


def test_hedge_kept_passes():
    result, _ = build_pair(
        claims=[(OCTOPUS_SMARTEST, "inference")],
        verdicts={OCTOPUS_SMARTEST: "PASS"},
    )
    claim_id = result.approved_claims[0]
    script = make_script(
        result,
        lines=[
            _hook(claim_id),
            {"text": OCTOPUS_SMARTEST, "purpose": "fact", "claim_refs": (claim_id,)}
        ],
    )
    assert _validate(script, result).valid


def test_framing_lines_exempt_from_token_coverage():
    result, _ = build_octopus_pair(approved=("fact",))
    claim_id = result.approved_claims[0]
    script = make_script(
        result,
        lines=[
            {"text": "This is entirely fresh framing wording.", "purpose": "hook", "claim_refs": ()},
            {"text": OCTOPUS_THREE_HEARTS, "purpose": "fact", "claim_refs": (claim_id,)},
        ],
    )
    assert _validate(script, result).valid


def test_validation_has_to_dict_report():
    result, _ = build_octopus_pair(approved=("fact",))
    script = make_script(result, lines=_approved_lines(result))
    validation = _validate(script, result)
    report = validation.to_dict()
    assert report["valid"] is True
    assert isinstance(report["errors"], list)
    assert isinstance(report["warnings"], list)