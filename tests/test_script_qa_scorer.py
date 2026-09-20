"""Stage-5 QA scorer (deterministic weighted scoring) unit tests."""

from __future__ import annotations

import pytest

from agents.qa.script.models import (
    DEFAULT_SCORE_WEIGHTS,
    QA_CATEGORIES,
    QA_DIMENSIONS,
    QAIssue,
    ScriptQAConfig,
)
from agents.qa.script.scorer import (
    compute_score,
    validate_weights,
)
from agents.qa.script.validator import evaluate_script
from script_qa_test_utils import (
    FIXED_AT,
    octopus_ok_lines,
    make_octopus_pair,
    make_script,
    run_qa_eval,
)


def _issue(severity, gate="soft"):
    return QAIssue(category="structure", severity=severity, gate=gate, message="x")


def test_clean_script_scores_full_marks():
    result, package = make_octopus_pair()
    qa = run_qa_eval(make_script(result, lines=octopus_ok_lines(result)), result, package)
    assert qa.score == 100.0
    assert set(qa.score_detail["dimensions"]) == set(QA_DIMENSIONS)
    assert qa.score_detail["weights"] == DEFAULT_SCORE_WEIGHTS


def test_score_is_deterministic():
    result, package = make_octopus_pair()
    script = make_script(result, lines=[
        {"text": octopus_ok_lines(result)[2]["text"], "purpose": "fact",
         "claim_refs": (result.approved_claims[0],)},
        {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
    ])
    first = evaluate_script(script, result, {c.claim_id: c for c in package.claims},
                            run_id="x", created_at=FIXED_AT)
    second = evaluate_script(script, result, {c.claim_id: c for c in package.claims},
                             run_id="x", created_at=FIXED_AT)
    assert first.score == second.score
    assert first.score_detail == second.score_detail


def test_penalty_ordering_hard_soft_warning_info():
    base = compute_score([], weights=DEFAULT_SCORE_WEIGHTS).score
    assert base == 100.0
    hard = compute_score([_issue("ERROR", "hard")], weights=DEFAULT_SCORE_WEIGHTS).score
    soft = compute_score([_issue("ERROR", "soft")], weights=DEFAULT_SCORE_WEIGHTS).score
    warn = compute_score([_issue("WARNING")], weights=DEFAULT_SCORE_WEIGHTS).score
    info = compute_score([_issue("INFO")], weights=DEFAULT_SCORE_WEIGHTS).score
    assert hard < soft < warn < info <= base


def test_penalty_formula():
    # One hard error on a clarity issue hits only the clarity dimension (-45).
    score = compute_score(
        [QAIssue(category="clarity", severity="ERROR", gate="hard", message="x")],
        weights=DEFAULT_SCORE_WEIGHTS,
    )
    assert score.dimensions["clarity"] == pytest.approx(100 - 45)
    overall = round(100.0 - 0.10 * 45, 1)
    assert score.score == pytest.approx(overall)


def test_score_floor_at_zero():
    # Enough hard errors in EVERY category clamps every dimension (and the
    # overall score) to zero - demonstrating the clamp, not real-world scoring.
    issues = [
        QAIssue(category=category, severity="ERROR", gate="hard", message="x").validate()
        for category in QA_CATEGORIES
        for _ in range(1000)
    ]
    score = compute_score(issues, weights=DEFAULT_SCORE_WEIGHTS)
    assert score.score == 0.0
    assert all(dimension == 0.0 for dimension in score.dimensions.values())


def test_score_detail_records_dimension_values():
    result, package = make_octopus_pair()
    long_sentence = (
        "Octopuses are smart animals that can solve complex puzzles, navigate "
        "mazes, remember individual people they meet, and learn entirely new "
        "behaviors simply by watching other octopuses."
    )
    assert len(long_sentence.split()) > 25
    assert " " + "." + " " not in long_sentence  # single sentence for the check
    script = make_script(result, lines=[
        {"text": octopus_ok_lines(result)[2]["text"], "purpose": "fact",
         "claim_refs": (result.approved_claims[0],)},
        {"text": long_sentence, "purpose": "call_to_action", "claim_refs": ()},
    ])
    qa = run_qa_eval(script, result, package)
    # clarity should drop below 100 from the long-sentence warning
    assert qa.score_detail["dimensions"]["clarity"] < 100.0
    assert qa.score < 100.0


def test_custom_weights_change_score():
    issues = [_issue("ERROR", "hard")]
    default = compute_score(issues, weights=DEFAULT_SCORE_WEIGHTS).score
    custom = {dim: 0.0 for dim in QA_DIMENSIONS}
    custom["structure"] = 1.0
    other = compute_score(issues, weights=custom).score
    assert other != default
    assert other == pytest.approx(100.0 - 45)


def test_validate_weights_accepts_default():
    assert validate_weights(DEFAULT_SCORE_WEIGHTS) is None


def test_validate_weights_rejects_bad_sum():
    weights = dict(DEFAULT_SCORE_WEIGHTS)
    weights["clarity"] += 0.5
    with pytest.raises(ValueError):
        compute_score([], weights=weights)
    with pytest.raises(ValueError):
        ScriptQAConfig(
            weights=tuple((k, v) for k, v in weights.items())
        ).validate()


def test_validate_weights_rejects_missing_dimension():
    weights = dict(DEFAULT_SCORE_WEIGHTS)
    weights.pop("pacing")
    weights["clarity"] += 0.10
    with pytest.raises(ValueError):
        compute_score([], weights=weights)


def test_weights_float_rounding_stable():
    result, package = make_octopus_pair()
    script = make_script(result, lines=[
        {"text": octopus_ok_lines(result)[2]["text"], "purpose": "fact",
         "claim_refs": (result.approved_claims[0],)},
        {"text": "Thanks for watching.", "purpose": "call_to_action", "claim_refs": ()},
    ])
    runs = {run_qa_eval(script, result, package).score for _ in range(3)}
    assert len(runs) == 1
    assert isinstance(runs.pop(), float)