"""Stage-5 QA models1 unit tests."""

from __future__ import annotations

import pytest

from agents.qa.script.models import (
    CATEGORY_DIMENSION,
    QA_CATEGORIES,
    QA_DIMENSIONS,
    QA_SEVERITIES,
    QA_VERDICTS,
    SCHEMA,
    SCHEMA_VERSION,
    STAGE,
    ClaimCoverage,
    DEFAULT_SCORE_WEIGHTS,
    QACheckRecord,
    QAIssue,
    ScriptQAConfig,
    ScriptQAResult,
    ScriptQAError,
    qa_result_id_for,
)
from script_qa_test_utils import (
    load_mock_package,
    load_mock_script_and_result,
    run_qa_eval,
)


def test_result_id_format_and_determinism():
    first = qa_result_id_for("run-1", "sc_abc123")
    second = qa_result_id_for("run-1", "sc_abc123")
    other = qa_result_id_for("run-1", "sc_xyz999")
    assert first == second
    assert first != other
    assert first.startswith("sqa_")
    assert len(first[len("sqa_"):]) == 12


def test_stage_constants():
    assert SCHEMA == "ScriptQAResult"
    assert SCHEMA_VERSION == 1
    assert STAGE == "script_qa"
    assert len(QA_CATEGORIES) == 13
    assert len(QA_DIMENSIONS) == 10
    assert set(QA_VERDICTS) == {"PASS", "REVISE", "REJECT"}
    assert set(QA_SEVERITIES) == {"INFO", "WARNING", "ERROR"}
    assert set(CATEGORY_DIMENSION.values()) <= set(QA_DIMENSIONS)


def test_default_weights_cover_dimensions_and_sum_to_one():
    assert set(DEFAULT_SCORE_WEIGHTS) == set(QA_DIMENSIONS)
    assert round(sum(DEFAULT_SCORE_WEIGHTS.values()), 9) == 1.0


def test_config_accepts_defaults():
    config = ScriptQAConfig()
    assert config.schema_version == "script_qa.v1"
    assert config.min_duration_seconds == 30
    assert config.max_duration_seconds == 60
    assert config.hard_min_duration_seconds == 15
    assert config.hard_max_duration_seconds == 90
    assert config.require_hook is True
    assert config.critical_unused_is_error is True


def test_config_rejects_bad_weights():
    config = ScriptQAConfig()
    weights = dict(DEFAULT_SCORE_WEIGHTS)
    weights["clarity"] = weights["clarity"] + 0.1
    with pytest.raises(ValueError):
        ScriptQAConfig(weights=tuple(weights.items())).validate()


def test_config_rejects_unknown_dimension_weight():
    weights = list(DEFAULT_SCORE_WEIGHTS.items()) + [("nowhere", 0.0)]
    with pytest.raises(ValueError):
        ScriptQAConfig(weights=weights).validate()


def test_qa_issue_round_trip_and_defaults():
    issue = QAIssue(
        category="structure",
        severity="ERROR",
        gate="hard",
        message="bad structure",
        affected_line_ids=("ln_00",),
        affected_claim_ids=("cl_x",),
    ).validate()
    assert issue.issue_id == ""
    assert issue.dimension == ""  # derived only by the check registry
    assert issue.is_error is True
    repaired = QAIssue.from_dict(issue.to_dict())
    assert repaired == issue
    assert repaired.category == "structure"
    assert repaired.severity == "ERROR"
    assert repaired.gate == "hard"


def test_qa_issue_validate_rejects_unknown_category():
    with pytest.raises(ScriptQAError):
        QAIssue(category="nonsense", severity="ERROR", gate="hard", message="x").validate()


def test_claim_coverage_compute():
    coverage = ClaimCoverage.compute(
        approved_claim_ids=("a", "b", "c"),
        used_claim_ids=("a",),
        critical_ids=("b",),
    )
    assert coverage.coverage_ratio == pytest.approx(0.333, abs=0.001)
    assert coverage.used_claim_ids == ("a",)
    assert coverage.approved_claim_ids == ("a", "b", "c")
    assert coverage.unused_approved_claim_ids == ("b", "c")
    assert coverage.critical_unused_claim_ids == ("b",)
    assert coverage.to_dict()["coverage_ratio"] == pytest.approx(0.333, abs=0.001)


def test_check_record_round_trip():
    record = QACheckRecord(
        check="clarity",
        category="clarity",
        dimension="clarity",
        hard_gate=False,
        warnings=1,
    )
    restored = QACheckRecord.from_dict(record.to_dict())
    assert restored == record
    assert restored.passed is True


def test_result_json_round_trip():
    script, fc = load_mock_script_and_result()
    package = load_mock_package()
    result = run_qa_eval(script, fc, package, run_id="round-trip")
    restored = ScriptQAResult.from_json(result.to_json())
    assert restored.to_json() == result.to_json()
    assert restored.overall_verdict == result.overall_verdict


def test_result_envelope():
    script, fc = load_mock_script_and_result()
    package = load_mock_package()
    result = run_qa_eval(script, fc, package, run_id="envelope")
    assert result.schema == SCHEMA
    assert result.schema_version == SCHEMA_VERSION
    assert result.stage == STAGE
    assert result.script_id == script.script_id
    assert result.result_id == fc.result_id
    assert result.produced_by["model"] == "ScriptQAValidator"