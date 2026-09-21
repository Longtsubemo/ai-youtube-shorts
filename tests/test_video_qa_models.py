"""Model + artifact-envelope tests for the Stage-8 video-QA package.

Covers the versioned header/schema constants, the issue/check/config model
behaviour, and the VideoQAResult JSON round-trip (the envelope that lands in
data/<run_id>/production/qa_result.json).
"""

from __future__ import annotations

import json

import pytest

from agents.qa.video.models import (
    QA_GATES,
    QA_SEVERITIES,
    QA_VERDICTS,
    SCHEMA,
    SCHEMA_VERSION,
    STAGE,
    STATE_STAGE,
    VideoQAConfig,
    VideoQAError,
    VideoQACheckRecord,
    VideoQAIssue,
    VideoQAResult,
    qa_result_id_for,
)
from agents.qa.video.validator import evaluate_video
from pipeline.state import Stage
from video_qa_test_utils import healthy_pair


def test_schema_constants_are_stable():
    assert SCHEMA == "VideoQAResult"
    assert SCHEMA_VERSION == 1
    assert STAGE == "video"
    assert STATE_STAGE == Stage.QUALITY_CHECK
    assert QA_VERDICTS == ("PASS", "REVIEW", "REJECT")
    assert QA_SEVERITIES == ("ERROR", "WARNING", "INFO")
    assert QA_GATES == ("hard", "soft")


def test_qa_result_id_for_is_deterministic_and_scoped():
    assert qa_result_id_for("mock", "pp_e99181e73b5d").startswith("qa_")
    assert qa_result_id_for("mock", "pp_e99181e73b5d") == qa_result_id_for(
        "mock", "pp_e99181e73b5d"
    )
    assert qa_result_id_for("a", "x") != qa_result_id_for("b", "x")


def test_issue_validate_and_roundtrip():
    issue = VideoQAIssue(
        issue_id="vqa_00",
        category="duration_window",
        severity="ERROR",
        message="duration outside hard bounds",
        affected_scene_ids=("scn_01",),
        affected_artifact_ids=("mockva_0",),
        gate="hard",
        check="duration_window",
    ).validate()
    assert issue.is_error
    assert issue.refs == ("scn_01", "mockva_0")
    restored = VideoQAIssue.from_dict(issue.to_dict())
    assert restored == issue


def test_issue_rejects_unknown_vocabulary():
    with pytest.raises(VideoQAError):
        VideoQAIssue(category="nope", severity="ERROR", message="x").validate()
    with pytest.raises(VideoQAError):
        VideoQAIssue(category="timing_fidelity", severity="BANANA", message="x").validate()
    with pytest.raises(VideoQAError):
        VideoQAIssue(category="timing_fidelity", severity="ERROR", message="").validate()


def test_check_record_outcomes():
    assert VideoQACheckRecord(check="c", category="c", hard_gate=True).passed is True
    assert VideoQACheckRecord(check="c", category="c", hard_gate=True).outcome == "PASS"
    soft = VideoQACheckRecord(check="c", category="c", hard_gate=False, errors=2, warnings=1)
    assert soft.outcome == "REVIEW"
    hard = VideoQACheckRecord(check="c", category="c", hard_gate=True, errors=2, hard_errors=1)
    assert hard.outcome == "REJECT"
    assert hard.passed is False
    record = VideoQACheckRecord(check="c", category="c", hard_gate=True, errors=1)
    assert VideoQACheckRecord.from_dict(record.to_dict()) == record


def test_config_defaults_and_validate():
    config = VideoQAConfig()
    assert config.hard_min_duration_seconds == 15.0
    assert config.hard_max_duration_seconds == 90.0
    assert config.min_duration_seconds == 30.0
    assert config.max_duration_seconds == 60.0
    assert config.expected_aspect_ratio == "9:16"
    assert config.provider_isolation_denylist  # reuses the Stage-7 vocabulary
    config.validate()
    with pytest.raises(ValueError):
        VideoQAConfig(hard_min_duration_seconds=40.0).validate()
    with pytest.raises(ValueError):
        VideoQAConfig(max_duration_seconds=100.0).validate()


def test_result_roundtrip_driven_by_evaluate_video():
    plan, result = healthy_pair()
    qa = evaluate_video(result, plan, run_id="video-qa-test", created_at="2026-09-20T08:00:00+00:00")
    restored = VideoQAResult.from_json(qa.to_json())
    assert restored.validate() is restored
    assert restored == qa
    assert restored.target_ref == plan.production_plan_id
    assert restored.video_generation_result_id == result.video_generation_result_id
    assert restored.stage == "video"
    assert restored.schema == SCHEMA
    assert restored.overall_verdict in QA_VERDICTS
    assert any(check.check == "media_honesty" for check in restored.checks)
    assert all(isinstance(record, VideoQACheckRecord) for record in restored.checks)


def test_result_validate_rejects_bad_verdict_and_missing_id():
    plan, result = healthy_pair()
    qa = evaluate_video(result, plan, run_id="video-qa-test")
    with pytest.raises(VideoQAError):
        qa.__class__(**{**qa.__dict__, "overall_verdict": "BANANA"}).validate()
    with pytest.raises(VideoQAError):
        qa.__class__(**{**qa.__dict__, "qa_result_id": ""}).validate()
    with pytest.raises(VideoQAError):
        qa.__class__(**{**qa.__dict__, "provenance": None}).validate()


def test_result_from_dict_rejects_broken_envelope():
    plan, result = healthy_pair()
    qa = evaluate_video(result, plan, run_id="video-qa-test")
    assert VideoQAResult.from_json(qa.to_json()) == qa  # sane input round-trips
    broken = json.loads(qa.to_json())
    del broken["data"]["checked_at"]
    with pytest.raises(VideoQAError):
        VideoQAResult.from_json(json.dumps(broken))