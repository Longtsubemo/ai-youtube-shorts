"""Overall-verdict + artifact-assembly tests for the Stage-8 video-QA gate."""

from __future__ import annotations

import dataclasses

from agents.qa.video.models import VideoQAIssue, qa_result_id_for
from agents.qa.video.validator import (
    VERDICT_PASS,
    VERDICT_REJECT,
    VERDICT_REVIEW,
    aggregate_verdict,
    evaluate_video,
)
from video_qa_test_utils import (
    healthy_pair,
    result_failed_scene,
    result_mock_with_uri,
    result_non_mock_artifact,
    result_provider_marker,
)


def _issue(severity, gate):
    return VideoQAIssue(
        issue_id="vqa_00",
        category="duration_window",
        severity=severity,
        message="x",
        gate=gate,
        check="duration_window",
    )


def test_aggregate_verdict_rule():
    assert aggregate_verdict([]) == VERDICT_PASS
    assert aggregate_verdict([_issue("ERROR", "hard")]) == VERDICT_REJECT
    assert aggregate_verdict([_issue("ERROR", "soft")]) == VERDICT_REVIEW
    assert aggregate_verdict([_issue("ERROR", "hard"), _issue("ERROR", "soft")]) == VERDICT_REJECT


def test_healthy_grades_pass_and_is_deterministic():
    plan, result = healthy_pair()
    first = evaluate_video(result, plan, run_id="video-qa-test", created_at="2026-09-20T08:00:00+00:00")
    second = evaluate_video(result, plan, run_id="video-qa-test", created_at="2026-09-20T08:00:00+00:00")
    assert first.overall_verdict == VERDICT_PASS
    assert first.passed
    assert len(first.checks) == 15
    assert first.issues == first.warnings == first.infos == ()
    assert first.to_dict() == second.to_dict()
    assert first.qa_result_id == qa_result_id_for("video-qa-test", plan.production_plan_id)
    assert first.provenance.generated_by == "video_qa_agent_v1"
    assert first.provenance.source == "DeterministicVideoQA"
    assert first.run_id == "video-qa-test"


def test_run_id_fallback_uses_result_run():
    plan, result = healthy_pair(run_id="carrier-run")
    qa = evaluate_video(result, plan)
    assert qa.run_id == "carrier-run"
    assert qa.created_at == result.created_at


def test_soft_error_yields_review():
    plan, result = healthy_pair()
    qa = evaluate_video(result_mock_with_uri(plan, result), plan, run_id="video-qa-test")
    assert qa.overall_verdict == VERDICT_REVIEW
    assert len(qa.issues) >= 1
    assert all(issue.severity == "ERROR" for issue in qa.issues)


def test_hard_error_yields_reject():
    plan, result = healthy_pair()
    for scenario in (
        result_failed_scene(plan, result),
        result_provider_marker(plan, result),
        result_non_mock_artifact(plan, result),
    ):
        qa = evaluate_video(scenario, plan, run_id="video-qa-test")
        assert qa.overall_verdict == VERDICT_REJECT
        assert any(issue.gate == "hard" for issue in qa.issues)


def test_issue_ids_are_stable_and_sequential():
    plan, result = healthy_pair()
    qa = evaluate_video(result_failed_scene(plan, result), plan, run_id="video-qa-test")
    ids = [issue.issue_id for issue in qa.issues]
    assert ids == sorted(ids)
    assert all(issue_id.startswith("vqa_") for issue_id in ids)


def test_structurally_broken_result_is_graded_not_raised():
    plan, result = healthy_pair()
    rows = list(result.scenes)
    rows[0], rows[1] = rows[1], rows[0]  # non-contiguous orders, no renumber
    broken = dataclasses.replace(result, scenes=tuple(rows))
    qa = evaluate_video(broken, plan, run_id="video-qa-test")
    assert qa.overall_verdict == VERDICT_REJECT
    assert any(issue.category == "structure_integrity" for issue in qa.issues)


def test_scene_list_echoes_contract_ids():
    plan, result = healthy_pair()
    qa = evaluate_video(result, plan, run_id="video-qa-test")
    assert qa.target_ref == plan.production_plan_id
    assert qa.video_generation_result_id == result.video_generation_result_id
    assert qa.topic_id == plan.topic_id
    assert qa.script_id == plan.script_id