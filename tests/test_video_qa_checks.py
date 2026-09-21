"""Per-rule tests for the 15 Stage-8 video-QA checks.

Each scenario from video_qa_test_utils triggers exactly the registered check
under test; the healthy pair triggers NONE (proving the controls are tight).
"""

from __future__ import annotations

import dataclasses

import pytest

from agents.qa.video.checks import CHECKS, run_checks
from agents.qa.video.models import VIDEO_QA_CATEGORIES, VideoQAConfig
from agents.video_generation.models import VideoGenerationResult
from video_qa_test_utils import (
    healthy_pair,
    result_bad_overall_status,
    result_failed_scene,
    result_media_claim,
    result_missing_scene,
    result_mock_with_uri,
    result_non_mock_artifact,
    result_pending_scene,
    result_phantom_scene,
    result_provider_marker,
    result_scaled_duration,
    result_swapped_ordering,
    result_timing_mismatch,
)


def _run(result, plan, config=None):
    return run_checks(result, plan, config or VideoQAConfig())


def _by_check(issues, check):
    return [issue for issue in issues if issue.check == check]


def test_registry_matches_categories():
    assert len(CHECKS) == 15
    assert [check.name for check in CHECKS] == list(VIDEO_QA_CATEGORIES)
    for check in CHECKS:
        assert check.category == check.name
        assert check.fn is not None
        assert check.note.strip()


def test_healthy_result_is_clean():
    plan, result = healthy_pair()
    issues, records = _run(result, plan)
    assert issues == []
    assert all(record.outcome == "PASS" for record in records)


def test_duration_window_soft_and_hard():
    plan, result = healthy_pair()
    soft, records = _run(result_scaled_duration(plan, result, target_total=70.0), plan)
    found = _by_check(soft, "duration_window")
    assert len(found) == 1
    assert found[0].gate == "soft" and found[0].severity == "ERROR"
    assert [r.outcome for r in records if r.check == "duration_window"] == ["REVIEW"]
    hard, _ = _run(result_scaled_duration(plan, result, target_total=10.0), plan)
    found = _by_check(hard, "duration_window")
    assert len(found) == 1
    assert found[0].gate == "hard"
    assert any(r.outcome == "REJECT" for r in _run(result_scaled_duration(plan, result, target_total=10.0), plan)[1])


def test_aspect_ratio_metadata():
    plan, result = healthy_pair()
    bad_style = dataclasses.replace(plan, global_style=dataclasses.replace(plan.global_style, aspect_ratio="4:5"))
    issues, _ = _run(result, bad_style)
    found = _by_check(issues, "aspect_ratio_metadata")
    assert len(found) == 1
    assert found[0].gate == "hard"
    soft_style = dataclasses.replace(plan, global_style=dataclasses.replace(plan.global_style, fps=24))
    issues, _ = _run(result, soft_style)
    found = _by_check(issues, "aspect_ratio_metadata")
    assert len(found) == 1 and found[0].gate == "soft"


def test_scene_coverage():
    plan, result = healthy_pair()
    issues, _ = _run(result_missing_scene(plan, result), plan)
    found = _by_check(issues, "scene_coverage")
    assert len(found) == 1
    assert found[0].gate == "hard"


def test_scene_pending():
    plan, result = healthy_pair()
    issues, _ = _run(result_pending_scene(plan, result), plan)
    found = _by_check(issues, "scene_pending")
    assert len(found) == 1 and found[0].gate == "hard"


def test_scene_failures():
    plan, result = healthy_pair()
    issues, _ = _run(result_failed_scene(plan, result), plan)
    found = _by_check(issues, "scene_failures")
    assert len(found) == 1 and found[0].gate == "hard"
    assert "scn_02" in found[0].message


def test_scene_ordering():
    plan, result = healthy_pair()
    issues, _ = _run(result_swapped_ordering(plan, result), plan)
    found = _by_check(issues, "scene_ordering")
    assert len(found) >= 2
    assert all(issue.gate == "hard" for issue in found)


def test_timing_fidelity():
    plan, result = healthy_pair()
    issues, _ = _run(result_timing_mismatch(plan, result), plan)
    found = _by_check(issues, "timing_fidelity")
    assert len(found) == 2  # request + artifact drift both flagged
    assert all(issue.gate == "hard" for issue in found)


def test_artifact_assignment():
    plan, result = healthy_pair()
    rows = list(result.scenes)
    rows[1] = dataclasses.replace(rows[1], artifact=None)
    broken = dataclasses.replace(result, scenes=tuple(rows))
    issues, _ = _run(broken, plan)
    found = _by_check(issues, "artifact_assignment")
    assert len(found) == 1 and found[0].gate == "hard"


def test_structure_integrity_empty_and_noncontiguous():
    plan, result = healthy_pair()
    empty = VideoGenerationResult(
        video_generation_result_id=result.video_generation_result_id,
        production_plan_id=result.production_plan_id,
        topic_id=result.topic_id,
        topic_title=result.topic_title,
        script_id=result.script_id,
        script_qa_result_id=result.script_qa_result_id,
        scenes=(),
        overall_status="failed",
        provenance=result.provenance,
        created_at=result.created_at,
        run_id=result.run_id,
    )
    issues, _ = _run(empty, plan)
    found = _by_check(issues, "structure_integrity")
    assert any("no scene records" in issue.message for issue in found)
    rows = list(result.scenes)
    rows[0], rows[1] = rows[1], rows[0]  # orders no longer contiguous 1..N
    issues, _ = _run(dataclasses.replace(result, scenes=tuple(rows)), plan)
    found = _by_check(issues, "structure_integrity")
    assert any("contiguous" in issue.message for issue in found)


def test_reference_integrity():
    plan, result = healthy_pair()
    issues, _ = _run(result_phantom_scene(plan, result), plan)
    found = _by_check(issues, "reference_integrity")
    assert len(found) == 1 and found[0].gate == "hard"
    assert "scn_99" in found[0].message


def test_provider_isolation():
    plan, result = healthy_pair()
    issues, _ = _run(result_provider_marker(plan, result), plan)
    found = _by_check(issues, "provider_isolation")
    assert len(found) >= 1
    assert all(issue.gate == "hard" for issue in found)


def test_overall_status_consistency():
    plan, result = healthy_pair()
    issues, _ = _run(result_bad_overall_status(plan, result), plan)
    found = _by_check(issues, "overall_status_consistency")
    assert len(found) == 1 and found[0].gate == "hard"


def test_mock_recognition():
    plan, result = healthy_pair()
    issues, _ = _run(result_mock_with_uri(plan, result), plan)
    found = _by_check(issues, "mock_recognition")
    assert len(found) == 1 and found[0].gate == "soft"
    issues, _ = _run(result_non_mock_artifact(plan, result), plan)
    found = _by_check(issues, "mock_recognition")
    assert any(issue.gate == "hard" for issue in found)


def test_media_honesty():
    plan, result = healthy_pair()
    issues, _ = _run(result_media_claim(plan, result), plan)
    found = _by_check(issues, "media_honesty")
    assert len(found) == 1 and found[0].gate == "hard"
    assert "uploaded" in found[0].message


def test_generation_validation_rejects_broken_result():
    plan, result = healthy_pair()
    broken = result_timing_mismatch(plan, result)
    issues, _ = _run(broken, plan)
    found = _by_check(issues, "generation_validation")
    assert len(found) >= 1
    assert all(issue.gate == "hard" for issue in found)
    assert any("g0" in issue.message for issue in found)


def test_records_flags_per_check():
    plan, result = healthy_pair()
    issues, records = _run(result_failed_scene(plan, result), plan)
    record = next(record for record in records if record.check == "scene_failures")
    assert record.outcome == "REJECT"
    assert record.hard_errors == 1
    assert record.errors == 1
    assert record.category == "scene_failures"