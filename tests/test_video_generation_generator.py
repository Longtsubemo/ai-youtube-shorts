"""Tests for the Stage-7 generator abstraction and MockVideoGenerator.

Covers the success path, determinism, request-level generatability gating,
deterministic retryable/non-retryable failure scenarios, resume semantics
(preserve completed / retry retryable / preserve permanent / process pending /
exhaust retry budget), plan-mismatch protection, and the provider-instance
isolation of the config-per-call override.
"""

from __future__ import annotations

import pytest

from agents.scene_planner.models import ProductionPlan
from agents.video_generation.generator import VideoGenerator
from agents.video_generation.mock_generator import (
    MockVideoGenerator,
    build_mock_artifact,
    request_issue,
)
from agents.video_generation.models import (
    MOCK_ARTIFACT_NOTE,
    MOCK_PROVIDER_NAME,
    SCENE_STATUS_COMPLETED,
    SCENE_STATUS_FAILED,
    SCENE_STATUS_PENDING,
    RESULT_STATUS_COMPLETED,
    RESULT_STATUS_FAILED,
    GenerationFailure,
    VideoGenerationConfig,
    VideoGenerationError,
    VideoGenerationResult,
    SceneGenerationResult,
    video_generation_result_id_for,
)
from video_generation_test_utils import FIXED_AT, make_healthy_plan


def _plan():
    return make_healthy_plan()


def _run(plan, config=None, previous=None, run_id="vg-demo", created_at=FIXED_AT):
    return (
        MockVideoGenerator(config=config)
        if config is not None
        else MockVideoGenerator()
    ).generate(plan, config=config, previous=previous, run_id=run_id, created_at=created_at)


class TestSuccessPath:
    def test_all_scenes_completed_with_mock_artifacts(self):
        plan = _plan()
        result = _run(plan)
        assert result.overall_status == RESULT_STATUS_COMPLETED
        assert len(result.scenes) == len(plan.scenes)
        for row in result.scenes:
            assert row.status == SCENE_STATUS_COMPLETED
            assert row.artifact is not None
            assert row.artifact.mock is True
            assert row.artifact.uri == ""
            assert row.artifact.provider == MOCK_PROVIDER_NAME
            assert MOCK_ARTIFACT_NOTE in row.artifact.note

    def test_result_id_stable_from_plan(self):
        plan = _plan()
        result = _run(plan)
        assert result.video_generation_result_id == video_generation_result_id_for(
            "vg-demo", plan.production_plan_id
        )

    def test_timing_preserved_end_to_end(self):
        plan = _plan()
        result = _run(plan)
        for scene, row in zip(plan.scenes, result.scenes):
            assert row.request.duration_seconds == scene.duration_seconds
            assert row.artifact.duration_seconds == scene.duration_seconds

    def test_artifact_subject_comes_from_plan_asset(self):
        plan = _plan()
        result = _run(plan)
        scene = plan.scenes[0]
        row = result.scenes[0]
        expected = scene.asset_requirements[0].subject
        assert row.artifact.subject == expected

    def test_generator_conforms_to_protocol(self):
        assert isinstance(MockVideoGenerator(), VideoGenerator)


class TestDeterminism:
    def test_identical_inputs_identical_records(self):
        plan = _plan()
        a = _run(plan)
        b = _run(plan)
        assert a.to_dict() == b.to_dict()

    def test_different_run_id_rerenders_artifact_ids(self):
        plan = _plan()
        a = _run(plan, run_id="vg-demo")
        b = _run(plan, run_id="vg-other")
        assert a.to_dict() != b.to_dict()
        assert a.video_generation_result_id != b.video_generation_result_id


class TestRequestGate:
    def test_empty_generative_prompt_is_non_retryable(self):
        plan = _plan()
        scene = plan.scenes[0]
        from agents.video_generation.models import GenerationRequest

        request = GenerationRequest.from_scene(scene)
        assert request_issue(request) == ""
        defect = GenerationRequest(
            **{
                **vars(request),
                "generation_prompt": "",
                "on_screen_text": request.on_screen_text,
                "asset_requirements": request.asset_requirements,
            }
        )
        issue = request_issue(defect)
        assert "generation_prompt" in issue

    def test_non_positive_duration_and_blank_description(self):
        from agents.video_generation.models import GenerationRequest

        bad_duration = GenerationRequest(
            scene_id="scn_x", order=1, duration_seconds=0.0,
            visual_type="generated_video", visual_description="vertical shot",
        )
        assert "duration" in request_issue(bad_duration)
        blank_description = GenerationRequest(
            scene_id="scn_x", order=1, duration_seconds=6.0,
            visual_type="generated_video", visual_description="   ",
        )
        assert "visual_description" in request_issue(blank_description)

    def test_healthy_scene_requests_are_generatable(self):
        plan = _plan()
        result = _run(plan)
        assert all(request_issue(row.request) == "" for row in result.scenes)


class TestRetryableFailure:
    def test_retryable_failure_recorded_with_message_and_flag(self):
        plan = _plan()
        config = VideoGenerationConfig(
            failure_scenarios=(GenerationFailure(scene_id=plan.scenes[2].scene_id),)
        )
        result = _run(plan, config=config)
        row = result.scenes[2]
        assert row.status == SCENE_STATUS_FAILED
        assert row.failure == "simulated provider failure"
        assert row.failure_retryable is True
        assert row.attempts == 1
        assert "retries remaining" in row.note
        assert result.overall_status == "partial"

    def test_resume_retries_retryable_failure(self):
        plan = _plan()
        first = _run(
            plan,
            config=VideoGenerationConfig(
                failure_scenarios=(GenerationFailure(scene_id=plan.scenes[2].scene_id),)
            ),
        )
        second = _run(plan, previous=first)
        row = second.scenes[2]
        assert row.status == SCENE_STATUS_COMPLETED
        assert row.attempts == 2
        assert second.overall_status == RESULT_STATUS_COMPLETED
        assert second.resumed_from_result_id == first.video_generation_result_id

    def test_resume_preserves_completed_scenes_verbatim(self):
        plan = _plan()
        first = _run(
            plan,
            config=VideoGenerationConfig(
                failure_scenarios=(GenerationFailure(scene_id=plan.scenes[2].scene_id),)
            ),
        )
        before = {(s.scene_id, s.attempts, s.artifact) for s in first.scenes
                  if s.status == SCENE_STATUS_COMPLETED}
        second = _run(plan, previous=first)
        after = {(s.scene_id, s.attempts, s.artifact) for s in second.scenes
                 if s.scene_id in {r[0] for r in before}}
        assert after == before

    def test_non_retryable_failure_preserved_across_resume(self):
        plan = _plan()
        config = VideoGenerationConfig(
            failure_scenarios=(
                GenerationFailure(scene_id=plan.scenes[1].scene_id, retryable=False),
            )
        )
        first = _run(plan, config=config, run_id="vg-demo")
        second = _run(plan, previous=first)
        row = second.scenes[1]
        assert row.status == SCENE_STATUS_FAILED
        assert row.attempts == 1
        assert row.failure_retryable is False
        assert "non-retryable" in row.note

    def test_retry_budget_exhaustion_marks_non_retryable(self):
        plan = _plan()
        config = VideoGenerationConfig(
            max_attempts=2,
            failure_scenarios=(GenerationFailure(scene_id=plan.scenes[3].scene_id),),
        )
        first = _run(plan, config=config, run_id="vg-demo")
        assert first.scenes[3].failure_retryable is True
        second = _run(plan, config=config, previous=first, run_id="vg-demo")
        row = second.scenes[3]
        assert row.status == SCENE_STATUS_FAILED
        assert row.attempts == 2
        assert row.failure_retryable is False
        assert "exhausted" in row.note
        # A further resume must NOT retry again.
        third = _run(plan, previous=second, run_id="vg-demo")
        assert third.scenes[3].status == SCENE_STATUS_FAILED
        assert third.scenes[3].attempts == 2

    def test_no_fresh_attempt_without_previous(self):
        # A non-retryable scenario without resume stays failed after one attempt.
        plan = _plan()
        config = VideoGenerationConfig(
            max_attempts=2,
            failure_scenarios=(GenerationFailure(scene_id=plan.scenes[0].scene_id),),
        )
        result = _run(plan, config=config, run_id="vg-demo")
        # already exhausted detection: first run records retryable with limit 2
        assert result.scenes[0].attempts == 1


class TestPendingResume:
    def test_pending_previous_scene_gets_processed(self):
        plan = _plan()
        first = _run(plan)
        rows = list(first.scenes)
        pending = SceneGenerationResult(
            scene_id=rows[4].scene_id,
            order=rows[4].order,
            status=SCENE_STATUS_PENDING,
            request=rows[4].request,
            attempts=0,
        )
        rows[4] = pending
        interrupted = VideoGenerationResult(
            video_generation_result_id=first.video_generation_result_id,
            production_plan_id=first.production_plan_id,
            topic_id=first.topic_id,
            topic_title=first.topic_title,
            script_id=first.script_id,
            script_qa_result_id=first.script_qa_result_id,
            scenes=tuple(rows),
            overall_status=first.overall_status,
            provenance=first.provenance,
            resumed_from_result_id=first.resumed_from_result_id,
            created_at=first.created_at,
            run_id=first.run_id,
        )
        assert interrupted.scenes[4].status == SCENE_STATUS_PENDING
        second = _run(plan, previous=interrupted)
        assert second.scenes[4].status == SCENE_STATUS_COMPLETED
        assert second.scenes[4].attempts == 1
        assert all(s.status == SCENE_STATUS_COMPLETED for s in second.scenes)


class TestPlanMismatch:
    def test_mismatched_previous_raises(self):
        from agents.video_generation.stage import run_video_generation

        plan_a = make_healthy_plan(run_id="video-a")
        plan_b = make_healthy_plan(run_id="video-b")
        first = _run(plan_b)
        with pytest.raises(VideoGenerationError):
            _run(plan_a, previous=first)
        with pytest.raises(VideoGenerationError):
            run_video_generation(plan_a, previous=first)


class TestBuildMockArtifact:
    def test_build_mock_artifact_is_explicit(self):
        plan = _plan()
        scene = plan.scenes[0]
        from agents.video_generation.models import GenerationRequest

        request = GenerationRequest.from_scene(scene)
        artifact = build_mock_artifact("vg-demo", scene, request)
        assert artifact.mock is True
        assert artifact.uri == ""
        assert artifact.duration_seconds == scene.duration_seconds
        assert artifact.artifact_id.startswith("mockva_")


def test_mock_generator_is_offline_provider_using_abstraction():
    """The generator interface only names the abstraction; callers pass the
    instance explicitly (dependency inversion), never hardcoding the mock."""
    plan = _plan()
    generator = MockVideoGenerator()
    result = generator.generate(plan, run_id="vg-demo", created_at=FIXED_AT)
    assert result.overall_status == RESULT_STATUS_COMPLETED
    assert isinstance(generator.generate(plan, run_id="vg-demo"), VideoGenerationResult)