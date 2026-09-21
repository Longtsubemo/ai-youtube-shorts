"""Tests for the Stage-7 video-generation data models.

Covers the versioned model/status surface, stable id derivation, request
snapshotting (pure copy, no raising), per-scene/overall status derivation,
serialization round-trips, and defensive config/failure invariants.
"""

from __future__ import annotations

import pytest

from agents.scene_planner.models import AssetRequirement, OnScreenText, Scene
from agents.video_generation.models import (
    GENERATION_CONFIG_VERSION,
    MOCK_ARTIFACT_NOTE,
    MOCK_PROVIDER_NAME,
    SCENE_STATUS_COMPLETED,
    SCENE_STATUS_FAILED,
    SCENE_STATUS_PENDING,
    RESULT_STATUS_COMPLETED,
    RESULT_STATUS_FAILED,
    RESULT_STATUS_PARTIAL,
    RESULT_STATUS_PENDING,
    GeneratedArtifact,
    GenerationFailure,
    GenerationRequest,
    SceneGenerationResult,
    VideoGenerationConfig,
    VideoGenerationError,
    VideoGenerationResult,
    derive_overall_status,
    mock_artifact_id_for,
    video_generation_result_id_for,
)
from agents.seo.models import Provenance

from video_generation_test_utils import FIXED_AT, make_healthy_plan


def _request(scene_id="scn_01", order=1, visual_type="generated_video"):
    return GenerationRequest(
        scene_id=scene_id,
        order=order,
        duration_seconds=6.0,
        visual_type=visual_type,
        visual_description="Vertical underwater octopus close-up with hearts.",
        generation_prompt="cinematic vertical underwater octopus shot" if visual_type in ("generated_video", "generated_image") else "",
    ).validate()


def _artifact(scene_id="scn_01"):
    return GeneratedArtifact(
        artifact_id=mock_artifact_id_for("vg-demo", scene_id),
        asset_type="generated_video",
        subject="octopus close-up",
        duration_seconds=6.0,
        provider=MOCK_PROVIDER_NAME,
        mock=True,
        uri="",
        note=MOCK_ARTIFACT_NOTE,
    ).validate()


def _completed_row(scene_id="scn_01", order=1):
    return SceneGenerationResult(
        scene_id=scene_id,
        order=order,
        status=SCENE_STATUS_COMPLETED,
        request=_request(scene_id, order),
        attempts=1,
        artifact=_artifact(scene_id),
    ).validate()


def _provenance(created_at=FIXED_AT):
    return Provenance(
        generated_by="video_generation_agent_v1",
        source=MOCK_PROVIDER_NAME,
        created_at=created_at,
        provider=None,
        model=MOCK_PROVIDER_NAME,
        prompt_version=GENERATION_CONFIG_VERSION,
    )


def _result(scenes=None, overall=None, run_id="vg-demo"):
    scenes = scenes or (_completed_row(),)
    return VideoGenerationResult(
        video_generation_result_id=video_generation_result_id_for(
            run_id, "pp_plan"
        ),
        production_plan_id="pp_plan",
        topic_id="t_octopus",
        topic_title="Why octopuses have three hearts",
        script_id="sc_1",
        script_qa_result_id="sqa_1",
        scenes=scenes,
        overall_status=overall or derive_overall_status(scenes),
        provenance=_provenance(),
        created_at=FIXED_AT,
        run_id=run_id,
    ).validate()


class TestIds:
    def test_result_id_stable_and_prefixed(self):
        a = video_generation_result_id_for("vg-demo", "pp_a")
        b = video_generation_result_id_for("vg-demo", "pp_a")
        c = video_generation_result_id_for("vg-demo", "pp_b")
        assert a == b
        assert a != c
        assert a.startswith("vg_")
        assert len(a) == len("vg_") + 12

    def test_artifact_id_stable_and_prefixed(self):
        a = mock_artifact_id_for("vg-demo", "scn_01")
        b = mock_artifact_id_for("vg-demo", "scn_01")
        assert a == b
        assert a != mock_artifact_id_for("vg-demo", "scn_02")
        assert a.startswith("mockva_")

    def test_id_pairs_are_distinct(self):
        result_id = video_generation_result_id_for("vg-demo", "pp_plan")
        artifact_id = mock_artifact_id_for("vg-demo", "scn_01")
        assert result_id != artifact_id


class TestGenerationRequest:
    def test_from_scene_is_pure_copy(self):
        scene = Scene(
            scene_id="scn_01",
            order=1,
            duration_seconds=6.0,
            script_line_ids=("sl_1",),
            narration="Octopuses have three hearts.",
            visual_type="generated_video",
            visual_description="Vertical underwater octopus close-up.",
            generation_prompt="cinematic vertical underwater shot",
            claim_refs=("fc_1",),
            on_screen_text=(OnScreenText(kind="lower_third", text="Three hearts"),),
            transition="crossfade",
            visual_continuity_group="octopus_hero",
            asset_requirements=(
                AssetRequirement(asset_type="graphic", subject="octopus", quantity=1),
            ),
            provenance={},
        )
        request = GenerationRequest.from_scene(scene)
        assert request.scene_id == "scn_01"
        assert request.duration_seconds == 6.0
        assert request.transition == "crossfade"
        assert request.on_screen_text == scene.on_screen_text
        assert request.asset_requirements == scene.asset_requirements

    def test_from_scene_never_raises_on_malformed_scene(self):
        # Generational requests are copied structurally; generation-time
        # validation catches problems when the result is assembled.
        scene = Scene(
            scene_id="scn_09",
            order=-5,  # invalid per request.validate()
            duration_seconds=-2.0,  # invalid per request.validate()
            script_line_ids=(),
            narration="",
            visual_type="generated_video",
            visual_description="   ",
            claim_refs=(),
            provenance={},
        )
        request = GenerationRequest.from_scene(scene)  # must NOT raise
        with pytest.raises(VideoGenerationError):
            request.validate()

    def test_validate_rejects_unknown_visual_type(self):
        with pytest.raises(VideoGenerationError):
            _request(visual_type="sora_video")

    def test_serialization_round_trip(self):
        request = _request()
        assert GenerationRequest.from_dict(request.to_dict()) == request


class TestGeneratedArtifact:
    def test_defaults_are_explicitly_mock(self):
        artifact = _artifact()
        assert artifact.mock is True
        assert artifact.uri == ""
        assert artifact.provider == MOCK_PROVIDER_NAME
        assert MOCK_ARTIFACT_NOTE in artifact.note

    def test_validate_rejects_missing_fields(self):
        with pytest.raises(VideoGenerationError):
            GeneratedArtifact(
                artifact_id="", asset_type="generated_video", subject="x"
            ).validate()

    def test_round_trip(self):
        artifact = _artifact()
        assert GeneratedArtifact.from_dict(artifact.to_dict()) == artifact


class TestSceneGenerationResult:
    def test_completed_row_requires_artifact(self):
        with pytest.raises(VideoGenerationError):
            SceneGenerationResult(
                scene_id="scn_01",
                order=1,
                status=SCENE_STATUS_COMPLETED,
                request=_request(),
                attempts=1,
                artifact=None,
            ).validate()

    def test_non_completed_row_may_not_carry_artifact(self):
        with pytest.raises(VideoGenerationError):
            SceneGenerationResult(
                scene_id="scn_01",
                order=1,
                status=SCENE_STATUS_FAILED,
                request=_request(),
                attempts=1,
                artifact=_artifact(),
                failure="nope",
                failure_retryable=False,
            ).validate()

    def test_failed_row_requires_message_and_flag(self):
        with pytest.raises(VideoGenerationError):
            SceneGenerationResult(
                scene_id="scn_01",
                order=1,
                status=SCENE_STATUS_FAILED,
                request=_request(),
                attempts=1,
            ).validate()

    def test_pending_row_allows_zero_attempts(self):
        row = SceneGenerationResult(
            scene_id="scn_05",
            order=5,
            status=SCENE_STATUS_PENDING,
            request=_request("scn_05", 5),
            attempts=0,
        ).validate()
        assert row.status == SCENE_STATUS_PENDING

    def test_round_trip(self):
        row = _completed_row()
        assert SceneGenerationResult.from_dict(row.to_dict()) == row


class TestDeriveOverallStatus:
    def test_all_completed(self):
        assert (
            derive_overall_status((_completed_row(), _completed_row("scn_02", 2)))
            == RESULT_STATUS_COMPLETED
        )

    def test_mixed_is_partial(self):
        failed = SceneGenerationResult(
            scene_id="scn_02",
            order=2,
            status=SCENE_STATUS_FAILED,
            request=_request("scn_02", 2),
            attempts=1,
            failure="boom",
            failure_retryable=True,
        )
        assert derive_overall_status((_completed_row(), failed)) == RESULT_STATUS_PARTIAL

    def test_all_failed(self):
        failed = SceneGenerationResult(
            scene_id="scn_01",
            order=1,
            status=SCENE_STATUS_FAILED,
            request=_request(),
            attempts=1,
            failure="boom",
            failure_retryable=False,
        )
        assert derive_overall_status((failed,)) == RESULT_STATUS_FAILED

    def test_all_pending(self):
        pending = SceneGenerationResult(
            scene_id="scn_01",
            order=1,
            status=SCENE_STATUS_PENDING,
            request=_request(),
            attempts=0,
        )
        assert derive_overall_status((pending,)) == RESULT_STATUS_PENDING


class TestVideoGenerationResult:
    def test_default_config_for_result_constants(self):
        config = VideoGenerationConfig().validate()
        assert config.schema_version == GENERATION_CONFIG_VERSION
        assert config.max_attempts == 3
        assert config.duration_tolerance_seconds > 0

    def test_result_serialization_round_trip_is_stable(self):
        result = _result()
        cloned = VideoGenerationResult.from_json(result.to_json())
        assert cloned == result

    def test_result_requires_contiguous_orders(self):
        scenes = (_completed_row("scn_01", 1), SceneGenerationResult(
            scene_id="scn_02",
            order=9,  # non-contiguous
            status=SCENE_STATUS_COMPLETED,
            request=_request("scn_02", 9),
            attempts=1,
            artifact=_artifact("scn_02"),
        ))
        with pytest.raises(VideoGenerationError):
            VideoGenerationResult(
                video_generation_result_id="vg_x",
                production_plan_id="pp_plan",
                topic_id="t",
                topic_title="T",
                script_id="sc_1",
                script_qa_result_id="sqa_1",
                scenes=scenes,
                overall_status=RESULT_STATUS_COMPLETED,
                provenance=_provenance(),
                run_id="vg-demo",
            ).validate()

    def test_result_rejects_status_mismatch(self):
        with pytest.raises(VideoGenerationError):
            _result(overall=RESULT_STATUS_PARTIAL)  # scenes are all completed

    def test_result_requires_provenance(self):
        with pytest.raises(VideoGenerationError):
            _result_ = _result()
            VideoGenerationResult(
                video_generation_result_id=_result_.video_generation_result_id,
                production_plan_id="pp_plan",
                topic_id="t_octopus",
                topic_title="Why octopuses have three hearts",
                script_id="sc_1",
                script_qa_result_id="sqa_1",
                scenes=_result_.scenes,
                overall_status=RESULT_STATUS_COMPLETED,
                provenance=None,
                run_id="vg-demo",
            ).validate()

    def test_result_header_and_payload_shape(self):
        result = _result()
        data = result.to_dict()
        assert data["schema"] == "VideoGenerationResult"
        assert data["stage"] == "video_generation"
        assert data["produced_by"]["model"] == MOCK_PROVIDER_NAME
        assert data["data"]["video_generation_result_id"].startswith("vg_")


class TestGenerationFailure:
    def test_defaults_retryable(self):
        failure = GenerationFailure(scene_id="scn_03").validate()
        assert failure.retryable is True
        assert failure.message == "simulated provider failure"

    def test_validate_requires_bool_flag(self):
        with pytest.raises(VideoGenerationError):
            GenerationFailure(scene_id="scn_03", retryable="yes").validate()


def test_healthy_plan_builder_reproducible():
    """The shared plan builder yields a deterministic 7-scene plan."""
    plan = make_healthy_plan()
    again = make_healthy_plan()
    assert plan.to_dict() == again.to_dict()
    assert len(plan.scenes) == 7
    assert [s.order for s in plan.scenes] == list(range(1, 8))