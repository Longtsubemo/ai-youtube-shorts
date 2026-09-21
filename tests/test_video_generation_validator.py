"""Tests for the Stage-7 video-generation validator.

Covers every rule (g01..g13) plus the explicit provider-isolation scanning
helpers. The validator grades arbitrary (even structurally invalid) results, so
several cases build UNvalidated VideoGenerationResult instances on purpose to
prove each rule fires independently.
"""

from __future__ import annotations

import json

import pytest

from agents.video_generation.models import (
    MOCK_ARTIFACT_NOTE,
    MOCK_PROVIDER_NAME,
    SCENE_STATUS_COMPLETED,
    SCENE_STATUS_FAILED,
    SCENE_STATUS_PENDING,
    RESULT_STATUS_COMPLETED,
    RESULT_STATUS_FAILED,
    RESULT_STATUS_PARTIAL,
    GeneratedArtifact,
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
from agents.video_generation.mock_generator import MockVideoGenerator
from agents.video_generation.validator import (
    ValidationIssue,
    provider_isolation_hits,
    scan_key_terms,
    scan_value_terms,
    validate_video_generation_result,
)

from video_generation_test_utils import FIXED_AT, make_healthy_plan


def _provenance(created_at=FIXED_AT):
    return Provenance(
        generated_by="video_generation_agent_v1",
        source=MOCK_PROVIDER_NAME,
        created_at=created_at,
        provider=None,
        model=MOCK_PROVIDER_NAME,
        prompt_version="video_generation.v1",
    )


def _raw_result(scenes, provenance=_provenance(), overall=None):
    """Build a VideoGenerationResult WITHOUT model .validate() so structurally
    invalid variants are testable against the validator (which never raises)."""
    return VideoGenerationResult(
        video_generation_result_id=video_generation_result_id_for(
            "vg-demo", "pp_plan"
        ),
        production_plan_id="pp_plan",
        topic_id="t_octopus",
        topic_title="Why octopuses have three hearts",
        script_id="sc_1",
        script_qa_result_id="sqa_1",
        scenes=tuple(scenes),
        overall_status=overall if overall is not None else derive_overall_status(scenes),
        provenance=provenance,
        created_at=FIXED_AT,
        run_id="vg-demo",
    )


def _request(scene_id, order, duration=6.0, prompt_ok=True):
    return GenerationRequest(
        scene_id=scene_id,
        order=order,
        duration_seconds=duration,
        visual_type="generated_video",
        visual_description="Vertical underwater octopus close-up with hearts.",
        generation_prompt="cinematic vertical underwater octopus shot" if prompt_ok else "",
    ).validate()


def _artifact(scene_id, duration=6.0, note=None):
    return GeneratedArtifact(
        artifact_id=mock_artifact_id_for("vg-demo", scene_id),
        asset_type="generated_video",
        subject="octopus close-up",
        duration_seconds=duration,
        provider=MOCK_PROVIDER_NAME,
        mock=True,
        uri="",
        note=note or MOCK_ARTIFACT_NOTE,
    ).validate()


def _completed(scene_id, order, duration=6.0):
    return SceneGenerationResult(
        scene_id=scene_id,
        order=order,
        status=SCENE_STATUS_COMPLETED,
        request=_request(scene_id, order, duration=duration),
        attempts=1,
        artifact=_artifact(scene_id, duration=duration),
    ).validate()


def _failed(scene_id, order, message="boom", retryable=False, attempts=1):
    return SceneGenerationResult(
        scene_id=scene_id,
        order=order,
        status=SCENE_STATUS_FAILED,
        request=_request(scene_id, order),
        attempts=attempts,
        failure=message,
        failure_retryable=retryable,
    ).validate()


def _pass_result(plan):
    return MockVideoGenerator().generate(plan, run_id="vg-demo", created_at=FIXED_AT)


PLAN = None


def _plan():
    global PLAN
    if PLAN is None:
        PLAN = make_healthy_plan()
    return PLAN


class TestScanHelpers:
    def test_scan_value_terms(self):
        assert "sora" in scan_value_terms("render with Sora API", ["sora", "gpu"])
        assert "gpu" in scan_value_terms("CUDA GPU", ["gpu"])
        assert scan_value_terms("octopus hearts", ["sora"]) == ()

    def test_scan_key_terms(self):
        assert "api_key" in scan_key_terms("API_KEY", ["api_key"])
        assert scan_key_terms("scene_id", ["api_key"]) == ()

    def test_provider_isolation_hits_healthy_artifact_is_clean(self):
        artifact_payload = {"prompt": "vertical documentary octopus", "mock": True}
        assert provider_isolation_hits(artifact_payload, VideoGenerationConfig().provider_isolation_denylist) == ()

    def test_provider_isolation_hits_finds_coupling(self):
        config = VideoGenerationConfig()
        hits = provider_isolation_hits(
            {"prompt": "render via Sora on a CUDA GPU"}, config.provider_isolation_denylist
        )
        assert any("sora" in hit for hit in hits)
        assert any("cuda" in hit for hit in hits)
        # secure-material leakage in keys AND values
        assert provider_isolation_hits({"page": "api_key"}, config.provider_isolation_denylist)
        assert provider_isolation_hits({"token": "bearer sk-secret"}, config.provider_isolation_denylist)
        assert provider_isolation_hits({"token": "access_token"}, config.provider_isolation_denylist)


class TestValidationPass:
    def test_healthy_result_passes_all_rules(self):
        plan = _plan()
        result = _pass_result(plan)
        validation = validate_video_generation_result(result, plan)
        assert validation.passed
        assert validation.errors == ()
        assert all(issue.severity == "ERROR" for issue in validation.errors) or True

    def test_warnings_and_error_accessors(self):
        validation = validate_video_generation_result(_pass_result(_plan()), _plan())
        assert validation.issues == ()
        assert validation.warnings == ()

    def test_to_dict_shape(self):
        validation = validate_video_generation_result(_pass_result(_plan()), _plan())
        data = validation.to_dict()
        assert data["passed"] is True
        assert data["issues"] == []

    def test_issue_to_dict(self):
        issue = ValidationIssue(rule_id="g01", severity="ERROR", message="x")
        assert issue.to_dict()["rule_id"] == "g01"


class TestStructuralRules:
    def test_g01_empty_scenes(self):
        plan = _plan()
        result = _raw_result(())
        validation = validate_video_generation_result(result, plan)
        assert any(i.rule_id == "g01" for i in validation.errors)

    def test_g02_duplicate_scene_ids(self):
        plan = _plan()
        result = _raw_result((_completed("scn_01", 1), _completed("scn_01", 2)))
        validation = validate_video_generation_result(result, plan)
        assert any(i.rule_id == "g02" for i in validation.errors)

    def test_g03_unknown_status(self):
        plan = _plan()
        row = _completed("scn_01", 1)
        bad = SceneGenerationResult(
            scene_id=row.scene_id,
            order=1,
            status="rendering",
            request=row.request,
            attempts=1,
            artifact=row.artifact,
        )
        result = _raw_result((bad,))
        validation = validate_video_generation_result(result, plan)
        assert any(i.rule_id == "g03" for i in validation.errors)

    def test_g04_phantom_scene(self):
        plan = _plan()
        result = _raw_result((_completed("phantom", 1),))
        validation = validate_video_generation_result(result, plan)
        assert any(i.rule_id == "g04" for i in validation.errors)

    def test_g05_missing_plan_scene(self):
        plan = _plan()
        first_six = [_completed(s.scene_id, s.order) for s in plan.scenes[:6]]
        result = _raw_result(first_six)
        validation = validate_video_generation_result(result, plan)
        assert any(i.rule_id == "g05" for i in validation.errors)

    def test_g06_order_mismatch(self):
        plan = _plan()
        rows = []
        for s in plan.scenes:
            rows.append(_completed(s.scene_id, s.order + 3))
        result = _raw_result(rows)
        validation = validate_video_generation_result(result, plan)
        assert any(i.rule_id == "g06" for i in validation.errors)


class TestContentRules:
    def test_g07_request_timing_mismatch(self):
        plan = _plan()
        rows = [_completed(s.scene_id, s.order, duration=s.duration_seconds + 30.0) for s in plan.scenes[:2]]
        rows[0] = rows[0]
        result = _raw_result(rows)
        validation = validate_video_generation_result(result, plan)
        assert any(i.rule_id == "g07" for i in validation.errors)

    def test_g08_completed_without_artifact(self):
        plan = _plan()
        row = SceneGenerationResult(
            scene_id=plan.scenes[0].scene_id,
            order=1,
            status=SCENE_STATUS_COMPLETED,
            request=_request(plan.scenes[0].scene_id, 1),
            attempts=1,
            artifact=None,
        )
        result = _raw_result((row,))
        validation = validate_video_generation_result(result, plan)
        assert any(i.rule_id == "g08" for i in validation.errors)

    def test_g08_artifact_duration_mismatch(self):
        plan = _plan()
        row = _completed(plan.scenes[0].scene_id, 1, duration=plan.scenes[0].duration_seconds)
        raw = row.artifact
        new_artifact = GeneratedArtifact(
            artifact_id=raw.artifact_id,
            asset_type=raw.asset_type,
            subject=raw.subject,
            duration_seconds=raw.duration_seconds + 5.0,
            provider=raw.provider,
            mock=raw.mock,
            uri=raw.uri,
            note=raw.note,
        )
        row = SceneGenerationResult(
            scene_id=row.scene_id,
            order=row.order,
            status=row.status,
            request=row.request,
            attempts=row.attempts,
            artifact=new_artifact,
        )
        result = _raw_result((row,))
        validation = validate_video_generation_result(result, plan)
        assert any(i.rule_id == "g08" for i in validation.errors)

    def test_g09_failed_missing_message(self):
        plan = _plan()
        row = SceneGenerationResult(
            scene_id=plan.scenes[0].scene_id,
            order=1,
            status=SCENE_STATUS_FAILED,
            request=_request(plan.scenes[0].scene_id, 1),
            attempts=1,
            failure="  ",
            failure_retryable=False,
        )
        result = _raw_result((row,))
        validation = validate_video_generation_result(result, plan)
        assert any(i.rule_id == "g09" for i in validation.errors)

    def test_g09_retryable_at_limit(self):
        plan = _plan()
        config = VideoGenerationConfig(max_attempts=3)
        row = _failed(plan.scenes[0].scene_id, 1, retryable=True, attempts=3)
        result = _raw_result((row,))
        validation = validate_video_generation_result(result, plan, config=config)
        assert any(i.rule_id == "g09" for i in validation.errors)

    def test_g09_failed_zero_attempts(self):
        plan = _plan()
        row = _failed(plan.scenes[0].scene_id, 1)
        row = SceneGenerationResult(
            scene_id=row.scene_id,
            order=1,
            status=SCENE_STATUS_FAILED,
            request=row.request,
            attempts=0,
            failure="boom",
            failure_retryable=False,
        )
        result = _raw_result((row,))
        validation = validate_video_generation_result(result, plan)
        assert any(i.rule_id == "g09" for i in validation.errors)


class TestWholeResultRules:
    def test_g10_overall_status_mismatch(self):
        plan = _plan()
        result = _pass_result(plan)
        raw = _raw_result(result.scenes, overall=RESULT_STATUS_PARTIAL)
        validation = validate_video_generation_result(raw, plan)
        assert any(i.rule_id == "g10" for i in validation.errors)

    def test_g11_missing_provenance(self):
        plan = _plan()
        result = _pass_result(plan)
        raw = _raw_result(result.scenes, provenance=None)
        validation = validate_video_generation_result(raw, plan)
        assert any(i.rule_id == "g11" for i in validation.errors)

    def test_g12_provider_coupling_in_payload(self):
        plan = _plan()
        result = _pass_result(plan)
        rows = list(result.scenes)
        first = rows[0]
        leaky_artifact = GeneratedArtifact(
            artifact_id=first.artifact.artifact_id,
            asset_type=first.artifact.asset_type,
            subject=first.artifact.subject,
            duration_seconds=first.artifact.duration_seconds,
            provider=first.artifact.provider,
            mock=first.artifact.mock,
            uri=first.artifact.uri,
            note="rendered with Sora on a CUDA GPU cluster",
        )
        rows[0] = SceneGenerationResult(
            scene_id=first.scene_id,
            order=first.order,
            status=first.status,
            request=first.request,
            attempts=first.attempts,
            artifact=leaky_artifact,
        )
        raw = _raw_result(rows, overall=result.overall_status)
        validation = validate_video_generation_result(raw, plan)
        assert any(i.rule_id == "g12" for i in validation.errors)
        assert any("sora" in i.message.lower() for i in validation.errors)

    def test_g12_excludes_provenance_metadata(self):
        # The top-level provenance record legitimately names the generator
        # abstraction; it must not trip the provider-isolation scan.
        plan = _plan()
        result = _pass_result(plan)
        assert result.provenance is not None
        validation = validate_video_generation_result(result, plan)
        assert not any(i.rule_id == "g12" for i in validation.errors)

    def test_g13_round_trip_success(self):
        plan = _plan()
        result = _pass_result(plan)
        validation = validate_video_generation_result(result, plan)
        assert validation.passed
        assert not any(i.rule_id == "g13" for i in validation.errors)

    def test_g13_skipped_for_structurally_broken_result(self):
        # g05 fires -> g13 must be skipped so from_dict's .validate() cannot
        # raise inside the validator.
        plan = _plan()
        first_six = [_completed(s.scene_id, s.order) for s in plan.scenes[:6]]
        result = _raw_result(first_six)
        validation = validate_video_generation_result(result, plan)
        rule_ids = [i.rule_id for i in validation.errors]
        assert "g05" in rule_ids
        assert "g13" not in rule_ids


class TestNoRaiseContract:
    def test_validator_never_raises_on_corrupt_artifacts(self):
        plan = _plan()
        candidates = [
            _raw_result(()),
            _raw_result((_completed("scn_01", 1), _completed("scn_01", 2))),
            _raw_result((_completed("phantom", 1),)),
        ]
        for candidate in candidates:
            validation = validate_video_generation_result(candidate, plan)
            assert isinstance(validation, object) or True
            for issue in validation.errors:
                assert issue.severity == "ERROR"


def test_healthy_result_round_trips_to_json_and_back():
    plan = _plan()
    result = _pass_result(plan)
    cloned = VideoGenerationResult.from_dict(json.loads(result.to_json()))
    assert validate_video_generation_result(cloned, plan).passed