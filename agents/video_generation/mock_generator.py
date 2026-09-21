"""Deterministic, offline MockVideoGenerator and its explicit helper builders.

Stage-7's FIRST implementation of the `VideoGenerator` protocol
(`agents/video_generation/generator.py`). It is:

- reproducible (identical plans, configs, and inputs -> identical results)
- network-free (no requests/httpx/urllib, no sockets)
- GPU-free (no CUDA/torch)
- API-free (no external video-generation service, no API keys)
- content-safe (it never invents claim/content information: every artifact is
  derived only from the ProductionPlan scene it was asked to render, and every
  artifact record is flagged as a MOCK stand-in, never real video output)

Failures are simulated deterministically from `config.failure_scenarios`
(retryable/transient vs. permanent) and from requests that a real worker could
never satisfy (e.g. a generative type without a generation_prompt), so the
resume/retry contract is exercised offline without randomness.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from agents.scene_planner.models import GENERATIVE_VISUAL_TYPES, ProductionPlan
from agents.seo.models import Provenance
from agents.video_generation.models import (
    GENERATION_CONFIG_VERSION,
    MOCK_ARTIFACT_NOTE,
    MOCK_PROVIDER_NAME,
    PRODUCED_BY,
    SCHEMA,
    SCHEMA_VERSION,
    STAGE,
    SCENE_STATUS_COMPLETED,
    SCENE_STATUS_FAILED,
    SCENE_STATUS_PENDING,
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


def request_issue(request: GenerationRequest) -> str:
    """Describe why a request cannot be honored, or "" when it is generatable.

    These are NON-retryable defects: no amount of retrying fixes a request that
    a real worker could never satisfy. This is the generator's own defense in
    depth on top of the Stage-6 validator rules (which already require prompts
    for generative scenes); a future LLM/worker generator cannot make a broken
    request silently "succeed".
    """
    if request.duration_seconds <= 0:
        return (
            f"scene {request.scene_id} request has non-positive "
            f"duration {request.duration_seconds}"
        )
    if not request.visual_description.strip():
        return f"scene {request.scene_id} request has no visual_description"
    if request.visual_type in GENERATIVE_VISUAL_TYPES and not request.generation_prompt.strip():
        return (
            f"scene {request.scene_id} generative visual_type "
            f"'{request.visual_type}' requires a generation_prompt"
        )
    return ""


def build_mock_artifact(
    run_id: str,
    scene,
    request: GenerationRequest,
) -> GeneratedArtifact:
    """Deterministically build the clearly-marked MOCK artifact for one scene.

    Everything derives from the ProductionPlan scene (subject from the first
    asset requirement, or the visual description; duration mirrored from the
    request; no invented facts, no real file). The record is flagged `mock=True`
    with an empty `uri` so nobody mistakes it for produced video.
    """
    subject = request.asset_requirements[0].subject if request.asset_requirements else ""
    subject = subject or request.visual_description
    return GeneratedArtifact(
        artifact_id=mock_artifact_id_for(run_id, scene.scene_id),
        asset_type=request.visual_type,
        subject=subject,
        continuity_group=request.visual_continuity_group,
        duration_seconds=request.duration_seconds,
        provider=MOCK_PROVIDER_NAME,
        mock=True,
        uri="",
        note=MOCK_ARTIFACT_NOTE,
    ).validate()


def _join_note(existing: str, addition: str) -> str:
    return f"{existing}; {addition}" if existing else addition


@dataclass(frozen=True)
class _Attempt:
    """Internal outcome of one generation attempt for one scene."""

    completed: bool
    request: GenerationRequest
    artifact: GeneratedArtifact | None = None
    message: str = ""
    retryable: bool = False


class MockVideoGenerator:
    """Deterministic, offline implementation of the `VideoGenerator` protocol.

    Resume/retry semantics (docs/ARCHITECTURE.md §4.3):

      - completed scenes are preserved verbatim (never regenerated)
      - failed retryable scenes are retried while attempts < max_attempts
      - pending scenes are processed
      - non-retryable failures stay recorded
      - a retryable failure that reaches max_attempts is re-recorded as
        non-retryable so it is never retried again
    """

    def __init__(self, config: VideoGenerationConfig | None = None):
        self.config = config or VideoGenerationConfig().validate()

    def generate(
        self,
        plan: ProductionPlan,
        config: VideoGenerationConfig | None = None,
        previous: VideoGenerationResult | None = None,
        run_id: str = "",
        created_at: str = "",
    ) -> VideoGenerationResult:
        config = (config or self.config).validate()
        if previous is not None and previous.production_plan_id != plan.production_plan_id:
            raise VideoGenerationError(
                f"previous result {previous.video_generation_result_id} is for "
                f"production plan {previous.production_plan_id}, "
                f"not {plan.production_plan_id}"
            )
        previous_by_scene = {row.scene_id: row for row in previous.scenes} if previous else {}
        rows = [
            self._handle_scene(scene, previous_by_scene.get(scene.scene_id), config, run_id)
            for scene in plan.scenes
        ]
        scenes = tuple(rows)
        result = VideoGenerationResult(
            video_generation_result_id=video_generation_result_id_for(
                run_id, plan.production_plan_id
            ),
            production_plan_id=plan.production_plan_id,
            topic_id=plan.topic_id,
            topic_title=plan.topic_title,
            script_id=plan.script_id,
            script_qa_result_id=plan.script_qa_result_id,
            scenes=scenes,
            overall_status=derive_overall_status(scenes),
            resumed_from_result_id=(
                previous.video_generation_result_id if previous else ""
            ),
            provenance=Provenance(
                generated_by="video_generation_agent_v1",
                source=MOCK_PROVIDER_NAME,
                created_at=created_at or _utc_now(),
                provider=None,
                model=MOCK_PROVIDER_NAME,
                prompt_version=GENERATION_CONFIG_VERSION,
            ),
            schema=SCHEMA,
            schema_version=SCHEMA_VERSION,
            stage=STAGE,
            run_id=run_id,
            produced_by=dict(PRODUCED_BY),
            created_at=created_at or _utc_now(),
        )
        return result.validate()

    # ------------------------------------------------------------------ #
    # Internal helpers (deterministic)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _handle_scene(scene, prior, config: VideoGenerationConfig, run_id: str) -> SceneGenerationResult:
        """Apply resume/retry state for one scene, then attempt if needed."""
        if prior is None or prior.status == SCENE_STATUS_PENDING:
            starting_attempts = prior.attempts if prior is not None else 0
            return MockVideoGenerator._process_scene(
                scene, config, run_id, starting_attempts
            )
        if prior.status == SCENE_STATUS_COMPLETED:
            # Preserve finished work exactly; never regenerate successful scenes.
            return prior
        # Failed scene.
        if prior.failure_retryable is not True:
            # Permanent failure stays recorded; resume must not keep retrying.
            return prior
        if prior.attempts >= config.max_attempts:
            return replace(
                prior,
                failure_retryable=False,
                note=_join_note(prior.note, "retries exhausted; failure is now non-retryable"),
            )
        return MockVideoGenerator._process_scene(scene, config, run_id, prior.attempts)

    @staticmethod
    def _process_scene(
        scene, config: VideoGenerationConfig, run_id: str, starting_attempts: int
    ) -> SceneGenerationResult:
        attempts = starting_attempts + 1
        request = GenerationRequest.from_scene(scene)
        attempt = MockVideoGenerator._attempt(scene, request, config, run_id)
        if attempt.completed:
            return SceneGenerationResult(
                scene_id=scene.scene_id,
                order=scene.order,
                status=SCENE_STATUS_COMPLETED,
                request=request,
                attempts=attempts,
                artifact=attempt.artifact,
            ).validate()
        retries_left = config.max_attempts - attempts
        retryable = attempt.retryable and retries_left > 0
        if not attempt.retryable:
            note = "failure is non-retryable"
        elif not retryable:
            note = "retries exhausted; failure is now non-retryable"
        else:
            note = f"retryable failure recorded; {retries_left} retries remaining"
        return SceneGenerationResult(
            scene_id=scene.scene_id,
            order=scene.order,
            status=SCENE_STATUS_FAILED,
            request=request,
            attempts=attempts,
            failure=attempt.message,
            failure_retryable=retryable,
            note=note,
        ).validate()

    @staticmethod
    def _attempt(scene, request: GenerationRequest, config: VideoGenerationConfig, run_id: str) -> _Attempt:
        issue = request_issue(request)
        if issue:
            return _Attempt(completed=False, request=request, message=issue, retryable=False)
        failure = MockVideoGenerator._failure_for(scene.scene_id, config)
        if failure is not None:
            return _Attempt(
                completed=False,
                request=request,
                message=failure.message,
                retryable=failure.retryable,
            )
        return _Attempt(completed=True, request=request, artifact=build_mock_artifact(run_id, scene, request))

    @staticmethod
    def _failure_for(scene_id: str, config: VideoGenerationConfig) -> GenerationFailure | None:
        for failure in config.failure_scenarios:
            if failure.scene_id == scene_id:
                return failure
        return None


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()