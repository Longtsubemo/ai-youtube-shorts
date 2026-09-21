"""Versioned Stage-7 (video generation) models.

Stage 7 is the provider-independent boundary between the Stage-6 ProductionPlan
(the CONTENT -> VIDEO boundary, docs/DATA_CONTRACTS.md §10) and the eventual
real video-generation worker. The content pipeline depends on the
`VideoGenerator` abstraction (agents/video_generation/generator.py), never on a
specific API provider, GPU stack, or cloud vendor.

The Stage-7 artifact is a `VideoGenerationResult`: one record per ProductionPlan
scene capturing:

    ProductionPlan scene
        -> GenerationRequest        (what a provider is asked to render)
        -> generated artifact       (what came back)
        -> per-scene status         (completed / failed / pending)

Everything here is deterministic and offline. The initial implementation is a
MockVideoGenerator; the artifact contract is written so a future Colab-hosted /
open-source GPU worker can implement the same protocol without touching the
stage runner, the validator, or the data contracts.

Follows the project's artifact envelope (docs/DATA_CONTRACTS.md §1):
header (schema/version/created_at/run_id/stage/produced_by) + data payload.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agents.scene_planner.models import (
    TRANSITIONS,
    VISUAL_TYPES,
    AssetRequirement,
    OnScreenText,
)
from agents.seo.models import Provenance
from pipeline.state import Stage

SCHEMA = "VideoGenerationResult"
SCHEMA_VERSION = 1
STAGE = Stage.VIDEO_GENERATION.value

# Version of the Stage-7 ruleset/configuration (recorded in provenance).
GENERATION_CONFIG_VERSION = "video_generation.v1"

# Provider abstraction the pipeline actually depends on. `MockVideoGenerator`
# is the deterministic, offline implementation; a real provider replaces it
# without changing the contract.
MOCK_PROVIDER_NAME = "MockVideoGenerator"

# Per-scene statuses. `pending` only appears on an interrupted previous result a
# resume is working through; a finished synchronous generation never leaves a
# scene pending (it is always completed or failed).
SCENE_STATUS_COMPLETED = "completed"
SCENE_STATUS_FAILED = "failed"
SCENE_STATUS_PENDING = "pending"
SCENE_STATUSES = (SCENE_STATUS_COMPLETED, SCENE_STATUS_FAILED, SCENE_STATUS_PENDING)

# Whole-result statuses (see derive_overall_status).
RESULT_STATUS_COMPLETED = "completed"
RESULT_STATUS_PARTIAL = "partial"
RESULT_STATUS_FAILED = "failed"
RESULT_STATUS_PENDING = "pending"
RESULT_STATUSES = (
    RESULT_STATUS_COMPLETED,
    RESULT_STATUS_PARTIAL,
    RESULT_STATUS_FAILED,
    RESULT_STATUS_PENDING,
)

# The `produced_by` envelope helper records which generator produced the result.
PRODUCED_BY = {"prompt_version": None, "provider": None, "model": MOCK_PROVIDER_NAME}

# Note attached to every mock artifact. It makes unambiguously clear that no
# video bytes exist: the record is a deterministic stand-in, never a real asset.
MOCK_ARTIFACT_NOTE = (
    "MOCK artifact: no video bytes were produced. This record is a deterministic "
    "offline stand-in for a future video provider's generated asset."
)


class VideoGenerationError(ValueError):
    """Raised when a video-generation artifact is missing, malformed, or
    inconsistent, or when resume/retry state cannot be applied safely."""


def _require(value: Any, name: str) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise VideoGenerationError(f"missing required field: {name}")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def video_generation_result_id_for(run_id: str, production_plan_id: str) -> str:
    """Stable VideoGenerationResult id per (run_id, production_plan_id)."""
    digest = hashlib.sha1(
        f"{run_id or ''}:{production_plan_id or ''}".encode("utf-8")
    ).hexdigest()[:12]
    return f"vg_{digest}"


def mock_artifact_id_for(run_id: str, scene_id: str) -> str:
    """Stable mock artifact id per (run_id, scene_id); deterministic."""
    digest = hashlib.sha1(
        f"{run_id or ''}:{scene_id or ''}".encode("utf-8")
    ).hexdigest()[:12]
    return f"mockva_{digest}"


# --------------------------------------------------------------------------- #
# Generation request
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GenerationRequest:
    """The provider-independent request snapshot for one ProductionPlan scene.

    This is what a real video worker would be asked to render: the scene's
    timing, narration, visual treatment, and provider-independent style. It
    carries no model names, no API fields, no provider syntax (`docs/
    DATA_CONTRACTS.md` §10 rules also apply to this snapshot).
    """

    scene_id: str
    order: int
    duration_seconds: float
    visual_type: str
    visual_description: str
    generation_prompt: str = ""
    negative_prompt: str = ""
    narration: str = ""
    claim_refs: tuple[str, ...] = ()
    script_line_ids: tuple[str, ...] = ()
    on_screen_text: tuple[OnScreenText, ...] = ()
    transition: str = "cut"
    visual_continuity_group: str = ""
    asset_requirements: tuple[AssetRequirement, ...] = ()

    def validate(self) -> "GenerationRequest":
        _require(self.scene_id, "scene_id")
        if self.order < 1:
            raise VideoGenerationError(
                f"request {self.scene_id} has invalid order {self.order}"
            )
        if self.duration_seconds <= 0:
            raise VideoGenerationError(
                f"request {self.scene_id} has non-positive "
                f"duration_seconds {self.duration_seconds}"
            )
        if self.visual_type not in VISUAL_TYPES:
            allowed = ", ".join(VISUAL_TYPES)
            raise VideoGenerationError(
                f"request {self.scene_id} has unknown visual_type "
                f"'{self.visual_type}' (allowed: {allowed})"
            )
        if self.transition not in TRANSITIONS:
            allowed = ", ".join(TRANSITIONS)
            raise VideoGenerationError(
                f"request {self.scene_id} has unknown transition "
                f"'{self.transition}' (allowed: {allowed})"
            )
        for item in self.on_screen_text:
            item.validate()
        for asset in self.asset_requirements:
            asset.validate()
        return self

    @classmethod
    def from_scene(cls, scene) -> "GenerationRequest":
        """Copy a ProductionPlan scene into a provider-independent request.

        Structural invariants are enforced at result assembly time
        (`SceneGenerationResult.validate` -> `request.validate()`); whether a
        request is actually generatable is a separate concern checked by the
        generator (`request_issue` in `mock_generator.py`).
        """
        return cls(
            scene_id=scene.scene_id,
            order=scene.order,
            duration_seconds=scene.duration_seconds,
            visual_type=scene.visual_type,
            visual_description=scene.visual_description,
            generation_prompt=scene.generation_prompt,
            negative_prompt=scene.negative_prompt,
            narration=scene.narration,
            claim_refs=scene.claim_refs,
            script_line_ids=scene.script_line_ids,
            on_screen_text=scene.on_screen_text,
            transition=scene.transition,
            visual_continuity_group=scene.visual_continuity_group,
            asset_requirements=scene.asset_requirements,
        )

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "order": self.order,
            "duration_seconds": self.duration_seconds,
            "visual_type": self.visual_type,
            "visual_description": self.visual_description,
            "generation_prompt": self.generation_prompt,
            "negative_prompt": self.negative_prompt,
            "narration": self.narration,
            "claim_refs": list(self.claim_refs),
            "script_line_ids": list(self.script_line_ids),
            "on_screen_text": [item.to_dict() for item in self.on_screen_text],
            "transition": self.transition,
            "visual_continuity_group": self.visual_continuity_group,
            "asset_requirements": [asset.to_dict() for asset in self.asset_requirements],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GenerationRequest":
        if not isinstance(data, dict):
            raise VideoGenerationError(f"generation request must be a mapping, got {data!r}")
        request = cls(
            scene_id=_require(data.get("scene_id"), "scene_id"),
            order=int(data.get("order") or 0),
            duration_seconds=float(data.get("duration_seconds") or 0.0),
            visual_type=_require(data.get("visual_type"), "visual_type"),
            visual_description=data.get("visual_description") or "",
            generation_prompt=data.get("generation_prompt") or "",
            negative_prompt=data.get("negative_prompt") or "",
            narration=data.get("narration") or "",
            claim_refs=tuple(data.get("claim_refs") or ()),
            script_line_ids=tuple(data.get("script_line_ids") or ()),
            on_screen_text=tuple(
                OnScreenText.from_dict(item) for item in data.get("on_screen_text") or ()
            ),
            transition=data.get("transition") or "cut",
            visual_continuity_group=data.get("visual_continuity_group") or "",
            asset_requirements=tuple(
                AssetRequirement.from_dict(asset)
                for asset in data.get("asset_requirements") or ()
            ),
        )
        return request.validate()


# --------------------------------------------------------------------------- #
# Generated artifact
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GeneratedArtifact:
    """The record of a generated scene artifact.

    The MockVideoGenerator records a placeholder (mock=True, empty uri) so the
    contract is exercised end-to-end without pretending real video exists. A
    future real provider fills `uri` and sets mock=False while keeping the same
    shape.
    """

    artifact_id: str
    asset_type: str
    subject: str
    continuity_group: str = ""
    duration_seconds: float = 0.0
    provider: str = MOCK_PROVIDER_NAME
    mock: bool = True
    uri: str = ""
    note: str = MOCK_ARTIFACT_NOTE

    def validate(self) -> "GeneratedArtifact":
        _require(self.artifact_id, "artifact_id")
        _require(self.asset_type, "asset_type")
        _require(self.subject, "subject")
        if self.duration_seconds <= 0:
            raise VideoGenerationError(
                f"artifact {self.artifact_id} has non-positive "
                f"duration_seconds {self.duration_seconds}"
            )
        if not isinstance(self.mock, bool):
            raise VideoGenerationError(
                f"artifact {self.artifact_id} mock flag must be a bool"
            )
        return self

    def to_dict(self) -> dict:
        return {
            "artifact_id": self.artifact_id,
            "asset_type": self.asset_type,
            "subject": self.subject,
            "continuity_group": self.continuity_group,
            "duration_seconds": self.duration_seconds,
            "provider": self.provider,
            "mock": self.mock,
            "uri": self.uri,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GeneratedArtifact":
        if not isinstance(data, dict):
            raise VideoGenerationError(f"generated artifact must be a mapping, got {data!r}")
        artifact = cls(
            artifact_id=_require(data.get("artifact_id"), "artifact_id"),
            asset_type=_require(data.get("asset_type"), "asset_type"),
            subject=_require(data.get("subject"), "subject"),
            continuity_group=data.get("continuity_group") or "",
            duration_seconds=float(data.get("duration_seconds") or 0.0),
            provider=data.get("provider") or MOCK_PROVIDER_NAME,
            mock=bool(data.get("mock", True)),
            uri=data.get("uri") or "",
            note=data.get("note") or MOCK_ARTIFACT_NOTE,
        )
        return artifact.validate()


# --------------------------------------------------------------------------- #
# Scene generation result
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SceneGenerationResult:
    """One scene's row in the VideoGenerationResult.

    `status` is completed / failed / pending. Failed rows carry a message and a
    `failure_retryable` flag so resume can distinguish retryable from permanent
    failures. `attempts` counts how many times generation has been attempted
    (across runs, mirroring docs/ARCHITECTURE.md §4.3 retry semantics).
    """

    scene_id: str
    order: int
    status: str
    request: GenerationRequest
    attempts: int = 1
    artifact: GeneratedArtifact | None = None
    failure: str = ""
    failure_retryable: bool | None = None
    note: str = ""

    def validate(self) -> "SceneGenerationResult":
        _require(self.scene_id, "scene_id")
        if self.status not in SCENE_STATUSES:
            allowed = ", ".join(SCENE_STATUSES)
            raise VideoGenerationError(
                f"scene {self.scene_id} has unknown status "
                f"'{self.status}' (allowed: {allowed})"
            )
        if self.status != SCENE_STATUS_PENDING and self.attempts < 1:
            raise VideoGenerationError(
                f"scene {self.scene_id} status '{self.status}' needs attempts >= 1"
            )
        if self.attempts < 0:
            raise VideoGenerationError(f"scene {self.scene_id} has negative attempts")
        self.request.validate()
        if self.status == SCENE_STATUS_COMPLETED:
            if self.artifact is None:
                raise VideoGenerationError(
                    f"completed scene {self.scene_id} has no generated artifact"
                )
            self.artifact.validate()
        if self.status != SCENE_STATUS_COMPLETED and self.artifact is not None:
            raise VideoGenerationError(
                f"non-completed scene {self.scene_id} may not carry an artifact"
            )
        if self.status == SCENE_STATUS_FAILED:
            if not self.failure.strip():
                raise VideoGenerationError(
                    f"failed scene {self.scene_id} has no failure message"
                )
            if self.failure_retryable is None:
                raise VideoGenerationError(
                    f"failed scene {self.scene_id} has no failure_retryable flag"
                )
        if self.status != SCENE_STATUS_FAILED and self.failure_retryable is not None:
            raise VideoGenerationError(
                f"scene {self.scene_id} is not failed but carries a "
                "failure_retryable flag"
            )
        return self

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "order": self.order,
            "status": self.status,
            "request": self.request.to_dict(),
            "attempts": self.attempts,
            "artifact": self.artifact.to_dict() if self.artifact else None,
            "failure": self.failure,
            "failure_retryable": self.failure_retryable,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SceneGenerationResult":
        if not isinstance(data, dict):
            raise VideoGenerationError(
                f"scene generation result must be a mapping, got {data!r}"
            )
        item = cls(
            scene_id=_require(data.get("scene_id"), "scene_id"),
            order=int(data.get("order") or 0),
            status=_require(data.get("status"), "status"),
            request=GenerationRequest.from_dict(data.get("request") or {}),
            attempts=int(data.get("attempts") or 0),
            artifact=GeneratedArtifact.from_dict(data["artifact"])
            if data.get("artifact")
            else None,
            failure=data.get("failure") or "",
            failure_retryable=data.get("failure_retryable"),
            note=data.get("note") or "",
        )
        return item.validate()


# --------------------------------------------------------------------------- #
# Deterministic failure scenarios (mock instrumentation)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GenerationFailure:
    """A deterministic failure scenario for one scene.

    Consumed by the MockVideoGenerator to simulate retryable/transient vs.
    permanent failures. A real provider reports failures natively; these
    scenarios exist so the resume/retry contract is testable offline.
    """

    scene_id: str
    message: str = "simulated provider failure"
    retryable: bool = True

    def validate(self) -> "GenerationFailure":
        _require(self.scene_id, "scene_id")
        _require(self.message, "message")
        if not isinstance(self.retryable, bool):
            raise VideoGenerationError(
                f"failure retryable flag must be a bool for {self.scene_id}"
            )
        return self


# --------------------------------------------------------------------------- #
# VideoGenerationConfig
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VideoGenerationConfig:
    """Deterministic knobs for the Stage-7 video-generation stage.

    `max_attempts` bounds retries per scene across runs (resume may retry a
    retryable failure until this limit; afterwards it is recorded as
    non-retryable). `duration_tolerance_seconds` is used by the validator to
    compare scene timings. `provider_isolation_denylist` is the vocabulary the
    validator scans the result contract for so the pipeline never becomes
    coupled to a specific API provider, API key, GPU/CUDA requirement, or cloud
    vendor. `failure_scenarios` lets the deterministic mock simulate failures.
    """

    schema_version: str = GENERATION_CONFIG_VERSION
    max_attempts: int = 3
    duration_tolerance_seconds: float = 0.001
    failure_scenarios: tuple[GenerationFailure, ...] = ()
    provider_isolation_denylist: tuple[str, ...] = (
        # video-generation model/vendor names
        "runway",
        "veo",
        "kling",
        "pika",
        "sora",
        "luma",
        "dall-e",
        "dalle",
        "midjourney",
        "stable diffusion",
        "stable-diffusion",
        "flux",
        "comfyui",
        "openai",
        "gpt-4",
        "gpt-5",
        "anthropic",
        "claude",
        "gemini",
        "replicate",
        "huggingface",
        "hugging face",
        "fireworks",
        # cloud / GPU execution assumptions
        "aws",
        "azure",
        "gcp",
        "google cloud",
        "cloud gpu",
        "colab",
        "cuda",
        "gpu",
        "tpu",
        "vram",
        # API keys / auth / runtime coupling
        "api key",
        "api-key",
        "api_key",
        "apikey",
        "bearer",
        "authorization",
        "secret",
        "access_token",
        "auth_token",
        "endpoint",
        "base_url",
    )

    def validate(self) -> "VideoGenerationConfig":
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.duration_tolerance_seconds <= 0:
            raise ValueError("duration_tolerance_seconds must be > 0")
        empty = [term for term in self.provider_isolation_denylist if not term.strip()]
        if empty:
            raise ValueError("provider_isolation_denylist contains empty terms")
        for failure in self.failure_scenarios:
            failure.validate()
        return self


# --------------------------------------------------------------------------- #
# Overall-status derivation
# --------------------------------------------------------------------------- #


def derive_overall_status(scenes) -> str:
    """Whole-result status from per-scene statuses.

        all completed  -> completed
        none completed -> pending if nothing attempted yet, else failed
        mixed          -> partial
    """
    statuses = set(scene.status for scene in scenes)
    if statuses == {SCENE_STATUS_COMPLETED}:
        return RESULT_STATUS_COMPLETED
    if SCENE_STATUS_COMPLETED not in statuses:
        if statuses == {SCENE_STATUS_PENDING}:
            return RESULT_STATUS_PENDING
        return RESULT_STATUS_FAILED
    return RESULT_STATUS_PARTIAL


# --------------------------------------------------------------------------- #
# VideoGenerationResult
# --------------------------------------------------------------------------- #


@dataclass
class VideoGenerationResult:
    """The Stage-7 artifact: per-scene generation status over a ProductionPlan.

    Written to `data/<run_id>/videos/video_generation_result.json`. The result
    is the provider-independent answer to the ProductionPlan: one row per scene
    with the request, the produced artifact, and the status, so a resume/retry
    run can preserve completed scenes and retry only what actually failed.
    """

    video_generation_result_id: str
    production_plan_id: str
    topic_id: str
    topic_title: str
    script_id: str
    script_qa_result_id: str
    scenes: tuple[SceneGenerationResult, ...]
    overall_status: str
    provenance: Provenance | None = None
    resumed_from_result_id: str = ""
    created_at: str = field(default_factory=_utc_now)
    schema: str = SCHEMA
    schema_version: int = SCHEMA_VERSION
    stage: str = STAGE
    run_id: str = ""
    produced_by: dict = field(default_factory=lambda: dict(PRODUCED_BY))

    @property
    def scene_ids(self) -> list[str]:
        return [row.scene_id for row in self.scenes]

    def validate(self) -> "VideoGenerationResult":
        _require(self.video_generation_result_id, "video_generation_result_id")
        _require(self.production_plan_id, "production_plan_id")
        _require(self.topic_id, "topic_id")
        _require(self.topic_title, "topic_title")
        _require(self.script_id, "script_id")
        if not self.scenes:
            raise VideoGenerationError(
                "video generation result must contain at least one scene record"
            )
        scene_ids = [row.scene_id for row in self.scenes]
        if len(set(scene_ids)) != len(scene_ids):
            raise VideoGenerationError("duplicate scene_id in video generation result")
        if [row.order for row in self.scenes] != list(range(1, len(self.scenes) + 1)):
            raise VideoGenerationError(
                "scene orders must be the contiguous sequence 1..N"
            )
        for row in self.scenes:
            row.validate()
        if self.overall_status not in RESULT_STATUSES:
            allowed = ", ".join(RESULT_STATUSES)
            raise VideoGenerationError(
                f"unknown overall_status '{self.overall_status}' (allowed: {allowed})"
            )
        derived = derive_overall_status(self.scenes)
        if self.overall_status != derived:
            raise VideoGenerationError(
                f"overall_status '{self.overall_status}' does not match "
                f"per-scene statuses (derived '{derived}')"
            )
        if self.provenance is None:
            raise VideoGenerationError(
                "video generation result provenance is required"
            )
        return self

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "run_id": self.run_id,
            "stage": self.stage,
            "produced_by": dict(self.produced_by),
            "data": {
                "video_generation_result_id": self.video_generation_result_id,
                "production_plan_id": self.production_plan_id,
                "topic_id": self.topic_id,
                "topic_title": self.topic_title,
                "script_id": self.script_id,
                "script_qa_result_id": self.script_qa_result_id,
                "scenes": [row.to_dict() for row in self.scenes],
                "overall_status": self.overall_status,
                "resumed_from_result_id": self.resumed_from_result_id,
                "provenance": self.provenance.to_dict() if self.provenance else None,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "VideoGenerationResult":
        header = dict(data)
        payload = header.pop("data")
        result = cls(
            video_generation_result_id=_require(
                payload.get("video_generation_result_id"), "video_generation_result_id"
            ),
            production_plan_id=_require(
                payload.get("production_plan_id"), "production_plan_id"
            ),
            topic_id=_require(payload.get("topic_id"), "topic_id"),
            topic_title=_require(payload.get("topic_title"), "topic_title"),
            script_id=_require(payload.get("script_id"), "script_id"),
            script_qa_result_id=_require(
                payload.get("script_qa_result_id"), "script_qa_result_id"
            ),
            scenes=tuple(
                SceneGenerationResult.from_dict(item) for item in payload.get("scenes") or []
            ),
            overall_status=_require(payload.get("overall_status"), "overall_status"),
            resumed_from_result_id=payload.get("resumed_from_result_id") or "",
            provenance=Provenance.from_dict(payload["provenance"])
            if payload.get("provenance")
            else None,
            schema=header.get("schema", SCHEMA),
            schema_version=header.get("schema_version", SCHEMA_VERSION),
            created_at=header["created_at"],
            stage=header.get("stage", STAGE),
            run_id=header.get("run_id", ""),
            produced_by=header.get("produced_by") or dict(PRODUCED_BY),
        )
        return result.validate()

    @classmethod
    def from_json(cls, text: str) -> "VideoGenerationResult":
        return cls.from_dict(json.loads(text))