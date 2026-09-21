"""Stage 7 - Video Generation: ProductionPlan -> provider-independent records.

Docs: docs/DATA_CONTRACTS.md §10 (production boundary) + the Stage-7 section of
docs/PROJECT_PROGRESS.md.
CLI: python -m agents.video_generation.

NOTE: this package imports only the light models/validator surface here. The
generator protocol, mock generator, and stage runner live in
`agents.video_generation.generator` / `agents.video_generation.mock_generator` /
`agents.video_generation.stage` and are imported by submodule path so that
`python -m agents.video_generation` runs without a double-import RuntimeWarning
(mirrors agents.scene_planner.__init__).
"""
from __future__ import annotations

from agents.video_generation.models import (
    GENERATION_CONFIG_VERSION,
    MOCK_ARTIFACT_NOTE,
    MOCK_PROVIDER_NAME,
    PRODUCED_BY,
    RESULT_STATUS_COMPLETED,
    RESULT_STATUS_FAILED,
    RESULT_STATUS_PARTIAL,
    RESULT_STATUS_PENDING,
    RESULT_STATUSES,
    SCENE_STATUSES,
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
from agents.video_generation.validator import (
    VideoGenerationValidation,
    ValidationIssue,
    provider_isolation_hits,
    scan_key_terms,
    scan_value_terms,
    validate_video_generation_result,
)

__all__ = [
    "GENERATION_CONFIG_VERSION",
    "GeneratedArtifact",
    "GenerationFailure",
    "GenerationRequest",
    "MOCK_ARTIFACT_NOTE",
    "MOCK_PROVIDER_NAME",
    "PRODUCED_BY",
    "RESULT_STATUS_COMPLETED",
    "RESULT_STATUS_FAILED",
    "RESULT_STATUS_PARTIAL",
    "RESULT_STATUS_PENDING",
    "RESULT_STATUSES",
    "SCENE_STATUSES",
    "SceneGenerationResult",
    "ValidationIssue",
    "VideoGenerationConfig",
    "VideoGenerationError",
    "VideoGenerationResult",
    "VideoGenerationValidation",
    "derive_overall_status",
    "mock_artifact_id_for",
    "provider_isolation_hits",
    "scan_key_terms",
    "scan_value_terms",
    "validate_video_generation_result",
    "video_generation_result_id_for",
]