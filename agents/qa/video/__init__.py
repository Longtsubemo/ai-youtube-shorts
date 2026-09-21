"""Stage 8 - Video QA: quality gate over the Stage-7 VideoGenerationResult.

Docs: docs/DATA_CONTRACTS.md §11, docs/PROJECT_PROGRESS.md Stage 8,
docs/VIDEO_PIPELINE.md §6.
CLI: python -m agents.qa.video.

NOTE: this package imports only models + validator here (same convention as
agents.qa.script): the checks and stage runner live in submodules imported by
path so `python -m agents.qa.video` runs without a double-import RuntimeWarning.

Stage 8 is VIDEO QA, NOT video assembly: no media is assembled, edited, or
published. The artifact header `stage` label is "video"; pipeline STATE is
advanced to the existing generic Stage.QUALITY_CHECK (`STATE_STAGE` in
models.py) -- no Stage enum member was added or renamed.
"""

from __future__ import annotations

from agents.qa.video.models import (
    QA_GATES,
    QA_SEVERITIES,
    QA_VERDICTS,
    SCHEMA,
    SCHEMA_VERSION,
    STAGE,
    STATE_STAGE,
    VIDEO_QA_CATEGORIES,
    VideoQAConfig,
    VideoQAError,
    VideoQACheckRecord,
    VideoQAIssue,
    VideoQAResult,
    qa_result_id_for,
)
from agents.qa.video.validator import (
    VERDICT_PASS,
    VERDICT_REJECT,
    VERDICT_REVIEW,
    aggregate_verdict,
    evaluate_video,
)

__all__ = [
    "QA_GATES",
    "QA_SEVERITIES",
    "QA_VERDICTS",
    "SCHEMA",
    "SCHEMA_VERSION",
    "STAGE",
    "STATE_STAGE",
    "VIDEO_QA_CATEGORIES",
    "VERDICT_PASS",
    "VERDICT_REJECT",
    "VERDICT_REVIEW",
    "VideoQAConfig",
    "VideoQAError",
    "VideoQACheckRecord",
    "VideoQAIssue",
    "VideoQAResult",
    "aggregate_verdict",
    "evaluate_video",
    "qa_result_id_for",
]