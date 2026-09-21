"""Versioned Stage-8 (video QA) models: VideoQAIssue, VideoQACheckRecord, VideoQAResult.

Stage 8 is the deterministic, network-free, GPU-free, API-free, provider-
independent technical QUALITY GATE over the Stage-7 `VideoGenerationResult`
(docs/PROJECT_PROGRESS.md Stage 8 / docs/DATA_CONTRACTS.md §11 /
docs/VIDEO_PIPELINE.md §6). It is a gate, not a new content/creation stage:
it never inspects pixels, audio, or subtitles, because in the current
mock-first pipeline no assembled media bytes exist. Every check is provable
from the Stage-7 result records and the Stage-6 ProductionPlan metadata.

Stage label vs. pipeline state: the artifact header `stage` is the generic
QAResult label `"video"` (docs/DATA_CONTRACTS.md §11 uses `script`/`video`/
`policy` for the shared QA shape). The numbered Stage 8 advances pipeline
STATE to the existing generic `Stage.QUALITY_CHECK` (pipeline/state.py) -
there is deliberately NO dedicated `VIDEO_QA` enum member, and existing stage
names are never renamed. `STATE_STAGE` records that mapping explicitly.

Follows the project's artifact envelope (docs/DATA_CONTRACTS.md §1):
header (schema/version/created_at/run_id/stage/produced_by) + data payload.

Verdict + issue model mirrors Stage-5 Script QA (§8.1): issues carry a
severity (ERROR / WARNING / INFO) and a hard/soft gate; any HARD ERROR is a
fatal REJECT, any other ERROR is a REVIEW, and a clean result is PASS. There
is deliberately NO scored output: docs/DATA_CONTRACTS.md §11's generic
QAResult has no score field, so nothing is invented here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agents.seo.models import Provenance
from agents.video_generation.models import VideoGenerationConfig
from pipeline.state import Stage

SCHEMA = "VideoQAResult"
SCHEMA_VERSION = 1

# Artifact-header `stage` label. DATA_CONTRACTS.md §11 distinguishes
# script/video/policy QA by this label; mirroring §8.1, only the value is not
# a pipeline/state.py Stage member (that mapping is `STATE_STAGE`).
STAGE = "video"

# Numbered Stage 8 (Video QA) advances PipelineState to the EXISTING generic
# quality-check stage. No new Stage enum member is introduced, and existing
# members are not renamed. Documented in DATA_CONTRACTS.md §11 and
# PROJECT_PROGRESS.md Stage 8.
STATE_STAGE = Stage.QUALITY_CHECK

# Version of the Stage-8 ruleset/configuration (recorded in provenance).
QA_CONFIG_VERSION = "video_qa.v1"

QA_VERDICTS = ("PASS", "REVIEW", "REJECT")
QA_SEVERITIES = ("ERROR", "WARNING", "INFO")
QA_GATES = ("hard", "soft")

# The 15 Stage-8 video-QA check categories (docs/PROJECT_PROGRESS.md Stage 8 /
# docs/VIDEO_PIPELINE.md §6). Names equal the registered check names so the
# per-check records stay self-describing. Several checks are metadata/contract
# level: they verify what the Stage-7 result records PROVE, and they do not
# pretend to inspect an assembled video that does not exist yet.
VIDEO_QA_CATEGORIES = (
    "duration_window",
    "aspect_ratio_metadata",
    "scene_coverage",
    "scene_pending",
    "scene_failures",
    "scene_ordering",
    "timing_fidelity",
    "artifact_assignment",
    "structure_integrity",
    "reference_integrity",
    "provider_isolation",
    "generation_validation",
    "overall_status_consistency",
    "mock_recognition",
    "media_honesty",
)

# The `produced_by` envelope helper records the QA agent that ran.
PRODUCED_BY = {
    "prompt_version": None,
    "provider": None,
    "model": "VideoQAValidator",
}


class VideoQAError(ValueError):
    """Raised when a video-QA artifact is missing, malformed, or inconsistent."""


def _require(value: Any, name: str) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise VideoQAError(f"missing required field: {name}")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def qa_result_id_for(run_id: str, production_plan_id: str) -> str:
    """Stable VideoQAResult id per (run_id, production_plan_id); deterministic."""
    digest = hashlib.sha1(
        f"{run_id or ''}:{production_plan_id or ''}".encode("utf-8")
    ).hexdigest()[:12]
    return f"qa_{digest}"


# --------------------------------------------------------------------------- #
# Issue
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VideoQAIssue:
    """One Stage-8 QA finding.

    `severity`: ERROR (must be fixed before PASS), WARNING (should be weighed;
    may still allow a PASS), INFO (observation only).
    `gate`: whether a severity-ERROR issue is a HARD gate. Any hard ERROR
    fails the video (verdict REJECT); non-hard ERRORS cause REVIEW.
    `affected_scene_ids` / `affected_artifact_ids` keep the audit chain
    ProductionPlan -> scene -> VideoGenerationResult -> artifact -> QA issue.
    """

    issue_id: str = ""
    category: str = ""
    severity: str = "INFO"
    message: str = ""
    affected_scene_ids: tuple[str, ...] = ()
    affected_artifact_ids: tuple[str, ...] = ()
    gate: str = "soft"
    check: str = ""

    def validate(self) -> "VideoQAIssue":
        if self.category not in VIDEO_QA_CATEGORIES:
            raise VideoQAError(
                f"unknown category {self.category!r} (allowed: {VIDEO_QA_CATEGORIES})"
            )
        if self.severity not in QA_SEVERITIES:
            raise VideoQAError(
                f"unknown severity {self.severity!r} (allowed: {QA_SEVERITIES})"
            )
        if self.gate not in QA_GATES:
            raise VideoQAError(f"unknown gate {self.gate!r} (allowed: {QA_GATES})")
        if not self.message.strip():
            raise VideoQAError("issue message is required")
        return self

    @property
    def is_error(self) -> bool:
        return self.severity == "ERROR"

    @property
    def refs(self) -> tuple[str, ...]:
        """All referenced contract objects, matching §11's `refs`."""
        return tuple(dict.fromkeys((*self.affected_scene_ids, *self.affected_artifact_ids)))

    def to_dict(self) -> dict:
        return {
            "issue_id": self.issue_id,
            "category": self.category,
            "severity": self.severity,
            "message": self.message,
            "affected_scene_ids": list(self.affected_scene_ids),
            "affected_artifact_ids": list(self.affected_artifact_ids),
            "refs": list(self.refs),
            "gate": self.gate,
            "check": self.check,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VideoQAIssue":
        if not isinstance(data, dict):
            raise VideoQAError(f"QA issue must be a mapping, got {data!r}")
        issue = cls(
            issue_id=data.get("issue_id") or "",
            category=data.get("category") or "",
            severity=_require(data.get("severity"), "severity"),
            message=_require(data.get("message"), "message"),
            affected_scene_ids=tuple(data.get("affected_scene_ids") or ()),
            affected_artifact_ids=tuple(data.get("affected_artifact_ids") or ()),
            gate=data.get("gate") or "soft",
            check=data.get("check") or "",
        )
        return issue.validate()


# --------------------------------------------------------------------------- #
# Check run record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VideoQACheckRecord:
    """A single check's outcome so reviewers can see each rule ran.

    `outcome` follows the QAResult gates: PASS when the check produced no
    errors, REJECT when any hard ERROR fired, REVIEW for soft errors only.
    `hard_errors` counts fatal findings so the outcome is independently
    verifiable without re-walking the issues.
    """

    check: str
    category: str
    hard_gate: bool
    errors: int = 0
    warnings: int = 0
    infos: int = 0
    hard_errors: int = 0
    note: str = ""

    @property
    def passed(self) -> bool:
        return self.errors == 0

    @property
    def outcome(self) -> str:
        if self.errors == 0:
            return "PASS"
        if self.hard_errors > 0:
            return "REJECT"
        return "REVIEW"

    def to_dict(self) -> dict:
        return {
            "check": self.check,
            "category": self.category,
            "hard_gate": self.hard_gate,
            "errors": self.errors,
            "warnings": self.warnings,
            "infos": self.infos,
            "hard_errors": self.hard_errors,
            "passed": self.passed,
            "outcome": self.outcome,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VideoQACheckRecord":
        if not isinstance(data, dict):
            raise VideoQAError(f"QA check record must be a mapping, got {data!r}")
        return cls(
            check=_require(data.get("check"), "check"),
            category=data.get("category") or "",
            hard_gate=bool(data.get("hard_gate", False)),
            errors=int(data.get("errors") or 0),
            warnings=int(data.get("warnings") or 0),
            infos=int(data.get("infos") or 0),
            hard_errors=int(data.get("hard_errors") or 0),
            note=data.get("note") or "",
        )


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VideoQAConfig:
    """Deterministic knobs for the Stage-8 video-QA gate.

    All thresholds are documented in VIDEO_PIPELINE.md §6. The preferred
    duration window is the 30-60s Shorts target; hard bounds catch clearly
    unusable output. `duration_tolerance_seconds` comes from the Stage-7
    contract so timing comparisons are consistent. `provider_isolation_denylist`
    reuses the Stage-7 vocabulary: video QA must enforce the same provider
    independence on the content it is gating.
    """

    schema_version: str = QA_CONFIG_VERSION
    min_duration_seconds: float = 30.0
    max_duration_seconds: float = 60.0
    hard_min_duration_seconds: float = 15.0
    hard_max_duration_seconds: float = 90.0
    duration_tolerance_seconds: float = 0.001
    expected_aspect_ratio: str = "9:16"
    expected_resolution: str = "1080x1920"
    expected_fps: int = 30
    provider_isolation_denylist: tuple[str, ...] = tuple(
        VideoGenerationConfig().provider_isolation_denylist
    )

    def validate(self) -> "VideoQAConfig":
        if not 0 < self.hard_min_duration_seconds <= self.min_duration_seconds:
            raise ValueError("hard min duration must satisfy 0 < hard_min <= preferred_min")
        if not self.max_duration_seconds <= self.hard_max_duration_seconds:
            raise ValueError("hard_max_duration_seconds must be >= preferred max")
        if self.min_duration_seconds <= 0 or self.max_duration_seconds <= 0:
            raise ValueError("duration bounds must be positive")
        if self.duration_tolerance_seconds <= 0:
            raise ValueError("duration_tolerance_seconds must be > 0")
        if not self.expected_aspect_ratio.strip():
            raise ValueError("expected_aspect_ratio must not be empty")
        if self.expected_fps <= 0:
            raise ValueError("expected_fps must be > 0")
        empty = [term for term in self.provider_isolation_denylist if not term.strip()]
        if empty:
            raise ValueError("provider_isolation_denylist contains empty terms")
        return self


# --------------------------------------------------------------------------- #
# VideoQAResult
# --------------------------------------------------------------------------- #


@dataclass
class VideoQAResult:
    """The Stage-8 artifact: verdict + per-check outcomes + issues + provenance.

    Written to `data/<run_id>/production/qa_result.json`. `issues` /
    `warnings` / `infos` split the same issue stream by severity. `target_ref`
    is the ProductionPlan id the QA gate was applied to; the trace chain
    ProductionPlan -> scene -> VideoGenerationResult -> artifact -> QA issue is
    preserved through `checks` and every issue's affected ids.
    """

    qa_result_id: str
    target_ref: str
    topic_id: str
    topic_title: str
    script_id: str
    video_generation_result_id: str
    overall_verdict: str
    checked_at: str
    checks: tuple[VideoQACheckRecord, ...] = ()
    issues: tuple[VideoQAIssue, ...] = ()
    warnings: tuple[VideoQAIssue, ...] = ()
    infos: tuple[VideoQAIssue, ...] = ()
    resumed_from_qa_result_id: str = ""
    provenance: Provenance | None = None
    created_at: str = field(default_factory=_utc_now)
    schema: str = SCHEMA
    schema_version: int = SCHEMA_VERSION
    stage: str = STAGE
    run_id: str = ""
    produced_by: dict = field(default_factory=lambda: dict(PRODUCED_BY))

    @property
    def passed(self) -> bool:
        return self.overall_verdict == "PASS"

    def all_issues(self) -> tuple[VideoQAIssue, ...]:
        return self.issues + self.warnings + self.infos

    def validate(self) -> "VideoQAResult":
        _require(self.qa_result_id, "qa_result_id")
        _require(self.target_ref, "target_ref")
        _require(self.video_generation_result_id, "video_generation_result_id")
        if self.overall_verdict not in QA_VERDICTS:
            raise VideoQAError(
                f"overall_verdict must be one of {QA_VERDICTS}, got {self.overall_verdict!r}"
            )
        if self.provenance is None:
            raise VideoQAError("VideoQAResult provenance is required")
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
                "qa_result_id": self.qa_result_id,
                "target_ref": self.target_ref,
                "topic_id": self.topic_id,
                "topic_title": self.topic_title,
                "script_id": self.script_id,
                "video_generation_result_id": self.video_generation_result_id,
                "overall_verdict": self.overall_verdict,
                "checked_at": self.checked_at,
                "checks": [record.to_dict() for record in self.checks],
                "issues": [issue.to_dict() for issue in self.issues],
                "warnings": [issue.to_dict() for issue in self.warnings],
                "infos": [issue.to_dict() for issue in self.infos],
                "resumed_from_qa_result_id": self.resumed_from_qa_result_id,
                "provenance": self.provenance.to_dict() if self.provenance else None,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "VideoQAResult":
        header = dict(data)
        payload = header.pop("data")
        result = cls(
            qa_result_id=_require(payload.get("qa_result_id"), "qa_result_id"),
            target_ref=_require(payload.get("target_ref"), "target_ref"),
            topic_id=payload.get("topic_id") or "",
            topic_title=payload.get("topic_title") or "",
            script_id=payload.get("script_id") or "",
            video_generation_result_id=_require(
                payload.get("video_generation_result_id"), "video_generation_result_id"
            ),
            overall_verdict=_require(payload.get("overall_verdict"), "overall_verdict"),
            checked_at=_require(payload.get("checked_at"), "checked_at"),
            checks=tuple(
                VideoQACheckRecord.from_dict(item) for item in payload.get("checks") or []
            ),
            issues=tuple(VideoQAIssue.from_dict(item) for item in payload.get("issues") or []),
            warnings=tuple(
                VideoQAIssue.from_dict(item) for item in payload.get("warnings") or []
            ),
            infos=tuple(VideoQAIssue.from_dict(item) for item in payload.get("infos") or []),
            resumed_from_qa_result_id=payload.get("resumed_from_qa_result_id") or "",
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
    def from_json(cls, text: str) -> "VideoQAResult":
        return cls.from_dict(json.loads(text))