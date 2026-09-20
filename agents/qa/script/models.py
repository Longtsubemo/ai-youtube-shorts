"""Versioned Stage-5 (script QA) models: QAIssue, ClaimCoverage, ScriptQAResult.

Stage 5 is QUALITY CONTROL for the Stage-4 Script artifact (docs/PROJECT_PROGRESS.md
Stage 5 / docs/DATA_CONTRACTS.md §8.1). It is NOT another research or
fact-finding stage and must not introduce new factual information: the approved
claims from Stage 3 remain the factual source of truth, and this stage evaluates
whether the script stays faithful to them, is coherent/clear/engaging, has
sane pacing and repetition, preserves uncertainty, and fits the Shorts format.

It never rewrites the script: it emits a PASS / REVISE / REJECT verdict plus
actionable issues/warnings that a future revision agent (or a human) consumes.

Follows the project's artifact envelope (docs/DATA_CONTRACTS.md §1):
header (schema/version/created_at/run_id/stage/produced_by) + data payload.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agents.seo.models import Provenance
from pipeline.state import Stage

SCHEMA = "ScriptQAResult"
SCHEMA_VERSION = 1
STAGE = Stage.SCRIPT_QA.value

# Version of the QA ruleset/configuration (recorded in provenance + envelope).
QA_CONFIG_VERSION = "script_qa.v1"

QA_VERDICTS = ("PASS", "REVISE", "REJECT")
QA_SEVERITIES = ("ERROR", "WARNING", "INFO")
QA_GATES = ("hard", "soft")

# The 13 QA check categories (docs/PROJECT_PROGRESS.md Stage 5).
QA_CATEGORIES = (
    "factual_fidelity",
    "claim_traceability",
    "claim_coverage",
    "length_duration",
    "pacing",
    "clarity",
    "repetition",
    "structure",
    "hook_quality",
    "uncertainty_preservation",
    "misleading_language",
    "cta_quality",
    "format_shorts",
)

# Scoring dimensions and their weights. Deterministic and documented; the
# score is an internal QA quality score, never a prediction of performance.
# Watch: uncertainty preservation gets its own dimension because dropping a
# hedge silently changes a claim's meaning.
QA_DIMENSIONS = (
    "factual_fidelity",
    "uncertainty",
    "traceability",
    "coverage",
    "clarity",
    "structure",
    "pacing",
    "hook",
    "repetition",
    "format",
)

DEFAULT_SCORE_WEIGHTS: dict[str, float] = {
    "factual_fidelity": 0.20,
    "uncertainty": 0.15,
    "traceability": 0.15,
    "coverage": 0.05,
    "clarity": 0.10,
    "structure": 0.10,
    "pacing": 0.10,
    "hook": 0.05,
    "repetition": 0.05,
    "format": 0.05,
}

# category -> scoring dimension
CATEGORY_DIMENSION: dict[str, str] = {
    "factual_fidelity": "factual_fidelity",
    "uncertainty_preservation": "uncertainty",
    "misleading_language": "clarity",
    "claim_traceability": "traceability",
    "claim_coverage": "coverage",
    "length_duration": "format",
    "pacing": "pacing",
    "clarity": "clarity",
    "repetition": "repetition",
    "structure": "structure",
    "hook_quality": "hook",
    "cta_quality": "format",
    "format_shorts": "format",
}

# The `produced_by` envelope helper records the QA agent that ran.
PRODUCED_BY = {
    "prompt_version": None,
    "provider": None,
    "model": "ScriptQAValidator",
}


class ScriptQAError(ValueError):
    """Raised when a QA artifact is missing, malformed, or inconsistent."""


def _require(value: Any, name: str) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ScriptQAError(f"missing required field: {name}")
    return value


def _check_iso(value: Any, name: str) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    candidate = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        datetime.fromisoformat(candidate)
    except ValueError:
        raise ScriptQAError(f"{name} must be an ISO date/datetime, got {text!r}") from None
    return text


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def qa_result_id_for(run_id: str, script_id: str) -> str:
    """Stable ScriptQAResult id per (run_id, script_id); deterministic."""
    digest = hashlib.sha1(f"{run_id or ''}:{script_id}".encode("utf-8")).hexdigest()[:12]
    return f"sqa_{digest}"


# --------------------------------------------------------------------------- #
# Issue
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class QAIssue:
    """One QA finding.

    `severity`: ERROR (must be fixed before PASS), WARNING (should be weighed;
    may still allow a PASS), INFO (quality suggestion only).
    `gate`: whether a severity-ERROR issue is a HARD gate. Any hard ERROR
    fails the script (verdict REJECT); non-hard ERRORS cause REVISE.
    `dimension`: the scoring dimension this issue feeds (derived from category).
    """

    issue_id: str = ""
    category: str = ""
    severity: str = "INFO"
    message: str = ""
    affected_line_ids: tuple[str, ...] = ()
    affected_claim_ids: tuple[str, ...] = ()
    gate: str = "soft"
    dimension: str = ""
    check: str = ""

    def validate(self) -> "QAIssue":
        if self.category not in QA_CATEGORIES:
            raise ScriptQAError(f"unknown category {self.category!r} (allowed: {QA_CATEGORIES})")
        if self.severity not in QA_SEVERITIES:
            raise ScriptQAError(f"unknown severity {self.severity!r} (allowed: {QA_SEVERITIES})")
        if self.gate not in QA_GATES:
            raise ScriptQAError(f"unknown gate {self.gate!r} (allowed: {QA_GATES})")
        if not self.message.strip():
            raise ScriptQAError("issue message is required")
        return self

    @property
    def is_error(self) -> bool:
        return self.severity == "ERROR"

    def to_dict(self) -> dict:
        return {
            "issue_id": self.issue_id,
            "category": self.category,
            "severity": self.severity,
            "message": self.message,
            "affected_line_ids": list(self.affected_line_ids),
            "affected_claim_ids": list(self.affected_claim_ids),
            "gate": self.gate,
            "dimension": self.dimension,
            "check": self.check,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QAIssue":
        if not isinstance(data, dict):
            raise ScriptQAError(f"QA issue must be a mapping, got {data!r}")
        issue = cls(
            issue_id=data.get("issue_id") or "",
            category=data.get("category") or "",
            severity=_require(data.get("severity"), "severity"),
            message=_require(data.get("message"), "message"),
            affected_line_ids=tuple(data.get("affected_line_ids") or ()),
            affected_claim_ids=tuple(data.get("affected_claim_ids") or ()),
            gate=data.get("gate") or "soft",
            dimension=data.get("dimension") or "",
            check=data.get("check") or "",
        )
        return issue.validate()


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ClaimCoverage:
    """How much of the approved material the script actually used.

    A Short is allowed to select its strongest claims: non-critical unused
    claims are fine. `coverage_ratio` is informational; only *critical* unused
    claims create a requirement (handled by the claim_coverage check).
    """

    approved_claim_ids: tuple[str, ...]
    used_claim_ids: tuple[str, ...]
    unused_approved_claim_ids: tuple[str, ...]
    critical_unused_claim_ids: tuple[str, ...] = ()
    coverage_ratio: float | None = None

    @classmethod
    def compute(
        cls,
        approved_claim_ids: tuple[str, ...],
        used_claim_ids: tuple[str, ...],
        critical_ids: tuple[str, ...] = (),
    ) -> "ClaimCoverage":
        approved = set(approved_claim_ids)
        used = set(used_claim_ids)
        ratio = round(len(approved & used) / len(approved), 3) if approved else None
        return cls(
            approved_claim_ids=tuple(sorted(approved)),
            used_claim_ids=tuple(sorted(used)),
            unused_approved_claim_ids=tuple(sorted(approved - used)),
            critical_unused_claim_ids=tuple(sorted(set(critical_ids) - used)),
            coverage_ratio=ratio,
        )

    def to_dict(self) -> dict:
        return {
            "approved_claim_ids": list(self.approved_claim_ids),
            "used_claim_ids": list(self.used_claim_ids),
            "unused_approved_claim_ids": list(self.unused_approved_claim_ids),
            "critical_unused_claim_ids": list(self.critical_unused_claim_ids),
            "coverage_ratio": self.coverage_ratio,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClaimCoverage":
        if not isinstance(data, dict):
            raise ScriptQAError(f"claim coverage must be a mapping, got {data!r}")
        return cls(
            approved_claim_ids=tuple(data.get("approved_claim_ids") or ()),
            used_claim_ids=tuple(data.get("used_claim_ids") or ()),
            unused_approved_claim_ids=tuple(data.get("unused_approved_claim_ids") or ()),
            critical_unused_claim_ids=tuple(data.get("critical_unused_claim_ids") or ()),
            coverage_ratio=(
                float(data["coverage_ratio"]) if data.get("coverage_ratio") is not None else None
            ),
        )


# --------------------------------------------------------------------------- #
# Check run record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class QACheckRecord:
    """A single check's outcome, so reviewers can see each rule ran."""

    check: str
    category: str
    dimension: str
    hard_gate: bool
    errors: int = 0
    warnings: int = 0
    infos: int = 0

    @property
    def passed(self) -> bool:
        return self.errors == 0

    def to_dict(self) -> dict:
        return {
            "check": self.check,
            "category": self.category,
            "dimension": self.dimension,
            "hard_gate": self.hard_gate,
            "errors": self.errors,
            "warnings": self.warnings,
            "infos": self.infos,
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QACheckRecord":
        if not isinstance(data, dict):
            raise ScriptQAError(f"QA check record must be a mapping, got {data!r}")
        return cls(
            check=_require(data.get("check"), "check"),
            category=data.get("category") or "",
            dimension=data.get("dimension") or "",
            hard_gate=bool(data.get("hard_gate", False)),
            errors=int(data.get("errors") or 0),
            warnings=int(data.get("warnings") or 0),
            infos=int(data.get("infos") or 0),
        )


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ScriptQAConfig:
    """Deterministic knobs for the QA stage (mirrors ScriptConfig).

    All thresholds are heuristics and are documented in DATA_CONTRACTS.md §8.1.
    `weights` is the score weighting (dimension -> weight, summing to 1).
    """

    schema_version: str = QA_CONFIG_VERSION
    words_per_minute: int = 150
    min_duration_seconds: float = 30.0
    max_duration_seconds: float = 60.0
    hard_min_duration_seconds: float = 15.0
    hard_max_duration_seconds: float = 90.0
    fidelity_coverage_threshold: float = 0.5
    min_lines: int = 3
    max_lines: int = 14
    max_line_words: int = 30
    hard_max_line_words: int = 45
    max_sentence_words: int = 25
    max_hook_words: int = 15
    max_cta_words: int = 14
    max_setup_words: int = 35
    max_block_words: int = 90
    require_hook: bool = True
    critical_unused_is_error: bool = True
    weights: tuple[tuple[str, float], ...] = tuple(DEFAULT_SCORE_WEIGHTS.items())

    def weight_map(self) -> dict[str, float]:
        return dict(self.weights)

    def validate(self) -> "ScriptQAConfig":
        if self.words_per_minute <= 0:
            raise ValueError("words_per_minute must be > 0")
        if not 0 < self.hard_min_duration_seconds <= self.min_duration_seconds:
            raise ValueError("hard min duration must satisfy 0 < hard_min <= preferred_min")
        if not self.max_duration_seconds <= self.hard_max_duration_seconds:
            raise ValueError("hard_max_duration_seconds must be >= preferred max")
        if self.hard_max_duration_seconds <= 0 or self.max_duration_seconds <= 0:
            raise ValueError("duration bounds must be positive")
        weights = self.weight_map()
        if set(weights) != set(QA_DIMENSIONS):
            raise ValueError(f"weights must cover exactly QA_DIMENSIONS: {QA_DIMENSIONS}")
        if abs(sum(weights.values()) - 1.0) > 1e-6:
            raise ValueError("weights must sum to 1.0")
        for value in weights.values():
            if value < 0:
                raise ValueError("weights must be >= 0")
        return self


# --------------------------------------------------------------------------- #
# ScriptQAResult
# --------------------------------------------------------------------------- #


@dataclass
class ScriptQAResult:
    """The Stage-5 artifact: verdict + checks + issues/warnings + score + provenance.

    `issues`/`warnings`/`infos` split the same issue stream by severity so
    consumers can take different actions on each.
    """

    qa_result_id: str
    script_id: str
    topic_id: str
    result_id: str
    package_id: str
    overall_verdict: str
    score: float
    score_detail: dict
    checks: tuple[QACheckRecord, ...] = ()
    issues: tuple[QAIssue, ...] = ()
    warnings: tuple[QAIssue, ...] = ()
    infos: tuple[QAIssue, ...] = ()
    approved_claim_coverage: ClaimCoverage | None = None
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

    def all_issues(self) -> tuple[QAIssue, ...]:
        return self.issues + self.warnings + self.infos

    def validate(self) -> "ScriptQAResult":
        _require(self.qa_result_id, "qa_result_id")
        _require(self.script_id, "script_id")
        _require(self.topic_id, "topic_id")
        _require(self.result_id, "result_id")
        if self.overall_verdict not in QA_VERDICTS:
            raise ScriptQAError(
                f"overall_verdict must be one of {QA_VERDICTS}, got {self.overall_verdict!r}"
            )
        if not 0.0 <= self.score <= 100.0:
            raise ScriptQAError(f"score must be in [0,100], got {self.score}")
        if self.provenance is None:
            raise ScriptQAError("ScriptQAResult provenance is required")
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
                "script_id": self.script_id,
                "topic_id": self.topic_id,
                "result_id": self.result_id,
                "package_id": self.package_id,
                "overall_verdict": self.overall_verdict,
                "score": self.score,
                "score_detail": dict(self.score_detail),
                "checks": [record.to_dict() for record in self.checks],
                "issues": [issue.to_dict() for issue in self.issues],
                "warnings": [issue.to_dict() for issue in self.warnings],
                "infos": [issue.to_dict() for issue in self.infos],
                "approved_claim_coverage": (
                    self.approved_claim_coverage.to_dict()
                    if self.approved_claim_coverage
                    else None
                ),
                "provenance": self.provenance.to_dict() if self.provenance else None,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "ScriptQAResult":
        header = dict(data)
        payload = header.pop("data")
        result = cls(
            qa_result_id=_require(payload.get("qa_result_id"), "qa_result_id"),
            script_id=_require(payload.get("script_id"), "script_id"),
            topic_id=_require(payload.get("topic_id"), "topic_id"),
            result_id=_require(payload.get("result_id"), "result_id"),
            package_id=payload.get("package_id") or "",
            overall_verdict=_require(payload.get("overall_verdict"), "overall_verdict"),
            score=float(payload.get("score") or 0.0),
            score_detail=payload.get("score_detail") or {},
            checks=tuple(QACheckRecord.from_dict(item) for item in payload.get("checks") or []),
            issues=tuple(QAIssue.from_dict(item) for item in payload.get("issues") or []),
            warnings=tuple(QAIssue.from_dict(item) for item in payload.get("warnings") or []),
            infos=tuple(QAIssue.from_dict(item) for item in payload.get("infos") or []),
            approved_claim_coverage=(
                ClaimCoverage.from_dict(payload["approved_claim_coverage"])
                if payload.get("approved_claim_coverage")
                else None
            ),
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
    def from_json(cls, text: str) -> "ScriptQAResult":
        return cls.from_dict(json.loads(text))