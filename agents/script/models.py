"""Versioned Stage-4 (script) models: ScriptLine, Script, ScriptError.

Stage 4 turns ONLY the fact-check result's `approved_claims` into a 30-60s
YouTube Shorts explainer script. The boundary is strict (docs/FACT_VERIFICATION.md
"consumption boundary"): rejected and REVIEW claims never reach this stage, and
every factual line must cite the approved claim(s) it came from so the audit
chain script line -> claim -> source -> evidence stays intact.

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

SCHEMA = "Script"
SCHEMA_VERSION = 1
STAGE = Stage.SCRIPT_GENERATION.value

# Allowed line purposes. Factual lines (FACTUAL_PURPOSES) MUST carry claim_refs;
# framing lines (hook/context/transition/cta) are boilerplate and carry none.
PURPOSE_VALUES = (
    "hook",
    "context",
    "explanation",
    "fact",
    "transition",
    "conclusion",
    "call_to_action",
)
FACTUAL_PURPOSES = ("fact", "explanation", "conclusion")

# The `produced_by` envelope helper records which generator produced the script.
PRODUCED_BY = {
    "prompt_version": None,
    "provider": None,
    "model": "MockScriptGenerator",
}

# Rounding tolerance for duration checks (derived durations vs declared total).
DURATION_TOLERANCE_SECONDS = 0.5


class ScriptError(ValueError):
    """Raised when a script artifact is missing, malformed, or inconsistent."""


def _require(value: Any, name: str) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ScriptError(f"missing required field: {name}")
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
        raise ScriptError(f"{name} must be an ISO date/datetime, got {text!r}") from None
    return text


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def script_id_for(run_id: str, topic_id: str) -> str:
    """Stable Script id per (run_id, topic_id); deterministic."""
    digest = hashlib.sha1(f"{run_id or ''}:{topic_id}".encode("utf-8")).hexdigest()[:12]
    return f"sc_{digest}"


def line_id_for(index: int) -> str:
    """Stable sequential line id; deterministic (ln_01..ln_NN)."""
    return f"ln_{int(index):02d}"


# --------------------------------------------------------------------------- #
# Script line
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ScriptLine:
    """One line of the script.

    `claim_refs` links the line to the approved claim(s) that support it;
    `estimated_duration_seconds` is that line's share of the total runtime,
    derived deterministically from its word count and the configured speaking
    rate (defaults: 150 words/min, 30-60s target).
    """

    line_id: str
    text: str
    claim_refs: tuple[str, ...]
    purpose: str
    estimated_duration_seconds: float = 0.0

    def is_factual(self) -> bool:
        return self.purpose in FACTUAL_PURPOSES

    def validate(self) -> "ScriptLine":
        _require(self.line_id, "line_id")
        _require(self.text, "text")
        for claim_id in self.claim_refs:
            _require(claim_id, "claim_refs entry")
        if self.purpose not in PURPOSE_VALUES:
            allowed = ", ".join(PURPOSE_VALUES)
            raise ScriptError(f"unknown purpose '{self.purpose}' (allowed: {allowed})")
        if self.estimated_duration_seconds < 0:
            raise ScriptError(
                f"line {self.line_id} has negative estimated_duration_seconds"
            )
        return self

    def to_dict(self) -> dict:
        return {
            "line_id": self.line_id,
            "text": self.text,
            "claim_refs": list(self.claim_refs),
            "purpose": self.purpose,
            "estimated_duration_seconds": self.estimated_duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScriptLine":
        if not isinstance(data, dict):
            raise ScriptError(f"script line must be a mapping, got {data!r}")
        line = cls(
            line_id=_require(data.get("line_id"), "line_id"),
            text=_require(data.get("text"), "text"),
            claim_refs=tuple(data.get("claim_refs") or ()),
            purpose=_require(data.get("purpose"), "purpose"),
            estimated_duration_seconds=float(data.get("estimated_duration_seconds") or 0.0),
        )
        return line.validate()


# --------------------------------------------------------------------------- #
# Script
# --------------------------------------------------------------------------- #


@dataclass
class Script:
    """The Stage-4 artifact: a validated short-form explainer script.

    Every line that states a fact or draws a conclusion (`fact`, `explanation`,
    `conclusion`) must reference the approved claim(s) it came from. `claim_refs`
    are the only factual source; lines may contain no facts of the generator's
    own making.

    `used_claim_ids` / `unused_approved_claim_ids` split the result's approved
    claims so reviewers and downstream stages can see exactly what was consumed
    and what was left on the table (docs/DATA_CONTRACTS.md §8).
    """

    script_id: str
    topic_id: str
    topic_title: str
    title: str
    hook: str
    lines: tuple[ScriptLine, ...]
    estimated_duration_seconds: float
    used_claim_ids: tuple[str, ...]
    unused_approved_claim_ids: tuple[str, ...]
    result_id: str = ""
    package_id: str = ""
    provenance: Provenance | None = None
    created_at: str = field(default_factory=_utc_now)
    schema: str = SCHEMA
    schema_version: int = SCHEMA_VERSION
    stage: str = STAGE
    run_id: str = ""
    produced_by: dict = field(default_factory=lambda: dict(PRODUCED_BY))

    @property
    def line_ids(self) -> list[str]:
        return [line.line_id for line in self.lines]

    def all_claim_refs(self) -> set[str]:
        return {claim_id for line in self.lines for claim_id in line.claim_refs}

    def validate(self) -> "Script":
        _require(self.script_id, "script_id")
        _require(self.topic_id, "topic_id")
        _require(self.topic_title, "topic_title")
        _require(self.title, "title")
        _require(self.hook, "hook")
        if not self.lines:
            raise ScriptError("script must contain at least one line")
        line_ids = [line.line_id for line in self.lines]
        if len(set(line_ids)) != len(line_ids):
            raise ScriptError("duplicate line_id in lines")
        for line in self.lines:
            line.validate()
        if self.estimated_duration_seconds < 0:
            raise ScriptError("estimated_duration_seconds must be >= 0")
        derived = round(sum(line.estimated_duration_seconds for line in self.lines), 2)
        if abs(derived - self.estimated_duration_seconds) > DURATION_TOLERANCE_SECONDS:
            raise ScriptError(
                f"estimated_duration_seconds {self.estimated_duration_seconds} "
                f"does not match derived sum {derived}"
            )
        used = self.all_claim_refs()
        if set(self.used_claim_ids) != used:
            raise ScriptError(
                "used_claim_ids must be the exact set of claim_refs across all lines"
            )
        if set(self.unused_approved_claim_ids) & used:
            raise ScriptError(
                "unused_approved_claim_ids overlaps used_claim_ids; "
                "a claim may not be both used and unused"
            )
        if self.provenance is None:
            raise ScriptError("script provenance is required")
        for claim_id in set(self.used_claim_ids) | set(self.unused_approved_claim_ids):
            _require(claim_id, "claim_ids entry")
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
                "script_id": self.script_id,
                "topic_id": self.topic_id,
                "topic_title": self.topic_title,
                "title": self.title,
                "hook": self.hook,
                "estimated_duration_seconds": self.estimated_duration_seconds,
                "used_claim_ids": list(self.used_claim_ids),
                "unused_approved_claim_ids": list(self.unused_approved_claim_ids),
                "result_id": self.result_id,
                "package_id": self.package_id,
                "lines": [line.to_dict() for line in self.lines],
                "provenance": self.provenance.to_dict() if self.provenance else None,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "Script":
        header = dict(data)
        payload = header.pop("data")
        script = cls(
            script_id=_require(payload.get("script_id"), "script_id"),
            topic_id=_require(payload.get("topic_id"), "topic_id"),
            topic_title=_require(payload.get("topic_title"), "topic_title"),
            title=_require(payload.get("title"), "title"),
            hook=_require(payload.get("hook"), "hook"),
            lines=tuple(ScriptLine.from_dict(item) for item in payload.get("lines") or []),
            estimated_duration_seconds=float(payload.get("estimated_duration_seconds") or 0.0),
            used_claim_ids=tuple(payload.get("used_claim_ids") or ()),
            unused_approved_claim_ids=tuple(payload.get("unused_approved_claim_ids") or ()),
            result_id=payload.get("result_id") or "",
            package_id=payload.get("package_id") or "",
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
        return script.validate()

    @classmethod
    def from_json(cls, text: str) -> "Script":
        return cls.from_dict(json.loads(text))