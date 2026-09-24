"""Versioned Stage-9 (policy/safety QA) models.

Stage 9 is the deterministic, network-free, GPU-free, API-free, provider-
independent pre-publication POLICY / SAFETY GATE over the end-to-end artifact
chain: Stage-4 Script, Stage-3 FactCheckResult, Stage-6 ProductionPlan,
Stage-7 VideoGenerationResult, and Stage-8 VideoQAResult
(docs/PROJECT_PROGRESS.md Stage 9 / docs/POLICY_AND_SAFETY.md / docs/
DATA_CONTRACTS.md §11). It is a gate, not a new creation stage: it evaluates
plans and metadata plus the deterministic policy registry in `rules.py`. Every
verdict is provable from the upstream artifacts; no external policy engine or
LLM is consulted, and going to YouTube/uploading is a LATER stage that this
stage deliberately does NOT implement.

Verdict model mirrors Stages 5/8: issues carry severity (ERROR / WARNING /
INFO) and a hard/soft gate; any hard ERROR is a fatal REJECT, any other ERROR
is a REVIEW, a clean result is PASS, and uncertainty NEVER resolves to PASS
(borderline classifications are soft errors -> REVIEW). There is no score
field and no YouTube-compliance guarantee.

Stage label vs. pipeline state: the artifact header `stage` is `"policy_qa"`
(per the Stage-9 task). The numbered Stage 9 advances pipeline STATE to the
existing generic `Stage.QUALITY_CHECK` (pipeline/state.py) -- the same mapping
Stage 8 uses; there is deliberately NO `POLICY_QA` enum member.
`STATE_STAGE` records that mapping explicitly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agents.qa.fact_checker.models import FactCheckResult
from agents.qa.policy.rules import (
    POLICY_CATEGORIES,
    POLICY_CHECK_ORDER,
    POLICY_RULES_BY_ID,
    POLICY_RULESET_NOTE,
    POLICY_VERSION,
)
from agents.qa.video.models import VideoQAResult
from agents.scene_planner.models import ProductionPlan
from agents.script.models import Script
from agents.seo.models import Provenance
from agents.video_generation.models import VideoGenerationResult
from pipeline.state import Stage

SCHEMA = "PolicyQAResult"
SCHEMA_VERSION = 1

#: Artifact-header `stage` label (per the Stage-9 task; documented in §11 and
#: PROJECT_PROGRESS.md Stage 9). Not a pipeline/state.py Stage member -- that
#: mapping is `STATE_STAGE`.
STAGE = "policy_qa"

#: Numbered Stage 9 (Policy QA) advances PipelineState to the EXISTING generic
#: quality-check stage, exactly as Stage 8 does. No new enum member.
STATE_STAGE = Stage.QUALITY_CHECK

#: Version of the Stage-9 ruleset (recorded in provenance and data.policy_version).
POLICY_QA_CONFIG_VERSION = "policy_qa.v1"

POLICY_VERDICTS = ("PASS", "REVIEW", "REJECT")
POLICY_SEVERITIES = ("ERROR", "WARNING", "INFO")
POLICY_GATES = ("hard", "soft")

#: The `produced_by` envelope helper records the QA agent that ran.
PRODUCED_BY = {
    "prompt_version": None,
    "provider": None,
    "model": "PolicyQAValidator",
}


class PolicyQAError(ValueError):
    """Raised when a policy-QA artifact is missing, malformed, or inconsistent."""


def _require(value: Any, name: str) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise PolicyQAError(f"missing required field: {name}")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def policy_result_id_for(run_id: str, target_ref: str) -> str:
    """Stable PolicyQAResult id per (run_id, target_ref); deterministic."""
    digest = hashlib.sha1(
        f"{run_id or ''}:{target_ref or ''}".encode("utf-8")
    ).hexdigest()[:12]
    return f"pr_{digest}"


# --------------------------------------------------------------------------- #
# Policy issue
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PolicyIssue:
    """One Stage-9 policy/safety finding.

    `severity`: ERROR (must resolve/justify before PASS), WARNING (should be
    weighed; may still allow a PASS), INFO (observation only).
    `gate`: whether a severity-ERROR issue is a HARD gate. Any hard ERROR
    fails (verdict REJECT); non-hard ERRORS cause REVIEW. Uncertainty is
    encoded as a soft ERROR so it can never resolve to PASS.
    `policy_rule`: the rule id + ruleset version that fired (P05@policy_rules_v1).
    `evidence`: a short quoted snippet / count table so the finding is
    independently re-checkable. `resolution_hint` tells a reviewer what a fix
    looks like. Affected ids keep the audit chain Script line -> claim ->
    ProductionPlan scene -> asset/artifact -> QA issue.
    """

    issue_id: str = ""
    check_id: str = ""
    category: str = ""
    severity: str = "INFO"
    gate: str = "soft"
    message: str = ""
    rationale: str = ""
    affected_scene_ids: tuple[str, ...] = ()
    affected_line_ids: tuple[str, ...] = ()
    affected_claim_ids: tuple[str, ...] = ()
    affected_asset_ids: tuple[str, ...] = ()
    policy_rule: str = ""
    evidence: str = ""
    resolution_hint: str = ""

    def validate(self) -> "PolicyIssue":
        if self.check_id not in POLICY_CHECK_ORDER:
            raise PolicyQAError(
                f"unknown check_id {self.check_id!r} (allowed: {POLICY_CHECK_ORDER})"
            )
        if not self.category:
            rule = POLICY_RULES_BY_ID.get(self.check_id)
            if rule is not None:
                object.__setattr__(self, "category", rule.category)
        if self.category not in POLICY_CATEGORIES:
            raise PolicyQAError(
                f"unknown category {self.category!r} (allowed: {POLICY_CATEGORIES})"
            )
        if self.severity not in POLICY_SEVERITIES:
            raise PolicyQAError(
                f"unknown severity {self.severity!r} (allowed: {POLICY_SEVERITIES})"
            )
        if self.gate not in POLICY_GATES:
            raise PolicyQAError(f"unknown gate {self.gate!r} (allowed: {POLICY_GATES})")
        if not self.message.strip():
            raise PolicyQAError("issue message is required")
        return self

    @property
    def is_error(self) -> bool:
        return self.severity == "ERROR"

    @property
    def is_hard_error(self) -> bool:
        return self.is_error and self.gate == "hard"

    @property
    def refs(self) -> tuple[str, ...]:
        """All referenced contract objects, matching §11's `refs`."""
        return tuple(
            dict.fromkeys(
                (
                    *self.affected_scene_ids,
                    *self.affected_line_ids,
                    *self.affected_claim_ids,
                    *self.affected_asset_ids,
                )
            )
        )

    def to_dict(self) -> dict:
        return {
            "issue_id": self.issue_id,
            "check_id": self.check_id,
            "category": self.category,
            "severity": self.severity,
            "gate": self.gate,
            "message": self.message,
            "rationale": self.rationale,
            "affected_scene_ids": list(self.affected_scene_ids),
            "affected_line_ids": list(self.affected_line_ids),
            "affected_claim_ids": list(self.affected_claim_ids),
            "affected_asset_ids": list(self.affected_asset_ids),
            "refs": list(self.refs),
            "policy_rule": self.policy_rule,
            "evidence": self.evidence,
            "resolution_hint": self.resolution_hint,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PolicyIssue":
        if not isinstance(data, dict):
            raise PolicyQAError(f"policy issue must be a mapping, got {data!r}")
        issue = cls(
            issue_id=data.get("issue_id") or "",
            check_id=data.get("check_id") or "",
            category=data.get("category") or "",
            severity=_require(data.get("severity"), "severity"),
            gate=data.get("gate") or "soft",
            message=_require(data.get("message"), "message"),
            rationale=data.get("rationale") or "",
            affected_scene_ids=tuple(data.get("affected_scene_ids") or ()),
            affected_line_ids=tuple(data.get("affected_line_ids") or ()),
            affected_claim_ids=tuple(data.get("affected_claim_ids") or ()),
            affected_asset_ids=tuple(data.get("affected_asset_ids") or ()),
            policy_rule=data.get("policy_rule") or "",
            evidence=data.get("evidence") or "",
            resolution_hint=data.get("resolution_hint") or "",
        )
        return issue.validate()


# --------------------------------------------------------------------------- #
# Check run record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PolicyCheckRecord:
    """A single policy check's outcome so reviewers can see each rule ran.

    `outcome`: PASS when the check produced no errors, REJECT when any hard
    ERROR fired, REVIEW for soft errors only. `hard_errors` counts fatal
    findings so the outcome is independently verifiable without re-walking the
    issues. `hard_gate` is the rule's registry `hard_default`.
    """

    check_id: str
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
            "check_id": self.check_id,
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
    def from_dict(cls, data: dict) -> "PolicyCheckRecord":
        if not isinstance(data, dict):
            raise PolicyQAError(f"policy check record must be a mapping, got {data!r}")
        return cls(
            check_id=_require(data.get("check_id"), "check_id"),
            category=data.get("category") or "",
            hard_gate=bool(data.get("hard_gate", False)),
            errors=int(data.get("errors") or 0),
            warnings=int(data.get("warnings") or 0),
            infos=int(data.get("infos") or 0),
            hard_errors=int(data.get("hard_errors") or 0),
            note=data.get("note") or "",
        )


# --------------------------------------------------------------------------- #
# Risk summary
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RiskSummary:
    """Human-readable rollup the Stage-9 reviewer page / CLI prints.

    `highest_risk_severity` is the worst severity observed across the issue
    stream ("none" | "INFO" | "WARNING" | "ERROR"). `flagged_categories` lists
    the categories that produced at least one ERROR so reviewers can triage
    without re-walking every issue. `reviewer_note` restates the scope and
    honesty of the ruleset.
    """

    verdict: str
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    flagged_categories: tuple[str, ...] = ()
    highest_risk_severity: str = "none"
    reviewer_note: str = POLICY_RULESET_NOTE

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "flagged_categories": list(self.flagged_categories),
            "highest_risk_severity": self.highest_risk_severity,
            "reviewer_note": self.reviewer_note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RiskSummary":
        if not isinstance(data, dict):
            raise PolicyQAError(f"risk summary must be a mapping, got {data!r}")
        return cls(
            verdict=_require(data.get("verdict"), "verdict"),
            error_count=int(data.get("error_count") or 0),
            warning_count=int(data.get("warning_count") or 0),
            info_count=int(data.get("info_count") or 0),
            flagged_categories=tuple(data.get("flagged_categories") or ()),
            highest_risk_severity=data.get("highest_risk_severity") or "none",
            reviewer_note=data.get("reviewer_note") or POLICY_RULESET_NOTE,
        )


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PolicyQAConfig:
    """Deterministic knobs for the Stage-9 policy gate.

    `policy_version` names the ruleset the evaluation must run under; a value
    outside `POLICY_VERSIONS` is caught by check P14 and emits a hard ERROR
    (REJECT) instead of a silent PASS. Config is frozen to keep run-to-run
    results reproducible.
    """

    policy_version: str = POLICY_VERSION
    repetition_token_overlap_threshold: float = 0.8

    def validate(self) -> "PolicyQAConfig":
        if not self.policy_version.strip():
            raise PolicyQAError("policy_version must not be empty")
        if not 0.0 < self.repetition_token_overlap_threshold < 1.0:
            raise PolicyQAError(
                "repetition_token_overlap_threshold must be in (0, 1)"
            )
        return self


# --------------------------------------------------------------------------- #
# Input bundle
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PresentationText:
    """One piece of presented copy with its provenance, for marker scanning.

    `source` is e.g. "script.title", "script.line", "plan.scene.narration",
    "plan.scene.visual_description", "plan.scene.on_screen_text",
    "video.artifact.note", "production_notes".
    """

    source: str
    text: str
    scene_id: str = ""
    line_id: str = ""
    purpose: str = ""
    claim_refs: tuple[str, ...] = ()
    asset_id: str = ""

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "text": self.text,
            "scene_id": self.scene_id,
            "line_id": self.line_id,
            "purpose": self.purpose,
            "claim_refs": list(self.claim_refs),
            "asset_id": self.asset_id,
        }


@dataclass(frozen=True)
class PolicyInputs:
    """Everything the Stage-9 gate evaluates over.

    Any artifact may legitimately be absent (that is itself a P13/P15-relevant
    policy signal), so all fields are optional and checks use `is None` guards.
    """

    script: Script | None = None
    fact_check: FactCheckResult | None = None
    plan: ProductionPlan | None = None
    video_result: VideoGenerationResult | None = None
    video_qa: VideoQAResult | None = None

    def presentation_items(self) -> tuple[PresentationText, ...]:
        """All presented copy + the artifact notes, each tagged with origin."""
        items: list[PresentationText] = []
        script = self.script
        if script is not None:
            if getattr(script, "title", "") or getattr(script, "hook", ""):
                items.append(PresentationText("script.title", script.title or ""))
                items.append(PresentationText("script.hook", script.hook or ""))
            for line in getattr(script, "lines", ()) or ():
                text = getattr(line, "text", "") or ""
                if text:
                    items.append(
                        PresentationText(
                            "script.line",
                            text,
                            line_id=getattr(line, "line_id", ""),
                            purpose=getattr(line, "purpose", ""),
                            claim_refs=tuple(getattr(line, "claim_refs", ()) or ()),
                        )
                    )
        plan = self.plan
        if plan is not None:
            for scene in getattr(plan, "scenes", ()) or ():
                scene_id = getattr(scene, "scene_id", "")
                narration = getattr(scene, "narration", "") or ""
                if narration:
                    items.append(PresentationText("plan.scene.narration", narration, scene_id=scene_id))
                visual = getattr(scene, "visual_description", "") or ""
                if visual:
                    items.append(PresentationText("plan.scene.visual_description", visual, scene_id=scene_id))
                prompt = getattr(scene, "generation_prompt", "") or ""
                if prompt:
                    items.append(PresentationText("plan.scene.generation_prompt", prompt, scene_id=scene_id))
                for chunk in getattr(scene, "on_screen_text", ()) or ():
                    text = getattr(chunk, "text", "") or ""
                    if text:
                        items.append(
                            PresentationText("plan.scene.on_screen_text", text, scene_id=scene_id)
                        )
            notes = getattr(plan, "production_notes", ()) or ()
            if isinstance(notes, str):
                notes = (notes,)
            notes_text = " ".join(str(note) for note in notes if str(note).strip())
            if notes_text:
                items.append(PresentationText("production_notes", notes_text))
        result = self.video_result
        if result is not None:
            for row in getattr(result, "scenes", ()) or ():
                artifact = getattr(row, "artifact", None)
                if artifact is None:
                    continue
                note = getattr(artifact, "note", "") or ""
                subject = getattr(artifact, "subject", "") or ""
                scene_id = getattr(row, "scene_id", "")
                asset_id = getattr(artifact, "artifact_id", "")
                if note:
                    items.append(
                        PresentationText(
                            "video.artifact.note", note, scene_id=scene_id, asset_id=asset_id
                        )
                    )
                if subject:
                    items.append(
                        PresentationText(
                            "video.artifact.subject", subject, scene_id=scene_id, asset_id=asset_id
                        )
                    )
        return tuple(items)


# --------------------------------------------------------------------------- #
# PolicyQAResult
# --------------------------------------------------------------------------- #


@dataclass
class PolicyQAResult:
    """The Stage-9 artifact: verdict + per-check outcomes + issues + policy version.

    Written to `data/<run_id>/production/policy_qa_result.json`. `issues` /
    `warnings` / `infos` split the same issue stream by severity. `target_ref`
    is the ProductionPlan id the policy gate was applied to. `policy_version`
    records the exact ruleset used; the trace chain Script -> claim -> scene ->
    asset/artifact -> QA issue is preserved through `checks` and every issue's
    affected ids.
    """

    policy_result_id: str
    target_ref: str
    topic_id: str
    topic_title: str
    script_id: str
    policy_version: str
    overall_verdict: str
    checked_at: str
    checks: tuple[PolicyCheckRecord, ...] = ()
    issues: tuple[PolicyIssue, ...] = ()
    warnings: tuple[PolicyIssue, ...] = ()
    infos: tuple[PolicyIssue, ...] = ()
    risk_summary: RiskSummary | None = None
    classifier_suggestion: str = ""
    resumed_from_policy_result_id: str = ""
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

    def all_issues(self) -> tuple[PolicyIssue, ...]:
        return self.issues + self.warnings + self.infos

    def issue_for(self, check_id: str) -> PolicyIssue | None:
        return next((issue for issue in self.all_issues() if issue.check_id == check_id), None)

    def validate(self) -> "PolicyQAResult":
        _require(self.policy_result_id, "policy_result_id")
        _require(self.target_ref, "target_ref")
        _require(self.policy_version, "policy_version")
        if self.overall_verdict not in POLICY_VERDICTS:
            raise PolicyQAError(
                f"overall_verdict must be one of {POLICY_VERDICTS}, got {self.overall_verdict!r}"
            )
        if self.provenance is None:
            raise PolicyQAError("PolicyQAResult provenance is required")
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
                "policy_result_id": self.policy_result_id,
                "target_ref": self.target_ref,
                "topic_id": self.topic_id,
                "topic_title": self.topic_title,
                "script_id": self.script_id,
                "policy_version": self.policy_version,
                "overall_verdict": self.overall_verdict,
                "checked_at": self.checked_at,
                "checks": [record.to_dict() for record in self.checks],
                "issues": [issue.to_dict() for issue in self.issues],
                "warnings": [issue.to_dict() for issue in self.warnings],
                "infos": [issue.to_dict() for issue in self.infos],
                "risk_summary": self.risk_summary.to_dict() if self.risk_summary else None,
                "classifier_suggestion": self.classifier_suggestion,
                "resumed_from_policy_result_id": self.resumed_from_policy_result_id,
                "provenance": self.provenance.to_dict() if self.provenance else None,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "PolicyQAResult":
        header = dict(data)
        payload = header.pop("data")
        result = cls(
            policy_result_id=_require(payload.get("policy_result_id"), "policy_result_id"),
            target_ref=_require(payload.get("target_ref"), "target_ref"),
            topic_id=payload.get("topic_id") or "",
            topic_title=payload.get("topic_title") or "",
            script_id=payload.get("script_id") or "",
            policy_version=_require(payload.get("policy_version"), "policy_version"),
            overall_verdict=_require(payload.get("overall_verdict"), "overall_verdict"),
            checked_at=_require(payload.get("checked_at"), "checked_at"),
            checks=tuple(
                PolicyCheckRecord.from_dict(item) for item in payload.get("checks") or []
            ),
            issues=tuple(PolicyIssue.from_dict(item) for item in payload.get("issues") or []),
            warnings=tuple(
                PolicyIssue.from_dict(item) for item in payload.get("warnings") or []
            ),
            infos=tuple(PolicyIssue.from_dict(item) for item in payload.get("infos") or []),
            risk_summary=RiskSummary.from_dict(payload["risk_summary"])
            if payload.get("risk_summary")
            else None,
            classifier_suggestion=payload.get("classifier_suggestion") or "",
            resumed_from_policy_result_id=payload.get("resumed_from_policy_result_id") or "",
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
    def from_json(cls, text: str) -> "PolicyQAResult":
        return cls.from_dict(json.loads(text))