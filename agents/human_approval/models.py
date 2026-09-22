"""Versioned Stage-10 (human approval) models.

Stage 10 is the HUMAN APPROVAL GATE on top of the fully QA'd artifact chain:
the Stage-3 Script, the Stage-5 FactCheckResult, the Stage-5 ScriptQAResult,
the Stage-6 ProductionPlan, the Stage-7 VideoGenerationResult, the Stage-8
VideoQAResult, and the Stage-9 PolicyQAResult
(docs/PROJECT_PROGRESS.md Stage 10 / docs/POLICY_AND_SAFETY.md / docs/
DATA_CONTRACTS.md). It records an explicit HUMAN decision over the EXACT
artifact versions that were reviewed (each reviewed artifact is bound to its
stable id + a canonical-content SHA-256 digest), so approval can never drift
from what was actually looked at.

Decision model is deliberately NOT the QA verdict tuple (PASS/REVIEW/REJECT):
a human decides APPROVED, CHANGES_REQUESTED, or REJECTED, with PENDING meaning
"no human action yet". A PENDING record is never written by the CLI -- no
written record means there is no approval (fail closed). Derived states
(CURRENT / STALE / SUPERSEDED) describe whether an existing APPROVED record
is still trustworthy: digests that no longer match the current artifacts make
it STALE, and a later decision supersedes it via `supersedes_approval_id`.

Hard gates: a Stage-8 or Stage-9 hard REJECT, missing/malformed upstream
artifacts, an uncompleted video, mismatched digests, or an unresolved soft
REVIEW (without a recorded, version-bound override) block APPROVED. Soft
REVIEW findings may be overridden by a human, but the override must be
explicitly recorded and bound to the exact reviewed artifact version -- never
a silent REVIEW -> PASS.

Provenance: the human is NOT an AI model/provider. `produced_by` records
provider=None / model=None and provenance.generated_by is
"human_approval_gate_v1" with source "HumanReviewer".

Stage label vs. pipeline state: the artifact header `stage` is
`"human_approval"` (per the Stage-10 task). The numbered Stage 10 advances
pipeline STATE to the existing generic `Stage.PUBLISH` (pipeline/state.py) as
a bookkeeping marker that the run is approved for publishing; no new enum
member is added or renamed. `STATE_STAGE` records that mapping explicitly.
This stage never uploads, schedules, or touches YouTube -- Stage 11 is not
implemented anywhere in this package.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agents.qa.fact_checker.models import FactCheckResult
from agents.qa.policy.models import PolicyQAResult
from agents.qa.script.models import ScriptQAResult
from agents.qa.video.models import VideoQAResult
from agents.scene_planner.models import ProductionPlan
from agents.script.models import Script
from agents.seo.models import Provenance
from agents.video_generation.models import VideoGenerationResult
from pipeline.state import Stage

SCHEMA = "HumanApprovalResult"
SCHEMA_VERSION = 1

#: Artifact-header `stage` label (per the Stage-10 task; documented in
#: DATA_CONTRACTS.md and PROJECT_PROGRESS.md Stage 10). Not a
#: pipeline/state.py Stage member -- that mapping is `STATE_STAGE`.
STAGE = "human_approval"

#: Numbered Stage 10 (Human Approval) advances PipelineState to the EXISTING
#: generic publish stage as a bookkeeping marker that the run is human-approved
#: for the manual-upload endpoint. No new enum member was added or renamed.
STATE_STAGE = Stage.PUBLISH

#: Version of the Stage-10 process (recorded in provenance and produced_by).
HUMAN_APPROVAL_CONFIG_VERSION = "human_approval.v1"

#: Human decision vocabulary (deliberately distinct from QA verdicts: a human
#: approves or requests changes; they never "PASS" or "REVIEW" a video).
DECISIONS = ("PENDING", "APPROVED", "CHANGES_REQUESTED", "REJECTED")

#: Decisions the CLI can actually write. PENDING is only the neutral "no human
#: action yet" state; the CLI refuses to fabricate a PENDING approval record.
WRITABLE_DECISIONS = ("APPROVED", "CHANGES_REQUESTED", "REJECTED")

#: Derived states for an existing approval record (not decisions).
DERIVED_STATES = ("CURRENT", "STALE", "SUPERSEDED")

#: Roles reviewed by the gate, in stable record order. `production_plan`
#: maps to the ApprovalInputs field `plan` (see ROLE_ATTR).
ROLES = (
    "policy_qa",
    "video_qa",
    "video_result",
    "production_plan",
    "script",
    "script_qa",
    "fact_check",
)

#: Roles that must be present and well-formed before ANY human decision (human
#: or machine) can be recorded. Missing required artifacts are a fail-closed
#: error, not something a human can approve around.
REQUIRED_ROLES = (
    "policy_qa",
    "video_qa",
    "video_result",
    "production_plan",
    "script",
)

#: ApprovalInputs attribute for each role.
ROLE_ATTR = {
    "policy_qa": "policy_qa",
    "video_qa": "video_qa",
    "video_result": "video_result",
    "production_plan": "plan",
    "script": "script",
    "script_qa": "script_qa",
    "fact_check": "fact_check",
}

#: Stable-id field on each upstream artifact (used for binding + freshness).
ROLE_ID_FIELD = {
    "policy_qa": "policy_result_id",
    "video_qa": "qa_result_id",
    "video_result": "video_generation_result_id",
    "production_plan": "production_plan_id",
    "script": "script_id",
    "script_qa": "qa_result_id",
    "fact_check": "result_id",
}

#: Schema label recorded for each reviewed artifact.
ROLE_SCHEMA = {
    "policy_qa": "PolicyQAResult",
    "video_qa": "VideoQAResult",
    "video_result": "VideoGenerationResult",
    "production_plan": "ProductionPlan",
    "script": "Script",
    "script_qa": "ScriptQAResult",
    "fact_check": "FactCheckResult",
}

#: The `produced_by` envelope helper. A human is NOT an AI model/provider, so
#: provider and model are explicitly null.
PRODUCED_BY = {
    "prompt_version": HUMAN_APPROVAL_CONFIG_VERSION,
    "provider": None,
    "model": None,
}

#: Provenance of the approval record itself.
PROVENANCE_GENERATED_BY = "human_approval_gate_v1"
PROVENANCE_SOURCE = "HumanReviewer"

#: Manual-upload readiness note (the pipeline ends at a manual upload, not at
#: an automated YouTube publish).
MANUAL_UPLOAD_NOTE = (
    "No automated upload is implemented. Endpoint is a manual upload only."
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class HumanApprovalError(ValueError):
    """Raised when a Stage-10 artifact is missing, malformed, or inconsistent."""


def _require(value: Any, name: str) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise HumanApprovalError(f"missing required field: {name}")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def human_approval_id_for(run_id: str, target_ref: str) -> str:
    """Stable HumanApprovalResult id per (run_id, target_ref); deterministic.

    Content-independent on purpose: content changes are detected through the
    reviewed-artifact digests, so the id never silently changes underneath an
    audit trail.
    """
    digest = hashlib.sha1(
        f"{run_id or ''}:{target_ref or ''}".encode("utf-8")
    ).hexdigest()[:12]
    return f"ha_{digest}"


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical JSON serialization for digesting (indent/order independent)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def content_sha256(value: Any) -> str:
    """Canonical-content SHA-256 of a serializable artifact dict payload."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


# --------------------------------------------------------------------------- #
# Reviewed artifact binding
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReviewedArtifact:
    """One upstream artifact as it was EXACTLY reviewed.

    `role` is the stable role id (ROLES), `artifact_id` is the artifact's own
    stable content-independent id (e.g. `pr_...`, `qa_...`), and
    `content_sha256` is the canonical-content digest of the reviewed version.
    Binding both id AND digest means any upstream change after review makes the
    approval STALE instead of silently approving a different version.
    """

    role: str
    artifact_id: str
    content_sha256: str
    schema: str = ""

    def validate(self) -> "ReviewedArtifact":
        if self.role not in ROLES:
            raise HumanApprovalError(
                f"unknown reviewed-artifact role {self.role!r} (allowed: {ROLES})"
            )
        object.__setattr__(self, "artifact_id", _require(self.artifact_id, "artifact_id"))
        digest = self.content_sha256 or ""
        if not _SHA256_RE.fullmatch(digest):
            raise HumanApprovalError(
                f"content_sha256 must be a 64-char hex sha256, got {digest!r}"
            )
        if self.schema and self.schema != ROLE_SCHEMA.get(self.role):
            raise HumanApprovalError(
                f"schema {self.schema!r} does not match role {self.role!r}"
            )
        return self

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "artifact_id": self.artifact_id,
            "content_sha256": self.content_sha256,
            "schema": self.schema or ROLE_SCHEMA.get(self.role, ""),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewedArtifact":
        if not isinstance(data, dict):
            raise HumanApprovalError(f"reviewed artifact must be a mapping, got {data!r}")
        role = _require(data.get("role"), "role")
        return cls(
            role=role,
            artifact_id=_require(data.get("artifact_id"), "artifact_id"),
            content_sha256=_require(data.get("content_sha256"), "content_sha256"),
            schema=data.get("schema") or ROLE_SCHEMA.get(role, ""),
        ).validate()


# --------------------------------------------------------------------------- #
# Approving human
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Approver:
    """The human making the decision.

    `approver_id` is required (free-form: reviewer handle, initials, ticket).
    `role` is optional context. The human is never modeled as an AI provider.
    """

    approver_id: str
    role: str = ""

    def validate(self) -> "Approver":
        object.__setattr__(self, "approver_id", _require(self.approver_id, "approver_id"))
        return self

    def to_dict(self) -> dict:
        return {"approver_id": self.approver_id, "role": self.role}

    @classmethod
    def from_dict(cls, data: dict) -> "Approver":
        if not isinstance(data, dict):
            raise HumanApprovalError(f"approver must be a mapping, got {data!r}")
        return cls(
            approver_id=_require(data.get("approver_id"), "approver_id"),
            role=data.get("role") or "",
        ).validate()


# --------------------------------------------------------------------------- #
# Overridden soft-REVIEW verdicts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OverriddenVerdict:
    """An explicit human resolution of a soft-REVIEW finding.

    Soft REVIEW findings (e.g. policy check P15 / P13, Stage-8 REVIEW) may be
    resolved by a human, but the resolution MUST be recorded: which gate/check
    was overridden, the verdict that was overridden, a justification, the human
    decision, and the exact reviewed artifact version (artifact_id). Without
    this record a REVIEW can never become an approval.
    """

    gate: str
    check_id: str = ""
    verdict: str = "REVIEW"
    decision: str = "APPROVED"
    justification: str = ""
    artifact_id: str = ""

    def validate(self) -> "OverriddenVerdict":
        if self.gate not in ("policy_qa", "video_qa"):
            raise HumanApprovalError(f"override gate must be policy_qa or video_qa, got {self.gate!r}")
        if self.verdict not in ("PASS", "REVIEW", "REJECT"):
            raise HumanApprovalError(f"overridden verdict must be PASS/REVIEW/REJECT, got {self.verdict!r}")
        if self.decision not in WRITABLE_DECISIONS:
            raise HumanApprovalError(f"override decision must be writable, got {self.decision!r}")
        object.__setattr__(
            self, "justification", _require(self.justification, "justification")
        )
        return self

    def to_dict(self) -> dict:
        return {
            "gate": self.gate,
            "check_id": self.check_id,
            "verdict": self.verdict,
            "decision": self.decision,
            "justification": self.justification,
            "artifact_id": self.artifact_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OverriddenVerdict":
        if not isinstance(data, dict):
            raise HumanApprovalError(f"override must be a mapping, got {data!r}")
        return cls(
            gate=_require(data.get("gate"), "gate"),
            check_id=data.get("check_id") or "",
            verdict=data.get("verdict") or "REVIEW",
            decision=data.get("decision") or "APPROVED",
            justification=_require(data.get("justification"), "justification"),
            artifact_id=data.get("artifact_id") or "",
        ).validate()


# --------------------------------------------------------------------------- #
# Blocking conditions + requested changes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BlockingCondition:
    """One deterministically-evaluated hard-gate condition for APPROVED.

    `satisfied=False` blocks APPROVED (the record is still written for
    CHANGES_REQUESTED / REJECTED so the audit shows exactly what failed).
    """

    code: str
    satisfied: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return {"code": self.code, "satisfied": self.satisfied, "detail": self.detail}

    @classmethod
    def from_dict(cls, data: dict) -> "BlockingCondition":
        if not isinstance(data, dict):
            raise HumanApprovalError(f"blocking condition must be a mapping, got {data!r}")
        return cls(
            code=_require(data.get("code"), "code"),
            satisfied=bool(data.get("satisfied", False)),
            detail=data.get("detail") or "",
        )


@dataclass(frozen=True)
class RequestedChange:
    """One concrete change the human requested before (re)approval.

    `requirement` is required; `issue_ref` optionally names the upstream
    issue/check it addresses (e.g. "P06", "P13", a scene id).
    """

    issue_ref: str = ""
    requirement: str = ""

    def validate(self) -> "RequestedChange":
        object.__setattr__(
            self, "requirement", _require(self.requirement, "requirement")
        )
        return self

    def to_dict(self) -> dict:
        return {"issue_ref": self.issue_ref, "requirement": self.requirement}

    @classmethod
    def from_dict(cls, data: dict) -> "RequestedChange":
        if not isinstance(data, dict):
            raise HumanApprovalError(f"requested change must be a mapping, got {data!r}")
        return cls(
            issue_ref=data.get("issue_ref") or "",
            requirement=_require(data.get("requirement"), "requirement"),
        ).validate()


# --------------------------------------------------------------------------- #
# Manual-upload readiness
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ManualUploadReadiness:
    """Whether the reviewed bundle is ready for a MANUAL upload.

    `ready` is true only for an APPROVED decision whose gates all cleared.
    `artifacts` lists the exact reviewed artifact ids that make up the ready
    package. There is never an automated YouTube publish.
    """

    ready: bool
    required: bool = True
    note: str = MANUAL_UPLOAD_NOTE
    artifacts: tuple[str, ...] = ()
    title: str = ""

    def to_dict(self) -> dict:
        return {
            "ready": self.ready,
            "required": self.required,
            "note": self.note,
            "artifacts": list(self.artifacts),
            "title": self.title,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ManualUploadReadiness":
        if not isinstance(data, dict):
            raise HumanApprovalError(
                f"manual-upload readiness must be a mapping, got {data!r}"
            )
        return cls(
            ready=bool(data.get("ready", False)),
            required=bool(data.get("required", True)),
            note=data.get("note") or MANUAL_UPLOAD_NOTE,
            artifacts=tuple(data.get("artifacts") or ()),
            title=data.get("title") or "",
        )


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HumanApprovalConfig:
    """Deterministic knobs for the Stage-10 gate. Frozen for reproducibility."""

    schema_version: str = HUMAN_APPROVAL_CONFIG_VERSION
    required_roles: tuple[str, ...] = REQUIRED_ROLES
    require_override_for_review: bool = True

    def validate(self) -> "HumanApprovalConfig":
        if not self.schema_version.strip():
            raise HumanApprovalError("schema_version must not be empty")
        if not self.required_roles:
            raise HumanApprovalError("required_roles must not be empty")
        for role in self.required_roles:
            if role not in ROLES:
                raise HumanApprovalError(
                    f"required role {role!r} not in ROLES (allowed: {ROLES})"
                )
        return self


# --------------------------------------------------------------------------- #
# Input bundle
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ApprovalInputs:
    """Everything the Stage-10 gate reviews.

    `policy_qa`, `video_qa`, `video_result`, `plan`, and `script` are required
    for ANY decision (fail closed). `script_qa` and `fact_check` are optional
    and, when present, are bound into the reviewed set.
    """

    policy_qa: PolicyQAResult | None = None
    video_qa: VideoQAResult | None = None
    video_result: VideoGenerationResult | None = None
    plan: ProductionPlan | None = None
    script: Script | None = None
    script_qa: ScriptQAResult | None = None
    fact_check: FactCheckResult | None = None


# --------------------------------------------------------------------------- #
# HumanApprovalResult
# --------------------------------------------------------------------------- #


@dataclass
class HumanApprovalResult:
    """The Stage-10 artifact: human decision + exact reviewed artifact bindings.

    Written to `data/<run_id>/approval/human_approval_result.json`. The header
    (schema/schema_version/created_at/run_id/stage/produced_by) follows the
    repo envelope; `data` holds the decision, the approving human, the reviewed
    artifacts (id + content digest), the prior gate verdicts, any recorded
    overrides, the blocking-condition evaluation, manual-upload readiness,
    notes, and any requested changes. `target_ref` is the ProductionPlan id
    that was reviewed.
    """

    approval_id: str
    target_ref: str
    decision: str
    decision_at: str
    approver: Approver
    reviewed_artifacts: tuple[ReviewedArtifact, ...] = ()
    blocking_conditions: tuple[BlockingCondition, ...] = ()
    prior_gate_verdicts: dict = field(default_factory=dict)
    overridden_verdicts: tuple[OverriddenVerdict, ...] = ()
    manual_upload: ManualUploadReadiness | None = None
    notes: str = ""
    changes_requested: tuple[RequestedChange, ...] = ()
    supersedes_approval_id: str = ""
    provenance: Provenance | None = None
    created_at: str = field(default_factory=_utc_now)
    schema: str = SCHEMA
    schema_version: int = SCHEMA_VERSION
    stage: str = STAGE
    run_id: str = ""
    produced_by: dict = field(default_factory=lambda: dict(PRODUCED_BY))

    @property
    def is_approved(self) -> bool:
        return self.decision == "APPROVED"

    @property
    def gates_cleared(self) -> bool:
        return all(condition.satisfied for condition in self.blocking_conditions)

    def validate(self) -> "HumanApprovalResult":
        _require(self.approval_id, "approval_id")
        _require(self.target_ref, "target_ref")
        _require(self.decision_at, "decision_at")
        if self.decision not in DECISIONS:
            raise HumanApprovalError(
                f"decision must be one of {DECISIONS}, got {self.decision!r}"
            )
        if not self.reviewed_artifacts:
            raise HumanApprovalError("reviewed_artifacts must not be empty")
        roles = [artifact.role for artifact in self.reviewed_artifacts]
        if len(set(roles)) != len(roles):
            raise HumanApprovalError("reviewed_artifacts roles must be unique")
        self.approver.validate()
        for artifact in self.reviewed_artifacts:
            artifact.validate()
        for condition in self.blocking_conditions:
            condition.code
        for override in self.overridden_verdicts:
            override.validate()
        if self.decision == "APPROVED" and not self.gates_cleared:
            failed = [
                f"{c.code} ({c.detail})" for c in self.blocking_conditions if not c.satisfied
            ]
            raise HumanApprovalError(
                "cannot APPROVE with unsatisfied blocking conditions: " + "; ".join(failed)
            )
        if self.decision == "CHANGES_REQUESTED":
            if not self.changes_requested and not self.notes.strip():
                raise HumanApprovalError(
                    "CHANGES_REQUESTED needs at least one requested change or a note"
                )
        if self.provenance is None:
            raise HumanApprovalError("HumanApprovalResult provenance is required")
        if self.manual_upload is None:
            raise HumanApprovalError("manual-upload readiness is required")
        if self.supersedes_approval_id and self.supersedes_approval_id == self.approval_id:
            raise HumanApprovalError("an approval cannot supersede itself")
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
                "approval_id": self.approval_id,
                "target_ref": self.target_ref,
                "decision": self.decision,
                "decision_at": self.decision_at,
                "approver": self.approver.to_dict(),
                "reviewed_artifacts": [
                    artifact.to_dict() for artifact in self.reviewed_artifacts
                ],
                "blocking_conditions": [
                    condition.to_dict() for condition in self.blocking_conditions
                ],
                "prior_gate_verdicts": dict(self.prior_gate_verdicts),
                "overridden_verdicts": [
                    override.to_dict() for override in self.overridden_verdicts
                ],
                "manual_upload": self.manual_upload.to_dict() if self.manual_upload else None,
                "notes": self.notes,
                "changes_requested": [
                    change.to_dict() for change in self.changes_requested
                ],
                "supersedes_approval_id": self.supersedes_approval_id,
                "provenance": self.provenance.to_dict() if self.provenance else None,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "HumanApprovalResult":
        header = dict(data)
        payload = header.pop("data")
        result = cls(
            approval_id=_require(payload.get("approval_id"), "approval_id"),
            target_ref=_require(payload.get("target_ref"), "target_ref"),
            decision=_require(payload.get("decision"), "decision"),
            decision_at=_require(payload.get("decision_at"), "decision_at"),
            approver=Approver.from_dict(payload["approver"]),
            reviewed_artifacts=tuple(
                ReviewedArtifact.from_dict(item)
                for item in payload.get("reviewed_artifacts") or []
            ),
            blocking_conditions=tuple(
                BlockingCondition.from_dict(item)
                for item in payload.get("blocking_conditions") or []
            ),
            prior_gate_verdicts=dict(payload.get("prior_gate_verdicts") or {}),
            overridden_verdicts=tuple(
                OverriddenVerdict.from_dict(item)
                for item in payload.get("overridden_verdicts") or []
            ),
            manual_upload=ManualUploadReadiness.from_dict(payload["manual_upload"])
            if payload.get("manual_upload")
            else None,
            notes=payload.get("notes") or "",
            changes_requested=tuple(
                RequestedChange.from_dict(item)
                for item in payload.get("changes_requested") or []
            ),
            supersedes_approval_id=payload.get("supersedes_approval_id") or "",
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
    def from_json(cls, text: str) -> "HumanApprovalResult":
        return cls.from_dict(json.loads(text))