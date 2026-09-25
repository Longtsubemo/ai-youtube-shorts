"""Versioned Stage-10 (human approval) models.

Stage 10 is the HUMAN APPROVAL GATE on top of the fully QA'd artifact chain:
the Stage-4 Script, the Stage-5 FactCheckResult, the Stage-5 ScriptQAResult,
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
pipeline bookkeeping to the existing generic `Stage.PUBLISH` for an approved
decision; non-approved decisions remain at `Stage.QUALITY_CHECK`. No new enum
member is added or renamed. `STATE_STAGE` records the approved-state mapping
explicitly. This stage never uploads, schedules, or touches YouTube -- Stage
11 is not implemented anywhere in this package.
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
#: pipeline/state.py Stage member -- approved decisions map to `STATE_STAGE`,
#: while non-approved decisions remain at `Stage.QUALITY_CHECK`.
STAGE = "human_approval"

#: Numbered Stage 10 (Human Approval) advances PipelineState to the EXISTING
#: generic publish stage for an approved decision as a bookkeeping marker. No
#: new enum member was added or renamed.
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
BLOCKING_CONDITION_CODES = (
    "required_artifacts",
    "policy_qa_hard_reject",
    "video_qa_hard_reject",
    "video_generation_completed",
    "review_overrides",
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
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_APPROVAL_ID_RE = re.compile(r"^ha_[0-9a-f]{12}$")


class HumanApprovalError(ValueError):
    """Raised when a Stage-10 artifact is missing, malformed, or inconsistent."""


def _require(value: Any, name: str) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise HumanApprovalError(f"missing required field: {name}")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_run_id(value: Any) -> str:
    """Validate an opaque run identifier before it is used in a path."""
    if not isinstance(value, str) or not _RUN_ID_RE.fullmatch(value):
        raise HumanApprovalError(
            "run_id must be 1-128 characters using letters, digits, '.', '_' "
            f"or '-', and must start with a letter or digit; got {value!r}"
        )
    if ".." in value:
        raise HumanApprovalError(f"run_id contains a path traversal sequence: {value!r}")
    return value


def human_approval_id_for(run_id: str, target_ref: str) -> str:
    """Stable HumanApprovalResult id per (run_id, target_ref); deterministic.

    Content-independent on purpose: content changes are detected through the
    reviewed-artifact digests, so the id never silently changes underneath an
    audit trail.
    """
    run_id = validate_run_id(run_id)
    if not isinstance(target_ref, str) or not target_ref.strip():
        raise HumanApprovalError(f"target_ref must be non-empty text, got {target_ref!r}")
    digest = hashlib.sha1(f"{run_id}:{target_ref}".encode("utf-8")).hexdigest()[:12]
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
        if not isinstance(self.role, str) or self.role not in ROLES:
            raise HumanApprovalError(
                f"unknown reviewed-artifact role {self.role!r} (allowed: {ROLES})"
            )
        if not isinstance(self.artifact_id, str):
            raise HumanApprovalError("artifact_id must be text")
        if not isinstance(self.content_sha256, str):
            raise HumanApprovalError("content_sha256 must be text")
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
        if not isinstance(self.approver_id, str):
            raise HumanApprovalError("approver_id must be text")
        if not isinstance(self.role, str):
            raise HumanApprovalError("approver role must be text")
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
        if not isinstance(self.gate, str) or self.gate not in ("policy_qa", "video_qa"):
            raise HumanApprovalError(
                f"override gate must be policy_qa or video_qa, got {self.gate!r}"
            )
        if not isinstance(self.check_id, str):
            raise HumanApprovalError(f"override check_id must be text, got {self.check_id!r}")
        if not isinstance(self.verdict, str) or self.verdict not in ("PASS", "REVIEW", "REJECT"):
            raise HumanApprovalError(
                f"overridden verdict must be PASS/REVIEW/REJECT, got {self.verdict!r}"
            )
        if not isinstance(self.decision, str) or self.decision not in WRITABLE_DECISIONS:
            raise HumanApprovalError(
                f"override decision must be writable, got {self.decision!r}"
            )
        if not isinstance(self.artifact_id, str):
            raise HumanApprovalError(
                f"override artifact_id must be text, got {self.artifact_id!r}"
            )
        if not isinstance(self.justification, str):
            raise HumanApprovalError("override justification must be text")
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
        satisfied = data.get("satisfied")
        if not isinstance(satisfied, bool):
            raise HumanApprovalError(
                f"blocking condition satisfied must be a boolean, got {satisfied!r}"
            )
        detail = data.get("detail", "")
        if not isinstance(detail, str):
            raise HumanApprovalError(f"blocking condition detail must be text, got {detail!r}")
        return cls(
            code=_require(data.get("code"), "code"),
            satisfied=satisfied,
            detail=detail,
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
        if not isinstance(self.issue_ref, str):
            raise HumanApprovalError("requested-change issue_ref must be text")
        if not isinstance(self.requirement, str):
            raise HumanApprovalError("requested-change requirement must be text")
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

    def validate(self) -> "ManualUploadReadiness":
        if not isinstance(self.ready, bool) or not isinstance(self.required, bool):
            raise HumanApprovalError("manual-upload readiness flags must be booleans")
        if not isinstance(self.note, str) or not isinstance(self.title, str):
            raise HumanApprovalError("manual-upload note/title must be text")
        if not isinstance(self.artifacts, tuple) or not all(
            isinstance(item, str) and item for item in self.artifacts
        ):
            raise HumanApprovalError("manual-upload artifacts must be a tuple of ids")
        if len(set(self.artifacts)) != len(self.artifacts):
            raise HumanApprovalError("manual-upload artifacts must be unique")
        return self

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
        ready = data.get("ready")
        required = data.get("required", True)
        if not isinstance(ready, bool) or not isinstance(required, bool):
            raise HumanApprovalError("manual-upload readiness flags must be booleans")
        artifacts = data.get("artifacts", [])
        if not isinstance(artifacts, list) or not all(
            isinstance(item, str) and item for item in artifacts
        ):
            raise HumanApprovalError("manual-upload artifacts must be a list of ids")
        note = data.get("note", MANUAL_UPLOAD_NOTE)
        title = data.get("title", "")
        if not isinstance(note, str) or not isinstance(title, str):
            raise HumanApprovalError("manual-upload note/title must be text")
        return cls(
            ready=ready,
            required=required,
            note=note or MANUAL_UPLOAD_NOTE,
            artifacts=tuple(artifacts),
            title=title,
        ).validate()


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
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise HumanApprovalError("schema_version must not be empty")
        if not isinstance(self.required_roles, tuple) or not self.required_roles:
            raise HumanApprovalError("required_roles must be a non-empty tuple")
        if not all(isinstance(role, str) and role in ROLES for role in self.required_roles):
            raise HumanApprovalError(
                "required_roles must contain only known role names "
                f"(allowed: {ROLES})"
            )
        if len(set(self.required_roles)) != len(self.required_roles):
            raise HumanApprovalError("required_roles must not contain duplicates")
        if not isinstance(self.require_override_for_review, bool):
            raise HumanApprovalError("require_override_for_review must be a boolean")
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
        if not isinstance(self.approval_id, str) or not _APPROVAL_ID_RE.fullmatch(self.approval_id):
            raise HumanApprovalError(f"invalid approval_id: {self.approval_id!r}")
        if not isinstance(self.target_ref, str):
            raise HumanApprovalError("target_ref must be text")
        _require(self.target_ref, "target_ref")
        if not isinstance(self.decision_at, str):
            raise HumanApprovalError("decision_at must be text")
        _require(self.decision_at, "decision_at")
        validate_run_id(self.run_id)
        if self.schema != SCHEMA:
            raise HumanApprovalError(f"schema must be {SCHEMA!r}, got {self.schema!r}")
        if isinstance(self.schema_version, bool) or self.schema_version != SCHEMA_VERSION:
            raise HumanApprovalError(
                f"schema_version must be {SCHEMA_VERSION}, got {self.schema_version!r}"
            )
        if self.stage != STAGE:
            raise HumanApprovalError(f"stage must be {STAGE!r}, got {self.stage!r}")
        if not isinstance(self.produced_by, dict):
            raise HumanApprovalError("produced_by must be a mapping")
        if self.produced_by.get("provider") is not None or self.produced_by.get("model") is not None:
            raise HumanApprovalError("human approval produced_by provider/model must be null")
        if self.produced_by.get("prompt_version") != HUMAN_APPROVAL_CONFIG_VERSION:
            raise HumanApprovalError("human approval produced_by prompt_version is invalid")
        if not isinstance(self.decision, str) or self.decision not in DECISIONS:
            raise HumanApprovalError(
                f"decision must be one of {DECISIONS}, got {self.decision!r}"
            )
        if not isinstance(self.notes, str):
            raise HumanApprovalError("notes must be text")
        if not isinstance(self.prior_gate_verdicts, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in self.prior_gate_verdicts.items()
        ):
            raise HumanApprovalError("prior_gate_verdicts must map text to text")
        if set(self.prior_gate_verdicts) != {
            "policy_qa",
            "video_qa",
            "video_generation",
            "script_qa",
        }:
            raise HumanApprovalError("prior_gate_verdicts has an unexpected key set")
        if not isinstance(self.reviewed_artifacts, tuple):
            raise HumanApprovalError("reviewed_artifacts must be a tuple")
        if not self.reviewed_artifacts:
            raise HumanApprovalError("reviewed_artifacts must not be empty")
        if not all(isinstance(artifact, ReviewedArtifact) for artifact in self.reviewed_artifacts):
            raise HumanApprovalError("reviewed_artifacts contains an invalid artifact")
        if not all(isinstance(artifact.role, str) for artifact in self.reviewed_artifacts):
            raise HumanApprovalError("reviewed artifact roles must be text")
        roles = [artifact.role for artifact in self.reviewed_artifacts]
        if len(set(roles)) != len(roles):
            raise HumanApprovalError("reviewed_artifacts roles must be unique")
        expected_order = [role for role in ROLES if role in roles]
        if roles != expected_order:
            raise HumanApprovalError("reviewed_artifacts must follow the stable role order")
        missing_roles = set(REQUIRED_ROLES) - set(roles)
        if missing_roles:
            raise HumanApprovalError(
                "reviewed_artifacts missing required roles: " + ", ".join(sorted(missing_roles))
            )
        for artifact in self.reviewed_artifacts:
            artifact.validate()
        if not isinstance(self.blocking_conditions, tuple):
            raise HumanApprovalError("blocking_conditions must be a tuple")
        if not isinstance(self.approver, Approver):
            raise HumanApprovalError("approver must be an Approver")
        self.approver.validate()
        for condition in self.blocking_conditions:
            if not isinstance(condition, BlockingCondition):
                raise HumanApprovalError("blocking_conditions contains an invalid condition")
            if not isinstance(condition.code, str):
                raise HumanApprovalError("blocking condition code must be text")
            _require(condition.code, "blocking condition code")
            if not isinstance(condition.satisfied, bool):
                raise HumanApprovalError("blocking condition satisfied must be boolean")
            if not isinstance(condition.detail, str):
                raise HumanApprovalError("blocking condition detail must be text")
        condition_codes = [condition.code for condition in self.blocking_conditions]
        if len(set(condition_codes)) != len(condition_codes):
            raise HumanApprovalError("blocking condition codes must be unique")
        if set(condition_codes) != set(BLOCKING_CONDITION_CODES):
            raise HumanApprovalError("blocking conditions must contain the complete gate set")
        if not isinstance(self.overridden_verdicts, tuple) or not all(
            isinstance(override, OverriddenVerdict) for override in self.overridden_verdicts
        ):
            raise HumanApprovalError("overridden_verdicts must contain valid overrides")
        for override in self.overridden_verdicts:
            override.validate()
            if not override.check_id or not override.artifact_id:
                raise HumanApprovalError(
                    "recorded overrides must include check_id and artifact_id"
                )
        if not isinstance(self.changes_requested, tuple) or not all(
            isinstance(change, RequestedChange) for change in self.changes_requested
        ):
            raise HumanApprovalError("changes_requested must contain valid changes")
        for change in self.changes_requested:
            change.validate()
        if self.decision == "APPROVED" and not self.blocking_conditions:
            raise HumanApprovalError("APPROVED requires evaluated blocking conditions")
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
        if not isinstance(self.provenance, Provenance):
            raise HumanApprovalError("HumanApprovalResult provenance is required")
        if (
            self.provenance.generated_by != PROVENANCE_GENERATED_BY
            or self.provenance.source != PROVENANCE_SOURCE
            or self.provenance.provider is not None
            or self.provenance.model is not None
            or self.provenance.prompt_version != HUMAN_APPROVAL_CONFIG_VERSION
        ):
            raise HumanApprovalError("HumanApprovalResult provenance is not human-review provenance")
        if not isinstance(self.manual_upload, ManualUploadReadiness):
            raise HumanApprovalError("manual-upload readiness must be a ManualUploadReadiness")
        self.manual_upload.validate()
        if not self.manual_upload.required:
            raise HumanApprovalError("manual_upload.required must be true")
        expected_artifact_ids = tuple(artifact.artifact_id for artifact in self.reviewed_artifacts)
        if self.manual_upload.artifacts != expected_artifact_ids:
            raise HumanApprovalError("manual_upload.artifacts must match reviewed_artifacts")
        if self.manual_upload.ready != (self.decision == "APPROVED" and self.gates_cleared):
            raise HumanApprovalError(
                "manual_upload.ready must exactly match the approved, gates-cleared decision"
            )
        if not isinstance(self.supersedes_approval_id, str):
            raise HumanApprovalError("supersedes_approval_id must be text")
        if self.supersedes_approval_id:
            if not _APPROVAL_ID_RE.fullmatch(self.supersedes_approval_id):
                raise HumanApprovalError("supersedes_approval_id must be a valid approval id")
            if self.supersedes_approval_id == self.approval_id:
                raise HumanApprovalError("an approval cannot supersede itself")
        if self.approval_id != human_approval_id_for(self.run_id, self.target_ref):
            raise HumanApprovalError("approval_id does not match run_id and target_ref")
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
        if not isinstance(data, dict):
            raise HumanApprovalError(f"HumanApprovalResult must be a mapping, got {data!r}")
        header = dict(data)
        required_header = {
            "schema",
            "schema_version",
            "created_at",
            "run_id",
            "stage",
            "produced_by",
            "data",
        }
        missing_header = sorted(required_header - set(header))
        if missing_header:
            raise HumanApprovalError(
                "HumanApprovalResult missing envelope fields: " + ", ".join(missing_header)
            )
        if header["schema"] != SCHEMA:
            raise HumanApprovalError(f"schema must be {SCHEMA!r}, got {header['schema']!r}")
        if (
            isinstance(header["schema_version"], bool)
            or not isinstance(header["schema_version"], int)
            or header["schema_version"] != SCHEMA_VERSION
        ):
            raise HumanApprovalError(
                f"schema_version must be {SCHEMA_VERSION}, got {header['schema_version']!r}"
            )
        if not isinstance(header["created_at"], str) or not header["created_at"].strip():
            raise HumanApprovalError("HumanApprovalResult created_at must be an ISO timestamp")
        if not isinstance(header["run_id"], str):
            raise HumanApprovalError("HumanApprovalResult run_id must be text")
        validate_run_id(header["run_id"])
        if header["stage"] != STAGE:
            raise HumanApprovalError(f"stage must be {STAGE!r}, got {header['stage']!r}")
        if not isinstance(header["produced_by"], dict):
            raise HumanApprovalError("HumanApprovalResult produced_by must be a mapping")
        payload = header["data"]
        if not isinstance(payload, dict):
            raise HumanApprovalError("HumanApprovalResult data must be a mapping")
        required_payload = {
            "approval_id",
            "target_ref",
            "decision",
            "decision_at",
            "approver",
            "reviewed_artifacts",
            "blocking_conditions",
            "prior_gate_verdicts",
            "overridden_verdicts",
            "manual_upload",
            "notes",
            "changes_requested",
            "supersedes_approval_id",
            "provenance",
        }
        missing_payload = sorted(required_payload - set(payload))
        if missing_payload:
            raise HumanApprovalError(
                "HumanApprovalResult missing data fields: " + ", ".join(missing_payload)
            )
        for field_name in (
            "reviewed_artifacts",
            "blocking_conditions",
            "overridden_verdicts",
            "changes_requested",
        ):
            if not isinstance(payload[field_name], list):
                raise HumanApprovalError(f"{field_name} must be a JSON array")
        if not isinstance(payload["prior_gate_verdicts"], dict):
            raise HumanApprovalError("prior_gate_verdicts must be a mapping")
        if not isinstance(payload["provenance"], dict):
            raise HumanApprovalError("provenance must be a mapping")
        if not isinstance(payload["notes"], str):
            raise HumanApprovalError("notes must be text")
        if not isinstance(payload["supersedes_approval_id"], str):
            raise HumanApprovalError("supersedes_approval_id must be text")
        try:
            result = cls(
                approval_id=_require(payload["approval_id"], "approval_id"),
                target_ref=_require(payload["target_ref"], "target_ref"),
                decision=_require(payload["decision"], "decision"),
                decision_at=_require(payload["decision_at"], "decision_at"),
                approver=Approver.from_dict(payload["approver"]),
                reviewed_artifacts=tuple(
                    ReviewedArtifact.from_dict(item)
                    for item in payload["reviewed_artifacts"]
                ),
                blocking_conditions=tuple(
                    BlockingCondition.from_dict(item)
                    for item in payload["blocking_conditions"]
                ),
                prior_gate_verdicts=dict(payload["prior_gate_verdicts"]),
                overridden_verdicts=tuple(
                    OverriddenVerdict.from_dict(item)
                    for item in payload["overridden_verdicts"]
                ),
                manual_upload=ManualUploadReadiness.from_dict(payload["manual_upload"]),
                notes=payload["notes"],
                changes_requested=tuple(
                    RequestedChange.from_dict(item)
                    for item in payload["changes_requested"]
                ),
                supersedes_approval_id=payload["supersedes_approval_id"],
                provenance=Provenance.from_dict(payload["provenance"]),
                schema=header["schema"],
                schema_version=header["schema_version"],
                created_at=header["created_at"],
                stage=header["stage"],
                run_id=header["run_id"],
                produced_by=dict(header["produced_by"]),
            )
        except HumanApprovalError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise HumanApprovalError(f"malformed HumanApprovalResult: {exc}") from exc
        return result.validate()

    @classmethod
    def from_json(cls, text: str) -> "HumanApprovalResult":
        if not isinstance(text, str):
            raise HumanApprovalError("HumanApprovalResult JSON must be text")
        try:
            data = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise HumanApprovalError(f"invalid HumanApprovalResult JSON: {exc}") from exc
        return cls.from_dict(data)
