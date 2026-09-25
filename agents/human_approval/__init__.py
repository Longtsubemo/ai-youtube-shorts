"""Stage 10 - Human Approval: records explicit human sign-off over exact versions.

Docs: docs/DATA_CONTRACTS.md §12, docs/PROJECT_PROGRESS.md Stage 10,
docs/POLICY_AND_SAFETY.md.
CLI: python -m agents.human_approval.

NOTE: this package imports only models + validator here (same convention as
agents.qa.policy / agents.qa.script / agents.qa.video): the stage runner and
manual runner live in submodules imported by path so `python -m
agents.human_approval` runs without a double-import RuntimeWarning.

Stage 10 is HUMAN APPROVAL, a GATE -- it does NOT implement Stage 11 (YouTube
upload/scheduling). It is deterministic, network-free, GPU-free, API-free, and
is NOT a YouTube compliance guarantee. The artifact header `stage` label is
"human_approval"; an approved decision advances pipeline bookkeeping to the
existing generic Stage.PUBLISH (`STATE_STAGE` in models.py), while a
non-approved decision remains at Stage.QUALITY_CHECK. No Stage enum member was
added or renamed.
"""

from __future__ import annotations

from agents.human_approval.models import (
    DECISIONS,
    DERIVED_STATES,
    HUMAN_APPROVAL_CONFIG_VERSION,
    MANUAL_UPLOAD_NOTE,
    REQUIRED_ROLES,
    ROLES,
    SCHEMA,
    SCHEMA_VERSION,
    STAGE,
    STATE_STAGE,
    WRITABLE_DECISIONS,
    ApprovalInputs,
    Approver,
    BlockingCondition,
    HumanApprovalConfig,
    HumanApprovalError,
    HumanApprovalResult,
    ManualUploadReadiness,
    OverriddenVerdict,
    RequestedChange,
    ReviewedArtifact,
    content_sha256,
    human_approval_id_for,
    validate_run_id,
)
from agents.human_approval.validator import (
    DECISION_APPROVED,
    DECISION_CHANGES_REQUESTED,
    DECISION_REJECTED,
    approval_is_effective,
    build_blocking_conditions,
    build_prior_gate_verdicts,
    build_reviewed_artifacts,
    check_approval_freshness,
    evaluate_human_approval,
    is_superseded_by,
    validate_approval_inputs,
)

__all__ = [
    "DECISIONS",
    "DERIVED_STATES",
    "HUMAN_APPROVAL_CONFIG_VERSION",
    "MANUAL_UPLOAD_NOTE",
    "REQUIRED_ROLES",
    "ROLES",
    "SCHEMA",
    "SCHEMA_VERSION",
    "STAGE",
    "STATE_STAGE",
    "WRITABLE_DECISIONS",
    "DECISION_APPROVED",
    "DECISION_CHANGES_REQUESTED",
    "DECISION_REJECTED",
    "ApprovalInputs",
    "Approver",
    "BlockingCondition",
    "HumanApprovalConfig",
    "HumanApprovalError",
    "HumanApprovalResult",
    "ManualUploadReadiness",
    "OverriddenVerdict",
    "RequestedChange",
    "ReviewedArtifact",
    "approval_is_effective",
    "build_blocking_conditions",
    "build_prior_gate_verdicts",
    "build_reviewed_artifacts",
    "check_approval_freshness",
    "content_sha256",
    "evaluate_human_approval",
    "human_approval_id_for",
    "is_superseded_by",
    "validate_approval_inputs",
    "validate_run_id",
]