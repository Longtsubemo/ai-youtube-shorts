"""Stage-10 verdict gating, override handling, and freshness evaluation.

The gate's core guarantee (docs/POLICY_AND_SAFETY.md): a human decision is
recorded ONLY over the exact artifact versions that were reviewed, and APPROVED
is only possible when every deterministic hard-gate condition is satisfied
(fail closed). No written record means there is no approval.

Hard gates for APPROVED (each is a BlockingCondition):
    required_artifacts          - every required role is present & loaded
    policy_qa_hard_reject       - Stage-9 policy verdict is not a hard REJECT
    video_qa_hard_reject        - Stage-8 QA verdict is not a hard REJECT
    video_generation_completed  - the Stage-7 video was actually completed
    review_overrides            - every soft-REVIEW finding either does not
                                  exist or has an explicit, version-bound
                                  human override (when configured)

CHANGES_REQUESTED and REJECTED are ALWAYS valid human decisions: they record
the blocking conditions (and any requested changes) for audit. APPROVED under
an unsatisfied condition raises HumanApprovalError and writes nothing.

Freshness: `check_approval_freshness` recomputes the content digests of the
current artifacts and compares id + digest against a recorded approval, so an
APPROVED record whose reviewed versions no longer match current content is
STALE and must not be treated as alive (`approval_is_effective`).
"""

from __future__ import annotations

import dataclasses

from agents.human_approval.models import (
    MANUAL_UPLOAD_NOTE,
    PRODUCED_BY,
    PROVENANCE_GENERATED_BY,
    PROVENANCE_SOURCE,
    ROLES,
    ROLE_ATTR,
    ROLE_ID_FIELD,
    ROLE_SCHEMA,
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
    _utc_now,
    content_sha256,
    human_approval_id_for,
)
from agents.seo.models import Provenance

DECISION_APPROVED = "APPROVED"
DECISION_CHANGES_REQUESTED = "CHANGES_REQUESTED"
DECISION_REJECTED = "REJECTED"

VERDICT_REJECT = "REJECT"
VERDICT_REVIEW = "REVIEW"
STATUS_COMPLETED = "completed"

#: Roles with a deterministic prior verdict that can hard-block APPROVED.
GATED_ROLES = ("policy_qa", "video_qa")


def _artifact_for(inputs: ApprovalInputs, role: str):
    return getattr(inputs, ROLE_ATTR[role])


def build_reviewed_artifacts(
    inputs: ApprovalInputs,
) -> tuple[ReviewedArtifact, ...]:
    """Bind the EXACT versions present: role + stable id + content digest."""
    artifacts: list[ReviewedArtifact] = []
    for role in ROLES:
        obj = _artifact_for(inputs, role)
        if obj is None:
            continue
        artifact_id = getattr(obj, ROLE_ID_FIELD[role], "") or ""
        digest = content_sha256(obj.to_dict())
        artifacts.append(
            ReviewedArtifact(
                role=role,
                artifact_id=artifact_id,
                content_sha256=digest,
                schema=ROLE_SCHEMA.get(role, ""),
            ).validate()
        )
    return tuple(artifacts)


def build_prior_gate_verdicts(inputs: ApprovalInputs) -> dict:
    """Prior-stage verdicts the human overrode or left untouched (for audit)."""
    policy = inputs.policy_qa
    video_qa = inputs.video_qa
    video_result = inputs.video_result
    script_qa = inputs.script_qa
    return {
        "policy_qa": policy.overall_verdict if policy is not None else "",
        "video_qa": video_qa.overall_verdict if video_qa is not None else "",
        "video_generation": video_result.overall_status if video_result is not None else "",
        "script_qa": script_qa.overall_verdict if script_qa is not None else "",
    }


def _review_gates_awaiting(inputs: ApprovalInputs) -> tuple[str, ...]:
    awaiting: list[str] = []
    for role in GATED_ROLES:
        obj = _artifact_for(inputs, role)
        if obj is not None and getattr(obj, "overall_verdict", "") == VERDICT_REVIEW:
            awaiting.append(role)
    return tuple(awaiting)


def build_blocking_conditions(
    inputs: ApprovalInputs,
    config: HumanApprovalConfig,
    overridden_verdicts: tuple[OverriddenVerdict, ...],
    decision: str,
) -> tuple[BlockingCondition, ...]:
    """Deterministic hard-gate evaluation for APPROVED (all roles present)."""
    policy = inputs.policy_qa
    video_qa = inputs.video_qa
    video_result = inputs.video_result

    awaiting = _review_gates_awaiting(inputs)
    missing_overrides = tuple(
        role for role in awaiting
        if config.require_override_for_review
        and not any(override.gate == role for override in overridden_verdicts)
    )

    conditions = (
        BlockingCondition(
            code="required_artifacts",
            satisfied=True,
            detail="required roles present: " + ", ".join(config.required_roles),
        ),
        BlockingCondition(
            code="policy_qa_hard_reject",
            satisfied=policy is not None and policy.overall_verdict != VERDICT_REJECT,
            detail=policy.overall_verdict if policy is not None else "missing",
        ),
        BlockingCondition(
            code="video_qa_hard_reject",
            satisfied=video_qa is not None and video_qa.overall_verdict != VERDICT_REJECT,
            detail=video_qa.overall_verdict if video_qa is not None else "missing",
        ),
        BlockingCondition(
            code="video_generation_completed",
            satisfied=video_result is not None
            and video_result.overall_status == STATUS_COMPLETED,
            detail=video_result.overall_status if video_result is not None else "missing",
        ),
        BlockingCondition(
            code="review_overrides",
            satisfied=not missing_overrides,
            detail=(
                "no unresolved soft REVIEW findings"
                if not awaiting
                else "soft REVIEW findings awaiting recorded override: "
                + ", ".join(awaiting)
            ),
        ),
    )
    if decision != DECISION_APPROVED:
        return conditions
    return tuple(
        dataclasses.replace(
            condition,
            detail=condition.detail + " (blocking for APPROVED)",
        )
        for condition in conditions
    )


def _bind_override_artifacts(
    overridden_verdicts: tuple[OverriddenVerdict, ...],
    reviewed: tuple[ReviewedArtifact, ...],
) -> tuple[OverriddenVerdict, ...]:
    """Bind each override to the EXACT reviewed artifact version it resolved."""
    bound: list[OverriddenVerdict] = []
    by_role = {artifact.role: artifact.artifact_id for artifact in reviewed}
    for override in overridden_verdicts:
        bound.append(
            dataclasses.replace(
                override,
                artifact_id=override.artifact_id or by_role.get(override.gate, ""),
            ).validate()
        )
    return tuple(bound)


def evaluate_human_approval(
    inputs: ApprovalInputs,
    config: HumanApprovalConfig | None = None,
    *,
    decision: str,
    approver: Approver,
    changes_requested: tuple[RequestedChange, ...] = (),
    overridden_verdicts: tuple[OverriddenVerdict, ...] = (),
    notes: str = "",
    supersedes_approval_id: str = "",
    run_id: str = "",
    created_at: str = "",
) -> HumanApprovalResult:
    """Evaluate a human decision over `inputs`; returns the Stage-10 artifact.

    Raises HumanApprovalError when the review bundle is incomplete or when an
    APPROVED decision hits an unsatisfied hard gate (fail closed: nothing is
    recorded, and no approval can be implied).
    """
    config = (config or HumanApprovalConfig()).validate()
    approver.validate()

    if decision not in WRITABLE_DECISIONS:
        raise HumanApprovalError(
            f"decision must be one of {WRITABLE_DECISIONS} "
            f"(PENDING is not a writable decision), got {decision!r}"
        )

    for role in config.required_roles:
        if _artifact_for(inputs, role) is None:
            raise HumanApprovalError(
                f"cannot review: missing required {role} artifact "
                "(fail closed; no decision is recorded without a complete bundle)"
            )

    policy = inputs.policy_qa
    target_ref = getattr(policy, "target_ref", "") or getattr(inputs.plan, "production_plan_id", "")
    if not target_ref:
        raise HumanApprovalError("approval needs a production-plan target_ref")

    reviewed_artifacts = build_reviewed_artifacts(inputs)
    prior_gate_verdicts = build_prior_gate_verdicts(inputs)
    blocking_conditions = build_blocking_conditions(
        inputs, config, overridden_verdicts, decision
    )

    # Overrides may only resolve an existing soft REVIEW, never re-label a
    # verdict that did not occur (prevents misapplied or fabricated overrides).
    for override in overridden_verdicts:
        override.validate()
        prior_verdict = prior_gate_verdicts.get(override.gate, "")
        if prior_verdict != VERDICT_REVIEW:
            raise HumanApprovalError(
                f"override for gate {override.gate!r} is invalid: its prior "
                f"verdict is {prior_verdict!r}, not REVIEW"
            )
    bound_overrides = _bind_override_artifacts(overridden_verdicts, reviewed_artifacts)

    def gates_cleared() -> bool:
        return all(condition.satisfied for condition in blocking_conditions)

    if decision == DECISION_APPROVED and not gates_cleared():
        failed = [
            f"{c.code} ({c.detail})" for c in blocking_conditions if not c.satisfied
        ]
        raise HumanApprovalError(
            "cannot APPROVE; blocking conditions unsatisfied: " + "; ".join(failed)
        )

    if decision == DECISION_CHANGES_REQUESTED:
        if not changes_requested and not notes.strip():
            raise HumanApprovalError(
                "CHANGES_REQUESTED needs at least one requested change or a note"
            )
    for change in changes_requested:
        change.validate()

    approval_id = human_approval_id_for(run_id, target_ref)
    if supersedes_approval_id == approval_id:
        raise HumanApprovalError("an approval cannot supersede itself")

    script = inputs.script
    title = getattr(script, "title", "") or ""
    if not title and policy is not None:
        title = getattr(policy, "topic_title", "") or ""
    manual_upload = ManualUploadReadiness(
        ready=decision == DECISION_APPROVED and gates_cleared(),
        required=True,
        note=MANUAL_UPLOAD_NOTE,
        artifacts=tuple(artifact.artifact_id for artifact in reviewed_artifacts),
        title=title,
    )

    provenance = Provenance(
        generated_by=PROVENANCE_GENERATED_BY,
        source=PROVENANCE_SOURCE,
        created_at=created_at or _utc_now(),
        provider=None,
        model=None,
        prompt_version=config.schema_version,
    )

    result = HumanApprovalResult(
        approval_id=approval_id,
        target_ref=target_ref,
        decision=decision,
        decision_at=created_at or _utc_now(),
        approver=approver,
        reviewed_artifacts=reviewed_artifacts,
        blocking_conditions=blocking_conditions,
        prior_gate_verdicts=prior_gate_verdicts,
        overridden_verdicts=bound_overrides,
        manual_upload=manual_upload,
        notes=notes,
        changes_requested=changes_requested,
        supersedes_approval_id=supersedes_approval_id,
        provenance=provenance,
        created_at=created_at or _utc_now(),
        run_id=run_id,
        produced_by=dict(PRODUCED_BY),
    )
    return result.validate()


# --------------------------------------------------------------------------- #
# Freshness / derived state
# --------------------------------------------------------------------------- #


def _current_digest(obj) -> str:
    if obj is None:
        return ""
    return content_sha256(obj.to_dict())


def check_approval_freshness(
    approval: HumanApprovalResult,
    current_artifacts: dict[str, object],
) -> bool:
    """True when every reviewed artifact still matches (id + content digest).

    `current_artifacts` maps role -> the CURRENT artifact object. A missing
    role, a changed stable id, or a changed content digest makes the approval
    STALE (returns False).
    """
    for artifact in approval.reviewed_artifacts:
        current = current_artifacts.get(artifact.role)
        if current is None:
            return False
        current_id = getattr(current, ROLE_ID_FIELD[artifact.role], "") or ""
        if current_id != artifact.artifact_id:
            return False
        if _current_digest(current) != artifact.content_sha256:
            return False
    return True


def is_superseded_by(
    approval: HumanApprovalResult,
    later: HumanApprovalResult,
) -> bool:
    """True when a later decision explicitly replaces this record."""
    return (
        later.supersedes_approval_id == approval.approval_id
        and later.approval_id != approval.approval_id
    )


def approval_is_effective(
    approval: HumanApprovalResult,
    current_artifacts: dict[str, object],
    later: HumanApprovalResult | None = None,
) -> bool:
    """An APPROVED record is effective only if fresh, gates clear, not replaced.

    This is the derived-state check: APPROVED + all blocking conditions
    satisfied + digests still match the current artifacts + not superseded by
    a later decision. Anything less is STALE / SUPERSEDED and must not be
    treated as a live approval.
    """
    if not approval.is_approved:
        return False
    if not approval.gates_cleared:
        return False
    if not check_approval_freshness(approval, current_artifacts):
        return False
    if later is not None and is_superseded_by(approval, later):
        return False
    return True