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
    validate_run_id,
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
from agents.qa.fact_checker.models import result_id_for
from agents.qa.policy.models import policy_result_id_for
from agents.qa.script.models import qa_result_id_for as script_qa_result_id_for
from agents.qa.video.models import qa_result_id_for as video_qa_result_id_for
from agents.scene_planner.models import production_plan_id_for
from agents.script.models import script_id_for
from agents.seo.models import Provenance
from agents.video_generation.models import video_generation_result_id_for

DECISION_APPROVED = "APPROVED"
DECISION_CHANGES_REQUESTED = "CHANGES_REQUESTED"
DECISION_REJECTED = "REJECTED"

VERDICT_REJECT = "REJECT"
VERDICT_REVIEW = "REVIEW"
STATUS_COMPLETED = "completed"

#: Roles with a deterministic prior verdict that can hard-block APPROVED.
GATED_ROLES = ("policy_qa", "video_qa")
_ROLE_STAGE = {
    "policy_qa": "policy_qa",
    "video_qa": "video",
    "video_result": "video_generation",
    "production_plan": "scene_planning",
    "script": "script_generation",
    "script_qa": "script_qa",
    "fact_check": "fact_check",
}


def _artifact_for(inputs: ApprovalInputs, role: str):
    return getattr(inputs, ROLE_ATTR[role])


def _same(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise HumanApprovalError(f"{label}: {actual!r} does not match {expected!r}")


def _validate_typed_artifact(role: str, artifact: object) -> None:
    expected_schema = ROLE_SCHEMA[role]
    if getattr(artifact, "schema", None) != expected_schema:
        raise HumanApprovalError(
            f"{role} schema must be {expected_schema!r}, got {getattr(artifact, 'schema', None)!r}"
        )
    schema_version = getattr(artifact, "schema_version", None)
    if isinstance(schema_version, bool) or schema_version != 1:
        raise HumanApprovalError(
            f"{role} schema_version must be 1, got {schema_version!r}"
        )
    if getattr(artifact, "stage", None) != _ROLE_STAGE[role]:
        raise HumanApprovalError(
            f"{role} stage must be {_ROLE_STAGE[role]!r}, got {getattr(artifact, 'stage', None)!r}"
        )
    if not isinstance(getattr(artifact, "produced_by", None), dict):
        raise HumanApprovalError(f"{role} produced_by must be a mapping")
    if not isinstance(getattr(artifact, "created_at", None), str) or not artifact.created_at:
        raise HumanApprovalError(f"{role} created_at must be non-empty text")
    validate_run_id(getattr(artifact, "run_id", None))
    artifact_id = getattr(artifact, ROLE_ID_FIELD[role], None)
    if not isinstance(artifact_id, str) or not artifact_id:
        raise HumanApprovalError(f"{role} artifact id must be non-empty text")
    validator = getattr(artifact, "validate", None)
    if not callable(validator):
        raise HumanApprovalError(f"{role} must provide a validate() method")
    try:
        validator()
    except HumanApprovalError:
        raise
    except (AttributeError, TypeError, ValueError) as exc:
        raise HumanApprovalError(f"{role} validation failed: {exc}") from exc


def _gate_consistency(inputs: ApprovalInputs) -> None:
    for role in GATED_ROLES:
        artifact = _artifact_for(inputs, role)
        if artifact is None:
            continue
        issues = tuple(getattr(artifact, "all_issues")())
        errors = tuple(issue for issue in issues if issue.severity == "ERROR")
        hard = any(issue.gate == "hard" for issue in errors)
        derived = "REJECT" if hard else "REVIEW" if errors else "PASS"
        if artifact.overall_verdict != derived:
            raise HumanApprovalError(
                f"{role} overall_verdict {artifact.overall_verdict!r} is inconsistent "
                f"with its issue stream (derived {derived!r})"
            )


def validate_approval_inputs(
    inputs: ApprovalInputs,
    config: HumanApprovalConfig | None = None,
) -> str:
    """Validate the complete Stage-10 bundle and return its source run id."""
    if not isinstance(inputs, ApprovalInputs):
        raise HumanApprovalError("inputs must be ApprovalInputs")
    config = (config or HumanApprovalConfig()).validate()
    for role in config.required_roles:
        if _artifact_for(inputs, role) is None:
            raise HumanApprovalError(f"missing required {role} artifact")
    run_ids: list[tuple[str, str]] = []
    for role in ROLES:
        artifact = _artifact_for(inputs, role)
        if artifact is None:
            continue
        _validate_typed_artifact(role, artifact)
        run_id = getattr(artifact, "run_id", None)
        validate_run_id(run_id)
        run_ids.append((role, run_id))
    if not run_ids:
        return ""
    source_run_id = run_ids[0][1]
    mismatched = [(role, value) for role, value in run_ids if value != source_run_id]
    if mismatched:
        details = ", ".join(f"{role}={value!r}" for role, value in run_ids)
        raise HumanApprovalError(
            "Stage-10 inputs must share one run_id before evaluation; " + details
        )

    policy = inputs.policy_qa
    video_qa = inputs.video_qa
    video_result = inputs.video_result
    plan = inputs.plan
    script = inputs.script
    script_qa = inputs.script_qa
    fact_check = inputs.fact_check

    if policy is not None:
        for field_name in ("topic_id", "topic_title", "script_id", "target_ref"):
            if not isinstance(getattr(policy, field_name, None), str) or not getattr(
                policy, field_name, ""
            ).strip():
                raise HumanApprovalError(
                    f"policy_qa is missing required binding field {field_name}"
                )
    if video_qa is not None:
        for field_name in (
            "topic_id",
            "topic_title",
            "script_id",
            "target_ref",
            "video_generation_result_id",
        ):
            if not isinstance(getattr(video_qa, field_name, None), str) or not getattr(
                video_qa, field_name, ""
            ).strip():
                raise HumanApprovalError(
                    f"video_qa is missing required binding field {field_name}"
                )
    if video_result is not None:
        for field_name in (
            "topic_id",
            "topic_title",
            "script_id",
            "production_plan_id",
            "script_qa_result_id",
        ):
            if not isinstance(getattr(video_result, field_name, None), str) or not getattr(
                video_result, field_name, ""
            ).strip():
                raise HumanApprovalError(
                    f"video_result is missing required binding field {field_name}"
                )
    if plan is not None:
        for field_name in ("topic_id", "topic_title", "script_id", "script_qa_result_id"):
            if not isinstance(getattr(plan, field_name, None), str) or not getattr(
                plan, field_name, ""
            ).strip():
                raise HumanApprovalError(
                    f"production_plan is missing required binding field {field_name}"
                )
    if script is not None:
        for field_name in ("script_id", "topic_id", "topic_title"):
            if not isinstance(getattr(script, field_name, None), str) or not getattr(
                script, field_name, ""
            ).strip():
                raise HumanApprovalError(
                    f"script is missing required binding field {field_name}"
                )

    if policy is not None and plan is not None:
        if plan.production_plan_id != policy.target_ref:
            raise HumanApprovalError(
                f"production plan {plan.production_plan_id} does not match policy "
                f"target_ref {policy.target_ref}"
            )
    if video_qa is not None and plan is not None and video_qa.target_ref != plan.production_plan_id:
        raise HumanApprovalError(
            f"video QA {video_qa.qa_result_id} targets {video_qa.target_ref}, "
            f"not {plan.production_plan_id}"
        )
    if video_result is not None and plan is not None:
        if video_result.production_plan_id != plan.production_plan_id:
            raise HumanApprovalError(
                f"video result {video_result.video_generation_result_id} targets "
                f"production plan {video_result.production_plan_id}, not "
                f"{plan.production_plan_id}"
            )
        _same("video result script_id", video_result.script_id, plan.script_id)
        _same("video result topic_id", video_result.topic_id, plan.topic_id)
        _same("video result topic_title", video_result.topic_title, plan.topic_title)
        _same(
            "video result script_qa_result_id",
            video_result.script_qa_result_id,
            plan.script_qa_result_id,
        )
    if video_qa is not None and video_result is not None:
        _same(
            "video QA video_generation_result_id",
            video_qa.video_generation_result_id,
            video_result.video_generation_result_id,
        )
        _same("video QA script_id", video_qa.script_id, video_result.script_id)
        _same("video QA topic_id", video_qa.topic_id, video_result.topic_id)
        _same("video QA topic_title", video_qa.topic_title, video_result.topic_title)
    if plan is not None and script is not None:
        if plan.script_id != script.script_id:
            raise HumanApprovalError(
                f"production plan {plan.production_plan_id} expects script "
                f"{plan.script_id}, not {script.script_id}"
            )
        _same("script topic_id", script.topic_id, plan.topic_id)
        _same("script topic_title", script.topic_title, plan.topic_title)
    if policy is not None and script is not None:
        _same("policy script_id", policy.script_id, script.script_id)
        _same("policy topic_id", policy.topic_id, script.topic_id)
        _same("policy topic_title", policy.topic_title, script.topic_title)
    if video_qa is not None and policy is not None:
        _same("video QA script_id", video_qa.script_id, policy.script_id)
        _same("video QA topic_id", video_qa.topic_id, policy.topic_id)
        _same("video QA topic_title", video_qa.topic_title, policy.topic_title)
        _same("video QA target_ref", video_qa.target_ref, policy.target_ref)
    if script_qa is not None:
        if script is not None:
            _same("script QA script_id", script_qa.script_id, script.script_id)
            _same("script QA topic_id", script_qa.topic_id, script.topic_id)
        if plan is not None:
            _same(
                "production plan script_qa_result_id",
                plan.script_qa_result_id,
                script_qa.qa_result_id,
            )
        if fact_check is not None:
            _same("script QA fact-check result_id", script_qa.result_id, fact_check.result_id)
            _same("script QA fact-check package_id", script_qa.package_id, fact_check.package_id)
    if fact_check is not None and script is not None:
        _same("script fact-check result_id", script.result_id, fact_check.result_id)
        _same("script fact-check package_id", script.package_id, fact_check.package_id)
        _same("fact-check topic_id", fact_check.topic_id, script.topic_id)
        _same("fact-check topic_title", fact_check.topic_title, script.topic_title)

    expected_ids = {
        "policy_qa": (
            policy is not None,
            policy_result_id_for(source_run_id, policy.target_ref) if policy is not None else None,
        ),
        "video_qa": (
            video_qa is not None,
            video_qa_result_id_for(source_run_id, video_qa.target_ref)
            if video_qa is not None
            else None,
        ),
        "video_result": (
            video_result is not None,
            video_generation_result_id_for(source_run_id, video_result.production_plan_id)
            if video_result is not None
            else None,
        ),
        "production_plan": (
            plan is not None,
            production_plan_id_for(source_run_id, plan.script_id) if plan is not None else None,
        ),
        "script": (
            script is not None,
            script_id_for(source_run_id, script.topic_id) if script is not None else None,
        ),
        "script_qa": (
            script_qa is not None,
            script_qa_result_id_for(source_run_id, script_qa.script_id)
            if script_qa is not None
            else None,
        ),
        "fact_check": (
            fact_check is not None,
            result_id_for(source_run_id, fact_check.package_id)
            if fact_check is not None
            else None,
        ),
    }
    for role, (present, expected_id) in expected_ids.items():
        if not present:
            continue
        artifact = _artifact_for(inputs, role)
        actual_id = getattr(artifact, ROLE_ID_FIELD[role], None)
        _same(f"{role} deterministic id", actual_id, expected_id)

    topic_sources = [
        (role, getattr(obj, "topic_id", ""))
        for role, obj in (
            ("policy_qa", policy),
            ("video_qa", video_qa),
            ("video_result", video_result),
            ("production_plan", plan),
            ("script", script),
            ("script_qa", script_qa),
            ("fact_check", fact_check),
        )
        if obj is not None
    ]
    if topic_sources:
        topic_id = topic_sources[0][1]
        for role, value in topic_sources[1:]:
            _same(f"{role} topic_id", value, topic_id)
    title_sources = [
        (role, getattr(obj, "topic_title", ""))
        for role, obj in (
            ("policy_qa", policy),
            ("video_qa", video_qa),
            ("video_result", video_result),
            ("production_plan", plan),
            ("script", script),
            ("fact_check", fact_check),
        )
        if obj is not None
    ]
    if title_sources:
        topic_title = title_sources[0][1]
        for role, value in title_sources[1:]:
            _same(f"{role} topic_title", value, topic_title)

    _gate_consistency(inputs)
    return source_run_id


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


def _review_findings(inputs: ApprovalInputs) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    findings: dict[str, set[str]] = {role: set() for role in GATED_ROLES}
    inconsistent: list[str] = []
    for role in GATED_ROLES:
        artifact = _artifact_for(inputs, role)
        if artifact is None or artifact.overall_verdict != VERDICT_REVIEW:
            continue
        for issue in artifact.all_issues():
            if issue.severity != "ERROR" or issue.gate != "soft":
                continue
            check_id = (
                getattr(issue, "check_id", "")
                if role == "policy_qa"
                else getattr(issue, "check", "") or getattr(issue, "category", "")
            )
            if check_id:
                findings[role].add(check_id)
        if not findings[role]:
            inconsistent.append(role)
    return (
        {role: tuple(sorted(values)) for role, values in findings.items()},
        tuple(inconsistent),
    )


def _validate_overrides(
    inputs: ApprovalInputs,
    overridden_verdicts: tuple[OverriddenVerdict, ...],
) -> None:
    if not isinstance(inputs, ApprovalInputs):
        raise HumanApprovalError("inputs must be ApprovalInputs")
    if not isinstance(overridden_verdicts, tuple):
        raise HumanApprovalError("overridden_verdicts must be a tuple")
    findings, inconsistent = _review_findings(inputs)
    seen: set[tuple[str, str]] = set()
    for override in overridden_verdicts:
        override.validate()
        prior_verdict = build_prior_gate_verdicts(inputs).get(override.gate, "")
        if prior_verdict != VERDICT_REVIEW:
            raise HumanApprovalError(
                f"override for gate {override.gate!r} is invalid: its prior "
                f"verdict is {prior_verdict!r}, not REVIEW"
            )
        if override.verdict != VERDICT_REVIEW:
            raise HumanApprovalError(
                f"override for {override.gate}/{override.check_id} must record "
                f"the actual REVIEW verdict, got {override.verdict!r}"
            )
        if override.decision != DECISION_APPROVED:
            raise HumanApprovalError(
                f"override for {override.gate}/{override.check_id} must resolve "
                f"to APPROVED, got {override.decision!r}"
            )
        key = (override.gate, override.check_id)
        if not override.check_id or override.check_id not in findings.get(override.gate, ()):
            raise HumanApprovalError(
                f"override for {override.gate}/{override.check_id!r} does not identify "
                "an actual soft-REVIEW finding"
            )
        if key in seen:
            raise HumanApprovalError(
                f"duplicate override for {override.gate}/{override.check_id}"
            )
        seen.add(key)
        artifact = _artifact_for(inputs, override.gate)
        artifact_id = getattr(artifact, ROLE_ID_FIELD[override.gate], "")
        if override.artifact_id and override.artifact_id != artifact_id:
            raise HumanApprovalError(
                f"override for {override.gate}/{override.check_id} is bound to "
                f"{override.artifact_id!r}, not reviewed artifact {artifact_id!r}"
            )
    if inconsistent:
        raise HumanApprovalError(
            "soft-REVIEW gate has no concrete finding: " + ", ".join(inconsistent)
        )


def build_blocking_conditions(
    inputs: ApprovalInputs,
    config: HumanApprovalConfig,
    overridden_verdicts: tuple[OverriddenVerdict, ...],
    decision: str,
) -> tuple[BlockingCondition, ...]:
    """Deterministic hard-gate evaluation for APPROVED (all roles present)."""
    if not isinstance(inputs, ApprovalInputs):
        raise HumanApprovalError("inputs must be ApprovalInputs")
    if not isinstance(config, HumanApprovalConfig):
        raise HumanApprovalError("config must be HumanApprovalConfig")
    config.validate()
    if not isinstance(overridden_verdicts, tuple):
        raise HumanApprovalError("overridden_verdicts must be a tuple")
    for override in overridden_verdicts:
        if not isinstance(override, OverriddenVerdict):
            raise HumanApprovalError("overridden_verdicts contains an invalid override")
        override.validate()
    policy = inputs.policy_qa
    video_qa = inputs.video_qa
    video_result = inputs.video_result

    findings, inconsistent_reviews = _review_findings(inputs)
    override_keys = {
        (override.gate, override.check_id)
        for override in overridden_verdicts
    }
    missing_overrides = tuple(
        f"{role}/{check_id}"
        for role in GATED_ROLES
        for check_id in findings[role]
        if config.require_override_for_review
        and (role, check_id) not in override_keys
    )
    review_gate_satisfied = not inconsistent_reviews and not missing_overrides

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
            satisfied=review_gate_satisfied,
            detail=(
                "no unresolved soft REVIEW findings"
                if review_gate_satisfied
                else "soft REVIEW findings awaiting recorded override: "
                + ", ".join(missing_overrides or inconsistent_reviews)
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
        artifact_id = by_role.get(override.gate, "")
        if not artifact_id:
            raise HumanApprovalError(
                f"override gate {override.gate!r} has no reviewed artifact"
            )
        if override.artifact_id and override.artifact_id != artifact_id:
            raise HumanApprovalError(
                f"override for {override.gate} is bound to {override.artifact_id!r}, "
                f"not {artifact_id!r}"
            )
        bound.append(
            dataclasses.replace(override, artifact_id=artifact_id).validate()
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
    source_run_id = validate_approval_inputs(inputs, config)
    if not isinstance(approver, Approver):
        raise HumanApprovalError("approver must be an Approver")
    if not isinstance(notes, str):
        raise HumanApprovalError("notes must be text")
    if not isinstance(changes_requested, tuple):
        raise HumanApprovalError("changes_requested must be a tuple")
    if not isinstance(overridden_verdicts, tuple):
        raise HumanApprovalError("overridden_verdicts must be a tuple")
    if not isinstance(supersedes_approval_id, str):
        raise HumanApprovalError("supersedes_approval_id must be text")
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
    effective_run_id = validate_run_id(run_id or source_run_id)

    reviewed_artifacts = build_reviewed_artifacts(inputs)
    prior_gate_verdicts = build_prior_gate_verdicts(inputs)
    _validate_overrides(inputs, overridden_verdicts)
    blocking_conditions = build_blocking_conditions(
        inputs, config, overridden_verdicts, decision
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

    approval_id = human_approval_id_for(effective_run_id, target_ref)
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

    timestamp = created_at or _utc_now()
    provenance = Provenance(
        generated_by=PROVENANCE_GENERATED_BY,
        source=PROVENANCE_SOURCE,
        created_at=timestamp,
        provider=None,
        model=None,
        prompt_version=config.schema_version,
    )

    result = HumanApprovalResult(
        approval_id=approval_id,
        target_ref=target_ref,
        decision=decision,
        decision_at=timestamp,
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
        created_at=timestamp,
        run_id=effective_run_id,
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
    try:
        approval.validate()
    except (HumanApprovalError, TypeError, ValueError, AttributeError):
        return False
    if not isinstance(current_artifacts, dict):
        return False
    for artifact in approval.reviewed_artifacts:
        current = current_artifacts.get(artifact.role)
        if current is None:
            return False
        current_id = getattr(current, ROLE_ID_FIELD[artifact.role], "") or ""
        if current_id != artifact.artifact_id:
            return False
        try:
            current_digest = _current_digest(current)
        except (HumanApprovalError, TypeError, ValueError, AttributeError):
            return False
        if current_digest != artifact.content_sha256:
            return False
    return True


def is_superseded_by(
    approval: HumanApprovalResult,
    later: HumanApprovalResult,
) -> bool:
    """True when a later decision explicitly replaces this record."""
    if not isinstance(approval, HumanApprovalResult) or not isinstance(
        later, HumanApprovalResult
    ):
        return False
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
    a later decision. Current gate inputs are re-evaluated as well, so a
    tampered record cannot turn a current hard REJECT into an effective approval.
    """
    if not isinstance(approval, HumanApprovalResult):
        return False
    if not isinstance(current_artifacts, dict):
        return False
    try:
        if not approval.validate().is_approved:
            return False
        if not approval.gates_cleared:
            return False
    except (HumanApprovalError, TypeError, ValueError, AttributeError):
        return False
    if not check_approval_freshness(approval, current_artifacts):
        return False
    try:
        current_inputs = ApprovalInputs(
            policy_qa=current_artifacts.get("policy_qa"),
            video_qa=current_artifacts.get("video_qa"),
            video_result=current_artifacts.get("video_result"),
            plan=current_artifacts.get("production_plan"),
            script=current_artifacts.get("script"),
            script_qa=current_artifacts.get("script_qa"),
            fact_check=current_artifacts.get("fact_check"),
        )
        config = HumanApprovalConfig().validate()
        validate_approval_inputs(current_inputs, config)
        if approval.prior_gate_verdicts != build_prior_gate_verdicts(current_inputs):
            return False
        _validate_overrides(current_inputs, approval.overridden_verdicts)
        conditions = build_blocking_conditions(
            current_inputs,
            config,
            approval.overridden_verdicts,
            DECISION_APPROVED,
        )
        if not all(condition.satisfied for condition in conditions):
            return False
    except (HumanApprovalError, TypeError, ValueError, AttributeError, KeyError):
        return False
    if later is not None and is_superseded_by(approval, later):
        return False
    return True
