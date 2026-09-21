"""Stage-9 verdict aggregation and the deterministic policy evaluation engine.

The gate's core guarantee (docs/POLICY_AND_SAFETY.md §7): the deterministic
verdict is authoritative. A classifier suggestion (see `classifier.py`) is
advisory only and may only push the verdict toward REJECT; it can never turn a
REJECT into a REVIEW/PASS or a REVIEW into a PASS.
"""

from __future__ import annotations

from typing import Iterable

from agents.qa.policy.classifier import (
    POLICY_VERDICT_RANK,
    PolicyRiskClassifier,
    build_policy_context,
)
from agents.qa.policy.checks import CHECK_FUNCTIONS
from agents.qa.policy.models import (
    POLICY_CHECK_ORDER,
    PolicyCheckRecord,
    PolicyInputs,
    PolicyIssue,
    PolicyQAConfig,
    PolicyQAError,
    PolicyQAResult,
    Provenance,
    RiskSummary,
    _utc_now,
    policy_result_id_for,
)
from agents.qa.policy.rules import POLICY_RULES_BY_ID

VERDICT_PASS = "PASS"
VERDICT_REVIEW = "REVIEW"
VERDICT_REJECT = "REJECT"


def aggregate_verdict(issues: Iterable[PolicyIssue]) -> str:
    """PASS when clean, REVIEW on any soft ERROR, REJECT on any hard ERROR."""
    has_hard_error = False
    has_error = False
    for issue in issues:
        if issue.is_hard_error:
            has_hard_error = True
        elif issue.is_error:
            has_error = True
    if has_hard_error:
        return "REJECT"
    if has_error:
        return "REVIEW"
    return "PASS"


def run_checks(
    inputs: PolicyInputs,
    config: PolicyQAConfig,
) -> tuple[tuple[PolicyIssue, ...], tuple[PolicyCheckRecord, ...]]:
    """Run every registered check in policy order; return (issues, records).

    Issue ids (`pol_00`, `pol_01`, ...) are assigned in stable traversal order
    so repeated runs on identical inputs produce byte-identical artifacts.
    """
    issues: list[PolicyIssue] = []
    records: list[PolicyCheckRecord] = []
    for check_id in POLICY_CHECK_ORDER:
        check_fn = CHECK_FUNCTIONS[check_id]
        check_issues = tuple(check_fn(inputs, config))
        for index, issue in enumerate(check_issues):
            issues.append(
                _with_issue_id(issue, f"pol_{index:02d}")
            )
        rule = POLICY_RULES_BY_ID[check_id]
        records.append(
            PolicyCheckRecord(
                check_id=check_id,
                category=rule.category,
                hard_gate=rule.hard_default,
                errors=sum(1 for i in check_issues if i.is_error),
                warnings=sum(1 for i in check_issues if i.severity == "WARNING"),
                infos=sum(1 for i in check_issues if i.severity == "INFO"),
                hard_errors=sum(1 for i in check_issues if i.is_hard_error),
            )
        )
    return tuple(issues), tuple(records)


def _with_issue_id(issue: PolicyIssue, issue_id: str) -> PolicyIssue:
    return PolicyIssue(
        issue_id=issue_id,
        check_id=issue.check_id,
        category=issue.category,
        severity=issue.severity,
        gate=issue.gate,
        message=issue.message,
        rationale=issue.rationale,
        affected_scene_ids=issue.affected_scene_ids,
        affected_line_ids=issue.affected_line_ids,
        affected_claim_ids=issue.affected_claim_ids,
        affected_asset_ids=issue.affected_asset_ids,
        policy_rule=issue.policy_rule,
        evidence=issue.evidence,
        resolution_hint=issue.resolution_hint,
    ).validate()


def _combined_verdict(deterministic: str, suggestion_verdict: str | None) -> str:
    if suggestion_verdict is None:
        return deterministic
    if POLICY_VERDICT_RANK[suggestion_verdict] > POLICY_VERDICT_RANK[deterministic]:
        return suggestion_verdict
    return deterministic


def evaluate_policy(
    inputs: PolicyInputs,
    config: PolicyQAConfig | None = None,
    run_id: str = "",
    created_at: str = "",
    classifier: PolicyRiskClassifier | None = None,
    resumed_from_policy_result_id: str = "",
) -> PolicyQAResult:
    """Evaluate the full policy gate over `inputs`; returns the artifact.

    `classifier` is optional and advisory; the deterministic verdict is always
    authoritative (see module docstring).
    """
    config = (config or PolicyQAConfig()).validate()

    issues, checks = run_checks(inputs, config)
    deterministic_verdict = aggregate_verdict(issues)

    suggestion_verdict: str | None = None
    classifier_notes = ""
    if classifier is not None:
        suggestion = classifier.classify(build_policy_context(inputs)).validate()
        suggestion_verdict = suggestion.suggested_verdict
        classifier_notes = suggestion.notes
        if POLICY_VERDICT_RANK[suggestion_verdict] > POLICY_VERDICT_RANK[deterministic_verdict]:
            w = PolicyIssue(
                check_id="P15",
                category="prior_gate_consistency",
                severity="WARNING",
                gate="soft",
                message=(
                    f"Classifier suggestion '{suggestion_verdict}' is more "
                    "conservative than the deterministic verdict; the stricter "
                    "value was recorded."
                ),
                rationale="Advisory suggestions may only make the gate more strict.",
                policy_rule=f"P15@{config.policy_version}",
                evidence=f"model={suggestion.model_name}; notes={classifier_notes}",
            )
            issues = (*issues, w.validate())
    overall_verdict = _combined_verdict(deterministic_verdict, suggestion_verdict)

    risk_summary = _build_risk_summary(issues, overall_verdict)

    plan = inputs.plan
    video_qa = inputs.video_qa
    target_ref = ""
    if plan is not None and getattr(plan, "production_plan_id", ""):
        target_ref = plan.production_plan_id
    elif video_qa is not None and getattr(video_qa, "target_ref", ""):
        target_ref = video_qa.target_ref
    if not target_ref:
        raise PolicyQAError("policy gate needs a production-plan target_ref")

    script = inputs.script
    topic_id = getattr(script, "topic_id", "") if script is not None else ""
    topic_title = (
        getattr(script, "topic_title", "") if script is not None else ""
    )
    if not topic_id and plan is not None:
        topic_id = getattr(plan, "topic_id", "") or topic_id
        topic_title = topic_title or getattr(plan, "topic_title", "") or topic_title
    script_id = getattr(script, "script_id", "") if script is not None else ""

    policy_result_id = policy_result_id_for(run_id, target_ref)

    provenance = Provenance(
        generated_by="policy_qa_agent_v1",
        source="DeterministicPolicyQA",
        created_at=created_at or _utc_now(),
        provider=None,
        model="PolicyQAValidator",
        prompt_version=config.policy_version,
    )

    result = PolicyQAResult(
        policy_result_id=policy_result_id,
        target_ref=target_ref,
        topic_id=topic_id,
        topic_title=topic_title,
        script_id=script_id,
        policy_version=config.policy_version,
        overall_verdict=overall_verdict,
        checked_at=created_at or _utc_now(),
        checks=checks,
        issues=tuple(i for i in issues if i.severity == "ERROR"),
        warnings=tuple(i for i in issues if i.severity == "WARNING"),
        infos=tuple(i for i in issues if i.severity == "INFO"),
        risk_summary=risk_summary,
        classifier_suggestion=suggestion_verdict or "",
        resumed_from_policy_result_id=resumed_from_policy_result_id,
        provenance=provenance,
        created_at=created_at or _utc_now(),
        run_id=run_id,
        produced_by={
            "prompt_version": config.policy_version,
            "provider": None,
            "model": "PolicyQAValidator",
        },
    )
    return result.validate()


def _build_risk_summary(issues: tuple[PolicyIssue, ...], verdict: str):
    error_count = sum(1 for i in issues if i.severity == "ERROR")
    warning_count = sum(1 for i in issues if i.severity == "WARNING")
    info_count = sum(1 for i in issues if i.severity == "INFO")
    flagged = tuple(
        dict.fromkeys(i.category for i in issues if i.severity == "ERROR")
    )
    if error_count:
        highest = "ERROR"
    elif warning_count:
        highest = "WARNING"
    elif info_count:
        highest = "INFO"
    else:
        highest = "none"
    return RiskSummary(
        verdict=verdict,
        error_count=error_count,
        warning_count=warning_count,
        info_count=info_count,
        flagged_categories=flagged,
        highest_risk_severity=highest,
    )