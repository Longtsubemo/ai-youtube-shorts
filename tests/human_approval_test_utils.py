"""Shared builders for the Stage-10 (human approval) test suite.

Everything here is MOCK, built deterministically from the committed
`data/mock/*` fixtures (the same rule as the Stage-3..9 builders): no fake
claims, no real-world evidence, no external services. The committed fixtures
evaluate CLEAN: the PolicyQAResult (data/mock/policy_qa_result.json) is a
deterministic mock Stage-9 PASS over the same bundle, so a healthy approval
over `committed_inputs()` clears every hard gate. Each scenario below
introduces exactly one defect on top of that baseline with
`dataclasses.replace`, isolating the rule under test.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agents.human_approval.models import (
    ApprovalInputs,
    Approver,
    HumanApprovalConfig,
    HumanApprovalResult,
    OverriddenVerdict,
    RequestedChange,
)
from agents.qa.policy.models import PolicyIssue
from agents.qa.video.models import VideoQAIssue
from agents.human_approval.validator import evaluate_human_approval

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MOCK = REPO_ROOT / "data" / "mock"

#: Fixed genesis timestamp shared by the committed data/mock fixtures.
GENESIS = "2026-09-20T12:00:00.000000+00:00"

__all__ = [
    "REPO_ROOT",
    "MOCK",
    "GENESIS",
    "committed_inputs",
    "build_inputs",
    "build_config",
    "healthy_approval",
    "artifact_map",
    "policy_verdict",
    "video_qa_verdict",
    "video_result_status",
    "script_qa_verdict",
    "tampered_video_result",
    "wrong_id_video_result",
]


def committed_inputs() -> ApprovalInputs:
    """The committed data/mock fixtures (clean baseline for an APPROVED).

    Loaded from absolute paths so it works regardless of cwd (e.g. under
    ``monkeypatch.chdir`` in purity tests).
    """
    from agents.human_approval.stage import (
        build_approval_inputs,
        load_script_qa_result,
    )
    from agents.qa.policy.stage import (
        load_policy_qa_result,
        load_video_generation_result,
        load_video_qa_result,
    )
    from agents.qa.script.stage import load_fact_check_result, load_script
    from agents.video_generation.stage import load_production_plan

    return build_approval_inputs(
        policy_qa=load_policy_qa_result(MOCK / "policy_qa_result.json"),
        video_qa=load_video_qa_result(MOCK / "qa_result.json"),
        video_result=load_video_generation_result(MOCK / "video_generation_result.json"),
        plan=load_production_plan(MOCK / "production.json"),
        script=load_script(MOCK / "script.json"),
        script_qa=load_script_qa_result(MOCK / "script_qa_result.json"),
        fact_check=load_fact_check_result(MOCK / "fact_check_result.json"),
    )


def build_inputs(**changes) -> ApprovalInputs:
    """Committed inputs with selected artifacts swapped out (None = missing)."""
    import dataclasses

    return dataclasses.replace(committed_inputs(), **changes)


def build_config(**changes) -> HumanApprovalConfig:
    import dataclasses

    return dataclasses.replace(HumanApprovalConfig(), **changes)


def artifact_map(inputs: ApprovalInputs) -> dict:
    """role -> artifact object, matching the CLI's current-artifact map."""
    return {
        "policy_qa": inputs.policy_qa,
        "video_qa": inputs.video_qa,
        "video_result": inputs.video_result,
        "production_plan": inputs.plan,
        "script": inputs.script,
        "script_qa": inputs.script_qa,
        "fact_check": inputs.fact_check,
    }


def _replace(obj, **changes):
    import dataclasses

    return dataclasses.replace(obj, **changes)


def policy_verdict(verdict: str):
    result = committed_inputs().policy_qa
    if verdict == result.overall_verdict:
        return result
    if verdict not in ("PASS", "REVIEW", "REJECT"):
        raise ValueError(f"unsupported policy verdict: {verdict}")
    check_id = "P15"
    check = next(record for record in result.checks if record.check_id == check_id)
    gate = "hard" if verdict == "REJECT" else "soft"
    issue = PolicyIssue(
        issue_id=f"test-policy-{verdict.lower()}",
        check_id=check_id,
        category=check.category,
        severity="ERROR",
        gate=gate,
        message=f"deterministic {verdict.lower()} test finding",
        rationale="Synthetic test defect for Stage-10 gate coverage",
        policy_rule="test-only",
    ).validate()
    checks = tuple(
        _replace(
            record,
            errors=record.errors + 1,
            hard_errors=record.hard_errors + (1 if gate == "hard" else 0),
        )
        if record.check_id == check_id
        else record
        for record in result.checks
    )
    risk_summary = result.risk_summary
    if risk_summary is not None:
        risk_summary = _replace(
            risk_summary,
            verdict=verdict,
            error_count=risk_summary.error_count + 1,
            highest_risk_severity="ERROR",
            flagged_categories=tuple(
                dict.fromkeys((*risk_summary.flagged_categories, check.category))
            ),
        )
    return _replace(
        result,
        overall_verdict=verdict,
        checks=checks,
        issues=(*result.issues, issue),
        risk_summary=risk_summary,
    )


def video_qa_verdict(verdict: str):
    result = committed_inputs().video_qa
    if verdict == result.overall_verdict:
        return result
    if verdict not in ("PASS", "REVIEW", "REJECT"):
        raise ValueError(f"unsupported video QA verdict: {verdict}")
    check_id = "provider_isolation"
    check = next(record for record in result.checks if record.check == check_id)
    gate = "hard" if verdict == "REJECT" else "soft"
    issue = VideoQAIssue(
        issue_id=f"test-video-{verdict.lower()}",
        category=check.category,
        severity="ERROR",
        message=f"deterministic {verdict.lower()} test finding",
        gate=gate,
        check=check_id,
    ).validate()
    checks = tuple(
        _replace(
            record,
            errors=record.errors + 1,
            hard_errors=record.hard_errors + (1 if gate == "hard" else 0),
        )
        if record.check == check_id
        else record
        for record in result.checks
    )
    return _replace(
        result,
        overall_verdict=verdict,
        checks=checks,
        issues=(*result.issues, issue),
    )


def video_result_status(status: str):
    result = committed_inputs().video_result
    if status == "failed":
        rows = tuple(
            _replace(
                row,
                status="failed",
                artifact=None,
                failure="deterministic failed test result",
                failure_retryable=True,
            )
            for row in result.scenes
        )
        return _replace(result, scenes=rows, overall_status=status)
    if status == "pending":
        rows = list(result.scenes)
        rows[0] = _replace(
            rows[0],
            status="pending",
            artifact=None,
            failure="",
            failure_retryable=None,
        )
        return _replace(result, scenes=tuple(rows), overall_status=status)
    if status == "completed":
        return result
    raise ValueError(f"unsupported video result status: {status}")


def script_qa_verdict(verdict: str):
    return _replace(committed_inputs().script_qa, overall_verdict=verdict)


def tampered_video_result(same_id: bool = True):
    """Same stable id, different reviewed content (for digest staleness)."""
    result = committed_inputs().video_result
    rows = list(result.scenes)
    rows[0] = _replace(
        rows[0],
        artifact=_replace(rows[0].artifact, note="re-rendered for the re-review"),
    )
    return _replace(result, scenes=tuple(rows))


def wrong_id_video_result():
    """Same content, different stable id (for id staleness)."""
    return _replace(committed_inputs().video_result, video_generation_result_id="vg_other")


def healthy_approval(
    *,
    decision: str = "APPROVED",
    approver_id: str = "reviewer-01",
    run_id: str = "mock",
    created_at: str = GENESIS,
    notes: str = "",
    changes: tuple[RequestedChange, ...] = (),
    overrides: tuple[OverriddenVerdict, ...] = (),
    supersedes_approval_id: str = "",
    config: HumanApprovalConfig | None = None,
    inputs: ApprovalInputs | None = None,
    **input_changes,
) -> HumanApprovalResult:
    """A Stage-10 record over the committed fixtures (self-consistent)."""
    inputs = inputs or build_inputs(**input_changes)
    return evaluate_human_approval(
        inputs,
        config=config or HumanApprovalConfig(),
        decision=decision,
        approver=Approver(approver_id=approver_id, role="test-reviewer"),
        changes_requested=changes,
        overridden_verdicts=overrides,
        notes=notes,
        supersedes_approval_id=supersedes_approval_id,
        run_id=run_id,
        created_at=created_at,
    )


def policy_result_dict(**changes) -> dict:
    """Serializable Stage-9 PolicyQAResult for CLI scenario tests (clean)."""
    changes = dict(changes)
    verdict = changes.pop("overall_verdict", None)
    policy = policy_verdict(verdict) if verdict is not None else committed_inputs().policy_qa
    if changes:
        policy = _replace(policy, **changes)
    return policy.to_dict()


def qa_result_dict(**changes) -> dict:
    """Serializable Stage-8 VideoQAResult for CLI scenario tests (clean)."""
    changes = dict(changes)
    verdict = changes.pop("overall_verdict", None)
    qa = video_qa_verdict(verdict) if verdict is not None else committed_inputs().video_qa
    if changes:
        qa = _replace(qa, **changes)
    return qa.to_dict()
