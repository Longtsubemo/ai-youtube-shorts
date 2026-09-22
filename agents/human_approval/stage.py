"""Stage-10 entrypoint: artifacts + human decision -> HumanApprovalResult.

QA-side only (docs/ARCHITECTURE.md): the stage loads the fully QA'd chain
(Stage-3 Script, Stage-5 FactCheckResult, Stage-5 ScriptQAResult, Stage-6
ProductionPlan, Stage-7 VideoGenerationResult, Stage-8 VideoQAResult, Stage-9
PolicyQAResult), verifies the reviewed bundle is coherent, and records the
human decision as a `HumanApprovalResult` under
`data/<run_id>/approval/human_approval_result.json`.

It never uploads, schedules, calls a provider, or touches YouTube. Stage 11
(YouTube upload) is out of scope for this stage and is not implemented here.
Approval here means "ready for a MANUAL upload" -- nothing is published
automatically.

Pairing guard: the reviewed bundle must all target the SAME ProductionPlan
(the policy target_ref, the Stage-8 target_ref, and the Stage-7
production_plan_id must agree, and the Stage-6 plan's script_id must match the
Stage-3 script). A mismatched bundle raises before any decision is recorded,
mirroring Stages 8/9.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.human_approval.models import (
    ApprovalInputs,
    Approver,
    HumanApprovalConfig,
    HumanApprovalError,
    HumanApprovalResult,
    OverriddenVerdict,
    RequestedChange,
)
from agents.human_approval.validator import evaluate_human_approval
from agents.qa.policy.stage import (
    load_policy_qa_result,
    load_video_generation_result,
    load_video_qa_result,
)
from agents.qa.script.models import ScriptQAResult
from agents.qa.script.stage import load_fact_check_result, load_script
from agents.video_generation.stage import load_production_plan

DEFAULT_SCRIPT_PATH = Path("data") / "mock" / "script.json"
DEFAULT_FACT_CHECK_PATH = Path("data") / "mock" / "fact_check_result.json"
DEFAULT_SCRIPT_QA_PATH = Path("data") / "mock" / "script_qa_result.json"
DEFAULT_PRODUCTION_PLAN_PATH = Path("data") / "mock" / "production.json"
DEFAULT_VIDEO_RESULT_PATH = Path("data") / "mock" / "video_generation_result.json"
DEFAULT_QA_RESULT_PATH = Path("data") / "mock" / "qa_result.json"
DEFAULT_POLICY_RESULT_PATH = Path("data") / "mock" / "policy_qa_result.json"
DEFAULT_APPROVAL_RESULT_NAME = "human_approval_result.json"


def load_script_qa_result(path: str | Path) -> ScriptQAResult:
    return ScriptQAResult.from_json(Path(path).read_text(encoding="utf-8"))


def load_approval_result(path: str | Path) -> HumanApprovalResult:
    return HumanApprovalResult.from_json(Path(path).read_text(encoding="utf-8"))


def build_approval_inputs(
    policy_qa=None,
    video_qa=None,
    video_result=None,
    plan=None,
    script=None,
    script_qa=None,
    fact_check=None,
    **_: Any,
) -> ApprovalInputs:
    """Assemble the typed review bundle the gate evaluates over."""
    return ApprovalInputs(
        policy_qa=policy_qa,
        video_qa=video_qa,
        video_result=video_result,
        plan=plan,
        script=script,
        script_qa=script_qa,
        fact_check=fact_check,
    )


def run_human_approval(
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
    """Record a human decision over `inputs` (guarded, fail closed)."""
    policy = inputs.policy_qa
    if policy is None:
        raise HumanApprovalError("policy QA result is required for approval")
    target_ref = policy.target_ref

    if inputs.plan is not None and inputs.plan.production_plan_id != target_ref:
        raise HumanApprovalError(
            f"production plan {inputs.plan.production_plan_id} does not match "
            f"policy target_ref {target_ref}"
        )
    if inputs.video_qa is not None and inputs.video_qa.target_ref != target_ref:
        raise HumanApprovalError(
            f"video QA {inputs.video_qa.qa_result_id} targets "
            f"{inputs.video_qa.target_ref}, not {target_ref}"
        )
    if (
        inputs.video_result is not None
        and inputs.video_result.production_plan_id != target_ref
    ):
        raise HumanApprovalError(
            f"video result {inputs.video_result.video_generation_result_id} "
            f"targets production plan {inputs.video_result.production_plan_id}, "
            f"not {target_ref}"
        )
    if (
        inputs.plan is not None
        and inputs.script is not None
        and inputs.plan.script_id != inputs.script.script_id
    ):
        raise HumanApprovalError(
            f"production plan {inputs.plan.production_plan_id} expects script "
            f"{inputs.plan.script_id}, not {inputs.script.script_id}"
        )

    return evaluate_human_approval(
        inputs,
        config=config,
        decision=decision,
        approver=approver,
        changes_requested=changes_requested,
        overridden_verdicts=overridden_verdicts,
        notes=notes,
        supersedes_approval_id=supersedes_approval_id,
        run_id=run_id,
        created_at=created_at,
    )


def load_committed_fixtures() -> ApprovalInputs:
    """Load the committed data/mock artifacts (incl. policy + script-QA)."""
    return build_approval_inputs(
        policy_qa=load_policy_qa_result(DEFAULT_POLICY_RESULT_PATH),
        video_qa=load_video_qa_result(DEFAULT_QA_RESULT_PATH),
        video_result=load_video_generation_result(DEFAULT_VIDEO_RESULT_PATH),
        plan=load_production_plan(DEFAULT_PRODUCTION_PLAN_PATH),
        script=load_script(DEFAULT_SCRIPT_PATH),
        script_qa=load_script_qa_result(DEFAULT_SCRIPT_QA_PATH),
        fact_check=load_fact_check_result(DEFAULT_FACT_CHECK_PATH),
    )


def approval_inputs_from_paths(
    policy_result_path: str | Path,
    video_qa_path: str | Path,
    video_result_path: str | Path,
    plan_path: str | Path,
    script_path: str | Path,
    script_qa_path: str | Path | None = None,
    fact_check_path: str | Path | None = None,
    **_: Any,
) -> ApprovalInputs:
    """Load a review bundle from explicit artifact JSON paths (for the CLI)."""
    script_qa = None
    if script_qa_path is not None:
        script_qa = load_script_qa_result(script_qa_path)
    fact_check = None
    if fact_check_path is not None:
        fact_check = load_fact_check_result(fact_check_path)
    return build_approval_inputs(
        policy_qa=load_policy_qa_result(policy_result_path),
        video_qa=load_video_qa_result(video_qa_path),
        video_result=load_video_generation_result(video_result_path),
        plan=load_production_plan(plan_path),
        script=load_script(script_path),
        script_qa=script_qa,
        fact_check=fact_check,
    )