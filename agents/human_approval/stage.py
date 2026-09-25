"""Stage-10 entrypoint: artifacts + human decision -> HumanApprovalResult.

QA-side only (docs/ARCHITECTURE.md): the stage loads the fully QA'd chain
(Stage-4 Script, Stage-5 FactCheckResult, Stage-5 ScriptQAResult, Stage-6
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
Stage-4 script). A mismatched bundle raises before any decision is recorded,
mirroring Stages 8/9.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from agents.human_approval.models import (
    SCHEMA as APPROVAL_SCHEMA,
    STAGE as APPROVAL_STAGE,
    ApprovalInputs,
    Approver,
    HumanApprovalConfig,
    HumanApprovalError,
    HumanApprovalResult,
    OverriddenVerdict,
    RequestedChange,
    validate_run_id,
)
from agents.human_approval.validator import evaluate_human_approval
from agents.qa.policy.models import PolicyQAResult
from agents.qa.script.models import ScriptQAResult
from agents.qa.video.models import VideoQAResult
from agents.scene_planner.models import ProductionPlan
from agents.video_generation.models import VideoGenerationResult

DEFAULT_SCRIPT_PATH = Path("data") / "mock" / "script.json"
DEFAULT_FACT_CHECK_PATH = Path("data") / "mock" / "fact_check_result.json"
DEFAULT_SCRIPT_QA_PATH = Path("data") / "mock" / "script_qa_result.json"
DEFAULT_PRODUCTION_PLAN_PATH = Path("data") / "mock" / "production.json"
DEFAULT_VIDEO_RESULT_PATH = Path("data") / "mock" / "video_generation_result.json"
DEFAULT_QA_RESULT_PATH = Path("data") / "mock" / "qa_result.json"
DEFAULT_POLICY_RESULT_PATH = Path("data") / "mock" / "policy_qa_result.json"
DEFAULT_APPROVAL_RESULT_NAME = "human_approval_result.json"


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str):
    raise ValueError(f"non-finite JSON constant: {value}")


def _load_envelope(
    path: str | Path,
    *,
    label: str,
    schema: str,
    stage: str,
    parser: Callable[[dict], Any],
) -> Any:
    artifact_path = Path(path)
    try:
        text = artifact_path.read_text(encoding="utf-8")
        data = json.loads(
            text,
            object_pairs_hook=_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise HumanApprovalError(f"cannot load {label} {artifact_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise HumanApprovalError(f"{label} must be a JSON object: {artifact_path}")
    if data.get("schema") != schema:
        raise HumanApprovalError(
            f"{label} schema must be {schema!r}, got {data.get('schema')!r}"
        )
    if data.get("stage") != stage:
        raise HumanApprovalError(
            f"{label} stage must be {stage!r}, got {data.get('stage')!r}"
        )
    if data.get("schema_version") != 1 or isinstance(data.get("schema_version"), bool):
        raise HumanApprovalError(
            f"{label} schema_version must be 1, got {data.get('schema_version')!r}"
        )
    if not isinstance(data.get("created_at"), str) or not data["created_at"].strip():
        raise HumanApprovalError(f"{label} created_at must be an ISO timestamp")
    if not isinstance(data.get("produced_by"), dict):
        raise HumanApprovalError(f"{label} produced_by must be an object")
    if not isinstance(data.get("data"), dict):
        raise HumanApprovalError(f"{label} data must be an object")
    run_id = data.get("run_id")
    validate_run_id(run_id)
    try:
        artifact = parser(data)
    except HumanApprovalError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise HumanApprovalError(f"malformed {label} {artifact_path}: {exc}") from exc
    if getattr(artifact, "run_id", None) != run_id:
        raise HumanApprovalError(
            f"{label} run_id changed while loading {artifact_path}"
        )
    return artifact


def load_policy_result(path: str | Path) -> PolicyQAResult:
    return _load_envelope(
        path,
        label="PolicyQAResult",
        schema="PolicyQAResult",
        stage="policy_qa",
        parser=PolicyQAResult.from_dict,
    )


def load_video_qa(path: str | Path) -> VideoQAResult:
    return _load_envelope(
        path,
        label="VideoQAResult",
        schema="VideoQAResult",
        stage="video",
        parser=VideoQAResult.from_dict,
    )


def load_video_result(path: str | Path) -> VideoGenerationResult:
    return _load_envelope(
        path,
        label="VideoGenerationResult",
        schema="VideoGenerationResult",
        stage="video_generation",
        parser=VideoGenerationResult.from_dict,
    )


def load_plan(path: str | Path) -> ProductionPlan:
    return _load_envelope(
        path,
        label="ProductionPlan",
        schema="ProductionPlan",
        stage="scene_planning",
        parser=ProductionPlan.from_dict,
    )


def load_script_artifact(path: str | Path):
    from agents.script.models import Script

    return _load_envelope(
        path,
        label="Script",
        schema="Script",
        stage="script_generation",
        parser=Script.from_dict,
    )


def load_fact_check(path: str | Path):
    from agents.qa.fact_checker.models import FactCheckResult

    return _load_envelope(
        path,
        label="FactCheckResult",
        schema="FactCheckResult",
        stage="fact_check",
        parser=FactCheckResult.from_dict,
    )


def load_script_qa_result(path: str | Path) -> ScriptQAResult:
    return _load_envelope(
        path,
        label="ScriptQAResult",
        schema="ScriptQAResult",
        stage="script_qa",
        parser=ScriptQAResult.from_dict,
    )


def load_approval_result(path: str | Path) -> HumanApprovalResult:
    return _load_envelope(
        path,
        label="HumanApprovalResult",
        schema=APPROVAL_SCHEMA,
        stage=APPROVAL_STAGE,
        parser=HumanApprovalResult.from_dict,
    )


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
    if not isinstance(inputs, ApprovalInputs):
        raise HumanApprovalError("inputs must be ApprovalInputs")
    if inputs.policy_qa is None:
        raise HumanApprovalError("policy QA result is required for approval")
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
        policy_qa=load_policy_result(DEFAULT_POLICY_RESULT_PATH),
        video_qa=load_video_qa(DEFAULT_QA_RESULT_PATH),
        video_result=load_video_result(DEFAULT_VIDEO_RESULT_PATH),
        plan=load_plan(DEFAULT_PRODUCTION_PLAN_PATH),
        script=load_script_artifact(DEFAULT_SCRIPT_PATH),
        script_qa=load_script_qa_result(DEFAULT_SCRIPT_QA_PATH),
        fact_check=load_fact_check(DEFAULT_FACT_CHECK_PATH),
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
    if script_qa_path is not None and str(script_qa_path).lower() != "none":
        script_qa = load_script_qa_result(script_qa_path)
    fact_check = None
    if fact_check_path is not None and str(fact_check_path).lower() != "none":
        fact_check = load_fact_check(fact_check_path)
    return build_approval_inputs(
        policy_qa=load_policy_result(policy_result_path),
        video_qa=load_video_qa(video_qa_path),
        video_result=load_video_result(video_result_path),
        plan=load_plan(plan_path),
        script=load_script_artifact(script_path),
        script_qa=script_qa,
        fact_check=fact_check,
    )
