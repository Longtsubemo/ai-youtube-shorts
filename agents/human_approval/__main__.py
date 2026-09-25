"""Manual runner for the Stage-10 human-approval gate (no external services).

Usage:
    python -m agents.human_approval \

    # record an APPROVED decision over the committed mock chain:
    python -m agents.human_approval \
        --decision APPROVED --approver-id alice --notes "ready for manual upload"

    # record CHANGES_REQUESTED (must include --notes or --change):
    python -m agents.human_approval \
        --decision CHANGES_REQUESTED --approver-id alice \
        --change "P13: add the synthetic-content disclosure"

    # explicit artifacts:
    python -m agents.human_approval \
        --policy-result data/<run_id>/production/policy_qa_result.json \
        --qa-result data/<run_id>/production/qa_result.json \
        --video-result data/<run_id>/videos/video_generation_result.json \
        --plan data/<run_id>/production/production.json \
        --script data/<run_id>/scripts/script.json \
        --script-qa data/<run_id>/production/script_qa_result.json \
        --fact-check data/<run_id>/research/fact_check_result.json \
        --decision APPROVED --approver-id alice

    # soft-REVIEW override (resolves a Stage-8/9 REVIEW with a recorded,
    # version-bound justification; key/value form "gate;check;verdict;why"):
    python -m agents.human_approval --decision APPROVED --approver-id alice \
        --override "video_qa;P15;REVIEW;accepted: disclosure warning reviewed on qa_9b7d098451d3"

    # audit: is an existing approval still effective against the CURRENT artifacts?
    python -m agents.human_approval --check \
        --approval data/<run_id>/approval/human_approval_result.json

Loads the full QA'd chain (Stage-4 Script, Stage-5 FactCheckResult +
ScriptQAResult, Stage-6 ProductionPlan, Stage-7 VideoGenerationResult,
Stage-8 VideoQAResult, Stage-9 PolicyQAResult), verifies the bundle is
coherent and complete, evaluates the deterministic hard gates for APPROVED,
and writes a HumanApprovalResult artifact under
data/<run_id>/approval/human_approval_result.json. Approved decisions advance
run bookkeeping to Stage.PUBLISH; non-approved decisions remain at
Stage.QUALITY_CHECK.

Stage note: Stage 10 is HUMAN APPROVAL, a gate -- it does NOT implement
Stage 11 (YouTube upload/scheduling). Approving marks the reviewed bundle as
ready for a MANUAL upload; nothing is ever uploaded or published by this stage.

Exit codes: 0 = decision recorded as APPROVED; 1 = CHANGES_REQUESTED or
REJECTED (artifact still written) or --check found the approval STALE /
not effective; 2 = error (missing/corrupt/incoherent inputs, missing or
invalid decision/approver, or an APPROVED attempt blocked by an unsatisfied
hard gate).
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from agents.human_approval.models import (
    Approver,
    HumanApprovalConfig,
    HumanApprovalError,
    HumanApprovalResult,
    OverriddenVerdict,
    RequestedChange,
    WRITABLE_DECISIONS,
    human_approval_id_for,
    validate_run_id,
)
from agents.human_approval.stage import (
    DEFAULT_APPROVAL_RESULT_NAME,
    DEFAULT_FACT_CHECK_PATH,
    DEFAULT_POLICY_RESULT_PATH,
    DEFAULT_PRODUCTION_PLAN_PATH,
    DEFAULT_QA_RESULT_PATH,
    DEFAULT_SCRIPT_PATH,
    DEFAULT_SCRIPT_QA_PATH,
    DEFAULT_VIDEO_RESULT_PATH,
    approval_inputs_from_paths,
    load_approval_result,
    run_human_approval,
)
from agents.human_approval.validator import (
    approval_is_effective,
    check_approval_freshness,
    validate_approval_inputs,
)
from pipeline.state import PipelineState, Stage


def _current_artifacts(inputs) -> dict:
    return {
        "policy_qa": inputs.policy_qa,
        "video_qa": inputs.video_qa,
        "video_result": inputs.video_result,
        "production_plan": inputs.plan,
        "script": inputs.script,
        "script_qa": inputs.script_qa,
        "fact_check": inputs.fact_check,
    }


def _parse_override(value: str) -> OverriddenVerdict:
    """Parse "gate;check;verdict;justification" (justification may hold ';')."""
    parts = value.split(";", maxsplit=3)
    if len(parts) != 4:
        raise ValueError(
            f"override must be 'gate;check;verdict;justification', got {value!r}"
        )
    gate, check_id, verdict, justification = parts
    return OverriddenVerdict(
        gate=gate,
        check_id=check_id,
        verdict=verdict,
        decision="APPROVED",
        justification=justification,
    )


def _optional_path(value: Path | None) -> Path | None:
    if value is None or str(value).lower() in ("", "none"):
        return None
    return value


def _load_cli_inputs(args) -> tuple[object, str]:
    inputs = approval_inputs_from_paths(
        policy_result_path=args.policy_result,
        video_qa_path=args.qa_result,
        video_result_path=args.video_result,
        plan_path=args.plan,
        script_path=args.script,
        script_qa_path=_optional_path(args.script_qa),
        fact_check_path=_optional_path(args.fact_check),
    )
    source_run_id = validate_approval_inputs(inputs)
    return inputs, source_run_id


def _temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(descriptor)
    return Path(name)


def _restore_file(path: Path, original: bytes | None) -> None:
    if original is None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return
    temporary = _temporary_path(path)
    try:
        temporary.write_bytes(original)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_pair(
    artifact_path: Path,
    artifact_text: str,
    state_path: Path,
    state_text: str,
) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    old_artifact = artifact_path.read_bytes() if artifact_path.exists() else None
    old_state = state_path.read_bytes() if state_path.exists() else None
    artifact_tmp: Path | None = None
    state_tmp: Path | None = None
    try:
        artifact_tmp = _temporary_path(artifact_path)
        state_tmp = _temporary_path(state_path)
        artifact_tmp.write_text(artifact_text, encoding="utf-8")
        state_tmp.write_text(state_text, encoding="utf-8")
        os.replace(artifact_tmp, artifact_path)
        os.replace(state_tmp, state_path)
    except OSError:
        for temporary in (artifact_tmp, state_tmp):
            if temporary is not None and temporary.exists():
                temporary.unlink()
        _restore_file(artifact_path, old_artifact)
        _restore_file(state_path, old_state)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agents.human_approval",
        description=(
            "Record a human approval decision over the exact QA'd artifact "
            "versions (fail closed; no written record = no approval)."
        ),
    )
    parser.add_argument(
        "--policy-result",
        type=Path,
        default=DEFAULT_POLICY_RESULT_PATH,
        help="Stage-9 PolicyQAResult JSON (default: data/mock/policy_qa_result.json)",
    )
    parser.add_argument(
        "--qa-result",
        type=Path,
        default=DEFAULT_QA_RESULT_PATH,
        help="Stage-8 VideoQAResult JSON (default: data/mock/qa_result.json)",
    )
    parser.add_argument(
        "--video-result",
        type=Path,
        default=DEFAULT_VIDEO_RESULT_PATH,
        help="Stage-7 VideoGenerationResult JSON "
        "(default: data/mock/video_generation_result.json)",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=DEFAULT_PRODUCTION_PLAN_PATH,
        help="Stage-6 ProductionPlan JSON (default: data/mock/production.json)",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=DEFAULT_SCRIPT_PATH,
        help="Stage-4 Script JSON (default: data/mock/script.json)",
    )
    parser.add_argument(
        "--script-qa",
        type=Path,
        default=DEFAULT_SCRIPT_QA_PATH,
        help=(
            "Stage-5 ScriptQAResult JSON (default: data/mock/script_qa_result.json); "
            "pass 'none' to omit it from the reviewed set"
        ),
    )
    parser.add_argument(
        "--fact-check",
        type=Path,
        default=DEFAULT_FACT_CHECK_PATH,
        help=(
            "Stage-5 FactCheckResult JSON "
            "(default: data/mock/fact_check_result.json); pass 'none' to omit it"
        ),
    )
    parser.add_argument(
        "--decision",
        default=None,
        help="human decision: APPROVED | CHANGES_REQUESTED | REJECTED",
    )
    parser.add_argument("--approver-id", default=None, help="who is deciding (required)")
    parser.add_argument("--approver-role", default="", help="approver context role")
    parser.add_argument("--notes", default="", help="human notes for the decision")
    parser.add_argument(
        "--change",
        action="append",
        default=[],
        dest="changes",
        help="a requested change (repeatable) for CHANGES_REQUESTED",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        dest="overrides",
        help=(
            "recorded soft-REVIEW override, form "
            "'gate;check;verdict;justification' (repeatable)"
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "approval run id (default: the common run_id in the input bundle); "
            "must be path-safe"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "output root under which <run_id>/approval/"
            "human_approval_result.json is written"
        ),
    )
    parser.add_argument(
        "--previous",
        type=Path,
        default=None,
        help=(
            "a previous HumanApprovalResult JSON being re-reviewed; its approval_id "
            "is recorded as supersedes_approval_id on the new record"
        ),
    )
    parser.add_argument(
        "--approval",
        type=Path,
        default=None,
        help="an existing HumanApprovalResult JSON to audit (requires --check)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "audit mode: verify an existing approval is still effective against "
            "the current artifacts (digest + id freshness); writes nothing"
        ),
    )
    args = parser.parse_args()

    if args.run_id is not None:
        try:
            validate_run_id(args.run_id)
        except HumanApprovalError as exc:
            print(f"[human_approval] error: {exc}")
            return 2

    core_paths = (
        ("PolicyQAResult", args.policy_result),
        ("VideoQAResult", args.qa_result),
        ("VideoGenerationResult", args.video_result),
        ("ProductionPlan", args.plan),
        ("Script", args.script),
    )
    for name, path in core_paths:
        if not path.is_file():
            print(f"[human_approval] {name} not found: {path}")
            return 2

    script_qa_path = _optional_path(args.script_qa)
    fact_check_path = _optional_path(args.fact_check)
    if script_qa_path is not None and not script_qa_path.is_file():
        print(f"[human_approval] ScriptQAResult not found: {script_qa_path}")
        return 2
    if fact_check_path is not None and not fact_check_path.is_file():
        print(f"[human_approval] FactCheckResult not found: {fact_check_path}")
        return 2

    if args.check:
        if args.approval is None:
            print("[human_approval] --check requires --approval <path>")
            return 2
        if any(
            (
                args.decision is not None,
                args.approver_id is not None,
                bool(args.changes),
                bool(args.overrides),
                args.previous is not None,
            )
        ):
            print("[human_approval] --check cannot be combined with decision options")
            return 2
        if not args.approval.is_file():
            print(f"[human_approval] approval not found: {args.approval}")
            return 2
        try:
            approval = load_approval_result(args.approval)
            if args.run_id is not None and approval.run_id != args.run_id:
                raise HumanApprovalError(
                    f"approval run_id {approval.run_id!r} does not match --run-id "
                    f"{args.run_id!r}"
                )
            inputs, _ = _load_cli_inputs(args)
            current_plan = inputs.plan
            if current_plan is None:
                raise HumanApprovalError("current production plan is required for --check")
            if approval.target_ref != current_plan.production_plan_id:
                raise HumanApprovalError(
                    f"approval target {approval.target_ref} does not match current "
                    f"plan {current_plan.production_plan_id}"
                )
            current = _current_artifacts(inputs)
            fresh = check_approval_freshness(approval, current)
            effective = approval_is_effective(approval, current)
        except (HumanApprovalError, OSError, TypeError, ValueError, KeyError, AttributeError) as exc:
            print(f"[human_approval] error: {exc}")
            return 2
        print(f"[human_approval] approval_id={approval.approval_id}")
        print(f"[human_approval] decision={approval.decision}")
        print(f"[human_approval] reviewed_artifacts={len(approval.reviewed_artifacts)}")
        print(f"[human_approval] digests_fresh={'yes' if fresh else 'NO'}")
        print(f"[human_approval] effective={'yes' if effective else 'NO'}")
        return 0 if effective else 1

    if args.approval is not None:
        print("[human_approval] --approval is only valid with --check")
        return 2
    if args.decision is None:
        print(
            "[human_approval] --decision is required "
            f"({', '.join(WRITABLE_DECISIONS)})"
        )
        return 2
    if args.decision not in WRITABLE_DECISIONS:
        print(
            f"[human_approval] decision {args.decision!r} is not writable "
            f"(allowed: {', '.join(WRITABLE_DECISIONS)})"
        )
        return 2
    if not args.approver_id or not args.approver_id.strip():
        print("[human_approval] --approver-id is required")
        return 2

    try:
        overrides = tuple(_parse_override(value) for value in args.overrides)
        changes = tuple(
            RequestedChange(requirement=value).validate() for value in args.changes
        )
        inputs, source_run_id = _load_cli_inputs(args)
        run_id = validate_run_id(args.run_id or source_run_id)
        previous = None
        if args.previous is not None:
            if not args.previous.is_file():
                raise HumanApprovalError(
                    f"previous approval not found: {args.previous}"
                )
            previous = load_approval_result(args.previous)
            current_plan = inputs.plan
            if current_plan is None:
                raise HumanApprovalError("current production plan is required for --previous")
            if previous.target_ref != current_plan.production_plan_id:
                raise HumanApprovalError(
                    f"previous approval target {previous.target_ref} does not match "
                    f"current plan {current_plan.production_plan_id}"
                )
            if previous.approval_id == human_approval_id_for(
                run_id, current_plan.production_plan_id
            ):
                raise HumanApprovalError(
                    "re-review of the same approval slot must omit --previous; "
                    "an approval cannot supersede itself"
                )
        approval = run_human_approval(
            inputs,
            config=HumanApprovalConfig(),
            decision=args.decision,
            approver=Approver(
                approver_id=args.approver_id, role=args.approver_role or ""
            ),
            changes_requested=changes,
            overridden_verdicts=overrides,
            notes=args.notes,
            supersedes_approval_id=previous.approval_id if previous is not None else "",
            run_id=run_id,
        )
    except (HumanApprovalError, OSError, TypeError, ValueError, KeyError, AttributeError) as exc:
        print(f"[human_approval] error: {exc}")
        return 2

    try:
        out_root = args.out or Path("data")
        out_path = out_root / run_id / "approval" / DEFAULT_APPROVAL_RESULT_NAME
        state_path = out_root / run_id / "state.json"
        try:
            state = PipelineState.load(state_path)
        except FileNotFoundError:
            state = PipelineState(run_id=run_id)
        if state.run_id != run_id:
            raise HumanApprovalError(
                f"state run_id {state.run_id!r} does not match approval run {run_id!r}"
            )
        registry_stage = Stage.PUBLISH if approval.is_approved else Stage.QUALITY_CHECK
        state.advance_to(registry_stage)
        state.artifacts.setdefault(registry_stage.value, [])
        artifact_ref = str(out_path)
        if artifact_ref not in state.artifacts[registry_stage.value]:
            state.artifacts[registry_stage.value].append(artifact_ref)
        _atomic_write_pair(
            out_path,
            approval.to_json(),
            state_path,
            state.to_json(),
        )
    except (HumanApprovalError, OSError, TypeError, ValueError, KeyError, AttributeError) as exc:
        print(f"[human_approval] error: {exc}")
        return 2

    _print_report(approval, chained=previous is not None)
    print(f"[human_approval] artifact written to {out_path}")
    print(f"[human_approval] run state written to {state_path}")
    return 0 if approval.decision == "APPROVED" else 1


def _print_report(approval: HumanApprovalResult, chained: bool) -> None:
    print(f"[human_approval] run_id={approval.run_id}")
    print(f"[human_approval] approval_id={approval.approval_id}")
    print(f"[human_approval] target_ref={approval.target_ref}")
    print(f"[human_approval] decision={approval.decision}")
    print(
        f"[human_approval] approver={approval.approver.approver_id}"
        + (f" ({approval.approver.role})" if approval.approver.role else "")
    )
    if chained:
        print(
            f"[human_approval] supersedes={approval.supersedes_approval_id}"
        )
    for condition in approval.blocking_conditions:
        print(
            f"[human_approval]   gate {condition.code}: "
            f"{'OK' if condition.satisfied else 'BLOCKED'} {condition.detail}"
        )
    print(
        f"[human_approval] reviewed_artifacts={len(approval.reviewed_artifacts)} "
        f"(id + content_sha256 bound)"
    )
    for artifact in approval.reviewed_artifacts:
        print(
            f"[human_approval]   {artifact.role}: {artifact.artifact_id} "
            f"@ {artifact.content_sha256[:12]}..."
        )
    for override in approval.overridden_verdicts:
        print(
            f"[human_approval]   override {override.gate}"
            + (f"/{override.check_id}" if override.check_id else "")
            + f" {override.verdict} -> {override.decision} on {override.artifact_id}"
        )
    manual = approval.manual_upload
    if manual is not None:
        print(
            f"[human_approval] manual_upload.ready={manual.ready} "
            f"(artifacts in package: {len(manual.artifacts)})"
        )
    for change in approval.changes_requested:
        ref = f" [{change.issue_ref}]" if change.issue_ref else ""
        print(f"[human_approval]   requested change{ref}: {change.requirement}")
    if approval.notes:
        print(f"[human_approval] notes: {approval.notes}")


if __name__ == "__main__":
    raise SystemExit(main())