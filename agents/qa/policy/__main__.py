"""Manual runner for the Stage-9 policy/safety-QA stage (no external APIs).

Usage:
    python -m agents.qa.policy

    # explicit artifacts:
    python -m agents.qa.policy \
        --script data/<run_id>/scripts/script.json \
        --fact-check data/<run_id>/research/fact_check_result.json \
        --plan data/<run_id>/production/production.json \
        --video-result data/<run_id>/videos/video_generation_result.json \
        --qa-result data/<run_id>/production/qa_result.json

    # chain from a previous PolicyQAResult (re-review after a fix):
    python -m agents.qa.policy --previous data/<run_id>/production/policy_qa_result.json

Loads the Stage-4 Script, Stage-3 FactCheckResult, Stage-6 ProductionPlan,
Stage-7 VideoGenerationResult, and Stage-8 VideoQAResult, runs all 15
deterministic policy/safety checks, prints the verdict/issues/risk summary, and
writes a PolicyQAResult artifact (plus run state advanced to
Stage.QUALITY_CHECK) under data/<run_id>/production/policy_qa_result.json.

Stage note: Stage 9 is POLICY / SAFETY QA, a gate -- it does NOT implement
Stage 10 (human approval) or Stage 11 (YouTube upload/scheduling), and it is
NOT a YouTube compliance guarantee. The artifact-header `stage` label is
"policy_qa"; the pipeline STATE advances to the existing generic
Stage.QUALITY_CHECK (pipeline/state.py) -- the mapping is documented, and no
state enum member was added or renamed.

Exit codes: 0 = PASS; 1 = REVIEW or REJECT (artifact still written); 2 = error
(e.g. missing/corrupt inputs, mismatched plan/video-result pairing).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import uuid
from pathlib import Path

from agents.qa.policy.models import (
    PolicyQAConfig,
    PolicyQAError,
    PolicyQAResult,
)
from agents.qa.policy.stage import (
    DEFAULT_FACT_CHECK_PATH,
    DEFAULT_PRODUCTION_PLAN_PATH,
    DEFAULT_QA_RESULT_PATH,
    DEFAULT_SCRIPT_PATH,
    DEFAULT_VIDEO_RESULT_PATH,
    load_policy_qa_result,
    policy_inputs_from_paths,
    run_policy_qa,
)
from pipeline.state import PipelineState, Stage


def _fail(message: str) -> int:
    """Print a clean fatal error to stderr and return the input-error exit code."""
    print(f"[policy_qa] error: {message}", file=sys.stderr)
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agents.qa.policy",
        description=(
            "Run the deterministic Stage-9 policy/safety gate over the Script + "
            "FactCheckResult + ProductionPlan + VideoGenerationResult + "
            "VideoQAResult artifacts."
        ),
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=DEFAULT_SCRIPT_PATH,
        help="Stage-4 Script artifact JSON (default: data/mock/script.json)",
    )
    parser.add_argument(
        "--fact-check",
        type=Path,
        default=DEFAULT_FACT_CHECK_PATH,
        help="Stage-3 FactCheckResult JSON (default: data/mock/fact_check_result.json)",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=DEFAULT_PRODUCTION_PLAN_PATH,
        help="Stage-6 ProductionPlan JSON (default: data/mock/production.json)",
    )
    parser.add_argument(
        "--video-result",
        type=Path,
        default=DEFAULT_VIDEO_RESULT_PATH,
        help="Stage-7 VideoGenerationResult JSON "
        "(default: data/mock/video_generation_result.json)",
    )
    parser.add_argument(
        "--qa-result",
        type=Path,
        default=DEFAULT_QA_RESULT_PATH,
        help="Stage-8 VideoQAResult JSON (default: data/mock/qa_result.json)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="run id (default: 'pol-demo' for the mock fixtures, else the script's run id)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "output root under which "
            "<run_id>/production/policy_qa_result.json is written"
        ),
    )
    parser.add_argument(
        "--previous",
        type=Path,
        default=None,
        help=(
            "previous PolicyQAResult JSON to chain from (its policy_result_id is "
            "recorded as resumed_from_policy_result_id on the new artifact)"
        ),
    )
    parser.add_argument(
        "--policy-version",
        default=None,
        help="policy ruleset version (default: policy_rules_v1)",
    )
    args = parser.parse_args()

    for name, path in (
        ("Script", args.script),
        ("FactCheckResult", args.fact_check),
        ("ProductionPlan", args.plan),
        ("VideoGenerationResult", args.video_result),
    ):
        if not path.is_file():
            return _fail(f"{name} not found: {path}")

    previous = None
    if args.previous is not None:
        if not args.previous.is_file():
            return _fail(f"previous policy result not found: {args.previous}")
        try:
            previous = load_policy_qa_result(args.previous)
        except (AttributeError, ValueError, KeyError, TypeError, OSError) as exc:
            return _fail(f"could not load previous policy result {args.previous}: {exc}")

    config = PolicyQAConfig(policy_version=args.policy_version or "policy_rules_v1")

    if args.run_id:
        run_id = args.run_id
    elif (
        args.script == DEFAULT_SCRIPT_PATH
        and args.fact_check == DEFAULT_FACT_CHECK_PATH
        and args.plan == DEFAULT_PRODUCTION_PLAN_PATH
        and args.video_result == DEFAULT_VIDEO_RESULT_PATH
    ):
        # Default demo invocation: keep the committed data/mock fixtures clean.
        run_id = "pol-demo"
    else:
        run_id = uuid.uuid4().hex[:12]

    try:
        inputs = policy_inputs_from_paths(
            script_path=args.script,
            fact_check_path=args.fact_check,
            plan_path=args.plan,
            video_result_path=args.video_result,
            video_qa_path=args.qa_result,
        )
        policy_result = run_policy_qa(inputs, config=config, run_id=run_id)
    except PolicyQAError as exc:
        return _fail(str(exc))
    except (AttributeError, ValueError, KeyError, TypeError, OSError) as exc:
        return _fail(f"could not load input artifacts: {exc}")

    if previous is not None:
        policy_result = dataclasses.replace(
            policy_result, resumed_from_policy_result_id=previous.policy_result_id
        )

    out_root = args.out or Path("data")
    out_path = out_root / run_id / "production" / "policy_qa_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(policy_result.to_json(), encoding="utf-8")

    state_path = out_root / run_id / "state.json"
    try:
        state = PipelineState.load(state_path)
    except FileNotFoundError:
        state = PipelineState(run_id=run_id)
    # Numbered Stage 9 (Policy QA) maps to the existing generic quality-check
    # stage, exactly as Stage 8 does. No new Stage enum member exists and none
    # was added (see models.py).
    state.advance_to(Stage.QUALITY_CHECK)
    state.add_artifact(Stage.QUALITY_CHECK, out_path)
    state.save(state_path)

    _print_report(policy_result, chained=previous is not None)
    print(f"[policy_qa] artifact written to {out_path}")
    print(f"[policy_qa] run state written to {state_path}")
    return 0 if policy_result.overall_verdict == "PASS" else 1


def _print_report(policy_result: PolicyQAResult, chained: bool) -> None:
    print(f"[policy_qa] run_id={policy_result.run_id}")
    print(f"[policy_qa] policy_result_id={policy_result.policy_result_id}")
    print(f"[policy_qa] target_ref={policy_result.target_ref}")
    print(f"[policy_qa] policy_version={policy_result.policy_version}")
    print(f"[policy_qa] overall_verdict={policy_result.overall_verdict}")
    if chained:
        print(f"[policy_qa] resumed_from={policy_result.resumed_from_policy_result_id}")
    summary = policy_result.risk_summary
    if summary is not None:
        print(
            f"[policy_qa] risk: errors={summary.error_count} "
            f"warnings={summary.warning_count} infos={summary.info_count}; "
            f"flagged={summary.flagged_categories or '-'}"
        )
    for record in policy_result.checks:
        print(
            f"[policy_qa]   check {record.check_id}: outcome={record.outcome} "
            f"(errors={record.errors}, warnings={record.warnings}, infos={record.infos})"
        )
    for issue in policy_result.issues:
        print(
            f"[policy_qa]   issue {issue.issue_id} [{issue.check_id}]"
            f"[{issue.gate}] {issue.message}"
        )
    for warning in policy_result.warnings:
        print(f"[policy_qa]   warning {warning.issue_id} [{warning.check_id}] {warning.message}")
    for info in policy_result.infos:
        print(f"[policy_qa]   info {info.issue_id} [{info.check_id}] {info.message}")
    summary = policy_result.risk_summary
    if summary is not None:
        print(f"[policy_qa] reviewer note: {summary.reviewer_note}")


if __name__ == "__main__":
    raise SystemExit(main())