"""Manual runner for the Stage-8 video-QA stage (no external APIs).

Usage:
    python -m agents.qa.video

    # explicit artifacts:
    python -m agents.qa.video \
        --result data/<run_id>/videos/video_generation_result.json \
        --plan data/<run_id>/production/production.json

    # chain from a previous VideoQAResult (re-QA after an assembly fix):
    python -m agents.qa.video --previous data/<run_id>/production/qa_result.json

Loads a Stage-7 VideoGenerationResult + its Stage-6 ProductionPlan, runs all
15 deterministic video-QA checks, prints the verdict/issues, and writes a
VideoQAResult artifact (plus run state advanced to Stage.QUALITY_CHECK) under
data/<run_id>/production/qa_result.json.

Stage note: Stage 8 is VIDEO QA. It is NOT video assembly/editing: no media is
assembled, uploaded, or published here. The artifact-header `stage` label is
"video" (DATA_CONTRACTS.md §11); the pipeline STATE advances to the existing
generic Stage.QUALITY_CHECK (pipeline/state.py) -- the mapping is documented,
and no state enum member was added or renamed.

Exit codes: 0 = PASS; 1 = REVIEW or REJECT (artifact still written); 2 = error
(e.g. missing/corrupt inputs, mismatched plan/result pairing).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import uuid
from pathlib import Path

from agents.qa.video.models import VideoQAConfig, VideoQAError, VideoQAResult
from agents.qa.video.stage import (
    DEFAULT_PRODUCTION_PLAN_PATH,
    DEFAULT_VIDEO_RESULT_PATH,
    load_video_generation_result,
    load_video_qa_result,
    run_video_qa,
)
from agents.video_generation.stage import load_production_plan
from pipeline.state import PipelineState, Stage


def _fail(message: str) -> int:
    """Print a clean fatal error to stderr and return the input-error exit code."""
    print(f"[video_qa] error: {message}", file=sys.stderr)
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agents.qa.video",
        description=(
            "Run the deterministic Stage-8 video-QA gate over a Stage-7 "
            "VideoGenerationResult + its Stage-6 ProductionPlan."
        ),
    )
    parser.add_argument(
        "--result",
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
        "--run-id",
        default=None,
        help="run id (default: 'vqa-demo' for the mock fixtures, else the result's run id)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output root under which <run_id>/production/qa_result.json is written",
    )
    parser.add_argument(
        "--previous",
        type=Path,
        default=None,
        help=(
            "previous VideoQAResult JSON to chain from (its qa_result_id is "
            "recorded as resumed_from_qa_result_id on the new artifact)"
        ),
    )
    args = parser.parse_args()

    if not args.result.is_file():
        return _fail(f"VideoGenerationResult not found: {args.result}")
    if not args.plan.is_file():
        return _fail(f"ProductionPlan not found: {args.plan}")
    try:
        plan = load_production_plan(args.plan)
    except (AttributeError, ValueError, KeyError, TypeError, OSError) as exc:
        return _fail(f"could not load ProductionPlan {args.plan}: {exc}")
    try:
        result = load_video_generation_result(args.result)
    except (AttributeError, ValueError, KeyError, TypeError, OSError) as exc:
        return _fail(f"could not load VideoGenerationResult {args.result}: {exc}")

    previous = None
    if args.previous is not None:
        if not args.previous.is_file():
            return _fail(f"previous QA result not found: {args.previous}")
        try:
            previous = load_video_qa_result(args.previous)
        except (AttributeError, ValueError, KeyError, TypeError, OSError) as exc:
            return _fail(f"could not load previous QA result {args.previous}: {exc}")

    if args.run_id:
        run_id = args.run_id
    elif args.result == DEFAULT_VIDEO_RESULT_PATH and args.plan == DEFAULT_PRODUCTION_PLAN_PATH:
        # Default demo invocation: keep the committed data/mock fixtures clean.
        run_id = "vqa-demo"
    else:
        run_id = result.run_id or plan.run_id or uuid.uuid4().hex[:12]

    try:
        qa_result = run_video_qa(plan, result, config=VideoQAConfig(), run_id=run_id)
    except VideoQAError as exc:
        print(f"[video_qa] error: {exc}")
        return 2

    if previous is not None:
        qa_result = dataclasses.replace(qa_result, resumed_from_qa_result_id=previous.qa_result_id)

    out_root = args.out or Path("data")
    out_path = out_root / run_id / "production" / "qa_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(qa_result.to_json(), encoding="utf-8")

    state_path = out_root / run_id / "state.json"
    try:
        state = PipelineState.load(state_path)
    except FileNotFoundError:
        state = PipelineState(run_id=run_id)
    # Numbered Stage 8 (Video QA) maps to the existing generic quality-check
    # stage. No new Stage enum member exists and none was added (see models.py).
    state.advance_to(Stage.QUALITY_CHECK)
    state.add_artifact(Stage.QUALITY_CHECK, out_path)
    state.save(state_path)

    _print_report(qa_result, result, previous is not None)
    print(f"[video_qa] artifact written to {out_path}")
    print(f"[video_qa] run state written to {state_path}")
    return 0 if qa_result.overall_verdict == "PASS" else 1


def _print_report(qa_result: VideoQAResult, result, chained: bool) -> None:
    print(f"[video_qa] run_id={qa_result.run_id}")
    print(f"[video_qa] qa_result_id={qa_result.qa_result_id}")
    print(f"[video_qa] target_ref={qa_result.target_ref}")
    print(f"[video_qa] video_generation_result_id={qa_result.video_generation_result_id}")
    print(f"[video_qa] overall_verdict={qa_result.overall_verdict}")
    completed = sum(1 for row in result.scenes if row.status == "completed")
    print(f"[video_qa] scenes={len(result.scenes)} (completed={completed})")
    if chained:
        print(f"[video_qa] resumed_from={qa_result.resumed_from_qa_result_id}")
    for record in qa_result.checks:
        print(
            f"[video_qa]   check {record.check}: outcome={record.outcome} "
            f"(errors={record.errors}, warnings={record.warnings}, infos={record.infos})"
        )
    for issue in qa_result.issues:
        print(
            f"[video_qa]   issue {issue.issue_id} [{issue.category}][{issue.gate}] "
            f"{issue.message}"
        )
    for warning in qa_result.warnings:
        print(f"[video_qa]   warning {warning.issue_id} [{warning.category}] {warning.message}")
    for info in qa_result.infos:
        print(f"[video_qa]   info {info.issue_id} [{info.category}] {info.message}")


if __name__ == "__main__":
    raise SystemExit(main())