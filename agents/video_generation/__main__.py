"""Manual runner for the Stage-7 video-generation stage (no external APIs).

Usage:
    python -m agents.video_generation

    # explicit ProductionPlan (e.g. a real Stage-6 run):
    python -m agents.video_generation --plan data/<run_id>/production/production.json \
        --run-id <run_id>

    # resume/retry from a previous VideoGenerationResult:
    python -m agents.video_generation \
        --plan data/<run_id>/production/production.json \
        --previous data/<run_id>/videos/video_generation_result.json

Loads the Stage-6 ProductionPlan (or the deterministic mock plan), drives it
through the `VideoGenerator` abstraction (default: the offline, GPU-free, API-
free MockVideoGenerator), validates the generated VideoGenerationResult, and
writes data/<run_id>/videos/video_generation_result.json plus run state.

Exit codes: 0 = success; 1 = generation blocked / result validation failed (no
artifact written); 2 = error (e.g. missing/corrupt inputs, mismatched previous).
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from agents.video_generation.models import VideoGenerationConfig, VideoGenerationError
from agents.video_generation.stage import (
    DEFAULT_PRODUCTION_PLAN_PATH,
    DEFAULT_VIDEO_RESULT_NAME,
    load_production_plan,
    load_video_generation_result,
    run_video_generation,
)
from pipeline.state import PipelineState, Stage


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agents.video_generation",
        description=(
            "Generate per-scene video-generation records from a Stage-6 "
            "ProductionPlan using a deterministic, offline mock generator."
        ),
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=DEFAULT_PRODUCTION_PLAN_PATH,
        help="ProductionPlan artifact JSON "
        "(default: data/mock/production.json)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="run id (default: 'vg-demo' for the mock plan, else the plan's run id)",
    )
    parser.add_argument(
        "--previous",
        type=Path,
        default=None,
        help=(
            "previous VideoGenerationResult JSON to resume/retry from "
            "(completed scenes preserved, retryable failures retried, "
            "pending scenes processed)"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output root under which <run_id>/videos/video_generation_result.json is written",
    )
    args = parser.parse_args()

    if not args.plan.is_file():
        raise SystemExit(f"[video_generation] ProductionPlan not found: {args.plan}")
    plan = load_production_plan(args.plan)

    previous = None
    if args.previous is not None:
        if not args.previous.is_file():
            raise SystemExit(f"[video_generation] previous result not found: {args.previous}")
        previous = load_video_generation_result(args.previous)

    if args.run_id:
        run_id = args.run_id
    elif args.plan == DEFAULT_PRODUCTION_PLAN_PATH:
        # Default demo invocation: keep the committed data/mock fixture clean.
        run_id = "vg-demo"
    else:
        run_id = plan.run_id or uuid.uuid4().hex[:12]

    try:
        outcome = run_video_generation(
            plan,
            config=VideoGenerationConfig(),
            previous=previous,
            run_id=run_id,
        )
    except VideoGenerationError as exc:
        print(f"[video_generation] error: {exc}")
        return 2

    if not outcome.generated:
        print(f"[video_generation] generation status=FAILED")
        print(f"[video_generation] reason={outcome.reason}")
        if outcome.validation is not None:
            for issue in outcome.validation.errors:
                print(f"[video_generation]   [{issue.rule_id}] {issue.message}")
        return 1

    result = outcome.result
    out_root = args.out or Path("data")
    out_path = out_root / run_id / "videos" / DEFAULT_VIDEO_RESULT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.to_json(), encoding="utf-8")

    state_path = out_root / run_id / "state.json"
    try:
        state = PipelineState.load(state_path)
    except FileNotFoundError:
        state = PipelineState(run_id=run_id)
    state.advance_to(Stage.VIDEO_GENERATION)
    state.add_artifact(Stage.VIDEO_GENERATION, out_path)
    state.save(state_path)

    _print_report(result, outcome, run_id, previous is not None)
    print(f"[video_generation] artifact written to {out_path}")
    print(f"[video_generation] run state written to {state_path}")
    return 0


def _print_report(result, outcome, run_id: str, resumed: bool) -> None:
    completed = sum(1 for row in result.scenes if row.status == "completed")
    failed = sum(1 for row in result.scenes if row.status == "failed")
    pending = sum(1 for row in result.scenes if row.status == "pending")
    print(f"[video_generation] generation status={result.overall_status}")
    print(f"[video_generation] video_generation_result_id={result.video_generation_result_id}")
    print(f"[video_generation] production_plan_id={result.production_plan_id}")
    print(f"[video_generation] run_id={run_id}")
    print(f"[video_generation] scenes={len(result.scenes)} "
          f"(completed={completed}, failed={failed}, pending={pending})")
    if resumed:
        print(f"[video_generation] resumed_from={result.resumed_from_result_id}")
    print(f"[video_generation] validation passed={outcome.validation.passed}")
    for row in result.scenes:
        status = row.status
        detail = row.artifact.artifact_id if row.status == "completed" else row.failure
        print(f"[video_generation]   scene {row.scene_id} ({row.order}) "
              f"status={status} attempts={row.attempts} {detail}")


if __name__ == "__main__":
    raise SystemExit(main())