"""Manual runner for the Stage-6 scene-planning stage (no external APIs).

Usage:
    python -m agents.scene_planner

    # explicit artifacts:
    python -m agents.scene_planner --script data/<run_id>/scripts/script.json \
        --qa-result data/<run_id>/scripts/script_qa_result.json \
        --result data/<run_id>/research/fact_check_result.json

Loads the QA-passed Script artifact + its ScriptQAResult + FactCheckResult,
enforces the PASS gate, runs the deterministic MockScenePlanner, validates the
ProductionPlan, and writes data/<run_id>/production/production.json (plus run
state). Exits non-zero when planning is blocked or the plan is invalid.
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from agents.scene_planner.models import ScenePlanningConfig
from agents.scene_planner.stage import (
    DEFAULT_FACT_CHECK_RESULT_PATH,
    DEFAULT_SCRIPT_PATH,
    DEFAULT_SCRIPT_QA_RESULT_PATH,
    load_fact_check_result,
    load_script,
    load_script_qa_result,
    run_scene_planning,
)
from agents.research.models import ResearchPackage
from agents.research.provider import MockResearchProvider
from agents.research.stage import load_topic_selection, run_research
from pipeline.state import PipelineState, Stage


def build_mock_package():
    """Deterministically produce the mock Stage-2 ResearchPackage in memory."""
    selection = load_topic_selection()
    return run_research(selection, MockResearchProvider(), run_id="mock")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agents.scene_planner",
        description="Plan provider-independent scenes for a QA-passed Script.",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=DEFAULT_SCRIPT_PATH,
        help="Script artifact JSON (default: data/mock/script.json)",
    )
    parser.add_argument(
        "--qa-result",
        type=Path,
        default=DEFAULT_SCRIPT_QA_RESULT_PATH,
        help="ScriptQAResult artifact JSON (default: data/mock/script_qa_result.json)",
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=DEFAULT_FACT_CHECK_RESULT_PATH,
        help="FactCheckResult artifact JSON (default: data/mock/fact_check_result.json)",
    )
    parser.add_argument("--run-id", default=None, help="run id (default: script's run id)")
    parser.add_argument(
        "--research",
        type=Path,
        default=None,
        help=(
            "Stage-2 ResearchPackage artifact JSON. Supply this for a real run: "
            "scene planning resolves approved claim ids against THIS package, so "
            "the default mock package cannot plan a real Script."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output root under which <run_id>/production/production.json is written",
    )
    args = parser.parse_args()

    for label, path in (
        ("Script", args.script),
        ("ScriptQAResult", args.qa_result),
        ("FactCheckResult", args.result),
    ):
        if not path.is_file():
            raise SystemExit(f"[scene_planner] {label} not found: {path}")

    script = load_script(args.script)
    qa_result = load_script_qa_result(args.qa_result)
    result = load_fact_check_result(args.result)
    if args.research is not None:
        if not args.research.is_file():
            raise SystemExit(f"[scene_planner] ResearchPackage not found: {args.research}")
        package = ResearchPackage.from_json(args.research.read_text(encoding="utf-8"))
    else:
        package = build_mock_package()
    if args.run_id:
        run_id = args.run_id
    elif (
        args.script == DEFAULT_SCRIPT_PATH
        and args.qa_result == DEFAULT_SCRIPT_QA_RESULT_PATH
        and args.result == DEFAULT_FACT_CHECK_RESULT_PATH
    ):
        # Default demo invocation: keep data/mock/ (committed fixtures) clean.
        run_id = "sp-demo"
    else:
        run_id = script.run_id or result.run_id or uuid.uuid4().hex[:12]

    planning = run_scene_planning(
        script,
        qa_result,
        result,
        package=package,
        config=ScenePlanningConfig(),
        run_id=run_id,
    )

    if not planning.planned:
        print(f"[scene_planner] planning status=BLOCKED")
        print(f"[scene_planner] reason={planning.reason}")
        if planning.validation is not None:
            for issue in planning.validation.errors:
                print(f"[scene_planner]   [{issue.rule_id}] {issue.message}")
        return 1

    plan = planning.plan
    out_root = args.out or Path("data")
    out_path = out_root / run_id / "production" / "production.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(plan.to_json(), encoding="utf-8")

    state_path = out_root / run_id / "state.json"
    try:
        state = PipelineState.load(state_path)
    except FileNotFoundError:
        state = PipelineState(run_id=run_id)
    state.advance_to(Stage.SCENE_PLANNING)
    state.add_artifact(Stage.SCENE_PLANNING, out_path)
    state.save(state_path)

    _print_report(plan, planning)
    print(f"[scene_planner] artifact written to {out_path}")
    print(f"[scene_planner] run state written to {state_path}")
    return 0


def _print_report(plan, planning) -> None:
    covered = len(plan.covered_script_line_ids)
    total = covered + len(plan.uncovered_script_line_ids)
    coverage = covered / total if total else 0.0
    print(f"[scene_planner] planning status=PLANNED")
    print(f"[scene_planner] production_plan_id={plan.production_plan_id}")
    print(f"[scene_planner] scene count={len(plan.scenes)}")
    print(f"[scene_planner] total duration={plan.total_duration_seconds}s")
    print(f"[scene_planner] covered script lines={covered}/{total}")
    print(f"[scene_planner] coverage={coverage}")
    print(f"[scene_planner] validation passed={planning.validation.passed}")
    if plan.scenes:
        mapping = ", ".join(
            f"scene {scene.scene_id} -> {','.join(scene.script_line_ids)}"
            for scene in plan.scenes
        )
        print(f"[scene_planner] script->scene mapping: {mapping}")


if __name__ == "__main__":
    raise SystemExit(main())