"""Stage 4 CLI: Scene Planning.

Reads ``data/<run_id>/script.json`` and writes ``data/<run_id>/scene_plan.json``.

Usage:
    python -m agents.scene_planner.scene_cli --run-id <run_id>
    python -m agents.scene_planner.scene_cli --run-id <run_id> --provider gemini
"""

from __future__ import annotations

import argparse
from pathlib import Path
from uuid import uuid4

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from agents.scene_planner.provider import (
    ScenePlannerProviderError,
    build_scene_planner_provider,
)
from agents.scene_planner.stage import (
    ScenePlanningError,
    load_script,
    run_scene_planning,
    write_scene_plan,
)
from agents.scene_planner.validator import validate_scene_plan_generation
from agents.seo.config import DEFAULT_CONFIG_PATH, load_config
from pipeline.state import PipelineState, Stage

PROVIDERS = ("mock", "gemini")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agents.scene_planner.scene_cli",
        description="Plan scenes from a script (Stage 4).",
    )
    parser.add_argument("--run-id", default=None, help="run id (default: generated)")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="topic-selection/config YAML",
    )
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        default="mock",
        help="scene planner provider (default: mock)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path for the ScenePlan artifact",
    )
    args = parser.parse_args()

    if load_dotenv is not None:
        load_dotenv()

    run_id = args.run_id or uuid4().hex[:12]
    run_dir = Path("data") / run_id

    script_path = run_dir / "script.json"
    if not script_path.exists():
        print(f"[scene_planner] Missing Stage 3 artifact: {script_path}")
        return 2

    try:
        config = load_config(args.config)
    except Exception as exc:  # noqa: BLE001
        print(f"[scene_planner] config error: {exc}")
        return 2

    try:
        script = load_script(script_path)
        provider = build_scene_planner_provider(args.provider, config=config)
        plan = run_scene_planning(script, provider, run_id=run_id)
    except (ScenePlanningError, ScenePlannerProviderError) as exc:
        print(f"[scene_planner] input error: {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"[scene_planner] provider error: {type(exc).__name__}: {exc}")
        return 2

    out_path = args.out or (run_dir / "scene_plan.json")
    for issue in validate_scene_plan_generation(plan).warnings:
        print(f"[scene_planner] warning [{issue.code}] {issue.message}")
    write_scene_plan(plan, out_path)

    state = PipelineState(run_id=run_id)
    state.advance_to(Stage.SCENE_PLANNING)
    state.add_artifact(Stage.SCENE_PLANNING, out_path)
    state.save(run_dir / "state.json")

    print(f"[scene_planner] run_id={run_id}")
    print(f"[scene_planner] provider={args.provider} topic={plan.topic_id}")
    print(f"[scene_planner] scenes={len(plan.scenes)} total={plan.total_scene_duration_seconds}s")
    print(f"[scene_planner] artifact written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())