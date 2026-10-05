"""Stage 4 CLI: Scene Planning.

Reads ``data/<run_id>/script.json`` and writes ``data/<run_id>/scene_plan.json``.

Usage:
    python -m agents.scene_planner.scene_cli --run-id <run_id>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from uuid import uuid4

from agents.scene_planner.scene_models import ScenePlan
from agents.scene_planner.stage_simple import write_scene_plan


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agents.scene_planner.scene_cli",
        description="Plan scenes from a script (Stage 4).",
    )
    parser.add_argument("--run-id", default=None, help="run id (default: generated)")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path for the ScenePlan artifact (default: data/<run_id>/scene_plan.json)",
    )
    args = parser.parse_args()

    run_id = args.run_id or uuid4().hex[:12]
    run_dir = Path("data") / run_id

    script_path = run_dir / "script.json"
    if not script_path.exists():
        print(f"[scene_planner] Missing Stage 3 artifact: {script_path}")
        return 2

    # NOT IMPLEMENTED: the Gemini-backed scene planner. Until it exists this
    # stage emits an empty placeholder so the artifact chain can be exercised.
    plan = ScenePlan(topic_id="", scenes=[])

    out_path = args.out or (run_dir / "scene_plan.json")
    write_scene_plan(plan, out_path)

    print(f"[scene_planner] run_id={run_id}")
    print(f"[scene_planner] artifact written to {out_path}")
    print("[scene_planner] NOTE: scene planning is a placeholder, not yet Gemini-backed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())