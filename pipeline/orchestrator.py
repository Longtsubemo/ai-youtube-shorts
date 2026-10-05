"""Minimal pipeline orchestrator entry point.

Scaffold only: it creates a run directory under ``data/`` with a fresh
pipeline state file. It executes no stages and calls no external services.
The artifact layout is flat - one JSON file per stage directly in
``data/<run_id>/``:

    data/<run_id>/
        topic_selection.json   Stage 1
        research.json          Stage 2
        script.json            Stage 3
        scene_plan.json        Stage 4
        video.mp4              Stage 5
        state.json

Run from the project root with::

    python -m pipeline.orchestrator
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from uuid import uuid4

from pipeline.state import PipelineState

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Flat layout: the artifact each stage produces, keyed by stage name.
STAGE_ARTIFACTS: dict[str, str] = {
    "topic_selection": "topic_selection.json",
    "ai_research": "research.json",
    "script_generation": "script.json",
    "scene_planning": "scene_plan.json",
    "video_generation": "video.mp4",
}


def run_dir_for(run_id: str, base: Path = DATA_DIR) -> Path:
    """Return the flat per-run directory for ``run_id``."""
    return base / run_id


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.orchestrator",
        description="AI YouTube Shorts pipeline orchestrator (scaffold).",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=DATA_DIR,
        help="Base data directory; a per-run subdirectory is created inside it.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Explicit run id (default: generated).",
    )
    args = parser.parse_args()

    run_id = args.run_id or uuid4().hex[:12]
    run_dir = run_dir_for(run_id, args.workdir)
    run_dir.mkdir(parents=True, exist_ok=True)

    state = PipelineState(run_id=run_id)
    state_path = state.save(run_dir / "state.json")

    print(f"[orchestrator] Run {run_id} initialized at {run_dir}")
    print(f"[orchestrator] State written to {state_path}")
    print("[orchestrator] Artifact layout is flat; expected files:")
    for stage, filename in STAGE_ARTIFACTS.items():
        print(f"[orchestrator]   {stage:<20} {run_dir / filename}")
    print("[orchestrator] Scaffold only: stages are run individually via their CLIs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())