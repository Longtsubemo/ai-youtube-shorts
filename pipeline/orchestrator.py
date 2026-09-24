"""Minimal pipeline orchestrator entry point.

Scaffold only: creating a run directory under ``data/`` with a fresh
pipeline state file is all it does for now. No stages are executed and
no external services are called.

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

RUN_SUBDIRS = ("research", "scripts", "production", "videos")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.orchestrator",
        description="AI YouTube Shorts pipeline orchestrator.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=DATA_DIR,
        help="Base data directory; a per-run subdirectory is created inside it.",
    )
    args = parser.parse_args()

    run_id = uuid4().hex[:12]
    run_dir = args.workdir / run_id
    for name in RUN_SUBDIRS:
        (run_dir / name).mkdir(parents=True, exist_ok=True)

    state = PipelineState(run_id=run_id)
    state_path = state.save(run_dir / "state.json")

    print(f"[orchestrator] Run {run_id} initialized at {run_dir}")
    print(f"[orchestrator] State written to {state_path}")
    print("[orchestrator] Scaffold only: stages are run individually via their CLIs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())