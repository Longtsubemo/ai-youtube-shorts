"""Stage 5 CLI: Video Generation.

Reads ``data/<run_id>/scene_plan.json`` and writes ``data/<run_id>/video.mp4``.

Usage:
    python -m agents.video_generation.video_cli --run-id <run_id>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from uuid import uuid4

from agents.video_generation.stage_simple import PLACEHOLDER_BYTES, write_video_placeholder


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agents.video_generation.video_cli",
        description="Render a scene plan into a video (Stage 5).",
    )
    parser.add_argument("--run-id", default=None, help="run id (default: generated)")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path for the video (default: data/<run_id>/video.mp4)",
    )
    args = parser.parse_args()

    run_id = args.run_id or uuid4().hex[:12]
    run_dir = Path("data") / run_id

    plan_path = run_dir / "scene_plan.json"
    if not plan_path.exists():
        print(f"[video_generation] Missing Stage 4 artifact: {plan_path}")
        return 2

    # NOT IMPLEMENTED: no renderer exists. This writes a byte placeholder so
    # the artifact chain can be exercised; it is not a playable video.
    out_path = args.out or (run_dir / "video.mp4")
    write_video_placeholder(out_path)

    print(f"[video_generation] run_id={run_id}")
    print(f"[video_generation] wrote {len(PLACEHOLDER_BYTES)} placeholder bytes to {out_path}")
    print("[video_generation] NOTE: no real renderer; this is NOT a playable video.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())