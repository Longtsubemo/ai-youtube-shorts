"""CLI: convert a Stage 4 scene plan into an image-job batch.

Usage:
    python -m agents.image_jobs --run-id <run_id>
    python -m agents.image_jobs --run-id <run_id> --out path/to/image_jobs.json

Reads ``data/<run_id>/scene_plan.json`` and writes ``data/<run_id>/image_jobs.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from uuid import uuid4

from agents.image_jobs.converter import (
    ImageJobError,
    build_image_job_batch,
    load_scene_plan,
    write_image_jobs,
)
from agents.scene_planner.validator import estimate_clip_tokens


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agents.image_jobs",
        description="Build deterministic SDXL image jobs from a Stage 4 scene plan.",
    )
    parser.add_argument("--run-id", default=None, help="run id (default: generated)")
    parser.add_argument(
        "--plan",
        type=Path,
        default=None,
        help="scene plan path (default: data/<run_id>/scene_plan.json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path for the image job batch (default: data/<run_id>/image_jobs.json)",
    )
    args = parser.parse_args()

    run_id = args.run_id or uuid4().hex[:12]
    run_dir = Path("data") / run_id
    plan_path = args.plan or (run_dir / "scene_plan.json")
    if not plan_path.exists():
        print(f"[image_jobs] Missing Stage 4 artifact: {plan_path}")
        return 2

    try:
        plan = load_scene_plan(plan_path)
        batch = build_image_job_batch(plan, run_id=run_id)
    except ImageJobError as exc:
        print(f"[image_jobs] input error: {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"[image_jobs] error: {type(exc).__name__}: {exc}")
        return 2

    for warning in batch.warnings:
        print(
            f"[image_jobs] warning [{warning.get('code')}] {warning.get('message')}"
        )

    out_path = args.out or (run_dir / "image_jobs.json")
    write_image_jobs(batch, out_path)

    print(f"[image_jobs] run_id={run_id} jobs={len(batch.jobs)}")
    for job in batch.jobs:
        print(
            f"[image_jobs] {job.scene_id} seed={job.seed} "
            f"final=~{estimate_clip_tokens(job.positive_prompt)}tok "
            f"negative=~{estimate_clip_tokens(job.negative_prompt)}tok "
            f"-> {job.output_path}"
        )
    print(f"[image_jobs] artifact written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
