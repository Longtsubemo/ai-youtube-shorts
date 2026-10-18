"""Deterministic image-generation jobs: Stage 4 scene plan -> Colab SDXL worker.

This package converts a validated Stage 4 ``scene_plan.json`` into a compact,
deterministic job batch (``image_jobs.json``) that an external SDXL-Lightning
batch worker (e.g. a Colab notebook) can consume without any pipeline code.

Contract highlights:

- The final SDXL positive prompt is composed as
  ``channel_image_style + ", " + scene.generation_prompt`` - the channel style
  lives once (channel config / scene plan) and is never repeated inside the
  scene-specific compact prompt.
- The mascot is NEVER part of any SDXL prompt. It stays a separate asset
  (``assets/channels/finance_mascot/mascot.png``) composited later by Stage 5.
- Narration is NEVER part of any SDXL prompt.
- Per-scene seeds are deterministic: derived from ``run_id`` + ``scene_id``
  via SHA-256 (no randomness anywhere).
- The combined positive prompt (not just the scene prompt) is validated
  against the CLIP budget. Overlong prompts fail clearly; nothing is ever
  silently truncated.
"""

from agents.image_jobs.converter import (
    DEFAULT_NEGATIVE_PROMPT,
    FINAL_PROMPT_HARD_CEILING_TOKENS,
    FINAL_PROMPT_TARGET_TOKENS,
    ImageJobError,
    build_image_job_batch,
    check_final_prompt,
    check_negative_prompt,
    derive_seed,
    load_scene_plan,
    write_image_jobs,
)
from agents.image_jobs.jobs import (
    JOB_SCHEMA_VERSION,
    ImageJob,
    ImageJobBatch,
    MascotJobInfo,
)

__all__ = [
    "DEFAULT_NEGATIVE_PROMPT",
    "FINAL_PROMPT_HARD_CEILING_TOKENS",
    "FINAL_PROMPT_TARGET_TOKENS",
    "ImageJob",
    "ImageJobBatch",
    "ImageJobError",
    "JOB_SCHEMA_VERSION",
    "MascotJobInfo",
    "build_image_job_batch",
    "check_final_prompt",
    "check_negative_prompt",
    "derive_seed",
    "load_scene_plan",
    "write_image_jobs",
]
