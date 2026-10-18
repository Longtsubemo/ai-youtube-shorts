"""Convert a Stage 4 scene plan into a deterministic SDXL image-job batch.

Composition rule (see ``config/channel.py``)::

    positive_prompt = channel_image_style + ", " + scene.generation_prompt

Validation is applied to the COMBINED positive prompt (not only the scene
prompt), and to the negative prompt separately. The CLIP text encoder accepts
at most 77 tokens (including 2 special tokens), so:

- ``FINAL_PROMPT_TARGET_TOKENS`` (60) - soft target; above it a warning is
  recorded, never an error.
- ``FINAL_PROMPT_HARD_CEILING_TOKENS`` (75) - hard ceiling; exceeding it is an
  error and the conversion FAILS CLEARLY. Prompts are never truncated.

Token counting uses the same dependency-free, deliberately conservative
``estimate_clip_tokens`` approximation as Stage 4 (sum of
``ceil(len(word)/4)``), so an accepted prompt is safely under the real
77-token CLIP limit.

Seeds are deterministic per scene: ``derive_seed(run_id, scene_id)`` takes the
first 4 bytes of ``sha256(f"{run_id}:{scene_id}")`` as a big-endian uint32.
The same run and scene always yield the same seed; nothing is random.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from agents.image_jobs.jobs import (
    ImageJob,
    ImageJobBatch,
    MascotJobInfo,
)
from agents.scene_planner.scene_models import (
    CHANNEL_IMAGE_STYLE,
    MASCOT_ASSET_PATH,
    ScenePlan,
)
from agents.scene_planner.validator import (
    FORBIDDEN_PROMPT_CONCEPTS,
    ValidationIssue,
    contains_narration,
    estimate_clip_tokens,
)
from config.channel import CHANNEL

# Soft target for the COMBINED positive prompt and for the negative prompt.
FINAL_PROMPT_TARGET_TOKENS = 60
# Hard ceiling (never exceeded; CLIP limit is 77 including 2 special tokens).
FINAL_PROMPT_HARD_CEILING_TOKENS = 75

# Visual-quality negative prompt for the channel's illustration style. It is
# validated ONLY against the token budget - concept words like "photorealistic"
# or "text" belong in a negative prompt by design.
DEFAULT_NEGATIVE_PROMPT = (
    "photorealistic, 3d render, animation, text, words, logo, watermark, "
    "people, faces, fox, mascot, blurry, deformed, extra limbs"
)

# Stage 5 resolves scene backgrounds as data/<run_id>/assets/<scene_id>_main.*
DEFAULT_OUTPUT_EXTENSION = ".png"


class ImageJobError(ValueError):
    """Raised when a scene plan cannot be converted into a valid job batch."""


def derive_seed(run_id: str, scene_id: str) -> int:
    """Deterministic uint32 seed for one scene (SHA-256 of run_id + scene_id)."""
    digest = hashlib.sha256(f"{run_id}:{scene_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def check_final_prompt(
    prompt: str,
    *,
    scene_id: str = "",
    narration: str = "",
    channel_style: str = "",
) -> list[ValidationIssue]:
    """Validate the COMBINED positive prompt for one scene."""
    issues: list[ValidationIssue] = []
    label = f"scene {scene_id}" if scene_id else "scene"
    text = prompt if isinstance(prompt, str) else ""

    if not text.strip():
        issues.append(
            ValidationIssue("ij01", f"{label}: final positive prompt is empty")
        )
        return issues

    # The composition rule must hold: style first, then the scene prompt.
    style = (channel_style or "").strip()
    if style and not text.startswith(style):
        issues.append(
            ValidationIssue(
                "ij01",
                f"{label}: final positive prompt does not start with the "
                "channel image style (composition rule violated)",
            )
        )

    estimated = estimate_clip_tokens(text)
    if estimated > FINAL_PROMPT_HARD_CEILING_TOKENS:
        issues.append(
            ValidationIssue(
                "ij02",
                f"{label}: final positive prompt is ~{estimated} tokens, above "
                f"the hard ceiling of {FINAL_PROMPT_HARD_CEILING_TOKENS} CLIP "
                "tokens; refusing (prompts are never truncated)",
            )
        )
    elif estimated > FINAL_PROMPT_TARGET_TOKENS:
        issues.append(
            ValidationIssue(
                "ij03",
                f"{label}: final positive prompt is ~{estimated} tokens, above "
                f"the target of {FINAL_PROMPT_TARGET_TOKENS} CLIP tokens",
                severity="warning",
            )
        )

    if contains_narration(text, narration):
        issues.append(
            ValidationIssue(
                "ij04",
                f"{label}: final positive prompt embeds the scene narration; "
                "narration is never part of an SDXL prompt",
            )
        )

    for concept, pattern in FORBIDDEN_PROMPT_CONCEPTS.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            if concept == "mascot":
                issues.append(
                    ValidationIssue(
                        "ij05",
                        f"{label}: final positive prompt mentions the "
                        "mascot/fox; the mascot is a separate composited asset",
                    )
                )
            else:
                issues.append(
                    ValidationIssue(
                        "ij06",
                        f"{label}: final positive prompt contains forbidden "
                        f"concept {concept!r}",
                    )
                )

    return issues


def check_negative_prompt(prompt: str) -> list[ValidationIssue]:
    """Validate the negative prompt separately (budget + non-empty only)."""
    issues: list[ValidationIssue] = []
    text = prompt if isinstance(prompt, str) else ""
    if not text.strip():
        issues.append(
            ValidationIssue("ij07", "negative prompt is empty")
        )
        return issues

    estimated = estimate_clip_tokens(text)
    if estimated > FINAL_PROMPT_HARD_CEILING_TOKENS:
        issues.append(
            ValidationIssue(
                "ij08",
                f"negative prompt is ~{estimated} tokens, above the hard "
                f"ceiling of {FINAL_PROMPT_HARD_CEILING_TOKENS} CLIP tokens",
            )
        )
    elif estimated > FINAL_PROMPT_TARGET_TOKENS:
        issues.append(
            ValidationIssue(
                "ij09",
                f"negative prompt is ~{estimated} tokens, above the target "
                f"of {FINAL_PROMPT_TARGET_TOKENS} CLIP tokens",
                severity="warning",
            )
        )
    return issues


def load_scene_plan(path: str | Path) -> ScenePlan:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ScenePlan.from_dict(data)


def write_image_jobs(batch: ImageJobBatch, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(batch.to_dict(), indent=2), encoding="utf-8")
    return out_path


def build_image_job_batch(
    plan: ScenePlan,
    run_id: str = "",
    *,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
) -> ImageJobBatch:
    """Build a validated, deterministic job batch from a Stage 4 plan.

    Raises ``ImageJobError`` (never truncates) when any final positive prompt
    or the negative prompt is invalid at the error severity.
    """
    resolved_run_id = run_id or plan.run_id
    if not resolved_run_id or not resolved_run_id.strip():
        raise ImageJobError("run_id is required to build image jobs")

    style = (plan.channel_image_style or CHANNEL_IMAGE_STYLE).strip()
    if not style:
        raise ImageJobError("channel image style is empty; nothing to compose")
    if estimate_clip_tokens(style) > FINAL_PROMPT_HARD_CEILING_TOKENS:
        raise ImageJobError(
            "channel image style alone exceeds the final-prompt hard ceiling; "
            "fix the channel style before building image jobs"
        )

    scenes = sorted(plan.scenes, key=lambda s: s.order)
    if not scenes:
        raise ImageJobError("scene plan has no scenes")

    issues: list[ValidationIssue] = []
    jobs: list[ImageJob] = []

    for scene in scenes:
        scene_prompt = (scene.generation_prompt or "").strip()
        if not scene_prompt:
            issues.append(
                ValidationIssue(
                    "ij01",
                    f"scene {scene.scene_id}: generation_prompt is empty; "
                    "nothing to compose into a final positive prompt",
                )
            )
        positive_prompt = f"{style}, {scene_prompt}" if scene_prompt else style
        scene_issues = check_final_prompt(
            positive_prompt,
            scene_id=scene.scene_id,
            narration=scene.narration,
            channel_style=style,
        )
        issues.extend(scene_issues)

        asset_id = f"{scene.scene_id}_main"
        jobs.append(
            ImageJob(
                job_id=f"{resolved_run_id}-{scene.scene_id}",
                scene_id=scene.scene_id,
                order=scene.order,
                duration_seconds=scene.duration_seconds,
                scene_generation_prompt=scene_prompt,
                channel_image_style=style,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                seed=derive_seed(resolved_run_id, scene.scene_id),
                asset_id=asset_id,
                output_path=(
                    f"data/{resolved_run_id}/assets/{asset_id}"
                    f"{DEFAULT_OUTPUT_EXTENSION}"
                ),
                mascot=MascotJobInfo(
                    present=scene.mascot.present,
                    action=scene.mascot.action,
                    asset=scene.mascot.asset or MASCOT_ASSET_PATH,
                    in_positive_prompt=False,
                ),
            )
        )

    issues.extend(check_negative_prompt(negative_prompt))

    errors = [i for i in issues if i.severity == "error"]
    if errors:
        details = "; ".join(f"[{i.code}] {i.message}" for i in errors)
        raise ImageJobError(f"invalid image jobs: {details}")

    return ImageJobBatch(
        run_id=resolved_run_id,
        topic_id=plan.topic_id,
        title=plan.title,
        channel_id=plan.channel_id,
        channel_image_style=style,
        negative_prompt=negative_prompt,
        jobs=jobs,
        warnings=[i.to_dict() for i in issues if i.severity == "warning"],
    )
