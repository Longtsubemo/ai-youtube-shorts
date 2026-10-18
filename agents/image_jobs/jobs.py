"""Data models for the image-generation job batch.

The batch is a pure data contract: no provider SDK, no network, no randomness.
A batch worker downloads ``image_jobs.json``, generates one image per job at
``output_path`` using ``positive_prompt`` / ``negative_prompt`` / ``seed``, and
the pipeline (Stage 5) later resolves those files as scene backgrounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

JOB_SCHEMA_VERSION = "image-jobs/1.0"

PROMPT_COMPOSITION_RULE = (
    "positive_prompt = channel_image_style + ', ' + scene_generation_prompt"
)

MASCOT_NOTE = (
    "The mascot is never part of any SDXL prompt. It is a separate asset "
    "(see mascot.asset) composited onto the generated background later by "
    "Stage 5."
)


@dataclass(frozen=True)
class MascotJobInfo:
    """Mascot presence for a scene, kept OUTSIDE every SDXL prompt."""

    present: bool = False
    action: str = ""
    asset: str = ""
    in_positive_prompt: bool = False
    composited_by: str = "stage5"

    def to_dict(self) -> dict:
        return {
            "present": self.present,
            "action": self.action,
            "asset": self.asset,
            "in_positive_prompt": self.in_positive_prompt,
            "composited_by": self.composited_by,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MascotJobInfo":
        return cls(
            present=bool(data.get("present", False)),
            action=str(data.get("action", "")),
            asset=str(data.get("asset", "")),
            in_positive_prompt=bool(data.get("in_positive_prompt", False)),
            composited_by=str(data.get("composited_by", "stage5")),
        )


@dataclass(frozen=True)
class ImageJob:
    """One deterministic SDXL generation job for one scene background."""

    job_id: str
    scene_id: str
    order: int
    duration_seconds: int
    scene_generation_prompt: str
    channel_image_style: str
    positive_prompt: str
    negative_prompt: str
    seed: int
    asset_id: str
    output_path: str
    mascot: MascotJobInfo = field(default_factory=MascotJobInfo)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "scene_id": self.scene_id,
            "order": self.order,
            "duration_seconds": self.duration_seconds,
            "scene_generation_prompt": self.scene_generation_prompt,
            "channel_image_style": self.channel_image_style,
            "positive_prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
            "asset_id": self.asset_id,
            "output_path": self.output_path,
            "mascot": self.mascot.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ImageJob":
        return cls(
            job_id=str(data.get("job_id", "")),
            scene_id=str(data.get("scene_id", "")),
            order=int(data.get("order", 0)),
            duration_seconds=int(data.get("duration_seconds", 0)),
            scene_generation_prompt=str(data.get("scene_generation_prompt", "")),
            channel_image_style=str(data.get("channel_image_style", "")),
            positive_prompt=str(data.get("positive_prompt", "")),
            negative_prompt=str(data.get("negative_prompt", "")),
            seed=int(data.get("seed", 0)),
            asset_id=str(data.get("asset_id", "")),
            output_path=str(data.get("output_path", "")),
            mascot=MascotJobInfo.from_dict(data.get("mascot") or {}),
        )


@dataclass
class ImageJobBatch:
    """Whole-run image job batch, one job per scene, fully deterministic."""

    schema: str = JOB_SCHEMA_VERSION
    run_id: str = ""
    topic_id: str = ""
    title: str = ""
    channel_id: str = ""
    channel_image_style: str = ""
    target_model: str = "sdxl-lightning"
    prompt_composition: str = PROMPT_COMPOSITION_RULE
    negative_prompt: str = ""
    mascot_note: str = MASCOT_NOTE
    jobs: list[ImageJob] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "topic_id": self.topic_id,
            "title": self.title,
            "channel_id": self.channel_id,
            "channel_image_style": self.channel_image_style,
            "target_model": self.target_model,
            "prompt_composition": self.prompt_composition,
            "negative_prompt": self.negative_prompt,
            "mascot_note": self.mascot_note,
            "jobs": [job.to_dict() for job in self.jobs],
            "warnings": [dict(w) for w in self.warnings],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ImageJobBatch":
        return cls(
            schema=str(data.get("schema", JOB_SCHEMA_VERSION)),
            run_id=str(data.get("run_id", "")),
            topic_id=str(data.get("topic_id", "")),
            title=str(data.get("title", "")),
            channel_id=str(data.get("channel_id", "")),
            channel_image_style=str(data.get("channel_image_style", "")),
            target_model=str(data.get("target_model", "sdxl-lightning")),
            prompt_composition=str(
                data.get("prompt_composition", PROMPT_COMPOSITION_RULE)
            ),
            negative_prompt=str(data.get("negative_prompt", "")),
            mascot_note=str(data.get("mascot_note", MASCOT_NOTE)),
            jobs=[ImageJob.from_dict(j) for j in data.get("jobs", [])],
            warnings=[dict(w) for w in data.get("warnings", [])],
        )
