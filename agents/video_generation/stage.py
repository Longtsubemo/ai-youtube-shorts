"""Stage-7 entrypoint: ProductionPlan -> VideoGenerationResult.

Consumes ONLY the Stage-6 ProductionPlan (the stable CONTENT -> VIDEO boundary)
and drives it through the `VideoGenerator` abstraction. The deterministic,
offline `MockVideoGenerator` is the default; a future real provider implements
the same protocol and drops in behind this stage without changing the runner,
validator, resume/retry logic, or CLI.

Resume/retry (`previous`): completed scenes are preserved, failed retryable
scenes may be retried (bounded by VideoGenerationConfig.max_attempts), pending
scenes are processed, and non-retryable failures stay recorded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agents.scene_planner.models import ProductionPlan
from agents.video_generation.generator import VideoGenerator
from agents.video_generation.mock_generator import MockVideoGenerator
from agents.video_generation.models import (
    VideoGenerationConfig,
    VideoGenerationError,
    VideoGenerationResult,
)
from agents.video_generation.validator import validate_video_generation_result

DEFAULT_PRODUCTION_PLAN_PATH = Path("data") / "mock" / "production.json"
DEFAULT_VIDEO_RESULT_NAME = "video_generation_result.json"


def load_production_plan(path: str | Path = DEFAULT_PRODUCTION_PLAN_PATH) -> ProductionPlan:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return ProductionPlan.from_dict(raw)


def load_video_generation_result(path: str | Path) -> VideoGenerationResult:
    return VideoGenerationResult.from_json(Path(path).read_text(encoding="utf-8"))


@dataclass
class VideoGenerationOutcome:
    """Stage runner outcome: a validated result, or a failed generation."""

    generated: bool
    result: VideoGenerationResult | None = None
    validation: object | None = None
    reason: str = ""


def run_video_generation(
    plan: ProductionPlan,
    generator: VideoGenerator | None = None,
    config: VideoGenerationConfig | None = None,
    previous: VideoGenerationResult | None = None,
    run_id: str = "",
    created_at: str = "",
) -> VideoGenerationOutcome:
    """Run the Stage-7 stage over a ProductionPlan.

    `previous` (when given) must target the SAME ProductionPlan; a mismatched
    previous result raises VideoGenerationError before anything is generated,
    so a wrong resume can never silently mix plans. The generated result is
    validated against the plan before it is returned as `generated=True`.
    """
    if previous is not None and previous.production_plan_id != plan.production_plan_id:
        raise VideoGenerationError(
            f"previous result {previous.video_generation_result_id} targets "
            f"production plan {previous.production_plan_id}, "
            f"not {plan.production_plan_id}"
        )
    generator = generator or MockVideoGenerator(config=config)
    result = generator.generate(
        plan,
        config=config,
        previous=previous,
        run_id=run_id,
        created_at=created_at,
    )
    validation = validate_video_generation_result(result, plan, config=config)
    if not validation.passed:
        error_messages = "; ".join(issue.message for issue in validation.errors)
        return VideoGenerationOutcome(
            generated=False,
            result=result,
            validation=validation,
            reason=f"video generation result validation failed: {error_messages}",
        )
    return VideoGenerationOutcome(generated=True, result=result, validation=validation)


def generate_mock_production_plan(
    run_id: str = "mock",
    created_at: str = "",
) -> ProductionPlan:
    """Deterministically build the mock ProductionPlan in memory (no files)."""
    from agents.scene_planner.stage import generate_mock_production_plan as _plan

    return _plan(run_id=run_id, created_at=created_at)