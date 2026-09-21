"""Stage-8 entrypoint: VideoGenerationResult + ProductionPlan -> VideoQAResult.

QA-side only (docs/ARCHITECTURE.md): the stage consumes the Stage-7
`VideoGenerationResult` and the Stage-6 `ProductionPlan` the result was
generated from, runs the deterministic 15-check video-QA gate, and emits a
`VideoQAResult`. It never touches video bytes, never calls a provider, and
never inspects media that does not exist in the mock-first pipeline.

The plan/result pairing is guarded: QAing a result against a DIFFERENT plan
than the one it was generated from raises before anything is evaluated, so a
wrong pairing can never silently produce a verdict.
"""

from __future__ import annotations

from pathlib import Path

from agents.qa.video.models import VideoQAResult, VideoQAError
from agents.qa.video.validator import evaluate_video
from agents.scene_planner.models import ProductionPlan
from agents.video_generation.models import VideoGenerationResult
from agents.video_generation.stage import load_production_plan

DEFAULT_PRODUCTION_PLAN_PATH = Path("data") / "mock" / "production.json"
DEFAULT_VIDEO_RESULT_PATH = Path("data") / "mock" / "video_generation_result.json"
DEFAULT_QA_RESULT_NAME = "qa_result.json"


def load_video_generation_result(path: str | Path) -> VideoGenerationResult:
    return VideoGenerationResult.from_json(Path(path).read_text(encoding="utf-8"))


def load_video_qa_result(path: str | Path) -> VideoQAResult:
    return VideoQAResult.from_json(Path(path).read_text(encoding="utf-8"))


def run_video_qa(
    plan: ProductionPlan,
    result: VideoGenerationResult,
    config=None,
    run_id: str = "",
    created_at: str = "",
) -> VideoQAResult:
    """Run the Stage-8 video-QA gate over a Stage-7 result + its ProductionPlan.

    `result` must target the SAME ProductionPlan; a mismatched pairing raises
    VideoQAError before anything is evaluated.
    """
    if result.production_plan_id != plan.production_plan_id:
        raise VideoQAError(
            f"result {result.video_generation_result_id} targets production "
            f"plan {result.production_plan_id}, not {plan.production_plan_id}"
        )
    return evaluate_video(
        result,
        plan,
        config=config,
        run_id=run_id,
        created_at=created_at,
    )


def load_committed_fixtures():
    """Load the committed data/mock plan and its generated video result."""
    return load_production_plan(DEFAULT_PRODUCTION_PLAN_PATH), load_video_generation_result(
        DEFAULT_VIDEO_RESULT_PATH
    )