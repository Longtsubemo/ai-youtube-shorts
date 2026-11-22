"""Stage 5: Video Generation.

Generic, data-driven video rendering for the channel:

    scene_plan.json -> translator -> render spec -> generic scene renderer
        -> data/<run_id>/video/scNN.mp4 -> video/final.mp4 -> video.mp4

Providers:

    VideoGenerator
        ├── LocalMotionRenderer   (implemented: local deterministic renderer)
        └── AIVideoRenderer       (future seam, not implemented)

CLI: ``python -m agents.video_generation.video_cli --run-id <run_id> --provider local``
"""

from agents.video_generation.models import SceneClip, VideoResult
from agents.video_generation.provider import (
    LocalMotionRenderer,
    VideoGenerator,
    VideoGeneratorError,
    available_providers,
    build_video_generator,
)
from agents.video_generation.render_spec import (
    ANIMATION_TYPES,
    ELEMENT_TYPES,
    RenderSpec,
    RenderSpecError,
    SceneRenderSpec,
)
from agents.video_generation.stage import (
    VideoGenerationError,
    load_scene_plan,
    run_video_generation,
)
from agents.video_generation.translator import TRANSLATOR_VERSION, translate_scene_plan

__all__ = [
    "ANIMATION_TYPES",
    "ELEMENT_TYPES",
    "LocalMotionRenderer",
    "RenderSpec",
    "RenderSpecError",
    "SceneClip",
    "SceneRenderSpec",
    "TRANSLATOR_VERSION",
    "VideoGenerationError",
    "VideoGenerator",
    "VideoGeneratorError",
    "VideoResult",
    "available_providers",
    "build_video_generator",
    "load_scene_plan",
    "run_video_generation",
    "translate_scene_plan",
]
