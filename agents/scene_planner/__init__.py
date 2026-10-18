"""Stage 4: Scene Planning.

Reads ``script.json`` and writes ``scene_plan.json``.

CLI: ``python -m agents.scene_planner.scene_cli --run-id <run_id>``
"""

from agents.scene_planner.scene_models import (
    CHANNEL_IMAGE_STYLE,
    MASCOT_ASSET_PATH,
    MASCOT_SPEAKS,
    Mascot,
    MascotScene,
    Scene,
    ScenePlan,
    ScenePlanningError,
)
from agents.scene_planner.provider import (
    MockScenePlannerProvider,
    ScenePlannerProvider,
    ScenePlannerProviderError,
    build_scene_planner_provider,
)
from agents.scene_planner.stage import (
    build_scene_plan_context,
    load_script,
    run_scene_planning,
    write_scene_plan,
)
from agents.scene_planner.validator import (
    GENERATION_PROMPT_HARD_CEILING_TOKENS,
    GENERATION_PROMPT_TARGET_TOKENS,
    ScenePlanningValidation,
    check_generation_prompt,
    estimate_clip_tokens,
    validate_scene_plan_generation,
)

__all__ = [
    "CHANNEL_IMAGE_STYLE",
    "GENERATION_PROMPT_HARD_CEILING_TOKENS",
    "GENERATION_PROMPT_TARGET_TOKENS",
    "MASCOT_ASSET_PATH",
    "MASCOT_SPEAKS",
    "Mascot",
    "MascotScene",
    "MockScenePlannerProvider",
    "Scene",
    "ScenePlan",
    "ScenePlanningError",
    "ScenePlanningValidation",
    "ScenePlannerProvider",
    "ScenePlannerProviderError",
    "build_scene_plan_context",
    "build_scene_planner_provider",
    "check_generation_prompt",
    "estimate_clip_tokens",
    "load_script",
    "run_scene_planning",
    "validate_scene_plan_generation",
    "write_scene_plan",
]