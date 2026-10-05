"""Stage 4: Scene Planning.

Reads ``script.json`` and writes ``scene_plan.json``.

CLI: ``python -m agents.scene_planner.scene_cli --run-id <run_id>``
"""

from agents.scene_planner.scene_models import (
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
from agents.scene_planner.validator import ScenePlanningValidation

__all__ = [
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
    "load_script",
    "run_scene_planning",
    "write_scene_plan",
]