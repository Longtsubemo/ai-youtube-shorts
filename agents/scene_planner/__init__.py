"""Stage 4: Scene Planning.

Reads ``script.json`` and writes ``scene_plan.json``.

CLI: ``python -m agents.scene_planner.scene_cli --run-id <run_id>``
"""

from agents.scene_planner.scene_models import Mascot, Scene, ScenePlan
from agents.scene_planner.stage_simple import write_scene_plan
from agents.scene_planner.validator import ScenePlanningValidation

__all__ = [
    "Mascot",
    "Scene",
    "ScenePlan",
    "ScenePlanningValidation",
    "write_scene_plan",
]