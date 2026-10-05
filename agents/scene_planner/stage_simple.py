"""Simplified scene planning."""

from __future__ import annotations

import json
from pathlib import Path

from agents.scene_planner.scene_models import ScenePlan


def write_scene_plan(plan: ScenePlan, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
    return out_path

