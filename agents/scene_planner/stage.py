"""Stage 4: Scene Planning (content stage).

Consumes the Stage 3 `script.json` artifact and produces a flat
`scene_plan.json`. This is a visual planning stage: narration is never
rewritten, only split across scenes. One script section may become
multiple scenes. No sources, no citations, no QA, no PASS/FAIL.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from agents.script.script_models import Script
from agents.scene_planner.scene_models import (
    CHANNEL_IMAGE_STYLE,
    MASCOT_ASSET_PATH,
    Scene,
    ScenePlan,
    ScenePlanningError,
)
from agents.scene_planner.validator import validate_scene_plan_generation
from agents.seo.config import TopicSelectionConfig


def load_script(path: str | Path) -> Script:
    return Script.from_json(Path(path).read_text(encoding="utf-8"))


def write_scene_plan(plan: ScenePlan, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
    return out_path


def _norm(s: str) -> str:
    return " ".join(s.split()).strip()


def build_scene_plan_context(script: Script) -> dict:
    if not script.topic_id or not script.topic_id.strip():
        raise ScenePlanningError("script has no topic_id")
    if not script.narration or not script.narration.strip():
        raise ScenePlanningError("script has no narration")
    if not script.sections:
        raise ScenePlanningError("script has no sections")
    return {
        "script": script,
    }


def run_scene_planning(
    script: Script,
    provider,
    *,
    run_id: str = "",
    created_at: str | None = None,
    config: TopicSelectionConfig | None = None,
) -> ScenePlan:
    context = build_scene_plan_context(script)
    plan = provider.generate(context)
    updates: dict = {
        "run_id": run_id or script.run_id,
        "topic_id": plan.topic_id or script.topic_id,
        "title": plan.title or script.title,
        "target_duration_seconds": plan.target_duration_seconds or script.duration_target_seconds,
        "channel_id": plan.channel_id or (config.channel.channel_id if config else (script.channel_id or "finance_mascot")),
        "mascot_asset_path": plan.mascot_asset_path or MASCOT_ASSET_PATH,
        "channel_image_style": plan.channel_image_style or CHANNEL_IMAGE_STYLE,
    }
    if created_at is not None:
        updates["created_at"] = created_at
    plan = replace(plan, **updates)
    if plan.total_scene_duration_seconds <= 0:
        plan = replace(plan, total_scene_duration_seconds=plan.estimated_seconds)
    plan = plan.validate()

    # SDXL/CLIP image-prompt gate: fail clearly on invalid generation_prompt
    # values (empty, over the hard ceiling, forbidden concepts, narration or
    # style leakage). Nothing is ever silently truncated.
    validation = validate_scene_plan_generation(plan)
    if not validation.passed:
        details = "; ".join(
            f"[{issue.code}] {issue.message}" for issue in validation.errors
        )
        raise ScenePlanningError(f"invalid generation_prompt: {details}")
    return plan
