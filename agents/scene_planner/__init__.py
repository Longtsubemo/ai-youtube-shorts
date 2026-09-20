"""Stage 6 - Scene Planning: approved script -> provider-independent ProductionPlan.

Docs: docs/DATA_CONTRACTS.md §10, docs/PROJECT_PROGRESS.md Stage 6.
CLI: python -m agents.scene_planner.

NOTE: this package imports only the light models/validator surface here. The
planner, stage runner, and CLI live in `agents.scene_planner.planner` /
`agents.scene_planner.stage` and are imported by submodule path so that
`python -m agents.scene_planner` runs without a double-import RuntimeWarning.
"""
from __future__ import annotations

from agents.scene_planner.models import (
    DEFAULT_ASPECT_RATIO,
    DEFAULT_FPS,
    DEFAULT_RESOLUTION,
    FACTUAL_TEXT_KINDS,
    GENERATIVE_VISUAL_TYPES,
    ON_SCREEN_TEXT_KINDS,
    PLAN_CONFIG_VERSION,
    PRODUCED_BY,
    TRANSITIONS,
    VISUAL_TYPES,
    AssetRequirement,
    ContinuityAsset,
    GlobalStyle,
    OnScreenText,
    PlanningBlockedError,
    PlanningError,
    ProductionPlan,
    Scene,
    ScenePlanningConfig,
    production_plan_id_for,
    scene_id_for,
)
from agents.scene_planner.validator import (
    ScenePlanningValidation,
    ValidationIssue,
    validate_production_plan,
)

__all__ = [
    "AssetRequirement",
    "ContinuityAsset",
    "DEFAULT_ASPECT_RATIO",
    "DEFAULT_FPS",
    "DEFAULT_RESOLUTION",
    "FACTUAL_TEXT_KINDS",
    "GENERATIVE_VISUAL_TYPES",
    "GlobalStyle",
    "ON_SCREEN_TEXT_KINDS",
    "OnScreenText",
    "PLAN_CONFIG_VERSION",
    "PRODUCED_BY",
    "PlanningBlockedError",
    "PlanningError",
    "ProductionPlan",
    "Scene",
    "ScenePlanningConfig",
    "ScenePlanningValidation",
    "TRANSITIONS",
    "VISUAL_TYPES",
    "ValidationIssue",
    "production_plan_id_for",
    "scene_id_for",
    "validate_production_plan",
]