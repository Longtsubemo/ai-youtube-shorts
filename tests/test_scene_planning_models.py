"""Stage-6 model tests: ProductionPlan envelope, sub-model rules, ids, round-trip."""

from __future__ import annotations

from dataclasses import replace

import pytest

from agents.scene_planner.models import (
    DEFAULT_ASPECT_RATIO,
    DEFAULT_FPS,
    DEFAULT_RESOLUTION,
    FACTUAL_TEXT_KINDS,
    GENERATIVE_VISUAL_TYPES,
    ON_SCREEN_TEXT_KINDS,
    TRANSITIONS,
    VISUAL_TYPES,
    AssetRequirement,
    ContinuityAsset,
    GlobalStyle,
    OnScreenText,
    PlanningError,
    ProductionPlan,
    Scene,
    ScenePlanningConfig,
    production_plan_id_for,
    scene_id_for,
)
from agents.scene_planner.models import SCHEMA, STAGE
from pipeline.state import Stage
from scene_planning_test_utils import FIXED_AT, build_healthy_plan


class TestIdentifiers:
    def test_production_plan_id_is_deterministic_and_prefixed(self):
        assert production_plan_id_for("run-a", "sc_x") == production_plan_id_for("run-a", "sc_x")
        first = production_plan_id_for("run-a", "sc_x")
        assert first.startswith("pp_")
        assert len(first) == 15  # "pp_" + 12 hex
        assert production_plan_id_for("run-b", "sc_x") != first
        assert production_plan_id_for("run-a", "sc_y") != first

    def test_scene_id_formatting(self):
        assert scene_id_for(1) == "scn_01"
        assert scene_id_for(12) == "scn_12"


class TestOnScreenText:
    def test_valid_round_trip(self):
        item = OnScreenText(kind="label", text="Three hearts", claim_refs=("cl_a",)).validate()
        clone = OnScreenText.from_dict(item.to_dict())
        assert clone == item

    def test_unknown_kind_rejected(self):
        with pytest.raises(PlanningError):
            OnScreenText(kind="caption2", text="x").validate()

    def test_missing_text_rejected(self):
        with pytest.raises(PlanningError):
            OnScreenText(kind="headline", text=" ").validate()

    def test_factual_kinds_and_allowed_kinds(self):
        assert FACTUAL_TEXT_KINDS == ("label", "caption")
        assert set(FACTUAL_TEXT_KINDS).issubset(ON_SCREEN_TEXT_KINDS)


class TestAssetRequirement:
    def test_valid_round_trip(self):
        asset = AssetRequirement(
            asset_type="generated_video", subject="octopus swimming", quantity=2, continuity_group="octopus"
        ).validate()
        clone = AssetRequirement.from_dict(asset.to_dict())
        assert clone == asset

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"asset_type": "hologram"},
            {"subject": "  "},
            {"quantity": 0},
        ],
    )
    def test_invalid_fields_rejected(self, kwargs):
        fields = {"asset_type": "generated_video", "subject": "octopus", "quantity": 1}
        fields.update(kwargs)
        with pytest.raises(PlanningError):
            AssetRequirement(**fields).validate()


class TestContinuityAsset:
    def test_valid_round_trip(self):
        asset = ContinuityAsset(
            group="octopus",
            subject="common octopus",
            appearance="realistic reddish-brown",
        ).validate()
        clone = ContinuityAsset.from_dict(asset.to_dict())
        assert clone == asset

    def test_missing_group_rejected(self):
        with pytest.raises(PlanningError):
            ContinuityAsset(group="", subject="octopus").validate()

    def test_missing_subject_rejected(self):
        with pytest.raises(PlanningError):
            ContinuityAsset(group="octopus", subject="").validate()


class TestGlobalStyle:
    def test_shorts_defaults(self):
        style = GlobalStyle().validate()
        assert style.aspect_ratio == DEFAULT_ASPECT_RATIO == "9:16"
        assert style.target_resolution == DEFAULT_RESOLUTION == "1080x1920"
        assert style.fps == DEFAULT_FPS == 30
        assert style.default_transition == "cut"

    def test_bad_transition_rejected(self):
        with pytest.raises(PlanningError):
            GlobalStyle(default_transition="swipe").validate()

    def test_non_positive_fps_rejected(self):
        with pytest.raises(PlanningError):
            GlobalStyle(fps=0).validate()

    def test_round_trip(self):
        style = GlobalStyle(consistency_notes="same octopus everywhere").validate()
        assert GlobalStyle.from_dict(style.to_dict()) == style


class TestScene:
    def test_valid_round_trip(self):
        scene = Scene(
            scene_id="scn_01",
            order=1,
            duration_seconds=4.0,
            script_line_ids=("ln_00",),
            narration="hello there",
            visual_type="generated_video",
            visual_description="a nice octopus",
        ).validate()
        clone = Scene.from_dict(scene.to_dict())
        assert clone == scene

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"scene_id": ""},
            {"order": 0},
            {"duration_seconds": 0.0},
            {"visual_type": "hologram"},
            {"visual_description": ""},
            {"transition": "swipe"},
        ],
    )
    def test_invalid_scene_rejected(self, kwargs):
        fields = dict(
            scene_id="scn_01",
            order=1,
            duration_seconds=4.0,
            script_line_ids=("ln_00",),
            narration="hello there",
            visual_type="generated_video",
            visual_description="a nice octopus",
        )
        fields.update(kwargs)
        with pytest.raises(PlanningError):
            Scene(**fields).validate()

    def test_transition_scene_does_not_require_narration(self):
        Scene(
            scene_id="scn_x",
            order=1,
            duration_seconds=0.5,
            script_line_ids=(),
            narration="",
            visual_type="transition",
            visual_description="a quick fade",
        ).validate()


class TestScenePlanningConfig:
    def test_default_validate(self):
        config = ScenePlanningConfig().validate()
        assert config.duration_tolerance_seconds == 0.5
        assert config.require_full_coverage is True
        assert config.caption_traceability is True
        assert set(config.generation_prompt_required_types) == set(GENERATIVE_VISUAL_TYPES)

    def test_bad_tolerance_rejected(self):
        with pytest.raises(ValueError):
            ScenePlanningConfig(duration_tolerance_seconds=0).validate()

    def test_bad_denylist_item_rejected(self):
        with pytest.raises(ValueError):
            ScenePlanningConfig(provider_hint_denylist=("sora", "")).validate()


class TestProductionPlan:
    def test_healthy_plan_round_trips(self):
        plan = build_healthy_plan()
        clone = ProductionPlan.from_json(plan.to_json())
        assert clone.to_dict() == plan.to_dict()

    def test_envelope(self):
        plan = build_healthy_plan()
        assert plan.schema == SCHEMA == "ProductionPlan"
        assert plan.stage == STAGE == Stage.SCENE_PLANNING.value == "scene_planning"
        assert plan.schema_version == 1
        assert plan.produced_by["model"] == "MockScenePlanner"
        assert plan.provenance.generated_by == "scene_planner_agent_v1"

    def test_empty_scenes_rejected(self):
        plan = replace(build_healthy_plan(), scenes=())
        with pytest.raises(PlanningError):
            plan.validate()

    def test_duplicate_scene_ids_rejected(self):
        plan = build_healthy_plan()
        dup = replace(plan.scenes[1], scene_id=plan.scenes[0].scene_id)
        broke = replace(plan, scenes=(plan.scenes[0], dup) + plan.scenes[2:])
        with pytest.raises(PlanningError, match="duplicate scene_id"):
            broke.validate()

    def test_bad_order_rejected(self):
        plan = build_healthy_plan()
        bad = list(plan.scenes)
        bad[2] = replace(bad[2], order=9)
        with pytest.raises(PlanningError, match="contiguous"):
            replace(plan, scenes=tuple(bad)).validate()

    def test_total_duration_mismatch_rejected(self):
        plan = build_healthy_plan()
        with pytest.raises(PlanningError, match="does not match"):
            replace(plan, total_duration_seconds=999.0).validate()

    def test_covered_uncovered_overlap_rejected(self):
        plan = build_healthy_plan()
        with pytest.raises(PlanningError, match="overlaps"):
            replace(plan, uncovered_script_line_ids=(plan.covered_script_line_ids[0],)).validate()

    def test_missing_provenance_rejected(self):
        plan = build_healthy_plan()
        with pytest.raises(PlanningError, match="provenance"):
            replace(plan, provenance=None).validate()


class TestEnums:
    def test_visual_types_are_controlled(self):
        assert "generated_video" in VISUAL_TYPES
        assert "text_only" in VISUAL_TYPES
        assert "sora_video" not in VISUAL_TYPES

    def test_transitions_are_controlled(self):
        assert "cut" in TRANSITIONS
        assert "fade" in TRANSITIONS
        assert "match_cut" in TRANSITIONS
        assert "ripple" not in TRANSITIONS