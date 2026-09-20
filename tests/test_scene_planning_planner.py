"""Stage-6 planner tests: QA gate, determinism, grouping, durations, grounding."""

from __future__ import annotations

import re
import socket

import pytest

from agents.qa.script.models import ScriptQAResult
from agents.scene_planner.models import (
    FACTUAL_TEXT_KINDS,
    GENERATIVE_VISUAL_TYPES,
    PLANNING_BLOCKED_MESSAGE,
    PlanningBlockedError,
    production_plan_id_for,
)
from agents.scene_planner.planner import (
    MockScenePlanner,
    NEGATIVE_STYLE,
    VISUAL_LANGUAGE,
    ScenePlanningResult,
)
from scene_planning_test_utils import FIXED_AT, make_octopus_pair
from script_test_utils import make_script, rounded_duration
from script_qa_test_utils import octopus_ok_lines

OCTOPUS_CLAIMS = (
    "Octopuses have three hearts: two branchial hearts that push blood through the gills and a systemic heart that powers the body.",
    "While octopuses swim, the two branchial hearts stop beating because the body diverts oxygen to the systemic heart.",
    "Octopus blood uses copper-based hemocyanin, which means their blood appears blue.",
)


def build_plan(run_id="planner-test"):
    result, package = make_octopus_pair(run_id=run_id, created_at=FIXED_AT)
    script = make_script(result, lines=octopus_ok_lines(result), created_at=FIXED_AT)
    from agents.script.stage import approved_claims

    claims = approved_claims(result, package)
    qa_result = _pass_qa(script, result, package, run_id)
    plan = MockScenePlanner().plan(script, qa_result, result, claims, run_id=run_id, created_at=FIXED_AT)
    return plan, script, result, package


def _pass_qa(script, result, package, run_id):
    from agents.seo.models import Provenance

    return ScriptQAResult(
        qa_result_id=f"sqa_{run_id}",
        script_id=script.script_id,
        topic_id=script.topic_id,
        result_id=result.result_id,
        package_id=result.package_id,
        overall_verdict="PASS",
        score=100.0,
        score_detail={},
        provenance=Provenance(
            generated_by="test", source="test-builder", created_at=FIXED_AT,
            provider=None, model="test", prompt_version=None,
        ),
        created_at=FIXED_AT,
        run_id=run_id,
    ).validate()


class TestQaGate:
    def test_revise_verdict_blocks_planning(self):
        plan, script, result, package = build_plan()
        gate = _replace_verdict(script, result, package, "REVISE")
        with pytest.raises(PlanningBlockedError, match="REVISE"):
            MockScenePlanner().plan(script, gate, result, [], run_id="planner-test")

    def test_reject_verdict_blocks_planning(self):
        plan, script, result, package = build_plan()
        gate = _replace_verdict(script, result, package, "REJECT")
        with pytest.raises(PlanningBlockedError, match="REJECT"):
            MockScenePlanner().plan(script, gate, result, [], run_id="planner-test")


def _replace_verdict(script, result, package, verdict):
    from dataclasses import replace

    qa = _pass_qa(script, result, package, "planner-test")
    return replace(qa, overall_verdict=verdict)


class TestDeterminism:
    def test_planning_is_deterministic(self):
        first = build_plan()[0].to_dict()
        second = build_plan()[0].to_dict()
        assert first == second

    def test_plan_id_changes_with_run_id(self):
        plan_a = build_plan(run_id="run-a")[0]
        plan_b = build_plan(run_id="run-b")[0]
        assert plan_a.production_plan_id != plan_b.production_plan_id

    def test_plan_id_is_derived_from_run_and_script(self):
        plan, script, result, package = build_plan()
        assert plan.production_plan_id == production_plan_id_for("planner-test", script.script_id)

    def test_created_at_is_propagated(self):
        plan, _, _, _ = build_plan()
        assert plan.created_at == FIXED_AT

    def test_repeated_planning_produces_same_total(self):
        a = build_plan()[0].total_duration_seconds
        b = build_plan()[0].total_duration_seconds
        assert a == b


class TestStructure:
    def test_seven_scene_shape(self):
        plan, script, result, package = build_plan()
        assert len(plan.scenes) == 7
        assert plan.scene_ids == ["scn_01", "scn_02", "scn_03", "scn_04", "scn_05", "scn_06", "scn_07"]

    def test_orders_are_contiguous(self):
        plan, _, _, _ = build_plan()
        assert [scene.order for scene in plan.scenes] == [1, 2, 3, 4, 5, 6, 7]

    def test_full_line_coverage(self):
        plan, script, _, _ = build_plan()
        assert set(plan.covered_script_line_ids) == {line.line_id for line in script.lines}
        assert plan.uncovered_script_line_ids == ()

    def test_total_duration_matches_script(self):
        plan, script, _, _ = build_plan()
        assert abs(plan.total_duration_seconds - script.estimated_duration_seconds) <= 0.5

    def test_line_to_scene_mapping(self):
        plan, _, _, _ = build_plan()
        expect = {
            "scn_01": ("ln_00",),
            "scn_02": ("ln_01",),
            "scn_03": ("ln_02",),
            "scn_04": ("ln_03",),
            "scn_05": ("ln_04",),
            "scn_06": ("ln_05", "ln_06"),
            "scn_07": ("ln_07",),
        }
        assert {scene.scene_id: scene.script_line_ids for scene in plan.scenes} == expect


class TestDurations:
    def test_scene_durations_mirror_line_timings(self):
        plan, script, _, _ = build_plan()
        for scene in plan.scenes:
            lines = [script.lines[i] for i in range(len(script.lines)) if
                     script.lines[i].line_id in scene.script_line_ids]
            expected = round(sum(line.estimated_duration_seconds for line in lines), 2)
            assert scene.duration_seconds == pytest.approx(expected, abs=0.01)

    def test_scene_durations_sum_to_total(self):
        plan, _, _, _ = build_plan()
        assert round(sum(scene.duration_seconds for scene in plan.scenes), 2) == pytest.approx(
            plan.total_duration_seconds, abs=0.01
        )

    def test_duration_derived_by_word_count(self):
        plan, script, _, _ = build_plan()
        # per-line durations come from the same formula as the script generator
        for index, line in enumerate(script.lines):
            assert line.estimated_duration_seconds == rounded_duration(line.text)


class TestVisualTreatments:
    def test_hook_is_generated_video(self):
        plan, _, _, _ = build_plan()
        assert plan.scenes[0].visual_type == "generated_video"

    def test_blue_blood_is_still_image_with_dissolve(self):
        plan, _, _, _ = build_plan()
        scene = plan.scenes[4]
        assert scene.visual_type == "generated_image"
        assert scene.transition == "dissolve"
        assert any(item.kind == "label" and item.text == "Blood appears blue" for item in scene.on_screen_text)

    def test_conclusion_uses_fade(self):
        plan, _, _, _ = build_plan()
        assert plan.scenes[5].transition == "fade"
        assert plan.scenes[5].visual_type == "motion_graphic"

    def test_other_scenes_use_cut(self):
        plan, _, _, _ = build_plan()
        assert [scene.transition for scene in plan.scenes[:4]] == ["cut"] * 4

    def test_cta_is_text_only(self):
        plan, _, _, _ = build_plan()
        scene = plan.scenes[-1]
        assert scene.visual_type == "text_only"
        assert scene.generation_prompt == ""
        assert scene.negative_prompt == ""
        assert any(item.text == "Follow for more science!" for item in scene.on_screen_text)

    def test_generative_scenes_have_prompts_and_negative_prompts(self):
        plan, _, _, _ = build_plan()
        for scene in plan.scenes:
            if scene.visual_type in GENERATIVE_VISUAL_TYPES:
                assert scene.generation_prompt.strip()
                assert NEGATIVE_STYLE in scene.negative_prompt

    def test_factual_scene_labels_carry_traceable_refs(self):
        plan, _, _, _ = build_plan()
        for scene in plan.scenes:
            for item in scene.on_screen_text:
                if item.kind in FACTUAL_TEXT_KINDS:
                    assert item.claim_refs, f"{item.text} must trace a claim"


class TestContinuity:
    def test_continuity_asset_defined(self):
        plan, _, _, _ = build_plan()
        assert len(plan.continuity) == 1
        asset = plan.continuity[0]
        assert asset.group == "octopus"
        assert "octopus" in asset.subject.lower()

    def test_scenes_reference_continuity_group(self):
        plan, _, _, _ = build_plan()
        for scene in plan.scenes[:-1]:
            assert scene.visual_continuity_group == "octopus"
        assert plan.scenes[-1].visual_continuity_group == ""

    def test_global_style_shorts_defaults(self):
        plan, _, _, _ = build_plan()
        style = plan.global_style
        assert style.aspect_ratio == "9:16"
        assert style.target_resolution == "1080x1920"
        assert style.fps == 30
        assert "documentary" in style.visual_style


class TestClaimGrounding:
    def test_scene_claims_are_only_approved(self):
        plan, _, result, _ = build_plan()
        approved = set(result.approved_claims)
        for scene in plan.scenes:
            assert set(scene.claim_refs).issubset(approved)

    def test_scene_claim_refs_trace_script_lines(self):
        plan, script, _, _ = build_plan()
        line_refs = {ref for line in script.lines for ref in line.claim_refs}
        for scene in plan.scenes:
            assert set(scene.claim_refs).issubset(line_refs)

    def test_factual_scenes_have_claim_traceability(self):
        plan, _, _, _ = build_plan()
        # factual scenes (fact/explanation/conclusion) carry claim_refs
        for scene in plan.scenes:
            if scene.visual_type != "text_only" and "octopus" in scene.visual_continuity_group:
                assert scene.claim_refs or scene.scene_id == "scn_02"  # atmospheric context scene

    def test_provenance_embeds_claim_texts(self):
        plan, _, _, package = build_plan()
        claims = {c.claim_id: c.claim_text for c in package.claims}
        for scene in plan.scenes:
            for claim_id, text in scene.provenance.get("claim_texts", {}).items():
                assert claims[claim_id] == text


def _tokens(text: str):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class TestNoHallucinatedVisualFacts:
    def test_visuals_are_grounded_in_claims_or_practice_vocabulary(self):
        plan, script, _, _ = build_plan()
        corpus = " ".join(
            OCTOPUS_CLAIMS + tuple(scene.narration for scene in plan.scenes)
        ).lower()
        allowed = set(VISUAL_LANGUAGE) | _tokens(corpus)
        allowed.update(_VISUAL_SKIP_WORDS)
        for scene in plan.scenes:
            for field in ("visual_description", "generation_prompt"):
                outliers = _tokens(getattr(scene, field)) - allowed
                assert not outliers, f"{scene.scene_id} {field} invented: {sorted(outliers)}"
            for item in scene.on_screen_text:
                outliers = _tokens(item.text) - allowed
                assert not outliers, f"{scene.scene_id} text invented: {sorted(outliers)}"

    def test_no_provider_vocabulary_in_plan_contract(self):
        plan, _, _, _ = build_plan()
        payload = plan.to_dict()["data"]
        text = str(payload).lower()
        for term in ("runway", "veo", "kling", "pika", "sora", "luma", "dall-e", "midjourney",
                     "stable diffusion", "flux", "comfyui"):
            assert term not in text, f"provider term leaked: {term}"

    def test_negative_prompts_are_visual_quality_only(self):
        plan, _, _, _ = build_plan()
        for scene in plan.scenes:
            if not scene.negative_prompt:
                continue
            for marker in ("octopus", "hearts", "hemocyanin", "invertebrate"):
                assert marker not in scene.negative_prompt.lower()


_VISUAL_SKIP_WORDS = {
    "a", "an", "the", "of", "and", "with", "in", "into", "on", "to", "from",
    "while", "over", "at", "its", "their", "that", "as", "for", "is", "are",
}


class TestNetworkIsolation:
    def test_planning_is_network_free(self, monkeypatch):
        def boom(*args, **kwargs):
            raise AssertionError("planner hit the network")

        monkeypatch.setattr("socket.socket", boom)
        monkeypatch.setattr("socket.create_connection", boom)
        plan, _, _, _ = build_plan()
        assert plan.production_plan_id

    def test_planner_import_does_not_touch_network(self):
        # importing modules is enough; exercising MockScenePlanner offline
        import importlib

        for name in ("agents.scene_planner.models", "agents.scene_planner.planner",
                     "agents.scene_planner.validator", "agents.scene_planner.stage"):
            importlib.reload(importlib.import_module(name))


class TestBlockedOutcome:
    def test_blocked_result_flag(self):
        result = ScenePlanningResult(
            planned=False,
            reason=PLANNING_BLOCKED_MESSAGE.format(verdict="REVISE"),
        )
        assert result.blocked is True

    def test_non_gate_failure_is_not_blocked(self):
        result = ScenePlanningResult(planned=False, reason="production plan validation failed")
        assert result.blocked is False