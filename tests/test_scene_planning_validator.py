"""Stage-6 validator tests: rule-by-rule checks (r01..r20) of ProductionPlans."""

from __future__ import annotations

from dataclasses import replace

from agents.qa.fact_checker.models import ClaimVerdict
from agents.scene_planner.models import OnScreenText, Scene
from agents.scene_planner.validator import ValidationIssue, validate_production_plan
from scene_planning_test_utils import build_healthy_plan, make_octopus_pair

import script_qa_test_utils as qa_utils


def make_r11_pair():
    """Octopus pair where one claim is known (has a verdict) but not approved.

    A 'known but not approved' claim should trip rule r11 (defense in depth:
    unapproved-but-recorded claims must never reach a scene).
    """
    from agents.qa.fact_checker.models import FactCheckResult

    result, package = make_octopus_pair()
    extra = ClaimVerdict(
        claim_id="cl_known_unused",
        verdict="REVIEW",
        evidence_ids=(),
        supporting_source_ids=(),
        contradiction_source_ids=(),
        rationale="mock: recorded but never approved",
        confidence=0.4,
        critical=False,
    ).validate()
    result = replace(result, claim_results=result.claim_results + (extra,))
    return result, package


def scene_issue_ids(validation):
    return {issue.rule_id for issue in validation.errors}


class TestHealthyPlan:
    def test_mock_plan_is_valid(self):
        assert build_healthy_plan().validate() is not None

    def test_validator_passes_healthy_plan(self):
        plan, script, result = repair_plan()
        validation = validate_production_plan(plan, script, result)
        assert validation.passed
        assert validation.errors == ()


def repair_plan(result=None):
    """Healthy inputs for validator tests: (plan, script, result)."""
    from script_test_utils import make_script
    from script_qa_test_utils import octopus_ok_lines

    result, package = result or make_octopus_pair(run_id="scene-test")
    script = make_script(result, lines=octopus_ok_lines(result), created_at=qa_utils.FIXED_AT)
    plan = build_healthy_plan(result=result, package=package, script=script)
    return plan, script, result


class TestStructuralRules:
    def test_r01_empty_plan_blocks_early(self):
        plan, script, result = repair_plan()
        broke = replace(plan, scenes=())
        validation = validate_production_plan(broke, script, result)
        assert not validation.passed
        assert "r01" in scene_issue_ids(validation)
        # r20 must be skipped for structurally broken plans
        assert "r20" not in scene_issue_ids(validation)

    def test_r02_duplicate_scene_ids(self):
        plan, script, result = repair_plan()
        dup = replace(plan.scenes[1], scene_id=plan.scenes[0].scene_id)
        broke = replace(plan, scenes=(plan.scenes[0], dup) + plan.scenes[2:])
        validation = validate_production_plan(broke, script, result)
        assert "r02" in scene_issue_ids(validation)
        assert "r20" not in scene_issue_ids(validation)

    def test_r03_scene_ordering(self):
        plan, script, result = repair_plan()
        bad = list(plan.scenes)
        bad[2] = replace(bad[2], order=9)
        validation = validate_production_plan(replace(plan, scenes=tuple(bad)), script, result)
        assert "r03" in scene_issue_ids(validation)

    def test_r04_non_positive_duration(self):
        plan, script, result = repair_plan()
        broke = replace(plan, scenes=(replace(plan.scenes[0], duration_seconds=-1.0),) + plan.scenes[1:])
        validation = validate_production_plan(replace(broke, total_duration_seconds=0.0), script, result)
        assert "r04" in scene_issue_ids(validation)

    def test_r05_total_duration_consistency(self):
        plan, script, result = repair_plan()
        broke = replace(
            plan,
            scenes=(replace(plan.scenes[0], duration_seconds=plan.scenes[0].duration_seconds + 1.0),) + plan.scenes[1:],
        )
        validation = validate_production_plan(broke, script, result)
        assert "r05" in scene_issue_ids(validation)

    def test_r06_duration_within_script_tolerance(self):
        plan, script, result = repair_plan()
        # keep internal totals consistent, but push total outside the script window
        drift = script.estimated_duration_seconds + 5.0 - plan.total_duration_seconds
        scenes = (replace(plan.scenes[0], duration_seconds=plan.scenes[0].duration_seconds + drift),) + plan.scenes[1:]
        total = plan.total_duration_seconds + drift
        broke = replace(plan, scenes=scenes, total_duration_seconds=total)
        validation = validate_production_plan(broke, script, result)
        assert "r06" in scene_issue_ids(validation)


class TestCoverageRules:
    def test_r07_scene_without_lines_is_rejected_unless_transition(self):
        plan, script, result = repair_plan()
        broke = replace(plan, scenes=(replace(plan.scenes[0], script_line_ids=()),) + plan.scenes[1:])
        validation = validate_production_plan(broke, script, result)
        assert "r07" in scene_issue_ids(validation)

    def test_transition_scene_may_have_no_lines(self):
        plan, script, result = repair_plan()
        transition = Scene(
            scene_id="scn_08",
            order=8,
            duration_seconds=0.5,
            script_line_ids=(),
            narration="",
            visual_type="transition",
            visual_description="a quick dissolve",
        ).validate()
        total = round(plan.total_duration_seconds + 0.5, 2)
        broke = replace(
            plan,
            scenes=plan.scenes + (transition,),
            total_duration_seconds=total,
        )
        validation = validate_production_plan(broke, script, result)
        assert "r07" not in scene_issue_ids(validation)

    def test_r08_uncovered_line_errors_when_required(self):
        plan, script, result = repair_plan()
        # Remap the hook scene onto the CTA line (same 4.0s slot, no total
        # drift) so script line ln_00 is left uncovered.
        broke = replace(
            plan,
            scenes=(replace(plan.scenes[0], script_line_ids=("ln_07",)),) + plan.scenes[1:],
        )
        validation = validate_production_plan(broke, script, result)
        assert "r08" in scene_issue_ids(validation)
        assert not validation.passed

    def test_r08_uncovered_line_is_warning_when_allowed(self):
        from agents.scene_planner.models import ScenePlanningConfig

        plan, script, result = repair_plan()
        broke = replace(
            plan,
            scenes=(replace(plan.scenes[0], script_line_ids=("ln_07",)),) + plan.scenes[1:],
        )
        config = ScenePlanningConfig(require_full_coverage=False)
        validation = validate_production_plan(broke, script, result, config=config)
        warning_rules = {issue.rule_id for issue in validation.warnings}
        assert "r08" in warning_rules
        assert validation.passed  # WARNING only

    def test_r09_unknown_script_line_reference(self):
        plan, script, result = repair_plan()
        pieces = list(plan.covered_script_line_ids)
        broke = replace(plan, scenes=(replace(plan.scenes[0], script_line_ids=("ln_nope",)),) + plan.scenes[1:])
        validation = validate_production_plan(broke, script, result)
        assert "r09" in scene_issue_ids(validation)

    def test_all_script_lines_are_covered_by_default_mock_plan(self):
        plan, script, result = repair_plan()
        covered = {line_id for scene in plan.scenes for line_id in scene.script_line_ids}
        assert covered == {line.line_id for line in script.lines}


class TestClaimRules:
    def test_r10_unknown_claim_reference(self):
        plan, script, result = repair_plan()
        broke = replace(plan, scenes=(replace(plan.scenes[0], claim_refs=("cl_unknown",)),) + plan.scenes[1:])
        validation = validate_production_plan(broke, script, result)
        assert "r10" in scene_issue_ids(validation)

    def test_r11_known_but_unapproved_claim_reference(self):
        plan, script, result = repair_plan(make_r11_pair())
        broke = replace(plan, scenes=(replace(plan.scenes[0], claim_refs=("cl_known_unused",)),) + plan.scenes[1:])
        validation = validate_production_plan(broke, script, result)
        assert "r11" in scene_issue_ids(validation)

    def test_r12_rejected_claim_reference(self):
        plan, script, result = repair_plan()
        rejected_id = result.rejected_claims[0]
        broke = replace(plan, scenes=(replace(plan.scenes[0], claim_refs=(rejected_id,)),) + plan.scenes[1:])
        validation = validate_production_plan(broke, script, result)
        assert "r12" in scene_issue_ids(validation)

    def test_r13_review_claim_reference(self):
        plan, script, result = repair_plan()
        review_id = result.review_items[0]
        broke = replace(plan, scenes=(replace(plan.scenes[0], claim_refs=(review_id,)),) + plan.scenes[1:])
        validation = validate_production_plan(broke, script, result)
        assert "r13" in scene_issue_ids(validation)

    def test_r14_factual_scene_without_claim_refs(self):
        plan, script, result = repair_plan()
        broke = replace(plan, scenes=plan.scenes[:2] + (replace(plan.scenes[2], claim_refs=()),) + plan.scenes[3:])
        validation = validate_production_plan(broke, script, result)
        assert "r14" in scene_issue_ids(validation)

    def test_r14_missing_claim_ref_for_factual_line(self):
        from script_test_utils import rounded_duration

        plan, script, result = repair_plan()
        # Merge ln_02 (fact, refs {a0}) + ln_03 (explanation, refs {a1}) into
        # one factual scene that only carries a0 -> a1 is silently dropped.
        merged = replace(
            plan.scenes[2],
            script_line_ids=(script.lines[2].line_id, script.lines[3].line_id),
            duration_seconds=round(
                rounded_duration(script.lines[2].text) + rounded_duration(script.lines[3].text), 2
            ),
            claim_refs=(plan.scenes[2].claim_refs[0],),
            narration=script.lines[2].text + " " + script.lines[3].text,
        )
        broke = replace(plan, scenes=plan.scenes[:2] + (merged,) + plan.scenes[3:])
        validation = validate_production_plan(broke, script, result)
        assert filtered(validation, "r14")

    def test_r14_atmospheric_scene_may_be_without_refs(self):
        plan, script, result = repair_plan()
        # context scene (ln_01) has no factual purpose -> r14 never fires there
        assert plan.scenes[1].claim_refs == ()
        validation = validate_production_plan(plan, script, result)
        assert "r14" not in scene_issue_ids(validation)

    def test_r15_factual_text_without_traceability(self):
        plan, script, result = repair_plan()
        no_trace = (OnScreenText(kind="label", text="Three hearts", claim_refs=()),)
        broke = replace(plan, scenes=plan.scenes[:2] + (replace(plan.scenes[2], on_screen_text=no_trace),) + plan.scenes[3:])
        validation = validate_production_plan(broke, script, result)
        assert "r15" in scene_issue_ids(validation)

    def test_r15_disabled_when_caption_traceability_off(self):
        from agents.scene_planner.models import ScenePlanningConfig

        plan, script, result = repair_plan()
        no_trace = (OnScreenText(kind="label", text="Three hearts", claim_refs=()),)
        broke = replace(plan, scenes=plan.scenes[:2] + (replace(plan.scenes[2], on_screen_text=no_trace),) + plan.scenes[3:])
        validation = validate_production_plan(broke, script, result, config=ScenePlanningConfig(caption_traceability=False))
        assert "r15" not in scene_issue_ids(validation)

    def test_stylistic_headline_without_refs_is_fine(self):
        plan, script, result = repair_plan()
        validation = validate_production_plan(plan, script, result)
        # CTA headline (scn_07) carries no claim refs and must pass
        assert plan.scenes[-1].on_screen_text[0].kind == "headline"
        assert "r15" not in scene_issue_ids(validation)


class TestPresentationRules:
    def test_r16_generative_type_requires_prompt(self):
        plan, script, result = repair_plan()
        broke = replace(plan, scenes=(replace(plan.scenes[0], generation_prompt=""),) + plan.scenes[1:])
        validation = validate_production_plan(broke, script, result)
        assert "r16" in scene_issue_ids(validation)

    def test_r16_text_only_does_not_require_prompt(self):
        plan, script, result = repair_plan()
        assert plan.scenes[-1].visual_type == "text_only"
        validation = validate_production_plan(plan, script, result)
        assert "r16" not in scene_issue_ids(validation)

    def test_r17_rejects_invalid_visual_type(self):
        plan, script, result = repair_plan()
        broke = replace(plan, scenes=(replace(plan.scenes[0], visual_type="hologram"),) + plan.scenes[1:])
        validation = validate_production_plan(broke, script, result)
        assert "r17" in scene_issue_ids(validation)

    def test_r18_provider_word_in_prompt(self):
        plan, script, result = repair_plan()
        broke = replace(plan, scenes=(replace(plan.scenes[0], generation_prompt="render via Sora API"),) + plan.scenes[1:])
        validation = validate_production_plan(broke, script, result)
        assert "r18" in scene_issue_ids(validation)

    def test_r18_provider_key_in_scene_provenance(self):
        plan, script, result = repair_plan()
        prov = dict(plan.scenes[0].provenance)
        prov["provider"] = "ignored-ish"
        broke = replace(plan, scenes=(replace(plan.scenes[0], provenance=prov),) + plan.scenes[1:])
        validation = validate_production_plan(broke, script, result)
        assert "r18" in scene_issue_ids(validation)

    def test_r19_missing_provenance(self):
        plan, script, result = repair_plan()
        broke = replace(plan, provenance=None)
        validation = validate_production_plan(broke, script, result)
        assert "r19" in scene_issue_ids(validation)


class TestRoundTripRule:
    def test_r20_passes_for_healthy_plan(self):
        plan, script, result = repair_plan()
        validation = validate_production_plan(plan, script, result)
        assert "r20" not in scene_issue_ids(validation)
        assert validation.passed

    def test_r20_skipped_after_structural_failures(self):
        plan, script, result = repair_plan()
        broke = replace(plan, scenes=())
        validation = validate_production_plan(broke, script, result)
        assert "r20" not in scene_issue_ids(validation)


class TestValidationIssueModel:
    def test_issue_dict(self):
        issue = ValidationIssue(rule_id="r07", severity="ERROR", message="no lines", scene_id="scn_01")
        payload = issue.to_dict()
        assert payload["rule_id"] == "r07"
        assert payload["severity"] == "ERROR"
        assert payload["scene_id"] == "scn_01"
        assert payload["message"] == "no lines"

    def test_errors_warnings_split(self):
        plan, script, result = repair_plan()
        broke = replace(plan, scenes=(replace(plan.scenes[2], claim_refs=()),) + (plan.scenes[:2] + plan.scenes[3:]))
        validation = validate_production_plan(broke, script, result)
        assert validation.errors
        assert not validation.passed


def filtered(validation, rule_id):
    return next((issue for issue in validation.errors if issue.rule_id == rule_id), None)