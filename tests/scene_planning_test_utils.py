"""Shared builders/stubs for the Stage-6 (scene planning) test suite.

Everything here is synthetic and MOCK - same rule as the Stage-3/4/5 builders.
The helpers build exact PASS/blocked/invalid planning scenarios offline, using
the same octopus claim set and the same line-grouping the mock planner uses.
"""

from __future__ import annotations

from agents.research.models import FactClaim
from agents.scene_planner.models import scene_id_for
from agents.seo.models import Provenance
from fact_check_test_utils import FIXED_AT
from script_qa_test_utils import make_octopus_pair, octopus_ok_lines
from script_test_utils import make_script


def build_healthy_plan(
    result=None,
    package=None,
    script=None,
    qa_result=None,
    run_id="scene-test",
    created_at=FIXED_AT,
):
    """Deterministically produce a HEALTHY ProductionPlan (mock planner path).

    Mirrors the MockScenePlanner's grouping so tests can predict scene names,
    durations, claim refs, and text without touching the planner directly.
    """
    from agents.scene_planner.planner import MockScenePlanner

    if result is None or package is None:
        result, package = make_octopus_pair(run_id=run_id, created_at=created_at)
    if script is None:
        script = make_script(result, lines=octopus_ok_lines(result), created_at=created_at)
    if qa_result is None:
        qa_result = _pass_qa(script, result, package, run_id, created_at)
    claims = _approved_records(result, package)
    return MockScenePlanner().plan(
        script,
        qa_result,
        result,
        claims,
        run_id=run_id,
        created_at=created_at,
    )


def _pass_qa(script, result, package, run_id, created_at):
    from agents.qa.script.models import ScriptQAResult

    # A minimal PASS ScriptQAResult is enough for the planner gate.
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
            generated_by="test",
            source="test-builder",
            created_at=created_at,
            provider=None,
            model="test",
            prompt_version=None,
        ),
        created_at=created_at,
        run_id=run_id,
    ).validate()


def _approved_records(result, package):
    from agents.script.stage import approved_claims

    return approved_claims(result, package)


def claim_index(package) -> dict[str, FactClaim]:
    return {claim.claim_id: claim for claim in package.claims}


def scene_ids(count: int) -> tuple[str, ...]:
    return tuple(scene_id_for(index) for index in range(1, count + 1))