"""Shared builders/stubs for the Stage-7 (video generation) test suite.

Everything here is synthetic and MOCK - same rule as the Stage-3..6 builders:
no source stands behind these claims, nothing is asserted true, and the mock
video records we build always carry `mock=True` with an empty URI so nobody
mistakes them for produced video. The helpers build exact healthy / retryable /
non-retryable / pending scenarios offline, without touching a provider.

NOTE: this builder deliberately re-implements a healthy ProductionPlan
construction locally instead of importing `scene_planning_test_utils`, so the
Stage-7 suite can never break because a Stage-6 helper changed.
"""

from __future__ import annotations

from agents.qa.script.models import ScriptQAResult
from agents.scene_planner.planner import MockScenePlanner
from agents.seo.models import Provenance
from fact_check_test_utils import FIXED_AT
from script_qa_test_utils import make_octopus_pair, octopus_ok_lines
from script_test_utils import make_script


def make_healthy_plan(
    run_id: str = "video-test",
    created_at: str = FIXED_AT,
):
    """Deterministically produce a HEALTHY 7-scene ProductionPlan.

    Mirrors the MockScenePlanner path exactly (approved claims + PASS QA
    verdict + octopus line set), so tests can predict scene ids, orders,
    durations, and the derived video-generation centerpiece ids.
    """
    from agents.script.stage import approved_claims

    result, package = make_octopus_pair(run_id=run_id, created_at=created_at)
    script = make_script(result, lines=octopus_ok_lines(result), created_at=created_at)
    qa_result = _pass_qa(script, result, run_id, created_at)
    claims = approved_claims(result, package)
    return MockScenePlanner().plan(
        script,
        qa_result,
        result,
        claims,
        run_id=run_id,
        created_at=created_at,
    )


def _pass_qa(script, result, run_id: str, created_at: str) -> ScriptQAResult:
    """A minimal PASS ScriptQAResult is enough for the planner gate."""
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