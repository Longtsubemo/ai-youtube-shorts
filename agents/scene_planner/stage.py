"""Stage-6 entrypoint: Script + ScriptQAResult + FactCheckResult -> ProductionPlan.

Content-system side only (docs/ARCHITECTURE.md): the planner consumes the
QA-passed Stage-4 Script, gates on the Stage-5 ScriptQAResult
(overall_verdict must be PASS), and joins the Stage-3 approved claim records
(the whitelist join, same as Stages 4/5). It only *visualizes* approved
content; it never introduces new factual information.

Uses local/mocked providers exclusively; no network, no APIs, no LLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from agents.qa.fact_checker.models import FactCheckResult
from agents.qa.script.models import ScriptQAResult
from agents.research.models import FactClaim, ResearchPackage
from agents.scene_planner.models import (
    PLANNING_BLOCKED_MESSAGE,
    ProductionPlan,
    ScenePlanningConfig,
)
from agents.scene_planner.planner import (
    MockScenePlanner,
    ScenePlanner,
    ScenePlanningResult,
)
from agents.scene_planner.validator import validate_production_plan
from agents.script.models import Script, ScriptError
from agents.script.stage import (
    DEFAULT_FACT_CHECK_RESULT_PATH,
    approved_claims,
)

DEFAULT_SCRIPT_PATH = Path("data") / "mock" / "script.json"
DEFAULT_SCRIPT_QA_RESULT_PATH = Path("data") / "mock" / "script_qa_result.json"


def load_script(path: str | Path = DEFAULT_SCRIPT_PATH) -> Script:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return Script.from_dict(raw)


def load_script_qa_result(path: str | Path = DEFAULT_SCRIPT_QA_RESULT_PATH) -> ScriptQAResult:
    return ScriptQAResult.from_json(Path(path).read_text(encoding="utf-8"))


def load_fact_check_result(path: str | Path = DEFAULT_FACT_CHECK_RESULT_PATH) -> FactCheckResult:
    return FactCheckResult.from_json(Path(path).read_text(encoding="utf-8"))


def claims_index(package: ResearchPackage) -> dict[str, FactClaim]:
    """Index every claim record in the package by claim id."""
    return {claim.claim_id: claim for claim in package.claims}


def run_scene_planning(
    script: Script,
    qa_result: ScriptQAResult,
    result: FactCheckResult,
    package: ResearchPackage | None = None,
    claims: Sequence[FactClaim] | None = None,
    planner: ScenePlanner | None = None,
    config: ScenePlanningConfig | None = None,
    run_id: str = "",
    created_at: str = "",
) -> ScenePlanningResult:
    """Run the scene-planning stage over a QA-passed Script + FactCheckResult.

    The Stage-5 PASS gate is enforced here, before any planning happens. The
    approved-claim join (`package` or `claims`) provides the ONLY claim records
    the planner may visualize; they are used for traceability and grounding,
    never as new factual content.
    """
    if qa_result.overall_verdict != "PASS":
        return ScenePlanningResult(
            planned=False,
            reason=PLANNING_BLOCKED_MESSAGE.format(verdict=qa_result.overall_verdict),
        )
    if package is not None:
        claim_records = approved_claims(result, package)
    else:
        claim_records = tuple(claims or ())
    if not claim_records and result.approved_claims:
        raise ScriptError(
            "no claim records provided for scene planning (approved_claims "
            "cannot be resolved against an empty package/claims sequence)"
        )

    planner = planner or MockScenePlanner(config=config)
    plan = planner.plan(
        script,
        qa_result,
        result,
        claim_records,
        config=config,
        run_id=run_id,
        created_at=created_at,
    )
    validation = validate_production_plan(plan, script, result, config=config)
    if not validation.passed:
        error_messages = "; ".join(issue.message for issue in validation.errors)
        return ScenePlanningResult(
            planned=False,
            plan=plan,
            validation=validation,
            reason=f"production plan validation failed: {error_messages}",
        )
    return ScenePlanningResult(planned=True, plan=plan, validation=validation)


def generate_mock_qa_result(
    result_path: str | Path = DEFAULT_FACT_CHECK_RESULT_PATH,
    run_id: str = "mock",
    created_at: str = "",
) -> ScriptQAResult:
    """Deterministically reproduce the committed mock ScriptQAResult fixture."""
    from agents.qa.script.stage import run_script_qa
    from agents.research.provider import MockResearchProvider
    from agents.research.stage import load_topic_selection, run_research

    selection = load_topic_selection()
    package = run_research(selection, MockResearchProvider(), run_id="mock")
    return run_script_qa(
        load_script(),
        load_fact_check_result(result_path),
        package=package,
        run_id=run_id,
        created_at=created_at,
    )


def generate_mock_production_plan(
    run_id: str = "mock",
    created_at: str = "",
) -> ProductionPlan:
    """Deterministically build the mock ProductionPlan in memory (no files)."""
    from agents.research.provider import MockResearchProvider
    from agents.research.stage import load_topic_selection, run_research

    selection = load_topic_selection()
    package = run_research(selection, MockResearchProvider(), run_id="mock")
    result = load_fact_check_result()
    qa_result = load_script_qa_result()
    plan = MockScenePlanner().plan(
        load_script(),
        qa_result,
        result,
        approved_claims(result, package),
        run_id=run_id,
        created_at=created_at,
    )
    return plan