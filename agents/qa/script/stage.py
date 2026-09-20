"""Stage-5 entrypoint: Script + FactCheckResult -> ScriptQAResult.

Content-system side only (docs/ARCHITECTURE.md): the QA stage consumes the
Stage-4 Script artifact plus the Stage-3 FactCheckResult and the approved
claim records (whitelist join, same as Stage 4). It never re-runs research and
never consults unapproved claims as fact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from agents.qa.fact_checker.models import FactCheckResult
from agents.qa.script.models import ScriptQAConfig, ScriptQAResult
from agents.qa.script.validator import evaluate_script
from agents.research.models import FactClaim, ResearchPackage
from agents.script.models import Script, ScriptError
from agents.script.stage import (
    DEFAULT_FACT_CHECK_RESULT_PATH,
    approved_claims,
)
from agents.script.script_generator import (
    ScriptConfig,
    MockScriptGenerator,
)

DEFAULT_SCRIPT_PATH = Path("data") / "mock" / "script.json"


def load_script(path: str | Path = DEFAULT_SCRIPT_PATH) -> Script:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return Script.from_dict(raw)


def load_fact_check_result(path: str | Path = DEFAULT_FACT_CHECK_RESULT_PATH) -> FactCheckResult:
    return FactCheckResult.from_json(Path(path).read_text(encoding="utf-8"))


def claims_index(package: ResearchPackage) -> dict[str, FactClaim]:
    """Index every claim record in the package by claim id."""
    return {claim.claim_id: claim for claim in package.claims}


def run_script_qa(
    script: Script,
    result: FactCheckResult,
    package: ResearchPackage | None = None,
    claims: Sequence[FactClaim] | None = None,
    config: ScriptQAConfig | None = None,
    run_id: str = "",
    created_at: str = "",
) -> ScriptQAResult:
    """Run the QA stage over a Script artifact + FactCheckResult.

    `claims` (or the resolve-able `package`) provides the claim records that
    pass the approved-claim join; they are used ONLY for validation and
    traceability and never introduce new factual content.
    """
    if package is not None:
        claim_records = approved_claims(result, package)
    else:
        claim_records = tuple(claims or ())
    if not claim_records and result.approved_claims:
        raise ScriptError(
            "no claim records provided for the QA stage (approved_claims "
            "cannot be resolved against an empty package/claims sequence)"
        )
    claims_by_id = {claim.claim_id: claim for claim in claim_records}
    return evaluate_script(
        script,
        result,
        claims_by_id,
        config=config,
        run_id=run_id,
        created_at=created_at,
    )


def generate_mock_script(
    result_path: str | Path = DEFAULT_FACT_CHECK_RESULT_PATH,
    config: ScriptConfig | None = None,
    run_id: str = "mock",
    created_at: str = "",
) -> Script:
    """Deterministically reproduce the committed mock Script artifact."""
    from agents.research.provider import MockResearchProvider
    from agents.research.stage import load_topic_selection, run_research

    selection = load_topic_selection()
    package = run_research(selection, MockResearchProvider(), run_id="mock")
    result = load_fact_check_result(result_path)
    generation = MockScriptGenerator(config=config).generate(
        result,
        package.claims,
        config=config,
        run_id=run_id,
        created_at=created_at,
    )
    if not generation.generated or generation.script is None:
        raise ScriptError(f"mock script generation failed: {generation.message}")
    return generation.script