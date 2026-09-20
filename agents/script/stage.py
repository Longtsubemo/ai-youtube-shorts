"""Stage-4 entrypoint: FactCheckResult -> Script.

Content-system side only (docs/ARCHITECTURE.md): this stage consumes the
Stage-3 FactCheckResult and joins its `approved_claims` against the Stage-2
ResearchPackage records, then hands the generator ONLY the approved claim
texts. The join is a whitelist: a claim's text is materialized exclusively
because - and only if - its id appears in `approved_claims`, so rejected,
REVIEW, and unlisted claims can never leak into a script (docs/FACT_VERIFICATION.md,
"consumption boundary").

Uses local/mocked providers exclusively; no network, no APIs, no LLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from agents.qa.fact_checker.models import FactCheckError, FactCheckResult
from agents.research.models import FactClaim, ResearchPackage
from agents.script.models import ScriptError
from agents.script.script_generator import (
    MockScriptGenerator,
    ScriptConfig,
    ScriptGenerationResult,
    ScriptGenerator,
)

DEFAULT_FACT_CHECK_RESULT_PATH = Path("data") / "mock" / "fact_check_result.json"


def load_fact_check_result(path: str | Path = DEFAULT_FACT_CHECK_RESULT_PATH) -> FactCheckResult:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return FactCheckResult.from_dict(raw)


def approved_claims(
    result: FactCheckResult, package: ResearchPackage
) -> tuple[FactClaim, ...]:
    """Resolve `result.approved_claims` to claim records (whitelist join).

    Order follows `result.approved_claims` so the script's line order is
    deterministic. Raises ScriptError if any approved id has no claim record -
    the join must never silently drop approved content.
    """
    claims_by_id = {claim.claim_id: claim for claim in package.claims}
    missing = [claim_id for claim_id in result.approved_claims if claim_id not in claims_by_id]
    if missing:
        raise ScriptError(
            f"approved claim(s) missing from ResearchPackage: {', '.join(missing)}"
        )
    return tuple(claims_by_id[claim_id] for claim_id in result.approved_claims)


def run_script_generation(
    result: FactCheckResult,
    package: ResearchPackage,
    generator: ScriptGenerator | None = None,
    config: ScriptConfig | None = None,
    run_id: str = "",
    created_at: str = "",
) -> ScriptGenerationResult:
    """Run the script-generation stage over a FactCheckResult + ResearchPackage."""
    generator = generator or MockScriptGenerator(config=config)
    claims = approved_claims(result, package)
    return generator.generate(
        result,
        claims,
        config=config,
        run_id=run_id,
        created_at=created_at,
    )