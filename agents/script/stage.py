"""Stage-4 entrypoint: FactCheckResult -> Script.

Content-system side only (docs/ARCHITECTURE.md): this stage consumes the
Stage-3 FactCheckResult and joins its `approved_claims` against the Stage-2
ResearchPackage records, then hands the generator ONLY the approved claim
texts. The join is a whitelist: a claim's text is materialized exclusively
because - and only if - its id appears in `approved_claims`, so rejected,
REVIEW, and unlisted claims can never leak into a script (docs/FACT_VERIFICATION.md,
"consumption boundary").

`bind_fact_check_result` and `fact_check_gate` are provider-neutral pieces of
that same boundary, shared by every `ScriptProvider` (mock and Gemini) so the
real path cannot drift from the mock one: a result that does not belong to the
supplied package is rejected, and a package verdict that did not survive fact
checking blocks generation.

Uses local/mocked providers exclusively; no network, no APIs, no LLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from agents.qa.fact_checker.models import FactCheckError, FactCheckResult
from agents.research.models import FactClaim, ResearchPackage
from agents.script.models import ScriptError
from agents.script.script_generator import (
    STATUS_BLOCKED,
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


def bind_fact_check_result(
    package: ResearchPackage, result: FactCheckResult
) -> FactCheckResult:
    """Check that `result` was produced for exactly `package`, and return it.

    The Stage-3 result is the binding between research and script: it carries
    `result_id` (which the Script must inherit) and `package_id`/`topic_id`
    (which prove it judged *these* claims). A mismatch means the caller paired
    a result with the wrong research package, which would silently validate the
    script against unrelated verdicts - so it fails loudly instead.
    """
    try:
        result.validate()
    except FactCheckError as exc:
        raise ScriptError(f"invalid FactCheckResult: {exc}") from exc
    if result.package_id != package.package_id:
        raise ScriptError(
            f"fact-check result {result.result_id!r} is bound to package "
            f"{result.package_id!r}, not {package.package_id!r}"
        )
    if result.topic_id != package.topic_id:
        raise ScriptError(
            f"fact-check result {result.result_id!r} is bound to topic "
            f"{result.topic_id!r}, not {package.topic_id!r}"
        )
    return result


def fact_check_gate(
    result: FactCheckResult, config: Any
) -> ScriptGenerationResult | None:
    """Package-verdict gate shared by every provider; None means "may proceed".

    Mirrors the mock generator's first gate so the real Gemini path enforces
    exactly what the deterministic path does: a REJECTed package has no
    validated content to script, and a REVIEWed package is blocked unless the
    caller explicitly opted in via `config.allow_review_content`.
    """
    if result.package_verdict == "REJECT":
        return ScriptGenerationResult(
            status=STATUS_BLOCKED,
            message="package verdict REJECT: no validated content to script",
            blocked_reasons=("package_verdict=REJECT",),
        )
    if result.package_verdict == "REVIEW" and not config.allow_review_content:
        return ScriptGenerationResult(
            status=STATUS_BLOCKED,
            message="package verdict REVIEW: review claims blocked by default",
            blocked_reasons=("package_verdict=REVIEW",),
        )
    return None


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