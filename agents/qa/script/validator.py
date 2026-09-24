"""Stage-5 QA orchestration: checks -> verdict -> artifact.

This module owns the overall verdict (PASS / REVISE / REJECT), the issue id
assignment, and the assembly of the ScriptQAResult artifact. It is fully
deterministic and network-free.

Verdict rules (docs/DATA_CONTRACTS.md §8.1 / PROJECT_PROGRESS Stage 5):

    REJECT  any severity-ERROR issue that is a HARD gate
    REVISE  any severity-ERROR issue that is not a hard gate
    PASS    otherwise (warnings/info only, or clean)

Hard gates (per-issue `gate=hard`) cover the fail-closed conditions: no
factual traceability; rejected/review/unknown claim references; material
contradiction; material certainty strengthening; invented factual detail;
empty/unusable or malformed script (duplicate line ids, unsupported purpose);
package not approved for scripting; missing script provenance; script package
binding not matching the fact-check result; and inconsistent claim accounting
(used | unused must exactly equal the approved claim set).
"""

from __future__ import annotations

import dataclasses

from agents.qa.script.checks import GATE_HARD, run_checks
from agents.qa.script.models import (
    PRODUCED_BY,
    ClaimCoverage,
    QAIssue,
    ScriptQAConfig,
    ScriptQAResult,
    qa_result_id_for,
)
from agents.qa.script.scorer import compute_score
from agents.research.models import FactClaim
from agents.seo.models import Provenance, utc_now

VERDICT_REJECT = "REJECT"
VERDICT_REVISE = "REVISE"
VERDICT_PASS = "PASS"


def aggregate_verdict(issues) -> str:
    """Overall verdict from a severity-ERROR stream (hard gate wins)."""
    errors = [issue for issue in issues if issue.severity == "ERROR"]
    if any(issue.gate == GATE_HARD for issue in errors):
        return VERDICT_REJECT
    if errors:
        return VERDICT_REVISE
    return VERDICT_PASS


def evaluate_script(
    script,
    fact_check_result,
    claims_by_id: dict[str, FactClaim],
    config: ScriptQAConfig | None = None,
    run_id: str = "",
    created_at: str = "",
) -> ScriptQAResult:
    """Run every check against a Script artifact and produce a ScriptQAResult.

    Deliberately does NOT call `script.validate()`: Stage 5 is a gate, not a
    re-validation of Stage 4. Structurally broken artifacts (empty lines,
    missing provenance, foreign claim ids) must produce QA issues rather than
    raise, so checks read attributes defensively.
    """
    config = (config or ScriptQAConfig()).validate()
    run_id = run_id or script.run_id or fact_check_result.run_id
    created_at = created_at or script.created_at or utc_now()

    issues, check_records = run_checks(script, fact_check_result, claims_by_id, config)

    # Stable, content-independent issue ids (deterministic ordering).
    issues = [
        dataclasses.replace(issue, issue_id=f"qa_{number:02d}")
        for number, issue in enumerate(issues)
    ]

    verdict = aggregate_verdict(issues)
    coverage = ClaimCoverage.compute(
        approved_claim_ids=tuple(fact_check_result.approved_claims),
        used_claim_ids=tuple(sorted(script.all_claim_refs())),
        critical_ids=tuple(_critical_ids(fact_check_result, claims_by_id)),
    )
    score = compute_score(issues, config.weight_map())

    result = ScriptQAResult(
        qa_result_id=qa_result_id_for(run_id, script.script_id),
        script_id=script.script_id,
        topic_id=script.topic_id,
        result_id=fact_check_result.result_id,
        package_id=fact_check_result.package_id,
        overall_verdict=verdict,
        score=score.score,
        score_detail=score.to_dict(),
        checks=tuple(check_records),
        issues=tuple(issue for issue in issues if issue.severity == "ERROR"),
        warnings=tuple(issue for issue in issues if issue.severity == "WARNING"),
        infos=tuple(issue for issue in issues if issue.severity == "INFO"),
        approved_claim_coverage=coverage,
        provenance=Provenance(
            generated_by="script_qa_agent_v1",
            source="DeterministicScriptQA",
            created_at=created_at,
            provider=None,
            model="ScriptQAValidator",
            prompt_version=config.schema_version,
        ),
        created_at=created_at,
        run_id=run_id,
        produced_by=dict(PRODUCED_BY),
    )
    return result.validate()


def _critical_ids(fact_check_result, claims_by_id: dict[str, FactClaim]) -> list[str]:
    """Approved claim ids that are critical (research record or verdict)."""
    verdict_by_id = {v.claim_id: v for v in fact_check_result.claim_results}
    critical = []
    for claim_id in fact_check_result.approved_claims:
        claim = claims_by_id.get(claim_id)
        if claim is not None and claim.critical:
            critical.append(claim_id)
        else:
            verdict = verdict_by_id.get(claim_id)
            if verdict is not None and verdict.critical:
                critical.append(claim_id)
    return critical