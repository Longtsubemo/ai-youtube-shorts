"""Deterministic verification gates + config for Stage 3.

The verifier produces an *evidence analysis*; the gates turn it into a final
verdict. Every gate is a pure, deterministic rule over {claim, evidence,
assessment, source metadata}. Config keeps the policy knobs explicit and
testable - nothing is hard-coded in the matcher or the checker.

A verdict can only stay the same or get worse as it passes through the gates:
gates never *upgrade* REVIEW/REJECT to PASS. REJECT wins over REVIEW, which
wins over PASS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from agents.qa.fact_checker.evidence import meaningful_tokens
from agents.qa.fact_checker.models import Evidence
from agents.qa.fact_checker.verifier import (
    CONTRADICTORY,
    DIRECT,
    INSUFFICIENT,
    NONE,
    PARTIAL,
    ClaimAssessment,
)
from agents.research.models import FactClaim, Source

VERDICT_RANK = {"PASS": 0, "REVIEW": 1, "REJECT": 2}

_NON_DISCOVERY_TIERS = ("primary", "high_quality_secondary")


@dataclass(frozen=True)
class FactCheckConfig:
    """Deterministic policy for the Stage-3 gates.

    Defaults are fail-safe: insufficient or weak evidence can never sabotage
    a claim; the verdict is REVIEW (not silently PASS), and unsupported or
    speculative claims are REJECTed rather than let through.
    """

    corroboration_min_sources: int = 2      # independent non-discovery sources needed to PASS a critical claim
    discovery_only_verdict: str = "REVIEW"  # evidence only from discovery-tier sources
    insufficient_verdict: str = "REVIEW"    # source exists but its evidence does not settle the claim
    unsupported_verdict: str = "REJECT"     # evidence retrieved but nothing supports the claim
    stronger_than_evidence_verdict: str = "REVIEW"  # claim asserts more strongly than the evidence
    contradiction_verdict: str = "REVIEW"   # credible contradictory evidence present
    speculation_as_fact_verdict: str = "REJECT"     # speculation/opinion claimed as fact
    no_evidence_verdict: str = "REVIEW"     # no evidence retrieved at all
    min_approved_claims: int = 1            # approved claims needed for the package to proceed


@dataclass(frozen=True)
class GateVerdict:
    """Final per-claim outcome after the deterministic gates run."""

    verdict: str
    checks: tuple[str, ...]
    rationale: str
    confidence: float
    evidence_ids: tuple[str, ...]
    supporting_source_ids: tuple[str, ...]
    contradiction_source_ids: tuple[str, ...]


def _better(incoming: str, current: str) -> str:
    return current if VERDICT_RANK[incoming] <= VERDICT_RANK[current] else incoming


def _coverage_for(claim_text: str, excerpt: str) -> float:
    claim = meaningful_tokens(claim_text)
    excerpt_tokens = meaningful_tokens(excerpt)
    if not claim:
        return 0.0
    return len(claim & excerpt_tokens) / len(claim)


def _confidence(claim, assessment, verdict: str, sources_by_id) -> float:
    supporting_tiers = [
        sources_by_id[sid].credibility_tier
        for sid in assessment.supporting_source_ids
        if sid in sources_by_id
    ]
    tier_base = {"primary": 0.85, "high_quality_secondary": 0.6, "discovery": 0.3}
    confidence = max((tier_base.get(tier, 0.2) for tier in supporting_tiers), default=0.3)
    non_discovery = sum(1 for tier in supporting_tiers if tier in _NON_DISCOVERY_TIERS)
    if non_discovery >= 2:
        confidence = min(0.95, confidence + 0.05 * (non_discovery - 1))
    if verdict == "REVIEW":
        confidence = min(confidence, 0.5)
    elif verdict == "REJECT":
        confidence = min(confidence, 0.3)
    return round(confidence, 2)


def evaluate_claim(
    claim: FactClaim,
    evidence: Sequence[Evidence],
    assessment: ClaimAssessment,
    sources_by_id: Mapping[str, Source],
    config: FactCheckConfig = FactCheckConfig(),
) -> GateVerdict:
    """Run all deterministic gates and return the final claim verdict."""
    verdict = "PASS"
    gates: list[str] = []
    rationale: list[str] = []

    # Gate 1: claim references no sources -> REJECT.
    if not claim.source_ids:
        verdict = _better(verdict, config.unsupported_verdict)
        verdict = "REJECT"
        rationale.append("claim references no sources")
    else:
        # Gate 2: every referenced source must exist.
        missing = [sid for sid in claim.source_ids if sid not in sources_by_id]
        if missing:
            verdict = _better(verdict, "REJECT")
            rationale.append(f"claim references unknown source(s): {', '.join(missing)}")

    # Gate 3: no evidence retrieved at all.
    if not assessment.evidence_ids:
        verdict = _better(verdict, config.no_evidence_verdict)
        rationale.append("no evidence retrieved for the claim's source(s)")
    else:
        # Gate 4: a credible contradiction outweighs confirmations.
        if assessment.support_level == CONTRADICTORY:
            verdict = _better(verdict, config.contradiction_verdict)
            rationale.append("credible contradictory evidence present")
        # Gate 5: speculation/opinion may never silently become factual PASS.
        elif assessment.speculation_as_fact:
            verdict = _better(verdict, config.speculation_as_fact_verdict)
            rationale.append("claim is speculation/opinion, not a settled fact")
        # Gate 6: unsupported claim (evidence retrieved, none supports it).
        elif assessment.support_level == NONE:
            verdict = _better(verdict, config.unsupported_verdict)
            rationale.append("none of the retrieved evidence supports the claim wording")
        # Gate 7: insufficient evidence from existing sources.
        elif assessment.support_level == INSUFFICIENT:
            verdict = _better(verdict, config.insufficient_verdict)
            rationale.append("retrieved evidence does not settle the claim (insufficient)")
        # Gate 8: claim worded more strongly than its evidence.
        elif assessment.stronger_than_evidence:
            verdict = _better(verdict, config.stronger_than_evidence_verdict)
            rationale.append("claim is worded more strongly than the supplied evidence supports")
        # Gate 9: partial wording coverage.
        elif assessment.support_level == PARTIAL:
            verdict = _better(verdict, config.insufficient_verdict)
            rationale.append("evidence covers only part of the claim's exact wording")
        # Gate 10: discovery-tier evidence cannot establish a claim on its own.
        elif assessment.discovery_only:
            verdict = _better(verdict, config.discovery_only_verdict)
            rationale.append("evidence comes only from discovery-tier source(s)")

    # Critical corroboration requirement (independent non-discovery sources).
    if claim.critical and assessment.support_level in (DIRECT, PARTIAL):
        non_discovery_support = [
            sid
            for sid in assessment.supporting_source_ids
            if sources_by_id.get(sid) and sources_by_id[sid].credibility_tier in _NON_DISCOVERY_TIERS
        ]
        if len(non_discovery_support) < config.corroboration_min_sources:
            verdict = _better(verdict, "REVIEW")
            rationale.append(
                f"critical claim corroborated by only {len(non_discovery_support)} "
                f"independent non-discovery source(s); required "
                f"{config.corroboration_min_sources}"
            )

    # Deterministic rationale: report exactly what evidence was used and why.
    evidence_desc = ", ".join(assessment.evidence_ids) if assessment.evidence_ids else "none"
    support_desc = ", ".join(assessment.supporting_source_ids) if assessment.supporting_source_ids else "none"
    contradiction_desc = (
        ", ".join(assessment.contradiction_source_ids) if assessment.contradiction_source_ids else "none"
    )
    rationale.insert(
        0,
        f"evidence examined: {evidence_desc}; supporting sources: {support_desc}; "
        f"contradicting sources: {contradiction_desc}",
    )

    confidence = _confidence(claim, assessment, verdict, sources_by_id)
    gates = list(assessment.checks)
    return GateVerdict(
        verdict=verdict,
        checks=tuple(gates),
        rationale=" ".join(rationale),
        confidence=confidence,
        evidence_ids=tuple(assessment.evidence_ids),
        supporting_source_ids=tuple(assessment.supporting_source_ids),
        contradiction_source_ids=tuple(assessment.contradiction_source_ids),
    )


# --------------------------------------------------------------------------- #
# Package-level aggregation
# --------------------------------------------------------------------------- #


def aggregate_package(
    claim_results: Sequence,
    config: FactCheckConfig = FactCheckConfig(),
) -> tuple[dict, list[str], list[str], list[str], list[str]]:
    """Aggregate per-claim verdicts into the package verdict + claim lists.

    Returns (summary, review_items, rejected_claims, approved_claims,
    critical_claim_failures) - the four lists Stage 4 and the audit chain rely
    on. `approved_claims` is exactly what Stage 4 may consume.
    """
    reviewed = [
        v.claim_id for v in claim_results if v.verdict == "REVIEW"
    ]
    rejected = [
        v.claim_id for v in claim_results if v.verdict == "REJECT"
    ]
    approved = [
        v.claim_id for v in claim_results if v.verdict == "PASS"
    ]
    critical_failures = [
        v.claim_id for v in claim_results if v.critical and v.verdict != "PASS"
    ]

    if any(v.critical and v.verdict == "REJECT" for v in claim_results):
        package_verdict = "REJECT"
        reason = "at least one critical claim was rejected"
    elif any(v.critical and v.verdict == "REVIEW" for v in claim_results):
        package_verdict = "REVIEW"
        reason = "at least one critical claim is under review and cannot be safely approved"
    elif not claim_results:
        package_verdict = "REJECT"
        reason = "no claims to fact check"
    elif len(approved) < config.min_approved_claims:
        package_verdict = "REVIEW"
        reason = f"only {len(approved)} approved claim(s); need {config.min_approved_claims}"
    else:
        package_verdict = "PASS"
        reason = (
            f"{len(approved)} approved claim(s); rejected and REVIEW claims excluded "
            f"from script consumption"
        )

    summary = {
        "package_verdict": package_verdict,
        "reason": reason,
        "approved_count": len(approved),
        "review_count": len(reviewed),
        "rejected_count": len(rejected),
        "critical_claim_failures": list(critical_failures),
    }
    return summary, reviewed, rejected, approved, critical_failures