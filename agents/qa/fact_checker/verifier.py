"""Verifier abstraction and the deterministic MockVerifier.

The verifier answers one question and one question only:

    "Does the supplied evidence support the exact wording of this claim?"

It never answers "do I personally know whether this claim is true?". The
MockVerifier is a hash-based, network-free text matcher: it compares the
claim's own wording against the excerpts actually retrieved from the claim's
cited sources. No model, no world knowledge, no invented facts.

The `Verifier` protocol is the future LLM boundary (docs/FACT_VERIFICATION.md):
an LLM judge can be plugged in later, but its prompt must be constrained to
{claim, supplied evidence, source metadata} and must never fall back on
outside knowledge as evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

from agents.qa.fact_checker.evidence import meaningful_tokens
from agents.qa.fact_checker.models import EVIDENCE_TYPES, Evidence
from agents.research.models import FactClaim, Source

# Words that assert a factual discovery/causal claim with certainty. If the
# claim uses these but the supplied evidence does not, the claim is being
# worded more strongly than the evidence (e.g. claim "proved X causes Y" vs
# evidence "observed a correlation between X and Y").
_STRONG_CLAIM_WORDS = frozenset(
    {
        "proved", "proven", "proves", "proving", "proof", "confirmed",
        "confirm", "demonstrated", "demonstrate", "demonstrates",
        "established", "establishes", "discovered", "discovery",
        "definitively", "conclusively", "guarantees", "caused", "causes",
        "causing", "exactly", "definitely", "certainly", "fact",
    }
)

# Words that make a statement an observation/hedge rather than a proof. An
# excerpt containing these is weaker than a claim that flatly asserts a cause.
_HEDGE_WORDS = frozenset(
    {
        "suggest", "suggests", "suggested", "indicating", "indicates",
        "indicated", "may", "might", "could", "possibly", "probably",
        "likely", "correlat", "associat", "appears", "seems", "reportedly",
        "believed", "according", "observed", "observes", "observations",
        "tentatively", "debated", "unclear", "possi", "theorizes",
    }
)

# Phrases that negate the claim directly ("no evidence that X", "X was not
# observed"...). Their presence in an excerpt signals a contradiction even
# when the evidence_type metadata missed it.
_NEGATION_PHRASES = frozenset(
    {
        "no evidence",
        "do not",
        "does not",
        "did not",
        "not true",
        "not proven",
        "no proof",
        "disprove",
        "disproved",
        "contradicts",
        "untrue",
        "debunked",
        "never",
        "fails to",
        "found no",
        "not confirmed",
    }
)

# Coverage thresholds, as fractions of the claim's meaningful tokens found
# verbatim inside an excerpt. Deterministic, no vectors/models.
DIRECT_COVERAGE_THRESHOLD = 0.8
PARTIAL_COVERAGE_THRESHOLD = 0.3
CONTRADICTION_COVERAGE_THRESHOLD = 0.2

# Support levels the matcher can produce for a claim.
DIRECT = "direct"          # excerpt covers (almost) the full claim wording
PARTIAL = "partial"        # excerpt covers part of the claim wording only
CONTRADICTORY = "contradictory"
INSUFFICIENT = "insufficient"
NONE = "none"


class Verifier(Protocol):
    """Constrained evaluator: does the supplied evidence back the claim?"""

    def verify(
        self,
        claim: FactClaim,
        evidence: Sequence[Evidence],
        sources_by_id: Mapping[str, Source],
    ) -> "ClaimAssessment": ...


@dataclass(frozen=True)
class ClaimAssessment:
    """Structured evidence-analysis produced by a Verifier, before gating.

    All fields derive only from the supplied claim, evidence, and source
    metadata. `support_level` is the matcher's opinion; the deterministic
    gates (gates.py) turn it into a final PASS/REVIEW/REJECT verdict.
    """

    claim_id: str
    support_level: str
    stronger_than_evidence: bool = False
    speculation_as_fact: bool = False
    discovery_only: bool = False
    contradiction_present: bool = False
    evidence_ids: tuple[str, ...] = ()
    supporting_evidence_ids: tuple[str, ...] = ()
    supporting_source_ids: tuple[str, ...] = ()
    contradiction_source_ids: tuple[str, ...] = ()
    insufficient_source_ids: tuple[str, ...] = ()
    checks: tuple[str, ...] = ()


def _contains_any(text: str, markers: frozenset[str]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def _coverage(claim_meaningful: set[str], excerpt_meaningful: set[str]) -> float:
    if not claim_meaningful:
        return 0.0
    overlap = claim_meaningful & excerpt_meaningful
    return len(overlap) / len(claim_meaningful)


def _stance(evidence: Evidence, coverage: float, negates: bool) -> str:
    if evidence.evidence_type == "contradictory_statement":
        return CONTRADICTORY if coverage >= CONTRADICTION_COVERAGE_THRESHOLD else NONE
    if evidence.evidence_type == "insufficient":
        return INSUFFICIENT
    if coverage >= DIRECT_COVERAGE_THRESHOLD:
        return DIRECT
    if coverage >= PARTIAL_COVERAGE_THRESHOLD:
        return PARTIAL
    if negates and coverage >= CONTRADICTION_COVERAGE_THRESHOLD:
        return CONTRADICTORY
    return NONE


class MockVerifier:
    """Deterministic, offline, hash-free (regex-free) text matcher.

    Network-free by construction: it only string-matches. Deterministic: the
    same (claim, evidence, sources) always produce the same assessment.
    """

    def verify(
        self,
        claim: FactClaim,
        evidence: Sequence[Evidence],
        sources_by_id: Mapping[str, Source],
    ) -> ClaimAssessment:
        claim_meaningful = meaningful_tokens(claim.claim_text)
        claim_strong_words = [w for w in _STRONG_CLAIM_WORDS if w in claim.claim_text.lower()]

        stances: list[str] = []
        supporting_ids: list[str] = []
        supporting_sources: list[str] = []
        contradiction_sources: list[str] = []
        insufficient_sources: list[str] = []
        evidence_ids: list[str] = []
        max_coverage = 0.0

        exhibit_strong_words: list[str] = []
        exhibit_hedged: list[str] = []

        for item in evidence:
            evidence_ids.append(item.evidence_id)
            excerpt_meaningful = meaningful_tokens(item.excerpt)
            overlap = _coverage(claim_meaningful, excerpt_meaningful)
            max_coverage = max(max_coverage, overlap)
            negates = _contains_any(item.excerpt, _NEGATION_PHRASES)
            stance = _stance(item, overlap, negates)
            stances.append(stance)
            source = sources_by_id.get(item.source_id)
            tier = source.credibility_tier if source else "discovery"

            if stance in (DIRECT, PARTIAL):
                supporting_ids.append(item.evidence_id)
                if source is not None and source.source_id not in supporting_sources:
                    supporting_sources.append(source.source_id)
                if _contains_any(item.excerpt, _STRONG_CLAIM_WORDS):
                    exhibit_strong_words.append(item.evidence_id)
                else:
                    exhibit_hedged.append(item.evidence_id)
            elif (
                stance == CONTRADICTORY
                and source is not None
                and source.credibility_tier != "discovery"
                and source.source_id not in contradiction_sources
            ):
                contradiction_sources.append(source.source_id)
            elif stance == INSUFFICIENT and source is not None and source.source_id not in insufficient_sources:
                insufficient_sources.append(source.source_id)

        # A credible (non-discovery-tier) contradiction outweighs any support:
        # an unresolved dispute can never be a silent PASS.
        contradiction_present = bool(contradiction_sources)

        if contradiction_present:
            support_level = CONTRADICTORY
        elif any(stance == DIRECT for stance in stances):
            support_level = DIRECT
        elif any(stance == PARTIAL for stance in stances):
            support_level = PARTIAL
        elif any(stance == INSUFFICIENT for stance in stances):
            support_level = INSUFFICIENT
        else:
            support_level = NONE

        supporting_sources_tiers = [
            sources_by_id[sid].credibility_tier for sid in supporting_sources if sid in sources_by_id
        ]
        discovery_only = bool(
            supporting_sources and supporting_sources_tiers
            and all(tier == "discovery" for tier in supporting_sources_tiers)
        )

        claim_asserts_strong = bool(claim_strong_words)
        exhibit_has_strong = bool(exhibit_strong_words)
        evidence_is_hedged = bool(exhibit_hedged)
        stronger_than_evidence = (
            claim_asserts_strong
            and not exhibit_has_strong
            and bool(supporting_ids)
        )

        speculation_as_fact = claim.claim_type in ("speculation", "opinion")

        checks = (
            f"coverage_max={max_coverage:.2f}",
            f"stance_overall={support_level}",
            f"strong_in_claim={','.join(sorted(set(claim_strong_words))) or 'none'}",
            f"strong_in_evidence={'yes' if exhibit_has_strong else 'no'}",
            f"hedged_in_evidence={'yes' if evidence_is_hedged else 'no'}",
            f"contradiction_present={'yes' if contradiction_present else 'no'}",
            f"discovery_only={'yes' if discovery_only else 'no'}",
            f"speculation_as_fact={'yes' if speculation_as_fact else 'no'}",
        )

        return ClaimAssessment(
            claim_id=claim.claim_id,
            support_level=support_level,
            stronger_than_evidence=stronger_than_evidence,
            speculation_as_fact=speculation_as_fact,
            discovery_only=discovery_only,
            contradiction_present=contradiction_present,
            supporting_evidence_ids=tuple(sorted(set(supporting_ids))),
            supporting_source_ids=tuple(sorted(set(supporting_sources))),
            contradiction_source_ids=tuple(sorted(set(contradiction_sources))),
            insufficient_source_ids=tuple(sorted(set(insufficient_sources))),
            evidence_ids=tuple(sorted(set(evidence_ids))),
            checks=checks,
        )