"""Deterministic claim classification and provisional confidence.

The research agent has no LLM (and must not need one). Instead, claim
classification and provisional confidence are deterministic, rule-based
heuristics so the whole research stage is testable offline and swappable
for a provider-backed one later without changing the agent's contract.
"""

from __future__ import annotations

from agents.research.models import CLAIM_TYPES, Source

# Cue word sets for heuristic classification. Order matters: opinion is
# checked first (valuation), then speculation (unsupported/forward-looking),
# then inference (hedged reasoning drawn from facts), then explanation
# (mechanism/how-why), and anything else is a plain fact.
_OPINION_CUES = (
    "i believe",
    "in my view",
    "most fascinating",
    "most interesting",
    "most important",
    "best ",
    "fascinating",
    "amazing",
    "beautiful",
    "unfortunately",
)
_SPECULATION_CUES = (
    "may someday",
    "might someday",
    "could someday",
    "could one day",
    "may one day",
    "might one day",
    "in the future",
    "could eventually",
    "imagine",
    "maybe",
    "possibly",
)
_INFERENCE_CUES = (
    "probably",
    "likely",
    "suggests",
    "suggested",
    "implies",
    "therefore",
    "given that",
    "indicates",
    "seems",
    "tends to",
)
_EXPLANATION_CUES = (
    "because",
    "which means",
    "that means",
    "as a result",
    "works by",
    "causes",
    "caused by",
    "this is why",
    "step by step",
    "by capturing",
)


def classify_claim(claim_text: str) -> str:
    """Return one of CLAIM_TYPES for a claim based on cue words.

    Deterministic and rule-based (no model, no network). Unmatched text
    defaults to `fact`.
    """
    lowered = claim_text.lower()
    if any(cue in lowered for cue in _OPINION_CUES):
        return "opinion"
    if any(cue in lowered for cue in _SPECULATION_CUES):
        return "speculation"
    if any(cue in lowered for cue in _INFERENCE_CUES):
        return "inference"
    if any(cue in lowered for cue in _EXPLANATION_CUES):
        return "explanation"
    return "fact"


def compute_confidence(claim_type: str, sources: tuple[Source, ...]) -> float:
    """Provisional confidence from backing-source tiers (0..1).

    Strongest source dominates. A corroboration boost applies only when at
    least two *non-discovery* sources back the claim - discovery leads add
    coverage, not trust. Speculation and opinion are hard-capped low because
    they are inherently not evidence-backed.
    """
    if not sources:
        return 0.0
    tier_base = {"primary": 0.85, "high_quality_secondary": 0.6, "discovery": 0.3}
    bases = [tier_base.get(source.credibility_tier, 0.2) for source in sources]
    non_discovery = [base for source, base in zip(sources, bases) if source.is_non_discovery]
    confidence = max(bases)
    if len(non_discovery) >= 2:
        confidence = min(0.95, confidence + 0.05 * (len(non_discovery) - 1))
    if claim_type in ("speculation", "opinion"):
        confidence = min(confidence, 0.4)
    return round(confidence, 2)


def validate_claim_type(claim_type: str) -> str:
    """Raise a clean error when caller supplies an unknown claim type."""
    if claim_type not in CLAIM_TYPES:
        allowed = ", ".join(CLAIM_TYPES)
        raise ValueError(f"unknown claim_type '{claim_type}' (allowed: {allowed})")
    return claim_type