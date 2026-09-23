"""Deterministic Stage-4 validation: scripts may not invent, strengthen, or dodge.

The validator is a set of pure, deterministic checks run after (and within)
generation, so a generated script either satisfies every rule or fails loudly.
It never consults the network or the generator's world knowledge - only the
script text, the fact-check result, and the approved claim texts it was given.

Checks implemented (docs/DATA_CONTRACTS.md §8 / PROJECT_PROGRESS.md Stage 4):

  structure   non-empty lines; title/hook present; hook + fact/explanation lines exist
  refs        factual lines carry claim_refs; every ref exists in the result;
              every ref is an APPROVED claim (never REVIEW/REJECT/unlisted)
  tracking    used == union of refs; used+wasted == approved; disjoint
  duration    within [min,max]; derived sum matches declared total
  provenance  provenance present; result_id matches the source fact-check result
  semantics   factual lines stay inside their claims' token coverage and keep
              hedges; no new certainty words that the claims do not contain

The token-coverage heuristic has one job: catch lines that smuggle in wording
(or entire facts) that the approved claims never said. Hedged claims must keep
a hedge so "probably P" cannot silently become "P".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from agents.qa.fact_checker.evidence import meaningful_tokens
from agents.qa.fact_checker.models import FactCheckResult
from agents.research.models import FactClaim, ResearchPackage
from agents.script.models import (
    FACTUAL_PURPOSES,
    DURATION_TOLERANCE_SECONDS,
    PURPOSE_VALUES,
    Script,
)

# Words that overstate the evidence if the referenced claims never say them.
_CERTAINTY_WORDS = frozenset(
    {
        "proved", "proven", "proves", "confirmed", "conclusively", "definitely",
        "certainly", "guaranteed", "established", "discovered", "demonstrates",
        "demonstrated", "undeniably", "irrefutable", "always", "never", "proved",
    }
)

# Words that signal a claim is hedged. If a referenced claim is hedged and the
# line drops every hedge, the line overstates its source.
_HEDGE_WORDS = frozenset(
    {
        "probably", "likely", "may", "might", "possibly", "could", "maybe",
        "suggests", "suggested", "reportedly", "estimated", "approximately",
        "one", "some", "according", "appears", "seems", "tends",
    }
)

_PUNCTUATION = re.compile(r"[^a-z0-9]+")
_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return meaningful_tokens(_PUNCTUATION.sub(" ", text.lower()))


@dataclass(frozen=True)
class ScriptValidation:
    """Result of validating a script; `valid` is False whenever errors exist."""

    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def _covered(ref_claims: Sequence[FactClaim], line_text: str, threshold: float) -> float | None:
    """Fraction of the line's meaningful tokens that appear in its claims.

    Returns None when there are no claim words to compare against (nothing to
    measure); otherwise a ratio in [0, 1].
    """
    claim_tokens = set()
    for claim in ref_claims:
        claim_tokens |= _tokens(claim.claim_text)
    line_tokens = _tokens(line_text)
    if not line_tokens or not claim_tokens:
        return None
    return len(line_tokens & claim_tokens) / len(line_tokens)


def _strengthened(ref_claims: Sequence[FactClaim], line_text: str) -> str | None:
    """Return an overstating word if the line asserts certainty beyond its claims."""
    claim_text = " ".join(claim.claim_text.lower() for claim in ref_claims)
    line_words = _WORD.findall(line_text.lower())
    claim_words = _WORD.findall(claim_text)
    for word in _CERTAINTY_WORDS:
        if word in line_words and word not in claim_words:
            return word
    return None


def _hedge_dropped(ref_claims: Sequence[FactClaim], line_text: str) -> bool:
    """True if a hedged claim had its hedge(s) stripped by this line."""
    hedged = [claim for claim in ref_claims if _HEDGE_WORDS & _tokens(claim.claim_text)]
    if not hedged:
        return False
    return not (_HEDGE_WORDS & _tokens(line_text))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def validate_script(
    script: Script,
    fact_check_result: FactCheckResult,
    approved_claims: Sequence[FactClaim],
    config: Any,
) -> ScriptValidation:
    """Validate a generated script against its fact-check result and claims.

    `approved_claims` must be the exact claim records the generator consumed
    (only approved claims - see stage.py). `config` provides the duration
    bounds and semantic thresholds (ScriptConfig from script_generator).
    """
    errors: list[str] = []
    warnings: list[str] = []

    claims_by_id = {claim.claim_id: claim for claim in approved_claims}
    verdict_by_id = {
        result.claim_id: result for result in fact_check_result.claim_results
    }
    approved = set(fact_check_result.approved_claims)

    # Provenance / binding to the exact fact-check result.
    if script.provenance is None:
        errors.append("script has no provenance")
    if script.result_id != fact_check_result.result_id:
        errors.append(
            f"script.result_id {script.result_id!r} does not match "
            f"fact-check result {fact_check_result.result_id!r}"
        )

    # Structural requirements.
    if not script.lines:
        errors.append("script has no lines")
    if not script.title.strip():
        errors.append("script title is empty")
    if not script.hook.strip():
        errors.append("script hook is empty")
    if not any(line.purpose == "hook" for line in script.lines):
        errors.append("script has no hook line")
    if not any(line.purpose in ("fact", "explanation") for line in script.lines):
        errors.append("script has no fact/explanation line")

    # Claim references: factual lines must be grounded in approved claims.
    # Framing lines (hook/context/transition/cta) are boilerplate and carry
    # no refs by design - they must simply not introduce facts.
    for line in script.lines:
        if line.purpose not in PURPOSE_VALUES:
            errors.append(f"line {line.line_id} has unknown purpose {line.purpose!r}")
            continue
        if not line.claim_refs:
            if line.is_factual():
                errors.append(
                    f"factual line {line.line_id} has no claim_refs"
                )
            continue
        for claim_id in line.claim_refs:
            if claim_id not in verdict_by_id:
                errors.append(
                    f"line {line.line_id} references unknown claim {claim_id}"
                )
            elif claim_id in fact_check_result.rejected_claims:
                errors.append(
                    f"line {line.line_id} references REJECTED claim {claim_id}"
                )
            elif claim_id in fact_check_result.review_items:
                errors.append(
                    f"line {line.line_id} references REVIEW claim {claim_id}"
                )
            elif claim_id not in approved:
                errors.append(
                    f"line {line.line_id} references unapproved claim {claim_id}"
                )

        # Semantic checks only apply to factual lines (framing lines are
        # boilerplate and legitimately add non-claim words).
        if not line.is_factual():
            continue
        ref_claims = [claims_by_id[cid] for cid in line.claim_refs if cid in claims_by_id]
        if len(ref_claims) != len(line.claim_refs):
            continue  # missing claim already reported above
        coverage = _covered(ref_claims, line.text, config.semantic_coverage_threshold)
        if coverage is not None and coverage < config.semantic_coverage_threshold:
            errors.append(
                f"line {line.line_id} uses wording not in its approved claims "
                f"(token coverage {coverage:.2f})"
            )
        word = _strengthened(ref_claims, line.text)
        if word:
            errors.append(
                f"line {line.line_id} strengthens its claims with unsupported "
                f"certainty word {word!r}"
            )
        if _hedge_dropped(ref_claims, line.text):
            errors.append(
                f"line {line.line_id} drops the hedge(s) of its approved claims"
            )

    # Claim usage tracking must be exact and complete.
    used = script.all_claim_refs()
    if set(script.used_claim_ids) != used:
        errors.append("used_claim_ids is not the exact set of claim_refs")
    if set(script.unused_approved_claim_ids) & used:
        errors.append("unused_approved_claim_ids overlaps used_claim_ids")
    if (set(script.used_claim_ids) | set(script.unused_approved_claim_ids)) != approved:
        errors.append(
            "used + unused approved claim ids do not cover the approved_claims set"
        )

    # Duration: declared total must match the derived sum; bounds are warnings
    # (a reviewer can still decide an out-of-range script is acceptable).
    derived = round(sum(line.estimated_duration_seconds for line in script.lines), 2)
    if abs(derived - script.estimated_duration_seconds) > DURATION_TOLERANCE_SECONDS:
        errors.append(
            "estimated_duration_seconds does not match the derived sum of line durations"
        )
    if script.estimated_duration_seconds < config.min_duration_seconds:
        warnings.append(
            f"duration {script.estimated_duration_seconds:.1f}s is below the "
            f"{config.min_duration_seconds:.0f}s target"
        )
    if script.estimated_duration_seconds > config.max_duration_seconds:
        warnings.append(
            f"duration {script.estimated_duration_seconds:.1f}s exceeds the "
            f"{config.max_duration_seconds:.0f}s Shorts target"
        )

    return ScriptValidation(errors=tuple(errors), warnings=tuple(warnings))


def validate_script_from_package(
    script: Script,
    package: ResearchPackage,
    config: Any,
) -> ScriptValidation:
    """Validate a generated script directly against its ResearchPackage.

    The package-based path (no fact-check result yet): every package claim is
    the working set, so a line may only cite claims present in the package.
    All other guarantees are identical to `validate_script` — factual lines
    must carry refs, claim usage must be tracked exactly and completely,
    factual wording must stay inside the referenced claims' token coverage,
    and hedges must survive (never strengthened into certainty). This is the
    deterministic final gate for script generation over a package.
    """
    errors: list[str] = []
    warnings: list[str] = []

    claims_by_id = {claim.claim_id: claim for claim in package.claims}
    package_claims = {claim.claim_id for claim in package.claims}

    # Provenance / binding to the exact research package.
    if script.provenance is None:
        errors.append("script has no provenance")
    if script.package_id != package.package_id:
        errors.append(
            f"script.package_id {script.package_id!r} does not match "
            f"research package {package.package_id!r}"
        )

    # Structural requirements (identical to validate_script).
    if not script.lines:
        errors.append("script has no lines")
    if not script.title.strip():
        errors.append("script title is empty")
    if not script.hook.strip():
        errors.append("script hook is empty")
    if not any(line.purpose == "hook" for line in script.lines):
        errors.append("script has no hook line")
    if not any(line.purpose in ("fact", "explanation") for line in script.lines):
        errors.append("script has no fact/explanation line")

    # Claim references: factual lines must be grounded in package claims.
    for line in script.lines:
        if line.purpose not in PURPOSE_VALUES:
            errors.append(f"line {line.line_id} has unknown purpose {line.purpose!r}")
            continue
        if not line.claim_refs:
            if line.is_factual():
                errors.append(f"factual line {line.line_id} has no claim_refs")
            continue
        for claim_id in line.claim_refs:
            if claim_id not in claims_by_id:
                errors.append(f"line {line.line_id} references unknown claim {claim_id}")

        if not line.is_factual():
            continue
        ref_claims = [claims_by_id[cid] for cid in line.claim_refs if cid in claims_by_id]
        if len(ref_claims) != len(line.claim_refs):
            continue  # missing claim already reported above
        coverage = _covered(ref_claims, line.text, config.semantic_coverage_threshold)
        if coverage is not None and coverage < config.semantic_coverage_threshold:
            errors.append(
                f"line {line.line_id} uses wording not in its approved claims "
                f"(token coverage {coverage:.2f})"
            )
        word = _strengthened(ref_claims, line.text)
        if word:
            errors.append(
                f"line {line.line_id} strengthens its claims with unsupported "
                f"certainty word {word!r}"
            )
        if _hedge_dropped(ref_claims, line.text):
            errors.append(
                f"line {line.line_id} drops the hedge(s) of its approved claims"
            )

    # Claim usage tracking must be exact and complete over the package.
    used = script.all_claim_refs()
    if set(script.used_claim_ids) != used:
        errors.append("used_claim_ids is not the exact set of claim_refs")
    if set(script.unused_approved_claim_ids) & used:
        errors.append("unused_approved_claim_ids overlaps used_claim_ids")
    if (set(script.used_claim_ids) | set(script.unused_approved_claim_ids)) != package_claims:
        errors.append(
            "used + unused claim ids do not cover the package claim set"
        )

    # Duration: declared total must match the derived sum; bounds are warnings.
    derived = round(sum(line.estimated_duration_seconds for line in script.lines), 2)
    if abs(derived - script.estimated_duration_seconds) > DURATION_TOLERANCE_SECONDS:
        errors.append(
            "estimated_duration_seconds does not match the derived sum of line durations"
        )
    if script.estimated_duration_seconds < config.min_duration_seconds:
        warnings.append(
            f"duration {script.estimated_duration_seconds:.1f}s is below the "
            f"{config.min_duration_seconds:.0f}s target"
        )
    if script.estimated_duration_seconds > config.max_duration_seconds:
        warnings.append(
            f"duration {script.estimated_duration_seconds:.1f}s exceeds the "
            f"{config.max_duration_seconds:.0f}s Shorts target"
        )

    return ScriptValidation(errors=tuple(errors), warnings=tuple(warnings))