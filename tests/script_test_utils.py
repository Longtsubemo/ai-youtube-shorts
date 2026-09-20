"""Shared builders/stubs for the Stage-4 (script generation) test suite.

Everything here is synthetic and MOCK - same rule as the Stage-3 builders:
no source stands behind these claims and nothing is asserted true. Builders
exist so tests can construct exact approved/review/rejected/insufficient
scenarios offline.
"""

from __future__ import annotations

from agents.qa.fact_checker.models import ClaimVerdict, FactCheckResult, result_id_for
from agents.script.models import Script, ScriptLine, line_id_for, script_id_for
from agents.seo.models import Provenance
from agents.research.models import FactClaim
from fact_check_test_utils import FIXED_AT, make_claim, make_package

# The deterministic octopus claim set the mock research stage produces.
OCTOPUS_THREE_HEARTS = (
    "Octopuses have three hearts: two branchial hearts that push blood through "
    "the gills and a systemic heart that powers the body."
)
OCTOPUS_SWIM = (
    "While octopuses swim, the two branchial hearts stop beating because the "
    "body diverts oxygen to the systemic heart."
)
OCTOPUS_BLUE = (
    "Octopus blood uses copper-based hemocyanin, which means their blood appears blue."
)
OCTOPUS_SMARTEST = "Octopuses are probably the smartest invertebrates in the sea."
OCTOPUS_TANK = "Some owners say a pet octopus may someday learn to open a tank lid."

DEFAULT_WORDS_PER_MINUTE = 150

_MISSING = object()


def make_verdict(
    claim_id: str,
    verdict: str = "PASS",
    confidence: float | None = None,
    critical: bool = False,
) -> ClaimVerdict:
    """Deterministic mock ClaimVerdict (no evidence linkage needed for scripts)."""
    defaults = {"PASS": 0.8, "REVIEW": 0.5, "REJECT": 0.3}
    return ClaimVerdict(
        claim_id=claim_id,
        verdict=verdict,
        evidence_ids=(),
        supporting_source_ids=(),
        contradiction_source_ids=(),
        rationale=f"mock rationale for {claim_id}",
        confidence=confidence if confidence is not None else defaults[verdict],
        critical=critical,
        checks_performed=(),
        verifier="MockVerifier",
    ).validate()


def build_pair(
    claims=(),
    verdicts=None,
    package_verdict="PASS",
    topic_id="t_test",
    topic_title="Why octopuses have three hearts",
    run_id="script-test",
    created_at=FIXED_AT,
) -> tuple[FactCheckResult, object]:
    """Build a (FactCheckResult, ResearchPackage) pair from (text, type) claims.

    `verdicts` maps claim TEXT to PASS/REVIEW/REJECT; anything not listed is
    PASS. The resulting result/package keeps approved/review/rejected lists and
    the package verdict consistent, and the package resolves long claim texts.
    """
    claims = [make_claim(text, claim_type=ctype) for text, ctype in claims]
    verdicts = verdicts or {}
    claim_results = tuple(
        make_verdict(claim.claim_id, verdict=verdicts.get(claim.claim_text, "PASS"))
        for claim in claims
    )
    approved = tuple(c.claim_id for c in claims if verdicts.get(c.claim_text, "PASS") == "PASS")
    review = tuple(c.claim_id for c in claims if verdicts.get(c.claim_text, "PASS") == "REVIEW")
    rejected = tuple(c.claim_id for c in claims if verdicts.get(c.claim_text, "PASS") == "REJECT")

    package = make_package(
        claims=claims, topic_id=topic_id, topic_title=topic_title
    )
    result = FactCheckResult(
        result_id=result_id_for(run_id, package.package_id),
        topic_id=topic_id,
        topic_title=topic_title,
        package_id=package.package_id,
        claim_results=claim_results,
        package_verdict=package_verdict,
        review_items=review,
        rejected_claims=rejected,
        approved_claims=approved,
        created_at=created_at,
        run_id=run_id,
    )
    return result.validate(), package


def build_octopus_pair(
    approved=("fact", "explanation", "explanation"),
    package_verdict="PASS",
    run_id="script-test",
    created_at=FIXED_AT,
) -> tuple[FactCheckResult, object]:
    """Octopus scenario; verdicts given per claim index via the `approved` map.

    `approved` lists claim_types for the three verified facts; the last two
    claims (inference with living-speculation hedge, speculation) stay unapproved.
    """
    claims = [
        (OCTOPUS_THREE_HEARTS, "fact"),
        (OCTOPUS_SWIM, "explanation"),
        (OCTOPUS_BLUE, "explanation"),
        (OCTOPUS_SMARTEST, "inference"),
        (OCTOPUS_TANK, "speculation"),
    ]
    approved_types = set(approved)
    verdicts = {}
    for index, (text, ctype) in enumerate(claims):
        verdicts[text] = "PASS" if ctype in approved_types else ("REVIEW" if index == 3 else "REJECT")
    return build_pair(
        claims=claims,
        verdicts=verdicts,
        package_verdict=package_verdict,
        topic_id="t_octopus_hearts",
        topic_title="Why octopuses have three hearts",
        run_id=run_id,
        created_at=created_at,
    )


def rounded_duration(text: str, words_per_minute: int = DEFAULT_WORDS_PER_MINUTE) -> float:
    return round(len(text.split()) / words_per_minute * 60, 2)


def make_script(
    result: FactCheckResult,
    lines=None,
    title=None,
    hook=None,
    used=None,
    unused=None,
    provenance=_MISSING,
    result_id=None,
    estimated_duration=None,
    created_at=FIXED_AT,
    run_id="script-test",
) -> Script:
    """Deterministic Script builder; derived fields can be overridden to corrupt.

    `lines` is an iterable of ScriptLine-like dicts with keys
    text/purpose/claim_refs (line_id and duration are derived). Pass
    `provenance=None` to build a script without provenance (validation error).
    """
    lines = (
        [{"text": OCTOPUS_THREE_HEARTS, "purpose": "fact", "claim_refs": ()}]
        if lines is None
        else list(lines)
    )
    built = []
    for index, spec in enumerate(lines):
        text = spec["text"]
        durations = rounded_duration(
            text, DEFAULT_WORDS_PER_MINUTE
        ) if "estimated_duration_seconds" not in spec else spec["estimated_duration_seconds"]
        built.append(
            ScriptLine(
                line_id=line_id_for(index),
                text=text,
                claim_refs=tuple(spec.get("claim_refs", ())),
                purpose=spec["purpose"],
                estimated_duration_seconds=durations,
            ).validate()
        )
    used_ids = tuple(sorted({ref for line in built for ref in line.claim_refs})) if used is None else used
    unused_ids = tuple(sorted(set(result.approved_claims) - set(used_ids))) if unused is None else unused
    return Script(
        script_id=script_id_for(run_id, result.topic_id),
        topic_id=result.topic_id,
        topic_title=result.topic_title,
        title=title or result.topic_title,
        hook=hook if hook is not None else (built[0].text if built else ""),
        lines=tuple(built),
        estimated_duration_seconds=(
            round(sum(line.estimated_duration_seconds for line in built), 1)
            if estimated_duration is None
            else estimated_duration
        ),
        used_claim_ids=used_ids,
        unused_approved_claim_ids=unused_ids,
        result_id=result_id or result.result_id,
        package_id=result.package_id,
        provenance=(
            Provenance(
                generated_by="test",
                source="test-builder",
                created_at=created_at,
                provider=None,
                model="test",
                prompt_version=None,
            )
            if provenance is _MISSING
            else provenance
        ),
        created_at=created_at,
        run_id=run_id,
    )


def fact_result_from_fixture() -> FactCheckResult:
    """Load the committed mock fixture (data/mock/fact_check_result.json)."""
    from agents.script.stage import load_fact_check_result

    return load_fact_check_result()


def mock_octopus_package():
    """Rebuild the deterministic mock octopus package in memory (no files)."""
    from agents.research.stage import load_topic_selection, run_research
    from agents.research.provider import MockResearchProvider

    selection = load_topic_selection()
    return run_research(selection, MockResearchProvider(), run_id="mock")