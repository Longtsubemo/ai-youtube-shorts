"""Stage 2 emits under-corroborated critical claims; Stage 3 grades them.

History
-------
A producer-side guard, `ensure_critical_claim_is_corroborable()`, used to reject
a `ResearchPackage` whose critical claim was bound to fewer than
`FactCheckConfig.corroboration_min_sources` eligible non-discovery sources. In
the real run `e2e-20261002-192946` the critical claim `cl_1b58fabb1b` was bound
to three sources but only one was non-discovery, so the whole run aborted in
Stage 2 with no artifact at all.

That guard was redundant, not additional, safety: Stage 3 already applies the
identical policy in `evaluate_claim` (the corroboration gate) and
`aggregate_package`, and it does so against the evidence Stage 3 actually
retrieves. Because the guard ran *before* emission it could only ever replace a
graded REVIEW with an ungraded crash. It has been removed.

The contract these tests lock in now:

    Stage 2  ->  emits any critical claim whose source_ids all resolve
    Stage 3  ->  decides PASS / REVIEW / REJECT, unchanged

Stage 2's only claim-level obligation is traceability. Source tiering,
corroboration, evidence grounding and package aggregation are all untouched and
all still belong to Stage 3.

Coverage:
   A. under-corroborated critical claims are emitted, not rejected
   B. Stage 3 still grades them REVIEW (regression on the removed guard)
   C. binding mechanics are unchanged
   D. discovery sources are neither promoted nor made authoritative
   E. non-critical claims are unchanged
   F. existing/legacy artifacts still load, and gain no new field
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agents.qa.fact_checker.checker import FactChecker
from agents.qa.fact_checker.gates import FactCheckConfig
from agents.research.agent import ResearchAgent
from agents.research.models import (
    FactClaim,
    ResearchPackage,
    ResearchPackageError,
    Source,
    source_id_for,
)
from agents.research.provider import ClaimItem, SourceCandidate, TopicDescriptor

from tests.fact_check_test_utils import (
    FIXED_AT,
    StaticEvidenceProvider,
    make_evidence,
    make_source,
)

URL_A = "https://example.test/papers/octopus"
URL_B = "https://example.test/news/octopus"
URL_C = "https://example.test/journals/octopus"
URL_UNRELATED = "https://example.test/news/weather"

CLAIM = "Octopuses have three hearts."
OTHER = "Octopuses are molluscs."
UNRELATED_CLAIM = "Rain is wet."

PRIMARY = "primary"
SECONDARY = "high_quality_secondary"
DISCOVERY = "discovery"


def required_corroboration_sources() -> int:
    """The Stage-3 requirement, read from the real (unmodified) gate config."""
    return FactCheckConfig().corroboration_min_sources


def eligible_non_discovery_support(
    source_ids: tuple[str, ...], sources_by_id: dict[str, Source]
) -> tuple[str, ...]:
    """Bound ids whose own `credibility_tier` is not discovery.

    This is the eligibility rule the removed producer guard used. It is
    re-expressed here against `Source.is_non_discovery` - the same property
    `evaluate_claim` and `evaluate_research_status` use - so these tests keep
    describing the real policy without depending on deleted production code.
    """
    return tuple(
        source_id
        for source_id in source_ids
        if (source := sources_by_id.get(source_id)) is not None and source.is_non_discovery
    )


def selection(topic_id: str = "t_octopus_hearts", title: str = "Why octopuses have three hearts") -> dict:
    return {
        "data": {
            "selected_topic_id": topic_id,
            "selected_topic": {"topic_id": topic_id, "title": title},
        }
    }


def candidate(
    url: str,
    claim_text: str,
    *extra_urls: str,
    tier: str = PRIMARY,
    critical: bool = False,
) -> SourceCandidate:
    """A source candidate whose claim may declare extra supporting URLs."""
    return SourceCandidate(
        title="Page",
        url=url,
        publisher="pub",
        source_type="web_page",
        publication_date="2024-01-05",
        credibility_tier=tier,
        notes="",
        claims=(
            ClaimItem(
                claim_text=claim_text,
                dates=(),
                suggested_claim_type="fact",
                critical=critical,
                also_supported_by_urls=tuple(extra_urls),
            ),
        ),
    )


class _CandidateProvider:
    def __init__(self, candidates: list[SourceCandidate]) -> None:
        self._candidates = candidates

    def search(self, topic: TopicDescriptor) -> list[SourceCandidate]:
        return self._candidates

    def knowable_topics(self) -> list[str]:
        return ["t1"]


def run_with(*candidates: SourceCandidate) -> ResearchPackage:
    agent = ResearchAgent(provider=_CandidateProvider(list(candidates)), run_id="r1")
    return agent.research(selection())


def claim_text_of(package: ResearchPackage, text: str) -> FactClaim:
    return next(claim for claim in package.claims if claim.claim_text == text)


def tiered_source(url: str, tier: str) -> Source:
    return Source(
        source_id=source_id_for(url),
        url=url,
        title="Page",
        publisher="pub",
        source_type="web_page",
        publication_date="2024-01-05",
        credibility_tier=tier,
    )


def evidence_for(package: ResearchPackage) -> dict:
    """Direct, verbatim evidence for every claim, drawn from its own sources."""
    mapping: dict[str, list] = {}
    for claim in package.claims:
        for source in package.sources:
            if source.source_id in claim.source_ids:
                mapping.setdefault(source.url, []).append(
                    make_evidence(source, claim.claim_text)
                )
    return mapping


def two_source_ids() -> tuple[tuple[str, ...], dict[str, Source]]:
    source_ids = (source_id_for(URL_A), source_id_for(URL_B))
    sources_by_id = {
        source_id_for(URL_A): tiered_source(URL_A, PRIMARY),
        source_id_for(URL_B): tiered_source(URL_B, SECONDARY),
    }
    return source_ids, sources_by_id


# ---------------------------------------------------------------------- A --
# The regression: what the removed guard used to reject must now be emitted.


class TestUnderCorroboratedCriticalClaimsAreEmitted:
    """A. Stage 2 no longer fails a package over corroboration."""

    def test_single_source_critical_claim_is_emitted(self):
        """The exact defect shape: critical claim, one source, nothing else."""
        package = run_with(candidate(URL_A, CLAIM, critical=True), candidate(URL_B, OTHER))
        claim = claim_text_of(package, CLAIM)
        assert claim.critical is True
        assert claim.source_ids == (source_id_for(URL_A),)

    def test_zero_eligible_source_critical_claim_is_emitted(self):
        """Even with no non-discovery support at all, emission succeeds."""
        package = run_with(
            candidate(URL_A, CLAIM, tier=DISCOVERY, critical=True),
            candidate(URL_B, OTHER, tier=DISCOVERY),
        )
        claim = claim_text_of(package, CLAIM)
        assert claim.critical is True
        assert claim.source_ids == (source_id_for(URL_A),)

    def test_one_of_three_non_discovery_is_emitted(self):
        """Mirrors the real run: 3 bound sources, 1 eligible."""
        package = run_with(
            candidate(URL_A, CLAIM, URL_B, URL_C, tier=PRIMARY, critical=True),
            candidate(URL_B, OTHER, tier=DISCOVERY),
            candidate(URL_C, UNRELATED_CLAIM, tier=DISCOVERY),
        )
        claim = claim_text_of(package, CLAIM)
        assert len(claim.source_ids) == 3

    def test_emitted_package_validates(self):
        package = run_with(candidate(URL_A, CLAIM, critical=True), candidate(URL_B, OTHER))
        assert package.validate() is package

    def test_emitted_critical_claim_really_is_under_corroborated(self):
        """Guards the test itself: the fixture must still be under the bar."""
        package = run_with(candidate(URL_A, CLAIM, critical=True), candidate(URL_B, OTHER))
        claim = claim_text_of(package, CLAIM)
        sources_by_id = {source.source_id: source for source in package.sources}
        eligible = eligible_non_discovery_support(claim.source_ids, sources_by_id)
        assert len(eligible) == 1
        assert len(eligible) < required_corroboration_sources()

    def test_research_status_still_reports_backing(self):
        """Emission does not fabricate a `ready_for_fact_check` status."""
        package = run_with(candidate(URL_A, CLAIM, critical=True), candidate(URL_B, OTHER))
        assert package.research_status == "ready_for_fact_check"
        all_discovery = run_with(
            candidate(URL_A, CLAIM, tier=DISCOVERY, critical=True),
            candidate(URL_B, OTHER, tier=DISCOVERY),
        )
        assert all_discovery.research_status == "provisional"


# ---------------------------------------------------------------------- B --
# Removing the producer guard must not remove the policy it duplicated.


class TestStage3StillGradesTheSameClaim:
    """B. The Stage-3 corroboration gate is untouched and still applies."""

    def test_single_source_critical_claim_is_reviewed_by_stage_3(self):
        package = run_with(candidate(URL_A, CLAIM, critical=True), candidate(URL_B, OTHER))
        result = FactChecker(StaticEvidenceProvider(evidence_for(package))).check(package)
        claim = claim_text_of(package, CLAIM)
        verdict = result.claim_verdict(claim.claim_id)
        assert verdict.verdict == "REVIEW"
        assert "corroborated by only 1" in verdict.rationale
        assert claim.claim_id not in result.approved_claims

    def test_package_aggregation_is_unchanged_by_the_removal(self):
        """A critical REVIEW still makes the whole package REVIEW."""
        package = run_with(candidate(URL_A, CLAIM, critical=True), candidate(URL_B, OTHER))
        result = FactChecker(StaticEvidenceProvider(evidence_for(package))).check(package)
        assert result.package_verdict == "REVIEW"
        assert claim_text_of(package, CLAIM).claim_id in result.critical_claim_failures

    def test_two_eligible_sources_still_reaches_pass(self):
        package = run_with(
            candidate(URL_A, CLAIM, URL_B, critical=True),
            candidate(URL_B, OTHER),
        )
        claim = claim_text_of(package, CLAIM)
        assert len(claim.source_ids) == 2
        result = FactChecker(StaticEvidenceProvider(evidence_for(package))).check(package)
        assert result.claim_verdict(claim.claim_id).verdict == "PASS"
        assert result.package_verdict == "PASS"
        assert result.critical_claim_failures == ()

    def test_corroboration_min_sources_is_still_two(self):
        assert required_corroboration_sources() == 2

    def test_stage_3_does_not_import_the_removed_guard(self):
        """Stage 3 never depended on the producer-side helper."""
        import agents.qa.fact_checker.gates as gates_module

        assert not hasattr(gates_module, "ensure_critical_claim_is_corroborable")
        assert FactCheckConfig().corroboration_min_sources == 2


# ---------------------------------------------------------------------- C --
# Binding mechanics are untouched by this change.


class TestBindingUnchanged:
    """C. Source binding / id resolution behave exactly as before."""

    def test_binds_two_eligible_sources(self):
        package = run_with(
            candidate(URL_A, CLAIM, URL_B, critical=True),
            candidate(URL_B, OTHER),
        )
        assert claim_text_of(package, CLAIM).source_ids == (
            source_id_for(URL_A),
            source_id_for(URL_B),
        )

    def test_binds_three_eligible_sources(self):
        package = run_with(
            candidate(URL_A, CLAIM, URL_B, URL_C, critical=True),
            candidate(URL_B, OTHER),
            candidate(URL_C, UNRELATED_CLAIM),
        )
        assert claim_text_of(package, CLAIM).source_ids == (
            source_id_for(URL_A),
            source_id_for(URL_B),
            source_id_for(URL_C),
        )

    def test_binding_order_is_preserved(self):
        package = run_with(
            candidate(URL_A, CLAIM, URL_B, URL_C),
            candidate(URL_B, OTHER),
            candidate(URL_C, UNRELATED_CLAIM),
        )
        assert claim_text_of(package, CLAIM).source_ids[0] == source_id_for(URL_A)

    def test_primary_source_is_still_first(self):
        package = run_with(
            candidate(URL_B, CLAIM, URL_A),
            candidate(URL_A, OTHER),
        )
        assert claim_text_of(package, CLAIM).source_ids[0] == source_id_for(URL_B)

    def test_trailing_slash_variant_still_resolves(self):
        package = run_with(
            candidate(URL_A, CLAIM, URL_B + "/"),
            candidate(URL_B, OTHER),
        )
        assert source_id_for(URL_B) in claim_text_of(package, CLAIM).source_ids

    def test_duplicate_urls_still_deduplicated(self):
        package = run_with(
            candidate(URL_A, CLAIM, URL_B, URL_B),
            candidate(URL_B, OTHER),
        )
        assert claim_text_of(package, CLAIM).source_ids == (
            source_id_for(URL_A),
            source_id_for(URL_B),
        )

    def test_self_reference_still_deduplicated(self):
        package = run_with(
            candidate(URL_A, CLAIM, URL_A),
            candidate(URL_B, OTHER),
        )
        assert claim_text_of(package, CLAIM).source_ids == (source_id_for(URL_A),)

    def test_unresolvable_declared_url_still_rejected(self):
        """Binding integrity is independent of the removed guard."""
        with pytest.raises(ResearchPackageError, match="not in the package"):
            run_with(
                candidate(URL_A, CLAIM, "https://fabricated.test/nope", URL_B),
                candidate(URL_B, OTHER),
            )

    def test_binding_is_deterministic(self):
        first = run_with(candidate(URL_A, CLAIM, URL_B), candidate(URL_B, OTHER))
        second = run_with(candidate(URL_A, CLAIM, URL_B), candidate(URL_B, OTHER))
        assert [c.source_ids for c in first.claims] == [c.source_ids for c in second.claims]

    def test_no_unrelated_source_is_auto_attached(self):
        """Emission must not pad a claim to satisfy a count."""
        package = run_with(
            candidate(URL_A, CLAIM, critical=True),
            candidate(URL_B, OTHER),
            candidate(URL_UNRELATED, UNRELATED_CLAIM),
        )
        assert source_id_for(URL_UNRELATED) not in claim_text_of(package, CLAIM).source_ids


# ---------------------------------------------------------------------- D --
# Removing the guard must not make weak sources authoritative.


class TestDiscoverySourcesStayWeak:
    """D. Nothing about tiering changed; discovery is still not authority."""

    def test_discovery_excluded_from_eligibility(self):
        source_ids = (source_id_for(URL_A), source_id_for(URL_B))
        sources_by_id = {
            source_id_for(URL_A): tiered_source(URL_A, DISCOVERY),
            source_id_for(URL_B): tiered_source(URL_B, DISCOVERY),
        }
        assert eligible_non_discovery_support(source_ids, sources_by_id) == ()

    def test_high_quality_secondary_counts(self):
        source_ids, sources_by_id = two_source_ids()
        assert eligible_non_discovery_support(source_ids, sources_by_id) == source_ids

    def test_eligibility_preserves_bound_order(self):
        source_ids = (source_id_for(URL_B), source_id_for(URL_A), source_id_for(URL_C))
        sources_by_id = {
            source_id_for(URL_A): tiered_source(URL_A, PRIMARY),
            source_id_for(URL_B): tiered_source(URL_B, SECONDARY),
            source_id_for(URL_C): tiered_source(URL_C, DISCOVERY),
        }
        assert eligible_non_discovery_support(source_ids, sources_by_id) == source_ids[:2]

    def test_unknown_bound_id_is_not_counted(self):
        source_ids = (source_id_for(URL_A), "src_notarealsource00")
        sources_by_id = {source_id_for(URL_A): tiered_source(URL_A, PRIMARY)}
        assert eligible_non_discovery_support(source_ids, sources_by_id) == (source_id_for(URL_A),)

    def test_emitting_a_discovery_only_claim_does_not_upgrade_its_tier(self):
        package = run_with(
            candidate(URL_A, CLAIM, tier=DISCOVERY, critical=True),
            candidate(URL_B, OTHER, tier=DISCOVERY),
        )
        tiers = {source.url: source.credibility_tier for source in package.sources}
        assert tiers[URL_A] == DISCOVERY

    def test_discovery_only_critical_claim_is_not_approved_by_stage_3(self):
        package = run_with(
            candidate(URL_A, CLAIM, tier=DISCOVERY, critical=True),
            candidate(URL_B, OTHER, tier=DISCOVERY),
        )
        result = FactChecker(StaticEvidenceProvider(evidence_for(package))).check(package)
        claim = claim_text_of(package, CLAIM)
        assert result.claim_verdict(claim.claim_id).verdict != "PASS"
        assert claim.claim_id not in result.approved_claims

    def test_confidence_still_reflects_tier(self):
        """A lone discovery source still yields a low-confidence claim."""
        discovery = run_with(
            candidate(URL_A, CLAIM, tier=DISCOVERY, critical=True),
            candidate(URL_B, OTHER, tier=DISCOVERY),
        )
        primary = run_with(candidate(URL_A, CLAIM, critical=True), candidate(URL_B, OTHER))
        assert (
            claim_text_of(discovery, CLAIM).confidence
            < claim_text_of(primary, CLAIM).confidence
        )


# ---------------------------------------------------------------------- E --


class TestNonCriticalClaimsUnchanged:
    """E. Criticality itself is still recorded, and still means nothing extra."""

    def test_non_critical_single_source_claim_still_binds(self):
        package = run_with(
            candidate(URL_A, CLAIM, critical=False),
            candidate(URL_B, OTHER),
        )
        assert claim_text_of(package, CLAIM).source_ids == (source_id_for(URL_A),)
        assert claim_text_of(package, CLAIM).critical is False

    def test_non_critical_discovery_only_claim_allowed(self):
        package = run_with(
            candidate(URL_A, CLAIM, tier=DISCOVERY, critical=False),
            candidate(URL_B, OTHER, tier=DISCOVERY),
        )
        assert claim_text_of(package, CLAIM).source_ids == (source_id_for(URL_A),)

    def test_mixed_package_records_criticality_per_claim(self):
        package = run_with(
            candidate(URL_A, CLAIM, URL_B, critical=True),
            candidate(URL_B, OTHER, critical=False),
        )
        assert claim_text_of(package, CLAIM).critical is True
        assert claim_text_of(package, OTHER).critical is False
        assert claim_text_of(package, OTHER).source_ids == (source_id_for(URL_B),)

    def test_real_mock_fixture_still_researches(self, research_agent, mock_topic_selection):
        package = research_agent.research(mock_topic_selection)
        assert package.claims
        assert package.sources


# ---------------------------------------------------------------------- F --


class TestExistingArtifactsRemainLoadable:
    """F. No schema change, and no new artifact field."""

    def test_legacy_payload_with_underbound_critical_claim_loads(self):
        legacy = {
            "package_id": "p1",
            "run_id": "r1",
            "topic_id": "t1",
            "topic_title": "Octopus hearts",
            "research_status": "ready_for_fact_check",
            "claims": [
                {
                    "claim_id": "cl_legacy",
                    "claim_text": CLAIM,
                    "claim_type": "fact",
                    "source_ids": [source_id_for(URL_A)],
                    "confidence": 0.5,
                    "verification_status": "provisional",
                    "dates": [],
                    "notes": "",
                    "critical": True,
                }
            ],
            "sources": [
                {
                    "source_id": source_id_for(URL_A),
                    "url": URL_A,
                    "title": "Legacy",
                    "publisher": "pub",
                    "source_type": "web_page",
                    "credibility_tier": PRIMARY,
                }
            ],
        }
        package = ResearchPackage.from_dict(
            {"created_at": FIXED_AT, "run_id": "r1", "data": legacy}
        )
        package.validate(traceability=True)
        assert package.claims[0].critical is True
        assert package.claims[0].source_ids == (source_id_for(URL_A),)

    def test_model_validation_is_not_gated_by_corroboration(self):
        package = ResearchPackage.build(
            topic_id="t1",
            topic_title="Octopus hearts",
            research_status="ready_for_fact_check",
            claims=[
                FactClaim(
                    claim_id="cl_store",
                    claim_text=CLAIM,
                    claim_type="fact",
                    source_ids=(source_id_for(URL_A),),
                    critical=True,
                )
            ],
            sources=[tiered_source(URL_A, PRIMARY)],
            run_id="r1",
            created_at=FIXED_AT,
        )
        assert package.validate() is package

    def test_real_run_with_the_defect_still_loads(self):
        """The exact artifact that motivated the removal must stay readable."""
        path = Path("data/e2e-20261002-192946/research/research_package.json")
        if not path.is_file():
            pytest.skip("real-run artifact not available")
        package = ResearchPackage.from_dict(json.loads(path.read_text(encoding="utf-8")))
        assert package.claims
        sources_by_id = {source.source_id: source for source in package.sources}
        underbound = [
            claim.claim_id
            for claim in package.claims
            if claim.critical
            and len(eligible_non_discovery_support(claim.source_ids, sources_by_id))
            < required_corroboration_sources()
        ]
        assert underbound == ["cl_1b58fabb1b"]

    def test_real_artifacts_still_load(self):
        artifacts = sorted(Path("data").glob("*/research/research_package.json"))
        if not artifacts:
            pytest.skip("no checked-out research artifacts available")
        loaded = 0
        for path in artifacts:
            try:
                ResearchPackage.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (ResearchPackageError, ValueError, KeyError):
                continue
            loaded += 1
        assert loaded, "no research artifact could be loaded"

    def test_package_serializes_without_guard_specific_fields(self):
        package = run_with(candidate(URL_A, CLAIM, URL_B), candidate(URL_B, OTHER))
        keys = set(claim_text_of(package, CLAIM).to_dict())
        assert keys == {
            "claim_id",
            "claim_text",
            "claim_type",
            "source_ids",
            "confidence",
            "verification_status",
            "dates",
            "notes",
            "critical",
        }


# --------------------------------------------------------------------------- #
# The removed guard must stay removed.


class TestProducerGuardIsGone:
    """Regression against reintroduction under another name."""

    def test_agent_module_exposes_no_producer_guard(self):
        import agents.research.agent as agent_module

        for name in (
            "ensure_critical_claim_is_corroborable",
            "required_corroboration_sources",
            "eligible_non_discovery_support",
        ):
            assert not hasattr(agent_module, name), name

    def test_research_agent_module_does_not_import_stage_3_gates(self):
        """Stage 2 must not depend on Stage 3, checked by real import behaviour.

        This deliberately does NOT grep the source text: a docstring or comment
        mentioning `agents.qa.fact_checker` would be a false positive, and an
        indirect or aliased import would be a false negative. Instead a clean
        interpreter imports `agents.research.agent` and we inspect what that
        actually pulled in. Any real dependency - module-level, function-local
        or lazy - necessarily loads its module into `sys.modules`.
        """
        probe = (
            "import sys, json\n"
            "import agents.research.agent\n"
            "print(json.dumps(sorted("
            "m for m in sys.modules if m.startswith('agents.qa.fact_checker')"
            ")))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout.strip()) == []

    def test_the_probe_would_detect_a_real_dependency(self):
        """Negative control: the probe above is not vacuously true.

        Guards against the import check silently degrading into a tautology -
        e.g. if the prefix being scanned stopped matching anything.
        """
        probe = (
            "import sys, json\n"
            "import agents.qa.fact_checker.gates\n"
            "print(json.dumps(sorted("
            "m for m in sys.modules if m.startswith('agents.qa.fact_checker')"
            ")))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        assert result.returncode == 0, result.stderr
        loaded = json.loads(result.stdout.strip())
        assert "agents.qa.fact_checker.gates" in loaded

    def test_research_agent_namespace_exposes_no_gate_objects(self):
        """No Stage-3 policy object is reachable from the research agent."""
        import agents.research.agent as agent_module

        for name in ("FactCheckConfig", "evaluate_claim", "aggregate_package"):
            assert not hasattr(agent_module, name), name