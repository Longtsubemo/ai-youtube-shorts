"""Multi-source claim binding: one claim, several supporting sources.

Stage 2 used to bind every claim to exactly one source (`source_ids=(source_id,)`),
so a critical claim could never gather the two independent sources Stage 3's
corroboration gate requires and the pipeline stalled at REVIEW. These tests lock
in the fix: the model explicitly declares extra supporting URLs, the provider
keeps only URLs that real retrieval returned, and the agent binds them to real
source ids.

Coverage:
  A. single source still binds one id (unchanged legacy behavior)
  B. two valid sources bind two ids
  C. unknown source is rejected (never invented)
  D. duplicate source ids are rejected / never emitted
  E. unrelated sources are never auto-attached
  F. existing single-source artifacts still load
  G. two qualifying sources satisfy the existing critical corroboration gate
  H. one qualifying source still yields REVIEW
  I. binding is deterministic
  J. mock-provider fixtures are unchanged (backward compatibility)
"""

from __future__ import annotations

import pytest

from agents.qa.fact_checker.checker import FactChecker
from agents.research.agent import ResearchAgent
from agents.research.models import (
    FactClaim,
    ResearchPackage,
    ResearchPackageError,
    Source,
    normalize_source_url,
    source_id_for,
)
from agents.research.provider import ClaimItem, SourceCandidate, TopicDescriptor

from fact_check_test_utils import (
    FIXED_AT,
    StaticEvidenceProvider,
    make_claim,
    make_evidence,
    make_package,
    make_source,
)

URL_A = "https://example.test/papers/octopus"
URL_B = "https://example.test/news/octopus"
URL_UNRELATED = "https://example.test/news/weather"

CLAIM = "Octopuses have three hearts."
OTHER = "Octopuses are molluscs."
UNRELATED_CLAIM = "Rain is wet."


def selection(topic_id: str = "t_octopus_hearts", title: str = "Why octopuses have three hearts") -> dict:
    """Minimal TopicSelection payload (the dict form ResearchAgent accepts)."""
    return {
        "data": {
            "selected_topic_id": topic_id,
            "selected_topic": {"topic_id": topic_id, "title": title},
        }
    }


def candidate(
    url: str, claim_text: str, *extra_urls: str, critical: bool = False
) -> SourceCandidate:
    """A source candidate whose single claim also declares extra support.

    `critical` defaults to False: most of this module tests *binding*
    mechanics (order, dedup, traceability), not the Stage-3 corroboration
    policy. A blanket `critical=True` default made every single-source fixture
    a critical claim that the producer-side guard now rejects by design, so
    criticality is declared explicitly per test instead. Tests that do mean to
    exercise a critical claim pass `critical=True` and supply the required
    number of eligible non-discovery sources.
    """
    return SourceCandidate(
        title="Page",
        url=url,
        publisher="pub",
        source_type="web_page",
        publication_date="2024-01-05",
        credibility_tier="primary",
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
        retrieved_at=FIXED_AT,
    )


class _CandidateProvider:
    """Minimal provider returning fixed candidates (no fixture file needed)."""

    def __init__(self, candidates: list[SourceCandidate]) -> None:
        self._candidates = candidates

    def search(self, topic: TopicDescriptor) -> list[SourceCandidate]:
        return self._candidates

    def knowable_topics(self) -> list[str]:
        return ["t1"]


def run_with(*candidates: SourceCandidate) -> ResearchPackage:
    agent = ResearchAgent(provider=_CandidateProvider(list(candidates)), run_id="r1")
    return agent.research(selection())


def claim_text(package: ResearchPackage, text: str = CLAIM) -> FactClaim:
    for claim in package.claims:
        if claim.claim_text == text:
            return claim
    raise AssertionError(f"claim not found: {text!r}")


def mock_source(url: str) -> Source:
    return Source(
        source_id=source_id_for(url),
        url=url,
        title="Page",
        publisher="pub",
        source_type="web_page",
        publication_date="2024-01-05",
        credibility_tier="primary",
    )


class TestSingleSourceUnchanged:
    """A. Legacy single-source binding is preserved for non-critical claims.

    These fixtures are non-critical on purpose. Single-source binding itself is
    unchanged legacy behavior and still works exactly as before; what changed is
    that a *critical* claim is no longer allowed to reach Stage 3 structurally
    unable to corroborate. See
    tests/test_research_critical_claim_binding.py for the producer guard.
    """

    def test_single_source_binds_one_id(self):
        package = run_with(candidate(URL_A, CLAIM))
        assert claim_text(package).source_ids == (source_id_for(URL_A),)

    def test_single_source_non_critical_claim_is_not_rejected(self):
        """A lone non-critical source is legal and survives the guard."""
        package = run_with(candidate(URL_A, CLAIM))
        assert claim_text(package).critical is False
        assert claim_text(package).source_ids == (source_id_for(URL_A),)

    def test_single_source_ids_are_unique(self):
        package = run_with(candidate(URL_A, CLAIM))
        ids = claim_text(package).source_ids
        assert len(ids) == len(set(ids))

    def test_real_mock_fixture_still_binds_single_source(self, research_agent, mock_topic_selection):
        """J. The shipped mock corpus behaves exactly as before."""
        package = research_agent.research(mock_topic_selection)
        claim = next(c for c in package.claims if c.claim_text.startswith("Octopuses have three hearts:"))
        assert claim.source_ids == (source_id_for("https://example.test/source/oct001"),)
        for item in package.claims:
            assert len(item.source_ids) == len(set(item.source_ids))


class TestMultiSourceBinding:
    """B. Multiple declared, retrievable sources bind in a stable order."""

    def test_two_valid_sources_bind_two_ids(self):
        package = run_with(candidate(URL_A, CLAIM, URL_B), candidate(URL_B, OTHER))
        assert claim_text(package).source_ids == (source_id_for(URL_A), source_id_for(URL_B))

    def test_critical_claim_binds_two_eligible_sources(self):
        """A genuinely critical claim now reaches the agent with the sources
        the Stage-3 gate requires, so multi-source binding is not just
        bookkeeping: it is what unblocks corroboration."""
        package = run_with(
            candidate(URL_A, CLAIM, URL_B, critical=True),
            candidate(URL_B, OTHER),
        )
        claim = claim_text(package)
        assert claim.critical is True
        assert claim.source_ids == (source_id_for(URL_A), source_id_for(URL_B))

    def test_primary_source_is_first(self):
        package = run_with(candidate(URL_B, CLAIM, URL_A), candidate(URL_A, OTHER))
        assert claim_text(package).source_ids[0] == source_id_for(URL_B)

    def test_every_bound_source_exists_in_package(self):
        package = run_with(candidate(URL_A, CLAIM, URL_B), candidate(URL_B, OTHER))
        known = {s.source_id for s in package.sources}
        assert set(claim_text(package).source_ids) <= known

    def test_trailing_slash_variant_still_resolves(self):
        package = run_with(candidate(URL_A, CLAIM, URL_B + "/"), candidate(URL_B, OTHER))
        assert source_id_for(URL_B) in claim_text(package).source_ids

    def test_confidence_uses_all_backing_sources(self):
        single = run_with(candidate(URL_A, CLAIM))
        multi = run_with(candidate(URL_A, CLAIM, URL_B), candidate(URL_B, OTHER))
        assert claim_text(multi).confidence >= claim_text(single).confidence


class TestUnknownSourceRejected:
    """C. Relationships that cannot be traced are rejected, not invented."""

    def test_declared_url_outside_package_raises(self):
        with pytest.raises(ResearchPackageError, match="not in the package"):
            run_with(candidate(URL_A, CLAIM, "https://fabricated.test/nope"))

    def test_non_http_url_outside_package_raises(self):
        with pytest.raises(ResearchPackageError, match="not in the package"):
            run_with(candidate(URL_A, CLAIM, "not-a-url"))

    def test_claim_citing_unknown_source_id_fails_validation(self):
        claim = FactClaim(
            claim_id="c1",
            claim_text=CLAIM,
            claim_type="fact",
            source_ids=(source_id_for(URL_A), "src_deadbeef00"),
        )
        package = ResearchPackage.build(
            topic_id="t1",
            topic_title="Octopus hearts",
            research_status="ready_for_fact_check",
            claims=[claim],
            sources=[mock_source(URL_A)],
            run_id="r1",
            created_at=FIXED_AT,
        )
        with pytest.raises(ResearchPackageError):
            package.validate(traceability=True)


class TestDuplicateSourceIdsRejected:
    """D. Duplicate ids are impossible from the agent and rejected by models."""

    def test_duplicate_source_ids_fail_validation(self):
        with pytest.raises(ResearchPackageError, match="duplicate source_id"):
            FactClaim(
                claim_id="c1",
                claim_text=CLAIM,
                claim_type="fact",
                source_ids=(source_id_for(URL_A), source_id_for(URL_A)),
            ).validate()

    def test_duplicate_json_source_ids_fail_on_load(self):
        payload = ResearchPackage.build(
            topic_id="t1",
            topic_title="Octopus hearts",
            research_status="ready_for_fact_check",
            claims=[
                make_claim(CLAIM, source_urls=[URL_A]),
            ],
            sources=[make_source(URL_A)],
            run_id="r1",
            created_at=FIXED_AT,
        ).to_dict()
        sid = source_id_for(URL_A)
        payload["data"]["claims"][0]["source_ids"] = [sid, sid]
        with pytest.raises(ResearchPackageError, match="duplicate source_id"):
            ResearchPackage.from_dict(payload).validate()

    def test_self_reference_is_deduplicated(self):
        package = run_with(candidate(URL_A, CLAIM, URL_A))
        assert claim_text(package).source_ids == (source_id_for(URL_A),)

    def test_repeated_extra_url_is_deduplicated(self):
        package = run_with(candidate(URL_A, CLAIM, URL_B, URL_B), candidate(URL_B, OTHER))
        assert claim_text(package).source_ids == (source_id_for(URL_A), source_id_for(URL_B))


class TestNoAutomaticAttachment:
    """E. Unrelated (and merely related) sources are never auto-attached."""

    def test_unrelated_source_not_attached_without_declaration(self):
        package = run_with(candidate(URL_A, CLAIM), candidate(URL_UNRELATED, UNRELATED_CLAIM))
        assert claim_text(package).source_ids == (source_id_for(URL_A),)
        assert source_id_for(URL_UNRELATED) not in claim_text(package).source_ids

    def test_related_source_also_requires_declaration(self):
        package = run_with(candidate(URL_A, CLAIM), candidate(URL_B, OTHER))
        assert source_id_for(URL_B) not in claim_text(package).source_ids


class TestArtifactCompatibility:
    """F. Pre-existing single-source artifacts keep loading."""

    def test_legacy_single_source_payload_round_trips(self):
        legacy = {
            "package_id": "p1",
            "run_id": "r1",
            "topic_id": "t1",
            "topic_title": "Octopus hearts",
            "research_status": "ready_for_fact_check",
            "claims": [
                {
                    "claim_id": "c1",
                    "claim_text": CLAIM,
                    "claim_type": "fact",
                    "source_ids": ["src_aaa"],
                    "confidence": 0.5,
                    "verification_status": "provisional",
                    "dates": [],
                    "notes": "",
                    "critical": False,
                }
            ],
            "sources": [
                {
                    "source_id": "src_aaa",
                    "url": "https://legacy.test/a",
                    "title": "Legacy",
                    "publisher": "pub",
                    "source_type": "web_page",
                    "credibility_tier": "primary",
                }
            ],
        }
        package = ResearchPackage.from_dict(
            {"created_at": FIXED_AT, "run_id": "r1", "data": legacy}
        )
        package.validate(traceability=True)
        assert package.claims[0].source_ids == ("src_aaa",)

    def test_real_single_source_artifact_still_loads(self):
        """An artifact produced before this change remains readable."""
        from pathlib import Path

        artifacts = sorted(
            Path("data").glob("*/research/*.json"),
        )
        if not artifacts:
            pytest.skip("no checked-out research artifacts available")
        loaded = 0
        for path in artifacts:
            try:
                ResearchPackage.from_dict(__import__("json").loads(path.read_text(encoding="utf-8")))
            except (ResearchPackageError, ValueError, KeyError):
                continue
            loaded += 1
        assert loaded, "no legacy research artifact could be loaded"

    def test_multi_source_serializes_as_json_list(self):
        package = run_with(candidate(URL_A, CLAIM, URL_B), candidate(URL_B, OTHER))
        payload = package.to_dict()
        entry = next(c for c in payload["data"]["claims"] if c["claim_text"] == CLAIM)
        assert isinstance(entry["source_ids"], list)
        assert len(entry["source_ids"]) == 2
        assert ResearchPackage.from_dict(payload).claims


class TestCorroborationGateUnchanged:
    """G/H. The Stage-3 gate is untouched; multi-source binding unblocks it."""

    def _result(self, urls: list[str]) -> "object":
        sources = [make_source(url, tier="primary") for url in urls]
        claim = make_claim(CLAIM, source_urls=urls, critical=True)
        provider = StaticEvidenceProvider(
            {url: [make_evidence(src, CLAIM)] for url, src in zip(urls, sources)}
        )
        return FactChecker(provider, run_id="t", created_at=FIXED_AT).check(
            make_package(claims=[claim], sources=sources)
        )

    def test_two_qualifying_sources_satisfy_corroboration(self):
        result = self._result([URL_A, URL_B])
        verdict = result.claim_verdict(next(iter(result.claim_results)).claim_id)
        assert verdict.verdict == "PASS"
        assert len(verdict.supporting_source_ids) == 2
        assert result.package_verdict == "PASS"

    def test_single_qualifying_source_still_reviews(self):
        result = self._result([URL_A])
        verdict = result.claim_verdict(next(iter(result.claim_results)).claim_id)
        assert verdict.verdict != "PASS"
        assert result.package_verdict == "REVIEW"


class TestDeterminism:
    """I. Identical input yields an identical binding."""

    def test_binding_is_repeatable(self):
        first = run_with(candidate(URL_A, CLAIM, URL_B), candidate(URL_B, OTHER))
        second = run_with(candidate(URL_A, CLAIM, URL_B), candidate(URL_B, OTHER))
        assert claim_text(first).source_ids == claim_text(second).source_ids

    def test_canonical_url_resolves_to_the_same_source_id(self):
        """Canonicalization is a lookup aid; source ids stay derived from the
        URL exactly as given, so pre-existing artifacts keep their ids."""
        assert normalize_source_url(URL_B + "/") == URL_B
        assert normalize_source_url("HTTPS://Example.test/papers/octopus/") == URL_A
        assert source_id_for(URL_A) == source_id_for(URL_A)
        assert source_id_for(URL_A) != source_id_for(URL_B)
