"""End-to-end tests for ResearchAgent over the mock provider."""

from __future__ import annotations

import pytest

from agents.research.agent import ResearchAgent
from agents.research.models import PROVISIONAL, ResearchPackage, ResearchPackageError, source_id_for
from agents.research.provider import ClaimItem, SourceCandidate, TopicDescriptor

OCTOPUS = "https://example.test/source/oct001"


def claim_by_text(package: ResearchPackage, needle: str):
    for claim in package.claims:
        if claim.claim_text.startswith(needle):
            return claim
    raise AssertionError(f"claim not found: {needle!r}")


class TestResearchAgentCore:
    def test_valid_selection_produces_ready_package(self, research_agent, mock_topic_selection):
        package = research_agent.research(mock_topic_selection)
        assert package.research_status == "ready_for_fact_check"
        assert package.topic_id == "t_octopus_hearts"
        assert package.topic_title == "Why octopuses have three hearts"
        assert len(package.claims) == 5
        assert len(package.sources) == 3

    def test_all_claims_are_provisional(self, research_agent, mock_topic_selection):
        package = research_agent.research(mock_topic_selection)
        assert all(claim.verification_status == PROVISIONAL for claim in package.claims)
        assert all("verified" != claim.verification_status for claim in package.claims)

    def test_claim_classification_and_provisional_confidence(self, research_agent, mock_topic_selection):
        package = research_agent.research(mock_topic_selection)
        expected = {
            "Octopuses have three hearts:": ("fact", 0.85),
            "While octopuses swim,": ("explanation", 0.85),
            "Octopus blood uses copper-based": ("explanation", 0.6),
            "Octopuses are probably": ("inference", 0.6),
            "Some owners say a pet octopus may someday": ("speculation", 0.3),
        }
        for prefix, (claim_type, confidence) in expected.items():
            claim = claim_by_text(package, prefix)
            assert claim.claim_type == claim_type, prefix
            assert claim.confidence == confidence, prefix

    def test_claim_to_source_traceability(self, research_agent, mock_topic_selection):
        package = research_agent.research(mock_topic_selection)
        source_ids = {source.source_id for source in package.sources}
        for claim in package.claims:
            assert claim.source_ids, "every claim must name a backing source"
            assert set(claim.source_ids) <= source_ids
        claim = claim_by_text(package, "Octopuses have three hearts:")
        assert claim.source_ids == (source_id_for(OCTOPUS),)
        assert package.source_by_id(source_id_for(OCTOPUS)).credibility_tier == "primary"
        assert package.claims_for_source(source_id_for(OCTOPUS)) and all(
            source_id_for(OCTOPUS) in c.source_ids
            for c in package.claims_for_source(source_id_for(OCTOPUS))
        )

    def test_all_claim_types_reachable_across_corpus(self, mock_research_provider, minimal_selection):
        agent = ResearchAgent(mock_research_provider, run_id="types")
        seen = set()
        for topic_id in mock_research_provider.knowable_topics():
            package = agent.research(minimal_selection(topic_id, topic_id))
            seen.update(claim.claim_type for claim in package.claims)
        assert {"fact", "inference", "explanation", "speculation", "opinion"} <= seen


class TestResearchStatuses:
    def test_insufficient_evidence(self, research_agent, minimal_selection):
        package = research_agent.research(minimal_selection("t_unknown_topic", "Completely unknown"))
        assert package.research_status == "insufficient_evidence"
        assert package.claims == ()
        assert package.sources == ()

    def test_provisional_when_only_discovery_evidence(self, research_agent, minimal_selection):
        package = research_agent.research(minimal_selection("t_first_bug", "The first computer bug was a literal moth"))
        assert package.research_status == "provisional"
        assert package.claims
        assert package.sources
        assert all(not source.is_non_discovery for source in package.sources)

    def test_research_notes_explain_status(self, research_agent, mock_topic_selection):
        package = research_agent.research(mock_topic_selection)
        joined = " ".join(package.research_notes).lower()
        assert "ready_for_fact_check" in joined
        assert "fact-check" in joined


class TestDates:
    def test_claim_discovery_date_not_conflated_with_source_publication(self, research_agent, mock_topic_selection):
        package = research_agent.research(mock_topic_selection)
        claim = claim_by_text(package, "While octopuses swim,")
        assert claim.dates[0].date_kind == "discovery_date"
        assert claim.dates[0].date == "2019-04-01"
        source = package.source_by_id(claim.source_ids[0])
        assert source.publication_date == "2019-04-12"
        assert source.publication_date != claim.dates[0].date

    def test_black_hole_announcement_date(self, research_agent, minimal_selection):
        package = research_agent.research(minimal_selection("t_black_hole", "Why can't we see inside a black hole?"))
        claim = claim_by_text(package, "The first picture of a black hole")
        assert claim.dates[0].date_kind == "announcement_date"
        assert claim.dates[0].date == "2019-04-10"

    def test_event_date_on_first_bug(self, research_agent, minimal_selection):
        package = research_agent.research(minimal_selection("t_first_bug", "The first computer bug was a literal moth"))
        claim = claim_by_text(package, "A moth was taped")
        assert claim.dates[0].date_kind == "event_date"


class TestSerializationAndProvenance:
    def test_json_round_trip(self, research_agent, minimal_selection):
        package = research_agent.research(minimal_selection("t_noise_cancellation", "How does noise cancellation actually work?"))
        restored = ResearchPackage.from_json(package.to_json())
        assert restored.to_dict() == package.to_dict()

    def test_provenance_metadata(self, research_agent, mock_topic_selection):
        package = research_agent.research(mock_topic_selection)
        assert package.stage == "research"
        assert package.run_id == "test-run-1"
        assert package.schema == "ResearchPackage"
        assert package.produced_by == {"prompt_version": None, "provider": None, "model": None}
        assert package.provenance.generated_by == "research_agent_v1"
        assert package.provenance.source == "MockResearchProvider"
        assert package.created_at

    def test_deterministic_output(self, mock_research_provider, mock_topic_selection):
        agent_a = ResearchAgent(mock_research_provider, run_id="det", created_at="2026-09-20T08:00:00+00:00")
        agent_b = ResearchAgent(mock_research_provider, run_id="det", created_at="2026-09-20T08:00:00+00:00")
        assert agent_a.research(mock_topic_selection).to_dict() == agent_b.research(mock_topic_selection).to_dict()


class TestCleanRejection:
    def test_malformed_source_rejected_cleanly(self, research_agent, minimal_selection):
        agent = ResearchAgent(_MalformedSourceProvider(), run_id="bad")
        with pytest.raises(ResearchPackageError, match="malformed source at index 0"):
            agent.research(minimal_selection("t_x", "Topic X"))

    def test_malformed_claim_rejected_cleanly(self, research_agent, minimal_selection):
        agent = ResearchAgent(_MalformedClaimProvider(), run_id="bad")
        with pytest.raises(ResearchPackageError, match="malformed claim at candidate 0, claim 0"):
            agent.research(minimal_selection("t_x", "Topic X"))

    def test_provider_swappable_without_agent_changes(self, minimal_selection):
        source = SourceCandidate(
            title="Custom stub",
            url="https://example.test/source/stub001",
            publisher="Stub Publisher",
            source_type="news_article",
            publication_date="2024-01-01",
            credibility_tier="high_quality_secondary",
            claims=(
                ClaimItem(claim_text="Stub providers are straightforward to plug in."),
            ),
        )
        agent = ResearchAgent(_StubProvider(source), run_id="stub-run")
        package = agent.research(minimal_selection("t_stub", "Stub topic"))
        assert package.research_status == "ready_for_fact_check"
        assert package.provenance.source == "_StubProvider"
        assert package.claims[0].claim_type == "fact"

    def test_missing_source_reference_in_serialized_package_rejected(self, research_agent, mock_topic_selection):
        package = research_agent.research(mock_topic_selection)
        raw = package.to_dict()
        raw["data"]["claims"][0]["source_ids"] = ["src_ghost001"]
        with pytest.raises(ResearchPackageError, match="unknown source"):
            ResearchPackage.from_dict(raw)


class _MalformedSourceProvider:
    """Search returning a source candidate missing its URL."""

    def search(self, topic: TopicDescriptor) -> list[SourceCandidate]:
        return [SourceCandidate(title="T", url="", publisher="P", source_type="paper")]


class _MalformedClaimProvider:
    """Search returning a candidate whose claim has an unknown date kind."""

    def search(self, topic: TopicDescriptor) -> list[SourceCandidate]:
        return [
            SourceCandidate(
                title="T",
                url="https://example.test/source/bad001",
                publisher="P",
                source_type="paper",
                claims=(ClaimItem(claim_text="Some claim.", dates=[{"date_kind": "publish_on", "date": "2020-01-01"}]),),
            )
        ]


class _StubProvider:
    """A custom provider driving the same ResearchAgent (replaceability)."""

    def __init__(self, *candidates: SourceCandidate) -> None:
        self._candidates = list(candidates)

    def search(self, topic: TopicDescriptor) -> list[SourceCandidate]:
        return self._candidates