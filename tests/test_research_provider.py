"""Behavioral tests for MockResearchProvider (deterministic, offline)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.research.provider import MockResearchProvider, ResearchProviderError, TopicDescriptor

RESEARCH_EVIDENCE = Path(__file__).resolve().parent.parent / "data" / "mock" / "research_evidence.json"


def descriptor(topic_id: str, title: str = "") -> TopicDescriptor:
    return TopicDescriptor(topic_id=topic_id, title=title)


class TestMockResearchProvider:
    def test_knowable_topics_are_fixture_fixed(self, mock_research_provider):
        assert set(mock_research_provider.knowable_topics()) == {
            "t_octopus_hearts",
            "t_black_hole",
            "t_ai_agents_computers",
            "t_noise_cancellation",
            "t_first_bug",
        }

    def test_octopus_search_returns_sources_across_tiers(self, mock_research_provider):
        results = mock_research_provider.search(descriptor("t_octopus_hearts", "Why octopuses have three hearts"))
        assert len(results) == 3
        tiers = {candidate.credibility_tier for candidate in results}
        assert {"primary", "high_quality_secondary", "discovery"} <= tiers
        assert all(candidate.url.startswith("https://example.test/") for candidate in results)

    def test_deterministic_across_calls(self, mock_research_provider):
        topic = descriptor("t_black_hole", "Why can't we see inside a black hole?")
        first = [candidate for candidate in mock_research_provider.search(topic)]
        second = [candidate for candidate in mock_research_provider.search(topic)]
        assert first == second
        assert [c.url for c in first] == [c.url for c in second]
        assert [item.claim_text for c in first for item in c.claims] == [
            item.claim_text for c in second for item in c.claims
        ]

    def test_finds_by_title_when_id_unknown(self, mock_research_provider):
        results = mock_research_provider.search(
            descriptor("t_unknown_id", "Why does noise cancellation actually work?")
        )
        assert results and results[0].url == "https://example.test/source/nc001"

    def test_unknown_topic_returns_nothing(self, mock_research_provider):
        assert mock_research_provider.search(descriptor("t_europa_salt", "Salt on Europa")) == []

    def test_fixture_file_missing_raises_clean_error(self, tmp_path):
        with pytest.raises(ResearchProviderError, match="not found"):
            MockResearchProvider(data_path=tmp_path / "missing.json")

    def test_injected_data_works(self):
        provider = MockResearchProvider(
            data={
                "topics": [
                    {
                        "topic_id": "t_x",
                        "title": "Topic X",
                        "sources": [
                            {
                                "title": "S",
                                "url": "https://example.test/source/x001",
                                "publisher": "P",
                                "source_type": "news_article",
                                "publication_date": "2024-01-01",
                                "credibility_tier": "primary",
                                "claims": [],
                            }
                        ],
                    }
                ]
            }
        )
        results = provider.search(descriptor("t_x", "Topic X"))
        assert len(results) == 1
        assert results[0].title == "S"