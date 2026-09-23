"""Offline tests for FirecrawlResearchRetriever.

Determinism and safety under test: bounded results/content, domain allow/deny
filtering, metadata extraction, and — critically — that every Firecrawl
failure is translated to `ResearchProviderError` so no raw SDK exception ever
escapes to the stage layer.
"""

from __future__ import annotations

import pytest

from agents.research.provider import ResearchProviderError, TopicDescriptor
from providers.firecrawl.config import FirecrawlConfig
from providers.firecrawl.errors import FirecrawlAPIError
from providers.firecrawl.retriever import FirecrawlResearchRetriever
from tests.firecrawl_test_utils import (
    FakeFirecrawlRawClient,
    FakeWebResult,
    fake_metadata,
    make_firecrawl_client,
    make_http_error,
    web_bucket,
)

PAGE = "https://example.test/papers/octopus"
TOPIC = TopicDescriptor(
    topic_id="t_octopus_hearts",
    title="Why octopuses have three hearts",
    description="Test description",
    pillar="science",
    topic_type="surprising_fact",
)


def _retriever(raw, *, allow_domains=(), deny_domains=(), max_content_chars=6000, **cfg):
    config = FirecrawlConfig(max_retries=1, **cfg)
    client = make_firecrawl_client(config, raw_client=raw)
    return FirecrawlResearchRetriever(
        client=client,
        config=config,
        allow_domains=allow_domains,
        deny_domains=deny_domains,
        max_content_chars=max_content_chars,
    )


class TestRetrieve:
    def test_builds_source_with_metadata(self):
        raw = FakeFirecrawlRawClient(
            outcome=web_bucket(
                FakeWebResult(PAGE, title="Grounded paper", markdown="Body text.", metadata=fake_metadata(PAGE))
            )
        )
        (source,) = _retriever(raw).retrieve(TOPIC)
        assert source.url == PAGE
        assert source.title == "Grounded paper"
        assert source.domain == "example.test"
        assert source.publisher == "example.test"
        assert source.source_type == "web_page"
        assert source.content == "Body text."

    def test_content_is_truncated_to_max_content_chars(self):
        raw = FakeFirecrawlRawClient(outcome=web_bucket(FakeWebResult(PAGE, markdown="x" * 10_000)))
        (source,) = _retriever(raw, max_content_chars=100).retrieve(TOPIC)
        assert len(source.content) < 200
        assert source.content.endswith("[truncated]")

    def test_empty_results_yield_no_sources(self):
        raw = FakeFirecrawlRawClient(outcome=web_bucket())
        assert _retriever(raw).retrieve(TOPIC) == []

    def test_query_uses_topic_title_and_bounded_limit(self):
        raw = FakeFirecrawlRawClient(outcome=web_bucket())
        retriever = _retriever(raw, max_search_results=4)
        retriever.retrieve(TOPIC)
        query, kwargs = raw.calls[0]
        assert query == TOPIC.title
        assert kwargs["limit"] == 4


class TestDomainFiltering:
    def test_deny_domains_drop_results(self):
        raw = FakeFirecrawlRawClient(outcome=web_bucket(FakeWebResult(PAGE)))
        assert _retriever(raw, deny_domains=("example.test",)).retrieve(TOPIC) == []

    def test_allow_domains_keep_only_matches(self):
        raw = FakeFirecrawlRawClient(
            outcome=web_bucket(
                FakeWebResult(PAGE),
                FakeWebResult("https://other.example/news"),
            )
        )
        sources = _retriever(raw, allow_domains=("example.test",)).retrieve(TOPIC)
        assert [s.url for s in sources] == [PAGE]

    def test_subdomain_matches_base_allow_domain(self):
        raw = FakeFirecrawlRawClient(
            outcome=web_bucket(FakeWebResult("https://a.example.test/page"), FakeWebResult(PAGE))
        )
        sources = _retriever(raw, allow_domains=("example.test",)).retrieve(TOPIC)
        assert [s.url for s in sources] == ["https://a.example.test/page", PAGE]

    def test_empty_allowed_is_unfiltered(self):
        raw = FakeFirecrawlRawClient(outcome=web_bucket(FakeWebResult(PAGE)))
        assert len(_retriever(raw).retrieve(TOPIC)) == 1


class TestErrorTranslation:
    def test_client_failure_maps_to_research_provider_error(self):
        raw = FakeFirecrawlRawClient(outcome=make_http_error(500))
        with pytest.raises(ResearchProviderError, match="Firecrawl retrieval failed"):
            _retriever(raw).retrieve(TOPIC)
        assert raw.attempt_count == 2  # max_retries=1 -> bounded 2 attempts

    def test_timeout_maps_to_research_provider_error(self):
        raw = FakeFirecrawlRawClient(outcome=TimeoutError("slow"))
        with pytest.raises(ResearchProviderError):
            _retriever(raw).retrieve(TOPIC)

    def test_no_raw_sdk_error_escapes(self):
        raw = FakeFirecrawlRawClient(outcome=FirecrawlAPIError("HTTP 500"))
        with pytest.raises(ResearchProviderError):
            _retriever(raw).retrieve(TOPIC)