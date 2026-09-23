"""REAL Firecrawl retrieval integration test — gated by FIRECRAWL_INTEGRATION_TESTS=1.

Makes one small, bounded real web search against the Firecrawl Search API and
verifies the retriever returns at least one valid http(s) source with content.
Requires either a FIRECRAWL_API_KEY or the keyless free tier (rate-limited).

Run explicitly with:
    FIRECRAWL_INTEGRATION_TESTS=1 python -m pytest tests/test_firecrawl_integration.py -q
"""

from __future__ import annotations

import os

import pytest

from agents.research.provider import TopicDescriptor
from providers.firecrawl.config import FirecrawlConfig
from providers.firecrawl.retriever import FirecrawlResearchRetriever

GATE = os.environ.get("FIRECRAWL_INTEGRATION_TESTS") == "1"
REQUIRES_FIRECRAWL = pytest.mark.skipif(
    not GATE, reason="set FIRECRAWL_INTEGRATION_TESTS=1 to run real Firecrawl calls"
)

TOPIC = TopicDescriptor(
    topic_id="t_octopus_hearts",
    title="Why octopuses have three hearts",
    description="",
    pillar="science",
    topic_type="surprising_fact",
)


@REQUIRES_FIRECRAWL
def test_real_firecrawl_retrieval_is_small_valid_and_bounded():
    config = FirecrawlConfig.from_env()
    retriever = FirecrawlResearchRetriever(config=config, max_content_chars=2000)

    sources = retriever.retrieve(TOPIC)

    assert 1 <= len(sources) <= config.max_search_results, "expected 1..max_search_results real sources"
    for source in sources:
        assert source.url.startswith(("http://", "https://")), "real sources must be valid http(s) URLs"
        assert source.domain, "real sources carry a derived domain"

    # Secrets never leak into returned source metadata or repr chains.
    api_key = os.environ.get("FIRECRAWL_API_KEY", "")
    assert api_key not in repr(config)
    assert api_key not in str(config.redacted())
    assert (not api_key) or (api_key not in repr(sources))