"""Factory-wiring tests for the `hybrid` retriever.

Construction only, never retrieval: these assert that `--retriever hybrid`
builds the right retriever through the existing Stage-2 and Stage-3 seams, and
that building it imports no SDK.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from agents.qa.fact_checker.provider import EvidenceProviderError, build_fact_check_components
from agents.research.provider import ResearchProviderError, build_research_provider
from providers.firecrawl.client import FirecrawlClient, FirecrawlResult
from providers.firecrawl.config import FirecrawlConfig
from providers.hybrid.retriever import HybridResearchRetriever
from providers.searxng.client import client_from_env

GEMINI_ENV = {"GEMINI_API_KEY": "test-key"}


class _StubFallback:
    """Stand-in for the expensive tier; construction only, never called."""

    def scrape(self, url: str, *, only_main_content: bool = True) -> FirecrawlResult:
        raise AssertionError("scrape must not be called in these tests")


class TestResearchFactory:
    def test_hybrid_retriever_is_wired(self):
        from providers.hybrid.retriever import HybridResearchRetriever

        provider = build_research_provider("gemini", env=GEMINI_ENV, retriever="hybrid")
        assert provider.retriever_name == "hybrid"
        assert isinstance(provider._retriever, HybridResearchRetriever)

    def test_firecrawl_retriever_still_works(self):
        provider = build_research_provider("gemini", env=GEMINI_ENV, retriever="firecrawl")
        assert provider.retriever_name == "firecrawl"

    def test_hybrid_construction_imports_no_sdk(self):
        """Building the hybrid provider must not import the Gemini or Firecrawl SDKs."""
        code = (
            "import sys\n"
            "from agents.research.provider import build_research_provider\n"
            "build_research_provider('gemini', env={'GEMINI_API_KEY': 'test-key'}, retriever='hybrid')\n"
            "assert 'google.genai' not in sys.modules, 'google.genai imported transitively'\n"
            "assert 'firecrawl' not in sys.modules, 'firecrawl imported transitively'\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)

    def test_hybrid_construction_makes_no_network_call(self):
        """Construction must not touch SearXNG or Firecrawl."""
        code = (
            "import socket\n"
            "def deny(*a, **k):\n"
            "    raise AssertionError('construction attempted a network connection')\n"
            "socket.socket.connect = deny\n"
            "from agents.research.provider import build_research_provider\n"
            "build_research_provider('gemini', env={'GEMINI_API_KEY': 'test-key'}, retriever='hybrid')\n"
            "print('ok')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
        assert "ok" in result.stdout

    def test_unknown_retriever_still_raises(self):
        with pytest.raises(ResearchProviderError, match="unknown retriever"):
            build_research_provider("gemini", env=GEMINI_ENV, retriever="nonsense")

    def test_mock_provider_rejects_a_retriever(self):
        with pytest.raises(ResearchProviderError, match="--retriever is only supported"):
            build_research_provider("mock", retriever="hybrid", env={})


class TestFactCheckFactory:
    def test_hybrid_retriever_is_wired(self):
        from providers.hybrid.retriever import HybridResearchRetriever

        components = build_fact_check_components("gemini", retriever="hybrid", env=GEMINI_ENV)
        assert components.provider.retriever_name == "hybrid"
        assert isinstance(components.provider._retriever, HybridResearchRetriever)

    def test_firecrawl_retriever_still_works(self):
        components = build_fact_check_components("gemini", retriever="firecrawl", env=GEMINI_ENV)
        assert components.provider.retriever_name == "firecrawl"

    def test_unknown_retriever_still_raises(self):
        with pytest.raises(EvidenceProviderError, match="unknown retriever"):
            build_fact_check_components("gemini", retriever="nonsense", env=GEMINI_ENV)

    def test_mock_provider_rejects_a_retriever(self):
        with pytest.raises(EvidenceProviderError, match="--retriever is only supported"):
            build_fact_check_components("mock", retriever="hybrid", env={})


class TestDefaultHybridConfiguration:
    """The default hybrid retriever must honour retrieval env vars and never
    reach the hosted Firecrawl endpoint when no local URL is configured."""

    def test_default_searcher_reads_searxng_env(self):
        env = {
            "SEARXNG_URL": "http://127.0.0.1:9999",
            "SEARXNG_CATEGORIES": "general",
            "SEARXNG_MAX_RESULTS": "7",
        }
        client = client_from_env(env)
        assert client._base_url == "http://127.0.0.1:9999"
        assert client._categories == "general"
        assert client._max_results == 7

    def test_constructed_retriever_uses_env_configured_searcher(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://127.0.0.1:9999")
        monkeypatch.setenv("SEARXNG_CATEGORIES", "science")
        retriever = HybridResearchRetriever()
        assert retriever._searcher._base_url == "http://127.0.0.1:9999"
        assert retriever._searcher._categories == "science"

    def test_no_api_url_means_no_cloud_fallback(self):
        """An unset FIRECRAWL_API_URL must not construct a Firecrawl client."""
        retriever = HybridResearchRetriever(firecrawl_config=FirecrawlConfig())
        assert retriever._resolve_fallback() is None

    def test_configured_api_url_resolves_fallback(self):
        retriever = HybridResearchRetriever(
            firecrawl_config=FirecrawlConfig(api_url="http://127.0.0.1:3002")
        )
        assert isinstance(retriever._resolve_fallback(), FirecrawlClient)

    def test_injected_fallback_still_wins(self):
        sentinel = _StubFallback()
        retriever = HybridResearchRetriever(
            fallback=sentinel,
            firecrawl_config=FirecrawlConfig(),
        )
        assert retriever._resolve_fallback() is sentinel


class TestStage3GateUntouched:
    """The retrieval change must not have altered corroboration policy."""

    def test_corroboration_min_sources_is_still_two(self):
        from agents.qa.fact_checker.gates import FactCheckConfig

        assert FactCheckConfig().corroboration_min_sources == 2
