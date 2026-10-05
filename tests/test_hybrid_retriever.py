"""Integration tests for HybridResearchRetriever.

Every test here is network-free: the searcher, fetcher, and Firecrawl fallback
are injected fakes, so the whole tier/dedup/fallback policy is exercised without
touching SearXNG, Firecrawl, or the open web. Real-service behaviour is covered
separately by the gated `test_retrieval_network_smoke.py`.
"""

from __future__ import annotations

import pytest

from agents.research.provider import TopicDescriptor
from agents.research.retriever import RetrievedSource
from providers.firecrawl.client import FirecrawlResult
from providers.firecrawl.errors import FirecrawlAPIError
from providers.hybrid.retriever import (
    CHEAP_TIER_SOURCE_TYPE,
    FALLBACK_ERROR_KINDS,
    FALLBACK_TIER_SOURCE_TYPE,
    HybridResearchRetriever,
)
from providers.html.fetcher import FetchedPage
from providers.searxng.client import SearchResult
from providers.searxng.errors import SearxngConnectionError

BODY = "Real prose about the subject. " * 12  # comfortably over min_usable_chars


def _topic(**kwargs) -> TopicDescriptor:
    base = {"topic_id": "t_test", "title": "Why Is the Sky Blue?", "description": "Rayleigh scattering."}
    base.update(kwargs)
    return TopicDescriptor(**base)


class FakeSearcher:
    """Returns a per-query result table; records the queries it was asked."""

    def __init__(self, table: dict[str, list[str]] | None = None, *, default: list[str] | None = None):
        self.table = table or {}
        self.default = default if default is not None else []
        self.queries: list[str] = []
        self.calls: list[str] = []

    def search(self, query: str, *, max_results: int | None = None) -> tuple[SearchResult, ...]:
        self.queries.append(query)
        self.calls.append(query)
        urls = self.table.get(query, self.default)
        return tuple(SearchResult(url=url, title=f"title for {url}", snippet="snippet") for url in urls)


class ExplodingSearcher:
    def __init__(self, error: Exception) -> None:
        self._error = error
        self.queries: list[str] = []

    def search(self, query: str, *, max_results: int | None = None):
        self.queries.append(query)
        raise self._error


class FakeFetcher:
    """Maps URL -> FetchedPage; unlisted URLs succeed with generic prose."""

    def __init__(self, outcomes: dict[str, FetchedPage] | None = None) -> None:
        self.outcomes = outcomes or {}
        self.fetched: list[str] = []

    def fetch(self, url: str) -> FetchedPage:
        self.fetched.append(url)
        if url in self.outcomes:
            return self.outcomes[url]
        return FetchedPage(url=url, ok=True, title=f"page {url}", text=BODY, domain="example.test")


class FakeFallback:
    def __init__(self, outcomes: dict[str, FirecrawlResult] | None = None, *, error: Exception | None = None):
        self.outcomes = outcomes or {}
        self.error = error
        self.scraped: list[str] = []

    def scrape(self, url: str, *, only_main_content: bool = True) -> FirecrawlResult:
        self.scraped.append(url)
        if self.error is not None:
            raise self.error
        return self.outcomes.get(url, FirecrawlResult(url=url, title="rendered", markdown=BODY))


def _retriever(**kwargs) -> HybridResearchRetriever:
    kwargs.setdefault("searcher", FakeSearcher(default=["https://a.test/1", "https://b.test/2"]))
    kwargs.setdefault("fetcher", FakeFetcher())
    return HybridResearchRetriever(**kwargs)


# --------------------------------------------------------------------------- #
# Query planning + multi-query recall
# --------------------------------------------------------------------------- #


class TestQueryPlanning:
    def test_uses_the_factual_query_not_the_rhetorical_title(self):
        """The factual query must lead; the raw title survives only as a low-ranked recall variant."""
        searcher = FakeSearcher(default=["https://a.test/1"])
        _retriever(searcher=searcher, enable_fallback=False).retrieve(
            _topic(title="Do Black Holes Actually Shrink?", description="Explore Hawking radiation.")
        )
        assert searcher.queries[0] == "Hawking radiation"
        assert searcher.queries[-1] == "Do Black Holes Actually Shrink?"

    def test_searches_every_planned_query(self):
        searcher = FakeSearcher(default=["https://a.test/1"])
        _retriever(searcher=searcher, enable_fallback=False).retrieve(_topic())
        assert len(searcher.queries) >= 2
        assert "Sky Blue" in searcher.queries

    def test_no_derivable_query_returns_nothing(self):
        """An empty topic must yield an empty result, never an invented query."""
        searcher = FakeSearcher(default=["https://a.test/1"])
        assert _retriever(searcher=searcher).retrieve(_topic(title="", description="")) == []
        assert searcher.queries == []


class TestRecall:
    def test_merges_results_across_queries(self):
        searcher = FakeSearcher(
            table={
                "Rayleigh scattering": ["https://a.test/1"],
                "Sky Blue": ["https://b.test/2"],
            }
        )
        sources = _retriever(searcher=searcher, enable_fallback=False).retrieve(_topic())
        assert [source.domain for source in sources] == ["a.test", "b.test"]

    def test_deduplicates_the_same_page_found_twice(self):
        searcher = FakeSearcher(default=["https://a.test/1", "https://a.test/1/"])
        sources = _retriever(searcher=searcher, enable_fallback=False).retrieve(_topic())
        assert [source.url for source in sources] == ["https://a.test/1"]

    def test_deduplicates_case_and_trailing_slash_differences(self):
        searcher = FakeSearcher(default=["https://A.test/Page", "https://a.test/Page/"])
        retriever = _retriever(searcher=searcher, enable_fallback=False)
        assert len(retriever.retrieve(_topic())) == 1

    def test_drops_hits_that_are_not_absolute_http_urls(self):
        searcher = FakeSearcher(default=["javascript:void(0)", "https://a.test/1", "not a url"])
        sources = _retriever(searcher=searcher, enable_fallback=False).retrieve(_topic())
        assert [source.url for source in sources] == ["https://a.test/1"]

    def test_respects_max_sources(self):
        searcher = FakeSearcher(default=[f"https://a.test/{i}" for i in range(20)])
        retriever = _retriever(searcher=searcher, enable_fallback=False, max_sources=3)
        assert len(retriever.retrieve(_topic())) == 3

    def test_zero_max_sources_returns_empty(self):
        searcher = FakeSearcher(default=["https://a.test/1"])
        retriever = _retriever(searcher=searcher, enable_fallback=False, max_sources=0)
        assert retriever.retrieve(_topic()) == []


# --------------------------------------------------------------------------- #
# Domain filtering
# --------------------------------------------------------------------------- #


class TestDomainFiltering:
    def test_deny_list_excludes_a_domain_and_its_subdomains(self):
        searcher = FakeSearcher(default=["https://social.test/x", "https://a.test/1"])
        retriever = _retriever(searcher=searcher, enable_fallback=False, deny_domains=("social.test",))
        assert [source.domain for source in retriever.retrieve(_topic())] == ["a.test"]

    def test_deny_list_matches_subdomains(self):
        searcher = FakeSearcher(default=["https://news.social.test/x", "https://a.test/1"])
        retriever = _retriever(searcher=searcher, enable_fallback=False, deny_domains=("social.test",))
        assert [source.domain for source in retriever.retrieve(_topic())] == ["a.test"]

    def test_allow_list_restricts_to_permitted_domains(self):
        searcher = FakeSearcher(default=["https://allowed.test/1", "https://other.test/2"])
        retriever = _retriever(searcher=searcher, enable_fallback=False, allow_domains=("allowed.test",))
        assert [source.domain for source in retriever.retrieve(_topic())] == ["allowed.test"]

    def test_no_lists_permits_everything(self):
        searcher = FakeSearcher(default=["https://a.test/1", "https://b.test/2"])
        assert len(_retriever(searcher=searcher, enable_fallback=False).retrieve(_topic())) == 2


# --------------------------------------------------------------------------- #
# The cheap tier
# --------------------------------------------------------------------------- #


class TestCheapTier:
    def test_successful_fetch_reports_the_cheap_tier(self):
        sources = _retriever(enable_fallback=False).retrieve(_topic())
        assert sources[0].source_type == CHEAP_TIER_SOURCE_TYPE

    def test_content_is_carried_through(self):
        sources = _retriever(enable_fallback=False).retrieve(_topic())
        assert sources[0].content.startswith("Real prose")

    def test_publisher_and_domain_come_from_the_url_host(self):
        sources = _retriever(enable_fallback=False).retrieve(_topic())
        assert sources[0].publisher == "a.test"
        assert sources[0].domain == "a.test"

    def test_content_is_truncated_to_the_configured_bound(self):
        sources = _retriever(enable_fallback=False, max_content_chars=100).retrieve(_topic())
        assert len(sources[0].content) < len(BODY)
        assert sources[0].content.endswith("[truncated]")

    def test_page_title_wins_over_the_search_snippet_title(self):
        fetcher = FakeFetcher({"https://a.test/1": FetchedPage(url="https://a.test/1", ok=True, title="Page Title", text=BODY)})
        sources = _retriever(fetcher=fetcher, enable_fallback=False, max_sources=1).retrieve(_topic())
        assert sources[0].title == "Page Title"

    def test_search_title_is_used_when_the_page_has_none(self):
        fetcher = FakeFetcher({"https://a.test/1": FetchedPage(url="https://a.test/1", ok=True, title="", text=BODY)})
        sources = _retriever(fetcher=fetcher, enable_fallback=False, max_sources=1).retrieve(_topic())
        assert sources[0].title == "title for https://a.test/1"


# --------------------------------------------------------------------------- #
# The Firecrawl fallback
# --------------------------------------------------------------------------- #


def _blocked(url: str, kind: str = "blocked") -> FetchedPage:
    return FetchedPage(url=url, ok=False, error_kind=kind, error=f"{kind} page")


class TestFallback:
    @pytest.mark.parametrize("kind", sorted(FALLBACK_ERROR_KINDS - {"invalid_url"}))
    def test_escalates_every_difficult_page_kind(self, kind):
        fetcher = FakeFetcher({"https://a.test/1": _blocked("https://a.test/1", kind)})
        fallback = FakeFallback()
        searcher = FakeSearcher(default=["https://a.test/1"])
        retriever = _retriever(searcher=searcher, fetcher=fetcher, fallback=fallback, max_sources=1)
        sources = retriever.retrieve(_topic())
        assert fallback.scraped == ["https://a.test/1"]
        assert sources[0].source_type == FALLBACK_TIER_SOURCE_TYPE

    def test_a_cheap_success_is_never_escalated(self):
        fallback = FakeFallback()
        _retriever(fetcher=FakeFetcher(), fallback=fallback, enable_fallback=True).retrieve(_topic())
        assert fallback.scraped == []

    def test_does_not_escalate_invalid_urls(self):
        """Firecrawl cannot fix a malformed URL, so it must not be asked."""
        fallback = FakeFallback()
        fetcher = FakeFetcher({"https://a.test/1": _blocked("https://a.test/1", "invalid_url")})
        searcher = FakeSearcher(default=["https://a.test/1"])
        retriever = _retriever(searcher=searcher, fetcher=fetcher, fallback=fallback, max_sources=1)
        assert retriever.retrieve(_topic()) == []
        assert fallback.scraped == []

    def test_fallback_budget_is_bounded(self):
        urls = [f"https://a.test/{i}" for i in range(5)]
        fetcher = FakeFetcher({url: _blocked(url) for url in urls})
        fallback = FakeFallback()
        searcher = FakeSearcher(default=urls)
        retriever = _retriever(
            searcher=searcher,
            fetcher=fetcher,
            fallback=fallback,
            max_sources=5,
            max_firecrawl_fallbacks=2,
        )
        retriever.retrieve(_topic())
        assert len(fallback.scraped) == 2

    def test_zero_budget_never_escalates(self):
        fetcher = FakeFetcher({"https://a.test/1": _blocked("https://a.test/1")})
        fallback = FakeFallback()
        searcher = FakeSearcher(default=["https://a.test/1"])
        retriever = _retriever(
            searcher=searcher, fetcher=fetcher, fallback=fallback, max_sources=1, max_firecrawl_fallbacks=0
        )
        assert retriever.retrieve(_topic()) == []
        assert fallback.scraped == []

    def test_disable_flag_never_escalates(self):
        fetcher = FakeFetcher({"https://a.test/1": _blocked("https://a.test/1")})
        fallback = FakeFallback()
        searcher = FakeSearcher(default=["https://a.test/1"])
        retriever = _retriever(searcher=searcher, fetcher=fetcher, fallback=fallback, enable_fallback=False)
        assert retriever.retrieve(_topic()) == []
        assert fallback.scraped == []

    def test_a_failed_escalation_drops_the_page_but_keeps_the_rest(self):
        """One unreadable page must not abort retrieval of the others."""
        fetcher = FakeFetcher({"https://a.test/1": _blocked("https://a.test/1")})
        fallback = FakeFallback(error=FirecrawlAPIError("boom"))
        searcher = FakeSearcher(default=["https://a.test/1", "https://b.test/2"])
        sources = _retriever(searcher=searcher, fetcher=fetcher, fallback=fallback, max_sources=5).retrieve(_topic())
        assert [source.domain for source in sources] == ["b.test"]

    def test_empty_fallback_content_drops_the_page(self):
        fetcher = FakeFetcher({"https://a.test/1": _blocked("https://a.test/1")})
        fallback = FakeFallback({"https://a.test/1": FirecrawlResult(url="https://a.test/1", markdown="  ")})
        searcher = FakeSearcher(default=["https://a.test/1"])
        assert _retriever(searcher=searcher, fetcher=fetcher, fallback=fallback, max_sources=1).retrieve(_topic()) == []

    def test_fallback_content_is_truncated_to_the_bound(self):
        fetcher = FakeFetcher({"https://a.test/1": _blocked("https://a.test/1")})
        searcher = FakeSearcher(default=["https://a.test/1"])
        sources = _retriever(
            searcher=searcher, fetcher=fetcher, fallback=FakeFallback(), enable_fallback=True, max_sources=1, max_content_chars=100
        ).retrieve(_topic())
        assert len(sources[0].content) < len(BODY)

    def test_a_page_unreadable_by_both_tiers_is_dropped_not_faked(self):
        """The bug this retriever exists to prevent: empty-content sources."""
        fetcher = FakeFetcher({"https://a.test/1": _blocked("https://a.test/1")})
        fallback = FakeFallback(error=FirecrawlAPIError("boom"))
        searcher = FakeSearcher(default=["https://a.test/1"])
        retriever = _retriever(searcher=searcher, fetcher=fetcher, fallback=fallback, max_sources=1)
        sources = retriever.retrieve(_topic())
        assert sources == []
        assert all(source.content for source in sources)


# --------------------------------------------------------------------------- #
# Search failures
# --------------------------------------------------------------------------- #


class TestSearchFailure:
    def test_a_total_search_outage_yields_no_sources(self):
        searcher = ExplodingSearcher(SearxngConnectionError("down"))
        retriever = _retriever(searcher=searcher, enable_fallback=False)
        assert retriever.retrieve(_topic()) == []

    def test_one_failing_query_does_not_discard_the_others(self):
        class FlakySearcher(FakeSearcher):
            def search(self, query: str, *, max_results: int | None = None):
                if query == "Rayleigh scattering":
                    raise SearxngConnectionError("down")
                return super().search(query, max_results=max_results)

        searcher = FlakySearcher(table={"Sky Blue": ["https://a.test/1"]})
        sources = _retriever(searcher=searcher, enable_fallback=False).retrieve(_topic())
        assert [source.domain for source in sources] == ["a.test"]

    def test_a_search_outage_does_not_raise(self):
        """Retrieval degrades to zero sources; it does not fabricate them."""
        searcher = ExplodingSearcher(SearxngConnectionError("down"))
        assert _retriever(searcher=searcher).retrieve(_topic()) == []


# --------------------------------------------------------------------------- #
# Traceability
# --------------------------------------------------------------------------- #


class TestTrace:
    def test_trace_records_the_query_plan(self):
        retriever = _retriever(enable_fallback=False)
        retriever.retrieve(_topic())
        assert "description_factual_clause" in retriever.last_trace.as_lines()[0]

    def test_trace_records_hits_per_query(self):
        retriever = _retriever(searcher=FakeSearcher(default=["https://a.test/1"]), enable_fallback=False)
        retriever.retrieve(_topic())
        assert retriever.last_trace.per_query_hits["Rayleigh scattering"] == 1

    def test_trace_counts_cheap_failures_by_kind(self):
        fetcher = FakeFetcher({"https://a.test/1": _blocked("https://a.test/1", "blocked")})
        searcher = FakeSearcher(default=["https://a.test/1"])
        retriever = _retriever(searcher=searcher, fetcher=fetcher, enable_fallback=False, max_sources=1)
        retriever.retrieve(_topic())
        assert retriever.last_trace.cheap_failed == {"https://a.test/1": "blocked"}

    def test_trace_records_a_successful_escalation(self):
        fetcher = FakeFetcher({"https://a.test/1": _blocked("https://a.test/1")})
        searcher = FakeSearcher(default=["https://a.test/1"])
        retriever = _retriever(
            searcher=searcher, fetcher=fetcher, fallback=FakeFallback(), enable_fallback=True, max_sources=1
        )
        retriever.retrieve(_topic())
        assert retriever.last_trace.fallback_ok == ["https://a.test/1"]

    def test_trace_records_a_skipped_escalation_with_a_reason(self):
        fetcher = FakeFetcher({"https://a.test/1": _blocked("https://a.test/1", "invalid_url")})
        searcher = FakeSearcher(default=["https://a.test/1"])
        retriever = _retriever(searcher=searcher, fetcher=fetcher, fallback=FakeFallback(), max_sources=1)
        retriever.retrieve(_topic())
        assert "invalid_url" in retriever.last_trace.fallback_skipped["https://a.test/1"]

    def test_trace_counts_the_emitted_sources(self):
        retriever = _retriever(enable_fallback=False, max_sources=1)
        retriever.retrieve(_topic())
        assert retriever.last_trace.emitted == 1


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #


class TestContract:
    def test_satisfies_the_research_retriever_protocol(self):
        retriever = _retriever(enable_fallback=False)
        assert retriever.retriever_name == "hybrid"
        assert callable(retriever.retrieve)

    def test_every_emitted_source_carries_real_content(self):
        sources = _retriever(enable_fallback=False).retrieve(_topic())
        assert sources
        for source in sources:
            assert isinstance(source, RetrievedSource)
            assert source.url.startswith("http")
            assert source.content.strip()

    def test_retrieval_never_invents_a_url(self):
        """Every returned URL must be one the searcher actually supplied."""
        supplied = {"https://a.test/1", "https://b.test/2"}
        searcher = FakeSearcher(default=sorted(supplied))
        for source in _retriever(searcher=searcher, enable_fallback=False).retrieve(_topic()):
            assert source.url in supplied

    def test_retrieved_at_is_left_empty_not_invented(self):
        sources = _retriever(enable_fallback=False).retrieve(_topic())
        assert all(source.retrieved_at == "" for source in sources)
