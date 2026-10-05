"""REAL local-stack smoke tests for the retrieval experiment.

Gated by `RETRIEVAL_NETWORK_TESTS=1` so a normal `pytest tests` run never touches
the network or requires Docker.

    RETRIEVAL_NETWORK_TESTS=1 python -m pytest tests/test_retrieval_network_smoke.py -q

These are the automated form of the manual verification done in Phases 2-6. They
assert only *structural* properties of real responses -- that a real local
SearXNG returns http(s) URLs, that the adapter maps them onto the production
`RetrievedSource` contract, that a real page fetch extracts prose. They assert
NOTHING about specific result counts, ordering, or text, because upstream
engines rotate and intermittently CAPTCHA: pinning those would make this test
fail for reasons unrelated to the code under test.

Requires, for the corresponding tests:
  - SearXNG   : `docker compose -f experiments/retrieval/searxng/docker-compose.yml up -d`
  - Firecrawl : `docker compose -f experiments/retrieval/firecrawl_selfhost/docker-compose.yml up -d`

If a stack is down the test SKIPS with the reason. A local service being
unavailable is an environment fact, not a code defect, and must never be
reported as a passing "0 results".
"""

from __future__ import annotations

import os

import pytest

from experiments.retrieval.adapter import results_to_sources
from experiments.retrieval.comparison import run_comparison
from experiments.retrieval.firecrawl_selfhost import SelfHostedFirecrawlClient, production_adapter_probe
from experiments.retrieval.html_fetcher import HtmlFetcher
from experiments.retrieval.searxng_client import DEFAULT_BASE_URL, SearxngClient, host_for

GATE = os.environ.get("RETRIEVAL_NETWORK_TESTS") == "1"
NETWORK = pytest.mark.skipif(
    not GATE, reason="set RETRIEVAL_NETWORK_TESTS=1 to run real local-stack calls"
)

SEARXNG_URL = os.environ.get("RETRIEVAL_SEARXNG_URL", DEFAULT_BASE_URL)
FIRECRAWL_URL = os.environ.get("RETRIEVAL_FIRECRAWL_URL", "http://127.0.0.1:3002")

# A stable public page that carries real prose. Deliberately NOT example.com:
# that page really does contain only ~167 characters of text, and the fetcher
# correctly refuses it as `thin_content`. Asserting it would succeed would
# contradict the anti-fabrication rule this experiment is built on.
SIMPLE_PAGE = "https://science.nasa.gov/earth/"

# Answers 403 to an honest User-Agent. The documented finding, not a defect.
BLOCKED_PAGE = "https://en.wikipedia.org/wiki/Triple_point"

# Answers a Cloudflare "Just a moment..." interstitial with HTTP 403.
ANTI_BOT_PAGE = "https://www.kroneckerwallis.com/"

# A real HTTP 404. The self-hosted stack still answers HTTP 200 for it.
MISSING_PAGE = "https://www.nasa.gov/real-mars/"


@pytest.fixture(scope="module")
def searxng() -> SearxngClient:
    client = SearxngClient(base_url=SEARXNG_URL)
    if not client.probe():
        pytest.skip(f"local SearXNG is not reachable at {SEARXNG_URL}")
    return client


@pytest.fixture(scope="module")
def firecrawl() -> SelfHostedFirecrawlClient:
    client = SelfHostedFirecrawlClient(base_url=FIRECRAWL_URL)
    if not client.is_ready():
        pytest.skip(f"self-hosted Firecrawl is not reachable at {FIRECRAWL_URL}")
    return client


class TestRealSearxng:
    @NETWORK
    def test_health_endpoint_reports_ready(self, searxng: SearxngClient):
        assert searxng.probe() is True

    @NETWORK
    def test_real_search_returns_real_http_results(self, searxng: SearxngClient):
        results, _dead = searxng.search_with_diagnostics("what is the triple point of water", max_results=5)
        assert results, "a healthy local SearXNG must return at least one result"
        for result in results:
            assert result.normalized_url.startswith(("http://", "https://"))
            assert host_for(result.normalized_url), "every result must yield a real host"
            assert result.title or result.snippet, "a real result carries some text"

    @NETWORK
    def test_results_are_deduplicated(self, searxng: SearxngClient):
        results, _dead = searxng.search_with_diagnostics("what is the triple point of water", max_results=10)
        urls = [r.normalized_url for r in results]
        assert len(urls) == len(set(urls)), "normalized URLs must already be unique"

    @NETWORK
    def test_real_results_map_onto_the_production_contract(self, searxng: SearxngClient):
        """The integration question: real SearXNG output -> real RetrievedSource."""
        results, _dead = searxng.search_with_diagnostics("what is the triple point of water", max_results=5)
        assert results
        mapped = results_to_sources(results)
        assert len(mapped) == len(results)

        for item in mapped:
            source = item.source
            # Exactly the 8 contract fields; nothing extra invented.
            assert set(source.__dataclass_fields__) == {
                "url",
                "title",
                "publisher",
                "publication_date",
                "source_type",
                "content",
                "domain",
                "retrieved_at",
            }
            assert source.url.startswith(("http://", "https://"))
            assert source.domain == host_for(source.url)
            assert source.publisher == source.domain
            # Retrieved from a search snippet, not a fetched page: distinguishable.
            assert source.source_type == "search_result"
            # No tier is invented anywhere: that stays a model judgement.
            assert not hasattr(source, "tier")
            # A publication date is only present when SearXNG gave a real ISO one.
            date = source.publication_date
            assert date == "" or ("-" in date and len(date) >= 10)
            # Timestamp is caller-supplied and empty by default, exactly as
            # providers/firecrawl/retriever.py leaves it; the adapter never
            # invents "now".
            assert source.retrieved_at == ""

    @NETWORK
    def test_caller_supplied_timestamp_flows_through(self, searxng: SearxngClient):
        results, _dead = searxng.search_with_diagnostics("what is the triple point of water", max_results=2)
        assert results
        mapped = results_to_sources(results, retrieved_at="2026-10-01T00:00:00Z")
        assert all(item.source.retrieved_at == "2026-10-01T00:00:00Z" for item in mapped)

    @NETWORK
    def test_snippet_is_not_promoted_to_content_by_default(self, searxng: SearxngClient):
        """A search snippet must never masquerade as fetched page text."""
        results, _dead = searxng.search_with_diagnostics("what is the triple point of water", max_results=5)
        mapped = results_to_sources(results)
        assert all(item.source.content == "" for item in mapped)
        assert all(item.came_from_fetch is False for item in mapped)


class TestRealBeautifulSoupFetch:
    @NETWORK
    def test_simple_page_extracts_prose(self):
        page = HtmlFetcher().fetch(SIMPLE_PAGE)
        assert page.ok is True, f"expected a readable page, got {page.error_kind}: {page.error}"
        assert page.http_status == 200
        assert page.domain == "science.nasa.gov"
        assert page.content_length > 200
        assert page.title

    @NETWORK
    def test_genuinely_thin_page_is_refused_rather_than_returned_as_article(self):
        """example.com really is ~167 chars. Passing it off as usable text is the
        failure mode this experiment exists to prevent, so it must be refused."""
        page = HtmlFetcher().fetch("https://example.com/")
        assert page.ok is False
        assert page.error_kind == "thin_content"
        assert page.error, "the reason must be recorded, not swallowed"

    @NETWORK
    def test_blocked_page_is_reported_as_blocked_not_silently_empty(self):
        """Wikipedia 403s an honest UA. That must be a typed failure, never ''."""
        page = HtmlFetcher().fetch(BLOCKED_PAGE)
        assert page.ok is False
        assert page.error_kind in {"blocked", "http_error"}
        assert page.text == "" or page.ok is False


class TestRealSelfHostedFirecrawl:
    @NETWORK
    def test_readiness(self, firecrawl: SelfHostedFirecrawlClient):
        assert firecrawl.is_ready() is True

    @NETWORK
    def test_search_returns_real_urls(self, firecrawl: SelfHostedFirecrawlClient):
        items = firecrawl.search("what is the triple point of water", limit=3)
        assert items, "a healthy instance must return at least one item"
        assert any(item.ok for item in items), "at least one real hit must succeed"
        for item in items:
            assert item.url.startswith(("http://", "https://"))

    @NETWORK
    def test_scrape_produces_markdown(self, firecrawl: SelfHostedFirecrawlClient):
        outcome = firecrawl.scrape(SIMPLE_PAGE)
        assert outcome.ok is True, f"expected a scrape, got {outcome.error_kind}: {outcome.error}"
        assert outcome.content_length > 0

    @NETWORK
    def test_no_cloud_traffic_is_possible(self, firecrawl: SelfHostedFirecrawlClient):
        """A cloud key in the environment must not change what this client does."""
        before = os.environ.get("FIRECRAWL_API_KEY")
        os.environ["FIRECRAWL_API_KEY"] = "sk-should-never-be-used"
        try:
            outcome = firecrawl.scrape(SIMPLE_PAGE)
            assert outcome.ok is True
            assert "api.firecrawl.dev" not in str(outcome.url)
        finally:
            if before is None:
                os.environ.pop("FIRECRAWL_API_KEY", None)
            else:
                os.environ["FIRECRAWL_API_KEY"] = before

    @NETWORK
    def test_unmodified_production_adapter_works_against_local_stack(self):
        """The adoption question: is self-hosting a config change or a rewrite?"""
        result = production_adapter_probe(base_url=FIRECRAWL_URL, limit=2)
        assert result.get("checked") is True, result.get("reason")
        assert result.get("ok") is True, result.get("reason")

    @NETWORK
    def test_anti_bot_wall_is_rejected_by_name_not_returned_as_content(self, firecrawl):
        """The bug this guards.

        kroneckerwallis.com answers HTTP 403 with a Cloudflare interstitial.
        The self-hosted stack renders it and reports `success: true`, HTTP 200,
        metadata.statusCode 403 and 483 characters of "Just a moment..." markup.
        Before the fix that was accepted as a real source: a bot-challenge page
        laundered into evidence.
        """
        from providers.firecrawl.client import FirecrawlClient, FirecrawlConfig

        client = FirecrawlClient(config=FirecrawlConfig(api_url=FIRECRAWL_URL))
        with pytest.raises(Exception) as excinfo:
            client.scrape(ANTI_BOT_PAGE)
        message = str(excinfo.value)
        assert "403" in message or "challenge" in message, message

    @NETWORK
    def test_real_page_is_still_readable_after_the_guards(self, firecrawl):
        """The guards must not break the happy path: a genuinely readable page
        is still returned, in full, with real content."""
        from providers.firecrawl.client import FirecrawlClient, FirecrawlConfig

        client = FirecrawlClient(config=FirecrawlConfig(api_url=FIRECRAWL_URL))
        result = client.scrape(SIMPLE_PAGE)
        assert result.markdown.strip()
        assert len(result.markdown) > 200
        assert result.url == SIMPLE_PAGE

    @NETWORK
    def test_missing_page_is_not_returned_as_article(self):
        """nasa.gov/real-mars/ is a real 404. The self-hosted stack answers HTTP
        200 with metadata.statusCode 404 and ~25k characters of "Page Not Found"
        navigation. Accepting that would fabricate a source that says nothing
        about the requested topic."""
        from providers.firecrawl.client import FirecrawlClient, FirecrawlConfig

        client = FirecrawlClient(config=FirecrawlConfig(api_url=FIRECRAWL_URL))
        with pytest.raises(Exception) as excinfo:
            client.scrape(MISSING_PAGE)
        assert "404" in str(excinfo.value)


class TestRealComparison:
    @NETWORK
    def test_comparison_runs_end_to_end_and_reports_every_method(self):
        report = run_comparison(["what is the triple point of water"], max_results=2)
        assert set(report.overall) == {
            "searxng_search",
            "beautifulsoup_fetch",
            "firecrawl_selfhost_scrape",
            "firecrawl_selfhost_search",
        }
        # Whatever the outcome, the report must be self-describing.
        rendered = report.render()
        assert "NOTES" in rendered
        assert report.notes

    @NETWORK
    def test_comparison_json_is_serialisable(self):
        import json

        report = run_comparison(["what is the triple point of water"], max_results=2)
        assert json.loads(json.dumps(report.to_dict()))
