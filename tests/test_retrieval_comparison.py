"""Offline tests for the comparison experiment and the self-hosted client.

`run_comparison` accepts injected clients, so the whole reporting path is
verified without Docker or network. These tests check that the report reflects
observations and nothing else: no invented metrics, no fabricated successes, and
an honest "unavailable" for a backend that is down.
"""

from __future__ import annotations

import json

import httpx
import pytest

from experiments.retrieval.comparison import (
    BEAUTIFULSOUP,
    FIRECRAWL_SCRAPE,
    FIRECRAWL_SEARCH,
    METHODS,
    SEARXNG,
    run_comparison,
)
from experiments.retrieval.firecrawl_selfhost import (
    SelfHostedFirecrawlClient,
    SelfHostedFirecrawlError,
    production_adapter_probe,
)
from experiments.retrieval.html_fetcher import HtmlFetcher
from experiments.retrieval.searxng_client import SearxngClient

ARTICLE = (
    "<html lang='en'><head><title>Triple Point</title></head><body><article>"
    "<p>The triple point of water is the unique combination of temperature and "
    "pressure at which liquid water, water vapor, and ice coexist in equilibrium, "
    "which makes it a convenient calibration reference for thermodynamics.</p>"
    "<p>It was used to define the kelvin before the 2019 SI revision took effect "
    "and remains a standard example in introductory physics courses worldwide.</p>"
    "</article></body></html>"
)

SAMPLE_PAYLOAD = {
    "results": [
        {
            "url": "https://good.test/triple",
            "title": "Triple Point",
            "content": "snippet",
            "engine": "duckduckgo",
            "engines": ["duckduckgo"],
            "score": 1.0,
        },
        {
            "url": "https://blocked.test/triple",
            "title": "Blocked",
            "content": "snippet",
            "engine": "brave",
            "engines": ["brave"],
            "score": 0.5,
        },
    ]
}


def _searxng(payload=None, status=200, ready=True):
    payload = SAMPLE_PAYLOAD if payload is None else payload

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/healthz":
            return httpx.Response(200 if ready else 503, text="OK")
        return httpx.Response(status, json=payload)

    return SearxngClient(transport=httpx.MockTransport(handler))


def _html_fetcher():
    def handler(request: httpx.Request) -> httpx.Response:
        if "good.test" in str(request.url):
            return httpx.Response(200, text=ARTICLE, headers={"content-type": "text/html"})
        return httpx.Response(403, text="Forbidden", headers={"content-type": "text/html"})

    return HtmlFetcher(transport=httpx.MockTransport(handler))


def _firecrawl(ready=True, scrape_status=200, search_payload=None, search_status=200):
    search_payload = search_payload if search_payload is not None else {
        "success": True,
        "data": {
            "web": [
                {"url": "https://good.test/triple", "title": "Triple Point", "markdown": "# Triple\n\n" + "x" * 500},
            ]
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/v0/health/readiness"):
            return httpx.Response(200 if ready else 503, json={"status": "ok"})
        if request.url.path.endswith("/v2/scrape"):
            requested = json.loads(request.content).get("url", "")
            if "blocked.test" in requested or scrape_status >= 400:
                return httpx.Response(scrape_status or 500, json={"success": False})
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {
                        # The real instance echoes the URL it actually rendered.
                        "sourceURL": requested,
                        "markdown": "# Triple\n\n" + "x" * 900,
                        "metadata": {"title": "Triple Point"},
                    },
                },
            )
        if search_status >= 400:
            return httpx.Response(search_status, json={"success": False})
        return httpx.Response(200, json=search_payload)

    return SelfHostedFirecrawlClient(transport=httpx.MockTransport(handler))


class TestSelfHostedClient:
    def test_ready_probe_true_and_false(self):
        assert _firecrawl(ready=True).is_ready() is True
        assert _firecrawl(ready=False).is_ready() is False

    def test_ready_probe_never_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        assert SelfHostedFirecrawlClient(transport=httpx.MockTransport(handler)).is_ready() is False

    def test_scrape_success(self):
        outcome = _firecrawl().scrape("https://good.test/triple")
        assert outcome.ok is True
        assert outcome.title == "Triple Point"
        assert outcome.domain == "good.test"
        assert outcome.content_length == len(outcome.markdown)
        assert outcome.error == ""

    def test_scrape_http_error_is_a_failure_not_an_exception(self):
        outcome = _firecrawl(scrape_status=500).scrape("https://good.test/triple")
        assert outcome.ok is False
        assert outcome.error_kind == "http_error"
        assert outcome.http_status == 500
        assert outcome.markdown == ""

    def test_scrape_success_false_payload_is_a_failure(self):
        client = SelfHostedFirecrawlClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json={"success": False, "error": {"message": "nope"}})
            )
        )
        outcome = client.scrape("https://a.test/p")
        assert outcome.ok is False
        assert outcome.error_kind == "api_error"
        assert "nope" in outcome.error

    def test_scrape_non_json_is_a_failure(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>oops</html>", headers={"content-type": "text/html"})

        outcome = SelfHostedFirecrawlClient(transport=httpx.MockTransport(handler)).scrape("https://a.test/p")
        assert outcome.ok is False
        assert outcome.error_kind == "malformed"

    def test_scrape_connection_failure_is_a_failure(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        outcome = SelfHostedFirecrawlClient(transport=httpx.MockTransport(handler), timeout_seconds=1).scrape(
            "https://a.test/p"
        )
        assert outcome.ok is False
        assert outcome.error_kind == "connection"

    @pytest.mark.parametrize("url", ["", "not-a-url", "ftp://a.test/x"])
    def test_scrape_invalid_url(self, url):
        outcome = _firecrawl().scrape(url)
        assert outcome.ok is False
        assert outcome.error_kind == "invalid_url"

    def test_search_returns_one_outcome_per_hit(self):
        items = _firecrawl().search("q", limit=3)
        assert len(items) == 1
        assert items[0].ok is True
        assert items[0].url == "https://good.test/triple"

    def test_search_failure_surfaces_one_failed_outcome_not_an_empty_list(self):
        """"Instance down" must never look like "the search found nothing"."""
        items = _firecrawl(search_status=503).search("q")
        assert len(items) == 1
        assert items[0].ok is False
        assert items[0].error_kind == "http_error"

    def test_search_malformed_response_is_a_failed_outcome(self):
        client = SelfHostedFirecrawlClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, text="not json"))
        )
        items = client.search("q")
        assert len(items) == 1
        assert items[0].ok is False

    def test_empty_query_rejected(self):
        with pytest.raises(SelfHostedFirecrawlError):
            _firecrawl().search("   ")

    def test_no_api_key_is_ever_sent(self):
        """USE_DB_AUTHENTICATION=false; forwarding a cloud key would leak it."""
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json={"success": True, "data": {"web": []}})

        SelfHostedFirecrawlClient(transport=httpx.MockTransport(handler)).search("q")
        assert seen["auth"] is None


class TestComparisonReporting:
    def test_all_methods_reported(self):
        report = run_comparison(
            ["triple point"],
            max_results=2,
            searxng=_searxng(),
            fetcher=_html_fetcher(),
            firecrawl=_firecrawl(),
        )
        for name in METHODS:
            assert name in report.overall
        assert set(report.overall) == set(METHODS)

    def test_counts_are_observations_not_invented(self):
        report = run_comparison(
            ["triple point"],
            max_results=2,
            searxng=_searxng(),
            fetcher=_html_fetcher(),
            firecrawl=_firecrawl(),
        )
        searx = report.overall[SEARXNG]
        assert searx.search_results_returned == 2
        assert searx.unique_urls == 2

        bs = report.overall[BEAUTIFULSOUP]
        assert bs.fetch_attempted == 2
        assert bs.fetch_succeeded == 1, "only good.test is readable"
        assert bs.fetch_failed == 1
        assert bs.error_kinds == {"blocked": 1}
        assert bs.http_status_counts == {"200": 1, "403": 1}

    def test_both_extraction_methods_see_the_same_input_set(self):
        """Discovery is held fixed so differences are attributable to the method.

        Note: SearXNG discovery outcomes carry `ok=False` by design — they record
        that a URL was *found*, not that text was extracted — so the discovered set
        is the raw outcome list, not `succeeded_urls()`.
        """
        report = run_comparison(
            ["triple point"],
            max_results=2,
            searxng=_searxng(),
            fetcher=_html_fetcher(),
            firecrawl=_firecrawl(),
        )
        discovered = {o.url for o in report.overall[SEARXNG].outcomes}
        assert discovered, "SearXNG must have discovered something for this test to mean anything"
        for method in (BEAUTIFULSOUP, FIRECRAWL_SCRAPE):
            attempted = {o.url for o in report.overall[method].outcomes}
            assert attempted == discovered, f"{method} saw a different URL set"

    def test_overall_rollup_keeps_searxng_discovery_outcomes(self):
        """Regression: discovery rows used to be dropped when rolling up to `overall`.

        They were only attached to the per-query report, so `overall[SEARXNG].outcomes`
        came back empty and `unique_urls` was carried by a separate running counter.
        """
        report = run_comparison(
            ["q"], max_results=2, searxng=_searxng(), fetcher=_html_fetcher(), firecrawl=_firecrawl()
        )
        searx = report.overall[SEARXNG]
        assert {o.url for o in searx.outcomes} == {
            "https://good.test/triple",
            "https://blocked.test/triple",
        }
        assert searx.unique_urls == len(searx.outcomes), "unique_urls must be derived from the outcomes"

    def test_unique_urls_are_deduplicated_across_queries(self):
        """The same URL returned by two queries is one unique URL."""
        shared = {
            "results": [{"url": "https://good.test/triple", "title": "T", "engine": "x"}]
        }
        client = _searxng(payload=shared)
        report = run_comparison(
            ["query one", "query two"],
            max_results=2,
            searxng=client,
            fetcher=_html_fetcher(),
            firecrawl=_firecrawl(),
        )
        searx = report.overall[SEARXNG]
        assert searx.search_results_returned == 2, "two observations were made"
        assert len(searx.outcomes) == 2
        assert searx.unique_urls == 1, "but the URL is the same"

    def test_searxng_discovery_produces_no_phantom_error_tally(self):
        """Discovery rows carry no status/error, so they must not be tallied."""
        report = run_comparison(
            ["q"], max_results=2, searxng=_searxng(), fetcher=_html_fetcher(), firecrawl=_firecrawl()
        )
        searx = report.overall[SEARXNG]
        assert searx.error_kinds == {}
        assert searx.http_status_counts == {}
        assert searx.fetch_attempted == 0
        assert searx.fetch_failed == 0

    def test_search_discovery_outcomes_are_not_marked_extracted(self):
        """Discovery must not masquerade as a successful extraction."""
        report = run_comparison(
            ["q"], max_results=2, searxng=_searxng(), fetcher=_html_fetcher(), firecrawl=_firecrawl()
        )
        assert report.overall[SEARXNG].succeeded_urls() == ()
        assert report.overall[SEARXNG].unique_urls == 2

    def test_coverage_aggregates_canonical_urls(self):
        """Trailing-slash variants from different tiers must be one row."""
        payload = {
            "results": [
                {"url": "https://good.test/triple/", "title": "t", "engine": "x"},
            ]
        }
        report = run_comparison(
            ["q"],
            max_results=2,
            searxng=_searxng(payload=payload),
            fetcher=_html_fetcher(),
            firecrawl=_firecrawl(),
        )
        coverage = report.coverage()
        assert set(coverage) == {"https://good.test/triple"}

    def test_coverage_records_which_methods_succeeded(self):
        report = run_comparison(
            ["triple point"],
            max_results=2,
            searxng=_searxng(),
            fetcher=_html_fetcher(),
            firecrawl=_firecrawl(),
        )
        coverage = report.coverage()
        good = coverage["https://good.test/triple"]
        assert BEAUTIFULSOUP in good
        assert FIRECRAWL_SCRAPE in good
        assert FIRECRAWL_SEARCH in good

    def test_discovered_but_unreadable_urls_are_kept_as_none(self):
        """A URL nobody could read is still a real observation, so it is retained."""
        report = run_comparison(
            ["q"], max_results=2, searxng=_searxng(), fetcher=_html_fetcher(), firecrawl=_firecrawl()
        )
        coverage = report.coverage()
        assert coverage["https://blocked.test/triple"] == ()
        assert "NONE" in report.render(), "the rendered report must say so explicitly"

    def test_search_only_method_claims_no_fetches(self):
        report = run_comparison(
            ["q"], max_results=2, searxng=_searxng(), fetcher=_html_fetcher(), firecrawl=_firecrawl()
        )
        searx = report.overall[SEARXNG]
        assert searx.fetch_attempted == 0
        assert searx.fetch_succeeded == 0
        assert searx.total_content_chars == 0

    def test_firecrawl_marked_unavailable_when_down(self):
        report = run_comparison(
            ["q"], max_results=2, searxng=_searxng(), fetcher=_html_fetcher(), firecrawl=_firecrawl(ready=False)
        )
        assert report.overall[FIRECRAWL_SCRAPE].available is False
        assert report.overall[FIRECRAWL_SEARCH].available is False
        assert report.overall[FIRECRAWL_SCRAPE].unavailable_reason
        assert any("not reachable" in note for note in report.notes)

    def test_searxng_down_is_reported_not_silently_zero(self):
        report = run_comparison(
            ["q"], max_results=2, searxng=_searxng(status=500), fetcher=_html_fetcher(), firecrawl=_firecrawl(ready=False)
        )
        assert report.overall[SEARXNG].available is False
        assert report.overall[BEAUTIFULSOUP].fetch_attempted == 0
        assert any("SearXNG search failed" in note for note in report.notes)

    def test_notes_always_state_nondeterminism(self):
        report = run_comparison(
            ["q"], max_results=2, searxng=_searxng(), fetcher=_html_fetcher(), firecrawl=_firecrawl(ready=False)
        )
        assert any("NOT reproducible" in note for note in report.notes)

    def test_unresponsive_engines_surfaced_in_notes(self):
        payload = {
            "results": SAMPLE_PAYLOAD["results"],
            "unresponsive_engines": ["brave : Suspended: too many requests"],
        }
        report = run_comparison(
            ["q"], max_results=2, searxng=_searxng(payload=payload), fetcher=_html_fetcher(), firecrawl=_firecrawl(ready=False)
        )
        assert report.overall[SEARXNG].unresponsive_engines == ("brave : Suspended: too many requests",)
        assert any("unavailable during this run" in note for note in report.notes)

    def test_no_invented_quality_metrics(self):
        """The report exposes counts only; no score/rank/relevance field exists."""
        report = run_comparison(
            ["q"], max_results=2, searxng=_searxng(), fetcher=_html_fetcher(), firecrawl=_firecrawl()
        )
        payload = report.to_dict()
        banned = {"score", "relevance", "quality", "rank", "confidence", "rating"}
        for name, data in payload["overall"].items():
            assert not (banned & set(data)), f"{name} exposes an invented metric"
            for outcome in data["outcomes"]:
                assert not (banned & set(outcome))

    def test_to_dict_is_json_serialisable(self):
        import json

        report = run_comparison(
            ["q"], max_results=2, searxng=_searxng(), fetcher=_html_fetcher(), firecrawl=_firecrawl()
        )
        assert json.loads(json.dumps(report.to_dict()))

    def test_render_mentions_every_method(self):
        report = run_comparison(
            ["q"], max_results=2, searxng=_searxng(), fetcher=_html_fetcher(), firecrawl=_firecrawl()
        )
        rendered = report.render()
        for name in METHODS:
            assert name in rendered


class TestProductionAdapterProbe:
    def test_reports_failure_without_raising(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        # Probe against an unreachable local port; must not raise and must not
        # reach the cloud (no api key is ever supplied).
        result = production_adapter_probe(base_url="http://127.0.0.1:1", limit=1, timeout_seconds=2.0)
        assert result["checked"] is True
        assert result["ok"] is False
        assert result["reason"]