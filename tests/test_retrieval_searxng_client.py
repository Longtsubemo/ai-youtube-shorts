"""Offline tests for the SearXNG client: normalization, dedup, failures.

No Docker, no network, no running SearXNG. HTTP behaviour is exercised through
an injected `httpx.MockTransport`, so the transport/parse/retry paths are all
covered deterministically.

Real-network behaviour lives in tests/test_retrieval_network_smoke.py, which is
gated behind an environment variable.
"""

from __future__ import annotations

import httpx
import pytest

from agents.research.models import normalize_source_url, source_id_for
from experiments.retrieval.errors import (
    SearxngAPIError,
    SearxngConfigurationError,
    SearxngConnectionError,
    SearxngFormatDisabledError,
    SearxngMalformedResponseError,
    SearxngTimeoutError,
)
from experiments.retrieval.searxng_client import (
    DEFAULT_CATEGORIES,
    SearxngClient,
    client_from_env,
    dedupe_results,
    host_for,
    normalize_result,
    parse_results,
    unresponsive_engines,
)


def _item(**overrides):
    base = {
        "url": "https://en.wikipedia.org/wiki/Triple_point",
        "title": "Triple point - Wikipedia",
        "content": "A snippet about the triple point.",
        "engine": "duckduckgo",
        "engines": ["duckduckgo"],
        "score": 1.0,
        "category": "general",
        "positions": [1],
        "publishedDate": None,
        "pubdate": "",
        "author": "",
    }
    base.update(overrides)
    return base


def _client(handler, **kwargs):
    return SearxngClient(transport=httpx.MockTransport(handler), **kwargs)


def _json_handler(payload, status=200, content_type="application/json"):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload, headers={"content-type": content_type})

    return handler


# --------------------------------------------------------------------------- #
# normalize_result
# --------------------------------------------------------------------------- #


class TestNormalizeResult:
    def test_maps_the_real_searxng_field_names(self):
        result = normalize_result(_item(publishedDate=None, pubdate="Jan 2024"))
        assert result is not None
        assert result.url == "https://en.wikipedia.org/wiki/Triple_point"
        assert result.normalized_url == "https://en.wikipedia.org/wiki/Triple_point"
        assert result.title == "Triple point - Wikipedia"
        assert result.snippet == "A snippet about the triple point."
        assert result.engine == "duckduckgo"
        assert result.engines == ("duckduckgo",)
        assert result.score == 1.0
        assert result.positions == (1,)
        # "Jan 2024" is preserved verbatim and NOT parsed as a date here.
        assert result.published_date_raw == "Jan 2024"

    def test_merges_single_engine_into_engines_without_duplicates(self):
        result = normalize_result(_item(engines=["google cse"], engine="google cse"))
        assert result is not None
        assert result.engines == ("google cse",)

    def test_adds_engine_when_engines_list_omits_it(self):
        result = normalize_result(_item(engines=[], engine="brave"))
        assert result is not None
        assert result.engines == ("brave",)

    def test_missing_score_is_none_not_zero(self):
        """A missing score must never become 0.0, which would invent a ranking."""
        item = _item()
        del item["score"]
        result = normalize_result(item)
        assert result is not None
        assert result.score is None

    def test_absent_metadata_stays_empty(self):
        item = _item()
        for key in ("title", "content", "category", "author", "publishedDate", "pubdate", "positions", "score"):
            del item[key]
        result = normalize_result(item)
        assert result is not None
        assert result.title == ""
        assert result.snippet == ""
        assert result.category == ""
        assert result.author == ""
        assert result.published_date_raw == ""
        assert result.positions == ()
        assert result.score is None

    @pytest.mark.parametrize(
        "item",
        [
            {},
            {"url": ""},
            {"url": "   "},
            {"url": "not-a-url"},
            {"url": "ftp://example.test/x"},
            {"url": "javascript:alert(1)"},
            {"url": "example.test/no-scheme"},
        ],
    )
    def test_unusable_urls_are_dropped_not_repaired(self, item):
        assert normalize_result(item) is None

    @pytest.mark.parametrize("item", [None, [], "text", 7, object()])
    def test_non_mapping_items_are_dropped(self, item):
        assert normalize_result(item) is None

    def test_structured_values_are_refused_not_stringified(self):
        """A list in a text field must not become "['a', 'b']"."""
        result = normalize_result(_item(title={"nested": "dict"}))
        assert result is not None
        assert result.title == ""

    def test_numeric_score_string_is_parsed(self):
        result = normalize_result(_item(score="0.75"))
        assert result is not None
        assert result.score == 0.75

    def test_unparseable_score_is_none(self):
        result = normalize_result(_item(score="high"))
        assert result is not None
        assert result.score is None

    def test_bool_is_not_treated_as_number(self):
        result = normalize_result(_item(score=True))
        assert result is not None
        assert result.score is None


# --------------------------------------------------------------------------- #
# parse_results + dedup
# --------------------------------------------------------------------------- #


class TestParseResults:
    def test_empty_results_list(self):
        assert parse_results({"query": "x", "results": []}) == ()

    def test_absent_results_key_is_zero_results(self):
        assert parse_results({"query": "x"}) == ()

    @pytest.mark.parametrize("payload", [None, [], "text", 3, object()])
    def test_non_object_payload_is_malformed(self, payload):
        with pytest.raises(SearxngMalformedResponseError):
            parse_results(payload)

    def test_results_of_wrong_type_is_malformed(self):
        with pytest.raises(SearxngMalformedResponseError):
            parse_results({"results": {"url": "https://a.test"}})
        with pytest.raises(SearxngMalformedResponseError):
            parse_results({"results": "https://a.test"})

    def test_malformed_entries_are_skipped_best_effort(self):
        payload = {"results": [_item(), {"url": ""}, "garbage", None, _item(url="https://b.test/x")]}
        results = parse_results(payload)
        assert [r.normalized_url for r in results] == [
            "https://en.wikipedia.org/wiki/Triple_point",
            "https://b.test/x",
        ]

    def test_trailing_slash_variants_deduplicate(self):
        payload = {
            "results": [
                _item(url="https://a.test/page/"),
                _item(url="https://a.test/page"),
            ]
        }
        results = parse_results(payload)
        assert len(results) == 1
        # First occurrence wins, matching the repository's normalize_source_url.
        assert results[0].url == "https://a.test/page/"

    def test_tracking_parameters_deduplicate(self):
        """normalize_source_url drops the query string, so tracking variants collapse."""
        payload = {
            "results": [
                _item(url="https://a.test/page?utm_source=x"),
                _item(url="https://a.test/page"),
            ]
        }
        assert len(parse_results(payload)) == 1

    def test_scheme_and_host_case_deduplicate(self):
        payload = {
            "results": [
                _item(url="HTTPS://Example.test/Page"),
                _item(url="https://example.test/Page"),
            ]
        }
        assert len(parse_results(payload)) == 1

    def test_duplicate_engines_are_merged(self):
        payload = {
            "results": [
                _item(url="https://a.test/p", engines=["duckduckgo"]),
                _item(url="https://a.test/p/", engines=["google cse"]),
            ]
        }
        results = parse_results(payload)
        assert len(results) == 1
        assert set(results[0].engines) == {"duckduckgo", "google cse"}

    def test_ranking_order_is_preserved(self):
        payload = {"results": [_item(url=f"https://a.test/{i}") for i in range(5)]}
        results = parse_results(payload)
        assert [r.normalized_url for r in results] == [f"https://a.test/{i}" for i in range(5)]

    def test_dedupe_results_helper_on_constructed_results(self):
        payload = {"results": [_item(url="https://a.test/p"), _item(url="https://a.test/p/")]}
        results = parse_results(payload)
        assert len(dedupe_results(results)) == 1

    def test_unresponsive_engines_parsed(self):
        payload = {"unresponsive_engines": ["brave : Suspended: too many requests", "duckduckgo : CAPTCHA"]}
        assert unresponsive_engines(payload) == (
            "brave : Suspended: too many requests",
            "duckduckgo : CAPTCHA",
        )

    @pytest.mark.parametrize("payload", [None, [], "x", 3, {}, {"unresponsive_engines": []}, {"unresponsive_engines": 5}])
    def test_unresponsive_engines_tolerates_anything(self, payload):
        assert unresponsive_engines(payload) == ()

    def test_unresponsive_engines_parses_real_engine_reason_pairs(self):
        """Real SearXNG answers with [engine, reason] pairs, not strings.

        Returning () for that shape would silently hide that a query ran with
        several engines suspended, which is exactly the diagnostic a caller
        needs when a search comes back empty.
        """
        payload = {
            "unresponsive_engines": [
                ["brave", "Suspended: too many requests"],
                ["duckduckgo", "CAPTCHA"],
            ]
        }
        assert unresponsive_engines(payload) == (
            "brave : Suspended: too many requests",
            "duckduckgo : CAPTCHA",
        )

    def test_unresponsive_engines_renders_a_partial_pair(self):
        assert unresponsive_engines({"unresponsive_engines": [["brave", ""]]}) == ("brave",)

    def test_unresponsive_engines_drops_unusable_pairs(self):
        assert unresponsive_engines({"unresponsive_engines": [[], {}, None]}) == ()

    def test_unresponsive_engines_accepts_a_single_engine_string(self):
        assert unresponsive_engines({"unresponsive_engines": "brave : CAPTCHA"}) == ("brave : CAPTCHA",)


# --------------------------------------------------------------------------- #
# host_for
# --------------------------------------------------------------------------- #


class TestHostFor:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://en.wikipedia.org/wiki/X", "en.wikipedia.org"),
            ("HTTPS://En.Wikipedia.ORG/wiki/X", "en.wikipedia.org"),
            ("http://a.test:8080/p", "a.test:8080"),
            ("ftp://a.test/p", ""),
            ("not-a-url", ""),
            ("", ""),
        ],
    )
    def test_host_for(self, url, expected):
        assert host_for(url) == expected


# --------------------------------------------------------------------------- #
# HTTP behaviour
# --------------------------------------------------------------------------- #


class TestClientConfig:
    @pytest.mark.parametrize("base", ["", "   ", "localhost:8080", "ftp://x.test", "/search"])
    def test_bad_base_url_rejected(self, base):
        with pytest.raises(SearxngConfigurationError):
            SearxngClient(base)

    @pytest.mark.parametrize(
        ("kwargs",),
        [
            ({"timeout_seconds": 0},),
            ({"timeout_seconds": -1},),
            ({"max_results": 0},),
            ({"max_retries": -1},),
        ],
    )
    def test_bounds_are_validated(self, kwargs):
        with pytest.raises(SearxngConfigurationError):
            SearxngClient("http://127.0.0.1:8080", **kwargs)

    @pytest.mark.parametrize("query", ["", "   "])
    def test_empty_query_rejected(self, query):
        client = _client(_json_handler({"results": []}))
        with pytest.raises(SearxngConfigurationError):
            client.search(query)

    def test_exposed_urls(self):
        client = SearxngClient("http://127.0.0.1:8080/")
        assert client.base_url == "http://127.0.0.1:8080"
        assert client.search_url == "http://127.0.0.1:8080/search"


class TestCategorySelection:
    """The bug this guards: SearXNG shipped a default engine set with no working
    general web-search engine, so every query returned zero results.

    `categories` is a DISCOVERY hint and must never be read as an authority
    signal - which results may back a claim is Stage 3's decision alone.
    """

    def test_default_categories_are_general_only(self):
        """`science` was measured and rejected: it reduced domain diversity."""
        assert DEFAULT_CATEGORIES == "general"

    def test_search_sends_the_categories_asking_for(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["params"] = dict(request.url.params)
            return httpx.Response(200, json={"results": [_item()]})

        _client(handler).search("triple point of water")
        assert seen["params"]["categories"] == DEFAULT_CATEGORIES

    def test_categories_are_overridable_without_touching_other_settings(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["params"] = dict(request.url.params)
            return httpx.Response(200, json={"results": []})

        _client(handler, categories="general").search("triple point")
        assert seen["params"]["categories"] == "general"

    def test_client_from_env_defaults_to_the_general_category(self):
        client = client_from_env({})
        assert client._categories == DEFAULT_CATEGORIES

    def test_client_from_env_honours_an_explicit_override(self):
        client = client_from_env({"SEARXNG_CATEGORIES": "news"})
        assert client._categories == "news"


class TestClientSearch:
    def test_successful_search(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["params"] = dict(request.url.params)
            return httpx.Response(200, json={"results": [_item()]})

        results = _client(handler).search("triple point of water")
        assert len(results) == 1
        assert seen["params"]["q"] == "triple point of water"
        assert seen["params"]["format"] == "json"
        assert seen["params"]["language"] == "en"

    def test_max_results_truncates_after_dedup(self):
        payload = {"results": [_item(url=f"https://a.test/{i}") for i in range(10)]}
        results = _client(_json_handler(payload)).search("q", max_results=3)
        assert len(results) == 3

    def test_with_diagnostics_returns_dead_engines(self):
        payload = {"results": [_item()], "unresponsive_engines": ["brave : Suspended: too many requests"]}
        results, dead = _client(_json_handler(payload)).search_with_diagnostics("q")
        assert len(results) == 1
        assert dead == ("brave : Suspended: too many requests",)

    def test_403_raises_the_actionable_format_error(self):
        """SearXNG answers a disallowed format=json with 403; it needs its own error."""
        handler = lambda request: httpx.Response(403, text="Forbidden", headers={"content-type": "text/html"})
        with pytest.raises(SearxngFormatDisabledError) as excinfo:
            _client(handler).search("q")
        assert "search.formats" in str(excinfo.value)

    def test_html_body_with_200_is_malformed(self):
        handler = lambda request: httpx.Response(
            200, text="<html>not json</html>", headers={"content-type": "text/html; charset=utf-8"}
        )
        with pytest.raises(SearxngMalformedResponseError):
            _client(handler).search("q")

    def test_json_content_type_but_invalid_body_is_malformed(self):
        handler = lambda request: httpx.Response(200, text="{not json", headers={"content-type": "application/json"})
        with pytest.raises(SearxngMalformedResponseError):
            _client(handler).search("q")

    def test_json_array_top_level_is_malformed(self):
        handler = lambda request: httpx.Response(200, json=[1, 2, 3])
        with pytest.raises(SearxngMalformedResponseError):
            _client(handler).search("q")

    @pytest.mark.parametrize("status", [401, 407])
    def test_auth_statuses_are_api_errors(self, status):
        handler = lambda request: httpx.Response(status)
        with pytest.raises(SearxngAPIError):
            _client(handler).search("q")

    @pytest.mark.parametrize("status", [400, 404, 418])
    def test_non_retryable_statuses_are_api_errors(self, status):
        handler = lambda request: httpx.Response(status)
        with pytest.raises(SearxngAPIError):
            _client(handler).search("q")

    @pytest.mark.parametrize("status", [500, 502, 503, 504])
    def test_retryable_statuses_are_retried_then_raise(self, status):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(status_code=status)

        with pytest.raises(SearxngAPIError):
            _client(handler, max_retries=1, sleep=lambda _s: None).search("q")
        assert calls["n"] == 2, "one initial attempt plus one retry"

    def test_retry_recovers_on_second_attempt(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(status_code=503)
            return httpx.Response(200, json={"results": [_item()]})

        results = _client(handler, max_retries=1, sleep=lambda _s: None).search("q")
        assert calls["n"] == 2
        assert len(results) == 1

    def test_timeout_is_typed(self):
        handler = lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("too slow"))
        with pytest.raises(SearxngTimeoutError):
            _client(handler, max_retries=0, sleep=lambda _s: None).search("q")

    def test_connection_refused_explains_how_to_start_searxng(self):
        handler = lambda request: (_ for _ in ()).throw(httpx.ConnectError("refused"))
        with pytest.raises(SearxngConnectionError) as excinfo:
            _client(handler, max_retries=0, sleep=lambda _s: None).search("q")
        message = str(excinfo.value)
        assert "docker compose" in message
        assert "searxng" in message

    def test_probe_true_and_false(self):
        ok = _client(lambda request: httpx.Response(200, text="OK"))
        assert ok.probe() is True

        down = _client(lambda request: (_ for _ in ()).throw(httpx.ConnectError("refused")))
        assert down.probe() is False, "probe reports, never raises"

    def test_probe_false_on_non_200(self):
        assert _client(lambda request: httpx.Response(503)).probe() is False


class TestDeterminism:
    def test_identical_payload_gives_identical_ids(self):
        """source_id_for must be stable across runs for the same canonical URL."""
        url = "https://en.wikipedia.org/wiki/Triple_point"
        assert source_id_for(url) == source_id_for(url)

    def test_adapter_ids_match_the_repository_helper(self):
        payload = {"results": [_item()]}
        results = parse_results(payload)
        assert results[0].normalized_url == normalize_source_url(results[0].url)

    def test_repeated_parses_are_identical(self):
        payload = {
            "results": [
                _item(url="https://a.test/p"),
                _item(url="https://a.test/p/", engines=["google cse"]),
                _item(url="https://b.test/q", score=0.5),
            ]
        }
        first = parse_results(payload)
        second = parse_results(payload)
        assert first == second