"""Offline tests for FirecrawlClient (response normalization + errors).

No Firecrawl SDK and no network are touched: the fake raw client mimics the
v2 `search` return shape, and every interesting failure mode is scripted.
"""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from providers.firecrawl.client import FirecrawlResult
from providers.firecrawl.config import FirecrawlConfig
from providers.firecrawl.errors import (
    FirecrawlAPIError,
    FirecrawlAuthenticationError,
    FirecrawlConfigurationError,
    FirecrawlMalformedResponseError,
    FirecrawlTimeoutError,
    FirecrawlUnexpectedError,
)
from tests.firecrawl_test_utils import (
    FakeDocument,
    FakeFirecrawlRawClient,
    FakeWebResult,
    fake_metadata,
    make_firecrawl_client,
    make_http_error,
    web_bucket,
    web_dict,
)

URL = "https://example.test/page"
MAX_RETRIES = 2
SECRET = "fc-top-secret-key"


class _FakeTimeout(TimeoutError):
    pass


class _ScrapeDocument:
    """Mimic the v2 `Document` returned by `client.scrape(...)`."""

    def __init__(self, markdown: str, metadata=None) -> None:
        self.markdown = markdown
        self.metadata = metadata


class _ScrapeRawClient:
    """Mimics `firecrawl.v2.FirecrawlClient.scrape(url, **options)`."""

    def __init__(self, document=None) -> None:
        self.document = document
        self.outcomes: list[Exception] = []
        self.calls: list[tuple] = []
        self._idx = 0

    def add_error(self, exc: Exception) -> None:
        self.outcomes.append(exc)

    @property
    def attempt_count(self) -> int:
        return len(self.calls)

    def scrape(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        if self._idx < len(self.outcomes):
            exc = self.outcomes[self._idx]
            self._idx += 1
            raise exc
        return self.document


def _client(raw, **cfg_kwargs):
    config = FirecrawlConfig(max_retries=MAX_RETRIES, **cfg_kwargs)
    return make_firecrawl_client(config, raw_client=raw)


def _args(raw):
    return raw.calls[0]


class TestNormalization:
    def test_web_results_are_normalized(self):
        raw = FakeFirecrawlRawClient()
        raw.add_result(
            web_bucket(
                FakeWebResult(
                    URL,
                    title="Page title",
                    description="Snippet",
                    markdown="Full markdown content.",
                    metadata=fake_metadata(URL),
                )
            )
        )
        (result,) = _client(raw).search("octopus hearts", limit=3)
        assert isinstance(result, FirecrawlResult)
        assert result.url == URL
        assert result.title == "Page title"
        assert result.description == "Snippet"
        assert result.markdown == "Full markdown content."
        assert result.source_url == URL

    def test_default_limit_is_config_max_search_results(self):
        raw = FakeFirecrawlRawClient(outcome=web_bucket())
        client = _client(raw, max_search_results=7)
        client.search("octopus")
        assert _args(raw)[1]["limit"] == 7

    def test_explicit_limit_wins(self):
        raw = FakeFirecrawlRawClient(outcome=web_bucket())
        client = _client(raw, max_search_results=7)
        client.search("octopus", limit=2)
        assert _args(raw)[1]["limit"] == 2

    def test_search_requests_markdown_content_in_one_call(self):
        raw = FakeFirecrawlRawClient(outcome=web_bucket())
        _client(raw).search("octopus")
        kwargs = _args(raw)[1]
        assert kwargs["sources"] == ["web"]
        assert kwargs["ignore_invalid_urls"] is True
        assert kwargs["timeout"] == int(FirecrawlConfig().timeout_seconds * 1000)
        scrape_options = kwargs["scrape_options"]
        assert scrape_options["formats"] == ["markdown"]
        assert "crawl" not in kwargs

    def test_dict_response_supported(self):
        raw = FakeFirecrawlRawClient(outcome=web_dict({"url": URL, "metadata": {"sourceURL": URL}}))
        (result,) = _client(raw).search("octopus")
        assert result.url == URL
        assert result.source_url == URL

    def test_hydrated_document_from_metadata_is_normalized(self):
        """Real v2 `Document` results carry url/title via `metadata`, not top-level."""
        raw = FakeFirecrawlRawClient(
            outcome=web_bucket(FakeDocument(metadata=fake_metadata(URL), markdown="# md"))
        )
        (result,) = _client(raw).search("octopus")
        assert result.url == URL
        assert result.title == "Metadata title"
        assert result.markdown == "# md"
        assert result.source_url == URL

    def test_document_without_resolvable_url_is_skipped(self):
        raw = FakeFirecrawlRawClient(outcome=web_bucket(FakeDocument(metadata=fake_metadata())))
        assert _client(raw).search("octopus") == ()

    def test_web_item_url_falls_back_to_metadata_source_url(self):
        raw = FakeFirecrawlRawClient(outcome=web_dict({"metadata": {"source_url": URL}}))
        (result,) = _client(raw).search("octopus")
        assert result.url == URL
        assert result.source_url == URL

    def test_metadata_title_fills_missing_top_level_title(self):
        raw = FakeFirecrawlRawClient(
            outcome=web_dict({"url": URL, "metadata": {"source_url": URL, "title": "Meta title"}})
        )
        (result,) = _client(raw).search("octopus")
        assert result.title == "Meta title"

    def test_legacy_data_bucket_fallback(self):
        raw = FakeFirecrawlRawClient(outcome=SimpleNamespace(data=[FakeWebResult(URL)]))
        (result,) = _client(raw).search("octopus")
        assert result.url == URL

    def test_missing_bucket_is_empty_result_set(self):
        raw = FakeFirecrawlRawClient(outcome=SimpleNamespace())
        assert _client(raw).search("octopus") == ()

    def test_invalid_item_url_is_skipped(self):
        raw = FakeFirecrawlRawClient(outcome=web_bucket(FakeWebResult("not a url"), FakeWebResult(URL)))
        results = _client(raw).search("octopus")
        assert [r.url for r in results] == [URL]

    def test_item_without_url_is_skipped(self):
        raw = FakeFirecrawlRawClient(outcome=web_bucket(FakeWebResult(""), FakeWebResult(URL)))
        results = _client(raw).search("octopus")
        assert [r.url for r in results] == [URL]

    def test_non_list_bucket_raises_malformed_response(self):
        raw = FakeFirecrawlRawClient(outcome=SimpleNamespace(web="nope"))
        with pytest.raises(FirecrawlMalformedResponseError):
            _client(raw).search("octopus")


class TestErrorsAndRetries:
    def test_transient_error_retries_then_succeeds(self):
        raw = FakeFirecrawlRawClient()
        raw.add_error(make_http_error(429))
        raw.add_result(web_bucket(FakeWebResult(URL)))
        results = _client(raw).search("octopus")
        assert [r.url for r in results] == [URL]
        assert raw.attempt_count == 2

    def test_timeout_retries_then_raises_after_bounded_budget(self):
        raw = FakeFirecrawlRawClient(outcome=_FakeTimeout("boom"))
        client = _client(raw)
        with pytest.raises(FirecrawlTimeoutError):
            client.search("octopus")
        assert raw.attempt_count == MAX_RETRIES + 1

    def test_transient_status_exhausts_budget_then_raises(self):
        raw = FakeFirecrawlRawClient(outcome=make_http_error(503))
        with pytest.raises(FirecrawlAPIError):
            _client(raw).search("octopus")
        assert raw.attempt_count == MAX_RETRIES + 1

    def test_authentication_error_does_not_retry(self):
        raw = FakeFirecrawlRawClient(outcome=make_http_error(401))
        with pytest.raises(FirecrawlAuthenticationError):
            _client(raw).search("octopus")
        assert raw.attempt_count == 1

    def test_provider_error_propagates_without_retry(self):
        raw = FakeFirecrawlRawClient(
            outcome=FirecrawlMalformedResponseError("boom")
        )
        with pytest.raises(FirecrawlMalformedResponseError):
            _client(raw).search("octopus")
        assert raw.attempt_count == 1

    def test_missing_sdk_maps_to_configuration_error(self):
        raw = FakeFirecrawlRawClient(outcome=ImportError("No module named 'firecrawl'"))
        with pytest.raises(FirecrawlConfigurationError, match="Firecrawl SDK is not installed"):
            _client(raw).search("octopus")
        assert raw.attempt_count == 1

    def test_missing_sdk_at_resolution_maps_to_configuration_error(self):
        """A real client without the SDK must fail cleanly, never leak ImportError."""
        from providers.firecrawl.client import FirecrawlClient

        client = FirecrawlClient(FirecrawlConfig(max_retries=0), sleep=lambda *_: None)

        def boom():
            raise ModuleNotFoundError("No module named 'firecrawl'")

        client._get_client = boom  # instance attribute shadows the real method
        with pytest.raises(FirecrawlConfigurationError, match="Firecrawl SDK is not installed"):
            client.search("octopus")

    def test_unknown_exception_maps_to_unexpected_error(self):
        raw = FakeFirecrawlRawClient(outcome=RuntimeError("weird"))
        with pytest.raises(FirecrawlUnexpectedError):
            _client(raw).search("octopus")


class TestSDKCompatibility:
    def test_keyless_resolution_builds_v2_client(self):
        """firecrawl>=4.44.0's top-level `Firecrawl` requires an API key; the v2
        `FirecrawlClient` backs the documented keyless path."""
        firecrawl_v2 = pytest.importorskip("firecrawl.v2")  # skip only if SDK absent
        from providers.firecrawl.client import FirecrawlClient

        client = FirecrawlClient(FirecrawlConfig())
        raw = client._get_client()
        assert isinstance(raw, firecrawl_v2.FirecrawlClient)
        assert callable(raw.search)

    def test_keyed_resolution_builds_v2_client_too(self):
        pytest.importorskip("firecrawl.v2")
        from providers.firecrawl.client import FirecrawlClient

        client = FirecrawlClient(FirecrawlConfig(api_key=SECRET))
        raw = client._get_client()
        assert raw.http_client.api_key == SECRET


class TestSecretsDiscipline:
    def test_key_never_appears_in_error_messages(self):
        raw = FakeFirecrawlRawClient(outcome=make_http_error(403))
        with pytest.raises(FirecrawlAuthenticationError) as exc:
            _client(raw, api_key=SECRET).search("octopus")
        assert SECRET not in str(exc.value)

    def test_export_redacted_never_contains_key(self):
        client = _client(FakeFirecrawlRawClient(outcome=web_bucket()), api_key=SECRET)
        data = client.export_redacted()
        assert SECRET not in str(data)
        assert data["provider"] == "firecrawl"


class TestScrapeTier:
    """The expensive single-page tier, including the two ways it used to
    manufacture evidence out of failure."""

    def _scrape_client(self, document, raw_client=None):
        raw = raw_client if raw_client is not None else _ScrapeRawClient(document)
        return _client(raw)

    def test_real_page_is_returned_in_full(self):
        client = self._scrape_client(
            _ScrapeDocument(
                markdown="Quantum entanglement is a physical phenomenon.",
                metadata=SimpleNamespace(statusCode=200, title="Quantum entanglement", sourceURL=URL),
            )
        )
        result = client.scrape(URL)
        assert result.markdown == "Quantum entanglement is a physical phenomenon."
        assert result.url == URL

    def test_scrape_pins_the_basic_proxy(self):
        """`proxy="auto"` is unusable self-hosted: on a 403 the fetch engine
        raises AddFeatureError(["stealthProxy"]), a cloud-only capability, and
        the request ends as an opaque HTTP 500. Verified against the live
        self-hosted stack."""
        raw = _ScrapeRawClient(
            _ScrapeDocument(
                markdown="Real prose about the triple point.",
                metadata=SimpleNamespace(statusCode=200, title="Triple point", sourceURL=URL),
            )
        )
        self._scrape_client(None, raw_client=raw).scrape(URL)
        assert _args(raw)[1]["proxy"] == "basic"

    def test_requested_url_wins_over_a_lying_metadata_url(self):
        client = self._scrape_client(
            _ScrapeDocument(
                markdown="Prose.",
                metadata=SimpleNamespace(statusCode=200, title="T", sourceURL="https://elsewhere.test/x"),
            )
        )
        assert client.scrape(URL).url == URL

    def test_empty_markdown_is_a_typed_failure_not_an_empty_source(self):
        client = self._scrape_client(
            _ScrapeDocument(markdown="   ", metadata=SimpleNamespace(statusCode=200, title="T", sourceURL=URL))
        )
        with pytest.raises(FirecrawlMalformedResponseError):
            client.scrape(URL)

    def test_http_404_page_is_rejected_despite_an_http_200_envelope(self):
        """The live self-hosted stack answers HTTP 200 with
        metadata.statusCode 404 and ~25k characters of "Page Not Found"
        navigation. Accepting it would fabricate a source."""
        client = self._scrape_client(
            _ScrapeDocument(
                markdown="Search\n\nSuggested Searches\n* Artemis\n* ISS\n",
                metadata=SimpleNamespace(statusCode=404, title="Page Not Found - NASA", sourceURL=URL),
            )
        )
        with pytest.raises(FirecrawlAPIError) as excinfo:
            client.scrape(URL)
        assert "404" in str(excinfo.value)

    @pytest.mark.parametrize("status", [403, 429, 500, 503])
    def test_any_error_status_is_rejected(self, status):
        client = self._scrape_client(
            _ScrapeDocument(
                markdown="Forbidden",
                metadata=SimpleNamespace(statusCode=status, title="Forbidden", sourceURL=URL),
            )
        )
        with pytest.raises(FirecrawlAPIError):
            client.scrape(URL)

    def test_cloudflare_interstitial_is_rejected_by_name(self):
        """kroneckerwallis.com renders a Cloudflare wall: HTTP 200 envelope,
        metadata.statusCode 403, title "Just a moment...", body "Performing
        security verification". Open-source self-hosted Firecrawl has no stealth
        proxy, so this page is genuinely unreachable and must be dropped."""
        client = self._scrape_client(
            _ScrapeDocument(
                markdown=(
                    "![Icon](https://x.test/favicon.ico)\n\nwww.kroneckerwallis.com\n"
                    "===\n\nPerforming security verification\n---\n\n"
                    "This website uses a security service to protect against malicious bots.\n"
                ),
                metadata=SimpleNamespace(
                    statusCode=200, title="Just a moment...", sourceURL="https://www.kroneckerwallis.com/"
                ),
            )
        )
        with pytest.raises(FirecrawlAPIError) as excinfo:
            client.scrape("https://www.kroneckerwallis.com/")
        assert "challenge" in str(excinfo.value)

    def test_challenge_guard_does_not_reject_ordinary_articles(self):
        """The marker list must not fire on real prose that merely mentions
        verification or JavaScript."""
        client = self._scrape_client(
            _ScrapeDocument(
                markdown=(
                    "Rayleigh scattering\n\nThe verification of this equation required "
                    "careful work. Enable JavaScript and cookies to continue reading the "
                    "full derivation of the scattering cross-section."
                ),
                metadata=SimpleNamespace(statusCode=200, title="Rayleigh scattering", sourceURL=URL),
            )
        )
        assert "Rayleigh scattering" in client.scrape(URL).markdown

    def test_dict_shaped_response_is_supported(self):
        client = self._scrape_client(
            {"markdown": "Prose from a dict.", "metadata": {"statusCode": 200, "title": "T"}}
        )
        assert client.scrape(URL).markdown == "Prose from a dict."

    def test_non_http_url_is_refused_before_any_call(self):
        raw = _ScrapeRawClient(None)
        with pytest.raises(FirecrawlConfigurationError):
            _client(raw).scrape("file:///etc/passwd")
        assert raw.attempt_count == 0

    def test_all_engines_failed_is_not_retried(self):
        """A blocked page is the page's answer, not a transient fault. The
        cloud-only `stealthProxy` engine does not exist here, so retrying only
        spends the budget on a guaranteed identical failure."""

        class Coded(Exception):
            code = "SCRAPE_ALL_ENGINES_FAILED"

        raw = _ScrapeRawClient(None)
        raw.add_error(Coded("all engines failed"))
        raw.add_error(Coded("all engines failed"))
        with pytest.raises(FirecrawlAPIError):
            _client(raw).scrape(URL)
        assert raw.attempt_count == 1

    def test_genuine_transient_status_is_still_retried(self):
        """The retry policy must not have been broken by the change above."""
        raw = _ScrapeRawClient(
            _ScrapeDocument(
                markdown="Prose recovered on the second attempt.",
                metadata=SimpleNamespace(statusCode=200, title="T", sourceURL=URL),
            )
        )
        raw.add_error(make_http_error(503))
        assert "recovered" in _client(raw).scrape(URL).markdown
        assert raw.attempt_count == 2


class TestImportIsolation:
    def test_module_import_never_loads_firecrawl_sdk(self):
        """Importing the package must not import the firecrawl SDK (subprocess)."""
        code = (
            "import sys\n"
            "import providers.firecrawl  # noqa: F401\n"
            "assert 'firecrawl' not in sys.modules, 'firecrawl SDK imported transitively'\n"
            "print('ok')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=".")
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ok"