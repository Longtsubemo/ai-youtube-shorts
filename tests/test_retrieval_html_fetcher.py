"""Offline tests for the BeautifulSoup fetcher.

No network: HTTP behaviour goes through an injected `httpx.MockTransport`, and
HTML extraction is tested against inline fixture strings.

The central property under test is honesty — a page that was blocked, was not
HTML, was empty, or was an interstitial must never come back as a success.
"""

from __future__ import annotations

import httpx
import pytest

from experiments.retrieval.errors import FetcherConfigurationError
from experiments.retrieval.html_fetcher import (
    BLOCKING_STATUSES,
    HtmlFetcher,
    extract_document,
    is_scriptish,
    looks_blocked,
    summarize,
)

ARTICLE = """
<!doctype html>
<html lang="en">
<head>
  <title>Triple Point of Water - Example Science</title>
  <meta name="description" content="The triple point of water explained.">
  <meta name="author" content="Example Author">
  <link rel="canonical" href="https://example.test/triple-point">
  <script>var tracking = 1; console.log("SHOULD NOT APPEAR");</script>
  <style>body { color: red; } /* SHOULD NOT APPEAR */</style>
</head>
<body>
  <nav>Home About Contact SHOULD NOT APPEAR</nav>
  <header>Site header SHOULD NOT APPEAR</header>
  <main>
    <article>
      <h1>Triple Point of Water</h1>
      <p>The triple point of water is the unique combination of temperature and
         pressure at which liquid water, water vapor, and ice coexist in
         equilibrium.</p>
      <p>At this point all three phases have the same chemical potential, so any
         further change in temperature or pressure will reduce the total Gibbs
         free energy of the system by converting some of one phase into another.</p>
      <p>The International Bureau of Weights and Measures used the triple point
         of water to define the kelvin before the 2019 revision of the SI.</p>
      <p>Because the equilibrium is exact and reproducible, the triple point is a
         convenient physical reference point for calibration and for teaching
         phase diagrams in introductory physics courses.</p>
    </article>
  </main>
  <aside>Related links SHOULD NOT APPEAR</aside>
  <footer>Copyright SHOULD NOT APPEAR</footer>
</body>
</html>
"""


def _fetcher(handler, **kwargs):
    return HtmlFetcher(transport=httpx.MockTransport(handler), **kwargs)


def _html_handler(body, status=200, content_type="text/html; charset=utf-8"):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=body, headers={"content-type": content_type})

    return handler


# --------------------------------------------------------------------------- #
# extract_document (pure)
# --------------------------------------------------------------------------- #


class TestExtractDocument:
    def test_extracts_title_and_metadata(self):
        doc = extract_document(ARTICLE)
        assert doc.title == "Triple Point of Water - Example Science"
        assert doc.meta_description == "The triple point of water explained."
        assert doc.language == "en"
        assert doc.author == "Example Author"
        assert doc.canonical_url == "https://example.test/triple-point"

    def test_extracts_readable_prose(self):
        doc = extract_document(ARTICLE)
        assert "unique combination of temperature" in doc.text
        assert "coexist in" in doc.text
        assert doc.content_length == len(doc.text)

    @pytest.mark.parametrize("noise", ["SHOULD NOT APPEAR"])
    def test_script_style_and_chrome_are_removed(self, noise):
        text = extract_document(ARTICLE).text
        assert noise not in text
        assert "tracking" not in text
        assert "console.log" not in text

    def test_headings_and_paragraphs_stay_separated(self):
        text = extract_document(ARTICLE).text
        assert "Triple Point of Water" in text
        assert "\n" in text

    def test_article_root_is_preferred_over_navigation(self):
        text = extract_document(ARTICLE).text
        assert "About Contact" not in text

    def test_base_url_used_when_no_canonical_link(self):
        doc = extract_document("<html><body><p>x</p></body></html>", base_url="https://a.test/p")
        assert doc.canonical_url == "https://a.test/p"

    def test_title_falls_back_to_og_then_h1(self):
        og = extract_document(
            '<html><head><meta property="og:title" content="OG Title"></head>'
            "<body><h1>H1 Title</h1><p>text</p></body></html>"
        )
        assert og.title == "OG Title"

        h1 = extract_document("<html><body><h1>H1 Title</h1><p>text</p></body></html>")
        assert h1.title == "H1 Title"

    @pytest.mark.parametrize("html", ["", "   ", "\n\t"])
    def test_empty_input_yields_empty_document(self, html):
        doc = extract_document(html)
        assert doc.text == ""
        assert doc.title == ""
        assert doc.content_length == 0

    @pytest.mark.parametrize(
        "html",
        [
            "<html><body><p>unclosed",
            "<p>no html or body tag",
            "<html><head><title>Only a title</title></head></html>",
            "<<<>>>not markup at all",
            "plain text with no tags",
        ],
    )
    def test_malformed_or_non_html_never_raises(self, html):
        doc = extract_document(html)
        assert isinstance(doc.text, str)

    def test_body_with_no_tags_produces_no_text(self):
        assert extract_document("<html><body></body></html>").text == ""


# --------------------------------------------------------------------------- #
# blocked / script detection
# --------------------------------------------------------------------------- #


class TestDetection:
    @pytest.mark.parametrize("status", sorted(BLOCKING_STATUSES))
    def test_blocking_status_is_blocked(self, status):
        assert looks_blocked(status=status, title="A Fine Page", text="plenty of real prose here") is True

    @pytest.mark.parametrize("status", [200, 201, 301, 404, 410, 500])
    def test_non_blocking_status_is_not_blocked(self, status):
        assert looks_blocked(status=status, title="A Fine Page", text="plenty of real prose") is False

    @pytest.mark.parametrize(
        ("title", "text"),
        [
            ("Just a moment...", "Checking your browser before accessing"),
            ("Attention Required! | Cloudflare", "blocked"),
            ("Security Check", "Please enable JavaScript and cookies to continue"),
            ("", "To continue, please confirm you are human"),
            ("Google Search", "If you're having trouble accessing Google Search, please click here"),
            ("Redirecting", "click here if you are redirected"),
            ("", "Verifying you are human"),
        ],
    )
    def test_known_interstitials_are_blocked(self, title, text):
        assert looks_blocked(status=200, title=title, text=text) is True

    def test_ordinary_article_is_not_blocked(self):
        doc = extract_document(ARTICLE)
        assert looks_blocked(status=200, title=doc.title, text=doc.text) is False

    def test_short_page_without_marker_is_not_blocked(self):
        """Length alone must never condemn a page."""
        assert looks_blocked(status=200, title="Short", text="Only three words") is False

    def test_is_scriptish(self):
        assert is_scriptish("") is True
        assert is_scriptish("var x = function (a) { return a; }") is True
        assert is_scriptish("The triple point of water is a unique thermodynamic state.") is False


# --------------------------------------------------------------------------- #
# HtmlFetcher.fetch
# --------------------------------------------------------------------------- #


class TestFetcherConfig:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"timeout_seconds": 0},
            {"timeout_seconds": -5},
            {"max_content_chars": 0},
            {"max_retries": -1},
            {"min_usable_chars": -1},
        ],
    )
    def test_bounds_are_validated(self, kwargs):
        with pytest.raises(FetcherConfigurationError):
            HtmlFetcher(**kwargs)

    @pytest.mark.parametrize("url", ["", "   ", "not-a-url", "ftp://a.test/x", "/relative/path", "example.test/x"])
    def test_invalid_urls_fail_as_invalid_url_not_an_exception(self, url):
        page = HtmlFetcher(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=ARTICLE))).fetch(url)
        assert page.ok is False
        assert page.error_kind == "invalid_url"
        assert page.text == ""


class TestFetcherSuccess:
    def test_successful_fetch(self):
        page = _fetcher(_html_handler(ARTICLE)).fetch("https://example.test/triple-point")
        assert page.ok is True
        assert page.http_status == 200
        assert page.domain == "example.test"
        assert page.title == "Triple Point of Water - Example Science"
        assert page.language == "en"
        assert "unique combination" in page.text
        assert page.content_length == len(page.text)
        assert page.error == ""
        assert page.error_kind == ""
        assert page.elapsed_ms is not None

    def test_content_is_bounded_and_marked(self):
        page = _fetcher(_html_handler(ARTICLE), max_content_chars=50).fetch("https://example.test/p")
        assert page.ok is True
        assert page.text.endswith(" …[truncated]")
        assert page.content_length == len(page.text)

    def test_to_source_fields_are_contract_shaped(self):
        page = _fetcher(_html_handler(ARTICLE)).fetch("https://example.test/p")
        fields = page.to_source_fields()
        assert fields["title"] == page.title
        assert fields["publisher"] == "example.test"
        assert fields["publication_date"] == "", "a date is never inferred from markup"
        assert fields["content"] == page.text

    def test_text_plain_is_accepted(self):
        body = (
            "The triple point of water is the unique combination of temperature and "
            "pressure at which liquid water, water vapor, and ice coexist in equilibrium. "
            "At this point all three phases have the same chemical potential, so any "
            "further change in temperature or pressure will reduce the total Gibbs free "
            "energy of the system by converting some of one phase into another entirely."
        )
        handler = _html_handler(body, content_type="text/plain")
        page = _fetcher(handler).fetch("https://example.test/p.txt")
        assert page.ok is True
        assert "plain" not in page.text
        assert "coexist in equilibrium" in page.text

    def test_sends_an_identifying_user_agent(self):
        """The fetcher must not impersonate a browser."""
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["ua"] = request.headers.get("user-agent", "")
            return httpx.Response(200, text=ARTICLE, headers={"content-type": "text/html"})

        _fetcher(handler).fetch("https://example.test/p")
        assert "Mozilla" not in seen["ua"]
        assert "ai-youtube-channel-retrieval" in seen["ua"]

    def test_fetch_many_preserves_order(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if "good" in str(request.url):
                return httpx.Response(200, text=ARTICLE, headers={"content-type": "text/html"})
            return httpx.Response(404)

        pages = _fetcher(handler).fetch_many(
            ["https://good.test/p", "https://bad.test/p", "https://good.test/q"]
        )
        assert [p.ok for p in pages] == [True, False, True]


class TestFetcherFailures:
    def test_403_is_blocked_not_a_success(self):
        handler = lambda request: httpx.Response(403, text="Forbidden", headers={"content-type": "text/html"})
        page = _fetcher(handler).fetch("https://example.test/p")
        assert page.ok is False
        assert page.error_kind == "blocked"
        assert page.http_status == 403
        assert page.text == ""
        assert page.title == ""

    def test_429_is_blocked(self):
        page = _fetcher(lambda r: httpx.Response(429)).fetch("https://example.test/p")
        assert page.ok is False
        assert page.error_kind == "blocked"

    def test_404_is_http_error(self):
        page = _fetcher(lambda r: httpx.Response(404)).fetch("https://example.test/p")
        assert page.ok is False
        assert page.error_kind == "http_error"
        assert page.http_status == 404

    def test_500_is_http_error(self):
        page = _fetcher(lambda r: httpx.Response(500)).fetch("https://example.test/p")
        assert page.ok is False
        assert page.error_kind == "http_error"

    def test_timeout_is_typed_and_retried(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            raise httpx.ReadTimeout("too slow")

        page = _fetcher(handler, max_retries=1, sleep=lambda _s: None).fetch("https://example.test/p")
        assert page.ok is False
        assert page.error_kind == "timeout"
        assert calls["n"] == 2

    def test_connect_error_is_typed(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        page = _fetcher(handler, max_retries=0, sleep=lambda _s: None).fetch("https://example.test/p")
        assert page.ok is False
        assert page.error_kind == "connection"

    def test_non_html_content_type_is_rejected(self):
        handler = _html_handler("%PDF-1.4 fake pdf body", content_type="application/pdf")
        page = _fetcher(handler).fetch("https://example.test/doc.pdf")
        assert page.ok is False
        assert page.error_kind == "content_type"
        assert "not HTML" in page.error
        assert page.text == "", "a PDF is never passed off as parsed content"

    def test_json_content_type_is_rejected(self):
        page = _fetcher(_html_handler('{"a":1}', content_type="application/json")).fetch("https://a.test/x")
        assert page.ok is False
        assert page.error_kind == "content_type"

    def test_empty_body_is_empty_content(self):
        page = _fetcher(_html_handler("")).fetch("https://example.test/p")
        assert page.ok is False
        assert page.error_kind == "empty_content"

    def test_captcha_interstitial_is_blocked_despite_http_200(self):
        body = "<html><head><title>Just a moment...</title></head><body><h1>Checking your browser</h1></body></html>"
        page = _fetcher(_html_handler(body)).fetch("https://example.test/p")
        assert page.ok is False
        assert page.error_kind == "blocked"
        assert page.http_status == 200, "the 200 is still reported, the page is just not usable"

    def test_thin_content_is_reported_as_a_failure(self):
        body = "<html><head><title>Tiny</title></head><body><p>Too short.</p></body></html>"
        page = _fetcher(_html_handler(body)).fetch("https://example.test/p")
        assert page.ok is False
        assert page.error_kind == "thin_content"
        # The text is retained for inspection even though it is not usable.
        assert "Too short" in page.text

    def test_thin_content_threshold_is_disableable(self):
        body = "<html><head><title>Tiny</title></head><body><p>Too short.</p></body></html>"
        page = _fetcher(_html_handler(body), min_usable_chars=0).fetch("https://example.test/p")
        assert page.ok is True

    def test_blocked_page_to_source_fields_are_all_empty(self):
        page = _fetcher(lambda r: httpx.Response(403)).fetch("https://example.test/p")
        fields = page.to_source_fields()
        assert fields == {"title": "", "publisher": "", "publication_date": "", "content": ""}

    def test_fetch_never_raises_on_network_errors(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise RuntimeError("something entirely unexpected")

        page = _fetcher(handler, max_retries=0, sleep=lambda _s: None).fetch("https://example.test/p")
        assert page.ok is False
        assert page.error_kind == "connection"
        assert "RuntimeError" in page.error


class TestSummarize:
    def test_counts_match_observations(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if "good" in str(request.url):
                return httpx.Response(200, text=ARTICLE, headers={"content-type": "text/html"})
            return httpx.Response(403)

        pages = _fetcher(handler).fetch_many(
            ["https://good.test/p", "https://bad.test/p", "https://bad.test/q"]
        )
        stats = summarize(pages)
        assert stats["attempted"] == 3
        assert stats["succeeded"] == 1
        assert stats["failed"] == 2
        assert stats["titles_available"] == 1
        assert stats["http_status_counts"] == {"200": 1, "403": 2}
        assert stats["error_kinds"] == {"blocked": 2}
        assert stats["total_content_chars"] > 0

    def test_no_status_is_reported_as_none(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        pages = _fetcher(handler, max_retries=0, sleep=lambda _s: None).fetch_many(["https://a.test/p"])
        assert summarize(pages)["http_status_counts"] == {"none": 1}

    def test_empty_input(self):
        assert summarize([]) == {
            "attempted": 0,
            "succeeded": 0,
            "failed": 0,
            "titles_available": 0,
            "total_content_chars": 0,
            "http_status_counts": {},
            "error_kinds": {},
        }