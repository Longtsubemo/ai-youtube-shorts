"""FirecrawlClient — thin wrapper over the official `firecrawl` SDK.

- The SDK is imported **lazily** so importing this module never forces the
  SDK or a network call; stage packages stay SDK-free (the provider boundary
  lives here).
- We use the v2 Search endpoint (`firecrawl.v2.FirecrawlClient.search`) WITH
  `scrape_options.formats=["markdown"]`: the API returns full-page Markdown
  content for each result in the same call, so no second scrape is needed.
  Results live under `SearchData.web` and may be plain `SearchResultWeb` hits
  or hydrated `Document` objects (url/title/description sourced via
  `metadata` in that case).
- Every SDK/network/parse failure is mapped onto the provider error hierarchy
  with a small bounded retry policy (timeouts and transient statuses only).
  Only normalized plain dataclasses ever escape this module; raw SDK objects
  never leave the provider boundary.
- `scrape()` is the expensive single-page tier, used by the hybrid retriever
  only for pages the cheap BeautifulSoup path could not read. It shares the
  config, the retry budget, and the error mapping with `search()`.
- A scrape is trusted only if it is actually the requested page. Firecrawl
  answers HTTP 200 even when the *page* was a 404 or a bot-challenge wall, so
  `scrape()` checks `metadata.statusCode` and rejects challenge/interstitial
  markup. Without that check a "Page Not Found" body or a Cloudflare
  verification page would be laundered into a real-looking source.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from urllib.parse import urlparse

from providers.firecrawl.config import FirecrawlConfig
from providers.firecrawl.errors import (
    FirecrawlAPIError,
    FirecrawlAuthenticationError,
    FirecrawlConfigurationError,
    FirecrawlMalformedResponseError,
    FirecrawlProviderError,
    FirecrawlTimeoutError,
    FirecrawlUnexpectedError,
)

RETRYABLE_STATUSES = (408, 429, 500, 502, 503, 504)

# Firecrawl's scrape waterfall ends in `SCRAPE_ALL_ENGINES_FAILED` (HTTP 500)
# when a page blocks the request. Retrying cannot help: the block is the page's
# answer, not a transient fault. Mapping it to a retryable error only spends the
# retry budget on a guaranteed second failure.
_ALL_ENGINES_FAILED = "scrape_all_engines_failed"

# Markup that means "you reached an anti-bot wall, not the article".
#
# Split by confidence, because a false positive here silently discards a real
# source. Observed verbatim from this self-hosted stack: kroneckerwallis.com
# returns HTTP 200 with metadata.statusCode 403, `<title>Just a moment...</title>`
# and a body containing "Performing security verification".
#
# STRONG markers are interstitial boilerplate that essentially never appears in
# an article about science or technology, so they are matched anywhere in the
# document.
_STRONG_BOT_CHALLENGE_MARKERS = (
    "performing security verification",
    "verify you are human",
    "verifying you are human",
    "checking your browser before accessing",
    "attention required! | cloudflare",
    "ddos protection by cloudflare",
    "_cf_chl_opt",
)

# WEAK markers are phrases a genuine article may contain when it is *about* the
# web (this file's own test suite has such a sentence). They are only treated as
# a challenge when they appear in the page title, which is what an interstitial
# actually sets.
_WEAK_BOT_CHALLENGE_TITLE_MARKERS = (
    "just a moment",
    "access denied",
    "attention required",
    "are you a robot",
    "security verification",
    "bot verification",
    "enable javascript and cookies to continue",
)

_TIMEOUT_EXCEPTIONS: tuple[type, ...] = ()


@dataclass(frozen=True)
class FirecrawlResult:
    """One normalized search hit from the Firecrawl Search API."""

    url: str
    title: str = ""
    description: str = ""
    markdown: str = ""
    source_url: str = ""


def _timeouts() -> tuple[type, ...]:
    """Resolve network-timeout exception types lazily (SDK uses httpx)."""
    global _TIMEOUT_EXCEPTIONS
    if not _TIMEOUT_EXCEPTIONS:
        try:
            import httpx

            _TIMEOUT_EXCEPTIONS = (
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadTimeout,
                httpx.PoolTimeout,
            )
        except ImportError:  # pragma: no cover - only when httpx is absent
            _TIMEOUT_EXCEPTIONS = (TimeoutError,)
    return _TIMEOUT_EXCEPTIONS


def _bot_challenge_marker(markdown: str, metadata: object) -> str | None:
    """Return the anti-bot challenge phrase that marks this page as a wall, if any.

    Strong markers are matched against the whole document; weak markers only
    against the title, because a real article may legitimately discuss phrases
    like "enable JavaScript and cookies to continue".
    """
    lowered_body = markdown.lower()
    for marker in _STRONG_BOT_CHALLENGE_MARKERS:
        if marker in lowered_body:
            return marker

    lowered_title = _metadata_title(metadata).lower()
    for marker in _WEAK_BOT_CHALLENGE_TITLE_MARKERS:
        if marker in lowered_title:
            return marker
    return None


def _metadata_title(metadata: object) -> str:
    if isinstance(metadata, dict):
        value = metadata.get("title")
    else:
        value = getattr(metadata, "title", "") if metadata is not None else ""
    return value if isinstance(value, str) else ""


def _is_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    return parsed.scheme.lower() in ("http", "https") and bool(parsed.netloc)


class FirecrawlClient:
    """Minimal Firecrawl wrapper. Inject `raw_client` for offline tests."""

    def __init__(
        self,
        config: FirecrawlConfig,
        *,
        raw_client: object | None = None,
        sleep: object | None = None,
    ) -> None:
        self._config = config
        self._raw_client = raw_client
        self._sleep = sleep if sleep is not None else time.sleep

    @property
    def config(self) -> FirecrawlConfig:
        return self._config

    def export_redacted(self) -> dict:
        """Redacted metadata for provenance (never the API key)."""
        return self._config.redacted()

    # -- public ------------------------------------------------------------- #

    def search(self, query: str, *, limit: int | None = None) -> tuple[FirecrawlResult, ...]:
        """Web-search `query` and return normalized, bounded results.

        Defaults to `config.max_search_results` hits and requests Markdown
        content in the same call (no second scrape). Raises a typed
        `FirecrawlProviderError` on failure after the bounded retry budget.
        """
        limit = limit if limit is not None else self._config.max_search_results
        attempts = self._config.max_retries + 1
        last_error: FirecrawlProviderError | None = None

        for attempt in range(attempts):
            try:
                client = self._get_client()  # inside try: a missing SDK maps, never leaks
                raw = client.search(
                    query=query,
                    sources=["web"],
                    limit=limit,
                    timeout=int(self._config.timeout_seconds * 1000),
                    ignore_invalid_urls=True,
                    scrape_options={"formats": ["markdown"], "only_main_content": True},
                )
                return self._normalize(raw)
            except FirecrawlProviderError as exc:
                raise exc
            except Exception as exc:  # noqa: BLE001 - all SDK/network failures land here
                mapped, retryable = self._classify(exc)
                last_error = mapped
                if retryable and attempt < attempts - 1:
                    self._sleep(0.25 * (2**attempt))
                    continue
                raise mapped from exc

        assert last_error is not None  # pragma: no cover - unreachable
        raise last_error

    def scrape(self, url: str, *, only_main_content: bool = True) -> FirecrawlResult:
        """Scrape one URL and return it as a normalized `FirecrawlResult`.

        This is the expensive tier of retrieval: a real browser-backed fetch used
        only for pages the cheap BeautifulSoup path could not read (blocked,
        script-rendered, or non-HTML). It is deliberately *not* part of
        `search()`'s happy path.

        A page that yields no usable Markdown is reported as a typed
        `FirecrawlMalformedResponseError` rather than an empty successful result,
        so a caller can never mistake "Firecrawl could not read this either" for
        "this page is an empty but valid source".

        `proxy` is pinned to `"basic"`. The default `"auto"` is unusable on a
        self-hosted (open-source) deployment: when the fetch engine sees a
        401/403/429 it throws `AddFeatureError(["stealthProxy"])`, and
        `stealthProxy` is a Firecrawl *cloud* capability the self-hosted build
        cannot provide. The request therefore retries and then fails with HTTP
        500 `SCRAPE_ALL_ENGINES_FAILED`, turning every blocked page into an
        opaque 500 and hiding the real reason. Verified on this stack: with
        `auto`, kroneckerwallis.com and bmitc-bioassets.com both 500; with
        `basic` both come back as HTTP 200 and the block is visible as
        `metadata.statusCode == 403` plus challenge markup, which
        `_normalize_scrape` then rejects by name. Unblocked pages are byte
        identical either way.
        """
        if not _is_http_url(url):
            raise FirecrawlConfigurationError(f"refusing to scrape a non-http URL: {url!r}")

        attempts = self._config.max_retries + 1
        last_error: FirecrawlProviderError | None = None

        for attempt in range(attempts):
            try:
                client = self._get_client()  # inside try: a missing SDK maps, never leaks
                raw = client.scrape(
                    url=url,
                    formats=["markdown"],
                    only_main_content=only_main_content,
                    timeout=int(self._config.timeout_seconds * 1000),
                    proxy="basic",
                )
                return self._normalize_scrape(raw, url)
            except FirecrawlProviderError as exc:
                raise exc
            except Exception as exc:  # noqa: BLE001 - all SDK/network failures land here
                mapped, retryable = self._classify(exc)
                last_error = mapped
                if retryable and attempt < attempts - 1:
                    self._sleep(0.25 * (2**attempt))
                    continue
                raise mapped from exc

        assert last_error is not None  # pragma: no cover - unreachable
        raise last_error

    # -- internals ---------------------------------------------------------- #

    def _get_client(self) -> object:
        if self._raw_client is not None:
            return self._raw_client

        # Lazy import so the SDK is only needed for real calls. In firecrawl>=4.x
        # the supported v2 API (Search -> SearchData.web) lives on
        # `firecrawl.v2.FirecrawlClient`, which also supports keyless mode.
        # The top-level `firecrawl.Firecrawl` class requires an API key on 4.44.0,
        # so it cannot back the documented keyless path.
        from firecrawl.v2 import FirecrawlClient  # lazy: SDK only needed for real calls

        kwargs: dict = {"api_key": self._config.api_key or None}
        if self._config.api_url:
            kwargs["api_url"] = self._config.api_url
        return FirecrawlClient(**kwargs)

    def _normalize(self, raw: object) -> tuple[FirecrawlResult, ...]:
        """Normalize a raw SearchData (or dict) into plain FirecrawlResults.

        Web results live under `.web` in the v2 SDK; we also accept the legacy
        `.data` bucket. A missing bucket is an empty result set (not an
        error); a malformed bucket raises FirecrawlMalformedResponseError;
        individual malformed items are skipped best-effort.
        """
        items: object | None = None
        if isinstance(raw, dict):
            items = raw.get("web")
            if items is None:
                items = raw.get("data")
        else:
            items = getattr(raw, "web", None)
            if items is None:
                items = getattr(raw, "data", None)
        if items is None:
            return ()
        if not isinstance(items, (list, tuple)):
            raise FirecrawlMalformedResponseError("Firecrawl search response did not contain a list of results")

        results: list[FirecrawlResult] = []
        for item in items:
            normalized = self._normalize_item(item)
            if normalized is not None:
                results.append(normalized)
        return tuple(results)

    def _normalize_item(self, item: object) -> FirecrawlResult | None:
        metadata = self._field(item, "metadata", default=None)
        metadata_url = self._metadata_source_url(metadata)
        url = (self._field(item, "url") or "").strip() or metadata_url
        if not url or not _is_http_url(url):
            return None
        return FirecrawlResult(
            url=url,
            title=self._field(item, "title") or self._metadata_field(metadata, "title", "og_title"),
            description=self._field(item, "description")
            or self._metadata_field(metadata, "description", "og_description"),
            markdown=self._field(item, "markdown"),
            source_url=metadata_url,
        )

    def _normalize_scrape(self, raw: object, requested_url: str) -> FirecrawlResult:
        """Normalize a raw scrape Document into one FirecrawlResult.

        The v2 SDK returns a pydantic `Document`; a dict is accepted too so this
        stays testable without the SDK. `requested_url` is authoritative for the
        URL because a scrape is explicitly for one page, and the response's own
        `source_url` may be empty or absent.
        """
        markdown = self._field(raw, "markdown")
        if not isinstance(markdown, str) or not markdown.strip():
            raise FirecrawlMalformedResponseError(
                f"Firecrawl scrape of {requested_url!r} returned no Markdown content"
            )
        metadata = self._field(raw, "metadata", default=None)

        # The API answered 200, but did it actually return the requested page?
        # A 404 body and a Cloudflare challenge wall both arrive as HTTP 200
        # with real-looking Markdown, so both are rejected here by name. Verified
        # on this stack: https://www.nasa.gov/real-mars/ -> HTTP 200,
        # metadata.statusCode 404, title "Page Not Found - NASA", 24988 chars of
        # navigation markup.
        page_status = self._metadata_field(metadata, "statusCode", "status_code")
        if page_status.isdigit() and int(page_status) >= 400:
            raise FirecrawlAPIError(
                f"Firecrawl scrape of {requested_url!r} returned a page with HTTP {page_status}"
                " (the scrape request succeeded; the page itself did not)"
            )
        challenge = _bot_challenge_marker(markdown, metadata)
        if challenge is not None:
            raise FirecrawlAPIError(
                f"Firecrawl scrape of {requested_url!r} hit an anti-bot challenge ({challenge!r})"
                " instead of article content; open-source self-hosted Firecrawl has no"
                " stealth proxy to clear it"
            )

        return FirecrawlResult(
            url=requested_url,
            title=self._field(raw, "title") or self._metadata_field(metadata, "title", "og_title"),
            description=self._field(raw, "description")
            or self._metadata_field(metadata, "description", "og_description"),
            markdown=markdown,
            source_url=self._metadata_source_url(metadata) or requested_url,
        )

    @staticmethod
    def _metadata_source_url(metadata: object) -> str:
        return FirecrawlClient._metadata_field(metadata, "source_url", "sourceURL")
    @staticmethod
    def _metadata_field(metadata: object, *names: str) -> str:
        """Read the first non-empty string across `names` from a metadata dict/obj."""
        for name in names:
            if isinstance(metadata, dict):
                value = metadata.get(name)
            else:
                value = getattr(metadata, name, "") if metadata is not None else ""
            if value:
                return str(value)
        return ""

    @staticmethod
    def _field(item: object, name: str, default: str = "") -> object:
        """Read a field from a dict-ish or object-ish response item."""
        try:
            value = item.get(name, default) if isinstance(item, dict) else getattr(item, name, default)
        except (AttributeError, TypeError, ValueError):
            return default
        if value is None:
            return default
        return value

    def _classify(self, exc: Exception) -> tuple[FirecrawlProviderError, bool]:
        """Classify a raw SDK/network exception into (error, retryable)."""
        code = getattr(exc, "status_code", None)
        if code is None:
            code = getattr(exc, "code", None)
        # Treat the specific "all engines failed" case as non-retryable: the
        # self-hosted stack does not have the cloud-only `stealthProxy` engine,
        # so retrying only burns the budget and returns the same opaque 500.
        if isinstance(code, str):
            if code.lower() == _ALL_ENGINES_FAILED:
                return FirecrawlAPIError(f"Firecrawl API error ({code})"), False

        if isinstance(code, int):
            if code in (401, 403):
                return FirecrawlAuthenticationError(f"Firecrawl rejected the API key (HTTP {code})"), False
            if code in RETRYABLE_STATUSES:
                return FirecrawlAPIError(f"Firecrawl API error (HTTP {code})"), True
            return FirecrawlAPIError(f"Firecrawl API error (HTTP {code})"), False

        if isinstance(exc, ImportError):
            return (
                FirecrawlConfigurationError(
                    "Firecrawl SDK is not installed; add 'firecrawl' to requirements.txt "
                    "to use --retriever firecrawl"
                ),
                False,
            )

        exc_name = type(exc).__name__
        if isinstance(exc, _timeouts()) or isinstance(exc, TimeoutError) or "timeout" in exc_name.lower():
            return FirecrawlTimeoutError(f"Firecrawl request timed out after {self._config.timeout_seconds:.0f}s"), True
        if exc_name in ("RateLimitError",):
            return FirecrawlAPIError("Firecrawl rate limit exceeded"), True
        if exc_name in ("AuthenticationError", "AuthorizationError", "APIKeyError"):
            return FirecrawlAuthenticationError(f"Firecrawl authentication error: {exc_name}"), False

        try:
            from firecrawl.errors import FirecrawlStatusError  # type: ignore[attr-defined]

            if isinstance(exc, FirecrawlStatusError):
                status = getattr(exc, "status_code", None) or getattr(exc, "code", 0)
                status = int(status) if str(status).isdigit() else 0
                retryable = status in RETRYABLE_STATUSES if status else True
                return FirecrawlAPIError(f"Firecrawl API error (HTTP {status})"), retryable
        except ImportError:  # pragma: no cover - firecrawl absent
            pass

        return FirecrawlUnexpectedError(f"Unexpected Firecrawl provider failure ({exc_name})"), False