"""FirecrawlClient — thin wrapper over the official `firecrawl` SDK (v2 Search).

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