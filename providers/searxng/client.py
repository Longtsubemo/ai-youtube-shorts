"""SearchResult + SearxngClient — local, self-hosted SearXNG discovery.

What it does
------------
Calls a LOCAL/self-hosted SearXNG instance over HTTP with `format=json`,
normalizes each hit into a plain frozen `SearchResult`, and deduplicates by
canonical URL.

Design rules this module holds itself to
-----------------------------------------
1. **Nothing is fabricated.** A field SearXNG does not supply stays empty (or
   `None` for `score`). Absent metadata is never guessed, defaulted to a
   plausible-looking value, or back-filled from another source.
2. **URL identity comes from the repository, not from here.** Dedup and
   `normalized_url` both call `agents.research.models.normalize_source_url()`,
   the single source of truth already used by the Stage-2 agent and the Stage-3
   evidence provider. This module deliberately defines no URL logic of its own.
3. **Search results stay discovery-tier.** A `SearchResult` is a *search hit*
   (title + short snippet). It is not a document, and nothing here pretends
   otherwise — turning one into evidence is the retrieval adapter's job, and
   even there the snippet is not passed off as page content.
4. **Failures are typed and visible.** No raw `httpx` exception escapes, and a
   malformed payload is a typed error rather than an empty result list that
   looks like "no results found".

Offline testability
-------------------
The parsing and dedup functions are module-level and pure, so the whole
normalization path is testable with zero network. `SearxngClient` accepts an
injected `httpx` transport, so HTTP behaviour is testable without a container.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

import httpx

from agents.research.models import normalize_source_url
from providers.searxng.errors import (
    SearxngAPIError,
    SearxngConfigurationError,
    SearxngConnectionError,
    SearxngError,
    SearxngFormatDisabledError,
    SearxngMalformedResponseError,
    SearxngTimeoutError,
)

# Matches the port published by experiments/retrieval/searxng/docker-compose.yml.
DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_TIMEOUT_SECONDS = 45.0
DEFAULT_MAX_RESULTS = 20
DEFAULT_MAX_RETRIES = 1

# Categories asked of SearXNG, NOT a list of trusted sources.
#
# `general` only. That is where the working web-search engines live (see the
# `engines` block in experiments/retrieval/searxng/settings.yml for why the
# packaged default set has to be extended).
#
# Adding `science` was measured against the four probe queries and made domain
# diversity WORSE on all of them (12/16/16/13 domains with `general` alone vs
# 11/15/15/10 with `general,science`). The academic engines return topically
# loose arxiv/europepmc papers that crowd out the pages actually about the
# topic, which then starve Stage 2 extraction. The scientific engines stay
# enabled in settings.yml and remain reachable via `SEARXNG_CATEGORIES` for a
# caller that genuinely wants papers.
#
# This is a DISCOVERY request only. Which results may later act as evidence is
# decided by the Stage 3 gate, never by the category asked here; asking for a
# category does not promote anything.
DEFAULT_CATEGORIES = "general"

RETRYABLE_STATUSES = (408, 429, 500, 502, 503, 504)

# SearXNG's own outbox is `request_timeout: 10.0` in our settings.yml; anything
# much beyond that means engines are hanging rather than answering.
_TIMEOUT_EXCEPTIONS: tuple[type[BaseException], ...] = ()


@dataclass(frozen=True)
class SearchResult:
    """One normalized SearXNG hit.

    `url` is the URL exactly as SearXNG reported it. `normalized_url` is that
    URL run through the repository's `normalize_source_url()`, and is `""` when
    the URL is not an absolute http(s) URL (such results are dropped).

    `score` is `float | None` rather than `float` on purpose: SearXNG omits the
    field on some engines, and defaulting it to `0.0` would invent a ranking
    signal that was never reported.

    `published_date_raw` is deliberately NOT named `publication_date`. Engines
    return free text there ("3 days ago", "Jan 2024"), and passing that straight
    into the `RetrievedSource.publication_date` field would launder non-ISO junk
    into a typed contract field. The adapter ISO-validates it before use; the
    raw value is kept here so nothing is lost.
    """

    url: str
    normalized_url: str = ""
    title: str = ""
    snippet: str = ""
    engine: str = ""
    engines: tuple[str, ...] = ()
    category: str = ""
    score: float | None = None
    positions: tuple[int, ...] = ()
    published_date_raw: str = ""
    author: str = ""


# --------------------------------------------------------------------------- #
# Pure coercion helpers. These never guess: a value of an unexpected type is
# refused ("" / None) instead of being stringified into something plausible.
# --------------------------------------------------------------------------- #


def _as_text(value: Any) -> str:
    """Coerce only genuine scalars to text; refuse lists/dicts/objects."""
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    return ""


def _as_optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _as_int_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    out: list[int] = []
    for item in value:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            out.append(item)
        elif isinstance(item, str) and item.strip().lstrip("-").isdigit():
            out.append(int(item.strip()))
    return tuple(out)


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if not isinstance(value, (list, tuple)):
        return ()
    out: list[str] = []
    for item in value:
        text = _as_text(item)
        if text and text not in out:
            out.append(text)
    return tuple(out)


def host_for(url: str) -> str:
    """Lowercased host of an absolute http(s) URL, else ''.

    Mirrors `providers/firecrawl/retriever.py:_host()` (netloc lowercased, port
    retained) so domain values stay consistent with the existing retriever.
    """
    try:
        parsed = urlparse(url)
    except ValueError:  # pragma: no cover - urlparse is extremely tolerant
        return ""
    if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
        return ""
    return parsed.netloc.lower()


# --------------------------------------------------------------------------- #
# Pure normalization / dedup (no network, directly unit-testable)
# --------------------------------------------------------------------------- #


def normalize_result(item: Any) -> SearchResult | None:
    """Normalize one raw SearXNG result object, or None if it is unusable.

    A result is unusable when it has no `url`, or when that URL is not an
    absolute http(s) URL. Such entries are skipped rather than repaired.
    """
    if not isinstance(item, Mapping):
        return None

    url = _as_text(item.get("url"))
    if not url:
        return None
    normalized = normalize_source_url(url)
    if not normalized:
        return None

    engines = list(_as_str_tuple(item.get("engines")))
    single = _as_text(item.get("engine"))
    if single and single not in engines:
        engines.append(single)

    # Engines report dates under several names, none of them reliably ISO.
    published_raw = _as_text(item.get("publishedDate")) or _as_text(item.get("pubdate"))

    return SearchResult(
        url=url,
        normalized_url=normalized,
        title=_as_text(item.get("title")),
        snippet=_as_text(item.get("content")),
        engine=single,
        engines=tuple(engines),
        category=_as_text(item.get("category")),
        score=_as_optional_float(item.get("score")),
        positions=_as_int_tuple(item.get("positions")),
        published_date_raw=published_raw,
        author=_as_text(item.get("author")),
    )


def parse_results(payload: Any) -> tuple[SearchResult, ...]:
    """Normalize a decoded SearXNG JSON payload into deduplicated results.

    - payload not a mapping                  -> SearxngMalformedResponseError
    - `results` absent                       -> treated as zero results
    - `results` present but not a list       -> SearxngMalformedResponseError
    - individual malformed entries           -> skipped (best effort)

    Dedup key is `normalize_source_url()`, which lowercases scheme+host and
    strips trailing slashes *and* the query string. That means
    `https://a.test/p?utm_source=x` and `https://a.test/p` collapse to one
    result, which is the behaviour we want for provenance: the same page found
    twice through two different tracking links is one source, not two.

    Ranking order from SearXNG is preserved: the first occurrence of a canonical
    URL wins, and later duplicates contribute only their engine names (a URL
    genuinely returned by several engines is recorded as such — this mirrors what
    SearXNG itself does in its own `engines` field).
    """
    if not isinstance(payload, Mapping):
        raise SearxngMalformedResponseError(
            f"SearXNG response was {type(payload).__name__}, expected a JSON object"
        )

    raw = payload.get("results")
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise SearxngMalformedResponseError(
            f"SearXNG 'results' was {type(raw).__name__}, expected a list"
        )

    order: list[str] = []
    by_key: dict[str, SearchResult] = {}
    for item in raw:
        result = normalize_result(item)
        if result is None:
            continue
        key = result.normalized_url
        existing = by_key.get(key)
        if existing is None:
            order.append(key)
            by_key[key] = result
            continue
        merged = tuple(existing.engines) + tuple(r for r in result.engines if r not in existing.engines)
        by_key[key] = replace(existing, engines=merged)

    return tuple(by_key[key] for key in order)


def dedupe_results(results: Iterable[SearchResult]) -> tuple[SearchResult, ...]:
    """Deduplicate already-normalized results by canonical URL (first wins)."""
    order: list[str] = []
    by_key: dict[str, SearchResult] = {}
    for result in results:
        key = result.normalized_url or result.url
        if key in by_key:
            continue
        order.append(key)
        by_key[key] = result
    return tuple(by_key[key] for key in order)


def _unresponsive_entry(item: Any) -> str:
    """Render one `unresponsive_engines` entry as `"engine : reason"`.

    SearXNG reports these in two shapes: a plain string (`"brave : CAPTCHA"`)
    and a `[engine, reason]` pair. Both must be handled, because returning
    nothing for the pair shape would silently hide the fact that a query ran
    with several engines suspended.
    """
    if isinstance(item, (list, tuple)) and len(item) == 2:
        engine = _as_text(item[0])
        reason = _as_text(item[1])
        if engine and reason:
            return f"{engine} : {reason}"
        return engine or reason
    return _as_text(item)


def unresponsive_engines(payload: Any) -> tuple[str, ...]:
    """Engines SearXNG reported as failed/captcha'd/suspended for this query.

    This is a real signal, not noise: a query that returns 20 results while
    4 engines are suspended has a narrower evidence base than it appears to.
    Surfaced separately so the comparison report can show it instead of
    silently implying full coverage.
    """
    if not isinstance(payload, Mapping):
        return ()
    raw = payload.get("unresponsive_engines")
    if raw is None or isinstance(raw, bool) or isinstance(raw, (int, float)):
        return ()
    if isinstance(raw, str):
        text = raw.strip()
        return (text,) if text else ()
    if not isinstance(raw, (list, tuple)):
        return ()
    out: list[str] = []
    for item in raw:
        text = _unresponsive_entry(item)
        if text and text not in out:
            out.append(text)
    return tuple(out)


def _timeouts() -> tuple[type[BaseException], ...]:
    global _TIMEOUT_EXCEPTIONS
    if not _TIMEOUT_EXCEPTIONS:
        _TIMEOUT_EXCEPTIONS = (
            httpx.TimeoutException,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.PoolTimeout,
            httpx.WriteTimeout,
        )
    return _TIMEOUT_EXCEPTIONS


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


class SearxngClient:
    """Minimal HTTP client for a local, self-hosted SearXNG instance.

    The HTTP transport is injectable so that HTTP behaviour (timeouts, 403,
    non-JSON bodies, 5xx) can be tested offline without a running container.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_results: int = DEFAULT_MAX_RESULTS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        categories: str = DEFAULT_CATEGORIES,
        language: str = "en",
        user_agent: str = "ai-youtube-channel-retrieval-experiment/0.1 (local SearXNG client)",
        transport: httpx.BaseTransport | None = None,
        sleep: Any = None,
    ) -> None:
        parsed = urlparse(base_url.strip())
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise SearxngConfigurationError(f"base_url must be an absolute http(s) URL, got {base_url!r}")
        if timeout_seconds <= 0:
            raise SearxngConfigurationError("timeout_seconds must be > 0")
        if max_results < 1:
            raise SearxngConfigurationError("max_results must be >= 1")
        if max_retries < 0:
            raise SearxngConfigurationError("max_retries must be >= 0")

        self._base_url = base_url.strip().rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._max_results = max_results
        self._max_retries = max_retries
        self._categories = categories
        self._language = language
        self._user_agent = user_agent
        self._transport = transport
        self._sleep = sleep if sleep is not None else time.sleep

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def search_url(self) -> str:
        return f"{self._base_url}/search"

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_seconds

    @property
    def max_results(self) -> int:
        return self._max_results

    def search(self, query: str, *, max_results: int | None = None) -> tuple[SearchResult, ...]:
        """Run `query` and return at most `max_results` deduplicated hits.

        Note on bounds: SearXNG has no server-side result cap, so the limit is
        applied client-side *after* normalization and dedup. That means we always
        transfer more than we keep — the tradeoff is deliberate, since trimming
        server-side would change which results rank highest.
        """
        text = (query or "").strip()
        if not text:
            raise SearxngConfigurationError("query must be a non-empty string")

        limit = max_results if max_results is not None else self._max_results
        if limit < 1:
            raise SearxngConfigurationError("max_results must be >= 1")

        payload = self._request_payload(text)
        results = parse_results(payload)
        return dedupe_results(results)[:limit]

    def search_with_diagnostics(self, query: str, *, max_results: int | None = None) -> tuple[tuple[SearchResult, ...], tuple[str, ...]]:
        """Same as `search`, but also returns the unresponsive-engine list."""
        text = (query or "").strip()
        if not text:
            raise SearxngConfigurationError("query must be a non-empty string")
        payload = self._request_payload(text)
        limit = max_results if max_results is not None else self._max_results
        return dedupe_results(parse_results(payload))[:limit], unresponsive_engines(payload)

    def probe(self) -> bool:
        """True when the instance answers `/healthz` with HTTP 200."""
        try:
            with httpx.Client(
                transport=self._transport, timeout=min(self._timeout_seconds, 10.0), follow_redirects=True
            ) as client:
                response = client.get(f"{self._base_url}/healthz", headers={"User-Agent": self._user_agent})
            return response.status_code == 200
        except Exception:  # noqa: BLE001 - a probe never raises; it reports False
            return False

    # -- internals ---------------------------------------------------------- #

    def _request_payload(self, query: str) -> Mapping[str, Any]:
        params: dict[str, str] = {
            "q": query,
            "format": "json",
            "categories": self._categories,
            "language": self._language,
        }
        attempts = self._max_retries + 1
        last: Exception | None = None

        for attempt in range(attempts):
            try:
                with httpx.Client(
                    transport=self._transport,
                    timeout=self._timeout_seconds,
                    follow_redirects=True,
                ) as client:
                    response = client.get(
                        self.search_url,
                        params=params,
                        headers={"User-Agent": self._user_agent, "Accept": "application/json"},
                    )
                return self._decode(response)
            except Exception as exc:  # noqa: BLE001 - everything is mapped below
                mapped, retryable = self._classify(exc)
                last = mapped
                if retryable and attempt < attempts - 1:
                    self._sleep(0.25 * (2**attempt))
                    continue
                raise mapped from exc

        assert last is not None  # pragma: no cover - unreachable
        raise last

    def _decode(self, response: httpx.Response) -> Mapping[str, Any]:
        status = response.status_code

        if status == 403:
            # SearXNG answers a disallowed `format=json` with 403 + HTML. This is
            # a configuration problem, not a rejection, so it gets its own error.
            raise SearxngFormatDisabledError(
                "SearXNG returned HTTP 403 for format=json. Add 'json' to "
                "search.formats in experiments/retrieval/searxng/settings.yml and "
                "restart the container."
            )

        if status in (401, 407):
            raise SearxngAPIError(f"SearXNG returned HTTP {status}", status=status, retryable=False)
        if status >= 400:
            raise SearxngAPIError(
                f"SearXNG returned HTTP {status}",
                status=status,
                retryable=status in RETRYABLE_STATUSES,
            )

        content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
        if content_type and not (content_type == "application/json" or content_type.endswith("+json")):
            raise SearxngMalformedResponseError(
                f"expected a JSON response but SearXNG sent content-type {content_type!r}. "
                "The JSON format is probably not enabled."
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise SearxngMalformedResponseError(f"SearXNG response was not valid JSON: {exc}") from exc

        if not isinstance(payload, Mapping):
            raise SearxngMalformedResponseError(
                f"SearXNG JSON was {type(payload).__name__}, expected an object"
            )
        return payload

    def _classify(self, exc: Exception) -> tuple[Exception, bool]:
        if isinstance(exc, SearxngError):
            # Already a typed experiment error; honour the retryable flag it carries.
            return exc, bool(getattr(exc, "retryable", False))
        if isinstance(exc, _timeouts()):
            return SearxngTimeoutError(
                f"SearXNG did not answer within {self._timeout_seconds:.0f}s at {self._base_url}"
            ), True
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            return SearxngAPIError(f"SearXNG returned HTTP {status}", status=status), status in RETRYABLE_STATUSES
        if isinstance(exc, (httpx.ConnectError, httpx.NetworkError, httpx.RemoteProtocolError)):
            return SearxngConnectionError(
                f"could not reach SearXNG at {self._base_url} ({type(exc).__name__}). "
                "Is the container running? "
                "docker compose -f experiments/retrieval/searxng/docker-compose.yml up -d"
            ), True
        return SearxngAPIError(f"unexpected SearXNG failure ({type(exc).__name__}: {exc})"), False


def client_from_env(env: Mapping[str, str] | None = None) -> SearxngClient:
    """Build a client from env vars, falling back to the compose defaults.

    Reads ONLY `SEARXNG_*` variables and never touches `.env` or any other
    provider's configuration, so retrieval search cannot perturb the rest of the
    pipeline's provider settings.
    """
    import os

    source = os.environ if env is None else env
    return SearxngClient(
        base_url=source.get("SEARXNG_URL") or DEFAULT_BASE_URL,
        timeout_seconds=float(source.get("SEARXNG_TIMEOUT") or DEFAULT_TIMEOUT_SECONDS),
        max_results=int(source.get("SEARXNG_MAX_RESULTS") or DEFAULT_MAX_RESULTS),
        categories=source.get("SEARXNG_CATEGORIES") or DEFAULT_CATEGORIES,
        language=source.get("SEARXNG_LANGUAGE") or "en",
    )


__all__: Sequence[str] = (
    "DEFAULT_BASE_URL",
    "DEFAULT_CATEGORIES",
    "DEFAULT_MAX_RESULTS",
    "DEFAULT_TIMEOUT_SECONDS",
    "SearxngClient",
    "SearchResult",
    "client_from_env",
    "dedupe_results",
    "host_for",
    "normalize_result",
    "parse_results",
    "unresponsive_engines",
)