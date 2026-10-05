"""Self-hosted (open-source) Firecrawl client for the retrieval experiment.

Scope: isolated. The production pipeline does not import this module, and the
production `providers/firecrawl` adapter is NOT modified or replaced.

What Phase 5 actually established (all verified on this machine)
----------------------------------------------------------------
The open-source Firecrawl stack DOES run locally, but not with its shipped
configuration. Findings, with the evidence for each:

1. **It is 5 containers, not one.** Official compose (fetched 2026-10-01) defines
   `api`, `playwright-service`, `nuq-postgres`, `redis`, `rabbitmq`
   (+ optional FoundationDB). There is no standalone pgvector image — pgvector
   and pg_cron are baked into the custom `nuq-postgres` image.

2. **The documented `docker compose up --build` is not viable here.** Upstream
   ships three services with `build: apps/...` and the `ghcr.io` alternatives
   *commented out*, so following the docs builds three images from source. The
   prebuilt GHCR images exist and pull anonymously, so this compose uses them.

3. **Declared memory exceeds the machine.** Upstream declares `mem_limit` 8G
   (api) + 4G (playwright) = 12 GiB, with `memswap_limit` equal to
   `mem_limit`, so hitting a cap is an OOM kill, not a slowdown. This host's
   Docker VM has 7.6 GiB.

4. **`NUQ_WORKER_COUNT` is the undocumented lever that matters.** Measured: with
   the default of 5, `harness.js --start-docker` spawns 10 Node processes inside
   the single api container and the container was **OOM-killed with exit 137**
   even after the cap was lowered to 3g. The api harness spawns every worker as
   a separate Node process (api, worker, extract-worker, NUQ_WORKER_COUNT
   nuq-workers, prefetch worker, reconciler). Critically, `NUM_WORKERS_PER_QUEUE`
   — the concurrency variable upstream's compose documents — does **not** control
   process count. Setting `NUQ_WORKER_COUNT=1` reduces it to 6 processes and it
   boots cleanly. Measured steady-state after boot: api 1.71 GiB, playwright
   183 MiB, postgres 103 MiB, rabbitmq 158 MiB, redis 6 MiB.

5. **It can reuse the experiment's own SearXNG.** Setting
   `SEARXNG_ENDPOINT=http://host.docker.internal:8080` makes `/v2/search` use
   SearXNG instead of silently falling back to DuckDuckGo scraping. Verified: the
   URLs and ranking returned by self-hosted Firecrawl matched what the Python
   SearXNG client returned for the same query.

6. **The production adapter works against it, unmodified.** With
   `FirecrawlConfig(api_url="http://127.0.0.1:3002", api_key="")`,
   `providers.firecrawl.client.FirecrawlClient.search(...)` returned real results
   with real Markdown, unchanged. Only `api_url` differs. The installed SDK's
   default base URL is `https://api.firecrawl.dev` (`firecrawl/v2/client.py`),
   and the SDK reads **no** base-URL environment variable at all — the repo's own
   `providers/firecrawl/config.py` is what bridges `FIRECRAWL_API_URL` into the
   constructor, so the plumbing is already correct.

7. **It solves the case BeautifulSoup cannot.** Wikipedia answers HTTP 403 to
   the fetcher's honest, identifying User-Agent. The self-hosted stack rendered
   the same URL to 39,579 characters of Markdown via Playwright. That is the
   concrete justification for keeping Firecrawl as the fallback tier.

Known regressions vs the hosted API (from source inspection + upstream issues)
-----------------------------------------------------------------------------
- `ignore_invalid_urls` — which `providers/firecrawl/client.py` always sends —
  is accepted by the schema but never forwarded to `executeSearch`, so it is
  silently dropped self-hosted. Expect more scrape failures than cloud.
- Long-standing upstream issue #1576: self-hosted Playwright underperforms the
  cloud stack against anti-bot-protected sites.
- Cloud-only: Agent/Browser/Interact, screenshots, page actions, the Fire-engine
  anti-bot layer, Developer/Research search categories, caching/proxying, and
  `/v2/team/*` credit endpoints. Not needed for `/v2/search`.

Security note
-------------
This client deliberately NEVER reads `FIRECRAWL_API_KEY` and never sends an
`Authorization` header. The self-hosted stack runs with
`USE_DB_AUTHENTICATION=false`; forwarding a cloud key to an unauthenticated
local server would be both pointless and a credential leak.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import httpx

from experiments.retrieval.errors import RetrievalExperimentError
from experiments.retrieval.searxng_client import host_for

DEFAULT_BASE_URL = "http://127.0.0.1:3002"
# A self-hosted scrape drives a headless Chromium page; the cloud SDK default of
# 30s is not enough for a cold browser pool on a laptop-class VM.
DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_MAX_RESULTS = 5


class SelfHostedFirecrawlError(RetrievalExperimentError):
    """The local self-hosted Firecrawl instance could not serve a request."""


@dataclass(frozen=True)
class ScrapeOutcome:
    """One scrape/search outcome. `ok=False` never carries usable content."""

    url: str
    ok: bool = False
    http_status: int | None = None
    title: str = ""
    domain: str = ""
    markdown: str = ""
    content_length: int = 0
    elapsed_ms: int | None = None
    error: str = ""
    error_kind: str = ""

    @property
    def content(self) -> str:
        return self.markdown


def _text(value: Any) -> str:
    if value is None or isinstance(value, bool):
        return ""
    return value.strip() if isinstance(value, str) else str(value)


def _dig(payload: Any, *keys: str, default: Any = None) -> Any:
    """Fetch a nested value from a decoded JSON object, tolerating absence."""
    current = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return default
        if key not in current:
            return default
        current = current[key]
    return current if current is not None else default


class SelfHostedFirecrawlClient:
    """Minimal HTTP client for a local self-hosted Firecrawl instance.

    Talks to `/v2/search` and `/v2/scrape` directly with httpx. It deliberately
    avoids the `firecrawl` SDK so the experiment's fallback tier is independent
    of the production provider and of any cloud SDK behavior. (`production_adapter_probe`
    below separately verifies the production adapter against this same instance.)
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_results: int = DEFAULT_MAX_RESULTS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._max_results = max_results
        self._transport = transport

    @property
    def base_url(self) -> str:
        return self._base_url

    def is_ready(self) -> bool:
        """True when `/v0/health/readiness` answers 200. Never raises."""
        try:
            with httpx.Client(transport=self._transport, timeout=15.0) as client:
                return client.get(f"{self._base_url}/v0/health/readiness").status_code == 200
        except Exception:  # noqa: BLE001 - a probe reports, never raises
            return False

    def scrape(self, url: str) -> ScrapeOutcome:
        """Scrape one URL to Markdown via the local instance."""
        target = (url or "").strip()
        if not host_for(target):
            return ScrapeOutcome(url=target, ok=False, error="not an absolute http(s) URL", error_kind="invalid_url")
        body: dict[str, Any] = {
            "url": target,
            "formats": ["markdown"],
            "onlyMainContent": True,
        }
        response = self._post("/v2/scrape", body, fallback_url=target)
        if not response.ok and response.failure is not None:
            return response.failure
        return response.items[0] if response.items else ScrapeOutcome(
            url=target, ok=False, error="scrape returned no document", error_kind="empty_content"
        )

    def search(self, query: str, *, limit: int | None = None) -> tuple[ScrapeOutcome, ...]:
        """Run a search and scrape each hit, exactly like the cloud adapter does.

        Requesting `scrapeOptions` in the same call matters: it is what lets the
        local stack return Markdown without a second scrape round-trip.
        """
        text = (query or "").strip()
        if not text:
            raise SelfHostedFirecrawlError("query must be a non-empty string")
        count = limit if limit is not None else self._max_results
        body: dict[str, Any] = {
            "query": text,
            "sources": ["web"],
            "limit": count,
            "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": True},
        }
        outcome = self._post("/v2/search", body)
        if not outcome.ok:
            # Surface the failure as a single failed outcome rather than an
            # empty list, so "the instance was unreachable" can never be read as
            # "the search found nothing".
            assert outcome.failure is not None
            return (outcome.failure,)
        return tuple(item for item in outcome.items if item.url)

    # -- internals ---------------------------------------------------------- #

    def _post(self, path: str, body: Mapping[str, Any], *, fallback_url: str = "") -> "_SearchResponse":
        started = time.perf_counter()
        try:
            with httpx.Client(transport=self._transport, timeout=self._timeout_seconds) as client:
                response = client.post(
                    f"{self._base_url}{path}",
                    json=dict(body),
                    # No Authorization header: USE_DB_AUTHENTICATION=false.
                    headers={"Accept": "application/json"},
                )
        except Exception as exc:  # noqa: BLE001 - every failure becomes an outcome
            kind = "timeout" if isinstance(exc, (httpx.TimeoutException, TimeoutError)) else "connection"
            outcome = ScrapeOutcome(
                url=fallback_url,
                ok=False,
                elapsed_ms=self._elapsed_ms(started),
                error=f"{type(exc).__name__}: {exc}",
                error_kind=kind,
            )
            return _SearchResponse(False, (), outcome)

        elapsed = self._elapsed_ms(started)
        if response.status_code >= 400:
            outcome = ScrapeOutcome(
                url=fallback_url,
                ok=False,
                http_status=response.status_code,
                elapsed_ms=elapsed,
                error=f"HTTP {response.status_code}: {response.text[:200]}",
                error_kind="http_error",
            )
            return _SearchResponse(False, (), outcome)

        try:
            payload = response.json()
        except ValueError as exc:
            outcome = ScrapeOutcome(
                url=fallback_url,
                ok=False,
                http_status=response.status_code,
                elapsed_ms=elapsed,
                error=f"response was not JSON: {exc}",
                error_kind="malformed",
            )
            return _SearchResponse(False, (), outcome)

        return self._parse(payload, fallback_url, response.status_code, elapsed)

    def _parse(self, payload: Any, fallback_url: str, status: int, elapsed: int) -> "_SearchResponse":
        success = _dig(payload, "success", default=True)
        errors = _dig(payload, "error")
        if success is False:
            message = _text(_dig(errors, "message")) or _text(errors) or "instance reported success=false"
            return _SearchResponse(
                False,
                (),
                ScrapeOutcome(
                    url=fallback_url,
                    ok=False,
                    http_status=status,
                    elapsed_ms=elapsed,
                    error=message,
                    error_kind="api_error",
                ),
            )

        web = _dig(payload, "data", "web", default=None)
        if web is None:
            web = _dig(payload, "data", default=None)

        if isinstance(web, list):
            items = tuple(
                ScrapeOutcome(
                    url=_text(_dig(item, "url")),
                    ok=True,
                    http_status=status,
                    title=_text(_dig(item, "title")) or _text(_dig(item, "metadata", "title")),
                    domain=host_for(_text(_dig(item, "url"))),
                    markdown=_text(_dig(item, "markdown")),
                    content_length=len(_text(_dig(item, "markdown"))),
                    elapsed_ms=elapsed,
                )
                for item in web
                if isinstance(item, Mapping)
            )
            return _SearchResponse(True, items, None)

        # Single-document shape, i.e. /v2/scrape.
        url = _text(_dig(payload, "data", "sourceURL")) or _text(_dig(payload, "data", "url")) or fallback_url
        markdown = _text(_dig(payload, "data", "markdown"))
        title = _text(_dig(payload, "data", "metadata", "title"))
        if not url and not markdown:
            return _SearchResponse(
                False,
                (),
                ScrapeOutcome(
                    url=fallback_url,
                    ok=False,
                    http_status=status,
                    elapsed_ms=elapsed,
                    error="response had neither a web list nor a document body",
                    error_kind="malformed",
                ),
            )
        outcome = ScrapeOutcome(
            url=url,
            ok=bool(markdown),
            http_status=status,
            title=title,
            domain=host_for(url),
            markdown=markdown,
            content_length=len(markdown),
            elapsed_ms=elapsed,
            error="" if markdown else "instance returned no markdown",
            error_kind="" if markdown else "empty_content",
        )
        return _SearchResponse(outcome.ok, (outcome,), None)

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return int((time.perf_counter() - started) * 1000)


@dataclass(frozen=True)
class _SearchResponse:
    """Internal envelope so one parser serves both endpoints."""

    ok: bool
    items: tuple[ScrapeOutcome, ...]
    failure: ScrapeOutcome | None = None


def production_adapter_probe(
    base_url: str = DEFAULT_BASE_URL,
    *,
    query: str = "what is the triple point of water",
    limit: int = 2,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Verify the UNMODIFIED production Firecrawl adapter against a local instance.

    This is the integration question that matters: if the production adapter
    already speaks to self-hosted Firecrawl, adopting it later is a config
    change (`FIRECRAWL_API_URL`) rather than a rewrite.

    Uses `providers.firecrawl.client.FirecrawlClient` exactly as the pipeline
    does, but with `api_key=""` (keyless) and `api_url` pointed at localhost.
    No cloud request is possible: no API key is supplied and the base URL is
    local.
    """
    try:
        from providers.firecrawl.client import FirecrawlClient
        from providers.firecrawl.config import FirecrawlConfig
    except ImportError as exc:  # pragma: no cover - repo package is present
        return {"checked": False, "reason": f"production adapter not importable: {exc}"}

    config = FirecrawlConfig(
        api_key="",
        api_url=base_url,
        timeout_seconds=timeout_seconds,
        max_search_results=limit,
    )
    try:
        results = FirecrawlClient(config).search(query, limit=limit)
    except Exception as exc:  # noqa: BLE001 - a probe reports, never raises
        return {"checked": True, "ok": False, "reason": f"{type(exc).__name__}: {exc}"}

    return {
        "checked": True,
        "ok": bool(results),
        "results_returned": len(results),
        "urls": [item.url for item in results],
        "markdown_chars": [len(item.markdown) for item in results],
        "note": "production providers/firecrawl client used unmodified; only api_url differs",
    }


__all__: Sequence[str] = (
    "DEFAULT_BASE_URL",
    "DEFAULT_MAX_RESULTS",
    "DEFAULT_TIMEOUT_SECONDS",
    "ScrapeOutcome",
    "SelfHostedFirecrawlClient",
    "SelfHostedFirecrawlError",
    "production_adapter_probe",
)