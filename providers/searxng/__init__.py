"""SearXNG discovery provider (Stage-2 web search discovery).

Provides a bounded client for a *self-hosted* SearXNG instance and normalizes each hit into a plain frozen `SearchResult`. Search results are discovery-tier: they carry a title and short snippet, never page content. Nothing in this package fabricates metadata a result did not supply, and URL identity/deduplication is delegated to the repository's single source of truth, `agents.research.models.normalize_source_url()`. The HTTP transport is injectable, so the whole normalization path is testable offline.
"""

from providers.searxng.client import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_RESULTS,
    DEFAULT_TIMEOUT_SECONDS,
    SearxngClient,
    SearchResult,
    client_from_env,
    dedupe_results,
    host_for,
    normalize_result,
    parse_results,
    unresponsive_engines,
)
from providers.searxng.errors import (
    SearxngAPIError,
    SearxngConfigurationError,
    SearxngConnectionError,
    SearxngError,
    SearxngFormatDisabledError,
    SearxngMalformedResponseError,
    SearxngTimeoutError,
)

__all__ = [
    "DEFAULT_BASE_URL",
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
    "SearxngError",
    "SearxngConfigurationError",
    "SearxngConnectionError",
    "SearxngTimeoutError",
    "SearxngAPIError",
    "SearxngFormatDisabledError",
    "SearxngMalformedResponseError",
]
