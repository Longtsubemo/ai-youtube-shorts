"""Re-export shim: the SearXNG client now lives in `providers.searxng`.

The implementation was promoted to production because the production Stage-2
retriever now uses it. This module remains so the experiment's CLI, comparison
harness, and original test files keep importing and running unchanged, and so
there is exactly one implementation rather than two copies that can drift.
"""

from providers.searxng.client import (
    DEFAULT_BASE_URL,
    DEFAULT_CATEGORIES,
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

__all__ = [
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
]
