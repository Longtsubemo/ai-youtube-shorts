"""Re-export shim: the HTML fetcher now lives in `providers.html`.

The implementation was promoted to production because the production Stage-2
retriever now uses it as its cheap fetch tier. This module remains so the
experiment's comparison harness, CLI, and original test files keep importing and
running unchanged, and so there is exactly one implementation rather than two
copies that can drift.
"""

from providers.html.errors import FetcherConfigurationError, ParserError
from providers.html.fetcher import (
    BLOCKING_STATUSES,
    DEFAULT_MAX_CONTENT_CHARS,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_USER_AGENT,
    FetchedPage,
    HtmlFetcher,
    ParsedDocument,
    extract_document,
    is_scriptish,
    looks_blocked,
    summarize,
)

__all__ = [
    "BLOCKING_STATUSES",
    "DEFAULT_MAX_CONTENT_CHARS",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_USER_AGENT",
    "FetchedPage",
    "FetcherConfigurationError",
    "HtmlFetcher",
    "ParsedDocument",
    "ParserError",
    "extract_document",
    "is_scriptish",
    "looks_blocked",
    "summarize",
]
