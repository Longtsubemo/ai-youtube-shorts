"""HTML page fetching provider (the cheap tier of Stage-2 retrieval).

`HtmlFetcher` is the cheap path for ordinary, static, non-hostile pages: a plain
`httpx` GET followed by offline BeautifulSoup extraction. It is completely honest
about everything else — a blocked page, a 403, a timeout, or a PDF served with a
200 all come back with `ok=False` and a typed `error_kind`, never as "content".

That honesty is what lets the caller route only genuinely-difficult pages to the
expensive self-hosted Firecrawl tier, rather than sending every URL to Firecrawl
or quietly counting blocked pages as usable sources.

`extract_document()` is pure and offline; only `HtmlFetcher` touches the network,
and its transport is injectable for tests.
"""

from providers.html.errors import FetcherConfigurationError, HtmlFetcherError, ParserError
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
    "HtmlFetcher",
    "ParsedDocument",
    "extract_document",
    "is_scriptish",
    "looks_blocked",
    "summarize",
    "FetcherConfigurationError",
    "HtmlFetcherError",
    "ParserError",
]
