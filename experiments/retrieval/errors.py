"""Error shim: SearXNG and HTML errors now live in their production providers.

Kept so `experiments.retrieval.firecrawl_selfhost` (still experiment-only) can
keep a base error to derive from, and so the original experiment tests importing
`FetcherConfigurationError` / `Searxng*Error` from here keep working.
"""

from providers.html.errors import FetcherConfigurationError, HtmlFetcherError, ParserError
from providers.searxng.errors import (
    SearxngAPIError,
    SearxngConfigurationError,
    SearxngConnectionError,
    SearxngError,
    SearxngFormatDisabledError,
    SearxngMalformedResponseError,
    SearxngTimeoutError,
)


class RetrievalExperimentError(Exception):
    """Base class for failures raised by experiment-only modules.

    Only `firecrawl_selfhost` still derives from this; the SearXNG and HTML
    clients now use their own production error families.
    """


__all__ = [
    "RetrievalExperimentError",
    "FetcherConfigurationError",
    "HtmlFetcherError",
    "ParserError",
    "SearxngAPIError",
    "SearxngConfigurationError",
    "SearxngConnectionError",
    "SearxngError",
    "SearxngFormatDisabledError",
    "SearxngMalformedResponseError",
    "SearxngTimeoutError",
]
