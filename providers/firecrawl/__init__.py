"""Firecrawl retrieval provider (Stage-2 real web retrieval).

Exposes a bounded, keyless-capable retriever over the Firecrawl Search API.
The Firecrawl SDK is imported lazily by `FirecrawlClient`, so importing this
package never forces the SDK or a network call (requirements.txt marks
`firecrawl` as an optional runtime dependency, like `google-genai`).
"""

from providers.firecrawl.client import FirecrawlClient, FirecrawlResult
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
from providers.firecrawl.retriever import FirecrawlResearchRetriever

__all__ = [
    "FirecrawlClient",
    "FirecrawlResult",
    "FirecrawlConfig",
    "FirecrawlResearchRetriever",
    "FirecrawlProviderError",
    "FirecrawlConfigurationError",
    "FirecrawlAuthenticationError",
    "FirecrawlAPIError",
    "FirecrawlTimeoutError",
    "FirecrawlMalformedResponseError",
    "FirecrawlUnexpectedError",
]