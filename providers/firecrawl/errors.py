"""Firecrawl retrieval provider error hierarchy.

Every failure that can happen inside `providers.firecrawl` is surfaced as a
`FirecrawlProviderError` so stage code (which imports no SDK) can handle
Firecrawl failures uniformly. Messages never contain the API key.
"""

from __future__ import annotations


class FirecrawlProviderError(Exception):
    """Base class for all Firecrawl retrieval provider failures."""


class FirecrawlConfigurationError(FirecrawlProviderError):
    """Missing/invalid configuration (e.g. SDK not installed)."""


class FirecrawlAuthenticationError(FirecrawlProviderError):
    """API key rejected by the provider (HTTP 401/403)."""


class FirecrawlAPIError(FirecrawlProviderError):
    """Upstream API failure (non-retryable or after bounded retries)."""


class FirecrawlTimeoutError(FirecrawlProviderError):
    """Request timed out after the bounded retry budget."""


class FirecrawlMalformedResponseError(FirecrawlProviderError):
    """Response was not parseable / did not contain the expected structure."""


class FirecrawlUnexpectedError(FirecrawlProviderError):
    """Any other unexpected provider/SDK failure."""