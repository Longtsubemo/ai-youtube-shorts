"""Gemini provider error hierarchy.

Every failure that can happen inside `providers.gemini` is surfaced as a
`GeminiProviderError` so stage code (which imports no SDK) can handle Gemini
failures uniformly. Messages never contain the API key.
"""

from __future__ import annotations


class GeminiProviderError(Exception):
    """Base class for all Gemini provider failures."""


class GeminiConfigurationError(GeminiProviderError):
    """Missing/invalid configuration (e.g. GEMINI_API_KEY not set)."""


class GeminiAuthenticationError(GeminiProviderError):
    """API key rejected by the provider (HTTP 401/403)."""


class GeminiAPIError(GeminiProviderError):
    """Upstream API failure (non-retryable or after bounded retries)."""


class GeminiTimeoutError(GeminiProviderError):
    """Request timed out after the bounded retry budget."""


class GeminiMalformedResponseError(GeminiProviderError):
    """Response was not parseable / did not contain expected JSON."""


class GeminiSchemaValidationError(GeminiProviderError):
    """Response parsed but did not satisfy the requested schema."""


class GeminiUnexpectedError(GeminiProviderError):
    """Any other unexpected provider/SDK failure."""