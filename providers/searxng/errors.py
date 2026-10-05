"""Typed error hierarchy for the local SearXNG discovery client.

Mirrors the shape of `providers/firecrawl/errors.py` (a base error plus typed
subclasses) so failures cross the Stage-2 retriever boundary as one typed family
rather than a raw `httpx` exception.

The most important member here is `SearxngFormatDisabledError`. SearXNG ships
with the JSON API disabled (`search.formats: [html]`), and the failure it
produces when JSON is requested anyway is an HTTP 403 with an HTML body. Without
a dedicated error, that looks like "SearXNG rejected us" instead of "add `json`
to search.formats in the settings file and restart the container".
"""

from __future__ import annotations


class SearxngError(Exception):
    """Base class for local SearXNG client failures."""


class SearxngConfigurationError(SearxngError):
    """The client was configured with unusable values (e.g. a non-http base URL)."""


class SearxngConnectionError(SearxngError):
    """The local SearXNG instance was not reachable (not running / wrong port)."""


class SearxngTimeoutError(SearxngError):
    """The local SearXNG instance did not answer within the bounded budget."""


class SearxngAPIError(SearxngError):
    """SearXNG answered with an unexpected HTTP status.

    Carries `status` and `retryable` so the client can apply a bounded retry
    policy without re-inspecting the exception type.
    """

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class SearxngFormatDisabledError(SearxngAPIError):
    """SearXNG refused the JSON format because it is not enabled.

    Raised on HTTP 403 when asking for `format=json`.
    """


class SearxngMalformedResponseError(SearxngError):
    """The response was not JSON, or was JSON without a usable `results` list."""


__all__ = (
    "SearxngAPIError",
    "SearxngConfigurationError",
    "SearxngConnectionError",
    "SearxngError",
    "SearxngFormatDisabledError",
    "SearxngMalformedResponseError",
    "SearxngTimeoutError",
)
