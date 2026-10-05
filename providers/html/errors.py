"""Typed error hierarchy for the BeautifulSoup HTML page fetcher.

Kept separate from `providers/searxng/errors.py` so each provider owns one error
family, mirroring `providers/firecrawl/errors.py`.
"""

from __future__ import annotations


class HtmlFetcherError(Exception):
    """Base class for HTML fetching/extraction failures."""


class FetcherConfigurationError(HtmlFetcherError):
    """The fetcher was configured with unusable values."""


class ParserError(HtmlFetcherError):
    """The readable-text extractor was given something it cannot parse at all."""


__all__ = ("FetcherConfigurationError", "HtmlFetcherError", "ParserError")
