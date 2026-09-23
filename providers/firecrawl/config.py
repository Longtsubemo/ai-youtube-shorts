"""Firecrawl configuration loaded from the environment.

FIRECRAWL_API_KEY   optional — Firecrawl works on a keyless free tier for
                    search/scrape/interact (rate-limited per IP). Set it for
                    higher rate limits.
FIRECRAWL_API_URL   optional — API root (defaults to Firecrawl's endpoint).

Secrets discipline mirrors providers/gemini/config.py: the key is never
printed, logged, or serialized. `__repr__` and `redacted()` hide it, and
exceptions built here never interpolate the key value.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from providers.firecrawl.errors import FirecrawlConfigurationError

ENV_API_KEY = "FIRECRAWL_API_KEY"
ENV_API_URL = "FIRECRAWL_API_URL"

DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_SEARCH_RESULTS = 5


@dataclass(frozen=True)
class FirecrawlConfig:
    """Immutable retrieval provider settings.

    The API key is optional: keyless mode is a first-class, supported path for
    the search endpoint we use. `api_key` is never stringified by `repr`,
    `str`, or `redacted()`.
    """

    api_key: str = ""
    api_url: str = ""
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    max_search_results: int = DEFAULT_MAX_SEARCH_RESULTS

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise FirecrawlConfigurationError("max_retries must be >= 0")
        if self.timeout_seconds <= 0:
            raise FirecrawlConfigurationError("timeout_seconds must be > 0")
        if self.max_search_results < 1:
            raise FirecrawlConfigurationError("max_search_results must be >= 1")
        object.__setattr__(self, "api_key", self.api_key.strip())
        object.__setattr__(self, "api_url", self.api_url.strip())

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "FirecrawlConfig":
        """Build config from an env mapping (defaults to os.environ).

        Keyless mode is supported: a missing FIRECRAWL_API_KEY is NOT an
        error (the free tier allows search without a key). Construction is
        therefore safe even when no key is configured.
        """
        env = os.environ if env is None else env
        return cls(
            api_key=env.get(ENV_API_KEY) or "",
            api_url=env.get(ENV_API_URL) or "",
        )

    def __repr__(self) -> str:
        return (
            f"FirecrawlConfig(timeout_seconds={self.timeout_seconds}, "
            f"max_retries={self.max_retries}, "
            f"max_search_results={self.max_search_results}, "
            f"keyless={not self.api_key}, api_key=<redacted>)"
        )

    def redacted(self) -> dict:
        """Provider-safe provenance/metadata (never contains the key)."""
        return {
            "provider": "firecrawl",
            "keyless": not bool(self.api_key),
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "max_search_results": self.max_search_results,
        }