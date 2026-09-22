"""Gemini configuration loaded from the environment.

GEMINI_API_KEY   required — the Google AI Studio API key.
GEMINI_MODEL     optional — model id (default: a current Gemini Flash-Lite).

Secrets discipline: the key is read from the environment (or a provided env
mapping) and never printed, logged, or serialized. `__repr__` and `to_dict`
are redacted; exceptions built here never interpolate the key value.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from providers.gemini.errors import GeminiConfigurationError

ENV_API_KEY = "GEMINI_API_KEY"
ENV_MODEL = "GEMINI_MODEL"

DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_OUTPUT_TOKENS = 4096


def _non_empty(value: str | None, name: str) -> str:
    if value is None or not value.strip():
        raise GeminiConfigurationError(
            f"{name} is not set. Run with a real provider only after configuring it "
            f"(see docs/GEMINI_PROVIDER_DESIGN.md)."
        )
    return value.strip()


@dataclass(frozen=True)
class GeminiConfig:
    """Immutable provider settings. The API key is never stringified."""

    api_key: str
    model: str = DEFAULT_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    temperature: float = DEFAULT_TEMPERATURE
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS

    def __post_init__(self) -> None:
        if not self.api_key or not self.api_key.strip():
            raise GeminiConfigurationError(
                f"{ENV_API_KEY} is not set. Run with a real provider only after configuring it "
                f"(see docs/GEMINI_PROVIDER_DESIGN.md)."
            )
        if not self.model or not self.model.strip():
            raise GeminiConfigurationError(f"{ENV_MODEL} must not be empty")
        if self.max_retries < 0:
            raise GeminiConfigurationError("max_retries must be >= 0")
        if self.timeout_seconds <= 0:
            raise GeminiConfigurationError("timeout_seconds must be > 0")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "GeminiConfig":
        """Build config from an env mapping (defaults to os.environ).

        Raises GeminiConfigurationError when GEMINI_API_KEY is missing.
        """
        env = os.environ if env is None else env
        key = _non_empty(env.get(ENV_API_KEY), ENV_API_KEY)
        model = (env.get(ENV_MODEL) or DEFAULT_MODEL).strip()
        return cls(api_key=key, model=model)

    def __repr__(self) -> str:
        return (
            f"GeminiConfig(model={self.model!r}, timeout_seconds={self.timeout_seconds}, "
            f"max_retries={self.max_retries}, api_key=<redacted>)"
        )

    def redacted(self) -> dict:
        """Provider-safe provenance/metadata (never contains the key)."""
        return {
            "provider": "gemini",
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
        }