"""GeminiClient — thin wrapper over the official `google-genai` SDK.

- The SDK is imported **lazily** so importing this module never forces the
  SDK or a network call; stage packages stay SDK-free (the provider boundary
  lives here).
- `generate_structured` maps every SDK/network/parse/validation failure onto
  the provider error hierarchy and applies a small bounded retry policy
  (timeouts and transient server/rate-limit errors only).
- Only validated Pydantic models ever escape this module; raw SDK objects
  never leave the provider boundary.
"""

from __future__ import annotations

import time
from typing import TypeVar

from pydantic import BaseModel

from providers.gemini.config import GeminiConfig
from providers.gemini.errors import (
    GeminiAPIError,
    GeminiAuthenticationError,
    GeminiProviderError,
    GeminiTimeoutError,
    GeminiUnexpectedError,
)
from providers.gemini.structured import parse_response_text

T = TypeVar("T", bound=BaseModel)

_TIMEOUT_EXCEPTIONS: tuple[type, ...] = ()

RETRYABLE_STATUSES = (408, 429, 500, 502, 503, 504)


def _timeouts() -> tuple[type, ...]:
    """Resolve network-timeout exception types lazily."""
    global _TIMEOUT_EXCEPTIONS
    if not _TIMEOUT_EXCEPTIONS:
        try:
            import httpx

            _TIMEOUT_EXCEPTIONS = (
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadTimeout,
                httpx.PoolTimeout,
            )
        except ImportError:  # pragma: no cover - only when httpx is absent
            _TIMEOUT_EXCEPTIONS = (TimeoutError,)
    return _TIMEOUT_EXCEPTIONS


class GeminiClient:
    """Minimal provider wrapper. Inject `raw_client` for offline tests."""

    def __init__(
        self,
        config: GeminiConfig,
        *,
        raw_client: object | None = None,
        sleep: object | None = None,
    ) -> None:
        self._config = config
        self._raw_client = raw_client
        self._sleep = sleep if sleep is not None else time.sleep

    @property
    def config(self) -> GeminiConfig:
        return self._config

    @property
    def model(self) -> str:
        return self._config.model

    def export_redacted(self) -> dict:
        """Redacted metadata for provenance (never the API key)."""
        return self._config.redacted()

    # -- public ------------------------------------------------------------- #

    def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> T:
        """Ask the model for a structured JSON response validated as `response_model`."""
        client = self._get_client()
        attempts = self._config.max_retries + 1
        last_error: GeminiProviderError | None = None

        for attempt in range(attempts):
            try:
                response = client.models.generate_content(
                    model=self._config.model,
                    contents=prompt,
                    config=self._request_config(
                        response_model,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                    ),
                )
                return parse_response_text(response, response_model)
            except GeminiProviderError as exc:
                raise exc
            except Exception as exc:  # noqa: BLE001 - all SDK/network failures land here
                mapped, retryable = self._classify(exc)
                last_error = mapped
                if retryable and attempt < attempts - 1:
                    self._sleep(0.25 * (2**attempt))
                    continue
                raise mapped from exc

        assert last_error is not None  # pragma: no cover - unreachable
        raise last_error

    # -- internals ---------------------------------------------------------- #

    def _get_client(self) -> object:
        if self._raw_client is not None:
            return self._raw_client
        from google import genai  # lazy: SDK only needed for real calls

        return genai.Client(
            api_key=self._config.api_key,
            http_options=genai.types.HttpOptions(
                timeout=int(self._config.timeout_seconds * 1000),
                retry_options=genai.types.HttpRetryOptions(attempts=0),
            ),
        )

    def _request_config(
        self,
        response_model: type[T],
        *,
        system_prompt: str | None,
        temperature: float | None,
        max_output_tokens: int | None,
    ) -> object:
        from google import genai  # noqa: F401 - re-import inside boundary

        params: dict = {
            "response_mime_type": "application/json",
            "response_schema": response_model,
            "temperature": temperature if temperature is not None else self._config.temperature,
        }
        if system_prompt is not None:
            params["system_instruction"] = system_prompt
        if max_output_tokens is not None or self._config.max_output_tokens:
            params["max_output_tokens"] = (
                max_output_tokens if max_output_tokens is not None else self._config.max_output_tokens
            )
        return genai.types.GenerateContentConfig(**params)

    def _classify(self, exc: Exception) -> tuple[GeminiProviderError, bool]:
        """Classify a raw SDK/network exception into (error, retryable)."""
        try:
            from google.genai import errors as genai_errors

            if isinstance(exc, genai_errors.APIError):
                code = exc.code
                if code in (401, 403):
                    return GeminiAuthenticationError(f"Gemini rejected the API key (HTTP {code})"), False
                if isinstance(code, int):
                    retryable = code in RETRYABLE_STATUSES
                    return (
                        GeminiAPIError(f"Gemini API error (HTTP {code}): {exc.message}"),
                        retryable,
                    )
                return GeminiAPIError(f"Gemini API error: {exc}"), False
        except ImportError:  # pragma: no cover
            pass

        if isinstance(exc, _timeouts()):
            return GeminiTimeoutError(f"Gemini request timed out after {self._config.timeout_seconds:.0f}s"), True

        try:
            import grpc

            if isinstance(exc, grpc.RpcError):  # pragma: no cover - transport v1 legacy
                return GeminiAPIError(f"Gemini transport error: {exc}"), True
        except ImportError:  # pragma: no cover
            pass

        return GeminiUnexpectedError(f"Unexpected Gemini provider failure ({type(exc).__name__})"), False