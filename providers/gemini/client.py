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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class GroundingChunkInfo:
    """A real search-grounding hit returned by Gemini for the request.

    `url`/`title`/`domain` come from the retrieval metadata, not model
    memory, so they are the evidence boundary for research sources.
    """

    url: str
    title: str = ""
    domain: str = ""


@dataclass(frozen=True)
class GroundedStructured:
    """Structured model output PLUS the retrieval metadata it was grounded on.

    `data` is what the model proposed (claims, source metadata); `grounding`
    and `url_context_urls` are the actual URLs the retrieval tools returned.
    The two must be cross-checked before anything is treated as evidence.
    """

    data: BaseModel
    grounding: tuple[GroundingChunkInfo, ...]
    url_context_urls: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "grounding",
            tuple(sorted(self.grounding, key=lambda item: item.url)),
        )
        object.__setattr__(self, "url_context_urls", tuple(sorted(self.url_context_urls)))


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

    def generate_research(
        self,
        prompt: str,
        response_model: type[T],
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        enable_url_context: bool = False,
    ) -> GroundedStructured:
        """Research request with Google Search grounding.

        Same retry/error discipline as `generate_structured`, but the request
        turns on the Google Search grounding tool so the model can retrieve
        real sources. The returned `GroundedStructured` carries the parsed
        model output together with the retrieval metadata (grounding chunks
        and optional URL-context hits), which is the evidence boundary.

        `enable_url_context` also enables the URL Context retrieval tool
        (deeper retrieval of supplied URLs). It is off by default: the free
        tier behaviour of grounding + URL context together is unverified.
        """
        client = self._get_client()
        attempts = self._config.max_retries + 1
        last_error: GeminiProviderError | None = None

        for attempt in range(attempts):
            try:
                response = client.models.generate_content(
                    model=self._config.model,
                    contents=prompt,
                    config=self._research_config(
                        response_model,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        enable_url_context=enable_url_context,
                    ),
                )
                parsed = parse_response_text(response, response_model)
                return GroundedStructured(
                    data=parsed,
                    grounding=self._extract_grounding(response),
                    url_context_urls=self._extract_url_context(response),
                )
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

    def _base_params(
        self,
        response_model: type[T],
        *,
        system_prompt: str | None,
        temperature: float | None,
        max_output_tokens: int | None,
    ) -> dict:
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
        return params

    def _request_config(
        self,
        response_model: type[T],
        *,
        system_prompt: str | None,
        temperature: float | None,
        max_output_tokens: int | None,
    ) -> object:
        from google import genai  # noqa: F401 - re-import inside boundary

        return genai.types.GenerateContentConfig(
            **self._base_params(
                response_model,
                system_prompt=system_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        )

    def _research_config(
        self,
        response_model: type[T],
        *,
        system_prompt: str | None,
        temperature: float | None,
        max_output_tokens: int | None,
        enable_url_context: bool,
    ) -> object:
        from google import genai  # noqa: F401 - re-import inside boundary

        return genai.types.GenerateContentConfig(
            **self._base_params(
                response_model,
                system_prompt=system_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
            tools=self._research_tools(enable_url_context),
        )

    def _research_tools(self, enable_url_context: bool) -> list:
        from google import genai  # noqa: F401 - re-import inside boundary

        tools = [genai.types.Tool(google_search=genai.types.GoogleSearch())]
        if enable_url_context:
            tools.append(genai.types.Tool(url_context=genai.types.UrlContext()))
        return tools

    def _extract_grounding(self, response: object) -> tuple[GroundingChunkInfo, ...]:
        """Pull real web hits out of `candidate.grounding_metadata`.

        Best-effort: if the metadata is missing/malformed we return empty and
        callers fail closed (no grounding -> no fabricated sources).
        """
        try:
            candidates = getattr(response, "candidates", None) or []
            metadata = getattr(candidates[0], "grounding_metadata", None)
            chunks: list[GroundingChunkInfo] = []
            for chunk in getattr(metadata, "grounding_chunks", None) or []:
                web = getattr(chunk, "web", None)
                uri = getattr(web, "uri", None) if web is not None else None
                if not isinstance(uri, str) or not uri.strip():
                    continue
                chunks.append(
                    GroundingChunkInfo(
                        url=uri,
                        title=str(getattr(web, "title", "") or ""),
                        domain=str(getattr(web, "domain", "") or ""),
                    )
                )
            return tuple(chunks)
        except (AttributeError, IndexError, TypeError, ValueError):  # pragma: no cover - defensive
            return ()

    def _extract_url_context(self, response: object) -> tuple[str, ...]:
        """Pull retrieved URLs from `candidate.url_context_metadata`."""
        try:
            candidates = getattr(response, "candidates", None) or []
            metadata = getattr(candidates[0], "url_context_metadata", None)
            urls: list[str] = []
            for item in getattr(metadata, "url_metadata", None) or []:
                url = getattr(item, "retrieved_url", None)
                if isinstance(url, str) and url.strip():
                    urls.append(url)
            return tuple(urls)
        except (AttributeError, IndexError, TypeError, ValueError):  # pragma: no cover - defensive
            return ()

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