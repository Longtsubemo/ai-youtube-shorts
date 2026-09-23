"""Gemini provider package (provider-specific code only).

Stage packages never import this module's SDK path; they depend only on the
provider errors raised across the boundary and on the Pydantic models they
pass in. The Gemini SDK is imported lazily inside `client.py`.
"""

from providers.gemini.client import GeminiClient, GroundedStructured, GroundingChunkInfo
from providers.gemini.config import GeminiConfig
from providers.gemini.errors import (
    GeminiAPIError,
    GeminiAuthenticationError,
    GeminiConfigurationError,
    GeminiMalformedResponseError,
    GeminiProviderError,
    GeminiSchemaValidationError,
    GeminiTimeoutError,
    GeminiUnexpectedError,
)
from providers.gemini.research_provider import GeminiResearchProvider
from providers.gemini.script_provider import GeminiScriptProvider, ScriptResultAdapter

__all__ = [
    "GeminiClient",
    "GeminiConfig",
    "GeminiAPIError",
    "GeminiAuthenticationError",
    "GeminiConfigurationError",
    "GeminiMalformedResponseError",
    "GeminiProviderError",
    "GeminiSchemaValidationError",
    "GeminiTimeoutError",
    "GeminiUnexpectedError",
    "GeminiResearchProvider",
    "GeminiScriptProvider",
    "ScriptResultAdapter",
    "GroundedStructured",
    "GroundingChunkInfo",
]