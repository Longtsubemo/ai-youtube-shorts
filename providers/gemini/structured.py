"""Structured-output helpers for the Gemini provider.

The provider asks the SDK for `response_mime_type="application/json"` +
`response_schema=<pydantic model>` (MODEL_STRATEGY.md §5), then validates the
returned JSON against the same model before anything is returned. All raw
response parsing/validation is centralized here so clients stay thin.
"""

from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from providers.gemini.errors import GeminiMalformedResponseError, GeminiSchemaValidationError

T = TypeVar("T", bound=BaseModel)


def parse_json_payload(text: str, response_model: type[T]) -> T:
    """Parse a JSON text payload and validate it as `response_model`.

    Raises GeminiMalformedResponseError for non-JSON text and
    GeminiSchemaValidationError when the payload fails schema validation.
    """
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise GeminiMalformedResponseError(
            f"Gemini returned non-JSON structured output ({type(exc).__name__})"
        ) from exc
    try:
        return response_model.model_validate(payload)
    except ValidationError as exc:
        raise GeminiSchemaValidationError(
            f"Gemini structured output failed {response_model.__name__} validation: {exc}"
        ) from exc


def parse_response_text(response: object, response_model: type[T]) -> T:
    """Extract `.text` from an SDK response object and validate it."""
    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise GeminiMalformedResponseError("Gemini response carried no text payload")
    return parse_json_payload(text, response_model)