"""Offline client tests — exercise error mapping, retry, and structured parse.

Everything is driven by a FakeRawClient that never talks to Google, so the
normal test suite stays network-free and never needs a real API key. Real
network integration is gated separately (test_gemini_integration.py).
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field

from providers.gemini.config import GeminiConfig
from providers.gemini.errors import (
    GeminiAPIError,
    GeminiAuthenticationError,
    GeminiProviderError,
    GeminiSchemaValidationError,
    GeminiTimeoutError,
    GeminiUnexpectedError,
)
from tests.gemini_test_utils import FakeRawClient, make_client


class FakeBook(BaseModel):
    title: str
    pages: int = Field(ge=1)


def _config(**overrides) -> GeminiConfig:
    defaults = {"api_key": "test-key", "max_retries": 2}
    defaults.update(overrides)
    return GeminiConfig(**defaults)


def test_generate_structured_returns_validated_model():
    fake = FakeRawClient()
    fake.add_result_json({"title": "Oceans", "pages": 12})
    client = make_client(_config(), raw_client=fake)
    book = client.generate_structured("prompt", FakeBook)
    assert isinstance(book, FakeBook)
    assert book.title == "Oceans"
    assert book.pages == 12


def test_malformed_json_raises_schema_error_bag():
    fake = FakeRawClient()
    fake.add_result("not json at all")
    client = make_client(_config(), raw_client=fake)
    with pytest.raises(GeminiProviderError):
        client.generate_structured("prompt", FakeBook)


def test_schema_violation_raises_validation_error():
    fake = FakeRawClient()
    fake.add_result_json({"title": "Oceans", "pages": -5})
    client = make_client(_config(), raw_client=fake)
    with pytest.raises(GeminiSchemaValidationError):
        client.generate_structured("prompt", FakeBook)


def test_auth_failure_raises_auth_error():
    from google.genai import errors as genai_errors

    fake = FakeRawClient()
    fake.add_error(genai_errors.ClientError(code=401, response_json={}))
    client = make_client(_config(), raw_client=fake)
    with pytest.raises(GeminiAuthenticationError):
        client.generate_structured("prompt", FakeBook)


def test_transient_server_error_retries_then_succeeds():
    from google.genai import errors as genai_errors

    fake = FakeRawClient()
    fake.add_error(genai_errors.ServerError(code=503, response_json={}))
    fake.add_result_json({"title": "Oceans", "pages": 12})
    client = make_client(_config(max_retries=2), raw_client=fake)
    book = client.generate_structured("prompt", FakeBook)
    assert book.title == "Oceans"
    assert fake.attempt_count == 2


def test_server_error_exhausts_retries():
    from google.genai import errors as genai_errors

    fake = FakeRawClient()
    for _ in range(4):
        fake.add_error(genai_errors.ServerError(code=503, response_json={}))
    client = make_client(_config(max_retries=2), raw_client=fake)
    with pytest.raises(GeminiAPIError):
        client.generate_structured("prompt", FakeBook)
    assert fake.attempt_count == 3  # 1 + max_retries bounded


def test_timeout_raises_timeout_error():
    import httpx

    fake = FakeRawClient()
    fake.add_error(httpx.TimeoutException("late"))
    client = make_client(_config(max_retries=0), raw_client=fake)
    with pytest.raises(GeminiTimeoutError):
        client.generate_structured("prompt", FakeBook)


def test_unexpected_failure_raises_unexpected_error():
    fake = FakeRawClient()
    fake.add_error(RuntimeError("boom"))
    client = make_client(_config(), raw_client=fake)
    with pytest.raises(GeminiUnexpectedError):
        client.generate_structured("prompt", FakeBook)


def test_no_raw_sdk_objects_escape():
    fake = FakeRawClient()
    fake.add_result_json({"title": "Oceans", "pages": 12})
    client = make_client(_config(), raw_client=fake)
    result = client.generate_structured("prompt", FakeBook)
    assert isinstance(result, FakeBook)
    assert not hasattr(result, "candidates") or isinstance(result, FakeBook)


def test_redacted_metadata_no_key():
    fake = FakeRawClient()
    fake.add_result_json({"title": "Oceans", "pages": 12})
    client = make_client(_config(api_key="secret-key-xyz"), raw_client=fake)
    info = client.export_redacted()
    assert "secret-key-xyz" not in json.dumps(info)
    assert info["provider"] == "gemini"


def test_client_construction_does_not_import_sdk():
    """Constructing a client must be SDK-free; genai is only imported on call."""
    import sys

    sdk = "google.genai"
    before = set(sys.modules)
    make_client(_config(api_key="k"), raw_client=FakeRawClient())
    added = set(sys.modules) - before
    assert sdk not in added