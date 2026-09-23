"""Security tests for the Stage-4 Gemini script provider.

The provider must never leak the API key into inspectable state (error
messages, serialized scripts, provenance) and the default/mock path must work
without any key or the SDK.
"""

from __future__ import annotations

import json

import pytest

from agents.script.provider import MockScriptProvider, build_script_provider
from providers.gemini.config import GeminiConfig
from providers.gemini.errors import (
    GeminiAuthenticationError,
    GeminiConfigurationError,
    GeminiUnexpectedError,
)
from tests.gemini_test_utils import FakeRawClient, make_client, valid_research_payload
from test_gemini_script_provider import build_package, _valid_payload

KEY = "sk-secret-key-abc123"


def _provider(fake, *, key=KEY):
    from providers.gemini.script_provider import GeminiScriptProvider

    client = make_client(GeminiConfig(api_key=key), raw_client=fake, sleep=lambda *_: None)
    return GeminiScriptProvider(client=client)


def test_key_never_appears_in_script_artifact():
    package = build_package()
    fake = FakeRawClient()
    fake.add_result_json(_valid_payload(package))
    generation = _provider(fake).generate(package)
    assert generation.generated
    assert KEY not in generation.script.to_json()


def test_key_never_appears_in_auth_error():
    from google.genai import errors as genai_errors

    package = build_package()
    fake = FakeRawClient()
    fake.add_error(genai_errors.ClientError(code=401, response_json={}))
    with pytest.raises(GeminiAuthenticationError) as exc_info:
        _provider(fake).generate(package)
    assert KEY not in str(exc_info.value)
    assert KEY not in exc_info.value.args[0]


def test_key_never_appears_in_unexpected_error():
    package = build_package()
    fake = FakeRawClient()
    fake.add_error(RuntimeError("boom"))
    with pytest.raises(GeminiUnexpectedError) as exc_info:
        _provider(fake).generate(package)
    assert KEY not in str(exc_info.value)


def test_export_redacted_excludes_key():
    client = make_client(GeminiConfig(api_key=KEY), raw_client=FakeRawClient())
    info = client.export_redacted()
    assert KEY not in json.dumps(info)


def test_mock_provider_needs_no_key_and_no_network():
    package = build_package()
    generation = MockScriptProvider().generate(package)
    assert generation.generated


def test_gemini_factory_without_key_raises_configuration_error():
    with pytest.raises(GeminiConfigurationError):
        build_script_provider("gemini", env={})


def test_mock_factory_without_key_works():
    provider = build_script_provider("mock", env={})
    assert isinstance(provider, MockScriptProvider)