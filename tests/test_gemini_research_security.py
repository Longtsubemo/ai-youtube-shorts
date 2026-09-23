"""Security tests for the Stage-2 Gemini research provider.

The provider must never leak the API key into inspectable state (records,
error messages, packages, collected data) and the default/mock path must work
without any key or the SDK.
"""

from __future__ import annotations

import json

import pytest

from agents.research.agent import ResearchAgent
from agents.research.provider import (
    MockResearchProvider,
    TopicDescriptor,
    build_research_provider,
)
from providers.gemini.config import GeminiConfig
from providers.gemini.errors import GeminiAuthenticationError, GeminiConfigurationError, GeminiUnexpectedError
from tests.gemini_test_utils import (
    FakeRawClient,
    grounded_result,
    make_client,
    valid_research_payload,
)

KEY = "sk-secret-key-abc123"
TOPIC = TopicDescriptor(
    topic_id="t_octopus_hearts",
    title="Why octopuses have three hearts",
    description="Test description",
    pillar="science",
    topic_type="surprising_fact",
)


def _provider(fake, *, key=KEY):
    client = make_client(GeminiConfig(api_key=key), raw_client=fake, sleep=lambda *_: None)
    from providers.gemini.research_provider import GeminiResearchProvider

    return GeminiResearchProvider(client=client)


def _grounded_fake():
    fake = FakeRawClient()
    fake.outcomes.append(
        grounded_result(
            valid_research_payload(),
            chunks=[("https://example.test/papers/octopus", "Grounded paper", "example.test")],
        )
    )
    return fake


def test_key_never_appears_in_candidate_state():
    from dataclasses import asdict

    provider = _provider(_grounded_fake())
    candidates = provider.search(TOPIC)
    assert KEY not in json.dumps([asdict(c) for c in candidates])


def test_key_never_appears_in_package_artifact(minimal_selection):
    provider = _provider(_grounded_fake())
    package = ResearchAgent(provider, run_id="sec").research(minimal_selection("t_octopus_hearts", "Three hearts"))
    package.produced_by = dict(provider.produced_by)
    package.validate()
    serialized = package.to_json() + json.dumps(package.to_dict())
    assert KEY not in serialized


def test_key_never_appears_in_auth_error():
    from google.genai import errors as genai_errors

    fake = FakeRawClient()
    fake.add_error(genai_errors.ClientError(code=401, response_json={}))
    provider = _provider(fake)
    with pytest.raises(GeminiAuthenticationError) as exc_info:
        provider.search(TOPIC)
    assert KEY not in str(exc_info.value)
    assert KEY not in exc_info.value.args[0]


def test_key_never_appears_in_unexpected_error():
    fake = FakeRawClient()
    fake.add_error(RuntimeError("boom"))
    provider = _provider(fake)
    with pytest.raises(GeminiUnexpectedError) as exc_info:
        provider.search(TOPIC)
    assert KEY not in str(exc_info.value)


def test_export_redacted_excludes_key():
    client = make_client(GeminiConfig(api_key=KEY), raw_client=_grounded_fake())
    info = client.export_redacted()
    assert KEY not in json.dumps(info)


def test_mock_provider_needs_no_key_and_no_network():
    provider = MockResearchProvider()
    candidates = provider.search(TOPIC)
    assert isinstance(candidates, list)


def test_gemini_factory_without_key_raises_configuration_error():
    with pytest.raises(GeminiConfigurationError):
        build_research_provider("gemini", env={})


def test_mock_factory_without_key_works():
    provider = build_research_provider("mock", env={})
    assert isinstance(provider, MockResearchProvider)