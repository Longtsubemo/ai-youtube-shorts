"""Stage-1 candidate-provider wiring tests (mock path + builder).

The default provider must remain the deterministic mock; requesting the real
provider must require explicit opt-in plus a configured key.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from agents.seo.provider import (
    MockTopicCandidateProvider,
    TopicCandidateProviderError,
    build_topic_candidate_provider,
)
from agents.seo.stage import MOCK_CANDIDATES_PATH, load_candidates


def test_default_provider_is_mock(default_config):
    provider = build_topic_candidate_provider(None, config=default_config)
    assert isinstance(provider, MockTopicCandidateProvider)


def test_mock_provider_returns_same_candidates_as_loader(default_config):
    provider = MockTopicCandidateProvider(MOCK_CANDIDATES_PATH, default_config)
    assert provider.list_candidates() == load_candidates(MOCK_CANDIDATES_PATH, default_config)


def test_unknown_provider_raises(default_config):
    with pytest.raises(TopicCandidateProviderError):
        build_topic_candidate_provider("openai", config=default_config)


def test_gemini_provider_requires_api_key(default_config, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(Exception) as excinfo:
        build_topic_candidate_provider("gemini", config=default_config, env={})
    assert "GEMINI_API_KEY" in str(excinfo.value)


def test_builder_accepts_injected_schema_classes():
    """Schema models used by the provider are plain pydantic classes that
    the stage can import without the SDK."""

    from providers.gemini.topic_provider import CandidateListSchema, TopicCandidateSchema

    assert issubclass(CandidateListSchema, BaseModel)
    assert issubclass(TopicCandidateSchema, BaseModel)