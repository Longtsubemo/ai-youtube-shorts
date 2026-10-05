"""Offline Stage-1 Gemini provider tests.

Verify that Gemini produces candidates that survive the existing
TopicCandidate validation gate, that provenance is stamped, and that a
full topic-selection run works with Gemini-sourced candidates.
"""

from __future__ import annotations

import pytest

from agents.seo.config import load_config
from agents.seo.history import SelectionHistory
from agents.seo.stage import run_topic_selection
from providers.gemini.errors import GeminiProviderError, GeminiSchemaValidationError
from providers.gemini.topic_provider import PROMPT_VERSION, GeminiTopicProvider
from tests.gemini_test_utils import FakeRawClient, make_client, valid_candidates_payload

DEFAULT_CONFIG = "config/topic_selection.yaml"


def _provider(config, fake) -> GeminiTopicProvider:
    gemini_config = make_client_config()
    client = make_client(gemini_config, raw_client=fake)
    return GeminiTopicProvider(config=config, client=client)


def make_client_config():
    from providers.gemini.config import GeminiConfig

    return GeminiConfig(api_key="test-key")


def test_provides_candidates_through_validation_gate(default_config):
    fake = FakeRawClient()
    fake.add_result_json(valid_candidates_payload(count=4))
    provider = _provider(default_config, fake)

    candidates = provider.list_candidates()
    assert len(candidates) == 4
    for candidate in candidates:
        # The existing gate already ran inside GeminiTopicProvider.
        assert candidate.validate(default_config.pillars, default_config.topic_types) is candidate
        assert candidate.topic_id.startswith("t_")
        assert candidate.signals.interest >= 0.0
        assert candidate.content_scores.curiosity <= 1.0


def test_gemini_candidates_participate_in_real_selection(default_config):
    fake = FakeRawClient()
    fake.add_result_json(valid_candidates_payload(count=4))
    provider = _provider(default_config, fake)
    candidates = provider.list_candidates()

    selection = run_topic_selection(candidates, SelectionHistory(), default_config, run_id="gem_run")
    selection.produced_by = dict(provider.produced_by)
    assert selection.selected_topic_id.startswith("t_")
    assert selection.ranked_candidates
    header = selection.to_dict()["produced_by"]
    assert header == provider.produced_by
    assert header["provider"] == "gemini"
    assert header["model"] == provider.model
    assert header["prompt_version"] == PROMPT_VERSION


def test_provided_provenance_header():
    fake = FakeRawClient()
    fake.add_result_json(valid_candidates_payload(count=1))
    provider = _provider(load_config(DEFAULT_CONFIG), fake)
    assert provider.produced_by == {
        "prompt_version": PROMPT_VERSION,
        "provider": "gemini",
        "model": provider.model,
    }


def test_candidate_provenance_stamped():
    fake = FakeRawClient()
    fake.add_result_json(valid_candidates_payload(count=2))
    provider = _provider(load_config(DEFAULT_CONFIG), fake)
    candidate = provider.list_candidates()[0]
    assert candidate.provenance is not None
    assert candidate.provenance.generated_by == "gemini_topic_provider_v2"
    assert candidate.provenance.provider == "gemini"
    assert candidate.provenance.model == provider.model
    assert candidate.provenance.prompt_version == PROMPT_VERSION


def test_invalid_pillar_fails_validation_gate(default_config):
    payload = valid_candidates_payload(count=1)
    payload["candidates"][0]["pillar"] = "not_a_pillar"
    fake = FakeRawClient()
    fake.add_result_json(payload)
    provider = _provider(default_config, fake)
    with pytest.raises((GeminiSchemaValidationError, ValueError)):
        provider.list_candidates()


def test_schema_violation_raises_provider_error(default_config):
    payload = valid_candidates_payload(count=1)
    payload["candidates"][0]["signals"]["interest"] = 7.0  # out of [0,1]
    fake = FakeRawClient()
    fake.add_result_json(payload)
    provider = _provider(default_config, fake)
    with pytest.raises(GeminiProviderError):
        provider.list_candidates()


def test_mock_path_untouched_when_gemini_requested_without_key(monkeypatch):
    """Requesting the gemini provider without a key must fail loudly, never
    silently fall back to the mock path (data/mock/...)."""
    from agents.seo.provider import build_topic_candidate_provider

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    config = load_config(DEFAULT_CONFIG)
    with pytest.raises(GeminiProviderError):
        build_topic_candidate_provider("gemini", config=config, env={})