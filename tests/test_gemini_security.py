"""Security regression tests: the API key must never leak.

Covered surfaces: exceptions, repr, combined error text, provider redacted
metadata, and full artifact JSON produced by a Gemini-backed run.
"""

from __future__ import annotations

import json

import pytest

from agents.seo.config import load_config
from agents.seo.history import SelectionHistory
from agents.seo.stage import run_topic_selection
from providers.gemini.client import GeminiClient
from providers.gemini.config import GeminiConfig
from providers.gemini.errors import GeminiConfigurationError
from providers.gemini.topic_provider import GeminiTopicProvider
from tests.gemini_test_utils import FakeRawClient, make_client, valid_candidates_payload

KEY = "sk-secret-alpha-0001"


def _config() -> GeminiConfig:
    return GeminiConfig(api_key=KEY)


def _provider(fake) -> GeminiTopicProvider:
    client = make_client(_config(), raw_client=fake, sleep=lambda *_: None)
    return GeminiTopicProvider(config=load_config("config/topic_selection.yaml"), client=client)


def test_key_never_in_repr():
    assert KEY not in repr(_config())
    assert KEY not in str(_config())


def test_key_not_in_config_error_text():
    with pytest.raises(GeminiConfigurationError) as excinfo:
        GeminiConfig.from_env({})
    assert KEY not in str(excinfo.value)


def test_key_not_in_client_redacted_metadata():
    fake = FakeRawClient()
    fake.add_result_json(valid_candidates_payload(count=1))
    client = make_client(_config(), raw_client=fake, sleep=lambda *_: None)
    payload = json.dumps(client.export_redacted())
    assert KEY not in payload


def test_key_not_in_gemini_exception_text():
    fake = FakeRawClient()
    fake.add_error(RuntimeError("boom"))
    client = make_client(_config(), raw_client=fake, sleep=lambda *_: None)
    with pytest.raises(Exception) as excinfo:
        client.generate_structured("prompt", dict)
    assert KEY not in str(excinfo.value)


def test_key_not_in_artifact_json():
    fake = FakeRawClient()
    fake.add_result_json(valid_candidates_payload(count=3))
    provider = _provider(fake)
    selection = run_topic_selection(
        provider.list_candidates(),
        SelectionHistory(),
        load_config("config/topic_selection.yaml"),
        run_id="sec_run",
    )
    selection.produced_by = dict(provider.produced_by)
    artifact = selection.to_json()
    assert KEY not in artifact
    assert '"provider": "gemini"' in artifact
    assert "api_key" not in artifact


def test_key_not_in_prompt_file_text():
    from pathlib import Path

    prompt = Path("prompts/seo/topic_candidates-v1.txt").read_text(encoding="utf-8")
    assert KEY not in prompt