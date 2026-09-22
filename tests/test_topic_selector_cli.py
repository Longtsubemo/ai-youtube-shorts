"""CLI wiring tests for agents/seo/topic_selector.py.

Exercises main() end-to-end for the mock and gemini opt-in paths without
network. State writes and candidate loading are redirected so the repo's
data/ tree stays untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agents.seo.config import load_config
from agents.seo.topic_selector import main
from agents.seo.provider import MockTopicCandidateProvider

VALID_CANDIDATES = Path("data") / "mock" / "topic_candidates.json"


class StubGeminiProvider:
    provider_name = "gemini"

    @property
    def produced_by(self):
        return {"prompt_version": "seo/topic_candidates-v1", "provider": "gemini", "model": "stub-model"}

    @property
    def model(self):
        return "stub-model"

    def list_candidates(self):
        return MockTopicCandidateProvider(
            VALID_CANDIDATES, load_config("config/topic_selection.yaml")
        ).list_candidates()


def _install_argv(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["topic_selector", *argv])


def _stub_state_save(monkeypatch):
    from pipeline.state import PipelineState

    def fake_save(self, path):
        return Path(path)

    monkeypatch.setattr(PipelineState, "save", fake_save)


def _builder(name, *, config, candidates_path=None):
    """Return the stub Gemini provider for the gemini path."""
    if name == "gemini":
        return StubGeminiProvider()
    raise AssertionError("CLI mock path should not construct a provider via builder")


def test_mock_default_no_network(monkeypatch, tmp_path):
    from agents.seo.provider import MockTopicCandidateProvider
    from agents.seo.topic_selector import build_topic_candidate_provider as real_builder

    _install_argv(monkeypatch, "--out", str(tmp_path / "sel.json"), "--run-id", "cli_mock")
    _stub_state_save(monkeypatch)
    monkeypatch.setattr(
        "agents.seo.topic_selector.build_topic_candidate_provider",
        lambda name, config, candidates_path=None: MockTopicCandidateProvider(
            VALID_CANDIDATES,
            load_config("config/topic_selection.yaml"),
        ),
    )
    assert main() == 0
    artifact = json.loads((tmp_path / "sel.json").read_text(encoding="utf-8"))
    assert artifact["schema"] == "TopicSelection"
    assert artifact["produced_by"] == {"prompt_version": None, "provider": None, "model": None}


def test_gemini_provider_stamps_provenance(monkeypatch, tmp_path):
    _install_argv(monkeypatch, "--provider", "gemini", "--out", str(tmp_path / "sel.json"), "--run-id", "cli_gem")
    _stub_state_save(monkeypatch)
    monkeypatch.setattr("agents.seo.topic_selector.build_topic_candidate_provider", _builder)
    assert main() == 0
    artifact = json.loads((tmp_path / "sel.json").read_text(encoding="utf-8"))
    assert artifact["produced_by"] == {
        "prompt_version": "seo/topic_candidates-v1",
        "provider": "gemini",
        "model": "stub-model",
    }


def test_gemini_without_key_exits_error_and_writes_nothing(monkeypatch, tmp_path, capsys):
    from providers.gemini.errors import GeminiConfigurationError

    def _fail_builder(name, *, config, candidates_path=None):
        raise GeminiConfigurationError("GEMINI_API_KEY is not set.")

    _install_argv(monkeypatch, "--provider", "gemini", "--out", str(tmp_path / "sel.json"), "--run-id", "cli_gem")
    _stub_state_save(monkeypatch)
    monkeypatch.setattr("agents.seo.topic_selector.build_topic_candidate_provider", _fail_builder)
    assert main() == 2
    assert not (tmp_path / "sel.json").exists()
    captured = capsys.readouterr().out
    assert "GEMINI_API_KEY" in captured or "error" in captured