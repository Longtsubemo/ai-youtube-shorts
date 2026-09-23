"""CLI wiring tests for the research stage provider opt-in.

Exercises main() end-to-end for the mock default and the gemini opt-in path
without network or a key. State writes are redirected so the repo's data/
tree stays untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agents.research.research_agent import main
from agents.research.provider import DEFAULT_RESEARCH_DATA_PATH, MockResearchProvider

VALID_EVIDENCE = Path("data") / "mock" / "research_evidence.json"


class StubGeminiProvider:
    """A search-capable stand-in for GeminiResearchProvider."""

    provider_name = "gemini"

    @property
    def produced_by(self):
        return {"prompt_version": "research/facts-v1", "provider": "gemini", "model": "stub-model"}

    @property
    def model(self):
        return "stub-model"

    def search(self, topic):
        return MockResearchProvider(data_path=VALID_EVIDENCE).search(topic)


def _install_argv(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["research_agent", *argv])


def _stub_state_save(monkeypatch):
    from pipeline.state import PipelineState

    def fake_save(self, path):
        return Path(path)

    monkeypatch.setattr(PipelineState, "save", fake_save)


def _stub_builder(monkeypatch, provider=None, *, raises=None):
    if provider is not None:

        def builder(name, *, data_path=None, env=None, enable_url_context=False):
            if name == "gemini":
                captured["provider"] = name
                captured["enable_url_context"] = enable_url_context
                return provider
            raise AssertionError("CLI mock path should not construct a provider via builder")

    elif raises is not None:

        def builder(name, *, data_path=None, env=None, enable_url_context=False):
            raise raises

    else:  # pragma: no cover - tests always provide one
        raise AssertionError

    captured = {}
    monkeypatch.setattr("agents.research.research_agent.build_research_provider", builder)
    return captured


def test_mock_default_no_network(monkeypatch, tmp_path):
    _install_argv(monkeypatch, "--out", str(tmp_path / "research.json"), "--run-id", "cli_res_mock")
    _stub_state_save(monkeypatch)
    assert main() == 0
    artifact = json.loads((tmp_path / "research.json").read_text(encoding="utf-8"))
    assert artifact["schema"] == "ResearchPackage"
    assert artifact["produced_by"] == {"prompt_version": None, "provider": None, "model": None}
    assert artifact["data"]["research_status"] == "ready_for_fact_check"


def test_gemini_provider_stamps_provenance_and_honest_notes(monkeypatch, tmp_path):
    _install_argv(monkeypatch, "--provider", "gemini", "--out", str(tmp_path / "research.json"), "--run-id", "cli_res_gem")
    _stub_state_save(monkeypatch)
    captured = _stub_builder(monkeypatch, StubGeminiProvider())
    assert main() == 0
    artifact = json.loads((tmp_path / "research.json").read_text(encoding="utf-8"))
    assert artifact["produced_by"] == {
        "prompt_version": "research/facts-v1",
        "provider": "gemini",
        "model": "stub-model",
    }
    first_note = artifact["data"]["research_notes"][0]
    assert "grounded in Gemini Google-Search retrieval" in first_note
    assert "no invented URLs" in first_note
    assert "provisional" in first_note


def test_gemini_provider_forwards_enable_url_context_flag(monkeypatch, tmp_path):
    _install_argv(
        monkeypatch,
        "--provider",
        "gemini",
        "--enable-url-context",
        "--out",
        str(tmp_path / "research.json"),
        "--run-id",
        "cli_res_ctx",
    )
    _stub_state_save(monkeypatch)
    captured = _stub_builder(monkeypatch, StubGeminiProvider())
    assert main() == 0
    assert captured["provider"] == "gemini"
    assert captured["enable_url_context"] is True


def test_gemini_without_key_exits_error_and_writes_nothing(monkeypatch, tmp_path, capsys):
    from providers.gemini.errors import GeminiConfigurationError

    _stub_state_save(monkeypatch)
    _stub_builder(monkeypatch, raises=GeminiConfigurationError("GEMINI_API_KEY is required."))
    _install_argv(monkeypatch, "--provider", "gemini", "--out", str(tmp_path / "research.json"), "--run-id", "cli_res_bad")
    assert main() == 2
    assert not (tmp_path / "research.json").exists()
    captured = capsys.readouterr().out
    assert "GEMINI_API_KEY" in captured or "error" in captured