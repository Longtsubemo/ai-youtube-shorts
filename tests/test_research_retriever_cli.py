"""CLI wiring tests for --retriever firecrawl (Stage-2 retrieval opt-in).

Exercises main() end-to-end without network or keys: the mock+retriever combo
is rejected, the gemini+firecrawl combo stamps retriever provenance, and
retrieval/runtime failures exit cleanly with no artifact written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agents.research.research_agent import main
from agents.research.provider import DEFAULT_RESEARCH_DATA_PATH, MockResearchProvider, ResearchProviderError

VALID_EVIDENCE = Path("data") / "mock" / "research_evidence.json"


class StubRetrievalGeminiProvider:
    """Stand-in for GeminiResearchProvider with a firecrawl-style retriever."""

    provider_name = "gemini"

    @property
    def produced_by(self):
        return {
            "prompt_version": "research/facts-retrieval-v1",
            "provider": "gemini",
            "model": "stub-model",
            "retrieved_by": "firecrawl",
        }

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
    captured = {}
    if provider is not None:

        def builder(name, *, data_path=None, env=None, enable_url_context=False, retriever=None):
            captured["name"] = name
            captured["retriever"] = retriever
            if name == "gemini":
                return provider
            raise AssertionError("CLI mock path should not construct a provider via builder")

    elif raises is not None:

        def builder(name, *, data_path=None, env=None, enable_url_context=False, retriever=None):
            raise raises

    else:  # pragma: no cover - tests always provide one
        raise AssertionError

    monkeypatch.setattr("agents.research.research_agent.build_research_provider", builder)
    return captured


def test_retriever_with_mock_provider_is_rejected(monkeypatch, tmp_path, capsys):
    _install_argv(
        monkeypatch,
        "--provider",
        "mock",
        "--retriever",
        "firecrawl",
        "--out",
        str(tmp_path / "research.json"),
        "--run-id",
        "cli_fr_bad",
    )
    _stub_state_save(monkeypatch)
    assert main() == 2
    assert not (tmp_path / "research.json").exists()
    assert "requires --provider gemini" in capsys.readouterr().out


def test_gemini_retriever_stamps_provenance_and_note(monkeypatch, tmp_path):
    _install_argv(
        monkeypatch,
        "--provider",
        "gemini",
        "--retriever",
        "firecrawl",
        "--out",
        str(tmp_path / "research.json"),
        "--run-id",
        "cli_fr_ok",
    )
    _stub_state_save(monkeypatch)
    captured = _stub_builder(monkeypatch, StubRetrievalGeminiProvider())
    assert main() == 0
    assert captured["retriever"] == "firecrawl"

    artifact = json.loads((tmp_path / "research.json").read_text(encoding="utf-8"))
    assert artifact["produced_by"] == {
        "prompt_version": "research/facts-retrieval-v1",
        "provider": "gemini",
        "model": "stub-model",
        "retrieved_by": "firecrawl",
    }
    first_note = artifact["data"]["research_notes"][0]
    assert "via Firecrawl" in first_note
    assert "retrieval metadata" in first_note
    assert "no invented URLs" in first_note
    assert "provisional" in first_note


def test_gemini_retriever_unknown_retriever_exits_error(monkeypatch, tmp_path, capsys):
    _stub_state_save(monkeypatch)
    _stub_builder(monkeypatch, raises=ResearchProviderError("unknown retriever: 'bogus'"))
    _install_argv(
        monkeypatch,
        "--provider",
        "gemini",
        "--retriever",
        "firecrawl",
        "--out",
        str(tmp_path / "research.json"),
        "--run-id",
        "cli_fr_unk",
    )
    assert main() == 2
    assert not (tmp_path / "research.json").exists()
    assert "error" in capsys.readouterr().out


def test_gemini_retriever_runtime_failure_exits_error(monkeypatch, tmp_path, capsys):
    _stub_state_save(monkeypatch)
    _stub_builder(
        monkeypatch,
        raises=ResearchProviderError("Firecrawl retrieval failed for topic 't_x': request timed out"),
    )
    _install_argv(
        monkeypatch,
        "--provider",
        "gemini",
        "--retriever",
        "firecrawl",
        "--out",
        str(tmp_path / "research.json"),
        "--run-id",
        "cli_fr_fail",
    )
    assert main() == 2
    assert not (tmp_path / "research.json").exists()
    out = capsys.readouterr().out
    assert "Firecrawl retrieval failed" in out