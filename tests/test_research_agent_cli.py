"""CLI wiring tests for the research stage provider opt-in.

Exercises main() end-to-end for the mock default and the gemini opt-in path
without network or a key. State writes are redirected so the repo's data/
tree stays untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from agents.research.research_agent import default_selection_path, main
from agents.research.provider import DEFAULT_RESEARCH_DATA_PATH, MockResearchProvider
from agents.seo.result import TopicSelection

VALID_EVIDENCE = Path("data") / "mock" / "research_evidence.json"

# --selection no longer falls back to this fixture. It defaults to the current
# run's own data/<run_id>/research/topic_selection.json, which these CLI tests
# do not create, so every case names the selection explicitly.
MOCK_SELECTION = Path("data") / "mock" / "topic_selection.json"

# Absolute forms, for the cases that chdir into tmp_path to exercise the
# run-derived default without writing anything into the repo's data/ tree.
REPO_ROOT = Path(__file__).resolve().parent.parent
MOCK_DATA_ABS = REPO_ROOT / "data" / "mock" / "research_evidence.json"
MOCK_SELECTION_ABS = REPO_ROOT / "data" / "mock" / "topic_selection.json"


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
    _install_argv(
        monkeypatch,
        "--selection",
        str(MOCK_SELECTION),
        "--out",
        str(tmp_path / "research.json"),
        "--run-id",
        "cli_res_mock",
    )
    _stub_state_save(monkeypatch)
    assert main() == 0
    artifact = json.loads((tmp_path / "research.json").read_text(encoding="utf-8"))
    assert artifact["schema"] == "ResearchPackage"
    assert artifact["produced_by"] == {"prompt_version": None, "provider": None, "model": None}
    assert artifact["data"]["research_status"] == "ready_for_fact_check"


def test_gemini_provider_stamps_provenance_and_honest_notes(monkeypatch, tmp_path):
    _install_argv(
        monkeypatch,
        "--provider",
        "gemini",
        "--selection",
        str(MOCK_SELECTION),
        "--out",
        str(tmp_path / "research.json"),
        "--run-id",
        "cli_res_gem",
    )
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
        "--selection",
        str(MOCK_SELECTION),
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
    _install_argv(
        monkeypatch,
        "--provider",
        "gemini",
        "--selection",
        str(MOCK_SELECTION),
        "--out",
        str(tmp_path / "research.json"),
        "--run-id",
        "cli_res_bad",
    )
    assert main() == 2
    assert not (tmp_path / "research.json").exists()
    captured = capsys.readouterr().out
    assert "GEMINI_API_KEY" in captured or "error" in captured


# --------------------------------------------------------------------------- #
# --selection resolution: the run-derived default and its safety invariants.
#
# Regresses the observed defect where a run recorded one Stage-1 topic in
# data/<run_id>/research/topic_selection.json while Stage 2 researched the
# hardcoded data/mock/topic_selection.json fixture instead.
# --------------------------------------------------------------------------- #


def _write_selection(path: Path, *, run_id: str, topic_id: str = "t_fixture") -> Path:
    """Write a minimal loadable TopicSelection carrying `run_id`."""
    selection = TopicSelection(
        selection_id="sel_fixture",
        selected_topic_id=topic_id,
        selected_topic={
            "topic_id": topic_id,
            "title": f"Fixture topic {topic_id}",
            "description": "",
            "pillar": "science",
            "topic_type": "explainer",
            "total_score": 0.5,
        },
        ranked_candidates=[],
        selection_mode="cold_start",
        historical_data_available=False,
        selection_reason="test fixture",
        scoring_configuration={},
        created_at="2026-01-01T00:00:00+00:00",
        run_id=run_id,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(selection.to_json(), encoding="utf-8")
    return path


def _forbid_builder(monkeypatch) -> list:
    """Record and reject any provider construction (used by the guard cases)."""
    calls: list = []

    def builder(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("provider must not be constructed")

    monkeypatch.setattr("agents.research.research_agent.build_research_provider", builder)
    return calls


def test_default_selection_path_is_the_runs_own_stage1_artifact():
    assert default_selection_path("run-abc") == Path("data") / "run-abc" / "research" / "topic_selection.json"


def test_omitted_selection_derives_the_runs_own_stage1_artifact(monkeypatch, tmp_path):
    """No --selection: the run's own data/<run_id>/research/topic_selection.json is used."""
    run_id = "cli_sel_derived"
    derived = _write_selection(
        tmp_path / "data" / run_id / "research" / "topic_selection.json",
        run_id=run_id,
        topic_id="t_derived_topic",
    )
    monkeypatch.chdir(tmp_path)
    _install_argv(
        monkeypatch,
        "--data",
        str(MOCK_DATA_ABS),
        "--out",
        str(tmp_path / "research.json"),
        "--run-id",
        run_id,
    )
    _stub_state_save(monkeypatch)

    assert main() == 0
    assert derived.is_file()
    artifact = json.loads((tmp_path / "research.json").read_text(encoding="utf-8"))
    assert artifact["run_id"] == run_id
    assert artifact["data"]["topic_id"] == "t_derived_topic"
    assert artifact["data"]["topic_title"] == "Fixture topic t_derived_topic"


def test_omitted_selection_with_missing_stage1_artifact_fails_before_provider(monkeypatch, tmp_path):
    run_id = "cli_sel_missing"
    monkeypatch.chdir(tmp_path)
    _install_argv(
        monkeypatch,
        "--data",
        str(MOCK_DATA_ABS),
        "--out",
        str(tmp_path / "research.json"),
        "--run-id",
        run_id,
    )
    _stub_state_save(monkeypatch)
    calls = _forbid_builder(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        main()

    message = str(excinfo.value)
    assert str(Path("data") / run_id / "research" / "topic_selection.json") in message
    assert "never fabricated" in message
    # The provider must not be constructed, and nothing may be written.
    assert calls == []
    assert not (tmp_path / "research.json").exists()


def test_gemini_cannot_fabricate_a_mock_selection_when_selection_is_omitted(monkeypatch, tmp_path):
    """--provider gemini must not silently fall back to (or create) the mock fixture."""
    run_id = "cli_sel_gemini_nofallback"
    monkeypatch.chdir(tmp_path)
    _install_argv(
        monkeypatch,
        "--provider",
        "gemini",
        "--out",
        str(tmp_path / "research.json"),
        "--run-id",
        run_id,
    )
    _stub_state_save(monkeypatch)
    calls = _forbid_builder(monkeypatch)

    with pytest.raises(SystemExit):
        main()

    assert calls == []
    # Neither a fabricated mock artifact nor a package may appear.
    assert not (tmp_path / "data" / "mock" / "topic_selection.json").exists()
    assert not (tmp_path / "research.json").exists()


def test_explicit_selection_is_still_honoured(monkeypatch, tmp_path):
    """An explicit --selection keeps working, including the mock fixture."""
    monkeypatch.chdir(tmp_path)
    _install_argv(
        monkeypatch,
        "--selection",
        str(MOCK_SELECTION_ABS),
        "--data",
        str(MOCK_DATA_ABS),
        "--out",
        str(tmp_path / "research.json"),
        "--run-id",
        "cli_sel_explicit",
    )
    _stub_state_save(monkeypatch)

    assert main() == 0
    artifact = json.loads((tmp_path / "research.json").read_text(encoding="utf-8"))
    assert artifact["data"]["topic_id"] == "t_octopus_hearts"
    assert artifact["data"]["topic_title"] == "Why octopuses have three hearts"


def test_selection_from_another_run_is_rejected_before_provider(monkeypatch, tmp_path):
    """A selection recorded under a different, non-placeholder run must not be used."""
    run_id = "cli_sel_wrongrun"
    foreign = _write_selection(
        tmp_path / "foreign" / "topic_selection.json",
        run_id="some-other-run",
        topic_id="t_foreign_topic",
    )
    monkeypatch.chdir(tmp_path)
    _install_argv(
        monkeypatch,
        "--selection",
        str(foreign),
        "--data",
        str(MOCK_DATA_ABS),
        "--out",
        str(tmp_path / "research.json"),
        "--run-id",
        run_id,
    )
    _stub_state_save(monkeypatch)
    calls = _forbid_builder(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        main()

    message = str(excinfo.value)
    assert "different run" in message
    assert "some-other-run" in message
    assert run_id in message
    assert calls == []
    assert not (tmp_path / "research.json").exists()


def test_mock_fixture_selection_is_allowed_despite_placeholder_run_id(monkeypatch, tmp_path):
    """data/mock/topic_selection.json carries run_id='mock'; that is not a mismatch."""
    monkeypatch.chdir(tmp_path)
    _install_argv(
        monkeypatch,
        "--selection",
        str(MOCK_SELECTION_ABS),
        "--data",
        str(MOCK_DATA_ABS),
        "--out",
        str(tmp_path / "research.json"),
        "--run-id",
        "cli_sel_mock_fixture",
    )
    _stub_state_save(monkeypatch)

    assert main() == 0
    artifact = json.loads((tmp_path / "research.json").read_text(encoding="utf-8"))
    assert artifact["data"]["topic_id"] == "t_octopus_hearts"


def test_selection_without_a_run_id_is_allowed(monkeypatch, tmp_path):
    """An empty selection run_id cannot be compared, so it is not treated as a mismatch."""
    run_id = "cli_sel_norunid"
    selection = _write_selection(
        tmp_path / "norunid" / "topic_selection.json",
        run_id="cli_sel_norunid_src",
        topic_id="t_norunid_topic",
    )
    payload = json.loads(selection.read_text(encoding="utf-8"))
    payload["run_id"] = ""
    selection.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    _install_argv(
        monkeypatch,
        "--selection",
        str(selection),
        "--data",
        str(MOCK_DATA_ABS),
        "--out",
        str(tmp_path / "research.json"),
        "--run-id",
        run_id,
    )
    _stub_state_save(monkeypatch)

    assert main() == 0
    artifact = json.loads((tmp_path / "research.json").read_text(encoding="utf-8"))
    assert artifact["data"]["topic_id"] == "t_norunid_topic"


def test_explicit_missing_selection_still_reports_not_found(monkeypatch, tmp_path):
    """Explicit-but-missing --selection keeps its original 'not found' failure."""
    monkeypatch.chdir(tmp_path)
    missing = tmp_path / "nope" / "topic_selection.json"
    _install_argv(
        monkeypatch,
        "--selection",
        str(missing),
        "--out",
        str(tmp_path / "research.json"),
        "--run-id",
        "cli_sel_explicit_missing",
    )
    _stub_state_save(monkeypatch)
    calls = _forbid_builder(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert "TopicSelection not found" in str(excinfo.value)
    assert calls == []