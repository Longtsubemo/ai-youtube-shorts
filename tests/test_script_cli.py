"""CLI wiring tests for the package-based script generation path.

Exercises main() end-to-end for the mock default and the gemini opt-in path
without network or a key. State writes are redirected so the repo's data/
tree stays untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from agents.script.provider import MockScriptProvider
from agents.script.script_generator import main
from fact_check_test_utils import make_claim, make_package, make_source
from script_test_utils import (
    OCTOPUS_BLUE,
    OCTOPUS_SMARTEST,
    OCTOPUS_SWIM,
    OCTOPUS_TANK,
    OCTOPUS_THREE_HEARTS,
)

CLI_SPECS = (
    (OCTOPUS_THREE_HEARTS, "fact"),
    (OCTOPUS_SWIM, "explanation"),
    (OCTOPUS_BLUE, "explanation"),
    (OCTOPUS_SMARTEST, "inference"),
    (OCTOPUS_TANK, "speculation"),
)


def _cli_package_path(tmp_path) -> Path:
    source = make_source(url="https://example.test/papers/octopus", tier="primary")
    claims = [
        make_claim(text, claim_type=ctype, source_urls=[source.url])
        for text, ctype in CLI_SPECS
    ]
    package = make_package(
        claims=claims,
        sources=[source],
        run_id="cli-pkg-source",
        topic_id="t_octopus_hearts",
        topic_title="Why octopuses have three hearts",
    ).validate()
    path = tmp_path / "research_package.json"
    path.write_text(package.to_json(), encoding="utf-8")
    return path


class StubGeminiScriptProvider(MockScriptProvider):
    """A gemini-stamped stand-in for GeminiScriptProvider (no network)."""

    provider_name = "gemini"

    @property
    def produced_by(self):
        return {
            "prompt_version": "script/script-v1",
            "provider": "gemini",
            "model": "stub-model",
        }


def _install_argv(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["script_generator", *argv])


def _stub_state_save(monkeypatch):
    from pipeline.state import PipelineState

    def fake_save(self, path):
        return Path(path)

    monkeypatch.setattr(PipelineState, "save", fake_save)


def _stub_builder(monkeypatch, provider=None, *, raises=None):
    captured = {}
    if provider is not None:

        def builder(name, *, env=None, config=None):
            captured["provider"] = name
            return provider

        monkeypatch.setattr("agents.script.provider.build_script_provider", builder)
    elif raises is not None:

        def builder(name, *, env=None, config=None):
            raise raises

        monkeypatch.setattr("agents.script.provider.build_script_provider", builder)
    else:  # pragma: no cover - tests always provide one
        raise AssertionError
    return captured


def _read_script(tmp_path, run_id) -> dict:
    path = tmp_path / run_id / "scripts" / "script.json"
    assert path.is_file(), f"missing artifact: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_mock_default_from_research_package(monkeypatch, tmp_path):
    research = _cli_package_path(tmp_path)
    _install_argv(
        monkeypatch,
        "--provider",
        "mock",
        "--research",
        str(research),
        "--preview",
        "--run-id",
        "pkg_cli_mock",
        "--out",
        str(tmp_path),
    )
    _stub_state_save(monkeypatch)
    assert main() == 0
    artifact = _read_script(tmp_path, "pkg_cli_mock")
    assert artifact["schema"] == "Script"
    assert artifact["produced_by"]["model"] == "MockScriptGenerator"
    assert artifact["data"]["result_id"] == ""
    assert artifact["data"]["package_id"]
    assert artifact["data"]["lines"]


def test_gemini_provider_stamps_provenance(monkeypatch, tmp_path):
    research = _cli_package_path(tmp_path)
    _install_argv(
        monkeypatch,
        "--provider",
        "gemini",
        "--research",
        str(research),
        "--preview",
        "--run-id",
        "pkg_cli_gem",
        "--out",
        str(tmp_path),
    )
    _stub_state_save(monkeypatch)
    captured = _stub_builder(monkeypatch, StubGeminiScriptProvider())
    assert main() == 0
    assert captured["provider"] == "gemini"
    artifact = _read_script(tmp_path, "pkg_cli_gem")
    assert artifact["produced_by"] == {
        "prompt_version": "script/script-v1",
        "provider": "gemini",
        "model": "stub-model",
    }


def test_gemini_without_key_exits_error_and_writes_nothing(monkeypatch, tmp_path, capsys):
    from providers.gemini.errors import GeminiConfigurationError

    research = _cli_package_path(tmp_path)
    _stub_state_save(monkeypatch)
    _stub_builder(monkeypatch, raises=GeminiConfigurationError("GEMINI_API_KEY is required."))
    _install_argv(
        monkeypatch,
        "--provider",
        "gemini",
        "--research",
        str(research),
        "--preview",
        "--run-id",
        "pkg_cli_bad",
        "--out",
        str(tmp_path),
    )
    assert main() == 2
    assert not (tmp_path / "pkg_cli_bad" / "scripts" / "script.json").exists()
    captured = capsys.readouterr().out
    assert "GEMINI_API_KEY" in captured or "error" in captured


def test_generate_api_error_exits_error_and_writes_nothing(monkeypatch, tmp_path, capsys):
    from providers.gemini.errors import GeminiAPIError

    research = _cli_package_path(tmp_path)

    class ExplodingProvider:
        def generate(self, package, config=None, run_id="", created_at="", fact_check_result=None):
            raise GeminiAPIError("upstream quota exhausted")

        produced_by = {}

    _stub_state_save(monkeypatch)
    _stub_builder(monkeypatch, ExplodingProvider())
    _install_argv(
        monkeypatch,
        "--provider",
        "gemini",
        "--research",
        str(research),
        "--preview",
        "--run-id",
        "pkg_cli_api",
        "--out",
        str(tmp_path),
    )
    assert main() == 2
    assert not (tmp_path / "pkg_cli_api" / "scripts" / "script.json").exists()
    assert "quota exhausted" in capsys.readouterr().out


def test_gemini_requires_research_flag(monkeypatch, tmp_path, capsys):
    _install_argv(monkeypatch, "--provider", "gemini", "--run-id", "pkg_cli_noresearch", "--out", str(tmp_path))
    _stub_state_save(monkeypatch)
    assert main() == 2
    assert not (tmp_path / "pkg_cli_noresearch" / "scripts" / "script.json").exists()
    assert "--provider gemini requires --research" in capsys.readouterr().out


def test_insufficient_package_exits_zero_writes_nothing(monkeypatch, tmp_path):
    source = make_source(url="https://example.test/papers/octopus", tier="primary")
    package = make_package(
        claims=[make_claim(OCTOPUS_BLUE, claim_type="explanation", source_urls=[source.url])],
        sources=[source],
        run_id="cli-pkg-tiny",
        topic_id="t_octopus_hearts",
        topic_title="Why octopuses have three hearts",
    ).validate()
    research = tmp_path / "tiny_package.json"
    research.write_text(package.to_json(), encoding="utf-8")

    _install_argv(
        monkeypatch,
        "--provider",
        "mock",
        "--research",
        str(research),
        "--preview",
        "--run-id",
        "pkg_cli_tiny",
        "--out",
        str(tmp_path),
    )
    _stub_state_save(monkeypatch)
    assert main() == 0
    assert not (tmp_path / "pkg_cli_tiny" / "scripts" / "script.json").exists()


def test_missing_research_package_exits_error(monkeypatch, tmp_path):
    _install_argv(
        monkeypatch,
        "--provider",
        "mock",
        "--research",
        str(tmp_path / "does_not_exist.json"),
        "--preview",
        "--out",
        str(tmp_path),
    )
    with pytest.raises(SystemExit):
        main()
