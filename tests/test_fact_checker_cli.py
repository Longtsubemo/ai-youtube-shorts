"""CLI wiring tests for the fact-check stage provider opt-in.

Exercises main() end-to-end for the mock default and the gemini opt-in path
without network or a key. State writes are redirected so the repo's data/
tree stays untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from agents.qa.fact_checker.__main__ import main
from agents.qa.fact_checker.verifier import ClaimAssessment
from fact_check_test_utils import StaticEvidenceProvider, make_claim, make_package, make_source

PAPER = "https://example.test/papers/octopus"


class StubGeminiEvidenceProvider(StaticEvidenceProvider):
    """A gemini-stamped stand-in for GeminiEvidenceProvider (no network)."""

    @property
    def produced_by(self):
        return {"prompt_version": "fact_check/evidence-v1", "provider": "gemini", "model": "stub-model"}


class StubGeminiVerifier:
    """A gemini-stamped stand-in for GeminiVerifier (no network)."""

    @property
    def produced_by(self):
        return {"prompt_version": "fact_check/verifier-v1", "provider": "gemini", "model": "stub-model"}

    def verify(self, claim, evidence, sources_by_id):
        return ClaimAssessment(claim_id=claim.claim_id, support_level="none")


class ExplodingEvidenceProvider(StubGeminiEvidenceProvider):
    def get_evidence(self, source):
        raise AssertionError("evidence provider must be stubbed for error tests")


class ExplodingVerifier(StubGeminiVerifier):
    def verify(self, claim, evidence, sources_by_id):
        from providers.gemini.errors import GeminiAPIError

        raise GeminiAPIError("upstream quota exhausted")


def _cli_package_path(tmp_path) -> Path:
    source = make_source(url=PAPER, tier="primary")
    claim = make_claim("Octopuses have three hearts.", source_urls=[source.url])
    package = make_package(
        claims=[claim],
        sources=[source],
        run_id="fc-cli",
        topic_id="t_octopus_hearts",
        topic_title="Why octopuses have three hearts",
    ).validate()
    path = tmp_path / "research_package.json"
    path.write_text(package.to_json(), encoding="utf-8")
    return path


def _install_argv(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["fact_checker", *argv])


def _stub_state_save(monkeypatch):
    from pipeline.state import PipelineState

    def fake_save(self, path):
        return Path(path)

    monkeypatch.setattr(PipelineState, "save", fake_save)


def _stub_gemini_builder(monkeypatch, components=None, *, raises=None):
    captured = {}
    if raises is not None:
        def builder(args):
            raise raises

        monkeypatch.setattr("agents.qa.fact_checker.__main__._build_gemini", builder)
    else:
        def builder(args):
            captured["retriever"] = args.retriever
            return components

        monkeypatch.setattr("agents.qa.fact_checker.__main__._build_gemini", builder)
    return captured


def _gemini_components(provider=None, verifier=None):
    return SimpleNamespace(
        provider=provider or StubGeminiEvidenceProvider(),
        verifier=verifier or StubGeminiVerifier(),
    )


def _read_result(tmp_path, name) -> dict:
    path = tmp_path / name
    assert path.is_file(), f"missing artifact: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_mock_default_runs_offline(monkeypatch, tmp_path):
    research = _cli_package_path(tmp_path)
    _install_argv(
        monkeypatch,
        "--package",
        str(research),
        "--run-id",
        "fc_cli_mock",
        "--out",
        str(tmp_path / "fc_mock.json"),
    )
    _stub_state_save(monkeypatch)
    assert main() == 0
    artifact = _read_result(tmp_path, "fc_mock.json")
    assert artifact["schema"] == "FactCheckResult"
    assert artifact["produced_by"]["model"] == "MockVerifier"
    assert artifact["produced_by"]["provider"] == "MockEvidenceProvider"
    assert "REVIEW" in artifact["data"]["package_verdict"] or "REJECT" in artifact["data"]["package_verdict"]


def test_gemini_provider_stamps_provenance(monkeypatch, tmp_path):
    research = _cli_package_path(tmp_path)
    components = _gemini_components()
    _stub_state_save(monkeypatch)
    captured = _stub_gemini_builder(monkeypatch, components)
    _install_argv(
        monkeypatch,
        "--provider",
        "gemini",
        "--retriever",
        "firecrawl",
        "--package",
        str(research),
        "--run-id",
        "fc_cli_gem",
        "--out",
        str(tmp_path / "fc_gem.json"),
    )
    assert main() == 0
    assert captured["retriever"] == "firecrawl"
    artifact = _read_result(tmp_path, "fc_gem.json")
    assert artifact["produced_by"] == {
        "prompt_version": "fact_check/verifier-v1",
        "provider": "gemini",
        "model": "stub-model",
    }
    assert artifact["data"]["provenance"]["model"] == "stub-model"


def test_gemini_without_key_exits_error_and_writes_nothing(monkeypatch, tmp_path, capsys):
    from agents.qa.fact_checker import provider as provider_factory
    from providers.gemini.errors import GeminiConfigurationError

    research = _cli_package_path(tmp_path)
    _stub_state_save(monkeypatch)

    def keyless_builder(name, *, evidence_data_path=None, env=None, retriever=None):
        raise GeminiConfigurationError("GEMINI_API_KEY is required.")

    monkeypatch.setattr(provider_factory, "build_fact_check_components", keyless_builder)
    _install_argv(
        monkeypatch,
        "--provider",
        "gemini",
        "--package",
        str(research),
        "--run-id",
        "fc_cli_bad",
        "--out",
        str(tmp_path / "fc_bad.json"),
    )
    assert main() == 2
    assert not (tmp_path / "fc_bad.json").exists()
    assert "GEMINI_API_KEY" in capsys.readouterr().err


def test_gemini_api_error_exits_error_and_writes_nothing(monkeypatch, tmp_path, capsys):
    research = _cli_package_path(tmp_path)
    components = _gemini_components(verifier=ExplodingVerifier())
    _stub_state_save(monkeypatch)
    _stub_gemini_builder(monkeypatch, components)
    _install_argv(
        monkeypatch,
        "--provider",
        "gemini",
        "--package",
        str(research),
        "--run-id",
        "fc_cli_api",
        "--out",
        str(tmp_path / "fc_api.json"),
    )
    assert main() == 2
    assert not (tmp_path / "fc_api.json").exists()
    assert "quota exhausted" in capsys.readouterr().err + capsys.readouterr().out


def test_gemini_evidence_failure_is_fatal_and_writes_nothing(monkeypatch, tmp_path, capsys):
    from providers.gemini.errors import GeminiAPIError

    research = _cli_package_path(tmp_path)

    class RaisingEvidenceProvider(StubGeminiEvidenceProvider):
        def get_evidence(self, source):
            raise GeminiAPIError("retrieval quota exhausted")

    components = _gemini_components(provider=RaisingEvidenceProvider())
    _stub_state_save(monkeypatch)
    _stub_gemini_builder(monkeypatch, components)
    _install_argv(
        monkeypatch,
        "--provider",
        "gemini",
        "--package",
        str(research),
        "--run-id",
        "fc_cli_ev",
        "--out",
        str(tmp_path / "fc_ev.json"),
    )
    assert main() == 2
    assert not (tmp_path / "fc_ev.json").exists()
    assert "aborting without artifact" in capsys.readouterr().err + capsys.readouterr().out


def test_retriever_with_mock_exits_error(monkeypatch, tmp_path, capsys):
    research = _cli_package_path(tmp_path)
    _install_argv(
        monkeypatch,
        "--provider",
        "mock",
        "--retriever",
        "firecrawl",
        "--package",
        str(research),
        "--run-id",
        "fc_cli_rm",
        "--out",
        str(tmp_path / "fc_rm.json"),
    )
    assert main() == 2
    assert not (tmp_path / "fc_rm.json").exists()
    assert "retriever" in (capsys.readouterr().err + capsys.readouterr().out).lower()


def test_unknown_retriever_exits_error(monkeypatch, tmp_path, capsys):
    research = _cli_package_path(tmp_path)
    _install_argv(
        monkeypatch,
        "--provider",
        "gemini",
        "--retriever",
        "bing",
        "--package",
        str(research),
        "--run-id",
        "fc_cli_rb",
        "--out",
        str(tmp_path / "fc_rb.json"),
    )
    assert main() == 2
    assert not (tmp_path / "fc_rb.json").exists()
    assert "unknown retriever" in (capsys.readouterr().err + capsys.readouterr().out).lower()


# --------------------------------------------------------------------------- #
# .env loading: the gemini path must see GEMINI_API_KEY / GEMINI_MODEL
# without the caller exporting them by hand.
# --------------------------------------------------------------------------- #


class TestDotEnvLoading:
    def test_dotenv_is_loaded_before_the_gemini_provider_is_built(
        self, monkeypatch, tmp_path
    ):
        """A .env-provided key must reach the provider builder."""
        import os

        from agents.qa.fact_checker import __main__ as fc_main

        research = _cli_package_path(tmp_path)
        seen = {}

        def fake_load_dotenv(*_a, **_kw):
            os.environ["GEMINI_API_KEY"] = "key-from-dotenv"
            os.environ["GEMINI_MODEL"] = "gemini-3.5-flash-lite"
            return True

        monkeypatch.setattr(fc_main, "load_dotenv", fake_load_dotenv)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_MODEL", raising=False)

        def builder(args):
            seen["key"] = os.environ.get("GEMINI_API_KEY")
            seen["model"] = os.environ.get("GEMINI_MODEL")
            return _gemini_components()

        monkeypatch.setattr(fc_main, "_build_gemini", builder)
        _stub_state_save(monkeypatch)
        _install_argv(
            monkeypatch,
            "--provider", "gemini",
            "--retriever", "firecrawl",
            "--package", str(research),
            "--run-id", "fc_cli_env",
            "--out", str(tmp_path / "fc_env.json"),
        )
        assert main() == 0
        assert seen["key"] == "key-from-dotenv"
        assert seen["model"] == "gemini-3.5-flash-lite"

    def test_real_load_dotenv_reads_a_repo_style_env_file(self, monkeypatch, tmp_path):
        """Exercise the real python-dotenv call against a repo-shaped .env."""
        import os

        from dotenv import load_dotenv as real_load_dotenv

        from agents.qa.fact_checker import __main__ as fc_main

        env_file = tmp_path / ".env"
        env_file.write_text(
            "GEMINI_API_KEY=key-read-from-file\n"
            "GEMINI_MODEL=gemini-3.5-flash-lite\n",
            encoding="utf-8",
        )
        research = _cli_package_path(tmp_path)
        seen = {}

        monkeypatch.setattr(
            fc_main, "load_dotenv", lambda *a, **kw: real_load_dotenv(env_file)
        )
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_MODEL", raising=False)

        def builder(args):
            seen["key"] = os.environ.get("GEMINI_API_KEY")
            seen["model"] = os.environ.get("GEMINI_MODEL")
            return _gemini_components()

        monkeypatch.setattr(fc_main, "_build_gemini", builder)
        _stub_state_save(monkeypatch)
        _install_argv(
            monkeypatch,
            "--provider", "gemini",
            "--retriever", "firecrawl",
            "--package", str(research),
            "--run-id", "fc_cli_env2",
            "--out", str(tmp_path / "fc_env2.json"),
        )
        assert main() == 0
        assert seen["key"] == "key-read-from-file"
        assert seen["model"] == "gemini-3.5-flash-lite"

    def test_explicit_env_var_wins_over_dotenv_file(self, monkeypatch, tmp_path):
        """load_dotenv must not clobber a value the caller already exported."""
        import os

        from dotenv import load_dotenv as real_load_dotenv

        from agents.qa.fact_checker import __main__ as fc_main

        env_file = tmp_path / ".env"
        env_file.write_text("GEMINI_MODEL=model-from-file\n", encoding="utf-8")
        research = _cli_package_path(tmp_path)
        seen = {}

        monkeypatch.setattr(
            fc_main, "load_dotenv", lambda *a, **kw: real_load_dotenv(env_file)
        )
        monkeypatch.setenv("GEMINI_MODEL", "model-from-process")

        def builder(args):
            seen["model"] = os.environ.get("GEMINI_MODEL")
            return _gemini_components()

        monkeypatch.setattr(fc_main, "_build_gemini", builder)
        _stub_state_save(monkeypatch)
        _install_argv(
            monkeypatch,
            "--provider", "gemini",
            "--retriever", "firecrawl",
            "--package", str(research),
            "--run-id", "fc_cli_env3",
            "--out", str(tmp_path / "fc_env3.json"),
        )
        assert main() == 0
        assert seen["model"] == "model-from-process"

    def test_mock_provider_needs_no_dotenv_and_no_key(self, monkeypatch, tmp_path):
        """The offline default must keep working when dotenv is unavailable."""
        from agents.qa.fact_checker import __main__ as fc_main

        research = _cli_package_path(tmp_path)
        monkeypatch.setattr(fc_main, "load_dotenv", None)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        _stub_state_save(monkeypatch)
        _install_argv(
            monkeypatch,
            "--provider", "mock",
            "--package", str(research),
            "--run-id", "fc_cli_nodotenv",
            "--out", str(tmp_path / "fc_nodotenv.json"),
        )
        assert main() == 0
        artifact = _read_result(tmp_path, "fc_nodotenv.json")
        assert artifact["schema"] == "FactCheckResult"
        assert artifact["produced_by"]["provider"] == "MockEvidenceProvider"
