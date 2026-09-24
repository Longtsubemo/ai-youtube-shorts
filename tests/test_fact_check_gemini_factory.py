"""Factory tests for the Stage-3 fact-check provider pair.

`build_fact_check_components` must: default to the deterministic mock pair,
only import the requested provider (SDK never imported for mock/default runs),
raise typed config errors for a real run without a key, and reject an unknown
provider or a retriever on the mock path.
"""

from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

import pytest

from agents.qa.fact_checker.evidence import EvidenceProviderError, MockEvidenceProvider
from agents.qa.fact_checker.provider import FactCheckComponents, build_fact_check_components
from agents.qa.fact_checker.verifier import MockVerifier
from providers.gemini.errors import GeminiConfigurationError
from providers.gemini.evidence_provider import GeminiEvidenceProvider
from providers.gemini.verifier_provider import GeminiVerifier

ROOT = Path(__file__).resolve().parent.parent
KEY = "sk-test-fc-key"


class TestMockPair:
    def test_default_is_mock(self):
        components = build_fact_check_components(None)
        assert isinstance(components, FactCheckComponents)
        assert isinstance(components.provider, MockEvidenceProvider)
        assert isinstance(components.verifier, MockVerifier)

    def test_explicit_mock(self):
        components = build_fact_check_components("mock")
        assert isinstance(components.provider, MockEvidenceProvider)
        assert isinstance(components.verifier, MockVerifier)

    def test_mock_rejects_retriever(self):
        with pytest.raises(EvidenceProviderError, match="retriever"):
            build_fact_check_components("mock", retriever="firecrawl")

    def test_mock_protocol_compliance(self):
        components = build_fact_check_components("mock")
        assert callable(getattr(components.provider, "get_evidence"))
        assert callable(getattr(components.verifier, "verify"))


class TestGeminiPair:
    def test_gemini_builds_provider_and_verifier(self):
        components = build_fact_check_components("gemini", env={"GEMINI_API_KEY": KEY})
        assert isinstance(components.provider, GeminiEvidenceProvider)
        assert isinstance(components.verifier, GeminiVerifier)
        assert components.provider.model == components.verifier.model
        assert components.provider.produced_by["provider"] == "gemini"
        assert components.verifier.produced_by["prompt_version"] == "fact_check/verifier-v1"

    def test_gemini_without_key_raises_config_error(self):
        with pytest.raises(GeminiConfigurationError):
            build_fact_check_components("gemini", env={})

    def test_gemini_firecrawl_retriever_built(self):
        components = build_fact_check_components(
            "gemini", env={"GEMINI_API_KEY": KEY, "FIRECRAWL_API_KEY": "fc-key"}, retriever="firecrawl"
        )
        assert components.provider.retriever_name == "firecrawl"

    def test_gemini_keyless_firecrawl_retriever_built(self):
        # Firecrawl is keyless-capable; only GEMINI_API_KEY is required.
        components = build_fact_check_components("gemini", env={"GEMINI_API_KEY": KEY}, retriever="firecrawl")
        assert components.provider.retriever_name == "firecrawl"

    def test_gemini_unknown_retriever_raises(self):
        with pytest.raises(EvidenceProviderError, match="retriever"):
            build_fact_check_components("gemini", env={"GEMINI_API_KEY": KEY}, retriever="bing")


class TestUnknownAndIsolation:
    def test_unknown_provider_raises(self):
        with pytest.raises(EvidenceProviderError, match="unknown fact-check provider"):
            build_fact_check_components("bogus")

    def test_mock_run_never_imports_sdk(self):
        # Subprocess: a mock build must not import the Gemini/Firecrawl SDKs.
        code = (
            "from agents.qa.fact_checker.provider import build_fact_check_components;"
            "build_fact_check_components('mock');"
            "import sys;"
            "assert 'google.genai' not in sys.modules, 'google.genai imported on the mock path';"
            "assert 'providers.gemini.client' not in sys.modules, 'gemini client imported on the mock path';"
            "assert 'firecrawl' not in sys.modules and 'firecrawl.v2' not in sys.modules, 'firecrawl imported on the mock path'"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env={"Path": __import__("os").environ.get("Path", "")},
        )
        assert result.returncode == 0, result.stderr

    def test_gemini_build_never_imports_sdk(self, monkeypatch):
        # Building the gemini pair is configuration only; the SDK stays lazy.
        def _blocked(*args, **kwargs):
            raise AssertionError("network called during provider construction")

        monkeypatch.setattr(socket, "socket", _blocked)
        components = build_fact_check_components("gemini", env={"GEMINI_API_KEY": KEY})
        assert isinstance(components.provider, GeminiEvidenceProvider)