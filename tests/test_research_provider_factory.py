"""Tests for build_research_provider factory wiring.

The factory is what keeps the default/research path SDK- and key-free. These
tests verify selection, error handling, and the gemini opt-in path (lazy
SDK import, key required).
"""

from __future__ import annotations

import pytest

from agents.research.provider import (
    MockResearchProvider,
    ResearchProviderError,
    build_research_provider,
)
from providers.gemini.errors import GeminiConfigurationError


def test_default_provider_is_mock():
    assert isinstance(build_research_provider(None, env={}), MockResearchProvider)


def test_empty_string_provider_is_mock():
    assert isinstance(build_research_provider("", env={}), MockResearchProvider)


def test_explicit_mock_provider():
    assert isinstance(build_research_provider("mock", env={}), MockResearchProvider)


def test_mock_provider_uses_data_path():
    provider = build_research_provider("mock", data_path="data/mock/research_evidence.json", env={})
    assert isinstance(provider, MockResearchProvider)
    assert provider._data_path.name == "research_evidence.json"


def test_unknown_provider_raises():
    with pytest.raises(ResearchProviderError, match="unknown research provider"):
        build_research_provider("nonsense", env={})


def test_gemini_without_key_raises_configuration_error():
    with pytest.raises(GeminiConfigurationError):
        build_research_provider("gemini", env={})


def test_gemini_with_key_constructs():
    from providers.gemini.research_provider import GeminiResearchProvider

    provider = build_research_provider("gemini", env={"GEMINI_API_KEY": "test-key"})
    assert isinstance(provider, GeminiResearchProvider)
    assert provider.provider_name == "gemini"


def test_gemini_factory_construction_does_not_import_sdk():
    """Building the provider must not import the SDK (only a real call does)."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "from agents.research.provider import build_research_provider\n"
        "build_research_provider('gemini', env={'GEMINI_API_KEY': 'test-key'})\n"
        "assert 'google.genai' not in sys.modules, 'google.genai imported transitively'\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=".")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


# --------------------------------------------------------------------------- #
# --retriever firecrawl wiring
# --------------------------------------------------------------------------- #


def test_mock_with_retriever_raises():
    with pytest.raises(ResearchProviderError, match="--retriever is only supported"):
        build_research_provider("mock", retriever="firecrawl", env={})


def test_gemini_with_firecrawl_retriever_constructs_keyless():
    from providers.gemini.research_provider import GeminiResearchProvider

    provider = build_research_provider(
        "gemini",
        env={"GEMINI_API_KEY": "test-key"},  # no FIRECRAWL_API_KEY: keyless OK
        retriever="firecrawl",
    )
    assert isinstance(provider, GeminiResearchProvider)
    assert provider.retriever_name == "firecrawl"
    assert provider.prompt_version == "research/facts-retrieval-v1"
    assert "retrieved_by" in provider.produced_by


def test_gemini_construction_does_not_import_any_sdk_with_retriever():
    """Building the firecrawl-wired provider must not import either SDK."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "from agents.research.provider import build_research_provider\n"
        "build_research_provider('gemini', env={'GEMINI_API_KEY': 'test-key'}, retriever='firecrawl')\n"
        "assert 'google.genai' not in sys.modules, 'google.genai imported transitively'\n"
        "assert 'firecrawl' not in sys.modules, 'firecrawl SDK imported transitively'\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=".")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_unknown_retriever_raises():
    with pytest.raises(ResearchProviderError, match="unknown retriever"):
        build_research_provider("gemini", env={"GEMINI_API_KEY": "test-key"}, retriever="nonsense")