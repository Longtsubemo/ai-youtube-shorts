"""Wiring tests for build_script_provider.

The factory must be lazy: the mock/default path never imports the Gemini SDK,
gemini requires a key and fails loudly without one, and unknown names raise a
ScriptProviderError instead of silently falling back.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from agents.script.provider import (
    MockScriptProvider,
    ScriptProviderError,
    build_script_provider,
)
from providers.gemini.config import GeminiConfig
from providers.gemini.errors import GeminiConfigurationError


def test_default_is_mock():
    assert isinstance(build_script_provider(None), MockScriptProvider)
    assert isinstance(build_script_provider(""), MockScriptProvider)
    assert isinstance(build_script_provider("mock"), MockScriptProvider)


def test_unknown_provider_raises():
    with pytest.raises(ScriptProviderError):
        build_script_provider("nonesuch")


def test_gemini_without_key_raises_configuration_error():
    with pytest.raises(GeminiConfigurationError):
        build_script_provider("gemini", env={})


def test_mock_never_imports_sdk():
    code = (
        "import sys\n"
        "from agents.script.provider import build_script_provider\n"
        "assert 'google.genai' not in sys.modules, 'google.genai imported transitively'\n"
        "provider = build_script_provider('mock')\n"
        "print(type(provider).__name__)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=".")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "MockScriptProvider"


def test_gemini_construction_is_lazy_and_offline():
    """Building the gemini provider must not call the network or import the SDK."""
    code = (
        "import sys\n"
        "from agents.script.provider import build_script_provider\n"
        "provider = build_script_provider('gemini', env={'GEMINI_API_KEY': 'k-test'})\n"
        "assert 'google.genai' not in sys.modules, 'google.genai imported transitively'\n"
        "assert provider.produced_by['provider'] == 'gemini'\n"
        "assert provider.model\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=".")
    assert result.returncode == 0, result.stderr


def test_gemini_via_factory_reuses_env_whitelist():
    from providers.gemini.script_provider import GeminiScriptProvider

    provider = build_script_provider(
        "gemini", env={"GEMINI_API_KEY": "k-test", "GEMINI_MODEL": "gemini-3.5-flash-lite"}
    )
    assert isinstance(provider, GeminiScriptProvider)
    assert provider.model == "gemini-3.5-flash-lite"
    assert provider.produced_by["prompt_version"] == "script/script-v1"