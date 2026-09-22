"""Offline config tests for the Gemini provider (no network, no real key)."""

from __future__ import annotations

import pytest

from providers.gemini.config import DEFAULT_MODEL, ENV_API_KEY, ENV_MODEL, GeminiConfig
from providers.gemini.errors import GeminiConfigurationError


def test_from_env_reads_key_and_default_model():
    config = GeminiConfig.from_env({ENV_API_KEY: "test-key"})
    assert config.api_key == "test-key"
    assert config.model == DEFAULT_MODEL


def test_from_env_uses_model_override():
    config = GeminiConfig.from_env({ENV_API_KEY: "k", ENV_MODEL: "custom-model"})
    assert config.model == "custom-model"


def test_from_env_missing_key_raises():
    with pytest.raises(GeminiConfigurationError) as excinfo:
        GeminiConfig.from_env({})
    assert ENV_API_KEY in str(excinfo.value)
    assert "k" not in str(excinfo.value)


def test_config_block_blank_key_raises():
    with pytest.raises(GeminiConfigurationError):
        GeminiConfig(api_key="   ")


def test_repr_redacts_api_key():
    config = GeminiConfig(api_key="super-secret-key-1234")
    assert "super-secret-key-1234" not in repr(config)
    assert "redacted" in repr(config)


def test_redacted_never_contains_key():
    config = GeminiConfig(api_key="super-secret-key-1234")
    redacted = config.redacted()
    assert "api_key" not in redacted
    assert "super-secret-key-1234" not in repr(redacted)
    assert redacted["provider"] == "gemini"
    assert redacted["model"] == config.model