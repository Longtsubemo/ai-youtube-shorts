"""Offline tests for FirecrawlConfig.

Key contract: the API key is OPTIONAL (keyless free tier is first-class) and
is never leaked by repr/str/redacted(). Mirror of the Gemini config tests.
"""

from __future__ import annotations

import json

import pytest

from providers.firecrawl.config import ENV_API_KEY, ENV_API_URL, FirecrawlConfig
from providers.firecrawl.errors import FirecrawlConfigurationError


class TestKeylessMode:
    def test_empty_env_is_keyless(self):
        config = FirecrawlConfig.from_env({})
        assert config.api_key == ""
        assert config.api_url == ""
        assert config.redacted()["keyless"] is True

    def test_missing_key_is_not_an_error(self):
        assert FirecrawlConfig.from_env({}).api_key == ""

    def test_from_env_reads_key_and_url(self):
        config = FirecrawlConfig.from_env(
            {ENV_API_KEY: "  fc-test-key  ", ENV_API_URL: "https://api.firecrawl.dev"}
        )
        assert config.api_key == "fc-test-key"
        assert config.api_url == "https://api.firecrawl.dev"

    def test_from_env_defaults_to_os_environ(self, monkeypatch):
        monkeypatch.setenv(ENV_API_KEY, "fc-from-os")
        assert FirecrawlConfig.from_env().api_key == "fc-from-os"

    def test_defaults_are_bounded(self):
        config = FirecrawlConfig()
        assert config.max_search_results >= 1
        assert config.max_retries >= 0
        assert config.timeout_seconds > 0


class TestSecretsDiscipline:
    def test_repr_never_leaks_key(self):
        config = FirecrawlConfig(api_key="fc-top-secret")
        assert "fc-top-secret" not in repr(config)
        assert "<redacted>" in repr(config)

    def test_str_never_leaks_key(self):
        assert "fc-top-secret" not in str(FirecrawlConfig(api_key="fc-top-secret"))

    def test_redacted_never_leaks_key(self):
        data = FirecrawlConfig(api_key="fc-top-secret").redacted()
        assert "fc-top-secret" not in json.dumps(data)
        assert data["keyless"] is False
        assert data["provider"] == "firecrawl"


class TestValidation:
    def test_zero_timeout_raises(self):
        with pytest.raises(FirecrawlConfigurationError):
            FirecrawlConfig(timeout_seconds=0)

    def test_negative_timeout_raises(self):
        with pytest.raises(FirecrawlConfigurationError):
            FirecrawlConfig(timeout_seconds=-1)

    def test_negative_max_retries_raises(self):
        with pytest.raises(FirecrawlConfigurationError):
            FirecrawlConfig(max_retries=-1)

    def test_zero_max_search_results_raises(self):
        with pytest.raises(FirecrawlConfigurationError):
            FirecrawlConfig(max_search_results=0)