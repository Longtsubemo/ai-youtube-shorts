"""Offline tests for GeminiClient.generate_research (grounded requests).

Driven by a FakeRawClient so the suite never talks to Google and never needs
a real API key. Covers grounding/URL-context extraction, tool wiring, parse
errors, and the shared retry/error discipline.
"""

from __future__ import annotations

import json

import pytest

from providers.gemini.client import GroundedStructured, GroundingChunkInfo
from providers.gemini.config import GeminiConfig
from providers.gemini.errors import (
    GeminiAPIError,
    GeminiAuthenticationError,
    GeminiProviderError,
    GeminiSchemaValidationError,
    GeminiTimeoutError,
    GeminiUnexpectedError,
)
from providers.gemini.research_provider import ResearchResultSchema
from tests.gemini_test_utils import (
    FakeRawClient,
    grounded_result,
    make_client,
    valid_research_payload,
)

GROUNDED_URL = "https://example.test/papers/octopus"


def _config(**overrides) -> GeminiConfig:
    defaults = {"api_key": "test-key", "max_retries": 2}
    defaults.update(overrides)
    return GeminiConfig(**defaults)


def _grounded_fake(payload=None, *, chunks=None, url_context_urls=()):
    fake = FakeRawClient()
    fake.outcomes.append(
        grounded_result(
            payload if payload is not None else valid_research_payload(),
            chunks=GROUNDED_CHUNKS if chunks is None else chunks,
            url_context_urls=url_context_urls,
        )
    )
    return fake


GROUNDED_CHUNKS = [(GROUNDED_URL, "Octopus hearts paper", "example.test")]


class TestGroundedRequests:
    def test_returns_parsed_data_with_grounding(self):
        fake = _grounded_fake()
        client = make_client(_config(), raw_client=fake)
        result = client.generate_research("prompt", ResearchResultSchema)
        assert isinstance(result, GroundedStructured)
        assert isinstance(result.data, ResearchResultSchema)
        assert result.data.sources[0].url == GROUNDED_URL
        assert result.grounding == (GroundingChunkInfo(GROUNDED_URL, "Octopus hearts paper", "example.test"),)

    def test_url_context_urls_carried_through(self):
        fake = _grounded_fake(url_context_urls=[GROUNDED_URL])
        client = make_client(_config(), raw_client=fake)
        result = client.generate_research("prompt", ResearchResultSchema, enable_url_context=True)
        assert result.url_context_urls == (GROUNDED_URL,)

    def test_response_without_candidates_yields_empty_grounding(self):
        fake = FakeRawClient()
        fake.outcomes.append(grounded_result(valid_research_payload()))
        client = make_client(_config(), raw_client=fake)
        result = client.generate_research("prompt", ResearchResultSchema)
        assert result.grounding == ()
        assert result.url_context_urls == ()

    def test_grounding_and_context_extraction_are_defensive(self):
        """Missing/malformed metadata must not raise."""

        class Odd:
            def __init__(self, **kw):
                self.__dict__.update(kw)

        fake = FakeRawClient()
        from types import SimpleNamespace

        payload = json.dumps(valid_research_payload())
        response = SimpleNamespace(
            text=payload,
            candidates=(
                SimpleNamespace(
                    grounding_metadata=SimpleNamespace(
                        grounding_chunks=(
                            SimpleNamespace(web=Odd(uri="https://example.test/a")),  # no title/domain
                            None,  # malformed chunk
                            SimpleNamespace(),  # no web
                        )
                    ),
                    url_context_metadata=SimpleNamespace(url_metadata=(None,)),
                ),
                None,  # malformed candidate
            ),
        )
        fake.outcomes.append(response)
        client = make_client(_config(), raw_client=fake)
        result = client.generate_research("prompt", ResearchResultSchema)
        assert result.grounding == (GroundingChunkInfo("https://example.test/a"),)
        assert result.url_context_urls == ()


class TestToolWiring:
    def test_google_search_tool_enabled_by_default(self):
        fake = _grounded_fake()
        client = make_client(_config(), raw_client=fake)
        client.generate_research("prompt", ResearchResultSchema)
        tools = fake.calls[0][2].tools
        assert len(tools) == 1
        assert tools[0].google_search is not None

    def test_url_context_tool_added_when_enabled(self):
        fake = _grounded_fake()
        client = make_client(_config(), raw_client=fake)
        client.generate_research("prompt", ResearchResultSchema, enable_url_context=True)
        tools = fake.calls[0][2].tools
        assert len(tools) == 2
        assert tools[0].google_search is not None
        assert tools[1].url_context is not None

    def test_system_prompt_and_temperature_forwarded(self):
        fake = _grounded_fake()
        client = make_client(_config(), raw_client=fake)
        client.generate_research("prompt", ResearchResultSchema, system_prompt="be rigorous", temperature=0.3)
        config = fake.calls[0][2]
        assert config.system_instruction == "be rigorous"
        assert config.temperature == 0.3


class TestErrorsAndRetries:
    def test_malformed_json_raises_provider_error(self):
        fake = FakeRawClient()
        fake.add_result("not json at all")
        client = make_client(_config(), raw_client=fake)
        with pytest.raises(GeminiProviderError):
            client.generate_research("prompt", ResearchResultSchema)

    def test_schema_violation_raises_validation_error(self):
        fake = FakeRawClient()
        fake.add_result_json(
            {
                "sources": [
                    {
                        "url": GROUNDED_URL,
                        "title": "T",
                        "publisher": "P",
                        "source_type": "paper",
                        "credibility_tier": "random_blog",  # not in Literal
                        "claims": [{"claim_text": "A claim."}],
                    }
                ]
            }
        )
        client = make_client(_config(), raw_client=fake)
        with pytest.raises(GeminiSchemaValidationError):
            client.generate_research("prompt", ResearchResultSchema)

    def test_auth_failure_raises_auth_error(self):
        from google.genai import errors as genai_errors

        fake = FakeRawClient()
        fake.add_error(genai_errors.ClientError(code=401, response_json={}))
        client = make_client(_config(), raw_client=fake)
        with pytest.raises(GeminiAuthenticationError):
            client.generate_research("prompt", ResearchResultSchema)

    def test_transient_server_error_retries_then_succeeds(self):
        from google.genai import errors as genai_errors

        fake = FakeRawClient()
        fake.add_error(genai_errors.ServerError(code=503, response_json={}))
        fake.outcomes.append(grounded_result(valid_research_payload(), chunks=GROUNDED_CHUNKS))
        client = make_client(_config(max_retries=2), raw_client=fake)
        result = client.generate_research("prompt", ResearchResultSchema)
        assert isinstance(result, GroundedStructured)
        assert fake.attempt_count == 2

    def test_server_error_exhausts_retries(self):
        from google.genai import errors as genai_errors

        fake = FakeRawClient()
        for _ in range(4):
            fake.add_error(genai_errors.ServerError(code=503, response_json={}))
        client = make_client(_config(max_retries=2), raw_client=fake)
        with pytest.raises(GeminiAPIError):
            client.generate_research("prompt", ResearchResultSchema)
        assert fake.attempt_count == 3  # 1 + max_retries bounded

    def test_timeout_raises_timeout_error(self):
        import httpx

        fake = FakeRawClient()
        fake.add_error(httpx.TimeoutException("late"))
        client = make_client(_config(max_retries=0), raw_client=fake)
        with pytest.raises(GeminiTimeoutError):
            client.generate_research("prompt", ResearchResultSchema)

    def test_unexpected_failure_raises_unexpected_error(self):
        fake = FakeRawClient()
        fake.add_error(RuntimeError("boom"))
        client = make_client(_config(), raw_client=fake)
        with pytest.raises(GeminiUnexpectedError):
            client.generate_research("prompt", ResearchResultSchema)


def test_no_raw_sdk_objects_escape():
    fake = _grounded_fake()
    client = make_client(_config(), raw_client=fake)
    result = client.generate_research("prompt", ResearchResultSchema)
    assert isinstance(result.data, ResearchResultSchema)
    assert all(isinstance(item, GroundingChunkInfo) for item in result.grounding)
    assert isinstance(result.url_context_urls, tuple)
