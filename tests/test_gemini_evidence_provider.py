"""Offline tests for GeminiEvidenceProvider (Stage-3 evidence extraction).

The provider must: bind evidence ONLY to the package source, quote only
verbatim (whitespace/punctuation-normalized) retrieved content, never invent
sources/URLs, stay bounded, and map every SDK failure to the typed hierarchy.
No network, no SDK, no key.
"""

from __future__ import annotations

import socket

import pytest

from agents.research.retriever import MockResearchRetriever, RetrievedSource
from fact_check_test_utils import make_source
from providers.gemini.config import GeminiConfig
from providers.gemini.errors import (
    GeminiProviderError,
    GeminiSchemaValidationError,
)
from providers.gemini.evidence_provider import (
    EVIDENCE_NOTE_PREFIX,
    PROMPT_VERSION,
    GeminiEvidenceProvider,
)
from tests.gemini_test_utils import FakeRawClient, make_client, noop_sleep

PAPER = "https://example.test/papers/octopus"
NEWS = "https://example.test/news/octopus"

CONTENT = (
    "Octopuses have three hearts: two branchial hearts and a systemic heart. "
    "Their blood uses copper-based hemocyanin, which makes it appear blue."
)

RETRIEVED_PAPER = RetrievedSource(
    url=PAPER,
    title="Why octopuses have three hearts",
    publisher="example.test",
    source_type="peer_reviewed_paper",
    content=CONTENT,
    domain="example.test",
)


def _provider(fake: FakeRawClient, *, sources=None, **kwargs) -> GeminiEvidenceProvider:
    client = make_client(GeminiConfig(api_key="k"), raw_client=fake, sleep=noop_sleep)
    retriever = MockResearchRetriever(list(sources) if sources is not None else [RETRIEVED_PAPER])
    return GeminiEvidenceProvider(client=client, retriever=retriever, **kwargs)


def _source(url=PAPER, tier="primary"):
    return make_source(url=url, tier=tier)


def _ev_payload(excerpts=("Octopuses have three hearts.",)):
    return {
        "excerpts": [
            {
                "excerpt": excerpt,
                "locator": f"loc:{index}",
                "evidence_type": "direct_statement",
                "notes": "",
            }
            for index, excerpt in enumerate(excerpts)
        ]
    }


def _structured_fake(payload=None):
    fake = FakeRawClient()
    fake.add_result_json(payload if payload is not None else _ev_payload())
    return fake


class TestEvidenceExtraction:
    def test_extracts_grounded_evidence_bound_to_source(self):
        fake = _structured_fake()
        source = _source()
        evidence = _provider(fake).get_evidence(source)
        assert len(evidence) == 1
        item = evidence[0]
        assert item.source_id == source.source_id
        assert item.excerpt == "Octopuses have three hearts."
        assert item.locator == "loc:0"
        assert item.evidence_type == "direct_statement"
        assert item.retrieved_at == source.retrieved_at
        assert EVIDENCE_NOTE_PREFIX in item.notes

    def test_evidence_type_from_model_is_preserved(self):
        payload = {
            "excerpts": [
                {
                    "excerpt": "Their blood uses copper-based hemocyanin.",
                    "locator": "p2",
                    "evidence_type": "supporting_context",
                    "notes": "",
                }
            ]
        }
        fake = _structured_fake(payload)
        (item,) = _provider(fake).get_evidence(_source())
        assert item.evidence_type == "supporting_context"
        assert "Their blood uses copper-based hemocyanin." in item.excerpt

    def test_punctuation_edge_difference_still_grounded(self):
        # Excerpt drops the trailing period that exists in the page text.
        payload = _ev_payload(excerpts=("Octopuses have three hearts",))
        fake = _structured_fake(payload)
        (item,) = _provider(fake).get_evidence(_source())
        assert item.excerpt == "Octopuses have three hearts"

    def test_duplicate_excerpts_collapse_to_one(self):
        fake = _structured_fake(
            _ev_payload(excerpts=("Octopuses have three hearts.", "Octopuses have three hearts."))
        )
        evidence = _provider(fake).get_evidence(_source())
        assert len(evidence) == 1

    def test_bounded_by_max_evidence_per_source(self):
        fake = _structured_fake(
            _ev_payload(excerpts=("Octopuses have three hearts.", "Their blood uses copper."))
        )
        provider = _provider(fake, max_evidence_per_source=1)
        evidence = provider.get_evidence(_source())
        assert len(evidence) == 1

    def test_excerpt_not_in_retrieved_content_is_dropped(self):
        fake = _structured_fake(_ev_payload(excerpts=("This fact was never in the page.",)))
        evidence = _provider(fake).get_evidence(_source())
        assert evidence == []


class TestEvidenceBoundary:
    def test_no_retriever_is_no_evidence_and_model_not_called(self):
        client = make_client(GeminiConfig(api_key="k"), raw_client=FakeRawClient(), sleep=noop_sleep)
        provider = GeminiEvidenceProvider(client=client)
        assert provider.get_evidence(_source()) == []
        assert client._raw_client.calls == []

    def test_empty_retrieval_returns_no_evidence_and_model_not_called(self):
        fake = FakeRawClient()
        provider = _provider(fake, sources=[])
        assert provider.get_evidence(_source()) == []
        assert fake.calls == []

    def test_retrieved_url_not_matching_source_returns_no_evidence(self):
        fake = FakeRawClient()
        other = RetrievedSource(
            url=NEWS,
            title="Unrelated",
            content="Octopus blood is blue.",
            domain="example.test",
        )
        provider = _provider(fake, sources=[other])
        assert provider.get_evidence(_source(PAPER)) == []
        assert fake.calls == []

    def test_url_normalization_matches_trailing_slash_variant(self):
        trailing = RetrievedSource(
            url=PAPER + "/",
            title="Trailing slash",
            content=CONTENT,
            domain="example.test",
        )
        fake = _structured_fake()
        evidence = _provider(fake, sources=[trailing]).get_evidence(_source(PAPER))
        assert evidence, "normalized URL match should produce evidence"

    def test_multiple_matching_pages_are_all_valid_content(self):
        second = RetrievedSource(
            url=PAPER,
            title="Duplicate hit",
            content="An entirely different page text about octopus hearts.",
            domain="example.test",
        )
        payload = _ev_payload(excerpts=("An entirely different page text about octopus hearts.",))
        fake = _structured_fake(payload)
        (item,) = _provider(fake, sources=[RETRIEVED_PAPER, second]).get_evidence(_source())
        assert item.excerpt.startswith("An entirely different")


class TestSchemaStrictness:
    def test_invalid_evidence_type_raises_schema_error(self):
        payload = {
            "excerpts": [
                {
                    "excerpt": "Octopuses have three hearts.",
                    "locator": "p1",
                    "evidence_type": "fabricated_type",
                    "notes": "",
                }
            ]
        }
        fake = _structured_fake(payload)
        with pytest.raises(GeminiSchemaValidationError):
            _provider(fake).get_evidence(_source())

    def test_empty_locator_raises_schema_error(self):
        payload = {
            "excerpts": [
                {
                    "excerpt": "Octopuses have three hearts.",
                    "locator": "",
                    "evidence_type": "direct_statement",
                    "notes": "",
                }
            ]
        }
        fake = _structured_fake(payload)
        with pytest.raises(GeminiSchemaValidationError):
            _provider(fake).get_evidence(_source())

    def test_malformed_json_raises_provider_error(self):
        fake = FakeRawClient()
        fake.add_result("this is not json")
        with pytest.raises(GeminiProviderError):
            _provider(fake).get_evidence(_source())

    def test_api_error_maps_to_provider_error(self):
        fake = FakeRawClient()
        fake.add_error(RuntimeError("boom"))
        with pytest.raises(GeminiProviderError):
            _provider(fake).get_evidence(_source())


class TestProvenanceAndPrompt:
    def test_produced_by_metadata(self):
        provider = _provider(FakeRawClient())
        assert provider.provider_name == "gemini"
        assert provider.produced_by == {
            "prompt_version": PROMPT_VERSION,
            "provider": "gemini",
            "model": provider.model,
        }
        assert provider.retriever_name == "mock"
        assert provider.prompt_version == PROMPT_VERSION

    def test_rendered_prompt_injects_source_and_content(self):
        fake = _structured_fake()
        source = _source()
        prompt = _provider(fake)._render_prompt(source, [RETRIEVED_PAPER])
        assert PAPER in prompt
        assert source.source_id in prompt
        assert CONTENT in prompt
        assert "{retrieved_sources}" not in prompt
        assert "{max_evidence}" not in prompt

    def test_missing_prompt_template_raises(self):
        fake = _structured_fake()
        with pytest.raises(FileNotFoundError):
            _provider(fake, prompt_path="prompts/missing/evidence.txt").get_evidence(_source())


class TestOfflineOnly:
    def test_sockets_blocked_full_run(self, monkeypatch):
        def _blocked(*args, **kwargs):
            raise OSError("network blocked in tests")

        monkeypatch.setattr(socket, "socket", _blocked)
        fake = _structured_fake()
        evidence = _provider(fake).get_evidence(_source())
        assert evidence
        assert fake.calls  # only the injected fake was used, no real network