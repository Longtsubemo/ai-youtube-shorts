"""Offline tests for GeminiResearchProvider (evidence-boundary behaviour).

The key safety guarantee under test: ONLY URLs the retrieval metadata
actually returned become sources. Model-proposed URLs without grounding are
dropped, no grounding at all yields no candidates (honest
`insufficient_evidence`), and claims are deduplicated so packages validate.
"""

from __future__ import annotations

import socket
import subprocess
import sys

import pytest

from agents.research.agent import ResearchAgent
from agents.research.provider import TopicDescriptor
from providers.gemini.config import GeminiConfig
from providers.gemini.errors import (
    GeminiAPIError,
    GeminiProviderError,
    GeminiSchemaValidationError,
)
from providers.gemini.research_provider import (
    PROMPT_VERSION,
    GeminiResearchProvider,
)
from tests.gemini_test_utils import (
    FakeRawClient,
    grounded_result,
    make_client,
    valid_research_payload,
)

PAPER_URL = "https://example.test/papers/octopus"
NEWS_URL = "https://example.test/news/octopus"
PAPER_CHUNK = (PAPER_URL, "Grounded paper title", "example.test")
NEWS_CHUNK = (NEWS_URL, "Grounded news title", "example.test")


def _topic() -> TopicDescriptor:
    return TopicDescriptor(
        topic_id="t_octopus_hearts",
        title="Why octopuses have three hearts",
        description="Test description",
        pillar="science",
        topic_type="surprising_fact",
    )


def _provider(fake: FakeRawClient, **kwargs) -> GeminiResearchProvider:
    client = make_client(GeminiConfig(api_key="k"), raw_client=fake, sleep=lambda *_: None)
    return GeminiResearchProvider(client=client, **kwargs)


def _ground_fake(payload=None, *, chunks=None, url_context_urls=()):
    fake = FakeRawClient()
    fake.outcomes.append(
        grounded_result(
            payload if payload is not None else valid_research_payload((PAPER_URL, NEWS_URL)),
            chunks=chunks if chunks is not None else [PAPER_CHUNK, NEWS_CHUNK],
            url_context_urls=url_context_urls,
        )
    )
    return fake


class TestGroundedCandidates:
    def test_search_returns_source_candidates(self):
        provider = _provider(_ground_fake())
        candidates = provider.search(_topic())
        assert [c.url for c in candidates] == [PAPER_URL, NEWS_URL]
        assert all(c.credibility_tier in ("primary", "high_quality_secondary", "discovery") for c in candidates)
        assert all(c.claims for c in candidates)
        assert all("not invented" in c.notes for c in candidates)

    def test_chunk_title_fills_empty_source_title(self):
        fake = _ground_fake()
        fake.outcomes[0] = grounded_result(
            {
                "sources": [
                    {
                        "url": PAPER_URL,
                        "publisher": "",
                        "source_type": "",
                        "credibility_tier": "primary",
                        "claims": [{"claim_text": "Grounded claim."}],
                    }
                ]
            },
            chunks=[PAPER_CHUNK],
        )
        provider = _provider(fake)
        (candidate,) = provider.search(_topic())
        assert candidate.title == "Grounded paper title"
        assert candidate.publisher == "example.test"
        assert candidate.source_type == "web_page"
        assert candidate.credibility_tier == "primary"

    def test_non_iso_publication_date_dropped(self):
        payload = valid_research_payload((PAPER_URL,))
        payload["sources"][0]["publication_date"] = "some time in May"
        fake = _ground_fake(payload, chunks=[PAPER_CHUNK])
        provider = _provider(fake)
        (candidate,) = provider.search(_topic())
        assert candidate.publication_date == ""


class TestEvidenceBoundary:
    def test_ungrounded_model_source_is_dropped(self):
        payload = valid_research_payload((PAPER_URL, "https://invented.example/fake"))
        fake = _ground_fake(payload, chunks=[PAPER_CHUNK])
        provider = _provider(fake)
        candidates = provider.search(_topic())
        assert [c.url for c in candidates] == [PAPER_URL]

    def test_url_context_urls_count_as_evidence(self):
        payload = valid_research_payload((NEWS_URL,))
        fake = _ground_fake(payload, chunks=[], url_context_urls=[NEWS_URL])
        provider = _provider(fake, enable_url_context=True)
        candidates = provider.search(_topic())
        assert [c.url for c in candidates] == [NEWS_URL]

    def test_no_grounding_yields_no_candidates(self):
        fake = FakeRawClient()
        fake.outcomes.append(grounded_result(valid_research_payload((PAPER_URL,))))
        provider = _provider(fake)
        assert provider.search(_topic()) == []

    def test_research_agent_lands_on_insufficient_evidence_without_grounding(self, minimal_selection):
        fake = FakeRawClient()
        fake.outcomes.append(grounded_result(valid_research_payload((PAPER_URL,))))
        provider = _provider(fake)
        package = ResearchAgent(provider, run_id="dng").research(minimal_selection("t_octopus_hearts", "Three hearts"))
        assert package.research_status == "insufficient_evidence"
        assert package.sources == ()
        assert package.claims == ()

    def test_grounded_only_package_is_ready_for_fact_check(self, minimal_selection):
        provider = _provider(_ground_fake())
        package = ResearchAgent(provider, run_id="gr").research(minimal_selection("t_octopus_hearts", "Three hearts"))
        package.produced_by = dict(provider.produced_by)
        package.validate()
        assert package.research_status == "ready_for_fact_check"
        assert {source.url for source in package.sources} == {PAPER_URL, NEWS_URL}
        assert all("not invented" in source.notes for source in package.sources)

    def test_duplicate_claim_text_collapses_to_one(self):
        payload = valid_research_payload((PAPER_URL, NEWS_URL))
        duplicate = dict(payload["sources"][0]["claims"][0])
        duplicate["claim_text"] = "Octopuses have three hearts: two branchial hearts and a systemic heart."
        payload["sources"][1]["claims"].append(duplicate)
        provider = _provider(_ground_fake(payload, chunks=[PAPER_CHUNK, NEWS_CHUNK]))
        candidates = provider.search(_topic())
        texts = [claim.claim_text for candidate in candidates for claim in candidate.claims]
        assert len(texts) == 2  # the duplicate three-hearts claim collapses into one
        assert len(set(t.lower() for t in texts)) == len(texts)
        assert texts.count("Octopuses have three hearts: two branchial hearts and a systemic heart.") == 1


class TestSchemaStrictness:
    def test_invalid_credibility_tier_raises_schema_error(self):
        payload = valid_research_payload((PAPER_URL,))
        payload["sources"][0]["credibility_tier"] = "random_blog"
        fake = _ground_fake(payload, chunks=[PAPER_CHUNK])
        with pytest.raises(GeminiSchemaValidationError):
            _provider(fake).search(_topic())

    def test_invalid_date_kind_raises_schema_error(self):
        payload = valid_research_payload((PAPER_URL,))
        payload["sources"][0]["claims"][0]["dates"] = [{"date_kind": "publish_on", "date": "2020-01-01"}]
        fake = _ground_fake(payload, chunks=[PAPER_CHUNK])
        with pytest.raises(GeminiSchemaValidationError):
            _provider(fake).search(_topic())

    def test_unknown_suggested_claim_type_raises_schema_error(self):
        payload = valid_research_payload((PAPER_URL,))
        payload["sources"][0]["claims"][0]["suggested_claim_type"] = "gossip"
        fake = _ground_fake(payload, chunks=[PAPER_CHUNK])
        with pytest.raises(GeminiSchemaValidationError):
            _provider(fake).search(_topic())

    def test_malformed_output_raises_provider_error(self):
        fake = FakeRawClient()
        fake.add_result("this is not json")
        with pytest.raises(GeminiProviderError):
            _provider(fake).search(_topic())

    def test_api_error_maps_to_provider_error(self):
        from google.genai import errors as genai_errors

        fake = FakeRawClient()
        fake.add_error(genai_errors.ServerError(code=503, response_json={}))
        with pytest.raises(GeminiAPIError):
            _provider(fake).search(_topic())


class TestProvenanceAndWiring:
    def test_produced_by_metadata(self):
        provider = _provider(_ground_fake())
        assert provider.produced_by == {
            "prompt_version": PROMPT_VERSION,
            "provider": "gemini",
            "model": provider.model,
        }
        assert provider.model

    def test_prompt_rendering_injects_topic(self):
        provider = _provider(_ground_fake())
        prompt = provider._render_prompt(_topic())
        assert "{title}" not in prompt and "Why octopuses have three hearts" in prompt
        assert "science" in prompt and "surprising_fact" in prompt and "t_octopus_hearts" in prompt

    def test_missing_prompt_template_raises(self):
        provider = _provider(_ground_fake(), prompt_path="prompts/missing/prompt.txt")
        with pytest.raises(FileNotFoundError):
            provider.search(_topic())

    def test_sockets_blocked_full_offline_run(self, monkeypatch, minimal_selection):
        """No network: even socket creation is blocked during the run."""

        def _blocked(*args, **kwargs):
            raise OSError("network blocked in tests")

        monkeypatch.setattr(socket, "socket", _blocked)
        provider = _provider(_ground_fake())
        package = ResearchAgent(provider, run_id="off").research(minimal_selection("t_octopus_hearts", "Three hearts"))
        package.validate()
        assert package.research_status == "ready_for_fact_check"


class TestImportIsolation:
    def test_module_import_never_loads_sdk(self):
        """Importing the provider must not import the Google SDK (subprocess)."""
        code = (
            "import sys\n"
            "import providers.gemini.research_provider  # noqa: F401\n"
            "assert 'google.genai' not in sys.modules, 'google.genai imported transitively'\n"
            "print('ok')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=".")
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ok"