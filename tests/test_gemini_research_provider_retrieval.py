"""Offline tests for the GeminiResearchProvider retrieval path (Firecrawl).

The provider must: use the injected retriever's evidence as the only allowed
source set, drop model-proposed URLs not in it, stay bounded, and keep the
grounding path byte-for-byte untouched. No network, no SDK, no key.
"""

from __future__ import annotations

import socket

import pytest

from agents.research.agent import ResearchAgent
from agents.research.provider import TopicDescriptor
from agents.research.retriever import MockResearchRetriever, RetrievedSource
from providers.gemini.config import GeminiConfig
from providers.gemini.errors import GeminiProviderError, GeminiSchemaValidationError
from providers.gemini.research_provider import (
    PROMPT_VERSION,
    RETRIEVAL_PROMPT_VERSION,
    GeminiResearchProvider,
)
from tests.gemini_test_utils import FakeRawClient, make_client, valid_research_payload

PAPER = "https://example.test/papers/octopus"
NEWS = "https://example.test/news/octopus"

RETRIEVED = (
    RetrievedSource(
        url=PAPER,
        title="Retrieved paper title",
        publisher="example.test",
        source_type="peer_reviewed_paper",
        content="Octopuses have three hearts.",
        domain="example.test",
    ),
    RetrievedSource(
        url=NEWS,
        title="Retrieved news title",
        publisher="example.test",
        source_type="news_article",
        content="Octopus blood is blue.",
        domain="example.test",
    ),
)


def _topic() -> TopicDescriptor:
    return TopicDescriptor(
        topic_id="t_octopus_hearts",
        title="Why octopuses have three hearts",
        description="Test description",
        pillar="science",
        topic_type="surprising_fact",
    )


def _provider(fake: FakeRawClient, *, sources=RETRIEVED, **kwargs) -> GeminiResearchProvider:
    client = make_client(GeminiConfig(api_key="k"), raw_client=fake, sleep=lambda *_: None)
    retriever = MockResearchRetriever(list(sources))
    return GeminiResearchProvider(client=client, retriever=retriever, **kwargs)


def _structured_fake(payload=None, *, sources=()):
    fake = FakeRawClient()
    fake.add_result_json(payload if payload is not None else valid_research_payload(tuple(sources)))
    return fake


class TestRetrievalCandidates:
    def test_returns_only_retrieved_urls(self):
        fake = _structured_fake(payload=valid_research_payload((PAPER, NEWS, "https://invented.example/fake")), sources=(PAPER, NEWS))
        candidates = _provider(fake, sources=RETRIEVED).search(_topic())
        assert [c.url for c in candidates] == [PAPER, NEWS]
        assert all(c.claims for c in candidates)
        assert all("not invented" in c.notes for c in candidates)

    def test_retrieved_metadata_fills_empty_model_fields(self):
        payload = {
            "sources": [
                {
                    "url": PAPER,
                    "title": "",
                    "publisher": "",
                    "source_type": "",
                    "credibility_tier": "primary",
                    "claims": [{"claim_text": "Three hearts claim."}],
                }
            ]
        }
        fake = _structured_fake(payload=payload, sources=(PAPER,))
        (candidate,) = _provider(fake).search(_topic())
        assert candidate.title == "Retrieved paper title"
        assert candidate.publisher == "example.test"
        assert candidate.source_type == "web_page"
        assert candidate.credibility_tier == "primary"

    def test_duplicate_claims_collapse(self):
        payload = valid_research_payload((PAPER, NEWS))
        dup = dict(payload["sources"][0]["claims"][0])
        dup["claim_text"] = "Octopuses have three hearts: two branchial hearts and a systemic heart."
        payload["sources"][1]["claims"].append(dup)
        fake = _structured_fake(payload=payload, sources=(PAPER, NEWS))
        candidates = _provider(fake).search(_topic())
        texts = [claim.claim_text for candidate in candidates for claim in candidate.claims]
        assert len(texts) == 2
        assert len(set(t.lower() for t in texts)) == len(texts)


class TestDuplicateClaimMerging:
    """A duplicate statement is retained once and corroborates across sources.

    Provider-level coverage of the Defect-3 fix: previously the second
    occurrence was discarded outright, leaving the corroborating source with
    zero claims. PAPER and NEWS are both `example.test`, so same-registrable-
    domain merging is deliberately refused here; the cross-domain case lives in
    `tests/test_research_multi_source_merge.py`.
    """

    def test_duplicate_claim_is_retained_once(self):
        payload = valid_research_payload((PAPER, NEWS))
        dup = dict(payload["sources"][0]["claims"][0])
        dup["claim_text"] = "Octopuses have three hearts: two branchial hearts and a systemic heart."
        payload["sources"][1]["claims"].append(dup)
        fake = _structured_fake(payload=payload, sources=(PAPER, NEWS))
        candidates = _provider(fake).search(_topic())
        texts = [claim.claim_text for candidate in candidates for claim in candidate.claims]
        assert len(texts) == 2  # the shared statement appears exactly once

    def test_same_domain_duplicate_is_not_merged(self):
        """`example.test` and `example.test` are one registrable domain."""
        payload = valid_research_payload((PAPER, NEWS))
        dup = dict(payload["sources"][0]["claims"][0])
        payload["sources"][1]["claims"].append(dup)
        fake = _structured_fake(payload=payload, sources=(PAPER, NEWS))
        candidates = _provider(fake).search(_topic())
        for candidate in candidates:
            for claim in candidate.claims:
                assert claim.also_supported_by_urls == ()

    def test_cross_domain_duplicate_is_merged(self):
        other = RetrievedSource(
            url="https://news.test/octopus",
            title="Other outlet",
            publisher="news.test",
            source_type="news_article",
            content="Octopuses have three hearts.",
            domain="news.test",
        )
        sources = (RETRIEVED[0], other)
        payload = valid_research_payload((PAPER, other.url))
        dup = dict(payload["sources"][0]["claims"][0])
        payload["sources"][1]["claims"].append(dup)
        fake = _structured_fake(payload=payload, sources=(PAPER, other.url))
        candidates = _provider(fake, sources=sources).search(_topic())
        merged = [
            claim
            for candidate in candidates
            for claim in candidate.claims
            if claim.also_supported_by_urls
        ]
        assert len(merged) == 1
        owner = next(
            c
            for c in candidates
            if any(claim is merged[0] for claim in c.claims)
        )
        assert owner.url == PAPER
        assert list(merged[0].also_supported_by_urls) == [other.url]

    def test_merge_end_to_end_reaches_the_agent(self):
        """The merged URL must become a real second source_id in the package."""
        from agents.research.models import source_id_for

        other = RetrievedSource(
            url="https://news.test/octopus",
            title="Other outlet",
            publisher="news.test",
            source_type="news_article",
            content="Octopuses have three hearts.",
            domain="news.test",
        )
        sources = (RETRIEVED[0], other)
        payload = valid_research_payload((PAPER, other.url))
        dup = dict(payload["sources"][0]["claims"][0])
        payload["sources"][1]["claims"].append(dup)
        fake = _structured_fake(payload=payload, sources=(PAPER, other.url))
        provider = _provider(fake, sources=sources)
        package = ResearchAgent(provider, run_id="merge").research(
            {
                "data": {
                    "selected_topic_id": "t_octopus_hearts",
                    "selected_topic": {
                        "topic_id": "t_octopus_hearts",
                        "title": "Why octopuses have three hearts",
                    },
                }
            }
        )
        multi = [c for c in package.claims if len(c.source_ids) == 2]
        assert len(multi) == 1
        assert set(multi[0].source_ids) == {source_id_for(PAPER), source_id_for(other.url)}
        # The corroborating source keeps its own claims and gains the shared one.
        assert multi[0] in package.claims_for_source(source_id_for(other.url))


class TestEvidenceBoundary:
    def test_empty_retrieval_returns_no_candidates_and_skips_model(self):
        fake = FakeRawClient()
        provider = _provider(fake, sources=[])
        assert provider.search(_topic()) == []
        assert fake.calls == []  # the model is never called without evidence

    def test_model_url_outside_retrieval_is_dropped(self):
        fake = _structured_fake(
            payload=valid_research_payload((PAPER, "https://other.example/fake")),
            sources=(PAPER,),
        )
        candidates = _provider(fake).search(_topic())
        assert [c.url for c in candidates] == [PAPER]

    def test_retrieval_sliced_to_max_retrieved_sources(self):
        extra = RetrievedSource(
            url="https://example.test/extra",
            title="Extra",
            content="Extra content",
            domain="example.test",
        )
        fake = _structured_fake(
            payload=valid_research_payload((RETRIEVED[0].url, RETRIEVED[1].url, extra.url)),
            sources=(RETRIEVED[0], RETRIEVED[1], extra),
        )
        provider = _provider(fake, sources=RETRIEVED + (extra,), max_retrieved_sources=2)
        candidates = provider.search(_topic())
        assert [c.url for c in candidates] == [PAPER, NEWS]
        prompt = fake.calls[0][1]
        assert PAPER in prompt and NEWS in prompt
        assert extra.url not in prompt

    def test_bounded_agent_package_ready_for_fact_check(self, minimal_selection):
        fake = _structured_fake(payload=valid_research_payload((PAPER, NEWS)), sources=(PAPER, NEWS))
        provider = _provider(fake)
        package = ResearchAgent(provider, run_id="fr").research(
            minimal_selection("t_octopus_hearts", "Three hearts")
        )
        package.produced_by = dict(provider.produced_by)
        package.validate()
        assert package.research_status == "ready_for_fact_check"
        assert {source.url for source in package.sources} == {PAPER, NEWS}
        assert all("not invented" in source.notes for source in package.sources)

    def test_sockets_blocked_full_offline_run(self, monkeypatch, minimal_selection):
        def _blocked(*args, **kwargs):
            raise OSError("network blocked in tests")

        monkeypatch.setattr(socket, "socket", _blocked)
        fake = _structured_fake(payload=valid_research_payload((PAPER,)), sources=(PAPER,))
        package = ResearchAgent(_provider(fake), run_id="offfr").research(
            minimal_selection("t_octopus_hearts", "Three hearts")
        )
        package.validate()
        assert package.research_status == "ready_for_fact_check"


class TestProvenanceAndPrompt:
    def test_retrieval_produced_by_includes_retriever(self):
        provider = _provider(FakeRawClient())
        assert provider.produced_by == {
            "prompt_version": RETRIEVAL_PROMPT_VERSION,
            "provider": "gemini",
            "model": provider.model,
            "retrieved_by": "mock",
        }
        assert provider.prompt_version == RETRIEVAL_PROMPT_VERSION
        assert provider.retriever_name == "mock"

    def test_grounding_produced_by_is_unchanged(self):
        bare = GeminiResearchProvider(
            client=make_client(GeminiConfig(api_key="k"), raw_client=FakeRawClient(), sleep=lambda *_: None)
        )
        assert bare.produced_by == {
            "prompt_version": PROMPT_VERSION,
            "provider": "gemini",
            "model": bare.model,
        }
        assert "retrieved_by" not in bare.produced_by
        assert bare.retriever_name is None

    def test_rendered_prompt_injects_retrieved_content_and_topic(self):
        fake = FakeRawClient()
        provider = _provider(fake)
        prompt = provider._render_retrieval_prompt(_topic(), list(RETRIEVED))
        assert PAPER in prompt and NEWS in prompt
        assert "Octopuses have three hearts." in prompt
        assert "Why octopuses have three hearts" in prompt
        assert "{retrieved_sources}" not in prompt

    def test_missing_retrieval_prompt_template_raises(self):
        fake = _structured_fake(payload=valid_research_payload((PAPER,)), sources=(PAPER,))
        provider = _provider(fake, retrieval_prompt_path="prompts/missing/r.txt")
        with pytest.raises(FileNotFoundError):
            provider.search(_topic())


class TestSchemaStrictness:
    def test_invalid_credibility_tier_raises_schema_error(self):
        payload = valid_research_payload((PAPER,))
        payload["sources"][0]["credibility_tier"] = "random_blog"
        fake = _structured_fake(payload=payload, sources=(PAPER,))
        with pytest.raises(GeminiSchemaValidationError):
            _provider(fake).search(_topic())

    def test_malformed_json_raises_provider_error(self):
        fake = FakeRawClient()
        fake.add_result("this is not json")
        with pytest.raises(GeminiProviderError):
            _provider(fake).search(_topic())


class TestMultiSourceDeclaration:
    """`also_supported_by` must be grounded in real retrieval evidence."""

    @staticmethod
    def _payload(paper_extras):
        return {
            "sources": [
                {
                    "url": PAPER,
                    "title": "",
                    "publisher": "",
                    "source_type": "",
                    "credibility_tier": "primary",
                    "claims": [
                        {
                            "claim_text": "Octopuses have three hearts.",
                            "also_supported_by": paper_extras,
                        }
                    ],
                },
                {
                    "url": NEWS,
                    "title": "",
                    "publisher": "",
                    "source_type": "",
                    "credibility_tier": "primary",
                    "claims": [{"claim_text": "Octopus blood is blue."}],
                },
            ]
        }

    def _extras(self, paper_extras):
        fake = _structured_fake(payload=self._payload(paper_extras), sources=(PAPER, NEWS))
        candidates = _provider(fake, sources=RETRIEVED).search(_topic())
        return candidates[0].claims[0].also_supported_by_urls

    def test_retrieved_url_is_kept(self):
        assert NEWS in self._extras([NEWS])

    def test_url_outside_retrieval_evidence_is_dropped(self):
        assert self._extras(["https://invented.example/fake"]) == ()

    def test_mixed_list_keeps_only_grounded_urls(self):
        assert self._extras([NEWS, "https://invented.example/fake"]) == (NEWS,)

    def test_duplicate_extras_are_deduplicated(self):
        assert self._extras([NEWS, NEWS]) == (NEWS,)

    def test_absent_field_defaults_to_empty(self):
        payload = self._payload([])
        del payload["sources"][0]["claims"][0]["also_supported_by"]
        fake = _structured_fake(payload=payload, sources=(PAPER, NEWS))
        candidates = _provider(fake, sources=RETRIEVED).search(_topic())
        assert candidates[0].claims[0].also_supported_by_urls == ()

    def test_grounded_declaration_binds_two_sources_end_to_end(self):
        """The whole point: a real two-source claim reaches Stage 3."""
        from agents.seo.result import TopicSelection

        fake = _structured_fake(payload=self._payload([NEWS]), sources=(PAPER, NEWS))
        provider = _provider(fake, sources=RETRIEVED)
        agent = ResearchAgent(provider, run_id="r1")
        selection = TopicSelection(
            selection_id="s1",
            selected_topic_id="t_octopus_hearts",
            selected_topic={"title": "Why octopuses have three hearts"},
            ranked_candidates=[],
            selection_mode="manual",
            historical_data_available=False,
            selection_reason="test",
            scoring_configuration={},
            created_at="2026-01-01T00:00:00+00:00",
        )
        package = agent.research(selection)
        claim = next(c for c in package.claims if c.claim_text == "Octopuses have three hearts.")
        assert len(claim.source_ids) == 2
        package.validate(traceability=True)