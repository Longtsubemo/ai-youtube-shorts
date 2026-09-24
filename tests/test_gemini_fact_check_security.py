"""Security regression tests for the Stage-3 Gemini fact-check slice.

Non-negotiable properties:
- the GEMINI_API_KEY never reaches prompts, exceptions, metadata, or the
  FactCheckResult artifact;
- model-generated text/URLs/sources are never evidence (verbatim
  containment + source binding + no URL/source-id output fields);
- only typed GeminiProviderError exceptions escape the providers;
- source attribution is derived deterministically, never model-proposed.
"""

from __future__ import annotations

import pytest

from agents.qa.fact_checker.checker import FactChecker
from agents.qa.fact_checker.models import evidence_id_for
from agents.research.retriever import MockResearchRetriever, RetrievedSource
from fact_check_test_utils import make_claim, make_evidence, make_package, make_source
from providers.gemini.config import GeminiConfig
from providers.gemini.errors import GeminiProviderError
from providers.gemini.evidence_provider import GeminiEvidenceProvider
from providers.gemini.verifier_provider import GeminiVerifier
from tests.gemini_test_utils import FakeRawClient, FakeResponse, make_client, noop_sleep

KEY = "sk-secret-fc-0001"
PAPER = "https://example.test/papers/octopus"
CONTENT = "Octopuses have three hearts: two branchial hearts and a systemic heart."


def _client(fake: FakeRawClient):
    return make_client(GeminiConfig(api_key=KEY), raw_client=fake, sleep=noop_sleep)


def _evidence_provider(fake: FakeRawClient) -> GeminiEvidenceProvider:
    retrieved = RetrievedSource(
        url=PAPER,
        title="Why octopuses have three hearts",
        content=CONTENT,
        domain="example.test",
    )
    return GeminiEvidenceProvider(
        client=_client(fake),
        retriever=MockResearchRetriever([retrieved]),
    )


def _verifier(fake: FakeRawClient) -> GeminiVerifier:
    return GeminiVerifier(client=_client(fake))


def _source():
    return make_source(url=PAPER, tier="primary")


class TestKeyNeverLeaks:
    def test_key_not_in_evidence_prompt(self):
        prompt = _evidence_provider(FakeRawClient())._render_prompt(_source(), [])
        assert KEY not in prompt

    def test_key_not_in_verifier_prompt(self):
        verifier = _verifier(FakeRawClient())
        claim = make_claim(source_urls=[PAPER])
        prompt = verifier._render_prompt(claim, [], {})
        assert KEY not in prompt

    def test_key_not_in_produced_by_or_artifact(self):
        source = _source()
        claim = make_claim("Octopuses have three hearts.", source_urls=[PAPER])
        package = make_package(claims=[claim], sources=[source], run_id="fc_sec")
        excerpt = "Octopuses have three hearts."
        ev_id = evidence_id_for(source.source_id, excerpt)
        fake = FakeRawClient()
        fake.add_result_json(
            {
                "excerpts": [
                    {
                        "excerpt": excerpt,
                        "locator": "p1",
                        "evidence_type": "direct_statement",
                        "notes": "",
                    }
                ]
            }
        )
        fake.add_result_json(
            {
                "claim_id": claim.claim_id,
                "support_level": "direct",
                "stronger_than_evidence": False,
                "supporting_evidence_ids": [ev_id],
                "contradiction_evidence_ids": [],
                "insufficient_evidence_ids": [],
            }
        )
        result = FactChecker(
            _evidence_provider(fake),
            verifier=_verifier(fake),
            run_id="sec-run",
        ).check(package)
        artifact = result.to_json()
        assert KEY not in artifact
        assert "api_key" not in artifact
        assert result.produced_by["provider"] == "gemini"
        assert result.produced_by["model"] == result.provenance.model


class TestEvidenceIsNeverModelFabrication:
    def test_invented_quote_is_not_evidence(self):
        fake = FakeRawClient()
        fake.add_result_json(
            {
                "excerpts": [
                    {
                        "excerpt": "A confident statement that never appeared in the page.",
                        "locator": "p1",
                        "evidence_type": "direct_statement",
                        "notes": "",
                    }
                ]
            }
        )
        evidence = _evidence_provider(fake).get_evidence(_source())
        assert evidence == []

    def test_evidence_has_no_url_or_source_field(self):
        fake = FakeRawClient()
        fake.add_result_json(
            {
                "excerpts": [
                    {
                        "excerpt": CONTENT.split(":")[0] + ".",
                        "locator": "p1",
                        "evidence_type": "direct_statement",
                        "notes": "",
                    }
                ]
            }
        )
        (item,) = _evidence_provider(fake).get_evidence(_source())
        payload = item.to_dict()
        assert "url" not in payload
        assert "source_id" in payload
        assert payload["source_id"] == _source().source_id

    def test_invented_evidence_id_is_rejected(self):
        source = _source()
        claim = make_claim(source_urls=[PAPER])
        package = make_package(claims=[claim], sources=[source], run_id="fc_sec")
        fake = FakeRawClient()
        fake.add_result_json(
            {
                "excerpts": [
                    {
                        "excerpt": "Octopuses have three hearts.",
                        "locator": "p1",
                        "evidence_type": "direct_statement",
                        "notes": "",
                    }
                ]
            }
        )
        fake.add_result_json(
            {
                "claim_id": claim.claim_id,
                "support_level": "direct",
                "stronger_than_evidence": False,
                "supporting_evidence_ids": ["ev_fabricated_999"],
                "contradiction_evidence_ids": [],
                "insufficient_evidence_ids": [],
            }
        )
        checker = FactChecker(_evidence_provider(fake), verifier=_verifier(fake))
        with pytest.raises(GeminiProviderError):
            checker.check(package)


class TestNoRawSdkEscape:
    def test_evidence_api_error_is_typed(self):
        fake = FakeRawClient()
        fake.add_error(RuntimeError("boom"))
        with pytest.raises(GeminiProviderError):
            _evidence_provider(fake).get_evidence(_source())

    def test_verifier_api_error_is_typed(self):
        fake = FakeRawClient()
        fake.add_error(RuntimeError("boom"))
        with pytest.raises(GeminiProviderError):
            _verifier(fake).verify(make_claim(), [], {})


class TestSourceAttributionIsDeterministic:
    def test_sources_derived_from_supplied_evidence(self):
        source = _source()
        claim = make_claim(source_urls=[PAPER])
        evidence = [make_evidence(source, "Octopuses have three hearts.", "direct_statement")]
        ev_id = evidence[0].evidence_id
        fake = FakeRawClient()
        fake.add_result_json(
            {
                "claim_id": claim.claim_id,
                "support_level": "direct",
                "stronger_than_evidence": False,
                "supporting_evidence_ids": [ev_id],
                "contradiction_evidence_ids": [],
                "insufficient_evidence_ids": [],
            }
        )
        assessment = _verifier(fake).verify(claim, evidence, {source.source_id: source})
        assert assessment.supporting_source_ids == (source.source_id,)