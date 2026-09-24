"""Offline tests for GeminiVerifier (Stage-3 claim judge).

The verifier must: limit its inputs to {claim, supplied evidence, source
metadata}, derive every source-level field deterministically, reject any
evidence id outside the supplied set, and map SDK/parse failures to the typed
hierarchy. No network, no SDK, no key.
"""

from __future__ import annotations

import pytest

from agents.research.models import FactClaim
from fact_check_test_utils import make_claim, make_evidence, make_source
from providers.gemini.config import GeminiConfig
from providers.gemini.errors import (
    GeminiProviderError,
    GeminiSchemaValidationError,
)
from providers.gemini.verifier_provider import (
    PROMPT_VERSION,
    GeminiVerifier,
)
from tests.gemini_test_utils import FakeRawClient, make_client, noop_sleep

PAPER = "https://example.test/papers/octopus"
NEWS = "https://example.test/news/octopus"


def _sources():
    primary = make_source(url=PAPER, tier="primary")
    discovery = make_source(url=NEWS, tier="discovery")
    return {primary.source_id: primary, discovery.source_id: discovery}


def _evidence():
    sources = _sources()
    primary = sources[[sid for sid in sources if sources[sid].url == PAPER][0]]
    discovery = sources[[sid for sid in sources if sources[sid].url == NEWS][0]]
    ev1 = make_evidence(primary, "Octopuses have three hearts.", "direct_statement")
    ev2 = make_evidence(primary, "Their blood appears blue.", "supporting_context")
    ev3 = make_evidence(discovery, "Newspaper report mentions three hearts.", "direct_statement")
    return [ev1, ev2, ev3], sources


def _claim(claim_text="Octopuses have three hearts.", source_urls=(PAPER,), **overrides):
    return make_claim(claim_text=claim_text, source_urls=source_urls, **overrides)


def _verifier(fake: FakeRawClient) -> GeminiVerifier:
    client = make_client(GeminiConfig(api_key="k"), raw_client=fake, sleep=noop_sleep)
    return GeminiVerifier(client=client)


def _payload(
    claim,
    *,
    support_level="direct",
    supporting=(),
    contradiction=(),
    insufficient=(),
    stronger=False,
):
    return {
        "claim_id": claim.claim_id,
        "support_level": support_level,
        "stronger_than_evidence": stronger,
        "supporting_evidence_ids": list(supporting),
        "contradiction_evidence_ids": list(contradiction),
        "insufficient_evidence_ids": list(insufficient),
    }


def _structured_fake(payload):
    fake = FakeRawClient()
    fake.add_result_json(payload)
    return fake


def _verify(fake, claim=None, evidence=None, sources=None):
    evidence = evidence if evidence is not None else _evidence()[0]
    sources = sources if sources is not None else _evidence()[1]
    claim = claim or _claim()
    return _verifier(fake).verify(claim, evidence, sources)


class TestSupported:
    def test_direct_support_uses_primary_source(self):
        ev, sources = _evidence()
        claim = _claim()
        fake = _structured_fake(_payload(claim, supporting=(ev[0].evidence_id,)))
        assessment = _verify(fake, claim, ev, sources)
        assert assessment.support_level == "direct"
        assert assessment.supporting_evidence_ids == (ev[0].evidence_id,)
        assert assessment.supporting_source_ids == (ev[0].source_id,)
        assert assessment.contradiction_present is False
        assert assessment.discovery_only is False
        assert assessment.evidence_ids == tuple(sorted({item.evidence_id for item in ev}))

    def test_partial_label_propagates(self):
        ev, sources = _evidence()
        claim = _claim()
        fake = _structured_fake(_payload(claim, support_level="partial", supporting=(ev[0].evidence_id,)))
        assessment = _verify(fake, claim, ev, sources)
        assert assessment.support_level == "partial"

    def test_supporting_ids_win_over_conflicting_label(self):
        ev, sources = _evidence()
        claim = _claim()
        fake = _structured_fake(_payload(claim, support_level="none", supporting=(ev[0].evidence_id,)))
        assessment = _verify(fake, claim, ev, sources)
        assert assessment.support_level == "partial"

    def test_confident_label_without_supporting_ids_downgraded(self):
        ev, sources = _evidence()
        claim = _claim()
        fake = _structured_fake(_payload(claim, support_level="direct"))
        assessment = _verify(fake, claim, ev, sources)
        assert assessment.support_level == "none"

    def test_none_label_with_no_ids(self):
        ev, sources = _evidence()
        claim = _claim()
        fake = _structured_fake(_payload(claim, support_level="none"))
        assessment = _verify(fake, claim, ev, sources)
        assert assessment.support_level == "none"

    def test_insufficient_label(self):
        ev, sources = _evidence()
        claim = _claim()
        fake = _structured_fake(_payload(claim, support_level="insufficient"))
        assessment = _verify(fake, claim, ev, sources)
        assert assessment.support_level == "insufficient"


class TestContradiction:
    def test_credible_contradiction_overrides_support(self):
        ev, sources = _evidence()
        claim = _claim()
        fake = _structured_fake(
            _payload(
                claim,
                support_level="contradictory",
                supporting=(ev[0].evidence_id,),
                contradiction=(ev[1].evidence_id,),
            )
        )
        assessment = _verify(fake, claim, ev, sources)
        assert assessment.support_level == "contradictory"
        assert assessment.contradiction_present is True
        assert ev[1].source_id in assessment.contradiction_source_ids

    def test_discovery_tier_contradiction_is_not_credible(self):
        ev, sources = _evidence()
        discovery_ev = ev[2]
        claim = _claim(source_urls=(NEWS,))
        fake = _structured_fake(
            _payload(claim, support_level="contradictory", contradiction=(discovery_ev.evidence_id,))
        )
        assessment = _verify(fake, claim, [discovery_ev], sources)
        assert assessment.contradiction_source_ids == ()
        assert assessment.contradiction_present is False
        assert assessment.support_level == "none"

    def test_insufficient_evidence_ids_recorded(self):
        ev, sources = _evidence()
        claim = _claim()
        fake = _structured_fake(
            _payload(claim, support_level="insufficient", insufficient=(ev[1].evidence_id,))
        )
        assessment = _verify(fake, claim, ev, sources)
        assert assessment.support_level == "insufficient"
        assert ev[1].source_id in assessment.insufficient_source_ids


class TestDeterministicFields:
    def test_speculation_claim_is_not_fact(self):
        ev, sources = _evidence()
        claim = _claim(claim_text="Octopuses might feel happy.", claim_type="speculation")
        fake = _structured_fake(_payload(claim, supporting=(ev[0].evidence_id,)))
        assessment = _verify(fake, claim, ev, sources)
        assert assessment.speculation_as_fact is True

    def test_fact_claim_is_not_speculation(self):
        ev, sources = _evidence()
        claim = _claim()
        fake = _structured_fake(_payload(claim, supporting=(ev[0].evidence_id,)))
        assessment = _verify(fake, claim, ev, sources)
        assert assessment.speculation_as_fact is False

    def test_discovery_only_detected(self):
        ev, sources = _evidence()
        discovery_ev = ev[2]
        claim = _claim(source_urls=(NEWS,))
        fake = _structured_fake(_payload(claim, supporting=(discovery_ev.evidence_id,)))
        assessment = _verify(fake, claim, [discovery_ev], sources)
        assert assessment.discovery_only is True
        assert assessment.support_level == "direct"

    def test_stronger_than_evidence_needs_supporting_ids(self):
        ev, sources = _evidence()
        claim = _claim()
        fake = _structured_fake(_payload(claim, supporting=(ev[0].evidence_id,), stronger=True))
        assessment = _verify(fake, claim, ev, sources)
        assert assessment.stronger_than_evidence is True

    def test_stronger_flagged_but_no_support_ignored(self):
        ev, sources = _evidence()
        claim = _claim()
        fake = _structured_fake(_payload(claim, stronger=True))
        assessment = _verify(fake, claim, ev, sources)
        assert assessment.stronger_than_evidence is False


class TestSchemaStrictness:
    def test_unknown_evidence_id_rejected(self):
        ev, sources = _evidence()
        claim = _claim()
        fake = _structured_fake(_payload(claim, supporting=("ev_fabricated",)))
        with pytest.raises(GeminiSchemaValidationError):
            _verify(fake, claim, ev, sources)

    def test_claim_id_mismatch_rejected(self):
        ev, sources = _evidence()
        claim = _claim()
        payload = _payload(claim, supporting=(ev[0].evidence_id,))
        payload["claim_id"] = "cl_some_other_claim"
        with pytest.raises(GeminiSchemaValidationError):
            _verify(_structured_fake(payload), claim, ev, sources)

    def test_invalid_support_level_raises_schema_error(self):
        ev, sources = _evidence()
        claim = _claim()
        payload = _payload(claim, supporting=(ev[0].evidence_id,))
        payload["support_level"] = "SUPPORTED"
        with pytest.raises(GeminiSchemaValidationError):
            _verify(_structured_fake(payload), claim, ev, sources)

    def test_malformed_json_raises_provider_error(self):
        fake = FakeRawClient()
        fake.add_result("nope not json")
        with pytest.raises(GeminiProviderError):
            _verify(fake)

    def test_api_error_maps_to_provider_error(self):
        fake = FakeRawClient()
        fake.add_error(RuntimeError("boom"))
        with pytest.raises(GeminiProviderError):
            _verify(fake)


class TestProvenanceAndPrompt:
    def test_produced_by_metadata(self):
        verifier = _verifier(FakeRawClient())
        assert verifier.provider_name == "gemini"
        assert verifier.prompt_version == PROMPT_VERSION
        assert verifier.produced_by == {
            "prompt_version": PROMPT_VERSION,
            "provider": "gemini",
            "model": verifier.model,
        }

    def test_rendered_prompt_is_limited_to_claim_evidence_sources(self):
        ev, sources = _evidence()
        claim = _claim()
        prompt = _verifier(FakeRawClient())._render_prompt(claim, ev, sources)
        assert claim.claim_id in prompt
        assert claim.claim_text in prompt
        assert ev[0].evidence_id in prompt
        assert ev[0].excerpt in prompt
        assert sources[ev[0].source_id].url in prompt
        assert "{evidence_items}" not in prompt and "{sources_metadata}" not in prompt

    def test_missing_prompt_template_raises(self):
        ev, sources = _evidence()
        claim = _claim()
        fake = FakeRawClient()
        fake.add_result_json(_payload(claim))
        verifier = GeminiVerifier(
            make_client(GeminiConfig(api_key="k"), raw_client=fake, sleep=noop_sleep),
            prompt_path="prompts/missing/verifier.txt",
        )
        with pytest.raises(FileNotFoundError):
            verifier.verify(claim, ev, sources)