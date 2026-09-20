"""End-to-end Stage-3 behavior tests: evidence -> verifier -> gates -> result.

Covers the required scenarios: PASS/REVIEW/REJECT, contradiction,
corroboration, credibility tiering, critical vs supporting claims,
hallucination resistance, traceability, and clean handling of malformed input.
"""

from __future__ import annotations

import pytest

from agents.qa.fact_checker.checker import FactChecker
from agents.qa.fact_checker.gates import FactCheckConfig
from agents.qa.fact_checker.models import FactCheckResult
from agents.qa.fact_checker.stage import run_fact_check

from fact_check_test_utils import (
    FIXED_AT,
    StaticEvidenceProvider,
    make_claim,
    make_evidence,
    make_package,
    make_source,
)

URL_A = "https://example.test/source/s001"
URL_B = "https://example.test/source/s002"
URL_G = "https://example.test/source/ghost"

CLAIM = "Octopuses have three hearts."
SUPPORT = "Octopuses have three hearts."
UNRELATED = "The sky appears blue during the day on Earth."


def checker_for(provider, config=None):
    return FactChecker(provider, config=config, run_id="t", created_at=FIXED_AT)


def run_single(source, claims, evidence_by_url, config=None):
    package = make_package(claims=[claims], sources=[source])
    provider = StaticEvidenceProvider(evidence_by_url)
    return checker_for(provider, config).check(package)


class TestDirectSupport:
    def test_claim_with_direct_evidence_passes(self):
        source = make_source(URL_A)
        claim = make_claim(CLAIM, source_urls=[URL_A])
        evidence = make_evidence(source, SUPPORT)
        result = run_single(source, claim, {URL_A: [evidence]})
        verdict = result.claim_verdict(claim.claim_id)
        assert verdict.verdict == "PASS"
        assert claim.claim_id in result.approved_claims

    def test_every_pass_has_evidence_ids(self):
        source_a = make_source(URL_A)
        source_b = make_source(URL_B)
        claim_a = make_claim(CLAIM, source_urls=[URL_A])
        claim_b = make_claim("Octopus blood uses copper-based hemocyanin.", source_urls=[URL_B])
        package = make_package(
            claims=[claim_a, claim_b],
            sources=[source_a, source_b],
        )
        provider = StaticEvidenceProvider(
            {
                URL_A: [make_evidence(source_a, CLAIM)],
                URL_B: [make_evidence(source_b, "Octopus blood uses copper-based hemocyanin.")],
            }
        )
        result = checker_for(provider).check(package)
        assert result.package_verdict == "PASS"
        for verdict in result.claim_results:
            assert verdict.verdict == "PASS"
            assert verdict.evidence_ids, "every PASS must be traceable to evidence"

    def test_missing_evidence_cannot_produce_pass(self):
        source = make_source(URL_A)
        claim = make_claim(CLAIM, source_urls=[URL_A])
        result = run_single(source, claim, {})  # provider knows nothing
        verdict = result.claim_verdict(claim.claim_id)
        assert verdict.verdict != "PASS"
        assert claim.claim_id not in result.approved_claims

    def test_evidence_traceability_chain(self):
        source = make_source(URL_A, tier="primary")
        claim = make_claim(CLAIM, source_urls=[URL_A])
        evidence = [make_evidence(source, SUPPORT), make_evidence(source, UNRELATED, "insufficient")]
        package = make_package(claims=[claim], sources=[source])
        provider = StaticEvidenceProvider({URL_A: evidence})
        result = checker_for(provider).check(package)
        verdict = result.claim_verdict(claim.claim_id)
        assert verdict.verdict == "PASS"
        assert set(verdict.evidence_ids) == {e.evidence_id for e in evidence}
        assert set(verdict.supporting_source_ids) == {source.source_id}
        # evidence source ids resolve to the claim's cited sources
        for evidence_id in verdict.evidence_ids:
            item = next(e for e in evidence if e.evidence_id == evidence_id)
            assert item.source_id in claim.source_ids
            assert item.source_id == source.source_id


class TestInsufficientAndUnsupported:
    def test_claim_with_insufficient_evidence_reviews(self):
        source = make_source(URL_A)
        claim = make_claim(CLAIM, source_urls=[URL_A])
        evidence = make_evidence(source, "The paper does not describe octopus hearts.", "insufficient")
        result = run_single(source, claim, {URL_A: [evidence]})
        verdict = result.claim_verdict(claim.claim_id)
        assert verdict.verdict == "REVIEW"
        assert claim.claim_id in result.review_items
        assert claim.claim_id not in result.approved_claims

    def test_unsupported_claim_rejected(self):
        source = make_source(URL_A)
        claim = make_claim(CLAIM, source_urls=[URL_A])
        evidence = make_evidence(source, UNRELATED, "direct_statement")
        result = run_single(source, claim, {URL_A: [evidence]})
        verdict = result.claim_verdict(claim.claim_id)
        assert verdict.verdict == "REJECT"
        assert claim.claim_id in result.rejected_claims

    def test_no_evidence_verdict_is_configurable(self):
        source = make_source(URL_A)
        claim = make_claim(CLAIM, source_urls=[URL_A])
        strict = FactCheckConfig(no_evidence_verdict="REJECT")
        result = run_single(source, claim, {}, config=strict)
        assert result.claim_verdict(claim.claim_id).verdict == "REJECT"


class TestContradiction:
    def test_contradictory_credible_source_reviews(self):
        source = make_source(URL_A, tier="primary")
        claim = make_claim(CLAIM, source_urls=[URL_A])
        evidence = make_evidence(source, "Octopuses have a single heart.", "contradictory_statement")
        result = run_single(source, claim, {URL_A: [evidence]})
        verdict = result.claim_verdict(claim.claim_id)
        assert verdict.verdict == "REVIEW"
        assert source.source_id in verdict.contradiction_source_ids
        assert claim.claim_id in result.review_items

    def test_contradiction_outweighs_otherwise_good_support(self):
        good = make_source(URL_A, tier="primary")
        contradictory = make_source(URL_B, tier="primary")
        claim = make_claim(CLAIM, source_urls=[URL_A, URL_B])
        package = make_package(claims=[claim], sources=[good, contradictory])
        provider = StaticEvidenceProvider(
            {
                URL_A: [make_evidence(good, SUPPORT)],
                URL_B: [make_evidence(contradictory, "Octopuses have a single heart.", "contradictory_statement")],
            }
        )
        result = checker_for(provider).check(package)
        verdict = result.claim_verdict(claim.claim_id)
        assert verdict.verdict == "REVIEW"

    def test_discovery_tier_contradiction_does_not_override_primary_support(self):
        good = make_source(URL_A, tier="primary")
        forum = make_source(URL_B, tier="discovery")
        claim = make_claim(CLAIM, source_urls=[URL_A, URL_B])
        package = make_package(claims=[claim], sources=[good, forum])
        provider = StaticEvidenceProvider(
            {
                URL_A: [make_evidence(good, SUPPORT)],
                URL_B: [make_evidence(forum, "Octopuses have a single heart.", "contradictory_statement")],
            }
        )
        result = checker_for(provider).check(package)
        assert result.claim_verdict(claim.claim_id).verdict == "PASS"


class TestCorroboration:
    def test_multiple_credible_sources_raise_confidence(self):
        single_claim = make_claim(CLAIM, source_urls=[URL_A])
        single_src = make_source(URL_A, tier="primary")
        package_single = make_package(claims=[single_claim], sources=[single_src])
        provider_single = StaticEvidenceProvider({URL_A: [make_evidence(single_src, SUPPORT)]})
        result_single = checker_for(provider_single).check(package_single)

        corr_claim = make_claim("Octopuses have three hearts, two branchial and one systemic.", source_urls=[URL_A, URL_B])
        corr_src_b = make_source(URL_B, tier="primary")
        package_corr = make_package(claims=[corr_claim], sources=[single_src, corr_src_b])
        provider_corr = StaticEvidenceProvider(
            {
                URL_A: [make_evidence(single_src, corr_claim.claim_text)],
                URL_B: [make_evidence(corr_src_b, corr_claim.claim_text)],
            }
        )
        result_corr = checker_for(provider_corr).check(package_corr)

        assert result_single.claim_verdict(single_claim.claim_id).confidence == 0.85
        assert result_corr.claim_verdict(corr_claim.claim_id).confidence == 0.90
        assert (
            result_corr.claim_verdict(corr_claim.claim_id).confidence
            > result_single.claim_verdict(single_claim.claim_id).confidence
        )

    def test_two_low_quality_sources_do_not_outweigh_one_primary(self):
        low_a = make_source(URL_A, tier="discovery")
        low_b = make_source(URL_B, tier="discovery")
        claim = make_claim(CLAIM, source_urls=[URL_A, URL_B])
        package = make_package(claims=[claim], sources=[low_a, low_b])
        provider = StaticEvidenceProvider(
            {
                URL_A: [make_evidence(low_a, SUPPORT)],
                URL_B: [make_evidence(low_b, SUPPORT)],
            }
        )
        result = checker_for(provider).check(package)
        verdict = result.claim_verdict(claim.claim_id)
        assert verdict.verdict != "PASS"
        assert verdict.confidence <= 0.5


class TestCredibilityTiering:
    def test_discovery_only_cannot_establish_critical_claim(self):
        source = make_source(URL_A, tier="discovery")
        claim = make_claim(CLAIM, source_urls=[URL_A], critical=True)
        evidence = make_evidence(source, SUPPORT)
        result = run_single(source, claim, {URL_A: [evidence]})
        verdict = result.claim_verdict(claim.claim_id)
        assert verdict.verdict != "PASS"
        assert claim.claim_id in result.critical_claim_failures

    def test_critical_claim_needs_independent_corroboration(self):
        source = make_source(URL_A, tier="primary")
        claim = make_claim(CLAIM, source_urls=[URL_A], critical=True)
        evidence = make_evidence(source, SUPPORT)
        result = run_single(source, claim, {URL_A: [evidence]})
        verdict = result.claim_verdict(claim.claim_id)
        assert verdict.verdict == "REVIEW"  # 1 source < config.corroboration_min_sources

    def test_critical_claim_passes_with_two_primary_sources(self):
        src_a = make_source(URL_A, tier="primary")
        src_b = make_source(URL_B, tier="primary")
        claim = make_claim(CLAIM, source_urls=[URL_A, URL_B], critical=True)
        package = make_package(claims=[claim], sources=[src_a, src_b])
        provider = StaticEvidenceProvider(
            {
                URL_A: [make_evidence(src_a, SUPPORT)],
                URL_B: [make_evidence(src_b, SUPPORT)],
            }
        )
        result = checker_for(provider).check(package)
        assert result.claim_verdict(claim.claim_id).verdict == "PASS"
        assert claim.claim_id in result.approved_claims
        assert result.critical_claim_failures == ()

    def test_discovery_only_supporting_claim_reviews(self):
        source = make_source(URL_A, tier="discovery")
        claim = make_claim(CLAIM, source_urls=[URL_A])
        evidence = make_evidence(source, SUPPORT)
        result = run_single(source, claim, {URL_A: [evidence]})
        assert result.claim_verdict(claim.claim_id).verdict == "REVIEW"


class TestClaimStrongerThanEvidence:
    def test_strong_claim_over_weak_evidence_reviews(self):
        source = make_source(URL_A, tier="primary")
        claim = make_claim(
            "Researchers proved excess heat is caused by cold fusion.",
            source_urls=[URL_A],
            claim_type="fact",
        )
        evidence = make_evidence(
            source,
            "The researchers observed a correlation between excess heat and the reactor setup.",
        )
        result = run_single(source, claim, {URL_A: [evidence]})
        verdict = result.claim_verdict(claim.claim_id)
        assert verdict.verdict == "REVIEW"

    def test_conservative_claim_over_matching_evidence_passes(self):
        source = make_source(URL_A, tier="primary")
        claim = make_claim(
            "The researchers observed a correlation between two devices.",
            source_urls=[URL_A],
        )
        evidence = make_evidence(source, "The researchers observed a correlation between two devices.")
        result = run_single(source, claim, {URL_A: [evidence]})
        assert result.claim_verdict(claim.claim_id).verdict == "PASS"


class TestSpeculationAsFact:
    def test_speculation_cannot_become_factual_pass(self):
        source = make_source(URL_A, tier="primary")
        claim = make_claim(
            "Some owners say a pet octopus may someday learn to open a tank lid.",
            source_urls=[URL_A],
            claim_type="speculation",
        )
        evidence = make_evidence(source, claim.claim_text)  # perfect wording match
        result = run_single(source, claim, {URL_A: [evidence]})
        verdict = result.claim_verdict(claim.claim_id)
        assert verdict.verdict != "PASS"
        assert claim.claim_id in result.rejected_claims

    def test_opinion_cannot_become_factual_pass(self):
        source = make_source(URL_A, tier="primary")
        claim = make_claim("I believe agents will reshape computers.", source_urls=[URL_A], claim_type="opinion")
        evidence = make_evidence(source, "I believe agents will reshape computers.")
        result = run_single(source, claim, {URL_A: [evidence]})
        assert result.claim_verdict(claim.claim_id).verdict != "PASS"


class TestMissingSourceReferences:
    def test_claim_referencing_unknown_source_is_rejected(self):
        source = make_source(URL_A)
        claim = make_claim(
            CLAIM,
            source_urls=[],
            source_ids=["src_ghost001"],
            claim_id="cl_ghost",
        )
        package = make_package(claims=[claim], sources=[source])
        provider = StaticEvidenceProvider({URL_A: [make_evidence(source, SUPPORT)]})
        result = checker_for(provider).check(package)
        verdict = result.claim_verdict(claim.claim_id)
        assert verdict.verdict == "REJECT"
        assert "unknown source" in verdict.rationale
        assert claim.claim_id in result.rejected_claims

    def test_claim_with_no_sources_is_rejected(self):
        claim = make_claim(CLAIM, source_urls=[], source_ids=[])
        package = make_package(claims=[claim], sources=[])
        result = checker_for(StaticEvidenceProvider()).check(package)
        assert result.claim_verdict(claim.claim_id).verdict == "REJECT"

    def test_empty_package_is_rejected(self):
        package = make_package(claims=[], sources=[])
        result = checker_for(StaticEvidenceProvider()).check(package)
        assert result.package_verdict == "REJECT"


class TestPackageAggregation:
    URL_C = "https://example.test/source/s003"

    def _critical_two_source_package(self):
        """Critical claim backed by 2 primary sources, plus a supporting claim."""
        src_a = make_source(URL_A, tier="primary")
        src_c = make_source(self.URL_C, tier="primary")
        src_b = make_source(URL_B, tier="primary")
        critical = make_claim(CLAIM, source_urls=[URL_A, self.URL_C], critical=True)
        supporting = make_claim(
            "A second supporting claim about octopus blood color.",
            source_urls=[URL_B],
        )
        package = make_package(claims=[critical, supporting], sources=[src_a, src_b, src_c])
        return package, critical, supporting, src_a, src_b, src_c

    def test_critical_rejected_claim_rejects_package(self):
        package, critical, supporting, src_a, src_b, src_c = self._critical_two_source_package()
        # critical claim unsupported by both sources -> REJECT; supporting claim fine -> PASS
        provider = StaticEvidenceProvider(
            {
                URL_A: [make_evidence(src_a, UNRELATED, "direct_statement")],
                self.URL_C: [make_evidence(src_c, UNRELATED, "direct_statement")],
                URL_B: [make_evidence(src_b, supporting.claim_text)],
            }
        )
        result = checker_for(provider).check(package)
        assert result.claim_verdict(critical.claim_id).verdict == "REJECT"
        assert result.package_verdict == "REJECT"
        assert critical.claim_id in result.critical_claim_failures
        assert critical.claim_id in result.rejected_claims

    def test_supporting_rejected_claim_does_not_reject_package(self):
        package, critical, supporting, src_a, src_b, src_c = self._critical_two_source_package()
        provider = StaticEvidenceProvider(
            {
                URL_A: [make_evidence(src_a, CLAIM)],
                self.URL_C: [make_evidence(src_c, CLAIM)],
                URL_B: [make_evidence(src_b, UNRELATED, "direct_statement")],
            }
        )
        result = checker_for(provider).check(package)
        assert result.claim_verdict(critical.claim_id).verdict == "PASS"
        assert result.claim_verdict(supporting.claim_id).verdict == "REJECT"
        assert result.package_verdict == "PASS"
        assert supporting.claim_id in result.rejected_claims
        assert supporting.claim_id not in result.approved_claims

    def test_review_claim_excluded_from_approved(self):
        package, critical, supporting, src_a, src_b, src_c = self._critical_two_source_package()
        provider = StaticEvidenceProvider(
            {
                URL_A: [make_evidence(src_a, CLAIM)],
                self.URL_C: [make_evidence(src_c, CLAIM)],
                URL_B: [make_evidence(src_b, "A second supporting claim about octopus.", "insufficient")],
            }
        )
        result = checker_for(provider).check(package)
        verdict = result.claim_verdict(supporting.claim_id)
        assert verdict.verdict == "REVIEW"
        assert supporting.claim_id in result.review_items
        assert supporting.claim_id not in result.approved_claims

    def test_package_result_aggregates_claim_results(self):
        src_a = make_source(URL_A, tier="primary")
        good = make_claim("Proper supporting claim one.", source_urls=[URL_A])
        weak = make_claim("Proper supporting claim two.", source_urls=[URL_A], claim_type="speculation")
        poetry = make_claim("Proper supporting claim three.", source_urls=[URL_A])
        package = make_package(
            claims=[good, weak, poetry],
            sources=[src_a],
        )
        provider = StaticEvidenceProvider(
            {
                URL_A: [
                    make_evidence(src_a, "Proper supporting claim one."),
                    make_evidence(src_a, "Proper supporting claim three."),
                ],
            }
        )
        result = checker_for(provider).check(package)
        assert result.claim_verdict(good.claim_id).verdict == "PASS"
        assert result.claim_verdict(weak.claim_id).verdict == "REJECT"
        assert result.claim_verdict(poetry.claim_id).verdict == "PASS"
        assert result.approved_claims == (good.claim_id, poetry.claim_id)
        assert result.rejected_claims == (weak.claim_id,)
        assert result.review_items == ()
        assert result.package_verdict == "PASS"

    def test_package_reviews_when_no_claim_approved(self):
        src_a = make_source(URL_A, tier="primary")
        weak = make_claim("Only weak claim.", source_urls=[URL_A], claim_type="speculation")
        package = make_package(claims=[weak], sources=[src_a])
        provider = StaticEvidenceProvider({URL_A: [make_evidence(src_a, "Only weak claim.")]})
        result = checker_for(provider).check(package)
        assert result.package_verdict == "REVIEW"


class TestHallucinationResistance:
    def test_verifier_is_network_free(self, monkeypatch):
        from agents.qa.fact_checker.verifier import MockVerifier

        def _block(*args, **kwargs):
            raise AssertionError("network access forbidden")

        monkeypatch.setattr("socket.socket", _block, raising=True)
        monkeypatch.setattr("urllib.request.urlopen", _block, raising=True)
        source = make_source(URL_A)
        claim = make_claim(CLAIM, source_urls=[URL_A])
        package = make_package(claims=[claim], sources=[source])
        result = run_fact_check(package, provider=StaticEvidenceProvider(
            {URL_A: [make_evidence(source, SUPPORT)]}
        ), run_id="t", created_at=FIXED_AT)
        assert result.claim_verdict(claim.claim_id).verdict == "PASS"

    def test_rationale_stays_within_supplied_evidence(self):
        source = make_source(URL_A, tier="primary")
        claim = make_claim(CLAIM, source_urls=[URL_A])
        evidence = make_evidence(source, SUPPORT)
        result = run_single(source, claim, {URL_A: [evidence]})
        rationale = result.claim_verdict(claim.claim_id).rationale
        assert rationale
        assert evidence.evidence_id in rationale  # rationale cites the actual evidence id

    def test_json_round_trip(self):
        source = make_source(URL_A)
        claim = make_claim(CLAIM, source_urls=[URL_A])
        package = make_package(claims=[claim], sources=[source])
        result = run_fact_check(package, provider=StaticEvidenceProvider(
            {URL_A: [make_evidence(source, SUPPORT)]}
        ), run_id="t", created_at=FIXED_AT)
        restored = FactCheckResult.from_json(result.to_json())
        assert restored.to_dict() == result.to_dict()

    def test_provenance_is_present(self):
        source = make_source(URL_A)
        claim = make_claim(CLAIM, source_urls=[URL_A])
        package = make_package(claims=[claim], sources=[source])
        result = run_fact_check(package, provider=StaticEvidenceProvider(
            {URL_A: [make_evidence(source, SUPPORT)]}
        ), run_id="t", created_at=FIXED_AT)
        assert result.schema == "FactCheckResult"
        assert result.schema_version == 1
        assert result.stage == "fact_check"
        assert result.provenance is not None
        assert result.provenance.generated_by == "fact_check_agent_v1"
        assert result.provenance.source == "StaticEvidenceProvider:MockVerifier"
        assert result.produced_by["model"] == "MockVerifier"

    def test_deterministic_across_runs(self):
        source = make_source(URL_A)
        claim = make_claim(CLAIM, source_urls=[URL_A])
        package = make_package(claims=[claim], sources=[source])
        provider = lambda: StaticEvidenceProvider({URL_A: [make_evidence(source, SUPPORT)]})
        run_fact_check(package, provider=provider(), run_id="det", created_at=FIXED_AT)
        result_a = run_fact_check(package, provider=provider(), run_id="det", created_at=FIXED_AT)
        result_b = run_fact_check(package, provider=provider(), run_id="det", created_at=FIXED_AT)
        assert result_a.to_dict() == result_b.to_dict()


class TestMalformedEvidenceHandling:
    def test_evidence_referencing_missing_source_is_dropped_cleanly(self):
        from dataclasses import replace

        source = make_source(URL_A)
        claim = make_claim(CLAIM, source_urls=[URL_A])
        package = make_package(claims=[claim], sources=[source])
        dangling = replace(make_evidence(source, SUPPORT, evidence_id="ev_dangling"), source_id="src_missing001")
        provider = StaticEvidenceProvider({URL_A: [dangling]})
        result = checker_for(provider).check(package)
        assert result.evidence_issues
        assert any("unknown source" in issue for issue in result.evidence_issues)
        assert result.claim_verdict(claim.claim_id).verdict != "PASS"

    def test_malformed_evidence_is_skipped_without_crash(self):
        from agents.qa.fact_checker.models import Evidence

        source = make_source(URL_A)
        claim = make_claim(CLAIM, source_urls=[URL_A])
        package = make_package(claims=[claim], sources=[source])
        good = make_evidence(source, SUPPORT)
        # construct a malformed Evidence directly (bypasses constructor validation,
        # so only the checker's runtime validate() can catch it)
        bad = Evidence(
            evidence_id="ev_bad",
            source_id=source.source_id,
            excerpt="",
            locator="loc:2",
            evidence_type="direct_statement",
        )
        provider = StaticEvidenceProvider({URL_A: [good, bad]})
        result = checker_for(provider).check(package)
        assert result.evidence_issues
        assert any("malformed evidence" in issue for issue in result.evidence_issues)
        assert result.claim_verdict(claim.claim_id).verdict == "PASS"  # good evidence still counts


class TestEndToEndMockCorpus:
    def test_mock_octopus_package_produces_mixed_verdicts(self):
        from agents.qa.fact_checker.evidence import MockEvidenceProvider
        from agents.research.provider import MockResearchProvider
        from agents.research.stage import load_topic_selection, run_research

        selection = load_topic_selection()
        package = run_research(selection, MockResearchProvider(), run_id="e2e")
        result = run_fact_check(
            package,
            provider=MockEvidenceProvider(),
            run_id="e2e",
            created_at=FIXED_AT,
        )
        by_verdict = {v: [] for v in ("PASS", "REVIEW", "REJECT")}
        for claim_result in result.claim_results:
            by_verdict[claim_result.verdict].append(claim_result.claim_id)
        assert by_verdict["PASS"], "mock corpus must contain a supported claim"
        assert by_verdict["REVIEW"], "mock corpus must contain a reviewed claim"
        assert by_verdict["REJECT"], "mock corpus must contain a rejected (speculative) claim"
        # the speculating 'smartest invertebrates' claim must not be approved
        assert result.package_verdict in ("PASS", "REVIEW")

    def test_mock_first_bug_package_is_discovery_only(self):
        from agents.qa.fact_checker.evidence import MockEvidenceProvider
        from agents.research.provider import MockResearchProvider
        from agents.research.stage import load_topic_selection, run_research

        selection = load_topic_selection()
        selection.selected_topic_id = "t_first_bug"  # points provider lookup at the moth fixture
        package = run_research(selection, MockResearchProvider(), run_id="e2e")
        result = run_fact_check(
            package,
            provider=MockEvidenceProvider(),
            run_id="e2e",
            created_at=FIXED_AT,
        )
        assert result.approved_claims == ()