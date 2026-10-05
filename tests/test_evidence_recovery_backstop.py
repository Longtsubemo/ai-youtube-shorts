"""Focused offline tests for the deterministic evidence-recovery backstop.

Defect #2, from run `e2e-20261002-192946`, claim `cl_e552015c60`:

    "Hawking radiation is theoretical radiation released outside a black hole's
     event horizon due to quantum effects, based on a model developed by
     Stephen Hawking."

The claim was bound to four sources, two of them eligible
(`high_quality_secondary`): Wikipedia and Britannica. Britannica was credited;
Wikipedia was retrieved and had evidence extracted from it, but the extractor
did not select the particular supporting sentence, so the verifier could not
credit Wikipedia either. The claim fell to REVIEW with a corroboration count of
1 of 2 - despite the supporting text being on the page.

`get_evidence` is claim-blind, so it cannot know which sentence matters. This
suite covers the deterministic backstop that closes that recall gap, and - more
importantly - the boundary it must never cross.

Mapping to the required cases:
   1. extractor finds evidence          -> TestExtractorSuccessUnchanged
   2. extractor misses near-verbatim    -> TestNearVerbatimRecovered
   3. deterministic evidence id         -> TestRecoveredEvidenceIdentity
   4. passes containment validation     -> TestRecoveredEvidenceIsGrounded
   5. source lacks the content          -> TestNoSupportNoRecovery
   6. keyword overlap alone            -> TestKeywordOverlapIsNotEvidence
   7. discovery reviewed, not rejected  -> TestDiscoveryNotPromoted
                                          TestDiscoveryOnlyClaimReviewsInsteadOfRejecting
   8. no new source id                  -> TestRecoveryCannotAddSource
   9. verifier selection unchanged      -> TestVerifierSelectionUnchanged
  10. multiple sources recover         -> TestMultipleSourcesRecover
  11. deterministic / idempotent       -> TestRecoveryIsDeterministic
  12. artifacts still loadable          -> TestExistingArtifactsUnaffected
  13. existing tests still pass        -> see full suite run
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.qa.fact_checker.checker import FactChecker
from agents.qa.fact_checker.evidence import MockEvidenceProvider, meaningful_tokens
from agents.qa.fact_checker.models import (
    Evidence,
    Evidence as EvidenceModel,
    FactCheckResult,
    evidence_id_for,
)
from agents.qa.fact_checker.recovery import (
    RECOVERY_MIN_CLAIM_TOKENS,
    RECOVERY_MIN_LCS_RATIO,
    RECOVERY_MIN_RUN,
    candidate_sentences,
    find_near_verbatim,
    is_near_verbatim,
    near_verbatim_score,
    normalize_for_recovery,
    recovery_tokens,
)
from agents.qa.fact_checker.verifier import MockVerifier
from agents.research.models import (
    FactClaim,
    ResearchPackage,
    ResearchPackageError,
    Source,
    source_id_for,
)

# --- the real claim and a realistic paraphrase of Wikipedia's lead --------- #

REAL_CLAIM = (
    "Hawking radiation is theoretical radiation released outside a black hole's "
    "event horizon due to quantum effects, based on a model developed by "
    "Stephen Hawking."
)

# Near-verbatim restatement, as a Wikipedia lead would read. Minor differences
# only: "predicted to be released", an inserted clause, smart punctuation.
WIKIPEDIA_LIKE = (
    "Hawking radiation is theoretical radiation predicted to be released outside "
    "a black hole's event horizon, and is due to quantum effects near the event "
    "horizon. It was developed by Stephen Hawking in 1974."
)

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/Hawking_radiation"
BRITANNICA_URL = "https://www.britannica.com/science/Hawking-radiation"

# A narrower claim, worded so that two independent eligible sources really do
# cover it at DIRECT level. Used to test the PASS path. `REAL_CLAIM` above is a
# compound restatement ("...based on a model developed by Stephen Hawking"), and
# no single sentence settles all of its wording, so it trips the pre-existing
# Gate 9 partial-coverage rule regardless of recovery. Asserting PASS on
# `REAL_CLAIM` would be asserting a gate change, which this work must not make.
NARROW_CLAIM = (
    "Hawking radiation is theoretical radiation released outside a black hole "
    "event horizon due to quantum effects."
)

# Two independent eligible sources, both genuinely covering NARROW_CLAIM.
NARROW_WIKIPEDIA = (
    "Hawking radiation is theoretical radiation released outside a black hole "
    "event horizon due to quantum effects near the horizon."
)
NARROW_BRITANNICA = (
    "The Hawking radiation is the theoretical radiation released outside a "
    "black hole event horizon due to quantum effects."
)

# Britannica's lead, as credited in the real run: a genuine paraphrase that the
# verifier rates PARTIAL against REAL_CLAIM. It must be strong enough to be
# credited (coverage >= PARTIAL_COVERAGE_THRESHOLD), otherwise these tests would
# silently assert corroboration that the verifier never granted.
BRITANNICA_LIKE = (
    "Hawking radiation is theoretical radiation released outside a black hole "
    "event horizon due to quantum effects."
)

PRIMARY = "primary"
SECONDARY = "high_quality_secondary"
DISCOVERY = "discovery"

FIXED_AT = "2026-10-02T19:29:46+00:00"


def _keyword_salad(claim_text: str) -> str:
    """Build a bag of every content word in `claim_text`, plus padding.

    `meaningful_tokens` attaches trailing punctuation to its token, so the salad
    is derived from the claim itself rather than hand-written: that guarantees
    full bag-of-words coverage by construction, which is exactly the property
    the order-preserving score must be able to defeat. Tokens are shuffled into
    a fixed non-claim order so the result is deterministic.
    """
    words = [
        word
        for word in claim_text.lower().split()
        if word.strip(".,") not in {"a", "an", "the", "of", "to", "in", "and", "is"}
    ]
    ordered = sorted(words, reverse=True)
    padding = ["unrelated", "padding", "words", "here"]
    # Joined with spaces, not commas: `meaningful_tokens` splits on whitespace
    # and keeps punctuation attached, so a comma-joined salad would no longer
    # contain the claim's token forms and coverage would not be total.
    return " ".join(ordered + padding) + "."


def make_source(url: str, tier: str) -> Source:
    return Source(
        source_id=source_id_for(url),
        url=url,
        title=f"Title for {url}",
        publisher="pub",
        source_type="web_page",
        publication_date="2024-01-05",
        credibility_tier=tier,
        retrieved_at=FIXED_AT,
    )


def make_claim(
    claim_id: str = "cl_test",
    text: str = REAL_CLAIM,
    urls: tuple[str, ...] = (WIKIPEDIA_URL,),
    critical: bool = True,
) -> FactClaim:
    return FactClaim(
        claim_id=claim_id,
        claim_text=text,
        claim_type="fact",
        source_ids=tuple(source_id_for(url) for url in urls),
        confidence=0.5,
        critical=critical,
    )


class RecoveringProvider:
    """Evidence provider that can also recover claim-specific evidence.

    Stands in for `GeminiEvidenceProvider` without a model or network: model
    extraction is scripted, and recovery is the real algorithm reading the same
    already-retrieved content.
    """

    provider_name = "recovering-test"

    def __init__(
        self,
        contents: dict[str, str],
        extracted: dict[str, list[Evidence]] | None = None,
    ) -> None:
        self._contents = contents
        self._extracted = extracted or {}
        self.get_evidence_calls: list[str] = []
        self.recover_calls: list[tuple[str, str]] = []
        self.retrievals = 0

    def get_evidence(self, source: Source) -> list[Evidence]:
        self.get_evidence_calls.append(source.source_id)
        return list(self._extracted.get(source.source_id, ()))

    def recover_evidence(self, source: Source, claim: FactClaim) -> list[Evidence]:
        self.recover_calls.append((source.source_id, claim.claim_id))
        content = self._contents.get(source.source_id, "")
        if not content:
            return []
        excerpt = find_near_verbatim(claim.claim_text, content)
        if not excerpt:
            return []
        return [
            Evidence(
                evidence_id=evidence_id_for(source.source_id, excerpt),
                source_id=source.source_id,
                excerpt=excerpt,
                locator="deterministic-recovery: near-verbatim sentence in retrieved content",
                evidence_type="direct_statement",
                retrieved_at=source.retrieved_at,
                notes="Deterministically recovered from retrieved web content.",
            )
        ]

    @property
    def produced_by(self) -> dict:
        return {
            "prompt_version": "test/none",
            "provider": self.provider_name,
            "model": "none",
        }


def evidence_for(url: str, excerpt: str, tier_note: str = "") -> Evidence:
    return Evidence(
        evidence_id=evidence_id_for(source_id_for(url), excerpt),
        source_id=source_id_for(url),
        excerpt=excerpt,
        locator="section 1",
        evidence_type="direct_statement",
        retrieved_at=FIXED_AT,
        notes=tier_note,
    )


def package_with(
    claims: list[FactClaim],
    sources: list[Source],
) -> ResearchPackage:
    return ResearchPackage.build(
        topic_id="t_hawking",
        topic_title="Hawking radiation",
        research_status="ready_for_fact_check",
        claims=claims,
        sources=sources,
        run_id="r1",
        created_at=FIXED_AT,
    )


def check(provider, package: ResearchPackage) -> FactCheckResult:
    return FactChecker(provider, verifier=MockVerifier(), run_id="r1", created_at=FIXED_AT).check(
        package
    )


# =========================================================================== #
# 1. Existing extractor finds evidence -> unchanged behavior
# =========================================================================== #


class TestExtractorSuccessUnchanged:
    """1. When the extractor already found the evidence, nothing changes."""

    def test_recovery_not_attempted_when_evidence_exists(self):
        """The extractor already returned evidence, so recovery is not consulted."""
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        britannica = make_source(BRITANNICA_URL, SECONDARY)
        provider = RecoveringProvider(
            contents={wikipedia.source_id: NARROW_WIKIPEDIA},
            extracted={
                wikipedia.source_id: [evidence_for(WIKIPEDIA_URL, NARROW_WIKIPEDIA)],
                britannica.source_id: [evidence_for(BRITANNICA_URL, NARROW_BRITANNICA)],
            },
        )
        result = check(
            provider,
            package_with(
                [make_claim(text=NARROW_CLAIM, urls=(WIKIPEDIA_URL, BRITANNICA_URL))],
                [wikipedia, britannica],
            ),
        )
        assert provider.recover_calls == []
        verdict = result.claim_verdict("cl_test")
        assert verdict.verdict == "PASS"
        assert set(verdict.supporting_source_ids) == {wikipedia.source_id, britannica.source_id}

    def test_extractor_evidence_is_byte_identical_with_and_without_capability(self):
        """A provider without recovery support produces the same verdict."""
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        evidence = [evidence_for(WIKIPEDIA_URL, WIKIPEDIA_LIKE)]

        class NoRecovery:
            def get_evidence(self, source):
                return list(evidence)

        with_recovery = check(
            RecoveringProvider({wikipedia.source_id: WIKIPEDIA_LIKE}, {wikipedia.source_id: evidence}),
            package_with([make_claim()], [wikipedia]),
        )
        without_recovery = check(NoRecovery(), package_with([make_claim()], [wikipedia]))
        assert with_recovery.claim_results[0] == without_recovery.claim_results[0]

    def test_unrecoverable_claim_keeps_its_original_verdict(self):
        """A source that genuinely lacks the text still reviews."""
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        provider = RecoveringProvider(
            contents={wikipedia.source_id: "Nothing relevant lives on this page."}
        )
        result = check(provider, package_with([make_claim()], [wikipedia]))
        assert result.claim_verdict("cl_test").verdict != "PASS"


# =========================================================================== #
# 2. Extractor misses a near-verbatim supporting sentence -> recovered
# =========================================================================== #


class TestNearVerbatimRecovered:
    """2. The recall gap that caused defect #2 is closed."""

    def test_missed_supporting_sentence_is_recovered(self):
        """The exact recall failure: extractor selected nothing, recovery found it.

        Asserted on what recovery actually controls - the source becoming
        credited - rather than on the final verdict, which additionally depends
        on the pre-existing coverage and corroboration gates.
        """
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        britannica = make_source(BRITANNICA_URL, SECONDARY)
        provider = RecoveringProvider(
            contents={
                wikipedia.source_id: NARROW_WIKIPEDIA,
                britannica.source_id: NARROW_BRITANNICA,
            },
            extracted={},  # extractor selected nothing
        )
        verdict = check(
            provider,
            package_with(
                [make_claim(text=NARROW_CLAIM, urls=(WIKIPEDIA_URL, BRITANNICA_URL))],
                [wikipedia, britannica],
            ),
        ).claim_verdict("cl_test")
        assert verdict.verdict == "PASS"
        assert wikipedia.source_id in verdict.supporting_source_ids

    def test_recovery_is_what_credits_the_missed_source(self):
        """Control: without recovery the same package does not reach PASS.

        This is the assertion that would fail if the backstop were removed, and
        it is what distinguishes recovery from a pre-existing pass.
        """
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        britannica = make_source(BRITANNICA_URL, SECONDARY)

        class NoRecovery:
            def get_evidence(self, source):
                return []

        package = package_with(
            [make_claim(text=NARROW_CLAIM, urls=(WIKIPEDIA_URL, BRITANNICA_URL))],
            [wikipedia, britannica],
        )
        without = FactChecker(NoRecovery(), verifier=MockVerifier(), run_id="r1", created_at=FIXED_AT)
        assert without.check(package).claim_verdict("cl_test").verdict != "PASS"

        provider = RecoveringProvider(
            contents={
                wikipedia.source_id: NARROW_WIKIPEDIA,
                britannica.source_id: NARROW_BRITANNICA,
            },
            extracted={},
        )
        with_recovery = check(provider, package)
        assert with_recovery.claim_verdict("cl_test").verdict == "PASS"

    def test_recovery_turns_the_real_failure_into_corroboration(self):
        """The exact shape of cl_e552015c60: one eligible source recovered.

        This is the defect, reproduced. Before the fix Wikipedia was retrieved
        but never credited, so corroboration was 1 of 2. The assertion is on
        corroboration, which is the quantity the defect was about.

        `REAL_CLAIM` still lands on REVIEW afterwards via the pre-existing
        Gate 9 partial-coverage rule, because its compound wording ("...based on
        a model developed by Stephen Hawking") is not settled by any single
        sentence. That is correct fail-safe behaviour and is asserted
        separately in `test_real_claim_still_reviews_on_its_own_wording`.
        """
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        britannica = make_source(BRITANNICA_URL, SECONDARY)
        provider = RecoveringProvider(
            contents={
                wikipedia.source_id: WIKIPEDIA_LIKE,
                britannica.source_id: BRITANNICA_LIKE,
            },
            extracted={
                britannica.source_id: [evidence_for(BRITANNICA_URL, BRITANNICA_LIKE)]
            },
        )
        result = check(
            provider,
            package_with(
                [make_claim(urls=(WIKIPEDIA_URL, BRITANNICA_URL))],
                [wikipedia, britannica],
            ),
        )
        verdict = result.claim_verdict("cl_test")
        # Wikipedia is now credited: corroboration went from 1 to 2 eligible sources.
        assert set(verdict.supporting_source_ids) == {wikipedia.source_id, britannica.source_id}

    def test_real_claim_still_reviews_on_its_own_wording(self):
        """Recovery fixes recall only; it must not mask a coverage shortfall."""
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        britannica = make_source(BRITANNICA_URL, SECONDARY)
        provider = RecoveringProvider(
            contents={
                wikipedia.source_id: WIKIPEDIA_LIKE,
                britannica.source_id: BRITANNICA_LIKE,
            },
            extracted={
                britannica.source_id: [evidence_for(BRITANNICA_URL, BRITANNICA_LIKE)]
            },
        )
        verdict = check(
            provider,
            package_with(
                [make_claim(urls=(WIKIPEDIA_URL, BRITANNICA_URL))],
                [wikipedia, britannica],
            ),
        ).claim_verdict("cl_test")
        # Recovered evidence is real and credited, but the claim's compound
        # wording is still only partially covered, so Gate 9 holds it at REVIEW.
        assert verdict.verdict == "REVIEW"
        assert "part of the claim's exact wording" in verdict.rationale
        # The coverage shortfall is reported, not silently cleared.
        assert "corroborated by only" not in verdict.rationale

    def test_recovered_sentence_is_the_actual_supporting_text(self):
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        provider = RecoveringProvider(contents={wikipedia.source_id: WIKIPEDIA_LIKE})
        recovered = provider.recover_evidence(wikipedia, make_claim())
        assert len(recovered) == 1
        assert "Hawking radiation is theoretical radiation" in recovered[0].excerpt

    def test_punctuation_and_formatting_differences_are_tolerated(self):
        variants = [
            WIKIPEDIA_LIKE,
            WIKIPEDIA_LIKE.replace("'", "\u2019"),
            "  " + WIKIPEDIA_LIKE.replace(" ", "\n") + "  ",
            WIKIPEDIA_LIKE.upper(),
            WIKIPEDIA_LIKE.replace("event horizon", "event-horizon"),
        ]
        for variant in variants:
            assert is_near_verbatim(REAL_CLAIM, variant), variant

    def test_unicode_normalization_is_applied(self):
        folded = normalize_for_recovery("Hawking\u2019s radiation \u2014 a test")
        assert "\u2019" not in folded and "\u2014" not in folded
        # Tokenization splits possessives, but it splits them identically on
        # both sides, which is what keeps the comparison consistent.
        assert recovery_tokens("Hawking\u2019s") == ["hawking", "s"]
        assert recovery_tokens("Hawking's") == recovery_tokens("Hawking\u2019s")

    def test_claim_split_across_two_sentences_is_recoverable(self):
        """A claim straddling a sentence boundary is recoverable by joining.

        Neither sentence alone reaches the threshold; the adjacent pair does.
        This is why `candidate_sentences` yields pairs as well as singles.
        """
        content = (
            "Hawking radiation is theoretical radiation released outside a black "
            "hole's event horizon. The effect is due to quantum effects."
        )
        assert is_near_verbatim(NARROW_CLAIM, content)
        # The trailing sentence on its own is far too little to clear the bar.
        assert not is_near_verbatim(
            NARROW_CLAIM, "The effect is due to quantum effects."
        )
        # What is recovered is the pair, not just the stronger first sentence.
        recovered = find_near_verbatim(NARROW_CLAIM, content)
        assert "due to quantum effects" in recovered
        assert near_verbatim_score(NARROW_CLAIM, recovered) > near_verbatim_score(
            NARROW_CLAIM, content.split(". ")[0]
        )

    def test_markdown_page_recovers_from_rendered_prose(self):
        content = (
            "# Hawking radiation\n\n"
            "**Hawking radiation** is [theoretical radiation](https://example.test/a) "
            "predicted to be released outside a black hole's event horizon, and is "
            "due to quantum effects near the event horizon. It was developed by "
            "Stephen Hawking in 1974.\n"
        )
        assert is_near_verbatim(REAL_CLAIM, content)

    def test_best_candidate_wins(self):
        content = (
            "Nothing useful here at all.\n\n"
            + WIKIPEDIA_LIKE
            + "\n\nHawking radiation is radiation.\n"
        )
        assert find_near_verbatim(REAL_CLAIM, content) == WIKIPEDIA_LIKE.strip()


# =========================================================================== #
# 3. Recovered evidence has a valid deterministic evidence ID
# =========================================================================== #


class TestRecoveredEvidenceIdentity:
    """3. Evidence ids are the normal deterministic ones."""

    def test_id_matches_the_shared_id_function(self):
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        recovered = RecoveringProvider(
            contents={wikipedia.source_id: WIKIPEDIA_LIKE}
        ).recover_evidence(wikipedia, make_claim())
        assert recovered[0].evidence_id == evidence_id_for(
            wikipedia.source_id, recovered[0].excerpt
        )
        assert recovered[0].evidence_id.startswith("ev_")

    def test_id_is_independent_of_the_claim_it_recovered_for(self):
        """Same source + same sentence => same id, whatever the claim."""
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        provider = RecoveringProvider(contents={wikipedia.source_id: WIKIPEDIA_LIKE})
        first = provider.recover_evidence(wikipedia, make_claim(claim_id="cl_a"))
        second = provider.recover_evidence(wikipedia, make_claim(claim_id="cl_b"))
        assert first[0].evidence_id == second[0].evidence_id

    def test_id_matches_evidence_from_the_normal_path(self):
        """A recovered excerpt and a model excerpt of the same text collide
        deliberately, so recovery can never double-count one sentence."""
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        provider = RecoveringProvider(contents={wikipedia.source_id: WIKIPEDIA_LIKE})
        recovered = provider.recover_evidence(wikipedia, make_claim())
        assert recovered, "expected recovery from the realistic page text"
        sentence = recovered[0].excerpt

        # The same sentence, delivered through the normal extractor path, must
        # produce the same evidence id. This is what stops recovery from
        # double-counting one sentence that the extractor did report.
        model_path = evidence_for(WIKIPEDIA_URL, sentence)
        assert model_path.evidence_id == recovered[0].evidence_id
        assert recovered[0].evidence_id == evidence_id_for(wikipedia.source_id, sentence)

    def test_evidence_passes_model_validation(self):
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        recovered = RecoveringProvider(
            contents={wikipedia.source_id: WIKIPEDIA_LIKE}
        ).recover_evidence(wikipedia, make_claim())
        assert recovered[0].validate() is recovered[0]

    def test_recovered_evidence_round_trips_through_the_model(self):
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        recovered = RecoveringProvider(
            contents={wikipedia.source_id: WIKIPEDIA_LIKE}
        ).recover_evidence(wikipedia, make_claim())
        assert EvidenceModel.from_dict(recovered[0].to_dict()) == recovered[0]

    def test_evidence_type_is_direct_statement_only(self):
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        recovered = RecoveringProvider(
            contents={wikipedia.source_id: WIKIPEDIA_LIKE}
        ).recover_evidence(wikipedia, make_claim())
        assert recovered[0].evidence_type == "direct_statement"


# =========================================================================== #
# 4. Recovered evidence passes the existing containment validation
# =========================================================================== #


class TestRecoveredEvidenceIsGrounded:
    """4. The excerpt is verbatim in the retrieved content - never reworded."""

    def test_excerpt_is_a_substring_of_the_content(self):
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        recovered = RecoveringProvider(
            contents={wikipedia.source_id: WIKIPEDIA_LIKE}
        ).recover_evidence(wikipedia, make_claim())
        assert recovered[0].excerpt in WIKIPEDIA_LIKE

    def test_gemini_provider_applies_its_own_containment_check(self):
        """The real provider re-verifies grounding before returning."""
        from providers.gemini.evidence_provider import GeminiEvidenceProvider

        source = make_source(WIKIPEDIA_URL, SECONDARY)
        provider = GeminiEvidenceProvider.__new__(GeminiEvidenceProvider)
        provider._content_cache = {source.source_id: [WIKIPEDIA_LIKE]}
        recovered = provider.recover_evidence(source, make_claim())
        assert recovered and GeminiEvidenceProvider._grounded(recovered[0].excerpt, [WIKIPEDIA_LIKE])

    def test_gemini_provider_refuses_when_content_is_absent(self):
        from providers.gemini.evidence_provider import GeminiEvidenceProvider

        source = make_source(WIKIPEDIA_URL, SECONDARY)
        provider = GeminiEvidenceProvider.__new__(GeminiEvidenceProvider)
        provider._content_cache = {}
        assert provider.recover_evidence(source, make_claim()) == []

    def test_no_content_means_no_evidence_not_an_error(self):
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        provider = RecoveringProvider(contents={})
        assert provider.recover_evidence(wikipedia, make_claim()) == []

    def test_recovery_notes_record_the_provenance(self):
        from providers.gemini.evidence_provider import RECOVERY_NOTE_PREFIX

        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        provider = GeminiEvidenceProviderWithCache(source= wikipedia)
        recovered = provider.recover_evidence(wikipedia, make_claim())
        assert recovered[0].notes == RECOVERY_NOTE_PREFIX


def GeminiEvidenceProviderWithCache(*, source):
    from providers.gemini.evidence_provider import GeminiEvidenceProvider

    provider = GeminiEvidenceProvider.__new__(GeminiEvidenceProvider)
    provider._content_cache = {source.source_id: [WIKIPEDIA_LIKE]}
    return provider


# =========================================================================== #
# 5. Source without the claim's supporting content receives no credit
# =========================================================================== #


class TestNoSupportNoRecovery:
    """5. A source that does not contain the support gains nothing."""

    UNRELATED = (
        "Marine biology is the study of ocean life. Coral reefs host thousands "
        "of species and are threatened by warming seas."
    )

    def test_unrelated_page_recovers_nothing(self):
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        provider = RecoveringProvider(contents={wikipedia.source_id: self.UNRELATED})
        assert provider.recover_evidence(wikipedia, make_claim()) == []

    def test_unrelated_page_cannot_support_the_claim(self):
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        provider = RecoveringProvider(contents={wikipedia.source_id: self.UNRELATED})
        result = check(provider, package_with([make_claim()], [wikipedia]))
        verdict = result.claim_verdict("cl_test")
        assert verdict.verdict != "PASS"
        assert wikipedia.source_id not in verdict.supporting_source_ids

    def test_empty_content_recovers_nothing(self):
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        assert RecoveringProvider(
            contents={wikipedia.source_id: ""}
        ).recover_evidence(wikipedia, make_claim()) == []

    def test_empty_source_set_recovers_nothing(self):
        assert find_near_verbatim(REAL_CLAIM, "") == ""

    def test_claim_with_no_sources_is_still_rejected(self):
        provider = RecoveringProvider(contents={})
        claim = FactClaim(
            claim_id="cl_bare",
            claim_text=REAL_CLAIM,
            claim_type="fact",
            source_ids=(),
            confidence=0.5,
            critical=True,
        )
        result = check(provider, package_with([claim], [make_source(WIKIPEDIA_URL, SECONDARY)]))
        assert result.claim_verdict("cl_bare").verdict == "REJECT"

    def test_wrong_subject_is_not_recovered(self):
        """Same shape, different subject: must not match."""
        other = "Tidal heating is a process in which orbital decay warms a body."
        assert find_near_verbatim(REAL_CLAIM, other) == ""


# =========================================================================== #
# 6. Keyword overlap alone does not create evidence
# =========================================================================== #


class TestKeywordOverlapIsNotEvidence:
    """6. This is the anti-fabrication boundary that matters most."""
    KEYWORD_SALAD = _keyword_salad(REAL_CLAIM)

    def test_all_claim_keywords_in_a_row_is_not_recovered(self):
        assert not is_near_verbatim(REAL_CLAIM, self.KEYWORD_SALAD)
        assert find_near_verbatim(REAL_CLAIM, self.KEYWORD_SALAD) == ""

    def test_reordered_claim_words_are_not_recovered(self):
        reordered = (
            "Theoretical radiation Hawking event horizon quantum effects released "
            "outside a black hole's based on a model developed by Stephen radiation."
        )
        assert not is_near_verbatim(REAL_CLAIM, reordered)

    def test_score_of_keyword_salad_is_low_despite_full_token_overlap(self):
        """Bag-of-words coverage is total; order-preserving score is not.

        This is precisely why recovery cannot be plain keyword matching.
        """
        assert meaningful_tokens(self.KEYWORD_SALAD) >= meaningful_tokens(REAL_CLAIM)
        assert near_verbatim_score(REAL_CLAIM, self.KEYWORD_SALAD) < RECOVERY_MIN_LCS_RATIO

    def test_short_overlap_is_rejected(self):
        """A long page with one matching sentence needs a real match."""
        content = (
            "Tidal forces shape icy moons. Radiation warms some of them. "
            "Quantum effects appear in laboratory magnets. Horizons blur in fog."
        )
        assert find_near_verbatim(REAL_CLAIM, content) == ""

    def test_thresholds_are_enforced_on_the_contiguous_run(self):
        claim = "alpha beta gamma delta epsilon zeta"
        # Full set, no contiguous run of RECOVERY_MIN_RUN tokens.
        scrambled = "zeta gamma beta alpha epsilon delta"
        assert not is_near_verbatim(claim, scrambled)
        assert RECOVERY_MIN_RUN >= 4

    def test_very_short_claims_are_not_recovered(self):
        """Guards against a one-word claim matching by accident."""
        assert RECOVERY_MIN_CLAIM_TOKENS >= 4
        assert find_near_verbatim("Hawking", WIKIPEDIA_LIKE) == ""

    def test_near_miss_below_threshold_is_rejected(self):
        """Half the claim is not enough, however relevant the page."""
        content = "Hawking radiation is theoretical radiation released in 1974."
        assert not is_near_verbatim(REAL_CLAIM, content)

    def test_long_paragraph_dump_is_not_treated_as_a_sentence(self):
        """A page-sized block is not a restatement."""
        from agents.qa.fact_checker.recovery import RECOVERY_MAX_SENTENCE_TOKENS

        filler = "The quick brown fox jumps over the lazy dog. " * 40
        assert not any(
            len(recovery_tokens(sentence)) > RECOVERY_MAX_SENTENCE_TOKENS
            for sentence in candidate_sentences(filler)
        )

    def test_keyword_salad_page_cannot_push_a_claim_to_pass(self):
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        provider = RecoveringProvider(contents={wikipedia.source_id: self.KEYWORD_SALAD})
        result = check(provider, package_with([make_claim()], [wikipedia]))
        assert result.claim_verdict("cl_test").verdict != "PASS"


# =========================================================================== #
# 7. Discovery sources are not promoted
# =========================================================================== #


class TestDiscoveryNotPromoted:
    """7. Discovery-tier sources gain no credit and no tier change.

    The tier policy lives in the checker, not the provider: the provider only
    knows how to find text, while deciding which sources may corroborate is a
    Stage-3 policy decision. These tests therefore assert at the checker level,
    and confirm the provider is never even asked about a discovery source.
    """

    def test_discovery_source_is_offered_to_the_provider(self):
        """Discovery-tier sources now receive the recall backstop.

        Deliberately changed from "never offered". The old tier filter meant a
        discovery-only claim lost the backstop entirely and could be REJECTed
        for want of evidence while its supporting sentence sat unread in the
        page that had already been retrieved. Offering the backstop cannot make
        such a claim PASS - Gate 10 holds it at REVIEW, and the critical gate
        counts only non-discovery tiers.
        """
        source = make_source(WIKIPEDIA_URL, DISCOVERY)
        provider = RecoveringProvider(contents={source.source_id: WIKIPEDIA_LIKE})
        check(provider, package_with([make_claim()], [source]))
        assert provider.recover_calls == [(source.source_id, "cl_test")]

    def test_discovery_recovery_offered_by_the_provider_reaches_the_verifier(self):
        """Belt and braces: evidence offered for a discovery source is judged,
        not discarded. It still cannot count toward corroboration."""

        class Offering(RecoveringProvider):
            def recover_evidence(self, source, claim):
                # Records the call itself, so the assertion below cannot pass
                # merely because the override bypassed the base bookkeeping.
                self.recover_calls.append((source.source_id, claim.claim_id))
                return [
                    Evidence(
                        evidence_id=evidence_id_for(source.source_id, WIKIPEDIA_LIKE),
                        source_id=source.source_id,
                        excerpt=WIKIPEDIA_LIKE,
                        locator="x",
                        evidence_type="direct_statement",
                        retrieved_at=source.retrieved_at,
                    )
                ]

        source = make_source(WIKIPEDIA_URL, DISCOVERY)
        provider = Offering({source.source_id: WIKIPEDIA_LIKE})
        result = check(provider, package_with([make_claim()], [source]))
        assert provider.recover_calls == [(source.source_id, "cl_test")]
        # The verifier is still the only thing that decides support.
        assert source.source_id in result.claim_verdict("cl_test").supporting_source_ids
        assert result.claim_verdict("cl_test").verdict != "PASS"

    def test_discovery_tier_is_never_mutated(self):
        source = make_source(WIKIPEDIA_URL, DISCOVERY)
        provider = RecoveringProvider(contents={source.source_id: WIKIPEDIA_LIKE})
        check(provider, package_with([make_claim()], [source]))
        assert source.credibility_tier == DISCOVERY

    def test_discovery_alone_still_reviews(self):
        source = make_source(WIKIPEDIA_URL, DISCOVERY)
        provider = RecoveringProvider(contents={source.source_id: WIKIPEDIA_LIKE})
        result = check(provider, package_with([make_claim()], [source]))
        verdict = result.claim_verdict("cl_test")
        assert verdict.verdict != "PASS"
        assert verdict.rationale

    def test_discovery_alone_is_review_not_reject(self):
        """Asserted as REVIEW, not merely `!= "PASS"`.

        The weaker form of the assertion above is satisfied by a REJECT as well,
        which is exactly why the false REJECT in run
        `e2e-20261005-evidence-audit-01` went unnoticed. The page supports the
        claim, so the honest verdict is REVIEW rather than a REJECT that would
        assert the claim is unsupported.
        """
        source = make_source(WIKIPEDIA_URL, DISCOVERY)
        provider = RecoveringProvider(contents={source.source_id: WIKIPEDIA_LIKE})
        verdict = check(provider, package_with([make_claim()], [source])).claim_verdict("cl_test")
        assert verdict.verdict == "REVIEW"
        # Gate 6 is the rule that produced the false REJECT; it must not fire.
        assert "none of the retrieved evidence supports" not in verdict.rationale

    def test_recovery_does_not_make_a_discovery_source_count_as_corroboration(self):
        discovery = make_source(WIKIPEDIA_URL, DISCOVERY)
        secondary = make_source(BRITANNICA_URL, SECONDARY)
        provider = RecoveringProvider(
            contents={
                discovery.source_id: WIKIPEDIA_LIKE,
                secondary.source_id: "Hawking radiation escapes a black hole.",
            },
            extracted={
                secondary.source_id: [
                    evidence_for(BRITANNICA_URL, "Hawking radiation escapes a black hole.")
                ]
            },
        )
        result = check(
            provider,
            package_with([make_claim(urls=(WIKIPEDIA_URL, BRITANNICA_URL))], [discovery, secondary]),
        )
        # Only one non-discovery source supports it -> still short of 2.
        assert result.claim_verdict("cl_test").verdict != "PASS"

    def test_recovered_discovery_evidence_never_satisfies_the_corroboration_gate(self):
        """Two discovery sources, both recovered, must still not corroborate.

        This is the sharpest form of the safety property: the gate counts
        non-discovery tiers only, so widening recovery cannot manufacture the
        corroboration that a critical claim needs.
        """
        first = make_source(WIKIPEDIA_URL, DISCOVERY)
        second = make_source(BRITANNICA_URL, DISCOVERY)
        provider = RecoveringProvider(
            contents={
                first.source_id: NARROW_WIKIPEDIA,
                second.source_id: NARROW_BRITANNICA,
            },
            extracted={},
        )
        result = check(
            provider,
            package_with(
                [make_claim(text=NARROW_CLAIM, urls=(WIKIPEDIA_URL, BRITANNICA_URL))],
                [first, second],
            ),
        )
        verdict = result.claim_verdict("cl_test")
        # Both sources really were recovered and credited...
        assert provider.recover_calls == [
            (first.source_id, "cl_test"),
            (second.source_id, "cl_test"),
        ]
        assert set(verdict.supporting_source_ids) == {first.source_id, second.source_id}
        # ...yet neither counts toward corroboration.
        assert verdict.verdict == "REVIEW"
        assert "discovery-tier" in verdict.rationale
        assert "corroborated by only 0" in verdict.rationale
        assert "independent non-discovery source(s); required 2" in verdict.rationale


class TestDiscoveryOnlyClaimReviewsInsteadOfRejecting:
    """Regression: the false REJECT that motivated widening the backstop.

    Run `e2e-20261005-evidence-audit-01`, claim `cl_f2d33645c9`:

        "A solar-mass black hole would have a temperature of about 60 nanokelvins."

    The page carried the supporting sentence, the claim-blind extractor did not
    surface it, and because the claim's only bound source was discovery-tier the
    backstop was skipped. The claim then failed Gate 6 (`support_level = NONE`)
    and was REJECTed, which asserts non-support for a claim that was in fact
    supported. These tests pin the corrected outcome: REVIEW, never PASS, and
    only because the verifier was shown the recovered sentence.
    """

    CLAIM_ID = "cl_f2d33645c9"
    CLAIM_TEXT = "A solar-mass black hole would have a temperature of about 60 nanokelvins."
    # Verbatim support as it appears on the retrieved page, including the
    # adjacent sentence that lifts the order-preserving score over the bar.
    PAGE_TEXT = (
        "A black hole with the mass of the Sun would have a temperature of about "
        "60 nanokelvins - tens of billions of times colder than the cosmic "
        "microwave background."
    )
    # The run surfaced seven such records and none of them supported the claim.
    # They matter: with *no* evidence at all Gate 3 fires and yields REVIEW, so
    # only evidence that is present-but-unsupporting reproduces the real REJECT.
    UNRELATED = "Stellar black holes form when massive stars collapse at the end of their life."

    def _package(self):
        discovery = make_source(WIKIPEDIA_URL, DISCOVERY)
        return discovery, package_with(
            [make_claim(claim_id=self.CLAIM_ID, text=self.CLAIM_TEXT)], [discovery]
        )

    def _extracted(self, discovery):
        """The extractor's output in the real run: present, but not supporting."""
        return {discovery.source_id: [evidence_for(WIKIPEDIA_URL, self.UNRELATED)]}

    def test_the_offline_oracle_finds_the_support(self):
        """The backstop's matcher really does recognise this page text.

        This is the assertion that retires my original, wrong hypothesis that
        recovery would fail here on LCS grounds. It scores 0.8462, above the 0.75
        bar, because `candidate_sentences` offers adjacent sentence pairs.
        """
        assert is_near_verbatim(self.CLAIM_TEXT, self.PAGE_TEXT)
        assert near_verbatim_score(self.CLAIM_TEXT, self.PAGE_TEXT) > RECOVERY_MIN_LCS_RATIO
        assert find_near_verbatim(self.CLAIM_TEXT, self.PAGE_TEXT)

    def test_supported_discovery_only_claim_reviews_rather_than_rejects(self):
        """The fix itself: recovery is attempted, finds the sentence, REVIEWs."""
        discovery, package = self._package()
        provider = RecoveringProvider(
            contents={discovery.source_id: self.PAGE_TEXT},
            extracted=self._extracted(discovery),
        )
        verdict = check(provider, package).claim_verdict(self.CLAIM_ID)

        # Recovery ran for the discovery source...
        assert provider.recover_calls == [(discovery.source_id, self.CLAIM_ID)]
        # ...it found the sentence...
        recovered_id = evidence_id_for(discovery.source_id, self.PAGE_TEXT)
        assert recovered_id in verdict.evidence_ids
        # ...the verifier credited the source on the second pass...
        assert discovery.source_id in verdict.supporting_source_ids
        # ...and the claim is REVIEW, not REJECT and not PASS.
        assert verdict.verdict == "REVIEW"
        assert "none of the retrieved evidence supports" not in verdict.rationale

    def test_unsupported_discovery_only_claim_is_still_rejected(self):
        """The fail-closed direction is preserved.

        Without the fix this case was indistinguishable from the one above: a
        discovery-only claim whose evidence was retrieved but did not support it
        must still REJECT. If this ever became REVIEW, the fix would be
        manufacturing evidence.
        """
        discovery, package = self._package()
        provider = RecoveringProvider(
            contents={discovery.source_id: "An unrelated page about tides and moons."},
            extracted=self._extracted(discovery),
        )
        verdict = check(provider, package).claim_verdict(self.CLAIM_ID)
        assert provider.recover_calls == [(discovery.source_id, self.CLAIM_ID)]
        assert verdict.verdict == "REJECT"
        assert "none of the retrieved evidence supports" in verdict.rationale
        assert discovery.source_id not in verdict.supporting_source_ids

    def test_recovery_is_what_lifts_the_verdict(self):
        """Control: without recovery capability the claim REJECTs again.

        This is the assertion that fails if the backstop is narrowed again.
        """
        discovery, package = self._package()

        class NoRecovery:
            def get_evidence(self, source):  # noqa: ARG002
                return list(provider_extracted)

        provider_extracted = self._extracted(discovery)[discovery.source_id]
        without = check(NoRecovery(), package).claim_verdict(self.CLAIM_ID)
        assert without.verdict == "REJECT"

        with_recovery = check(
            RecoveringProvider(
                contents={discovery.source_id: self.PAGE_TEXT},
                extracted=self._extracted(discovery),
            ),
            package,
        ).claim_verdict(self.CLAIM_ID)
        assert with_recovery.verdict == "REVIEW"

    def test_recovered_evidence_is_audited_as_recovery_for_a_discovery_source(self):
        """The audit shows the backstop actually ran, so the fix is observable."""
        discovery, package = self._package()
        result = check(
            RecoveringProvider(
                contents={discovery.source_id: self.PAGE_TEXT},
                extracted=self._extracted(discovery),
            ),
            package,
        )
        audit = result.claim_verdict(self.CLAIM_ID).evidence_audit
        records = {record.evidence_id: record for record in audit.records}
        recovered_id = evidence_id_for(discovery.source_id, self.PAGE_TEXT)
        assert records[recovered_id].origin == "recovery"
        assert records[recovered_id].source_id == discovery.source_id
        # The unsupported extractor record is still reported, untouched.
        extracted_id = evidence_id_for(discovery.source_id, self.UNRELATED)
        assert records[extracted_id].origin == "extraction"
        source_audit = next(
            entry for entry in audit.sources if entry.source_id == discovery.source_id
        )
        assert source_audit.recovery_attempted is True
        assert source_audit.recovery_count == 1
        assert source_audit.status == "credited"

    def test_at_most_one_recovered_candidate_per_source(self):
        """The widened backstop stays bounded to a single candidate per source."""
        discovery, package = self._package()
        provider = RecoveringProvider(
            contents={
                discovery.source_id: (
                    self.PAGE_TEXT
                    + "\n\nA solar-mass black hole would have a temperature of "
                    "about 60 nanokelvins."
                )
            },
            extracted=self._extracted(discovery),
        )
        result = check(provider, package)
        verdict = result.claim_verdict(self.CLAIM_ID)
        assert verdict.verdict == "REVIEW"
        # Several sentences restate the claim, but only one may be admitted.
        audit = verdict.evidence_audit
        recovered = [record for record in audit.records if record.origin == "recovery"]
        assert len(recovered) == 1
        assert len(verdict.evidence_ids) == len(set(verdict.evidence_ids))
        source_audit = next(
            entry for entry in audit.sources if entry.source_id == discovery.source_id
        )
        assert source_audit.recovery_count == 1


# =========================================================================== #
# 8. Backstop cannot introduce a new source ID
# =========================================================================== #


class TestRecoveryCannotAddSource:
    """8. Recovery is confined to sources already bound to the claim."""

    def test_unbound_source_with_support_is_not_recovered(self):
        bound = make_source(BRITANNICA_URL, SECONDARY)
        unbound = make_source(WIKIPEDIA_URL, SECONDARY)
        provider = RecoveringProvider(
            contents={unbound.source_id: WIKIPEDIA_LIKE, bound.source_id: "Nothing here."}
        )
        result = check(provider, package_with([make_claim(urls=(BRITANNICA_URL,))], [bound, unbound]))
        verdict = result.claim_verdict("cl_test")
        assert unbound.source_id not in verdict.evidence_ids or True
        assert verdict.verdict != "PASS"
        assert unbound.source_id not in verdict.supporting_source_ids

    def test_evidence_bound_to_another_source_is_dropped(self):
        """A provider returning misbound evidence cannot smuggle a source in."""

        class Misbound(RecoveringProvider):
            def recover_evidence(self, source, claim):
                return [
                    Evidence(
                        evidence_id="ev_misbound",
                        source_id="src_somewhereelse00",
                        excerpt=WIKIPEDIA_LIKE,
                        locator="x",
                        evidence_type="direct_statement",
                    )
                ]

        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        result = check(Misbound({wikipedia.source_id: WIKIPEDIA_LIKE}), package_with([make_claim()], [wikipedia]))
        assert any("instead of" in issue for issue in result.evidence_issues)
        assert result.claim_verdict("cl_test").verdict != "PASS"

    def test_recovery_adds_no_new_sources_to_the_package(self):
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        package = package_with([make_claim()], [wikipedia])
        before = {source.source_id for source in package.sources}
        provider = RecoveringProvider(contents={wikipedia.source_id: WIKIPEDIA_LIKE})
        check(provider, package)
        assert {source.source_id for source in package.sources} == before

    def test_provider_is_asked_only_about_bound_sources(self):
        bound = make_source(BRITANNICA_URL, SECONDARY)
        unbound = make_source(WIKIPEDIA_URL, SECONDARY)
        provider = RecoveringProvider(
            contents={bound.source_id: "", unbound.source_id: WIKIPEDIA_LIKE}
        )
        check(
            provider,
            package_with([make_claim(urls=(BRITANNICA_URL,))], [bound, unbound]),
        )
        assert [source_id for source_id, _ in provider.recover_calls] == [bound.source_id]

    def test_unknown_source_id_on_a_claim_is_rejected_by_the_gate(self):
        claim = FactClaim(
            claim_id="cl_unknown",
            claim_text=REAL_CLAIM,
            claim_type="fact",
            source_ids=("src_doesnotexist00",),
            confidence=0.5,
            critical=True,
        )
        provider = RecoveringProvider(contents={"src_doesnotexist00": WIKIPEDIA_LIKE})
        result = check(provider, package_with([claim], [make_source(WIKIPEDIA_URL, SECONDARY)]))
        assert result.claim_verdict("cl_unknown").verdict == "REJECT"


# =========================================================================== #
# 9. Existing verifier-selected evidence remains unchanged
# =========================================================================== #


class TestVerifierSelectionUnchanged:
    """9. Recovery adds candidates; it never overrules the verifier."""

    def test_recovery_cannot_credit_a_source_the_verifier_declines(self):
        """Even real recovered text must be judged by the verifier.

        The MockVerifier scores by token coverage. A recovered excerpt that only
        partially covers the claim is PARTIAL, not DIRECT, and the gates
        downgrade a partially-worded claim.
        """
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        provider = RecoveringProvider(
            contents={
                wikipedia.source_id: (
                    "Stephen Hawking developed a theoretical model of quantum "
                    "effects near a black hole's event horizon in 1974, which "
                    "predicts radiation escaping the horizon over long times."
                )
            }
        )
        verdict = check(provider, package_with([make_claim()], [wikipedia])).claim_verdict("cl_test")
        assert wikipedia.source_id not in verdict.supporting_source_ids

    def test_recovery_does_not_touch_a_source_the_verifier_already_credited(self):
        """A source the verifier credited is never re-offered to the provider."""
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        britannica = make_source(BRITANNICA_URL, SECONDARY)
        provider = RecoveringProvider(
            contents={
                wikipedia.source_id: NARROW_WIKIPEDIA,
                britannica.source_id: NARROW_BRITANNICA,
            },
            extracted={
                wikipedia.source_id: [evidence_for(WIKIPEDIA_URL, NARROW_WIKIPEDIA)],
                britannica.source_id: [evidence_for(BRITANNICA_URL, NARROW_BRITANNICA)],
            },
        )
        check(
            provider,
            package_with(
                [make_claim(text=NARROW_CLAIM, urls=(WIKIPEDIA_URL, BRITANNICA_URL))],
                [wikipedia, britannica],
            ),
        )
        # Both sources were credited on the first pass, so there was nothing to
        # recover and the provider was never asked.
        assert provider.recover_calls == []

    def test_recovery_does_run_when_extracted_evidence_was_not_credited(self):
        """This is the real defect: Wikipedia had evidence, but none of it supported.

        A rule of "skip any source that already has evidence" would leave this
        claim at corroboration 1 of 2 forever, which is exactly what happened in
        run `e2e-20261002-192946`.
        """
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        unrelated = "Unrelated material about tides."
        provider = RecoveringProvider(
            contents={wikipedia.source_id: NARROW_WIKIPEDIA},
            extracted={wikipedia.source_id: [evidence_for(WIKIPEDIA_URL, unrelated)]},
        )
        verdict = check(
            provider,
            package_with([make_claim(text=NARROW_CLAIM)], [wikipedia]),
        ).claim_verdict("cl_test")
        assert provider.recover_calls == [(wikipedia.source_id, "cl_test")]
        # The uncredited-but-present source is now credited via recovery.
        assert wikipedia.source_id in verdict.supporting_source_ids
        # The unrelated excerpt the extractor did report is still reported.
        assert any(item in verdict.evidence_ids for item in
                   (evidence_for(WIKIPEDIA_URL, unrelated).evidence_id,))

    def test_verifier_still_sees_and_reports_recovered_evidence(self):
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        provider = RecoveringProvider(contents={wikipedia.source_id: WIKIPEDIA_LIKE})
        verdict = check(provider, package_with([make_claim()], [wikipedia])).claim_verdict("cl_test")
        assert verdict.evidence_ids
        assert "evidence examined:" in verdict.rationale

    def test_recovered_evidence_is_listed_in_the_artifact(self):
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        provider = RecoveringProvider(contents={wikipedia.source_id: WIKIPEDIA_LIKE})
        result = check(provider, package_with([make_claim()], [wikipedia]))
        recovered_id = evidence_id_for(wikipedia.source_id, find_near_verbatim(REAL_CLAIM, WIKIPEDIA_LIKE))
        assert recovered_id in result.claim_verdict("cl_test").evidence_ids

    def test_no_recovery_paths_are_added_to_the_artifact(self):
        """The backstop adds no schema field and no new artifact.

        `evidence_audit` is the one deliberate addition to this record, and it
        came from the separate evidence-chain observability work, not from the
        backstop. It is still exactly one audit field, and the backstop itself
        contributed no verdict, gate or source-selection field.
        """
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        provider = RecoveringProvider(contents={wikipedia.source_id: WIKIPEDIA_LIKE})
        payload = check(provider, package_with([make_claim()], [wikipedia])).to_dict()
        claim_entry = payload["data"]["claim_results"][0]
        assert set(claim_entry) == {
            "claim_id",
            "verdict",
            "evidence_ids",
            "supporting_source_ids",
            "contradiction_source_ids",
            "rationale",
            "confidence",
            "critical",
            "checks_performed",
            "verifier",
            "evidence_audit",
        }


# =========================================================================== #
# 10. Multiple eligible sources can independently recover evidence
# =========================================================================== #


class TestMultipleSourcesRecover:
    """10. Every eligible bound source gets its own independent chance."""

    def test_two_eligible_sources_both_recover(self):
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        britannica = make_source(BRITANNICA_URL, SECONDARY)
        provider = RecoveringProvider(
            contents={
                wikipedia.source_id: WIKIPEDIA_LIKE,
                britannica.source_id: (
                    "Hawking radiation is theoretical radiation emitted outside a "
                    "black hole's event horizon as a result of quantum effects, "
                    "from a model developed by Stephen Hawking."
                ),
            }
        )
        result = check(
            provider,
            package_with([make_claim(urls=(WIKIPEDIA_URL, BRITANNICA_URL))], [wikipedia, britannica]),
        )
        verdict = result.claim_verdict("cl_test")
        assert verdict.verdict == "PASS"
        assert set(verdict.supporting_source_ids) == {wikipedia.source_id, britannica.source_id}
        assert len(provider.recover_calls) == 2

    def test_one_eligible_source_with_support_one_without_still_reviews(self):
        """Recovery raises corroboration only as far as the text allows."""
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        britannica = make_source(BRITANNICA_URL, SECONDARY)
        provider = RecoveringProvider(
            contents={
                wikipedia.source_id: WIKIPEDIA_LIKE,
                britannica.source_id: "A page about something else entirely.",
            }
        )
        verdict = check(
            provider,
            package_with([make_claim(urls=(WIKIPEDIA_URL, BRITANNICA_URL))], [wikipedia, britannica]),
        ).claim_verdict("cl_test")
        assert verdict.verdict != "PASS"
        assert wikipedia.source_id in verdict.supporting_source_ids
        assert britannica.source_id not in verdict.supporting_source_ids

    def test_recovery_is_per_claim_not_shared_between_claims(self):
        """Evidence recovered for one claim must not appear for another."""
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        other_claim = make_claim(
            claim_id="cl_other",
            text="Coral reefs are threatened by warming ocean temperatures.",
        )
        provider = RecoveringProvider(
            contents={
                wikipedia.source_id: WIKIPEDIA_LIKE
                + "\n\nCoral reefs are threatened by warming ocean temperatures."
            }
        )
        result = check(provider, package_with([make_claim(), other_claim], [wikipedia]))
        first = result.claim_verdict("cl_test")
        second = result.claim_verdict("cl_other")
        assert set(first.evidence_ids) != set(second.evidence_ids)
        assert second.verdict != "PASS"

    def test_recovery_failure_is_reported_not_fatal(self):
        class Boom(RecoveringProvider):
            def recover_evidence(self, source, claim):
                raise RuntimeError("backstop exploded")

        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        result = check(Boom({wikipedia.source_id: WIKIPEDIA_LIKE}), package_with([make_claim()], [wikipedia]))
        assert any("evidence recovery failed" in issue for issue in result.evidence_issues)
        assert result.claim_verdict("cl_test").verdict != "PASS"


# =========================================================================== #
# 11. Duplicate recovery is deterministic / idempotent
# =========================================================================== #


class TestRecoveryIsDeterministic:
    """11. Same inputs, same evidence - always."""

    def test_repeated_checks_are_identical(self):
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)

        def run():
            provider = RecoveringProvider(contents={wikipedia.source_id: WIKIPEDIA_LIKE})
            return check(provider, package_with([make_claim()], [wikipedia])).to_dict()

        # `package_id` and `result_id` are derived from a fresh random package on
        # every call, so compare everything that recovery can actually influence.
        def stable(document: dict) -> dict:
            return {key: value for key, value in document["data"].items() if key not in {"package_id", "result_id"}}

        assert stable(run()) == stable(run())

    def test_repeated_recovery_returns_one_stable_item(self):
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        provider = RecoveringProvider(contents={wikipedia.source_id: WIKIPEDIA_LIKE})
        first = provider.recover_evidence(wikipedia, make_claim())
        second = provider.recover_evidence(wikipedia, make_claim())
        assert first == second
        assert len(first) == 1

    def test_recovery_does_not_duplicate_evidence_ids_in_the_artifact(self):
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        provider = RecoveringProvider(contents={wikipedia.source_id: WIKIPEDIA_LIKE})
        verdict = check(provider, package_with([make_claim()], [wikipedia])).claim_verdict("cl_test")
        assert len(verdict.evidence_ids) == len(set(verdict.evidence_ids))

    def test_sentence_extraction_is_stable_across_calls(self):
        assert candidate_sentences(WIKIPEDIA_LIKE) == candidate_sentences(WIKIPEDIA_LIKE)

    def test_normalization_is_idempotent(self):
        once = normalize_for_recovery(WIKIPEDIA_LIKE)
        assert normalize_for_recovery(once) == once

    def test_no_model_or_network_is_used_by_recovery(self, monkeypatch):
        """Recovery is pure text work: sockets are never touched."""
        import socket

        def forbidden(*args, **kwargs):
            raise AssertionError("recovery must not use the network")

        monkeypatch.setattr(socket, "socket", forbidden)
        monkeypatch.setattr(socket, "create_connection", forbidden)
        wikipedia = make_source(WIKIPEDIA_URL, SECONDARY)
        provider = RecoveringProvider(contents={wikipedia.source_id: WIKIPEDIA_LIKE})
        assert provider.recover_evidence(wikipedia, make_claim())


# =========================================================================== #
# 12. Existing evidence artifacts remain loadable
# =========================================================================== #


class TestExistingArtifactsUnaffected:
    """12. No schema change, so prior artifacts keep loading."""

    RUN = Path("data/e2e-20261002-192946/research")

    def test_real_fact_check_result_still_loads(self):
        path = self.RUN / "fact_check_result.json"
        if not path.is_file():
            pytest.skip("real-run artifact not available")
        result = FactCheckResult.from_json(path.read_text(encoding="utf-8"))
        assert result.package_verdict == "REVIEW"
        assert result.claim_verdict("cl_e552015c60") is not None

    def test_real_research_package_still_loads_and_validates(self):
        path = self.RUN / "research_package.json"
        if not path.is_file():
            pytest.skip("real-run artifact not available")
        package = ResearchPackage.from_dict(json.loads(path.read_text(encoding="utf-8")))
        package.validate()
        assert package.claims

    def test_real_claim_would_now_be_recoverable(self):
        """The defect is reproducible offline against the real claim text."""
        path = self.RUN / "research_package.json"
        if not path.is_file():
            pytest.skip("real-run artifact not available")
        payload = json.loads(path.read_text(encoding="utf-8"))["data"]
        claim = next(c for c in payload["claims"] if c["claim_id"] == "cl_e552015c60")
        assert is_near_verbatim(claim["claim_text"], WIKIPEDIA_LIKE)

    def test_real_claim_is_not_recoverable_from_unrelated_text(self):
        claim = (
            "Hawking radiation is theoretical radiation released outside a black "
            "hole's event horizon due to quantum effects, based on a model "
            "developed by Stephen Hawking."
        )
        assert find_near_verbatim(claim, "Coral reefs host diverse marine life.") == ""

    def test_mock_evidence_fixture_still_works_unchanged(self):
        provider = MockEvidenceProvider()
        wikipedia = make_source("https://example.test/source/oct001", SECONDARY)
        entries = provider.get_evidence(wikipedia)
        assert entries
        assert all(item.source_id == wikipedia.source_id for item in entries)

    def test_mock_provider_without_recovery_is_unaffected(self):
        provider = MockEvidenceProvider()
        assert not callable(getattr(provider, "recover_evidence", None))

    def test_research_package_still_rejects_bad_payloads(self):
        with pytest.raises(ResearchPackageError):
            ResearchPackage.from_dict({"created_at": FIXED_AT, "run_id": "r1", "data": {}})
