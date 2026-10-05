"""Focused offline tests for the evidence-chain audit fields.

Observability only. Nothing here may assert a verdict, a gate outcome, or a
retrieval decision; these tests exist so a finished run can be re-read and a
reviewer can tell apart:

    A. another retrieved source genuinely supports the claim
    B. the model did not declare that source
    C. the source was bound but produced no evidence
    D. evidence existed but the verifier did not credit it
    E. the source was never relevant

Everything is synthetic. No network, no model, no real source.
"""

from __future__ import annotations

import os

import pytest

from agents.qa.fact_checker.checker import FactChecker
from agents.qa.fact_checker.gates import FactCheckConfig
from agents.qa.fact_checker.models import (
    NO_EVIDENCE_ALL_REJECTED,
    NO_EVIDENCE_PROVIDER_EMPTY,
    NO_EVIDENCE_PROVIDER_FAILED,
    NO_EVIDENCE_UNKNOWN,
    ClaimEvidenceAudit,
    Evidence,
    FactCheckError,
    evidence_id_for,
)
from agents.research.models import FactClaim, Source, source_id_for
from agents.research.retriever import RetrievedSource
from fact_check_test_utils import (
    FIXED_AT,
    StaticEvidenceProvider,
    make_claim,
    make_evidence,
    make_package,
    make_source,
)
from providers.hybrid.retriever import (
    MAX_AUDIT_EXCERPT_CHARS,
    HybridRetrievalTrace,
    bounded_audit_excerpt,
)

ALPHA_URL = "https://alpha.test/article/a"
BETA_URL = "https://beta.test/article/b"

ALPHA_EXCERPT_A = "The octopus has three hearts and two of them stop beating while it swims."
ALPHA_EXCERPT_B = "Blue blood results from hemocyanin rather than hemoglobin iron."
BETA_EXCERPT = "An octopus propels itself by drawing water into and out of a mantle cavity."
# A beta excerpt that restates the claim, used where corroboration is needed.
BETA_SUPPORTING = "An octopus has three hearts as well"

# No trailing punctuation: `MockVerifier` tokenizes on whitespace, so a final
# period would become part of the token ("hearts.") and silently cap coverage.
CLAIM_TEXT = "An octopus has three hearts"


def claim_with(urls, claim_id="cl_audit", text=CLAIM_TEXT, critical=True):
    return make_claim(text, source_urls=urls, claim_id=claim_id, critical=critical)


def run_check(provider, package, config=None):
    return FactChecker(provider=provider, config=config).check(package)


def make_trace(url=ALPHA_URL, content="", title="T"):
    trace = HybridRetrievalTrace()
    trace.record_emitted(None, RetrievedSource(url=url, title=title, content=content))
    return trace


# =========================================================================== #
# 1-3. Retrieval trace: source excerpt, deterministic truncation, truncation flag
# =========================================================================== #


class TestTraceSourceExcerpt:
    def test_excerpt_persisted_for_an_emitted_source(self):
        trace = make_trace(content="Deep body text about cephalopod anatomy.")
        record = trace.to_dict()["emitted_records"][0]
        assert record["audit_excerpt"] == "Deep body text about cephalopod anatomy."
        assert record["audit_excerpt_chars"] == len(record["audit_excerpt"])
        assert record["audit_excerpt_truncated"] is False

    def test_excerpt_is_truncated_at_the_documented_cap(self):
        body = "x" * (MAX_AUDIT_EXCERPT_CHARS + 500)
        record = make_trace(content=body).to_dict()["emitted_records"][0]
        assert len(record["audit_excerpt"]) <= MAX_AUDIT_EXCERPT_CHARS
        assert record["audit_excerpt"] == "x" * MAX_AUDIT_EXCERPT_CHARS

    def test_truncation_flag_is_set_only_when_content_exceeds_the_cap(self):
        exact = "y" * MAX_AUDIT_EXCERPT_CHARS
        assert make_trace(content=exact).to_dict()["emitted_records"][0][
            "audit_excerpt_truncated"
        ] is False
        assert make_trace(content=exact + "z").to_dict()["emitted_records"][0][
            "audit_excerpt_truncated"
        ] is True

    def test_truncation_is_deterministic_across_calls(self):
        body = "deterministic " * 200
        first = make_trace(content=body).to_dict()["emitted_records"][0]
        second = make_trace(content=body).to_dict()["emitted_records"][0]
        assert first["audit_excerpt"] == second["audit_excerpt"]
        assert first["audit_excerpt_truncated"] == second["audit_excerpt_truncated"]

    def test_content_chars_still_reports_the_full_length(self):
        body = "q" * 4321
        record = make_trace(content=body).to_dict()["emitted_records"][0]
        # Truncating the audit excerpt must not change what retrieval reports.
        assert record["content_chars"] == 4321
        assert len(record["audit_excerpt"]) == MAX_AUDIT_EXCERPT_CHARS

    def test_bounded_audit_excerpt_edge_cases(self):
        assert bounded_audit_excerpt("") == ("", False)
        assert bounded_audit_excerpt("   ") == ("", False)
        assert bounded_audit_excerpt("abc", limit=0) == ("", True)
        assert bounded_audit_excerpt("abc", limit=10) == ("abc", False)

    def test_excerpt_text_is_stored_once_not_duplicated_in_the_summary(self):
        payload = make_trace(content="only once please").to_dict()
        # `emitted_summary` carries identity only, so the page text is stored
        # exactly once per emitted source.
        assert "audit_excerpt" not in payload["emitted_summary"][0]
        assert payload["emitted_records"][0]["audit_excerpt"] == "only once please"

    def test_cap_is_self_documented_in_the_artifact(self):
        payload = make_trace(content="cap me").to_dict()
        assert payload["audit_excerpt_max_chars"] == MAX_AUDIT_EXCERPT_CHARS

    def test_source_id_joins_the_trace_to_the_research_package(self):
        record = make_trace(url=ALPHA_URL).to_dict()["emitted_records"][0]
        assert record["source_id"] == source_id_for(ALPHA_URL)

    def test_existing_trace_keys_and_version_are_preserved(self):
        payload = make_trace(content="keep the old fields").to_dict()
        assert payload["schema_version"] == 3
        for key in (
            "plan",
            "per_query_hits",
            "candidates",
            "duplicates_dropped",
            "domain_filtered",
            "cheap_ok",
            "cheap_failed",
            "fallback_attempted",
            "fallback_ok",
            "fallback_skipped",
            "emitted",
            "raw_hits",
            "candidate_records",
            "emitted_records",
            "emitted_summary",
            "model_discarded",
            "not_attempted",
            "selection",
        ):
            assert key in payload, key
        record = payload["emitted_records"][0]
        for key in (
            "order",
            "index",
            "url",
            "identity",
            "title",
            "source_type",
            "retrieval_path",
            "obtained_via",
            "content_chars",
            "query",
            "rank",
            "source_tier",
        ):
            assert key in record, key


# =========================================================================== #
# 4-5. Evidence excerpt persistence and evidence_id reproducibility
# =========================================================================== #


class TestEvidenceExcerptPersistence:
    def _verdict(self):
        alpha = make_source(ALPHA_URL, "primary")
        provider = StaticEvidenceProvider(
            {ALPHA_URL: [make_evidence(alpha, ALPHA_EXCERPT_A)]}
        )
        package = make_package([claim_with([ALPHA_URL])], [alpha])
        return run_check(provider, package).claim_verdict("cl_audit")

    def test_evidence_excerpt_is_persisted_verbatim(self):
        record = self._verdict().evidence_audit.record(
            evidence_id_for(source_id_for(ALPHA_URL), ALPHA_EXCERPT_A)
        )
        assert record.excerpt == ALPHA_EXCERPT_A

    def test_excerpt_is_never_truncated(self):
        long_excerpt = "L" * 5000
        alpha = make_source(ALPHA_URL, "primary")
        provider = StaticEvidenceProvider({ALPHA_URL: [make_evidence(alpha, long_excerpt)]})
        package = make_package([claim_with([ALPHA_URL])], [alpha])
        verdict = run_check(provider, package).claim_verdict("cl_audit")
        record = verdict.evidence_audit.record(
            evidence_id_for(source_id_for(ALPHA_URL), long_excerpt)
        )
        assert len(record.excerpt) == 5000

    def test_evidence_id_is_reproducible_from_persisted_fields(self):
        record = self._verdict().evidence_audit.record(
            evidence_id_for(source_id_for(ALPHA_URL), ALPHA_EXCERPT_A)
        )
        assert record.reproduces_evidence_id()
        assert evidence_id_for(record.source_id, record.excerpt) == record.evidence_id

    def test_reproducibility_fails_when_the_excerpt_is_altered(self):
        record = self._verdict().evidence_audit.record(
            evidence_id_for(source_id_for(ALPHA_URL), ALPHA_EXCERPT_A)
        )
        tampered = type(record)(
            evidence_id=record.evidence_id,
            source_id=record.source_id,
            excerpt=record.excerpt + " extra",
            origin=record.origin,
            evidence_type=record.evidence_type,
            locator=record.locator,
            credited=record.credited,
        )
        assert not tampered.reproduces_evidence_id()

    def test_every_persisted_record_reproduces_its_id_after_a_json_round_trip(self):
        alpha = make_source(ALPHA_URL, "primary")
        beta = make_source(BETA_URL, "high_quality_secondary")
        provider = StaticEvidenceProvider(
            {
                ALPHA_URL: [
                    make_evidence(alpha, ALPHA_EXCERPT_A),
                    make_evidence(alpha, ALPHA_EXCERPT_B, locator="loc:2"),
                ],
                BETA_URL: [make_evidence(beta, BETA_EXCERPT)],
            }
        )
        package = make_package([claim_with([ALPHA_URL, BETA_URL])], [alpha, beta])
        result = run_check(provider, package)
        restored = type(result).from_json(result.to_json())
        audit = restored.claim_verdict("cl_audit").evidence_audit
        assert audit.records
        for record in audit.records:
            assert record.reproduces_evidence_id()

    def test_record_carries_existing_evidence_metadata(self):
        record = self._verdict().evidence_audit.records[0]
        assert record.locator == "loc:1"
        assert record.evidence_type == "direct_statement"
        assert record.retrieved_at == FIXED_AT
        assert record.source_id == source_id_for(ALPHA_URL)


# =========================================================================== #
# 6. Zero-evidence source records
# =========================================================================== #


class ExplodingProvider:
    def get_evidence(self, source):
        raise RuntimeError("provider exploded")


class RejectingProvider:
    def get_evidence(self, source):
        return [
            Evidence(
                evidence_id=evidence_id_for("src_does_not_exist", "dangling"),
                source_id="src_does_not_exist",
                excerpt="dangling",
                locator="loc:1",
                evidence_type="direct_statement",
            )
        ]


class TestZeroEvidenceRecords:
    def _audit_for(self, provider, tier="primary"):
        alpha = make_source(ALPHA_URL, tier)
        package = make_package([claim_with([ALPHA_URL])], [alpha])
        return run_check(provider, package).claim_verdict("cl_audit").evidence_audit

    def test_zero_evidence_source_is_recorded_explicitly(self):
        audit = self._audit_for(StaticEvidenceProvider({}))
        row = audit.source(source_id_for(ALPHA_URL))
        assert row is not None
        assert row.evidence_count == 0
        assert row.status == "no_evidence"
        assert row.is_empty

    def test_empty_provider_is_reported_as_provider_returned_empty(self):
        audit = self._audit_for(StaticEvidenceProvider({}))
        assert audit.source(source_id_for(ALPHA_URL)).reason == NO_EVIDENCE_PROVIDER_EMPTY

    def test_provider_failure_is_reported_as_provider_failed(self):
        audit = self._audit_for(ExplodingProvider())
        assert audit.source(source_id_for(ALPHA_URL)).reason == NO_EVIDENCE_PROVIDER_FAILED

    def test_all_records_dropped_is_reported_distinctly(self):
        audit = self._audit_for(RejectingProvider())
        assert audit.source(source_id_for(ALPHA_URL)).reason == NO_EVIDENCE_ALL_REJECTED

    def test_a_zero_evidence_source_is_never_reported_as_credited(self):
        audit = self._audit_for(StaticEvidenceProvider({}))
        row = audit.source(source_id_for(ALPHA_URL))
        assert row.status != "credited"
        assert row.status != "not_credited"

    def test_status_zero_count_combination_is_rejected(self):
        with pytest.raises(FactCheckError, match="zero evidence"):
            type(self._audit_for(StaticEvidenceProvider({})).sources[0])(
                source_id="src_x", evidence_count=0, status="credited"
            ).validate()

    def test_unknown_reason_is_accepted_and_never_invented(self):
        from agents.qa.fact_checker.models import SourceEvidenceAudit

        row = SourceEvidenceAudit(
            source_id="src_x", evidence_count=0, status="no_evidence", reason=NO_EVIDENCE_UNKNOWN
        )
        assert row.validate().reason == NO_EVIDENCE_UNKNOWN

    def test_every_bound_source_gets_a_row(self):
        alpha = make_source(ALPHA_URL, "primary")
        beta = make_source(BETA_URL, "high_quality_secondary")
        provider = StaticEvidenceProvider({BETA_URL: [make_evidence(beta, BETA_EXCERPT)]})
        package = make_package([claim_with([ALPHA_URL, BETA_URL])], [alpha, beta])
        audit = run_check(provider, package).claim_verdict("cl_audit").evidence_audit
        assert [row.source_id for row in audit.sources] == [
            source_id_for(ALPHA_URL),
            source_id_for(BETA_URL),
        ]
        assert audit.source(source_id_for(ALPHA_URL)).status == "no_evidence"
        assert audit.source(source_id_for(BETA_URL)).evidence_count == 1

    def test_discovery_only_source_zero_evidence_is_still_recorded(self):
        audit = self._audit_for(StaticEvidenceProvider({}), tier="discovery")
        assert audit.source(source_id_for(ALPHA_URL)).status == "no_evidence"


# =========================================================================== #
# 7. Recovery evidence attribution
# =========================================================================== #


class RecoveringProvider:
    """Provider that returns nothing then recovers one excerpt on demand."""

    provider_name = "recovering-audit"

    def __init__(self, contents, extracted=None):
        self._contents = contents
        self._extracted = extracted or {}
        self.recover_calls = []

    def get_evidence(self, source):
        return list(self._extracted.get(source.source_id, ()))

    def recover_evidence(self, source, claim):
        self.recover_calls.append((source.source_id, claim.claim_id))
        content = self._contents.get(source.source_id, "")
        excerpt = self._contents.get(f"excerpt:{source.source_id}", "")
        if not content or not excerpt:
            return []
        return [
            Evidence(
                evidence_id=evidence_id_for(source.source_id, excerpt),
                source_id=source.source_id,
                excerpt=excerpt,
                locator=f"recovered:{excerpt[:12]}",
                evidence_type="direct_statement",
                retrieved_at=FIXED_AT,
            )
        ]


class TestRecoveryAttribution:
    def _audit(self):
        alpha = make_source(ALPHA_URL, "primary")
        provider = RecoveringProvider(
            {alpha.source_id: "full page body", f"excerpt:{alpha.source_id}": ALPHA_EXCERPT_A}
        )
        package = make_package([claim_with([ALPHA_URL])], [alpha])
        return run_check(provider, package).claim_verdict("cl_audit"), provider

    def test_recovered_evidence_is_marked_with_recovery_origin(self):
        verdict, _ = self._audit()
        assert verdict.evidence_audit.recovered_evidence_ids
        record = verdict.evidence_audit.records[0]
        assert record.origin == "recovery"

    def test_extracted_evidence_is_marked_with_extraction_origin(self):
        alpha = make_source(ALPHA_URL, "primary")
        provider = StaticEvidenceProvider(
            {ALPHA_URL: [make_evidence(alpha, ALPHA_EXCERPT_A)]}
        )
        package = make_package([claim_with([ALPHA_URL])], [alpha])
        verdict = run_check(provider, package).claim_verdict("cl_audit")
        assert verdict.evidence_audit.records[0].origin == "extraction"
        assert verdict.evidence_audit.recovered_evidence_ids == ()

    def test_recovered_locator_is_preserved(self):
        verdict, _ = self._audit()
        assert verdict.evidence_audit.records[0].locator.startswith("recovered:")

    def test_recovered_record_reproduces_its_evidence_id(self):
        verdict, _ = self._audit()
        for record in verdict.evidence_audit.records:
            assert record.reproduces_evidence_id()

    def test_source_row_counts_recovery(self):
        verdict, _ = self._audit()
        row = verdict.evidence_audit.source(source_id_for(ALPHA_URL))
        assert row.recovery_attempted is True
        assert row.recovery_count == 1

    def test_no_recovery_attempted_when_nothing_eligible(self):
        alpha = make_source(ALPHA_URL, "primary")
        provider = StaticEvidenceProvider(
            {ALPHA_URL: [make_evidence(alpha, ALPHA_EXCERPT_A)]}
        )
        package = make_package([claim_with([ALPHA_URL])], [alpha])
        verdict = run_check(provider, package).claim_verdict("cl_audit")
        assert verdict.evidence_audit.source(source_id_for(ALPHA_URL)).recovery_attempted is False

    def test_recovery_attempt_is_recorded_even_when_nothing_is_recovered(self):
        alpha = make_source(ALPHA_URL, "primary")
        provider = RecoveringProvider({})
        package = make_package([claim_with([ALPHA_URL])], [alpha])
        verdict = run_check(provider, package).claim_verdict("cl_audit")
        row = verdict.evidence_audit.source(source_id_for(ALPHA_URL))
        assert row.recovery_attempted is True
        assert row.recovery_count == 0


# =========================================================================== #
# 8. claim -> source -> evidence linkage (and requirement 10)
# =========================================================================== #


class TestLinkage:
    def _result(self):
        alpha = make_source(ALPHA_URL, "primary")
        beta = make_source(BETA_URL, "high_quality_secondary")
        provider = StaticEvidenceProvider(
            {
                ALPHA_URL: [
                    make_evidence(alpha, ALPHA_EXCERPT_A, locator="loc:1"),
                    make_evidence(alpha, ALPHA_EXCERPT_B, locator="loc:2"),
                ],
                BETA_URL: [make_evidence(beta, BETA_EXCERPT)],
            }
        )
        package = make_package([claim_with([ALPHA_URL, BETA_URL])], [alpha, beta])
        return run_check(provider, package), alpha, beta

    def test_full_chain_is_reconstructable_from_the_artifact(self):
        result, alpha, beta = self._result()
        verdict = result.claim_verdict("cl_audit")
        audit = verdict.evidence_audit

        # claim_id
        assert audit.claim_id == verdict.claim_id == "cl_audit"
        # -> per-source rows, one for every source the claim is bound to
        assert {row.source_id for row in audit.sources} == {alpha.source_id, beta.source_id}
        # -> evidence_id -> excerpt, each tied to its source
        for record in audit.records:
            row = audit.source(record.source_id)
            assert row is not None
            assert record.reproduces_evidence_id()
        # -> supporting_source_ids -> verdict
        assert set(verdict.supporting_source_ids) <= {alpha.source_id, beta.source_id}
        assert verdict.verdict in ("PASS", "REVIEW", "REJECT")

    def test_two_excerpts_from_one_source_is_one_source(self):
        result, alpha, _beta = self._result()
        audit = result.claim_verdict("cl_audit").evidence_audit
        alpha_records = [r for r in audit.records if r.source_id == alpha.source_id]
        assert len(alpha_records) == 2
        assert len({r.evidence_id for r in alpha_records}) == 2
        row = audit.source(alpha.source_id)
        assert row.evidence_count == 2
        assert row.status == "credited"
        assert row.status.count("credited") == 1  # exactly one source row

    def test_one_excerpt_each_from_two_sources_is_two_sources(self):
        alpha = make_source(ALPHA_URL, "primary")
        beta = make_source(BETA_URL, "high_quality_secondary")
        provider = StaticEvidenceProvider(
            {
                ALPHA_URL: [make_evidence(alpha, ALPHA_EXCERPT_A)],
                BETA_URL: [make_evidence(beta, BETA_EXCERPT)],
            }
        )
        package = make_package([claim_with([ALPHA_URL, BETA_URL])], [alpha, beta])
        audit = run_check(provider, package).claim_verdict("cl_audit").evidence_audit
        assert audit.source(alpha.source_id).evidence_count == 1
        assert audit.source(beta.source_id).evidence_count == 1
        # Total records (2) and total sources (2) are separately readable.
        assert len(audit.records) == 2
        assert len({row.source_id for row in audit.sources}) == 2

    def test_evidence_count_is_never_exposed_as_a_corroboration_count(self):
        result, alpha, beta = self._result()
        verdict = result.claim_verdict("cl_audit")
        audit = verdict.evidence_audit
        # Alpha supplied TWO excerpts; beta supplied one that does not settle
        # the claim. So the package holds 3 evidence records but only ONE
        # credited source.
        assert len(audit.records) == 3
        assert len(verdict.supporting_source_ids) == 1
        row = audit.source(alpha.source_id)
        assert row.evidence_count == 2
        assert row.status == "credited"
        # The audit exposes no corroboration field at all.
        assert not any("corroborat" in key for key in row.to_dict())

    def test_credited_flag_reflects_source_credit_not_record_credit(self):
        result, alpha, _beta = self._result()
        audit = result.claim_verdict("cl_audit").evidence_audit
        supporting = set(result.claim_verdict("cl_audit").supporting_source_ids)
        for record in audit.records:
            assert record.credited is (record.source_id in supporting)

    def test_not_credited_source_with_evidence_is_distinguishable(self):
        alpha = make_source(ALPHA_URL, "primary")
        beta = make_source(BETA_URL, "high_quality_secondary")
        provider = StaticEvidenceProvider(
            {
                ALPHA_URL: [make_evidence(alpha, ALPHA_EXCERPT_A)],
                BETA_URL: [
                    make_evidence(beta, BETA_EXCERPT, evidence_type="insufficient")
                ],
            }
        )
        package = make_package([claim_with([ALPHA_URL, BETA_URL])], [alpha, beta])
        verdict = run_check(provider, package).claim_verdict("cl_audit")
        audit = verdict.evidence_audit
        # Beta produced evidence but was not credited -> case D.
        beta_row = audit.source(beta.source_id)
        assert beta_row.evidence_count == 1
        assert beta_row.status == "not_credited"
        assert beta.source_id not in verdict.supporting_source_ids


# =========================================================================== #
# 9-10. No raw model output, no secrets
# =========================================================================== #


SENTINELS = [
    "GEMINI_API_KEY",
    "FIRECRAWL_API_KEY",
    "SEARXNG_BASE_URL",
]


class TestNoSecretsOrRawModelOutput:
    def test_audit_contains_no_env_var_names_or_values(self):
        for name in SENTINELS:
            os.environ.pop(name, None)
        os.environ["GEMINI_API_KEY"] = "sentinel-gemini-key-value"
        try:
            alpha = make_source(ALPHA_URL, "primary")
            provider = StaticEvidenceProvider(
                {ALPHA_URL: [make_evidence(alpha, ALPHA_EXCERPT_A)]}
            )
            package = make_package([claim_with([ALPHA_URL])], [alpha])
            blob = run_check(provider, package).to_json()
            assert "sentinel-gemini-key-value" not in blob
            assert "GEMINI_API_KEY" not in blob
        finally:
            os.environ.pop("GEMINI_API_KEY", None)

    def test_audit_carries_no_free_form_model_output_field(self):
        alpha = make_source(ALPHA_URL, "primary")
        provider = StaticEvidenceProvider({ALPHA_URL: [make_evidence(alpha, ALPHA_EXCERPT_A)]})
        package = make_package([claim_with([ALPHA_URL])], [alpha])
        audit = run_check(provider, package).claim_verdict("cl_audit").evidence_audit
        for row in audit.to_dict()["records"]:
            assert set(row) == {
                "evidence_id",
                "source_id",
                "excerpt",
                "origin",
                "evidence_type",
                "locator",
                "credited",
                "retrieved_at",
                "notes",
            }
        for row in audit.to_dict()["sources"]:
            assert set(row) == {
                "source_id",
                "evidence_count",
                "status",
                "reason",
                "recovery_attempted",
                "recovery_count",
            }

    def test_audit_stores_only_a_bounded_source_excerpt_in_the_trace(self):
        payload = make_trace(content="s" * 6000).to_dict()
        # 6000 chars of page text in, at most MAX_AUDIT_EXCERPT_CHARS out, and
        # the full length is still reported separately as `content_chars`.
        assert len(payload["emitted_records"][0]["audit_excerpt"]) == MAX_AUDIT_EXCERPT_CHARS
        assert payload["emitted_records"][0]["content_chars"] == 6000

    def test_trace_stores_page_text_verbatim_and_is_not_a_redaction_boundary(self):
        # Honest statement of the boundary: the excerpt is the real page prefix,
        # stored verbatim, because the audit must be quotable. The trace is NOT
        # a secret-redaction layer, and must not be treated as one. What the
        # trace does guarantee is that no credential material is ever written by
        # the recording code itself (see the env-var test above).
        body = "AKIAIOSFODNN7EXAMPLE" * 5
        record = make_trace(content=body).to_dict()["emitted_records"][0]
        assert record["audit_excerpt"] == body[:MAX_AUDIT_EXCERPT_CHARS]


# =========================================================================== #
# 11-12. Gate and artifact semantics unchanged
# =========================================================================== #


class TestGatesUnchanged:
    def test_corroboration_min_sources_is_still_two(self):
        assert FactCheckConfig().corroboration_min_sources == 2

    def test_critically_supported_critical_claim_with_two_sources_still_passes(self):
        alpha = make_source(ALPHA_URL, "primary")
        beta = make_source(BETA_URL, "high_quality_secondary")
        provider = StaticEvidenceProvider(
            {
                ALPHA_URL: [make_evidence(alpha, ALPHA_EXCERPT_A)],
                BETA_URL: [make_evidence(beta, BETA_SUPPORTING)],
            }
        )
        package = make_package([claim_with([ALPHA_URL, BETA_URL])], [alpha, beta])
        verdict = run_check(provider, package).claim_verdict("cl_audit")
        assert len(verdict.supporting_source_ids) == 2
        assert verdict.verdict == "PASS"

    def test_single_source_critical_claim_still_goes_to_review(self):
        alpha = make_source(ALPHA_URL, "primary")
        provider = StaticEvidenceProvider(
            {ALPHA_URL: [make_evidence(alpha, ALPHA_EXCERPT_A)]}
        )
        package = make_package([claim_with([ALPHA_URL])], [alpha])
        verdict = run_check(provider, package).claim_verdict("cl_audit")
        assert verdict.verdict == "REVIEW"
        assert "corroborated by only 1" in verdict.rationale

    def test_audit_does_not_change_the_verdict_versus_no_audit(self):
        alpha = make_source(ALPHA_URL, "primary")
        provider = StaticEvidenceProvider(
            {ALPHA_URL: [make_evidence(alpha, ALPHA_EXCERPT_A)]}
        )
        package = make_package([claim_with([ALPHA_URL])], [alpha])
        with_audit = run_check(provider, package)
        stripped = type(with_audit)(
            result_id=with_audit.result_id,
            topic_id=with_audit.topic_id,
            topic_title=with_audit.topic_title,
            package_id=with_audit.package_id,
            claim_results=tuple(
                type(v)(
                    claim_id=v.claim_id,
                    verdict=v.verdict,
                    evidence_ids=v.evidence_ids,
                    supporting_source_ids=v.supporting_source_ids,
                    contradiction_source_ids=v.contradiction_source_ids,
                    rationale=v.rationale,
                    confidence=v.confidence,
                    critical=v.critical,
                    checks_performed=v.checks_performed,
                    verifier=v.verifier,
                )
                for v in with_audit.claim_results
            ),
            package_verdict=with_audit.package_verdict,
            critical_claim_failures=with_audit.critical_claim_failures,
            review_items=with_audit.review_items,
            rejected_claims=with_audit.rejected_claims,
            approved_claims=with_audit.approved_claims,
            evidence_issues=with_audit.evidence_issues,
            provenance=with_audit.provenance,
            created_at=with_audit.created_at,
            run_id=with_audit.run_id,
            produced_by=with_audit.produced_by,
        )
        assert stripped.to_dict() == with_audit.to_dict() | {
            "data": {
                **with_audit.to_dict()["data"],
                "claim_results": [
                    {**entry, "evidence_audit": None}
                    for entry in with_audit.to_dict()["data"]["claim_results"]
                ],
            }
        }

    def test_approved_claims_gate_input_is_unchanged(self):
        alpha = make_source(ALPHA_URL, "primary")
        beta = make_source(BETA_URL, "high_quality_secondary")
        provider = StaticEvidenceProvider(
            {
                ALPHA_URL: [make_evidence(alpha, ALPHA_EXCERPT_A)],
                BETA_URL: [make_evidence(beta, BETA_SUPPORTING)],
            }
        )
        package = make_package([claim_with([ALPHA_URL, BETA_URL])], [alpha, beta])
        result = run_check(provider, package)
        assert result.approved_claims == ("cl_audit",)
        assert result.package_verdict == "PASS"


class TestAuditModelValidation:
    def test_unknown_origin_is_rejected(self):
        from agents.qa.fact_checker.models import EvidenceAuditRecord

        with pytest.raises(FactCheckError, match="origin"):
            EvidenceAuditRecord(
                evidence_id="ev_1",
                source_id="src_1",
                excerpt="text",
                origin="guess",
                evidence_type="direct_statement",
                locator="loc:1",
                credited=False,
            ).validate()

    def test_unknown_status_is_rejected(self):
        from agents.qa.fact_checker.models import SourceEvidenceAudit

        with pytest.raises(FactCheckError, match="status"):
            SourceEvidenceAudit(
                source_id="src_1", evidence_count=1, status="maybe"
            ).validate()

    def test_unknown_no_evidence_reason_is_rejected(self):
        from agents.qa.fact_checker.models import SourceEvidenceAudit

        with pytest.raises(FactCheckError, match="reason"):
            SourceEvidenceAudit(
                source_id="src_1",
                evidence_count=0,
                status="no_evidence",
                reason="provider_had_a_bad_day",
            ).validate()

    def test_negative_evidence_count_is_rejected(self):
        from agents.qa.fact_checker.models import SourceEvidenceAudit

        with pytest.raises(FactCheckError, match="evidence_count"):
            SourceEvidenceAudit(
                source_id="src_1", evidence_count=-1, status="credited"
            ).validate()

    def test_audit_lookup_helpers(self):
        audit = ClaimEvidenceAudit(
            claim_id="cl_1",
            sources=(),
        )
        assert audit.record("ev_missing") is None
        assert audit.source("src_missing") is None