"""Validation, serialization, and provenance tests for Stage-3 models."""

from __future__ import annotations

import pytest

from agents.qa.fact_checker.models import (
    VERDICTS,
    ClaimVerdict,
    Evidence,
    FactCheckError,
    FactCheckResult,
    evidence_id_for,
    result_id_for,
)
from agents.seo.models import Provenance

from fact_check_test_utils import FIXED_AT

ALL_EVIDENCE_TYPES = (
    "direct_statement",
    "supporting_context",
    "contradictory_statement",
    "insufficient",
)


def sample_evidence(**overrides) -> dict:
    data = {
        "evidence_id": "ev_0001",
        "source_id": "src_0001",
        "excerpt": "Octopuses have three hearts.",
        "locator": "p.3",
        "evidence_type": "direct_statement",
        "retrieved_at": FIXED_AT,
        "notes": "",
    }
    data.update(overrides)
    return data


def sample_verdict(**overrides) -> dict:
    data = {
        "claim_id": "cl_0001",
        "verdict": "PASS",
        "evidence_ids": ["ev_0001"],
        "supporting_source_ids": ["src_0001"],
        "contradiction_source_ids": [],
        "rationale": "Direct evidence supports the claim wording.",
        "confidence": 0.85,
        "critical": False,
        "checks_performed": ["stance_overall=direct"],
        "verifier": "MockVerifier",
    }
    data.update(overrides)
    return data


class TestEvidenceModel:
    def test_round_trip(self):
        evidence = Evidence.from_dict(sample_evidence())
        assert evidence.evidence_id == "ev_0001"
        assert evidence.evidence_type == "direct_statement"
        assert evidence.source_id == "src_0001"
        assert Evidence.from_dict(evidence.to_dict()) == evidence

    @pytest.mark.parametrize(
        "field", ["evidence_id", "source_id", "excerpt", "locator"]
    )
    def test_required_fields(self, field):
        with pytest.raises(FactCheckError, match="missing required field"):
            Evidence.from_dict(sample_evidence(**{field: ""}))

    @pytest.mark.parametrize("value", ["unknown", "", "PROOF"])
    def test_evidence_type_restricted(self, value):
        with pytest.raises(FactCheckError, match="evidence_type"):
            Evidence.from_dict(sample_evidence(evidence_type=value))

    @pytest.mark.parametrize("value", ALL_EVIDENCE_TYPES)
    def test_all_evidence_types_accepted(self, value):
        evidence = Evidence.from_dict(sample_evidence(evidence_type=value))
        assert evidence.evidence_type == value

    def test_bad_retrieved_at(self):
        with pytest.raises(FactCheckError, match="retrieved_at"):
            Evidence.from_dict(sample_evidence(retrieved_at="soon"))

    def test_ids_are_deterministic(self):
        assert evidence_id_for("src_1", "same excerpt") == evidence_id_for("src_1", "same excerpt")
        assert evidence_id_for("src_1", "a") != evidence_id_for("src_1", "b")
        assert evidence_id_for("src_1", "x") != evidence_id_for("src_2", "x")

    def test_non_mapping_rejected(self):
        with pytest.raises(FactCheckError, match="mapping"):
            Evidence.from_dict("nope")


class TestClaimVerdict:
    def test_round_trip(self):
        verdict = ClaimVerdict.from_dict(sample_verdict())
        assert verdict.verdict == "PASS"
        assert verdict.is_approved is True
        assert ClaimVerdict.from_dict(verdict.to_dict()) == verdict

    @pytest.mark.parametrize("verdict", VERDICTS)
    def test_all_verdicts_accepted(self, verdict):
        assert ClaimVerdict.from_dict(sample_verdict(verdict=verdict)).verdict == verdict

    def test_unknown_verdict_rejected(self):
        with pytest.raises(FactCheckError, match="verdict"):
            ClaimVerdict.from_dict(sample_verdict(verdict="MAYBE"))

    def test_review_is_not_approved(self):
        assert ClaimVerdict.from_dict(sample_verdict(verdict="REVIEW")).is_approved is False

    def test_confidence_range(self):
        with pytest.raises(FactCheckError, match="confidence"):
            ClaimVerdict.from_dict(sample_verdict(confidence=1.2))

    def test_requires_rationale(self):
        with pytest.raises(FactCheckError, match="rationale"):
            ClaimVerdict.from_dict(sample_verdict(rationale=""))


def sample_result(**overrides) -> dict:
    header = {
        "schema": "FactCheckResult",
        "schema_version": 1,
        "created_at": FIXED_AT,
        "run_id": "run-1",
        "stage": "fact_check",
        "produced_by": {"prompt_version": None, "provider": "MockEvidenceProvider", "model": "MockVerifier"},
    }
    data = {
        "result_id": "fc_0001",
        "topic_id": "t_octopus_hearts",
        "topic_title": "Why octopuses have three hearts",
        "package_id": "rp_0001",
        "package_verdict": "PASS",
        "claim_results": [sample_verdict()],
        "critical_claim_failures": [],
        "review_items": [],
        "rejected_claims": [],
        "approved_claims": ["cl_0001"],
        "evidence_issues": [],
        "provenance": {
            "generated_by": "fact_check_agent_v1",
            "source": "MockEvidenceProvider:MockVerifier",
            "created_at": FIXED_AT,
        },
    }
    for key, value in overrides.items():
        (header if key in header else data)[key] = value
    header["data"] = data
    return header


class TestFactCheckResult:
    def test_json_round_trip(self):
        result = FactCheckResult.from_dict(sample_result())
        assert result.schema == "FactCheckResult"
        assert result.schema_version == 1
        assert result.stage == "fact_check"
        restored = FactCheckResult.from_json(result.to_json())
        assert restored.to_dict() == result.to_dict()

    @pytest.mark.parametrize("verdict", VERDICTS)
    def test_all_package_verdicts_accepted(self, verdict):
        result = FactCheckResult.from_dict(sample_result(package_verdict=verdict))
        assert result.package_verdict == verdict

    def test_unknown_package_verdict_rejected(self):
        with pytest.raises(FactCheckError, match="package_verdict"):
            FactCheckResult.from_dict(sample_result(package_verdict="APPROVED"))

    def test_duplicate_claim_results_rejected(self):
        data = sample_result(claim_results=[sample_verdict(), sample_verdict()])
        with pytest.raises(FactCheckError, match="duplicate claim_id"):
            FactCheckResult.from_dict(data)

    def test_review_item_must_exist_in_results(self):
        data = sample_result(review_items=["cl_nope"])
        with pytest.raises(FactCheckError, match="missing from claim_results"):
            FactCheckResult.from_dict(data)

    def test_result_id_is_deterministic(self):
        assert result_id_for("run-1", "rp_1") == result_id_for("run-1", "rp_1")
        assert result_id_for("run-1", "rp_1") != result_id_for("run-1", "rp_2")

    def test_provenance_metadata(self):
        result = FactCheckResult.from_dict(sample_result(run_id="run-abc"))
        assert result.provenance.generated_by == "fact_check_agent_v1"
        assert result.produced_by["model"] == "MockVerifier"
        assert result.topic_id == "t_octopus_hearts"
        assert result.approved_claims == ("cl_0001",)