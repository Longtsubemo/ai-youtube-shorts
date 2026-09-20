"""Validation, serialization, and provenance tests for research models."""

from __future__ import annotations

import pytest

from agents.research.models import (
    PROVISIONAL,
    ClaimDate,
    FactClaim,
    ResearchPackage,
    ResearchPackageError,
    Source,
    claim_id_for,
    source_id_for,
)


def sample_source(**overrides) -> dict:
    data = {
        "source_id": "src_0001",
        "title": "Circulatory anatomy (mock fixture)",
        "url": "https://example.test/source/001",
        "publisher": "Mock Marine Biology Journal",
        "source_type": "scientific_paper",
        "publication_date": "2019-04-12",
        "credibility_tier": "primary",
        "retrieved_at": "2026-09-20T08:00:00+00:00",
        "notes": "Mock fixture - not real.",
    }
    data.update(overrides)
    return data


def sample_claim(**overrides) -> dict:
    data = {
        "claim_id": "cl_0001",
        "claim_text": "Octopuses have three hearts.",
        "claim_type": "fact",
        "source_ids": ["src_0001"],
        "confidence": 0.85,
        "verification_status": PROVISIONAL,
        "dates": [],
        "notes": "",
    }
    data.update(overrides)
    return data


_HEADER_KEYS = ("schema", "schema_version", "created_at", "run_id", "stage", "produced_by")


def sample_package(**overrides) -> dict:
    header = {
        "schema": "ResearchPackage",
        "schema_version": 1,
        "created_at": "2026-09-20T08:00:00+00:00",
        "run_id": "run-1",
        "stage": "research",
        "produced_by": {"prompt_version": None, "provider": None, "model": None},
    }
    data = {
        "package_id": "rp_0001",
        "topic_id": "t_octopus_hearts",
        "topic_title": "Why octopuses have three hearts",
        "research_status": "ready_for_fact_check",
        "claims": [sample_claim()],
        "sources": [sample_source()],
        "research_notes": ["Mock evidence only."],
        "provenance": {
            "generated_by": "research_agent_v1",
            "source": "MockResearchProvider",
            "created_at": "2026-09-20T08:00:00+00:00",
        },
    }
    for key, value in overrides.items():
        (header if key in _HEADER_KEYS else data)[key] = value
    header["data"] = data
    return header


class TestSourceValidation:
    def test_from_dict_round_trip(self):
        source = Source.from_dict(sample_source())
        assert source.source_id == "src_0001"
        assert source.credibility_tier == "primary"
        assert source.is_non_discovery is True
        assert Source.from_dict(source.to_dict()) == source

    @pytest.mark.parametrize("field", ["source_id", "title", "url", "publisher", "source_type"])
    def test_required_fields(self, field):
        with pytest.raises(ResearchPackageError):
            Source.from_dict(sample_source(**{field: ""}))

    def test_bad_credibility_tier(self):
        with pytest.raises(ResearchPackageError, match="credibility_tier"):
            Source.from_dict(sample_source(credibility_tier="random_blog"))

    def test_bad_url_scheme(self):
        with pytest.raises(ResearchPackageError, match="URL"):
            Source.from_dict(sample_source(url="not-a-url"))

    def test_bad_publication_date(self):
        with pytest.raises(ResearchPackageError, match="publication_date"):
            Source.from_dict(sample_source(publication_date="yesterday-ish"))

    def test_discovery_not_non_discovery(self):
        source = Source.from_dict(sample_source(credibility_tier="discovery"))
        assert source.is_non_discovery is False


class TestClaimDateValidation:
    def test_date_kind_restricted(self):
        with pytest.raises(ResearchPackageError, match="date_kind"):
            ClaimDate.from_dict({"date_kind": "publish_on", "date": "2020-01-01"})

    @pytest.mark.parametrize("kind", ["publication_date", "discovery_date", "announcement_date", "event_date"])
    def test_all_kinds_accepted(self, kind):
        date = ClaimDate.from_dict({"date_kind": kind, "date": "2020-01-01"})
        assert date.date_kind == kind

    def test_date_required(self):
        with pytest.raises(ResearchPackageError, match="missing required field: date"):
            ClaimDate.from_dict({"date_kind": "event_date", "date": ""})


class TestFactClaimValidation:
    def test_from_dict_round_trip(self):
        claim = FactClaim.from_dict(sample_claim())
        assert claim.verification_status == PROVISIONAL
        assert claim.confidence == 0.85
        assert FactClaim.from_dict(claim.to_dict()) == claim

    @pytest.mark.parametrize("claim_type", ["fact", "inference", "explanation", "speculation", "opinion"])
    def test_all_claim_types_accepted(self, claim_type):
        claim = FactClaim.from_dict(sample_claim(claim_type=claim_type))
        assert claim.claim_type == claim_type

    def test_unknown_claim_type(self):
        with pytest.raises(ResearchPackageError, match="claim_type"):
            FactClaim.from_dict(sample_claim(claim_type="gossip"))

    def test_confidence_out_of_range(self):
        with pytest.raises(ResearchPackageError, match="confidence"):
            FactClaim.from_dict(sample_claim(confidence=1.5))

    def test_cannot_mark_verified(self):
        with pytest.raises(ResearchPackageError, match="verification_status"):
            FactClaim.from_dict(sample_claim(verification_status="verified"))

    def test_dates_round_trip(self):
        claim = FactClaim.from_dict(
            sample_claim(dates=[{"date_kind": "announcement_date", "date": "2019-04-10"}])
        )
        assert claim.dates[0].date_kind == "announcement_date"
        assert claim.dates[0].date == "2019-04-10"


class TestResearchPackageValidation:
    def test_build_and_validate(self):
        source = Source.from_dict(sample_source())
        claim = FactClaim.from_dict(sample_claim())
        package = ResearchPackage.build(
            topic_id="t_octopus_hearts",
            topic_title="Why octopuses have three hearts",
            research_status="ready_for_fact_check",
            claims=[claim],
            sources=[source],
            run_id="run-1",
            created_at="2026-09-20T08:00:00+00:00",
        ).validate()
        assert package.stage == "research"

    @pytest.mark.parametrize("status", ["provisional", "insufficient_evidence", "ready_for_fact_check"])
    def test_all_research_statuses_accepted(self, status):
        package = ResearchPackage.from_dict(sample_package(research_status=status))
        assert package.research_status == status

    def test_unknown_research_status(self):
        with pytest.raises(ResearchPackageError, match="research_status"):
            ResearchPackage.from_dict(sample_package(research_status="approved"))

    def test_missing_source_reference_rejected(self):
        data = sample_package(
            claims=[sample_claim(source_ids=["src_ghost001"])],
        )
        with pytest.raises(ResearchPackageError, match="unknown source"):
            ResearchPackage.from_dict(data)

    def test_duplicate_source_rejected(self):
        data = sample_package(sources=[sample_source(), sample_source()])
        with pytest.raises(ResearchPackageError, match="duplicate source_id"):
            ResearchPackage.from_dict(data)

    def test_envelope_and_provenance(self):
        package = ResearchPackage.from_dict(
            sample_package(run_id="run-abc", provenance={"generated_by": "research_agent_v1"})
        )
        assert package.schema == "ResearchPackage"
        assert package.schema_version == 1
        assert package.stage == "research"
        assert package.run_id == "run-abc"
        assert package.produced_by == {"prompt_version": None, "provider": None, "model": None}
        assert package.created_at == "2026-09-20T08:00:00+00:00"
        assert package.provenance.generated_by == "research_agent_v1"

    def test_json_round_trip(self):
        package = ResearchPackage.from_dict(sample_package())
        restored = ResearchPackage.from_json(package.to_json())
        assert restored.to_dict() == package.to_dict()

    def test_traceability_helpers(self):
        package = ResearchPackage.from_dict(sample_package())
        assert package.source_by_id("src_0001").title == sample_source()["title"]
        assert package.claims_for_source("src_0001")[0].claim_id == "cl_0001"
        assert package.claim_by_id("cl_0001").claim_type == "fact"
        assert package.source_by_id("nope") is None
        assert package.claim_by_id("nope") is None


def test_id_helpers_are_stable():
    assert source_id_for("https://example.test/source/001") == source_id_for("https://example.test/source/001")
    assert claim_id_for("Same text.") == claim_id_for("Same text.")
    assert source_id_for("a") != source_id_for("b")