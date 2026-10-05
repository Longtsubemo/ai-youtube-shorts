"""Stage 2 claim-date robustness: a bare year must not abort the package.

Real Gemini answers with the year it actually knows. On run
``e2e-20260930-008`` the model emitted ``dates[].date = "1986"`` for an
AI-hallucination history claim, and ``ClaimDate.from_dict`` raised
``ResearchPackageError: date must be an ISO date/datetime, got '1986'``,
killing the *entire* ResearchPackage — a single imprecise date on one claim
threw away every other good claim in the run.

The contract (``docs/DATA_CONTRACTS.md`` §5) defines ``dates`` as "zero-or-more
{date_kind, date} entries" with no partial-date representation, so a year-only
value can only be *omitted*. This test file locks in that decision:

  A. valid ISO date survives unchanged
  B. valid ISO datetime survives unchanged (incl. trailing Z)
  C. bare year "1986" is dropped, and never widened to a fake exact date
  D. other malformed values still raise (the tolerance is NOT a blanket drop)
  E. one bad year does not take down the rest of the package
  F. the strict artifact loader is unchanged (from_dict still raises)
  G. legacy artifacts still load and round-trip
  H. multi-source claim binding is untouched by the date change
  I. no unrelated source is ever auto-attached
  J. determinism
"""

from __future__ import annotations

import json

import pytest

from agents.research.agent import ResearchAgent
from agents.research.models import (
    ClaimDate,
    FactClaim,
    ResearchPackage,
    ResearchPackageError,
    is_bare_year,
    normalize_source_url,
)
from agents.research.provider import ClaimItem, SourceCandidate, TopicDescriptor

URL_A = "https://example.test/papers/hallucination"
URL_B = "https://example.test/news/hallucination"

CLAIM = "Early chatbots produced plausible-sounding falsehoods."
CLAIM_TWO = "Modern chatbots still produce confident falsehoods."


def _selection() -> dict:
    return {
        "data": {
            "selected_topic_id": "t_hallucination",
            "selected_topic": {"topic_id": "t_hallucination", "title": "Why AI Makes Things Up"},
        }
    }


def _candidate(
    url: str,
    claim_text: str,
    dates: tuple[dict, ...],
    *extra_urls: str,
    critical: bool = False,
) -> SourceCandidate:
    """A dated claim candidate.

    `critical` defaults to False: this module tests *date* handling end to end
    (parsing, dropping, round-tripping), not the Stage-3 corroboration policy.
    A blanket `critical=True` default made every single-source fixture a
    critical claim that the producer-side guard rejects by design, so
    criticality is declared explicitly per test. Date behavior is unaffected.
    """
    return SourceCandidate(
        title="Page",
        url=url,
        publisher="pub",
        source_type="web_page",
        publication_date="",
        credibility_tier="high_quality_secondary",
        notes="",
        claims=(
            ClaimItem(
                claim_text=claim_text,
                dates=dates,
                suggested_claim_type="fact",
                critical=critical,
                also_supported_by_urls=tuple(extra_urls),
            ),
        ),
        retrieved_at="2026-01-01T00:00:00+00:00",
    )


class _CandidateProvider:
    def __init__(self, candidates: list[SourceCandidate]) -> None:
        self._candidates = candidates

    def search(self, topic: TopicDescriptor) -> list[SourceCandidate]:
        return self._candidates

    def knowable_topics(self) -> list[str]:
        return ["t1"]


def _run(*candidates: SourceCandidate) -> ResearchPackage:
    return ResearchAgent(provider=_CandidateProvider(list(candidates)), run_id="r1").research(_selection())


def _claim(package: ResearchPackage, text: str) -> FactClaim:
    for claim in package.claims:
        if claim.claim_text == text:
            return claim
    raise AssertionError(f"claim not found: {text}")


# --------------------------------------------------------------------------- #
# A. valid ISO date
# --------------------------------------------------------------------------- #


class TestValidIsoDates:
    def test_iso_date_preserved_exactly(self):
        parsed = ClaimDate.from_dict({"date_kind": "discovery_date", "date": "1986-06-15"})
        assert parsed.date == "1986-06-15"
        assert parsed.date_kind == "discovery_date"

    def test_iso_date_flows_through_the_agent(self):
        package = _run(
            _candidate(URL_A, CLAIM, ({"date_kind": "discovery_date", "date": "1986-06-15"},))
        )
        claim = _claim(package, CLAIM)
        assert len(claim.dates) == 1
        assert claim.dates[0].date == "1986-06-15"
        assert claim.dates[0].date_kind == "discovery_date"

    def test_optional_parser_keeps_valid_date(self):
        assert ClaimDate.from_optional_dict({"date_kind": "event_date", "date": "2019-12-31"}) is not None


# --------------------------------------------------------------------------- #
# B. valid ISO datetime
# --------------------------------------------------------------------------- #


class TestValidIsoDatetimes:
    def test_iso_datetime_preserved(self):
        parsed = ClaimDate.from_dict({"date_kind": "announcement_date", "date": "2023-05-04T11:30:00"})
        assert parsed.date == "2023-05-04T11:30:00"

    def test_trailing_z_datetime_preserved(self):
        parsed = ClaimDate.from_dict({"date_kind": "event_date", "date": "2023-05-04T11:30:00Z"})
        assert parsed.date == "2023-05-04T11:30:00Z"

    def test_datetime_flows_through_the_agent(self):
        package = _run(
            _candidate(URL_A, CLAIM, ({"date_kind": "event_date", "date": "2023-05-04T11:30:00Z"},))
        )
        assert _claim(package, CLAIM).dates[0].date == "2023-05-04T11:30:00Z"

    def test_optional_parser_keeps_valid_datetime(self):
        parsed = ClaimDate.from_optional_dict({"date_kind": "event_date", "date": "2023-05-04T11:30:00"})
        assert parsed is not None and parsed.date == "2023-05-04T11:30:00"


# --------------------------------------------------------------------------- #
# C. bare year
# --------------------------------------------------------------------------- #


class TestBareYearIsOmitted:
    def test_bare_year_detected(self):
        assert is_bare_year("1986") is True
        assert is_bare_year(" 1986 ") is True

    def test_optional_parser_drops_bare_year(self):
        assert ClaimDate.from_optional_dict({"date_kind": "discovery_date", "date": "1986"}) is None

    def test_bare_year_does_not_abort_the_package(self):
        package = _run(_candidate(URL_A, CLAIM, ({"date_kind": "discovery_date", "date": "1986"},)))
        assert len(package.claims) == 1
        assert package.claims[0].claim_text == CLAIM

    def test_bare_year_entry_is_omitted_not_fabricated(self):
        package = _run(_candidate(URL_A, CLAIM, ({"date_kind": "discovery_date", "date": "1986"},)))
        claim = _claim(package, CLAIM)
        assert claim.dates == ()
        # no invented "1986-01-01" and no 1986 anywhere in the serialized artifact
        assert "1986" not in claim.to_dict()["dates"].__str__()
        assert "1986-01-01" not in str(claim.to_dict())

    def test_bare_year_mixed_with_a_valid_date_keeps_the_valid_one(self):
        package = _run(
            _candidate(
                URL_A,
                CLAIM,
                (
                    {"date_kind": "discovery_date", "date": "1986"},
                    {"date_kind": "publication_date", "date": "2018-06-19"},
                ),
            )
        )
        dates = _claim(package, CLAIM).dates
        assert len(dates) == 1
        assert dates[0].date == "2018-06-19"

    def test_multiple_bare_years_all_dropped(self):
        package = _run(
            _candidate(
                URL_A,
                CLAIM,
                (
                    {"date_kind": "discovery_date", "date": "1986"},
                    {"date_kind": "event_date", "date": "1995"},
                ),
            )
        )
        assert _claim(package, CLAIM).dates == ()

    @pytest.mark.parametrize("year", ["1000", "1986", "2024", "9999"])
    def test_plausible_years_are_dropped(self, year):
        assert ClaimDate.from_optional_dict({"date_kind": "discovery_date", "date": year}) is None

    @pytest.mark.parametrize("text", ["0999", "0001", "abcd", "19x6", "12345", "19860", ""])
    def test_out_of_range_or_nonnumeric_is_not_a_bare_year(self, text):
        assert is_bare_year(text) is False


# --------------------------------------------------------------------------- #
# D. other malformed values still raise
# --------------------------------------------------------------------------- #


class TestOtherMalformedStillRaise:
    @pytest.mark.parametrize(
        "date",
        [
            "sometime in 1986",
            "1986 or so",
            "circa 1986",
            "19th century",
            "1986-13-45",
            "not-a-date",
            "1986.0",
            "0",
            "12",
            "123",
        ],
    )
    def test_optional_parser_still_raises_on_junk(self, date):
        with pytest.raises(ResearchPackageError):
            ClaimDate.from_optional_dict({"date_kind": "discovery_date", "date": date})

    def test_partial_iso_is_still_rejected(self):
        # "1986-07" is not a bare year and not a full ISO date: still an error.
        with pytest.raises(ResearchPackageError):
            ClaimDate.from_optional_dict({"date_kind": "discovery_date", "date": "1986-07"})

    def test_unknown_date_kind_still_raises(self):
        with pytest.raises(ResearchPackageError):
            ClaimDate.from_optional_dict({"date_kind": "invented_date", "date": "1986-06-15"})

    def test_bad_date_kind_still_raises_in_the_agent(self):
        with pytest.raises(ResearchPackageError):
            _run(_candidate(URL_A, CLAIM, ({"date_kind": "invented_date", "date": "2018-06-19"},)))

    def test_junk_date_still_aborts_the_agent_run(self):
        with pytest.raises(ResearchPackageError):
            _run(_candidate(URL_A, CLAIM, ({"date_kind": "discovery_date", "date": "sometime in 1986"},)))

    def test_missing_date_still_raises(self):
        with pytest.raises(ResearchPackageError):
            ClaimDate.from_optional_dict({"date_kind": "discovery_date"})

    def test_non_mapping_still_raises(self):
        with pytest.raises(ResearchPackageError):
            ClaimDate.from_optional_dict("1986")


# --------------------------------------------------------------------------- #
# E. one bad year does not take down the rest of the package
# --------------------------------------------------------------------------- #


class TestPackageSurvivesOneBadYear:
    def test_other_claims_and_sources_survive(self):
        package = _run(
            _candidate(URL_A, CLAIM, ({"date_kind": "discovery_date", "date": "1986"},)),
            _candidate(URL_B, CLAIM_TWO, ({"date_kind": "discovery_date", "date": "2003-11-01"},)),
        )
        assert len(package.claims) == 2
        assert len(package.sources) == 2
        assert _claim(package, CLAIM).dates == ()
        assert _claim(package, CLAIM_TWO).dates[0].date == "2003-11-01"

    def test_status_still_ready_for_fact_check(self):
        package = _run(_candidate(URL_A, CLAIM, ({"date_kind": "discovery_date", "date": "1986"},)))
        assert package.research_status in ("provisional", "ready_for_fact_check")

    def test_artifact_round_trips_after_a_dropped_year(self):
        package = _run(
            _candidate(
                URL_A,
                CLAIM,
                (
                    {"date_kind": "discovery_date", "date": "1986"},
                    {"date_kind": "publication_date", "date": "2018-06-19"},
                ),
            )
        )
        reloaded = ResearchPackage.from_json(package.to_json())
        assert reloaded.claims[0].dates[0].date == "2018-06-19"
        assert len(reloaded.claims[0].dates) == 1


# --------------------------------------------------------------------------- #
# F. the strict artifact loader is unchanged
# --------------------------------------------------------------------------- #


class TestStrictLoaderUnchanged:
    def test_from_dict_still_raises_on_bare_year(self):
        # The tolerance is scoped to the model-boundary build path only.
        # Artifact loading stays strict so bad data on disk is never silently
        # accepted.
        with pytest.raises(ResearchPackageError):
            ClaimDate.from_dict({"date_kind": "discovery_date", "date": "1986"})


# --------------------------------------------------------------------------- #
# G. legacy artifacts still load
# --------------------------------------------------------------------------- #


class TestLegacyArtifactsStillLoad:
    def test_legacy_empty_dates_list(self):
        package = _run(_candidate(URL_A, CLAIM, ()))
        reloaded = ResearchPackage.from_json(package.to_json())
        assert reloaded.claims[0].dates == ()

    def test_legacy_full_iso_dates_list(self):
        package = _run(
            _candidate(
                URL_A,
                CLAIM,
                (
                    {"date_kind": "publication_date", "date": "2018-06-19"},
                    {"date_kind": "discovery_date", "date": "2012-10-01"},
                ),
            )
        )
        reloaded = ResearchPackage.from_json(package.to_json())
        assert [d.date for d in reloaded.claims[0].dates] == ["2018-06-19", "2012-10-01"]

    def test_real_artifact_from_a_previous_run_still_loads(self):
        from pathlib import Path

        candidates = sorted(Path("data").glob("*/research/research_package.json"))
        if not candidates:
            pytest.skip("no research_package.json artifacts on disk")
        for path in candidates:
            ResearchPackage.from_json(path.read_text(encoding="utf-8"))

    def test_real_claim_date_model_round_trips(self):
        parsed = ClaimDate.from_dict({"date_kind": "event_date", "date": "2020-01-02"})
        assert ClaimDate.from_dict(parsed.to_dict()) == parsed


# --------------------------------------------------------------------------- #
# H. multi-source binding is untouched
# --------------------------------------------------------------------------- #


class TestMultiSourceBindingUnaffected:
    def test_declared_extra_url_still_binds_second_source(self):
        package = _run(
            _candidate(URL_A, CLAIM, (), URL_B, critical=True),
            _candidate(URL_B, CLAIM_TWO, ()),
        )
        claim = _claim(package, CLAIM)
        assert claim.critical is True
        assert len(claim.source_ids) == 2
        assert claim.source_ids[0] != claim.source_ids[1]

    def test_critical_claim_keeps_dates_with_required_sources(self):
        """A critical, dated claim binds two eligible sources and keeps its
        date: criticality does not disturb date extraction."""
        dated = {"date_kind": "discovery_date", "date": "1986-02-14"}
        package = _run(
            _candidate(URL_A, CLAIM, (dated,), URL_B, critical=True),
            _candidate(URL_B, CLAIM_TWO, ()),
        )
        claim = _claim(package, CLAIM)
        assert claim.critical is True
        assert [d.date for d in claim.dates] == ["1986-02-14"]
        assert len(claim.source_ids) == 2

    def test_bare_year_plus_extra_url_still_binds(self):
        package = _run(
            _candidate(URL_A, CLAIM, ({"date_kind": "discovery_date", "date": "1986"},), URL_B),
            _candidate(URL_B, CLAIM_TWO, ()),
        )
        claim = _claim(package, CLAIM)
        assert claim.dates == ()
        assert len(claim.source_ids) == 2

    def test_unretrieved_extra_url_still_raises(self):
        with pytest.raises(ResearchPackageError):
            _run(_candidate(URL_A, CLAIM, (), "https://example.test/never-retrieved"))

    def test_unrelated_source_is_not_auto_attached(self):
        # Two retrieved sources exist, but the model never declared the second
        # one as support, so neither claim may gain it.
        package = _run(
            _candidate(URL_A, CLAIM, ()),
            _candidate(URL_B, CLAIM_TWO, ()),
        )
        assert len({s.url for s in package.sources}) == 2
        assert len(_claim(package, CLAIM).source_ids) == 1
        assert len(_claim(package, CLAIM_TWO).source_ids) == 1

    def test_url_normalization_helper_unchanged(self):
        assert normalize_source_url("HTTPS://Example.test/A/") == "https://example.test/A"


# --------------------------------------------------------------------------- #
# J. determinism
# --------------------------------------------------------------------------- #


class TestDeterminism:
    def _build(self) -> ResearchPackage:
        return _run(
            _candidate(URL_A, CLAIM, ({"date_kind": "discovery_date", "date": "1986"},), URL_B),
            _candidate(URL_B, CLAIM_TWO, ()),
        )

    def test_repeated_builds_are_byte_identical(self):
        # created_at (envelope and provenance) is wall-clock by design, so
        # determinism is asserted on the content the date handling produces.
        def fingerprint(package: ResearchPackage) -> str:
            return json.dumps(
                {
                    "claims": [
                        {
                            "claim_id": c.claim_id,
                            "source_ids": list(c.source_ids),
                            "dates": [d.to_dict() for d in c.dates],
                            "critical": c.critical,
                        }
                        for c in package.claims
                    ],
                    "sources": sorted((s.source_id, s.url) for s in package.sources),
                },
                sort_keys=True,
            )

        assert fingerprint(self._build()) == fingerprint(self._build())

    def test_dropped_year_is_stable_across_repeats(self):
        assert self._build().claims[0].dates == self._build().claims[0].dates

    def test_bare_year_detection_is_stable(self):
        assert [is_bare_year(v) for v in ("1986", "1986", "junk")] == [True, True, False]
