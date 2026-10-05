"""Offline tests for the RetrievedSource adapter.

These assert the two things that matter most for Stage 2: that the mapping is
deterministic and contract-exact, and that a search result is never upgraded
into something it is not.
"""

from __future__ import annotations

import pytest

from agents.research.models import source_id_for
from agents.research.retriever import RetrievedSource
from experiments.retrieval.adapter import (
    SEARCH_RESULT_SOURCE_TYPE,
    iso_date_or_empty,
    result_to_source,
    results_to_sources,
    source_ids,
    truncate_content,
)
from experiments.retrieval.searxng_client import SearchResult

CONTRACT_FIELDS = {
    "url",
    "title",
    "publisher",
    "publication_date",
    "source_type",
    "content",
    "domain",
    "retrieved_at",
}


def _result(**overrides) -> SearchResult:
    base = {
        "url": "https://en.wikipedia.org/wiki/Triple_point",
        "normalized_url": "https://en.wikipedia.org/wiki/Triple_point",
        "title": "Triple point - Wikipedia",
        "snippet": "A snippet about the triple point of water.",
        "engine": "duckduckgo",
        "engines": ("duckduckgo",),
        "score": 1.0,
    }
    base.update(overrides)
    return SearchResult(**base)


class TestContractShape:
    def test_produces_a_real_retrieved_source_with_exactly_the_contract_fields(self):
        mapped = result_to_source(_result())
        assert mapped is not None
        assert isinstance(mapped.source, RetrievedSource)
        # The production contract must not have grown an experiment-only field.
        assert set(mapped.source.__dataclass_fields__) == CONTRACT_FIELDS

    def test_contract_is_unmodified_by_the_experiment(self):
        """The adapter imports RetrievedSource; it must not redefine or subclass it."""
        import agents.research.retriever as production

        assert RetrievedSource is production.RetrievedSource
        assert len(CONTRACT_FIELDS) == 8

    def test_source_type_marks_discovery_not_a_fetched_page(self):
        mapped = result_to_source(_result())
        assert mapped is not None
        assert mapped.source.source_type == SEARCH_RESULT_SOURCE_TYPE == "search_result"
        # Distinct from the Firecrawl retriever's value, so a consumer can tell.
        assert mapped.source.source_type != "web_page"


class TestMapping:
    def test_url_is_the_canonical_form_and_raw_is_preserved(self):
        mapped = result_to_source(
            _result(
                url="https://En.Wikipedia.org/wiki/Triple_point/",
                normalized_url="https://en.wikipedia.org/wiki/Triple_point",
            )
        )
        assert mapped is not None
        assert mapped.source.url == "https://en.wikipedia.org/wiki/Triple_point"
        assert mapped.original_url == "https://En.Wikipedia.org/wiki/Triple_point/"

    def test_source_id_is_the_repository_helper_on_the_contract_url(self):
        mapped = result_to_source(_result())
        assert mapped is not None
        assert mapped.source_id == source_id_for(mapped.source.url)
        assert mapped.source_id.startswith("src_")

    def test_source_id_is_deterministic_across_calls(self):
        ids = {result_to_source(_result()).source_id for _ in range(25)}
        assert len(ids) == 1

    def test_source_id_differs_per_url(self):
        a = result_to_source(_result(normalized_url="https://a.test/x"))
        b = result_to_source(_result(normalized_url="https://b.test/x"))
        assert a is not None and b is not None
        assert a.source_id != b.source_id

    def test_publisher_and_domain_are_the_host(self):
        mapped = result_to_source(_result(normalized_url="https://en.wikipedia.org/wiki/Triple_point"))
        assert mapped is not None
        assert mapped.source.domain == "en.wikipedia.org"
        assert mapped.source.publisher == "en.wikipedia.org"

    def test_title_is_preserved_verbatim(self):
        mapped = result_to_source(_result(title="  Triple point - Wikipedia  "))
        assert mapped is not None
        assert mapped.source.title == "Triple point - Wikipedia"

    def test_unusable_url_maps_to_none(self):
        assert result_to_source(_result(normalized_url="")) is None


class TestNoFabrication:
    def test_content_is_empty_by_default(self):
        """A search snippet is not page content, so it must not fill `content`."""
        mapped = result_to_source(_result(snippet="A snippet."))
        assert mapped is not None
        assert mapped.source.content == ""
        assert mapped.came_from_fetch is False

    def test_snippet_only_when_explicitly_requested(self):
        mapped = result_to_source(_result(snippet="A snippet."), include_snippet=True)
        assert mapped is not None
        assert mapped.source.content == "A snippet."
        # Still not a fetch, and the type still says so.
        assert mapped.came_from_fetch is False
        assert mapped.source.source_type == SEARCH_RESULT_SOURCE_TYPE

    def test_supplied_fetched_content_is_used_and_marked(self):
        mapped = result_to_source(_result(), content="Real fetched page text.")
        assert mapped is not None
        assert mapped.source.content == "Real fetched page text."
        assert mapped.came_from_fetch is True

    def test_fetched_content_wins_over_snippet(self):
        mapped = result_to_source(_result(snippet="snippet"), content="fetched", include_snippet=True)
        assert mapped is not None
        assert mapped.source.content == "fetched"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("2024-01-15", "2024-01-15"),
            ("2024-01-15T10:30:00+00:00", "2024-01-15T10:30:00+00:00"),
            ("2024-01-15T10:30:00Z", "2024-01-15T10:30:00Z"),
            ("2024-01-15T10:30:00", "2024-01-15T10:30:00"),
            ("1986", ""),  # bare year: valid for CLAIM dates, not for publication_date
            ("", ""),
            ("3 days ago", ""),
            ("Jan 2024", ""),
            ("yesterday", ""),
            ("2024-13-45", ""),
            ("not a date", ""),
        ],
    )
    def test_only_real_iso_dates_reach_publication_date(self, raw, expected):
        """Free-text engine dates must not be laundered into a typed ISO field."""
        assert iso_date_or_empty(raw) == expected
        mapped = result_to_source(_result(published_date_raw=raw))
        assert mapped is not None
        assert mapped.source.publication_date == expected

    def test_missing_everything_stays_empty(self):
        mapped = result_to_source(
            SearchResult(url="https://a.test/x", normalized_url="https://a.test/x")
        )
        assert mapped is not None
        source = mapped.source
        assert source.title == ""
        assert source.publication_date == ""
        assert source.content == ""
        assert source.retrieved_at == ""
        # Only url/domain/publisher/source_type are populated, all derivable facts.
        assert source.domain == "a.test"
        assert source.publisher == "a.test"
        assert source.source_type == SEARCH_RESULT_SOURCE_TYPE

    def test_author_is_not_promoted_to_publisher(self):
        mapped = result_to_source(_result(author="Jane Doe"))
        assert mapped is not None
        assert mapped.source.publisher == "en.wikipedia.org"
        assert "Jane" not in mapped.source.publisher

    def test_retrieved_at_is_empty_unless_supplied(self):
        assert result_to_source(_result()).source.retrieved_at == ""
        stamped = result_to_source(_result(), retrieved_at="2026-10-01T00:00:00+00:00")
        assert stamped is not None
        assert stamped.source.retrieved_at == "2026-10-01T00:00:00+00:00"

    def test_no_tier_field_exists_so_nothing_can_be_promoted(self):
        """Tiering is model-judged; the contract has no tier field to abuse."""
        assert "credibility_tier" not in CONTRACT_FIELDS
        assert "tier" not in CONTRACT_FIELDS

    def test_snippet_does_not_leak_into_other_fields(self):
        mapped = result_to_source(_result(snippet="SECRET-SNIPPET"), include_snippet=True)
        assert mapped is not None
        assert mapped.source.url.find("SECRET") == -1
        assert mapped.source.title.find("SECRET") == -1
        assert mapped.source.domain.find("SECRET") == -1


class TestTruncation:
    def test_short_text_untouched(self):
        assert truncate_content("hello", 100) == "hello"

    def test_empty_text(self):
        assert truncate_content("", 100) == ""
        assert truncate_content("   ", 100) == ""

    def test_truncation_is_marked_not_silent(self):
        out = truncate_content("x" * 50, 10)
        assert out.endswith(" …[truncated]")
        assert out.startswith("x" * 10)

    def test_adapter_respects_max_content_chars(self):
        mapped = result_to_source(_result(), content="y" * 5000, max_content_chars=100)
        assert mapped is not None
        assert mapped.source.content.endswith(" …[truncated]")
        assert len(mapped.source.content) < 200


class TestBatchMapping:
    def test_maps_one_to_one_in_rank_order(self):
        results = [_result(normalized_url=f"https://a.test/{i}") for i in range(4)]
        mapped = results_to_sources(results)
        assert [m.source.url for m in mapped] == [f"https://a.test/{i}" for i in range(4)]

    def test_unusable_results_are_dropped_not_zeroed(self):
        results = [_result(normalized_url=""), _result(normalized_url="https://a.test/ok")]
        mapped = results_to_sources(results)
        assert len(mapped) == 1
        assert mapped[0].source.url == "https://a.test/ok"

    def test_contents_are_keyed_by_canonical_url(self):
        results = [_result(normalized_url="https://a.test/p")]
        mapped = results_to_sources(results, contents={"https://a.test/p": "fetched body"})
        assert mapped[0].source.content == "fetched body"
        assert mapped[0].came_from_fetch is True

    def test_url_without_contents_gets_no_content_not_a_neighbours(self):
        results = [_result(normalized_url="https://a.test/p"), _result(normalized_url="https://a.test/q")]
        mapped = results_to_sources(results, contents={"https://a.test/p": "only for p"})
        assert mapped[0].source.content == "only for p"
        assert mapped[1].source.content == "", "no content leaks between URLs"

    def test_no_sources_are_attached_to_each_other(self):
        """The adapter is strictly one result -> one source, with no cross-linking."""
        results = [_result(normalized_url=f"https://a.test/{i}") for i in range(3)]
        mapped = results_to_sources(results)
        assert len(mapped) == 3
        for item in mapped:
            # RetrievedSource has no field that could carry a related source.
            assert not hasattr(item.source, "related_sources")
            assert not hasattr(item.source, "corroborating_sources")
            assert not hasattr(item.source, "supports")

    def test_source_ids_helper_accepts_both_shapes(self):
        raw = RetrievedSource(url="https://a.test/p")
        mapped = result_to_source(_result(normalized_url="https://a.test/p"))
        assert mapped is not None
        assert source_ids([raw, mapped]) == (
            source_id_for("https://a.test/p"),
            source_id_for("https://a.test/p"),
        )

    def test_source_ids_skips_empty_urls(self):
        assert source_ids([RetrievedSource(url="")]) == ()