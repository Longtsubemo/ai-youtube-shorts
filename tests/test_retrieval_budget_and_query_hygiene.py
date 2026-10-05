"""Tests for retrieval budget allocation and query hygiene.

Two defects found by the `e2e-20261004-193251-v3trace` run:

1. **Budget starvation.** The source cap was one global budget consumed in strict
   query order, so the planner's first query took all 8 slots and the remaining
   three planned queries (including the verbatim title) contributed nothing:
   40 of 52 candidates were never fetched.
2. **Format-vocabulary leakage.** "animation-friendly explainer", copied straight
   out of the Stage-1 description, put animation-tool marketing pages into the
   candidate pool.

Every test is network-free: the searcher, fetcher and Firecrawl fallback are
injected fakes.
"""

from __future__ import annotations

import copy

import pytest

from agents.research.provider import TopicDescriptor
from providers.firecrawl.client import FirecrawlResult
from providers.hybrid.query import (
    _factual_sentences,
    _is_production_token,
    _keywords,
    plan_research_queries,
)
from providers.hybrid.retriever import (
    DEFAULT_MAX_FIRECRAWL_FALLBACKS,
    DEFAULT_MAX_SOURCES,
    HybridResearchRetriever,
    _interleave_by_query,
)
from providers.html.fetcher import FetchedPage

BODY = "Real prose about the subject. " * 12

Q_FACTUAL = "Rayleigh scattering"
Q_SUBJECT = "Sky Blue"
Q_KEYWORDS = "Sky Blue Rayleigh scattering"
Q_VERBATIM = "Why Is the Sky Blue?"


def _topic(**kwargs) -> TopicDescriptor:
    base = {"topic_id": "t_test", "title": "Why Is the Sky Blue?", "description": "Rayleigh scattering."}
    base.update(kwargs)
    return TopicDescriptor(**base)


class FakeSearcher:
    """Per-query URL table; records the queries it was actually asked."""

    def __init__(self, table: dict[str, list[str]] | None = None) -> None:
        self.table = table or {}
        self.queries: list[str] = []

    def search(self, query: str, *, max_results: int | None = None):
        self.queries.append(query)
        from providers.searxng.client import SearchResult

        return tuple(
            SearchResult(url=url, title=f"title {url}", snippet="s") for url in self.table.get(query, [])
        )


class FakeFetcher:
    def __init__(self, failing: set[str] | None = None) -> None:
        self.failing = failing or set()
        self.fetched: list[str] = []

    def fetch(self, url: str) -> FetchedPage:
        self.fetched.append(url)
        if url in self.failing:
            return FetchedPage(url=url, ok=False, error_kind="blocked", domain="")
        return FetchedPage(url=url, ok=True, title=f"page {url}", text=BODY, domain="example.test")


class FakeFallback:
    def __init__(self) -> None:
        self.scraped: list[str] = []

    def scrape(self, url: str, *, only_main_content: bool = True) -> FirecrawlResult:
        self.scraped.append(url)
        return FirecrawlResult(url=url, title="rendered", markdown=BODY)


def _four_query_table(per_query: int) -> dict[str, list[str]]:
    """Each planned query gets its own disjoint block of URLs, so nothing dedupes."""
    return {
        Q_FACTUAL: [f"https://q1.test/{i}" for i in range(per_query)],
        Q_SUBJECT: [f"https://q2.test/{i}" for i in range(per_query)],
        Q_KEYWORDS: [f"https://q3.test/{i}" for i in range(per_query)],
        Q_VERBATIM: [f"https://q4.test/{i}" for i in range(per_query)],
    }


def _retriever(table=None, **kwargs) -> HybridResearchRetriever:
    kwargs.setdefault("searcher", FakeSearcher(table if table is not None else _four_query_table(6)))
    kwargs.setdefault("fetcher", FakeFetcher())
    return HybridResearchRetriever(**kwargs)


# --------------------------------------------------------------------------- #
# Budget allocation: no first-query starvation
# --------------------------------------------------------------------------- #


class TestBudgetAllocation:
    def test_a_flooding_first_query_cannot_take_every_slot(self):
        """The defect: query 1 alone used to fill all 8 slots."""
        table = _four_query_table(6)
        retriever = _retriever(table, enable_fallback=False)
        sources = retriever.retrieve(_topic())
        assert len(sources) == DEFAULT_MAX_SOURCES

        contributed = {source.url.split("/")[2] for source in sources}
        assert len(contributed) > 1, f"only {contributed} contributed"

    def test_every_planned_query_is_attempted_before_any_gets_a_second(self):
        retriever = _retriever(enable_fallback=False)
        retriever.retrieve(_topic())
        fetched = retriever._fetcher.fetched  # noqa: SLF001 - asserting on the fake
        # Round one of the interleave is one candidate per query, so with a cap of
        # 8 and 4 queries the first four attempts are the four queries' best hit.
        assert [url.split("/")[2] for url in fetched[:4]] == [
            "q1.test",
            "q2.test",
            "q3.test",
            "q4.test",
        ]

    def test_later_queries_still_contribute_sources(self):
        retriever = _retriever(enable_fallback=False)
        sources = retriever.retrieve(_topic())
        assert any(source.url.startswith("https://q4.test/") for source in sources)
        assert any(source.url.startswith("https://q2.test/") for source in sources)

    def test_the_top_ranked_query_still_goes_first(self):
        """Query precedence is preserved: the highest-precision query leads."""
        retriever = _retriever(enable_fallback=False)
        retriever.retrieve(_topic())
        assert retriever._fetcher.fetched[0] == "https://q1.test/0"  # noqa: SLF001

    def test_rank_order_inside_each_query_is_preserved(self):
        retriever = _retriever(enable_fallback=False)
        retriever.retrieve(_topic())
        fetched = retriever._fetcher.fetched  # noqa: SLF001
        q1 = [url for url in fetched if url.startswith("https://q1.test/")]
        assert q1 == sorted(q1, key=lambda u: int(u.rsplit("/", 1)[1]))

    def test_ordering_is_deterministic_across_runs(self):
        first = [source.url for source in _retriever(enable_fallback=False).retrieve(_topic())]
        second = [source.url for source in _retriever(enable_fallback=False).retrieve(_topic())]
        assert first == second

    def test_failed_fetches_do_not_consume_a_slot(self):
        """The cap counts emitted sources, so a broken page still lets others in."""
        table = _four_query_table(6)
        fetcher = FakeFetcher(failing={f"https://q1.test/{i}" for i in range(6)})
        retriever = _retriever(table, fetcher=fetcher, enable_fallback=False)
        sources = retriever.retrieve(_topic())
        assert len(sources) == DEFAULT_MAX_SOURCES
        assert any(source.url.startswith("https://q2.test/") for source in sources)

    def test_still_records_the_untouched_candidates(self):
        retriever = _retriever(enable_fallback=False)
        retriever.retrieve(_topic())
        trace = retriever.last_trace
        assert len(trace.not_attempted) == trace.candidates - trace.emitted
        assert all(entry["reason"] == "source_cap_reached" for entry in trace.not_attempted)

    def test_every_candidate_is_visited_exactly_once(self):
        """The interleave is a permutation, so nothing is fetched twice or dropped."""
        retriever = _retriever(enable_fallback=False)
        retriever.retrieve(_topic())
        records = retriever.last_trace.candidate_records
        touched = [
            record["index"]
            for record in records
            if record["cheap_attempted"] or record["not_attempted_reason"]
        ]
        assert sorted(touched) == list(range(len(records)))


class TestInterleaveHelper:
    def _records(self, queries: list[str]) -> list[dict]:
        return [{"index": i, "query": q} for i, q in enumerate(queries)]

    def test_interleaves_round_robin_in_plan_order(self):
        order = _interleave_by_query(self._records(["a", "b", "c", "a", "b", "a"]))
        assert order == [0, 1, 2, 3, 4, 5]

    def test_handles_unequal_bucket_sizes(self):
        order = _interleave_by_query(self._records(["a", "a", "a", "b"]))
        assert order == [0, 3, 1, 2]

    def test_is_a_permutation_of_the_input(self):
        records = self._records(["a", "b", "a", "c", "b", "a"])
        assert sorted(_interleave_by_query(records)) == list(range(len(records)))

    def test_empty_input_is_empty(self):
        assert _interleave_by_query([]) == []

    def test_single_query_is_unchanged(self):
        assert _interleave_by_query(self._records(["a"] * 5)) == [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------- #
# Budgets must not have moved
# --------------------------------------------------------------------------- #


class TestBudgetsUnchanged:
    def test_source_cap_is_exactly_eight(self):
        assert DEFAULT_MAX_SOURCES == 8

    def test_firecrawl_budget_is_exactly_three(self):
        assert DEFAULT_MAX_FIRECRAWL_FALLBACKS == 3

    def test_default_retriever_still_emits_at_most_eight(self):
        retriever = _retriever(_four_query_table(20), enable_fallback=False)
        assert len(retriever.retrieve(_topic())) == DEFAULT_MAX_SOURCES

    def test_fallback_budget_is_still_consumed_at_most_three_times(self):
        table = {Q_FACTUAL: [f"https://q1.test/{i}" for i in range(10)]}
        fetcher = FakeFetcher(failing={url for urls in table.values() for url in urls})
        fallback = FakeFallback()
        retriever = _retriever(table, fetcher=fetcher, fallback=fallback, enable_fallback=True)
        retriever.retrieve(_topic())
        assert len(fallback.scraped) == DEFAULT_MAX_FIRECRAWL_FALLBACKS

    def test_fallback_attempts_still_prefer_the_capped_order(self):
        """Reordering candidates must not reorder which pages get the expensive tier."""
        table = {Q_FACTUAL: [f"https://q1.test/{i}" for i in range(10)]}
        fetcher = FakeFetcher(failing={url for urls in table.values() for url in urls})
        fallback = FakeFallback()
        retriever = _retriever(table, fetcher=fetcher, fallback=fallback, enable_fallback=True)
        retriever.retrieve(_topic())
        assert fallback.scraped[:3] == [
            "https://q1.test/0",
            "https://q1.test/1",
            "https://q1.test/2",
        ]


# --------------------------------------------------------------------------- #
# Query hygiene: production vocabulary must not reach a search engine
# --------------------------------------------------------------------------- #

DEEP_SEA = (
    "An animation-friendly explainer on extreme pressure at oceanic depths and how "
    "gas behaves under thousands of pounds of force"
)


class TestProductionVocabularyIsStripped:
    def test_the_exact_e2e_query_no_longer_carries_the_format_words(self):
        assert _factual_sentences(DEEP_SEA) == [
            "extreme pressure at oceanic depths and how gas behaves under thousands "
            "of pounds of force"
        ]

    def test_generated_plan_queries_are_free_of_format_words(self):
        topic = _topic(
            title="What Happens to Air Bubbles at the Bottom of the Ocean?",
            description=DEEP_SEA,
        )
        for text in plan_research_queries(topic).texts:
            for token in text.lower().replace("?", " ").replace(",", " ").split():
                assert not _is_production_token(token), f"{token!r} leaked into {text!r}"

    def test_actual_topical_terms_survive(self):
        topic = _topic(
            title="What Happens to Air Bubbles at the Bottom of the Ocean?",
            description=DEEP_SEA,
        )
        queries = " ".join(plan_research_queries(topic).texts).lower()
        for term in ("extreme pressure", "oceanic depths", "gas", "bubbles", "ocean"):
            assert term in queries, term

    def test_it_is_a_category_not_a_single_word(self):
        """Other artifact words go too, not just the one that happened to fail."""
        sentence = (
            "A cinematic documentary episode with narration and a voiceover covers "
            "enzyme denaturation and protein folding."
        )
        result = " ".join(_factual_sentences(sentence))
        for word in ("cinematic", "documentary", "episode", "narration", "voiceover"):
            assert word not in result, word
        for term in ("enzyme denaturation", "protein folding"):
            assert term in result, term

    def test_hyphenated_compounds_are_handled_generically(self):
        assert _is_production_token("animation-friendly")
        assert _is_production_token("video-driven")
        assert _is_production_token("Explainer")
        assert _is_production_token("animations")
        assert _is_production_token("narrated")
        assert not _is_production_token("peer-reviewed")
        assert not _is_production_token("state-of-the-art")

    def test_ordinary_compounds_are_left_verbatim(self):
        assert _keywords(["a peer-reviewed study of tidal forces"]) == (
            "peer-reviewed study tidal forces"
        )

    def test_ambiguous_subject_words_are_not_over_filtered(self):
        """Over-filtering would delete real topical terms, which is the other failure."""
        for word in (
            "film",
            "motion",
            "visual",
            "graphics",
            "narrative",
            "score",
            "story",
            "stories",
            "series",
            "segment",
            "script",
            "editing",
            "cut",
        ):
            assert not _is_production_token(word), word
        result = " ".join(
            _factual_sentences(
                "Cohesion between water molecules produces a film across a bubble, "
                "and the visual cortex perceives motion."
            )
        )
        for term in ("film", "visual cortex", "motion", "bubble"):
            assert term in result, term

    @pytest.mark.parametrize(
        ("text", "term"),
        [
            ("The Fourier series converges for smooth functions.", "series"),
            ("CRISPR gene editing rewrites a DNA segment.", "editing"),
            ("CRISPR gene editing rewrites a DNA segment.", "segment"),
            ("A Python script parses the telemetry log.", "script"),
            ("The bee's abdomen has a striped segment pattern.", "segment"),
        ],
    )
    def test_scientific_terms_that_look_like_format_words_survive(self, text, term):
        assert not _is_production_token(term)
        assert term in _keywords([text])
        assert term in " ".join(_factual_sentences(text))

    def test_keywords_drop_format_words(self):
        keywords = _keywords(["an animation-friendly explainer about ocean pressure"])
        assert "animation-friendly" not in keywords
        assert "explainer" not in keywords
        assert "ocean pressure" in keywords

    def test_a_clause_that_is_only_about_the_artifact_is_dropped(self):
        assert _factual_sentences("An animation-friendly explainer video.") == []

    def test_orphaned_function_words_are_cleaned_up(self):
        clause = _factual_sentences(DEEP_SEA)[0]
        assert not clause.lower().startswith(("an ", "on ", "the "))
        assert clause[0].islower()

    def test_a_normal_description_is_returned_byte_identical(self):
        text = "Hawking radiation makes black holes lose mass over long times."
        assert _factual_sentences(text) == ["Hawking radiation makes black holes lose mass over long times"]


class TestPlannerBehaviourPreserved:
    NORMAL = (
        "Explore the theoretical concept of Hawking radiation and how black holes "
        "might slowly lose mass over unimaginable spans of time."
    )

    def test_rules_and_precedence_are_unchanged_for_normal_topics(self):
        rules = [q.rule for q in plan_research_queries(_topic()).queries]
        assert rules == [
            "description_factual_clause",
            "title_subject",
            "title_description_keywords",
            "title_verbatim",
        ]

    def test_normal_topic_produces_the_unchanged_queries(self):
        plan = plan_research_queries(
            _topic(title="Do Black Holes Actually Shrink?", description=self.NORMAL)
        )
        assert plan.texts[0] == (
            "Hawking radiation and how black holes might slowly lose mass over "
            "unimaginable spans of time"
        )
        assert plan.texts[1] == "Black Holes Shrink"
        assert plan.texts[-1] == "Do Black Holes Actually Shrink?"

    def test_planning_is_still_deterministic(self):
        topic = _topic(title="Why Is the Sky Blue?", description=DEEP_SEA)
        assert plan_research_queries(topic).texts == plan_research_queries(topic).texts

    def test_no_query_is_invented_and_the_title_still_appears_verbatim(self):
        plan = plan_research_queries(_topic(title="Why Is the Sky Blue?", description="Rayleigh scattering."))
        assert plan.texts[-1] == "Why Is the Sky Blue?"
        assert plan_research_queries(_topic(title="", description="")).queries == ()

    def test_queries_still_only_use_words_from_the_topic(self):
        """Sanitization may remove words; it may never introduce any."""
        topic = _topic(
            title="What Happens to Air Bubbles at the Bottom of the Ocean?",
            description=DEEP_SEA,
        )
        topic_words = {
            token.strip(".,?!'\"")
            for token in f"{topic.title} {topic.description}".lower().split()
        }
        for query in plan_research_queries(topic).texts:
            for token in query.lower().split():
                cleaned = token.strip(".,?!'\"-")
                if cleaned:
                    assert cleaned in topic_words, cleaned


# --------------------------------------------------------------------------- #
# No accidental mutation
# --------------------------------------------------------------------------- #


class TestNoMutation:
    def test_planning_does_not_mutate_the_topic(self):
        topic = _topic(title="What Happens to Air Bubbles at the Bottom of the Ocean?", description=DEEP_SEA)
        before = copy.deepcopy((topic.topic_id, topic.title, topic.description))
        plan_research_queries(topic)
        assert (topic.topic_id, topic.title, topic.description) == before

    def test_sanitizing_does_not_mutate_the_input_string(self):
        before = DEEP_SEA
        _factual_sentences(DEEP_SEA)
        assert DEEP_SEA == before

    def test_planning_does_not_mutate_the_previous_plan(self):
        plan = plan_research_queries(_topic())
        snapshot = copy.deepcopy(plan)
        plan_research_queries(_topic(title="Why Do Octopuses Have Three Hearts?", description="Hemocyanin."))
        assert plan == snapshot

    def test_retrieval_does_not_mutate_the_searcher_table(self):
        table = _four_query_table(6)
        snapshot = copy.deepcopy(table)
        _retriever(table, enable_fallback=False).retrieve(_topic())
        assert table == snapshot

    def test_retrieval_does_not_mutate_the_fetcher_failure_set(self):
        failing = {"https://q1.test/0"}
        snapshot = set(failing)
        fetcher = FakeFetcher(failing=failing)
        _retriever(_four_query_table(6), fetcher=fetcher, enable_fallback=False).retrieve(_topic())
        assert failing == snapshot

    def test_emitted_sources_are_not_shared_between_retrieve_calls(self):
        retriever = _retriever(enable_fallback=False)
        first = retriever.retrieve(_topic())
        first.append("mutation")  # type: ignore[arg-type]
        second = retriever.retrieve(_topic())
        assert "mutation" not in second

    def test_trace_records_are_not_aliased_between_calls(self):
        retriever = _retriever(enable_fallback=False)
        retriever.retrieve(_topic())
        first = retriever.last_trace
        retriever.retrieve(_topic())
        assert first is not retriever.last_trace
        assert first.emitted == DEFAULT_MAX_SOURCES


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))