"""Tests for the deterministic research-query planner.

Fully offline and network-free: the planner is pure string logic by design.
"""

from __future__ import annotations

import pytest

from agents.research.provider import TopicDescriptor
from providers.hybrid.query import (
    ResearchQueryPlan,
    _factual_sentences,
    _keywords,
    _title_subject,
    plan_research_queries,
)


def _topic(**kwargs) -> TopicDescriptor:
    base = {"topic_id": "t_test", "title": "", "description": ""}
    base.update(kwargs)
    return TopicDescriptor(**base)


def _rules(plan: ResearchQueryPlan) -> list[str]:
    return [query.rule for query in plan.queries]


# --------------------------------------------------------------------------- #
# Title framing
# --------------------------------------------------------------------------- #


class TestTitleSubject:
    @pytest.mark.parametrize(
        ("title", "expected"),
        [
            ("Do Black Holes Actually Shrink?", "Black Holes Shrink"),
            ("Why Is the Sky Blue?", "Sky Blue"),
            ("How Does AI Actually Think Next Words?", "AI Think Next Words"),
            ("What Causes Ocean Tides?", "Causes Ocean Tides"),
        ],
    )
    def test_strips_interrogative_frame(self, title, expected):
        assert _title_subject(title) == expected

    def test_preserves_word_order(self):
        """The planner removes framing, it does not rewrite the subject."""
        assert _title_subject("What Are the Main Causes of Ice Ages?") == (
            "Main Causes of Ice Ages"
        )

    def test_keeps_a_bare_topic_unchanged(self):
        assert _title_subject("Antikythera mechanism") == "Antikythera mechanism"

    def test_empty_title_is_empty(self):
        assert _title_subject("") == ""
        assert _title_subject("   ") == ""

    def test_title_with_only_framing_is_empty(self):
        """Nothing left to search for: the caller must not invent a query."""
        assert _title_subject("Why?") == ""


# --------------------------------------------------------------------------- #
# Description handling
# --------------------------------------------------------------------------- #


class TestFactualSentences:
    def test_drops_meta_sentences_about_the_video(self):
        sentences = _factual_sentences(
            "Hawking radiation makes black holes lose mass. This topic covers astrophysics."
        )
        assert sentences == ["Hawking radiation makes black holes lose mass"]

    def test_strips_instruction_opener(self):
        assert _factual_sentences(
            "An intuitive look at tokenization and probability."
        ) == ["tokenization and probability"]

    def test_strips_abstract_noun_frame(self):
        """'the theoretical concept of X' is production language, not subject matter."""
        assert _factual_sentences(
            "The theoretical concept of Hawking radiation explains mass loss."
        ) == ["Hawking radiation explains mass loss"]

    def test_keeps_short_content_when_frame_would_eat_it(self):
        """The frame is only stripped when real content survives."""
        assert _factual_sentences("The idea of it.") == ["The idea of it"]

    def test_empty_description_yields_nothing(self):
        assert _factual_sentences("") == []


# --------------------------------------------------------------------------- #
# Keywords
# --------------------------------------------------------------------------- #


class TestKeywords:
    def test_drops_stopwords_and_intensifiers(self):
        keywords = _keywords(["Do Black Holes Actually Shrink?"])
        assert "Do" not in keywords
        assert "Actually" not in keywords
        assert "Black Holes Shrink" in keywords

    def test_deduplicates_preserving_first_seen_order(self):
        assert _keywords(["mass loss", "mass loss again"]) == "mass loss"

    def test_respects_the_limit(self):
        assert len(_keywords([" ".join(f"term{i}" for i in range(100))], limit=5).split()) == 5


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #


class TestPlanResearchQueries:
    def test_separates_a_rhetorical_title_from_a_factual_query(self):
        """The headline case: the title is a poor query, the description is not."""
        plan = plan_research_queries(
            _topic(
                title="Do Black Holes Actually Shrink?",
                description="Explore the theoretical concept of Hawking radiation and "
                "how black holes might slowly lose mass over unimaginable spans of time.",
            )
        )
        factual = plan.queries[0]
        assert factual.rule == "description_factual_clause"
        assert "Hawking radiation" in factual.text
        assert "Do" not in factual.text
        # The raw title is still present, but only as the lowest-ranked variant.
        assert plan.texts[-1] == "Do Black Holes Actually Shrink?"

    def test_ranks_factual_clause_above_the_verbatim_title(self):
        plan = plan_research_queries(
            _topic(title="How Does AI Actually Think Next Words?", description="Tokenization.")
        )
        assert _rules(plan).index("description_factual_clause") < _rules(plan).index("title_verbatim")

    def test_is_deterministic(self):
        topic = _topic(title="Why Is the Sky Blue?", description="Rayleigh scattering.")
        assert plan_research_queries(topic).texts == plan_research_queries(topic).texts

    def test_emits_no_duplicate_queries(self):
        plan = plan_research_queries(_topic(title="Tides", description="Tides."))
        assert len(set(plan.texts)) == len(plan.texts)

    def test_falls_back_to_the_title_when_description_is_empty(self):
        plan = plan_research_queries(_topic(title="Why Is the Sky Blue?", description=""))
        assert plan.texts
        assert "Sky Blue" in plan.texts[0]

    def test_empty_topic_plans_nothing_rather_than_inventing(self):
        plan = plan_research_queries(_topic(title="", description=""))
        assert plan.queries == ()
        assert plan.texts == ()
        assert "no research query" in plan.as_trace()

    def test_respects_max_queries(self):
        plan = plan_research_queries(
            _topic(title="Why Is the Sky Blue?", description="Rayleigh scattering. Molecules."), max_queries=1
        )
        assert len(plan.queries) == 1

    def test_caps_at_least_one_query(self):
        plan = plan_research_queries(_topic(title="Tides", description="Tides."), max_queries=0)
        assert len(plan.queries) == 1

    def test_trace_names_the_rule_for_each_query(self):
        trace = plan_research_queries(
            _topic(title="Why Is the Sky Blue?", description="Rayleigh scattering.")
        ).as_trace()
        assert "title_subject=" in trace
        assert "description_factual_clause=" in trace


class TestNoFabrication:
    def test_queries_only_contain_words_from_the_topic(self):
        """The planner may not introduce subject matter the topic did not supply."""
        topic = _topic(
            title="Do Black Holes Actually Shrink?",
            description="Explore Hawking radiation.",
        )
        punctuation = ".,?!'\""
        topic_words = {
            token.strip(punctuation)
            for token in f"{topic.title} {topic.description}".lower().split()
        }
        for query in plan_research_queries(topic).texts:
            for token in query.lower().split():
                cleaned = token.strip(punctuation)
                if cleaned:
                    assert cleaned in topic_words, cleaned
