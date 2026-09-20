"""Unit tests for the deterministic claim classifier and confidence heuristic."""

from __future__ import annotations

import pytest

from agents.research.classifier import classify_claim, compute_confidence
from agents.research.models import Source


def source(tier: str, source_id: str = "src_x") -> Source:
    return Source(
        source_id=source_id,
        title="Mock source",
        url=f"https://example.test/source/{source_id}",
        publisher="Mock",
        source_type="news_article",
        publication_date="2024-01-01",
        credibility_tier=tier,
    )


class TestClassifier:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Octopuses have three hearts.", "fact"),
            ("The event horizon blocks all light.", "fact"),
            ("ANC headphones use small microphones.", "fact"),
            ("The hearts stop because oxygen is diverted.", "explanation"),
            ("The loop re-observes the screen step by step.", "explanation"),
            ("This works by inverting the incoming wave.", "explanation"),
            ("Octopuses are probably the cleverest.", "inference"),
            ("Humans are likely to supervise the loop.", "inference"),
            ("This anecdote is probably the most cited one.", "inference"),
            ("A probe could someday cross the horizon.", "speculation"),
            ("A pet octopus may someday open a tank lid.", "speculation"),
            ("Early adopters maybe count on agents.", "speculation"),
            ("I believe agents will reshape computing.", "opinion"),
            ("This is the most interesting animal.", "opinion"),
        ],
    )
    def test_classifies_heuristic_cases(self, text, expected):
        assert classify_claim(text) == expected


class TestConfidence:
    def test_tier_based_confidence(self):
        assert compute_confidence("fact", (source("primary"),)) == 0.85
        assert compute_confidence("fact", (source("high_quality_secondary"),)) == 0.6
        assert compute_confidence("fact", (source("discovery"),)) == 0.3

    def test_corroboration_boost(self):
        assert compute_confidence("fact", (source("primary"), source("primary"))) == 0.9
        assert compute_confidence("fact", (source("primary"), source("discovery"))) == 0.85

    def test_speculation_and_opinion_are_capped(self):
        assert compute_confidence("speculation", (source("primary"),)) == 0.4
        assert compute_confidence("opinion", (source("high_quality_secondary"),)) == 0.4

    def test_no_sources_gives_zero(self):
        assert compute_confidence("fact", ()) == 0.0