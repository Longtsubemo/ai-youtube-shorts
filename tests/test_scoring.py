"""Scoring tests (requirements 2, 3, 9) + competition/history handling."""

from __future__ import annotations

import pytest

from agents.seo.competition import CompetitionMappingError, competition_to_opportunity
from agents.seo.history import HistoricalRecord, SelectionHistory
from agents.seo.scorer import TopicScorer


def _expected_base(candidate, config) -> float:
    w = config.scoring.weights
    return (
        w["interest"] * candidate.signals.interest
        + w["trend_momentum"] * candidate.signals.trend_momentum
        + w["competition"] * (1 - candidate.signals.competition)  # inverse = opportunity
        + w["curiosity"] * candidate.content_scores.curiosity
        + w["explainer_potential"] * candidate.content_scores.explainer_potential
        + w["visual_potential"] * candidate.content_scores.visual_potential
        + w["audience_fit"] * candidate.content_scores.audience_fit
    )


class TestCompetitionOpportunity:
    def test_inverse_map_reduces_opportunity(self):
        assert competition_to_opportunity(0.0, "inverse") == pytest.approx(1.0)
        assert competition_to_opportunity(1.0, "inverse") == pytest.approx(0.0)
        assert competition_to_opportunity(0.5, "inverse") == pytest.approx(0.5)
        assert competition_to_opportunity(0.8, "inverse") == pytest.approx(0.2)

    def test_unknown_transformation_raises(self):
        with pytest.raises(CompetitionMappingError):
            competition_to_opportunity(0.5, "made_up")

    def test_out_of_range_raises(self):
        with pytest.raises(CompetitionMappingError):
            competition_to_opportunity(1.5, "inverse")


class TestScoreCalculation:
    def test_base_score_matches_weighted_formula(self, default_config, candidate):
        scorer = TopicScorer(default_config)
        scored = scorer.score(candidate)  # no history -> base only
        assert scored.total_score == pytest.approx(_expected_base(candidate, default_config))
        assert scored.historical_fit is None

    def test_weights_are_respected_single_weight(self, candidate_factory, config_factory):
        candidate = candidate_factory()
        weights = {
            "interest": 1.0,
            "trend_momentum": 0.0,
            "competition": 0.0,
            "curiosity": 0.0,
            "explainer_potential": 0.0,
            "visual_potential": 0.0,
            "audience_fit": 0.0,
        }
        config = config_factory(weights=weights)
        scored = TopicScorer(config).score(candidate)
        assert scored.total_score == pytest.approx(candidate.signals.interest)

    def test_weights_are_respected_content_only(self, candidate_factory, config_factory):
        candidate = candidate_factory()
        weights = {
            "interest": 0.0,
            "trend_momentum": 0.0,
            "competition": 0.0,
            "curiosity": 1.0,
            "explainer_potential": 0.0,
            "visual_potential": 0.0,
            "audience_fit": 0.0,
        }
        config = config_factory(weights=weights)
        scored = TopicScorer(config).score(candidate)
        assert scored.total_score == pytest.approx(candidate.content_scores.curiosity)

    def test_competition_weight_scores_opportunity(self, candidate_factory, config_factory):
        candidate = candidate_factory()
        weights = {
            "interest": 0.0,
            "trend_momentum": 0.0,
            "competition": 1.0,
            "curiosity": 0.0,
            "explainer_potential": 0.0,
            "visual_potential": 0.0,
            "audience_fit": 0.0,
        }
        config = config_factory(weights=weights)
        scored = TopicScorer(config).score(candidate)
        assert scored.total_score == pytest.approx(1 - candidate.signals.competition)

    def test_cold_start_no_history_keeps_historical_fit_null(self, default_config, candidate):
        scored = TopicScorer(default_config).score(candidate, SelectionHistory())
        assert scored.historical_fit is None
        assert scored.total_score == pytest.approx(_expected_base(candidate, default_config))

    def test_history_sets_historical_fit_and_blends(self, default_config, candidate, config_factory):
        history = SelectionHistory.from_records(
            [
                HistoricalRecord(video_id="v1", topic_id="t1", pillar="science", topic_type="surprising_fact", views=4000),
                HistoricalRecord(video_id="v2", topic_id="t2", pillar="science", topic_type="surprising_fact", views=2000),
            ]
        )
        scored = TopicScorer(default_config).score(candidate, history)
        # exact topic match, normalized vs peak views -> 1.0
        assert scored.historical_fit == pytest.approx(1.0)
        base = _expected_base(candidate, default_config)
        weight = default_config.scoring.historical_weight
        assert scored.total_score == pytest.approx((1 - weight) * base + weight * 1.0)

    def test_no_matching_record_keeps_fit_null(self, default_config, candidate_factory):
        candidate = candidate_factory(topic_id="nobody")
        history = SelectionHistory.from_records(
            [HistoricalRecord(video_id="v1", topic_id="t1", pillar="technology", topic_type="technology_development", views=4000)]
        )
        scored = TopicScorer(default_config).score(candidate, history)
        assert scored.historical_fit is None
        assert scored.total_score == pytest.approx(_expected_base(candidate, default_config))

    def test_historical_disabled_ignores_history(self, default_config, candidate, config_factory):
        history = SelectionHistory.from_records(
            [HistoricalRecord(video_id="v1", topic_id="t1", pillar="science", topic_type="surprising_fact", views=4000)]
        )
        config = config_factory(historical_enabled=False, historical_weight=0.25)
        scored = TopicScorer(config).score(candidate, history)
        assert scored.historical_fit is None
        assert scored.total_score == pytest.approx(_expected_base(candidate, config))