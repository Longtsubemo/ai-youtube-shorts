"""Selector tests (requirements 3, 4, 5, 6, 7, 8)."""

from __future__ import annotations

import pytest

from agents.seo.history import SelectionHistory
from agents.seo.scorer import TopicScorer
from agents.seo.selector import TopicSelector, TopicSelectorError


def _scored(candidates, config, history=None):
    scorer = TopicScorer(config)
    return [scorer.score(c, history) for c in candidates]


def _top_candidate(candidates, config, history=None):
    return max(_scored(candidates, config, history), key=lambda c: c.total_score or 0.0)


class TestSelectionBehavior:
    def test_empty_history_runs_and_reports_unavailable(self, default_config, mock_candidates):
        scorer = TopicScorer(default_config)
        scored = [scorer.score(c, SelectionHistory()) for c in mock_candidates]
        selection = TopicSelector(default_config).select(scored, SelectionHistory())
        assert selection.historical_data_available is False
        assert selection.selection_mode == "cold_start"

    def test_cold_start_keeps_historical_fit_null_in_artifact(self, default_config, mock_candidates):
        scorer = TopicScorer(default_config)
        scored = [scorer.score(c, SelectionHistory()) for c in mock_candidates]
        selection = TopicSelector(default_config).select(scored, SelectionHistory())
        assert all(candidate["historical_fit"] is None for candidate in selection.ranked_candidates)

    def test_history_accepted_sets_historical_data_flag(self, default_config, mock_candidates, sample_history):
        scored = _scored(mock_candidates, default_config, sample_history)
        selector = TopicSelector(default_config)
        selection = selector.select(scored, sample_history)
        assert selection.historical_data_available is True
        assert selection.selection_mode in ("cold_start", "exploration", "exploitation")

    def test_history_provider_interface_used(self, default_config, mock_candidates, history_provider):
        history = history_provider.get_history()
        scored = _scored(mock_candidates, default_config, history)
        selection = TopicSelector(default_config).select(scored, history)
        assert selection.historical_data_available is True

    def test_selection_returns_valid_topic_selection(self, default_config, mock_candidates):
        scored = _scored(mock_candidates, default_config, SelectionHistory())
        selection = TopicSelector(default_config).select(scored, SelectionHistory())
        data = selection.to_dict()["data"]
        for key in (
            "selection_id",
            "selected_topic_id",
            "selected_topic",
            "ranked_candidates",
            "selection_mode",
            "historical_data_available",
            "selection_reason",
            "scoring_configuration",
        ):
            assert key in data, f"missing artifact field: {key}"
        assert data["selected_topic"]["topic_id"] == data["selected_topic_id"]
        assert data["selected_topic_id"] in {c["topic_id"] for c in data["ranked_candidates"]}

    def test_ranking_sorted_desc(self, default_config, mock_candidates):
        scored = _scored(mock_candidates, default_config, SelectionHistory())
        selection = TopicSelector(default_config).select(scored, SelectionHistory())
        scores = [c["total_score"] for c in selection.ranked_candidates]
        assert scores == sorted(scores, reverse=True)

    def test_unscored_candidates_rejected_cleanly(self, default_config, mock_candidates):
        with pytest.raises(TopicSelectorError, match="not scored"):
            TopicSelector(default_config).select(mock_candidates)

    def test_no_candidates_raises(self, default_config):
        with pytest.raises(TopicSelectorError, match="no candidates"):
            TopicSelector(default_config).select([], SelectionHistory())


class TestModeRecording:
    def test_mode_cold_start_when_history_empty(self, default_config, mock_candidates):
        scored = _scored(mock_candidates, default_config, SelectionHistory())
        selection = TopicSelector(default_config).select(scored, SelectionHistory())
        assert selection.selection_mode == "cold_start"

    def test_mode_exploitation_when_history_and_zero_exploration(self, config_factory, mock_candidates, sample_history):
        config = config_factory(exploration_fraction=0.0, seed=42)
        scored = _scored(mock_candidates, config, sample_history)
        selection = TopicSelector(config).select(scored, sample_history)
        assert selection.selection_mode == "exploitation"
        top = max(scored, key=lambda c: c.total_score or 0.0)
        assert selection.selected_topic_id == top.topic_id

    def test_mode_exploration_when_history_and_full_exploration(self, config_factory, mock_candidates, sample_history):
        config = config_factory(exploration_fraction=1.0, seed=42)
        scored = _scored(mock_candidates, config, sample_history)
        selection = TopicSelector(config).select(scored, sample_history)
        assert selection.selection_mode == "exploration"

    def test_exploitation_picks_top_ranked(self, config_factory, mock_candidates, sample_history):
        config = config_factory(exploration_fraction=0.0, seed=42)
        scored = _scored(mock_candidates, config, sample_history)
        selection = TopicSelector(config).select(scored, sample_history)
        assert selection.ranked_candidates[0]["topic_id"] == selection.selected_topic_id

    def test_cold_start_reason_mentions_no_historical_data(self, default_config, mock_candidates):
        scored = _scored(mock_candidates, default_config, SelectionHistory())
        selection = TopicSelector(default_config).select(scored, SelectionHistory())
        assert "no historical data" in selection.selection_reason


class TestExplorationDiversity:
    def test_exploration_does_not_always_pick_highest_pillar(self, config_factory, mock_candidates):
        """Across seeds, cold-start exploration must not produce only the
        single highest-scoring candidate/pillar (`top_n` pool has several
        eligible pillars)."""
        empty = SelectionHistory()
        chosen = []
        for seed in range(20):
            config = config_factory(seed=seed)
            scored = _scored(mock_candidates, config, empty)
            selection = TopicSelector(config).select(scored, empty)
            chosen.append(selection)
            assert selection.selection_mode == "cold_start"

        selected_pillars = {selection.selected_topic["pillar"] for selection in chosen}
        selected_ids = {selection.selected_topic_id for selection in chosen}

        # multiple eligible pillars -> selection varies
        assert len(selected_pillars) > 1
        assert len(selected_ids) > 1

        top_id = _top_candidate(mock_candidates, config_factory(), empty).topic_id
        # exploration must NOT always pick the global highest-scoring candidate
        assert any(c.selected_topic_id != top_id for c in chosen)