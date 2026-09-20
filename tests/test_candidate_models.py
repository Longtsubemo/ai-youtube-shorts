"""Candidate validation tests (requirements 1 and 12)."""

from __future__ import annotations

import json

import pytest

from agents.seo.models import ContentScores, SignalScores, TopicCandidate, TopicCandidateError, Provenance


class TestCandidateValidation:
    def test_valid_candidate_passes_validation_and_keeps_fields(self, default_config, candidate):
        candidate.validate(default_config.pillars, default_config.topic_types)
        assert candidate.topic_id
        assert candidate.title
        assert candidate.description
        assert candidate.pillar in default_config.pillars
        assert candidate.topic_type in default_config.topic_types
        assert candidate.historical_fit is None  # cold start default
        assert candidate.total_score is None

    def test_missing_description_is_rejected(self, default_config, candidate):
        data = candidate.to_dict()
        del data["description"]
        with pytest.raises(TopicCandidateError, match="description"):
            TopicCandidate.from_dict(data, default_config.pillars, default_config.topic_types)

    def test_missing_signals_are_rejected(self, default_config, candidate):
        data = candidate.to_dict()
        data["signals"].pop("interest")
        with pytest.raises(TopicCandidateError, match="interest"):
            TopicCandidate.from_dict(data, default_config.pillars, default_config.topic_types)

    def test_unknown_pillar_is_rejected(self, default_config, candidate):
        data = candidate.to_dict()
        data["pillar"] = "cooking"
        with pytest.raises(TopicCandidateError, match="pillar"):
            TopicCandidate.from_dict(data, default_config.pillars, default_config.topic_types)

    def test_unknown_topic_type_is_rejected(self, default_config, candidate):
        data = candidate.to_dict()
        data["topic_type"] = "clickbait"
        with pytest.raises(TopicCandidateError, match="topic_type"):
            TopicCandidate.from_dict(data, default_config.pillars, default_config.topic_types)

    def test_out_of_range_signal_is_rejected(self, default_config, candidate):
        data = candidate.to_dict()
        data["signals"]["competition"] = 1.7
        with pytest.raises(TopicCandidateError, match="range"):
            TopicCandidate.from_dict(data, default_config.pillars, default_config.topic_types)

    def test_negative_content_score_is_rejected(self, default_config, candidate):
        data = candidate.to_dict()
        data["content_scores"]["curiosity"] = -0.1
        with pytest.raises(TopicCandidateError, match="range"):
            TopicCandidate.from_dict(data, default_config.pillars, default_config.topic_types)

    def test_invalid_candidate_json_is_rejected_cleanly(self, default_config, candidate):
        data = candidate.to_dict()
        data["topic_id"] = ""
        with pytest.raises(TopicCandidateError, match="topic_id"):
            TopicCandidate.from_dict(data, default_config.pillars, default_config.topic_types)

    def test_round_trip_preserves_candidate(self, default_config, candidate):
        data = json.loads(json.dumps(candidate.to_dict()))
        rebuilt = TopicCandidate.from_dict(data, default_config.pillars, default_config.topic_types)
        assert rebuilt == candidate


class TestModelComponents:
    def test_signal_scores_round_trip(self):
        signals = SignalScores(interest=0.5, trend_momentum=0.4, competition=0.7, freshness=0.2)
        assert SignalScores.from_dict(signals.to_dict()) == signals

    def test_content_scores_round_trip(self):
        content = ContentScores(curiosity=0.9, explainer_potential=0.8, visual_potential=0.7, audience_fit=0.6)
        assert ContentScores.from_dict(content.to_dict()) == content

    def test_provenance_round_trip(self):
        provenance = Provenance(generated_by="x", source="y", created_at="2026-01-01", provider=None)
        assert Provenance.from_dict(provenance.to_dict()) == provenance