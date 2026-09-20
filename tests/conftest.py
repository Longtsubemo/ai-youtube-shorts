"""Shared fixtures for the Stage-1 topic-selection test suite."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agents.seo.config import load_config
from agents.seo.history import HistoricalRecord, MockHistoryProvider, SelectionHistory
from agents.seo.models import ContentScores, Provenance, SignalScores, TopicCandidate

ROOT = Path(__file__).resolve().parent.parent
MOCK_DATA = ROOT / "data" / "mock"


def _default_config_dict() -> dict:
    return {
        "version": 1,
        "scoring": {
            "weights": {
                "interest": 0.20,
                "trend_momentum": 0.15,
                "competition": 0.15,
                "curiosity": 0.15,
                "explainer_potential": 0.15,
                "visual_potential": 0.10,
                "audience_fit": 0.10,
            },
            "competition": {"interpretation": "opportunity", "transformation": "inverse"},
            "historical": {"enabled": True, "weight": 0.25},
        },
        "selection": {
            "exploration_fraction": 0.3,
            "seed": 42,
            "top_n": 3,
            "coverage_penalty": 0.25,
        },
        "pillars": ["ai", "technology", "science", "space", "curiosity"],
        "topic_types": [
            "evergreen",
            "trending",
            "recent_discovery",
            "historical",
            "surprising_fact",
            "technology_development",
            "scientific_concept",
            "unusual_phenomenon",
        ],
    }


@pytest.fixture(scope="session")
def default_config():
    return load_config(ROOT / "config" / "topic_selection.yaml")


@pytest.fixture
def config_factory(tmp_path):
    """Build a config on disk from overrides (so tests prove the values
    flow through config -> scorer/selector, not hard-coded Python)."""

    def build(
        *,
        weights=None,
        exploration_fraction=None,
        seed=None,
        top_n=None,
        coverage_penalty=None,
        historical_enabled=None,
        historical_weight=None,
    ) -> "TopicSelectionConfig":
        data = _default_config_dict()
        if weights is not None:
            data["scoring"]["weights"] = weights
        if exploration_fraction is not None:
            data["selection"]["exploration_fraction"] = exploration_fraction
        if seed is not None:
            data["selection"]["seed"] = seed
        if top_n is not None:
            data["selection"]["top_n"] = top_n
        if coverage_penalty is not None:
            data["selection"]["coverage_penalty"] = coverage_penalty
        if historical_enabled is not None:
            data["scoring"]["historical"]["enabled"] = historical_enabled
        if historical_weight is not None:
            data["scoring"]["historical"]["weight"] = historical_weight
        path = tmp_path / "topic_selection_test.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return load_config(path)

    return build


def _make_candidate(
    topic_id="t1",
    title="Sample topic",
    pillar="science",
    topic_type="surprising_fact",
    signals=None,
    content=None,
) -> TopicCandidate:
    signals = signals or {"interest": 0.7, "trend_momentum": 0.4, "competition": 0.5, "freshness": 0.3}
    content = content or {"curiosity": 0.8, "explainer_potential": 0.8, "visual_potential": 0.7, "audience_fit": 0.8}
    return TopicCandidate(
        topic_id=topic_id,
        title=title,
        description="Test description for a locally mocked candidate.",
        pillar=pillar,
        topic_type=topic_type,
        signals=SignalScores(**signals),
        content_scores=ContentScores(**content),
        provenance=Provenance(
            generated_by="test_fixture",
            source="unit_test",
            created_at="2026-09-20T00:00:00+00:00",
        ),
    )


@pytest.fixture
def candidate():
    return _make_candidate()


@pytest.fixture
def candidate_factory():
    return _make_candidate


@pytest.fixture
def mock_candidates(default_config):
    from agents.seo.stage import load_candidates

    return load_candidates(MOCK_DATA / "topic_candidates.json", default_config)


@pytest.fixture
def sample_history():
    return SelectionHistory.from_records(
        [
            HistoricalRecord(
                video_id="vid_mock_0001",
                topic_id="t_octopus_hearts",
                pillar="science",
                topic_type="surprising_fact",
                views=5200,
                retention=0.58,
                likes=340,
                comments=41,
                shares=22,
                subscribers_gained=12,
                published_at="2026-08-18T20:00:00+00:00",
            ),
            HistoricalRecord(
                video_id="vid_mock_0002",
                topic_id="t_black_hole",
                pillar="space",
                topic_type="scientific_concept",
                views=3100,
                retention=0.51,
                likes=210,
                comments=30,
                shares=15,
                subscribers_gained=8,
                published_at="2026-08-25T20:00:00+00:00",
            ),
        ]
    )


@pytest.fixture
def history_provider(sample_history):
    return MockHistoryProvider(sample_history.records)