"""TopicScorer: weights signals/content into a total_score.

Pure and replaceable: it depends on the competition map and the
candidate model, nothing else. Historical performance contributes ONLY
when a historical_fit exists (never during cold start).
"""

from __future__ import annotations

from agents.seo.competition import competition_to_opportunity
from agents.seo.config import TopicSelectionConfig
from agents.seo.history import SelectionHistory
from agents.seo.models import TopicCandidate


class TopicScorerError(ValueError):
    """Raised when scoring cannot be performed."""


class TopicScorer:
    def __init__(self, config: TopicSelectionConfig) -> None:
        self._config = config

    def score(self, candidate: TopicCandidate, history: SelectionHistory | None = None) -> TopicCandidate:
        """Fill candidate.historical_fit and .total_score in place."""
        if candidate.total_score is not None:
            return candidate  # already scored; avoid double-blending history

        base = self._weighted_base_score(candidate)

        historical_fit = None
        if (
            self._config.scoring.historical_enabled
            and history
            and not history.is_empty
        ):
            historical_fit = _compute_historical_fit(candidate, history)

        candidate.historical_fit = historical_fit
        candidate.total_score = _blend(base, historical_fit, historical_weight=self._config.scoring.historical_weight)
        return candidate.validate(self._config.pillars, self._config.topic_types)

    def _weighted_base_score(self, candidate: TopicCandidate) -> float:
        weights = self._config.scoring.weights
        total_weight = sum(weights.values())
        if total_weight <= 0:
            raise TopicScorerError("scoring weights sum to zero; nothing to score with")

        transformed = {
            "interest": candidate.signals.interest,
            "trend_momentum": candidate.signals.trend_momentum,
            "competition": competition_to_opportunity(
                candidate.signals.competition,
                transformation=self._config.scoring.competition_transformation,
            ),
            "curiosity": candidate.content_scores.curiosity,
            "explainer_potential": candidate.content_scores.explainer_potential,
            "visual_potential": candidate.content_scores.visual_potential,
            "audience_fit": candidate.content_scores.audience_fit,
        }
        return sum(weights[key] * transformed[key] for key in transformed) / total_weight

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._config.scoring.weights)


def _compute_historical_fit(candidate: TopicCandidate, history: SelectionHistory) -> float | None:
    """Provisional historical signal: normalized views for matching videos.

    Matches by topic_id first, then falls back to pillar. Returns None
    (caller keeps it null) when nothing matches. This is intentionally a
    simple proxy until real analytics exist (docs/ANALYTICS.md).
    """
    records = history.records
    matching = [record for record in records if record.topic_id == candidate.topic_id]
    if not matching:
        matching = [record for record in records if record.pillar == candidate.pillar]
    if not matching:
        return None
    peak = history.peak_views()
    if peak <= 0:
        return 0.0
    normalized_mean = sum(record.views / peak for record in matching) / len(matching)
    return max(0.0, min(1.0, normalized_mean))


def _blend(base: float, historical_fit: float | None, historical_weight: float) -> float:
    if historical_fit is None:
        return base
    return (1.0 - historical_weight) * base + historical_weight * historical_fit