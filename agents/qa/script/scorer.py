"""Deterministic weighted quality score for the Stage-5 QA result.

The score is an INTERNAL QA QUALITY SCORE - it never predicts YouTube
performance (docs/DATA_CONTRACTS.md §8.1). It is fully deterministic: the same
script/issues produce the same score, and the dimension breakdown is recorded
in the artifact so the number can be audited.

Score model:

    dimension_score = clamp(100 - penalties, 0, 100)
        each ERROR issue  -45 (hard gate) or -20 (soft gate)
        each WARNING      -6
        each INFO         -1
    overall = round(weighted_average(dimension_scores, weights), 1)

Weights sum to 1 (configurable; DEFAULT_SCORE_WEIGHTS in models.py):
    factual_fidelity 0.20   traceability 0.15   uncertainty 0.15
    structure 0.10         pacing 0.10         clarity 0.10
    hook 0.05              coverage 0.05       repetition 0.05
    format 0.05
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agents.qa.script.checks import GATE_HARD
from agents.qa.script.models import (
    CATEGORY_DIMENSION,
    DEFAULT_SCORE_WEIGHTS,
    QA_DIMENSIONS,
)


@dataclass(frozen=True)
class ScoreResult:
    """Deterministic score summary, stored in the artifact."""

    score: float
    dimensions: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "dimensions": dict(self.dimensions),
            "weights": dict(self.weights),
        }


def _penalty(severity: str, gate: str) -> float:
    if severity == "ERROR":
        return 45.0 if gate == GATE_HARD else 20.0
    if severity == "WARNING":
        return 6.0
    if severity == "INFO":
        return 1.0
    return 0.0


def compute_score(issues, weights: dict[str, float] | None = None) -> ScoreResult:
    """Aggregate an ordered issue stream into a weighted dimension score.

    `weights` must cover every QA_DIMENSIONS key and sum to 1; when omitted,
    the documented DEFAULT_SCORE_WEIGHTS are used.
    """
    weights = dict(DEFAULT_SCORE_WEIGHTS) if weights is None else dict(weights)
    validate_weights(weights)

    penalties = {dimension: 0.0 for dimension in QA_DIMENSIONS}
    for issue in issues:
        dimension = issue.dimension or CATEGORY_DIMENSION.get(issue.category, "clarity")
        penalties[dimension] += _penalty(issue.severity, issue.gate)

    dimensions = {}
    for dimension in QA_DIMENSIONS:
        raw = 100.0 - penalties[dimension]
        dimensions[dimension] = round(max(0.0, min(100.0, raw)), 1)

    weighted_sum = sum(weights.get(dimension, 0.0) * dimensions[dimension] for dimension in QA_DIMENSIONS)
    return ScoreResult(score=round(weighted_sum, 1), dimensions=dimensions, weights=weights)


def validate_weights(weights: dict[str, float]) -> None:
    """Assert configured weights cover exactly QA_DIMENSIONS and sum to 1."""
    if set(weights) != set(QA_DIMENSIONS):
        raise ValueError(f"weights must cover exactly QA_DIMENSIONS: {QA_DIMENSIONS}")
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError("weights must sum to 1.0")


__all__ = ["DEFAULT_SCORE_WEIGHTS", "ScoreResult", "compute_score", "validate_weights"]