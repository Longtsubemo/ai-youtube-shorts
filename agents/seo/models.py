"""Versioned TopicCandidate model and related value blocks.

Bundles objective/external signals, content-evaluation scores, and
historical performance separately. `historical_fit` is `null` at
cold start and stays `null` for any candidate with no matching history -
it is never silently zeroed. `total_score` is filled by the TopicScorer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# Every artifact carries these provenance fields (see docs/DATA_CONTRACTS.md).
PROVENANCE_FIELDS = ("generated_by", "source", "created_at", "provider", "model", "prompt_version")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TopicCandidateError(ValueError):
    """Raised when a candidate is missing fields, is malformed, or fails
    validation. Carries a human-readable message so stage-1 failures are
    diagnosable."""


def _require(value: Any, name: str) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise TopicCandidateError(f"missing required field: {name}")
    return value


def _check_score(value: Any, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise TopicCandidateError(f"{name} must be a number, got {value!r}") from None
    if not 0.0 <= value <= 1.0:
        raise TopicCandidateError(f"{name} out of range [0,1]: {value}")
    return value


@dataclass(frozen=True)
class Provenance:
    """Where/how an artifact was generated. Replaceable generator info."""

    generated_by: str
    source: str
    created_at: str
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None

    def to_dict(self) -> dict:
        return {
            "generated_by": self.generated_by,
            "source": self.source,
            "created_at": self.created_at,
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Provenance":
        return cls(**{k: data.get(k) for k in PROVENANCE_FIELDS})


@dataclass(frozen=True)
class SignalScores:
    """Objective/external signals (0..1)."""

    interest: float
    trend_momentum: float
    competition: float
    freshness: float

    def to_dict(self) -> dict:
        return {
            "interest": self.interest,
            "trend_momentum": self.trend_momentum,
            "competition": self.competition,
            "freshness": self.freshness,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SignalScores":
        for key in ("interest", "trend_momentum", "competition", "freshness"):
            if key not in data:
                raise TopicCandidateError(f"signals missing field: {key}")
        return cls(
            interest=_check_score(data["interest"], "signals.interest"),
            trend_momentum=_check_score(data["trend_momentum"], "signals.trend_momentum"),
            competition=_check_score(data["competition"], "signals.competition"),
            freshness=_check_score(data["freshness"], "signals.freshness"),
        )


@dataclass(frozen=True)
class ContentScores:
    """AI/content-evaluation scores (0..1)."""

    curiosity: float
    explainer_potential: float
    visual_potential: float
    audience_fit: float

    def to_dict(self) -> dict:
        return {
            "curiosity": self.curiosity,
            "explainer_potential": self.explainer_potential,
            "visual_potential": self.visual_potential,
            "audience_fit": self.audience_fit,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContentScores":
        for key in ("curiosity", "explainer_potential", "visual_potential", "audience_fit"):
            if key not in data:
                raise TopicCandidateError(f"content_scores missing field: {key}")
        return cls(
            curiosity=_check_score(data["curiosity"], "content_scores.curiosity"),
            explainer_potential=_check_score(data["explainer_potential"], "content_scores.explainer_potential"),
            visual_potential=_check_score(data["visual_potential"], "content_scores.visual_potential"),
            audience_fit=_check_score(data["audience_fit"], "content_scores.audience_fit"),
        )


@dataclass
class TopicCandidate:
    """A single candidate topic ready to be scored and ranked.

    `historical_fit`: null when no historical data exists for this topic
    (cold start) or when the historical signal is disabled.
    `total_score`: null until scored.
    """

    topic_id: str
    title: str
    description: str
    pillar: str
    topic_type: str
    signals: SignalScores
    content_scores: ContentScores
    historical_fit: float | None = None
    total_score: float | None = None
    provenance: Provenance | None = None

    def validate(self, pillars: tuple[str, ...], topic_types: tuple[str, ...]) -> "TopicCandidate":
        _require(self.topic_id, "topic_id")
        _require(self.title, "title")
        _require(self.description, "description")
        _require(self.pillar, "pillar")
        _require(self.topic_type, "topic_type")
        if self.pillar not in pillars:
            allowed = ", ".join(pillars)
            raise TopicCandidateError(f"unknown pillar '{self.pillar}' (allowed: {allowed})")
        if self.topic_type not in topic_types:
            allowed = ", ".join(topic_types)
            raise TopicCandidateError(f"unknown topic_type '{self.topic_type}' (allowed: {allowed})")
        if self.historical_fit is not None:
            self.historical_fit = _check_score(self.historical_fit, "historical_fit")
        if self.total_score is not None:
            self.total_score = _check_score(self.total_score, "total_score")
        return self

    def to_dict(self) -> dict:
        return {
            "topic_id": self.topic_id,
            "title": self.title,
            "description": self.description,
            "pillar": self.pillar,
            "topic_type": self.topic_type,
            "signals": self.signals.to_dict(),
            "content_scores": self.content_scores.to_dict(),
            "historical_fit": self.historical_fit,
            "total_score": self.total_score,
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }

    @classmethod
    def from_dict(cls, data: dict, pillars: tuple[str, ...], topic_types: tuple[str, ...]) -> "TopicCandidate":
        candidate = cls(
            topic_id=_require(data.get("topic_id"), "topic_id"),
            title=_require(data.get("title"), "title"),
            description=_require(data.get("description"), "description"),
            pillar=_require(data.get("pillar"), "pillar"),
            topic_type=_require(data.get("topic_type"), "topic_type"),
            signals=SignalScores.from_dict(data.get("signals") or {}),
            content_scores=ContentScores.from_dict(data.get("content_scores") or {}),
            historical_fit=data.get("historical_fit"),
            total_score=data.get("total_score"),
            provenance=Provenance.from_dict(data["provenance"]) if data.get("provenance") else None,
        )
        return candidate.validate(pillars, topic_types)