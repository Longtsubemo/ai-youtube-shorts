"""Stage-1 Gemini candidate provider.

Implements the `TopicCandidateProvider` protocol from `agents/seo/provider.py`
using the provider foundation in `providers/gemini/client.py`. Candidates are
*proposed* by Gemini; scoring/ranking/replay stay deterministic in the stage,
and every candidate must still survive the existing `TopicCandidate.from_dict`
validation gate (pillars, topic_types, score ranges).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field, TypeAdapter

from agents.seo.config import TopicSelectionConfig
from agents.seo.models import Provenance, TopicCandidate, utc_now
from providers.gemini.client import GeminiClient
from providers.gemini.errors import GeminiSchemaValidationError

PROMPT_VERSION = "seo/topic_candidates-v1"
DEFAULT_PROMPT_PATH = Path("prompts") / "seo" / "topic_candidates-v1.txt"
GENERATED_BY = "gemini_topic_provider_v1"
DEFAULT_CANDIDATE_COUNT = 8

_SLUG_CLEANUP = re.compile(r"[^a-z0-9]+")


class _SignalScores(BaseModel):
    interest: float = Field(ge=0.0, le=1.0)
    trend_momentum: float = Field(ge=0.0, le=1.0)
    competition: float = Field(ge=0.0, le=1.0)
    freshness: float = Field(ge=0.0, le=1.0)


class _ContentScores(BaseModel):
    curiosity: float = Field(ge=0.0, le=1.0)
    explainer_potential: float = Field(ge=0.0, le=1.0)
    visual_potential: float = Field(ge=0.0, le=1.0)
    audience_fit: float = Field(ge=0.0, le=1.0)


class TopicCandidateSchema(BaseModel):
    """Response schema mirroring the Stage-1 TopicCandidate contract."""

    topic_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    pillar: str = Field(min_length=1)
    topic_type: str = Field(min_length=1)
    signals: _SignalScores
    content_scores: _ContentScores


class CandidateListSchema(BaseModel):
    candidates: list[TopicCandidateSchema] = Field(default_factory=list)


def _normalize_topic_id(raw: str) -> str:
    slug = _SLUG_CLEANUP.sub("_", raw.strip().lower()).strip("_")
    return f"t_{slug}" if not slug.startswith("t_") else slug


class GeminiTopicProvider:
    """Topic candidate provider backed by Gemini structured output."""

    provider_name = "gemini"

    def __init__(
        self,
        config: TopicSelectionConfig,
        client: GeminiClient,
        *,
        prompt_path: str | Path = DEFAULT_PROMPT_PATH,
        candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    ) -> None:
        self._config = config
        self._client = client
        self._prompt_path = Path(prompt_path)
        self._candidate_count = candidate_count

    # -- metadata --------------------------------------------------------- #

    @property
    def model(self) -> str:
        return self._client.model

    @property
    def prompt_version(self) -> str:
        return PROMPT_VERSION

    @property
    def produced_by(self) -> dict:
        """Header provenance for the TopicSelection envelope."""
        return {
            "prompt_version": PROMPT_VERSION,
            "provider": self.provider_name,
            "model": self.model,
        }

    # -- protocol --------------------------------------------------------- #

    def list_candidates(self) -> list[TopicCandidate]:
        prompt = self._render_prompt()
        schema = self._client.generate_structured(
            prompt,
            CandidateListSchema,
            system_prompt=self._system_prompt(),
        )
        try:
            return [self._to_candidate(item) for item in schema.candidates]
        except (TypeError, ValueError) as exc:
            raise GeminiSchemaValidationError(
                f"Gemini candidate failed Stage-1 validation: {exc}"
            ) from exc

    # -- internals --------------------------------------------------------- #

    def _system_prompt(self) -> str:
        return (
            "You are a stage-1 topic candidate generator. You only propose ideas. "
            "You must not claim research was performed, must not invent evidence, "
            "and must not fabricate URLs or sources. Return JSON only."
        )

    def _render_prompt(self) -> str:
        path = Path(self._prompt_path)
        if not path.is_file():
            raise FileNotFoundError(f"prompt template not found: {path}")
        return (
            path.read_text(encoding="utf-8")
            .replace("{candidate_count}", str(self._candidate_count))
            .replace("{pillars}", ", ".join(self._config.pillars))
            .replace("{topic_types}", ", ".join(self._config.topic_types))
        )

    def _to_candidate(self, item: TopicCandidateSchema) -> TopicCandidate:
        candidate = TopicCandidate.from_dict(
            {
                "topic_id": _normalize_topic_id(item.topic_id),
                "title": item.title,
                "description": item.description,
                "pillar": item.pillar,
                "topic_type": item.topic_type,
                "signals": item.signals.model_dump(),
                "content_scores": item.content_scores.model_dump(),
            },
            self._config.pillars,
            self._config.topic_types,
        )
        candidate.provenance = Provenance(
            generated_by=GENERATED_BY,
            source="gemini",
            created_at=utc_now(),
            provider=self.provider_name,
            model=self.model,
            prompt_version=PROMPT_VERSION,
        )
        return candidate


# Convenience re-export for schema validation in tests.
CandidateListAdapter = TypeAdapter(CandidateListSchema)


def iter_schemas(candidates: Iterable[dict]) -> list[TopicCandidateSchema]:
    """Schema-validate raw candidate dicts (test/CLI helper)."""
    return [TopicCandidateSchema.model_validate(item) for item in candidates]