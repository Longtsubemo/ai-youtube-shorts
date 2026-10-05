"""Stage-2 Gemini research provider.

Expands the selected finance topic into structured research material using the
existing Gemini client/config infrastructure - there is no second client. The
model's own knowledge is the research basis: this provider calls
``generate_structured`` (NOT ``generate_research``), so no Google Search
grounding and no URL context are enabled and nothing is retrieved.

Every field of the response schema is optional with an empty default. That is
deliberate: the prompt instructs Gemini not to fabricate precise figures, so an
empty ``numbers`` list must validate successfully rather than forcing an
invented statistic to satisfy the schema.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from agents.research.ai_research_models import AIResearch, ResearchTopic
from agents.seo.config import TopicSelectionConfig
from providers.gemini.client import GeminiClient
from providers.gemini.errors import GeminiSchemaValidationError

PROMPT_VERSION = "research/topic_research-v1"
DEFAULT_PROMPT_PATH = Path("prompts") / "research" / "topic_research-v1.txt"
GENERATED_BY = "gemini_research_provider_v1"

_TITLE_SLUG_CLEANUP = re.compile(r"[^a-z0-9]+")


class AIResearchSchema(BaseModel):
    """Structured research response for one selected topic.

    Mirrors the Stage 2 `AIResearch` content contract. Every field is optional
    so the model can return an empty `numbers` list instead of inventing a
    precise figure.
    """

    key_concepts: list[str] = Field(default_factory=list)
    explanation: str = ""
    examples: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)
    interesting_angles: list[str] = Field(default_factory=list)
    common_misconceptions: list[str] = Field(default_factory=list)
    story_ideas: list[str] = Field(default_factory=list)


class GeminiAIResearchProvider:
    """Research provider backed by Gemini structured output."""

    provider_name = "gemini"

    def __init__(
        self,
        config: TopicSelectionConfig | None,
        client: GeminiClient,
        *,
        prompt_path: str | Path = DEFAULT_PROMPT_PATH,
    ) -> None:
        self._config = config
        self._client = client
        self._prompt_path = Path(prompt_path)

    # -- metadata --------------------------------------------------------- #

    @property
    def model(self) -> str:
        return self._client.model

    @property
    def prompt_version(self) -> str:
        return PROMPT_VERSION

    @property
    def produced_by(self) -> dict:
        """Header provenance for the research artifact."""
        return {
            "prompt_version": PROMPT_VERSION,
            "provider": self.provider_name,
            "model": self.model,
        }

    # -- protocol --------------------------------------------------------- #

    def generate(self, topic: ResearchTopic) -> AIResearch:
        schema = self._client.generate_structured(
            self._render_prompt(topic),
            AIResearchSchema,
            system_prompt=self._system_prompt(),
        )
        try:
            research = AIResearch(
                topic_id=topic.topic_id,
                key_concepts=list(schema.key_concepts),
                explanation=schema.explanation,
                examples=list(schema.examples),
                numbers=list(schema.numbers),
                interesting_angles=list(schema.interesting_angles),
                common_misconceptions=list(schema.common_misconceptions),
                story_ideas=list(schema.story_ideas),
                pillar=topic.pillar,
                produced_by=self.produced_by,
            )
            return research.validate()
        except (TypeError, ValueError) as exc:
            raise GeminiSchemaValidationError(
                f"Gemini research failed Stage-2 validation: {exc}"
            ) from exc

    # -- internals --------------------------------------------------------- #

    def _system_prompt(self) -> str:
        channel = self._config.channel if self._config else None
        promise = channel.channel_promise if channel else "Understand money. Grow wealth."
        mascot_role = channel.mascot.role if channel else "financial_detective"
        return (
            "You expand one selected finance topic into research material for a "
            "short-form video script. You answer from your own knowledge only: you "
            "never browse, never search, never cite sources, and never invent "
            f"studies, institutions, quotes, or precise statistics. The channel promise is: "
            f"{promise} The on-screen character is a {mascot_role} that never speaks, so "
            "flag moments that can be shown on screen. Return JSON only."
        )

    def _render_prompt(self, topic: ResearchTopic) -> str:
        path = Path(self._prompt_path)
        if not path.is_file():
            raise FileNotFoundError(f"prompt template not found: {path}")

        channel = self._config.channel if self._config else None
        pillar_info = None
        if self._config is not None:
            try:
                pillar_info = self._config.pillar_info(topic.pillar)
            except Exception:
                pillar_info = None

        replacements = {
            "{title}": topic.title,
            "{pillar}": topic.pillar,
            "{pillar_label}": pillar_info.label if pillar_info else topic.pillar,
            "{pillar_focus}": pillar_info.focus if pillar_info else topic.pillar,
            "{pillar_guidance}": pillar_info.guidance if pillar_info else "",
            "{angle}": topic.angle or "(none supplied)",
            "{hook_direction}": topic.hook_direction or "(none supplied)",
            "{topic_type}": topic.topic_type or "(none supplied)",
            "{topic_description}": topic.description or "(none supplied)",
            "{channel_id}": channel.channel_id if channel else "finance_mascot",
            "{channel_promise}": channel.channel_promise if channel else "Understand money. Grow wealth.",
            "{audience}": channel.audience if channel else "",
            "{tone}": channel.tone if channel else "",
            "{target_duration_seconds}": str(channel.target_duration_seconds) if channel else "70",
            "{mascot_role}": channel.mascot.role if channel else "financial_detective",
        }

        rendered = path.read_text(encoding="utf-8")
        for token, value in replacements.items():
            rendered = rendered.replace(token, value)
        return rendered


def slugify_topic_id(title: str) -> str:
    """Stable fallback id derived from a title (test/CLI helper)."""
    slug = _TITLE_SLUG_CLEANUP.sub("_", title.strip().lower()).strip("_")
    return f"t_{slug}" if not slug.startswith("t_") else slug