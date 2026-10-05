"""AIResearchProvider abstraction for Stage 2.

Stage 2 expands the selected finance topic into research material. The
provider seam keeps the deterministic mock as the default so unit tests never
need a real GEMINI_API_KEY, while allowing Gemini (or anything else) to drop in
without the stage knowing.

Like Stage 1, only the requested provider is imported: a mock/default run
never imports the Gemini SDK. Requesting a real provider without a key raises
`GeminiConfigurationError` rather than silently falling back to mock.

There is intentionally no retrieval here. The mock is template-generated and
the Gemini path asks the model to use its own knowledge; neither browses,
harvests sources, or corroborates anything.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Protocol

from agents.research.ai_research_models import AIResearch, ResearchTopic
from agents.seo.config import TopicSelectionConfig


class AIResearchProviderError(ValueError):
    """Raised when a research provider cannot fulfill the request."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class AIResearchProvider(Protocol):
    """Anything that can turn a selected topic into research material.

    Implementations may wrap an LLM, a local template, or a file on disk.
    The stage never sees the concrete class.
    """

    provider_name: str

    def generate(self, topic: ResearchTopic) -> AIResearch: ...


def _pillar_info(config: TopicSelectionConfig | None, pillar: str) -> Any:
    """Best-effort editorial lookup; unknown pillars degrade to no context."""
    if config is None:
        return None
    try:
        return config.pillar_info(pillar)
    except Exception:
        return None


class MockAIResearchProvider:
    """Deterministic, network-free research provider.

    Material is templated from the topic's own title/pillar/angle plus the
    pillar catalog, so the same topic always yields byte-identical output.
    Nothing is retrieved and no statistic is asserted: the mock supplies
    explanatory scaffolding for a later stage to fill with real model output.
    """

    provider_name = "mock"

    def __init__(self, config: TopicSelectionConfig | None = None) -> None:
        self._config = config

    @property
    def produced_by(self) -> dict:
        return {"prompt_version": None, "provider": "mock", "model": None}

    def generate(self, topic: ResearchTopic) -> AIResearch:
        info = _pillar_info(self._config, topic.pillar)
        label = info.label if info else topic.pillar
        focus = info.focus if info else topic.pillar
        guidance = info.guidance if info else ""

        return AIResearch(
            topic_id=topic.topic_id,
            key_concepts=[
                f"{label}: what it covers and why it matters to the viewer's money.",
                f"The single mechanism behind '{topic.title}'.",
                "The one thing a non-expert gets wrong about this.",
            ],
            explanation=(
                f"Mock research for '{topic.title}'. "
                f"Pillar '{label}' focuses on {focus} "
                f"Mock material explains the mechanism in plain language for a "
                f"broad English-speaking audience; a real provider supplies the "
                f"actual substance here."
            ),
            examples=[
                f"A concrete everyday example of {topic.pillar} in action.",
                f"A contrast case showing what the opposite behaviour looks like.",
            ],
            # Deliberately empty: the mock must never fabricate a statistic,
            # and an empty `numbers` list is a valid research result.
            numbers=[],
            interesting_angles=[
                f"Angle from Stage 1: {topic.angle}",
                f"Hook direction from Stage 1: {topic.hook_direction}",
                f"Framing hint: {guidance}" if guidance else "Framing hint: detect the mechanism.",
            ],
            common_misconceptions=[
                "The widely repeated belief this topic quietly contradicts.",
                "A second misconception that makes the same topic feel intuitive.",
            ],
            story_ideas=[
                f"Open on the {topic.pillar} mystery and reveal the mechanism.",
                f"Visualize the contrast: the visible number versus what actually happened.",
            ],
            pillar=topic.pillar,
            produced_by=self.produced_by,
        ).validate()


def build_ai_research_provider(
    name: str | None,
    *,
    config: TopicSelectionConfig | None = None,
    env: Mapping[str, str] | None = None,
    client: object | None = None,
) -> AIResearchProvider:
    """Return the research provider selected by `name` (default: mock)."""
    if name in (None, "", "mock"):
        return MockAIResearchProvider(config)

    if name == "gemini":
        from providers.gemini.client import GeminiClient  # lazy import
        from providers.gemini.config import GeminiConfig
        from providers.gemini.research_provider import GeminiAIResearchProvider  # lazy import

        if client is None:
            gemini_config = GeminiConfig.from_env(env if env is not None else os.environ)
            client = GeminiClient(gemini_config)
        return GeminiAIResearchProvider(config=config, client=client)

    raise AIResearchProviderError(f"unknown AI research provider: {name!r}")