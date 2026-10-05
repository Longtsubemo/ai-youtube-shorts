"""ScriptProvider abstraction and its deterministic mock implementation.

Stage 3 shapes Stage 2 research into a finished spoken script. The provider
seam keeps the deterministic mock as the default so unit tests never need a
real GEMINI_API_KEY, while letting Gemini (or anything else) drop in without
the stage knowing.

Like Stage 1 and Stage 2, only the requested provider is imported, so a mock
run never imports the Gemini SDK. Requesting a real provider without a key
raises `GeminiConfigurationError` rather than silently falling back to mock.

Nothing here browses, cites, or validates the script: the mock is templated
from the supplied topic and research, and the Gemini path asks the model to
answer from its own knowledge.
"""

from __future__ import annotations

import os
from typing import Mapping, Protocol

from agents.research.ai_research_models import AIResearch
from agents.script.script_models import (
    EDUCATIONAL_NOTICE,
    MASCOT_NOTE,
    MASCOT_SPEAKS,
    TARGET_DURATION_SECONDS,
    Script,
    ScriptSection,
)
from agents.seo.config import TopicSelectionConfig


class ScriptProviderError(ValueError):
    """Raised when a script provider cannot fulfill the request."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ScriptProvider(Protocol):
    """Anything that can turn a topic + research into a spoken script.

    `context` is the dict produced by
    `agents.script.stage.build_script_context`: the Stage 1 topic fields plus
    the Stage 2 `AIResearch` under the key "research".
    """

    provider_name: str

    def generate(self, context: dict) -> Script: ...


def _first(items, fallback: str = "") -> str:
    """First non-empty string in `items`, else `fallback`."""
    for item in items or ():
        text = str(item).strip()
        if text:
            return text
    return fallback


class MockScriptProvider:
    """Deterministic, network-free script provider.

    Assembles an educational narration from the supplied topic and Stage 2
    research so tests and offline runs get a valid, non-empty, structured
    artifact without a Gemini key.

    The mock deliberately states no market view and issues no directive: it is
    general educational framing only, matching the channel's safety posture.
    """

    provider_name = "mock"

    def __init__(self, config: TopicSelectionConfig | None = None) -> None:
        self._config = config

    @property
    def produced_by(self) -> dict:
        return {"prompt_version": None, "provider": "mock", "model": None}

    def _channel_id(self) -> str:
        if self._config is not None:
            return self._config.channel.channel_id
        return "finance_mascot"

    def generate(self, context: dict) -> Script:
        research: AIResearch = context["research"]
        pillar = str(context.get("pillar") or "")
        title = str(context["title"])

        hook = str(context.get("hook_direction") or "").strip()
        example = _first(research.examples)
        misconception = _first(research.common_misconceptions)
        story = _first(research.story_ideas)

        sections = [
            ScriptSection(
                section_id="s1",
                role="hook",
                narration=hook or "Here's something about money almost nobody explains.",
                estimated_seconds=10,
                visual_hint="The detective fox opens a case file on a desk.",
            ),
            ScriptSection(
                section_id="s2",
                role="setup",
                narration=f"{research.explanation}",
                estimated_seconds=20,
                visual_hint="The core mechanism appears on screen as a simple diagram.",
            ),
            ScriptSection(
                section_id="s3",
                role="example",
                narration=(
                    f"{example} {misconception}".strip()
                    if (example or misconception)
                    else "A simple everyday example makes the idea concrete."
                ),
                estimated_seconds=18,
                visual_hint="A before-and-after comparison of two balances.",
            ),
            ScriptSection(
                section_id="s4",
                role="payoff",
                narration=(
                    f"{story} That's the real story behind {title}."
                    if story
                    else f"That's the real story behind {title}."
                ),
                estimated_seconds=18,
                visual_hint="The reveal resolves onto the single key takeaway.",
            ),
        ]

        narration = " ".join(section.narration.strip() for section in sections if section.narration.strip())

        return Script(
            topic_id=str(context["topic_id"]),
            title=str(context["title"]),
            duration_target_seconds=TARGET_DURATION_SECONDS,
            narration=narration,
            sections=sections,
            pillar=pillar,
            channel_id=self._channel_id(),
            produced_by=self.produced_by,
            educational_notice=EDUCATIONAL_NOTICE,
            mascot_note=MASCOT_NOTE,
            mascot_speaks=MASCOT_SPEAKS,
        ).validate()


def build_script_provider(
    name: str | None,
    *,
    config: TopicSelectionConfig | None = None,
    env: Mapping[str, str] | None = None,
    client: object | None = None,
) -> ScriptProvider:
    """Return the script provider selected by `name` (default: mock)."""
    if name in (None, "", "mock"):
        return MockScriptProvider(config)

    if name == "gemini":
        from providers.gemini.client import GeminiClient  # lazy import
        from providers.gemini.config import GeminiConfig
        from providers.gemini.script_provider import GeminiScriptProvider  # lazy import

        if client is None:
            gemini_config = GeminiConfig.from_env(env if env is not None else os.environ)
            client = GeminiClient(gemini_config)
        return GeminiScriptProvider(config=config, client=client)

    raise ScriptProviderError(f"unknown script provider: {name!r}")