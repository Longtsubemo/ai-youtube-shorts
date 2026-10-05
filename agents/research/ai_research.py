"""Stage 2: AI Research.

Reads the normal Stage 1 ``topic_selection.json`` artifact, expands the
selected finance topic into research material for a later script stage, and
writes ``research.json``.

Gemini's own knowledge is the research basis. This module contains no
retrieval, no source harvesting, no corroboration and no factuality gate. The
only provider boundary is `AIResearchProvider`, so the Gemini implementation
stays replaceable.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from agents.research.ai_research_models import (
    AIResearch,
    AIResearchError,
    ResearchTopic,
    utc_now,
)
from agents.research.provider import AIResearchProvider
from agents.seo.result import TopicSelection

REQUIRED_TOPIC_FIELDS = ("topic_id", "title", "pillar", "angle", "hook_direction")


class ResearchTopicError(ValueError):
    """Raised when the Stage 1 artifact cannot supply a usable topic.

    This covers missing/blank required fields, a missing or mismatched
    ``selected_topic_id``, and absent selection data. It is an input-shape
    check, not a content-quality gate.
    """


def load_topic_selection(path: str | Path) -> TopicSelection:
    """Load the Stage 1 ``topic_selection.json`` artifact."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return TopicSelection.from_dict(raw)


def write_ai_research(art: AIResearch, out_path: str | Path) -> Path:
    """Write the Stage 2 ``research.json`` artifact."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(art.to_dict(), indent=2), encoding="utf-8")
    return out_path


def build_research_topic(selection: TopicSelection) -> ResearchTopic:
    """Extract the selected topic from a Stage 1 selection artifact.

    Requires ``topic_id``, ``title``, ``pillar``, ``angle`` and
    ``hook_direction`` on the selected topic, plus a consistent
    ``selected_topic_id``. Fails loudly rather than researching a blank topic.
    """
    selected = selection.selected_topic
    if not isinstance(selected, dict) or not selected:
        raise ResearchTopicError("topic_selection.json has no selected_topic")

    missing = [field for field in REQUIRED_TOPIC_FIELDS if not str(selected.get(field) or "").strip()]
    if missing:
        raise ResearchTopicError(
            "selected_topic missing required field(s): " + ", ".join(missing)
        )

    if not str(selection.selected_topic_id or "").strip():
        raise ResearchTopicError("topic_selection.json has no selected_topic_id")
    if str(selected["topic_id"]).strip() != str(selection.selected_topic_id).strip():
        raise ResearchTopicError(
            "selected_topic.topic_id does not match selected_topic_id: "
            f"{selected['topic_id']!r} != {selection.selected_topic_id!r}"
        )

    return ResearchTopic(
        topic_id=str(selected["topic_id"]).strip(),
        title=str(selected["title"]).strip(),
        pillar=str(selected["pillar"]).strip(),
        angle=str(selected["angle"]).strip(),
        hook_direction=str(selected["hook_direction"]).strip(),
        topic_type=str(selected.get("topic_type") or "").strip(),
        description=str(selected.get("description") or "").strip(),
    )


def run_ai_research(
    selection: TopicSelection,
    provider: AIResearchProvider,
    *,
    run_id: str = "",
    created_at: str | None = None,
) -> AIResearch:
    """Expand the selected topic into research material via `provider`.

    The stage owns artifact concerns (run id, timestamp, structural
    validation); the provider owns how the material is produced.
    """
    topic = build_research_topic(selection)
    research = provider.generate(topic)
    updates: dict = {"run_id": run_id or selection.run_id}
    if created_at is not None:
        updates["created_at"] = created_at
    # AIResearch is frozen; rebuild rather than mutate.
    return replace(research, **updates).validate()


def generate_ai_research(
    selection: TopicSelection,
    provider: AIResearchProvider | None = None,
    *,
    run_id: str = "",
    created_at: str | None = None,
) -> AIResearch:
    """Produce the research artifact for a selected topic.

    Defaults to the deterministic mock provider so the stage is runnable and
    testable without a Gemini key.
    """
    if provider is None:
        from agents.research.provider import MockAIResearchProvider

        provider = MockAIResearchProvider()
    return run_ai_research(
        selection, provider, run_id=run_id, created_at=created_at or utc_now()
    )


__all__ = [
    "AIResearch",
    "AIResearchError",
    "ResearchTopic",
    "ResearchTopicError",
    "build_research_topic",
    "generate_ai_research",
    "load_topic_selection",
    "run_ai_research",
    "write_ai_research",
]