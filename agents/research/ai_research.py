"""Stage 2: AI Research (artifact IO + generation entry point)."""

from __future__ import annotations

import json
from pathlib import Path

from agents.research.ai_research_models import AIResearch
from agents.seo.result import TopicSelection


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


def generate_ai_research(topic: TopicSelection) -> AIResearch:
    """Produce the AI-research artifact for a selected topic.

    NOT IMPLEMENTED: no Gemini call is made yet. The research basis for this
    architecture is Gemini's own knowledge, but that provider does not exist.
    Only ``topic_id`` is populated for now.
    """
    return AIResearch(topic_id=topic.selected_topic_id)