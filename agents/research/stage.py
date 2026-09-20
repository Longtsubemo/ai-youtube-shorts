"""Stage-2 entrypoint: TopicSelection -> ResearchPackage.

Content-system side only (docs/ARCHITECTURE.md): consumes a Stage-1
TopicSelection artifact and produces the versioned ResearchPackage that the
future fact-check stage consumes. Uses local/mocked providers exclusively -
no network, no APIs, no LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

from agents.research.agent import ResearchAgent
from agents.research.models import ResearchPackage
from agents.research.provider import MockResearchProvider, ResearchProvider, DEFAULT_RESEARCH_DATA_PATH
from agents.seo.result import TopicSelection

DEFAULT_SELECTION_PATH = Path("data") / "mock" / "topic_selection.json"


def load_topic_selection(path: str | Path = DEFAULT_SELECTION_PATH) -> TopicSelection:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return TopicSelection.from_dict(raw)


def run_research(
    selection: TopicSelection,
    provider: ResearchProvider | None = None,
    run_id: str = "",
    created_at: str | None = None,
    research_data_path: str | Path = DEFAULT_RESEARCH_DATA_PATH,
) -> ResearchPackage:
    """Run the research stage over a selection, returning a ResearchPackage."""
    provider = provider or MockResearchProvider(data_path=str(research_data_path))
    agent = ResearchAgent(provider, run_id=run_id, created_at=created_at)
    return agent.research(selection)