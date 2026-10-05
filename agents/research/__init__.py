"""Stage 2: AI Research.

Reads the Stage 1 ``topic_selection.json`` artifact and writes
``research.json``. Gemini's own knowledge is the research basis: there is no
web retrieval, no source harvesting and no corroboration in this stage.

CLI: ``python -m agents.research.ai_research_cli --run-id <run_id>``
"""

from agents.research.ai_research import (
    generate_ai_research,
    load_topic_selection,
    write_ai_research,
)
from agents.research.ai_research_models import AIResearch

__all__ = [
    "AIResearch",
    "generate_ai_research",
    "load_topic_selection",
    "write_ai_research",
]