"""Stage 2: AI Research.

Reads the Stage 1 ``topic_selection.json`` artifact and writes
``research.json``. Gemini's own knowledge is the research basis: there is no
web retrieval, no source harvesting, no corroboration, and no factuality gate
in this stage.

CLI: ``python -m agents.research.ai_research_cli --run-id <run_id>``
"""

from agents.research.ai_research import (
    ResearchTopicError,
    build_research_topic,
    generate_ai_research,
    load_topic_selection,
    run_ai_research,
    write_ai_research,
)
from agents.research.ai_research_models import (
    AIResearch,
    AIResearchError,
    ResearchTopic,
)
from agents.research.provider import (
    AIResearchProvider,
    AIResearchProviderError,
    MockAIResearchProvider,
    build_ai_research_provider,
)

__all__ = [
    "AIResearch",
    "AIResearchError",
    "AIResearchProvider",
    "AIResearchProviderError",
    "MockAIResearchProvider",
    "ResearchTopic",
    "ResearchTopicError",
    "build_ai_research_provider",
    "build_research_topic",
    "generate_ai_research",
    "load_topic_selection",
    "run_ai_research",
    "write_ai_research",
]