"""Topic discovery / scoring / selection (Stage 1).

Data flow (docs/TOPIC_SELECTION.md):

    candidate source -> TopicCandidate -> TopicScorer -> TopicSelector
                        -> TopicSelection -> (future) Research Agent

Everything here works against local/mocked data; no YouTube, trends,
LLM, or database dependencies.
"""

from agents.seo.config import TopicSelectionConfig, load_config
from agents.seo.history import (
    HistoricalRecord,
    HistoryProvider,
    MockHistoryProvider,
    SelectionHistory,
)
from agents.seo.models import (
    Provenance,
    SignalScores,
    ContentScores,
    TopicCandidate,
    TopicCandidateError,
)
from agents.seo.result import TopicSelection
from agents.seo.scorer import TopicScorer
from agents.seo.selector import TopicSelector

__all__ = [
    "TopicSelectionConfig",
    "load_config",
    "HistoricalRecord",
    "HistoryProvider",
    "MockHistoryProvider",
    "SelectionHistory",
    "Provenance",
    "SignalScores",
    "ContentScores",
    "TopicCandidate",
    "TopicCandidateError",
    "TopicSelection",
    "TopicScorer",
    "TopicSelector",
]