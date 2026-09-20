"""Stage-1 entrypoint: candidates -> scored -> selected -> TopicSelection.

Content-system side only (docs/ARCHITECTURE.md): produces the
TopicSelection artifact that a future research agent consumes. Uses only
local/mocked data and configuration; no network, no APIs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from agents.seo.config import TopicSelectionConfig, load_config
from agents.seo.history import HistoricalRecord, SelectionHistory
from agents.seo.models import TopicCandidate
from agents.seo.result import TopicSelection
from agents.seo.scorer import TopicScorer
from agents.seo.selector import TopicSelector

MOCK_CANDIDATES_PATH = Path("data") / "mock" / "topic_candidates.json"


def load_candidates(
    path: str | Path = MOCK_CANDIDATES_PATH,
    config: TopicSelectionConfig | None = None,
) -> list[TopicCandidate]:
    """Load and validate candidate dicts from a JSON file."""
    import json

    config = config or load_config()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"candidate file must contain a JSON list: {path}")
    return [TopicCandidate.from_dict(item, config.pillars, config.topic_types) for item in raw]


def load_history(path: str | Path) -> SelectionHistory:
    """Load a JSON list of HistoricalRecords (may be an empty list)."""
    import json

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"history file must contain a JSON list: {path}")
    return SelectionHistory.from_records(HistoricalRecord.from_dict(item) for item in raw)


def run_topic_selection(
    candidates: Iterable[TopicCandidate],
    history: SelectionHistory | None,
    config: TopicSelectionConfig,
    run_id: str = "",
    created_at: str | None = None,
    seed: int | None = None,
) -> TopicSelection:
    """Score every candidate, then select one (see also topic_selector CLI)."""
    scorer = TopicScorer(config)
    scored = [scorer.score(candidate, history) for candidate in candidates]

    import random

    rng = random.Random(seed) if seed is not None else None
    selector = TopicSelector(config, run_id=run_id, created_at=created_at, rng=rng)
    return selector.select(scored, history)