"""TopicSelector: deterministic, configurable exploration/exploitation.

Conceptual interface (docs/TOPIC_SELECTION.md):

    selection = TopicSelector(config).select(candidates, history)

Configuration is injected at construction; the call matches the planned
`select(candidates, history)` boundary so a bandit/ML optimizer can be
dropped in later behind the same signature.

Modes:
    cold_start   - no historical data; always explores across pillars.
    exploration  - historical data exists AND the seeded roll landed in
                   the exploration fraction; explores with coverage bias.
    exploitation - historical data exists and the roll did not; picks the
                   highest total_score candidate.

Candidates MUST be scored (by TopicScorer) before calling select.
"""

from __future__ import annotations

import random
from typing import Sequence

from agents.seo.config import TopicSelectionConfig
from agents.seo.history import SelectionHistory
from agents.seo.models import TopicCandidate
from agents.seo.result import TopicSelection

COLD_START = "cold_start"
EXPLORATION = "exploration"
EXPLOITATION = "exploitation"


class TopicSelectorError(ValueError):
    """Raised when selection cannot be performed."""


class TopicSelector:
    def __init__(
        self,
        config: TopicSelectionConfig,
        run_id: str = "",
        created_at: str | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._config = config
        self._run_id = run_id
        self._created_at = created_at
        self._rng = rng or random.Random(config.selection.seed)

    def select(
        self,
        candidates: Sequence[TopicCandidate],
        history: SelectionHistory | None = None,
    ) -> TopicSelection:
        history = history or SelectionHistory()
        self._ensure_scored(candidates)

        ranked = sorted(candidates, key=lambda c: c.total_score or 0.0, reverse=True)
        if not ranked:
            raise TopicSelectorError("select() called with no candidates")

        historical_available = not history.is_empty
        mode, mode_reason = self._determine_mode(historical_available, history)

        if mode == EXPLOITATION:
            selected = ranked[0]
        else:
            selected, explore_reason = self._explore(ranked, history)
            mode_reason = f"{mode_reason} {explore_reason}"

        reason = (
            f"{mode}: {mode_reason} | "
            f"selected '{selected.title}' (pillar={selected.pillar}, "
            f"topic_type={selected.topic_type}, total_score={selected.total_score:.4f}) | "
            f"historical_data_available={historical_available}"
        )

        return TopicSelection.build(
            run_id=self._run_id,
            selected=selected,
            ranked_candidates=ranked,
            selection_mode=mode,
            historical_data_available=historical_available,
            selection_reason=reason,
            scoring_configuration=self._config.to_dict(),
            created_at=self._created_at,
        )

    def _ensure_scored(self, candidates: Sequence[TopicCandidate]) -> None:
        unscored = [c.topic_id for c in candidates if c.total_score is None]
        if unscored:
            raise TopicSelectorError(
                "candidates not scored yet - run TopicScorer.score() first: " + ", ".join(unscored)
            )

    def _determine_mode(
        self, historical_available: bool, history: SelectionHistory
    ) -> tuple[str, str]:
        if not historical_available:
            coverage = ""
            if history and not history.is_empty:
                coverage = f" (history present but empty: {len(history.records)})"
            return COLD_START, f"no historical data{coverage}; exploring to build channel signal"

        roll = self._rng.random()
        fraction = self._config.selection.exploration_fraction
        if roll < fraction:
            return EXPLORATION, f"roll={roll:.4f} < exploration_fraction={fraction}; exploring"
        return EXPLOITATION, f"roll={roll:.4f} >= exploration_fraction={fraction}; exploit top score"

    def _explore(
        self, ranked: list[TopicCandidate], history: SelectionHistory
    ) -> tuple[TopicCandidate, str]:
        coverage = history.pillar_coverage() if not history.is_empty else {}
        penalty = self._config.selection.coverage_penalty
        top_n = self._config.selection.top_n

        adjusted = [
            (
                candidate,
                (candidate.total_score or 0.0) * _coverage_factor(coverage.get(candidate.pillar, 0), penalty),
            )
            for candidate in ranked
        ]
        adjusted.sort(key=lambda pair: (pair[1], pair[0].topic_id), reverse=True)

        pool = [candidate for candidate, _ in adjusted[:top_n]]
        chosen = self._rng.choice(pool)
        seen = ", ".join(sorted({candidate.pillar for candidate in pool}))
        return chosen, (
            f"explored top-{len(pool)} candidates across pillars [{seen}] "
            f"with coverage penalty {penalty:.2f}"
        )


def _coverage_factor(pillar_count: int, penalty: float) -> float:
    """Degrade scores for pillars already heavily covered."""
    dampened = 1.0 - penalty * pillar_count
    return max(0.0, dampened)