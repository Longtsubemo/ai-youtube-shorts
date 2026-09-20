"""Competition -> opportunity mapping (isolated and replaceable).

Competition must NOT be read as "more competition = better". By default
higher competition lowers opportunity. This single step is walled off so
a more sophisticated model (share-of-voice, saturation curves, etc.) can
slot in without touching the scorer.
"""

from __future__ import annotations


class CompetitionMappingError(ValueError):
    """Raised for unknown transformation configurations."""


def competition_to_opportunity(competition: float, transformation: str = "inverse") -> float:
    """Map a raw competition signal (0=no competition, 1=saturated) to
    an opportunity value used in scoring."""
    if not 0.0 <= competition <= 1.0:
        raise CompetitionMappingError(f"competition out of range [0,1]: {competition}")
    if transformation == "inverse":
        # Saturating loss of opportunity as competition rises.
        return max(0.0, min(1.0, 1.0 - competition))
    raise CompetitionMappingError(f"unknown competition transformation: {transformation}")