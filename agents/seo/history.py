"""Lightweight historical representation for topic selection.

The selector depends only on the HistoryProvider protocol - never on a
database or YouTube Analytics directly. In Stage 1 the provider is fed
from mock/local JSON; later it will read the pipelines analytics store
(docs/ANALYTICS.md).

Cold start = empty SelectionHistory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol


class TopicSelectionHistoryError(ValueError):
    """Raised when history data is malformed."""


@dataclass(frozen=True)
class HistoricalRecord:
    """One published video's performance (lightweight subset for now).

    Later fields (retention curve, impressions, ctr) are additive.
    """

    video_id: str
    topic_id: str
    pillar: str
    topic_type: str
    views: int
    retention: float | None = None
    likes: int = 0
    comments: int = 0
    shares: int = 0
    subscribers_gained: int = 0
    published_at: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "HistoricalRecord":
        required = ("video_id", "topic_id", "pillar", "topic_type", "views")
        missing = [key for key in required if key not in data]
        if missing:
            raise TopicSelectionHistoryError(f"history record missing fields: {', '.join(missing)}")
        return cls(**{key: data.get(key, default) for key, default in cls._defaults().items()})

    @staticmethod
    def _defaults() -> dict:
        return {
            "video_id": "",
            "topic_id": "",
            "pillar": "",
            "topic_type": "",
            "views": 0,
            "retention": None,
            "likes": 0,
            "comments": 0,
            "shares": 0,
            "subscribers_gained": 0,
            "published_at": "",
        }

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "topic_id": self.topic_id,
            "pillar": self.pillar,
            "topic_type": self.topic_type,
            "views": self.views,
            "retention": self.retention,
            "likes": self.likes,
            "comments": self.comments,
            "shares": self.shares,
            "subscribers_gained": self.subscribers_gained,
            "published_at": self.published_at,
        }


@dataclass(frozen=True)
class SelectionHistory:
    """Immutable collection of HistoricalRecords with fast aggregates."""

    records: tuple[HistoricalRecord, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.records)

    @property
    def is_empty(self) -> bool:
        return not self.records

    def pillar_coverage(self) -> dict[str, int]:
        """How many published videos each pillar already has."""
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.pillar] = counts.get(record.pillar, 0) + 1
        return counts

    def topic_coverage(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.topic_id] = counts.get(record.topic_id, 0) + 1
        return counts

    @classmethod
    def from_records(cls, records: Iterable[HistoricalRecord]) -> "SelectionHistory":
        return cls(tuple(records))

    @classmethod
    def from_dict(cls, data: list) -> "SelectionHistory":
        return cls.from_records(HistoricalRecord.from_dict(item) for item in data)

    def to_dict(self) -> list:
        return [record.to_dict() for record in self.records]

    def peak_views(self) -> int:
        return max((record.views for record in self.records), default=0)


class HistoryProvider(Protocol):
    """Anything that can hand the pipeline selection history.

    Implementations may later hide a database, the YouTube Analytics API,
    or a file on disk. The selector never sees the implementation.
    """

    def get_history(self) -> SelectionHistory: ...


class MockHistoryProvider:
    """static history provider for local/mocked data."""

    def __init__(self, records: Iterable[HistoricalRecord] = ()) -> None:
        self._history = SelectionHistory.from_records(records)

    def get_history(self) -> SelectionHistory:
        return self._history