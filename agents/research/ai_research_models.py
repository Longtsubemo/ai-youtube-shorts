"""AI Research output for simplified pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class AIResearch:
    topic_id: str
    key_concepts: list[str] = field(default_factory=list)
    explanation: str = ""
    examples: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    interesting_angles: list[str] = field(default_factory=list)
    common_misconceptions: list[str] = field(default_factory=list)
    story_ideas: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AIResearch":
        return cls(
            topic_id=str(data.get("topic_id", "")),
            key_concepts=list(data.get("key_concepts", [])),
            explanation=str(data.get("explanation", "")),
            examples=list(data.get("examples", [])),
            numbers=list(data.get("numbers", [])),
            interesting_angles=list(data.get("interesting_angles", [])),
            common_misconceptions=list(data.get("common_misconceptions", [])),
            story_ideas=list(data.get("story_ideas", [])),
        )

