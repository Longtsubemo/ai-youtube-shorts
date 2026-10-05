"""Script models for simplified pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class ScriptSection:
    type: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ScriptSection":
        return cls(type=str(data.get("type", "")), text=str(data.get("text", "")))


@dataclass(frozen=True)
class Script:
    topic_id: str
    title: str
    duration_target_seconds: int = 70
    narration: str = ""
    sections: list[ScriptSection] = field(default_factory=list)

    def to_dict(self) -> dict:
        result = {
            "topic_id": self.topic_id,
            "title": self.title,
            "duration_target_seconds": self.duration_target_seconds,
            "narration": self.narration,
            "sections": [s.to_dict() for s in self.sections],
        }
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "Script":
        sections = []
        for s in data.get("sections", []):
            sections.append(ScriptSection.from_dict(s))
        return cls(
            topic_id=str(data.get("topic_id", "")),
            title=str(data.get("title", "")),
            duration_target_seconds=int(data.get("duration_target_seconds", 70)),
            narration=str(data.get("narration", "")),
            sections=sections,
        )

