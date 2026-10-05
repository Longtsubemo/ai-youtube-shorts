"""Scene models for simplified pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class Mascot:
    present: bool = True
    action: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Mascot":
        return cls(
            present=bool(data.get("present", True)),
            action=str(data.get("action", "")),
        )


@dataclass(frozen=True)
class Scene:
    scene_id: str
    duration: int = 7
    narration: str = ""
    visual: str = ""
    mascot: Mascot = field(default_factory=Mascot)
    generation_prompt: str = ""

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "duration": self.duration,
            "narration": self.narration,
            "visual": self.visual,
            "mascot": self.mascot.to_dict(),
            "generation_prompt": self.generation_prompt,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Scene":
        mascot = Mascot.from_dict(data.get("mascot", {}))
        return cls(
            scene_id=str(data.get("scene_id", "")),
            duration=int(data.get("duration", 7)),
            narration=str(data.get("narration", "")),
            visual=str(data.get("visual", "")),
            mascot=mascot,
            generation_prompt=str(data.get("generation_prompt", "")),
        )


@dataclass(frozen=True)
class ScenePlan:
    topic_id: str
    scenes: list[Scene] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "topic_id": self.topic_id,
            "scenes": [s.to_dict() for s in self.scenes],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScenePlan":
        scenes = []
        for s in data.get("scenes", []):
            scenes.append(Scene.from_dict(s))
        return cls(
            topic_id=str(data.get("topic_id", "")),
            scenes=scenes,
        )

