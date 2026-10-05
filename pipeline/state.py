"""Minimal pipeline state model for 5-stage finance_mascot pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Stage(str, Enum):
    """Ordered pipeline stages."""

    TOPIC_SELECTION = "topic_selection"
    AI_RESEARCH = "ai_research"
    SCRIPT_GENERATION = "script_generation"
    SCENE_PLANNING = "scene_planning"
    VIDEO_GENERATION = "video_generation"


@dataclass
class PipelineState:
    """Serializable state for a single production run."""

    run_id: str
    status: str = "initialized"
    current_stage: Stage = Stage.TOPIC_SELECTION
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    artifacts: dict[str, list[str]] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = _utc_now()
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PipelineState":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["current_stage"] = Stage(data["current_stage"])
        return cls(**data)

    def advance_to(self, stage: Stage) -> None:
        self.current_stage = stage
        self.updated_at = _utc_now()

    def add_artifact(self, stage: Stage, path: str | Path) -> None:
        self.artifacts.setdefault(stage.value, []).append(str(path))
        self.updated_at = _utc_now()

