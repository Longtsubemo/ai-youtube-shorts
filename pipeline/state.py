"""Minimal pipeline state model.

Tracks one production run through the pipeline stages. Agents and
stages communicate through structured file-based data contracts (JSON),
and the state records where each run is and what it has produced.

Only the scaffolding is intentional here; stage implementations and
artifact contracts are added as each stage is built.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Stage(str, Enum):
    """Ordered pipeline stages, matching the documented pipeline."""

    TOPIC_DISCOVERY = "topic_discovery"
    SEO_RESEARCH = "seo_research"
    TOPIC_SELECTION = "topic_selection"
    RESEARCH = "research"
    FACT_CHECK = "fact_check"
    SCRIPT_GENERATION = "script_generation"
    SCENE_PLANNING = "scene_planning"
    PRODUCTION = "production"
    VIDEO_GENERATION = "video_generation"
    AUDIO = "audio"
    VIDEO_ASSEMBLY = "video_assembly"
    QUALITY_CHECK = "quality_check"
    PUBLISH = "publish"
    ANALYTICS = "analytics"


@dataclass
class PipelineState:
    """Serializable state for a single production run."""

    run_id: str
    status: str = "initialized"
    current_stage: Stage = Stage.TOPIC_DISCOVERY
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
        """Record that the run has moved to a later pipeline stage."""
        self.current_stage = stage
        self.updated_at = _utc_now()

    def add_artifact(self, stage: Stage, path: str | Path) -> None:
        """Record an artifact produced by a stage."""
        self.artifacts.setdefault(stage.value, []).append(str(path))
        self.updated_at = _utc_now()