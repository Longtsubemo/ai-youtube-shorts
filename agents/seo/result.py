"""Versioned TopicSelection artifact.

Uses the project's artifact envelope (docs/DATA_CONTRACTS.md):
header (schema/version/created_at/run_id/stage/produced_by) + data.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from agents.seo.models import TopicCandidate, utc_now
from pipeline.state import Stage

SCHEMA = "TopicSelection"
SCHEMA_VERSION = 1
STAGE = Stage.TOPIC_SELECTION.value


@dataclass
class TopicSelection:
    selection_id: str
    selected_topic_id: str
    selected_topic: dict
    ranked_candidates: list[dict]
    selection_mode: str
    historical_data_available: bool
    selection_reason: str
    scoring_configuration: dict
    created_at: str
    schema: str = SCHEMA
    schema_version: int = SCHEMA_VERSION
    stage: str = STAGE
    run_id: str = ""
    produced_by: dict = field(
        default_factory=lambda: {"prompt_version": None, "provider": None, "model": None}
    )

    @classmethod
    def build(
        cls,
        *,
        run_id: str,
        selected: TopicCandidate,
        ranked_candidates: list[TopicCandidate],
        selection_mode: str,
        historical_data_available: bool,
        selection_reason: str,
        scoring_configuration: dict,
        selection_id: str | None = None,
        created_at: str | None = None,
    ) -> "TopicSelection":
        return cls(
            selection_id=selection_id or f"ts_{uuid.uuid4().hex[:12]}",
            selected_topic_id=selected.topic_id,
            selected_topic=selected.to_dict(),
            ranked_candidates=[candidate.to_dict() for candidate in ranked_candidates],
            selection_mode=selection_mode,
            historical_data_available=historical_data_available,
            selection_reason=selection_reason,
            scoring_configuration=scoring_configuration,
            created_at=created_at or utc_now(),
            run_id=run_id,
        )

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "run_id": self.run_id,
            "stage": self.stage,
            "produced_by": dict(self.produced_by),
            "data": {
                "selection_id": self.selection_id,
                "selected_topic_id": self.selected_topic_id,
                "selected_topic": self.selected_topic,
                "ranked_candidates": self.ranked_candidates,
                "selection_mode": self.selection_mode,
                "historical_data_available": self.historical_data_available,
                "selection_reason": self.selection_reason,
                "scoring_configuration": self.scoring_configuration,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "TopicSelection":
        header = dict(data)
        payload = header.pop("data")
        return cls(
            selection_id=payload["selection_id"],
            selected_topic_id=payload["selected_topic_id"],
            selected_topic=payload["selected_topic"],
            ranked_candidates=payload["ranked_candidates"],
            selection_mode=payload["selection_mode"],
            historical_data_available=payload["historical_data_available"],
            selection_reason=payload["selection_reason"],
            scoring_configuration=payload["scoring_configuration"],
            schema=header.get("schema", SCHEMA),
            schema_version=header.get("schema_version", SCHEMA_VERSION),
            created_at=header["created_at"],
            stage=header.get("stage", STAGE),
            run_id=header.get("run_id", ""),
            produced_by=header.get("produced_by") or {"prompt_version": None, "provider": None, "model": None},
        )

    @classmethod
    def from_json(cls, text: str) -> "TopicSelection":
        return cls.from_dict(json.loads(text))