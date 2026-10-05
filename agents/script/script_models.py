"""Stage 3 (Script Generation) output contract.

`Script` is a finished spoken YouTube Shorts script built from the Stage 1
selection plus the Stage 2 research. It is deliberately a *content* contract:
no sources, no citations, no QA score, no verdict.

The script is structured rather than one giant string. `narration` is the full
spoken text for convenience, and `sections` carries the same narration split
into ordered, individually-timed segments. Each section holds exactly the words
belonging to it, so a later Scene Planning stage can map scenes onto narration
without re-parsing the blob.

Section `role` is descriptive, not a gate: the model may structure the script
as a mystery, a question, a story, a comparison, or a reveal. `role` simply
records the beat so downstream stages know what each segment is doing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Channel format constants for a YouTube Short.
TARGET_DURATION_SECONDS = 70
DURATION_MIN_SECONDS = 60
DURATION_TARGET_RANGE = (65, 80)

# Roles a section may carry. Descriptive labels, not an enforced template.
SECTION_ROLES = (
    "hook",
    "setup",
    "explanation",
    "example",
    "reveal",
    "payoff",
    "closing",
)

EDUCATIONAL_NOTICE = (
    "Educational content only. This script explains general financial concepts "
    "and is not personalized investment, tax, or financial advice."
)

MASCOT_SPEAKS = False

MASCOT_NOTE = (
    "The financial-detective fox mascot appears visually alongside the "
    "narration and does not speak."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ScriptError(ValueError):
    """Raised when a script is missing the material Scene Planning needs.

    Structural only: a topic, a title, narration, and at least one section that
    actually has words. There is no quality score and no PASS/FAIL verdict.
    """


@dataclass(frozen=True)
class ScriptSection:
    """One ordered, individually-timed narration segment."""

    section_id: str
    role: str
    narration: str
    estimated_seconds: int = 0
    visual_hint: str = ""

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "role": self.role,
            "narration": self.narration,
            "estimated_seconds": self.estimated_seconds,
            "visual_hint": self.visual_hint,
        }

    @classmethod
    def from_dict(cls, data: dict, index: int = 0) -> "ScriptSection":
        # Tolerate the older `type`/`text` spelling so pre-existing artifacts
        # still load.
        return cls(
            section_id=str(data.get("section_id") or f"s{index + 1}"),
            role=str(data.get("role") or data.get("type") or ""),
            narration=str(data.get("narration") or data.get("text") or ""),
            estimated_seconds=int(data.get("estimated_seconds") or 0),
            visual_hint=str(data.get("visual_hint") or ""),
        )


@dataclass(frozen=True)
class Script:
    topic_id: str
    title: str
    duration_target_seconds: int = TARGET_DURATION_SECONDS
    narration: str = ""
    sections: list[ScriptSection] = field(default_factory=list)
    pillar: str = ""
    channel_id: str = ""
    mascot_speaks: bool = MASCOT_SPEAKS
    mascot_note: str = MASCOT_NOTE
    educational_notice: str = EDUCATIONAL_NOTICE
    run_id: str = ""
    created_at: str = field(default_factory=utc_now)
    produced_by: dict = field(
        default_factory=lambda: {"prompt_version": None, "provider": None, "model": None}
    )

    def validate(self) -> "Script":
        """Structural validation only - no score, no PASS/FAIL verdict, no QA gate."""
        if not self.topic_id or not self.topic_id.strip():
            raise ScriptError("missing required field: topic_id")
        if not self.title or not self.title.strip():
            raise ScriptError("missing required field: title")
        if not self.narration or not self.narration.strip():
            raise ScriptError("missing required field: narration")
        if not self.sections:
            raise ScriptError("script has no sections for Scene Planning")
        for section in self.sections:
            if not section.narration or not section.narration.strip():
                raise ScriptError(f"section {section.section_id} has no narration")
        if self.duration_target_seconds <= 0:
            raise ScriptError("duration_target_seconds must be > 0")
        return self

    @property
    def estimated_seconds(self) -> int:
        return sum(section.estimated_seconds for section in self.sections)

    def to_dict(self) -> dict:
        return {
            "topic_id": self.topic_id,
            "title": self.title,
            "duration_target_seconds": self.duration_target_seconds,
            "narration": self.narration,
            "sections": [section.to_dict() for section in self.sections],
            "pillar": self.pillar,
            "channel_id": self.channel_id,
            "mascot_speaks": self.mascot_speaks,
            "mascot_note": self.mascot_note,
            "educational_notice": self.educational_notice,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "produced_by": dict(self.produced_by),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Script":
        sections = [
            ScriptSection.from_dict(section, index)
            for index, section in enumerate(data.get("sections") or [])
        ]
        return cls(
            topic_id=str(data.get("topic_id", "")),
            title=str(data.get("title", "")),
            duration_target_seconds=int(
                data.get("duration_target_seconds") or TARGET_DURATION_SECONDS
            ),
            narration=str(data.get("narration", "")),
            sections=sections,
            pillar=str(data.get("pillar", "")),
            channel_id=str(data.get("channel_id", "")),
            mascot_speaks=bool(data.get("mascot_speaks", MASCOT_SPEAKS)),
            mascot_note=str(data.get("mascot_note", MASCOT_NOTE)),
            educational_notice=str(data.get("educational_notice", EDUCATIONAL_NOTICE)),
            run_id=str(data.get("run_id", "")),
            created_at=str(data.get("created_at") or utc_now()),
            produced_by=dict(
                data.get("produced_by")
                or {"prompt_version": None, "provider": None, "model": None}
            ),
        )

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "Script":
        import json

        return cls.from_dict(json.loads(text))