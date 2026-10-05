"""Stage 2 (AI Research) output contract.

`AIResearch` is the structured material a later script-generation stage
consumes directly. It is deliberately a *content* contract: it carries no
sources, no evidence, no citations and no score. Gemini's own knowledge is the
research basis, so there is nothing here to corroborate and no factuality gate
to pass.

Fields are permissive on purpose. `numbers` may legitimately be empty - the
model is instructed never to invent precise figures just to fill the list, so
an empty list is a valid, correct result rather than a failure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

RESEARCH_FIELDS = (
    "key_concepts",
    "explanation",
    "examples",
    "numbers",
    "interesting_angles",
    "common_misconceptions",
    "story_ideas",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AIResearchError(ValueError):
    """Raised when research output is missing its required material.

    This is a *structural* check only (a topic and some explanation). It is
    not a quality score, a PASS/FAIL gate, or a factuality gate.
    """


@dataclass(frozen=True)
class ResearchTopic:
    """The Stage 1 selection distilled into what research actually needs.

    Built from the normal ``topic_selection.json`` artifact - there is no
    separate research input format.
    """

    topic_id: str
    title: str
    pillar: str
    angle: str
    hook_direction: str
    topic_type: str = ""
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "topic_id": self.topic_id,
            "title": self.title,
            "pillar": self.pillar,
            "angle": self.angle,
            "hook_direction": self.hook_direction,
            "topic_type": self.topic_type,
            "description": self.description,
        }


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
    pillar: str = ""
    run_id: str = ""
    created_at: str = field(default_factory=utc_now)
    produced_by: dict = field(
        default_factory=lambda: {"prompt_version": None, "provider": None, "model": None}
    )

    def validate(self) -> "AIResearch":
        """Structural validation only.

        Requires a topic id and an explanation. Deliberately does NOT require
        numbers, examples, sources, or any minimum count: there is no quality
        bar and no factuality gate in this stage.
        """
        if not self.topic_id or not self.topic_id.strip():
            raise AIResearchError("missing required field: topic_id")
        if not self.explanation or not self.explanation.strip():
            raise AIResearchError("missing required field: explanation")
        return self

    def to_dict(self) -> dict:
        data: dict[str, Any] = {"topic_id": self.topic_id}
        data.update({name: getattr(self, name) for name in RESEARCH_FIELDS})
        data["pillar"] = self.pillar
        data["run_id"] = self.run_id
        data["created_at"] = self.created_at
        data["produced_by"] = dict(self.produced_by)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "AIResearch":
        return cls(
            topic_id=str(data.get("topic_id", "")),
            key_concepts=list(data.get("key_concepts") or []),
            explanation=str(data.get("explanation", "")),
            examples=list(data.get("examples") or []),
            numbers=list(data.get("numbers") or []),
            interesting_angles=list(data.get("interesting_angles") or []),
            common_misconceptions=list(data.get("common_misconceptions") or []),
            story_ideas=list(data.get("story_ideas") or []),
            pillar=str(data.get("pillar", "")),
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
    def from_json(cls, text: str) -> "AIResearch":
        import json

        return cls.from_dict(json.loads(text))