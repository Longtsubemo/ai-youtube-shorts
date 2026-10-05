"""Scene Planning models for Stage 4.

Scene maps a slice of the script's narration to a concrete visual
plan for video generation. One script section may become multiple
scenes. The narration in each scene must come directly from the
script - never rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

MASCOT_ASSET_PATH = "assets/channels/finance_mascot/mascot.png"
MASCOT_SPEAKS = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ScenePlanningError(ValueError):
    """Raised when a scene plan is missing required material."""


@dataclass(frozen=True)
class MascotScene:
    present: bool = True
    action: str = ""
    asset: str = MASCOT_ASSET_PATH

    def to_dict(self) -> dict:
        return {
            "present": self.present,
            "action": self.action,
            "asset": self.asset,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MascotScene":
        if not data:
            return cls()
        return cls(
            present=bool(data.get("present", True)),
            action=str(data.get("action", "")),
            asset=str(data.get("asset", MASCOT_ASSET_PATH)),
        )


@dataclass(frozen=True)
class Mascot:
    present: bool = True
    action: str = ""
    asset: str = MASCOT_ASSET_PATH

    def to_dict(self) -> dict:
        return {
            "present": self.present,
            "action": self.action,
            "asset": self.asset,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Mascot":
        if not data:
            return cls()
        return cls(
            present=bool(data.get("present", True)),
            action=str(data.get("action", "")),
            asset=str(data.get("asset", MASCOT_ASSET_PATH)),
        )


@dataclass(frozen=True)
class Scene:
    scene_id: str
    order: int = 0
    duration_seconds: int = 7
    narration: str = ""
    source_section_id: str = ""
    visual_description: str = ""
    camera: str = ""
    composition: str = ""
    environment: str = ""
    generation_prompt: str = ""
    required_assets: list[str] = field(default_factory=list)
    transition_hint: str = ""
    mascot: Mascot = field(default_factory=Mascot)

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "order": self.order,
            "duration_seconds": self.duration_seconds,
            "narration": self.narration,
            "source_section_id": self.source_section_id,
            "visual_description": self.visual_description,
            "camera": self.camera,
            "composition": self.composition,
            "environment": self.environment,
            "generation_prompt": self.generation_prompt,
            "required_assets": list(self.required_assets),
            "transition_hint": self.transition_hint,
            "mascot": self.mascot.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Scene":
        mascot = Mascot.from_dict(data.get("mascot", {}))
        required = data.get("required_assets") or []
        return cls(
            scene_id=str(data.get("scene_id", "")),
            order=int(data.get("order", data.get("index", 0))),
            duration_seconds=int(data.get("duration_seconds", data.get("duration", 7))),
            narration=str(data.get("narration", "")),
            source_section_id=str(data.get("source_section_id", "")),
            visual_description=str(data.get("visual_description", data.get("visual", ""))),
            camera=str(data.get("camera", "")),
            composition=str(data.get("composition", "")),
            environment=str(data.get("environment", "")),
            generation_prompt=str(data.get("generation_prompt", "")),
            required_assets=[str(x) for x in required],
            transition_hint=str(data.get("transition_hint", "")),
            mascot=mascot,
        )


@dataclass(frozen=True)
class ScenePlan:
    topic_id: str
    title: str = ""
    target_duration_seconds: int = 70
    total_scene_duration_seconds: int = 0
    scenes: list[Scene] = field(default_factory=list)
    channel_id: str = "finance_mascot"
    mascot_asset_path: str = MASCOT_ASSET_PATH
    run_id: str = ""
    created_at: str = field(default_factory=utc_now)
    produced_by: dict = field(
        default_factory=lambda: {"prompt_version": None, "provider": None, "model": None}
    )

    def validate(self) -> "ScenePlan":
        if not self.topic_id or not self.topic_id.strip():
            raise ScenePlanningError("missing required field: topic_id")
        if not self.scenes:
            raise ScenePlanningError("scene plan has no scenes")
        for scene in self.scenes:
            if not scene.narration or not scene.narration.strip():
                raise ScenePlanningError(f"scene {scene.scene_id} has no narration")
            if scene.duration_seconds <= 0:
                raise ScenePlanningError(f"scene {scene.scene_id} has invalid duration")
        return self

    @property
    def estimated_seconds(self) -> int:
        return sum(s.duration_seconds for s in self.scenes)

    def to_dict(self) -> dict:
        return {
            "topic_id": self.topic_id,
            "title": self.title,
            "target_duration_seconds": self.target_duration_seconds,
            "total_scene_duration_seconds": self.total_scene_duration_seconds,
            "scenes": [s.to_dict() for s in self.scenes],
            "channel_id": self.channel_id,
            "mascot_asset_path": self.mascot_asset_path,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "produced_by": dict(self.produced_by),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScenePlan":
        scenes = []
        for idx, s in enumerate(data.get("scenes") or []):
            sc = Scene.from_dict(s)
            scenes.append(Scene(
                scene_id=sc.scene_id or f"sc{idx+1}",
                order=idx,
                duration_seconds=sc.duration_seconds,
                narration=sc.narration,
                source_section_id=sc.source_section_id,
                visual_description=sc.visual_description,
                camera=sc.camera,
                composition=sc.composition,
                environment=sc.environment,
                generation_prompt=sc.generation_prompt,
                required_assets=sc.required_assets,
                transition_hint=sc.transition_hint,
                mascot=sc.mascot,
            ))
        return cls(
            topic_id=str(data.get("topic_id", "")),
            title=str(data.get("title", "")),
            target_duration_seconds=int(data.get("target_duration_seconds", 70)),
            total_scene_duration_seconds=int(data.get("total_scene_duration_seconds", 0)),
            scenes=scenes,
            channel_id=str(data.get("channel_id", "finance_mascot")),
            mascot_asset_path=str(data.get("mascot_asset_path", MASCOT_ASSET_PATH)),
            run_id=str(data.get("run_id", "")),
            created_at=str(data.get("created_at") or utc_now()),
            produced_by=dict(data.get("produced_by") or {"prompt_version": None, "provider": None, "model": None}),
        )

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "ScenePlan":
        import json

        return cls.from_dict(json.loads(text))