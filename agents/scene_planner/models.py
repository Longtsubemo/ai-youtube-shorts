"""Versioned Stage-6 (scene planning) models: Scene, GlobalStyle, ProductionPlan.

Stage 6 converts a Stage-5-approved Script into a provider-independent
ProductionPlan: the stable boundary between the CONTENT SYSTEM and the
VIDEO GENERATION SYSTEM (docs/ARCHITECTURE.md). The planner must NOT introduce
new factual information - it visualizes approved content, and every factual
visual traces back scene -> script line -> approved claim -> evidence -> source.

The ProductionPlan describes what to show, never how some provider should
render it: no video-model names, no API fields, no provider syntax (those
belong to Stage 7).

Follows the project's artifact envelope (docs/DATA_CONTRACTS.md §1):
header (schema/version/created_at/run_id/stage/produced_by) + data payload.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agents.seo.models import Provenance
from pipeline.state import Stage

SCHEMA = "ProductionPlan"
SCHEMA_VERSION = 1
STAGE = Stage.SCENE_PLANNING.value

# Version of the planning ruleset/configuration (recorded in provenance).
PLAN_CONFIG_VERSION = "scene_planning.v1"

# Controlled, provider-independent visual types (docs/DATA_CONTRACTS.md §10).
VISUAL_TYPES = (
    "generated_video",
    "generated_image",
    "stock_asset",
    "diagram",
    "text_only",
    "motion_graphic",
    "existing_asset",
    "transition",
)
# Visual types that require a generation prompt (they will be generated).
GENERATIVE_VISUAL_TYPES = ("generated_video", "generated_image", "motion_graphic", "diagram")

# Controlled transition set; a simple cut is the default (docs/DATA_CONTRACTS.md §10).
TRANSITIONS = ("cut", "fade", "dissolve", "zoom", "match_cut", "none")

# On-screen text kinds. `label`/`caption` annotate factual content and therefore
# require claim traceability; `headline`/`emphasis` are stylistic text.
ON_SCREEN_TEXT_KINDS = ("headline", "emphasis", "label", "caption")
FACTUAL_TEXT_KINDS = ("label", "caption")

DEFAULT_ASPECT_RATIO = "9:16"
DEFAULT_RESOLUTION = "1080x1920"
DEFAULT_FPS = 30

# The `produced_by` envelope helper records which planner produced the plan.
PRODUCED_BY = {
    "prompt_version": None,
    "provider": None,
    "model": "MockScenePlanner",
}

# Standard reason text used when the Stage-5 QA gate blocks planning.
PLANNING_BLOCKED_MESSAGE = (
    "QA gate: ScriptQAResult.overall_verdict is '{verdict}', not PASS - "
    "scene planning is blocked until the script passes QA"
)


def planning_blocked_reason(verdict: str) -> str:
    return PLANNING_BLOCKED_MESSAGE.format(verdict=verdict)


class PlanningError(ValueError):
    """Raised when a plan artifact is missing, malformed, or inconsistent."""


class PlanningBlockedError(PlanningError):
    """Raised when the QA gate blocks planning (ScriptQAResult is not PASS)."""


def _require(value: Any, name: str) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise PlanningError(f"missing required field: {name}")
    return value


def _check_iso(value: Any, name: str) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    candidate = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        datetime.fromisoformat(candidate)
    except ValueError:
        raise PlanningError(f"{name} must be an ISO date/datetime, got {text!r}") from None
    return text


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def production_plan_id_for(run_id: str, script_id: str) -> str:
    """Stable ProductionPlan id per (run_id, script_id); deterministic."""
    digest = hashlib.sha1(f"{run_id or ''}:{script_id}".encode("utf-8")).hexdigest()[:12]
    return f"pp_{digest}"


def scene_id_for(index: int) -> str:
    """Stable sequential scene id; deterministic (scn_01..scn_NN)."""
    return f"scn_{int(index):02d}"


# --------------------------------------------------------------------------- #
# On-screen text
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OnScreenText:
    """A short piece of on-screen text for a scene.

    Tracked separately from narration. `label`/`caption` items annotate factual
    content and must carry `claim_refs` (enforced by the scene-planning
    validator); `headline`/`emphasis` are stylistic. Never duplicates the full
    narration.
    """

    kind: str
    text: str
    claim_refs: tuple[str, ...] = ()

    def validate(self) -> "OnScreenText":
        _require(self.text, "text")
        if self.kind not in ON_SCREEN_TEXT_KINDS:
            allowed = ", ".join(ON_SCREEN_TEXT_KINDS)
            raise PlanningError(f"unknown on_screen_text kind '{self.kind}' (allowed: {allowed})")
        for claim_id in self.claim_refs:
            _require(claim_id, "claimed refs entry")
        return self

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "text": self.text,
            "claim_refs": list(self.claim_refs),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OnScreenText":
        if not isinstance(data, dict):
            raise PlanningError(f"on-screen text must be a mapping, got {data!r}")
        item = cls(
            kind=_require(data.get("kind"), "kind"),
            text=_require(data.get("text"), "text"),
            claim_refs=tuple(data.get("claim_refs") or ()),
        )
        return item.validate()


# --------------------------------------------------------------------------- #
# Asset requirements
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AssetRequirement:
    """A required visual asset for a scene, described provider-independently.

    `asset_type` reuses the controlled visual types; `continuity_group` lets a
    later stage re-use/regenerate the same subject across scenes.
    """

    asset_type: str
    subject: str
    quantity: int = 1
    continuity_group: str = ""

    def validate(self) -> "AssetRequirement":
        if self.asset_type not in VISUAL_TYPES:
            allowed = ", ".join(VISUAL_TYPES)
            raise PlanningError(f"unknown asset_type '{self.asset_type}' (allowed: {allowed})")
        _require(self.subject, "subject")
        if self.quantity < 1:
            raise PlanningError(f"asset quantity must be >= 1, got {self.quantity}")
        return self

    def to_dict(self) -> dict:
        return {
            "asset_type": self.asset_type,
            "subject": self.subject,
            "quantity": self.quantity,
            "continuity_group": self.continuity_group,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AssetRequirement":
        if not isinstance(data, dict):
            raise PlanningError(f"asset requirement must be a mapping, got {data!r}")
        asset = cls(
            asset_type=_require(data.get("asset_type"), "asset_type"),
            subject=_require(data.get("subject"), "subject"),
            quantity=int(data.get("quantity") or 1),
            continuity_group=data.get("continuity_group") or "",
        )
        return asset.validate()


# --------------------------------------------------------------------------- #
# Visual continuity
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ContinuityAsset:
    """A consistent visual identity reused across scenes of one short.

    The first scene that shows a subject establishes its appearance; later
    scenes reference the same `group` instead of reinventing the subject.
    """

    group: str
    subject: str
    appearance: str = ""
    environment: str = ""
    palette: str = ""
    recurring_elements: str = ""

    def validate(self) -> "ContinuityAsset":
        _require(self.group, "group")
        _require(self.subject, "subject")
        return self

    def to_dict(self) -> dict:
        return {
            "group": self.group,
            "subject": self.subject,
            "appearance": self.appearance,
            "environment": self.environment,
            "palette": self.palette,
            "recurring_elements": self.recurring_elements,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContinuityAsset":
        if not isinstance(data, dict):
            raise PlanningError(f"continuity asset must be a mapping, got {data!r}")
        asset = cls(
            group=_require(data.get("group"), "group"),
            subject=_require(data.get("subject"), "subject"),
            appearance=data.get("appearance") or "",
            environment=data.get("environment") or "",
            palette=data.get("palette") or "",
            recurring_elements=data.get("recurring_elements") or "",
        )
        return asset.validate()


# --------------------------------------------------------------------------- #
# Global style
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GlobalStyle:
    """Whole-video style configuration (Shorts default: vertical  9:16)."""

    aspect_ratio: str = DEFAULT_ASPECT_RATIO
    visual_style: str = "educational documentary"
    target_resolution: str = DEFAULT_RESOLUTION
    fps: int = DEFAULT_FPS
    color_palette_notes: str = ""
    consistency_notes: str = ""
    default_transition: str = "cut"
    visual_continuity_requirements: str = ""

    def validate(self) -> "GlobalStyle":
        _require(self.aspect_ratio, "aspect_ratio")
        _require(self.visual_style, "visual_style")
        _require(self.target_resolution, "target_resolution")
        if self.default_transition not in TRANSITIONS:
            allowed = ", ".join(TRANSITIONS)
            raise PlanningError(
                f"unknown default_transition '{self.default_transition}' (allowed: {allowed})"
            )
        if self.fps <= 0:
            raise PlanningError(f"fps must be > 0, got {self.fps}")
        return self

    def to_dict(self) -> dict:
        return {
            "aspect_ratio": self.aspect_ratio,
            "visual_style": self.visual_style,
            "target_resolution": self.target_resolution,
            "fps": self.fps,
            "color_palette_notes": self.color_palette_notes,
            "consistency_notes": self.consistency_notes,
            "default_transition": self.default_transition,
            "visual_continuity_requirements": self.visual_continuity_requirements,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GlobalStyle":
        if not isinstance(data, dict):
            raise PlanningError(f"global style must be a mapping, got {data!r}")
        style = cls(
            aspect_ratio=data.get("aspect_ratio") or DEFAULT_ASPECT_RATIO,
            visual_style=data.get("visual_style") or "educational documentary",
            target_resolution=data.get("target_resolution") or DEFAULT_RESOLUTION,
            fps=int(data.get("fps") or DEFAULT_FPS),
            color_palette_notes=data.get("color_palette_notes") or "",
            consistency_notes=data.get("consistency_notes") or "",
            default_transition=data.get("default_transition") or "cut",
            visual_continuity_requirements=data.get("visual_continuity_requirements") or "",
        )
        return style.validate()


# --------------------------------------------------------------------------- #
# Scene
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Scene:
    """One scene of the production plan.

    `duration_seconds` derives from the mapped script lines' timings; the
    planner never changes narration timing. `claim_refs` keep the audit chain
    scene -> script line -> claim -> evidence -> source. Purely
    stylistic/atmospheric/transition scenes may carry no claim_refs.

    `generation_prompt`/`negative_prompt` are provider-independent descriptions:
    no provider-specific syntax or parameters (those belong to Stage 7).
    """

    scene_id: str
    order: int
    duration_seconds: float
    script_line_ids: tuple[str, ...]
    narration: str
    visual_type: str
    visual_description: str
    claim_refs: tuple[str, ...] = ()
    generation_prompt: str = ""
    negative_prompt: str = ""
    on_screen_text: tuple[OnScreenText, ...] = ()
    transition: str = "cut"
    visual_continuity_group: str = ""
    asset_requirements: tuple[AssetRequirement, ...] = ()
    provenance: dict = field(default_factory=dict)

    def validate(self) -> "Scene":
        _require(self.scene_id, "scene_id")
        if self.order < 1:
            raise PlanningError(f"scene {self.scene_id} has invalid order {self.order}")
        if self.duration_seconds <= 0:
            raise PlanningError(
                f"scene {self.scene_id} has non-positive duration_seconds {self.duration_seconds}"
            )
        for line_id in self.script_line_ids:
            _require(line_id, "script_line_ids entry")
        for claim_id in self.claim_refs:
            _require(claim_id, "claim_refs entry")
        if self.visual_type not in VISUAL_TYPES:
            allowed = ", ".join(VISUAL_TYPES)
            raise PlanningError(f"unknown visual_type '{self.visual_type}' (allowed: {allowed})")
        _require(self.visual_description, "visual_description")
        if self.transition not in TRANSITIONS:
            allowed = ", ".join(TRANSITIONS)
            raise PlanningError(f"unknown transition '{self.transition}' (allowed: {allowed})")
        if self.visual_type != "transition" and not self.narration.strip():
            raise PlanningError(f"scene {self.scene_id} requires narration")
        for item in self.on_screen_text:
            item.validate()
        for asset in self.asset_requirements:
            asset.validate()
        return self

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "order": self.order,
            "duration_seconds": self.duration_seconds,
            "script_line_ids": list(self.script_line_ids),
            "narration": self.narration,
            "claim_refs": list(self.claim_refs),
            "visual_type": self.visual_type,
            "visual_description": self.visual_description,
            "generation_prompt": self.generation_prompt,
            "negative_prompt": self.negative_prompt,
            "on_screen_text": [item.to_dict() for item in self.on_screen_text],
            "transition": self.transition,
            "visual_continuity_group": self.visual_continuity_group,
            "asset_requirements": [asset.to_dict() for asset in self.asset_requirements],
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Scene":
        if not isinstance(data, dict):
            raise PlanningError(f"scene must be a mapping, got {data!r}")
        scene = cls(
            scene_id=_require(data.get("scene_id"), "scene_id"),
            order=int(data.get("order") or 0),
            duration_seconds=float(data.get("duration_seconds") or 0.0),
            script_line_ids=tuple(data.get("script_line_ids") or ()),
            narration=data.get("narration") or "",
            claim_refs=tuple(data.get("claim_refs") or ()),
            visual_type=_require(data.get("visual_type"), "visual_type"),
            visual_description=_require(data.get("visual_description"), "visual_description"),
            generation_prompt=data.get("generation_prompt") or "",
            negative_prompt=data.get("negative_prompt") or "",
            on_screen_text=tuple(OnScreenText.from_dict(item) for item in data.get("on_screen_text") or ()),
            transition=data.get("transition") or "cut",
            visual_continuity_group=data.get("visual_continuity_group") or "",
            asset_requirements=tuple(
                AssetRequirement.from_dict(asset) for asset in data.get("asset_requirements") or ()
            ),
            provenance=dict(data.get("provenance") or {}),
        )
        return scene.validate()


# --------------------------------------------------------------------------- #
# ScenePlanningConfig
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ScenePlanningConfig:
    """Deterministic knobs for the scene-planning stage.

    All thresholds are documented in DATA_CONTRACTS.md §10. `provider_hint_denylist`
    is the lexicon the validator scans plan text for; matching terms fail the
    plan because provider-specific wording must not leak into the contract.
    """

    schema_version: str = PLAN_CONFIG_VERSION
    duration_tolerance_seconds: float = 0.5
    default_aspect_ratio: str = DEFAULT_ASPECT_RATIO
    default_visual_style: str = "educational documentary"
    default_transition: str = "cut"
    default_resolution: str = DEFAULT_RESOLUTION
    fps: int = DEFAULT_FPS
    require_full_coverage: bool = True
    caption_traceability: bool = True
    generation_prompt_required_types: tuple[str, ...] = GENERATIVE_VISUAL_TYPES
    provider_hint_denylist: tuple[str, ...] = (
        "runway",
        "veo",
        "kling",
        "pika",
        "sora",
        "luma",
        "dall-e",
        "dalle",
        "midjourney",
        "stable diffusion",
        "stable-diffusion",
        "flux",
        "comfyui",
    )

    def validate(self) -> "ScenePlanningConfig":
        if self.duration_tolerance_seconds <= 0:
            raise ValueError("duration_tolerance_seconds must be > 0")
        if self.default_transition not in TRANSITIONS:
            raise ValueError(f"default_transition must be one of {TRANSITIONS}")
        if self.fps <= 0:
            raise ValueError("fps must be > 0")
        empty = [term for term in self.provider_hint_denylist if not term.strip()]
        if empty:
            raise ValueError("provider_hint_denylist contains empty terms")
        return self


# --------------------------------------------------------------------------- #
# ProductionPlan
# --------------------------------------------------------------------------- #


@dataclass
class ProductionPlan:
    """The Stage-6 artifact: provider-independent scenes for a QA-passed script.

    This artifact is the stable CONTENT -> VIDEO boundary. Future
    video-generation stages consume it instead of reading Script / ResearchPackage
    directly. `covered_script_line_ids` / `uncovered_script_line_ids` make the
    script -> scene mapping explicit; `script_qa_result_id` records which PASS
    gated this plan.
    """

    production_plan_id: str
    topic_id: str
    topic_title: str
    script_id: str
    script_qa_result_id: str
    scenes: tuple[Scene, ...]
    total_duration_seconds: float
    global_style: GlobalStyle
    covered_script_line_ids: tuple[str, ...] = ()
    uncovered_script_line_ids: tuple[str, ...] = ()
    continuity: tuple[ContinuityAsset, ...] = ()
    production_notes: tuple[str, ...] = ()
    provenance: Provenance | None = None
    created_at: str = field(default_factory=_utc_now)
    schema: str = SCHEMA
    schema_version: int = SCHEMA_VERSION
    stage: str = STAGE
    run_id: str = ""
    produced_by: dict = field(default_factory=lambda: dict(PRODUCED_BY))

    @property
    def scene_ids(self) -> list[str]:
        return [scene.scene_id for scene in self.scenes]

    def validate(self) -> "ProductionPlan":
        _require(self.production_plan_id, "production_plan_id")
        _require(self.topic_id, "topic_id")
        _require(self.topic_title, "topic_title")
        _require(self.script_id, "script_id")
        if not self.scenes:
            raise PlanningError("production plan must contain at least one scene")
        scene_ids = [scene.scene_id for scene in self.scenes]
        if len(set(scene_ids)) != len(scene_ids):
            raise PlanningError("duplicate scene_id in scenes")
        if [scene.order for scene in self.scenes] != list(range(1, len(self.scenes) + 1)):
            raise PlanningError("scene orders must be the contiguous sequence 1..N")
        for scene in self.scenes:
            scene.validate()
        if self.total_duration_seconds <= 0:
            raise PlanningError("total_duration_seconds must be > 0")
        derived = round(sum(scene.duration_seconds for scene in self.scenes), 2)
        if abs(derived - self.total_duration_seconds) > 0.01:
            raise PlanningError(
                f"total_duration_seconds {self.total_duration_seconds} "
                f"does not match derived scene sum {derived}"
            )
        covered = set(self.covered_script_line_ids)
        if set(self.uncovered_script_line_ids) & covered:
            raise PlanningError(
                "covered_script_line_ids overlaps uncovered_script_line_ids; "
                "a line may not be both covered and uncovered"
            )
        for line_id in self.covered_script_line_ids:
            _require(line_id, "covered_script_line_ids entry")
        for line_id in self.uncovered_script_line_ids:
            _require(line_id, "uncovered_script_line_ids entry")
        self.global_style.validate()
        for asset in self.continuity:
            asset.validate()
        if self.provenance is None:
            raise PlanningError("production plan provenance is required")
        return self

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "run_id": self.run_id,
            "stage": self.stage,
            "produced_by": dict(self.produced_by),
            "data": {
                "production_plan_id": self.production_plan_id,
                "topic_id": self.topic_id,
                "topic_title": self.topic_title,
                "script_id": self.script_id,
                "script_qa_result_id": self.script_qa_result_id,
                "scenes": [scene.to_dict() for scene in self.scenes],
                "total_duration_seconds": self.total_duration_seconds,
                "global_style": self.global_style.to_dict(),
                "covered_script_line_ids": list(self.covered_script_line_ids),
                "uncovered_script_line_ids": list(self.uncovered_script_line_ids),
                "continuity": [asset.to_dict() for asset in self.continuity],
                "production_notes": list(self.production_notes),
                "provenance": self.provenance.to_dict() if self.provenance else None,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "ProductionPlan":
        header = dict(data)
        payload = header.pop("data")
        plan = cls(
            production_plan_id=_require(payload.get("production_plan_id"), "production_plan_id"),
            topic_id=_require(payload.get("topic_id"), "topic_id"),
            topic_title=_require(payload.get("topic_title"), "topic_title"),
            script_id=_require(payload.get("script_id"), "script_id"),
            script_qa_result_id=_require(payload.get("script_qa_result_id"), "script_qa_result_id"),
            scenes=tuple(Scene.from_dict(item) for item in payload.get("scenes") or []),
            total_duration_seconds=float(payload.get("total_duration_seconds") or 0.0),
            global_style=GlobalStyle.from_dict(payload.get("global_style") or {}),
            covered_script_line_ids=tuple(payload.get("covered_script_line_ids") or ()),
            uncovered_script_line_ids=tuple(payload.get("uncovered_script_line_ids") or ()),
            continuity=tuple(
                ContinuityAsset.from_dict(asset) for asset in payload.get("continuity") or ()
            ),
            production_notes=tuple(payload.get("production_notes") or ()),
            provenance=Provenance.from_dict(payload["provenance"])
            if payload.get("provenance")
            else None,
            schema=header.get("schema", SCHEMA),
            schema_version=header.get("schema_version", SCHEMA_VERSION),
            created_at=header["created_at"],
            stage=header.get("stage", STAGE),
            run_id=header.get("run_id", ""),
            produced_by=header.get("produced_by") or dict(PRODUCED_BY),
        )
        return plan.validate()

    @classmethod
    def from_json(cls, text: str) -> "ProductionPlan":
        return cls.from_dict(json.loads(text))