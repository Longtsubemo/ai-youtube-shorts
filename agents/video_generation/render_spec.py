"""Render specification contract for Stage 5 (Video Generation).

A render spec is the deterministic, provider-independent intermediate form
between the creative Stage 4 ``ScenePlan`` and the generic scene renderer.
It describes *how* a scene is composed (elements, transforms, animations,
timing) and never *what the content means*. Nothing in this module or in the
renderer knows about scene topics; a new scene is only new data.

Coordinate convention
---------------------

All positions are **normalized** and expressed as fractions of the output
frame:

* ``x = 0.0`` is the left edge, ``x = 1.0`` is the right edge.
* ``y = 0.0`` is the top edge, ``y = 1.0`` is the bottom edge.
* ``(x, y)`` is the **center point** of the element, not its top-left corner.
* ``scale`` is a multiplier applied to the element's *fit size* (see below),
  so ``scale = 1.0`` is the fit size and ``scale = 0.5`` is half of it.
* ``rotation`` is degrees **clockwise** about the element center.
* ``opacity`` ranges from ``0.0`` (invisible) to ``1.0`` (fully opaque).

Base transform positions are validated inside ``[-1.0, 2.0]`` so an element
may legally start or end outside the frame (used by slide-ins), while the
translator only ever emits positions inside ``[0.0, 1.0]``.

Fit size
--------

An image element has a natural pixel size. Its *fit size* is derived from the
frame and the element's ``fit`` mode:

* ``fit = "contain"`` - the image is scaled to fit entirely inside the frame
  (one dimension touches the frame edge). Default for mascots and props.
* ``fit = "cover"`` - the image is scaled to cover the whole frame (one
  dimension overflows). Default for full-frame scene imagery.

``transform.scale`` multiplies the fit size.

Animations
----------

An animation is a generic, data-driven motion applied over a time window::

    {"type": "zoom", "start": 0.0, "duration": 8.0,
     "from": 1.0, "to": 1.06, "easing": "linear"}

Supported types: ``zoom``, ``pan``, ``slide``, ``fade``, ``rotate``.

Value semantics:

* ``zoom``: ``from``/``to`` are multipliers combined with
  ``transform.scale`` (``1.0`` means "no change relative to fit size").
* ``pan``: ``from``/``to`` are absolute normalized centers
  (``{"x": ..., "y": ...}``; a missing component keeps the current value).
* ``slide``: ``from``/``to`` are offsets relative to the element's current
  position (``{"dx": ..., "dy": ...}``).
* ``fade``: ``from``/``to`` are opacity values.
* ``rotate``: ``from``/``to`` are absolute clockwise degrees.

Time behaviour: before ``start`` the ``from`` value is held, between ``start``
and ``start + duration`` the value is interpolated, after the window the ``to``
value is held. ``duration = 0`` snaps to ``to`` at ``start``.

Element timing
--------------

``timing.start``/``timing.duration`` describe when an element exists at all.
A ``duration`` of ``0`` means "until the end of the scene".

Element types
-------------

``image``   a generic image (scene asset id, or an explicit path)
``mascot``  the canonical channel character, rendered from its own asset
            with appearance preserved (resize/move/rotate/opacity only)
``text``    a generic text overlay driven by ``text`` + ``style``
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

SPEC_VERSION = 1

ELEMENT_TYPES = ("image", "mascot", "text")
ANIMATION_TYPES = ("zoom", "pan", "slide", "fade", "rotate")
FIT_MODES = ("contain", "cover")
EASINGS = ("linear", "ease_in_out", "ease_out")

DEFAULT_WIDTH = 1080
DEFAULT_HEIGHT = 1920
DEFAULT_FPS = 30
DEFAULT_BACKGROUND = (14, 16, 22)

# Base positions may sit slightly outside the frame so elements can slide in.
POSITION_LIMIT = (-1.0, 2.0)


class RenderSpecError(ValueError):
    """Raised when a render specification is structurally invalid."""


def _as_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RenderSpecError(f"{label} must be a number, got {value!r}")
    return float(value)


def _as_range(value: float, low: float, high: float, label: str) -> float:
    if not (low <= value <= high):
        raise RenderSpecError(f"{label} must be within [{low}, {high}], got {value}")
    return value


def _as_color(value: Any, label: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(isinstance(c, bool) or not isinstance(c, (int, float)) for c in value)
    ):
        raise RenderSpecError(f"{label} must be an [r, g, b] triple, got {value!r}")
    return tuple(max(0, min(255, int(round(c)))) for c in value)  # type: ignore[return-value]


@dataclass
class Transform:
    """Static placement of an element, in normalized frame coordinates."""

    x: float = 0.5
    y: float = 0.5
    scale: float = 1.0
    rotation: float = 0.0
    opacity: float = 1.0
    fit: str = "contain"

    def validate(self, label: str = "transform") -> None:
        _as_range(
            _as_number(self.x, f"{label}.x"),
            POSITION_LIMIT[0],
            POSITION_LIMIT[1],
            f"{label}.x",
        )
        _as_range(
            _as_number(self.y, f"{label}.y"),
            POSITION_LIMIT[0],
            POSITION_LIMIT[1],
            f"{label}.y",
        )
        scale = _as_number(self.scale, f"{label}.scale")
        if scale <= 0:
            raise RenderSpecError(f"{label}.scale must be > 0, got {scale}")
        _as_number(self.rotation, f"{label}.rotation")
        _as_range(
            _as_number(self.opacity, f"{label}.opacity"),
            0.0,
            1.0,
            f"{label}.opacity",
        )
        if self.fit not in FIT_MODES:
            raise RenderSpecError(f"{label}.fit must be one of {FIT_MODES}, got {self.fit!r}")

    def to_dict(self) -> dict:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "scale": float(self.scale),
            "rotation": float(self.rotation),
            "opacity": float(self.opacity),
            "fit": self.fit,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "Transform":
        data = dict(data or {})
        return cls(
            x=float(data.get("x", 0.5)),
            y=float(data.get("y", 0.5)),
            scale=float(data.get("scale", 1.0)),
            rotation=float(data.get("rotation", 0.0)),
            opacity=float(data.get("opacity", 1.0)),
            fit=str(data.get("fit", "contain")),
        )


@dataclass
class Animation:
    """A generic motion applied to an element over a time window."""

    type: str
    start: float = 0.0
    duration: float = 0.0
    from_value: Any = None
    to_value: Any = None
    easing: str = "linear"

    def validate(self, label: str = "animation") -> None:
        if self.type not in ANIMATION_TYPES:
            raise RenderSpecError(
                f"{label}.type must be one of {ANIMATION_TYPES}, got {self.type!r}"
            )
        start = _as_number(self.start, f"{label}.start")
        if start < 0:
            raise RenderSpecError(f"{label}.start must be >= 0, got {start}")
        duration = _as_number(self.duration, f"{label}.duration")
        if duration < 0:
            raise RenderSpecError(f"{label}.duration must be >= 0, got {duration}")
        if self.easing not in EASINGS:
            raise RenderSpecError(
                f"{label}.easing must be one of {EASINGS}, got {self.easing!r}"
            )
        if self.type in ("zoom", "fade", "rotate"):
            _as_number(self.from_value, f"{label}.from")
            _as_number(self.to_value, f"{label}.to")
        if self.type == "pan":
            self._validate_point(self.from_value, f"{label}.from")
            self._validate_point(self.to_value, f"{label}.to")
        if self.type == "slide":
            self._validate_delta(self.from_value, f"{label}.from")
            self._validate_delta(self.to_value, f"{label}.to")

    @staticmethod
    def _validate_point(value: Any, label: str) -> None:
        if value is None:
            return
        if not isinstance(value, dict):
            raise RenderSpecError(f"{label} must be a dict like {{'x': 0.5, 'y': 0.5}}")
        for key in value:
            if key not in ("x", "y"):
                raise RenderSpecError(f"{label} has unsupported key {key!r}")
            _as_number(value[key], f"{label}.{key}")

    @staticmethod
    def _validate_delta(value: Any, label: str) -> None:
        if value is None:
            return
        if not isinstance(value, dict):
            raise RenderSpecError(f"{label} must be a dict like {{'dx': 0.1, 'dy': 0.0}}")
        for key in value:
            if key not in ("dx", "dy"):
                raise RenderSpecError(f"{label} has unsupported key {key!r}")
            _as_number(value[key], f"{label}.{key}")

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "start": float(self.start),
            "duration": float(self.duration),
            "from": self.from_value,
            "to": self.to_value,
            "easing": self.easing,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Animation":
        if not isinstance(data, dict):
            raise RenderSpecError(f"animation must be a dict, got {type(data).__name__}")
        return cls(
            type=str(data.get("type", "")),
            start=float(data.get("start", 0.0)),
            duration=float(data.get("duration", 0.0)),
            from_value=data.get("from"),
            to_value=data.get("to"),
            easing=str(data.get("easing", "linear")),
        )


@dataclass
class ElementTiming:
    """When an element exists inside its scene."""

    start: float = 0.0
    duration: float = 0.0  # 0 => until the end of the scene

    def validate(self, label: str = "timing") -> None:
        start = _as_number(self.start, f"{label}.start")
        if start < 0:
            raise RenderSpecError(f"{label}.start must be >= 0, got {start}")
        duration = _as_number(self.duration, f"{label}.duration")
        if duration < 0:
            raise RenderSpecError(f"{label}.duration must be >= 0, got {duration}")

    def end(self, scene_duration: float) -> float:
        if self.duration <= 0:
            return float(scene_duration)
        return float(self.start + self.duration)

    def to_dict(self) -> dict:
        return {"start": float(self.start), "duration": float(self.duration)}

    @classmethod
    def from_dict(cls, data: dict | None) -> "ElementTiming":
        data = dict(data or {})
        return cls(
            start=float(data.get("start", 0.0)),
            duration=float(data.get("duration", 0.0)),
        )


@dataclass
class RenderElement:
    """One drawable thing inside a scene."""

    id: str
    type: str
    asset: str = ""
    text: str = ""
    layer: int = 0
    transform: Transform = field(default_factory=Transform)
    animations: list[Animation] = field(default_factory=list)
    timing: ElementTiming = field(default_factory=ElementTiming)
    style: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def validate(self, label: str = "element") -> None:
        if not self.id or not str(self.id).strip():
            raise RenderSpecError(f"{label} is missing an id")
        if self.type not in ELEMENT_TYPES:
            raise RenderSpecError(
                f"{label} ({self.id}) type must be one of {ELEMENT_TYPES}, "
                f"got {self.type!r}"
            )
        if self.type in ("image", "mascot") and not self.asset:
            raise RenderSpecError(f"{label} ({self.id}) of type {self.type!r} needs an asset")
        if self.type == "text" and not str(self.text).strip():
            raise RenderSpecError(f"{label} ({self.id}) of type 'text' needs text")
        if not isinstance(self.layer, int) or isinstance(self.layer, bool):
            raise RenderSpecError(f"{label} ({self.id}) layer must be an int")
        self.transform.validate(f"{label}.{self.id}.transform")
        self.timing.validate(f"{label}.{self.id}.timing")
        if not isinstance(self.animations, list):
            raise RenderSpecError(f"{label} ({self.id}) animations must be a list")
        for idx, animation in enumerate(self.animations):
            animation.validate(f"{label}.{self.id}.animations[{idx}]")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "asset": self.asset,
            "text": self.text,
            "layer": int(self.layer),
            "transform": self.transform.to_dict(),
            "animations": [a.to_dict() for a in self.animations],
            "timing": self.timing.to_dict(),
            "style": dict(self.style),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RenderElement":
        if not isinstance(data, dict):
            raise RenderSpecError(f"element must be a dict, got {type(data).__name__}")
        return cls(
            id=str(data.get("id", "")),
            type=str(data.get("type", "")),
            asset=str(data.get("asset", "")),
            text=str(data.get("text", "")),
            layer=int(data.get("layer", 0)),
            transform=Transform.from_dict(data.get("transform")),
            animations=[Animation.from_dict(a) for a in (data.get("animations") or [])],
            timing=ElementTiming.from_dict(data.get("timing")),
            style=dict(data.get("style") or {}),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class SceneRenderSpec:
    """Everything needed to render one scene clip."""

    scene_id: str
    order: int = 0
    duration_seconds: float = 7.0
    background: tuple[int, int, int] = DEFAULT_BACKGROUND
    elements: list[RenderElement] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def validate(self, label: str = "scene") -> None:
        if not self.scene_id or not str(self.scene_id).strip():
            raise RenderSpecError(f"{label} is missing a scene_id")
        duration = _as_number(self.duration_seconds, f"{label} ({self.scene_id}).duration_seconds")
        if duration <= 0:
            raise RenderSpecError(
                f"{label} {self.scene_id} duration_seconds must be > 0, got {duration}"
            )
        if isinstance(self.order, bool) or not isinstance(self.order, int):
            raise RenderSpecError(f"{label} {self.scene_id} order must be an int")
        _as_color(list(self.background), f"{label} {self.scene_id}.background")
        if not isinstance(self.elements, list):
            raise RenderSpecError(f"{label} {self.scene_id} elements must be a list")
        seen: set[str] = set()
        for element in self.elements:
            element.validate(f"{label} {self.scene_id}")
            if element.id in seen:
                raise RenderSpecError(
                    f"{label} {self.scene_id} has duplicate element id {element.id!r}"
                )
            seen.add(element.id)

    @property
    def frame_count_hint(self) -> int:
        return max(1, int(round(self.duration_seconds)))

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "order": int(self.order),
            "duration_seconds": float(self.duration_seconds),
            "background": list(self.background),
            "elements": [e.to_dict() for e in self.elements],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SceneRenderSpec":
        if not isinstance(data, dict):
            raise RenderSpecError(f"scene must be a dict, got {type(data).__name__}")
        return cls(
            scene_id=str(data.get("scene_id", "")),
            order=int(data.get("order", 0)),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            background=_as_color(data.get("background", list(DEFAULT_BACKGROUND)), "background"),
            elements=[RenderElement.from_dict(e) for e in (data.get("elements") or [])],
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class RenderSpec:
    """A whole video: ordered scenes plus output settings."""

    run_id: str = ""
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    fps: int = DEFAULT_FPS
    background: tuple[int, int, int] = DEFAULT_BACKGROUND
    scenes: list[SceneRenderSpec] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    version: int = SPEC_VERSION

    def validate(self) -> "RenderSpec":
        if int(self.version) != SPEC_VERSION:
            raise RenderSpecError(
                f"unsupported render spec version {self.version}, expected {SPEC_VERSION}"
            )
        for name, value in (("width", self.width), ("height", self.height), ("fps", self.fps)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RenderSpecError(f"{name} must be a positive int, got {value!r}")
        _as_color(list(self.background), "background")
        if not self.scenes:
            raise RenderSpecError("render spec has no scenes")
        seen: set[str] = set()
        for scene in self.scenes:
            scene.validate("scene")
            if scene.scene_id in seen:
                raise RenderSpecError(f"duplicate scene_id {scene.scene_id!r}")
            seen.add(scene.scene_id)
        return self

    @property
    def ordered_scenes(self) -> list[SceneRenderSpec]:
        """Scenes in playback order (by order, then scene_id)."""
        return sorted(self.scenes, key=lambda s: (s.order, s.scene_id))

    @property
    def total_duration_seconds(self) -> float:
        return float(sum(s.duration_seconds for s in self.ordered_scenes))

    @property
    def scene_ids(self) -> list[str]:
        return [s.scene_id for s in self.ordered_scenes]

    def get_scene(self, scene_id: str) -> SceneRenderSpec:
        for scene in self.scenes:
            if scene.scene_id == scene_id:
                return scene
        raise RenderSpecError(f"unknown scene_id {scene_id!r}")

    def to_dict(self) -> dict:
        return {
            "version": int(self.version),
            "run_id": self.run_id,
            "width": int(self.width),
            "height": int(self.height),
            "fps": int(self.fps),
            "background": list(self.background),
            "scenes": [s.to_dict() for s in self.ordered_scenes],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RenderSpec":
        if not isinstance(data, dict):
            raise RenderSpecError(f"render spec must be a dict, got {type(data).__name__}")
        return cls(
            run_id=str(data.get("run_id", "")),
            width=int(data.get("width", DEFAULT_WIDTH)),
            height=int(data.get("height", DEFAULT_HEIGHT)),
            fps=int(data.get("fps", DEFAULT_FPS)),
            background=_as_color(data.get("background", list(DEFAULT_BACKGROUND)), "background"),
            scenes=[SceneRenderSpec.from_dict(s) for s in (data.get("scenes") or [])],
            metadata=dict(data.get("metadata") or {}),
            version=int(data.get("version", SPEC_VERSION)),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "RenderSpec":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RenderSpecError(f"render spec is not valid JSON: {exc}") from exc
        return cls.from_dict(data)
