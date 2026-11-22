"""Deterministic translation from the Stage 4 ``ScenePlan`` to a render spec.

This is the interpreter between the creative plan and the generic renderer.
It is **provider independent** (no Gemini, no network) and **content
agnostic**: it maps fields that every scene has - duration, narration,
camera, composition, environment, required assets, mascot, transition hint -
onto generic elements, transforms, animations and timing.

Nothing here branches on scene ids, topics or visual text. The only tables are
generic film vocabulary (camera movement words) and generic layout defaults,
so a new scene renders from data alone.

Creative fields are not thrown away: they are carried into each scene's
``metadata`` block of the render spec.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from agents.scene_planner.scene_models import MASCOT_ASSET_PATH, Scene, ScenePlan
from agents.video_generation.assets import is_channel_asset_ref
from agents.video_generation.render_spec import (
    DEFAULT_BACKGROUND,
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    Animation,
    ElementTiming,
    RenderElement,
    RenderSpec,
    SceneRenderSpec,
    Transform,
)

TRANSLATOR_NAME = "local-deterministic"
TRANSLATOR_VERSION = "stage5-translator/1.0"

# Fields a scene may use to put words on screen. Stage 4 does not emit them
# yet; when it does, text becomes data rather than renderer code.
SCENE_TEXT_KEYS = ("on_screen_text", "text_overlay", "caption", "headline", "text")

TEXT_STYLE_DEFAULTS: dict[str, Any] = {
    "size": 0.05,  # fraction of frame height
    "color": "#FFFFFF",
    "align": "center",
}

# Generic camera vocabulary -> generic motion. Unknown words fall through to
# DEFAULT_CAMERA_MOTION, so no scene is ever special-cased.
CAMERA_MOTION_TABLE: tuple[tuple[tuple[str, ...], tuple[dict, ...]], ...] = (
    (
        ("push in", "zoom in", "dolly in"),
        ({"type": "zoom", "from": 1.0, "to": 1.07},),
    ),
    (
        ("pull out", "zoom out", "dolly out"),
        ({"type": "zoom", "from": 1.07, "to": 1.0},),
    ),
    (
        ("pan left", "left pan"),
        (
            {
                "type": "slide",
                "from": {"dx": 0.05, "dy": 0.0},
                "to": {"dx": -0.05, "dy": 0.0},
            },
        ),
    ),
    (
        ("pan right", "right pan"),
        (
            {
                "type": "slide",
                "from": {"dx": -0.05, "dy": 0.0},
                "to": {"dx": 0.05, "dy": 0.0},
            },
        ),
    ),
    (
        ("tracking", "track"),
        (
            {
                "type": "slide",
                "from": {"dx": -0.04, "dy": 0.0},
                "to": {"dx": 0.04, "dy": 0.0},
            },
        ),
    ),
    (("static", "locked", "still", "tripod"), ()),
)

DEFAULT_CAMERA_MOTION: tuple[dict, ...] = ({"type": "zoom", "from": 1.0, "to": 1.04},)

MASCOT_POSITION = {"x": 0.5, "y": 0.7, "scale": 0.45}
MAIN_VISUAL_POSITION = {"x": 0.5, "y": 0.45, "scale": 1.0}
SECONDARY_POSITION = {"x": 0.5, "y": 0.32, "scale": 0.6}
TEXT_POSITION = {"x": 0.5, "y": 0.16, "scale": 1.0}

LAYER_MAIN_VISUAL = 0
LAYER_SECONDARY = 10
LAYER_MASCOT = 50
LAYER_TEXT = 100

_SLUG_RE = re.compile(r"[^a-z0-9_]+")


def _slug(value: str) -> str:
    slug = _SLUG_RE.sub("_", str(value).lower()).strip("_")
    return slug or "asset"


def _animation(template: dict, duration: float) -> Animation:
    return Animation(
        type=str(template["type"]),
        start=0.0,
        duration=float(duration),
        from_value=template.get("from"),
        to_value=template.get("to"),
        easing=str(template.get("easing", "linear")),
    )


def camera_animations(camera: str, duration: float) -> list[Animation]:
    """Map free-form camera wording onto generic motion primitives."""
    key = str(camera or "").lower()
    for keywords, templates in CAMERA_MOTION_TABLE:
        if any(word in key for word in keywords):
            return [_animation(t, duration) for t in templates]
    return [_animation(t, duration) for t in DEFAULT_CAMERA_MOTION]


def mascot_idle_animations(duration: float) -> list[Animation]:
    """Generic motion for the mascot stand-in.

    The semantic action recorded by Stage 4 (pointing, nodding, ...) is kept
    as element metadata for a future character-animation seam; the first
    renderer only fades the mascot in and gives it a slow, subtle zoom.
    """
    fade_duration = max(0.2, min(0.5, duration * 0.25))
    return [
        Animation(type="fade", start=0.0, duration=fade_duration, from_value=0.0, to_value=1.0),
        Animation(type="zoom", start=0.0, duration=duration, from_value=1.0, to_value=1.03),
    ]


def _is_mascot_ref(ref: str, plan: ScenePlan) -> bool:
    ref = str(ref or "").replace("\\", "/")
    if not ref:
        return False
    if is_channel_asset_ref(ref):
        return True
    return bool(plan.mascot_asset_path) and ref == plan.mascot_asset_path.replace("\\", "/")


def _required_visual_assets(scene: Scene, plan: ScenePlan) -> tuple[str, list[str]]:
    """Split ``required_assets`` into (primary, secondary), mascot excluded."""
    primary = ""
    secondary: list[str] = []
    for ref in scene.required_assets:
        if not ref or _is_mascot_ref(ref, plan):
            continue
        if not primary:
            primary = ref
        else:
            secondary.append(ref)
    if not primary:
        primary = f"{scene.scene_id}_main"
    return primary, secondary


def _text_for_scene(
    scene: Scene,
    raw: dict,
    text_overrides: dict[str, str],
) -> str:
    if scene.scene_id in text_overrides:
        return str(text_overrides[scene.scene_id])
    for key in SCENE_TEXT_KEYS:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def translate_scene(
    scene: Scene,
    plan: ScenePlan,
    *,
    order: int | None = None,
    raw: dict | None = None,
    text_overrides: dict[str, str] | None = None,
) -> SceneRenderSpec:
    """Translate one scene into a generic, renderable scene spec."""
    raw = dict(raw or {})
    text_overrides = dict(text_overrides or {})
    duration = float(scene.duration_seconds)
    primary, secondary = _required_visual_assets(scene, plan)

    elements: list[RenderElement] = [
        RenderElement(
            id="main_visual",
            type="image",
            asset=primary,
            layer=LAYER_MAIN_VISUAL,
            transform=Transform(
                x=MAIN_VISUAL_POSITION["x"],
                y=MAIN_VISUAL_POSITION["y"],
                scale=MAIN_VISUAL_POSITION["scale"],
                fit="cover",
            ),
            animations=camera_animations(scene.camera, duration),
            timing=ElementTiming(start=0.0, duration=0.0),
            metadata={"origin": "scene.required_assets"},
        )
    ]

    for index, ref in enumerate(secondary, start=1):
        elements.append(
            RenderElement(
                id=f"asset_{index}_{_slug(ref)}",
                type="image",
                asset=ref,
                layer=LAYER_SECONDARY,
                transform=Transform(
                    x=SECONDARY_POSITION["x"],
                    y=SECONDARY_POSITION["y"],
                    scale=SECONDARY_POSITION["scale"],
                    fit="contain",
                ),
                animations=camera_animations(scene.camera, duration),
                timing=ElementTiming(start=0.0, duration=0.0),
                metadata={"origin": "scene.required_assets"},
            )
        )

    if scene.mascot.present:
        mascot_ref = scene.mascot.asset or plan.mascot_asset_path or MASCOT_ASSET_PATH
        elements.append(
            RenderElement(
                id="mascot",
                type="mascot",
                asset=mascot_ref,
                layer=LAYER_MASCOT,
                transform=Transform(
                    x=MASCOT_POSITION["x"],
                    y=MASCOT_POSITION["y"],
                    scale=MASCOT_POSITION["scale"],
                    fit="contain",
                ),
                animations=mascot_idle_animations(duration),
                timing=ElementTiming(start=0.0, duration=0.0),
                metadata={
                    "origin": "scene.mascot",
                    "action": scene.mascot.action,
                    "speaks": False,
                    "animation_seam": "future character animation reads metadata.action",
                },
            )
        )

    text_value = _text_for_scene(scene, raw, text_overrides)
    if text_value:
        elements.append(
            RenderElement(
                id="text_main",
                type="text",
                text=text_value,
                layer=LAYER_TEXT,
                transform=Transform(
                    x=TEXT_POSITION["x"],
                    y=TEXT_POSITION["y"],
                    scale=TEXT_POSITION["scale"],
                ),
                animations=[
                    Animation(
                        type="fade",
                        start=0.0,
                        duration=max(0.2, min(0.6, duration * 0.3)),
                        from_value=0.0,
                        to_value=1.0,
                    )
                ],
                timing=ElementTiming(start=0.0, duration=0.0),
                style=dict(TEXT_STYLE_DEFAULTS),
                metadata={"origin": "scene text field"},
            )
        )

    metadata = {
        "narration": scene.narration,
        "visual_description": scene.visual_description,
        "camera": scene.camera,
        "composition": scene.composition,
        "environment": scene.environment,
        "generation_prompt": scene.generation_prompt,
        "transition_hint": scene.transition_hint,
        "required_assets": list(scene.required_assets),
        "mascot_action": scene.mascot.action,
        "mascot_asset": scene.mascot.asset,
    }

    return SceneRenderSpec(
        scene_id=scene.scene_id,
        order=int(scene.order if order is None else order),
        duration_seconds=duration,
        background=DEFAULT_BACKGROUND,
        elements=elements,
        metadata=metadata,
    )


def _raw_scene_index(raw_plan: dict) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for position, item in enumerate(raw_plan.get("scenes") or []):
        if not isinstance(item, dict):
            continue
        scene_id = str(item.get("scene_id") or f"sc{position + 1}")
        index[scene_id] = item
    return index


def translate_scene_plan(
    plan: ScenePlan | dict,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: int = DEFAULT_FPS,
    run_id: str | None = None,
    background: tuple[int, int, int] = DEFAULT_BACKGROUND,
    text_overrides: dict[str, str] | None = None,
) -> RenderSpec:
    """Translate a full ``ScenePlan`` (or its raw dict) into a ``RenderSpec``.

    Deterministic: the same input always produces byte-identical JSON.
    """
    if isinstance(plan, dict):
        raw_plan = plan
        plan_obj = ScenePlan.from_dict(plan)
    else:
        plan_obj = plan
        raw_plan = plan.to_dict()

    raw_scenes = _raw_scene_index(raw_plan)
    entries: list[tuple[int, str, Scene]] = []
    for scene in plan_obj.scenes:
        raw = raw_scenes.get(scene.scene_id, {})
        order = int(raw.get("order", scene.order))
        entries.append((order, scene.scene_id, scene))
    entries.sort(key=lambda item: (item[0], item[1]))

    scenes = [
        translate_scene(
            scene,
            plan_obj,
            order=order,
            raw=raw_scenes.get(scene.scene_id, {}),
            text_overrides=text_overrides,
        )
        for order, _scene_id, scene in entries
    ]

    spec = RenderSpec(
        run_id=run_id if run_id is not None else plan_obj.run_id,
        width=int(width),
        height=int(height),
        fps=int(fps),
        background=tuple(background),
        scenes=scenes,
        metadata={
            "translator": TRANSLATOR_NAME,
            "translator_version": TRANSLATOR_VERSION,
            "topic_id": plan_obj.topic_id,
            "title": plan_obj.title,
            "channel_id": plan_obj.channel_id,
            "mascot_asset_path": plan_obj.mascot_asset_path,
            "target_duration_seconds": plan_obj.target_duration_seconds,
            "scene_plan_duration_seconds": plan_obj.total_scene_duration_seconds
            or plan_obj.estimated_seconds,
        },
    )
    return spec.validate()


def collect_asset_ids(spec: RenderSpec) -> list[str]:
    """Unique asset ids referenced by the spec, in scene order."""
    seen: list[str] = []
    for scene in spec.ordered_scenes:
        for element in scene.elements:
            if element.type in ("image", "mascot") and element.asset:
                if element.asset not in seen:
                    seen.append(element.asset)
    return seen


def collect_scene_asset_ids(spec: RenderSpec) -> list[str]:
    """Bare scene asset ids only - never channel assets like the mascot."""
    return [
        asset_id
        for asset_id in collect_asset_ids(spec)
        if "/" not in asset_id and "\\" not in asset_id
    ]
