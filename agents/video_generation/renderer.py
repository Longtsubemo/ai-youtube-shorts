"""The generic scene renderer.

Composes one frame at a time from a ``SceneRenderSpec``:

    background -> elements ordered by layer -> RGB frame

The renderer only understands generic concepts: image, mascot, text,
position, scale, rotation, opacity, zoom, pan, slide, fade, rotate. It never
branches on scene ids, topics or visual wording - a scene it cannot fully
render falls back to the nearest generic behaviour (available imagery,
camera motion, duration, neutral background) and the deviation is recorded in
``RenderNotes`` instead of being silently claimed as success.

Mascot guarantee: the mascot asset is resized, moved, rotated and alpha-blended
only. No color transform, filter or channel manipulation is ever applied, so
the character's original appearance survives rendering.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw

from agents.video_generation.assets import AssetResolver, ResolvedAsset
from agents.video_generation.encoder import (
    FrameWriter,
    open_frame_writer,
    probe_video,
)
from agents.video_generation.fonts import load_font
from agents.video_generation.motion import (
    collect_unsupported_animations,
    compute_state,
    element_visible,
)
from agents.video_generation.placeholder_assets import placeholder_image
from agents.video_generation.render_spec import (
    ELEMENT_TYPES,
    RenderElement,
    SceneRenderSpec,
    Transform,
)

MAX_TARGET_DIMENSION_FACTOR = 4


class RenderError(RuntimeError):
    """Raised when a scene clip could not be rendered into a playable MP4."""


@dataclass
class RenderNotes:
    """Honest record of everything the renderer could not do as specified."""

    missing_assets: list[dict] = field(default_factory=list)
    unsupported_animations: list[dict] = field(default_factory=list)
    skipped_elements: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "missing_assets": list(self.missing_assets),
            "unsupported_animations": list(self.unsupported_animations),
            "skipped_elements": list(self.skipped_elements),
        }

    @property
    def has_notes(self) -> bool:
        return bool(self.missing_assets or self.unsupported_animations or self.skipped_elements)


def parse_color(value, default: tuple[int, int, int] = (255, 255, 255)) -> tuple[int, int, int]:
    """Parse ``#RRGGBB`` or ``[r, g, b]`` into an RGB triple."""
    if isinstance(value, str):
        try:
            return tuple(ImageColor.getrgb(value))[:3]  # type: ignore[return-value]
        except ValueError:
            return default
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(int(max(0, min(255, int(c)))) for c in value)  # type: ignore[return-value]
    return default


class SceneRenderer:
    """Deterministic, data-driven renderer for a single output resolution."""

    def __init__(
        self,
        resolver: AssetResolver,
        *,
        width: int,
        height: int,
        fps: int,
        notes: RenderNotes | None = None,
        asset_manifest: dict[str, dict] | None = None,
    ) -> None:
        self.resolver = resolver
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.notes = notes if notes is not None else RenderNotes()
        self.asset_manifest = asset_manifest if asset_manifest is not None else {}
        self._images: dict[tuple[str, str], tuple[Image.Image, ResolvedAsset]] = {}
        self._layer_cache: OrderedDict[tuple, Image.Image] = OrderedDict()
        self._recorded_animations: set[tuple[str, str]] = set()

    # ------------------------------------------------------------------ assets

    def _record_asset(self, resolved: ResolvedAsset, element: RenderElement) -> None:
        entry = self.asset_manifest.get(resolved.asset_id)
        if entry is None:
            entry = resolved.to_dict()
            entry["elements"] = [element.id]
            entry["used_in_scenes"] = [resolved.scene_id] if resolved.scene_id else []
            self.asset_manifest[resolved.asset_id] = entry
            return
        if element.id not in entry["elements"]:
            entry["elements"].append(element.id)
        if resolved.scene_id and resolved.scene_id not in entry["used_in_scenes"]:
            entry["used_in_scenes"].append(resolved.scene_id)

    def _element_image(self, element: RenderElement, scene: SceneRenderSpec) -> Image.Image:
        """Load (or substitute) the image for an element, once per asset."""
        key = (element.type, element.asset)
        if key not in self._images:
            resolved = self.resolver.resolve(element.asset, scene_id=scene.scene_id)
            image = self.resolver.open_image(resolved)
            if image is None:
                kind = "mascot" if element.type == "mascot" else "scene"
                image = placeholder_image(element.asset, (self.width, self.height), kind=kind)
                self._record_missing(resolved, element, scene)
            self._images[key] = (image, resolved)
        image, resolved = self._images[key]
        self._record_asset(resolved, element)
        return image

    def _record_missing(
        self, resolved: ResolvedAsset, element: RenderElement, scene: SceneRenderSpec
    ) -> None:
        for entry in self.notes.missing_assets:
            if entry["asset_id"] == resolved.asset_id and scene.scene_id in entry["scenes"]:
                return
        self.notes.missing_assets.append(
            {
                "asset_id": resolved.asset_id,
                "kind": resolved.kind.value,
                "expected_path": str(resolved.path),
                "scenes": [scene.scene_id],
                "elements": [element.id],
                "substituted": "in-memory deterministic placeholder",
            }
        )

    # ------------------------------------------------------------------ layers

    @staticmethod
    def _fit_scale(image: Image.Image, width: int, height: int, fit: str) -> float:
        iw, ih = image.size
        if iw <= 0 or ih <= 0:
            return 1.0
        contain = min(width / iw, height / ih)
        cover = max(width / iw, height / ih)
        return cover if fit == "cover" else contain

    def _target_size(
        self, image: Image.Image, fit: str, scale_multiplier: float, transform: Transform
    ) -> tuple[int, int]:
        iw, ih = image.size
        factor = self._fit_scale(image, self.width, self.height, fit) * transform.scale * scale_multiplier
        max_dim = MAX_TARGET_DIMENSION_FACTOR * max(self.width, self.height)
        target_w = max(1, min(max_dim, int(round(iw * factor))))
        target_h = max(1, min(max_dim, int(round(ih * factor))))
        return target_w, target_h

    @staticmethod
    def _resize(image: Image.Image, size: tuple[int, int]) -> Image.Image:
        if image.size == size:
            return image
        shrinking = size[0] < image.size[0] and size[1] < image.size[1]
        resample = Image.Resampling.LANCZOS if shrinking else Image.Resampling.BICUBIC
        return image.resize(size, resample=resample)

    @staticmethod
    def _apply_opacity(layer: Image.Image, opacity: float) -> Image.Image:
        if opacity >= 0.999:
            return layer
        alpha = layer.getchannel("A")
        lut = [int(value * opacity) for value in range(256)]
        alpha = alpha.point(lut)
        layer = layer.copy()
        layer.putalpha(alpha)
        return layer

    def _place(
        self,
        canvas: Image.Image,
        layer: Image.Image,
        state_x: float,
        state_y: float,
        opacity: float,
    ) -> None:
        layer = self._apply_opacity(layer, opacity)
        if layer.getchannel("A").getextrema()[1] == 0:
            return
        center_x = state_x * self.width
        center_y = state_y * self.height
        left = int(round(center_x - layer.size[0] / 2))
        top = int(round(center_y - layer.size[1] / 2))
        canvas.paste(layer, (left, top), layer)

    def _cached_layer(self, key: tuple, builder) -> Image.Image:
        """Tiny LRU for resized/rotated layers (sizes repeat frame to frame)."""
        hit = self._layer_cache.get(key)
        if hit is not None:
            self._layer_cache.move_to_end(key)
            return hit
        layer = builder()
        self._layer_cache[key] = layer
        if len(self._layer_cache) > 16:
            self._layer_cache.popitem(last=False)
        return layer

    @staticmethod
    def _rotate_layer(layer: Image.Image, rotation: float) -> Image.Image:
        if abs(rotation) <= 0.01:
            return layer
        # Pillow rotates counter-clockwise for positive angles; our convention
        # is clockwise degrees, so the angle is negated.
        return layer.rotate(-rotation, resample=Image.Resampling.BICUBIC, expand=True)

    def _image_layer(
        self, element: RenderElement, scene: SceneRenderSpec, state
    ) -> Image.Image:
        base = self._element_image(element, scene)
        target = self._target_size(base, element.transform.fit, state.scale, element.transform)
        rotation = round(float(state.rotation), 2)
        key = ("image", element.type, element.asset, target, rotation)
        return self._cached_layer(
            key,
            lambda: self._rotate_layer(self._resize(base, target), float(rotation)),
        )

    def _text_layer(self, element: RenderElement, state) -> Image.Image:
        style = dict(element.style or {})
        size_fraction = float(style.get("size", 0.05))
        pixel_size = max(8, int(round(size_fraction * self.height * state.scale)))
        color = parse_color(style.get("color", "#FFFFFF"))
        font_path = str(style.get("font", ""))
        rotation = round(float(state.rotation), 2)
        key = ("text", element.text, pixel_size, color, font_path, rotation)

        def build() -> Image.Image:
            layer = self._render_text(element.text, pixel_size, color, font_path)
            return self._rotate_layer(layer, float(rotation))

        return self._cached_layer(key, build)

    @staticmethod
    def _render_text(
        text: str, pixel_size: int, color: tuple[int, int, int], font_path: str
    ) -> Image.Image:
        font = load_font(pixel_size, font_path)
        lines = str(text).split("\n")
        widths = []
        heights = []
        measure = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        for line in lines:
            bbox = measure.textbbox((0, 0), line or " ", font=font)
            widths.append(bbox[2] - bbox[0])
            heights.append(bbox[3] - bbox[1])
        line_gap = max(4, pixel_size // 4)
        width = max(widths) if widths else 1
        height = (sum(heights) + line_gap * (len(lines) - 1)) if heights else 1
        layer = Image.new("RGBA", (max(1, width) + 4, max(1, height) + 4), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        cursor = 2
        for line, line_height, line_width in zip(lines, heights, widths):
            draw.text(
                (2 + (width - line_width) / 2, cursor),
                line,
                font=font,
                fill=color + (255,),
            )
            cursor += line_height + line_gap
        return layer

    def _layer_for(self, element: RenderElement, scene: SceneRenderSpec, state) -> Image.Image:
        if element.type in ("image", "mascot"):
            return self._image_layer(element, scene, state)
        if element.type == "text":
            return self._text_layer(element, state)
        raise RenderError(
            f"unsupported element type {element.type!r} (element {element.id!r})"
        )

    # ------------------------------------------------------------------ frames

    def compose_frame(self, scene: SceneRenderSpec, t: float) -> Image.Image:
        """Compose one RGB frame of ``scene`` at time ``t`` (seconds)."""
        background = tuple(int(c) for c in scene.background)
        canvas = Image.new("RGBA", (self.width, self.height), background + (255,))

        ordered = sorted(scene.elements, key=lambda e: (e.layer, e.id))
        for element in ordered:
            unsupported = collect_unsupported_animations(element)
            if unsupported:
                marker = (scene.scene_id, element.id)
                if marker not in self._recorded_animations:
                    self._recorded_animations.add(marker)
                    self.notes.unsupported_animations.append(
                        {
                            "scene_id": scene.scene_id,
                            "element_id": element.id,
                            "types": unsupported,
                            "handled": "ignored; remaining motion and duration preserved",
                        }
                    )
            if element.type not in ELEMENT_TYPES:
                marker = (scene.scene_id, element.id)
                if marker not in self._recorded_animations:
                    self._recorded_animations.add(marker)
                    self.notes.skipped_elements.append(
                        {
                            "scene_id": scene.scene_id,
                            "element_id": element.id,
                            "type": element.type,
                            "reason": "unknown element type",
                        }
                    )
                continue
            if not element_visible(element, t, scene.duration_seconds):
                continue
            state = compute_state(element, t)
            if state.opacity <= 0.002:
                continue
            try:
                layer = self._layer_for(element, scene, state)
            except RenderError as exc:
                self.notes.skipped_elements.append(
                    {
                        "scene_id": scene.scene_id,
                        "element_id": element.id,
                        "type": element.type,
                        "reason": str(exc),
                    }
                )
                continue
            self._place(canvas, layer, state.x, state.y, state.opacity)
        return canvas.convert("RGB")

    def render_scene(
        self,
        scene: SceneRenderSpec,
        out_path: str | Path,
        *,
        backend_info: dict,
    ) -> dict:
        """Render every frame of one scene into an MP4, then probe the file."""
        out_path = Path(out_path)
        frame_count = max(1, int(round(scene.duration_seconds * self.fps)))
        writer: FrameWriter = open_frame_writer(
            out_path, self.width, self.height, self.fps, backend_info=backend_info
        )
        write_error: Exception | None = None
        for index in range(frame_count):
            t = index / self.fps
            try:
                frame = self.compose_frame(scene, t)
                writer.write(np.asarray(frame))
            except Exception as exc:  # noqa: BLE001 - converted to RenderError below
                write_error = exc
                break
        try:
            encode_info = writer.close()
        except Exception as exc:  # noqa: BLE001
            out_path.unlink(missing_ok=True)
            if write_error is None:
                raise RenderError(
                    f"scene {scene.scene_id}: encoding failed: {exc}"
                ) from exc
            raise RenderError(
                f"scene {scene.scene_id}: frame {index} failed: {write_error}"
            ) from exc
        if write_error is not None:
            out_path.unlink(missing_ok=True)
            raise RenderError(
                f"scene {scene.scene_id}: frame {index} failed: {write_error}"
            ) from write_error

        probe = probe_video(out_path)
        minimum = max(1, int(frame_count * 0.9))
        if not probe["readable"] or probe["frames"] < minimum:
            out_path.unlink(missing_ok=True)
            raise RenderError(
                f"scene {scene.scene_id}: encoded file failed playback probe "
                f"({probe['frames']} frames, expected about {frame_count})"
            )
        duration = frame_count / self.fps
        return {
            "scene_id": scene.scene_id,
            "path": str(out_path),
            "frames": frame_count,
            "duration_seconds": duration,
            "probe": probe,
            "encode": encode_info,
        }

