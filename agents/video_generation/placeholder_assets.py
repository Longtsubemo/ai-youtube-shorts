"""Deterministic local test assets and in-memory placeholders.

Two distinct jobs:

* **Placeholders** stand in for an asset that is missing at render time. They
  are generated in memory, clearly labelled, and reported through the asset
  manifest - never written into ``data/<run_id>/assets/`` so nothing ever
  masquerades as a generated production image.
* **Synthetic test assets** give the renderer real files to read when no image
  provider exists yet (unit tests, the first smoke render). They are written
  only when explicitly requested and are recorded with the provenance
  ``deterministic_test_asset``.

Both are fully deterministic: the same asset id always yields the same pixels.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageDraw

from agents.video_generation.fonts import load_font


def asset_digest(asset_id: str) -> bytes:
    """Stable digest used to derive colors and shapes from an asset id."""
    return hashlib.sha256(str(asset_id).encode("utf-8")).digest()


def placeholder_color(asset_id: str) -> tuple[int, int, int]:
    """Muted deterministic color for a placeholder."""
    digest = asset_digest(asset_id)
    return (40 + digest[0] // 3, 44 + digest[1] // 3, 56 + digest[2] // 3)


def _truncate(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> str:
    if draw.textlength(text, font=font) <= max_width:
        return text
    while text and draw.textlength(text + "...", font=font) > max_width:
        text = text[:-1]
    return text + "..." if text else ""


def placeholder_image(
    asset_id: str,
    size: tuple[int, int] = (720, 1280),
    *,
    kind: str = "scene",
) -> Image.Image:
    """An obvious, deterministic stand-in for a missing asset (RGBA)."""
    width, height = size
    base = placeholder_color(asset_id)
    stripe = tuple(min(255, c + 26) for c in base)
    img = Image.new("RGBA", (width, height), base + (255,))
    draw = ImageDraw.Draw(img)

    period = max(24, width // 8)
    for offset in range(-(height // 2), width + height, period):
        draw.line(
            [(offset, 0), (offset + height, height)],
            fill=stripe + (255,),
            width=max(2, period // 6),
        )

    border = max(4, width // 90)
    draw.rectangle(
        [border, border, width - border - 1, height - border - 1],
        outline=(235, 235, 240, 255),
        width=border,
    )

    label_font = load_font(max(14, width // 22))
    caption_font = load_font(max(12, width // 30))
    label = _truncate(draw, str(asset_id), label_font, int(width * 0.86))
    caption = f"missing {kind} placeholder"
    lw = draw.textlength(label, font=label_font)
    cw = draw.textlength(caption, font=caption_font)
    cx, cy = width / 2, height / 2
    draw.text((cx - cw / 2, cy - 4), caption, font=caption_font, fill=(245, 245, 250, 255))
    draw.text((cx - lw / 2, cy + 26), label, font=label_font, fill=(245, 245, 250, 255))
    return img


def synthetic_scene_image(
    asset_id: str,
    size: tuple[int, int] = (1080, 1920),
) -> Image.Image:
    """Deterministic local test art standing in for a generated scene image.

    Purely geometric: a vertical gradient, a horizon band and a few shapes
    whose parameters all derive from the asset id. It contains no text that
    could be mistaken for narration content.
    """
    width, height = size
    digest = asset_digest(asset_id)
    top = (18 + digest[0] // 5, 24 + digest[1] // 6, 46 + digest[2] // 4)
    bottom = (12 + digest[3] // 6, 58 + digest[4] // 4, 70 + digest[5] // 5)

    img = Image.new("RGBA", (width, height), top + (255,))
    draw = ImageDraw.Draw(img)
    for step in range(height):
        p = step / max(1, height - 1)
        color = tuple(int(top[i] + (bottom[i] - top[i]) * p) for i in range(3))
        draw.line([(0, step), (width, step)], fill=color + (255,))

    horizon = int(height * (0.55 + (digest[6] % 10) / 100))
    band = tuple(min(255, c + 40) for c in bottom)
    draw.rectangle([0, horizon, width, horizon + max(6, height // 60)], fill=band + (255,))

    shape_colors = [
        (200 + digest[7] % 55, 160 + digest[8] % 80, 60 + digest[9] % 120),
        (70 + digest[10] % 120, 190 + digest[11] % 60, 200 + digest[12] % 55),
        (235, 235, 240),
    ]
    for idx in range(3):
        cx = int(width * (0.2 + 0.3 * idx) + (digest[13 + idx] % 40) - 20)
        cy = int(horizon - height * (0.06 + 0.05 * (digest[16 + idx] % 5)))
        radius = max(8, int(width * (0.05 + 0.02 * (digest[19 + idx] % 4))))
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=shape_colors[idx] + (255,),
        )
    return img


def synthesize_scene_assets(
    run_dir: str | Path,
    asset_ids: list[str],
    *,
    size: tuple[int, int] = (1080, 1920),
    force: bool = False,
) -> list[Path]:
    """Write deterministic test assets for missing scene asset ids.

    Only bare scene ids (no path separators) are written, and only into
    ``<run_dir>/assets/``. Channel assets - the mascot above all - are never
    written, copied or renamed here.
    """
    run_dir = Path(run_dir)
    assets_dir = run_dir / "assets"
    written: list[Path] = []
    for asset_id in asset_ids:
        if not asset_id or "/" in asset_id or "\\" in asset_id:
            continue
        out_path = assets_dir / f"{asset_id}.png"
        if out_path.exists() and not force:
            continue
        assets_dir.mkdir(parents=True, exist_ok=True)
        synthetic_scene_image(asset_id, size).save(out_path, format="PNG")
        written.append(out_path)
    return written
