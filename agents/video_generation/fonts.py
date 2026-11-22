"""Font lookup shared by the renderer and the deterministic test assets.

Uses a real TrueType font when the platform provides one and falls back to
Pillow's bundled scalable default, so text rendering never requires a network
download or an extra dependency.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

FONT_CANDIDATES = (
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
)


@lru_cache(maxsize=256)
def load_font(size: int, font_path: str = "") -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a font rasterized at ``size`` pixels."""
    size = max(1, int(size))
    if font_path:
        candidate = Path(font_path)
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:
                pass
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - very old Pillow
        return ImageFont.load_default()


def font_fingerprint(size: int, font_path: str = "") -> str:
    """Stable cache key describing which font produced a text layer."""
    font = load_font(size, font_path)
    return f"{getattr(font, 'path', 'default')}:{size}"
