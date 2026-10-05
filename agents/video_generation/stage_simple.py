"""Simplified video generation (Stage 5 artifact writer)."""

from __future__ import annotations

from pathlib import Path

PLACEHOLDER_BYTES = b"video-placeholder"


def write_video_placeholder(out_path: str | Path) -> Path:
    """Write a non-playable placeholder file for the Stage 5 video artifact.

    No renderer exists yet. This exists only so the five-artifact chain can be
    created and inspected end to end.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(PLACEHOLDER_BYTES)
    return out_path