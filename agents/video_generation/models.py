"""Stage 5 result objects.

Small, serializable containers handed from the provider back to the stage so
``video.json`` is written from real measurements, not from expectations.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SceneClip:
    """One rendered scene MP4."""

    scene_id: str
    path: str
    frames: int
    duration_seconds: float
    probe: dict = field(default_factory=dict)

    @property
    def playable(self) -> bool:
        return bool(self.probe.get("valid"))

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "path": self.path,
            "frames": int(self.frames),
            "duration_seconds": float(self.duration_seconds),
            "probe": dict(self.probe),
            "playable": self.playable,
        }

    @classmethod
    def from_render(cls, data: dict) -> "SceneClip":
        return cls(
            scene_id=str(data["scene_id"]),
            path=str(data["path"]),
            frames=int(data["frames"]),
            duration_seconds=float(data["duration_seconds"]),
            probe=dict(data.get("probe") or {}),
        )


@dataclass
class VideoResult:
    """Everything Stage 5 produced for one run."""

    run_id: str
    provider: str
    renderer: str
    renderer_version: str
    width: int
    height: int
    fps: int
    duration_seconds: float
    scene_clips: list[SceneClip] = field(default_factory=list)
    final_video: str = ""
    video_path: str = ""
    metadata_path: str = ""
    render_spec_path: str = ""
    encoder: dict = field(default_factory=dict)
    assembly: dict = field(default_factory=dict)
    asset_manifest: list[dict] = field(default_factory=list)
    notes: dict = field(default_factory=dict)
    translator_version: str = ""

    @property
    def scene_count(self) -> int:
        return len(self.scene_clips)

    @property
    def scene_ids(self) -> list[str]:
        return [clip.scene_id for clip in self.scene_clips]

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "stage": "video_generation",
            "provider": self.provider,
            "renderer": self.renderer,
            "renderer_version": self.renderer_version,
            "translator_version": self.translator_version,
            "scene_count": self.scene_count,
            "scene_clips": [clip.to_dict() for clip in self.scene_clips],
            "final_video": self.final_video,
            "video_path": self.video_path,
            "metadata_path": self.metadata_path,
            "render_spec_path": self.render_spec_path,
            "duration_seconds": float(self.duration_seconds),
            "resolution": {"width": int(self.width), "height": int(self.height)},
            "fps": int(self.fps),
            "encoder": dict(self.encoder),
            "assembly": dict(self.assembly),
            "asset_manifest": list(self.asset_manifest),
            "notes": dict(self.notes),
        }

    @staticmethod
    def as_metadata(result: "VideoResult") -> dict:
        return result.to_dict()
