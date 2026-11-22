"""Video generation providers for Stage 5.

    VideoGenerator
        ├── LocalMotionRenderer   (this task: deterministic local 2D renderer)
        └── AIVideoRenderer       (future seam - not implemented here)

The stage contract is stable: hand over a ``RenderSpec``, a run directory, an
asset resolver and an encoding backend, get back a ``VideoResult`` describing
what was actually written. A future AI video provider plugs in behind the same
interface without changing the stage, the CLI or the artifacts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from agents.video_generation.assets import AssetResolver
from agents.video_generation.encoder import (
    VideoEncodingError,
    concat_videos,
    probe_video,
)
from agents.video_generation.models import SceneClip, VideoResult
from agents.video_generation.render_spec import RenderSpec
from agents.video_generation.renderer import RenderError, SceneRenderer


class VideoGeneratorError(RuntimeError):
    """Raised when a provider cannot produce the requested video."""


class VideoGenerator(ABC):
    """Provider contract for turning a render spec into video files."""

    name: str = "abstract"
    renderer_name: str = "abstract"
    renderer_version: str = "0.0.0"

    @abstractmethod
    def generate(
        self,
        spec: RenderSpec,
        *,
        run_dir: str | Path,
        resolver: AssetResolver,
        backend_info: dict,
    ) -> VideoResult:
        """Render every scene, assemble the final clip and return results."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r} version={self.renderer_version}>"


class LocalMotionRenderer(VideoGenerator):
    """Deterministic Pillow/OpenCV/FFmpeg renderer driven purely by data."""

    name = "local"
    renderer_name = "local-motion-renderer"
    renderer_version = "1.0.0"

    def generate(
        self,
        spec: RenderSpec,
        *,
        run_dir: str | Path,
        resolver: AssetResolver,
        backend_info: dict,
    ) -> VideoResult:
        spec.validate()
        run_dir = Path(run_dir)
        video_dir = run_dir / "video"
        video_dir.mkdir(parents=True, exist_ok=True)

        scene_renderer = SceneRenderer(
            resolver,
            width=spec.width,
            height=spec.height,
            fps=spec.fps,
        )

        clips: list[SceneClip] = []
        for scene in spec.ordered_scenes:
            out_path = video_dir / f"{scene.scene_id}.mp4"
            try:
                rendered = scene_renderer.render_scene(
                    scene, out_path, backend_info=backend_info
                )
            except RenderError as exc:
                raise VideoGeneratorError(str(exc)) from exc
            clips.append(SceneClip.from_render(rendered))

        if not clips:
            raise VideoGeneratorError("render spec produced no scene clips")

        final_path = video_dir / "final.mp4"
        try:
            assembly = concat_videos(
                [clip.path for clip in clips],
                final_path,
                width=spec.width,
                height=spec.height,
                fps=spec.fps,
                backend_info=backend_info,
            )
        except VideoEncodingError as exc:
            raise VideoGeneratorError(
                f"scene assembly failed: {exc}"
            ) from exc

        final_probe = probe_video(final_path)
        expected_frames = sum(clip.frames for clip in clips)
        minimum = max(1, int(expected_frames * 0.9))
        if not final_probe["valid"] or final_probe["frames"] < minimum:
            raise VideoGeneratorError(
                "assembled video failed the playback probe "
                f"({final_probe['frames']} frames, expected about {expected_frames}): "
                f"{final_path}"
            )
        assembly.update({"probe": final_probe, "expected_frames": expected_frames})

        manifest = sorted(
            scene_renderer.asset_manifest.values(), key=lambda entry: entry["asset_id"]
        )
        return VideoResult(
            run_id=spec.run_id,
            provider=self.name,
            renderer=self.renderer_name,
            renderer_version=self.renderer_version,
            width=spec.width,
            height=spec.height,
            fps=spec.fps,
            duration_seconds=float(sum(clip.duration_seconds for clip in clips)),
            scene_clips=clips,
            final_video=str(final_path),
            encoder=dict(backend_info),
            assembly=assembly,
            asset_manifest=manifest,
            notes=scene_renderer.notes.to_dict(),
        )


PROVIDERS: dict[str, type[VideoGenerator]] = {
    LocalMotionRenderer.name: LocalMotionRenderer,
}


def available_providers() -> list[str]:
    return sorted(PROVIDERS)


def build_video_generator(provider_name: str) -> VideoGenerator:
    """Instantiate a video provider by name."""
    key = str(provider_name or "").strip().lower()
    provider_cls = PROVIDERS.get(key)
    if provider_cls is None:
        raise VideoGeneratorError(
            f"unknown video provider {provider_name!r}; available: "
            f"{', '.join(available_providers())}"
        )
    return provider_cls()
