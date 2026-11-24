"""Stage 5: Video Generation.

Deterministic local stage that turns the Stage 4 ``scene_plan.json`` into real
MP4 files:

    data/<run_id>/scene_plan.json
            -> translate (provider independent)
            -> render specification
            -> generic scene renderer
            -> data/<run_id>/video/<scene_id>.mp4 ... -> video/final.mp4
            -> data/<run_id>/video/final.mp4
            -> data/<run_id>/video.mp4
            -> data/<run_id>/video.json
            -> state.json = VIDEO_GENERATION

With ``narration_voice`` set, Kokoro TTS (see ``narration.py``) narrates each
scene first, scene lengths follow the audio, and the track is muxed into the MP4.

No network access, no hosted model calls, no image-generation API, no scoring and no output gating.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from agents.scene_planner.scene_models import ScenePlan, ScenePlanningError
from agents.video_generation.assets import AssetResolver
from agents.video_generation.encoder import VideoEncodingError, detect_ffmpeg, resolve_backend
from agents.video_generation import narration as tts
from agents.video_generation.models import VideoResult
from agents.video_generation.placeholder_assets import synthesize_scene_assets
from agents.video_generation.provider import (
    VideoGenerator,
    VideoGeneratorError,
    build_video_generator,
)
from agents.video_generation.render_spec import (
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    RenderSpec,
)
from agents.video_generation.translator import (
    MASCOT_LAYOUTS,
    TRANSLATOR_VERSION,
    collect_scene_asset_ids,
    translate_scene_plan,
)
from pipeline.state import PipelineState, Stage

RENDER_SPEC_FILENAME = "render_spec.json"
VIDEO_METADATA_FILENAME = "video.json"
FINAL_VIDEO_NAME = "final.mp4"


class VideoGenerationError(RuntimeError):
    """Raised for input problems the stage refuses to render."""


def load_scene_plan(path: str | Path) -> ScenePlan:
    """Load and structurally validate the Stage 4 artifact."""
    path = Path(path)
    if not path.exists():
        raise VideoGenerationError(f"missing Stage 4 artifact: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VideoGenerationError(f"scene plan is not valid JSON: {path} ({exc})") from exc
    if not isinstance(data, dict):
        raise VideoGenerationError(f"scene plan must be a JSON object: {path}")
    try:
        plan = ScenePlan.from_dict(data)
        plan.validate()
    except (ScenePlanningError, ValueError, TypeError, KeyError) as exc:
        raise VideoGenerationError(f"invalid scene plan: {exc}") from exc

    seen: set[str] = set()
    for scene in plan.scenes:
        if scene.scene_id in seen:
            raise VideoGenerationError(f"scene plan has duplicate scene_id {scene.scene_id!r}")
        seen.add(scene.scene_id)
    return plan


def write_render_spec(spec: RenderSpec, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(spec.to_json(), encoding="utf-8")
    return path


def write_video_metadata(result: VideoResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return path


def update_state(run_dir: str | Path, run_id: str, result: VideoResult) -> Path:
    """Record Stage 5 completion in ``state.json``."""
    run_dir = Path(run_dir)
    state_path = run_dir / "state.json"
    state = PipelineState.load_or_create(state_path, run_id)
    state.advance_to(Stage.VIDEO_GENERATION)
    state.add_artifact(Stage.VIDEO_GENERATION, result.final_video)
    if result.video_path:
        state.add_artifact(Stage.VIDEO_GENERATION, result.video_path)
    if result.metadata_path:
        state.add_artifact(Stage.VIDEO_GENERATION, result.metadata_path)
    if result.render_spec_path:
        state.add_artifact(Stage.VIDEO_GENERATION, result.render_spec_path)
    return state.save(state_path)


def pick_variant(run_id: str) -> dict:
    """Per-video look (mascot side, caption colour), stable for a run id."""
    # shortcut: hash-based, so consecutive runs can repeat a look; rotate from run history if
    # that becomes visible on the channel.
    n = int.from_bytes(hashlib.sha256(run_id.encode("utf-8")).digest()[:4], "big")
    layouts, colours = sorted(MASCOT_LAYOUTS), sorted(tts.CAPTION_HIGHLIGHTS)
    return {"layout": layouts[n % len(layouts)],
            "caption_colour": colours[(n // len(layouts)) % len(colours)]}


def _even_dimensions(width: int, height: int) -> tuple[int, int, str | None]:
    """MP4 pixel formats need even dimensions; normalize and say so."""
    even_w = max(2, int(width) - int(width) % 2)
    even_h = max(2, int(height) - int(height) % 2)
    if (even_w, even_h) == (int(width), int(height)):
        return int(width), int(height), None
    return even_w, even_h, f"dimensions adjusted to even {even_w}x{even_h} for MP4 encoding"


def run_video_generation(
    plan: ScenePlan,
    *,
    run_dir: str | Path,
    run_id: str | None = None,
    provider: str = "local",
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: int = DEFAULT_FPS,
    encoder: str = "auto",
    synthesize_test_assets: bool = False,
    text_overrides: dict[str, str] | None = None,
    narration_voice: str | None = None,
    narration_speed: float = tts.DEFAULT_SPEED,
    tts_python: str | Path | None = None,
    captions: bool = True,
) -> VideoResult:
    """Translate, render, assemble and write every Stage 5 artifact."""
    run_dir = Path(run_dir)
    if int(width) <= 0 or int(height) <= 0 or int(fps) <= 0:
        raise VideoGenerationError(
            f"width/height/fps must be positive, got {width}x{height} @ {fps}"
        )
    plan.validate()
    resolved_run_id = run_id or plan.run_id or run_dir.name

    width, height, dimension_note = _even_dimensions(int(width), int(height))
    fps = int(fps)
    variant = pick_variant(resolved_run_id)

    narration_info = None
    if narration_voice:
        ffmpeg = detect_ffmpeg()
        if not ffmpeg:
            raise tts.NarrationError("narration needs an FFmpeg executable to mux audio")
        tts_dir = run_dir / "video" / "narration"
        timing = tts.synthesize(plan, tts_dir, tts_python=tts.resolve_tts_python(tts_python),
                                voice=narration_voice, speed=narration_speed)
        planned = {sc.scene_id: sc.duration_seconds for sc in plan.scenes}
        plan = tts.fit_plan_to_narration(plan, timing, fps)
        track = tts.build_narration_track(tts_dir, plan, run_dir / "video" / "narration.wav")
        words_path = run_dir / "video" / "narration.json"
        words = tts.word_timeline(timing, plan)
        words_path.write_text(json.dumps({
            "voice": narration_voice, "speed": narration_speed, "words": words}, indent=2), encoding="utf-8")
        captions_path = (tts.write_captions(words, run_dir / "video" / "narration.ass", width, height,
                                            highlight=tts.CAPTION_HIGHLIGHTS[variant["caption_colour"]])
                         if captions else None)
        narration_info = {
            "engine": "kokoro", "voice": narration_voice, "speed": narration_speed,
            "track": str(track), "words": str(words_path),
            "captions": str(captions_path) if captions_path else None,
            "scene_seconds": {sc.scene_id: {"planned": planned[sc.scene_id], "narrated": sc.duration_seconds}
                              for sc in plan.scenes},
        }

    spec = translate_scene_plan(
        plan,
        width=width,
        height=height,
        fps=fps,
        run_id=resolved_run_id,
        text_overrides=text_overrides,
        layout=variant["layout"],
    )
    spec_path = write_render_spec(spec, run_dir / "video" / RENDER_SPEC_FILENAME)

    backend_info = resolve_backend(encoder)

    synthesized: list[Path] = []
    if synthesize_test_assets:
        synthesized = synthesize_scene_assets(
            run_dir, collect_scene_asset_ids(spec), size=(width, height)
        )
    synthesized_ids = {path.stem for path in synthesized}

    resolver = AssetResolver(run_dir)
    generator: VideoGenerator = build_video_generator(provider)
    result = generator.generate(spec, run_dir=run_dir, resolver=resolver, backend_info=backend_info)

    if synthesized_ids:
        for entry in result.asset_manifest:
            if entry["asset_id"] in synthesized_ids:
                entry["provenance"] = "deterministic_test_asset"
        result.notes["synthesized_test_assets"] = sorted(synthesized_ids)
    if dimension_note:
        result.notes["dimension_normalization"] = dimension_note
    result.notes["variant"] = variant

    final_path = Path(result.final_video)
    if narration_info:
        narrated = final_path.with_name("final_narrated.mp4")
        tts.mux_narration(final_path, Path(narration_info["track"]), narrated, ffmpeg, captions=captions_path)
        narrated.replace(final_path)
        result.notes["narration"] = narration_info
    video_path = run_dir / "video.mp4"
    shutil.copyfile(final_path, video_path)

    result.run_id = resolved_run_id
    result.translator_version = TRANSLATOR_VERSION
    result.video_path = str(video_path)
    result.render_spec_path = str(spec_path)
    result.metadata_path = str(run_dir / VIDEO_METADATA_FILENAME)
    write_video_metadata(result, result.metadata_path)
    update_state(run_dir, resolved_run_id, result)
    return result
