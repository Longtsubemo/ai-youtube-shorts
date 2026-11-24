"""Stage 5 CLI: Video Generation.

Reads ``data/<run_id>/scene_plan.json`` and renders real MP4 files::

    python -m agents.video_generation.video_cli --run-id <run_id> --provider local

Outputs::

    data/<run_id>/video/<scene_id>.mp4 ...
    data/<run_id>/video/final.mp4
    data/<run_id>/video/render_spec.json
    data/<run_id>/video.mp4
    data/<run_id>/video.json
    data/<run_id>/state.json   -> video_generation

Exit codes: 0 success, 2 input/structure problem, 3 render/encode problem.

No network access and no hosted model calls are made. ``--narration kokoro`` runs
local Kokoro TTS in a Python 3.12 subprocess (see agents/video_generation/narration.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from uuid import uuid4

from agents.video_generation.encoder import BACKENDS, VideoEncodingError, detect_ffmpeg
from agents.video_generation.models import VideoResult
from agents.video_generation.narration import DEFAULT_SPEED, DEFAULT_VOICE, NarrationError
from agents.video_generation.provider import (
    VideoGeneratorError,
    available_providers,
)
from agents.video_generation.stage import (
    VideoGenerationError,
    load_scene_plan,
    run_video_generation,
)
from agents.video_generation.translator import TRANSLATOR_VERSION
from agents.video_generation.render_spec import DEFAULT_FPS, DEFAULT_HEIGHT, DEFAULT_WIDTH


def _print_summary(result: VideoResult) -> None:
    print(f"[video_generation] run_id={result.run_id}")
    print(f"[video_generation] provider={result.provider} renderer={result.renderer} v{result.renderer_version}")
    print(f"[video_generation] translator={TRANSLATOR_VERSION}")
    encoder = result.encoder
    ffmpeg_note = encoder.get("ffmpeg_executable") or "not found"
    print(
        f"[video_generation] encoder backend={encoder.get('selected')} "
        f"(ffmpeg: {ffmpeg_note}; codec: {encoder.get('codec')})"
    )
    print(
        f"[video_generation] scenes={result.scene_count} "
        f"duration={result.duration_seconds:.2f}s "
        f"{result.width}x{result.height} @{result.fps}fps"
    )
    for clip in result.scene_clips:
        status = "ok" if clip.playable else "BAD"
        print(
            f"[video_generation]   {clip.scene_id}: {clip.frames} frames "
            f"{clip.duration_seconds:.2f}s [{status}] {clip.path}"
        )
    narration = result.notes.get("narration")
    if narration:
        print(f"[video_generation] narration: kokoro {narration['voice']} x{narration['speed']} -> {narration['track']}")
    print(f"[video_generation] final video: {result.final_video}")
    print(f"[video_generation] video.mp4:   {result.video_path}")
    print(f"[video_generation] metadata:    {result.metadata_path}")
    missing = result.notes.get("missing_assets") or []
    if missing:
        ids = ", ".join(sorted({entry['asset_id'] for entry in missing}))
        print(f"[video_generation] note: missing assets used placeholders: {ids}")
    unsupported = result.notes.get("unsupported_animations") or []
    if unsupported:
        print(f"[video_generation] note: unsupported animations skipped: {len(unsupported)}")
    skipped = result.notes.get("skipped_elements") or []
    if skipped:
        print(f"[video_generation] note: elements skipped: {len(skipped)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agents.video_generation.video_cli",
        description="Render a scene plan into MP4 files (Stage 5).",
    )
    parser.add_argument("--run-id", default=None, help="run id (default: generated)")
    parser.add_argument(
        "--scene-plan",
        type=Path,
        default=None,
        help="scene plan path (default: data/<run_id>/scene_plan.json)",
    )
    parser.add_argument(
        "--provider",
        choices=available_providers(),
        default="local",
        help="video generator provider (default: local)",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="output width")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="output height")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="output frame rate")
    parser.add_argument(
        "--encoder",
        choices=BACKENDS,
        default="auto",
        help="encoding backend: auto prefers an FFmpeg executable, else OpenCV",
    )
    parser.add_argument(
        "--synthesize-test-assets",
        action="store_true",
        help=(
            "write deterministic local test images for scene assets that do not "
            "exist yet (recorded as deterministic_test_asset in video.json; "
            "channel assets such as the mascot are never written)"
        ),
    )
    parser.add_argument(
        "--narration",
        choices=["none", "kokoro"],
        default="none",
        help="narrate scenes with local Kokoro TTS; scene lengths then follow the audio (default: none)",
    )
    parser.add_argument("--voice", default=DEFAULT_VOICE, help=f"Kokoro voice (default: {DEFAULT_VOICE})")
    parser.add_argument("--narration-speed", type=float, default=DEFAULT_SPEED,
                        help=f"Kokoro speaking speed (default: {DEFAULT_SPEED})")
    parser.add_argument("--no-captions", action="store_true",
                        help="with --narration, skip the burned-in word captions")
    parser.add_argument("--tts-python", default=None,
                        help="Python 3.12 with kokoro installed (default: KOKORO_PYTHON or .venv-tts)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    run_id = args.run_id or uuid4().hex[:12]
    run_dir = Path("data") / run_id
    scene_plan_path = args.scene_plan or (run_dir / "scene_plan.json")

    ffmpeg_path = detect_ffmpeg()
    print(
        f"[video_generation] ffmpeg executable: "
        f"{ffmpeg_path if ffmpeg_path else 'not found (OpenCV writer will be used)'}"
    )

    try:
        plan = load_scene_plan(scene_plan_path)
    except VideoGenerationError as exc:
        print(f"[video_generation] input error: {exc}")
        return 2

    print(
        f"[video_generation] scene plan: {scene_plan_path} "
        f"({len(plan.scenes)} scenes, {plan.estimated_seconds}s)"
    )

    try:
        result = run_video_generation(
            plan,
            run_dir=run_dir,
            run_id=run_id,
            provider=args.provider,
            width=args.width,
            height=args.height,
            fps=args.fps,
            encoder=args.encoder,
            synthesize_test_assets=args.synthesize_test_assets,
            narration_voice=args.voice if args.narration == "kokoro" else None,
            narration_speed=args.narration_speed,
            tts_python=args.tts_python,
            captions=not args.no_captions,
        )
    except VideoGenerationError as exc:
        print(f"[video_generation] input error: {exc}")
        return 2
    except (VideoGeneratorError, VideoEncodingError, NarrationError) as exc:
        print(f"[video_generation] render error: {exc}")
        return 3
    except OSError as exc:
        print(f"[video_generation] render error: {type(exc).__name__}: {exc}")
        return 3

    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
