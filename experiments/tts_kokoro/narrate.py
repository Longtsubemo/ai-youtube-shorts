"""Experiment: narrate a run's scene plan with Kokoro-82M (Apache-2.0).

Not production code and not imported by the pipeline. It answers three questions
before any Stage 5 change: how the voices sound, how long the real narration is
per scene (vs. the planned scene durations), and whether Kokoro's per-word
timestamps are usable for captions.

    experiments/tts_kokoro/.venv/Scripts/python.exe experiments/tts_kokoro/narrate.py \
        --run-id e2e-20261123-gemini-02

Defaults are the chosen channel voice: bm_george at speed 1.25, then an ffmpeg
clarity pass (loudness to -14 LUFS, presence EQ, light compression, 48 kHz).

Writes to experiments/tts_kokoro/output/<run_id>/ (WAVs are gitignored):
    <voice>_<speed>x.raw.wav      Kokoro output, 24 kHz
    <voice>_<speed>x.wav          after the clarity pass (needs ffmpeg on PATH)
    <voice>_<speed>x.timing.json  per-scene audio duration vs. plan, per-word timestamps
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline

SAMPLE_RATE = 24_000  # Kokoro's fixed output rate
REPO_ID = "hexgrad/Kokoro-82M"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VOICE = "bm_george"
DEFAULT_SPEED = 1.25
# Kokoro's raw output sits near -25 LUFS and sounds muffled; YouTube plays at about -14.
CLARITY_FILTER = (
    "highpass=f=80,equalizer=f=250:t=q:w=1:g=-2,equalizer=f=3500:t=q:w=1:g=3,"
    "acompressor=threshold=-20dB:ratio=3:attack=5:release=80,"
    "aresample=48000:resampler=soxr,loudnorm=I=-14:TP=-1.5:LRA=7"
)


def clarify(raw: Path, out: Path) -> bool:
    """Run the clarity pass; timing is unchanged (no tempo filters)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw),
                    "-af", CLARITY_FILTER, "-ar", "48000", str(out)], check=True)
    return True


def load_scenes(run_id: str) -> list[dict]:
    plan = json.loads((ROOT / "data" / run_id / "scene_plan.json").read_text(encoding="utf-8"))
    return plan.get("data", plan)["scenes"]


def narrate(pipeline: KPipeline, scenes: list[dict], voice: str, speed: float) -> tuple[np.ndarray, dict]:
    chunks: list[np.ndarray] = []
    timing = {"voice": voice, "speed": speed, "scenes": []}
    offset = 0.0
    for scene in scenes:
        scene_audio: list[np.ndarray] = []
        words: list[dict] = []
        chunk_offset = offset
        for result in pipeline(scene["narration"], voice=voice, speed=speed):
            audio = result.audio.numpy() if result.audio is not None else np.zeros(0, np.float32)
            for tok in result.tokens or []:
                if tok.start_ts is None or tok.end_ts is None:
                    continue  # punctuation/whitespace tokens carry no timing
                words.append({
                    "word": tok.text,
                    "start": round(chunk_offset + tok.start_ts, 3),
                    "end": round(chunk_offset + tok.end_ts, 3),
                })
            scene_audio.append(audio)
            chunk_offset += len(audio) / SAMPLE_RATE
        audio = np.concatenate(scene_audio) if scene_audio else np.zeros(0, np.float32)
        seconds = len(audio) / SAMPLE_RATE
        timing["scenes"].append({
            "scene_id": scene["scene_id"],
            "planned_seconds": scene["duration_seconds"],
            "audio_seconds": round(seconds, 2),
            "start": round(offset, 3),
            "words": words,
        })
        chunks.append(audio)
        offset += seconds
    timing["total_audio_seconds"] = round(offset, 2)
    timing["total_planned_seconds"] = sum(s["duration_seconds"] for s in scenes)
    return np.concatenate(chunks), timing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--voices", default=DEFAULT_VOICE)
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    args = parser.parse_args()

    scenes = load_scenes(args.run_id)
    out_dir = Path(__file__).parent / "output" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pipelines: dict[str, KPipeline] = {}
    for voice in [v.strip() for v in args.voices.split(",") if v.strip()]:
        lang = voice[0]  # Kokoro voice ids start with their language code: a=US, b=UK English
        pipeline = pipelines.setdefault(lang, KPipeline(lang_code=lang, repo_id=REPO_ID))
        started = time.perf_counter()
        audio, timing = narrate(pipeline, scenes, voice, args.speed)
        timing["synthesis_seconds"] = round(time.perf_counter() - started, 2)
        stem = f"{voice}_{args.speed:g}x"
        raw = out_dir / f"{stem}.raw.wav"
        sf.write(raw, audio, SAMPLE_RATE)
        if not clarify(raw, out_dir / f"{stem}.wav"):
            print("warning: ffmpeg not on PATH, clarity pass skipped (raw WAV only)")
        (out_dir / f"{stem}.timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
        per_scene = " ".join(f"{s['scene_id']}={s['audio_seconds']}/{s['planned_seconds']}" for s in timing["scenes"])
        words = sum(len(s["words"]) for s in timing["scenes"])
        print(
            f"{voice}: audio {timing['total_audio_seconds']}s vs plan {timing['total_planned_seconds']}s, "
            f"{words} timed words, synth {timing['synthesis_seconds']}s | {per_scene}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
