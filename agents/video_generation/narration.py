"""Stage 5 narration: Kokoro TTS through a Python 3.12 subprocess.

Kokoro does not install on the project's Python 3.13, so ``tts_worker.py`` runs
in a separate venv (``.venv-tts`` by default, or ``KOKORO_PYTHON``). Each
scene's length is then set from its narration audio, and the padded narration
track is muxed into the final MP4 with a loudness/clarity pass. Optional
captions are burned in from Kokoro's word timestamps (ASS via ffmpeg/libass).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import wave
from dataclasses import replace
from pathlib import Path

from agents.scene_planner.scene_models import ScenePlan

ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name("tts_worker.py")
DEFAULT_VOICE = "bm_george"  # channel voice chosen 2026-11-24
DEFAULT_SPEED = 1.25
# Kokoro's raw output is ~-25 LUFS and dull; YouTube plays at about -14 LUFS.
CLARITY_FILTER = (
    "highpass=f=80,equalizer=f=250:t=q:w=1:g=-2,equalizer=f=3500:t=q:w=1:g=3,"
    "acompressor=threshold=-20dB:ratio=3:attack=5:release=80,"
    "aresample=48000:resampler=soxr,loudnorm=I=-14:TP=-1.5:LRA=7"
)
WORKER_TIMEOUT_SECONDS = 1800
CAPTION_MAX_WORDS = 3
CAPTION_MAX_GAP = 0.4  # seconds of silence that starts a new caption
CAPTION_Y = 0.45  # above the mascot's head (~0.53 at 1080x1920) and clear of the Shorts UI
# ASS colours are &HAABBGGRR. One is picked per video for the word being spoken.
CAPTION_HIGHLIGHTS = {"gold": "&H0000D7FF&", "cyan": "&H00FFE100&", "lime": "&H0050FF7A&", "coral": "&H00607BFF&"}
CAPTION_HIGHLIGHT = CAPTION_HIGHLIGHTS["gold"]


class NarrationError(RuntimeError):
    """Raised when narration cannot be synthesized or muxed."""


def resolve_tts_python(explicit: str | Path | None = None) -> Path:
    candidates = [explicit, os.environ.get("KOKORO_PYTHON"),
                  ROOT / ".venv-tts" / "Scripts" / "python.exe", ROOT / ".venv-tts" / "bin" / "python"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise NarrationError(
        "no Kokoro Python found: create .venv-tts with Python 3.12 "
        "(see requirements-tts.txt), set KOKORO_PYTHON, or pass --tts-python"
    )


def synthesize(plan: ScenePlan, out_dir: Path, *, tts_python: Path, voice: str, speed: float,
               runner=subprocess.run) -> dict:
    """Run the worker; return its timing.json (per-scene seconds and words)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    request = out_dir / "tts_request.json"
    request.write_text(json.dumps({
        "voice": voice, "speed": speed,
        "scenes": [{"scene_id": s.scene_id, "narration": s.narration} for s in plan.scenes],
    }, indent=2), encoding="utf-8")
    try:
        done = runner([str(tts_python), str(WORKER), str(request), str(out_dir)],
                      capture_output=True, text=True, timeout=WORKER_TIMEOUT_SECONDS)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NarrationError(f"TTS worker failed to run: {exc}") from exc
    if done.returncode != 0:
        tail = "\n".join((done.stderr or "").strip().splitlines()[-5:])
        raise NarrationError(f"TTS worker exited {done.returncode}: {tail}")
    timing = json.loads((out_dir / "timing.json").read_text(encoding="utf-8"))
    got = [s["scene_id"] for s in timing["scenes"]]
    if got != [s.scene_id for s in plan.scenes]:
        raise NarrationError(f"TTS worker returned scenes {got}, expected the plan's order")
    return timing


def fit_plan_to_narration(plan: ScenePlan, timing: dict, fps: int) -> ScenePlan:
    """Each scene lasts exactly as long as its narration, rounded up to a whole frame."""
    seconds = {s["scene_id"]: s["seconds"] for s in timing["scenes"]}
    scenes = [replace(sc, duration_seconds=math.ceil(seconds[sc.scene_id] * fps) / fps)
              for sc in plan.scenes]
    return replace(plan, scenes=scenes)


def build_narration_track(wav_dir: Path, plan: ScenePlan, out_path: Path) -> Path:
    """Concatenate per-scene WAVs, padding each with silence to its scene's length."""
    params = None
    with wave.open(str(out_path), "wb") as out:
        for scene in plan.scenes:
            with wave.open(str(wav_dir / f"{scene.scene_id}.wav"), "rb") as part:
                if params is None:
                    params = part.getparams()
                    out.setparams(params)
                frames = part.readframes(part.getnframes())
            target = round(scene.duration_seconds * params.framerate) * params.sampwidth * params.nchannels
            out.writeframes(frames[:target] + b"\x00" * max(0, target - len(frames)))
    return out_path


def caption_chunks(words: list[dict]) -> list[list[dict]]:
    """Group words into short captions; never across scenes or long pauses."""
    chunks: list[list[dict]] = []
    for word in words:
        last = chunks[-1][-1] if chunks else None
        if (last is None or len(chunks[-1]) >= CAPTION_MAX_WORDS or word["scene_id"] != last["scene_id"]
                or word["start"] - last["end"] > CAPTION_MAX_GAP):
            chunks.append([])
        chunks[-1].append(word)
    return chunks


def _ass_time(t: float) -> str:
    cs = max(0, round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _ass_text(word: str) -> str:
    return word.replace("\\", "").replace("{", "").replace("}", "").upper()


def write_captions(words: list[dict], path: Path, width: int, height: int,
                   highlight: str = CAPTION_HIGHLIGHT) -> Path:
    """One ASS event per spoken word: the chunk, with that word highlighted."""
    events = []
    chunks = caption_chunks(words)
    for i, chunk in enumerate(chunks):
        nxt = chunks[i + 1][0]["start"] if i + 1 < len(chunks) else None
        chunk_end = chunk[-1]["end"]
        if nxt is not None and nxt - chunk_end <= CAPTION_MAX_GAP:
            chunk_end = nxt  # hold the caption through short pauses instead of flickering
        for j, word in enumerate(chunk):
            start = word["start"] if j else chunk[0]["start"]
            end = chunk[j + 1]["start"] if j + 1 < len(chunk) else chunk_end
            text = " ".join(f"{{\\c{highlight}}}{_ass_text(w['word'])}{{\\c&H00FFFFFF&}}" if k == j
                            else _ass_text(w["word"]) for k, w in enumerate(chunk))
            events.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Caption,,0,0,0,,"
                          f"{{\\pos({width // 2},{round(height * CAPTION_Y)})}}{text}")
    size = round(height * 0.05)
    path.write_text("\n".join([
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}", f"PlayResY: {height}",
        "WrapStyle: 0", "", "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Caption,Arial,{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
        f"-1,0,0,0,100,100,0,0,1,{max(2, size // 12)},{max(1, size // 30)},5,60,60,0,1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        *events, ""]), encoding="utf-8")
    return path


def mux_narration(video: Path, track: Path, out_path: Path, ffmpeg: str,
                  captions: Path | None = None) -> Path:
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video.resolve()),
           "-i", str(track.resolve()), "-map", "0:v:0", "-map", "1:a:0"]
    if captions:
        # Relative name + cwd avoids escaping Windows drive colons inside the filter graph.
        cmd += ["-vf", f"subtitles={captions.name}", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "18", "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c:v", "copy"]
    cmd += ["-af", CLARITY_FILTER, "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart", str(out_path.resolve())]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                              cwd=str(captions.parent) if captions else None)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NarrationError(f"ffmpeg mux failed to run: {exc}") from exc
    if done.returncode != 0 or not out_path.exists():
        raise NarrationError(f"ffmpeg mux failed: {(done.stderr or '').strip()[-400:]}")
    return out_path


def word_timeline(timing: dict, plan: ScenePlan) -> list[dict]:
    """Per-word timestamps on the video timeline, used for captions."""
    words, start = [], 0.0
    for entry, scene in zip(timing["scenes"], plan.scenes):
        # Kokoro gives some punctuation tokens ("." ",") timestamps too; they are not words.
        words += [{**w, "start": round(start + w["start"], 3), "end": round(start + w["end"], 3),
                   "scene_id": scene.scene_id} for w in entry["words"] if any(c.isalnum() for c in w["word"])]
        start += scene.duration_seconds
    return words

