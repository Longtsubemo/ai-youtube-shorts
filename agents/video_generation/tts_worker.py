"""Kokoro-82M narration worker, run as a subprocess by Stage 5.

Kokoro needs Python 3.12, so this script runs in the separate TTS venv and
imports nothing from the project. ``narration.py`` is the caller.

    <tts-python> tts_worker.py <request.json> <out_dir>

request.json: {"voice": str, "speed": float, "scenes": [{"scene_id", "narration"}]}
Writes <out_dir>/<scene_id>.wav (16-bit PCM, 24 kHz mono) per scene and
<out_dir>/timing.json with per-scene seconds and per-word timestamps
(seconds from the start of that scene).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SAMPLE_RATE = 24_000
REPO_ID = "hexgrad/Kokoro-82M"


def main(argv: list[str]) -> int:
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    request = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    out_dir = Path(argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    voice, speed = request["voice"], float(request["speed"])
    pipeline = KPipeline(lang_code=voice[0], repo_id=REPO_ID)  # voice ids start with the language code

    timing = {"voice": voice, "speed": speed, "sample_rate": SAMPLE_RATE, "scenes": []}
    for scene in request["scenes"]:
        chunks, words, offset = [], [], 0.0
        for result in pipeline(scene["narration"], voice=voice, speed=speed):
            audio = result.audio.numpy() if result.audio is not None else np.zeros(0, np.float32)
            for tok in result.tokens or []:
                if tok.start_ts is not None and tok.end_ts is not None:
                    words.append({"word": tok.text, "start": round(offset + tok.start_ts, 3),
                                  "end": round(offset + tok.end_ts, 3)})
            chunks.append(audio)
            offset += len(audio) / SAMPLE_RATE
        audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        if audio.size == 0:
            print(f"no audio produced for scene {scene['scene_id']}", file=sys.stderr)
            return 1
        sf.write(out_dir / f"{scene['scene_id']}.wav", audio, SAMPLE_RATE, subtype="PCM_16")
        timing["scenes"].append({"scene_id": scene["scene_id"],
                                 "seconds": round(len(audio) / SAMPLE_RATE, 3), "words": words})
    (out_dir / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
