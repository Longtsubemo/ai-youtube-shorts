# Kokoro TTS experiment

Isolated experiment (not imported by the pipeline): narrate a run's
`scene_plan.json` with [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
(Apache-2.0 code and weights) on CPU, and record per-scene audio length and
per-word timestamps.

## Setup (Python 3.12)

```powershell
py -3.12 -m venv experiments/tts_kokoro/.venv
experiments/tts_kokoro/.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
experiments/tts_kokoro/.venv/Scripts/python.exe -m pip install -r experiments/tts_kokoro/requirements.txt
```

The first run downloads the model (~330 MB, Hugging Face cache) and the spaCy
`en_core_web_sm` model (into the venv).

## Run

```powershell
experiments/tts_kokoro/.venv/Scripts/python.exe experiments/tts_kokoro/narrate.py --run-id <run_id>
```

**Channel voice (decided 2026-11-24): `bm_george` at speed 1.25**, followed by an
ffmpeg clarity pass (-14 LUFS, presence EQ, light compression, 48 kHz). These
are the defaults; `--voices` and `--speed` override them. ffmpeg must be on
PATH for the clarity pass, otherwise only the raw WAV is written.

Output (gitignored) in `experiments/tts_kokoro/output/<run_id>/`:
`<voice>_<speed>x.raw.wav`, `<voice>_<speed>x.wav` (processed) and
`<voice>_<speed>x.timing.json`.
