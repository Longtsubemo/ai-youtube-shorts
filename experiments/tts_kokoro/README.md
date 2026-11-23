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
experiments/tts_kokoro/.venv/Scripts/python.exe experiments/tts_kokoro/narrate.py --run-id <run_id> --voices af_heart,am_michael,bm_george
```

Output: `experiments/tts_kokoro/output/<run_id>/<voice>.wav` and
`<voice>.timing.json` (gitignored).
