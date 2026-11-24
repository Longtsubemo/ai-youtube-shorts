"""Stage 5 narration bridge: worker contract, timing fit, padded track (no Kokoro needed)."""

from __future__ import annotations

import json
import subprocess
import wave
from dataclasses import replace
from pathlib import Path

import pytest

from agents.scene_planner.scene_models import Scene, ScenePlan
from agents.video_generation import narration as tts
from agents.video_generation.video_cli import build_parser

RATE = 24_000


def _plan() -> ScenePlan:
    scenes = [Scene(scene_id=f"sc0{i}", narration=f"line {i}", duration_seconds=5) for i in (1, 2)]
    return ScenePlan(topic_id="t", scenes=scenes)


def _write_wav(path: Path, seconds: float) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(b"\x01\x00" * round(seconds * RATE))


def _fake_worker(seconds: dict[str, float]):
    def run(cmd, **kwargs):
        request, out_dir = json.loads(Path(cmd[2]).read_text(encoding="utf-8")), Path(cmd[3])
        scenes = []
        for s in request["scenes"]:
            _write_wav(out_dir / f"{s['scene_id']}.wav", seconds[s["scene_id"]])
            scenes.append({"scene_id": s["scene_id"], "seconds": seconds[s["scene_id"]],
                           "words": [{"word": "line", "start": 0.1, "end": 0.4}]})
        (out_dir / "timing.json").write_text(json.dumps({"scenes": scenes}), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return run


def test_scenes_follow_narration_and_track_is_padded_to_match(tmp_path):
    plan = _plan()
    timing = tts.synthesize(plan, tmp_path, tts_python=Path("py"), voice="bm_george", speed=1.25,
                            runner=_fake_worker({"sc01": 2.01, "sc02": 3.5}))
    fitted = tts.fit_plan_to_narration(plan, timing, fps=30)
    assert [s.duration_seconds for s in fitted.scenes] == [61 / 30, 3.5]  # 2.01s rounds up to frame 61

    track = tts.build_narration_track(tmp_path, fitted, tmp_path / "narration.wav")
    with wave.open(str(track)) as w:
        assert w.getnframes() == round(61 / 30 * RATE) + round(3.5 * RATE)

    words = tts.word_timeline(timing, fitted)
    assert [w["start"] for w in words] == [0.1, round(61 / 30 + 0.1, 3)]
    assert words[1]["scene_id"] == "sc02"


def test_worker_failure_and_scene_mismatch_raise(tmp_path):
    plan = _plan()
    failing = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "boom")  # noqa: E731
    with pytest.raises(tts.NarrationError, match="boom"):
        tts.synthesize(plan, tmp_path, tts_python=Path("py"), voice="v", speed=1.0, runner=failing)

    reordered = replace(plan, scenes=list(reversed(plan.scenes)))
    run = _fake_worker({"sc01": 1.0, "sc02": 1.0})
    tts.synthesize(reordered, tmp_path, tts_python=Path("py"), voice="v", speed=1.0, runner=run)
    with pytest.raises(tts.NarrationError, match="order"):
        tts.synthesize(plan, tmp_path, tts_python=Path("py"), voice="v", speed=1.0,
                       runner=lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""))


def test_missing_tts_python_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.delenv("KOKORO_PYTHON", raising=False)
    monkeypatch.setattr(tts, "ROOT", tmp_path)
    with pytest.raises(tts.NarrationError, match="venv-tts"):
        tts.resolve_tts_python(None)


def test_cli_narration_is_opt_in_with_channel_voice_defaults():
    args = build_parser().parse_args([])
    assert args.narration == "none"
    assert (args.voice, args.narration_speed) == ("bm_george", 1.25)
