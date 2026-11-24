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


def test_cli_narrates_with_channel_voice_by_default():
    args = build_parser().parse_args([])
    assert args.narration == "kokoro" and not args.no_captions
    assert (args.voice, args.narration_speed) == ("bm_george", 1.25)


def test_captions_group_words_and_highlight_each_spoken_word(tmp_path):
    w = lambda word, start, end, scene="sc01": {"word": word, "start": start, "end": end, "scene_id": scene}  # noqa: E731
    words = [w("a", 0.0, 0.2), w("b", 0.2, 0.4), w("c", 0.4, 0.6), w("d", 0.6, 0.8),  # 4th word -> new chunk
             w("e", 2.0, 2.2),                                                       # long pause -> new chunk
             w("f", 2.2, 2.4, "sc02")]                                               # new scene -> new chunk
    assert [[x["word"] for x in c] for c in tts.caption_chunks(words)] == [["a", "b", "c"], ["d"], ["e"], ["f"]]

    ass = tts.write_captions(words, tmp_path / "c.ass", 1080, 1920).read_text(encoding="utf-8")
    events = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
    assert len(events) == len(words)
    assert "0:00:00.00,0:00:00.20" in events[0] and r"{\c&H0000D7FF&}A{\c&H00FFFFFF&} B C" in events[0]
    assert "0:00:00.40,0:00:00.60" in events[2]       # caption held to the next chunk's start (short gap)
    assert "0:00:00.60,0:00:00.80" in events[3]       # not held across the long pause
    assert "PlayResY: 1920" in ass and r"\pos(540,864)" in events[0]


def test_variant_is_stable_per_run_and_spreads_across_runs():
    from agents.video_generation.stage import pick_variant
    from agents.video_generation.translator import MASCOT_LAYOUTS

    assert pick_variant("run-a") == pick_variant("run-a")
    looks = {tuple(pick_variant(f"run-{i}").values()) for i in range(60)}
    assert {layout for layout, _ in looks} == set(MASCOT_LAYOUTS)
    assert {colour for _, colour in looks} == set(tts.CAPTION_HIGHLIGHTS)


def test_layout_moves_the_mascot(tmp_path):
    from agents.scene_planner.provider import MockScenePlannerProvider
    from agents.video_generation.translator import translate_scene_plan
    from tests.test_scene_planner_generation_prompts import build_script

    plan = MockScenePlannerProvider().generate({"script": build_script()})
    xs = {}
    for layout in ("left", "center", "right"):
        spec = translate_scene_plan(plan, layout=layout).to_dict()
        xs[layout] = next(e["transform"]["x"] for sc in spec["scenes"] for e in sc["elements"] if e["id"] == "mascot")
    assert xs["left"] < xs["center"] < xs["right"]
