"""Stage 5 tests: stage orchestration, artifacts, CLI, determinism, isolation."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import cv2
import pytest

from agents.scene_planner.scene_models import Mascot, Scene, ScenePlan
from agents.video_generation.encoder import detect_ffmpeg, probe_video
from agents.video_generation.provider import VideoGeneratorError, build_video_generator
from agents.video_generation.stage import load_scene_plan, run_video_generation
from agents.video_generation.video_cli import main as video_cli_main

CANONICAL_MASCOT = "assets/channels/finance_mascot/mascot.png"
REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_PLAN_PATH = REPO_ROOT / "data" / "e2e-20261005-stage4-live-01" / "scene_plan.json"

WIDTH, HEIGHT, FPS = 160, 288, 10


def make_plan_dict(scene_count: int = 3, duration: int = 1) -> dict:
    scenes = []
    for index in range(scene_count):
        scenes.append(
            Scene(
                scene_id=f"sc{index + 1:02d}",
                order=index,
                duration_seconds=duration,
                narration=f"Narration line {index + 1} for the test scene.",
                visual_description="generic test visual",
                camera="push in" if index % 2 else "medium shot",
                required_assets=[CANONICAL_MASCOT],
                transition_hint="cut",
                mascot=Mascot(present=True, action="standing", asset=CANONICAL_MASCOT),
            ).to_dict()
        )
    return {
        "topic_id": "t_test",
        "title": "Test plan",
        "target_duration_seconds": 3,
        "scenes": scenes,
        "channel_id": "finance_mascot",
        "mascot_asset_path": CANONICAL_MASCOT,
        "run_id": "run_test",
    }


def write_run(run_dir: Path, scene_count: int = 3, duration: int = 1) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = make_plan_dict(scene_count, duration)
    (run_dir / "scene_plan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def first_frame(path: Path):
    capture = cv2.VideoCapture(str(path))
    ok, frame = capture.read()
    capture.release()
    assert ok, f"could not read a frame from {path}"
    return frame


# 17 + 18 + 21 - assembly, scene count, flat layout ---------------------------


def test_stage_renders_scenes_assembles_final_and_keeps_flat_layout(tmp_path):
    run_dir = tmp_path / "run"
    write_run(run_dir, scene_count=3, duration=1)
    plan = load_scene_plan(run_dir / "scene_plan.json")

    result = run_video_generation(
        plan, run_dir=run_dir, run_id="run", width=WIDTH, height=HEIGHT, fps=FPS
    )

    video_dir = run_dir / "video"
    expected_clips = [video_dir / f"sc0{i}.mp4" for i in (1, 2, 3)]
    for clip in expected_clips:
        assert clip.exists(), f"missing scene clip {clip}"
        assert probe_video(clip)["valid"], f"scene clip not playable: {clip}"
    assert result.scene_count == 3
    assert result.scene_ids == ["sc01", "sc02", "sc03"]

    final = video_dir / "final.mp4"
    assert final.exists()
    final_probe = probe_video(final)
    assert final_probe["valid"]
    assert final_probe["frames"] >= int(3 * FPS * 0.9)
    assert (final_probe["width"], final_probe["height"]) == (WIDTH, HEIGHT)
    assert final_probe["fps"] == pytest.approx(FPS)

    video_mp4 = run_dir / "video.mp4"
    assert video_mp4.exists()
    assert video_mp4.stat().st_size == final.stat().st_size

    # flat artifact layout
    top_level = {p.name for p in run_dir.iterdir()}
    assert {"scene_plan.json", "video", "video.mp4", "video.json", "state.json"} <= top_level
    assert video_dir.joinpath("render_spec.json").exists()
    for nested in ("research", "scripts", "production", "videos", "audio"):
        assert not (run_dir / nested).exists(), f"nested layout leaked: {nested}"

    metadata = json.loads((run_dir / "video.json").read_text(encoding="utf-8"))
    assert metadata["run_id"] == "run"
    assert metadata["scene_count"] == 3
    assert len(metadata["scene_clips"]) == 3
    assert metadata["final_video"].endswith(str(Path("video") / "final.mp4"))
    assert metadata["video_path"].endswith("video.mp4")
    assert metadata["duration_seconds"] == pytest.approx(3.0)
    assert metadata["resolution"] == {"width": WIDTH, "height": HEIGHT}
    assert metadata["fps"] == FPS
    assert metadata["renderer_version"]
    assert metadata["translator_version"] == "stage5-translator/1.0"
    assert metadata["encoder"]["selected"] in ("opencv", "ffmpeg")
    assert isinstance(metadata["asset_manifest"], list)
    assert any(e["asset_id"] == CANONICAL_MASCOT for e in metadata["asset_manifest"])

    # spec preserved creative fields instead of discarding them
    spec = json.loads((video_dir / "render_spec.json").read_text(encoding="utf-8"))
    assert [s["scene_id"] for s in spec["scenes"]] == ["sc01", "sc02", "sc03"]
    assert spec["scenes"][0]["metadata"]["narration"]
    assert spec["scenes"][0]["metadata"]["visual_description"]
    assert spec["scenes"][0]["metadata"]["transition_hint"]


# 22 - state progression ------------------------------------------------------


def test_state_json_advances_to_video_generation(tmp_path):
    run_dir = tmp_path / "run"
    write_run(run_dir, scene_count=2, duration=1)
    plan = load_scene_plan(run_dir / "scene_plan.json")
    run_video_generation(
        plan, run_dir=run_dir, run_id="run", width=WIDTH, height=HEIGHT, fps=FPS
    )

    from pipeline.state import Stage

    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert state["current_stage"] == Stage.VIDEO_GENERATION.value
    assert state["run_id"] == "run"
    artifacts = state["artifacts"]["video_generation"]
    assert any(a.endswith("video.mp4") for a in artifacts)
    assert any(a.endswith("final.mp4") for a in artifacts)
    assert any(a.endswith("video.json") for a in artifacts)


def test_state_progression_preserves_earlier_stage_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    write_run(run_dir, scene_count=1, duration=1)
    from pipeline.state import PipelineState, Stage

    prior = PipelineState(run_id="run")
    prior.advance_to(Stage.SCENE_PLANNING)
    prior.add_artifact(Stage.SCENE_PLANNING, str(run_dir / "scene_plan.json"))
    prior.save(run_dir / "state.json")

    plan = load_scene_plan(run_dir / "scene_plan.json")
    run_video_generation(
        plan, run_dir=run_dir, run_id="run", width=WIDTH, height=HEIGHT, fps=FPS
    )
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert "scene_planning" in state["artifacts"]
    assert "video_generation" in state["artifacts"]


# 20 - deterministic output ---------------------------------------------------


def test_stage_output_is_deterministic_for_same_inputs(tmp_path):
    run_dir = tmp_path / "run"
    write_run(run_dir, scene_count=2, duration=1)
    plan = load_scene_plan(run_dir / "scene_plan.json")

    run_video_generation(
        plan, run_dir=run_dir, run_id="run", width=WIDTH, height=HEIGHT, fps=FPS
    )
    spec_first = (run_dir / "video" / "render_spec.json").read_bytes()
    meta_first = (run_dir / "video.json").read_bytes()
    frame_first = first_frame(run_dir / "video" / "sc01.mp4")

    run_video_generation(
        plan, run_dir=run_dir, run_id="run", width=WIDTH, height=HEIGHT, fps=FPS
    )
    spec_second = (run_dir / "video" / "render_spec.json").read_bytes()
    meta_second = (run_dir / "video.json").read_bytes()
    frame_second = first_frame(run_dir / "video" / "sc01.mp4")

    assert spec_first == spec_second
    assert meta_first == meta_second
    assert (frame_first == frame_second).all()


# 24 - no network requirement -------------------------------------------------


def test_stage_runs_with_all_sockets_blocked(tmp_path, monkeypatch):
    def blocked(*_args, **_kwargs):
        raise AssertionError("network access attempted during Stage 5")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)

    run_dir = tmp_path / "run"
    write_run(run_dir, scene_count=2, duration=1)
    plan = load_scene_plan(run_dir / "scene_plan.json")
    result = run_video_generation(
        plan, run_dir=run_dir, run_id="run", width=WIDTH, height=HEIGHT, fps=FPS
    )
    assert result.scene_count == 2
    assert probe_video(result.final_video)["valid"]


# CLI -------------------------------------------------------------------------


def test_cli_reports_missing_scene_plan(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    exit_code = video_cli_main(["--run-id", "does_not_exist"])
    assert exit_code == 2
    assert "input error" in capsys.readouterr().out


def test_cli_renders_run_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    run_id = "cli_run"
    write_run(Path("data") / run_id, scene_count=2, duration=1)

    exit_code = video_cli_main(
        ["--run-id", run_id, "--width", str(WIDTH), "--height", str(HEIGHT), "--fps", str(FPS)]
    )
    out = capsys.readouterr().out
    assert exit_code == 0, out
    assert "final.mp4" in out
    run_dir = Path("data") / run_id
    assert (run_dir / "video" / "final.mp4").exists()
    assert (run_dir / "video.mp4").exists()
    assert (run_dir / "video.json").exists()
    assert probe_video(run_dir / "video" / "final.mp4")["valid"]


@pytest.mark.skipif(
    detect_ffmpeg() is not None, reason="FFmpeg executable present; absence path only"
)
def test_cli_encoder_ffmpeg_reports_clear_error_when_absent(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    run_id = "ffmpeg_run"
    write_run(Path("data") / run_id, scene_count=1, duration=1)

    exit_code = video_cli_main(
        [
            "--run-id",
            run_id,
            "--encoder",
            "ffmpeg",
            "--width",
            str(WIDTH),
            "--height",
            str(HEIGHT),
            "--fps",
            str(FPS),
        ]
    )
    out = capsys.readouterr().out
    assert exit_code == 3
    assert "FFmpeg executable not found" in out


def test_unknown_provider_is_rejected():
    with pytest.raises(VideoGeneratorError) as excinfo:
        build_video_generator("future_ai_video")
    assert "future_ai_video" in str(excinfo.value)
    assert "local" in str(excinfo.value)
    assert build_video_generator("local").name == "local"


def test_provider_contract_is_provider_based():
    from agents.video_generation.provider import VideoGenerator

    generator = build_video_generator("local")
    assert isinstance(generator, VideoGenerator)
    assert generator.renderer_name == "local-motion-renderer"
    # future AIVideoRenderer would register the same way, without stage changes
    from agents.video_generation import provider as provider_module

    assert set(provider_module.PROVIDERS) == {"local"}


# deterministic test-asset seam -----------------------------------------------


def test_synthesized_test_assets_are_marked_and_mascot_is_never_written(tmp_path):
    run_dir = tmp_path / "run"
    write_run(run_dir, scene_count=2, duration=1)
    plan = load_scene_plan(run_dir / "scene_plan.json")

    result = run_video_generation(
        plan,
        run_dir=run_dir,
        run_id="run",
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        synthesize_test_assets=True,
    )

    for name in ("sc01_main.png", "sc02_main.png"):
        assert (run_dir / "assets" / name).exists()
    manifest = {entry["asset_id"]: entry for entry in result.asset_manifest}
    assert manifest["sc01_main"]["provenance"] == "deterministic_test_asset"
    assert result.notes["synthesized_test_assets"] == ["sc01_main", "sc02_main"]

    # channel mascot: reported, never fabricated
    mascot_entry = manifest[CANONICAL_MASCOT]
    expected = "channel_asset" if (REPO_ROOT / CANONICAL_MASCOT).exists() else "missing_placeholder"
    assert mascot_entry["provenance"] == expected
    assert not (REPO_ROOT / CANONICAL_MASCOT).exists() or expected == "channel_asset"
    assert not (run_dir / CANONICAL_MASCOT).exists()


def test_scene_assets_are_never_written_into_channel_asset_locations(tmp_path):
    run_dir = tmp_path / "run"
    write_run(run_dir, scene_count=1, duration=1)
    plan = load_scene_plan(run_dir / "scene_plan.json")
    run_video_generation(
        plan,
        run_dir=run_dir,
        run_id="run",
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        synthesize_test_assets=True,
    )
    assert not (run_dir / "assets" / "channels").exists()
    assert sorted(p.name for p in (run_dir / "assets").iterdir()) == ["sc01_main.png"]


# real Stage 4 artifact -------------------------------------------------------


@pytest.mark.skipif(not REAL_PLAN_PATH.exists(), reason="real Stage 4 artifact not present")
def test_smoke_run_of_real_seven_scene_plan(tmp_path):
    """Mini smoke: the real 7-scene plan rendered locally at test resolution."""
    run_dir = tmp_path / "real_run"
    run_dir.mkdir()
    (run_dir / "scene_plan.json").write_bytes(REAL_PLAN_PATH.read_bytes())

    plan = load_scene_plan(run_dir / "scene_plan.json")
    result = run_video_generation(
        plan,
        run_dir=run_dir,
        run_id="real_run",
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        synthesize_test_assets=True,
    )

    assert result.scene_ids == ["sc01", "sc02", "sc03", "sc04", "sc05", "sc06", "sc07"]
    assert result.scene_count == 7
    assert result.duration_seconds == pytest.approx(68.0)
    final_probe = probe_video(result.final_video)
    assert final_probe["valid"]
    assert final_probe["frames"] >= int(68 * FPS * 0.9)
    for clip in result.scene_clips:
        assert clip.playable, clip.path
    assert (run_dir / "video.mp4").exists()
    assert (run_dir / "video.json").exists()
    # the source artifact was used read-only
    assert (run_dir / "scene_plan.json").read_bytes() == REAL_PLAN_PATH.read_bytes()
