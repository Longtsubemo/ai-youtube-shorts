"""Stage/CLI tests for Stage 8: fixtures, pairing guard, artifact + state output."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agents.qa.video.models import VideoQAConfig, VideoQAError
from agents.qa.video.stage import (
    DEFAULT_VIDEO_RESULT_PATH,
    load_committed_fixtures,
    load_video_generation_result,
    load_video_qa_result,
    run_video_qa,
)
from agents.video_generation.mock_generator import MockVideoGenerator
from pipeline.state import PipelineState
from video_qa_test_utils import healthy_pair, result_failed_scene, result_mock_with_uri

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_committed_fixtures_are_healthy_and_pass():
    plan, result = load_committed_fixtures()
    assert result.production_plan_id == plan.production_plan_id
    qa = run_video_qa(plan, result, config=VideoQAConfig(), run_id="vqa-demo")
    assert qa.overall_verdict == "PASS"
    assert len(qa.checks) == 15
    assert qa.target_ref == plan.production_plan_id


def test_mismatched_pairing_raises():
    plan_a, _ = healthy_pair(run_id="pair-a")
    _, result_b = healthy_pair(run_id="pair-b")
    with pytest.raises(VideoQAError):
        run_video_qa(plan_a, result_b)


def test_loaders_round_trip(tmp_path):
    plan, result = load_committed_fixtures()
    assert load_video_generation_result(DEFAULT_VIDEO_RESULT_PATH) == result
    qa = run_video_qa(plan, result, config=VideoQAConfig(), run_id="roundtrip")
    (tmp_path / "qa.json").write_text(qa.to_json(), encoding="utf-8")
    assert load_video_qa_result(tmp_path / "qa.json") == qa


def _write_scenario(tmp_path, scenario):
    plan, result = scenario
    (tmp_path / "plan.json").write_text(plan.to_json(), encoding="utf-8")
    (tmp_path / "result.json").write_text(result.to_json(), encoding="utf-8")
    return tmp_path / "plan.json", tmp_path / "result.json"


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "agents.qa.video", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_cli_defaults_write_artifact_and_advance_state(tmp_path):
    proc = _run_cli("--out", str(tmp_path))
    out = tmp_path / "vqa-demo" / "production" / "qa_result.json"
    assert proc.returncode == 0, proc.stderr
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["data"]["overall_verdict"] == "PASS"
    assert payload["run_id"] == "vqa-demo"
    state = PipelineState.load(tmp_path / "vqa-demo" / "state.json")
    assert state.current_stage.value == "quality_check"
    assert state.artifacts.get("quality_check")


def test_cli_explicit_run_id(tmp_path):
    plan, result = healthy_pair()
    _write_scenario(tmp_path, (plan, result))
    proc = _run_cli(
        "--plan", str(tmp_path / "plan.json"),
        "--result", str(tmp_path / "result.json"),
        "--run-id", "explicit-1",
        "--out", str(tmp_path),
    )
    assert proc.returncode == 0
    payload = json.loads(
        (tmp_path / "explicit-1" / "production" / "qa_result.json").read_text(encoding="utf-8")
    )
    assert payload["run_id"] == "explicit-1"
    assert payload["data"]["overall_verdict"] == "PASS"


def test_cli_review_and_reject_scenarios(tmp_path):
    plan, healthy_result = healthy_pair()
    review_plan, review_result = plan, result_mock_with_uri(plan, healthy_result)
    _write_scenario(tmp_path, (review_plan, review_result))
    proc = _run_cli(
        "--plan", str(tmp_path / "plan.json"),
        "--result", str(tmp_path / "result.json"),
        "--run-id", "review-1",
        "--out", str(tmp_path),
    )
    assert proc.returncode == 1
    payload = json.loads(
        (tmp_path / "review-1" / "production" / "qa_result.json").read_text(encoding="utf-8")
    )
    assert payload["data"]["overall_verdict"] == "REVIEW"

    reject_plan, reject_result = plan, result_failed_scene(plan, healthy_result)
    _write_scenario(tmp_path, (reject_plan, reject_result))
    proc = _run_cli(
        "--plan", str(tmp_path / "plan.json"),
        "--result", str(tmp_path / "result.json"),
        "--run-id", "reject-1",
        "--out", str(tmp_path),
    )
    assert proc.returncode == 1
    payload = json.loads(
        (tmp_path / "reject-1" / "production" / "qa_result.json").read_text(encoding="utf-8")
    )
    assert payload["data"]["overall_verdict"] == "REJECT"


def test_cli_previous_chains_qa_result(tmp_path):
    first_plan, first_result = healthy_pair()
    _write_scenario(tmp_path, (first_plan, first_result))
    first_proc = _run_cli(
        "--plan", str(tmp_path / "plan.json"),
        "--result", str(tmp_path / "result.json"),
        "--run-id", "chain-a",
        "--out", str(tmp_path),
    )
    assert first_proc.returncode == 0
    first_payload = json.loads(
        (tmp_path / "chain-a" / "production" / "qa_result.json").read_text(encoding="utf-8")
    )

    second_plan, _ = healthy_pair()
    second_result = MockVideoGenerator().generate(
        second_plan, run_id="chain-b", created_at=second_plan.created_at
    )
    _write_scenario(tmp_path, (second_plan, second_result))
    second_proc = _run_cli(
        "--plan", str(tmp_path / "plan.json"),
        "--result", str(tmp_path / "result.json"),
        "--run-id", "chain-b",
        "--out", str(tmp_path),
        "--previous", str(tmp_path / "chain-a" / "production" / "qa_result.json"),
    )
    assert second_proc.returncode == 0
    second_payload = json.loads(
        (tmp_path / "chain-b" / "production" / "qa_result.json").read_text(encoding="utf-8")
    )
    assert second_payload["data"]["resumed_from_qa_result_id"] == first_payload["data"][
        "qa_result_id"
    ]


def test_cli_missing_result_is_an_error(tmp_path):
    proc = _run_cli("--result", str(tmp_path / "missing.json"), "--out", str(tmp_path))
    assert proc.returncode == 2
    assert "not found" in proc.stderr
    assert not list(tmp_path.rglob("state.json"))


def test_cli_missing_plan_is_an_error(tmp_path):
    proc = _run_cli("--plan", str(tmp_path / "missing.json"), "--out", str(tmp_path))
    assert proc.returncode == 2
    assert "not found" in proc.stderr
    assert not list(tmp_path.rglob("state.json"))


def test_cli_missing_previous_is_an_error(tmp_path):
    plan, result = healthy_pair()
    _write_scenario(tmp_path, (plan, result))
    proc = _run_cli(
        "--plan", str(tmp_path / "plan.json"),
        "--result", str(tmp_path / "result.json"),
        "--run-id", "missing-prev",
        "--out", str(tmp_path),
        "--previous", str(tmp_path / "missing_prev.json"),
    )
    assert proc.returncode == 2
    assert "not found" in proc.stderr
    assert not (tmp_path / "missing-prev" / "production" / "qa_result.json").exists()
    assert not (tmp_path / "missing-prev" / "state.json").exists()


def test_cli_corrupt_result_exits_2_without_traceback(tmp_path):
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    proc = _run_cli("--result", str(tmp_path / "bad.json"), "--out", str(tmp_path))
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert "error" in proc.stderr
    assert not list(tmp_path.rglob("qa_result.json"))
    assert not list(tmp_path.rglob("state.json"))


def test_cli_corrupt_plan_exits_2_without_traceback(tmp_path):
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    proc = _run_cli("--plan", str(tmp_path / "bad.json"), "--out", str(tmp_path))
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert "error" in proc.stderr
    assert not list(tmp_path.rglob("qa_result.json"))
    assert not list(tmp_path.rglob("state.json"))


def test_cli_corrupt_previous_exits_2_without_traceback(tmp_path):
    plan, result = healthy_pair()
    _write_scenario(tmp_path, (plan, result))
    (tmp_path / "bad_prev.json").write_text("{not json", encoding="utf-8")
    proc = _run_cli(
        "--plan", str(tmp_path / "plan.json"),
        "--result", str(tmp_path / "result.json"),
        "--run-id", "corrupt-prev",
        "--out", str(tmp_path),
        "--previous", str(tmp_path / "bad_prev.json"),
    )
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert "error" in proc.stderr
    assert not (tmp_path / "corrupt-prev" / "production" / "qa_result.json").exists()
    assert not (tmp_path / "corrupt-prev" / "state.json").exists()


def test_cli_invalid_json_schema_exits_2_without_traceback(tmp_path):
    (tmp_path / "empty.json").write_text("{}", encoding="utf-8")
    proc = _run_cli("--result", str(tmp_path / "empty.json"), "--out", str(tmp_path))
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert "error" in proc.stderr
    assert not list(tmp_path.rglob("qa_result.json"))
    assert not list(tmp_path.rglob("state.json"))


def test_cli_loader_round_trip(tmp_path):
    plan, result = healthy_pair()
    (tmp_path / "result.json").write_text(result.to_json(), encoding="utf-8")
    loaded = load_video_generation_result(tmp_path / "result.json")
    assert loaded.video_generation_result_id == result.video_generation_result_id
    assert loaded.production_plan_id == result.production_plan_id
    assert loaded.overall_status == result.overall_status
    assert [row.scene_id for row in loaded.scenes] == [row.scene_id for row in result.scenes]
    assert [row.artifact.artifact_id for row in loaded.scenes if row.artifact] == [
        row.artifact.artifact_id for row in result.scenes if row.artifact
    ]