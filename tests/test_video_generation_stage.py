"""Stage-7 stage runner + CLI integration tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from agents.video_generation.generator import VideoGenerator
from agents.video_generation.models import (
    SCENE_STATUS_FAILED,
    SCENE_STATUS_PENDING,
    RESULT_STATUS_COMPLETED,
    RESULT_STATUS_PARTIAL,
    GenerationFailure,
    VideoGenerationConfig,
    VideoGenerationResult,
    video_generation_result_id_for,
)
from agents.video_generation.mock_generator import MockVideoGenerator
from agents.video_generation.stage import (
    DEFAULT_PRODUCTION_PLAN_PATH,
    generate_mock_production_plan,
    load_production_plan,
    load_video_generation_result,
    run_video_generation,
)
from pipeline.state import PipelineState, Stage

from video_generation_test_utils import FIXED_AT, make_healthy_plan

REPO_ROOT = Path(__file__).resolve().parents[1]


class BrokenGenerator:
    """A (deliberately broken) VideoGenerator: drops plan scene 7."""

    def generate(self, plan, config=None, previous=None, run_id="", created_at=""):
        rows = []
        for scene in plan.scenes[:-1]:
            from agents.video_generation.models import (
                GenerationRequest,
                SceneGenerationResult,
            )

            request = GenerationRequest.from_scene(scene)
            rows.append(
                SceneGenerationResult(
                    scene_id=scene.scene_id,
                    order=scene.order,
                    status=SCENE_STATUS_PENDING,
                    request=request,
                    attempts=0,
                )
            )
        return VideoGenerationResult(
            video_generation_result_id=video_generation_result_id_for(
                run_id, plan.production_plan_id
            ),
            production_plan_id=plan.production_plan_id,
            topic_id=plan.topic_id,
            topic_title=plan.topic_title,
            script_id=plan.script_id,
            script_qa_result_id=plan.script_qa_result_id,
            scenes=tuple(rows),
            overall_status="partial",
            run_id=run_id,
            created_at=created_at or FIXED_AT,
        )


class TestStageEntrypoint:
    def test_healthy_plan_generates(self):
        plan = make_healthy_plan()
        outcome = run_video_generation(plan, run_id="vg-demo", created_at=FIXED_AT)
        assert outcome.generated is True
        assert outcome.result is not None
        assert outcome.result.overall_status == RESULT_STATUS_COMPLETED
        assert outcome.validation.passed

    def test_outcome_reports_blocked_on_invalid_result(self):
        plan = make_healthy_plan()
        outcome = run_video_generation(plan, generator=BrokenGenerator(), run_id="vg-demo")
        assert outcome.generated is False
        assert outcome.result is None or outcome.result is not None
        assert "validation failed" in outcome.reason
        assert outcome.validation.passed is False

    def test_partial_result_is_valid_when_statuses_are_sound(self):
        plan = make_healthy_plan()
        config = VideoGenerationConfig(
            failure_scenarios=(GenerationFailure(scene_id=plan.scenes[0].scene_id),)
        )
        outcome = run_video_generation(plan, config=config, run_id="vg-demo")
        assert outcome.generated is True
        assert outcome.result.overall_status == RESULT_STATUS_PARTIAL

    def test_previous_plan_mismatch_raises(self):
        plan_a = make_healthy_plan(run_id="video-a")
        plan_b = make_healthy_plan(run_id="video-b")
        previous = MockVideoGenerator().generate(plan_b, run_id="vg-b")
        try:
            run_video_generation(plan_a, previous=previous)
            raised = False
        except ValueError:
            raised = True
        assert raised

    def test_resume_via_stage(self):
        plan = make_healthy_plan()
        first = run_video_generation(
            plan,
            config=VideoGenerationConfig(
                failure_scenarios=(GenerationFailure(scene_id=plan.scenes[2].scene_id),)
            ),
            run_id="vg-demo",
        )
        assert first.generated and first.result.overall_status == RESULT_STATUS_PARTIAL
        second = run_video_generation(plan, previous=first.result, run_id="vg-demo")
        assert second.generated
        assert second.result.overall_status == RESULT_STATUS_COMPLETED
        assert second.result.resumed_from_result_id == first.result.video_generation_result_id


class TestLoaders:
    def test_generate_mock_production_plan(self):
        # Lazy import: a Stage-6 test reloads agents.scene_planner.models, which
        # would otherwise give us a stale class reference for isinstance checks.
        from agents.scene_planner.models import ProductionPlan

        plan = generate_mock_production_plan(created_at="2026-09-20T12:00:00.000000+00:00")
        assert isinstance(plan, ProductionPlan)
        assert plan.run_id == "mock"
        assert len(plan.scenes) == 7

    def test_load_production_plan_from_committed_fixture(self):
        plan = load_production_plan(DEFAULT_PRODUCTION_PLAN_PATH)
        assert plan.run_id == "mock"
        assert len(plan.scenes) == 7

    def test_result_loader_round_trip(self, tmp_path):
        plan = make_healthy_plan()
        outcome = run_video_generation(plan, run_id="vg-demo", created_at=FIXED_AT)
        path = tmp_path / "video_generation_result.json"
        path.write_text(outcome.result.to_json(), encoding="utf-8")
        loaded = load_video_generation_result(path)
        assert loaded.to_dict() == outcome.result.to_dict()


class TestCliIntegration:
    def test_cli_writes_result_and_state(self, tmp_path):
        out = tmp_path / "out"
        proc = subprocess.run(
            [sys.executable, "-m", "agents.video_generation", "--out", str(out)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert proc.returncode == 0, proc.stderr
        artifact = out / "vg-demo" / "videos" / "video_generation_result.json"
        state_path = out / "vg-demo" / "state.json"
        assert artifact.is_file()
        assert state_path.is_file()
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        assert payload["schema"] == "VideoGenerationResult"
        assert payload["data"]["overall_status"] == "completed"
        assert payload["data"]["video_generation_result_id"].startswith("vg_")
        state = PipelineState.load(state_path)
        assert state.current_stage == Stage.VIDEO_GENERATION

    def test_cli_run_id_flag(self, tmp_path):
        out = tmp_path / "out"
        proc = subprocess.run(
            [sys.executable, "-m", "agents.video_generation",
             "--run-id", "my-run", "--out", str(out)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert proc.returncode == 0, proc.stderr
        assert (out / "my-run" / "videos" / "video_generation_result.json").is_file()

    def test_cli_explicit_plan(self, tmp_path):
        out = tmp_path / "out"
        proc = subprocess.run(
            [sys.executable, "-m", "agents.video_generation",
             "--plan", str(DEFAULT_PRODUCTION_PLAN_PATH), "--out", str(out)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert proc.returncode == 0, proc.stderr
        # Passing the default fixture path counts as the demo invocation, so the
        # run id is 'vg-demo' (keeps the committed data/mock fixture clean).
        assert (out / "vg-demo" / "videos" / "video_generation_result.json").is_file()

    def test_cli_resume(self, tmp_path):
        out = tmp_path / "out"
        first = subprocess.run(
            [sys.executable, "-m", "agents.video_generation", "--out", str(out)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert first.returncode == 0, first.stderr
        artifact = out / "vg-demo" / "videos" / "video_generation_result.json"
        second = subprocess.run(
            [sys.executable, "-m", "agents.video_generation",
             "--previous", str(artifact), "--out", str(out)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert second.returncode == 0, second.stderr
        assert "resumed_from" in second.stdout
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        assert payload["data"]["resumed_from_result_id"].startswith("vg_")

    def test_cli_errors_on_missing_plan(self):
        proc = subprocess.run(
            [sys.executable, "-m", "agents.video_generation",
             "--plan", str(Path("does-not-exist.json"))],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert proc.returncode != 0
        assert "not found" in proc.stderr

    def test_cli_errors_on_missing_previous(self, tmp_path):
        proc = subprocess.run(
            [sys.executable, "-m", "agents.video_generation",
             "--previous", str(tmp_path / "nope.json")],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert proc.returncode != 0
        assert "not found" in proc.stderr