"""Stage-6 stage runner + CLI integration tests."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from agents.scene_planner.stage import (
    generate_mock_production_plan,
    generate_mock_qa_result,
    load_fact_check_result,
    load_script,
    load_script_qa_result,
    run_scene_planning,
)
from agents.script.models import ScriptError
from pipeline.state import PipelineState, Stage
from script_test_utils import make_script
from script_qa_test_utils import (
    make_octopus_pair,
    octopus_ok_lines,
    load_mock_package,
)
from scene_planning_test_utils import FIXED_AT

REPO_ROOT = Path(__file__).resolve().parents[1]


def healthy_inputs(run_id="scene-test"):
    result, package = make_octopus_pair(run_id=run_id, created_at=FIXED_AT)
    script = make_script(result, lines=octopus_ok_lines(result), created_at=FIXED_AT)
    from agents.qa.script.stage import run_script_qa

    qa = run_script_qa(script, result, package=package, run_id=run_id, created_at=FIXED_AT)
    assert qa.overall_verdict == "PASS"
    return script, qa, result, package


class TestStageEntrypoint:
    def test_pass_plans(self):
        script, qa, result, package = healthy_inputs()
        planning = run_scene_planning(script, qa, result, package=package, run_id="scene-test", created_at=FIXED_AT)
        assert planning.planned is True
        assert planning.plan is not None
        assert planning.validation.passed
        assert planning.blocked is False

    def test_revise_is_blocked_before_planning(self):
        script, qa, result, package = healthy_inputs()
        qa = replace(qa, overall_verdict="REVISE")
        planning = run_scene_planning(script, qa, result, package=package, run_id="scene-test")
        assert planning.planned is False
        assert planning.plan is None
        assert "QA gate" in planning.reason
        assert planning.blocked is True

    def test_reject_is_blocked_before_planning(self):
        script, qa, result, package = healthy_inputs()
        qa = replace(qa, overall_verdict="REJECT")
        planning = run_scene_planning(script, qa, result, package=package, run_id="scene-test")
        assert planning.planned is False
        assert planning.blocked is True

    def test_accepts_claims_sequence_directly(self):
        from agents.script.stage import approved_claims

        script, qa, result, package = healthy_inputs()
        claims = approved_claims(result, package)
        planning = run_scene_planning(script, qa, result, claims=claims, run_id="scene-test", created_at=FIXED_AT)
        assert planning.planned is True

    def test_raises_without_claim_records(self):
        script, qa, result, package = healthy_inputs()
        with pytest.raises(ScriptError):
            run_scene_planning(script, qa, result, claims=())

    def test_plan_records_qa_gate_id(self):
        script, qa, result, package = healthy_inputs()
        planning = run_scene_planning(script, qa, result, package=package, run_id="scene-test", created_at=FIXED_AT)
        assert planning.plan.script_qa_result_id == qa.qa_result_id
        assert planning.plan.script_id == script.script_id


class TestMockFixtures:
    def test_mock_qa_limit_reproducible(self):
        reproduce = generate_mock_qa_result(created_at=load_script().created_at)
        committed = load_script_qa_result()
        assert reproduce.to_dict() == committed.to_dict()

    def test_mock_qa_is_pass(self):
        qa = load_script_qa_result()
        assert qa.overall_verdict == "PASS"
        assert qa.qa_result_id.startswith("sqa_")

    def test_generate_mock_production_plan(self):
        plan = generate_mock_production_plan(created_at=load_script().created_at)
        assert plan.production_plan_id.startswith("pp_")
        assert plan.run_id == "mock"
        assert len(plan.scenes) == 7
        assert plan.total_duration_seconds == pytest.approx(45.2, abs=0.01)


class TestCliIntegration:
    def test_cli_writes_plan_and_state(self, tmp_path):
        out = tmp_path / "out"
        proc = subprocess.run(
            [sys.executable, "-m", "agents.scene_planner", "--out", str(out)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert proc.returncode == 0, proc.stderr
        artifact = out / "sp-demo" / "production" / "production.json"
        state_path = out / "sp-demo" / "state.json"
        assert artifact.is_file()
        assert state_path.is_file()
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        assert payload["schema"] == "ProductionPlan"
        assert payload["stage"] == "scene_planning"
        state = _load_state(state_path)
        assert state.current_stage == Stage.SCENE_PLANNING
        assert "production_plan_id" in json.dumps(payload["data"])

    def test_cli_blocks_on_revise_qa(self, tmp_path):
        out = tmp_path / "blocked"
        revise = tmp_path / "revise.json"
        qa = load_script_qa_result()
        reduce = replace(qa, overall_verdict="REVISE", score=60.0)
        revise.write_text(reduce.to_json(), encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, "-m", "agents.scene_planner",
             "--qa-result", str(revise), "--out", str(out)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert proc.returncode != 0
        assert "BLOCKED" in proc.stdout
        assert not (out / "sp-demo" / "production").exists()

    def test_cli_fails_on_missing_artifact(self, tmp_path):
        proc = subprocess.run(
            [sys.executable, "-m", "agents.scene_planner",
             "--script", str(tmp_path / "nope.json")],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert proc.returncode != 0
        assert "not found" in proc.stderr


def _load_state(path):
    return PipelineState.load(path)