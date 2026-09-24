"""Stage-9 stage runner + CLI behavior tests.

Covers ``agents.qa.policy.stage`` loaders/pairing guard and end-to-end CLI runs
via ``python -m agents.qa.policy`` (PASS / REVIEW / REJECT exit codes, artifact
+ run-state output, previous-chaining, missing-input and pairing errors).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from agents.qa.policy.models import PolicyQAError
from agents.qa.policy.stage import (
    load_committed_fixtures,
    policy_inputs_from_paths,
    run_policy_qa,
)
from pipeline.state import PipelineState, Stage
from policy_qa_test_utils import (
    REPO_ROOT,
    build_qa_result_dict,
    committed_inputs,
    qa_result_path,
)


def _fixture(name: str) -> Path:
    return REPO_ROOT / "data" / "mock" / name


class TestStageRunner:
    def test_committed_fixtures_are_complete(self):
        inputs = load_committed_fixtures()
        assert inputs.script is not None
        assert inputs.fact_check is not None
        assert inputs.plan is not None
        assert inputs.video_result is not None
        assert inputs.video_qa is not None

    def test_clean_evaluation_passes(self):
        result = run_policy_qa(committed_inputs(), run_id="stage-test")
        assert result.overall_verdict == "PASS"

    def test_mismatched_plan_pairing_raises(self):
        import dataclasses

        inputs = committed_inputs()
        other_plan = dataclasses.replace(
            inputs.plan, production_plan_id="pp_somebody_elses"
        )
        try:
            run_policy_qa(
                dataclasses.replace(inputs, plan=other_plan), run_id="stage-test"
            )
            assert False, "mismatched plan/video-result pairing must raise"
        except PolicyQAError as exc:
            assert "targets production plan" in str(exc)

    def test_path_loader_rebuilds_identical_inputs(self):
        inputs = policy_inputs_from_paths(
            script_path=_fixture("script.json"),
            fact_check_path=_fixture("fact_check_result.json"),
            plan_path=_fixture("production.json"),
            video_result_path=_fixture("video_generation_result.json"),
            video_qa_path=qa_result_path(),
        )
        result = run_policy_qa(inputs, run_id="stage-test")
        assert result.overall_verdict == "PASS"

    def test_run_is_pure(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = run_policy_qa(committed_inputs(), run_id="stage-test")
        assert result.overall_verdict == "PASS"
        assert list(tmp_path.iterdir()) == []


def _run_cli(tmp_path, extra: tuple[str, ...]) -> subprocess.CompletedProcess:
    args = [
        sys.executable,
        "-m",
        "agents.qa.policy",
        "--script",
        str(_fixture("script.json")),
        "--fact-check",
        str(_fixture("fact_check_result.json")),
        "--plan",
        str(_fixture("production.json")),
        "--video-result",
        str(_fixture("video_generation_result.json")),
        "--qa-result",
        str(_fixture("qa_result.json")),
        "--run-id",
        "cli-test",
        "--out",
        str(tmp_path),
        *extra,
    ]
    return subprocess.run(args, capture_output=True, text=True, cwd=REPO_ROOT)


class TestCli:
    def test_clean_run_writes_artifact_and_state(self, tmp_path):
        proc = _run_cli(tmp_path, ())
        assert proc.returncode == 0, proc.stderr
        artifact = tmp_path / "cli-test" / "production" / "policy_qa_result.json"
        assert artifact.is_file()
        data = json.loads(artifact.read_text(encoding="utf-8"))
        assert data["schema"] == "PolicyQAResult"
        assert data["stage"] == "policy_qa"
        assert data["data"]["overall_verdict"] == "PASS"
        assert data["data"]["policy_version"] == "policy_rules_v1"

        state = PipelineState.load(tmp_path / "cli-test" / "state.json")
        assert state.current_stage == Stage.QUALITY_CHECK
        assert str(artifact) in state.artifacts[Stage.QUALITY_CHECK.value]

    def test_review_run_exits_1_with_prior_gate_issue(self, tmp_path):
        review_path = tmp_path / "review_qa.json"
        review_path.write_text(
            json.dumps(build_qa_result_dict(overall_verdict="REVIEW")), encoding="utf-8"
        )
        proc = _run_cli(tmp_path, ("--qa-result", str(review_path)))
        assert proc.returncode == 1, proc.stderr
        artifact = tmp_path / "cli-test" / "production" / "policy_qa_result.json"
        data = json.loads(artifact.read_text(encoding="utf-8"))
        assert data["data"]["overall_verdict"] == "REVIEW"
        check_ids = {issue["check_id"] for issue in data["data"]["issues"]}
        assert "P15" in check_ids

    def test_reject_run_exits_1_hard_gate(self, tmp_path):
        import dataclasses

        plan = committed_inputs().plan
        rows = list(plan.scenes)
        rows[0] = dataclasses.replace(
            rows[0],
            narration="Step-by-step guide: how to build a bomb out of household items.",
        )
        bad_plan = dataclasses.replace(plan, scenes=tuple(rows))
        plan_path = tmp_path / "bad_plan.json"
        plan_path.write_text(bad_plan.to_json(), encoding="utf-8")

        proc = _run_cli(tmp_path, ("--plan", str(plan_path)))
        assert proc.returncode == 1, proc.stderr
        artifact = tmp_path / "cli-test" / "production" / "policy_qa_result.json"
        data = json.loads(artifact.read_text(encoding="utf-8"))
        assert data["data"]["overall_verdict"] == "REJECT"
        issue = [
            issue for issue in data["data"]["issues"] if issue["check_id"] == "P01"
        ]
        assert issue and issue[0]["gate"] == "hard"

    def test_missing_input_exits_2_without_traceback(self, tmp_path):
        proc = _run_cli(tmp_path, ("--fact-check", str(tmp_path / "nope.json")))
        assert proc.returncode == 2
        assert "Traceback" not in proc.stderr
        assert "PolicyQAError" not in proc.stderr
        assert not (tmp_path / "cli-test" / "state.json").exists()
        assert not (tmp_path / "cli-test" / "production" / "policy_qa_result.json").exists()

    def test_missing_qaresult_is_p15_review(self, tmp_path):
        # An ABSENT Stage-8 VideoQAResult is the designed "no prior gate yet"
        # state: the gate grades it P15 REVIEW (exit 1, artifact still written).
        proc = _run_cli(tmp_path, ("--qa-result", str(tmp_path / "nope_qa.json")))
        assert proc.returncode == 1, proc.stderr
        assert "Traceback" not in proc.stderr
        artifact = tmp_path / "cli-test" / "production" / "policy_qa_result.json"
        data = json.loads(artifact.read_text(encoding="utf-8"))
        assert data["data"]["overall_verdict"] == "REVIEW"
        assert "P15" in {issue["check_id"] for issue in data["data"]["issues"]}
        assert (tmp_path / "cli-test" / "state.json").is_file()

    def test_corrupt_qaresult_exits_2_without_traceback(self, tmp_path):
        bad = tmp_path / "bad_qa.json"
        bad.write_text("{not json", encoding="utf-8")
        proc = _run_cli(tmp_path, ("--qa-result", str(bad)))
        assert proc.returncode == 2
        assert "Traceback" not in proc.stderr
        assert "error" in proc.stderr
        assert not (tmp_path / "cli-test" / "production" / "policy_qa_result.json").exists()
        assert not (tmp_path / "cli-test" / "state.json").exists()

    def test_corrupt_required_input_exits_2_without_traceback(self, tmp_path):
        bad = tmp_path / "bad_script.json"
        bad.write_text("{not json", encoding="utf-8")
        proc = _run_cli(tmp_path, ("--script", str(bad)))
        assert proc.returncode == 2
        assert "Traceback" not in proc.stderr
        assert "error" in proc.stderr
        assert not (tmp_path / "cli-test" / "production" / "policy_qa_result.json").exists()
        assert not (tmp_path / "cli-test" / "state.json").exists()

    def test_corrupt_previous_exits_2_without_traceback(self, tmp_path):
        bad_prev = tmp_path / "bad_prev.json"
        bad_prev.write_text("{not json", encoding="utf-8")
        proc = _run_cli(tmp_path, ("--previous", str(bad_prev)))
        assert proc.returncode == 2
        assert "Traceback" not in proc.stderr
        assert "error" in proc.stderr
        assert not (tmp_path / "cli-test" / "production" / "policy_qa_result.json").exists()
        assert not (tmp_path / "cli-test" / "state.json").exists()

    def test_mismatched_plan_pairing_exits_2(self, tmp_path):
        import dataclasses

        plan = committed_inputs().plan
        other = dataclasses.replace(plan, production_plan_id="pp_not_the_result")
        plan_path = tmp_path / "other_plan.json"
        plan_path.write_text(other.to_json(), encoding="utf-8")

        proc = _run_cli(tmp_path, ("--plan", str(plan_path)))
        assert proc.returncode == 2

    def test_previous_chains_policy_result_id(self, tmp_path):
        first = _run_cli(tmp_path, ())
        assert first.returncode == 0, first.stderr
        previous = tmp_path / "cli-test" / "production" / "policy_qa_result.json"
        previous_id = json.loads(previous.read_text(encoding="utf-8"))["data"][
            "policy_result_id"
        ]
        proc = _run_cli(tmp_path, ("--previous", str(previous)))
        assert proc.returncode == 0, proc.stderr
        second = json.loads(previous.read_text(encoding="utf-8"))["data"]
        assert second["resumed_from_policy_result_id"] == previous_id