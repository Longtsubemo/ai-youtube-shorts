"""Stage-10 stage tests: loaders, pairing guards, purity, and the manual CLI.

The Stage-10 stage must (1) only ever review a coherent, self-consistent
bundle (the plan the policy was run against, the QA it was run against, the
video that was produced from that plan, and the script that plan was built
from), (2) be a pure function (never write files), and (3) have a manual CLI
that writes the HumanApprovalResult artifact + advances pipeline state to the
existing generic publish stage.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agents.human_approval.models import Approver, HumanApprovalError
from agents.human_approval.stage import (
    DEFAULT_APPROVAL_RESULT_NAME,
    approval_inputs_from_paths,
    load_approval_result,
    load_committed_fixtures,
    run_human_approval,
)
from human_approval_test_utils import (
    GENESIS,
    MOCK,
    REPO_ROOT,
    artifact_map,
    build_inputs,
    committed_inputs,
    qa_result_dict,
)
from pipeline.state import PipelineState, Stage
from agents.video_generation.models import VideoGenerationResult


def _fixture(name: str) -> Path:
    return MOCK / name


def _run_cli(tmp_path: Path, extra: tuple[str, ...]) -> subprocess.CompletedProcess:
    args = [
        sys.executable,
        "-m",
        "agents.human_approval",
        "--policy-result",
        str(_fixture("policy_qa_result.json")),
        "--qa-result",
        str(_fixture("qa_result.json")),
        "--video-result",
        str(_fixture("video_generation_result.json")),
        "--plan",
        str(_fixture("production.json")),
        "--script",
        str(_fixture("script.json")),
        "--script-qa",
        str(_fixture("script_qa_result.json")),
        "--fact-check",
        str(_fixture("fact_check_result.json")),
        "--run-id",
        "cli-test",
        "--out",
        str(tmp_path),
        *extra,
    ]
    return subprocess.run(args, capture_output=True, text=True, cwd=REPO_ROOT)


def _write_json(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _approval_file(tmp_path: Path) -> Path:
    return tmp_path / "cli-test" / "approval" / DEFAULT_APPROVAL_RESULT_NAME


class TestLoaders:
    def test_committed_fixtures_include_full_chain(self):
        inputs = load_committed_fixtures()
        assert inputs.policy_qa.policy_result_id == "pr_9b7d098451d3"
        assert inputs.video_qa.qa_result_id == "qa_9b7d098451d3"
        assert inputs.plan.production_plan_id == "pp_e99181e73b5d"
        assert inputs.script.script_id == "sc_9fc41d7a46b7"
        assert inputs.script_qa is not None
        assert inputs.fact_check is not None

    def test_path_loader_rebuilds_identical_bundle(self):
        inputs = approval_inputs_from_paths(
            policy_result_path=_fixture("policy_qa_result.json"),
            video_qa_path=_fixture("qa_result.json"),
            video_result_path=_fixture("video_generation_result.json"),
            plan_path=_fixture("production.json"),
            script_path=_fixture("script.json"),
            script_qa_path=_fixture("script_qa_result.json"),
            fact_check_path=_fixture("fact_check_result.json"),
        )
        assert artifact_map(inputs) == artifact_map(committed_inputs())

    def test_optional_roles_can_be_omitted(self):
        inputs = approval_inputs_from_paths(
            policy_result_path=_fixture("policy_qa_result.json"),
            video_qa_path=_fixture("qa_result.json"),
            video_result_path=_fixture("video_generation_result.json"),
            plan_path=_fixture("production.json"),
            script_path=_fixture("script.json"),
        )
        assert inputs.script_qa is None
        assert inputs.fact_check is None

    def test_committed_policy_fixture_is_still_consistent(self):
        from agents.qa.policy.stage import load_committed_fixtures as load_policy_fixtures
        from agents.qa.policy.validator import evaluate_policy

        result = evaluate_policy(load_policy_fixtures(), run_id="mock")
        assert result.overall_verdict == "PASS"
        assert result.policy_result_id == "pr_9b7d098451d3"


class TestPairingGuards:
    def test_mismatched_plan_raises(self):
        import dataclasses

        inputs = build_inputs()
        broken = dataclasses.replace(
            inputs, plan=dataclasses.replace(inputs.plan, production_plan_id="pp_other")
        )
        with pytest.raises(HumanApprovalError, match="does not match policy target_ref"):
            run_human_approval(
                broken,
                decision="APPROVED",
                approver=Approver(approver_id="alice", role="reviewer"),
                run_id="stage-test",
            )

    def test_mismatched_video_qa_raises(self):
        import dataclasses

        inputs = build_inputs()
        broken = dataclasses.replace(
            inputs, video_qa=dataclasses.replace(inputs.video_qa, target_ref="pp_other")
        )
        with pytest.raises(HumanApprovalError, match="targets"):
            run_human_approval(
                broken,
                decision="APPROVED",
                approver=Approver(approver_id="alice", role="reviewer"),
                run_id="stage-test",
            )

    def test_mismatched_video_result_raises(self):
        import dataclasses

        inputs = build_inputs()
        broken = dataclasses.replace(
            inputs,
            video_result=dataclasses.replace(
                inputs.video_result, production_plan_id="pp_other"
            ),
        )
        with pytest.raises(HumanApprovalError, match="targets production plan"):
            run_human_approval(
                broken,
                decision="APPROVED",
                approver=Approver(approver_id="alice", role="reviewer"),
                run_id="stage-test",
            )

    def test_mismatched_script_raises(self):
        import dataclasses

        inputs = build_inputs()
        broken = dataclasses.replace(
            inputs, script=dataclasses.replace(inputs.script, script_id="sc_other")
        )
        with pytest.raises(HumanApprovalError, match="expects script"):
            run_human_approval(
                broken,
                decision="APPROVED",
                approver=Approver(approver_id="alice", role="reviewer"),
                run_id="stage-test",
            )

    def test_healthy_bundle_records_approval(self):
        result = run_human_approval(
            build_inputs(),
            decision="APPROVED",
            approver=Approver(approver_id="alice", role="reviewer"),
            run_id="stage-test",
            created_at=GENESIS,
        )
        assert result.is_approved
        assert result.created_at == GENESIS


class TestRunPurityAndRoundTrip:
    def test_run_is_pure(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = run_human_approval(
            committed_inputs(),
            decision="APPROVED",
            approver=Approver(approver_id="alice", role="reviewer"),
            run_id="stage-test",
            created_at=GENESIS,
        )
        assert result.is_approved
        assert list(tmp_path.iterdir()) == []

    def test_loader_round_trip(self, tmp_path):
        result = run_human_approval(
            committed_inputs(),
            decision="APPROVED",
            approver=Approver(approver_id="alice", role="reviewer"),
            run_id="stage-test",
            created_at=GENESIS,
        )
        path = tmp_path / DEFAULT_APPROVAL_RESULT_NAME
        path.write_text(result.to_json(), encoding="utf-8")
        restored = load_approval_result(path)
        assert restored == result
        assert restored.to_json() == result.to_json()


class TestCli:
    def test_approve_writes_artifact_and_advances_state(self, tmp_path):
        proc = _run_cli(
            tmp_path,
            ("--decision", "APPROVED", "--approver-id", "alice", "--notes", "ready"),
        )
        assert proc.returncode == 0, proc.stderr
        artifact = _approval_file(tmp_path)
        assert artifact.is_file()
        data = json.loads(artifact.read_text(encoding="utf-8"))
        assert data["schema"] == "HumanApprovalResult"
        assert data["stage"] == "human_approval"
        assert data["data"]["decision"] == "APPROVED"
        assert data["data"]["approver"]["approver_id"] == "alice"
        assert len(data["data"]["reviewed_artifacts"]) == 7

        state = PipelineState.load(tmp_path / "cli-test" / "state.json")
        assert state.current_stage == Stage.PUBLISH
        assert str(artifact) in state.artifacts[Stage.PUBLISH.value]

    def test_changes_requested_exit_1(self, tmp_path):
        proc = _run_cli(
            tmp_path,
            (
                "--decision",
                "CHANGES_REQUESTED",
                "--approver-id",
                "alice",
                "--notes",
                "fix the disclosure first",
            ),
        )
        assert proc.returncode == 1, proc.stderr
        data = json.loads(_approval_file(tmp_path).read_text(encoding="utf-8"))
        assert data["data"]["decision"] == "CHANGES_REQUESTED"
        assert data["data"]["manual_upload"]["ready"] is False

    def test_changes_requested_via_change_flag(self, tmp_path):
        proc = _run_cli(
            tmp_path,
            (
                "--decision",
                "CHANGES_REQUESTED",
                "--approver-id",
                "alice",
                "--change",
                "P13: add the synthetic-content disclosure",
            ),
        )
        assert proc.returncode == 1, proc.stderr
        data = json.loads(_approval_file(tmp_path).read_text(encoding="utf-8"))
        assert data["data"]["changes_requested"][0]["requirement"].startswith("P13:")

    def test_rejected_exit_1(self, tmp_path):
        proc = _run_cli(
            tmp_path,
            (
                "--decision",
                "REJECTED",
                "--approver-id",
                "alice",
                "--notes",
                "not fit to publish",
            ),
        )
        assert proc.returncode == 1, proc.stderr
        data = json.loads(_approval_file(tmp_path).read_text(encoding="utf-8"))
        assert data["data"]["decision"] == "REJECTED"

    def test_approve_blocked_on_hard_reject(self, tmp_path):
        reject_path = _write_json(
            tmp_path, "qa_reject.json", qa_result_dict(overall_verdict="REJECT")
        )
        proc = _run_cli(
            tmp_path,
            (
                "--decision",
                "APPROVED",
                "--approver-id",
                "alice",
                "--qa-result",
                str(reject_path),
            ),
        )
        assert proc.returncode == 2, proc.stdout
        assert not _approval_file(tmp_path).exists()

    def test_changes_requested_allowed_under_hard_reject(self, tmp_path):
        reject_path = _write_json(
            tmp_path, "qa_reject.json", qa_result_dict(overall_verdict="REJECT")
        )
        proc = _run_cli(
            tmp_path,
            (
                "--decision",
                "CHANGES_REQUESTED",
                "--approver-id",
                "alice",
                "--notes",
                "must fix the QA blockers first",
                "--qa-result",
                str(reject_path),
            ),
        )
        assert proc.returncode == 1, proc.stderr
        data = json.loads(_approval_file(tmp_path).read_text(encoding="utf-8"))
        blocked = {c["code"]: c for c in data["data"]["blocking_conditions"]}
        assert blocked["video_qa_hard_reject"]["satisfied"] is False
        assert data["data"]["decision"] == "CHANGES_REQUESTED"

    def test_approve_with_recorded_override(self, tmp_path):
        review_path = _write_json(
            tmp_path, "qa_review.json", qa_result_dict(overall_verdict="REVIEW")
        )
        proc = _run_cli(
            tmp_path,
            (
                "--decision",
                "APPROVED",
                "--approver-id",
                "alice",
                "--override",
                "video_qa;Q-PROVIDER_SUBJECT;REVIEW;accepted the disclosure warning",
                "--qa-result",
                str(review_path),
            ),
        )
        assert proc.returncode == 0, proc.stderr
        data = json.loads(_approval_file(tmp_path).read_text(encoding="utf-8"))
        assert data["data"]["overridden_verdicts"][0]["artifact_id"] == "qa_9b7d098451d3"
        assert data["data"]["manual_upload"]["ready"] is True

    def test_missing_decision_is_error(self, tmp_path):
        proc = _run_cli(tmp_path, ("--approver-id", "alice"))
        assert proc.returncode == 2

    def test_pending_is_not_writable(self, tmp_path):
        proc = _run_cli(tmp_path, ("--decision", "PENDING", "--approver-id", "alice"))
        assert proc.returncode == 2
        assert not _approval_file(tmp_path).exists()

    def test_missing_approver_is_error(self, tmp_path):
        proc = _run_cli(tmp_path, ("--decision", "APPROVED"))
        assert proc.returncode == 2

    def test_missing_input_is_error(self, tmp_path):
        proc = _run_cli(
            tmp_path,
            (
                "--decision",
                "APPROVED",
                "--approver-id",
                "alice",
                "--policy-result",
                str(tmp_path / "nope.json"),
            ),
        )
        assert proc.returncode == 2

    def test_supersede_chain_via_previous(self, tmp_path):
        first = _run_cli(
            tmp_path,
            ("--decision", "APPROVED", "--approver-id", "alice", "--notes", "v1"),
        )
        assert first.returncode == 0, first.stderr
        first_path = _approval_file(tmp_path)
        first_id = json.loads(first_path.read_text(encoding="utf-8"))["data"]["approval_id"]

        proc = _run_cli(
            tmp_path,
            (
                "--run-id",
                "cli-test2",
                "--decision",
                "APPROVED",
                "--approver-id",
                "bob",
                "--notes",
                "re-reviewed after fix",
                "--previous",
                str(first_path),
            ),
        )
        assert proc.returncode == 0, proc.stderr
        second_path = tmp_path / "cli-test2" / "approval" / DEFAULT_APPROVAL_RESULT_NAME
        second = json.loads(second_path.read_text(encoding="utf-8"))
        assert second["data"]["supersedes_approval_id"] == first_id
        assert second["data"]["approval_id"] != first_id

    def test_check_reports_effective_then_stale(self, tmp_path):
        approve = _run_cli(
            tmp_path,
            ("--decision", "APPROVED", "--approver-id", "alice", "--notes", "ok"),
        )
        assert approve.returncode == 0, approve.stderr
        approval_path = _approval_file(tmp_path)

        current = _run_cli(
            tmp_path, ("--check", "--approval", str(approval_path))
        )
        assert current.returncode == 0, current.stderr

        vg = VideoGenerationResult.from_json(
            _fixture("video_generation_result.json").read_text(encoding="utf-8")
        )
        import dataclasses

        rows = list(vg.scenes)
        rows[0] = dataclasses.replace(
            rows[0], artifact=dataclasses.replace(rows[0].artifact, note="edited later")
        )
        tampered_path = tmp_path / "vg_tampered.json"
        tampered_path.write_text(
            dataclasses.replace(vg, scenes=tuple(rows)).to_json(), encoding="utf-8"
        )

        stale = _run_cli(
            tmp_path,
            (
                "--check",
                "--approval",
                str(approval_path),
                "--video-result",
                str(tampered_path),
            ),
        )
        assert stale.returncode == 1, stale.stdout
        assert "digests_fresh=NO" in stale.stdout