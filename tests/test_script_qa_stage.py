"""Stage-5 QA stage + CLI integration tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from agents.qa.script.models import ScriptQAConfig, ScriptQAResult
from agents.qa.script.stage import (
    DEFAULT_FACT_CHECK_RESULT_PATH,
    DEFAULT_SCRIPT_PATH,
    approved_claims,
    claims_index,
    generate_mock_script,
    load_fact_check_result,
    load_script,
    run_script_qa,
)
from agents.script.models import ScriptError
from pipeline.state import PipelineState, Stage
from script_qa_test_utils import load_mock_package

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_fixture_artifacts_load():
    assert DEFAULT_SCRIPT_PATH.is_file()
    script = load_script()
    result = load_fact_check_result()
    assert script.result_id == result.result_id
    assert script.topic_id == result.topic_id


def test_stage_end_to_end_pass():
    script = load_script()
    result = load_fact_check_result()
    package = load_mock_package()
    qa = run_script_qa(script, result, package=package)
    assert qa.overall_verdict == "PASS"
    assert qa.score == 100.0
    assert qa.approved_claim_coverage.coverage_ratio == 1.0
    assert isinstance(qa, ScriptQAResult)


def test_stage_accepts_claims_sequence_directly():
    script = load_script()
    result = load_fact_check_result()
    package = load_mock_package()
    approved = approved_claims(result, package)
    qa = run_script_qa(script, result, claims=approved)
    assert qa.overall_verdict == "PASS"


def test_stage_raises_without_claim_records():
    script = load_script()
    result = load_fact_check_result()
    with pytest.raises(ScriptError):
        run_script_qa(script, result, claims=())


def test_generate_mock_script_reproduces_fixture():
    script = load_script()
    regenerated = generate_mock_script(created_at=script.created_at)
    assert regenerated.to_dict() == script.to_dict()


def test_claims_index_covers_package():
    package = load_mock_package()
    index = claims_index(package)
    assert {c.claim_id for c in package.claims} == set(index)


def test_cli_writes_artifact_and_advances_state(tmp_path):
    out = tmp_path / "out"
    cmd = [
        sys.executable,
        "-m",
        "agents.qa.script",
        "--script", str(DEFAULT_SCRIPT_PATH),
        "--result", str(DEFAULT_FACT_CHECK_RESULT_PATH),
        "--out", str(out),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    # Default demo invocation keeps the committed data/mock/ fixtures clean,
    # so the run id falls back to the CLI demo id.
    artifact_path = out / "sqa-demo" / "scripts" / "script_qa_result.json"
    assert artifact_path.is_file()
    qa = ScriptQAResult.from_json(artifact_path.read_text(encoding="utf-8"))
    assert qa.overall_verdict == "PASS"

    state = PipelineState.load(out / "sqa-demo" / "state.json")
    assert state.current_stage == Stage.SCRIPT_QA
    assert any(str(artifact_path) in value for value in state.artifacts.values())


def test_cli_rejects_missing_artifacts(tmp_path):
    cmd = [
        sys.executable,
        "-m",
        "agents.qa.script",
        "--script", str(tmp_path / "nope.json"),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode != 0
    assert "not found" in proc.stderr


def test_stage_is_network_free_monkeypatched():
    import socket

    script = load_script()
    result = load_fact_check_result()
    package = load_mock_package()

    def boom(*args, **kwargs):
        raise AssertionError("QL stage opened the network")

    original = socket.socket
    socket.socket = boom
    try:
        qa = run_script_qa(script, result, package=package)
    finally:
        socket.socket = original
    assert qa.overall_verdict == "PASS"