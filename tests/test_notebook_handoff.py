"""Offline tests for the Colab handoff worker notebook and its core logic.

Everything here is offline: temp directories simulate the mounted Drive, dummy
bytes stand in for PNGs, and the GPU gate is exercised with stub facts. The
notebook itself is never executed (no Colab, no torch, no GPU); its static
structure, vendored sources and reproducibility are checked instead.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import nbformat
import pytest

from agents.image_jobs import handoff
from agents.image_jobs.jobs import JOB_SCHEMA_VERSION
from notebooks import build_notebook
from notebooks import handoff_notebook_core as core

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = build_notebook.NOTEBOOK_PATH
HANDOFF_SOURCE = build_notebook.HANDOFF_SOURCE
CORE_SOURCE = build_notebook.CORE_SOURCE

RUN_ID = "nb-run-01"
ATTEMPT_ID = "nb-run-01-attempt"
RESOLUTION = (768, 1344)
SENTINEL_VALUE = "test-sentinel-value-0001"


# --- helpers -----------------------------------------------------------------


def _job(scene_id: str, *, run_id: str = RUN_ID, asset_id=None) -> dict:
    asset_id = asset_id if asset_id is not None else f"{scene_id}_main"
    return {
        "scene_id": scene_id,
        "positive_prompt": "style, coins on a desk",
        "negative_prompt": "blurry",
        "seed": 7,
        "asset_id": asset_id,
        "output_path": f"data/{run_id}/assets/{asset_id}.png",
    }


def _make_handoff_root(base: Path, *, name: str = "ai-youtube-channel-handoff") -> Path:
    root = base / name
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_sentinel(
    root: Path,
    value: str = SENTINEL_VALUE,
    *,
    project: str = "ai-youtube-channel",
    rootname: str = "ai-youtube-channel-handoff",
    version: int = 1,
) -> Path:
    path = root / core.SENTINEL_FILENAME
    path.write_text(
        json.dumps(
            {
                "project": project,
                "root": rootname,
                "sentinel_version": version,
                "verification_value": value,
            }
        ),
        encoding="utf-8",
    )
    return path


def _make_input(
    root: Path,
    *,
    run_id: str = RUN_ID,
    attempt_id: str = ATTEMPT_ID,
    jobs=None,
):
    jobs = jobs if jobs is not None else [_job("sc01")]
    input_dir = root / run_id / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    jobs_path = input_dir / "image_jobs.json"
    jobs_path.write_text(
        json.dumps({"schema": JOB_SCHEMA_VERSION, "run_id": run_id, "jobs": jobs}),
        encoding="utf-8",
    )
    manifest = handoff.build_input_manifest(
        run_id, jobs_path, attempt_id=attempt_id, expected_resolution=RESOLUTION
    )
    handoff.dump_json(manifest, input_dir / "input_manifest.json")
    handoff.write_marker(input_dir, "READY", content=attempt_id)
    return manifest


def _dummy_pngs(tmp_path: Path, asset_ids) -> dict:
    out = {}
    for asset_id in asset_ids:
        path = tmp_path / f"{asset_id}.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + asset_id.encode("utf-8"))
        out[asset_id] = path
    return out


# --- root discovery ----------------------------------------------------------


def test_discover_root_ok(tmp_path):
    root = _make_handoff_root(tmp_path)
    assert core.discover_handoff_root(tmp_path) == root


def test_discover_root_missing_fails_closed(tmp_path):
    with pytest.raises(core.NotebookHandoffError):
        core.discover_handoff_root(tmp_path)


def test_discover_root_case_mismatch_fails_closed(tmp_path):
    _make_handoff_root(tmp_path, name="AI-YouTube-Channel-Handoff")
    with pytest.raises(core.NotebookHandoffError):
        core.discover_handoff_root(tmp_path)


def test_discover_root_not_a_directory_fails_closed(tmp_path):
    (tmp_path / "ai-youtube-channel-handoff").write_text("not a dir", encoding="utf-8")
    with pytest.raises(core.NotebookHandoffError):
        core.discover_handoff_root(tmp_path)


def test_discover_root_missing_base_fails_closed(tmp_path):
    with pytest.raises(core.NotebookHandoffError):
        core.discover_handoff_root(tmp_path / "does-not-exist")


def test_discover_root_duplicate_candidates_fails_closed(tmp_path, monkeypatch):
    # Windows filesystems cannot hold two names differing only by case, so the
    # >1-candidate branch is exercised via a synthetic directory listing.
    class _Entry:
        def __init__(self, name):
            self.name = name

    base = tmp_path
    real_iterdir = Path.iterdir

    def fake_iterdir(self):
        if self == base:
            return iter(
                [_Entry("ai-youtube-channel-handoff"), _Entry("AI-YOUTUBE-CHANNEL-HANDOFF")]
            )
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", fake_iterdir)
    with pytest.raises(core.NotebookHandoffError):
        core.discover_handoff_root(base)


# --- sentinel trust ----------------------------------------------------------


def test_verify_sentinel_ok(tmp_path):
    root = _make_handoff_root(tmp_path)
    _write_sentinel(root)
    doc = core.verify_sentinel(root, SENTINEL_VALUE)
    assert doc["verification_value"] == SENTINEL_VALUE


def test_verify_sentinel_missing_value_fails_closed(tmp_path):
    root = _make_handoff_root(tmp_path)
    _write_sentinel(root)
    with pytest.raises(core.NotebookHandoffError):
        core.verify_sentinel(root, "")


def test_verify_sentinel_missing_file_fails_closed(tmp_path):
    root = _make_handoff_root(tmp_path)
    with pytest.raises(core.NotebookHandoffError):
        core.verify_sentinel(root, SENTINEL_VALUE)


def test_verify_sentinel_mismatch_fails_closed(tmp_path):
    root = _make_handoff_root(tmp_path)
    _write_sentinel(root, value="other-value")
    with pytest.raises(core.NotebookHandoffError):
        core.verify_sentinel(root, SENTINEL_VALUE)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"project": "someone-else"},
        {"rootname": "other-root"},
        {"version": 2},
    ],
)
def test_verify_sentinel_wrong_fields_fail_closed(tmp_path, kwargs):
    root = _make_handoff_root(tmp_path)
    _write_sentinel(root, **kwargs)
    with pytest.raises(core.NotebookHandoffError):
        core.verify_sentinel(root, SENTINEL_VALUE)


def test_verify_sentinel_malformed_json_fails_closed(tmp_path):
    root = _make_handoff_root(tmp_path)
    (root / core.SENTINEL_FILENAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(core.NotebookHandoffError):
        core.verify_sentinel(root, SENTINEL_VALUE)


def test_verify_sentinel_non_object_fails_closed(tmp_path):
    root = _make_handoff_root(tmp_path)
    (root / core.SENTINEL_FILENAME).write_text("[]", encoding="utf-8")
    with pytest.raises(core.NotebookHandoffError):
        core.verify_sentinel(root, SENTINEL_VALUE)


# --- explicit run selection + input validation -------------------------------


def test_verify_run_input_ok(tmp_path):
    root = _make_handoff_root(tmp_path)
    manifest = _make_input(root)
    assert core.verify_run_input(root, RUN_ID)["attempt_id"] == manifest["attempt_id"]


@pytest.mark.parametrize("bad_run_id", ["", "  ", "../escape", "a/b", "..", "a\\b"])
def test_select_run_rejects_unsafe_ids(tmp_path, bad_run_id):
    root = _make_handoff_root(tmp_path)
    with pytest.raises(core.NotebookHandoffError):
        core.select_run(root, bad_run_id)


def test_select_run_missing_run_fails_closed(tmp_path):
    root = _make_handoff_root(tmp_path)
    with pytest.raises(core.NotebookHandoffError):
        core.select_run(root, RUN_ID)


def test_verify_run_input_missing_ready_fails_closed(tmp_path):
    root = _make_handoff_root(tmp_path)
    _make_input(root)
    (root / RUN_ID / "input" / "READY").unlink()
    with pytest.raises(handoff.HandoffError):
        core.verify_run_input(root, RUN_ID)


def test_verify_run_input_stale_ready_fails_closed(tmp_path):
    root = _make_handoff_root(tmp_path)
    _make_input(root)
    handoff.write_marker(root / RUN_ID / "input", "READY", content="a-different-attempt")
    with pytest.raises(handoff.HandoffError):
        core.verify_run_input(root, RUN_ID)


def test_verify_run_input_tampered_jobs_fails_closed(tmp_path):
    root = _make_handoff_root(tmp_path)
    _make_input(root)
    jobs_path = root / RUN_ID / "input" / "image_jobs.json"
    jobs_path.write_text(
        json.dumps(
            {"schema": JOB_SCHEMA_VERSION, "run_id": RUN_ID, "jobs": [_job("sc02")]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(handoff.HandoffError):
        core.verify_run_input(root, RUN_ID)


# --- GPU preflight gate ------------------------------------------------------


def _facts(**overrides) -> dict:
    facts = {
        "cuda_available": True,
        "device_count": 1,
        "gpu_name": "Tesla T4",
        "total_vram_gb": 15.0,
        "free_vram_gb": 12.0,
        "cuda_op_ok": True,
        "nvidia_smi": "Tesla T4",
    }
    facts.update(overrides)
    return facts


def test_preflight_ok():
    decision = core.preflight_decision(_facts())
    assert decision["ok"] is True
    assert decision["error_code"] is None


def test_preflight_no_gpu():
    assert core.preflight_decision(
        _facts(cuda_available=False, device_count=0, total_vram_gb=None, free_vram_gb=None)
    )["error_code"] == "NO_GPU"


def test_preflight_cuda_init_failed():
    assert core.preflight_decision(_facts(cuda_op_ok=False))["error_code"] == "CUDA_INIT_FAILED"


@pytest.mark.parametrize(
    "overrides",
    [
        {"total_vram_gb": 12.0},        # below the 14 GiB total threshold
        {"free_vram_gb": 8.0},          # below the 10 GiB free threshold
        {"total_vram_gb": None},        # unknown -> fail closed
        {"free_vram_gb": None},         # unknown -> fail closed
        {"total_vram_gb": float("nan")},  # non-finite -> fail closed
        {"free_vram_gb": float("inf")},   # non-finite -> fail closed
    ],
)
def test_preflight_insufficient_vram(overrides):
    decision = core.preflight_decision(_facts(**overrides))
    assert decision["ok"] is False
    assert decision["error_code"] == "INSUFFICIENT_VRAM"


# --- prompt token budget -----------------------------------------------------


@pytest.mark.parametrize(
    "longest,expected_kind",
    [(10, None), (60, None), (61, "warning"), (77, "warning"), (78, "error")],
)
def test_prompt_token_finding(longest, expected_kind):
    kind, _message = core.prompt_token_finding(longest)
    assert kind == expected_kind


# --- result document ---------------------------------------------------------


def test_build_result_document_schema():
    batch = {"run_id": RUN_ID, "channel_id": "finance_mascot", "title": "T", "jobs": [_job("sc01")]}
    results = [
        {
            "scene_id": "sc01",
            "asset_id": "sc01_main",
            "filename": "sc01_main.png",
            "status": "generated",
            "seed": 7,
            "width": RESOLUTION[0],
            "height": RESOLUTION[1],
        }
    ]
    doc = core.build_result_document(
        RUN_ID,
        batch,
        results,
        resolution=RESOLUTION,
        generation_config=handoff.default_generation_config(*RESOLUTION),
    )
    assert doc["schema"] == handoff.SCHEMA_RESULT_MANIFEST
    assert doc["status"] == "completed"
    assert doc["generated_count"] == 1
    assert doc["failed_count"] == 0
    assert doc["resolution"] == {"width": 768, "height": 1344}


# --- progress / failure status ----------------------------------------------


def test_write_progress_status_running(tmp_path):
    status = core.write_progress_status(
        tmp_path,
        run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        image_jobs_sha256="a" * 64,
        generation_config=handoff.default_generation_config(*RESOLUTION),
        completed_stages=["gpu_preflight"],
        current_stage="env_setup",
    )
    assert status["overall_status"] == "RUNNING"
    handoff.validate_status(
        json.loads((tmp_path / handoff.STATUS_FILENAME).read_text(encoding="utf-8"))
    )


@pytest.mark.parametrize(
    "code,expected_overall",
    [("GENERATION_FAILED", "BLOCKED"), ("OUTPUT_VALIDATION_FAILED", "FAILED")],
)
def test_write_failure_status(tmp_path, code, expected_overall):
    stage = handoff.ERROR_STAGE_HINTS[code]
    status = core.write_failure_status(
        tmp_path,
        run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        image_jobs_sha256="a" * 64,
        generation_config=handoff.default_generation_config(*RESOLUTION),
        code=code,
        stage=stage,
        message="boom",
    )
    assert status["overall_status"] == expected_overall
    handoff.validate_status(
        json.loads((tmp_path / handoff.STATUS_FILENAME).read_text(encoding="utf-8"))
    )


# --- output packaging + marker-last publication ------------------------------


def _publish_kwargs(tmp_path, asset_ids=("sc01_main",), generated=None, expected=None):
    generated = _dummy_pngs(tmp_path, asset_ids if generated is None else generated)
    expected = list(asset_ids if expected is None else expected)
    return {
        "run_id": RUN_ID,
        "attempt_id": ATTEMPT_ID,
        "image_jobs_sha256": "b" * 64,
        "generated_files": generated,
        "result_document": {
            "schema": handoff.SCHEMA_RESULT_MANIFEST,
            "run_id": RUN_ID,
            "status": "completed",
            "results": [],
        },
        "expected_asset_ids": expected,
        "resolution": RESOLUTION,
        "generation_config": handoff.default_generation_config(*RESOLUTION),
    }


def test_publish_outputs_creates_valid_zip_and_done(tmp_path):
    output_dir = tmp_path / "output"
    manifest = core.publish_outputs(output_dir, **_publish_kwargs(tmp_path))

    assert manifest["schema"] == handoff.SCHEMA_OUTPUT_MANIFEST
    zip_path = output_dir / f"{RUN_ID}{core.OUTPUT_ZIP_SUFFIX}"
    assert zip_path.is_file()
    with zipfile.ZipFile(zip_path) as archive:
        assert sorted(archive.namelist()) == sorted(["sc01_main.png", "result.json"])

    # The output is fetchable via the existing contract verifier.
    handoff.verify_done(
        output_dir,
        run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        image_jobs_sha256="b" * 64,
        expected_asset_ids=["sc01_main"],
    )
    assert handoff.read_marker(output_dir, handoff.DONE_MARKER) == ATTEMPT_ID


def test_publish_outputs_zip_name(tmp_path):
    output_dir = tmp_path / "output"
    core.publish_outputs(output_dir, **_publish_kwargs(tmp_path))
    assert (output_dir / f"{RUN_ID}_generated_assets.zip").is_file()


def test_publish_outputs_marker_is_written_last(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    seen = {}
    real_write_marker = handoff.write_marker

    def checking_write_marker(directory, name, *, content=""):
        if name == handoff.DONE_MARKER:
            # Everything DONE refers to must already exist.
            seen["manifest"] = (Path(directory) / handoff.OUTPUT_MANIFEST_FILENAME).is_file()
            seen["status"] = (Path(directory) / handoff.STATUS_FILENAME).is_file()
            seen["zip"] = (Path(directory) / f"{RUN_ID}{core.OUTPUT_ZIP_SUFFIX}").is_file()
        return real_write_marker(directory, name, content=content)

    monkeypatch.setattr(handoff, "write_marker", checking_write_marker)
    core.publish_outputs(output_dir, **_publish_kwargs(tmp_path))
    assert seen == {"manifest": True, "status": True, "zip": True}


def test_publish_outputs_rejects_incomplete_asset_set(tmp_path):
    output_dir = tmp_path / "output"
    kwargs = _publish_kwargs(tmp_path, asset_ids=("sc01_main",), expected=["sc01_main", "sc02_main"])
    with pytest.raises(core.NotebookHandoffError):
        core.publish_outputs(output_dir, **kwargs)
    assert not (output_dir / handoff.DONE_MARKER).exists()


def test_publish_outputs_missing_file_writes_no_done(tmp_path):
    output_dir = tmp_path / "output"
    kwargs = _publish_kwargs(tmp_path)
    kwargs["generated_files"]["sc01_main"] = tmp_path / "does-not-exist.png"
    with pytest.raises(core.NotebookHandoffError):
        core.publish_outputs(output_dir, **kwargs)
    assert not (output_dir / handoff.DONE_MARKER).exists()


def test_publish_outputs_clears_stale_completion(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True)
    (output_dir / handoff.DONE_MARKER).write_text("stale-attempt", encoding="utf-8")
    (output_dir / handoff.OUTPUT_MANIFEST_FILENAME).write_text("{}", encoding="utf-8")

    core.publish_outputs(output_dir, **_publish_kwargs(tmp_path))

    assert handoff.read_marker(output_dir, handoff.DONE_MARKER) == ATTEMPT_ID


# --- producer ↔ contract cross-check (result document) ------------------------


def _producer_result_document(*, failed: bool = False) -> dict:
    batch = {
        "run_id": RUN_ID,
        "channel_id": "finance_mascot",
        "title": "T",
        "jobs": [_job("sc01")],
    }
    results = [
        {
            "scene_id": "sc01",
            "asset_id": "sc01_main",
            "filename": "sc01_main.png",
            "status": "failed" if failed else "generated",
            "seed": 7,
            "width": RESOLUTION[0],
            "height": RESOLUTION[1],
        }
    ]
    return core.build_result_document(
        RUN_ID,
        batch,
        results,
        resolution=RESOLUTION,
        generation_config=handoff.default_generation_config(*RESOLUTION),
    )


def test_published_producer_document_satisfies_contract(tmp_path):
    """The document the notebook really writes passes the CLI verifier."""
    output_dir = tmp_path / "output"
    kwargs = _publish_kwargs(tmp_path)
    kwargs["result_document"] = _producer_result_document()
    manifest = core.publish_outputs(output_dir, **kwargs)

    handoff.verify_zip_result_document(
        output_dir / f"{RUN_ID}{core.OUTPUT_ZIP_SUFFIX}",
        output_manifest=manifest,
        run_id=RUN_ID,
    )


def test_producer_partial_failure_document_rejected_for_published_artifact(tmp_path):
    doc = _producer_result_document(failed=True)
    assert doc["status"] == "partial_failure"
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_result_document(
            doc,
            expected_run_id=RUN_ID,
            expected_asset_ids=["sc01_main"],
            expected_resolution=RESOLUTION,
            expected_generation_config=handoff.default_generation_config(*RESOLUTION),
            require_completed=True,
        )
    assert exc.value.code == "hf50"


def test_minimal_result_document_fails_contract_despite_valid_publish(tmp_path):
    # publish_outputs packages whatever document it is handed; the consumption
    # boundary (validate/import) is what fails closed on a non-contract one.
    output_dir = tmp_path / "output"
    manifest = core.publish_outputs(output_dir, **_publish_kwargs(tmp_path))
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.verify_zip_result_document(
            output_dir / f"{RUN_ID}{core.OUTPUT_ZIP_SUFFIX}",
            output_manifest=manifest,
            run_id=RUN_ID,
        )
    assert exc.value.code == "hf49"


# --- Issue 1: rerun lifecycle / stale completion -----------------------------

PUBLISH_SHA = "b" * 64


def _seed_valid_completion(tmp_path, output_dir, *, attempt_id=ATTEMPT_ID):
    """Publish a genuine, verifiable output so a prior DONE exists."""
    core.publish_outputs(output_dir, **_publish_kwargs(tmp_path))
    handoff.verify_done(
        output_dir,
        run_id=RUN_ID,
        attempt_id=attempt_id,
        image_jobs_sha256=PUBLISH_SHA,
        expected_asset_ids=["sc01_main"],
    )
    return output_dir


def _assert_no_stale_success(output_dir):
    assert handoff.read_marker(output_dir, handoff.DONE_MARKER) is None
    assert not (output_dir / handoff.OUTPUT_MANIFEST_FILENAME).exists()
    assert not (output_dir / handoff.STATUS_FILENAME).exists()
    with pytest.raises(handoff.HandoffError):
        handoff.verify_done(
            output_dir,
            run_id=RUN_ID,
            attempt_id=ATTEMPT_ID,
            image_jobs_sha256=PUBLISH_SHA,
            expected_asset_ids=["sc01_main"],
        )


def test_invalidate_stale_completion_clears_prior_success(tmp_path):
    output_dir = _seed_valid_completion(tmp_path, tmp_path / "output")
    core.invalidate_stale_completion(output_dir, attempt_id=ATTEMPT_ID)
    _assert_no_stale_success(output_dir)


def test_invalidate_stale_completion_keeps_zip_artifact(tmp_path):
    output_dir = _seed_valid_completion(tmp_path, tmp_path / "output")
    zip_path = output_dir / f"{RUN_ID}{core.OUTPUT_ZIP_SUFFIX}"
    assert zip_path.is_file()
    core.invalidate_stale_completion(output_dir, attempt_id=ATTEMPT_ID)
    assert zip_path.is_file()  # artifacts are preserved, only completion is cleared
    _assert_no_stale_success(output_dir)


def test_invalidate_stale_completion_requires_attempt_id(tmp_path):
    with pytest.raises(core.NotebookHandoffError):
        core.invalidate_stale_completion(tmp_path / "output", attempt_id="")


@pytest.mark.parametrize(
    "code,completed_stages",
    [
        ("NO_GPU", []),
        ("INSUFFICIENT_VRAM", []),
        ("GENERATION_FAILED", ["gpu_preflight", "env_setup", "model_load"]),
        (
            "OUTPUT_VALIDATION_FAILED",
            ["gpu_preflight", "env_setup", "model_load", "generation"],
        ),
    ],
)
def test_failed_rerun_with_invalidation_never_exposes_stale_success(
    tmp_path, code, completed_stages
):
    run_dir = tmp_path / "run"
    output_dir = run_dir / "output"
    status_dir = run_dir / "status"
    _seed_valid_completion(tmp_path, output_dir)

    # The worker's lifecycle: invalidate first, then fail at a later stage.
    core.invalidate_stale_completion(output_dir, attempt_id=ATTEMPT_ID)
    status = core.write_failure_status(
        status_dir,
        run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        image_jobs_sha256=PUBLISH_SHA,
        generation_config=handoff.default_generation_config(*RESOLUTION),
        code=code,
        stage=handoff.ERROR_STAGE_HINTS[code],
        message="synthetic failure",
        completed_stages=completed_stages,
    )

    _assert_no_stale_success(output_dir)
    assert status["attempt_id"] == ATTEMPT_ID
    assert status["overall_status"] == handoff.overall_status_for_error(code)
    written = json.loads(
        (status_dir / handoff.STATUS_FILENAME).read_text(encoding="utf-8")
    )
    assert written["overall_status"] == handoff.overall_status_for_error(code)


def test_failed_rerun_without_invalidation_would_expose_stale_success(tmp_path):
    """Documents the hazard the early invalidation removes.

    With no early invalidation, a stale ``DONE`` survives a failed rerun and
    still verifies for the (unchanged) attempt id.
    """
    run_dir = tmp_path / "run"
    output_dir = run_dir / "output"
    status_dir = run_dir / "status"
    _seed_valid_completion(tmp_path, output_dir)

    # A GPU failure that only writes the progress status (no invalidation).
    core.write_failure_status(
        status_dir,
        run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        image_jobs_sha256=PUBLISH_SHA,
        generation_config=handoff.default_generation_config(*RESOLUTION),
        code="NO_GPU",
        stage="gpu_preflight",
        message="synthetic failure",
        completed_stages=[],
    )

    # The old completion still verifies for the unchanged attempt id: the bug.
    handoff.verify_done(
        output_dir,
        run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        image_jobs_sha256=PUBLISH_SHA,
        expected_asset_ids=["sc01_main"],
    )


def test_notebook_invalidates_completion_before_gpu_preflight():
    cells = [cell.source for cell in _notebook().cells if cell.cell_type == "code"]
    invalidate_idx = next(
        i for i, s in enumerate(cells) if "invalidate_stale_completion(" in s
    )
    gpu_idx = next(i for i, s in enumerate(cells) if "GPU PREFLIGHT" in s)
    assert invalidate_idx < gpu_idx


# --- Issue 2: prompt-token gate + documented status incompatibility ----------


def test_prompt_validation_findings_over_limit_is_error():
    errors, warnings = core.prompt_validation_findings([_job("sc01")], lambda _p: 78)
    assert errors and not warnings


def test_prompt_validation_findings_empty_positive_prompt_is_error():
    job = _job("sc01")
    job["positive_prompt"] = "   "
    errors, _warnings = core.prompt_validation_findings([job], lambda _p: 5)
    assert any("empty positive prompt" in message for message in errors)


def test_prompt_validation_findings_warns_above_soft_target():
    errors, warnings = core.prompt_validation_findings([_job("sc01")], lambda _p: 61)
    assert not errors and warnings


def test_prompt_validation_findings_ok():
    errors, warnings = core.prompt_validation_findings([_job("sc01")], lambda _p: 40)
    assert errors == [] and warnings == []


def test_input_validation_failed_code_represents_prompt_defects():
    """Phase 2.2/2.3 replaces the documented Issue 2 gap.

    A prompt-token failure is a deterministic *input* defect, so it maps to
    ``INPUT_VALIDATION_FAILED``: an unrecoverable (``FAILED``) code valid at the
    pre-generation ``env_setup`` stage, distinct from the recoverable
    infrastructure ``ENV_SETUP_FAILED``.
    """
    assert "INPUT_VALIDATION_FAILED" in handoff.ERROR_CODES
    assert "INPUT_VALIDATION_FAILED" in handoff.UNRECOVERABLE_ERROR_CODES
    assert "INPUT_VALIDATION_FAILED" not in handoff.RECOVERABLE_ERROR_CODES
    assert handoff.ERROR_STAGE_HINTS["INPUT_VALIDATION_FAILED"] == "env_setup"
    assert handoff.overall_status_for_error("INPUT_VALIDATION_FAILED") == "FAILED"
    assert handoff.ERROR_STAGE_HINTS["OUTPUT_VALIDATION_FAILED"] == "output_validation"
    assert "ENV_SETUP_FAILED" in handoff.RECOVERABLE_ERROR_CODES
    assert handoff.ERROR_STAGE_HINTS["ENV_SETUP_FAILED"] == "env_setup"


def test_prompt_gate_precedes_model_load_and_writes_a_terminal_status():
    cells = [cell.source for cell in _notebook().cells if cell.cell_type == "code"]
    prompt_idx = next(i for i, s in enumerate(cells) if "PROMPT TOKEN CHECK" in s)
    model_idx = next(i for i, s in enumerate(cells) if "MODEL LOAD" in s)
    assert prompt_idx < model_idx

    prompt_cell = cells[prompt_idx]
    # The gate cell must not download/construct the heavy pipeline.
    for forbidden in (
        "StableDiffusionXLPipeline",
        "UNet2DConditionModel",
        "hf_hub_download",
        "diffusers",
    ):
        assert forbidden not in prompt_cell
    # On a prompt defect it must write the terminal FAILED status via the shared
    # fail helper (never a progress status, never publish, never DONE).
    assert "fail(" in prompt_cell
    assert '"INPUT_VALIDATION_FAILED"' in prompt_cell
    assert '"env_setup"' in prompt_cell
    assert 'completed_stages=["gpu_preflight"]' in prompt_cell
    for forbidden in (
        "publish_outputs",
        "write_progress_status",
        "write_marker",
        "DONE_MARKER",
    ):
        assert forbidden not in prompt_cell


def test_prompt_failure_surfaces_after_invalidation_before_model_load():
    cells = [cell.source for cell in _notebook().cells if cell.cell_type == "code"]
    invalidate_idx = next(
        i for i, s in enumerate(cells) if "invalidate_stale_completion(" in s
    )
    prompt_idx = next(i for i, s in enumerate(cells) if "PROMPT TOKEN CHECK" in s)
    model_idx = next(i for i, s in enumerate(cells) if "MODEL LOAD" in s)
    assert invalidate_idx < prompt_idx < model_idx


def test_input_validation_failure_is_terminal_and_leaves_no_done(tmp_path):
    run_dir = tmp_path / "run"
    output_dir = run_dir / "output"
    status_dir = run_dir / "status"
    _seed_valid_completion(tmp_path, output_dir)

    core.invalidate_stale_completion(output_dir, attempt_id=ATTEMPT_ID)
    status = core.write_failure_status(
        status_dir,
        run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        image_jobs_sha256=PUBLISH_SHA,
        generation_config=handoff.default_generation_config(*RESOLUTION),
        code="INPUT_VALIDATION_FAILED",
        stage="env_setup",
        message="prompt validation failed",
        diagnostics={"prompt_errors": ["sc01 | positive_prompt: 78 tokens"]},
        completed_stages=["gpu_preflight"],
    )

    assert status["overall_status"] == "FAILED"
    assert status["error"]["code"] == "INPUT_VALIDATION_FAILED"
    assert status["error"]["stage"] == "env_setup"
    assert status["error"]["recoverable"] is False
    assert status["error"]["diagnostics"]["prompt_errors"]
    assert status["completed_stages"] == ["gpu_preflight"]

    written = json.loads(
        (status_dir / handoff.STATUS_FILENAME).read_text(encoding="utf-8")
    )
    handoff.validate_status(written)
    _assert_no_stale_success(output_dir)


# --- notebook contract integrity / drift -------------------------------------


def _notebook():
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


def _vendored_sources():
    code_cells = [cell.source for cell in _notebook().cells if cell.cell_type == "code"]
    return code_cells[0], code_cells[1], code_cells[2]


def test_notebook_is_valid_nbformat():
    nb = _notebook()
    nbformat.validate(nb)
    assert nb.nbformat == 4
    assert any(cell.cell_type == "markdown" for cell in nb.cells)


def test_vendored_handoff_matches_source():
    _shim, vendored_handoff, _vendored_core = _vendored_sources()
    assert vendored_handoff == HANDOFF_SOURCE.read_text(encoding="utf-8")


def test_vendored_core_matches_source():
    _shim, _vendored_handoff, vendored_core = _vendored_sources()
    assert vendored_core == CORE_SOURCE.read_text(encoding="utf-8")


def test_shim_uses_real_jobs_schema_version():
    shim, _handoff_cell, _core_cell = _vendored_sources()
    assert f"_JOBS_SCHEMA = {JOB_SCHEMA_VERSION!r}" in shim


def test_vendored_sources_compile():
    shim, vendored_handoff, vendored_core = _vendored_sources()
    for source in (shim, vendored_handoff, vendored_core):
        compile(source, "<vendored>", "exec")


def test_notebook_is_reproducible_from_sources():
    on_disk = _notebook()
    rebuilt = build_notebook.build_notebook(commit=on_disk.metadata["aiyt_vendored_commit"])
    assert rebuilt == on_disk


def test_recorded_commit_present():
    commit = _notebook().metadata.get("aiyt_vendored_commit")
    assert isinstance(commit, str) and commit
    assert commit == "unknown" or len(commit) == 40


# --- vendored bundle runs flat (the notebook's actual execution mode) --------


def test_vendored_bundle_executes_flat_and_smoke_passes(tmp_path):
    """Execute the vendored cells exactly as the notebook does (flat, no repo).

    Runs in a fresh interpreter with no repository on the path, so the core's
    flat fallback (`_handoff = sys.modules[__name__]`) is genuinely exercised.
    """
    script = f"""
import json, sys, types
from pathlib import Path
import nbformat

nb = nbformat.read(r"{NOTEBOOK_PATH}", as_version=4)
code = [c.source for c in nb.cells if c.cell_type == "code"]
mod = types.ModuleType("aiyt_nb_runtime")
mod.__dict__["__name__"] = "aiyt_nb_runtime"
sys.modules["aiyt_nb_runtime"] = mod
for source in code[:3]:
    exec(compile(source, "<cell>", "exec"), mod.__dict__)

assert mod.__dict__["_handoff"] is mod, "flat fallback was not used"

ns = mod.__dict__
tmp = Path(r"{tmp_path}")
base = tmp / "drive"
base.mkdir(parents=True)
root = base / "ai-youtube-channel-handoff"
root.mkdir()
ns["discover_handoff_root"](base)
(root / ns["SENTINEL_FILENAME"]).write_text(
    json.dumps({{"project": "ai-youtube-channel", "root": "ai-youtube-channel-handoff",
                 "sentinel_version": 1, "verification_value": "V1"}}), encoding="utf-8")
ns["verify_sentinel"](root, "V1")
assert ns["preflight_decision"]({{
    "cuda_available": True, "device_count": 1, "gpu_name": "T4",
    "total_vram_gb": 15.0, "free_vram_gb": 12.0, "cuda_op_ok": True, "nvidia_smi": "T4",
}})["ok"] is True
assert ns["prompt_token_finding"](78)[0] == "error"
print("FLAT_OK")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    assert completed.returncode == 0, completed.stderr
    assert "FLAT_OK" in completed.stdout


def test_contract_helper_is_not_a_reimplementation():
    # The core module must rely on the shared contract, not duplicate it.
    assert core._handoff is handoff
    assert issubclass(core.NotebookHandoffError, ValueError)
