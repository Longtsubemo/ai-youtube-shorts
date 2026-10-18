"""End-to-end CLI tests for ``tools/colab_handoff.py`` (Milestone 2).

Everything runs against temporary directories and synthetic fixtures: no
network, no Google Drive, no rclone, no GPU. The suite drives the CLI through
``main(argv)`` and asserts on exit codes, output and on-disk effects.

The Milestone 1 contract tests live in ``tests/test_colab_handoff.py``; this file
only covers the local CLI and is intentionally separate.
"""

from __future__ import annotations

import ast
import io
import json
import shutil
import zipfile
from pathlib import Path

import pytest
from PIL import Image

import tools.colab_handoff as cli
from agents.image_jobs.handoff import (
    STAGES,
    build_output_manifest,
    build_status,
    dump_json,
    make_error,
    write_marker,
)

RUN = "e2e-cli-run-01"
ASSETS = ["sc01_main", "sc02_main"]
PNG_SIZE = (768, 1344)


# --- fixtures / helpers -------------------------------------------------------


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    return tmp_path / "proj"


@pytest.fixture
def handoff_root(tmp_path: Path) -> Path:
    return tmp_path / "handoff"


def _write_jobs(project_root: Path, run_id: str = RUN, assets=None,
                output_path_fn=None) -> Path:
    assets = assets or ASSETS
    jobs = []
    for i, asset in enumerate(assets, start=1):
        if output_path_fn is not None:
            output_path = output_path_fn(asset)
        else:
            output_path = f"data/{run_id}/assets/{asset}.png"
        jobs.append(
            {
                "job_id": f"{run_id}-sc{i:02d}",
                "scene_id": f"sc{i:02d}",
                "order": i - 1,
                "duration_seconds": 8,
                "scene_generation_prompt": "a thing",
                "channel_image_style": "style",
                "positive_prompt": "style, a thing",
                "negative_prompt": "no",
                "seed": 100 + i,
                "asset_id": asset,
                "output_path": output_path,
            }
        )
    data = {"schema": "image-jobs/1.0", "run_id": run_id, "jobs": jobs, "warnings": []}
    path = project_root / "data" / run_id / "image_jobs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _png_bytes(size: tuple[int, int] = PNG_SIZE) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (12, 34, 56)).save(buf, "PNG")
    return buf.getvalue()


def _result_document(run_id: str = RUN, assets=None, *, status: str = "completed") -> dict:
    """A producer-faithful ``image-generation-results/1.0`` document.

    Mirrors ``build_result_document`` + the worker's per-item result shape.
    """
    assets = list(assets if assets is not None else ASSETS)
    results = [
        {
            "scene_id": f"sc{i:02d}",
            "asset_id": asset,
            "filename": f"{asset}.png",
            "status": "generated",
            "seed": 100 + i,
            "width": PNG_SIZE[0],
            "height": PNG_SIZE[1],
        }
        for i, asset in enumerate(assets, start=1)
    ]
    if status == "partial_failure" and results:
        results[-1] = dict(results[-1], status="failed")
    failed = sum(1 for item in results if item["status"] == "failed")
    return {
        "schema": "image-generation-results/1.0",
        "run_id": run_id,
        "source_json": "image_jobs.json",
        "channel_id": "finance_mascot",
        "title": "A video",
        "status": "completed" if failed == 0 else "partial_failure",
        "job_count": len(results),
        "generated_count": len(results) - failed,
        "skipped_count": 0,
        "failed_count": failed,
        "resolution": {"width": PNG_SIZE[0], "height": PNG_SIZE[1]},
        "steps": 4,
        "guidance_scale": 0.0,
        "results": results,
    }


def _make_zip(dest: Path, assets=ASSETS, *, size=PNG_SIZE, result=True,
              extra_members=None, dup_first=False, result_doc: dict | None = None,
              run_id: str = RUN) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w") as zf:
        first = None
        for asset in assets:
            data = _png_bytes(size)
            if first is None:
                first = data
            zf.writestr(f"{asset}.png", data)
        if result:
            document = result_doc if result_doc is not None else _result_document(
                run_id, assets
            )
            zf.writestr("result.json", json.dumps(document))
        for name in extra_members or []:
            zf.writestr(name, b"x")
        if dup_first and assets:
            zf.writestr(f"./{assets[0]}.png", first)
    return dest


def _read_input_manifest(handoff_root: Path, run_id: str = RUN) -> dict:
    path = handoff_root / run_id / "input" / "input_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _publish_output(handoff_root: Path, run_id: str, *, zip_path: Path,
                    members: list[str], generated: list[str], attempt: str | None = None,
                    manifest_run_id: str | None = None, status: dict | None = None,
                    completed_asset_ids: list[str] | None = None,
                    done_content: str | None = None) -> tuple[dict, dict]:
    """Simulate the Colab worker publishing a completed output."""
    in_manifest = _read_input_manifest(handoff_root, run_id)
    attempt = attempt if attempt is not None else in_manifest["attempt_id"]
    out_dir = handoff_root / run_id / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_output_manifest(
        manifest_run_id or run_id,
        attempt_id=attempt,
        image_jobs_sha256=in_manifest["image_jobs_sha256"],
        zip_path=zip_path,
        members=members,
        generated_asset_ids=generated,
    )
    dump_json(manifest, out_dir / "output_manifest.json")

    if status is None:
        status = build_status(
            run_id,
            attempt_id=attempt,
            image_jobs_sha256=in_manifest["image_jobs_sha256"],
            overall_status="PASS",
            current_stage=STAGES[-1],
            completed_stages=list(STAGES),
            completed_asset_ids=list(completed_asset_ids or in_manifest["expected_asset_ids"]),
        )
    dump_json(status, out_dir / "status.json")
    write_marker(out_dir, "DONE", content=done_content if done_content is not None else attempt)
    return manifest, status


def _complete_valid(handoff_root: Path, run_id: str = RUN) -> Path:
    """Publish a fully valid completed output and return the ZIP path."""
    out_dir = handoff_root / run_id / "output"
    zip_path = _make_zip(out_dir / f"{run_id}_assets.zip")
    in_manifest = _read_input_manifest(handoff_root, run_id)
    assets = list(in_manifest["expected_asset_ids"])
    _publish_output(
        handoff_root, run_id, zip_path=zip_path,
        members=[f"{a}.png" for a in assets] + ["result.json"],
        generated=assets,
    )
    return zip_path


def _complete_with_result(handoff_root: Path, result_doc: dict | None, *,
                          run_id: str = RUN, assets=ASSETS,
                          raw_result: str | None = None) -> Path:
    """Publish a completed output whose ZIP carries ``result_doc``/``raw_result``."""
    out_dir = handoff_root / run_id / "output"
    zip_path = out_dir / f"{run_id}_assets.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for asset in assets:
            zf.writestr(f"{asset}.png", _png_bytes())
        payload = raw_result if raw_result is not None else json.dumps(result_doc)
        zf.writestr("result.json", payload)
    _publish_output(
        handoff_root, run_id, zip_path=zip_path,
        members=[f"{a}.png" for a in assets] + ["result.json"],
        generated=assets,
    )
    return zip_path


# --- prepare ------------------------------------------------------------------


def test_prepare_creates_layout_and_writes_ready_last(project_root, handoff_root):
    src = _write_jobs(project_root)
    original = src.read_bytes()
    rc = cli.main(["--project-root", str(project_root), "--handoff-root", str(handoff_root),
                   "prepare", "--run-id", RUN])
    assert rc == 0

    input_dir = handoff_root / RUN / "input"
    assert (input_dir / "image_jobs.json").is_file()
    manifest = _read_input_manifest(handoff_root)
    ready = (input_dir / "READY").read_text(encoding="utf-8")
    assert ready == manifest["attempt_id"]
    progress = json.loads((handoff_root / RUN / "status" / "status.json").read_text(encoding="utf-8"))
    assert progress["overall_status"] == "PENDING"
    assert progress["attempt_id"] == manifest["attempt_id"]
    # source preserved; no assets dir created
    assert src.read_bytes() == original
    assert not (project_root / "data" / RUN / "assets").exists()


def test_prepare_rejects_run_id_mismatch(project_root, handoff_root, capsys):
    src = _write_jobs(project_root)  # placed under data/RUN
    data = json.loads(src.read_text(encoding="utf-8"))
    data["run_id"] = "some-other-run"  # inner contract run id disagrees
    for job in data["jobs"]:  # keep each job internally consistent
        job["output_path"] = f"data/some-other-run/assets/{job['asset_id']}.png"
    src.write_text(json.dumps(data), encoding="utf-8")
    rc = cli.main(["--project-root", str(project_root), "--handoff-root", str(handoff_root),
                   "prepare", "--run-id", RUN])
    assert rc == 2
    assert "hf04" in capsys.readouterr().err


def test_prepare_rejects_unsafe_output_path(project_root, handoff_root, capsys):
    _write_jobs(project_root, output_path_fn=lambda a: f"C:/evil/{a}.png")
    rc = cli.main(["--project-root", str(project_root), "--handoff-root", str(handoff_root),
                   "prepare", "--run-id", RUN])
    assert rc == 2
    assert "hf06" in capsys.readouterr().err


def test_prepare_rejects_wrong_parent_output_path(project_root, handoff_root, capsys):
    _write_jobs(project_root, output_path_fn=lambda a: f"data/{RUN}/elsewhere/{a}.png")
    rc = cli.main(["--project-root", str(project_root), "--handoff-root", str(handoff_root),
                   "prepare", "--run-id", RUN])
    assert rc == 2
    assert "hf07" in capsys.readouterr().err


def test_prepare_missing_source(project_root, handoff_root, capsys):
    rc = cli.main(["--project-root", str(project_root), "--handoff-root", str(handoff_root),
                   "prepare", "--run-id", RUN])
    assert rc == 2
    assert "hf01" in capsys.readouterr().err


def test_prepare_requires_force_then_new_attempt(project_root, handoff_root):
    _write_jobs(project_root)
    base = ["--project-root", str(project_root), "--handoff-root", str(handoff_root),
            "prepare", "--run-id", RUN]
    assert cli.main(base) == 0
    first = _read_input_manifest(handoff_root)["attempt_id"]

    assert cli.main(base) == 2  # already prepared
    assert cli.main(base + ["--force"]) == 0
    second = _read_input_manifest(handoff_root)["attempt_id"]
    assert second != first


# --- status -------------------------------------------------------------------


def test_status_reports_pending_after_prepare(project_root, handoff_root, capsys):
    _write_jobs(project_root)
    cli.main(["--project-root", str(project_root), "--handoff-root", str(handoff_root),
              "prepare", "--run-id", RUN])
    capsys.readouterr()
    rc = cli.main(["--project-root", str(project_root), "--handoff-root", str(handoff_root),
                   "status", "--run-id", RUN])
    out = capsys.readouterr().out
    assert rc == 0
    assert "input_ready=yes" in out
    assert "progress_status=PENDING" in out
    assert "output_done=no" in out


def test_status_reports_pass_after_completion(project_root, handoff_root, capsys):
    _write_jobs(project_root)
    cli.main(["--project-root", str(project_root), "--handoff-root", str(handoff_root),
              "prepare", "--run-id", RUN])
    _complete_valid(handoff_root)
    capsys.readouterr()
    rc = cli.main(["--project-root", str(project_root), "--handoff-root", str(handoff_root),
                   "status", "--run-id", RUN])
    out = capsys.readouterr().out
    assert rc == 0
    assert "output_done=yes" in out
    assert "output_status=PASS" in out


def test_status_missing_run(project_root, handoff_root, capsys):
    rc = cli.main(["--project-root", str(project_root), "--handoff-root", str(handoff_root),
                   "status", "--run-id", RUN])
    assert rc == 2
    assert "not found" in capsys.readouterr().err


# --- fetch --------------------------------------------------------------------


def _prepared(project_root, handoff_root):
    _write_jobs(project_root)
    assert cli.main(["--project-root", str(project_root), "--handoff-root", str(handoff_root),
                     "prepare", "--run-id", RUN]) == 0


def test_fetch_verifies_and_stages(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    _complete_valid(handoff_root)
    staging = handoff_root / "staging"
    rc = cli.main(["--project-root", str(project_root), "--handoff-root", str(handoff_root),
                   "fetch", "--run-id", RUN, "--staging-dir", str(staging)])
    out = capsys.readouterr().out
    assert rc == 0
    staged = staging / f"{RUN}_assets.zip"
    assert staged.is_file()
    assert "fetch verified" in out
    assert not (project_root / "data" / RUN / "assets").exists()  # not extracted


def test_fetch_missing_done(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    _complete_valid(handoff_root)
    (handoff_root / RUN / "output" / "DONE").unlink()
    rc = cli.main(["--project-root", str(project_root), "--handoff-root", str(handoff_root),
                   "fetch", "--run-id", RUN])
    assert rc == 2
    assert "hf22" in capsys.readouterr().err


def test_fetch_stale_marker(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    _complete_valid(handoff_root)
    write_marker(handoff_root / RUN / "output", "DONE", content="stale-attempt")
    rc = cli.main(["--project-root", str(project_root), "--handoff-root", str(handoff_root),
                   "fetch", "--run-id", RUN])
    assert rc == 2
    assert "hf25" in capsys.readouterr().err


def test_fetch_bad_zip_hash(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    zip_path = _complete_valid(handoff_root)
    with open(zip_path, "ab") as handle:
        handle.write(b"tamper")
    rc = cli.main(["--project-root", str(project_root), "--handoff-root", str(handoff_root),
                   "fetch", "--run-id", RUN])
    assert rc == 2
    assert "hf32" in capsys.readouterr().err


def test_fetch_wrong_run_id(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    out_dir = handoff_root / RUN / "output"
    zip_path = _make_zip(out_dir / f"{RUN}_assets.zip")
    _publish_output(handoff_root, RUN, zip_path=zip_path,
                    members=[f"{a}.png" for a in ASSETS] + ["result.json"],
                    generated=ASSETS, manifest_run_id="other-run")
    rc = cli.main(["--project-root", str(project_root), "--handoff-root", str(handoff_root),
                   "fetch", "--run-id", RUN])
    assert rc == 2
    assert "hf18" in capsys.readouterr().err


def test_fetch_unexpected_asset(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    out_dir = handoff_root / RUN / "output"
    zip_path = _make_zip(out_dir / f"{RUN}_assets.zip", assets=ASSETS + ["sc03_main"])
    _publish_output(handoff_root, RUN, zip_path=zip_path,
                    members=[f"{a}.png" for a in ASSETS + ["sc03_main"]] + ["result.json"],
                    generated=ASSETS + ["sc03_main"])
    rc = cli.main(["--project-root", str(project_root), "--handoff-root", str(handoff_root),
                   "fetch", "--run-id", RUN])
    assert rc == 2
    assert "hf43" in capsys.readouterr().err


def test_fetch_missing_asset(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    out_dir = handoff_root / RUN / "output"
    zip_path = _make_zip(out_dir / f"{RUN}_assets.zip", assets=[ASSETS[0]])
    _publish_output(handoff_root, RUN, zip_path=zip_path,
                    members=[f"{ASSETS[0]}.png", "result.json"], generated=[ASSETS[0]])
    rc = cli.main(["--project-root", str(project_root), "--handoff-root", str(handoff_root),
                   "fetch", "--run-id", RUN])
    assert rc == 2
    assert "hf43" in capsys.readouterr().err


def test_fetch_duplicate_zip_member(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    out_dir = handoff_root / RUN / "output"
    zip_path = _make_zip(out_dir / f"{RUN}_assets.zip", dup_first=True)
    _publish_output(handoff_root, RUN, zip_path=zip_path,
                    members=[f"{a}.png" for a in ASSETS] + ["result.json"],
                    generated=ASSETS)
    rc = cli.main(["--project-root", str(project_root), "--handoff-root", str(handoff_root),
                   "fetch", "--run-id", RUN])
    assert rc == 2
    assert "hf36" in capsys.readouterr().err


def test_fetch_unsafe_zip_path(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    out_dir = handoff_root / RUN / "output"
    zip_path = _make_zip(out_dir / f"{RUN}_assets.zip", extra_members=["../evil.png"])
    _publish_output(handoff_root, RUN, zip_path=zip_path,
                    members=[f"{a}.png" for a in ASSETS] + ["result.json"],
                    generated=ASSETS)
    rc = cli.main(["--project-root", str(project_root), "--handoff-root", str(handoff_root),
                   "fetch", "--run-id", RUN])
    assert rc == 2
    assert "hf34" in capsys.readouterr().err


def test_fetch_status_not_pass(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    out_dir = handoff_root / RUN / "output"
    zip_path = _make_zip(out_dir / f"{RUN}_assets.zip")
    blocked = build_status(
        RUN, attempt_id=_read_input_manifest(handoff_root)["attempt_id"],
        image_jobs_sha256=_read_input_manifest(handoff_root)["image_jobs_sha256"],
        overall_status="BLOCKED", current_stage="gpu_preflight",
        error=make_error("NO_GPU", "gpu_preflight", "no gpu"),
    )
    _publish_output(handoff_root, RUN, zip_path=zip_path,
                    members=[f"{a}.png" for a in ASSETS] + ["result.json"],
                    generated=ASSETS, status=blocked)
    rc = cli.main(["--project-root", str(project_root), "--handoff-root", str(handoff_root),
                   "fetch", "--run-id", RUN])
    assert rc == 2
    assert "hf48" in capsys.readouterr().err


def test_fetch_status_attempt_mismatch(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    out_dir = handoff_root / RUN / "output"
    zip_path = _make_zip(out_dir / f"{RUN}_assets.zip")
    bad_status = build_status(
        RUN, attempt_id="a-different-attempt",
        image_jobs_sha256=_read_input_manifest(handoff_root)["image_jobs_sha256"],
        overall_status="PASS", current_stage=STAGES[-1], completed_stages=list(STAGES),
        completed_asset_ids=list(ASSETS),
    )
    _publish_output(handoff_root, RUN, zip_path=zip_path,
                    members=[f"{a}.png" for a in ASSETS] + ["result.json"],
                    generated=ASSETS, status=bad_status)
    rc = cli.main(["--project-root", str(project_root), "--handoff-root", str(handoff_root),
                   "fetch", "--run-id", RUN])
    assert rc == 2
    assert "hf26" in capsys.readouterr().err


# --- validate -----------------------------------------------------------------


def _validate(project_root, handoff_root, *extra: str) -> int:
    return cli.main(["--project-root", str(project_root), "--handoff-root", str(handoff_root),
                     "validate", "--run-id", RUN, *extra])


def test_validate_ok_is_non_destructive(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    _complete_valid(handoff_root)
    capsys.readouterr()
    rc = _validate(project_root, handoff_root)
    out = capsys.readouterr().out
    assert rc == 0
    assert "import_status=validated" in out
    assert "validate OK" in out
    assert not (project_root / "data" / RUN / "assets").exists()


def test_validate_detects_corrupt_image(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    out_dir = handoff_root / RUN / "output"
    zip_path = out_dir / f"{RUN}_assets.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(f"{ASSETS[0]}.png", b"not a png")
        zf.writestr(f"{ASSETS[1]}.png", _png_bytes())
        zf.writestr("result.json", json.dumps(_result_document()))
    _publish_output(
        handoff_root, RUN, zip_path=zip_path,
        members=[f"{a}.png" for a in ASSETS] + ["result.json"],
        generated=ASSETS,
    )
    rc = _validate(project_root, handoff_root)
    assert rc == 2
    assert "validate FAILED" in capsys.readouterr().out


def test_validate_detects_wrong_dimensions(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    out_dir = handoff_root / RUN / "output"
    zip_path = _make_zip(out_dir / f"{RUN}_assets.zip", size=(64, 64))
    _publish_output(
        handoff_root, RUN, zip_path=zip_path,
        members=[f"{a}.png" for a in ASSETS] + ["result.json"],
        generated=ASSETS,
    )
    rc = _validate(project_root, handoff_root)
    assert rc == 2
    assert "validate FAILED" in capsys.readouterr().out


def test_validate_without_zip(project_root, handoff_root, capsys):
    # A completed run whose artifact ZIP has gone missing cannot validate.
    _prepared(project_root, handoff_root)
    _complete_valid(handoff_root).unlink()
    rc = _validate(project_root, handoff_root)
    assert rc == 2
    assert "hf30" in capsys.readouterr().err


# --- validate: completion binding (N2) ----------------------------------------


def test_validate_rejects_arbitrary_zip_without_completion(project_root, handoff_root, capsys):
    # N2 regression: a ZIP merely sitting under output/ must NOT validate.
    _prepared(project_root, handoff_root)
    _make_zip(handoff_root / RUN / "output" / f"{RUN}_assets.zip")
    capsys.readouterr()
    rc = _validate(project_root, handoff_root)
    captured = capsys.readouterr()
    assert rc == 2
    assert "hf22" in captured.err  # DONE marker missing
    assert "validate OK" not in captured.out
    assert "import_status" not in captured.out


def test_validate_rejects_missing_done(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    _complete_valid(handoff_root)
    (handoff_root / RUN / "output" / "DONE").unlink()
    rc = _validate(project_root, handoff_root)
    assert rc == 2
    assert "hf22" in capsys.readouterr().err


def test_validate_rejects_non_success_status(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    out_dir = handoff_root / RUN / "output"
    zip_path = _make_zip(out_dir / f"{RUN}_assets.zip")
    blocked = build_status(
        RUN, attempt_id=_read_input_manifest(handoff_root)["attempt_id"],
        image_jobs_sha256=_read_input_manifest(handoff_root)["image_jobs_sha256"],
        overall_status="BLOCKED", current_stage="gpu_preflight",
        error=make_error("NO_GPU", "gpu_preflight", "no gpu"),
    )
    _publish_output(
        handoff_root, RUN, zip_path=zip_path,
        members=[f"{a}.png" for a in ASSETS] + ["result.json"],
        generated=ASSETS, status=blocked,
    )
    rc = _validate(project_root, handoff_root)
    assert rc == 2
    assert "hf48" in capsys.readouterr().err


def test_validate_rejects_attempt_id_mismatch(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    out_dir = handoff_root / RUN / "output"
    zip_path = _make_zip(out_dir / f"{RUN}_assets.zip")
    _publish_output(
        handoff_root, RUN, zip_path=zip_path,
        members=[f"{a}.png" for a in ASSETS] + ["result.json"],
        generated=ASSETS, attempt="stale-attempt", done_content="stale-attempt",
    )
    rc = _validate(project_root, handoff_root)
    assert rc == 2
    assert "hf28" in capsys.readouterr().err


def test_validate_rejects_missing_output_manifest(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    _complete_valid(handoff_root)
    (handoff_root / RUN / "output" / "output_manifest.json").unlink()
    rc = _validate(project_root, handoff_root)
    assert rc == 2
    assert "hf23" in capsys.readouterr().err


def test_validate_rejects_stale_output_zip(project_root, handoff_root, capsys):
    # The completed artifact's ZIP was tampered with after publication.
    _prepared(project_root, handoff_root)
    zip_path = _complete_valid(handoff_root)
    with open(zip_path, "ab") as handle:
        handle.write(b"tamper")
    rc = _validate(project_root, handoff_root)
    assert rc == 2
    assert "hf32" in capsys.readouterr().err


def test_validate_rejects_swapped_zip_missing_asset(project_root, handoff_root, capsys):
    # A same-named ZIP that lacks an expected asset is not the verified artifact.
    _prepared(project_root, handoff_root)
    zip_path = _complete_valid(handoff_root)
    swapped = _make_zip(zip_path.parent / f"{RUN}_swapped.zip", assets=[ASSETS[0]])
    zip_path.unlink()
    swapped.rename(zip_path)
    rc = _validate(project_root, handoff_root)
    assert rc == 2
    assert "hf32" in capsys.readouterr().err


def test_validate_rejects_multiple_output_zips(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    _complete_valid(handoff_root)
    _make_zip(handoff_root / RUN / "output" / "stale.zip")
    rc = _validate(project_root, handoff_root)
    assert rc == 2
    assert "multiple ZIPs" in capsys.readouterr().err


def test_validate_rejects_stale_staged_zip(project_root, handoff_root, capsys):
    # A staged copy from an earlier run shadows the output ZIP but fails the
    # manifest hash. (A byte-identical copy is the same artifact and passes.)
    _prepared(project_root, handoff_root)
    _complete_valid(handoff_root)
    staging = project_root / "handoff_staging" / RUN
    _make_zip(staging / f"{RUN}_assets.zip", run_id="an-older-run")
    rc = _validate(project_root, handoff_root)
    assert rc == 2
    assert "hf32" in capsys.readouterr().err


def test_validate_explicit_zip_must_match_completed_artifact(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    _complete_valid(handoff_root)
    other = _make_zip(handoff_root / "elsewhere.zip", run_id="an-older-run")
    rc = _validate(project_root, handoff_root, "--zip", str(other))
    assert rc == 2
    assert "hf32" in capsys.readouterr().err


def test_validate_staged_zip_passes_when_verified(project_root, handoff_root, capsys):
    # A staged copy of the exact verified artifact still validates.
    _prepared(project_root, handoff_root)
    _complete_valid(handoff_root)
    staging = project_root / "handoff_staging" / RUN
    staging.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        handoff_root / RUN / "output" / f"{RUN}_assets.zip",
        staging / f"{RUN}_assets.zip",
    )
    capsys.readouterr()
    rc = _validate(project_root, handoff_root)
    out = capsys.readouterr().out
    assert rc == 0
    assert "validate OK" in out


# --- validate: in-ZIP result.json contract (N3) -------------------------------


def test_validate_rejects_malformed_result_json(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    _complete_with_result(handoff_root, None, raw_result="{not json")
    rc = _validate(project_root, handoff_root)
    assert rc == 2
    assert "hf49" in capsys.readouterr().err


def test_validate_rejects_result_schema_mismatch(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    doc = _result_document()
    doc["schema"] = "image-generation-results/2.0"
    _complete_with_result(handoff_root, doc)
    rc = _validate(project_root, handoff_root)
    assert rc == 2
    assert "hf49" in capsys.readouterr().err


def test_validate_rejects_result_run_id_mismatch(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    _complete_with_result(handoff_root, _result_document(run_id="some-other-run"))
    rc = _validate(project_root, handoff_root)
    assert rc == 2
    assert "hf50" in capsys.readouterr().err


def test_validate_rejects_result_asset_set_mismatch(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    doc = _result_document(assets=[ASSETS[0]])  # missing sc02_main
    _complete_with_result(handoff_root, doc)
    rc = _validate(project_root, handoff_root)
    assert rc == 2
    assert "hf50" in capsys.readouterr().err


def test_validate_rejects_result_partial_failure(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    _complete_with_result(handoff_root, _result_document(status="partial_failure"))
    rc = _validate(project_root, handoff_root)
    assert rc == 2
    assert "hf50" in capsys.readouterr().err


def test_validate_missing_result_member_rejected_by_member_gate(project_root, handoff_root, capsys):
    # A ZIP without result.json is already rejected by the manifest member gate.
    _prepared(project_root, handoff_root)
    out_dir = handoff_root / RUN / "output"
    zip_path = out_dir / f"{RUN}_assets.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for asset in ASSETS:
            zf.writestr(f"{asset}.png", _png_bytes())
    _publish_output(
        handoff_root, RUN, zip_path=zip_path,
        members=[f"{a}.png" for a in ASSETS],
        generated=ASSETS,
    )
    rc = _validate(project_root, handoff_root)
    assert rc == 2
    err = capsys.readouterr().err
    assert "hf35" in err or "hf49" in err


def test_validate_ignores_loose_output_result_json(project_root, handoff_root, capsys):
    # The loose output/result.json is never trusted as a substitute for the
    # in-ZIP member: a valid loose copy cannot rescue an invalid in-ZIP one.
    _prepared(project_root, handoff_root)
    zip_path = _complete_with_result(handoff_root, None, raw_result="{not json")
    (zip_path.parent / "result.json").write_text(
        json.dumps(_result_document()), encoding="utf-8"
    )
    rc = _validate(project_root, handoff_root)
    assert rc == 2
    assert "hf49" in capsys.readouterr().err


# --- import -------------------------------------------------------------------


def test_import_requires_authorization(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    _complete_valid(handoff_root)
    rc = cli.main(["--project-root", str(project_root), "--handoff-root", str(handoff_root),
                   "import", "--run-id", RUN])
    assert rc == 2
    assert "--authorize-import" in capsys.readouterr().err
    assert not (project_root / "data" / RUN / "assets").exists()


def test_import_authorized_writes_assets(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    _complete_valid(handoff_root)
    # fetch first so the staged ZIP path is exercised
    cli.main(["--project-root", str(project_root), "--handoff-root", str(handoff_root),
              "fetch", "--run-id", RUN])
    capsys.readouterr()
    rc = cli.main(["--project-root", str(project_root), "--handoff-root", str(handoff_root),
                   "import", "--run-id", RUN, "--authorize-import"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "import_status=imported" in out
    for asset in ASSETS:
        assert (project_root / "data" / RUN / "assets" / f"{asset}.png").is_file()
    assert "Stage 5 is NOT started" in out


def test_import_does_not_overwrite_by_default(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    _complete_valid(handoff_root)
    dest = project_root / "data" / RUN / "assets" / f"{ASSETS[0]}.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"original-bytes")
    rc = cli.main(["--project-root", str(project_root), "--handoff-root", str(handoff_root),
                   "import", "--run-id", RUN, "--authorize-import"])
    assert rc == 2
    assert dest.read_bytes() == b"original-bytes"  # untouched


def test_import_overwrite_flag_replaces(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    _complete_valid(handoff_root)
    dest = project_root / "data" / RUN / "assets" / f"{ASSETS[0]}.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"original-bytes")
    rc = cli.main(["--project-root", str(project_root), "--handoff-root", str(handoff_root),
                   "import", "--run-id", RUN, "--authorize-import", "--overwrite"])
    assert rc == 0
    assert dest.read_bytes() != b"original-bytes"


def test_import_refuses_unverified_zip(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    zip_path = _complete_valid(handoff_root)
    with open(zip_path, "ab") as handle:
        handle.write(b"tamper")
    rc = cli.main(["--project-root", str(project_root), "--handoff-root", str(handoff_root),
                   "import", "--run-id", RUN, "--authorize-import"])
    assert rc == 2
    assert not (project_root / "data" / RUN / "assets").exists()


def test_import_rejects_invalid_result_document(project_root, handoff_root, capsys):
    # Import stays fail-closed: a malformed in-ZIP result.json writes nothing.
    _prepared(project_root, handoff_root)
    _complete_with_result(handoff_root, None, raw_result="{not json")
    rc = cli.main(["--project-root", str(project_root), "--handoff-root", str(handoff_root),
                   "import", "--run-id", RUN, "--authorize-import"])
    assert rc == 2
    assert "hf49" in capsys.readouterr().err
    assert not (project_root / "data" / RUN / "assets").exists()


# --- offline / scope ----------------------------------------------------------


def test_cli_imports_only_stdlib_and_agents():
    source = Path(cli.__file__).read_text(encoding="utf-8")
    mods = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    assert mods <= {
        "__future__", "argparse", "shutil", "sys", "pathlib", "agents", "tools",
    }


def test_root_is_reused_not_duplicated(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    cli.main(["--project-root", str(project_root), "--handoff-root", str(handoff_root),
              "prepare", "--run-id", RUN, "--force"])
    run_dirs = [p for p in handoff_root.iterdir() if p.is_dir()]
    assert [p.name for p in run_dirs] == [RUN]


# --- run-id confinement (F1) --------------------------------------------------


BAD_RUN_IDS = [
    "",
    " ",
    "  padded  ",
    ".",
    "..",
    "../evil",
    "..\\evil",
    "a/b",
    "a\\b",
    "/abs",
    "\\abs",
    "C:/evil",
    "C:\\evil",
    "C:evil",
]


@pytest.mark.parametrize("bad", BAD_RUN_IDS)
def test_prepare_rejects_unconfined_run_id(project_root, handoff_root, capsys, bad):
    _write_jobs(project_root)
    rc = cli.main(["--project-root", str(project_root),
                   "--handoff-root", str(handoff_root),
                   "prepare", "--run-id", bad])
    assert rc == 2
    assert "hf08" in capsys.readouterr().err
    # No run folder and no traversal target were created under the root.
    assert not (handoff_root / "a").exists()
    assert not (handoff_root / "C:").exists()


@pytest.mark.parametrize("command", ["prepare", "status", "fetch", "validate", "import"])
def test_run_id_validation_applies_to_every_subcommand(
    project_root, handoff_root, capsys, command
):
    # Deliberately no --authorize-import: run-id validation must run first.
    rc = cli.main(["--project-root", str(project_root),
                   "--handoff-root", str(handoff_root),
                   command, "--run-id", "../evil"])
    assert rc == 2
    assert "hf08" in capsys.readouterr().err


def test_valid_run_id_with_dots_and_dashes(project_root, handoff_root):
    rid = "e2e-2026.10_v2-01"
    _write_jobs(project_root, run_id=rid)
    rc = cli.main(["--project-root", str(project_root),
                   "--handoff-root", str(handoff_root),
                   "prepare", "--run-id", rid])
    assert rc == 0
    assert (handoff_root / rid / "input" / "READY").is_file()
    assert not (handoff_root / "e2e-2026").exists()


# --- status diagnostics (F2/F3) -----------------------------------------------


def _status(project_root, handoff_root):
    return cli.main(["--project-root", str(project_root),
                     "--handoff-root", str(handoff_root),
                     "status", "--run-id", RUN])


def _write_progress(handoff_root, *, overall_status="FAILED", current_stage="env_setup",
                    completed_stages=None, error=None, attempt=None, run_id=RUN):
    """Overwrite the progress ``status/status.json`` as the worker would."""
    in_manifest = _read_input_manifest(handoff_root, run_id)
    status = build_status(
        run_id,
        attempt_id=attempt if attempt is not None else in_manifest["attempt_id"],
        image_jobs_sha256=in_manifest["image_jobs_sha256"],
        overall_status=overall_status,
        current_stage=current_stage,
        completed_stages=completed_stages,
        error=error,
    )
    dump_json(status, handoff_root / run_id / "status" / "status.json")


def test_status_flags_stale_ready_mismatch(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    write_marker(handoff_root / RUN / "input", "READY", content="stale-attempt")
    rc = _status(project_root, handoff_root)
    out = capsys.readouterr().out
    assert rc == 2
    assert "input_ready=mismatch" in out
    assert "status problem" in out


def test_status_flags_stale_done_mismatch(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    _complete_valid(handoff_root)
    write_marker(handoff_root / RUN / "output", "DONE", content="stale-attempt")
    rc = _status(project_root, handoff_root)
    out = capsys.readouterr().out
    assert rc == 2
    assert "output_done=yes" in out
    assert "output_done_matches=no" in out
    assert "status problem" in out


def test_status_flags_done_vs_output_manifest_mismatch(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    _complete_valid(handoff_root)
    out_dir = handoff_root / RUN / "output"
    manifest = json.loads((out_dir / "output_manifest.json").read_text(encoding="utf-8"))
    manifest["attempt_id"] = "different-attempt"
    (out_dir / "output_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rc = _status(project_root, handoff_root)
    out = capsys.readouterr().out
    assert rc == 2
    assert "output_done_matches=no" in out
    assert "output_attempts_consistent=no" in out


def test_status_flags_malformed_progress_but_completes_report(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    (handoff_root / RUN / "status" / "status.json").write_text("{not json", encoding="utf-8")
    rc = _status(project_root, handoff_root)
    out = capsys.readouterr().out
    assert rc == 2
    assert "progress_status=malformed" in out
    assert "status problem" in out
    # The report still ran to completion (the output section was reached).
    assert "output_done=no" in out


def test_status_flags_malformed_input_manifest(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    (handoff_root / RUN / "input" / "input_manifest.json").write_text(
        "{not json", encoding="utf-8"
    )
    rc = _status(project_root, handoff_root)
    out = capsys.readouterr().out
    assert rc == 2
    assert "input_manifest=malformed" in out
    assert "status problem" in out


def test_status_surfaces_progress_error_for_prompt_failure(project_root, handoff_root, capsys):
    # A prompt defect writes a terminal FAILED status to status/status.json and no
    # output status exists (early completion invalidation removed it). `status`
    # must still surface the error code/stage so the operator sees the cause.
    _prepared(project_root, handoff_root)
    _write_progress(
        handoff_root,
        overall_status="FAILED",
        current_stage="env_setup",
        completed_stages=["gpu_preflight"],
        error=make_error("INPUT_VALIDATION_FAILED", "env_setup", "prompt over token budget"),
    )
    rc = _status(project_root, handoff_root)
    out = capsys.readouterr().out
    assert rc == 0
    assert "progress_status=FAILED" in out
    assert "progress_error_code=INPUT_VALIDATION_FAILED" in out
    assert "progress_error_stage=env_setup" in out
    assert "output_done=no" in out


def test_status_flags_conflicting_progress_attempt(project_root, handoff_root, capsys):
    # A prompt-failure status bound to a different attempt must not be treated as
    # the current run's status.
    _prepared(project_root, handoff_root)
    _write_progress(
        handoff_root,
        overall_status="FAILED",
        current_stage="env_setup",
        completed_stages=["gpu_preflight"],
        error=make_error("INPUT_VALIDATION_FAILED", "env_setup", "prompt over token budget"),
        attempt="a-conflicting-attempt",
    )
    rc = _status(project_root, handoff_root)
    out = capsys.readouterr().out
    assert rc == 2
    assert "progress_status=FAILED" in out
    assert "progress_attempt_matches_input=no" in out
    assert "status problem" in out


def test_status_clean_run_exits_zero(project_root, handoff_root, capsys):
    _prepared(project_root, handoff_root)
    rc = _status(project_root, handoff_root)
    out = capsys.readouterr().out
    assert rc == 0
    assert "status problem" not in out
