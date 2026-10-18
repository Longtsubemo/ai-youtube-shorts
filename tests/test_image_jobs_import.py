"""Tests for the Colab SDXL asset importer (agents/image_jobs/import_assets).

The importer validates a Colab-generated ZIP against the real image-job
contract and copies scene backgrounds into ``data/<run_id>/assets/``. These
tests use synthetic PNGs and temp directories only; no network, no real run
artifacts are modified.
"""

from __future__ import annotations

import hashlib
import io
import json
import warnings
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from agents.image_jobs.import_assets import (
    DEFAULT_IMAGE_SIZE,
    REPORT_FILENAME,
    REPORT_SCHEMA_VERSION,
    AssetImportError,
    import_generated_assets,
)
from agents.image_jobs.import_assets import main as import_main

RUN_ID = "run-import-01"
EXPECTED_SIZE = (8, 6)

REAL_JOBS = Path("data/e2e-20261012-sceneprompt-v2-01/image_jobs.json")
REAL_RUN_ID = "e2e-20261012-sceneprompt-v2-01"


# --- helpers -----------------------------------------------------------------


def _png_bytes(size=EXPECTED_SIZE, color=(10, 20, 30), fmt="PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


def _job(scene_id: str, *, asset_id=None, output_path=None, run_id=RUN_ID) -> dict:
    asset_id = asset_id if asset_id is not None else f"{scene_id}_main"
    output_path = (
        output_path
        if output_path is not None
        else f"data/{run_id}/assets/{asset_id}.png"
    )
    return {
        "job_id": f"{run_id}-{scene_id}",
        "scene_id": scene_id,
        "order": 0,
        "duration_seconds": 8,
        "scene_generation_prompt": "coins on a desk",
        "channel_image_style": "style",
        "positive_prompt": "style, coins on a desk",
        "negative_prompt": "blurry",
        "seed": 1,
        "asset_id": asset_id,
        "output_path": output_path,
        "mascot": {"present": False},
    }


def _write_jobs(
    run_dir: Path,
    jobs: list[dict],
    *,
    run_id=RUN_ID,
    schema="image-jobs/1.0",
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "image_jobs.json"
    path.write_text(
        json.dumps({"schema": schema, "run_id": run_id, "jobs": jobs}, indent=2),
        encoding="utf-8",
    )
    return path


def _make_zip(path: Path, members: list[tuple[str, bytes]]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                archive.writestr(name, data)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_dir(project_root: Path, run_id=RUN_ID) -> Path:
    return project_root / "data" / run_id


# --- constants ---------------------------------------------------------------


def test_default_expected_size_is_sdxl_portrait():
    assert DEFAULT_IMAGE_SIZE == (768, 1344)
    assert REPORT_FILENAME == "asset_import.json"
    assert REPORT_SCHEMA_VERSION == "asset-import/1.0"


# --- successful imports ------------------------------------------------------


def test_import_flat_layout_succeeds(tmp_path):
    run_dir = _run_dir(tmp_path)
    _write_jobs(run_dir, [_job("sc01"), _job("sc02")])
    png_a = _png_bytes(color=(1, 2, 3))
    png_b = _png_bytes(color=(4, 5, 6))
    zip_path = _make_zip(
        tmp_path / "assets.zip",
        [("sc01_main.png", png_a), ("sc02_main.png", png_b)],
    )

    report = import_generated_assets(
        zip_path, RUN_ID, project_root=tmp_path, expected_size=EXPECTED_SIZE
    )

    assert report.status == "imported"
    assert report.ok
    assert (run_dir / "assets" / "sc01_main.png").read_bytes() == png_a
    assert (run_dir / "assets" / "sc02_main.png").read_bytes() == png_b
    assert (run_dir / REPORT_FILENAME).is_file()
    assert report.report_path.endswith(REPORT_FILENAME)
    data = json.loads((run_dir / REPORT_FILENAME).read_text(encoding="utf-8"))
    assert data["schema"] == REPORT_SCHEMA_VERSION
    assert data["status"] == "imported"
    assert data["counts"]["expected"] == 2
    assert data["counts"]["written"] == 2


def test_import_assets_prefixed_layout_succeeds(tmp_path):
    run_dir = _run_dir(tmp_path)
    _write_jobs(run_dir, [_job("sc01")])
    png = _png_bytes()
    zip_path = _make_zip(tmp_path / "assets.zip", [("assets/sc01_main.png", png)])

    report = import_generated_assets(
        zip_path, RUN_ID, project_root=tmp_path, expected_size=EXPECTED_SIZE
    )

    assert report.status == "imported"
    assert (run_dir / "assets" / "sc01_main.png").read_bytes() == png
    assert report.unexpected == []


def test_result_manifest_is_ignored_not_reported(tmp_path):
    run_dir = _run_dir(tmp_path)
    _write_jobs(run_dir, [_job("sc01")])
    zip_path = _make_zip(
        tmp_path / "assets.zip",
        [("result.json", b'{"ok": true}'), ("sc01_main.png", _png_bytes())],
    )

    report = import_generated_assets(
        zip_path, RUN_ID, project_root=tmp_path, expected_size=EXPECTED_SIZE
    )

    assert report.status == "imported"
    assert report.unexpected == []
    assert not (run_dir / "assets" / "result.json").exists()


def test_report_records_destinations_and_ids(tmp_path):
    run_dir = _run_dir(tmp_path)
    _write_jobs(run_dir, [_job("sc01")])
    zip_path = _make_zip(tmp_path / "assets.zip", [("sc01_main.png", _png_bytes())])

    report = import_generated_assets(
        zip_path, RUN_ID, project_root=tmp_path, expected_size=EXPECTED_SIZE
    )

    entry = report.assets[0]
    assert entry["scene_id"] == "sc01"
    assert entry["asset_id"] == "sc01_main"
    assert entry["status"] == "imported"
    assert entry["destination"] == str(run_dir / "assets" / "sc01_main.png")


# --- missing / unexpected / duplicates ---------------------------------------


def test_missing_image_fails_without_partial_import(tmp_path):
    run_dir = _run_dir(tmp_path)
    _write_jobs(run_dir, [_job("sc01"), _job("sc02")])
    zip_path = _make_zip(tmp_path / "assets.zip", [("sc01_main.png", _png_bytes())])

    report = import_generated_assets(
        zip_path, RUN_ID, project_root=tmp_path, expected_size=EXPECTED_SIZE
    )

    assert report.status == "failed"
    assert not report.ok
    assert "missing" in {a["status"] for a in report.assets}
    assert any(i.code == "im10" for i in report.errors)
    # no partial import: neither image was written, assets dir not even created
    assert not (run_dir / "assets").exists()


def test_unexpected_member_warns_but_still_imports(tmp_path):
    run_dir = _run_dir(tmp_path)
    _write_jobs(run_dir, [_job("sc01")])
    zip_path = _make_zip(
        tmp_path / "assets.zip",
        [("sc01_main.png", _png_bytes()), ("notes/extra.png", _png_bytes())],
    )

    report = import_generated_assets(
        zip_path, RUN_ID, project_root=tmp_path, expected_size=EXPECTED_SIZE
    )

    assert report.status == "imported"
    assert report.unexpected == ["notes/extra.png"]
    assert any(w.code == "im12" for w in report.warnings)
    assert (run_dir / "assets" / "sc01_main.png").is_file()
    assert not (run_dir / "assets" / "extra.png").exists()


def test_duplicate_zip_member_rejected(tmp_path):
    run_dir = _run_dir(tmp_path)
    _write_jobs(run_dir, [_job("sc01")])
    zip_path = _make_zip(
        tmp_path / "assets.zip",
        [("sc01_main.png", _png_bytes()), ("sc01_main.png", _png_bytes())],
    )

    report = import_generated_assets(
        zip_path, RUN_ID, project_root=tmp_path, expected_size=EXPECTED_SIZE
    )

    assert report.status == "failed"
    assert report.assets[0]["status"] == "duplicate"
    assert any(i.code == "im11" for i in report.errors)
    assert not (run_dir / "assets").exists()


def test_normalized_duplicate_member_rejected(tmp_path):
    run_dir = _run_dir(tmp_path)
    _write_jobs(run_dir, [_job("sc01")])
    zip_path = _make_zip(
        tmp_path / "assets.zip",
        [("sc01_main.png", _png_bytes()), ("./sc01_main.png", _png_bytes())],
    )

    report = import_generated_assets(
        zip_path, RUN_ID, project_root=tmp_path, expected_size=EXPECTED_SIZE
    )

    assert report.status == "failed"
    assert report.assets[0]["status"] == "duplicate"
    assert not (run_dir / "assets").exists()


# --- corrupt / wrong dimensions ----------------------------------------------


def test_corrupt_image_fails(tmp_path):
    run_dir = _run_dir(tmp_path)
    _write_jobs(run_dir, [_job("sc01")])
    zip_path = _make_zip(tmp_path / "assets.zip", [("sc01_main.png", b"not an image")])

    report = import_generated_assets(
        zip_path, RUN_ID, project_root=tmp_path, expected_size=EXPECTED_SIZE
    )

    assert report.status == "failed"
    assert report.assets[0]["status"] == "corrupt"
    assert any(i.code == "im13" for i in report.errors)
    assert not (run_dir / "assets").exists()


def test_jpeg_masquerading_as_png_is_corrupt(tmp_path):
    run_dir = _run_dir(tmp_path)
    _write_jobs(run_dir, [_job("sc01")])
    jpeg = _png_bytes(fmt="JPEG")
    zip_path = _make_zip(tmp_path / "assets.zip", [("sc01_main.png", jpeg)])

    report = import_generated_assets(
        zip_path, RUN_ID, project_root=tmp_path, expected_size=EXPECTED_SIZE
    )

    assert report.status == "failed"
    assert report.assets[0]["status"] == "corrupt"
    assert not (run_dir / "assets").exists()


def test_wrong_dimensions_fails(tmp_path):
    run_dir = _run_dir(tmp_path)
    _write_jobs(run_dir, [_job("sc01")])
    zip_path = _make_zip(
        tmp_path / "assets.zip", [("sc01_main.png", _png_bytes(size=(9, 6)))]
    )

    report = import_generated_assets(
        zip_path, RUN_ID, project_root=tmp_path, expected_size=EXPECTED_SIZE
    )

    assert report.status == "failed"
    assert report.assets[0]["status"] == "invalid_dimension"
    assert any(i.code == "im14" for i in report.errors)
    assert not (run_dir / "assets").exists()


# --- overwrite / conflict ----------------------------------------------------


def test_existing_target_conflicts_by_default(tmp_path):
    run_dir = _run_dir(tmp_path)
    _write_jobs(run_dir, [_job("sc01")])
    target = run_dir / "assets" / "sc01_main.png"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"sentinel-original")
    zip_path = _make_zip(tmp_path / "assets.zip", [("sc01_main.png", _png_bytes())])

    report = import_generated_assets(
        zip_path, RUN_ID, project_root=tmp_path, expected_size=EXPECTED_SIZE
    )

    assert report.status == "failed"
    assert report.assets[0]["status"] == "conflict"
    assert any(i.code == "im15" for i in report.errors)
    assert target.read_bytes() == b"sentinel-original"


def test_overwrite_replaces_existing_target(tmp_path):
    run_dir = _run_dir(tmp_path)
    _write_jobs(run_dir, [_job("sc01")])
    target = run_dir / "assets" / "sc01_main.png"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"sentinel-original")
    png = _png_bytes(color=(7, 8, 9))
    zip_path = _make_zip(tmp_path / "assets.zip", [("sc01_main.png", png)])

    report = import_generated_assets(
        zip_path,
        RUN_ID,
        project_root=tmp_path,
        expected_size=EXPECTED_SIZE,
        overwrite=True,
    )

    assert report.status == "imported"
    assert report.assets[0]["status"] == "overwritten"
    assert target.read_bytes() == png


def test_overwrite_never_writes_when_validation_fails(tmp_path):
    run_dir = _run_dir(tmp_path)
    _write_jobs(run_dir, [_job("sc01")])
    target = run_dir / "assets" / "sc01_main.png"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"sentinel-original")
    zip_path = _make_zip(tmp_path / "assets.zip", [("sc01_main.png", b"bad")])

    report = import_generated_assets(
        zip_path,
        RUN_ID,
        project_root=tmp_path,
        expected_size=EXPECTED_SIZE,
        overwrite=True,
    )

    assert report.status == "failed"
    assert target.read_bytes() == b"sentinel-original"


# --- dry run -----------------------------------------------------------------


def test_dry_run_writes_nothing(tmp_path):
    run_dir = _run_dir(tmp_path)
    _write_jobs(run_dir, [_job("sc01")])
    zip_path = _make_zip(tmp_path / "assets.zip", [("sc01_main.png", _png_bytes())])

    report = import_generated_assets(
        zip_path, RUN_ID, project_root=tmp_path, expected_size=EXPECTED_SIZE, dry_run=True
    )

    assert report.status == "validated"
    assert report.ok
    assert report.assets[0]["status"] == "validated"
    assert report.report_path == ""
    assert not (run_dir / "assets").exists()
    assert not (run_dir / REPORT_FILENAME).exists()


def test_dry_run_reports_failures_without_writing(tmp_path):
    run_dir = _run_dir(tmp_path)
    _write_jobs(run_dir, [_job("sc01")])
    zip_path = _make_zip(tmp_path / "assets.zip", [("sc01_main.png", b"bad")])

    report = import_generated_assets(
        zip_path, RUN_ID, project_root=tmp_path, expected_size=EXPECTED_SIZE, dry_run=True
    )

    assert report.status == "failed"
    assert not (run_dir / "assets").exists()
    assert not (run_dir / REPORT_FILENAME).exists()


# --- unsafe contract / path traversal ----------------------------------------


def test_output_path_outside_assets_dir_rejected(tmp_path):
    run_dir = _run_dir(tmp_path)
    _write_jobs(
        run_dir,
        [_job("sc01", output_path=f"data/{RUN_ID}/other/sc01_main.png")],
    )
    zip_path = _make_zip(tmp_path / "assets.zip", [("sc01_main.png", _png_bytes())])

    report = import_generated_assets(
        zip_path, RUN_ID, project_root=tmp_path, expected_size=EXPECTED_SIZE
    )

    assert report.status == "failed"
    assert report.assets[0]["status"] == "unsafe"
    assert any(i.code == "im06" for i in report.errors)
    assert not (run_dir / "assets").exists()


def test_output_path_with_traversal_rejected(tmp_path):
    run_dir = _run_dir(tmp_path)
    _write_jobs(
        run_dir,
        [_job("sc01", output_path=f"data/{RUN_ID}/assets/../sc01_main.png")],
    )
    zip_path = _make_zip(tmp_path / "assets.zip", [("sc01_main.png", _png_bytes())])

    report = import_generated_assets(
        zip_path, RUN_ID, project_root=tmp_path, expected_size=EXPECTED_SIZE
    )

    assert report.status == "failed"
    assert report.assets[0]["status"] == "unsafe"
    assert not (run_dir / "assets").exists()


def test_absolute_output_path_rejected(tmp_path):
    run_dir = _run_dir(tmp_path)
    _write_jobs(run_dir, [_job("sc01", output_path="/tmp/evil.png")])
    zip_path = _make_zip(tmp_path / "assets.zip", [("sc01_main.png", _png_bytes())])

    report = import_generated_assets(
        zip_path, RUN_ID, project_root=tmp_path, expected_size=EXPECTED_SIZE
    )

    assert report.status == "failed"
    assert report.assets[0]["status"] == "unsafe"


def test_unsafe_asset_id_rejected(tmp_path):
    run_dir = _run_dir(tmp_path)
    _write_jobs(
        run_dir,
        [
            _job(
                "sc01",
                asset_id="../evil",
                output_path=f"data/{RUN_ID}/assets/../evil.png",
            )
        ],
    )
    zip_path = _make_zip(tmp_path / "assets.zip", [("../evil.png", _png_bytes())])

    report = import_generated_assets(
        zip_path, RUN_ID, project_root=tmp_path, expected_size=EXPECTED_SIZE
    )

    assert report.status == "failed"
    assert report.assets[0]["status"] == "unsafe"
    assert not (run_dir / "assets").exists()


def test_zip_traversal_member_is_not_written_outside(tmp_path):
    run_dir = _run_dir(tmp_path)
    _write_jobs(run_dir, [_job("sc01")])
    zip_path = _make_zip(tmp_path / "assets.zip", [("../sc01_main.png", _png_bytes())])

    report = import_generated_assets(
        zip_path, RUN_ID, project_root=tmp_path, expected_size=EXPECTED_SIZE
    )

    assert report.status == "failed"
    assert "../sc01_main.png" in report.unexpected
    assert not (tmp_path / "sc01_main.png").exists()
    assert not (run_dir / "assets").exists()


def test_duplicate_asset_id_in_contract_rejected(tmp_path):
    run_dir = _run_dir(tmp_path)
    _write_jobs(run_dir, [_job("sc01"), _job("sc01")])
    zip_path = _make_zip(tmp_path / "assets.zip", [("sc01_main.png", _png_bytes())])

    report = import_generated_assets(
        zip_path, RUN_ID, project_root=tmp_path, expected_size=EXPECTED_SIZE
    )

    assert report.status == "failed"
    assert any(i.code == "im07" for i in report.errors)
    assert not (run_dir / "assets").exists()


def test_unsafe_contract_reported_even_when_zip_missing(tmp_path):
    run_dir = _run_dir(tmp_path)
    _write_jobs(run_dir, [_job("sc01", output_path="elsewhere/sc01_main.png")])

    report = import_generated_assets(
        tmp_path / "does-not-exist.zip",
        RUN_ID,
        project_root=tmp_path,
        expected_size=EXPECTED_SIZE,
    )

    assert report.status == "failed"
    assert report.assets[0]["status"] == "unsafe"


# --- read-only source guarantees ---------------------------------------------


def test_source_artifacts_are_byte_identical_after_import(tmp_path):
    run_dir = _run_dir(tmp_path)
    jobs_path = _write_jobs(run_dir, [_job("sc01")])
    zip_path = _make_zip(tmp_path / "assets.zip", [("sc01_main.png", _png_bytes())])
    jobs_hash = _sha256(jobs_path)
    zip_hash = _sha256(zip_path)

    import_generated_assets(
        zip_path, RUN_ID, project_root=tmp_path, expected_size=EXPECTED_SIZE
    )

    assert _sha256(jobs_path) == jobs_hash
    assert _sha256(zip_path) == zip_hash


# --- structural input errors -------------------------------------------------


def test_missing_zip_raises_im08(tmp_path):
    run_dir = _run_dir(tmp_path)
    _write_jobs(run_dir, [_job("sc01")])
    with pytest.raises(AssetImportError) as exc:
        import_generated_assets(
            tmp_path / "nope.zip", RUN_ID, project_root=tmp_path, expected_size=EXPECTED_SIZE
        )
    assert exc.value.code == "im08"


def test_non_zip_file_raises_im09(tmp_path):
    run_dir = _run_dir(tmp_path)
    _write_jobs(run_dir, [_job("sc01")])
    not_zip = tmp_path / "not.zip"
    not_zip.write_bytes(b"definitely not a zip")
    with pytest.raises(AssetImportError) as exc:
        import_generated_assets(
            not_zip, RUN_ID, project_root=tmp_path, expected_size=EXPECTED_SIZE
        )
    assert exc.value.code == "im09"


def test_run_id_mismatch_raises_im04(tmp_path):
    run_dir = _run_dir(tmp_path)
    _write_jobs(run_dir, [_job("sc01", run_id="other-run")], run_id="other-run")
    zip_path = _make_zip(tmp_path / "assets.zip", [("sc01_main.png", _png_bytes())])
    with pytest.raises(AssetImportError) as exc:
        import_generated_assets(
            zip_path, RUN_ID, project_root=tmp_path, expected_size=EXPECTED_SIZE
        )
    assert exc.value.code == "im04"


def test_bad_schema_raises_im03(tmp_path):
    run_dir = _run_dir(tmp_path)
    _write_jobs(run_dir, [_job("sc01")], schema="image-jobs/9.9")
    zip_path = _make_zip(tmp_path / "assets.zip", [("sc01_main.png", _png_bytes())])
    with pytest.raises(AssetImportError) as exc:
        import_generated_assets(
            zip_path, RUN_ID, project_root=tmp_path, expected_size=EXPECTED_SIZE
        )
    assert exc.value.code == "im03"


def test_malformed_jobs_json_raises_im02(tmp_path):
    run_dir = _run_dir(tmp_path)
    run_dir.mkdir(parents=True)
    (run_dir / "image_jobs.json").write_text("{ not json", encoding="utf-8")
    zip_path = _make_zip(tmp_path / "assets.zip", [("sc01_main.png", _png_bytes())])
    with pytest.raises(AssetImportError) as exc:
        import_generated_assets(
            zip_path, RUN_ID, project_root=tmp_path, expected_size=EXPECTED_SIZE
        )
    assert exc.value.code == "im02"


def test_missing_jobs_file_raises_im01(tmp_path):
    zip_path = _make_zip(tmp_path / "assets.zip", [("sc01_main.png", _png_bytes())])
    with pytest.raises(AssetImportError) as exc:
        import_generated_assets(
            zip_path, RUN_ID, project_root=tmp_path, expected_size=EXPECTED_SIZE
        )
    assert exc.value.code == "im01"


def test_empty_jobs_raises_im16(tmp_path):
    run_dir = _run_dir(tmp_path)
    _write_jobs(run_dir, [])
    zip_path = _make_zip(tmp_path / "assets.zip", [("sc01_main.png", _png_bytes())])
    with pytest.raises(AssetImportError) as exc:
        import_generated_assets(
            zip_path, RUN_ID, project_root=tmp_path, expected_size=EXPECTED_SIZE
        )
    assert exc.value.code == "im16"


# --- CLI ---------------------------------------------------------------------


def _cli_args(tmp_path, zip_path, *extra):
    return [
        "--run-id",
        RUN_ID,
        "--zip",
        str(zip_path),
        "--project-root",
        str(tmp_path),
        "--expected-width",
        str(EXPECTED_SIZE[0]),
        "--expected-height",
        str(EXPECTED_SIZE[1]),
        *extra,
    ]


def test_cli_success_returns_0(tmp_path, capsys):
    run_dir = _run_dir(tmp_path)
    _write_jobs(run_dir, [_job("sc01")])
    zip_path = _make_zip(tmp_path / "assets.zip", [("sc01_main.png", _png_bytes())])

    assert import_main(_cli_args(tmp_path, zip_path)) == 0
    assert (run_dir / "assets" / "sc01_main.png").is_file()
    out = capsys.readouterr().out
    assert "status=imported" in out


def test_cli_failure_returns_2(tmp_path, capsys):
    run_dir = _run_dir(tmp_path)
    _write_jobs(run_dir, [_job("sc01")])
    zip_path = _make_zip(tmp_path / "assets.zip", [("sc01_main.png", b"bad")])

    assert import_main(_cli_args(tmp_path, zip_path)) == 2
    out = capsys.readouterr().out
    assert "status=failed" in out


def test_cli_input_error_returns_2(tmp_path, capsys):
    run_dir = _run_dir(tmp_path)
    _write_jobs(run_dir, [_job("sc01")])
    assert import_main(_cli_args(tmp_path, tmp_path / "missing.zip")) == 2
    out = capsys.readouterr().out
    assert "[im08]" in out


def test_cli_dry_run_returns_0_and_writes_nothing(tmp_path, capsys):
    run_dir = _run_dir(tmp_path)
    _write_jobs(run_dir, [_job("sc01")])
    zip_path = _make_zip(tmp_path / "assets.zip", [("sc01_main.png", _png_bytes())])

    assert import_main(_cli_args(tmp_path, zip_path, "--dry-run")) == 0
    assert not (run_dir / "assets").exists()
    assert not (run_dir / REPORT_FILENAME).exists()
    out = capsys.readouterr().out
    assert "no files written" in out


# --- real contract -----------------------------------------------------------


@pytest.mark.skipif(not REAL_JOBS.is_file(), reason="live image-job artifact missing")
def test_real_contract_derives_expected_assets(tmp_path):
    raw = json.loads(REAL_JOBS.read_text(encoding="utf-8"))
    asset_ids = [job["asset_id"] for job in raw["jobs"]]
    run_dir = _run_dir(tmp_path, REAL_RUN_ID)
    run_dir.mkdir(parents=True)
    (run_dir / "image_jobs.json").write_text(
        REAL_JOBS.read_text(encoding="utf-8"), encoding="utf-8"
    )
    zip_path = _make_zip(
        tmp_path / "assets.zip",
        [(f"{asset_id}.png", _png_bytes()) for asset_id in asset_ids],
    )

    report = import_generated_assets(
        zip_path, REAL_RUN_ID, project_root=tmp_path, expected_size=EXPECTED_SIZE
    )

    assert report.status == "imported"
    assert len(report.assets) == len(asset_ids)
    assert {a["asset_id"] for a in report.assets} == set(asset_ids)
    assert report.unexpected == []
