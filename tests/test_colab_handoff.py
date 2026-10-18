"""Tests for the Colab SDXL handoff contract helpers.

These tests exercise ``agents/image_jobs/handoff.py`` offline: temp files and
synthetic ZIPs only, no network, no Drive, no GPU, no real run artifacts.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

import pytest

from agents.image_jobs import handoff
from agents.image_jobs.handoff import (
    DONE_MARKER,
    ERROR_CODES,
    ERROR_STAGE_HINTS,
    HANDOFF_ROOT_NAME,
    IMAGE_JOBS_FILENAME,
    INPUT_MANIFEST_FILENAME,
    OUTPUT_MANIFEST_FILENAME,
    OUTPUT_STATUSES,
    READY_MARKER,
    RECOVERABLE_ERROR_CODES,
    SCHEMA_INPUT_MANIFEST,
    SCHEMA_OUTPUT_MANIFEST,
    SCHEMA_RESULT_MANIFEST,
    SCHEMA_STATUS,
    STAGES,
    STATUS_FILENAME,
    UNRECOVERABLE_ERROR_CODES,
)
from agents.image_jobs.jobs import JOB_SCHEMA_VERSION

RUN_ID = "run-hoff-01"
ATTEMPT_ID = "run-hoff-01-attempt-0001"
RESOLUTION = (768, 1344)


# --- helpers -----------------------------------------------------------------


def _job(scene_id: str, *, run_id=RUN_ID, asset_id=None, output_path=None) -> dict:
    asset_id = asset_id if asset_id is not None else f"{scene_id}_main"
    output_path = (
        output_path
        if output_path is not None
        else f"data/{run_id}/assets/{asset_id}.png"
    )
    return {
        "scene_id": scene_id,
        "positive_prompt": "style, coins on a desk",
        "negative_prompt": "blurry",
        "seed": 7,
        "asset_id": asset_id,
        "output_path": output_path,
    }


def _write_jobs(path: Path, jobs, *, run_id=RUN_ID, schema=JOB_SCHEMA_VERSION) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema": schema, "run_id": run_id, "jobs": jobs}),
        encoding="utf-8",
    )
    return path


def _jobs_path(tmp_path: Path, jobs=None, **kwargs) -> Path:
    jobs = jobs if jobs is not None else [_job("sc01")]
    return _write_jobs(tmp_path / "data" / RUN_ID / IMAGE_JOBS_FILENAME, jobs, **kwargs)


def _make_zip(path: Path, members: dict, *, compression=zipfile.ZIP_STORED) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _output_fixtures(
    tmp_path,
    *,
    jobs=None,
    generated=None,
    members=None,
    result_filename="result.json",
):
    jobs = jobs if jobs is not None else [_job("sc01")]
    generated = list(generated) if generated is not None else ["sc01_main"]
    if members is None:
        members = [f"{a}.png" for a in generated] + [result_filename]
    jobs_path = _jobs_path(tmp_path, jobs)
    input_manifest = handoff.build_input_manifest(
        RUN_ID, jobs_path, attempt_id=ATTEMPT_ID
    )
    zip_path = _make_zip(
        tmp_path / "output" / f"{RUN_ID}_assets.zip",
        {m: b"png-bytes" for m in members},
    )
    output_manifest = handoff.build_output_manifest(
        RUN_ID,
        attempt_id=ATTEMPT_ID,
        image_jobs_sha256=input_manifest["image_jobs_sha256"],
        zip_path=zip_path,
        members=members,
        generated_asset_ids=generated,
        result_filename=result_filename,
    )
    return zip_path, input_manifest, output_manifest


# --- constants / schemas -----------------------------------------------------


def test_schema_strings_are_versioned_and_distinct():
    schemas = {
        SCHEMA_INPUT_MANIFEST,
        SCHEMA_OUTPUT_MANIFEST,
        SCHEMA_STATUS,
        SCHEMA_RESULT_MANIFEST,
    }
    assert len(schemas) == 4
    for schema in schemas:
        assert re.match(r"^[a-z0-9-]+/\d+\.\d+$", schema)


def test_schemas_are_1_1():
    assert SCHEMA_INPUT_MANIFEST.endswith("/1.1")
    assert SCHEMA_OUTPUT_MANIFEST.endswith("/1.1")
    assert SCHEMA_STATUS.endswith("/1.1")
    assert SCHEMA_RESULT_MANIFEST == "image-generation-results/1.0"


def test_image_jobs_schema_mirrors_contract():
    assert handoff.JOB_SCHEMA_VERSION == JOB_SCHEMA_VERSION == "image-jobs/1.0"


def test_default_generation_config_matches_approved_decisions():
    config = handoff.default_generation_config()
    assert config["checkpoint"] == "sdxl_lightning_4step_unet.safetensors"
    assert config["steps"] == 4
    assert config["guidance_scale"] == 0.0
    assert config["dtype"] == "float16"
    assert (config["width"], config["height"]) == RESOLUTION
    assert handoff.generation_config_problems(config) == []


def test_generation_config_problems_detects_wrong_steps():
    config = handoff.default_generation_config()
    config["steps"] = 8
    problems = handoff.generation_config_problems(config)
    assert any("steps" in p for p in problems)


def test_error_code_partition_is_complete_and_disjoint():
    recoverable = set(RECOVERABLE_ERROR_CODES)
    unrecoverable = set(UNRECOVERABLE_ERROR_CODES)
    assert recoverable.isdisjoint(unrecoverable)
    assert recoverable | unrecoverable == set(ERROR_CODES)


def test_output_statuses_is_completed_only():
    assert OUTPUT_STATUSES == ("completed",)


def test_stages_and_statuses_are_unique():
    assert len(STAGES) == len(set(STAGES))
    assert len(handoff.OVERALL_STATUSES) == len(set(handoff.OVERALL_STATUSES))


def test_gpu_threshold_constants():
    assert handoff.VRAM_TOTAL_MIN_GIB == 14.0
    assert handoff.VRAM_FREE_MIN_GIB == 10.0


def test_marker_and_folder_names():
    assert READY_MARKER == "READY"
    assert DONE_MARKER == "DONE"
    assert HANDOFF_ROOT_NAME == "ai-youtube-channel-handoff"


def test_new_attempt_id_is_unique_and_prefixed():
    first = handoff.new_attempt_id(RUN_ID)
    second = handoff.new_attempt_id(RUN_ID)
    assert first.startswith(RUN_ID + "-")
    assert first != second


def test_new_attempt_id_requires_run_id():
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.new_attempt_id("")
    assert exc.value.code == "hf24"


# --- hashing -----------------------------------------------------------------


def test_sha256_bytes_matches_hashlib():
    assert handoff.sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()


def test_sha256_file_and_fingerprint(tmp_path):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"payload")
    assert handoff.sha256_file(path) == hashlib.sha256(b"payload").hexdigest()
    assert handoff.file_size(path) == 7
    assert handoff.file_fingerprint(path) == {
        "filename": "blob.bin",
        "sha256": hashlib.sha256(b"payload").hexdigest(),
        "bytes": 7,
    }


# --- JSON I/O ----------------------------------------------------------------


def test_load_json_object_roundtrip(tmp_path):
    path = handoff.dump_json({"a": 1}, tmp_path / "nested" / "doc.json")
    assert handoff.load_json_object(path) == {"a": 1}


def test_load_json_object_missing_file(tmp_path):
    with pytest.raises(handoff.HandoffError) as exc:
        handoff.load_json_object(tmp_path / "nope.json")
    assert exc.value.code == "hf01"


def test_load_json_object_malformed(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.load_json_object(path)
    assert exc.value.code == "hf02"


def test_load_json_object_non_object(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.load_json_object(path)
    assert exc.value.code == "hf02"


# --- load_image_jobs ---------------------------------------------------------


def test_load_image_jobs_valid(tmp_path):
    path = _jobs_path(tmp_path, [_job("sc01"), _job("sc02")])
    data = handoff.load_image_jobs(path)
    assert handoff.expected_asset_ids(data) == ["sc01_main", "sc02_main"]


def test_load_image_jobs_wrong_schema(tmp_path):
    path = _jobs_path(tmp_path, [_job("sc01")], schema="image-jobs/9.9")
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.load_image_jobs(path)
    assert exc.value.code == "hf03"


def test_load_image_jobs_missing_run_id(tmp_path):
    path = _jobs_path(tmp_path, [_job("sc01")], run_id="")
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.load_image_jobs(path)
    assert exc.value.code == "hf03"


def test_load_image_jobs_empty_jobs(tmp_path):
    path = _jobs_path(tmp_path, [])
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.load_image_jobs(path)
    assert exc.value.code == "hf03"


def test_load_image_jobs_missing_field(tmp_path):
    job = _job("sc01")
    del job["seed"]
    path = _jobs_path(tmp_path, [job])
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.load_image_jobs(path)
    assert exc.value.code == "hf03"


def test_load_image_jobs_unsafe_asset_id(tmp_path):
    path = _jobs_path(tmp_path, [_job("sc01", asset_id="..evil")])
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.load_image_jobs(path)
    assert exc.value.code == "hf05"


def test_load_image_jobs_duplicate_asset_id(tmp_path):
    jobs = [_job("sc01"), _job("sc02", asset_id="sc01_main")]
    path = _jobs_path(tmp_path, jobs)
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.load_image_jobs(path)
    assert exc.value.code == "hf05"


def test_load_image_jobs_unsafe_output_path_is_absolute(tmp_path):
    job = _job(
        "sc01", asset_id="sc01_main", output_path="C:/abs/sc01_main.png"
    )
    path = _jobs_path(tmp_path, [job])
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.load_image_jobs(path)
    assert exc.value.code == "hf06"


def test_load_image_jobs_unsafe_output_path_traversal(tmp_path):
    job = _job(
        "sc01", asset_id="sc01_main", output_path="data/../sc01_main.png"
    )
    path = _jobs_path(tmp_path, [job])
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.load_image_jobs(path)
    assert exc.value.code == "hf06"


def test_load_image_jobs_wrong_extension(tmp_path):
    job = _job(
        "sc01",
        asset_id="sc01_main",
        output_path=f"data/{RUN_ID}/assets/sc01_main.jpg",
    )
    path = _jobs_path(tmp_path, [job])
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.load_image_jobs(path)
    assert exc.value.code == "hf06"


def test_load_image_jobs_stem_mismatch(tmp_path):
    job = _job(
        "sc01",
        asset_id="sc01_main",
        output_path=f"data/{RUN_ID}/assets/other.png",
    )
    path = _jobs_path(tmp_path, [job])
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.load_image_jobs(path)
    assert exc.value.code == "hf06"


@pytest.mark.parametrize(
    "output_path",
    [
        "foo/sc01_main.png",
        "data/x/sc01_main.png",
        "assets/sc01_main.png",
        "sc01_main.png",
    ],
)
def test_load_image_jobs_output_path_wrong_parent(tmp_path, output_path):
    job = _job("sc01", asset_id="sc01_main", output_path=output_path)
    path = _jobs_path(tmp_path, [job])
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.load_image_jobs(path)
    assert exc.value.code == "hf07"


# --- input manifest ----------------------------------------------------------


def test_build_input_manifest_fields(tmp_path):
    path = _jobs_path(tmp_path, [_job("sc01"), _job("sc02")])
    manifest = handoff.build_input_manifest(RUN_ID, path, attempt_id=ATTEMPT_ID)

    assert manifest["schema"] == SCHEMA_INPUT_MANIFEST
    assert manifest["run_id"] == RUN_ID
    assert manifest["attempt_id"] == ATTEMPT_ID
    assert "input_sha256" not in manifest
    assert manifest["image_jobs"]["filename"] == IMAGE_JOBS_FILENAME
    assert manifest["image_jobs"]["schema"] == JOB_SCHEMA_VERSION
    assert manifest["image_jobs"]["bytes"] == handoff.file_size(path)
    assert manifest["image_jobs_sha256"] == handoff.sha256_file(path)
    assert manifest["job_count"] == 2
    assert manifest["expected_asset_ids"] == ["sc01_main", "sc02_main"]
    assert manifest["expected_resolution"] == {"width": 768, "height": 1344}
    assert manifest["generation_config"]["steps"] == 4


def test_build_input_manifest_requires_attempt_id(tmp_path):
    path = _jobs_path(tmp_path, [_job("sc01")])
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.build_input_manifest(RUN_ID, path, attempt_id="")
    assert exc.value.code == "hf10"


def test_build_input_manifest_run_id_mismatch(tmp_path):
    path = _jobs_path(tmp_path, [_job("sc01")])
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.build_input_manifest("different-run", path, attempt_id=ATTEMPT_ID)
    assert exc.value.code == "hf04"


def test_verify_input_manifest_success(tmp_path):
    path = _jobs_path(tmp_path, [_job("sc01"), _job("sc02")])
    manifest = handoff.build_input_manifest(RUN_ID, path, attempt_id=ATTEMPT_ID)
    handoff.verify_input_manifest(
        manifest,
        image_jobs_path=path,
        expected_run_id=RUN_ID,
        expected_attempt_id=ATTEMPT_ID,
    )


def test_verify_input_manifest_structural_only(tmp_path):
    path = _jobs_path(tmp_path, [_job("sc01")])
    manifest = handoff.build_input_manifest(RUN_ID, path, attempt_id=ATTEMPT_ID)
    handoff.verify_input_manifest(manifest, expected_run_id=RUN_ID)


def test_verify_input_manifest_wrong_schema(tmp_path):
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.verify_input_manifest({"schema": "nope"})
    assert exc.value.code == "hf10"


def test_verify_input_manifest_rejects_legacy_input_sha256(tmp_path):
    path = _jobs_path(tmp_path, [_job("sc01")])
    manifest = handoff.build_input_manifest(RUN_ID, path, attempt_id=ATTEMPT_ID)
    manifest["input_sha256"] = "deadbeef"
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.verify_input_manifest(manifest, expected_run_id=RUN_ID)
    assert exc.value.code == "hf16"


def test_verify_input_manifest_missing_attempt_id(tmp_path):
    path = _jobs_path(tmp_path, [_job("sc01")])
    manifest = handoff.build_input_manifest(RUN_ID, path, attempt_id=ATTEMPT_ID)
    del manifest["attempt_id"]
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.verify_input_manifest(manifest, expected_run_id=RUN_ID)
    assert exc.value.code == "hf10"


def test_verify_input_manifest_run_id_mismatch(tmp_path):
    path = _jobs_path(tmp_path, [_job("sc01")])
    manifest = handoff.build_input_manifest(RUN_ID, path, attempt_id=ATTEMPT_ID)
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_input_manifest(manifest, expected_run_id="other")
    assert exc.value.code == "hf11"


def test_verify_input_manifest_attempt_mismatch(tmp_path):
    path = _jobs_path(tmp_path, [_job("sc01")])
    manifest = handoff.build_input_manifest(RUN_ID, path, attempt_id=ATTEMPT_ID)
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_input_manifest(
            manifest, expected_run_id=RUN_ID, expected_attempt_id="other"
        )
    assert exc.value.code == "hf25"


def test_verify_input_manifest_resolution_mismatch(tmp_path):
    path = _jobs_path(tmp_path, [_job("sc01")])
    manifest = handoff.build_input_manifest(RUN_ID, path, attempt_id=ATTEMPT_ID)
    manifest["expected_resolution"] = {"width": 512, "height": 512}
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_input_manifest(manifest, expected_run_id=RUN_ID)
    assert exc.value.code == "hf12"


def test_verify_input_manifest_stale_hash(tmp_path):
    path = _jobs_path(tmp_path, [_job("sc01")])
    manifest = handoff.build_input_manifest(RUN_ID, path, attempt_id=ATTEMPT_ID)
    _write_jobs(path, [_job("sc01"), _job("sc02")])
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_input_manifest(
            manifest, image_jobs_path=path, expected_run_id=RUN_ID
        )
    assert exc.value.code == "hf13"


def test_verify_input_manifest_byte_mismatch(tmp_path):
    path = _jobs_path(tmp_path, [_job("sc01")])
    manifest = handoff.build_input_manifest(RUN_ID, path, attempt_id=ATTEMPT_ID)
    manifest["image_jobs"]["bytes"] = 999999
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_input_manifest(
            manifest, image_jobs_path=path, expected_run_id=RUN_ID
        )
    assert exc.value.code == "hf13"


def test_verify_input_manifest_filename_mismatch(tmp_path):
    path = _jobs_path(tmp_path, [_job("sc01")])
    manifest = handoff.build_input_manifest(RUN_ID, path, attempt_id=ATTEMPT_ID)
    manifest["image_jobs"]["filename"] = "somewhere_else.json"
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_input_manifest(
            manifest, image_jobs_path=path, expected_run_id=RUN_ID
        )
    assert exc.value.code == "hf15"


def test_verify_input_manifest_ids_mismatch(tmp_path):
    path = _jobs_path(tmp_path, [_job("sc01")])
    manifest = handoff.build_input_manifest(RUN_ID, path, attempt_id=ATTEMPT_ID)
    manifest["expected_asset_ids"] = ["sc99_main"]
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_input_manifest(
            manifest, image_jobs_path=path, expected_run_id=RUN_ID
        )
    assert exc.value.code == "hf14"


def test_verify_input_manifest_job_count_mismatch(tmp_path):
    path = _jobs_path(tmp_path, [_job("sc01")])
    manifest = handoff.build_input_manifest(RUN_ID, path, attempt_id=ATTEMPT_ID)
    manifest["job_count"] = 5
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_input_manifest(
            manifest, image_jobs_path=path, expected_run_id=RUN_ID
        )
    assert exc.value.code == "hf14"


# --- output manifest ---------------------------------------------------------


def test_build_output_manifest_fields(tmp_path):
    zip_path, input_manifest, manifest = _output_fixtures(tmp_path)
    assert manifest["schema"] == SCHEMA_OUTPUT_MANIFEST
    assert manifest["run_id"] == RUN_ID
    assert manifest["attempt_id"] == ATTEMPT_ID
    assert "input_sha256" not in manifest
    assert manifest["image_jobs_sha256"] == input_manifest["image_jobs_sha256"]
    assert manifest["status"] == "completed"
    assert manifest["zip"]["filename"] == zip_path.name
    assert manifest["zip"]["sha256"] == handoff.sha256_file(zip_path)
    assert manifest["members"] == ["sc01_main.png", "result.json"]
    assert manifest["result_manifest"] == {
        "filename": "result.json",
        "schema": SCHEMA_RESULT_MANIFEST,
    }


def test_build_output_manifest_missing_zip(tmp_path):
    with pytest.raises(handoff.HandoffError) as exc:
        handoff.build_output_manifest(
            RUN_ID,
            attempt_id=ATTEMPT_ID,
            image_jobs_sha256="x",
            zip_path=tmp_path / "missing.zip",
            members=["a.png"],
            generated_asset_ids=["a"],
        )
    assert exc.value.code == "hf30"


def test_build_output_manifest_invalid_status(tmp_path):
    zip_path = _make_zip(tmp_path / "assets.zip", {"a.png": b"x"})
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.build_output_manifest(
            RUN_ID,
            attempt_id=ATTEMPT_ID,
            image_jobs_sha256="x",
            zip_path=zip_path,
            members=["a.png"],
            generated_asset_ids=["a"],
            status="partial",
        )
    assert exc.value.code == "hf45"


def test_verify_output_manifest_success(tmp_path):
    zip_path, input_manifest, manifest = _output_fixtures(tmp_path)
    members = handoff.verify_output_manifest(
        manifest,
        expected_run_id=RUN_ID,
        expected_attempt_id=ATTEMPT_ID,
        expected_image_jobs_sha256=input_manifest["image_jobs_sha256"],
        zip_path=zip_path,
    )
    assert members == ["result.json", "sc01_main.png"]


def test_verify_output_manifest_without_zip_returns_manifest_members(tmp_path):
    _, _, manifest = _output_fixtures(tmp_path)
    members = handoff.verify_output_manifest(manifest, expected_run_id=RUN_ID)
    assert members == ["sc01_main.png", "result.json"]


def test_verify_output_manifest_wrong_schema(tmp_path):
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.verify_output_manifest({"schema": "nope"})
    assert exc.value.code == "hf17"


def test_verify_output_manifest_rejects_legacy_input_sha256(tmp_path):
    _, _, manifest = _output_fixtures(tmp_path)
    manifest["input_sha256"] = "deadbeef"
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.verify_output_manifest(manifest, expected_run_id=RUN_ID)
    assert exc.value.code == "hf16"


def test_verify_output_manifest_run_id_mismatch(tmp_path):
    _, _, manifest = _output_fixtures(tmp_path)
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_output_manifest(manifest, expected_run_id="other")
    assert exc.value.code == "hf18"


def test_verify_output_manifest_missing_attempt_id(tmp_path):
    _, _, manifest = _output_fixtures(tmp_path)
    del manifest["attempt_id"]
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.verify_output_manifest(manifest, expected_run_id=RUN_ID)
    assert exc.value.code == "hf24"


def test_verify_output_manifest_attempt_mismatch(tmp_path):
    _, _, manifest = _output_fixtures(tmp_path)
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_output_manifest(
            manifest, expected_run_id=RUN_ID, expected_attempt_id="other"
        )
    assert exc.value.code == "hf28"


def test_verify_output_manifest_input_sha_mismatch(tmp_path):
    _, _, manifest = _output_fixtures(tmp_path)
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_output_manifest(
            manifest, expected_image_jobs_sha256="deadbeef"
        )
    assert exc.value.code == "hf19"


def test_verify_output_manifest_invalid_status(tmp_path):
    _, _, manifest = _output_fixtures(tmp_path)
    manifest["status"] = "partial"
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.verify_output_manifest(manifest, expected_run_id=RUN_ID)
    assert exc.value.code == "hf45"


def test_verify_output_manifest_bad_result_schema(tmp_path):
    _, _, manifest = _output_fixtures(tmp_path)
    manifest["result_manifest"]["schema"] = "bogus/1.0"
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.verify_output_manifest(manifest, expected_run_id=RUN_ID)
    assert exc.value.code == "hf47"


def test_verify_output_manifest_unsafe_zip_filename(tmp_path):
    _, _, manifest = _output_fixtures(tmp_path)
    manifest["zip"]["filename"] = "../evil.zip"
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.verify_output_manifest(manifest, expected_run_id=RUN_ID)
    assert exc.value.code == "hf27"


def test_verify_output_manifest_member_set_mismatch(tmp_path):
    zip_path, input_manifest, manifest = _output_fixtures(tmp_path)
    manifest["generated_asset_ids"] = ["sc01_main", "ghost"]
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_output_manifest(
            manifest,
            expected_image_jobs_sha256=input_manifest["image_jobs_sha256"],
            zip_path=zip_path,
        )
    assert exc.value.code == "hf35"


def test_verify_output_manifest_duplicate_generated_ids(tmp_path):
    _, _, manifest = _output_fixtures(tmp_path)
    manifest["generated_asset_ids"] = ["sc01_main", "sc01_main"]
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_output_manifest(manifest, expected_run_id=RUN_ID)
    assert exc.value.code == "hf43"


def test_verify_output_manifest_zip_hash_mismatch(tmp_path):
    zip_path, input_manifest, manifest = _output_fixtures(tmp_path)
    manifest["zip"]["sha256"] = "0" * 64
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_output_manifest(
            manifest,
            expected_image_jobs_sha256=input_manifest["image_jobs_sha256"],
            zip_path=zip_path,
        )
    assert exc.value.code == "hf32"


# --- handoff pair ------------------------------------------------------------


def test_verify_handoff_pair_success(tmp_path):
    zip_path, input_manifest, output_manifest = _output_fixtures(tmp_path)
    jobs_path = tmp_path / "data" / RUN_ID / IMAGE_JOBS_FILENAME
    handoff.verify_handoff_pair(
        input_manifest,
        output_manifest,
        image_jobs_path=jobs_path,
        expected_run_id=RUN_ID,
    )


def test_verify_handoff_pair_run_id_mismatch(tmp_path):
    _, input_manifest, output_manifest = _output_fixtures(tmp_path)
    output_manifest["run_id"] = "other"
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_handoff_pair(input_manifest, output_manifest)
    assert exc.value.code in ("hf18", "hf44")


def test_verify_handoff_pair_hash_mismatch(tmp_path):
    _, input_manifest, output_manifest = _output_fixtures(tmp_path)
    output_manifest["image_jobs_sha256"] = "0" * 64
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_handoff_pair(input_manifest, output_manifest)
    assert exc.value.code == "hf19"


def test_verify_handoff_pair_attempt_mismatch(tmp_path):
    _, input_manifest, output_manifest = _output_fixtures(tmp_path)
    output_manifest["attempt_id"] = input_manifest["attempt_id"] + "-x"
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_handoff_pair(input_manifest, output_manifest)
    assert exc.value.code == "hf28"


def test_verify_handoff_pair_resolution_mismatch(tmp_path):
    _, input_manifest, output_manifest = _output_fixtures(tmp_path)
    output_manifest["resolution"] = {"width": 512, "height": 512}
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_handoff_pair(input_manifest, output_manifest)
    assert exc.value.code == "hf29"


def test_verify_handoff_pair_generation_config_mismatch(tmp_path):
    _, input_manifest, output_manifest = _output_fixtures(tmp_path)
    output_manifest["generation_config"] = handoff.default_generation_config()
    output_manifest["generation_config"]["steps"] = 8
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_handoff_pair(input_manifest, output_manifest)
    assert exc.value.code == "hf42"


def test_verify_handoff_pair_asset_missing(tmp_path):
    zip_path, input_manifest, output_manifest = _output_fixtures(tmp_path)
    output_manifest["generated_asset_ids"] = []
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.verify_handoff_pair(input_manifest, output_manifest)
    assert exc.value.code == "hf17"


def test_verify_handoff_pair_asset_unexpected(tmp_path):
    zip_path, input_manifest, output_manifest = _output_fixtures(tmp_path)
    output_manifest["generated_asset_ids"] = ["sc01_main", "sc99_main"]
    output_manifest["members"] = [
        "sc01_main.png",
        "sc99_main.png",
        "result.json",
    ]
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_handoff_pair(input_manifest, output_manifest)
    assert exc.value.code == "hf43"


# --- status / error ----------------------------------------------------------


def _status(**overrides):
    kwargs = dict(
        attempt_id=ATTEMPT_ID,
        image_jobs_sha256=_sha("jobs"),
        overall_status="PENDING",
        current_stage="gpu_preflight",
    )
    kwargs.update(overrides)
    return handoff.build_status(RUN_ID, **kwargs)


def _pass_status(**overrides):
    kwargs = dict(
        attempt_id=ATTEMPT_ID,
        image_jobs_sha256=_sha("jobs"),
        overall_status="PASS",
        current_stage="upload",
        completed_stages=list(STAGES),
        completed_asset_ids=["sc01_main"],
    )
    kwargs.update(overrides)
    return handoff.build_status(RUN_ID, **kwargs)


def test_build_and_validate_pending_status():
    status = _status()
    assert status["schema"] == SCHEMA_STATUS
    assert status["run_id"] == RUN_ID
    assert status["attempt_id"] == ATTEMPT_ID
    assert "input_sha256" not in status
    assert status["overall_status"] == "PENDING"
    assert status["error"] is None
    handoff.validate_status(status)


def test_validate_status_rejects_legacy_input_sha256():
    status = _status()
    status["input_sha256"] = "deadbeef"
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.validate_status(status)
    assert exc.value.code == "hf41"


def test_validate_status_missing_attempt_id():
    status = _status()
    del status["attempt_id"]
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.validate_status(status)
    assert exc.value.code == "hf41"


def test_validate_pending_with_completed_stage_fails():
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.validate_status(_status(completed_stages=["gpu_preflight"]))
    assert exc.value.code == "hf41"


def test_validate_running_next_stage():
    handoff.validate_status(
        _status(
            overall_status="RUNNING",
            current_stage="generation",
            completed_stages=["gpu_preflight", "env_setup", "model_load"],
        )
    )


def test_validate_running_wrong_stage_fails():
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.validate_status(
            _status(
                overall_status="RUNNING",
                current_stage="upload",
                completed_stages=["gpu_preflight"],
            )
        )
    assert exc.value.code == "hf41"


def test_validate_pass_status():
    handoff.validate_status(
        _pass_status(), expected_asset_ids=["sc01_main"]
    )


def test_validate_pass_requires_expected_asset_ids():
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.validate_status(_pass_status())
    assert exc.value.code == "hf41"


def test_validate_pass_incomplete_stages_fails():
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.validate_status(
            _pass_status(completed_stages=list(STAGES[:-1])),
            expected_asset_ids=["sc01_main"],
        )
    assert exc.value.code == "hf41"


def test_validate_pass_incomplete_assets_fails():
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.validate_status(
            _pass_status(completed_asset_ids=[]),
            expected_asset_ids=["sc01_main"],
        )
    assert exc.value.code == "hf41"


def test_validate_blocked_status_requires_error():
    error = handoff.make_error("NO_GPU", "gpu_preflight", "no GPU assigned")
    handoff.validate_status(_status(overall_status="BLOCKED", error=error))


def test_validate_blocked_without_error_fails():
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.validate_status(_status(overall_status="BLOCKED"))
    assert exc.value.code == "hf41"


def test_validate_blocked_error_stage_must_match_current():
    error = handoff.make_error("ENV_SETUP_FAILED", "env_setup", "x")
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.validate_status(
            _status(
                overall_status="BLOCKED",
                current_stage="gpu_preflight",
                error=error,
            )
        )
    assert exc.value.code == "hf41"


def test_validate_failed_status():
    error = handoff.make_error(
        "OUTPUT_VALIDATION_FAILED", "output_validation", "bad output"
    )
    completed = list(STAGES[: STAGES.index("output_validation")])
    handoff.validate_status(
        _status(
            overall_status="FAILED",
            current_stage="output_validation",
            completed_stages=completed,
            error=error,
        )
    )


def test_validate_pass_with_error_fails():
    error = handoff.make_error("NO_GPU", "gpu_preflight", "x")
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.validate_status(
            _pass_status(error=error), expected_asset_ids=["sc01_main"]
        )
    assert exc.value.code == "hf41"


def test_validate_invalid_overall_status():
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.validate_status(_status(overall_status="DONE"))
    assert exc.value.code == "hf41"


def test_validate_invalid_stage():
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.validate_status(_status(current_stage="nope"))
    assert exc.value.code == "hf41"


def test_validate_completed_stages_must_be_known():
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.validate_status(_status(completed_stages=["bogus"]))
    assert exc.value.code == "hf41"


def test_validate_completed_stages_must_be_prefix():
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.validate_status(
            _status(
                overall_status="RUNNING",
                current_stage="model_load",
                completed_stages=["env_setup"],
            )
        )
    assert exc.value.code == "hf41"


def test_make_error_unknown_code():
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.make_error("BOGUS", "gpu_preflight", "x")
    assert exc.value.code == "hf40"


def test_make_error_unknown_stage():
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.make_error("NO_GPU", "bogus", "x")
    assert exc.value.code == "hf40"


def test_make_error_code_stage_consistency():
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.make_error("NO_GPU", "upload", "x")
    assert exc.value.code == "hf40"


def test_make_error_marks_recoverability():
    blocked = handoff.make_error("NO_GPU", "gpu_preflight", "x")
    failed = handoff.make_error(
        "OUTPUT_VALIDATION_FAILED", "output_validation", "x"
    )
    assert blocked["recoverable"] is True
    assert failed["recoverable"] is False


def test_make_error_passes_diagnostics_and_traceback():
    error = handoff.make_error(
        "NO_GPU",
        "gpu_preflight",
        "x",
        diagnostics={"device_count": 0},
        traceback_text="trace",
    )
    assert error["diagnostics"] == {"device_count": 0}
    assert error["traceback"] == "trace"


def test_input_validation_failed_partition_and_hint():
    assert "INPUT_VALIDATION_FAILED" in ERROR_CODES
    assert ERROR_STAGE_HINTS["INPUT_VALIDATION_FAILED"] == "env_setup"
    recoverable = set(RECOVERABLE_ERROR_CODES)
    unrecoverable = set(UNRECOVERABLE_ERROR_CODES)
    assert "INPUT_VALIDATION_FAILED" in unrecoverable
    assert "INPUT_VALIDATION_FAILED" not in recoverable
    assert recoverable.isdisjoint(unrecoverable)
    assert recoverable | unrecoverable == set(ERROR_CODES)


def test_make_error_input_validation_failed_is_unrecoverable():
    error = handoff.make_error("INPUT_VALIDATION_FAILED", "env_setup", "x")
    assert error["recoverable"] is False
    assert error["stage"] == "env_setup"


def test_make_error_input_validation_failed_only_valid_at_env_setup():
    handoff.make_error("INPUT_VALIDATION_FAILED", "env_setup", "x")
    for stage in ("gpu_preflight", "model_load", "generation", "output_validation"):
        with pytest.raises(handoff.HandoffSchemaError) as exc:
            handoff.make_error("INPUT_VALIDATION_FAILED", stage, "x")
        assert exc.value.code == "hf40"


def test_validate_failed_input_validation_status():
    error = handoff.make_error("INPUT_VALIDATION_FAILED", "env_setup", "prompt long")
    handoff.validate_status(
        _status(
            overall_status="FAILED",
            current_stage="env_setup",
            completed_stages=["gpu_preflight"],
            error=error,
        )
    )


def test_stale_reader_rejects_input_validation_failed(monkeypatch):
    status = _status(
        overall_status="FAILED",
        current_stage="env_setup",
        completed_stages=["gpu_preflight"],
        error=handoff.make_error("INPUT_VALIDATION_FAILED", "env_setup", "x"),
    )
    # The updated reader accepts the new code (fail-closed for stale readers).
    handoff.validate_status(status)
    stale_codes = tuple(c for c in ERROR_CODES if c != "INPUT_VALIDATION_FAILED")
    monkeypatch.setattr(handoff, "ERROR_CODES", stale_codes)
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.validate_status(status)
    assert exc.value.code == "hf41"


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("NO_GPU", "BLOCKED"),
        ("CUDA_INIT_FAILED", "BLOCKED"),
        ("INSUFFICIENT_VRAM", "BLOCKED"),
        ("RUNTIME_DISCONNECTED", "BLOCKED"),
        ("ENV_SETUP_FAILED", "BLOCKED"),
        ("INPUT_VALIDATION_FAILED", "FAILED"),
        ("MODEL_LOAD_FAILED", "BLOCKED"),
        ("GENERATION_FAILED", "BLOCKED"),
        ("OUTPUT_VALIDATION_FAILED", "FAILED"),
    ],
)
def test_overall_status_for_error(code, expected):
    assert handoff.overall_status_for_error(code) == expected


def test_overall_status_for_error_unknown():
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.overall_status_for_error("BOGUS")
    assert exc.value.code == "hf40"


# --- markers -----------------------------------------------------------------


def test_read_write_has_marker(tmp_path):
    assert handoff.read_marker(tmp_path, READY_MARKER) is None
    assert handoff.has_marker(tmp_path, READY_MARKER) is False
    path = handoff.write_marker(tmp_path, READY_MARKER, content=ATTEMPT_ID)
    assert path.read_text(encoding="utf-8") == ATTEMPT_ID
    assert handoff.read_marker(tmp_path, READY_MARKER) == ATTEMPT_ID
    assert handoff.has_marker(tmp_path, READY_MARKER) is True


def test_begin_attempt_clears_stale_output(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    zip_path = output_dir / f"{RUN_ID}_assets.zip"
    zip_path.write_bytes(b"zip")
    handoff.write_marker(output_dir, DONE_MARKER, content=ATTEMPT_ID)
    handoff.dump_json({"x": 1}, output_dir / OUTPUT_MANIFEST_FILENAME)
    handoff.dump_json({"x": 1}, output_dir / STATUS_FILENAME)

    handoff.begin_attempt(output_dir, ATTEMPT_ID)

    assert handoff.has_marker(output_dir, DONE_MARKER) is False
    assert not (output_dir / OUTPUT_MANIFEST_FILENAME).exists()
    assert not (output_dir / STATUS_FILENAME).exists()
    assert zip_path.exists()


def test_verify_ready_missing_marker(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_ready(input_dir, run_id=RUN_ID)
    assert exc.value.code == "hf20"


def test_verify_ready_premature(tmp_path):
    input_dir = tmp_path / "input"
    _write_jobs(input_dir / IMAGE_JOBS_FILENAME, [_job("sc01")])
    manifest = handoff.build_input_manifest(
        RUN_ID, input_dir / IMAGE_JOBS_FILENAME, attempt_id=ATTEMPT_ID
    )
    handoff.dump_json(manifest, input_dir / INPUT_MANIFEST_FILENAME)
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_ready(input_dir, run_id=RUN_ID)
    assert exc.value.code == "hf20"


def test_verify_ready_success(tmp_path):
    input_dir = tmp_path / "input"
    _write_jobs(input_dir / IMAGE_JOBS_FILENAME, [_job("sc01"), _job("sc02")])
    manifest = handoff.build_input_manifest(
        RUN_ID, input_dir / IMAGE_JOBS_FILENAME, attempt_id=ATTEMPT_ID
    )
    handoff.dump_json(manifest, input_dir / INPUT_MANIFEST_FILENAME)
    handoff.write_marker(input_dir, READY_MARKER, content=ATTEMPT_ID)
    assert (
        handoff.verify_ready(
            input_dir, run_id=RUN_ID, expected_attempt_id=ATTEMPT_ID
        )["run_id"]
        == RUN_ID
    )


def test_verify_ready_marker_content_mismatch(tmp_path):
    input_dir = tmp_path / "input"
    _write_jobs(input_dir / IMAGE_JOBS_FILENAME, [_job("sc01")])
    manifest = handoff.build_input_manifest(
        RUN_ID, input_dir / IMAGE_JOBS_FILENAME, attempt_id=ATTEMPT_ID
    )
    handoff.dump_json(manifest, input_dir / INPUT_MANIFEST_FILENAME)
    handoff.write_marker(input_dir, READY_MARKER, content="other-attempt")
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_ready(input_dir, run_id=RUN_ID)
    assert exc.value.code == "hf25"


def test_verify_ready_stale_hash(tmp_path):
    input_dir = tmp_path / "input"
    _write_jobs(input_dir / IMAGE_JOBS_FILENAME, [_job("sc01")])
    manifest = handoff.build_input_manifest(
        RUN_ID, input_dir / IMAGE_JOBS_FILENAME, attempt_id=ATTEMPT_ID
    )
    handoff.dump_json(manifest, input_dir / INPUT_MANIFEST_FILENAME)
    handoff.write_marker(input_dir, READY_MARKER, content=ATTEMPT_ID)
    _write_jobs(input_dir / IMAGE_JOBS_FILENAME, [_job("sc01"), _job("sc02")])
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_ready(input_dir, run_id=RUN_ID)
    assert exc.value.code == "hf13"


def _write_output_bundle(tmp_path, *, status_overrides=None):
    zip_path, input_manifest, manifest = _output_fixtures(tmp_path)
    output_dir = zip_path.parent
    handoff.dump_json(manifest, output_dir / OUTPUT_MANIFEST_FILENAME)
    status_kwargs = dict(
        attempt_id=ATTEMPT_ID,
        image_jobs_sha256=input_manifest["image_jobs_sha256"],
        overall_status="PASS",
        current_stage="upload",
        completed_stages=list(STAGES),
        completed_asset_ids=list(input_manifest["expected_asset_ids"]),
    )
    if status_overrides:
        status_kwargs.update(status_overrides)
    status = handoff.build_status(RUN_ID, **status_kwargs)
    handoff.dump_json(status, output_dir / STATUS_FILENAME)
    handoff.write_marker(output_dir, DONE_MARKER, content=ATTEMPT_ID)
    return zip_path, input_manifest, manifest, output_dir


def test_verify_done_missing_marker(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_done(output_dir, run_id=RUN_ID)
    assert exc.value.code == "hf22"


def test_verify_done_success(tmp_path):
    zip_path, input_manifest, manifest, output_dir = _write_output_bundle(tmp_path)
    result = handoff.verify_done(
        output_dir,
        run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        image_jobs_sha256=input_manifest["image_jobs_sha256"],
        expected_asset_ids=["sc01_main"],
    )
    assert result["run_id"] == RUN_ID


def test_verify_done_marker_content_mismatch(tmp_path):
    zip_path, input_manifest, manifest, output_dir = _write_output_bundle(tmp_path)
    handoff.write_marker(output_dir, DONE_MARKER, content="other-attempt")
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_done(output_dir, run_id=RUN_ID)
    assert exc.value.code == "hf25"


def test_verify_done_missing_status(tmp_path):
    zip_path, input_manifest, manifest = _output_fixtures(tmp_path)
    output_dir = zip_path.parent
    handoff.dump_json(manifest, output_dir / OUTPUT_MANIFEST_FILENAME)
    handoff.write_marker(output_dir, DONE_MARKER, content=ATTEMPT_ID)
    with pytest.raises(handoff.HandoffError) as exc:
        handoff.verify_done(output_dir, run_id=RUN_ID)
    assert exc.value.code == "hf26"


def test_verify_done_status_attempt_mismatch(tmp_path):
    _, _, _, output_dir = _write_output_bundle(
        tmp_path, status_overrides={"attempt_id": "other-attempt"}
    )
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_done(output_dir, run_id=RUN_ID)
    assert exc.value.code == "hf26"


def test_verify_done_status_not_pass(tmp_path):
    _, _, _, output_dir = _write_output_bundle(
        tmp_path,
        status_overrides={
            "overall_status": "BLOCKED",
            "current_stage": "gpu_preflight",
            "completed_stages": [],
            "completed_asset_ids": [],
            "error": handoff.make_error("NO_GPU", "gpu_preflight", "x"),
        },
    )
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_done(output_dir, run_id=RUN_ID)
    assert exc.value.code == "hf48"


def test_verify_done_unsafe_zip_filename(tmp_path):
    zip_path, input_manifest, manifest = _output_fixtures(tmp_path)
    output_dir = zip_path.parent
    manifest["zip"]["filename"] = "../evil.zip"
    handoff.dump_json(manifest, output_dir / OUTPUT_MANIFEST_FILENAME)
    status = handoff.build_status(
        RUN_ID,
        attempt_id=ATTEMPT_ID,
        image_jobs_sha256=input_manifest["image_jobs_sha256"],
        overall_status="PASS",
        current_stage="upload",
        completed_stages=list(STAGES),
        completed_asset_ids=list(input_manifest["expected_asset_ids"]),
    )
    handoff.dump_json(status, output_dir / STATUS_FILENAME)
    handoff.write_marker(output_dir, DONE_MARKER, content=ATTEMPT_ID)
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.verify_done(output_dir, run_id=RUN_ID)
    assert exc.value.code == "hf27"


def test_verify_done_stale_done_invalidated_by_new_attempt(tmp_path):
    zip_path, input_manifest, manifest, output_dir = _write_output_bundle(tmp_path)
    assert handoff.verify_done(
        output_dir, run_id=RUN_ID, expected_asset_ids=["sc01_main"]
    )["run_id"] == RUN_ID
    handoff.begin_attempt(output_dir, "new-attempt")
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_done(output_dir, run_id=RUN_ID)
    assert exc.value.code == "hf22"


# --- ZIP integrity -----------------------------------------------------------


def test_verify_zip_success(tmp_path):
    path = _make_zip(
        tmp_path / "assets.zip",
        {"sc01_main.png": b"a", "result.json": b"{}"},
    )
    members = handoff.verify_zip(path)
    assert members == ["result.json", "sc01_main.png"]


def test_verify_zip_missing(tmp_path):
    with pytest.raises(handoff.HandoffError) as exc:
        handoff.verify_zip(tmp_path / "nope.zip")
    assert exc.value.code == "hf30"


def test_verify_zip_not_a_zip(tmp_path):
    path = tmp_path / "fake.zip"
    path.write_bytes(b"not a zip")
    with pytest.raises(handoff.HandoffError) as exc:
        handoff.verify_zip(path)
    assert exc.value.code == "hf31"


def test_verify_zip_sha_mismatch(tmp_path):
    path = _make_zip(tmp_path / "assets.zip", {"a.png": b"a"})
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_zip(path, expected_sha256="0" * 64)
    assert exc.value.code == "hf32"


def test_verify_zip_bytes_mismatch(tmp_path):
    path = _make_zip(tmp_path / "assets.zip", {"a.png": b"a"})
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_zip(path, expected_bytes=999999)
    assert exc.value.code == "hf32"


def test_verify_zip_corrupt_member(tmp_path):
    payload = b"the-quick-brown-fox-payload"
    path = _make_zip(tmp_path / "assets.zip", {"a.png": payload})
    raw = path.read_bytes()
    corrupted = raw.replace(payload, b"XXX-quick-brown-fox-payload")
    assert corrupted != raw
    path.write_bytes(corrupted)
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_zip(path)
    assert exc.value.code == "hf33"


def test_verify_zip_unsafe_member(tmp_path):
    path = _make_zip(tmp_path / "assets.zip", {"../evil.png": b"x"})
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_zip(path)
    assert exc.value.code == "hf34"


def test_verify_zip_duplicate_normalized_member(tmp_path):
    path = _make_zip(
        tmp_path / "assets.zip",
        {"a.png": b"x", "./a.png": b"y"},
    )
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_zip(path)
    assert exc.value.code == "hf36"


def test_verify_zip_member_set_mismatch(tmp_path):
    path = _make_zip(tmp_path / "assets.zip", {"a.png": b"a"})
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_zip(path, expected_members=["a.png", "b.png"])
    assert exc.value.code == "hf35"


# --- result document (image-generation-results/1.0) ---------------------------


def _result_item(asset_id="sc01_main", **overrides) -> dict:
    item = {
        "scene_id": "sc01",
        "asset_id": asset_id,
        "filename": f"{asset_id}.png",
        "status": "generated",
        "seed": 7,
        "width": RESOLUTION[0],
        "height": RESOLUTION[1],
    }
    item.update(overrides)
    return item


def _result_doc(**overrides) -> dict:
    """A producer-faithful ``image-generation-results/1.0`` document."""
    doc = {
        "schema": SCHEMA_RESULT_MANIFEST,
        "run_id": RUN_ID,
        "source_json": IMAGE_JOBS_FILENAME,
        "channel_id": "finance_mascot",
        "title": "T",
        "status": "completed",
        "job_count": 1,
        "generated_count": 1,
        "skipped_count": 0,
        "failed_count": 0,
        "resolution": {"width": RESOLUTION[0], "height": RESOLUTION[1]},
        "steps": handoff.DEFAULT_STEPS,
        "guidance_scale": handoff.DEFAULT_GUIDANCE_SCALE,
        "results": [_result_item()],
    }
    doc.update(overrides)
    return doc


def _verify_result(doc, **kwargs) -> None:
    kwargs.setdefault("expected_run_id", RUN_ID)
    kwargs.setdefault("expected_asset_ids", ["sc01_main"])
    kwargs.setdefault("expected_resolution", RESOLUTION)
    kwargs.setdefault(
        "expected_generation_config", handoff.default_generation_config(*RESOLUTION)
    )
    handoff.verify_result_document(doc, **kwargs)


def test_verify_result_document_accepts_producer_document():
    _verify_result(_result_doc(), require_completed=True)


def test_verify_result_document_rejects_non_object():
    for bad in (None, [], "x", 3):
        with pytest.raises(handoff.HandoffSchemaError) as exc:
            handoff.verify_result_document(bad)
        assert exc.value.code == "hf49"


def test_verify_result_document_rejects_wrong_schema():
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        _verify_result(_result_doc(schema="image-generation-results/2.0"))
    assert exc.value.code == "hf49"


def test_verify_result_document_requires_produced_fields():
    for field in handoff.RESULT_DOCUMENT_REQUIRED_FIELDS:
        doc = _result_doc()
        del doc[field]
        with pytest.raises(handoff.HandoffSchemaError) as exc:
            handoff.verify_result_document(doc)
        assert exc.value.code == "hf49"


def test_verify_result_document_run_id_mismatch():
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        _verify_result(_result_doc(run_id="another-run"))
    assert exc.value.code == "hf50"


def test_verify_result_document_source_json_mismatch():
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        _verify_result(_result_doc(source_json="jobs.json"))
    assert exc.value.code == "hf49"


def test_verify_result_document_invalid_status():
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        _verify_result(_result_doc(status="running"))
    assert exc.value.code == "hf49"


def test_verify_result_document_require_completed_rejects_partial_failure():
    doc = _result_doc(
        status="partial_failure",
        job_count=2,
        generated_count=1,
        failed_count=1,
        results=[
            _result_item("sc01_main"),
            _result_item("sc02_main", status="failed", filename="sc02_main.png"),
        ],
    )
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        _verify_result(doc, require_completed=True)
    assert exc.value.code == "hf50"
    # Without require_completed the same document is structurally consistent.
    _verify_result(doc, expected_asset_ids=["sc01_main", "sc02_main"])


def test_verify_result_document_completed_with_failures_contradicts():
    doc = _result_doc(
        status="completed",
        failed_count=1,
        generated_count=0,
        results=[_result_item(status="failed")],
    )
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_result_document(doc)
    assert exc.value.code == "hf50"


@pytest.mark.parametrize(
    "field,value",
    [
        ("job_count", 2),
        ("generated_count", 2),
        ("failed_count", 1),
        ("skipped_count", 1),
    ],
)
def test_verify_result_document_count_mismatch(field, value):
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_result_document(_result_doc(**{field: value}))
    assert exc.value.code == "hf50"


@pytest.mark.parametrize("field,value", [("job_count", -1), ("generated_count", "1")])
def test_verify_result_document_count_type_and_sign(field, value):
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.verify_result_document(_result_doc(**{field: value}))
    assert exc.value.code == "hf49"


def test_verify_result_document_resolution_mismatch():
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        _verify_result(_result_doc(), expected_resolution=(64, 64))
    assert exc.value.code == "hf50"


def test_verify_result_document_invalid_resolution_axis():
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.verify_result_document(
            _result_doc(resolution={"width": 768, "height": "1344"})
        )
    assert exc.value.code == "hf49"


def test_verify_result_document_generation_config_mismatch():
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        _verify_result(_result_doc(), expected_generation_config={"steps": 8})
    assert exc.value.code == "hf50"


@pytest.mark.parametrize(
    "ids",
    [
        [],  # missing
        ["sc01_main", "sc02_main"],  # unexpected extra
    ],
)
def test_verify_result_document_asset_set_mismatch(ids):
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        _verify_result(_result_doc(), expected_asset_ids=ids or ["sc02_main"])
    assert exc.value.code == "hf50"


def test_verify_result_document_duplicate_asset():
    doc = _result_doc(
        job_count=2,
        generated_count=2,
        results=[_result_item(), _result_item()],
    )
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_result_document(doc)
    assert exc.value.code == "hf50"


@pytest.mark.parametrize("asset_id", ["../evil", "a/b", "", ".hidden"])
def test_verify_result_document_unsafe_asset_id(asset_id):
    doc = _result_doc(results=[_result_item(asset_id)])
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.verify_result_document(doc)
    assert exc.value.code == "hf49"


def test_verify_result_document_filename_mismatch():
    doc = _result_doc(results=[_result_item(filename="other.png")])
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_result_document(doc)
    assert exc.value.code == "hf50"


def test_verify_result_document_item_size_mismatch():
    doc = _result_doc(results=[_result_item(width=64, height=64)])
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_result_document(doc)
    assert exc.value.code == "hf50"


@pytest.mark.parametrize("field", handoff.RESULT_ITEM_REQUIRED_FIELDS)
def test_verify_result_document_item_requires_fields(field):
    item = _result_item()
    del item[field]
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.verify_result_document(_result_doc(results=[item]))
    assert exc.value.code == "hf49"


# --- in-ZIP result document ---------------------------------------------------


def _zip_with_result(tmp_path: Path, result_payload: bytes, *, name="result.json") -> Path:
    return _make_zip(
        tmp_path / "assets.zip",
        {"sc01_main.png": b"png", name: result_payload},
    )


def test_read_result_document_from_zip_valid(tmp_path):
    path = _zip_with_result(tmp_path, json.dumps(_result_doc()).encode("utf-8"))
    assert handoff.read_result_document_from_zip(path) == _result_doc()


def test_read_result_document_from_zip_missing_member(tmp_path):
    path = _make_zip(tmp_path / "assets.zip", {"sc01_main.png": b"png"})
    with pytest.raises(handoff.HandoffError) as exc:
        handoff.read_result_document_from_zip(path)
    assert exc.value.code == "hf49"


def test_read_result_document_from_zip_malformed_json(tmp_path):
    path = _zip_with_result(tmp_path, b"{not json")
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.read_result_document_from_zip(path)
    assert exc.value.code == "hf49"


def test_read_result_document_from_zip_non_object(tmp_path):
    path = _zip_with_result(tmp_path, b"[1, 2]")
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.read_result_document_from_zip(path)
    assert exc.value.code == "hf49"


def test_read_result_document_from_zip_rejects_unsafe_filename(tmp_path):
    path = _zip_with_result(tmp_path, b"{}")
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.read_result_document_from_zip(path, result_filename="../result.json")
    assert exc.value.code == "hf49"


def _manifest_for_zip(zip_path: Path, generated=("sc01_main",), **overrides) -> dict:
    manifest = handoff.build_output_manifest(
        RUN_ID,
        attempt_id=ATTEMPT_ID,
        image_jobs_sha256="b" * 64,
        zip_path=zip_path,
        members=[f"{a}.png" for a in generated] + ["result.json"],
        generated_asset_ids=list(generated),
        resolution=RESOLUTION,
        generation_config=handoff.default_generation_config(*RESOLUTION),
    )
    manifest.update(overrides)
    return manifest


def test_verify_zip_result_document_accepts_published_artifact(tmp_path):
    path = _zip_with_result(tmp_path, json.dumps(_result_doc()).encode("utf-8"))
    handoff.verify_zip_result_document(
        path, output_manifest=_manifest_for_zip(path), run_id=RUN_ID
    )


def test_verify_zip_result_document_uses_manifest_run_id(tmp_path):
    path = _zip_with_result(tmp_path, json.dumps(_result_doc()).encode("utf-8"))
    handoff.verify_zip_result_document(
        path, output_manifest=_manifest_for_zip(path)
    )


def test_verify_zip_result_document_run_mismatch(tmp_path):
    path = _zip_with_result(tmp_path, json.dumps(_result_doc()).encode("utf-8"))
    with pytest.raises(handoff.HandoffIntegrityError) as exc:
        handoff.verify_zip_result_document(
            path, output_manifest=_manifest_for_zip(path), run_id="another-run"
        )
    assert exc.value.code == "hf50"


def test_verify_zip_result_document_rejects_non_output_manifest(tmp_path):
    path = _zip_with_result(tmp_path, json.dumps(_result_doc()).encode("utf-8"))
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.verify_zip_result_document(path, output_manifest={"schema": "nope"})
    assert exc.value.code == "hf17"


def test_verify_zip_result_document_ignores_loose_result_json(tmp_path):
    # A valid loose output/result.json must not substitute for a bad in-ZIP one.
    path = _zip_with_result(tmp_path, b"{not json")
    (path.parent / "result.json").write_text(
        json.dumps(_result_doc()), encoding="utf-8"
    )
    with pytest.raises(handoff.HandoffSchemaError) as exc:
        handoff.verify_zip_result_document(
            path, output_manifest=_manifest_for_zip(path)
        )
    assert exc.value.code == "hf49"


def test_verify_zip_result_document_honours_manifest_result_filename(tmp_path):
    path = _zip_with_result(
        tmp_path, json.dumps(_result_doc()).encode("utf-8"), name="gen_result.json"
    )
    manifest = _manifest_for_zip(
        path,
        members=["sc01_main.png", "gen_result.json"],
        result_manifest={"filename": "gen_result.json", "schema": SCHEMA_RESULT_MANIFEST},
    )
    handoff.verify_zip_result_document(path, output_manifest=manifest)
    # The default member name is no longer the one to read.
    path_default = _zip_with_result(tmp_path, b"{}", name="result.json")
    manifest_default = _manifest_for_zip(path_default)
    with pytest.raises(handoff.HandoffError) as exc:
        handoff.verify_zip_result_document(
            path_default, output_manifest=manifest_default
        )
    assert exc.value.code == "hf49"


# --- GPU preflight -----------------------------------------------------------


def test_gpu_preflight_no_cuda():
    result = handoff.evaluate_gpu_preflight(cuda_available=False)
    assert result["ok"] is False
    assert result["error_code"] == "NO_GPU"


def test_gpu_preflight_zero_devices():
    result = handoff.evaluate_gpu_preflight(cuda_available=True, device_count=0)
    assert result["error_code"] == "NO_GPU"


def test_gpu_preflight_cuda_op_failed():
    result = handoff.evaluate_gpu_preflight(
        cuda_available=True, device_count=1, cuda_op_ok=False
    )
    assert result["error_code"] == "CUDA_INIT_FAILED"


def test_gpu_preflight_total_vram_too_low():
    result = handoff.evaluate_gpu_preflight(
        cuda_available=True,
        device_count=1,
        cuda_op_ok=True,
        total_vram_gb=10.0,
        free_vram_gb=9.0,
    )
    assert result["error_code"] == "INSUFFICIENT_VRAM"


def test_gpu_preflight_free_vram_too_low():
    result = handoff.evaluate_gpu_preflight(
        cuda_available=True,
        device_count=1,
        cuda_op_ok=True,
        total_vram_gb=15.0,
        free_vram_gb=8.0,
    )
    assert result["error_code"] == "INSUFFICIENT_VRAM"


def test_gpu_preflight_total_unknown_is_insufficient():
    result = handoff.evaluate_gpu_preflight(
        cuda_available=True,
        device_count=1,
        cuda_op_ok=True,
        total_vram_gb=15.0,
        free_vram_gb=None,
    )
    assert result["ok"] is False
    assert result["error_code"] == "INSUFFICIENT_VRAM"


def test_gpu_preflight_total_vram_unknown_is_insufficient():
    result = handoff.evaluate_gpu_preflight(
        cuda_available=True,
        device_count=1,
        cuda_op_ok=True,
        total_vram_gb=None,
        free_vram_gb=14.5,
    )
    assert result["error_code"] == "INSUFFICIENT_VRAM"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_gpu_preflight_non_finite_is_insufficient(bad):
    total = handoff.evaluate_gpu_preflight(
        cuda_available=True,
        device_count=1,
        cuda_op_ok=True,
        total_vram_gb=bad,
        free_vram_gb=14.5,
    )
    free = handoff.evaluate_gpu_preflight(
        cuda_available=True,
        device_count=1,
        cuda_op_ok=True,
        total_vram_gb=15.0,
        free_vram_gb=bad,
    )
    assert total["error_code"] == "INSUFFICIENT_VRAM"
    assert free["error_code"] == "INSUFFICIENT_VRAM"


def test_gpu_preflight_ok():
    result = handoff.evaluate_gpu_preflight(
        cuda_available=True,
        device_count=1,
        cuda_op_ok=True,
        gpu_name="Tesla T4",
        total_vram_gb=15.0,
        free_vram_gb=14.5,
    )
    assert result["ok"] is True
    diagnostics = result["diagnostics"]
    assert diagnostics["gpu_name"] == "Tesla T4"
    assert diagnostics["required_total_vram_gib"] == 14.0
    assert diagnostics["required_free_vram_gib"] == 10.0


def test_gpu_preflight_custom_thresholds():
    result = handoff.evaluate_gpu_preflight(
        cuda_available=True,
        device_count=1,
        cuda_op_ok=True,
        total_vram_gb=8.0,
        free_vram_gb=7.0,
        total_min_gib=6.0,
        free_min_gib=5.0,
    )
    assert result["ok"] is True


# --- purity ------------------------------------------------------------------


def test_handoff_module_imports_no_third_party():
    source = Path(handoff.__file__).read_text(encoding="utf-8")
    forbidden = r"(?m)^\s*(?:import|from)\s+(numpy|torch|PIL|requests|google|httpx|boto3|pydantic)\b"
    assert re.search(forbidden, source) is None


# --- drift guards against the importer ---------------------------------------


def test_safe_asset_id_re_matches_importer():
    from agents.image_jobs import import_assets

    assert handoff.SAFE_ASSET_ID_RE.pattern == import_assets.SAFE_ASSET_ID_RE.pattern


@pytest.mark.parametrize(
    "name",
    ["a.png", "./a.png", "dir/a.png", ".\\dir\\a.png", "a.png"],
)
def test_normalize_member_matches_importer(name):
    from agents.image_jobs import import_assets

    assert handoff._normalize_member(name) == import_assets._norm_member(name)
