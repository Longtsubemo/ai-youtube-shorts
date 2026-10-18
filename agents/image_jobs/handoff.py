"""Handoff contracts and integrity helpers for the Colab SDXL workflow.

This module is a **pure standard-library** contract layer for the manual Google
Drive handoff between the local pipeline and a free-GPU Colab SDXL worker. It
defines the versioned JSON documents exchanged through the shared Drive folder,
plus the SHA-256, marker, manifest and ZIP integrity helpers used to validate
them.

It deliberately contains no third-party SDK, no network access, no model and no
GPU code. Every path is a plain filesystem path, so the same helpers work
against a real mounted/synced Drive folder or a local directory that simulates
Drive in tests.

What lives here:

- schema constants and JSON builders for the input manifest, output manifest and
  status/error document;
- SHA-256 helpers for bytes and files;
- a content-bearing, marker-last protocol (``READY`` / ``DONE``) bound to an
  attempt id, so a stale marker from an earlier attempt can never be reused;
- input<->output cross-binding (run id, ``image_jobs`` hash, resolution,
  generation config, asset completeness);
- ZIP integrity checks (existence, hash/size, readability, duplicate normalized
  members, unsafe members, confined filenames);
- in-ZIP ``result.json`` (``image-generation-results/1.0``) validation against
  the worker's publisher contract;
- a pure GPU-preflight evaluator mapping observed GPU facts to a decision, with
  fail-closed handling of unknown or non-finite VRAM readings.

What does **not** live here (later milestones): the local CLI, rclone/Drive
integration, the automation notebook, asset import and Stage 5. Those consume
this module; they do not change it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from agents.image_jobs.jobs import JOB_SCHEMA_VERSION

# --- schema versions ---------------------------------------------------------

# Milestone 1.1 binds input<->output<->status and carries a content-bearing
# attempt id. Version 1.0 documents are rejected outright (the readers require
# exact equality), which is the intended migration boundary: a 1.0 handoff must
# be re-built with the new fields rather than silently accepted.
SCHEMA_INPUT_MANIFEST = "sdxl-input-manifest/1.1"
SCHEMA_OUTPUT_MANIFEST = "sdxl-output-manifest/1.1"
SCHEMA_STATUS = "sdxl-status/1.1"
SCHEMA_RESULT_MANIFEST = "image-generation-results/1.0"

# --- canonical handoff file / folder names -----------------------------------

HANDOFF_ROOT_NAME = "ai-youtube-channel-handoff"
INPUT_DIR_NAME = "input"
STATUS_DIR_NAME = "status"
OUTPUT_DIR_NAME = "output"

INPUT_MANIFEST_FILENAME = "input_manifest.json"
OUTPUT_MANIFEST_FILENAME = "output_manifest.json"
STATUS_FILENAME = "status.json"
IMAGE_JOBS_FILENAME = "image_jobs.json"
RESULT_FILENAME = "result.json"
READY_MARKER = "READY"
DONE_MARKER = "DONE"

TOOL_NAME = "colab_handoff"
TOOL_VERSION = "0.1.1"

# --- approved generation defaults (decision D1) ------------------------------

DEFAULT_CHECKPOINT = "sdxl_lightning_4step_unet.safetensors"
DEFAULT_STEPS = 4
DEFAULT_GUIDANCE_SCALE = 0.0
DEFAULT_DTYPE = "float16"
DEFAULT_EXPECTED_WIDTH = 768
DEFAULT_EXPECTED_HEIGHT = 1344
DEFAULT_EXPECTED_RESOLUTION = (DEFAULT_EXPECTED_WIDTH, DEFAULT_EXPECTED_HEIGHT)

GENERATION_CONFIG_REQUIRED_FIELDS = (
    "checkpoint",
    "steps",
    "guidance_scale",
    "dtype",
    "width",
    "height",
)

# --- approved GPU preflight threshold (decision D4) --------------------------

VRAM_TOTAL_MIN_GIB = 14.0
VRAM_FREE_MIN_GIB = 10.0

# --- status vocabulary -------------------------------------------------------

STAGES = (
    "gpu_preflight",
    "env_setup",
    "model_load",
    "generation",
    "output_validation",
    "packaging",
    "upload",
)

OVERALL_STATUSES = ("PENDING", "RUNNING", "PASS", "BLOCKED", "FAILED")

# Allowed values of the output manifest's own ``status`` field. Only a fully
# completed output may be published (the marker-last protocol means ``DONE``
# implies this value).
OUTPUT_STATUSES = ("completed",)

ERROR_CODES = (
    "NO_GPU",
    "CUDA_INIT_FAILED",
    "INSUFFICIENT_VRAM",
    "RUNTIME_DISCONNECTED",
    "ENV_SETUP_FAILED",
    "INPUT_VALIDATION_FAILED",
    "MODEL_LOAD_FAILED",
    "GENERATION_FAILED",
    "OUTPUT_VALIDATION_FAILED",
)

# Recoverable infrastructure failures -> overall status ``BLOCKED`` (free GPU
# unavailable or runtime faulted; the human retries later, never automatically).
RECOVERABLE_ERROR_CODES = frozenset(
    {
        "NO_GPU",
        "CUDA_INIT_FAILED",
        "INSUFFICIENT_VRAM",
        "RUNTIME_DISCONNECTED",
        "ENV_SETUP_FAILED",
        "MODEL_LOAD_FAILED",
        "GENERATION_FAILED",
    }
)

# Deterministic defects -> overall status ``FAILED`` (a retry that is not preceded
# by a fix cannot help): output validation, and deterministic pre-generation
# input defects (e.g. a prompt that exceeds the token budget).
UNRECOVERABLE_ERROR_CODES = frozenset(
    {
        "INPUT_VALIDATION_FAILED",
        "OUTPUT_VALIDATION_FAILED",
    }
)

# Where each error code is expected to occur. ``RUNTIME_DISCONNECTED`` may occur
# at any stage, so it is absent (unconstrained).
ERROR_STAGE_HINTS = {
    "NO_GPU": "gpu_preflight",
    "CUDA_INIT_FAILED": "gpu_preflight",
    "INSUFFICIENT_VRAM": "gpu_preflight",
    "ENV_SETUP_FAILED": "env_setup",
    "INPUT_VALIDATION_FAILED": "env_setup",
    "MODEL_LOAD_FAILED": "model_load",
    "GENERATION_FAILED": "generation",
    "OUTPUT_VALIDATION_FAILED": "output_validation",
}

IMAGE_JOB_REQUIRED_FIELDS = (
    "scene_id",
    "positive_prompt",
    "negative_prompt",
    "seed",
    "asset_id",
    "output_path",
)

# A bare, safe scene asset id such as ``sc01_main``. Mirrors the importer so a
# handoff manifest can never describe an asset the importer would reject.
SAFE_ASSET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Errors carried in the status document (informational code for HandoffError).
HANDOFF_ERROR_BASE_CODE = "hf00"


class HandoffError(ValueError):
    """Base error for structural handoff-contract problems."""

    def __init__(self, message: str, code: str = HANDOFF_ERROR_BASE_CODE) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class HandoffSchemaError(HandoffError):
    """A document is missing, malformed, or declares an unsupported schema."""


class HandoffIntegrityError(HandoffError):
    """A document exists and parses but contradicts observed facts."""


# --- small internal guards ---------------------------------------------------


def _reject_legacy_hash(document: dict, code: str) -> None:
    """Refuse the retired, ambiguous ``input_sha256`` field."""
    if not isinstance(document, dict):
        return
    if "input_sha256" in document:
        raise HandoffSchemaError(
            "legacy 'input_sha256' is no longer accepted; use "
            "'image_jobs_sha256' (schema 1.1)",
            code,
        )


def _safe_bare_name(name: object) -> bool:
    """Return whether ``name`` is a single, non-empty, separator-free filename."""
    if not isinstance(name, str) or not name:
        return False
    if name in (".", "..") or "\\" in name or "/" in name:
        return False
    path = Path(name)
    if path.name != name or path.is_absolute() or path.drive:
        return False
    return True


def _finite(value: object) -> bool:
    """Return whether ``value`` is a real (non-bool) finite number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


# --- time / hashing ----------------------------------------------------------


def utc_now_iso() -> str:
    """Return the current UTC time as a timezone-aware ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def new_attempt_id(run_id: str) -> str:
    """Return a unique attempt id for ``run_id`` (binds one handoff attempt)."""
    run_id = str(run_id)
    if not run_id:
        raise HandoffSchemaError("cannot build an attempt id without a run_id", "hf24")
    return f"{run_id}-{uuid.uuid4().hex}"


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the lowercase hex SHA-256 of a file, streamed in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_size(path: str | Path) -> int:
    """Return the size in bytes of a file."""
    return Path(path).stat().st_size


def file_fingerprint(path: str | Path) -> dict:
    """Return ``{"filename", "sha256", "bytes"}`` for a file."""
    path = Path(path)
    return {
        "filename": path.name,
        "sha256": sha256_file(path),
        "bytes": file_size(path),
    }


# --- JSON I/O ----------------------------------------------------------------


def load_json_object(path: str | Path, *, code: str = "hf01") -> dict:
    """Read a JSON object from ``path`` or raise a :class:`HandoffError`."""
    path = Path(path)
    if not path.is_file():
        raise HandoffError(f"file not found: {path}", code)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HandoffError(f"cannot read {path}: {exc}", code) from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HandoffSchemaError(f"{path} is not valid JSON: {exc}", "hf02") from exc
    if not isinstance(data, dict):
        raise HandoffSchemaError(f"{path} must contain a JSON object", "hf02")
    return data


def dump_json(document: dict, path: str | Path) -> Path:
    """Write ``document`` as pretty JSON to ``path`` (parents created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


# --- generation config -------------------------------------------------------


def default_generation_config(
    width: int = DEFAULT_EXPECTED_WIDTH,
    height: int = DEFAULT_EXPECTED_HEIGHT,
) -> dict:
    """Return the approved (D1) SDXL-Lightning generation configuration."""
    return {
        "checkpoint": DEFAULT_CHECKPOINT,
        "steps": DEFAULT_STEPS,
        "guidance_scale": DEFAULT_GUIDANCE_SCALE,
        "dtype": DEFAULT_DTYPE,
        "width": int(width),
        "height": int(height),
    }


def generation_config_problems(
    config: object,
    *,
    resolution: tuple[int, int] | None = None,
    require_approved: bool = True,
) -> list[str]:
    """Return a list of problems with a generation-config object (empty = OK)."""
    problems: list[str] = []
    if not isinstance(config, dict):
        return ["generation_config must be an object"]

    missing = [f for f in GENERATION_CONFIG_REQUIRED_FIELDS if f not in config]
    if missing:
        problems.append(f"generation_config is missing: {', '.join(missing)}")
        return problems

    if not isinstance(config["checkpoint"], str) or not config["checkpoint"]:
        problems.append("generation_config.checkpoint must be a non-empty string")
    if config["steps"] != DEFAULT_STEPS:
        problems.append(
            f"generation_config.steps is {config['steps']!r}, expected {DEFAULT_STEPS}"
        )
    if config["guidance_scale"] != DEFAULT_GUIDANCE_SCALE:
        problems.append(
            "generation_config.guidance_scale is "
            f"{config['guidance_scale']!r}, expected {DEFAULT_GUIDANCE_SCALE}"
        )
    if config["dtype"] != DEFAULT_DTYPE:
        problems.append(
            f"generation_config.dtype is {config['dtype']!r}, expected {DEFAULT_DTYPE}"
        )
    for axis in ("width", "height"):
        if not isinstance(config[axis], int) or isinstance(config[axis], bool):
            problems.append(f"generation_config.{axis} must be an integer")

    if require_approved and config["checkpoint"] != DEFAULT_CHECKPOINT:
        problems.append(
            f"generation_config.checkpoint is {config['checkpoint']!r}, expected "
            f"{DEFAULT_CHECKPOINT}"
        )

    if resolution is not None and not problems:
        actual = (config["width"], config["height"])
        expected = (int(resolution[0]), int(resolution[1]))
        if actual != expected:
            problems.append(
                f"generation_config resolution {actual} != expected {expected}"
            )
    return problems


# --- image-jobs contract -----------------------------------------------------


def load_image_jobs(path: str | Path) -> dict:
    """Load and structurally validate an ``image_jobs.json`` batch.

    Validates the schema string, a non-empty ``run_id``, non-empty ``jobs``,
    required per-job fields, safe asset ids, and ``output_path`` values confined
    to ``data/<run_id>/assets/<asset_id>.png`` (the same shape the importer
    enforces). Raises :class:`HandoffSchemaError` for structural problems and
    :class:`HandoffIntegrityError` for unsafe ids/paths.
    """
    data = load_json_object(path, code="hf01")
    schema = data.get("schema")
    if schema != JOB_SCHEMA_VERSION:
        raise HandoffSchemaError(
            f"unsupported image-jobs schema {schema!r}; expected {JOB_SCHEMA_VERSION}",
            "hf03",
        )

    batch_run_id = str(data.get("run_id", "")).strip()
    if not batch_run_id:
        raise HandoffSchemaError(
            "image-jobs batch is missing run_id (needed to check output_path)",
            "hf03",
        )
    rel_assets = Path("data") / batch_run_id / "assets"

    jobs = data.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise HandoffSchemaError("image-jobs batch contains no jobs", "hf03")

    seen: set[str] = set()
    for index, job in enumerate(jobs, start=1):
        if not isinstance(job, dict):
            raise HandoffSchemaError(f"job {index} is not an object", "hf03")
        missing = [f for f in IMAGE_JOB_REQUIRED_FIELDS if f not in job]
        if missing:
            raise HandoffSchemaError(
                f"job {index} is missing {', '.join(missing)}", "hf03"
            )

        asset_id = str(job["asset_id"]).strip()
        if not SAFE_ASSET_ID_RE.match(asset_id):
            raise HandoffIntegrityError(
                f"job {index}: unsafe asset_id {asset_id!r}", "hf05"
            )
        if asset_id in seen:
            raise HandoffIntegrityError(
                f"job {index}: duplicate asset_id {asset_id!r}", "hf05"
            )
        seen.add(asset_id)

        output_path = str(job["output_path"]).replace("\\", "/")
        op = Path(output_path)
        if op.is_absolute() or op.drive or ".." in op.parts:
            raise HandoffIntegrityError(
                f"job {index}: unsafe output_path {job['output_path']!r}", "hf06"
            )
        if op.parent != rel_assets:
            raise HandoffIntegrityError(
                f"job {index}: output_path {job['output_path']!r} must be directly "
                f"under {rel_assets.as_posix()}/",
                "hf07",
            )
        if op.suffix != ".png" or op.stem != asset_id:
            raise HandoffIntegrityError(
                f"job {index}: output_path {job['output_path']!r} must end in "
                f"'.png' and match asset_id {asset_id!r}",
                "hf06",
            )
    return data


def expected_asset_ids(data: dict) -> list[str]:
    """Return the ordered ``asset_id`` list from a validated jobs dict."""
    return [str(job["asset_id"]) for job in data["jobs"]]


# --- input manifest ----------------------------------------------------------


def build_input_manifest(
    run_id: str,
    image_jobs_path: str | Path,
    *,
    attempt_id: str,
    created_at: str | None = None,
    expected_resolution: tuple[int, int] = DEFAULT_EXPECTED_RESOLUTION,
    generation_config: dict | None = None,
    tool_version: str = TOOL_VERSION,
) -> dict:
    """Build the ``sdxl-input-manifest/1.1`` document for one run attempt.

    The manifest fingerprints the exact ``image_jobs.json`` handed to Colab and
    records the attempt id that binds it to the matching output/status.
    """
    if not attempt_id:
        raise HandoffSchemaError("input manifest requires an attempt_id", "hf10")
    data = load_image_jobs(image_jobs_path)
    contract_run = str(data.get("run_id", ""))
    if contract_run and contract_run != run_id:
        raise HandoffIntegrityError(
            f"image-jobs run_id {contract_run!r} does not match requested run {run_id!r}",
            "hf04",
        )

    path = Path(image_jobs_path)
    ids = expected_asset_ids(data)
    return {
        "schema": SCHEMA_INPUT_MANIFEST,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "created_at": created_at or utc_now_iso(),
        "tool": {"name": TOOL_NAME, "version": tool_version},
        "image_jobs": {
            "filename": path.name,
            "schema": JOB_SCHEMA_VERSION,
            "bytes": file_size(path),
        },
        "image_jobs_sha256": sha256_file(path),
        "job_count": len(ids),
        "expected_asset_ids": ids,
        "expected_resolution": {
            "width": int(expected_resolution[0]),
            "height": int(expected_resolution[1]),
        },
        "generation_config": generation_config
        or default_generation_config(*expected_resolution),
    }


def verify_input_manifest(
    manifest: dict,
    *,
    image_jobs_path: str | Path | None = None,
    expected_run_id: str | None = None,
    expected_attempt_id: str | None = None,
    expected_resolution: tuple[int, int] = DEFAULT_EXPECTED_RESOLUTION,
) -> None:
    """Validate an input manifest, raising on any structural/integrity problem.

    When ``image_jobs_path`` is given the manifest's recorded filename, hash and
    expected ids are re-checked against the actual batch (stale-manifest guard).
    """
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA_INPUT_MANIFEST:
        raise HandoffSchemaError(
            f"not a {SCHEMA_INPUT_MANIFEST} document", "hf10"
        )
    _reject_legacy_hash(manifest, "hf16")

    run_id = manifest.get("run_id")
    if not run_id:
        raise HandoffSchemaError("input manifest is missing run_id", "hf10")
    if expected_run_id is not None and run_id != expected_run_id:
        raise HandoffIntegrityError(
            f"input manifest run_id {run_id!r} != expected {expected_run_id!r}", "hf11"
        )

    attempt_id = manifest.get("attempt_id")
    if not attempt_id:
        raise HandoffSchemaError("input manifest is missing attempt_id", "hf10")
    if expected_attempt_id is not None and attempt_id != expected_attempt_id:
        raise HandoffIntegrityError(
            f"input manifest attempt_id {attempt_id!r} != expected "
            f"{expected_attempt_id!r}",
            "hf25",
        )

    ids = manifest.get("expected_asset_ids")
    if not isinstance(ids, list) or not ids:
        raise HandoffSchemaError("input manifest has no expected_asset_ids", "hf10")
    if len(set(ids)) != len(ids):
        raise HandoffSchemaError(
            "input manifest expected_asset_ids contains duplicates", "hf10"
        )

    resolution = manifest.get("expected_resolution") or {}
    actual_resolution = (resolution.get("width"), resolution.get("height"))
    expected_pair = (int(expected_resolution[0]), int(expected_resolution[1]))
    if actual_resolution != expected_pair:
        raise HandoffIntegrityError(
            f"input manifest resolution {actual_resolution} != expected {expected_pair}",
            "hf12",
        )

    problems = generation_config_problems(manifest.get("generation_config"))
    if problems:
        raise HandoffSchemaError(
            "input manifest generation_config invalid: " + "; ".join(problems), "hf10"
        )

    jobs_meta = manifest.get("image_jobs") or {}
    if not isinstance(jobs_meta, dict):
        raise HandoffSchemaError("input manifest image_jobs must be an object", "hf10")
    if jobs_meta.get("schema") not in (None, JOB_SCHEMA_VERSION):
        raise HandoffSchemaError(
            f"input manifest image_jobs.schema {jobs_meta.get('schema')!r} is invalid",
            "hf10",
        )

    if image_jobs_path is not None:
        jobs_path = Path(image_jobs_path)
        if jobs_meta.get("filename") not in (None, jobs_path.name):
            raise HandoffIntegrityError(
                f"input manifest image_jobs.filename {jobs_meta.get('filename')!r} "
                f"!= {jobs_path.name!r}",
                "hf15",
            )
        expected_sha = manifest.get("image_jobs_sha256")
        if not expected_sha:
            raise HandoffSchemaError(
                "input manifest is missing image_jobs_sha256", "hf10"
            )
        if expected_sha != sha256_file(jobs_path):
            raise HandoffIntegrityError(
                "input manifest image_jobs_sha256 does not match the batch "
                "(stale or swapped manifest)",
                "hf13",
            )
        if jobs_meta.get("bytes") is not None and int(jobs_meta["bytes"]) != file_size(
            jobs_path
        ):
            raise HandoffIntegrityError(
                "input manifest image_jobs.bytes does not match the batch", "hf13"
            )
        data = load_image_jobs(jobs_path)
        actual_ids = expected_asset_ids(data)
        if sorted(ids) != sorted(actual_ids):
            raise HandoffIntegrityError(
                "input manifest expected_asset_ids do not match the batch", "hf14"
            )
        if manifest.get("job_count") != len(actual_ids):
            raise HandoffIntegrityError(
                "input manifest job_count does not match the batch", "hf14"
            )
        if data.get("run_id") and data["run_id"] != run_id:
            raise HandoffIntegrityError(
                f"batch run_id {data['run_id']!r} != manifest run_id {run_id!r}",
                "hf11",
            )


# --- output manifest ---------------------------------------------------------


def build_output_manifest(
    run_id: str,
    *,
    attempt_id: str,
    image_jobs_sha256: str,
    zip_path: str | Path,
    members: list[str],
    generated_asset_ids: list[str],
    resolution: tuple[int, int] = DEFAULT_EXPECTED_RESOLUTION,
    generation_config: dict | None = None,
    result_filename: str = RESULT_FILENAME,
    status: str = "completed",
    created_at: str | None = None,
) -> dict:
    """Build the ``sdxl-output-manifest/1.1`` document for one run's ZIP."""
    if not attempt_id:
        raise HandoffSchemaError("output manifest requires an attempt_id", "hf24")
    if status not in OUTPUT_STATUSES:
        raise HandoffSchemaError(
            f"invalid output status {status!r}; expected one of {OUTPUT_STATUSES}",
            "hf45",
        )
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise HandoffError(f"output ZIP not found: {zip_path}", "hf30")

    return {
        "schema": SCHEMA_OUTPUT_MANIFEST,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "created_at": created_at or utc_now_iso(),
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION},
        "image_jobs_sha256": image_jobs_sha256,
        "status": status,
        "zip": {
            "filename": zip_path.name,
            "sha256": sha256_file(zip_path),
            "bytes": file_size(zip_path),
        },
        "members": list(members),
        "generated_asset_ids": list(generated_asset_ids),
        "resolution": {
            "width": int(resolution[0]),
            "height": int(resolution[1]),
        },
        "generation_config": generation_config
        or default_generation_config(*resolution),
        "result_manifest": {
            "filename": result_filename,
            "schema": SCHEMA_RESULT_MANIFEST,
        },
    }


def _expected_output_members(
    generated_asset_ids: list[str], result_filename: str
) -> set[str]:
    return {f"{asset_id}.png" for asset_id in generated_asset_ids} | {result_filename}


def verify_output_manifest(
    manifest: dict,
    *,
    expected_run_id: str | None = None,
    expected_attempt_id: str | None = None,
    expected_image_jobs_sha256: str | None = None,
    zip_path: str | Path | None = None,
) -> list[str]:
    """Validate an output manifest and, if given, the referenced ZIP.

    Returns the ZIP member list (from the archive when ``zip_path`` is given,
    otherwise from the manifest).
    """
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != SCHEMA_OUTPUT_MANIFEST
    ):
        raise HandoffSchemaError(f"not a {SCHEMA_OUTPUT_MANIFEST} document", "hf17")
    _reject_legacy_hash(manifest, "hf16")

    run_id = manifest.get("run_id")
    if not run_id:
        raise HandoffSchemaError("output manifest is missing run_id", "hf17")
    if expected_run_id is not None and run_id != expected_run_id:
        raise HandoffIntegrityError(
            f"output manifest run_id {run_id!r} != expected {expected_run_id!r}",
            "hf18",
        )

    attempt_id = manifest.get("attempt_id")
    if not attempt_id:
        raise HandoffSchemaError("output manifest is missing attempt_id", "hf24")
    if expected_attempt_id is not None and attempt_id != expected_attempt_id:
        raise HandoffIntegrityError(
            f"output manifest attempt_id {attempt_id!r} != expected "
            f"{expected_attempt_id!r}",
            "hf28",
        )

    image_jobs_sha256 = manifest.get("image_jobs_sha256")
    if not image_jobs_sha256:
        raise HandoffSchemaError(
            "output manifest is missing image_jobs_sha256", "hf17"
        )
    if (
        expected_image_jobs_sha256 is not None
        and image_jobs_sha256 != expected_image_jobs_sha256
    ):
        raise HandoffIntegrityError(
            "output manifest image_jobs_sha256 does not match the input manifest",
            "hf19",
        )

    status = manifest.get("status")
    if status not in OUTPUT_STATUSES:
        raise HandoffSchemaError(
            f"invalid output status {status!r}; expected one of {OUTPUT_STATUSES}",
            "hf45",
        )

    members = manifest.get("members")
    if not isinstance(members, list) or not members:
        raise HandoffSchemaError("output manifest has no members", "hf17")

    result_manifest = manifest.get("result_manifest")
    if not isinstance(result_manifest, dict) or result_manifest.get(
        "schema"
    ) != SCHEMA_RESULT_MANIFEST:
        raise HandoffSchemaError(
            f"output manifest result_manifest.schema must be {SCHEMA_RESULT_MANIFEST}",
            "hf47",
        )
    result_filename = result_manifest.get("filename")
    if not _safe_bare_name(result_filename):
        raise HandoffSchemaError(
            f"output manifest result_manifest.filename {result_filename!r} is unsafe",
            "hf47",
        )

    zip_meta = manifest.get("zip") or {}
    if not _safe_bare_name(zip_meta.get("filename")):
        raise HandoffSchemaError(
            f"output manifest zip.filename {zip_meta.get('filename')!r} is unsafe",
            "hf27",
        )

    generated = manifest.get("generated_asset_ids")
    if not isinstance(generated, list) or not generated:
        raise HandoffSchemaError(
            "output manifest has no generated_asset_ids", "hf17"
        )
    if len(set(generated)) != len(generated):
        raise HandoffIntegrityError(
            "output manifest generated_asset_ids contains duplicates", "hf43"
        )

    normalized_members = [_normalize_member(m) for m in members]
    if len(set(normalized_members)) != len(normalized_members):
        raise HandoffIntegrityError(
            "output manifest members contains duplicate normalized names", "hf36"
        )
    expected_members = _expected_output_members(generated, result_filename)
    if set(normalized_members) != expected_members:
        raise HandoffIntegrityError(
            "output manifest members do not match generated_asset_ids + "
            f"result_manifest (members={sorted(set(normalized_members))}, "
            f"expected={sorted(expected_members)})",
            "hf35",
        )

    if zip_path is not None:
        return verify_zip(
            zip_path,
            expected_sha256=zip_meta.get("sha256"),
            expected_bytes=zip_meta.get("bytes"),
            expected_members=normalized_members,
        )
    return list(members)


def verify_handoff_pair(
    input_manifest: dict,
    output_manifest: dict,
    *,
    image_jobs_path: str | Path | None = None,
    expected_run_id: str | None = None,
    expected_resolution: tuple[int, int] = DEFAULT_EXPECTED_RESOLUTION,
) -> None:
    """Cross-bind an input manifest to its output manifest.

    Enforces that the output belongs to the approved input: same run id, same
    ``image_jobs`` hash, same attempt id, same resolution, the approved
    generation config, and an asset set identical to the approved input.
    """
    if (
        not isinstance(input_manifest, dict)
        or input_manifest.get("schema") != SCHEMA_INPUT_MANIFEST
    ):
        raise HandoffSchemaError(
            f"pair: not a {SCHEMA_INPUT_MANIFEST} document", "hf44"
        )
    if (
        not isinstance(output_manifest, dict)
        or output_manifest.get("schema") != SCHEMA_OUTPUT_MANIFEST
    ):
        raise HandoffSchemaError(
            f"pair: not a {SCHEMA_OUTPUT_MANIFEST} document", "hf44"
        )
    _reject_legacy_hash(input_manifest, "hf16")
    _reject_legacy_hash(output_manifest, "hf16")

    verify_input_manifest(
        input_manifest,
        image_jobs_path=image_jobs_path,
        expected_run_id=expected_run_id,
        expected_resolution=expected_resolution,
    )
    verify_output_manifest(
        output_manifest,
        expected_run_id=input_manifest.get("run_id"),
        expected_attempt_id=input_manifest.get("attempt_id"),
        expected_image_jobs_sha256=input_manifest.get("image_jobs_sha256"),
    )

    if output_manifest.get("run_id") != input_manifest.get("run_id"):
        raise HandoffIntegrityError(
            "pair run_id mismatch: "
            f"{output_manifest.get('run_id')!r} != {input_manifest.get('run_id')!r}",
            "hf44",
        )
    if output_manifest.get("image_jobs_sha256") != input_manifest.get(
        "image_jobs_sha256"
    ):
        raise HandoffIntegrityError(
            "pair image_jobs_sha256 mismatch", "hf19"
        )
    if output_manifest.get("attempt_id") != input_manifest.get("attempt_id"):
        raise HandoffIntegrityError("pair attempt_id mismatch", "hf28")

    in_res = input_manifest.get("expected_resolution") or {}
    out_res = output_manifest.get("resolution") or {}
    if (in_res.get("width"), in_res.get("height")) != (
        out_res.get("width"),
        out_res.get("height"),
    ):
        raise HandoffIntegrityError(
            "pair resolution mismatch: "
            f"{(in_res.get('width'), in_res.get('height'))} != "
            f"{(out_res.get('width'), out_res.get('height'))}",
            "hf29",
        )

    problems = generation_config_problems(
        output_manifest.get("generation_config"),
        resolution=expected_resolution,
    )
    if problems:
        raise HandoffIntegrityError(
            "pair generation_config invalid: " + "; ".join(problems), "hf42"
        )
    if output_manifest.get("generation_config") != input_manifest.get(
        "generation_config"
    ):
        raise HandoffIntegrityError(
            "pair generation_config does not match the input manifest", "hf42"
        )

    expected_ids = list(input_manifest.get("expected_asset_ids") or [])
    generated = list(output_manifest.get("generated_asset_ids") or [])
    if len(set(generated)) != len(generated):
        raise HandoffIntegrityError(
            "pair generated_asset_ids contains duplicates", "hf43"
        )
    missing = sorted(set(expected_ids) - set(generated))
    unexpected = sorted(set(generated) - set(expected_ids))
    if missing or unexpected:
        raise HandoffIntegrityError(
            "pair asset set mismatch "
            f"(missing={missing}, unexpected={unexpected})",
            "hf43",
        )

    if output_manifest.get("status") != "completed":
        raise HandoffIntegrityError(
            f"pair output status {output_manifest.get('status')!r} is not "
            "'completed'",
            "hf45",
        )


# --- status / error document -------------------------------------------------


def make_error(
    code: str,
    stage: str,
    message: str,
    *,
    diagnostics: dict | None = None,
    traceback_text: str = "",
    timestamp: str | None = None,
) -> dict:
    """Build a structured error entry for the status document."""
    if code not in ERROR_CODES:
        raise HandoffSchemaError(f"unknown error code {code!r}", "hf40")
    if stage not in STAGES:
        raise HandoffSchemaError(f"unknown stage {stage!r}", "hf40")
    hint = ERROR_STAGE_HINTS.get(code)
    if hint is not None and stage != hint:
        raise HandoffSchemaError(
            f"error code {code!r} belongs to stage {hint!r}, not {stage!r}", "hf40"
        )
    if not message:
        raise HandoffSchemaError("error message must not be empty", "hf40")
    return {
        "code": code,
        "stage": stage,
        "message": message,
        "recoverable": code in RECOVERABLE_ERROR_CODES,
        "timestamp": timestamp or utc_now_iso(),
        "diagnostics": dict(diagnostics or {}),
        "traceback": traceback_text,
    }


def overall_status_for_error(code: str) -> str:
    """Map an error code to ``BLOCKED`` (recoverable) or ``FAILED`` (defect)."""
    if code in RECOVERABLE_ERROR_CODES:
        return "BLOCKED"
    if code in UNRECOVERABLE_ERROR_CODES:
        return "FAILED"
    raise HandoffSchemaError(f"unknown error code {code!r}", "hf40")


def build_status(
    run_id: str,
    *,
    attempt_id: str,
    image_jobs_sha256: str,
    generation_config: dict | None = None,
    overall_status: str = "PENDING",
    current_stage: str = STAGES[0],
    completed_stages: list[str] | None = None,
    completed_asset_ids: list[str] | None = None,
    failed_assets: list[dict] | None = None,
    error: dict | None = None,
    environment: dict | None = None,
    resume_instructions: str = "",
    created_at: str | None = None,
    updated_at: str | None = None,
) -> dict:
    """Build the ``sdxl-status/1.1`` document for one run attempt."""
    now = utc_now_iso()
    return {
        "schema": SCHEMA_STATUS,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "created_at": created_at or now,
        "updated_at": updated_at or now,
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION},
        "image_jobs_sha256": image_jobs_sha256,
        "overall_status": overall_status,
        "current_stage": current_stage,
        "completed_stages": list(completed_stages or []),
        "completed_asset_ids": list(completed_asset_ids or []),
        "failed_assets": list(failed_assets or []),
        "error": error,
        "environment": dict(environment or {}),
        "generation_config": generation_config or default_generation_config(),
        "resume_instructions": resume_instructions,
    }


def _validate_error_shape(error: object) -> str:
    """Validate an error object and return its code."""
    if not isinstance(error, dict):
        raise HandoffSchemaError("status error must be an object", "hf41")
    code = error.get("code")
    if code not in ERROR_CODES:
        raise HandoffSchemaError("status error has an unknown code", "hf41")
    stage = error.get("stage")
    if stage not in STAGES:
        raise HandoffSchemaError("status error has an unknown stage", "hf41")
    hint = ERROR_STAGE_HINTS.get(code)
    if hint is not None and stage != hint:
        raise HandoffSchemaError(
            f"status error code {code!r} belongs to stage {hint!r}, not {stage!r}",
            "hf41",
        )
    if not error.get("message"):
        raise HandoffSchemaError("status error has no message", "hf41")
    if bool(error.get("recoverable")) != (code in RECOVERABLE_ERROR_CODES):
        raise HandoffSchemaError(
            "status error recoverable flag does not match its code", "hf41"
        )
    return str(code)


def validate_status(
    status: dict,
    *,
    expected_asset_ids: list[str] | None = None,
    required_stages: tuple[str, ...] | list[str] = STAGES,
) -> None:
    """Validate a status document's structure and internal consistency.

    Enforces the resume invariants: ``completed_stages`` is a prefix of the
    stage order; ``PENDING`` has completed nothing; ``RUNNING`` is at the next
    stage; ``PASS`` has completed every required stage with exactly the expected
    assets; ``BLOCKED``/``FAILED`` carry an error at the current stage.
    """
    if not isinstance(status, dict) or status.get("schema") != SCHEMA_STATUS:
        raise HandoffSchemaError(f"not a {SCHEMA_STATUS} document", "hf41")
    _reject_legacy_hash(status, "hf41")

    if not status.get("run_id"):
        raise HandoffSchemaError("status is missing run_id", "hf41")
    if not status.get("attempt_id"):
        raise HandoffSchemaError("status is missing attempt_id", "hf41")
    if not status.get("image_jobs_sha256"):
        raise HandoffSchemaError("status is missing image_jobs_sha256", "hf41")

    overall = status.get("overall_status")
    if overall not in OVERALL_STATUSES:
        raise HandoffSchemaError(
            f"invalid overall_status {overall!r}; expected one of {OVERALL_STATUSES}",
            "hf41",
        )

    current = status.get("current_stage")
    if current not in STAGES:
        raise HandoffSchemaError(f"invalid current_stage {current!r}", "hf41")

    completed = status.get("completed_stages")
    if not isinstance(completed, list) or any(s not in STAGES for s in completed):
        raise HandoffSchemaError("completed_stages must be a list of known stages", "hf41")
    if len(set(completed)) != len(completed):
        raise HandoffSchemaError("completed_stages contains duplicates", "hf41")
    if list(completed) != list(STAGES[: len(completed)]):
        raise HandoffSchemaError(
            "completed_stages must be a prefix of the stage order", "hf41"
        )

    required = list(required_stages)
    completed_ids = status.get("completed_asset_ids")
    if not isinstance(completed_ids, list) or any(
        not isinstance(a, str) for a in completed_ids
    ):
        raise HandoffSchemaError(
            "completed_asset_ids must be a list of strings", "hf41"
        )
    if len(set(completed_ids)) != len(completed_ids):
        raise HandoffSchemaError("completed_asset_ids contains duplicates", "hf41")

    failed_assets = status.get("failed_assets")
    if not isinstance(failed_assets, list) or any(
        not isinstance(a, dict) or not a.get("asset_id") for a in failed_assets
    ):
        raise HandoffSchemaError(
            "failed_assets must be a list of objects with asset_id", "hf41"
        )

    environment = status.get("environment")
    if not isinstance(environment, dict):
        raise HandoffSchemaError("status environment must be an object", "hf41")

    problems = generation_config_problems(status.get("generation_config"))
    if problems:
        raise HandoffSchemaError(
            "status generation_config invalid: " + "; ".join(problems), "hf41"
        )

    error = status.get("error")

    if overall == "PENDING":
        if error is not None:
            raise HandoffSchemaError("status PENDING must not carry an error", "hf41")
        if completed or completed_ids or failed_assets:
            raise HandoffSchemaError(
                "status PENDING must have completed nothing", "hf41"
            )
        if current != STAGES[0]:
            raise HandoffSchemaError(
                f"status PENDING must be at {STAGES[0]!r}", "hf41"
            )

    elif overall == "RUNNING":
        if error is not None:
            raise HandoffSchemaError("status RUNNING must not carry an error", "hf41")
        if failed_assets:
            raise HandoffSchemaError(
                "status RUNNING must not carry failed_assets", "hf41"
            )
        if len(completed) >= len(STAGES) or current != STAGES[len(completed)]:
            raise HandoffSchemaError(
                "status RUNNING current_stage must be the stage after the last "
                "completed stage",
                "hf41",
            )

    elif overall == "PASS":
        if error is not None:
            raise HandoffSchemaError("status PASS must not carry an error", "hf41")
        if failed_assets:
            raise HandoffSchemaError("status PASS must not carry failed_assets", "hf41")
        if list(completed) != required:
            raise HandoffSchemaError(
                "status PASS must have completed every required stage", "hf41"
            )
        if current != required[-1]:
            raise HandoffSchemaError(
                f"status PASS current_stage must be {required[-1]!r}", "hf41"
            )
        if expected_asset_ids is None:
            raise HandoffSchemaError(
                "status PASS requires expected_asset_ids for completeness", "hf41"
            )
        if sorted(completed_ids) != sorted(set(expected_asset_ids)):
            raise HandoffSchemaError(
                "status PASS completed_asset_ids must equal the expected asset set",
                "hf41",
            )

    else:  # BLOCKED / FAILED
        code = _validate_error_shape(error)
        if error.get("stage") != current:
            raise HandoffSchemaError(
                "status error.stage must equal current_stage", "hf41"
            )
        if overall_status_for_error(code) != overall:
            raise HandoffSchemaError(
                f"status {overall} contradicts error code {code!r}", "hf41"
            )


# --- markers (content-bearing, marker-last protocol) -------------------------


def marker_path(directory: str | Path, name: str) -> Path:
    """Return the path of a marker file inside ``directory``."""
    return Path(directory) / name


def read_marker(directory: str | Path, name: str) -> str | None:
    """Return the stripped marker content, or ``None`` if it does not exist."""
    path = marker_path(directory, name)
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").strip()


def write_marker(
    directory: str | Path, name: str, *, content: str = ""
) -> Path:
    """Write a marker file (parents created) and return its path.

    A content-bearing marker records the attempt id so a marker from an earlier
    attempt cannot be mistaken for the current one.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(str(content), encoding="utf-8")
    return path


def has_marker(directory: str | Path, name: str) -> bool:
    """Return whether a marker file exists inside ``directory``."""
    return marker_path(directory, name).is_file()


def begin_attempt(output_dir: str | Path, attempt_id: str) -> None:
    """Clear stale output markers/documents for a fresh attempt.

    Removes ``DONE``, ``output_manifest.json`` and ``status.json`` from
    ``output_dir`` so a previous attempt's completion can never be observed as
    the new attempt's. Artifacts (e.g. the ZIP) are left untouched.
    """
    if not attempt_id:
        raise HandoffSchemaError("begin_attempt requires an attempt_id", "hf24")
    output_dir = Path(output_dir)
    for name in (DONE_MARKER, OUTPUT_MANIFEST_FILENAME, STATUS_FILENAME):
        path = output_dir / name
        if path.is_file():
            path.unlink()


def verify_ready(
    input_dir: str | Path,
    *,
    run_id: str | None = None,
    expected_attempt_id: str | None = None,
    image_jobs_path: str | Path | None = None,
) -> dict:
    """Validate a Colab input folder under the marker-last protocol.

    The ``READY`` marker must exist (written last), its content must equal the
    input manifest's attempt id, and the input manifest must be present and
    consistent with the batch.
    """
    input_dir = Path(input_dir)
    ready_content = read_marker(input_dir, READY_MARKER)
    if ready_content is None:
        raise HandoffIntegrityError(
            f"{input_dir}: READY marker is missing (input not complete)", "hf20"
        )
    manifest_path = input_dir / INPUT_MANIFEST_FILENAME
    manifest = load_json_object(manifest_path, code="hf21")
    attempt_id = manifest.get("attempt_id")
    if not attempt_id or ready_content != attempt_id:
        raise HandoffIntegrityError(
            f"{input_dir}: READY marker content {ready_content!r} does not match "
            f"the input manifest attempt_id {attempt_id!r}",
            "hf25",
        )
    jobs_path = (
        Path(image_jobs_path)
        if image_jobs_path is not None
        else input_dir / IMAGE_JOBS_FILENAME
    )
    verify_input_manifest(
        manifest,
        image_jobs_path=jobs_path,
        expected_run_id=run_id,
        expected_attempt_id=expected_attempt_id,
    )
    return manifest


def verify_done(
    output_dir: str | Path,
    *,
    run_id: str | None = None,
    attempt_id: str | None = None,
    image_jobs_sha256: str | None = None,
    expected_asset_ids: list[str] | None = None,
    zip_path: str | Path | None = None,
) -> dict:
    """Validate a Colab output folder under the marker-last protocol.

    Requires the ``DONE`` marker, an output manifest, and a ``status.json`` that
    all agree on the same attempt id and a ``PASS`` status, plus a ZIP whose
    filename is confined to ``output_dir``.
    """
    output_dir = Path(output_dir)
    done_content = read_marker(output_dir, DONE_MARKER)
    if done_content is None:
        raise HandoffIntegrityError(
            f"{output_dir}: DONE marker is missing (output not complete)", "hf22"
        )
    manifest = load_json_object(
        output_dir / OUTPUT_MANIFEST_FILENAME, code="hf23"
    )
    manifest_attempt = manifest.get("attempt_id")
    if not manifest_attempt:
        raise HandoffSchemaError(
            "output manifest is missing attempt_id", "hf24"
        )
    if done_content != manifest_attempt:
        raise HandoffIntegrityError(
            f"{output_dir}: DONE marker content {done_content!r} does not match "
            f"the output manifest attempt_id {manifest_attempt!r}",
            "hf25",
        )
    if attempt_id is not None and manifest_attempt != attempt_id:
        raise HandoffIntegrityError(
            f"output manifest attempt_id {manifest_attempt!r} != expected "
            f"{attempt_id!r}",
            "hf28",
        )

    zip_meta = manifest.get("zip") or {}
    if not _safe_bare_name(zip_meta.get("filename")):
        raise HandoffSchemaError(
            f"output manifest zip.filename {zip_meta.get('filename')!r} is unsafe",
            "hf27",
        )

    status = load_json_object(output_dir / STATUS_FILENAME, code="hf26")
    if status.get("attempt_id") != manifest_attempt:
        raise HandoffIntegrityError(
            f"{output_dir}: status.json attempt_id {status.get('attempt_id')!r} "
            f"does not match the output manifest attempt_id {manifest_attempt!r}",
            "hf26",
        )
    validate_status(status, expected_asset_ids=expected_asset_ids)
    if status.get("overall_status") != "PASS":
        raise HandoffIntegrityError(
            f"{output_dir}: output status is {status.get('overall_status')!r}, "
            "not PASS",
            "hf48",
        )

    resolved_zip = (
        Path(zip_path) if zip_path is not None else output_dir / zip_meta["filename"]
    )
    verify_output_manifest(
        manifest,
        expected_run_id=run_id,
        expected_image_jobs_sha256=image_jobs_sha256,
        zip_path=resolved_zip,
    )
    return manifest


# --- ZIP integrity -----------------------------------------------------------


def _normalize_member(name: str) -> str:
    normalized = str(name).replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def verify_zip(
    zip_path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
    expected_members: list[str] | None = None,
) -> list[str]:
    """Validate a ZIP archive and return its file members.

    Checks existence, optional SHA-256/size fingerprint, readability, member
    safety (no absolute paths, no ``..``), duplicate normalized members, and the
    member set when expected. Raises :class:`HandoffError` /
    :class:`HandoffIntegrityError` on failure.
    """
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise HandoffError(f"ZIP not found: {zip_path}", "hf30")
    if not zipfile.is_zipfile(zip_path):
        raise HandoffError(f"not a valid ZIP archive: {zip_path}", "hf31")

    if expected_sha256 is not None and sha256_file(zip_path) != expected_sha256:
        raise HandoffIntegrityError("ZIP sha256 does not match the manifest", "hf32")
    if expected_bytes is not None and file_size(zip_path) != int(expected_bytes):
        raise HandoffIntegrityError("ZIP byte size does not match the manifest", "hf32")

    with zipfile.ZipFile(zip_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise HandoffIntegrityError(
                f"ZIP contains a corrupt member: {bad_member}", "hf33"
            )
        members: list[str] = []
        seen: set[str] = set()
        for info in archive.infolist():
            if info.is_dir():
                continue
            normalized = _normalize_member(info.filename)
            if normalized.startswith("/") or ".." in Path(normalized).parts:
                raise HandoffIntegrityError(
                    f"ZIP member has an unsafe path: {info.filename}", "hf34"
                )
            if normalized in seen:
                raise HandoffIntegrityError(
                    f"ZIP has a duplicate normalized member: {info.filename}", "hf36"
                )
            seen.add(normalized)
            members.append(normalized)

    if expected_members is not None:
        expected_set = {_normalize_member(m) for m in expected_members}
        actual_set = set(members)
        missing = sorted(expected_set - actual_set)
        unexpected = sorted(actual_set - expected_set)
        if missing or unexpected:
            raise HandoffIntegrityError(
                f"ZIP members do not match the manifest "
                f"(missing={missing}, unexpected={unexpected})",
                "hf35",
            )
    return sorted(members)


# --- result document (image-generation-results/1.0) --------------------------

# The in-ZIP ``result.json`` document. Its content contract is defined by the
# worker's publisher: ``build_result_document``/``publish_outputs`` in
# ``notebooks/handoff_notebook_core.py`` (the only producer). The checks below
# enforce exactly what that producer emits and what a completed (``DONE``)
# artifact implies; they do not invent fields the producer never writes.

RESULT_DOCUMENT_STATUSES = ("completed", "partial_failure")
RESULT_ITEM_STATUSES = ("generated", "failed")

RESULT_DOCUMENT_REQUIRED_FIELDS = (
    "schema",
    "run_id",
    "source_json",
    "channel_id",
    "title",
    "status",
    "job_count",
    "generated_count",
    "skipped_count",
    "failed_count",
    "resolution",
    "steps",
    "guidance_scale",
    "results",
)

RESULT_ITEM_REQUIRED_FIELDS = (
    "scene_id",
    "asset_id",
    "filename",
    "status",
    "seed",
    "width",
    "height",
)


def verify_result_document(
    document: object,
    *,
    expected_run_id: str | None = None,
    expected_asset_ids: list[str] | None = None,
    expected_resolution: tuple[int, int] | None = None,
    expected_generation_config: dict | None = None,
    require_completed: bool = False,
) -> None:
    """Validate an ``image-generation-results/1.0`` result document.

    Enforces the producer contract (``build_result_document``): the exact
    top-level field set the worker writes, a supported ``status``, count
    consistency with ``results``, per-item field/asset/filename/status/size
    consistency, and — when given — the expected run id, asset set, resolution
    and generation config. ``require_completed`` additionally rejects the
    ``partial_failure`` status, which the worker never packages.

    The document carries no attempt id (the producer does not write one); the
    attempt binding of the artifact is proven by the output manifest and its
    ZIP hash instead. The result document is therefore validated as a
    consistency cross-check, never as the source of the expected asset set.
    """
    if not isinstance(document, dict):
        raise HandoffSchemaError("result document must be a JSON object", "hf49")
    if document.get("schema") != SCHEMA_RESULT_MANIFEST:
        raise HandoffSchemaError(f"not a {SCHEMA_RESULT_MANIFEST} document", "hf49")

    missing = [f for f in RESULT_DOCUMENT_REQUIRED_FIELDS if f not in document]
    if missing:
        raise HandoffSchemaError(
            f"result document is missing {', '.join(missing)}", "hf49"
        )

    run_id = document.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise HandoffSchemaError(
            "result document run_id must be a non-empty string", "hf49"
        )
    if expected_run_id is not None and run_id != expected_run_id:
        raise HandoffIntegrityError(
            f"result document run_id {run_id!r} != expected {expected_run_id!r}",
            "hf50",
        )

    if document.get("source_json") != IMAGE_JOBS_FILENAME:
        raise HandoffSchemaError(
            f"result document source_json {document.get('source_json')!r} != "
            f"{IMAGE_JOBS_FILENAME!r}",
            "hf49",
        )
    if not isinstance(document.get("channel_id"), str):
        raise HandoffSchemaError(
            "result document channel_id must be a string", "hf49"
        )
    title = document.get("title")
    if not (title is None or isinstance(title, str)):
        raise HandoffSchemaError(
            "result document title must be a string or null", "hf49"
        )

    status = document.get("status")
    if status not in RESULT_DOCUMENT_STATUSES:
        raise HandoffSchemaError(
            f"invalid result status {status!r}; expected one of "
            f"{RESULT_DOCUMENT_STATUSES}",
            "hf49",
        )
    if require_completed and status != "completed":
        raise HandoffIntegrityError(
            f"result document status {status!r} is not 'completed' for a "
            "published artifact",
            "hf50",
        )

    resolution = document.get("resolution")
    if not isinstance(resolution, dict):
        raise HandoffSchemaError("result document resolution must be an object", "hf49")
    for axis in ("width", "height"):
        value = resolution.get(axis)
        if not isinstance(value, int) or isinstance(value, bool):
            raise HandoffSchemaError(
                f"result document resolution.{axis} must be an integer", "hf49"
            )
    if expected_resolution is not None:
        actual = (resolution.get("width"), resolution.get("height"))
        expected_pair = (int(expected_resolution[0]), int(expected_resolution[1]))
        if actual != expected_pair:
            raise HandoffIntegrityError(
                f"result document resolution {actual} != expected {expected_pair}",
                "hf50",
            )

    counts: dict[str, int] = {}
    for field in ("job_count", "generated_count", "skipped_count", "failed_count"):
        value = document.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise HandoffSchemaError(
                f"result document {field} must be a non-negative integer", "hf49"
            )
        counts[field] = value

    results = document.get("results")
    if not isinstance(results, list):
        raise HandoffSchemaError("result document results must be a list", "hf49")

    seen: set[str] = set()
    item_status_counts = {name: 0 for name in RESULT_ITEM_STATUSES}
    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            raise HandoffSchemaError(f"result item {index} is not an object", "hf49")
        missing_item = [f for f in RESULT_ITEM_REQUIRED_FIELDS if f not in item]
        if missing_item:
            raise HandoffSchemaError(
                f"result item {index} is missing {', '.join(missing_item)}", "hf49"
            )

        asset_id = item.get("asset_id")
        if not isinstance(asset_id, str) or not SAFE_ASSET_ID_RE.match(asset_id):
            raise HandoffSchemaError(
                f"result item {index} has unsafe asset_id {asset_id!r}", "hf49"
            )
        if asset_id in seen:
            raise HandoffIntegrityError(
                f"result document results contains duplicate asset_id {asset_id!r}",
                "hf50",
            )
        seen.add(asset_id)

        filename = item.get("filename")
        if filename != f"{asset_id}.png":
            raise HandoffIntegrityError(
                f"result item {index} filename {filename!r} does not match "
                f"asset_id {asset_id!r}",
                "hf50",
            )

        item_status = item.get("status")
        if item_status not in RESULT_ITEM_STATUSES:
            raise HandoffSchemaError(
                f"result item {index} has invalid status {item_status!r}; expected "
                f"one of {RESULT_ITEM_STATUSES}",
                "hf49",
            )
        item_status_counts[item_status] += 1

        scene_id = item.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise HandoffSchemaError(
                f"result item {index} scene_id must be a non-empty string", "hf49"
            )
        seed = item.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise HandoffSchemaError(
                f"result item {index} seed must be an integer", "hf49"
            )
        for axis in ("width", "height"):
            value = item.get(axis)
            if not isinstance(value, int) or isinstance(value, bool):
                raise HandoffSchemaError(
                    f"result item {index} {axis} must be an integer", "hf49"
                )
        if (item["width"], item["height"]) != (
            resolution.get("width"),
            resolution.get("height"),
        ):
            raise HandoffIntegrityError(
                f"result item {index} size ({item['width']}x{item['height']}) does "
                "not match the document resolution",
                "hf50",
            )

    if counts["job_count"] != len(results):
        raise HandoffIntegrityError(
            f"result document job_count {counts['job_count']} != len(results) "
            f"{len(results)}",
            "hf50",
        )
    if counts["generated_count"] != item_status_counts["generated"]:
        raise HandoffIntegrityError(
            f"result document generated_count {counts['generated_count']} != "
            f"{item_status_counts['generated']} generated result item(s)",
            "hf50",
        )
    if counts["failed_count"] != item_status_counts["failed"]:
        raise HandoffIntegrityError(
            f"result document failed_count {counts['failed_count']} != "
            f"{item_status_counts['failed']} failed result item(s)",
            "hf50",
        )
    # The producer hardcodes skipped_count to 0 and never skips a job.
    if counts["skipped_count"] != 0:
        raise HandoffIntegrityError(
            "result document skipped_count must be 0 (the worker never skips a job)",
            "hf50",
        )
    # The producer sets status from the failed count; the two must agree.
    if status == "completed" and counts["failed_count"] != 0:
        raise HandoffIntegrityError(
            "result document claims 'completed' but has failed_count "
            f"{counts['failed_count']}",
            "hf50",
        )
    if status == "partial_failure" and counts["failed_count"] == 0:
        raise HandoffIntegrityError(
            "result document claims 'partial_failure' but has failed_count 0",
            "hf50",
        )

    if expected_asset_ids is not None:
        expected_set = set(expected_asset_ids)
        missing_assets = sorted(expected_set - seen)
        unexpected_assets = sorted(seen - expected_set)
        if missing_assets or unexpected_assets:
            raise HandoffIntegrityError(
                "result document asset set mismatch "
                f"(missing={missing_assets}, unexpected={unexpected_assets})",
                "hf50",
            )

    if expected_generation_config is not None:
        for field in ("steps", "guidance_scale"):
            if field not in expected_generation_config:
                continue
            expected_value = expected_generation_config[field]
            if document.get(field) != expected_value:
                raise HandoffIntegrityError(
                    f"result document {field} {document.get(field)!r} does not "
                    f"match the generation_config {expected_value!r}",
                    "hf50",
                )


def read_result_document_from_zip(
    zip_path: str | Path, *, result_filename: str = RESULT_FILENAME
) -> dict:
    """Read and parse the result-document member of a ZIP (never extracted).

    The member must exist, be UTF-8 JSON and contain a JSON object. The loose
    ``output/result.json`` sibling is deliberately ignored: only the in-ZIP
    member (bound by the manifest's ZIP hash) is trusted.
    """
    zip_path = Path(zip_path)
    if not _safe_bare_name(result_filename):
        raise HandoffSchemaError(
            f"result document filename {result_filename!r} is unsafe", "hf49"
        )
    if not zipfile.is_zipfile(zip_path):
        raise HandoffError(f"not a valid ZIP archive: {zip_path}", "hf31")
    with zipfile.ZipFile(zip_path) as archive:
        try:
            raw = archive.read(result_filename)
        except KeyError as exc:
            raise HandoffError(
                f"ZIP is missing the result document member {result_filename!r}",
                "hf49",
            ) from exc
    try:
        data = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise HandoffSchemaError(
            f"result document {result_filename!r} is not valid UTF-8", "hf49"
        ) from exc
    except json.JSONDecodeError as exc:
        raise HandoffSchemaError(
            f"result document {result_filename!r} is not valid JSON: {exc}", "hf49"
        ) from exc
    if not isinstance(data, dict):
        raise HandoffSchemaError(
            f"result document {result_filename!r} must contain a JSON object",
            "hf49",
        )
    return data


def verify_zip_result_document(
    zip_path: str | Path,
    *,
    output_manifest: dict,
    run_id: str | None = None,
) -> None:
    """Validate the in-ZIP result document of a published output ZIP.

    Reads the member named by the output manifest's ``result_manifest.filename``
    from ``zip_path`` itself (the ZIP whose hash the manifest pins — never the
    loose ``output/result.json``) and validates it against the manifest's run
    id, generated asset set, resolution and generation config. The document is
    a consistency cross-check only: the expected import assets still come from
    ``image_jobs.json``, never from this untrusted-side metadata.
    """
    if (
        not isinstance(output_manifest, dict)
        or output_manifest.get("schema") != SCHEMA_OUTPUT_MANIFEST
    ):
        raise HandoffSchemaError(
            f"not a {SCHEMA_OUTPUT_MANIFEST} document", "hf17"
        )
    result_meta = output_manifest.get("result_manifest") or {}
    result_filename = result_meta.get("filename") or RESULT_FILENAME

    resolution = output_manifest.get("resolution") or {}
    expected_resolution = None
    if resolution.get("width") is not None and resolution.get("height") is not None:
        expected_resolution = (resolution.get("width"), resolution.get("height"))

    document = read_result_document_from_zip(
        zip_path, result_filename=result_filename
    )
    verify_result_document(
        document,
        expected_run_id=run_id if run_id is not None else output_manifest.get("run_id"),
        expected_asset_ids=list(output_manifest.get("generated_asset_ids") or []),
        expected_resolution=expected_resolution,
        expected_generation_config=output_manifest.get("generation_config"),
        require_completed=True,
    )


# --- GPU preflight (pure) ----------------------------------------------------


def evaluate_gpu_preflight(
    *,
    cuda_available: bool,
    device_count: int = 0,
    gpu_name: str = "",
    total_vram_gb: float | None = None,
    free_vram_gb: float | None = None,
    cuda_op_ok: bool = False,
    nvidia_smi: str = "",
    total_min_gib: float = VRAM_TOTAL_MIN_GIB,
    free_min_gib: float = VRAM_FREE_MIN_GIB,
) -> dict:
    """Map observed GPU facts to a preflight decision (no torch, no CUDA).

    Fail-closed: unknown (``None``) or non-finite (NaN/±inf) VRAM readings are
    treated as insufficient rather than as passing. Returns
    ``{"ok": bool, "error_code": None | str, "diagnostics": {...}}``; the caller
    uses :func:`overall_status_for_error` to derive the run status.
    """
    diagnostics = {
        "cuda_available": bool(cuda_available),
        "device_count": int(device_count),
        "gpu_name": gpu_name,
        "total_vram_gb": total_vram_gb,
        "free_vram_gb": free_vram_gb,
        "cuda_op_ok": bool(cuda_op_ok),
        "nvidia_smi": nvidia_smi,
        "required_total_vram_gib": float(total_min_gib),
        "required_free_vram_gib": float(free_min_gib),
    }

    error_code: str | None = None
    if not cuda_available or int(device_count) < 1:
        error_code = "NO_GPU"
    elif not cuda_op_ok:
        error_code = "CUDA_INIT_FAILED"
    elif not _finite(total_vram_gb) or float(total_vram_gb) < float(total_min_gib):
        error_code = "INSUFFICIENT_VRAM"
    elif not _finite(free_vram_gb) or float(free_vram_gb) < float(free_min_gib):
        error_code = "INSUFFICIENT_VRAM"

    diagnostics["ok"] = error_code is None
    return {"ok": error_code is None, "error_code": error_code, "diagnostics": diagnostics}
