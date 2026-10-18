"""Pure-standard-library logic for the Colab handoff worker notebook.

This module is the **single source of truth** for every part of the notebook
that does not need ``torch``, Google Drive, or PIL. It is:

* importable and unit-tested offline (``tests/test_notebook_handoff.py``); and
* vendored **verbatim** into ``notebooks/SDXL_Lightning_handoff.ipynb``.

Vendoring contract
------------------

The notebook executes this file and ``agents/image_jobs/handoff.py`` flat, in the
same namespace, rather than importing them from the repository (Colab has no
checkout, and the notebook must not fetch project code from the network). To let
the *same source* work both as an importable module and as flat notebook code,
the contract is resolved through the ``_handoff`` alias below:

* imported normally -> ``from agents.image_jobs import handoff``;
* executed flat in the notebook -> the name resolves to the running namespace,
  which already holds the flat, vendored contract names.

Because the code is byte-for-byte identical in both places, an offline drift
check can assert that the notebook's vendored cells equal these source files
exactly.

Nothing here mints an attempt id, upgrades a schema, or starts another stage.
"""

from __future__ import annotations

import hmac
import json
from pathlib import Path

try:  # importable module (repository / pytest)
    from agents.image_jobs import handoff as _handoff
except ImportError:  # pragma: no cover - only hit when executed flat
    import sys as _sys

    _handoff = _sys.modules[__name__]
    if not hasattr(_handoff, "verify_ready"):
        # Not flat execution: a genuine import failure we must not swallow.
        raise

# --- constants mirrored from tools/handoff_transport.py ----------------------
#
# The sentinel contract lives in the (non-vendored) rclone transport module, so
# the notebook keeps its own copy of the *constants only* and implements the
# same fail-closed check locally. The values are asserted against
# tools/handoff_transport.py by the offline tests.

SENTINEL_FILENAME = "handoff_sentinel.json"
SENTINEL_SCHEMA_VERSION = 1
HANDOFF_PROJECT_ID = "ai-youtube-channel"

# The Drive base scanned for the single handoff root (fail-closed discovery).
DRIVE_HANDOFF_BASE = "/content/drive/MyDrive"

# Flat output ZIP name (approved D1/Phase-2 decision; the local verifier reads
# the name from the output manifest, so this is a naming convention only).
OUTPUT_ZIP_SUFFIX = "_generated_assets.zip"

# Prompt budget: hard error above 77 tokens, warning above 60 (original notebook).
HARD_PROMPT_TOKEN_LIMIT = 77
WARN_PROMPT_TOKEN_LIMIT = 60

SAFE_RUN_ID_RE = _handoff.SAFE_ASSET_ID_RE


class NotebookHandoffError(ValueError):
    """A fail-closed problem detected by the notebook worker (no side effects)."""


# --- root discovery ----------------------------------------------------------


def discover_handoff_root(base_dir: str | Path) -> Path:
    """Return the single ``ai-youtube-channel-handoff`` directory under ``base_dir``.

    Fail-closed: a missing root, more than one candidate, a case-mismatched
    name, or a non-directory stops with :class:`NotebookHandoffError`. The
    notebook never creates a root (that is the local ``prepare`` workflow's job)
    and never picks the newest or first match.
    """
    base = Path(base_dir)
    if not base.is_dir():
        raise NotebookHandoffError(f"Drive base is not a directory: {base}")

    root_name = _handoff.HANDOFF_ROOT_NAME
    candidates = [entry.name for entry in base.iterdir() if entry.name.lower() == root_name.lower()]

    if not candidates:
        raise NotebookHandoffError(
            f"handoff root {root_name!r} not found under {base}"
        )
    if len(candidates) > 1:
        raise NotebookHandoffError(
            f"ambiguous handoff root under {base}: multiple entries match "
            f"{root_name!r}: {sorted(candidates)}"
        )

    found = candidates[0]
    if found != root_name:
        raise NotebookHandoffError(
            f"ambiguous handoff root {found!r} under {base}: expected exactly "
            f"{root_name!r} (case mismatch)"
        )
    root = base / found
    if not root.is_dir():
        raise NotebookHandoffError(f"handoff root is not a directory: {root}")
    return root


# --- sentinel trust ----------------------------------------------------------


def verify_sentinel(root_dir: str | Path, expected_value: str) -> dict:
    """Verify ``root_dir/handoff_sentinel.json`` against an independent value.

    The expected value must come from a trusted source outside the remote root
    (the operator's Colab Secret, ultimately the Windows DPAPI store). The remote
    file is never trusted on its own. Fails closed on a missing, malformed,
    mismatched, or wrong-project/root/version sentinel.
    """
    if not isinstance(expected_value, str) or not expected_value:
        raise NotebookHandoffError(
            "a trusted sentinel value is required (Colab Secret "
            "HANDOFF_SENTINEL_VALUE); refusing to proceed"
        )

    path = Path(root_dir) / SENTINEL_FILENAME
    if not path.is_file():
        raise NotebookHandoffError(f"handoff sentinel is missing: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NotebookHandoffError(f"handoff sentinel is unreadable: {exc}") from exc
    if not isinstance(document, dict):
        raise NotebookHandoffError("handoff sentinel must be a JSON object")

    if document.get("project") != HANDOFF_PROJECT_ID:
        raise NotebookHandoffError(
            f"sentinel project {document.get('project')!r} != {HANDOFF_PROJECT_ID!r}"
        )
    if document.get("root") != _handoff.HANDOFF_ROOT_NAME:
        raise NotebookHandoffError(
            f"sentinel root {document.get('root')!r} != {_handoff.HANDOFF_ROOT_NAME!r}"
        )
    if document.get("sentinel_version") != SENTINEL_SCHEMA_VERSION:
        raise NotebookHandoffError(
            f"sentinel version {document.get('sentinel_version')!r} != "
            f"{SENTINEL_SCHEMA_VERSION!r}"
        )
    remote_value = document.get("verification_value")
    if not isinstance(remote_value, str) or not remote_value:
        raise NotebookHandoffError("sentinel has no verification_value")
    if not hmac.compare_digest(remote_value, expected_value):
        raise NotebookHandoffError(
            "sentinel verification_value does not match the trusted expected value "
            "(the Drive account/root does not match)"
        )
    return document


# --- explicit run selection + input validation -------------------------------


def select_run(root_dir: str | Path, run_id: str) -> Path:
    """Return ``<root_dir>/<run_id>`` for an explicit, valid, existing run.

    The operator names the run; the notebook never selects the newest run. A bare
    safe run id is required (no separators / ``..`` / absolute paths).
    """
    if not isinstance(run_id, str) or not run_id.strip() or run_id != run_id.strip():
        raise NotebookHandoffError("an explicit run_id is required")
    if not SAFE_RUN_ID_RE.match(run_id):
        raise NotebookHandoffError(f"unsafe run_id {run_id!r}")
    run_dir = Path(root_dir) / run_id
    if not run_dir.is_dir():
        raise NotebookHandoffError(f"run folder not found: {run_dir}")
    return run_dir


def verify_run_input(root_dir: str | Path, run_id: str) -> dict:
    """Validate the run's input via the schema-1.1 contract and return the manifest.

    Delegates to :func:`handoff.verify_ready`, which requires ``input/READY``
    (content == the input manifest ``attempt_id``) and re-checks the manifest
    against ``input/image_jobs.json`` (schema, attempt id, SHA-256, asset set).
    """
    run_dir = select_run(root_dir, run_id)
    input_dir = run_dir / _handoff.INPUT_DIR_NAME
    jobs_path = input_dir / _handoff.IMAGE_JOBS_FILENAME
    return _handoff.verify_ready(input_dir, run_id=run_id, image_jobs_path=jobs_path)


def invalidate_stale_completion(
    output_dir: str | Path, *, attempt_id: str
) -> Path:
    """Clear a previous attempt's completion before a new attempt can fail.

    ``handoff.begin_attempt`` is contract-compatible and removes ``DONE``,
    ``output_manifest.json`` and ``output/status.json`` while leaving artifacts
    (such as the ZIP) in place. The notebook calls this immediately after input
    validation and **before** GPU preflight, so a rerun that later fails can
    never leave an earlier attempt's ``DONE`` visible as the current result —
    even when the attempt id is unchanged (e.g. the run was not re-prepared).

    This is the same invalidation the local ``prepare --force`` path performs;
    the notebook performs it again because a worker may be re-run without a new
    ``prepare``.
    """
    if not isinstance(attempt_id, str) or not attempt_id:
        raise NotebookHandoffError(
            "invalidate_stale_completion requires an attempt_id"
        )
    output_dir = Path(output_dir)
    _handoff.begin_attempt(output_dir, attempt_id)
    return output_dir


# --- GPU preflight gate (pure mapping) ---------------------------------------


def preflight_decision(facts: dict) -> dict:
    """Map observed GPU facts to a fail-closed decision (no torch/CUDA here).

    Thin wrapper over :func:`handoff.evaluate_gpu_preflight` so the notebook and
    the offline tests share one gate. Unknown / non-finite VRAM is insufficient.
    """
    return _handoff.evaluate_gpu_preflight(**facts)


# --- prompt token budget (pure) ----------------------------------------------


def prompt_token_finding(longest_tokens: int) -> tuple[str | None, str]:
    """Classify a prompt's longest token count.

    Returns ``("error"|"warning"|None, message)``: above the hard limit is an
    error, above the soft target is a warning, otherwise no finding. The caller
    supplies counts from the CLIP tokenizers.
    """
    if longest_tokens > HARD_PROMPT_TOKEN_LIMIT:
        return (
            "error",
            f"prompt has {longest_tokens} tokens; limit is {HARD_PROMPT_TOKEN_LIMIT}",
        )
    if longest_tokens > WARN_PROMPT_TOKEN_LIMIT:
        return (
            "warning",
            f"prompt has {longest_tokens} tokens (above the preferred "
            f"{WARN_PROMPT_TOKEN_LIMIT}-token target)",
        )
    return (None, "")


def prompt_validation_findings(
    jobs: list[dict], longest_tokens_for
) -> tuple[list[str], list[str]]:
    """Return ``(errors, warnings)`` for a batch's positive/negative prompts.

    ``longest_tokens_for(prompt) -> int`` supplies the CLIP token count; the
    notebook passes a tokenizer-backed callable and tests pass a stub. An empty
    positive prompt is an error. The caller decides what to do with the result;
    this function has no side effects.
    """
    errors: list[str] = []
    warnings: list[str] = []
    for item in jobs:
        for field in ("positive_prompt", "negative_prompt"):
            prompt = item.get(field, "")
            if field == "positive_prompt" and not str(prompt).strip():
                errors.append(f"{item['scene_id']}: empty positive prompt")
                continue
            kind, message = prompt_token_finding(longest_tokens_for(prompt))
            if kind == "error":
                errors.append(f"{item['scene_id']} | {field}: {message}")
            elif kind == "warning":
                warnings.append(f"{item['scene_id']} | {field}: {message}")
    return errors, warnings


# --- result document ---------------------------------------------------------


def build_result_document(
    run_id: str,
    batch: dict,
    results: list[dict],
    *,
    resolution: tuple[int, int],
    generation_config: dict,
) -> dict:
    """Build the ``image-generation-results/1.0`` ``result.json`` document.

    Publication requires the complete asset set, so ``status`` is always
    ``"completed"`` here; a partial run is never packaged (see
    :func:`publish_outputs`).
    """
    generated = sum(1 for item in results if item.get("status") == "generated")
    failed = sum(1 for item in results if item.get("status") == "failed")
    return {
        "schema": _handoff.SCHEMA_RESULT_MANIFEST,
        "run_id": run_id,
        "source_json": _handoff.IMAGE_JOBS_FILENAME,
        "channel_id": batch.get("channel_id", ""),
        "title": batch.get("title"),
        "status": "completed" if failed == 0 else "partial_failure",
        "job_count": len(batch.get("jobs", [])),
        "generated_count": generated,
        "skipped_count": 0,
        "failed_count": failed,
        "resolution": {"width": int(resolution[0]), "height": int(resolution[1])},
        "steps": generation_config.get("steps"),
        "guidance_scale": generation_config.get("guidance_scale"),
        "results": list(results),
    }


# --- progress / terminal status ----------------------------------------------


def write_progress_status(
    status_dir: str | Path,
    *,
    run_id: str,
    attempt_id: str,
    image_jobs_sha256: str,
    generation_config: dict,
    completed_stages: list[str],
    current_stage: str,
    completed_asset_ids: list[str] | None = None,
    environment: dict | None = None,
) -> dict:
    """Write a ``RUNNING`` ``status/status.json`` (progress) and validate it."""
    status = _handoff.build_status(
        run_id,
        attempt_id=attempt_id,
        image_jobs_sha256=image_jobs_sha256,
        generation_config=generation_config,
        overall_status="RUNNING",
        current_stage=current_stage,
        completed_stages=list(completed_stages),
        completed_asset_ids=list(completed_asset_ids or []),
        environment=environment,
    )
    _handoff.validate_status(status)
    _handoff.dump_json(status, Path(status_dir) / _handoff.STATUS_FILENAME)
    return status


def write_failure_status(
    status_dir: str | Path,
    *,
    run_id: str,
    attempt_id: str,
    image_jobs_sha256: str,
    generation_config: dict,
    code: str,
    stage: str,
    message: str,
    diagnostics: dict | None = None,
    completed_stages: list[str] | None = None,
    environment: dict | None = None,
) -> dict:
    """Write a terminal ``BLOCKED``/``FAILED`` progress status (never a DONE)."""
    error = _handoff.make_error(code, stage, message, diagnostics=diagnostics)
    overall = _handoff.overall_status_for_error(code)
    status = _handoff.build_status(
        run_id,
        attempt_id=attempt_id,
        image_jobs_sha256=image_jobs_sha256,
        generation_config=generation_config,
        overall_status=overall,
        current_stage=stage,
        completed_stages=list(completed_stages or []),
        error=error,
        environment=environment,
    )
    _handoff.validate_status(status)
    _handoff.dump_json(status, Path(status_dir) / _handoff.STATUS_FILENAME)
    return status


# --- output validation, packaging, marker-last publication -------------------


def _expected_members(asset_ids: list[str]) -> list[str]:
    members = [f"{asset_id}.png" for asset_id in asset_ids]
    members.append(_handoff.RESULT_FILENAME)
    return sorted(members)


def publish_outputs(
    output_dir: str | Path,
    *,
    run_id: str,
    attempt_id: str,
    image_jobs_sha256: str,
    generated_files: dict[str, str | Path],
    result_document: dict,
    expected_asset_ids: list[str],
    resolution: tuple[int, int],
    generation_config: dict,
    environment: dict | None = None,
) -> dict:
    """Validate, package and publish one run's output, marker-last.

    Steps (all before the ``DONE`` marker):

    1. require the generated set to equal ``expected_asset_ids`` exactly;
    2. ``begin_attempt`` to clear any stale completion from an earlier attempt;
    3. write ``result.json`` and build ``<run_id>_generated_assets.zip``
       (payload);
    4. build, write and verify ``output_manifest.json`` (schema 1.1);
    5. write and verify the final ``PASS`` ``output/status.json``;
    6. write ``DONE`` **last** (content == ``attempt_id``).

    Any failure before step 6 raises and leaves no ``DONE``. The generated assets
    are never written into the local ``data/`` tree and are never imported here.
    """
    output_dir = Path(output_dir)
    expected = list(expected_asset_ids)
    if sorted(generated_files) != sorted(expected):
        raise NotebookHandoffError(
            "generated asset set does not match the expected set "
            f"(generated={sorted(generated_files)}, expected={sorted(expected)})"
        )
    if sorted(set(expected)) != sorted(expected):
        raise NotebookHandoffError("expected_asset_ids contains duplicates")

    for asset_id in expected:
        if not Path(generated_files[asset_id]).is_file():
            raise NotebookHandoffError(f"generated asset missing: {asset_id}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _handoff.begin_attempt(output_dir, attempt_id)

    result_path = output_dir / _handoff.RESULT_FILENAME
    _handoff.dump_json(result_document, result_path)

    import zipfile

    zip_path = output_dir / f"{run_id}{OUTPUT_ZIP_SUFFIX}"
    if zip_path.is_file():
        zip_path.unlink()
    members = _expected_members(expected)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for asset_id in expected:
            archive.write(Path(generated_files[asset_id]), arcname=f"{asset_id}.png")
        archive.write(result_path, arcname=_handoff.RESULT_FILENAME)

    manifest = _handoff.build_output_manifest(
        run_id,
        attempt_id=attempt_id,
        image_jobs_sha256=image_jobs_sha256,
        zip_path=zip_path,
        members=members,
        generated_asset_ids=expected,
        resolution=resolution,
        generation_config=generation_config,
    )
    _handoff.dump_json(manifest, output_dir / _handoff.OUTPUT_MANIFEST_FILENAME)
    _handoff.verify_output_manifest(
        manifest,
        expected_run_id=run_id,
        expected_attempt_id=attempt_id,
        expected_image_jobs_sha256=image_jobs_sha256,
        zip_path=zip_path,
    )

    status = _handoff.build_status(
        run_id,
        attempt_id=attempt_id,
        image_jobs_sha256=image_jobs_sha256,
        generation_config=generation_config,
        overall_status="PASS",
        current_stage=_handoff.STAGES[-1],
        completed_stages=list(_handoff.STAGES),
        completed_asset_ids=expected,
        environment=environment,
    )
    _handoff.validate_status(status, expected_asset_ids=expected)
    _handoff.dump_json(status, output_dir / _handoff.STATUS_FILENAME)

    # DONE is written strictly last; its content binds it to this attempt.
    _handoff.write_marker(output_dir, _handoff.DONE_MARKER, content=attempt_id)
    return manifest
