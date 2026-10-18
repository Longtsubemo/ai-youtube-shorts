"""Local handoff CLI for the manual Colab SDXL -> Google Drive workflow.

This is the **local** side of the handoff described in
``docs/COLAB_HANDOFF.md``. By default it operates on a plain local directory
that simulates the shared Google Drive folder, using an offline backend. It does
**not** talk to Google Drive, does not use rclone, does not authenticate, and
does not run any GPU code. Swapping in a Drive/rclone backend later means
changing only how the handoff root is materialised locally; the manifest,
marker and ZIP integrity logic is unchanged and lives in
``agents/image_jobs/handoff.py`` (pure standard library).

An optional ``--remote local:<dir>`` mode (Milestone 3B.1) uses the same
offline code against a directory-backed "remote": the run folder is mirrored to
a temporary local directory for the duration of the command and published back
with ``READY`` written last. This is still fully offline.

``--remote rclone:<remote>`` (Milestone 3B.2) plugs the same seam into the
rclone-backed transport (e.g. ``--remote rclone:gdrive:``). Before any transfer
it runs a read-only preflight: it checks remote accessibility, discovers exactly
one ``ai-youtube-channel-handoff`` root (creating it only with an explicit
``--create-root`` on ``prepare``), and verifies the root's sentinel against the
trusted ``--sentinel-value`` supplied by the operator. A missing/ambiguous root
or a missing/mismatched sentinel fails closed. rclone is only ever invoked when
an rclone transport method runs; importing this module never spawns a process.

Subcommands::

    prepare  --run-id <id>            # build input manifest + READY (marker last)
    status   --run-id <id>            # show current stage / attempt / completion
    fetch    --run-id <id>            # verify completion, stage the ZIP locally
    validate --run-id <id>            # verify completion + ZIP, then importer dry-run (non-destructive)
    import   --run-id <id> --authorize-import  # real import (overwrite off unless --overwrite)

Folder layout under ``--handoff-root`` (default ``ai-youtube-channel-handoff``)::

    ai-youtube-channel-handoff/
      <run_id>/
        input/
          image_jobs.json          # copied from data/<run_id>/image_jobs.json
          input_manifest.json      # sdxl-input-manifest/1.1
          READY                    # marker; content = attempt_id; written LAST
        status/
          status.json              # sdxl-status/1.1 (progress)
        output/
          status.json              # final sdxl-status/1.1
          output_manifest.json     # sdxl-output-manifest/1.1
          <run_id>_generated_assets.zip   # flat: <asset_id>.png + result.json
          DONE                     # marker; content = attempt_id; written LAST

Integrity rules enforced here (all delegated to the hardened helpers):

- schema 1.1 and the attempt-id contract;
- ``READY`` is written only after the copied input files and hashes verify;
- ``DONE`` is trusted only after ZIP, result manifest, output manifest, status
  and hashes validate (``verify_done``);
- the approved ``expected_asset_ids`` are always passed to ``verify_done``;
- the input/output pair must cross-bind (``verify_handoff_pair``);
- stale markers, mismatched attempts, invalid manifests, corrupt ZIPs, unsafe
  paths and unexpected assets are rejected;
- ``validate`` and ``import`` only ever act on the completed attempt's
  artifact: the resolved ZIP (explicit, staged or in ``output/``) must match
  the output manifest's SHA-256/size, and its in-ZIP ``result.json`` must
  satisfy the ``image-generation-results/1.0`` contract for this run;
- the run id must be a bare directory name (checked before any path is built),
  so it can never escape the handoff root;
- ``fetch`` stages the ZIP outside ``data/`` and never extracts it;
- ``validate`` is non-destructive; ``import`` requires ``--authorize-import``
  and never overwrites unless ``--overwrite`` is given;
- the source ``image_jobs.json`` and the source ZIP are preserved;
- Stage 5 and later stages are never started.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Allow ``python tools/colab_handoff.py`` (script) as well as
# ``import tools.colab_handoff`` (tests): ensure the repository root is importable.
if __package__ in (None, ""):  # pragma: no cover - exercised by the script entrypoint
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from agents.image_jobs.handoff import (
    DONE_MARKER,
    HANDOFF_ROOT_NAME,
    IMAGE_JOBS_FILENAME,
    INPUT_MANIFEST_FILENAME,
    OUTPUT_MANIFEST_FILENAME,
    READY_MARKER,
    SAFE_ASSET_ID_RE,
    STAGES,
    STATUS_FILENAME,
    HandoffError,
    begin_attempt,
    build_input_manifest,
    build_status,
    dump_json,
    load_image_jobs,
    load_json_object,
    new_attempt_id,
    read_marker,
    sha256_file,
    verify_done,
    verify_handoff_pair,
    verify_input_manifest,
    verify_ready,
    verify_zip,
    verify_zip_result_document,
    write_marker,
)
from agents.image_jobs.import_assets import (
    DEFAULT_IMAGE_SIZE,
    AssetImportError,
    import_generated_assets,
)
from tools.handoff_transport import (
    STAGING_DIR_NAME,
    LocalDirectoryTransport,
    RcloneTransport,
    RemoteMirrorBackend,
    RemoteTransport,
    RootDiscoveryError,
    TransportError,
    preflight_remote,
)

EXIT_OK = 0
EXIT_FAILURE = 2


class HandoffCLIError(Exception):
    """A CLI-level failure with an optional informational code."""

    def __init__(self, message: str, code: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


def _fail(message: str, code: str = "") -> None:
    raise HandoffCLIError(message, code)


def _validate_run_id(run_id: object) -> str:
    """Return ``run_id`` if it is a safe bare directory name, else fail ``hf08``.

    Every subcommand funnels through here *before* any path is constructed, so a
    run id can never escape ""<handoff-root>/<run_id>/"" (no separators, no
    traversal, no absolute or drive-qualified paths, no deceptively-whitespaced
    or empty names). The accepted shape is exactly that of a safe asset id
    (``SAFE_ASSET_ID_RE``), which is the same bare-name rule the importer and the
    contract already enforce for asset ids.
    """
    if not isinstance(run_id, str):
        _fail(f"invalid run id {run_id!r}: must be a string", "hf08")
    name = run_id.strip()
    if not name or name != run_id:
        _fail(
            f"invalid run id {run_id!r}: must be a non-empty bare directory name "
            "with no surrounding whitespace",
            "hf08",
        )
    if name in (".", "..") or "/" in name or "\\" in name:
        _fail(
            f"invalid run id {run_id!r}: must not contain path separators or '.'/'..'",
            "hf08",
        )
    if not SAFE_ASSET_ID_RE.match(name):
        _fail(
            f"invalid run id {run_id!r}: must match {SAFE_ASSET_ID_RE.pattern}",
            "hf08",
        )
    path = Path(name)
    if path.is_absolute() or path.drive or path.name != name:
        _fail(f"invalid run id {run_id!r}: must be a bare directory name", "hf08")
    return name


# --- backend: a local directory that simulates the shared Drive folder --------


class LocalHandoffBackend:
    """Offline backend over a local directory shaped like the Drive root.

    This is the only place that knows *where* the handoff lives. A future
    rclone/Drive backend can implement the same path accessors without touching
    the manifest/marker/ZIP logic.
    """

    def __init__(self, root: str | Path, project_root: str | Path = ".") -> None:
        self.root = Path(root)
        self.project_root = Path(project_root)

    def ensure_root(self) -> Path:
        if self.root.exists() and not self.root.is_dir():
            _fail(f"handoff root is not a directory: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def input_dir(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "input"

    def status_dir(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "status"

    def output_dir(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "output"

    def source_jobs_path(self, run_id: str) -> Path:
        return self.project_root / "data" / run_id / IMAGE_JOBS_FILENAME

    def staging_dir(self, run_id: str, override: str | Path | None = None) -> Path:
        if override is not None:
            return Path(override)
        return self.project_root / STAGING_DIR_NAME / run_id


# Either backend exposes the same path accessors; the commands are backend
# agnostic (the offline local backend, or the mirror backend over a remote).
HandoffBackend = LocalHandoffBackend | RemoteMirrorBackend


# --- shared helpers -----------------------------------------------------------


def _print(message: str = "") -> None:
    print(message)


def _status_line(label: str, value: object) -> None:
    _print(f"{label}={value}")


def _load_input_manifest(backend: HandoffBackend, run_id: str) -> dict:
    path = backend.input_dir(run_id) / INPUT_MANIFEST_FILENAME
    if not path.is_file():
        _fail(f"input manifest not found: {path} (run 'prepare' first)", "hf21")
    return load_json_object(path, code="hf21")


def _verify_completed(backend: HandoffBackend, run_id: str) -> tuple[dict, dict]:
    """Verify the input READY + output DONE and the input<->output pair.

    Returns ``(input_manifest, output_manifest)``. Raises :class:`HandoffError`
    (or :class:`HandoffCLIError`) when the handoff is not complete/consistent.
    """
    input_dir = backend.input_dir(run_id)
    output_dir = backend.output_dir(run_id)

    input_manifest = verify_ready(input_dir, run_id=run_id)
    expected_ids = list(input_manifest.get("expected_asset_ids") or [])
    if not expected_ids:
        _fail("input manifest has no expected_asset_ids", "hf10")

    output_manifest = verify_done(
        output_dir,
        run_id=run_id,
        attempt_id=input_manifest.get("attempt_id"),
        image_jobs_sha256=input_manifest.get("image_jobs_sha256"),
        expected_asset_ids=expected_ids,
    )
    verify_handoff_pair(
        input_manifest,
        output_manifest,
        image_jobs_path=input_dir / IMAGE_JOBS_FILENAME,
        expected_run_id=run_id,
    )
    return input_manifest, output_manifest


def _resolve_zip(
    backend: HandoffBackend,
    run_id: str,
    explicit: str | Path | None,
    staging_override: str | Path | None,
) -> Path:
    """Locate the ZIP to validate/import: explicit, staged, else run output."""
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            _fail(f"ZIP not found: {path}", "hf30")
        return path

    staging = backend.staging_dir(run_id, staging_override)
    if staging.is_dir():
        staged = sorted(staging.glob("*.zip"))
        if len(staged) == 1:
            return staged[0]
        if len(staged) > 1:
            _fail(f"multiple staged ZIPs in {staging}; pass --zip explicitly")

    output_zips = sorted(backend.output_dir(run_id).glob("*.zip"))
    if len(output_zips) == 1:
        return output_zips[0]
    if len(output_zips) > 1:
        _fail(f"multiple ZIPs in {backend.output_dir(run_id)}; pass --zip explicitly")

    _fail(
        f"no ZIP found for run {run_id!r}; run 'fetch' first or pass --zip",
        "hf30",
    )


def _resolve_verified_zip(
    backend: HandoffBackend, run_id: str, args: argparse.Namespace
) -> tuple[Path, dict]:
    """Verify completion, then resolve the completed attempt's exact ZIP.

    ``validate`` and ``import`` both go through here so neither can ever act on
    an arbitrary ZIP that merely happens to live under ``output/`` or in the
    staging directory. The resolved ZIP (explicit ``--zip``, staged copy, or
    the run's output ZIP) must be the artifact the completed output manifest
    describes: same SHA-256 and byte size, and its in-ZIP ``result.json`` must
    satisfy the ``image-generation-results/1.0`` contract for this run/attempt.
    Returns ``(zip_path, output_manifest)``.
    """
    _, output_manifest = _verify_completed(backend, run_id)
    zip_path = _resolve_zip(backend, run_id, args.zip, args.staging_dir)
    zip_meta = output_manifest["zip"]
    verify_zip(
        zip_path,
        expected_sha256=zip_meta["sha256"],
        expected_bytes=zip_meta["bytes"],
    )
    verify_zip_result_document(
        zip_path, output_manifest=output_manifest, run_id=run_id
    )
    return zip_path, output_manifest


def _print_import_report(report) -> None:
    for issue in report.warnings:
        _print(f"[colab_handoff] warning [{issue.code}] {issue.message}")
    for issue in report.errors:
        _print(f"[colab_handoff] error [{issue.code}] {issue.message}")
    counts: dict[str, int] = {}
    for asset in report.assets:
        counts[asset["status"]] = counts.get(asset["status"], 0) + 1
    _status_line("import_status", report.status)
    _status_line("expected", len(report.assets))
    for key in ("imported", "overwritten", "validated", "missing", "duplicate",
                "corrupt", "invalid_dimension", "conflict", "unsafe"):
        if counts.get(key):
            _status_line(key, counts[key])
    if report.unexpected:
        _status_line("unexpected", len(report.unexpected))
    if report.report_path:
        _status_line("report", report.report_path)


# --- subcommands --------------------------------------------------------------


def cmd_prepare(backend: HandoffBackend, args: argparse.Namespace) -> int:
    run_id = args.run_id
    source_jobs = backend.source_jobs_path(run_id)
    if not source_jobs.is_file():
        _fail(f"image-job JSON not found: {source_jobs}", "hf01")

    # Validate the source contract and confirm its run_id matches the request.
    data = load_image_jobs(source_jobs)
    contract_run = str(data.get("run_id", "")).strip()
    if contract_run and contract_run != run_id:
        _fail(
            f"image-jobs run_id {contract_run!r} does not match requested run "
            f"{run_id!r}",
            "hf04",
        )

    backend.ensure_root()
    input_dir = backend.input_dir(run_id)
    output_dir = backend.output_dir(run_id)
    status_dir = backend.status_dir(run_id)

    already_prepared = read_marker(input_dir, READY_MARKER) is not None
    if already_prepared and not args.force:
        _fail(
            f"run {run_id!r} is already prepared ({READY_MARKER} exists); "
            "re-run with --force to start a new attempt",
        )

    attempt_id = new_attempt_id(run_id)

    # A fresh attempt must not inherit an earlier attempt's completion markers.
    begin_attempt(output_dir, attempt_id)

    input_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preserve the source job contract: copy, never move.
    dest_jobs = input_dir / IMAGE_JOBS_FILENAME
    shutil.copy2(source_jobs, dest_jobs)

    manifest = build_input_manifest(run_id, dest_jobs, attempt_id=attempt_id)
    dump_json(manifest, input_dir / INPUT_MANIFEST_FILENAME)

    # Verify the copied input files and hashes *before* signalling READY.
    verify_input_manifest(
        manifest,
        image_jobs_path=dest_jobs,
        expected_run_id=run_id,
        expected_attempt_id=attempt_id,
    )
    write_marker(input_dir, READY_MARKER, content=attempt_id)

    status = build_status(
        run_id,
        attempt_id=attempt_id,
        image_jobs_sha256=manifest["image_jobs_sha256"],
        overall_status="PENDING",
        current_stage=STAGES[0],
    )
    dump_json(status, status_dir / STATUS_FILENAME)

    _status_line("run_id", run_id)
    _status_line("handoff_dir", backend.run_dir(run_id))
    _status_line("attempt_id", attempt_id)
    _status_line("image_jobs_sha256", manifest["image_jobs_sha256"])
    _status_line("expected_asset_ids", ",".join(manifest["expected_asset_ids"]))
    _print("[colab_handoff] prepared; READY written last")
    return EXIT_OK


def _load_for_report(path: Path, *, code: str) -> tuple[dict | None, str]:
    """Load a JSON object for diagnostics; return ``(obj, error_text)``.

    Never raises: a malformed/unreadable document is reported alongside the rest
    of the status instead of aborting the report halfway through.
    """
    try:
        return load_json_object(path, code=code), ""
    except HandoffError as exc:
        return None, f"[{exc.code}] {exc.message}"


def cmd_status(backend: HandoffBackend, args: argparse.Namespace) -> int:
    """Print a complete, self-consistent status report for one run.

    The report cross-checks the content-bearing markers against the documents
    they refer to (``READY`` vs the input manifest ``attempt_id``; ``DONE`` vs
    the output manifest and output ``status.json`` ``attempt_id``; progress vs
    input attempt). A missing run is a hard error (exit 2). A malformed document
    or any detected marker/attempt inconsistency is reported explicitly and
    makes the command exit 2; the report is still printed in full. Absent
    markers/documents (e.g. a not-yet-started output) are normal and exit 0.
    """
    run_id = args.run_id
    run_dir = backend.run_dir(run_id)
    if not run_dir.is_dir():
        _fail(f"run {run_id!r} not found in handoff root {backend.root}", "hf01")

    input_dir = backend.input_dir(run_id)
    output_dir = backend.output_dir(run_id)
    problems: list[str] = []

    _status_line("run_id", run_id)
    _status_line("handoff_dir", run_dir)

    # --- input: manifest + READY marker (must agree on the attempt id) --------
    input_manifest_path = input_dir / INPUT_MANIFEST_FILENAME
    input_manifest: dict | None = None
    if input_manifest_path.is_file():
        input_manifest, err = _load_for_report(input_manifest_path, code="hf21")
        if err:
            _status_line("input_manifest", "malformed")
            _status_line("input_manifest_error", err)
            problems.append(f"input manifest: {err}")
    else:
        _status_line("input_manifest", "missing")

    ready_content = read_marker(input_dir, READY_MARKER)
    if input_manifest is not None:
        attempt = input_manifest.get("attempt_id")
        _status_line("input_attempt_id", attempt)
        if ready_content is None:
            _status_line("input_ready", "no")
        elif not attempt or ready_content != attempt:
            _status_line("input_ready", "mismatch")
            _status_line("input_ready_content", ready_content)
            problems.append(
                f"input READY content {ready_content!r} does not match the input "
                f"manifest attempt_id {attempt!r}"
            )
        else:
            _status_line("input_ready", "yes")
    elif ready_content is not None:
        _status_line("input_ready", "unknown")
        _status_line("input_ready_content", ready_content)

    # --- progress document (/status/status.json) ------------------------------
    progress_path = backend.status_dir(run_id) / STATUS_FILENAME
    progress: dict | None = None
    if progress_path.is_file():
        progress, err = _load_for_report(progress_path, code="hf41")
        if err:
            _status_line("progress_status", "malformed")
            _status_line("progress_error", err)
            problems.append(f"progress status: {err}")
        else:
            _status_line("progress_status", progress.get("overall_status"))
            _status_line("current_stage", progress.get("current_stage"))
            progress_error = progress.get("error") or {}
            if progress_error:
                _status_line("progress_error_code", progress_error.get("code"))
                _status_line("progress_error_stage", progress_error.get("stage"))
            if input_manifest is not None and (
                progress.get("attempt_id") != input_manifest.get("attempt_id")
            ):
                _status_line("progress_attempt_matches_input", "no")
                problems.append(
                    f"progress status attempt_id {progress.get('attempt_id')!r} does "
                    f"not match the input manifest attempt_id "
                    f"{input_manifest.get('attempt_id')!r}"
                )
    else:
        _status_line("progress_status", "none")

    # --- output: manifest + final status + DONE marker (all must agree) -------
    output_manifest_path = output_dir / OUTPUT_MANIFEST_FILENAME
    output_manifest: dict | None = None
    if output_manifest_path.is_file():
        output_manifest, err = _load_for_report(output_manifest_path, code="hf23")
        if err:
            _status_line("output_manifest", "malformed")
            _status_line("output_manifest_error", err)
            problems.append(f"output manifest: {err}")
    else:
        _status_line("output_manifest", "missing")

    final_path = output_dir / STATUS_FILENAME
    final: dict | None = None
    if final_path.is_file():
        final, err = _load_for_report(final_path, code="hf41")
        if err:
            _status_line("output_status", "malformed")
            _status_line("output_status_error", err)
            problems.append(f"output status: {err}")
        else:
            _status_line("output_status", final.get("overall_status"))
            error = final.get("error") or {}
            if error:
                _status_line("error_code", error.get("code"))
                _status_line("error_stage", error.get("stage"))
    else:
        _status_line("output_status", "none")

    done_content = read_marker(output_dir, DONE_MARKER)
    if done_content is None:
        _status_line("output_done", "no")
    else:
        _status_line("output_done", "yes")
        _status_line("output_attempt_id", done_content)
        refs: list[tuple[str, object]] = []
        if output_manifest is not None:
            refs.append(("output manifest", output_manifest.get("attempt_id")))
        if final is not None:
            refs.append(("output status", final.get("attempt_id")))
        mismatched = [
            (name, value)
            for name, value in refs
            if value is not None and value != done_content
        ]
        if mismatched:
            _status_line("output_done_matches", "no")
            details = ", ".join(f"{name} attempt_id {value!r}" for name, value in mismatched)
            problems.append(
                f"output DONE content {done_content!r} does not match {details}"
            )
        else:
            _status_line("output_done_matches", "yes" if refs else "unknown")
        if (
            output_manifest is not None
            and final is not None
            and output_manifest.get("attempt_id") != final.get("attempt_id")
        ):
            _status_line("output_attempts_consistent", "no")
            problems.append(
                "output manifest attempt_id "
                f"{output_manifest.get('attempt_id')!r} does not match output status "
                f"attempt_id {final.get('attempt_id')!r}"
            )

    if problems:
        for problem in problems:
            _print(f"[colab_handoff] status problem: {problem}")
        return EXIT_FAILURE
    return EXIT_OK


def cmd_fetch(backend: HandoffBackend, args: argparse.Namespace) -> int:
    run_id = args.run_id
    input_manifest, output_manifest = _verify_completed(backend, run_id)

    zip_name = output_manifest["zip"]["filename"]
    source_zip = backend.output_dir(run_id) / zip_name
    staged_dir = backend.staging_dir(run_id, args.staging_dir)
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged_zip = staged_dir / zip_name

    # Stage a copy, then re-verify the copy's fingerprint against the manifest.
    shutil.copy2(source_zip, staged_zip)
    expected_sha = output_manifest["zip"]["sha256"]
    if sha256_file(staged_zip) != expected_sha:
        _fail(f"staged ZIP hash mismatch: {staged_zip}", "hf32")

    _status_line("run_id", run_id)
    _status_line("attempt_id", input_manifest.get("attempt_id"))
    _status_line("output_manifest", backend.output_dir(run_id) / OUTPUT_MANIFEST_FILENAME)
    _status_line("staged_zip", staged_zip)
    _status_line("zip_sha256", expected_sha)
    _status_line("expected_asset_ids", ",".join(input_manifest.get("expected_asset_ids") or []))
    _print("[colab_handoff] fetch verified; ZIP staged (not extracted)")
    return EXIT_OK


def cmd_validate(backend: HandoffBackend, args: argparse.Namespace) -> int:
    """Importer dry-run against the *completed* attempt's verified artifact.

    Non-destructive: nothing is written or imported. Before the dry-run the
    run's completion is verified (READY/DONE markers, attempt binding, PASS
    status, output manifest, pair cross-binding) and the resolved ZIP must be
    the exact artifact the output manifest pins by hash — so an arbitrary or
    stale ZIP under ``output/`` can never receive a successful validation.
    """
    run_id = args.run_id
    zip_path, _ = _resolve_verified_zip(backend, run_id, args)
    report = import_generated_assets(
        zip_path,
        run_id,
        project_root=backend.project_root,
        expected_size=DEFAULT_IMAGE_SIZE,
        overwrite=False,
        dry_run=True,
    )
    _status_line("run_id", run_id)
    _status_line("zip", zip_path)
    _print_import_report(report)
    if report.ok:
        _print("[colab_handoff] validate OK (no files written)")
        return EXIT_OK
    _print("[colab_handoff] validate FAILED (no files written)")
    return EXIT_FAILURE


def cmd_import(backend: HandoffBackend, args: argparse.Namespace) -> int:
    run_id = args.run_id
    if not args.authorize_import:
        _fail(
            "refusing to import without --authorize-import "
            "(this writes into data/<run_id>/assets/)"
        )

    # Never import an unverified artifact: re-check completion, the pair, the
    # ZIP hash and the in-ZIP result document, all against the same path that
    # is then handed to the importer (no time-of-check/time-of-use gap).
    zip_path, _ = _resolve_verified_zip(backend, run_id, args)

    report = import_generated_assets(
        zip_path,
        run_id,
        project_root=backend.project_root,
        expected_size=DEFAULT_IMAGE_SIZE,
        overwrite=args.overwrite,
        dry_run=False,
    )
    _status_line("run_id", run_id)
    _status_line("zip", zip_path)
    _print_import_report(report)
    if report.ok:
        _print(
            "[colab_handoff] import complete; Stage 5 is NOT started "
            "automatically (use a new run id for Stage 5)"
        )
        return EXIT_OK
    _print("[colab_handoff] import FAILED (existing files are not overwritten by default)")
    return EXIT_FAILURE


# --- argument parsing ---------------------------------------------------------


def _add_common(parser: argparse.ArgumentParser, *, suppress_default: bool = False) -> None:
    project_default = argparse.SUPPRESS if suppress_default else Path(".")
    handoff_default = argparse.SUPPRESS if suppress_default else Path(HANDOFF_ROOT_NAME)
    remote_default = argparse.SUPPRESS if suppress_default else None
    create_default = argparse.SUPPRESS if suppress_default else False
    sentinel_default = argparse.SUPPRESS if suppress_default else None
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_default,
        help="project root containing data/ (default: current directory)",
    )
    parser.add_argument(
        "--handoff-root",
        type=Path,
        default=handoff_default,
        help=(
            "local directory simulating the shared Drive root "
            f"(default: {HANDOFF_ROOT_NAME}); ignored with --remote"
        ),
    )
    parser.add_argument(
        "--remote",
        default=remote_default,
        metavar="SPEC",
        help=(
            "remote handoff base, e.g. local:<dir> (offline directory transport). "
            "The handoff root is discovered inside it; default uses --handoff-root"
        ),
    )
    parser.add_argument(
        "--create-root",
        action="store_true",
        default=create_default,
        help="create the handoff root if missing (only meaningful for prepare with --remote)",
    )
    parser.add_argument(
        "--sentinel-value",
        default=sentinel_default,
        metavar="VALUE",
        help=(
            "trusted sentinel verification value from your local handoff notes; "
            "required for --remote rclone:<remote>: to prove the remote account/root"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    _add_common(common)
    # Subparsers use suppressed defaults so the shared options may be given
    # either before the subcommand (global position) or after it.
    common_optional = argparse.ArgumentParser(add_help=False)
    _add_common(common_optional, suppress_default=True)

    parser = argparse.ArgumentParser(
        prog="python tools/colab_handoff.py",
        parents=[common],
        description=(
            "Local (offline) handoff CLI for the manual Colab SDXL -> Drive "
            "workflow. Operates on a local directory; no Drive, no rclone."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", parents=[common_optional], help="build the input manifest + READY")
    p_prepare.add_argument("--run-id", required=True)
    p_prepare.add_argument(
        "--force",
        action="store_true",
        help="re-prepare an already-prepared run as a new attempt",
    )
    p_prepare.set_defaults(func=cmd_prepare)

    p_status = sub.add_parser("status", parents=[common_optional], help="show run status and attempt id")
    p_status.add_argument("--run-id", required=True)
    p_status.set_defaults(func=cmd_status)

    p_fetch = sub.add_parser("fetch", parents=[common_optional], help="verify completion and stage the ZIP")
    p_fetch.add_argument("--run-id", required=True)
    p_fetch.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="local staging directory (default: <project-root>/handoff_staging/<run_id>)",
    )
    p_fetch.set_defaults(func=cmd_fetch)

    p_validate = sub.add_parser("validate", parents=[common_optional], help="importer dry-run (non-destructive)")
    p_validate.add_argument("--run-id", required=True)
    p_validate.add_argument("--zip", type=Path, default=None, help="explicit ZIP to validate")
    p_validate.add_argument("--staging-dir", type=Path, default=None)
    p_validate.set_defaults(func=cmd_validate)

    p_import = sub.add_parser("import", parents=[common_optional], help="import assets (requires --authorize-import)")
    p_import.add_argument("--run-id", required=True)
    p_import.add_argument("--zip", type=Path, default=None, help="explicit ZIP to import")
    p_import.add_argument("--staging-dir", type=Path, default=None)
    p_import.add_argument(
        "--authorize-import",
        action="store_true",
        help="explicitly authorize writing images into data/<run_id>/assets/",
    )
    p_import.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing asset files (off by default)",
    )
    p_import.set_defaults(func=cmd_import)

    return parser


def _make_transport(spec: str) -> RemoteTransport:
    """Build a transport from a ``--remote`` spec.

    Supported specs:

    * ``local:<dir>`` — offline directory transport (no network, no rclone);
    * ``rclone:<remote>`` — rclone-backed transport, where ``<remote>`` is an
      rclone remote ending in ``:`` (e.g. ``rclone:gdrive:``). The handoff root
      is discovered inside ``<remote>``; ``--sentinel-value`` is then required.
    """
    if spec.startswith("local:"):
        base = spec[len("local:"):]
        if not base:
            _fail("--remote 'local:' requires a directory path", "hf09")
        return LocalDirectoryTransport(base)
    if spec.startswith("rclone:"):
        remote = spec[len("rclone:"):]
        if not remote or ":" not in remote:
            _fail(
                "--remote 'rclone:' requires an rclone remote ending in ':' "
                "(e.g. 'rclone:gdrive:')",
                "hf09",
            )
        try:
            return RcloneTransport(remote)
        except TransportError as exc:
            _fail(str(exc), "hf09")
    scheme = spec.split(":", 1)[0] or spec
    _fail(
        f"unsupported --remote scheme {scheme!r}: available schemes are "
        "'local:<dir>' and 'rclone:<remote>:'",
        "hf09",
    )


def _make_backend(args: argparse.Namespace) -> HandoffBackend:
    remote = getattr(args, "remote", None)
    if not remote:
        return LocalHandoffBackend(args.handoff_root, args.project_root)
    return RemoteMirrorBackend(
        _make_transport(remote),
        args.project_root,
        create_root=bool(getattr(args, "create_root", False)),
    )


# Commands that read the handoff root but never write to it. ``prepare`` is the
# only command that publishes back (marker last).
_REMOTE_READ_COMMANDS = {"status", "fetch", "validate", "import"}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    backend: HandoffBackend | None = None
    try:
        # Reject an unconfined run id before any subcommand builds a path.
        _validate_run_id(args.run_id)
        backend = _make_backend(args)

        # Fail-closed remote preflight: accessibility, exact root discovery and
        # sentinel trust, before *any* transfer. The root may be created only
        # for ``prepare`` and only with an explicit ``--create-root``; a root
        # that exists but lacks a matching sentinel stops here.
        transport = getattr(backend, "transport", None)
        if transport is not None and getattr(transport, "requires_sentinel", False):
            create = (
                bool(getattr(args, "create_root", False))
                and args.command == "prepare"
            )
            preflight_remote(
                transport,
                expected_sentinel_value=getattr(args, "sentinel_value", None),
                create_root=create,
            )

        if isinstance(backend, RemoteMirrorBackend):
            # Download the run folder into the mirror before reading it (and,
            # for prepare, so an existing READY is detected). Read-only: a
            # missing root leaves the mirror empty and the command reports it.
            if args.command in _REMOTE_READ_COMMANDS or args.command == "prepare":
                backend.begin_read(args.run_id)

        rc = args.func(backend, args)

        if (
            isinstance(backend, RemoteMirrorBackend)
            and args.command == "prepare"
            and rc == EXIT_OK
        ):
            backend.publish(args.run_id, upload_last=("input/READY",))
        return rc
    except HandoffCLIError as exc:
        prefix = f"[colab_handoff] error [{exc.code}]" if exc.code else "[colab_handoff] error:"
        print(f"{prefix} {exc.message}", file=sys.stderr)
        return EXIT_FAILURE
    except TransportError as exc:
        print(f"[colab_handoff] error: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    except HandoffError as exc:
        print(f"[colab_handoff] error [{exc.code}] {exc.message}", file=sys.stderr)
        return EXIT_FAILURE
    except AssetImportError as exc:
        print(f"[colab_handoff] input error [{exc.code}] {exc.message}", file=sys.stderr)
        return EXIT_FAILURE
    finally:
        if isinstance(backend, RemoteMirrorBackend):
            backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
