"""Offline tests for the Milestone 3B.2 rclone transport and remote preflight.

Everything here is offline: no network, no Google Drive, no real ``rclone``
process, no GPU, no OAuth. The rclone transport is driven through an **injected
fake runner** that interprets exactly the rclone subcommands the transport
issues (``lsf``, ``mkdir``, ``copy``, ``copyto``, ``sync``, ``cat``, ``about``)
and applies them to local temporary directories.

This lets us exercise, deterministically and offline: root discovery and
ambiguity, path traversal / run-id validation, upload/download, partial and
interrupted transfers, bounded retries for transient failures only, marker-last
ordering, fail-closed sentinel verification, no deletion of unrelated roots or
runs, and command failure/exit-code handling.
"""

from __future__ import annotations

import io
import json
import shutil
import zipfile
from pathlib import Path

import pytest
from PIL import Image

import tools.colab_handoff as cli
import tools.handoff_transport as transport_mod
from agents.image_jobs.handoff import (
    HANDOFF_ROOT_NAME,
    STAGES,
    build_output_manifest,
    build_status,
    dump_json,
    write_marker,
)

REMOTE = "gdrive:"
RUN = "rclone-run-01"
ASSETS = ["sc01_main"]
PNG_SIZE = (768, 1344)
SENTINEL_VALUE = "deadbeefcafebabe0123456789abcdef"


# ---------------------------------------------------------------------------
# Fake rclone CLI
# ---------------------------------------------------------------------------


class _RemoteError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class FakeRclone:
    """In-process stand-in for the rclone CLI, backed by local directories."""

    def __init__(self, remotes: dict[str, Path]) -> None:
        self.remotes = {name: Path(path) for name, path in remotes.items()}
        self.calls: list[list[str]] = []
        self.sleeps: list[float] = []
        self._injected: list[dict] = []
        # Listing can be overridden to simulate directory layouts that the host
        # filesystem cannot represent (e.g. two roots differing only by case).
        self.list_overrides: dict[str, list[str]] = {}

    def override_lsf(self, remote_path: str, lines: list[str]) -> None:
        self.list_overrides[remote_path] = list(lines)

    # -- failure injection ---------------------------------------------------

    def fail(
        self,
        command: str,
        *,
        times: int = 1,
        code: int = 5,
        stderr: str = "rate limit exceeded (429)",
        partial: bool = False,
    ) -> None:
        self._injected.append(
            {
                "command": command,
                "times": times,
                "code": code,
                "stderr": stderr,
                "partial": partial,
            }
        )

    def _next_failure(self, command: str) -> dict | None:
        for item in self._injected:
            if item["command"] == command and item["times"] > 0:
                item["times"] -= 1
                return item
        return None

    # -- translation ---------------------------------------------------------

    def _local(self, remote_path: str) -> Path:
        name, sep, rest = remote_path.partition(":")
        key = f"{name}:"
        if sep and key in self.remotes:
            base = self.remotes[key]
            rest = rest.lstrip("/")
            return base / rest if rest else base
        # Not a configured remote: a plain local path. Windows drive letters
        # contain a colon, so e.g. ``C:\\tmp`` must not be read as remote ``C``.
        return Path(remote_path)

    # -- entry point ---------------------------------------------------------

    def __call__(self, argv):
        argv = list(argv)
        self.calls.append(argv)
        command = argv[1]
        failure = self._next_failure(command)
        if failure is not None:
            if failure["partial"]:
                self._do_partial(argv)
            return transport_mod.CommandResult(
                argv, failure["code"], "", failure["stderr"]
            )
        handler = getattr(self, f"_cmd_{command}", None)
        if handler is None:
            return transport_mod.CommandResult(argv, 2, "", f"unknown fake command {command}")
        try:
            return handler(argv)
        except _RemoteError as exc:
            return transport_mod.CommandResult(argv, exc.code, "", exc.message)

    def _do_partial(self, argv) -> None:
        if argv[1] in ("copy", "sync"):
            src = self._local(argv[2])
            dst = self._local(argv[3])
            for path in sorted(src.rglob("*")):
                if path.is_file():
                    target = dst / path.relative_to(src)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                    break

    # -- subcommands ---------------------------------------------------------

    def _cmd_about(self, argv):
        return transport_mod.CommandResult(argv, 0, "Total: 15 GiB\n", "")

    def _cmd_lsf(self, argv):
        if argv[2] in self.list_overrides:
            lines = self.list_overrides[argv[2]]
            return transport_mod.CommandResult(
                argv, 0, "\n".join(lines) + ("\n" if lines else ""), ""
            )
        path = self._local(argv[2])
        dirs_only = "--dirs-only" in argv
        if not path.is_dir():
            return transport_mod.CommandResult(argv, 3, "", "directory not found")
        lines: list[str] = []
        for child in sorted(path.iterdir()):
            if child.is_dir():
                lines.append(child.name + "/")
            elif not dirs_only:
                lines.append(child.name)
        return transport_mod.CommandResult(
            argv, 0, "\n".join(lines) + ("\n" if lines else ""), ""
        )

    def _cmd_mkdir(self, argv):
        self._local(argv[2]).mkdir(parents=True, exist_ok=True)
        return transport_mod.CommandResult(argv, 0, "", "")

    def _cmd_cat(self, argv):
        path = self._local(argv[2])
        if not path.is_file():
            return transport_mod.CommandResult(argv, 4, "", "file not found")
        return transport_mod.CommandResult(argv, 0, path.read_text(encoding="utf-8"), "")

    def _cmd_copy(self, argv):
        src = self._local(argv[2])
        dst = self._local(argv[3])
        if not src.is_dir():
            return transport_mod.CommandResult(argv, 3, "", "directory not found")
        for path in sorted(src.rglob("*")):
            if path.is_file():
                target = dst / path.relative_to(src)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        return transport_mod.CommandResult(argv, 0, "", "")

    def _cmd_copyto(self, argv):
        src = self._local(argv[2])
        dst = self._local(argv[3])
        if not src.is_file():
            return transport_mod.CommandResult(argv, 4, "", "file not found")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return transport_mod.CommandResult(argv, 0, "", "")

    def _cmd_sync(self, argv):
        src = self._local(argv[2])
        dst = self._local(argv[3])
        excludes = [argv[i + 1] for i, a in enumerate(argv) if a == "--exclude"]

        def excluded(rel: str) -> bool:
            for pattern in excludes:
                stripped = pattern.lstrip("/").rstrip("/")
                if rel == stripped or rel.startswith(stripped + "/"):
                    return True
            return False

        if dst.is_dir():
            for path in sorted(dst.rglob("*"), reverse=True):
                if path.is_file():
                    rel = path.relative_to(dst).as_posix()
                    if excluded(rel):
                        continue
                    if not (src / rel).is_file():
                        path.unlink()
        for path in sorted(src.rglob("*")):
            if path.is_file():
                rel = path.relative_to(src).as_posix()
                if excluded(rel):
                    continue
                target = dst / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        return transport_mod.CommandResult(argv, 0, "", "")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _transport(fake: FakeRclone, **kwargs) -> transport_mod.RcloneTransport:
    kwargs.setdefault("sleep", fake.sleeps.append)
    return transport_mod.RcloneTransport(REMOTE, runner=fake, **kwargs)


def _write_sentinel(
    base: Path,
    value: str = SENTINEL_VALUE,
    *,
    root_name: str = HANDOFF_ROOT_NAME,
    contents: str | None = None,
) -> Path:
    root = base / root_name
    root.mkdir(parents=True, exist_ok=True)
    sentinel = root / transport_mod.SENTINEL_FILENAME
    if contents is None:
        doc = {
            "project": transport_mod.HANDOFF_PROJECT_ID,
            "root": root_name,
            "sentinel_version": transport_mod.SENTINEL_SCHEMA_VERSION,
            "verification_value": value,
            "created_utc": "2026-10-14T00:00:00Z",
        }
        contents = json.dumps(doc)
    sentinel.write_text(contents, encoding="utf-8")
    return sentinel


def _write_jobs(project_root: Path, run_id: str = RUN) -> Path:
    jobs = []
    for i, asset in enumerate(ASSETS, start=1):
        jobs.append(
            {
                "scene_id": f"sc{i:02d}",
                "positive_prompt": "style, a thing",
                "negative_prompt": "no",
                "seed": 100 + i,
                "asset_id": asset,
                "output_path": f"data/{run_id}/assets/{asset}.png",
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


def _make_zip(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w") as zf:
        for asset in ASSETS:
            zf.writestr(f"{asset}.png", _png_bytes())
        zf.writestr("result.json", json.dumps({"schema": "image-generation-results/1.0"}))
    return dest


def _run_dir(base: Path, run_id: str = RUN) -> Path:
    return base / HANDOFF_ROOT_NAME / run_id


def _read_input_manifest(base: Path, run_id: str = RUN) -> dict:
    path = _run_dir(base, run_id) / "input" / "input_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _publish_output(base: Path, run_id: str, *, zip_path: Path) -> None:
    manifest = _read_input_manifest(base, run_id)
    out_dir = _run_dir(base, run_id) / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = build_output_manifest(
        run_id,
        attempt_id=manifest["attempt_id"],
        image_jobs_sha256=manifest["image_jobs_sha256"],
        zip_path=zip_path,
        members=[f"{a}.png" for a in ASSETS] + ["result.json"],
        generated_asset_ids=list(ASSETS),
    )
    dump_json(output_manifest, out_dir / "output_manifest.json")
    status = build_status(
        run_id,
        attempt_id=manifest["attempt_id"],
        image_jobs_sha256=manifest["image_jobs_sha256"],
        overall_status="PASS",
        current_stage=STAGES[-1],
        completed_stages=list(STAGES),
        completed_asset_ids=list(ASSETS),
    )
    dump_json(status, out_dir / "status.json")
    write_marker(out_dir, "DONE", content=manifest["attempt_id"])


def _patch_transport(monkeypatch, fake: FakeRclone) -> None:
    monkeypatch.setattr(
        cli, "_make_transport", lambda spec: _transport(fake)
    )


# ---------------------------------------------------------------------------
# path traversal / construction
# ---------------------------------------------------------------------------


def test_transport_requires_a_remote_name(tmp_path):
    with pytest.raises(transport_mod.TransportError):
        transport_mod.RcloneTransport("gdrive")  # no trailing colon


def test_constructing_transport_runs_nothing(tmp_path):
    fake = FakeRclone({REMOTE: tmp_path})
    _transport(fake)
    assert fake.calls == []  # no subprocess at construction time


@pytest.mark.parametrize("bad", ["../evil", "a/../../b", "C:/evil", "/abs", "sub/-flag"])
def test_unsafe_relative_paths_are_rejected(tmp_path, bad):
    fake = FakeRclone({REMOTE: tmp_path})
    transport = _transport(fake)
    with pytest.raises(transport_mod.TransportError):
        transport.upload_tree(bad, tmp_path)
    with pytest.raises(transport_mod.TransportError):
        transport.download_tree(bad, tmp_path / "dest")
    assert fake.calls == []  # rejected before any command ran


# ---------------------------------------------------------------------------
# root discovery
# ---------------------------------------------------------------------------


def test_root_discovery_reuses_single(tmp_path):
    _write_sentinel(tmp_path)
    fake = FakeRclone({REMOTE: tmp_path})
    transport = _transport(fake)
    assert transport_mod.discover_root(transport, create=False) == HANDOFF_ROOT_NAME


def test_root_discovery_missing_without_create_raises(tmp_path):
    fake = FakeRclone({REMOTE: tmp_path})
    transport = _transport(fake)
    with pytest.raises(transport_mod.RootDiscoveryError, match="not found"):
        transport_mod.discover_root(transport, create=False)


def test_root_discovery_creates_once(tmp_path):
    fake = FakeRclone({REMOTE: tmp_path})
    transport = _transport(fake)
    assert transport_mod.discover_root(transport, create=True) == HANDOFF_ROOT_NAME
    assert (tmp_path / HANDOFF_ROOT_NAME).is_dir()
    # re-discovery reuses rather than creating a second root
    assert transport_mod.discover_root(transport, create=True) == HANDOFF_ROOT_NAME
    dirs = [p.name for p in tmp_path.iterdir() if p.is_dir()]
    assert dirs == [HANDOFF_ROOT_NAME]


def test_root_discovery_ambiguous_stops(tmp_path):
    # The host filesystem is case-insensitive on Windows, so the two case-variant
    # roots are simulated through the listing override instead of real dirs.
    fake = FakeRclone({REMOTE: tmp_path})
    fake.override_lsf(
        REMOTE, [f"{HANDOFF_ROOT_NAME}/", f"{HANDOFF_ROOT_NAME.upper()}/"]
    )
    transport = _transport(fake)
    with pytest.raises(transport_mod.RootDiscoveryError, match="ambiguous"):
        transport_mod.discover_root(transport, create=True)


def test_root_discovery_case_mismatch_stops(tmp_path):
    fake = FakeRclone({REMOTE: tmp_path})
    fake.override_lsf(REMOTE, [f"{HANDOFF_ROOT_NAME.upper()}/"])
    transport = _transport(fake)
    with pytest.raises(transport_mod.RootDiscoveryError, match="ambiguous"):
        transport_mod.discover_root(transport, create=False)


def test_root_discovery_file_is_not_a_directory(tmp_path):
    (tmp_path / HANDOFF_ROOT_NAME).write_text("not a dir", encoding="utf-8")
    fake = FakeRclone({REMOTE: tmp_path})
    transport = _transport(fake)
    with pytest.raises(transport_mod.RootDiscoveryError, match="not a directory"):
        transport_mod.discover_root(transport, create=False)


# ---------------------------------------------------------------------------
# upload / download
# ---------------------------------------------------------------------------


def _make_source(tmp_path: Path, files: dict[str, str]) -> Path:
    src = tmp_path / "src"
    for rel, text in files.items():
        path = src / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return src


def test_upload_then_download_roundtrip(tmp_path):
    base = tmp_path / "base"
    _write_sentinel(base)
    fake = FakeRclone({REMOTE: base})
    transport = _transport(fake)

    src = _make_source(tmp_path, {"input/image_jobs.json": "jobs", "status/status.json": "s"})
    transport.upload_tree(f"{HANDOFF_ROOT_NAME}/{RUN}", src)

    dest = tmp_path / "download"
    transport.download_tree(f"{HANDOFF_ROOT_NAME}/{RUN}", dest)
    assert (dest / "input" / "image_jobs.json").read_text(encoding="utf-8") == "jobs"
    assert (dest / "status" / "status.json").read_text(encoding="utf-8") == "s"


def test_upload_tree_marker_last_ordering(tmp_path):
    base = tmp_path / "base"
    _write_sentinel(base)
    fake = FakeRclone({REMOTE: base})
    transport = _transport(fake)

    src = _make_source(tmp_path, {"input/image_jobs.json": "jobs", "input/READY": "attempt-1"})
    transport.upload_tree(f"{HANDOFF_ROOT_NAME}/{RUN}", src, upload_last=("input/READY",))

    # READY is excluded from the payload sync ...
    sync_call = next(call for call in fake.calls if call[1] == "sync")
    assert "/input/READY" in sync_call
    # ... and published by a copyto that is the final command.
    write_commands = [c for c in fake.calls if c[1] in ("sync", "copy", "copyto")]
    assert write_commands[-1][1] == "copyto"
    assert write_commands[-1][-1].endswith("/input/READY")
    assert (_run_dir(base) / "input" / "READY").read_text(encoding="utf-8") == "attempt-1"


def test_failed_payload_transfer_publishes_no_marker(tmp_path):
    base = tmp_path / "base"
    _write_sentinel(base)
    fake = FakeRclone({REMOTE: base})
    fake.fail("sync", times=1, code=1, stderr="permission denied (403)")
    transport = _transport(fake)

    src = _make_source(tmp_path, {"input/image_jobs.json": "jobs", "input/READY": "attempt-1"})
    with pytest.raises(transport_mod.CommandError):
        transport.upload_tree(f"{HANDOFF_ROOT_NAME}/{RUN}", src, upload_last=("input/READY",))
    assert not (_run_dir(base) / "input" / "READY").exists()


def test_partial_transfer_leaves_no_marker(tmp_path):
    base = tmp_path / "base"
    _write_sentinel(base)
    fake = FakeRclone({REMOTE: base})
    # Fail transiently *after* copying one payload file, exhausting retries.
    fake.fail("sync", times=99, code=5, stderr="connection reset", partial=True)
    transport = _transport(fake, max_attempts=2)

    src = _make_source(
        tmp_path,
        {"input/image_jobs.json": "jobs", "input/input_manifest.json": "m", "input/READY": "attempt-1"},
    )
    with pytest.raises(transport_mod.CommandError):
        transport.upload_tree(f"{HANDOFF_ROOT_NAME}/{RUN}", src, upload_last=("input/READY",))
    # A partial payload may exist, but the marker must not.
    assert not (_run_dir(base) / "input" / "READY").exists()
    assert not any(c[1] == "copyto" for c in fake.calls)


# ---------------------------------------------------------------------------
# retries (transient only)
# ---------------------------------------------------------------------------


def test_transient_failure_is_retried_within_limit(tmp_path):
    base = tmp_path / "base"
    _write_sentinel(base)
    fake = FakeRclone({REMOTE: base})
    fake.fail("sync", times=2, code=5, stderr="rate limit exceeded (429)")
    transport = _transport(fake, max_attempts=3)

    src = _make_source(tmp_path, {"a.txt": "a"})
    transport.upload_tree(f"{HANDOFF_ROOT_NAME}/{RUN}", src)
    assert sum(1 for c in fake.calls if c[1] == "sync") == 3
    assert fake.sleeps  # backoff was applied


def test_transient_failure_exhausts_retry_limit(tmp_path):
    base = tmp_path / "base"
    _write_sentinel(base)
    fake = FakeRclone({REMOTE: base})
    fake.fail("sync", times=99, code=5, stderr="connection reset by peer")
    transport = _transport(fake, max_attempts=3)

    src = _make_source(tmp_path, {"a.txt": "a"})
    with pytest.raises(transport_mod.CommandError):
        transport.upload_tree(f"{HANDOFF_ROOT_NAME}/{RUN}", src)
    assert sum(1 for c in fake.calls if c[1] == "sync") == 3


def test_non_transient_failure_is_not_retried(tmp_path):
    base = tmp_path / "base"
    _write_sentinel(base)
    fake = FakeRclone({REMOTE: base})
    fake.fail("sync", times=99, code=1, stderr="permission denied (403)")
    transport = _transport(fake, max_attempts=3)

    src = _make_source(tmp_path, {"a.txt": "a"})
    with pytest.raises(transport_mod.CommandError):
        transport.upload_tree(f"{HANDOFF_ROOT_NAME}/{RUN}", src)
    assert sum(1 for c in fake.calls if c[1] == "sync") == 1


def test_auth_failure_is_not_retried(tmp_path):
    base = tmp_path / "base"
    _write_sentinel(base)
    fake = FakeRclone({REMOTE: base})
    fake.fail("sync", times=99, code=5, stderr="invalid_grant: token expired")
    transport = _transport(fake, max_attempts=3)

    src = _make_source(tmp_path, {"a.txt": "a"})
    with pytest.raises(transport_mod.CommandError):
        transport.upload_tree(f"{HANDOFF_ROOT_NAME}/{RUN}", src)
    assert sum(1 for c in fake.calls if c[1] == "sync") == 1


def test_command_exit_code_is_surfaced(tmp_path):
    base = tmp_path / "base"
    _write_sentinel(base)
    fake = FakeRclone({REMOTE: base})
    fake.fail("copy", times=1, code=2, stderr="fatal: something deterministic")
    transport = _transport(fake)
    with pytest.raises(transport_mod.CommandError) as excinfo:
        transport.download_tree(f"{HANDOFF_ROOT_NAME}/{RUN}", tmp_path / "dest")
    assert excinfo.value.returncode == 2


def test_check_access_failure_surfaces(tmp_path):
    base = tmp_path / "base"
    _write_sentinel(base)
    fake = FakeRclone({REMOTE: base})
    fake.fail("about", times=1, code=2, stderr="invalid_grant")
    transport = _transport(fake)
    with pytest.raises(transport_mod.CommandError):
        transport.check_access()


# ---------------------------------------------------------------------------
# sentinel / preflight
# ---------------------------------------------------------------------------


def test_preflight_requires_sentinel_value(tmp_path):
    base = tmp_path / "base"
    _write_sentinel(base)
    fake = FakeRclone({REMOTE: base})
    transport = _transport(fake)
    with pytest.raises(transport_mod.RemoteVerificationError, match="sentinel"):
        transport_mod.preflight_remote(transport, expected_sentinel_value=None)
    assert fake.calls == []  # never touched the remote without a trusted value


def test_preflight_ok(tmp_path):
    base = tmp_path / "base"
    _write_sentinel(base)
    fake = FakeRclone({REMOTE: base})
    transport = _transport(fake)
    assert (
        transport_mod.preflight_remote(
            transport, expected_sentinel_value=SENTINEL_VALUE
        )
        == HANDOFF_ROOT_NAME
    )


def test_preflight_sentinel_missing_fails_closed(tmp_path):
    base = tmp_path / "base"
    (base / HANDOFF_ROOT_NAME).mkdir(parents=True)  # root exists, no sentinel
    fake = FakeRclone({REMOTE: base})
    transport = _transport(fake)
    with pytest.raises(transport_mod.RemoteVerificationError, match="missing"):
        transport_mod.preflight_remote(transport, expected_sentinel_value=SENTINEL_VALUE)


def test_preflight_sentinel_mismatch_fails_closed(tmp_path):
    base = tmp_path / "base"
    _write_sentinel(base, value="a-different-value")
    fake = FakeRclone({REMOTE: base})
    transport = _transport(fake)
    with pytest.raises(transport_mod.RemoteVerificationError, match="does not match"):
        transport_mod.preflight_remote(transport, expected_sentinel_value=SENTINEL_VALUE)


def test_preflight_sentinel_malformed_fails_closed(tmp_path):
    base = tmp_path / "base"
    _write_sentinel(base, contents="{not json")
    fake = FakeRclone({REMOTE: base})
    transport = _transport(fake)
    with pytest.raises(transport_mod.RemoteVerificationError):
        transport_mod.preflight_remote(transport, expected_sentinel_value=SENTINEL_VALUE)


@pytest.mark.parametrize(
    "field,value",
    [("project", "other-project"), ("root", "other-root"), ("sentinel_version", 99)],
)
def test_preflight_sentinel_wrong_field_fails_closed(tmp_path, field, value):
    base = tmp_path / "base"
    _write_sentinel(base)
    sentinel = base / HANDOFF_ROOT_NAME / transport_mod.SENTINEL_FILENAME
    doc = json.loads(sentinel.read_text(encoding="utf-8"))
    doc[field] = value
    sentinel.write_text(json.dumps(doc), encoding="utf-8")
    fake = FakeRclone({REMOTE: base})
    transport = _transport(fake)
    with pytest.raises(transport_mod.RemoteVerificationError):
        transport_mod.preflight_remote(transport, expected_sentinel_value=SENTINEL_VALUE)


def test_preflight_creates_root_and_sentinel_when_authorized(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    fake = FakeRclone({REMOTE: base})
    transport = _transport(fake)
    root = transport_mod.preflight_remote(
        transport, expected_sentinel_value=SENTINEL_VALUE, create_root=True
    )
    assert root == HANDOFF_ROOT_NAME
    sentinel = base / HANDOFF_ROOT_NAME / transport_mod.SENTINEL_FILENAME
    assert sentinel.is_file()
    doc = json.loads(sentinel.read_text(encoding="utf-8"))
    assert doc["verification_value"] == SENTINEL_VALUE
    assert doc["project"] == transport_mod.HANDOFF_PROJECT_ID
    assert doc["root"] == HANDOFF_ROOT_NAME
    # exactly one root was created
    assert [p.name for p in base.iterdir() if p.is_dir()] == [HANDOFF_ROOT_NAME]
    # running it again verifies (does not recreate/overwrite)
    assert (
        transport_mod.preflight_remote(
            transport, expected_sentinel_value=SENTINEL_VALUE, create_root=True
        )
        == HANDOFF_ROOT_NAME
    )


def test_preflight_never_overwrites_an_existing_sentinel(tmp_path):
    base = tmp_path / "base"
    sentinel = _write_sentinel(base, value="original-value")
    original_bytes = sentinel.read_bytes()
    fake = FakeRclone({REMOTE: base})
    transport = _transport(fake)
    with pytest.raises(transport_mod.RemoteVerificationError, match="does not match"):
        transport_mod.preflight_remote(
            transport, expected_sentinel_value=SENTINEL_VALUE, create_root=True
        )
    assert sentinel.read_bytes() == original_bytes  # untouched


def test_preflight_missing_root_without_create_fails(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    fake = FakeRclone({REMOTE: base})
    transport = _transport(fake)
    with pytest.raises(transport_mod.RootDiscoveryError, match="not found"):
        transport_mod.preflight_remote(transport, expected_sentinel_value=SENTINEL_VALUE)


def test_preflight_ambiguity_fails(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    fake = FakeRclone({REMOTE: base})
    fake.override_lsf(
        REMOTE, [f"{HANDOFF_ROOT_NAME}/", f"{HANDOFF_ROOT_NAME.upper()}/"]
    )
    transport = _transport(fake)
    with pytest.raises(transport_mod.RootDiscoveryError, match="ambiguous"):
        transport_mod.preflight_remote(
            transport, expected_sentinel_value=SENTINEL_VALUE, create_root=True
        )


# ---------------------------------------------------------------------------
# non-destructive guarantees
# ---------------------------------------------------------------------------


def test_upload_does_not_delete_unrelated_roots_or_runs(tmp_path):
    base = tmp_path / "base"
    _write_sentinel(base)
    other_run = _run_dir(base, "old-run-existing")
    other_run.mkdir(parents=True)
    (other_run / "keep.txt").write_text("keep", encoding="utf-8")
    sibling = base / "finance_mascot"
    sibling.mkdir()
    (sibling / "mascot.png").write_bytes(b"mascot")

    fake = FakeRclone({REMOTE: base})
    transport = _transport(fake)
    src = _make_source(tmp_path, {"input/READY": "attempt", "input/a.txt": "a"})
    transport.upload_tree(f"{HANDOFF_ROOT_NAME}/{RUN}", src, upload_last=("input/READY",))

    assert (other_run / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert (sibling / "mascot.png").read_bytes() == b"mascot"
    assert (base / HANDOFF_ROOT_NAME).is_dir()


def test_upload_replaces_only_the_named_run(tmp_path):
    base = tmp_path / "base"
    _write_sentinel(base)
    stale = _run_dir(base, RUN) / "output"
    stale.mkdir(parents=True)
    (stale / "DONE").write_text("stale-attempt", encoding="utf-8")

    fake = FakeRclone({REMOTE: base})
    transport = _transport(fake)
    src = _make_source(tmp_path, {"input/READY": "new-attempt", "input/a.txt": "a"})
    transport.upload_tree(f"{HANDOFF_ROOT_NAME}/{RUN}", src, upload_last=("input/READY",))

    # the stale marker inside the republished run folder is gone (replace), and
    # the new marker is present
    assert not (_run_dir(base, RUN) / "output" / "DONE").exists()
    assert (_run_dir(base, RUN) / "input" / "READY").read_text(encoding="utf-8") == "new-attempt"


# ---------------------------------------------------------------------------
# CLI integration (offline, fake rclone)
# ---------------------------------------------------------------------------


def test_cli_rclone_prepare_publishes_ready_last(tmp_path, monkeypatch, capsys):
    project_root = tmp_path / "proj"
    base = tmp_path / "base"
    base.mkdir()
    _write_jobs(project_root)
    fake = FakeRclone({REMOTE: base})
    _patch_transport(monkeypatch, fake)

    rc = cli.main([
        "--project-root", str(project_root),
        "--remote", f"rclone:{REMOTE}",
        "--sentinel-value", SENTINEL_VALUE,
        "prepare", "--run-id", RUN, "--create-root",
    ])
    assert rc == 0, capsys.readouterr().err
    ready = _run_dir(base) / "input" / "READY"
    assert ready.is_file()
    assert ready.read_text(encoding="utf-8") == _read_input_manifest(base)["attempt_id"]
    assert (base / HANDOFF_ROOT_NAME / transport_mod.SENTINEL_FILENAME).is_file()
    assert [p.name for p in base.iterdir() if p.is_dir()] == [HANDOFF_ROOT_NAME]


def test_cli_rclone_requires_sentinel_value(tmp_path, monkeypatch, capsys):
    project_root = tmp_path / "proj"
    base = tmp_path / "base"
    base.mkdir()
    _write_jobs(project_root)
    fake = FakeRclone({REMOTE: base})
    _patch_transport(monkeypatch, fake)

    rc = cli.main([
        "--project-root", str(project_root),
        "--remote", f"rclone:{REMOTE}",
        "prepare", "--run-id", RUN, "--create-root",
    ])
    assert rc == 2
    assert "sentinel" in capsys.readouterr().err
    assert not (base / HANDOFF_ROOT_NAME).exists()


def test_cli_rclone_sentinel_mismatch_stops(tmp_path, monkeypatch, capsys):
    project_root = tmp_path / "proj"
    base = tmp_path / "base"
    _write_sentinel(base, value="the-real-value")
    _write_jobs(project_root)
    fake = FakeRclone({REMOTE: base})
    _patch_transport(monkeypatch, fake)

    rc = cli.main([
        "--project-root", str(project_root),
        "--remote", f"rclone:{REMOTE}",
        "--sentinel-value", "wrong-value",
        "prepare", "--run-id", RUN, "--create-root",
    ])
    assert rc == 2
    assert "does not match" in capsys.readouterr().err


def test_cli_rclone_run_id_traversal_rejected(tmp_path, monkeypatch, capsys):
    project_root = tmp_path / "proj"
    base = tmp_path / "base"
    _write_sentinel(base)
    _write_jobs(project_root)
    fake = FakeRclone({REMOTE: base})
    _patch_transport(monkeypatch, fake)

    rc = cli.main([
        "--project-root", str(project_root),
        "--remote", f"rclone:{REMOTE}",
        "--sentinel-value", SENTINEL_VALUE,
        "prepare", "--run-id", "../evil", "--create-root",
    ])
    assert rc == 2
    assert "hf08" in capsys.readouterr().err
    assert fake.calls == []


def test_cli_rclone_ambiguous_root_stops(tmp_path, monkeypatch, capsys):
    project_root = tmp_path / "proj"
    base = tmp_path / "base"
    base.mkdir()
    _write_jobs(project_root)
    fake = FakeRclone({REMOTE: base})
    fake.override_lsf(
        REMOTE, [f"{HANDOFF_ROOT_NAME}/", f"{HANDOFF_ROOT_NAME.upper()}/"]
    )
    _patch_transport(monkeypatch, fake)

    rc = cli.main([
        "--project-root", str(project_root),
        "--remote", f"rclone:{REMOTE}",
        "--sentinel-value", SENTINEL_VALUE,
        "prepare", "--run-id", RUN, "--create-root",
    ])
    assert rc == 2
    assert "ambiguous" in capsys.readouterr().err


def test_cli_rclone_status_and_fetch_roundtrip(tmp_path, monkeypatch, capsys):
    project_root = tmp_path / "proj"
    base = tmp_path / "base"
    base.mkdir()
    _write_jobs(project_root)
    fake = FakeRclone({REMOTE: base})
    _patch_transport(monkeypatch, fake)

    assert cli.main([
        "--project-root", str(project_root),
        "--remote", f"rclone:{REMOTE}",
        "--sentinel-value", SENTINEL_VALUE,
        "prepare", "--run-id", RUN, "--create-root",
    ]) == 0

    _publish_output(base, RUN, zip_path=_make_zip(_run_dir(base) / "output" / f"{RUN}_assets.zip"))
    capsys.readouterr()

    rc = cli.main([
        "--project-root", str(project_root),
        "--remote", f"rclone:{REMOTE}",
        "--sentinel-value", SENTINEL_VALUE,
        "status", "--run-id", RUN,
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "output_done=yes" in out
    assert "output_status=PASS" in out

    staging = tmp_path / "staging"
    rc = cli.main([
        "--project-root", str(project_root),
        "--remote", f"rclone:{REMOTE}",
        "--sentinel-value", SENTINEL_VALUE,
        "fetch", "--run-id", RUN, "--staging-dir", str(staging),
    ])
    assert rc == 0, capsys.readouterr().err
    assert (staging / f"{RUN}_assets.zip").is_file()
