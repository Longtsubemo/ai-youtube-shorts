"""Transport layer for the Colab handoff root (Milestone 3B.1, offline scaffold).

This module isolates *where* the handoff root lives from the schema-1.1
integrity logic in ``agents/image_jobs/handoff.py``. The integrity logic runs on
plain local filesystem paths; this layer moves those paths to and from a remote
handoff root.

Milestone 3B.1 is deliberately **offline**: it provides

* the :class:`RemoteTransport` protocol,
* :class:`LocalDirectoryTransport`, a directory-backed implementation used as an
  offline "remote" (tests, and a local mirror of a Drive folder),
* :func:`discover_root`, the fail-closed handoff-root discovery rule, and
* :class:`RemoteMirrorBackend`, which materialises a remote run folder into a
  temporary local mirror, reuses the existing local integrity code unchanged,
  and publishes the result back without ever replacing another run.

Milestone 3B.2 adds :class:`RcloneTransport` behind the exact same
:class:`RemoteTransport` protocol, so the CLI and the integrity logic do not
change. The rclone transport shells out to the ``rclone`` binary through an
**injectable runner**, so:

* importing this module never runs a subprocess, never touches the network and
  never reads any rclone configuration;
* tests substitute a fake runner (see ``tests/test_rclone_transport.py``) and
  remain fully offline;
* the real runner is the only place a subprocess is created, and it is invoked
  only when an rclone transport method is actually called.

Preflight (:func:`preflight_remote`) is read-only except for the explicit,
authorized case of creating a missing root (never a second root) together with
its sentinel. The sentinel is always verified against an **independently
supplied trusted value**; the remote file is never trusted on its own.
"""

from __future__ import annotations

import abc
import json
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path

from agents.image_jobs.handoff import HANDOFF_ROOT_NAME, IMAGE_JOBS_FILENAME

STAGING_DIR_NAME = "handoff_staging"


class TransportError(Exception):
    """A transport-level failure (unsafe path, missing remote tree, ...)."""


class RootDiscoveryError(TransportError):
    """The handoff root is missing, ambiguous, or not a directory.

    Raised instead of guessing so that a duplicate or malformed root stops the
    operation rather than causing a second root to be created.
    """


def _safe_rel(rel: str) -> str:
    """Return a normalised relative path, or raise for an unsafe one.

    Rejects absolute/drive-qualified paths and any ``..`` component so a remote
    relative path can never escape the transport's base directory.
    """
    if rel in ("", "."):
        return ""
    if rel.startswith(("/", "\\")):
        # ``Path('/abs').is_absolute()`` is False on Windows (no drive), so a
        # leading separator is rejected explicitly rather than relying on the OS.
        raise TransportError(f"unsafe remote path {rel!r}: must be relative")
    path = Path(rel)
    if path.is_absolute() or path.drive:
        raise TransportError(f"unsafe remote path {rel!r}: must be relative")
    parts = list(path.parts)
    if any(part in ("", "..") for part in parts):
        raise TransportError(
            f"unsafe remote path {rel!r}: must not contain '..' or empty components"
        )
    return "/".join(parts)


class RemoteTransport(abc.ABC):
    """Where the handoff root lives. Only this layer knows about "remote".

    Paths passed to these methods are relative to the transport's base
    directory. The concrete implementation may be a local directory (offline),
    an rclone remote, or anything else; the callers only use these methods.
    """

    scheme: str = ""

    #: Whether this transport requires a verified sentinel before any transfer.
    #: Only account-bound remotes (rclone/Drive) set this; the offline directory
    #: transport does not, so default offline behaviour is unchanged.
    requires_sentinel: bool = False

    @abc.abstractmethod
    def base_display(self) -> str:
        """Human-readable description of the base, for messages/reports."""

    @abc.abstractmethod
    def list_entries(self, rel: str = "") -> list[str]:
        """Names of the entries directly under ``<base>/<rel>`` (empty if absent)."""

    @abc.abstractmethod
    def is_dir(self, rel: str) -> bool:
        """True if ``<base>/<rel>`` is a directory."""

    @abc.abstractmethod
    def exists(self, rel: str) -> bool:
        """True if ``<base>/<rel>`` exists (file or directory)."""

    @abc.abstractmethod
    def mkdir(self, rel: str) -> None:
        """Create ``<base>/<rel>`` (and parents); fail if it already exists."""

    @abc.abstractmethod
    def download_tree(self, rel: str, dest: Path) -> None:
        """Copy the remote tree ``<base>/<rel>`` into local ``dest``."""

    @abc.abstractmethod
    def upload_tree(
        self, rel: str, src: Path, *, upload_last: Iterable[str] = ()
    ) -> None:
        """Replace remote ``<base>/<rel>`` with local ``src``.

        ``upload_last`` names relative files inside ``src`` that must be copied
        **after** all other files, so a content-bearing marker (``READY``/``DONE``)
        is always published last.
        """

    def read_bytes(self, rel: str) -> bytes:
        """Return the bytes of the file at ``<base>/<rel>``.

        Concrete (non-abstract) so existing transports/tests that do not need
        file reads remain valid; transports that support preflight override it.
        """
        raise TransportError(f"{type(self).__name__} does not support read_bytes")

    def write_bytes(self, rel: str, data: bytes) -> None:
        """Create the file at ``<base>/<rel>``; refuse to overwrite an existing one.

        Concrete (non-abstract); used only by the rclone preflight to publish the
        sentinel on an explicitly authorized root creation.
        """
        raise TransportError(f"{type(self).__name__} does not support write_bytes")


class LocalDirectoryTransport(RemoteTransport):
    """Directory-backed transport: an offline "remote" and the test double.

    Everything lives under ``base``. This is intentionally the only concrete
    transport in 3B.1; the rclone transport added later implements the same
    methods.
    """

    scheme = "local"

    def __init__(self, base: str | Path) -> None:
        self.base = Path(base)

    def base_display(self) -> str:
        return f"local:{self.base}"

    def _path(self, rel: str) -> Path:
        safe = _safe_rel(rel)
        return self.base / safe if safe else self.base

    def list_entries(self, rel: str = "") -> list[str]:
        path = self._path(rel)
        if not path.is_dir():
            return []
        return sorted(entry.name for entry in path.iterdir())

    def is_dir(self, rel: str) -> bool:
        return self._path(rel).is_dir()

    def exists(self, rel: str) -> bool:
        return self._path(rel).exists()

    def mkdir(self, rel: str) -> None:
        path = self._path(rel)
        if path.exists():
            raise TransportError(f"remote path already exists: {path}")
        path.mkdir(parents=True)

    def download_tree(self, rel: str, dest: Path) -> None:
        source = self._path(rel)
        if not source.is_dir():
            raise TransportError(f"remote run folder not found: {source}")
        shutil.copytree(source, Path(dest), dirs_exist_ok=True)

    def upload_tree(
        self, rel: str, src: Path, *, upload_last: Iterable[str] = ()
    ) -> None:
        source_root = Path(src)
        if not source_root.is_dir():
            raise TransportError(f"local source folder not found: {source_root}")

        last = [_safe_rel(entry) for entry in upload_last]
        last_set = set(last)

        dest_root = self._path(rel)
        if dest_root.exists():
            if dest_root.is_dir():
                shutil.rmtree(dest_root)
            else:
                dest_root.unlink()
        dest_root.mkdir(parents=True)

        for path in sorted(p for p in source_root.rglob("*") if p.is_file()):
            relative = path.relative_to(source_root).as_posix()
            if relative in last_set:
                continue
            self._copy_file(path, dest_root / relative)

        for relative in last:
            path = source_root / relative
            if not path.is_file():
                raise TransportError(f"upload_last file missing locally: {relative}")
            self._copy_file(path, dest_root / relative)

    def _copy_file(self, source: Path, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)


def discover_root(
    transport: RemoteTransport,
    *,
    create: bool,
    root_name: str = HANDOFF_ROOT_NAME,
) -> str:
    """Find the single handoff root, or fail closed.

    * exactly one matching directory -> reuse it and return its name;
    * none and ``create`` is false -> :class:`RootDiscoveryError`;
    * none and ``create`` is true -> create exactly ``root_name`` once;
    * more than one match, a case mismatch, or a non-directory -> fail with
      :class:`RootDiscoveryError`; never create a second root.
    """
    entries = transport.list_entries("")
    candidates = [name for name in entries if name.lower() == root_name.lower()]

    if not candidates:
        if not create:
            raise RootDiscoveryError(
                f"handoff root {root_name!r} not found at {transport.base_display()}; "
                "pass --create-root to create it"
            )
        transport.mkdir(root_name)
        return root_name

    if len(candidates) > 1:
        raise RootDiscoveryError(
            f"ambiguous handoff root at {transport.base_display()}: "
            f"multiple entries match {root_name!r}: {candidates}"
        )

    found = candidates[0]
    if found != root_name:
        raise RootDiscoveryError(
            f"ambiguous handoff root {found!r} at {transport.base_display()}: "
            f"expected exactly {root_name!r}"
        )
    if not transport.is_dir(found):
        raise RootDiscoveryError(
            f"handoff root {found!r} at {transport.base_display()} is not a directory"
        )
    return found


class RemoteMirrorBackend:
    """Backend over a remote root, materialised as a temporary local mirror.

    It exposes the same path accessors as the offline
    ``LocalHandoffBackend``, but those paths live in a per-invocation temporary
    directory. The caller downloads the run folder before a read, runs the
    existing local integrity code against the mirror, and publishes the mirror
    back for ``prepare``. The integrity logic never learns it is remote.

    The mirror copy of a run is used only as a staging area: the local source
    ``data/<run_id>/image_jobs.json`` and the local staging directory remain on
    the local filesystem, exactly as they are for the offline backend.
    """

    def __init__(
        self,
        transport: RemoteTransport,
        project_root: str | Path = ".",
        *,
        create_root: bool = False,
    ) -> None:
        self.transport = transport
        self.project_root = Path(project_root)
        self.create_root = create_root
        self.mirror_root = Path(tempfile.mkdtemp(prefix="handoff_mirror_"))
        self._root_name: str | None = None
        self.root = f"{transport.base_display()}/{HANDOFF_ROOT_NAME}"

    # --- discovery / path accessors (mirror-local) ---------------------------

    def ensure_root(self) -> Path:
        if self._root_name is None:
            self._root_name = discover_root(self.transport, create=self.create_root)
        return self.mirror_root

    def run_dir(self, run_id: str) -> Path:
        return self.mirror_root / run_id

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

    # --- remote sync ---------------------------------------------------------

    def begin_read(self, run_id: str) -> None:
        """Download ``<root>/<run_id>`` into the local mirror, if it exists.

        A missing/ambiguous root or a missing run leaves the mirror empty; the
        wrapped command then reports "run not found" in the normal way, so a
        read never creates anything remotely.
        """
        if self._root_name is None:
            try:
                self._root_name = discover_root(self.transport, create=False)
            except RootDiscoveryError:
                return
        rel = f"{self._root_name}/{run_id}"
        if self.transport.exists(rel):
            self.transport.download_tree(rel, self.run_dir(run_id))

    def publish(self, run_id: str, *, upload_last: Iterable[str] = ()) -> None:
        """Replace ``<root>/<run_id>`` with the mirror (markers last).

        Only the named run folder is touched; every other run stays untouched.
        """
        if self._root_name is None:
            self._root_name = discover_root(self.transport, create=self.create_root)
        rel = f"{self._root_name}/{run_id}"
        self.transport.upload_tree(rel, self.run_dir(run_id), upload_last=upload_last)

    def close(self) -> None:
        shutil.rmtree(self.mirror_root, ignore_errors=True)


# --- rclone-backed transport (Milestone 3B.2) --------------------------------
#
# Everything below is opt-in: nothing here runs at import time, and the real
# subprocess runner is only used when an :class:`RcloneTransport` method is
# called with a transport whose runner is the default. Tests inject a fake
# runner and stay fully offline.

RCLONE_BIN_DEFAULT = "rclone"

#: Sentinel file (inside the handoff root) proving access to *this* account.
SENTINEL_FILENAME = "handoff_sentinel.json"
SENTINEL_SCHEMA_VERSION = 1
HANDOFF_PROJECT_ID = "ai-youtube-channel"

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = 2.0

# rclone documents exit code 5 ("temporary error") and 6 ("less serious error")
# as transient. Authentication/authorisation and integrity problems are *never*
# retried (see ``_NONRETRY_MARKERS`` and the requirement not to retry auth).
_TRANSIENT_EXIT_CODES = frozenset({5, 6})
_TRANSIENT_MARKERS = (
    "429",
    "rate limit",
    "ratelimit",
    "ratelimitexceeded",
    "temporarily unavailable",
    "resource_exhausted",
    "connection reset",
    "connection refused",
    "connection aborted",
    "timed out",
    "timeout",
    "try again",
    "backing off",
    "broken pipe",
    "unexpected eof",
    " 500",
    " 502",
    " 503",
    " 504",
)
_NONRETRY_MARKERS = (
    "401",
    "403",
    "invalid_grant",
    "unauthorized",
    "forbidden",
    "insufficientpermission",
    "insufficient permission",
    "permission denied",
    "checksum",
    "corrupt",
    "size mismatch",
    "not authorized",
    "does not have",
)


class CommandResult:
    """A finished rclone invocation: ``returncode`` + captured ``stdout``/``stderr``."""

    __slots__ = ("argv", "returncode", "stdout", "stderr")

    def __init__(
        self,
        argv,
        returncode: int,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        self.argv = list(argv)
        self.returncode = int(returncode)
        self.stdout = stdout or ""
        self.stderr = stderr or ""

    def combined_output(self) -> str:
        return f"{self.stdout}\n{self.stderr}"


class CommandError(TransportError):
    """An rclone invocation failed and is not retryable (or retries exhausted)."""

    def __init__(self, argv, result: CommandResult) -> None:
        self.argv = list(argv)
        self.returncode = result.returncode
        self.stdout = result.stdout
        self.stderr = result.stderr
        super().__init__(
            f"rclone command failed (exit {result.returncode}): {' '.join(self.argv)}"
        )


class RemoteVerificationError(TransportError):
    """Remote access, root, or sentinel verification failed (fail closed).

    A missing, malformed, mismatched, duplicate, or ambiguous root/sentinel state
    stops the operation. Matching folder names alone are never treated as proof
    of account identity.
    """


#: A runner is any callable ``argv -> CommandResult``. Injecting it is how the
#: tests stay offline; the default is the real subprocess runner.
RcloneRunner = Callable[[list[str]], CommandResult]


def default_rclone_runner(argv: list[str]) -> CommandResult:
    """Run ``argv`` via :mod:`subprocess` (no shell) and capture the result.

    The argument vector is passed directly (never a shell string), so remote
    paths cannot inject shell commands. No tokens or configuration values are
    ever placed on the command line; ``rclone config`` is never invoked.
    """
    completed = subprocess.run(  # noqa: S603 - fixed binary, list args, no shell
        list(argv),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return CommandResult(
        argv, completed.returncode, completed.stdout or "", completed.stderr or ""
    )


class RcloneTransport(RemoteTransport):
    """Remote transport backed by the ``rclone`` command-line tool.

    The remote base is an rclone remote spec ending in ``:`` (e.g. ``gdrive:``).
    Every operation is confined to ``<remote>/<safe relative path>``; paths are
    validated by :func:`_safe_rel` before any command is built, so a run id or
    relative path can never escape the base. Commands are always passed as an
    argument list (never a shell string).

    Transfer-level retries are **bounded** and only for transient transport
    failures. Authentication, permission and integrity failures are never
    retried; nor is any generation or whole-job work (this layer only moves
    bytes).
    """

    scheme = "rclone"
    requires_sentinel = True

    def __init__(
        self,
        remote: str,
        *,
        root_name: str = HANDOFF_ROOT_NAME,
        runner: RcloneRunner | None = None,
        rclone_bin: str = RCLONE_BIN_DEFAULT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if not isinstance(remote, str) or not remote.strip() or ":" not in remote:
            raise TransportError(
                f"invalid rclone remote {remote!r}: expected a remote name like "
                "'gdrive:'"
            )
        self.remote = remote.strip()
        self.root_name = root_name
        self._runner: RcloneRunner = runner or default_rclone_runner
        self.rclone_bin = rclone_bin
        self.max_attempts = max(1, int(max_attempts))
        self.backoff_seconds = float(backoff_seconds)
        self._sleep: Callable[[float], None] = sleep or time.sleep

    # --- path / command plumbing --------------------------------------------

    def base_display(self) -> str:
        return f"rclone:{self.remote}"

    def _path(self, rel: str) -> str:
        safe = _safe_rel(rel)
        if safe and any(part.startswith("-") for part in safe.split("/")):
            raise TransportError(
                f"unsafe remote path {rel!r}: components must not start with '-'"
            )
        if not safe:
            return self.remote
        return f"{self.remote.rstrip('/')}/{safe}"

    def _is_transient(self, result: CommandResult) -> bool:
        if result.returncode == 0:
            return False
        text = result.combined_output().lower()
        if any(marker in text for marker in _NONRETRY_MARKERS):
            return False
        if result.returncode in _TRANSIENT_EXIT_CODES:
            return True
        return any(marker in text for marker in _TRANSIENT_MARKERS)

    @staticmethod
    def _is_not_found(result: CommandResult) -> bool:
        if result.returncode in (3, 4):
            return True
        return "not found" in result.combined_output().lower()

    def _invoke(self, args: list[str], *, retry: bool = True) -> CommandResult:
        """Run ``rclone <args>`` with bounded retries for transient failures only."""
        argv = [self.rclone_bin, *args]
        attempts = self.max_attempts if retry else 1
        result = self._runner(argv)
        for attempt in range(attempts):
            if result.returncode == 0:
                return result
            if not retry or not self._is_transient(result) or attempt + 1 >= attempts:
                return result
            self._sleep(self.backoff_seconds * (2**attempt))
            result = self._runner(argv)
        return result

    def _run(self, args: list[str], *, retry: bool = True) -> CommandResult:
        result = self._invoke(args, retry=retry)
        if result.returncode != 0:
            raise CommandError([self.rclone_bin, *args], result)
        return result

    # --- RemoteTransport protocol -------------------------------------------

    def list_entries(self, rel: str = "") -> list[str]:
        self._path(rel)  # validate before running (raises on unsafe paths)
        result = self._invoke(["lsf", self._path(rel)])
        if result.returncode != 0:
            if self._is_not_found(result):
                return []
            raise CommandError([self.rclone_bin, "lsf", self._path(rel)], result)
        names: list[str] = []
        for line in result.stdout.splitlines():
            entry = line.strip()
            if not entry:
                continue
            names.append(entry[:-1] if entry.endswith("/") else entry)
        return sorted(names)

    def is_dir(self, rel: str) -> bool:
        safe = _safe_rel(rel)
        if not safe:
            return self._invoke(["lsf", self._path("")]).returncode == 0
        result = self._invoke(["lsf", self._path(safe), "--dirs-only"])
        return result.returncode == 0

    def exists(self, rel: str) -> bool:
        safe = _safe_rel(rel)
        if not safe:
            return self._invoke(["lsf", self._path("")]).returncode == 0
        parent, _, name = safe.rpartition("/")
        return name in self.list_entries(parent)

    def mkdir(self, rel: str) -> None:
        safe = _safe_rel(rel)
        if not safe:
            raise TransportError("cannot create the rclone base itself")
        if self.exists(safe):
            raise TransportError(f"remote path already exists: {self._path(safe)}")
        self._run(["mkdir", self._path(safe)])

    def download_tree(self, rel: str, dest: Path) -> None:
        self._run(["copy", self._path(rel), str(Path(dest))])

    def upload_tree(
        self, rel: str, src: Path, *, upload_last: Iterable[str] = ()
    ) -> None:
        source_root = Path(src)
        if not source_root.is_dir():
            raise TransportError(f"local source folder not found: {source_root}")

        safe = _safe_rel(rel)
        last = [_safe_rel(entry) for entry in upload_last]
        for entry in last:
            if not (source_root / entry).is_file():
                raise TransportError(f"upload_last file missing locally: {entry}")

        dest = self._path(safe)
        # Phase 1: mirror all payload files, excluding the markers. ``sync`` is
        # confined to this one run folder; no other run or root is touched.
        args = ["sync", str(source_root), dest]
        for entry in last:
            args += ["--exclude", f"/{entry}"]
        self._run(args)

        # Phase 2: publish the content-bearing markers strictly last. If phase 1
        # raised, no marker is written.
        for entry in last:
            self._run(["copyto", str(source_root / entry), f"{dest}/{entry}"])

    # --- preflight helpers ---------------------------------------------------

    def check_access(self) -> None:
        """Confirm the remote is reachable (read-only ``rclone about``).

        Auth/permission failures are not retried (see :meth:`_is_transient`).
        """
        self._run(["about", self.remote])

    def read_bytes(self, rel: str) -> bytes:
        safe = _safe_rel(rel)
        if not safe:
            raise TransportError("cannot read the rclone base itself")
        result = self._run(["cat", self._path(safe)])
        return result.stdout.encode("utf-8")

    def write_bytes(self, rel: str, data: bytes) -> None:
        safe = _safe_rel(rel)
        if not safe:
            raise TransportError("cannot write the rclone base itself")
        if self.exists(safe):
            raise TransportError(
                f"refusing to overwrite existing remote file: {self._path(safe)}"
            )
        with tempfile.TemporaryDirectory(prefix="rclone_write_") as tmp:
            payload = Path(tmp) / "payload"
            payload.write_bytes(data)
            self._run(["copyto", str(payload), self._path(safe)])


# --- sentinel + remote preflight (Milestone 3B.2) ----------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_sentinel(
    *,
    verification_value: str,
    root_name: str = HANDOFF_ROOT_NAME,
    project_id: str = HANDOFF_PROJECT_ID,
    created_utc: str | None = None,
) -> dict:
    """Build the sentinel document (deterministic shape, random value)."""
    if not isinstance(verification_value, str) or not verification_value:
        raise TransportError("sentinel requires a non-empty verification value")
    return {
        "project": project_id,
        "root": root_name,
        "sentinel_version": SENTINEL_SCHEMA_VERSION,
        "verification_value": verification_value,
        "created_utc": created_utc or _utc_now_iso(),
    }


def _dump_json_bytes(document: dict) -> bytes:
    return (json.dumps(document, indent=2) + "\n").encode("utf-8")


def _parse_sentinel(raw: bytes) -> dict:
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RemoteVerificationError(f"handoff sentinel is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise RemoteVerificationError("handoff sentinel must be a JSON object")
    return document


def check_sentinel(
    document: dict,
    *,
    expected_value: str,
    root_name: str = HANDOFF_ROOT_NAME,
    project_id: str = HANDOFF_PROJECT_ID,
) -> None:
    """Validate a sentinel against an independently supplied trusted value.

    Fails closed on any mismatch. The remote file is never trusted on its own:
    the caller must supply the expected value from a trusted local source.
    """
    if not isinstance(expected_value, str) or not expected_value:
        raise RemoteVerificationError(
            "a trusted sentinel value is required to verify the remote account"
        )
    if document.get("project") != project_id:
        raise RemoteVerificationError(
            f"sentinel project {document.get('project')!r} != expected {project_id!r}"
        )
    if document.get("root") != root_name:
        raise RemoteVerificationError(
            f"sentinel root {document.get('root')!r} != expected {root_name!r}"
        )
    if document.get("sentinel_version") != SENTINEL_SCHEMA_VERSION:
        raise RemoteVerificationError(
            "sentinel version "
            f"{document.get('sentinel_version')!r} != expected "
            f"{SENTINEL_SCHEMA_VERSION!r}"
        )
    remote_value = document.get("verification_value")
    if not isinstance(remote_value, str) or not remote_value:
        raise RemoteVerificationError("sentinel has no verification_value")
    if remote_value != expected_value:
        raise RemoteVerificationError(
            "sentinel verification_value does not match the trusted expected value "
            "(the remote account/root does not match)"
        )


def preflight_remote(
    transport: RemoteTransport,
    *,
    expected_sentinel_value: str | None,
    create_root: bool = False,
    root_name: str = HANDOFF_ROOT_NAME,
) -> str:
    """Read-only preflight: accessibility, exact root, and sentinel trust.

    * the remote must be reachable (``check_access`` when available);
    * exactly one matching ``<root_name>`` directory must exist, or the root is
      created **once** when ``create_root`` is explicitly authorized;
    * the sentinel must exist and match ``expected_sentinel_value``;
    * a root that exists but has no sentinel (or a wrong/malformed one) fails
      closed — an existing sentinel is never overwritten.

    Returns the discovered root name. Raises :class:`RemoteVerificationError` or
    :class:`RootDiscoveryError` (both :class:`TransportError`) on any problem.
    """
    if not isinstance(expected_sentinel_value, str) or not expected_sentinel_value:
        raise RemoteVerificationError(
            "an rclone remote requires --sentinel-value (the trusted sentinel value "
            "from your local handoff notes); refusing to transfer without it"
        )

    check_access = getattr(transport, "check_access", None)
    if callable(check_access):
        check_access()

    candidates_before = [
        name for name in transport.list_entries("") if name.lower() == root_name.lower()
    ]
    root = discover_root(transport, create=create_root, root_name=root_name)
    sentinel_rel = f"{root}/{SENTINEL_FILENAME}"

    if not transport.exists(sentinel_rel):
        if create_root and not candidates_before:
            transport.write_bytes(
                sentinel_rel,
                _dump_json_bytes(
                    build_sentinel(verification_value=expected_sentinel_value, root_name=root)
                ),
            )
        else:
            raise RemoteVerificationError(
                f"handoff sentinel {SENTINEL_FILENAME!r} is missing under {root!r} at "
                f"{transport.base_display()}; refusing to proceed"
            )

    document = _parse_sentinel(transport.read_bytes(sentinel_rel))
    check_sentinel(
        document,
        expected_value=expected_sentinel_value,
        root_name=root,
    )
    return root
