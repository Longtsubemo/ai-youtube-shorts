"""Run the SDXL image worker on a free Colab T4 through Google's Colab CLI.

No browser and no Drive: the run is prepared into a local handoff root, the
existing worker notebook (with its Drive-mount cell swapped for an env-var
read) is executed on a fresh Colab VM with ``colab exec``, the published output
is downloaded back, and the existing ``tools/colab_handoff.py`` fetch/validate/
import verify it exactly as they do for the Drive route.

The Colab CLI is Linux/macOS only, so by default it is called inside WSL
(``~/.colab-cli-venv/bin/colab`` in the ``Ubuntu`` distro). Override the whole
command with ``COLAB_CLI`` (shell-split), e.g. ``COLAB_CLI="colab"`` on Linux.

    python tools/colab_cli_run.py --run-id <id> [--authorize-import]

Free tier only: one ``colab new --gpu T4`` attempt, no retries; the VM is always
stopped afterwards (``--keep-session`` to debug).
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - script entrypoint
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nbformat as nbf

from agents.image_jobs.handoff import (
    DONE_MARKER,
    HANDOFF_ROOT_NAME,
    IMAGE_JOBS_FILENAME,
    INPUT_MANIFEST_FILENAME,
    OUTPUT_MANIFEST_FILENAME,
    READY_MARKER,
    STATUS_FILENAME,
)
from notebooks import build_notebook as nb
from notebooks.handoff_notebook_core import OUTPUT_ZIP_SUFFIX, SENTINEL_FILENAME
from tools import colab_handoff
from tools.handoff_transport import build_sentinel

ROOT = Path(__file__).resolve().parent.parent
# --exec skips the shell so arguments arrive unchanged; the path is relative to the WSL home.
DEFAULT_CLI = ["wsl", "-d", "Ubuntu", "--cd", "~", "--exec", ".colab-cli-venv/bin/colab"]
REMOTE_BASE = "/content/handoff_base"
SESSION = "aiyt-images"
EXEC_TIMEOUT_SECONDS = 3600

_HEADLESS_MOUNT = '''# HEADLESS (Colab CLI): no Drive mount; the handoff root was uploaded to the VM.
import os

TRUSTED_SENTINEL = os.environ.get("HANDOFF_SENTINEL_VALUE")
if not TRUSTED_SENTINEL:
    raise SystemExit("HANDOFF_SENTINEL_VALUE is not set (pass it with colab exec --env)")
'''


class ColabRunError(RuntimeError):
    pass


def colab_cli() -> list[str]:
    return shlex.split(os.environ["COLAB_CLI"]) if os.environ.get("COLAB_CLI") else list(DEFAULT_CLI)


def cli_path(path: Path, cli: list[str]) -> str:
    """Local path as the CLI sees it: /mnt/<drive>/... when it runs inside WSL."""
    path = Path(path).resolve()
    if cli[0].lower() != "wsl" or not path.drive:
        return str(path)
    return f"/mnt/{path.drive[0].lower()}" + path.as_posix()[len(path.drive):]


def headless_notebook(run_id: str) -> nbf.NotebookNode:
    """The worker notebook with RUN_ID set and Drive/Colab-Secret access removed."""
    notebook = nb.build_notebook()
    swapped = 0
    for cell in notebook.cells:
        if cell.source == nb._MOUNT:
            cell.source, swapped = _HEADLESS_MOUNT, swapped + 1
        elif cell.source == nb._CONFIG:
            cell.source = cell.source.replace('RUN_ID = "REPLACE_WITH_RUN_ID"', f"RUN_ID = {run_id!r}").replace(
                'DRIVE_BASE = "/content/drive/MyDrive"', f"DRIVE_BASE = {REMOTE_BASE!r}")
            swapped += 1
    if swapped != 2 or any("from google.colab import" in c.source for c in notebook.cells):
        raise ColabRunError("worker notebook layout changed; update tools/colab_cli_run.py")
    return notebook


def run(run_id: str, *, authorize_import: bool = False, keep_session: bool = False,
        runner=subprocess.run, project_root: Path = ROOT) -> int:
    run_id = colab_handoff._validate_run_id(run_id)
    if not (project_root / "data" / run_id / IMAGE_JOBS_FILENAME).is_file():
        raise ColabRunError(f"data/{run_id}/{IMAGE_JOBS_FILENAME} not found; run `python -m agents.image_jobs` first")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    work = project_root / "handoff_staging" / "colab_cli" / f"{run_id}-{stamp}"
    handoff_root = work / "base" / HANDOFF_ROOT_NAME
    handoff_root.mkdir(parents=True)
    # Fresh per-run value: the VM is private, but the worker still checks the root it reads.
    sentinel_value = secrets.token_urlsafe(24)
    sentinel = build_sentinel(verification_value=sentinel_value)
    (handoff_root / SENTINEL_FILENAME).write_text(json.dumps(sentinel, indent=2) + "\n", encoding="utf-8")

    common = ["--project-root", str(project_root), "--handoff-root", str(handoff_root)]
    if colab_handoff.main([*common, "prepare", "--run-id", run_id]) != 0:
        raise ColabRunError("prepare failed")
    worker = work / "worker.ipynb"
    with worker.open("w", encoding="utf-8") as handle:
        nbf.write(headless_notebook(run_id), handle)

    cli = colab_cli()
    remote_run = f"{REMOTE_BASE}/{HANDOFF_ROOT_NAME}/{run_id}"
    local_run = handoff_root / run_id

    def colab(*args: str, timeout: float = 600, check: bool = True, quiet: bool = False):
        done = runner([*cli, *args], capture_output=quiet, text=True, timeout=timeout)
        if check and done.returncode != 0:
            raise ColabRunError(f"colab {args[0]} exited {done.returncode}")
        return done

    print(f"[colab_cli_run] requesting a free T4 (session {SESSION})")
    colab("new", "-s", SESSION, "--gpu", "T4")
    try:
        mkdirs = work / "mkdirs.py"
        mkdirs.write_text(f"import os\nfor d in ('input', 'status', 'output'):\n"
                          f"    os.makedirs({remote_run!r} + '/' + d, exist_ok=True)\n", encoding="utf-8")
        colab("exec", "-s", SESSION, "-f", cli_path(mkdirs, cli))
        uploads = [handoff_root / SENTINEL_FILENAME] + [
            local_run / sub / name for sub, name in (("input", IMAGE_JOBS_FILENAME), ("input", INPUT_MANIFEST_FILENAME),
                                                     ("input", READY_MARKER), ("status", STATUS_FILENAME))]
        for path in uploads:
            remote = f"{REMOTE_BASE}/{path.relative_to(handoff_root.parent).as_posix()}"
            colab("upload", "-s", SESSION, cli_path(path, cli), remote)

        print("[colab_cli_run] running the worker (model download + generation; several minutes)")
        exec_done = colab("exec", "-s", SESSION, "-f", cli_path(worker, cli), "--timeout", "0",
                          "--env", f"HANDOFF_SENTINEL_VALUE={sentinel_value}",
                          timeout=EXEC_TIMEOUT_SECONDS, check=False)

        # Bring back whatever the worker published; DONE exists only on success.
        wanted = [("status", STATUS_FILENAME), ("output", STATUS_FILENAME), ("output", OUTPUT_MANIFEST_FILENAME),
                  ("output", run_id + OUTPUT_ZIP_SUFFIX), ("output", DONE_MARKER)]
        for sub, name in wanted:
            target = local_run / sub / name
            target.parent.mkdir(parents=True, exist_ok=True)
            colab("download", "-s", SESSION, f"{remote_run}/{sub}/{name}", cli_path(target, cli),
                  check=False, quiet=True)
    finally:
        if keep_session:
            print(f"[colab_cli_run] session {SESSION} left running (--keep-session); stop it with `colab stop -s {SESSION}`")
        else:
            colab("stop", "-s", SESSION, check=False)

    if not (local_run / "output" / DONE_MARKER).is_file():
        status = local_run / "status" / STATUS_FILENAME
        detail = status.read_text(encoding="utf-8")[-800:] if status.is_file() else "no status downloaded"
        raise ColabRunError(f"worker did not publish DONE (exec exit {exec_done.returncode}); status: {detail}")

    staged = ["--staging-dir", str(work / "staged")]  # never mixes with the Drive route's staging
    for step in [["fetch"], ["validate"]] + ([["import", "--authorize-import"]] if authorize_import else []):
        if colab_handoff.main([*common, step[0], "--run-id", run_id, *staged, *step[1:]]) != 0:
            raise ColabRunError(f"{step[0]} failed")
    if not authorize_import:
        print("[colab_cli_run] validated; re-run import with --authorize-import to write data/<run_id>/assets/")
    print(f"[colab_cli_run] handoff kept at {work}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python tools/colab_cli_run.py", description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--authorize-import", action="store_true",
                        help="import the verified images into data/<run_id>/assets/")
    parser.add_argument("--keep-session", action="store_true", help="do not stop the Colab VM (debugging)")
    args = parser.parse_args(argv)
    try:
        return run(args.run_id, authorize_import=args.authorize_import, keep_session=args.keep_session)
    except (ColabRunError, colab_handoff.HandoffCLIError, subprocess.TimeoutExpired, OSError) as exc:
        print(f"[colab_cli_run] FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
