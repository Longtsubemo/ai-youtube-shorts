"""Offline tests for the Milestone 3B.1 transport scaffold.

Everything here is offline: no network, no Google Drive, no rclone, no GPU. The
tests drive the CLI through ``main(argv)`` with ``--remote local:<dir>`` against
a temporary directory that acts as the remote handoff base, and unit-test the
transport layer (`tools/handoff_transport.py`) directly with a synthetic
transport for the discovery/ambiguity rules.

The rclone-backed transport (3B.2) is not present yet; these tests pin the
protocol and the offline behavior it must preserve.
"""

from __future__ import annotations

import ast
import io
import json
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

RUN = "e2e-drive-run-01"
ASSETS = ["sc01_main", "sc02_main"]
PNG_SIZE = (768, 1344)


# --- fixtures / helpers -------------------------------------------------------


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    return tmp_path / "proj"


@pytest.fixture
def remote_base(tmp_path: Path) -> Path:
    return tmp_path / "remote"


def _write_jobs(project_root: Path, run_id: str = RUN, assets=None) -> Path:
    assets = assets or ASSETS
    jobs = []
    for i, asset in enumerate(assets, start=1):
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


def _make_zip(dest: Path, assets=ASSETS, *, size=PNG_SIZE, run_id: str = RUN) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
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
    result_document = {
        "schema": "image-generation-results/1.0",
        "run_id": run_id,
        "source_json": "image_jobs.json",
        "channel_id": "finance_mascot",
        "title": "A video",
        "status": "completed",
        "job_count": len(results),
        "generated_count": len(results),
        "skipped_count": 0,
        "failed_count": 0,
        "resolution": {"width": PNG_SIZE[0], "height": PNG_SIZE[1]},
        "steps": 4,
        "guidance_scale": 0.0,
        "results": results,
    }
    with zipfile.ZipFile(dest, "w") as zf:
        for asset in assets:
            zf.writestr(f"{asset}.png", _png_bytes(size))
        zf.writestr("result.json", json.dumps(result_document))
    return dest


def _run_dir(remote_base: Path, run_id: str = RUN) -> Path:
    return remote_base / HANDOFF_ROOT_NAME / run_id


def _read_input_manifest(remote_base: Path, run_id: str = RUN) -> dict:
    path = _run_dir(remote_base, run_id) / "input" / "input_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _publish_output(remote_base: Path, run_id: str, *, zip_path: Path) -> None:
    """Simulate the Colab worker publishing a completed output into the remote."""
    in_manifest = _read_input_manifest(remote_base, run_id)
    assets = list(in_manifest["expected_asset_ids"])
    out_dir = _run_dir(remote_base, run_id) / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_output_manifest(
        run_id,
        attempt_id=in_manifest["attempt_id"],
        image_jobs_sha256=in_manifest["image_jobs_sha256"],
        zip_path=zip_path,
        members=[f"{a}.png" for a in assets] + ["result.json"],
        generated_asset_ids=assets,
    )
    dump_json(manifest, out_dir / "output_manifest.json")
    status = build_status(
        run_id,
        attempt_id=in_manifest["attempt_id"],
        image_jobs_sha256=in_manifest["image_jobs_sha256"],
        overall_status="PASS",
        current_stage=STAGES[-1],
        completed_stages=list(STAGES),
        completed_asset_ids=assets,
    )
    dump_json(status, out_dir / "status.json")
    write_marker(out_dir, "DONE", content=in_manifest["attempt_id"])


def _remote(project_root: Path, remote_base: Path, *rest: str) -> list[str]:
    return ["--project-root", str(project_root), "--remote", f"local:{remote_base}", *rest]


# --- discovery / transport unit tests -----------------------------------------


class _ListingTransport(transport_mod.RemoteTransport):
    """Synthetic transport for the discovery rules (no real filesystem)."""

    scheme = "fake"

    def __init__(self, entries, dirs):
        self._entries = list(entries)
        self._dirs = set(dirs)

    def base_display(self) -> str:
        return "fake:"

    def list_entries(self, rel: str = "") -> list[str]:
        return list(self._entries)

    def is_dir(self, rel: str) -> bool:
        return rel in self._dirs

    def exists(self, rel: str) -> bool:
        return rel in self._dirs or rel in self._entries

    def mkdir(self, rel: str) -> None:
        self._entries.append(rel)
        self._dirs.add(rel)

    def download_tree(self, rel: str, dest: Path) -> None:  # pragma: no cover
        raise NotImplementedError

    def upload_tree(self, rel, src, *, upload_last=()):  # pragma: no cover
        raise NotImplementedError


def test_discover_root_creates_when_absent(tmp_path):
    transport = transport_mod.LocalDirectoryTransport(tmp_path / "base")
    assert transport_mod.discover_root(transport, create=True) == HANDOFF_ROOT_NAME
    assert (tmp_path / "base" / HANDOFF_ROOT_NAME).is_dir()


def test_discover_root_missing_without_create_raises(tmp_path):
    transport = transport_mod.LocalDirectoryTransport(tmp_path / "base")
    with pytest.raises(transport_mod.RootDiscoveryError, match="not found"):
        transport_mod.discover_root(transport, create=False)


def test_discover_root_reuses_single_existing(tmp_path):
    (tmp_path / "base" / HANDOFF_ROOT_NAME).mkdir(parents=True)
    transport = transport_mod.LocalDirectoryTransport(tmp_path / "base")
    assert transport_mod.discover_root(transport, create=False) == HANDOFF_ROOT_NAME


def test_discover_root_ambiguous_stops():
    transport = _ListingTransport(
        [HANDOFF_ROOT_NAME, HANDOFF_ROOT_NAME.upper()],
        [HANDOFF_ROOT_NAME, HANDOFF_ROOT_NAME.upper()],
    )
    with pytest.raises(transport_mod.RootDiscoveryError, match="ambiguous"):
        transport_mod.discover_root(transport, create=True)


def test_discover_root_case_mismatch_stops():
    transport = _ListingTransport([HANDOFF_ROOT_NAME.upper()], [HANDOFF_ROOT_NAME.upper()])
    with pytest.raises(transport_mod.RootDiscoveryError, match="ambiguous"):
        transport_mod.discover_root(transport, create=False)


def test_discover_root_file_is_not_a_directory():
    transport = _ListingTransport([HANDOFF_ROOT_NAME], [])
    with pytest.raises(transport_mod.RootDiscoveryError, match="not a directory"):
        transport_mod.discover_root(transport, create=False)


def test_unsafe_relative_path_rejected(tmp_path):
    transport = transport_mod.LocalDirectoryTransport(tmp_path / "base")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "x").write_text("x", encoding="utf-8")
    with pytest.raises(transport_mod.TransportError, match="unsafe"):
        transport.upload_tree("../evil", tmp_path / "src")


def test_transport_module_imports_only_stdlib_and_agents():
    # Milestone 3B.2 adds the rclone transport here, so the module now imports
    # ``subprocess``/``json``/``time``/``datetime``. Those are standard library;
    # imports remain free of third-party SDKs. The rclone subprocess runs only
    # through the injected runner when a transport method is called (see
    # ``tests/test_rclone_transport.py``), never at import time.
    source = Path(transport_mod.__file__).read_text(encoding="utf-8")
    mods = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    assert mods <= {
        "__future__", "abc", "json", "shutil", "subprocess", "tempfile", "time",
        "datetime", "collections", "pathlib", "agents",
    }


# --- remote CLI flow ----------------------------------------------------------


def test_remote_prepare_publishes_ready_last(project_root, remote_base, monkeypatch):
    _write_jobs(project_root)
    order: list[str] = []
    original = transport_mod.LocalDirectoryTransport._copy_file

    def _record(self, source, dest):
        order.append(dest.name)
        return original(self, source, dest)

    monkeypatch.setattr(transport_mod.LocalDirectoryTransport, "_copy_file", _record)

    rc = cli.main(_remote(project_root, remote_base, "prepare", "--run-id", RUN, "--create-root"))
    assert rc == 0
    ready = _run_dir(remote_base) / "input" / "READY"
    assert ready.is_file()
    assert ready.read_text(encoding="utf-8") == _read_input_manifest(remote_base)["attempt_id"]
    assert order[-1] == "READY"
    # the default local handoff root was not consulted
    assert not (project_root / HANDOFF_ROOT_NAME).exists()


def test_remote_prepare_requires_create_root(project_root, remote_base, capsys):
    _write_jobs(project_root)
    rc = cli.main(_remote(project_root, remote_base, "prepare", "--run-id", RUN))
    assert rc == 2
    assert "not found" in capsys.readouterr().err
    assert not (remote_base / HANDOFF_ROOT_NAME).exists()


def test_remote_prepare_failure_does_not_create_root(project_root, remote_base, capsys):
    # no source image_jobs.json -> prepare fails before ensure_root
    rc = cli.main(_remote(project_root, remote_base, "prepare", "--run-id", RUN, "--create-root"))
    assert rc == 2
    assert "hf01" in capsys.readouterr().err
    assert not (remote_base / HANDOFF_ROOT_NAME).exists()


def test_remote_root_reused_and_force_makes_new_attempt(project_root, remote_base, capsys):
    _write_jobs(project_root)
    base = _remote(project_root, remote_base, "prepare", "--run-id", RUN, "--create-root")
    assert cli.main(base) == 0
    first = _read_input_manifest(remote_base)["attempt_id"]

    assert cli.main(_remote(project_root, remote_base, "prepare", "--run-id", RUN)) == 2
    assert cli.main(_remote(project_root, remote_base, "prepare", "--run-id", RUN, "--force")) == 0
    second = _read_input_manifest(remote_base)["attempt_id"]
    assert second != first

    entries = [p for p in (remote_base).iterdir() if p.is_dir()]
    assert [p.name for p in entries] == [HANDOFF_ROOT_NAME]


def test_remote_status_and_fetch_roundtrip(project_root, remote_base, tmp_path, capsys):
    _write_jobs(project_root)
    assert cli.main(_remote(project_root, remote_base, "prepare", "--run-id", RUN, "--create-root")) == 0
    _publish_output(
        remote_base, RUN, zip_path=_make_zip(_run_dir(remote_base) / "output" / f"{RUN}_assets.zip")
    )
    capsys.readouterr()

    rc = cli.main(_remote(project_root, remote_base, "status", "--run-id", RUN))
    out = capsys.readouterr().out
    assert rc == 0
    assert "output_done=yes" in out
    assert "output_status=PASS" in out

    staging = tmp_path / "staging"
    rc = cli.main(_remote(
        project_root, remote_base, "fetch", "--run-id", RUN, "--staging-dir", str(staging)
    ))
    assert rc == 0
    assert (staging / f"{RUN}_assets.zip").is_file()


def test_remote_import_writes_local_only(project_root, remote_base, capsys):
    _write_jobs(project_root)
    assert cli.main(_remote(project_root, remote_base, "prepare", "--run-id", RUN, "--create-root")) == 0
    _publish_output(
        remote_base, RUN, zip_path=_make_zip(_run_dir(remote_base) / "output" / f"{RUN}_assets.zip")
    )
    capsys.readouterr()
    rc = cli.main(_remote(
        project_root, remote_base, "import", "--run-id", RUN, "--authorize-import"
    ))
    assert rc == 0
    for asset in ASSETS:
        assert (project_root / "data" / RUN / "assets" / f"{asset}.png").is_file()
    # nothing was uploaded back: no assets/ under the remote run folder
    assert not (_run_dir(remote_base) / "assets").exists()
    assert not (_run_dir(remote_base) / "data").exists()


def test_remote_unsupported_scheme(project_root, capsys):
    _write_jobs(project_root)
    rc = cli.main(["--project-root", str(project_root), "--remote", "s3:bucket",
                   "prepare", "--run-id", RUN, "--create-root"])
    assert rc == 2
    assert "hf09" in capsys.readouterr().err


def test_remote_ambiguous_root_stops(project_root, remote_base, monkeypatch, capsys):
    _write_jobs(project_root)
    fake = _ListingTransport(
        [HANDOFF_ROOT_NAME, HANDOFF_ROOT_NAME.upper()],
        [HANDOFF_ROOT_NAME, HANDOFF_ROOT_NAME.upper()],
    )
    monkeypatch.setattr(cli, "_make_transport", lambda spec: fake)
    rc = cli.main(_remote(project_root, remote_base, "prepare", "--run-id", RUN, "--create-root"))
    assert rc == 2
    assert "ambiguous" in capsys.readouterr().err


def test_remote_missing_run_status(project_root, remote_base, capsys):
    (remote_base / HANDOFF_ROOT_NAME).mkdir(parents=True)
    rc = cli.main(_remote(project_root, remote_base, "status", "--run-id", RUN))
    assert rc == 2
    assert "not found" in capsys.readouterr().err
