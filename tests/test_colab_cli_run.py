"""Headless Colab CLI image runner: offline, with a fake `colab` command and VM filesystem."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tests.test_colab_handoff_cli import ASSETS, RUN, _make_zip, _publish_output, _write_jobs
from tools import colab_cli_run as runner_mod


class FakeColab:
    """Stands in for the Colab CLI; `/content/...` paths live under `vm`."""

    def __init__(self, vm: Path, *, worker_succeeds: bool = True):
        self.vm, self.worker_succeeds, self.calls = vm, worker_succeeds, []

    def _vm(self, remote: str) -> Path:
        assert remote.startswith("/content/")
        return self.vm / remote[len("/content/"):]

    def __call__(self, cmd, **kwargs):
        assert cmd[0] == "fakecolab"
        sub, args = cmd[1], cmd[2:]
        self.calls.append(sub)
        rc = 0
        if sub == "exec":
            path = Path(args[args.index("-f") + 1])
            if path.name == "mkdirs.py":
                exec(compile(path.read_text(encoding="utf-8").replace("'/content/", f"'{self.vm.as_posix()}/"),
                             "mkdirs", "exec"))
            else:
                text = path.read_text(encoding="utf-8")
                assert "from google.colab import" not in text and RUN in text
                assert any(a.startswith("HANDOFF_SENTINEL_VALUE=") for a in args)
                if self.worker_succeeds:
                    root = self.vm / "handoff_base" / "ai-youtube-channel-handoff"
                    zip_path = _make_zip(root / RUN / "output" / f"{RUN}_generated_assets.zip")
                    _publish_output(root, RUN, zip_path=zip_path, generated=ASSETS,
                                    members=[f"{a}.png" for a in ASSETS] + ["result.json"])
                else:
                    rc = 1
        elif sub == "upload":
            target = self._vm(args[-1])
            assert target.parent.is_dir(), f"remote dir missing for {args[-1]}"
            shutil.copy2(args[-2], target)
        elif sub == "download":
            source = self._vm(args[-2])
            if source.is_file():
                shutil.copy2(source, args[-1])
            else:
                rc = 1
        return subprocess.CompletedProcess(cmd, rc, "", "")


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_CLI", "fakecolab")
    root = tmp_path / "proj"
    _write_jobs(root)
    return root


def test_success_downloads_verifies_and_imports(project, tmp_path):
    fake = FakeColab(tmp_path / "vm")
    assert runner_mod.run(RUN, authorize_import=True, runner=fake, project_root=project) == 0
    assert fake.calls[0] == "new" and fake.calls[-1] == "stop"
    assert fake.calls.count("upload") == 5
    for asset in ASSETS:
        assert (project / "data" / RUN / "assets" / f"{asset}.png").is_file()


def test_validate_only_without_authorization(project, tmp_path):
    assert runner_mod.run(RUN, runner=FakeColab(tmp_path / "vm"), project_root=project) == 0
    assert not (project / "data" / RUN / "assets").exists()


def test_worker_failure_still_stops_the_vm_and_imports_nothing(project, tmp_path):
    fake = FakeColab(tmp_path / "vm", worker_succeeds=False)
    with pytest.raises(runner_mod.ColabRunError, match="did not publish DONE"):
        runner_mod.run(RUN, authorize_import=True, runner=fake, project_root=project)
    assert fake.calls[-1] == "stop"
    assert not (project / "data" / RUN / "assets").exists()


def test_no_gpu_fails_without_retry(project):
    calls = []

    def no_gpu(cmd, **kwargs):
        calls.append(cmd[1])
        return subprocess.CompletedProcess(cmd, 1 if cmd[1] == "new" else 0, "", "")

    with pytest.raises(runner_mod.ColabRunError, match="colab new"):
        runner_mod.run(RUN, runner=no_gpu, project_root=project)
    assert calls == ["new"]


def test_headless_notebook_and_wsl_paths():
    notebook = runner_mod.headless_notebook("run-x")
    text = "".join(c.source for c in notebook.cells)
    assert 'RUN_ID = \'run-x\'' in text and "DRIVE_BASE = '/content/handoff_base'" in text
    assert 'drive.mount("/content/drive")' not in text and "userdata.get(" not in text
    assert runner_mod.cli_path(Path("C:/Users/x/a b.txt"), ["wsl"]) == "/mnt/c/Users/x/a b.txt"
