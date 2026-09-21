"""Stage-8 safety regression tests.

The video-QA stage must stay a pure offline boundary, just like Stage 7: no
network, no GPU/CUDA, no API keys, no provider coupling, no file writes, and no
Stage-9 work. These tests enforce that by (1) statically auditing the package
source (executable-code tokens only, so docstrings/denylist vocabulary in string
literals are not false positives), (2) proving a full QA run succeeds with
sockets blocked, and (3) proving healthy outputs never trip the provider
denylist and never pretend a real video exists.
"""

from __future__ import annotations

import io
import tokenize
from pathlib import Path

import pytest

from agents.qa.video.models import VideoQAConfig
from agents.qa.video.validator import evaluate_video

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "agents" / "qa" / "video"
PACKAGE_PY = sorted(str(p) for p in PACKAGE_DIR.glob("*.py"))

FORBIDDEN_MODULES = ("requests", "httpx", "urllib", "urllib3", "socket", "http.client")
FORBIDDEN_HARDWARE = ("torch", "cuda", "gpu", "tpu", "vram", "cuDNN", "tensorflow")
FORBIDDEN_ENV_READS = ("os.environ", "os.getenv", "getenv(")
FORBIDDEN_SECRET_NAMES = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "HUGGINGFACE_TOKEN",
    "REPLICATE_API_TOKEN",
)
# Stage 9 (video assembly/editing) is explicitly out of scope for Stage 8.
FORBIDDEN_STAGE9 = ("video_assembly", "video_editing", "ffmpeg")


def code_tokens(path: str) -> str:
    """Return normalized source restricted to executable code tokens.

    String literals (docstrings, denylist vocabulary) and comments are dropped:
    they are data, not behavior, and must not trip the audit.
    """
    text = ""
    for token in tokenize.tokenize(io.BytesIO(Path(path).read_bytes()).readline):
        if token.type in (tokenize.NAME, tokenize.OP, tokenize.NUMBER):
            text += token.string
    return text.lower()


class TestStaticAudit:
    @pytest.mark.parametrize("path", PACKAGE_PY)
    def test_no_network_imports(self, path):
        source = code_tokens(path)
        for module in FORBIDDEN_MODULES:
            assert module not in source, f"{path} references network module '{module}'"

    @pytest.mark.parametrize("path", PACKAGE_PY)
    def test_no_hardware_acceleration_references(self, path):
        source = code_tokens(path)
        for term in FORBIDDEN_HARDWARE:
            assert term not in source, f"{path} references hardware term '{term}'"

    @pytest.mark.parametrize("path", PACKAGE_PY)
    def test_no_secret_or_env_reads(self, path):
        source = code_tokens(path)
        for term in FORBIDDEN_ENV_READS:
            assert term not in source, f"{path} reads environment via '{term}'"
        raw = Path(path).read_text(encoding="utf-8").upper()
        for name in FORBIDDEN_SECRET_NAMES:
            assert name not in raw, f"{path} references secret name '{name}'"

    @pytest.mark.parametrize("path", PACKAGE_PY)
    def test_no_stage9_references(self, path):
        source = code_tokens(path)
        for term in FORBIDDEN_STAGE9:
            assert term not in source, f"{path} references Stage-9 term '{term}'"

    def test_audit_detects_real_forbidden_imports(self, tmp_path):
        sneaky = tmp_path / "sneaky.py"
        sneaky.write_text(
            "import requests\n"
            "os.getenv('OPENAI_API_KEY')\n"
            "import ffmpeg\n"
            "import socket\n",
            encoding="utf-8",
        )
        source = code_tokens(str(sneaky))
        assert "requests" in source
        assert "os.getenv" in source
        assert "ffmpeg" in source
        assert "socket" in source
        raw = sneaky.read_text(encoding="utf-8").upper()
        assert "OPENAI_API_KEY" in raw

    def test_package_files_exist(self):
        expected = {
            "__init__.py",
            "__main__.py",
            "checks.py",
            "models.py",
            "stage.py",
            "validator.py",
        }
        assert expected <= {Path(p).name for p in PACKAGE_PY}


class TestNetworkFreedom:
    def test_qa_succeeds_with_sockets_blocked(self, monkeypatch):
        def deny(*args, **kwargs):
            raise RuntimeError("network access is forbidden in the offline stage")

        monkeypatch.setattr("socket.socket", deny)
        monkeypatch.setattr("socket.create_connection", deny)
        from video_qa_test_utils import healthy_pair

        plan, result = healthy_pair()
        qa = evaluate_video(result, plan, config=VideoQAConfig(), run_id="vg-demo")
        assert qa.overall_verdict == "PASS"

    def test_qa_never_writes_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from video_qa_test_utils import healthy_pair

        plan, result = healthy_pair()
        evaluate_video(result, plan, config=VideoQAConfig(), run_id="vg-demo")
        assert len(list(tmp_path.iterdir())) == 0


class TestProviderIndependence:
    def test_healthy_qa_artifact_never_trips_denylist(self):
        from agents.video_generation.validator import provider_isolation_hits
        from video_qa_test_utils import healthy_pair

        plan, result = healthy_pair()
        qa = evaluate_video(result, plan, config=VideoQAConfig(), run_id="vg-demo")
        payload = dict(qa.to_dict()["data"])
        payload.pop("provenance", None)
        hits = provider_isolation_hits(payload, VideoQAConfig().provider_isolation_denylist)
        assert hits == ()

    def test_healthy_mock_artifacts_never_pretend_to_be_real(self):
        from video_qa_test_utils import healthy_pair

        plan, result = healthy_pair()
        for row in result.scenes:
            assert row.artifact.mock is True
            assert row.artifact.uri == ""
        qa = evaluate_video(result, plan, config=VideoQAConfig(), run_id="vg-demo")
        assert qa.overall_verdict == "PASS"