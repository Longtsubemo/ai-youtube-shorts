"""Stage-9 safety regression tests.

The policy stage must stay a pure offline boundary, exactly like Stages 7 and
8: no network, no GPU/CUDA, no API keys, no env reads, no provider coupling, no
file writes, no Stage-10/11 work (upload/scheduling/analytics/OAuth), and no
YouTube-side calls. These tests (1) statically audit the package source on
executable-code tokens only (so docstrings/notes in string literals are not
false positives), (2) prove a full evaluation succeeds with sockets blocked,
and (3) prove the policy artifact is deterministic and self-contained.
"""

from __future__ import annotations

import io
import tokenize
from pathlib import Path

import pytest

from agents.qa.policy.validator import evaluate_policy
from policy_qa_test_utils import committed_inputs

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "agents" / "qa" / "policy"
PACKAGE_PY = sorted(str(p) for p in PACKAGE_DIR.glob("*.py"))

FORBIDDEN_MODULES = ("requests", "httpx", "urllib", "urllib3", "socket", "http.client")
FORBIDDEN_HARDWARE = ("torch", "cuda", "gpu", "tpu", "vram", "cuDNN", "tensorflow")
FORBIDDEN_ENV_READS = ("os.environ", "os.getenv", "getenv(")
FORBIDDEN_SECRET_NAMES = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "HUGGINGFACE_TOKEN",
    "REPLICATE_API_TOKEN",
    "YOUTUBE_API_KEY",
    "GOOGLE_API_KEY",
)
# Stages 10 (human approval) and 11 (YouTube upload/scheduling/analytics) are
# explicitly OUT OF SCOPE for the Stage-9 gate.
FORBIDDEN_STAGE10_11 = (
    "oauth",
    "youtube_upload",
    "upload_video",
    "publish_to_youtube",
    "channel_management",
    "video_scheduling",
    "live_stream",
    "analytics_ingest",
)


def code_tokens(path: str) -> str:
    """Return normalized source restricted to executable code tokens.

    String literals (docstrings, marker vocabulary, reviewer notes) are data,
    not behavior, so they never trip the audit.
    """
    text = ""
    for token in tokenize.tokenize(io.BytesIO(Path(path).read_bytes()).readline):
        if token.type in (tokenize.NAME, tokenize.OP, tokenize.NUMBER):
            text += token.string
    return text.lower()


def name_tokens(path: str) -> set[str]:
    """The package's identifier tokens (NAME), lowercased."""
    names = set()
    for token in tokenize.tokenize(io.BytesIO(Path(path).read_bytes()).readline):
        if token.type == tokenize.NAME:
            names.add(token.string.lower())
    return names


class TestStaticAudit:
    @pytest.mark.parametrize("path", PACKAGE_PY)
    def test_no_network_imports(self, path):
        source = code_tokens(path)
        for module in FORBIDDEN_MODULES:
            assert module not in source, f"{path} references network module '{module}'"

    @pytest.mark.parametrize("path", PACKAGE_PY)
    def test_no_hardware_acceleration_references(self, path):
        # Identifier-token based: the concatenated-stream form of the generator
        # can produce accidental substrings (e.g. "tpu" inside "purpose"), so
        # hardware terms are only flagged when they appear inside an identifier.
        names = name_tokens(path)
        for term in FORBIDDEN_HARDWARE:
            assert not any(term in name for name in names), (
                f"{path} references hardware term '{term}'"
            )

    @pytest.mark.parametrize("path", PACKAGE_PY)
    def test_no_secret_or_env_reads(self, path):
        source = code_tokens(path)
        for term in FORBIDDEN_ENV_READS:
            assert term not in source, f"{path} reads environment via '{term}'"
        raw = Path(path).read_text(encoding="utf-8").upper()
        for name in FORBIDDEN_SECRET_NAMES:
            assert name not in raw, f"{path} references secret name '{name}'"

    @pytest.mark.parametrize("path", PACKAGE_PY)
    def test_no_stage_10_11_references(self, path):
        source = code_tokens(path)
        for term in FORBIDDEN_STAGE10_11:
            assert term not in source, f"{path} references Stage-10/11 term '{term}'"

    @pytest.mark.parametrize("path", PACKAGE_PY)
    def test_identifier_tokens_are_ascii(self, path):
        # Docstrings/conventions allow non-ASCII (e.g. "§11" page refs, matching
        # other QA stages), but executable identifiers must stay ASCII.
        names = name_tokens(path)
        assert all(name.isascii() for name in names), f"{path} has non-ASCII identifiers"

    def test_audit_detects_real_forbidden_imports(self, tmp_path):
        sneaky = tmp_path / "sneaky.py"
        sneaky.write_text(
            "import requests\n"
            "os.getenv('YOUTUBE_API_KEY')\n"
            "import socket\n"
            "def upload_video(): ...\n",
            encoding="utf-8",
        )
        source = code_tokens(str(sneaky))
        assert "requests" in source
        assert "os.getenv" in source
        assert "socket" in source
        assert "upload_video" in source
        raw = sneaky.read_text(encoding="utf-8").upper()
        assert "YOUTUBE_API_KEY" in raw

    def test_package_files_exist(self):
        expected = {
            "__init__.py",
            "__main__.py",
            "checks.py",
            "classifier.py",
            "models.py",
            "rules.py",
            "stage.py",
            "validator.py",
        }
        assert expected <= {Path(p).name for p in PACKAGE_PY}


class TestNetworkFreedom:
    def test_policy_gate_succeeds_with_sockets_blocked(self, monkeypatch):
        def deny(*args, **kwargs):
            raise RuntimeError("network access is forbidden in the offline stage")

        monkeypatch.setattr("socket.socket", deny)
        monkeypatch.setattr("socket.create_connection", deny)
        result = evaluate_policy(committed_inputs(), run_id="pol-demo")
        assert result.overall_verdict == "PASS"

    def test_policy_gate_never_writes_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        evaluate_policy(committed_inputs(), run_id="pol-demo")
        assert len(list(tmp_path.iterdir())) == 0


class TestDeterminism:
    def test_policy_artifact_is_deterministic_given_fixed_clock(self):
        first = evaluate_policy(
            committed_inputs(), run_id="pol-demo", created_at="2026-09-20T08:00:00+00:00"
        )
        second = evaluate_policy(
            committed_inputs(), run_id="pol-demo", created_at="2026-09-20T08:00:00+00:00"
        )
        assert first.to_json() == second.to_json()

    def test_health_state_stays_reproducible(self):
        result = evaluate_policy(committed_inputs(), run_id="pol-demo")
        assert result.risk_summary.flagged_categories == ()
        assert result.overall_verdict == "PASS"