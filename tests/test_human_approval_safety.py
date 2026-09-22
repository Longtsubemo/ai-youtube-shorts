"""Stage-10 safety regression tests.

The human-approval stage must stay a pure offline boundary, just like the QA
stages before it: no network, no GPU/CUDA, no API keys, no provider coupling,
no file writes, and no Stage-11 (YouTube upload/scheduling) work. These tests
enforce that by (1) statically auditing the package source (executable-code
tokens only, so docstrings/denylist vocabulary in string literals are not
false positives), (2) proving an approval run succeeds with sockets blocked,
and (3) proving an approval never fabricates automation, never trips the
provider denylist, and marks the endpoint as a MANUAL upload only.
"""

from __future__ import annotations

import io
import json
import tokenize
from pathlib import Path

import pytest

from agents.human_approval.models import HumanApprovalConfig
from agents.human_approval.validator import evaluate_human_approval
from human_approval_test_utils import GENESIS, build_inputs

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "agents" / "human_approval"
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
# Stage 11 (YouTube upload/scheduling) is explicitly out of scope for Stage 10.
FORBIDDEN_STAGE11 = (
    "youtube_upload",
    "upload_video",
    "publish_to_youtube",
    "channel_management",
    "video_scheduling",
    "live_stream",
    "analytics_ingest",
    "googleapiclient",
    "oauth2",
)


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


def healthy_approval_inputs():
    return build_inputs()


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
    def test_no_stage11_references(self, path):
        source = code_tokens(path)
        for term in FORBIDDEN_STAGE11:
            assert term not in source, f"{path} references Stage-11 term '{term}'"

    def test_audit_detects_real_forbidden_imports(self, tmp_path):
        sneaky = tmp_path / "sneaky.py"
        sneaky.write_text(
            "import requests\n"
            "os.getenv('OPENAI_API_KEY')\n"
            "import socket\n"
            "def x(): youtube_upload()\n",
            encoding="utf-8",
        )
        source = code_tokens(str(sneaky))
        assert "requests" in source
        assert "os.getenv" in source
        assert "socket" in source
        assert "youtube_upload" in source
        raw = sneaky.read_text(encoding="utf-8").upper()
        assert "OPENAI_API_KEY" in raw

    def test_package_files_exist(self):
        expected = {
            "__init__.py",
            "__main__.py",
            "models.py",
            "stage.py",
            "validator.py",
        }
        assert expected <= {Path(p).name for p in PACKAGE_PY}


class TestNetworkFreedom:
    def test_approval_succeeds_with_sockets_blocked(self, monkeypatch):
        def deny(*args, **kwargs):
            raise RuntimeError("network access is forbidden in the offline stage")

        monkeypatch.setattr("socket.socket", deny)
        monkeypatch.setattr("socket.create_connection", deny)
        approval = evaluate_human_approval(
            healthy_approval_inputs(),
            config=HumanApprovalConfig(),
            decision="APPROVED",
            approver=_make_approver(),
            run_id="offline-test",
            created_at=GENESIS,
        )
        assert approval.is_approved

    def test_approval_never_writes_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        evaluate_human_approval(
            healthy_approval_inputs(),
            config=HumanApprovalConfig(),
            decision="APPROVED",
            approver=_make_approver(),
            run_id="offline-test",
            created_at=GENESIS,
        )
        assert len(list(tmp_path.iterdir())) == 0


def _make_approver():
    from agents.human_approval.models import Approver

    return Approver(approver_id="safety-reviewer", role="audit")


class TestProviderIndependence:
    def test_approval_never_fabricates_a_provider(self):
        approval = evaluate_human_approval(
            healthy_approval_inputs(),
            config=HumanApprovalConfig(),
            decision="APPROVED",
            approver=_make_approver(),
            run_id="purity-test",
            created_at=GENESIS,
        )
        # The human is not an AI model/provider: envelope and provenance are
        # both explicitly null for provider/model.
        assert approval.produced_by["provider"] is None
        assert approval.produced_by["model"] is None
        assert approval.provenance.provider is None
        assert approval.provenance.model is None
        assert approval.provenance.generated_by == "human_approval_gate_v1"
        assert approval.provenance.source == "HumanReviewer"

    def test_payload_never_mentions_stage11_endpoint(self):
        approval = evaluate_human_approval(
            healthy_approval_inputs(),
            config=HumanApprovalConfig(),
            decision="APPROVED",
            approver=_make_approver(),
            run_id="purity-test",
            created_at=GENESIS,
        )
        payload = json.dumps(approval.to_dict()["data"]).lower()
        for term in FORBIDDEN_STAGE11 + ("youtube", "api_key"):
            assert term not in payload, f"approval payload mentions '{term}'"

    def test_endpoint_is_manual_upload_only(self):
        approval = evaluate_human_approval(
            healthy_approval_inputs(),
            config=HumanApprovalConfig(),
            decision="APPROVED",
            approver=_make_approver(),
            run_id="purity-test",
            created_at=GENESIS,
        )
        assert approval.manual_upload.ready is True
        assert "manual upload" in approval.manual_upload.note.lower()
        assert approval.manual_upload.title == "Why octopuses have three hearts"


class TestNeverFabricatedApproval:
    def test_non_approved_decisions_never_claim_approval(self):
        for decision in ("CHANGES_REQUESTED", "REJECTED"):
            approval = evaluate_human_approval(
                healthy_approval_inputs(),
                config=HumanApprovalConfig(),
                decision=decision,
                approver=_make_approver(),
                notes="recorded for audit" if decision == "CHANGES_REQUESTED" else "no",
                run_id="purity-test",
                created_at=GENESIS,
            )
            assert approval.is_approved is False
            assert approval.manual_upload.ready is False
            assert approval.gates_cleared

    def test_approved_records_every_reviewed_digest(self):
        approval = evaluate_human_approval(
            healthy_approval_inputs(),
            config=HumanApprovalConfig(),
            decision="APPROVED",
            approver=_make_approver(),
            run_id="purity-test",
            created_at=GENESIS,
        )
        # Every bound digest is a full canonical-content SHA-256 -> the record
        # can prove exactly which versions were reviewed.
        for artifact in approval.reviewed_artifacts:
            assert len(artifact.content_sha256) == 64
            int(artifact.content_sha256, 16)