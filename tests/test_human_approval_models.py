"""Stage-10 model tests: schema, decisions, digest binding, round trips.

The Stage-10 artifact must never reuse the QA verdict vocabulary (PASS /
REVIEW / REJECT), must bind each reviewed artifact to (stable id, content
SHA-256), and must round-trip the envelope + data payload losslessly.
"""

from __future__ import annotations

import json

import pytest

from agents.human_approval.models import (
    DECISIONS,
    SCHEMA,
    SCHEMA_VERSION,
    STAGE,
    STATE_STAGE,
    Approver,
    BlockingCondition,
    HumanApprovalConfig,
    HumanApprovalError,
    HumanApprovalResult,
    ManualUploadReadiness,
    OverriddenVerdict,
    RequestedChange,
    ReviewedArtifact,
    canonical_json_bytes,
    content_sha256,
    human_approval_id_for,
)
from human_approval_test_utils import GENESIS, healthy_approval

from pipeline.state import Stage


class TestIds:
    def test_stable_id_prefix_and_determinism(self):
        one = human_approval_id_for("run-1", "pp_e99181e73b5d")
        assert one.startswith("ha_")
        assert len(one) == 3 + 12
        assert one == human_approval_id_for("run-1", "pp_e99181e73b5d")

    def test_id_changes_with_run_or_target(self):
        base = human_approval_id_for("run-1", "pp_a")
        assert base != human_approval_id_for("run-2", "pp_a")
        assert base != human_approval_id_for("run-1", "pp_b")

    def test_id_rejects_path_unsafe_run_id(self):
        with pytest.raises(HumanApprovalError, match="run_id"):
            human_approval_id_for("../escape", "pp_a")

    def test_id_rejects_empty_target(self):
        with pytest.raises(HumanApprovalError, match="target_ref"):
            human_approval_id_for("run-1", "")

    def test_record_uses_stable_id(self):
        result = healthy_approval(run_id="mock")
        assert result.approval_id == human_approval_id_for("mock", result.target_ref)


class TestContentDigest:
    def test_digest_is_sha256(self):
        digest = content_sha256({"a": 1})
        assert len(digest) == 64
        int(digest, 16)

    def test_canonical_ignores_key_order(self):
        first = content_sha256({"a": 1, "b": {"c": 2}})
        second = content_sha256({"b": {"c": 2}, "a": 1})
        assert first == second

    def test_canonical_bytes_stable(self):
        encoded = json.dumps({"z": 1, "a": [1, 2]}, sort_keys=True, separators=(",", ":"))
        assert canonical_json_bytes({"a": [1, 2], "z": 1}) == encoded.encode("utf-8")

    def test_digest_is_content_sensitive(self):
        assert content_sha256({"note": "a"}) != content_sha256({"note": "b"})


class TestReviewedArtifact:
    def test_round_trip(self):
        ra = ReviewedArtifact(
            role="script",
            artifact_id="sc_9fc41d7a46b7",
            content_sha256="a" * 64,
            schema="Script",
        ).validate()
        restored = ReviewedArtifact.from_dict(ra.to_dict())
        assert restored == ra

    def test_unknown_role_rejected(self):
        with pytest.raises(HumanApprovalError, match="unknown reviewed-artifact role"):
            ReviewedArtifact(
                role="bogus", artifact_id="x", content_sha256="a" * 64
            ).validate()

    def test_missing_artifact_id_rejected(self):
        with pytest.raises(HumanApprovalError, match="missing required field"):
            ReviewedArtifact(role="script", artifact_id="", content_sha256="a" * 64).validate()

    def test_bad_digest_rejected(self):
        with pytest.raises(HumanApprovalError, match="content_sha256"):
            ReviewedArtifact(role="script", artifact_id="sc_x", content_sha256="zz").validate()

    def test_non_text_role_is_rejected_cleanly(self):
        with pytest.raises(HumanApprovalError, match="unknown reviewed-artifact role"):
            ReviewedArtifact(role=[], artifact_id="sc_x", content_sha256="a" * 64).validate()

    def test_wrong_schema_for_role_rejected(self):
        with pytest.raises(HumanApprovalError, match="does not match role"):
            ReviewedArtifact(
                role="script",
                artifact_id="sc_x",
                content_sha256="a" * 64,
                schema="ScriptQAResult",
            ).validate()


class TestApprover:
    def test_round_trip(self):
        approver = Approver(approver_id="alice", role="content lead").validate()
        assert Approver.from_dict(approver.to_dict()) == approver

    def test_approver_id_required(self):
        with pytest.raises(HumanApprovalError, match="approver_id"):
            Approver(approver_id="").validate()


class TestOverriddenVerdict:
    def test_round_trip(self):
        override = OverriddenVerdict(
            gate="video_qa",
            check_id="Q-PROVIDER_SUBJECT",
            verdict="REVIEW",
            decision="APPROVED",
            justification="reviewed and accepted on qa_x",
            artifact_id="qa_x",
        ).validate()
        assert OverriddenVerdict.from_dict(override.to_dict()) == override

    def test_invalid_gate_rejected(self):
        with pytest.raises(HumanApprovalError, match="override gate"):
            OverriddenVerdict(
                gate="script", justification="nope"
            ).validate()

    def test_justification_required(self):
        with pytest.raises(HumanApprovalError, match="justification"):
            OverriddenVerdict(gate="video_qa", justification="").validate()


class TestBlockingConditionAndChanges:
    def test_blocking_condition_round_trip(self):
        condition = BlockingCondition(code="policy_qa_hard_reject", satisfied=False, detail="REJECT")
        assert BlockingCondition.from_dict(condition.to_dict()) == condition

    def test_requested_change_round_trip(self):
        change = RequestedChange(issue_ref="P13", requirement="add the disclosure").validate()
        assert RequestedChange.from_dict(change.to_dict()) == change

    def test_requested_change_requires_requirement(self):
        with pytest.raises(HumanApprovalError, match="requirement"):
            RequestedChange(requirement="").validate()


class TestManualUploadReadiness:
    def test_round_trip(self):
        readiness = ManualUploadReadiness(
            ready=True,
            required=True,
            note="manual upload only",
            artifacts=("pr_x", "qa_y"),
            title="Why octopuses have three hearts",
        )
        assert ManualUploadReadiness.from_dict(readiness.to_dict()) == readiness

    def test_note_is_manual_never_automated(self):
        note = ManualUploadReadiness(ready=True).note.lower()
        assert "manual upload" in note
        assert "youtube" not in note
        assert "api" not in note


class TestResultEnvelope:
    def test_config_rejects_non_text_role_cleanly(self):
        with pytest.raises(HumanApprovalError, match="known role names"):
            HumanApprovalConfig(required_roles=([],)).validate()

    def test_healthy_approval_round_trip_json(self):
        result = healthy_approval(
            decision="APPROVED",
            notes="all good",
            run_id="mock",
            created_at=GENESIS,
        )
        restored = HumanApprovalResult.from_json(result.to_json())
        assert restored == result
        assert restored.to_json() == result.to_json()

    def test_changes_requested_round_trip(self):
        result = healthy_approval(
            decision="CHANGES_REQUESTED",
            notes="fix the disclosure first",
            changes=(RequestedChange(issue_ref="P13", requirement="add disclosure"),),
            run_id="mock",
            created_at=GENESIS,
        )
        restored = HumanApprovalResult.from_json(result.to_json())
        assert restored.decision == "CHANGES_REQUESTED"
        assert restored.changes_requested[0].requirement == "add disclosure"
        assert restored.notes == "fix the disclosure first"

    def test_envelope_constants(self):
        assert SCHEMA == "HumanApprovalResult"
        assert SCHEMA_VERSION == 1
        assert STAGE == "human_approval"
        assert STATE_STAGE == Stage.PUBLISH
        assert healthy_approval().schema == SCHEMA

    def test_decision_vocabulary_never_reuses_qa_verdicts(self):
        for word in ("PASS", "REVIEW", "REJECT"):
            assert word not in DECISIONS, f"{word} must not be a human decision"
        assert DECISIONS == ("PENDING", "APPROVED", "CHANGES_REQUESTED", "REJECTED")

    def test_approve_with_unsatisfied_condition_fails_validation(self):
        import dataclasses

        result = healthy_approval()
        broken = dataclasses.replace(
            result,
            blocking_conditions=(
                BlockingCondition(code="policy_qa_hard_reject", satisfied=False, detail="REJECT"),
            ),
        )
        with pytest.raises(HumanApprovalError, match="blocking conditions"):
            broken.validate()

    def test_changes_requested_requires_changes_or_notes(self):
        with pytest.raises(HumanApprovalError, match="CHANGES_REQUESTED needs"):
            healthy_approval(decision="CHANGES_REQUESTED", notes="")

    def test_pending_is_a_model_value_but_not_an_approval(self):
        approved = healthy_approval()  # APPROVED, gates clear
        assert approved.is_approved
        assert approved.gates_cleared

    def test_missing_provenance_rejected(self):
        import dataclasses

        result = healthy_approval()
        broken = dataclasses.replace(result, provenance=None)
        with pytest.raises(HumanApprovalError, match="provenance is required"):
            broken.validate()

    def test_supersede_self_rejected(self):
        import dataclasses

        result = healthy_approval(run_id="mock")
        broken = dataclasses.replace(
            result, supersedes_approval_id=result.approval_id
        )
        with pytest.raises(HumanApprovalError, match="supersede itself"):
            broken.validate()

    def test_result_requires_complete_condition_set(self):
        import dataclasses

        result = healthy_approval()
        broken = dataclasses.replace(
            result,
            blocking_conditions=result.blocking_conditions[:1],
        )
        with pytest.raises(HumanApprovalError, match="complete gate set"):
            broken.validate()

    def test_result_rejects_tampered_approval_id(self):
        import dataclasses

        result = healthy_approval()
        broken = dataclasses.replace(result, approval_id="ha_" + "0" * 12)
        with pytest.raises(HumanApprovalError, match="does not match"):
            broken.validate()

    def test_result_rejects_tampered_manual_package(self):
        import dataclasses

        result = healthy_approval()
        broken = dataclasses.replace(
            result,
            manual_upload=dataclasses.replace(
                result.manual_upload,
                artifacts=(result.manual_upload.artifacts[0],),
            ),
        )
        with pytest.raises(HumanApprovalError, match="manual_upload.artifacts"):
            broken.validate()

    def test_malformed_provenance_is_wrapped(self):
        payload = healthy_approval().to_dict()
        payload["data"]["provenance"] = None
        with pytest.raises(HumanApprovalError, match="provenance must be a mapping"):
            HumanApprovalResult.from_dict(payload)
