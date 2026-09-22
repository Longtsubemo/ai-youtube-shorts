"""Stage-10 validator tests: hard gates, overrides, digests, staleness.

The gate's guarantees (docs/POLICY_AND_SAFETY.md):
- APPROVED requires EVERY deterministic hard gate to be satisfied (fail closed).
- CHANGES_REQUESTED / REJECTED are always valid human decisions and record the
  evaluated blocking conditions for audit.
- Soft-REVIEW findings may be overridden, but only with an explicit,
  version-bound override; a hard REJECT can never be overridden.
- Every reviewed artifact is bound to (stable id, content SHA-256), so any
  upstream change afterwards makes the approval STALE.
"""

from __future__ import annotations

import pytest

from agents.human_approval.models import (
    Approver,
    BlockingCondition,
    HumanApprovalConfig,
    HumanApprovalError,
    HumanApprovalResult,
    OverriddenVerdict,
    RequestedChange,
    content_sha256,
    human_approval_id_for,
)
from agents.human_approval.validator import (
    approval_is_effective,
    build_blocking_conditions,
    check_approval_freshness,
    evaluate_human_approval,
    is_superseded_by,
)
from human_approval_test_utils import (
    GENESIS,
    artifact_map,
    build_config,
    build_inputs,
    healthy_approval,
    policy_verdict,
    tampered_video_result,
    video_qa_verdict,
    video_result_status,
    wrong_id_video_result,
)


class TestBlockingConditions:
    def test_healthy_baseline_clears_every_gate(self):
        result = healthy_approval(decision="APPROVED")
        assert result.is_approved
        assert result.gates_cleared
        codes = {condition.code for condition in result.blocking_conditions}
        assert codes == {
            "required_artifacts",
            "policy_qa_hard_reject",
            "video_qa_hard_reject",
            "video_generation_completed",
            "review_overrides",
        }

    def test_build_blocking_conditions_is_deterministic(self):
        inputs = build_inputs()
        config = HumanApprovalConfig()
        first = build_blocking_conditions(inputs, config, (), "APPROVED")
        second = build_blocking_conditions(inputs, config, (), "APPROVED")
        assert [c.to_dict() for c in first] == [c.to_dict() for c in second]


class TestHardGatesBlockApproved:
    def test_stage9_hard_reject_blocks_approve(self):
        with pytest.raises(HumanApprovalError, match="policy_qa_hard_reject"):
            healthy_approval(decision="APPROVED", inputs=build_inputs(policy_qa=policy_verdict("REJECT")))

    def test_stage8_hard_reject_blocks_approve(self):
        with pytest.raises(HumanApprovalError, match="video_qa_hard_reject"):
            healthy_approval(decision="APPROVED", inputs=build_inputs(video_qa=video_qa_verdict("REJECT")))

    def test_uncompleted_video_blocks_approve(self):
        with pytest.raises(HumanApprovalError, match="video_generation_completed"):
            healthy_approval(
                decision="APPROVED",
                inputs=build_inputs(video_result=video_result_status("failed")),
            )

    def test_stage8_hard_reject_is_never_overrideable(self):
        # A hard REJECT never becomes approvable: the override is rejected
        # outright (only soft REVIEW verdicts are overrideable).
        override = OverriddenVerdict(
            gate="video_qa", verdict="REJECT", justification="rescind the reject"
        )
        with pytest.raises(HumanApprovalError, match="not REVIEW"):
            healthy_approval(
                decision="APPROVED",
                overrides=(override,),
                inputs=build_inputs(video_qa=video_qa_verdict("REJECT")),
            )

    def test_override_for_non_review_verdict_is_rejected(self):
        override = OverriddenVerdict(
            gate="policy_qa", justification="accepted, no review existed"
        )
        with pytest.raises(HumanApprovalError, match="not REVIEW"):
            healthy_approval(
                decision="APPROVED", overrides=(override,)
            )

    def test_multiple_hard_rejects_listed(self):
        with pytest.raises(HumanApprovalError, match="policy_qa_hard_reject.*video_qa_hard_reject"):
            healthy_approval(
                decision="APPROVED",
                inputs=build_inputs(
                    policy_qa=policy_verdict("REJECT"),
                    video_qa=video_qa_verdict("REJECT"),
                ),
            )


class TestMissingRequiredArtifactsFailClosed:
    def test_missing_policy_result_blocks(self):
        for decision in ("APPROVED", "CHANGES_REQUESTED", "REJECTED"):
            with pytest.raises(HumanApprovalError, match="missing required policy_qa"):
                healthy_approval(decision=decision, inputs=build_inputs(policy_qa=None))

    def test_missing_video_qa_blocks(self):
        with pytest.raises(HumanApprovalError, match="missing required video_qa"):
            healthy_approval(decision="APPROVED", inputs=build_inputs(video_qa=None))

    def test_missing_video_result_blocks(self):
        with pytest.raises(HumanApprovalError, match="missing required video_result"):
            healthy_approval(decision="APPROVED", inputs=build_inputs(video_result=None))

    def test_missing_plan_blocks(self):
        for decision in ("APPROVED", "CHANGES_REQUESTED", "REJECTED"):
            with pytest.raises(HumanApprovalError, match="missing required production_plan"):
                healthy_approval(decision=decision, inputs=build_inputs(plan=None))

    def test_missing_script_blocks(self):
        with pytest.raises(HumanApprovalError, match="missing required script"):
            healthy_approval(decision="APPROVED", inputs=build_inputs(script=None))

    def test_optional_roles_ok_when_absent(self):
        result = healthy_approval(
            decision="APPROVED", inputs=build_inputs(script_qa=None, fact_check=None)
        )
        roles = {artifact.role for artifact in result.reviewed_artifacts}
        assert "script_qa" not in roles
        assert "fact_check" not in roles
        assert result.is_approved


class TestChangesRequestedAndRejectedAllowed:
    def test_changes_requested_allowed_under_stage8_reject(self):
        result = healthy_approval(
            decision="CHANGES_REQUESTED",
            notes="fix the QA blockers first",
            inputs=build_inputs(video_qa=video_qa_verdict("REJECT")),
        )
        assert not result.is_approved
        assert not result.gates_cleared
        conditions = {c.code: c for c in result.blocking_conditions}
        assert conditions["video_qa_hard_reject"].satisfied is False

    def test_changes_requested_allowed_under_stage9_reject(self):
        result = healthy_approval(
            decision="CHANGES_REQUESTED",
            changes=(RequestedChange(issue_ref="P06", requirement="fix the framing"),),
            inputs=build_inputs(policy_qa=policy_verdict("REJECT")),
        )
        assert result.decision == "CHANGES_REQUESTED"
        assert result.changes_requested[0].requirement == "fix the framing"

    def test_rejected_allowed_under_hard_reject(self):
        result = healthy_approval(
            decision="REJECTED",
            notes="not fit to publish",
            inputs=build_inputs(video_qa=video_qa_verdict("REJECT")),
        )
        assert result.decision == "REJECTED"
        assert not result.is_approved
        assert result.manual_upload.ready is False

    def test_changes_requested_with_unfinished_video_is_recorded(self):
        result = healthy_approval(
            decision="CHANGES_REQUESTED",
            notes="finish rendering first",
            inputs=build_inputs(video_result=video_result_status("failed")),
        )
        assert result.decision == "CHANGES_REQUESTED"


class TestSoftReviewOverrides:
    def test_policy_review_without_override_blocks_approve(self):
        with pytest.raises(HumanApprovalError, match="review_overrides"):
            healthy_approval(
                decision="APPROVED",
                inputs=build_inputs(policy_qa=policy_verdict("REVIEW")),
            )

    def test_policy_review_with_override_approves(self):
        override = OverriddenVerdict(
            gate="policy_qa",
            check_id="P15",
            verdict="REVIEW",
            decision="APPROVED",
            justification="reviewed the prior-gate warning on pr_9b7d098451d3 and accept",
        )
        result = healthy_approval(
            decision="APPROVED",
            overrides=(override,),
            inputs=build_inputs(policy_qa=policy_verdict("REVIEW")),
        )
        assert result.is_approved
        conditions = {c.code: c for c in result.blocking_conditions}
        assert conditions["review_overrides"].satisfied is True

    def test_video_review_with_override_approves_and_binds_version(self):
        override = OverriddenVerdict(
            gate="video_qa",
            check_id="Q-PROVIDER_SUBJECT",
            verdict="REVIEW",
            justification="the disclosure warning was reviewed and accepted",
        )
        result = healthy_approval(
            decision="APPROVED",
            overrides=(override,),
            inputs=build_inputs(video_qa=video_qa_verdict("REVIEW")),
        )
        assert result.is_approved
        assert result.overridden_verdicts[0].artifact_id == "qa_9b7d098451d3"

    def test_both_reviews_need_both_overrides(self):
        policy_override = OverriddenVerdict(
            gate="policy_qa", justification="accepted"
        )
        with pytest.raises(HumanApprovalError, match="review_overrides"):
            healthy_approval(
                decision="APPROVED",
                overrides=(policy_override,),
                inputs=build_inputs(
                    policy_qa=policy_verdict("REVIEW"),
                    video_qa=video_qa_verdict("REVIEW"),
                ),
            )

    def test_override_disabled_when_configured_off(self):
        result = healthy_approval(
            decision="APPROVED",
            config=build_config(require_override_for_review=False),
            inputs=build_inputs(video_qa=video_qa_verdict("REVIEW")),
        )
        assert result.is_approved


class TestExactVersionBinding:
    def test_every_reviewed_artifact_is_digest_bound(self):
        result = healthy_approval(decision="APPROVED")
        current = artifact_map(build_inputs())
        for artifact in result.reviewed_artifacts:
            obj = current[artifact.role]
            assert artifact.artifact_id == getattr(obj, {
                "policy_qa": "policy_result_id",
                "video_qa": "qa_result_id",
                "video_result": "video_generation_result_id",
                "production_plan": "production_plan_id",
                "script": "script_id",
                "script_qa": "qa_result_id",
                "fact_check": "result_id",
            }[artifact.role])
            assert artifact.content_sha256 == content_sha256(obj.to_dict())

    def test_required_and_optional_roles_recorded(self):
        result = healthy_approval(decision="APPROVED")
        roles = [artifact.role for artifact in result.reviewed_artifacts]
        assert roles == [
            "policy_qa",
            "video_qa",
            "video_result",
            "production_plan",
            "script",
            "script_qa",
            "fact_check",
        ]

    def test_prior_gate_verdicts_recorded(self):
        result = healthy_approval(decision="APPROVED")
        assert result.prior_gate_verdicts["policy_qa"] == "PASS"
        assert result.prior_gate_verdicts["video_qa"] == "PASS"
        assert result.prior_gate_verdicts["video_generation"] == "completed"
        assert result.prior_gate_verdicts["script_qa"] == "PASS"


class TestProvenanceAndReadiness:
    def test_human_not_an_ai_provider(self):
        result = healthy_approval(decision="APPROVED")
        assert result.produced_by["provider"] is None
        assert result.produced_by["model"] is None
        assert result.produced_by["prompt_version"] == "human_approval.v1"
        assert result.provenance.generated_by == "human_approval_gate_v1"
        assert result.provenance.source == "HumanReviewer"
        assert result.provenance.provider is None
        assert result.provenance.model is None

    def test_manual_upload_ready_only_when_approved(self):
        approved = healthy_approval(decision="APPROVED")
        assert approved.manual_upload.ready is True
        assert approved.manual_upload.required is True
        assert "manual upload" in approved.manual_upload.note.lower()
        assert approved.manual_upload.title == "Why octopuses have three hearts"

        not_approved = healthy_approval(
            decision="CHANGES_REQUESTED", notes="fix first"
        )
        assert not_approved.manual_upload.ready is False

    def test_writable_decisions_only(self):
        with pytest.raises(HumanApprovalError, match="not a writable decision"):
            healthy_approval(decision="PENDING", notes="not fabricated")

    def test_bogus_decision_rejected(self):
        with pytest.raises(HumanApprovalError, match="writable"):
            healthy_approval(decision="MAYBE", notes="?")


class TestFreshnessAndStaleness:
    def test_fresh_when_unchanged(self):
        result = healthy_approval(decision="APPROVED")
        assert check_approval_freshness(result, artifact_map(build_inputs())) is True

    def test_stale_on_content_change(self):
        result = healthy_approval(decision="APPROVED")
        current = artifact_map(build_inputs(video_result=tampered_video_result()))
        assert check_approval_freshness(result, current) is False
        assert approval_is_effective(result, current) is False

    def test_stale_on_id_change(self):
        result = healthy_approval(decision="APPROVED")
        current = artifact_map(build_inputs(video_result=wrong_id_video_result()))
        assert check_approval_freshness(result, current) is False

    def test_stale_on_missing_role(self):
        result = healthy_approval(decision="APPROVED")
        current = artifact_map(build_inputs())
        del current["script_qa"]
        assert check_approval_freshness(result, current) is False

    def test_effective_only_when_approved_and_fresh(self):
        result = healthy_approval(decision="APPROVED")
        assert approval_is_effective(result, artifact_map(build_inputs())) is True
        not_approved = healthy_approval(decision="CHANGES_REQUESTED", notes="n")
        assert (
            approval_is_effective(not_approved, artifact_map(build_inputs())) is False
        )

    def test_superseded_not_effective(self):
        first = healthy_approval(run_id="run-a", decision="APPROVED", created_at=GENESIS)
        second = healthy_approval(
            run_id="run-b",
            decision="APPROVED",
            created_at=GENESIS,
            supersedes_approval_id=first.approval_id,
        )
        assert first.approval_id != second.approval_id
        assert is_superseded_by(first, second) is True
        assert (
            approval_is_effective(
                first, artifact_map(build_inputs()), later=second
            )
            is False
        )
        assert approval_is_effective(second, artifact_map(build_inputs())) is True


class TestDecisionModelDuties:
    def test_no_written_record_means_no_approval(self):
        # An APPROVED attempt that trips a gate raises (nothing written, no
        # output), so a missing artifact can never be read as an approval.
        with pytest.raises(HumanApprovalError):
            evaluate_human_approval(
                build_inputs(video_qa=video_qa_verdict("REJECT")),
                decision="APPROVED",
                approver=Approver(approver_id="alice", role="reviewer"),
                run_id="mock",
            )

    def test_approval_only_ever_over_the_policy_target(self):
        result = healthy_approval(decision="APPROVED")
        assert result.target_ref == "pp_e99181e73b5d"
        assert result.approval_id == human_approval_id_for("mock", "pp_e99181e73b5d")

    def test_changes_requested_carries_audit_blocking_conditions(self):
        result = healthy_approval(
            decision="CHANGES_REQUESTED",
            notes="fix disclosure",
            inputs=build_inputs(video_result=video_result_status("failed")),
        )
        assert any(
            c.code == "video_generation_completed" and not c.satisfied
            for c in result.blocking_conditions
        )