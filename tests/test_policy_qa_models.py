"""Stage-9 model/envelope tests (PolicyIssue, PolicyCheckRecord, PolicyQAResult).

Covers the DATA_CONTRACTS.md §11/§1 shape: schema header (schema/schema_version/
run_id/stage/produced_by) + data payload, stable `pr_` result ids, verdict/
severity/gate vocab containers, risk summary, config validation, and full JSON
round-trips.
"""

from __future__ import annotations

import pytest

from agents.qa.policy.models import (
    POLICY_QA_CONFIG_VERSION,
    POLICY_VERDICTS,
    PolicyCheckRecord,
    PolicyIssue,
    PolicyQAConfig,
    PolicyQAError,
    PolicyQAResult,
    RiskSummary,
    policy_result_id_for,
)
from agents.qa.policy.rules import POLICY_CHECK_ORDER, POLICY_RULES, POLICY_VERSION
from agents.qa.policy.validator import evaluate_policy
from pipeline.state import Stage
from policy_qa_test_utils import committed_inputs


class TestEnvelope:
    def test_stage_label_is_task_value(self):
        from agents.qa.policy.models import STAGE

        assert STAGE == "policy_qa"

    def test_state_mapping_reuses_quality_check(self):
        from agents.qa.policy.models import STATE_STAGE

        assert STATE_STAGE == Stage.QUALITY_CHECK

    def test_schema_constants(self):
        from agents.qa.policy.models import SCHEMA, SCHEMA_VERSION

        assert SCHEMA == "PolicyQAResult"
        assert SCHEMA_VERSION == 1

    def test_produced_by_model(self):
        from agents.qa.policy.models import PRODUCED_BY

        assert PRODUCED_BY["model"] == "PolicyQAValidator"

    def test_config_version_and_ruleset_version_are_distinct_but_recorded(self):
        assert POLICY_QA_CONFIG_VERSION == "policy_qa.v1"
        assert POLICY_VERSION == "policy_rules_v1"

    def test_registry_has_expected_rules(self):
        assert [rule.check_id for rule in POLICY_RULES] == [
            f"P{number:02d}" for number in range(1, 16)
        ]

    def test_check_order_matches_registry(self):
        assert POLICY_CHECK_ORDER == tuple(rule.check_id for rule in POLICY_RULES)


class TestPolicyResultId:
    def test_prefix_and_length(self):
        result_id = policy_result_id_for("run-1", "pp_e99181e73b5d")
        assert result_id.startswith("pr_")
        assert len(result_id) == 15  # pr_ + 12 hex

    def test_deterministic_identical_inputs(self):
        assert policy_result_id_for("run-1", "pp_e99181e73b5d") == policy_result_id_for(
            "run-1", "pp_e99181e73b5d"
        )

    def test_changes_with_run_or_target(self):
        base = policy_result_id_for("run-1", "pp_e99181e73b5d")
        assert policy_result_id_for("run-2", "pp_e99181e73b5d") != base
        assert policy_result_id_for("run-1", "pp_other000000") != base


class TestPolicyIssue:
    def test_round_trip(self):
        issue = PolicyIssue(
            check_id="P05",
            severity="ERROR",
            gate="hard",
            message="Content claims real footage that the record proves is mock.",
            affected_scene_ids=("scn_01",),
            affected_asset_ids=("mockva_00000001",),
        ).validate()
        restored = PolicyIssue.from_dict(issue.to_dict())
        assert restored == issue

    def test_refs_merge_all_affected_ids_deduplicated(self):
        issue = PolicyIssue(
            check_id="P03",
            message="overstatement",
            affected_line_ids=("ln_02",),
            affected_claim_ids=("cl_97628e259b", "cl_b4ca50a10d"),
        )
        assert issue.refs == ("ln_02", "cl_97628e259b", "cl_b4ca50a10d")

    def test_is_hard_error(self):
        assert PolicyIssue(check_id="P01", severity="ERROR", gate="hard", message="x").is_hard_error
        assert not PolicyIssue(check_id="P01", severity="ERROR", gate="soft", message="x").is_hard_error
        assert not PolicyIssue(check_id="P01", severity="WARNING", gate="hard", message="x").is_hard_error

    def test_rejects_unknown_check_id(self):
        with pytest.raises(PolicyQAError):
            PolicyIssue(check_id="P99", message="x").validate()

    def test_rejects_unknown_category(self):
        with pytest.raises(PolicyQAError):
            PolicyIssue(check_id="P01", category="nope", message="x").validate()

    def test_rejects_empty_message(self):
        with pytest.raises(PolicyQAError):
            PolicyIssue(check_id="P01", message="").validate()


class TestPolicyCheckRecord:
    def test_outcome_pass_review_reject(self):
        assert PolicyCheckRecord(check_id="P01", category="c", hard_gate=True).outcome == "PASS"
        record = PolicyCheckRecord(
            check_id="P01", category="c", hard_gate=True, errors=1, hard_errors=0
        )
        assert record.outcome == "REVIEW"
        record = PolicyCheckRecord(
            check_id="P01", category="c", hard_gate=True, errors=1, hard_errors=1
        )
        assert record.outcome == "REJECT"

    def test_round_trip(self):
        record = PolicyCheckRecord(
            check_id="P06", category="synthetic_media_disclosure", hard_gate=False,
            errors=1, warnings=2, infos=0, hard_errors=0,
        )
        assert PolicyCheckRecord.from_dict(record.to_dict()) == record


class TestRiskSummary:
    def test_round_trip(self):
        summary = RiskSummary(verdict="REVIEW", error_count=2, flagged_categories=("c",))
        assert RiskSummary.from_dict(summary.to_dict()) == summary

    def test_defaults(self):
        summary = RiskSummary(verdict="PASS")
        assert summary.highest_risk_severity == "none"
        assert summary.reviewer_note  # honesty note always present


class TestPolicyQAConfig:
    def test_defaults(self):
        config = PolicyQAConfig().validate()
        assert config.policy_version == POLICY_VERSION

    def test_rejects_empty_version(self):
        with pytest.raises(PolicyQAError):
            PolicyQAConfig(policy_version="").validate()

    def test_rejects_bad_threshold(self):
        with pytest.raises(PolicyQAError):
            PolicyQAConfig(repetition_token_overlap_threshold=1.0).validate()
        with pytest.raises(PolicyQAError):
            PolicyQAConfig(repetition_token_overlap_threshold=0.0).validate()


class TestPolicyQAResult:
    def test_clean_baseline_is_pass(self):
        result = evaluate_policy(committed_inputs(), run_id="model-test")
        assert result.overall_verdict == "PASS"
        assert result.passed is True

    def test_json_round_trip_preserves_policy_version(self):
        result = evaluate_policy(committed_inputs(), run_id="model-test")
        restored = PolicyQAResult.from_json(result.to_json())
        assert restored.policy_result_id == result.policy_result_id
        assert restored.policy_version == POLICY_VERSION
        assert restored.target_ref == result.target_ref
        assert restored.stage == "policy_qa"
        assert restored.schema == "PolicyQAResult"
        assert restored.risk_summary == result.risk_summary

    def test_data_payload_carries_required_keys(self):
        result = evaluate_policy(committed_inputs(), run_id="model-test")
        data = result.to_dict()["data"]
        for key in (
            "policy_result_id",
            "target_ref",
            "overall_verdict",
            "checks",
            "issues",
            "risk_summary",
            "policy_version",
            "provenance",
        ):
            assert key in data

    def test_produced_by_matches_evaluator(self):
        result = evaluate_policy(committed_inputs(), run_id="model-test")
        assert result.produced_by["model"] == "PolicyQAValidator"
        assert result.produced_by["prompt_version"] == POLICY_VERSION
        assert result.provenance.prompt_version == POLICY_VERSION

    def test_validate_rejects_unknown_verdict(self):
        result = evaluate_policy(committed_inputs(), run_id="model-test")
        bad = PolicyQAResult(
            policy_result_id=result.policy_result_id,
            target_ref=result.target_ref,
            topic_id=result.topic_id,
            topic_title=result.topic_title,
            script_id=result.script_id,
            policy_version=result.policy_version,
            overall_verdict="SUCCESS",
            checked_at=result.checked_at,
            checks=result.checks,
            provenance=result.provenance,
        )
        with pytest.raises(PolicyQAError):
            bad.validate()

    def test_validate_requires_provenance(self):
        result = evaluate_policy(committed_inputs(), run_id="model-test")
        bad = PolicyQAResult(
            policy_result_id=result.policy_result_id,
            target_ref=result.target_ref,
            topic_id=result.topic_id,
            topic_title=result.topic_title,
            script_id=result.script_id,
            policy_version=result.policy_version,
            overall_verdict="PASS",
            checked_at=result.checked_at,
            checks=result.checks,
            provenance=None,
        )
        with pytest.raises(PolicyQAError):
            bad.validate()


class TestDeterministicOutput:
    def test_identical_inputs_produce_identical_artifacts(self):
        created_at = "2026-09-20T08:00:00+00:00"
        first = evaluate_policy(
            committed_inputs(), run_id="det-test", created_at=created_at
        )
        second = evaluate_policy(
            committed_inputs(), run_id="det-test", created_at=created_at
        )
        assert first.to_json() == second.to_json()

    def test_verdicts_vocabulary(self):
        assert POLICY_VERDICTS == ("PASS", "REVIEW", "REJECT")