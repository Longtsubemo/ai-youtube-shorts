"""Stage-9 validator tests: verdict aggregation, risk summary, classifier
boundary, determinism, and the full evaluate_policy path.
"""

from __future__ import annotations

import pytest

from agents.qa.policy.classifier import (
    PolicyClassifierSuggestion,
    build_policy_context,
)
from agents.qa.policy.models import PolicyQAConfig
from agents.qa.policy.rules import POLICY_VERSION
from agents.qa.policy.validator import (
    VERDICT_PASS,
    VERDICT_REJECT,
    VERDICT_REVIEW,
    aggregate_verdict,
    evaluate_policy,
    run_checks,
)
from agents.qa.policy.models import PolicyIssue
from policy_qa_test_utils import (
    build_fake_footage_inputs,
    build_harmful_inputs,
    build_missing_plan_inputs,
    build_no_qa_inputs,
    build_overstatement_inputs,
    build_review_qa_inputs,
    build_watermarked_inputs,
    committed_inputs,
    invalid_version_config,
)


class TestAggregateVerdict:
    def test_clean_is_pass(self):
        assert aggregate_verdict([]) == "PASS"

    def test_soft_error_is_review(self):
        issues = [PolicyIssue(check_id="P06", severity="ERROR", gate="soft", message="x")]
        assert aggregate_verdict(issues) == "REVIEW"

    def test_hard_error_is_reject(self):
        issues = [PolicyIssue(check_id="P05", severity="ERROR", gate="hard", message="x")]
        assert aggregate_verdict(issues) == VERDICT_REJECT

    def test_hard_error_beats_soft_error(self):
        issues = [
            PolicyIssue(check_id="P06", severity="ERROR", gate="soft", message="x"),
            PolicyIssue(check_id="P01", severity="ERROR", gate="hard", message="y"),
        ]
        assert aggregate_verdict(issues) == "REJECT"

    def test_warning_and_info_are_not_errors(self):
        issues = [
            PolicyIssue(check_id="P05", severity="WARNING", gate="hard", message="x"),
            PolicyIssue(check_id="P06", severity="INFO", gate="soft", message="y"),
        ]
        assert aggregate_verdict(issues) == "PASS"


class TestEvaluateClean:
    def test_baseline_passes(self):
        result = evaluate_policy(committed_inputs(), run_id="val-test")
        assert result.overall_verdict == "PASS"
        assert result.issues == ()
        assert result.risk_summary.verdict == "PASS"
        assert result.risk_summary.flagged_categories == ()
        assert result.classifier_suggestion == ""

    def test_provenance_and_envelope(self):
        result = evaluate_policy(committed_inputs(), run_id="val-test")
        assert result.provenance.generated_by == "policy_qa_agent_v1"
        assert result.provenance.model == "PolicyQAValidator"
        assert result.produced_by["prompt_version"] == POLICY_VERSION

    def test_target_ref_is_the_plan(self):
        result = evaluate_policy(committed_inputs(), run_id="val-test")
        assert result.target_ref == committed_inputs().plan.production_plan_id

    def test_all_15_checks_recorded(self):
        result = evaluate_policy(committed_inputs(), run_id="val-test")
        assert len(result.checks) == 15
        assert all(record.outcome == "PASS" for record in result.checks)


class TestEvaluateScenarios:
    def test_review_from_overstatement(self):
        result = evaluate_policy(build_overstatement_inputs(), run_id="val-test")
        assert result.overall_verdict == "REVIEW"
        assert any(issue.check_id == "P03" for issue in result.issues)
        assert "misleading_presentation" in result.risk_summary.flagged_categories

    def test_reject_from_harmful_marker(self):
        result = evaluate_policy(build_harmful_inputs(), run_id="val-test")
        assert result.overall_verdict == "REJECT"
        assert any(issue.check_id == "P01" for issue in result.issues)

    def test_reject_from_fake_footage(self):
        result = evaluate_policy(build_fake_footage_inputs(), run_id="val-test")
        assert result.overall_verdict == "REJECT"

    def test_reject_from_watermarked_asset(self):
        result = evaluate_policy(build_watermarked_inputs(), run_id="val-test")
        assert result.overall_verdict == "REJECT"

    def test_stage8_review_cannot_become_pass(self):
        result = evaluate_policy(build_review_qa_inputs(), run_id="val-test")
        assert result.overall_verdict == "REVIEW"
        assert any(issue.check_id == "P15" for issue in result.issues)

    def test_missing_qa_is_review_not_pass(self):
        result = evaluate_policy(build_no_qa_inputs(), run_id="val-test")
        assert result.overall_verdict == "REVIEW"

    def test_missing_plan_is_review(self):
        result = evaluate_policy(build_missing_plan_inputs(), run_id="val-test")
        assert result.overall_verdict == "REVIEW"

    def test_invalid_policy_version_rejects_closed(self):
        result = evaluate_policy(
            committed_inputs(), config=invalid_version_config(), run_id="val-test"
        )
        assert result.overall_verdict == "REJECT"
        issue = result.issue_for("P14")
        assert issue is not None
        assert issue.gate == "hard"


class _StubClassifier:
    """Deterministic classifier stub: always returns a fixed suggestion."""

    def __init__(self, suggested_verdict: str, model_name: str = "stub"):
        self._suggested_verdict = suggested_verdict
        self._model_name = model_name

    def classify(self, context) -> PolicyClassifierSuggestion:
        return PolicyClassifierSuggestion(
            suggested_verdict=self._suggested_verdict,
            model_name=self._model_name,
            notes="test stub",
        )


class TestClassifierBoundary:
    def test_suggestion_can_only_make_gate_stricter(self):
        # deterministic PASS + advisory REVIEW -> REVIEW.
        result = evaluate_policy(
            committed_inputs(),
            classifier=_StubClassifier("REVIEW"),
            run_id="val-test",
        )
        assert result.overall_verdict == "REVIEW"
        assert result.classifier_suggestion == "REVIEW"

    def test_suggestion_cannot_downgrade_deterministic_review(self):
        result = evaluate_policy(
            build_review_qa_inputs(),
            classifier=_StubClassifier("PASS"),
            run_id="val-test",
        )
        assert result.overall_verdict == "REVIEW"

    def test_suggestion_cannot_override_deterministic_reject(self):
        for suggestion in ("PASS", "REVIEW", "REJECT"):
            result = evaluate_policy(
                build_harmful_inputs(),
                classifier=_StubClassifier(suggestion),
                run_id="val-test",
            )
            assert result.overall_verdict == "REJECT"

    def test_no_classifier_records_no_suggestion(self):
        result = evaluate_policy(committed_inputs(), run_id="val-test")
        assert result.classifier_suggestion == ""

    def test_context_is_bounded(self):
        context = build_policy_context(committed_inputs())
        assert context.title
        assert context.line_count == 8
        assert context.scene_count == 7
        assert context.video_qa_verdict == "PASS"
        assert isinstance(context.to_dict(), dict)

    def test_classifier_errors_are_not_swallowed(self, monkeypatch):
        class ExplodingClassifier:
            def classify(self, context):
                raise RuntimeError("provider exploded")

        with pytest.raises(RuntimeError):
            evaluate_policy(
                committed_inputs(),
                classifier=ExplodingClassifier(),
                run_id="val-test",
            )


class TestRunChecks:
    def test_issue_ids_are_stable_and_sequential(self):
        issues, records = run_checks(build_missing_plan_inputs(), PolicyQAConfig())
        ids = [issue.issue_id for issue in issues]
        assert ids == [f"pol_{index:02d}" for index in range(len(ids))]
        assert len(records) == 15

    def test_record_counts_match_issues(self):
        issues, records = run_checks(build_harmful_inputs(), PolicyQAConfig())
        total_errors = sum(issue.is_error for issue in issues)
        assert sum(record.errors for record in records) == total_errors