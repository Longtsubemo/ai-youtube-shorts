"""Stage-9 per-check behavior tests (P01..P15), run directly on `checks`.

Each scenario builder introduces exactly one defect on the clean octopus
baseline, and the corresponding check must fire with the documented severity +
gate. Separately verify the harmless baseline clears every check and that
issues carry the audit chain (rule, affected ids, evidence).
"""

from __future__ import annotations

import pytest

from agents.qa.policy import checks
from agents.qa.policy.models import PolicyQAConfig
from agents.qa.policy.rules import POLICY_CHECK_ORDER, POLICY_RULES_BY_ID, POLICY_VERSION
from policy_qa_test_utils import (
    committed_inputs,
    build_adsensitive_inputs,
    build_dangerous_inputs,
    build_fake_footage_inputs,
    build_harmful_inputs,
    build_licensed_external_inputs,
    build_misinfo_framing_inputs,
    build_missing_plan_inputs,
    build_missing_provenance_inputs,
    build_near_duplicate_inputs,
    build_no_qa_inputs,
    build_overstatement_inputs,
    build_privacy_leak_inputs,
    build_privacy_pii_inputs,
    build_review_qa_inputs,
    build_sensational_title_inputs,
    build_spam_metadata_inputs,
    build_stale_qa_inputs,
    build_synthetic_no_disclosure_inputs,
    build_unknown_rights_inputs,
    build_watermarked_inputs,
    invalid_version_config,
)

CONFIG = PolicyQAConfig()


def first_issue(check_id, inputs, config=CONFIG):
    issues = checks.CHECK_FUNCTIONS[check_id](inputs, config)
    return tuple(issues)


class TestRegistry:
    def test_every_registered_check_has_an_implementation(self):
        assert set(checks.CHECK_FUNCTIONS) == set(POLICY_CHECK_ORDER)

    def test_clean_baseline_clears_every_check(self):
        inputs = committed_inputs()
        for check_id in POLICY_CHECK_ORDER:
            issues = first_issue(check_id, inputs)
            assert issues == (), f"{check_id} flagged the healthy octopus fixture: {issues}"


class TestP01HarmfulContent:
    def test_hard_marker_rejects(self):
        issues = first_issue("P01", build_harmful_inputs())
        assert len(issues) == 1
        assert issues[0].severity == "ERROR"
        assert issues[0].gate == "hard"
        assert issues[0].check_id == "P01"
        assert issues[0].affected_scene_ids


class TestP02DangerousInstructions:
    def test_hard_instructional_phrase_rejects(self):
        issues = first_issue("P02", build_dangerous_inputs())
        assert len(issues) == 1
        assert issues[0].severity == "ERROR"
        assert issues[0].gate == "hard"


class TestP03MisleadingPresentation:
    def test_overstatement_on_weak_claim_reviews(self):
        issues = first_issue("P03", build_overstatement_inputs())
        assert len(issues) == 1
        assert issues[0].severity == "ERROR"
        assert issues[0].gate == "soft"
        assert issues[0].affected_line_ids
        assert issues[0].affected_claim_ids
        assert issues[0].policy_rule.startswith("P03@")

    def test_strong_language_without_claim_refs_reviews(self):
        from agents.qa.policy.stage import load_committed_fixtures
        import dataclasses

        script = load_committed_fixtures().script
        lines = list(script.lines)
        base = [l for l in lines if l.purpose in ("fact", "explanation")][0]
        new_line = dataclasses.replace(
            base,
            claim_refs=(),
            text="This guide proves octopuses definitely have three hearts.",
        )
        lines = [new_line if l == base else l for l in lines]
        inputs = dataclasses.replace(
            load_committed_fixtures(), script=dataclasses.replace(script, lines=tuple(lines))
        )
        issues = first_issue("P03", inputs)
        assert issues, "Missing claim refs must be treated as unverifiable -> REVIEW"


class TestP04MisinfoFraming:
    def test_definitive_sensitive_framing_reviews(self):
        issues = first_issue("P04", build_misinfo_framing_inputs())
        assert len(issues) == 1
        assert issues[0].severity == "ERROR"
        assert issues[0].gate == "soft"


class TestP05FabricatedMedia:
    def test_real_media_claim_on_mock_artifact_rejects(self):
        issues = first_issue("P05", build_fake_footage_inputs())
        assert len(issues) == 1
        assert issues[0].severity == "ERROR"
        assert issues[0].gate == "hard"
        assert issues[0].affected_scene_ids
        assert issues[0].affected_asset_ids


class TestP06SyntheticDisclosure:
    def test_real_person_in_generated_visual_without_disclosure_reviews(self):
        issues = first_issue("P06", build_synthetic_no_disclosure_inputs())
        assert len(issues) == 1
        assert issues[0].severity == "ERROR"
        assert issues[0].gate == "soft"

    def test_harmless_ai_octopus_visuals_pass(self):
        assert first_issue("P06", committed_inputs()) == ()


class TestP07AssetRights:
    def test_unknown_rights_reviews(self):
        issues = first_issue("P07", build_unknown_rights_inputs())
        assert len(issues) == 1
        assert issues[0].severity == "ERROR"
        assert issues[0].gate == "soft"
        assert issues[0].affected_asset_ids

    def test_watermarked_asset_rejects(self):
        issues = first_issue("P07", build_watermarked_inputs())
        assert len(issues) == 1
        assert issues[0].severity == "ERROR"
        assert issues[0].gate == "hard"

    def test_licensed_external_asset_passes(self):
        assert first_issue("P07", build_licensed_external_inputs()) == ()


class TestP08Privacy:
    def test_plain_pii_reviews(self):
        issues = first_issue("P08", build_privacy_pii_inputs())
        assert len(issues) == 1
        assert issues[0].severity == "ERROR"
        assert issues[0].gate == "soft"

    def test_pii_with_leak_framing_rejects(self):
        issues = first_issue("P08", build_privacy_leak_inputs())
        assert len(issues) == 1
        assert issues[0].severity == "ERROR"
        assert issues[0].gate == "hard"


class TestP09SpamMetadata:
    def test_bait_title_and_hook_review(self):
        issues = first_issue("P09", build_spam_metadata_inputs())
        assert len(issues) == 1
        assert issues[0].severity == "ERROR"
        assert issues[0].gate == "soft"


class TestP10RepetitiveContent:
    def test_near_duplicate_factual_lines_review(self):
        issues = first_issue("P10", build_near_duplicate_inputs())
        assert len(issues) == 1
        assert issues[0].severity == "ERROR"
        assert issues[0].gate == "soft"
        assert len(issues[0].affected_line_ids) == 2


class TestP11AdvertiserSuitability:
    def test_sensitive_topic_without_disclaimer_reviews(self):
        issues = first_issue("P11", build_adsensitive_inputs())
        assert len(issues) == 1
        assert issues[0].severity == "ERROR"
        assert issues[0].gate == "soft"


class TestP12TitleContentConsistency:
    def test_sensational_title_without_matching_content_reviews(self):
        issues = first_issue("P12", build_sensational_title_inputs())
        assert len(issues) >= 1
        assert all(issue.severity == "ERROR" for issue in issues)
        assert all(issue.gate == "soft" for issue in issues)


class TestP13ProvenanceCompleteness:
    def test_missing_plan_provenance_reviews(self):
        issues = first_issue("P13", build_missing_provenance_inputs())
        assert len(issues) == 1
        assert issues[0].severity == "ERROR"
        assert issues[0].gate == "soft"

    def test_missing_plan_artifact_reviews(self):
        issues = first_issue("P13", build_missing_plan_inputs())
        assert len(issues) == 1
        assert issues[0].severity == "ERROR"


class TestP14PolicyVersionValidity:
    def test_unregistered_version_hard_rejects(self):
        issues = first_issue("P14", committed_inputs(), config=invalid_version_config())
        assert len(issues) == 1
        assert issues[0].severity == "ERROR"
        assert issues[0].gate == "hard"


class TestP15PriorGateConsistency:
    def test_missing_qa_reviews(self):
        issues = first_issue("P15", build_no_qa_inputs())
        assert len(issues) == 1
        assert issues[0].severity == "ERROR"
        assert issues[0].gate == "soft"

    def test_prior_stage8_review_does_not_pass(self):
        issues = first_issue("P15", build_review_qa_inputs())
        assert len(issues) == 1
        assert issues[0].severity == "ERROR"
        assert issues[0].gate == "soft"
        assert "stage-8".casefold() in issues[0].message.casefold() or "PASS" in issues[0].message

    def test_stale_qa_reviews(self):
        issues = first_issue("P15", build_stale_qa_inputs())
        assert len(issues) == 1
        assert issues[0].severity == "ERROR"
        assert issues[0].gate == "soft"

    def test_clean_gate_passes(self):
        assert first_issue("P15", committed_inputs()) == ()


class TestTraceability:
    def test_every_rule_is_traced_in_every_issue(self):
        scenarios = [
            build_harmful_inputs(),
            build_dangerous_inputs(),
            build_overstatement_inputs(),
            build_misinfo_framing_inputs(),
            build_fake_footage_inputs(),
            build_synthetic_no_disclosure_inputs(),
            build_unknown_rights_inputs(),
            build_privacy_pii_inputs(),
            build_spam_metadata_inputs(),
            build_near_duplicate_inputs(),
            build_adsensitive_inputs(),
            build_sensational_title_inputs(),
            build_review_qa_inputs(),
        ]
        seen = 0
        for inputs in scenarios:
            for check_id in POLICY_CHECK_ORDER:
                for issue in checks.CHECK_FUNCTIONS[check_id](inputs, CONFIG):
                    assert issue.policy_rule == f"{check_id}@{POLICY_VERSION}"
                    assert issue.category == POLICY_RULES_BY_ID[check_id].category
                    assert issue.evidence
                    seen += 1
        assert seen >= len(scenarios)  # every scenario produced at least one issue