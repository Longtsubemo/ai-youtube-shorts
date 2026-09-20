"""Generator behavior tests (Stage 4): valid output, blocked/insufficient
gates, strict approved-only consumption, determinism, and network-freedom."""

from __future__ import annotations

import json
import socket
import urllib.request

import pytest

from agents.script.script_generator import (
    STATUS_BLOCKED,
    STATUS_GENERATED,
    STATUS_INSUFFICIENT,
    MockScriptGenerator,
    ScriptConfig,
    ScriptGenerationResult,
)
from agents.script.stage import run_script_generation

from script_test_utils import (
    FIXED_AT,
    OCTOPUS_BLUE,
    OCTOPUS_SMARTEST,
    OCTOPUS_SWIM,
    OCTOPUS_TANK,
    OCTOPUS_THREE_HEARTS,
    build_octopus_pair,
    build_pair,
    fact_result_from_fixture,
    mock_octopus_package,
)


def test_valid_generation_shape():
    result, package = build_octopus_pair(approved=("fact", "explanation", "explanation"))
    generation = run_script_generation(
        result, package, config=ScriptConfig(), run_id="gen-shape", created_at=FIXED_AT
    )
    assert generation.status == STATUS_GENERATED, generation.message
    assert generation.generated
    assert generation.validation is not None and generation.validation.valid
    script = generation.script
    assert script.script_id.startswith("sc_")
    assert script.title == "Why octopuses have three hearts"
    assert script.hook
    assert any(line.purpose in ("fact", "explanation") for line in script.lines)
    assert set(script.used_claim_ids) == set(result.approved_claims)
    assert script.unused_approved_claim_ids == ()
    assert isinstance(script.estimated_duration_seconds, float)
    assert 30.0 <= script.estimated_duration_seconds <= 60.0
    assert script.provenance is not None and script.provenance.model == "MockScriptGenerator"
    assert script.produced_by["model"] == "MockScriptGenerator"


def test_only_approved_claims_consumed():
    result, package = build_octopus_pair(approved=("fact", "explanation", "explanation"))
    rejected_id = result.rejected_claims[0]
    review_id = result.review_items[0]
    generation = run_script_generation(result, package, config=ScriptConfig(), run_id="only-approved")
    assert generation.generated
    script = generation.script
    referenced = set(script.all_claim_refs())
    assert referenced == set(result.approved_claims)
    assert rejected_id not in referenced
    assert review_id not in referenced
    assert rejected_id not in script.used_claim_ids
    assert review_id not in script.used_claim_ids
    body = script.to_json()
    assert "tank lid" not in body
    assert "smartest invertebrates" not in body


def test_rejected_package_blocked():
    result, package = build_octopus_pair(
        approved=("fact", "explanation", "explanation"),
        package_verdict="REJECT",
    )
    generation = run_script_generation(result, package, config=ScriptConfig(), run_id="rej")
    assert generation.status == STATUS_BLOCKED
    assert not generation.generated
    assert "REJECT" in generation.message
    assert generation.blocked_reasons == ("package_verdict=REJECT",)


def test_review_package_blocked_by_default():
    result, package = build_octopus_pair(
        approved=("fact", "explanation", "explanation"),
        package_verdict="REVIEW",
    )
    generation = run_script_generation(result, package, config=ScriptConfig(), run_id="rev")
    assert generation.status == STATUS_BLOCKED
    assert "REVIEW" in generation.message
    assert generation.blocked_reasons == ("package_verdict=REVIEW",)


def test_review_content_opt_in_generates():
    result, package = build_octopus_pair(
        approved=("fact", "explanation", "explanation"),
        package_verdict="REVIEW",
    )
    generation = run_script_generation(
        result,
        package,
        config=ScriptConfig(allow_review_content=True),
        run_id="rev-optin",
    )
    assert generation.generated, generation.message
    assert generation.script.result_id == result.result_id
    assert not any(ref in result.review_items for ref in generation.script.all_claim_refs())


def test_insufficient_when_too_few_approved():
    result, package = build_pair(
        claims=[(OCTOPUS_THREE_HEARTS, "fact")],
        verdicts={OCTOPUS_THREE_HEARTS: "REVIEW"},
    )
    generation = run_script_generation(result, package, config=ScriptConfig(), run_id="insuf")
    assert generation.status == STATUS_INSUFFICIENT
    assert not generation.generated
    assert "minimum" in generation.message.lower()


def test_insufficient_when_too_short():
    result, package = build_pair(
        claims=[(OCTOPUS_BLUE, "explanation")],
        verdicts={OCTOPUS_BLUE: "PASS"},
    )
    generation = run_script_generation(
        result,
        package,
        config=ScriptConfig(min_approved_claims=1),
        run_id="short",
    )
    assert generation.status == STATUS_INSUFFICIENT
    assert "below the" in generation.message
    assert generation.blocked_reasons == ("too_short",)


def test_deterministic_across_runs():
    result, package = build_octopus_pair(approved=("fact", "explanation", "explanation"))
    a = run_script_generation(
        result, package, config=ScriptConfig(), run_id="det", created_at=FIXED_AT
    )
    b = run_script_generation(
        result, package, config=ScriptConfig(), run_id="det", created_at=FIXED_AT
    )
    assert a.script.to_dict() == b.script.to_dict()


def test_speaking_rate_changes_duration():
    result, package = build_octopus_pair(approved=("fact", "explanation", "explanation"))
    slow = run_script_generation(
        result, package, config=ScriptConfig(words_per_minute=125), run_id="slow", created_at=FIXED_AT
    )
    fast = run_script_generation(
        result, package, config=ScriptConfig(words_per_minute=200), run_id="fast", created_at=FIXED_AT
    )
    assert slow.script.estimated_duration_seconds > fast.script.estimated_duration_seconds


def test_exceeding_max_duration_warns_but_generates():
    result, package = build_octopus_pair(approved=("fact", "explanation", "explanation"))
    generation = run_script_generation(
        result, package, config=ScriptConfig(words_per_minute=30), run_id="warn", created_at=FIXED_AT
    )
    assert generation.generated
    assert generation.validation.valid
    assert any("exceeds the" in w for w in generation.validation.warnings)


def test_future_protocol_input_bounds_to_approved():
    result, package = build_octopus_pair(approved=("fact", "explanation", "explanation"))
    generator = MockScriptGenerator()
    generation = generator.generate(
        result,
        claims=package.claims,  # caller passes EVERYTHING - generator must filter
        config=ScriptConfig(),
        run_id="defense",
        created_at=FIXED_AT,
    )
    assert generation.generated
    assert set(generation.script.all_claim_refs()) == set(result.approved_claims)


def test_missing_approved_record_blocks_rather_than_drop():
    result, package = build_octopus_pair(approved=("fact", "explanation", "explanation"))
    result.approved_claims = result.approved_claims + ("cl_missing_record",)
    generation = MockScriptGenerator().generate(
        result,
        claims=package.claims,
        config=ScriptConfig(),
        run_id="missing",
        created_at=FIXED_AT,
    )
    assert generation.status == STATUS_BLOCKED
    assert any("missing claim record" in r for r in generation.blocked_reasons)


def test_hedged_claim_keeps_hedge():
    result, package = build_octopus_pair(approved=("fact", "explanation", "explanation"))
    generation = run_script_generation(result, package, run_id="hedge", created_at=FIXED_AT)
    assert generation.generated
    body = generation.script.to_json()
    assert "three hearts" in body


def test_no_network_and_no_external_knowledge(monkeypatch):
    def _block(*args, **kwargs):
        raise RuntimeError("network access is forbidden in tests")

    monkeypatch.setattr(socket, "socket", _block)
    monkeypatch.setattr(socket, "create_connection", _block)
    monkeypatch.setattr(urllib.request, "urlopen", _block)

    result, package = build_octopus_pair(approved=("fact", "explanation", "explanation"))
    generation = run_script_generation(result, package, config=ScriptConfig(), run_id="offline")
    assert generation.generated
    assert set(generation.script.all_claim_refs()) == set(result.approved_claims)


def test_end_to_end_with_committed_mock_fixture():
    result = fact_result_from_fixture()
    package = mock_octopus_package()
    generation = run_script_generation(
        result, package, config=ScriptConfig(), run_id="e2e", created_at=FIXED_AT
    )
    assert generation.generated, generation.message
    assert generation.validation.valid
    script = generation.script
    assert set(script.used_claim_ids) == set(result.approved_claims)
    assert 30.0 <= script.estimated_duration_seconds <= 60.0
    # Round-trips through the artifact contract.
    restored = json.loads(script.to_json())
    assert restored["schema"] == "Script" and restored["data"]["script_id"] == script.script_id
    # Every line refs only approved claims.
    assert set(script.all_claim_refs()) <= set(result.approved_claims)


def test_line_durations_sum_to_total():
    result, package = build_octopus_pair(approved=("fact", "explanation", "explanation"))
    script = run_script_generation(result, package, run_id="sum", created_at=FIXED_AT).script
    derived = round(sum(line.estimated_duration_seconds for line in script.lines), 1)
    assert derived == script.estimated_duration_seconds