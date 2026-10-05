"""Offline tests for the Stage-4 FactCheckResult binding (full pipeline).

The real Gemini package path used to generate straight from a ResearchPackage
with `result_id=""`, so a rejected/REVIEW claim could reach a script with no
fact-check verdict behind it. These tests pin the fixed contract:

- `Script.result_id` is inherited from the supplied FactCheckResult,
- only `approved_claims` reach the prompt / the working set,
- REJECT/REVIEW/unknown claims can never become factual script material,
- a result belonging to a different package (or naming a claim the package does
  not contain) is refused,
- the full-pipeline CLI path refuses to run without a FactCheckResult, and
  writes no partial artifact when it fails,
- the explicitly opted-in `--preview` path keeps its old unbound behavior, and
  the mock provider is unchanged.

Everything here is offline: Gemini is a `FakeRawClient` and no key, network, or
Firecrawl call is involved.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from agents.qa.fact_checker.models import FactCheckResult
from agents.research.models import ResearchPackage
from agents.script.models import ScriptError
from agents.script.provider import MockScriptProvider
from agents.script.script_generator import (
    STATUS_BLOCKED,
    STATUS_GENERATED,
    STATUS_INSUFFICIENT,
    ScriptConfig,
    main,
)
from agents.script.stage import (
    approved_claims,
    bind_fact_check_result,
    fact_check_gate,
)
from fact_check_test_utils import FIXED_AT, make_claim, make_package, make_source
from providers.gemini.config import GeminiConfig
from providers.gemini.script_provider import GeminiScriptProvider
from script_test_utils import (
    OCTOPUS_BLUE,
    OCTOPUS_SMARTEST,
    OCTOPUS_SWIM,
    OCTOPUS_TANK,
    OCTOPUS_THREE_HEARTS,
    build_octopus_pair,
)
from tests.gemini_test_utils import FakeRawClient, make_client, noop_sleep

SOURCE_URL = "https://example.test/papers/octopus"

# The three verified claims, plus one REVIEW and one REJECT claim.
ALL_SPECS = (
    (OCTOPUS_THREE_HEARTS, "fact"),
    (OCTOPUS_SWIM, "explanation"),
    (OCTOPUS_BLUE, "explanation"),
    (OCTOPUS_SMARTEST, "inference"),
    (OCTOPUS_TANK, "speculation"),
)


def build_five_claim_package() -> ResearchPackage:
    source = make_source(url=SOURCE_URL, tier="primary")
    claims = [make_claim(text, claim_type=ctype) for text, ctype in ALL_SPECS]
    return make_package(
        claims=claims,
        sources=[source],
        run_id="bind-test",
        topic_id="t_octopus_hearts",
        topic_title="Why octopuses have three hearts",
    ).validate()


def bound_pair(package_verdict: str = "PASS"):
    """A (package, result) pair: 3 approved, 1 REVIEW, 1 REJECT."""
    result, package = build_octopus_pair(
        approved=("fact", "explanation", "explanation"),
        package_verdict=package_verdict,
        run_id="bind-test",
        created_at=FIXED_AT,
    )
    return package, result


def _provider(fake: FakeRawClient) -> GeminiScriptProvider:
    client = make_client(GeminiConfig(api_key="k"), raw_client=fake, sleep=noop_sleep)
    return GeminiScriptProvider(client=client)


def _payload_for(claims, refs, *, title="Why octopuses have three hearts") -> dict:
    """Schema-valid payload: two framing lines + one verbatim line per ref."""
    lines = [
        {"purpose": "hook", "text": "You probably didn't know this.", "claim_refs": []},
        {"purpose": "context", "text": "Here is what the evidence showed.", "claim_refs": []},
    ]
    by_id = {claim.claim_id: claim for claim in claims}
    for claim_id in refs:
        lines.append(
            {
                "purpose": "fact",
                "text": by_id[claim_id].claim_text,
                "claim_refs": [claim_id],
            }
        )
    lines.append({"purpose": "call_to_action", "text": "Follow for more.", "claim_refs": []})
    return {"title": title, "lines": lines}


def _generate(package, result, payload=None, *, config=None, run_id="bind-test"):
    approved = [claim.claim_id for claim in approved_claims(result, package)]
    fake = FakeRawClient()
    fake.add_result_json(
        payload if payload is not None else _payload_for(package.claims, approved)
    )
    return _provider(fake).generate(
        package,
        config=config,
        run_id=run_id,
        created_at=FIXED_AT,
        fact_check_result=result,
    )


# --------------------------------------------------------------------------- #
# 1. result_id inheritance
# --------------------------------------------------------------------------- #


class TestResultIdBinding:
    def test_script_inherits_fact_check_result_id(self):
        package, result = bound_pair()
        generation = _generate(package, result)
        assert generation.status == STATUS_GENERATED, generation.message
        assert generation.script.result_id == result.result_id
        assert generation.script.result_id != ""

    def test_result_id_tracks_a_different_result_for_same_package(self):
        """A second verdict on the same package must re-bind the script."""
        package, result = bound_pair()
        from agents.qa.fact_checker.models import result_id_for

        other = FactCheckResult(
            result_id=result_id_for("bind-test-2", package.package_id),
            topic_id=result.topic_id,
            topic_title=result.topic_title,
            package_id=result.package_id,
            claim_results=result.claim_results,
            package_verdict=result.package_verdict,
            review_items=result.review_items,
            rejected_claims=result.rejected_claims,
            approved_claims=result.approved_claims,
            run_id="bind-test-2",
        ).validate()
        generation = _generate(package, other)
        assert generation.status == STATUS_GENERATED, generation.message
        assert generation.script.result_id == other.result_id
        assert generation.script.result_id != result.result_id


# --------------------------------------------------------------------------- #
# 2-3. approved-only working set; REVIEW/REJECT cannot become script content
# --------------------------------------------------------------------------- #


class TestApprovedOnlyWorkingSet:
    def test_only_approved_claims_are_used(self):
        package, result = bound_pair()
        generation = _generate(package, result)
        assert generation.generated, generation.message
        script = generation.script
        assert set(script.used_claim_ids) <= set(result.approved_claims)
        assert set(script.used_claim_ids) == set(result.approved_claims)

    def test_review_and_rejected_claims_never_appear_in_the_prompt(self):
        package, result = bound_pair()
        approved = [claim.claim_id for claim in approved_claims(result, package)]
        fake = FakeRawClient()
        fake.add_result_json(_payload_for(package.claims, approved))
        _provider(fake).generate(
            package, run_id="bind-test", created_at=FIXED_AT, fact_check_result=result
        )
        prompt = fake.calls[0][1]
        for claim in package.claims:
            if claim.claim_id not in result.approved_claims:
                assert claim.claim_id not in prompt
                assert claim.claim_text not in prompt

    def test_model_referencing_a_rejected_claim_is_blocked(self):
        package, result = bound_pair()
        rejected = result.rejected_claims[0]
        # The model cheats and scripts the rejected claim; the gate must refuse.
        generation = _generate(
            package,
            result,
            payload=_payload_for(package.claims, [rejected] * 3),
        )
        assert generation.status == STATUS_BLOCKED
        assert generation.script is None or not generation.validation.valid
        assert any("REJECTED" in reason for reason in generation.blocked_reasons)

    def test_model_referencing_a_review_claim_is_blocked(self):
        package, result = bound_pair()
        review = result.review_items[0]
        generation = _generate(
            package, result, payload=_payload_for(package.claims, [review] * 3)
        )
        assert generation.status == STATUS_BLOCKED
        assert any("REVIEW" in reason for reason in generation.blocked_reasons)

    def test_unlisted_package_claim_is_blocked(self):
        """A claim with a PASS verdict that is absent from approved_claims."""
        package, result = bound_pair()
        dropped = next(
            claim.claim_id
            for claim in package.claims
            if claim.claim_id in set(result.approved_claims)
        )
        narrowed = FactCheckResult(
            result_id=result.result_id,
            topic_id=result.topic_id,
            topic_title=result.topic_title,
            package_id=result.package_id,
            claim_results=result.claim_results,
            package_verdict=result.package_verdict,
            approved_claims=tuple(
                cid for cid in result.approved_claims if cid != dropped
            ),
            review_items=result.review_items,
            rejected_claims=result.rejected_claims,
            run_id=result.run_id,
        ).validate()
        generation = _generate(
            package, narrowed, payload=_payload_for(package.claims, [dropped] * 3)
        )
        assert generation.status == STATUS_BLOCKED
        assert any("unapproved claim" in r for r in generation.blocked_reasons)

    def test_model_invented_claim_id_is_blocked(self):
        package, result = bound_pair()
        approved = list(result.approved_claims)
        payload = _payload_for(package.claims, approved)
        payload["lines"][2]["claim_refs"] = ["cl_never_saw_this"]
        generation = _generate(package, result, payload=payload)
        assert generation.status == STATUS_BLOCKED
        assert any("unknown claim" in r for r in generation.blocked_reasons)

    def test_unused_approved_ids_are_computed_against_approved_only(self):
        package, result = bound_pair()
        approved = list(result.approved_claims)
        generation = _generate(
            package, result, payload=_payload_for(package.claims, approved[:2])
        )
        assert generation.generated, generation.message
        script = generation.script
        assert set(script.unused_approved_claim_ids) == set(approved[2:])
        # Never a rejected/REVIEW id, even though they are in the package.
        assert not set(script.unused_approved_claim_ids) & set(result.rejected_claims)
        assert not set(script.unused_approved_claim_ids) & set(result.review_items)


# --------------------------------------------------------------------------- #
# 4-5. binding refusals
# --------------------------------------------------------------------------- #


class TestBindingRefusals:
    def test_unknown_claim_in_approved_set_is_refused(self):
        """An approved id the package cannot resolve must not be dropped silently."""
        package, result = bound_pair()
        ghost = FactCheckResult(
            result_id=result.result_id,
            topic_id=result.topic_id,
            topic_title=result.topic_title,
            package_id=result.package_id,
            claim_results=result.claim_results,
            package_verdict=result.package_verdict,
            approved_claims=result.approved_claims + ("cl_ghost",),
            run_id=result.run_id,
        ).validate()
        with pytest.raises(ScriptError, match="missing from ResearchPackage"):
            approved_claims(ghost, package)

    def test_package_mismatch_is_refused(self):
        package, result = bound_pair()
        # build_octopus_pair returns (result, package); the other run's package
        # has a different package_id than the one this result is bound to.
        _, other_package = build_octopus_pair(run_id="a-different-run")
        assert other_package.package_id != package.package_id
        with pytest.raises(ScriptError, match="bound to package"):
            bind_fact_check_result(other_package, result)
        with pytest.raises(ScriptError, match="bound to package"):
            _provider(FakeRawClient()).generate(other_package, fact_check_result=result)

    def test_topic_mismatch_is_refused(self):
        package, result = bound_pair()
        mismatched = FactCheckResult(
            result_id=result.result_id,
            topic_id="t_a_completely_different_topic",
            topic_title=result.topic_title,
            package_id=package.package_id,
            claim_results=result.claim_results,
            package_verdict=result.package_verdict,
            approved_claims=result.approved_claims,
            run_id=result.run_id,
        ).validate()
        with pytest.raises(ScriptError, match="bound to topic"):
            bind_fact_check_result(package, mismatched)

    def test_reject_verdict_blocks_generation(self):
        package, result = bound_pair(package_verdict="REJECT")
        generation = _generate(package, result)
        assert generation.status == STATUS_BLOCKED
        assert generation.blocked_reasons == ("package_verdict=REJECT",)

    def test_review_verdict_blocks_by_default_and_allows_on_opt_in(self):
        package, result = bound_pair(package_verdict="REVIEW")
        blocked = _generate(package, result)
        assert blocked.status == STATUS_BLOCKED
        assert blocked.blocked_reasons == ("package_verdict=REVIEW",)

        allowed = _generate(
            package, result, config=ScriptConfig(allow_review_content=True)
        )
        assert allowed.status == STATUS_GENERATED, allowed.message
        # Even opted in, REVIEW claims stay out of the script.
        assert not set(allowed.script.used_claim_ids) & set(result.review_items)

    def test_too_few_approved_claims_is_insufficient(self):
        package, result = bound_pair()
        starved = FactCheckResult(
            result_id=result.result_id,
            topic_id=result.topic_id,
            topic_title=result.topic_title,
            package_id=result.package_id,
            claim_results=result.claim_results,
            package_verdict=result.package_verdict,
            approved_claims=result.approved_claims[:1],
            run_id=result.run_id,
        ).validate()
        generation = _generate(package, starved)
        assert generation.status == STATUS_INSUFFICIENT
        assert generation.blocked_reasons == ("too_few_approved_claims",)


class TestFactCheckGateHelper:
    """The shared gate returns a blocked result for REJECT/REVIEW, else None."""

    def test_pass_returns_none(self):
        _, result = bound_pair(package_verdict="PASS")
        assert fact_check_gate(result, ScriptConfig()) is None

    def test_reject_is_blocked(self):
        _, result = bound_pair(package_verdict="REJECT")
        blocked = fact_check_gate(result, ScriptConfig())
        assert blocked is not None
        assert blocked.status == STATUS_BLOCKED
        assert blocked.blocked_reasons == ("package_verdict=REJECT",)
        # REJECT is final: the opt-in does not unlock it.
        assert fact_check_gate(result, ScriptConfig(allow_review_content=True)) is not None

    def test_review_blocked_by_default_unlocked_by_opt_in(self):
        _, result = bound_pair(package_verdict="REVIEW")
        blocked = fact_check_gate(result, ScriptConfig())
        assert blocked is not None
        assert blocked.status == STATUS_BLOCKED
        assert blocked.blocked_reasons == ("package_verdict=REVIEW",)
        assert (
            fact_check_gate(result, ScriptConfig(allow_review_content=True)) is None
        )


def _result_for(package: ResearchPackage, template: FactCheckResult) -> FactCheckResult:
    """A result bound to `package` but carrying `template`'s verdicts."""
    return FactCheckResult(
        result_id="fc_othertopic000",
        topic_id=package.topic_id,
        topic_title=package.topic_title,
        package_id=package.package_id,
        claim_results=template.claim_results,
        package_verdict=template.package_verdict,
        approved_claims=template.approved_claims,
        run_id=template.run_id,
    ).validate()


# --------------------------------------------------------------------------- #
# 8-9. preview path and mock provider unchanged
# --------------------------------------------------------------------------- #


class TestPreviewPathUnchanged:
    def test_no_fact_check_result_keeps_result_id_empty_and_full_package(self):
        package = build_five_claim_package()
        ids = [claim.claim_id for claim in package.claims]
        fake = FakeRawClient()
        fake.add_result_json(_payload_for(package.claims, ids[:3]))
        generation = _provider(fake).generate(
            package, run_id="preview", created_at=FIXED_AT
        )
        assert generation.status == STATUS_GENERATED, generation.message
        script = generation.script
        assert script.result_id == ""
        assert set(script.unused_approved_claim_ids) == set(ids[3:])

    def test_mock_provider_without_result_is_unchanged(self):
        package = build_five_claim_package()
        generation = MockScriptProvider().generate(
            package, run_id="preview", created_at=FIXED_AT
        )
        assert generation.generated, generation.message
        assert generation.script.result_id == ""
        assert generation.script.package_id == package.package_id

    def test_mock_provider_honours_the_gate_when_bound(self):
        package, result = bound_pair()
        generation = MockScriptProvider().generate(
            package, run_id="bind-test", created_at=FIXED_AT, fact_check_result=result
        )
        assert generation.generated, generation.message
        assert generation.script.result_id == result.result_id
        assert set(generation.script.used_claim_ids) <= set(result.approved_claims)


# --------------------------------------------------------------------------- #
# 6-7. CLI gate + no partial artifact
# --------------------------------------------------------------------------- #


class TestCliRequiresFactCheckResult:
    def _package_path(self, tmp_path) -> Path:
        path = tmp_path / "research_package.json"
        path.write_text(build_five_claim_package().to_json(), encoding="utf-8")
        return path

    def _result_path(self, tmp_path) -> Path:
        from script_test_utils import fact_result_from_fixture

        path = tmp_path / "fact_check_result.json"
        path.write_text(fact_result_from_fixture().to_json(), encoding="utf-8")
        return path

    def test_missing_result_exits_error_and_writes_nothing(
        self, monkeypatch, tmp_path, capsys
    ):
        package_path = self._package_path(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "script_generator",
                "--provider", "mock",
                "--research", str(package_path),
                "--run-id", "cli_no_result",
                "--out", str(tmp_path),
            ],
        )
        assert main() == 2
        assert not (tmp_path / "cli_no_result" / "scripts" / "script.json").exists()
        assert "requires --result" in capsys.readouterr().out

    def test_missing_result_file_exits_error_and_writes_nothing(
        self, monkeypatch, tmp_path, capsys
    ):
        package_path = self._package_path(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "script_generator",
                "--provider", "mock",
                "--research", str(package_path),
                "--result", str(tmp_path / "nope.json"),
                "--run-id", "cli_bad_result",
                "--out", str(tmp_path),
            ],
        )
        assert main() == 2
        assert not (tmp_path / "cli_bad_result" / "scripts" / "script.json").exists()
        assert "FactCheckResult not found" in capsys.readouterr().out

    def test_invalid_result_file_exits_error_and_writes_nothing(
        self, monkeypatch, tmp_path, capsys
    ):
        package_path = self._package_path(tmp_path)
        bad = tmp_path / "bad_result.json"
        bad.write_text(json.dumps({"schema": "FactCheckResult"}), encoding="utf-8")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "script_generator",
                "--provider", "mock",
                "--research", str(package_path),
                "--result", str(bad),
                "--run-id", "cli_invalid_result",
                "--out", str(tmp_path),
            ],
        )
        assert main() == 2
        assert not (tmp_path / "cli_invalid_result" / "scripts" / "script.json").exists()
        assert "invalid FactCheckResult" in capsys.readouterr().out

    def test_preview_and_result_are_mutually_exclusive(
        self, monkeypatch, tmp_path, capsys
    ):
        package_path = self._package_path(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "script_generator",
                "--provider", "mock",
                "--research", str(package_path),
                "--result", str(self._result_path(tmp_path)),
                "--preview",
                "--run-id", "cli_both",
                "--out", str(tmp_path),
            ],
        )
        assert main() == 2
        assert "mutually exclusive" in capsys.readouterr().out

    def test_bound_cli_run_writes_a_bound_script(
        self, monkeypatch, tmp_path, capsys
    ):
        """The happy path: --research + --result produces a bound Script."""
        from pipeline.state import PipelineState

        monkeypatch.setattr(PipelineState, "save", lambda self, path: Path(path))
        package, result = bound_pair()
        package_path = tmp_path / "pkg.json"
        package_path.write_text(package.to_json(), encoding="utf-8")
        result_path = tmp_path / "fc.json"
        result_path.write_text(result.to_json(), encoding="utf-8")

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "script_generator",
                "--provider", "mock",
                "--research", str(package_path),
                "--result", str(result_path),
                "--run-id", "cli_bound",
                "--out", str(tmp_path),
            ],
        )
        assert main() == 0, capsys.readouterr().out
        artifact = json.loads(
            (tmp_path / "cli_bound" / "scripts" / "script.json").read_text(encoding="utf-8")
        )
        assert artifact["data"]["result_id"] == result.result_id
        assert artifact["data"]["package_id"] == package.package_id
        approved = set(result.approved_claims)
        refs = {
            ref
            for line in artifact["data"]["lines"]
            for ref in line["claim_refs"]
        }
        assert refs <= approved


# --------------------------------------------------------------------------- #
# Security: nothing secret may leak into the bound artifact
# --------------------------------------------------------------------------- #


class TestNoSecretLeak:
    def test_api_key_never_appears_in_a_bound_script(self):
        package, result = bound_pair()
        secret = "super-secret-key-value"
        client = make_client(
            GeminiConfig(api_key=secret), raw_client=_scripted(package, result), sleep=noop_sleep
        )
        generation = GeminiScriptProvider(client=client).generate(
            package,
            run_id="bind-test",
            created_at=FIXED_AT,
            fact_check_result=result,
        )
        assert generation.generated, generation.message
        serialized = generation.script.to_json()
        assert secret not in serialized
        assert secret not in " ".join(generation.blocked_reasons)


def _scripted(package, result) -> FakeRawClient:
    approved = [claim.claim_id for claim in approved_claims(result, package)]
    fake = FakeRawClient()
    fake.add_result_json(_payload_for(package.claims, approved))
    return fake
