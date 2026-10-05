"""Offline tests for GeminiScriptProvider.

Exercises the full provider boundary with a fake raw client: valid output
generates a deterministic Script that passes the package validator; unknown
refs, off-claim wording, dropped hedges, and strengthened certainty all fail
closed as a `blocked` generation; malformed output/API errors surface as the
provider error hierarchy, never raw SDK exceptions.
"""

from __future__ import annotations

import socket
import subprocess
import sys

import pytest

from agents.research.models import ResearchPackage
from agents.script.script_generator import (
    STATUS_BLOCKED,
    STATUS_GENERATED,
    ScriptConfig,
)
from agents.script.models import PURPOSE_VALUES
from providers.gemini.config import GeminiConfig
from providers.gemini.errors import (
    GeminiAPIError,
    GeminiMalformedResponseError,
    GeminiProviderError,
    GeminiSchemaValidationError,
)
from providers.gemini.script_provider import (
    DEFAULT_PROMPT_PATH,
    GENERATED_BY,
    PROMPT_VERSION,
    GeminiScriptProvider,
)
from tests.gemini_test_utils import FakeRawClient, make_client, noop_sleep
from fact_check_test_utils import FIXED_AT, make_claim, make_package, make_source
from script_test_utils import (
    OCTOPUS_BLUE,
    OCTOPUS_SMARTEST,
    OCTOPUS_SWIM,
    OCTOPUS_THREE_HEARTS,
)

SOURCE_URL = "https://example.test/papers/octopus"

DEFAULT_SPECS = (
    (OCTOPUS_THREE_HEARTS, "fact"),
    (OCTOPUS_SWIM, "explanation"),
    (OCTOPUS_BLUE, "explanation"),
)


def build_package(
    specs=DEFAULT_SPECS,
    *,
    run_id="gem-script-test",
    topic_id="t_octopus_hearts",
    topic_title="Why octopuses have three hearts",
) -> ResearchPackage:
    source = make_source(url=SOURCE_URL, tier="primary")
    claims = [
        make_claim(text, claim_type=ctype, source_urls=[SOURCE_URL])
        for text, ctype in specs
    ]
    return make_package(
        claims=claims,
        sources=[source],
        run_id=run_id,
        topic_id=topic_id,
        topic_title=topic_title,
    ).validate()


def _provider(fake: FakeRawClient, **kwargs) -> GeminiScriptProvider:
    client = make_client(GeminiConfig(api_key="k"), raw_client=fake, sleep=noop_sleep)
    return GeminiScriptProvider(client=client, **kwargs)


def _claim_ids(package: ResearchPackage) -> list[str]:
    return [claim.claim_id for claim in package.claims]


def _valid_payload(
    package: ResearchPackage,
    *,
    fact_indices=(0, 1, 2),
    title="Why octopuses have three hearts",
) -> dict:
    """A schema-valid payload whose fact lines quote package claims verbatim."""
    claims = list(package.claims)
    ids = _claim_ids(package)
    lines = [
        {"purpose": "hook", "text": "You probably didn't know this.", "claim_refs": []},
        {"purpose": "context", "text": "Here is the answer we gathered.", "claim_refs": []},
    ]
    for purpose, index in zip(("fact", "fact", "fact"), fact_indices):
        lines.append(
            {
                "purpose": purpose,
                "text": claims[index].claim_text,
                "claim_refs": [ids[index]],
            }
        )
    lines.append({"purpose": "transition", "text": "Here is one more detail.", "claim_refs": []})
    lines.append({"purpose": "call_to_action", "text": "Follow for more.", "claim_refs": []})
    return {"title": title, "lines": lines}


class TestGeneratedOutput:
    def test_valid_payload_generates(self):
        package = build_package()
        fake = FakeRawClient()
        fake.add_result_json(_valid_payload(package))
        generation = _provider(fake).generate(package, run_id="gem-gen", created_at=FIXED_AT)
        assert generation.status == STATUS_GENERATED, generation.message
        assert generation.generated
        assert generation.validation is not None and generation.validation.valid
        script = generation.script
        assert script.title == "Why octopuses have three hearts"
        assert script.result_id == ""
        assert script.package_id == package.package_id
        assert script.run_id == "gem-gen"
        assert script.produced_by == {
            "prompt_version": PROMPT_VERSION,
            "provider": "gemini",
            "model": script.provenance.model,
        }
        assert script.provenance is not None
        assert script.provenance.generated_by == GENERATED_BY
        assert script.provenance.provider == "gemini"
        assert script.provenance.model == script.produced_by["model"]
        assert script.provenance.prompt_version == PROMPT_VERSION
        assert {p for line in script.lines for p in (line.purpose,)} <= set(PURPOSE_VALUES)
        assert set(script.used_claim_ids) == set(_claim_ids(package))
        assert script.unused_approved_claim_ids == ()

    def test_durations_derived_deterministically_from_words(self):
        package = build_package()
        fake = FakeRawClient()
        fake.add_result_json(_valid_payload(package))
        script = _provider(fake).generate(package).script
        for line in script.lines:
            expected = round(len(line.text.split()) / 150 * 60, 2)
            assert line.estimated_duration_seconds == expected

    def test_partial_usage_tracks_unused_claims(self):
        package = build_package()
        fake = FakeRawClient()
        fake.add_result_json(_valid_payload(package, fact_indices=(0, 1)))
        generation = _provider(fake).generate(package)
        assert generation.generated, generation.message
        script = generation.script
        assert set(script.used_claim_ids) == set(_claim_ids(package)[:2])
        assert script.unused_approved_claim_ids == (_claim_ids(package)[2],)
        assert script.estimated_duration_seconds > 0

    def test_title_truncated_to_seventy(self):
        package = build_package()
        fake = FakeRawClient()
        fake.add_result_json(_valid_payload(package, title="x" * 100))
        script = _provider(fake).generate(package).script
        assert len(script.title) <= 70

    def test_package_run_id_used_when_no_run_given(self):
        package = build_package(run_id="pkg-gem-run")
        fake = FakeRawClient()
        fake.add_result_json(_valid_payload(package))
        script = _provider(fake).generate(package).script
        assert script.run_id == "pkg-gem-run"


class TestFinalGateBlocks:
    def test_unknown_claim_ref_blocks(self):
        package = build_package()
        payload = _valid_payload(package)
        payload["lines"][2]["claim_refs"] = ["cl_no_such_claim"]
        fake = FakeRawClient()
        fake.add_result_json(payload)
        generation = _provider(fake).generate(package)
        assert generation.status == STATUS_BLOCKED
        assert not generation.generated
        assert any("references unknown claim" in error for error in generation.blocked_reasons)

    def test_out_of_coverage_wording_blocks(self):
        package = build_package()
        payload = _valid_payload(package)
        payload["lines"][2]["text"] = (
            "These creatures boast unrelated details nobody included in our claims at all."
        )
        fake = FakeRawClient()
        fake.add_result_json(payload)
        generation = _provider(fake).generate(package)
        assert generation.status == STATUS_BLOCKED
        assert any("wording not in its approved claims" in error for error in generation.blocked_reasons)

    def test_dropped_hedge_blocks(self):
        package = build_package(specs=((OCTOPUS_SMARTEST, "inference"), (OCTOPUS_BLUE, "explanation")))
        payload = _valid_payload(package, fact_indices=(0, 1))
        payload["lines"][2]["text"] = "Octopuses are the smartest invertebrates in the sea."
        payload["lines"][2]["claim_refs"] = [_claim_ids(package)[0]]
        fake = FakeRawClient()
        fake.add_result_json(payload)
        generation = _provider(fake).generate(package)
        assert generation.status == STATUS_BLOCKED
        assert any("drops the hedge" in error for error in generation.blocked_reasons)

    def test_retained_hedge_generates(self):
        package = build_package(specs=((OCTOPUS_SMARTEST, "inference"), (OCTOPUS_BLUE, "explanation")))
        payload = _valid_payload(package, fact_indices=(0, 1))
        payload["lines"][2]["text"] = OCTOPUS_SMARTEST
        payload["lines"][2]["claim_refs"] = [_claim_ids(package)[0]]
        fake = FakeRawClient()
        fake.add_result_json(payload)
        generation = _provider(fake).generate(package)
        assert generation.status == STATUS_GENERATED, generation.message

    def test_strengthened_certainty_blocks(self):
        package = build_package()
        payload = _valid_payload(package)
        payload["lines"][2]["text"] = OCTOPUS_THREE_HEARTS + " This is proved beyond doubt."
        fake = FakeRawClient()
        fake.add_result_json(payload)
        generation = _provider(fake).generate(package)
        assert generation.status == STATUS_BLOCKED
        assert any("strengthens its claims" in error for error in generation.blocked_reasons)


class TestSchemaStrictness:
    def test_invalid_purpose_raises_schema_error(self):
        package = build_package()
        payload = _valid_payload(package)
        payload["lines"][2]["purpose"] = "gossip"
        fake = FakeRawClient()
        fake.add_result_json(payload)
        with pytest.raises(GeminiSchemaValidationError):
            _provider(fake).generate(package, run_id="bad-purpose")

    def test_empty_lines_list_raises_schema_error(self):
        package = build_package()
        fake = FakeRawClient()
        fake.add_result_json({"title": "T", "lines": []})
        with pytest.raises(GeminiSchemaValidationError):
            _provider(fake).generate(package)

    def test_whitespace_line_maps_to_schema_error(self):
        package = build_package()
        fake = FakeRawClient()
        fake.add_result_json(
            {"title": "T", "lines": [{"purpose": "hook", "text": "   ", "claim_refs": []}]}
        )
        with pytest.raises(GeminiSchemaValidationError):
            _provider(fake).generate(package)

    def test_malformed_output_raises_provider_error(self):
        package = build_package()
        fake = FakeRawClient()
        fake.add_result("this is not json")
        with pytest.raises(GeminiProviderError):
            _provider(fake).generate(package)

    def test_malformed_output_retries_then_generates(self):
        package = build_package()
        fake = FakeRawClient()
        fake.add_result("this is not json")
        fake.add_result_json(_valid_payload(package))
        generation = _provider(fake).generate(package)
        assert generation.status == STATUS_GENERATED, generation.message
        assert fake.attempt_count == 2

    def test_malformed_output_raises_after_bounded_retries(self):
        package = build_package()
        fake = FakeRawClient()
        fake.add_result("this is not json")
        with pytest.raises(GeminiMalformedResponseError):
            _provider(fake).generate(package)
        assert fake.attempt_count == 2  # 1 + malformed_retries=1

    def test_api_error_maps_to_provider_error(self):
        from google.genai import errors as genai_errors

        package = build_package()
        fake = FakeRawClient()
        fake.add_error(genai_errors.ServerError(code=503, response_json={}))
        with pytest.raises(GeminiAPIError):
            _provider(fake).generate(package)

    def test_missing_prompt_template_raises(self):
        package = build_package()
        fake = FakeRawClient()
        fake.add_result_json(_valid_payload(package))
        with pytest.raises(FileNotFoundError):
            _provider(fake, prompt_path="prompts/missing/prompt.txt").generate(package)


class TestProvenanceAndWiring:
    def test_produced_by_metadata(self):
        provider = _provider(FakeRawClient())
        assert provider.produced_by == {
            "prompt_version": PROMPT_VERSION,
            "provider": "gemini",
            "model": provider.model,
        }
        assert provider.model
        assert provider.prompt_version == PROMPT_VERSION

    def test_prompt_rendering_injects_claims_only(self):
        package = build_package()
        provider = _provider(FakeRawClient())
        prompt = provider._render_prompt(package, ScriptConfig(), package.claims)
        assert "{claims}" not in prompt and "{topic_id}" not in prompt
        assert "{topic_title}" not in prompt and "{max_words}" not in prompt
        assert "Why octopuses have three hearts" in prompt
        for claim in package.claims:
            assert claim.claim_id in prompt
            assert claim.claim_text.split()[0] in prompt

    def test_sockets_blocked_full_offline_run(self, monkeypatch):
        def _blocked(*args, **kwargs):
            raise OSError("network blocked in tests")

        monkeypatch.setattr(socket, "socket", _blocked)
        package = build_package()
        fake = FakeRawClient()
        fake.add_result_json(_valid_payload(package))
        generation = _provider(fake).generate(package)
        assert generation.status == STATUS_GENERATED
        assert generation.script.produced_by["provider"] == "gemini"


class TestImportIsolation:
    def test_module_import_never_loads_sdk(self):
        code = (
            "import sys\n"
            "import providers.gemini.script_provider  # noqa: F401\n"
            "assert 'google.genai' not in sys.modules, 'google.genai imported transitively'\n"
            "print('ok')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=".")
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ok"