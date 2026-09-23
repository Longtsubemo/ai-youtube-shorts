"""Offline tests for MockScriptProvider (package-based script generation).

The mock provider is the default lens for the fact-check-free path: a whole
ResearchPackage is the working set, the deterministic package validator is the
final gate, and generation makes no network requests.
"""

from __future__ import annotations

import socket

from agents.research.models import ResearchPackage
from agents.script.provider import MockScriptProvider
from agents.script.script_generator import (
    STATUS_GENERATED,
    STATUS_INSUFFICIENT,
    ScriptConfig,
)
from agents.script.validator import validate_script_from_package
from fact_check_test_utils import make_claim, make_package, make_source
from script_test_utils import (
    OCTOPUS_BLUE,
    OCTOPUS_SMARTEST,
    OCTOPUS_SWIM,
    OCTOPUS_TANK,
    OCTOPUS_THREE_HEARTS,
)

SOURCE_URL = "https://example.test/papers/octopus"

FIVE_CLAIMS = (
    (OCTOPUS_THREE_HEARTS, "fact"),
    (OCTOPUS_SWIM, "explanation"),
    (OCTOPUS_BLUE, "explanation"),
    (OCTOPUS_SMARTEST, "inference"),
    (OCTOPUS_TANK, "speculation"),
)


def build_package(
    specs=FIVE_CLAIMS,
    *,
    run_id="pkg-test",
    topic_id="t_octopus_hearts",
    topic_title="Why octopuses have three hearts",
) -> ResearchPackage:
    """Deterministic package whose claims cite one real-ish (mock) source."""
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


def test_generated_shape():
    package = build_package()
    generation = MockScriptProvider().generate(package, run_id="mock-shape")
    assert generation.status == STATUS_GENERATED, generation.message
    assert generation.generated
    assert generation.validation is not None and generation.validation.valid
    script = generation.script
    assert script.script_id.startswith("sc_")
    assert script.title == "Why octopuses have three hearts"
    assert script.result_id == ""  # no fact-check result exists on this path
    assert script.package_id == package.package_id
    assert script.provenance is not None and script.provenance.model == "MockScriptProvider"
    assert script.produced_by["model"] == "MockScriptGenerator"
    assert set(script.used_claim_ids) == {c.claim_id for c in package.claims}
    assert script.unused_approved_claim_ids == ()
    assert 30.0 <= script.estimated_duration_seconds <= 60.0


def test_defaults_run_id_to_package():
    package = build_package(run_id="pkg-run")
    script = MockScriptProvider().generate(package).script
    assert script.run_id == "pkg-run"
    explicit = MockScriptProvider().generate(package, run_id="other-run").script
    assert explicit.run_id == "other-run"


def test_too_few_claims_insufficient():
    package = build_package(specs=((OCTOPUS_BLUE, "explanation"),))
    generation = MockScriptProvider().generate(package)
    assert generation.status == STATUS_INSUFFICIENT
    assert not generation.generated
    assert generation.blocked_reasons == ("too_few_approved_claims",)
    assert generation.script is None


def test_too_short_insufficient():
    package = build_package(specs=((OCTOPUS_BLUE, "explanation"),))
    config = ScriptConfig(min_approved_claims=1)
    generation = MockScriptProvider(config=config).generate(package)
    assert generation.status == STATUS_INSUFFICIENT
    assert generation.blocked_reasons == ("too_short",)


def test_generated_script_passes_package_validator():
    package = build_package()
    generation = MockScriptProvider().generate(package)
    validation = validate_script_from_package(
        generation.script, package, ScriptConfig()
    )
    assert validation.valid, validation.errors


def test_deterministic_across_runs():
    package = build_package()
    first = MockScriptProvider().generate(package, run_id="det", created_at="2026-09-20T08:00:00+00:00")
    second = MockScriptProvider().generate(package, run_id="det", created_at="2026-09-20T08:00:00+00:00")
    assert first.script.to_json() == second.script.to_json()


def test_sockets_blocked_full_offline_run(monkeypatch):
    def _blocked(*args, **kwargs):
        raise OSError("network blocked in tests")

    monkeypatch.setattr(socket, "socket", _blocked)
    package = build_package()
    generation = MockScriptProvider().generate(package)
    assert generation.generated