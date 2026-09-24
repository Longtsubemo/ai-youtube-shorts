"""REAL Gemini integration tests — gated by GEMINI_INTEGRATION_TESTS=1.

These require a genuine GEMINI_API_KEY and make real network calls, so they
are excluded from the normal offline suite. They validate one end-to-end
Stage-1 run against the real provider.

Run explicitly with:
    GEMINI_INTEGRATION_TESTS=1 python -m pytest tests/test_gemini_integration.py -q
"""

from __future__ import annotations

import os

import pytest

from agents.seo.config import load_config
from agents.seo.history import SelectionHistory
from agents.seo.stage import run_topic_selection

GATE = os.environ.get("GEMINI_INTEGRATION_TESTS") == "1"
REQUIRES_KEY = pytest.mark.skipif(
    not GATE, reason="set GEMINI_INTEGRATION_TESTS=1 to run real Gemini calls"
)


@REQUIRES_KEY
def test_real_stage1_run_with_gemini_provider():
    from agents.seo.provider import build_topic_candidate_provider

    config = load_config("config/topic_selection.yaml")
    provider = build_topic_candidate_provider("gemini", config=config)
    candidates = provider.list_candidates()
    assert len(candidates) >= 1

    selection = run_topic_selection(candidates, SelectionHistory(), config, run_id="gem_int_01")
    selection.produced_by = dict(provider.produced_by)
    artifact = selection.to_dict()
    assert artifact["produced_by"]["provider"] == "gemini"
    assert artifact["produced_by"]["model"] == provider.model
    assert artifact["produced_by"]["prompt_version"] == provider.prompt_version
    assert artifact["data"]["selected_topic"]["topic_id"].startswith("t_")
    # Secrets never leak into the artifact even on a real run.
    api_key = os.environ.get("GEMINI_API_KEY", "")
    assert api_key not in selection.to_json()


@REQUIRES_KEY
def test_real_stage2_research_with_gemini_provider():
    from agents.research.provider import build_research_provider
    from agents.research.stage import load_topic_selection, run_research

    selection = load_topic_selection("data/mock/topic_selection.json")
    provider = build_research_provider("gemini")
    package = run_research(selection, provider, run_id="gem_int_research")
    package.produced_by = dict(provider.produced_by)
    package.validate()

    artifact = package.to_dict()
    assert artifact["produced_by"]["provider"] == "gemini"
    assert artifact["produced_by"]["model"] == provider.model
    assert artifact["produced_by"]["prompt_version"] == provider.prompt_version
    assert artifact["data"]["research_status"] in ("provisional", "insufficient_evidence", "ready_for_fact_check")
    assert len(artifact["data"]["research_notes"]) >= 1
    assert any("grounded" in note or "provisional" in note for note in artifact["data"]["research_notes"])

    # Every source URL must be a real http(s) URL returned by grounding
    # (never a fabricated URL) and no two claims share a claim id.
    claim_ids = set()
    for source in artifact["data"]["sources"]:
        assert source["url"].startswith(("http://", "https://"))
    for claim in artifact["data"]["claims"]:
        assert claim["claim_id"] not in claim_ids
        claim_ids.add(claim["claim_id"])
        assert claim["source_ids"], "every real claim must be grounded to a source"

    api_key = os.environ.get("GEMINI_API_KEY", "")
    assert api_key not in package.to_json()


@REQUIRES_KEY
def test_real_stage3_fact_check_with_gemini_provider():
    from agents.qa.fact_checker.checker import FactChecker
    from agents.qa.fact_checker.provider import build_fact_check_components
    from agents.research.models import ResearchPackage

    components = build_fact_check_components("gemini", retriever="firecrawl")
    package = ResearchPackage.from_json(
        "data/857acc781211-firecrawl-lite/research/research_package.json"
    )
    provider, verifier = components.provider, components.verifier
    result = FactChecker(provider, verifier=verifier).check(package)

    artifact = result.to_dict()
    assert artifact["produced_by"]["provider"] == "gemini"
    assert artifact["produced_by"]["model"] == provider.model
    assert artifact["produced_by"]["prompt_version"] == provider.prompt_version
    assert artifact["data"]["provenance"]["model"] == provider.model
    assert artifact["data"]["package_verdict"] in ("pass", "review", "reject")
    assert artifact["data"]["claim_results"], "every claim must get a verdict"
    for claim in artifact["data"]["claim_results"]:
        assert claim["verdict"] in ("PASS", "REVIEW", "REJECT")
        assert isinstance(claim["rationale"], str) and claim["rationale"]
    # Secrets never leak into the artifact even on a real run.
    api_key = os.environ.get("GEMINI_API_KEY", "")
    assert api_key not in result.to_json()


@REQUIRES_KEY
def test_real_stage4_script_with_gemini_provider():
    from agents.research.models import ResearchPackage
    from agents.script.provider import build_script_provider

    package = ResearchPackage.from_json(
        "data/857acc781211-firecrawl-lite/research/research_package.json"
    )
    provider = build_script_provider("gemini")
    result = provider.generate(package, run_id="gem_int_script")

    assert result.status in ("generated", "blocked")
    assert result.script is not None or result.blocked_reasons
    if result.generated:
        script = result.script
        artifact = script.to_dict()
        assert artifact["produced_by"]["provider"] == "gemini"
        assert artifact["produced_by"]["model"] == provider.model
        assert artifact["produced_by"]["prompt_version"] == provider.prompt_version
        assert artifact["data"]["package_id"] == package.package_id
        assert artifact["data"]["result_id"] == ""
        # Secrets never leak into the artifact even on a real run.
        api_key = os.environ.get("GEMINI_API_KEY", "")
        assert api_key not in script.to_json()
    else:
        assert result.blocked_reasons
        assert all(isinstance(reason, str) and reason for reason in result.blocked_reasons)