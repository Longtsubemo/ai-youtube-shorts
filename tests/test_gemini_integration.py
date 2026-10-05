"""REAL Gemini integration test - gated by GEMINI_INTEGRATION_TESTS=1.

Requires a genuine GEMINI_API_KEY and makes a real network call, so it is
excluded from the normal offline suite. It validates one end-to-end Stage 1
(topic selection) run against the real provider.

Run explicitly with:
    GEMINI_INTEGRATION_TESTS=1 python -m pytest tests/test_gemini_integration.py -q

The removed stages (AI research via retrieval, fact checking, script QA,
video QA) are not covered here: they are no longer part of the architecture.
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