"""Serialization + provenance tests (requirements 10 and 11)."""

from __future__ import annotations

import json

from agents.seo.result import TopicSelection
from agents.seo.history import SelectionHistory
from agents.seo.stage import run_topic_selection


def _build_selection(mock_candidates, config, run_id):
    return run_topic_selection(
        mock_candidates,
        SelectionHistory(),
        config,
        run_id=run_id,
        seed=7,
        created_at="2026-09-20T08:00:00+00:00",
    )


class TestProvenance:
    def test_artifact_header_has_provenance_metadata(self, default_config, mock_candidates):
        selection = _build_selection(mock_candidates, default_config, run_id="run_test_0001")
        artifact = selection.to_dict()
        for key in ("schema", "schema_version", "created_at", "run_id", "stage", "produced_by"):
            assert key in artifact, f"missing provenance header field: {key}"
        assert artifact["schema"] == "TopicSelection"
        assert artifact["schema_version"] == 1
        assert artifact["stage"] in ("topic_selection", "topic_discovery", "seo_research")
        assert artifact["run_id"] == "run_test_0001"
        assert artifact["created_at"] == "2026-09-20T08:00:00+00:00"
        assert set(artifact["produced_by"]) == {"prompt_version", "provider", "model"}

    def test_candidate_provenance_kept_in_artifact(self, default_config, mock_candidates):
        selection = _build_selection(mock_candidates, default_config, run_id="run_test_0002")
        selected = selection.to_dict()["data"]["selected_topic"]
        provenance = selected.get("provenance")
        assert provenance is not None
        assert provenance["generated_by"] == "mock_catalog_v1"
        assert provenance["source"] == "internal_mock"
        assert provenance["created_at"]

    def test_scoring_configuration_included(self, default_config, mock_candidates):
        selection = _build_selection(mock_candidates, default_config, run_id="run_test_0003")
        scoring = selection.to_dict()["data"]["scoring_configuration"]
        assert scoring["version"] == default_config.version
        assert scoring["scoring"]["weights"] == default_config.scoring.weights
        assert scoring["selection"]["seed"] == default_config.selection.seed


class TestSerialization:
    def test_json_round_trip_preserves_artifact(self, default_config, mock_candidates):
        selection = _build_selection(mock_candidates, default_config, run_id="run_test_0004")
        dumped = json.dumps(selection.to_dict())
        rebuilt = TopicSelection.from_json(dumped)
        assert rebuilt.to_dict() == selection.to_dict()

    def test_json_round_trip_via_string(self, default_config, mock_candidates):
        selection = _build_selection(mock_candidates, default_config, run_id="run_test_0005")
        rebuilt = TopicSelection.from_json(selection.to_json())
        assert rebuilt.to_json() == selection.to_json()