"""Behavioral tests for the evidence layer (deterministic, offline)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.qa.fact_checker.evidence import (
    DEFAULT_EVIDENCE_DATA_PATH,
    EvidenceProvider,
    EvidenceProviderError,
    MockEvidenceProvider,
)
from agents.qa.fact_checker.models import Evidence
from agents.research.models import source_id_for

from fact_check_test_utils import make_source

EVIDENCE_FIXTURE = Path(__file__).resolve().parent.parent / "data" / "mock" / "fact_check_evidence.json"

OCT001 = "https://example.test/source/oct001"
OCT003 = "https://example.test/source/oct003"
UNKNOWN = "https://example.test/source/never-seen"


@pytest.fixture
def mock_evidence_provider():
    return MockEvidenceProvider(data_path=EVIDENCE_FIXTURE)


class TestMockEvidenceProvider:
    def test_fixture_loads_and_returns_evidence(self, mock_evidence_provider):
        source = make_source(OCT001)
        entries = mock_evidence_provider.get_evidence(source)
        assert len(entries) == 2
        for entry in entries:
            assert entry.source_id == source.source_id
            assert entry.source_id == source_id_for(OCT001)
            assert entry.retrieved_at == source.retrieved_at
            assert entry.evidence_type in (
                "direct_statement",
                "supporting_context",
                "contradictory_statement",
                "insufficient",
            )

    def test_unknown_source_returns_no_evidence(self, mock_evidence_provider):
        assert mock_evidence_provider.get_evidence(make_source(UNKNOWN)) == []

    def test_deterministic_across_calls(self, mock_evidence_provider):
        source = make_source(OCT001)
        first = mock_evidence_provider.get_evidence(source)
        second = mock_evidence_provider.get_evidence(source)
        assert [e.to_dict() for e in first] == [e.to_dict() for e in second]

    def test_discovery_source_evidence_usable(self, mock_evidence_provider):
        entries = mock_evidence_provider.get_evidence(make_source(OCT003))
        assert entries
        assert all(entry.evidence_type == "insufficient" for entry in entries)

    def test_fixture_file_missing_raises_clean_error(self, tmp_path):
        with pytest.raises(EvidenceProviderError, match="not found"):
            MockEvidenceProvider(data_path=tmp_path / "missing.json")

    def test_injected_data_works(self):
        provider = MockEvidenceProvider(
            data={"schema": "mock_fact_check_evidence", "schema_version": 1, "sources": [
                {"url": OCT001, "excerpts": [
                    {"excerpt": "A moth was taped into the Mark II logbook in 1947.",
                     "locator": "post 1", "evidence_type": "direct_statement"},
                ]},
            ]},
        )
        entries = provider.get_evidence(make_source(OCT001))
        assert entries[0].excerpt.startswith("A moth was taped")

    def test_malformed_fixture_raises_clean_error(self):
        provider = MockEvidenceProvider(
            data={"schema": "mock_fact_check_evidence", "schema_version": 1, "sources": [
                {"url": OCT001, "excerpts": [
                    {"locator": "p.1", "evidence_type": "direct_statement"},
                ]},
            ]},
        )
        with pytest.raises(EvidenceProviderError, match="malformed evidence excerpt 0"):
            provider.get_evidence(make_source(OCT001))

    def test_bad_fixture_layout_raises_clean_error(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('{"schema": "mock_fact_check_evidence"}', encoding="utf-8")
        with pytest.raises(EvidenceProviderError, match="layout"):
            MockEvidenceProvider(data_path=path)


class TestProviderProtocol:
    def test_protocol_duck_type(self):
        """get_evidence(source) is the only required method."""
        assert "get_evidence" in EvidenceProvider.__dict__ or hasattr(EvidenceProvider, "get_evidence")

    def test_mock_is_network_free(self, monkeypatch):
        """Prove the mock provider never touches the network."""

        def _block(*args, **kwargs):
            raise AssertionError("network access forbidden in tests")

        monkeypatch.setattr("socket.socket", _block, raising=True)
        monkeypatch.setattr("urllib.request.urlopen", _block, raising=True)
        provider = MockEvidenceProvider(data_path=EVIDENCE_FIXTURE)
        entries = provider.get_evidence(make_source(OCT001))
        assert entries
        assert all(isinstance(entry, Evidence) for entry in entries)