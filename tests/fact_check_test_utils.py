"""Shared builders/stubs for the Stage-3 (fact-check) test suite.

Everything here is synthetic and MOCK - no real source exists behind these
URLs and no claim is asserted to be true. The builders exist so tests can
construct precise claim/evidence scenarios offline.
"""

from __future__ import annotations

from agents.qa.fact_checker.evidence import EvidenceProvider
from agents.qa.fact_checker.models import Evidence, evidence_id_for
from agents.research.models import (
    FactClaim,
    ResearchPackage,
    Source,
    claim_id_for,
    source_id_for,
)

FIXED_AT = "2026-09-20T08:00:00+00:00"


def make_source(url="https://example.test/source/s001", tier="primary", **overrides) -> Source:
    """Deterministic mock Source for a fake URL."""
    data = {
        "source_id": source_id_for(url),
        "title": "Mock source title",
        "url": url,
        "publisher": "Mock Publisher",
        "source_type": "scientific_paper",
        "publication_date": "2024-01-01",
        "credibility_tier": tier,
        "retrieved_at": FIXED_AT,
        "notes": "MOCK fixture - not a real source.",
    }
    data.update(overrides)
    return Source.from_dict(data)


def make_claim(claim_text="Mock claim text.", source_urls=(), **overrides) -> FactClaim:
    """Deterministic mock FactClaim citing deterministic source ids."""
    data = {
        "claim_id": claim_id_for(claim_text),
        "claim_text": claim_text,
        "claim_type": "fact",
        "source_ids": [source_id_for(url) for url in source_urls],
        "confidence": 0.5,
        "verification_status": "provisional",
        "dates": [],
        "notes": "",
        "critical": False,
    }
    data.update(overrides)
    return FactClaim.from_dict(data)


def make_evidence(
    source: Source,
    excerpt: str,
    evidence_type: str = "direct_statement",
    locator: str = "loc:1",
    **overrides,
) -> Evidence:
    """Deterministic mock Evidence attached to a real (mock) Source."""
    data = {
        "evidence_id": evidence_id_for(source.source_id, excerpt),
        "source_id": source.source_id,
        "excerpt": excerpt,
        "locator": locator,
        "evidence_type": evidence_type,
        "retrieved_at": source.retrieved_at,
        "notes": "",
    }
    data.update(overrides)
    return Evidence.from_dict(data)


def make_package(
    claims=(),
    sources=(),
    topic_id="t_test",
    topic_title="Test topic",
    research_status="ready_for_fact_check",
) -> ResearchPackage:
    return ResearchPackage.build(
        topic_id=topic_id,
        topic_title=topic_title,
        research_status=research_status,
        claims=list(claims),
        sources=list(sources),
        run_id="test-run",
        created_at=FIXED_AT,
    )


class StaticEvidenceProvider:
    """Offline EvidenceProvider stub returning fixed evidence per source URL."""

    def __init__(self, mapping: dict | None = None) -> None:
        self._mapping = mapping or {}
        self.calls: list[str] = []

    def get_evidence(self, source: Source) -> list[Evidence]:
        self.calls.append(source.url)
        return list(self._mapping.get(source.url, []))