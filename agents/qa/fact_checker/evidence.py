"""EvidenceProvider abstraction and its deterministic mock implementation.

Stage 3's golden rule: a **source is not evidence**. The fact checker never
operates on bare URLs; it operates on `Evidence` objects the provider returns
for a given source. This module owns that boundary.

    EvidenceProvider.get_evidence(source) -> tuple[Evidence, ...]

A real provider (web scraper, search API, PDF extractor, RAG store) can be
dropped in later without touching the checker's core logic. The mock provider
reads local fixture text only (data/mock/fact_check_evidence.json): it makes
no network requests and returns clearly-marked fake evidence (example.test
sources).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from agents.qa.fact_checker.models import Evidence, FactCheckError, evidence_id_for
from agents.research.models import FactClaim, Source

DEFAULT_EVIDENCE_DATA_PATH = Path("data") / "mock" / "fact_check_evidence.json"

# Which evidence types count as *supporting* material for a claim.
SUPPORTING_EVIDENCE_TYPES = ("direct_statement", "supporting_context")


class EvidenceProviderError(ValueError):
    """Raised when an evidence provider cannot fulfill a request."""


class EvidenceProvider(Protocol):
    """Anything that can return evidence content for a single source."""

    def get_evidence(self, source: Source) -> list[Evidence]: ...


class EvidenceRecoverer(Protocol):
    """Optional claim-aware recall backstop for an evidence provider.

    `get_evidence` is claim-blind: it returns whatever excerpts the extractor
    chose, so a source whose supporting sentence was simply not selected looks
    exactly like a source that does not support the claim. A provider that also
    holds the retrieved content can implement this to recover a near-verbatim
    restatement of a specific claim from text it already has.

    Implementations MUST NOT retrieve new content, search for new sources, call
    a model, change a source tier, or return evidence bound to any source other
    than the one passed in. Recovered evidence is ordinary `Evidence`: it is
    still credited only by the verifier and still faces the full gates.
    """

    def recover_evidence(self, source: Source, claim: FactClaim) -> list[Evidence]: ...


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


_STOPWORDS = frozenset(
    {
        "a", "an", "the", "of", "to", "in", "for", "on", "with", "and", "or", "by",
        "how", "why", "what", "when", "where", "who", "which", "does", "do", "did",
        "is", "are", "was", "were", "be", "been", "actually", "really", "can", "could",
        "will", "would", "it", "its", "their", "we", "you", "that", "this", "than",
    }
)


def meaningful_tokens(text: str) -> set[str]:
    """Tokenize text into meaningful (non-stopword) lowercase tokens."""
    return {token for token in _normalize(text).split() if token not in _STOPWORDS}


class MockEvidenceProvider:
    """Deterministic, network-free evidence provider backed by fixture text.

    Fixture layout (data/mock/fact_check_evidence.json):

        {"schema": "mock_fact_check_evidence", "schema_version": 1,
         "sources": [ { "url": "...", "excerpts": [ {...}, ... ] }, ... ]}

    Each excerpt maps to an `Evidence` for the source whose URL matches. The
    provider never invents, paraphrases from memory, or goes online; if the
    source is not in the fixture the answer is "no evidence" (empty list).
    """

    def __init__(
        self,
        data_path: str | Path = DEFAULT_EVIDENCE_DATA_PATH,
        data: dict | None = None,
    ) -> None:
        self._data_path = Path(data_path)
        self._data = data if data is not None else self._load(self._data_path)

    def _load(self, path: Path) -> dict:
        if not path.is_file():
            raise EvidenceProviderError(f"mock evidence fixture not found: {path}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise EvidenceProviderError(
                f"invalid evidence fixture {path}: {exc}"
            ) from None
        if not isinstance(raw, dict) or not isinstance(raw.get("sources"), list):
            raise EvidenceProviderError(f"invalid evidence fixture layout: {path}")
        return raw

    @property
    def sources(self) -> list[dict]:
        return self._data["sources"]

    def get_evidence(self, source: Source) -> list[Evidence]:
        entries = []
        for raw_source in self.sources:
            if raw_source.get("url") != source.url:
                continue
            for index, raw_excerpt in enumerate(raw_source.get("excerpts") or []):
                try:
                    entries.append(
                        Evidence(
                            evidence_id=evidence_id_for(
                                source.source_id,
                                raw_excerpt.get("excerpt") or "",
                            ),
                            source_id=source.source_id,
                            excerpt=str(raw_excerpt.get("excerpt") or ""),
                            locator=str(raw_excerpt.get("locator") or ""),
                            evidence_type=str(raw_excerpt.get("evidence_type") or ""),
                            retrieved_at=source.retrieved_at,
                            notes=str(raw_excerpt.get("notes") or ""),
                        ).validate()
                    )
                except (FactCheckError,) as exc:
                    raise EvidenceProviderError(
                        f"malformed evidence excerpt {index} for "
                        f"{raw_source.get('url')}: {exc}"
                    ) from None
        return entries