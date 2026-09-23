"""ResearchProvider abstraction and its deterministic mock implementation.

The ResearchAgent depends only on the `ResearchProvider` protocol (docs/-
RESEARCH_PIPELINE.md flow):

    TopicSelection -> ResearchAgent -> ResearchProvider

A real implementation (e.g. a WebResearchProvider over a search API) can be
dropped in later without touching the agent's core logic. The mock provider
serves local fixture data only; it makes no network requests and returns
realistic but clearly-marked fake sources (example.test URLs).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Protocol

from agents.research.models import ClaimDate

DEFAULT_RESEARCH_DATA_PATH = Path("data") / "mock" / "research_evidence.json"


class ResearchProviderError(ValueError):
    """Raised when a research provider cannot fulfill a search."""


@dataclass(frozen=True)
class TopicDescriptor:
    """Normalized topic identification passed to a provider search."""

    topic_id: str
    title: str
    description: str = ""
    pillar: str = ""
    topic_type: str = ""


@dataclass(frozen=True)
class ClaimItem:
    """A claim candidate extracted from one source page.

    The provider returns claim *candidates*; the agent decides the final
    claim (classification, confidence, id, verification status).
    """

    claim_text: str
    dates: tuple[dict, ...] = ()
    suggested_claim_type: str | None = None
    critical: bool = False


@dataclass(frozen=True)
class SourceCandidate:
    """One raw search-result page, before the agent normalizes it into a Source."""

    title: str
    url: str
    publisher: str
    source_type: str
    publication_date: str = ""
    credibility_tier: str = "primary"
    notes: str = ""
    claims: tuple[ClaimItem, ...] = ()
    retrieved_at: str = ""


class ResearchProvider(Protocol):
    """Anything that can return research source candidates for a topic.

    Implementations may later wrap a search API, a database, or an LLM
    extractor. The agent never sees the concrete class.
    """

    def search(self, topic: TopicDescriptor) -> list[SourceCandidate]: ...


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


_STOPWORDS = frozenset(
    {
        "a", "an", "the", "of", "to", "in", "for", "on", "with", "and", "or", "by",
        "how", "why", "what", "when", "where", "who", "which", "does", "do", "did",
        "is", "are", "was", "were", "be", "been", "actually", "really", "can", "could",
        "will", "would", "it", "its", "their", "we", "you",
    }
)


def _meaningful_tokens(text: str) -> set[str]:
    return {token for token in _normalize(text).split() if token not in _STOPWORDS}


class MockResearchProvider:
    """Deterministic, network-free research provider backed by fixture data.

    Fixture layout (data/mock/research_evidence.json):

        {"schema": "mock_research_evidence", "schema_version": 1,
         "topics": [{"topic_id": ..., "title": ..., "sources": [source...]}]}

    Each source entry mirrors SourceCandidate fields; `claims` are ClaimItem
    dicts. Lookup is by exact topic_id first, then by normalized title, then
    by keyword overlap of the topic title against fixture titles.
    """

    def __init__(
        self,
        data_path: str | Path = DEFAULT_RESEARCH_DATA_PATH,
        data: dict | None = None,
    ) -> None:
        self._data_path = Path(data_path)
        self._data = data if data is not None else self._load(self._data_path)

    def _load(self, path: Path) -> dict:
        if not path.is_file():
            raise ResearchProviderError(f"mock research fixture not found: {path}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ResearchProviderError(f"invalid research fixture {path}: {exc}") from None
        if not isinstance(raw, dict) or not isinstance(raw.get("topics"), list):
            raise ResearchProviderError(f"invalid research fixture layout: {path}")
        return raw

    @property
    def topics(self) -> list[dict]:
        return self._data["topics"]

    def _find(self, topic: TopicDescriptor) -> dict | None:
        for entry in self.topics:
            if entry.get("topic_id") == topic.topic_id:
                return entry
        needle = _meaningful_tokens(topic.title)
        if needle:
            exact = next(
                (
                    entry
                    for entry in self.topics
                    if _meaningful_tokens(entry.get("title") or "") == needle
                ),
                None,
            )
            if exact is not None:
                return exact
            best, best_score = None, 0
            for entry in self.topics:
                overlap = len(needle & _meaningful_tokens(entry.get("title") or ""))
                if overlap > best_score:
                    best, best_score = entry, overlap
            if best is not None and best_score > 0:
                return best
        return None

    def search(self, topic: TopicDescriptor) -> list[SourceCandidate]:
        entry = self._find(topic)
        if entry is None or not isinstance(entry.get("sources"), list):
            return []
        candidates: list[SourceCandidate] = []
        for raw_source in entry["sources"]:
            raw_claims = raw_source.get("claims") or []
            candidates.append(
                SourceCandidate(
                    title=str(raw_source.get("title") or ""),
                    url=str(raw_source.get("url") or ""),
                    publisher=str(raw_source.get("publisher") or ""),
                    source_type=str(raw_source.get("source_type") or ""),
                    publication_date=str(raw_source.get("publication_date") or ""),
                    credibility_tier=str(raw_source.get("credibility_tier") or "primary"),
                    notes=str(raw_source.get("notes") or ""),
                    claims=tuple(
                        ClaimItem(
                            claim_text=str(item.get("claim_text") or ""),
                            dates=tuple(item.get("dates") or ()),
                            suggested_claim_type=item.get("suggested_claim_type"),
                            critical=bool(item.get("critical", False)),
                        )
                        for item in raw_claims
                    ),
                    retrieved_at=str(raw_source.get("retrieved_at") or ""),
                )
            )
        return candidates

    def knowable_topics(self) -> list[str]:
        return [entry["topic_id"] for entry in self.topics]


def build_research_provider(
    name: str | None,
    *,
    data_path: str | Path = DEFAULT_RESEARCH_DATA_PATH,
    env: Mapping[str, str] | None = None,
    enable_url_context: bool = False,
    retriever: str | None = None,
) -> ResearchProvider:
    """Return the research provider selected by `name` (default: mock).

    Only the requested provider is imported, so a mock/default run never
    needs or imports the Gemini/Firecrawl SDKs. Providing a real provider
    requires the GEMINI_API_KEY environment variable; a missing key raises
    GeminiConfigurationError rather than silently failing over to mock.

    `retriever` selects real web retrieval for the gemini provider:
    `"firecrawl"` builds a FirecrawlResearchRetriever (keyless-capable). No
    other provider accepts a retriever.
    """
    if name in (None, "", "mock"):
        if retriever is not None:
            raise ResearchProviderError("--retriever is only supported with the gemini provider")
        return MockResearchProvider(data_path=data_path)

    if name == "gemini":
        from providers.gemini.client import GeminiClient
        from providers.gemini.config import GeminiConfig
        from providers.gemini.research_provider import GeminiResearchProvider

        gemini_config = GeminiConfig.from_env(env if env is not None else os.environ)
        client = GeminiClient(gemini_config)

        built_retriever = None
        if retriever == "firecrawl":
            from providers.firecrawl.config import FirecrawlConfig
            from providers.firecrawl.retriever import FirecrawlResearchRetriever

            firecrawl_config = FirecrawlConfig.from_env(env if env is not None else os.environ)
            built_retriever = FirecrawlResearchRetriever(config=firecrawl_config)
        elif retriever is not None:
            raise ResearchProviderError(f"unknown retriever: {retriever!r}")

        return GeminiResearchProvider(
            client=client,
            enable_url_context=enable_url_context,
            retriever=built_retriever,
        )

    raise ResearchProviderError(f"unknown research provider: {name!r}")