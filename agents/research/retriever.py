"""ResearchRetriever abstraction: real web retrieval, decoupled from Gemini.

The Gemini research provider chooses between two retrieval strategies:

1. Google Search grounding built into the Gemini request (no retriever), or
2. an explicit retrieval step (`ResearchRetriever`) whose returned sources
   are handed to Gemini for claim extraction in a separate structured call.

Firecrawl is such a retriever. `RetrievedSource` is the evidence boundary:
URLs, titles, and publishers come from real search-result metadata, never from
model memory. `MockResearchRetriever` is a deterministic, network-free stand-in
for offline tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agents.research.provider import TopicDescriptor


@dataclass(frozen=True)
class RetrievedSource:
    """One real web hit returned by a retriever, before claim extraction.

    `domain` is derived from the URL's host. `content` holds bounded Markdown
    (so the extraction prompt stays deterministic and small). This is the
    evidence a downstream claim-extractor is allowed to ground on — anything
    not here is NOT evidence.
    """

    url: str
    title: str = ""
    publisher: str = ""
    publication_date: str = ""
    source_type: str = "web_page"
    content: str = ""
    domain: str = ""
    retrieved_at: str = ""


class ResearchRetriever(Protocol):
    """Anything that can fetch real, bounded web results for a topic.

    Implementations must be deterministic in their bounds (bounded result
    count, bounded content, explicit timeout, explicit failure handling) and
    must never fabricate URLs or content.
    """

    retriever_name: str

    def retrieve(self, topic: TopicDescriptor) -> list[RetrievedSource]: ...


class MockResearchRetriever:
    """Deterministic, network-free retriever for offline tests.

    Returns the injected sources verbatim, or a fixed example.test fixture set
    when none are provided. URLs use `example.test` so packages never touch
    the open web in tests.
    """

    retriever_name = "mock"

    _FIXTURE = (
        RetrievedSource(
            url="https://example.test/papers/octopus-hearts",
            title="Why octopuses have three hearts",
            publisher="example.test",
            source_type="peer_reviewed_paper",
            content="Octopuses have three hearts: two branchial hearts and a "
            "systemic heart. Their blood uses copper-based hemocyanin.",
            domain="example.test",
        ),
        RetrievedSource(
            url="https://example.test/news/octopus-blood",
            title="Octopus blood is blue",
            publisher="example.test",
            source_type="news_article",
            content="Octopus blood appears blue because it is based on copper, "
            "not iron.",
            domain="example.test",
        ),
    )

    def __init__(self, sources: list[RetrievedSource] | None = None) -> None:
        self._sources = list(sources) if sources is not None else list(self._FIXTURE)

    def retrieve(self, topic: TopicDescriptor) -> list[RetrievedSource]:
        return list(self._sources)