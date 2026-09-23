"""Stage-2 Gemini research provider.

Implements the `ResearchProvider` protocol from `agents/research/provider.py`
using the provider foundation in `providers/gemini/client.py`.

Retrieval strategies (docs/GEMINI_PROVIDER_DESIGN.md §I):

    TopicDescriptor
        -> Gemini + Google Search grounding (primary, no retriever)
        -> optional URL Context (deeper retrieval of supplied URLs, opt-in)
        OR, with a `ResearchRetriever` injected (Firecrawl):
        -> retriever.retrieve(topic)  (real web search, bounded)
        -> Gemini structured extraction over the retrieved content
        -> model-proposed claims cross-checked against REAL retrieval metadata
        -> SourceCandidate[] for the existing deterministic ResearchAgent

Evidence boundary (identical for both strategies):

- Only URLs that the retrieval actually returned count as sources. Google
  Search grounding/URL-context for the grounding path; the Firecrawl
  retrieved-source list for the retrieval path.
- The model's own generated text/URLs are NOT evidence. A source proposed by
  the model whose URL is not in the retrieval evidence is dropped (never
  invented); if no evidence exists at all the provider returns no candidates,
  so the package honestly lands on `insufficient_evidence`.
- Claims live inside a source record, so a claim only ever attaches to a
  source the retrieval actually returned.
- Claim/source fields are Pydantic-validated against the existing research
  contract (claim types, credibility tiers, date kinds), so malformed output
  fails cleanly as a GeminiProviderError, never a raw SDK exception.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, TypeAdapter

from agents.research.provider import ClaimItem, SourceCandidate, TopicDescriptor
from agents.research.retriever import RetrievedSource, ResearchRetriever
from providers.gemini.client import GeminiClient

PROMPT_VERSION = "research/facts-v1"
RETRIEVAL_PROMPT_VERSION = "research/facts-retrieval-v1"
DEFAULT_PROMPT_PATH = Path("prompts") / "research" / "facts-v1.txt"
RETRIEVAL_PROMPT_PATH = Path("prompts") / "research" / "facts-retrieval-v1.txt"
GENERATED_BY = "gemini_research_provider_v1"
DEFAULT_MAX_SOURCES = 8
DEFAULT_MAX_RETRIEVED_SOURCES = 4

_SYSTEM_PROMPT = (
    "You are a research assistant for a YouTube Shorts channel. The enabled "
    "Google Search grounding tool retrieves real sources; your generated text "
    "is NOT evidence — only the sources the retrieval actually returned count. "
    "Never invent URLs, sources, citations, or dates. You never verify facts: "
    "your output is provisional for a downstream fact-check stage. Return "
    "JSON only."
)

_SYSTEM_PROMPT_RETRIEVAL = (
    "You are a research assistant for a YouTube Shorts channel. Real web "
    "content was retrieved for you and is provided in the prompt; your "
    "generated text is NOT evidence — only the retrieved sources provided in "
    "the prompt count. Never invent URLs, sources, citations, or dates. You "
    "never verify facts: your output is provisional for a downstream "
    "fact-check stage. Return JSON only."
)

_GROUNDING_NOTE = "Grounded by Gemini Google-Search retrieval (URL returned by grounding, not invented)."
_RETRIEVAL_NOTE = "Sourced from Firecrawl web retrieval (URL returned by retrieval metadata, not invented)."

_CLAIM_TYPES = ("fact", "inference", "explanation", "speculation", "opinion")
_SOURCE_TIERS = ("primary", "high_quality_secondary", "discovery")
_DATE_KINDS = ("publication_date", "discovery_date", "announcement_date", "event_date")


class _ResearchDateSchema(BaseModel):
    date_kind: Literal["publication_date", "discovery_date", "announcement_date", "event_date"]
    date: str = Field(min_length=1)


class _ResearchClaimSchema(BaseModel):
    claim_text: str = Field(min_length=1)
    dates: list[_ResearchDateSchema] = Field(default_factory=list)
    suggested_claim_type: Optional[Literal["fact", "inference", "explanation", "speculation", "opinion"]] = None
    critical: bool = False


class _ResearchSourceSchema(BaseModel):
    url: str = Field(min_length=1)
    title: str = ""
    publisher: str = ""
    source_type: str = "web_page"
    publication_date: str = ""
    credibility_tier: Literal["primary", "high_quality_secondary", "discovery"] = "discovery"
    notes: str = ""
    claims: list[_ResearchClaimSchema] = Field(default_factory=list)


class ResearchResultSchema(BaseModel):
    """The response schema mirroring the existing ResearchPackage contract.

    Sources are proposed by the model but only survive if their URL is in the
    retrieval metadata; claims live inside each source record.
    """

    sources: list[_ResearchSourceSchema] = Field(default_factory=list)


# Convenience re-export for tests.
ResearchResultAdapter = TypeAdapter(ResearchResultSchema)


def _normalize_url(url: str) -> str:
    """Lowercase, strip trailing slash, require an http(s) scheme."""
    url = url.strip()
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
            return ""
        path = parsed.path.rstrip("/") or "/"
        return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}"
    except ValueError:
        return ""


def _is_iso(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    candidate = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        datetime.fromisoformat(candidate)
    except ValueError:
        return False
    return True


def _netloc(url: str) -> str:
    try:
        return urlparse(url).netloc
    except ValueError:  # pragma: no cover - defensive
        return ""


class GeminiResearchProvider:
    """Research provider backed by Gemini + Google Search grounding."""

    provider_name = "gemini"

    def __init__(
        self,
        client: GeminiClient,
        *,
        prompt_path: str | Path = DEFAULT_PROMPT_PATH,
        max_sources: int = DEFAULT_MAX_SOURCES,
        enable_url_context: bool = False,
        retriever: ResearchRetriever | None = None,
        retrieval_prompt_path: str | Path = RETRIEVAL_PROMPT_PATH,
        max_retrieved_sources: int = DEFAULT_MAX_RETRIEVED_SOURCES,
    ) -> None:
        self._client = client
        self._prompt_path = Path(prompt_path)
        self._max_sources = max_sources
        self._enable_url_context = enable_url_context
        self._retriever = retriever
        self._retrieval_prompt_path = Path(retrieval_prompt_path)
        self._max_retrieved_sources = max_retrieved_sources

    # -- metadata --------------------------------------------------------- #

    @property
    def model(self) -> str:
        return self._client.model

    @property
    def prompt_version(self) -> str:
        return RETRIEVAL_PROMPT_VERSION if self._retriever is not None else PROMPT_VERSION

    @property
    def retriever_name(self) -> str | None:
        return getattr(self._retriever, "retriever_name", None) if self._retriever is not None else None

    @property
    def produced_by(self) -> dict:
        """Header provenance for the ResearchPackage envelope."""
        info: dict = {
            "prompt_version": self.prompt_version,
            "provider": self.provider_name,
            "model": self.model,
        }
        if self._retriever is not None:
            info["retrieved_by"] = self.retriever_name
        return info

    # -- protocol --------------------------------------------------------- #

    def search(self, topic: TopicDescriptor) -> list[SourceCandidate]:
        if self._retriever is not None:
            return self._search_retrieval(topic)
        return self._search_grounding(topic)

    # -- internals --------------------------------------------------------- #

    def _render_prompt(self, topic: TopicDescriptor) -> str:
        path = Path(self._prompt_path)
        if not path.is_file():
            raise FileNotFoundError(f"prompt template not found: {path}")
        return (
            path.read_text(encoding="utf-8")
            .replace("{topic_id}", topic.topic_id)
            .replace("{title}", topic.title)
            .replace("{pillar}", topic.pillar)
            .replace("{topic_type}", topic.topic_type)
            .replace("{description}", topic.description)
            .replace("{max_sources}", str(self._max_sources))
        )

    def _search_grounding(self, topic: TopicDescriptor) -> list[SourceCandidate]:
        prompt = self._render_prompt(topic)
        response = self._client.generate_research(
            prompt,
            ResearchResultSchema,
            system_prompt=_SYSTEM_PROMPT,
            enable_url_context=self._enable_url_context,
        )

        evidence: set[str] = set()
        meta_by_url: dict[str, object] = {}
        for chunk in response.grounding:
            key = _normalize_url(chunk.url)
            if key:
                evidence.add(key)
                meta_by_url[key] = chunk
        for url in response.url_context_urls:
            key = _normalize_url(url)
            if key:
                evidence.add(key)
        return self._map_candidates(response.data.sources, evidence, meta_by_url, _GROUNDING_NOTE)

    def _search_retrieval(self, topic: TopicDescriptor) -> list[SourceCandidate]:
        """Firecrawl path: retrieve real evidence first, then extract claims.

        Retrieved sources are sliced to `max_retrieved_sources` so the prompt
        stays bounded and deterministic. If the retriever returns nothing we
        return no candidates (honest `insufficient_evidence`) and never call
        the model.
        """
        retrieved = self._retriever.retrieve(topic)
        if not retrieved:
            return []

        selected = retrieved[: self._max_retrieved_sources]
        evidence: set[str] = set()
        meta_by_url: dict[str, RetrievedSource] = {}
        for source in selected:
            key = _normalize_url(source.url)
            if key:
                evidence.add(key)
                meta_by_url[key] = source

        prompt = self._render_retrieval_prompt(topic, selected)
        extracted = self._client.generate_structured(
            prompt,
            ResearchResultSchema,
            system_prompt=_SYSTEM_PROMPT_RETRIEVAL,
        )
        return self._map_candidates(extracted.sources, evidence, meta_by_url, _RETRIEVAL_NOTE)

    def _map_candidates(
        self,
        sources: list,
        evidence: set[str],
        meta_by_url: dict[str, object],
        note: str,
    ) -> list[SourceCandidate]:
        """Normalize model output into SourceCandidates gated by `evidence`.

        A model-proposed source is kept ONLY when its URL is in the retrieval
        evidence. Sources without a retrieved URL are dropped — never invented.
        Claims are embedded in their source, and duplicate claim text across
        the result is collapsed to one (claim ids derive from text, so
        duplicates would break the package).
        """
        candidates: list[SourceCandidate] = []
        seen_claims: set[str] = set()
        for source in sources:
            key = _normalize_url(source.url)
            if not key or key not in evidence:
                continue
            candidates.append(self._candidate(source, meta_by_url.get(key), seen_claims, note))
        return candidates

    def _candidate(
        self,
        source,
        meta: object | None,
        seen_claims: set[str],
        note: str,
    ) -> SourceCandidate:
        url = source.url.strip()
        meta_title = str(getattr(meta, "title", "") or "")
        meta_domain = str(getattr(meta, "domain", "") or "")
        meta_publisher = str(getattr(meta, "publisher", "") or "")
        title = source.title.strip() or meta_title.strip()
        publisher = (
            source.publisher.strip()
            or meta_publisher.strip()
            or meta_domain.strip()
            or _netloc(url)
        )
        publication_date = source.publication_date.strip() if _is_iso(source.publication_date) else ""

        claims: list[ClaimItem] = []
        for item in source.claims:
            text = item.claim_text.strip()
            if not text or text.lower() in seen_claims:
                continue
            seen_claims.add(text.lower())
            claims.append(
                ClaimItem(
                    claim_text=text,
                    dates=tuple(date.model_dump() for date in item.dates),
                    suggested_claim_type=item.suggested_claim_type,
                    critical=item.critical,
                )
            )

        notes = note
        if source.notes.strip():
            notes = f"{notes} | {source.notes.strip()}"

        return SourceCandidate(
            title=title,
            url=url,
            publisher=publisher,
            source_type=source.source_type.strip() or "web_page",
            publication_date=publication_date,
            credibility_tier=source.credibility_tier,
            claims=tuple(claims),
            notes=notes,
            retrieved_at="",
        )

    # -- retrieval prompt rendering ---------------------------------------- #

    def _render_retrieval_prompt(self, topic: TopicDescriptor, selected: list[RetrievedSource]) -> str:
        path = Path(self._retrieval_prompt_path)
        if not path.is_file():
            raise FileNotFoundError(f"prompt template not found: {path}")
        return (
            path.read_text(encoding="utf-8")
            .replace("{retrieved_sources}", self._render_retrieved_sources(selected))
            .replace("{max_sources}", str(len(selected)))
            .replace("{topic_id}", topic.topic_id)
            .replace("{title}", topic.title)
            .replace("{pillar}", topic.pillar)
            .replace("{topic_type}", topic.topic_type)
            .replace("{description}", topic.description)
        )

    @staticmethod
    def _render_retrieved_sources(selected: list[RetrievedSource]) -> str:
        """Render retrieved sources as a bounded, deterministic text block."""
        lines: list[str] = []
        for index, source in enumerate(selected, start=1):
            lines.append(f"[{index}] {source.url}")
            if source.title:
                lines.append(f"    title: {source.title}")
            if source.publisher:
                lines.append(f"    publisher: {source.publisher}")
            if source.publication_date:
                lines.append(f"    publication_date: {source.publication_date}")
            content = (source.content or "").strip()
            if content:
                lines.append("    content:")
                lines.extend(f"    {line}" for line in content.splitlines())
        return "\n".join(lines)


def claim_text_digest(text: str) -> str:
    """Stable digest helper reused by tests for dedup assertions."""
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:10]