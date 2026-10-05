"""Stage-3 Gemini evidence provider.

Implements the `EvidenceProvider` protocol from
`agents/qa/fact_checker/evidence.py` using the provider foundation in
`providers/gemini/client.py`.

The checker calls `get_evidence(source)` once per source in the ResearchPackage.
Per-source verification retrieval (docs/GEMINI_PROVIDER_DESIGN.md §L):

    Source (url + metadata)
        -> retriever.retrieve(topic_for(source))   (Firecrawl; bounded, real web)
        -> the retrieved page that matches the source URL (strict normalized match)
        -> Gemini structured extraction of *verbatim* evidence excerpts
        -> deterministic containment check -> Evidence[] bound to the source

Evidence boundary:

- The model never proposes URLs, source ids, or sources. `Evidence.source_id`
  is bound to the package source being verified; `locator` and `notes` are the
  only free text fields, and neither is a URL.
- Every excerpt must be a verbatim (whitespace/markdown-normalized) substring
  of the retrieved content; anything else is model fabrication and is dropped.
- If the source's own URL is not among the retrieved pages, or no retriever is
  configured, the provider returns no evidence (honest `insufficient` ->
  REVIEW) and never calls the model.
- Excerpts are Pydantic-validated (evidence_type vocabulary, non-empty
  locator), so malformed model output fails cleanly as a GeminiProviderError,
  never a raw SDK exception.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from agents.qa.fact_checker.models import Evidence, evidence_id_for
from agents.qa.fact_checker.recovery import find_near_verbatim
from agents.research.models import FactClaim, Source
from agents.research.provider import TopicDescriptor
from agents.research.retriever import ResearchRetriever
from providers.gemini.client import GeminiClient

PROMPT_VERSION = "fact_check/evidence-v1"
DEFAULT_PROMPT_PATH = Path("prompts") / "fact_check" / "evidence-v1.txt"
GENERATED_BY = "gemini_evidence_provider_v1"
DEFAULT_MAX_EVIDENCE = 8

_SYSTEM_PROMPT = (
    "You are an evidence extractor for a fact-checking pipeline. "
    "Real web content was retrieved for you and is provided in the prompt; "
    "your generated text is NOT evidence. Only quote the retrieved content "
    "verbatim - every excerpt must appear word-for-word in the retrieved "
    "content. Never invent URLs, sources, or facts. Return JSON only."
)

EVIDENCE_NOTE_PREFIX = "Gemini-extracted from retrieved web content (verbatim; not invented)."
RECOVERY_NOTE_PREFIX = (
    "Deterministically recovered from retrieved web content "
    "(verbatim near-verbatim restatement; not invented)."
)
RECOVERY_LOCATOR_PREFIX = "deterministic-recovery"


class _EvidenceExcerptSchema(BaseModel):
    excerpt: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    evidence_type: Literal[
        "direct_statement",
        "supporting_context",
        "contradictory_statement",
        "insufficient",
    ] = "direct_statement"
    notes: str = ""


class EvidenceSetSchema(BaseModel):
    """Structured evidence extraction for one source page.

    The schema deliberately has NO url/source_id field: the model must never
    propose sources; `Evidence.source_id` is bound by the provider to the
    package source being verified.
    """

    excerpts: list[_EvidenceExcerptSchema] = Field(default_factory=list)


# Convenience re-export for tests.
EvidenceSetAdapter = EvidenceSetSchema


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
    except ValueError:  # pragma: no cover - defensive
        return ""


_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
# The label may contain escaped brackets (Wikipedia footnote refs: `[\[1\]](url)`),
# so the label body accepts `\[` / `\]` escapes rather than a bare `]`.
_MARKDOWN_LINK = re.compile(r"\[((?:[^\]\\]|\\.)*)\]\([^)]*\)")
_MARKDOWN_EMPHASIS = re.compile(r"(\*{1,3}|_{1,3})(\S(?:.*?\S)?)\1", re.DOTALL)
_MARKDOWN_ESCAPED_BRACKET = re.compile(r"\\([\[\]])")
_MARKDOWN_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_MARKDOWN_BULLET = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)


def _strip_markdown(text: str) -> str:
    """Render markdown markup down to the prose a reader actually sees.

    Retrieved pages arrive as markdown, but a model quoting "verbatim" text
    reproduces the rendered prose (it does not re-emit `[label](url)` link
    syntax or `**bold**` markers). Comparing a rendered excerpt against raw
    markdown source therefore fails on pages that are heavily linked, even
    though the words are genuinely present. Stripping markup on BOTH sides
    keeps the containment check strictly word-for-word while comparing the
    same representation: no word is added, reordered, or removed, so
    fabricated text still cannot pass.
    """
    text = _MARKDOWN_IMAGE.sub(" ", text)
    text = _MARKDOWN_LINK.sub(r"\1", text)
    text = _MARKDOWN_ESCAPED_BRACKET.sub(r"\1", text)
    text = _MARKDOWN_HEADING.sub("", text)
    text = _MARKDOWN_BULLET.sub("", text)
    text = _MARKDOWN_EMPHASIS.sub(r"\2", text)
    return text


def _normalize_whitespace(text: str) -> str:
    """Lowercase/collapse whitespace and strip edge punctuation before matching.

    Both sides are normalized the same way (including markdown markup), so a
    verbatim excerpt that differs only in markup or edge punctuation (a trailing
    '.', ':', '...') still counts as grounded text — but interior words and
    punctuation must match exactly.
    """
    collapsed = " ".join(_strip_markdown(text).strip().lower().split())
    return collapsed.strip(" .,;:!?\"'()[]{}\u2013\u2014")


class GeminiEvidenceProvider:
    """Evidence provider backed by Gemini + verification web retrieval."""

    provider_name = "gemini"

    def __init__(
        self,
        client: GeminiClient,
        *,
        prompt_path: str | Path = DEFAULT_PROMPT_PATH,
        retriever: ResearchRetriever | None = None,
        max_evidence_per_source: int = DEFAULT_MAX_EVIDENCE,
    ) -> None:
        self._client = client
        self._prompt_path = Path(prompt_path)
        self._retriever = retriever
        self._max_evidence_per_source = max_evidence_per_source
        # Retrieved content cached during get_evidence, keyed by source_id.
        # The deterministic backstop reads from here only: recovery must reuse
        # text that was ALREADY retrieved for this source and must never issue
        # a new retrieval, a new search, or a new model call.
        self._content_cache: dict[str, list[str]] = {}

    # -- metadata --------------------------------------------------------- #

    @property
    def model(self) -> str:
        return self._client.model

    @property
    def prompt_version(self) -> str:
        return PROMPT_VERSION

    @property
    def retriever_name(self) -> str | None:
        return (
            getattr(self._retriever, "retriever_name", None) if self._retriever is not None else None
        )

    @property
    def produced_by(self) -> dict:
        """Header provenance stamped by the fact-check envelope."""
        return {
            "prompt_version": self.prompt_version,
            "provider": self.provider_name,
            "model": self.model,
        }

    # -- protocol --------------------------------------------------------- #

    def get_evidence(self, source: Source) -> list[Evidence]:
        """Verify-oriented evidence extraction for a single package source.

        Returns `[]` when there is no retriever, the source's own URL was not
        retrieved, or the retrieved content yielded no grounded excerpts - in
        every case the deterministic gates turn that into an honest REVIEW.
        """
        if self._retriever is None:
            return []

        retrieved = self._retriever.retrieve(self._topic_for(source))
        if not retrieved:
            return []

        target = _normalize_url(source.url)
        if not target:
            return []
        matched = [
            item for item in retrieved if _normalize_url(item.url) == target and (item.content or "").strip()
        ]
        if not matched:
            return []

        self._content_cache[source.source_id] = [
            item.content for item in matched if (item.content or "").strip()
        ]

        prompt = self._render_prompt(source, matched)
        extracted = self._client.generate_structured(
            prompt,
            EvidenceSetSchema,
            system_prompt=_SYSTEM_PROMPT,
        )
        return self._map_evidence(source, extracted.excerpts, matched)

    # -- deterministic recall backstop ------------------------------------ #

    def recover_evidence(self, source: Source, claim: FactClaim) -> list[Evidence]:
        """Recover a near-verbatim restatement of `claim` from retrieved text.

        The recall backstop for the claim-blind extractor. `get_evidence` cannot
        know which sentence matters for a given claim, so a source whose
        supporting sentence simply was not selected is indistinguishable from a
        source that does not support the claim at all - which forces an honest
        but unhelpful REVIEW.

        Guarantees, in order:

        1. Operates only on content already retrieved for THIS source; no
           retrieval, no search, no model call (see `_content_cache`).
        2. Returns at most one candidate, and only a genuine near-verbatim
           restatement (order-preserving token match plus a contiguous run).
        3. The excerpt is a span of the retrieved content and must pass the same
           `_grounded` verbatim containment check the model-extracted excerpts
           pass. Nothing is reworded.
        4. `source_id` is bound to `source` - recovery cannot introduce a source.
        5. `evidence_type` is always `direct_statement`; recovery never claims
           context, contradiction, or insufficiency, which only the extractor
           and the verifier may assert.
        6. The id is the normal deterministic `evidence_id_for`, so a repeat
           call is idempotent and de-duplicates against existing evidence.
        7. The result is ordinary `Evidence`. It still has to be credited by the
           verifier and pass the gates; recovery never grants support itself.

        Returns `[]` when nothing qualifies, so behavior is unchanged whenever
        the extractor already did its job.
        """
        contents = self._content_cache.get(source.source_id, [])
        if not contents:
            return []
        excerpt = find_near_verbatim(claim.claim_text, "\n\n".join(contents))
        if not excerpt:
            return []
        if not self._grounded(excerpt, contents):
            return []
        return [
            Evidence(
                evidence_id=evidence_id_for(source.source_id, excerpt),
                source_id=source.source_id,
                excerpt=excerpt,
                locator=f"{RECOVERY_LOCATOR_PREFIX}: near-verbatim sentence in retrieved content",
                evidence_type="direct_statement",
                retrieved_at=source.retrieved_at,
                notes=RECOVERY_NOTE_PREFIX,
            ).validate()
        ]

    # -- internals --------------------------------------------------------- #

    def _topic_for(self, source: Source) -> TopicDescriptor:
        """Verification query targets the source's own URL, not free research."""
        return TopicDescriptor(
            topic_id=source.source_id,
            title=source.url,
            description=source.title,
        )

    def _map_evidence(
        self,
        source: Source,
        excerpts: list,
        matched: list,
    ) -> list[Evidence]:
        """Bind grounded, validated model excerpts to `source`.

        Deterministic rules applied after the Pydantic schema gate:
        - excerpt must appear verbatim (whitespace/markdown-normalized) in one of
          the retrieved page contents, otherwise it is dropped (never evidence);
        - excerpts are deduplicated by evidence_id;
        - output is bounded by `max_evidence_per_source`.
        """
        contents = [item.content for item in matched]
        evidence: list[Evidence] = []
        seen: set[str] = set()
        for item in excerpts:
            excerpt = item.excerpt.strip()
            if not excerpt or not self._grounded(excerpt, contents):
                continue
            evidence_id = evidence_id_for(source.source_id, excerpt)
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            notes = EVIDENCE_NOTE_PREFIX
            if item.notes.strip():
                notes = f"{notes} | {item.notes.strip()}"
            evidence.append(
                Evidence(
                    evidence_id=evidence_id,
                    source_id=source.source_id,
                    excerpt=excerpt,
                    locator=item.locator.strip(),
                    evidence_type=item.evidence_type,
                    retrieved_at=source.retrieved_at,
                    notes=notes,
                ).validate()
            )
            if len(evidence) >= self._max_evidence_per_source:
                break
        return evidence

    @staticmethod
    def _grounded(excerpt: str, contents: list[str]) -> bool:
        """Excerpt must be a verbatim substring of some retrieved content.

        Both sides pass through the same markdown/whitespace normalization, so
        rendered prose is compared against rendered prose.
        """
        needle = _normalize_whitespace(excerpt)
        if not needle:
            return False
        return any(needle in _normalize_whitespace(content) for content in contents)

    def _render_prompt(self, source: Source, matched: list) -> str:
        path = Path(self._prompt_path)
        if not path.is_file():
            raise FileNotFoundError(f"prompt template not found: {path}")
        return (
            path.read_text(encoding="utf-8")
            .replace("{source_id}", source.source_id)
            .replace("{source_url}", source.url)
            .replace("{source_title}", source.title)
            .replace("{max_evidence}", str(self._max_evidence_per_source))
            .replace("{retrieved_sources}", self._render_retrieved_sources(matched))
        )

    @staticmethod
    def _render_retrieved_sources(matched: list) -> str:
        """Render matched pages as a bounded, deterministic text block."""
        lines: list[str] = []
        for index, item in enumerate(matched, start=1):
            lines.append(f"[{index}] {item.url}")
            if item.title:
                lines.append(f"    title: {item.title}")
            if item.publisher:
                lines.append(f"    publisher: {item.publisher}")
            content = (item.content or "").strip()
            if content:
                lines.append("    content:")
                lines.extend(f"    {line}" for line in content.splitlines())
        return "\n".join(lines)