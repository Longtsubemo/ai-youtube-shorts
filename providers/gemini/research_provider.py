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
- One claim may cite several sources, via two independent mechanisms:

  1. Explicit - the model names other retrieved URLs in `also_supported_by`.
  2. Implicit - the model asserts the identical claim text under a second
     retrieved source. The claim is retained once under the first source that
     asserted it, and the later source becomes additional support. Requiring
     different registrable domains is what stops a syndicated copy of a single
     article from manufacturing corroboration.

  Only mechanism 2 is domain-gated here. Mechanism 1 keeps its previous
  behavior deliberately: it is the model asserting deliberate cross-source
  support, and narrowing it is a separate independence-policy decision, not part
  of this fix. See the note on `also_supported_by` in the module below.
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

from agents.research.models import normalize_source_url
from agents.research.provider import ClaimItem, SourceCandidate, TopicDescriptor
from agents.research.retriever import RetrievedSource, ResearchRetriever
from providers.gemini.client import GeminiClient

PROMPT_VERSION = "research/facts-v1"
RETRIEVAL_PROMPT_VERSION = "research/facts-retrieval-v1"
DEFAULT_PROMPT_PATH = Path("prompts") / "research" / "facts-v1.txt"
RETRIEVAL_PROMPT_PATH = Path("prompts") / "research" / "facts-retrieval-v1.txt"
GENERATED_BY = "gemini_research_provider_v1"
DEFAULT_MAX_SOURCES = 8
# Measured against real hybrid runs: slicing the retrieval prompt to 4 pages
# made Stage 2 emit only 1-3 sources, which cannot satisfy Stage 3's
# two-independent-source corroboration requirement for even one critical
# claim. Raising the slice to the retriever's own `DEFAULT_MAX_SOURCES` let the
# same prompts yield 5 sourced claims. This bounds the PROMPT, not the gates.
DEFAULT_MAX_RETRIEVED_SOURCES = 8

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
_RETRIEVAL_NOTE = "Sourced from web retrieval (URL returned by retrieval metadata, not invented)."

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
    # URLs of OTHER retrieved sources that support this same claim. Kept as
    # URLs (not ids) so the provider can re-check every one against the real
    # retrieval evidence before it reaches the agent; anything not returned
    # by retrieval is dropped here rather than invented downstream.
    #
    # NOTE: unlike the implicit same-statement merge, this explicit field is NOT
    # subject to the different-registrable-domain rule. Narrowing it would change
    # an existing, deliberately model-driven contract and is left as a separate
    # independence-policy decision rather than folded into this fix.
    also_supported_by: list[str] = Field(default_factory=list)


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
    """Lowercase, strip trailing slash, require an http(s) scheme.

    Delegates to the shared contract helper so the provider's evidence gate
    and the agent's claim->source resolution cannot disagree on URL identity.
    """
    return normalize_source_url(url)


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


# Approximate eTLD+1 for the implicit-corroboration independence rule. Stdlib
# only: a full public-suffix list is a dependency this project does not need,
# and the rule below is deliberately biased so that an unrecognized suffix can
# only ever make merging *stricter*.
_MULTI_LABEL_PUBLIC_SUFFIXES = frozenset(
    {
        "co.uk", "org.uk", "ac.uk", "gov.uk", "me.uk", "net.uk", "sch.uk",
        "com.au", "net.au", "org.au", "edu.au", "gov.au",
        "co.nz", "co.jp", "ne.jp", "or.jp", "ac.jp", "go.jp",
        "co.za", "org.za", "com.br", "com.cn", "net.cn", "org.cn",
        "gov.cn", "edu.cn", "com.hk", "com.tw", "com.sg", "com.mx",
        "com.ar", "com.tr", "com.pl", "com.ua", "co.kr", "co.in",
        "net.in", "org.in", "co.il", "com.my", "com.ph", "com.vn",
        "com.pk", "com.sa", "com.eg", "co.id",
    }
)


def _registrable_domain(url: str) -> str:
    """Approximate registrable domain (eTLD+1) of an http(s) URL.

    Deterministic and stdlib-only (`urlparse`), with no new dependency.

    Conservative by construction. Every uncertain case resolves to *the same*
    answer for two hosts that may not really be independent, so an
    unrecognized host can never gain implicit support:

    - an IP literal is compared whole, so `1.2.3.4` and `5.6.3.4` stay distinct;
    - a multi-label public suffix missing from the table falls back to the last
      two labels, so two different sites under the same unknown suffix look
      identical and are refused;
    - a host with no dot, or one that cannot be parsed, is returned as-is.

    Merging is only ever permitted between two hosts this function reports as
    different, and a false "different" would have to survive the first and
    second rules above, which is why they fail closed.
    """
    text = str(url or "").strip()
    if not text:
        return ""
    try:
        parsed = urlparse(text)
    except ValueError:
        return ""
    host = (parsed.hostname or "").lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return ""
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    if labels[-1].isdigit():  # IPv4/IPv6 literal: never share a numeric tail
        return host
    tail_two = ".".join(labels[-2:])
    if tail_two in _MULTI_LABEL_PUBLIC_SUFFIXES:
        return ".".join(labels[-3:])
    return tail_two


def _claim_dedup_key(claim_text: str) -> str:
    """Normalization used to decide two claims are the same statement.

    Case-folded and whitespace-collapsed, so a re-worded casing or a re-wrapped
    sentence from a second source is recognised as the same claim. Deliberately
    conservative: anything short of this (a paraphrase, a different sentence)
    stays a separate claim, which can only *under*-report corroboration.
    """
    return " ".join(str(claim_text).strip().lower().split())


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
    def last_retrieval_trace(self):
        """Audit record for the most recent `retrieve()` call, if any.

        `retrieve()` runs at most once per `search()` and there is no retry
        loop, so `last_trace` is the whole story for a run rather than the tail of
        several attempts. Returns None when no retriever is configured, or when
        the retriever records no trace (the Firecrawl retriever does not). Read
        only: nothing here re-runs or alters retrieval, so persisting the trace
        cannot perturb the sources it describes.
        """
        if self._retriever is None:
            return None
        return getattr(self._retriever, "last_trace", None)

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
        return self._map_candidates(
            response.data.sources, evidence, meta_by_url, _GROUNDING_NOTE, {}
        )

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
        merge_stats: dict = {}
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
        candidates = self._map_candidates(
            extracted.sources, evidence, meta_by_url, _RETRIEVAL_NOTE, merge_stats
        )
        self._record_selection(selected, extracted.sources, candidates, evidence, merge_stats)
        return candidates

    def _record_selection(
        self,
        selected: list[RetrievedSource],
        model_sources: list,
        candidates: list[SourceCandidate],
        evidence: set[str],
        merge_stats: dict | None = None,
    ) -> None:
        """Record which offered sources the model kept, for the retrieval trace.

        Observation only. Nothing here influences the returned candidates, and a
        retriever without a trace (or with a trace that does not accept the
        payload) simply records nothing, so the grounding gate is unchanged
        whether or not instrumentation is present.
        """
        record = getattr(self.last_retrieval_trace, "record_selection", None)
        if not callable(record):
            return

        kept_keys = [_normalize_url(candidate.url) for candidate in candidates]
        kept_set = set(kept_keys)

        selected_records = [
            {
                "url": source.url,
                "normalized_url": _normalize_url(source.url),
                "source_type": source.source_type,
                "content_chars": len(source.content or ""),
                "kept": _normalize_url(source.url) in kept_set,
            }
            for source in selected
        ]

        proposed_records = []
        dropped_by_evidence_gate = []
        discarded = []
        record_discarded = getattr(self.last_retrieval_trace, "record_model_discarded", None)
        for source in model_sources:
            key = _normalize_url(getattr(source, "url", ""))
            entry = {
                "url": getattr(source, "url", ""),
                "normalized_url": key,
                "source_type": getattr(source, "source_type", ""),
                "credibility_tier": getattr(source, "credibility_tier", None),
            }
            proposed_records.append(entry)
            if not key or key not in evidence:
                reason = "not_in_retrieval_evidence"
                dropped_by_evidence_gate.append({**entry, "reason": reason})
                discarded.append(
                    {
                        "url": entry["url"],
                        "canonical_url": key or entry["url"],
                        "tier": getattr(source, "credibility_tier", None),
                        "reason": reason,
                    }
                )
                if callable(record_discarded):
                    try:
                        record_discarded(
                            url=entry["url"],
                            canonical_url=key or entry["url"],
                            tier=getattr(source, "credibility_tier", None),
                            reason=reason,
                        )
                    except Exception:
                        pass
            elif key not in kept_set:
                reason = "discarded_by_model_selection"
                discarded.append(
                    {
                        "url": entry["url"],
                        "canonical_url": key,
                        "tier": getattr(source, "credibility_tier", None),
                        "reason": reason,
                    }
                )
                if callable(record_discarded):
                    try:
                        record_discarded(
                            url=entry["url"],
                            canonical_url=key,
                            tier=getattr(source, "credibility_tier", None),
                            reason=reason,
                        )
                    except Exception:
                        pass

        record(
            {
                "offered_count": len(selected),
                "offered": selected_records,
                "model_proposed_count": len(proposed_records),
                "model_proposed": proposed_records,
                "kept_count": len(candidates),
                "kept": [
                    {
                        "url": candidate.url,
                        "normalized_url": key,
                        "source_type": candidate.source_type,
                        "credibility_tier": candidate.credibility_tier,
                    }
                    for candidate, key in zip(candidates, kept_keys)
                ],
                "discarded_count": sum(1 for item in selected_records if not item["kept"]),
                "discarded": [item for item in selected_records if not item["kept"]],
                "dropped_by_evidence_gate_count": len(dropped_by_evidence_gate),
                "dropped_by_evidence_gate": dropped_by_evidence_gate,
                "model_discarded": discarded,
                "corroboration": self._corroboration_view(candidates, merge_stats),
            }
        )
        # Also attach to trace's model_discarded list directly if it has one
        trace = self.last_retrieval_trace
        if trace is not None:
            try:
                if not hasattr(trace, "model_discarded"):
                    setattr(trace, "model_discarded", [])
                # already appended via record_model_discarded above in some cases; ensure list exists
            except Exception:
                pass

    @staticmethod
    def _corroboration_view(candidates: list[SourceCandidate], merge_stats: dict | None) -> dict:
        """Aggregate claim-level corroboration counters for the retrieval trace.

        Additive instrumentation, added because the previous trace could not
        distinguish "Gemini declared no extra support" from "extra support was
        declared and then dropped" — which is exactly the ambiguity that made
        live under-binding undiagnosable.

        Counts only. No model claim text, no URLs, no credentials: the block is
        integers derived from the parsed structured output, so it stays safe to
        persist and deterministic across runs.

        Field meanings:
          model_claims_seen              claim objects in the model output
          claims_emitted                 distinct statements after de-duplication
          claims_single_source           under-declared: no extra support at all
          claims_multi_source            bound to 2+ sources by either path
          max_sources_per_claim          widest corroboration achieved
          explicit_support_declared      URLs Gemini put in also_supported_by
          explicit_support_kept          those that survived retrieval evidence
          duplicate_claims               identical statement seen under 2+ sources
          duplicate_merged               of those, merged as corroboration
          duplicate_rejected_same_domain of those, refused as same-domain
          merged_support_kept            merged URLs that survived the evidence gate
        """
        stats = merge_stats or {}
        widths = [1 + len(claim.also_supported_by_urls) for c in candidates for claim in c.claims]
        return {
            "model_claims_seen": int(stats.get("model_claims_seen", 0)),
            "claims_emitted": int(stats.get("claims_emitted", 0)),
            "claims_single_source": int(stats.get("claims_single_source", 0)),
            "claims_multi_source": int(stats.get("claims_multi_source", 0)),
            "max_sources_per_claim": max(widths) if widths else 0,
            "explicit_support_declared": int(stats.get("explicit_support_declared", 0)),
            "explicit_support_kept": int(stats.get("explicit_support_kept", 0)),
            "duplicate_claims": int(stats.get("duplicate_claims", 0)),
            "duplicate_merged": int(stats.get("duplicate_merged", 0)),
            "duplicate_rejected_same_domain": int(stats.get("duplicate_rejected_same_domain", 0)),
            "merged_support_kept": int(stats.get("merged_support_kept", 0)),
        }

    def _map_candidates(
        self,
        sources: list,
        evidence: set[str],
        meta_by_url: dict[str, object],
        note: str,
        merge_stats: dict | None = None,
    ) -> list[SourceCandidate]:
        """Normalize model output into SourceCandidates gated by `evidence`.

        A model-proposed source is kept ONLY when its URL is in the retrieval
        evidence. Sources without a retrieved URL are dropped — never invented.

        Duplicate claim text is **merged, not discarded**. When the model
        asserts the same claim under two different retrieved sources, the claim
        is retained once under the first source that asserted it (first-wins
        ownership, so provenance and ordering are unchanged) and the later
        source is recorded as additional support via the existing
        `also_supported_by` path. That keeps one claim per statement — which
        `claim_id_for` requires, since claim ids derive from the text — while
        turning corroboration the model already expressed into real source
        multiplicity instead of throwing it away.

        Merging requires two *independent* registrable domains, so a syndicated
        or mirrored copy of one article cannot manufacture corroboration.
        Explicitly declared `also_supported_by` is a separate mechanism and is
        deliberately NOT subject to that extra rule here; see the module
        docstring note on independence policy.

        `merge_stats`, when supplied, is filled with aggregate counters only —
        never model text — so the retrieval trace can show whether corroboration
        came from the model's declarations, from merging, or from neither.
        """
        stats = merge_stats if merge_stats is not None else {}

        def bump(name: str, amount: int = 1) -> None:
            stats[name] = stats.get(name, 0) + amount

        # Pass 1: admit evidence-gated sources, in the model's own order, which
        # is what makes "first asserting source wins" deterministic.
        admitted: list[tuple[str, object, str]] = []
        for source in sources:
            key = _normalize_url(source.url)
            if not key or key not in evidence:
                continue
            admitted.append((key, source, key))

        # Pass 2: index claims by statement, recording owner + extra support.
        registry: dict[str, dict] = {}
        for _key, source, canonical in admitted:
            for item in source.claims:
                text = item.claim_text.strip()
                if not text:
                    continue
                bump("model_claims_seen")
                declared = tuple(item.also_supported_by or ())
                bump("explicit_support_declared", len(declared))
                dedup = _claim_dedup_key(text)
                record = registry.get(dedup)
                if record is None:
                    registry[dedup] = {
                        "owner": canonical,
                        "item": item,
                        "declared": declared,
                        "merged": [],
                    }
                    continue
                # Same statement already asserted under another admitted source.
                bump("duplicate_claims")
                if _registrable_domain(record["owner"]) == _registrable_domain(canonical):
                    bump("duplicate_rejected_same_domain")
                    continue
                record["merged"].append(canonical)
                bump("duplicate_merged")

        claims_by_owner: dict[str, list[dict]] = {}
        for record in registry.values():
            claims_by_owner.setdefault(record["owner"], []).append(record)

        # Pass 3: build candidates in model order; each claim rides on its owner.
        candidates: list[SourceCandidate] = []
        for key, source, canonical in admitted:
            claims = tuple(
                self._merged_claim_item(record, evidence, bump)
                for record in claims_by_owner.get(canonical, ())
            )
            candidates.append(
                self._candidate(source, meta_by_url.get(key), note, evidence, claims)
            )

        bump("claims_emitted", len(registry))
        return candidates

    def _merged_claim_item(self, record: dict, evidence: set[str], bump) -> ClaimItem:
        """Build one ClaimItem from its owner claim plus every support path.

        Explicit declarations and merged duplicates are unioned and then passed
        through `_extra_support` together, so neither path can widen the
        admissible source set beyond real retrieval evidence, and neither can
        bypass the anti-fabrication gate.

        The two paths are also counted separately through the same gate, so a
        URL that is both declared and merged is attributed to the path that
        introduced it rather than guessed from the union length.
        """
        item = record["item"]
        declared = tuple(record["declared"])
        merged = tuple(record["merged"])
        kept = self._extra_support(declared + merged, evidence)
        bump("explicit_support_kept", len(self._extra_support(declared, evidence)))
        bump("merged_support_kept", len(self._extra_support(merged, evidence)))
        bump("claims_multi_source" if kept else "claims_single_source")
        return ClaimItem(
            claim_text=item.claim_text.strip(),
            dates=tuple(date.model_dump() for date in item.dates),
            suggested_claim_type=item.suggested_claim_type,
            critical=item.critical,
            also_supported_by_urls=kept,
        )

    def _candidate(
        self,
        source,
        meta: object | None,
        note: str,
        evidence: set[str],
        claims: tuple[ClaimItem, ...],
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
            claims=claims,
            notes=notes,
            retrieved_at="",
        )

    @staticmethod
    def _extra_support(raw_urls, evidence: set[str]) -> tuple[str, ...]:
        """Keep only declared supporting URLs that retrieval actually returned.

        This is the anti-fabrication boundary for multi-source claims: a URL
        the model names that is not in the retrieval evidence is dropped
        (exactly like a model-proposed source), never passed downstream. The
        agent independently re-resolves what survives against the sources it
        built, so a dropped/foreign URL can never become a source id.
        """
        kept: list[str] = []
        seen: set[str] = set()
        for raw in raw_urls or ():
            key = _normalize_url(str(raw))
            if not key or key not in evidence or key in seen:
                continue
            seen.add(key)
            kept.append(key)
        return tuple(kept)

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