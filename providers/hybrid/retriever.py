"""HybridResearchRetriever — the production Stage-2 retrieval path.

Pipeline
--------
    topic title/description
      -> providers.hybrid.query   (deterministic factual queries, traceable)
      -> providers.searxng        (self-hosted discovery, several queries)
      -> dedupe by canonical URL  (agents.research.models.normalize_source_url)
      -> domain allow/deny filter
      -> providers.html           (cheap httpx + BeautifulSoup fetch)
      -> providers.firecrawl      (self-hosted fallback, bounded, only on
                                   genuinely difficult pages)
      -> RetrievedSource          (the existing evidence boundary)

Why this shape
--------------
A single search query over a rhetorical video title retrieves video-essay and
social/UGC pages; that is why the previous Firecrawl-only run retained just one
usable source and could not satisfy Stage 3's `corroboration_min_sources = 2`.
Planning several *factual* queries widens recall, and tiering the fetch means
the expensive browser-backed path is spent only where it actually helps.

Tier honesty
------------
Cheap-tier successes report `source_type="web_page"` and Firecrawl-recovered
pages report `source_type="web_page_rendered"`, so a downstream reader can tell
how a source was actually obtained. This records *provenance of the retrieval*,
not the authority of the source: nothing here assigns credibility. Tiering
remains the claim-extractor's judgement (`SourceCandidate.credibility_tier`),
exactly as before.

A search result that could not be fetched is **dropped, not emitted**. It would
otherwise reach Stage 2 as a source with empty content — which is precisely the
failure this retriever exists to fix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence
from urllib.parse import urlparse

from agents.research.models import normalize_source_url, source_id_for
from agents.research.provider import ResearchProviderError, TopicDescriptor
from agents.research.retriever import RetrievedSource
from providers.firecrawl.client import FirecrawlClient, FirecrawlResult
from providers.firecrawl.config import ENV_API_URL, FirecrawlConfig
from providers.firecrawl.errors import FirecrawlProviderError
from providers.html.fetcher import DEFAULT_MAX_CONTENT_CHARS, FetchedPage, HtmlFetcher
from providers.hybrid.query import DEFAULT_MAX_QUERIES, ResearchQueryPlan, plan_research_queries
from providers.searxng.client import DEFAULT_MAX_RESULTS, SearchResult, client_from_env
from providers.searxng.errors import SearxngError

logger = logging.getLogger(__name__)

DEFAULT_MAX_SOURCES = 8
DEFAULT_MAX_FIRECRAWL_FALLBACKS = 3

# `HtmlFetcher` error kinds that mean "the page is probably real, but a plain
# GET could not read it". These, and only these, justify the expensive tier.
# `invalid_url` is deliberately excluded: Firecrawl cannot fix a malformed URL.
FALLBACK_ERROR_KINDS = frozenset(
    {
        "timeout",
        "connection",
        "http_error",
        "blocked",
        "content_type",
        "thin_content",
        "empty_content",
        "scriptish",
    }
)

CHEAP_TIER_SOURCE_TYPE = "web_page"
FALLBACK_TIER_SOURCE_TYPE = "web_page_rendered"

# Trace format. Version 2 adds per-candidate, per-fetch and per-selection
# identity on top of version 1's counters; every version-1 key is still
# emitted unchanged so an older reader keeps working.
#
# Version 3 adds the provider's source-selection outcome. The bounded audit
# excerpt below is additive on top of that: it appends keys to existing records
# and adds one top-level key, so per docs/DATA_CONTRACTS.md §1 ("bump on
# breaking changes only; additive fields do not bump") the version stays 3.
TRACE_SCHEMA_VERSION = 3

# Longest audit excerpt persisted per emitted source. Audit-only: retrieval and
# the Stage 2 prompt still use full `DEFAULT_MAX_CONTENT_CHARS` content, so this
# cannot change what was fetched or what the model saw. 600 chars keeps all 8
# emitted sources under ~5 KB while still showing the passage that decided
# whether a source was relevant to a claim.
MAX_AUDIT_EXCERPT_CHARS = 600

# Deterministic drop reasons recorded for candidates that never reached a fetch.
SOURCE_CAP_REACHED = "source_cap_reached"
DISPOSITION_INVALID_URL = "invalid_url"
DISPOSITION_DUPLICATE = "duplicate"
DISPOSITION_DOMAIN_FILTERED = "domain_filtered"
DISPOSITION_CANDIDATE = "candidate"


def _host(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except ValueError:  # pragma: no cover - defensive
        return ""


def _domain_allowed(host: str, allow: frozenset[str], deny: frozenset[str]) -> bool:
    if not host:
        return False
    if host in deny or any(host.endswith("." + denied) for denied in deny):
        return False
    if not allow:
        return True
    return host in allow or any(host.endswith("." + allowed) for allowed in allow)


def _interleave_by_query(records: Sequence[dict]) -> list[int]:
    """Order candidate indices round-robin across the queries that produced them.

    `_discover` appends candidates query-major, so consuming them in raw order
    hands the entire source budget to whichever query the planner ranked first: if
    that query returns enough fetchable hits, every later query — including the
    high-precision title variants — is never attempted at all.

    Round-robin fixes that without touching the budget. It preserves each query's
    internal rank order (so ranking semantics within a query are unchanged), it
    keeps the plan's query precedence for the first round (the highest-precision
    query still contributes the first candidate), and it is a pure function of the
    already-recorded candidate list, so a run stays reproducible.

    The guarantee it buys is deliberately modest and precise: before any query
    gets its *second* candidate, every planned query gets its first. Which
    candidates are worth fetching is unchanged — only who competes for the budget.
    """
    buckets: dict[str, list[int]] = {}
    for record in records:
        buckets.setdefault(str(record.get("query") or ""), []).append(int(record["index"]))

    order: list[int] = []
    depth = 0
    while True:
        added = False
        for indices in buckets.values():
            if depth < len(indices):
                order.append(indices[depth])
                added = True
        if not added:
            return order
        depth += 1


def _truncate(text: str, limit: int) -> str:
    body = (text or "").strip()
    if not body:
        return ""
    if len(body) <= limit:
        return body
    return body[:limit].rstrip() + " …[truncated]"


def bounded_audit_excerpt(
    text: str,
    limit: int = MAX_AUDIT_EXCERPT_CHARS,
) -> tuple[str, bool]:
    """Return a deterministic bounded prefix of retrieved content.

    Audit-only. The retrieval path still consumes full content
    (`DEFAULT_MAX_CONTENT_CHARS`); this exists purely so a finished run can be
    re-read without re-fetching anything.

    The returned flag is True when `text` was longer than `limit`, so a reader
    never has to infer truncation from length alone. The excerpt is the plain
    prefix with no synthetic marker appended, which keeps it quotable as the
    real page text; the boolean carries the truncation fact instead.
    """
    body = (text or "").strip()
    cap = max(0, limit)
    if len(body) <= cap:
        return body, False
    return body[:cap].rstrip(), True


class _Searcher(Protocol):
    def search(self, query: str, *, max_results: int | None = None) -> tuple[SearchResult, ...]: ...


class _Fallback(Protocol):
    def scrape(self, url: str, *, only_main_content: bool = True) -> FirecrawlResult: ...


@dataclass
class HybridRetrievalTrace:
    """What actually happened during one `retrieve()` call.

    This is the audit record for the query planner and the fetch tiers. It is
    exposed separately from `RetrievedSource` because the evidence contract must
    not grow fields, and it is empty-by-default so nothing depends on it.
    """

    plan: ResearchQueryPlan | None = None
    per_query_hits: dict[str, int] = field(default_factory=dict)
    candidates: int = 0
    duplicates_dropped: int = 0
    domain_filtered: int = 0
    cheap_ok: int = 0
    cheap_failed: dict[str, str] = field(default_factory=dict)
    fallback_attempted: list[str] = field(default_factory=list)
    fallback_ok: list[str] = field(default_factory=list)
    fallback_skipped: dict[str, str] = field(default_factory=dict)
    emitted: int = 0
    raw_hits: list[dict] = field(default_factory=list)
    candidate_records: list[dict] = field(default_factory=list)
    emitted_records: list[dict] = field(default_factory=list)
    not_attempted: list[dict] = field(default_factory=list)
    selection: dict | None = None
    emitted_summary: list[dict] = field(default_factory=list)
    model_discarded: list[dict] = field(default_factory=list)

    # -- identity recording ------------------------------------------------ #
    #
    # Bookkeeping only. Every method here appends to or annotates a record and
    # returns nothing the caller branches on, so no recording path can change
    # which candidate is fetched next, in what order, or how many are emitted.

    def _make_identity(self, url: str, canonical_url: str, index: int | None = None) -> dict:
        canon = canonical_url or url or ""
        identity = {
            "id": canon,
            "canonical_url": canon,
            "url": url or canon,
            "index": index,
        }
        return identity

    def record_raw_hit(
        self,
        *,
        query: str,
        rank: int,
        url: str,
        canonical_url: str,
        disposition: str,
        source_tier: str | None = None,
        was_deduplicated: bool = False,
    ) -> None:
        """Record one SearXNG hit and what happened to it before fetching."""
        self.raw_hits.append(
            {
                "query": query,
                "rank": rank,
                "url": url,
                "canonical_url": canonical_url,
                "disposition": disposition,
                "discovery_provider": "searxng",
                "source_tier": source_tier,
                "was_deduplicated": was_deduplicated,
            }
        )

    def begin_candidate(
        self,
        *,
        index: int,
        url: str,
        canonical_url: str,
        title: str,
        query: str,
        rank: int,
        engines: Sequence[str],
        score: float | None,
        source_tier: str | None = None,
    ) -> None:
        """Open the record for a candidate that survived dedupe and filtering."""
        identity = self._make_identity(url, canonical_url, index)
        self.candidate_records.append(
            {
                "index": index,
                "url": url,
                "canonical_url": canonical_url,
                "identity": identity,
                "title": title,
                "query": query,
                "rank": rank,
                "engines": list(engines),
                "score": score,
                "source_tier": source_tier,
                "discovery_provider": "searxng",
                "cheap_attempted": False,
                "cheap_ok": False,
                "cheap_error_kind": None,
                "fallback_attempted": False,
                "fallback_ok": False,
                "fallback_skip_reason": None,
                "source_type": None,
                "retrieval_path": None,
                "obtained_via": None,
                "content_chars": 0,
                "not_attempted_reason": None,
                "deduplicated": False,
            }
        )

    def _record(self, index: int | None) -> dict | None:
        if index is None or not (0 <= index < len(self.candidate_records)):
            return None
        return self.candidate_records[index]

    def record_cheap_success(self, index: int | None, *, url: str, content_chars: int, source_tier: str | None = None) -> None:
        record = self._record(index)
        if record is None:
            return
        record["cheap_attempted"] = True
        record["cheap_ok"] = True
        record["source_type"] = CHEAP_TIER_SOURCE_TYPE
        record["retrieval_path"] = "cheap_html"
        record["obtained_via"] = "cheap"
        record["content_chars"] = content_chars
        if source_tier is not None:
            record["source_tier"] = source_tier

    def record_cheap_failure(self, index: int | None, *, url: str, error_kind: str, source_tier: str | None = None) -> None:
        record = self._record(index)
        if record is None:
            return
        record["cheap_attempted"] = True
        record["cheap_ok"] = False
        record["cheap_error_kind"] = error_kind
        if source_tier is not None:
            record["source_tier"] = source_tier

    def record_fallback_attempted(self, index: int | None) -> None:
        record = self._record(index)
        if record is not None:
            record["fallback_attempted"] = True

    def record_fallback_success(self, index: int | None, *, content_chars: int, source_tier: str | None = None) -> None:
        record = self._record(index)
        if record is None:
            return
        record["fallback_attempted"] = True
        record["fallback_ok"] = True
        record["source_type"] = FALLBACK_TIER_SOURCE_TYPE
        record["retrieval_path"] = "firecrawl"
        record["obtained_via"] = "firecrawl"
        record["content_chars"] = content_chars
        if source_tier is not None:
            record["source_tier"] = source_tier

    def record_fallback_skip(self, index: int | None, *, reason: str, source_tier: str | None = None) -> None:
        record = self._record(index)
        if record is not None:
            record["fallback_skip_reason"] = reason
            if source_tier is not None:
                record["source_tier"] = source_tier

    def record_not_attempted(self, index: int, *, reason: str, source_tier: str | None = None) -> None:
        """Mark a candidate that was never fetched because the cap was already met."""
        record = self._record(index)
        if record is None:
            return
        record["not_attempted_reason"] = reason
        if source_tier is not None:
            record["source_tier"] = source_tier
        self.not_attempted.append(
            {
                "index": index,
                "url": record["url"],
                "identity": record.get("identity", self._make_identity(record["url"], record.get("canonical_url", record["url"]), index)),
                "reason": reason,
                "source_tier": record.get("source_tier", source_tier),
            }
        )

    def record_emitted(self, index: int | None, source: RetrievedSource) -> None:
        """Record an emitted source and how its content was obtained.

        Also persists a bounded excerpt of the content this source actually
        carried, so the model-selection boundary can be audited later without
        re-fetching. Audit-only: nothing here is read back by `retrieve()`.
        """
        obtained_via = (
            "firecrawl" if source.source_type == FALLBACK_TIER_SOURCE_TYPE else "cheap"
        )
        retrieval_path = "firecrawl" if obtained_via == "firecrawl" else "cheap_html"
        record = self._record(index)
        if record is not None:
            record["obtained_via"] = obtained_via
            record["retrieval_path"] = retrieval_path
            record["source_type"] = source.source_type
            if not record["content_chars"]:
                record["content_chars"] = len(source.content or "")
            identity = record.get("identity", self._make_identity(source.url, record.get("canonical_url", source.url), index))
        else:
            identity = self._make_identity(source.url, source.url, index)
        excerpt, truncated = bounded_audit_excerpt(source.content)
        # Minted exactly the way the research agent mints it, so an audit can
        # join a trace record to a ResearchPackage Source by id and not only by
        # URL.
        source_id = source_id_for(source.url)
        self.emitted_records.append(
            {
                "order": len(self.emitted_records),
                "index": index,
                "url": source.url,
                "identity": identity,
                "source_id": source_id,
                "title": source.title,
                "source_type": source.source_type,
                "retrieval_path": retrieval_path,
                "obtained_via": obtained_via,
                "content_chars": len(source.content or ""),
                "audit_excerpt": excerpt,
                "audit_excerpt_chars": len(excerpt),
                "audit_excerpt_truncated": truncated,
                "query": record.get("query") if record is not None else None,
                "rank": record.get("rank") if record is not None else None,
                "source_tier": record.get("source_tier") if record is not None else None,
            }
        )
        self.emitted_summary.append(
            {
                "stable_source_identity": identity.get("id"),
                "identity": identity,
                "url": source.url,
                # Identity only - the excerpt text lives once, in
                # `emitted_records`, so the artifact never stores it twice.
                "source_id": source_id,
                "tier": record.get("source_tier") if record is not None else None,
                "retrieval_path": retrieval_path,
                "query": record.get("query") if record is not None else None,
                "rank": record.get("rank") if record is not None else None,
            }
        )

    def record_selection(self, payload: dict) -> None:
        """Attach the provider's source-selection outcome.

        Called by the research provider once the model has chosen sources, so
        the trace can name which fetched sources were offered, kept, or thrown
        away. Storing a plain dict keeps the retrieval package free of any
        dependency on the provider's candidate types.
        """
        self.selection = dict(payload)

    def record_model_discarded(self, *, url: str, canonical_url: str | None = None, tier: str | None = None, reason: str | None = None) -> None:
        canon = canonical_url or url or ""
        self.model_discarded.append(
            {
                "url": url,
                "canonical_url": canon,
                "identity": self._make_identity(url, canon),
                "tier": tier,
                "reason": reason or "discarded_without_reason",
            }
        )

    @staticmethod
    def _candidate_view(record: dict) -> dict:
        """Derive a candidate's final status from its recorded outcomes.

        Derived rather than set per branch, so the status cannot drift from the
        outcome fields it summarises.
        """
        view = dict(record)
        if record.get("not_attempted_reason"):
            status = "not_attempted"
        elif record.get("cheap_ok"):
            status = "emitted"
        elif record.get("fallback_ok"):
            status = "emitted"
        elif record.get("fallback_attempted"):
            status = "fallback_failed"
        elif record.get("cheap_attempted"):
            status = "dropped"
        else:
            status = "not_attempted"
        view["status"] = status
        view["reason"] = (
            record.get("not_attempted_reason")
            or record.get("cheap_error_kind")
            or record.get("fallback_skip_reason")
        )
        return view

    def as_lines(self) -> list[str]:
        """Human-readable, log/artifact friendly summary."""
        lines = [f"query_plan: {self.plan.as_trace() if self.plan else 'none'}"]
        lines += [f"query_hits[{query!r}]={hits}" for query, hits in self.per_query_hits.items()]
        lines.append(
            f"candidates={self.candidates} duplicates_dropped={self.duplicates_dropped} "
            f"domain_filtered={self.domain_filtered}"
        )
        lines.append(f"cheap_ok={self.cheap_ok} cheap_failed={len(self.cheap_failed)}")
        lines += [f"cheap_failure[{url}]={kind}" for url, kind in sorted(self.cheap_failed.items())]
        lines.append(
            f"fallback_attempted={len(self.fallback_attempted)} fallback_ok={len(self.fallback_ok)}"
        )
        lines += [f"fallback_ok[{url}]" for url in sorted(self.fallback_ok)]
        lines += [f"fallback_skipped[{url}]={reason}" for url, reason in sorted(self.fallback_skipped.items())]
        lines.append(f"emitted={self.emitted}")
        lines.append(
            f"identities raw_hits={len(self.raw_hits)} candidates={len(self.candidate_records)} "
            f"emitted_records={len(self.emitted_records)} not_attempted={len(self.not_attempted)}"
        )
        lines.append(f"selection={'recorded' if self.selection else 'absent'}")
        return lines

    def to_dict(self) -> dict:
        """JSON-serialisable snapshot for artifact persistence.

        Persistence only. This reads fields `retrieve()` already recorded and
        changes no retrieval behaviour, so the audit trail can be stored without
        risking the evidence it describes. The plan is expanded structurally as
        well as via `as_trace()` so consumers never have to parse the summary
        line back into queries.

        Version 2 keeps every version-1 key unchanged and adds identity: each
        raw hit with its disposition, each candidate with its fetch and fallback
        outcome, each emitted source with how its content was obtained, each
        candidate that was never attempted with the reason, and the provider's
        source-selection outcome. That closes the version-1 gap where a fetched
        or dropped page was only visible as a count.

        `selection` is `None` whenever no provider recorded one, so a reader can
        tell "nothing selected" apart from "selection never ran".

        Additive on top of version 3: each `emitted_records` entry also carries
        `source_id`, `audit_excerpt`, `audit_excerpt_chars` and
        `audit_excerpt_truncated`, and `emitted_summary` carries `source_id`, so
        the model-selection boundary can be audited without re-fetching. The
        excerpt is capped at `MAX_AUDIT_EXCERPT_CHARS`, also reported here so the
        cap is visible from the artifact alone. Retrieval still feeds full
        content to the model, so none of this can change what was fetched.
        """
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "audit_excerpt_max_chars": MAX_AUDIT_EXCERPT_CHARS,
            "plan": (
                None
                if self.plan is None
                else {
                    "topic_id": self.plan.topic_id,
                    "queries": [
                        {"text": query.text, "rule": query.rule}
                        for query in self.plan.queries
                    ],
                    "as_trace": self.plan.as_trace(),
                }
            ),
            "per_query_hits": dict(self.per_query_hits),
            "candidates": self.candidates,
            "duplicates_dropped": self.duplicates_dropped,
            "domain_filtered": self.domain_filtered,
            "cheap_ok": self.cheap_ok,
            "cheap_failed": dict(self.cheap_failed),
            "fallback_attempted": list(self.fallback_attempted),
            "fallback_ok": list(self.fallback_ok),
            "fallback_skipped": dict(self.fallback_skipped),
            "emitted": self.emitted,
            "raw_hits": [dict(hit) for hit in self.raw_hits],
            "candidate_records": [self._candidate_view(record) for record in self.candidate_records],
            "emitted_records": [dict(record) for record in self.emitted_records],
            "emitted_summary": [dict(record) for record in self.emitted_summary],
            "model_discarded": [dict(record) for record in self.model_discarded],
            "not_attempted": [dict(record) for record in self.not_attempted],
            "selection": None if self.selection is None else dict(self.selection),
        }


class HybridResearchRetriever:
    """Retrieve real, bounded, independently-useful sources for a topic."""

    retriever_name = "hybrid"

    def __init__(
        self,
        *,
        searcher: _Searcher | None = None,
        fetcher: HtmlFetcher | None = None,
        fallback: _Fallback | None = None,
        firecrawl_config: FirecrawlConfig | None = None,
        allow_domains: Iterable[str] = (),
        deny_domains: Iterable[str] = (),
        max_sources: int = DEFAULT_MAX_SOURCES,
        max_queries: int = DEFAULT_MAX_QUERIES,
        search_limit_per_query: int = DEFAULT_MAX_RESULTS,
        max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
        max_firecrawl_fallbacks: int = DEFAULT_MAX_FIRECRAWL_FALLBACKS,
        enable_fallback: bool = True,
    ) -> None:
        # Built from `SEARXNG_*` env so a run can point discovery at a
        # non-default instance or restrict categories. Constructing the client
        # performs no I/O.
        self._searcher = searcher if searcher is not None else client_from_env()
        self._fetcher = fetcher if fetcher is not None else HtmlFetcher()
        self._firecrawl_config = firecrawl_config
        self._fallback = fallback
        # Resolved lazily: a run that never needs the expensive tier must not
        # require the Firecrawl SDK to be installed at all.
        self._fallback_enabled = enable_fallback and max_firecrawl_fallbacks > 0
        self._allow = frozenset(domain.strip().lower() for domain in allow_domains if domain.strip())
        self._deny = frozenset(domain.strip().lower() for domain in deny_domains if domain.strip())
        self._max_sources = max(0, max_sources)
        self._max_queries = max(1, max_queries)
        self._search_limit = max(1, search_limit_per_query)
        self._max_content_chars = max(0, max_content_chars)
        self._max_fallbacks = max(0, max_firecrawl_fallbacks)
        self._last_trace = HybridRetrievalTrace()

    @property
    def last_trace(self) -> HybridRetrievalTrace:
        """Audit record for the most recent `retrieve()` call."""
        return self._last_trace

    def retrieve(self, topic: TopicDescriptor) -> list[RetrievedSource]:
        plan = plan_research_queries(topic, max_queries=self._max_queries)
        trace = HybridRetrievalTrace(plan=plan)
        self._last_trace = trace

        if not plan.queries:
            # No title and no description means no honest query exists. Returning
            # nothing is correct; inventing one would fabricate a search.
            logger.warning("no research query derivable for topic %s", topic.topic_id)
            return []

        candidates = self._discover(plan, trace)
        trace.candidates = len(candidates)

        # Round-robin over the originating queries so no single query can spend the
        # whole `max_sources` budget before the others are tried at all.
        order = _interleave_by_query(trace.candidate_records)

        sources: list[RetrievedSource] = []
        for position, index in enumerate(order):
            if len(sources) >= self._max_sources:
                # The cap is why the remaining candidates are unknown today.
                # Name each of them, then stop exactly as before.
                for pending in order[position:]:
                    trace.record_not_attempted(pending, reason=SOURCE_CAP_REACHED)
                break
            source = self._retrieve_one(candidates[index], trace, index)
            if source is not None:
                sources.append(source)
                trace.record_emitted(index, source)

        trace.emitted = len(sources)
        for line in trace.as_lines():
            logger.debug("hybrid retrieval %s", line)
        return sources

    # -- discovery ---------------------------------------------------------- #

    def _discover(self, plan: ResearchQueryPlan, trace: HybridRetrievalTrace) -> list[SearchResult]:
        """Search every planned query, merge, and deduplicate by canonical URL."""
        merged: list[SearchResult] = []
        seen: set[str] = set()

        for query in plan.texts:
            try:
                hits = tuple(self._searcher.search(query, max_results=self._search_limit))
            except SearxngError as exc:
                # One failed query must not discard the others; a total outage
                # surfaces as an empty (not fabricated) candidate list.
                logger.warning("SearXNG query %r failed: %s", query, exc)
                trace.per_query_hits[query] = 0
                continue
            trace.per_query_hits[query] = len(hits)
            for rank, hit in enumerate(hits, start=1):
                canonical = normalize_source_url(hit.url)
                if not canonical:
                    trace.record_raw_hit(
                        query=query,
                        rank=rank,
                        url=hit.url or "",
                        canonical_url="",
                        disposition=DISPOSITION_INVALID_URL,
                    )
                    continue
                if not _domain_allowed(_host(canonical), self._allow, self._deny):
                    trace.domain_filtered += 1
                    trace.record_raw_hit(
                        query=query,
                        rank=rank,
                        url=hit.url,
                        canonical_url=canonical,
                        disposition=DISPOSITION_DOMAIN_FILTERED,
                    )
                    continue
                if canonical in seen:
                    trace.duplicates_dropped += 1
                    trace.record_raw_hit(
                        query=query,
                        rank=rank,
                        url=hit.url,
                        canonical_url=canonical,
                        disposition=DISPOSITION_DUPLICATE,
                        was_deduplicated=True,
                    )
                    continue
                seen.add(canonical)
                trace.record_raw_hit(
                    query=query,
                    rank=rank,
                    url=hit.url,
                    canonical_url=canonical,
                    disposition=DISPOSITION_CANDIDATE,
                )
                # Tier assignment is not changed by this instrumentation - keep existing behavior implicit
                trace.begin_candidate(
                    index=len(merged),
                    url=hit.url,
                    canonical_url=canonical,
                    title=hit.title or "",
                    query=query,
                    rank=rank,
                    engines=tuple(hit.engines or ()),
                    score=hit.score,
                )
                merged.append(hit)
        return merged

    # -- fetch + fallback --------------------------------------------------- #

    def _retrieve_one(
        self,
        candidate: SearchResult,
        trace: HybridRetrievalTrace,
        index: int | None = None,
    ) -> RetrievedSource | None:
        url = (candidate.url or "").strip()
        if not url:
            return None

        page = self._fetcher.fetch(url)
        if page.ok:
            trace.cheap_ok += 1
            trace.record_cheap_success(index, url=url, content_chars=len(page.text or ""))
            return self._to_source(url, page.title or candidate.title, page.text, CHEAP_TIER_SOURCE_TYPE)

        error_kind = page.error_kind or "unknown"
        trace.cheap_failed[url] = error_kind
        trace.record_cheap_failure(index, url=url, error_kind=error_kind)
        return self._try_fallback(url, candidate, page, trace, index)

    def _try_fallback(
        self,
        url: str,
        candidate: SearchResult,
        page: FetchedPage,
        trace: HybridRetrievalTrace,
        index: int | None = None,
    ) -> RetrievedSource | None:
        """Escalate one genuinely difficult page to self-hosted Firecrawl."""
        if page.error_kind and page.error_kind not in FALLBACK_ERROR_KINDS:
            reason = f"not_a_fallback_kind:{page.error_kind}"
            trace.fallback_skipped[url] = reason
            trace.record_fallback_skip(index, reason=reason)
            return None
        if not self._fallback_enabled:
            trace.fallback_skipped[url] = "fallback_disabled"
            trace.record_fallback_skip(index, reason="fallback_disabled")
            return None
        if len(trace.fallback_attempted) >= self._max_fallbacks:
            trace.fallback_skipped[url] = "fallback_budget_exhausted"
            trace.record_fallback_skip(index, reason="fallback_budget_exhausted")
            return None

        client = self._resolve_fallback()
        if client is None:
            trace.fallback_skipped[url] = "fallback_unavailable"
            trace.record_fallback_skip(index, reason="fallback_unavailable")
            return None

        trace.fallback_attempted.append(url)
        trace.record_fallback_attempted(index)
        try:
            result = client.scrape(url)
        except FirecrawlProviderError as exc:
            # A failed escalation is an honest miss, not an error for the run:
            # one unreadable page must not abort retrieval of the others.
            logger.warning("self-hosted Firecrawl fallback failed for %s: %s", url, exc)
            reason = f"fallback_error:{type(exc).__name__}"
            trace.fallback_skipped[url] = reason
            trace.record_fallback_skip(index, reason=reason)
            return None

        content = result.markdown
        if not (content or "").strip():
            trace.fallback_skipped[url] = "fallback_returned_no_content"
            trace.record_fallback_skip(index, reason="fallback_returned_no_content")
            return None

        trace.fallback_ok.append(url)
        trace.record_fallback_success(index, content_chars=len(content))
        return self._to_source(
            url,
            (result.title or page.title or candidate.title).strip(),
            content,
            FALLBACK_TIER_SOURCE_TYPE,
        )

    def _resolve_fallback(self) -> _Fallback | None:
        if self._fallback is not None:
            return self._fallback
        config = self._firecrawl_config or FirecrawlConfig()
        if not config.api_url:
            # An empty FIRECRAWL_API_URL makes the Firecrawl SDK fall back to
            # its hosted endpoint. The expensive tier here is meant to be the
            # self-hosted instance, so refuse instead of silently sending a run
            # to the cloud. Missing fallback is an honest miss: the cheap tier
            # already emitted what it could actually read.
            logger.warning(
                "self-hosted Firecrawl fallback disabled: %s is not set, and the "
                "SDK default would be the hosted endpoint",
                ENV_API_URL,
            )
            return None
        try:
            self._fallback = FirecrawlClient(config)
        except FirecrawlProviderError as exc:  # pragma: no cover - config is validated in-process
            logger.warning("self-hosted Firecrawl unavailable: %s", exc)
            return None
        return self._fallback

    def _to_source(self, url: str, title: str, content: str, source_type: str) -> RetrievedSource:
        content = _truncate(content, self._max_content_chars)
        if not content:
            # Belt and braces: never emit a source we could not actually read.
            raise RuntimeError("_to_source called with empty content")  # pragma: no cover
        host = _host(url)
        return RetrievedSource(
            url=url,
            title=(title or "").strip(),
            publisher=host,
            source_type=source_type,
            content=content,
            domain=host,
            retrieved_at="",
        )


__all__ = (
    "CHEAP_TIER_SOURCE_TYPE",
    "DEFAULT_MAX_FIRECRAWL_FALLBACKS",
    "DEFAULT_MAX_SOURCES",
    "FALLBACK_ERROR_KINDS",
    "FALLBACK_TIER_SOURCE_TYPE",
    "MAX_AUDIT_EXCERPT_CHARS",
    "SOURCE_CAP_REACHED",
    "TRACE_SCHEMA_VERSION",
    "HybridResearchRetriever",
    "HybridRetrievalTrace",
    "bounded_audit_excerpt",
)
