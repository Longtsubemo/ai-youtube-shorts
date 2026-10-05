"""Adapter: SearXNG `SearchResult` -> production `RetrievedSource`.

This is the only place in the experiment that produces contract objects. It is
deliberately shaped so a future integration is a wiring change, not a redesign:

- `RetrievedSource` itself is **not** modified. It is imported from
  `agents.research.retriever` and constructed with its existing 8 fields only.
- URL identity uses the repository's existing logic: `normalize_source_url()`
  for canonicalization and `source_id_for()` for the deterministic id. No new
  hashing or normalization scheme is introduced.
- Unknown metadata stays empty. Nothing is inferred, defaulted, or back-filled.

The honesty rules this module enforces
--------------------------------------
1. **A search result is never dressed up as a fetched document.** SearXNG
   returns a title plus a short snippet; the `content` field of the production
   contract is meant to hold bounded page text. By default this adapter leaves
   `content` empty and only fills it when the caller supplies text it actually
   fetched (see `experiments.retrieval.html_fetcher`). Pass
   `include_snippet=True` to opt into snippet-as-content, which is off by
   default on purpose.
2. **A search result is never treated as authoritative.** `source_type` is
   `"search_result"`, distinct from the `"web_page"` the Firecrawl retriever
   emits, and no tier is assigned — `RetrievedSource` has no tier field and
   tiering in this project is model-judged (docs/RESEARCH_PIPELINE.md §4). This
   adapter therefore cannot promote anything, and it never assigns a
   `SourceCandidate.credibility_tier`.
3. **No unrelated sources are attached to anything.** This adapter maps results
   to sources one-to-one. Deciding which sources corroborates which claim is the
   Stage-2 agent's job, and is unchanged.
4. **A publication date is only passed on if it is really a date.** Engines
   return "3 days ago" / "Jan 2024" in `publishedDate`. Those are preserved on
   `SearchResult.published_date_raw`, but only ISO date/datetime strings are
   copied into `RetrievedSource.publication_date`, because that field is
   contractually ISO and a bare-year or free-text value would fail validation
   downstream anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Sequence

from agents.research.models import source_id_for
from agents.research.retriever import RetrievedSource
from experiments.retrieval.searxng_client import SearchResult, host_for

# Distinct from the Firecrawl retriever's "web_page" so a consumer can always
# tell "this came from a search-engine snippet" from "this is fetched page text".
SEARCH_RESULT_SOURCE_TYPE = "search_result"

# Same marker the production Firecrawl retriever uses, so bounded content reads
# identically wherever it appears.
TRUNCATION_SUFFIX = " …[truncated]"

DEFAULT_MAX_CONTENT_CHARS = 6000


@dataclass(frozen=True)
class MappedSource:
    """A `RetrievedSource` plus the provenance the experiment wants to keep.

    `RetrievedSource` is a fixed 8-field frozen dataclass and this experiment
    does not change it, so the extra traceability lives here instead of being
    squeezed into the contract:

    - `source_id`: `source_id_for(source.url)`, computed once.
    - `original_url`: the URL exactly as SearXNG reported it, before
      canonicalization. Retained because `url` on the contract holds the
      canonical form (see below).
    - `came_from_fetch`: whether `content` came from a real page fetch or from a
      search snippet.
    """

    source: RetrievedSource
    source_id: str
    original_url: str
    came_from_fetch: bool = False

    # Convenience passthroughs so callers do not unwrap by hand.
    @property
    def url(self) -> str:
        return self.source.url

    @property
    def title(self) -> str:
        return self.source.title

    @property
    def domain(self) -> str:
        return self.source.domain

    @property
    def content(self) -> str:
        return self.source.content


def truncate_content(text: str, limit: int) -> str:
    """Bound `text` to `limit` chars, marking truncation (never silently)."""
    body = (text or "").strip()
    if not body:
        return ""
    if len(body) <= limit:
        return body
    return body[:limit].rstrip() + TRUNCATION_SUFFIX


def iso_date_or_empty(raw: str) -> str:
    """Return `raw` only if it is a real ISO date/datetime, else "".

    A bare 4-digit year such as "1986" is deliberately REJECTED here, even though
    the repository tolerates bare years elsewhere (`is_bare_year` in
    `agents/research/models.py`). That tolerance applies to *claim* dates, which
    have an explicit partial-date representation. `Source.publication_date` does
    not: it is validated by `_check_iso`, which rejects a bare year. Emitting one
    here would produce a value that fails validation the moment it is persisted.
    The raw value is not lost — it stays on `SearchResult.published_date_raw`.
    """
    text = (raw or "").strip()
    if not text:
        return ""
    candidate = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        datetime.fromisoformat(candidate)
    except ValueError:
        return ""
    return text


def result_to_source(
    result: SearchResult,
    *,
    content: str = "",
    include_snippet: bool = False,
    retrieved_at: str = "",
    max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
) -> MappedSource | None:
    """Map one search hit to the production contract, or None if unusable.

    `url` holds the **canonical** URL (from `normalize_source_url()`), not the
    raw one. Reason: `source_id_for()` hashes the string exactly as given, so
    handing it the canonical form makes ids stable across the trailing-slash and
    tracking-parameter variants the Stage-2 agent already tries to reconcile
    (`agents/research/agent.py` canonical-URL index). The raw URL is preserved on
    `MappedSource.original_url` so nothing is lost.

    Returns None when the result has no usable absolute http(s) URL, so an
    unusable hit can never become a source.
    """
    canonical = result.normalized_url or ""
    if not canonical:
        return None

    domain = host_for(canonical)

    body = (content or "").strip()
    came_from_fetch = bool(body)
    if not body and include_snippet:
        body = (result.snippet or "").strip()

    source = RetrievedSource(
        url=canonical,
        title=(result.title or "").strip(),
        # `publisher` is the site, matching providers/firecrawl/retriever.py.
        # An `author` from the search result is NOT promoted to publisher: an
        # article author is not the publishing organisation.
        publisher=domain,
        publication_date=iso_date_or_empty(result.published_date_raw),
        source_type=SEARCH_RESULT_SOURCE_TYPE,
        content=truncate_content(body, max_content_chars),
        domain=domain,
        # Left empty by default, exactly as providers/firecrawl/retriever.py does;
        # the caller that knows the real retrieval time supplies it.
        retrieved_at=retrieved_at,
    )

    return MappedSource(
        source=source,
        source_id=source_id_for(source.url),
        original_url=result.url,
        came_from_fetch=came_from_fetch,
    )


def results_to_sources(
    results: Iterable[SearchResult],
    *,
    contents: dict[str, str] | None = None,
    include_snippet: bool = False,
    retrieved_at: str = "",
    max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
) -> tuple[MappedSource, ...]:
    """Map many search hits to contract sources, one-to-one, in rank order.

    `contents` optionally supplies real fetched page text keyed by canonical URL
    (that is what `experiments.retrieval.comparison` does). A URL with no entry
    simply gets empty content rather than inheriting another page's text.
    """
    lookup = contents or {}
    mapped: list[MappedSource] = []
    for result in results:
        item = result_to_source(
            result,
            content=lookup.get(result.normalized_url, ""),
            include_snippet=include_snippet,
            retrieved_at=retrieved_at,
            max_content_chars=max_content_chars,
        )
        if item is not None:
            mapped.append(item)
    return tuple(mapped)


def source_ids(sources: Iterable[MappedSource | RetrievedSource]) -> tuple[str, ...]:
    """Deterministic ids for mapped sources or raw contract objects."""
    out: list[str] = []
    for item in sources:
        url = item.url if isinstance(item, (MappedSource, RetrievedSource)) else ""
        if url:
            out.append(source_id_for(url))
    return tuple(out)


__all__: Sequence[str] = (
    "DEFAULT_MAX_CONTENT_CHARS",
    "MappedSource",
    "SEARCH_RESULT_SOURCE_TYPE",
    "TRUNCATION_SUFFIX",
    "iso_date_or_empty",
    "result_to_source",
    "results_to_sources",
    "source_ids",
    "truncate_content",
)