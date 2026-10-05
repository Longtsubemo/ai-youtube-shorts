"""Phase 6 comparison: SearXNG vs BeautifulSoup vs self-hosted Firecrawl.

Scope: isolated. The production pipeline does not import this module.

Experimental design
-------------------
The interesting question is not "which search engine is best" — it is **which
extraction tier can actually read a page**. So the discovery set is held fixed:

    SearXNG (one search)  ->  the SAME URL list  ->  BeautifulSoup
                                                  ->  self-hosted Firecrawl

Both extraction methods are handed an identical input set, so any difference in
the outcome table is attributable to the method, not to which pages each one
happened to find. Self-hosted Firecrawl's own `/v2/search` is additionally
recorded as a separate method, because that is what tells us whether it adds
anything on top of SearXNG discovery or only on top of page reading.

Determinism
-----------
Deliberately narrow, and the limits are stated rather than hidden:

- The query list, the per-query result cap, and the fetch order are fixed in
  code, so the *shape* of a run is reproducible.
- The *contents* are not reproducible, and the module says so in its report.
  Upstream engines (DuckDuckGo, Google CSE, Brave, ...) rotate, return
  different orderings, and intermittently return CAPTCHA pages. Measured
  directly: the same query yielded 24 results once and a different top-3 later,
  with `brave` suspended ("too many requests") on one run and `duckduckgo`
  CAPTCHA'd on another. Wikipedia's Markdown also changed size between runs
  (30,447 / 30,693 / 39,579 chars). So: counts are measurements, not fixtures.

No invented metrics
-------------------
Everything reported is counted from an actual observation. There is no
relevance score, no "quality" number, and no proxy metric standing in for one.
If a field could not be measured it is reported as empty/`None`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from experiments.retrieval.firecrawl_selfhost import SelfHostedFirecrawlClient
from experiments.retrieval.html_fetcher import HtmlFetcher
from experiments.retrieval.searxng_client import SearxngClient

# Reused for canonical URL identity in the coverage table; not redefined here.
from agents.research.models import normalize_source_url

SEARXNG = "searxng_search"
BEAUTIFULSOUP = "beautifulsoup_fetch"
FIRECRAWL_SCRAPE = "firecrawl_selfhost_scrape"
FIRECRAWL_SEARCH = "firecrawl_selfhost_search"

METHODS = (SEARXNG, BEAUTIFULSOUP, FIRECRAWL_SCRAPE, FIRECRAWL_SEARCH)

# Factual queries with unambiguous answers, so a missing page is a retrieval
# failure rather than an argument about the fact.
DEFAULT_QUERIES: tuple[str, ...] = (
    "what is the triple point of water",
    "how many bones are in the adult human body",
    "what is the boiling point of ethanol at sea level",
)

DEFAULT_MAX_RESULTS = 5


@dataclass(frozen=True)
class UrlOutcome:
    """What one method did with one URL. All fields are observations."""

    url: str
    domain: str = ""
    http_status: int | None = None
    ok: bool = False
    content_length: int = 0
    title_available: bool = False
    elapsed_ms: int | None = None
    error_kind: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "domain": self.domain,
            "http_status": self.http_status,
            "ok": self.ok,
            "content_length": self.content_length,
            "title_available": self.title_available,
            "elapsed_ms": self.elapsed_ms,
            "error_kind": self.error_kind,
        }


@dataclass(frozen=True)
class MethodReport:
    """Aggregated observations for one method across all queries."""

    method: str
    available: bool = True
    unavailable_reason: str = ""
    search_results_returned: int = 0
    unique_urls: int = 0
    fetch_attempted: int = 0
    fetch_succeeded: int = 0
    fetch_failed: int = 0
    total_content_chars: int = 0
    titles_available: int = 0
    http_status_counts: Mapping[str, int] = field(default_factory=dict)
    error_kinds: Mapping[str, int] = field(default_factory=dict)
    unresponsive_engines: tuple[str, ...] = ()
    outcomes: tuple[UrlOutcome, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "search_results_returned": self.search_results_returned,
            "unique_urls": self.unique_urls,
            "fetch_attempted": self.fetch_attempted,
            "fetch_succeeded": self.fetch_succeeded,
            "fetch_failed": self.fetch_failed,
            "total_content_chars": self.total_content_chars,
            "titles_available": self.titles_available,
            "http_status_counts": dict(self.http_status_counts),
            "error_kinds": dict(self.error_kinds),
            "unresponsive_engines": list(self.unresponsive_engines),
            "outcomes": [o.to_dict() for o in self.outcomes],
        }

    def succeeded_urls(self) -> tuple[str, ...]:
        return tuple(o.url for o in self.outcomes if o.ok)


def _tally_statuses(outcomes: Sequence[UrlOutcome]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for outcome in outcomes:
        key = str(outcome.http_status) if outcome.http_status is not None else "none"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _tally_errors(outcomes: Sequence[UrlOutcome]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for outcome in outcomes:
        if not outcome.ok:
            kind = outcome.error_kind or "unknown"
            counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items()))


@dataclass(frozen=True)
class ComparisonReport:
    """Full result of one comparison run."""

    queries: tuple[str, ...]
    per_query: Mapping[str, Mapping[str, MethodReport]] = field(default_factory=dict)
    overall: Mapping[str, MethodReport] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "queries": list(self.queries),
            "notes": list(self.notes),
            "per_query": {
                query: {name: report.to_dict() for name, report in methods.items()}
                for query, methods in self.per_query.items()
            },
            "overall": {name: report.to_dict() for name, report in self.overall.items()},
        }

    def coverage(self) -> dict[str, tuple[str, ...]]:
        """For every URL any method saw: which methods actually extracted text.

        Aggregated on `normalize_source_url()`, the repository's canonical form,
        because the tiers report URLs differently: SearXNG's canonical form drops
        a trailing slash while Firecrawl echoes whatever `sourceURL` the page
        served. Without this the same page appeared as two rows (measured: 6 such
        pairs in the first real run), which would understate agreement.
        """
        table: dict[str, list[str]] = {}
        for report in self.overall.values():
            if not report.available:
                continue
            for outcome in report.outcomes:
                key = normalize_source_url(outcome.url)
                if not key:
                    continue
                entry = table.setdefault(key, [])
                if outcome.ok and report.method not in entry:
                    entry.append(report.method)
        return {url: tuple(methods) for url, methods in sorted(table.items())}

    def render(self) -> str:
        """Human-readable report. Counts only; no quality judgements."""
        lines: list[str] = []
        lines.append("RETRIEVAL COMPARISON (SearXNG vs BeautifulSoup vs self-hosted Firecrawl)")
        lines.append(f"queries: {len(self.queries)}")
        for query in self.queries:
            lines.append(f"  - {query}")
        lines.append("")

        header = (
            f"{'method':30} {'avail':6} {'results':8} {'uniq':5} {'try':4} "
            f"{'ok':4} {'fail':5} {'chars':8} {'titles':7}"
        )
        lines.append("OVERALL")
        lines.append(header)
        lines.append("-" * len(header))
        for name in METHODS:
            report = self.overall.get(name)
            if report is None:
                lines.append(f"{name:30} {'n/a':6}")
                continue
            if not report.available:
                lines.append(f"{name:30} {'NO':6} {report.unavailable_reason}")
                continue
            lines.append(
                f"{name:30} {'yes':6} {report.search_results_returned:8} "
                f"{report.unique_urls:5} {report.fetch_attempted:4} {report.fetch_succeeded:4} "
                f"{report.fetch_failed:5} {report.total_content_chars:8} {report.titles_available:7}"
            )
            if report.error_kinds:
                lines.append(f"{'':30} error_kinds: {report.error_kinds}")
            if report.http_status_counts:
                lines.append(f"{'':30} http_status: {report.http_status_counts}")
            if report.unresponsive_engines:
                lines.append(f"{'':30} unresponsive_engines: {list(report.unresponsive_engines)}")

        lines.append("")
        lines.append("PER-URL COVERAGE (which method actually extracted text)")
        coverage = self.coverage()
        if not coverage:
            lines.append("  no URLs observed")
        for url, methods in coverage.items():
            lines.append(f"  {'+'.join(methods) if methods else 'NONE':46} {url}")

        if self.notes:
            lines.append("")
            lines.append("NOTES")
            for note in self.notes:
                lines.append(f"  - {note}")
        return "\n".join(lines)


def run_comparison(
    queries: Sequence[str] = DEFAULT_QUERIES,
    *,
    max_results: int = DEFAULT_MAX_RESULTS,
    searxng: SearxngClient | None = None,
    fetcher: HtmlFetcher | None = None,
    firecrawl: SelfHostedFirecrawlClient | None = None,
    max_content_chars: int = 6000,
) -> ComparisonReport:
    """Run the three-way comparison and return measured observations.

    Injectable clients so the same code path can be exercised offline. A method
    whose backend is down is reported as `available=False` with the reason; it
    never silently reports zero results.
    """
    search_client = searxng if searxng is not None else SearxngClient()
    html = fetcher if fetcher is not None else HtmlFetcher(max_content_chars=max_content_chars)
    local_firecrawl = firecrawl if firecrawl is not None else SelfHostedFirecrawlClient()

    notes: list[str] = []
    firecrawl_ready = local_firecrawl.is_ready()
    if not firecrawl_ready:
        notes.append(
            f"self-hosted Firecrawl not reachable at {local_firecrawl.base_url} "
            "(v0/health/readiness); its two methods were skipped"
        )

    per_query: dict[str, dict[str, MethodReport]] = {}
    all_outcomes: dict[str, list[UrlOutcome]] = {name: [] for name in METHODS}
    all_unresponsive: set[str] = set()

    for query in queries:
        query_reports: dict[str, MethodReport] = {}

        # -- 1. discovery via local SearXNG ------------------------------- #
        discovered: list[str] = []
        try:
            results, dead = search_client.search_with_diagnostics(query, max_results=max_results)
            discovered = [r.normalized_url for r in results]
            all_unresponsive.update(dead)
            discovery_outcomes = tuple(
                UrlOutcome(url=url, domain="", http_status=None, ok=False, title_available=False)
                for url in discovered
            )
            all_outcomes[SEARXNG].extend(discovery_outcomes)
            query_reports[SEARXNG] = MethodReport(
                method=SEARXNG,
                search_results_returned=len(results),
                unique_urls=len(discovered),
                unresponsive_engines=dead,
                outcomes=discovery_outcomes,
            )
        except Exception as exc:  # noqa: BLE001 - a down backend is a reportable result
            notes.append(f"SearXNG search failed for {query!r}: {type(exc).__name__}: {exc}")
            query_reports[SEARXNG] = MethodReport(
                method=SEARXNG,
                available=False,
                unavailable_reason=f"{type(exc).__name__}: {exc}",
            )

        # -- 2. BeautifulSoup over the SAME urls -------------------------- #
        if discovered:
            pages = html.fetch_many(discovered)
            bs_outcomes = tuple(
                UrlOutcome(
                    url=page.final_url or page.url,
                    domain=page.domain,
                    http_status=page.http_status,
                    ok=page.ok,
                    content_length=page.content_length,
                    title_available=bool(page.title),
                    elapsed_ms=page.elapsed_ms,
                    error_kind=page.error_kind,
                )
                for page in pages
            )
        else:
            bs_outcomes = ()
        all_outcomes[BEAUTIFULSOUP].extend(bs_outcomes)
        query_reports[BEAUTIFULSOUP] = MethodReport(
            method=BEAUTIFULSOUP,
            search_results_returned=0,
            unique_urls=len({o.url for o in bs_outcomes}),
            fetch_attempted=len(bs_outcomes),
            fetch_succeeded=sum(1 for o in bs_outcomes if o.ok),
            fetch_failed=sum(1 for o in bs_outcomes if not o.ok),
            total_content_chars=sum(o.content_length for o in bs_outcomes),
            titles_available=sum(1 for o in bs_outcomes if o.title_available),
            http_status_counts=_tally_statuses(bs_outcomes),
            error_kinds=_tally_errors(bs_outcomes),
            outcomes=bs_outcomes,
        )

        # -- 3. self-hosted Firecrawl over the SAME urls ------------------ #
        if not firecrawl_ready:
            reason = f"not reachable at {local_firecrawl.base_url}"
            fc_outcomes: tuple[UrlOutcome, ...] = ()
            scrape_report = MethodReport(method=FIRECRAWL_SCRAPE, available=False, unavailable_reason=reason)
        elif discovered:
            outcomes = []
            for url in discovered:
                outcome = local_firecrawl.scrape(url)
                markdown = outcome.markdown
                if len(markdown) > max_content_chars:
                    markdown = markdown[:max_content_chars]
                outcomes.append(
                    UrlOutcome(
                        url=outcome.url or url,
                        domain=outcome.domain,
                        http_status=outcome.http_status,
                        ok=outcome.ok and bool(markdown),
                        content_length=len(markdown),
                        title_available=bool(outcome.title),
                        elapsed_ms=outcome.elapsed_ms,
                        error_kind=outcome.error_kind,
                    )
                )
            fc_outcomes = tuple(outcomes)
            scrape_report = MethodReport(
                method=FIRECRAWL_SCRAPE,
                search_results_returned=0,
                unique_urls=len({o.url for o in fc_outcomes}),
                fetch_attempted=len(fc_outcomes),
                fetch_succeeded=sum(1 for o in fc_outcomes if o.ok),
                fetch_failed=sum(1 for o in fc_outcomes if not o.ok),
                total_content_chars=sum(o.content_length for o in fc_outcomes),
                titles_available=sum(1 for o in fc_outcomes if o.title_available),
                http_status_counts=_tally_statuses(fc_outcomes),
                error_kinds=_tally_errors(fc_outcomes),
                outcomes=fc_outcomes,
            )
        else:
            fc_outcomes = ()
            scrape_report = MethodReport(method=FIRECRAWL_SCRAPE)

        all_outcomes[FIRECRAWL_SCRAPE].extend(fc_outcomes)
        query_reports[FIRECRAWL_SCRAPE] = scrape_report

        # -- 4. self-hosted Firecrawl's own search ------------------------ #
        if not firecrawl_ready:
            query_reports[FIRECRAWL_SEARCH] = MethodReport(
                method=FIRECRAWL_SEARCH,
                available=False,
                unavailable_reason=f"not reachable at {local_firecrawl.base_url}",
            )
        else:
            found = local_firecrawl.search(query, limit=max_results)
            real = tuple(
                UrlOutcome(
                    url=item.url,
                    domain=item.domain,
                    http_status=item.http_status,
                    ok=item.ok,
                    content_length=item.content_length,
                    title_available=bool(item.title),
                    elapsed_ms=item.elapsed_ms,
                    error_kind=item.error_kind,
                )
                for item in found
            )
            all_outcomes[FIRECRAWL_SEARCH].extend(real)
            query_reports[FIRECRAWL_SEARCH] = MethodReport(
                method=FIRECRAWL_SEARCH,
                search_results_returned=len(real),
                unique_urls=len({o.url for o in real}),
                fetch_attempted=len(real),
                fetch_succeeded=sum(1 for o in real if o.ok),
                fetch_failed=sum(1 for o in real if not o.ok),
                total_content_chars=sum(o.content_length for o in real),
                titles_available=sum(1 for o in real if o.title_available),
                http_status_counts=_tally_statuses(real),
                error_kinds=_tally_errors(real),
                outcomes=real,
            )

        per_query[query] = query_reports

    overall: dict[str, MethodReport] = {}
    for name in METHODS:
        outcomes = tuple(all_outcomes.get(name, ()))
        available = any(
            report.get(name) is not None and report[name].available for report in per_query.values()
        )
        reason = ""
        if not available:
            first = next((report.get(name) for report in per_query.values() if report.get(name)), None)
            reason = first.unavailable_reason if first else "never evaluated"
        # SearXNG rows record *discovery*: they have no HTTP status and no error
        # kind, so tallies over them would read as a phantom "unknown" error.
        # Every method's `unique_urls` is derived from its own outcome URLs, so
        # URLs repeated across queries are counted once.
        tallied = outcomes if name != SEARXNG else ()
        overall[name] = MethodReport(
            method=name,
            available=available,
            unavailable_reason=reason,
            search_results_returned=sum(
                (report.get(name).search_results_returned if report.get(name) else 0)
                for report in per_query.values()
            ),
            unique_urls=len({o.url for o in outcomes}),
            fetch_attempted=len(tallied),
            fetch_succeeded=sum(1 for o in tallied if o.ok),
            fetch_failed=sum(1 for o in tallied if not o.ok),
            total_content_chars=sum(o.content_length for o in tallied),
            titles_available=sum(1 for o in tallied if o.title_available),
            http_status_counts=_tally_statuses(tallied),
            error_kinds=_tally_errors(tallied),
            unresponsive_engines=tuple(sorted(all_unresponsive)) if name == SEARXNG else (),
            outcomes=outcomes,
        )

    notes.append(
        "Result ordering and content size are NOT reproducible: upstream engines rotate "
        "and intermittently return CAPTCHA pages (observed: brave 'too many requests', "
        "duckduckgo CAPTCHA). Counts here are measurements, not fixtures."
    )
    if all_unresponsive:
        notes.append(
            "Some engines were unavailable during this run, so the discovery base is narrower "
            f"than the result count suggests: {sorted(all_unresponsive)}"
        )

    return ComparisonReport(
        queries=tuple(queries),
        per_query=per_query,
        overall=overall,
        notes=tuple(notes),
    )


__all__: Sequence[str] = (
    "BEAUTIFULSOUP",
    "DEFAULT_MAX_RESULTS",
    "DEFAULT_QUERIES",
    "ComparisonReport",
    "FIRECRAWL_SCRAPE",
    "FIRECRAWL_SEARCH",
    "METHODS",
    "MethodReport",
    "SEARXNG",
    "UrlOutcome",
    "run_comparison",
)