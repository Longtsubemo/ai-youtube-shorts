"""Isolated retrieval experiment: SearXNG + BeautifulSoup + self-hosted Firecrawl.

NOT production code. Nothing under `agents/`, `providers/`, or `pipeline/`
imports this package, and this package must not be wired into a stage without an
explicit decision (see docs/PROJECT_PROGRESS.md).

The experiment exists to answer one question with measurements instead of
assumption: can a fully self-hosted retrieval stack produce better Stage-2
evidence than the hosted Firecrawl search that currently returns almost entirely
discovery-tier sources?

Intended routing being evaluated (implemented here only as far as Phase 6):

    SearXNG -> search results -> ordinary page fetch -> BeautifulSoup
            -> (only if the page is blocked or too thin)
            -> self-hosted Firecrawl -> normalized RetrievedSource

Contract discipline
-------------------
Every component reuses the production contracts unchanged:
`RetrievedSource` (8 fields, imported from `agents.research.retriever`),
`source_id_for()` and `normalize_source_url()` (imported from
`agents.research.models`). This package defines no competing source type, no
competing id scheme, and no URL canonicalization.
"""

from __future__ import annotations

__all__ = (
    "adapter",
    "comparison",
    "errors",
    "firecrawl_selfhost",
    "html_fetcher",
    "searxng_client",
)