"""Hybrid retrieval provider (the production Stage-2 web retrieval path).

Composes the three providers into one bounded `ResearchRetriever`:

    providers.searxng  -> self-hosted search discovery, several queries
    providers.html     -> cheap httpx + BeautifulSoup page fetch
    providers.firecrawl-> bounded fallback, only for pages the cheap tier failed on

`plan_research_queries` converts a Stage-1 video topic into deterministic,
traceable factual queries; `HybridResearchRetriever` runs the tiers and emits the
existing `RetrievedSource` contract unchanged.
"""

from providers.hybrid.query import (
    DEFAULT_MAX_QUERIES,
    ResearchQuery,
    ResearchQueryPlan,
    plan_research_queries,
)
from providers.hybrid.retriever import (
    CHEAP_TIER_SOURCE_TYPE,
    DEFAULT_MAX_FIRECRAWL_FALLBACKS,
    DEFAULT_MAX_SOURCES,
    FALLBACK_ERROR_KINDS,
    FALLBACK_TIER_SOURCE_TYPE,
    HybridResearchRetriever,
    HybridRetrievalTrace,
)

__all__ = [
    "DEFAULT_MAX_QUERIES",
    "ResearchQuery",
    "ResearchQueryPlan",
    "plan_research_queries",
    "CHEAP_TIER_SOURCE_TYPE",
    "DEFAULT_MAX_FIRECRAWL_FALLBACKS",
    "DEFAULT_MAX_SOURCES",
    "FALLBACK_ERROR_KINDS",
    "FALLBACK_TIER_SOURCE_TYPE",
    "HybridResearchRetriever",
    "HybridRetrievalTrace",
    "TRACE_SCHEMA_VERSION",
    "SOURCE_CAP_REACHED",
    "DISPOSITION_INVALID_URL",
    "DISPOSITION_DUPLICATE",
    "DISPOSITION_DOMAIN_FILTERED",
    "DISPOSITION_CANDIDATE",
]
