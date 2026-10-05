"""REAL local-stack smoke test for the production `hybrid` retriever.

Gated by `RETRIEVAL_NETWORK_TESTS=1`, so a normal `pytest -q` run stays
network-free and needs no Docker:

    RETRIEVAL_NETWORK_TESTS=1 python -m pytest tests/test_hybrid_network_smoke.py -q

This exercises the composed production path end to end against the *self-hosted*
stacks only — never a cloud API:

    query planning -> SearXNG -> BeautifulSoup fetch -> (bounded Firecrawl fallback)

It asserts only structural properties: real http(s) URLs, a non-empty plan, at
most the configured number of sources, every emitted source carrying non-empty
content, and the two tiers being labelled distinctly. It asserts nothing about
result counts, ordering, or text, because upstream engines rotate and
intermittently CAPTCHA — pinning those would make this fail for reasons
unrelated to the code under test.

If a stack is down the test SKIPS with the reason. A local service being
unavailable is an environment fact, not a code defect, and must never be
reported as a passing "0 results".

Requires:
  - SearXNG   : docker compose -f experiments/retrieval/searxng/docker-compose.yml up -d
  - Firecrawl : docker compose -f experiments/retrieval/firecrawl_selfhost/docker-compose.yml up -d
"""

from __future__ import annotations

import os

import pytest

from agents.research.provider import TopicDescriptor
from providers.firecrawl.config import FirecrawlConfig
from providers.hybrid.retriever import (
    CHEAP_TIER_SOURCE_TYPE,
    FALLBACK_TIER_SOURCE_TYPE,
    HybridResearchRetriever,
)
from providers.html.fetcher import HtmlFetcher
from providers.searxng.client import DEFAULT_BASE_URL, SearxngClient

GATE = os.environ.get("RETRIEVAL_NETWORK_TESTS") == "1"
NETWORK = pytest.mark.skipif(
    not GATE, reason="set RETRIEVAL_NETWORK_TESTS=1 to run real local-stack calls"
)

SEARXNG_URL = os.environ.get("RETRIEVAL_SEARXNG_URL", DEFAULT_BASE_URL)
FIRECRAWL_URL = os.environ.get("RETRIEVAL_FIRECRAWL_URL", "http://127.0.0.1:3002")

TOPIC = TopicDescriptor(
    topic_id="t_network_smoke",
    title="Why Is the Sky Blue?",
    description="Rayleigh scattering makes shorter wavelengths of visible light scatter "
    "more strongly than longer ones in the atmosphere.",
    pillar="science",
    topic_type="scientific_concept",
)


def _retriever(**kwargs) -> HybridResearchRetriever:
    kwargs.setdefault("max_sources", 4)
    kwargs.setdefault("max_firecrawl_fallbacks", 1)
    kwargs.setdefault("firecrawl_config", FirecrawlConfig(api_url=FIRECRAWL_URL, api_key=""))
    return HybridResearchRetriever(
        searcher=SearxngClient(base_url=SEARXNG_URL),
        fetcher=HtmlFetcher(),
        **kwargs,
    )


def _searxng_healthy() -> bool:
    """True when the local container itself answers /healthz."""
    try:
        return SearxngClient(base_url=SEARXNG_URL).probe()
    except Exception:  # noqa: BLE001 - availability probe only
        return False


def _unresponsive_engines() -> str:
    """Upstream engines that refused this container's most recent search."""
    try:
        engines = SearxngClient(base_url=SEARXNG_URL, max_retries=0).search_with_diagnostics(
            "Rayleigh scattering", max_results=1
        )[1]
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        return f"diagnostics unavailable ({type(exc).__name__})"
    if not engines:
        return "none reported"
    return "; ".join(engines)


@NETWORK
def test_hybrid_retrieval_against_real_local_stacks():
    if not _searxng_healthy():
        pytest.skip(f"local SearXNG not healthy at {SEARXNG_URL}")

    retriever = _retriever()
    sources = retriever.retrieve(TOPIC)
    trace = retriever.last_trace

    # The plan must be real and traceable, whatever the search returns.
    assert trace.plan is not None
    assert trace.plan.queries
    assert trace.plan.texts[0]  # a factual query leads

    if not sources:
        # SearXNG being healthy does NOT mean upstream engines answered: they
        # rate-limit and CAPTCHA routinely. A zero-source run is an environment
        # fact, and reporting it as a passing assertion would be dishonest, so
        # surface the reason and skip rather than assert on nothing.
        unresponsive = _unresponsive_engines()
        pytest.skip(
            "SearXNG is healthy but returned no usable hits for any planned query; "
            f"unresponsive upstream engines: {unresponsive}"
        )

    # Bounded, and no more than we asked for.
    assert len(sources) <= 4

    for source in sources:
        assert source.url.startswith(("http://", "https://"))
        # The core fix: no source may reach Stage 2 with empty content.
        assert source.content.strip(), f"empty content leaked for {source.url}"
        assert source.domain
        assert source.source_type in (CHEAP_TIER_SOURCE_TYPE, FALLBACK_TIER_SOURCE_TYPE)

    # Real retrieval must mean real *volume*: one hit would satisfy the
    # assertions above while proving the local stack cannot actually feed Stage 2,
    # which is the failure mode this whole session was spent fixing. Stage 3
    # corroboration needs several independent sources, so a run that cannot
    # produce several must not be reported as a working stack.
    assert len(sources) >= 2, (
        "the local stack produced only one usable source; that cannot support "
        f"corroboration. trace: {' | '.join(trace.as_lines())}"
    )
    domains = {source.domain for source in sources}
    assert len(domains) >= 2, f"all {len(sources)} sources came from one domain: {sorted(domains)}"

    print("\n[hybrid network smoke]")
    for line in trace.as_lines():
        print(" ", line)
    for source in sources:
        print(f"  source {source.source_type} {source.domain} chars={len(source.content)}")


@NETWORK
def test_firecrawl_fallback_recovers_a_blocked_page():
    """Escalate a page the cheap tier cannot read; assert it either works or is dropped."""
    from providers.firecrawl.client import FirecrawlClient

    # example.com is reliably blocked/thin for a plain GET, which is exactly the
    # escalation case. The assertion is deliberately weak: the page may become
    # readable from here, and then a fallback is not required.
    url = "https://example.com/"
    page = HtmlFetcher().fetch(url)
    if page.ok:
        pytest.skip(f"cheap fetch unexpectedly succeeded for {url}; no fallback case to test")

    try:
        result = FirecrawlClient(FirecrawlConfig(api_url=FIRECRAWL_URL, api_key="")).scrape(url)
    except Exception as exc:  # noqa: BLE001 - self-hosted stack may be down
        pytest.skip(f"self-hosted Firecrawl not usable at {FIRECRAWL_URL}: {type(exc).__name__}")

    assert result.markdown.strip()
    print(f"\n[fallback probe] {url} cheap_error_kind={page.error_kind!r} fallback_chars={len(result.markdown)}")
