"""CLI for the isolated retrieval experiment.

Scope: isolated. Not part of the pipeline, not wired to any stage.

Subcommands
-----------
  probe      Check that local SearXNG is up and run one real search.
  fetch      Fetch one or more URLs with BeautifulSoup and show what was read.
  adapter    Map real SearXNG results to the production RetrievedSource contract.
  compare    Run the three-way comparison (SearXNG / BeautifulSoup / Firecrawl).
  firecrawl  Probe the self-hosted Firecrawl stack and the production adapter.

Every subcommand talks to real local services. Nothing here fabricates results:
if a backend is down the command says so and exits non-zero.

Examples
--------
  python -m experiments.retrieval probe
  python -m experiments.retrieval probe --query "what is the boiling point of ethanol" -n 5
  python -m experiments.retrieval fetch https://en.wikipedia.org/wiki/Triple_point
  python -m experiments.retrieval adapter -n 3
  python -m experiments.retrieval compare
  python -m experiments.retrieval firecrawl
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from experiments.retrieval.adapter import results_to_sources
from experiments.retrieval.comparison import DEFAULT_QUERIES, run_comparison
from experiments.retrieval.firecrawl_selfhost import (
    SelfHostedFirecrawlClient,
    production_adapter_probe,
)
from experiments.retrieval.html_fetcher import HtmlFetcher
from experiments.retrieval.searxng_client import SearxngClient, host_for

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_BACKEND_DOWN = 2


def _cmd_probe(args: argparse.Namespace) -> int:
    client = SearxngClient(base_url=args.url, max_results=args.limit)
    print(f"endpoint: {client.base_url}")
    if not client.probe():
        print("FAIL: /healthz did not answer 200.")
        print("Start it with:")
        print("  docker compose -f experiments/retrieval/searxng/docker-compose.yml up -d")
        return EXIT_BACKEND_DOWN

    print("healthz: ok")
    print(f"query: {args.query!r}")
    try:
        results, dead = client.search_with_diagnostics(args.query, max_results=args.limit)
    except Exception as exc:  # noqa: BLE001 - the CLI reports, it does not crash
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return EXIT_BACKEND_DOWN

    print(f"results returned: {len(results)}")
    if dead:
        print(f"unresponsive engines: {list(dead)}")
    print()
    for index, item in enumerate(results, start=1):
        print(f"{index:2}. {item.normalized_url}")
        print(f"    title={item.title!r}")
        print(f"    domain={host_for(item.url)}")
        print(f"    engines={list(item.engines)} score={item.score}")
        if item.snippet:
            print(f"    snippet={item.snippet[:110]!r}")
        print(f"    published_date_raw={item.published_date_raw!r} author={item.author!r}")
    if not results:
        print("no results")
    return EXIT_OK


def _cmd_fetch(args: argparse.Namespace) -> int:
    fetcher = HtmlFetcher(max_content_chars=args.chars)
    for url in args.urls:
        page = fetcher.fetch(url)
        status = page.http_status if page.http_status is not None else "---"
        print(f"{url}")
        print(f"  ok={page.ok} status={status} kind={page.error_kind or '-'} domain={page.domain}")
        if page.ok:
            print(f"  title={page.title!r}")
            print(f"  language={page.language!r} content_length={page.content_length} elapsed_ms={page.elapsed_ms}")
            print(f"  text[:300]={page.text[:300]!r}")
        else:
            print(f"  error={page.error}")
        print()
    return EXIT_OK


def _cmd_adapter(args: argparse.Namespace) -> int:
    client = SearxngClient(base_url=args.url, max_results=args.limit)
    if not client.probe():
        print("FAIL: local SearXNG is not reachable")
        return EXIT_BACKEND_DOWN

    results = client.search(args.query, max_results=args.limit)
    mapped = results_to_sources(results)
    print(f"query: {args.query!r}")
    print(f"search results: {len(results)} -> RetrievedSource objects: {len(mapped)}")
    print()
    for item in mapped:
        source = item.source
        print(f"source_id={item.source_id}")
        print(f"  url            = {source.url}")
        print(f"  original_url   = {item.original_url}")
        print(f"  title          = {source.title!r}")
        print(f"  publisher      = {source.publisher!r}")
        print(f"  domain         = {source.domain!r}")
        print(f"  source_type    = {source.source_type!r}")
        print(f"  publication_date={source.publication_date!r}  retrieved_at={source.retrieved_at!r}")
        print(f"  content        = {len(source.content)} chars  (empty: search snippet is NOT page content)")
        print(f"  came_from_fetch= {item.came_from_fetch}")
    if args.json:
        print()
        print(json.dumps([item.source.url for item in mapped], indent=2))
    return EXIT_OK


def _cmd_compare(args: argparse.Namespace) -> int:
    queries = args.query or list(DEFAULT_QUERIES)
    report = run_comparison(queries, max_results=args.limit)
    print(report.render())
    if args.json:
        print()
        print(json.dumps(report.to_dict(), indent=2))
    return EXIT_OK


def _cmd_firecrawl(args: argparse.Namespace) -> int:
    client = SelfHostedFirecrawlClient(base_url=args.url)
    print(f"endpoint: {client.base_url}")
    if not client.is_ready():
        print("FAIL: /v0/health/readiness did not answer 200.")
        print("Start it with:")
        print("  docker compose -f experiments/retrieval/firecrawl_selfhost/docker-compose.yml up -d")
        return EXIT_BACKEND_DOWN

    print("readiness: ok")
    for url in args.urls:
        outcome = client.scrape(url)
        status = outcome.http_status if outcome.http_status is not None else "---"
        print(f"  scrape {url}")
        print(f"    ok={outcome.ok} status={status} kind={outcome.error_kind or '-'} chars={outcome.content_length}")
        print(f"    title={outcome.title!r}")
        if not outcome.ok:
            print(f"    error={outcome.error}")
        print()

    print("production adapter probe (providers/firecrawl, unmodified):")
    probe = production_adapter_probe(base_url=args.url, limit=args.limit)
    print(json.dumps(probe, indent=2))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.retrieval",
        description="Isolated retrieval experiment (SearXNG + BeautifulSoup + self-hosted Firecrawl).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--url",
        default="http://127.0.0.1:8080",
        help="local SearXNG base URL (default: %(default)s)",
    )

    p_probe = sub.add_parser("probe", parents=[common], help="check SearXNG and run one real search")
    p_probe.add_argument("--query", default=DEFAULT_QUERIES[0])
    p_probe.add_argument("-n", "--limit", type=int, default=5)
    p_probe.set_defaults(func=_cmd_probe)

    p_fetch = sub.add_parser("fetch", help="fetch URLs with BeautifulSoup")
    p_fetch.add_argument("urls", nargs="+")
    p_fetch.add_argument("--chars", type=int, default=6000)
    p_fetch.set_defaults(func=_cmd_fetch)

    p_adapter = sub.add_parser("adapter", parents=[common], help="map real SearXNG results to RetrievedSource")
    p_adapter.add_argument("--query", default=DEFAULT_QUERIES[0])
    p_adapter.add_argument("-n", "--limit", type=int, default=5)
    p_adapter.add_argument("--json", action="store_true")
    p_adapter.set_defaults(func=_cmd_adapter)

    p_compare = sub.add_parser("compare", parents=[common], help="three-way comparison")
    p_compare.add_argument("--query", action="append", help="repeatable; default is the built-in query list")
    p_compare.add_argument("-n", "--limit", type=int, default=5)
    p_compare.add_argument("--json", action="store_true")
    p_compare.set_defaults(func=_cmd_compare)

    p_fc = sub.add_parser("firecrawl", help="probe self-hosted Firecrawl + the production adapter")
    p_fc.add_argument("--url", default="http://127.0.0.1:3002", help="local Firecrawl base URL")
    p_fc.add_argument(
        "urls",
        nargs="*",
        default=["https://en.wikipedia.org/wiki/Triple_point"],
        help="URLs to scrape",
    )
    p_fc.add_argument("-n", "--limit", type=int, default=2)
    p_fc.set_defaults(func=_cmd_firecrawl)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("interrupted", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())