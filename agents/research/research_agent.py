"""Manual runner for the research stage.

Usage:
    python -m agents.research.research_agent
    python -m agents.research.research_agent --selection data/<run_id>/research/topic_selection.json
    python -m agents.research.research_agent --data data/mock/research_evidence.json
    python -m agents.research.research_agent --provider gemini
    python -m agents.research.research_agent --provider gemini --retriever firecrawl

Loads a TopicSelection artifact, runs the ResearchAgent against the selected
ResearchProvider (default mock), prints a concise research summary, and writes
a ResearchPackage artifact (plus run state) under data/<run_id>/research/.
"""

from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is a declared dep
    load_dotenv = None

from agents.research.provider import DEFAULT_RESEARCH_DATA_PATH
from agents.research.provider import ResearchProviderError
from agents.research.provider import build_research_provider
from agents.research.stage import DEFAULT_SELECTION_PATH, load_topic_selection, run_research
from pipeline.state import PipelineState, Stage


def build_mock_selection(path: Path) -> None:
    """Create a cold-start TopicSelection artifact from the mock candidates."""
    from agents.seo.config import load_config
    from agents.seo.history import SelectionHistory
    from agents.seo.stage import load_candidates, run_topic_selection

    config = load_config()
    candidates = load_candidates(Path("data") / "mock" / "topic_candidates.json", config)
    selection = run_topic_selection(candidates, SelectionHistory(), config, run_id="mock")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(selection.to_json(), encoding="utf-8")
    print(f"[research_agent] wrote mock TopicSelection fixture: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agents.research.research_agent",
        description="Run research over a TopicSelection locally (mock or Gemini).",
    )
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION_PATH, help="TopicSelection artifact JSON")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_RESEARCH_DATA_PATH,
        help="mock research evidence JSON (ignored when --provider gemini)",
    )
    parser.add_argument(
        "--provider",
        choices=("mock", "gemini"),
        default="mock",
        help="research provider (default: mock/local JSON; gemini requires GEMINI_API_KEY)",
    )
    parser.add_argument(
        "--enable-url-context",
        action="store_true",
        help="with --provider gemini, also enable the URL Context retrieval tool (experimental)",
    )
    parser.add_argument(
        "--retriever",
        choices=("firecrawl",),
        default=None,
        help="with --provider gemini, retrieve real web sources via Firecrawl before "
        "claim extraction (keyless-capable; FIRECRAWL_API_KEY optional)",
    )
    parser.add_argument("--run-id", default=None, help="run id (default: generated)")
    parser.add_argument("--out", type=Path, default=None, help="output path for the ResearchPackage artifact")
    args = parser.parse_args()

    if args.retriever and args.provider != "gemini":
        print("[research_agent] --retriever firecrawl requires --provider gemini")
        return 2

    if not args.selection.is_file():
        if str(args.selection) != str(DEFAULT_SELECTION_PATH):
            raise SystemExit(f"[research_agent] TopicSelection not found: {args.selection}")
        build_mock_selection(args.selection)

    if load_dotenv is not None:
        load_dotenv()  # honour GEMINI_API_KEY/GEMINI_MODEL from .env (ignored by git)

    selection = load_topic_selection(args.selection)
    print(
        f"[research_agent] selected topic: {selection.selected_topic['title']!r} "
        f"(pillar={selection.selected_topic.get('pillar', '?')})"
    )

    try:
        builder_kwargs = dict(data_path=args.data, enable_url_context=args.enable_url_context)
        if args.retriever is not None:
            builder_kwargs["retriever"] = args.retriever
        provider = build_research_provider(args.provider, **builder_kwargs)
        run_id = args.run_id or uuid.uuid4().hex[:12]
        package = run_research(selection, provider, run_id=run_id)
    except ResearchProviderError as exc:
        print(f"[research_agent] provider error: {exc}")
        return 2
    except Exception as exc:
        print(f"[research_agent] provider error: {type(exc).__name__}: {exc}")
        return 2

    if args.provider == "gemini":
        package.produced_by = dict(provider.produced_by)
        package.research_notes = (
            _research_note(args.retriever),
            *package.research_notes[1:],
        )

    out_path = args.out or Path("data") / run_id / "research" / "research_package.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(package.to_json(), encoding="utf-8")

    try:
        state = PipelineState.load(Path("data") / run_id / "state.json")
    except FileNotFoundError:
        state = PipelineState(run_id=run_id)
    state.advance_to(Stage.RESEARCH)
    state.add_artifact(Stage.RESEARCH, out_path)
    state.save(Path("data") / run_id / "state.json")

    print(f"[research_agent] run_id={run_id}")
    print(f"[research_agent] provider={args.provider}")
    print(f"[research_agent] retriever={args.retriever or 'none'}")
    print(f"[research_agent] research_status={package.research_status}")
    print(
        f"[research_agent] {len(package.claims)} claim(s), {len(package.sources)} source(s) "
        f"(tiers={_tier_counts(package)})"
    )
    for claim in package.claims:
        print(
            f"  - [{claim.claim_type:<11}] conf={claim.confidence:.2f} {claim.claim_text!r}"
        )
    print(f"[research_agent] artifact written to {out_path}")
    print(f"[research_agent] run state written to {Path('data') / run_id / 'state.json'}")
    return 0


def _tier_counts(package) -> dict:
    from collections import Counter

    return dict(Counter(source.credibility_tier for source in package.sources))


def _research_note(retriever: str | None) -> str:
    if retriever:
        return (
            "Research evidence retrieved from the web via Firecrawl and extracted by "
            "Gemini; every source URL comes from retrieval metadata (no invented URLs). "
            "Claims remain provisional until the fact-check stage."
        )
    return (
        "Research evidence grounded in Gemini Google-Search retrieval; every "
        "source URL comes from retrieval metadata (no invented URLs). "
        "Claims remain provisional until the fact-check stage."
    )


if __name__ == "__main__":
    raise SystemExit(main())