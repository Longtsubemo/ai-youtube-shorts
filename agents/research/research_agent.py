"""Manual runner for the research stage.

Usage:
    python -m agents.research.research_agent --run-id <run_id>
    python -m agents.research.research_agent --run-id <run_id> --provider gemini
    python -m agents.research.research_agent --run-id <run_id> --provider gemini --retriever hybrid
    python -m agents.research.research_agent --selection data/<run_id>/research/topic_selection.json --run-id <run_id>
    python -m agents.research.research_agent --selection data/mock/topic_selection.json --run-id mock
    python -m agents.research.research_agent --data data/mock/research_evidence.json

Loads a TopicSelection artifact, runs the ResearchAgent against the selected
ResearchProvider (default mock), prints a concise research summary, and writes
a ResearchPackage artifact (plus run state) under data/<run_id>/research/.

``--selection`` defaults to the current run's own Stage-1 artifact,
``data/<run_id>/research/topic_selection.json``. That default is never
fabricated: when the file is absent the CLI fails before any provider is
constructed, so a run cannot silently research a topic selected for a
different run. An explicit ``--selection`` still accepts any path, including
the ``data/mock/topic_selection.json`` fixture.

As a second line of defence, a selection whose recorded ``run_id`` belongs to
a different, non-placeholder run is rejected before the provider runs.

When the retriever records one, a retrieval_trace.json audit record is written
alongside the package; retrieval itself is unchanged by its presence.
"""

from __future__ import annotations

import argparse
import json
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
from agents.research.stage import load_topic_selection, run_research
from pipeline.state import PipelineState, Stage

SELECTION_SUBPATH = Path("research") / "topic_selection.json"

# A selection carrying one of these run ids is a fixture rather than the output
# of a real run, so its run id cannot be compared against the run being written.
PLACEHOLDER_SELECTION_RUN_IDS = frozenset({"mock"})


def default_selection_path(run_id: str) -> Path:
    """The Stage-1 artifact this run owns: data/<run_id>/research/topic_selection.json."""
    return Path("data") / run_id / SELECTION_SUBPATH


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agents.research.research_agent",
        description="Run research over a TopicSelection locally (mock or Gemini).",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=None,
        help="TopicSelection artifact JSON (default: this run's own Stage-1 artifact, "
        "data/<run_id>/research/topic_selection.json; never fabricated)",
    )
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
        choices=("firecrawl", "hybrid"),
        default=None,
        help="with --provider gemini, retrieve real web sources before claim extraction. "
        "'firecrawl' uses the Firecrawl Search API (keyless-capable; FIRECRAWL_API_KEY optional). "
        "'hybrid' uses self-hosted SearXNG discovery plus a BeautifulSoup fetch, falling back to "
        "self-hosted Firecrawl only for pages it could not read",
    )
    parser.add_argument("--run-id", default=None, help="run id (default: generated)")
    parser.add_argument("--out", type=Path, default=None, help="output path for the ResearchPackage artifact")
    args = parser.parse_args()

    if args.retriever and args.provider != "gemini":
        print("[research_agent] --retriever requires --provider gemini")
        return 2

    run_id = args.run_id or uuid.uuid4().hex[:12]
    selection_path = _resolve_selection_path(args.selection, run_id)
    selection = load_topic_selection(selection_path)
    _check_selection_run_id(selection, run_id, selection_path)

    if load_dotenv is not None:
        load_dotenv()  # honour GEMINI_API_KEY/GEMINI_MODEL from .env (ignored by git)

    print(
        f"[research_agent] selected topic: {selection.selected_topic['title']!r} "
        f"(pillar={selection.selected_topic.get('pillar', '?')})"
    )

    try:
        builder_kwargs = dict(data_path=args.data, enable_url_context=args.enable_url_context)
        if args.retriever is not None:
            builder_kwargs["retriever"] = args.retriever
        provider = build_research_provider(args.provider, **builder_kwargs)
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

    trace_path = _write_retrieval_trace(provider, out_path.parent)

    try:
        state = PipelineState.load(Path("data") / run_id / "state.json")
    except FileNotFoundError:
        state = PipelineState(run_id=run_id)
    state.advance_to(Stage.RESEARCH)
    state.add_artifact(Stage.RESEARCH, out_path)
    if trace_path is not None:
        state.add_artifact(Stage.RESEARCH, trace_path)
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
    if trace_path is not None:
        print(f"[research_agent] retrieval trace written to {trace_path}")
    print(f"[research_agent] run state written to {Path('data') / run_id / 'state.json'}")
    return 0


def _resolve_selection_path(explicit: Path | None, run_id: str) -> Path:
    """Resolve which TopicSelection this run researches, or fail loudly.

    An explicit ``--selection`` is honoured as given. Otherwise the run's own
    Stage-1 artifact is used. That default is deliberately never created: a
    missing Stage-1 artifact means the run has no selected topic, and silently
    substituting another run's selection would research the wrong topic under
    this run's id and artifacts.

    Raises SystemExit before any provider is constructed.
    """
    path = Path(explicit) if explicit is not None else default_selection_path(run_id)
    if path.is_file():
        return path
    if explicit is None:
        raise SystemExit(
            f"[research_agent] no TopicSelection for run {run_id!r}: {path} not found\n"
            f"[research_agent] run Stage 1 first, or pass --selection <path> "
            f"explicitly (the default is never fabricated)"
        )
    raise SystemExit(f"[research_agent] TopicSelection not found: {path}")


def _check_selection_run_id(selection, run_id: str, path: Path) -> None:
    """Reject a TopicSelection that belongs to a different real run.

    Guards the wrong-run/wrong-topic case from the other direction: pointing
    ``--selection`` at another run's artifact. A selection with no recorded run
    id, or a fixture run id, cannot be compared and is allowed through.

    Raises SystemExit before any provider is constructed.
    """
    selection_run_id = str(getattr(selection, "run_id", "") or "")
    if not selection_run_id or selection_run_id in PLACEHOLDER_SELECTION_RUN_IDS:
        return
    if selection_run_id != run_id:
        raise SystemExit(
            f"[research_agent] TopicSelection belongs to a different run: {path} "
            f"has run_id={selection_run_id!r} but this run is {run_id!r}\n"
            f"[research_agent] research the topic selected for this run, or pass "
            f"--selection data/{run_id}/research/topic_selection.json"
        )


def _write_retrieval_trace(provider, out_dir: Path) -> Path | None:
    """Persist the retriever's audit record beside the research package.

    Observability only. The trace is produced during retrieval and read here
    afterwards, so writing it cannot change which sources were fetched, ranked or
    emitted. It is a separate file rather than a new `ResearchPackage` field so
    the evidence contract does not grow.

    Returns None - leaving the artifact absent rather than empty - when the
    provider exposes no trace: the mock provider has no retriever, and the
    Firecrawl retriever records no trace.
    """
    trace = getattr(provider, "last_retrieval_trace", None)
    to_dict = getattr(trace, "to_dict", None)
    if not callable(to_dict):
        return None

    payload = {
        "retriever": getattr(provider, "retriever_name", None),
        "provider": getattr(provider, "provider_name", None),
        "model": getattr(provider, "model", None),
        "trace": to_dict(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "retrieval_trace.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _tier_counts(package) -> dict:
    from collections import Counter

    return dict(Counter(source.credibility_tier for source in package.sources))


def _research_note(retriever: str | None) -> str:
    if retriever == "hybrid":
        return (
            "Research evidence retrieved from the web via self-hosted SearXNG "
            "discovery and page fetching, with a bounded self-hosted Firecrawl "
            "fallback for pages the cheap fetch could not read, then extracted by "
            "Gemini; every source URL comes from retrieval metadata (no invented "
            "URLs). Claims remain provisional until the fact-check stage."
        )
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