"""Manual runner for the research stage (no external APIs).

Usage:
    python -m agents.research.research_agent
    python -m agents.research.research_agent --selection data/<run_id>/research/topic_selection.json
    python -m agents.research.research_agent --data data/mock/research_evidence.json

Loads a TopicSelection artifact, runs the ResearchAgent against the
(mock) ResearchProvider, prints a concise research summary, and writes a
ResearchPackage artifact (plus run state) under data/<run_id>/research/.
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from agents.research.provider import MockResearchProvider, DEFAULT_RESEARCH_DATA_PATH
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
        description="Run research over a TopicSelection locally with mock data.",
    )
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION_PATH, help="TopicSelection artifact JSON")
    parser.add_argument("--data", type=Path, default=DEFAULT_RESEARCH_DATA_PATH, help="mock research evidence JSON")
    parser.add_argument("--run-id", default=None, help="run id (default: generated)")
    parser.add_argument("--out", type=Path, default=None, help="output path for the ResearchPackage artifact")
    args = parser.parse_args()

    if not args.selection.is_file():
        if str(args.selection) != str(DEFAULT_SELECTION_PATH):
            raise SystemExit(f"[research_agent] TopicSelection not found: {args.selection}")
        build_mock_selection(args.selection)

    selection = load_topic_selection(args.selection)
    print(
        f"[research_agent] selected topic: {selection.selected_topic['title']!r} "
        f"(pillar={selection.selected_topic.get('pillar', '?')})"
    )

    provider = MockResearchProvider(data_path=args.data)
    run_id = args.run_id or uuid.uuid4().hex[:12]
    package = run_research(selection, provider, run_id=run_id)

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


if __name__ == "__main__":
    raise SystemExit(main())