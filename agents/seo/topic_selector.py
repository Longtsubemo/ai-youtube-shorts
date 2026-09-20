"""Manual runner for the topic-selection stage (no external APIs).

Usage:
    python -m agents.seo.topic_selector
    python -m agents.seo.topic_selector --history data/mock/history_sample.json

Loads mock/local candidates + history, scores them, selects a topic,
prints the result, and writes a TopicSelection artifact (plus run state)
under data/<run_id>/.
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from agents.seo.config import DEFAULT_CONFIG_PATH, load_config
from agents.seo.history import SelectionHistory
from agents.seo.result import TopicSelection
from agents.seo.stage import load_candidates, load_history, run_topic_selection
from pipeline.state import PipelineState, Stage


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agents.seo.topic_selector",
        description="Run topic discovery/scoring/selection locally with mock data.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="topic-selection config YAML")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("data") / "mock" / "topic_candidates.json",
        help="candidate JSON (default: mock dataset)",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=None,
        help="history JSON (default: cold start, empty history)",
    )
    parser.add_argument("--run-id", default=None, help="run id (default: generated)")
    parser.add_argument("--out", type=Path, default=None, help="output path for the TopicSelection artifact")
    args = parser.parse_args()

    config = load_config(args.config)
    candidates = load_candidates(args.candidates, config)

    if args.history:
        history = load_history(args.history)
        print(f"[topic_selector] history: {len(history.records)} record(s) from {args.history}")
    else:
        history = SelectionHistory()
        print("[topic_selector] no --history given; running as cold start")

    run_id = args.run_id or uuid.uuid4().hex[:12]
    selection = run_topic_selection(candidates, history, config, run_id=run_id)

    out_path = args.out or Path("data") / run_id / "research" / "topic_selection.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(selection.to_json(), encoding="utf-8")

    state = PipelineState(run_id=run_id)
    state.advance_to(Stage.TOPIC_SELECTION)
    state.add_artifact(Stage.TOPIC_SELECTION, out_path)
    state.save(Path("data") / run_id / "state.json")

    print(f"[topic_selector] run_id={run_id}")
    print(f"[topic_selector] mode={selection.selection_mode} historical_data_available={selection.historical_data_available}")
    print(f"[topic_selector] selected: #{_rank(selection, selection.selected_topic_id)} "
          f"{selection.selected_topic['title']!r} "
          f"(pillar={selection.selected_topic['pillar']}, score={selection.selected_topic['total_score']:.4f})")
    print("[topic_selector] ranking:")
    for rank, candidate in enumerate(selection.ranked_candidates, start=1):
        marker = " <--" if candidate["topic_id"] == selection.selected_topic_id else ""
        print(
            f"  {rank:>2}. {candidate['title']!r} "
            f"pillar={candidate['pillar']:<10} score={candidate['total_score']:.4f}{marker}"
        )
    print(f"[topic_selector] reason: {selection.selection_reason}")
    print(f"[topic_selector] artifact written to {out_path}")
    print(f"[topic_selector] run state written to {Path('data') / run_id / 'state.json'}")
    return 0


def _rank(selection: TopicSelection, topic_id: str) -> int:
    for rank, candidate in enumerate(selection.ranked_candidates, start=1):
        if candidate["topic_id"] == topic_id:
            return rank
    return 0


if __name__ == "__main__":
    raise SystemExit(main())