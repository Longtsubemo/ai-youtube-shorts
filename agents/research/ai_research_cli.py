"""Stage 2 CLI: AI Research.

Reads ``data/<run_id>/topic_selection.json`` (written by Stage 1) and writes
``data/<run_id>/research.json``. Gemini's own knowledge is the research
basis - there is no web retrieval and no source/corroboration validation.

Usage:
    python -m agents.research.ai_research_cli --run-id <run_id>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from uuid import uuid4

from agents.research.ai_research import generate_ai_research, load_topic_selection, write_ai_research


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agents.research.ai_research_cli",
        description="Run AI research for a selected topic (Stage 2).",
    )
    parser.add_argument("--run-id", default=None, help="run id (default: generated)")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path for the AIResearch artifact (default: data/<run_id>/research.json)",
    )
    args = parser.parse_args()

    run_id = args.run_id or uuid4().hex[:12]
    run_dir = Path("data") / run_id

    selection_path = run_dir / "topic_selection.json"
    if not selection_path.exists():
        print(f"[ai_research] Missing Stage 1 artifact: {selection_path}")
        return 2

    selection = load_topic_selection(selection_path)
    research = generate_ai_research(selection)
    out_path = args.out or (run_dir / "research.json")
    write_ai_research(research, out_path)

    print(f"[ai_research] run_id={run_id}")
    print(f"[ai_research] topic={research.topic_id}")
    print(f"[ai_research] artifact written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())