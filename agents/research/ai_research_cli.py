"""Stage 2 CLI: AI Research.

Reads ``data/<run_id>/topic_selection.json`` (written by Stage 1) and writes
the flat artifact ``data/<run_id>/research.json``. Gemini's own knowledge is
the research basis - there is no web retrieval, no source harvesting, and no
corroboration or factuality validation in this stage.

Usage:
    python -m agents.research.ai_research_cli --run-id <run_id>
    python -m agents.research.ai_research_cli --run-id <run_id> --provider gemini
"""

from __future__ import annotations

import argparse
from pathlib import Path
from uuid import uuid4

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is a declared dep
    load_dotenv = None

from agents.research.ai_research import (
    generate_ai_research,
    load_topic_selection,
    write_ai_research,
)
from agents.research.provider import (
    AIResearchProviderError,
    build_ai_research_provider,
)
from agents.seo.config import DEFAULT_CONFIG_PATH, load_config
from pipeline.state import PipelineState, Stage

PROVIDERS = ("mock", "gemini")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agents.research.ai_research_cli",
        description="Run AI research for a selected topic (Stage 2).",
    )
    parser.add_argument("--run-id", default=None, help="run id (default: generated)")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="topic-selection config YAML (supplies channel + pillar context)",
    )
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        default="mock",
        help="research source (default: mock; gemini requires GEMINI_API_KEY)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path for the research artifact (default: data/<run_id>/research.json)",
    )
    args = parser.parse_args()

    if load_dotenv is not None:
        load_dotenv()  # honour GEMINI_API_KEY/GEMINI_MODEL from .env (ignored by git)

    run_id = args.run_id or uuid4().hex[:12]
    run_dir = Path("data") / run_id

    selection_path = run_dir / "topic_selection.json"
    if not selection_path.exists():
        print(f"[ai_research] Missing Stage 1 artifact: {selection_path}")
        return 2

    try:
        config = load_config(args.config)
    except Exception as exc:  # noqa: BLE001 - surfaced as a CLI error, not a traceback
        print(f"[ai_research] config error: {exc}")
        return 2

    selection = load_topic_selection(selection_path)

    try:
        provider = build_ai_research_provider(args.provider, config=config)
        research = generate_ai_research(selection, provider, run_id=run_id)
    except AIResearchProviderError as exc:
        print(f"[ai_research] provider error: {exc.message}")
        return 2
    except Exception as exc:  # noqa: BLE001 - provider failures must not traceback
        print(f"[ai_research] provider error: {type(exc).__name__}: {exc}")
        return 2

    out_path = args.out or (run_dir / "research.json")
    write_ai_research(research, out_path)

    state = PipelineState(run_id=run_id)
    state.advance_to(Stage.AI_RESEARCH)
    state.add_artifact(Stage.AI_RESEARCH, out_path)
    state.save(run_dir / "state.json")

    print(f"[ai_research] run_id={run_id}")
    print(f"[ai_research] provider={args.provider}")
    print(f"[ai_research] topic={research.topic_id} pillar={research.pillar}")
    print(
        f"[ai_research] concepts={len(research.key_concepts)} examples={len(research.examples)} "
        f"numbers={len(research.numbers)} angles={len(research.interesting_angles)} "
        f"misconceptions={len(research.common_misconceptions)} stories={len(research.story_ideas)}"
    )
    print(f"[ai_research] artifact written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())