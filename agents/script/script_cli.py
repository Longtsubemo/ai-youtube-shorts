"""Stage 3 CLI: Script Generation.

Reads ``data/<run_id>/topic_selection.json`` (Stage 1) and
``data/<run_id>/research.json`` (Stage 2), then writes
``data/<run_id>/script.json`` - a finished spoken YouTube Shorts script.

Gemini's own knowledge carries the material through from Stage 2; this stage
only shapes it into narration. There is no web retrieval, no source harvesting,
and no QA score or PASS/FAIL verdict here.

Usage:
    python -m agents.script.script_cli --run-id <run_id>
    python -m agents.script.script_cli --run-id <run_id> --provider gemini
"""

from __future__ import annotations

import argparse
from pathlib import Path
from uuid import uuid4

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is a declared dep
    load_dotenv = None

from agents.script.provider import ScriptProviderError, build_script_provider
from agents.script.stage import (
    ScriptInputError,
    load_research,
    load_topic_selection,
    run_script_generation,
    write_script,
)
from agents.seo.config import DEFAULT_CONFIG_PATH, load_config
from pipeline.state import PipelineState, Stage

PROVIDERS = ("mock", "gemini")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agents.script.script_cli",
        description="Generate a spoken script from AI research (Stage 3).",
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
        help="script writer (default: mock; gemini requires GEMINI_API_KEY)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path for the script artifact (default: data/<run_id>/script.json)",
    )
    args = parser.parse_args()

    if load_dotenv is not None:
        load_dotenv()  # honour GEMINI_API_KEY/GEMINI_MODEL from .env (ignored by git)

    run_id = args.run_id or uuid4().hex[:12]
    run_dir = Path("data") / run_id

    selection_path = run_dir / "topic_selection.json"
    if not selection_path.exists():
        print(f"[script] Missing Stage 1 artifact: {selection_path}")
        return 2

    research_path = run_dir / "research.json"
    if not research_path.exists():
        print(f"[script] Missing Stage 2 artifact: {research_path}")
        return 2

    try:
        config = load_config(args.config)
    except Exception as exc:  # noqa: BLE001 - surfaced as a CLI error, not a traceback
        print(f"[script] config error: {exc}")
        return 2

    try:
        selection = load_topic_selection(selection_path)
        research = load_research(research_path)
        provider = build_script_provider(args.provider, config=config)
        script = run_script_generation(selection, research, provider, run_id=run_id)
    except (ScriptInputError, ScriptProviderError) as exc:
        print(f"[script] input error: {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001 - provider failures must not traceback
        print(f"[script] provider error: {type(exc).__name__}: {exc}")
        return 2

    out_path = args.out or (run_dir / "script.json")
    write_script(script, out_path)

    state = PipelineState(run_id=run_id)
    state.advance_to(Stage.SCRIPT_GENERATION)
    state.add_artifact(Stage.SCRIPT_GENERATION, out_path)
    state.save(run_dir / "state.json")

    roles = ",".join(section.role for section in script.sections)
    print(f"[script] run_id={run_id}")
    print(f"[script] provider={args.provider} topic={script.topic_id} pillar={script.pillar}")
    print(
        f"[script] sections={len(script.sections)} roles={roles} "
        f"target={script.duration_target_seconds}s estimated={script.estimated_seconds}s"
    )
    print(f"[script] artifact written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())