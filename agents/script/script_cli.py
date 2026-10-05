"""Stage 3 CLI: Script Generation.

Reads ``data/<run_id>/research.json`` and writes ``data/<run_id>/script.json``.

Usage:
    python -m agents.script.script_cli --run-id <run_id>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from uuid import uuid4

from agents.script.script_models import Script, ScriptSection
from agents.script.stage_simple import write_script


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agents.script.script_cli",
        description="Generate a spoken script from AI research (Stage 3).",
    )
    parser.add_argument("--run-id", default=None, help="run id (default: generated)")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path for the Script artifact (default: data/<run_id>/script.json)",
    )
    args = parser.parse_args()

    run_id = args.run_id or uuid4().hex[:12]
    run_dir = Path("data") / run_id

    research_path = run_dir / "research.json"
    if not research_path.exists():
        print(f"[script] Missing Stage 2 artifact: {research_path}")
        return 2

    # NOT IMPLEMENTED: the Gemini-backed script writer. Until it exists this
    # stage emits an empty placeholder so the artifact chain can be exercised.
    script = Script(topic_id="", title="", sections=[])

    out_path = args.out or (run_dir / "script.json")
    write_script(script, out_path)

    print(f"[script] run_id={run_id}")
    print(f"[script] artifact written to {out_path}")
    print("[script] NOTE: script generation is a placeholder, not yet Gemini-backed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())