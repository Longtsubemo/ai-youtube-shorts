"""Manual runner for the Stage-5 script-QA stage (no external APIs).

Usage:
    python -m agents.qa.script

    # explicit artifacts:
    python -m agents.qa.script --script data/<run_id>/scripts/script.json \
        --result data/<run_id>/research/fact_check_result.json

Loads the Script artifact + its FactCheckResult (and the approved claim
records), runs all 13 QA checks, prints the verdict/score/issues/warnings, and
writes a ScriptQAResult artifact (plus run state) under data/<run_id>/scripts/.
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from agents.qa.script.models import ScriptQAResult
from agents.qa.script.stage import (
    DEFAULT_FACT_CHECK_RESULT_PATH,
    DEFAULT_SCRIPT_PATH,
    claims_index,
    load_fact_check_result,
    load_script,
    run_script_qa,
)
from agents.research.models import ResearchPackage
from agents.research.provider import MockResearchProvider
from agents.research.stage import load_topic_selection, run_research
from pipeline.state import PipelineState, Stage


def build_mock_package():
    """Deterministically produce the mock Stage-2 ResearchPackage in memory."""
    selection = load_topic_selection()
    return run_research(selection, MockResearchProvider(), run_id="mock")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agents.qa.script",
        description="Run deterministic script QA over a Stage-4 Script artifact.",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=DEFAULT_SCRIPT_PATH,
        help="Script artifact JSON (default: data/mock/script.json)",
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=DEFAULT_FACT_CHECK_RESULT_PATH,
        help="FactCheckResult artifact JSON (default: data/mock/fact_check_result.json)",
    )
    parser.add_argument("--run-id", default=None, help="run id (default: script's run id)")
    parser.add_argument(
        "--research",
        type=Path,
        default=None,
        help=(
            "Stage-2 ResearchPackage artifact JSON. Supply this for a real run: "
            "QA resolves approved claim ids against THIS package, so the default "
            "mock package cannot validate a real Script."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output root under which <run_id>/scripts/script_qa_result.json is written",
    )
    args = parser.parse_args()

    if not args.script.is_file():
        raise SystemExit(f"[script_qa] Script not found: {args.script}")
    if not args.result.is_file():
        raise SystemExit(f"[script_qa] FactCheckResult not found: {args.result}")

    script = load_script(args.script)
    result = load_fact_check_result(args.result)
    if args.research is not None:
        if not args.research.is_file():
            raise SystemExit(f"[script_qa] ResearchPackage not found: {args.research}")
        package = ResearchPackage.from_json(args.research.read_text(encoding="utf-8"))
    else:
        package = build_mock_package()
    if args.run_id:
        run_id = args.run_id
    elif args.script == DEFAULT_SCRIPT_PATH and args.result == DEFAULT_FACT_CHECK_RESULT_PATH:
        # Default demo invocation: keep data/mock/ (committed fixtures) clean.
        run_id = "sqa-demo"
    else:
        run_id = script.run_id or result.run_id or uuid.uuid4().hex[:12]

    qa_result = run_script_qa(
        script,
        result,
        package=package,
        run_id=run_id,
    )

    out_root = args.out or Path("data")
    out_path = out_root / run_id / "scripts" / "script_qa_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(qa_result.to_json(), encoding="utf-8")

    state_path = out_root / run_id / "state.json"
    try:
        state = PipelineState.load(state_path)
    except FileNotFoundError:
        state = PipelineState(run_id=run_id)
    state.advance_to(Stage.SCRIPT_QA)
    state.add_artifact(Stage.SCRIPT_QA, out_path)
    state.save(state_path)

    _print_report(qa_result)
    print(f"[script_qa] artifact written to {out_path}")
    print(f"[script_qa] run state written to {state_path}")
    return 0 if qa_result.overall_verdict == "PASS" else 1


def _print_report(qa_result: ScriptQAResult) -> None:
    print(f"[script_qa] run_id={qa_result.run_id}")
    print(f"[script_qa] script_id={qa_result.script_id}")
    print(f"[script_qa] overall_verdict={qa_result.overall_verdict}")
    print(f"[script_qa] score={qa_result.score:.1f}")
    coverage = qa_result.approved_claim_coverage
    if coverage is not None:
        print(
            f"[script_qa] coverage={coverage.coverage_ratio} "
            f"(used {len(coverage.used_claim_ids)}/"
            f"{len(coverage.approved_claim_ids)} approved, "
            f"critical unused: {coverage.critical_unused_claim_ids or '-'})"
        )
    if qa_result.issues:
        print("[script_qa] errors:")
        for issue in qa_result.issues:
            print(
                f"  {issue.issue_id} [{issue.category}][{issue.gate}] "
                f"{issue.message}"
            )
    if qa_result.warnings:
        print("[script_qa] warnings:")
        for issue in qa_result.warnings:
            print(f"  {issue.issue_id} [{issue.category}] {issue.message}")
    if qa_result.infos:
        print("[script_qa] infos:")
        for issue in qa_result.infos:
            print(f"  {issue.issue_id} [{issue.category}] {issue.message}")
    print(
        "[script_qa] score detail: "
        + ", ".join(f"{dim}={value:.1f}" for dim, value in qa_result.score_detail.get("dimensions", {}).items())
    )


if __name__ == "__main__":
    raise SystemExit(main())