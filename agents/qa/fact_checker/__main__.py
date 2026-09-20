"""Manual runner for the Stage-3 fact-check stage (no external APIs).

Usage:
    python -m agents.qa.fact_checker

    # existing research package:
    python -m agents.qa.fact_checker --package data/<run_id>/research/research_package.json
    python -m agents.qa.fact_checker --evidence data/mock/fact_check_evidence.json

Loads a ResearchPackage (building the mock one deterministically if none is
given), runs the checker against the mock evidence provider, prints claim-level
results + the package verdict, and writes a FactCheckResult artifact (plus run
state) under data/<run_id>/research/.
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from agents.qa.fact_checker.evidence import (
    DEFAULT_EVIDENCE_DATA_PATH,
    MockEvidenceProvider,
)
from agents.qa.fact_checker.models import FactCheckResult
from agents.qa.fact_checker.stage import load_research_package, run_fact_check
from agents.research.models import ResearchPackage
from agents.research.provider import MockResearchProvider
from agents.research.stage import load_topic_selection, run_research
from pipeline.state import PipelineState, Stage

DEFAULT_SELECTION_PATH = Path("data") / "mock" / "topic_selection.json"


def build_mock_package() -> ResearchPackage:
    """Deterministically produce the mock Stage-2 ResearchPackage in memory."""
    selection = load_topic_selection(DEFAULT_SELECTION_PATH)
    return run_research(selection, MockResearchProvider(), run_id="mock")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agents.qa.fact_checker",
        description="Run evidence-grounded fact checking locally with mock data.",
    )
    parser.add_argument(
        "--package",
        type=Path,
        default=None,
        help="ResearchPackage artifact JSON (default: build the mock package)",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE_DATA_PATH,
        help="mock evidence fixture JSON",
    )
    parser.add_argument("--run-id", default=None, help="run id (default: generated)")
    parser.add_argument("--out", type=Path, default=None, help="output path for the FactCheckResult artifact")
    args = parser.parse_args()

    if args.package is not None:
        if not args.package.is_file():
            raise SystemExit(f"[fact_checker] ResearchPackage not found: {args.package}")
        package = load_research_package(args.package)
        run_id = args.run_id or package.run_id or uuid.uuid4().hex[:12]
        print(f"[fact_checker] loaded package from {args.package}")
    else:
        package = build_mock_package()
        run_id = args.run_id or uuid.uuid4().hex[:12]
        print(
            "[fact_checker] built mock ResearchPackage "
            f"{package.topic_title!r} ({len(package.claims)} claims, {len(package.sources)} sources)"
        )

    provider = MockEvidenceProvider(data_path=args.evidence)
    result = run_fact_check(package, provider=provider, run_id=run_id)

    out_path = args.out or Path("data") / run_id / "research" / "fact_check_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.to_json(), encoding="utf-8")

    try:
        state = PipelineState.load(Path("data") / run_id / "state.json")
    except FileNotFoundError:
        state = PipelineState(run_id=run_id)
    state.advance_to(Stage.FACT_CHECK)
    state.add_artifact(Stage.FACT_CHECK, out_path)
    state.save(Path("data") / run_id / "state.json")

    print(f"[fact_checker] run_id={run_id}")
    print(f"[fact_checker] package_verdict={result.package_verdict}")
    for claim_result in result.claim_results:
        verdict = f"{claim_result.verdict:>6}"
        marker = " (CRITICAL)" if claim_result.critical else ""
        print(
            f"  {verdict} conf={claim_result.confidence:.2f} "
            f"{claim_result.claim_id}{marker} {_short(claim_result.rationale)}"
        )
    print(f"[fact_checker] approved={len(result.approved_claims)} "
          f"review={len(result.review_items)} rejected={len(result.rejected_claims)}")
    if result.critical_claim_failures:
        print(f"[fact_checker] critical claim failures: {', '.join(result.critical_claim_failures)}")
    print(f"[fact_checker] artifact written to {out_path}")
    print(f"[fact_checker] run state written to {Path('data') / run_id / 'state.json'}")
    return 0


def _short(text: str, limit: int = 140) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


if __name__ == "__main__":
    raise SystemExit(main())