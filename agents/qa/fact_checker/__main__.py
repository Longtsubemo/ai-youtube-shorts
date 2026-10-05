"""Manual runner for the Stage-3 fact-check stage.

Usage:
    python -m agents.qa.fact_checker

    # existing research package:
    python -m agents.qa.fact_checker --package data/<run_id>/research/research_package.json
    python -m agents.qa.fact_checker --evidence data/mock/fact_check_evidence.json

    # real Gemini fact checking (Firecrawl verification retrieval):
    python -m agents.qa.fact_checker --provider gemini \
        --package data/<run_id>/research/research_package.json \
        --retriever firecrawl

Loads a ResearchPackage (building the mock one deterministically if none is
given), runs the checker, prints claim-level results + the package verdict, and
writes a FactCheckResult artifact (plus run state) under data/<run_id>/research/.

The default provider is the deterministic mock (no network, no key). The gemini
provider requires GEMINI_API_KEY and — for real evidence — a retriever; provider
errors (missing key, auth, quota, API or parse/validation failures, unknown
provider/retriever) exit with code 2 and write no partial artifact.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is a declared dep
    load_dotenv = None

from agents.qa.fact_checker.evidence import (
    DEFAULT_EVIDENCE_DATA_PATH,
    EvidenceProviderError,
    MockEvidenceProvider,
)
from agents.qa.fact_checker.models import FactCheckResult
from agents.qa.fact_checker.stage import load_research_package, run_fact_check
from agents.research.models import ResearchPackage
from agents.research.provider import MockResearchProvider, ResearchProviderError
from agents.research.stage import load_topic_selection, run_research
from pipeline.state import PipelineState, Stage
from providers.gemini.errors import GeminiProviderError

DEFAULT_SELECTION_PATH = Path("data") / "mock" / "topic_selection.json"


def build_mock_package() -> ResearchPackage:
    """Deterministically produce the mock Stage-2 ResearchPackage in memory."""
    selection = load_topic_selection(DEFAULT_SELECTION_PATH)
    return run_research(selection, MockResearchProvider(), run_id="mock")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agents.qa.fact_checker",
        description="Run evidence-grounded fact checking with the mock or Gemini provider.",
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
        help="mock evidence fixture JSON (mock provider only)",
    )
    parser.add_argument(
        "--provider",
        choices=("mock", "gemini"),
        default="mock",
        help="fact-check provider (default: mock, deterministic and network-free)",
    )
    parser.add_argument(
        "--retriever",
        default=None,
        help="real web retrieval for --provider gemini (firecrawl | hybrid)",
    )
    parser.add_argument("--run-id", default=None, help="run id (default: generated)")
    parser.add_argument("--out", type=Path, default=None, help="output path for the FactCheckResult artifact")
    args = parser.parse_args()

    if load_dotenv is not None:
        load_dotenv()  # honour GEMINI_API_KEY/GEMINI_MODEL from .env (ignored by git)

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

    if args.provider == "gemini":
        if args.retriever is not None and args.retriever not in ("firecrawl", "hybrid"):
            print(f"[fact_checker] unknown retriever: {args.retriever!r}", file=sys.stderr)
            return 2
        components = _build_gemini(args)
        if components is None:
            return 2
        provider, verifier = components.provider, components.verifier
    else:
        if args.retriever is not None:
            print("[fact_checker] --retriever is only supported with --provider gemini", file=sys.stderr)
            return 2
        provider = MockEvidenceProvider(data_path=args.evidence)
        verifier = None

    try:
        result = run_fact_check(package, provider=provider, verifier=verifier, run_id=run_id)
    except (EvidenceProviderError, ResearchProviderError, GeminiProviderError) as exc:
        if args.provider == "gemini":
            print(f"[fact_checker] provider error: {exc}", file=sys.stderr)
            return 2
        raise
    if _evidence_provider_failed(args, result):
        return 2

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


def _build_gemini(args) -> "object | None":
    """Build the gemini (provider, verifier) pair; returns None on error."""
    from agents.qa.fact_checker.provider import build_fact_check_components

    try:
        return build_fact_check_components("gemini", retriever=args.retriever)
    except (EvidenceProviderError, ResearchProviderError, GeminiProviderError) as exc:
        print(f"[fact_checker] provider error: {exc}", file=sys.stderr)
        return None


def _evidence_provider_failed(args, result) -> bool:
    """True when a gemini evidence-provider failure was recorded as an issue.

    The checker records a per-source provider failure as an `evidence_issue`
    (no evidence -> REVIEW) so the shared mock path stays graceful, but the
    gemini slice treats a provider/API/parse/validation failure as fatal:
    abort with code 2 and never write a partial artifact (no false REVIEW).
    """
    if args.provider != "gemini":
        return False
    for issue in result.evidence_issues:
        if issue.startswith("evidence provider failed"):
            print(
                "[fact_checker] provider error during evidence collection; "
                f"aborting without artifact: {issue}",
                file=sys.stderr,
            )
            return True
    return False


def _short(text: str, limit: int = 140) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


if __name__ == "__main__":
    raise SystemExit(main())