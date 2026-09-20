"""Stage-3 entrypoint: ResearchPackage -> FactCheckResult.

Content-system side only (docs/ARCHITECTURE.md): consumes a Stage-2
ResearchPackage artifact and evaluates each claim against *supplied evidence*
only - never against the model's own world knowledge. Uses local/mocked
providers exclusively; no network, no APIs, no LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

from agents.qa.fact_checker.checker import FactChecker
from agents.qa.fact_checker.evidence import (
    DEFAULT_EVIDENCE_DATA_PATH,
    EvidenceProvider,
    MockEvidenceProvider,
)
from agents.qa.fact_checker.gates import FactCheckConfig
from agents.qa.fact_checker.models import FactCheckResult
from agents.qa.fact_checker.verifier import MockVerifier, Verifier
from agents.research.models import ResearchPackage

DEFAULT_PACKAGE_PATH = Path("data") / "mock" / "research_package.json"


def load_research_package(path: str | Path = DEFAULT_PACKAGE_PATH) -> ResearchPackage:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return ResearchPackage.from_dict(raw)


def run_fact_check(
    package: ResearchPackage,
    provider: EvidenceProvider | None = None,
    verifier: Verifier | None = None,
    config: FactCheckConfig | None = None,
    run_id: str = "",
    created_at: str | None = None,
    evidence_data_path: str | Path = DEFAULT_EVIDENCE_DATA_PATH,
) -> FactCheckResult:
    """Run the fact-check stage over a ResearchPackage, returning a FactCheckResult."""
    provider = provider or MockEvidenceProvider(data_path=str(evidence_data_path))
    verifier = verifier or MockVerifier()
    checker = FactChecker(provider, verifier=verifier, config=config, run_id=run_id, created_at=created_at)
    return checker.check(package)