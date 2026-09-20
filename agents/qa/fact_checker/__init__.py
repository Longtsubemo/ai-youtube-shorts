"""Stage 3 - evidence-grounded fact checking.

Public surface of the fact-check stage:

    run_fact_check(package) -> FactCheckResult
    FactChecker / FactCheckConfig / MockVerifier / MockEvidenceProvider

The stage evaluates whether the *supplied evidence* supports the exact wording
of each research claim (docs/FACT_VERIFICATION.md). It is an evidence verifier,
not a source of truth: it never invents citations, excerpts, or facts, and it
never falls back on world knowledge to settle a claim.
"""

from __future__ import annotations

from agents.qa.fact_checker.checker import FactChecker
from agents.qa.fact_checker.evidence import (
    DEFAULT_EVIDENCE_DATA_PATH,
    EvidenceProvider,
    EvidenceProviderError,
    MockEvidenceProvider,
    SUPPORTING_EVIDENCE_TYPES,
)
from agents.qa.fact_checker.gates import (
    FactCheckConfig,
    GateVerdict,
    aggregate_package,
    evaluate_claim,
)
from agents.qa.fact_checker.models import (
    EVIDENCE_TYPES,
    VERDICTS,
    ClaimVerdict,
    Evidence,
    FactCheckError,
    FactCheckResult,
)
from agents.qa.fact_checker.stage import load_research_package, run_fact_check
from agents.qa.fact_checker.verifier import (
    ClaimAssessment,
    MockVerifier,
    Verifier,
)

__all__ = [
    "ClaimAssessment",
    "ClaimVerdict",
    "EVIDENCE_TYPES",
    "Evidence",
    "EvidenceProvider",
    "EvidenceProviderError",
    "VERDICTS",
    "Verifier",
    "FactCheckConfig",
    "FactChecker",
    "FactCheckError",
    "FactCheckResult",
    "GateVerdict",
    "MockEvidenceProvider",
    "MockVerifier",
    "SUPPORTING_EVIDENCE_TYPES",
    "aggregate_package",
    "evaluate_claim",
    "load_research_package",
    "run_fact_check",
]