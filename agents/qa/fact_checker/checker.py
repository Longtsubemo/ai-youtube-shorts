"""FactChecker: ResearchPackage -> evidence -> verifier -> gates -> FactCheckResult.

The checker is a pure orchestration shell. It does four things:

  1. Collects evidence from the provider for every source in the package,
     dropping (and recording) any evidence that does not resolve to a real
     source or fails validation - a dangling citation can never back a PASS.
  2. Runs the (constrained) verifier per claim.
  3. Runs the deterministic gates per claim to mint a final verdict.
  4. Aggregates the package verdict and the Stage-4-approved claim lists.

The checker itself contains no world knowledge and makes no network calls.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from agents.qa.fact_checker.evidence import EvidenceProvider
from agents.qa.fact_checker.gates import FactCheckConfig, aggregate_package, evaluate_claim
from agents.qa.fact_checker.models import (
    ClaimVerdict,
    Evidence,
    FactCheckError,
    FactCheckResult,
    result_id_for,
)
from agents.qa.fact_checker.verifier import MockVerifier, Verifier
from agents.research.models import ResearchPackage
from agents.seo.models import Provenance, utc_now

GENERATED_BY = "fact_check_agent_v1"


class FactChecker:
    """Orchestrates Stage 3 over a ResearchPackage."""

    def __init__(
        self,
        provider: EvidenceProvider,
        verifier: Verifier | None = None,
        config: FactCheckConfig | None = None,
        run_id: str = "",
        created_at: str | None = None,
    ) -> None:
        self._provider = provider
        self._verifier = verifier or MockVerifier()
        self._config = config or FactCheckConfig()
        self._run_id = run_id
        self._created_at = created_at

    def check(self, package: ResearchPackage) -> FactCheckResult:
        package.validate(traceability=False)

        source_ids = {source.source_id for source in package.sources}
        evidence_index, issues = self._collect_evidence(package, source_ids)

        claim_results = tuple(
            self._check_claim(claim, evidence_index, package) for claim in package.claims
        )

        summary, review_items, rejected_claims, approved_claims, critical_failures = aggregate_package(
            claim_results, self._config
        )

        provider_meta = getattr(self._provider, "produced_by", None)
        verifier_meta = getattr(self._verifier, "produced_by", None)
        if provider_meta is not None:
            provider_label = provider_meta.get("provider") or self._provider.__class__.__name__
            model_label = (verifier_meta or {}).get("model") or self._verifier.__class__.__name__
            prompt_version = (verifier_meta or {}).get("prompt_version") or provider_meta.get("prompt_version")
            produced_by = {
                "prompt_version": prompt_version,
                "provider": provider_label,
                "model": model_label,
            }
            provenance_source = f"{provider_label}:{model_label}"
            provenance_provider = provider_label
            provenance_model = model_label
            provenance_prompt_version = prompt_version
        else:
            produced_by = {
                "prompt_version": None,
                "provider": self._provider.__class__.__name__,
                "model": self._verifier.__class__.__name__,
            }
            provenance_source = f"{self._provider.__class__.__name__}:{self._verifier.__class__.__name__}"
            provenance_provider = None
            provenance_model = "MockVerifier"
            provenance_prompt_version = None

        provenance = Provenance(
            generated_by=GENERATED_BY,
            source=provenance_source,
            created_at=self._created_at or utc_now(),
            provider=provenance_provider,
            model=provenance_model,
            prompt_version=provenance_prompt_version,
        )
        result = FactCheckResult(
            result_id=result_id_for(self._run_id, package.package_id),
            topic_id=package.topic_id,
            topic_title=package.topic_title,
            package_id=package.package_id,
            claim_results=claim_results,
            package_verdict=summary["package_verdict"],
            critical_claim_failures=tuple(critical_failures),
            review_items=tuple(review_items),
            rejected_claims=tuple(rejected_claims),
            approved_claims=tuple(approved_claims),
            evidence_issues=tuple(issues),
            provenance=provenance,
            created_at=self._created_at or provenance.created_at,
            run_id=self._run_id or package.run_id,
            produced_by=produced_by,
        )
        return result.validate()

    # ------------------------------------------------------------------ #

    def _collect_evidence(self, package: ResearchPackage, source_ids: set[str]) -> tuple[dict[str, list[Evidence]], list[str]]:
        """Fetch + validate evidence; drop anything that cannot be traced."""
        index: dict[str, list[Evidence]] = {}
        issues: list[str] = []
        for source in package.sources:
            try:
                entries = self._provider.get_evidence(source)
            except Exception as exc:  # provider misbehaves -> clean issue, not a crash
                issues.append(f"evidence provider failed for {source.source_id}: {exc}")
                entries = []
            for evidence in entries:
                try:
                    evidence.validate()
                except FactCheckError as exc:
                    issues.append(f"malformed evidence for {source.source_id}: {exc}")
                    continue
                if evidence.source_id not in source_ids:
                    issues.append(
                        f"evidence {evidence.evidence_id} references unknown source "
                        f"{evidence.source_id!r}"
                    )
                    continue
                index.setdefault(source.source_id, []).append(evidence)
        return index, issues

    def _check_claim(self, claim, evidence_index: Mapping[str, Sequence[Evidence]], package: ResearchPackage) -> ClaimVerdict:
        sources_by_id = {source.source_id: source for source in package.sources}
        evidence = [
            item for source_id in claim.source_ids for item in evidence_index.get(source_id, ())
        ]
        assessment = self._verifier.verify(claim, evidence, sources_by_id)
        gate_verdict = evaluate_claim(claim, evidence, assessment, sources_by_id, self._config)
        return ClaimVerdict(
            claim_id=claim.claim_id,
            verdict=gate_verdict.verdict,
            evidence_ids=gate_verdict.evidence_ids,
            supporting_source_ids=gate_verdict.supporting_source_ids,
            contradiction_source_ids=gate_verdict.contradiction_source_ids,
            rationale=gate_verdict.rationale,
            confidence=gate_verdict.confidence,
            critical=claim.critical,
            checks_performed=gate_verdict.checks,
            verifier=self._verifier.__class__.__name__,
        ).validate()