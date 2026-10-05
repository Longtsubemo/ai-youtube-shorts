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
    NO_EVIDENCE_ALL_REJECTED,
    NO_EVIDENCE_PROVIDER_EMPTY,
    NO_EVIDENCE_PROVIDER_FAILED,
    NO_EVIDENCE_UNKNOWN,
    ClaimEvidenceAudit,
    ClaimVerdict,
    Evidence,
    EvidenceAuditRecord,
    FactCheckError,
    FactCheckResult,
    SourceEvidenceAudit,
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
        evidence_index, issues, evidence_reasons = self._collect_evidence(package, source_ids)

        claim_results = []
        for claim in package.claims:
            verdict, recovery_issues = self._check_claim(
                claim, evidence_index, package, evidence_reasons
            )
            claim_results.append(verdict)
            issues.extend(recovery_issues)
        claim_results = tuple(claim_results)

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

    def _collect_evidence(
        self, package: ResearchPackage, source_ids: set[str]
    ) -> tuple[dict[str, list[Evidence]], list[str], dict[str, str]]:
        """Fetch + validate evidence; drop anything that cannot be traced.

        Returns the per-source index, the issue list, and - for audit only - a
        map of `source_id` to why it produced no usable evidence. The reason map
        is derived purely from what this method already observed, so it can only
        name `provider_failed`, `provider_returned_empty` or
        `all_records_rejected`; a source with no recorded reason falls back to
        `unknown` rather than a guess. Nothing here decides what is accepted.
        """
        index: dict[str, list[Evidence]] = {}
        issues: list[str] = []
        reasons: dict[str, str] = {}
        for source in package.sources:
            sid = source.source_id
            returned = 0
            accepted = 0
            try:
                entries = self._provider.get_evidence(source)
            except Exception as exc:  # provider misbehaves -> clean issue, not a crash
                issues.append(f"evidence provider failed for {sid}: {exc}")
                entries = []
                reasons[sid] = NO_EVIDENCE_PROVIDER_FAILED
            for evidence in entries:
                returned += 1
                try:
                    evidence.validate()
                except FactCheckError as exc:
                    issues.append(f"malformed evidence for {sid}: {exc}")
                    continue
                if evidence.source_id not in source_ids:
                    issues.append(
                        f"evidence {evidence.evidence_id} references unknown source "
                        f"{evidence.source_id!r}"
                    )
                    continue
                index.setdefault(sid, []).append(evidence)
                accepted += 1
            if sid not in reasons:
                if returned == 0:
                    reasons[sid] = NO_EVIDENCE_PROVIDER_EMPTY
                elif accepted == 0:
                    reasons[sid] = NO_EVIDENCE_ALL_REJECTED
        return index, issues, reasons

    def _recover_evidence(
        self,
        package: ResearchPackage,
        claim,
        candidate_source_ids: Sequence[str],
        claim_evidence: dict[str, list[Evidence]],
        sources_by_id: Mapping[str, Source],
    ) -> tuple[list[Evidence], list[str]]:
        """Deterministic recall backstop for one claim.

        Called only for sources the verifier did *not* credit on its first
        assessment. That distinction is the whole point of this backstop: the
        real failure was a source that supplied plenty of evidence but none of
        the sentence actually supporting the claim, so a rule like "skip any
        source that already has evidence" would never have fired.

        Deliberately conservative scope:

        - Only sources bound to `claim` and present in the package are
          considered; recovery can never introduce a source.
        - Only sources the verifier declined to credit are considered. Sources
          it did credit are left completely untouched, so verifier selection
          over existing evidence is never second-guessed.
        - Discovery-tier sources are offered the backstop too. They still cannot
          count toward corroboration, because that decision belongs to the
          unchanged gate, but excluding them from recall meant a claim bound
          only to discovery sources could be REJECTed for want of evidence while
          the supporting sentence sat unread in the page that was already
          retrieved. Recovered discovery evidence can therefore move such a
          claim to REVIEW, never to PASS.
        - Evidence is only ever *added* to the verifier's input. Support is
          decided by the ordinary verifier on a second pass and by the unchanged
          gates, so recovery cannot award a source any credit on its own.
        - An evidence id already present is dropped, so a sentence the extractor
          did report is never counted twice.

        Returns the recovered evidence plus any issues; never raises.
        """
        recover = getattr(self._provider, "recover_evidence", None)
        if not callable(recover):
            return [], []

        recovered: list[Evidence] = []
        issues: list[str] = []
        package_source_ids = {source.source_id for source in package.sources}
        existing_ids = {
            evidence.evidence_id
            for entries in claim_evidence.values()
            for evidence in entries
        }

        for source_id in candidate_source_ids:
            source = sources_by_id.get(source_id)
            if source is None or source_id not in package_source_ids:
                continue
            # Tier is deliberately not filtered here. Discovery-tier sources are
            # offered the backstop so that a discovery-only claim can reach
            # REVIEW on real evidence instead of being REJECTed for lack of it.
            # Whether that evidence counts toward anything is the gate's call.
            try:
                entries = list(recover(source, claim))
            except Exception as exc:  # recovery is best-effort, never fatal
                issues.append(f"evidence recovery failed for {source_id}: {exc}")
                continue
            for evidence in entries:
                try:
                    evidence.validate()
                except FactCheckError as exc:
                    issues.append(f"malformed recovered evidence for {source_id}: {exc}")
                    continue
                if evidence.source_id != source_id:
                    # Recovery may never introduce a source.
                    issues.append(
                        f"recovered evidence {evidence.evidence_id} bound to "
                        f"{evidence.source_id!r} instead of {source_id!r}; dropped"
                    )
                    continue
                if evidence.evidence_id in existing_ids:
                    continue  # the extractor already reported this sentence
                existing_ids.add(evidence.evidence_id)
                recovered.append(evidence)
        return recovered, issues

    def _check_claim(
        self,
        claim,
        evidence_index: Mapping[str, Sequence[Evidence]],
        package: ResearchPackage,
        evidence_reasons: Mapping[str, str],
    ) -> tuple[ClaimVerdict, list[str]]:
        """Verify one claim and mint its verdict.

        Evidence for this claim is assembled into a claim-local mapping, never
        into the shared index: recovered evidence is claim-specific (a sentence
        that restates claim A says nothing about claim B), so letting it leak
        between claims would both skip a legitimate recovery and let one
        claim's text be judged as another's evidence.

        The backstop runs as a second pass. The first assessment decides which
        bound, eligible sources were left uncredited; only those are offered to
        the provider, and any recovered evidence goes back through the ordinary
        verifier rather than being credited directly.

        `evidence_reasons` is audit input only: it supplies the reason attached
        to a bound source that produced no evidence. It is never consulted
        before the verifier runs, so it cannot influence a verdict.
        """
        sources_by_id = {source.source_id: source for source in package.sources}
        claim_evidence: dict[str, list[Evidence]] = {
            source_id: list(evidence_index.get(source_id, ()))
            for source_id in claim.source_ids
        }

        def assemble() -> list[Evidence]:
            return [
                item
                for source_id in claim.source_ids
                for item in claim_evidence.get(source_id, ())
            ]

        assessment = self._verifier.verify(claim, assemble(), sources_by_id)
        recovery_issues: list[str] = []
        recovered_ids: set[str] = set()

        credited = set(assessment.supporting_source_ids)
        # Tier is deliberately not filtered here. The backstop exists to recover
        # text the claim-blind extractor missed, and discovery-tier sources are
        # bound to claims too. Excluding them let a discovery-only claim be
        # REJECTed while its supporting sentence was present in the retrieved
        # page. Discovery evidence still cannot satisfy corroboration: Gate 10
        # holds such a claim at REVIEW and the critical gate counts only
        # non-discovery tiers, both unchanged.
        eligible_uncredited = [
            source_id
            for source_id in claim.source_ids
            if source_id not in credited
            and sources_by_id.get(source_id) is not None
        ]

        if eligible_uncredited:
            recovered, recovery_issues = self._recover_evidence(
                package, claim, eligible_uncredited, claim_evidence, sources_by_id
            )
            if recovered:
                for evidence in recovered:
                    claim_evidence.setdefault(evidence.source_id, []).append(evidence)
                # Recovered records are marked for the audit only. Recovery
                # already drops any id the extractor reported, so every id here
                # is genuinely new to this claim's evidence set.
                recovered_ids.update(evidence.evidence_id for evidence in recovered)
                # Re-run the ordinary verifier over the fuller evidence set. The
                # gates are untouched, so recovered text can still be declined.
                assessment = self._verifier.verify(claim, assemble(), sources_by_id)

        evidence = assemble()
        gate_verdict = evaluate_claim(claim, evidence, assessment, sources_by_id, self._config)
        supporting = set(gate_verdict.supporting_source_ids)
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
            evidence_audit=self._build_audit(
                claim, evidence, gate_verdict.supporting_source_ids,
                eligible_uncredited, recovered_ids, evidence_reasons,
            ),
        ).validate(), recovery_issues

    def _build_audit(
        self,
        claim,
        evidence: Sequence[Evidence],
        supporting_source_ids: Sequence[str],
        recovery_candidates: Sequence[str],
        recovered_ids: set[str],
        evidence_reasons: Mapping[str, str],
    ) -> ClaimEvidenceAudit:
        """Describe the claim-local evidence chain. Audit only.

        `evidence_count` is a count of Evidence RECORDS and is never used as a
        corroboration count; corroboration stays the sole responsibility of
        `supporting_source_ids` and the unchanged gate. `credited` on a record
        means the verifier credited that record's SOURCE for this claim, since
        per-excerpt credit is not exposed by the verifier.
        """
        supporting = set(supporting_source_ids)
        recovery_offered = set(recovery_candidates)

        records = tuple(
            EvidenceAuditRecord.from_evidence(
                item,
                origin="recovery" if item.evidence_id in recovered_ids else "extraction",
                credited=item.source_id in supporting,
            )
            for item in evidence
        )

        recovered_per_source: dict[str, int] = {}
        for item in records:
            if item.origin == "recovery":
                recovered_per_source[item.source_id] = (
                    recovered_per_source.get(item.source_id, 0) + 1
                )

        sources = tuple(
            SourceEvidenceAudit(
                source_id=source_id,
                evidence_count=sum(1 for item in records if item.source_id == source_id),
                status=(
                    "no_evidence"
                    if not any(item.source_id == source_id for item in records)
                    else ("credited" if source_id in supporting else "not_credited")
                ),
                reason=(
                    ""
                    if any(item.source_id == source_id for item in records)
                    else evidence_reasons.get(source_id, NO_EVIDENCE_UNKNOWN)
                ),
                recovery_attempted=source_id in recovery_offered,
                recovery_count=recovered_per_source.get(source_id, 0),
            )
            for source_id in claim.source_ids
        )

        return ClaimEvidenceAudit(
            claim_id=claim.claim_id, records=records, sources=sources
        ).validate()