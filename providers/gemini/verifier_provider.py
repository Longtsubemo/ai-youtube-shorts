"""Stage-3 Gemini verifier (the fact-check judge).

Implements the `Verifier` protocol from `agents/qa/fact_checker/verifier.py`
using the provider foundation in `providers/gemini/client.py`.

The verifier answers exactly one question, constrained to the supplied inputs:

    Does the SUPPLIED EVIDENCE support the exact wording of this claim?

Strict constraints (docs/GEMINI_PROVIDER_DESIGN.md §L, FACT_VERIFICATION.md):

- The model prompt is limited to {claim, supplied evidence, source metadata}.
  No world knowledge, no extra retrieval, no invented facts.
- The model may propose which SUPPLIED evidence ids support / contradict a
  claim, but it never names sources: `supporting_source_ids` and
  `contradiction_source_ids` are derived deterministically from the evidence
  -> source mapping (same rules as MockVerifier).
- Any evidence id the model references that is NOT in the supplied set is a
  schema error (fail closed - never let the model cite invisible evidence).
- `speculation_as_fact`, `discovery_only`, and `contradiction_present` are
  computed deterministically, mirroring MockVerifier.
- The final worldwide verdict is the deterministic gates' (gates.py), not the
  model's label; the model label is reconciled into the ClaimAssessment the
  gates already understand (DIRECT/PARTIAL/INSUFFICIENT/NONE/CONTRADICTORY).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Mapping, Sequence

from pydantic import BaseModel, Field

from agents.qa.fact_checker.models import Evidence
from agents.qa.fact_checker.verifier import (
    CONTRADICTORY,
    DIRECT,
    INSUFFICIENT,
    NONE,
    PARTIAL,
    ClaimAssessment,
    Verifier,
)
from agents.research.models import FactClaim, Source
from providers.gemini.client import GeminiClient
from providers.gemini.errors import GeminiSchemaValidationError

PROMPT_VERSION = "fact_check/verifier-v1"
DEFAULT_PROMPT_PATH = Path("prompts") / "fact_check" / "verifier-v1.txt"
GENERATED_BY = "gemini_verifier_v1"

_SYSTEM_PROMPT = (
    "You are a fact-checking judge in a YouTube Shorts explainer pipeline. "
    "Your ONLY inputs are the claim and the supplied evidence below. Never use "
    "your own knowledge, never invent evidence, and never name sources - cite "
    "only the supplied evidence ids. Return JSON only."
)

_SUPPORT_LEVELS = ("direct", "partial", "contradictory", "insufficient", "none")


class _VerifierSchema(BaseModel):
    """The model's claim-vs-evidence judgement, validated before use."""

    claim_id: str = Field(min_length=1)
    support_level: Literal[
        "direct",
        "partial",
        "contradictory",
        "insufficient",
        "none",
    ] = "none"
    stronger_than_evidence: bool = False
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradiction_evidence_ids: list[str] = Field(default_factory=list)
    insufficient_evidence_ids: list[str] = Field(default_factory=list)


# Convenience re-export for tests.
VerifierSchemaAdapter = _VerifierSchema


class GeminiVerifier:
    """Claim judge backed by Gemini structured output over supplied evidence."""

    provider_name = "gemini"

    def __init__(
        self,
        client: GeminiClient,
        *,
        prompt_path: str | Path = DEFAULT_PROMPT_PATH,
    ) -> None:
        self._client = client
        self._prompt_path = Path(prompt_path)

    # -- metadata --------------------------------------------------------- #

    @property
    def model(self) -> str:
        return self._client.model

    @property
    def prompt_version(self) -> str:
        return PROMPT_VERSION

    @property
    def produced_by(self) -> dict:
        """Header provenance stamped by the fact-check envelope."""
        return {
            "prompt_version": self.prompt_version,
            "provider": self.provider_name,
            "model": self.model,
        }

    # -- protocol --------------------------------------------------------- #

    def verify(
        self,
        claim: FactClaim,
        evidence: Sequence[Evidence],
        sources_by_id: Mapping[str, Source],
    ) -> ClaimAssessment:
        prompt = self._render_prompt(claim, evidence, sources_by_id)
        response = self._client.generate_structured(
            prompt,
            _VerifierSchema,
            system_prompt=_SYSTEM_PROMPT,
        )

        supplied = tuple(sorted({item.evidence_id for item in evidence}))
        supplied_set = set(supplied)
        referenced = (
            set(response.supporting_evidence_ids)
            | set(response.contradiction_evidence_ids)
            | set(response.insufficient_evidence_ids)
        )
        unknown = sorted(referenced - supplied_set)
        if unknown:
            raise GeminiSchemaValidationError(
                "verifier referenced evidence ids outside the supplied set: "
                + ", ".join(unknown)
            )
        if response.claim_id != claim.claim_id:
            raise GeminiSchemaValidationError(
                f"verifier returned claim_id {response.claim_id!r} for claim {claim.claim_id!r}"
            )

        return self._assess(claim, evidence, sources_by_id, response, supplied)

    # -- internals --------------------------------------------------------- #

    def _assess(
        self,
        claim: FactClaim,
        evidence: Sequence[Evidence],
        sources_by_id: Mapping[str, Source],
        response,
        supplied: tuple[str, ...],
    ) -> ClaimAssessment:
        """Reconcile the model judgement into a MockVerifier-shaped assessment.

        Source-level fields are deterministic; the model only proposes which
        supplied evidence ids relate in what way.
        """
        supporting = set(response.supporting_evidence_ids)
        contra = set(response.contradiction_evidence_ids)
        insufficient = set(response.insufficient_evidence_ids)

        supporting_ids: list[str] = []
        supporting_sources: list[str] = []
        contradiction_sources: list[str] = []
        insufficient_sources: list[str] = []

        for item in evidence:
            source = sources_by_id.get(item.source_id)
            if item.evidence_id in contra:
                if (
                    source is not None
                    and source.credibility_tier != "discovery"
                    and source.source_id not in contradiction_sources
                ):
                    contradiction_sources.append(source.source_id)
            if item.evidence_id in supporting:
                supporting_ids.append(item.evidence_id)
                if source is not None and source.source_id not in supporting_sources:
                    supporting_sources.append(source.source_id)
            if item.evidence_id in insufficient:
                if source is not None and source.source_id not in insufficient_sources:
                    insufficient_sources.append(source.source_id)

        contradiction_present = bool(contradiction_sources)

        if contradiction_present:
            support_level = CONTRADICTORY
        elif supporting_ids:
            # Evidence ids the model cited as supporting win over a conflicting
            # label ("direct" with no cited support, or "none"/"insufficient"
            # while citing support ids); the cited ids are the concrete artifact.
            support_level = response.support_level if response.support_level in (DIRECT, PARTIAL) else PARTIAL
        elif insufficient_sources or response.support_level == INSUFFICIENT:
            support_level = INSUFFICIENT
        else:
            support_level = NONE

        stronger_than_evidence = bool(response.stronger_than_evidence) and bool(supporting_ids)
        speculation_as_fact = claim.claim_type in ("speculation", "opinion")
        discovery_only = bool(supporting_sources) and all(
            sources_by_id[sid].credibility_tier == "discovery" for sid in supporting_sources
        )

        checks = (
            "verifier=GeminiVerifier",
            f"support_level={support_level}",
            f"model_support_level={response.support_level}",
            f"supporting_evidence={len(supporting_ids)}",
            f"contradiction_present={'yes' if contradiction_present else 'no'}",
            f"discovery_only={'yes' if discovery_only else 'no'}",
            f"speculation_as_fact={'yes' if speculation_as_fact else 'no'}",
            f"stronger_than_evidence={'yes' if stronger_than_evidence else 'no'}",
        )

        return ClaimAssessment(
            claim_id=claim.claim_id,
            support_level=support_level,
            stronger_than_evidence=stronger_than_evidence,
            speculation_as_fact=speculation_as_fact,
            discovery_only=discovery_only,
            contradiction_present=contradiction_present,
            evidence_ids=supplied,
            supporting_evidence_ids=tuple(sorted(supporting_ids)),
            supporting_source_ids=tuple(sorted(set(supporting_sources))),
            contradiction_source_ids=tuple(sorted(set(contradiction_sources))),
            insufficient_source_ids=tuple(sorted(set(insufficient_sources))),
            checks=checks,
        )

    def _render_prompt(
        self,
        claim: FactClaim,
        evidence: Sequence[Evidence],
        sources_by_id: Mapping[str, Source],
    ) -> str:
        path = Path(self._prompt_path)
        if not path.is_file():
            raise FileNotFoundError(f"prompt template not found: {path}")
        return (
            path.read_text(encoding="utf-8")
            .replace("{claim_id}", claim.claim_id)
            .replace("{claim_text}", claim.claim_text)
            .replace("{claim_type}", claim.claim_type)
            .replace("{critical}", "true" if claim.critical else "false")
            .replace("{evidence_items}", self._render_evidence(evidence))
            .replace("{sources_metadata}", self._render_sources(sources_by_id))
        )

    @staticmethod
    def _render_evidence(evidence: Sequence[Evidence]) -> str:
        lines: list[str] = []
        for index, item in enumerate(evidence, start=1):
            lines.append(f"[{index}] evidence_id: {item.evidence_id}")
            lines.append(f"    source_id: {item.source_id}")
            lines.append(f"    evidence_type: {item.evidence_type}")
            lines.append(f"    locator: {item.locator}")
            lines.append(f"    excerpt: \"{item.excerpt}\"")
        return "\n".join(lines)

    @staticmethod
    def _render_sources(sources_by_id: Mapping[str, Source]) -> str:
        lines: list[str] = []
        for source in sources_by_id.values():
            lines.append(f"- {source.source_id}: url={source.url} tier={source.credibility_tier}")
        return "\n".join(lines)