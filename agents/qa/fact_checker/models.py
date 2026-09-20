"""Versioned Stage-3 (fact-check) models: Evidence, ClaimVerdict, FactCheckResult.

Stage 3 is an *evidence verifier*, not a source of truth (docs/FACT_VERIFICATION.md).
Everything here evaluates whether the supplied evidence supports the exact wording
of each research claim. The key distinction enforced throughout:

    SOURCE  !=  EVIDENCE.

A source URL alone is not proof that the source supports a claim; the fact checker
operates on `Evidence` objects - content actually retrieved from a source.

Follows the project's artifact envelope (docs/DATA_CONTRACTS.md §1):
header (schema/version/created_at/run_id/stage/produced_by) + data payload.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agents.seo.models import Provenance
from pipeline.state import Stage

SCHEMA = "FactCheckResult"
SCHEMA_VERSION = 1
STAGE = Stage.FACT_CHECK.value

# Claim-level and package-level verdicts a fact check may issue.
VERDICTS = ("PASS", "REVIEW", "REJECT")

# How an evidence excerpt relates to the claim it is being checked against.
# `insufficient` means the source exists but does not settle the claim.
EVIDENCE_TYPES = (
    "direct_statement",
    "supporting_context",
    "contradictory_statement",
    "insufficient",
)

# The `produced_by` envelope helper records the verifier/provider that ran.
PRODUCED_BY = {
    "prompt_version": None,
    "provider": None,
    "model": "MockVerifier",
}


class FactCheckError(ValueError):
    """Raised when a fact-check artifact is missing, malformed, or inconsistent."""


def _require(value: Any, name: str) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise FactCheckError(f"missing required field: {name}")
    return value


def _check_confidence(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise FactCheckError(f"confidence must be a number, got {value!r}") from None
    if not 0.0 <= value <= 1.0:
        raise FactCheckError(f"confidence out of range [0,1]: {value}")
    return value


def _check_verdict(value: Any, name: str = "verdict") -> str:
    text = _require(value, name).strip().upper()
    if text not in VERDICTS:
        allowed = ", ".join(VERDICTS)
        raise FactCheckError(f"{name} must be one of {{{allowed}}}, got {value!r}")
    return text


def _check_iso(value: Any, name: str) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    candidate = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        datetime.fromisoformat(candidate)
    except ValueError:
        raise FactCheckError(f"{name} must be an ISO date/datetime, got {text!r}") from None
    return text


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def evidence_id_for(source_id, excerpt: str) -> str:
    """Stable evidence id derived from (source_id, excerpt); deterministic."""
    digest = hashlib.sha1(f"{source_id}:{excerpt.strip()}".encode("utf-8")).hexdigest()[:10]
    return f"ev_{digest}"


def result_id_for(run_id: str, package_id: str) -> str:
    """Stable FactCheckResult id per (run_id, package_id); deterministic."""
    digest = hashlib.sha1(f"{run_id or ''}:{package_id}".encode("utf-8")).hexdigest()[:12]
    return f"fc_{digest}"


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Evidence:
    """Content actually obtained from a source page.

    `source_id` must resolve to a Source inside the ResearchPackage being
    checked; the checker drops (and reports) evidence whose source it cannot
    resolve so a dangling citation can never back a PASS.

    `evidence_type` describes how the excerpt relates to the claim:
    direct_statement (asserts it), supporting_context (background that backs
    it), contradictory_statement (asserts against it), insufficient (does not
    settle it).
    """

    evidence_id: str
    source_id: str
    excerpt: str
    locator: str
    evidence_type: str
    retrieved_at: str = ""
    notes: str = ""

    def validate(self) -> "Evidence":
        _require(self.evidence_id, "evidence_id")
        _require(self.source_id, "source_id")
        _require(self.excerpt, "excerpt")
        _require(self.locator, "locator")
        if self.evidence_type not in EVIDENCE_TYPES:
            allowed = ", ".join(EVIDENCE_TYPES)
            raise FactCheckError(
                f"unknown evidence_type '{self.evidence_type}' (allowed: {allowed})"
            )
        _check_iso(self.retrieved_at, "retrieved_at")
        return self

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "source_id": self.source_id,
            "excerpt": self.excerpt,
            "locator": self.locator,
            "evidence_type": self.evidence_type,
            "retrieved_at": self.retrieved_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Evidence":
        if not isinstance(data, dict):
            raise FactCheckError(f"evidence must be a mapping, got {data!r}")
        evidence = cls(
            evidence_id=_require(data.get("evidence_id"), "evidence_id"),
            source_id=_require(data.get("source_id"), "source_id"),
            excerpt=_require(data.get("excerpt"), "excerpt"),
            locator=_require(data.get("locator"), "locator"),
            evidence_type=_require(data.get("evidence_type"), "evidence_type"),
            retrieved_at=_check_iso(data.get("retrieved_at"), "retrieved_at"),
            notes=data.get("notes") or "",
        )
        return evidence.validate()


# --------------------------------------------------------------------------- #
# Claim verdict
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ClaimVerdict:
    """One claim's fact-check outcome.

    `evidence_ids` are the evidence items the verifier actually examined;
    `supporting_source_ids` / `contradiction_source_ids` are the sources whose
    supplied evidence backed / contradicted the claim wording. The rationale
    explains the verdict using ONLY the supplied evidence and metadata - it is
    never allowed to introduce outside knowledge.
    """

    claim_id: str
    verdict: str
    evidence_ids: tuple[str, ...]
    supporting_source_ids: tuple[str, ...]
    contradiction_source_ids: tuple[str, ...]
    rationale: str
    confidence: float
    critical: bool
    checks_performed: tuple[str, ...] = ()
    verifier: str = ""

    def validate(self) -> "ClaimVerdict":
        _require(self.claim_id, "claim_id")
        _check_verdict(self.verdict)
        for evidence_id in self.evidence_ids:
            _require(evidence_id, "evidence_ids entry")
        for source_id in self.supporting_source_ids:
            _require(source_id, "supporting_source_ids entry")
        for source_id in self.contradiction_source_ids:
            _require(source_id, "contradiction_source_ids entry")
        _require(self.rationale, "rationale")
        _check_confidence(self.confidence)
        return self

    @property
    def is_approved(self) -> bool:
        return self.verdict == "PASS"

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "verdict": self.verdict,
            "evidence_ids": list(self.evidence_ids),
            "supporting_source_ids": list(self.supporting_source_ids),
            "contradiction_source_ids": list(self.contradiction_source_ids),
            "rationale": self.rationale,
            "confidence": self.confidence,
            "critical": self.critical,
            "checks_performed": list(self.checks_performed),
            "verifier": self.verifier,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClaimVerdict":
        if not isinstance(data, dict):
            raise FactCheckError(f"claim verdict must be a mapping, got {data!r}")
        verdict = cls(
            claim_id=_require(data.get("claim_id"), "claim_id"),
            verdict=_check_verdict(data.get("verdict")),
            evidence_ids=tuple(data.get("evidence_ids") or ()),
            supporting_source_ids=tuple(data.get("supporting_source_ids") or ()),
            contradiction_source_ids=tuple(data.get("contradiction_source_ids") or ()),
            rationale=_require(data.get("rationale"), "rationale"),
            confidence=data.get("confidence", 0.0),
            critical=bool(data.get("critical", False)),
            checks_performed=tuple(data.get("checks_performed") or ()),
            verifier=data.get("verifier") or "",
        )
        return verdict.validate()


# --------------------------------------------------------------------------- #
# Package-level result
# --------------------------------------------------------------------------- #


@dataclass
class FactCheckResult:
    """The Stage-3 artifact: claim verdicts + a package verdict + provenance.

    `approved_claims` is exactly what Stage 4 may consume: rejected and REVIEW
    claims are excluded and must never reach the script stage.
    """

    result_id: str
    topic_id: str
    topic_title: str
    package_id: str
    claim_results: tuple[ClaimVerdict, ...] = ()
    package_verdict: str = "REVIEW"
    critical_claim_failures: tuple[str, ...] = ()
    review_items: tuple[str, ...] = ()
    rejected_claims: tuple[str, ...] = ()
    approved_claims: tuple[str, ...] = ()
    evidence_issues: tuple[str, ...] = ()
    provenance: Provenance | None = None
    created_at: str = field(default_factory=_utc_now)
    schema: str = SCHEMA
    schema_version: int = SCHEMA_VERSION
    stage: str = STAGE
    run_id: str = ""
    produced_by: dict = field(default_factory=lambda: dict(PRODUCED_BY))

    def claim_verdict(self, claim_id: str) -> ClaimVerdict | None:
        return next((result for result in self.claim_results if result.claim_id == claim_id), None)

    def validate(self) -> "FactCheckResult":
        _require(self.result_id, "result_id")
        _require(self.topic_id, "topic_id")
        _require(self.topic_title, "topic_title")
        _require(self.package_id, "package_id")
        _check_verdict(self.package_verdict, "package_verdict")
        claim_ids = [result.claim_id for result in self.claim_results]
        if len(set(claim_ids)) != len(claim_ids):
            raise FactCheckError("duplicate claim_id in claim_results")
        for claim_id in self.review_items + self.rejected_claims:
            if claim_id not in claim_ids:
                raise FactCheckError(
                    f"review/rejected claim {claim_id} missing from claim_results"
                )
        return self

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "run_id": self.run_id,
            "stage": self.stage,
            "produced_by": dict(self.produced_by),
            "data": {
                "result_id": self.result_id,
                "topic_id": self.topic_id,
                "topic_title": self.topic_title,
                "package_id": self.package_id,
                "package_verdict": self.package_verdict,
                "claim_results": [result.to_dict() for result in self.claim_results],
                "critical_claim_failures": list(self.critical_claim_failures),
                "review_items": list(self.review_items),
                "rejected_claims": list(self.rejected_claims),
                "approved_claims": list(self.approved_claims),
                "evidence_issues": list(self.evidence_issues),
                "provenance": self.provenance.to_dict() if self.provenance else None,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "FactCheckResult":
        header = dict(data)
        payload = header.pop("data")
        result = cls(
            result_id=_require(payload.get("result_id"), "result_id"),
            topic_id=_require(payload.get("topic_id"), "topic_id"),
            topic_title=_require(payload.get("topic_title"), "topic_title"),
            package_id=_require(payload.get("package_id"), "package_id"),
            claim_results=tuple(ClaimVerdict.from_dict(item) for item in payload.get("claim_results") or []),
            package_verdict=_check_verdict(payload.get("package_verdict"), "package_verdict"),
            critical_claim_failures=tuple(payload.get("critical_claim_failures") or ()),
            review_items=tuple(payload.get("review_items") or ()),
            rejected_claims=tuple(payload.get("rejected_claims") or ()),
            approved_claims=tuple(payload.get("approved_claims") or ()),
            evidence_issues=tuple(payload.get("evidence_issues") or ()),
            provenance=Provenance.from_dict(payload["provenance"]) if payload.get("provenance") else None,
            schema=header.get("schema", SCHEMA),
            schema_version=header.get("schema_version", SCHEMA_VERSION),
            created_at=header["created_at"],
            stage=header.get("stage", STAGE),
            run_id=header.get("run_id", ""),
            produced_by=header.get("produced_by") or dict(PRODUCED_BY),
        )
        return result.validate()

    @classmethod
    def from_json(cls, text: str) -> "FactCheckResult":
        return cls.from_dict(json.loads(text))