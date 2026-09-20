"""Versioned research models: Source, FactClaim, ClaimDate, ResearchPackage.

The research stage changes unstructured search results into *structured,
evidence-backed claims* that later stages (fact check, script) are allowed
to consume. Nothing here verifies facts: verification starts and stays
"provisional"; settling it is the fact-check stage's job.

Follows the project's artifact envelope (docs/DATA_CONTRACTS.md §1):
header (schema/version/created_at/run_id/stage/produced_by) + data payload.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from agents.seo.models import Provenance, utc_now
from pipeline.state import Stage

SCHEMA = "ResearchPackage"
SCHEMA_VERSION = 1
STAGE = Stage.RESEARCH.value

# Evidence-gathering statuses the research stage can produce.
RESEARCH_STATUSES = ("provisional", "insufficient_evidence", "ready_for_fact_check")

# Claim classification the research agent must distinguish.
CLAIM_TYPES = ("fact", "inference", "explanation", "speculation", "opinion")

# Source credibility tiers (docs/RESEARCH_PIPELINE.md §4).
SOURCE_TIERS = ("primary", "high_quality_secondary", "discovery")

# Date kinds a claim may cite. Independent of a source's publication date:
# an article published in 2026 about a discovery made in 2019 is not fresh.
CLAIM_DATE_KINDS = ("publication_date", "discovery_date", "announcement_date", "event_date")

# Verification values the research stage is allowed to stamp. Values that
# imply final verification are NOT in this set - the agent must never label
# a claim as fully verified; that is exclusive to the fact-check stage.
PROVISIONAL = "provisional"
VERIFICATION_STATUSES = ("provisional", "pending", "single_source", "multiple_sources", "disputed", "unsupported")


class ResearchPackageError(ValueError):
    """Raised when research data is missing, malformed, or internally inconsistent."""


def _require(value: Any, name: str) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ResearchPackageError(f"missing required field: {name}")
    return value


def _check_confidence(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ResearchPackageError(f"confidence must be a number, got {value!r}") from None
    if not 0.0 <= value <= 1.0:
        raise ResearchPackageError(f"confidence out of range [0,1]: {value}")
    return value


def _check_iso(value: Any, name: str) -> str:
    """Normalize an ISO date/datetime to a non-empty string, or '' when None."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    candidate = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        datetime.fromisoformat(candidate)
    except ValueError:
        raise ResearchPackageError(f"{name} must be an ISO date/datetime, got {text!r}") from None
    return text


def _check_url(value: Any, name: str = "url") -> str:
    text = _require(value, name).strip()
    parsed = urlparse(text)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ResearchPackageError(f"{name} must be a valid http(s) URL, got {value!r}")
    return text


def source_id_for(url: str) -> str:
    """Stable source id derived from the source URL (deterministic across runs)."""
    return "src_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]


def claim_id_for(claim_text: str) -> str:
    """Stable claim id derived from claim text (deterministic across runs)."""
    return "cl_" + hashlib.sha1(claim_text.strip().encode("utf-8")).hexdigest()[:10]


@dataclass(frozen=True)
class ClaimDate:
    """One relevant date attached to a claim, with an explicit date kind.

    `date_kind` distinguishes what the date means (discovery vs publication
    vs announcement vs event); an article's publication_date is a property
    of the Source, never silently promoted to a discovery_date.
    """

    date_kind: str
    date: str

    def to_dict(self) -> dict:
        return {"date_kind": self.date_kind, "date": self.date}

    @classmethod
    def from_dict(cls, data: dict) -> "ClaimDate":
        if not isinstance(data, dict):
            raise ResearchPackageError(f"claim date must be a mapping, got {data!r}")
        date_kind = _require(data.get("date_kind"), "date_kind")
        if date_kind not in CLAIM_DATE_KINDS:
            allowed = ", ".join(CLAIM_DATE_KINDS)
            raise ResearchPackageError(f"unknown date_kind '{date_kind}' (allowed: {allowed})")
        date = _check_iso(_require(data.get("date"), "date"), "date")
        return cls(date_kind=date_kind, date=date)


@dataclass(frozen=True)
class Source:
    """A single research source: origin metadata + where it sits in evidence.

    `credibility_tier` is the research stage's own assessment of the
    source's weight (primary / high_quality_secondary / discovery); it does
    not claim the source has been verified.
    """

    source_id: str
    title: str
    url: str
    publisher: str
    source_type: str
    publication_date: str
    credibility_tier: str
    retrieved_at: str = ""
    notes: str = ""

    @property
    def is_non_discovery(self) -> bool:
        return self.credibility_tier != "discovery"

    def validate(self) -> "Source":
        _require(self.source_id, "source_id")
        _require(self.title, "title")
        _check_url(self.url)
        _require(self.publisher, "publisher")
        _require(self.source_type, "source_type")
        _check_iso(self.publication_date, "publication_date")
        if self.credibility_tier not in SOURCE_TIERS:
            allowed = ", ".join(SOURCE_TIERS)
            raise ResearchPackageError(f"unknown credibility_tier '{self.credibility_tier}' (allowed: {allowed})")
        _check_iso(self.retrieved_at, "retrieved_at")
        return self

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "title": self.title,
            "url": self.url,
            "publisher": self.publisher,
            "source_type": self.source_type,
            "publication_date": self.publication_date,
            "credibility_tier": self.credibility_tier,
            "retrieved_at": self.retrieved_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Source":
        if not isinstance(data, dict):
            raise ResearchPackageError(f"source must be a mapping, got {data!r}")
        source = cls(
            source_id=_require(data.get("source_id"), "source_id"),
            title=_require(data.get("title"), "title"),
            url=_require(data.get("url"), "url"),
            publisher=_require(data.get("publisher"), "publisher"),
            source_type=_require(data.get("source_type"), "source_type"),
            publication_date=_check_iso(data.get("publication_date"), "publication_date"),
            credibility_tier=_require(data.get("credibility_tier"), "credibility_tier"),
            retrieved_at=_check_iso(data.get("retrieved_at"), "retrieved_at"),
            notes=data.get("notes") or "",
        )
        return source.validate()


@dataclass(frozen=True)
class FactClaim:
    """One evidence-backed claim extracted during research.

    `verification_status` is provisional by design; the research agent is
    not allowed to mark a claim as verified (that is the fact-check stage's
    role). `confidence` is a deterministic heuristic based on the backing
    source tiers.
    """

    claim_id: str
    claim_text: str
    claim_type: str
    source_ids: tuple[str, ...] = ()
    confidence: float = 0.5
    verification_status: str = PROVISIONAL
    dates: tuple[ClaimDate, ...] = ()
    notes: str = ""
    critical: bool = False

    def validate(self) -> "FactClaim":
        _require(self.claim_id, "claim_id")
        _require(self.claim_text, "claim_text")
        _require(self.claim_type, "claim_type")
        if self.claim_type not in CLAIM_TYPES:
            allowed = ", ".join(CLAIM_TYPES)
            raise ResearchPackageError(f"unknown claim_type '{self.claim_type}' (allowed: {allowed})")
        _check_confidence(self.confidence)
        if self.verification_status not in VERIFICATION_STATUSES:
            allowed = ", ".join(VERIFICATION_STATUSES)
            raise ResearchPackageError(
                f"verification_status '{self.verification_status}' not allowed at research stage "
                f"(allowed: {allowed})"
            )
        for source_id in self.source_ids:
            _require(source_id, "source_ids entry")
        return self

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "claim_type": self.claim_type,
            "source_ids": list(self.source_ids),
            "confidence": self.confidence,
            "verification_status": self.verification_status,
            "dates": [date.to_dict() for date in self.dates],
            "notes": self.notes,
            "critical": self.critical,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FactClaim":
        if not isinstance(data, dict):
            raise ResearchPackageError(f"claim must be a mapping, got {data!r}")
        dates = _require(data.get("dates"), "dates")
        if not isinstance(dates, (list, tuple)):
            raise ResearchPackageError("claim dates must be a list")
        claim = cls(
            claim_id=_require(data.get("claim_id"), "claim_id"),
            claim_text=_require(data.get("claim_text"), "claim_text"),
            claim_type=_require(data.get("claim_type"), "claim_type"),
            source_ids=tuple(data.get("source_ids") or ()),
            confidence=data.get("confidence", 0.5),
            verification_status=data.get("verification_status") or PROVISIONAL,
            dates=tuple(ClaimDate.from_dict(item) for item in dates),
            notes=data.get("notes") or "",
            critical=bool(data.get("critical", False)),
        )
        return claim.validate()


@dataclass
class ResearchPackage:
    """The research stage's artifact: claims, their sources, and provenance.

    Mirrors the artifact envelope used by Stage 1 (TopicSelection): the
    header repeats schema/version/timestamps/run, and the evidence lives in
    `data`. `provenance` describes which agent/provider produced the package.
    """

    package_id: str
    topic_id: str
    topic_title: str
    research_status: str
    claims: tuple[FactClaim, ...] = ()
    sources: tuple[Source, ...] = ()
    research_notes: tuple[str, ...] = ()
    provenance: Provenance | None = None
    created_at: str = field(default_factory=utc_now)
    schema: str = SCHEMA
    schema_version: int = SCHEMA_VERSION
    stage: str = STAGE
    run_id: str = ""
    produced_by: dict = field(
        default_factory=lambda: {"prompt_version": None, "provider": None, "model": None}
    )

    @classmethod
    def build(
        cls,
        *,
        topic_id: str,
        topic_title: str,
        research_status: str,
        claims: list[FactClaim] | tuple[FactClaim, ...],
        sources: list[Source] | tuple[Source, ...],
        research_notes: list[str] | tuple[str, ...] = (),
        package_id: str | None = None,
        provenance: Provenance | None = None,
        created_at: str | None = None,
        run_id: str = "",
        produced_by: dict | None = None,
    ) -> "ResearchPackage":
        return cls(
            package_id=package_id or f"rp_{uuid.uuid4().hex[:12]}",
            topic_id=topic_id,
            topic_title=topic_title,
            research_status=research_status,
            claims=tuple(claims),
            sources=tuple(sources),
            research_notes=tuple(research_notes),
            provenance=provenance,
            created_at=created_at or utc_now(),
            run_id=run_id,
            produced_by=produced_by or {"prompt_version": None, "provider": None, "model": None},
        )

    def source_by_id(self, source_id: str) -> Source | None:
        return next((source for source in self.sources if source.source_id == source_id), None)

    def claim_by_id(self, claim_id: str) -> FactClaim | None:
        return next((claim for claim in self.claims if claim.claim_id == claim_id), None)

    def claims_for_source(self, source_id: str) -> list[FactClaim]:
        return [claim for claim in self.claims if source_id in claim.source_ids]

    def validate(self, traceability: bool = True) -> "ResearchPackage":
        _require(self.package_id, "package_id")
        _require(self.topic_id, "topic_id")
        _require(self.topic_title, "topic_title")
        if self.research_status not in RESEARCH_STATUSES:
            allowed = ", ".join(RESEARCH_STATUSES)
            raise ResearchPackageError(f"unknown research_status '{self.research_status}' (allowed: {allowed})")
        source_ids = [source.source_id for source in self.sources]
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(set(source_ids)) != len(source_ids):
            raise ResearchPackageError("duplicate source_id in package")
        if len(set(claim_ids)) != len(claim_ids):
            raise ResearchPackageError("duplicate claim_id in package")
        if traceability:
            for claim in self.claims:
                missing = [source_id for source_id in claim.source_ids if source_id not in source_ids]
                if missing:
                    raise ResearchPackageError(
                        f"claim {claim.claim_id} references unknown source(s): {', '.join(missing)}"
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
                "package_id": self.package_id,
                "topic_id": self.topic_id,
                "topic_title": self.topic_title,
                "research_status": self.research_status,
                "claims": [claim.to_dict() for claim in self.claims],
                "sources": [source.to_dict() for source in self.sources],
                "research_notes": list(self.research_notes),
                "provenance": self.provenance.to_dict() if self.provenance else None,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict, traceability: bool = True) -> "ResearchPackage":
        header = dict(data)
        payload = header.pop("data")
        package = cls(
            package_id=_require(payload.get("package_id"), "package_id"),
            topic_id=_require(payload.get("topic_id"), "topic_id"),
            topic_title=_require(payload.get("topic_title"), "topic_title"),
            research_status=_require(payload.get("research_status"), "research_status"),
            claims=tuple(FactClaim.from_dict(item) for item in (payload.get("claims") or [])),
            sources=tuple(Source.from_dict(item) for item in (payload.get("sources") or [])),
            research_notes=tuple(payload.get("research_notes") or ()),
            provenance=Provenance.from_dict(payload["provenance"]) if payload.get("provenance") else None,
            schema=header.get("schema", SCHEMA),
            schema_version=header.get("schema_version", SCHEMA_VERSION),
            created_at=header["created_at"],
            stage=header.get("stage", STAGE),
            run_id=header.get("run_id", ""),
            produced_by=header.get("produced_by") or {"prompt_version": None, "provider": None, "model": None},
        )
        return package.validate(traceability=traceability)

    @classmethod
    def from_json(cls, text: str, traceability: bool = True) -> "ResearchPackage":
        return cls.from_dict(json.loads(text), traceability=traceability)