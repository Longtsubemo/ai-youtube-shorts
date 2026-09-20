"""ResearchAgent: TopicSelection -> ResearchPackage.

Turns a selected topic into a structured, evidence-backed ResearchPackage:

    1. Receives a TopicSelection.
    2. Requests source candidates from a ResearchProvider.
    3. Validates/normalizes source data (rejects malformed data cleanly).
    4. Extracts structured claims and associates each with its source ids.
    5. Records relevant dates (date-kind aware).
    6. Classifies claims (fact/inference/explanation/speculation/opinion).
    7. Assigns provisional confidence from source credibility tiers.
    8. Produces a ResearchPackage with a provisional research_status.

The agent NEVER performs final fact verification: every claim keeps the
provisional verification status, and settling it is deferred to the
fact-check stage.
"""

from __future__ import annotations

from typing import Mapping

from agents.research.classifier import classify_claim, compute_confidence
from agents.research.models import (
    PROVISIONAL,
    FactClaim,
    Provenance,
    ResearchPackage,
    ResearchPackageError,
    Source,
    claim_id_for,
    source_id_for,
    utc_now,
)
from agents.research.provider import ClaimItem, ResearchProvider, TopicDescriptor, SourceCandidate
from agents.seo.result import TopicSelection


def evaluate_research_status(claims: tuple[FactClaim, ...], sources: tuple[Source, ...]) -> str:
    """Provisional evidence-gathering status for a package.

    - "insufficient_evidence"  : no claims or no sources at all.
    - "provisional"            : evidence exists but nothing is backed by a
                                 non-discovery source (leads only).
    - "ready_for_fact_check"   : at least one fact/explanation/inference claim
                                 is backed by a primary or high-quality
                                 secondary source.
    """
    if not claims or not sources:
        return "insufficient_evidence"
    sources_by_id = {source.source_id: source for source in sources}
    core_backed = any(
        claim.claim_type in ("fact", "explanation", "inference")
        and any(
            (source := sources_by_id.get(source_id)) is not None and source.is_non_discovery
            for source_id in claim.source_ids
        )
        for claim in claims
    )
    return "ready_for_fact_check" if core_backed else "provisional"


# ------------------------------------------------------------------ #

def _package_id(run_id: str, topic_id: str) -> str:
    """Deterministic package id per (run, topic), stable across re-runs."""
    import hashlib

    digest = hashlib.sha1(f"{run_id}:{topic_id}".encode("utf-8")).hexdigest()[:12]
    return f"rp_{digest}"


class ResearchAgent:
    def __init__(
        self,
        provider: ResearchProvider,
        run_id: str = "",
        created_at: str | None = None,
    ) -> None:
        self._provider = provider
        self._run_id = run_id
        self._created_at = created_at

    def research(self, selection: TopicSelection | Mapping) -> ResearchPackage:
        topic_id, topic_title, descriptor = self._make_topic(selection)
        candidates = self._provider.search(descriptor)
        sources = self._build_sources(candidates)
        claims = self._build_claims(candidates, sources)
        research_status = evaluate_research_status(claims, sources)
        research_notes = self._compose_notes(claims, sources, research_status)

        provenance = Provenance(
            generated_by="research_agent_v1",
            source=f"{self._provider.__class__.__name__}",
            created_at=self._created_at or utc_now(),
        )
        return ResearchPackage.build(
            topic_id=topic_id,
            topic_title=topic_title,
            research_status=research_status,
            claims=claims,
            sources=sources,
            research_notes=research_notes,
            package_id=_package_id(self._run_id or "", topic_id),
            provenance=provenance,
            created_at=self._created_at,
            run_id=self._run_id or getattr(selection, "run_id", "") or "",
        )

    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_topic(selection: TopicSelection | Mapping) -> tuple[str, str, TopicDescriptor]:
        """Extract topic identity + provider search descriptor from a selection."""
        if hasattr(selection, "to_dict") and not isinstance(selection, Mapping):
            data = selection.to_dict()["data"]
            selected_topic = data["selected_topic"]
            return (
                str(data["selected_topic_id"]),
                str(selected_topic["title"]),
                TopicDescriptor(
                    topic_id=str(data["selected_topic_id"]),
                    title=str(selected_topic["title"]),
                    description=str(selected_topic.get("description") or ""),
                    pillar=str(selected_topic.get("pillar") or ""),
                    topic_type=str(selected_topic.get("topic_type") or ""),
                ),
            )
        payload = selection.get("data") if "data" in selection else selection
        selected_topic = payload["selected_topic"]
        return (
            str(payload["selected_topic_id"]),
            str(selected_topic["title"]),
            TopicDescriptor(
                topic_id=str(payload["selected_topic_id"]),
                title=str(selected_topic["title"]),
                description=str(selected_topic.get("description") or ""),
                pillar=str(selected_topic.get("pillar") or ""),
                topic_type=str(selected_topic.get("topic_type") or ""),
            ),
        )

    def _build_sources(self, candidates: list[SourceCandidate]) -> tuple[Source, ...]:
        sources: list[Source] = []
        for index, candidate in enumerate(candidates):
            try:
                sources.append(self._source_from_candidate(candidate))
            except ResearchPackageError as exc:
                raise ResearchPackageError(f"malformed source at index {index}: {exc}") from None
        return tuple(sources)

    def _source_from_candidate(self, candidate: SourceCandidate) -> Source:
        source = Source(
            source_id=source_id_for(candidate.url),
            title=candidate.title,
            url=candidate.url,
            publisher=candidate.publisher,
            source_type=candidate.source_type,
            publication_date=candidate.publication_date,
            credibility_tier=candidate.credibility_tier,
            retrieved_at=candidate.retrieved_at or (self._created_at or utc_now()),
            notes=candidate.notes,
        )
        return source.validate()

    def _build_claims(
        self, candidates: list[SourceCandidate], sources: tuple[Source, ...]
    ) -> tuple[FactClaim, ...]:
        sources_by_id = {source.source_id: source for source in sources}
        claims: list[FactClaim] = []
        for index, candidate in enumerate(candidates):
            claim_items = candidate.claims
            if isinstance(claim_items, ClaimItem):
                claim_items = (claim_items,)
            source_id = source_id_for(candidate.url)
            for claim_index, item in enumerate(claim_items):
                try:
                    claims.append(self._claim_from_item(item, source_id, sources_by_id))
                except ResearchPackageError as exc:
                    raise ResearchPackageError(
                        f"malformed claim at candidate {index}, claim {claim_index}: {exc}"
                    ) from None
        return tuple(claims)

    def _claim_from_item(self, item, source_id: str, sources_by_id: dict) -> FactClaim:
        from agents.research.models import ClaimDate

        if not item or not getattr(item, "claim_text", "").strip():
            raise ResearchPackageError("claim_text is empty")
        dates = tuple(ClaimDate.from_dict(entry) for entry in item.dates)
        claim_type = (
            item.suggested_claim_type
            if item.suggested_claim_type
            else classify_claim(item.claim_text)
        )
        if claim_type not in (
            "fact",
            "inference",
            "explanation",
            "speculation",
            "opinion",
        ):
            raise ResearchPackageError(f"unknown claim_type '{claim_type}'")
        backing = (sources_by_id[source_id],)
        return FactClaim(
            claim_id=claim_id_for(item.claim_text),
            claim_text=item.claim_text,
            claim_type=claim_type,
            source_ids=(source_id,),
            confidence=compute_confidence(claim_type, backing),
            verification_status=PROVISIONAL,
            dates=dates,
            notes=item.notes if getattr(item, "notes", None) else "",
            critical=getattr(item, "critical", False),
        )

    @staticmethod
    def _compose_notes(
        claims: tuple[FactClaim, ...], sources: tuple[Source, ...], research_status: str
    ) -> tuple[str, ...]:
        from collections import Counter

        counts = Counter(claim.claim_type for claim in claims)
        breakdown = ", ".join(f"{claim_type}={count}" for claim_type, count in sorted(counts.items()))
        notes: list[str] = []
        notes.append("Research evidence gathered from local/mocked sources only.")
        notes.append(f"{len(claims)} claim(s) from {len(sources)} source(s): {breakdown}.")
        if research_status == "insufficient_evidence":
            notes.append("No structured evidence found for this topic; status set to insufficient_evidence.")
        elif research_status == "provisional":
            notes.append("Evidence is discovery-tier only (leads); status kept provisional.")
        else:
            notes.append("Core evidence present; package marked ready_for_fact_check.")
        notes.append(
            "Claims carry provisional verification only; final verification is deferred "
            "to the fact-check stage."
        )
        return tuple(notes)