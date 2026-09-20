"""Research stage (Stage 2).

Data flow (docs/RESEARCH_PIPELINE.md):

    TopicSelection -> ResearchAgent -> ResearchProvider
                    -> SourceCandidate -> Source/FactClaim
                    -> ResearchPackage -> (future) Fact Checker

Everything here works against local/mocked evidence; the research agent is
deliberately provider-agnostic so a real web provider can be swapped in
without touching agent logic. Claims leave the stage provisional - final
verification belongs to the fact-check stage.
"""

from agents.research.agent import ResearchAgent
from agents.research.classifier import classify_claim, compute_confidence
from agents.research.models import (
    ClaimDate,
    FactClaim,
    ResearchPackage,
    ResearchPackageError,
    Source,
)
from agents.research.provider import (
    ClaimItem,
    MockResearchProvider,
    ResearchProvider,
    ResearchProviderError,
    SourceCandidate,
    TopicDescriptor,
)

__all__ = [
    "ResearchAgent",
    "classify_claim",
    "compute_confidence",
    "ClaimDate",
    "FactClaim",
    "ResearchPackage",
    "ResearchPackageError",
    "Source",
    "ClaimItem",
    "MockResearchProvider",
    "ResearchProvider",
    "ResearchProviderError",
    "SourceCandidate",
    "TopicDescriptor",
]