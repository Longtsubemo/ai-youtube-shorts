"""Factory for the Stage-3 fact-check providers (evidence provider + verifier).

Mirrors `build_research_provider` (`agents/research/provider.py`): only the
requested provider is imported, so a mock/default run never needs or imports
the Gemini/Firecrawl SDKs. A real run requires the GEMINI_API_KEY environment
variable; a missing key raises GeminiConfigurationError rather than silently
failing over to mock.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from agents.qa.fact_checker.evidence import (
    DEFAULT_EVIDENCE_DATA_PATH,
    EvidenceProvider,
    EvidenceProviderError,
)
from agents.qa.fact_checker.verifier import Verifier


@dataclass(frozen=True)
class FactCheckComponents:
    """The pair of Stage-3 providers: an EvidenceProvider plus a Verifier.

    `provider` collects evidence per source; `verifier` judges each claim
    against that evidence (agents/qa/fact_checker/checker.py consumes both).
    """

    provider: EvidenceProvider
    verifier: Verifier


def build_fact_check_components(
    name: str | None,
    *,
    evidence_data_path: str | Path = DEFAULT_EVIDENCE_DATA_PATH,
    env: Mapping[str, str] | None = None,
    retriever: str | None = None,
) -> FactCheckComponents:
    """Return the (evidence provider, verifier) pair selected by `name`.

    `name` in (None, "", "mock") builds the deterministic mock pair from the
    fixture at `evidence_data_path`. `name == "gemini"` builds the real Gemini
    pair; `retriever="firecrawl"` wires verification web
    retrieval for the gemini evidence provider behind a `ResearchRetriever`
    (reusing the Stage-2 Firecrawl seam). No other provider accepts a retriever.
    """
    if name in (None, "", "mock"):
        if retriever is not None:
            raise EvidenceProviderError("--retriever is only supported with the gemini provider")
        from agents.qa.fact_checker.evidence import MockEvidenceProvider
        from agents.qa.fact_checker.verifier import MockVerifier

        return FactCheckComponents(
            provider=MockEvidenceProvider(data_path=evidence_data_path),
            verifier=MockVerifier(),
        )

    if name == "gemini":
        from providers.gemini.client import GeminiClient
        from providers.gemini.config import GeminiConfig
        from providers.gemini.evidence_provider import GeminiEvidenceProvider
        from providers.gemini.verifier_provider import GeminiVerifier

        gemini_config = GeminiConfig.from_env(env if env is not None else os.environ)
        client = GeminiClient(gemini_config)

        built_retriever = None
        if retriever == "firecrawl":
            from providers.firecrawl.config import FirecrawlConfig
            from providers.firecrawl.retriever import FirecrawlResearchRetriever

            firecrawl_config = FirecrawlConfig.from_env(env if env is not None else os.environ)
            built_retriever = FirecrawlResearchRetriever(config=firecrawl_config)
        elif retriever is not None:
            raise EvidenceProviderError(f"unknown retriever: {retriever!r}")

        return FactCheckComponents(
            provider=GeminiEvidenceProvider(client=client, retriever=built_retriever),
            verifier=GeminiVerifier(client=client),
        )

    raise EvidenceProviderError(f"unknown fact-check provider: {name!r}")