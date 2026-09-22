"""TopicCandidateProvider abstraction and its deterministic mock implementation.

Stage 1 (topic selection) currently sources candidates from a local JSON file
(`data/mock/topic_candidates.json`) through `load_candidates`. The provider
seam keeps that mock path as the default while allowing a real candidate
source (e.g. Gemini) to drop in later without touching scoring/selection.

The mock candidates are deterministic and clearly marked as internal mock
data (`generated_by="mock_catalog_v1"`, `source="internal_mock"`). Scaling,
ranking, and replay behavior are unchanged: candidates flow through the same
scorer/selector as before.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Protocol

from agents.seo.config import TopicSelectionConfig
from agents.seo.models import TopicCandidate
from agents.seo.stage import MOCK_CANDIDATES_PATH, load_candidates


class TopicCandidateProviderError(ValueError):
    """Raised when a candidate provider cannot fulfill the request."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class TopicCandidateProvider(Protocol):
    """Anything that can return topic candidates for scoring.

    Implementations may later wrap a research API, an LLM, or a file on disk.
    The stage never sees the concrete class.
    """

    def list_candidates(self) -> list[TopicCandidate]: ...


class MockTopicCandidateProvider:
    """Deterministic, network-free candidate provider backed by the mock JSON."""

    def __init__(
        self,
        path: str | Path = MOCK_CANDIDATES_PATH,
        config: TopicSelectionConfig | None = None,
    ) -> None:
        self._path = Path(path)
        self._config = config

    def list_candidates(self) -> list[TopicCandidate]:
        if not self._path.is_file():
            raise TopicCandidateProviderError(f"mock candidate fixture not found: {self._path}")
        return load_candidates(self._path, self._config)


def build_topic_candidate_provider(
    name: str,
    *,
    config: TopicSelectionConfig,
    candidates_path: str | Path = MOCK_CANDIDATES_PATH,
    env: Mapping[str, str] | None = None,
) -> TopicCandidateProvider:
    """Return the candidate provider selected by `name` (default: mock).

    Only the requested provider is imported, so a mock/default run never
    needs or imports the Gemini SDK. Providing a real provider requires the
    GEMINI_API_KEY environment variable; a missing key raises
    GeminiConfigurationError rather than silently failing over to mock.
    """
    if name in (None, "", "mock"):
        return MockTopicCandidateProvider(candidates_path, config)

    if name == "gemini":
        from providers.gemini.config import GeminiConfig
        from providers.gemini.topic_provider import GeminiTopicProvider  # lazy import
        from providers.gemini.client import GeminiClient

        gemini_config = GeminiConfig.from_env(env if env is not None else os.environ)
        client = GeminiClient(gemini_config)
        return GeminiTopicProvider(config=config, client=client)

    raise TopicCandidateProviderError(f"unknown topic candidate provider: {name!r}")