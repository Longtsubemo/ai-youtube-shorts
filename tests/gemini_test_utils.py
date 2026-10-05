"""Shared offline fixtures for Gemini provider tests.

None of these make a network call or require a real API key. The fake raw
client mimics the small slice of `google.genai.Client` that
`GeminiClient.generate_structured` uses so the provider can be exercised
without touching Google.
"""

from __future__ import annotations

from types import SimpleNamespace


class FakeResponse:
    """Mimics GenerateContentResponse enough for `.text`."""

    def __init__(self, text: str | None = None) -> None:
        self.text = text

    @classmethod
    def json(cls, payload) -> "FakeResponse":
        import json

        return cls(json.dumps(payload))


class FakeRawClient:
    """Mimics `client.models.generate_content` with scripted outcomes."""

    def __init__(self, outcomes=None) -> None:
        # outcomes: list of callables(raw_client, *, model, contents, config) -> FakeResponse
        # or one fixed FakeResponse, or Exception instances to raise.
        self.outcomes = [] if outcomes is None else list(outcomes)
        self.calls: list[tuple] = []
        self._idx = 0

    def add_result(self, text: str) -> None:
        self.outcomes.append(FakeResponse(text))

    def add_result_json(self, payload) -> None:
        self.outcomes.append(FakeResponse.json(payload))

    def add_error(self, exc: Exception) -> None:
        self.outcomes.append(exc)

    @property
    def models(self):
        return self

    def generate_content(self, *, model, contents, config=None):
        self.calls.append((model, contents, config))
        if self.outcomes:
            outcome = self.outcomes[min(self._idx, len(self.outcomes) - 1)]
        else:  # pragma: no cover - always script outcomes in tests
            outcome = FakeResponse()
        self._idx += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @property
    def attempt_count(self) -> int:
        return len(self.calls)


# --------------------------------------------------------------------------- #
# Grounding-aware fakes (Stage-2 research)
# --------------------------------------------------------------------------- #


class FakeGroundingChunkWeb:
    """Mimics `GroundingChunk.Web` (uri/title/domain)."""

    def __init__(self, uri: str, title: str = "", domain: str = "") -> None:
        self.uri = uri
        self.title = title
        self.domain = domain


class FakeGroundingChunk:
    def __init__(self, web: FakeGroundingChunkWeb) -> None:
        self.web = web


class FakeGroundingMetadata:
    def __init__(self, chunks) -> None:
        self.grounding_chunks = chunks


class FakeUrlContextItem:
    def __init__(self, url: str) -> None:
        self.retrieved_url = url


class FakeUrlContextMetadata:
    def __init__(self, urls) -> None:
        self.url_metadata = [FakeUrlContextItem(url) for url in urls]


class FakeCandidate:
    """Mimics the grounding/url-context slice of a `Candidate`."""

    def __init__(self, grounding=None, url_context=None) -> None:
        self.grounding_metadata = grounding
        self.url_context_metadata = url_context


class FakeGroundedResponse(FakeResponse):
    """A response that carries a `candidates` list (grounding metadata)."""

    def __init__(self, text: str | None = None, candidates=()) -> None:
        super().__init__(text)
        self.candidates = tuple(candidates)


def grounded_result(
    payload=None,
    *,
    chunks=(),
    url_context_urls=(),
    text: str | None = None,
) -> FakeGroundedResponse:
    """Build a fake grounded response a FakeRawClient can return.

    `chunks` is an iterable of (uri, title, domain) triples for the
    `grounding_metadata.grounding_chunks`. `url_context_urls` populate the
    URL-context metadata. When both are empty the response has no candidates
    (as if the model did not ground anything).
    """
    import json as _json

    if text is None:
        text = _json.dumps({} if payload is None else payload)
    if not chunks and not url_context_urls:
        return FakeGroundedResponse(text=text)
    grounding = None
    if chunks:
        grounding = FakeGroundingMetadata(
            [FakeGroundingChunk(FakeGroundingChunkWeb(uri, title, domain)) for uri, title, domain in chunks]
        )
    url_context = FakeUrlContextMetadata(url_context_urls) if url_context_urls else None
    return FakeGroundedResponse(
        text=text,
        candidates=[FakeCandidate(grounding=grounding, url_context=url_context)],
    )


def valid_research_payload(
    urls=("https://example.test/papers/octopus",),
    critical_index: int | None = None,
) -> dict:
    """A schema-valid research payload (sources + claims) for offline tests.

    `critical_index` marks one source's claim critical; it defaults to None so
    fixtures are non-critical by default. These payloads exercise provider
    grounding, provenance and schema handling, not the Stage-3 corroboration
    policy. A blanket critical default made every single-source payload a
    critical claim that the producer-side guard rejects by design, so callers
    that genuinely mean to exercise criticality pass the index explicitly and
    supply enough eligible non-discovery sources for it.
    """
    sources = []
    for index, url in enumerate(urls):
        sources.append(
            {
                "url": url,
                "title": f"Grounded source {index}",
                "publisher": f"Publisher {index}",
                "source_type": "peer_reviewed_paper" if index == 0 else "news_article",
                "publication_date": "2020-05-12",
                "credibility_tier": "primary" if index == 0 else "high_quality_secondary",
                "notes": "",
                "claims": [
                    {
                        "claim_text": (
                            "Octopuses have three hearts: two branchial hearts and a systemic heart."
                            if index == 0
                            else "Octopus blood uses copper-based hemocyanin, which makes it appear blue."
                        ),
                        "dates": (
                            [{"date_kind": "discovery_date", "date": "2019-04-01"}]
                            if index == 0
                            else [{"date_kind": "publication_date", "date": "2020-05-12"}]
                        ),
                        "suggested_claim_type": "fact",
                        "critical": index == critical_index,
                    }
                ],
            }
        )
    return {"sources": sources}


def valid_candidates_payload(count: int = 3) -> dict:
    """A schema-valid candidate list payload for offline provider tests."""
    return {
        "candidates": [
            {
                "topic_id": "t_ai_agents_computers",
                "title": f"Candidate {i}",
                "description": "A sample candidate description.",
                "pillar": "ai",
                "topic_type": "evergreen",
                "signals": {
                    "interest": 0.7,
                    "trend_momentum": 0.5,
                    "competition": 0.4,
                    "freshness": 0.3,
                },
                "content_scores": {
                    "curiosity": 0.8,
                    "explainer_potential": 0.8,
                    "visual_potential": 0.7,
                    "audience_fit": 0.8,
                },
            }
            for i in range(count)
        ]
    }


def make_client(config, raw_client=None, sleep=None) -> "object":
    from providers.gemini.client import GeminiClient

    sleep = sleep or (lambda *_: None)
    return GeminiClient(config, raw_client=raw_client, sleep=sleep)


def noop_sleep(*_) -> None:
    """Disable backoff sleeps in offline tests."""
    return None