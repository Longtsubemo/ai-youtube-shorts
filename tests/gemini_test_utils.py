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