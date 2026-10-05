"""Persistence of HybridRetrievalTrace as a run artifact.

Covers the three seams that make the retrieval audit record survive a run:

- `HybridRetrievalTrace.to_dict()` serialises the recorded trace.
- `GeminiResearchProvider.last_retrieval_trace` exposes it without importing a
  concrete retriever, and degrades to None when there is none.
- The research stage runner writes `retrieval_trace.json` beside the package.

The load-bearing property is that persisting the trace does not change
retrieval: these tests assert the emitted sources are identical whether or not
the artifact is written. All offline; no network, Gemini key or SearXNG.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from agents.research.research_agent import _write_retrieval_trace, main
from providers.hybrid.query import ResearchQuery, ResearchQueryPlan
from providers.hybrid.retriever import HybridResearchRetriever, HybridRetrievalTrace

VALID_EVIDENCE = Path("data") / "mock" / "research_evidence.json"

# --selection no longer defaults to this fixture; it defaults to the current
# run's own data/<run_id>/research/topic_selection.json. Trace persistence is
# orthogonal to topic selection, so these cases name the fixture explicitly.
MOCK_SELECTION = Path("data") / "mock" / "topic_selection.json"


class _Searcher:
    def __init__(self, urls):
        self._urls = list(urls)

    def search(self, query, *, max_results=None):
        from providers.searxng.client import SearchResult

        return tuple(SearchResult(url=url, title=f"title for {url}", snippet="snippet") for url in self._urls)


class _Fetcher:
    def __init__(self, results):
        self._results = results
        self.calls: list[str] = []

    def fetch(self, url):
        self.calls.append(url)
        return self._results[url]


class _TraceProvider:
    """Minimal provider surface the trace writer relies on."""

    provider_name = "gemini"
    retriever_name = "hybrid"
    model = "stub-model"

    def __init__(self, trace):
        self.last_retrieval_trace = trace


class _NoTraceProvider:
    """A provider with no retriever at all (mock / Firecrawl shapes)."""

    provider_name = "gemini"
    retriever_name = None
    model = "stub-model"


def _trace() -> HybridRetrievalTrace:
    plan = ResearchQueryPlan(
        topic_id="t1",
        queries=(
            ResearchQuery(text="why is the sky blue", rule="description_factual_clause"),
            ResearchQuery(text="sky blue", rule="title_subject"),
        ),
    )
    return HybridRetrievalTrace(
        plan=plan,
        per_query_hits={"why is the sky blue": 3, "sky blue": 1},
        candidates=4,
        duplicates_dropped=1,
        domain_filtered=1,
        cheap_ok=2,
        cheap_failed={"https://blocked.test/a": "blocked", "https://dead.test/b": "http_error"},
        fallback_attempted=["https://blocked.test/a", "https://dead.test/b"],
        fallback_ok=["https://blocked.test/a"],
        fallback_skipped={"https://dead.test/b": "invalid_url"},
        emitted=2,
    )


# --------------------------------------------------------------------------- #
# to_dict
# --------------------------------------------------------------------------- #


def test_to_dict_captures_every_recorded_field():
    payload = _trace().to_dict()

    assert payload["schema_version"] in (2, 3)
    assert payload["candidates"] == 4
    assert payload["duplicates_dropped"] == 1
    assert payload["domain_filtered"] == 1
    assert payload["cheap_ok"] == 2
    assert payload["emitted"] == 2
    assert payload["per_query_hits"] == {"why is the sky blue": 3, "sky blue": 1}
    assert payload["cheap_failed"] == {
        "https://blocked.test/a": "blocked",
        "https://dead.test/b": "http_error",
    }
    assert payload["fallback_attempted"] == ["https://blocked.test/a", "https://dead.test/b"]
    assert payload["fallback_ok"] == ["https://blocked.test/a"]
    assert payload["fallback_skipped"] == {"https://dead.test/b": "invalid_url"}


def test_to_dict_expands_plan_structurally_not_only_as_text():
    """Consumers must not have to parse the human summary line back into queries."""
    plan = _trace().to_dict()["plan"]

    assert plan["topic_id"] == "t1"
    assert plan["queries"] == [
        {"text": "why is the sky blue", "rule": "description_factual_clause"},
        {"text": "sky blue", "rule": "title_subject"},
    ]
    assert "description_factual_clause" in plan["as_trace"]


def test_to_dict_handles_absent_and_empty_plans():
    assert HybridRetrievalTrace().to_dict()["plan"] is None

    empty = HybridRetrievalTrace(plan=ResearchQueryPlan(topic_id="t1", queries=()))
    assert empty.to_dict()["plan"]["queries"] == []


def test_to_dict_is_json_serialisable_and_copies_mutable_state():
    trace = _trace()
    payload = trace.to_dict()

    # A snapshot, not a live view: later mutation must not rewrite history.
    trace.cheap_failed["https://late.test/c"] = "timeout"
    trace.per_query_hits["extra"] = 9

    assert "https://late.test/c" not in payload["cheap_failed"]
    assert "extra" not in payload["per_query_hits"]
    json.dumps(payload)


# --------------------------------------------------------------------------- #
# provider accessor
# --------------------------------------------------------------------------- #


def test_provider_exposes_trace_without_concrete_retriever_import():
    from providers.gemini.research_provider import GeminiResearchProvider

    class _Retriever:
        retriever_name = "hybrid"

        def __init__(self):
            self.last_trace = _trace()

    provider = GeminiResearchProvider.__new__(GeminiResearchProvider)
    provider._retriever = _Retriever()

    assert provider.last_retrieval_trace is provider._retriever.last_trace


@pytest.mark.parametrize(
    "retriever",
    [None, object()],
    ids=["no_retriever", "retriever_without_trace"],
)
def test_provider_returns_none_when_no_trace_is_recorded(retriever):
    """Firecrawl retriever has no last_trace; getattr must not explode."""
    from providers.gemini.research_provider import GeminiResearchProvider

    provider = GeminiResearchProvider.__new__(GeminiResearchProvider)
    provider._retriever = retriever

    assert provider.last_retrieval_trace is None


# --------------------------------------------------------------------------- #
# artifact writing
# --------------------------------------------------------------------------- #


def test_writer_emits_json_with_provenance(tmp_path):
    path = _write_retrieval_trace(_TraceProvider(_trace()), tmp_path)

    assert path == tmp_path / "retrieval_trace.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["retriever"] == "hybrid"
    assert payload["provider"] == "gemini"
    assert payload["model"] == "stub-model"
    assert payload["trace"]["candidates"] == 4
    assert payload["trace"]["plan"]["topic_id"] == "t1"


def test_writer_is_a_noop_without_a_trace(tmp_path):
    """Artifact absent, not empty, so its absence stays meaningful."""
    assert _write_retrieval_trace(_NoTraceProvider(), tmp_path) is None
    assert _write_retrieval_trace(_TraceProvider(None), tmp_path) is None
    assert not (tmp_path / "retrieval_trace.json").exists()


def test_writer_creates_missing_directory(tmp_path):
    target = tmp_path / "nested" / "research"
    assert _write_retrieval_trace(_TraceProvider(_trace()), target) is not None
    assert (target / "retrieval_trace.json").exists()


# --------------------------------------------------------------------------- #
# the load-bearing guarantee: retrieval is unchanged
# --------------------------------------------------------------------------- #


def test_persisting_the_trace_does_not_change_emitted_sources(tmp_path):
    """Same fetches, same emitted URLs, with and without the artifact written."""
    urls = ["https://a.test/1", "https://b.test/2", "https://c.test/3"]

    def _run(*, persist: bool, out_dir) -> tuple[list[str], list[str], dict]:
        fetcher = _Fetcher({url: _page(url) for url in urls})
        retriever = HybridResearchRetriever(
            searcher=_Searcher(urls),
            fetcher=fetcher,
            enable_fallback=False,
        )
        emitted = [source.url for source in retriever.retrieve(_topic())]
        snapshot = retriever.last_trace.to_dict()

        if persist:
            _write_retrieval_trace(_TraceProvider(retriever.last_trace), out_dir)

        assert retriever.last_trace.to_dict() == snapshot, (
            "writing the artifact must not mutate the trace"
        )
        return list(fetcher.calls), emitted, snapshot

    # Identical retrieval inputs; the only difference is whether the trace is
    # persisted afterwards.
    control = tmp_path / "control"
    persisted = tmp_path / "persisted"

    calls_a, emitted_a, trace = _run(persist=False, out_dir=control)
    calls_b, emitted_b, _ = _run(persist=True, out_dir=persisted)

    assert calls_a == calls_b == urls, "fetch order must not change"
    assert emitted_a == emitted_b == urls, "emitted sources must not change"
    assert trace["emitted"] == 3
    assert trace["cheap_ok"] == 3

    assert not (control / "retrieval_trace.json").exists()
    assert (persisted / "retrieval_trace.json").exists()


# --------------------------------------------------------------------------- #
# CLI wiring
# --------------------------------------------------------------------------- #


def _page(url: str):
    from providers.html.fetcher import FetchedPage

    return FetchedPage(
        url=url,
        ok=True,
        title=f"page {url}",
        text="Real prose about the subject. " * 12,
        domain=url.split("/")[2],
    )


def _topic():
    from agents.research.provider import TopicDescriptor

    return TopicDescriptor(topic_id="t1", title="Why is the sky blue", description="")


def test_cli_writes_trace_for_gemini(monkeypatch, tmp_path):
    from agents.research.provider import MockResearchProvider

    class _StubGemini:
        provider_name = "gemini"
        retriever_name = "hybrid"
        model = "stub-model"
        last_retrieval_trace = _trace()

        @property
        def produced_by(self):
            return {"prompt_version": "research/facts-retrieval-v1", "provider": "gemini", "model": "stub-model"}

        def search(self, topic):
            return MockResearchProvider(data_path=VALID_EVIDENCE).search(topic)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "research_agent",
            "--provider",
            "gemini",
            "--selection",
            str(MOCK_SELECTION),
            "--out",
            str(tmp_path / "research.json"),
            "--run-id",
            "cli_trace",
        ],
    )
    monkeypatch.setattr(
        "agents.research.research_agent.build_research_provider",
        lambda *a, **k: _StubGemini(),
    )
    from pipeline.state import PipelineState

    monkeypatch.setattr(PipelineState, "save", lambda self, path: Path(path))

    assert main() == 0

    payload = json.loads((tmp_path / "retrieval_trace.json").read_text(encoding="utf-8"))
    assert payload["retriever"] == "hybrid"
    assert payload["trace"]["plan"]["topic_id"] == "t1"
    assert payload["trace"]["emitted"] == 2


def test_cli_writes_no_trace_for_mock(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "research_agent",
            "--selection",
            str(MOCK_SELECTION),
            "--out",
            str(tmp_path / "research.json"),
            "--run-id",
            "cli_no_trace",
        ],
    )
    from pipeline.state import PipelineState

    monkeypatch.setattr(PipelineState, "save", lambda self, path: Path(path))

    assert main() == 0
    assert not (tmp_path / "retrieval_trace.json").exists()