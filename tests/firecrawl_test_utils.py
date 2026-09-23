"""Shared offline fixtures for Firecrawl retrieval tests.

None of these make a network call or require a Firecrawl API key. The fake raw
client mimics the small slice of `firecrawl.v2.FirecrawlClient.search` that
`FirecrawlClient` uses so retrievers and providers can be exercised without
touching the Firecrawl API — and without the `firecrawl` SDK being installed.
"""

from __future__ import annotations

from types import SimpleNamespace

from providers.firecrawl.config import FirecrawlConfig


class FakeWebResult:
    """Mimics a v2 web search hit (SearchResultWeb / hydrated Document)."""

    def __init__(self, url: str, title: str = "", description: str = "", markdown: str = "", metadata=None) -> None:
        self.url = url
        self.title = title
        self.description = description
        self.markdown = markdown
        self.metadata = metadata


def fake_metadata(source_url: str = "", status_code: int = 200) -> SimpleNamespace:
    """Mimic `Document.metadata` (sourceURL/source_url/statusCode/title)."""
    return SimpleNamespace(
        sourceURL=source_url,
        source_url=source_url,
        statusCode=status_code,
        title=(_TITLE if source_url else ""),
    )


_TITLE = "Metadata title"


class FakeDocument:
    """Mimic a hydrated v2 `Document` result.

    The Firecrawl Search API returns `Document` objects when markdown/scrape
    formats are requested; those carry `markdown` + `metadata` and usually do
    NOT expose a top-level `url`/`title` (they live on `metadata`).
    """

    def __init__(self, metadata=None, markdown: str = "") -> None:
        self.metadata = metadata if metadata is not None else fake_metadata()
        self.markdown = markdown


def web_bucket(*results) -> SimpleNamespace:
    """Wrap results in the v2 SearchData shape (`result.web`)."""
    return SimpleNamespace(web=list(results))


def web_dict(*results) -> dict:
    """Wrap results assuming a dict response (`{"web": [...]}`)."""
    return {"web": list(results)}


class FakeFirecrawlRawClient:
    """Mimics `firecrawl.Firecrawl.search(query, **options)`.

    `outcomes` is a list of callables/values/exceptions; when it is exhausted
    the last outcome repeats (so retry exhaustion behaves like a real client).
    """

    def __init__(self, outcome=None) -> None:
        self.outcomes = [] if outcome is None else [outcome]
        self.calls: list[tuple] = []
        self._idx = 0

    def add_result(self, value) -> None:
        self.outcomes.append(value)

    def add_error(self, exc: Exception) -> None:
        self.outcomes.append(exc)

    @property
    def attempt_count(self) -> int:
        return len(self.calls)

    def search(self, query: str, **kwargs):
        self.calls.append((query, kwargs))
        if not self.outcomes:
            return web_bucket()
        outcome = self.outcomes[min(self._idx, len(self.outcomes) - 1)]
        self._idx += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_firecrawl_client(config: FirecrawlConfig | None = None, raw_client=None, sleep=None) -> "object":
    from providers.firecrawl.client import FirecrawlClient

    sleep = sleep or (lambda *_: None)
    return FirecrawlClient(config or FirecrawlConfig(), raw_client=raw_client, sleep=sleep)


def make_http_error(status_code: int):
    """A real exception that mimicks an SDK status error (`.status_code`)."""

    class FakeHttpError(Exception):
        def __init__(self, code: int) -> None:
            super().__init__(f"HTTP {code}")
            self.status_code = code

    return FakeHttpError(status_code)