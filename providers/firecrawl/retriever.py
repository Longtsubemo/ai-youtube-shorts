"""FirecrawlResearchRetriever: bounded real-web retrieval via the Firecrawl API.

Owns everything that touches the open web: search, content cleaning, source
metadata, and domain allow/deny filtering. It produces `RetrievedSource`
records (the evidence boundary) that a downstream extractor turns into claims.

Determinism / safety properties:

- bounded search results (`config.max_search_results`),
- bounded per-source content (`max_content_chars`),
- explicit timeout + bounded retries handled by the client,
- domain allow/deny filtering,
- explicit failure handling: every Firecrawl failure is translated to
  `ResearchProviderError` so no raw SDK exception escapes the stage boundary.
"""

from __future__ import annotations

from urllib.parse import urlparse

from agents.research.provider import ResearchProviderError, TopicDescriptor
from agents.research.retriever import RetrievedSource
from providers.firecrawl.client import FirecrawlClient
from providers.firecrawl.config import FirecrawlConfig
from providers.firecrawl.errors import FirecrawlProviderError

DEFAULT_MAX_CONTENT_CHARS = 6000


def _host(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except ValueError:  # pragma: no cover - defensive
        return ""


def _domain_allowed(host: str, allow: frozenset[str], deny: frozenset[str]) -> bool:
    if host in deny or any(host.endswith("." + denied) for denied in deny):
        return False
    if not allow:
        return True
    return host in allow or any(host.endswith("." + allowed) for allowed in allow)


def _truncate_markdown(markdown: str, limit: int) -> str:
    text = (markdown or "").strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " …[truncated]"


class FirecrawlResearchRetriever:
    """Fetch real search results for a topic via the Firecrawl Search API."""

    retriever_name = "firecrawl"

    def __init__(
        self,
        client: FirecrawlClient | None = None,
        *,
        config: FirecrawlConfig | None = None,
        allow_domains: tuple[str, ...] = (),
        deny_domains: tuple[str, ...] = (),
        max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
    ) -> None:
        self._config = config or FirecrawlConfig()
        self._client = client if client is not None else FirecrawlClient(self._config)
        self._allow = frozenset(domain.strip().lower() for domain in allow_domains if domain.strip())
        self._deny = frozenset(domain.strip().lower() for domain in deny_domains if domain.strip())
        self._max_content_chars = max_content_chars

    @property
    def client(self) -> FirecrawlClient:
        return self._client

    @property
    def config(self) -> FirecrawlConfig:
        return self._config

    # -- protocol --------------------------------------------------------- #

    def retrieve(self, topic: TopicDescriptor) -> list[RetrievedSource]:
        """Search the web for `topic` and return bounded, filtered sources.

        Raises `ResearchProviderError` on any Firecrawl failure (typed,
        never a raw SDK exception) — including when the SDK is not installed,
        so a missing `firecrawl` dependency fails cleanly.
        """
        try:
            results = self._client.search(self._build_query(topic), limit=self._config.max_search_results)
        except FirecrawlProviderError as exc:
            raise ResearchProviderError(
                f"Firecrawl retrieval failed for topic {topic.topic_id!r}: {exc}"
            ) from exc

        sources: list[RetrievedSource] = []
        for result in results:
            url = (result.url or "").strip()
            host = _host(url)
            if not url or not _domain_allowed(host, self._allow, self._deny):
                continue
            sources.append(
                RetrievedSource(
                    url=url,
                    title=(result.title or "").strip(),
                    publisher=host,
                    source_type="web_page",
                    content=_truncate_markdown(result.markdown, self._max_content_chars),
                    domain=host,
                    retrieved_at="",
                )
            )
        return sources

    # -- internals --------------------------------------------------------- #

    @staticmethod
    def _build_query(topic: TopicDescriptor) -> str:
        """A deterministic query for the search API (no free-form prompt)."""
        return (topic.title or "").strip()