"""BeautifulSoup fetcher for ordinary HTML pages — the cheap tier of retrieval.

The point of this fetcher is the cheap tier of the intended routing:

    SearXNG -> search results -> ordinary page fetch -> BeautifulSoup
             -> (only if inadequate) self-hosted Firecrawl

So it must be good at plain, static, non-hostile pages and it must be completely
honest about everything else.

The honesty rule
----------------
`HtmlFetcher.fetch()` **never raises for a network or HTTP problem and never
returns a "successful" result for a page it did not actually read.** Every
outcome is a `FetchedPage` with an explicit `ok` flag and, when `ok` is False,
an `error_kind` plus a human-readable `error`. A captcha interstitial, a 403, a
timeout, a PDF served with a 200 — all of them come back `ok=False`. This is
what lets a caller route only genuinely-difficult pages to the expensive tier
instead of quietly counting blocked pages as content.

Layout
------
- `extract_document()` — pure, offline, no network. Parses HTML into a
  `ParsedDocument`. Directly unit-testable with fixture HTML strings.
- `HtmlFetcher` — the only part that touches the network. Injectable transport
  so HTTP behaviour is testable without a live site.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from providers.html.errors import FetcherConfigurationError
from providers.searxng.client import host_for

DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_CONTENT_CHARS = 6000
DEFAULT_USER_AGENT = (
    "ai-youtube-channel-retrieval/1.0 "
    "(+local BeautifulSoup page fetcher; contact: repository owner)"
)

# Statuses that mean "you were not served the page", regardless of body.
BLOCKING_STATUSES = frozenset({401, 402, 403, 406, 407, 408, 429, 451})

# Tags whose text is chrome, not content.
_NOISE_TAGS = (
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "canvas",
    "iframe",
    "object",
    "embed",
    "form",
    "button",
    "nav",
    "header",
    "footer",
    "aside",
    "select",
    "textarea",
)

# Markers of an anti-bot / paywall interstitial. Matched against the *title*
# and the first part of the extracted text. Deliberately a list of specific
# phrases rather than "short page": a short page is still a real page.
_BLOCK_MARKERS = (
    "just a moment",
    "checking your browser",
    "verify you are human",
    "verifying you are human",
    "are you a robot",
    "enable javascript and cookies to continue",
    "please enable js and disable any ad blocker",
    "unusual traffic from your computer network",
    "access to this page has been denied",
    "attention required! | cloudflare",
    "ddos protection by",
    "captcha",
    "recaptcha",
    "rate limit",
    "too many requests",
    "sign in to continue",
    "subscribe to continue reading",
    "this content is only available to our subscribers",
    "request unsuccessful. incapsula",
    # Consent / JS-redirect interstitials. These answer HTTP 200 and look like a
    # successful fetch, but serve no article text. Verified against a live
    # google.com/search response during the experiment: 200 OK, 88 characters of
    # "click here" boilerplate.
    "click here if you are redirected",
    "before you continue to google search",
    "if you're having trouble accessing google search",
    "here are a few simple ways",
    "to continue, please confirm",
)

# Below this many characters of prose a "page" cannot serve as retrieval
# evidence. This is NOT "short pages are invalid": the extracted text is kept
# on the returned FetchedPage for inspection, and the threshold is a constructor
# argument (pass 0 to disable). It exists because HTTP 200 interstitials that no
# marker catches otherwise read as successes.
DEFAULT_MIN_USABLE_CHARS = 200

_WHITESPACE_RUN = re.compile(r"[ \t\f\v]+")
_BLANK_RUN = re.compile(r"\n{3,}")
_SCRIPTISH = re.compile(r"(javascript|function\s*\(|<!doctype|\{\{|\$\()")


@dataclass(frozen=True)
class ParsedDocument:
    """Readable content extracted from an HTML string. Pure data, no network."""

    title: str = ""
    meta_description: str = ""
    language: str = ""
    author: str = ""
    text: str = ""
    canonical_url: str = ""
    content_length: int = 0

    def to_source_fields(self) -> dict[str, str]:
        """The contract-shaped fields this document can honestly supply."""
        return {
            "title": self.title,
            "publisher": "",  # publisher is a site fact; the fetcher sets it from the host
            "publication_date": "",  # never inferred from markup - would be a guess
            "content": self.text,
        }


@dataclass(frozen=True)
class FetchedPage:
    """The result of one fetch attempt, successful or not.

    `ok=False` always carries `error` and `error_kind`. `text`/`title` are empty
    on failure, so a blocked page cannot leak into results as content.
    """

    url: str
    ok: bool = False
    final_url: str = ""
    http_status: int | None = None
    content_type: str = ""
    title: str = ""
    domain: str = ""
    language: str = ""
    meta_description: str = ""
    text: str = ""
    content_length: int = 0
    elapsed_ms: int | None = None
    error: str = ""
    error_kind: str = ""
    document: ParsedDocument | None = None

    def to_source_fields(self) -> dict[str, str]:
        """Contract fields for a successful page, all else empty."""
        if not self.ok:
            return {"title": "", "publisher": "", "publication_date": "", "content": ""}
        return {
            "title": self.title,
            "publisher": self.domain,
            "publication_date": "",  # not inferred from markup
            "content": self.text,
        }


@dataclass
class FetchStats:
    """Counters for a batch of fetches (used by the comparison experiment)."""

    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    errors: dict[str, int] = field(default_factory=dict)

    def record_success(self) -> None:
        self.succeeded += 1

    def record_failure(self, kind: str) -> None:
        self.failed += 1
        self.errors[kind] = self.errors.get(kind, 0) + 1


# --------------------------------------------------------------------------- #
# Pure HTML extraction (no network)
# --------------------------------------------------------------------------- #


def _soup(html: str, parser: str | None = None) -> BeautifulSoup:
    """Parse with lxml when available, else the stdlib parser."""
    if parser:
        return BeautifulSoup(html, parser)
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:  # noqa: BLE001 - lxml missing or unbuildable parser
        return BeautifulSoup(html, "html.parser")


def _meta(soup: BeautifulSoup, *names: str) -> str:
    """First non-empty `<meta>` content among `names` (attribute lookup order)."""
    for name in names:
        for attr in ("property", "name", "itemprop"):
            tag = soup.find("meta", attrs={attr: name})
            if tag is None:
                continue
            value = tag.get("content")
            if isinstance(value, list):  # pragma: no cover - defensive
                value = " ".join(str(v) for v in value)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _first_text(node: Any) -> str:
    if node is None:
        return ""
    text = node.get_text(" ", strip=True)
    return _WHITESPACE_RUN.sub(" ", text).strip() if isinstance(text, str) else ""


def looks_blocked(*, status: int | None, title: str, text: str) -> bool:
    """True when this looks like an anti-bot/paywall interstitial, not a page.

    Two independent signals, either sufficient: a blocking HTTP status, or a
    known interstitial marker in the title / opening text. A short page with no
    marker is NOT treated as blocked.
    """
    if status is not None and status in BLOCKING_STATUSES:
        return True
    haystack = f"{title}\n{text[:1200]}".lower()
    return any(marker in haystack for marker in _BLOCK_MARKERS)


def is_scriptish(text: str) -> bool:
    """True when extracted text is obviously not human prose."""
    sample = text[:600]
    if not sample:
        return True
    return bool(_SCRIPTISH.search(sample))


def extract_document(html: str, *, base_url: str = "") -> ParsedDocument:
    """Extract readable text and basic metadata from an HTML string.

    Never raises for bad markup: empty, malformed, or non-HTML input yields an
    empty `ParsedDocument`. That keeps "this page had no extractable text" a
    reportable outcome rather than an exception.
    """
    if not html or not html.strip():
        return ParsedDocument()

    soup = _soup(html)

    title = ""
    if soup.title and soup.title.string:
        title = _WHITESPACE_RUN.sub(" ", str(soup.title.string)).strip()
    if not title:
        title = _meta(soup, "og:title", "twitter:title")
    if not title:
        heading = soup.find("h1")
        title = _first_text(heading)

    canonical = ""
    link = soup.find("link", rel=lambda value: bool(value) and "canonical" in " ".join(value if isinstance(value, list) else [value]).lower())
    if link is not None:
        href = link.get("href")
        if isinstance(href, str):
            canonical = href.strip()

    language = ""
    html_tag = soup.find("html")
    if html_tag is not None:
        raw_lang = html_tag.get("lang")
        if isinstance(raw_lang, str):
            language = raw_lang.strip()
    if not language:
        language = _meta(soup, "og:locale")

    # Strip chrome before choosing the content root.
    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()

    root = None
    for finder in (
        lambda: soup.find("article"),
        lambda: soup.find("main"),
        lambda: soup.find(attrs={"role": "main"}),
        lambda: soup.body,
        lambda: soup,
    ):
        candidate = finder()
        if candidate is not None:
            root = candidate
            break

    text = ""
    if root is not None:
        # Block-aware separators keep paragraphs readable instead of running
        # headings and paragraphs together.
        text = root.get_text("\n", strip=True)
        text = _WHITESPACE_RUN.sub(" ", text)
        text = "\n".join(line.strip() for line in text.split("\n"))
        text = _BLANK_RUN.sub("\n\n", text).strip()

    return ParsedDocument(
        title=title,
        meta_description=_meta(soup, "description", "og:description", "twitter:description"),
        language=language,
        author=_meta(soup, "author", "article:author"),
        text=text,
        canonical_url=canonical or base_url,
        content_length=len(text),
    )


# --------------------------------------------------------------------------- #
# Network fetching
# --------------------------------------------------------------------------- #


def _looks_like_html(content_type: str, body: str) -> bool:
    ctype = content_type.split(";")[0].strip().lower()
    if ctype:
        if ctype in ("text/html", "application/xhtml+xml"):
            return True
        if ctype.startswith("text/"):
            # text/plain and friends are readable as-is.
            return ctype in ("text/plain",)
        return False
    # No content-type at all: fall back to sniffing the body.
    head = body[:2048].lstrip().lower()
    return "<html" in head or "<!doctype html" in head or "<body" in head or "<head" in head


class HtmlFetcher:
    """Fetch an ordinary web page and extract readable text with BeautifulSoup.

    `transport` is injectable so HTTP behaviour (timeouts, 403, 404, non-HTML)
    can be tested offline. `max_content_chars` bounds the returned text.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
        max_retries: int = 1,
        min_usable_chars: int = DEFAULT_MIN_USABLE_CHARS,
        user_agent: str = DEFAULT_USER_AGENT,
        transport: httpx.BaseTransport | None = None,
        sleep: Any = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise FetcherConfigurationError("timeout_seconds must be > 0")
        if max_content_chars < 1:
            raise FetcherConfigurationError("max_content_chars must be >= 1")
        if max_retries < 0:
            raise FetcherConfigurationError("max_retries must be >= 0")
        if min_usable_chars < 0:
            raise FetcherConfigurationError("min_usable_chars must be >= 0")

        self._timeout_seconds = timeout_seconds
        self._max_content_chars = max_content_chars
        self._max_retries = max_retries
        self._min_usable_chars = min_usable_chars
        self._user_agent = user_agent
        self._transport = transport
        self._sleep = sleep if sleep is not None else time.sleep

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_seconds

    def fetch(self, url: str) -> FetchedPage:
        """Fetch `url` and extract its text. Never raises for network failure.

        Every outcome is a `FetchedPage`. A page that was blocked, timed out,
        returned an error status, or was not HTML comes back `ok=False` with an
        `error_kind` of `invalid_url`, `timeout`, `connection`, `http_error`,
        `blocked`, `content_type`, or `empty_content`.
        """
        target = (url or "").strip()
        parsed = urlparse(target)
        if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
            return FetchedPage(
                url=target,
                ok=False,
                error=f"not an absolute http(s) URL: {target!r}",
                error_kind="invalid_url",
            )

        attempts = self._max_retries + 1
        last: FetchedPage | None = None

        for attempt in range(attempts):
            started = time.perf_counter()
            page, retryable = self._attempt(target, started)
            if page.ok or not retryable or attempt >= attempts - 1:
                return page
            last = page
            self._sleep(0.25 * (2**attempt))

        assert last is not None  # pragma: no cover - unreachable
        return last

    def fetch_many(self, urls: Iterable[str]) -> tuple[FetchedPage, ...]:
        """Fetch several URLs sequentially, preserving input order."""
        return tuple(self.fetch(url) for url in urls)

    # -- internals ---------------------------------------------------------- #

    def _attempt(self, target: str, started: float) -> tuple[FetchedPage, bool]:
        """One fetch attempt. Returns (page, retryable)."""
        try:
            with httpx.Client(
                transport=self._transport,
                timeout=self._timeout_seconds,
                follow_redirects=True,
            ) as client:
                response = client.get(
                    target,
                    headers={
                        # An honest, identifying User-Agent. This experiment does
                        # NOT impersonate a browser: some sites will refuse it,
                        # and that refusal is a real measurement worth keeping.
                        "User-Agent": self._user_agent,
                        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
                    },
                )
        except Exception as exc:  # noqa: BLE001 - every failure becomes a FetchedPage
            return self._failure(target, started, exc), self._is_retryable(exc)

        elapsed = self._elapsed_ms(started)
        status = response.status_code
        content_type = response.headers.get("content-type", "") or ""

        if status >= 400:
            retryable = status in (408, 429, 500, 502, 503, 504)
            kind = "blocked" if status in BLOCKING_STATUSES else "http_error"
            return (
                FetchedPage(
                    url=target,
                    ok=False,
                    final_url=str(response.url),
                    http_status=status,
                    content_type=content_type,
                    domain=host_for(str(response.url)) or host_for(target),
                    elapsed_ms=elapsed,
                    error=f"HTTP {status} from {host_for(target) or target}",
                    error_kind=kind,
                ),
                retryable,
            )

        body = response.text
        if not _looks_like_html(content_type, body):
            return (
                FetchedPage(
                    url=target,
                    ok=False,
                    final_url=str(response.url),
                    http_status=status,
                    content_type=content_type,
                    domain=host_for(str(response.url)) or host_for(target),
                    elapsed_ms=elapsed,
                    error=(
                        "response was not HTML "
                        f"(content-type={content_type or 'unset'!r}, {len(body)} bytes)"
                    ),
                    error_kind="content_type",
                ),
                False,
            )

        document = extract_document(body, base_url=str(response.url))

        if looks_blocked(status=status, title=document.title, text=document.text):
            return (
                FetchedPage(
                    url=target,
                    ok=False,
                    final_url=str(response.url),
                    http_status=status,
                    content_type=content_type,
                    domain=host_for(str(response.url)) or host_for(target),
                    elapsed_ms=elapsed,
                    error="page looks like an anti-bot / paywall interstitial, not content",
                    error_kind="blocked",
                    document=document,
                ),
                False,
            )

        if not document.text.strip() or is_scriptish(document.text):
            return (
                FetchedPage(
                    url=target,
                    ok=False,
                    final_url=str(response.url),
                    http_status=status,
                    content_type=content_type,
                    title=document.title,
                    domain=host_for(str(response.url)) or host_for(target),
                    elapsed_ms=elapsed,
                    error="no readable prose extracted (empty or script-like body)",
                    error_kind="empty_content",
                    document=document,
                ),
                False,
            )

        if self._min_usable_chars and len(document.text) < self._min_usable_chars:
            # HTTP 200, no marker, but too little prose to be evidence. Reported
            # as a failure so it cannot inflate a success count, with the text
            # retained for inspection.
            return (
                FetchedPage(
                    url=target,
                    ok=False,
                    final_url=str(response.url),
                    http_status=status,
                    content_type=content_type,
                    title=document.title,
                    domain=host_for(str(response.url)) or host_for(target),
                    language=document.language,
                    text=document.text,
                    content_length=len(document.text),
                    elapsed_ms=elapsed,
                    error=(
                        f"only {len(document.text)} chars of prose "
                        f"(< min_usable_chars={self._min_usable_chars}); "
                        "looks like an interstitial, not an article"
                    ),
                    error_kind="thin_content",
                    document=document,
                ),
                False,
            )

        text = document.text
        if len(text) > self._max_content_chars:
            text = text[: self._max_content_chars].rstrip() + " …[truncated]"

        return (
            FetchedPage(
                url=target,
                ok=True,
                final_url=str(response.url),
                http_status=status,
                content_type=content_type,
                title=document.title,
                domain=host_for(str(response.url)) or host_for(target),
                language=document.language,
                meta_description=document.meta_description,
                text=text,
                content_length=len(text),
                elapsed_ms=elapsed,
                error="",
                error_kind="",
                document=document,
            ),
            False,
        )

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return int((time.perf_counter() - started) * 1000)

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)):
            return True
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return isinstance(status, int) and status in (408, 429, 500, 502, 503, 504)

    def _failure(self, target: str, started: float, exc: Exception) -> FetchedPage:
        name = type(exc).__name__
        if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
            kind, message = "timeout", f"timed out after {self._timeout_seconds:.0f}s"
        elif isinstance(exc, (httpx.ConnectError,)):
            kind, message = "connection", "could not connect"
        elif isinstance(exc, httpx.InvalidURL):
            kind, message = "invalid_url", "invalid URL"
        else:
            kind, message = "connection", f"{name}: {exc}"
        return FetchedPage(
            url=target,
            ok=False,
            http_status=getattr(getattr(exc, "response", None), "status_code", None),
            domain=host_for(target),
            elapsed_ms=self._elapsed_ms(started),
            error=message,
            error_kind=kind,
        )


def summarize(pages: Sequence[FetchedPage]) -> dict[str, Any]:
    """Aggregate counters for a batch of `FetchedPage`s (Phase 6 reporting)."""
    stats = FetchStats()
    total_chars = 0
    titles = 0
    statuses: dict[str, int] = {}
    for page in pages:
        stats.attempted += 1
        if page.ok:
            stats.record_success()
            total_chars += page.content_length
            titles += 1 if page.title else 0
        else:
            stats.record_failure(page.error_kind or "unknown")
        key = str(page.http_status) if page.http_status is not None else "none"
        statuses[key] = statuses.get(key, 0) + 1
    return {
        "attempted": stats.attempted,
        "succeeded": stats.succeeded,
        "failed": stats.failed,
        "titles_available": titles,
        "total_content_chars": total_chars,
        "http_status_counts": dict(sorted(statuses.items())),
        "error_kinds": dict(sorted(stats.errors.items())),
    }


__all__: Sequence[str] = (
    "BLOCKING_STATUSES",
    "DEFAULT_MAX_CONTENT_CHARS",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_USER_AGENT",
    "FetchedPage",
    "FetchStats",
    "HtmlFetcher",
    "ParsedDocument",
    "extract_document",
    "is_scriptish",
    "looks_blocked",
    "summarize",
)