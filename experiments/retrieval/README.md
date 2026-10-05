# Retrieval Experiment

This directory holds a retrieval experiment that compares three extraction methods over the same discovery set, using **only local/self-hosted infrastructure**.

No file under `agents/`, `providers/` or `pipeline/` imports `experiments/`; only `tests/` do. The isolation that mattered during the experiment has since been relaxed in the other direction: the SearXNG client, the BeautifulSoup fetcher and the typed retrieval errors were **promoted into `providers/`** and are now used by production Stage 2 (`providers/hybrid/retriever.py`). `searxng_client.py`, `html_fetcher.py` and `errors.py` here are retained as **re-export shims** onto those providers, so existing experiment tests and the CLI keep working. `adapter.py`, `comparison.py` and `firecrawl_selfhost.py` remain experiment-specific.

## Scope
- **SearXNG**: local metasearch (`experiments/retrieval/searxng/`). Returns real web results in JSON.
- **BeautifulSoup**: honest HTML fetcher. No browser, no impersonation of a browser. Respects content-type, detects CAPTCHA/interstitials, rejects thin/blocked pages. Implementation now lives in `providers/html/fetcher.py`; `experiments/retrieval/html_fetcher.py` re-exports it.
- **Self-hosted Firecrawl**: open-source Firecrawl on localhost (`experiments/retrieval/firecrawl_selfhost/`). Uses `/v2/search` and `/v2/scrape` via Playwright. Verified to work locally (tuned memory/worker count).

Production retrieval uses these providers. `agents/research/retriever.py` (which defines `RetrievedSource`) is unmodified; `providers/firecrawl/client.py` **has** since been hardened (status-code, bot-challenge and empty-Markdown rejection) as a result of the real runs, tracked in `docs/PROJECT_PROGRESS.md`.

## What was measured
The central design: hold discovery fixed so extraction differences are attributable to the method.

1. **SearXNG** (one search) → same URL list → **BeautifulSoup**
2. Same list → **self-hosted Firecrawl (scrape)**
3. **self-hosted Firecrawl (search)** recorded separately (uses its own discovery/scrape)

Constraints enforced during the experiment:
- No cloud Firecrawl API calls (no `FIRECRAWL_API_KEY` forwarded; local only)
- No fabrication of dates/sources/tiers (tiering is model-judged; the adapter assigns none)
- Contract preserved: maps to existing `RetrievedSource` using `agents/research/models.py::source_id_for()` and `normalize_source_url()`
- Externally nondeterministic results are stated explicitly (engines rotate, CAPTCHA, content sizes vary)

Note: "no production integration yet" no longer holds. Stage 2 now runs `--retriever hybrid`, which drives the same SearXNG → BeautifulSoup → bounded self-hosted Firecrawl chain.

## Architecture
- `searxng_client.py` – re-export shim onto `providers/searxng/client.py` (normalized `SearchResult`, retries, typed errors, dedup)
- `adapter.py` – `SearchResult` → `RetrievedSource` (strict: only real ISO dates reach `publication_date`; bare years rejected as partial claims belong elsewhere)
- `html_fetcher.py` – re-export shim onto `providers/html/fetcher.py` (honest fetch, boilerplate removal, blocked/thin-content detection)
- `firecrawl_selfhost.py` – direct HTTP to local `/v2/*`, no SDK dependency
- `comparison.py` – three-way comparison with canonical URL aggregation (`normalize_source_url()`) and explicit "unavailable" reporting
- `__main__.py` – CLI (`probe`, `fetch`, `adapter`, `compare`, `firecrawl`)

## Running (offline unit tests)
All unit tests run without network:
```bash
python -m pytest tests/test_retrieval_searxng_client.py tests/test_retrieval_adapter.py tests/test_retrieval_html_fetcher.py tests/test_retrieval_comparison.py tests/test_hybrid_retriever.py tests/test_hybrid_query_planner.py tests/test_hybrid_factory_wiring.py -q
```
Full suite: `python -m pytest -q` (network smoke tests are skipped unless explicitly gated).

## Network smoke tests (local stacks)
Gated to avoid accidental network calls:
```bash
RETRIEVAL_NETWORK_TESTS=1 python -m pytest tests/test_retrieval_network_smoke.py -q
```
Requires local SearXNG (`http://127.0.0.1:8080`) and self-hosted Firecrawl (`http://127.0.0.1:3002`) healthy. If either is down the tests skip (environment fact, not a defect).

## Key findings (recorded)
- Self-hosted Firecrawl is 5 services; `NUQ_WORKER_COUNT=1` is the effective lever for laptop memory (12 GiB declared vs ~7.6 GiB). Steady-state ~2.3 GiB total.
- BeautifulSoup gets HTTP 403 from Wikipedia (honest UA). Self-hosted Firecrawl renders it to Markdown (30k+ chars) — the concrete fallback justification.
- Production `providers/firecrawl.client.FirecrawlClient` targets localhost by configuration alone (only `api_url` differs). It has since been hardened for self-hosted quirks — `proxy="basic"` is pinned because cloud-only `stealthProxy` returns HTTP 500 — but its interface is otherwise unchanged, so adoption remains a config change rather than a rewrite.

## Isolation status (as of 2026-10-02)
- `experiments/__init__.py`, `experiments/retrieval/__init__.py` are experiment markers.
- Production code never imports `experiments/`; only `tests/` do.
- `experiments/retrieval/searxng/settings.yml` is **local machine tuning** (an engine block added while troubleshooting suspended/CAPTCHA engines), not a portable default. It has never been validated end-to-end.
- The promoted retrieval stack (`providers/searxng/`, `providers/hybrid/`, `providers/html/`) and this whole experiment directory are **untracked working-tree files**. They are not committed.
- Root `.env` unchanged (including `GEMINI_MODEL=gemini-3.5-flash-lite`). Experiment uses its own `.env` under `firecrawl_selfhost/` (gitignored).