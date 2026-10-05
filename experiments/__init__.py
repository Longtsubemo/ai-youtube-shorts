"""Isolated experiments that evaluate retrieval options for Stage 2.

Nothing in this package is imported by the production pipeline. `agents/`,
`providers/`, and `pipeline/` do not import `experiments.*`, and nothing here
should ever be wired into a stage without an explicit decision.

The experiment under `experiments/retrieval/` explores a self-hosted retrieval
stack: a local SearXNG instance for discovery, direct HTTP + BeautifulSoup for
ordinary pages, and self-hosted Firecrawl for pages that are difficult or
blocked. It reuses the production `RetrievedSource` contract, `source_id_for()`,
and `normalize_source_url()` as-is so that anything proven here can be integrated
later without changing the contract.
"""

from __future__ import annotations