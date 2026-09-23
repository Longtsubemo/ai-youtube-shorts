# Project Progress — Automated AI YouTube Shorts Channel

Handoff document. A new developer or AI coding agent should read this file
first to understand exactly where the project stands and how to continue,
without relying on previous conversation/session history.

## Project

Automated AI YouTube Shorts Channel.

Goal: a modular, eventually automated system that discovers topics,
researches them, verifies factual accuracy, generates 30–60 second
explainer Shorts, performs quality/policy checks, gets human approval,
publishes to YouTube, and eventually learns from analytics to improve
future topic selection.

## Current architecture

High-level pipeline (top to bottom):

```
Topic Discovery
    ↓
Topic Selection
    ↓
Research
    ↓
Fact Checking
    ↓
Script
    ↓
Script QA
    ↓
Scene Planning
    ↓
Production Plan
    ↓
Video Generation
    ↓
Video QA
    ↓
Policy / Publishing QA
    ↓
Human Approval
    ↓
YouTube
    ↓
Analytics
    ↓
Topic Selection feedback loop
```

The first **ten** stages are implemented (Topic Selection = Stage 1,
Research = Stage 2, Fact Checking = Stage 3, Script Generation = Stage 4,
Script QA = Stage 5, Scene Planning = Stage 6, Video Generation = Stage 7,
Video QA = Stage 8, Policy / Publishing QA = Stage 9, Human Approval =
Stage 10). Everything below Human Approval (YouTube publishing, analytics,
scheduling) is design-only documentation until implemented.

## Stage 1 — Topic Selection

**Status:** COMPLETE

**Implementation location:** `agents/seo/`

Implemented components:

- `TopicCandidate` — signal-scored candidate topic (`agents/seo/models.py`)
- Topic scoring — weighted signals + content scores + optional historical fit
  (`agents/seo/scorer.py`)
- `TopicSelector` — deterministic, configurable exploration/exploitation
  (`agents/seo/selector.py`)
- Cold-start behavior
- Exploration behavior
- Historical data interface — `HistoryProvider` abstraction + mock history
  (`agents/seo/history.py`)
- Mock candidate provider/data (`data/mock/`)
- `TopicSelection` artifact — versioned JSON (`agents/seo/stage.py`,
  `agents/seo/result.py`)
- Configurable scoring — all weights/knobs live in
  `config/topic_selection.yaml`, never hard-coded
- Provenance — every candidate and artifact records who/what produced it

### Cold start

- `historical_fit` is `null` for all candidates.
- selection mode is `cold_start`.
- Exploration occurs across eligible pillars (coverage bias).
- A deterministic seed is currently used for testing/reproducibility.

### Historical data

- `HistoryProvider` abstraction exists; mock history exists in tests.
- Historical contribution to scoring is configurable
  (`scoring.historical_enabled`, `scoring.historical_weight`).
- Future real analytics can replace the mock provider without touching the
  selector.

### Test status

42 Stage 1 tests (`tests/test_candidate_models.py`,
`test_scoring.py`, `test_selection.py`, `test_serialization.py`).

CLI: `python -m agents.seo.topic_selector`.

### Gemini integration — Stage 1: COMPLETE

The **provider foundation** and **Stage-1 candidate sourcing via Gemini** are
implemented, tested, and committed on this branch. Stage 2 (Gemini Research) is
now implemented on this branch too (see below). Stages 3, 4, and 6 are NOT yet
wired to Gemini (each keeps its mock-first default). Scope intentionally
limited: Stage 1 candidates and Stage 2 research only; scoring, ranking,
replay, and deterministic validation stay the final gate.

- New package `providers/gemini/` — `errors.py`, `config.py`,
  `structured.py`, `client.py`, `topic_provider.py` (provider-specific code;
  the SDK is imported lazily, never by an `agents/` stage).
- `agents/seo/provider.py` — `TopicCandidateProvider` protocol,
  `MockTopicCandidateProvider`, and `build_topic_candidate_provider()` factory.
  Default provider remains **mock**; the gemini path requires `GEMINI_API_KEY`.
- CLI opt-in: `python -m agents.seo.topic_selector --provider gemini`
  (default `mock`). A Gemini run stamps envelope `produced_by` and per-candidate
  `provenance` (`provider="gemini"`, `model`, `prompt_version`).
- Prompt: `prompts/seo/topic_candidates-v1.txt` (structured-output discipline:
  candidates only, no invented evidence, no fabricated URLs, no research claims).
- Env/config: `GEMINI_API_KEY` (required), `GEMINI_MODEL` (default
  `gemini-3.5-flash-lite`) via `.env`/environment. `.env.example` updated.
- Security: the key never appears in repr, exceptions, logs, or artifacts.
- Tests: `tests/test_gemini_config.py`, `test_gemini_client.py`,
  `test_gemini_topic_provider.py`, `test_topic_candidate_provider.py`,
  `test_topic_selector_cli.py`, `test_gemini_security.py` (all offline;
  no network, no real key). Real-network integration is gated by
  `GEMINI_INTEGRATION_TESTS=1` (`tests/test_gemini_integration.py`).

### Test status (updated)

Stage 1 now includes 42 original + 38 Gemini/provider tests =
**80 Stage-1-related tests**; the full suite (including the new Stage-2 Gemini
tests below) is **1084 passed, 3 skipped** at the Stage-2 checkpoint
(`python -m pytest tests -q`).

## Stage 2 — Research Pipeline

**Status:** COMPLETE

**Implementation location:** `agents/research/`

Implemented components (`agents/research/models.py`, `classifier.py`,
`provider.py`, `agent.py`, `stage.py`, `research_agent.py`):

- `ResearchPackage` — the research artifact (claims + sources + status +
  provenance)
- `FactClaim` — one evidence-backed claim
- `Source` — origin metadata + credibility tier
- `ClaimDate` — dated, typed date entries on a claim
- `ResearchProvider` protocol — swappable interface
- `MockResearchProvider` — deterministic, network-free implementation
- `ResearchAgent` — runs the stage against any `ResearchProvider`
- Research status evaluation
- Deterministic classifier (rule/cue-word based; no LLM)
- Provenance
- JSON serialization/deserialization (round-trip with validation)
- Claim → source traceability (enforced by package validation)

### Research statuses

- `provisional`
- `insufficient_evidence`
- `ready_for_fact_check`

### Important

Research does **NOT** establish that a claim is true. All research claims
remain provisional. Final verification belongs exclusively to Stage 3.

### Claim types

- `fact`
- `inference`
- `explanation`
- `speculation`
- `opinion`

### Source credibility tiers

- `primary`
- `high_quality_secondary`
- `discovery`

### Claim date kinds

- `publication_date`
- `discovery_date`
- `announcement_date`
- `event_date`

### IDs

Claim/source IDs use deterministic prefixes + hashing:
- Source: `src_<sha1(url)[:10]>`
- Claim: `cl_<sha1(text)[:10]>`
- Package: `rp_<uuid[:12]>`

### Claim → source audit chain

```
script sentence → claim_id → source_id → original source
```

`FactClaim.source_ids` must resolve to `Source`s inside the same package;
`ResearchPackage.validate()` rejects dangling references.

### Provider status

Stage 2 has **two** research providers behind the `ResearchProvider` protocol:

- `MockResearchProvider` — deterministic, network-free, the **default**.
- `GeminiResearchProvider` — Gemini + Google Search grounding (opt-in,
  **implemented and tested on this branch**; see `GEMINI_PROVIDER_DESIGN.md`).

### Gemini integration — Stage 2: COMPLETE

The Stage-2 real research provider is implemented, offline-tested, and wired
into the research CLI on this branch (not committed yet):

- `providers/gemini/research_provider.py` — `GeminiResearchProvider` +
  `ResearchResultSchema` (sources + claims, Pydantic-validated against the
  existing research contract: claim types, credibility tiers, date kinds).
- `providers/gemini/client.py` — `GeminiClient.generate_research(...)` with
  **Google Search grounding** (+ optional **URL Context** retrieval, off by
  default) and the same retry/error discipline as `generate_structured`; returns
  `GroundedStructured` (parsed data + grounding chunks + URL-context hits).
- Prompt: `prompts/research/facts-v1.txt` (`PROMPT_VERSION = "research/facts-v1"`).
- Evidence boundary: **only URLs the retrieval metadata actually returned become
  sources.** Model-proposed URLs without grounding are dropped (never invented);
  no grounding at all → no candidates → the package honestly lands on
  `insufficient_evidence`. Claims live inside a source, and duplicate claim text
  is collapsed (claim ids derive from text).
- `agents/research/provider.py` — `build_research_provider(name, ..., env=...)`
  factory. Default (no/empty/`mock`) → `MockResearchProvider` (key- and
  SDK-free); `gemini` requires `GEMINI_API_KEY` (raises
  `GeminiConfigurationError` otherwise, never a silent fallback).
- CLI opt-in: `python -m agents.research.research_agent --provider gemini`
  (default `mock`; optional `--enable-url-context`). A Gemini run stamps
  envelope `produced_by` and replaces `research_notes[0]` with an honest
  grounding note ("every source URL comes from retrieval metadata …"). Provider
  config errors exit 2 without writing an artifact (mirrors
  `agents/seo/topic_selector.py`).
- Security: the API key never leaks into artifacts, exceptions, repr, or logs;
  importing `providers.gemini.*` or building the provider never imports the SDK
  (subprocess-verified); the mock/default path works with **no key and no
  network** (sockets-blocked test).
- Tests (all offline): `tests/test_gemini_research_client.py`,
  `tests/test_gemini_research_provider.py`,
  `tests/test_gemini_research_security.py`,
  `tests/test_research_provider_factory.py`,
  `tests/test_research_agent_cli.py`; grounding-aware fakes added to
  `tests/gemini_test_utils.py`. Real-network Stage-2 integration is gated by
  `GEMINI_INTEGRATION_TESTS=1` (`tests/test_gemini_integration.py`) and was
  **NOT run** (no `GEMINI_API_KEY`).

### Firecrawl retriever layer — Stage 2: COMPLETE

Real web retrieval behind the same `GeminiResearchProvider`: an optional
`--retriever firecrawl` opt-in routes research through the **Firecrawl Search
API** (full-page Markdown in the same call) → Gemini structured extraction.
Implemented, offline-tested, and wired into the research CLI on this branch
(not committed yet):

- `providers/firecrawl/` — new provider package (**SDK isolated, import-lazy**,
  same discipline as `providers/gemini/`): `errors.py` (`FirecrawlError`
  hierarchy), `config.py` (`FirecrawlConfig`, **keyless** — `FIRECRAWL_API_KEY`
  optional, `from_env`, redacted repr, bounded defaults), `client.py` (lazy
  `from firecrawl.v2 import FirecrawlClient` — keyless-capable, the v2 Search
  API home; bounded retries, error classification incl.
  missing-SDK → `FirecrawlConfigurationError`, tolerant `.web`/`.data` bucket
  normalization, `FirecrawlResult`), `retriever.py`
  (`FirecrawlResearchRetriever`, domain allow/deny, content truncation at
  `DEFAULT_MAX_CONTENT_CHARS=6000`, maps `Firecrawl*Error` →
  `ResearchProviderError` at the `agents/` boundary).
- `agents/research/retriever.py` — `ResearchRetriever` protocol +
  `RetrievedSource` (the evidence-boundary type) + deterministic
  `MockResearchRetriever` (offline fixtures).
- `providers/gemini/research_provider.py` — `retriever` parameter: retrieval
  path uses `prompts/research/facts-retrieval-v1.txt`, stamps
  `produced_by.retrieved_by`, slices results to `max_retrieved_sources` (4),
  and renders retrieved content into the prompt; **grounding path is
  byte-for-byte unchanged when `retriever=None`**.
- `agents/research/provider.py` — `build_research_provider(..., retriever=...)`:
  `mock` + `--retriever` → `ResearchProviderError`; `firecrawl` builds the
  retriever **keyless and SDK-free** (import-isolation subprocess-verified).
- `agents/research/research_agent.py` — `--retriever firecrawl`; validation
  exits 2 without writing an artifact.
- Evidence boundary preserved: **only URLs the retriever actually returned may
  become sources**; empty retrieval → no candidates →
  `insufficient_evidence`; key never in repr/exceptions/logs/artifacts.
- Tests (all offline, 66 new): `tests/firecrawl_test_utils.py`,
  `tests/test_firecrawl_config.py`, `test_firecrawl_client.py`,
  `test_firecrawl_retriever.py`, `test_gemini_research_provider_retrieval.py`,
  `test_research_retriever_cli.py` + extended `test_research_provider_factory.py`.
  Real-network integration is gated by `FIRECRAWL_INTEGRATION_TESTS=1`
  (`tests/test_firecrawl_integration.py`); the live Firecrawl verification is
  recorded in the "Firecrawl 4.44.0 repair + real verification" block below.

CLI: `python -m agents.research.research_agent --provider gemini --retriever firecrawl`
(`FIRECRAWL_API_KEY` optional — keyless free tier; requires `GEMINI_API_KEY`).

#### Firecrawl 4.44.0 repair + real verification (2026-09-23)

- **Failure observed:** the first real run (`--provider gemini --retriever
  firecrawl`, run-id `...-firecrawl`) failed with `Firecrawl retrieval failed
  ...: Unexpected Firecrawl provider failure (ValidationError)`.
- **Root cause (confirmed by SDK inspection + offline reproduction):** the
  installed `firecrawl==4.44.0` raises a pydantic `ValidationError` *inside*
  `FirecrawlClient.search` while parsing the `/v2/search` response —
  `SearchResultWeb.url` is required, and a non-Document web hit missing a
  top-level `url` reproduces `ValidationError: url Field required`
  deterministically offline. The request shape was proven valid (builds and
  sends as `ignoreInvalidURLs`/`scrapeOptions` etc.); it is a data-dependent
  response-parsing fault, so it did **not** reproduce across 9 live calls for
  the actual topic after the fix.
- **Second confirmed 4.44.0 incompatibility:** the top-level
  `from firecrawl import Firecrawl` class **requires an API key** and raises
  `ValueError: No API key provided` in keyless mode — breaking the documented
  keyless path. The adapter now resolves `firecrawl.v2.FirecrawlClient`
  (keyless-capable; the supported home of the v2 Search API / `SearchData.web`).
- **Adapter fix (`providers/firecrawl/client.py`):** v2 client resolution +
  `_normalize_item` handles hydrated v2 `Document` hits (real responses carry
  `url=''`/`title=''` top-level; url/title/description resolve via
  `metadata.source_url`/`.title`, so markdown content is no longer dropped).
- **Tests:** +6 (v2 `Document` shapes, metadata fallbacks, keyless/keyed v2
  client resolution). Full suite at that point: **1084 passed, 3 skipped**.
- **Real verification (2026-09-23):** a live `FirecrawlResearchRetriever`
  retrieval returned 5 real sources (youtube/instructables/acsh/reddit) with
  url, title, publisher, domain, and truncated markdown
  (`…[truncated]`; one reddit hit with empty markdown). `retrieved_at` remains
  `""` by design (minor limitation).
- **Gemini leg of the end-to-end CLI smoke (run-id `857acc781211-firecrawl-fixed`)
  blocked by Google-side throttling, not a code defect:** `gemini-3-flash` →
  HTTP 429 `generate_content_free_tier_requests` quota (limit 20);
  `gemini-2.5-flash` → HTTP 404 (deprecated for new users; API recommends
  `gemini-3.6-flash`); `gemini-3.6-flash` → HTTP 503 high demand. The CLI fails
  cleanly in every case (exit 2, typed `GeminiAPIError`, no artifact written).
  Re-run the CLI command when Google's free-tier quota/demand resets.

### Test status

80 Stage 2 tests (`tests/test_research_models.py`,
`test_research_classifier.py`, `test_research_provider.py`,
`test_research_agent.py`) + **54 new offline Stage-2 Gemini/provider tests**
listed above.

CLI: `python -m agents.research.research_agent`
(`--provider mock|gemini`, default mock).

## Documentation status

Synchronized with the implementation (recently updated to remove stale
design-only assumptions and match the implemented contracts):

- `docs/DATA_CONTRACTS.md`
- `docs/RESEARCH_PIPELINE.md`
- `docs/FACT_VERIFICATION.md`
- `README.md`

`docs/DATA_CONTRACTS.md` is the authoritative contract reference for every
artifact shape (`Source`, `FactClaim`, `ResearchPackage`, `TopicCandidate`,
`TopicSelection`, `FactCheckResult`, `Script`, ...).

## Current verification

Full suite: **1133 tests passing, 4 skipped** (`python -m pytest tests -q`):
42 Stage 1 + 80 Stage 2 + 79 Stage 3 + 102 Stage 4 + 92 Stage 5 + 120 Stage 6 +
117 Stage 7 + 74 Stage 8 + 140 Stage 9 + 123 Stage 10 = 920 deterministic
tests, plus **207 offline Gemini/Firecrawl provider tests** (38 Stage-1
foundation/topic + 54 Stage-2 Gemini research + 66 Firecrawl retriever layer +
49 Stage-4 Gemini script provider/CLI — `test_script_provider_factory.py`,
`test_mock_script_provider.py`, `test_gemini_script_provider.py`,
`test_gemini_script_security.py`, `test_script_cli.py`).
The 4 skips are the real-network integration tests (Gemini Stage 1 + Stage 2 +
Stage 4 + Firecrawl), gated by `GEMINI_INTEGRATION_TESTS=1` /
`FIRECRAWL_INTEGRATION_TESTS=1`.

## Stage 3 — Fact Checking

**Status:** COMPLETE

**Implementation location:** `agents/qa/fact_checker/`

Implemented components:

- `Evidence` — content excerpt retrieved from a source (≠ source URL)
  (`models.py`), with deterministic id `ev_<sha1(source_id:excerpt)[:10]>`
- `ClaimVerdict` / `FactCheckResult` — versioned Stage-3 artifact
  (`models.py`), id `fc_<sha1(run_id:package_id)[:12]>`
- `EvidenceProvider` protocol + `MockEvidenceProvider` — deterministic,
  network-free, fixture-backed (`evidence.py`)
- `Verifier` protocol + `MockVerifier` — rule-based claim↔evidence coverage
  matcher; the future LLM boundary (`verifier.py`)
- `FactCheckConfig` / `evaluate_claim` / `aggregate_package` — deterministic
  gates + confidence + package verdict (`gates.py`)
- `FactChecker` — ResearchPackage → evidence → verifier → gates →
  FactCheckResult; drops + records malformed/dangling evidence
  (`checker.py`)
- `run_fact_check` stage entrypoint + `__main__` CLI + `pipeline/state.py`
  `Stage.FACT_CHECK`
- Mock evidence fixture `data/mock/fact_check_evidence.json` (clearly marked
  MOCK; aligned with the Stage-2 mock research fixtures)

### Behavior

- Verdicts: `PASS` / `REVIEW` / `REJECT`; rank `PASS < REVIEW < REJECT`;
  gates only downgrade, never upgrade to PASS.
- Critical claims need ≥2 independent non-discovery sources; critical
  `REVIEW` → package `REVIEW`, critical `REJECT` → package `REJECT`.
- `approved_claims` is exactly what Stage 4 may consume; rejected/REVIEW
  claims are excluded.
- Documented policy: `docs/FACT_VERIFICATION.md`, `docs/DATA_CONTRACTS.md` §7.

### Test status

79 Stage 3 tests (`tests/test_fact_check_models.py`,
`tests/test_fact_check_evidence.py`, `tests/test_fact_checker.py`, plus
`tests/fact_check_test_utils.py` builders). CLI:
`python -m agents.qa.fact_checker` (writes `data/<run_id>/research/fact_check_result.json`
+ run state).

## Stage 4 — Script Generation

**Status:** COMPLETE

**Implementation location:** `agents/script/`

Implemented components:

- `ScriptLine` / `Script` — versioned Stage-4 artifact (`models.py`), id
  `sc_<sha1(run_id:topic_id)[:12]>`, lines `ln_00, ln_01, ...`
- `ScriptGenerator` protocol + `MockScriptGenerator` — deterministic,
  network-free generator (`script_generator.py`); the future LLM boundary
- `ScriptConfig` — wpm / duration bounds / min approved claims /
  `allow_review_content` (`script_generator.py`)
- `validate_script` — deterministic validator (`validator.py`): factual
  lines must carry `claim_refs` to approved claims only; REVIEW/REJECT /
  unknown / unapproved refs rejected; exact used/unused approved-claim
  tracking; token-coverage semantic check; hedge preservation; no
  unsupported certainty strengthening; duration bounds
- `run_script_generation` + `approved_claims` whitelist join (`stage.py`)
- CLI `python -m agents.script.script_generator`
  (writes `data/<run_id>/scripts/script.json` + run state)
- Mock fixture `data/mock/fact_check_result.json` — deterministic MOCK
  result authoring 3 approved octopus claims for script demos (clearly
  marked MOCK; claim 4 stays REVIEW, claim 5 stays REJECT)

### Behavior

- Consumes ONLY `FactCheckResult.approved_claims`; rejected/REVIEW claims
  are never materialized (the generator re-filters even if handed the full
  package — defense in depth).
- Package `REJECT` → `blocked`; package `REVIEW` → `blocked` unless
  `allow_review_content`; too few approved claims or too-short material →
  structured `insufficient_content` (never invents filler).
- Lines: hook (approved claim, curiosity-framed), context (topic title
  bridge), one near-verbatim line per approved claim (purpose from claim
  type), transition/CTA (inert boilerplate), conclusion (restates the first
  claim's payload, with refs).
- Duration estimated deterministically: Σ line words ÷ wpm × 60
  (default 150 wpm); target 30–60s.
- Documented policy: `docs/DATA_CONTRACTS.md` §8.

### Test status

53 Stage 4 tests (`tests/test_script_models.py`,
`tests/test_script_generator.py`, `tests/test_script_validator.py`, plus
`tests/script_test_utils.py` builders). CLI:
`python -m agents.script.script_generator`.

### Gemini provider slice (package-based path)

The deterministic mock path above is unchanged. A real Gemini provider was
added behind a new seam (`agents/script/provider.py`) plus a CLI opt-in, both
fully wired and smoke-tested:

- `ScriptProvider` protocol + `MockScriptProvider` + `build_script_provider`
  (`agents/script/provider.py`). Has no fact-check result: the WHOLE package
  claim set is the working set; scripts carry `result_id=""` and bind to the
  research package via `package_id`.
- `GeminiScriptProvider` (`providers/gemini/script_provider.py`) renders
  `prompts/script/script-v1.txt` (claims-only prompt; `purpose` is a strict
  `Literal`), uses the shared `GeminiClient.generate_structured` with a bounded
  malformed-output retry, and maps structured output into the existing `Script`
  contract (deterministic durations, exact used/unused tracking, provenance).
- New deterministic gate `validate_script_from_package` (`validator.py`)
  mirrors `validate_script` against the package: unknown refs, off-claim
  wording, dropped hedges, and strengthened certainty all fail closed as a
  `blocked` generation — never a persisted artifact.
- CLI: `python -m agents.script --provider <mock|gemini> --research
  <research_package.json>`; `--provider gemini` requires `--research` (exit 2
  otherwise). Exit codes: insufficient 0, blocked 1, provider/config error 2.
- New offline tests: `tests/test_script_provider_factory.py`,
  `tests/test_mock_script_provider.py`, `tests/test_gemini_script_provider.py`,
  `tests/test_gemini_script_security.py`, `tests/test_script_cli.py`; real
  key-gated case in `tests/test_gemini_integration.py`.
- Real smoke produced
  `data/857acc781211-firecrawl-lite/scripts/script.json` (see
  `docs/GEMINI_PROVIDER_DESIGN.md` §K for the verified provenance/summary).

## Stage 5 — Script QA

**Status:** COMPLETE

**Implementation location:** `agents/qa/script/`

Implemented components:

- `QAIssue` / `ClaimCoverage` / `QACheckRecord` / `ScriptQAConfig` /
  `ScriptQAResult` — versioned Stage-5 artifact (`models.py`); issue ids
  `qa_00, qa_01, ...`, result id `sqa_<sha1(run_id:script_id)[:12]>`; 13 QA
  categories; 10 scoring dimensions under `QA_DIMENSIONS`; documented
  `DEFAULT_SCORE_WEIGHTS` (sum = 1)
- 13 deterministic check functions, no LLM, no network (`checks.py`):
  factual fidelity, claim traceability, claim coverage, length/duration,
  pacing, clarity, repetition, structure, hook quality, uncertainty
  preservation, misleading language, CTA quality, format/Shorts suitability;
  registered in `CHECKS` for independent rule additions
- Verdict rules (`validator.py`): any severity-ERROR issue with `gate=hard`
  → **REJECT**; any soft ERROR → **REVISE**; otherwise → **PASS**. The 9 hard
  gates cover: no factual traceability; REJECTED/REVIEW/unknown claim refs;
  material contradiction; certainty strengthening; invented factual detail;
  empty/unusable script; package not approved for scripting
- Deterministic weighted quality score (`scorer.py`):
  dimension = `clamp(100 - penalties, 0, 100)`; ERROR −45 hard / −20 soft,
  WARNING −6, INFO −1; overall = weighted mean rounded to 1dp; score is an
  internal quality number, never a performance prediction
- Coverage accounting — `approved_claim_coverage`: ratio + used/unused/
  critical-unused ids; only *critical* unused claims create requirements
- `evaluate_script` deliberately does NOT call `script.validate()`: Stage 5
  is a gate over arbitrary artifacts, so corrupt inputs produce issues rather
  than exceptions
- `run_script_qa` + whitelist join `approved_claims` (same as Stage 4) +
  `generate_mock_script` (`stage.py`); CLI `python -m agents.qa.script`
  (writes `data/<run_id>/scripts/script_qa_result.json` + run state advanced
  to `Stage.SCRIPT_QA`)
- Mock fixture `data/mock/script.json` — deterministic MOCK Script artifact
  (45.2s, 3 approved claims used, clean PASS at score 100.0)

### Behavior

- QA never re-runs research and never consults unapproved claims as fact:
  the approved claims from Stage 3 remain the factual source of truth.
- QA never rewrites the script — it emits the verdict plus actionable
  issues/warnings/infos with line/claim ids for a future revision agent.
- Uncertainty preservation is a hard gate and a dedicated score dimension:
  dropping a hedge or adding unsupported certainty words silently changes a
  claim's meaning.
- Intentional hook/conclusion reinforcement and repeat of the first claim are
  NOT flagged as repetition failures.
- Optimized phrase/hedge/negation sets are disjoint so a single word is never
  double-reported at two severities; `appears`/`seems` count as hedges only
  when followed by "to".
- Documented policy: `docs/DATA_CONTRACTS.md` §8.1.

### Test status

92 Stage 5 tests (`tests/test_script_qa_models.py`,
`tests/test_script_qa_checks.py`, `tests/test_script_qa_scorer.py`,
`tests/test_script_qa_validator.py`, `tests/test_script_qa_stage.py`, plus
`tests/script_qa_test_utils.py` builders). CLI:
`python -m agents.qa.script`.

## Stage 6 — Scene Planning

**Status:** COMPLETE

**Implementation location:** `agents/scene_planner/`

Implemented components:

- `Scene` / `GlobalStyle` / `OnScreenText` / `AssetRequirement` /
  `ContinuityAsset` / `ScenePlanningConfig` / `ProductionPlan` — versioned
  Stage-6 artifact (`models.py`), the stable **CONTENT → VIDEO boundary**; id
  `pp_<sha1(run_id:script_id)[:12]>`, scenes `scn_01..scn_NN`; envelope
  header + `data` payload with per-scene `provenance` (planning rule +
  embedded claim texts)
- Controlled, provider-independent vocabulary (`models.py`): `visual_type` ∈
  {generated_video, generated_image, stock_asset, diagram, text_only,
  motion_graphic, existing_asset, transition}; `transition` ∈ {cut, fade,
  dissolve, zoom, match_cut, none}; `on_screen_text.kind` ∈ {headline,
  emphasis, label, caption} (label/caption are factual → require claim_refs)
- `ScenePlanner` protocol + `MockScenePlanner` — deterministic, offline
  planner over approved claim records only (`planner.py`); groups lines into
  scenes (7 scenes for the octopus demo), mirrors the script's line timings,
  and appends grounded visual treatments. No web, no LLM, no API, no GPU
- `validate_production_plan` — deterministic validator with **21 rules
  r01–r21** (`validator.py`): non-empty, unique/ordered scenes, durations,
  script-duration tolerance, line coverage, claim whitelist (unknown/
  unapproved/rejected/review each have their own rule), factual visuals +
  factual text traceability, prompts where required, valid types, no
  provider-specific wording/keys in the contract (plan-level provenance
  record excluded), JSON round-trip; **r21 determinism** is a planner-level
  guarantee asserted in tests
- `run_scene_planning` + whitelist join `approved_claims` (same as Stages
  4/5) + `generate_mock_qa_result` / `generate_mock_production_plan`
  (`stage.py`); CLI `python -m agents.scene_planner` (writes
  `data/<run_id>/production/production.json` + run state advanced to
  `Stage.SCENE_PLANNING`); blocked gate or invalid plan → exit non-zero, no
  artifact written
- Mock fixture `data/mock/script_qa_result.json` — deterministic MOCK
  ScriptQAResult (run `mock`, PASS score 100.0, genesis 2026-09-20) so the
  demo run `data/sp-demo/` reproduces the octopus plan

### Behavior

- **QA gate:** `ScriptQAResult.overall_verdict` must be `PASS`; REVISE/REJECT
  block planning before anything is generated (no silent fallback).
- **Grounding:** the planner receives only approved claim records; every
  factual scene and every factual `label`/`caption` traces to an approved
  claim (scene → script line → claim → evidence → source). Scene provenance
  embeds the claim texts for audit.
- **Duration:** scene rules sum the mapped lines' timings; total must stay
  within `duration_tolerance_seconds` (0.5) of the script duration.
- **Negative prompts are visual-quality ONLY** — factual constraints are
  never placed in negative prompt text.
- **Provider-agnostic:** prompts/descriptions carry no model names, no API
  fields, no provider syntax; the validator scans the contract for leaked
  provider vocabulary (runway/veo/sora/luma/...).
- Documented policy: `docs/DATA_CONTRACTS.md` §10; design history incl. the
  superseded `ScenePlan` in §9.

### Test status

120 Stage 6 tests (`tests/test_scene_planning_models.py`,
`tests/test_scene_planning_validator.py`,
`tests/test_scene_planning_planner.py`,
`tests/test_scene_planning_stage.py`, plus
`tests/scene_planning_test_utils.py` builders). CLI:
`python -m agents.scene_planner`.

## Stage 7 — Video Generation

**Status:** COMPLETE

**Implementation location:** `agents/video_generation/`

Implemented components:

- `GenerationRequest` / `GeneratedArtifact` / `SceneGenerationResult` /
  `GenerationFailure` / `VideoGenerationConfig` / `VideoGenerationResult` —
  versioned Stage-7 model surface (`models.py`): the provider-independent
  answer to `ProductionPlan` — one per-scene record with the request, the
  produced artifact (always clearly `mock=True`, empty `uri`, MOCK note for
  this first implementation), and per-scene status. Result id
  `vg_<sha1(run_id:production_plan_id)[:12]>`, mock artifact ids
  `mockva_<sha1(run_id:scene_id)[:12]>`; envelope header + `data` payload
  with `provenance` (generated_by `video_generation_agent_v1`,
  source/model `MockVideoGenerator`, versioned ruleset
  `video_generation.v1`)
- `VideoGenerator` protocol — the provider-independent generation boundary
  (`generator.py`); a future Colab / GPU / open-source worker implements the
  same `generate(plan, config, previous, run_id, created_at)` signature and
  drops in without changing the stage runner, validator, resume logic, or CLI
- `MockVideoGenerator` — deterministic, offline, GPU-free, network-free,
  API-free FIRST implementation (`mock_generator.py`); consumes ONLY the
  Stage-6 `ProductionPlan` (never invents facts), simulates failures
  deterministically from `config.failure_scenarios`, and gates un-generatable
  requests (`request_issue`: generative type without a `generation_prompt`,
  non-positive duration, blank description → non-retryable failure)
- `validate_video_generation_result` — deterministic validator with **13
  rules g01–g13** (`validator.py`): non-empty, unique/ordered scenes, valid
  statuses, result↔plan scene correspondence (no phantom, no missing), order
  matching, request/artifact timing preservation, completed-scene artifact
  invariants, failed-scene message/retry/attempt invariants (a retryable
  flag at the retry limit is an error), `overall_status` consistency,
  provenance, **provider isolation** (scans keys AND values against the
  denylist: runway/veo/sora/luma/algorithms/GPU/CUDA/Colab/cloud/API keys/
  auth vocab — excludes the top-level provenance audit record), JSON
  round-trip (skipped for structurally broken results)
- Resume/retry semantics (`stage.py` + `mock_generator.py`): completed
  scenes are preserved verbatim and never regenerated; failed retryable
  scenes are retried while `attempts < config.max_attempts`; pending scenes
  are processed; non-retryable failures persist; an exhausted retry budget
  re-records a failure as non-retryable. A `previous` result for a different
  `production_plan_id` raises `VideoGenerationError` before anything runs
- `run_video_generation` — stage entrypoint returning a `VideoGenerationOutcome`
  (`generated=True` only after the result validates against the plan) +
  `generate_mock_production_plan` re-export + `load_production_plan` /
  `load_video_generation_result` loaders (`stage.py`)
- CLI `python -m agents.video_generation` (`__main__.py`; flags
  `--plan` default `data/mock/production.json`, `--run-id`, `--previous`,
  `--out`) — writes `data/<run_id>/videos/video_generation_result.json` +
  run state advanced to `Stage.VIDEO_GENERATION`; exit 0 success, 1 blocked /
  validation failed (no artifact), 2 error (e.g. mismatched previous)
- Mock fixture `data/mock/production.json` — deterministic MOCK
  ProductionPlan (run `mock`, genesis 2026-09-20, 7 octopus scenes) so the
  demo run `data/vg-demo/` reproduces a full completed result without any
  provider

### Behavior

- **Boundary:** Stage 7 consumes ONLY the Stage-6 `ProductionPlan`
  (`production.json`) — the stable CONTENT → VIDEO contract
  (`docs/DATA_CONTRACTS.md` §10) — and produces a provider-independent
  `VideoGenerationResult`. It never calls a video API and never renders bytes.
- **Mock honesty:** every artifact record is flagged `mock=True` with an
  empty `uri` and a note stating no video bytes were produced. Nothing is ever
  made to look like real produced video.
- **Resume/retry:** completed work is preserved exactly; only retryable
  failures are retried, bounded by `max_attempts` (3 default); permanent
  failures persist. Retry happens across runs (resume via `--previous`), not
  inside a run, so deterministic failure scenarios are testable offline.
- **Provider isolation:** the artifact contract must not leak provider names,
  API keys, GPU/CUDA requirements, cloud vendors, or runtime assumptions; the
  validator scans key names and string values against `provider_isolation_denylist`.
- **Determinism:** identical plan + config + inputs → identical result
  (`vg_...` / `mockva_...` ids are stable sha1 digests).
- **Safety:** the package statically contains no network modules, no GPU/CUDA
  references, no secret/env reads; tests prove a full generation still
  succeeds with sockets blocked and that it never writes files.
- Documented policy: `docs/DATA_CONTRACTS.md` §10 (ProductionPlan boundary);
  future real worker design notes remain in `docs/VIDEO_PIPELINE.md`.

### Test status

117 Stage 7 tests (`tests/test_video_generation_models.py`,
`tests/test_video_generation_validator.py`,
`tests/test_video_generation_generator.py`,
`tests/test_video_generation_stage.py`,
`tests/test_video_generation_safety.py`, plus
`tests/video_generation_test_utils.py` builders). CLI:
`python -m agents.video_generation`.

## Stage 8 — Video QA

**Status:** COMPLETE

**Implementation location:** `agents/qa/video/`

Implemented components:

- `VideoQAIssue` / `VideoQACheckRecord` / `VideoQAConfig` / `VideoQAResult` —
  versioned Stage-8 artifact (`models.py`): the quality gate over the Stage-7
  `VideoGenerationResult`. Issue ids `vqa_00, vqa_01, ...` (stable,
  content-independent), result id
  `qa_<sha1(run_id:production_plan_id)[:12]>`, envelope header (`stage="video"`)
  + `data` payload; `STATE_STAGE = Stage.QUALITY_CHECK` records the stage-label
  ↔ pipeline-state mapping (see enum decision below)
- **15 deterministic, network-free, GPU-free, API-free checks** (`checks.py`,
  registered in `CHECKS`): duration window (preferred 30–60s, hard 15–90s),
  aspect-ratio metadata (9:16 hard / 1080x1920@30fps soft, from the plan's
  global style), scene coverage, pending/failed scenes, scene ordering, timing
  fidelity, artifact assignment, structure integrity, reference integrity,
  provider isolation (scans the result contract, excluding audit provenance),
  generation validation (re-runs Stage-7's own `validate_video_generation_result`
  as a hard gate), overall-status consistency, mock recognition (non-mock
  artifact = hard error; mock artifact with a URI = soft error), and media
  honesty (no real/uploaded/published video may be claimed). Several checks are
  deliberately metadata/contract-level because no assembled media bytes exist.
- Verdict rules (`validator.py`), mirroring §5/§8.1: any severity-ERROR issue
  with `gate=hard` → **REJECT**; any soft ERROR → **REVIEW**; otherwise →
  **PASS**. No score field (records nothing §11 has no notion of); `evaluate_video`
  deliberately does NOT call `result.validate()` so structurally broken results
  are graded, not raised.
- `run_video_qa` (`stage.py`): consumes the Stage-7
  `VideoGenerationResult` + its Stage-6 `ProductionPlan`; a result that targets
  a DIFFERENT `production_plan_id` raises `VideoQAError` before anything runs.
- CLI `python -m agents.qa.video` (`__main__.py`; flags `--result` default
  `data/mock/video_generation_result.json`, `--plan` default
  `data/mock/production.json`, `--run-id`, `--previous` chains
  `resumed_from_qa_result_id`, `--out`) — writes
  `data/<run_id>/production/qa_result.json` + run state advanced to
  `Stage.QUALITY_CHECK`; exit 0 PASS, 1 REVIEW/REJECT (artifact still written),
  2 error.
- Mock fixture `data/mock/video_generation_result.json` — deterministic MOCK
  Stage-7 result (run `mock`, 7/7 completed, 45.2s, PASS) generated from
  `data/mock/production.json` so the demo run `data/vqa-demo/` reproduces a
  clean PASS end-to-end.

### Behavior

- **Boundary:** Stage 8 is VIDEO QA, NOT video assembly. It never inspects
  pixels/audio/subtitles (none exist yet), never calls a provider, and never
  makes anything look like real produced video. It grades what the Stage-7
  records PROVE, keeping the trace chain
  ProductionPlan → scene → VideoGenerationResult → artifact → QA issue.
- **Verdict gating mirrors Stage 5:** hard errors REJECT, soft errors REVIEW,
  clean PASS; per-check `outcome` is independently derivable from the recorded
  `hard_errors` count.
- **Provider isolation:** the QA artifact itself must not leak provider/API
  vocabulary (registry notes avoid denylist terms), so a re-QA never
  self-flags.
- **Safety:** the package statically contains no network modules, no GPU/CUDA
  references, no secret/env reads, and no Stage-9 vocabulary (video_assembly /
  video_editing / ffmpeg); tests prove a full QA run succeeds with sockets
  blocked and never writes files.
- **Stage label vs. pipeline state:** the artifact header `stage` is the
  generic QAResult label `"video"` (DATA_CONTRACTS.md §11). The numbered Stage 8
  advances pipeline STATE to the existing generic `Stage.QUALITY_CHECK`; no new
  `Stage` enum member was added and none was renamed (`STATE_STAGE` = models.py
  constant, documented in DATA_CONTRACTS.md §11).
- Documented policy: `docs/DATA_CONTRACTS.md` §11; `docs/VIDEO_PIPELINE.md` §6.

### Test status

74 Stage 8 tests (`tests/test_video_qa_models.py`,
`tests/test_video_qa_checks.py`, `tests/test_video_qa_validator.py`,
`tests/test_video_qa_stage.py`, `tests/test_video_qa_safety.py`, plus
`tests/video_qa_test_utils.py` builders). CLI:
`python -m agents.qa.video`.

## Stage 9 — Policy / Publishing QA

**Status:** COMPLETE

**Implementation location:** `agents/qa/policy/`

NOTE: mirrors the QA-stage naming convention (like Stages 5/8), but is a
**deterministic pre-publication POLICY / SAFETY GATE**, not a quality grade.

Implemented components:

- `PolicyIssue` / `PolicyCheckRecord` / `RiskSummary` / `PolicyQAConfig` /
  `PolicyQAResult` — versioned Stage-9 artifact (`models.py`): the risk gate
  over the full chain Script → FactCheckResult → ProductionPlan →
  VideoGenerationResult → VideoQAResult. Issue ids `pol_00, pol_01, ...`,
  result id `pr_<sha1(run_id:target_ref)[:12]>`, envelope header
  (`stage="policy_qa"`) + `data` payload; `STATE_STAGE = Stage.QUALITY_CHECK`
  records the stage-label ↔ pipeline-state mapping (same enum mapping as
  Stage 8; marked decision below)
- `POLICY_RULES` P01–P15 — a versioned, deterministic policy registry
  (`rules.py`, `policy_version="policy_rules_v1"`), Python data (not YAML),
  exposing `POLICY_CHECK_ORDER`, `POLICY_CATEGORIES` and `POLICY_RULESET_NOTE`:
  harmful content (P01, hard markers REJECT / soft REVIEW); dangerous
  instructions with "don't try this at home" negator guard (P02); overstatement
  of weak facts via the recorded `strong_in_claim=none` (P03, soft); definitive
  misinformation framing (P04, soft); "actual footage" claims on mock artifacts
  (P05, hard REJECT); synthetic media of real people/events without disclosure
  (P06, soft — people/event markers only, not generic "realistic"); asset
  rights/provenance (P07); PII/doxxing (P08); spam metadata (P09, ≥2 markers);
  near-duplicate/reused content (P10, overlap ≥0.8 excluding hook/conclusion/
  transition/CTA purposes); advertiser-suitability (P11, soft without a
  disclaimer); sensational title/hook absent from content (P12); provenance
  completeness (P13, soft); unregistered policy version (P14, hard REJECT);
  prior-gate consistency with Stage-8 (P15: missing QA, non-PASS QA, or a
  target_ref mismatch → soft REVIEW — a Stage-8 REVIEW can never silently pass)
- Verdict rules (`validator.py`) mirroring Stages 5/8: any severity-ERROR issue
  with `gate=hard` → **REJECT**; any soft ERROR → **REVIEW**; otherwise →
  **PASS**; uncertainty NEVER resolves to PASS. No score field. `run_checks`
  assigns stable sequential issue ids. Optional deterministic
  `PolicyRiskClassifier` boundary (`classifier.py`): a bounded, context only
  suggestion that can only make the gate STRICTER (a deterministic REVIEW/REJECT
  is never downgraded); default run uses no classifier
- `evaluate_policy` deliberately does NOT validate the upstream artifacts: it
  is a gate over arbitrary record shapes, and corrupt/missing inputs produce
  issues (P13/P15), not exceptions
- `run_policy_qa` (`stage.py`): consumes the Stage-8 `VideoQAResult` + its
  Stage-7 `VideoGenerationResult` + Stage-6 `ProductionPlan` + Stage-5
  `FactCheckResult` + Stage-4 `Script`; a video result that targets a DIFFERENT
  `production_plan_id` than the plan raises `PolicyQAError` before anything
  runs (exit 2)
- CLI `python -m agents.qa.policy` (`__main__.py`; flags `--script`
  `--fact-check --plan --video-result --qa-result`, defaults to the committed
  `data/mock/*`, `--run-id` (default `pol-demo`), `--previous` chains
  `resumed_from_policy_result_id` (loaded as a `PolicyQAResult`), `--policy-version`,
  `--out`) — writes `data/<run_id>/production/policy_qa_result.json` + run
  state advanced to `Stage.QUALITY_CHECK`; exit 0 PASS, 1 REVIEW/REJECT
  (artifact still written), 2 error
- Mock fixture `data/mock/qa_result.json` — deterministic MOCK Stage-8 result
  (run `mock`, PASS, `qa_9b7d098451d3`) generated from
  `data/mock/video_generation_result.json` so the demo run `data/pol-demo/`
  reproduces a clean PASS end-to-end
- Safety (`tests/test_policy_qa_safety.py`): the package statically contains
  no network modules, no GPU/CUDA/hardware tokens (identifier-level audit, so
  substring artifacts like "tpu" in "purpose" are not false positives), no
  secret/env reads, and no Stage 10/11 vocabulary (oauth / youtube upload /
  scheduling / analytics); tests prove a full policy evaluation succeeds with
  sockets blocked and never writes files

### Behavior

- **Boundary:** Stage 9 is a PRE-PUBLICATION POLICY GATE, not a YouTube
  compliance engine and not the publishing stage. It grades what the recorded
  artifacts PROVE. `POLICY_RULESET_NOTE` states explicitly it is "*Not a
  YouTube or platform compliance guarantee*"; No Stage 10 (human approval) or
  Stage 11 (upload) code exists anywhere in the package.
- **Prior-gate honesty:** a Stage-8 REVIEW or a missing/mismatched QA result is
  a REVIEW under P15 — the policy gate can never quietly convert an unresolved
  quality finding into a publishable PASS.
- **Markers are word-aware:** P05/P06 vocabulary only fires for real
  media/people/event claims (e.g. "actual footage", "the president"), so a
  harmless "realistic rendering of an octopus" stays benign.
- **Stage label vs. pipeline state:** the artifact header `stage` is
  `"policy_qa"` (Stage-9 task label). Numbered Stage 9 advances pipeline STATE
  to the existing generic `Stage.QUALITY_CHECK`; no `Stage` enum member was
  added for policy QA (`STATE_STAGE` constant, documented in §11).
- Documented policy: `docs/DATA_CONTRACTS.md` §11; `docs/POLICY_AND_SAFETY.md`.

### Test status

140 Stage 9 tests (`tests/test_policy_qa_models.py`,
`tests/test_policy_qa_checks.py`, `tests/test_policy_qa_validator.py`,
`tests/test_policy_qa_stage.py`, `tests/test_policy_qa_safety.py`, plus
`tests/policy_qa_test_utils.py` builders). CLI:
`python -m agents.qa.policy`.

## Stage 10 — Human Approval

**Status:** COMPLETE

**Implementation location:** `agents/human_approval/`

The mandatory human decision gate between the completed content pipeline and
any publication. Stage 10 produces a **machine-verifiable record of an
explicit human decision over the exact reviewed artifact versions**. Design:
`docs/HUMAN_APPROVAL_STAGE10_DESIGN.md`; contract: `docs/DATA_CONTRACTS.md`
§12.

Implemented components:

- `ROLES` (7, stable order) + `ROLE_ATTR` / `ROLE_ID_FIELD` mappings
  (`models.py`): `policy_qa`, `video_qa`, `video_result`, `production_plan`,
  `script`, `script_qa`, `fact_check`. `REQUIRED_ROLES` (5) = policy_qa,
  video_qa, video_result, production_plan, script; script_qa/fact_check are
  optional (`--script-qa none` / `--fact-check none`). A missing REQUIRED role
  fails CLOSED for every decision (`HumanApprovalError`, nothing recorded).
- `Approver` / `ReviewedArtifact` / `OverriddenVerdict` / `RequestedChange` /
  `ManualUploadReadiness` / `HumanApprovalResult` — versioned Stage-10
  artifact (`models.py`), id `ha_<sha1(run_id:target_ref)[:12]>`; every
  reviewed artifact records `role`, `artifact_id`, and canonical
  `content_sha256` (SHA-256 over `json.dumps(to_dict(), sort_keys=True,
  separators={",", ":"})`).
- Decision vocabulary `DECISIONS = (PENDING, APPROVED, CHANGES_REQUESTED,
  REJECTED)` — deliberately DISJOINT from the QA PASS/REVIEW/REJECT brand.
  `WRITABLE_DECISIONS` excludes PENDING: the CLI refuses to write "no
  decision" (no written approval = not approved). `is_approved` ⇔
  `decision == APPROVED`.
- Blocking conditions `build_blocking_conditions` (`validator.py`): codes
  `required_artifacts`, `policy_qa_hard_reject`, `video_qa_hard_reject`,
  `video_generation_completed`, `review_overrides`. `APPROVED` requires ALL
  satisfied else `HumanApprovalError` (nothing written); CHANGES_REQUESTED /
  REJECTED are valid under any state and record the unsatisfied conditions
  for audit. CHANGES_REQUESTED needs `changes_requested` requirements or
  `notes`.
- Pairing guards (`stage.py`, fail closed, all before any evaluation): plan id
  == policy `target_ref` == video-QA `target_ref`; video_result
  `production_plan_id` == plan id; video_result result_id == video-QA
  `video_generation_result_id`; script `script_id` == plan `script_id`; QA
  `script_id` == script id; `supersedes_approval_id` != own id.
- Overrides (soft REVIEW only, hard REJECT never): an `--override
  "gate;check;verdict;justification"` is validated against the prior
  REVIEW verdict and auto-bound to the reviewed artifact's `artifact_id`, so
  the record proves exactly which version the override reviewed.
- Freshness / supersede (`validator.py`): `check_approval_freshness(approval,
  current_artifacts)` re-derives the digests and ids of every bound role and
  flags STALE on any mismatch; `is_superseded_by` / `approval_is_effective`
  chain re-review; the CLI `--check --approval <path>` audit mode re-reads the
  current fixtures and reports `digests_fresh` / whether the approval is
  effective and gates-clear without writing anything.
- Provenance honesty: the approver is a human, NOT a model/provider —
  `produced_by`/`provenance` have `provider == model == None`,
  `generated_by="human_approval_gate_v1"`, `source="HumanReviewer"`,
  `prompt_version="human_approval.v1"`.
- CLI `python -m agents.human_approval` (`__main__.py`; defaults to the
  committed `data/mock/*` fixtures; `--decision --approver-id --notes
  --change --override --previous --check --approval --out`; optional
  `--policy-result / --qa-result / --video-result / --plan / --script /
  --script-qa / --fact-check`) — writes
  `data/<run_id>/approval/human_approval_result.json` + run state advanced to
  the existing generic **`Stage.PUBLISH`** (via `STATE_STAGE`; no new `Stage`
  enum member, no upload performed). Exit 0 APPROVED, 1 CHANGES_REQUESTED /
  REJECTED (artifact still written) or STALE in `--check` mode, 2 error
  (missing/incoherent inputs, invalid decision/approver/override, blocked
  APPROVED under a hard gate).
- Committed fixture `data/mock/policy_qa_result.json` — deterministic MOCK
  Stage-9 result (run `mock`, PASS, `pr_9b7d098451d3`) generated by the
  Stage-9 validator over the existing mock fixtures so the demo run
  `data/ha-demo/` reproduces an APPROVED manual-upload-ready end-to-end from
  Stages 1–10 without any provider.
- `manual_upload` readiness: `ready == is_approved`; `title` = Stage-4 Script
  title; note states metadata + manual upload remain human steps.

### Behavior

- **Fixed, honest brand:** the decisions are `APPROVED / CHANGES_REQUESTED /
  REJECTED / PENDING` — never QA's PASS/REVIEW/REJECT — so a "REVIEW" human
  decision can never be mistaken for an override or vice versa.
- **Fail-closed APPROVED:** an approval is only recorded when every hard gate
  is demonstrably clear, and the CLI refuses to write a `PENDING` "no
  decision". Absence of approval is never approval.
- **Exact-version review proof:** `content_sha256` binds the decision to the
  canonical bytes of the artifacts actually on disk; `--check` re-digests and
  reports STALE the moment anything changes (e.g. a later re-render or an
  edited script), independent of human memory.
- **Pairing is the anti-confusion boundary:** the bundle must target one plan
  end to end (policy, video QA, video result, script) or the stage raises
  before a decision can be recorded.
- **Overrides are soft-and-accountable only:** a soft Stage-8 REVIEW can be
  overridden with justification and is bound to the reviewed artifact; a hard
  REJECT can never be overridden into approval.
- **Stage label vs. pipeline state:** the artifact header `stage` is
  `"human_approval"`. Numbered Stage 10 advances pipeline STATE to the
  existing generic `Stage.PUBLISH`; no new `Stage` enum member was added and
  none was renamed.
- **Manual endpoint only:** the package statically contains no network
  modules, no GPU/CUDA tokens, no secret/env reads, no oauth/googleapiclient,
  and no Stage-11 vocabulary (youtube_upload/upload_video/scheduling/
  analytics); the safety suite proves a full approval succeeds with sockets
  blocked, never writes files during evaluation, and never fabricates a
  provider/model or a Stage-11 mention.
- Documented policy: `docs/DATA_CONTRACTS.md` §12;
  `docs/POLICY_AND_SAFETY.md` §7.

### Test status

123 Stage 10 tests (`tests/test_human_approval_models.py`,
`tests/test_human_approval_validator.py`,
`tests/test_human_approval_stage.py`, `tests/test_human_approval_safety.py`,
plus `tests/human_approval_test_utils.py` builders). CLI:
`python -m agents.human_approval`.

## Future stages

Planned, **not implemented**:

- Stage 11 — YouTube Publishing
- Stage 12 — Analytics
- Stage 13 — Automated Scheduling / Orchestration

Do not claim any of these are implemented.

## Important design decisions

1. File-based, versioned JSON contracts are the interface between stages.
2. `production.json` / `ProductionPlan` IS the stable boundary between
   content generation and video generation (implemented in Stage 6).
3. Components should be provider-agnostic and replaceable.
4. The eventual production LLM is NOT the same as OpenCode's coding model.
5. "Big Pickle" is currently used only as the coding agent.
6. No production LLM provider has been selected yet.
7. Google Colab is the initial planned GPU worker, but integration has NOT
   been implemented.
8. Google Drive is the likely initial transport between the local PC and
   Colab, but is NOT implemented.
9. The system initially requires human approval before publishing.
10. The system should eventually learn from channel analytics using
    exploration/exploitation rather than assuming a fixed niche forever.
11. Content is primarily 30–60 second vertical YouTube Shorts.
12. Content pillars: AI, technology, science, space, curiosity / interesting
    phenomena.
13. Content is NOT restricted to recent discoveries. Topics can be:
    evergreen, trending, recent discoveries, historical, surprising facts,
    scientific concepts, technological developments, unusual phenomena.
14. Factual accuracy is a first-class requirement.
15. AI-generated text is never automatically treated as truth.
16. The QA artifact-header `stage` label ("video") is the generic §11 label and
    is NOT a `pipeline/state.py` `Stage` member. Numbered Stage 8 advances
    pipeline STATE to the existing generic `Stage.QUALITY_CHECK`; Stage enum
    members are never added or renamed for QA stages.
17. Stage 9 labels its artifact header `stage` `"policy_qa"` (its task label),
    and like Stages 5/8 advances pipeline STATE to the existing generic
    `Stage.QUALITY_CHECK` — no new `Stage` member was added for policy QA.
18. Stage 9 is a deterministic pre-publication policy/safety GATE with a
    bounded, fully optional classifier boundary. It is explicitly NOT a
    YouTube/platform compliance guarantee, and human approval / publishing
    remain later stages that it never implements.
19. Stage 10 (human approval) brands its decisions with a vocabulary DISJOINT
    from QA's PASS/REVIEW/REJECT (`APPROVED` / `CHANGES_REQUESTED` /
    `REJECTED` / `PENDING`), binds every reviewed artifact by stable id +
    canonical `content_sha256`, fail-closes: APPROVED needs all gates clear
    and PENDING is never writable, and advances pipeline STATE to the existing
    generic `Stage.PUBLISH` while performing NO upload (the endpoint is a
    manual-upload readiness flag).

## Known future decisions (unresolved)

Do not decide these unless the decision is already implemented:

- YouTube data source
- Production LLM provider (beyond the Stage-1/2/4 Gemini provider)
- Real analytics provider
- Colab transport implementation
- Topic scoring weight tuning
- Exploration tuning
- Fact-check corroboration thresholds
- Audio/TTS provider
- Subtitle generation method
- Video generation model/provider
- YouTube publishing implementation
- Analytics storage implementation

## Continuity checkpoint — end of session

Single snapshot so a **fresh OpenCode session** — with **no reliance on any
previous OpenCode conversation history** — knows exactly where the project
stopped without guessing.

### Repository state

- **Branch:** `feat/real-gemini-pipeline`
- **HEAD commit:** `b57f2ee` — `feat: implement real gemini stages 1 and 2`
- Stage 1 Gemini (provider foundation + topic candidates) and Stage 2 Gemini
  Research (grounding + Firecrawl retriever) are committed at HEAD.
- The current working tree adds **Stage 4 — Gemini Script Provider**
  (implemented and fully offline-tested below) plus this documentation
  checkpoint. All of it is deliberately left **uncommitted** for the next
  session to verify and commit when asked. Do not commit anything unless
  explicitly asked.
- Prior committed baseline (pre-Gemini): `0b054d44` — `feat: implement stage 10
  human approval`, still reachable in history.

### Stage status

- **Stages 1–10 (deterministic / mock pipeline): COMPLETE**, unchanged, and
  still the default execution path. All existing deterministic tests stay
  green — full suite at this checkpoint: **1133 passed, 4 skipped**
  (`python -m pytest tests -q`; the 4 skips are the key-gated real-Gemini
  integration cases).
- **Stage 1 real Gemini foundation + integration: COMPLETE** on this branch
  (provider layer, `TopicCandidateProvider` integration, opt-in CLI, offline
  tests, provenance, API-key security).
- **Stage 2 real Gemini Research: COMPLETE** (implemented, tested, committed;
  real key-gated run performed — produced
  `data/857acc781211-firecrawl-lite/research/research_package.json`; run data
  is gitignored).
- **Stage 4 real Gemini Script: COMPLETE** on this branch (implemented, tested,
  NOT committed; real key-gated smoke produced
  `data/857acc781211-firecrawl-lite/scripts/script.json`).
- **Stages 3 and 6 real Gemini providers: NOT IMPLEMENTED** (design only —
  `GeminiEvidenceProvider`, `GeminiVerifier`, `GeminiScenePlanner` do not
  exist).
- **Stage 11 (YouTube publishing) and later: NOT IMPLEMENTED**, intentionally
  out of scope.

### Stage-1 Gemini implementation (committed)

- `providers/gemini/` — `errors.py`, `config.py`, `structured.py`, `client.py`,
  `topic_provider.py` (+ package `__init__.py`); SDK is import-lazy and
  **never imported by any `agents/` stage**.
- `agents/seo/provider.py` — `TopicCandidateProvider` protocol,
  `MockTopicCandidateProvider` (mock path preserved), and
  `build_topic_candidate_provider()` factory.
- `agents/seo/topic_selector.py` — `--provider mock|gemini` opt-in
  (default `mock`); a Gemini failure exits with code 2 and writes no artifact.
- `prompts/seo/topic_candidates-v1.txt` — versioned Stage-1 prompt.
- `requirements.txt` — `google-genai>=2.23,<3`, `pydantic>=2,<3`;
  `.env.example` documents `GEMINI_API_KEY` / `GEMINI_MODEL`.
- Provenance — Gemini runs stamp envelope `produced_by` and per-candidate
  `provenance` with `provider="gemini"`, `model`, and `prompt_version`.
- API-key security — `GEMINI_API_KEY` is required only for the opt-in path; the
  key never appears in `repr`, exceptions, logs, or artifacts
  (`tests/test_gemini_security.py`).

### Stage-2 Gemini implementation (implemented, NOT committed)

- `providers/gemini/research_provider.py` — `GeminiResearchProvider`,
  `ResearchResultSchema`/`ResearchResultAdapter` (Pydantic gate over the
  research contract: claim types, credibility tiers, date kinds),
  `_normalize_url`/`_is_iso`/`_netloc` helpers, `claim_text_digest`,
  `PROMPT_VERSION = "research/facts-v1"`.
- `providers/gemini/client.py` — `generate_research(...)` (Google Search
  grounding + optional URL Context), `GroundedStructured`/`GroundingChunkInfo`
  dataclasses, `_extract_grounding`/`_extract_url_context`; `_base_params`
  refactor (behavior preserved). Same retry/timeout/error discipline as Stage 1.
- `providers/gemini/__init__.py` — exports `GeminiResearchProvider`,
  `GroundedStructured`, `GroundingChunkInfo`.
- `agents/research/provider.py` — `build_research_provider(name, *,
  data_path=..., env=None, enable_url_context=False)` factory: `mock` default
  (no key/SDK), `gemini` requires `GEMINI_API_KEY`, unknown name →
  `ResearchProviderError`.
- `agents/research/research_agent.py` — `--provider mock|gemini` (default
  `mock`) + `--enable-url-context`; gemini path stamps `produced_by` and
  replaces `research_notes[0]` with an honest grounding note; provider errors
  exit 2 without writing an artifact.
- `prompts/research/facts-v1.txt` — versioned Stage-2 research prompt.
- Evidence boundary: only retrieval-metadata URLs become sources; ungrounded
  model sources are dropped; no grounding → empty candidates →
  `insufficient_evidence`; claims deduped by lowercased text.
- `tests/gemini_test_utils.py` — grounding-aware fakes (FakeGroundedResponse,
  FakeCandidate, FakeGroundingMetadata/Chunks, FakeUrlContextMetadata,
  `grounded_result`, `valid_research_payload`).
- New offline tests (54): `tests/test_gemini_research_client.py`,
  `tests/test_gemini_research_provider.py`,
  `tests/test_gemini_research_security.py`,
  `tests/test_research_provider_factory.py`,
  `tests/test_research_agent_cli.py`. Gated Stage-2 real-network test added to
  `tests/test_gemini_integration.py` (`REQUIRES_KEY`).

### Verification

- Full suite at that point: **1012 passed, 2 skipped**; the suite as of the
  Stage-4 Gemini script slice (current uncommitted work) is **1133 passed,
  4 skipped** (`python -m pytest tests -q`).
- **54 new offline Stage-2 Gemini/provider tests** (client language: grounding,
  tool wiring, errors/retries; provider: evidence boundary, schema strictness,
  dedup, provenance; security: no key/sdk leakage, mock works without key,
  sockets-blocked run; factory wiring; CLI opt-in). No network, no real key
  needed.
- Real-network integration is gated behind `GEMINI_INTEGRATION_TESTS=1`
  (`tests/test_gemini_integration.py`, Stage 1 + Stage 2 tests) and was **NOT
  run** — `GEMINI_API_KEY` was not configured.
- Verified behavior: default mock CLI unchanged and produces a valid
  `ResearchPackage` (`ready_for_fact_check`); `--provider gemini` without a key
  fails cleanly (exit 2, message names `GEMINI_API_KEY`) and writes no
  artifact; `--help` documents the new flags.

### Stage 2 research decisions (implemented)

- `GeminiResearchProvider` implements the existing `ResearchProvider` protocol
  (`agents/research/provider.py`).
- **Gemini + Google Search grounding is the selected first retrieval
  mechanism.**
- URL Context is implemented for deeper retrieval but **off by default**
  (`--enable-url-context`; free-tier behaviour unverified).
- Actual retrieved source metadata / URLs are the evidence boundary.
- Gemini's generated knowledge is **NOT** treated as verified evidence.
- Research claims remain **provisional** until Stage 3 Fact Checking.
- **No invented / fabricated URLs or sources.**
- Stage 3 remains the factual verification gate.
- `MockResearchProvider` remains the default / offline path; real Gemini is
  explicit opt-in.
- The existing `ResearchPackage`/data contract and deterministic validation
  (`docs/DATA_CONTRACTS.md`) are preserved unchanged.

### Next task

**Verify and (when asked) commit the Stage-4 Gemini script slice** (current
uncommitted work: `agents/script/provider.py`, `providers/gemini/script_provider.py`,
`prompts/script/script-v1.txt`, `agents/script/script_generator.py` CLI,
`agents/script/validator.py` gate, and the 49 new offline tests in the five
`tests/test_*_script*` files plus the gated real case in
`tests/test_gemini_integration.py`). A real key-gated smoke already produced
`data/857acc781211-firecrawl-lite/scripts/script.json`. After committing,
continue with single-step real-provider progression (next natural candidate:
Stage 3 — `GeminiEvidenceProvider` behind the `EvidenceProvider` protocol).

Practically, the next session should:

1. Confirm `python -m pytest tests -q` still reports **1133 passed, 4 skipped**.
2. Inspect `git status`/`git diff` to review the new Stage-4 script slice
   (files listed in the Stage-4 section above and in
   `docs/GEMINI_PROVIDER_DESIGN.md` §[K]).
3. If a real `GEMINI_API_KEY` is available and the user opts in, optionally
   re-run the Stage-4 smoke:
   `$env:GEMINI_MODEL="gemini-3.5-flash-lite"; python -m agents.script --provider gemini --research data\857acc781211-firecrawl-lite\research\research_package.json`
   (the `.env` default `gemini-3-flash` may be free-tier quota-exhausted;
   note the quota metric is per-model), or
   `GEMINI_INTEGRATION_TESTS=1 python -m pytest tests/test_gemini_integration.py -q`.
4. Commit only when explicitly asked.

Explicitly do NOT, unless asked:

- implement Stage 3 yet;
- implement Gemini providers for Stages 3 or 6 yet;
- change the deterministic / mock pipeline or any existing data contract;
- implement video generation / assembly / upload / automation work.

### How to continue

1. Read `docs/PROJECT_PROGRESS.md` (this file) first.
2. Read `docs/GEMINI_PROVIDER_DESIGN.md` (provider architecture + Stage 2 plan
   + §[K] Stage-4 script slice).
3. Read `docs/ARCHITECTURE.md`, `docs/DATA_CONTRACTS.md`,
   `docs/FACT_VERIFICATION.md`, `docs/RESEARCH_PIPELINE.md`.
4. Confirm the test suite passes
   (`python -m pytest tests -q` — expect **1133 passed, 4 skipped**).
5. Continue with the single next task above. Stages 3–11 stay out of scope.
6. Do not redo completed stages unless a concrete bug is discovered.
7. Do not rely on previous OpenCode conversation history.