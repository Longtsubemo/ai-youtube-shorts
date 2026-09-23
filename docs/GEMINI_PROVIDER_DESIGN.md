# Gemini Provider Integration — Design Report

**Status:**

- **Gemini Provider Architecture: IMPLEMENTED for Stages 1 & 2**
- **Stage 1 Gemini (topic selection): IMPLEMENTED**
- **Stage 2 Gemini Research: IMPLEMENTED** (offline-tested; a real key-gated
  run is the only unperformed verification)
- **Stage 2 web retrieval (Firecrawl retriever layer): IMPLEMENTED**
  (offline-tested; a real Firecrawl network run is the only unperformed
  verification — the `firecrawl` SDK is not installed in this environment)
- **Stages 3, 4, 6 Gemini providers (`GeminiEvidenceProvider`,
  `GeminiVerifier`, `GeminiScriptGenerator`, `GeminiScenePlanner`):
  NOT IMPLEMENTED** (design only)

This document records the inspection/design pass over the existing Stages 1–6
implementation and the plan for introducing a real Gemini provider
without breaking the deterministic mock providers, data contracts, provenance,
tests, or the Stage 1–6 boundaries and determinism.

Stages 1–10 are complete and deterministic (branch `feat/real-gemini-pipeline`,
HEAD `439a2bb`), **1084 tests pass (3 skipped)**. Stage 11 (YouTube upload)
remains out of scope.

The Stage-1 slice of this plan is implemented (see **[H] Implemented —
Stage 1 slice** below): a reusable `providers/gemini/` layer plus the
`agents/seo/provider.py` seam, used by Stage 1 (topic selection) via an explicit
`--provider gemini` CLI opt-in. **Stage 2 (research) is now implemented too**
(same layer + `agents/research/provider.py` seam; see **[I] Implemented —
Stage 2 slice** plus **[I₂] Stage 2 — Firecrawl web retrieval layer**),
leaving Stages 3, 4, and 6 with mock-first defaults; their
adapter classes (section B) are NOT yet implemented. The default mock paths are
byte-for-byte unchanged.

---

## A. Key finding: the LLM seam already exists

Every LLM-worthy stage already has a `Protocol` + deterministic `Mock`
implementation, and its stage runner already accepts the provider through an
optional injected argument. A real Gemini provider is therefore a **new adapter
class implementing an existing protocol**, passed explicitly — **not** a change
to any stage logic, mock, contract, ID rule, test, or default.

| Stage | Name | Seam (protocol / mock) | Injection point |
|---|---|---|---|
| 1 | Topic selection | **`TopicCandidateProvider` / `MockTopicCandidateProvider`** (`agents/seo/provider.py`) — seam added with the Stage-1 Gemini slice; `GeminiTopicProvider` implements it | `build_topic_candidate_provider()` + `--provider gemini` (implemented) |
| 2 | Research | `ResearchProvider` / `MockResearchProvider` (`agents/research/provider.py`) | `run_research(selection, provider=None, ...)` |
| 3 | Fact checking | `EvidenceProvider` / `MockEvidenceProvider` (`agents/qa/fact_checker/evidence.py`) + `Verifier` / `MockVerifier` (`agents/qa/fact_checker/verifier.py`) | `run_fact_check(package, provider=None, verifier=None, ...)` |
| 4 | Script | `ScriptGenerator` / `MockScriptGenerator` (`agents/script/script_generator.py`) | `run_script_generation(..., generator=...)` |
| 5 | Script QA | **no seam — deterministic 13 checks** (`agents/qa/script/checks.py`) | — |
| 6 | Scene planning | `ScenePlanner` / `MockScenePlanner` (`agents/scene_planner/planner.py`) | `run_scene_planning(..., planner=...)` |

Stage 1 originally had no LLM seam **by design** (scored topic selection;
direction comes from config). It now HAS a seam (`TopicCandidateProvider`),
added only by the Stage-1 Gemini slice, and stays deterministic in the mock
default: the existing scorer/selector remains the final gate regardless of how
candidates are sourced. Stage 5 has no LLM seam **by design** — it is the
machine QA gate for the script — and does not change.

## B. Target: what a “real Gemini provider” adds

Following `docs/MODEL_STRATEGY.md` — a provider is never hard-coded anywhere
agents import. Implementation keeps the Gemini SDK out of every `agents/`
package. The architecture is a small layered stack:

```
Gemini provider layer (providers/gemini/, SDK isolated, lazy import)
    ↓
per-stage provider adapter (GeminiTopicProvider, ...)
    ↓
existing stage protocol (TopicCandidateProvider, ResearchProvider, ...)
    ↓
existing deterministic validator / gates
```

New layer (implemented for Stage 1; the SDK specifics differ from the earlier
`agents/llm/` design below — actual layout is `providers/gemini/`):

```
providers/gemini/
├── errors.py         GeminiProviderError hierarchy (no raw SDK exception escapes)
├── config.py         GeminiConfig + from_env() (GEMINI_API_KEY / GEMINI_MODEL)
├── structured.py     parse + Pydantic-validate model JSON into the stage contract
├── client.py         GeminiClient — lazy google.genai import, bounded retry,
│                     error classification (injectable raw_client for tests);
│                     generate_structured (Stage 1) + generate_research (Stage 2)
├── topic_provider.py GeminiTopicProvider — Stage-1 adapter over TopicCandidateProvider
└── research_provider.py GeminiResearchProvider — Stage-2 adapter over ResearchProvider
```

Per-stage adapters implement the stage's **existing** protocol and receive
**only the same inputs the mock receives**. Output is validated by the stage's
existing validator/gates before the artifact can be persisted.

### `ModelProvider` (gateway) — NOT implemented as a separate layer

An abstract gateway (`ModelProvider` / `ModelRequest` / `ModelResponse`)
sketched by `docs/MODEL_STRATEGY.md` §2 remains **design-only**. The Stage-1
slice deliberately used the concrete `GeminiClient` instead:

```python
class ModelProvider(Protocol):
    def complete(self, request: ModelRequest) -> ModelResponse: ...

# ModelRequest
#   messages: str | list
#   system_prompt_ref: Optional[str]   # versioned prompt, e.g. research/facts-v1
#   temperature: Optional[float]
#   max_tokens: Optional[int]

# ModelResponse
#   text: str
#   usage: tokens_in/out
#   provider_ref: str                  # e.g. "gemini-2.x-flash@2026-<hash>"
#   latency_ms: int
```

`provider_ref` is what flows into every artifact’s `produced_by.provider` /
`produced_by.model` (`DATA_CONTRACTS.md` §1). A later stage may introduce the
gateway; nothing today depends on it, and the Stage-1 provider stamps the same
fields directly via `GeminiConfig.model`.

### Per-stage protocol adapters

Each adapter implements the stage’s **existing** protocol and receives **only
the same inputs the mock receives**. Output is validated by the stage’s existing
validator/gates before the artifact can be persisted.

| Adapter | Protocol it implements | Input bound | Validator it must pass | Status |
|---|---|---|---|---|
| `GeminiTopicProvider` | `TopicCandidateProvider` | `config/topic_selection.yaml` (pillars, topic types) | `TopicCandidate.validate()` / `run_topic_selection` | **IMPLEMENTED (Stage 1)** |
| `GeminiResearchProvider` | `ResearchProvider` | `TopicDescriptor` (one topic, one evidence request); grounding metadata is the evidence boundary | `agents/research/` package-building rules (provisional-only claims, dedup, source validation) | **IMPLEMENTED (Stage 2)** |
| `GeminiEvidenceProvider` | `EvidenceProvider` | `Source`; evidence constrained to the source's own content | `MeaningfulEvidence` / `SUPPORTING_EVIDENCE_TYPES` bounds | NOT IMPLEMENTED |
| `GeminiVerifier` | `Verifier` | **strictly `{claim, supplied evidence, source metadata}`** — never outside/world knowledge (`agents/qa/fact_checker/verifier.py` docstring); returns `ClaimAssessment` | verdict vocabulary `DIRECT/PARTIAL/INSUFFICIENT/NONE/CONTRADICTORY` | NOT IMPLEMENTED |
| `GeminiScriptGenerator` | `ScriptGenerator` | **only approved claims** (Stage 3 `approved_claims`); no raw research | `validate_script` (token coverage / no-hallucination / certainty preservation, deterministic) | NOT IMPLEMENTED |
| `GeminiScenePlanner` | `ScenePlanner` | `Script` (post Script-QA PASS); returns `ScenePlanningResult` | `validate_production_plan` rules r01–r20 | NOT IMPLEMENTED |

`GeminiTopicProvider` and `GeminiResearchProvider` exist.
`GeminiEvidenceProvider`, `GeminiVerifier`, `GeminiScriptGenerator`, and
`GeminiScenePlanner` are **FUTURE / NOT IMPLEMENTED** — do not claim they exist.

Constaints honored everywhere:

- Facts come from sources; the model only proposes (`MODEL_STRATEGY.md` §6,
  `RESEARCH_PIPELINE.md` §1).
- Fact check is rule-first; LLM is assist at most (`FACT_VERIFICATION.md` §4).
- Nothing in the stable worker contract (`production.json`) gains model info
  (`DATA_CONTRACTS.md` §10).

## C. Wiring — explicit opt-in only, defaults untouched

The injection points in section A already exist. Wiring adds:

1. A factory `builders.py: build_*_provider(stage_config, settings)` that
   returns a Gemini adapter **only when explicitly configured**.
2. An explicit switch on the stage CLIs, e.g. `--provider gemini`, or an env
   flag (`PROVIDER=gemini`), making the opt-in visible in the run.
3. `config/` YAML per stage selecting `provider: mock | gemini` +
   `model`/`prompt_version` (structure deferred to implementation; also covered
   by `MODEL_STRATEGY.md` §7: config in `config/` + `.env`, no SDKs/credentials
   committed).

Rule: **defaults stay `None → Mock*`.** Determinism tests and the
committed-fixture path are byte-for-byte unchanged. Consumers that pass a real
provider take responsibility for cost/latency — a run concern, not a code path
change.

## D. Provenance without contract changes

- Gemini adapters set the existing envelope fields only:
  `produced_by = {provider: "gemini", model: <model id>, prompt_version: <ref>}`
  (research package, evidence, verdicts, script `PRODUCED_BY`, plan `PRODUCED_BY`).
- Schema, `schema_version`, header versions, artifact ID derivations
  (`sc_…`, `pp_…`, `fc_…`, …) are untouched — ids stay content-…/run-key-based.
- The mock artifacts (`data/mock/*.json`, `produced_by.model="MockScriptGenerator"`,
  etc.) remain exactly as they are; Gemini runs produce *new* artifacts under
  their own run dir, with Gemini recorded in provenance.
- `produced_by.prompt_version` is required for every Gemini-produced artifact
  (prompts are versioned inputs, `MODEL_STRATEGY.md` §4).

## E. Structured-output discipline

`MODEL_STRATEGY.md` §5 applies: ask for the contract shape, then validate before
persisting. Concretely:

1. Prompt the model for the exact contract JSON (per-stage prompts under
   `prompts/<stage>/`, e.g. `research/facts-v1.txt` — dirs already exist).
2. Parse and validate against the existing contract models.
3. Only a fully validated artifact is persisted as completed (`None` on any
   parse/validation failure → stage fails closed; nothing partial written).
4. Validation dependency decision deferred (stdlib-only draft vs. a schema
   library — `MODEL_STRATEGY.md` §8).

## F. Safety / tests — nothing enters the audited stage packages

- Stages 1–6 currently have no `test_*_safety.py`; that convention starts at
  Stage 7 (`tests/test_*_safety.py`): static token audit (no network / GPU /
  env-read / secrets / downstream vocab), sockets-blocked healthy run, no-write
  purity.
- Keeping the Gemini SDK **only inside `providers/gemini/`** preserves the same
  guarantee for Stages 1–10: the audited `agents/` packages stay import-free of
  the SDK, network-free, and env-free. When Stage 1–6 safety tests are added
  later, the audit window is unchanged.
- New `providers/gemini/` tests are offline: a stubbed `raw_client` + recorded
  responses, `socket.blocked`, no network, no real key. `GeminiClient` is
  tested against a stubbed client that never talks to Google.
- Determinism: every existing test keeps using mocks; new tests cover only the
  new layer + adapter wiring.

## G. Explicit statement

**Stage 1 Gemini integration IS IMPLEMENTED; Stage 2 Gemini Research IS
IMPLEMENTED (offline-tested, real key-gated run NOT performed); the Stage-2
Firecrawl web-retrieval layer IS IMPLEMENTED (offline-tested, real Firecrawl
run NOT performed); Stage 3, 4, and 6 Gemini adapters are design only.**
Implemented files for the Stage-1 slice:
`providers/gemini/` + `agents/seo/provider.py` +
`prompts/seo/topic_candidates-v1.txt` + offline tests. Implemented files for the
Stage-2 slice: `providers/gemini/research_provider.py` + `generate_research` in
`providers/gemini/client.py` + `agents/research/provider.py` factory +
`agents/research/research_agent.py` CLI opt-in + `prompts/research/facts-v1.txt`
+ offline tests. Implemented for the Stage-2 Firecrawl slice:
`providers/firecrawl/` + `agents/research/retriever.py` + `retriever`
param/`--retriever firecrawl` wiring +
`prompts/research/facts-retrieval-v1.txt` + offline tests. Default paths stay
`None → Mock*`; the mock runner, contracts, provenance schemes, and ID rules are
unchanged; the full suite (1084 passed, 3 skipped) is green. Stages 3, 4, and 6
keep mock-first defaults.

---

## H. Implemented — Stage 1 slice

The following were implemented for Stage 1 (topic selection) only:

1. `providers/gemini/errors.py` — `GeminiProviderError` hierarchy
   (configuration / authentication / API / timeout / malformed-response /
   schema-validation / unexpected). No raw SDK exception escapes the provider.
2. `providers/gemini/config.py` — `GeminiConfig` (api_key, model, timeout,
   max_retries, temperature, max_output_tokens). Key from `GEMINI_API_KEY`
   (required) via `from_env()`; model from `GEMINI_MODEL` (default
   `gemini-3.5-flash-lite`). `repr`/`redacted()` never expose the key.
3. `providers/gemini/structured.py` — parse + Pydantic-validate the model
   response into the stage contract (`GeminiMalformedResponseError`,
   `GeminiSchemaValidationError`).
4. `providers/gemini/client.py` — `GeminiClient` (lazy `google.genai` import —
   importing the provider never imports the SDK), bounded retries on
   408/429/5xx + timeouts only, classification to the error hierarchy,
   injectable `raw_client`/`sleep` for offline tests.
5. `providers/gemini/topic_provider.py` — `GeminiTopicProvider` implementing
   Stage 1's `TopicCandidateProvider` seam; returns validated `TopicCandidate`s
   and stamps `produced_by = {provider: "gemini", model, prompt_version}`.
6. `agents/seo/provider.py` — `TopicCandidateProvider` protocol +
   `MockTopicCandidateProvider` (existing mock behavior) +
   `build_topic_candidate_provider()` factory (default `mock`; `gemini` requires a key).
7. `agents/seo/topic_selector.py` — `--provider {mock,gemini}` opt-in; provider
   failure exits with code 2 and never writes a partial artifact.
8. `prompts/seo/topic_candidates-v1.txt` — versioned Stage-1 prompt
   (`{candidate_count}`, `{pillars}`, `{topic_types}` placeholders).
9. `requirements.txt` — `google-genai>=2.23,<3`, `pydantic>=2,<3` (optional /
   only needed behind the opt-in). `.env.example` documents `GEMINI_API_KEY` /
   `GEMINI_MODEL`.
10. Offline tests: `tests/gemini_test_utils.py`, `test_gemini_config.py`,
    `test_gemini_client.py`, `test_gemini_topic_provider.py`,
    `test_topic_candidate_provider.py`, `test_topic_selector_cli.py`,
    `test_gemini_security.py` (key-leak regression). Real-network integration
    is gated by `GEMINI_INTEGRATION_TESTS=1` (`tests/test_gemini_integration.py`).

Key leak safety: the API key never appears in repr, exception text, redacted
metadata, or artifacts — enforced by `test_gemini_security.py` and covered in
the CLI gemini-path tests.

## H₂. Adoption checklist (remaining — Stages 2–6)

1. Decide whether a generic `ModelProvider` / `ModelRequest` / `ModelResponse`
   gateway (`MODEL_STRATEGY.md` §2) is worth introducing; Stage 1 used the
   concrete `GeminiClient` and stamped provenance directly.
2. Extend `providers/gemini/client.py` `GeminiClient` for the required stage
   shapes (research, evidence, verifier, script, scene planning) as needed —
   lazy import, `.env` `GEMINI_API_KEY`, already done for Stage 1.
3. Add the remaining stage adapters behind the existing protocols; Stage 1's
   `GeminiTopicProvider` is done and **Stage 2's `GeminiResearchProvider` is now
   implemented (this branch, offline-tested)**. **Next: Stage 3's
   `GeminiEvidenceProvider` (behind `EvidenceProvider`) only** — not Stages 4/6.
4. Add `providers/gemini/builders.py`-style factories + CLI/env opt-in
   (`--provider gemini`); defaults unchanged (Stage 1 factory lives in
   `agents/seo/provider.py`).
5. Add `prompts/<stage>/<name>.<ver>` templates + `config/` provider YAML
   (`provider: mock | gemini`, model, prompt refs).
6. Add offline tests (stubbed Gemini client, `socket.blocked`) — no network, no
   `test_*_safety.py` changes to Stages 1–6.
7. Record provenance (`provider` / `model` / `prompt_version`) per
   `DATA_CONTRACTS.md` §1 on Gemini-produced artifacts only.

## I. Implemented — Stage 2 slice (Gemini Research Provider)

**Status: IMPLEMENTED** (this branch, offline-tested and wiring-complete; the
only unperformed verification is a real key-gated network run, gated by
`GEMINI_INTEGRATION_TESTS=1`). `MockResearchProvider` remains the default /
offline path.

### Flow (implemented)

```
TopicDescriptor
    ↓
GeminiResearchProvider (implements existing ResearchProvider protocol)
    ↓
Gemini + Google Search grounding (primary retrieval)
    ↓
candidate source discovery + optional URL Context retrieval (opt-in)
    ↓
grounding metadata / URLs = the EVIDENCE BOUNDARY
    ↓
SourceCandidate[] → existing deterministic ResearchAgent → ResearchPackage
    ↓
existing deterministic research validation (provisional-only claims)
    ↓
Stage 3 Fact Checking
```

When `--retriever firecrawl` is passed, the retrieval step replaces Google
Search grounding with `FirecrawlResearchRetriever` (Firecrawl Search API +
full-page Markdown in the same call); section I₂ documents that path. With no
retriever the flow above is exactly what runs.

### What was implemented

- `providers/gemini/client.py` — `GeminiClient.generate_research(prompt,
  response_model, *, system_prompt, temperature, max_output_tokens,
  enable_url_context=False)` → `GroundedStructured` (parsed `data` +
  `grounding` tuples + `url_context_urls`). The request always enables the
  **Google Search grounding tool**; `enable_url_context` adds the **URL Context
  retrieval tool**. Same bounded-retry / timeout / error-classification
  discipline as `generate_structured`. `_request_config` refactored onto
  `_base_params` (Stage-1 behavior preserved).
- `providers/gemini/research_provider.py` — `GeminiResearchProvider` with
  `ResearchResultSchema` (sources + per-source claims; Pydantic `Literal`
  bounds on claim types, credibility tiers, date kinds — malformed output fails
  as `GeminiSchemaValidationError`, never a raw SDK error), `PROMPT_VERSION =
  "research/facts-v1"`, and helpers `_normalize_url`, `_is_iso`, `_netloc`,
  `claim_text_digest`.
- **Evidence boundary (the anti-fabrication guarantee):** only URLs present in
  the retrieval metadata (grounding chunks and/or URL-context hits) survive as
  sources. A model-proposed source whose URL is not grounded is **dropped**; if
  there is no grounding metadata at all the provider returns **no candidates**,
  so the package honestly lands on `insufficient_evidence`. Claims live inside
  a source record (never floating), and duplicate claim text is collapsed
  across the result (claim ids derive from text, so duplicates would break
  `ResearchPackage.validate()`).
- `agents/research/provider.py` — `build_research_provider(name, *,
  data_path=..., env=None, enable_url_context=False)` factory: `mock`/`None`
  default (key- and SDK-free), `gemini` requires `GEMINI_API_KEY`
  (`GeminiConfigurationError` otherwise — no silent fallback), unknown name →
  `ResearchProviderError`.
- `agents/research/research_agent.py` — `--provider mock|gemini` (default
  `mock`) + `--enable-url-context`. The gemini path stamps envelope
  `produced_by` and replaces `research_notes[0]` with an honest grounding note
  ("…every source URL comes from retrieval metadata … Claims remain
  provisional until the fact-check stage."). Provider errors exit 2 and write
  nothing (mirrors `agents/seo/topic_selector.py`).
- `prompts/research/facts-v1.txt` — versioned Stage-2 prompt
  (`{topic_id}`, `{title}`, `{pillar}`, `{topic_type}`, `{description}`,
  `{max_sources}` placeholders).
- Offline tests: grounding-aware fakes in `tests/gemini_test_utils.py`
  (`FakeGroundedResponse`, `grounded_result`, `valid_research_payload`, …) +
  `tests/test_gemini_research_client.py`,
  `tests/test_gemini_research_provider.py`,
  `tests/test_gemini_research_security.py`,
  `tests/test_research_provider_factory.py`,
  `tests/test_research_agent_cli.py`. Real-network Stage-2 integration test
  added to `tests/test_gemini_integration.py` (`REQUIRES_KEY`).

### Stage-2 rules (decided and enforced)

- **Gemini is a research/retrieval ASSISTANT, NOT the factual authority.**
- Search-grounded source metadata must be retained.
- **Do not fabricate URLs or sources** — evidence boundary drop (see above).
- Claims returned by research remain **provisional**.
- **Stage 3 performs factual verification**; it remains the factual gate.
- Evidence must be tied to actual retrieved sources (grounding metadata / URLs
  are the evidence boundary).
- The provider implements the existing `ResearchProvider` protocol.
- `MockResearchProvider` remains available and is the default / offline path.
- Real Gemini must be **explicitly selected** (opt-in).
- No stage package imports the Gemini SDK directly (SDK stays inside
  `providers/gemini/`; importing the provider never imports the SDK —
  subprocess-verified).
- API keys / secrets never enter artifacts, logs, exceptions, or committed
  files (key-leak regression tests).
- Compatible with Gemini **Free API** constraints; single request, no
  multi-call loops.
- Preserves the existing `ResearchPackage` data contract and deterministic
  validation (`docs/DATA_CONTRACTS.md`).
- Provenance: `produced_by = {provider: "gemini", model, prompt_version}`
  (`DATA_CONTRACTS.md` §1).

### Floating decision (resolved)

- JSON extraction for research structured output: **provider `response_schema`
  + Pydantic** (same as Stage 1) — open decision 2 below is RESOLVED.

## I₂. Implemented — Stage 2 Firecrawl web retrieval layer

**Status: IMPLEMENTED** (this branch, offline-tested and wiring-complete; the
real Firecrawl network run is NOT performed and gated by
`FIRECRAWL_INTEGRATION_TESTS=1` — the `firecrawl` SDK is not installed in this
environment).

### Flow (retrieval variant)

```
TopicDescriptor
    ↓
GeminiResearchProvider (implements existing ResearchProvider protocol)
    ↓
ResearchRetriever (FirecrawlResearchRetriever when --retriever firecrawl)
    → Firecrawl Search API: search + full-page Markdown in the same call
    ↓
RetrievedSource[] from the retriever = the EVIDENCE BOUNDARY
    ↓
sliced to max_retrieved_sources (4) → Gemini structured extraction
    (prompts/research/facts-retrieval-v1.txt)
    ↓
SourceCandidate[] → existing deterministic ResearchAgent → ResearchPackage
    ↓
existing deterministic research validation (provisional-only claims)
    ↓
Stage 3 Fact Checking
```

### What was implemented

- `providers/firecrawl/` — a second SDK-isolated provider package (import-lazy:
  `from firecrawl.v2 import FirecrawlClient` happens only inside
  `FirecrawlClient._get_client`), mirroring the `providers/gemini/` discipline:
  - `errors.py` — `FirecrawlProviderError` hierarchy (configuration /
    authentication / API / timeout / malformed-response / unexpected). Missing
    SDK → `FirecrawlConfigurationError`. No raw `firecrawl` exception escapes.
  - `config.py` — `FirecrawlConfig` (api_key, api_url, timeout_seconds 60,
    max_retries 2, max_search_results 5). **Keyless by default**:
    `FIRECRAWL_API_KEY` is optional (free-tier keyless mode). `from_env()`,
    `repr`/`redacted()` never expose credentials.
  - `client.py` — `FirecrawlClient.search(query, *, limit)` using the v2 SDK
    (`firecrawl.v2.FirecrawlClient` — keyless-capable; the top-level
    `firecrawl.Firecrawl` class requires an API key on 4.44.0 and does NOT back
    keyless mode; `client.search(query, sources=["web"], limit, timeout=ms,
    scrape_options={"formats": ["markdown"], "only_main_content": True})`) →
    `FirecrawlResult` (url/title/description/markdown/source_url). Tolerant
    normalization of the `.web` (v2) / `.data` (legacy) buckets, including
    hydrated v2 `Document` hits whose url/title/description resolve via
    `metadata`; malformed bucket → `FirecrawlMalformedResponseError`. Bounded
    retries on transient/429/timeout only; authentication errors do not retry.
  - `retriever.py` — `FirecrawlResearchRetriever` implementing the
    `ResearchRetriever` protocol: domain allow/deny filters, Markdown truncation
    (`DEFAULT_MAX_CONTENT_CHARS=6000`), maps every `Firecrawl*Error` to
    `ResearchProviderError` at the `agents/` boundary.
- `agents/research/retriever.py` — `ResearchRetriever` protocol
  (`retriever_name`, `retrieve(topic) → list[RetrievedSource]`),
  `RetrievedSource` dataclass
  (url/title/publisher/publication_date/source_type/content/domain/retrieved_at),
  and deterministic `MockResearchRetriever` for offline tests.
- `providers/gemini/research_provider.py` — `retriever` parameter added:
  `search()` branches to the grounding path (byte-for-byte unchanged,
  `retriever=None`) or the retrieval path (`_search_retrieval`): results sliced
  to `max_retrieved_sources=4`, rendered into the prompt (facts-retrieval-v1),
  evidence boundary = only retriever-returned URLs; empty retrieval → no model
  call → `insufficient_evidence`. `produced_by` gains `retrieved_by` only on the
  retrieval path; `prompt_version` = `research/facts-retrieval-v1`.
- `agents/research/provider.py` — `build_research_provider(..., retriever=...)`:
  `mock` + a retriever → `ResearchProviderError`; `firecrawl` builds the
  retriever **keyless and SDK-free** (import-isolation subprocess-verified);
  unknown retriever name → `ResearchProviderError`.
- `agents/research/research_agent.py` — `--retriever firecrawl` opt-in;
  invalid combinations exit 2 without writing an artifact.
- `prompts/research/facts-retrieval-v1.txt` — versioned retrieval prompt
  (`{retrieved_sources}` rendered from the retriever, `{topic_*}` placeholders;
  "only the retrieved URLs may become sources" rule).
- Tests (72 new offline):
  `tests/firecrawl_test_utils.py` (Fake* fixtures, real-raisable fake errors),
  `tests/test_firecrawl_config.py`, `tests/test_firecrawl_client.py` (incl.
  missing-SDK mapping, key redaction, subprocess import-isolation, v2
  `Document`-shape normalization, keyless/keyed `firecrawl.v2.FirecrawlClient`
  resolution),
  `tests/test_firecrawl_retriever.py`,
  `tests/test_gemini_research_provider_retrieval.py` (evidence boundary,
  slicing, provenance, prompt rendering, agent integration incl. a
  sockets-blocked run), `tests/test_research_retriever_cli.py`, extended
  `tests/test_research_provider_factory.py`. One gated real-network test in
  `tests/test_firecrawl_integration.py` (`FIRECRAWL_INTEGRATION_TESTS=1`).
  `requirements.txt` pins `firecrawl>=4.22,<5` (optional).

**Verified against firecrawl 4.44.0 (2026-09-23):** the top-level
`firecrawl.Firecrawl` class requires an API key on 4.44.0 (`ValueError: No API
key provided` when keyless), so the adapter resolves `firecrawl.v2.FirecrawlClient`
(keyless-capable; home of v2 Search). Live `/v2/search` responses return both
`SearchResultWeb` and hydrated `Document` entries — `Document` entries carry
`url=''`/`title=''` top-level, so `_normalize_item` resolves url/title/description
via `metadata`. The SDK itself can raise a pydantic `ValidationError` during
response parsing when a non-Document web hit lacks the required `url`; the
adapter's request shape is otherwise proven valid against the live API (9/9
successful real calls for the smoke topic), and all failures map to the typed
provider hierarchy. Live retriever run returned 5 real sources with populated
metadata.

### Stage-2 retrieval rules (decided and enforced)

- The retriever **returns evidence; Gemini proposes claims** — the evidence
  boundary never widens to model knowledge.
- **Do not fabricate URLs or sources** — only `RetrievedSource`s from the
  retriever may become `Source`s.
- Retrieval is **bounded and determinism-shaped**: max search results (5),
  sliced to at most 4 sources per package, content truncated at 6000 chars,
  bounded retries/timeout.
- `FIRECRAWL_API_KEY` is optional (keyless free tier, rate-limited); the key
  never enters artifacts, logs, exceptions, or committed files (key-leak
  regression tests).
- `providers/firecrawl/` stays SDK-isolated: neither `agents/` nor
  `providers/gemini/` imports the `firecrawl` SDK (subprocess-verified); a
  missing SDK fails cleanly as `ResearchProviderError`, never a raw
  `ModuleNotFoundError`.
- The existing `ResearchPackage` contract, the grounding path, and deterministic
  validation (`docs/DATA_CONTRACTS.md`) are unchanged; Google Search grounding
  remains the default when no retriever is passed.

## J. Open decisions (implementation time)

1. Validation dependency: Pydantic v2 is used by both Stage-1 and Stage-2
   slices; confirmed as the shared schema layer for research structured output
   (`MODEL_STRATEGY.md` §8).
2. JSON extraction: **RESOLVED** — provider `response_schema` + Pydantic for
   both Stage 1 (topic candidates) and Stage 2 (research).
3. Retry/backoff and token-budget policy per stage: bounded retries + free-API
   rate-limit shaping, same policy for research as Stage 1 (single request, no
   multi-call loops).
4. Prompt/version naming convention: `research/facts-v1` established
   (`seo/topic_candidates-v1` is the Stage-1 precedent).
5. RESOLVED (vs. previously open): Gemini Search grounding is the selected first
   retrieval mechanism for research; sources remain the authority — the model
   proposes, never the fact authority. Stage-3 eligibility for grounding stays
   open until Stage 3 is implemented.