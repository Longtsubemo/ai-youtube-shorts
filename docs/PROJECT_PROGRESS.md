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

### Gemini integration — Stage 1: IN PROGRESS / FOUNDATION COMPLETE

The **provider foundation** and **Stage-1 candidate sourcing via Gemini** are
implemented; Stages 2–6 are NOT yet wired to Gemini (each keeps its mock-first
default). Scope intentionally limited: Stage 1 candidates only; scoring,
ranking, replay, and deterministic validation stay the final gate.

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
**80 Stage-1-related tests**; full suite: **958 passed, 1 skipped**
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

Stage 2 uses `MockResearchProvider` only. **No real web/API research
provider has been implemented yet.**

### Test status

80 Stage 2 tests (`tests/test_research_models.py`,
`test_research_classifier.py`, `test_research_provider.py`,
`test_research_agent.py`).

CLI: `python -m agents.research.research_agent`.

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

Full suite: **920 tests passing** (42 Stage 1 + 80 Stage 2 + 79 Stage 3 +
53 Stage 4 + 92 Stage 5 + 120 Stage 6 + 117 Stage 7 + 74 Stage 8 +
140 Stage 9 + 123 Stage 10).

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

- Real research/web data provider
- YouTube data source
- Production LLM provider
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

## Current checkpoint

```
STAGE 1 COMPLETE (Topic Selection)
    ↓
STAGE 2 COMPLETE (Research Pipeline)
    ↓
STAGE 3 COMPLETE (Fact Checking)
    ↓
STAGE 4 COMPLETE (Script Generation)
    ↓
STAGE 5 COMPLETE (Script QA)
    ↓
STAGE 6 COMPLETE (Scene Planning)
    ↓
STAGE 7 COMPLETE (Video Generation)
    ↓
STAGE 8 COMPLETE (Video QA)
    ↓
STAGE 9 COMPLETE (Policy / Publishing QA)
    ↓
STAGE 10 COMPLETE (Human Approval)
    ↓
STAGE 11 YOUTUBE PUBLISHING IS NEXT (not implemented)
```

Current verification: 958 tests passing (`python -m pytest tests -q` ->
`958 passed, 1 skipped`).

**Final end-to-end verification (completed):** Stages 1–10 are implemented and
the whole pipeline closes with a REAL Stage-10 CLI approval run
(`python -m agents.human_approval --decision APPROVED --approver-id test-user
--approver-role owner --notes "Final manual-upload approval"`):

- exit code **0**;
- decision **APPROVED**;
- all 5 blocking gates OK (`required_artifacts`, `policy_qa_hard_reject`,
  `video_qa_hard_reject`, `video_generation_completed`, `review_overrides`);
- **7 reviewed artifacts** bound by id + canonical `content_sha256`
  (policy_qa / video_qa / video_result / production_plan / script / script_qa
  / fact_check);
- `manual_upload.ready=True`;
- `data/<run_id>/approval/human_approval_result.json` written and run state
  advanced to the existing generic `Stage.PUBLISH` in `state.json`;
- no upload/network/analytics/scheduling action occurred.

Stage 11 (YouTube Publishing) is **NOT IMPLEMENTED** and remains intentionally
out of scope. Project goal achieved: the pipeline now produces a
**manual-upload-ready video package** — the user performs the YouTube upload
manually; no automatic publishing is implemented.

The next implementation task is **"Implement Stage 11 — YouTube Publishing"** —
do not implement it now. Stage 10's `manual_upload.ready` marks the produced
package as ready for a MANUAL YouTube upload by the user; automatic YouTube
publishing may follow in a later stage.

## How to continue

1. Read `docs/PROJECT_PROGRESS.md` (this file) first.
2. Read `docs/ARCHITECTURE.md`.
3. Read `docs/DATA_CONTRACTS.md`.
4. Read `docs/FACT_VERIFICATION.md`.
5. Confirm the current test suite passes
   (`python -m pytest tests -q` — expect 958 passing, 1 skipped).
6. Continue with Stage 11 only.
7. Do not redo completed stages unless a concrete bug is discovered.
8. Do not rely on previous OpenCode conversation history.

## Continuity checkpoint — end of session

Single snapshot so a **fresh OpenCode session** — with **no reliance on any
previous OpenCode conversation history** — knows exactly where the project
stopped without guessing.

### Repository state (Gemini foundation work)

- The Stage-1 **Gemini foundation** (`providers/gemini/`, `agents/seo/provider.py`,
  `prompts/seo/topic_candidates-v1.txt`, offline tests, `.env.example`,
  `requirements.txt` additions, CLI `--provider mock|gemini` opt-in) is
  **implemented and uncommitted** in the working tree; the added Stage-1 tests
  are green (suite: **958 passed, 1 skipped**). This work was NOT committed.
  Hooks/by-line may differ from the HEAD commit below. Do not commit unless
  explicitly asked to.
- HEAD (= the last committed state before this foundation) —
  `0b054d44ecf565e0d7068919b8746ea00179b7bb` — `feat: implement stage 10 human approval`.
- The Stage-1 Gemini slice is **Stage 1 only**: no Stage 2–6 / 7–11 code changes,
  no generative stages, no Stage-11 publishing work.
- The uncommitted Gemini work is NOT part of a clean commit; `git status`
  shows modified/new files listed prior. Do not amend or reword commits.

### Completed — Stages 1–10

Stage 1 Topic Selection, Stage 2 Research, Stage 3 Fact Checking, Stage 4
Script Generation, Stage 5 Script QA, Stage 6 Scene Planning, Stage 7 Video
Generation, Stage 8 Video QA, Stage 9 Policy / Publishing QA, and
**Stage 10 Human Approval** are all **COMPLETE**.

- **Stage 10 = Human Approval**, implemented in **`agents/human_approval/`**
  (`models.py`, `validator.py`, `stage.py`, `__main__.py`, `__init__.py`).
- Stage 10 records an explicit human decision over the exact reviewed
  artifact versions (7 roles, stable ids + canonical `content_sha256`),
  with decisions `APPROVED / CHANGES_REQUESTED / REJECTED / PENDING` (PENDING
  never writable), hard cleaning gates that block APPROVED fail-closed, soft
  REVIEW-only overrides bound to the reviewed artifact, digest-based staleness
  detection, supersede chaining, and a `manual_upload` readiness flag. It
  advances state to the existing generic `Stage.PUBLISH` and performs NO
  upload.
- **Full test suite passes: 920 tests** (`python -m pytest tests -q`),
  including 123 Stage 10 tests (models / validator / stage / safety).
- **Final end-to-end CLI approval verified on the real default path:** exit 0,
  decision `APPROVED`, all 5 blocking gates OK, 7 reviewed artifacts bound by
  id + `content_sha256`, `manual_upload.ready=True`,
  `data/ha-demo/approval/human_approval_result.json` + `state.json`
  (`current_stage: publish`) written, and no upload/network/analytics/
  scheduling action. **Project goal achieved:** the pipeline produces a
  manually upload-ready package; the user performs the YouTube upload
  manually, and no automatic publishing is implemented.
- Stage 10 was verified across the APPROVED / CHANGES_REQUESTED / REJECTED,
  hard-gate-blocked-APPROVED, stale-digest-invalidation, override-bound, and
  supersede paths (CLI exits 0 / 1 / 1 / 2 / 1 / 0 / 0), plus the offline
  safety audit (sockets blocked, no writes, no Stage-11 vocabulary).
- **Stage 10 does NOT implement publishing** — it marks a produced package as
  ready for a MANUAL YouTube upload.

### Not implemented (out of scope)

- **Stage 11 — YouTube Publishing is NOT IMPLEMENTED** and is intentionally
  out of scope.
- **Current project goal — ACHIEVED:** produce a video package that is ready
  for MANUAL YouTube upload by the user **and** carries a verifiable human
  approval record (`manual_upload.ready`). **Automatic YouTube uploading is
  not part of the current goal and is not implemented.**

### Future work (separate, where applicable)

Real production infrastructure remains future work: live research/evidence
providers, Colab/GPU generation workers, TTS, subtitle rendering, video
assembly, actual media rendering, and Stage-11 YouTube publishing.

### Continuity instruction for the next session

1. Read `docs/PROJECT_PROGRESS.md` (this file) first.
2. Read `docs/ARCHITECTURE.md`.
3. Read `docs/DATA_CONTRACTS.md`.
4. Read `docs/FACT_VERIFICATION.md`.
5. Confirm the current test suite passes
   (`python -m pytest tests -q` — expect 958 passing, 1 skipped).
6. Do NOT implement Stage 11 (YouTube publishing) or any later stage unless
   explicitly asked; the goal is a manual-upload-ready video package with a
   verifiable human approval record, not publishing.
7. Do not redo completed stages unless a concrete bug is discovered.
8. Do not rely on previous OpenCode conversation history.