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

The first **three** stages are implemented (Topic Selection = Stage 1,
Research = Stage 2, Fact Checking = Stage 3). Everything below Fact Checking
is design-only documentation until implemented.

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
`TopicSelection`, ...).

## Current verification

Full suite: **201 tests passing** (42 Stage 1 + 80 Stage 2 + 79 Stage 3).

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

## Future stages

Planned, **not implemented**:

- Stage 4 — Script Generation
- Stage 5 — Script QA
- Stage 6 — Scene Planning
- Stage 7 — Production / Colab Video Generation
- Stage 8 — Video QA
- Stage 9 — YouTube Policy / Publishing QA
- Stage 10 — Human Approval
- Stage 11 — YouTube Publishing
- Stage 12 — Analytics
- Stage 13 — Automated Scheduling / Orchestration

Do not claim any of these are implemented.

## Important design decisions

1. File-based, versioned JSON contracts are the interface between stages.
2. `production.json` / `ProductionPlan` will eventually be the stable
   boundary between content generation and video generation.
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
- Exact policy-check implementation
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
STAGE 4 SCRIPT GENERATION IS NEXT
```

Current verification: 201 tests passing.

The next implementation task is **"Implement Stage 4 — Script Generation"** —
do not implement it now. It may consume the `FactCheckResult.approved_claims`
only.

## How to continue

1. Read `docs/PROJECT_PROGRESS.md` (this file) first.
2. Read `docs/ARCHITECTURE.md`.
3. Read `docs/DATA_CONTRACTS.md`.
4. Read `docs/FACT_VERIFICATION.md`.
5. Confirm the current test suite passes
   (`python -m pytest tests -q` — expect 201 passing).
6. Continue with Stage 4 only.
7. Do not redo completed stages unless a concrete bug is discovered.
8. Do not rely on previous OpenCode conversation history.