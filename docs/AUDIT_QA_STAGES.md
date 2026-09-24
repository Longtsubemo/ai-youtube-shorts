# Repo Audit — Stage 8 (Video QA) and Stage 9 (Policy / Publishing QA)

Inspection-only report, re-verified against the live repository. No source,
tests, or documentation were modified by the audit; nothing is staged or
committed. Git state at audit time: branch `feat/real-gemini-pipeline`, HEAD
`6a56390 feat: implement Gemini fact checking and script QA`; all tracked files
clean, with one untracked file (this doc).

## Canonical stage numbering

`pipeline/state.py` uses an old 15-band enum (`TOPIC_DISCOVERY … ANALYTICS`)
with **no** stage-8/9 members — both QA stages map to the generic
`Stage.QUALITY_CHECK` (state.py:39); Stage 10 maps to `Stage.PUBLISH`. The
numbered scheme is authoritative in PROJECT_PROGRESS.md section headers,
HUMAN_APPROVAL_STAGE10_DESIGN.md, and DATA_CONTRACTS.md §7/§8/§8.1/§10/§11/§12/§16:

| # | Stage | Artifact |
|---|---|---|
| 3 | Fact Checking | `FactCheckResult` |
| 4 | Script Generation | `Script` |
| 5 | Script QA | `ScriptQAResult` (uses **REVISE**) |
| 6 | Scene Planning | `ProductionPlan` |
| 7 | Video Generation | `VideoGenerationResult` |
| 8 | **Video QA** | `VideoQAResult` (header `stage="video"`) |
| 9 | **Policy / Publish QA** | `PolicyQAResult` (header `stage="policy_qa"`) |
| 10 | Human Approval | `HumanApprovalResult` (header `stage="human_approval"`) |

---

## STAGE 8 — Video QA

### 1. STATUS — COMPLETE

Deterministic/mock-only (network-, GPU-, API-, LLM-free). Executable today:
`python -m agents.qa.video` (defaults to the committed `data/mock` fixtures and
writes the demo run `data/vqa-demo/`).

### 2. RESPONSIBILITY

The deterministic quality gate over the Stage-7 `VideoGenerationResult` + the
Stage-6 `ProductionPlan` it was generated from; emits a PASS/REVIEW/REJECT
verdict. It is a QA/decision stage, not a renderer.

**Backward reach: none.** `agents/qa/video/stage.py` imports only
`scene_planner.models`, `video_generation.models`, `video_generation.stage`. It
never touches Script / ScriptQAResult / FactCheckResult / Research. **PASS.**

### 3. INPUT/OUTPUT CONTRACT (verified against DATA_CONTRACTS §11.1)

- **In:** `VideoGenerationResult` (+ `ProductionPlan` for the pairing guard and
  global style). Pairing: `result.production_plan_id` must equal
  `plan.production_plan_id`, else `VideoQAError` is raised before anything is
  evaluated (stage.py:49). **PASS.**
- **Out:** `data/<run_id>/production/qa_result.json`, envelope header
  `schema="VideoQAResult"` v1 + `data`, id
  `qa_<sha1(run_id:production_plan_id)[:12]>`, binds `target_ref`=plan id,
  `video_generation_result_id`, topic/title/script echo; issues carry
  scene/artifact ids; provenance `video_qa_agent_v1 / DeterministicVideoQA /
  VideoQAValidator`; optional `resumed_from_qa_result_id`. State:
  `advance_to(Stage.QUALITY_CHECK)` + artifact registry entry. Matches §11.1.
  **PASS.**

### 4. GATES — 15 checks (`checks.py`, registered in `CHECKS`)

Duration window (pref 30–60s / hard 15–90s) · aspect-ratio metadata (9:16 hard;
1080×1920@30fps soft, from plan style) · scene coverage · scene pending · scene
failures · scene ordering · timing fidelity (0.001s) · artifact assignment ·
structure integrity · reference integrity · provider isolation (denylist token
scan) · **generation validation (re-runs Stage-7's own
`validate_video_generation_result` as a hard gate)** · overall-status
consistency · mock recognition (non-mock artifact = hard; mock+URI = soft) ·
media honesty (no real/uploaded/published claims).

Failure classes:

- **Blocking (fatal → REJECT):** any `severity=ERROR` with `gate=hard`
  (missing/pending/failed scenes, wrong order, broken timing, malformed/phantom
  artifacts, provider coupling, failed Stage-7 validation, inconsistent overall
  status, mock/truthfulness violations).
- **Warning/review (→ REVIEW):** any soft ERROR (duration outside the preferred
  window, aspect-ratio soft, mock+URI, etc.).
- **Retryable:** a REJECT from generation-validation / scene-failure classes
  maps to "re-run Stage 7"; the framework itself has no stage-level retry
  (orchestrator is scaffold), so retryability is a reading of the verdict, not a
  mechanism.
- **Fatal error (exit 2):** missing/corrupt inputs, mismatched pairing
  (`VideoQAError`). Clean state → **PASS**. No score field. **PASS.**

### 5. FAIL-CLOSED BEHAVIOR

Hard errors REJECT; soft REVIEW; clean PASS; no score. `evaluate_video`
deliberately does **not** call `result.validate()` so structurally broken
results are *graded*, never crashed. Provider-isolation + mock-recognition +
media-honesty prevent any false "real video" claim.

**Partial artifacts:** on the error path (exit 2 / SystemExit) nothing is
written; artifact + state are written only after a verdict is produced. On
REVIEW/REJECT the artifact and state are *intentionally* fully written (exit 1)
— documented, not partial corruption. (See CONCERN C3 on sequential writes.)
**PASS with C3 note.**

### 6. STAGE 8 SPECIFICS

Scene completeness ✓ (`scene_coverage`/`pending`/`failures`); generated/failed
consistency ✓ (`overall_status_consistency` + per-scene status); timing/duration
✓ (`duration_window` + `timing_fidelity`); artifact validity ✓
(`structure_integrity`, `reference_integrity`, `artifact_assignment`);
provider/mock invariants ✓ (`provider_isolation`, `mock_recognition`,
`media_honesty`); plan/result binding ✓ (pairing guard + `target_ref` +
`video_generation_result_id`); provenance ✓; verdict semantics ✓
(PASS/REVIEW/REJECT, no score).

**Can QA pass incomplete video output? No** — any missing/pending/failed scene
or failed Stage-7 validation is a hard REJECT; confirmed by the negative-path
tests. **PASS.**

---

## STAGE 9 — Policy / Publish QA

### 1. STATUS — COMPLETE

Deterministic/mock-only. Executable today: `python -m agents.qa.policy`
(defaults → demo run `data/pol-demo/`).

### 2. RESPONSIBILITY

The deterministic pre-publication POLICY / SAFETY GATE over the full chain; a
gate, not publishing, not Stage 10/11. Consumes Script (#4), FactCheckResult
(#3), ProductionPlan (#6), VideoGenerationResult (#7), VideoQAResult (#8);
produces `PolicyQAResult`. All five are its documented inputs (this stage has
the widest legitimate scope); it does not reach *further* back into
research/SEO/topic artifacts. **PASS.**

### 3. INPUT/OUTPUT CONTRACT (verified against DATA_CONTRACTS §11.2)

`data/<run_id>/production/policy_qa_result.json`, header `schema="PolicyQAResult"`
v1 + `stage="policy_qa"`, id `pr_<sha1(run_id:target_ref)[:12]>`, binds
`target_ref`=plan id, `policy_version`, topic/script echo, per-issue `refs`
(line→claim→scene→asset) + `policy_rule="<check>@<version>"`, `RiskSummary`,
`classifier_suggestion` (default empty), `resumed_from_policy_result_id`,
provenance `policy_qa_agent_v1 / DeterministicPolicyQA / PolicyQAValidator`.
State: `Stage.QUALITY_CHECK`. Matches §11.2. **PASS.**

### 4. GATES — 15 versioned rules `P01–P15` (`rules.py`, `policy_rules_v1`)

P01 harmful content (hard) · P02 dangerous/mod instructions, negator-guarded
"don't try this at home" (hard) · P03 overstatement vs `strong_in_claim=none`
(soft) · P04 misinformation framing (soft) · P05 "actual footage" on mock
(hard) · P06 synthetic media of real people/events without AI disclosure (soft)
· P07 asset rights/provenance · P08 PII/doxxing · P09 spam metadata (≥2 markers)
· P10 near-duplicate narration (overlap ≥0.8, bookends excluded) · P11
advertiser suitability without disclaimer (soft) · P12 sensational title/hook
absent from content (soft) · P13 provenance completeness (soft) · P14
unregistered policy version (hard → REJECT) · P15 prior-gate consistency:
missing/non-PASS/stale-`target_ref` Stage-8 QA → **soft REVIEW**.

Failure classes:

- **Blocking:** any hard ERROR (P01/P02/P05/P14…) → **REJECT**.
- **Review:** any soft ERROR (incl. P15) → **REVIEW**.
- **Fatal error (exit 2):** plan↔video-result pairing mismatch, no `target_ref`
  available, uncaught input issues.
- Clean → **PASS**; **uncertainty never resolves to PASS** (borderline → soft →
  REVIEW). **PASS.**

### 5. FAIL-CLOSED BEHAVIOR

Unregistered policy version → REJECT (never silent PASS); missing artifacts →
P13 REVIEW; missing/non-PASS/stale Stage-8 QA → P15 REVIEW; `evaluate_policy`
grades malformed/absent upstream shapes as issues, never exceptions.

**Partial artifacts:** same as Stage 8 — error paths write nothing; verdict
paths write the full artifact; REVIEW/REJECT artifacts intentionally persisted.
**PASS (C3 note).**

### 6. STAGE 9 SPECIFICS

Policy/safety ✓ (P01–P10); prohibited-content handling ✓ (P01/P02/P05/P06/P08
hard/soft); unsupported-claims / factual safety ✓ (P03/P04); metadata/title/
description ✓ (P09 spam, P11 advertiser suitability, P12 title/hook
consistency); publishing prerequisites: **none as code** (the gate is
deterministic only; upload is Stage 11, design-only), but all honesty
guardrails apply; blocking vs review semantics ✓ (hard vs soft).

**Can Stage 9 approve what Stage 8 rejected? No.** P15 turns a Stage-8
REVIEW/REJECT (or missing/stale QA) into a Stage-9 REVIEW that can never
resolve to PASS; `run_policy_qa` aborts (exit 2) on plan↔video-result mismatch;
`evaluate_policy` raises `PolicyQAError` without a `target_ref`. It correctly
depends on upstream QA results. **PASS.**

---

## 7. TEST COVERAGE

Every Stage-8/9 test (10 files, all verified against pytest collection):

| File | Count | Covers |
|---|---|---|
| `test_video_qa_models.py` | 9 | id determinism/scoping, vocab, round-trips, config |
| `test_video_qa_checks.py` | 18 | every rule pos/neg, registry, record flags |
| `test_video_qa_validator.py` | 8 | verdict matrix, graded-not-raised, stable issue ids |
| `test_video_qa_stage.py` | 9 | pairing guard, CLI (defaults/explicit/REVIEW/REJECT/`--previous`/missing→nonzero), loader round-trip |
| `test_video_qa_safety.py` | 30 | no net/GPU/secret/Stage-9 vocab, sockets-blocked run, purity, denylist |
| `test_policy_qa_models.py` | 31 | ids, vocab, strictness-only classifier, bounded context |
| `test_policy_qa_checks.py` | 27 | each rule pos/neg incl. "harmless octopus" false-positive guard |
| `test_policy_qa_validator.py` | 25 | verdict matrix, prior-gate P15, graded-not-raised |
| `test_policy_qa_stage.py` | 11 | committed-fixture PASS, pairing, exit 1 (review/reject), exit 2 (mismatch), `--previous` chain |
| `test_policy_qa_safety.py` | 46 | no net/GPU/secret/Stage-10-11 vocab, sockets-blocked, purity |
| **Total** | **214** | |

Coverage present: fail-closed ✓; malformed-input ✓ (graded-not-raised in both
validators; missing-input → nonzero); cross-stage binding ✓ (P15 stale/missing,
pairing guards, `target_ref` echo — and `test_human_approval_*` consumes both
QA artifacts downstream); CLI ✓ (exit 0/1/2); state-transition ✓ (artifact +
state write to `QUALITY_CHECK`); security/safety ✓ (both suites).

**Important missing tests:** no exact-boundary edges (30/60s, 15/90s, 0.8
overlap); no `--out` custom-root test; no test asserting **exit 2 for
missing/corrupt input** (tests only assert "nonzero" — this is why BUG B1 is
unguarded); no single Stage-7→8→9→10 integration test; no
`state.json`-vs-artifact agreement test.

---

## 8. REAL vs MOCK

Neither stage uses Gemini, external APIs, network, media analysis, LLMs, or GPU.

- Stage 8: pure deterministic rules; zero provider code.
- Stage 9: `classifier.py` ships a `PolicyRiskClassifier` **protocol** with
  `classifier=None` default; it is advisory, sees a bounded 8k-char context,
  can only push toward REJECT, and is unused in every production path.
  Concluded **deterministic rules only** — no provider exists or is called
  (verified in source; the default needs no key/API). **PASS** — the seam is
  dormant design, not an external dependency.

---

## 9. ARTIFACTS

Committed (all verified valid this session; nothing modified):

- `data/mock/qa_result.json` — Stage-8 PASS, `qa_9b7d098451d3`,
  `target_ref=pp_e99181e73b5d`, 7/7 completed, 45.2s, 15 checks → proves a
  clean Stage-8 verdict on mock data.
- `data/mock/policy_qa_result.json` — Stage-9 PASS, `pr_9b7d098451d3`,
  `policy_rules_v1`, 15 checks → proves a clean pre-publication verdict over
  the mock chain.
- Demo runs `data/vqa-demo/` (`qa_1b31898a4988`) and `data/pol-demo/`
  (`pr_5d38af9365bd`) — CLI end-to-end PASS evidence.
- Matching upstream mocks (`script.json`, `fact_check_result.json`,
  `production.json`, `video_generation_result.json`) make the chain
  self-consistent.
- All are `mock=True` with empty URIs — no real media anywhere. **PASS.**

---

## 10. DOCUMENTATION CONSISTENCY

- PASS: README.md:26–27,92–99,138–139; PROJECT_PROGRESS.md §Stage 8 (line 765)
  / §Stage 9 (line 843); DATA_CONTRACTS.md §11/§11.1/§11.2 (lines 829–963);
  ARCHITECTURE.md §2/§4.4; VIDEO_PIPELINE.md §6; POLICY_AND_SAFETY.md — all
  agree with the implementation on status, checks, ids, paths, and exit codes.
- CONCERN — **Stage-numbering drift on FactCheckResult/Script** (confirmed as a
  repo-wide leak): DATA_CONTRACTS.md:934 and PROJECT_PROGRESS.md:889 call the
  inputs "Stage-5 `FactCheckResult` + Stage-4 `Script`" (FactCheck is
  canonically **Stage 3** per §7:293/§16 and the Stage-10 design doc); the
  Stage-9 package docstrings (`models.py:5`, `stage.py:3–4`,
  `__main__.py:17–18,72,74,78`) additionally call Script "Stage-3". Doc-only;
  ID bindings, not numbers, are what the code enforces.
- CONCERN — **GEMINI_PROVIDER_DESIGN.md header stale** (pre-existing): claims
  HEAD `b0f5f2d` / "1148 tests (4 skipped)" (real: `6a56390` / 1229 / 5) and
  "Stages 3 & 6 Gemini NOT IMPLEMENTED … Stage 3 next target" while its own §L
  documents the Stage-3 slice as implemented at HEAD. Not Stage-8/9-specific.

---

## 11. GIT STATE

- Branch: `feat/real-gemini-pipeline`
- HEAD: `6a56390 feat: implement Gemini fact checking and script QA`
- Working tree: clean for tracked files; 1 untracked file: `docs/AUDIT_QA_STAGES.md`
  (this document — nothing staged or committed).

---

## 12. FULL TEST SUITE (exact)

```
python -m pytest tests -q
→ 1229 passed, 5 skipped in 24.31s

python -m pytest tests/test_video_qa_* tests/test_policy_qa_* -q
→ 214 passed in 8.34s
  (Stage 8: 74 = models 9, checks 18, validator 8, stage 9, safety 30)
  (Stage 9: 140 = models 31, checks 27, validator 25, stage 11, safety 46)
```

---

## 13. BUG CLASSIFICATION

- **PASS** — both stages' gate logic: hard→REJECT, soft→REVIEW, clean→PASS,
  uncertainty-never-PASS; pairing guards; grade-don't-crash validators;
  provider/mock/media-honesty invariants; id/provenance binding vs §11;
  fixtures valid; documentation otherwise consistent; deterministic-only (no
  provider).
- **CONCERN** — no orchestrator (the stage chain is manual, CLI-only); `state.json`
  never records the QA verdict (PASS/REVIEW/REJECT indistinguishable from state
  alone); sequential non-transactional artifact→state writes; dormant
  classifier seam untested in production; boundary-edge and `--out` tests
  missing.
- **BUG B1** — CLI error-path exit codes violate the documented "2 = error"
  contract (empirically verified): `--result <missing>` → SystemExit, **exit 1**;
  `--result <corrupt.json>` → uncaught traceback, **exit 1**; policy
  `--qa-result <missing>` → uncaught `FileNotFoundError`, **exit 1** (and
  policy's fixtures-loader tolerates a missing QA as P15 while its path-loader
  throws — internally inconsistent); exit 2 is reached only from pairing
  mismatches. A broken run is thus indistinguishable from a REVIEW/REJECT by
  exit code. Untested. LOW severity (observability/robustness; nothing
  safety-relevant passes).

---

## 14. FINAL RECOMMENDATION

- **Stage 8: READY.**
- **Stage 9: READY.**

**What is actually DONE:** Stages 1–10 end-to-end as deterministic, mock-first,
offline gates; 214 QA tests green (74 + 140); full suite 1229 passed / 5
skipped; committed fixtures reproduce qa + policy PASS through real CLI runs
(`vqa-demo`, `pol-demo`); §11.1/§11.2 contracts implemented to spec;
Stage-7→8→9→10 id chain (`vg_→qa_→pr_→ha_`) bound and the Stage-10 package
consumes both QA artifacts.

**What remains incomplete:** any programmatic stage chaining
(`pipeline/orchestrator.py` is scaffold-only); failure/retry framing from §4.3
(pending/running/completed/failed) is not implemented anywhere; BUG B1's error
handling; boundary/`--out`/full-chain tests; real media + audio/subtitles/
assembly (design-only); Stage 11 upload + analytics (design-only); Gemini
providers exist for Stages 1/2/4 (+ the just-landed Stage-3/QA slice), none for
Stage 6.

**What to build next:** wire `pipeline/orchestrator.py` to chain the existing
CLIs (video_generation → qa.video → qa.policy → human_approval), advance state,
and stop on the first non-PASS — pure orchestration over existing deterministic
artifacts (no new providers, no media), which also gives the full-chain
integration test a home. Then fix BUG B1 and the docstring stage-numbering.

**Are Stages 8/9 supposed to remain deterministic/mock-only? Yes.** The
architecture (ARCHITECTURE.md: local PC runs QA; "QA gates have no seam —
deterministic" per GEMINI_PROVIDER_DESIGN.md:57; §11.2 honesty rules; the
advisory strictly-stricter classifier) explicitly and repeatedly restricts
these gates to deterministic rules over recorded artifacts. Nothing in the repo
plans Gemini/LLM/media analysis for video QA or policy QA.

---

| Stage | Status | Tests | Real/Mock | Major Concerns | Ready? |
|------|--------|-------|-----------|----------------|--------|
| 8 — Video QA | COMPLETE | 74 | Deterministic/mock-only | No orchestrator; `state.json` has no verdict; exit-code contract bug (B1); boundary tests missing | **Yes** |
| 9 — Policy/Publish QA | COMPLETE | 140 | Deterministic/mock-only (dormant advisory classifier, unused) | Same as 8 + stale docstring stage-numbers; P15 keep-REJECT guarantees already verified | **Yes** |

Full suite: `pytest tests -q` → **1229 passed, 5 skipped** (24.31s). Targeted QA
suites: **214 passed** (Stage 8 = 74, Stage 9 = 140).