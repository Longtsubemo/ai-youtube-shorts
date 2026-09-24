# Audit — Real Gemini Script Generation (repo Stage 4)

**Date:** 2026-09-24
**Commit under audit:** `b0f5f2d feat: implement real gemini script generation`
**Method:** inspection/review only — no files modified, nothing committed.

> **Key correction up front:** the audited implementation is **NOT uncommitted**.
> It was committed as `b0f5f2d` and the working tree is clean. All findings below
> were verified against the actual files.

---

## 1. Canonical pipeline

Sources: `pipeline/state.py:24-41` (`Stage` enum), `docs/PROJECT_PROGRESS.md:53-57`,
`docs/ARCHITECTURE.md:22-41`, `README.md:19-28`.

**The repository numbers Script Generation as Stage 4** (matches
`Stage.SCRIPT_GENERATION`). "Stage 3" in user-facing shorthand = Script Generation
is the **repo's Stage 4**.

| # | Name | Path | Status |
|---|---|---|---|
| 1 | Topic Selection | `agents/seo/`, `TOPIC_SELECTION` | COMPLETE — deterministic mock + **real Gemini** (`GeminiTopicProvider`, `--provider gemini`) |
| 2 | Research | `agents/research/`, `RESEARCH` | COMPLETE — mock + **real Gemini** (`GeminiResearchProvider`) + **Firecrawl** retriever |
| 3 | Fact Checking | `agents/qa/fact_checker/`, `FACT_CHECK` | COMPLETE — deterministic rules (no Gemini) |
| 4 | Script Generation | `agents/script/`, `SCRIPT_GENERATION` | COMPLETE — deterministic mock + **real Gemini** (`GeminiScriptProvider`) |
| 5 | Script QA | `agents/qa/script/`, `SCRIPT_QA` | COMPLETE — deterministic |
| 6 | Scene Planning | `agents/scene_planner/`, `SCENE_PLANNING` | COMPLETE — deterministic |
| 7 | Video Generation | `agents/video_generation/`, `VIDEO_GENERATION` | COMPLETE — mock-first, deterministic |
| 8 | Video QA | `agents/qa/video/`, `QUALITY_CHECK` | COMPLETE — deterministic |
| 9 | Policy/Publish QA | `agents/qa/policy/`, `QUALITY_CHECK` | COMPLETE — deterministic |
| 10 | Human Approval | `agents/human_approval/`, `PUBLISH` | COMPLETE — deterministic (no upload) |
| — | YouTube publishing / audio / assembly / analytics | — | NOT IMPLEMENTED (design-only) |

---

## 2. Architecture audit

**A. Existing interface implemented:**
`ScriptProvider` protocol — `agents/script/provider.py:39-54`:
`generate(package, config=None, run_id="", created_at="") -> ScriptGenerationResult`
reusing the existing generation contract `ScriptGenerationResult` / statuses /
`ScriptConfig` (`agents/script/script_generator.py:56-108`).
`MockScriptProvider` implements the same protocol.

**B. Gemini foundation reuse:**
Yes — `GeminiScriptProvider` calls the shared `GeminiClient.generate_structured`
(`providers/gemini/script_provider.py:168` on `providers/gemini/client.py:120`),
same `system_prompt` / `response_schema` mechanism as Stages 1–2. Error typology
reused from `providers/gemini/errors.py`.

**C. Direct SDK imports in stage/agent code:**
None. `agents/script/` imports only the seam (`provider.py`) and stage modules.
The `google-genai` SDK is imported only inside the gemini branch of
`build_script_provider` (`provider.py:193-200`). Verified by subprocess tests
asserting `google.genai` never enters `sys.modules`
(`tests/test_script_provider_factory.py:40-66`,
`tests/test_gemini_script_provider.py:334-343`).

**D. Provider selection:**
CLI `--provider mock|gemini` (default `mock`, `agents/script/script_generator.py:407-411`);
`--provider gemini` requires `--research` (exit 2 otherwise, lines 438-440).
`build_script_provider(name, env=, config=)` (`provider.py:177-202`); `gemini`
requires `GEMINI_API_KEY` (else `GeminiConfigurationError`); unknown name →
`ScriptProviderError`. The `--result` fact-check path is untouched and default.

**E. Mock behavior preserved:**
`MockScriptProvider` is the default and reuses `MockScriptGenerator._plan`
(`provider.py:93`). The `--result` path (`_run_from_fact_check`) is the unchanged
pre-existing flow. Default CLI run has no key/SDK requirement.

**F. Input artifact:**
A `ResearchPackage` JSON via `--research` (`agents/script/script_generator.py:476-477`).
Only the package's claims (id / type / verification / evidence tiers) are rendered
into the prompt (`providers/gemini/script_provider.py:181-205`). No
`FactCheckResult` on this path.

**G. Output artifact:**
`Script` written to `<out>/<run_id>/scripts/script.json` and `state.json` advanced
to `script_generation` via `_write_artifact`
(`agents/script/script_generator.py:535-548`) — only when `status == generated`.

**H. Structured-output validation (three layers):**
1. Pydantic `ScriptResultSchema` — `purpose` is a `Literal`, non-empty title/lines
   (`script_provider.py:62-76`).
2. `ScriptLine` / `Script.validate()` (`agents/script/models.py:115-232`).
3. Deterministic `validate_script_from_package` final gate
   (`agents/script/validator.py:254-359`).

**I. Malformed output handling:**
`GeminiMalformedResponseError` is retried a bounded number of times
(`malformed_retries`, default 1 → up to 2 total attempts) in `_request_schema`
(`script_provider.py:157-176`), then re-raised as a `GeminiProviderError` — never a
raw SDK exception. **Schema errors are NOT retried** (comment at lines 160-163).

**J. API failure handling:**
`client.generate_structured` classifies SDK/network failures into the typed hierarchy
(`client.py:343-369`: auth → `GeminiAuthenticationError`, server → `GeminiAPIError`,
timeout → `GeminiTimeoutError`, transport → `GeminiAPIError`, else
`GeminiUnexpectedError`). The provider propagates these; the CLI converts any
`ScriptProviderError` / `Exception` to exit 2 with **no artifact written**
(`script_generator.py:481-489`).

---

## 3. Script contract / data binding

Verified from `data/857acc781211-firecrawl-lite/scripts/script.json` +
`data/857acc781211-firecrawl-lite/research/research_package.json`:

- **`package_id`:** `rp_ed6872e5bcc3` on both sides. Binding enforced at
  `validator.py:278`.
- **`claim_refs`:** each line resolves to a package claim id
  (ln_02 → `cl_293cfd72f1`, ln_03 → `cl_cc34d568df`, ln_04 → `cl_cba0e1d80d`);
  framing lines carry `[]`.
- **`used_claim_ids`:** exactly the 3/3 package claims; `unused_approved_claim_ids`
  empty (whole package is the working set). Exact/disjoint/complete accounting
  enforced at `validator.py:332-340` and `models.py:219-227`.
- **`result_id`:** `""` — **intentional** for this path. `Script.result_id`
  denotes the upstream `FactCheckResult.result_id`; `validate_script` enforces the
  equality on the fact-check path (`validator.py:149-153`). On the package path
  there is no fact-check yet, so `""` is the honest sentinel (documented in
  `provider.py` docstring; asserted in `tests/test_gemini_script_provider.py:123`
  and `test_script_cli.py:125`). **It becomes a fc-id only if/when script generation
  is chained after repo Stage 3 Fact Checking.** Not a defect.
- **Provenance:** `generated_by="gemini_script_provider_v1"`,
  `source="GeminiScriptProvider"`, `provider="gemini"`, `model="gemini-3.5-flash-lite"`,
  `prompt_version="script/script-v1"`; envelope `produced_by` identical.
- **Duration:** deterministic word-count calculations (e.g. ln_02, 23 words → 9.2s);
  declared total 36.4s == Σ line durations (enforced by `models.py:212-217`).
- **Layout:** 6 chronological lines (`ln_00..ln_05`), valid `purpose` values,
  hook/context/CTA carry no refs, factual lines carry refs. Conforms to
  `docs/DATA_CONTRACTS.md` §8.

---

## 4. Deterministic safety gate

`validate_script_from_package` — `agents/script/validator.py`.

| Check | Verdict | Location |
|---|---|---|
| Unknown claim references | PASS | `validator.py:305-307` |
| Unsupported / off-coverage claims | PASS | `validator.py:314-319` (token coverage < threshold) |
| Dropped hedging | PASS | `validator.py:326-329` (`_hedge_dropped`; word list 51-57) |
| Strengthened certainty | PASS | `validator.py:320-325` (`_strengthened`; word list 41-47) |
| Package mismatch | PASS | `validator.py:278-282` (`script.package_id == package.package_id`) |
| Invalid used/unused accounting | PASS | `validator.py:332-340` + `models.py:219-227` |

Minor permissiveness (non-BUG): a **framing** line carrying valid package refs is
not rejected (framing refs simply count toward `used`). The prompt hard-rules `[]`
for framing lines (`prompts/script/script-v1.txt:96-97`); the validator is the
floor, not the prompt.

---

## 5. Test audit

Verified by pytest collection: **49 new tests** in 5 files.

| File | Count | Covers |
|---|---|---|
| `tests/test_script_provider_factory.py` | 6 | default=mock, unknown raises, no-key raises, SDK isolation subprocess, lazy/offline gemini construction, env whitelist |
| `tests/test_mock_script_provider.py` | 7 | generated shape (+`result_id=""`, package binding), run-id default, too-few → insufficient, too-short → insufficient, passes package validator, determinism, sockets-blocked |
| `tests/test_gemini_script_provider.py` | 22 | generated output (5); final-gate blocks: unknown ref / off-coverage / dropped hedge / retained hedge / strengthened certainty (5); schema strictness + malformed retry (`attempt_count==2`) + API error mapping + missing prompt (8); provenance/wiring (3); SDK isolation (1) |
| `tests/test_gemini_script_security.py` | 7 | no key in artifact / auth error / unexpected error / `export_redacted`; mock no-key no-network; factory no-key raises; mock factory ok |
| `tests/test_script_cli.py` | 7 | mock default artifact, gemini provenance stamp, key-missing → exit 2 no write, generate-API-error → exit 2 no write, gemini-requires-research, insufficient → 0 no write, missing package → `SystemExit` |

Integration: `test_real_stage4_script_with_gemini_provider`
(`tests/test_gemini_integration.py:82-107`) gated by `GEMINI_INTEGRATION_TESTS=1`
+ `REQUIRES_KEY` skipif.

**Missing (gaps identified, not addressed):**
- **No direct unit tests for `validate_script_from_package`.** All 22 tests in
  `tests/test_script_validator.py` target only `validate_script` (fact-check
  path); the package gate is exercised only end-to-end via the providers.
  Missing direct coverage: package_id mismatch, used/unused coverage-over-package,
  unknown purpose, framing-line-with-refs permissiveness.
- **No CLI test asserting real `state.json` advancement/save** — CLI tests stub
  `PipelineState.save` (`test_script_cli.py:72-78`).
- No negative test that `result_id=""` is tolerated by the package validator
  (only asserted as provider/CLI behavior).

---

## 6. Real smoke test audit

All three files exist and were inspected directly:

- **Script exists & valid:** schema `Script/v1`, stage `script_generation`, 6
  unique lines, valid purposes, durations sum to declared 36.4s, provenance
  present → `Script.validate()` passes.
- **`package_id` matches:** `rp_ed6872e5bcc3` on both sides.
- **3/3 claims used:** used = {`cl_293cfd72f1`, `cl_cc34d568df`, `cl_cba0e1d80d`}
  = the 3 package claims; refs resolve.
- **Provenance / model / prompt:** `gemini` / `gemini-3.5-flash-lite` /
  `script/script-v1`, `generated_by=gemini_script_provider_v1`.
- **`result_id`:** `""` — correct for this path (see §3).
- **State advancement:** `state.json` `current_stage="script_generation"`, artifacts
  contain both `research` and `script_generation` entries. Separation of concerns:
  run continued from Stage 2, matching the single-run convention used by
  `agents/research/research_agent.py:121-131`.
- **No API key:** none present in any of the three files.

---

## 7. Test suite run

```
python -m pytest tests -q
```
Result: **1133 passed, 4 skipped** (~12-18s). The 4 skips are the key-gated
real-Gemini + Firecrawl integration cases.

---

## 8. Git / change audit

`git status --short` → **empty (clean)**.
`git log -1 --oneline` → `b0f5f2d feat: implement real gemini script generation`.

The full 16-file change set is in commit `b0f5f2d`:

- **Implementation:** `agents/script/provider.py` (new), `agents/script/script_generator.py`,
  `agents/script/validator.py`, `providers/gemini/script_provider.py` (new),
  `providers/gemini/__init__.py` (modified)
- **Prompts:** `prompts/script/script-v1.txt` (new)
- **Tests:** `tests/test_script_provider_factory.py`, `tests/test_mock_script_provider.py`,
  `tests/test_gemini_script_provider.py`, `tests/test_gemini_script_security.py`,
  `tests/test_script_cli.py` (new); `tests/fact_check_test_utils.py`,
  `tests/test_gemini_integration.py` (modified)
- **Documentation:** `README.md`, `docs/GEMINI_PROVIDER_DESIGN.md`,
  `docs/PROJECT_PROGRESS.md`
- **Generated artifacts:** none staged — `data/857acc781211-firecrawl-lite/**` and
  `.env` are gitignored (confirmed via `git check-ignore`); only committed mock
  fixtures under `data/mock/`.
- **Unrelated changes:** none.

Nothing remaining to stage; nothing unsafe is tracked.

---

## 9. Documentation consistency

Consistent: README, PROJECT_PROGRESS, and GEMINI_PROVIDER_DESIGN all use
**Stage 4 = Script Generation** (canonical). Real-vs-mock and the new provider are
described accurately.

Stale / inaccurate (non-blocking):
- `docs/GEMINI_PROVIDER_DESIGN.md:23-25` still says HEAD `b57f2ee` — actual HEAD
  is `b0f5f2d`.
- `docs/GEMINI_PROVIDER_DESIGN.md:7-8` says the real Stage-2 key-gated run is "the
  only unperformed verification" — but the smoke consumed a real Stage-2
  Gemini+Firecrawl artifact (`research_package.json`, `retrieved_by: firecrawl`),
  so a real Stage-2 run did happen.
- `README.md:170-172` says the Stage-4 path renders "the approved research package
  claims" — there is no approval step on the package path; claims are provisional.
- Prompt-internal nit: `prompts/script/script-v1.txt:51` invites a hook "with refs"
  while the hard rule (96-97) mandates `[]` — cosmetic; the validator permits both.

None of these misstate the stage numbering.

---

## 10. Final verdict

| Area | PASS / CONCERN / BUG | Explanation |
|---|---|---|
| Architecture | PASS | Protocol seam (`ScriptProvider`) + factory + CLI opt-in; mock default intact; no SDK in agents |
| Gemini integration | PASS | Reuses shared `GeminiClient.generate_structured`; typed error hierarchy; provenance stamping |
| Script contract | PASS | Envelope/layout/purpose/duration conform to `models.py` + DATA_CONTRACTS §8; `result_id=""` intentional |
| Claim binding | PASS | `package_id` enforced; used/unused exact and complete; refs resolve; 3/3 smoke claims used |
| Deterministic validator | PASS | All 6 classes blocked; small permissiveness on framing-line refs (floor = validator, prompt stricter) |
| Error handling | PASS | Malformed → bounded retry → typed error; API errors → typed errors → exit 2, no artifact |
| Security | PASS | No key in artifact/errors/`export_redacted`; SDK isolation subprocess-tested; `.env` and run data gitignored |
| Tests | PASS | 49 offline + 1 gated integration; gaps: no direct `validate_script_from_package` unit tests, no real-save CLI assertion |
| Real smoke test | PASS | Artifact exists, valid, bound, provenance-correct, state advanced, no key |
| Documentation | CONCERN | Stale HEAD (`b57f2ee`), stale "Stage-2 real run not performed" note, loose "approved claims" wording — all minor; numbering correct |
| Git cleanliness | PASS | Working tree clean; all changes in `b0f5f2d`; no artifacts/secrets committed |

**A. What was actually implemented:**
Full Stage-4 real Gemini Script Generation — `ScriptProvider` seam +
`MockScriptProvider` + `GeminiScriptProvider` + package validator +
CLI opt-in + versioned prompt + state/artifact writing — with 49 offline tests,
1 gated integration test, and documentation updates.

**B. What was actually verified:**
1133 passed / 4 skipped; the real smoke artifact is intact and valid (3/3 claims,
correct provenance/model/prompt, matching `package_id`, correct state advance, no
key); commit `b0f5f2d` contains exactly the 16 intended files.

**C. What remains uncertain:**
1. Whether the package-path run dir can be continued into Stage 5+ (downstream
   stages whitelist-join a `FactCheckResult.approved_claims`, which this run lacks —
   the state jump `research → script_generation` skips `FACT_CHECK`).
2. Free-tier quota availability for the `.env` default model (`gemini-3-flash`).
3. Duration bounds are warnings, so a short script can still be committed with
   warnings.

**D. Bugs:** none found.

**E. Non-blocking concerns:**
Validator unit-test gap; stale doc HEAD / Stage-2 note; prompt hook-refs wording;
framing-line refs permissiveness; README "approved" wording.

**F. Ready to commit:** **already committed** (`b0f5f2d`). If the commit message /
timeline should reflect "Stage 4" per repo numbering, that is cosmetic; the change
set is complete, verified, and appropriately scoped (no Stage 5+ code touched).