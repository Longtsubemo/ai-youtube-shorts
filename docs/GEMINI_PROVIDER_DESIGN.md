# Gemini Provider Integration — Design Report

**Status:** RESEARCH / DESIGN ONLY — **NOT IMPLEMENTED.**

This document records the inspection/design pass over the existing Stages 1–6
implementation and the exact plan for introducing a real Gemini provider
without breaking the deterministic mock providers, data contracts, provenance,
tests, or the Stage 1–6 boundaries and determinism.

Stages 1–10 are complete and deterministic (HEAD `0b054d44`), 920 tests pass.
Stage 11 (YouTube upload) remains out of scope.

---

## A. Key finding: the LLM seam already exists

Every LLM-worthy stage already has a `Protocol` + deterministic `Mock`
implementation, and its stage runner already accepts the provider through an
optional injected argument. A real Gemini provider is therefore a **new adapter
class implementing an existing protocol**, passed explicitly — **not** a change
to any stage logic, mock, contract, ID rule, test, or default.

| Stage | Name | Seam (protocol / mock) | Injection point |
|---|---|---|---|
| 1 | Topic selection | **no seam — deterministic** (weighted scorer + seeded `TopicSelector`, `config/topic_selection.yaml`) | — |
| 2 | Research | `ResearchProvider` / `MockResearchProvider` (`agents/research/provider.py`) | `run_research(selection, provider=None, ...)` |
| 3 | Fact checking | `EvidenceProvider` / `MockEvidenceProvider` (`agents/qa/fact_checker/evidence.py`) + `Verifier` / `MockVerifier` (`agents/qa/fact_checker/verifier.py`) | `run_fact_check(package, provider=None, verifier=None, ...)` |
| 4 | Script | `ScriptGenerator` / `MockScriptGenerator` (`agents/script/script_generator.py`) | `run_script_generation(..., generator=...)` |
| 5 | Script QA | **no seam — deterministic 13 checks** (`agents/qa/script/checks.py`) | — |
| 6 | Scene planning | `ScenePlanner` / `MockScenePlanner` (`agents/scene_planner/planner.py`) | `run_scene_planning(..., planner=...)` |

Stages 1 and 5 have no LLM seam **by design** and stay deterministic. Stage 1
is scored topic selection (direction comes from config, not an LLM). Stage 5 is
the machine QA gate for the script. Neither changes.

## B. Target: what a “real Gemini provider” adds

Following `docs/MODEL_STRATEGY.md` — a provider is never hard-coded anywhere
agents import; agents depend only on the abstract `ModelProvider` interface.
Gemini is implemented as **one concrete `ModelProvider` adapter** plus
**per-stage protocol adapters** that wrap it. No stage package ever imports the
Gemini SDK.

New layer (not a new stage):

```
agents/llm/
├── __init__.py
├── gateway.py        ModelProvider protocol + ModelRequest/ModelResponse
│                     (canonical implementation of docs/MODEL_STRATEGY.md §2)
├── gemini.py         GeminiModelProvider: maps gateway ↔ google-generativeai
│                     (lazy import of the SDK; key from .env GEMINI_API_KEY)
├── prompts.py        resolves system_prompt_ref → prompts/<stage>/<name>.<ver>
├── builders.py       build_{research,evidence,verifier,script,scene}_provider()
│                     factory: stage → protocol adapter from config/env
tests/
├── test_model_gateway.py    gateway contract, without any provider
├── test_gemini_provider.py  offline adapter tests (stubbed client)
├── test_builders.py         factory wiring, no SDK import
└── llm_test_utils.py        FakeModelProvider + recorded fixtures
```

### `ModelProvider` (gateway)

`agents/llm/gateway.py` implements the interface sketched in
`docs/MODEL_STRATEGY.md` §2:

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
`produced_by.model` (`DATA_CONTRACTS.md` §1).

### Per-stage protocol adapters

Each adapter implements the stage’s **existing** protocol and receives **only
the same inputs the mock receives**. Output is validated by the stage’s existing
validator/gates before the artifact can be persisted.

| Adapter | Protocol it implements | Input bound | Validator it must pass |
|---|---|---|---|
| `GeminiResearchProvider` | `ResearchProvider` | `TopicDescriptor` (one topic, one evidence request); returns `ResearchPackage` envelope | `agents/research/` package-building rules (provisional-only claims) |
| `GeminiEvidenceProvider` | `EvidenceProvider` | `Source`; evidence constrained to the source’s own content | `MeaningfulEvidence` / `SUPPORTING_EVIDENCE_TYPES` bounds |
| `GeminiVerifier` | `Verifier` | **strictly `{claim, supplied evidence, source metadata}`** — never outside/world knowledge (`agents/qa/fact_checker/verifier.py` docstring); returns `ClaimAssessment` | verdict vocabulary `DIRECT/PARTIAL/INSUFFICIENT/NONE/CONTRADICTORY` |
| `GeminiScriptGenerator` | `ScriptGenerator` | **only approved claims** (Stage 3 `approved_claims`); no raw research | `validate_script` (token coverage / no-hallucination / certainty preservation, deterministic) |
| `GeminiScenePlanner` | `ScenePlanner` | `Script` (post Script-QA PASS); returns `ScenePlanningResult` | `validate_production_plan` rules r01–r20 |

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
- Keeping the Gemini SDK **only inside `agents/llm/`** preserves the same
  guarantee for Stages 1–6: those packages stay import-free of the SDK,
  network-free, and env-free. When Stage 1–6 safety tests are added later, the
  audit window is unchanged.
- New `agents/llm/` tests are offline: `FakeModelProvider` + recorded
  responses, `socket.blocked`, no network, no real key. `GeminiModelProvider`
  is tested against a stubbed client that never talks to Google.
- Determinism: every existing test keeps using mocks; new tests cover only the
  new layer + adapter wiring.

## G. Explicit statement

**Gemini provider integration is NOT IMPLEMENTED — design only.** No new files,
no SDK, no config changes, no requirement changes. Stage 1 and 5 remain
deterministic with no seam; Stages 2, 3, 4, 6 keep mock-first defaults; all
contracts, provenance schemes, ID rules, and the 920-test suite are preserved.

---

## H. Adoption checklist (when implementing — not now)

1. Add `agents/llm/gateway.py` — `ModelProvider` / `ModelRequest` /
   `ModelResponse` (canonicalizing `MODEL_STRATEGY.md` §2).
2. Add `agents/llm/gemini.py` — `GeminiModelProvider` (lazy import, `.env`
   `GEMINI_API_KEY`).
3. Add the 5 stage adapters (`GeminiResearchProvider`, `GeminiEvidenceProvider`,
   `GeminiVerifier`, `GeminiScriptGenerator`, `GeminiScenePlanner`) behind the
   existing protocols.
4. Add `agents/llm/builders.py` + CLI/env opt-in (`--provider gemini`);
   defaults unchanged.
5. Add `prompts/<stage>/<name>.<ver>` templates + `config/` provider YAML
   (`provider: mock | gemini`, model, prompt refs).
6. Add offline tests (`FakeModelProvider`, stubbed Gemini client,
   `socket.blocked`) — no network, no `test_*_safety.py` changes to Stages 1–6.
7. Record provenance (`provider` / `model` / `prompt_version`) per
   `DATA_CONTRACTS.md` §1 on Gemini-produced artifacts only.

## I. Open decisions (implementation time)

1. Validation dependency: stdlib-only schema draft vs. a schema library
   (`MODEL_STRATEGY.md` §8).
2. JSON extraction: prompt-only vs. provider structured-output decoders.
3. Retry/backoff and token-budget policy per stage.
4. Prompt/version naming convention (`research/facts-v1`, …).
5. Whether Gemini Search grounding is used for research/evidence, or sources
   remain external (rule: model proposes, never the fact authority).