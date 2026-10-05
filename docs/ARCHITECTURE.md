# Architecture

This document defines the target architecture for the automated faceless
YouTube Shorts channel. It is a **design document**; the pipeline up to and
including Stages 1–10 (topic selection → ... → human approval) is implemented
(see `README.md`), while the GPU worker, audio/subtitles/assembly, YouTube
publishing, and analytics remain design-only.

## 1. Purpose

Produce 30–60 second, 9:16 vertical explainer Shorts on a consistent
schedule with as little human involvement as possible, while keeping the
content evidence-backed, original, and policy-safe.

The local PC is the **orchestrator** (content system, state, assembly,
QA). Google Colab is initially the **GPU worker** for video generation,
and is replaceable later by another cloud GPU, a local GPU, or a
different video-generation system.

## 2. Target pipeline

```
topic discovery
→ seo research
→ topic selection / scoring
→ research
→ fact check
→ script generation
→ script QA
→ scene planning
→ production.json
→ video generation   [GPU worker: Google Colab]
→ audio
→ subtitles
→ assembly
→ video QA
→ policy / publish QA
→ human approval
→ YouTube upload
→ analytics feedback
```

Every stage reads its input from and writes its output to **structured
JSON files** (data contracts) under a per-run workspace in `data/`.
Stages communicate only through these files — never by calling each
other.

## 3. Components

| Component | Location | Role |
|---|---|---|
| Orchestrator | `pipeline/orchestrator.py` | Drives stages, tracks state, resumes runs |
| Run state | `pipeline/state.py` | Persisted per-run stage/status/artifacts |
| Agent stages | `agents/` | topic/seo/research/script/scene_planning/qa/human_approval logic |
| GPU worker | `video/generation/` | Consumes `production.json`, returns video artifacts |
| Audio | `video/audio/` | TTS / voice track |
| Subtitles | `video/subtitles/` | Caption generation |
| Assembly/editing | `video/editing/` | Final Short assembly |
| Prompts | `prompts/` | Prompt templates, versioned per stage |
| Config | `config/` + `.env` | Provider/worker settings, secrets |
| Contracts | docs + JSON schemas | Versioned data contracts |

Physical mapping of logical stages (e.g. fact check, policy) to module
paths will be decided when those stages are implemented; the design
treats them as independent stages regardless of which package hosts
them.

## 4. Key patterns

### 4.1 File-based data contracts
Each stage produces one or more contract JSON files. Consumers never
import producers. Contracts are versioned and carry provenance metadata
(creation time, run ID, stage, prompt version, model/provider). See
`DATA_CONTRACTS.md`.

### 4.2 Per-run workspace
A run owns a directory:

```
data/<run_id>/
├── state.json
├── research/      research package, fact-check results, sources, claims
├── scripts/       script, script-QA result
├── production/    scene plan, production.json, video-/policy-QA results
├── approval/      human approval record (human_approval_result.json)
└── videos/        generated assets and final Short
```

`pipeline/state.py` records which stages completed and which artifacts
each produced; `state.json` is the source of truth for resuming.

### 4.3 Resumability and failure recovery
- Stage statuses: `pending → running → completed | failed`.
- A stage persists its artifacts before the run is marked complete for
  that stage. On failure, the run keeps `state.json` in a non-terminal
  state.
- An **interrupted run** is detected when a previously `running` stage
  has no completion entry and a heartbeat/`updated_at` is stale.
- Recovery = start the orchestration **at the first non-completed stage**.
- Stages must be **idempotent enough to re-run safely**: outputs are
  written to the run workspace, and re-running an unfinished stage does
  not destroy downstream artifacts until the new output is committed.
- Per-stage **retries** are handled by the orchestrator wrapper around a
  stage call, with a configured max attempts; retriable vs. fatal
  failure classification is a stage responsibility exposed via a
  minimal stage result object.

### 4.4 Human approval gate
Publishing requires human approval until the pipeline has proven safe
and reliable. The gate sits between policy/publish QA and YouTube
upload. See `POLICY_AND_SAFETY.md`.

**Implemented as Stage 10** (`agents/human_approval/`): a machine-verifiable
record of an explicit human decision over the exact reviewed artifact
versions — 7 roles (`policy_qa`, video/policy QA targets, plan, script,
script QA, fact check), each bound by stable id + canonical `content_sha256`
digest, decisions `APPROVED / CHANGES_REQUESTED / REJECTED / PENDING`, a
fail-closed APPROVED (all hard gates clear, PENDING never writable), soft
REVIEW-only overrides, digest-based staleness checks, and supersede chaining.
An APPROVED decision advances run state to the existing generic
`Stage.PUBLISH`; non-approved decisions remain at `Stage.QUALITY_CHECK`. The
package is marked `manual_upload.ready` only for an approved decision — Stage 10
performs NO upload. Design:
`docs/HUMAN_APPROVAL_STAGE10_DESIGN.md`; contract: `docs/DATA_CONTRACTS.md`
§12.

## 5. Replaceability boundaries

| Boundary | Contract that must stay stable | Swappable implementations |
|---|---|---|
| Model provider | `ModelProvider` interface (`MODEL_STRATEGY.md`) | Different LLM providers/local models |
| GPU worker | `ProductionPlan` (`production.json`) + worker result envelope (`VIDEO_PIPELINE.md`) | Google Colab, cloud GPU, local GPU |
| Topic scoring | signal-based `TopicCandidate`/`TopicSelection` (`TOPIC_SELECTION.md`) | future bandit/ML optimizers |

The content system never calls a video-generation provider directly; it
writes `production.json` and waits for a worker result.

## 6. Process boundaries

- **Local PC:** orchestration, topic/SEO/research/script/scene stages, QA,
  assembly of assets into the final Short, publishing client logic.
- **GPU worker (Colab):** receives `production.json` plus input assets
  via a staging/exchange mechanism (upload), runs generation, and
  returns generated clips (download). The exact transport (e.g. Drive,
  direct transfer, worker API) is an **open decision**; the orchestrator
  only depends on a narrow worker interface.
- **Content system vs. video system:** the split is exactly at
  `production.json`.

## 7. Automation model

- Initial trigger is **manual** (`python -m pipeline.orchestrator`).
- Scheduling (e.g. daily 20:00) is a future orchestrator feature; the
  design only requires that a run is a self-contained, resumable unit so
  a scheduler can start it later. No Windows scheduling is implemented
  or designed beyond that.

## 8. Implementation order (guided by dependencies)

1. Model provider abstraction + one functioning agent stage (research)
2. Fact check stage and contracts
3. Script + script QA
4. Scene plan → `production.json`
5. GPU worker interface (Colab adapter) + video generation
6. Audio, subtitles, assembly, video QA
7. Policy/publish QA + human approval + YouTube upload
8. Analytics collection and feedback into topic selection
9. Scheduling

Implementation status as of the Stage-10 checkpoint: items 1–7 are implemented
through the human approval gate (Stages 1–10). Stages 1–4 have opt-in real
Gemini/Firecrawl paths; Stages 5–10 are deterministic/mock-first. The real
Colab GPU worker, audio/subtitles/assembly, YouTube publishing (Stage 11),
analytics, and scheduling remain future work.

## 9. Open decisions

- GPU worker transport (how `production.json`/assets/files move between
  local PC and Colab).
- Which model providers will back agents (deferred by design).
- File storage/retention policy for runs (cleanup, archives).
- Whether fact-check/policy stages live under `agents/qa/` or as new
  packages under `agents/`.
- Where downloaded/source-material assets are cached long term.