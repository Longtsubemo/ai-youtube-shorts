# AI YouTube Channel

Automated pipeline that researches a topic, analyzes SEO, generates a
script, plans a scene-by-scene production, generates video on a GPU
worker, assembles the final YouTube Short, and eventually publishes it —
end to end.

> **Status: implementing, stage by stage.** Ten stages are implemented with
> deterministic, network-free mock providers: topic selection, research, fact
> checking, script generation, script QA, scene planning, video generation,
> video QA, policy/publishing QA, and the human approval gate. Voice/audio,
> assembly, publishing, and analytics are not implemented yet.

## Long-term pipeline

```
topic discovery
→ SEO research
→ topic selection / scoring          Stage 1 ✅ implemented
→ research                           Stage 2 ✅ implemented
→ fact checking                      Stage 3 ✅ implemented
→ script generation                  Stage 4 ✅ implemented
→ script QA                           Stage 5 ✅ implemented
→ scene planning / production JSON    Stage 6 ✅ implemented
→ video generation (mock-first)       Stage 7 ✅ implemented
→ video QA                            Stage 8 ✅ implemented
→ policy / publishing QA              Stage 9 ✅ implemented
→ human approval                      Stage 10 ✅ implemented
→ Google Colab video generation (GPU worker)
→ voice / audio
→ video assembly
→ quality check
→ YouTube upload
→ analytics feedback
```

## Architecture

- **Local PC = orchestrator.** A central orchestrator drives the pipeline
  and keeps per-run state.
- **Replaceable GPU worker.** Google Colab starts as the video-generation
  worker; the interface is kept generic so a local GPU or another
  provider can be swapped in.
- **File-based data contracts.** Agents do not call each other directly.
  Each stage reads its input from and writes its output to structured
  JSON files under `data/`, so stages stay decoupled.
- **State per run.** `pipeline/state.py` models a single production run;
  the current orchestrator writes that state into a per-run directory
  under `data/`.
- **Prompts kept separate.** Prompt templates live in `prompts/`, not in
  agent code.
- **Manual trigger for now.** Scheduling is out of scope until the
  pipeline is stable.
- **Free / open-source first.** New dependencies and tools are chosen
  with that in mind.

## Repository layout

```
agents/            Implemented stage agents (seo, research, qa/*, script, scene_planner, video_generation, human_approval) + future ones
pipeline/          Orchestrator scaffold and per-run state model
video/             Future video build steps (generation, editing, subtitles, audio)
data/              Working artifacts per run + committed mock fixtures under data/mock/
prompts/           Prompt templates per agent
config/            Configuration (populated later; secrets go in .env)
tests/             Test suite for the implemented stages (920 passing)
```

## Current development stage

Implemented stages (all deterministic, mock-backed, offline):

- **Stage 1 — Topic Selection** (`agents/seo/`): signal-scored topic
  candidates + selection.
- **Stage 2 — Research** (`agents/research/`): evidence-backed claims +
  sources, provenance.
- **Stage 3 — Fact Checking** (`agents/qa/fact_checker/`): per-claim
  PASS/REVIEW/REJECT gating; `approved_claims` is what downstream may use.
- **Stage 4 — Script Generation** (`agents/script/`): approved claims only
  → validated 30–60s Short script with per-line claim traceability.
- **Stage 5 — Script QA** (`agents/qa/script/`): deterministic quality
  control (13 checks, 9 hard gates, weighted score) → PASS / REVISE /
  REJECT with actionable issues; never rewrites the script.
- **Stage 6 — Scene Planning** (`agents/scene_planner/`): QA-passed script →
  provider-independent versioned `ProductionPlan` (`production.json`), the
  stable CONTENT → VIDEO boundary; grounded in approved claims only.
- **Stage 7 — Video Generation** (`agents/video_generation/`): mock-first,
  provider-independent `VideoGenerationResult` per plan scene; every artifact
  is explicitly flagged `mock` — nothing looks like real video. Offline,
  GPU-free, API-free; resume/retry supported.
- **Stage 8 — Video QA** (`agents/qa/video/`): deterministic quality gate
  (15 checks) over the generated result → PASS / REVIEW / REJECT with
  actionable, artifact-traced issues; provider-isolation audit included. Never
  inspects media bytes (none exist yet).
- **Stage 9 — Policy / Publishing QA** (`agents/qa/policy/`): deterministic
  pre-publication policy/safety gate (15 rules P01–P15) over the full
  Script → FactCheck → ProductionPlan → VideoGeneration → VideoQA chain →
  `PolicyQAResult` (`policy_qa_result.json`); hard errors REJECT, soft errors
  REVIEW, uncertain never PASS; explicitly **NOT** a YouTube compliance
  guarantee, and human approval / publishing stay later stages.
- **Stage 10 — Human Approval** (`agents/human_approval/`): a
  machine-verifiable record of an explicit human decision over the exact
  reviewed artifact versions (role + stable id + canonical `content_sha256`
  digest). Decisions `PENDING / APPROVED / CHANGES_REQUESTED / REJECTED` (a
  disjoint brand from QA's PASS/REVIEW/REJECT). `APPROVED` fail-closes on any
  blocking condition (missing required artifact, Stage-8/9 hard REJECT, video
  not completed, unsatisfied review overrides); CLI refuses to write `PENDING`
  (no written decision = no approval). Soft REVIEW overrides are bound to the
  reviewed artifact id; hard REJECT is never overrideable. Staleness is
  detected by re-digesting current artifacts (`--check --approval <path>`),
  and re-review chains `supersedes_approval_id`. Advances state to the generic
  `Stage.PUBLISH`; the endpoint is a **MANUAL upload** (`manual_upload.ready`)
  — Stage 10 performs no YouTube/network/oauth work.

Not yet implemented: voice/audio, video assembly, publishing (Stage 11),
analytics.

## Running the implemented stages

Requires Python 3.10+.

```powershell
# optional: create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

# run per-stage demos against deterministic mock data:
python -m agents.seo.topic_selector          # Stage 1: topic selection
python -m agents.research.research_agent     # Stage 2: research package
python -m agents.qa.fact_checker             # Stage 3: fact check
python -m agents.script.script_generator     # Stage 4: script generation
python -m agents.qa.script                   # Stage 5: script QA
python -m agents.scene_planner               # Stage 6: scene planning (production.json)
python -m agents.video_generation            # Stage 7: video generation (mock artifacts)
python -m agents.qa.video                    # Stage 8: video QA (PASS/REVIEW/REJECT)
python -m agents.qa.policy                   # Stage 9: policy/publishing QA (PASS/REVIEW/REJECT)
python -m agents.human_approval              # Stage 10: human approval (APPROVED/CHANGES_REQUESTED/REJECTED)

# run the whole test suite (920 tests passing)
python -m pytest tests -q
```

Each CLI writes its artifact under `data/<run_id>/` and records run state
in `state.json`.

## Design documentation

The technical design is documented in `docs/`. Stages 1–10 are implemented;
the video-worker and marketplace docs remain design-only:

| Doc | Covers |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, pipeline stages, resumability, replaceability boundaries |
| [CONTENT_STRATEGY.md](docs/CONTENT_STRATEGY.md) | Channel identity, pillars, fixed 30–60 s story format |
| [TOPIC_SELECTION.md](docs/TOPIC_SELECTION.md) | External signals, cold start, exploration/exploitation |
| [RESEARCH_PIPELINE.md](docs/RESEARCH_PIPELINE.md) | Evidence-backed claims, source credibility tiers, claim/source/date model |
| [FACT_VERIFICATION.md](docs/FACT_VERIFICATION.md) | Separate fact-check stage: PASS/REVIEW/REJECT gating |
| [POLICY_AND_SAFETY.md](docs/POLICY_AND_SAFETY.md) | YouTube policy checks (implemented gate), human approval gate |
| [HUMAN_APPROVAL_STAGE10_DESIGN.md](docs/HUMAN_APPROVAL_STAGE10_DESIGN.md) | Stage-10 design: decision vocabulary, hard gates, digests, overrides |
| [VIDEO_PIPELINE.md](docs/VIDEO_PIPELINE.md) | Shorts target, production.json interface, Colab worker + swaps |
| [DATA_CONTRACTS.md](docs/DATA_CONTRACTS.md) | The versioned JSON contracts and artifact envelope |
| [MODEL_STRATEGY.md](docs/MODEL_STRATEGY.md) | Abstract model interface; coding model vs. production LLMs |
| [ANALYTICS.md](docs/ANALYTICS.md) | Future analytics interface feeding topic selection |

## Roadmap

1. Topic discovery and SEO research
2. Topic selection / scoring
3. Research → fact checking → script → script QA → scene planning → video
   generation → video QA → policy QA (Stages 1–9)
4. Human approval gate (Stage 10)
5. Real GPU/Colab video worker consuming the `production.json` contract
6. Voice, subtitles, video assembly
7. YouTube upload and analytics feedback
8. Scheduling

Each step keeps the file-based contracts so earlier stages can run and
be tested standalone.