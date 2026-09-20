# AI YouTube Channel

Automated pipeline that researches a topic, analyzes SEO, generates a
script, plans a scene-by-scene production, generates video on a GPU
worker, assembles the final YouTube Short, and eventually publishes it —
end to end.

> **Status: scaffold only.** No agents, no API calls, no video generation,
> and no scheduling yet. This project currently defines the structure and
> a minimal entry point.

## Long-term pipeline

```
topic discovery
→ SEO research
→ topic selection / scoring
→ research
→ script generation
→ scene planning
→ production JSON
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
agents/            Future agent modules (research, seo, script, scene_planner, qa)
pipeline/          Orchestrator and per-run state model
video/             Future video build steps (generation, editing, subtitles, audio)
data/              Working artifacts per run (research, scripts, production, videos)
prompts/           Prompt templates per agent
config/            Configuration (populated later; secrets go in .env)
tests/             Tests (added with the first implemented stage)
```

## Current development stage

Only the initial foundation exists:

- `pipeline/state.py` — minimal serializable run state (stage tracking,
  artifact registry, save/load).
- `pipeline/orchestrator.py` — minimal entry point that creates a fresh
  run directory and a state file, then stops.
- Directory skeleton, `.env.example`, `.gitignore`, and a minimal
  `requirements.txt`.

Implementations to come: each agent stage, the GPU worker contract,
video assembly, quality checks, publishing, and analytics.

## Running the minimal project

Requires Python 3.10+.

```powershell
# optional: create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

# start a new run
python -m pipeline.orchestrator
```

This creates `data/<run_id>/` (with `research/`, `scripts/`, `production/`,
`videos/` subdirectories) and writes ``state.json`` for the run.

## Design documentation

The technical design is documented in `docs/` (design only — no
implementation beyond the scaffold below):

| Doc | Covers |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, pipeline stages, resumability, replaceability boundaries |
| [CONTENT_STRATEGY.md](docs/CONTENT_STRATEGY.md) | Channel identity, pillars, fixed 30–60 s story format |
| [TOPIC_SELECTION.md](docs/TOPIC_SELECTION.md) | External signals, cold start, exploration/exploitation |
| [RESEARCH_PIPELINE.md](docs/RESEARCH_PIPELINE.md) | Evidence-backed claims, source credibility tiers, claim/source/date model |
| [FACT_VERIFICATION.md](docs/FACT_VERIFICATION.md) | Separate fact-check stage: PASS/REVIEW/REJECT gating |
| [POLICY_AND_SAFETY.md](docs/POLICY_AND_SAFETY.md) | YouTube policy checks, human approval gate |
| [VIDEO_PIPELINE.md](docs/VIDEO_PIPELINE.md) | Shorts target, production.json interface, Colab worker + swaps |
| [DATA_CONTRACTS.md](docs/DATA_CONTRACTS.md) | The 10 versioned JSON contracts and artifact envelope |
| [MODEL_STRATEGY.md](docs/MODEL_STRATEGY.md) | Abstract model interface; coding model vs. production LLMs |
| [ANALYTICS.md](docs/ANALYTICS.md) | Future analytics interface feeding topic selection |

## Roadmap

1. Topic discovery and SEO research
2. Topic selection / scoring
3. Research → script generation → scene planning
4. Production JSON contract + Colab worker hand-off
5. Voice, subtitles, video assembly, quality check
6. YouTube upload and analytics feedback
7. Scheduling

Each step keeps the file-based contracts so earlier stages can run and
be tested standalone.