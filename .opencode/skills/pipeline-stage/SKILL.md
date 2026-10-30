---
name: pipeline-stage
description: Use when running, debugging or documenting an individual pipeline stage. Knows the five stages, the verified CLI entry points, artifacts, implementation status, and that the orchestrator is only a scaffold.
---

# Pipeline Stage

## Authoritative references

Check these against live code before relying on them:

- `pipeline/state.py` — the `Stage` enum and its order
- `docs/ARCHITECTURE.md` — artifact layout and removed subsystems
- `AGENTS.md` §5 — stage table, entry points, providers; §6 — removals

Caveats:

- `docs/ARCHITECTURE.md`'s "Current implementation status" table is stale and
  still describes the original skeletons. Verify status in code.
- Other docs under `docs/` describe a retired ten-stage design with
  `agents.qa.*`, `agents.script.script_generator`, `agents.research.research_agent`
  and `agents.human_approval`. Those modules do not exist. Ignore them for stage
  mechanics.

## The five stages

| # | Stage | Entry point | Artifact in `data/<run_id>/` |
| --- | --- | --- | --- |
| 1 | Topic Selection | `python -m agents.seo.topic_selector` | `topic_selection.json` |
| 2 | AI Research | `python -m agents.research.ai_research_cli` | `research.json` |
| 3 | Script Generation | `python -m agents.script.script_cli` | `script.json` |
| 4 | Scene Planning | `python -m agents.scene_planner.scene_cli` | `scene_plan.json` |
| 5 | Video Generation | `python -m agents.video_generation.video_cli` | `video.mp4` |

Stage 1 lives in `agents/seo/`; there is no `agents/topic_selection/` package.

There is **no** per-stage `__main__.py`. The only `__main__.py` under `agents/`
is `agents/image_jobs/__main__.py`, which is Colab handoff tooling, not a stage.

Common flags: all five accept `--run-id`; Stages 1–4 also accept `--config` and
`--out`; Stage 5 accepts `--scene-plan`, `--encoder`, `--width`, `--height` and
`--fps`. Re-verify with `--help` before scripting a run.

## Implementation status

Verify in code; this reflects the repository when the skill was written.

| Area | Status |
| --- | --- |
| Stage 1 Topic Selection | Implemented; `mock` and `gemini` providers |
| Stage 2 AI Research | Implemented; `mock` and `gemini` providers |
| Stage 3 Script Generation | Implemented; `mock` and `gemini` providers |
| Stage 4 Scene Planning | Implemented; `mock` and `gemini` providers |
| Stage 5 Video Generation | Implemented with the local motion renderer; `local` is the only provider |
| AI video renderer | NOT IMPLEMENTED — explicit future seam in `agents/video_generation/provider.py` |
| Orchestrator | Scaffold only; creates the run directory and `state.json` |
| Ollama / Qwen fallback | PLANNED, NOT IMPLEMENTED |
| Fact checking, retrieval, QA, human approval | Removed by design |

Stage 5 has **no** `gemini` provider. `--provider gemini` applies to Stages 1–4
only.

## Rules

- Run stages individually via their CLIs. Do not add an orchestration layer
  (`AGENTS.md` §8).
- Do not invent stages, providers or artifacts. This skill documents the
  pipeline; it does not extend it.
- Use a fresh run ID and preserve existing `data/` artifacts (`AGENTS.md` §10).
- Do not add retrieval, evidence, QA or approval behavior (`AGENTS.md` §6).
- Defaults are offline and deterministic. Real model calls are opt-in and must
  be authorized by the task.
- If a documented path or flag does not exist, report the discrepancy instead of
  guessing.
