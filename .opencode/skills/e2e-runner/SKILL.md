---
name: e2e-runner
description: Use when running, resuming or auditing an end-to-end pipeline run. Enforces fresh run IDs, preservation of existing data/ artifacts, orchestrator-scaffold awareness, honest verification, and external-operation restrictions.
---

# E2E Runner

## Current architecture

Five stages, in order. `pipeline/state.py` and `docs/ARCHITECTURE.md` are the
authoritative references; check them against live code first.

| # | Stage | Artifact in `data/<run_id>/` |
| --- | --- | --- |
| 1 | Topic Selection | `topic_selection.json` |
| 2 | AI Research | `research.json` |
| 3 | Script Generation | `script.json` |
| 4 | Scene Planning | `scene_plan.json` |
| 5 | Video Generation | `video.mp4` |

Each stage reads the previous stage's artifact from the same run directory and
writes its own. Run metadata lives in `data/<run_id>/state.json`.

## Orchestrator reality

`pipeline/orchestrator.py` is a **scaffold**. It accepts `--workdir` and
`--run-id`, creates `data/<run_id>/`, writes `state.json`, prints the expected
artifact paths, and exits. It executes **no** stages and calls no services.

An end-to-end run is the five stage CLIs run in order — see `pipeline-stage`.
Never describe the orchestrator as executing the pipeline.

## Run IDs and artifacts

- Use a **new, unique run ID** for every run.
- Before choosing one, list `data/` and never pass a `--run-id` that already
  exists there. Reusing an ID risks overwriting historical artifacts.
- Preserve every existing run directory. Do not delete, rewrite, renumber or
  "migrate" them.
- `data/` also holds legacy directories from the retired ten-stage design
  (`research/`, `scripts/`, `production/`, `videos/`, `*-demo`, `e2e-*`). They
  are historical evidence, not the current layout; leave them alone.
- Never hand-edit a generated artifact to make validation pass.

## Verification

- Record the exact command per stage and the run ID in the task report.
- Report real counts, failures and timings. A stage that produced nothing is not
  a pass.
- There is no evidence-based `REVIEW` outcome in this architecture; do not
  report one. Report what the stages and their validation actually return.
- Distinguish working-tree verification from committed-snapshot verification.
- Use `VERIFIED` / `NOT VERIFIED` / `NOT TESTED` / `BLOCKED` labels per
  `AGENTS.md` §4.

## Hard restrictions

Unless the task explicitly authorizes them, do **not**:

- call external generation services or real model APIs
- run Colab, rclone, or Google Drive operations
- use a GPU, or download model weights
- modify `data/` artifacts, tests, gates or schemas to force a pass
- reuse or overwrite a historical run

Default providers are `mock` for Stages 1–4 and `local` for Stage 5, so a
default run stays offline and deterministic. Real model calls require
`--provider gemini` (Stages 1–4 only; Stage 5 has no Gemini provider) and a
configured API key — never print the key.

## Reporting

Finish with `task-report`: run ID, per-stage commands, artifact paths, result
counts, failures, skips and timings.
