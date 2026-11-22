# Architecture

## 5-stage production pipeline

```
1. Topic Selection   ->  topic_selection.json
2. AI Research       ->  research.json
3. Script Generation ->  script.json
4. Scene Planning    ->  scene_plan.json
5. Video Generation  ->  video.mp4
```

Each stage reads the previous stage's artifact and writes its own. There is no
orchestrator that executes stages; `pipeline/orchestrator.py` is a scaffold
that only creates a run directory and its `state.json`.

## Artifact layout (flat)

One artifact per stage, written directly into `data/<run_id>/`:

| Stage | Artifact |
| --- | --- |
| Topic Selection | `data/<run_id>/topic_selection.json` |
| AI Research | `data/<run_id>/research.json` |
| Script Generation | `data/<run_id>/script.json` |
| Scene Planning | `data/<run_id>/scene_plan.json` |
| Video Generation | `data/<run_id>/video.mp4` |
| Run metadata | `data/<run_id>/state.json` |

The previous nested layout (`research/`, `scripts/`, `production/`, `videos/`)
no longer applies.

## Run state

`pipeline/state.py` models exactly the five stages above, in order:

```
Stage.TOPIC_SELECTION -> Stage.AI_RESEARCH -> Stage.SCRIPT_GENERATION
                      -> Stage.SCENE_PLANNING -> Stage.VIDEO_GENERATION
```

`PipelineState` records `run_id`, `status`, `current_stage`, `created_at`,
`updated_at`, and a per-stage list of artifact paths. It carries no state for
fact checking, human approval, script QA or video QA, because those stages no
longer exist.

## Research basis

**Gemini's own knowledge is the research basis.** There is no web retrieval
and no external evidence in this architecture.

Intentionally removed:

- Fact checking and corroboration
- Evidence / source validation and credibility tiering
- SearXNG
- BeautifulSoup / httpx direct HTTP retrieval
- Firecrawl
- Hybrid retrieval and query planning
- Retrieval traces
- Script QA
- Video QA
- Human approval

## Configuration

- `config/channel.py` - `ChannelConfig` for the `finance_mascot` channel
  (language, promise, tone, format, target duration, mascot asset path).
- `config/topic_selection.yaml` - Stage 1 scoring weights, pillars and
  selection parameters.
- `providers/gemini/` - `GeminiClient`, `GeminiConfig` and the Stage 1
  `GeminiTopicProvider`.

## Current implementation status

| Stage | Status |
| --- | --- |
| Topic Selection | Implemented (mock default, Gemini opt-in) |
| AI Research | Implemented (mock default, Gemini opt-in; model knowledge only, no web retrieval) |
| Script Generation | Implemented (mock default, Gemini opt-in) |
| Scene Planning | Implemented (mock default, Gemini opt-in; prompt `short_scene-v2`) |
| Video Generation | Implemented (local renderer only; ffmpeg if present, else OpenCV) |

Each stage CLI exits 2 if its input artifact is missing. Every stage loads and
appends to `state.json` (`PipelineState.load_or_create`), so it accumulates
across the run. `PipelineState.status` is not yet updated by any stage.