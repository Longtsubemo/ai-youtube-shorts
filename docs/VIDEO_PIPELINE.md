# Video Pipeline

From scene plan to the finished, QA'd Short. Design only; no video
generation, no Colab integration, no uploads are implemented.

## 1. Target output

- YouTube Shorts
- **30–60 seconds**
- **9:16 vertical**
- consistent story format (`CONTENT_STRATEGY.md`)

## 2. Pipeline (content → video)

```
scene plan
→ production.json            ← stable interface (content ↔ video)
→ video generation           [GPU worker: Google Colab]
→ audio
→ subtitles
→ assembly
→ video QA
→ policy / publish QA
→ human approval
→ YouTube
```

The content system (topic/SEO/research/fact/script/scene) and the video
system (generation/audio/subtitles/assembly) communicate **only through
`production.json`**.

## 3. `production.json` as the stable interface

`ProductionPlan` (`DATA_CONTRACTS.md`) is a complete, self-contained
specification of the video to build:

- format specs (9:16, duration target, fps),
- ordered scenes with timing, per-scene visuals + narration + subtitle
  text,
- audio requirements (voice style, pacing),
- subtitle/style settings,
- output expectations (file name, container).

Properties that make it the stable boundary:

- **No provider knowledge:** it describes *what* to generate, not *how*
  or *by whom*. No model names, no provider URLs, no API keys inside.
- **Self-contained:** the worker needs `production.json` + provided
  assets and nothing else about the channel.
- **Versioned:** `schema_version` bump = coordinated worker update
  (`DATA_CONTRACTS.md`).

Replacing Colab with a local GPU or another video-generation system
means writing a different worker behind the same contract — the content
system never notices.

## 4. Worker abstraction

Minimal interface (design sketch):

```python
# video/generation/interface.py (future)
def submit(plan: ProductionPlan, input_artifacts: Path) -> JobResult: ...
def poll(job_id: str) -> JobState: ...
def fetch_artifacts(job_id: str, destination: Path) -> Path: ...
```

A `JobResult`/`JobState` envelope (small, versioned) carries status
(`submitted / running / succeeded / failed`), job ID, output file paths,
and logs for debugging. This envelope is defined as part of the worker
contract, alongside `production.json`, but is deliberately minimal.

### Google Colab adapter (initial worker)

- Uploads `production.json` + input assets to the Colab runtime; runs
  generation; downloads produced clips.
- The exact transport (Drive, direct transfer, tunnel, etc.) is an
  **open decision** (`ARCHITECTURE.md` §9).
- Colab limits (session lifetime, GPU hours) are handled by the adapter,
  not by the content system.

## 5. Video-asset stages

| Stage | Role | Notes |
|---|---|---|
| generation | produces raw clips per scene | driven entirely by `production.json` |
| audio | TTS voiceover track (and silence/breaks if generated separately) | voice is described semantically ("calm female, neutral accent") — provider-agnostic |
| subtitles | timed captions from script lines/scene text | styled per plan, must sync with narration |
| assembly | combine clips + audio + captions into the final Short | cuts to scene timing, concat order, vertical frame |

Stages run on the local PC after generation, consuming the same
`production.json` for scene timing (never re-generating content).

## 6. Video QA

`QAResult` with stage = `video`. Checks:

- duration within 30–60 s; 9:16 vertical frame confirmed
- audio present, not clipped; voice intelligible
- subtitles present and synchronized; no text cut off
- no black frames / missing scene at seams
- per-scene visual matches plan description
- file size/format valid for upload

Visual/audio issues are technical, unlike policy issues
(`POLICY_AND_SAFETY.md` §6).

## 7. Replaceability summary

| Swap | Only change needed |
|---|---|
| GPU worker: Colab → other cloud/local/different system | new adapter behind `submit`/`poll`/`fetch` |
| TTS voice provider | new audio stage implementation |
| Subtitle renderer | new subtitles stage implementation |
| Assembly tool | new editing stage implementation |

None of these changes touch the content system or the contracts.

## 8. Open questions

- Worker transport mechanism for the Colab adapter (§4).
- Whether audio is generated from the scene text locally or expected
  as part of worker output; either way `production.json` includes the
  narration text so both are possible.
- Whether subtitles are burned into the frame or sidecar/hard
  bands (visual style decision; influences assembly + QA checks).