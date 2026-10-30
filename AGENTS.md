# AGENTS.md

Persistent project instructions for OpenCode in this repository.

This file is the mechanism by which continuity is carried between sessions. It
records **rules that do not change**. Facts about the current state of the work
live in the two documents described below, not here.

---

## 0. Think and speak critically

Always think and speak critically. This applies to every task, every claim, and
every response.

- Do not accept a premise merely because it was stated. If a request, a
  document, or a prior session's conclusion appears wrong, risky, or
  unsupported, say so plainly and explain why.
- Separate what is **verified** from what is **assumed**, **intended**, or
  **planned**. Never let confidence stand in for evidence.
- Challenge your own work before presenting it. Look for the counter-case, the
  failure mode, and the edge condition rather than only confirming your first
  answer.
- Point out tradeoffs, alternatives, and downsides honestly, even when the user
  did not ask. Do not agree just to be agreeable, and do not soften a real
  problem into a non-problem.
- When uncertain or blocked, say so directly instead of guessing and presenting
  the guess as fact. Ask rather than fabricate.
- Report failures, gaps, and limitations as clearly as successes. A skipped
  test is not a pass; an unavailable service is not "0 results".

---

## 1. Project continuity

Before beginning any non-trivial task:

1. Read the **latest** `## Current continuity checkpoint` in
   `docs/PROJECT_PROGRESS.md` (see section 2 for the efficient reading rule).
2. Read the latest task report(s) under `reports/` if any exist.
3. Treat `docs/PROJECT_PROGRESS.md` as the authoritative long-term project state.
4. Treat `reports/*.md` as the execution/handoff record of their own task only.
5. Inspect the **actual** repository state before relying on historical
   documentation.
6. Never assume a documented feature is currently implemented without checking
   the repository when the task depends on it.

Documentation drifts from code. The code is the fact; the documents are the
record of intent. When they disagree, say so rather than silently trusting
either one.

---

## 2. Documentation roles

| File | Purpose |
| --- | --- |
| `docs/PROJECT_PROGRESS.md` | Long-term continuity: milestones, historical checkpoints, verified state, decisions, blockers, future direction. |
| `reports/<title>.md` | Per-task execution/handoff report for one task only. Pressure/space for detailed reports. |

Rules:

- `PROJECT_PROGRESS.md` is **historical and cumulative**.
- Each report under `reports/` is **task-scoped** and represents **only its own task**.
- Do not turn `reports/` into a second project-history document.
- Do not rewrite historical `PROJECT_PROGRESS.md` checkpoints merely to make
  them cleaner.
- New checkpoints are appended, or clearly supersede the appropriate prior
  checkpoint.

### Authoritative sources

When documents disagree, resolve in this order:

1. **Code and tests** — the fact.
2. `pipeline/state.py` and `docs/ARCHITECTURE.md` — the stage model and artifact
   layout.
3. This file (`AGENTS.md`) — standing rules.
4. `docs/PROJECT_PROGRESS.md` — long-term history and intent.
5. `reports/*.md` — one task's record only.

Many documents under `docs/` predate the current five-stage architecture and
still describe removed stages. Treat them as historical unless the code agrees.
Never assume a documented feature is implemented without checking the repository.

### Reading PROJECT_PROGRESS.md efficiently

`docs/PROJECT_PROGRESS.md` is large and cumulative. Unless the task genuinely
needs project history, read **only the last** `## Current continuity
checkpoint` section, plus any section the task explicitly depends on. Do not
read the entire file as a routine starting step.

`opencode_report.md` is **retired**. Do **not** add, edit, or replace anything
in it from now on; treat it as frozen historical output.

---

## 3. Task report requirement

After **every** task or session that performs meaningful work, write a report
covering **only** that task.

- Place it under `reports/`.
- Name the file after the report's title, e.g.
  `reports/colab_handoff_implementation_plan.md`.
- Use a descriptive title; it becomes the file name.
- One report per task. Do not append to or rewrite another task's report.
- Use the `reports/` folder specifically when the report is large or long; it
  exists to give reports room that chat and a single ephemeral file do not.
- Write the report to a Markdown file under `reports/` whenever it contains
  tables (or other structured content), rather than only answering in chat.

Do **not** add, edit, or replace anything in `opencode_report.md`. It is
retired and frozen; ignore it from now on.

The report must include, where applicable:

```
# <Report Title>

## Task
What was requested.

## Repository State
- branch
- HEAD
- working-tree state
- relevant pre-existing modifications
- relevant untracked files

## Environment
Relevant OS, Python, Docker, WSL, models, packages, services details.
Do not expose secrets.

## Work Completed
What actually happened.

## Files Changed
| File | Change |

## Files Not Changed
Components intentionally left untouched.

## Verification
| Check | Result |
Tests, test counts, compile checks, lint/type checks, Docker checks,
network checks, E2E checks, git diff --check, other relevant verification.

## Runtime / Experiment Results
Commands, run IDs, artifact paths, containers, endpoints, result counts,
failures, timings, network behavior.

## Current Status
Classify as COMPLETE / WORKING / BLOCKED / NOT STARTED / NOT TESTED / PLANNED.

## Problems / Blockers
Separate: code defects, environment problems, external service problems,
quota/rate-limit problems, configuration problems, design limitations.

## Important Decisions
Decisions future sessions need to know.

## External Operations
Which external operations were performed and which were deliberately not
performed (network calls, Drive/rclone, Colab, GPU, model downloads,
generation services). State "none" explicitly when none were used.

## Commit / Push
- Files staged and committed, or why not
- Commit hash and branch
- Pushed to which remote branch, or why not
- Any divergence between working-tree test results and committed-snapshot
  verification

## Next Recommended Action
The next logical action.

## Constraints / Do Not Do
Task-specific constraints future sessions must preserve.
```

Do not execute unrelated work merely to make the report complete.

### Report safety

**NEVER** place any of the following in any report under `reports/`:

- API keys
- passwords
- access tokens
- bearer tokens
- credentials
- secret environment variable values
- private authentication material

It is acceptable to state `GEMINI_API_KEY: configured`, but **never** print its
value. Apply the same rule to any other credential: describe its presence, never
its value.

If a secret must be discussed, describe its presence without exposing the value.

---

## 4. Verification honesty

**Never claim something is working unless it was actually verified.**

Use these labels explicitly and distinguish them:

- `VERIFIED`
- `NOT VERIFIED`
- `NOT TESTED`
- `BLOCKED`
- `PLANNED`
- `NOT CHANGED`

Do not hide failed attempts. Do not replace failed results with intended
results. Do not fabricate network responses, search results, API results,
artifacts, test results, or performance numbers.

A skipped test is not a passing test. An unavailable service reported as
"unavailable" is correct; the same service reported as "0 results" is a lie.

---

## 5. Project architecture

This project is an AI YouTube **Shorts** content-generation pipeline.

### Current stage model

The pipeline has **five** stages. `pipeline/state.py` is the authoritative
definition of the stage enum and its order; `docs/ARCHITECTURE.md` is the
authoritative description of the artifact layout.

| # | Stage | `Stage` enum value | Artifact under `data/<run_id>/` |
| --- | --- | --- | --- |
| 1 | Topic Selection | `topic_selection` | `topic_selection.json` |
| 2 | AI Research | `ai_research` | `research.json` |
| 3 | Script Generation | `script_generation` | `script.json` |
| 4 | Scene Planning | `scene_planning` | `scene_plan.json` |
| 5 | Video Generation | `video_generation` | `video.mp4` |

Run metadata is written to `data/<run_id>/state.json`. The artifact layout is
**flat**: one file per stage directly inside the run directory.

Stage 1 is implemented under `agents/seo/`, not `agents/topic_selection/`.
There is no Script QA, Video QA, Human Approval, or fact-checking stage or
package. See section 6.

### Stage entry points

Each stage is a module entry point run with `python -m`. There is **no**
per-stage `__main__.py`; the only `__main__.py` under `agents/` is
`agents/image_jobs/__main__.py`, which belongs to the Colab image-handoff
tooling, not to the pipeline.

| # | Entry point | Default `--provider` |
| --- | --- | --- |
| 1 | `python -m agents.seo.topic_selector` | `mock` |
| 2 | `python -m agents.research.ai_research_cli` | `mock` |
| 3 | `python -m agents.script.script_cli` | `mock` |
| 4 | `python -m agents.scene_planner.scene_cli` | `mock` |
| 5 | `python -m agents.video_generation.video_cli` | `local` |

All five accept `--run-id`. Stages 1–4 also accept `--config` and `--out`;
Stage 5 accepts `--scene-plan`, `--encoder`, `--width`, `--height` and `--fps`.

Provider support is **not** uniform across stages:

- Stages 1–4 accept `--provider mock` (the default) or `--provider gemini`.
- Stage 5 currently supports **only** `--provider local` (the default).
  `--provider gemini` is not supported by Stage 5 and is rejected at argument
  parsing. Gemini support in Stages 1–4 does not imply Gemini support in
  Stage 5.

Re-verify flags and provider choices against the source before relying on them.

### Providers

The only provider package is `providers/gemini/`: `client.py`, `config.py`,
`errors.py`, `structured.py`, and the per-stage `*_provider.py` modules. Those
modules serve Stages 1–4; Stage 5 has no Gemini provider. There is no Firecrawl,
retrieval, or web-search provider.

Stage 5's provider registry is `agents/video_generation/provider.py`, and it
currently registers a single entry: `local` (`LocalMotionRenderer`). Documented
provider lists must reflect actual runtime capabilities — `available_providers()`
and the CLI `--provider` choices — not an intended future set.

### Configuration

| Path | Contents |
| --- | --- |
| `config/channel.py` | `ChannelConfig` for the `finance_mascot` channel |
| `config/topic_selection.yaml` | Stage 1 scoring weights, pillars, selection parameters |
| `prompts/` | Versioned prompt templates, including `prompts/scene_planner/` |

### Current direction

1. The pipeline is **not** scheduled and must **not** be started automatically.
2. Get the five stages working end-to-end with real data.
3. Use the Stage 4 `ScenePlan` as the input for Stage 5 video work.
4. Only once the five-stage pipeline works end-to-end, consider larger
   architectural improvements.

Do not introduce scheduled pipeline execution unless explicitly requested.

### Documentation drift warning

`docs/ARCHITECTURE.md` is authoritative for stage structure and artifact
layout, but its "Current implementation status" table is stale and still
describes the original skeletons — verify implementation status in code.

Several older documents (`docs/DATA_CONTRACTS.md`, `docs/POLICY_AND_SAFETY.md`,
`docs/AUDIT_QA_STAGES.md`, `docs/E2E_REAL_PIPELINE_REPORT.md`) still reference a
former ten-stage design with `agents.qa.*`, `agents.script.script_generator`,
`agents.research.research_agent` and `agents.human_approval`. None of those
modules exist. Treat those references as **historical**.

---

## 6. Removed subsystems — do not reintroduce

Fact checking, evidence/corroboration, source validation and credibility
tiering, retrieval (SearXNG, BeautifulSoup/httpx, Firecrawl), Script QA,
Video QA, policy gates and human approval were **intentionally removed** from
this architecture. Gemini's own knowledge is the research basis: there is no web
retrieval and no external evidence in the pipeline.

These absences are enforced by the current test suite:

| Test | Enforces |
| --- | --- |
| `tests/test_video_render_spec.py::test_no_fact_check_evidence_or_qa_concepts_in_stage5` | Stage 5 sources contain no evidence, corroboration, fact-check, citation, verdict, PASS/FAIL, QA, approval, source, retrieval or web-search concepts |
| `tests/test_finance_script_generation.py::TestNoQaContract` | `agents.script` exposes no validation verdict or score; no `validator.py`, `qa.py`, `quality.py`, `fact_checker.py` |
| `tests/test_finance_ai_research.py::TestNoSourceMachinery` | Research contract has no source/evidence/citation/url fields and imports no retrieval modules |
| `tests/test_finance_topic_selection.py::test_prompt_has_no_fact_checking_stage_requirements` | The Stage 1 prompt states there is no fact-verification stage |

Rules:

- Do **not** reintroduce these subsystems without a separately approved scope
  change.
- Do **not** delete, skip, or weaken the tests above to make documentation or an
  implementation appear consistent.
- Do **not** cite files or settings that do not exist — for example
  `agents/qa/fact_checker/gates.py` or a `corroboration_min_sources` value.
  Neither exists in this repository.
- Do **not** defend behavior from the removed design. If a stale document says
  otherwise, the code and tests win.
- If asked to add fact checking, evidence, retrieval, QA, or human approval,
  treat it as an architecture change: stop, state that it conflicts with the
  current design and tests, and ask for explicit approval before writing code.

### Historical note

Earlier versions of this project used a ten-stage design with fact checking,
Script QA, Video QA, policy gates and human approval. An **isolated experiment**
under `experiments/retrieval/` (SearXNG → BeautifulSoup / self-hosted Firecrawl)
was completed and verified at that time.

That history is retained for context only. It is **not** the current
architecture, the experiment is not imported by production code, and no
retrieval work should be resumed without an approved scope change.

---

## 7. LLM provider direction

Current default Gemini model: `gemini-3.5-flash-lite` (`DEFAULT_MODEL` in
`providers/gemini/config.py`), overridable with the `GEMINI_MODEL` environment
variable. This is configuration-driven. **Do not change the model unless
explicitly requested.**

Real Gemini calls require a configured Gemini API key. Never print a key, token
or credential value; state only that it is configured.

Per-stage provider defaults are `mock` for Stages 1–4 and `local` for Stage 5.
Real model calls are opt-in with `--provider gemini` in Stages 1–4 only; Stage 5
has no Gemini provider and accepts only `--provider local`. Default runs must
stay offline and deterministic.

Future direction: Ollama + Qwen may be used as a fallback when Gemini API
quota or rate limits are encountered.

That fallback is **PLANNED, NOT IMPLEMENTED**. Do not implement it unless
explicitly requested, and do not redesign the LLM provider architecture merely
to accommodate it.

---

## 8. Orchestration

Do **NOT** introduce a separate orchestration layer yet. The current priority is
getting the existing five-stage pipeline working correctly.

`pipeline/orchestrator.py` is an acknowledged **scaffold**. It accepts
`--workdir` and `--run-id`, creates `data/<run_id>/`, writes `state.json`,
prints the expected artifact paths, and exits. It executes **no** stages and
calls no external services. Do not treat it as a working orchestrator and do not
describe it as one.

Stages are run individually via the CLIs listed in section 5. An orchestration
layer may be considered later, once the pipeline works end-to-end.

---

## 9. Git workflow

This is the single authoritative Git policy. It supersedes any older blanket
instruction elsewhere in this file or in task reports.

### Before starting work

1. Inspect the current branch, HEAD, configured remotes, staged changes,
   unstaged changes, and untracked files (`git status --short --branch`,
   `git log -1 --oneline`, `git remote -v`).
2. Preserve every pre-existing modification and unrelated untracked file. The
   working tree of this repository is dirty **on purpose**.
3. Work on the appropriate feature branch. Do **not** switch branches if doing
   so risks losing or mixing existing work.
4. Complete one clearly scoped task at a time.

### Committing and pushing a completed task

1. Run the relevant tests and verification **before** committing.
2. Review the diff (`git diff`, `git diff --cached`) and stage **only** the
   files that belong to the completed task. Never use `git add .` or
   `git add -A`.
3. Commit each completed and verified task to the feature branch.
4. Push that commit to the corresponding remote feature branch.
5. Report precisely what was tested, committed and pushed, and distinguish
   working-tree test results from committed-snapshot verification.

### Restrictions

- Never merge into `main`, push directly to `main`, or deploy without explicit
  user authorization.
- Never force-push, rewrite history, or run destructive cleanup commands
  (`git reset --hard`, `git clean`, `git checkout -- .`).
- Never revert, stage or commit unrelated work.
- Never add, edit or replace `opencode_report.md`.
- If the branch, the remote, the ownership of existing changes, or the safe file
  scope is unclear, **stop and report the blocker** rather than guessing.

### Versioned and gitignored files

`AGENTS.md` is **tracked** and versioned in Git, so a fresh clone receives the
project instructions. `docs/PROJECT_PROGRESS.md` and `reports/` are listed in
`.gitignore` and are local-only: they are **not** included in a task commit
unless the ignore policy is changed by an explicit, separately approved
decision.

- Do not re-add `AGENTS.md` to `.gitignore`.
- Do not change `.gitignore` as part of routine task work; the current ignore
  policy is deliberate.
- Do not change the ignore policy for `docs/PROJECT_PROGRESS.md` or `reports/`
  without an explicit, separately approved decision.
- Do not force-add an ignored path (`docs/PROJECT_PROGRESS.md`, `reports/`)
  unless the user explicitly authorizes it.
- If a required deliverable is ignored, complete the work, report the
  limitation, and ask before changing the policy.

### Historical Git blockers

Earlier checkpoints recorded that no remote was configured, which blocked
commit and push. A remote (`origin`) is now configured. Re-verify remote and
branch state at the start of every task rather than trusting that note.

---

## 10. Data / artifact safety

- Preserve existing run directories and artifacts under `data/`.
- **Do not reuse previous run IDs.** Use a new run ID for every new run, and
  never pass a `--run-id` that already exists under `data/`.
- Do not overwrite historical artifacts.
- Do not manually alter generated artifacts to make validation pass.
- `data/` also holds **legacy** directories from the retired ten-stage layout
  (`research/`, `scripts/`, `production/`, `videos/`, `*-demo`, `e2e-*`). They
  are historical evidence, not the current layout. Do not delete, rewrite or
  "migrate" them unless explicitly requested.

---

## 11. Experimental work

Experimental code belongs under an appropriate experiment directory unless it is
explicitly promoted into production.

Do not automatically convert an experiment into production code.

Always distinguish clearly between an **experiment**, a **prototype**, and a
**production implementation**.

`experiments/retrieval/` is an existing experiment of this kind. Production code
must not import it, and its presence is not a mandate to resume retrieval work
(see section 6).

---

## 12. End-of-task behavior

At the end of every meaningful task:

1. Verify the work.
2. Update `docs/PROJECT_PROGRESS.md` with a concise, verified continuity
   checkpoint when the task materially changes project state. Append; do not
   rewrite historical checkpoints.
3. Write exactly one task report at `reports/<title>.md`, following section 3.
   Never modify `opencode_report.md`.
4. Report exactly what changed.
5. Report exactly what did **not** change.
6. Report the tests and verification actually performed, including failures and
   skips.
7. Report blockers.
8. Report the next logical action.
9. Commit and push per section 9, or report why that was not possible.

Do **not** merge into `main`. Do **not** claim a check passed unless it was
actually run.

---

## 13. Repository quick reference

Observed on branch `feat/colab-handoff-integration`. Re-verify rather than trust.

| Path | Contents |
| --- | --- |
| `agents/seo/` | Stage 1 — topic selection (`topic_selector.py`, `config.py`, `scorer.py`, `selector.py`, `history.py`, `competition.py`). |
| `agents/research/` | Stage 2 — AI research (`ai_research_cli.py`, `ai_research.py`, `ai_research_models.py`, `provider.py`). |
| `agents/script/` | Stage 3 — script generation (`script_cli.py`, `stage.py`, `script_models.py`, `provider.py`). |
| `agents/scene_planner/` | Stage 4 — scene planning (`scene_cli.py`, `stage.py`, `scene_models.py`, `validator.py`, `provider.py`). |
| `agents/video_generation/` | Stage 5 — video generation (`video_cli.py`, `stage.py`, `renderer.py`, `render_spec.py`, `encoder.py`, `assets.py`, `fonts.py`, `motion.py`). |
| `agents/image_jobs/` | Colab image-handoff tooling (`__main__.py`, `handoff.py`, `import_assets.py`, `converter.py`, `jobs.py`). Not a pipeline stage. |
| `providers/gemini/` | Gemini client, config, errors, structured-output helper and per-stage providers. |
| `pipeline/` | `state.py` (authoritative five-stage model) and the scaffold `orchestrator.py`. |
| `prompts/` | Versioned prompt templates, including `prompts/scene_planner/`. |
| `config/` | `channel.py` and `topic_selection.yaml`. |
| `docs/` | Design docs and contracts; `ARCHITECTURE.md` and `PROJECT_PROGRESS.md`. Some docs are historical. |
| `experiments/` | Isolated experiments, including `experiments/retrieval/` (not production). |
| `tools/`, `notebooks/` | Colab handoff CLI and notebook generator. |
| `tests/` | `pytest` suite. Run with `python -m pytest -q`. |
| `reports/` | Per-task reports, one file per task, named after the report title. Gitignored. |
| `data/` | Per-run artifacts, mostly gitignored; contains legacy layout directories. |

Stage entry points are `python -m` module CLIs — see the table in section 5.
There is no per-stage `__main__.py`.
