# Colab SDXL Handoff Contract

Contract for the **manual** Google Drive handoff between the local pipeline and
a free-GPU Colab SDXL-Lightning worker.

This document records the approved contract. It is deliberately explicit about
what is implemented today versus what is planned, so no later session mistakes
a plan for code.

## Status

| Area | Status |
| --- | --- |
| Contract schemas, hashing, markers, ZIP integrity, GPU preflight evaluator (`agents/image_jobs/handoff.py`) | `VERIFIED` (Milestone 1, hardened to schema 1.1) |
| This document | `VERIFIED` (Milestone 1.1) |
| Local handoff CLI (`tools/colab_handoff.py`) | `IMPLEMENTED` (Milestone 2, offline local backend) |
| Offline transport scaffold (`tools/handoff_transport.py`, `--remote local:`) | `IMPLEMENTED` (Milestone 3B.1, offline directory transport) |
| rclone transport + Drive preflight (`tools/handoff_transport.py`, `--remote rclone:<remote>:`) | `IMPLEMENTED` + offline `VERIFIED` (Milestone 3B.2); real-Drive transfer `NOT TESTED` |
| Automation notebook (`notebooks/SDXL_Lightning_handoff.ipynb`) | `IMPLEMENTED` + offline `VERIFIED` (Milestone 3B.3); real Colab/GPU run `NOT TESTED` |
| rclone / Google Drive integration | `IMPLEMENTED` offline; real-Drive end-to-end `NOT TESTED` |
| Asset import reuse (`agents/image_jobs/import_assets.py`) | `NOT CHANGED` |
| Stage 5 execution | `NOT STARTED` |

The helpers in `agents/image_jobs/handoff.py` are **pure standard library**:
no network, no third-party SDK, no model, no GPU. They operate on plain
filesystem paths, so the same code works against a real mounted/synced Drive
folder or a local directory that simulates Drive in tests.

## Non-goals

- No scheduled or automatically-started execution.
- No automatic retry, no retry loop, no scheduler.
- No CPU fallback and no paid GPU. Free GPU only.
- No change to the original notebook or its local Downloads copy.
- No automatic start of Stage 5.

## Approved decisions

| ID | Decision |
| --- | --- |
| D1 | **4 inference steps** with `sdxl_lightning_4step_unet.safetensors` (not 8). The original notebook's `STEPS=8` is a stale post-execution edit; the executed run used 4. Original notebook and Downloads copy stay unchanged. |
| D2 | The new automation notebook is tracked in Git later via a **targeted `.gitignore` exception**. |
| D3 | Local Google Drive access uses **rclone** (later milestone). |
| D4 | GPU preflight requires **total VRAM >= 14 GiB** and **free VRAM >= 10 GiB**; diagnostics are recorded and the run stops if the threshold is not met. |
| D5 | Assets import into `data/<run_id>/assets/`. **Never overwrite by default.** Any Stage 5 work uses a new run ID. |
| D6 | Pure-stdlib handoff helpers live in `agents/image_jobs/handoff.py`. |
| D7 | Colab and rclone are assumed to use the **same Google account**; access is verified when the Drive-integration milestone begins. |

### Drive root behavior

After Drive integration is authorized: create `ai-youtube-channel-handoff/` if
it does not exist, reuse it if it exists, and **stop with a clear error** if
duplicate or ambiguous roots are found. Never scatter files across Drive and
never delete old run artifacts.

This is implemented for rclone in Milestone 3B.2 (`preflight_remote` +
`discover_root`). The discovered root is trusted only when the remote sentinel
`handoff_sentinel.json` also verifies against an **explicitly configured**
expected value supplied by the operator (see *rclone transport* below); the value
is never read from the remote file alone.

## Generation configuration

| Field | Value |
| --- | --- |
| Base model | `stabilityai/stable-diffusion-xl-base-1.0` |
| Adapter | `ByteDance/SDXL-Lightning` |
| Checkpoint | `sdxl_lightning_4step_unet.safetensors` |
| Steps | `4` |
| Guidance scale | `0.0` |
| dtype | `float16` |
| Resolution | `768 x 1344` |
| Scheduler | `EulerDiscreteScheduler(trailing)` |
| GPU | Colab T4 |

## Folder structure

```
ai-youtube-channel-handoff/
  <run_id>/
    input/
      image_jobs.json          # the local image-jobs/1.0 contract
      input_manifest.json      # sdxl-input-manifest/1.1
      READY                    # marker; content = attempt_id; written LAST
    status/
      status.json              # sdxl-status/1.1 (overwritten by the worker)
    output/
      status.json              # final sdxl-status/1.1
      output_manifest.json     # sdxl-output-manifest/1.1
      <run_id>_generated_assets.zip   # flat: <asset_id>.png + result.json
      DONE                     # marker; content = attempt_id; written LAST
```

`READY` and `DONE` are **content-bearing** markers: their content is the
`attempt_id` of the handoff attempt they belong to. Their existence is still the
signal, but the content binds the marker to one specific attempt, so a marker
left over from an earlier attempt can never be mistaken for the current one.
They are always written **after** every document and payload they refer to.

## Marker-last protocol

The protocol guarantees a reader never sees a half-written handoff.

1. The local side mints an `attempt_id` (`new_attempt_id(run_id)`), uploads
   `input/image_jobs.json` and `input/input_manifest.json`, then writes
   `input/READY` **last** with the `attempt_id` as its content.
2. The Colab worker treats a run as startable only when `input/READY` exists,
   its content equals `input_manifest.json`'s `attempt_id`, and the input
   manifest validates. It never consumes a partial input.
3. The worker writes `status/status.json` while running (progress/handoff), then
   calls `begin_attempt(output_dir, attempt_id)` to clear any stale
   `DONE`/`output_manifest.json`/`status.json` from an earlier attempt before
   publishing its own.
4. The worker packages `<run_id>_generated_assets.zip`, writes `output/output_manifest.json`
   and `output/status.json` (both carrying the `attempt_id`), then writes
   `output/DONE` **last** with the same `attempt_id` as content.
5. The local side treats an output as fetchable only when `output/DONE` exists,
   its content matches the output manifest's `attempt_id`, `status.json` is
   `PASS` for the same attempt, and `output_manifest.json` validates (including
   the ZIP fingerprint).
6. A `DONE` marker without a valid manifest/status, a marker/attempt mismatch, or
   a manifest whose ZIP hash/size does not match, is an integrity failure (see
   failure codes).

Validated by `verify_ready(...)`, `verify_done(...)` and `verify_handoff_pair(...)`
in `agents/image_jobs/handoff.py`.

### Attempt binding

`attempt_id` (format `<run_id>-<uuid4 hex>`) is the lynchpin of the hardened
protocol. It appears in the input manifest, the output manifest and
`status.json`, and is the content of both markers. Every cross-document check
compares attempt ids, so a retry that reuses `run_id` cannot pick up a previous
attempt's documents or its `DONE` marker. `begin_attempt(...)` removes the stale
completion state without touching artifacts (the ZIP is preserved).

## Schemas

All documents are JSON objects with a `schema` string. Version is part of the
string; a reader rejects any unknown version rather than guessing. The current
version is **1.1**; the retired 1.0 documents (which carried an ambiguous
`input_sha256` and no `attempt_id`) are rejected outright rather than guessed at
— a 1.0 handoff must be re-built.

### `sdxl-input-manifest/1.1`

```json
{
  "schema": "sdxl-input-manifest/1.1",
  "run_id": "<run_id>",
  "attempt_id": "<run_id>-<uuid4 hex>",
  "created_at": "<ISO-8601 UTC>",
  "tool": {"name": "colab_handoff", "version": "0.1.1"},
  "image_jobs": {
    "filename": "image_jobs.json",
    "schema": "image-jobs/1.0",
    "bytes": 12345
  },
  "image_jobs_sha256": "<64 hex, the image_jobs.json hash>",
  "job_count": 7,
  "expected_asset_ids": ["sc01_main"],
  "expected_resolution": {"width": 768, "height": 1344},
  "generation_config": {
    "checkpoint": "sdxl_lightning_4step_unet.safetensors",
    "steps": 4,
    "guidance_scale": 0.0,
    "dtype": "float16",
    "width": 768,
    "height": 1344
  }
}
```

### `sdxl-output-manifest/1.1`

```json
{
  "schema": "sdxl-output-manifest/1.1",
  "run_id": "<run_id>",
  "attempt_id": "<run_id>-<uuid4 hex>",
  "created_at": "<ISO-8601 UTC>",
  "tool": {"name": "colab_handoff", "version": "0.1.1"},
  "image_jobs_sha256": "<the input image_jobs.json sha256>",
  "status": "completed",
  "zip": {
    "filename": "<run_id>_generated_assets.zip",
    "sha256": "<64 hex>",
    "bytes": 204800
  },
  "members": ["sc01_main.png", "result.json"],
  "generated_asset_ids": ["sc01_main"],
  "resolution": {"width": 768, "height": 1344},
  "generation_config": {"checkpoint": "...", "steps": 4, "guidance_scale": 0.0, "dtype": "float16", "width": 768, "height": 1344},
  "result_manifest": {"filename": "result.json", "schema": "image-generation-results/1.0"}
}
```

`status` may only be `"completed"`. `image_jobs_sha256` is the explicit rename
of the retired, ambiguous `input_sha256`: it is the hash of the input
`image_jobs.json`, and it is the value the pair verifier compares between the
input and output manifests.

### `sdxl-status/1.1`

```json
{
  "schema": "sdxl-status/1.1",
  "run_id": "<run_id>",
  "attempt_id": "<run_id>-<uuid4 hex>",
  "created_at": "<ISO-8601 UTC>",
  "updated_at": "<ISO-8601 UTC>",
  "tool": {"name": "colab_handoff", "version": "0.1.1"},
  "image_jobs_sha256": "<the input image_jobs.json sha256>",
  "overall_status": "PENDING | RUNNING | PASS | BLOCKED | FAILED",
  "current_stage": "gpu_preflight",
  "completed_stages": ["gpu_preflight"],
  "completed_asset_ids": ["sc01_main"],
  "failed_assets": [{"asset_id": "sc02_main", "code": "GENERATION_FAILED", "message": "..."}],
  "error": null,
  "environment": {"gpu_name": "Tesla T4", "total_vram_gb": 15.0, "free_vram_gb": 14.8},
  "generation_config": {"...": "..."},
  "resume_instructions": ""
}
```

`overall_status` meaning:

| Value | Meaning |
| --- | --- |
| `PENDING` | input staged, not yet started |
| `RUNNING` | worker is executing |
| `PASS` | all assets generated, validated, packaged, marked `DONE` |
| `BLOCKED` | recoverable infrastructure failure; human retries later |
| `FAILED` | deterministic defect (e.g. output validation); fix required |

`error` is `null` for `PENDING`/`RUNNING`/`PASS`, and a structured object
otherwise (see failure codes). `validate_status(...)` enforces this **and** the
resume invariants:

- `completed_stages` is a prefix of the stage order (no skipped stages);
- `PENDING` has completed nothing and is at `gpu_preflight`;
- `RUNNING` is at the stage after the last completed one;
- `PASS` has completed every required stage, is at the last stage, has no
  failed assets, and `completed_asset_ids` equals the expected asset set
  (so `validate_status` requires `expected_asset_ids` for a `PASS`);
- `BLOCKED`/`FAILED` carry a well-formed error whose `stage` equals
  `current_stage` and whose recoverability matches the code.

### `image-generation-results/1.0`

The `result.json` member inside the output ZIP, written by the worker via
`build_result_document`:

```json
{
  "schema": "image-generation-results/1.0",
  "run_id": "<run_id>",
  "source_json": "image_jobs.json",
  "channel_id": "finance_mascot",
  "title": "<video title>",
  "status": "completed",
  "job_count": 1,
  "generated_count": 1,
  "skipped_count": 0,
  "failed_count": 0,
  "resolution": {"width": 768, "height": 1344},
  "steps": 4,
  "guidance_scale": 0.0,
  "results": [
    {"scene_id": "sc01", "asset_id": "sc01_main", "filename": "sc01_main.png",
     "status": "generated", "seed": 7, "width": 768, "height": 1344}
  ]
}
```

`verify_result_document(...)` enforces the producer contract: the exact
top-level field set above, `source_json == "image_jobs.json"`, `status` in
`completed | partial_failure`, non-negative integer counts consistent with
`results` (`job_count == len(results)`, generated/failed counts equal to the
item statuses, `skipped_count == 0` — the worker never skips, and `status`
agrees with `failed_count`), and per item: all fields present, a safe unique
`asset_id`, `filename == "<asset_id>.png"`, `status` in `generated | failed`,
integer `seed`/size, and a size equal to the document resolution. When the
expected run id, asset set, resolution or generation config are known (they
are: from the verified output manifest), they must match (`hf50`).

The document carries **no attempt id** — the producer does not write one. The
attempt/artifact binding of `result.json` therefore comes from the output
manifest's ZIP hash: the CLI reads the member *from the ZIP the manifest pins*
(never the loose `output/result.json`) and validates it against that manifest.
`verify_zip_result_document(zip_path, output_manifest=..., run_id=...)` is the
single entry point used by `validate` and `import`.

## Cross-binding the input and output

`verify_handoff_pair(input_manifest, output_manifest, ...)` binds an output to
the approved input. It requires: matching `run_id`, matching
`image_jobs_sha256`, matching `attempt_id`, matching resolution, the approved
`generation_config` (4 steps, the 4-step UNet, guidance 0.0, float16), an output
`status` of `completed`, and a `generated_asset_ids` set **identical** to the
input `expected_asset_ids` (no missing, unexpected or duplicate assets). This is
what enforces decision D1 and the 768x1344 resolution at the output gate.

## Failure codes

| Code | Stage | Overall | Meaning |
| --- | --- | --- | --- |
| `NO_GPU` | `gpu_preflight` | `BLOCKED` | No CUDA device / GPU runtime not assigned |
| `CUDA_INIT_FAILED` | `gpu_preflight` | `BLOCKED` | CUDA unavailable or a CUDA op failed |
| `INSUFFICIENT_VRAM` | `gpu_preflight` | `BLOCKED` | Total/free VRAM below threshold, or unknown/non-finite (fail-closed) |
| `RUNTIME_DISCONNECTED` | any | `BLOCKED` | Colab runtime dropped mid-run |
| `ENV_SETUP_FAILED` | `env_setup` | `BLOCKED` | Recoverable dependency/setup failure |
| `INPUT_VALIDATION_FAILED` | `env_setup` | `FAILED` | Deterministic pre-generation input defect (e.g. prompt over token budget); fix required |
| `MODEL_LOAD_FAILED` | `model_load` | `BLOCKED` | Base model / adapter failed to load |
| `GENERATION_FAILED` | `generation` | `BLOCKED` | Inference failure (e.g. transient OOM) |
| `OUTPUT_VALIDATION_FAILED` | `output_validation` | `FAILED` | Output malformed/incomplete; fix required |

Recoverable codes map to `BLOCKED`; `INPUT_VALIDATION_FAILED` and
`OUTPUT_VALIDATION_FAILED` map to `FAILED`. `overall_status_for_error(code)`
implements this and `make_error(...)` records the code, stage, message,
recoverability, timestamp, diagnostics and traceback. Note that
`INPUT_VALIDATION_FAILED` and `ENV_SETUP_FAILED` share the `env_setup` stage but
differ in recoverability: the former is a deterministic input defect
(`FAILED`, do not blind-retry), the latter a recoverable infrastructure fault
(`BLOCKED`).

Manifest/hash defects raised by the integrity helpers (`HandoffError` and
subclasses) are **deterministic** and are treated as `FAILED` by the eventual
CLI, not retried.

### Local integrity codes (informational)

| Code | Meaning |
| --- | --- |
| `hf01`–`hf03` | unreadable / malformed JSON / unsupported schema or missing batch `run_id` |
| `hf04`–`hf08` | run-id mismatch / unsafe asset id / unsafe output path / output path not under `data/<run_id>/assets/` / invalid (unconfined) run id |
| `hf10`–`hf16` | input manifest structural/integrity problems (incl. `hf15` filename mismatch, `hf16` legacy `input_sha256`) |
| `hf17`–`hf19` | output manifest structural / run-id / `image_jobs_sha256` mismatch |
| `hf20`–`hf26` | missing `READY`/`DONE` marker, unreadable manifest/status, or marker/attempt mismatch |
| `hf27`–`hf29` | unsafe ZIP/result filename / (pair) attempt mismatch / resolution mismatch |
| `hf30`–`hf36` | ZIP missing / not a ZIP / hash-size mismatch / corrupt member / unsafe path / member-set mismatch / duplicate normalized member |
| `hf40`–`hf48` | invalid error code/stage, invalid status document, pair generation-config/asset mismatch, invalid output status, invalid `result_manifest`, output status not `PASS` |
| `hf49` | in-ZIP result document structural defect: not a ZIP/missing member, not JSON/UTF-8, not an object, wrong schema, missing/invalid field, unsafe filename or asset id |
| `hf50` | in-ZIP result document consistency mismatch: run id, required-but-non-`completed` status, asset set, resolution, counts, generation config, duplicate/renamed/size-mismatched item |

The pair verifier uses `hf42` (generation-config mismatch), `hf43` (asset
completeness), `hf44` (pair structure/identity) and `hf45` (invalid output
status); `validate_status` uses `hf41`; the `DONE`-requires-`PASS` check uses
`hf48`. The result-document contract (`verify_result_document`,
`read_result_document_from_zip`, `verify_zip_result_document`) uses `hf49` for
"this is not a well-formed `image-generation-results/1.0` document" and `hf50`
for "it is well-formed but disagrees with the verified manifest/run".

## GPU threshold (D4)

`evaluate_gpu_preflight(...)` is a pure function: the notebook gathers the GPU
facts (via `torch` and `nvidia-smi`) and passes them in. It requires
**total VRAM >= 14 GiB** and **free VRAM >= 10 GiB** (configurable thresholds,
defaults `VRAM_TOTAL_MIN_GIB = 14.0`, `VRAM_FREE_MIN_GIB = 10.0`). It is
**fail-closed**: an unknown (`None`) or non-finite (NaN / ±inf) VRAM reading is
treated as `INSUFFICIENT_VRAM`, never as passing, so a caller that fails to read
the GPU cannot slip through. On failure it returns the matching code (`NO_GPU`,
`CUDA_INIT_FAILED` or `INSUFFICIENT_VRAM`) and the full diagnostics dict, which
is recorded in `status.json`.

## Local handoff CLI (Milestone 2, offline backend)

`tools/colab_handoff.py` is the **local** side of the handoff. It is deliberately
**offline**: it operates on a plain local directory that simulates the shared
Google Drive root. It does **not** use rclone, the Google Drive API, OAuth, or
any network. Swapping in a Drive/rclone backend later is a matter of how the
handoff root is materialised; the manifest/marker/ZIP logic in
`agents/image_jobs/handoff.py` is unchanged and remains pure standard library.

The backend is the only component that knows *where* the handoff lives
(`LocalHandoffBackend` in `tools/colab_handoff.py`). The CLI never extracts a ZIP
into `data/`; it stages the verified ZIP outside `data/` and leaves the actual
import to `agents/image_jobs/import_assets.py`.

### Commands

```bash
# 1. Build the input manifest + READY (marker written LAST) in the handoff root.
python tools/colab_handoff.py --handoff-root <root> prepare --run-id <run_id>

# 2. Show current stage, attempt id and completion state.
python tools/colab_handoff.py --handoff-root <root> status --run-id <run_id>

# 3. Verify completion (DONE + status PASS + output manifest + ZIP) and stage the
#    verified ZIP locally (default: <project-root>/handoff_staging/<run_id>/).
python tools/colab_handoff.py --handoff-root <root> fetch --run-id <run_id>

# 4. Verify the completed attempt, then run the existing importer in DRY-RUN
#    mode against that attempt's verified ZIP; write nothing.
python tools/colab_handoff.py --handoff-root <root> validate --run-id <run_id>

# 5. Same verification, then import the assets into data/<run_id>/assets/
#    (explicitly authorized).
python tools/colab_handoff.py --handoff-root <root> import --run-id <run_id> --authorize-import
```

`--project-root` (default `.`) is where `data/<run_id>/image_jobs.json` lives.
`--handoff-root` (default `ai-youtube-channel-handoff`) is the local directory
that simulates Drive and contains `<root>/<run_id>/{input,status,output}/`.
Both options may be given before or after the subcommand.

Useful options:

| Option | Command | Meaning |
| --- | --- | --- |
| `--remote` | all | select a transport: `local:<dir>` (default offline) or `rclone:<remote>:`; takes precedence over `--handoff-root` |
| `--sentinel-value` | all | required with `--remote rclone:<remote>:`; the operator-configured expected sentinel value used to trust the Drive root |
| `--create-root` | `prepare` | allow preflight to create the handoff root (and sentinel) once if absent |
| `--force` | `prepare` | re-prepare an already-prepared run as a **new** attempt (clears a stale output completion via `begin_attempt`) |
| `--staging-dir` | `fetch`/`validate`/`import` | override the staging directory |
| `--zip` | `validate`/`import` | use an explicit ZIP instead of the staged/output one; it must still be the manifest-verified artifact or the command fails (`hf32`) |
| `--authorize-import` | `import` | required; without it `import` refuses (exit 2) and writes nothing |
| `--overwrite` | `import` | replace existing asset files; **off by default** |

### Root folder behavior (offline)

- If the given root exists it is **reused**; otherwise it is created once.
- A run folder `"<root>/<run_id>/"` is created under it; existing runs are never
  deleted or renamed, and no duplicate root is created.
- `prepare` copies (never moves) `data/<run_id>/image_jobs.json` into
  `input/image_jobs.json`, so the source contract is preserved.

### Offline remote mode (Milestone 3B.1)

`--remote SPEC` selects a transport for the handoff root. `--remote local:<dir>`
is a directory-backed transport that is still fully offline; `--remote
rclone:<remote>:` (Milestone 3B.2, see below) drives a real rclone remote.
`--remote` and `--handoff-root` are mutually exclusive in practice: `--remote`
takes precedence.

Behavior:

- The handoff root `ai-youtube-channel-handoff/` is **discovered** inside the
  remote base (`tools/handoff_transport.py: discover_root`). Exactly one
  directory is reused; none is created **only** with `--create-root` (so
  `prepare --create-root`); more than one match, a case mismatch, or a file
  rather than a directory **stops with an error** — a second root is never
  created and nothing is overwritten.
- The named run folder is mirrored to a temporary local directory, the existing
  offline integrity code runs against the mirror unchanged, and (for `prepare`)
  the mirror is published back with `input/READY` written **last**
  (`upload_last`). Only the named run folder is touched; every other run is left
  alone.
- Read commands (`status`, `fetch`, `validate`, `import`) never write to the
  remote. This is the milestone that a same-account check (D7) will build on.

The transport protocol is `RemoteTransport` (`base_display`, `list_entries`,
`is_dir`, `exists`, `mkdir`, `read_bytes`, `write_bytes`, `download_tree`,
`upload_tree(upload_last=...)`). The rclone implementation added in 3B.2
implements the same methods; the CLI and the integrity helpers do not change.

### rclone / Google Drive transport (Milestone 3B.2)

`--remote rclone:<remote>:` (e.g. `rclone:gdrive:`) runs the handoff against a
real rclone remote. It is enabled only when a sentinel value is also supplied:

```bash
python tools/colab_handoff.py --project-root . \
  --remote rclone:gdrive: \
  --sentinel-value <expected-value> \
  prepare --run-id <run_id> --create-root
```

The transport shells out to `rclone` with explicit argument lists (never a shell
string) via an injectable runner (`default_rclone_runner`, subprocess with
`shell=False`). The runner is not invoked at import time; the module still
imports only standard library. Every remote path component is validated
(`_safe_rel`) so a `-`-leading or absolute component can never reach rclone as
an option or escape the base.

#### Preflight and root creation (read-mostly)

Before any transfer, `preflight_remote(transport, expected_sentinel_value=...)`
runs:

1. `check_access()` — a read-only `rclone about` (remote reachable, auth valid);
2. `discover_root(...)` — exactly one matching
   `ai-youtube-channel-handoff/` is reused; a duplicate match, a case mismatch,
   or a file-instead-of-directory **stops**; a missing root is created **only**
   when `create_root=True` (i.e. `prepare --create-root`) and only once;
3. sentinel verification — when the root exists, `handoff_sentinel.json` must
   parse, name the expected project/root, and match the **operator-supplied**
   `--sentinel-value`. A missing/unreadable/mismatching sentinel **stops** the
   run; the value is never taken from the remote document itself.

When the root is newly created, the sentinel is written with the supplied value
so later runs can verify it. The preflight itself does not create run folders.

#### Bounded, transient-only retries

`RcloneTransport` retries an rclone invocation only for **transient transport**
failures (rclone exit codes `5`/`6` and known rate-limit/timeout/network
markers), up to `max_attempts` (default `3`) with a bounded backoff. Auth,
permission, integrity, and any ambiguous outcome are **never** retried — they
propagate immediately as `CommandError`. A read-only `about` check is likewise
not retried on an auth failure.

#### Non-destructive publishing

`upload_tree` mirrors the payload with `rclone sync` **confined to the single
named run folder**, excluding the content-bearing markers, then writes each
marker last with `rclone copyto` (phase 2 runs only if phase 1 succeeded). No
other run and no sibling root is ever deleted or renamed. `write_bytes` refuses
to overwrite an existing remote file.

> **Note:** rclone reports that the shared Google Drive `client_id` used by the
> default remote configuration is being retired during 2026; a personal OAuth
> client may be required before long-term Drive use.

> **Real-Drive status:** the rclone transport, preflight, sentinel trust and
> retry logic are covered by offline fake-rclone tests only. No project-data
> transfer to real Google Drive has been performed or verified yet.

### Integrity rules enforced by the CLI

- Uses schema **1.1** and the attempt-id contract throughout.
- `READY` is written only after the copied input files and hashes verify.
- `DONE` is trusted only after the ZIP, result manifest, output manifest, status
  and hashes validate (`verify_done`).
- `expected_asset_ids` from the input manifest are always passed to
  `verify_done`.
- The input and output manifests must cross-bind (`verify_handoff_pair`).
- Stale markers, mismatched attempts, invalid manifests, corrupt ZIPs, unsafe
  paths, duplicate members and unexpected/missing assets are rejected.
- `fetch` stages the ZIP **without extracting** it; contents are never written
  into `data/` by the CLI.
- `validate` and `import` share one resolve-then-verify path: completion is
  verified first (READY/DONE markers, attempt binding, `PASS` status, output
  manifest, `verify_handoff_pair`), then the resolved ZIP — explicit `--zip`,
  staged copy, or the run's output ZIP — must match the completed output
  manifest by SHA-256 **and** byte size (`hf32`). An arbitrary, stale or swapped
  ZIP that merely sits under `output/`/staging therefore never validates or
  imports. Identity is by content, not location: a byte-identical copy of the
  artifact passes.
- The in-ZIP result document is validated next
  (`verify_zip_result_document`): the member named by the manifest's
  `result_manifest.filename`, read from the hash-pinned ZIP itself, must
  satisfy `image-generation-results/1.0` for this run (`hf49`/`hf50`). The
  loose `output/result.json` is never consulted as a substitute. `import` runs
  the same checks on the same path it then imports from (no
  time-of-check/time-of-use gap).
- `validate` is non-destructive (importer dry-run); `import` requires
  `--authorize-import` and never overwrites unless `--overwrite` is given.
- The source ZIP and the source `image_jobs.json` are preserved.
- Stage 5 and later stages are never started; Stage 5 always uses a new run id.
- The `--run-id` must be a **bare directory name** (the safe-asset-id shape):
  separators, `.`/`..`, absolute/drive-qualified names, surrounding whitespace
  and empty names are rejected (`hf08`) before any path is built, so a run id can
  never escape `<handoff-root>/<run_id>/`. All five subcommands validate it.

### `status` diagnostics

`status` prints a full report even when something is wrong. It cross-checks the
content-bearing markers against the documents they refer to:

- `READY` content vs the input manifest `attempt_id`
  (`input_ready=yes|no|mismatch`);
- `DONE` content vs the output manifest and output `status.json` `attempt_id`
  (`output_done_matches=yes|no`); and the output manifest vs output status
  attempt ids (`output_attempts_consistent=no` on mismatch);
- progress `status.json` attempt vs the input manifest attempt.

When the progress `status.json` carries an `error`, its `code` and `stage` are
surfaced as `progress_error_code` / `progress_error_stage`, so a prompt-stage
failure (`INPUT_VALIDATION_FAILED` at `env_setup`) is visible even after the
early completion invalidation has removed any output `status.json`.

A malformed/unreadable document is reported inline (`input_manifest=malformed`,
`progress_status=malformed`, ...) with its `[code] message`, and the report still
finishes. Absent markers/documents (e.g. a not-yet-started output) are normal.

Exit codes: `0` success; `2` on any integrity/import failure or refused action
(e.g. missing `--authorize-import`, an unconfined `--run-id`). For `status`
specifically, `2` also means the run is missing, a document is malformed, or a
marker/attempt inconsistency was detected.

### Colab worker notebook (Milestone 3B.3)

`notebooks/SDXL_Lightning_handoff.ipynb` is the free-GPU Colab worker. It is
**`IMPLEMENTED` and offline-tested but `NOT TESTED` on a real Colab GPU** (no run
has been authorized yet).

- The notebook does not fetch project code from the network. It **vendors the
  contract verbatim**: `agents/image_jobs/handoff.py` and the notebook's own
  pure-stdlib logic (`notebooks/handoff_notebook_core.py`) are embedded
  byte-for-byte as cells, and a bootstrap shim registers a minimal stand-in for
  `agents.image_jobs.jobs` so `JOB_SCHEMA_VERSION` resolves without a checkout.
  The exact source commit is recorded in the notebook metadata
  (`aiyt_vendored_commit`); the notebook is reproducible from that commit via
  `python -m notebooks.build_notebook`.
- The trusted sentinel is read from a **Colab Secret** named
  `HANDOFF_SENTINEL_VALUE` (via `google.colab.userdata.get`). It is never
  hardcoded, printed, or persisted, and is compared against the mounted
  `handoff_sentinel.json` before anything else runs.
- Startup: mount Drive -> fail-closed discovery of the single
  `ai-youtube-channel-handoff` root (missing / duplicate / case-mismatch /
  non-directory all stop) -> independent sentinel check -> explicit `RUN_ID`
  (never newest) -> input validation (`READY` + `image_jobs.json` +
  `input_manifest.json`, attempt binding, SHA-256) -> **early completion
  invalidation** -> **GPU preflight** (D4, before any weight download) -> prompt
  token budget (hard error > 77 tokens, warning > 60).
- **Rerun safety:** immediately after input validation the worker clears any
  previous attempt's `DONE`/`output_manifest.json`/`output/status.json`
  (`handoff.begin_attempt`), *before* GPU preflight. A rerun that later fails
  therefore cannot leave an earlier attempt's `DONE` visible — even when the
  attempt id is unchanged (a worker re-run without a new `prepare`). This mirrors
  the invalidation `prepare --force` already performs. Artifacts (the ZIP) are
  kept; only the selected run's completion documents/marker are cleared.
- Generation uses exactly 4 steps, 768x1344, guidance `0.0`, `float16`, and only
  the declared asset set. The output is validated against the exact expected
  asset set before publishing. Publication is **marker-last** (at application
  level, not an atomic filesystem guarantee): the `<run_id>_generated_assets.zip`
  (PNGs + `result.json`), `output_manifest.json` and a `PASS` `output/status.json`
  are written, then `DONE` (whose content must equal the attempt id) is written
  last. A failure writes a `BLOCKED`/`FAILED` status and never writes `DONE`.
  `result.json` is produced by `build_result_document` (the producer side of
  `image-generation-results/1.0`); `publish_outputs` packages whatever document
  it is handed, so the consumption boundary is what fails closed — `validate`
  and `import` reject a non-contract document with `hf49`/`hf50`.
- **Prompt-token failures write a terminal `FAILED` status.** An over-limit
  (or empty) prompt is a deterministic *input* defect. The worker writes an
  attempt-bound `FAILED` `status/status.json` with
  `code=INPUT_VALIDATION_FAILED` (stage `env_setup`, `recoverable=false`, the
  prompt findings under `error.diagnostics`), then stops *before* any model
  download or generation. This is distinct from the recoverable infrastructure
  fault `ENV_SETUP_FAILED` (`BLOCKED`): `INPUT_VALIDATION_FAILED` means the input
  must be fixed and re-prepared, not blind-retried. Because the early
  completion invalidation runs first and no `DONE`/output manifest is written,
  a stale success cannot survive the failure. The atomicity caveat for the
  completion markers is unchanged.
- The notebook never writes `data/` and never imports assets. Import stays a
  separate, explicitly authorized local step.

Offline tests live in `tests/test_notebook_handoff.py`.

## Manual resume procedure

There is no automatic retry. On a `BLOCKED` or `FAILED` status:

1. Read `status/output/status.json` (or `status/status.json`). Note
   `error.code`, `error.stage`, `completed_asset_ids` and diagnostics.
2. If `error.recoverable` is `true` (`BLOCKED`): provision a fresh free GPU,
   re-run the GPU preflight, then start a **new** attempt. Never auto-loop.
3. If `failed_assets` is non-empty, decide whether to regenerate those assets
   or abandon the run; do not patch generated artifacts by hand.
4. On any resume, the input must still validate (`input/READY` content ==
   `input_manifest.json` `attempt_id`, plus the `image_jobs_sha256` hash) and any
   fetched output must validate (`output/DONE` content == output manifest
   `attempt_id`, `status.json` `PASS`, `output_manifest.json` + ZIP fingerprint)
   before import. Call `begin_attempt(output_dir, new_attempt_id)` to clear a
   previous attempt's `DONE`/manifest/status so a stale completion cannot be
   re-fetched.
5. Verify the pair before import: `verify_handoff_pair(...)` must accept the
   input and output manifests (same run, hash, attempt, resolution, config, and
   exact asset set).
6. Import with the existing importer, never overwriting by default:
   `python -m agents.image_jobs.import_assets --run-id <run_id> --zip <zip>`.
7. Stage 5 is never started automatically and always uses a new run ID (D5).

## Implementation map

| Concern | Location |
| --- | --- |
| Constants, schemas, hashing | `agents/image_jobs/handoff.py` |
| Input/output manifest build + verify | `agents/image_jobs/handoff.py` |
| Input<->output pair binding (`verify_handoff_pair`) | `agents/image_jobs/handoff.py` |
| Status build + validate, error mapping | `agents/image_jobs/handoff.py` |
| Attempt id + content-bearing markers (`READY`/`DONE`, `begin_attempt`) | `agents/image_jobs/handoff.py` |
| ZIP integrity | `agents/image_jobs/handoff.py` |
| GPU preflight evaluator (fail-closed) | `agents/image_jobs/handoff.py` |
| Contract tests | `tests/test_colab_handoff.py` |
| Local CLI | `tools/colab_handoff.py` (implemented, offline backend) |
| Local CLI tests | `tests/test_colab_handoff_cli.py` |
| Transport protocol + offline directory transport + root discovery + mirror backend | `tools/handoff_transport.py` (Milestone 3B.1) |
| rclone transport, `preflight_remote`, sentinel build/verify | `tools/handoff_transport.py` (Milestone 3B.2) |
| Transport / remote-mode tests | `tests/test_colab_handoff_drive.py` |
| rclone transport / sentinel / preflight / retry tests (offline fake rclone) | `tests/test_rclone_transport.py` |
| Worker notebook | `notebooks/SDXL_Lightning_handoff.ipynb` (generated; Milestone 3B.3) |
| Notebook pure-stdlib logic (vendored verbatim) | `notebooks/handoff_notebook_core.py` |
| Notebook builder (deterministic, records source commit) | `notebooks/build_notebook.py` |
| Notebook contract / drift / offline logic tests | `tests/test_notebook_handoff.py` |
