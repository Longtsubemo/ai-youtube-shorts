"""Deterministically build ``notebooks/SDXL_Lightning_handoff.ipynb``.

The notebook is a generated artifact. Its two "vendored" cells are the exact
bytes of the source files it ships:

* ``agents/image_jobs/handoff.py`` (the schema-1.1 contract), and
* ``notebooks/handoff_notebook_core.py`` (the pure-stdlib notebook logic).

Everything else (bootstrap shim, Colab mount, GPU/model/generation cells) is a
static template in this file. Running this builder reproduces the notebook
byte-for-byte from the current sources, which is what the offline drift test in
``tests/test_notebook_handoff.py`` checks. No network, no Drive, no GPU.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import nbformat as nbf

from agents.image_jobs.jobs import JOB_SCHEMA_VERSION

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "SDXL_Lightning_handoff.ipynb"
HANDOFF_SOURCE = ROOT / "agents" / "image_jobs" / "handoff.py"
CORE_SOURCE = ROOT / "notebooks" / "handoff_notebook_core.py"


def recorded_commit() -> str:
    """Return the current git commit, or ``"unknown"`` outside a checkout."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source)


def _md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source)


# --- static cell templates ---------------------------------------------------

_INTRO = r"""# SDXL-Lightning Colab Handoff Worker

Manual, free-GPU worker for the **`ai-youtube-channel`** handoff. It consumes a
run that the local pipeline already prepared in the shared Google Drive root,
generates exactly the declared scene backgrounds, and publishes a verified
output for the Windows side to fetch and import.

**Vendored sources (do not edit inside the notebook):**

- `agents/image_jobs/handoff.py` — schema-1.1 contract (vendored verbatim).
- `notebooks/handoff_notebook_core.py` — this notebook's pure logic (vendored verbatim).

Source commit: `__COMMIT__` (see notebook metadata `aiyt_vendored_commit`).

**Rules**

- Free GPU only. Fail closed on any GPU/model/output problem. No CPU or paid fallback.
- No automatic retries, reconnect loops, schedulers, or Stage-5/import actions.
- The generated assets are **never** written to the local `data/` tree and are
  never imported here; the Windows-side importer stays separate.
- The notebook writes `DONE` last and only after full validation. It never
  publishes `DONE` on a failure.
- On a rerun it clears any previous attempt's completion before preflight, so a
  failed rerun can never leave a stale `DONE` visible.
"""

_SHIM = r"""# BOOTSTRAP — let the vendored contract import resolve without a repository.
# The contract imports `JOB_SCHEMA_VERSION` from `agents.image_jobs.jobs`; Colab
# has no checkout, so register a minimal stand-in. The value is asserted against
# the real module by the offline drift test.
import sys as _sys
import types as _types

_JOBS_SCHEMA = __JOB_SCHEMA_VERSION__

for _name in ("agents", "agents.image_jobs"):
    if _name not in _sys.modules:
        _mod = _types.ModuleType(_name)
        _mod.__path__ = []
        _sys.modules[_name] = _mod
        if _name == "agents.image_jobs":
            _sys.modules["agents"].image_jobs = _mod

if "agents.image_jobs.jobs" not in _sys.modules:
    _jobs = _types.ModuleType("agents.image_jobs.jobs")
    _jobs.JOB_SCHEMA_VERSION = _JOBS_SCHEMA
    _sys.modules["agents.image_jobs.jobs"] = _jobs
    _sys.modules["agents.image_jobs"].jobs = _jobs
"""

_CONFIG = r"""# CONFIG — operator settings and shared constants.

# >>> EDIT THIS: the exact run id to process. It is validated, never auto-selected.
RUN_ID = "REPLACE_WITH_RUN_ID"

# The Drive base that contains the single `ai-youtube-channel-handoff` root.
DRIVE_BASE = "/content/drive/MyDrive"

MODEL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
LIGHTNING_REPO = "ByteDance/SDXL-Lightning"
CHECKPOINT = "sdxl_lightning_4step_unet.safetensors"

if not isinstance(RUN_ID, str) or RUN_ID in ("", "REPLACE_WITH_RUN_ID"):
    raise SystemExit("Set RUN_ID to the explicit run id before running.")
"""

_INSTALL = r"""# INSTALL — pinned inference dependencies.
!pip uninstall -y torchao -q
!pip install -q diffusers==0.35.2 transformers accelerate safetensors huggingface_hub
"""

_MOUNT = r"""# MOUNT DRIVE + READ THE TRUSTED SENTINEL VALUE.
# The value is read from a Colab Secret and is never printed, logged, or written.
from google.colab import drive, userdata

drive.mount("/content/drive")
TRUSTED_SENTINEL = userdata.get("HANDOFF_SENTINEL_VALUE")
if not TRUSTED_SENTINEL:
    raise SystemExit(
        "Missing Colab Secret HANDOFF_SENTINEL_VALUE (enable notebook access)."
    )
"""

_DISCOVER = r"""# ROOT DISCOVERY + SENTINEL TRUST + EXPLICIT RUN + INPUT VALIDATION.
root = discover_handoff_root(DRIVE_BASE)        # fail-closed: single root only
verify_sentinel(root, TRUSTED_SENTINEL)         # independent trusted value
run_dir = select_run(root, RUN_ID)              # explicit run; never the newest
input_manifest = verify_run_input(root, RUN_ID) # schema 1.1 + attempt + SHA-256

ATTEMPT_ID = input_manifest["attempt_id"]
IMAGE_JOBS_SHA256 = input_manifest["image_jobs_sha256"]
GENERATION_CONFIG = input_manifest["generation_config"]
EXPECTED_ASSET_IDS = list(input_manifest["expected_asset_ids"])
RESOLUTION = (
    int(input_manifest["expected_resolution"]["width"]),
    int(input_manifest["expected_resolution"]["height"]),
)
INPUT_JOBS_PATH = run_dir / INPUT_DIR_NAME / IMAGE_JOBS_FILENAME
STATUS_DIR = run_dir / STATUS_DIR_NAME
OUTPUT_DIR = run_dir / OUTPUT_DIR_NAME

# Invalidate any previous attempt's completion NOW, after input validation and
# before any step that can fail (GPU preflight and beyond). Without this, a
# rerun of the same run id could fail while an earlier attempt's DONE/manifest/
# status.json remain visible and still verify as this attempt's result. This is
# the same invalidation `prepare --force` performs; the worker repeats it because
# it can be re-run without a new prepare. Artifacts (e.g. the ZIP) are kept.
invalidate_stale_completion(OUTPUT_DIR, attempt_id=ATTEMPT_ID)

print("root:", root)
print("run:", RUN_ID, "attempt:", ATTEMPT_ID)
print("assets:", EXPECTED_ASSET_IDS)
print("resolution:", RESOLUTION)
print("generation_config:", GENERATION_CONFIG)
"""

_GPU = r"""# GPU PREFLIGHT — inspect the assigned runtime BEFORE any model download.
import subprocess

import torch


def _gpu_facts():
    cuda = torch.cuda.is_available()
    count = torch.cuda.device_count() if cuda else 0
    name = torch.cuda.get_device_name(0) if count else ""
    total_gb = None
    free_gb = None
    cuda_op_ok = False
    if count:
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        try:
            free_bytes, _total = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024 ** 3)
        except Exception:
            free_gb = None
        try:
            _ = (torch.ones(1, device="cuda") * 1).sum().item()
            cuda_op_ok = True
        except Exception:
            cuda_op_ok = False
    try:
        smi = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        smi = ""
    return {
        "cuda_available": cuda,
        "device_count": count,
        "gpu_name": name,
        "total_vram_gb": total_gb,
        "free_vram_gb": free_gb,
        "cuda_op_ok": cuda_op_ok,
        "nvidia_smi": smi,
    }


GPU_FACTS = _gpu_facts()
GPU_DECISION = preflight_decision(GPU_FACTS)
ENVIRONMENT = {
    "gpu_name": GPU_FACTS["gpu_name"],
    "total_vram_gb": GPU_FACTS["total_vram_gb"],
    "free_vram_gb": GPU_FACTS["free_vram_gb"],
}
print("preflight ok:", GPU_DECISION["ok"], "code:", GPU_DECISION["error_code"])


def fail(code, stage, message, diagnostics=None, completed_stages=None):
    write_failure_status(
        STATUS_DIR,
        run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        image_jobs_sha256=IMAGE_JOBS_SHA256,
        generation_config=GENERATION_CONFIG,
        code=code,
        stage=stage,
        message=message,
        diagnostics=diagnostics,
        completed_stages=completed_stages,
        environment=ENVIRONMENT,
    )
    raise SystemExit(f"[{code}] {message}")


if not GPU_DECISION["ok"]:
    fail(
        GPU_DECISION["error_code"],
        "gpu_preflight",
        "GPU preflight failed (free GPU with >=14 GiB total and >=10 GiB free required)",
        diagnostics=GPU_DECISION["diagnostics"],
    )
"""

_PROMPT = r"""# PROMPT TOKEN CHECK — before heavy model downloads.
# Loads only the CLIP tokenizers (small), never the UNet/pipeline weights.
import json

from transformers import CLIPTokenizer

_tok = CLIPTokenizer.from_pretrained(
    MODEL_REPO, subfolder="tokenizer", cache_dir=str(HF_CACHE / "hub")
)
_tok2 = CLIPTokenizer.from_pretrained(
    MODEL_REPO, subfolder="tokenizer_2", cache_dir=str(HF_CACHE / "hub")
)
batch = json.loads(Path(INPUT_JOBS_PATH).read_text(encoding="utf-8"))


def _longest(prompt):
    a = len(_tok(prompt, add_special_tokens=True, truncation=False)["input_ids"])
    b = len(_tok2(prompt, add_special_tokens=True, truncation=False)["input_ids"])
    return max(a, b)


prompt_errors, prompt_warnings = prompt_validation_findings(batch["jobs"], _longest)

for warning in prompt_warnings:
    print("WARN:", warning)
if prompt_errors:
    # Write an attempt-bound FAILED progress status, then stop before any model
    # download or generation. INPUT_VALIDATION_FAILED (stage env_setup) is a
    # deterministic input defect -> overall FAILED, recoverable=False; it is NOT
    # the recoverable infrastructure ENV_SETUP_FAILED. The stale completion was
    # already invalidated in the DISCOVER cell (invalidate_stale_completion), and
    # fail() raises SystemExit, so no DONE and no output manifest are produced.
    fail(
        "INPUT_VALIDATION_FAILED",
        "env_setup",
        "Prompt validation failed; fix image_jobs.json and re-prepare:\n- "
        + "\n- ".join(prompt_errors),
        diagnostics={
            "prompt_errors": prompt_errors,
            "prompt_warnings": prompt_warnings,
        },
        completed_stages=["gpu_preflight"],
    )
"""

_MODEL = r"""# MODEL LOAD — FP16 SDXL-Lightning, 4-step UNet asserted vs the manifest.
from diffusers import (
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

assert GENERATION_CONFIG["steps"] == 4, GENERATION_CONFIG
assert GENERATION_CONFIG["checkpoint"] == CHECKPOINT, GENERATION_CONFIG
assert GENERATION_CONFIG["guidance_scale"] == 0.0, GENERATION_CONFIG
assert GENERATION_CONFIG["dtype"] == "float16", GENERATION_CONFIG
assert (GENERATION_CONFIG["width"], GENERATION_CONFIG["height"]) == RESOLUTION

try:
    _unet_path = hf_hub_download(
        repo_id=LIGHTNING_REPO, filename=CHECKPOINT, cache_dir=str(HF_CACHE / "hub")
    )
    _unet = UNet2DConditionModel.from_config(
        MODEL_REPO, subfolder="unet", cache_dir=str(HF_CACHE / "hub")
    )
    _unet.load_state_dict(load_file(_unet_path, device="cpu"), strict=True)
    _unet = _unet.to(dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_REPO,
        unet=_unet,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        cache_dir=str(HF_CACHE / "hub"),
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe = pipe.to("cuda")
except Exception as exc:
    fail("MODEL_LOAD_FAILED", "model_load", f"model load failed: {exc}")
print("SDXL-Lightning loaded (4 steps, float16).")
"""

_GENERATE = r"""# GENERATION — only the declared assets; no retries or reconnect loops.
from PIL import Image


def _is_valid_image(path, resolution):
    path = Path(path)
    if not path.is_file():
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            return img.size == resolution
    except Exception:
        return False


write_progress_status(
    STATUS_DIR,
    run_id=RUN_ID,
    attempt_id=ATTEMPT_ID,
    image_jobs_sha256=IMAGE_JOBS_SHA256,
    generation_config=GENERATION_CONFIG,
    completed_stages=["gpu_preflight", "env_setup", "model_load"],
    current_stage="generation",
    environment=ENVIRONMENT,
)

generated_files = {}
results = []
for item in batch["jobs"]:
    asset_id = item["asset_id"]
    out_path = GENERATED_DIR / f"{asset_id}.png"
    out_path.unlink(missing_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(int(item["seed"]))
    try:
        image = pipe(
            prompt=item["positive_prompt"],
            negative_prompt=item.get("negative_prompt", ""),
            width=RESOLUTION[0],
            height=RESOLUTION[1],
            num_inference_steps=GENERATION_CONFIG["steps"],
            guidance_scale=GENERATION_CONFIG["guidance_scale"],
            generator=generator,
        ).images[0]
        image.save(out_path)
    except Exception as exc:
        fail("GENERATION_FAILED", "generation", f"generation failed for {asset_id}: {exc}")
    if not _is_valid_image(out_path, RESOLUTION):
        fail("GENERATION_FAILED", "generation", f"invalid image for {asset_id}")
    generated_files[asset_id] = out_path
    results.append(
        {
            "scene_id": item["scene_id"],
            "asset_id": asset_id,
            "filename": f"{asset_id}.png",
            "status": "generated",
            "seed": int(item["seed"]),
            "width": RESOLUTION[0],
            "height": RESOLUTION[1],
        }
    )
print("generated:", sorted(generated_files))
"""

_PUBLISH = r"""# OUTPUT VALIDATION + PACKAGING + MARKER-LAST PUBLICATION.
result_document = build_result_document(
    RUN_ID, batch, results, resolution=RESOLUTION, generation_config=GENERATION_CONFIG
)
try:
    output_manifest = publish_outputs(
        OUTPUT_DIR,
        run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        image_jobs_sha256=IMAGE_JOBS_SHA256,
        generated_files=generated_files,
        result_document=result_document,
        expected_asset_ids=EXPECTED_ASSET_IDS,
        resolution=RESOLUTION,
        generation_config=GENERATION_CONFIG,
        environment=ENVIRONMENT,
    )
except NotebookHandoffError as exc:
    fail("OUTPUT_VALIDATION_FAILED", "output_validation", str(exc))
print("published; DONE written for attempt", ATTEMPT_ID)
print("zip:", RUN_ID + OUTPUT_ZIP_SUFFIX)
"""

# The pre-GENERATION cells need `torch` and config pieces defined before use.
# `GENERATED_DIR`, `HF_CACHE` are set in the CONFIG cell.
_EXTRA_CONFIG = r"""# TEMPORARY STORAGE.
import os

MODEL_ROOT = Path("/content/models")
HF_CACHE = MODEL_ROOT / "huggingface"
GENERATED_DIR = Path("/content/generated_assets")
HF_CACHE.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE / "transformers")
"""

_NOTES = r"""## Notes

- **Trusted value:** `HANDOFF_SENTINEL_VALUE` is a Colab Secret, read with
  `google.colab.userdata.get`. It must never be hardcoded, printed, or written to
  Drive. Locally it lives in a Windows-DPAPI CurrentUser store.
- **Vendoring:** the contract and core cells are byte-for-byte copies of their
  source files. Regenerate with `python notebooks/build_notebook.py`; the offline
  drift test fails if the vendored copies diverge.
- **Rerun safety:** immediately after input validation the worker clears any
  previous attempt's `DONE`/`output_manifest.json`/`output/status.json`
  (`begin_attempt`), before GPU preflight. A rerun that later fails therefore
  cannot leave an earlier `DONE` visible as the current result, even when the
  attempt id is unchanged. Artifacts (the ZIP) are kept.
- **Failure handling:** GPU/env/model/generation/output problems write a
  `BLOCKED`/`FAILED` status and stop; `DONE` is never written. A prompt-budget or
  empty-prompt failure writes an attempt-bound `FAILED` `status/status.json` with
  `code=INPUT_VALIDATION_FAILED` (stage `env_setup`, `recoverable=false`, prompt
  findings in `diagnostics`) and stops before any model download or generation.
  This deterministic input defect is distinct from a recoverable `env_setup`
  infrastructure fault (`ENV_SETUP_FAILED`, `BLOCKED`).
- **Import is separate:** fetch + import happen on Windows via
  `tools/colab_handoff.py` and `agents/image_jobs/import_assets.py`. This notebook
  never touches `data/`.
"""


def build_notebook(commit: str | None = None) -> nbf.NotebookNode:
    """Return the notebook as an nbformat document for ``commit``."""
    if commit is None:
        commit = recorded_commit()
    handoff_src = _read(HANDOFF_SOURCE)
    core_src = _read(CORE_SOURCE)

    cells = [
        _md(_INTRO.replace("__COMMIT__", commit)),
        _code(_SHIM.replace("__JOB_SCHEMA_VERSION__", repr(JOB_SCHEMA_VERSION))),
        _code(handoff_src),
        _code(core_src),
        _code(_INSTALL),
        _code(_CONFIG),
        _code(_EXTRA_CONFIG),
        _code(_MOUNT),
        _code(_DISCOVER),
        _code(_GPU),
        _code(_PROMPT),
        _code(_MODEL),
        _code(_GENERATE),
        _code(_PUBLISH),
        _md(_NOTES),
    ]

    # Deterministic cell ids so the notebook rebuilds byte-for-byte.
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:02d}"

    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3"}
    notebook.metadata["aiyt_vendored_commit"] = commit
    notebook.metadata["aiyt_vendored_sources"] = {
        "agents/image_jobs/handoff.py": "vendored",
        "notebooks/handoff_notebook_core.py": "vendored",
    }
    return notebook


def write_notebook(path: Path | None = None, commit: str | None = None) -> Path:
    """Write the built notebook to ``path`` (default the repo notebook)."""
    path = Path(path) if path is not None else NOTEBOOK_PATH
    notebook = build_notebook(commit)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        nbf.write(notebook, handle, version=4)
    return path


def main() -> int:
    path = write_notebook()
    print(f"wrote {path} (commit {recorded_commit()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
