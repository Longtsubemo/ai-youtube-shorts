# Policy And Safety

Quality-control stage (a.k.a. **Stage 9 — Policy / Publishing QA**) that
checks a finished Video against YouTube policy, advertiser suitability, and
the channel's own standards **before publishing**.

**Status:** IMPLEMENTED as a deterministic, network-free, GPU-free, API-free
gate in `agents/qa/policy/` (135+ lines of policy rules P01–P15, a CLI, and a
test suite). The gate is a *pre-publication risk assessment*; it is explicitly
**NOT a YouTube or platform compliance guarantee**, does not talk to YouTube,
and does not block or replace the mandatory human approval gate (§7). The
package deliberately contains no Stage 10 (human approval) or Stage 11
(YouTube publishing) code.

## 1. Position in pipeline

```
...
→ video QA
→ policy / publish QA
→ human approval
→ YouTube upload
```

The policy stage is the last automated gate. `PASS` does not bypass
human approval — publishing still requires the human gate (§7).

## 2. Checks

The policy checker evaluates each finished video against the risk areas
that matter for this channel. Outcomes are recorded per check.

| Risk area | What the checker looks for |
|---|---|
| Community Guidelines | Content that is prohibited without nuance (dangerous, hateful, misleading medical, etc.) |
| Misleading claims | Over-the-top, unverifiable, or type-mismatched claims presented as fact |
| Copyright / reused content | Unlicensed clips, images, audio, music, fonts; unoriginal reuse |
| Repetitive or mass-produced content | Near-duplicate videos reused across the channel or web at scale |
| AI-generated / synthetic media disclosure | Content that realistically depicts people/events → labeling/disclosure needed per policy |
| Advertiser suitability | Topics earning Limited Ads / not suitable status |
| Spam / deceptive behavior | Inflated claims, keyword-stuffing, engagement bait, manipulative descriptions |

**Design stance:** AI-generated content is not automatically
prohibited. What is prohibited is *faceless spam*: mass-produced,
near-identical, low-substance variations. The channel's defense is
genuinely substantive, original, single-topic explainers
(`CONTENT_STRATEGY.md`).

### 2.1 Implemented checks (Stage 9)

The implemented gate grades what the recorded artifacts PROVE — the
Script, FactCheckResult, ProductionPlan, VideoGenerationResult and
VideoQAResult files — with 15 deterministic rules (`policy_version =
policy_rules_v1`) in `agents/qa/policy/checks.py` (registry:
`rules.py`):

| Check | Risk area graded | Gate |
|---|---|---|
| P01 | Harmful content (hard markers) | hard → REJECT, soft → REVIEW |
| P02 | Dangerous instructions (negator-guarded) | soft → REVIEW |
| P03 | Overstatement vs `strong_in_claim=none` | soft → REVIEW |
| P04 | Definitive misinformation framing | soft → REVIEW |
| P05 | "Actual footage" claimed while artifact `mock=True` | hard → REJECT |
| P06 | Synthetic media of real people/events without disclosure | soft → REVIEW |
| P07 | Asset rights / provenance markers (watermark, copyrighted, ...) | hard/soft |
| P08 | PII / doxxing leaks | hard/soft |
| P09 | Spam metadata (≥2 markers) | soft → REVIEW |
| P10 | Near-duplicate / reused content (overlap ≥0.8) | soft → REVIEW |
| P11 | Advertiser-sensitive topic without disclaimer | soft → REVIEW |
| P12 | Sensational title/hook absent from content | soft → REVIEW |
| P13 | Missing provenance on recorded artifacts | soft → REVIEW |
| P14 | Unregistered `policy_version` | hard → REJECT |
| P15 | Stage-8 prior gate missing / non-PASS / stale target_ref | soft → REVIEW |

Uncertain classifications resolve to REVIEW, never to PASS.

## 3. Verdicts

| Verdict | Meaning | Action |
|---|---|---|
| `PASS` | No material risk found | Allowed to proceed to human approval |
| `REVIEW` | Ambiguous; needs a human decision | Block until human decides |
| `REJECT` | Material violation or unresolvable risk | Blocked; video not publishable as-is |

Rejected videos cannot proceed by any path. REVIEW blocks until a human
acts — never auto-approves after N seconds.

## 4. Inputs the checker needs

- final video file (assembly output),
- `ProductionPlan` metadata (scene/subtitle text, visuals used),
- source/material provenance of assets used (licensing),
- script + approved claims (to judge "is this claim type presented as
  fact?"),
- uploaded-status flags (thumbnail, description, tags, title drafts).

This is why asset provenance is tracked from research through
production (`DATA_CONTRACTS.md` + `ARCHITECTURE.md` traceability).

## 5. Output

`PolicyQAResult` (a §11 `QAResult`, `stage = "policy_qa"`) with:

- per-check outcomes (PASS/REVIEW/REJECT + counts) for P01–P15,
- overall verdict,
- a `risk_summary` (flagged categories + a `reviewer_note`),
- issues with affected scene/line/claim/asset ids (audit chain) and
  recommended actions (e.g. "add synthetic-media disclosure",
  "re-render asset X", "rewrite description").

Written to `data/<run_id>/production/policy_qa_result.json`. The header
`stage` `"policy_qa"` maps to the generic pipeline state
`Stage.QUALITY_CHECK` (the same mapping Stages 5/8 use — see `STATE_STAGE`).
CLI: `python -m agents.qa.policy` (exit 0 PASS, 1 REVIEW/REJECT, 2 error).

## 6. Positioning vs. other QA

| Stage | Scope |
|---|---|
| script QA | Script quality, claim-fact type alignment |
| video QA | Technical/visual/audio issues in the render |
| **policy/publish QA** | Eligibility to be published at all |

They are separate stages with the same `QAResult` shape, different
`stage` tags (`pipeline/state.py` `Stage` already enumerates them).

### 6.1 Boundary of the implemented gate

- It reads the **recorded artifacts only** (plan scenes, metadata, QA
  verdicts); the mock-first pipeline has no assembled media bytes, so
  checks that need rendered pixels/audio remain metadata/contract-level.
- It never calls YouTube, never reads secrets/env, never uses network,
  GPU, or an external API, and never writes beyond its own artifact.
- Marker vocabulary is deliberately *word-aware*: P05/P06 only flag real
  media / real people-event claims ("actual footage", "the president"),
  so innocuous "realistic rendering of an octopus" content stays benign
  and the clean committed fixtures PASS end-to-end.
- `PASS` never means "platform-approved": the reviewer note records
  "Not a YouTube or platform compliance guarantee", and §7's human gate
  remains mandatory.

## 7. Human approval gate

**Status:** IMPLEMENTED as **Stage 10** (`agents/human_approval/`). It is a
machine-verifiable record of an explicit human decision over the exact
reviewed artifact versions, **not** an automated gate and **not** the
publishing stage. Full contract in `docs/DATA_CONTRACTS.md` §12 and design in
`docs/HUMAN_APPROVAL_STAGE10_DESIGN.md`.

- **Mandatory** before any YouTube upload: setup, metadata, and manual upload
  happen by hand; the stage produces a `ManualUploadReadiness` ready flag
  (`BooleanOperand "manual_upload"` `True`) plus the upload title.
- The gate presents: policy report (Stage 9), video QA report (Stage 8),
  video generation result (Stage 7), production plan (Stage 6), script
  (Stage 4), and the optional script-QA / fact-check results.
- **Decisions** (`PENDING` / `APPROVED` / `CHANGES_REQUESTED` / `REJECTED`;
  reviewed-artifact digests are recorded with every decision):
  - `APPROVED` — only ever recorded when ALL hard gates are cleared (required
    artifacts present, no Stage-8/9 hard REJECT, video generation completed,
    review-verdict overrides satisfied). A blocked APPROVED error is the
    fail-closed default: nothing is written.
  - `CHANGES_REQUESTED` / `REJECTED` — valid under any gate state; the
    blocking conditions that made approval impossible are recorded for audit.
    `CHANGES_REQUESTED` requires at least one change note (or
    `changes_requested` requirement).
  - `PENDING` is a model value but the CLI refuses to write it — *no written
    decision = no approval* (absence of a rejection is never approval).
- **Overrides:** a soft `REVIEW` on Stage 8 (`video_qa`) or Stage 9
  (`policy_qa`) may be overridden to a recorded explainable decision; a hard
  `REJECT` can never be overridden. Every override is bound to the
  `artifact_id` of the exact artifact version it reviewed.
- **Staleness / supersede:** each reviewed artifact records
  `content_sha256` (canonical SHA-256 of the artifact payload) and `artifacts.bind`
  records ids; `check_approval_freshness` and `approval_is_effective` fail the
  digest/id comparison, so a stale approval is detected (`--check
  --approval <path>`); a newer approval supersedes an earlier one via
  `supersedes_approval_id`.
- CLI: `python -m agents.human_approval` (exit 0 APPROVED, 1
  CHANGES_REQUESTED/REJECTED or stale in `--check` mode, 2 error). Writes
  `data/<run_id>/approval/human_approval_result.json` + run state advanced to
  the existing generic `Stage.PUBLISH` (Stage 10 does not implement any upload).
- Automating approval remains possible **later, only after sufficient
  testing**; the stage stays a first-class record so a future automated
  approver produces the same contract.
- Fail-closed rule: absence of a rejection is not approval — and approval is
  only ever *recorded*, never *performed*: the endpoint is a MANUAL upload
  with no Stage-11 (YouTube/network/oauth) code.

## 8. Open questions

- Which licensing sources/registries we trust for assets (stock,
  CC-BY, generated+recorded) — must be settled before first upload.
- Synthetic-media disclosure policy specifics for *faceless* content
  (whether/how to label; default is to disclose per current policy).
- Extra channel metadata (description template, hashtags) rules —
  consistent and non-spammy.