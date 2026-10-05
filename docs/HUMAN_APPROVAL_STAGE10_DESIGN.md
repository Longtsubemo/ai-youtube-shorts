# Stage 10 — Human Approval: Research & Design Report

**Status:** IMPLEMENTED in `agents/human_approval/` (Stage 10 complete).

This document records the Stage-10 research/design pass and the implemented
contract over the current repository (Stages 1–10 complete; Stages 1–4 have
opt-in real provider paths, while Stages 5–10 are deterministic/mock-first).
Stage 11 (YouTube upload) is intentionally out of scope; the project goal is a
manual-upload-ready video package.

---

## A. Current architecture understanding

**Pipeline stages 1–10 are complete.** All stages communicate
only via file-based, versioned JSON contracts under `data/<run_id>/` (envelope
header + `data` payload, `docs/DATA_CONTRACTS.md` §1). Every artifact is
self-describing with `schema`, `schema_version`, `created_at`, `run_id`,
`stage`, `produced_by`, and a `provenance` record.

- **Stage 3** `FactCheckResult` (`agents/qa/fact_checker/`) — PASS/REVIEW/REJECT;
  `approved_claims` is the only thing Stages 4–6 may consume.
- **Stage 4** `Script`, **Stage 5** `ScriptQAResult` (PASS/**REVISE**/REJECT),
  **Stage 6** `ProductionPlan` (`pp_…`, the stable CONTENT→VIDEO boundary),
  **Stage 7** `VideoGenerationResult` (`vg_…`, mock artifacts, `mock=True`, empty
  `uri`), **Stage 8** `VideoQAResult` (`qa_…`, `target_ref`=plan id, 15 checks;
  hard→REJECT / soft→REVIEW), **Stage 9** `PolicyQAResult` (`pr_…`, 15 rules
  P01–P15; same gate model; explicitly *not* a platform-compliance guarantee),
  **Stage 10** `HumanApprovalResult` (`ha_…`; explicit human decision bound to
  reviewed artifact digests; manual-upload readiness).
- Gate aggregation everywhere: **hard ERROR → REJECT, soft ERROR → REVIEW
  (REVISE at Stage 5), clean → PASS**; gates only make things worse, never
  upgrade (`PASS < REVIEW < REJECT`).
- **Stage 9 P15** forces prior-gate honesty: a Stage-8 REVIEW/REJECT, missing
  QA, or stale `target_ref` surfaces as a Stage-9 REVIEW — a prior gate can
  never silently become a policy PASS.
- Media bytes do not exist yet (mock-first): QA results are permission-
  metadata/contract-level checks.
- Pipeline state (`pipeline/state.py`) has no dedicated human-approval enum
  member. Stages 8/9 deliberately advance state to the existing generic
  `Stage.QUALITY_CHECK`; an APPROVED Stage-10 decision advances to the existing
  generic `Stage.PUBLISH`, while non-approved decisions remain at
  `Stage.QUALITY_CHECK` (`STATE_STAGE`; decisions #16/#17 in PROJECT_PROGRESS).
  No enum member is added or renamed.
- High-level pipeline direction: `Human Approval` is the implemented Stage-10
  gate before the out-of-scope YouTube stage (`docs/PROJECT_PROGRESS.md`,
  `docs/ARCHITECTURE.md`, `docs/POLICY_AND_SAFETY.md`).

## B. Stage-10 purpose

Produce a **deterministic, machine-verifiable record of a human review
decision** that turns a QA-complete run into a **manual-upload-ready package**.
It is the last gate before the (out-of-scope) YouTube stage. Per
`POLICY_AND_SAFETY.md` §7:

- the gate is **mandatory** and **fail-closed** (“silent default =
  REJECT/PENDING”),
- it presents: policy report, video QA report, script, claims trace, and the
  finished video,
- it stays a first-class stage so it can be relaxed later without surgery.

Stage 10 must **not** do upload/scheduling/analytics, must not assume a real
assembled video exists (mock-first honesty), and must keep the whole chain
auditable: `sentence → claim → source` and `script → plan → generated artifacts
→ QA → human approval → exact reviewed version`.

## C. Inputs

Stage 10 consumes the approval-relevant bundle. The required roles are
`PolicyQAResult`, `VideoQAResult`, `VideoGenerationResult`, `ProductionPlan`,
and `Script`; `ScriptQAResult` and `FactCheckResult` are optional and are
validated and bound when present. The typed `ApprovalInputs` container mirrors
the cross-stage `PolicyInputs` pattern:

1. **Stage 9 `PolicyQAResult`** — the primary gate input (`policy_result_id`,
   `overall_verdict`, per-check records, `risk_summary`, `target_ref`).
2. **Stage 8 `VideoQAResult`** — quality report (`overall_verdict`, checks,
   issues).
3. **Stage 7 `VideoGenerationResult` + Stage 6 `ProductionPlan`** — confirm the
   video side is complete (`overall_status=="completed"`, all scenes done) and
   anchor `target_ref`.
4. **Stage 4 `Script` + Stage 3 `FactCheckResult` (+ research package)** —
   claims trace for the reviewer.
5. **Stage 5 `ScriptQAResult`** — script QA report (PASS required upstream,
   technically already enforced at Stage 6; kept for the reviewer page).
6. **Future assembled/rendered video package** — optional/absent in the current
   mock-first pipeline; Stage 10 grades what records prove (mirror of Stage
   8/9 “no media bytes exist, metadata-level checks”).

Implemented input container: a typed `ApprovalInputs` dataclass mirroring
`PolicyInputs`.

## D. Implemented output contract

`HumanApprovalResult` is a versioned artifact following the §1 envelope. The
authoritative serialization and validation are in `agents/human_approval/models.py`;
the canonical field list is maintained in `docs/DATA_CONTRACTS.md` §12.

```jsonc
{
  "schema": "HumanApprovalResult",
  "schema_version": 1,
  "created_at": "...iso...",
  "run_id": "<run_id>",
  "stage": "human_approval",            // task label; STATE mapping is STATE_STAGE (see I)
  "produced_by": {"prompt_version": "human_approval.v1", "provider": null, "model": null},
  "data": {
    "approval_id": "ha_<sha1(run_id:target_ref)[:12]>",
    "target_ref": "pp_...",
    "decision": "APPROVED",              // PENDING | APPROVED | CHANGES_REQUESTED | REJECTED
    "decision_at": "...iso...",
    "approver": {"approver_id": "...", "role": "owner"},
    "reviewed_artifacts": [               // EXACT reviewed version, incl. content digests
      {"role": "policy_qa", "artifact_id": "pr_...", "content_sha256": "...", "schema": "PolicyQAResult"},
      {"role": "video_qa",   "artifact_id": "qa_...", "content_sha256": "...", "schema": "VideoQAResult"},
      {"role": "video_result","artifact_id":"vg_...", "content_sha256": "...", "schema": "VideoGenerationResult"},
      {"role": "production_plan","artifact_id":"pp_...", "content_sha256": "...", "schema": "ProductionPlan"},
      {"role": "script",     "artifact_id": "sc_...",  "content_sha256": "...", "schema": "Script"},
      {"role": "script_qa",  "artifact_id": "sqa_...", "content_sha256": "...", "schema": "ScriptQAResult"},
      {"role": "fact_check", "artifact_id": "fc_...",  "content_sha256": "...", "schema": "FactCheckResult"}
    ],
    "prior_gate_verdicts": {             // relationship to previous QA results
      "policy_qa": "PASS", "video_qa": "PASS", "video_generation": "completed", "script_qa": "PASS"
    },
    "overridden_verdicts": [...],        // recorded when a soft REVIEW is resolved
    "blocking_conditions": [...],        // complete required condition set
    "manual_upload": {"ready": true, "required": true, "artifacts": ["pr_...", "..."], "note": "No automated upload is implemented. Endpoint is a manual upload only.", "title": "..."},
    "notes": "...",                      // reviewer notes
    "changes_requested": [...],          // only when decision == CHANGES_REQUESTED
    "supersedes_approval_id": "",        // re-review chain (mirror resumed_from_* convention)
    "provenance": {"generated_by": "human_approval_gate_v1", "source": "HumanReviewer", ...}
  }
}
```

- **Approval status** lives in `decision`; **state** is derived (PENDING
  default / STALE / SUPERSEDED, see E/G).
- **Approver info**: free-form `approver_id`/`role` (no auth system exists — see
  open decision M8).
- **Timestamp**: `decision_at` (CLI-provided or user-supplied; the record always
  carries one).
- **Reviewed artifact/version**: `reviewed_artifacts[]` with each artifact’s
  **stable id + content digest** (see G/H for why digests are essential).
- **Stable ID**: `ha_<sha1(run_id:target_ref)[:12]>`, deterministic per
  (run, target) exactly like `pr_`/`qa_`/`vg_`. Re-decisions overwrite the slot;
  the prior record id is chained via `supersedes_approval_id`.

## E. Approval state machine

**Implemented:** `PENDING → APPROVED | CHANGES_REQUESTED | REJECTED`, plus
derived/invariant states `STALE` and `SUPERSEDED`.

- `PENDING` — no human action yet (also the fail-closed default: a missing
  approval is never an approval).
- `APPROVED` — the run is manually-upload-ready (subject to F).
- `CHANGES_REQUESTED` — human wants upstream changes; nothing advances.
- `REJECTED` — terminal for the reviewed artifact version; docs say rejected
  videos “cannot proceed by any path” (`POLICY_AND_SAFETY.md` §3).
- Derived `STALE` — reviewed artifact/version changed since the decision (see G).
- Derived `SUPERSEDED` — a later `HumanApprovalResult` chained via
  `supersedes_approval_id` replaced this one.

Why **not** `APPROVED / REVIEW / REJECT`: the pipeline already owns
`PASS/REVIEW/REJECT` (and Stage 5’s `REVISE`) as deterministic *machine*
verdicts. Reuse would conflate a gate verdict with a human decision, make an
“Approve a REVIEW” instruction ambiguous, and muddy audit.
`CHANGES_REQUESTED` is also deliberately distinct from Stage-5 `REVISE`.
Human-decision vocabulary must be disjoint from QA vocabulary.

## F. Hard blocking rules

| Input | Auto gate | Human may at Stage 10 |
|---|---|---|
| Stage 8 PASS / Stage 9 PASS | approvable | APPROVED / CHANGES_REQUESTED / REJECTED |
| Stage 8 REVIEW (soft ERROR) | approvable but flagged | may override **only if no hard ERROR exists**; override must be recorded |
| Stage 9 REVIEW (soft ERROR, incl. P15/P13 surface + stricter classifier) | approvable but flagged | may resolve — this is exactly what `REVIEW = “block until human decides”` means |
| Stage 8 REJECT (hard ERROR) | **blocking** | cannot APPROVE; only CHANGES_REQUESTED/REJECTED |
| Stage 9 REJECT (hard ERROR: P01/P02/P05/P14 or any hard-gate issue) | **blocking** | cannot APPROVE — “Rejected videos cannot proceed by any path” |
| Missing / mismatched / malformed upstream artifacts | **blocking** | cannot APPROVE (fail-closed; missing P13/P15-grade inputs) |

**May approval ever bypass a failed deterministic safety/policy gate?** No for
any hard-gate failure. Soft REVIEW may be resolved by the human because the docs
*design it that way* (`POLICY_AND_SAFETY.md` §3 “Ambiguous; needs a human
decision”). The implemented rule is strict: `APPROVED` requires
`PolicyQAResult.overall_verdict == "PASS"` **or** a recorded, justified
`overridden_verdicts[]` entry for every concrete soft finding, with no
hard-gate ERROR present.

**Always-blocking regardless of human approval:** Stage-9 hard REJECT; Stage-8
hard REJECT; missing/dangling/mismatched artifacts; unsatisfied hard-gate
conditions. These must remain monotone (gates only downgrade), which also
preserves the project’s determinism invariants.

## G. Artifact/version invalidation rules

**Key finding (must be addressed):** every downstream id is *content-
independent* by construction — `sc_<sha1(run_id:topic_id)[:12]>`,
`pp_<sha1(run_id:script_id)[:12]>`, `vg_/qa_/pr_<sha1(run_id:key)[:12]>`.
Editing script/plan text **does not change any id**, so binding an approval to
ids alone cannot detect a changed package. Therefore:

- Stage 10 must bind to **content digests** (`content_sha256` of each reviewed
  artifact’s canonical JSON) in addition to ids.
- **Invalidation rule:** an approval is valid iff every recorded
  `(artifact_id, content_sha256)` still matches the corresponding current file.
  Any change → `STALE`.
- Changing any upstream artifact (re-run of ANY upstream stage) invalidates
  downstream approvals for that target (approval is not transitive-resilient).
- **Re-review:** a new decision on (possibly changed) artifacts overwrites the
  slot and chains `supersedes_approval_id`.
- **Interrupted runs:** fail-closed. No written `HumanApprovalResult` = PENDING;
  the CLI re-presents current artifacts and the human re-decides. No state
  beyond `pipeline/state.py` is required.

## H. Audit/provenance model

Extends the existing chain:

```
sentence → claim → source
script → production plan → generated artifacts → QA (video/policy)
human approval → exact reviewed artifact/version
```

Stage 10 records, for audit:

- the full `reviewed_artifacts[]` set (role + stable id + `content_sha256`),
- `prior_gate_verdicts` and any `overridden_verdicts` (which REVIEW findings
  were consciously accepted),
- `blocking_conditions` state, `approver`, `decision_at`, `notes`,
- provenance (`generated_by="human_approval_gate_v1"`,
  `source="HumanReviewer"`, version stamps).

This makes “what exactly did the human see/approve” independently re-checkable —
same philosophy as P01–P15’s “independently re-checkable” evidence.

## I. Pipeline/state implications

**Implemented:** no new `Stage` enum member. An `APPROVED` decision advances
pipeline bookkeeping to the existing `Stage.PUBLISH`; `CHANGES_REQUESTED` and
`REJECTED` remain under `Stage.QUALITY_CHECK`. The artifact header retains the
task label `stage="human_approval"`. This never triggers uploads — `state.json`
is bookkeeping only.


## J. Implemented package/file structure

The package follows the repository's `agents/<stage>/` pattern:

```
agents/human_approval/
├── __init__.py
├── models.py        HumanApprovalResult, Approver, ReviewedArtifact,
│                    ApprovalInputs, HumanApprovalConfig (frozen, versioned),
│                    HumanApprovalError
├── validator.py     deterministic: gate application, blocking checks,
│                    digest-staleness, decision validation
├── stage.py         run_human_approval(...), loaders, pairing/digest guards
├── __main__.py      CLI: writes data/<run_id>/approval/human_approval_result.json
│                    + decision-aware state.json update
tests/
├── test_human_approval_models.py / _validator.py / _stage.py / _safety.py
└── human_approval_test_utils.py
```

- Storage: `data/<run_id>/approval/human_approval_result.json`.
- CLI exit codes: 0 = APPROVED, 1 = CHANGES_REQUESTED/REJECTED (artifact still
  written), 2 = error — matching the established Stage 9 convention.
- `produced_by.model = null`; the “producer” is a human, never a
  model/provider — consistent with deterministic stages.

## K. Implemented test coverage (all deterministic; human input is injected, not simulated UI)

The implementation follows this suite shape (mirrors existing stage-test
patterns, e.g.
`tests/test_policy_qa_stage.py`):

1. **Approval** — clean PASS bundle → valid `HumanApprovalResult`,
   deterministic `ha_…` id, digest matches, round-trip JSON, CLI exit 0.
2. **Changes requested** — decision `CHANGES_REQUESTED` requires
   `changes_requested`/`notes`; never `APPROVED`.
3. **Rejection** — `REJECTED` written and terminal for that artifact version.
4. **Missing QA results** — empty/missing policy/QA input → fail-closed block,
   no approvals, error path.
5. **Failed upstream gates** — Stage-9 or Stage-8 hard REJECT → attempt
   `APPROVED` rejected (validator blocks), CHANGES_REQUESTED allowed.
6. **Artifact/version mismatch** — tampered artifact (content differs, id same)
   → digest check fails.
7. **Stale approval** — approve, then mutate upstream artifact → validity check
   reports `STALE`, never `APPROVED`.
8. **Resume/re-review** — `supersedes_approval_id` chain; slot overwrite;
   interrupted run leaves `PENDING` with no artifact.
9. **Provenance/audit chain** — all reviewed ids + digests + prior verdicts +
   overrides recorded; envelope self-describing.
10. **Safety invariants** — static package scan: no network/GPU/secret modules,
    no Stage 11 vocabulary (oauth/upload/scheduling/analytics/youtube-api);
    tests prove the stage runs with sockets blocked and never writes outside its
    artifact (mirror `test_policy_qa_safety.py` / `test_video_generation_safety.py`).

## L. Documentation changes completed

- `docs/PROJECT_PROGRESS.md` — Stage 10 status, package, state mapping, and
  verification notes.
- `docs/DATA_CONTRACTS.md` — implemented §12 contract, envelope, roles, gates,
  storage, and CLI semantics.
- `docs/POLICY_AND_SAFETY.md` — updated gate, override, freshness, and manual
  upload safety semantics.
- This design document — implementation status, package layout, and resolved
  decisions.

## M. Resolved implementation decisions

1. **State mapping:** approved records advance to `Stage.PUBLISH`; non-approved
   decisions remain at `Stage.QUALITY_CHECK`; no new enum member is added.
2. **Package placement:** `agents/human_approval/`.
3. **Approval with soft REVIEW:** allowed only with a concrete, version-bound
   override; hard REJECT is never overrideable.
4. **Human REJECTED:** terminal for the reviewed artifact version.
5. **Version binding:** stable id plus canonical-content SHA-256.
6. **Manual-upload-ready package:** a manifest/readiness section containing only
   references and the title; no upload code.
7. **CHANGES_REQUESTED structure:** structured requirements with optional issue
   references, or a non-empty note.
8. **Approver identity:** explicit free-form `approver_id` and optional role;
   no authentication system is implied.
9. **Artifact location:** `data/<run_id>/approval/`.
10. **Freshness re-check:** current artifact ids and digests are verified before
    an approval is considered effective.

## N. Explicit statement

**Stage 10 — Human Approval IS IMPLEMENTED.** `agents/human_approval/`
(`models.py`, `validator.py`, `stage.py`, `__main__.py`, `__init__.py`) contains
the deterministic gate, strict artifact loaders, freshness checks, and CLI.
The focused Stage 10 suite covers the implemented contract; Stage 11 (YouTube
upload) remains out of scope.

---

## Final checks

- **Focused Stage 10 tests:** **141 passed** with
  `python -m pytest -q tests/test_human_approval_models.py tests/test_human_approval_validator.py tests/test_human_approval_stage.py tests/test_human_approval_safety.py`.
- **Full suite:** **1257 passed, 5 skipped** (`python -m pytest tests -q`).
- **Compilation:** `python -m compileall -q agents tests` passed.
- **Git:** the Stage 10 implementation is committed at
  `80c9841134d9db3c623bf1b909c4f2752e676980`
  (`feat: complete human approval stage`); this documentation-only checkpoint
  is intentionally uncommitted and changes no source or test files.
