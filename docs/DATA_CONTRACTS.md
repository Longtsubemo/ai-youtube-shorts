# Data Contracts

Structured JSON contracts that stages exchange — the backbone of the
file-based data contracts architecture (`ARCHITECTURE.md` §4.1).

Schemas are intentionally **minimal**: enough to implement v1 cleanly,
not a premature enterprise model. They grow with the stages.

## 1. Common envelope and versioning

Every artifact is a JSON object with a contract header:

```json
{
  "schema": "ResearchPackage",
  "schema_version": 1,
  "created_at": "2026-09-20T08:00:00+00:00",
  "run_id": "1c34504001b0",
  "stage": "research",
  "produced_by": {
    "prompt_version": "research/facts-v1",
    "provider": null,
    "model": null
  },
  "data": { "...contract-specific fields..." }
}
```

- `schema_version` integer. Bump on **breaking** changes only; additive
  fields do not bump.
- `produced_by` records model/provider/prompt version where applicable;
  `null` when not applicable (deterministic stages).
- `stage` ranges over `pipeline/state.py` `Stage` enum values.

Examples below show the `data` portion only (header elided).

## 2. TopicCandidate

Signal-scored candidate topic from discovery (`TOPIC_SELECTION.md`).

```json
{
  "candidate_id": "tc_01J..",
  "topic": "How CRISPR bypasses the immune system",
  "pillar": "science",
  "topic_type": "evergreen",
  "summary": "One-sentence description.",
  "signals": {
    "search_interest": 0.61,
    "trend": 0.0,
    "competition": 0.42,
    "evergreen_potential": 0.9,
    "curiosity": 0.8,
    "explainer_fit": 0.9,
    "visual_potential": 0.75,
    "audience_fit": 0.8,
    "freshness": null
  },
  "source_ids": ["src_01J.."],
  "discovered_at": "2026-09-20T08:00:00+00:00"
}
```

- `signals` are 0–1 per key; keys may grow. `null` = not measured.
- `topic_type` distinguishes evergreen/trending/recent/historical/etc.

## 3. TopicSelection

The per-run decision (`TOPIC_SELECTION.md`).

```json
{
  "selection_id": "ts_01K..",
  "run_id": "1c34504001b0",
  "candidate_id": "tc_01J..",
  "mode": "explore",
  "composite_score": 0.71,
  "scores": {
    "signal_weighted": 0.78,
    "historical_fit": null,
    "coverage_bonus": 0.05
  },
  "rationale": "High explainer fit, low pillar recency; exploration mode.",
  "selected_at": "2026-09-20T08:00:00+00:00"
}
```

- `mode` in `explore` / `exploit` / `mixed` — records the
  exploration/exploitation policy actually used.
- `historical_fit` is `null` during cold start.

## 4. Source

`RESEARCH_PIPELINE.md` §4.

```json
{
  "source_id": "src_01J..",
  "title": "Off-target effects of CRISPR-Cas9",
  "url": "https://example.org/paper",
  "publisher": "Nature",
  "publication_date": "2018-06-19",
  "source_type": "peer_reviewed_paper",
  "hierarchy": "primary",
  "retrieval_date": "2026-09-20T00:00:00+00:00"
}
```

- `hierarchy` in `primary` / `high_quality_secondary` / `discovery`.
- `source_type` is the document kind (paper, preprint, press release,
  news article, encyclopedia, official statement, ...).

## 5. FactClaim

`RESEARCH_PIPELINE.md` §3.

```json
{
  "claim_id": "cl_01J..",
  "claim_text": "CRISPR editing can occasionally alter DNA at off-target sites.",
  "claim_type": "fact",
  "source_ids": ["src_01J..", "src_02K.."],
  "verification_status": "multiple_sources",
  "confidence": 0.9,
  "critical": true,
  "dates": {
    "event": "2018-06-19",
    "announcement": null,
    "discovery": null,
    "publication": "2018-06-19"
  }
}
```

- `claim_type` in `fact` / `inference` / `explanation` / `speculation` /
  `opinion`.
- `verification_status` is provisional in research; settled by
  `FactCheckResult`.
- `critical` = the script argument depends on it.

## 6. ResearchPackage

Set of claims + sources for one topic, submitted to fact check.

```json
{
  "package_id": "rp_01J..",
  "run_id": "1c34504001b0",
  "topic": "How CRISPR bypasses the immune system",
  "framing": "Explain the adaptivity of CRISPR systems in 45 seconds.",
  "claims": ["cl_01J..", "cl_02K.."],
  "sources": ["src_01J..", "src_02K..", "src_03L.."],
  "status": "pending_review"
}
```

- `status` in `pending_review` / `pending_remediation` / `fact_checked`
  / `approved` / `rejected`.
- Claims/sources referenced by ID; full objects live in the same run
  workspace (`data/<run>/research/`).

## 7. FactCheckResult

`FACT_VERIFICATION.md`.

```json
{
  "result_id": "fc_01J..",
  "run_id": "1c34504001b0",
  "package_id": "rp_01J..",
  "verdict": "PASS",
  "per_claim": {
    "cl_01J..": {"outcome": "PASS", "note": "2 independent primary/secondary sources; no contradiction."},
    "cl_02K..": {"outcome": "REVIEW", "note": "Only discovery sources found."}
  },
  "blocked": ["cl_02K.."],
  "critical_rejected": [],
  "checked_at": "2026-09-20T08:05:00+00:00"
}
```

- `verdict` in `PASS` / `REVIEW` / `REJECT`. `outcome` per claim in
  `PASS` / `REVIEW` / `REJECT`.
- A critical claim in `blocked` forces `REJECT`.

## 8. Script

Line-level traceability: every line references the claims it relies on.

```json
{
  "script_id": "sc_01J..",
  "run_id": "1c34504001b0",
  "title": "Why CRISPR edits the right genes, mostly",
  "hook": "Bacteria keep a mugshot of every virus they ever fought.",
  "lines": [
    {
      "line_id": "ln_01",
      "text": "Bacteria keep mugshots of past viruses in their DNA.",
      "claim_refs": ["cl_01J.."],
      "duration_ms": 3200
    }
  ],
  "estimated_duration_ms": 42000,
  "format_constraints": "30-60s vertical short"
}
```

- `lines` are chronological narration. `claim_refs` enable the audit
  chain: script sentence -> claim_id -> research claim -> source_id ->
  original source.
- `duration_ms` estimates inform scene planning.

## 9. ScenePlan

Visualization of the script into scenes (`VIDEO_PIPELINE.md`).

```json
{
  "scene_plan_id": "sp_01J..",
  "run_id": "1c34504001b0",
  "script_id": "sc_01J..",
  "scenes": [
    {
      "scene_id": "scn_01",
      "index": 1,
      "start_ms": 0,
      "duration_ms": 3200,
      "visual_description": "Microscope zoom on a bacterium with DNA loops.",
      "narration_line_ids": ["ln_01"],
      "subtitle_text": "Bacteria keep mugshots of past viruses.",
      "transition_out": "dissolve"
    }
  ]
}
```

## 10. ProductionPlan (production.json)

The stable content-to-video contract (`VIDEO_PIPELINE.md` §3).

```json
{
  "plan_id": "pp_01J..",
  "run_id": "1c34504001b0",
  "video": {
    "orientation": "vertical_9_16",
    "target_duration_s": 45,
    "fps": 30,
    "resolution": "1080x1920"
  },
  "scenes": [
    {
      "scene_id": "scn_01",
      "index": 1,
      "start_ms": 0,
      "duration_ms": 3200,
      "visual": {
        "description": "Microscope zoom on a bacterium with DNA loops.",
        "assets": ["asset_ref:1"]
      },
      "narration": "Bacteria keep mugshots of past viruses in their DNA.",
      "subtitle": "Bacteria keep mugshots of past viruses."
    }
  ],
  "audio": {
    "voice": "calm, neutral, unambiguous",
    "language": "en"
  },
  "subtitles": {"enabled": true, "style": "bold, lower-third, white"},
  "output": {"filename": "short_final.mp4", "container": "mp4"}
}
```

- Provider-agnostic: everything is a description; no model/provider/API
  details leak in.
- `output` is the worker's deliverable contract.

## 11. QAResult

Used by any QA stage (`script` / `video` / `policy`).

```json
{
  "qa_result_id": "qa_01J..",
  "run_id": "1c34504001b0",
  "stage": "policy",
  "target_ref": "pp_01J..",
  "verdict": "REVIEW",
  "checks": [
    {"check": "misleading_claims", "outcome": "PASS", "note": "All claims type-consistent."},
    {"check": "synthetic_media_disclosure", "outcome": "REVIEW", "note": "Features realistic scene; label required."}
  ],
  "issues": [
    {"severity": "REVIEW", "message": "Add synthetic-media disclosure.", "refs": []}
  ],
  "checked_at": "2026-09-20T08:30:00+00:00"
}
```

- Same shape for script/video/policy QA; distinguished by `stage`.

## 12. AnalyticsRecord

`ANALYTICS.md`. Collected after publishing; feeds topic selection.

```json
{
  "record_id": "an_01J..",
  "run_id": "1c34504001b0",
  "video_id": "yt_VIDEO_ID",
  "published_at": "2026-10-01T20:00:00+00:00",
  "collected_at": "2026-10-08T20:00:00+00:00",
  "metrics": {
    "views": 1234,
    "likes": 89,
    "comments": 12,
    "shares": 34,
    "subscribers_gained": 11,
    "avg_retention_pct": 0.62,
    "other": {}
  }
}
```

## 13. Versioning and extension policy

- Additive field additions: no version bump.
- Renames, removals, or semantic changes of existing fields: bump
  `schema_version` and document migration in the release notes.
- Consumers must reject unknown major versions; tolerate unknown extra
  fields.
- Every artifact keeps the header (§1) so any file is self-describing
  and auditable without external lookup.

## 14. Storage layout

```
data/<run_id>/
├── state.json          run status + artifact registry (pipeline/state.py)
├── research/           TopicCandidate, TopicSelection, Source, FactClaim,
│                       ResearchPackage, FactCheckResult
├── scripts/            Script, QAResult(script)
├── production/         ScenePlan, ProductionPlan (production.json), QAResult(video), QAResult(policy)
└── videos/             generated assets + final Short
```

## 15. Open questions

- Exact ID formats (prefix + short id). Current sketch:
  `tc_/ts_/src_/cl_/rp_/fc_/sc_/sp_/pp_/qa_/an_`.
- Whether QA results for all stages share one file each per run or one
  QAResult registry file with stage records.