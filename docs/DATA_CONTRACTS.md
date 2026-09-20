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

Signal-scored candidate topic from discovery (`TOPIC_SELECTION.md`). It does
not ship standalone: candidates are embedded in the `TopicSelection` artifact
(`data.ranked_candidates`, and the winner at `data.selected_topic`).

```json
{
  "topic_id": "t_octopus_hearts",
  "title": "Why octopuses have three hearts",
  "description": "One-sentence description.",
  "pillar": "science",
  "topic_type": "surprising_fact",
  "signals": {
    "interest": 0.75,
    "trend_momentum": 0.35,
    "competition": 0.3,
    "freshness": 0.1
  },
  "content_scores": {
    "curiosity": 0.9,
    "explainer_potential": 0.75,
    "visual_potential": 0.8,
    "audience_fit": 0.8
  },
  "historical_fit": null,
  "total_score": 0.72,
  "provenance": {
    "generated_by": "mock_catalog_v1",
    "source": "internal_mock",
    "created_at": "2026-09-20T08:00:00+00:00",
    "provider": null,
    "model": null,
    "prompt_version": null
  }
}
```

- `signals` (market/interest feel) and `content_scores` (content fit) are
  0–1 per key; keys may grow. `null` = not measured.
- `historical_fit` is `null` during cold start; `total_score` is `null`
  until the scorer runs.
- `topic_id` is a stable typed slug (`t_<slug>`); `topic_type` distinguishes
  evergreen/trending/recent_discovery/historical/surprising_fact/etc.
- `provenance` mirrors the common envelope header (§1) as a nested object.

## 3. TopicSelection

The per-run decision artifact (`TOPIC_SELECTION.md`). Full envelope + `data`:

```json
{
  "schema": "TopicSelection",
  "schema_version": 1,
  "created_at": "2026-09-20T08:00:00+00:00",
  "run_id": "1c34504001b0",
  "stage": "topic_selection",
  "produced_by": {"prompt_version": null, "provider": null, "model": null},
  "data": {
    "selection_id": "ts_a4ad44ea7fa9",
    "selected_topic_id": "t_octopus_hearts",
    "selected_topic": {
      "topic_id": "t_octopus_hearts",
      "title": "Why octopuses have three hearts",
      "description": "One-sentence description.",
      "pillar": "science",
      "topic_type": "surprising_fact",
      "signals": {"interest": 0.75, "trend_momentum": 0.35, "competition": 0.3, "freshness": 0.1},
      "content_scores": {"curiosity": 0.9, "explainer_potential": 0.75, "visual_potential": 0.8, "audience_fit": 0.8},
      "historical_fit": null,
      "total_score": 0.72,
      "provenance": {"generated_by": "mock_catalog_v1", "source": "internal_mock", "created_at": "2026-09-20T08:00:00+00:00", "provider": null, "model": null, "prompt_version": null}
    },
    "ranked_candidates": [ {"...": "TopicCandidate objects, descending score (§2)"} ],
    "selection_mode": "cold_start",
    "historical_data_available": false,
    "selection_reason": "cold_start: no historical data; selected 'Why octopuses have three hearts' (total_score=0.7150)",
    "scoring_configuration": {
      "version": 1,
      "scoring": {
        "weights": {"interest": 0.2, "trend_momentum": 0.15, "competition": 0.15, "curiosity": 0.15, "explainer_potential": 0.15, "visual_potential": 0.1, "audience_fit": 0.1},
        "competition_transformation": "inverse",
        "historical_enabled": true,
        "historical_weight": 0.25
      },
      "selection": {"exploration_fraction": 0.3, "seed": 42, "top_n": 3, "coverage_penalty": 0.25},
      "pillars": ["ai", "technology", "science", "space", "curiosity"],
      "topic_types": ["evergreen", "trending", "recent_discovery", "historical", "surprising_fact", "technology_development", "scientific_concept", "unusual_phenomenon"]
    }
  }
}
```

- `selection_mode` in `cold_start` / `exploration` / `exploitation` —
  records the exploration/exploitation policy actually used; `cold_start`
  when no historical data exists.
- `historical_data_available` is a boolean; candidates keep
  `historical_fit = null` during cold start.
- `selected_topic` is the full `TopicCandidate` (§2) that won;
  `ranked_candidates` are all scored candidates, descending.
- `scoring_configuration` copies the config actually used
  (`config/topic_selection.yaml`), so the run is reproducible without
  external lookup.

## 4. Source

`RESEARCH_PIPELINE.md` §4.

```json
{
  "source_id": "src_34ce4eb949",
  "title": "Off-target effects of CRISPR-Cas9",
  "url": "https://example.test/paper/crispr-off-target",
  "publisher": "Nature",
  "source_type": "peer_reviewed_paper",
  "publication_date": "2018-06-19",
  "credibility_tier": "primary",
  "retrieved_at": "2026-09-20T08:00:00+00:00",
  "notes": "MOCK FIXTURE - not a real source."
}
```

- `credibility_tier` in `primary` / `high_quality_secondary` / `discovery`
  (`RESEARCH_PIPELINE.md` §4).
- `source_type` is the document kind (paper, preprint, press release,
  news article, encyclopedia, official statement, ...).
- `publication_date` is when the source was published; `retrieved_at` is
  when *we* retrieved it. Both accept ISO dates/datetimes and are empty
  strings when unset.
- Required fields: `source_id`, `title`, `url`, `publisher`, `source_type`,
  `publication_date`, `credibility_tier`. `retrieved_at` and `notes` are
  optional (default empty).
- `source_id` is stable and deterministic: `src_<sha1(url)[:10]>`, shared
  by claims that cite the source.

## 5. FactClaim

`RESEARCH_PIPELINE.md` §3.

```json
{
  "claim_id": "cl_04e08cd0cb",
  "claim_text": "CRISPR editing can occasionally alter DNA at off-target sites.",
  "claim_type": "fact",
  "source_ids": ["src_34ce4eb949", "src_b6b2f2a64a"],
  "confidence": 0.9,
  "verification_status": "provisional",
  "dates": [
    {"date_kind": "publication_date", "date": "2018-06-19"},
    {"date_kind": "discovery_date", "date": "2012-10-01"}
  ],
  "notes": "Corroborated by a primary and a high-quality secondary source.",
  "critical": true
}
```

- `claim_type` in `fact` / `inference` / `explanation` / `speculation` /
  `opinion`.
- `verification_status` is **provisional at the research stage**: the
  allowed set is `provisional` (the default) / `pending` / `single_source`
  / `multiple_sources` / `disputed` / `unsupported`. Research must never
  label a claim as fully verified; settlement is exclusive to the fact check
  stage (`FACT_VERIFICATION.md`, `FactCheckResult` §7).
- `confidence` is a heuristic 0–1 derived from backing source tiers
  (`RESEARCH_PIPELINE.md`), **not** a fact-check result.
- `dates` is a list of zero-or-more `{date_kind, date}` entries; kinds are
  `publication_date` / `discovery_date` / `announcement_date` / `event_date`
  (`RESEARCH_PIPELINE.md` §5). A source's `publication_date` is never
  silently promoted to a claim date kind.
- `critical` = the script argument depends on it. `notes` is free text.
- `claim_id` is stable and deterministic: `cl_<sha1(text)[:10]>`, shared by
  the sources that back it (`source_ids`).

## 6. ResearchPackage

The research stage's artifact: evidence-backed claims and their sources for
one selected topic, plus provenance and status. Full envelope + `data`:

```json
{
  "schema": "ResearchPackage",
  "schema_version": 1,
  "created_at": "2026-09-20T08:00:00+00:00",
  "run_id": "1c34504001b0",
  "stage": "research",
  "produced_by": {"prompt_version": null, "provider": null, "model": null},
  "data": {
    "package_id": "rp_6003faf0a58c",
    "topic_id": "t_octopus_hearts",
    "topic_title": "Why octopuses have three hearts",
    "research_status": "ready_for_fact_check",
    "claims": [
      {
        "claim_id": "cl_04e08cd0cb",
        "claim_text": "Octopuses have three hearts: two branchial and one systemic.",
        "claim_type": "fact",
        "source_ids": ["src_34ce4eb949"],
        "confidence": 0.85,
        "verification_status": "provisional",
        "dates": [{"date_kind": "publication_date", "date": "2020-05-12"}],
        "notes": "",
        "critical": false
      }
    ],
    "sources": [
      {
        "source_id": "src_34ce4eb949",
        "title": "The cardiovascular system of the octopus",
        "url": "https://example.test/papers/octopus-hearts",
        "publisher": "Journal of Cephalopod Biology",
        "source_type": "peer_reviewed_paper",
        "publication_date": "2020-05-12",
        "credibility_tier": "primary",
        "retrieved_at": "2026-09-20T08:00:00+00:00",
        "notes": ""
      }
    ],
    "research_notes": [
      "Research evidence gathered from local/mocked sources only.",
      "Claims carry provisional verification only; final verification is deferred to the fact-check stage."
    ],
    "provenance": {
      "generated_by": "research_agent_v1",
      "source": "MockResearchProvider",
      "created_at": "2026-09-20T08:00:00+00:00",
      "provider": null,
      "model": null,
      "prompt_version": null
    }
  }
}
```

- `research_status` in `provisional` / `insufficient_evidence` /
  `ready_for_fact_check`.
  - `ready_for_fact_check` — core (non-discovery) evidence backs at least
    one fact/explanation/inference.
  - `provisional` — discovery-tier leads only.
  - `insufficient_evidence` — no usable claims and/or sources found.
- `claims` and `sources` are embedded **full objects** (not just IDs).
  Every claim's `source_ids` must resolve to a source in `sources`; the
  package validates this and rejects dangling references (this is the basis
  of the §8 script audit chain).
- `research_notes` are human/audit-readable strings; `provenance` records
  which agent/provider produced the package.
- `package_id` is `rp_<uuid[:12]>`. The package and all research artifacts
  live at `data/<run_id>/research/`.

## 7. FactCheckResult

The fact-check stage's artifact (`FACT_VERIFICATION.md`, Stage 3). It evaluates
every claim against **supplied evidence only** and issues an overall verdict.
Full envelope + `data`:

```json
{
  "schema": "FactCheckResult",
  "schema_version": 1,
  "created_at": "2026-09-20T08:05:00+00:00",
  "run_id": "1c34504001b0",
  "stage": "fact_check",
  "produced_by": {"prompt_version": null, "provider": "MockEvidenceProvider", "model": "MockVerifier"},
  "data": {
    "result_id": "fc_aa4ad44ea7fa9",
    "topic_id": "t_octopus_hearts",
    "topic_title": "Why octopuses have three hearts",
    "package_id": "rp_6003faf0a58c",
    "package_verdict": "PASS",
    "claim_results": [
      {
        "claim_id": "cl_04e08cd0cb",
        "verdict": "PASS",
        "evidence_ids": ["ev_56b649a6e9", "ev_9155cf348b"],
        "supporting_source_ids": ["src_34ce4eb949"],
        "contradiction_source_ids": [],
        "rationale": "evidence examined: ev_56b649a6e9, ev_9155cf348b; supporting sources: src_34ce4eb949; contradicting sources: none",
        "confidence": 0.85,
        "critical": false,
        "checks_performed": ["stance_overall=direct", "contradiction_present=no"],
        "verifier": "MockVerifier"
      },
      {
        "claim_id": "cl_02K..",
        "verdict": "REVIEW",
        "evidence_ids": ["ev_2392068451"],
        "supporting_source_ids": [],
        "contradiction_source_ids": ["src_21df285abb"],
        "rationale": "credible contradictory evidence present",
        "confidence": 0.3,
        "critical": false,
        "checks_performed": ["stance_overall=contradictory", "contradiction_present=yes"],
        "verifier": "MockVerifier"
      }
    ],
    "critical_claim_failures": [],
    "review_items": ["cl_02K.."],
    "rejected_claims": [],
    "approved_claims": ["cl_04e08cd0cb"],
    "evidence_issues": [],
    "provenance": {
      "generated_by": "fact_check_agent_v1",
      "source": "MockEvidenceProvider:MockVerifier",
      "created_at": "2026-09-20T08:05:00+00:00",
      "provider": null,
      "model": "MockVerifier",
      "prompt_version": null
    }
  }
}
```

- `package_verdict` and per-claim `verdict` in `PASS` / `REVIEW` / `REJECT`.
  Verdict rank: `PASS < REVIEW < REJECT`; gates only make a verdict worse,
  never upgrade REVIEW/REJECT to PASS.
- `approved_claims` is **exactly** what Stage 4 may consume — rejected and
  REVIEW claims are excluded and never reach the script stage.
- `critical_claim_failures` = critical claims that did not pass; any critical
  `REVIEW` forces package `REVIEW`, any critical `REJECT` forces package
  `REJECT`. Non-critical `REJECT`s may be dropped from the script instead.
- Every `PASS` is traceable: `evidence_ids` must be non-empty and each
  evidence's `source_id` resolves to a `Source` inside the package (dangling
  evidence is dropped and recorded in `evidence_issues`).
- `evidence_issues` records provider failures, malformed evidence, and
  evidence referencing unknown sources — the checker never crashes on bad
  input.
- `result_id` is stable/deterministic: `fc_<sha1(run_id:package_id)[:12]>`.
- `produced_by`/`provenance` record which provider and verifier ran; with a
  real LLM verifier it will carry the model + prompt version.

### Evidence (provider sub-contract)

`FactCheckResult` references `Evidence` by `evidence_id`; the full evidence
list is provided by an `EvidenceProvider` (mock fixture
`data/mock/fact_check_evidence.json`, keyed by source URL). Each evidence:

```json
{
  "evidence_id": "ev_56b649a6e9",
  "source_id": "src_34ce4eb949",
  "excerpt": "Octopuses have three hearts: two branchial and one systemic.",
  "locator": "section: cardiac anatomy",
  "evidence_type": "direct_statement",
  "retrieved_at": "2026-09-20T08:00:00+00:00",
  "notes": "MOCK"
}
```

- `evidence_type` in `direct_statement` / `supporting_context` /
  `contradictory_statement` / `insufficient`. A source URL alone is never
  evidence; only retrieved content (`excerpt`) counts.
- `evidence_id` is stable/deterministic:
  `ev_<sha1(source_id:excerpt)[:10]>`.
- `confidence` per claim: tier base (primary 0.85 / high-quality secondary
  0.6 / discovery 0.3), +0.05 per extra independent non-discovery source
  (cap 0.95), REVIEW capped at 0.5, REJECT capped at 0.3.

## 8. Script

**Implemented** (`agents/script/`). Stage 4 turns a `FactCheckResult` into a
30–60s short-form explainer script. Its only factual input is the fact-check
result's `approved_claims` (joined against the research package) — rejected,
REVIEW, and unlisted claims can never appear as fact (`FACT_VERIFICATION.md`,
"consumption boundary").

Envelope follows §1. Example (`data/<run_id>/scripts/script.json`):

```json
{
  "schema": "Script",
  "schema_version": 1,
  "created_at": "2026-09-20T16:31:21+00:00",
  "run_id": "sc-demo",
  "stage": "script_generation",
  "produced_by": { "prompt_version": null, "provider": null, "model": "MockScriptGenerator" },
  "data": {
    "script_id": "sc_41d39e3980db",
    "topic_id": "t_octopus_hearts",
    "topic_title": "Why octopuses have three hearts",
    "title": "Why octopuses have three hearts",
    "hook": "You probably didn't know this, but Octopuses have three hearts",
    "estimated_duration_seconds": 45.2,
    "used_claim_ids": ["cl_97628e259b", "cl_b4ca50a10d", "cl_f6089873c8"],
    "unused_approved_claim_ids": [],
    "result_id": "fc_41e052643b3a",
    "package_id": "rp_9fc41d7a46b7",
    "lines": [
      {
        "line_id": "ln_00",
        "text": "You probably didn't know this, but Octopuses have three hearts",
        "claim_refs": ["cl_b4ca50a10d"],
        "purpose": "hook",
        "estimated_duration_seconds": 4.0
      },
      {
        "line_id": "ln_02",
        "text": "Octopuses have three hearts: two branchial hearts that push blood through the gills and a systemic heart that powers the body.",
        "claim_refs": ["cl_b4ca50a10d"],
        "purpose": "fact",
        "estimated_duration_seconds": 8.4
      }
    ],
    "provenance": {
      "generated_by": "script_agent_v1",
      "source": "MockScriptGenerator",
      "created_at": "2026-09-20T16:31:21+00:00",
      "provider": null,
      "model": "MockScriptGenerator",
      "prompt_version": null
    }
  }
}
```

- `lines` are chronological narration. `purpose` in `hook` / `context` /
  `explanation` / `fact` / `transition` / `conclusion` / `call_to_action`.
- `claim_refs` enable the audit chain:
  `script line -> claim_id -> research claim -> source_id -> original source`.
- **Factual lines** (`fact`, `explanation`, `conclusion`) MUST carry
  `claim_refs` pointing at approved claims only; framing lines
  (`hook`/`context`/`transition`/`call_to_action`) are boilerplate and carry
  none.
- `used_claim_ids` + `unused_approved_claim_ids` split the result's
  `approved_claims` exactly (disjoint, together equal `approved_claims`), so
  reviewers can see what was consumed and what was left out.
- `estimated_duration_seconds` = Σ line word counts ÷ `words_per_minute` × 60
  (default 150 wpm), target 30–60s.
- IDs are deterministic: `sc_<sha1(run_id:topic_id)[:12]>`, lines
  `ln_00, ln_01, ...`.
- Validation (see `validate_script`) additionally enforces: no
  REVIEW/REJECT/unapproved refs, no referencing claims missing from the
  result, token-coverage (factual lines stay inside their claims' wording),
  preserved hedges, no unsupported certainty strengthening, duration bounds.

CLI: `python -m agents.script.script_generator` (writes
`data/<run_id>/scripts/script.json` + run state). Gate/failure behavior:
package `REJECT` → `blocked`; package `REVIEW` → `blocked` unless
`allow_review_content`; too few approved claims or too-short material →
structured `insufficient_content` result (never inventing filler).

## 8.1 ScriptQAResult (Script QA)

**Implemented** (`agents/qa/script/`). Stage 5 is quality control over the
Stage-4 `Script` artifact — it is NOT another research/fact-finding stage. Its
only factual input is (a) the `Script`, (b) the `FactCheckResult`, and (c) the
**approved claim records** via the same whitelist join as Stage 4
(`approved_claims`). It never re-runs research, never consults unapproved
claims as fact, and never rewrites the script: it emits a verdict plus
actionable issues for a future revision agent.

Example (committed fixture `data/mock/script_qa_result.json`):

```json
{
  "schema": "ScriptQAResult",
  "schema_version": 1,
  "created_at": "2026-09-20T12:00:00.000000+00:00",
  "run_id": "mock",
  "stage": "script_qa",
  "produced_by": { "prompt_version": null, "provider": null, "model": "ScriptQAValidator" },
  "data": {
    "qa_result_id": "sqa_e99181e73b5d",
    "script_id": "sc_9fc41d7a46b7",
    "topic_id": "t_octopus_hearts",
    "result_id": "fc_41e052643b3a",
    "package_id": "rp_9fc41d7a46b7",
    "overall_verdict": "PASS",
    "score": 100.0,
    "score_detail": {
      "score": 100.0,
      "dimensions": { "factual_fidelity": 100.0, "uncertainty": 100.0, "traceability": 100.0,
                      "coverage": 100.0, "clarity": 100.0, "structure": 100.0, "pacing": 100.0,
                      "hook": 100.0, "repetition": 100.0, "format": 100.0 },
      "weights": { "factual_fidelity": 0.2, "uncertainty": 0.15, "traceability": 0.15,
                   "clarity": 0.1, "structure": 0.1, "pacing": 0.1, "hook": 0.05,
                   "coverage": 0.05, "repetition": 0.05, "format": 0.05 }
    },
    "checks": [ { "check": "factual_fidelity", "category": "factual_fidelity", "dimension": "factual_fidelity",
                  "hard_gate": true, "errors": 0, "warnings": 0, "infos": 0, "passed": true } ],
    "issues": [],
    "warnings": [],
    "infos": [],
    "approved_claim_coverage": {
      "approved_claim_ids": ["cl_97628e259b", "cl_b4ca50a10d", "cl_f6089873c8"],
      "used_claim_ids": ["cl_97628e259b", "cl_b4ca50a10d", "cl_f6089873c8"],
      "unused_approved_claim_ids": [],
      "critical_unused_claim_ids": [],
      "coverage_ratio": 1.0
    },
    "provenance": {
      "generated_by": "script_qa_agent_v1",
      "source": "DeterministicScriptQA",
      "created_at": "2026-09-20T12:00:00.000000+00:00",
      "provider": null,
      "model": "ScriptQAValidator",
      "prompt_version": "script_qa.v1"
    }
  }
}
```

- **Verdict:** `REJECT` if any severity-ERROR issue is a HARD gate; `REVISE`
  if any (soft) severity-ERROR issue exists; otherwise `PASS`
  (warnings/INFO only, or fully clean). Warnings and INFO can never alone
  cause failure.
- **13 check categories:** `factual_fidelity`, `claim_traceability`,
  `claim_coverage`, `length_duration`, `pacing`, `clarity`, `repetition`,
  `structure`, `hook_quality`, `uncertainty_preservation`,
  `misleading_language`, `cta_quality`, `format_shorts`. Each issue carries
  `category`, `severity` (`ERROR`/`WARNING`/`INFO`), `gate`
  (`hard`/`soft`), `dimension`, `check`, and affected line/claim ids for the
  audit chain.
- **9 hard gates** (each is a `gate=hard` ERROR → REJECT):
  1. no factual traceability (factual line without claim refs)
  2. references a REJECTED claim
  3. references a REVIEW (unapproved) claim
  4. references an unknown claim id
  5. material contradiction (negation never said by the approved claims)
  6. material certainty strengthening (unsupported certainty word, hard gate)
  7. invented factual detail (token coverage below the 0.5 threshold)
  8. empty / structurally unusable script
  9. package not approved for scripting (package verdict REJECT or REVIEW)
- **Score (internal quality, NOT a performance prediction):**
  `dimension = clamp(100 − penalties, 0, 100)`; ERROR hard −45, ERROR soft −20,
  WARNING −6, INFO −1; `overall = round(weighted mean, 1)`. Default weights
  (sum = 1): factual_fidelity 0.20, uncertainty 0.15, traceability 0.15,
  clarity 0.10, structure 0.10, pacing 0.10, hook 0.05, coverage 0.05,
  repetition 0.05, format 0.05. Configurable via `ScriptQAConfig.validate()`
  (weights must cover exactly the 10 dimensions and sum to 1).
- **Coverage:** `approved_claim_coverage.coverage_ratio` = used ∕ approved.
  Non-critical unused claims are allowed (INFO). A *critical* unused claim is
  an ERROR by default (`critical_unused_is_error`); relaxed to a WARNING when
  configured.
- **Deterministic heuristics worth knowing** (all in `checks.py`):
  - Duration: <15s or >90s → hard ERROR; <30s or >60s → WARNING.
  - Line pacing: >45 words → ERROR (soft gate), >30 → WARNING; >2 sentences
    with >20 words, or ≥5 connectors → WARNING; >90-word uninterrupted
    explanation block → WARNING; >35 words of setup before the first fact →
    WARNING.
  - Clarity: sentence >25 words, unexplained acronym (`[A-Z]{2,6}`), term
    >18 chars, >4 parentheses, or a factual line opening with
    it/they/this/that/these/those/there → WARNING.
  - Repetition: identical or near-duplicate (Jaccard ≥ 0.8, non-adjacent)
    factual statements in two body lines → WARNING. Hook/conclusion
    reinforcement and reusing the opening claim are intentional and only INFO.
  - Uncertainty: hedged claims must keep their hedge; dropping it or adding a
    certainty word neither claim ever says → hard ERROR. `appears`/`seems`
    count as hedges only when followed by `to` (so "blood appears blue" is not
    treated as hedged).
  - Misleading/clickbait phrasing → WARNING (never fatal by itself); CTA
    missing → INFO only; long/duplicated/reference-free hook or CTA →
    WARNING.
  - Format: <3 or >14 lines, a >160-character line, >20 avg words/line, or
    >150 total words → WARNING.
- Deliberately does NOT call `script.validate()`: corrupt artifacts are
  reported as issues instead of raising, so QA can gate a broken input.
- IDs: `sqa_<sha1(run_id:script_id)[:12]>`; issue ids `qa_00, qa_01, ...`
  assigned in deterministic check order.

CLI: `python -m agents.qa.script` (writes
`data/<run_id>/scripts/script_qa_result.json` + run state advanced to
`Stage.SCRIPT_QA`). Config live in `ScriptQAConfig` /
`agents/qa/script/models.py` and are documented above.

## 9. ScenePlan (superseded)

The design-time `ScenePlan` shape has been consolidated into the implemented
`ProductionPlan` (§10): `ProductionPlan.scenes` carries the scene-level
description, narration, and transitions, so **no separate ScenePlan artifact is
produced** and the `sp_` id prefix is not used.

## 10. ProductionPlan (production.json)

**Implemented** (`agents/scene_planner/`). Stage 6 converts the QA-passed
Stage-4 `Script` into this provider-independent, versioned plan — the stable
CONTENT → VIDEO boundary (`VIDEO_PIPELINE.md` §3). A future video-generation
stage consumes *only* this artifact and never reads `Script` /
`ResearchPackage` directly.

The plan describes **what to show**, never **how some provider should render
it**: no video-model names, no API fields, no provider syntax (those belong to
the Stage-7 GPU worker).

Example (committed fixture / `generate_mock_production_plan()`):

```json
{
  "schema": "ProductionPlan",
  "schema_version": 1,
  "created_at": "2026-09-20T12:00:00.000000+00:00",
  "run_id": "mock",
  "stage": "scene_planning",
  "produced_by": { "prompt_version": null, "provider": null, "model": "MockScenePlanner" },
  "data": {
    "production_plan_id": "pp_e99181e73b5d",
    "topic_id": "t_octopus_hearts",
    "topic_title": "Why octopuses have three hearts",
    "script_id": "sc_9fc41d7a46b7",
    "script_qa_result_id": "sqa_e99181e73b5d",
    "scenes": [
      {
        "scene_id": "scn_01",
        "order": 1,
        "duration_seconds": 4.0,
        "script_line_ids": ["ln_00"],
        "narration": "You probably didn't know this, but Octopuses have three hearts",
        "claim_refs": ["cl_b4ca50a10d"],
        "visual_type": "generated_video",
        "visual_description": "Vertical close-up of a common octopus moving through clear ocean water; the camera slowly pushes toward the animal while three subtle heart indicators appear on its body.",
        "generation_prompt": "Vertical documentary-style underwater shot of a common octopus moving slowly across a rocky seabed, cinematic natural lighting, gentle camera push-in, three subtle heart indicators appearing on the body, realistic anatomy, educational visual style, 9:16 vertical composition, roughly four seconds of movement.",
        "negative_prompt": "distorted anatomy, extra limbs, text artifacts, watermark, blurry image, unrealistic deformation",
        "on_screen_text": [],
        "transition": "cut",
        "visual_continuity_group": "octopus",
        "asset_requirements": [
          { "asset_type": "generated_video", "subject": "octopus swimming on a rocky seabed", "quantity": 1, "continuity_group": "octopus" }
        ],
        "provenance": {
          "generated_by": "MockScenePlanner",
          "planning_rule": "hook_scene",
          "claim_texts": {
            "cl_b4ca50a10d": "Octopuses have three hearts: two branchial hearts that push blood through the gills and a systemic heart that powers the body."
          }
        }
      },
      {
        "scene_id": "scn_03",
        "order": 3,
        "duration_seconds": 8.4,
        "script_line_ids": ["ln_02"],
        "narration": "Octopuses have three hearts: two branchial hearts that push blood through the gills and a systemic heart that powers the body.",
        "claim_refs": ["cl_b4ca50a10d"],
        "visual_type": "motion_graphic",
        "visual_description": "Animated vertical diagram of an octopus with three heart icons highlighted: two small hearts near the gills and one larger heart in the body, clean educational labels, gentle pulsing highlights.",
        "generation_prompt": "Animated vertical motion graphic of an octopus with three heart icons adjacent to the gills and body, subtle pulsing highlights, clean science-explainer labeling, smooth motion-graphic animation, 9:16 vertical composition, about eight seconds.",
        "negative_prompt": "distorted anatomy, extra limbs, text artifacts, watermark, blurry image, unrealistic deformation",
        "on_screen_text": [ { "kind": "label", "text": "Three hearts", "claim_refs": ["cl_b4ca50a10d"] } ],
        "transition": "cut",
        "visual_continuity_group": "octopus",
        "asset_requirements": [
          { "asset_type": "motion_graphic", "subject": "three-heart anatomy diagram", "quantity": 1, "continuity_group": "octopus" }
        ],
        "provenance": {
          "generated_by": "MockScenePlanner",
          "planning_rule": "fact_motion_graphic",
          "claim_texts": {
            "cl_b4ca50a10d": "Octopuses have three hearts: two branchial hearts that push blood through the gills and a systemic heart that powers the body."
          }
        }
      }
    ],
    "total_duration_seconds": 45.2,
    "global_style": {
      "aspect_ratio": "9:16",
      "visual_style": "educational documentary",
      "target_resolution": "1080x1920",
      "fps": 30,
      "color_palette_notes": "muted natural documentary tones; accent highlights for facts",
      "consistency_notes": "same octopus design and lighting atmosphere across all scenes",
      "default_transition": "cut",
      "visual_continuity_requirements": "the octopus subject must keep one consistent appearance, environment and palette wherever it appears"
    },
    "covered_script_line_ids": ["ln_00", "ln_01", "ln_02", "ln_03", "ln_04", "ln_05", "ln_06", "ln_07"],
    "uncovered_script_line_ids": [],
    "continuity": [
      {
        "group": "octopus",
        "subject": "common octopus",
        "appearance": "realistic common octopus, natural reddish-brown coloration",
        "environment": "clear ocean water over a rocky seabed",
        "palette": "muted natural documentary tones",
        "recurring_elements": "three highlighted heart indicators"
      }
    ],
    "production_notes": [
      "Produced by the deterministic MockScenePlanner; every visual is a suggestion for a future provider, not an asset.",
      "Factual visuals derive solely from the Stage-3 approved claims and the Stage-4 script; no new factual information was added.",
      "Total duration mirrors the Stage-4 script within 0.5s tolerance."
    ],
    "provenance": {
      "generated_by": "scene_planner_agent_v1",
      "source": "MockScenePlanner",
      "created_at": "2026-09-20T12:00:00.000000+00:00",
      "provider": null,
      "model": "MockScenePlanner",
      "prompt_version": "scene_planning.v1"
    }
  }
}
```

- **QA gate:** `ScriptQAResult.overall_verdict` must be `PASS`; otherwise the
  stage is BLOCKED before any planning (REVISE/REJECT always block; an invalid
  plan is a structured failure, never a silent fallback). The gate's
  `qa_result_id` is recorded as `script_qa_result_id`.
- **Scene mapping:** each scene maps ≥ 1 script line (`script_line_ids`)
  unless its `visual_type` is `transition`; the mapped lines equal the full
  script by default (`require_full_coverage`), and
  `covered_script_line_ids` / `uncovered_script_line_ids` make the mapping
  explicit. `total_duration_seconds` mirrors the script duration within
  `duration_tolerance_seconds` (default 0.5).
- **Grounding rule (safety, defense in depth):** every factual scene and every
  factual `label`/`caption` on-screen text MUST carry `claim_refs` pointing
  only at **approved** claims (`result.approved_claims`). The planner receives
  only approved claim records (the Stage-3 whitelist join, same as Stages 4/5)
  and never introduces new factual information; each scene's `provenance`
  embeds the referenced claim texts so the chain
  scene → script line → approved claim → evidence → source is auditable.
  Validator rules r10–r15 re-check this even for a future LLM planner.
- **Controlled vocabulary:** `visual_type` ∈ {generated_video, generated_image,
  stock_asset, diagram, text_only, motion_graphic, existing_asset,
  transition}; `transition` ∈ {cut, fade, dissolve, zoom, match_cut, none};
  `on_screen_text.kind` ∈ {headline, emphasis, label, caption}. Generative
  types require a `generation_prompt`; `negative_prompt` is **visual quality
  only** and must never carry factual constraints.
- **Operator-agnostic style:** default 9:16 / 1080x1920 / 30 fps /
  "educational documentary"; configurable in `ScenePlanningConfig`.
- **Continuity:** `continuity` plus per-scene `visual_continuity_group` keep
  one subject/appearance/environment consistent across scenes.
- **Validation (rules r01–r21):** r01–r20 run in the validator
  (`agents/scene_planner/validator.py`) — non-empty plan, unique scene ids,
  contiguous order, positive durations, internally consistent total, total
  within script tolerance, ≥1 line per non-transition scene, full coverage,
  referenced lines exist, claim refs approved (unknown / unapproved / rejected
  / review each have their own rule), factual visuals trace claims, factual
  text traceability, prompts where required, valid visual types, no
  provider-specific wording or keys anywhere in the contract (the plan-level
  `provenance` record is audit metadata and is excluded), provenance present,
  JSON round-trip. **r21** (determinism) is a planner-level guarantee: running
  the same inputs twice yields the identical plan — asserted in tests.
- **Planner protocol:** `ScenePlanner.plan(script, qa_result, result, claims,
  config, run_id, created_at) -> ProductionPlan`. The deterministic, offline
  `MockScenePlanner` implements it; a future LLM planner implements the same
  protocol and must pass the same validator. Provider fields (e.g. seed,
  sampler, API keys, model names) are Stage-7 concerns, not part of this
  contract.

CLI: `python -m agents.scene_planner` (writes
`data/<run_id>/production/production.json` + run state advanced to
`Stage.SCENE_PLANNING`). A blocked gate or invalid plan exits non-zero without
writing an artifact. Config lives in `ScenePlanningConfig` /
`agents/scene_planner/models.py`.

## 11. QAResult

Design-time placeholder for generic quality-gate results. **Implemented so
far: `ScriptQAResult` (§8.1).** This generic shape is reserved for the future
video/policy QA stages; those stages will likely reuse §8.1's issue/severity/
gate model.

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
├── production/         ProductionPlan (production.json), QAResult(video), QAResult(policy)
└── videos/             generated assets + final Short
```

## 15. Open questions

- ID formats implemented so far: `ts_<uuid[:12]>` (TopicSelection),
  `src_<sha1(url)[:10]>` and `cl_<sha1(text)[:10]>` (research),
  `rp_<uuid[:12]>` (ResearchPackage, though deterministic `rp_<sha1(run:topic)[:12]>`
  is used by the research agent), `ev_<sha1(source_id:excerpt)[:10]>` (evidence),
  `fc_<sha1(run_id:package_id)[:12]>` (FactCheckResult), `sc_<sha1(run_id:topic_id)[:12]>`
  (Script), `sqa_<sha1(run_id:script_id)[:12]>` (ScriptQAResult), and the
  Stage-6 `pp_<sha1(run_id:script_id)[:12]>` (ProductionPlan) with sequential
  scene ids `scn_01..scn_NN`. Topic ids are stable typed slugs (`t_<slug>`).
  Remaining stages' prefixes (`qa_`/`an_` and the Stage-7 worker result) are
  still to be settled by those stages.
- Whether QA results for all stages share one file each per run or one
  QAResult registry file with stage records.