# Research Pipeline

How the system turns a selected topic into **evidence-backed claims**
that later stages (fact check, script) are allowed to consume.

The research stage is **implemented** as a deterministic, mock-backed
pipeline (no network/API/LLM calls). A pluggable `ResearchProvider`
interface (`agents/research/provider.py`) exists to swap in real search
sources later with no agent changes. The **fact-check stage remains design
only** (`FACT_VERIFICATION.md`).

## 1. Guiding rule

**The AI is not the source of truth.** The LLM proposes; evidence
decides. A claim that cannot be traced to sources must not flow into
the script.

## 2. Stage output

Research produces a `ResearchPackage` containing:

- the selected `topic_id` + `topic_title` (from the Stage 1
  `TopicSelection`)
- a set of `FactClaim`s (the claims the video might use)
- the `Source`s behind those claims
- provisional per-claim confidence and verification status
- `research_status` — `provisional` / `insufficient_evidence` /
  `ready_for_fact_check`
- `research_notes` and `provenance` (which agent/provider produced it)

The package is then handed to the **fact check stage** (separate stage,
see `FACT_VERIFICATION.md`). Research itself does not judge
"correct/wrong"; it gathers and attributes.

## 3. Claims

Every important factual claim carries:

- `claim_id` — stable, deterministic `cl_<sha1(text)[:10]>`
- claim text
- `claim_type` (see below)
- `source_ids` (supporting sources; each must resolve inside the package)
- `verification_status` — provisional, set in research; settled later
- `confidence` — a 0–1 heuristic from backing source tiers, **not** a
  fact-check result
- `dates` — a list of `{date_kind, date}` entries; kinds in
  `publication_date` / `discovery_date` / `announcement_date` /
  `event_date`. Zero or more, present when relevant (see §5)
- `notes` — optional free text
- `critical` flag — claims the video's argument depends on. Critical
  claims gate the pipeline (`FACT_VERIFICATION.md`).

### Claim types (must be distinguished)

| Type | Meaning | Example |
|---|---|---|
| `fact` | Directly supported by sources | "The Kepler-186f system hosts a rocky planet announced in 2014." |
| `inference` | Drawn from facts, not stated in any source | "Habitable-zone planets are probably common given this detection rate." |
| `explanation` | How/why, derived from sources | "Transit method dims the star during each orbit." |
| `speculation` | Unsupported or forward-looking | "This could make interstellar probes feasible someday." |
| `opinion` | Judgment/valuation | "It is the most exciting exoplanet find." |

The pipeline must **not** blur speculation into fact during script
generation. Script wording for non-fact claims is constrained to match
their type (e.g. "may", "suggests", "speculation").

## 4. Sources

| Field | Purpose |
|---|---|
| `source_id` | stable ID used by claims (`src_<sha1(url)[:10]>`) |
| `title` | title/metadata |
| `url` | origin |
| `publisher` | who published |
| `source_type` | document kind (research paper, preprint, press release, news article, encyclopedia, official statement, …) |
| `publication_date` | when the source was published |
| `credibility_tier` | primary / high_quality_secondary / discovery |
| `retrieved_at` | when *we* retrieved it |
| `notes` | optional free text (e.g. "MOCK FIXTURE" flags) |

### Source credibility tiers

1. **Primary** — the origin of the fact (peer-reviewed paper, original
   dataset, official announcement, instrument/web page record, primary
   documentation).
2. **High-quality secondary** — reputable synthesis/coverage of primary
   material (major scientific outlets, institutional summaries,
   reference works that cite sources).
3. **Discovery** — search results, aggregators, social/trend posts,
   casual content. Good for *finding* topics and leads; **not sufficient
   evidence by itself**.

### Rules

- A claim must be backed by at least one non-discovery source unless it
  is genuinely a primary-only fact.
- One bottle-of-wine rule: **one supporting article is not
  confirmation**; enforcement lives in fact check, but research must
  surface at least attempt at corroboration.
- Discovery sources may be attached to claims as leads, never counted
  as sufficient.

## 5. Dates: freshness must be earned

Fresh ≠ just discovered. Four distinct date kinds are tracked where
relevant; they map 1:1 to the `date_kind` values allowed on a claim's
`dates` list:

- **publication_date** — when a source was published.
- **discovery_date** — when the phenomenon/entity was discovered.
- **announcement_date** — when it was announced.
- **event_date** — when the underlying event happened.

A 2026 article about a 2019 discovery is **not fresh news**. A theorem
published in 1789 is only relevant today if it still matters. The
research stage labels claims with the correct date kind so topic
selection and script don't mistake document recency for subject
freshness.

## 6. Research does not verify

Research's `verification_status` is provisional by design. The full allowed
set is `provisional` (the default) / `pending` / `single_source` /
`multiple_sources` / `disputed` / `unsupported`; any value implying final
verification is rejected at the model boundary.

Settlement (verified / contradicted / unsupported → PASS/REVIEW/REJECT)
is the exclusive job of the **fact check stage**. The script generator
may only consume a package the fact checker approved.

## 7. Contract

- `ResearchPackage`, `FactClaim`, `Source` — see `DATA_CONTRACTS.md` (§4–§6).
- `research_status` in `provisional` / `insufficient_evidence` /
  `ready_for_fact_check`.
- Claim type and confidence are currently deterministic rule-based
  heuristics (`agents/research/classifier.py`); a real provider may later
  supply typed claims, which the agent validates into this model.
- Stored at `data/<run_id>/research/`.

## 8. Open questions

- Minimum corroboration policy (how many independent sources for a
  `critical` fact) — to be set as a rule, not an LLM whim.
- Which search / literature APIs back discovery and retrieval (first
  integration dependency; not chosen yet).
- Publisher allow/block list maturity over time.