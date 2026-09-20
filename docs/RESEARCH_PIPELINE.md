# Research Pipeline

How the system turns a selected topic into **evidence-backed claims**
that later stages (fact check, script) are allowed to consume.

Design only. Nothing is implemented.

## 1. Guiding rule

**The AI is not the source of truth.** The LLM proposes; evidence
decides. A claim that cannot be traced to sources must not flow into
the script.

## 2. Stage output

Research produces a `ResearchPackage` containing:

- topic + framing
- a set of `FactClaim`s (the claims the video might use)
- the `Source`s behind those claims
- provisional per-claim confidence and verification status

The package is then handed to the **fact check stage** (separate stage,
see `FACT_VERIFICATION.md`). Research itself does not judge
"correct/wrong"; it gathers and attributes.

## 3. Claims

Every important factual claim carries:

- `claim_id`
- claim text
- `claim_type` (see below)
- `source_ids` (supporting sources)
- `verification_status` (provisional, set in research; settled later)
- `confidence`
- relevant dates when applicable (`event`, `announcement`,
  `discovery`, `publication`)
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
| `source_id` | stable ID used by claims |
| `title` | title/metadata |
| `url` | origin |
| `publisher` | who published |
| `publication_date` | when the source was published |
| `source_type` | document kind (research paper, preprint, press release, news article, encyclopedia, official statement, …) |
| `hierarchy` | primary / high-quality secondary / discovery |
| `retrieval_date` | when *we* retrieved it |

### Source hierarchy

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

Fresh ≠ just discovered. Four distinct dates are tracked where relevant:

- **Publication date** — when a source was published.
- **Discovery date** — when the phenomenon/entity was discovered.
- **Announcement date** — when it was announced.
- **Event date** — when the underlying event happened.

A 2026 article about a 2019 discovery is **not fresh news**. A theorem
published in 1789 is only relevant today if it still matters. The
research stage labels claims with the correct date kind so topic
selection and script don't mistake document recency for subject
freshness.

## 6. Research does not verify

Research's `verification_status` values are provisional:
`pending`, `single_source`, `multiple_sources`, `disputed`,
`unsupported`.

Settlement (verified / contradicted / unsupported → PASS/REVIEW/REJECT)
is the exclusive job of the **fact check stage**. The script generator
may only consume a package the fact checker approved.

## 7. Contract

- `ResearchPackage`, `FactClaim`, `Source` — see `DATA_CONTRACTS.md`.
- Stored at `data/<run_id>/research/`.

## 8. Open questions

- Minimum corroboration policy (how many independent sources for a
  `critical` fact) — to be set as a rule, not an LLM whim.
- Which search / literature APIs back discovery and retrieval (first
  integration dependency; not chosen yet).
- Publisher allow/block list maturity over time.