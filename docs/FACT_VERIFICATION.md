# Fact Verification

A **separate stage from research** that settles whether claims are fit
for the script. Stage 3 (fact check) is **implemented** — deterministic,
mock-backed (`agents/qa/fact_checker/`), consuming the research stage's
`ResearchPackage` and producing a `FactCheckResult`. It is an *evidence
verifier*, not a truth source: every verdict derives only from the
evidence actually supplied for the claim's cited sources.

## 1. Responsibility boundary

- **Research** gathers claims and sources (provisional).
- **Fact check** decides soundness and issues a verdict.

Fact check never silently rewrites claims; it returns outcomes so a
human (or later an automated repair loop) can act.

Key rule enforced by the implementation: **a source is not evidence.**
The checker operates on `Evidence` objects (content excerpts retrieved per
source by an `EvidenceProvider`), never on bare URLs, and never on the
verifier's own world knowledge.

## 2. Verdicts

A `FactCheckResult` carries an overall verdict:

| Verdict | Meaning |
|---|---|
| `PASS` | All claims accepted; approved for script writing |
| `REVIEW` | Needs human/expert review (ties, ambiguous, partial support) |
| `REJECT` | Blocked: critical claim unsupported or contradicted, or fatal issues |

Per-claim outcomes map onto the same tri-state with notes:

| Outcome | Meaning |
|---|---|
| `PASS` | Supported by credible, non-discovery sources; no contradiction found |
| `REVIEW` | Partial support, conflicting sources, or only discovery sources |
| `REJECT` | Contradicted by better evidence, or critical with no support |

## 3. Gating rule

- One **critical** claim with outcome `REJECT` ⇒ overall `REJECT`.
- Multiple critical `REVIEW`s ⇒ overall `REVIEW` (human/expert
  decides), never silently shipped.
- Non-critical `REJECT` claims may be *dropped from the script* in
  remediation; the pipeline must not publish a script still carrying a
  rejected claim.

Rejected packages cannot reach script generation. Script generation may
only consume research that passed the fact-check stage (a `FactCheckResult`
verdict, `DATA_CONTRACTS.md` §7); it must never read raw, still-provisional
research directly.

## 4. Verification method (rules, not vibes)

The fact checker is fully deterministic and network-free: a rule-based
`MockVerifier` (hash/coverage text matching over the supplied excerpts)
produces an evidence analysis, and `gates.py` turns it into a verdict.
`agents/qa/fact_checker/`:

1. **Source credibility tier weight** — primary > high-quality secondary >
   discovery (`RESEARCH_PIPELINE.md` §4). Discovery alone ≠ support:
   discovery-only evidence reviews the claim.
2. **Corroboration** — critical claims need ≥2 independent non-discovery
   sources (`FactCheckConfig.corroboration_min_sources`, default 2) or the
   claim stays `REVIEW`.
3. **Contradiction scan** — a credible (non-discovery) source whose excerpt
   contradicts the claim forces support level `contradictory` and `REVIEW`,
   even against otherwise good support. A discovery-tier contradiction does
   not override primary support.
4. **Stronger-than-evidence** — a claim worded more strongly than its
   evidence (e.g. "proved X causes Y" vs "observed a correlation") reviews;
   speculation/opinion presented as fact rejects; unsupported claims reject.
5. **Coverage thresholds** — the portion of the claim's meaningful wording
   found verbatim in an excerpt decides `direct` (≥0.8), `partial` (≥0.3),
   `contradictory` (≥0.2), else `none`. Direct → PASS, partial → REVIEW.
6. **Type integrity** — a claim typed `speculation`/`opinion` can never be
   a factual PASS regardless of its evidence wording.
7. **Source sanity** — claims referencing unknown sources, claims with no
   sources, and empty packages reject. Evidence referencing unknown sources
   or failing validation is dropped and recorded in `evidence_issues`
   (never a crash, never a silent PASS).

Date alignment (`RESEARCH_PIPELINE.md` §5) is **not yet gated**: claim dates
are checked for freshness only by future stages. Verdict gates can only make
a verdict worse (`PASS < REVIEW < REJECT`); a gate never upgrades to PASS.

Confidence (0–1, per claim): tier base primary 0.85 / high-quality secondary
0.6 / discovery 0.3, +0.05 per extra independent non-discovery source (cap
0.95), REVIEW capped at 0.5, REJECT capped at 0.3.

Outputs are explainable: every outcome carries the exact evidence ids,
supporting/contradicting source ids, and the reason, so a human can audit in
seconds.

## 5. Claims and traceability

Every accepted claim keeps its `claim_id` so the chain survives:

```
script sentence → claim_id → research claim → source_id → original source
```

This chain is the audit trail for every published video and for later
retractions/corrections policy (`FACT_VERIFICATION.md` + `DATA_CONTRACTS.md`).

## 6. Human involvement

- `REVIEW` verdicts route to a human/expert path (currently manual, since
  no automation exists).
- `FAIL`→`REJECT` outputs record actionable reasons for the research
  stage / script stage to adjust. Automated remediation is future work;
  v1 is fail-closed to human gates.

## 7. Contracts

- `FactCheckResult` (result, per-claim outcomes, blocked claim IDs;
  `DATA_CONTRACTS.md` §7).
- Stored alongside its `ResearchPackage` at `data/<run_id>/research/`;
  script generation gates on the fact-check verdict, not on a
  research-stage status.

## 8. Open questions

- Whether a real LLM verifier is added later. The `Verifier` protocol is the
  future LLM boundary: a model can be plugged in, but its prompt must be
  constrained to `{claim, supplied evidence, source metadata}` and must never
  fall back on outside memory as evidence (`agents/qa/fact_checker/verifier.py`).
- Date/freshness gating (claim dates vs source/event dates) is future work.
- Contradiction evidence weighting (how many discovery-tier sources would
  outweigh one primary confirmation).