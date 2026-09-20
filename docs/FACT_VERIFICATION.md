# Fact Verification

A **separate stage from research** that settles whether claims are fit
for the script. Design only; nothing is implemented.

## 1. Responsibility boundary

- **Research** gathers claims and sources (provisional).
- **Fact check** decides soundness and issues a verdict.

Fact check never silently rewrites claims; it returns outcomes so a
human (or later an automated repair loop) can act.

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
only read **approved research** (`approved` status on
`ResearchPackage`), never raw unrestricted research.

## 4. Verification method (rules, not vibes)

The fact checker applies deterministic-ish checks around sources and
claims:

1. **Source hierarchy weight** — primary > high-quality secondary >
   discovery (`RESEARCH_PIPELINE.md`). Discovery alone ≠ support.
2. **Corroboration** — count of independent, non-discovery sources per
   critical claim (policy threshold set in configuration).
3. **Contradiction scan** — search for evidence against the claim; a
   single strongly sourced contradiction beats multiple weak
   confirmations.
4. **Date alignment** — claim dates must match source/event dates
   (`RESEARCH_PIPELINE.md` §5). A claim about "today" backed only by an
   old or pre-announcement source fails.
5. **Type integrity** — a claim labeled `fact` that is really an
   `inference`/`opinion` demotes or rejects per policy.
6. **Source sanity** — source metadata present, URL plausible, publisher
   known, not an LLM-generated source citation (hallucinated references).

Outputs must be explainable: every outcome carries the evidence and
reasoning so a human can audit in seconds.

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

- `FactCheckResult` (result, per-claim outcomes, blocked claim IDs).
- Attached to the `ResearchPackage` approval status before any script
  work. Stored at `data/<run_id>/research/`.

## 8. Open questions

- Corroboration threshold per claim type (config value; suggested
  default: ≥2 independent non-discovery sources for critical facts).
- Whether fact check is fully LLM-in-the-loop or rule-first with LLM
  assist (initial: rule-first, LLM assist).
- Contradiction evidence weighting (how many sources beat one).