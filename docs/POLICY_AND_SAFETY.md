# Policy And Safety

Quality-control stage that checks a finished Video against YouTube
policy, advertiser suitability, and the channel's own standards **before
publishing**. Design only; nothing is implemented.

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

`QAResult` with:

- stage = `policy`
- per-check outcomes (PASS/REVIEW/REJECT + notes)
- overall verdict
- recommended actions (e.g. "add synthetic-media disclosure",
  "re-render asset X", "rewrite description")

Stored at `data/<run_id>/` alongside the QA records.

## 6. Positioning vs. other QA

| Stage | Scope |
|---|---|
| script QA | Script quality, claim-fact type alignment |
| video QA | Technical/visual/audio issues in the render |
| **policy/publish QA** | Eligibility to be published at all |

They are separate stages with the same `QAResult` shape, different
`stage` tags (`pipeline/state.py` `Stage` already enumerates them).

## 7. Human approval gate

- Initially **mandatory** before any YouTube upload.
- The gate presents: policy report, video QA report, script, claims
  trace, and the finished video.
- Automating approval is possible **later, only after sufficient
  testing**; the design keeps the gate a first-class stage so it can be
  relaxed without surgery.
- Fail-closed rule: absence of a rejection is not approval.
  Silent default = REJECT/PENDING.

## 8. Open questions

- Which licensing sources/registries we trust for assets (stock,
  CC-BY, generated+recorded) — must be settled before first upload.
- Synthetic-media disclosure policy specifics for *faceless* content
  (whether/how to label; default is to disclose per current policy).
- Extra channel metadata (description template, hashtags) rules —
  consistent and non-spammy.