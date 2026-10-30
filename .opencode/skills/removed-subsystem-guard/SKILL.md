---
name: removed-subsystem-guard
description: Use when asked to add, restore or re-enable fact checking, evidence/corroboration, source validation, web retrieval, Script or Video QA, policy gates, or human approval. Explains that these were intentionally removed and requires explicit scope approval before any work.
---

# Removed Subsystem Guard

Several subsystems were deliberately removed from this project. This skill
exists to stop them being reintroduced by accident.

## Current exclusions

`docs/ARCHITECTURE.md` records these as **intentionally removed**:

- Fact checking and corroboration
- Evidence / source validation and credibility tiering
- SearXNG, BeautifulSoup / httpx direct retrieval, Firecrawl, hybrid retrieval,
  query planning, retrieval traces
- Script QA and Video QA
- Policy gates
- Human approval

The research basis is the model's own knowledge. There is no web retrieval and
no external evidence in the pipeline.

## What enforces it

Re-verify these tests against the live repository before relying on this list.

| Test | Asserts |
| --- | --- |
| `tests/test_video_render_spec.py::test_no_fact_check_evidence_or_qa_concepts_in_stage5` | Stage 5 sources contain no evidence, corroboration, fact-check, citation, verdict, PASS/FAIL, QA, approval, source, retrieval or web-search concepts |
| `tests/test_finance_script_generation.py::TestNoQaContract` | `agents.script` exposes no validation verdict or score; no `validator.py`, `qa.py`, `quality.py`, `fact_checker.py` |
| `tests/test_finance_ai_research.py::TestNoSourceMachinery` | The research contract has no source/evidence/citation/url fields and imports no retrieval modules |
| `tests/test_finance_topic_selection.py::test_prompt_has_no_fact_checking_stage_requirements` | The Stage 1 prompt states there is no fact-verification stage |

## When this skill applies

Someone asks to add or restore any item above, or a stale document implies one
already exists in the pipeline.

## Required response

1. Say plainly that the subsystem was intentionally removed and is enforced by
   the tests listed above.
2. Do **not** invent or cite files that do not exist. There is no `agents/qa/`
   package, no `agents/qa/fact_checker/gates.py`, and no
   `corroboration_min_sources` setting anywhere in this repository.
3. Do **not** describe fact checking as a current stage. The pipeline is five
   stages, and Script Generation is Stage 3; there is no fact-checking stage.
4. Treat the request as an architecture change: require explicit, separately
   approved scope before writing code, tests or documentation that assumes it.
5. If approval is given, plan the change explicitly — including which enforcing
   tests must change and why — instead of deleting or skipping enforcement to
   make a build pass.

## Boundaries

- Do not modify pipeline code or tests from this skill alone.
- Do not delete, skip or weaken the enforcing tests.
- The retired ten-stage design and `experiments/retrieval/` remain documented in
  `AGENTS.md` §6 for context only. Their existence is not a mandate to resume
  that work, and production code must not import the experiment.
