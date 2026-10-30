---
name: task-report
description: Use when finishing a task in this repository, before declaring it done, to write the report under reports/. Applies the AGENTS.md report template, secret safety, honest verification, and accurate commit/push status.
---

# Task Report

Owns one thing: producing the per-task report.

## Steps

1. Write exactly one report to `reports/<descriptive-title>.md`, named after its
   title. One report per task; never rewrite another task's report.
2. Use the template in `AGENTS.md` §3. Do not maintain a second copy of the
   template here; if the template changes, follow `AGENTS.md`.
3. Fill it with facts, not intentions.

## Required content

- Task and scope
- Starting repository state: branch, HEAD, working tree, pre-existing changes
- Files changed, and files deliberately not changed
- Verification **actually run**, with real results
- Failures, skips and known limitations, named explicitly
- External operations performed, and those deliberately not performed
- Commit and push status
- Blockers and the next logical action

## Honesty rules

- A skipped test is not a pass. An unavailable service is not "0 results".
- Use `VERIFIED` / `NOT VERIFIED` / `NOT TESTED` / `BLOCKED` / `PLANNED` /
  `NOT CHANGED` explicitly, per `AGENTS.md` §4.
- Never claim a task was committed or pushed unless the commit and push actually
  happened. If they were blocked, say so and why.
- Never fabricate commands, outputs, run IDs, counts or timings.

## Secret safety

Never write an API key, password, token, credential, secret environment value or
private authentication material into a report. Describe presence only, for
example `GEMINI_API_KEY: configured`.

## Git and file handling

- Do not force-add `reports/`; it is gitignored by design. State that the report
  is local-only when that is the case.
- Commit and push the task's code and docs per `AGENTS.md` §9, or report the
  blocker.
- Never merge into `main`.

## Closing checklist

- [ ] Exactly one report written under `reports/`
- [ ] Template from `AGENTS.md` §3 followed
- [ ] Verification results are real and complete, including failures and skips
- [ ] External operations stated explicitly
- [ ] Commit and push status accurate
- [ ] No secrets in the report
- [ ] `opencode_report.md` untouched
