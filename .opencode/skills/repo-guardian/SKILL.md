---
name: repo-guardian
description: Use at the start of any task in this repository, before proposing or making changes. Establishes AGENTS.md rules, the latest continuity checkpoint, relevant reports, and current Git state.
---

# Repo Guardian

Entry-point skill for work in this repository. Load it first; the other four
skills assume it has run.

## What to do, in order

1. Read `AGENTS.md` — the authoritative standing rules. Do not rely on a summary
   here. The sections that matter most are:
   - §2 documentation roles and authoritative-source order
   - §4 verification honesty
   - §5 architecture
   - §6 removed subsystems
   - §9 Git workflow
   - §10 data and artifact safety
2. Read **only the latest** `## Current continuity checkpoint` section of
   `docs/PROJECT_PROGRESS.md`. Do not read the whole file unless the task
   genuinely needs its history.
3. Consult `reports/*.md` selectively — only the reports relevant to this task,
   and treat each as the record of its own task only.
4. Inspect live Git state: `git status --short --branch`, `git log -1 --oneline`,
   `git remote -v`, and note staged, unstaged and untracked work.
5. Inspect the actual code and tests for any claim the task depends on.

## Standing rules

- Code and tests are the fact; documents are the record of intent. If they
  disagree, say so instead of silently trusting either.
- The working tree is dirty **on purpose**. Preserve every pre-existing
  modification and unrelated untracked file. Never revert, clean or reset.
- Stage only files belonging to the current task. Never `git add .` or
  `git add -A`. Follow `AGENTS.md` §9.
- Never add, edit or replace `opencode_report.md`; it is retired and frozen.
- Do not introduce scheduled pipeline execution, an orchestration layer, or the
  removed subsystems described in `AGENTS.md` §6 and §8.

## Stopping conditions

Stop and report instead of guessing when:

- the branch, the remote, or the ownership of existing changes is unclear
- a required deliverable is gitignored and therefore cannot be committed without
  a policy decision
- the code contradicts the documentation in a way that changes the plan
- the task would require an external service, GPU, Colab, Drive, rclone, model
  download, or generation call that the task has not authorized

## Related skills

- `pipeline-stage` — running or debugging an individual stage
- `e2e-runner` — end-to-end runs and artifact preservation
- `removed-subsystem-guard` — asked to add a subsystem that was removed
- `task-report` — finishing the task and writing its report

Descriptions only make a skill visible. A skill is applied only when it is
actually loaded; do not assume one has been applied because its topic came up.
