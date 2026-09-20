# Topic Selection

How the system finds candidate topics, scores them, and picks the topic
for a run — including the **cold-start** problem and the
**exploration/exploitation** behavior. Design only; nothing is
implemented.

## 1. Goal

Choose, for each run, a topic that is worth making a Short about, using
signals available **without any channel history** at first, and
gradually incorporating channel performance as it accumulates.

Two hard requirements:

1. It must work from day one with **no historical channel data**.
2. It must **not simply pick the highest-scoring topic every day**.

## 2. Stage: discovery → selection

Two logical stages:

| Stage | Output contract | Role |
|---|---|---|
| topic discovery + seo research | `TopicCandidate[]` | Find/score raw candidate topics using external signals |
| topic selection | `TopicSelection` | Decide which candidate is produced this run |

Discovery/research gather **external signals**; selection applies the
exploration/exploitation policy and emits the final decision.

## 3. External signals (no channel data needed)

Signals available from the moment the channel starts:

- **YouTube search / video data** — what people search, related videos.
- **Trend signals** — rising interest on the web/social/YouTube.
- **Competition** — how many/which videos already cover the topic, and
  their apparent quality.
- **Topic interest** — baseline audience size for the subject.
- **Evergreen potential** — does the topic age well?
- **Curiosity / click potential** — hook strength, gap between common
  belief and reality.
- **Explainer potential** — can it be explained in 30–60 s?
- **Visual potential** — can it be shown?
- **Audience fit** — on-pillar for the channel.
- **Freshness** — only relevant when the topic is time-sensitive.

These map to per-signal scores recorded on each `TopicCandidate`
(`DATA_CONTRACTS.md`).

## 4. Historical performance signal (optional, initially null)

Channel analytics (how past videos performed, `ANALYTICS.md`) are a
**pluggable signal** — not a hard-coded scoring branch.

- During cold start: historical performance is `null`/absent.
- Later: it becomes one weighted signal among the external ones.
- This keeps the door open for an ML-driven optimizer later without
  forcing one now.

## 5. Exploration / exploitation

The system should behave like an exploration/exploitation system
**without implementing a bandit or ML model yet**.

Design principles:

- **Exploration dominates early** (no data → you cannot know what
  performs; coverage across pillars and topic types builds the signal).
- **Greedy selection is prohibited** — picking the same style of
  highest-scored topic every run is worse than occasionally picking a
  slightly lower-scored one, because it starves the channel of data and
  risks monotony/spam patterns.
- **Balance shifts gradually**: as analytics accumulate, historical fit
  gets weight, while a **reserve of experimentation** remains to keep
  learning.
- Every decision records which mode it used so the choice can be
  audited and tuned.

### Conceptual behavior only

This is intentionally **not** a formal epsilon-greedy/UCB or Bayesian
bandit at this stage. The interface is designed so one can be dropped in
later behind the same boundaries.

A minimal `TopicSelector` interface (design sketch, not code to ship):

```python
def select(candidates: list[TopicCandidate],
           history: Optional[HistoryAggregate]) -> TopicSelection:
    ...
```

- `candidates` — the scored pool from discovery.
- `history` — channel-level performance summary; `None` during cold
  start.
- Returns a `TopicSelection` carrying the chosen candidate, composite
  scores, and the decision `mode` (`explore` / `exploit` / `mixed`).

This is the seam where a future bandit/ML optimizer plugs in: identical
signature, smarter internals.

## 6. Scoring model (v1, hand-tuned weights)

For v1 the composite score is a **weighted sum with a small amount of
controlled randomness/coverage logic** to honor the exploration
requirement. Explicitly out of scope for v1:

- learned weights
- value-function learning
- formal bandit algorithms

Decisions that must be tuned and remain explicit configuration:

- weight per signal,
- exploration fraction,
- pillar/topic-type coverage bucket (don't repeat the same pillar type
  too densely),
- minimum signal coverage (reject topics with zero data on key signals).

## 7. Data contracts used here

- `TopicCandidate` — one topic with per-signal scores and metadata.
- `TopicSelection` — the decision, scores, mode, and rationale.

Both are versioned and carry run provenance (`DATA_CONTRACTS.md`).

## 8. Open questions

- Actual weights and exploration fraction for v1 (tuned after first
  real runs; do not predecide).
- Definitions of "trend" data sources (which external source provides
  trend/search signals) — this determines the first API integration.
- How many candidates each discovery pass should surface.
- Coverage policy: minimum spacing between pillars / topic types.