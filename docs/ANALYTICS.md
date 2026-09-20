# Analytics

Future interface only. Nothing is implemented or collected yet — this
document defines *what the shape will be* so the pipeline can be built
to produce the data that analytics will later need.

## 1. Purpose

After publishing, collect per-video performance and feed it back into
**topic selection** (`TOPIC_SELECTION.md`) as the historical signal the
channel lacks during cold start.

## 2. Metrics of interest

| Metric | Note |
|---|---|
| views | |
| retention (average + curve where available) | the best explainer-quality signal |
| likes | |
| comments | |
| shares | |
| subscribers gained | |
| other useful YouTube analytics | impressions, CTR, watch time — kept in `other` until proven useful |

## 3. Feedback loop (design)

```
YouTube → AnalyticsRecord → history store → TopicScorer/TopicSelector
```

- The `AnalyticsRecord` contract (`DATA_CONTRACTS.md` §12) is the unit of
  exchange.
- A future collector stage fetches from the YouTube API, maps to
  `AnalyticsRecord`, stores it.
- Topic selection consumes *aggregated history* (`HistoryAggregate`,
  `TOPIC_SELECTION.md` §5). Analytics never calls topic selection
  directly; it writes records, the selector reads aggregates.

## 4. Interface boundary (no implementation)

Minimal interface the pipeline will eventually need:

```python
# design sketch, not code to ship
def collect(video_id: str) -> AnalyticsRecord: ...
def history(run_ids: list[str]) -> HistoryAggregate: ...
```

- `collect` — fetch + normalize one video's analytics.
- `history` — summarize store into the signal the selector consumes
  (per-pillar/per-type aggregates first; kept coarse, not an ML sink).

## 5. Cold start

While `history(run_ids)` returns nothing, topic selection runs in
exploration mode with `historical_fit = null` (`DATA_CONTRACTS.md` §3).
The pipeline is expected to publish a mix of topics early precisely so
this history becomes informative.

## 6. Storage

- Records stored flat per run (`data/<run_id>/analytics/` in the same
  per-run workspace), and a long-lived aggregate store for cross-run
  history (exact store choice deferred; CSV/SQLite are fine v1
  candidates).
- Records are immutable once written, but "same video re-collected"
  updates are permitted (growing retention/views) — latest record wins
  when building history.

## 7. Explicitly out of scope (until later)

- Dashboard/visualization.
- Regression/valuation models predicting success.
- Bandit integration (interface reserved, `TOPIC_SELECTION.md` §5).
- Auto-publish decisions driven by analytics.

## 8. Open questions

- Collection cadence (weekly first, since Shorts metrics stabilize
  within ~7 days).
- YouTube API credentials/scopes when upload + analytics are wired up
  (shared OAuth path with publishing).
- Which aggregates exist in v1 `HistoryAggregate` (pillar, type,
  duration bucket?) — keep small; grow with data.