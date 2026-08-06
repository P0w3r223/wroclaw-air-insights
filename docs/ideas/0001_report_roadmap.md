# Report roadmap — making the Pages report speak to a non-technical reader

Date: 2026-08-06
Status: draft (revised after review — the original diagnosis was half right)
Author: P0w3r223 + Claude
Related to: `src/wroclaw_air_insights/report.py`, `.github/workflows/refresh.yml`, PR #3

---

## Context

The published report showed three bare numbers — test MAE, RMSE and R². To a reader
without a modelling background they carry no information: no units to anchor on, no
sense of what "good" looks like, and no reference point to judge whether the model
adds anything at all.

The first pass (PR #3) added the explanation: model against its references on one table,
a plain-language headline, and a glossary defining each metric with its range. Explaining
the metrics forced them to be correct, which is how the finding below surfaced.

## Finding — and the correction to it

*Resolved in PR #3; kept because the reasoning error is worth not repeating.*

The then-deployed RandomForest beat the persistence baseline by 14.8% on MAE while scoring
R² 0.021.

**The original reading was that this is purely a low-variance-window artifact.** The window
is genuinely calm: sd(y_test) = 5.5 µg/m³ against sd(y_train) = 14.6, so there is little
variation available to explain and R² is squeezed toward zero. That part holds and is now
stated on the page with the numbers attached rather than as an assertion.

**But that explanation is incomplete, and the omission mattered.** On the *same* test
window — same denominator, same rows — `HistGradientBoosting` scores R² **0.23** against
RandomForest's 0.02 (`README.md`, model comparison table). If window variance were the
whole story, no model could reach 0.23 there. So the near-zero R² is part window artifact
and part model-selection gap, and the original roadmap reached "this is not a bug" by
quoting only two rows of a five-row table that sits in the repo's own README.

**A second omission ran the other way.** The single split does not only depress R², it also
flatters MAE: rolling-origin CV gives 7.18 ± 2.55 µg/m³ across the year (folds:
6.14 / 10.48 / 9.81 / 5.66 / 3.82) against the split's 4.15. One cause — an easy summer
window — moved both numbers, and the original draft noticed only the direction that looked
bad while proposing to publish the direction that looked good.

**Consequence for recommendation #1 as originally written.** Replacing R² with a
persistence skill score would have swapped 0.02 for 0.29 and changed the model not at all.
R² *is* a skill score — against climatology instead of persistence. Substituting one for
the other reads as metric-shopping to anyone who works that out. The correct move is
additive: show both references, name both as skill, and let the reader see that the model
beats the naive rule comfortably and a hindsight flat line barely.

### Fixed in PR #3

- The headline is now the year-round CV figure; the single-split number is labelled with
  its window and its season.
- A climatology row joined the table; R² and persistence skill are presented as one family
  with named references.
- MAE-relative-to-typical-reading now divides by the *test window's* mean, not the whole
  history's — the earlier version mixed two periods and flattered the ratio.
- The R² prose says "the average of the period being scored", not "the long-run average",
  and compares against the naive rule already in the table rather than naming a fixed rival —
  so it cannot go stale when selection picks a different winner.
- README's "learns the physics rather than memorizing noise" corrected: `pm25_lag_24`
  carries 0.35 of the importance, so the honest description is persistence plus a weather
  correction worth ~15% of MAE.
- **Model selection closed the loop.** `select_model` scores every candidate by rolling-origin
  CV and `train` fits the winner; `compare_models` no longer prints a winner nobody consumes.
  Selection runs on CV rather than on the held-out split on purpose — choosing the winner on
  the rows the report publishes would make those metrics a best-of-three, the same leak a
  random split causes, one level up. HistGradientBoosting now deploys (CV MAE 6.97 vs
  RandomForest 7.18 vs Ridge 8.19), by a margin narrower than the ±2.55 fold spread, and the
  page says so instead of implying the result was decisive.

## Recommendations

Ordered so that every credibility gap closes before anything cosmetic is added.

### Next — close the remaining credibility gaps

1. **Encode wind direction as u/v components.** `features.py` passes every weather column
   through unchanged; only hour/day/month get sin/cos. `wind_direction_10m` therefore enters
   as linear 0–360 with a discontinuity at north, and still ranks 4th (0.082) — above wind
   *speed* (0.040), which is itself the smell. `speed × sin/cos(direction)` is the physically
   correct encoding for advection, ~3 lines. This is the concrete instance of the
   feature-quality bottleneck the original draft gestured at without listing.

2. **Regime breakdown at the WHO 15 µg/m³ line**, plus mean error (bias). The original draft
   proposed a >25 µg/m³ threshold — that is `PM25_EU_ANNUAL`, an *annual mean* limit used as
   an *hourly* episode threshold, a category error a reviewer would catch. The page already
   draws the WHO 24h line; anchor there and add hit-rate / false-alarm for it. Bias matters
   because the model trains mostly on winter and is tested on summer: a systematic offset is
   the leading hypothesis and costs one line to test.

### Then — make the page show its work

3. **Backtest chart, last ~14 days.** Trap to avoid: the saved bundle holds the *all-data*
   model, so charting its fit over recent days is in-sample. Needs the split-trained model,
   which `train()` currently discards (`results, _ = model.run_experiment(...)`).

4. **Feature importances,** with the corrected caption from the finding above. Note the
   coupling: `HistGradientBoostingRegressor` has no `feature_importances_`, so the model choice decides
   whether this stays a one-liner or becomes `permutation_importance`.

5. **Multi-horizon with lead time as a feature.** Every serving row uses lags ≥24h from the
   same origin, so the +1h forecast is built from PM2.5 that is 23 hours old even though a
   fresh reading is in hand. Train/serve semantics are consistent so the metrics are honest,
   but the near-term hours — the ones a reader checks against their own window — are
   needlessly weak, and one MAE currently describes 24 different tasks. The MAE-vs-lead-time
   curve also *visually* settles the R² argument: high at lead 1, low at lead 24.

6. **Prediction intervals.** sklearn ≥1.5 is pinned, so
   `HistGradientBoostingRegressor(loss="quantile")` is two extra fits — but on RF it needs
   per-tree spread or a third-party quantile forest. Sequence after the model choice, now settled.

### Later

7. **Prospective forecast log,** not "metrics over time". The original framing was both
   under-costed and methodologically weak: `refresh.yml` has `contents: read`, gitignores
   `models/` and `*.db`, and re-ingests from scratch, so there is nowhere to append; and
   because the training window slides daily, "MAE over time" would track window composition
   rather than model health. Persist each day's *published* forecast and score it when the
   observations arrive. That is genuine prospective evaluation and it makes item 3 real
   rather than retrospective. Cost M, not S.

8. **Health context** — translate µg/m³ into what it means for a runner or someone with
   asthma. Low technical signal, real product signal.

9. **Polish version of the page.** Data and likely audience are Polish; code and docs stay
   English by project convention.

10. **Extract a pure `_render_page(...)` from `generate_report`.** Page composition is
    currently interleaved with network I/O and file writing, so the assembled HTML — including
    the legacy-metadata normalisation — can only be exercised with network plus a saved bundle.
    Flagged during test writing; the helper tests replicate that step and cannot catch it
    regressing.

### Re-scoped

11. **Three-station comparison is not available as described.** `docs/research/data-sources.md`
    is explicit: station 115's PM2.5 is manual (`getData` → `API-ERR-100003`, archive only,
    4–8 week delay) and station 114 has no PM2.5 at all. Options are 129+115 at mismatched
    resolution and latency in the notebook only, or a traffic-vs-background cross-pollutant
    comparison using 114's NOx. Pick one or drop the item.

## Not doing yet

Deep-learning sequence models (LSTM/temporal fusion). With ~8.5k hourly rows, a raw circular
feature, an unresolved model choice, and evaluation that until now disagreed between the
README and the published page, capacity is not the constraint. Revisit once items 1–3 land.
