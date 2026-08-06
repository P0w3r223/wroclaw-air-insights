# Report roadmap — making the Pages report speak to a non-technical reader

Date: 2026-08-06
Status: draft — second revision (recommendation #1 was measured and **rejected**)
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

## Second revision — recommendation #1, measured

*The wind-encoding item was written from a feature-importance ranking. Before implementing
it, the ranking itself was re-measured. It did not survive, and neither did the item.*

### The A/B: three encodings, same rows, same folds, same seeds

`wind_direction_10m` was re-expressed two ways and scored with the project's own
`cross_validate` (5 rolling folds, 8 584 rows, 2025-07-24 → 2026-07-17). CV MAE, µg/m³:

| Encoding | Ridge | HistGradientBoosting | RandomForest |
|----------|:-----:|:--------------------:|:------------:|
| A — raw 0–360 (current) | 8.19 | **6.97** | 7.18 |
| B — u/v components (`speed × sin/cos`) | 8.26 | 7.06 | 7.18 |
| C — `sin/cos` of direction + speed | 8.09 | 7.03 | 7.19 |

Every difference is ≤ 0.09 µg/m³ against a fold-to-fold spread of ±2.5. The physically
correct encoding is not measurably better, and for the deployed model it is marginally
worse. Even Ridge — the one candidate that genuinely *cannot* use a raw bearing, and so
should have gained the most — got **worse** under u/v. The discontinuity at north is real
and costs a gradient-boosted tree essentially nothing: it can already carve the circle with
a second split.

**Rejected.** Not deferred — measured, and it does not pay.

### The reason the item existed at all: the importance number was wrong

The roadmap's evidence — "direction ranks 4th at 0.082, above wind speed at 0.040" — came
from `RandomForestRegressor.feature_importances_`, an impurity-based score computed on
training rows. Measured instead as *degradation on held-out rows*, the ranking changes
shape (MAE increase in µg/m³, HistGradientBoosting, permutation with 10 repeats):

| Feature | RF impurity | Permutation (MAE ↑) |
|---------|:-----------:|:-------------------:|
| `pm25_lag_24` | **0.353** (1st) | **−0.016** |
| `temperature_2m` | 0.130 | 0.060 |
| `boundary_layer_height` | 0.118 | **0.358** (1st) |
| `wind_direction_10m` | 0.082 | 0.108 |
| `surface_pressure` | 0.055 | −0.006 |
| `wind_speed_10m` | 0.040 | 0.080 |

Wind direction *does* matter — that part of the premise held up, which is why re-encoding
it looked promising. What collapsed is everything else the ranking was used for.

### And a correction to the correction

The single-column permutation above says `pm25_lag_24` is worthless. That reading is also
wrong, in the opposite direction: the five PM2.5-history columns are near-duplicates, so
shuffling one leaves its information in the other four. Permuting and dropping whole
**groups** settles it (HistGradientBoosting, same held-out window, full MAE 3.64,
persistence 4.87):

| Group | MAE if permuted | MAE if retrained without it |
|-------|:---------------:|:---------------------------:|
| All weather (9 cols) | 4.98 (+1.34) | 4.39 (+0.75) |
| PM2.5 history (5 cols) | 3.84 (+0.20) | 4.36 (+0.72) |
| `boundary_layer_height` alone | 4.01 (+0.37) | 3.84 (+0.21) |
| Wind (speed + direction) | 3.81 (+0.18) | 3.82 (+0.18) |
| Calendar (7 cols) | 3.79 (+0.15) | 3.71 (+0.08) |

Weather and PM2.5 history contribute **the same amount** (+0.75 vs +0.72). Strip either one
completely and the model still beats persistence. Three methods, three orderings, and only
the last one answers the question anyone actually asks — *what would we lose without it?*

**Consequence for what is already published.** README currently says yesterday's reading
"dominates at 0.35" and the model is "persistence plus a weather correction". That sentence
was itself introduced in PR #3 as a *correction* to an earlier overclaim, and it rests on
the impurity number. On held-out rows the two halves are co-equal. It has to be fixed, and
`reports/figures/fig6_importances.png` with it.

## Recommendations

Ordered so that every credibility gap closes before anything cosmetic is added.

### Rule for anything in this list that claims an accuracy gain

State the CV MAE delta **next to the fold spread**, and treat a delta inside the spread as
a null result to be published, not buried. Item #1 above cost a few minutes to measure and
would have cost a reviewer's trust to ship on reasoning alone. The A/B harness that settled
it should become a small module rather than a scratch script, so the next feature idea is
one command, not one argument.

### Next — close the remaining credibility gaps

1. ~~**Encode wind direction as u/v components.**~~ **Rejected — measured, no gain.**
   See the second revision above. The reasoning was physically sound and empirically
   irrelevant; the write-up of *why* is worth more than the change would have been.

1a. **Publish importances the honest way.** Replace impurity ranking with held-out
    drop-column / group permutation, correct the README sentence and the figure, and keep
    the impurity column beside it — the contrast between the two is the finding. This was
    item #4 and it is now first, because a wrong claim is already live.

1b. ~~**Cross-pollutant lags.**~~ **Also rejected — measured, no gain.** Station 129 stores
    NO2 and CO (not PM10/O3/SO2 — it does not measure them), both at 100% coverage, so
    adding them costs no training rows. Two variants: NO2+CO at lag 24, and the full
    treatment PM2.5's own history gets (lags 24/48/168 + a 24h rolling mean at the origin).

    | Features | HistGradientBoosting | RandomForest |
    |----------|:--------------------:|:------------:|
    | current (21) | **6.97** | 7.18 |
    | + NO2/CO lag 24 (23) | 6.98 | 7.14 |
    | + NO2/CO full history (29) | 6.99 | 7.12 |

    ≤0.06 µg/m³ either way against a ±2.5 fold spread. The argument for this item — that it
    adds *information* rather than re-expressing it — was right in kind and wrong in
    magnitude: NO2 and PM2.5 in a city share their drivers (traffic, heating, the same
    boundary layer), so at a 24h lag NO2 mostly restates what the weather columns and
    `pm25_lag_24` already carry.

### What two consecutive nulls actually pointed at

Two feature ideas, each defensible, each worth ≤0.09 µg/m³. That is not two coincidences —
it says the feature set is not the binding constraint. The obvious suspect is the task
itself: every serving row uses PM2.5 that is at least `horizon` hours old, so at a 24h lead
the freshest observation available is a full day stale. Rebuilding the same matrix at
shorter horizons (lags floored at the horizon, so nothing leaks) and scoring the naive rule
**on the same folds** as the model:

| Lead | Naive CV MAE | Ridge | HistGradientBoosting | RandomForest | Best vs naive |
|------|:------------:|:-----:|:--------------------:|:------------:|:-------------:|
| 1 h  | **3.75** | 3.76 | 4.14 | 4.02 | −0.2% (naive) |
| 3 h  | 5.52 | **5.23** | 5.43 | 5.39 | +5.2% |
| 6 h  | 7.20 | 6.37 | **6.02** | 6.22 | +16.5% |
| 12 h | 8.52 | — | **6.56** | — | +23.0% |
| 24 h | 8.61 | 8.19 | **6.97** | 7.18 | +19.1% |

The horizon moves MAE by 2.8 µg/m³ — thirty times what any feature moved it.

*(Second revision of this table. The first version scored only HistGradientBoosting and
concluded "below about 3 hours the model loses to persistence, by 10% at a 1h lead". That
was a property of one candidate, not of the problem. Ridge ties the naive rule at 1h — it
loses by 0.008 µg/m³, a rounding error, not 10% — and beats it by 5.2% at 3h. The crossover
is at lead 1, not lead 3.)*

**The correction matters more than the number it replaced,** because of what the full table
shows: the winning family *changes with the lead*. Ridge takes 1–3h, HistGradientBoosting
takes 6h and beyond. A single model with lead time as a feature has to compromise across a
range where different model families win, and a lead feature alone cannot express "be a
linear extrapolator at 1h and a gradient-boosted tree at 24h". Item 5's proposed shape is
therefore not obviously the right one — see the entry itself.

1c. **A first correction fell out of measuring the above — now fixed.** Scoring the naive
    rule fold-by-fold required `cross_validate_baseline`, which did not exist. It did not
    exist because the project had never compared model to baseline on anything but the
    single split — while headlining the year-round CV error. The page was therefore pairing
    an all-seasons error with a summer improvement: **19.1% year-round, not the 25.4% it
    printed.** Both figures are now stored and the page states each with its period.

2. **Regime breakdown at the WHO 15 µg/m³ line, plus bias — done, and the stated hypothesis
   was wrong.** (The original draft proposed a >25 µg/m³ threshold — that is
   `PM25_EU_ANNUAL`, an *annual mean* limit used as an *hourly* episode threshold, a
   category error a reviewer would catch. Anchored at the WHO 24h line the chart already
   draws, and labelled as a reference level rather than a compliance test, since the
   guideline itself applies to daily means.)

   The prediction was "a systematic offset, because the model trains mostly on winter and is
   tested on summer". Aggregate bias is +0.91 µg/m³, which looks like a mild version of
   exactly that. Split at the line, it is not an offset at all:

   | Hours | Model MAE | Model bias | Naive MAE | Naive bias |
   |-------|:---------:|:----------:|:---------:|:----------:|
   | below 15 µg/m³ (1 280 h) | 3.30 | **+2.15** | 4.37 | +1.70 |
   | at or above 15 µg/m³ (437 h) | 4.61 | **−2.73** | 6.35 | −4.95 |

   The model runs high on clean air and low on dirty air — regression toward the mean, not a
   seasonal shift, and the two directions had been netting each other out into a single
   reassuring number. The naive rule does the same thing more strongly, which is the useful
   context: this is a property of the problem, not a defect introduced by the model.

   Detection at that line: the forecast flags **57%** of genuinely elevated hours (naive rule
   40%), and **42%** of its warnings are wrong (naive rule 60%). Better than the reference on
   both counts and nowhere near good enough to call it an alerting system — which is why the
   page states both numbers instead of only the flattering one.

   The seasonal-offset hypothesis is not baseless, it was aimed at the wrong predictor: the
   climatology row shows bias **+8.60**, a flat line trained on winter and scored on summer
   doing precisely what was predicted. Bias is now a column for every predictor in the main
   table, which is what makes that visible.

### Then — make the page show its work

3. **Backtest chart, last ~14 days — done.** The trap was correctly identified: the saved
   bundle holds the *all-data* model, so charting its fit over recent days would show it
   hours it trained on. The proposed fix — keep the split-trained model — was the wrong
   half of the answer. The chart needs that model's *output*, not the model: `run_experiment`
   already computes predictions on the held-out rows, so `backtest_series` stores the tail
   of them as parallel arrays. A few hundred floats in the bundle instead of a second
   serialised estimator, and no refit at report time.

   The naive rule is drawn alongside, because a forecast that simply repeats yesterday
   looks convincing on this chart — tracking PM2.5 a day late still tracks it. Two lines
   make that visible; one line would have made the chart decorative.

   Watch: `train` used to `json.dumps` the whole results dict, which would have printed
   several hundred hourly rows into every CI log. The bundle keeps them, the log gets a
   count.

4. ~~**Feature importances.**~~ Promoted to 1a — see above. The coupling noted here turned
   out to be the whole story rather than an implementation detail:
   `HistGradientBoostingRegressor` has no `feature_importances_`, which is why the published
   figure still shows RandomForest's — a ranking from a model that is no longer deployed,
   produced by a method that disagrees with held-out measurement.

5. **Multi-horizon with lead time as a feature — promoted, and now the largest measured
   lever in this document.** Every serving row uses lags ≥24h from the same origin, so the
   +1h forecast is built from PM2.5 that is 23 hours old even though a fresh reading is in
   hand. Train/serve semantics are consistent so the metrics are honest, but one MAE
   currently describes 24 different tasks. The sweep above prices the near-term hours at
   ~2.8 µg/m³, against ≤0.09 for the two feature ideas that were rejected.

   **Verified before starting, and it is a LARGE task standing on two untested assumptions.**

   - *The 2.8 µg/m³ is an upper bound, not a forecast.* It was measured with one specialist
     model per horizon. This item proposes one model learning 24 tasks at once, which is a
     different and normally weaker design. The number cannot be carried across without
     measuring the design that would actually ship.
   - *A lead feature may be the wrong shape.* The revised sweep shows the winning family
     changing with the lead — Ridge at 1–3h, HistGradientBoosting from 6h. One model plus a
     lead column cannot express that; per-lead selection might be the real answer, and it is
     a different implementation.
   - *Cost is bounded by one candidate.* An (origin × lead) matrix is 206 016 rows × 22
     columns. Measured, one fit: HistGradientBoosting **3.45 s**, RandomForest **90 s** on 12
     cores. Five-fold CV over the registry is ~17 s of gradient boosting against ~7.5 min of
     forest locally, and a GitHub runner has 4 vCPUs. The daily refresh job runs ~9 min today.
     RandomForest has never won a selection and loses at every lead measured above, so the
     scaling decision it forces should be made deliberately rather than discovered in a
     40-minute CI run.
   - *Blast radius.* `baseline.persistence_prediction` returns `pm25_lag_24` outright, so the
     current baseline is simply wrong for 23 of 24 leads — the per-lead baseline is a
     prerequisite, not a follow-up. Lags are computed relative to valid time `T`; an
     (origin, lead) matrix needs them relative to the origin, which rewrites `_assemble`,
     `build_features`, `build_inference_features`, `_BASELINES`, `compare_models`,
     `cross_validate_baseline`, `run_experiment`, `serving` and the report's single-MAE
     framing. Above five files and a restructure: LARGE by the project's own rule.

   Sequenced accordingly: prerequisites first (items 11–13, done), then a pilot that measures
   one-model-plus-lead against per-horizon specialists, then the restructure — or not,
   depending on what the pilot says.

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

### Correctness debts found while measuring the above — all three done

*Cleared as prerequisites for item 5: the first is required by any change to the feature
set, and the other two are the places where "which RandomForest" and "which model" had
already drifted.*

11. **A feature change silently invalidates the saved bundle — fixed.** `serving` did
    `feats[bundle["feature_names"]]`, so renaming a column made a stale
    `models/pm25_forecaster.joblib` fail with a bare pandas `KeyError` from inside the
    serving path — and `report.py` makes the same call, so it surfaced as a broken Pages
    build with no explanation. `model.align_features` now raises `FeatureMismatchError`
    naming what the model wanted, what the builder produces instead, and the retrain
    command. Verified against the real saved bundle: it passes today's contract and fires
    correctly when `wind_direction_10m` is renamed — the exact scenario item 5 would cause.

12. **Two definitions of the same RandomForest — fixed.** `train_forecaster` restated
    `n_estimators=300, min_samples_leaf=2`, which `build_models` also owns. Since
    `select_model` landed, only `build_models` reaches anything shipped, so the copy was a
    definition nothing read, free to drift. `train_forecaster` now takes the candidate from
    the registry and keeps its one real job: giving the notebook a concrete forest, the only
    estimator here that exposes impurity importances.

13. **`pipeline compare` hardcoded `"RandomForest"` — fixed.** It cross-validated one named
    model while `train` deployed whatever `select_model` picked. It now reports CV for every
    candidate, names the winner, and prints the naive rule scored on the same folds.

### Re-scoped

14. **Three-station comparison is not available as described.** `docs/research/data-sources.md`
    is explicit: station 115's PM2.5 is manual (`getData` → `API-ERR-100003`, archive only,
    4–8 week delay) and station 114 has no PM2.5 at all. Options are 129+115 at mismatched
    resolution and latency in the notebook only, or a traffic-vs-background cross-pollutant
    comparison using 114's NOx. Pick one or drop the item.

## Not doing yet

**Hyperparameter tuning.** Every candidate runs at library defaults —
`HistGradientBoostingRegressor()` untouched, RandomForest at 300 trees chosen by hand,
Ridge at `alpha=1.0`. This is a deliberate omission and should be stated as one rather than
left to look like an oversight: tuning on the same folds that already pick the winner would
make the CV figure a best-of-many, exactly the leak `select_model` was written to avoid one
level up. Doing it properly means nested CV — an inner loop to tune, an outer loop to score
— which multiplies fit count by the inner grid size. Given that the gap between first and
second place is already narrower than the fold spread, the expected gain is small and the
expected *reported* gain is optimistic. Revisit when there is a reason to believe the
defaults are the binding constraint; the measurement above suggests features, not
capacity, decide this problem.

**Deep-learning sequence models (LSTM/temporal fusion).** With ~8.5k hourly rows and a model
whose signal splits evenly between five weather columns and five lag columns, capacity is
not the constraint. The original entry also listed "a raw circular feature" as a reason to
wait — that reason is now retired, measured, and it was never costing anything. Revisit once
items 1a–2 land.
