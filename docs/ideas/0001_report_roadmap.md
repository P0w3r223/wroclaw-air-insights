# Report roadmap — making the Pages report speak to a non-technical reader

Date: 2026-08-06
Updated: 2026-08-11
Status: draft — sixth revision (item 5 **complete**: phase 1 measured, gated and served on a
re-measured band, whose gate then failed on fresher data — the null ships and the machinery
stands; item 6 **complete**: three interval constructions measured, one publishable, bundle
schema 4; item 7's forecast log **live** since 2026-08-08, now grading interval coverage too;
both published nulls re-measured under the paired rule that replaced the retracted fold-spread
test; item 14 dropped; item 15 **complete** — the page made legible, and the four defects that
took, measured at a real phone viewport. Every numbered item is measured, shipped or dropped.
What is open now comes from the running system rather than from this plan: see **Next**, where
item 16 gates the rest.)
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

Every difference is ≤ 0.10 µg/m³ against a fold-to-fold spread of ±2.5. The physically
correct encoding is not measurably better, and for the deployed model it is marginally
worse. Even Ridge — the one candidate that genuinely *cannot* use a raw bearing, and so
should have gained the most — got **worse** under u/v. The discontinuity at north is real
and costs a gradient-boosted tree essentially nothing: it can already carve the circle with
a second split.

*(Correction, on re-checking this table for the published page: the bound above read ≤ 0.09,
which the table refutes — Ridge under sin/cos moves 8.19 → 8.09, a full 0.10, and it is the
largest single effect here. It is also the one movement in the **helpful** direction, on
exactly the candidate the physical argument predicted would gain, which is why the scope
"worse **under u/v**" matters and a blanket "got worse" would be false. The conclusion is
unchanged — 0.10 against ±2.5 is still a null — but the number was wrong in the record and
had been copied into the README from here.)*

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

## Third revision — both published nulls re-measured under the rule that replaced the retracted one

*The two nulls above were judged by comparing their delta to the ±2.5 spread of MAE levels.
Phase 0 of item 5 retracted that test and left this note behind: "the two published nulls were
judged under the old reading and have not been re-measured paired; they stand as unrefuted,
not as confirmed." This closes that debt, and building the harness to do it is what the
"Rule" entry above asked for in the first place.*

`forecast/ab.py` scores feature-set variants on **identical rows** (intersected before
scoring, because a variant that adds a column can lose hours to `dropna` and would otherwise
be graded on an easier set), **identical folds**, and reports the paired difference per
candidate estimator. Reproduction check against the tables above: **14 of the 15 published
cells come back within 0.01 µg/m³**, including every linear one. The exception is
`HistGradientBoosting` under u/v — 6.982 here against the published 7.06.

*What accounts for that 0.078, measured rather than asserted.* The first draft of this entry
blamed column order: the original scratch script did not record it and a histogram-binned tree
is mildly sensitive to it. Measured over 12 random permutations of that exact frame, the cell
spans 6.982–7.016 — **0.034, well under half the gap** — and the alternative variant shape
(dropping wind speed) puts it at 7.109, so that is not it either. Column order is part of the
answer and not the whole of it; the likeliest remainder is a different scikit-learn/numpy
version in the original script, which is not recoverable. Stated as far as it was checked,
because asserting a cause a quick check only half-supports is the failure mode this project's
own rule targets. No verdict turns on it — that cell is a null under either figure.

*One reconstruction detail was load bearing and nearly got it wrong.* "u/v components" reads
naturally as *replacing* speed and direction, since u and v carry the magnitude between them.
The published table was measured with speed **kept** as its own column: reconstructed the tidy
way, Ridge lands on 8.32 rather than the 8.26 printed above. A re-measurement of a published
verdict has to score the thing that was published, so the harness reproduces the original shape
and says so in a comment.

CV MAE µg/m³ and the paired verdict, 8 584 rows, 5 folds, same window as above:

| Experiment | Variant | Ridge | HistGradientBoosting *(deployed)* | RandomForest |
|------------|---------|:-----:|:--------------------------------:|:------------:|
| wind | raw *(reference)* | 8.189 | **6.966** | 7.182 |
| wind | u/v | 8.260 · null | 6.982 · null | 7.173 · null |
| wind | sin/cos | **8.092 · improvement** | 7.030 · null | 7.189 · null |
| pollutants | current *(reference)* | 8.189 | **6.966** | 7.182 |
| pollutants | + NO2/CO lag 24 | 8.151 · null | 6.984 · null | **7.137 · improvement** |
| pollutants | + NO2/CO full history | 8.535 · null | 6.985 · null | 7.123 · null |

The two reference rows are identical here, and that is a fact about this station rather than a
duplicated row: NO2 and CO have full coverage, so adding them drops no hours and both
experiments intersect to the same 8 584. A pollutant with gaps would separate them, which is
why `align_variants` intersects per experiment instead of assuming one shared row set.

**Both shipped decisions are unchanged.** Nothing wins consistently for the deployed model in
either experiment — the folds disagree in both directions on all four of its variants. The
nulls were right about what ships.

**Both were wrong as blanket statements, and in the same direction.** Two cells sweep every
fold that separates them, and neither is the deployed model:

- **sin/cos on Ridge: +0.097 µg/m³, better on 4 folds of 5, one tie, none worse.** This is the
  candidate the physical argument named *in advance* — the one that structurally cannot read a
  bearing. The old write-up already noticed the movement ("the largest single effect here, and
  it is in the helpful direction, on exactly the candidate the physical argument predicted
  would gain") and then dismissed it with the retracted test. Under the paired rule it clears
  the bar — on a candidate trailing the deployed model by 1.13 µg/m³, so it changes no decision
  either way. What it does refute is the blanket wording: "the discontinuity costs nothing" was
  too strong. It plausibly costs the linear model something, and only the linear model.

  *How much this can carry on its own: not much.* Four wins, one tie, none against is a sign
  test at p ≈ 0.06 before any multiplicity adjustment — about the most five folds are able to
  show. The claim worth making is the scoped one (*who* it affects), not a quantified gain.
- **NO2/CO at lag 24 on RandomForest: +0.044 µg/m³, better on 5 folds of 5.** No fold against,
  worth four hundredths, on a candidate that has never won a selection — and predicted by
  nothing in advance. See the multiplicity note below: this is the one that reads as noise.

**The multiplicity caveat, and the first version of it was wrong in the flattering direction.**
The verdict rule asks that no fold *contradict* the direction; under a change that does nothing
and never ties, each fold's sign is a coin flip and a clean sweep is a **1-in-16** event at five
folds. Twelve comparisons then expect 0.75 sweeps from noise, "and it found two" — which is how
this entry originally read, and it made the two survivors look like more than chance provides.

**Ties raise the sweep rate, they do not lower it,** because a tie is not a contradiction: it
removes a chance to fail. With per-fold tie probability `t` the rate is
`2 · (((1+t)/2)^n − t^n)`, above `2/2^n` for every `t > 0`. This run tied **8 of its 60 folds**,
so the rate is 0.117 per comparison and twelve comparisons expect **~1.4** sweeps. It found two.
Verified analytically and by simulation (0.1164 against 0.1168 predicted), and it is still a
*floor*: rolling-origin folds share training rows, so their signs are positively correlated,
which pushes it higher again.

So the count of survivors is what chance gives, and neither result is evidence on its own. The
only thing that separates them is what was predicted beforehand — the Ridge result was named in
advance by a physical argument, the RandomForest result by nothing at all — and even the Ridge
one is 4 of 5 with a tie, about the most five folds can show. `ab.expected_by_chance` takes the
run's own tie rate and `pipeline ab` prints a run-level line after the last table, because a
reader picking a survivor is choosing from every comparison the run made, not from one table's
six.

**One consequence for the methodology, and it argues against a change rather than for one.**
The A/B verdict rule is deliberately *stricter* than the serving policy's: `horizon` hands a
lead to whichever predictor wins a majority of folds, because some predictor has to answer that
hour; a feature change can simply not happen, so there the bar is that no fold contradicts it.
The two disagree on real cases here — NO2/CO at lag 24 under Ridge wins 3 folds of 5 and loses
one, which the serving rule would call a win and the A/B rule calls a null. Sharing one function
between them was tried and reverted; the docstrings on both now say why they differ.

## Recommendations

Ordered so that every credibility gap closes before anything cosmetic is added.

### Rule for anything in this list that claims an accuracy gain

~~State the CV MAE delta **next to the fold spread**, and treat a delta inside the spread as
a null result.~~ **Retracted — see the third revision below.** That test compares against the
spread of MAE *levels*, which is seasonal and common to every predictor scored on those folds;
by it, this project's own headline is a null. The rule is now the mean and sign of the per-fold
*difference* (`model.paired_delta`), and a delta that changes sign across folds is the null.

The rest of the entry stands. Item #1 cost a few minutes to measure and would have cost a
reviewer's trust to ship on reasoning alone, and the A/B harness that settled it is now a
module rather than a scratch script — `forecast/ab.py`, run as `pipeline ab`. The next feature
idea is one command, not one argument.

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

   ### Pilot result — the proposed shape is refuted, the lever is real

   The (origin × lead) matrix was built for real (206 016 rows before dropna, 22 columns) and
   both designs were scored on **identical rows and folds**: one model fitted across all 24
   leads, versus one model per lead, versus the reading at the origin.

   *A leak had to be closed first, and it is specific to this design.* Rows from neighbouring
   origins share target hours — training row (origin `T`, lead 24) is labelled with the
   observation at `T+24`, and test row (origin `T+1`, lead 23) is scored on that same hour. A
   plain chronological cut trains on a label the test set is about to be graded against. Each
   fold therefore embargoes training rows whose target time reaches the first test origin
   (276 rows per fold). This does not arise in the current single-horizon design, where train
   and test targets are an hour apart and never coincide.

   Mean CV MAE, µg/m³:

   | Lead | Naive | Ridge pooled | Ridge per-lead | HGB pooled | HGB per-lead |
   |------|:-----:|:------------:|:--------------:|:----------:|:------------:|
   | 1 h  | **3.750** | 6.122 | **3.751** | 6.254 | 4.088 |
   | 3 h  | 5.517 | 6.475 | **5.271** | 6.393 | 5.386 |
   | 6 h  | 7.202 | 6.862 | 6.458 | 6.531 | **6.107** |
   | 12 h | 8.514 | 7.287 | 7.285 | 6.688 | **6.526** |
   | 24 h | 8.609 | 7.718 | 8.217 | **6.817** | 7.122 |

   **One model with a lead feature is the wrong design.** At lead 1 it lands at 6.1–6.3
   against the naive rule's 3.75 — 63% *worse* than doing nothing at all. Pooling forces one
   set of parameters onto tasks whose relationship to the inputs is not the same: at lead 1
   the answer is nearly "copy `pm25_origin`", at lead 24 that feature is almost noise, and a
   lead *column* cannot express the interaction. Ridge structurally cannot; a tree can in
   principle, so capacity was the obvious objection and it was tested — 255 leaves and 500
   iterations instead of 31 and 100 made the pooled model **worse at every lead** (lead 1:
   6.54, lead 24: 7.06). Overfitting, not underfitting. The weakness is structural.

   **The lever is real, and it belongs to specialists.** Per-lead, the near hours improve from
   the ~6.97 a single 24h model delivers at every lead to **3.75 at lead 1** and 5.27 at 3h —
   the 2.8 µg/m³ the sweep priced, collected by a different mechanism than item 5 proposed.

   **Two findings to carry into the implementation.** First, at lead 1 the best model *ties*
   the naive rule (3.751 vs 3.750): the honest product decision for the first hour or two is
   to serve persistence and say so, not to serve a model that merely matches it. Second,
   pooling wins at lead 24 (6.817 vs 7.122) — the hardest task benefits from the extra rows
   that related leads provide. So the endpoint is probably neither pure design but per-lead
   selection over {naive, per-lead model, pooled model}, which the existing `select_model`
   machinery already expresses, one lead at a time.

   ### Phase 0 — shipped, and the free half of the lever collected

   An architecture pass on the pilot disagreed with this item in three places, and the
   disagreements reshaped what got built.

   **The headline number priced the wrong quantity.** 2.8 µg/m³ is the span between the
   model's easiest task (lead 1) and its hardest (lead 24) — a measure of *task difficulty*,
   not of achievable improvement, and not comparable to a feature delta measured on a fixed
   task. The comparable quantity is today's constant error minus the best available
   predictor at each lead, averaged over the 24 hours the page publishes: **≈0.70 µg/m³**.
   Still ~8× the largest feature delta, still the biggest lever here, and a third of what
   this item claimed.

   **That 0.70 splits in two, and one half is nearly free.** ≈0.30 comes from serving the
   naive rule where it already wins — no new estimator, no per-lead fitting; ≈0.40 needs the
   per-lead specialists. Phase 0 is the first half.

   *"Nearly", not "zero", and the difference was worth catching.* Scoring the lead axis
   needs the winner's held-out predictions, and `select_model` discards the ones it already
   computed — so `horizon.measure` refits the winner once per fold: ~15 → ~20 fits per daily
   run, a handful of seconds of gradient boosting on a job that takes ~9 minutes. Accepted
   rather than plumbed around, because having `select_model` hand back per-fold prediction
   arrays would put non-serialisable numpy into the object that becomes bundle metadata.

   **The blast radius was overstated.** The leakage rule is that `pm25_lag_k` is legal for
   row `(O, l)` iff `k ≥ l`. With leads 1–24 and lags (24, 48, 168), *every existing lag is
   already legal at every lead* — nothing has to change coordinate systems, and Phase 0 needs
   no change to the training matrix at all.

   **The sharpest fact came out of the code rather than the measurements.** Every feature is
   anchored at the valid time `T`; the lead is not an input and cannot be, so the rows for
   lead 1 and lead 24 at the same `T` are the *identical row*. The model's per-lead error is
   therefore not approximately flat — it is **exactly** constant. Confirmed empirically: 6.966
   / 6.967 across all 24 leads, varying only where a missing origin observation changes which
   rows are scored.

   Measured on the project's own folds, model and naive rule on identical rows:

   | Lead | Model | Naive (reading at issue) | Paired Δ | Folds won | Served by |
   |------|:-----:|:------------------------:|:--------:|:---------:|:---------:|
   | 1 h  | 6.97 | **3.75** | −3.22 | 0/5 | naive |
   | 3 h  | 6.97 | **5.51** | −1.45 | 1/5 | naive |
   | 5 h  | 6.97 | **6.73** | −0.24 | 2/5 | naive |
   | 6 h  | 6.97 | 7.20 | +0.23 | 2/5 | naive |
   | 7 h  | 6.97 | 7.54 | +0.57 | 3/5 | model |
   | 12 h | 6.97 | 8.51 | +1.54 | 4/5 | model |
   | 24 h | 6.97 | 8.61 | +1.64 | 5/5 | model |

   **The crossover is lead 6: a quarter of the published chart was worse than doing nothing.**
   Those hours now serve the current reading, labelled as such on the page and in the chart.

   Lead 6 is the entry worth keeping: the model is ahead *on average* (+0.23) and loses three
   folds of five. Under the rule this document used to apply — compare the delta to the ±2.55
   fold spread — that is a null on a technicality. Under the paired rule it is a decision:
   a mean that holds in two weeks out of five does not earn the slot.

   **A review caught the page stating the blunt version of that.** The served-hours sentence
   was templated off the crossover alone and read "over that range the model is measurably
   worse than doing nothing" — directly contradicting the `+0.23` printed two rows above it,
   and wrong on exactly the case the design exists to handle. The sentence now derives from
   the boundary record: outright loss and mean-win/fold-loss get different prose. Two more
   from the same review, both the same shape — a claim the page's own numbers refute:
   `_selection_note` was still arguing from the ±fold spread this change declares refuted,
   a few hundred bytes from the corrected argument; and "this page used to print a single
   error figure for all 24 hours" was past tense about something the page still does.

   **The prefix rule is not justified by monotonicity, and the curve proves it.** The
   docstring claimed the naive error grows with the lead so a single crossover is monotone by
   construction. Measured: naive MAE peaks at lead 18 (8.93) and falls back to 8.57 by lead
   23, because "the reading now" re-aligns with the same hour of day as the lead approaches
   24 — and the fold tallies wobble too (leads 10/11/12 give 4/5/4). The rule stands on its
   second reason alone, which was always the real one: 24 independent argmins would be a
   best-of-N on the folds that also produce the published figures, and the served range has
   to be contiguous for the page to describe itself in a sentence.

   **A fold "win" now needs to be visible.** `paired_delta` counted any positive difference as
   a fold won, and the selection path pairs on fold MAEs already rounded to three decimals —
   so gradient boosting was credited with beating Ridge 5 folds of 5, one of which separates
   them by **0.003 µg/m³**, below the precision the page prints. Ties are now their own
   category, anchored on that published precision, and the tally carries the narrowest fold
   beside it. The corrected claim is 4 of 5 with one tied.

   **The methodology rule was wrong and is now fixed** (`CLAUDE.md`). "A delta inside the fold
   spread is a null" tests against the spread of MAE *levels*, which is seasonal and common to
   every predictor on those folds — by that test this project's own headline is a null
   (8.61 − 6.97 = 1.64, inside ±2.55). The correct statistic is the mean and spread of the
   per-fold *difference*. Two consequences landed immediately: the headline claim is now
   *measured in code* rather than assumed — `train` stores `cv_paired_vs_baseline`, and the
   model beats the naive rule on 5 folds of 5, closest margin 0.20 — and the model-selection
   prose was corrected. HistGradientBoosting beats RandomForest 4/5 and Ridge 4/5 with one
   tie, so "the top two are close" was the right conclusion from the wrong reason. The two
   published nulls were judged under the old reading and have not been re-measured paired;
   they stand as unrefuted, not as confirmed.

   Shipped: `config.FORECAST_LEADS` / `CV_SPLITS`, `features.observations_at_origin`,
   `model.paired_delta` / `fold_indices` / `cross_validate_predictions` (one set of fits
   serves all 24 leads — refitting per lead would produce 24 identical estimators),
   `forecast/horizon.py` (scoring, prefix crossover, `apply_policy`), bundle **schema 2**
   carrying the serving policy with a named error on older bundles, `serving` returning
   `lead` and `source`, and the report's MAE-vs-lead chart plus its section. Suite 211 → 262.

   Phase 1 — the per-lead specialists worth the other ≈0.40 — is **gated**: the paired delta
   of specialist against `max(today's model, naive)` must be positive on ≥4 folds of 5 across
   a majority of leads, or the null gets published and Phase 0 stands alone.

   **Cost, with the RandomForest question settled by the same numbers.** Per-lead selection is
   3 candidates × 5 folds × 24 leads = 360 fits. At ~8.5k rows each that is ~2 min without
   RandomForest and ~8 min with it, on top of a daily job that runs ~9 min today — for a
   candidate that wins no lead in the table above. Drop it from the multi-horizon path.

   ### Phase 1 — measured, and the gate clears

   *`forecast/specialists.py`, `pipeline specialists`. This entry is the **measurement**, and it
   is kept as it was written — when it landed, serving was a separate change and nothing on the
   page moved. That change has since shipped; see "Phase 1 — served" below.*

   **The gate had to be reformulated before anything could be measured against it.** As written
   above it compares the specialist to `max(today's model, naive)` — but that max is taken over
   the same folds that then score the comparison, so the reference is the better of two noisy
   estimates and reads better than it will behave. A winner's curse, one level down from the
   one `select_model` exists to avoid.

   The fix needs no nested cross-validation: require the specialist to beat **both fixed
   references separately** at the same lead, on the same rows and folds. That is at least as
   strong as beating their maximum and it selects nothing on the rows it scores. A lead where
   the references disagree is exactly where the max would have been *chosen* — now both bars
   have to be cleared instead of the taller one being picked after the fact.

   **The lever, mechanically.** Today's model is trained on the 24-hour task, so its freshest
   input is `pm25_lag_24`: at a one-hour lead it forecasts from a reading a full day old while
   a fresh one sits in hand. A specialist for lead `l` may legally use `pm25_lag_l` — that
   reading. The matrix is built by the *existing* feature builder with `horizon=l` and the lag
   set floored at `l`, so the rolling window ends at the origin too. Nothing is re-implemented;
   a second feature path is how a leak gets in.

   HistGradientBoosting, 5 folds, ~8.57k rows per lead, all three predictors on identical rows:

   | Lead | Specialist | Deployed model | Naive | vs model | vs naive | Earns it |
   |------|:----------:|:--------------:|:-----:|:--------:|:--------:|:--------:|
   | +1 h  | 4.138 | 6.998 | **3.750** | +2.860 (5/5) | −0.388 (2/5) | no |
   | +3 h  | 5.427 | 6.998 | 5.517 | +1.571 (5/5) | +0.090 (2/5) | no |
   | +4 h  | 5.605 | 7.058 | 6.217 | +1.454 (5/5) | +0.613 (3/5) | no |
   | +5 h  | **5.875** | 7.069 | 6.735 | +1.194 (5/5) | +0.860 (4/5) | **yes** |
   | +6 h  | **6.017** | 7.010 | 7.202 | +0.993 (5/5) | +1.185 (5/5) | **yes** |
   | +12 h | **6.556** | 7.025 | 8.515 | +0.469 (4/5) | +1.959 (5/5) | **yes** |
   | +17 h | **6.686** | 7.017 | 8.911 | +0.331 (5/5) | +2.225 (5/5) | **yes** |
   | +18 h | 6.863 | 6.991 | 8.935 | +0.128 (3/5) | +2.072 (5/5) | no |
   | +21 h | 7.050 | 6.982 | 8.747 | −0.068 (2/5) | +1.697 (5/5) | no |
   | +24 h | 6.966 | 6.966 | 8.607 | **+0.000 (0/5)** | +1.640 (5/5) | no |

   **Gate: PASS — 16 leads of 24 clear both references on ≥4 folds of 5, against 13 needed.**

   **The +24 h row is the control, and it is what makes the rest readable.** At the full horizon
   the specialist matrix *is* today's matrix, so the two are the same model on the same rows.
   The measured difference there is exactly **0.000 on 0 of 5 folds** — every fold a tie. An
   instrument that produced anything else there would be reporting its own noise, and no other
   row could be believed. Pinned by a test rather than observed once.

   **The shape is the evidence, not any single tally.** This run makes 48 paired comparisons,
   and by this document's own multiplicity rule a handful of those clear a bar by chance. What
   chance does not produce is a *monotone decay from +2.86 at lead 1 to exactly 0.00 at lead 24*
   with a known mechanism behind it: `pm25_lag_l` converges on `pm25_lag_24`, which the
   incumbent already has, so the specialist's advantage has to vanish precisely where it does.

   **What is worth serving, and it is a contiguous band.** Phase 0's argument against 24
   independent argmins applies unchanged — that would be a best-of-N on the folds that also
   produce the published figures. The longest unbroken run of leads clearing both bars is
   **+5 h to +17 h**, 13 leads. Against today's policy that is worth **0.330 µg/m³ averaged over
   the 24 published hours** — the roadmap priced phase 1 at ≈0.40, so the estimate was fair and
   slightly optimistic. Extending the band to +24 h would add **0.022**, which is why the band
   is the right shape rather than a compromise.

   Below +5 h the naive rule keeps its hours: the specialist reaches 4.138 at lead 1 against
   persistence's 3.750 and loses outright. Phase 0's prefix was not made redundant by phase 1,
   which is the second thing worth carrying — the freshest observation is *most* of the answer
   at short leads, and a model that also has it still cannot beat simply repeating it.

   **Not shipped by this change.** Serving 13 specialists means a bundle carrying 13 estimators,
   a per-lead selection path, a schema break, and a report that can describe three bands rather
   than two. That is the next change, and it should be sequenced with item 6 in mind — a
   specialist per lead is also what makes per-lead prediction intervals cheap.

   ### Phase 1 — served

   *Bundle **schema 3**, `horizon.serving_policy` on three sources, `serving.specialist_predictions`,
   and a third band on the page. The measurement above did not move; what changed is that it now
   decides what a reader sees.*

   **The band is re-measured every run, which is what made this expensive rather than hard.**
   Which leads a specialist earns moves with the data exactly as the crossover does, so the
   band the page publishes has to be the band that run measured — this document's own rule
   against asserting a movable decision as a number. That puts the full phase 1 measurement
   inside `train`: ~10 fits per lead, 24 leads, **~80 s locally on 12 cores**, and it is now the
   most expensive step in the command. Accepted rather than cached, because a cached band is a
   number in a file that no longer has a measurement behind it.

   **The one decision the measurement did not settle: what happens where the two bands
   overlap.** They do overlap — this is not hypothetical. The crossover falls at 6 and the band
   starts at 5, so leads 5 and 6 are claimed by both phase 0 (naive) and phase 1 (specialist).
   The bars are not equally strong, and that is what resolves it. The prefix rule only asks that
   the *incumbent* fail to win a majority of folds against the naive rule — it is a bar the
   incumbent fails, not one the naive rule passes. The gate asks the specialist to beat both
   references **separately** on ≥4 folds of 5, which includes beating the naive rule on those
   very hours. Leaving them to the naive rule would let the weaker measurement overrule the
   stronger one on the same hours, so the band wins and the naive prefix shrinks to what is
   below it. A specialist never takes an hour from the naive rule without having beaten it there.

   On the window phase 1 was measured on — 8 584 rows ending 2026-07-17 — the policy comes out
   in three bands, and the overlap is visible in it: the crossover is 6, the band starts at 5,
   so the naive rule keeps 1–4 rather than 1–6.

   | Hours | Served by | Why that predictor |
   |-------|-----------|--------------------|
   | +1 h → +4 h | the reading at issue | the model loses outright; the specialist does too |
   | +5 h → +17 h | a specialist per lead | beat both references on ≥4 folds of 5, every hour |
   | +18 h → +24 h | the 24-hour model | the specialist's edge has decayed into it |

   Thirteen estimators, each stored with the lag set it was fitted on, take the bundle from
   ~0.4 MB to **5.5 MB**. Storing the lag set is not redundancy: a specialist for lead `l` was
   trained on a matrix built with `horizon=l` and lags floored at `l`, and a serving path that
   re-derived that rule from the lead would be a second copy of the contract, free to drift
   from the one the estimator actually saw. The bundle records the recipe; serving reads it back.

   ### And then the gate failed — which is the most important result in this entry

   *The table above is not what the project serves today. It is what the phase 1 window
   produced, and it is kept because the contrast is the finding.*

   Re-running the whole thing on data ingested **twenty-four days later** — 8 558 rows ending
   2026-08-10, the same station, the same code, the same bar:

   | Window ends | Rows | Leads clearing the gate | Verdict | Band served |
   |-------------|-----:|:-----------------------:|:-------:|:-----------:|
   | 2026-07-17 | 8 584 | **16** of 24 | pass | +5 h → +17 h |
   | 2026-08-10 | 8 558 | **7** of 24 | **fail** | none |

   **Phase 1's headline does not replicate.** It was measured once, on one window, and reported
   as a pass with 16 leads clearing against 13 needed — a comfortable-looking margin. Three
   weeks of fresher data and the same measurement clears 7. Nothing about the mechanism changed;
   the mechanism was never the uncertain part. What moved is an estimate made on five folds of
   one year of a single station, and the honest reading of the two runs together is that phase
   1's gain is somewhere around the bar rather than clearly above it.

   **This is the outcome the design was built for, and it is worth being precise about why.**
   The roadmap fixed the bar *before* the measurement and wrote down what a failure would mean:
   "or the null gets published and phase 0 stands alone". Because the band is re-measured on
   every run rather than written into the code, the failing run did what the passing run would
   have done in reverse — it published the null, served the two-band policy, and said so on the
   page. Had the band been hardcoded from the passing run, the project would today be serving
   thirteen specialists on a station where its own gate no longer clears, with a page asserting
   they earned it.

   **What ships is therefore the machinery plus the gate, not the band.** That is a weaker claim
   than "specialists improve the forecast" and it is the one the evidence supports. The page
   states the null in the same place it would have stated the band, with the tally that produced
   it, because a column of specialist figures sitting beside no sentence invites a reader to
   draw the conclusion the run declined to draw.

   *One caveat on the comparison itself, since it is a comparison this document would demand of
   anything else.* The two runs differ in more than their end date: the later window drops the
   oldest three weeks as well as adding the newest, so this is not "the same data plus more". It
   is the same *procedure* on the window the pipeline would actually have used on each day, which
   is the quantity that matters for a system that retrains daily — but it is not a controlled
   experiment on window length, and no claim here rests on which of the two effects dominates.

   **Both replacements degrade to the incumbent, and the label degrades with them.** A naive
   hour needs an origin observation and a specialist hour needs its own matrix to assemble;
   thirteen estimators are thirteen chances for one hour to be unbuildable on a station gap.
   Falling back to the model is right — a published page beats a uniform one — but publishing
   the model's answer under a *label* naming a predictor that did not produce it is the one
   failure the `source` column exists to prevent, so the row is relabelled, not just refilled.

   ### The verification found a live defect in the serving path, and it had nothing to do with specialists

   Running `predict` against the fresh bundle printed **every lead as model-served**, while
   `train` had just announced that the naive rule serves leads 1–4. The policy was correct and
   the serving path was quietly discarding it.

   The cause is one line and it predates this change. `clean` reindexes PM2.5 onto a continuous
   hourly grid and interpolates only *interior* gaps, so a series whose newest slot has been
   published-but-empty ends in `NaN`. `predict_next_24h` took the origin as
   `pm25["timestamp"].max()` — the last hour on the *grid*, not the last hour with a *reading*.
   Three things then went wrong at once, none of them visibly:

   - every naive-served lead degraded to the model, because there was no reading to repeat, so
     **the entire phase 0 policy stopped applying** with nothing saying so;
   - the leads were numbered from an hour that does not exist as an observation, so "+1 h" was
     really +2 h;
   - the page's freshness note derives the anchor back out of `timestamp - lead`, so it would
     have named an empty hour as "the reading" the forecast is built on — a false sentence on
     the published page, which is the one thing this project's rules forbid outright.

   The origin is now the newest hour that actually has a value, and the history is truncated
   there so the trailing empty slots cannot collide with the future hours the inference builder
   appends. Nothing is lost: every feature is by construction knowable at the origin.

   **The forecast log had already recorded this happening in production, and it was misread.**
   The 2026-08-09 entry has 23 rows and not one naive-served hour, which the log entry above
   reads as "the naive prefix collapsed to zero — the boundary is re-measured every run and one
   run put it at the very start". That explanation is wrong. It is this defect: an empty newest
   slot, every naive hour degrading to the model, and a missing row at the far end. The log
   found a real fault three days before anyone went looking, and the first reading of it
   attributed the fault to the thing the log was built to measure.

   **The change surfaced one more defect, and that one was in a sentence rather than in the code.**
   The section's opening claims the model's error line is flat, which is true in the sense that
   matters — the lead is not an input. Adding the band edges to the printed table put +17 h on
   the page at **6.96** beside +24 h at **6.97**, so the page asserted "the same error at every
   hour" directly above a column that visibly disagreed. The cause is real and small: a lead
   whose origin reading is missing drops that hour from every predictor at once, so the mean is
   taken over slightly different rows. The page now says so **only when the rows it printed
   actually differ** — explaining a discrepancy the reader cannot see would be its own kind of
   noise. This is the fourth review round in this project to find the same defect class, and the
   first where the trigger was a *new row* rather than new prose.

6. **Prediction intervals — built, checked, and mostly withheld.** The entry as written
   proposed the construction (`HistGradientBoostingRegressor(loss="quantile")`, two extra
   fits) and one condition: publish only with a coverage check. The construction was the easy
   part. The condition is what the result turned out to be.

   **An interval is the only claim on this page a reader can falsify.** "The model is good" is
   not checkable; "80% of measured hours land inside this band" is, on hours the band was not
   fitted on. So the check runs on the same rolling folds as everything else, and a band ships
   only if it comes back close to what it promises.

   `forecast/intervals.py` measures three constructions against one bar:

   | Interval | Coverage | Per-fold / per-lead | Width | Verdict |
   |----------|:--------:|:-------------------:|:-----:|:-------:|
   | Model — quantile regression | 0.581 | 0.51 – 0.65 | 12.8 | withheld |
   | Model — conformal band from its own held-out errors | 0.853 | 0.70 – 0.92 | 24.2 | withheld |
   | Naive rule — per lead, from the record | 0.762 | 0.745 – 0.797 | 22.1 | **published** |

   *(Dated record, 8 558 rows ending 2026-08-10. Nominal 0.80 throughout.)*

   **Measuring only quantile regression would have produced a claim about a library, not about
   the problem.** It is the obvious tool and it covers 58% of hours while labelled 80% — a band
   that reads as precision and is not. Publishing that as "prediction intervals do not work
   here" would have repeated this project's own mistake of arguing from one measurement, so the
   standard alternative was measured too: calibrate on the *previous fold's* held-out residuals
   and apply to the next, which is split conformal prediction with time supplying the split.
   That is genuinely out of sample, costs no extra fits, and it is honest about resting on one
   fold fewer than every other figure on the page, since the first fold has nothing to
   calibrate from.

   **The conformal band is the interesting failure.** It averages 0.853 — it misses the
   tolerance by three thousandths, which on its own would be a coin-flip verdict not worth
   defending. What disqualifies it is the spread: **0.70 on one fold and 0.92 on another**. That
   is not an 80% interval that wobbles, it is a band that is too narrow in one season and too
   wide in another, and the average conceals exactly that. So the gate gained a second
   condition — no single fold, and for the per-lead construction no single lead, further than
   0.10 from nominal — which is the same rule this project already applies to every other
   comparison: **judge it fold by fold, not on the mean.** Adding it after seeing the numbers is
   worth flagging; it changes no verdict, since the conformal band fails the average check
   independently, and the quantile band fails it by 0.22.

   **What passes is the naive rule's band, and the reason it passes is the reason it exists.**
   It is anchored on the reading *in hand*, so it moves with the level of the air instead of
   carrying a fixed width learned from a training window that has since drifted. The model's
   bands are fitted on the past and applied to a later distribution; that is where both of them
   lose. And the per-lead construction confirmed, on the same folds, the physical argument it
   was built from: the band **does** widen with the lead — 11.3 µg/m³ at +1 h against 23.6 at
   +12 h. Two feature ideas in this document were physically sound and empirically null, so
   this one is stated as measured rather than as obvious.

   **The page therefore draws a band over four hours of twenty-four and says so**, lists the two
   constructions that failed with their numbers, and states plainly that a range covering far
   fewer hours than its label promises is worse than no range. That is a thinner deliverable
   than "the forecast now has uncertainty bands", and it is the one the evidence supports.

   **The log grades it too**, which is the part cross-validation structurally cannot do. Every
   published band is written to the forecast log with the row it belongs to, and `score-log`
   reports out-of-sample coverage by predictor. An 80% band is a promise about hours that had
   not happened when it was drawn; the log is where those hours arrive.

   *One defect found by looking at the live output.* The band ran below zero — a lower bound of
   **−0.3 µg/m³** at the fourth lead, because a drift offset added to a small reading has
   nothing stopping it. Clamped at zero, which cannot change the coverage the band was gated
   on: it moves the boundary only across values no observation can take.

### Later

7. **Prospective forecast log — shipped.** Not "metrics over time": the original framing was
   both under-costed and methodologically weak, because the training window slides daily, so
   "MAE over time" would track window composition rather than model health. A hard month would
   read as a worse model. What is reported instead is per-lead error over a *named* period, and
   the split by which predictor answered the hour.

   `forecast/prospective.py` + `pipeline score-log`. The blocker the entry named was real —
   `refresh.yml` had `contents: read` and re-ingests from scratch, so there was nowhere to
   append. It now writes to an append-only `forecast-log` data branch, checked out into
   `.forecast-log/` and never merged into main, where a daily commit would bury the code
   history.

   Three decisions worth recording:

   - **The log records the frame the page was built from, not a re-prediction.** Re-running the
     predictor for the log would usually agree and occasionally not — a new observation moves
     the origin — and the log would then grade a forecast nobody saw. Logging is a parameter of
     `generate_report`, so page and log come from one object.
   - **Keyed on `(station, origin, lead)`, not on the wall clock.** A workflow re-run publishes
     the same forecast from the same origin; keying on `issued_at` would let a re-run
     double-weight that day in every figure the log ever produces. The `pages` concurrency
     group is also what makes the push safe without retry machinery — two refresh runs cannot
     overlap, and a run cancelled mid-push is recovered by the next one *because* appending is
     idempotent rather than additive.
   - **`source` is logged per row, which is the point.** The crossover was chosen on folds;
     the per-source split says whether it survives contact with hours nobody had seen when it
     was chosen. That is a test cross-validation structurally cannot run.

   *One thing it does not claim.* Logging happens in the build job and the deploy is separate,
   so a failed deploy leaves a logged forecast that never reached a reader. That does not
   weaken the evaluation: what out-of-sample scoring rests on is that the forecast was fixed
   **before the outcome existed**, not that a web server served it — and the docstring says so
   rather than claiming the stronger thing.

   Grading reads the observations *currently* stored, so a GIOŚ revision moves a past grade
   instead of being frozen at first sight. Right direction — the log records what was forecast,
   the database what is now believed to have happened — but it means any figure quoted from
   this command is a figure as of today. Next: the page can carry it once there is enough of a
   record to be worth reading, which is the thing that makes item 3 prospective rather than
   retrospective.

   ### The log is running, and its first numbers are not evidence of anything

   *Dated record, from the daily run of 2026-08-10 — three issuances, and it is written down
   here so that a later reading has a starting point, not because it says anything yet.*

   71 logged forecasts, **47 graded**, covering valid hours 2026-08-08 08:00 → 2026-08-10 07:00.
   By predictor: model 1.898 µg/m³ over 42 hours, naive rule 2.320 over 5.

   **Three reasons not to read that as a result, all of which have to be stated beside it.**

   - *It is smaller than the difference it would be measuring.* Five naive hours is one prefix
     of one day repeated; the crossover moves with the daily retrain, so those five are not even
     a fixed lead set. The prospective test of the crossover this log was built for needs weeks.
   - *The level is a property of the days, not of the model.* 1.9 µg/m³ against the year-round
     CV figure of ~7 does not mean the deployed model is four times better than cross-validation
     says. Early August was calm; the same easy-window effect that made the single-split MAE
     4.15 against a rolling 7.18 is operating here with three days instead of ten weeks. A
     prospective figure is only readable against a period, which is why `prospective_summary`
     refuses to return one without naming its window.
   - *It is graded against observations that can still be revised.* Two days of GIOŚ data is
     exactly the age at which a revision is most likely.

   What the run does establish is that the mechanism works end to end on infrastructure rather
   than on fixtures: three daily runs appended three forecasts, the branch holds one file with
   **no duplicate `(station, origin, lead)` key**, and the grading step joins it to the freshly
   ingested observations without a clock conversion. That was the thing that could have been
   wrong and is now known not to be.

   **And in three days it has already shown two things cross-validation cannot.** Both are
   observations, not defects, and both are what a log is *for*:

   - **A published day is not always 24 gradable leads.** 2026-08-09 logged leads 1–23 and no
     +24 h row. Two things produce that and the log cannot tell them apart, by design: the row
     never reached the frame (a gap in `build_inference_features`), or it reached it as `NaN`
     and `forecast_rows` skipped it, because an hour the page prints as "n/a" is not a forecast
     to be graded on. Either way, leads are derived from the origin rather than from row
     position, so the 23 that are there are labelled correctly and nothing is silently shifted.
     Worth knowing before any section is written that assumes 24 — and worth resolving in the
     log itself, since "we published nothing for that hour" and "we published a number" are
     different events and only one of them is currently visible.
   - **The naive prefix served nothing on one of the three days.** Naive served 5 leads on
     08-08 and 4 on 08-10, and **none at all** on 08-09.

     *This entry first read that as the boundary moving — "re-measured every retrain, and one
     run put it at the very start" — and that was wrong.* The 08-09 row has the other symptom
     beside it, 23 leads instead of 24, and the two have one cause: the origin landed on a
     published-but-empty hour, so there was no reading to repeat and every naive lead degraded
     to the model. Found while verifying the specialist serving change, fixed there, and written
     up under "Phase 1 — served". The correction is the point worth keeping: **the log caught a
     production fault three days before anyone went looking, and the first thing done with the
     evidence was to explain it as the phenomenon the log exists to measure.** A prospective
     record is only as good as the willingness to read it as a fault report rather than as a
     result.

8. **Health context** — translate µg/m³ into what it means for a runner or someone with
   asthma. Low technical signal, real product signal.

9. **Polish version of the page.** Data and likely audience are Polish; code and docs stay
   English by project convention.

10. **Extract a pure `_render_page(...)` from `generate_report` — done.** Page composition was
    interleaved with network I/O and file writing, so the assembled HTML — including the
    legacy-metadata normalisation — could only be exercised with network plus a saved bundle.
    Flagged during test writing; the helper tests replicated that step and could not catch it
    regressing.

    `_render_page` now takes what it renders — station, forecast frame, index reading,
    metadata, and the timestamp — and returns HTML. `generate_report` keeps the three impure
    jobs: fetching, clocking, writing. Sequenced *before* the multi-horizon work rather than
    after it, so item 5's per-lead section is written against a testable page instead of being
    retrofitted into a 700-line function.

    Three things the extraction bought, none of which the helper tests could reach:

    - The legacy normalisation is now pinned. A bundle that stored its split metrics only
      under `"metrics"` renders every row; without the normalisation `_row` drops them all and
      the table would publish empty.
    - `_render_page` no longer mutates the metadata it is handed — `setdefault` rewrote the
      caller's dict, harmless while the caller was always a fresh `load_model()`, and a trap
      for anything that reuses a bundle.
    - The clock is an argument, so the page is deterministic under test. Passing a fixed
      instant is what makes the footer assertion possible at all.

    And one real defect the seam exposed: **the forecast peak had no NaN gate**. Every stored
    metric passes through `_number`, but the peak comes from the frame rather than from the
    metadata, so an empty or all-NaN `predicted_pm25` published `Forecast peak: nan µg/m³` —
    the largest number on the page. Not reachable today, because `predict_next_24h` raises
    instead of returning an empty frame; reachable as soon as item 5 adds callers to this seam.
    Now gated the same way as everything else.

    Watch, for anything asserting on the whole page: the charts are inlined as base64, and a
    payload contains arbitrary letter runs — a bare `"nan" not in html` fails on PNG bytes,
    not on prose. The page-level tests strip `data:image/png;base64,…` first.

15. **The page as something a stranger can read — done, 2026-08-11.** Every item above makes
    the page *correct*. None of them made it *readable*, and four defects in one day showed the
    two are measured separately.

    - **Presentation (PR #16).** 2 812 words in one undifferentiated card, with all three
      figures putting their legend over the data. Now one card per section, a jump list built
      only from sections that rendered, a four-tile figure strip, and ~1 150 words moved behind
      `<details>` rather than deleted. Visible prose 2 812 → 1 583.
    - **A unit a thousand times too large (PR #16).** `text-transform: uppercase` on table
      headers upper-cased the micro sign to a capital mu: "Typical width (µg/m³)" published as
      "(MG/M³)". A test asserts the rule is gone.
    - **The badge reported the wrong index (PR #17).** Documented in
      `docs/research/data-sources.md`, because both causes are properties of the GIOŚ payload
      rather than of this code. The one that is ours: the fixture was written from the
      documented spelling instead of a captured response, so a green suite never saw it.
    - **The page never said what it was (PR #18).** It opened on a chart. A reader arriving
      from the portfolio index could not tell a live artefact from a screenshot, see where the
      data came from, or find the code — the only repository link was in the footer, in
      0.85rem grey. Three sentences now run above the forecast card, and a test keeps them
      free of figures: it is the one standing block of prose on a page whose every number is
      rebuilt daily, so a number written into it is the single claim a later run could
      contradict silently.
    - **The tables did not fit a phone (PRs #19, #20).** The CSS claimed only the seven-column
      lead table should scroll; at a real 390 px viewport all four did. Two causes:
      `.metrics.regimes th + th` outranks the `.metrics th + th` that the 640 px block relaxes,
      so the regime table kept fixed 5.5rem columns; and five columns of numbers at body size
      need ~427 px against the ~336 a phone leaves inside a card.

    **Two measurement traps, both of which produced a wrong answer before the right one.**

    A headless screenshot at `--window-size=390` is not a 390 px layout: Chromium will not
    create a window under ~500 px, so it renders wide and *crops*, which looks exactly like
    content overflowing. That artefact was read as a responsiveness bug on two consecutive
    days. The tell is that a paragraph wraps identically at 360 and at 500. Measured over CDP
    with `Emulation.setDeviceMetricsOverride`, the page has no horizontal overflow at any width
    tried — the real finding, the tables, only surfaced once the instrument was fixed.

    And `table { width: 100% }` pins every table at exactly the card width, so a rendered-width
    check reports the same number whether there is room to spare or none. Forcing
    `width: min-content` per table is what exposes the floor — without it, #19's margins all
    read `+0` and the metrics table looked comfortable while sitting one pixel inside the card.

    **Sized for winter, not for today.** That one pixel was August paying for it: with
    two-digit MAE and RMSE the same table came to 346 px and scrolled again. At 0.78rem/4px the
    winter case is a min-content 324 against 336. Published state at 390 px — metrics 316
    (+20 on current data), interval 264 (+72), regimes 224 (+112), lead 360 (−24, scrolls by
    design). Below ~380 px the metrics table scrolls again: its floor is the longest
    unbreakable word in the label column, a model name, and the ways under that are
    hyphenating it mid-word or type below 12 px. A scrolling table is the better of the three,
    and the CSS comment says so rather than promising past it.

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

14. ~~**Three-station comparison.**~~ **Dropped — the data does not support the item, in any
    of its shapes.** `docs/research/data-sources.md` is explicit: station 115's PM2.5 is manual
    (`getData` → `API-ERR-100003`, archive only, 4–8 week delay) and station 114 has no PM2.5
    at all. That leaves 129+115 at mismatched resolution and a 4–8 week latency, which cannot
    say anything about the hours this project forecasts, or a traffic-vs-background comparison
    on 114's NOx — a different question, about a pollutant the forecast does not target and the
    page does not publish.

    Dropped rather than left open. An item that has been re-scoped once and still has no
    defensible shape reads as an oversight the longer it sits in a list of things to do, and
    the reason it fails is itself the finding: **the spatial question needs stations this city
    does not instrument for it.** One automatic PM2.5 sensor is what Wrocław has, and it is
    what the project is built on.

## Next — the state as of 2026-08-11, and what it points at

Every numbered item is measured, shipped or dropped. What follows is not a backlog of
features; it is what the running system has started saying and has not been answered yet.

**16. The log now over-weights days the workflow was run by hand.** `score-log` reports 215
logged forecasts across four origin days, and the split is 24 / 23 / 48 / 120. The last figure
is today: `refresh.yml` was dispatched five times while shipping items 15's PRs, each run
logging a fresh origin, so **56% of the record is one day** — and the aggregate is taken over
rows. The key `(station, origin, lead)` is doing exactly its job (a re-run of the *same* origin
cannot double-count), so this is not a bug in the log; it is a bug in reading the log as though
its days were equally weighted. Decide between reporting per-day means of per-lead error, or
stating the run count beside the figure. Until then no prospective figure should be quoted
without the origin-day counts beside it. **This is the first item to settle, because every
other prospective claim below is read off the same aggregate.**

**17. Prospective evidence is still too thin to publish, and it is worth naming how thin.**
104 forecasts graded, of which the served-predictor split is model 82 (MAE 2.573), naive 20
(2.505), **specialist 2 (4.400)**. Two graded hours is not a measurement of the specialist
band — it is a reminder that the phase 1 gate rarely assigns any leads on current data, which
is what item 5's null already said retrospectively. Item 3 (the backtest chart) becomes
prospective when the log holds weeks rather than four days; the honest date for that is late
August at the earliest, and item 16 has to land first or the weeks will be unevenly weighted.

**18. Interval coverage out of sample has not been compared with the 0.762 that cross-validation
gave it.** `score-log` grades published bounds. Nobody has yet put the prospective rate beside
the retrospective one — the single most informative thing this project could say about its own
uncertainty claim, and the one figure a coverage gate cannot fake, because the hours were not
in hand when the band was drawn.

**19. Whether the specialist gate ever clears again.** It cleared on 16 of 24 leads three weeks
before item 5 shipped and on 7 of 24 on fresh data the day it did. Each daily run re-measures
it (~80 s of `train`). Worth checking on a later window whether that was seasonal, and saying so
either way; the null is already published, so a clearance would be the news, not the failure.

~~**Housekeeping.** Nine merged branches survive locally and on origin.~~ **Done 2026-08-11** —
all nine deleted from both after checking each against `git branch --merged origin/main`. Only
`main` and the `forecast-log` data branch remain.

**Unnumbered and optional, unchanged:** item 8 (health context) and item 9 (a Polish page).
Item 9 is worth weighing against item 16 rather than beside it — a second language doubles the
prose that every future measurement has to keep true, and this file records four defects in one
day caused by prose drifting from the numbers next to it.

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
