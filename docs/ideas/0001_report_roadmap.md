# Report roadmap — making the Pages report speak to a non-technical reader

Date: 2026-08-06
Status: draft
Author: P0w3r223 + Claude
Related to: `src/wroclaw_air_insights/report.py`, `.github/workflows/refresh.yml`

---

## Context

The published report showed three bare numbers — test MAE, RMSE and R². To a reader
without a modelling background they carry no information: no units to anchor on, no
sense of what "good" looks like, and no reference point to judge whether the model
adds anything at all.

A first pass (2026-08-06) addressed the explanation gap: the model card now shows the
deployed model next to the persistence baseline on the same test split, states the
headline error in a plain sentence, and carries a collapsible glossary defining each
metric, its range, and how to read it.

That change surfaced a substantive problem rather than a cosmetic one, which shapes
the priorities below.

## Open finding — R² is misleading on this task

On the current chronological split:

| Predictor | MAE | RMSE | R² |
| --- | --- | --- | --- |
| Random forest | 4.152 | 5.419 | 0.021 |
| Persistence baseline (T−24h) | 4.874 | 6.411 | −0.370 |

The model beats the baseline by 14.8% on MAE, yet posts an R² of essentially zero.
This is not a bug. The test window is the last 20% of a year ingested in July — a
low-variance summer period. R² measures variance explained, so when the target barely
varies, R² collapses even though the absolute error is fine. The metric is unstable
across seasons and is the wrong headline number for this task.

Now that the report explains R² to the reader, that weakness is visible on the public
page. It needs a real answer, not softer wording.

## Recommendations

### High value, low cost

1. **Report a skill score alongside (or instead of) R².**
   `skill = 1 − MSE_model / MSE_baseline`. Scale-free, season-robust, and it answers the
   only question that matters here: how much better than the naive rule. R² answers a
   question nobody asked and answers it badly on seasonal data.

2. **Put a backtest chart on the page** — forecast vs. actual for the last ~14 days.
   For a non-technical reader this is far more persuasive than any metrics table: it is
   directly checkable against lived experience.

3. **Show feature importances.** `model.feature_importances` already exists; only the
   chart is missing. It answers "is this a black box?" and demonstrates that the physical
   drivers (boundary layer height, wind) actually carry signal.

### Medium

4. **Break metrics down by regime** — smog episodes (>25 µg/m³) vs. ordinary hours, and
   per season. A model that is accurate on average but blind during episodes fails exactly
   where a forecast has value. Expect this question in any technical interview.

5. **Prediction intervals** via quantile regression. "22 ± 8 µg/m³" is both more honest
   and more legible than a single line, and it makes the model's uncertainty part of the
   product rather than a footnote.

6. **Track metrics over time.** `refresh.yml` retrains daily; appending each run's metrics
   to a CSV and charting MAE over time turns the project into a demonstration of model
   monitoring, for roughly twenty lines of code.

### Scope extensions

7. **Health context.** Translate µg/m³ into what it means for a runner or someone with
   asthma. The project is called *Insights*; today it stops at numbers.

8. **Three stations instead of one.** Stations 115 and 114 are already in `config.py`;
   a spatial comparison is the natural next step for the EDA narrative.

9. **Polish version of the page.** The data and the likely audience are Polish, while the
   report is English by project convention. A language toggle would resolve the tension
   without breaking the convention for code and docs.

## Not doing yet

Deep-learning sequence models (LSTM/temporal fusion). With ~8.5k hourly rows and a
random forest that only just clears a naive baseline, the bottleneck is feature quality
and evaluation honesty, not model capacity. Revisit once items 1–4 are in place.
