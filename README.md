# wroclaw-air-insights

[![CI](https://github.com/P0w3r223/wroclaw-air-insights/actions/workflows/ci.yml/badge.svg)](https://github.com/P0w3r223/wroclaw-air-insights/actions/workflows/ci.yml)

**Air quality analysis for Wrocław from open GIOŚ data** — an ingestion pipeline, a
SQL database, an insights report, and a **24-hour PM2.5 forecast**.

> Portfolio project A1. Demonstrates pandas, SQL, visualization, working with public
> APIs, and a first scikit-learn model built with correct time-series methodology
> (a time-based split rather than a random one — a common junior mistake this project
> deliberately avoids).

## What it does

1. **Ingest** — pulls hourly measurements for every pollutant a Wrocław station reports
   (PM2.5, NO2, CO) directly from the GIOŚ API (live + up to a year of history), plus the
   current air-quality index, and hourly weather from Open-Meteo.
2. **Store** — writes tidy measurements into a local SQLite database.
3. **Analyze** — a notebook with a question → analysis → conclusion narrative:
   seasonality, exceedances of air-quality norms, and cross-pollutant/weather relations.
4. **Forecast** — predicts PM2.5 24 hours ahead from time + weather features, compares
   several models against naive baselines (single split **and** rolling-origin CV), and
   serves a **live next-24h forecast** from the saved model.
5. **Publish** — a scheduled GitHub Actions job refreshes the data daily and deploys an
   HTML report (live forecast + air-quality index) to GitHub Pages.

## Data sources

| Data | Source | License / attribution |
|------|--------|-----------------------|
| PM2.5 measurements (Wrocław) | [GIOŚ](https://powietrze.gios.gov.pl/pjp/content/api) — Główny Inspektorat Ochrony Środowiska | Public sector information — source: GIOŚ |
| Weather (history + forecast) | [Open-Meteo](https://open-meteo.com) (CAMS) | CC BY 4.0 — Open-Meteo + CAMS |

See [`docs/research/data-sources.md`](docs/research/data-sources.md) for station ids,
endpoint details, and the reasoning behind these choices.

## Results

Hourly PM2.5 shows the expected strong seasonality — low in summer, peaking in the
winter heating season, when the WHO 24-hour guideline is regularly exceeded:

![PM2.5 over time](reports/figures/fig1_timeseries.png)

**24-hour forecast — models vs. baselines** (chronological test split, ~1 year of hourly data):

| Model | MAE (µg/m³) | RMSE (µg/m³) | R² |
|-------|:-----------:|:------------:|:--:|
| **HistGradientBoosting** | **3.64** | **4.81** | **0.23** |
| RandomForest | 4.15 | 5.42 | 0.02 |
| baseline (persistence) | 4.87 | 6.41 | −0.37 |
| baseline (seasonal) | 5.95 | 7.70 | −0.98 |
| Ridge | 5.60 | 7.06 | −0.66 |

On that window gradient boosting lowers MAE by ~25% versus the naive persistence baseline.
**Year-round the figure is 19.1%** — 6.97 against the naive rule's 8.61 µg/m³, both scored
on the same rolling folds. The published page leads with the 19.1%: the split's 25% is a
summer number, and quoting it beside an all-seasons error credits the model with a season
it was not tested across. The gap holds **fold by fold, 5 of 5** (mean +1.64 µg/m³), which
is what makes it a result rather than an average that one hard winter fold could be
carrying.

**The pipeline picks the model itself**, and it picks on rolling-origin cross-validation,
never on the split above — choosing a winner on the rows the report then publishes would
turn those figures into a best-of-three rather than an honest estimate. CV MAE across the
year: HistGradientBoosting **6.97**, RandomForest 7.18, Ridge 8.19.

How close is that? The question is settled by the **per-fold difference**, not by comparing
the gap to the ±2.5 swing between folds — that swing is seasonal and common to every
candidate, so testing against it would call the model's own 1.64 win over the naive rule a
null too. Paired: gradient boosting beats RandomForest on 4 folds of 5 (+0.21, losing the first) and
Ridge on 4 of 5 with the second fold a **tie** — those two land 0.003 µg/m³ apart there, below the
precision this page even prints, and counting that as a fold won would be the same overclaim
one level down. So the honest summary is still "the top two are close", but for the right
reason: the winner changes hands on one fold in five, and because the pipeline retrains
daily on a rolling year it can change on the next run. The published report always names the
model it actually used.

**One error figure was describing twenty-four different tasks.** The model is trained on a
single task — predict 24 hours ahead — and the lead time is not one of its inputs. It cannot
be: every feature is anchored at the hour being predicted, so the rows for "+1h" and "+24h"
at the same valid time are the *same row*. The model's error is therefore not approximately
flat across the published chart, it is exactly constant. The naive reference is not:

| Lead | Model MAE | Naive MAE | Paired Δ | Folds won | Served by |
|------|:---------:|:---------:|:--------:|:---------:|:---------:|
| +1 h  | 6.97 | **3.75** | −3.22 | 0/5 | naive rule |
| +3 h  | 6.97 | **5.51** | −1.46 | 1/5 | naive rule |
| +6 h  | 6.97 | 7.20 | +0.23 | 2/5 | naive rule |
| +12 h | **6.97** | 8.51 | +1.54 | 4/5 | model |
| +24 h | **6.97** | 8.61 | +1.64 | 5/5 | model |

*(Naive rule = "the reading at the moment the forecast is issued". At a 24-hour lead that is
the same prediction as "the same hour yesterday", which is why one baseline sufficed until
the lead axis was measured.)*

**Over the first six hours the forecast was not worth serving**, so those hours now carry the
naive rule and the page says so. The reason is not uniform across that range, and the
distinction is the point. Through +5 h the naive rule simply wins — 3.75 against 6.97 at one
hour ahead, which is the model losing to "nothing changes" by 86%. At +6 h it flips: the
model is ahead *on average* (+0.23) while losing three folds of five. **A mean that holds in
two periods out of five does not earn the hour**, so the boundary sits above it rather than
below, and the published sentence is derived from that record instead of asserting a blanket
"the model is worse" the table would contradict.

Two things this measurement corrected on the way. The naive curve is **not** monotone in the
lead — it peaks at +18 h (8.93) and falls back to 8.57 by +23 h, because "the reading now"
re-aligns with the same hour of day as the lead approaches 24. And the served range is kept
contiguous by construction, not because the curve is smooth: 24 independent per-lead choices
would be a best-of-N on the same folds that produce the published figures.

**Rolling-origin cross-validation** also gives a far more sober picture than any single
split: **~7 µg/m³** against the split's 3.6, because winter folds are much harder than a
summer test window. A single split flatters the model; CV exposes the seasonal variance,
which is why the report headlines the CV figure.

**What it does when the air is actually bad.** One average over every hour is dominated by
calm ones, so the error is also reported either side of the WHO 24-hour guideline level
(15 µg/m³, used as a reference level for hourly readings — not as a compliance test, which
applies to daily means):

| Hours | Model MAE | Model bias | Naive MAE | Naive bias |
|-------|:---------:|:----------:|:---------:|:----------:|
| below 15 µg/m³ (1 280 h) | 3.30 | **+2.15** | 4.37 | +1.70 |
| at or above 15 µg/m³ (437 h) | 4.61 | **−2.73** | 6.35 | −4.95 |

The model runs high on clean air and low on dirty air — regression toward the mean, which
the aggregate bias of +0.91 hides by netting the two against each other. It flags **57%** of
genuinely elevated hours against the naive rule's 40%, and **42%** of its warnings are wrong
against the naive rule's 60%: better on both counts, and not an alerting system.

**Where the skill comes from — measured on held-out rows, not on training splits.** The
question worth answering is *what would the forecast lose without this?*, so each source of
information is removed and the model scored again (MAE, µg/m³; full model 3.64, persistence
4.87):

| Source | MAE if shuffled | MAE if retrained without it |
|--------|:---------------:|:---------------------------:|
| Weather (9 columns) | 4.96 (+1.32) | 4.39 (+0.75) |
| PM2.5 history (5 lag/rolling columns) | 3.84 (+0.20) | 4.36 (+0.72) |
| Calendar (7 columns) | 3.80 (+0.16) | 3.71 (+0.08) |

Weather and PM2.5 history are worth **the same** (+0.75 vs +0.72), and either one alone
still beats the naive rule. The single strongest column is `boundary_layer_height` —
mixing-layer depth, the physical control on how much air the pollution is diluted into.

The two columns disagree on purpose. *Shuffled* keeps the fitted model and destroys the
information at prediction time; *retrained* refits without it. PM2.5 history is cheap to
shuffle but expensive to drop, because the five lag columns are near-duplicates — shuffle
one and the others still carry it. That gap is why single-column importances mislead here,
and impurity importances mislead further still:

| Feature | Impurity (RF, training rows) | Permutation (MAE ↑, held-out) |
|---------|:----------------------------:|:-----------------------------:|
| `pm25_lag_24` | **0.353** — 1st | −0.016 |
| `boundary_layer_height` | 0.118 | **0.358** — 1st |
| `wind_direction_10m` | 0.082 | 0.108 |
| `wind_speed_10m` | 0.040 | 0.080 |

An earlier version of this section quoted the left column and concluded the model was
"persistence plus a weather correction". The right column, and the group table above, say
it is both in equal measure. Reproduce either with `pipeline importance`.

The figure below is the left column — the impurity ranking, from a RandomForest that is no
longer the deployed model. It is kept as the contrast, not as the answer:

![Feature importances](reports/figures/fig6_importances.png)

**A documented null result.** That impurity ranking put wind *direction* above wind *speed*,
which suggested the raw 0–360° encoding (discontinuous at north) was costing accuracy.
Re-encoding it as u/v components and as sin/cos was A/B-tested on the same folds: CV MAE
moved by ≤0.09 µg/m³ against a fold spread of ±2.5, and for the deployed model it moved the
wrong way. The change was rejected rather than shipped — see
[`docs/ideas/0001_report_roadmap.md`](docs/ideas/0001_report_roadmap.md).

The full narrative analysis — seasonality, norm exceedances, an hour × weekday heatmap,
and weather correlations — is in
[`notebooks/01_analysis.ipynb`](notebooks/01_analysis.ipynb).

## Project structure

```
src/wroclaw_air_insights/
  config.py  clean.py  db.py  pipeline.py  report.py
  ingest/    gios.py  weather.py
  forecast/  features.py  baseline.py  model.py  serving.py
notebooks/                  # 01_analysis.ipynb — EDA + figures
tests/                      # pytest — cleaning, parsing, db, forecast, save/load
docs/research/              # data-source research and decisions
reports/figures/            # generated charts used in this README
.github/workflows/          # ci.yml (tests) + refresh.yml (daily Pages deploy)
```

## Setup

```bash
python -m venv .venv
# Windows:
.venv/Scripts/python -m pip install -r requirements.txt
# Linux/macOS:
# source .venv/bin/activate && pip install -r requirements.txt
```

## Usage

```bash
# fetch ~1 year of all pollutants + weather into SQLite, then train + evaluate
python -m wroclaw_air_insights.pipeline all --days 365
# or run the steps separately:
python -m wroclaw_air_insights.pipeline ingest --days 365
python -m wroclaw_air_insights.pipeline train      # train + save the model
python -m wroclaw_air_insights.pipeline compare    # baselines vs models + rolling CV
python -m wroclaw_air_insights.pipeline importance # what each source of data is worth
python -m wroclaw_air_insights.pipeline predict    # live next-24h PM2.5 forecast
python -m wroclaw_air_insights.report              # build the HTML report

pytest                      # run the test suite

# reproduce the analysis notebook (figures + outputs)
jupyter nbconvert --to notebook --execute --inplace notebooks/01_analysis.ipynb
```

## Methodology highlights

- **Time-based split** for training and evaluation — no future leakage.
- **Rolling-origin cross-validation** (TimeSeriesSplit) alongside a single split, so the
  reported error reflects seasonal variance rather than one lucky window.
- **Model comparison** — baselines (persistence, seasonal) vs. Ridge, gradient boosting
  and random forest on identical test data.
- **Leakage-free inference** — training and live prediction share one feature contract:
  every feature is knowable at the forecast origin.
- **Importance measured by removal, on held-out rows** — grouped, because near-duplicate
  columns hide each other's contribution, and because impurity importances disagree.
- **Backtest drawn from held-out predictions** — the published chart uses the
  chronologically-trained model's output, never the deployed all-data model's fit over
  days it learned from, with the naive rule plotted beside it.
- **Explicit missing-data handling** — station gaps are treated, not ignored.

## Live report

A daily GitHub Actions job refreshes the data, retrains, and deploys an HTML report
(live 24h forecast + current air-quality index) to **GitHub Pages**:
<https://p0w3r223.github.io/wroclaw-air-insights/>.

## License

MIT. Air-quality data © GIOŚ; weather data © Open-Meteo / CAMS (CC BY 4.0).
