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

Gradient boosting lowers MAE by **~25%** versus the naive persistence baseline; the
random forest — kept as the interpretable default — by ~15%. Its feature importances show
it relies on recent PM2.5 (autocorrelation) plus dispersion drivers — boundary-layer
height, wind, temperature — so it learns the physics rather than memorizing noise:

![Feature importances](reports/figures/fig6_importances.png)

**Rolling-origin cross-validation** (5 folds) gives a more honest picture: RandomForest
MAE is **7.2 ± 2.5 µg/m³** — far above the single summer split, because winter folds are
much harder. A single split flatters the model; CV exposes the seasonal variance.

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
- **Explicit missing-data handling** — station gaps are treated, not ignored.

## Live report

A daily GitHub Actions job refreshes the data, retrains, and deploys an HTML report
(live 24h forecast + current air-quality index) to **GitHub Pages**:
<https://p0w3r223.github.io/wroclaw-air-insights/>.

## License

MIT. Air-quality data © GIOŚ; weather data © Open-Meteo / CAMS (CC BY 4.0).
