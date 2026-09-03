# wroclaw-air-insights

[![CI](https://github.com/P0w3r223/wroclaw-air-insights/actions/workflows/ci.yml/badge.svg)](https://github.com/P0w3r223/wroclaw-air-insights/actions/workflows/ci.yml)

**A live PM2.5 forecast for Wrocław that writes down what it predicted before the outcome
exists — and grades itself once those hours are measured.**

Every morning a GitHub Actions job pulls hourly GIOŚ readings and Open-Meteo weather into
SQLite, retrains the forecaster on the rolling year behind them, and rebuilds the published
page from that run. The forecast it shows is appended to an append-only log on a separate
branch and scored later against what actually happened.

**Live report: <https://p0w3r223.github.io/wroclaw-air-insights/>**

> Portfolio project A1 — data pipeline + time-series forecasting. Judged on methodological
> correctness rather than on model accuracy.

## What it does

1. **Ingest** — hourly measurements for every pollutant the Wrocław station reports (PM2.5,
   NO2, CO) from the GIOŚ API, plus the air-quality index and hourly weather from Open-Meteo.
2. **Store** — tidy measurements in a local SQLite database, gaps kept as gaps.
3. **Analyze** — a notebook with a question → analysis → conclusion narrative: seasonality,
   norm exceedances, cross-pollutant and weather relations.
4. **Forecast** — PM2.5 24 hours ahead from time + weather features, chosen against naive
   baselines on rolling-origin cross-validation, then served live from the saved bundle.
5. **Publish & grade** — a daily job deploys the report to GitHub Pages and logs the forecast
   it published, so past forecasts can be scored once their hours arrive.

## What it demonstrates

- **Time-series methodology.** Chronological splits and rolling-origin CV — a random split
  leaks the future into the past, and the CV figure is the one the page headlines, because a
  single summer window flatters the model.
- **Every claim reported against a baseline.** Two naive rules are scored on the same folds,
  and comparisons are judged on the paired per-fold difference rather than on the spread of
  error levels between folds, which is seasonal and common to every predictor.
- **A published claim carries the check that gates it.** A prediction interval states a
  coverage rate, so it ships only where measured coverage held on held-out folds — and the
  constructions that missed are published as misses, with their numbers.
- **No single predictor serves the whole chart.** The model cannot see the lead time, so its
  error is constant across the 24 hours while a naive rule's is not. Which predictor answers
  which hour is re-measured on every run under a stated rule, not fixed in code.
- **Held-out feature importance by removal, by group** — near-duplicate columns mask each
  other, and impurity importances disagree with measurement here.
- **Shipped and kept alive.** CI on every push, a daily refresh workflow, a pytest suite, and
  a page whose phone layout is measured at real viewport widths over CDP rather than eyeballed.

## Results

The pipeline retrains daily, so the current figures live on the page. As a **dated record of
one run** (8 584 hourly rows, 2025-07-24 → 2026-07-17), scored year-round on rolling folds:

| Predictor | CV MAE (µg/m³) |
|-----------|:--------------:|
| **HistGradientBoosting** (deployed) | **6.97** |
| RandomForest | 7.18 |
| naive rule — the reading at forecast time | 8.61 |

A 19.1% smaller miss than the naive rule, and the gap held on **5 folds of 5** — which is what
makes it a result rather than an average one hard winter fold could be carrying.

The reasoning behind each number, the lead-time analysis, the importance study and the two
documented null results are in [`docs/methodology.md`](docs/methodology.md).

## Stack

Python 3.12 · pandas · scikit-learn · SQLite · matplotlib · pytest · GitHub Actions + Pages

## Run it

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
pytest

python -m wroclaw_air_insights.pipeline all --days 365   # ingest + train
python -m wroclaw_air_insights.pipeline predict          # live next-24h forecast
python -m wroclaw_air_insights.report                    # build the Pages HTML
```

`pipeline --help` lists the measurement commands: `compare`, `importance`, `ab`,
`specialists`, `score-log`.

## Where the detail lives

| Document | What it holds |
|----------|---------------|
| [`docs/methodology.md`](docs/methodology.md) | every measurement behind the page, and what was rejected |
| [`docs/ideas/0001_report_roadmap.md`](docs/ideas/0001_report_roadmap.md) | the roadmap, with results — including the dropped items |
| [`docs/research/data-sources.md`](docs/research/data-sources.md) | station ids, endpoint traps, why these sources |
| [`notebooks/01_analysis.ipynb`](notebooks/01_analysis.ipynb) | the narrative EDA |

## Data sources

| Data | Source | License / attribution |
|------|--------|-----------------------|
| PM2.5 measurements (Wrocław) | [GIOŚ](https://powietrze.gios.gov.pl/pjp/content/api) | Public sector information — source: GIOŚ |
| Weather (history + forecast) | [Open-Meteo](https://open-meteo.com) (CAMS) | CC BY 4.0 — Open-Meteo + CAMS |

## License

MIT — see [LICENSE](LICENSE). Air-quality data © GIOŚ; weather data © Open-Meteo / CAMS
(CC BY 4.0).
