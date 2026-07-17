# wroclaw-air-insights

**Air quality analysis for Wrocław from open GIOŚ data** — an ingestion pipeline, a
SQL database, an insights report, and a **24-hour PM2.5 forecast**.

> Portfolio project A1. Demonstrates pandas, SQL, visualization, working with public
> APIs, and a first scikit-learn model built with correct time-series methodology
> (a time-based split rather than a random one — a common junior mistake this project
> deliberately avoids).

## What it does

1. **Ingest** — pulls hourly PM2.5 for Wrocław stations directly from the GIOŚ API
   (live + up to a year of history) and hourly weather from Open-Meteo.
2. **Store** — writes tidy measurements into a local SQLite database.
3. **Analyze** — a notebook with a question → analysis → conclusion narrative:
   seasonality, station comparison, and exceedances of air-quality norms.
4. **Forecast** — predicts PM2.5 24 hours ahead using time + weather features, and
   reports how much it beats a naive baseline.

## Data sources

| Data | Source | License / attribution |
|------|--------|-----------------------|
| PM2.5 measurements (Wrocław) | [GIOŚ](https://powietrze.gios.gov.pl/pjp/content/api) — Główny Inspektorat Ochrony Środowiska | Public sector information — source: GIOŚ |
| Weather (history + forecast) | [Open-Meteo](https://open-meteo.com) (CAMS) | CC BY 4.0 — Open-Meteo + CAMS |

See [`docs/research/data-sources.md`](docs/research/data-sources.md) for station ids,
endpoint details, and the reasoning behind these choices.

## Project structure

```
src/wroclaw_air_insights/   # config, ingest (gios/weather), db, clean, forecast
notebooks/                  # 01_analysis.ipynb — EDA + figures
tests/                      # pytest — cleaning & feature logic
docs/research/              # data-source research and decisions
reports/figures/            # generated charts used in this README
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
pytest                      # run the test suite
# pipeline / forecast entry points — see below (added as the project grows)
```

## Methodology highlights

- **Time-based split** for training and evaluation — no future leakage.
- **Baseline comparison** — the model is reported against a naive persistence
  baseline, with the improvement quantified.
- **Explicit missing-data handling** — station gaps are treated, not ignored.

## License

MIT. Air-quality data © GIOŚ; weather data © Open-Meteo / CAMS (CC BY 4.0).
