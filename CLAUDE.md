# CLAUDE.md — wroclaw-air-insights

Air quality analysis for Wrocław from open GIOŚ data: ingestion, a SQLite store, an analysis
notebook, and a PM2.5 forecast published daily to GitHub Pages. Portfolio project A1 — it is
judged on methodological correctness, not on model accuracy.

## Architecture

```
src/wroclaw_air_insights/
  config.py            # stations, pollutants, endpoints, norms, paths, lead grid, fold count
  ingest/gios.py       # GIOŚ client: current + archival measurements, air-quality index
  ingest/weather.py    # Open-Meteo client; I/O split from parsing
  db.py                # SQLite: measurements (station, pollutant, hour) + weather
  clean.py             # pure cleaning/validation
  forecast/
    features.py        # leakage-free features; build_features (train) / build_inference_features (serve)
    baseline.py        # naive rules + their published labels
    model.py           # split, selection, rolling CV, paired_delta, bundle persistence
    horizon.py         # per-lead scoring and the serving policy (which predictor answers which hour)
    serving.py         # live next-24h forecast from the saved bundle
  charts.py            # matplotlib figures as base64
  formatting.py        # the n/a gate every published number passes
  horizon_section.py   # the lead-axis section of the report
  report.py            # page composition; _render_page is pure, generate_report does the I/O
  pipeline.py          # CLI: ingest / train / compare / importance / predict / all
notebooks/01_analysis.ipynb
tests/                 # pytest
docs/ideas/            # roadmap: measured results, including the rejected ones
docs/research/         # data-source research
.github/workflows/     # ci.yml (tests) + refresh.yml (daily Pages deploy)
```

## Methodology rules

These override convenience. Each one exists because the project already got it wrong once.

- **Split chronologically, never randomly.** A random split on a time series leaks the future
  into the past.
- **Report against a baseline, with the delta.** A forecast means nothing on its own.
- **Judge a comparison on the paired per-fold difference** (`model.paired_delta`), not on the
  spread of MAE levels. That spread is seasonal and common to every predictor on those folds,
  so testing against it would call this project's own headline a null. A delta that changes
  sign across folds is a null result — publish it as one. Worked reasoning and the two
  outstanding nulls: `docs/ideas/0001_report_roadmap.md`.
- **Measure importance by removal, on held-out rows, by group** (`pipeline importance`).
  Near-duplicate columns mask each other, and impurity `feature_importances_` disagrees with
  held-out measurement here. Name the estimator beside any ranking.
- **One clock.** GIOŚ returns local Warsaw time and Open-Meteo is requested in `Europe/Warsaw`.
- **Missing data is real.** Station readings have gaps — handle them, don't assume continuity.
- **Train and serve from the same distribution.** Training weather comes from Open-Meteo's
  Historical Forecast API, the same models the live Forecast API runs.
- **No sentence on the published page may be contradicted by a number beside it.** Derive prose
  from the data it describes rather than templating it off a threshold; this has been the
  source of every defect a review has caught on the report.

## Conventions

- English for code, comments, docs and commit messages. Conventional Commits.
- I/O separated from logic; parsing functions pure. Configurable values live in `config.py`.
- Attribute the data: GIOŚ (air quality), Open-Meteo + CAMS (weather).
- Document the *why* of each modeling decision — the project is read as an explanation.

## How to run

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
pytest

python -m wroclaw_air_insights.pipeline all --days 365    # ingest + train
python -m wroclaw_air_insights.pipeline compare           # candidates + naive rules on shared folds
python -m wroclaw_air_insights.pipeline importance        # held-out importance by source
python -m wroclaw_air_insights.pipeline predict           # live next-24h forecast
python -m wroclaw_air_insights.report                     # build the Pages HTML
```

Interpreter: `.venv/Scripts/python.exe` (3.12). Set `PYTHONIOENCODING=utf-8` on Windows so
µg/m³ and Polish characters print.

`train` rewrites `models/pm25_forecaster.joblib`. The bundle carries a schema version, and an
older one is refused with the retrain command rather than read half-heartedly.

## Code graph

The repo carries two indexes: `.codegraph/` (queried with `codegraph_explore`, per the global
rule) and `.code-review-graph/`, whose MCP tools are declared in `.mcp.json` but are not always
loaded in a session — check what is actually available before planning around them. Neither
index has a hook: run `code-review-graph update` after changing code, or the graph answers
questions about the previous state of the repo.
