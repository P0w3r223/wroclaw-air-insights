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
    ab.py              # feature-idea A/B on shared rows and folds; the verdict rule for a gain
    specialists.py     # phase 1: a predictor per lead, the gate it must clear, the band it serves
    intervals.py       # prediction bands, and the coverage check that gates each one
    prospective.py     # the forecast log: what was published, graded once the hours arrive
    serving.py         # live next-24h forecast from the saved bundle
  accuracy_section.py  # the headline error, its two references, and how close selection was
  charts.py            # matplotlib figures as base64
  formatting.py        # the n/a gate every published number passes
  glossary_section.py  # what MAE/RMSE/R² mean, read against this run's own figures
  horizon_section.py   # the lead-axis section of the report
  interval_section.py  # the interval section: what the band claims, and whether it held
  regime_section.py    # the clean/elevated split at the WHO reference level
  rejected_section.py  # what was measured and not shipped (a dated record, not recomputed)
  report.py            # page composition; _render_page is pure, generate_report does the I/O
  pipeline.py          # CLI: ingest / train / compare / importance / ab / specialists /
                       #      score-log / predict / all
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
  sign across folds is a null result — publish it as one. Worked reasoning:
  `docs/ideas/0001_report_roadmap.md`.
- **The bar depends on whether nothing is an option.** A feature change can simply not happen,
  so it earns its column only if *no* fold contradicts it (`ab.verdict`). Some predictor has to
  answer a given hour, so the serving policy hands it over on a *majority* of folds
  (`horizon._model_earns_the_lead`). A *third* predictor is optional again, so a specialist
  takes an hour only by beating both references separately on ≥4 folds of 5
  (`specialists._clears`) — never their maximum, which would be chosen on the folds that then
  score it. These bars disagree on cases that occur here — do not collapse them into one
  function.
- **A published claim carries the check that tests it.** An interval states a coverage rate,
  so it ships only if measured coverage holds on held-out folds — on the average *and* fold by
  fold (`intervals.verdict`). Measure more than one construction before publishing a null:
  quantile regression missing here is a fact about quantile regression until a conformal band
  has been measured beside it.
- **A served range is a contiguous band, chosen once.** The naive prefix and the specialist
  band are each one decision (`horizon.crossover_lead`, `specialists.band`), not 24 argmins:
  per-lead selection would be a best-of-N on the same folds that produce the published
  figures, and the page could not describe a forecast with holes in it. Where the two overlap
  the stronger measurement wins — a specialist takes an hour from the naive rule only by
  having beaten it there.
- **Count the comparisons before believing one.** A sign-consistent verdict over 5 folds is a
  1-in-16 event under a change that does nothing — and **ties raise that rate, not lower it**,
  since a tie cannot contradict. At this project's own tie rate a grid of a dozen expects ~1.4
  such results from noise. `pipeline ab` prints the expectation per table and for the whole
  run; weigh a survivor by whether anything predicted it *in advance*, not by the sweep.
- **Measure importance by removal, on held-out rows, by group** (`pipeline importance`).
  Near-duplicate columns mask each other, and impurity `feature_importances_` disagrees with
  held-out measurement here. Name the estimator beside any ranking.
- **One clock.** GIOŚ returns local Warsaw time and Open-Meteo is requested in `Europe/Warsaw`.
- **Missing data is real.** Station readings have gaps — handle them, don't assume continuity.
- **Train and serve from the same distribution.** Training weather comes from Open-Meteo's
  Historical Forecast API, the same models the live Forecast API runs.
- **No sentence in a published artefact may be contradicted by a number beside it — or by the
  live page.** Derive prose from the data it describes rather than templating it off a
  threshold; this has been the source of every defect a review has caught on the report. The
  rule covers README too: the pipeline retrains daily, so a figure that describes the *current*
  system goes stale on the next run. Quote such figures only as a dated record of one run, and
  leave the current answer to the page. A *decision* that can move with the data (which lead
  the crossover falls on, which model deploys) is never asserted as a number outside the code
  that recomputes it — describe the rule that places it instead.

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
python -m wroclaw_air_insights.pipeline ab                # feature idea vs current set, paired
python -m wroclaw_air_insights.pipeline specialists       # phase 1: per-lead predictors + gate
python -m wroclaw_air_insights.pipeline predict           # live next-24h forecast
python -m wroclaw_air_insights.pipeline score-log         # grade the log, bands included
python -m wroclaw_air_insights.report                     # build the Pages HTML
```

Interpreter: `.venv/Scripts/python.exe` (3.12). Set `PYTHONIOENCODING=utf-8` on Windows so
µg/m³ and Polish characters print.

`train` rewrites `models/pm25_forecaster.joblib`. The bundle carries a schema version, and an
older one is refused with the retrain command rather than read half-heartedly. Since schema 3
it also carries one fitted estimator per lead in the served band, each with the lag set it was
trained on — so `train` re-measures the phase 1 gate on every run (~80 s locally, the most
expensive step in the command) and the bundle is ~5 MB rather than ~0.4.

## Verifying the published page

The page is the deliverable, so a claim about *it* is measured, not eyeballed — the same rule
the numbers on it live under. Two instruments have already returned confident wrong answers:

- **Confirm a deploy on fetched HTML, never on a reload.** `curl` the URL and assert on a
  build-specific marker (the `Generated …` stamp, or whatever the change added). A browser
  reload can serve a cached page that agrees with what you hoped.
- **A phone width needs CDP, not `--window-size`.** Chromium will not create a window under
  ~500 px, so `msedge --headless --window-size=390,900 --screenshot` renders wide and *crops* —
  which looks exactly like content overflowing, and was read that way on two consecutive days.
  Drive it over `--remote-debugging-port` with
  `Emulation.setDeviceMetricsOverride {width, height, deviceScaleFactor, mobile: true}`; the
  venv already has `websocket-client` and `requests`. The tell for the artefact: a paragraph
  wraps identically at 360 px and at 500 px.
- **`table { width: 100% }` hides how tight a table is.** Rendered width reports the card's
  width whether there is room to spare or none. Force `width: min-content` per table to read
  the floor, and size against the *widest realistic* content — two-digit winter MAE/RMSE, not
  today's single-digit summer figures. A layout that fits only while the air is clean fails in
  the season an hourly PM2.5 page is read.

## Code graph

The repo carries two indexes: `.codegraph/` (queried with `codegraph_explore`, per the global
rule) and `.code-review-graph/`, whose MCP tools are declared in `.mcp.json` but are not always
loaded in a session — check what is actually available before planning around them. Neither
index has a hook: run `code-review-graph update` after changing code, or the graph answers
questions about the previous state of the repo.
