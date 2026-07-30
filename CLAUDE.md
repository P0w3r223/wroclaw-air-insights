# CLAUDE.md — wroclaw-air-insights

Guidance for Claude Code (and any contributor) working in this repository.

## What this project is

Air quality analysis for Wrocław based on open GIOŚ data: an ingestion pipeline,
a SQL (SQLite) store, an analysis notebook with insights, and a **24h PM2.5
forecast** compared against a naive baseline. Portfolio project A1 — the opening
piece that proves solid data work + a first, methodologically correct ML model.

## Architecture

```
src/wroclaw_air_insights/
  config.py          # single source of truth: stations, pollutants, endpoints, norms, paths
  ingest/
    gios.py          # GIOŚ client: multi-pollutant current/archival + air-quality index
    weather.py       # Open-Meteo client (forecast + historical); I/O split from parsing
  db.py              # SQLite: measurements (station, pollutant, hour) + weather
  clean.py           # pure cleaning/validation functions (unit-tested)
  forecast/
    features.py      # leakage-free features; build_features (train) + build_inference_features (serve)
    baseline.py      # naive baselines (persistence / seasonal)
    model.py         # time-based split, model comparison, rolling CV, joblib persistence
    serving.py       # live next-24h forecast from the saved model
  report.py          # self-contained HTML report for GitHub Pages
  pipeline.py        # CLI: ingest / train / compare / predict / all
notebooks/01_analysis.ipynb   # narrative EDA + figures
tests/                # pytest — cleaning, parsing, db, forecast, save/load
docs/research/        # data-source research + decisions
.github/workflows/    # ci.yml (tests) + refresh.yml (daily Pages deploy)
```

## Methodology rules (do not violate)

- **Time-based split, never random.** This is time-series forecasting; a random
  train/test split leaks the future into the past. Split chronologically.
- **Always beat a baseline, and say by how much.** The forecast is only meaningful
  relative to a naive baseline (e.g. persistence). Report the delta and why.
- **One clock.** GIOŚ returns local Warsaw time; Open-Meteo is requested in
  `Europe/Warsaw`. Keep timestamps consistent before joining.
- **Missing data is real.** Station readings have gaps (`null`) — handle explicitly,
  never assume a continuous series.
- **Train/serve consistency.** Weather features for training come from Open-Meteo's
  Historical Forecast API (same models as the live Forecast API).

## Conventions

- **English** for code, comments, README, and commit messages.
- **Conventional Commits**; commits tell a story: skeleton → feature → tests → docs.
- **Small, testable units.** I/O separated from logic; parsing functions are pure.
  No hardcoded values — everything configurable lives in `config.py`.
- **Data attribution required:** GIOŚ (air quality), Open-Meteo + CAMS (weather).
- Every modeling decision must be explainable in an interview — document the *why*.

## How to run

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
pytest                                                    # tests

python -m wroclaw_air_insights.pipeline all --days 365    # ingest + train
python -m wroclaw_air_insights.pipeline compare           # models vs baselines + CV
python -m wroclaw_air_insights.pipeline predict           # live next-24h forecast
python -m wroclaw_air_insights.report                     # build the Pages HTML report
```

Interpreter used during development: `.venv/Scripts/python.exe` (Python 3.12).
On Windows, run the pipeline with `PYTHONIOENCODING=utf-8` (or rely on the built-in
`sys.stdout.reconfigure`) so µg/µm and Polish characters print.

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes_tool` or `query_graph_tool` instead of Grep
- **Understanding impact**: `get_impact_radius_tool` instead of manually tracing imports
- **Code review**: `detect_changes_tool` + `get_review_context_tool` instead of reading entire files
- **Finding relationships**: `query_graph_tool` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview_tool` + `list_communities_tool`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
| ------ | ---------- |
| `detect_changes_tool` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context_tool` | Need source snippets for review — token-efficient |
| `get_impact_radius_tool` | Understanding blast radius of a change |
| `get_affected_flows_tool` | Finding which execution paths are impacted |
| `query_graph_tool` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes_tool` | Finding functions/classes by name or keyword |
| `get_architecture_overview_tool` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. No hooks installed — run `code-review-graph update` after code changes.
2. Use `detect_changes_tool` for code review.
3. Use `get_affected_flows_tool` to understand impact.
4. Use `query_graph_tool` pattern="tests_for" to check coverage.
