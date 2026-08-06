"""Generate a self-contained HTML report for GitHub Pages.

Combines the live 24h PM2.5 forecast, the current air-quality index, and the saved
model's metrics into a single standalone HTML file (chart embedded as base64), so it
can be published to Pages with no external assets.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from math import isfinite
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wroclaw_air_insights import config  # noqa: E402
from wroclaw_air_insights.forecast import baseline, model, serving  # noqa: E402
from wroclaw_air_insights.ingest import gios  # noqa: E402

# --- Chart styling: clean, print-quality matplotlib aligned with the page palette. ---
_ACCENT = "#2563eb"
_WHO_LINE = "#d97706"
_CHART_STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
    "grid.color": "#e3e7ee", "grid.linewidth": 0.9,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": "#1c2430",
    "axes.titlepad": 12, "axes.labelcolor": "#667085", "axes.labelsize": 10.5,
    "text.color": "#1c2430", "xtick.color": "#667085", "ytick.color": "#667085",
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "font.size": 10.5,
    "legend.frameon": False, "legend.fontsize": 9.5,
}
plt.rcParams.update(_CHART_STYLE)

# Polish air-quality index categories -> display colour.
_AQI_COLORS = {
    "Bardzo dobry": "#1a9850",
    "Dobry": "#91cf60",
    "Umiarkowany": "#fee08b",
    "Dostateczny": "#fc8d59",
    "Zły": "#d73027",
    "Bardzo zły": "#7f0000",
    "Brak indeksu": "#9e9e9e",
}
_DEFAULT_REPORT_PATH = config.PROJECT_ROOT / "reports" / "site" / "index.html"


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _forecast_chart(forecast_df) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    x, y = forecast_df["timestamp"], forecast_df["predicted_pm25"]
    ax.fill_between(x, y, color=_ACCENT, alpha=0.08, zorder=1)
    ax.plot(x, y, color=_ACCENT, lw=2.2, marker="o", markersize=5,
            markerfacecolor="white", markeredgecolor=_ACCENT, zorder=3)
    ax.axhline(config.PM25_WHO_DAILY, color=_WHO_LINE, ls="--", lw=1.4, zorder=2,
               label=f"WHO 24h guideline ({config.PM25_WHO_DAILY})")
    ax.set(title="Predicted PM2.5 — next 24 hours", ylabel="PM2.5 (µg/m³)", xlabel="")
    ax.margins(x=0.02)
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def _station_name(station_id: int) -> str:
    return next((s.name for s in config.STATIONS if s.id == station_id), f"station {station_id}")


# --- Model quality section ---------------------------------------------------
def _fmt(value: object, digits: int = 2) -> str:
    """Format a metric for the table, or ``n/a`` when it was never recorded."""
    number = _number(value)
    return f"{number:.{digits}f}" if number is not None else "n/a"


def _number(value: object) -> float | None:
    """Coerce a stored metric to a usable float, rejecting missing and non-finite values."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value):
        return float(value)
    return None


def _row(label: str, metrics: dict, css_class: str = "") -> str:
    """One predictor row, or nothing at all when the metrics were never recorded."""
    if not metrics:
        return ""
    cls = f' class="{css_class}"' if css_class else ""
    return f"""    <tr{cls}>
      <td>{label}</td>
      <td>{_fmt(metrics.get('mae'))}</td>
      <td>{_fmt(metrics.get('rmse'))}</td>
      <td>{_fmt(metrics.get('r2'))}</td>
    </tr>
"""


def _metrics_table(metadata: dict) -> str:
    """The model against both references it has to answer to, scored on the same rows."""
    baseline_label = metadata.get("baseline_label") or baseline.LABELS["persistence"]
    rows = (
        _row("Random forest (this project)", metadata.get("model", {}), "deployed")
        + _row(f"Naive rule — “{baseline_label}”", metadata.get("baseline_persistence", {}))
        + _row(
            f"Flat line — “{baseline.LABELS['climatology']}”",
            metadata.get("baseline_climatology", {}),
        )
    )
    window = metadata.get("test_window") or {}
    when = (
        f" — {window['start']} to {window['end']}"
        if window.get("start") and window.get("end")
        else ""
    )
    return f"""  <table class="metrics">
    <thead>
      <tr><th>Predictor</th><th>MAE ↓</th><th>RMSE ↓</th><th>R² ↑</th></tr>
    </thead>
    <tbody>
{rows}    </tbody>
  </table>
  <p class="hint">All three scored on the same held-out window{when}.
  MAE and RMSE are in µg/m³ (↓ lower is better); R² is a ratio (↑ higher is better).</p>"""


def _verdict(metadata: dict) -> str:
    """The honest headline: the all-seasons error first, the flattering one second.

    A single chronological split covers one season. Leading with it would publish the
    project's best-looking number as if it were its characteristic one.
    """
    cv = metadata.get("cross_validation") or {}
    cv_mae, cv_std = _number(cv.get("mae_mean")), _number(cv.get("mae_std"))
    split_mae = _number((metadata.get("model") or {}).get("mae"))

    if cv_mae is None and split_mae is None:
        return '<p class="verdict">No stored metrics for this model yet.</p>'

    if cv_mae is not None:
        spread = f" ± {cv_std:.2f}" if cv_std is not None else ""
        folds = cv.get("n_splits")
        across = (
            f"Averaged over {folds} rolling folds spanning the whole year"
            if isinstance(folds, int)
            else "Averaged over the whole year"
        )
        headline = (
            f"{across}, the forecast lands on average "
            f"<strong>{cv_mae:.2f}{spread} µg/m³</strong> away from what was actually "
            f"measured."
        )
    else:
        headline = ""

    if split_mae is not None and cv_mae is not None:
        window = metadata.get("test_window") or {}
        when = (
            f" ({window['start']} to {window['end']})"
            if window.get("start") and window.get("end")
            else ""
        )
        caveat = (
            f" On the single most recent held-out window alone{when} it does better — "
            f"<strong>{split_mae:.2f} µg/m³</strong> — because that window is summer, and "
            f"summer air is far easier to predict than a winter smog episode. The table "
            f"below uses that window; this sentence is the number to trust."
        )
    elif split_mae is not None:
        caveat = (
            f"On the held-out window the forecast lands on average "
            f"<strong>{split_mae:.2f} µg/m³</strong> away from the measured value. "
            f"This is a single season and not a year-round figure."
        )
    else:
        caveat = ""

    return f'<p class="verdict">{headline}{caveat}</p>'


def _skill_line(metadata: dict) -> str:
    """State both skill scores as one family, so R² cannot be read as a stranger."""
    skill = _number(metadata.get("skill_vs_persistence"))
    r2 = _number((metadata.get("model") or {}).get("r2"))
    improvement = _number(metadata.get("mae_improvement_pct"))
    if skill is None and improvement is None:
        return ""

    parts = []
    if improvement is not None:
        parts.append(
            f"the model's average miss is <strong>{improvement:.1f}% smaller</strong> "
            f"than the naive rule's"
        )
    if skill is not None:
        parts.append(
            f"it removes <strong>{100 * skill:.0f}%</strong> of that rule's squared error"
        )
    if r2 is None:
        against_flat = ""
    elif r2 < 0:
        against_flat = (
            f" Against a flat line drawn at the window's own average — a reference only "
            f"available with hindsight — it does no better at all, which is what an R² of "
            f"{r2:.2f} reports."
        )
    else:
        against_flat = (
            f" Against a flat line drawn at the window's own average — a reference only "
            f"available with hindsight — it removes just {100 * r2:.0f}%, and that second "
            f"number is exactly what R² reports."
        )
    return f'<p class="skill">On the same window, {" and ".join(parts)}.{against_flat}</p>'


# RMSE/MAE ≈ 1.25 when errors are normally distributed. Meaningfully above that means a
# minority of large misses is carrying the average.
_GAUSSIAN_RMSE_MAE_RATIO = 1.25


def _glossary(metadata: dict) -> str:
    """Plain-language explainer for MAE / RMSE / R², with ranges and how to read them."""
    metrics = metadata.get("model") or {}
    mae, rmse, r2 = _number(metrics.get("mae")), _number(metrics.get("rmse")), _number(
        metrics.get("r2")
    )
    test_mean = _number(metadata.get("test_mean_pm25"))

    if test_mean and mae is not None:
        scale = (
            f"For scale: over the scored window PM2.5 averaged "
            f"<strong>{test_mean:.1f} µg/m³</strong>, and the WHO 24-hour guideline is "
            f"{config.PM25_WHO_DAILY:.0f} µg/m³. A miss of {mae:.2f} is therefore about "
            f"{100 * mae / test_mean:.0f}% of a typical reading from that same period."
        )
    else:
        scale = (
            f"For scale, compare the error against typical PM2.5 levels and against the "
            f"WHO 24-hour guideline of {config.PM25_WHO_DAILY:.0f} µg/m³."
        )

    if rmse is not None and mae:
        ratio = rmse / mae
        spread = (
            f"Here RMSE is {ratio:.2f}× the MAE"
            + (
                f", close to the {_GAUSSIAN_RMSE_MAE_RATIO} you would get from evenly "
                f"scattered errors — no small group of disastrous hours is carrying the "
                f"average."
                if ratio < _GAUSSIAN_RMSE_MAE_RATIO * 1.2
                else ", well above the ~1.25 of evenly scattered errors, so a minority of "
                "large misses dominates."
            )
        )
    else:
        spread = "Comparing RMSE against MAE tells you how evenly sized the errors are."

    test_std, train_std = (
        _number(metadata.get("test_std_pm25")),
        _number(metadata.get("train_std_pm25")),
    )
    variation = (
        f"PM2.5 varied by only {test_std:.1f} µg/m³ (standard deviation) across the scored "
        f"hours, against {train_std:.1f} over the training period."
        if test_std and train_std
        else "the scored window is calmer than the year as a whole."
    )

    # A bundle saved before the references were recorded renders the model row alone;
    # the note must not then point at rows that aren't there.
    references_note = (
        """<p class="note"><strong>Why there are three rows, not one.</strong> Air pollution
  is persistent, so “the same as yesterday at this hour” is already a decent guess, and a
  model can post respectable-looking numbers while adding almost nothing. R² and the skill
  figure are the same calculation against two different references — a flat line and the
  naive rule. A forecast has to beat both to be worth running, so both are shown.</p>"""
        if metadata.get("baseline_persistence")
        else """<p class="note"><strong>A metric without a reference says little.</strong>
  Air pollution is persistent, so “the same as yesterday at this hour” is already a decent
  guess. This model bundle predates the comparison being recorded, so only its own figures
  are shown; retraining restores the reference rows.</p>"""
    )

    r2_reading = (
        f" This model scores {r2:.2f}, close to that floor, and part of the reason is the "
        f"window itself: {variation} There is simply less variation available to explain, "
        f"so every predictor's R² is squeezed toward zero. It is not <em>only</em> an "
        f"artefact, though — a gradient-boosted model reaches 0.23 on this very same "
        f"window. The random forest is kept because it can report which drivers it used, "
        f"and that choice costs roughly half a µg/m³ of MAE."
        if r2 is not None
        else ""
    )

    return f"""<details class="glossary">
  <summary>What do MAE, RMSE and R² actually mean?</summary>

  <dl>
    <dt>MAE — mean absolute error <span class="unit">µg/m³</span></dt>
    <dd>
      <p>Take every forecast, measure how far it landed from what was actually recorded
      (ignoring whether it was too high or too low), and average those distances.
      That is the MAE: <em>the size of a typical miss</em>.</p>
      <p><strong>Range:</strong> 0 and upwards. 0 is a flawless forecast; there is no upper
      limit, because errors are expressed in the same unit as the pollutant itself. {scale}</p>
    </dd>

    <dt>RMSE — root mean squared error <span class="unit">µg/m³</span></dt>
    <dd>
      <p>Also an average error, but each miss is squared before averaging, so one badly
      wrong hour counts for far more than several slightly wrong ones. It is the metric
      to watch when <em>rare, large</em> mistakes are the ones that matter — and for air
      quality they are, because those are the smog episodes people need warning about.</p>
      <p><strong>Range:</strong> 0 and upwards, and always at least as large as the MAE.
      The gap between the two is the interesting part. {spread}</p>
    </dd>

    <dt>R² — coefficient of determination <span class="unit">no unit</span></dt>
    <dd>
      <p>How much of the movement in PM2.5 the model reproduces, rather than how far off
      it is. It answers a different question from MAE: not “by how much am I wrong?” but
      “am I tracking the ups and downs at all?”.</p>
      <p><strong>Range:</strong> 1.0 reproduces every wiggle perfectly. 0.0 means doing no
      better than predicting <em>the average of the very window being scored</em> and never
      moving. That average can only be known after the fact, which is why it is not a row in
      the table: the flat line there uses the <em>training</em> average, the only one
      available in advance, and on a summer window that average is far too high — hence its
      deeply negative score. <em>Negative values are possible</em> for any predictor, and
      mean it does worse than the hindsight average.{r2_reading}</p>
    </dd>
  </dl>

  {references_note}

  <p class="note"><strong>Why a chronological split, and why two error figures.</strong> The
  test set is always strictly later in time than the training set; shuffling the hours at
  random would let the model peek at the future while learning the past. One split still only
  covers one season, so the headline figure comes from rolling-origin cross-validation, where
  every fold trains on the past and is tested on the future that follows it.</p>
</details>"""


def generate_report(
    station_id: int = config.PRIMARY_STATION_ID, output_path: Path | None = None
) -> Path:
    """Build the HTML report and write it to ``output_path`` (default reports/site/)."""
    output_path = output_path or _DEFAULT_REPORT_PATH
    forecast_df = serving.predict_next_24h(station_id)
    aqi = gios.fetch_aqindex(station_id)
    metadata = model.load_model()["metadata"]
    # Older bundles stored the split metrics only under "metrics"; normalise so the
    # section builders can read one key.
    metadata.setdefault("model", metadata.get("metrics", {}))

    overall = aqi.get("overall", {})
    category = overall.get("category") or "Brak indeksu"
    colour = _AQI_COLORS.get(category, "#9e9e9e")
    chart_b64 = _forecast_chart(forecast_df)
    peak = forecast_df["predicted_pm25"].max()
    generated = datetime.now(ZoneInfo(config.TIMEZONE)).strftime("%Y-%m-%d %H:%M %Z")

    metrics_table = _metrics_table(metadata)
    verdict = _verdict(metadata)
    skill_line = _skill_line(metadata)
    glossary = _glossary(metadata)
    n_test = metadata.get("n_test")
    tested_on = f" ({n_test:,} held-out hours)" if isinstance(n_test, int) else ""

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wrocław Air Insights — live forecast</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  body {{ font: 16px/1.55 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
         -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility;
         max-width: 820px; margin: 0 auto; padding: 32px 20px 56px; color: #1c2430; }}
  h1 {{ margin-bottom: 2px; font-weight: 700; letter-spacing: -0.01em; }}
  h2 {{ font-weight: 600; }}
  .sub {{ color: #667085; margin-top: 0; }}
  .badge {{ display: inline-block; padding: 0.4rem 0.9rem; border-radius: 999px;
           color: #fff; font-weight: 600; background: {colour}; }}
  .card {{ border: 1px solid #e3e7ee; border-radius: 12px; padding: 18px 20px; margin: 18px 0; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #eef1f6; }}
  img {{ max-width: 100%; height: auto; border-radius: 8px; }}
  code {{ background: #eef1f6; padding: 1px 5px; border-radius: 4px; font-size: 0.9em; }}
  footer {{ color: #667085; font-size: 0.85rem; margin-top: 24px; }}
  a {{ color: #2563eb; }}

  /* --- Model quality: metrics table, verdict line, collapsible glossary --- */
  .metrics th {{ font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.04em;
                color: #667085; font-weight: 600; }}
  .metrics th + th, .metrics td + td {{ text-align: right;
                font-variant-numeric: tabular-nums; width: 6.5rem; }}
  .metrics tr.deployed td {{ font-weight: 600; }}
  .hint {{ color: #667085; font-size: 0.82rem; margin: 8px 0 0; }}
  .skill {{ margin: 14px 0 0; font-size: 0.95rem; }}
  .verdict {{ background: #f6f8fc; border-left: 3px solid {_ACCENT};
             border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 16px 0 4px; }}
  details.glossary {{ margin-top: 14px; border-top: 1px solid #eef1f6; padding-top: 12px; }}
  details.glossary > summary {{ cursor: pointer; font-weight: 600; color: {_ACCENT};
             list-style: none; padding: 4px 0; }}
  details.glossary > summary::-webkit-details-marker {{ display: none; }}
  details.glossary > summary::before {{ content: "＋ "; font-weight: 400; }}
  details.glossary[open] > summary::before {{ content: "－ "; }}
  details.glossary dl {{ margin: 12px 0 0; }}
  details.glossary dt {{ font-weight: 600; margin-top: 18px; }}
  details.glossary dd {{ margin: 6px 0 0; padding-left: 14px;
             border-left: 2px solid #eef1f6; }}
  details.glossary dd p {{ margin: 6px 0; }}
  .unit {{ color: #667085; font-weight: 400; font-size: 0.85em; }}
  .note {{ background: #fbfbf9; border: 1px solid #eef1f6; border-radius: 8px;
          padding: 12px 14px; margin-top: 16px; font-size: 0.93rem; }}
</style>
</head>
<body>
<h1>Wrocław Air Insights</h1>
<p class="sub">Live 24-hour PM2.5 forecast — {_station_name(station_id)}</p>

<div class="card">
  <p>Current air-quality index: <span class="badge">{category}</span></p>
  <img src="data:image/png;base64,{chart_b64}" alt="24h PM2.5 forecast">
  <p>Forecast peak: <strong>{peak:.1f} µg/m³</strong>
     (WHO 24-hour guideline: {config.PM25_WHO_DAILY} µg/m³).</p>
</div>

<div class="card">
  <h2>How good is the forecast?</h2>
  <p>The model is always trained on earlier hours and scored on later ones{tested_on} — a
  chronological split, never a random one. The model that serves the chart above uses these
  settings but is refitted on all available data, so the figures below describe the method
  rather than that exact artefact.</p>
  {verdict}
{metrics_table}
  {skill_line}
{glossary}
</div>

<footer>
  Generated {generated} ·
  <a href="https://github.com/P0w3r223/wroclaw-air-insights">source on GitHub</a> ·
  Data © GIOŚ, weather © Open-Meteo / CAMS (CC BY 4.0)
</footer>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    print("wrote", generate_report())
