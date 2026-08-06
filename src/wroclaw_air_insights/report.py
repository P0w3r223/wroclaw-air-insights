"""Generate a self-contained HTML report for GitHub Pages.

Combines the live 24h PM2.5 forecast, the current air-quality index, and the saved
model's metrics into a single standalone HTML file (chart embedded as base64), so it
can be published to Pages with no external assets.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wroclaw_air_insights import config, db  # noqa: E402
from wroclaw_air_insights.forecast import model, serving  # noqa: E402
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
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"


def _observed_pm25_mean(station_id: int) -> float | None:
    """Mean measured PM2.5 in the stored history — gives the error metrics a scale.

    An error of "4 µg/m³" means nothing on its own; it means a lot next to the level
    the pollutant actually sits at.
    """
    conn = db.connect()
    try:
        history = db.read_pm25(conn, station_id)
    finally:
        conn.close()
    return float(history["value"].mean()) if not history.empty else None


def _metrics_table(metrics: dict, baseline_metrics: dict) -> str:
    baseline_row = (
        f"""    <tr>
      <td>Naive baseline — “same hour, yesterday”</td>
      <td>{_fmt(baseline_metrics.get('mae'))}</td>
      <td>{_fmt(baseline_metrics.get('rmse'))}</td>
      <td>{_fmt(baseline_metrics.get('r2'))}</td>
    </tr>"""
        if baseline_metrics
        else ""
    )
    return f"""  <table class="metrics">
    <thead>
      <tr><th>Predictor</th><th>MAE ↓</th><th>RMSE ↓</th><th>R² ↑</th></tr>
    </thead>
    <tbody>
    <tr class="deployed">
      <td>Random forest <span class="tag">deployed</span></td>
      <td>{_fmt(metrics.get('mae'))}</td>
      <td>{_fmt(metrics.get('rmse'))}</td>
      <td>{_fmt(metrics.get('r2'))}</td>
    </tr>
{baseline_row}
    </tbody>
  </table>
  <p class="hint">MAE and RMSE are in µg/m³ (↓ lower is better);
  R² is a ratio (↑ higher is better).</p>"""


def _verdict(metrics: dict, improvement_pct: object) -> str:
    """One plain-language sentence: what the headline number means, and vs. what."""
    mae = metrics.get("mae")
    if not isinstance(mae, (int, float)):
        return "<p class=\"verdict\">No stored metrics for the deployed model yet.</p>"
    if isinstance(improvement_pct, (int, float)):
        comparison = (
            f" — <strong>{improvement_pct:.1f}% closer</strong> than the naive baseline, "
            "which just repeats yesterday’s reading for the same hour"
        )
    else:
        comparison = ""
    return (
        f'<p class="verdict">On hours the model had never seen, its forecast lands on average '
        f"<strong>{mae:.2f} µg/m³</strong> away from the measured value{comparison}.</p>"
    )


def _glossary(metrics: dict, observed_mean: float | None) -> str:
    """Plain-language explainer for MAE / RMSE / R², with ranges and how to read them."""
    mae, rmse, r2 = metrics.get("mae"), metrics.get("rmse"), metrics.get("r2")

    if observed_mean is not None and isinstance(mae, (int, float)) and observed_mean:
        scale = (
            f"For scale: PM2.5 at this station averages "
            f"<strong>{observed_mean:.1f} µg/m³</strong> "
            f"across the stored history, and the WHO 24-hour guideline is "
            f"{config.PM25_WHO_DAILY:.0f} µg/m³. An MAE of {mae:.2f} is therefore about "
            f"{100 * mae / observed_mean:.0f}% of a typical reading."
        )
    else:
        scale = (
            f"For scale, compare the error against typical PM2.5 levels and against the "
            f"WHO 24-hour guideline of {config.PM25_WHO_DAILY:.0f} µg/m³."
        )

    if isinstance(rmse, (int, float)) and isinstance(mae, (int, float)) and mae:
        spread = (
            f"Here RMSE is {rmse / mae:.1f}× the MAE"
            + (
                ", so the errors are fairly evenly sized — no single catastrophic miss "
                "is carrying the average."
                if rmse / mae < 1.5
                else ", so a minority of large misses (typically smog spikes) dominates "
                "the average."
            )
        )
    else:
        spread = "Comparing RMSE against MAE tells you how evenly sized the errors are."

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
      <p>How much of the hour-to-hour movement in PM2.5 the model actually reproduces,
      rather than how far off it is. It answers a different question from MAE:
      not “by how much am I wrong?” but “am I tracking the ups and downs at all?”.</p>
      <p><strong>Range:</strong> 1.0 means the model reproduces every wiggle perfectly.
      0.0 means it does no better than always guessing the long-run average and never
      moving. <em>Negative values are possible</em> and mean the model is worse than that
      flat guess. As a rough reading of this task: below 0.5 is weak, 0.5–0.75 is usable,
      0.75–0.9 is strong{
        f" — this model sits at {r2:.2f}" if isinstance(r2, (int, float)) else ""
      }.</p>
    </dd>
  </dl>

  <p class="note"><strong>Why the baseline row is the important one.</strong> Air pollution is
  persistent: “the same as yesterday at this hour” is already a decent guess, so a model can
  post a flattering R² while adding almost nothing. The honest question is not whether the
  numbers look good in isolation, but whether they beat that naive rule — and by how much.
  That is why both rows are shown on the same test split, scored the same way.</p>

  <p class="note"><strong>Why a chronological split.</strong> The test set is strictly later in
  time than the training set. Shuffling the hours at random would let the model peek at the
  future while learning the past, and every metric above would be flattering nonsense.</p>
</details>"""


def generate_report(
    station_id: int = config.PRIMARY_STATION_ID, output_path: Path | None = None
) -> Path:
    """Build the HTML report and write it to ``output_path`` (default reports/site/)."""
    output_path = output_path or _DEFAULT_REPORT_PATH
    forecast_df = serving.predict_next_24h(station_id)
    aqi = gios.fetch_aqindex(station_id)
    metadata = model.load_model()["metadata"]
    metrics = metadata.get("metrics", {})
    observed_mean = _observed_pm25_mean(station_id)

    overall = aqi.get("overall", {})
    category = overall.get("category") or "Brak indeksu"
    colour = _AQI_COLORS.get(category, "#9e9e9e")
    chart_b64 = _forecast_chart(forecast_df)
    peak = forecast_df["predicted_pm25"].max()
    generated = datetime.now(ZoneInfo(config.TIMEZONE)).strftime("%Y-%m-%d %H:%M %Z")

    metrics_table = _metrics_table(metrics, metadata.get("baseline_metrics", {}))
    verdict = _verdict(metrics, metadata.get("mae_improvement_pct"))
    glossary = _glossary(metrics, observed_mean)
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
  .tag {{ display: inline-block; margin-left: 6px; padding: 1px 7px; border-radius: 999px;
         background: #eef2ff; color: #4338ca; font-size: 0.7rem; font-weight: 600;
         text-transform: uppercase; letter-spacing: 0.04em; vertical-align: 1px; }}
  .hint {{ color: #667085; font-size: 0.82rem; margin: 8px 0 0; }}
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
  <p>The model was trained on the earlier part of the history and then scored on the later
  part it had never seen{tested_on} — a chronological split, never a random one. The deployed
  model is afterwards retrained on all available data.</p>
{metrics_table}
  {verdict}
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
