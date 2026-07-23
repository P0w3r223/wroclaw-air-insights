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


def generate_report(
    station_id: int = config.PRIMARY_STATION_ID, output_path: Path | None = None
) -> Path:
    """Build the HTML report and write it to ``output_path`` (default reports/site/)."""
    output_path = output_path or _DEFAULT_REPORT_PATH
    forecast_df = serving.predict_next_24h(station_id)
    aqi = gios.fetch_aqindex(station_id)
    metrics = model.load_model()["metadata"].get("metrics", {})

    overall = aqi.get("overall", {})
    category = overall.get("category") or "Brak indeksu"
    colour = _AQI_COLORS.get(category, "#9e9e9e")
    chart_b64 = _forecast_chart(forecast_df)
    peak = forecast_df["predicted_pm25"].max()
    generated = datetime.now(ZoneInfo(config.TIMEZONE)).strftime("%Y-%m-%d %H:%M %Z")

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
  body {{ font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
         -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility;
         max-width: 820px; margin: 2rem auto; padding: 0 1rem; color: #1c2430; }}
  h1 {{ margin-bottom: 0.2rem; font-weight: 700; letter-spacing: -0.01em; }}
  .sub {{ color: #667085; margin-top: 0; }}
  .badge {{ display: inline-block; padding: 0.4rem 0.9rem; border-radius: 999px;
           color: #fff; font-weight: 600; background: {colour}; }}
  img {{ max-width: 100%; height: auto; }}
  table {{ border-collapse: collapse; margin: 1rem 0; }}
  td, th {{ border: 1px solid #ddd; padding: 0.4rem 0.8rem; text-align: left; }}
  footer {{ color: #888; font-size: 0.85rem; margin-top: 2rem; }}
</style>
</head>
<body>
<h1>Wrocław Air Insights</h1>
<p class="sub">Live 24-hour PM2.5 forecast — {_station_name(station_id)}</p>

<p>Current air-quality index: <span class="badge">{category}</span></p>

<img src="data:image/png;base64,{chart_b64}" alt="24h PM2.5 forecast">

<p>Forecast peak: <strong>{peak:.1f} µg/m³</strong>
   (WHO 24h guideline: {config.PM25_WHO_DAILY} µg/m³).</p>

<h2>Model</h2>
<p>Metrics from the held-out chronological test split; the deployed model is retrained on
all data.</p>
<table>
  <tr><th>Test MAE</th><td>{metrics.get('mae', 'n/a')} µg/m³</td></tr>
  <tr><th>Test RMSE</th><td>{metrics.get('rmse', 'n/a')} µg/m³</td></tr>
  <tr><th>Test R²</th><td>{metrics.get('r2', 'n/a')}</td></tr>
</table>

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
