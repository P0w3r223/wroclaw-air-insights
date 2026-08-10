"""Matplotlib figures for the report, rendered to embeddable base64 PNGs.

Split out of ``report`` so that module composes prose and this one draws pictures. The
functions here take numbers and return an image; deciding *whether* there is anything worth
drawing — a metric that came back NaN, a series too short to be a curve — stays with the
section builders, where it can be tested without rendering anything.

Every figure is returned as base64 rather than written to disk: the published page is a
single self-contained HTML file with no external assets.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wroclaw_air_insights import config  # noqa: E402
from wroclaw_air_insights.forecast import baseline  # noqa: E402

# --- Chart styling: clean, print-quality matplotlib aligned with the page palette. ---
ACCENT = "#2563eb"
MUTED = "#98a2b3"
# The third predictor needs its own colour, not a shade of the model's: a specialist is a
# different estimator, not a variant of the incumbent, and the chart is where that reads.
_SPECIALIST = "#0f9d78"
_WHO_LINE = "#d97706"
_INK = "#1c2430"
_CHART_STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
    "grid.color": "#e3e7ee", "grid.linewidth": 0.9,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": _INK,
    "axes.titlepad": 12, "axes.labelcolor": "#667085", "axes.labelsize": 10.5,
    "text.color": _INK, "xtick.color": "#667085", "ytick.color": "#667085",
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "font.size": 10.5,
    "legend.frameon": False, "legend.fontsize": 9.5,
}
plt.rcParams.update(_CHART_STYLE)


def to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def forecast(forecast_df) -> str:
    """The next 24 hours, with any hour the model did not answer drawn as itself."""
    fig, ax = plt.subplots(figsize=(10, 4))
    x, y = forecast_df["timestamp"], forecast_df["predicted_pm25"]
    ax.fill_between(x, y, color=ACCENT, alpha=0.08, zorder=1)
    # The interval, over exactly the hours it was validated on. Rows whose band was withheld
    # come through as NaN and matplotlib leaves them empty, which is the right picture: a gap
    # in the shading is an honest statement that nothing here was checked.
    lower, upper = forecast_df.get("lower_pm25"), forecast_df.get("upper_pm25")
    if lower is not None and upper is not None and lower.notna().any():
        ax.fill_between(x, lower, upper, color=ACCENT, alpha=0.16, zorder=2, linewidth=0,
                        label="80% interval, where it passed its coverage check")
    ax.plot(x, y, color=ACCENT, lw=2.2, marker="o", markersize=5,
            markerfacecolor="white", markeredgecolor=ACCENT, zorder=3)
    # The page claims the model forecasts 24 hours. Over the earliest leads it does not —
    # the naive rule does — so those points are marked rather than blended into the line.
    source = forecast_df.get("source")
    if source is not None and (source == config.FORECAST_SOURCE_NAIVE).any():
        mask = (source == config.FORECAST_SOURCE_NAIVE).to_numpy()
        ax.plot(x[mask], y[mask], color=MUTED, lw=0, marker="o", markersize=6, zorder=4,
                label="naive rule serves these hours")
    ax.axhline(config.PM25_WHO_DAILY, color=_WHO_LINE, ls="--", lw=1.4, zorder=2,
               label=f"WHO 24h guideline ({config.PM25_WHO_DAILY})")
    ax.set(title="Predicted PM2.5 — next 24 hours", ylabel="PM2.5 (µg/m³)", xlabel="")
    ax.margins(x=0.02)
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    return to_base64(fig)


def lead_curve(
    leads: list[int],
    model_mae: list[float],
    naive_mae: list[float],
    crossover: int,
    specialist_mae: list[float | None] | None = None,
    band: list[int] | None = None,
) -> str:
    """Error against lead time: the model's flat line against the two curves that beat it.

    The most informative chart on the page, because it shows what a single number cannot:
    the model's error does not depend on the lead — it *cannot*, the lead is not an input —
    while the reference it has to beat gets steadily harder. Where the lines cross is where
    the forecast starts earning its keep.

    The specialist curve is what makes the third band legible. It starts below both other
    lines and converges on the flat one as the lead approaches 24, because its whole advantage
    is holding a fresher observation than ``pm25_lag_24`` — and at a 24-hour lead that *is*
    ``pm25_lag_24``. A reader who does not believe the band should be able to see the
    convergence and check it against the +24 h row of the table.
    """
    fig, ax = plt.subplots(figsize=(10, 3.6))
    if crossover >= leads[0]:
        ax.axvspan(leads[0] - 0.5, crossover + 0.5, color=MUTED, alpha=0.12, zorder=1,
                   label="served by the naive rule")
    if band:
        ax.axvspan(band[0] - 0.5, band[1] + 0.5, color=_SPECIALIST, alpha=0.10, zorder=1,
                   label="served by a per-lead specialist")
    ax.plot(leads, naive_mae, color=MUTED, lw=1.8, ls="--", marker="o", markersize=4,
            zorder=3, label=f"Naive — {baseline.LABELS['origin_persistence']}")
    if specialist_mae and any(value is not None for value in specialist_mae):
        # Gaps stay gaps: a lead the phase 1 run could not score is not a point to interpolate
        # through, and matplotlib breaks the line on NaN by itself.
        drawn = [float("nan") if value is None else value for value in specialist_mae]
        ax.plot(leads, drawn, color=_SPECIALIST, lw=1.8, marker="o", markersize=4,
                zorder=5, label="Specialist — one predictor per lead")
    ax.plot(leads, model_mae, color=ACCENT, lw=2.2, marker="o", markersize=4, zorder=4,
            label="This project's model")
    ax.set(title="Error against how far ahead the forecast reaches",
           ylabel="MAE (µg/m³)", xlabel="Hours ahead")
    ax.set_xticks([lead for lead in leads if lead % 3 == 0 or lead == leads[0]])
    ax.margins(x=0.02)
    ax.legend(loc="upper left", ncol=2)
    return to_base64(fig)


def backtest(series: dict) -> str | None:
    """Measured against forecast over the tail of the held-out window.

    Three series on purpose: without the naive rule drawn beside it, a forecast that
    simply repeats yesterday looks impressive here, because tracking PM2.5 a day late
    still tracks it.
    """
    stamps = [datetime.fromisoformat(t) for t in series.get("timestamps") or []]
    actual, predicted = series.get("actual") or [], series.get("predicted") or []
    if not stamps or len(stamps) != len(actual) or len(stamps) != len(predicted):
        return None

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.fill_between(stamps, actual, color=_INK, alpha=0.07, zorder=1)
    ax.plot(stamps, actual, color=_INK, lw=1.8, zorder=3, label="Measured")
    ax.plot(stamps, predicted, color=ACCENT, lw=1.8, zorder=4, label="Forecast (24h ahead)")

    naive = series.get("naive")
    if naive and len(naive) == len(stamps):
        ax.plot(stamps, naive, color=MUTED, lw=1.1, ls="--", zorder=2,
                label=f"Naive — {baseline.LABELS['persistence']}")

    ax.axhline(config.PM25_WHO_DAILY, color=_WHO_LINE, ls=":", lw=1.2, zorder=2)
    ax.set(title="Forecast against what was measured — held-out hours",
           ylabel="PM2.5 (µg/m³)", xlabel="")
    ax.margins(x=0.01)
    ax.legend(loc="upper right", ncol=3)
    fig.autofmt_xdate()
    return to_base64(fig)
