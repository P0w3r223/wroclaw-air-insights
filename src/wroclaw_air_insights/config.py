"""Central configuration: stations, endpoints, norms, paths, feature list.

No I/O happens here — only constants and immutable data. Keeping every station id,
URL, norm and path in one place means the rest of the codebase has a single source
of truth and no magic values scattered around.

All values under "GIOŚ stations" were verified against the live API on 2026-07-17;
see ``docs/research/data-sources.md`` for the probe and the reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- Paths -------------------------------------------------------------------
# parents[2] == project root:  src/wroclaw_air_insights/config.py -> project/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "air_quality.db"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# --- Location (Wrocław) ------------------------------------------------------
WROCLAW_LAT = 51.11
WROCLAW_LON = 17.03
# One clock everywhere. GIOŚ returns local Warsaw time; we ask Open-Meteo for the
# same zone so hourly joins line up without off-by-one errors.
TIMEZONE = "Europe/Warsaw"

# --- GIOŚ air-quality API ----------------------------------------------------
# New API (the pre-2025-06-30 services were retired). JSON-LD, Polish keys.
GIOS_API_BASE = "https://api.gios.gov.pl/pjp-api/v1/rest"


@dataclass(frozen=True)
class Station:
    """A GIOŚ measurement station in Wrocław.

    ``code`` is the station prefix used inside GIOŚ position codes
    (e.g. ``DsWrocAlWisn-PM2.5-1g``). Sensor ids are intentionally NOT stored
    here — they are fetched at runtime from ``station/sensors/{id}`` because the
    sensor→pollutant mapping can change between years.
    """

    id: int
    name: str
    code: str
    has_automatic_pm25: bool


STATIONS: tuple[Station, ...] = (
    Station(129, "Wrocław, al. Wiśniowa", "DsWrocAlWisn", has_automatic_pm25=True),
    Station(115, "Wrocław, Na Grobli", "DsWrocNaGrob", has_automatic_pm25=False),
    Station(114, "Wrocław, Bartnicza", "DsWrocBartni", has_automatic_pm25=False),
)

# Station whose automatic hourly sensor is the live PM2.5 target for the forecast.
PRIMARY_STATION_ID = 129

# Pollutant code as it appears in the GIOŚ "Wskaźnik - kod" field.
PM25_CODE = "PM2.5"

# --- Norms / thresholds (µg/m³) ----------------------------------------------
# Reference levels used for the exceedance analysis. WHO 2021 guidelines are the
# strict health-based reference; the EU annual limit is the binding legal one.
# PM2.5 has no Polish smog-alert level (those are defined for PM10), so exceedance
# analysis references WHO/EU rather than a national alarm threshold.
PM25_WHO_DAILY = 15.0  # WHO 2021, 24-hour mean guideline
PM25_WHO_ANNUAL = 5.0  # WHO 2021, annual mean guideline
PM25_EU_ANNUAL = 25.0  # EU annual limit value (Directive 2008/50/EC)

# --- Weather features (Open-Meteo) -------------------------------------------
# Train on Historical Forecast API and serve on Forecast API: both expose the same
# variables from the same models, so features are distribution-consistent between
# training and inference (no train/serve shift).
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HISTORICAL_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

WEATHER_HOURLY_VARS: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "precipitation",
    "cloud_cover",
    "boundary_layer_height",  # mixing-layer height — strong physical driver of PM2.5
)

# --- Forecast task -----------------------------------------------------------
FORECAST_HORIZON_HOURS = 24
