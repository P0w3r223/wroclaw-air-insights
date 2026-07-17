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
MODELS_DIR = PROJECT_ROOT / "models"

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

# --- Pollutants --------------------------------------------------------------
# GIOŚ codes ("Wskaźnik - kod") for the criteria pollutants we care about.
# Ingestion pulls whichever of these a station actually measures
# (see ingest.gios.get_sensor_map).
POLLUTANTS: tuple[str, ...] = ("PM2.5", "PM10", "NO2", "O3", "SO2", "CO")

# The pollutant the 24h forecast targets.
TARGET_POLLUTANT = "PM2.5"

# Backwards-compatible alias.
PM25_CODE = TARGET_POLLUTANT

# Physically plausible value ranges (µg/m³) per pollutant, used by cleaning to drop
# sensor errors. Anything not listed falls back to DEFAULT_VALUE_RANGE.
DEFAULT_VALUE_RANGE = (0.0, 10000.0)
POLLUTANT_RANGES: dict[str, tuple[float, float]] = {
    "PM2.5": (0.0, 1000.0),
    "PM10": (0.0, 2000.0),
    "NO2": (0.0, 1000.0),
    "O3": (0.0, 1000.0),
    "SO2": (0.0, 1500.0),
    "CO": (0.0, 50000.0),  # CO reported in µg/m³
}

# --- Norms / thresholds (µg/m³) ----------------------------------------------
# Reference levels for exceedance analysis. WHO 2021 guidelines are the strict
# health-based reference; EU limits are the binding legal ones. Only documented
# thresholds are listed per pollutant.
POLLUTANT_NORMS: dict[str, dict[str, float]] = {
    "PM2.5": {"who_daily": 15.0, "who_annual": 5.0, "eu_annual": 25.0},
    "PM10": {"who_daily": 45.0, "who_annual": 15.0, "eu_daily": 50.0, "eu_annual": 40.0},
    "NO2": {"who_annual": 10.0, "eu_annual": 40.0, "eu_hourly": 200.0},
    "O3": {"who_8h": 100.0, "eu_8h": 120.0},
    "SO2": {"who_daily": 40.0, "eu_hourly": 350.0},
    "CO": {"eu_8h": 10000.0},
}

# Backwards-compatible PM2.5 aliases (used by the analysis notebook).
PM25_WHO_DAILY = POLLUTANT_NORMS["PM2.5"]["who_daily"]
PM25_WHO_ANNUAL = POLLUTANT_NORMS["PM2.5"]["who_annual"]
PM25_EU_ANNUAL = POLLUTANT_NORMS["PM2.5"]["eu_annual"]

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
