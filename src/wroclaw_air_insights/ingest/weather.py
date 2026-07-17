"""Open-Meteo weather client — historical (training) and forecast (serving).

Same split as :mod:`gios`: ``fetch_*`` do the HTTP call, :func:`parse_hourly` is a
pure transform. Open-Meteo returns parallel arrays under ``hourly`` (a ``time`` array
plus one array per variable), which map cleanly onto a DataFrame.

Training uses the Historical Forecast API and serving uses the Forecast API on
purpose: both expose the same variables from the same models, so weather features
are distribution-consistent between training and inference.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import pandas as pd
import requests

from wroclaw_air_insights import config

_REQUEST_TIMEOUT_S = 30
_MAX_RETRIES = 3
_RETRY_BACKOFF_S = 5


class OpenMeteoError(RuntimeError):
    """Raised when the Open-Meteo API returns an error or unrecoverable status."""


def _get(url: str, params: dict) -> dict:
    """HTTP GET returning parsed JSON, retrying on transient 429/5xx."""
    last_error: str | None = None
    for _ in range(_MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT_S)
        except requests.RequestException as exc:
            last_error = str(exc)
            time.sleep(_RETRY_BACKOFF_S)
            continue

        if response.status_code == 429 or response.status_code >= 500:
            last_error = f"HTTP {response.status_code}"
            time.sleep(_RETRY_BACKOFF_S)
            continue
        if response.status_code != 200:
            # Open-Meteo returns {"error": true, "reason": "..."} on bad requests.
            reason = ""
            try:
                reason = response.json().get("reason", "")
            except ValueError:
                reason = response.text[:200]
            raise OpenMeteoError(f"Open-Meteo {url} -> HTTP {response.status_code}: {reason}")
        try:
            return response.json()
        except ValueError as exc:
            raise OpenMeteoError(
                f"Open-Meteo {url} -> non-JSON body on 200: {response.text[:200]}"
            ) from exc

    raise OpenMeteoError(f"Open-Meteo {url} failed after {_MAX_RETRIES} attempts ({last_error})")


def parse_hourly(payload: dict) -> pd.DataFrame:
    """Convert an Open-Meteo response into a tidy hourly frame.

    Pure function — no network. Returns a ``timestamp`` column plus one column per
    requested weather variable, sorted by time.
    """
    hourly = payload.get("hourly") or {}
    if "time" not in hourly:
        return pd.DataFrame(columns=["timestamp"])

    frame = pd.DataFrame(hourly).rename(columns={"time": "timestamp"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    return frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def _base_params(lat: float, lon: float, variables: Sequence[str]) -> dict:
    return {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(variables),
        "timezone": config.TIMEZONE,
    }


def fetch_forecast(
    forecast_days: int = 2,
    past_days: int = 0,
    lat: float = config.WROCLAW_LAT,
    lon: float = config.WROCLAW_LON,
    variables: Sequence[str] = config.WEATHER_HOURLY_VARS,
) -> pd.DataFrame:
    """Fetch hourly weather forecast (default 2 days — covers the 24h horizon).

    ``past_days`` prepends recent history from the same forecast model, so live
    inference has a continuous weather series spanning the deepest PM2.5 lag.
    """
    params = {**_base_params(lat, lon, variables), "forecast_days": forecast_days}
    if past_days:
        params["past_days"] = past_days
    return parse_hourly(_get(config.OPEN_METEO_FORECAST_URL, params))


def fetch_historical(
    start_date: str,
    end_date: str,
    lat: float = config.WROCLAW_LAT,
    lon: float = config.WROCLAW_LON,
    variables: Sequence[str] = config.WEATHER_HOURLY_VARS,
) -> pd.DataFrame:
    """Fetch hourly historical weather for ``[start_date, end_date]`` (ISO ``YYYY-MM-DD``)."""
    params = {
        **_base_params(lat, lon, variables),
        "start_date": start_date,
        "end_date": end_date,
    }
    return parse_hourly(_get(config.OPEN_METEO_HISTORICAL_URL, params))
