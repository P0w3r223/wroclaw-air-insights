"""GIOŚ air-quality API client (new v1/rest API, live from 2025-06-30).

Network I/O is deliberately separated from parsing:
- ``fetch_*`` functions do the HTTP call (side effects, retries, rate limiting);
- ``parse_measurements`` is a pure function (``dict -> DataFrame``) so it can be
  unit-tested without touching the network.

The API returns JSON-LD with Polish keys; the response list lives under a key that
starts with ``"Lista"`` and each row looks like::

    {"Kod stanowiska": "DsWrocAlWisn-PM2.5-1g", "Data": "2026-07-17 11:00:00", "Wartość": 15.7}

``Wartość`` may be ``null`` (missing reading) and ``Data`` is local Warsaw time.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

from wroclaw_air_insights import config

_REQUEST_TIMEOUT_S = 30
_MAX_RETRIES = 4
# GIOŚ throttles archival/sensor endpoints to ~2 req/min → back off ~30s on 429.
_RETRY_BACKOFF_S = 30
# Proactive pause between archival pages to stay under the 2 req/min limit.
_ARCHIVAL_PAGE_DELAY_S = 31
# Max window the archival endpoint accepts in a single request.
_ARCHIVAL_MAX_DAYS = 366

# GIOŚ JSON-LD field names (Polish).
_KEY_DATE = "Data"
_KEY_VALUE = "Wartość"
_KEY_STATION_CODE = "Kod stanowiska"
_KEY_SENSOR_ID = "Identyfikator stanowiska"
_KEY_INDICATOR_CODE = "Wskaźnik - kod"

MEASUREMENT_COLUMNS = ("timestamp", "value", "station_code")


class GiosApiError(RuntimeError):
    """Raised when the GIOŚ API returns an error payload or an unrecoverable status."""


def _get(url: str, params: dict | None = None) -> dict:
    """HTTP GET returning parsed JSON, retrying on rate-limit (429) and transient 5xx."""
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
            raise GiosApiError(
                f"GIOŚ {url} -> HTTP {response.status_code}: {response.text[:200]}"
            )

        payload = response.json()
        if isinstance(payload, dict) and "error_code" in payload:
            raise GiosApiError(
                f"GIOŚ {url} -> {payload.get('error_code')}: {payload.get('error_reason')}"
            )
        return payload

    raise GiosApiError(f"GIOŚ {url} failed after {_MAX_RETRIES} attempts ({last_error})")


def _extract_list(payload: dict) -> list[dict]:
    """Return the first JSON-LD list value (the ``"Lista ..."`` key)."""
    for key, value in payload.items():
        if not key.startswith("@") and isinstance(value, list):
            return value
    return []


def parse_measurements(payload: dict) -> pd.DataFrame:
    """Convert a getData / archivalData payload into a tidy measurements frame.

    Pure function — no network. Returns columns ``timestamp`` (datetime),
    ``value`` (float, NaN for missing) and ``station_code`` (str), sorted by time
    with unparseable timestamps dropped.
    """
    rows = _extract_list(payload)
    frame = pd.DataFrame(
        [
            {
                "timestamp": row.get(_KEY_DATE),
                "value": row.get(_KEY_VALUE),
                "station_code": row.get(_KEY_STATION_CODE),
            }
            for row in rows
        ],
        columns=list(MEASUREMENT_COLUMNS),
    )
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return (
        frame.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def get_pm25_sensor_id(station_id: int) -> int:
    """Look up the PM2.5 sensor (``Identyfikator stanowiska``) for a station.

    Sensor ids are resolved at runtime rather than hardcoded, because the
    sensor→pollutant mapping can change between years.
    """
    payload = _get(f"{config.GIOS_API_BASE}/station/sensors/{station_id}")
    for sensor in _extract_list(payload):
        if sensor.get(_KEY_INDICATOR_CODE) == config.PM25_CODE:
            return int(sensor[_KEY_SENSOR_ID])
    raise GiosApiError(f"No PM2.5 sensor found at station {station_id}")


def fetch_current_pm25(station_id: int) -> pd.DataFrame:
    """Fetch the last ~3 days of hourly PM2.5 for a station (live endpoint)."""
    sensor_id = get_pm25_sensor_id(station_id)
    payload = _get(f"{config.GIOS_API_BASE}/data/getData/{sensor_id}")
    return parse_measurements(payload)


def fetch_archival_pm25(
    station_id: int, days: int = _ARCHIVAL_MAX_DAYS, page_size: int = 5000
) -> pd.DataFrame:
    """Fetch up to ``days`` (≤366) of hourly PM2.5 history for a station.

    Pages through the archival endpoint, pausing between pages to respect the
    ~2 req/min limit. Returns the same tidy frame as :func:`parse_measurements`,
    de-duplicated on timestamp.
    """
    if days > _ARCHIVAL_MAX_DAYS:
        raise ValueError(
            f"archival endpoint accepts at most {_ARCHIVAL_MAX_DAYS} days per call, got {days}"
        )

    sensor_id = get_pm25_sensor_id(station_id)
    url = f"{config.GIOS_API_BASE}/archivalData/getDataBySensor/{sensor_id}"
    base_params = {"dayNumber": days, "size": page_size}

    frames: list[pd.DataFrame] = []
    page = 0
    while True:
        payload = _get(url, params={**base_params, "page": page})
        frames.append(parse_measurements(payload))
        total_pages = int(payload.get("totalPages", 1) or 1)
        page += 1
        if page >= total_pages:
            break
        time.sleep(_ARCHIVAL_PAGE_DELAY_S)

    if not frames:
        return pd.DataFrame(columns=list(MEASUREMENT_COLUMNS))
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
