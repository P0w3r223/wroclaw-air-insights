"""GIOŚ air-quality API client (new v1/rest API, live from 2025-06-30).

Network I/O is deliberately separated from parsing:
- ``fetch_*`` functions do the HTTP call (side effects, retries, rate limiting);
- ``parse_measurements`` / ``parse_aqindex`` are pure functions so they can be
  unit-tested without touching the network.

The API returns JSON-LD with Polish keys; measurement rows look like::

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

# Pollutants covered by the Polish air-quality index (CO is not part of it).
AQI_POLLUTANTS = ("SO2", "NO2", "PM10", "PM2.5", "O3")


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


def get_sensor_map(station_id: int) -> dict[str, int]:
    """Map pollutant code -> sensor id for every sensor at a station.

    Resolved at runtime rather than hardcoded, because the sensor→pollutant mapping
    can change between years.
    """
    payload = _get(f"{config.GIOS_API_BASE}/station/sensors/{station_id}")
    return {
        sensor[_KEY_INDICATOR_CODE]: int(sensor[_KEY_SENSOR_ID])
        for sensor in _extract_list(payload)
        if sensor.get(_KEY_INDICATOR_CODE) and sensor.get(_KEY_SENSOR_ID) is not None
    }


def get_sensor_id(station_id: int, pollutant: str) -> int:
    """Sensor id for a pollutant at a station, or raise if unavailable."""
    sensors = get_sensor_map(station_id)
    if pollutant not in sensors:
        raise GiosApiError(f"No {pollutant} sensor found at station {station_id}")
    return sensors[pollutant]


def get_pm25_sensor_id(station_id: int) -> int:
    """Backwards-compatible PM2.5 sensor lookup."""
    return get_sensor_id(station_id, config.PM25_CODE)


def fetch_current(
    station_id: int, pollutant: str, sensor_id: int | None = None
) -> pd.DataFrame:
    """Fetch the last ~3 days of hourly readings for a pollutant (live endpoint)."""
    sid = sensor_id if sensor_id is not None else get_sensor_id(station_id, pollutant)
    payload = _get(f"{config.GIOS_API_BASE}/data/getData/{sid}")
    return parse_measurements(payload)


def fetch_current_pm25(station_id: int) -> pd.DataFrame:
    """Backwards-compatible live PM2.5 fetch."""
    return fetch_current(station_id, config.PM25_CODE)


def fetch_archival(
    station_id: int,
    pollutant: str,
    days: int = _ARCHIVAL_MAX_DAYS,
    page_size: int = 5000,
    sensor_id: int | None = None,
) -> pd.DataFrame:
    """Fetch up to ``days`` (≤366) of hourly history for a pollutant at a station.

    Pages through the archival endpoint, pausing between pages to respect the
    ~2 req/min limit. Returns the tidy frame from :func:`parse_measurements`,
    de-duplicated on timestamp. ``sensor_id`` can be passed to skip the lookup.
    """
    if days > _ARCHIVAL_MAX_DAYS:
        raise ValueError(
            f"archival endpoint accepts at most {_ARCHIVAL_MAX_DAYS} days per call, got {days}"
        )

    sid = sensor_id if sensor_id is not None else get_sensor_id(station_id, pollutant)
    url = f"{config.GIOS_API_BASE}/archivalData/getDataBySensor/{sid}"
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


def fetch_archival_pm25(
    station_id: int, days: int = _ARCHIVAL_MAX_DAYS, page_size: int = 5000
) -> pd.DataFrame:
    """Backwards-compatible archival PM2.5 fetch."""
    return fetch_archival(station_id, config.PM25_CODE, days=days, page_size=page_size)


def parse_aqindex(payload: dict) -> dict:
    """Parse an ``aqindex/getIndex`` payload into a compact dict.

    Pure function — no network. Returns the overall index plus the per-pollutant
    sub-indices (value + Polish category name), skipping pollutants with no data.
    """
    aq = payload.get("AqIndex", {})
    result = {
        "station_id": aq.get("Identyfikator stacji pomiarowej"),
        "calculated_at": aq.get("Data wykonania obliczeń indeksu"),
        "overall": {
            "value": aq.get("Wartość indeksu"),
            "category": aq.get("Nazwa kategorii indeksu"),
        },
        "critical_pollutant": aq.get("Kod zanieczyszczenia krytycznego"),
        "pollutants": {},
    }
    for pollutant in AQI_POLLUTANTS:
        value = aq.get(f"Wartość indeksu dla wskaźnika {pollutant}")
        category = aq.get(f"Nazwa kategorii indeksu dla wskaźnika {pollutant}")
        if value is not None or category is not None:
            result["pollutants"][pollutant] = {"value": value, "category": category}
    return result


def fetch_aqindex(station_id: int) -> dict:
    """Fetch the current Polish air-quality index for a station."""
    return parse_aqindex(_get(f"{config.GIOS_API_BASE}/aqindex/getIndex/{station_id}"))
