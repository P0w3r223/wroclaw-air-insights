"""Tests for the pure parsing functions of the ingest clients (no network)."""

import pandas as pd

from wroclaw_air_insights.ingest import gios, weather

GIOS_PAYLOAD = {
    "@context": {"meta": "..."},
    "Lista danych pomiarowych": [
        {"Kod stanowiska": "X", "Data": "2026-07-17 11:00:00", "Wartość": 15.7},
        {"Kod stanowiska": "X", "Data": "2026-07-17 10:00:00", "Wartość": None},
        {"Kod stanowiska": "X", "Data": "2026-07-17 12:00:00", "Wartość": 14.1},
    ],
    "totalPages": 1,
}

OPEN_METEO_PAYLOAD = {
    "hourly": {
        "time": ["2026-07-17T00:00", "2026-07-17T01:00"],
        "temperature_2m": [22.5, 21.6],
        "boundary_layer_height": [15.0, 35.0],
    }
}


def test_parse_measurements_sorts_and_handles_null():
    df = gios.parse_measurements(GIOS_PAYLOAD)
    assert list(df.columns) == list(gios.MEASUREMENT_COLUMNS)
    assert len(df) == 3
    assert df["timestamp"].is_monotonic_increasing
    assert int(df["value"].isna().sum()) == 1  # null -> NaN


def test_parse_measurements_empty_payload():
    assert gios.parse_measurements({"@context": {}}).empty


def test_parse_hourly_maps_arrays_to_columns():
    df = weather.parse_hourly(OPEN_METEO_PAYLOAD)
    assert list(df.columns) == ["timestamp", "temperature_2m", "boundary_layer_height"]
    assert len(df) == 2
    assert df["timestamp"].is_monotonic_increasing


def test_parse_hourly_empty_payload():
    assert weather.parse_hourly({}).empty


AQINDEX_PAYLOAD = {
    "AqIndex": {
        "Identyfikator stacji pomiarowej": 129,
        "Data wykonania obliczeń indeksu": "2026-07-17 12:20:35",
        "Wartość indeksu": 1,
        "Nazwa kategorii indeksu": "Dobry",
        "Kod zanieczyszczenia krytycznego": "PM2.5",
        "Wartość indeksu dla wskaźnika NO2": 0,
        "Nazwa kategorii indeksu dla wskaźnika NO2": "Bardzo dobry",
        "Wartość indeksu dla wskaźnika PM2.5": 1,
        "Nazwa kategorii indeksu dla wskaźnika PM2.5": "Dobry",
        "Wartość indeksu dla wskaźnika O3": None,
        "Nazwa kategorii indeksu dla wskaźnika O3": None,
    }
}


def test_parse_aqindex_extracts_overall_and_subindices():
    result = gios.parse_aqindex(AQINDEX_PAYLOAD)
    assert result["station_id"] == 129
    assert result["overall"]["category"] == "Dobry"
    assert result["critical_pollutant"] == "PM2.5"
    assert result["pollutants"]["NO2"]["category"] == "Bardzo dobry"
    assert result["pollutants"]["PM2.5"]["value"] == 1
    assert "O3" not in result["pollutants"]  # pollutants with no data are skipped


def test_parse_aqindex_empty_payload():
    result = gios.parse_aqindex({})
    assert result["overall"]["value"] is None
    assert result["pollutants"] == {}
