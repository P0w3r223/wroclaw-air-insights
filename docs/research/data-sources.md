# Data Sources — wroclaw-air-insights

Date: 2026-07-17
Status: accepted
Author: P0w3r223 + Claude
Related to: portfolio plan (project A1), forecast module

---

Synthesis of the research + a live probe of the APIs (2026-07-17), used to decide
where the project gets its data. Two decisions were locked in with the project owner:
**target measurements come directly from GIOŚ** (Polish reference data — a CV
differentiator), and **weather features come from Open-Meteo**.

## Decision summary

| Purpose | Source | Why |
|---------|--------|-----|
| Target PM2.5 — history (training) | GIOŚ **API** `archivalData/getDataBySensor/{sensorId}` | Programmatic hourly history, up to **366 days per request** (`dateFrom`/`dateTo` or `dayNumber`) + pagination — no need to scrape the JS archive page |
| Target PM2.5 — recent / live | GIOŚ **API v1/rest** `data/getData/{sensorId}` | Last ~3 days, hourly; used for refresh / inference |
| Weather features — history (training) | Open-Meteo **Historical Forecast API** | Same models as live forecast → no train/serve distribution shift |
| Weather features — forecast (24h) | Open-Meteo **Forecast API** | Consistent variables incl. `boundary_layer_height` |

Weather is fetched for a single point (Wrocław ≈ 51.11, 17.03), hourly, `timezone=Europe/Warsaw`.

## Wrocław stations (probed live, GIOŚ API v1/rest)

New GIOŚ API base (the old services were retired 2025-06-30 — many old tutorials point at
dead endpoints): `https://api.gios.gov.pl/pjp-api/v1/rest/`. Response is JSON-LD with
**Polish keys** and pagination.

| Station | ID | PM2.5 | Notes |
|---------|-----|-------|-------|
| al. Wiśniowa / Powst. Śląskich | 129 | ✅ sensor **744** — automatic hourly (`DsWrocAlWisn-PM2.5-1g`) | Primary live source + target; also CO, NO, NO2, NOx |
| Na Grobli | 115 | ⚠️ sensor **645** — **manual** | `getData` returns error `API-ERR-100003`; results only via archive, 4–8 weeks delayed. History-only source |
| Bartnicza | 114 | ❌ none | Traffic station: NO, NO2, NOx, O3 |

> Do **not** hardcode sensor IDs long-term — fetch them from `station/sensors/{stationId}`,
> because sensor→pollutant mapping can change between years.

## Response formats (verified)

**GIOŚ `data/getData/{sensorId}`** — key `"Lista danych pomiarowych"`, list of:
```json
{ "Kod stanowiska": "DsWrocAlWisn-PM2.5-1g", "Data": "2026-07-17 11:00:00", "Wartość": 15.7 }
```
`Wartość` is a float **or `null`** (missing readings — must be handled). `Data` is local time.

**Open-Meteo** — parallel arrays under `hourly` (`time` + one array per variable) with
`hourly_units`; fields include `temperature_2m`, `relative_humidity_2m`, `dew_point_2m`,
`wind_speed_10m`, `wind_direction_10m`, `surface_pressure`, `precipitation`, `cloud_cover`,
`boundary_layer_height`.

## Rate limits & licensing

- **GIOŚ:** archive/sensors/getData throttled to ~**2 req/min**; live index up to 1500 req/min.
  Reuse of public sector information — attribution to **GIOŚ** required. For bulk history use
  the archive files, not the API.
- **Open-Meteo:** no API key for non-commercial use; ~10k req/day fair-use (unconstraining for
  one point). Licensed **CC BY 4.0** — attribution to Open-Meteo + CAMS required.

## Traps to remember

- **Time zone:** align GIOŚ (local time) and Open-Meteo (`Europe/Warsaw`) — keep everything on
  one consistent clock (UTC internally, or Europe/Warsaw throughout) to avoid off-by-one joins.
- **Missing values:** real gaps exist in station data (sensor downtime, calibration) — plan
  imputation; never assume a continuous series.
- **Archive file format** (CSV vs XLSX, exact columns, missing-value marker) must be confirmed on
  a downloaded file before finalizing the parser.
- **Train/serve shift:** train weather features on Historical Forecast API, not on ERA5 reanalysis,
  so training data matches what the model sees at inference.

## Sources

- GIOŚ API: https://powietrze.gios.gov.pl/pjp/content/api · Swagger: https://api.gios.gov.pl/pjp-api/swagger-ui/index.html
- GIOŚ archive: https://powietrze.gios.gov.pl/pjp/archives
- Open-Meteo: https://open-meteo.com/en/docs (forecast) · /historical-forecast-api · /docs/air-quality-api
- OpenAQ (fallback aggregator, GIOŚ-sourced): https://docs.openaq.org
