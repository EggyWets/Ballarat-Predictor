"""
data_fetcher.py
────────────────────────────────────────────────────────────────────────────
Fetches all raw data for the Ballarat Busyness Forecaster:
  - Pedestrian / foot-traffic counts from Ballarat Open Data (OpenDataSoft API)
  - Historical & forecast weather from Open-Meteo (completely free, no key)
  - Victoria public holidays via the `holidays` Python library
  - Curated Ballarat local events and school holiday calendar
"""

import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta
import holidays
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# GEO CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BALLARAT_LAT = -37.5622
BALLARAT_LON = 143.8503

BALLARAT_API_BASE = "https://data.ballarat.vic.gov.au/api/explore/v2.1/catalog/datasets"
OPEN_METEO_HISTORICAL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST   = "https://api.open-meteo.com/v1/forecast"

WEATHER_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "windspeed_10m_max",
    "weathercode",
    "sunshine_duration",
]

# ─────────────────────────────────────────────────────────────────────────────
# LOCAL EVENTS CALENDAR
# Edit this list to keep events current.
# impact: 1=minor bump | 2=moderate | 3=major crowd puller
# ─────────────────────────────────────────────────────────────────────────────
LOCAL_EVENTS = [
    {"name": "Ballarat Begonia Festival",              "month": 3,  "start_day": 7,  "end_day": 9,  "impact": 3},
    {"name": "Ballarat Show",                          "month": 10, "start_day": 24, "end_day": 25, "impact": 3},
    {"name": "White Night Ballarat",                   "month": 8,  "start_day": 16, "end_day": 17, "impact": 3},
    {"name": "Ballarat International Foto Biennale",   "month": 8,  "start_day": 1,  "end_day": 31, "impact": 2},
    {"name": "Ballarat Swap Meet",                     "month": 4,  "start_day": 26, "end_day": 27, "impact": 2},
    {"name": "Sovereign Hill Aura Night Show Season",  "month": 6,  "start_day": 1,  "end_day": 30, "impact": 2},
    {"name": "AFL Grand Final Eve",                    "month": 9,  "start_day": 26, "end_day": 26, "impact": 2},
    {"name": "Melbourne Cup Day",                      "month": 11, "start_day": 4,  "end_day": 4,  "impact": 2},
    {"name": "Christmas Market Season",                "month": 12, "start_day": 1,  "end_day": 24, "impact": 2},
    {"name": "New Year Period",                        "month": 12, "start_day": 28, "end_day": 31, "impact": 1},
    {"name": "Back to School Rush",                    "month": 1,  "start_day": 27, "end_day": 31, "impact": 1},
    {"name": "Easter Long Weekend",                    "month": 4,  "start_day": 17, "end_day": 21, "impact": 2},
    {"name": "Queen's Birthday Weekend",               "month": 6,  "start_day": 7,  "end_day": 9,  "impact": 1},
    {"name": "Ballarat Heritage Weekend",              "month": 5,  "start_day": 10, "end_day": 11, "impact": 2},
]

# Victorian school holiday approximate windows: (month, start_day, end_day_or_next_month_day, wraps)
# wraps=True means end_day is in the NEXT calendar month
SCHOOL_HOLIDAYS = [
    (1,  1,  31,  False, "Summer holidays"),
    (4,  5,  20,  False, "Autumn school holidays"),
    (7,  5,  20,  False, "Winter school holidays"),
    (9, 20,  4,   True,  "Spring school holidays"),   # wraps into October
    (12, 20, 31,  False, "Summer holidays"),
]


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC HOLIDAYS
# ─────────────────────────────────────────────────────────────────────────────
def get_vic_holidays(year: int) -> dict:
    """Return {date: holiday_name} for Victorian public holidays in a given year."""
    vic = holidays.Australia(state="VIC", years=year)
    return dict(vic)


def is_public_holiday(d: date):
    """Returns (bool, holiday_name_or_empty_string)."""
    vic = get_vic_holidays(d.year)
    name = vic.get(d, "")
    return bool(name), name


def is_eve_of_holiday(d: date) -> bool:
    """True if tomorrow is a Victorian public holiday."""
    tomorrow = d + timedelta(days=1)
    vic = get_vic_holidays(tomorrow.year)
    return tomorrow in vic


def is_long_weekend(d: date) -> bool:
    """True if the date falls within a 3-day long weekend involving a public holiday."""
    for offset in [-1, 0, 1, 2]:
        check = d + timedelta(days=offset)
        is_ph, _ = is_public_holiday(check)
        if is_ph and check.weekday() == 0:  # Monday public holiday = long weekend
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL EVENTS
# ─────────────────────────────────────────────────────────────────────────────
def get_events_for_date(d: date) -> list:
    """Return list of event dicts active on date d."""
    active = []
    for ev in LOCAL_EVENTS:
        start_month = ev["month"]
        s_day = ev["start_day"]
        e_day = ev["end_day"]
        year  = d.year

        # Handle year-end wrap (e.g. Dec 28 – Jan 2 — none currently but future-proof)
        try:
            start = date(year, start_month, s_day)
            if e_day >= s_day:
                end = date(year, start_month, e_day)
            else:
                # end_day in next month
                next_month = start_month % 12 + 1
                end_year   = year if next_month > 1 else year + 1
                end = date(end_year, next_month, e_day)

            if start <= d <= end:
                active.append(ev)
        except ValueError:
            continue
    return active


def get_event_impact_for_date(d: date) -> int:
    """Return highest event impact score (0–3) for the date."""
    events = get_events_for_date(d)
    if not events:
        return 0
    return max(e["impact"] for e in events)


def is_school_holiday(d: date) -> bool:
    """Approximate check for Victorian school holiday periods."""
    for month, start_day, end_day, wraps, _ in SCHOOL_HOLIDAYS:
        if not wraps:
            if d.month == month and start_day <= d.day <= end_day:
                return True
        else:
            next_month = month % 12 + 1
            if (d.month == month and d.day >= start_day) or \
               (d.month == next_month and d.day <= end_day):
                return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# BALLARAT OPEN DATA — FOOT TRAFFIC
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_dataset_paginated(dataset_id: str, max_records: int = 100000) -> pd.DataFrame:
    """
    Fetch all records from a Ballarat OpenDataSoft dataset using pagination.
    The API max per request is 100; we page through until done.
    """
    url = f"{BALLARAT_API_BASE}/{dataset_id}/records"
    page_size = 100
    all_records = []
    offset = 0

    try:
        while offset < max_records:
            params = {"limit": page_size, "offset": offset}
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            records = data.get("results", [])
            if not records:
                break
            all_records.extend(records)
            offset += len(records)
            # If we got fewer than page_size, we've hit the end
            if len(records) < page_size:
                break
            logger.info(f"{dataset_id}: fetched {offset} records so far...")

        if not all_records:
            logger.warning(f"Dataset '{dataset_id}' returned no records.")
            return pd.DataFrame()

        df = pd.json_normalize(all_records)
        df.columns = [c.lower().replace(" ", "_").replace(".", "_") for c in df.columns]
        logger.info(f"{dataset_id}: loaded {len(df)} total records. Columns: {df.columns.tolist()}")
        return df

    except Exception as e:
        logger.error(f"Error fetching '{dataset_id}': {e}")
        return pd.DataFrame()


def _find_col(df: pd.DataFrame, keywords: list) -> str | None:
    for kw in keywords:
        matches = [c for c in df.columns if kw in c]
        if matches:
            return matches[0]
    return None


def fetch_infrared_counters() -> pd.DataFrame:
    """
    Fetch infrared counter data. Real API fields: counter, datetime, count
    """
    df = _fetch_dataset_paginated("infrared-counters", max_records=400000)
    if df.empty:
        return pd.DataFrame(columns=["date", "total_count"])

    # Use known field names first, fall back to keyword search
    date_col  = "datetime" if "datetime" in df.columns else _find_col(df, ["datetime", "date", "time"])
    count_col = "count"    if "count"    in df.columns else _find_col(df, ["count", "total", "pedestrian", "cyclist"])

    if not date_col or not count_col:
        logger.warning(f"infrared-counters: unexpected columns {df.columns.tolist()}")
        return pd.DataFrame(columns=["date", "total_count"])

    df["date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_convert("Australia/Melbourne").dt.date
    df["total_count"] = pd.to_numeric(df[count_col], errors="coerce")
    daily = df.groupby("date")["total_count"].sum().reset_index()
    logger.info(f"infrared-counters: {len(daily)} daily records from {daily['date'].min()} to {daily['date'].max()}")
    return daily.dropna()


def fetch_people_counts() -> pd.DataFrame:
    """
    Fetch people count data. Real API fields: counter, datetime, count
    """
    df = _fetch_dataset_paginated("people-counts", max_records=400000)
    if df.empty:
        return pd.DataFrame(columns=["date", "total_count"])

    date_col  = "datetime" if "datetime" in df.columns else _find_col(df, ["datetime", "date", "time"])
    count_col = "count"    if "count"    in df.columns else _find_col(df, ["count", "total", "people"])

    if not date_col or not count_col:
        return pd.DataFrame(columns=["date", "total_count"])

    df["date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_convert("Australia/Melbourne").dt.date
    df["count"] = pd.to_numeric(df[count_col], errors="coerce")
    daily = df.groupby("date")["count"].sum().reset_index()
    daily.columns = ["date", "total_count"]
    logger.info(f"people-counts: {len(daily)} daily records")
    return daily.dropna()


def fetch_parking_transactions() -> pd.DataFrame:
    """
    Fetch parking transaction data.
    """
    df = _fetch_dataset_paginated("parking-transactions", max_records=100000)
    if df.empty:
        return pd.DataFrame(columns=["date", "parking_count"])

    date_col  = "datetime" if "datetime" in df.columns else _find_col(df, ["datetime", "date", "time"])
    count_col = "count"    if "count"    in df.columns else _find_col(df, ["count", "transaction", "number", "total"])

    if not date_col:
        return pd.DataFrame(columns=["date", "parking_count"])

    df["date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_convert("Australia/Melbourne").dt.date
    df["count"] = pd.to_numeric(df[count_col], errors="coerce") if count_col else 1
    daily = df.groupby("date")["count"].sum().reset_index()
    daily.columns = ["date", "parking_count"]
    logger.info(f"parking-transactions: {len(daily)} daily records")
    return daily.dropna()


def combine_traffic_data() -> pd.DataFrame:
    """
    Merge all traffic/people data sources into a single daily busyness index (0–100).
    """
    sources = [
        (fetch_infrared_counters(), "total_count"),
        (fetch_people_counts(),     "total_count"),
        (fetch_parking_transactions(), "parking_count"),
    ]

    frames = []
    for df, col in sources:
        if df.empty or col not in df.columns:
            continue
        mn, mx = df[col].min(), df[col].max()
        if mx > mn:
            df = df.copy()
            df["norm"] = (df[col] - mn) / (mx - mn) * 100.0
        else:
            df = df.copy()
            df["norm"] = 50.0
        frames.append(df[["date", "norm"]])

    if not frames:
        return pd.DataFrame(columns=["date", "busyness_index"])

    combined = (
        pd.concat(frames)
        .groupby("date")["norm"]
        .mean()
        .reset_index()
        .rename(columns={"norm": "busyness_index"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# WEATHER — OPEN-METEO (free, no API key required)
# ─────────────────────────────────────────────────────────────────────────────
def _meteo_request(url: str, params: dict) -> pd.DataFrame:
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data  = resp.json()
        daily = data.get("daily", {})
        df    = pd.DataFrame(daily)
        df["date"] = pd.to_datetime(df["time"]).dt.date
        return df.drop(columns=["time"])
    except Exception as e:
        logger.error(f"Open-Meteo request failed: {e}")
        return pd.DataFrame()


def fetch_historical_weather(start: date, end: date) -> pd.DataFrame:
    params = {
        "latitude":   BALLARAT_LAT,
        "longitude":  BALLARAT_LON,
        "start_date": start.isoformat(),
        "end_date":   end.isoformat(),
        "daily":      ",".join(WEATHER_VARS),
        "timezone":   "Australia/Melbourne",
    }
    return _meteo_request(OPEN_METEO_HISTORICAL, params)


def fetch_forecast_weather(days_ahead: int = 16) -> pd.DataFrame:
    params = {
        "latitude":      BALLARAT_LAT,
        "longitude":     BALLARAT_LON,
        "daily":         ",".join(WEATHER_VARS),
        "timezone":      "Australia/Melbourne",
        "forecast_days": min(days_ahead, 16),
    }
    return _meteo_request(OPEN_METEO_FORECAST, params)


def get_weather_for_date(d: date) -> dict:
    """
    Returns a weather dict for a specific date.
    Automatically chooses historical archive or forecast.
    """
    today = date.today()
    if d > today:
        df = fetch_forecast_weather(days_ahead=16)
    else:
        df = fetch_historical_weather(d - timedelta(days=1), d + timedelta(days=1))

    if df.empty:
        return {}
    row = df[df["date"] == d]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# WMO WEATHER CODE LOOKUP
# ─────────────────────────────────────────────────────────────────────────────
# (code: description, emoji, outdoor_crowd_multiplier 0–1)
WMO_CODES = {
    0:  ("Clear sky",              "☀️",  1.00),
    1:  ("Mainly clear",           "🌤️", 0.95),
    2:  ("Partly cloudy",          "⛅",  0.80),
    3:  ("Overcast",               "☁️",  0.65),
    45: ("Foggy",                  "🌫️", 0.55),
    48: ("Icy fog",                "🌫️", 0.45),
    51: ("Light drizzle",          "🌦️", 0.55),
    53: ("Moderate drizzle",       "🌧️", 0.40),
    55: ("Dense drizzle",          "🌧️", 0.30),
    61: ("Slight rain",            "🌧️", 0.50),
    63: ("Moderate rain",          "🌧️", 0.30),
    65: ("Heavy rain",             "🌧️", 0.15),
    71: ("Slight snow",            "🌨️", 0.35),
    73: ("Moderate snow",          "🌨️", 0.25),
    75: ("Heavy snow",             "❄️",  0.10),
    80: ("Rain showers",           "🌦️", 0.45),
    81: ("Moderate showers",       "🌧️", 0.30),
    82: ("Violent showers",        "⛈️", 0.10),
    95: ("Thunderstorm",           "⛈️", 0.10),
    99: ("Thunderstorm + hail",    "⛈️", 0.05),
}

def describe_weather_code(code) -> tuple:
    """Return (description, emoji, crowd_multiplier) for a WMO weather code."""
    try:
        code = int(code)
    except (TypeError, ValueError):
        return ("Unknown conditions", "❓", 0.70)
    return WMO_CODES.get(code, ("Unknown conditions", "❓", 0.70))
