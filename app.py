"""
app.py — Ballarat Busyness Forecaster
All logic merged into one file to eliminate circular import issues on Render.
Run with: streamlit run app.py
"""

# ── Standard library imports ──────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import logging
import calendar as cal_module
from datetime import date, timedelta

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONSTANTS & LOOKUP TABLES
# ═════════════════════════════════════════════════════════════════════════════
BALLARAT_LAT = -37.5622
BALLARAT_LON = 143.8503
BALLARAT_API  = "https://data.ballarat.vic.gov.au/api/explore/v2.1/catalog/datasets"
METEO_HIST    = "https://archive-api.open-meteo.com/v1/archive"
METEO_FORE    = "https://api.open-meteo.com/v1/forecast"

WEATHER_VARS = [
    "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
    "windspeed_10m_max", "weathercode", "sunshine_duration",
]

WMO_CODES = {
    0:  ("Clear sky",           "☀️",  1.00),
    1:  ("Mainly clear",        "🌤️", 0.95),
    2:  ("Partly cloudy",       "⛅",  0.80),
    3:  ("Overcast",            "☁️",  0.65),
    45: ("Foggy",               "🌫️", 0.55),
    48: ("Icy fog",             "🌫️", 0.45),
    51: ("Light drizzle",       "🌦️", 0.55),
    53: ("Moderate drizzle",    "🌧️", 0.40),
    55: ("Dense drizzle",       "🌧️", 0.30),
    61: ("Slight rain",         "🌧️", 0.50),
    63: ("Moderate rain",       "🌧️", 0.30),
    65: ("Heavy rain",          "🌧️", 0.15),
    71: ("Slight snow",         "🌨️", 0.35),
    73: ("Moderate snow",       "🌨️", 0.25),
    75: ("Heavy snow",          "❄️",  0.10),
    80: ("Rain showers",        "🌦️", 0.45),
    81: ("Moderate showers",    "🌧️", 0.30),
    82: ("Violent showers",     "⛈️", 0.10),
    95: ("Thunderstorm",        "⛈️", 0.10),
    99: ("Thunderstorm + hail", "⛈️", 0.05),
}

LOCAL_EVENTS = [
    {"name": "Ballarat Begonia Festival",            "month": 3,  "start_day": 7,  "end_day": 9,  "impact": 3},
    {"name": "Ballarat Show",                        "month": 10, "start_day": 24, "end_day": 25, "impact": 3},
    {"name": "White Night Ballarat",                 "month": 8,  "start_day": 16, "end_day": 17, "impact": 3},
    {"name": "Ballarat International Foto Biennale", "month": 8,  "start_day": 1,  "end_day": 31, "impact": 2},
    {"name": "Ballarat Swap Meet",                   "month": 4,  "start_day": 26, "end_day": 27, "impact": 2},
    {"name": "Sovereign Hill Aura Night Season",     "month": 6,  "start_day": 1,  "end_day": 30, "impact": 2},
    {"name": "AFL Grand Final Eve",                  "month": 9,  "start_day": 26, "end_day": 26, "impact": 2},
    {"name": "Melbourne Cup Day",                    "month": 11, "start_day": 4,  "end_day": 4,  "impact": 2},
    {"name": "Christmas Market Season",              "month": 12, "start_day": 1,  "end_day": 24, "impact": 2},
    {"name": "New Year Period",                      "month": 12, "start_day": 28, "end_day": 31, "impact": 1},
    {"name": "Back to School Rush",                  "month": 1,  "start_day": 27, "end_day": 31, "impact": 1},
    {"name": "Easter Long Weekend",                  "month": 4,  "start_day": 17, "end_day": 21, "impact": 2},
    {"name": "Ballarat Heritage Weekend",            "month": 5,  "start_day": 10, "end_day": 11, "impact": 2},
]

SCHOOL_HOLIDAYS = [
    (1,  1,  31,  False),
    (4,  5,  20,  False),
    (7,  5,  20,  False),
    (9,  20, 4,   True),   # wraps into October
    (12, 20, 31,  False),
]

FEATURE_COLS = [
    "day_of_week", "is_weekend", "month", "day_of_month", "week_of_year",
    "is_summer", "is_autumn", "is_winter", "is_spring",
    "is_public_holiday", "is_eve_of_holiday", "is_long_weekend", "is_school_holiday",
    "has_event", "event_impact", "num_events",
    "temp_max", "temp_min", "precipitation", "wind_max",
    "weather_code", "sunshine_hours",
    "is_rainy", "is_hot", "is_cold", "is_pleasant",
    "is_free_parking_day",
    # ── Lag features: historical same-day lookups ─────────────────────────
    "lag_same_week_last_year",   # observed score ~52 weeks ago, same day-of-week
    "lag_avg_month_dow",         # avg score for this month + day-of-week combo
    "lag_rolling_4wk_dow",       # avg of last 4 occurrences of this day-of-week
]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CALENDAR HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def get_vic_holidays(year):
    import holidays as hol
    return dict(hol.Australia(state="VIC", years=year))

def is_public_holiday(d):
    vic = get_vic_holidays(d.year)
    name = vic.get(d, "")
    return bool(name), name

def is_eve_of_holiday(d):
    tomorrow = d + timedelta(days=1)
    vic = get_vic_holidays(tomorrow.year)
    return tomorrow in vic

def is_long_weekend(d):
    for offset in [-1, 0, 1, 2]:
        check = d + timedelta(days=offset)
        is_ph, _ = is_public_holiday(check)
        if is_ph and check.weekday() == 0:
            return True
    return False

def is_school_holiday(d):
    for month, start_day, end_day, wraps in SCHOOL_HOLIDAYS:
        if not wraps:
            if d.month == month and start_day <= d.day <= end_day:
                return True
        else:
            next_month = month % 12 + 1
            if (d.month == month and d.day >= start_day) or \
               (d.month == next_month and d.day <= end_day):
                return True
    return False

def get_events_for_date(d):
    active = []
    for ev in LOCAL_EVENTS:
        try:
            start = date(d.year, ev["month"], ev["start_day"])
            end   = date(d.year, ev["month"], ev["end_day"])
            if start <= d <= end:
                active.append(ev)
        except ValueError:
            continue
    return active

def get_event_impact_for_date(d):
    events = get_events_for_date(d)
    return max((e["impact"] for e in events), default=0)

def describe_weather_code(code):
    try:
        code = int(code)
    except (TypeError, ValueError):
        return ("Unknown conditions", "❓", 0.70)
    return WMO_CODES.get(code, ("Unknown conditions", "❓", 0.70))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DATA FETCHING
# ═════════════════════════════════════════════════════════════════════════════
def _fetch_dataset_paginated(dataset_id, max_records=100000):
    url = f"{BALLARAT_API}/{dataset_id}/records"
    all_records, offset = [], 0
    try:
        while offset < max_records:
            resp = requests.get(url, params={"limit": 100, "offset": offset}, timeout=30)
            resp.raise_for_status()
            records = resp.json().get("results", [])
            if not records:
                break
            all_records.extend(records)
            offset += len(records)
            if len(records) < 100:
                break
        if not all_records:
            return pd.DataFrame()
        df = pd.json_normalize(all_records)
        df.columns = [c.lower().replace(" ", "_").replace(".", "_") for c in df.columns]
        return df
    except Exception as e:
        logger.error(f"Error fetching {dataset_id}: {e}")
        return pd.DataFrame()

def _find_col(df, keywords):
    for kw in keywords:
        matches = [c for c in df.columns if kw in c]
        if matches:
            return matches[0]
    return None

def _to_daily(df, date_col, count_col, out_col):
    df = df.copy()
    df["date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_convert("Australia/Melbourne").dt.date
    df["val"]  = pd.to_numeric(df[count_col], errors="coerce")
    daily = df.groupby("date")["val"].sum().reset_index()
    daily.columns = ["date", out_col]
    return daily.dropna()

def fetch_infrared_counters():
    df = _fetch_dataset_paginated("infrared-counters", max_records=400000)
    if df.empty:
        return pd.DataFrame(columns=["date", "total_count"])
    dc = "datetime" if "datetime" in df.columns else _find_col(df, ["datetime","date","time"])
    cc = "count"    if "count"    in df.columns else _find_col(df, ["count","total","pedestrian"])
    if not dc or not cc:
        return pd.DataFrame(columns=["date", "total_count"])
    return _to_daily(df, dc, cc, "total_count")

def fetch_people_counts():
    df = _fetch_dataset_paginated("people-counts", max_records=400000)
    if df.empty:
        return pd.DataFrame(columns=["date", "total_count"])
    dc = "datetime" if "datetime" in df.columns else _find_col(df, ["datetime","date","time"])
    cc = "count"    if "count"    in df.columns else _find_col(df, ["count","total","people"])
    if not dc or not cc:
        return pd.DataFrame(columns=["date", "total_count"])
    return _to_daily(df, dc, cc, "total_count")

def fetch_parking_transactions():
    df = _fetch_dataset_paginated("parking-transactions", max_records=100000)
    if df.empty:
        return pd.DataFrame(columns=["date", "parking_count"])
    dc = "datetime" if "datetime" in df.columns else _find_col(df, ["datetime","date","time"])
    cc = "count"    if "count"    in df.columns else _find_col(df, ["count","transaction","number","total"])
    if not dc:
        return pd.DataFrame(columns=["date", "parking_count"])
    return _to_daily(df, dc, cc or "count", "parking_count")

def combine_traffic_data():
    infrared = fetch_infrared_counters()
    people   = fetch_people_counts()
    parking  = fetch_parking_transactions()

    # Strip parking data on free-parking days (Sundays + public holidays)
    if not parking.empty:
        cache = {}
        def is_free(d):
            if isinstance(d, str): d = date.fromisoformat(d)
            if d.weekday() == 6: return True
            if d.year not in cache: cache[d.year] = get_vic_holidays(d.year)
            return d in cache[d.year]
        parking = parking[~parking["date"].apply(is_free)].copy()

    frames = []
    for df, col in [(infrared,"total_count"),(people,"total_count"),(parking,"parking_count")]:
        if df.empty or col not in df.columns: continue
        mn, mx = df[col].min(), df[col].max()
        df = df.copy()
        df["norm"] = (df[col]-mn)/(mx-mn)*100 if mx>mn else 50.0
        frames.append(df[["date","norm"]])

    if not frames:
        return pd.DataFrame(columns=["date","busyness_index"])

    return (pd.concat(frames).groupby("date")["norm"].mean()
            .reset_index().rename(columns={"norm":"busyness_index"})
            .sort_values("date").reset_index(drop=True))

def fetch_weather(start, end, forecast=False):
    url = METEO_FORE if forecast else METEO_HIST
    params = {
        "latitude": BALLARAT_LAT, "longitude": BALLARAT_LON,
        "daily": ",".join(WEATHER_VARS), "timezone": "Australia/Melbourne",
    }
    if forecast:
        params["forecast_days"] = 16
    else:
        params["start_date"] = start.isoformat()
        params["end_date"]   = end.isoformat()
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
        df = pd.DataFrame(daily)
        df["date"] = pd.to_datetime(df["time"]).dt.date
        return df.drop(columns=["time"])
    except Exception as e:
        logger.error(f"Weather fetch failed: {e}")
        return pd.DataFrame()

def get_weather_for_date(d):
    today = date.today()
    df = fetch_weather(None, None, forecast=True) if d >= today else fetch_weather(d, d)
    if df.empty: return {}
    row = df[df["date"] == d]
    return row.iloc[0].to_dict() if not row.empty else {}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════

def compute_lag_lookup(traffic_df):
    """
    Pre-compute a lookup dict from the historical traffic DataFrame so we can
    quickly retrieve lag features for any date without re-scanning the full table.

    Returns a dict with:
      - "by_date":     {date: busyness_index}  — raw daily lookup
      - "month_dow":   {(month, dow): mean}    — avg for each month+day-of-week combo
      - "by_iso_week_dow": {(iso_week, dow, year): busyness_index}
    """
    if traffic_df is None or traffic_df.empty:
        return {}

    df = traffic_df.copy()
    df["date"] = pd.to_datetime(df["date"].astype(str)).dt.date
    df = df.dropna(subset=["busyness_index"])

    by_date = dict(zip(df["date"], df["busyness_index"]))

    df["month"] = df["date"].apply(lambda d: d.month)
    df["dow"]   = df["date"].apply(lambda d: d.weekday())
    month_dow   = df.groupby(["month","dow"])["busyness_index"].mean().to_dict()

    return {
        "by_date":   by_date,
        "month_dow": month_dow,
        "df":        df,   # kept for rolling calc
    }


def get_lag_features(d, lag_lookup):
    """
    Compute the three lag features for date d using the pre-built lookup.
    Returns dict with the three lag values (or neutral 50.0 if no data available).
    """
    NEUTRAL = 50.0

    if not lag_lookup:
        return {
            "lag_same_week_last_year": NEUTRAL,
            "lag_avg_month_dow":       NEUTRAL,
            "lag_rolling_4wk_dow":     NEUTRAL,
        }

    by_date   = lag_lookup.get("by_date", {})
    month_dow = lag_lookup.get("month_dow", {})
    df_hist   = lag_lookup.get("df", pd.DataFrame())

    # ── Feature 1: same ISO week last year, same day-of-week ─────────────
    # Try exact 364-day lookback (52 weeks), then ±7 days if not found
    lag1 = NEUTRAL
    for offset in [364, 371, 357, 378, 350]:
        candidate = d - timedelta(days=offset)
        if candidate in by_date:
            lag1 = by_date[candidate]
            break

    # ── Feature 2: avg of this month + day-of-week across all history ────
    key  = (d.month, d.weekday())
    lag2 = month_dow.get(key, NEUTRAL)

    # ── Feature 3: rolling avg of last 4 same-weekday occurrences ────────
    lag3 = NEUTRAL
    if not df_hist.empty:
        same_dow = df_hist[df_hist["dow"] == d.weekday()].copy()
        same_dow = same_dow[same_dow["date"] < d].sort_values("date", ascending=False)
        recent4  = same_dow.head(4)
        if len(recent4) >= 2:
            lag3 = float(recent4["busyness_index"].mean())

    return {
        "lag_same_week_last_year": round(lag1, 2),
        "lag_avg_month_dow":       round(lag2, 2),
        "lag_rolling_4wk_dow":     round(lag3, 2),
    }


def build_features(d, weather=None, lag_lookup=None):
    if weather is None:
        weather = get_weather_for_date(d)
    is_ph, _ = is_public_holiday(d)
    events   = get_events_for_date(d)
    return {
        "day_of_week":         d.weekday(),
        "is_weekend":          int(d.weekday() >= 5),
        "month":               d.month,
        "day_of_month":        d.day,
        "week_of_year":        d.isocalendar()[1],
        "is_summer":           int(d.month in [12,1,2]),
        "is_autumn":           int(d.month in [3,4,5]),
        "is_winter":           int(d.month in [6,7,8]),
        "is_spring":           int(d.month in [9,10,11]),
        "is_public_holiday":   int(is_ph),
        "is_eve_of_holiday":   int(is_eve_of_holiday(d)),
        "is_long_weekend":     int(is_long_weekend(d)),
        "is_school_holiday":   int(is_school_holiday(d)),
        "has_event":           int(len(events) > 0),
        "event_impact":        get_event_impact_for_date(d),
        "num_events":          len(events),
        "temp_max":            float(weather.get("temperature_2m_max", 15) or 15),
        "temp_min":            float(weather.get("temperature_2m_min", 8)  or 8),
        "precipitation":       float(weather.get("precipitation_sum", 0)   or 0),
        "wind_max":            float(weather.get("windspeed_10m_max", 10)  or 10),
        "weather_code":        int(weather.get("weathercode", 2)           or 2),
        "sunshine_hours":      float(weather.get("sunshine_duration", 28800) or 28800) / 3600,
        "is_rainy":            int(float(weather.get("precipitation_sum",0) or 0) > 2.0),
        "is_hot":              int(float(weather.get("temperature_2m_max",15) or 15) > 28.0),
        "is_cold":             int(float(weather.get("temperature_2m_max",15) or 15) < 10.0),
        "is_pleasant":         int(15 <= float(weather.get("temperature_2m_max",15) or 15) <= 26
                                   and float(weather.get("precipitation_sum",0) or 0) < 1.0),
        "is_free_parking_day": int(d.weekday() == 6 or is_ph),
        # ── Lag features ─────────────────────────────────────────────────
        **get_lag_features(d, lag_lookup),
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SYNTHETIC DATA + MODEL
# ═════════════════════════════════════════════════════════════════════════════
def generate_synthetic_traffic(years=3):
    today, start = date.today(), date(date.today().year - years, 1, 1)
    rows, rng, d = [], np.random.default_rng(42), start
    monthly = {1:-5,2:-3,3:10,4:5,5:0,6:-8,7:-10,8:5,9:8,10:10,11:5,12:15}
    while d < today:
        base     = 50.0 + [0,5,5,8,15,30,18][d.weekday()]
        is_ph, _ = is_public_holiday(d)
        if d.weekday() == 6 or is_ph: base += 10
        base += monthly.get(d.month, 0)
        base += get_event_impact_for_date(d) * 12
        if is_ph: base += 20
        if is_school_holiday(d): base += 8
        rows.append({"date": d, "busyness_index": float(np.clip(base + rng.normal(0,8), 0, 100))})
        d += timedelta(days=1)
    return pd.DataFrame(rows)

def score_to_label(score):
    if score >= 80: return {"text":"Very Busy",  "colour":"#e74c3c","emoji":"🔴","tier":5}
    if score >= 65: return {"text":"Busy",        "colour":"#e67e22","emoji":"🟠","tier":4}
    if score >= 45: return {"text":"Moderate",    "colour":"#f1c40f","emoji":"🟡","tier":3}
    if score >= 25: return {"text":"Quiet",       "colour":"#2ecc71","emoji":"🟢","tier":2}
    return             {"text":"Very Quiet",   "colour":"#27ae60","emoji":"🟢","tier":1}

def build_reasons(d, feats, weather, score, data_stats=None, lag_lookup=None):
    reasons   = []
    dow_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow       = dow_names[d.weekday()]

    if d.weekday() == 5:
        reasons.append(("📅","Saturday","Saturdays are consistently the busiest day in Ballarat's CBD."))
    elif d.weekday() == 6:
        reasons.append(("📅","Sunday","Sundays bring moderate foot traffic — free parking encourages family visits."))
        reasons.append(("🅿️","Free Parking Day","Parking is free on Sundays. The model accounts for zero parking transactions and does not treat this as low activity."))
    elif d.weekday() == 4:
        reasons.append(("📅","Friday","Fridays are the second-busiest weekday, with after-work and evening activity."))
    else:
        reasons.append(("📅", dow, f"{dow}s are typically quieter weekdays in central Ballarat."))

    is_ph, ph_name = is_public_holiday(d)
    if is_ph:
        reasons.append(("🎉",f"Public Holiday: {ph_name}","Public holidays significantly boost CBD activity — markets, events and day-trippers."))
        reasons.append(("🅿️","Free Parking Day","Parking is free on public holidays. The model corrects for missing parking data."))
    elif is_eve_of_holiday(d):
        reasons.append(("🌙","Eve of Public Holiday","The day before a public holiday often sees elevated shopping and hospitality activity."))

    if feats.get("is_long_weekend") and not is_ph:
        reasons.append(("📆","Long Weekend","Long weekends draw visitors from Melbourne and surrounds to Ballarat."))
    if feats.get("is_school_holiday"):
        reasons.append(("🎒","School Holidays","School holiday periods increase family visits to Sovereign Hill and the Wildlife Park."))

    for ev in get_events_for_date(d):
        impact_desc = {1:"minor",2:"moderate",3:"major"}.get(ev["impact"],"notable")
        reasons.append(("🎪", ev["name"], f"This {impact_desc}-impact event increases visitor numbers in Ballarat."))

    code   = int(weather.get("weathercode", 2) or 2)
    desc, emoji, mult = describe_weather_code(code)
    temp   = float(weather.get("temperature_2m_max", 15) or 15)
    precip = float(weather.get("precipitation_sum", 0)   or 0)
    if mult >= 0.9:
        wx = f"Forecast: {desc}, {temp:.0f}°C — excellent conditions will encourage people into town."
    elif mult >= 0.6:
        wx = f"Forecast: {desc}, {temp:.0f}°C — reasonable conditions; mild impact on foot traffic."
    else:
        wx = f"Forecast: {desc}, {temp:.0f}°C with {precip:.1f}mm rain — poor conditions will reduce movement."
    reasons.append((emoji, f"Weather: {desc}", wx))

    if feats.get("is_hot"):  reasons.append(("🌡️","Hot Day (>28°C)","Very hot days can reduce CBD foot traffic as people seek cooler environments."))
    if feats.get("is_cold"): reasons.append(("🧊","Cold Day (<10°C)","Cold days tend to keep people at home and reduce casual shopping trips."))

    for key, (icon, label, detail) in {
        "is_summer": ("☀️","Summer Season","December–February: school holidays and warm weather boost tourism."),
        "is_winter": ("❄️","Winter Season","June–August: Ballarat winters are cold — the quietest period for CBD activity."),
        "is_spring": ("🌸","Spring Season","September–November: warming weather and events build momentum."),
        "is_autumn": ("🍂","Autumn Season","March–May: mild weather; post-summer visitor numbers remain solid."),
    }.items():
        if feats.get(key):
            reasons.append((icon, label, detail))
            break

    # ── Lag feature observations ─────────────────────────────────────────
    if lag_lookup and lag_lookup.get("by_date"):
        lag_feats   = get_lag_features(d, lag_lookup)
        dow_names_l = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow_label   = dow_names_l[d.weekday()]
        month_names_l = ["January","February","March","April","May","June",
                         "July","August","September","October","November","December"]
        month_label = month_names_l[d.month - 1]

        # Feature 1 — same week last year
        lag1 = lag_feats["lag_same_week_last_year"]
        if lag1 != 50.0:
            # Find which actual date was used
            actual_lag_date = None
            for offset in [364, 371, 357, 378, 350]:
                candidate = d - timedelta(days=offset)
                if candidate in lag_lookup.get("by_date", {}):
                    actual_lag_date = candidate
                    break
            date_str = actual_lag_date.strftime("%-d %B %Y") if actual_lag_date else "~1 year ago"
            lbl      = score_to_label(lag1)
            diff     = lag1 - feats.get("lag_avg_month_dow", 50)
            vs_str   = f"{abs(diff):.0f} points {'above' if diff > 0 else 'below'} the {month_label} {dow_label} average"
            reasons.append((
                "🔁", f"Same Week Last Year: {lbl['text']} ({lag1:.0f}/100)",
                f"On {date_str} — the equivalent {dow_label} in the same week last year — "
                f"the observed busyness score was {lag1:.0f}/100 ({lbl['text']}), "
                f"which was {vs_str}. This is the single strongest historical comparison available."
            ))
        else:
            reasons.append((
                "🔁", "Same Week Last Year: No Data",
                "No observed data found for the equivalent week last year. "
                "The model is relying on seasonal patterns and other signals for this date."
            ))

        # Feature 2 — avg for this month + day of week
        lag2    = lag_feats["lag_avg_month_dow"]
        lbl2    = score_to_label(lag2)
        by_date = lag_lookup.get("by_date", {})
        # Count how many matching month+dow observations exist
        n_obs = sum(1 for dt in by_date if dt.month == d.month and dt.weekday() == d.weekday())
        reasons.append((
            "📆", f"Historical {month_label} {dow_label}s: avg {lag2:.0f}/100",
            f"Across all {month_label} {dow_label}s in the training dataset ({n_obs} observations), "
            f"the average busyness score is {lag2:.0f}/100 ({lbl2['text']}). "
            f"This reflects the typical pattern for this specific month and day-of-week combination."
        ))

        # Feature 3 — rolling 4-week same-day average
        lag3 = lag_feats["lag_rolling_4wk_dow"]
        lbl3 = score_to_label(lag3)
        # Find what dates those 4 weeks were
        df_hist = lag_lookup.get("df", pd.DataFrame())
        if not df_hist.empty:
            recent = df_hist[df_hist["dow"] == d.weekday()]
            recent = recent[recent["date"] < d].sort_values("date", ascending=False).head(4)
            if len(recent) >= 2:
                date_range_str = f"{recent.iloc[-1]['date'].strftime('%-d %b')} – {recent.iloc[0]['date'].strftime('%-d %b %Y')}"
                reasons.append((
                    "📉", f"Recent {dow_label} Trend: avg {lag3:.0f}/100",
                    f"The last {len(recent)} {dow_label}s ({date_range_str}) averaged a busyness "
                    f"score of {lag3:.0f}/100 ({lbl3['text']}). "
                    f"This rolling trend captures any recent shifts in activity patterns."
                ))

    # ── Historical data observations ──────────────────────────────────────
    if data_stats and data_stats.get("source") == "live":
        dow_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow       = d.weekday()
        avg_by_dow = data_stats.get("avg_by_dow", {})
        overall    = data_stats.get("avg_overall", 50)
        total_days = data_stats.get("total_days", 0)
        start_str  = data_stats.get("date_range_start", "")
        end_str    = data_stats.get("date_range_end", "")
        busiest    = data_stats.get("busiest_dow", 5)
        quietest   = data_stats.get("quietest_dow", 1)
        pct_busy   = data_stats.get("pct_above_65", 0)

        # Day-of-week historical average
        if dow in avg_by_dow:
            dow_avg = avg_by_dow[dow]
            vs_overall = dow_avg - overall
            direction  = f"{abs(vs_overall):.0f} points above" if vs_overall > 0 else f"{abs(vs_overall):.0f} points below"
            reasons.append((
                "📊", f"Historical Pattern: {dow_names[dow]}s",
                f"Based on {total_days:,} days of real Ballarat sensor data "
                f"({start_str[:7]} to {end_str[:7]}), {dow_names[dow]}s average a busyness score of "
                f"{dow_avg:.0f}/100 — {direction} the overall daily average of {overall:.0f}. "
                f"The busiest day historically is {dow_names[busiest]}, "
                f"the quietest is {dow_names[quietest]}."
            ))

        # Infrared / pedestrian sensor note
        reasons.append((
            "🚶", "Pedestrian Sensor Data",
            f"This forecast draws on live infrared pedestrian counter data from Ballarat City Council sensors "
            f"located at Lake Wendouree, Loreto Point and key CBD locations. Across all training data, "
            f"{pct_busy:.0f}% of days exceeded the 'Busy' threshold (score ≥ 65)."
        ))

        # Parking data note
        is_ph, _ = is_public_holiday(d)
        if d.weekday() == 6 or is_ph:
            reasons.append((
                "🅿️", "Parking Data: Not Used Today",
                "Ballarat City Council parking transaction data is excluded for Sundays and public holidays "
                "as meters are not active. The forecast relies solely on pedestrian counter data and "
                "calendar/weather signals for these days."
            ))
        else:
            reasons.append((
                "🅿️", "Parking Transaction Data",
                "CBD parking transaction volumes from Ballarat City Council meters contribute to this forecast. "
                "Higher parking activity on weekdays strongly correlates with overall CBD busyness."
            ))

    return reasons

class BusynessModel:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.feature_importances = {}
        self.train_r2 = None
        self.data_source = "unknown"
        self.data_stats = {}   # stores observed averages for reason callouts
        self.lag_lookup = {}   # pre-computed historical lag lookup table

    def train(self, use_live_data=True):
        from xgboost import XGBRegressor
        from sklearn.model_selection import cross_val_score

        # Fetch each source individually so we can report diagnostics
        if use_live_data:
            infrared_raw = fetch_infrared_counters()
            people_raw   = fetch_people_counts()
            parking_raw  = fetch_parking_transactions()
            traffic      = combine_traffic_data()
        else:
            infrared_raw = pd.DataFrame()
            people_raw   = pd.DataFrame()
            parking_raw  = pd.DataFrame()
            traffic      = pd.DataFrame()

        if len(traffic) < 30:
            traffic = generate_synthetic_traffic(years=3)
            self.data_source = "synthetic"
        else:
            self.data_source = "live"

        min_date = min(traffic["date"])
        max_date = max(traffic["date"])
        if isinstance(min_date, str):
            min_date = date.fromisoformat(min_date)
            max_date = date.fromisoformat(max_date)

        weather_df = fetch_weather(min_date, max_date)

        # Build lag lookup once before the loop (not on every row)
        train_lag_lookup = compute_lag_lookup(traffic)

        rows = []
        for _, row in traffic.iterrows():
            d = row["date"]
            if isinstance(d, str): d = date.fromisoformat(d)
            w_row = weather_df[weather_df["date"] == d]
            w = w_row.iloc[0].to_dict() if not w_row.empty else {}
            feats = build_features(d, w, lag_lookup=train_lag_lookup)
            feats["busyness_index"] = row["busyness_index"]
            feats["date"] = d
            rows.append(feats)

        df = pd.DataFrame(rows).dropna(subset=FEATURE_COLS + ["busyness_index"])
        X  = df[FEATURE_COLS].astype(float)
        y  = df["busyness_index"].astype(float)

        self.model = XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0,
        )
        self.model.fit(X, y)
        cv = cross_val_score(self.model, X, y, cv=min(5, len(X)//20 or 2), scoring="r2")
        self.train_r2 = float(np.mean(cv))
        self.feature_importances = dict(sorted(zip(FEATURE_COLS, self.model.feature_importances_), key=lambda x: -x[1]))

        # Store lag lookup for use at prediction time
        self.lag_lookup = compute_lag_lookup(traffic)

        # ── Per-source day counts for diagnostics ─────────────────────────
        def _date_range(df_src):
            if df_src.empty or "date" not in df_src.columns:
                return "N/A", "N/A"
            dates = pd.to_datetime(df_src["date"].astype(str))
            return str(dates.min().date()), str(dates.max().date())

        infrared_start, infrared_end = _date_range(infrared_raw)
        people_start,   people_end   = _date_range(people_raw)
        parking_start,  parking_end  = _date_range(parking_raw)
        weather_start,  weather_end  = _date_range(weather_df)

        # Parking after free-parking-day filter
        if not parking_raw.empty:
            cache = {}
            def _is_free(d):
                if isinstance(d, str): d = date.fromisoformat(d)
                if d.weekday() == 6: return True
                if d.year not in cache: cache[d.year] = get_vic_holidays(d.year)
                return d in cache[d.year]
            parking_filtered = parking_raw[~parking_raw["date"].apply(_is_free)]
        else:
            parking_filtered = pd.DataFrame()

        # Cache observed data stats for use in reason summaries and diagnostics
        df["dow"] = df["date"].apply(lambda d: d.weekday() if hasattr(d, "weekday") else 0)
        self.data_stats = {
            "total_days":           len(df),
            "date_range_start":     str(min_date),
            "date_range_end":       str(max_date),
            "avg_by_dow":           df.groupby("dow")["busyness_index"].mean().to_dict(),
            "avg_overall":          float(df["busyness_index"].mean()),
            "busiest_dow":          int(df.groupby("dow")["busyness_index"].mean().idxmax()),
            "quietest_dow":         int(df.groupby("dow")["busyness_index"].mean().idxmin()),
            "pct_above_65":         float((df["busyness_index"] >= 65).mean() * 100),
            "source":               self.data_source,
            # Per-source diagnostics
            "infrared_days":        len(infrared_raw),
            "infrared_start":       infrared_start,
            "infrared_end":         infrared_end,
            "infrared_avg":         float(infrared_raw["total_count"].mean()) if not infrared_raw.empty and "total_count" in infrared_raw.columns else 0,
            "people_days":          len(people_raw),
            "people_start":         people_start,
            "people_end":           people_end,
            "parking_days_raw":     len(parking_raw),
            "parking_days_used":    len(parking_filtered),
            "parking_days_excluded":len(parking_raw) - len(parking_filtered),
            "parking_start":        parking_start,
            "parking_end":          parking_end,
            "parking_avg":          float(parking_raw["parking_count"].mean()) if not parking_raw.empty and "parking_count" in parking_raw.columns else 0,
            "weather_days":         len(weather_df),
            "weather_start":        weather_start,
            "weather_end":          weather_end,
        }
        self.is_trained = True

    def predict(self, d, weather=None):
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        if weather is None:
            weather = get_weather_for_date(d)
        feats = build_features(d, weather, lag_lookup=self.lag_lookup)
        X     = pd.DataFrame([feats])[FEATURE_COLS].astype(float)
        score = float(np.clip(self.model.predict(X)[0], 0, 100))
        return {
            "date": d, "score": score,
            "label": score_to_label(score),
            "feats": feats, "weather": weather,
            "events": get_events_for_date(d),
            "reasons": build_reasons(d, feats, weather, score, self.data_stats, self.lag_lookup),
        }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — STREAMLIT UI
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Ballarat Busyness Forecaster",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Source+Sans+3:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }
h1,h2,h3 { font-family: 'Playfair Display', Georgia, serif !important; }

/* Dark mode defaults (Streamlit ships dark) */
.main-title { font-family:'Playfair Display',Georgia,serif; font-size:2.6rem; font-weight:700; color:#ffffff; margin-bottom:0; }
.sub-title  { font-size:1.0rem; color:#c0ccd8; margin-top:0.2rem; margin-bottom:1.5rem; }
.sidebar-header { font-family:'Playfair Display',serif; font-size:1.3rem; color:#ffffff; margin-bottom:0.5rem; }
.reason-card  { background:#1e2a3a; border-left:4px solid #3a86ff; border-radius:0 8px 8px 0; padding:0.75rem 1rem; margin-bottom:0.6rem; }
.reason-heading { font-weight:600; color:#e8edf2; font-size:0.95rem; }
.reason-detail  { color:#8fa3b8; font-size:0.85rem; margin-top:0.15rem; }

/* Light mode overrides */
@media (prefers-color-scheme: light) {
    .main-title     { color:#1a1a2e; }
    .sub-title      { color:#5c6b7a; }
    .sidebar-header { color:#1a1a2e; }
    .reason-card    { background:#f8f9fc; }
    .reason-heading { color:#1a1a2e; }
    .reason-detail  { color:#5c6b7a; }
}

/* Shared (mode-independent) */
.score-card { background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%); border-radius:16px; padding:2rem; text-align:center; color:white; margin-bottom:1rem; }
.score-number { font-family:'Playfair Display',serif; font-size:5rem; font-weight:700; line-height:1; }
.score-label  { font-size:1.5rem; font-weight:600; margin-top:0.3rem; }
.score-date   { font-size:0.9rem; opacity:0.7; margin-top:0.5rem; }
.data-badge { display:inline-block; padding:2px 10px; border-radius:12px; font-size:0.75rem; font-weight:600; }
.badge-live  { background:#d4edda; color:#155724; }
.badge-synth { background:#fff3cd; color:#856404; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="🧠 Training forecast model — this takes ~30 seconds on first run...")
def load_model():
    m = BusynessModel()
    m.train(use_live_data=True)
    return m

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-header">🏙️ Ballarat Busyness</div>', unsafe_allow_html=True)
    st.caption("Forecast how busy Ballarat town centre will be on any day.")
    st.divider()
    st.markdown("### 📅 Select a Date")
    today = date.today()

    year_options        = list(range(today.year - 1, today.year + 3))
    selected_year       = st.selectbox("Year", year_options, index=year_options.index(today.year))
    month_names         = ["January","February","March","April","May","June","July","August","September","October","November","December"]
    selected_month_name = st.selectbox("Month", month_names, index=today.month - 1)
    selected_month      = month_names.index(selected_month_name) + 1
    days_in_month       = cal_module.monthrange(selected_year, selected_month)[1]
    default_day         = min(today.day, days_in_month) if (selected_year == today.year and selected_month == today.month) else 1
    selected_day        = st.selectbox("Day", list(range(1, days_in_month + 1)), index=default_day - 1)

    try:
        target_date = date(selected_year, selected_month, selected_day)
    except ValueError:
        target_date = today

    st.divider()
    show_week = st.checkbox("Show full week comparison", value=True)
    with st.expander("ℹ️ About this tool"):
        st.markdown("""
        **Data sources:**
        - 🚶 Ballarat Open Data (pedestrian & parking counts)
        - 🌦️ Open-Meteo weather (free, no API key)
        - 📅 Victorian public holidays
        - 🎪 Curated Ballarat events calendar

        **Model:** XGBoost trained on historical traffic + weather + calendar features.

        **Note:** Weather forecasts available ~16 days ahead.
        """)

# ── Main ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Ballarat Busyness Forecaster</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-powered busyness predictions for Ballarat, Victoria — powered by local open data, weather & events.</div>', unsafe_allow_html=True)

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.info("Make sure all dependencies are installed: `pip install -r requirements.txt`")
    st.stop()

badge_class = "badge-live" if model.data_source == "live" else "badge-synth"
badge_label = "Live Data" if model.data_source == "live" else "Synthetic Training Data"
badge_tip   = "Trained on real Ballarat pedestrian & parking data." if model.data_source == "live" \
              else "Ballarat Open Data API unavailable — model trained on realistic synthetic data."
st.markdown(f'<span class="data-badge {badge_class}">⚡ {badge_label}</span> <span style="font-size:0.8rem;color:#888">{badge_tip}</span>', unsafe_allow_html=True)

if model.data_source == "synthetic":
    st.warning("⚠️ The Ballarat Open Data API could not be reached. Predictions use realistic synthetic training data.", icon="⚠️")

st.divider()

with st.spinner(f"Forecasting {target_date.strftime('%A %-d %B %Y')}..."):
    try:
        result = model.predict(target_date)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

score   = result["score"]
label   = result["label"]
feats   = result["feats"]
weather = result["weather"]
reasons = result["reasons"]

col_score, col_reasons = st.columns([1, 2])

with col_score:
    st.markdown(f"""
    <div class="score-card">
        <div style="font-size:3rem">{label['emoji']}</div>
        <div class="score-number" style="color:{label['colour']}">{score:.0f}</div>
        <div style="font-size:0.85rem;opacity:0.6;margin-top:0.2rem">out of 100</div>
        <div class="score-label">{label['text']}</div>
        <div class="score-date">{target_date.strftime('%A, %-d %B %Y')}</div>
    </div>""", unsafe_allow_html=True)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"suffix":"/100","font":{"size":20}},
        gauge={
            "axis":{"range":[0,100],"tickwidth":1},
            "bar":{"color":label["colour"]},
            "steps":[
                {"range":[0,25],"color":"#e8f5e9"},{"range":[25,45],"color":"#c8e6c9"},
                {"range":[45,65],"color":"#fff9c4"},{"range":[65,80],"color":"#ffe0b2"},
                {"range":[80,100],"color":"#ffcdd2"},
            ],
            "threshold":{"line":{"color":"#1a1a2e","width":3},"thickness":0.8,"value":score},
        },
        domain={"x":[0,1],"y":[0,1]},
    ))
    fig_gauge.update_layout(height=220, margin=dict(l=20,r=20,t=20,b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_gauge, use_container_width=True)

    if weather:
        temp_max = weather.get("temperature_2m_max")
        precip   = weather.get("precipitation_sum")
        code     = weather.get("weathercode")
        wx_desc, wx_emoji, _ = describe_weather_code(code)
        m1, m2 = st.columns(2)
        with m1: st.metric("🌡️ Max Temp", f"{temp_max:.0f}°C" if temp_max else "N/A")
        with m2: st.metric("🌧️ Rain", f"{precip:.1f}mm" if precip is not None else "N/A")
        st.caption(f"{wx_emoji} {wx_desc}")

    if model.train_r2 is not None:
        st.caption(f"📊 Model accuracy (cross-val R²): {model.train_r2*100:.1f}%")

with col_reasons:
    st.markdown("### 🔍 Why this forecast?")
    st.caption("The factors below were considered by the model when generating this prediction.")
    for icon, heading, detail in reasons:
        st.markdown(f"""
        <div class="reason-card">
            <div><span style="font-size:1.2rem">{icon}</span>
            <span class="reason-heading"> {heading}</span></div>
            <div class="reason-detail">{detail}</div>
        </div>""", unsafe_allow_html=True)

    with st.expander("🧪 Top model features (global importance)"):
        top_feats = list(model.feature_importances.items())[:10]
        fi_df = pd.DataFrame(top_feats, columns=["Feature","Importance"])
        fig_fi = px.bar(fi_df.sort_values("Importance"), x="Importance", y="Feature",
                        orientation="h", color="Importance", color_continuous_scale=["#e8f4fd","#3a86ff"])
        fig_fi.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)

st.divider()

if show_week:
    st.markdown("### 📆 Week at a Glance")
    week_start  = target_date - timedelta(days=target_date.weekday())
    week_dates  = [week_start + timedelta(days=i) for i in range(7)]
    week_results = []
    for wd in week_dates:
        try:
            r = model.predict(wd)
            week_results.append({"date":wd,"day":wd.strftime("%A"),"label":r["label"]["text"],
                                  "colour":r["label"]["colour"],"score":r["score"],"emoji":r["label"]["emoji"]})
        except Exception:
            week_results.append({"date":wd,"day":wd.strftime("%A"),"label":"?","colour":"#ccc","score":0,"emoji":"❓"})

    wdf = pd.DataFrame(week_results)
    fig_week = go.Figure()
    for _, row in wdf.iterrows():
        is_sel = row["date"] == target_date
        fig_week.add_trace(go.Bar(
            x=[row["day"]], y=[row["score"]], marker_color=row["colour"],
            marker_line_color="#1a1a2e" if is_sel else "rgba(0,0,0,0)",
            marker_line_width=3 if is_sel else 0,
            text=[f"{row['score']:.0f}"], textposition="outside",
            textfont=dict(size=13,color="#1a1a2e"), name=row["day"], showlegend=False,
            hovertemplate=f"<b>{row['day']}</b><br>{row['label']}<br>Score: {row['score']:.0f}/100<extra></extra>",
        ))
    fig_week.update_layout(
        height=300, bargap=0.3, margin=dict(l=0,r=0,t=20,b=0),
        yaxis=dict(range=[0,110],title="Busyness Score",showgrid=True,gridcolor="#f0f0f0"),
        xaxis=dict(title=""), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    for y_val, lbl, col in [(65,"Busy threshold","#e67e22"),(25,"Quiet threshold","#2ecc71")]:
        fig_week.add_hline(y=y_val, line_dash="dot", line_color=col, line_width=1.5,
                           annotation_text=lbl, annotation_position="right",
                           annotation_font_size=11, annotation_font_color=col)
    st.plotly_chart(fig_week, use_container_width=True)

    cols = st.columns(7)
    for i, row in wdf.iterrows():
        with cols[i]:
            is_sel = row["date"] == target_date
            st.markdown(f"""
            <div style="background:{'#f8f9fc' if is_sel else 'white'};
                border:{'3px solid '+row['colour'] if is_sel else '1px solid #eee'};
                border-radius:10px; padding:0.6rem 0.3rem; text-align:center; margin-bottom:0.3rem">
                <div style="font-size:0.7rem;color:#888">{row['date'].strftime('%a')}</div>
                <div style="font-size:0.8rem;font-weight:600;color:#1a1a2e">{row['date'].strftime('%-d %b')}</div>
                <div style="font-size:1.4rem">{row['emoji']}</div>
                <div style="font-size:0.75rem;font-weight:600;color:{row['colour']}">{row['label']}</div>
                <div style="font-size:1.1rem;font-weight:700;color:#1a1a2e">{row['score']:.0f}</div>
            </div>""", unsafe_allow_html=True)

st.divider()

st.markdown("### 🗓️ Monthly Overview")
st.caption(f"Predicted busyness for every day in {selected_month_name} {selected_year}.")

@st.cache_data(ttl=3600, show_spinner="Building monthly forecast...")
def get_monthly_forecasts(year, month):
    _, n_days = cal_module.monthrange(year, month)
    results = []
    for d in [date(year, month, day) for day in range(1, n_days+1)]:
        try:
            r = model.predict(d)
            results.append({"date":d,"score":r["score"],"label":r["label"]["text"],"colour":r["label"]["colour"]})
        except Exception:
            results.append({"date":d,"score":0,"label":"?","colour":"#ccc"})
    return results

month_data  = get_monthly_forecasts(selected_year, selected_month)
mdf         = pd.DataFrame(month_data)
month_dates = [date(selected_year, selected_month, d) for d in range(1, cal_module.monthrange(selected_year, selected_month)[1]+1)]
first_dow   = month_dates[0].weekday()
day_labels  = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
grid = [None]*first_dow + month_dates
while len(grid) % 7 != 0: grid.append(None)
weeks = [grid[i:i+7] for i in range(0, len(grid), 7)]

hdr_cols = st.columns(7)
for i, dl in enumerate(day_labels):
    with hdr_cols[i]:
        st.markdown(f"<div style='text-align:center;font-weight:600;color:#888;font-size:0.8rem'>{dl}</div>", unsafe_allow_html=True)

for week in weeks:
    wk_cols = st.columns(7)
    for i, d in enumerate(week):
        with wk_cols[i]:
            if d is None:
                st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)
            else:
                row = mdf[mdf["date"] == d]
                if not row.empty:
                    s, c = row.iloc[0]["score"], row.iloc[0]["colour"]
                    is_sel   = d == target_date
                    is_today = d == date.today()
                    border   = "2px solid #1a1a2e" if is_sel else ("2px solid #3a86ff" if is_today else "1px solid #eee")
                    st.markdown(f"""
                    <div style="background:{c}22;border:{border};border-radius:8px;
                        padding:0.4rem 0.2rem;text-align:center;margin-bottom:2px;min-height:58px">
                        <div style="font-size:0.75rem;font-weight:{'700' if is_sel else '400'};color:#1a1a2e">{d.day}</div>
                        <div style="font-size:1.0rem;font-weight:700;color:{c}">{s:.0f}</div>
                    </div>""", unsafe_allow_html=True)

st.divider()

# ── Diagnostic Section ────────────────────────────────────────────────────
st.markdown("### 🔬 Model Training Diagnostics")
st.caption("A breakdown of the data sources used to train the forecast model.")

ds = model.data_stats
dow_names_diag = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

if ds.get("source") == "synthetic":
    st.warning("Model is running on synthetic training data — live API was unavailable at startup.", icon="⚠️")
else:
    # ── Overview metrics row ──────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("📅 Training Days", f"{ds.get('total_days', 0):,}")
        st.caption(f"{ds.get('date_range_start','')[:10]}  →  {ds.get('date_range_end','')[:10]}")
    with c2:
        st.metric("📊 Model Accuracy (R²)", f"{(model.train_r2 or 0)*100:.1f}%")
        st.caption("Cross-validated on held-out data")
    with c3:
        busiest = ds.get("busiest_dow", 5)
        quietest = ds.get("quietest_dow", 1)
        st.metric("🔴 Busiest Day", dow_names_diag[busiest])
        st.caption(f"Avg score: {ds.get('avg_by_dow',{}).get(busiest,0):.0f}/100")
    with c4:
        st.metric("🟢 Quietest Day", dow_names_diag[quietest])
        st.caption(f"Avg score: {ds.get('avg_by_dow',{}).get(quietest,0):.0f}/100")

    st.markdown("---")

    # ── Per-source breakdown ──────────────────────────────────────────────
    st.markdown("#### 📡 Data Sources Used in Training")
    src_c1, src_c2, src_c3, src_c4 = st.columns(4)

    with src_c1:
        st.markdown("**🚶 Infrared Counters**")
        days = ds.get("infrared_days", 0)
        st.metric("Observation Records", f"{days:,}")
        if days > 0:
            st.caption(f"📅 {ds.get('infrared_start','')[:10]} → {ds.get('infrared_end','')[:10]}")
            avg = ds.get("infrared_avg", 0)
            st.caption(f"Avg daily count: {avg:,.0f} people")
            st.success("✅ Active" if days > 0 else "")
        else:
            st.warning("No data returned")
        st.caption("Source: Ballarat Open Data infrared-counters dataset. Sensors at Lake Wendouree, Loreto Point and CBD locations.")

    with src_c2:
        st.markdown("**👥 People Counts**")
        days = ds.get("people_days", 0)
        st.metric("Observation Records", f"{days:,}")
        if days > 0:
            st.caption(f"📅 {ds.get('people_start','')[:10]} → {ds.get('people_end','')[:10]}")
            st.success("✅ Active")
        else:
            st.warning("No data returned")
        st.caption("Source: Ballarat Open Data people-counts dataset. 15-minute interval counts aggregated to daily totals.")

    with src_c3:
        st.markdown("**🅿️ Parking Transactions**")
        raw  = ds.get("parking_days_raw", 0)
        used = ds.get("parking_days_used", 0)
        excl = ds.get("parking_days_excluded", 0)
        st.metric("Records Used in Training", f"{used:,}")
        if raw > 0:
            st.caption(f"📅 {ds.get('parking_start','')[:10]} → {ds.get('parking_end','')[:10]}")
            st.caption(f"Raw records: {raw:,}")
            st.caption(f"Excluded (free parking days): {excl:,}")
            avg_p = ds.get("parking_avg", 0)
            st.caption(f"Avg daily transactions: {avg_p:,.0f}")
            st.success("✅ Active")
        else:
            st.warning("No data returned")
        st.caption("Source: Ballarat Open Data parking-transactions dataset. Sundays & public holidays excluded (free parking = zero transactions).")

    with src_c4:
        st.markdown("**🌦️ Weather (Open-Meteo)**")
        days = ds.get("weather_days", 0)
        st.metric("Days of Weather Data", f"{days:,}")
        if days > 0:
            st.caption(f"📅 {ds.get('weather_start','')[:10]} → {ds.get('weather_end','')[:10]}")
            st.success("✅ Active")
        else:
            st.warning("No weather data")
        st.caption("Source: Open-Meteo archive API. Variables: max temp, min temp, rainfall, wind speed, weather code, sunshine hours.")

    st.markdown("---")

    # ── Day-of-week average busyness bar chart ────────────────────────────
    st.markdown("#### 📈 Historical Average Busyness by Day of Week")
    st.caption("Observed average busyness index (0–100) per day of week across all training data.")

    avg_by_dow = ds.get("avg_by_dow", {})
    if avg_by_dow:
        dow_df = pd.DataFrame([
            {"Day": dow_names_diag[int(k)], "Avg Busyness": round(v, 1), "dow_order": int(k)}
            for k, v in avg_by_dow.items()
        ]).sort_values("dow_order")

        colours = []
        for _, r in dow_df.iterrows():
            if r["Avg Busyness"] >= 65:   colours.append("#e67e22")
            elif r["Avg Busyness"] >= 45: colours.append("#f1c40f")
            else:                          colours.append("#2ecc71")

        fig_dow = go.Figure(go.Bar(
            x=dow_df["Day"],
            y=dow_df["Avg Busyness"],
            marker_color=colours,
            text=[f"{v:.0f}" for v in dow_df["Avg Busyness"]],
            textposition="outside",
            textfont=dict(size=12),
            hovertemplate="<b>%{x}</b><br>Avg: %{y:.1f}/100<extra></extra>",
        ))
        fig_dow.update_layout(
            height=300,
            yaxis=dict(range=[0, 110], title="Avg Busyness Score", showgrid=True, gridcolor="#333"),
            xaxis=dict(title=""),
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            bargap=0.3,
        )
        fig_dow.add_hline(y=ds.get("avg_overall", 50), line_dash="dot", line_color="#3a86ff",
                          line_width=2, annotation_text=f"Overall avg: {ds.get('avg_overall',50):.0f}",
                          annotation_position="right", annotation_font_color="#3a86ff", annotation_font_size=11)
        st.plotly_chart(fig_dow, use_container_width=True)

    # ── How model uses the data — explainer ───────────────────────────────
    with st.expander("ℹ️ How does the model use this data?"):
        st.markdown(f"""
**The model does not simply average each day of the week.** Instead it uses XGBoost — a gradient
boosted decision tree algorithm — that learns complex interactions between all signals simultaneously.

**Training process:**
1. Each historical day becomes a row of ~27 features (day of week, month, weather, events, holidays, etc.)
2. The busyness index for that day (derived from the sensor data above) is the target value
3. XGBoost builds {400} decision trees, each correcting the errors of the previous
4. Trees vote together to produce a final prediction score

**Why `is_free_parking_day` dominates the feature importance chart:**
The parking dataset has zero transactions on Sundays and public holidays. This creates a very strong
structural signal — the model quickly learns that days with no parking data behave differently from
weekdays. This is a data artifact rather than a meaningful signal, which is why we added the
`is_free_parking_day` feature to explicitly flag it.

**Data quality note:**
With {ds.get('total_days', 0):,} training days and a cross-validated R² of {(model.train_r2 or 0)*100:.1f}%,
the model explains roughly {(model.train_r2 or 0)*100:.0f}% of the variance in observed busyness.
The remaining variance is due to factors not captured in the data (one-off events, road closures, etc.).
        """)

st.divider()
st.markdown("""
<div style="text-align:center;color:#aaa;font-size:0.8rem;padding:1rem 0">
    Ballarat Busyness Forecaster &nbsp;|&nbsp;
    Data: <a href="https://data.ballarat.vic.gov.au" style="color:#3a86ff">Ballarat Open Data</a>,
    <a href="https://open-meteo.com" style="color:#3a86ff">Open-Meteo</a> &nbsp;|&nbsp;
    Built with Streamlit + XGBoost &nbsp;|&nbsp;
    Made by <strong style="color:#ffffff">Justin Molik</strong>
</div>
""", unsafe_allow_html=True)
