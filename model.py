"""
model.py
────────────────────────────────────────────────────────────────────────────
Busyness forecast model for Ballarat.

All imports from data_fetcher are done INSIDE functions (lazy imports) to
prevent circular import errors when Streamlit loads app.py -> model.py ->
data_fetcher.py simultaneously.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# WMO WEATHER CODE LOOKUP (self-contained — no cross-module import needed)
# ─────────────────────────────────────────────────────────────────────────────
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

def _describe_weather(code) -> tuple:
    try:
        code = int(code)
    except (TypeError, ValueError):
        return ("Unknown conditions", "❓", 0.70)
    return WMO_CODES.get(code, ("Unknown conditions", "❓", 0.70))


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE COLUMNS
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "day_of_week", "is_weekend", "month", "day_of_month", "week_of_year",
    "is_summer", "is_autumn", "is_winter", "is_spring",
    "is_public_holiday", "is_eve_of_holiday", "is_long_weekend", "is_school_holiday",
    "has_event", "event_impact", "num_events",
    "temp_max", "temp_min", "precipitation", "wind_max",
    "weather_code", "sunshine_hours",
    "is_rainy", "is_hot", "is_cold", "is_pleasant",
    "is_free_parking_day",
]


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def build_features_for_date(d: date, weather: dict = None) -> dict:
    """Build the full feature dict for a single date."""
    from data_fetcher import (
        is_public_holiday, is_eve_of_holiday, is_long_weekend,
        is_school_holiday, get_events_for_date, get_event_impact_for_date,
        get_weather_for_date,
    )

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
        "is_summer":           int(d.month in [12, 1, 2]),
        "is_autumn":           int(d.month in [3, 4, 5]),
        "is_winter":           int(d.month in [6, 7, 8]),
        "is_spring":           int(d.month in [9, 10, 11]),
        "is_public_holiday":   int(is_ph),
        "is_eve_of_holiday":   int(is_eve_of_holiday(d)),
        "is_long_weekend":     int(is_long_weekend(d)),
        "is_school_holiday":   int(is_school_holiday(d)),
        "has_event":           int(len(events) > 0),
        "event_impact":        get_event_impact_for_date(d),
        "num_events":          len(events),
        "temp_max":            float(weather.get("temperature_2m_max", 15.0) or 15.0),
        "temp_min":            float(weather.get("temperature_2m_min", 8.0)  or 8.0),
        "precipitation":       float(weather.get("precipitation_sum", 0.0)   or 0.0),
        "wind_max":            float(weather.get("windspeed_10m_max", 10.0)  or 10.0),
        "weather_code":        int(weather.get("weathercode", 2)             or 2),
        "sunshine_hours":      float(weather.get("sunshine_duration", 28800) or 28800) / 3600,
        "is_rainy":            int(float(weather.get("precipitation_sum", 0) or 0) > 2.0),
        "is_hot":              int(float(weather.get("temperature_2m_max", 15) or 15) > 28.0),
        "is_cold":             int(float(weather.get("temperature_2m_max", 15) or 15) < 10.0),
        "is_pleasant":         int(
                                   15 <= float(weather.get("temperature_2m_max", 15) or 15) <= 26
                                   and float(weather.get("precipitation_sum", 0) or 0) < 1.0
                               ),
        "is_free_parking_day": int(d.weekday() == 6 or is_ph),
    }


def build_training_dataframe(traffic: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in traffic.iterrows():
        d = row["date"]
        if isinstance(d, str):
            d = date.fromisoformat(d)
        w_row   = weather_df[weather_df["date"] == d]
        weather = w_row.iloc[0].to_dict() if not w_row.empty else {}
        feats   = build_features_for_date(d, weather)
        feats["busyness_index"] = row["busyness_index"]
        feats["date"] = d
        rows.append(feats)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC TRAINING DATA FALLBACK
# ─────────────────────────────────────────────────────────────────────────────
def generate_synthetic_traffic(years: int = 3) -> pd.DataFrame:
    """Realistic synthetic daily busyness — used when live API is unavailable."""
    from data_fetcher import is_public_holiday, get_event_impact_for_date, is_school_holiday

    today = date.today()
    start = date(today.year - years, 1, 1)
    rows  = []
    rng   = np.random.default_rng(42)
    d     = start

    while d < today:
        base     = 50.0
        dow      = d.weekday()
        base    += [0, 5, 5, 8, 15, 30, 18][dow]
        is_ph, _ = is_public_holiday(d)
        if dow == 6 or is_ph:
            base += 10
        monthly  = {1: -5, 2: -3, 3: 10, 4: 5, 5: 0, 6: -8,
                    7: -10, 8: 5, 9: 8, 10: 10, 11: 5, 12: 15}
        base    += monthly.get(d.month, 0)
        base    += get_event_impact_for_date(d) * 12
        if is_ph:
            base += 20
        if is_school_holiday(d):
            base += 8
        rows.append({"date": d, "busyness_index": float(np.clip(base + rng.normal(0, 8), 0, 100))})
        d += timedelta(days=1)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CLASS
# ─────────────────────────────────────────────────────────────────────────────
class BusynessModel:
    def __init__(self):
        self.model               = None
        self.is_trained          = False
        self.feature_importances = {}
        self.train_r2            = None
        self.data_source         = "unknown"

    def train(self, use_live_data: bool = True):
        """Fetch data, engineer features, train XGBoost."""
        from data_fetcher import combine_traffic_data, fetch_historical_weather
        try:
            from xgboost import XGBRegressor
            from sklearn.model_selection import cross_val_score
        except ImportError as e:
            raise ImportError(f"Missing dependency: {e}. Run: pip install xgboost scikit-learn")

        logger.info("Fetching traffic data...")
        traffic = combine_traffic_data() if use_live_data else pd.DataFrame()

        if len(traffic) < 30:
            logger.warning(f"Insufficient live data ({len(traffic)} rows). Using synthetic data.")
            traffic          = generate_synthetic_traffic(years=3)
            self.data_source = "synthetic"
        else:
            self.data_source = "live"

        logger.info(f"Training on {len(traffic)} days (source: {self.data_source})")

        min_date = min(traffic["date"])
        max_date = max(traffic["date"])
        if isinstance(min_date, str):
            min_date = date.fromisoformat(min_date)
            max_date = date.fromisoformat(max_date)

        logger.info("Fetching historical weather for training period...")
        weather_df = fetch_historical_weather(min_date, max_date)

        df = build_training_dataframe(traffic, weather_df)
        df = df.dropna(subset=FEATURE_COLS + ["busyness_index"])

        X = df[FEATURE_COLS].astype(float)
        y = df["busyness_index"].astype(float)

        self.model = XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        )
        self.model.fit(X, y)

        cv_scores     = cross_val_score(self.model, X, y, cv=min(5, len(X) // 20 or 2), scoring="r2")
        self.train_r2 = float(np.mean(cv_scores))
        logger.info(f"Model trained. Cross-val R² = {self.train_r2:.3f}")

        self.feature_importances = dict(
            sorted(zip(FEATURE_COLS, self.model.feature_importances_), key=lambda x: -x[1])
        )
        self.is_trained = True

    def predict(self, d: date, weather: dict = None) -> dict:
        """Predict busyness for a given date."""
        from data_fetcher import get_weather_for_date, get_events_for_date

        if not self.is_trained:
            raise RuntimeError("Model not trained. Call .train() first.")

        if weather is None:
            weather = get_weather_for_date(d)

        feats     = build_features_for_date(d, weather)
        X         = pd.DataFrame([feats])[FEATURE_COLS].astype(float)
        raw_score = float(self.model.predict(X)[0])
        score     = float(np.clip(raw_score, 0, 100))

        return {
            "date":    d,
            "score":   score,
            "label":   score_to_label(score),
            "feats":   feats,
            "weather": weather,
            "events":  get_events_for_date(d),
            "reasons": build_reason_summary(d, feats, weather, score),
        }


# ─────────────────────────────────────────────────────────────────────────────
# SCORING LABELS
# ─────────────────────────────────────────────────────────────────────────────
def score_to_label(score: float) -> dict:
    if score >= 80:
        return {"text": "Very Busy",  "colour": "#e74c3c", "emoji": "🔴", "tier": 5}
    elif score >= 65:
        return {"text": "Busy",       "colour": "#e67e22", "emoji": "🟠", "tier": 4}
    elif score >= 45:
        return {"text": "Moderate",   "colour": "#f1c40f", "emoji": "🟡", "tier": 3}
    elif score >= 25:
        return {"text": "Quiet",      "colour": "#2ecc71", "emoji": "🟢", "tier": 2}
    else:
        return {"text": "Very Quiet", "colour": "#27ae60", "emoji": "🟢", "tier": 1}


# ─────────────────────────────────────────────────────────────────────────────
# REASON SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def build_reason_summary(d: date, feats: dict, weather: dict, score: float) -> list:
    """Return list of (icon, heading, detail) tuples explaining the forecast."""
    from data_fetcher import is_public_holiday, is_eve_of_holiday, get_events_for_date

    reasons   = []
    dow_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow       = dow_names[d.weekday()]

    if d.weekday() == 5:
        reasons.append(("📅", "Saturday", "Saturdays are consistently the busiest day in Ballarat's CBD."))
    elif d.weekday() == 6:
        reasons.append(("📅", "Sunday", "Sundays bring moderate foot traffic — free parking encourages family visits and leisure shopping."))
        reasons.append(("🅿️", "Free Parking Day", "Parking is free on Sundays in Ballarat. The model accounts for zero parking transactions and does not treat this as low activity."))
    elif d.weekday() == 4:
        reasons.append(("📅", "Friday", "Fridays are the second-busiest weekday, with after-work and evening activity."))
    else:
        reasons.append(("📅", dow, f"{dow}s are typically quieter weekdays in central Ballarat."))

    is_ph, ph_name = is_public_holiday(d)
    if is_ph:
        reasons.append(("🎉", f"Public Holiday: {ph_name}", "Public holidays significantly boost CBD activity — markets, events and day-trippers."))
        reasons.append(("🅿️", "Free Parking Day", "Parking is free on public holidays. The model accounts for the missing parking data."))
    elif is_eve_of_holiday(d):
        reasons.append(("🌙", "Eve of Public Holiday", "The day before a public holiday often sees elevated shopping and hospitality activity."))

    if feats.get("is_long_weekend") and not is_ph:
        reasons.append(("📆", "Long Weekend", "Long weekends draw visitors from Melbourne and surrounds to Ballarat."))

    if feats.get("is_school_holiday"):
        reasons.append(("🎒", "School Holidays", "School holiday periods increase family visits to Sovereign Hill and the Wildlife Park."))

    for ev in get_events_for_date(d):
        impact_desc = {1: "minor", 2: "moderate", 3: "major"}.get(ev["impact"], "notable")
        reasons.append(("🎪", ev["name"], f"This {impact_desc}-impact local event increases visitor numbers in the Ballarat area."))

    code   = int(weather.get("weathercode", 2) or 2)
    desc, emoji, mult = _describe_weather(code)
    temp   = float(weather.get("temperature_2m_max", 15) or 15)
    precip = float(weather.get("precipitation_sum", 0)   or 0)

    if mult >= 0.9:
        wx_detail = f"Forecast: {desc}, {temp:.0f}°C — excellent conditions will encourage people into town."
    elif mult >= 0.6:
        wx_detail = f"Forecast: {desc}, {temp:.0f}°C — reasonable conditions; mild impact on foot traffic."
    else:
        wx_detail = f"Forecast: {desc}, {temp:.0f}°C with {precip:.1f}mm rain — poor conditions will reduce outdoor movement."

    reasons.append((emoji, f"Weather: {desc}", wx_detail))

    if feats.get("is_hot"):
        reasons.append(("🌡️", "Hot Day (>28°C)", "Very hot days can reduce CBD foot traffic as people seek cooler environments."))
    if feats.get("is_cold"):
        reasons.append(("🧊", "Cold Day (<10°C max)", "Cold days tend to keep people at home and reduce casual shopping trips."))

    for key, (icon, label, detail) in {
        "is_summer": ("☀️", "Summer Season", "December–February: school holidays and warm weather boost tourism."),
        "is_winter": ("❄️", "Winter Season",  "June–August: Ballarat winters are cold — the quietest period for CBD activity."),
        "is_spring": ("🌸", "Spring Season",  "September–November: warming weather and events build momentum."),
        "is_autumn": ("🍂", "Autumn Season",  "March–May: mild weather; post-summer visitor numbers remain solid."),
    }.items():
        if feats.get(key):
            reasons.append((icon, label, detail))
            break

    return reasons
