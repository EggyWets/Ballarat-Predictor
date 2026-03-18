"""
model.py
────────────────────────────────────────────────────────────────────────────
Builds and runs the busyness forecast model for Ballarat.

Model: XGBoost Regressor
Target: normalised busyness index (0–100) derived from combined traffic data
Features: temporal, weather, calendar (holidays, events, school holidays)

The model is trained once on historical data then cached in-memory.
Call `predict_day()` to get a forecast for any date.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
import logging

from data_fetcher import (
    combine_traffic_data,
    fetch_historical_weather,
    get_vic_holidays,
    get_events_for_date,
    get_event_impact_for_date,
    is_public_holiday,
    is_eve_of_holiday,
    is_long_weekend,
    is_school_holiday,
    get_weather_for_date,
    describe_weather_code,
    LOCAL_EVENTS,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def build_features_for_date(d: date, weather: dict = None) -> dict:
    """
    Build the complete feature dictionary for a single date.
    Weather dict should contain Open-Meteo daily fields.
    If weather is None, weather features will be set to sensible defaults.
    """
    if weather is None:
        weather = get_weather_for_date(d)

    is_ph, ph_name = is_public_holiday(d)
    events = get_events_for_date(d)

    feats = {
        # ── Temporal ──────────────────────────────────────────────────────
        "day_of_week":       d.weekday(),           # 0=Mon … 6=Sun
        "is_weekend":        int(d.weekday() >= 5),
        "month":             d.month,
        "day_of_month":      d.day,
        "week_of_year":      d.isocalendar()[1],
        "is_summer":         int(d.month in [12, 1, 2]),
        "is_autumn":         int(d.month in [3, 4, 5]),
        "is_winter":         int(d.month in [6, 7, 8]),
        "is_spring":         int(d.month in [9, 10, 11]),

        # ── Calendar ──────────────────────────────────────────────────────
        "is_public_holiday":  int(is_ph),
        "is_eve_of_holiday":  int(is_eve_of_holiday(d)),
        "is_long_weekend":    int(is_long_weekend(d)),
        "is_school_holiday":  int(is_school_holiday(d)),

        # ── Events ────────────────────────────────────────────────────────
        "has_event":          int(len(events) > 0),
        "event_impact":       get_event_impact_for_date(d),
        "num_events":         len(events),

        # ── Weather ───────────────────────────────────────────────────────
        "temp_max":           float(weather.get("temperature_2m_max", 15.0) or 15.0),
        "temp_min":           float(weather.get("temperature_2m_min", 8.0)  or 8.0),
        "precipitation":      float(weather.get("precipitation_sum", 0.0)   or 0.0),
        "wind_max":           float(weather.get("windspeed_10m_max", 10.0)  or 10.0),
        "weather_code":       int(weather.get("weathercode", 2)             or 2),
        "sunshine_hours":     float(weather.get("sunshine_duration", 28800) or 28800) / 3600,

        # ── Derived weather ───────────────────────────────────────────────
        "is_rainy":           int(float(weather.get("precipitation_sum", 0) or 0) > 2.0),
        "is_hot":             int(float(weather.get("temperature_2m_max", 15) or 15) > 28.0),
        "is_cold":            int(float(weather.get("temperature_2m_max", 15) or 15) < 10.0),
        "is_pleasant":        int(
                                  15 <= float(weather.get("temperature_2m_max", 15) or 15) <= 26 and
                                  float(weather.get("precipitation_sum", 0) or 0) < 1.0
                              ),

        # ── Parking bias correction ───────────────────────────────────────
        # Ballarat has free parking on Sundays and public holidays, meaning
        # parking transaction counts drop to zero on these days — not because
        # the town is quiet, but because the meters aren't running.
        # This flag tells the model to discount the missing parking signal.
        "is_free_parking_day": int(d.weekday() == 6 or is_ph),
    }
    return feats


def build_training_dataframe(traffic: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join traffic data with weather and calendar features.
    """
    rows = []
    for _, row in traffic.iterrows():
        d = row["date"]
        if isinstance(d, str):
            d = date.fromisoformat(d)

        w_row = weather_df[weather_df["date"] == d]
        weather = w_row.iloc[0].to_dict() if not w_row.empty else {}

        feats = build_features_for_date(d, weather)
        feats["busyness_index"] = row["busyness_index"]
        feats["date"] = d
        rows.append(feats)

    return pd.DataFrame(rows)


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
# SYNTHETIC DATA FALLBACK
# If the Ballarat Open Data API has no data (or insufficient history),
# we generate realistic synthetic data so the app always works.
# ─────────────────────────────────────────────────────────────────────────────
def generate_synthetic_traffic(years: int = 3) -> pd.DataFrame:
    """
    Produce synthetic but realistic daily busyness data for Ballarat.
    Used as training fallback if the live API returns insufficient data.
    """
    today = date.today()
    start = date(today.year - years, 1, 1)
    rows  = []
    rng   = np.random.default_rng(42)

    d = start
    while d < today:
        base = 50.0
        # Weekday patterns — based on real foot traffic not parking transactions
        # Sunday is genuinely moderately busy (free parking day, families out)
        dow = d.weekday()
        day_boost = [0, 5, 5, 8, 15, 30, 18][dow]  # Mon–Sun
        base += day_boost

        # Free parking day (Sunday or public holiday) — parking data is zero
        # but actual busyness is NOT zero; add a correction boost
        is_ph, _ = is_public_holiday(d)
        if dow == 6 or is_ph:
            base += 10  # correct for missing parking signal

        # Monthly seasonality (Ballarat)
        monthly = {1: -5, 2: -3, 3: 10, 4: 5, 5: 0, 6: -8,
                   7: -10, 8: 5, 9: 8, 10: 10, 11: 5, 12: 15}
        base += monthly.get(d.month, 0)

        # Events
        event_impact = get_event_impact_for_date(d)
        base += event_impact * 12

        # Public holidays
        if is_ph:
            base += 20

        # School holidays
        if is_school_holiday(d):
            base += 8

        # Add noise
        noise = rng.normal(0, 8)
        busyness = float(np.clip(base + noise, 0, 100))
        rows.append({"date": d, "busyness_index": busyness})
        d += timedelta(days=1)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CLASS
# ─────────────────────────────────────────────────────────────────────────────
class BusynessModel:
    def __init__(self):
        self.model       = None
        self.is_trained  = False
        self.feature_importances = {}
        self.train_r2    = None
        self.data_source = "unknown"

    def train(self, use_live_data: bool = True):
        """
        Fetch data, engineer features, train XGBoost model.
        Falls back to synthetic data if live API returns <30 rows.
        """
        try:
            from xgboost import XGBRegressor
            from sklearn.model_selection import cross_val_score
            from sklearn.preprocessing import StandardScaler
        except ImportError as e:
            raise ImportError(f"Missing dependency: {e}. Run: pip install xgboost scikit-learn")

        logger.info("Fetching traffic data...")
        traffic = combine_traffic_data() if use_live_data else pd.DataFrame()

        if len(traffic) < 30:
            logger.warning(f"Insufficient live data ({len(traffic)} rows). Using synthetic data.")
            traffic = generate_synthetic_traffic(years=3)
            self.data_source = "synthetic"
        else:
            self.data_source = "live"

        logger.info(f"Training on {len(traffic)} days of data (source: {self.data_source})")

        # Fetch bulk weather for the training period
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

        # Cross-validated R²
        cv_scores = cross_val_score(self.model, X, y, cv=min(5, len(X)//20 or 2), scoring="r2")
        self.train_r2 = float(np.mean(cv_scores))
        logger.info(f"Model trained. Cross-val R² = {self.train_r2:.3f}")

        # Feature importances
        self.feature_importances = dict(
            sorted(
                zip(FEATURE_COLS, self.model.feature_importances_),
                key=lambda x: -x[1]
            )
        )
        self.is_trained = True

    def predict(self, d: date, weather: dict = None) -> dict:
        """
        Predict busyness for a date. Returns a rich result dict.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call .train() first.")

        if weather is None:
            weather = get_weather_for_date(d)

        feats = build_features_for_date(d, weather)
        X = pd.DataFrame([feats])[FEATURE_COLS].astype(float)
        raw_score = float(self.model.predict(X)[0])
        score = float(np.clip(raw_score, 0, 100))

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
        return {"text": "Very Busy",    "colour": "#e74c3c", "emoji": "🔴", "tier": 5}
    elif score >= 65:
        return {"text": "Busy",         "colour": "#e67e22", "emoji": "🟠", "tier": 4}
    elif score >= 45:
        return {"text": "Moderate",     "colour": "#f1c40f", "emoji": "🟡", "tier": 3}
    elif score >= 25:
        return {"text": "Quiet",        "colour": "#2ecc71", "emoji": "🟢", "tier": 2}
    else:
        return {"text": "Very Quiet",   "colour": "#27ae60", "emoji": "🟢", "tier": 1}


# ─────────────────────────────────────────────────────────────────────────────
# REASON SUMMARY — plain-English explanation of the prediction
# ─────────────────────────────────────────────────────────────────────────────
def build_reason_summary(d: date, feats: dict, weather: dict, score: float) -> list:
    """
    Return a list of (icon, heading, detail) tuples explaining the forecast.
    """
    reasons = []
    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = dow_names[d.weekday()]

    # Day of week
    if d.weekday() == 5:
        reasons.append(("📅", "Saturday", "Saturdays are consistently the busiest day in Ballarat's CBD."))
    elif d.weekday() == 6:
        reasons.append(("📅", "Sunday", "Sundays bring moderate foot traffic — free parking encourages family visits and leisure shopping."))
        reasons.append(("🅿️", "Free Parking Day", "Parking is free on Sundays in Ballarat, so parking transaction data is not available. The model accounts for this and does not treat the missing data as a sign of low activity."))
    elif d.weekday() == 4:
        reasons.append(("📅", "Friday", "Fridays are the second-busiest weekday, with after-work and evening activity."))
    else:
        reasons.append(("📅", dow, f"{dow}s are typically quieter weekdays in central Ballarat."))

    # Public holiday
    is_ph, ph_name = is_public_holiday(d)
    if is_ph:
        reasons.append(("🎉", f"Public Holiday: {ph_name}", "Public holidays significantly boost CBD activity — markets, events and day-trippers."))
        reasons.append(("🅿️", "Free Parking Day", "Parking is free on public holidays in Ballarat. The model accounts for the missing parking data and does not treat it as low activity."))
    elif is_eve_of_holiday(d):
        reasons.append(("🌙", "Eve of Public Holiday", "The day before a public holiday often sees elevated shopping and hospitality activity."))

    # Long weekend
    if feats.get("is_long_weekend") and not is_ph:
        reasons.append(("📆", "Long Weekend", "Long weekends draw visitors from Melbourne and surrounds to Ballarat."))

    # School holidays
    if feats.get("is_school_holiday"):
        reasons.append(("🎒", "School Holidays", "School holiday periods increase family visits to attractions like Sovereign Hill and the Wildlife Park."))

    # Events
    events = get_events_for_date(d)
    for ev in events:
        impact_desc = {1: "minor", 2: "moderate", 3: "major"}.get(ev["impact"], "notable")
        reasons.append(("🎪", ev["name"], f"This {impact_desc}-impact local event increases visitor numbers in the Ballarat area."))

    # Weather
    code = int(weather.get("weathercode", 2) or 2)
    desc, emoji, mult = describe_weather_code(code)
    temp  = float(weather.get("temperature_2m_max", 15) or 15)
    precip = float(weather.get("precipitation_sum", 0) or 0)

    if mult >= 0.9:
        wx_detail = f"Forecast: {desc}, {temp:.0f}°C — excellent outdoor conditions will encourage people into town."
    elif mult >= 0.6:
        wx_detail = f"Forecast: {desc}, {temp:.0f}°C — reasonable conditions; mild impact on foot traffic."
    else:
        wx_detail = f"Forecast: {desc}, {temp:.0f}°C with {precip:.1f}mm rain — poor conditions will reduce outdoor movement."

    reasons.append((emoji, f"Weather: {desc}", wx_detail))

    if feats.get("is_hot"):
        reasons.append(("🌡️", "Hot Day (>28°C)", "Very hot days in Ballarat can reduce CBD foot traffic as people seek cooler environments."))
    if feats.get("is_cold"):
        reasons.append(("🧊", "Cold Day (<10°C max)", "Cold days, especially in winter, tend to keep people at home and reduce casual shopping trips."))

    # Season
    season_notes = {
        "is_summer": ("☀️", "Summer Season", "December–February: school holidays and warm weather generally boost tourism."),
        "is_winter": ("❄️", "Winter Season",  "June–August: Ballarat winters are cold — generally the quietest period for CBD activity."),
        "is_spring": ("🌸", "Spring Season",  "September–November: warming weather and events like the Begonia Festival build momentum."),
        "is_autumn": ("🍂", "Autumn Season",  "March–May: mild weather; post-summer visitor numbers remain solid."),
    }
    for key, (icon, label, detail) in season_notes.items():
        if feats.get(key):
            reasons.append((icon, label, detail))
            break

    return reasons
