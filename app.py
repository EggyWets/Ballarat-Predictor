"""
app.py
────────────────────────────────────────────────────────────────────────────
Ballarat Busyness Forecaster — Streamlit Dashboard

Run with:   streamlit run app.py
Requires:   pip install -r requirements.txt
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
import sys
import os

# ── Inline weather code lookup (avoids circular import issues) ────────────
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

def describe_weather_code(code):
    try:
        code = int(code)
    except (TypeError, ValueError):
        return ("Unknown conditions", "❓", 0.70)
    return WMO_CODES.get(code, ("Unknown conditions", "❓", 0.70))

# ── Page config (must be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="Ballarat Busyness Forecaster",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Source+Sans+3:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
}

h1, h2, h3 {
    font-family: 'Playfair Display', Georgia, serif !important;
}

.main-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0;
}

.sub-title {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 1.0rem;
    color: #c0ccd8;
    margin-top: 0.2rem;
    margin-bottom: 1.5rem;
}

.score-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    color: white;
    margin-bottom: 1rem;
}

.score-number {
    font-family: 'Playfair Display', serif;
    font-size: 5rem;
    font-weight: 700;
    line-height: 1;
}

.score-label {
    font-size: 1.5rem;
    font-weight: 600;
    margin-top: 0.3rem;
}

.score-date {
    font-size: 0.9rem;
    opacity: 0.7;
    margin-top: 0.5rem;
}

.reason-card {
    background: #f8f9fc;
    border-left: 4px solid #3a86ff;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin-bottom: 0.6rem;
}

.reason-icon {
    font-size: 1.2rem;
}

.reason-heading {
    font-weight: 600;
    color: #1a1a2e;
    font-size: 0.95rem;
}

.reason-detail {
    color: #5c6b7a;
    font-size: 0.85rem;
    margin-top: 0.15rem;
}

.metric-box {
    background: white;
    border: 1px solid #e8ecf0;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

.stAlert {
    border-radius: 8px;
}

div[data-testid="stSelectbox"] > div {
    border-radius: 8px;
}

.sidebar-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    color: #ffffff;
    margin-bottom: 0.5rem;
}

.data-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
}

.badge-live   { background: #d4edda; color: #155724; }
.badge-synth  { background: #fff3cd; color: #856404; }
</style>
""", unsafe_allow_html=True)


# ── Lazy imports (so errors surface nicely) ───────────────────────────────
@st.cache_resource(show_spinner="🧠  Training forecast model — this takes ~30 seconds on first run...")
def load_model():
    from model import BusynessModel
    m = BusynessModel()
    m.train(use_live_data=True)
    return m


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-header">🏙️ Ballarat Busyness</div>', unsafe_allow_html=True)
    st.caption("Forecast how busy Ballarat town centre will be on any day.")
    st.divider()

    # Date selector
    st.markdown("### 📅 Select a Date")
    today = date.today()

    # Year selector
    year_options  = list(range(today.year - 1, today.year + 3))
    selected_year = st.selectbox("Year", year_options, index=year_options.index(today.year))

    # Month selector
    month_names = ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"]
    selected_month_name = st.selectbox("Month", month_names, index=today.month - 1)
    selected_month = month_names.index(selected_month_name) + 1

    # Day selector — dynamically correct number of days
    import calendar
    days_in_month = calendar.monthrange(selected_year, selected_month)[1]
    default_day   = min(today.day, days_in_month) if (selected_year == today.year and selected_month == today.month) else 1
    selected_day  = st.selectbox("Day", list(range(1, days_in_month + 1)), index=default_day - 1)

    try:
        target_date = date(selected_year, selected_month, selected_day)
    except ValueError:
        target_date = today

    st.divider()

    # Week view toggle
    show_week = st.checkbox("Show full week comparison", value=True)

    # About section
    with st.expander("ℹ️ About this tool"):
        st.markdown("""
        **Data sources:**
        - 🚶 Ballarat Open Data (pedestrian & parking counts)
        - 🌦️ Open-Meteo weather (free, no API key)
        - 📅 Victorian public holidays
        - 🎪 Curated Ballarat events calendar

        **Model:** XGBoost gradient boosting trained on historical
        traffic + weather + calendar features.

        **Note:** Predictions for distant future dates use forecast
        weather data (available ~16 days ahead) or seasonal averages.
        """)


# ── Main content ──────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Ballarat Busyness Forecaster</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-powered busyness predictions for Ballarat, Victoria — powered by local open data, weather & events.</div>', unsafe_allow_html=True)

# Load model
try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.info("Make sure all dependencies are installed: `pip install -r requirements.txt`")
    st.stop()

# Data source badge
badge_class = "badge-live" if model.data_source == "live" else "badge-synth"
badge_label = "Live Data" if model.data_source == "live" else "Synthetic Training Data"
badge_tip   = "Trained on real Ballarat pedestrian & parking data." if model.data_source == "live" \
              else "Ballarat Open Data API unavailable — model trained on realistic synthetic data."
st.markdown(
    f'<span class="data-badge {badge_class}">⚡ {badge_label}</span> '
    f'<span style="font-size:0.8rem; color:#888">{badge_tip}</span>',
    unsafe_allow_html=True
)

if model.data_source == "synthetic":
    st.warning("⚠️ The Ballarat Open Data API could not be reached. Predictions are based on realistic synthetic training data — patterns will be accurate but not calibrated to actual counts.", icon="⚠️")

st.divider()

# ── Run prediction ────────────────────────────────────────────────────────
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
events  = result["events"]
reasons = result["reasons"]

# ── Layout: score card + reasons ─────────────────────────────────────────
col_score, col_reasons = st.columns([1, 2])

with col_score:
    # Big score card
    st.markdown(f"""
    <div class="score-card">
        <div style="font-size:3rem">{label['emoji']}</div>
        <div class="score-number" style="color:{label['colour']}">{score:.0f}</div>
        <div style="font-size:0.85rem; opacity:0.6; margin-top:0.2rem">out of 100</div>
        <div class="score-label">{label['text']}</div>
        <div class="score-date">{target_date.strftime('%A, %-d %B %Y')}</div>
    </div>
    """, unsafe_allow_html=True)

    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 20}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": label["colour"]},
            "steps": [
                {"range": [0,  25],  "color": "#e8f5e9"},
                {"range": [25, 45],  "color": "#c8e6c9"},
                {"range": [45, 65],  "color": "#fff9c4"},
                {"range": [65, 80],  "color": "#ffe0b2"},
                {"range": [80, 100], "color": "#ffcdd2"},
            ],
            "threshold": {
                "line": {"color": "#1a1a2e", "width": 3},
                "thickness": 0.8,
                "value": score,
            },
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig_gauge.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Source Sans 3"},
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Key weather metrics
    if weather:
        temp_max = weather.get("temperature_2m_max")
        precip   = weather.get("precipitation_sum")
        code     = weather.get("weathercode")
        _, wx_emoji, _ = describe_weather_code(code)
        wx_desc, _, _  = describe_weather_code(code)

        m1, m2 = st.columns(2)
        with m1:
            st.metric("🌡️ Max Temp", f"{temp_max:.0f}°C" if temp_max else "N/A")
        with m2:
            st.metric("🌧️ Rain", f"{precip:.1f}mm" if precip is not None else "N/A")
        st.caption(f"{wx_emoji} {wx_desc}")

    # Model confidence note
    if model.train_r2 is not None:
        r2_pct = model.train_r2 * 100
        st.caption(f"📊 Model accuracy (cross-val R²): {r2_pct:.1f}%")


with col_reasons:
    st.markdown("### 🔍 Why this forecast?")
    st.caption("The factors below were considered by the model when generating this prediction.")

    for icon, heading, detail in reasons:
        st.markdown(f"""
        <div class="reason-card">
            <div>
                <span class="reason-icon">{icon}</span>
                <span class="reason-heading"> {heading}</span>
            </div>
            <div class="reason-detail">{detail}</div>
        </div>
        """, unsafe_allow_html=True)

    # Feature importance for this prediction
    with st.expander("🧪 Top model features (global importance)"):
        top_feats = list(model.feature_importances.items())[:10]
        fi_df = pd.DataFrame(top_feats, columns=["Feature", "Importance"])
        fig_fi = px.bar(
            fi_df.sort_values("Importance"),
            x="Importance", y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=["#e8f4fd", "#3a86ff"],
        )
        fig_fi.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_fi, use_container_width=True)


st.divider()

# ── Week comparison ───────────────────────────────────────────────────────
if show_week:
    st.markdown("### 📆 Week at a Glance")
    st.caption(f"Busyness forecast for the 7-day window around {target_date.strftime('%-d %B')}.")

    # Calculate week: Mon–Sun of the selected date's week
    week_start = target_date - timedelta(days=target_date.weekday())
    week_dates = [week_start + timedelta(days=i) for i in range(7)]

    week_results = []
    for wd in week_dates:
        with st.spinner(f"Forecasting {wd.strftime('%a %-d %b')}..."):
            try:
                r = model.predict(wd)
                week_results.append({
                    "date":    wd,
                    "day":     wd.strftime("%A"),
                    "label":   r["label"]["text"],
                    "colour":  r["label"]["colour"],
                    "score":   r["score"],
                    "emoji":   r["label"]["emoji"],
                })
            except Exception:
                week_results.append({"date": wd, "day": wd.strftime("%A"), "label": "?", "colour": "#ccc", "score": 0, "emoji": "❓"})

    wdf = pd.DataFrame(week_results)

    # Bar chart
    fig_week = go.Figure()
    for _, row in wdf.iterrows():
        is_selected = row["date"] == target_date
        fig_week.add_trace(go.Bar(
            x=[row["day"]],
            y=[row["score"]],
            marker_color=row["colour"],
            marker_line_color="#1a1a2e" if is_selected else "rgba(0,0,0,0)",
            marker_line_width=3 if is_selected else 0,
            text=[f"{row['score']:.0f}"],
            textposition="outside",
            textfont=dict(size=13, color="#1a1a2e"),
            name=row["day"],
            showlegend=False,
            hovertemplate=f"<b>{row['day']}</b><br>{row['label']}<br>Score: {row['score']:.0f}/100<extra></extra>",
        ))

    fig_week.update_layout(
        height=300,
        yaxis=dict(range=[0, 110], title="Busyness Score", showgrid=True, gridcolor="#f0f0f0"),
        xaxis=dict(title=""),
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.3,
        font=dict(family="Source Sans 3"),
    )
    # Add reference lines
    for y_val, label_text, colour in [(65, "Busy threshold", "#e67e22"), (25, "Quiet threshold", "#2ecc71")]:
        fig_week.add_hline(y=y_val, line_dash="dot", line_color=colour, line_width=1.5,
                           annotation_text=label_text, annotation_position="right",
                           annotation_font_size=11, annotation_font_color=colour)

    st.plotly_chart(fig_week, use_container_width=True)

    # Day tiles
    cols = st.columns(7)
    for i, row in wdf.iterrows():
        with cols[i]:
            is_selected = row["date"] == target_date
            border = f"3px solid {row['colour']}" if is_selected else "1px solid #eee"
            bg     = "#f8f9fc" if is_selected else "white"
            st.markdown(f"""
            <div style="background:{bg}; border:{border}; border-radius:10px;
                        padding:0.6rem 0.3rem; text-align:center; margin-bottom:0.3rem">
                <div style="font-size:0.7rem; color:#888">{row['date'].strftime('%a')}</div>
                <div style="font-size:0.8rem; font-weight:600; color:#1a1a2e">{row['date'].strftime('%-d %b')}</div>
                <div style="font-size:1.4rem">{row['emoji']}</div>
                <div style="font-size:0.75rem; font-weight:600; color:{row['colour']}">{row['label']}</div>
                <div style="font-size:1.1rem; font-weight:700; color:#1a1a2e">{row['score']:.0f}</div>
            </div>
            """, unsafe_allow_html=True)


st.divider()

# ── Monthly heatmap ───────────────────────────────────────────────────────
st.markdown("### 🗓️ Monthly Overview")
st.caption(f"Predicted busyness for every day in {selected_month_name} {selected_year}.")

import calendar as cal_module
_, days_this_month = cal_module.monthrange(selected_year, selected_month)
month_dates = [date(selected_year, selected_month, d) for d in range(1, days_this_month + 1)]

@st.cache_data(ttl=3600, show_spinner="Building monthly forecast...")
def get_monthly_forecasts(year: int, month: int):
    import calendar as cal_mod
    _, n_days = cal_mod.monthrange(year, month)
    dates = [date(year, month, d) for d in range(1, n_days + 1)]
    results = []
    for d in dates:
        try:
            r = model.predict(d)
            results.append({"date": d, "score": r["score"], "label": r["label"]["text"], "colour": r["label"]["colour"]})
        except Exception:
            results.append({"date": d, "score": 0, "label": "?", "colour": "#ccc"})
    return results

month_data = get_monthly_forecasts(selected_year, selected_month)
mdf = pd.DataFrame(month_data)

# Build calendar grid
first_dow = month_dates[0].weekday()  # 0=Mon
day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Pad start
grid = [None] * first_dow + month_dates
while len(grid) % 7 != 0:
    grid.append(None)

weeks = [grid[i:i+7] for i in range(0, len(grid), 7)]

week_header_cols = st.columns(7)
for i, dl in enumerate(day_labels):
    with week_header_cols[i]:
        st.markdown(f"<div style='text-align:center; font-weight:600; color:#888; font-size:0.8rem'>{dl}</div>", unsafe_allow_html=True)

for week in weeks:
    week_cols = st.columns(7)
    for i, d in enumerate(week):
        with week_cols[i]:
            if d is None:
                st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)
            else:
                row = mdf[mdf["date"] == d]
                if not row.empty:
                    s = row.iloc[0]["score"]
                    c = row.iloc[0]["colour"]
                    is_today = d == date.today()
                    is_sel   = d == target_date
                    border   = f"2px solid #1a1a2e" if is_sel else ("2px solid #3a86ff" if is_today else "1px solid #eee")
                    st.markdown(f"""
                    <div style="background:{c}22; border:{border}; border-radius:8px;
                                padding:0.4rem 0.2rem; text-align:center; margin-bottom:2px; min-height:58px">
                        <div style="font-size:0.75rem; font-weight:{'700' if is_sel else '400'}; color:#1a1a2e">{d.day}</div>
                        <div style="font-size:1.0rem; font-weight:700; color:{c}">{s:.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)


st.divider()

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#aaa; font-size:0.8rem; padding:1rem 0">
    Ballarat Busyness Forecaster &nbsp;|&nbsp;
    Data: <a href="https://data.ballarat.vic.gov.au" style="color:#3a86ff">Ballarat Open Data</a>,
    <a href="https://open-meteo.com" style="color:#3a86ff">Open-Meteo</a> &nbsp;|&nbsp;
    Built with Streamlit + XGBoost &nbsp;|&nbsp;
    Made by <strong style="color:#ffffff">Justin Molik</strong>
</div>
""", unsafe_allow_html=True)
