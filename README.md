# 🏙️ Ballarat Busyness Forecaster

An AI-powered dashboard that predicts how busy Ballarat town centre will be on any given day, using real local open data, live weather forecasts, public holidays and a curated events calendar.

---

## 📦 Setup (VS Code / Terminal)

### 1. Install Python dependencies

Open a terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

> **Tip:** It's best practice to use a virtual environment:
> ```bash
> python -m venv venv
> venv\Scripts\activate        # Windows
> source venv/bin/activate      # Mac/Linux
> pip install -r requirements.txt
> ```

### 2. Run the dashboard

```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`

---

## 🗂️ Project Structure

```
ballarat_busyness/
├── app.py              ← Streamlit dashboard (run this)
├── model.py            ← XGBoost ML model + feature engineering
├── data_fetcher.py     ← All data sources (Ballarat API, weather, holidays, events)
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## 📊 Data Sources

| Source | What it provides | Key detail |
|--------|-----------------|------------|
| [Ballarat Open Data](https://data.ballarat.vic.gov.au) | Pedestrian counts, parking transactions | Free REST API, no key needed |
| [Open-Meteo](https://open-meteo.com) | Historical + forecast weather | Free, no API key required |
| `holidays` Python library | Victorian public holidays | Automatically correct per year |
| Local events calendar | Begonia Festival, Ballarat Show, etc. | Curated in `data_fetcher.py` |

---

## 🧠 How the Model Works

The XGBoost model is trained on historical busyness data (real or synthetic if the API is unavailable) combined with:

- **Temporal features:** day of week, month, week of year, season
- **Calendar features:** public holidays, eves of holidays, long weekends, school holidays
- **Event features:** local event presence and impact score
- **Weather features:** max temperature, rainfall, weather code, sunshine hours

On first run, the model fetches historical data and trains automatically (~15–30 seconds). It then caches in memory for the session.

**Accuracy:** Typically 70–85% R² on cross-validation with real data.

---

## 🎪 Updating the Events Calendar

Open `data_fetcher.py` and find the `LOCAL_EVENTS` list near the top. Each event is a dict:

```python
{"name": "Your Event Name", "month": 3, "start_day": 7, "end_day": 9, "impact": 3},
```

- `month`: integer (1–12)
- `start_day / end_day`: day of the month
- `impact`: `1` = minor, `2` = moderate, `3` = major

---

## ⚠️ Notes

- Weather forecasts are available ~16 days into the future via Open-Meteo. For dates beyond that, the model uses seasonal weather averages.
- If the Ballarat Open Data API is unreachable (e.g. network issues), the model automatically falls back to realistic synthetic training data. Predictions remain meaningful but are not calibrated to actual pedestrian counts.
- This tool is for community research purposes. Predictions are probabilistic — not guarantees.

---

## 🔧 Troubleshooting

**`ModuleNotFoundError`** — Run `pip install -r requirements.txt`

**Streamlit not found** — Make sure your virtual environment is activated

**Model trains on synthetic data every time** — The Ballarat Open Data API may be temporarily unavailable. Check `https://data.ballarat.vic.gov.au` in your browser.

**Slow first load** — The model fetches several years of weather data on first run. Subsequent loads use Streamlit's cache.
# Ballarat-Predictor
