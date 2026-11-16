# Urban Flood Prediction and Early Warning Intelligence

A small, beginner-friendly Streamlit learning project. It explores a **synthetic demonstration dataset**, fits a baseline scikit-learn classifier, and also displays current weather model data for a searchable, selectable city or region.

> **Demonstration only.** All included observations are generated, not measured in a real city. Predictions and threshold messages are not official forecasts or emergency alerts. Do not use them to make safety decisions.

## What you will see

- A dashboard with a selectable model and estimated class probabilities for fictional conditions.
- Searchable city/region selection with current weather, recent rainfall, and next-day rain chance from Open-Meteo.
- A map marker and hourly rainfall/rain-chance charts for the selected region.
- A side-by-side comparison of Logistic Regression, Random Forest, and Gradient Boosting.
- Holdout and 5-fold cross-validation scores, a confusion matrix, and permutation-based feature influence.
- Configurable rainfall thresholds applied to selected-region weather data.
- Editable rainfall, short-term rainfall, river level, soil saturation, and drainage blockage inputs.
- Configurable thresholds that produce a clearly labeled *demo watch* panel.
- Dataset exploration: summary statistics, risk counts, and a rainfall-versus-risk chart.
- A reproducible model comparison, with all scores clearly tied to synthetic labels.
- A bundled CSV and a deterministic script to regenerate it.

## Requirements

- Python 3.10 or newer.
- Internet access for the first installation of the Python packages in `requirements.txt`.
- Internet access while using the dashboard to retrieve location and weather data from Open-Meteo.
- A terminal (PowerShell on Windows, Terminal on macOS/Linux).

## Setup and run

Open a terminal in this project folder, then run the commands for your operating system.

### Windows PowerShell

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal and activate again. You can also skip activation and use `.venv\Scripts\python.exe -m pip ...` and `.venv\Scripts\python.exe -m streamlit run app.py`.

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) to open in your browser. Stop the app with Ctrl+C.

## Project map

```text
urban-flood-intelligence/
├── app.py                    # Dashboard layout, controls, risk and demo watch
├── generate_dataset.py       # Rebuild the bundled synthetic CSV
├── requirements.txt          # Lightweight Python dependencies
├── .streamlit/
│   └── config.toml            # Teal, aqua, indigo, and amber visual theme
├── data/
│   └── demo_flood_data.csv   # Included synthetic observations
└── flood_intelligence/
    ├── __init__.py
    ├── data.py               # Generate or load the synthetic dataset
    ├── analysis.py           # Small reusable summaries and chart inputs
    ├── live_weather.py       # Place search and live weather API client
    └── model.py              # Three models, validation, diagnostics, prediction
```

## How it works

1. **Live weather:** `flood_intelligence/live_weather.py` searches for places and fetches current modeled weather, recent hourly precipitation, and rain probability using the no-key Open-Meteo geocoding and forecast APIs. Results are cached briefly to avoid repeated requests.
2. **Data:** `flood_intelligence/data.py` creates a reproducible set of fictional hourly weather and drainage observations. A transparent rule based on rainfall, river level, soil saturation, and drainage blockage assigns a synthetic label (`Low`, `Moderate`, or `High`). This label is a teaching target, not ground truth about a real flood.
3. **Exploration:** `analysis.py` calculates simple summaries. The dashboard displays class counts, descriptive statistics, and a scatter plot so you can inspect the generated patterns before modeling.
4. **Model comparison:** `model.py` compares scaled Logistic Regression, Random Forest, and Gradient Boosting on the same stratified 75/25 holdout, then calculates 5-fold cross-validation scores. It also creates a confusion matrix and permutation importance for the selected model. These measures describe synthetic labels only.
5. **Interactive estimate:** the dashboard lets the learner choose a model and passes fictional sidebar inputs to `predict_proba`. It shows a class and probabilities. None of the models consume the selected region's live weather.
6. **Forecast exploration:** hourly precipitation and rain-probability charts plus a map marker add geographic and time context from Open-Meteo. Weather-model values still do not establish local flooding.
7. **Rainfall watch:** configurable thresholds compare live model precipitation for the selected location with user-entered values. This is an informational indicator, not an official alert.

## Try these learning activities

- Change rainfall thresholds in the sidebar and compare the watch panel with the model output.
- Adjust `N_DEMO_ROWS` and the fixed seed in `data.py`; observe how a small dataset changes the metrics.
- Compare the cross-validation score and confusion matrix for each model.
- Change a scenario input and inspect the selected model's class probabilities.
- Add an input feature to `FEATURE_COLUMNS` in `data.py`, then add the matching widget in `app.py`.
- Compare the rule-generated label with model predictions. A high score can simply mean that the model learned the synthetic labeling rule; it does not validate real-world flood prediction.

## Limitations

The live feed contains weather-model outputs, not on-the-ground rain gauges, river levels, soil saturation sensors, drainage network data, or flood observations. The model remains trained only on synthetic data and must not be used to predict real events. The rainfall indicator is not calibrated as a public warning and has no alert delivery. Real flood warnings should come from official local authorities.
