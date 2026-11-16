"""Streamlit dashboard for Urban Flood Prediction and Early Warning Intelligence."""

import pandas as pd
import streamlit as st

from flood_intelligence.analysis import feature_summary, risk_counts
from flood_intelligence.data import FEATURE_COLUMNS, load_demo_data
from flood_intelligence.live_weather import (
    fetch_weather,
    rainfall_totals,
    search_locations,
    weather_description,
)
from flood_intelligence.model import MODEL_NAMES, predict_risk, train_model_suite


st.set_page_config(page_title="Urban Flood Intelligence", page_icon="🌧️", layout="wide")

st.html(
    """
    <style>
      .stApp {
        background: linear-gradient(128deg, #eef6ff 0%, #e8f8f4 48%, #fff6ea 100%);
        background-attachment: fixed;
      }
      .stSidebar {
        background: linear-gradient(180deg, #e1f4f1 0%, #edf4ff 100%);
      }
      .flood-app-hero {
        position: relative;
        display: flex;
        align-items: center;
        gap: 1.25rem;
        overflow: hidden;
        margin: .45rem 0 1.25rem;
        padding: 1.65rem 1.8rem;
        border: 1px solid rgba(255,255,255,.56);
        border-radius: 24px;
        color: #f7ffff;
        background: linear-gradient(115deg, #073b4c 0%, #087f8c 52%, #2aa889 100%);
        box-shadow: 0 18px 44px rgba(7, 83, 100, .19);
        animation: flood-hero-appear .65s ease-out both;
      }
      .flood-app-hero::after {
        content: "";
        position: absolute;
        width: 25rem;
        height: 25rem;
        right: -8rem;
        top: -15rem;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(255,255,255,.23), rgba(255,255,255,0) 68%);
        pointer-events: none;
      }
      .flood-hero-icon {
        display: grid;
        place-items: center;
        flex: 0 0 4.25rem;
        width: 4.25rem;
        height: 4.25rem;
        border: 1px solid rgba(255,255,255,.34);
        border-radius: 1.35rem;
        background: rgba(255,255,255,.15);
        font-size: 2rem;
        animation: flood-icon-float 4s ease-in-out infinite;
      }
      .flood-hero-eyebrow {
        margin: 0 0 .35rem;
        color: #baf7e6;
        font-size: .72rem;
        font-weight: 750;
        letter-spacing: .16em;
        text-transform: uppercase;
      }
      .flood-app-hero h1 {
        margin: 0;
        color: #fff;
        font-size: clamp(1.65rem, 3vw, 2.45rem);
        font-weight: 760;
        letter-spacing: -.035em;
        line-height: 1.12;
      }
      .flood-app-hero p {
        margin: .55rem 0 0;
        color: rgba(244,255,255,.86);
        font-size: .98rem;
      }
      .flood-hero-tags {
        display: flex;
        flex-wrap: wrap;
        gap: .5rem;
        margin-top: .95rem;
      }
      .flood-hero-tags span {
        padding: .3rem .65rem;
        border: 1px solid rgba(255,255,255,.24);
        border-radius: 999px;
        background: rgba(255,255,255,.12);
        color: #f2ffff;
        font-size: .76rem;
        font-weight: 600;
      }
      @keyframes flood-hero-appear {
        from { opacity: 0; transform: translateY(9px); }
        to { opacity: 1; transform: translateY(0); }
      }
      @keyframes flood-icon-float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-4px); }
      }
      @media (prefers-reduced-motion: reduce) {
        .flood-app-hero, .flood-hero-icon { animation: none; }
      }
    </style>
    <header class="flood-app-hero">
      <div class="flood-hero-icon" aria-hidden="true">🌧️</div>
      <div>
        <div class="flood-hero-eyebrow">Regional weather · urban resilience</div>
        <h1>Urban Flood Intelligence</h1>
        <p>Explore live weather context and learn how flood-risk models work.</p>
        <div class="flood-hero-tags">
          <span>🌍 Select a region</span>
          <span>🌦️ Live weather context</span>
          <span>🧪 Synthetic learning model</span>
        </div>
      </div>
    </header>
    """
)
st.caption("Student prototype · Live weather context plus a separate synthetic-data learning model")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_location_search(query: str):
    return search_locations(query)


@st.cache_data(ttl=900, show_spinner=False)
def cached_weather(latitude: float, longitude: float):
    return fetch_weather(latitude, longitude)

with st.sidebar:
    st.header("Live weather region")
    location_query = st.text_input("Search for a city or region", value="Mumbai")
    locations = []
    if len(location_query.strip()) >= 2:
        try:
            locations = cached_location_search(location_query.strip())
        except Exception as error:
            st.warning(f"Location search is temporarily unavailable: {error}")
    if locations:
        def location_label(location):
            parts = [location.get("name"), location.get("admin1"), location.get("country")]
            return ", ".join(dict.fromkeys(part for part in parts if part))

        chosen_location = st.selectbox(
            "Choose a matching place",
            options=locations,
            format_func=location_label,
        )
        st.caption(
            f"Coordinates: {chosen_location['latitude']:.3f}, "
            f"{chosen_location['longitude']:.3f} · {chosen_location.get('timezone', 'local time')}"
        )
        if st.button("Refresh live weather"):
            cached_weather.clear()
    else:
        chosen_location = None
        st.caption("Type at least two letters to search for places.")

    st.divider()
    st.header("Current scenario")
    st.caption("Adjust the fictional measurements to explore how inputs affect this baseline model.")
    selected_model = st.selectbox("Model to explore", MODEL_NAMES, index=0)
    rainfall_1h = st.slider("Rainfall in last hour (mm)", 0.0, 90.0, 24.0, 1.0)
    rainfall_24h = st.slider("Rainfall in last 24 hours (mm)", 0.0, 300.0, 95.0, 5.0)
    river_level = st.slider("River level (m)", 0.2, 6.0, 2.2, 0.1)
    soil = st.slider("Soil saturation (%)", 5.0, 100.0, 62.0, 1.0)
    blockage = st.slider("Drain blockage (%)", 0.0, 100.0, 30.0, 1.0)

    st.divider()
    st.header("Demo watch thresholds")
    st.caption("These simple rules are editable examples, separate from the model.")
    rain_threshold = st.number_input("1-hour rainfall trigger (mm)", 0.0, 90.0, 35.0, 1.0)
    river_threshold = st.number_input("River level trigger (m)", 0.2, 6.0, 3.2, 0.1)
    blockage_threshold = st.number_input("Drain blockage trigger (%)", 0.0, 100.0, 70.0, 1.0)

    st.divider()
    st.header("Live rainfall watch thresholds")
    live_rain_1h_threshold = st.number_input(
        "Rain in last hour trigger (mm)", 0.0, 300.0, 35.0, 1.0
    )
    live_rain_24h_threshold = st.number_input(
        "Rain in last 24 hours trigger (mm)", 0.0, 1000.0, 100.0, 5.0
    )

data = load_demo_data()
models, model_comparison, model_diagnostics = train_model_suite(data)
model = models[selected_model]
metrics = model_comparison.set_index("Model").loc[selected_model]

st.header("Live weather context")
st.caption(
    "Current weather model data for the selected region. This is not a river gauge, "
    "drain sensor, flood observation, or official warning."
)
if chosen_location is None:
    st.info("Search for a city or region in the sidebar to load weather data.")
else:
    try:
        live_weather = cached_weather(
            float(chosen_location["latitude"]), float(chosen_location["longitude"])
        )
        current = live_weather.get("current", {})
        rain_1h, rain_24h, rain_chance = rainfall_totals(live_weather)
        live_place = ", ".join(
            part
            for part in [chosen_location.get("name"), chosen_location.get("country")]
            if part
        )
        st.subheader(f"{live_place} · {current.get('time', 'latest available time')}")
        st.caption(
            f"Provider: Open-Meteo weather models · Display timezone: "
            f"{live_weather.get('timezone', 'API local time')} · Refreshes about every 15 minutes; "
            "cached here for up to 15 minutes."
        )
        weather_cols = st.columns(5)
        weather_cols[0].metric("Conditions", weather_description(current.get("weather_code")))
        weather_cols[1].metric("Temperature", f"{current.get('temperature_2m', '—')} °C")
        weather_cols[2].metric("Rain · last hour", f"{rain_1h:.1f} mm")
        weather_cols[3].metric("Rain · last 24 hours", f"{rain_24h:.1f} mm")
        weather_cols[4].metric(
            "Forecast rain chance · next 24h",
            f"{rain_chance:.0f}%" if rain_chance is not None else "—",
        )
        detail_cols = st.columns(3)
        detail_cols[0].metric("Humidity", f"{current.get('relative_humidity_2m', '—')}%")
        detail_cols[1].metric("Wind", f"{current.get('wind_speed_10m', '—')} km/h")
        detail_cols[2].metric("Wind gusts", f"{current.get('wind_gusts_10m', '—')} km/h")

        hourly = pd.DataFrame({
            "time": live_weather.get("hourly", {}).get("time", []),
            "Rainfall (mm)": live_weather.get("hourly", {}).get("precipitation", []),
            "Rain chance (%)": live_weather.get("hourly", {}).get("precipitation_probability", []),
        })
        if not hourly.empty:
            hourly["time"] = pd.to_datetime(hourly["time"])
            future = hourly[hourly["time"] > pd.Timestamp(current.get("time"))].head(24).set_index("time")
            if not future.empty:
                with st.expander("Explore the next 24 hours and map this location", expanded=True):
                    map_col, forecast_col = st.columns([1, 2])
                    with map_col:
                        st.map(pd.DataFrame({"lat": [chosen_location["latitude"]], "lon": [chosen_location["longitude"]]}), zoom=9)
                        st.caption("Selected city/region location")
                    with forecast_col:
                        st.subheader("Hourly rain forecast")
                        st.area_chart(future[["Rainfall (mm)"]], height=180)
                        st.line_chart(future[["Rain chance (%)"]], height=180)

        st.subheader("Rainfall threshold panel · selected region")
        live_triggers = []
        if rain_1h >= live_rain_1h_threshold:
            live_triggers.append(
                f"Last-hour rainfall reached {rain_1h:.1f} mm "
                f"(configured threshold: {live_rain_1h_threshold:g} mm)."
            )
        if rain_24h >= live_rain_24h_threshold:
            live_triggers.append(
                f"Last-24-hour rainfall reached {rain_24h:.1f} mm "
                f"(configured threshold: {live_rain_24h_threshold:g} mm)."
            )
        if live_triggers:
            st.warning("Rainfall threshold crossed · information only")
            for trigger in live_triggers:
                st.write(f"- {trigger}")
        else:
            st.success("Selected live rainfall readings are below the configured thresholds.")
        st.caption(
            "This is a user-configured rainfall indicator based on weather-model values. "
            "It is not a flood prediction or public emergency alert. Check official local guidance."
        )
    except Exception as error:
        st.error(f"Live weather could not be loaded: {error}")

st.divider()
st.header("Synthetic baseline model exercise")
st.caption(
    "This model is trained only on generated teaching data. It does not use the selected "
    "region's live weather values and is not a real flood-risk forecast."
)
scenario = {
    "rainfall_1h_mm": rainfall_1h,
    "rainfall_24h_mm": rainfall_24h,
    "river_level_m": river_level,
    "soil_saturation_pct": soil,
    "drain_blockage_pct": blockage,
}
prediction, probabilities = predict_risk(model, scenario)

st.warning(
    "SYNTHETIC DEMO ONLY — this dashboard does not use live observations and must not be used "
    "for emergency decisions. Follow official local authority guidance."
)

col1, col2, col3 = st.columns(3)
col1.metric(f"{selected_model} class", prediction)
col2.metric("Estimated class probability", f"{probabilities[prediction]:.0%}")
col3.metric("Held-out accuracy on synthetic labels", f"{metrics['Holdout accuracy']:.0%}")

left, right = st.columns(2)
with left:
    st.subheader("Model probabilities")
    st.bar_chart(pd.Series(probabilities, name="Estimated probability"))
with right:
    st.subheader("Early-warning panel · demo rules")
    triggers = []
    if rainfall_1h >= rain_threshold:
        triggers.append(f"1-hour rainfall is at or above {rain_threshold:g} mm")
    if river_level >= river_threshold:
        triggers.append(f"river level is at or above {river_threshold:g} m")
    if blockage >= blockage_threshold:
        triggers.append(f"drain blockage is at or above {blockage:g}%")
    if triggers:
        st.error("DEMO WATCH — configured threshold crossed")
        for trigger in triggers:
            st.write(f"- {trigger}")
    else:
        st.success("No configured demo threshold crossed for these inputs.")
    st.caption("This configurable teaching panel is not a public warning or emergency alert.")

st.divider()
st.header("Compare and understand the models")
st.caption("All scores below measure agreement with generated teaching labels, not real flood outcomes.")
st.dataframe(
    model_comparison.style.format({column: "{:.1%}" for column in model_comparison.columns if column != "Model"}),
    width="stretch",
)
st.bar_chart(model_comparison.set_index("Model")[["Weighted F1", "Balanced accuracy", "5-fold F1 mean"]])
diagnostic = model_diagnostics[selected_model]
diag_left, diag_right = st.columns(2)
with diag_left:
    st.subheader(f"{selected_model}: confusion matrix")
    st.dataframe(diagnostic["confusion"], width="stretch")
    st.caption("Rows are actual synthetic classes; columns are model guesses on held-out rows.")
with diag_right:
    st.subheader("Input influence (permutation importance)")
    st.bar_chart(diagnostic["importance"].set_index("Input feature")[["Importance"]])
    st.caption("A feature is more influential when shuffling it lowers the model's held-out score.")

st.divider()
st.header("Explore the synthetic dataset")
st.write(f"{len(data):,} fictional hourly records. Labels were created by a simple rule for learning.")
eda_left, eda_right = st.columns(2)
with eda_left:
    st.subheader("Generated risk classes")
    st.bar_chart(risk_counts(data))
    st.subheader("Feature summary")
    st.dataframe(feature_summary(data, FEATURE_COLUMNS), use_container_width=True)
with eda_right:
    st.subheader("Rainfall and synthetic risk labels")
    chart_data = data[["rainfall_24h_mm", "rainfall_1h_mm", "risk_label"]].sample(
        min(350, len(data)), random_state=42
    ).sort_values("risk_label")
    st.scatter_chart(
        chart_data,
        x="rainfall_24h_mm",
        y="rainfall_1h_mm",
        color="risk_label",
        height=330,
    )
    st.subheader("Model comparison notes")
    st.write(
        "Three models are compared: Logistic Regression, Random Forest, and Gradient Boosting. "
        "Each is evaluated on the same stratified 75/25 holdout and with 5-fold cross-validation. "
        "The selected model drives the fictional scenario above."
    )
    st.write(f"Held-out rows: {diagnostic['test_rows']:,} · 5-fold weighted F1: {metrics['5-fold F1 mean']:.1%} ± {metrics['5-fold F1 spread']:.1%}")
    st.caption("Synthetic-label metrics describe this generated exercise, not real-world accuracy.")

with st.expander("See the demo data"):
    st.dataframe(data.head(50), use_container_width=True)
    st.download_button(
        "Download synthetic demo CSV",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="demo_flood_data.csv",
        mime="text/csv",
    )
