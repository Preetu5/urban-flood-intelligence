"""Small no-key client for Open-Meteo geocoding and current weather data."""

from typing import Any

import requests


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


def search_locations(query: str) -> list[dict[str, Any]]:
    """Find selectable cities or regions by name; no account or API key needed."""
    response = requests.get(
        GEOCODING_URL,
        params={"name": query, "count": 10, "language": "en", "format": "json"},
        timeout=15,
    )
    response.raise_for_status()
    return response.json().get("results", []) or []


def fetch_weather(latitude: float, longitude: float) -> dict[str, Any]:
    """Get latest modeled weather, previous hourly rain and next-day outlook."""
    response = requests.get(
        WEATHER_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,relative_humidity_2m,precipitation,rain,showers,"
                "weather_code,wind_speed_10m,wind_gusts_10m"
            ),
            "hourly": "precipitation,precipitation_probability",
            "past_hours": 24,
            "forecast_hours": 24,
            "timezone": "auto",
        },
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def weather_description(code: int | None) -> str:
    """Translate the standard WMO weather code into a short label."""
    descriptions = {
        0: "Clear",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Freezing fog",
        51: "Light drizzle",
        53: "Drizzle",
        55: "Heavy drizzle",
        56: "Freezing drizzle",
        57: "Heavy freezing drizzle",
        61: "Light rain",
        63: "Rain",
        65: "Heavy rain",
        66: "Freezing rain",
        67: "Heavy freezing rain",
        71: "Light snow",
        73: "Snow",
        75: "Heavy snow",
        77: "Snow grains",
        80: "Light rain showers",
        81: "Rain showers",
        82: "Heavy rain showers",
        85: "Snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with hail",
        99: "Severe thunderstorm with hail",
    }
    return descriptions.get(code, "Conditions unavailable")


def rainfall_totals(weather: dict[str, Any]) -> tuple[float, float, float | None]:
    """Return latest hourly rain, trailing 24-hour total, and next-day rain chance."""
    hourly = weather.get("hourly", {})
    times = hourly.get("time", [])
    rain = hourly.get("precipitation", [])
    chance = hourly.get("precipitation_probability", [])
    now = weather.get("current", {}).get("time", "")

    past_values = [value for time, value in zip(times, rain) if time <= now and value is not None]
    future = [
        probability
        for time, probability in zip(times, chance)
        if time > now and probability is not None
    ]
    recent_24 = past_values[-24:]
    last_hour = recent_24[-1] if recent_24 else 0.0
    last_day = sum(recent_24)
    max_chance = max(future) if future else None
    return float(last_hour), float(last_day), (float(max_chance) if max_chance is not None else None)
