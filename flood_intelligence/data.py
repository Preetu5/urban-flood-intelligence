"""Create and load the project's fictional, reproducible demonstration data."""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "demo_flood_data.csv"
FEATURE_COLUMNS = [
    "rainfall_1h_mm",
    "rainfall_24h_mm",
    "river_level_m",
    "soil_saturation_pct",
    "drain_blockage_pct",
]


def generate_demo_data(rows: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Generate fictional hourly observations and rule-derived teaching labels."""
    rng = np.random.default_rng(seed)
    rainfall_1h = rng.gamma(shape=1.5, scale=7.0, size=rows).clip(0, 90)
    rainfall_24h = (rainfall_1h * rng.uniform(2, 7, rows) + rng.gamma(2, 8, rows)).clip(0, 300)
    soil = (25 + rainfall_24h * 0.24 + rng.normal(0, 12, rows)).clip(5, 100)
    river = (1.1 + rainfall_24h * 0.012 + rng.normal(0, 0.35, rows)).clip(0.2, 6)
    blockage = rng.beta(2, 4, rows) * 100

    # Weighted index is only a convenient way to create a learnable demo label.
    score = (
        rainfall_1h / 35
        + rainfall_24h / 170
        + river / 3.6
        + soil / 145
        + blockage / 220
    )
    risk = np.select([score >= 2.75, score >= 1.8], ["High", "Moderate"], default="Low")
    timestamps = pd.date_range("2025-06-01", periods=rows, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "rainfall_1h_mm": rainfall_1h.round(1),
            "rainfall_24h_mm": rainfall_24h.round(1),
            "river_level_m": river.round(2),
            "soil_saturation_pct": soil.round(1),
            "drain_blockage_pct": blockage.round(1),
            "risk_label": risk,
        }
    )


def load_demo_data() -> pd.DataFrame:
    """Load the bundled CSV; regenerate it if it is missing."""
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH, parse_dates=["timestamp_utc"])
    data = generate_demo_data()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(DATA_PATH, index=False)
    return data
