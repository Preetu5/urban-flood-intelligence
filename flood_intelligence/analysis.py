"""Small exploratory-data helpers used by the dashboard."""

import pandas as pd


def risk_counts(data: pd.DataFrame) -> pd.Series:
    """Count rows in each risk class, keeping the display order stable."""
    return data["risk_label"].value_counts().reindex(["Low", "Moderate", "High"], fill_value=0)


def feature_summary(data: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Return a rounded descriptive-statistics table for model inputs."""
    return data[feature_columns].describe().T.round(2)
