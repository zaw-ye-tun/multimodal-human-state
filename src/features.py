from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS = [
    "heart_rate",
    "hr_change",
    "hr_rolling_mean",
    "movement",
    "movement_rolling_mean",
]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple temporal context features used by the classifier."""
    result = df.copy()
    result["heart_rate"] = result["heart_rate"].astype(float)
    result["movement"] = result["movement"].astype(float)
    result["hr_change"] = result["heart_rate"].diff().fillna(0)
    result["hr_rolling_mean"] = (
        result["heart_rate"].rolling(10, min_periods=1).mean()
    )
    result["movement_rolling_mean"] = (
        result["movement"].rolling(10, min_periods=1).mean()
    )
    return result

