"""Data cleaning and feature engineering shared by training and prediction."""
import numpy as np
import pandas as pd

CATEGORICALS = ["pickup", "delivery", "equipment"]
FEATURES = [
    "distance", "great_circle_miles", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "weight", "market_index", "quote_signal", "day_of_year", "day_of_week", "month",
] + CATEGORICALS
MAX_WEIGHT = 47_500  # legal-ish cap; the data is clipped at this value


def great_circle_miles(df: pd.DataFrame) -> pd.Series:
    """Straight-line (haversine) miles between pickup and delivery coordinates."""
    lat1, lon1, lat2, lon2 = (np.radians(df[c]) for c in
                              ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 3958.8 * np.arcsin(np.sqrt(a))


def clean_inputs(df: pd.DataFrame, weight_median: float) -> pd.DataFrame:
    """Fix input-column issues. Applied identically to training and prediction data."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    # Negative weights are sign errors (magnitudes look like normal loads) -> take absolute value.
    df["weight"] = df["weight"].abs().clip(upper=MAX_WEIGHT)
    # ~0.6% of weights are missing -> fill with the training median.
    df["weight"] = df["weight"].fillna(weight_median)
    # market_index missing (~0.8%) is left as NaN: LightGBM handles missing values natively.
    return df


def remove_rate_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop training rows whose rate-per-mile is implausible for their lane.

    About 1.3% of rows are 2-7x above or below the typical rate for the same
    lane + equipment (likely data-entry / unit errors). We compare each row to the
    median rate-per-mile of its lane and drop rows outside 0.5x-2x of it.
    Only used on TRAINING rows - we never drop prediction rows.
    """
    rpm = df["posted_rate"] / df["distance"]
    lane_median = rpm.groupby([df["pickup"], df["delivery"], df["equipment"]]).transform("median")
    ratio = rpm / lane_median
    return df[(ratio > 0.5) & (ratio < 2.0)].copy()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["great_circle_miles"] = great_circle_miles(df)
    df["day_of_year"] = df["date"].dt.dayofyear
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    for col in CATEGORICALS:
        df[col] = df[col].astype("category")
    return df
