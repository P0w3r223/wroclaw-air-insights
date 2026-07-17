"""Naive baselines for the 24h PM2.5 forecast.

A model is only meaningful relative to a baseline. These require no training — they
just reuse a past observation as the prediction, which for air quality is a
surprisingly strong reference. The model has to beat them to earn its keep.
"""

from __future__ import annotations

import pandas as pd


def persistence_prediction(features: pd.DataFrame) -> pd.Series:
    """Predict PM2.5[T] = PM2.5[T-24h] — yesterday, same hour."""
    return features["pm25_lag_24"]


def seasonal_naive_prediction(features: pd.DataFrame) -> pd.Series:
    """Predict PM2.5[T] = PM2.5[T-168h] — same hour, last week."""
    return features["pm25_lag_168"]
