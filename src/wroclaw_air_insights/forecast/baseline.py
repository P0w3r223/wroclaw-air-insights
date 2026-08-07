"""Naive baselines for the 24h PM2.5 forecast.

A model is only meaningful relative to a baseline. These require no training — they
just reuse a past observation as the prediction, which for air quality is a
surprisingly strong reference. The model has to beat them to earn its keep.
"""

from __future__ import annotations

import pandas as pd


# Human-readable descriptions of each reference, so the report can name what the model
# was compared against instead of hardcoding the rule in prose.
LABELS = {
    "persistence": "same hour, yesterday",
    "seasonal": "same hour, last week",
    "climatology": "the training-period average, every hour",
    # The lead-aware rule, scored by `forecast.horizon`. Deliberately a separate entry
    # rather than a redefinition of "persistence": at a 24h lead the two are the same
    # prediction, and everywhere else they are not. Repointing the existing name would
    # have made every published figure that quotes it mean something new without saying so.
    "origin_persistence": "the reading at the moment the forecast is issued",
}


def persistence_prediction(features: pd.DataFrame) -> pd.Series:
    """Predict PM2.5[T] = PM2.5[T-24h] — yesterday, same hour."""
    return features["pm25_lag_24"]


def seasonal_naive_prediction(features: pd.DataFrame) -> pd.Series:
    """Predict PM2.5[T] = PM2.5[T-168h] — same hour, last week."""
    return features["pm25_lag_168"]
