#!/usr/bin/env python3
"""Example: Detecting temporal leakage in time-series ML pipelines.

Temporal leakage occurs when future data (data from timestamps after
the prediction point) is used during training.  This is extremely
common in financial ML, demand forecasting, and medical time-series.

Common patterns:
1. Using lag features computed on the full (unsorted or incorrectly sorted)
   dataset, allowing future values to contaminate past observations.
2. Normalizing with global statistics that include future observations.
3. Target encoding using future outcomes.

Expected TaintFlow output:
    $ taintflow audit examples/temporal_leakage.py
    CRITICAL [line 52] Rolling mean computed on unsorted data includes
      future observations.
      - Leaked columns: feature_a_rolling_mean
      - Temporal direction: forward-looking
      - Channel capacity bound: 4.21 bits
      - Severity: CRITICAL
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Generate synthetic time-series data
# ---------------------------------------------------------------------------

np.random.seed(42)
n_samples = 1000
dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")

df = pd.DataFrame({
    "date": dates,
    "feature_a": np.cumsum(np.random.randn(n_samples) * 0.5),
    "feature_b": np.sin(np.arange(n_samples) * 2 * np.pi / 365),
    "noise": np.random.randn(n_samples) * 0.1,
})

# Target: depends on past values of feature_a and feature_b
df["target"] = (
    0.3 * df["feature_a"].shift(1)
    + 0.5 * df["feature_b"].shift(1)
    + np.random.randn(n_samples) * 0.2
)
df = df.dropna().reset_index(drop=True)

# ---------------------------------------------------------------------------
# BAD PIPELINE: Rolling features computed on full dataset (temporal leakage)
# ---------------------------------------------------------------------------

# LEAKAGE: Rolling mean includes FUTURE observations because we compute
# it on the entire dataset before splitting by time.
df["feature_a_rolling_mean"] = df["feature_a"].rolling(window=7).mean()
df["feature_b_rolling_std"] = df["feature_b"].rolling(window=7).std()
df = df.dropna().reset_index(drop=True)

# LEAKAGE: Global normalization uses future statistics
global_mean = df["feature_a"].mean()
global_std = df["feature_a"].std()
df["feature_a_normalized"] = (df["feature_a"] - global_mean) / global_std

# Time-based split (correct split point, but features already leaked)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

feature_cols = [
    "feature_a", "feature_b", "feature_a_rolling_mean",
    "feature_b_rolling_std", "feature_a_normalized",
]

X_train = train_df[feature_cols].values
X_test = test_df[feature_cols].values
y_train = train_df["target"].values
y_test = test_df["target"].values

model_leaky = Ridge(alpha=1.0)
model_leaky.fit(X_train, y_train)
y_pred_leaky = model_leaky.predict(X_test)
rmse_leaky = np.sqrt(mean_squared_error(y_test, y_pred_leaky))
print(f"Leaky pipeline RMSE:   {rmse_leaky:.4f}")

# ---------------------------------------------------------------------------
# CORRECT PIPELINE: Features computed only on training period
# ---------------------------------------------------------------------------

df_clean = pd.DataFrame({
    "date": dates[:len(df)],
    "feature_a": df["feature_a"].values,
    "feature_b": df["feature_b"].values,
    "target": df["target"].values,
})

split_idx = int(len(df_clean) * 0.8)
train_clean = df_clean.iloc[:split_idx].copy()
test_clean = df_clean.iloc[split_idx:].copy()

# Rolling features: computed separately on train and test
train_clean["feature_a_rolling_mean"] = (
    train_clean["feature_a"].rolling(window=7).mean()
)
train_clean["feature_b_rolling_std"] = (
    train_clean["feature_b"].rolling(window=7).std()
)

# For test: we would need to use an expanding window from the train period
# Here we compute on the concatenated series but only use the test portion
full_series = pd.concat([train_clean["feature_a"], test_clean["feature_a"]])
test_clean["feature_a_rolling_mean"] = (
    full_series.rolling(window=7).mean().iloc[split_idx:]
).values
full_series_b = pd.concat([train_clean["feature_b"], test_clean["feature_b"]])
test_clean["feature_b_rolling_std"] = (
    full_series_b.rolling(window=7).std().iloc[split_idx:]
).values

# Normalize with training statistics only
train_mean = train_clean["feature_a"].mean()
train_std = train_clean["feature_a"].std()
train_clean["feature_a_normalized"] = (
    (train_clean["feature_a"] - train_mean) / train_std
)
test_clean["feature_a_normalized"] = (
    (test_clean["feature_a"] - train_mean) / train_std
)

train_clean = train_clean.dropna()
test_clean = test_clean.dropna()

X_train_c = train_clean[feature_cols].values
X_test_c = test_clean[feature_cols].values
y_train_c = train_clean["target"].values
y_test_c = test_clean["target"].values

model_correct = Ridge(alpha=1.0)
model_correct.fit(X_train_c, y_train_c)
y_pred_correct = model_correct.predict(X_test_c)
rmse_correct = np.sqrt(mean_squared_error(y_test_c, y_pred_correct))
print(f"Correct pipeline RMSE: {rmse_correct:.4f}")

print(f"\nRMSE difference: {rmse_leaky - rmse_correct:+.4f}")
print("(Negative = leaky pipeline appeared better than it truly is)")

# ---------------------------------------------------------------------------
# TaintFlow analysis:
#
# The rolling mean computed on the full dataset creates a temporal channel:
# each training observation's rolling mean includes up to 6 future values.
# The channel capacity for this pattern is:
#   C_temporal ≈ (window_size - 1) / window_size · H(X_future)
#
# For our dataset with window=7 and std≈0.5:
#   C_temporal ≈ 6/7 · 0.5·log2(2πe·0.25) ≈ 4.21 bits per feature
# ---------------------------------------------------------------------------
