"""Example: Detecting target encoding leakage with TaintFlow.

Target encoding replaces each categorical value with the mean of the target
variable for that category. When done naively on the full dataset, the
encoding for a test row directly incorporates that row's own y-value,
creating severe information leakage.

Expected TaintFlow output:
    $ taintflow audit examples/target_encoding_leakage.py
    WARNING [line 43] Manual target encoding uses target variable 'y'
      computed over full dataset (including test rows).
      - Leaked columns: city_encoded
      - Parameter taint: ⊤ (Both)
      - Channel capacity bound: 3.32 bits (CountingBound, k=10)
      - Severity: HIGH

    Summary: 1 leakage warning(s), 1 leaked column(s).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# BAD PIPELINE: target encoding on full dataset before split
# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)
n_samples = 2000
cities = [f"city_{i}" for i in range(10)]

df = pd.DataFrame({
    "city": rng.choice(cities, size=n_samples),
    "feature_a": rng.standard_normal(n_samples),
    "feature_b": rng.standard_normal(n_samples),
})
y = (
    pd.Categorical(df["city"]).codes * 0.3
    + df["feature_a"] * 0.5
    + rng.standard_normal(n_samples) * 0.2
)
y_binary = (y > y.median()).astype(int)

# LEAKAGE: target encoding computed on the FULL dataset.
# Each city's encoded value is the mean of y_binary across ALL rows,
# including test rows that should be unseen during training.
city_means = df.groupby("city").apply(
    lambda g: y_binary[g.index].mean()
)
df["city_encoded"] = df["city"].map(city_means)  # <-- TaintFlow flags this

X = df[["city_encoded", "feature_a", "feature_b"]].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42
)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

print(f"Leaky pipeline accuracy: {model.score(X_test, y_test):.3f}")
# This accuracy is inflated because city_encoded already encodes
# information about y_test.

# ---------------------------------------------------------------------------
# CORRECT approach: compute target encoding on training data only.
#
#   X_train, X_test, y_train, y_test = train_test_split(...)
#   train_df = df.iloc[train_idx]
#   city_means = train_df.groupby("city")["y"].mean()
#   train_df["city_encoded"] = train_df["city"].map(city_means)
#   test_df["city_encoded"]  = test_df["city"].map(city_means)  # uses train stats
#
# Or use sklearn's TargetEncoder inside a Pipeline (see correct_pipeline.py).
# ---------------------------------------------------------------------------
