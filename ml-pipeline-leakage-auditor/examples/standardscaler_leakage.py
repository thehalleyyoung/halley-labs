"""Example: Detecting StandardScaler leakage with TaintFlow.

This script demonstrates the most common form of ML pipeline leakage:
calling fit_transform on the entire dataset before performing the
train/test split. The scaler's mean and variance are computed using
test samples, so test-set information leaks into the training features.

Expected TaintFlow output:
    $ taintflow audit examples/standardscaler_leakage.py
    WARNING [line 24] StandardScaler.fit_transform called on full dataset
      before train_test_split at line 25.
      - Leaked columns: x0, x1, x2, x3, x4
      - Parameter taint: ⊤ (Both)
      - Channel capacity bound: 0.22 bits (GaussianChannel, d=5, n=1000, test=200)
      - Severity: LOW

    Summary: 1 leakage warning(s), 5 leaked column(s).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# BAD PIPELINE: fit_transform on full dataset before split
# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)
X = rng.standard_normal((1000, 5))
y = rng.integers(0, 2, size=1000)

# LEAKAGE: The scaler sees ALL rows (including future test rows) during fit.
# Its learned mean and std are contaminated with test-set statistics.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # <-- TaintFlow flags this line

# The split happens AFTER scaling, so X_test was already influenced by
# test-row statistics baked into the scaler parameters.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

print(f"Leaky pipeline accuracy: {model.score(X_test, y_test):.3f}")
# The accuracy looks fine, but the evaluation is optimistically biased
# because the scaler "peeked" at the test set.

# ---------------------------------------------------------------------------
# For comparison, here is the CORRECT approach (see correct_pipeline.py):
#
#   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#   scaler = StandardScaler()
#   X_train_scaled = scaler.fit_transform(X_train)   # fit on train only
#   X_test_scaled  = scaler.transform(X_test)         # transform test
# ---------------------------------------------------------------------------
