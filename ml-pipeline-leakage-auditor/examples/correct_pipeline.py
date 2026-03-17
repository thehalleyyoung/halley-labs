"""Example: A correct sklearn Pipeline with no leakage.

This script demonstrates the proper way to build an ML pipeline that
TaintFlow verifies as clean. By using sklearn's Pipeline object, all
fit operations are scoped to the training fold automatically.

Expected TaintFlow output:
    $ taintflow audit examples/correct_pipeline.py
    ✓ No leakage detected.

    Pipeline summary:
      - Nodes: 4 (StandardScaler, PCA, LogisticRegression, train_test_split)
      - Columns tracked: 20
      - All parameter taints: Train
      - Channel capacity: 0.00 bits

    Result: CLEAN
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Generate synthetic data
# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)
n_samples = 1000
n_features = 20

X = rng.standard_normal((n_samples, n_features))
true_weights = rng.standard_normal(n_features)
y = (X @ true_weights + rng.standard_normal(n_samples) * 2 > 0).astype(int)

# ---------------------------------------------------------------------------
# CORRECT: split FIRST, then build a Pipeline that fits only on training data
# ---------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# sklearn's Pipeline ensures that:
#   1. scaler.fit() sees only X_train
#   2. pca.fit() sees only the scaled X_train
#   3. classifier.fit() sees only the PCA-transformed X_train
# When calling pipeline.predict(X_test), each step calls only .transform()
# (not .fit_transform()), so test data never contaminates parameters.
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=10)),
    ("classifier", LogisticRegression(random_state=42, max_iter=200)),
])

pipeline.fit(X_train, y_train)

# Evaluate on the held-out test set
y_pred = pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {test_accuracy:.3f}")
print()
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------------------------
# ALSO CORRECT: cross-validation with Pipeline
# ---------------------------------------------------------------------------
# cross_val_score handles splitting internally; the Pipeline ensures
# fit is always scoped to the training fold.
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
print(f"Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ---------------------------------------------------------------------------
# Why this is safe (TaintFlow analysis):
#
# 1. train_test_split assigns:
#      X_train row_taint = Train
#      X_test  row_taint = Test
#
# 2. pipeline.fit(X_train, y_train):
#      scaler params      → taint = Train  (fit on Train rows only)
#      pca params          → taint = Train  (fit on Train rows only)
#      classifier params   → taint = Train  (fit on Train rows only)
#
# 3. pipeline.predict(X_test):
#      scaler.transform(X_test)    → param_taint=Train, row_taint=Test → OK
#      pca.transform(...)          → param_taint=Train, row_taint=Test → OK
#      classifier.predict(...)     → param_taint=Train, row_taint=Test → OK
#
# No parameter taint reaches ⊤, so TaintFlow reports: CLEAN.
# ---------------------------------------------------------------------------
