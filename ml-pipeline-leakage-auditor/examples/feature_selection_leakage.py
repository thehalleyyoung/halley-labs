#!/usr/bin/env python3
"""Example: Detecting feature selection leakage with TaintFlow.

This script demonstrates a subtle but common leakage pattern: performing
feature selection (e.g., SelectKBest, mutual information) on the entire
dataset before the train/test split.  The feature selector sees test
labels, so the selected features are biased toward the test set.

This pattern is particularly insidious because:
1. It doesn't change the data values — only which columns survive.
2. Standard code reviews rarely catch it.
3. The accuracy inflation can be substantial (5-15% on small datasets).

Expected TaintFlow output:
    $ taintflow audit examples/feature_selection_leakage.py
    WARNING [line 42] SelectKBest.fit called on full dataset before
      train_test_split at line 48.
      - Leaked columns: selected features (5 of 20)
      - Parameter taint: ⊤ (Both)
      - Channel capacity bound: 1.47 bits (feature_selection, k=5, d=20)
      - Severity: CRITICAL
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------
# Generate a dataset where most features are noise
# ---------------------------------------------------------------------------

X, y = make_classification(
    n_samples=500,
    n_features=20,
    n_informative=5,
    n_redundant=3,
    n_classes=2,
    random_state=42,
)

# ---------------------------------------------------------------------------
# BAD PIPELINE: feature selection on full dataset before split
# ---------------------------------------------------------------------------

# LEAKAGE: SelectKBest computes F-statistics using ALL rows (train + test).
# The selected features are influenced by test-set labels.
selector = SelectKBest(f_classif, k=5)
X_selected = selector.fit_transform(X, y)  # <-- TaintFlow flags this line

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.3, random_state=42
)

model = LogisticRegression(random_state=42, max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

leaky_accuracy = accuracy_score(y_test, y_pred)
print(f"Leaky pipeline accuracy:   {leaky_accuracy:.3f}")

# ---------------------------------------------------------------------------
# CORRECT PIPELINE: feature selection AFTER split, on training data only
# ---------------------------------------------------------------------------

X_train_raw, X_test_raw, y_train_c, y_test_c = train_test_split(
    X, y, test_size=0.3, random_state=42
)

selector_correct = SelectKBest(f_classif, k=5)
X_train_sel = selector_correct.fit_transform(X_train_raw, y_train_c)
X_test_sel = selector_correct.transform(X_test_raw)

model_correct = LogisticRegression(random_state=42, max_iter=200)
model_correct.fit(X_train_sel, y_train_c)
y_pred_correct = model_correct.predict(X_test_sel)

correct_accuracy = accuracy_score(y_test_c, y_pred_correct)
print(f"Correct pipeline accuracy: {correct_accuracy:.3f}")

print(f"\nAccuracy inflation from leakage: {leaky_accuracy - correct_accuracy:+.3f}")
print("(Positive value = leakage made results look artificially better)")

# ---------------------------------------------------------------------------
# TaintFlow analysis summary:
#
# In the leaky pipeline, SelectKBest.fit() computes F-statistics:
#   F_j = (between-group-variance_j) / (within-group-variance_j)
#
# These statistics use ALL rows, including test rows.  The channel capacity
# for feature selection leakage is bounded by:
#   C_select ≤ log2(C(d, k)) = log2(C(20, 5)) ≈ 14.1 bits total
#
# However, with provenance-parameterized bounds (ρ = 0.3):
#   C_select(ρ) ≈ k · I(X_test; selected | X_train) ≈ 1.47 bits
#
# This is enough to inflate accuracy by several percentage points.
# ---------------------------------------------------------------------------
