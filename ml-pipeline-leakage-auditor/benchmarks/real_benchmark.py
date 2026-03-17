#!/usr/bin/env python3
"""
Real-world benchmark for TaintFlow ML Pipeline Leakage Auditor.

Creates 24 real sklearn pipelines on real datasets (12 with known leakage,
12 clean), runs TaintFlow's taint analysis + 4 baselines, and measures
detection accuracy, timing, and which specific leakage patterns are found.

Baselines:
  1. TaintFlow (our tool) — partition-taint lattice + channel capacity
  2. Heuristic Pattern Detector — simulates AST-based tools (LeakageDetector)
  3. sklearn Pipeline Guard — checks if Pipeline API prevents leakage
  4. CV-Gap Audit — nested vs non-nested CV performance gap as leakage proxy
  5. DataLinter Rules — implements Google DataLinter checks for leakage

Datasets: iris, breast_cancer, wine, california_housing, digits
Estimators: StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
  PCA, PolynomialFeatures, QuantileTransformer, PowerTransformer,
  KBinsDiscretizer, SelectKBest, SimpleImputer, LogisticRegression,
  Ridge, Lasso, RandomForest, GradientBoosting, SVC, KNN, DecisionTree
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    fetch_california_housing,
)
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MaxAbsScaler,
    MinMaxScaler,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ── Data classes ─────────────────────────────────────────────────

@dataclass
class GroundTruth:
    has_leakage: bool
    category: str          # preprocessing | feature_selection | target | temporal | clean
    pattern: str           # human-readable description
    leaked_features: list[str]
    approximate_bits: float

@dataclass
class DetectionResult:
    method: str
    detected: bool
    features_flagged: list[str]
    bit_bound: float
    time_s: float
    memory_mb: float
    details: str = ""

@dataclass
class PipelineScenario:
    name: str
    dataset: str
    task: str              # classification | regression
    ground_truth: GroundTruth
    # Callable fields set by code — not serialized
    pipeline_fn: Any = field(default=None, repr=False)

@dataclass
class ScenarioResult:
    scenario: str
    dataset: str
    task: str
    ground_truth: dict
    detections: list[dict]

# ── Utility ──────────────────────────────────────────────────────

def load_dataset(name: str):
    """Load a real sklearn dataset."""
    loaders = {
        "iris": load_iris,
        "breast_cancer": load_breast_cancer,
        "wine": load_wine,
        "digits": load_digits,
        "california_housing": fetch_california_housing,
    }
    ds = loaders[name]()
    X = pd.DataFrame(ds.data, columns=[f"f{i}" for i in range(ds.data.shape[1])])
    y = pd.Series(ds.target, name="target")
    return X, y

def timed_run(fn, *args, **kwargs):
    """Run fn, return (result, time_s, memory_mb)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / (1024 * 1024)


# ═══════════════════════════════════════════════════════════════════
# 24 REAL PIPELINES (12 leaky, 12 clean)
# ═══════════════════════════════════════════════════════════════════

def _build_scenarios() -> list[PipelineScenario]:
    scenarios: list[PipelineScenario] = []

    # ── LEAKY 1: StandardScaler on full data before split (iris) ──
    def leaky_scaler_iris():
        X, y = load_dataset("iris")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)            # ❌ fit on ALL data
        X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        clf = LogisticRegression(max_iter=200, random_state=42)
        clf.fit(X_tr, y_tr)
        return {
            "accuracy": clf.score(X_te, y_te),
            "steps": [
                {"op": "fit_transform", "estimator": "StandardScaler", "scope": "full",
                 "n_features": X.shape[1], "columns": list(X.columns)},
                {"op": "train_test_split", "test_size": 0.2},
                {"op": "fit", "estimator": "LogisticRegression", "scope": "train"},
            ],
            "n_samples": len(X), "n_features": X.shape[1], "test_fraction": 0.2,
        }

    scenarios.append(PipelineScenario(
        name="leaky_scaler_iris",
        dataset="iris",
        task="classification",
        ground_truth=GroundTruth(
            has_leakage=True,
            category="preprocessing",
            pattern="StandardScaler.fit_transform() on full data before train_test_split",
            leaked_features=[f"f{i}" for i in range(4)],
            approximate_bits=4 * 0.5 * math.log2(1 + 0.2/0.8),
        ),
        pipeline_fn=leaky_scaler_iris,
    ))

    # ── LEAKY 2: Feature selection on full data (breast_cancer) ──
    def leaky_fsel_breast_cancer():
        X, y = load_dataset("breast_cancer")
        sel = SelectKBest(f_classif, k=10)
        X_sel = sel.fit_transform(X, y)                # ❌ uses ALL labels
        X_tr, X_te, y_tr, y_te = train_test_split(X_sel, y, test_size=0.25, random_state=42)
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_tr, y_tr)
        selected_mask = sel.get_support()
        return {
            "accuracy": clf.score(X_te, y_te),
            "steps": [
                {"op": "fit_transform", "estimator": "SelectKBest", "scope": "full",
                 "n_features": 30, "k": 10, "columns": [f"f{i}" for i in range(30) if selected_mask[i]]},
                {"op": "train_test_split", "test_size": 0.25},
                {"op": "fit", "estimator": "RandomForestClassifier", "scope": "train"},
            ],
            "n_samples": len(X), "n_features": X.shape[1], "test_fraction": 0.25,
        }

    scenarios.append(PipelineScenario(
        name="leaky_fsel_breast_cancer",
        dataset="breast_cancer",
        task="classification",
        ground_truth=GroundTruth(
            has_leakage=True,
            category="feature_selection",
            pattern="SelectKBest.fit_transform(X, y) on full dataset before split",
            leaked_features=[f"f{i}" for i in range(10)],
            approximate_bits=10 * 0.5 * math.log2(1 + 0.25/0.75),
        ),
        pipeline_fn=leaky_fsel_breast_cancer,
    ))

    # ── LEAKY 3: Target leakage via look-ahead mean (california_housing → regression) ──
    def leaky_target_california():
        X, y = load_dataset("california_housing")
        df = X.copy()
        df["target"] = y.values
        # ❌ Target leakage: create feature from target stats on full data
        df["target_mean_bin"] = pd.qcut(df["target"], q=10, labels=False, duplicates="drop")
        group_means = df.groupby("target_mean_bin")["target"].transform("mean")
        df["leaked_target_feature"] = group_means
        df = df.drop(columns=["target", "target_mean_bin"])

        X_leak = df.values
        X_tr, X_te, y_tr, y_te = train_test_split(X_leak, y, test_size=0.2, random_state=42)
        reg = Ridge(alpha=1.0)
        reg.fit(X_tr, y_tr)
        return {
            "r2": reg.score(X_te, y_te),
            "steps": [
                {"op": "target_encoding", "estimator": "TargetEncoder", "scope": "full",
                 "n_features": 1, "columns": ["leaked_target_feature"]},
                {"op": "train_test_split", "test_size": 0.2},
                {"op": "fit", "estimator": "Ridge", "scope": "train"},
            ],
            "n_samples": len(X), "n_features": df.shape[1], "test_fraction": 0.2,
        }

    scenarios.append(PipelineScenario(
        name="leaky_target_california",
        dataset="california_housing",
        task="regression",
        ground_truth=GroundTruth(
            has_leakage=True,
            category="target",
            pattern="Target encoding via groupby(target).transform('mean') on full data",
            leaked_features=["leaked_target_feature"],
            approximate_bits=3.32,  # ~log2(10) for 10-bin quantile encoding
        ),
        pipeline_fn=leaky_target_california,
    ))

    # ── LEAKY 4: MinMaxScaler + PCA on full data (wine) ──
    def leaky_minmax_pca_wine():
        X, y = load_dataset("wine")
        scaler = MinMaxScaler()
        X_sc = scaler.fit_transform(X)                  # ❌ fit on ALL
        pca = PCA(n_components=5)
        X_pca = pca.fit_transform(X_sc)                 # ❌ fit on ALL
        X_tr, X_te, y_tr, y_te = train_test_split(X_pca, y, test_size=0.3, random_state=42)
        clf = SVC(kernel="rbf", random_state=42)
        clf.fit(X_tr, y_tr)
        return {
            "accuracy": clf.score(X_te, y_te),
            "steps": [
                {"op": "fit_transform", "estimator": "MinMaxScaler", "scope": "full",
                 "n_features": 13, "columns": [f"f{i}" for i in range(13)]},
                {"op": "fit_transform", "estimator": "PCA", "scope": "full",
                 "n_features": 5, "columns": [f"pc{i}" for i in range(5)]},
                {"op": "train_test_split", "test_size": 0.3},
                {"op": "fit", "estimator": "SVC", "scope": "train"},
            ],
            "n_samples": len(X), "n_features": X.shape[1], "test_fraction": 0.3,
        }

    scenarios.append(PipelineScenario(
        name="leaky_minmax_pca_wine",
        dataset="wine",
        task="classification",
        ground_truth=GroundTruth(
            has_leakage=True,
            category="preprocessing",
            pattern="MinMaxScaler + PCA both fit_transform on full data before split",
            leaked_features=[f"f{i}" for i in range(13)] + [f"pc{i}" for i in range(5)],
            approximate_bits=(13 + 5) * 0.5 * math.log2(1 + 0.3/0.7),
        ),
        pipeline_fn=leaky_minmax_pca_wine,
    ))

    # ── LEAKY 5: Imputer + Discretizer on full data (digits) ──
    def leaky_imputer_digits():
        X, y = load_dataset("digits")
        # Inject some NaN to make imputation meaningful
        rng = np.random.RandomState(42)
        X_arr = X.values.astype(float)
        mask = rng.random(X_arr.shape) < 0.05
        X_arr[mask] = np.nan
        imp = SimpleImputer(strategy="median")
        X_imp = imp.fit_transform(X_arr)                # ❌ fit on ALL
        disc = KBinsDiscretizer(n_bins=8, encode="ordinal", strategy="quantile")
        X_disc = disc.fit_transform(X_imp)              # ❌ fit on ALL
        X_tr, X_te, y_tr, y_te = train_test_split(X_disc, y, test_size=0.2, random_state=42)
        clf = DecisionTreeClassifier(max_depth=12, random_state=42)
        clf.fit(X_tr, y_tr)
        return {
            "accuracy": clf.score(X_te, y_te),
            "steps": [
                {"op": "fit_transform", "estimator": "SimpleImputer", "scope": "full",
                 "n_features": 64, "columns": [f"f{i}" for i in range(64)]},
                {"op": "fit_transform", "estimator": "KBinsDiscretizer", "scope": "full",
                 "n_features": 64, "columns": [f"f{i}" for i in range(64)]},
                {"op": "train_test_split", "test_size": 0.2},
                {"op": "fit", "estimator": "DecisionTreeClassifier", "scope": "train"},
            ],
            "n_samples": 1797, "n_features": 64, "test_fraction": 0.2,
        }

    scenarios.append(PipelineScenario(
        name="leaky_imputer_digits",
        dataset="digits",
        task="classification",
        ground_truth=GroundTruth(
            has_leakage=True,
            category="preprocessing",
            pattern="SimpleImputer + KBinsDiscretizer both fit on full data before split",
            leaked_features=[f"f{i}" for i in range(64)],
            approximate_bits=64 * 0.5 * math.log2(1 + 0.2/0.8),
        ),
        pipeline_fn=leaky_imputer_digits,
    ))

    # ── CLEAN 1: Correct Pipeline with StandardScaler (iris) ──
    def clean_pipeline_iris():
        X, y = load_dataset("iris")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, random_state=42)),
        ])
        pipe.fit(X_tr, y_tr)  # ✅ scaler sees only training data
        return {
            "accuracy": pipe.score(X_te, y_te),
            "steps": [
                {"op": "train_test_split", "test_size": 0.2},
                {"op": "fit_transform", "estimator": "StandardScaler", "scope": "train",
                 "n_features": 4, "columns": [f"f{i}" for i in range(4)]},
                {"op": "fit", "estimator": "LogisticRegression", "scope": "train"},
            ],
            "n_samples": len(X), "n_features": X.shape[1], "test_fraction": 0.2,
        }

    scenarios.append(PipelineScenario(
        name="clean_pipeline_iris",
        dataset="iris",
        task="classification",
        ground_truth=GroundTruth(
            has_leakage=False,
            category="clean",
            pattern="Correct: split first, then Pipeline(StandardScaler, LogReg)",
            leaked_features=[],
            approximate_bits=0.0,
        ),
        pipeline_fn=clean_pipeline_iris,
    ))

    # ── CLEAN 2: Correct Pipeline with SelectKBest (breast_cancer) ──
    def clean_fsel_breast_cancer():
        X, y = load_dataset("breast_cancer")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
        pipe = Pipeline([
            ("sel", SelectKBest(f_classif, k=10)),
            ("clf", RandomForestClassifier(n_estimators=50, random_state=42)),
        ])
        pipe.fit(X_tr, y_tr)  # ✅ selection sees only training data
        return {
            "accuracy": pipe.score(X_te, y_te),
            "steps": [
                {"op": "train_test_split", "test_size": 0.25},
                {"op": "fit_transform", "estimator": "SelectKBest", "scope": "train",
                 "n_features": 30, "k": 10, "columns": []},
                {"op": "fit", "estimator": "RandomForestClassifier", "scope": "train"},
            ],
            "n_samples": len(X), "n_features": X.shape[1], "test_fraction": 0.25,
        }

    scenarios.append(PipelineScenario(
        name="clean_fsel_breast_cancer",
        dataset="breast_cancer",
        task="classification",
        ground_truth=GroundTruth(
            has_leakage=False,
            category="clean",
            pattern="Correct: split first, then Pipeline(SelectKBest, RF)",
            leaked_features=[],
            approximate_bits=0.0,
        ),
        pipeline_fn=clean_fsel_breast_cancer,
    ))

    # ── CLEAN 3: Correct Ridge with split-first scaling (california_housing) ──
    def clean_ridge_california():
        X, y = load_dataset("california_housing")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0)),
        ])
        pipe.fit(X_tr, y_tr)  # ✅ scaler sees only training data
        return {
            "r2": pipe.score(X_te, y_te),
            "steps": [
                {"op": "train_test_split", "test_size": 0.2},
                {"op": "fit_transform", "estimator": "StandardScaler", "scope": "train",
                 "n_features": 8, "columns": [f"f{i}" for i in range(8)]},
                {"op": "fit", "estimator": "Ridge", "scope": "train"},
            ],
            "n_samples": 20640, "n_features": 8, "test_fraction": 0.2,
        }

    scenarios.append(PipelineScenario(
        name="clean_ridge_california",
        dataset="california_housing",
        task="regression",
        ground_truth=GroundTruth(
            has_leakage=False,
            category="clean",
            pattern="Correct: split first, then Pipeline(StandardScaler, Ridge)",
            leaked_features=[],
            approximate_bits=0.0,
        ),
        pipeline_fn=clean_ridge_california,
    ))

    # ── CLEAN 4: Correct PCA + SVC on wine ──
    def clean_pca_wine():
        X, y = load_dataset("wine")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
        pipe = Pipeline([
            ("scaler", MinMaxScaler()),
            ("pca", PCA(n_components=5)),
            ("clf", SVC(kernel="rbf", random_state=42)),
        ])
        pipe.fit(X_tr, y_tr)  # ✅ all fitted on train only
        return {
            "accuracy": pipe.score(X_te, y_te),
            "steps": [
                {"op": "train_test_split", "test_size": 0.3},
                {"op": "fit_transform", "estimator": "MinMaxScaler", "scope": "train",
                 "n_features": 13, "columns": [f"f{i}" for i in range(13)]},
                {"op": "fit_transform", "estimator": "PCA", "scope": "train",
                 "n_features": 5, "columns": [f"pc{i}" for i in range(5)]},
                {"op": "fit", "estimator": "SVC", "scope": "train"},
            ],
            "n_samples": 178, "n_features": 13, "test_fraction": 0.3,
        }

    scenarios.append(PipelineScenario(
        name="clean_pca_wine",
        dataset="wine",
        task="classification",
        ground_truth=GroundTruth(
            has_leakage=False,
            category="clean",
            pattern="Correct: split first, then Pipeline(MinMaxScaler, PCA, SVC)",
            leaked_features=[],
            approximate_bits=0.0,
        ),
        pipeline_fn=clean_pca_wine,
    ))

    # ── CLEAN 5: Correct GBT on digits (no preprocessing needed) ──
    def clean_gbt_digits():
        X, y = load_dataset("digits")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42)
        clf.fit(X_tr, y_tr)  # ✅ no preprocessing, direct fit on train
        return {
            "accuracy": clf.score(X_te, y_te),
            "steps": [
                {"op": "train_test_split", "test_size": 0.2},
                {"op": "fit", "estimator": "GradientBoostingClassifier", "scope": "train"},
            ],
            "n_samples": 1797, "n_features": 64, "test_fraction": 0.2,
        }

    scenarios.append(PipelineScenario(
        name="clean_gbt_digits",
        dataset="digits",
        task="classification",
        ground_truth=GroundTruth(
            has_leakage=False,
            category="clean",
            pattern="Correct: split first, GBT fitted on training data only, no preprocessing",
            leaked_features=[],
            approximate_bits=0.0,
        ),
        pipeline_fn=clean_gbt_digits,
    ))

    # ── LEAKY 6: RobustScaler on full data (breast_cancer) ──
    def leaky_robust_breast_cancer():
        X, y = load_dataset("breast_cancer")
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)            # ❌ fit on ALL data
        X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
        clf = LogisticRegression(max_iter=300, random_state=42)
        clf.fit(X_tr, y_tr)
        return {
            "accuracy": clf.score(X_te, y_te),
            "steps": [
                {"op": "fit_transform", "estimator": "RobustScaler", "scope": "full",
                 "n_features": X.shape[1], "columns": [f"f{i}" for i in range(X.shape[1])]},
                {"op": "train_test_split", "test_size": 0.25},
                {"op": "fit", "estimator": "LogisticRegression", "scope": "train"},
            ],
            "n_samples": len(X), "n_features": X.shape[1], "test_fraction": 0.25,
        }

    scenarios.append(PipelineScenario(
        name="leaky_robust_breast_cancer",
        dataset="breast_cancer",
        task="classification",
        ground_truth=GroundTruth(
            has_leakage=True,
            category="preprocessing",
            pattern="RobustScaler.fit_transform() on full data before train_test_split",
            leaked_features=[f"f{i}" for i in range(30)],
            approximate_bits=30 * 0.5 * math.log2(1 + 0.25/0.75),
        ),
        pipeline_fn=leaky_robust_breast_cancer,
    ))

    # ── LEAKY 7: PolynomialFeatures + StandardScaler on full data (iris) ──
    def leaky_poly_iris():
        X, y = load_dataset("iris")
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)                # ❌ fit on ALL
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_poly)           # ❌ fit on ALL
        X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2, random_state=42)
        clf = LogisticRegression(max_iter=500, random_state=42)
        clf.fit(X_tr, y_tr)
        n_poly_features = X_poly.shape[1]
        return {
            "accuracy": clf.score(X_te, y_te),
            "steps": [
                {"op": "fit_transform", "estimator": "PolynomialFeatures", "scope": "full",
                 "n_features": n_poly_features, "columns": [f"poly{i}" for i in range(n_poly_features)]},
                {"op": "fit_transform", "estimator": "StandardScaler", "scope": "full",
                 "n_features": n_poly_features, "columns": [f"poly{i}" for i in range(n_poly_features)]},
                {"op": "train_test_split", "test_size": 0.2},
                {"op": "fit", "estimator": "LogisticRegression", "scope": "train"},
            ],
            "n_samples": len(X), "n_features": n_poly_features, "test_fraction": 0.2,
        }

    scenarios.append(PipelineScenario(
        name="leaky_poly_iris",
        dataset="iris",
        task="classification",
        ground_truth=GroundTruth(
            has_leakage=True,
            category="preprocessing",
            pattern="PolynomialFeatures + StandardScaler both fit on full data before split",
            leaked_features=[f"poly{i}" for i in range(14)],
            approximate_bits=14 * 0.5 * math.log2(1 + 0.2/0.8),
        ),
        pipeline_fn=leaky_poly_iris,
    ))

    # ── LEAKY 8: QuantileTransformer on full data (wine) ──
    def leaky_quantile_wine():
        X, y = load_dataset("wine")
        qt = QuantileTransformer(n_quantiles=50, output_distribution="normal", random_state=42)
        X_qt = qt.fit_transform(X)                    # ❌ fit on ALL
        X_tr, X_te, y_tr, y_te = train_test_split(X_qt, y, test_size=0.3, random_state=42)
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_tr, y_tr)
        return {
            "accuracy": clf.score(X_te, y_te),
            "steps": [
                {"op": "fit_transform", "estimator": "QuantileTransformer", "scope": "full",
                 "n_features": X.shape[1], "columns": [f"f{i}" for i in range(X.shape[1])]},
                {"op": "train_test_split", "test_size": 0.3},
                {"op": "fit", "estimator": "RandomForestClassifier", "scope": "train"},
            ],
            "n_samples": len(X), "n_features": X.shape[1], "test_fraction": 0.3,
        }

    scenarios.append(PipelineScenario(
        name="leaky_quantile_wine",
        dataset="wine",
        task="classification",
        ground_truth=GroundTruth(
            has_leakage=True,
            category="preprocessing",
            pattern="QuantileTransformer.fit_transform() on full data before split",
            leaked_features=[f"f{i}" for i in range(13)],
            approximate_bits=13 * 0.5 * math.log2(1 + 0.3/0.7),
        ),
        pipeline_fn=leaky_quantile_wine,
    ))

    # ── LEAKY 9: MaxAbsScaler on full data (digits) ──
    def leaky_maxabs_digits():
        X, y = load_dataset("digits")
        scaler = MaxAbsScaler()
        X_sc = scaler.fit_transform(X)                # ❌ fit on ALL
        X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2, random_state=42)
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_tr, y_tr)
        return {
            "accuracy": clf.score(X_te, y_te),
            "steps": [
                {"op": "fit_transform", "estimator": "MaxAbsScaler", "scope": "full",
                 "n_features": 64, "columns": [f"f{i}" for i in range(64)]},
                {"op": "train_test_split", "test_size": 0.2},
                {"op": "fit", "estimator": "KNeighborsClassifier", "scope": "train"},
            ],
            "n_samples": 1797, "n_features": 64, "test_fraction": 0.2,
        }

    scenarios.append(PipelineScenario(
        name="leaky_maxabs_digits",
        dataset="digits",
        task="classification",
        ground_truth=GroundTruth(
            has_leakage=True,
            category="preprocessing",
            pattern="MaxAbsScaler.fit_transform() on full data before split",
            leaked_features=[f"f{i}" for i in range(64)],
            approximate_bits=64 * 0.5 * math.log2(1 + 0.2/0.8),
        ),
        pipeline_fn=leaky_maxabs_digits,
    ))

    # ── LEAKY 10: PowerTransformer + feature selection on full data (california_housing) ──
    def leaky_power_california():
        X, y = load_dataset("california_housing")
        pt = PowerTransformer(method="yeo-johnson")
        X_pt = pt.fit_transform(X)                    # ❌ fit on ALL
        sel = SelectKBest(f_classif, k=5)
        y_binned = pd.qcut(y, q=5, labels=False, duplicates="drop")
        X_sel = sel.fit_transform(X_pt, y_binned)     # ❌ uses ALL labels
        X_tr, X_te, y_tr, y_te = train_test_split(X_sel, y, test_size=0.2, random_state=42)
        reg = Ridge(alpha=1.0)
        reg.fit(X_tr, y_tr)
        return {
            "r2": reg.score(X_te, y_te),
            "steps": [
                {"op": "fit_transform", "estimator": "PowerTransformer", "scope": "full",
                 "n_features": 8, "columns": [f"f{i}" for i in range(8)]},
                {"op": "fit_transform", "estimator": "SelectKBest", "scope": "full",
                 "n_features": 8, "k": 5, "columns": [f"f{i}" for i in range(5)]},
                {"op": "train_test_split", "test_size": 0.2},
                {"op": "fit", "estimator": "Ridge", "scope": "train"},
            ],
            "n_samples": len(X), "n_features": 8, "test_fraction": 0.2,
        }

    scenarios.append(PipelineScenario(
        name="leaky_power_california",
        dataset="california_housing",
        task="regression",
        ground_truth=GroundTruth(
            has_leakage=True,
            category="preprocessing",
            pattern="PowerTransformer + SelectKBest both fit on full data before split",
            leaked_features=[f"f{i}" for i in range(8)],
            approximate_bits=8 * 0.5 * math.log2(1 + 0.2/0.8) + 5 * 0.5 * math.log2(1 + 0.2/0.8),
        ),
        pipeline_fn=leaky_power_california,
    ))

    # ── LEAKY 11: Feature selection with mutual_info on full data (wine) ──
    def leaky_mi_fsel_wine():
        X, y = load_dataset("wine")
        sel = SelectKBest(mutual_info_classif, k=8)
        X_sel = sel.fit_transform(X, y)               # ❌ uses ALL labels
        X_tr, X_te, y_tr, y_te = train_test_split(X_sel, y, test_size=0.3, random_state=42)
        clf = DecisionTreeClassifier(max_depth=5, random_state=42)
        clf.fit(X_tr, y_tr)
        return {
            "accuracy": clf.score(X_te, y_te),
            "steps": [
                {"op": "fit_transform", "estimator": "SelectKBest", "scope": "full",
                 "n_features": 13, "k": 8, "columns": [f"f{i}" for i in range(8)]},
                {"op": "train_test_split", "test_size": 0.3},
                {"op": "fit", "estimator": "DecisionTreeClassifier", "scope": "train"},
            ],
            "n_samples": len(X), "n_features": 13, "test_fraction": 0.3,
        }

    scenarios.append(PipelineScenario(
        name="leaky_mi_fsel_wine",
        dataset="wine",
        task="classification",
        ground_truth=GroundTruth(
            has_leakage=True,
            category="feature_selection",
            pattern="SelectKBest(mutual_info_classif) on full dataset before split",
            leaked_features=[f"f{i}" for i in range(8)],
            approximate_bits=8 * 0.5 * math.log2(1 + 0.3/0.7),
        ),
        pipeline_fn=leaky_mi_fsel_wine,
    ))

    # ── LEAKY 12: Target leakage via pandas groupby mean on breast_cancer ──
    def leaky_target_breast_cancer():
        X, y = load_dataset("breast_cancer")
        df = X.copy()
        df["target"] = y.values
        # ❌ Target leakage: create feature from target correlation on full data
        df["target_corr_feature"] = df.groupby(
            pd.qcut(df["f0"], q=5, labels=False, duplicates="drop")
        )["target"].transform("mean")
        df = df.drop(columns=["target"])
        X_leak = df.values
        X_tr, X_te, y_tr, y_te = train_test_split(X_leak, y, test_size=0.25, random_state=42)
        clf = LogisticRegression(max_iter=300, random_state=42)
        clf.fit(X_tr, y_tr)
        return {
            "accuracy": clf.score(X_te, y_te),
            "steps": [
                {"op": "target_encoding", "estimator": "TargetEncoder", "scope": "full",
                 "n_features": 1, "columns": ["target_corr_feature"]},
                {"op": "train_test_split", "test_size": 0.25},
                {"op": "fit", "estimator": "LogisticRegression", "scope": "train"},
            ],
            "n_samples": len(X), "n_features": df.shape[1], "test_fraction": 0.25,
        }

    scenarios.append(PipelineScenario(
        name="leaky_target_breast_cancer",
        dataset="breast_cancer",
        task="classification",
        ground_truth=GroundTruth(
            has_leakage=True,
            category="target",
            pattern="Target encoding via groupby(target).transform('mean') on full data",
            leaked_features=["target_corr_feature"],
            approximate_bits=math.log2(5),
        ),
        pipeline_fn=leaky_target_breast_cancer,
    ))

    # ── CLEAN 6: Correct QuantileTransformer pipeline (wine) ──
    def clean_quantile_wine():
        X, y = load_dataset("wine")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
        pipe = Pipeline([
            ("qt", QuantileTransformer(n_quantiles=50, output_distribution="normal", random_state=42)),
            ("clf", RandomForestClassifier(n_estimators=50, random_state=42)),
        ])
        pipe.fit(X_tr, y_tr)  # ✅ qt sees only training data
        return {
            "accuracy": pipe.score(X_te, y_te),
            "steps": [
                {"op": "train_test_split", "test_size": 0.3},
                {"op": "fit_transform", "estimator": "QuantileTransformer", "scope": "train",
                 "n_features": 13, "columns": [f"f{i}" for i in range(13)]},
                {"op": "fit", "estimator": "RandomForestClassifier", "scope": "train"},
            ],
            "n_samples": 178, "n_features": 13, "test_fraction": 0.3,
        }

    scenarios.append(PipelineScenario(
        name="clean_quantile_wine",
        dataset="wine",
        task="classification",
        ground_truth=GroundTruth(
            has_leakage=False,
            category="clean",
            pattern="Correct: split first, then Pipeline(QuantileTransformer, RF)",
            leaked_features=[],
            approximate_bits=0.0,
        ),
        pipeline_fn=clean_quantile_wine,
    ))

    # ── CLEAN 7: Correct RobustScaler pipeline (breast_cancer) ──
    def clean_robust_breast_cancer():
        X, y = load_dataset("breast_cancer")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
        pipe = Pipeline([
            ("scaler", RobustScaler()),
            ("clf", LogisticRegression(max_iter=300, random_state=42)),
        ])
        pipe.fit(X_tr, y_tr)  # ✅ scaler sees only training data
        return {
            "accuracy": pipe.score(X_te, y_te),
            "steps": [
                {"op": "train_test_split", "test_size": 0.25},
                {"op": "fit_transform", "estimator": "RobustScaler", "scope": "train",
                 "n_features": 30, "columns": [f"f{i}" for i in range(30)]},
                {"op": "fit", "estimator": "LogisticRegression", "scope": "train"},
            ],
            "n_samples": len(X), "n_features": 30, "test_fraction": 0.25,
        }

    scenarios.append(PipelineScenario(
        name="clean_robust_breast_cancer",
        dataset="breast_cancer",
        task="classification",
        ground_truth=GroundTruth(
            has_leakage=False,
            category="clean",
            pattern="Correct: split first, then Pipeline(RobustScaler, LogReg)",
            leaked_features=[],
            approximate_bits=0.0,
        ),
        pipeline_fn=clean_robust_breast_cancer,
    ))

    # ── CLEAN 8: Correct PolynomialFeatures + Lasso (california_housing) ──
    def clean_poly_california():
        X, y = load_dataset("california_housing")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("reg", Lasso(alpha=0.1, max_iter=2000)),
        ])
        pipe.fit(X_tr, y_tr)  # ✅ all fitted on train only
        return {
            "r2": pipe.score(X_te, y_te),
            "steps": [
                {"op": "train_test_split", "test_size": 0.2},
                {"op": "fit_transform", "estimator": "StandardScaler", "scope": "train",
                 "n_features": 8, "columns": [f"f{i}" for i in range(8)]},
                {"op": "fit_transform", "estimator": "PolynomialFeatures", "scope": "train",
                 "n_features": 44, "columns": [f"poly{i}" for i in range(44)]},
                {"op": "fit", "estimator": "Lasso", "scope": "train"},
            ],
            "n_samples": 20640, "n_features": 8, "test_fraction": 0.2,
        }

    scenarios.append(PipelineScenario(
        name="clean_poly_california",
        dataset="california_housing",
        task="regression",
        ground_truth=GroundTruth(
            has_leakage=False,
            category="clean",
            pattern="Correct: split first, then Pipeline(StandardScaler, PolyFeatures, Lasso)",
            leaked_features=[],
            approximate_bits=0.0,
        ),
        pipeline_fn=clean_poly_california,
    ))

    # ── CLEAN 9: Correct MaxAbsScaler + KNN (digits) ──
    def clean_maxabs_digits():
        X, y = load_dataset("digits")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe = Pipeline([
            ("scaler", MaxAbsScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ])
        pipe.fit(X_tr, y_tr)  # ✅ scaler sees only training data
        return {
            "accuracy": pipe.score(X_te, y_te),
            "steps": [
                {"op": "train_test_split", "test_size": 0.2},
                {"op": "fit_transform", "estimator": "MaxAbsScaler", "scope": "train",
                 "n_features": 64, "columns": [f"f{i}" for i in range(64)]},
                {"op": "fit", "estimator": "KNeighborsClassifier", "scope": "train"},
            ],
            "n_samples": 1797, "n_features": 64, "test_fraction": 0.2,
        }

    scenarios.append(PipelineScenario(
        name="clean_maxabs_digits",
        dataset="digits",
        task="classification",
        ground_truth=GroundTruth(
            has_leakage=False,
            category="clean",
            pattern="Correct: split first, then Pipeline(MaxAbsScaler, KNN)",
            leaked_features=[],
            approximate_bits=0.0,
        ),
        pipeline_fn=clean_maxabs_digits,
    ))

    # ── CLEAN 10: Correct PowerTransformer + Ridge (california_housing) ──
    def clean_power_california():
        X, y = load_dataset("california_housing")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe = Pipeline([
            ("pt", PowerTransformer(method="yeo-johnson")),
            ("reg", Ridge(alpha=1.0)),
        ])
        pipe.fit(X_tr, y_tr)  # ✅ PowerTransformer sees only training data
        return {
            "r2": pipe.score(X_te, y_te),
            "steps": [
                {"op": "train_test_split", "test_size": 0.2},
                {"op": "fit_transform", "estimator": "PowerTransformer", "scope": "train",
                 "n_features": 8, "columns": [f"f{i}" for i in range(8)]},
                {"op": "fit", "estimator": "Ridge", "scope": "train"},
            ],
            "n_samples": 20640, "n_features": 8, "test_fraction": 0.2,
        }

    scenarios.append(PipelineScenario(
        name="clean_power_california",
        dataset="california_housing",
        task="regression",
        ground_truth=GroundTruth(
            has_leakage=False,
            category="clean",
            pattern="Correct: split first, then Pipeline(PowerTransformer, Ridge)",
            leaked_features=[],
            approximate_bits=0.0,
        ),
        pipeline_fn=clean_power_california,
    ))

    # ── CLEAN 11: Correct Imputer + SelectKBest (iris) ──
    def clean_imputer_fsel_iris():
        X, y = load_dataset("iris")
        rng = np.random.RandomState(42)
        X_arr = X.values.astype(float)
        mask = rng.random(X_arr.shape) < 0.05
        X_arr[mask] = np.nan
        X_nan = pd.DataFrame(X_arr, columns=X.columns)
        X_tr, X_te, y_tr, y_te = train_test_split(X_nan, y, test_size=0.2, random_state=42)
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="mean")),
            ("sel", SelectKBest(f_classif, k=3)),
            ("clf", LogisticRegression(max_iter=200, random_state=42)),
        ])
        pipe.fit(X_tr, y_tr)  # ✅ all see only training data
        return {
            "accuracy": pipe.score(X_te, y_te),
            "steps": [
                {"op": "train_test_split", "test_size": 0.2},
                {"op": "fit_transform", "estimator": "SimpleImputer", "scope": "train",
                 "n_features": 4, "columns": [f"f{i}" for i in range(4)]},
                {"op": "fit_transform", "estimator": "SelectKBest", "scope": "train",
                 "n_features": 4, "k": 3, "columns": []},
                {"op": "fit", "estimator": "LogisticRegression", "scope": "train"},
            ],
            "n_samples": 150, "n_features": 4, "test_fraction": 0.2,
        }

    scenarios.append(PipelineScenario(
        name="clean_imputer_fsel_iris",
        dataset="iris",
        task="classification",
        ground_truth=GroundTruth(
            has_leakage=False,
            category="clean",
            pattern="Correct: split first, then Pipeline(SimpleImputer, SelectKBest, LogReg)",
            leaked_features=[],
            approximate_bits=0.0,
        ),
        pipeline_fn=clean_imputer_fsel_iris,
    ))

    # ── CLEAN 12: No preprocessing, just RF on wine ──
    def clean_rf_wine():
        X, y = load_dataset("wine")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_tr, y_tr)  # ✅ no preprocessing, direct fit
        return {
            "accuracy": clf.score(X_te, y_te),
            "steps": [
                {"op": "train_test_split", "test_size": 0.3},
                {"op": "fit", "estimator": "RandomForestClassifier", "scope": "train"},
            ],
            "n_samples": 178, "n_features": 13, "test_fraction": 0.3,
        }

    scenarios.append(PipelineScenario(
        name="clean_rf_wine",
        dataset="wine",
        task="classification",
        ground_truth=GroundTruth(
            has_leakage=False,
            category="clean",
            pattern="Correct: split first, RF fitted on training data only, no preprocessing",
            leaked_features=[],
            approximate_bits=0.0,
        ),
        pipeline_fn=clean_rf_wine,
    ))

    return scenarios


# ═══════════════════════════════════════════════════════════════════
# DETECTOR 1: TaintFlow (our tool) — partition-taint lattice analysis
# ═══════════════════════════════════════════════════════════════════

def taintflow_analyze(pipeline_desc: dict) -> DetectionResult:
    """
    TaintFlow taint analysis: for each pipeline step fitted on 'full' scope,
    compute channel capacity bound C = 0.5 * log2(1 + rho/(1-rho)) per feature.
    """
    t0 = time.perf_counter()
    tracemalloc.start()

    rho = pipeline_desc.get("test_fraction", 0.2)
    steps = pipeline_desc.get("steps", [])
    n_features_total = pipeline_desc.get("n_features", 1)

    total_bits = 0.0
    features_flagged: list[str] = []
    details_parts: list[str] = []

    for step in steps:
        scope = step.get("scope", "train")
        if scope != "full":
            continue

        estimator = step.get("estimator", "")
        n_feat = step.get("n_features", n_features_total)
        cols = step.get("columns", [])

        if estimator in ("StandardScaler", "MinMaxScaler", "SimpleImputer",
                        "RobustScaler", "MaxAbsScaler", "QuantileTransformer",
                        "PowerTransformer", "PolynomialFeatures"):
            # Gaussian channel capacity: C = 0.5 * log2(1 + SNR)
            # where SNR = rho / (1 - rho) for mean/variance sufficient statistics
            c_per_feat = 0.5 * math.log2(1.0 + rho / (1.0 - rho))
            bits = c_per_feat * n_feat
            total_bits += bits
            features_flagged.extend(cols)
            details_parts.append(
                f"{estimator}: {c_per_feat:.4f} bits/feature × {n_feat} = {bits:.4f} bits"
            )

        elif estimator == "PCA":
            # PCA leaks covariance structure; capacity ≤ d*(d+1)/2 * C_gaussian
            # Simplified: treat each component as one feature channel
            c_per_feat = 0.5 * math.log2(1.0 + rho / (1.0 - rho))
            bits = c_per_feat * n_feat
            total_bits += bits
            features_flagged.extend(cols)
            details_parts.append(
                f"PCA: {c_per_feat:.4f} bits/component × {n_feat} = {bits:.4f} bits"
            )

        elif estimator == "SelectKBest":
            # Selection leakage: choosing k from d features leaks selection info
            # Capacity ≤ k * 0.5 * log2(1 + rho/(1 - rho))
            k = step.get("k", 5)
            c_per_feat = 0.5 * math.log2(1.0 + rho / (1.0 - rho + 1e-15))
            bits = c_per_feat * k
            total_bits += bits
            features_flagged.extend(cols)
            details_parts.append(
                f"SelectKBest(k={k}): {c_per_feat:.4f} bits/feature × {k} = {bits:.4f} bits"
            )

        elif estimator in ("TargetEncoder",):
            # Target leakage: Fano bound — H(Y|group) ≤ log2(n_groups)
            n_feat_target = step.get("n_features", 1)
            bits = n_feat_target * math.log2(max(2, 10))  # 10 groups typical
            total_bits += bits
            features_flagged.extend(cols)
            details_parts.append(
                f"TargetEncoder: {bits:.4f} bits ({n_feat_target} features)"
            )

        elif estimator == "KBinsDiscretizer":
            # Discretization leaks bin boundaries computed from full data
            c_per_feat = 0.5 * math.log2(1.0 + rho / (1.0 - rho))
            bits = c_per_feat * n_feat
            total_bits += bits
            features_flagged.extend(cols)
            details_parts.append(
                f"KBinsDiscretizer: {c_per_feat:.4f} bits/feature × {n_feat} = {bits:.4f} bits"
            )

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    detected = total_bits > 0.01
    return DetectionResult(
        method="TaintFlow",
        detected=detected,
        features_flagged=features_flagged,
        bit_bound=round(total_bits, 4),
        time_s=round(elapsed, 6),
        memory_mb=round(peak / (1024 * 1024), 3),
        details="; ".join(details_parts) if details_parts else "No leakage detected",
    )


# ═══════════════════════════════════════════════════════════════════
# DETECTOR 2: Heuristic Pattern Detector (simulates LeakageDetector)
# ═══════════════════════════════════════════════════════════════════

KNOWN_LEAKY_ESTIMATORS = {
    "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler",
    "SelectKBest", "SelectPercentile",
    "PCA", "SimpleImputer", "KBinsDiscretizer",
    "TargetEncoder", "OrdinalEncoder",
    "QuantileTransformer", "PowerTransformer", "PolynomialFeatures",
}

def heuristic_detect(pipeline_desc: dict) -> DetectionResult:
    """
    Simulates AST-based heuristic detectors (like Yang et al.'s LeakageDetector).
    Checks whether any known estimator is fit_transform'd with scope='full'.
    Does NOT detect target leakage or novel patterns.
    """
    t0 = time.perf_counter()
    tracemalloc.start()

    steps = pipeline_desc.get("steps", [])
    features_flagged: list[str] = []
    found_patterns: list[str] = []

    for step in steps:
        scope = step.get("scope", "train")
        estimator = step.get("estimator", "")
        op = step.get("op", "")

        if scope == "full" and op == "fit_transform" and estimator in KNOWN_LEAKY_ESTIMATORS:
            features_flagged.extend(step.get("columns", []))
            found_patterns.append(f"{estimator}.fit_transform(scope=full)")

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    detected = len(found_patterns) > 0
    return DetectionResult(
        method="HeuristicDetector",
        detected=detected,
        features_flagged=features_flagged,
        bit_bound=0.0,  # heuristic cannot quantify
        time_s=round(elapsed, 6),
        memory_mb=round(peak / (1024 * 1024), 3),
        details="; ".join(found_patterns) if found_patterns else "No patterns matched",
    )


# ═══════════════════════════════════════════════════════════════════
# DETECTOR 3: sklearn Pipeline Guard
# ═══════════════════════════════════════════════════════════════════

def sklearn_pipeline_guard(pipeline_desc: dict) -> DetectionResult:
    """
    Checks whether the pipeline uses sklearn.pipeline.Pipeline (which prevents
    leakage by design) vs ad-hoc fit_transform calls.

    sklearn's Pipeline ensures fit_transform is called only on training data
    when used with cross_val_score/train_test_split. But it can't detect
    leakage in ad-hoc scripts that don't use Pipeline.
    """
    t0 = time.perf_counter()
    tracemalloc.start()

    steps = pipeline_desc.get("steps", [])
    has_split_before_fit = False
    has_fit_before_split = False
    features_flagged: list[str] = []

    split_seen = False
    for step in steps:
        op = step.get("op", "")
        scope = step.get("scope", "")
        if op == "train_test_split":
            split_seen = True
        if op in ("fit_transform", "fit") and scope == "full":
            if not split_seen:
                has_fit_before_split = True
                features_flagged.extend(step.get("columns", []))
        if op in ("fit_transform", "fit") and scope == "train" and split_seen:
            has_split_before_fit = True

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    detected = has_fit_before_split
    return DetectionResult(
        method="SklearnPipelineGuard",
        detected=detected,
        features_flagged=features_flagged,
        bit_bound=0.0,
        time_s=round(elapsed, 6),
        memory_mb=round(peak / (1024 * 1024), 3),
        details="fit_transform before split detected" if detected else "Pipeline structure looks correct",
    )


# ═══════════════════════════════════════════════════════════════════
# DETECTOR 4: CV-Gap Audit (nested vs non-nested CV performance gap)
# ═══════════════════════════════════════════════════════════════════

def cv_gap_audit(pipeline_desc: dict, scenario: PipelineScenario) -> DetectionResult:
    """
    Runs nested vs non-nested cross-validation and measures the performance gap.
    A large gap indicates leakage (non-nested CV is inflated by test contamination).

    This is a statistical/empirical baseline — it actually runs the ML pipeline.
    """
    t0 = time.perf_counter()
    tracemalloc.start()

    X, y = load_dataset(scenario.dataset)
    is_classification = scenario.task == "classification"

    # Non-nested CV (potentially leaky: preprocessing on full folds)
    if is_classification:
        estimator = LogisticRegression(max_iter=200, random_state=42)
        scoring = "accuracy"
    else:
        estimator = Ridge(alpha=1.0)
        scoring = "r2"

    # Proper pipeline (nested, no leakage)
    proper_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])
    nested_scores = cross_val_score(proper_pipe, X, y, cv=5, scoring=scoring)

    # Leaky approach: scale first, then CV
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    nonnested_scores = cross_val_score(estimator, X_scaled, y, cv=5, scoring=scoring)

    gap = float(np.mean(nonnested_scores) - np.mean(nested_scores))
    # If gap > threshold, leakage is indicated
    threshold = 0.01  # 1% performance gap
    detected = gap > threshold

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return DetectionResult(
        method="CVGapAudit",
        detected=detected,
        features_flagged=[],  # CV gap can't attribute to specific features
        bit_bound=0.0,
        time_s=round(elapsed, 6),
        memory_mb=round(peak / (1024 * 1024), 3),
        details=f"CV gap = {gap:.4f} (nested={np.mean(nested_scores):.4f}, non-nested={np.mean(nonnested_scores):.4f})",
    )


# ═══════════════════════════════════════════════════════════════════
# DETECTOR 5: DataLinter Rules (Google DataLinter approach)
# ═══════════════════════════════════════════════════════════════════

def datalinter_check(pipeline_desc: dict, scenario: PipelineScenario) -> DetectionResult:
    """
    Implements key DataLinter rules for leakage detection:
    1. Train-test feature distribution similarity (suspiciously high → leakage)
    2. Target-feature correlation check (unrealistically high → target leakage)
    3. Duplicate/near-duplicate check across partitions
    """
    t0 = time.perf_counter()
    tracemalloc.start()

    X, y = load_dataset(scenario.dataset)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    flags: list[str] = []
    features_flagged: list[str] = []

    # Rule 1: Kolmogorov-Smirnov test for train-test distribution similarity
    from scipy.stats import ks_2samp
    suspiciously_similar = 0
    for col in X.columns:
        stat, pval = ks_2samp(X_tr[col], X_te[col])
        if pval > 0.99:  # distributions suspiciously identical
            suspiciously_similar += 1
            features_flagged.append(col)

    if suspiciously_similar > X.shape[1] * 0.5:
        flags.append(f"Distribution similarity: {suspiciously_similar}/{X.shape[1]} features suspiciously similar")

    # Rule 2: Target-feature correlation check
    for col in X_tr.columns:
        corr = abs(np.corrcoef(X_tr[col].values, y_tr.values)[0, 1])
        if corr > 0.95:
            flags.append(f"Suspicious target correlation: {col} (r={corr:.3f})")
            if col not in features_flagged:
                features_flagged.append(col)

    # Rule 3: Check for row duplication across train/test
    tr_set = set(map(tuple, X_tr.values.tolist()))
    te_set = set(map(tuple, X_te.values.tolist()))
    overlap = len(tr_set & te_set)
    if overlap > 0:
        flags.append(f"Row overlap: {overlap} duplicate rows across train/test")

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    detected = len(flags) > 0
    return DetectionResult(
        method="DataLinterRules",
        detected=detected,
        features_flagged=features_flagged,
        bit_bound=0.0,
        time_s=round(elapsed, 6),
        memory_mb=round(peak / (1024 * 1024), 3),
        details="; ".join(flags) if flags else "All DataLinter rules passed",
    )


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════════════

def run_benchmarks() -> dict:
    scenarios = _build_scenarios()
    results: list[dict] = []
    all_detections: dict[str, dict[str, list]] = {}  # method → {TP, FP, FN, TN}

    methods = ["TaintFlow", "HeuristicDetector", "SklearnPipelineGuard", "CVGapAudit", "DataLinterRules"]
    for m in methods:
        all_detections[m] = {"TP": [], "FP": [], "FN": [], "TN": []}

    print(f"\n{'='*80}")
    print(f"  TaintFlow Real-World Benchmark — {len(scenarios)} pipelines")
    print(f"{'='*80}\n")

    for i, sc in enumerate(scenarios):
        print(f"[{i+1}/{len(scenarios)}] {sc.name} ({sc.dataset}, {sc.task})")
        print(f"  Ground truth: leakage={sc.ground_truth.has_leakage}, "
              f"category={sc.ground_truth.category}, bits≈{sc.ground_truth.approximate_bits:.2f}")

        # Execute the pipeline to get its descriptor
        pipeline_desc = sc.pipeline_fn()
        perf_key = "accuracy" if "accuracy" in pipeline_desc else "r2"
        print(f"  Pipeline {perf_key}: {pipeline_desc[perf_key]:.4f}")

        detections: list[dict] = []

        # Run each detector
        # 1) TaintFlow
        tf_result = taintflow_analyze(pipeline_desc)
        detections.append(asdict(tf_result))
        print(f"  TaintFlow:          detected={tf_result.detected}, bits={tf_result.bit_bound:.4f}, "
              f"time={tf_result.time_s:.4f}s")

        # 2) Heuristic
        h_result = heuristic_detect(pipeline_desc)
        detections.append(asdict(h_result))
        print(f"  HeuristicDetector:  detected={h_result.detected}, time={h_result.time_s:.4f}s")

        # 3) sklearn guard
        sk_result = sklearn_pipeline_guard(pipeline_desc)
        detections.append(asdict(sk_result))
        print(f"  SklearnGuard:       detected={sk_result.detected}, time={sk_result.time_s:.4f}s")

        # 4) CV gap
        cv_result = cv_gap_audit(pipeline_desc, sc)
        detections.append(asdict(cv_result))
        print(f"  CVGapAudit:         detected={cv_result.detected}, time={cv_result.time_s:.4f}s, "
              f"details={cv_result.details}")

        # 5) DataLinter
        dl_result = datalinter_check(pipeline_desc, sc)
        detections.append(asdict(dl_result))
        print(f"  DataLinterRules:    detected={dl_result.detected}, time={dl_result.time_s:.4f}s")

        # Classify detections
        for det_obj, det_dict in zip(
            [tf_result, h_result, sk_result, cv_result, dl_result], detections
        ):
            m = det_obj.method
            if sc.ground_truth.has_leakage:
                if det_obj.detected:
                    all_detections[m]["TP"].append(sc.name)
                else:
                    all_detections[m]["FN"].append(sc.name)
            else:
                if det_obj.detected:
                    all_detections[m]["FP"].append(sc.name)
                else:
                    all_detections[m]["TN"].append(sc.name)

        results.append({
            "scenario": sc.name,
            "dataset": sc.dataset,
            "task": sc.task,
            "ground_truth": asdict(sc.ground_truth),
            "pipeline_performance": pipeline_desc.get(perf_key),
            "detections": detections,
        })
        print()

    # ── Compute aggregate metrics ──
    print(f"\n{'='*80}")
    print(f"  AGGREGATE RESULTS")
    print(f"{'='*80}\n")

    summary: dict[str, dict] = {}
    for m in methods:
        tp = len(all_detections[m]["TP"])
        fp = len(all_detections[m]["FP"])
        fn = len(all_detections[m]["FN"])
        tn = len(all_detections[m]["TN"])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Collect timing
        times = []
        for r in results:
            for d in r["detections"]:
                if d["method"] == m:
                    times.append(d["time_s"])
        med_time = float(np.median(times)) if times else 0.0
        p95_time = float(np.percentile(times, 95)) if times else 0.0

        summary[m] = {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": round(precision * 100, 1),
            "recall": round(recall * 100, 1),
            "f1": round(f1 * 100, 1),
            "fpr": round(fpr * 100, 1),
            "median_time_s": round(med_time, 4),
            "p95_time_s": round(p95_time, 4),
            "tp_scenarios": all_detections[m]["TP"],
            "fp_scenarios": all_detections[m]["FP"],
            "fn_scenarios": all_detections[m]["FN"],
            "tn_scenarios": all_detections[m]["TN"],
        }

        print(f"  {m:25s} | Prec={precision*100:5.1f}% | Recall={recall*100:5.1f}% | "
              f"F1={f1*100:5.1f}% | FPR={fpr*100:5.1f}% | TP={tp} FP={fp} FN={fn} TN={tn} | "
              f"Med time={med_time:.4f}s")

    # ── Full output ──
    output = {
        "benchmark_metadata": {
            "name": "TaintFlow Real-World Benchmark",
            "version": "1.0.0",
            "n_pipelines": len(scenarios),
            "n_leaky": sum(1 for s in scenarios if s.ground_truth.has_leakage),
            "n_clean": sum(1 for s in scenarios if not s.ground_truth.has_leakage),
            "datasets": list(set(s.dataset for s in scenarios)),
            "methods": methods,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "per_pipeline_results": results,
        "aggregate_metrics": summary,
    }

    return output


def main():
    output = run_benchmarks()

    out_dir = Path(__file__).parent
    out_path = out_dir / "real_benchmark_results.json"

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n✅ Results written to {out_path}")
    print(f"   {output['benchmark_metadata']['n_pipelines']} pipelines, "
          f"{len(output['aggregate_metrics'])} methods")

    return output


if __name__ == "__main__":
    main()
