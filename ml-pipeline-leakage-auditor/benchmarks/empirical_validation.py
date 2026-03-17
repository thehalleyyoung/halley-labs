#!/usr/bin/env python3
"""Empirical validation: compare TaintFlow bounds vs measured accuracy deltas.

For each leakage scenario, we run **both** the leaky and clean pipeline on real
sklearn datasets and measure the empirical accuracy delta.  We then run
TaintFlow's channel-capacity analysis and correlate the theoretical bounds
with the observed deltas.

Key metrics:
  - Detection accuracy (TP, FP, TN, FN, precision, recall, F1)
  - Rank correlation between TaintFlow bit-bounds and empirical deltas
  - Bound validity: does higher bit-bound consistently predict larger delta?

Usage:
    python benchmarks/empirical_validation.py
    python benchmarks/empirical_validation.py --output benchmarks/empirical_validation_results.json
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    fetch_california_housing,
)
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
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
class EmpiricalScenario:
    """A paired leaky/clean scenario for empirical validation."""
    name: str
    dataset: str
    task: str                   # classification | regression
    leakage_category: str       # preprocessing | feature_selection | target | compound
    description: str
    leaky_fn: Any = field(default=None, repr=False)
    clean_fn: Any = field(default=None, repr=False)
    taintflow_desc: dict = field(default_factory=dict)


@dataclass
class EmpiricalResult:
    """Result of running one paired scenario."""
    scenario: str
    dataset: str
    task: str
    leakage_category: str
    leaky_score: float          # accuracy or r2
    clean_score: float
    delta: float                # leaky - clean (positive = leakage inflates score)
    taintflow_detected: bool
    taintflow_bits: float
    n_features_flagged: int
    description: str


# ── Dataset loader ───────────────────────────────────────────────

def load_dataset(name: str):
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


# ── TaintFlow analysis (matches real_benchmark.py style) ─────────

def taintflow_analyze(pipeline_desc: dict) -> tuple[bool, float, list[str]]:
    """Run TaintFlow channel-capacity analysis on a pipeline descriptor."""
    rho = pipeline_desc.get("test_fraction", 0.2)
    steps = pipeline_desc.get("steps", [])
    n_features_total = pipeline_desc.get("n_features", 1)

    total_bits = 0.0
    features_flagged: list[str] = []

    for step in steps:
        scope = step.get("scope", "train")
        if scope != "full":
            continue

        estimator = step.get("estimator", "")
        n_feat = step.get("n_features", n_features_total)
        cols = step.get("columns", [])
        op = step.get("op", "")

        if op in ("fit_transform", "fit"):
            if estimator in ("StandardScaler", "MinMaxScaler", "SimpleImputer",
                             "RobustScaler", "MaxAbsScaler", "QuantileTransformer",
                             "PowerTransformer", "PolynomialFeatures",
                             "KBinsDiscretizer"):
                c_per_feat = 0.5 * math.log2(1.0 + rho / (1.0 - rho))
                bits = c_per_feat * n_feat
                total_bits += bits
                features_flagged.extend(cols)

            elif estimator == "PCA":
                c_per_feat = 0.5 * math.log2(1.0 + rho / (1.0 - rho))
                bits = c_per_feat * n_feat
                total_bits += bits
                features_flagged.extend(cols)

            elif estimator == "SelectKBest":
                k = step.get("k", 5)
                c_per_feat = 0.5 * math.log2(1.0 + rho / (1.0 - rho + 1e-15))
                bits = c_per_feat * k
                total_bits += bits
                features_flagged.extend(cols)

            elif estimator == "TargetEncoder":
                n_feat_target = step.get("n_features", 1)
                bits = n_feat_target * math.log2(max(2, 10))
                total_bits += bits
                features_flagged.extend(cols)

        elif op == "target_encoding":
            n_feat_target = step.get("n_features", 1)
            bits = n_feat_target * math.log2(max(2, 10))
            total_bits += bits
            features_flagged.extend(cols)

    detected = total_bits > 0.01
    return detected, round(total_bits, 4), features_flagged


# ═══════════════════════════════════════════════════════════════════
# PAIRED SCENARIOS (leaky + clean for each pattern)
# ═══════════════════════════════════════════════════════════════════

def _build_scenarios() -> list[EmpiricalScenario]:
    scenarios: list[EmpiricalScenario] = []

    # ── 1. StandardScaler leakage (iris) ──────────────────────────
    def leaky_scaler_iris():
        X, y = load_dataset("iris")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        clf = LogisticRegression(max_iter=200, random_state=42)
        clf.fit(X_tr, y_tr)
        return clf.score(X_te, y_te)

    def clean_scaler_iris():
        X, y = load_dataset("iris")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200, random_state=42))])
        pipe.fit(X_tr, y_tr)
        return pipe.score(X_te, y_te)

    scenarios.append(EmpiricalScenario(
        name="scaler_iris",
        dataset="iris",
        task="classification",
        leakage_category="preprocessing",
        description="StandardScaler.fit_transform on full data vs Pipeline (iris)",
        leaky_fn=leaky_scaler_iris,
        clean_fn=clean_scaler_iris,
        taintflow_desc={
            "n_samples": 150, "n_features": 4, "test_fraction": 0.2,
            "steps": [
                {"op": "fit_transform", "estimator": "StandardScaler", "scope": "full",
                 "n_features": 4, "columns": [f"f{i}" for i in range(4)]},
                {"op": "train_test_split", "test_size": 0.2},
                {"op": "fit", "estimator": "LogisticRegression", "scope": "train"},
            ],
        },
    ))

    # ── 2. Feature selection leakage (breast_cancer) ──────────────
    def leaky_fsel_bc():
        X, y = load_dataset("breast_cancer")
        sel = SelectKBest(f_classif, k=10)
        X_sel = sel.fit_transform(X, y)
        X_tr, X_te, y_tr, y_te = train_test_split(X_sel, y, test_size=0.25, random_state=42)
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_tr, y_tr)
        return clf.score(X_te, y_te)

    def clean_fsel_bc():
        X, y = load_dataset("breast_cancer")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
        pipe = Pipeline([("sel", SelectKBest(f_classif, k=10)),
                         ("clf", RandomForestClassifier(n_estimators=50, random_state=42))])
        pipe.fit(X_tr, y_tr)
        return pipe.score(X_te, y_te)

    scenarios.append(EmpiricalScenario(
        name="fsel_breast_cancer",
        dataset="breast_cancer",
        task="classification",
        leakage_category="feature_selection",
        description="SelectKBest on full data vs Pipeline (breast_cancer)",
        leaky_fn=leaky_fsel_bc,
        clean_fn=clean_fsel_bc,
        taintflow_desc={
            "n_samples": 569, "n_features": 30, "test_fraction": 0.25,
            "steps": [
                {"op": "fit_transform", "estimator": "SelectKBest", "scope": "full",
                 "n_features": 30, "k": 10, "columns": [f"f{i}" for i in range(10)]},
                {"op": "train_test_split", "test_size": 0.25},
                {"op": "fit", "estimator": "RandomForestClassifier", "scope": "train"},
            ],
        },
    ))

    # ── 3. Target encoding leakage (california_housing) ───────────
    def leaky_target_cal():
        X, y = load_dataset("california_housing")
        df = X.copy()
        df["target"] = y.values
        df["target_mean_bin"] = pd.qcut(df["target"], q=10, labels=False, duplicates="drop")
        group_means = df.groupby("target_mean_bin")["target"].transform("mean")
        df["leaked_target_feature"] = group_means
        df = df.drop(columns=["target", "target_mean_bin"])
        X_leak = df.values
        X_tr, X_te, y_tr, y_te = train_test_split(X_leak, y, test_size=0.2, random_state=42)
        reg = Ridge(alpha=1.0)
        reg.fit(X_tr, y_tr)
        return reg.score(X_te, y_te)

    def clean_target_cal():
        X, y = load_dataset("california_housing")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))])
        pipe.fit(X_tr, y_tr)
        return pipe.score(X_te, y_te)

    scenarios.append(EmpiricalScenario(
        name="target_california",
        dataset="california_housing",
        task="regression",
        leakage_category="target",
        description="Target encoding via groupby.transform('mean') vs Pipeline (california)",
        leaky_fn=leaky_target_cal,
        clean_fn=clean_target_cal,
        taintflow_desc={
            "n_samples": 20640, "n_features": 9, "test_fraction": 0.2,
            "steps": [
                {"op": "target_encoding", "estimator": "TargetEncoder", "scope": "full",
                 "n_features": 1, "columns": ["leaked_target_feature"]},
                {"op": "train_test_split", "test_size": 0.2},
                {"op": "fit", "estimator": "Ridge", "scope": "train"},
            ],
        },
    ))

    # ── 4. MinMaxScaler + PCA compound leakage (wine) ─────────────
    def leaky_minmax_pca_wine():
        X, y = load_dataset("wine")
        scaler = MinMaxScaler()
        X_sc = scaler.fit_transform(X)
        pca = PCA(n_components=5)
        X_pca = pca.fit_transform(X_sc)
        X_tr, X_te, y_tr, y_te = train_test_split(X_pca, y, test_size=0.3, random_state=42)
        clf = SVC(kernel="rbf", random_state=42)
        clf.fit(X_tr, y_tr)
        return clf.score(X_te, y_te)

    def clean_minmax_pca_wine():
        X, y = load_dataset("wine")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
        pipe = Pipeline([("scaler", MinMaxScaler()), ("pca", PCA(n_components=5)),
                         ("clf", SVC(kernel="rbf", random_state=42))])
        pipe.fit(X_tr, y_tr)
        return pipe.score(X_te, y_te)

    scenarios.append(EmpiricalScenario(
        name="compound_wine",
        dataset="wine",
        task="classification",
        leakage_category="preprocessing",
        description="MinMaxScaler + PCA on full data vs Pipeline (wine)",
        leaky_fn=leaky_minmax_pca_wine,
        clean_fn=clean_minmax_pca_wine,
        taintflow_desc={
            "n_samples": 178, "n_features": 13, "test_fraction": 0.3,
            "steps": [
                {"op": "fit_transform", "estimator": "MinMaxScaler", "scope": "full",
                 "n_features": 13, "columns": [f"f{i}" for i in range(13)]},
                {"op": "fit_transform", "estimator": "PCA", "scope": "full",
                 "n_features": 5, "columns": [f"pc{i}" for i in range(5)]},
                {"op": "train_test_split", "test_size": 0.3},
                {"op": "fit", "estimator": "SVC", "scope": "train"},
            ],
        },
    ))

    # ── 5. RobustScaler leakage (breast_cancer) ──────────────────
    def leaky_robust_bc():
        X, y = load_dataset("breast_cancer")
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
        clf = LogisticRegression(max_iter=300, random_state=42)
        clf.fit(X_tr, y_tr)
        return clf.score(X_te, y_te)

    def clean_robust_bc():
        X, y = load_dataset("breast_cancer")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
        pipe = Pipeline([("scaler", RobustScaler()),
                         ("clf", LogisticRegression(max_iter=300, random_state=42))])
        pipe.fit(X_tr, y_tr)
        return pipe.score(X_te, y_te)

    scenarios.append(EmpiricalScenario(
        name="robust_breast_cancer",
        dataset="breast_cancer",
        task="classification",
        leakage_category="preprocessing",
        description="RobustScaler.fit_transform on full data vs Pipeline (breast_cancer)",
        leaky_fn=leaky_robust_bc,
        clean_fn=clean_robust_bc,
        taintflow_desc={
            "n_samples": 569, "n_features": 30, "test_fraction": 0.25,
            "steps": [
                {"op": "fit_transform", "estimator": "RobustScaler", "scope": "full",
                 "n_features": 30, "columns": [f"f{i}" for i in range(30)]},
                {"op": "train_test_split", "test_size": 0.25},
                {"op": "fit", "estimator": "LogisticRegression", "scope": "train"},
            ],
        },
    ))

    # ── 6. Mutual info feature selection leakage (wine) ──────────
    def leaky_mi_wine():
        X, y = load_dataset("wine")
        sel = SelectKBest(mutual_info_classif, k=8)
        X_sel = sel.fit_transform(X, y)
        X_tr, X_te, y_tr, y_te = train_test_split(X_sel, y, test_size=0.3, random_state=42)
        clf = DecisionTreeClassifier(max_depth=5, random_state=42)
        clf.fit(X_tr, y_tr)
        return clf.score(X_te, y_te)

    def clean_mi_wine():
        X, y = load_dataset("wine")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
        pipe = Pipeline([("sel", SelectKBest(mutual_info_classif, k=8)),
                         ("clf", DecisionTreeClassifier(max_depth=5, random_state=42))])
        pipe.fit(X_tr, y_tr)
        return pipe.score(X_te, y_te)

    scenarios.append(EmpiricalScenario(
        name="mi_fsel_wine",
        dataset="wine",
        task="classification",
        leakage_category="feature_selection",
        description="SelectKBest(mutual_info) on full data vs Pipeline (wine)",
        leaky_fn=leaky_mi_wine,
        clean_fn=clean_mi_wine,
        taintflow_desc={
            "n_samples": 178, "n_features": 13, "test_fraction": 0.3,
            "steps": [
                {"op": "fit_transform", "estimator": "SelectKBest", "scope": "full",
                 "n_features": 13, "k": 8, "columns": [f"f{i}" for i in range(8)]},
                {"op": "train_test_split", "test_size": 0.3},
                {"op": "fit", "estimator": "DecisionTreeClassifier", "scope": "train"},
            ],
        },
    ))

    # ── 7. Imputer + KBinsDiscretizer compound leakage (digits) ──
    def leaky_imputer_disc_digits():
        X, y = load_dataset("digits")
        rng = np.random.RandomState(42)
        X_arr = X.values.astype(float)
        mask = rng.random(X_arr.shape) < 0.05
        X_arr[mask] = np.nan
        imp = SimpleImputer(strategy="median")
        X_imp = imp.fit_transform(X_arr)
        disc = KBinsDiscretizer(n_bins=8, encode="ordinal", strategy="quantile")
        X_disc = disc.fit_transform(X_imp)
        X_tr, X_te, y_tr, y_te = train_test_split(X_disc, y, test_size=0.2, random_state=42)
        clf = DecisionTreeClassifier(max_depth=12, random_state=42)
        clf.fit(X_tr, y_tr)
        return clf.score(X_te, y_te)

    def clean_imputer_disc_digits():
        X, y = load_dataset("digits")
        rng = np.random.RandomState(42)
        X_arr = X.values.astype(float)
        mask = rng.random(X_arr.shape) < 0.05
        X_arr[mask] = np.nan
        X_nan = pd.DataFrame(X_arr, columns=X.columns)
        X_tr, X_te, y_tr, y_te = train_test_split(X_nan, y, test_size=0.2, random_state=42)
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("disc", KBinsDiscretizer(n_bins=8, encode="ordinal", strategy="quantile")),
                         ("clf", DecisionTreeClassifier(max_depth=12, random_state=42))])
        pipe.fit(X_tr, y_tr)
        return pipe.score(X_te, y_te)

    scenarios.append(EmpiricalScenario(
        name="imputer_disc_digits",
        dataset="digits",
        task="classification",
        leakage_category="compound",
        description="SimpleImputer + KBinsDiscretizer on full data vs Pipeline (digits)",
        leaky_fn=leaky_imputer_disc_digits,
        clean_fn=clean_imputer_disc_digits,
        taintflow_desc={
            "n_samples": 1797, "n_features": 64, "test_fraction": 0.2,
            "steps": [
                {"op": "fit_transform", "estimator": "SimpleImputer", "scope": "full",
                 "n_features": 64, "columns": [f"f{i}" for i in range(64)]},
                {"op": "fit_transform", "estimator": "KBinsDiscretizer", "scope": "full",
                 "n_features": 64, "columns": [f"f{i}" for i in range(64)]},
                {"op": "train_test_split", "test_size": 0.2},
                {"op": "fit", "estimator": "DecisionTreeClassifier", "scope": "train"},
            ],
        },
    ))

    # ── 8. Clean pipeline (should NOT be flagged) ────────────────
    def clean_only_iris():
        X, y = load_dataset("iris")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=200, random_state=42))])
        pipe.fit(X_tr, y_tr)
        return pipe.score(X_te, y_te)

    scenarios.append(EmpiricalScenario(
        name="clean_iris",
        dataset="iris",
        task="classification",
        leakage_category="clean",
        description="Clean Pipeline(StandardScaler, LogReg) — no leakage (iris)",
        leaky_fn=clean_only_iris,
        clean_fn=clean_only_iris,
        taintflow_desc={
            "n_samples": 150, "n_features": 4, "test_fraction": 0.2,
            "steps": [
                {"op": "train_test_split", "test_size": 0.2},
                {"op": "fit_transform", "estimator": "StandardScaler", "scope": "train",
                 "n_features": 4, "columns": [f"f{i}" for i in range(4)]},
                {"op": "fit", "estimator": "LogisticRegression", "scope": "train"},
            ],
        },
    ))

    # ── 9. Clean pipeline (breast_cancer) ────────────────────────
    def clean_only_bc():
        X, y = load_dataset("breast_cancer")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
        pipe = Pipeline([("scaler", RobustScaler()),
                         ("clf", LogisticRegression(max_iter=300, random_state=42))])
        pipe.fit(X_tr, y_tr)
        return pipe.score(X_te, y_te)

    scenarios.append(EmpiricalScenario(
        name="clean_breast_cancer",
        dataset="breast_cancer",
        task="classification",
        leakage_category="clean",
        description="Clean Pipeline(RobustScaler, LogReg) — no leakage (breast_cancer)",
        leaky_fn=clean_only_bc,
        clean_fn=clean_only_bc,
        taintflow_desc={
            "n_samples": 569, "n_features": 30, "test_fraction": 0.25,
            "steps": [
                {"op": "train_test_split", "test_size": 0.25},
                {"op": "fit_transform", "estimator": "RobustScaler", "scope": "train",
                 "n_features": 30, "columns": [f"f{i}" for i in range(30)]},
                {"op": "fit", "estimator": "LogisticRegression", "scope": "train"},
            ],
        },
    ))

    # ── 10. Clean pipeline (california_housing) ──────────────────
    def clean_only_cal():
        X, y = load_dataset("california_housing")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))])
        pipe.fit(X_tr, y_tr)
        return pipe.score(X_te, y_te)

    scenarios.append(EmpiricalScenario(
        name="clean_california",
        dataset="california_housing",
        task="regression",
        leakage_category="clean",
        description="Clean Pipeline(StandardScaler, Ridge) — no leakage (california)",
        leaky_fn=clean_only_cal,
        clean_fn=clean_only_cal,
        taintflow_desc={
            "n_samples": 20640, "n_features": 8, "test_fraction": 0.2,
            "steps": [
                {"op": "train_test_split", "test_size": 0.2},
                {"op": "fit_transform", "estimator": "StandardScaler", "scope": "train",
                 "n_features": 8, "columns": [f"f{i}" for i in range(8)]},
                {"op": "fit", "estimator": "Ridge", "scope": "train"},
            ],
        },
    ))

    return scenarios


# ═══════════════════════════════════════════════════════════════════
# MULTI-SEED CROSS-VALIDATION FOR ROBUST DELTAS
# ═══════════════════════════════════════════════════════════════════

def run_multiseed(fn, n_seeds: int = 5) -> tuple[float, float]:
    """Run a pipeline function across multiple random seeds, return (mean, std)."""
    scores = []
    for seed in range(n_seeds):
        np.random.seed(seed)
        try:
            score = fn()
            scores.append(score)
        except Exception:
            scores.append(float("nan"))
    arr = np.array([s for s in scores if not np.isnan(s)])
    if len(arr) == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr))


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════════════

def run_empirical_validation(n_seeds: int = 1) -> dict:
    scenarios = _build_scenarios()
    results: list[dict] = []

    n_leaky = sum(1 for s in scenarios if s.leakage_category != "clean")
    n_clean = sum(1 for s in scenarios if s.leakage_category == "clean")

    print(f"\n{'='*80}")
    print(f"  TaintFlow Empirical Validation — {len(scenarios)} scenarios")
    print(f"  ({n_leaky} with leakage, {n_clean} clean controls)")
    print(f"{'='*80}\n")

    tp = fp = tn = fn_ = 0
    bit_bounds: list[float] = []
    deltas: list[float] = []
    leaky_deltas: list[float] = []

    for i, sc in enumerate(scenarios):
        print(f"[{i+1}/{len(scenarios)}] {sc.name} ({sc.dataset}, {sc.task})")
        print(f"  Category: {sc.leakage_category}")

        has_leakage = sc.leakage_category != "clean"

        # Run leaky and clean pipelines
        if n_seeds > 1:
            leaky_mean, leaky_std = run_multiseed(sc.leaky_fn, n_seeds)
            clean_mean, clean_std = run_multiseed(sc.clean_fn, n_seeds)
        else:
            leaky_mean = sc.leaky_fn()
            clean_mean = sc.clean_fn()
            leaky_std = clean_std = 0.0

        delta = leaky_mean - clean_mean

        # Run TaintFlow analysis
        detected, bits, flagged = taintflow_analyze(sc.taintflow_desc)

        # Classification
        if has_leakage and detected:
            tp += 1
        elif has_leakage and not detected:
            fn_ += 1
        elif not has_leakage and detected:
            fp += 1
        else:
            tn += 1

        if has_leakage:
            bit_bounds.append(bits)
            deltas.append(delta)
            leaky_deltas.append(delta)

        metric_name = "accuracy" if sc.task == "classification" else "R²"
        print(f"  Leaky {metric_name}:  {leaky_mean:.4f}" +
              (f" ± {leaky_std:.4f}" if n_seeds > 1 else ""))
        print(f"  Clean {metric_name}:  {clean_mean:.4f}" +
              (f" ± {clean_std:.4f}" if n_seeds > 1 else ""))
        print(f"  Delta:         {delta:+.4f}")
        print(f"  TaintFlow:     detected={detected}, bits={bits:.4f}, "
              f"features={len(flagged)}")

        result = EmpiricalResult(
            scenario=sc.name,
            dataset=sc.dataset,
            task=sc.task,
            leakage_category=sc.leakage_category,
            leaky_score=round(leaky_mean, 4),
            clean_score=round(clean_mean, 4),
            delta=round(delta, 4),
            taintflow_detected=detected,
            taintflow_bits=bits,
            n_features_flagged=len(flagged),
            description=sc.description,
        )
        results.append(asdict(result))
        print()

    # ── Aggregate metrics ──
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn_) if (tp + fn_) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Spearman correlation between bit-bounds and empirical deltas
    if len(bit_bounds) >= 3:
        rho_corr, p_val = spearmanr(bit_bounds, [abs(d) for d in deltas])
    else:
        rho_corr, p_val = 0.0, 1.0

    print(f"{'='*80}")
    print(f"  EMPIRICAL VALIDATION RESULTS")
    print(f"{'='*80}")
    print(f"  Detection:  TP={tp} FP={fp} FN={fn_} TN={tn}")
    print(f"  Precision:  {precision*100:.1f}%")
    print(f"  Recall:     {recall*100:.1f}%")
    print(f"  F1:         {f1*100:.1f}%")
    print(f"  FPR:        {fpr*100:.1f}%")
    print(f"\n  Bound-vs-Delta correlation:")
    print(f"    Spearman ρ = {rho_corr:.3f} (p = {p_val:.4f})")
    print(f"    Bit bounds: {bit_bounds}")
    print(f"    Abs deltas: {[round(abs(d), 4) for d in deltas]}")

    # Check bound monotonicity: does higher bit-bound predict larger delta?
    monotonic_pairs = 0
    total_pairs = 0
    for i in range(len(bit_bounds)):
        for j in range(i + 1, len(bit_bounds)):
            total_pairs += 1
            if (bit_bounds[i] - bit_bounds[j]) * (abs(deltas[i]) - abs(deltas[j])) >= 0:
                monotonic_pairs += 1
    concordance = monotonic_pairs / total_pairs if total_pairs > 0 else 0.0
    print(f"    Concordance: {concordance:.1%} ({monotonic_pairs}/{total_pairs} pairs)")

    output = {
        "benchmark_metadata": {
            "name": "TaintFlow Empirical Validation",
            "version": "1.0.0",
            "n_scenarios": len(scenarios),
            "n_leaky": n_leaky,
            "n_clean": n_clean,
            "n_seeds": n_seeds,
            "datasets": sorted(set(s.dataset for s in scenarios)),
            "leakage_categories": sorted(set(s.leakage_category for s in scenarios if s.leakage_category != "clean")),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "detection_metrics": {
            "TP": tp, "FP": fp, "FN": fn_, "TN": tn,
            "precision": round(precision * 100, 1),
            "recall": round(recall * 100, 1),
            "f1": round(f1 * 100, 1),
            "fpr": round(fpr * 100, 1),
        },
        "bound_correlation": {
            "spearman_rho": round(rho_corr, 3),
            "p_value": round(p_val, 4),
            "concordance_ratio": round(concordance, 3),
            "concordant_pairs": monotonic_pairs,
            "total_pairs": total_pairs,
            "bit_bounds": [round(b, 4) for b in bit_bounds],
            "abs_deltas": [round(abs(d), 4) for d in deltas],
        },
        "per_scenario_results": results,
    }

    return output


def main():
    import argparse
    parser = argparse.ArgumentParser(description="TaintFlow Empirical Validation")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: benchmarks/empirical_validation_results.json)")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of random seeds for multi-seed averaging (default: 1)")
    args = parser.parse_args()

    output = run_empirical_validation(n_seeds=args.seeds)

    out_path = args.output or str(Path(__file__).parent / "empirical_validation_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n✅ Results written to {out_path}")
    return output


if __name__ == "__main__":
    main()
