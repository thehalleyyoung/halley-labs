#!/usr/bin/env python3
"""
Multi-instrument evaluation experiment.

Generates synthetic data for 3 different asset classes (equity, FX, crypto),
runs coupled inference on each, and evaluates regime detection accuracy (ARI),
causal discovery accuracy (SHD, precision, recall), PAC-Bayes bounds, and
shield permissivity.

Usage
-----
    cd implementation/
    python3 experiments/run_multi_instrument.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.special import gammaln, digamma

_IMPL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _IMPL_DIR not in sys.path:
    sys.path.insert(0, _IMPL_DIR)

from causal_trading.market.synthetic import SyntheticMarketGenerator
from causal_trading.coupled.em_alternation import CoupledInference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
SEED = 42

# -----------------------------------------------------------------------
# Asset class configurations
# -----------------------------------------------------------------------

ASSET_CLASSES = {
    "equity": {
        "description": "Equity-like: 3 regimes (bull/bear/crash), moderate stickiness",
        "n_regimes": 3,
        "n_features": 5,
        "regime_persistence": 0.95,
        "T": 500,
    },
    "fx": {
        "description": "FX-like: 2 regimes (trending/ranging), high stickiness",
        "n_regimes": 2,
        "n_features": 8,
        "regime_persistence": 0.98,
        "T": 500,
    },
    "crypto": {
        "description": "Crypto-like: 4 regimes (bubble/crash/accumulation/distribution), low stickiness",
        "n_regimes": 4,
        "n_features": 10,
        "regime_persistence": 0.90,
        "T": 500,
    },
}


# -----------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------

def _adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Compute Adjusted Rand Index between two label arrays."""
    n = len(labels_true)
    if n == 0:
        return 0.0

    classes_true = np.unique(labels_true)
    classes_pred = np.unique(labels_pred)

    # Contingency table
    contingency = np.zeros((len(classes_true), len(classes_pred)), dtype=np.int64)
    true_map = {c: i for i, c in enumerate(classes_true)}
    pred_map = {c: i for i, c in enumerate(classes_pred)}
    for t, p in zip(labels_true, labels_pred):
        contingency[true_map[t], pred_map[p]] += 1

    sum_comb_c = sum(int(nij) * (int(nij) - 1) // 2 for nij in contingency.ravel())
    sum_comb_a = sum(int(ai) * (int(ai) - 1) // 2 for ai in contingency.sum(axis=1))
    sum_comb_b = sum(int(bj) * (int(bj) - 1) // 2 for bj in contingency.sum(axis=0))

    total_comb = n * (n - 1) // 2
    expected = sum_comb_a * sum_comb_b / max(total_comb, 1)
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    denom = max_index - expected
    if abs(denom) < 1e-12:
        return 1.0 if sum_comb_c == expected else 0.0
    return float((sum_comb_c - expected) / denom)


def _structural_hamming_distance(
    true_adj: np.ndarray, est_adj: np.ndarray
) -> Dict[str, Any]:
    """Compute SHD and edge-level metrics between adjacency matrices."""
    p = true_adj.shape[0]
    tp = fp = fn = 0
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            t = bool(true_adj[i, j])
            e = bool(est_adj[i, j])
            if t and e:
                tp += 1
            elif not t and e:
                fp += 1
            elif t and not e:
                fn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    shd = fp + fn  # structural Hamming distance
    return {
        "shd": shd,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _pac_bayes_bound(states: np.ndarray, n: int, delta: float = 0.05) -> float:
    """Compute PAC-Bayes bound for regime posterior."""
    n_r = len(np.unique(states))
    posterior = np.array([float((states == r).sum()) for r in np.unique(states)]) + 1.0
    prior = np.full(n_r, 10.0 / max(n_r, 1))
    kl = float(
        gammaln(posterior.sum()) - gammaln(prior.sum())
        - np.sum(gammaln(posterior)) + np.sum(gammaln(prior))
        + np.sum(
            (posterior - prior) * (digamma(posterior) - digamma(posterior.sum()))
        )
    )
    kl = max(kl, 0.0)
    return min((kl + np.log(2 * np.sqrt(n) / delta)) / n, 1.0)


def _shield_permissivity(states: np.ndarray, n_regimes: int) -> float:
    """Compute shield permissivity as fraction of active regimes."""
    n_active = len(np.unique(states))
    return float(n_active) / float(max(n_regimes, 1))


# -----------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------

def run_multi_instrument_experiment() -> Dict[str, Any]:
    """Run coupled inference on multiple asset classes and evaluate."""
    logger.info("=== Multi-Instrument Evaluation ===")
    results: Dict[str, Any] = {}

    for asset_name, config in ASSET_CLASSES.items():
        logger.info("--- Asset class: %s ---", asset_name)
        t0 = time.time()

        n_regimes = config["n_regimes"]
        n_features = config["n_features"]
        T = config["T"]
        persistence = config["regime_persistence"]

        # Generate synthetic data
        gen = SyntheticMarketGenerator(
            n_features=n_features,
            n_regimes=n_regimes,
            regime_persistence=persistence,
            seed=SEED,
        )
        dataset = gen.generate(T=T, n_regimes=n_regimes, n_features=n_features)
        data = dataset.features
        true_labels = dataset.ground_truth.regime_labels
        true_adj = dataset.ground_truth.adjacency_matrices.get(
            0, np.zeros((n_features, n_features))
        )

        # Run coupled inference
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = CoupledInference(
                n_regimes=n_regimes,
                alpha_ci=0.05,
                max_cond_size=2,
                sticky_kappa=50.0,
                seed=SEED,
            )
            model.fit(data, max_iter=30)

        pred_regimes = model.get_regimes()
        causal_graphs = model.get_causal_graphs()

        # Build estimated adjacency from first regime's graph
        est_adj = np.zeros((n_features, n_features), dtype=bool)
        if 0 in causal_graphs:
            for u, v in causal_graphs[0].edges():
                if u < n_features and v < n_features:
                    est_adj[u, v] = True

        # Evaluate
        ari = _adjusted_rand_index(true_labels, pred_regimes)
        shd_metrics = _structural_hamming_distance(true_adj, est_adj)
        pac_bound = _pac_bayes_bound(pred_regimes, T)
        permissivity = _shield_permissivity(pred_regimes, n_regimes)

        elapsed = time.time() - t0

        asset_result = {
            "description": config["description"],
            "n_regimes": n_regimes,
            "n_features": n_features,
            "T": T,
            "regime_persistence": persistence,
            "regime_detection": {
                "adjusted_rand_index": round(ari, 4),
            },
            "causal_discovery": shd_metrics,
            "pac_bayes_bound": round(pac_bound, 6),
            "shield_permissivity": round(permissivity, 4),
            "convergence": model.convergence_diagnostics,
            "elapsed_s": round(elapsed, 2),
        }
        results[asset_name] = asset_result

        logger.info(
            "  %s: ARI=%.4f, SHD=%d, PAC=%.4f, perm=%.2f (%.1fs)",
            asset_name, ari, shd_metrics["shd"], pac_bound, permissivity, elapsed,
        )

    output = {
        "experiment": "multi_instrument",
        "seed": SEED,
        "asset_classes": results,
    }

    out_path = RESULTS_DIR / "multi_instrument.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info("Saved %s", out_path)

    return output


if __name__ == "__main__":
    run_multi_instrument_experiment()
