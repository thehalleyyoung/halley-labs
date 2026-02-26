#!/usr/bin/env python3
"""
Run per-stage error decomposition experiment.

Generates synthetic data with known ground truth, runs the full
pipeline, computes error contributions from each stage (regime
detection, causal discovery, PAC-Bayes, shield), saves results as
JSON, and generates pgfplots-compatible data files.

Usage
-----
    python experiments/run_error_decomposition.py [--output-dir results/error_decomp]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.special import gammaln, digamma

# Ensure the implementation package is importable
_IMPL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _IMPL_DIR not in sys.path:
    sys.path.insert(0, _IMPL_DIR)

from causal_trading.market.synthetic import SyntheticMarketGenerator
from causal_trading.regime.sticky_hdp_hmm import StickyHDPHMM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error decomposition data structures
# ---------------------------------------------------------------------------

@dataclass
class StageError:
    """Error metrics for a single pipeline stage."""
    stage: str
    error: float
    relative_error: float
    details: Dict[str, float] = field(default_factory=dict)


@dataclass
class ErrorDecomposition:
    """Full per-stage error decomposition."""
    stages: List[StageError] = field(default_factory=list)
    total_error: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stages": [
                {
                    "stage": s.stage,
                    "error": s.error,
                    "relative_error": s.relative_error,
                    "details": s.details,
                }
                for s in self.stages
            ],
            "total_error": self.total_error,
            "metadata": self.metadata,
        }

    def to_pgfplots_data(self) -> str:
        """Generate pgfplots .dat content."""
        lines = ["stage error relative_error"]
        for i, s in enumerate(self.stages):
            lines.append(f"{i} {s.error:.6g} {s.relative_error:.6g}")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Error computation helpers
# ---------------------------------------------------------------------------

def _regime_error(
    predicted_states: np.ndarray,
    true_labels: np.ndarray,
) -> StageError:
    """Compute regime detection error via normalised mutual information."""
    from scipy.stats import entropy as sp_entropy

    T = len(true_labels)
    pred_unique = np.unique(predicted_states)
    true_unique = np.unique(true_labels)

    # Confusion-based accuracy (with best permutation alignment)
    # Simple approach: use cluster purity
    purity = 0.0
    for c in pred_unique:
        mask = predicted_states == c
        if mask.sum() == 0:
            continue
        counts = np.array([np.sum(true_labels[mask] == t) for t in true_unique])
        purity += counts.max()
    purity /= T

    # Adjusted Rand Index (simplified)
    error = 1.0 - purity

    # Duration error
    true_durations = _avg_duration(true_labels)
    pred_durations = _avg_duration(predicted_states)
    duration_err = abs(true_durations - pred_durations) / max(true_durations, 1.0)

    return StageError(
        stage="regime_detection",
        error=error,
        relative_error=error,
        details={
            "purity": purity,
            "n_regimes_true": float(len(true_unique)),
            "n_regimes_pred": float(len(pred_unique)),
            "duration_error": duration_err,
        },
    )


def _avg_duration(states: np.ndarray) -> float:
    durations = []
    cur = 1
    for t in range(1, len(states)):
        if states[t] == states[t - 1]:
            cur += 1
        else:
            durations.append(cur)
            cur = 1
    durations.append(cur)
    return float(np.mean(durations))


def _causal_error(
    data: np.ndarray,
    true_adj: np.ndarray,
    states: np.ndarray,
) -> StageError:
    """Compute causal discovery error using correlation-based proxy."""
    T, D = data.shape
    if D < 2:
        return StageError(
            stage="causal_discovery",
            error=0.0,
            relative_error=0.0,
            details={"n_true_edges": 0, "n_discovered_edges": 0},
        )

    # Discover edges using simple correlation threshold
    corr = np.corrcoef(data.T)
    threshold = 2.0 / np.sqrt(T)
    discovered = np.abs(corr) > threshold
    np.fill_diagonal(discovered, False)

    # Compare to ground truth (upper triangle)
    n_true = int(true_adj.sum())
    n_disc = int(discovered.sum()) // 2  # symmetric
    tp = 0
    fp = 0
    fn = 0
    for i in range(D):
        for j in range(i + 1, D):
            t = true_adj[i, j] or true_adj[j, i]
            d = discovered[i, j]
            if t and d:
                tp += 1
            elif not t and d:
                fp += 1
            elif t and not d:
                fn += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    error = 1.0 - f1

    return StageError(
        stage="causal_discovery",
        error=error,
        relative_error=error,
        details={
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_true_edges": float(n_true),
            "n_discovered_edges": float(n_disc),
        },
    )


def _pac_bayes_error(
    states: np.ndarray,
    n: int,
    delta: float = 0.05,
) -> StageError:
    """Compute PAC-Bayes bound tightness."""
    n_regimes = len(np.unique(states))
    posterior_counts = np.zeros(n_regimes)
    for i, r in enumerate(np.unique(states)):
        posterior_counts[i] = float((states == r).sum())
    posterior_counts += 1.0
    prior_alpha = np.full(n_regimes, 10.0 / max(n_regimes, 1))

    kl = float(
        gammaln(posterior_counts.sum()) - gammaln(prior_alpha.sum())
        - np.sum(gammaln(posterior_counts)) + np.sum(gammaln(prior_alpha))
        + np.sum((posterior_counts - prior_alpha)
                 * (digamma(posterior_counts) - digamma(posterior_counts.sum())))
    )
    kl = max(kl, 0.0)
    bound = min((kl + np.log(2 * np.sqrt(n) / delta)) / n, 1.0)

    return StageError(
        stage="pac_bayes",
        error=bound,
        relative_error=bound,
        details={"kl_divergence": kl, "bound": bound, "n_regimes": float(n_regimes)},
    )


def _shield_error(states: np.ndarray, K_max: int) -> StageError:
    """Compute shield permissivity gap."""
    n_active = len(np.unique(states))
    # Use n_active as effective K for informative prior
    effective_k = max(n_active, 1)
    permissivity = float(effective_k) / float(max(K_max, effective_k))
    error = 1.0 - permissivity

    return StageError(
        stage="shield",
        error=error,
        relative_error=error,
        details={"permissivity": permissivity, "n_active": float(n_active)},
    )


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_error_decomposition(
    n_regimes: int = 3,
    n_features: int = 5,
    T: int = 1000,
    seed: int = 42,
) -> ErrorDecomposition:
    """Run the full pipeline and decompose errors."""
    # 1. Generate data
    gen = SyntheticMarketGenerator(
        n_features=n_features,
        n_regimes=n_regimes,
        seed=seed,
    )
    dataset = gen.generate(T=T, n_regimes=n_regimes, n_features=n_features)
    data = dataset.features
    true_labels = dataset.ground_truth.regime_labels
    true_adj = dataset.ground_truth.adjacency_matrices.get(0, np.zeros((n_features, n_features)))

    # 2. Regime detection
    hmm = StickyHDPHMM(
        K_max=10,
        kappa=50.0,
        n_iter=100,
        burn_in=20,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hmm.fit(data)
    predicted_states = hmm.states_

    # 3. Compute per-stage errors
    decomp = ErrorDecomposition()
    decomp.stages.append(_regime_error(predicted_states, true_labels))
    decomp.stages.append(_causal_error(data, true_adj, predicted_states))
    decomp.stages.append(_pac_bayes_error(predicted_states, T))
    decomp.stages.append(_shield_error(predicted_states, K_max=10))

    decomp.total_error = sum(s.error for s in decomp.stages)
    decomp.metadata = {
        "n_regimes": n_regimes,
        "n_features": n_features,
        "T": T,
        "seed": seed,
    }
    return decomp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run error decomposition")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/error_decomp",
        help="Output directory",
    )
    parser.add_argument("--n-regimes", type=int, default=3)
    parser.add_argument("--n-features", type=int, default=5)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Running error decomposition ===")
    t0 = time.time()
    decomp = run_error_decomposition(
        n_regimes=args.n_regimes,
        n_features=args.n_features,
        T=args.T,
        seed=args.seed,
    )
    elapsed = time.time() - t0
    decomp.metadata["elapsed_seconds"] = elapsed

    # Save JSON
    json_path = output_dir / "error_decomposition.json"
    json_path.write_text(json.dumps(decomp.to_dict(), indent=2, default=str))
    logger.info("Saved JSON to %s", json_path)

    # Save pgfplots data
    dat_path = output_dir / "error_decomposition.dat"
    dat_path.write_text(decomp.to_pgfplots_data())
    logger.info("Saved pgfplots data to %s", dat_path)

    # Summary
    logger.info("=== Error Decomposition Summary ===")
    logger.info("Total error: %.4f", decomp.total_error)
    for s in decomp.stages:
        logger.info("  %-25s error=%.4f  details=%s", s.stage, s.error, s.details)
    logger.info("Done in %.1f s", elapsed)


if __name__ == "__main__":
    main()
