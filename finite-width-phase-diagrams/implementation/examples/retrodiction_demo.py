#!/usr/bin/env python3
"""Example: reproduce known results (Chizat & Bach) and compare predictions.

Demonstrates the retrodiction pipeline:
  1. Define known theoretical results
  2. Compute predictions using the phase diagram system
  3. Compare and report deviations
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.evaluation import RetrodictionValidator, KnownResult
from src.kernel_engine import AnalyticNTK
from src.corrections import FiniteWidthCorrector


def main():
    print("=" * 60)
    print("  Retrodiction Demo: Known Results Validation")
    print("=" * 60)

    rng = np.random.RandomState(42)
    X = rng.randn(20, 5)

    # ----- Compute NTKs for reference ----------------------------------------
    print("\nPreparing NTK data...")
    analytic = AnalyticNTK()
    widths = [32, 64, 128, 256, 512, 1024]
    ntk_data = {}
    for w in widths:
        ntk_data[w] = analytic.compute(X, depth=2, width=w, activation="relu")
    print(f"  Computed NTKs at widths: {widths}")

    # Fit corrections
    corrector = FiniteWidthCorrector(order_max=2)
    theta_0 = ntk_data[widths[-1]]
    corrections = corrector.compute_corrections_regression(ntk_data, theta_0=theta_0)

    # ----- Define prediction functions ----------------------------------------
    def chizat_bach_predictor(lr: float, width: int, **kw) -> float:
        """Predict lazy/rich transition scaling.

        Chizat & Bach (2019): transition at α ~ 1/√N for mean-field,
        where α is learning rate scale.
        """
        depth = kw.get("depth", 2)
        gamma = lr * depth / np.sqrt(max(width, 1))
        return float(gamma)

    def saxe_predictor(lr: float, width: int, **kw) -> float:
        """Predict learning dynamics scaling from Saxe et al.

        Deep linear networks: dynamics governed by singular values.
        """
        depth = kw.get("depth", 2)
        return float(lr * depth / max(width, 1))

    def mup_predictor(lr: float, width: int, **kw) -> float:
        """Predict μP scaling exponents.

        Yang & Hu (2021): maximal update parametrisation.
        """
        return float(lr / max(width, 1))

    def kernel_fp_predictor(lr: float, width: int, **kw) -> float:
        """Predict kernel fixed-point behaviour."""
        K = theta_0
        eigvals = np.linalg.eigvalsh(K)
        spectral_gap = float(eigvals[-1] - eigvals[-2]) if len(eigvals) >= 2 else 1.0
        return lr * spectral_gap / max(width, 1)

    # ----- Run retrodiction ---------------------------------------------------
    print("\nRunning retrodiction validation...")
    validator = RetrodictionValidator()

    compute_fns = {
        "chizat_bach": chizat_bach_predictor,
        "saxe_dynamics": saxe_predictor,
        "mup_exponents": mup_predictor,
        "kernel_fixed_point": kernel_fp_predictor,
    }

    results = validator.run_all(compute_fns)

    # ----- Report results -----------------------------------------------------
    print("\n" + "=" * 60)
    print("  Retrodiction Results")
    print("=" * 60)

    for r in results:
        name = r.name if hasattr(r, "name") else "?"
        passed = r.passed if hasattr(r, "passed") else None
        deviation = r.deviation if hasattr(r, "deviation") else None

        status = "PASS" if passed else "FAIL" if passed is not None else "N/A"
        dev_str = f"deviation={deviation:.4f}" if deviation is not None else ""
        print(f"  [{status}] {name:30s} {dev_str}")

    # ----- Deviation analysis -------------------------------------------------
    print("\nDeviation analysis:")
    analysis = validator.deviation_analysis(results)
    if isinstance(analysis, dict):
        for k, v in analysis.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    # ----- Summary report -----------------------------------------------------
    report = validator.summary_report(results)
    print(f"\n{report}")

    print("\nDone!")


if __name__ == "__main__":
    main()
