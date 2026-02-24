#!/usr/bin/env python3
"""Example: demonstrate the calibration pipeline at multiple widths.

Shows:
  1. Compute NTKs at several widths
  2. Run regression to extract 1/N coefficients
  3. Bootstrap confidence intervals
  4. Analyse residuals and convergence
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.kernel_engine import AnalyticNTK
from src.corrections import FiniteWidthCorrector
from src.calibration import (
    CalibrationConfig,
    CalibrationPipeline,
    CalibrationRegression,
    BootstrapCI,
)


def main():
    print("=" * 60)
    print("  Calibration Pipeline Demo")
    print("=" * 60)

    rng = np.random.RandomState(42)
    n_train, d_in = 30, 8
    X = rng.randn(n_train, d_in)
    print(f"\nData: n={n_train}, d={d_in}")

    # ----- 1. Compute NTKs at multiple widths --------------------------------
    print("\n[Step 1] Computing NTKs...")
    analytic = AnalyticNTK()
    widths = [32, 64, 128, 256, 512, 1024, 2048]
    ntk_data = {}
    for w in widths:
        K = analytic.compute(X, depth=2, width=w, activation="relu")
        ntk_data[w] = K
        trace = np.trace(K)
        cond = np.linalg.cond(K)
        print(f"  width={w:5d}:  tr(K)={trace:10.2f}  cond(K)={cond:.2e}")

    # ----- 2. Regression: extract 1/N coefficients --------------------------
    print("\n[Step 2] Regression fit...")
    reg = CalibrationRegression()
    reg_result = reg.fit(ntk_data)

    print(f"  θ₀ (infinite-width) norm: {np.linalg.norm(reg_result.theta_0):.4f}")
    if hasattr(reg_result, "theta_1") and reg_result.theta_1 is not None:
        print(f"  θ₁ (1/N correction) norm: {np.linalg.norm(reg_result.theta_1):.4f}")
        ratio = np.linalg.norm(reg_result.theta_1) / max(np.linalg.norm(reg_result.theta_0), 1e-10)
        print(f"  ||θ₁||/||θ₀|| ratio:      {ratio:.4e}")

    if hasattr(reg_result, "r_squared"):
        print(f"  R²: {reg_result.r_squared:.6f}")

    # ----- 3. Bootstrap confidence intervals ---------------------------------
    print("\n[Step 3] Bootstrap CI...")
    boot = BootstrapCI(n_samples=500, ci_level=0.95)
    boot_result = boot.compute(ntk_data)

    if hasattr(boot_result, "ci_lower") and boot_result.ci_lower is not None:
        ci_width = np.mean(np.abs(boot_result.ci_upper - boot_result.ci_lower))
        print(f"  Mean CI width: {ci_width:.4e}")
        print(f"  CI level: 95%")
    else:
        print("  Bootstrap results available")

    # ----- 4. Convergence analysis -------------------------------------------
    print("\n[Step 4] Convergence analysis...")
    corrector = FiniteWidthCorrector(order_max=2)
    theta_0_ref = ntk_data[widths[-1]]

    # Check how corrections change as we add more widths
    for n_widths in [3, 5, 7]:
        subset_widths = widths[:n_widths]
        subset_data = {w: ntk_data[w] for w in subset_widths}
        result = corrector.compute_corrections_regression(subset_data, theta_0=theta_0_ref)
        if result.theta_1 is not None:
            norm1 = np.linalg.norm(result.theta_1)
        else:
            norm1 = 0.0
        print(f"  {n_widths} widths ({subset_widths}): ||θ₁|| = {norm1:.4e}")

    # ----- 5. Residual analysis ----------------------------------------------
    print("\n[Step 5] Residual analysis...")
    if reg_result.theta_0 is not None:
        for w in widths:
            predicted = reg_result.theta_0
            if hasattr(reg_result, "theta_1") and reg_result.theta_1 is not None:
                predicted = predicted + reg_result.theta_1 / w
            residual = ntk_data[w] - predicted
            rel_resid = np.linalg.norm(residual, "fro") / np.linalg.norm(ntk_data[w], "fro")
            print(f"  width={w:5d}:  relative residual = {rel_resid:.4e}")

    # ----- 6. Pipeline end-to-end --------------------------------------------
    print("\n[Step 6] Full pipeline...")
    cfg = CalibrationConfig()
    pipeline = CalibrationPipeline(config=cfg)
    cal_result = pipeline.run_from_measurements(ntk_data)
    print(f"  Pipeline complete: {type(cal_result).__name__}")

    if hasattr(cal_result, "regression_result"):
        print(f"  Regression: {type(cal_result.regression_result).__name__}")
    if hasattr(cal_result, "bootstrap_result"):
        print(f"  Bootstrap:  {type(cal_result.bootstrap_result).__name__}")

    print("\nDone!")


if __name__ == "__main__":
    main()
