"""Histogram mechanism synthesis example.

Demonstrates:
  1. Defining a histogram query over d bins
  2. Synthesizing for both pure DP (δ=0) and approximate DP (δ>0)
  3. Comparing with Gaussian and Laplace baselines
  4. Visualizing noise distributions per bin
"""

from __future__ import annotations

import numpy as np

from dp_forge.types import (
    QuerySpec,
    MechanismFamily,
    SamplingConfig,
    SamplingMethod,
)
from dp_forge.cegis_loop import CEGISSynthesize, CEGISProgress
from dp_forge.baselines import LaplaceMechanism, GaussianMechanism
from dp_forge.extractor import extract_mechanism
from dp_forge.sampling import MechanismSampler
from dp_forge.verifier import verify_dp


def run_histogram_example():
    """Synthesize optimal histogram mechanisms under pure and approximate DP."""
    print("=" * 60)
    print("DP-Forge: Histogram Mechanism Synthesis")
    print("=" * 60)

    d = 5  # number of bins
    epsilon = 1.0
    k = 100

    # =====================================================================
    # Part 1: Pure DP (δ = 0)
    # =====================================================================
    print(f"\n--- Part 1: Pure DP (ε={epsilon}, δ=0) ---")
    print(f"Histogram with d={d} bins\n")

    spec_pure = QuerySpec.histogram(n_bins=d, epsilon=epsilon, delta=0.0, k=k)
    y_grid = np.linspace(0, 1000, k)

    print("Synthesizing...")
    result_pure = CEGISSynthesize(
        spec_pure,
        family=MechanismFamily.PIECEWISE_CONST,
    )
    mech_pure = extract_mechanism(result_pure, spec_pure)

    dp_ok = verify_dp(mech_pure.p_final, epsilon=epsilon, delta=0.0)
    print(f"  Iterations: {result_pure.iterations}")
    print(f"  Objective: {result_pure.obj_val:.6f}")
    print(f"  DP verified: {dp_ok.valid}")

    # MSE comparison
    sampler = MechanismSampler(
        mech_pure, SamplingConfig(method=SamplingMethod.ALIAS, seed=42)
    )
    n_mc = 10_000
    se_per_bin = []
    for i in range(d):
        samples = sampler.sample_mechanism(i, n_samples=n_mc)
        outputs = y_grid[samples]
        se = (outputs - spec_pure.query_values[i]) ** 2
        se_per_bin.append(float(np.mean(se)))
    mse_cegis_pure = float(np.mean(se_per_bin))

    laplace = LaplaceMechanism(epsilon, sensitivity=1.0)
    mse_laplace = laplace.mse() * d  # total MSE across all bins

    print(f"\n  MSE comparison (total across {d} bins):")
    print(f"    CEGIS:   {mse_cegis_pure:.4f}")
    print(f"    Laplace: {mse_laplace:.4f}")
    if mse_laplace > 0:
        print(f"    Ratio:   {mse_cegis_pure / mse_laplace:.4f}")

    # =====================================================================
    # Part 2: Approximate DP (δ = 1e-5)
    # =====================================================================
    delta = 1e-5
    print(f"\n--- Part 2: Approximate DP (ε={epsilon}, δ={delta}) ---")
    print(f"Histogram with d={d} bins\n")

    spec_approx = QuerySpec.histogram(
        n_bins=d, epsilon=epsilon, delta=delta, k=k,
    )

    print("Synthesizing...")
    result_approx = CEGISSynthesize(
        spec_approx,
        family=MechanismFamily.PIECEWISE_CONST,
    )
    mech_approx = extract_mechanism(result_approx, spec_approx)

    dp_ok = verify_dp(mech_approx.p_final, epsilon=epsilon, delta=delta)
    print(f"  Iterations: {result_approx.iterations}")
    print(f"  Objective: {result_approx.obj_val:.6f}")
    print(f"  DP verified: {dp_ok.valid}")

    # MSE comparison with Gaussian
    sampler_approx = MechanismSampler(
        mech_approx, SamplingConfig(method=SamplingMethod.ALIAS, seed=42)
    )
    se_per_bin = []
    for i in range(d):
        samples = sampler_approx.sample_mechanism(i, n_samples=n_mc)
        outputs = y_grid[samples]
        se = (outputs - spec_approx.query_values[i]) ** 2
        se_per_bin.append(float(np.mean(se)))
    mse_cegis_approx = float(np.mean(se_per_bin))

    gaussian = GaussianMechanism(epsilon, sensitivity=1.0, delta=delta)
    mse_gaussian = gaussian.mse() * d

    print(f"\n  MSE comparison (total across {d} bins):")
    print(f"    CEGIS:    {mse_cegis_approx:.4f}")
    print(f"    Gaussian: {mse_gaussian:.4f}")
    print(f"    Laplace:  {mse_laplace:.4f}")
    if mse_gaussian > 0:
        print(f"    CEGIS/Gaussian ratio: {mse_cegis_approx / mse_gaussian:.4f}")

    # =====================================================================
    # Visualize
    # =====================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # Row 1: Pure DP mechanism
        for i in range(min(d, 3)):
            ax = axes[0, i]
            ax.bar(y_grid, mech_pure.p_final[i],
                   width=y_grid[1] - y_grid[0], alpha=0.7, label="CEGIS")
            lap_disc, _ = laplace.discretize(k, y_grid=y_grid,
                                              center=spec_pure.query_values[i])
            ax.bar(y_grid, lap_disc, width=y_grid[1] - y_grid[0],
                   alpha=0.4, label="Laplace")
            ax.set_title(f"Pure DP: Bin {i}")
            ax.set_xlabel("Output")
            ax.set_ylabel("Prob")
            ax.legend(fontsize=8)

        # Row 2: Approximate DP mechanism
        for i in range(min(d, 3)):
            ax = axes[1, i]
            ax.bar(y_grid, mech_approx.p_final[i],
                   width=y_grid[1] - y_grid[0], alpha=0.7, label="CEGIS")
            gauss_disc, _ = gaussian.discretize(k, y_grid=y_grid,
                                                 center=spec_approx.query_values[i])
            ax.bar(y_grid, gauss_disc, width=y_grid[1] - y_grid[0],
                   alpha=0.4, label="Gaussian")
            ax.set_title(f"Approx DP: Bin {i}")
            ax.set_xlabel("Output")
            ax.set_ylabel("Prob")
            ax.legend(fontsize=8)

        fig.suptitle(
            f"Histogram Synthesis (d={d}, ε={epsilon})", fontsize=14
        )
        fig.tight_layout()
        fig.savefig("histogram_synthesis.pdf")
        print("\n  Figure saved to histogram_synthesis.pdf")
    except ImportError:
        print("\n  (matplotlib not available, skipping visualization)")

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Pure DP:   CEGIS MSE = {mse_cegis_pure:.4f} "
          f"(Laplace = {mse_laplace:.4f})")
    print(f"  Approx DP: CEGIS MSE = {mse_cegis_approx:.4f} "
          f"(Gaussian = {mse_gaussian:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    run_histogram_example()
