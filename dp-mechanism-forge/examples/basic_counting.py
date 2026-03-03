"""Basic counting query synthesis example.

Demonstrates:
  1. Defining a simple counting query
  2. Synthesizing an optimal mechanism with CEGIS
  3. Comparing with Laplace and Staircase baselines
  4. Visualizing the synthesized mechanism
  5. Printing a summary report
"""

from __future__ import annotations

import numpy as np

from dp_forge.types import (
    QuerySpec,
    MechanismFamily,
    LossFunction,
    SamplingConfig,
    SamplingMethod,
)
from dp_forge.cegis_loop import CEGISSynthesize, CEGISProgress
from dp_forge.baselines import LaplaceMechanism, StaircaseMechanism, GeometricMechanism
from dp_forge.extractor import extract_mechanism
from dp_forge.sampling import MechanismSampler
from dp_forge.verifier import verify_dp


def progress_callback(prog: CEGISProgress):
    """Print progress during synthesis."""
    print(
        f"  iter {prog.iteration:4d} | obj={prog.objective:.6f} | "
        f"witnesses={prog.n_witness_pairs:4d} | "
        f"violation={prog.violation_magnitude:.2e} | "
        f"time={prog.total_time:.1f}s"
    )


def run_basic_counting_example():
    """Synthesize an optimal mechanism for a single counting query."""
    # -----------------------------------------------------------------
    # 1. Define the counting query
    # -----------------------------------------------------------------
    print("=" * 60)
    print("DP-Forge: Basic Counting Query Example")
    print("=" * 60)

    epsilon = 1.0
    n = 100  # number of possible true values (databases)
    k = 100  # discretization of output space

    print(f"\nQuery: counting query with n={n} inputs")
    print(f"Privacy: ε={epsilon}, δ=0 (pure DP)")
    print(f"Discretization: k={k}")
    print()

    spec = QuerySpec.counting(n=n, epsilon=epsilon, delta=0.0, k=k)
    y_grid = np.linspace(spec.eta_min, 1.0 + spec.eta_min, k)

    # -----------------------------------------------------------------
    # 2. Synthesize with CEGIS
    # -----------------------------------------------------------------
    print("--- Synthesizing with CEGIS ---")
    result = CEGISSynthesize(
        spec,
        family=MechanismFamily.PIECEWISE_CONST,
        callback=progress_callback,
    )

    print(f"\nSynthesis complete:")
    print(f"  Iterations: {result.iterations}")
    print(f"  Objective (MSE): {result.obj_val:.6f}")
    print(f"  Converged: {result.converged}")

    if result.optimality_certificate:
        cert = result.optimality_certificate
        print(f"  Duality gap: {cert.duality_gap:.2e}")
        print(f"  Relative gap: {cert.relative_gap:.2e}")
        print(f"  Optimal: {cert.is_tight()}")

    # Extract deployable mechanism
    mechanism = extract_mechanism(result, spec)

    # -----------------------------------------------------------------
    # 3. Verify DP
    # -----------------------------------------------------------------
    print("\n--- Privacy Verification ---")
    dp_result = verify_dp(mechanism.p_final, epsilon=epsilon, delta=0.0)
    print(f"  DP verified: {dp_result.valid}")
    if not dp_result.valid:
        print(f"  Violation: {dp_result.violation}")

    # -----------------------------------------------------------------
    # 4. Compare with baselines
    # -----------------------------------------------------------------
    print("\n--- Baseline Comparison ---")

    laplace = LaplaceMechanism(epsilon, sensitivity=1.0)
    staircase = StaircaseMechanism(epsilon, sensitivity=1.0)
    geometric = GeometricMechanism(epsilon, sensitivity=1.0)

    # Estimate MSE via sampling
    sampler = MechanismSampler(
        mechanism, SamplingConfig(method=SamplingMethod.ALIAS, seed=42)
    )

    n_mc = 10_000
    cegis_se = []
    for i in range(n):
        samples = sampler.sample_mechanism(i, n_samples=n_mc)
        outputs = y_grid[samples]
        se = (outputs - spec.query_values[i]) ** 2
        cegis_se.append(float(np.mean(se)))
    mse_cegis = float(np.mean(cegis_se))

    mse_laplace = laplace.mse()
    mse_staircase = staircase.mse()
    mse_geometric = geometric.mse()

    print(f"  {'Mechanism':<20s} {'MSE':>12s} {'Ratio':>8s}")
    print("  " + "-" * 42)
    print(f"  {'CEGIS':<20s} {mse_cegis:12.6f} {1.0:8.4f}")
    print(
        f"  {'Laplace':<20s} {mse_laplace:12.6f} "
        f"{mse_laplace / mse_cegis:8.4f}" if mse_cegis > 0
        else f"  {'Laplace':<20s} {mse_laplace:12.6f} {'N/A':>8s}"
    )
    print(
        f"  {'Staircase':<20s} {mse_staircase:12.6f} "
        f"{mse_staircase / mse_cegis:8.4f}" if mse_cegis > 0
        else f"  {'Staircase':<20s} {mse_staircase:12.6f} {'N/A':>8s}"
    )
    print(
        f"  {'Geometric':<20s} {mse_geometric:12.6f} "
        f"{mse_geometric / mse_cegis:8.4f}" if mse_cegis > 0
        else f"  {'Geometric':<20s} {mse_geometric:12.6f} {'N/A':>8s}"
    )

    # -----------------------------------------------------------------
    # 5. Visualize (optional)
    # -----------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Plot mechanism heatmap
        ax = axes[0]
        im = ax.imshow(mechanism.p_final, aspect="auto", cmap="viridis")
        ax.set_xlabel("Output bin j")
        ax.set_ylabel("Input i")
        ax.set_title("CEGIS Mechanism P[i,j]")
        plt.colorbar(im, ax=ax)

        # Plot PDF for middle input
        ax = axes[1]
        mid = n // 2
        ax.bar(y_grid, mechanism.p_final[mid], width=y_grid[1] - y_grid[0],
               alpha=0.7, label="CEGIS")
        lap_disc, _ = laplace.discretize(k, y_grid=y_grid, center=spec.query_values[mid])
        ax.bar(y_grid, lap_disc, width=y_grid[1] - y_grid[0],
               alpha=0.4, label="Laplace")
        ax.set_xlabel("Output y")
        ax.set_ylabel("Probability")
        ax.set_title(f"PDF at input i={mid}")
        ax.legend()

        # Plot convergence
        ax = axes[2]
        if result.convergence_history:
            ax.plot(result.convergence_history, "b-")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Objective")
            ax.set_title("CEGIS Convergence")
            ax.set_yscale("log")

        fig.tight_layout()
        fig.savefig("basic_counting_example.pdf")
        print("\n  Figure saved to basic_counting_example.pdf")
    except ImportError:
        print("\n  (matplotlib not available, skipping visualization)")

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    improvement = (1.0 - mse_cegis / mse_laplace) * 100 if mse_laplace > 0 else 0
    print(f"CEGIS achieves {improvement:.1f}% MSE reduction vs Laplace")
    print("=" * 60)


if __name__ == "__main__":
    run_basic_counting_example()
