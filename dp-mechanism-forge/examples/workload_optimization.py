"""Workload-aware mechanism optimization example.

Demonstrates:
  1. Defining prefix and all-range workloads
  2. Synthesizing optimal Gaussian mechanisms via SDP
  3. Comparing with the Matrix Mechanism baseline
  4. Visualizing noise covariance structure
"""

from __future__ import annotations

import numpy as np

from dp_forge.types import (
    QuerySpec,
    WorkloadSpec,
    MechanismFamily,
    LossFunction,
    SamplingConfig,
    SamplingMethod,
    QueryType,
)
from dp_forge.cegis_loop import CEGISSynthesize
from dp_forge.baselines import GaussianMechanism, MatrixMechanism
from dp_forge.workloads import WorkloadGenerator, WorkloadAnalyzer
from dp_forge.extractor import extract_mechanism
from dp_forge.sampling import MechanismSampler
from dp_forge.verifier import verify_dp


def run_workload_example():
    """Demonstrate workload-aware mechanism synthesis."""
    print("=" * 60)
    print("DP-Forge: Workload Optimization Example")
    print("=" * 60)

    d = 10
    epsilon = 1.0
    delta = 1e-5
    k = 100

    # =====================================================================
    # Part 1: Prefix (cumulative sum) workload
    # =====================================================================
    print(f"\n--- Part 1: Prefix Workload (d={d}) ---")

    A_prefix = WorkloadGenerator.prefix_sums(d)
    print(f"  Workload matrix: {A_prefix.shape[0]} queries × {A_prefix.shape[1]} bins")
    print(f"  L1 sensitivity: {WorkloadAnalyzer.l1_sensitivity(A_prefix):.2f}")
    print(f"  L2 sensitivity: {WorkloadAnalyzer.l2_sensitivity(A_prefix):.2f}")
    print(f"  Rank: {WorkloadAnalyzer.rank(A_prefix)}")
    print(f"  Condition number: {WorkloadAnalyzer.condition_number(A_prefix):.2f}")

    structure = WorkloadAnalyzer.detect_structure(A_prefix)
    if structure:
        print(f"  Detected structure: {structure}")

    sens = WorkloadAnalyzer.l1_sensitivity(A_prefix)
    n_queries = A_prefix.shape[0]
    query_values = np.arange(n_queries, dtype=np.float64)

    spec_prefix = QuerySpec(
        query_values=query_values,
        domain=f"prefix_d{d}",
        sensitivity=sens,
        epsilon=epsilon,
        delta=delta,
        k=k,
        loss_fn=LossFunction.L2,
        query_type=QueryType.RANGE,
        metadata={"workload": "prefix_sums"},
    )

    # Synthesize with SDP (workload-aware Gaussian family)
    print("\n  Synthesizing with SDP (Gaussian workload)...")
    try:
        result_prefix = CEGISSynthesize(
            spec_prefix,
            family=MechanismFamily.GAUSSIAN_WORKLOAD,
        )
        mech_prefix = extract_mechanism(result_prefix, spec_prefix)
        print(f"  Iterations: {result_prefix.iterations}")
        print(f"  Objective: {result_prefix.obj_val:.6f}")
    except Exception as exc:
        print(f"  SDP failed ({exc}), falling back to LP...")
        result_prefix = CEGISSynthesize(
            spec_prefix,
            family=MechanismFamily.PIECEWISE_CONST,
        )
        mech_prefix = extract_mechanism(result_prefix, spec_prefix)
        print(f"  Iterations: {result_prefix.iterations}")
        print(f"  Objective: {result_prefix.obj_val:.6f}")

    # Baselines
    matrix_mech = MatrixMechanism(A_prefix, epsilon=epsilon, delta=delta)
    mse_matrix = matrix_mech.mse()

    gauss = GaussianMechanism(epsilon, sensitivity=sens, delta=delta)
    mse_gauss_naive = gauss.mse() * n_queries

    # Estimate CEGIS MSE
    y_grid = np.linspace(0, 1000, k)
    sampler = MechanismSampler(
        mech_prefix, SamplingConfig(method=SamplingMethod.ALIAS, seed=42)
    )
    se_list = []
    for i in range(spec_prefix.n):
        samples = sampler.sample_mechanism(i, n_samples=10_000)
        outputs = y_grid[samples]
        se_list.append(float(np.mean((outputs - spec_prefix.query_values[i]) ** 2)))
    mse_cegis_prefix = float(np.mean(se_list))

    print(f"\n  MSE comparison:")
    print(f"    CEGIS:           {mse_cegis_prefix:.4f}")
    print(f"    Matrix Mechanism: {mse_matrix:.4f}")
    print(f"    Naive Gaussian:  {mse_gauss_naive:.4f}")

    # =====================================================================
    # Part 2: All-range workload
    # =====================================================================
    print(f"\n--- Part 2: All-Range Workload (d={d}) ---")

    A_range = WorkloadGenerator.all_range(d)
    n_range_queries = A_range.shape[0]
    print(f"  Workload matrix: {n_range_queries} queries × {A_range.shape[1]} bins")
    print(f"  L1 sensitivity: {WorkloadAnalyzer.l1_sensitivity(A_range):.2f}")
    print(f"  L2 sensitivity: {WorkloadAnalyzer.l2_sensitivity(A_range):.2f}")

    sens_range = WorkloadAnalyzer.l1_sensitivity(A_range)
    query_values_range = np.arange(n_range_queries, dtype=np.float64)

    spec_range = QuerySpec(
        query_values=query_values_range,
        domain=f"allrange_d{d}",
        sensitivity=sens_range,
        epsilon=epsilon,
        delta=delta,
        k=k,
        loss_fn=LossFunction.L2,
        query_type=QueryType.RANGE,
        metadata={"workload": "all_range"},
    )

    print("\n  Synthesizing...")
    try:
        result_range = CEGISSynthesize(
            spec_range,
            family=MechanismFamily.GAUSSIAN_WORKLOAD,
        )
        mech_range = extract_mechanism(result_range, spec_range)
    except Exception:
        result_range = CEGISSynthesize(
            spec_range,
            family=MechanismFamily.PIECEWISE_CONST,
        )
        mech_range = extract_mechanism(result_range, spec_range)

    print(f"  Iterations: {result_range.iterations}")
    print(f"  Objective: {result_range.obj_val:.6f}")

    # Baselines
    matrix_range = MatrixMechanism(A_range, epsilon=epsilon, delta=delta)
    mse_matrix_range = matrix_range.mse()

    gauss_range = GaussianMechanism(epsilon, sensitivity=sens_range, delta=delta)
    mse_gauss_range = gauss_range.mse() * n_range_queries

    sampler_range = MechanismSampler(
        mech_range, SamplingConfig(method=SamplingMethod.ALIAS, seed=42)
    )
    se_list = []
    for i in range(spec_range.n):
        samples = sampler_range.sample_mechanism(i, n_samples=10_000)
        outputs = y_grid[samples]
        se_list.append(float(np.mean((outputs - spec_range.query_values[i]) ** 2)))
    mse_cegis_range = float(np.mean(se_list))

    print(f"\n  MSE comparison:")
    print(f"    CEGIS:           {mse_cegis_range:.4f}")
    print(f"    Matrix Mechanism: {mse_matrix_range:.4f}")
    print(f"    Naive Gaussian:  {mse_gauss_range:.4f}")

    # =====================================================================
    # Visualize
    # =====================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Workload matrices
        ax = axes[0]
        ax.imshow(A_prefix, aspect="auto", cmap="Blues")
        ax.set_title("Prefix Workload")
        ax.set_xlabel("Data dimension")
        ax.set_ylabel("Query index")

        ax = axes[1]
        ax.imshow(A_range[:20, :], aspect="auto", cmap="Blues")
        ax.set_title(f"All-Range Workload (first 20/{n_range_queries})")
        ax.set_xlabel("Data dimension")
        ax.set_ylabel("Query index")

        # MSE comparison bar chart
        ax = axes[2]
        labels = ["Prefix", "All-Range"]
        cegis_vals = [mse_cegis_prefix, mse_cegis_range]
        matrix_vals = [mse_matrix, mse_matrix_range]
        x = np.arange(len(labels))
        w = 0.3
        ax.bar(x - w, cegis_vals, w, label="CEGIS", alpha=0.8)
        ax.bar(x, matrix_vals, w, label="Matrix", alpha=0.8)
        ax.set_ylabel("MSE")
        ax.set_title("MSE Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        fig.suptitle(f"Workload Optimization (d={d}, ε={epsilon})", fontsize=14)
        fig.tight_layout()
        fig.savefig("workload_optimization.pdf")
        print("\n  Figure saved to workload_optimization.pdf")
    except ImportError:
        print("\n  (matplotlib not available, skipping visualization)")

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "=" * 60)
    print("Summary:")
    r1 = mse_cegis_prefix / mse_matrix if mse_matrix > 0 else float("inf")
    r2 = mse_cegis_range / mse_matrix_range if mse_matrix_range > 0 else float("inf")
    print(f"  Prefix:    CEGIS/Matrix ratio = {r1:.3f}")
    print(f"  All-Range: CEGIS/Matrix ratio = {r2:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    run_workload_example()
