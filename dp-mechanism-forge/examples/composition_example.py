"""Sequential composition example.

Demonstrates:
  1. Synthesizing mechanisms for multiple queries
  2. Tracking privacy budget via composition theorems
  3. Comparing basic vs advanced composition
  4. Showing per-query and total MSE
"""

from __future__ import annotations

import numpy as np

from dp_forge.types import (
    QuerySpec,
    MechanismFamily,
    PrivacyBudget,
    SamplingConfig,
    SamplingMethod,
)
from dp_forge.cegis_loop import CEGISSynthesize
from dp_forge.baselines import LaplaceMechanism
from dp_forge.extractor import extract_mechanism
from dp_forge.sampling import MechanismSampler
from dp_forge.privacy_accounting import BasicComposition, AdvancedComposition
from dp_forge.verifier import verify_dp


def run_composition_example():
    """Demonstrate sequential composition of multiple DP mechanisms."""
    print("=" * 60)
    print("DP-Forge: Sequential Composition Example")
    print("=" * 60)

    n_queries = 5
    per_eps = 0.5
    n_inputs = 100
    k = 100
    n_mc = 10_000

    print(f"\n  Queries: {n_queries} independent counting queries")
    print(f"  Per-query ε: {per_eps}")
    print(f"  Inputs per query: {n_inputs}")
    print(f"  Discretization: k={k}")

    # =====================================================================
    # Part 1: Synthesize mechanisms for each query
    # =====================================================================
    print(f"\n--- Synthesizing {n_queries} mechanisms ---")

    cegis_mechanisms = []
    cegis_specs = []
    cegis_mses = []
    laplace_mses = []

    laplace = LaplaceMechanism(per_eps, sensitivity=1.0)

    for q in range(n_queries):
        print(f"\n  Query {q + 1}/{n_queries}:")
        spec = QuerySpec.counting(
            n=n_inputs, epsilon=per_eps, delta=0.0, k=k,
        )
        y_grid = np.linspace(spec.eta_min, 1.0 + spec.eta_min, k)

        result = CEGISSynthesize(
            spec, family=MechanismFamily.PIECEWISE_CONST,
        )
        mech = extract_mechanism(result, spec)

        dp_ok = verify_dp(mech.p_final, epsilon=per_eps, delta=0.0)
        print(f"    Iterations: {result.iterations}, "
              f"Objective: {result.obj_val:.6f}, "
              f"DP: {'✓' if dp_ok.valid else '✗'}")

        # Estimate MSE
        sampler = MechanismSampler(
            mech, SamplingConfig(method=SamplingMethod.ALIAS, seed=42 + q)
        )
        se_vals = []
        for i in range(n_inputs):
            samples = sampler.sample_mechanism(i, n_samples=n_mc)
            outputs = y_grid[samples]
            se_vals.append(float(np.mean((outputs - spec.query_values[i]) ** 2)))
        mse_q = float(np.mean(se_vals))
        cegis_mses.append(mse_q)
        laplace_mses.append(laplace.mse())

        cegis_mechanisms.append(mech)
        cegis_specs.append(spec)

        print(f"    MSE: {mse_q:.6f} (Laplace: {laplace.mse():.6f})")

    # =====================================================================
    # Part 2: Composition accounting
    # =====================================================================
    print("\n--- Privacy Budget Accounting ---")

    budgets = [PrivacyBudget(per_eps, 0.0)] * n_queries

    # Basic composition
    composed_basic = BasicComposition.sequential(budgets)
    print(f"\n  Basic composition:")
    print(f"    Total ε = {composed_basic.epsilon:.4f} "
          f"({n_queries} × {per_eps} = {n_queries * per_eps:.4f})")
    print(f"    Total δ = {composed_basic.delta}")

    # Advanced composition (for approximate DP)
    delta_prime = 1e-5
    try:
        composed_adv = AdvancedComposition.compose_homogeneous(
            epsilon=per_eps, delta=0.0, k=n_queries, delta_prime=delta_prime,
        )
        print(f"\n  Advanced composition (δ'={delta_prime}):")
        print(f"    Total ε = {composed_adv.epsilon:.4f}")
        print(f"    Total δ = {composed_adv.delta:.2e}")
        print(f"    Savings vs basic: "
              f"{(1 - composed_adv.epsilon / composed_basic.epsilon) * 100:.1f}%")
    except Exception as exc:
        print(f"\n  Advanced composition: {exc}")

    # =====================================================================
    # Part 3: Per-query and total MSE comparison
    # =====================================================================
    print("\n--- MSE Summary ---")
    total_cegis = sum(cegis_mses)
    total_laplace = sum(laplace_mses)

    print(f"\n  {'Query':>8s} {'CEGIS MSE':>12s} {'Laplace MSE':>12s} {'Ratio':>8s}")
    print("  " + "-" * 45)
    for q in range(n_queries):
        ratio = cegis_mses[q] / laplace_mses[q] if laplace_mses[q] > 0 else float("inf")
        print(f"  {q + 1:8d} {cegis_mses[q]:12.6f} {laplace_mses[q]:12.6f} {ratio:8.4f}")
    print("  " + "-" * 45)
    total_ratio = total_cegis / total_laplace if total_laplace > 0 else float("inf")
    print(f"  {'Total':>8s} {total_cegis:12.6f} {total_laplace:12.6f} {total_ratio:8.4f}")

    # =====================================================================
    # Visualize
    # =====================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Per-query MSE
        ax = axes[0]
        x = np.arange(n_queries)
        w = 0.35
        ax.bar(x - w / 2, cegis_mses, w, label="CEGIS", alpha=0.8)
        ax.bar(x + w / 2, laplace_mses, w, label="Laplace", alpha=0.8)
        ax.set_xlabel("Query")
        ax.set_ylabel("MSE")
        ax.set_title("Per-Query MSE")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Q{i + 1}" for i in range(n_queries)])
        ax.legend()

        # Budget comparison
        ax = axes[1]
        labels = ["Basic", "Advanced"]
        eps_vals = [composed_basic.epsilon]
        try:
            eps_vals.append(composed_adv.epsilon)
        except NameError:
            eps_vals.append(composed_basic.epsilon)
        ax.bar(labels, eps_vals, alpha=0.8, color=["tab:blue", "tab:green"])
        ax.set_ylabel("Total ε")
        ax.set_title("Composition Budget")
        ax.axhline(n_queries * per_eps, color="r", linestyle="--",
                    label=f"Sum = {n_queries * per_eps}")
        ax.legend()

        fig.suptitle(
            f"Sequential Composition ({n_queries} queries, per-ε={per_eps})",
            fontsize=14,
        )
        fig.tight_layout()
        fig.savefig("composition_example.pdf")
        print("\n  Figure saved to composition_example.pdf")
    except ImportError:
        print("\n  (matplotlib not available, skipping visualization)")

    # Summary
    print("\n" + "=" * 60)
    improvement = (1 - total_cegis / total_laplace) * 100 if total_laplace > 0 else 0
    print(f"Total MSE improvement: {improvement:.1f}% over Laplace")
    print(f"Privacy budget: ε={composed_basic.epsilon:.2f} (basic)")
    print("=" * 60)


if __name__ == "__main__":
    run_composition_example()
