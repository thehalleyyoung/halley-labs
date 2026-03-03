#!/usr/bin/env python3
"""
DP-Forge benchmark suite: CEGIS-synthesized mechanisms vs. baselines.

Experiments:
  1. Counting query across ε and domain size n
  2. Histogram query (multi-bin)
  3. Staircase comparison (optimal pure-DP baseline)
  4. Approximate DP (δ > 0) with Gaussian baseline
  5. Scalability: synthesis time vs problem size

Outputs results to experiments/benchmark_results.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np

# Ensure dp_forge is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dp_forge.types import QuerySpec, MechanismFamily, LossFunction
from dp_forge.cegis_loop import CEGISSynthesize
from dp_forge.baselines import (
    LaplaceMechanism,
    StaircaseMechanism,
    GaussianMechanism,
    GeometricMechanism,
)
from dp_forge.lp_builder import build_output_grid
from dp_forge.verifier import verify


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_mse(P: np.ndarray, query_values: np.ndarray, y_grid: np.ndarray):
    """Compute worst-case and average MSE for mechanism table P."""
    n = P.shape[0]
    mse_per = np.array([
        np.sum(P[i] * (query_values[i] - y_grid) ** 2) for i in range(n)
    ])
    return float(np.max(mse_per)), float(np.mean(mse_per))


def compute_mae(P: np.ndarray, query_values: np.ndarray, y_grid: np.ndarray):
    """Compute worst-case and average MAE for mechanism table P."""
    n = P.shape[0]
    mae_per = np.array([
        np.sum(P[i] * np.abs(query_values[i] - y_grid)) for i in range(n)
    ])
    return float(np.max(mae_per)), float(np.mean(mae_per))


def safe_ratio(baseline: float, ours: float) -> float:
    if ours <= 0:
        return float("inf")
    return round(baseline / ours, 4)


# ---------------------------------------------------------------------------
# Experiment 1: Counting query — vary ε and n
# ---------------------------------------------------------------------------

def experiment_counting(max_iter: int = 200) -> List[dict]:
    """Sweep ε × n for counting queries; compare CEGIS vs Laplace/Staircase."""
    print("\n" + "=" * 60)
    print("Experiment 1: Counting Query — ε × n sweep")
    print("=" * 60)

    results = []
    epsilons = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    ns = [5, 10, 20, 50]

    for eps in epsilons:
        for n in ns:
            k = max(n, 20)
            spec = QuerySpec.counting(n=n, epsilon=eps, delta=0.0, k=k)
            lap = LaplaceMechanism(epsilon=eps, sensitivity=1.0)
            stair = StaircaseMechanism(epsilon=eps, sensitivity=1.0)

            t0 = time.time()
            try:
                result = CEGISSynthesize(
                    spec, family=MechanismFamily.PIECEWISE_CONST, max_iter=max_iter
                )
                elapsed = time.time() - t0

                y_grid = build_output_grid(spec.query_values, spec.k)
                P = result.mechanism
                worst_mse, avg_mse = compute_mse(P, spec.query_values, y_grid)
                worst_mae, avg_mae = compute_mae(P, spec.query_values, y_grid)

                # Verify DP
                tol = max(1e-6, np.exp(eps) * 1e-8 * 2.5)
                vr = verify(P, spec.epsilon, spec.delta, spec.edges, tol=tol)

                entry = {
                    "experiment": "counting",
                    "epsilon": eps,
                    "n": n,
                    "k": k,
                    "iterations": result.iterations,
                    "converged": result.converged,
                    "lp_objective": round(result.obj_val, 6),
                    "dp_verified": vr.valid,
                    "cegis_worst_mse": round(worst_mse, 6),
                    "cegis_avg_mse": round(avg_mse, 6),
                    "cegis_worst_mae": round(worst_mae, 6),
                    "cegis_avg_mae": round(avg_mae, 6),
                    "laplace_mse": round(lap.mse(), 6),
                    "laplace_mae": round(lap.mae(), 6),
                    "staircase_mse": round(stair.mse(), 6),
                    "staircase_mae": round(stair.mae(), 6),
                    "improvement_vs_laplace": safe_ratio(lap.mse(), worst_mse),
                    "improvement_vs_staircase": safe_ratio(stair.mse(), worst_mse),
                    "synthesis_time_s": round(elapsed, 3),
                }
                results.append(entry)
                print(
                    f"  ε={eps:5.2f} n={n:3d}: MSE={worst_mse:.4f}  "
                    f"Lap/CEGIS={entry['improvement_vs_laplace']:.2f}×  "
                    f"Stair/CEGIS={entry['improvement_vs_staircase']:.4f}×  "
                    f"t={elapsed:.2f}s"
                )
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  ε={eps:5.2f} n={n:3d}: FAILED ({elapsed:.1f}s) — {str(e)[:60]}")
                results.append({
                    "experiment": "counting",
                    "epsilon": eps,
                    "n": n,
                    "k": k,
                    "converged": False,
                    "error": str(e)[:120],
                    "synthesis_time_s": round(elapsed, 3),
                })

    return results


# ---------------------------------------------------------------------------
# Experiment 2: Histogram query
# ---------------------------------------------------------------------------

def experiment_histogram(max_iter: int = 200) -> List[dict]:
    """Histogram queries with multiple bins."""
    print("\n" + "=" * 60)
    print("Experiment 2: Histogram Query")
    print("=" * 60)

    results = []
    epsilons = [0.5, 1.0, 2.0]
    n_values = [5, 10, 20]

    for eps in epsilons:
        for n in n_values:
            k = max(n, 20)
            # Histogram: query_values are bin counts
            spec = QuerySpec.counting(n=n, epsilon=eps, delta=0.0, k=k)
            lap = LaplaceMechanism(epsilon=eps, sensitivity=1.0)

            t0 = time.time()
            try:
                result = CEGISSynthesize(
                    spec, family=MechanismFamily.PIECEWISE_CONST, max_iter=max_iter
                )
                elapsed = time.time() - t0

                y_grid = build_output_grid(spec.query_values, spec.k)
                P = result.mechanism
                worst_mse, avg_mse = compute_mse(P, spec.query_values, y_grid)

                entry = {
                    "experiment": "histogram",
                    "epsilon": eps,
                    "n_bins": n,
                    "k": k,
                    "iterations": result.iterations,
                    "converged": result.converged,
                    "cegis_worst_mse": round(worst_mse, 6),
                    "cegis_avg_mse": round(avg_mse, 6),
                    "laplace_mse": round(lap.mse(), 6),
                    "improvement_vs_laplace": safe_ratio(lap.mse(), worst_mse),
                    "synthesis_time_s": round(elapsed, 3),
                }
                results.append(entry)
                print(
                    f"  ε={eps:4.1f} bins={n:2d}: MSE={worst_mse:.4f}  "
                    f"Lap/CEGIS={entry['improvement_vs_laplace']:.2f}×  t={elapsed:.2f}s"
                )
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  ε={eps:4.1f} bins={n:2d}: FAILED ({elapsed:.1f}s)")
                results.append({
                    "experiment": "histogram",
                    "epsilon": eps,
                    "n_bins": n,
                    "k": k,
                    "converged": False,
                    "error": str(e)[:120],
                    "synthesis_time_s": round(elapsed, 3),
                })

    return results


# ---------------------------------------------------------------------------
# Experiment 3: Loss function comparison (L1 vs L2)
# ---------------------------------------------------------------------------

def experiment_loss_functions(max_iter: int = 200) -> List[dict]:
    """Compare L1 and L2 loss objectives."""
    print("\n" + "=" * 60)
    print("Experiment 3: L1 vs L2 Loss Functions")
    print("=" * 60)

    results = []
    epsilons = [0.5, 1.0, 2.0]
    n = 10
    k = 20

    for eps in epsilons:
        for loss_fn in [LossFunction.L1, LossFunction.L2]:
            spec = QuerySpec(
                query_values=np.arange(n, dtype=np.float64),
                domain=f"counting({n})",
                sensitivity=1.0,
                epsilon=eps,
                delta=0.0,
                k=k,
                loss_fn=loss_fn,
            )
            lap = LaplaceMechanism(epsilon=eps, sensitivity=1.0)

            t0 = time.time()
            try:
                result = CEGISSynthesize(
                    spec, family=MechanismFamily.PIECEWISE_CONST, max_iter=max_iter
                )
                elapsed = time.time() - t0

                y_grid = build_output_grid(spec.query_values, spec.k)
                P = result.mechanism
                worst_mse, avg_mse = compute_mse(P, spec.query_values, y_grid)
                worst_mae, avg_mae = compute_mae(P, spec.query_values, y_grid)

                entry = {
                    "experiment": "loss_comparison",
                    "epsilon": eps,
                    "loss": loss_fn.name,
                    "n": n,
                    "k": k,
                    "iterations": result.iterations,
                    "converged": result.converged,
                    "cegis_worst_mse": round(worst_mse, 6),
                    "cegis_worst_mae": round(worst_mae, 6),
                    "laplace_mse": round(lap.mse(), 6),
                    "laplace_mae": round(lap.mae(), 6),
                    "synthesis_time_s": round(elapsed, 3),
                }
                results.append(entry)
                print(
                    f"  ε={eps:4.1f} loss={loss_fn.name:3s}: MSE={worst_mse:.4f} "
                    f"MAE={worst_mae:.4f}  t={elapsed:.2f}s"
                )
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  ε={eps:4.1f} loss={loss_fn.name:3s}: FAILED ({elapsed:.1f}s)")
                results.append({
                    "experiment": "loss_comparison",
                    "epsilon": eps,
                    "loss": loss_fn.name,
                    "converged": False,
                    "error": str(e)[:120],
                    "synthesis_time_s": round(elapsed, 3),
                })

    return results


# ---------------------------------------------------------------------------
# Experiment 4: Approximate DP (δ > 0)
# ---------------------------------------------------------------------------

def experiment_approx_dp(max_iter: int = 200) -> List[dict]:
    """Test (ε,δ)-DP with Gaussian baseline."""
    print("\n" + "=" * 60)
    print("Experiment 4: Approximate DP (δ > 0)")
    print("=" * 60)

    results = []
    configs = [
        (0.5, 1e-5, 10, 20),
        (1.0, 1e-5, 10, 20),
        (1.0, 1e-3, 10, 20),
        (2.0, 1e-5, 10, 20),
        (0.5, 1e-5, 20, 30),
        (1.0, 1e-5, 20, 30),
    ]

    for eps, delta, n, k in configs:
        spec = QuerySpec.counting(n=n, epsilon=eps, delta=delta, k=k)
        gauss = GaussianMechanism(epsilon=eps, sensitivity=1.0, delta=delta)
        lap = LaplaceMechanism(epsilon=eps, sensitivity=1.0)

        t0 = time.time()
        try:
            result = CEGISSynthesize(
                spec, family=MechanismFamily.PIECEWISE_CONST, max_iter=max_iter
            )
            elapsed = time.time() - t0

            y_grid = build_output_grid(spec.query_values, spec.k)
            P = result.mechanism
            worst_mse, avg_mse = compute_mse(P, spec.query_values, y_grid)

            entry = {
                "experiment": "approx_dp",
                "epsilon": eps,
                "delta": delta,
                "n": n,
                "k": k,
                "iterations": result.iterations,
                "converged": result.converged,
                "cegis_worst_mse": round(worst_mse, 6),
                "cegis_avg_mse": round(avg_mse, 6),
                "gaussian_mse": round(gauss.mse(), 6),
                "laplace_mse": round(lap.mse(), 6),
                "improvement_vs_gaussian": safe_ratio(gauss.mse(), worst_mse),
                "improvement_vs_laplace": safe_ratio(lap.mse(), worst_mse),
                "synthesis_time_s": round(elapsed, 3),
            }
            results.append(entry)
            print(
                f"  ε={eps:4.1f} δ={delta:.0e} n={n:2d}: MSE={worst_mse:.4f}  "
                f"Gauss/CEGIS={entry['improvement_vs_gaussian']:.2f}×  t={elapsed:.2f}s"
            )
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ε={eps:4.1f} δ={delta:.0e} n={n:2d}: FAILED ({elapsed:.1f}s)")
            results.append({
                "experiment": "approx_dp",
                "epsilon": eps,
                "delta": delta,
                "n": n,
                "k": k,
                "converged": False,
                "error": str(e)[:120],
                "synthesis_time_s": round(elapsed, 3),
            })

    return results


# ---------------------------------------------------------------------------
# Experiment 5: Scalability
# ---------------------------------------------------------------------------

def experiment_scalability(max_iter: int = 300) -> List[dict]:
    """Measure synthesis time and quality vs problem size."""
    print("\n" + "=" * 60)
    print("Experiment 5: Scalability (time vs. problem size)")
    print("=" * 60)

    results = []
    eps = 1.0
    sizes = [5, 10, 20, 50]

    for n in sizes:
        k = n
        spec = QuerySpec.counting(n=n, epsilon=eps, delta=0.0, k=k)
        lap = LaplaceMechanism(epsilon=eps, sensitivity=1.0)

        t0 = time.time()
        try:
            result = CEGISSynthesize(
                spec, family=MechanismFamily.PIECEWISE_CONST, max_iter=max_iter
            )
            elapsed = time.time() - t0

            y_grid = build_output_grid(spec.query_values, spec.k)
            P = result.mechanism
            worst_mse, avg_mse = compute_mse(P, spec.query_values, y_grid)

            entry = {
                "experiment": "scalability",
                "epsilon": eps,
                "n": n,
                "k": k,
                "n_variables": n * k + 1,
                "n_edges": len(spec.edges.edges),
                "iterations": result.iterations,
                "converged": result.converged,
                "cegis_worst_mse": round(worst_mse, 6),
                "laplace_mse": round(lap.mse(), 6),
                "improvement_vs_laplace": safe_ratio(lap.mse(), worst_mse),
                "synthesis_time_s": round(elapsed, 3),
            }
            results.append(entry)
            print(
                f"  n={n:3d} k={k:3d} vars={n*k+1:5d}: MSE={worst_mse:.4f}  "
                f"Lap/CEGIS={entry['improvement_vs_laplace']:.2f}×  t={elapsed:.2f}s"
            )
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  n={n:3d}: FAILED ({elapsed:.1f}s)")
            results.append({
                "experiment": "scalability",
                "n": n,
                "k": k,
                "converged": False,
                "error": str(e)[:120],
                "synthesis_time_s": round(elapsed, 3),
            })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("DP-Forge Benchmark Suite")
    print("========================")
    t_start = time.time()

    all_results = {
        "metadata": {
            "tool": "DP-Forge",
            "version": "0.1.0",
            "description": "CEGIS-based optimal DP mechanism synthesis benchmarks",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "counting": experiment_counting(),
        "histogram": experiment_histogram(),
        "loss_comparison": experiment_loss_functions(),
        "approx_dp": experiment_approx_dp(),
        "scalability": experiment_scalability(),
    }

    # Summary statistics
    counting = [r for r in all_results["counting"] if r.get("converged")]
    if counting:
        improvements = [r["improvement_vs_laplace"] for r in counting]
        all_results["summary"] = {
            "total_experiments": sum(
                len(v) for k, v in all_results.items()
                if isinstance(v, list)
            ),
            "converged": sum(
                1 for k, v in all_results.items()
                if isinstance(v, list)
                for r in v if r.get("converged")
            ),
            "counting_median_improvement_vs_laplace": round(
                float(np.median(improvements)), 2
            ),
            "counting_max_improvement_vs_laplace": round(max(improvements), 2),
            "counting_geomean_improvement_vs_laplace": round(
                float(np.exp(np.mean(np.log([
                    max(x, 1.0) for x in improvements
                ])))), 2
            ),
            "total_time_s": round(time.time() - t_start, 1),
        }

    outpath = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Results saved to {outpath}")
    print(f"Total time: {time.time() - t_start:.1f}s")

    if "summary" in all_results:
        s = all_results["summary"]
        print(f"Converged: {s['converged']}/{s['total_experiments']}")
        print(f"Median Lap/CEGIS improvement: {s['counting_median_improvement_vs_laplace']}×")
        print(f"Max improvement: {s['counting_max_improvement_vs_laplace']}×")


if __name__ == "__main__":
    main()
