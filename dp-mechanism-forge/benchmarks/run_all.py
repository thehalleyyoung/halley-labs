#!/usr/bin/env python3
"""
DP-Forge comprehensive benchmark suite.

Tiered benchmarks for evaluating CEGIS mechanism synthesis:
  Tier 1 — Quick smoke tests (< 5s)
  Tier 2 — Standard configurations (< 30s)
  Tier 3 — Extended parameter sweeps (< 5min)
  Tier 4 — Stress tests and scalability (< 30min)

Usage:
    python benchmarks/run_all.py --tier 1
    python benchmarks/run_all.py --tier all --output results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

from dp_forge.types import QuerySpec, LossFunction, MechanismFamily
from dp_forge.cegis_loop import CEGISSynthesize
from dp_forge.baselines import (
    LaplaceMechanism,
    StaircaseMechanism,
    GaussianMechanism,
    GeometricMechanism,
)
from dp_forge.lp_builder import build_output_grid
from dp_forge.verifier import quick_verify


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def worst_case_mse(P, qv, ygrid):
    n = P.shape[0]
    return float(np.max([np.sum(P[i] * (qv[i] - ygrid) ** 2) for i in range(n)]))


def avg_mse(P, qv, ygrid):
    n = P.shape[0]
    return float(np.mean([np.sum(P[i] * (qv[i] - ygrid) ** 2) for i in range(n)]))


def worst_case_mae(P, qv, ygrid):
    n = P.shape[0]
    return float(np.max([np.sum(P[i] * np.abs(qv[i] - ygrid)) for i in range(n)]))


def safe_ratio(baseline: float, ours: float) -> float:
    if ours <= 0:
        return float("inf")
    return round(baseline / ours, 4)


def run_single(spec, max_iter=300):
    """Run CEGIS synthesis on a single spec; return (result, elapsed, P, ygrid)."""
    t0 = time.time()
    result = CEGISSynthesize(spec, max_iter=max_iter)
    elapsed = time.time() - t0
    ygrid = build_output_grid(spec.query_values, spec.k)
    P = result.mechanism
    return result, elapsed, P, ygrid


# ---------------------------------------------------------------------------
# Tier 1: Quick smoke tests
# ---------------------------------------------------------------------------

def tier1_counting_basic():
    """Basic counting queries at standard epsilons."""
    print("  [Tier 1] Counting queries — basic")
    results = []
    for eps in [0.5, 1.0, 2.0]:
        for n in [2, 5, 10]:
            k = max(n, 30)
            spec = QuerySpec.counting(n=n, epsilon=eps, delta=0.0, k=k)
            result, elapsed, P, ygrid = run_single(spec)
            wmse = worst_case_mse(P, spec.query_values, ygrid)
            lap_mse = LaplaceMechanism(epsilon=eps, sensitivity=1.0).mse()
            stair_mse = StaircaseMechanism(epsilon=eps, sensitivity=1.0).mse()
            verified = quick_verify(P, epsilon=eps, delta=0.0)
            entry = {
                "experiment": "counting_basic",
                "epsilon": eps, "n": n, "k": k,
                "cegis_mse": round(wmse, 6),
                "laplace_mse": round(lap_mse, 6),
                "staircase_mse": round(stair_mse, 6),
                "vs_laplace": safe_ratio(lap_mse, wmse),
                "vs_staircase": safe_ratio(stair_mse, wmse),
                "dp_verified": verified,
                "iterations": result.iterations,
                "time_s": round(elapsed, 4),
            }
            results.append(entry)
            print(f"    ε={eps:.1f} n={n:2d}: MSE={wmse:.4f}  Lap/C={entry['vs_laplace']:.2f}×  t={elapsed:.3f}s")
    return results


def tier1_approx_dp():
    """Quick approximate DP tests."""
    print("  [Tier 1] Approximate DP")
    results = []
    for eps, delta, n in [(1.0, 1e-5, 5), (0.5, 1e-5, 10)]:
        k = 30
        spec = QuerySpec.counting(n=n, epsilon=eps, delta=delta, k=k)
        result, elapsed, P, ygrid = run_single(spec)
        wmse = worst_case_mse(P, spec.query_values, ygrid)
        gauss_mse = GaussianMechanism(epsilon=eps, delta=delta, sensitivity=1.0).mse()
        entry = {
            "experiment": "approx_dp_basic",
            "epsilon": eps, "delta": delta, "n": n, "k": k,
            "cegis_mse": round(wmse, 6),
            "gaussian_mse": round(gauss_mse, 6),
            "vs_gaussian": safe_ratio(gauss_mse, wmse),
            "time_s": round(elapsed, 4),
        }
        results.append(entry)
        print(f"    ε={eps:.1f} δ={delta:.0e} n={n:2d}: MSE={wmse:.4f}  Gauss/C={entry['vs_gaussian']:.1f}×")
    return results


# ---------------------------------------------------------------------------
# Tier 2: Standard configurations
# ---------------------------------------------------------------------------

def tier2_counting_sweep():
    """Full ε × n sweep for counting queries."""
    print("  [Tier 2] Counting — ε × n sweep")
    results = []
    epsilons = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    ns = [2, 5, 10, 20]
    for eps in epsilons:
        for n in ns:
            k = max(n, 30)
            spec = QuerySpec.counting(n=n, epsilon=eps, delta=0.0, k=k)
            result, elapsed, P, ygrid = run_single(spec)
            wmse = worst_case_mse(P, spec.query_values, ygrid)
            amse = avg_mse(P, spec.query_values, ygrid)
            wmae = worst_case_mae(P, spec.query_values, ygrid)
            lap = LaplaceMechanism(epsilon=eps, sensitivity=1.0)
            stair = StaircaseMechanism(epsilon=eps, sensitivity=1.0)
            entry = {
                "experiment": "counting_sweep",
                "epsilon": eps, "n": n, "k": k,
                "cegis_worst_mse": round(wmse, 6),
                "cegis_avg_mse": round(amse, 6),
                "cegis_worst_mae": round(wmae, 6),
                "laplace_mse": round(lap.mse(), 6),
                "laplace_mae": round(lap.mae(), 6),
                "staircase_mse": round(stair.mse(), 6),
                "vs_laplace": safe_ratio(lap.mse(), wmse),
                "vs_staircase": safe_ratio(stair.mse(), wmse),
                "iterations": result.iterations,
                "converged": result.converged,
                "time_s": round(elapsed, 4),
            }
            results.append(entry)
            print(f"    ε={eps:5.2f} n={n:2d}: MSE={wmse:.4f}  Lap/C={entry['vs_laplace']:.2f}×  t={elapsed:.3f}s")
    return results


def tier2_approx_dp_sweep():
    """Comprehensive approximate DP sweep."""
    print("  [Tier 2] Approximate DP — comprehensive")
    results = []
    configs = [
        (0.1, 1e-5, 5), (0.5, 1e-5, 5), (1.0, 1e-5, 5),
        (0.1, 1e-3, 10), (0.5, 1e-3, 10), (1.0, 1e-3, 10),
        (0.5, 1e-5, 20), (1.0, 1e-5, 20),
    ]
    for eps, delta, n in configs:
        k = max(n, 30)
        spec = QuerySpec.counting(n=n, epsilon=eps, delta=delta, k=k)
        result, elapsed, P, ygrid = run_single(spec)
        wmse = worst_case_mse(P, spec.query_values, ygrid)
        gauss = GaussianMechanism(epsilon=eps, delta=delta, sensitivity=1.0)
        lap = LaplaceMechanism(epsilon=eps, sensitivity=1.0)
        entry = {
            "experiment": "approx_dp_sweep",
            "epsilon": eps, "delta": delta, "n": n, "k": k,
            "cegis_mse": round(wmse, 6),
            "gaussian_mse": round(gauss.mse(), 6),
            "laplace_mse": round(lap.mse(), 6),
            "vs_gaussian": safe_ratio(gauss.mse(), wmse),
            "vs_laplace": safe_ratio(lap.mse(), wmse),
            "time_s": round(elapsed, 4),
        }
        results.append(entry)
        print(f"    ε={eps:.1f} δ={delta:.0e} n={n:2d}: CEGIS={wmse:.4f}  Gauss/C={entry['vs_gaussian']:.1f}×")
    return results


def tier2_query_types():
    """Range, sum, and median queries."""
    print("  [Tier 2] Non-trivial query types")
    results = []

    # Range queries
    for n in [4, 8, 16]:
        qv = np.arange(n, dtype=float)
        k = max(n, 30)
        spec = QuerySpec(query_values=qv, domain=f"range({n})", sensitivity=1.0, epsilon=1.0, k=k)
        result, elapsed, P, ygrid = run_single(spec)
        wmse = worst_case_mse(P, qv, ygrid)
        lap_mse = LaplaceMechanism(epsilon=1.0, sensitivity=1.0).mse()
        results.append({
            "experiment": "range_query", "n": n, "k": k,
            "cegis_mse": round(wmse, 6), "laplace_mse": round(lap_mse, 6),
            "vs_laplace": safe_ratio(lap_mse, wmse),
            "time_s": round(elapsed, 4),
        })
        print(f"    Range n={n:2d}: MSE={wmse:.4f}  Lap/C={safe_ratio(lap_mse, wmse):.2f}×")

    # Sum queries
    for sens in [1.0, 5.0, 10.0, 50.0]:
        n = 5; k = 30
        qv = np.linspace(0, sens, n)
        spec = QuerySpec(query_values=qv, domain=f"sum(Δ={sens})", sensitivity=sens, epsilon=1.0, k=k)
        result, elapsed, P, ygrid = run_single(spec)
        wmse = worst_case_mse(P, qv, ygrid)
        lap_mse = LaplaceMechanism(epsilon=1.0, sensitivity=sens).mse()
        results.append({
            "experiment": "sum_query", "sensitivity": sens, "n": n, "k": k,
            "cegis_mse": round(wmse, 6), "laplace_mse": round(lap_mse, 6),
            "vs_laplace": safe_ratio(lap_mse, wmse),
            "time_s": round(elapsed, 4),
        })
        print(f"    Sum Δ={sens:5.1f}: MSE={wmse:.4f}  Lap/C={safe_ratio(lap_mse, wmse):.2f}×")

    # Median with L1 loss
    for n in [3, 5, 7, 11]:
        qv = np.arange(n, dtype=float)
        k = max(n, 30)
        spec = QuerySpec(query_values=qv, domain=f"median({n})", sensitivity=1.0,
                        epsilon=1.0, k=k, loss_fn=LossFunction.L1)
        result, elapsed, P, ygrid = run_single(spec)
        wmae = worst_case_mae(P, qv, ygrid)
        lap_mae = LaplaceMechanism(epsilon=1.0, sensitivity=1.0).mae()
        results.append({
            "experiment": "median_query_l1", "n": n, "k": k,
            "cegis_mae": round(wmae, 6), "laplace_mae": round(lap_mae, 6),
            "vs_laplace": safe_ratio(lap_mae, wmae),
            "time_s": round(elapsed, 4),
        })
        print(f"    Median n={n:2d} (L1): MAE={wmae:.4f}  Lap/C={safe_ratio(lap_mae, wmae):.2f}×")

    return results


def tier2_loss_comparison():
    """L1 vs L2 vs Linf loss objectives."""
    print("  [Tier 2] Loss function comparison")
    results = []
    n = 10; k = 30
    for eps in [0.5, 1.0, 2.0]:
        for loss_fn in [LossFunction.L1, LossFunction.L2, LossFunction.LINF]:
            qv = np.arange(n, dtype=float)
            spec = QuerySpec(query_values=qv, domain=f"counting({n})", sensitivity=1.0,
                            epsilon=eps, k=k, loss_fn=loss_fn)
            result, elapsed, P, ygrid = run_single(spec)
            wmse = worst_case_mse(P, qv, ygrid)
            wmae = worst_case_mae(P, qv, ygrid)
            results.append({
                "experiment": "loss_comparison",
                "epsilon": eps, "loss": loss_fn.name, "n": n, "k": k,
                "cegis_worst_mse": round(wmse, 6),
                "cegis_worst_mae": round(wmae, 6),
                "time_s": round(elapsed, 4),
            })
            print(f"    ε={eps:.1f} {loss_fn.name:4s}: MSE={wmse:.4f}  MAE={wmae:.4f}")
    return results


# ---------------------------------------------------------------------------
# Tier 3: Extended sweeps and discretization analysis
# ---------------------------------------------------------------------------

def tier3_discretization_convergence():
    """Study how MSE converges as k → ∞."""
    print("  [Tier 3] Discretization convergence (k → ∞)")
    results = []
    for n in [2, 5, 10]:
        for k in [10, 20, 30, 50, 100, 200]:
            eps = 1.0
            spec = QuerySpec.counting(n=n, epsilon=eps, delta=0.0, k=k)
            result, elapsed, P, ygrid = run_single(spec)
            wmse = worst_case_mse(P, spec.query_values, ygrid)
            stair = StaircaseMechanism(epsilon=eps, sensitivity=1.0)
            entry = {
                "experiment": "discretization_convergence",
                "n": n, "k": k,
                "cegis_mse": round(wmse, 6),
                "staircase_mse": round(stair.mse(), 6),
                "ratio_to_staircase": round(wmse / stair.mse(), 6),
                "time_s": round(elapsed, 4),
            }
            results.append(entry)
        print(f"    n={n:2d}: k=10→200 MSE converges to {results[-1]['cegis_mse']:.6f} (Staircase={results[-1]['staircase_mse']:.6f})")
    return results


def tier3_high_privacy():
    """High-privacy regime (very small ε)."""
    print("  [Tier 3] High-privacy regime")
    results = []
    for eps in [0.01, 0.05, 0.1, 0.25, 0.5]:
        for n in [2, 5, 10]:
            k = 50
            spec = QuerySpec.counting(n=n, epsilon=eps, delta=0.0, k=k)
            result, elapsed, P, ygrid = run_single(spec)
            wmse = worst_case_mse(P, spec.query_values, ygrid)
            lap_mse = LaplaceMechanism(epsilon=eps, sensitivity=1.0).mse()
            entry = {
                "experiment": "high_privacy",
                "epsilon": eps, "n": n, "k": k,
                "cegis_mse": round(wmse, 6),
                "laplace_mse": round(lap_mse, 6),
                "vs_laplace": safe_ratio(lap_mse, wmse),
                "time_s": round(elapsed, 4),
            }
            results.append(entry)
            print(f"    ε={eps:.2f} n={n:2d}: MSE={wmse:.4f}  Lap/C={entry['vs_laplace']:.1f}×")
    return results


# ---------------------------------------------------------------------------
# Tier 4: Scalability and stress
# ---------------------------------------------------------------------------

def tier4_scalability():
    """Synthesis time vs problem size."""
    print("  [Tier 4] Scalability")
    results = []
    for n in [2, 5, 10, 20, 50]:
        k = max(n, 20)
        spec = QuerySpec.counting(n=n, epsilon=1.0, delta=0.0, k=k)
        times = []
        for _ in range(3):
            t0 = time.time()
            result = CEGISSynthesize(spec, max_iter=500)
            times.append(time.time() - t0)
        avg_t = np.mean(times)
        ygrid = build_output_grid(spec.query_values, spec.k)
        P = result.mechanism
        wmse = worst_case_mse(P, spec.query_values, ygrid)
        entry = {
            "experiment": "scalability",
            "n": n, "k": k,
            "n_variables": n * k + 1,
            "n_constraints_upper_bound": 2 * (n - 1) * k,
            "avg_time_s": round(avg_t, 4),
            "std_time_s": round(float(np.std(times)), 4),
            "cegis_mse": round(wmse, 6),
            "converged": result.converged,
            "iterations": result.iterations,
        }
        results.append(entry)
        print(f"    n={n:3d} k={k:3d} vars={n*k+1:5d}: {avg_t:.4f}s ± {np.std(times):.4f}s  converged={result.converged}")
    return results


def tier4_speed_benchmark():
    """Repeated timing for statistical significance."""
    print("  [Tier 4] Speed benchmark (20 trials)")
    results = []
    n, k, eps = 10, 30, 1.0
    spec = QuerySpec.counting(n=n, epsilon=eps, delta=0.0, k=k)
    times = []
    for _ in range(20):
        t0 = time.time()
        CEGISSynthesize(spec, max_iter=300)
        times.append(time.time() - t0)
    times = np.array(times)
    entry = {
        "experiment": "speed_benchmark",
        "n": n, "k": k, "epsilon": eps,
        "trials": 20,
        "mean_s": round(float(times.mean()), 6),
        "median_s": round(float(np.median(times)), 6),
        "std_s": round(float(times.std()), 6),
        "p5_s": round(float(np.percentile(times, 5)), 6),
        "p95_s": round(float(np.percentile(times, 95)), 6),
    }
    results.append(entry)
    print(f"    n={n} k={k}: mean={times.mean()*1000:.2f}ms  median={np.median(times)*1000:.2f}ms  p95={np.percentile(times,95)*1000:.2f}ms")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TIERS = {
    1: [("counting_basic", tier1_counting_basic), ("approx_dp_basic", tier1_approx_dp)],
    2: [
        ("counting_sweep", tier2_counting_sweep),
        ("approx_dp_sweep", tier2_approx_dp_sweep),
        ("query_types", tier2_query_types),
        ("loss_comparison", tier2_loss_comparison),
    ],
    3: [
        ("discretization_convergence", tier3_discretization_convergence),
        ("high_privacy", tier3_high_privacy),
    ],
    4: [
        ("scalability", tier4_scalability),
        ("speed_benchmark", tier4_speed_benchmark),
    ],
}


def main():
    parser = argparse.ArgumentParser(description="DP-Forge benchmark suite")
    parser.add_argument("--tier", default="1", help="Tier to run: 1, 2, 3, 4, or all")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file")
    args = parser.parse_args()

    if args.tier == "all":
        tiers_to_run = [1, 2, 3, 4]
    else:
        tiers_to_run = [int(args.tier)]

    print(f"DP-Forge Benchmark Suite — Tiers: {tiers_to_run}")
    print("=" * 60)
    t_start = time.time()

    all_results: Dict[str, Any] = {
        "metadata": {
            "tool": "DP-Forge",
            "version": "0.1.0",
            "tiers": tiers_to_run,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    for tier in tiers_to_run:
        print(f"\n{'='*60}")
        print(f"Tier {tier}")
        print(f"{'='*60}")
        for name, fn in TIERS[tier]:
            try:
                all_results[name] = fn()
            except Exception as e:
                print(f"  FAILED: {name} — {e}")
                all_results[name] = [{"error": str(e)}]

    total_time = time.time() - t_start
    all_results["metadata"]["total_time_s"] = round(total_time, 1)

    # Summary
    all_entries = []
    for k, v in all_results.items():
        if isinstance(v, list):
            all_entries.extend(v)
    converged = [e for e in all_entries if e.get("converged", True) and "error" not in e]
    vs_laplace = [e["vs_laplace"] for e in all_entries if "vs_laplace" in e and e["vs_laplace"] < float("inf")]
    vs_gaussian = [e["vs_gaussian"] for e in all_entries if "vs_gaussian" in e]

    summary = {
        "total_experiments": len(all_entries),
        "successful": len(converged),
        "total_time_s": round(total_time, 1),
    }
    if vs_laplace:
        summary["median_vs_laplace"] = round(float(np.median(vs_laplace)), 2)
        summary["max_vs_laplace"] = round(max(vs_laplace), 2)
        summary["geomean_vs_laplace"] = round(float(np.exp(np.mean(np.log(
            [max(x, 1.0) for x in vs_laplace]
        )))), 2)
    if vs_gaussian:
        summary["median_vs_gaussian"] = round(float(np.median(vs_gaussian)), 2)
        summary["max_vs_gaussian"] = round(max(vs_gaussian), 2)
    all_results["summary"] = summary

    outpath = args.output or os.path.join(os.path.dirname(__file__), "results.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Results saved to {outpath}")
    print(f"Total time: {total_time:.1f}s")
    if "median_vs_laplace" in summary:
        print(f"Median improvement vs Laplace: {summary['median_vs_laplace']}×")
    if "max_vs_laplace" in summary:
        print(f"Max improvement vs Laplace: {summary['max_vs_laplace']}×")
    if "median_vs_gaussian" in summary:
        print(f"Median improvement vs Gaussian: {summary['median_vs_gaussian']}×")


if __name__ == "__main__":
    main()
