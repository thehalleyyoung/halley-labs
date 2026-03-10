#!/usr/bin/env python3
"""Compare CI-test methods, solver strategies, and convergence behaviour.

This script benchmarks CausalCert across different configuration axes:

  * **CI tests** — partial correlation, Spearman rank, kernel CI, CRT,
    Cauchy ensemble.
  * **Solvers** — ILP, LP relaxation, FPT-DP, CDCL, greedy search.
  * **Convergence** — how the robustness radius estimate tightens as the
    sample size grows.

Run::

    python examples/comparison_methods.py [--samples 1500] [--seed 42]
"""
from __future__ import annotations

import argparse
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from causalcert.data.synthetic import generate_linear_gaussian
from causalcert.pipeline.config import PipelineRunConfig
from causalcert.pipeline.orchestrator import CausalCertPipeline
from causalcert.types import AuditReport


# =====================================================================
# Helpers
# =====================================================================

def _build_test_dag() -> tuple[np.ndarray, list[str]]:
    """Small 6-node DAG used for all comparisons."""
    names = ["U", "X", "M", "Z", "W", "Y"]
    n = len(names)
    adj = np.zeros((n, n), dtype=np.int8)
    adj[0, 1] = 1  # U → X
    adj[0, 4] = 1  # U → W
    adj[1, 2] = 1  # X → M
    adj[2, 5] = 1  # M → Y
    adj[3, 1] = 1  # Z → X
    adj[3, 5] = 1  # Z → Y
    adj[4, 5] = 1  # W → Y
    return adj, names


def _generate_data(
    adj: np.ndarray,
    names: list[str],
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weights = adj.astype(np.float64) * rng.uniform(0.3, 0.9, size=adj.shape)
    data = generate_linear_gaussian(
        adj_matrix=adj, weights=weights,
        n_samples=n_samples, noise_scale=1.0, seed=seed,
    )
    return pd.DataFrame(data, columns=names)


@dataclass
class MethodResult:
    """Single benchmark row."""
    method: str
    category: str          # "ci_test" | "solver" | "convergence"
    radius_lower: int
    radius_upper: int
    n_fragile: int
    runtime_s: float
    extra: dict = field(default_factory=dict)


# =====================================================================
# 1. CI-Test Comparison
# =====================================================================

CI_TEST_METHODS = [
    "partial_correlation",
    "spearman",
    "kernel",
    "crt",
    "ensemble",
]


def compare_ci_tests(
    adj: np.ndarray,
    df: pd.DataFrame,
    treatment: int = 1,
    outcome: int = 5,
) -> list[MethodResult]:
    """Run CausalCert with each CI-test method and record results."""
    results: list[MethodResult] = []

    for method in CI_TEST_METHODS:
        print(f"  CI test: {method:<25s}", end="", flush=True)
        config = PipelineRunConfig(
            treatment=treatment,
            outcome=outcome,
            alpha=0.05,
            solver_strategy="auto",
            ci_test_method=method,
        )
        pipeline = CausalCertPipeline(config)
        t0 = time.perf_counter()
        report = pipeline.run(adj_matrix=adj, data=df)
        elapsed = time.perf_counter() - t0

        n_frag = sum(1 for fs in report.fragility_ranking if fs.score >= 0.4)
        results.append(MethodResult(
            method=method,
            category="ci_test",
            radius_lower=report.radius.lower_bound,
            radius_upper=report.radius.upper_bound,
            n_fragile=n_frag,
            runtime_s=elapsed,
        ))
        print(f"  r=[{report.radius.lower_bound},{report.radius.upper_bound}]  "
              f"fragile={n_frag}  ({elapsed:.2f}s)")

    return results


def print_ci_test_table(results: list[MethodResult]) -> None:
    """Pretty-print CI-test comparison."""
    print()
    print("=" * 70)
    print("CI-Test Method Comparison")
    print("=" * 70)
    header = f"{'Method':<25s}  {'r_lo':>4s}  {'r_hi':>4s}  {'#Frag':>5s}  {'Time':>7s}"
    print(header)
    print("-" * 70)
    for r in results:
        print(f"{r.method:<25s}  {r.radius_lower:>4d}  {r.radius_upper:>4d}  "
              f"{r.n_fragile:>5d}  {r.runtime_s:>6.2f}s")
    print()


# =====================================================================
# 2. Solver Comparison
# =====================================================================

SOLVER_STRATEGIES = ["ilp", "lp_relaxation", "fpt", "cdcl", "greedy"]


def compare_solvers(
    adj: np.ndarray,
    df: pd.DataFrame,
    treatment: int = 1,
    outcome: int = 5,
) -> list[MethodResult]:
    """Run CausalCert with each solver strategy."""
    results: list[MethodResult] = []

    for strategy in SOLVER_STRATEGIES:
        print(f"  Solver: {strategy:<20s}", end="", flush=True)
        config = PipelineRunConfig(
            treatment=treatment,
            outcome=outcome,
            alpha=0.05,
            solver_strategy=strategy,
        )
        pipeline = CausalCertPipeline(config)
        t0 = time.perf_counter()
        report = pipeline.run(adj_matrix=adj, data=df)
        elapsed = time.perf_counter() - t0

        n_frag = sum(1 for fs in report.fragility_ranking if fs.score >= 0.4)
        gap = report.radius.upper_bound - report.radius.lower_bound
        results.append(MethodResult(
            method=strategy,
            category="solver",
            radius_lower=report.radius.lower_bound,
            radius_upper=report.radius.upper_bound,
            n_fragile=n_frag,
            runtime_s=elapsed,
            extra={"gap": gap},
        ))
        print(f"  r=[{report.radius.lower_bound},{report.radius.upper_bound}]  "
              f"gap={gap}  ({elapsed:.2f}s)")

    return results


def print_solver_table(results: list[MethodResult]) -> None:
    """Pretty-print solver comparison."""
    print()
    print("=" * 70)
    print("Solver Strategy Comparison")
    print("=" * 70)
    header = (f"{'Solver':<20s}  {'r_lo':>4s}  {'r_hi':>4s}  "
              f"{'Gap':>4s}  {'#Frag':>5s}  {'Time':>7s}")
    print(header)
    print("-" * 70)
    for r in results:
        gap = r.extra.get("gap", "?")
        print(f"{r.method:<20s}  {r.radius_lower:>4d}  {r.radius_upper:>4d}  "
              f"{gap:>4}  {r.n_fragile:>5d}  {r.runtime_s:>6.2f}s")
    print()


# =====================================================================
# 3. Convergence with Sample Size
# =====================================================================

SAMPLE_SIZES = [200, 500, 1000, 2000, 5000]


def convergence_study(
    adj: np.ndarray,
    names: list[str],
    treatment: int = 1,
    outcome: int = 5,
    seed: int = 42,
) -> list[MethodResult]:
    """Track how the radius estimate changes with increasing data."""
    results: list[MethodResult] = []

    for n in SAMPLE_SIZES:
        print(f"  n={n:<6d}", end="", flush=True)
        df = _generate_data(adj, names, n, seed)

        config = PipelineRunConfig(
            treatment=treatment, outcome=outcome,
            alpha=0.05, solver_strategy="auto",
        )
        pipeline = CausalCertPipeline(config)
        t0 = time.perf_counter()
        report = pipeline.run(adj_matrix=adj, data=df)
        elapsed = time.perf_counter() - t0

        n_frag = sum(1 for fs in report.fragility_ranking if fs.score >= 0.4)
        gap = report.radius.upper_bound - report.radius.lower_bound
        results.append(MethodResult(
            method=f"n={n}",
            category="convergence",
            radius_lower=report.radius.lower_bound,
            radius_upper=report.radius.upper_bound,
            n_fragile=n_frag,
            runtime_s=elapsed,
            extra={"n_samples": n, "gap": gap},
        ))
        print(f"  r=[{report.radius.lower_bound},{report.radius.upper_bound}]  "
              f"gap={gap}  ({elapsed:.2f}s)")

    return results


def print_convergence_table(results: list[MethodResult]) -> None:
    """Pretty-print convergence results with ASCII plot."""
    print()
    print("=" * 70)
    print("Convergence: Robustness Radius vs. Sample Size")
    print("=" * 70)

    header = f"{'n':>6s}  {'r_lo':>4s}  {'r_hi':>4s}  {'Gap':>4s}  {'Plot'}"
    print(header)
    print("-" * 70)

    max_r = max((r.radius_upper for r in results), default=1) or 1
    for r in results:
        n = r.extra.get("n_samples", "?")
        gap = r.extra.get("gap", "?")
        lo_bar = "░" * int(r.radius_lower / max_r * 30)
        hi_bar = "█" * int((r.radius_upper - r.radius_lower) / max_r * 30)
        print(f"{n:>6}  {r.radius_lower:>4d}  {r.radius_upper:>4d}  "
              f"{gap:>4}  {lo_bar}{hi_bar}")
    print()


# =====================================================================
# 4. Summary Statistics
# =====================================================================

def print_combined_summary(
    ci_results: list[MethodResult],
    solver_results: list[MethodResult],
    conv_results: list[MethodResult],
) -> None:
    """Print a one-paragraph summary of all comparisons."""
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    fastest_ci = min(ci_results, key=lambda r: r.runtime_s)
    tightest_solver = min(solver_results, key=lambda r: r.extra.get("gap", 999))
    largest_n = conv_results[-1] if conv_results else None

    print(f"  • Fastest CI test     : {fastest_ci.method} ({fastest_ci.runtime_s:.2f}s)")
    print(f"  • Tightest solver gap : {tightest_solver.method} "
          f"(gap={tightest_solver.extra.get('gap')})")
    if largest_n is not None:
        print(f"  • Radius at largest n : [{largest_n.radius_lower}, "
              f"{largest_n.radius_upper}]")
    print()


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CausalCert CI tests, solvers, and convergence.",
    )
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(textwrap.dedent("""\
    ╔══════════════════════════════════════════════════════════╗
    ║    CausalCert — Method Comparison                       ║
    ╚══════════════════════════════════════════════════════════╝
    """))

    adj, names = _build_test_dag()
    df = _generate_data(adj, names, args.samples, args.seed)

    # 1. CI tests
    print("1. Comparing CI-test methods:\n")
    ci_results = compare_ci_tests(adj, df, treatment=1, outcome=5)
    print_ci_test_table(ci_results)

    # 2. Solvers
    print("2. Comparing solver strategies:\n")
    solver_results = compare_solvers(adj, df, treatment=1, outcome=5)
    print_solver_table(solver_results)

    # 3. Convergence
    print("3. Convergence study (varying sample size):\n")
    conv_results = convergence_study(adj, names, treatment=1, outcome=5, seed=args.seed)
    print_convergence_table(conv_results)

    # 4. Summary
    print_combined_summary(ci_results, solver_results, conv_results)

    print("Done.")


if __name__ == "__main__":
    main()
