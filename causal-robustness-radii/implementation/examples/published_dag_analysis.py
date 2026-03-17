#!/usr/bin/env python3
"""Analyse robustness of all 15 published benchmark DAGs.

For every DAG in the CausalCert benchmark library this script:

  1. Loads the graph and generates synthetic observational data.
  2. Runs the full CausalCert pipeline (CI testing → fragility → radius).
  3. Collects per-DAG results into a summary table.
  4. Identifies the most fragile edges across all DAGs.

Run::

    python examples/published_dag_analysis.py [--samples 1000] [--seed 42]
"""
from __future__ import annotations

import argparse
import sys
import textwrap
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from causalcert.data.synthetic import generate_linear_gaussian
from causalcert.evaluation.published_dags import (
    get_all_published_dags,
    get_published_dag,
    get_small_dags,
    list_published_dags,
)
from causalcert.pipeline.config import PipelineRunConfig
from causalcert.pipeline.orchestrator import CausalCertPipeline
from causalcert.types import SolverStrategy


# =====================================================================
# Data containers
# =====================================================================

@dataclass
class DAGResult:
    """Stores analysis results for a single published DAG."""
    name: str
    n_nodes: int
    n_edges: int
    radius_lower: int
    radius_upper: int
    top_fragile_edges: list[tuple[str, str, float]]
    n_load_bearing: int
    runtime_s: float
    ate_estimate: float | None = None
    ate_se: float | None = None


@dataclass
class CrossDAGSummary:
    """Aggregated summary across all DAGs."""
    results: list[DAGResult] = field(default_factory=list)
    all_fragile_edges: list[tuple[str, str, str, float]] = field(
        default_factory=list,
    )  # (dag_name, src, dst, score)


# =====================================================================
# Helpers
# =====================================================================

def _generate_data_for_dag(
    adj: np.ndarray,
    node_names: list[str],
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Generate linear-Gaussian data from a DAG."""
    data, _weights = generate_linear_gaussian(
        adj,
        n=n_samples,
        noise_scale=1.0,
        edge_weight_range=(0.3, 0.9),
        seed=seed,
    )
    data.columns = node_names
    return data


def _run_single_dag(
    dag_name: str,
    n_samples: int,
    seed: int,
) -> DAGResult:
    """Run CausalCert on one published DAG and return results."""
    dag_info = get_published_dag(dag_name)
    adj = dag_info.adj
    names = dag_info.node_names
    treatment = dag_info.default_treatment
    outcome = dag_info.default_outcome

    df = _generate_data_for_dag(adj, names, n_samples, seed)

    config = PipelineRunConfig(
        treatment=treatment,
        outcome=outcome,
        alpha=0.05,
        solver_strategy=SolverStrategy.AUTO,
    )
    pipeline = CausalCertPipeline(config)
    t0 = time.perf_counter()
    report = pipeline.run(adj_matrix=adj, data=df)
    elapsed = time.perf_counter() - t0

    top_edges = [
        (names[fs.edge[0]], names[fs.edge[1]], fs.total_score)
        for fs in report.fragility_ranking[:5]
    ]
    n_load = sum(1 for fs in report.fragility_ranking if fs.total_score >= 0.4)

    ate = None
    ate_se = None
    if report.baseline_estimate is not None:
        ate = report.baseline_estimate.ate
        ate_se = report.baseline_estimate.se

    return DAGResult(
        name=dag_name,
        n_nodes=len(names),
        n_edges=int(adj.sum()),
        radius_lower=report.radius.lower_bound,
        radius_upper=report.radius.upper_bound,
        top_fragile_edges=top_edges,
        n_load_bearing=n_load,
        runtime_s=elapsed,
        ate_estimate=ate,
        ate_se=ate_se,
    )


# =====================================================================
# Analysis routines
# =====================================================================

def analyse_all_dags(
    n_samples: int = 1000,
    seed: int = 42,
    max_nodes: int | None = None,
) -> CrossDAGSummary:
    """Run CausalCert on every published DAG and build a cross-DAG summary."""
    dag_names = (
        [dag.name for dag in get_small_dags(max_nodes)]
        if max_nodes is not None
        else [dag.name for dag in get_all_published_dags()]
    )
    summary = CrossDAGSummary()

    print(f"Analysing {len(dag_names)} published DAGs …\n")

    for idx, name in enumerate(dag_names, 1):
        print(f"  [{idx:>2d}/{len(dag_names)}]  {name:<20s}", end="", flush=True)
        try:
            result = _run_single_dag(name, n_samples, seed)
            summary.results.append(result)

            for src, dst, sc in result.top_fragile_edges:
                summary.all_fragile_edges.append((name, src, dst, sc))

            print(f"  radius=[{result.radius_lower},{result.radius_upper}]  "
                  f"load-bearing={result.n_load_bearing}  "
                  f"({result.runtime_s:.1f}s)")
        except Exception as exc:
            print(f"  ✗  {exc}")

    return summary


def print_summary_table(summary: CrossDAGSummary) -> None:
    """Print a formatted results table for all DAGs."""
    print()
    print("=" * 90)
    print("Cross-DAG Summary")
    print("=" * 90)
    header = (
        f"{'DAG':<20s}  {'|V|':>4s}  {'|E|':>4s}  "
        f"{'r_lo':>4s}  {'r_hi':>4s}  "
        f"{'#LB':>4s}  {'ATE':>8s}  {'Time':>6s}"
    )
    print(header)
    print("-" * 90)

    for r in summary.results:
        ate_str = f"{r.ate_estimate:.4f}" if r.ate_estimate is not None else "N/A"
        print(
            f"{r.name:<20s}  {r.n_nodes:>4d}  {r.n_edges:>4d}  "
            f"{r.radius_lower:>4d}  {r.radius_upper:>4d}  "
            f"{r.n_load_bearing:>4d}  {ate_str:>8s}  {r.runtime_s:>5.1f}s"
        )
    print()


def print_most_fragile_edges(summary: CrossDAGSummary, top_k: int = 20) -> None:
    """Rank all edges across all DAGs by fragility and print the top-k."""
    sorted_edges = sorted(summary.all_fragile_edges, key=lambda t: -t[3])

    print("=" * 70)
    print(f"Top-{top_k} Most Fragile Edges Across All DAGs")
    print("=" * 70)

    for idx, (dag, src, dst, sc) in enumerate(sorted_edges[:top_k], 1):
        bar = "█" * int(sc * 30)
        print(f"  {idx:>2d}. [{dag:<16s}] {src:<12s} → {dst:<12s}  {sc:.4f}  {bar}")
    print()


def print_radius_distribution(summary: CrossDAGSummary) -> None:
    """Print a histogram-style overview of the robustness radii."""
    print("=" * 60)
    print("Robustness-Radius Distribution")
    print("=" * 60)

    bins: dict[str, list[str]] = defaultdict(list)
    for r in summary.results:
        lo = r.radius_lower
        if lo == 0:
            bins["  r = 0  (fragile)"].append(r.name)
        elif lo <= 2:
            bins["  r ∈ [1,2]"].append(r.name)
        elif lo <= 5:
            bins["  r ∈ [3,5]"].append(r.name)
        else:
            bins["  r ≥ 6   (robust)"].append(r.name)

    for label in sorted(bins.keys()):
        dags = bins[label]
        print(f"{label}: {len(dags)}  — {', '.join(dags)}")
    print()


def print_severity_breakdown(summary: CrossDAGSummary) -> None:
    """Count how many edges fall into each severity category across all DAGs."""
    counts: dict[str, int] = defaultdict(int)
    for _, _, _, sc in summary.all_fragile_edges:
        if sc >= 0.7:
            counts["CRITICAL"] += 1
        elif sc >= 0.4:
            counts["IMPORTANT"] += 1
        elif sc >= 0.1:
            counts["MODERATE"] += 1
        else:
            counts["COSMETIC"] += 1

    print("=" * 50)
    print("Edge-Severity Breakdown (all DAGs pooled)")
    print("=" * 50)
    for sev in ["CRITICAL", "IMPORTANT", "MODERATE", "COSMETIC"]:
        print(f"  {sev:<12s}  {counts.get(sev, 0):>4d}")
    print()


def print_dag_comparison_matrix(summary: CrossDAGSummary) -> None:
    """Print a compact comparison of key metrics across DAGs."""
    print("=" * 60)
    print("DAG Comparison Matrix")
    print("=" * 60)

    # Density comparison
    for r in summary.results:
        max_edges = r.n_nodes * (r.n_nodes - 1)
        density = r.n_edges / max_edges if max_edges > 0 else 0.0
        robustness_per_edge = (
            r.radius_lower / r.n_edges if r.n_edges > 0 else 0.0
        )
        print(
            f"  {r.name:<20s}  density={density:.3f}  "
            f"radius/|E|={robustness_per_edge:.3f}  "
            f"load-bearing%={100 * r.n_load_bearing / max(r.n_edges, 1):.1f}%"
        )
    print()


def print_runtime_statistics(summary: CrossDAGSummary) -> None:
    """Summarise runtime across all DAGs."""
    if not summary.results:
        return
    times = [r.runtime_s for r in summary.results]
    print("=" * 50)
    print("Runtime Statistics")
    print("=" * 50)
    print(f"  Total     : {sum(times):.1f} s")
    print(f"  Mean      : {np.mean(times):.2f} s")
    print(f"  Median    : {np.median(times):.2f} s")
    print(f"  Max       : {max(times):.2f} s  ({summary.results[int(np.argmax(times))].name})")
    print(f"  Min       : {min(times):.2f} s  ({summary.results[int(np.argmin(times))].name})")
    print()


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robustness analysis of all published benchmark DAGs.",
    )
    parser.add_argument(
        "--samples", type=int, default=1000,
        help="Number of synthetic samples per DAG (default: 1000).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="Number of most-fragile edges to display (default: 20).",
    )
    parser.add_argument(
        "--max-nodes", type=int, default=None,
        help="Restrict to DAGs with at most this many nodes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(textwrap.dedent("""\
    ╔══════════════════════════════════════════════════════════╗
    ║    CausalCert — Published DAG Analysis                  ║
    ╚══════════════════════════════════════════════════════════╝
    """))

    summary = analyse_all_dags(
        n_samples=args.samples,
        seed=args.seed,
        max_nodes=args.max_nodes,
    )

    print_summary_table(summary)
    print_most_fragile_edges(summary, top_k=args.top_k)
    print_radius_distribution(summary)
    print_severity_breakdown(summary)
    print_dag_comparison_matrix(summary)
    print_runtime_statistics(summary)

    print("Analysis complete.")


if __name__ == "__main__":
    main()
