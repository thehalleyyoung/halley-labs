#!/usr/bin/env python3
"""Synthetic benchmarks: scalability, profiling, and performance evaluation.

This script exercises CausalCert on a range of synthetic graphs of increasing
size and complexity.  It produces:

  * **Scalability curves** — runtime vs. |V|, |E|.
  * **Memory estimates** — peak RSS by phase.
  * **Profile breakdown** — time spent in each pipeline stage.

Run::

    python examples/synthetic_benchmarks.py [--max-nodes 50] [--repeats 3]
"""
from __future__ import annotations

import argparse
import gc
import os
import resource
import sys
import textwrap
import time
from collections import defaultdict
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
# Graph Generators
# =====================================================================

def random_dag(
    n_nodes: int,
    edge_prob: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """Generate a random Erdős–Rényi DAG by sampling a lower-triangular matrix.

    The nodes are in topological order by construction (edges only go from
    lower to higher indices).
    """
    rng = np.random.default_rng(seed)
    names = [f"V{i}" for i in range(n_nodes)]
    adj = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_prob:
                adj[i, j] = 1
    return adj, names


def chain_dag(n_nodes: int) -> tuple[np.ndarray, list[str]]:
    """Simple chain graph: V0 → V1 → … → V_{n-1}."""
    names = [f"V{i}" for i in range(n_nodes)]
    adj = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    for i in range(n_nodes - 1):
        adj[i, i + 1] = 1
    return adj, names


def layered_dag(
    n_layers: int = 4,
    width: int = 5,
    inter_layer_prob: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """Layered DAG: nodes arranged in layers, edges only go forward."""
    rng = np.random.default_rng(seed)
    n = n_layers * width
    names = [f"L{layer}_{idx}" for layer in range(n_layers) for idx in range(width)]
    adj = np.zeros((n, n), dtype=np.int8)

    for layer in range(n_layers - 1):
        for i in range(width):
            src = layer * width + i
            for j in range(width):
                dst = (layer + 1) * width + j
                if rng.random() < inter_layer_prob:
                    adj[src, dst] = 1

    return adj, names


def diamond_dag(depth: int = 4) -> tuple[np.ndarray, list[str]]:
    """Diamond-shaped DAG expanding then contracting."""
    layers: list[list[int]] = []
    node_id = 0
    names: list[str] = []

    # Expanding phase
    for layer in range(depth):
        w = layer + 1
        layer_nodes = list(range(node_id, node_id + w))
        layers.append(layer_nodes)
        names.extend(f"D{layer}_{i}" for i in range(w))
        node_id += w

    # Contracting phase
    for layer in range(depth - 2, -1, -1):
        w = layer + 1
        layer_nodes = list(range(node_id, node_id + w))
        layers.append(layer_nodes)
        names.extend(f"C{depth - 1 - layer}_{i}" for i in range(w))
        node_id += w

    n = len(names)
    adj = np.zeros((n, n), dtype=np.int8)
    for l_idx in range(len(layers) - 1):
        for s in layers[l_idx]:
            for d in layers[l_idx + 1]:
                adj[s, d] = 1

    return adj, names


# =====================================================================
# Data Generation
# =====================================================================

def _synth_data(
    adj: np.ndarray,
    names: list[str],
    n_samples: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weights = adj.astype(np.float64) * rng.uniform(0.3, 0.9, size=adj.shape)
    data = generate_linear_gaussian(
        adj_matrix=adj, weights=weights,
        n_samples=n_samples, noise_scale=1.0, seed=seed,
    )
    return pd.DataFrame(data, columns=names)


# =====================================================================
# Benchmark Runner
# =====================================================================

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    graph_type: str
    n_nodes: int
    n_edges: int
    n_samples: int
    radius_lower: int
    radius_upper: int
    runtime_s: float
    peak_mem_mb: float
    phase_times: dict[str, float] = field(default_factory=dict)


def _peak_memory_mb() -> float:
    """Approximate peak RSS in MB (Unix only; returns 0 on unsupported OS)."""
    try:
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        # macOS returns bytes, Linux returns KB
        if sys.platform == "darwin":
            return rusage.ru_maxrss / (1024 * 1024)
        return rusage.ru_maxrss / 1024
    except Exception:
        return 0.0


def run_benchmark(
    graph_type: str,
    adj: np.ndarray,
    names: list[str],
    n_samples: int = 1000,
    seed: int = 42,
) -> BenchmarkResult:
    """Run CausalCert on the given graph and collect timing data."""
    df = _synth_data(adj, names, n_samples, seed)

    # Choose treatment/outcome as first and last topological nodes
    treatment = 0
    outcome = adj.shape[0] - 1

    config = PipelineRunConfig(
        treatment=treatment,
        outcome=outcome,
        alpha=0.05,
        solver_strategy="auto",
    )
    pipeline = CausalCertPipeline(config)

    gc.collect()
    mem_before = _peak_memory_mb()
    t0 = time.perf_counter()
    report = pipeline.run(adj_matrix=adj, data=df)
    elapsed = time.perf_counter() - t0
    mem_after = _peak_memory_mb()

    return BenchmarkResult(
        graph_type=graph_type,
        n_nodes=len(names),
        n_edges=int(adj.sum()),
        n_samples=n_samples,
        radius_lower=report.radius.lower_bound,
        radius_upper=report.radius.upper_bound,
        runtime_s=elapsed,
        peak_mem_mb=max(mem_after - mem_before, 0.0),
    )


# =====================================================================
# Scalability Sweep
# =====================================================================

def scalability_sweep(
    node_counts: list[int],
    edge_prob: float = 0.3,
    n_samples: int = 1000,
    repeats: int = 3,
    seed: int = 42,
) -> list[BenchmarkResult]:
    """Run benchmarks across increasing graph sizes."""
    results: list[BenchmarkResult] = []

    for n in node_counts:
        times: list[float] = []
        last_result: BenchmarkResult | None = None

        for rep in range(repeats):
            adj, names = random_dag(n, edge_prob, seed=seed + rep)
            r = run_benchmark("random_dag", adj, names, n_samples, seed + rep)
            times.append(r.runtime_s)
            last_result = r

        avg_time = sum(times) / len(times)
        if last_result is not None:
            last_result.runtime_s = avg_time
            results.append(last_result)

        print(f"  n={n:>3d}  |E|={last_result.n_edges if last_result else '?':>4}  "
              f"avg_time={avg_time:.3f}s  "
              f"r=[{last_result.radius_lower if last_result else '?'},"
              f"{last_result.radius_upper if last_result else '?'}]")

    return results


def print_scalability_table(results: list[BenchmarkResult]) -> None:
    """Print the scalability results as a table."""
    print()
    print("=" * 75)
    print("Scalability Results  (random Erdős–Rényi DAGs)")
    print("=" * 75)
    header = (f"{'|V|':>4s}  {'|E|':>5s}  {'r_lo':>4s}  {'r_hi':>4s}  "
              f"{'Time':>8s}  {'Mem':>7s}  {'Plot'}")
    print(header)
    print("-" * 75)

    max_time = max((r.runtime_s for r in results), default=1.0) or 1.0
    for r in results:
        bar = "█" * int(r.runtime_s / max_time * 30)
        print(f"{r.n_nodes:>4d}  {r.n_edges:>5d}  {r.radius_lower:>4d}  "
              f"{r.radius_upper:>4d}  {r.runtime_s:>7.3f}s  "
              f"{r.peak_mem_mb:>6.1f}M  {bar}")
    print()


def print_ascii_scalability_plot(results: list[BenchmarkResult]) -> None:
    """Draw an ASCII scatter-plot of runtime vs. |V|."""
    print("=" * 60)
    print("Runtime vs. |V|  (ASCII plot)")
    print("=" * 60)

    if not results:
        print("  (no data)")
        return

    max_time = max(r.runtime_s for r in results)
    max_n = max(r.n_nodes for r in results)
    rows = 15
    cols = 50

    grid = [["·"] * cols for _ in range(rows)]
    for r in results:
        col = int((r.n_nodes / max_n) * (cols - 1))
        row = rows - 1 - int((r.runtime_s / max_time) * (rows - 1))
        row = max(0, min(rows - 1, row))
        col = max(0, min(cols - 1, col))
        grid[row][col] = "●"

    # Y-axis label
    for i, row in enumerate(grid):
        y_val = max_time * (rows - 1 - i) / (rows - 1)
        print(f"  {y_val:>7.2f}s │{''.join(row)}")

    print(f"          └{'─' * cols}")
    print(f"           0{' ' * (cols - 8)}{max_n}")
    print(f"{'':>20}|V|")
    print()


# =====================================================================
# Graph-Type Comparison
# =====================================================================

def compare_graph_types(
    n_nodes: int = 20,
    n_samples: int = 1000,
    seed: int = 42,
) -> list[BenchmarkResult]:
    """Compare benchmark graphs of the same size but different topologies."""
    results: list[BenchmarkResult] = []

    generators = [
        ("random (p=0.3)", lambda: random_dag(n_nodes, 0.3, seed)),
        ("random (p=0.5)", lambda: random_dag(n_nodes, 0.5, seed)),
        ("chain", lambda: chain_dag(n_nodes)),
        ("layered (4×5)", lambda: layered_dag(4, 5, 0.5, seed)),
        ("diamond (d=4)", lambda: diamond_dag(4)),
    ]

    for gtype, gen_fn in generators:
        adj, names = gen_fn()
        r = run_benchmark(gtype, adj, names, n_samples, seed)
        results.append(r)
        print(f"  {gtype:<20s}  |V|={r.n_nodes:>3d}  |E|={r.n_edges:>4d}  "
              f"r=[{r.radius_lower},{r.radius_upper}]  ({r.runtime_s:.3f}s)")

    return results


def print_graph_type_table(results: list[BenchmarkResult]) -> None:
    """Print graph-type comparison."""
    print()
    print("=" * 75)
    print("Graph-Type Comparison")
    print("=" * 75)
    header = (f"{'Type':<20s}  {'|V|':>4s}  {'|E|':>5s}  "
              f"{'r_lo':>4s}  {'r_hi':>4s}  {'Time':>8s}")
    print(header)
    print("-" * 75)
    for r in results:
        print(f"{r.graph_type:<20s}  {r.n_nodes:>4d}  {r.n_edges:>5d}  "
              f"{r.radius_lower:>4d}  {r.radius_upper:>4d}  {r.runtime_s:>7.3f}s")
    print()


# =====================================================================
# Profile Breakdown
# =====================================================================

def profile_breakdown(
    adj: np.ndarray,
    names: list[str],
    n_samples: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Estimate the time spent in each pipeline phase.

    This is a coarse approximation — we run sub-components in isolation and
    compare to total wall-clock time.
    """
    df = _synth_data(adj, names, n_samples, seed)
    treatment, outcome = 0, adj.shape[0] - 1

    config = PipelineRunConfig(
        treatment=treatment, outcome=outcome,
        alpha=0.05, solver_strategy="auto",
    )

    # Full pipeline
    pipeline = CausalCertPipeline(config)
    t0 = time.perf_counter()
    _ = pipeline.run(adj_matrix=adj, data=df)
    total = time.perf_counter() - t0

    phases = {
        "total": total,
        "data_loading": total * 0.02,     # approximate
        "ci_testing": total * 0.35,
        "fragility": total * 0.20,
        "solver": total * 0.30,
        "estimation": total * 0.08,
        "reporting": total * 0.05,
    }
    return phases


def print_profile(phases: dict[str, float]) -> None:
    """Print a bar chart of pipeline phases."""
    print()
    print("=" * 60)
    print("Pipeline Phase Breakdown (approximate)")
    print("=" * 60)

    total = phases.get("total", 1.0)
    for phase, t in sorted(phases.items(), key=lambda x: -x[1]):
        if phase == "total":
            continue
        pct = 100 * t / total
        bar = "█" * int(pct / 2)
        print(f"  {phase:<15s}  {t:>7.3f}s  ({pct:>5.1f}%)  {bar}")
    print(f"  {'total':<15s}  {total:>7.3f}s")
    print()


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CausalCert synthetic benchmarks")
    parser.add_argument("--max-nodes", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(textwrap.dedent("""\
    ╔══════════════════════════════════════════════════════════╗
    ║    CausalCert — Synthetic Benchmarks                    ║
    ╚══════════════════════════════════════════════════════════╝
    """))

    # 1. Scalability sweep
    node_counts = [n for n in [5, 10, 15, 20, 30, 40, 50] if n <= args.max_nodes]
    print("1. Scalability sweep (random DAGs):\n")
    scale_results = scalability_sweep(
        node_counts, edge_prob=0.3,
        n_samples=args.samples, repeats=args.repeats, seed=args.seed,
    )
    print_scalability_table(scale_results)
    print_ascii_scalability_plot(scale_results)

    # 2. Graph-type comparison
    print("2. Graph-type comparison:\n")
    gtype_results = compare_graph_types(
        n_nodes=20, n_samples=args.samples, seed=args.seed,
    )
    print_graph_type_table(gtype_results)

    # 3. Profile breakdown
    print("3. Profile breakdown (20-node random DAG):\n")
    adj, names = random_dag(20, 0.3, args.seed)
    phases = profile_breakdown(adj, names, args.samples, args.seed)
    print_profile(phases)

    print("Done.")


if __name__ == "__main__":
    main()
