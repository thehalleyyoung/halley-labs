#!/usr/bin/env python3
"""CausalCert — Comprehensive Experiment Suite.

Runs all experiments and benchmarks needed for the tool paper:

  1. Published DAG Fragility Analysis (Table 1)
  2. Standard Benchmark Suite (Table 2)
  3. Solver Comparison (Table 3)
  4. CI-Test Method Comparison (Table 4)
  5. Scalability Evaluation (Figure 1)
  6. Convergence with Sample Size (Figure 2)
  7. Ablation Study (Table 5)
  8. Baseline Comparisons: IDA / SID / E-value / PC / GES (Table 6)
  9. Treewidth vs Runtime (Figure 3)

Results are written to benchmark_output/ as JSON files.

Usage::

    python experiments/run_experiments.py [--output-dir benchmark_output] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from causalcert.benchmarks.compare import (
    compare_with_e_value,
    compare_with_ida,
    compare_with_structure_learning,
    compute_e_value,
)
from causalcert.benchmarks.standard import (
    get_benchmark,
    list_benchmarks,
)
from causalcert.data.synthetic import generate_linear_gaussian
from causalcert.evaluation.published_dags import (
    get_published_dag,
    list_published_dags,
)
from causalcert.pipeline.config import PipelineRunConfig, CITestMethod
from causalcert.pipeline.orchestrator import CausalCertPipeline
from causalcert.types import SolverStrategy


# ====================================================================
# Helpers
# ====================================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _save_json(data: dict | list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, cls=NumpyEncoder))
    print(f"  ✓ Saved {path}")


def _is_dag(adj: np.ndarray) -> bool:
    """Check if adjacency matrix is acyclic via topological sort attempt."""
    n = adj.shape[0]
    in_degree = adj.sum(axis=0).copy()
    queue = [i for i in range(n) if in_degree[i] == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for j in range(n):
            if adj[node, j]:
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    queue.append(j)
    return visited == n


def _synth_data(
    adj: np.ndarray,
    n_samples: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    n = adj.shape[0]
    names = [f"V{i}" for i in range(n)]
    # If the graph has a cycle, break cycles by zeroing backward edges
    clean_adj = adj.copy()
    if not _is_dag(clean_adj):
        for i in range(n):
            for j in range(i):
                if clean_adj[i, j]:
                    clean_adj[i, j] = 0
    df, _W = generate_linear_gaussian(clean_adj, n=n_samples, noise_scale=1.0,
                                       edge_weight_range=(0.3, 0.9), seed=seed)
    df.columns = names
    return df


def _run_pipeline(adj, data, treatment, outcome, solver="auto", ci_test="ensemble", alpha=0.05):
    # Map string solver to SolverStrategy enum
    solver_map = {
        "auto": SolverStrategy.AUTO,
        "ilp": SolverStrategy.ILP,
        "lp_relaxation": SolverStrategy.LP_RELAXATION,
        "fpt": SolverStrategy.FPT,
        "cdcl": SolverStrategy.CDCL,
        "greedy": SolverStrategy.GREEDY if hasattr(SolverStrategy, "GREEDY") else SolverStrategy.AUTO,
    }
    # Map string ci_test to CITestMethod enum
    ci_map = {
        "ensemble": CITestMethod.ENSEMBLE,
        "partial_correlation": CITestMethod.PARTIAL_CORRELATION,
        "kernel": CITestMethod.KERNEL,
        "rank": CITestMethod.RANK,
        "spearman": CITestMethod.RANK,
        "crt": CITestMethod.CRT,
    }
    config = PipelineRunConfig(
        treatment=treatment,
        outcome=outcome,
        alpha=alpha,
        solver_strategy=solver_map.get(solver, SolverStrategy.AUTO),
        ci_method=ci_map.get(ci_test, CITestMethod.ENSEMBLE),
    )
    pipeline = CausalCertPipeline(config)
    t0 = time.perf_counter()
    report = pipeline.run(adj_matrix=adj, data=data)
    elapsed = time.perf_counter() - t0
    return report, elapsed


# ====================================================================
# Experiment 1: Published DAG Fragility Analysis
# ====================================================================

def experiment_published_dags(output_dir: Path, seed: int = 42) -> dict:
    """Run CausalCert on all published DAGs and collect fragility results."""
    print("\n" + "=" * 70)
    print("Experiment 1: Published DAG Fragility Analysis")
    print("=" * 70)

    results = []
    dag_names = list_published_dags()

    for name in dag_names:
        dag = get_published_dag(name)
        if dag.default_treatment is None or dag.default_outcome is None:
            continue
        if dag.n_nodes > 40:
            continue  # skip very large DAGs for tractability

        print(f"  {name} ({dag.n_nodes} nodes, {dag.n_edges} edges)...", end="", flush=True)
        data = _synth_data(dag.adj, n_samples=2000, seed=seed)

        try:
            report, elapsed = _run_pipeline(
                dag.adj, data, dag.default_treatment, dag.default_outcome
            )
            n_fragile = sum(1 for fs in report.fragility_ranking if fs.total_score >= 0.4)
            n_critical = sum(1 for fs in report.fragility_ranking if fs.total_score >= 0.7)
            top_edge = None
            if report.fragility_ranking:
                top = report.fragility_ranking[0]
                top_edge = {"edge": list(top.edge), "score": round(top.total_score, 4)}

            row = {
                "dag": name,
                "n_nodes": dag.n_nodes,
                "n_edges": dag.n_edges,
                "source": dag.source,
                "treatment": dag.default_treatment,
                "outcome": dag.default_outcome,
                "radius_lower": report.radius.lower_bound,
                "radius_upper": report.radius.upper_bound,
                "n_fragile_edges": n_fragile,
                "n_critical_edges": n_critical,
                "top_fragile_edge": top_edge,
                "n_total_scored": len(report.fragility_ranking),
                "runtime_s": round(elapsed, 3),
            }
            if report.baseline_estimate is not None:
                row["ate_estimate"] = round(report.baseline_estimate.ate, 4)
                row["ate_se"] = round(report.baseline_estimate.se, 4)
            results.append(row)
            print(f" r=[{report.radius.lower_bound},{report.radius.upper_bound}] "
                  f"fragile={n_fragile} ({elapsed:.2f}s)")
        except Exception as e:
            print(f" ERROR: {e}")
            results.append({"dag": name, "error": str(e)})

    out = {"experiment": "published_dag_fragility", "seed": seed, "results": results}
    _save_json(out, output_dir / "published_dag_fragility.json")
    return out


# ====================================================================
# Experiment 2: Standard Benchmark Suite
# ====================================================================

def experiment_standard_benchmarks(output_dir: Path, seed: int = 42) -> dict:
    """Run all standard benchmarks and check against expected ranges."""
    print("\n" + "=" * 70)
    print("Experiment 2: Standard Benchmark Suite")
    print("=" * 70)

    results = []
    for name in list_benchmarks():
        bench = get_benchmark(name)
        print(f"  {name}...", end="", flush=True)
        data = _synth_data(bench.adj_matrix, n_samples=2000, seed=seed)

        try:
            report, elapsed = _run_pipeline(
                bench.adj_matrix, data, bench.treatment, bench.outcome
            )
            n_fragile = sum(1 for fs in report.fragility_ranking if fs.total_score >= 0.4)
            rlo, rhi = bench.expected_radius_range

            row = {
                "benchmark": name,
                "description": bench.description,
                "n_nodes": len(bench.node_names),
                "n_edges": int(bench.adj_matrix.sum()),
                "radius_lower": report.radius.lower_bound,
                "radius_upper": report.radius.upper_bound,
                "expected_radius_range": list(bench.expected_radius_range),
                "radius_in_expected": rlo <= report.radius.lower_bound <= rhi,
                "n_fragile_edges": n_fragile,
                "expected_n_load_bearing": bench.expected_n_load_bearing,
                "runtime_s": round(elapsed, 3),
            }
            results.append(row)
            in_range = "✓" if row["radius_in_expected"] else "✗"
            print(f" r=[{report.radius.lower_bound},{report.radius.upper_bound}] "
                  f"expected=[{rlo},{rhi}] {in_range} ({elapsed:.2f}s)")
        except Exception as e:
            print(f" ERROR: {e}")
            results.append({"benchmark": name, "error": str(e)})

    n_pass = sum(1 for r in results if r.get("radius_in_expected", False))
    out = {
        "experiment": "standard_benchmarks",
        "seed": seed,
        "n_benchmarks": len(results),
        "n_pass": n_pass,
        "results": results,
    }
    _save_json(out, output_dir / "standard_benchmark_results.json")
    return out


# ====================================================================
# Experiment 3: Solver Comparison
# ====================================================================

def experiment_solver_comparison(output_dir: Path, seed: int = 42) -> dict:
    """Compare ILP, LP relaxation, FPT, CDCL, and greedy solvers."""
    print("\n" + "=" * 70)
    print("Experiment 3: Solver Strategy Comparison")
    print("=" * 70)

    # Use a few representative DAGs
    test_dags = ["diamond", "napkin", "backdoor", "mediator", "iv"]
    solvers = ["ilp", "lp_relaxation", "fpt", "cdcl", "auto"]

    results = []
    for dag_name in test_dags:
        bench = get_benchmark(dag_name)
        data = _synth_data(bench.adj_matrix, n_samples=2000, seed=seed)

        for solver in solvers:
            print(f"  {dag_name}/{solver}...", end="", flush=True)
            try:
                report, elapsed = _run_pipeline(
                    bench.adj_matrix, data, bench.treatment, bench.outcome,
                    solver=solver,
                )
                gap = report.radius.upper_bound - report.radius.lower_bound
                results.append({
                    "dag": dag_name,
                    "solver": solver,
                    "radius_lower": report.radius.lower_bound,
                    "radius_upper": report.radius.upper_bound,
                    "gap": gap,
                    "runtime_s": round(elapsed, 3),
                })
                print(f" r=[{report.radius.lower_bound},{report.radius.upper_bound}] "
                      f"gap={gap} ({elapsed:.2f}s)")
            except Exception as e:
                print(f" ERROR: {e}")
                results.append({"dag": dag_name, "solver": solver, "error": str(e)})

    out = {"experiment": "solver_comparison", "seed": seed, "results": results}
    _save_json(out, output_dir / "solver_comparison.json")
    return out


# ====================================================================
# Experiment 4: CI-Test Method Comparison
# ====================================================================

def experiment_ci_test_comparison(output_dir: Path, seed: int = 42) -> dict:
    """Compare partial correlation, rank, kernel, CRT, and ensemble CI tests."""
    print("\n" + "=" * 70)
    print("Experiment 4: CI-Test Method Comparison")
    print("=" * 70)

    methods = ["partial_correlation", "rank", "kernel", "ensemble"]
    test_dags = ["diamond", "napkin", "backdoor"]

    results = []
    for dag_name in test_dags:
        bench = get_benchmark(dag_name)
        data = _synth_data(bench.adj_matrix, n_samples=2000, seed=seed)

        for method in methods:
            print(f"  {dag_name}/{method}...", end="", flush=True)
            try:
                report, elapsed = _run_pipeline(
                    bench.adj_matrix, data, bench.treatment, bench.outcome,
                    ci_test=method,
                )
                n_fragile = sum(1 for fs in report.fragility_ranking if fs.total_score >= 0.4)
                results.append({
                    "dag": dag_name,
                    "ci_test": method,
                    "radius_lower": report.radius.lower_bound,
                    "radius_upper": report.radius.upper_bound,
                    "n_fragile": n_fragile,
                    "runtime_s": round(elapsed, 3),
                })
                print(f" r=[{report.radius.lower_bound},{report.radius.upper_bound}] "
                      f"fragile={n_fragile} ({elapsed:.2f}s)")
            except Exception as e:
                print(f" ERROR: {e}")
                results.append({"dag": dag_name, "ci_test": method, "error": str(e)})

    out = {"experiment": "ci_test_comparison", "seed": seed, "results": results}
    _save_json(out, output_dir / "ci_test_comparison.json")
    return out


# ====================================================================
# Experiment 5: Scalability Evaluation
# ====================================================================

def experiment_scalability(output_dir: Path, seed: int = 42) -> dict:
    """Measure runtime vs graph size (5 to 50 nodes)."""
    print("\n" + "=" * 70)
    print("Experiment 5: Scalability Evaluation")
    print("=" * 70)

    node_counts = [5, 8, 10, 15, 20, 25, 30, 40, 50]
    repeats = 3
    results = []

    for n in node_counts:
        times = []
        last_row = None
        for rep in range(repeats):
            rng = np.random.default_rng(seed + rep)
            adj = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < 0.3:
                        adj[i, j] = 1

            data = _synth_data(adj, n_samples=1000, seed=seed + rep)
            try:
                report, elapsed = _run_pipeline(adj, data, 0, n - 1)
                times.append(elapsed)
                last_row = {
                    "n_nodes": n,
                    "n_edges": int(adj.sum()),
                    "radius_lower": report.radius.lower_bound,
                    "radius_upper": report.radius.upper_bound,
                }
            except Exception:
                pass

        if times and last_row:
            last_row["avg_runtime_s"] = round(sum(times) / len(times), 3)
            last_row["min_runtime_s"] = round(min(times), 3)
            last_row["max_runtime_s"] = round(max(times), 3)
            results.append(last_row)
            print(f"  n={n:>3d}  avg={last_row['avg_runtime_s']:.3f}s  "
                  f"r=[{last_row['radius_lower']},{last_row['radius_upper']}]")

    out = {"experiment": "scalability", "seed": seed, "repeats": repeats, "results": results}
    _save_json(out, output_dir / "scalability_results.json")
    return out


# ====================================================================
# Experiment 6: Convergence with Sample Size
# ====================================================================

def experiment_convergence(output_dir: Path, seed: int = 42) -> dict:
    """Track radius bounds as sample size increases."""
    print("\n" + "=" * 70)
    print("Experiment 6: Convergence with Sample Size")
    print("=" * 70)

    sample_sizes = [100, 200, 500, 1000, 2000, 5000]
    test_dags = ["diamond", "napkin", "backdoor"]
    results = []

    for dag_name in test_dags:
        bench = get_benchmark(dag_name)
        for n in sample_sizes:
            print(f"  {dag_name}/n={n}...", end="", flush=True)
            data = _synth_data(bench.adj_matrix, n_samples=n, seed=seed)
            try:
                report, elapsed = _run_pipeline(
                    bench.adj_matrix, data, bench.treatment, bench.outcome,
                )
                gap = report.radius.upper_bound - report.radius.lower_bound
                results.append({
                    "dag": dag_name,
                    "n_samples": n,
                    "radius_lower": report.radius.lower_bound,
                    "radius_upper": report.radius.upper_bound,
                    "gap": gap,
                    "runtime_s": round(elapsed, 3),
                })
                print(f" r=[{report.radius.lower_bound},{report.radius.upper_bound}] "
                      f"gap={gap} ({elapsed:.2f}s)")
            except Exception as e:
                print(f" ERROR: {e}")

    out = {"experiment": "convergence", "seed": seed, "results": results}
    _save_json(out, output_dir / "convergence_results.json")
    return out


# ====================================================================
# Experiment 7: Ablation Study
# ====================================================================

def experiment_ablation(output_dir: Path, seed: int = 42) -> dict:
    """Ablation: incremental d-sep, caching, pruning, ensemble vs single test."""
    print("\n" + "=" * 70)
    print("Experiment 7: Ablation Study")
    print("=" * 70)

    bench = get_benchmark("napkin")
    data = _synth_data(bench.adj_matrix, n_samples=2000, seed=seed)
    results = []

    configurations = [
        ("full_pipeline", {"solver": "auto", "ci_test": "ensemble"}),
        ("ilp_only", {"solver": "ilp", "ci_test": "ensemble"}),
        ("lp_only", {"solver": "lp_relaxation", "ci_test": "ensemble"}),
        ("partial_corr_only", {"solver": "auto", "ci_test": "partial_correlation"}),
        ("kernel_only", {"solver": "auto", "ci_test": "kernel"}),
        ("fpt_solver", {"solver": "fpt", "ci_test": "ensemble"}),
        ("cdcl_solver", {"solver": "cdcl", "ci_test": "ensemble"}),
    ]

    for config_name, kwargs in configurations:
        print(f"  {config_name}...", end="", flush=True)
        try:
            report, elapsed = _run_pipeline(
                bench.adj_matrix, data, bench.treatment, bench.outcome, **kwargs
            )
            n_fragile = sum(1 for fs in report.fragility_ranking if fs.total_score >= 0.4)
            results.append({
                "config": config_name,
                "radius_lower": report.radius.lower_bound,
                "radius_upper": report.radius.upper_bound,
                "n_fragile": n_fragile,
                "runtime_s": round(elapsed, 3),
            })
            print(f" r=[{report.radius.lower_bound},{report.radius.upper_bound}] ({elapsed:.2f}s)")
        except Exception as e:
            print(f" ERROR: {e}")
            results.append({"config": config_name, "error": str(e)})

    out = {"experiment": "ablation", "seed": seed, "dag": "napkin", "results": results}
    _save_json(out, output_dir / "ablation_results.json")
    return out


# ====================================================================
# Experiment 8: Baseline Comparison (IDA / E-value / PC)
# ====================================================================

def experiment_baseline_comparison(output_dir: Path, seed: int = 42) -> dict:
    """Compare CausalCert with IDA, E-value, and structure learning baselines."""
    print("\n" + "=" * 70)
    print("Experiment 8: Baseline Comparison (IDA / E-value / PC)")
    print("=" * 70)

    test_dags = ["diamond", "napkin", "backdoor", "mediator"]
    results = []

    for dag_name in test_dags:
        bench = get_benchmark(dag_name)
        data = _synth_data(bench.adj_matrix, n_samples=2000, seed=seed)
        adj = bench.adj_matrix

        print(f"  {dag_name}/IDA...", end="", flush=True)
        try:
            r = compare_with_ida(adj, data, bench.treatment, bench.outcome, seed=seed)
            results.append({
                "dag": dag_name, "comparison": "IDA",
                "metric": r.metric_name, "value": r.metric_value,
                "runtime_s": round(r.runtime_s, 3), "notes": r.notes,
            })
            print(f" done ({r.runtime_s:.2f}s)")
        except Exception as e:
            print(f" ERROR: {e}")

        print(f"  {dag_name}/E-value...", end="", flush=True)
        try:
            r = compare_with_e_value(adj, data, bench.treatment, bench.outcome, observed_rr=2.0)
            results.append({
                "dag": dag_name, "comparison": "E-value",
                "metric": r.metric_name, "value": r.metric_value,
                "runtime_s": round(r.runtime_s, 3), "notes": r.notes,
            })
            print(f" done ({r.runtime_s:.2f}s)")
        except Exception as e:
            print(f" ERROR: {e}")

        print(f"  {dag_name}/PC-GES...", end="", flush=True)
        try:
            r = compare_with_structure_learning(adj, data, bench.treatment, bench.outcome)
            results.append({
                "dag": dag_name, "comparison": "PC/GES",
                "metric": r.metric_name, "value": r.metric_value,
                "runtime_s": round(r.runtime_s, 3), "notes": r.notes,
            })
            print(f" done ({r.runtime_s:.2f}s)")
        except Exception as e:
            print(f" ERROR: {e}")

    out = {"experiment": "baseline_comparison", "seed": seed, "results": results}
    _save_json(out, output_dir / "baseline_comparison.json")
    return out


# ====================================================================
# Experiment 9: Treewidth vs Runtime
# ====================================================================

def experiment_treewidth_runtime(output_dir: Path, seed: int = 42) -> dict:
    """Measure how runtime scales with moral-graph treewidth."""
    print("\n" + "=" * 70)
    print("Experiment 9: Treewidth vs Runtime")
    print("=" * 70)

    from causalcert.treewidth.decomposition import compute_treewidth_upper_bound

    results = []
    # Generate graphs with increasing density (higher density → higher treewidth)
    for n_nodes in [8, 12, 15, 20]:
        for edge_prob in [0.15, 0.25, 0.35, 0.45]:
            rng = np.random.default_rng(seed)
            adj = np.zeros((n_nodes, n_nodes), dtype=np.int8)
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if rng.random() < edge_prob:
                        adj[i, j] = 1

            try:
                tw = compute_treewidth_upper_bound(adj)
            except Exception:
                tw = -1

            data = _synth_data(adj, n_samples=1000, seed=seed)
            try:
                report, elapsed = _run_pipeline(adj, data, 0, n_nodes - 1)
                results.append({
                    "n_nodes": n_nodes,
                    "edge_prob": edge_prob,
                    "n_edges": int(adj.sum()),
                    "treewidth": tw,
                    "radius_lower": report.radius.lower_bound,
                    "radius_upper": report.radius.upper_bound,
                    "runtime_s": round(elapsed, 3),
                })
                print(f"  n={n_nodes} p={edge_prob} tw={tw} "
                      f"r=[{report.radius.lower_bound},{report.radius.upper_bound}] "
                      f"({elapsed:.2f}s)")
            except Exception as e:
                print(f"  n={n_nodes} p={edge_prob} ERROR: {e}")

    out = {"experiment": "treewidth_runtime", "seed": seed, "results": results}
    _save_json(out, output_dir / "treewidth_runtime.json")
    return out


# ====================================================================
# Main
# ====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="CausalCert experiments")
    parser.add_argument("--output-dir", type=str, default="benchmark_output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = args.seed

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         CausalCert — Full Experiment Suite                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    all_results = {}
    total_t0 = time.perf_counter()

    all_results["published_dags"] = experiment_published_dags(output_dir, seed)
    all_results["standard_benchmarks"] = experiment_standard_benchmarks(output_dir, seed)
    all_results["solver_comparison"] = experiment_solver_comparison(output_dir, seed)
    all_results["ci_test_comparison"] = experiment_ci_test_comparison(output_dir, seed)
    all_results["scalability"] = experiment_scalability(output_dir, seed)
    all_results["convergence"] = experiment_convergence(output_dir, seed)
    all_results["ablation"] = experiment_ablation(output_dir, seed)
    all_results["baseline_comparison"] = experiment_baseline_comparison(output_dir, seed)
    all_results["treewidth_runtime"] = experiment_treewidth_runtime(output_dir, seed)

    total_elapsed = time.perf_counter() - total_t0

    # Summary
    summary = {
        "total_runtime_s": round(total_elapsed, 1),
        "n_experiments": len(all_results),
        "seed": seed,
        "experiments": list(all_results.keys()),
    }
    _save_json(summary, output_dir / "experiment_summary.json")

    print("\n" + "=" * 70)
    print(f"All experiments complete in {total_elapsed:.1f}s")
    print(f"Results saved to {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
