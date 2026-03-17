#!/usr/bin/env python3
"""
Benchmark suite for the Spectral Decomposition Oracle.

Generates synthetic sparse matrices and simulates spectral feature extraction
to produce timing benchmarks across varying matrix sizes. All computations are
pure Python (no Rust FFI) — this is a simulation harness for CI and planning.

Usage:
    python run_benchmarks.py [--sizes 100,500,1000] [--trials 5] [--output results/]
"""

import argparse
import csv
import json
import math
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Synthetic sparse matrix generation
# ---------------------------------------------------------------------------

@dataclass
class SparseMatrixCOO:
    """Coordinate-format sparse matrix."""
    rows: int
    cols: int
    row_idx: List[int]
    col_idx: List[int]
    values: List[float]

    @property
    def nnz(self) -> int:
        return len(self.values)

    @property
    def density(self) -> float:
        return self.nnz / (self.rows * self.cols) if self.rows * self.cols > 0 else 0.0


def generate_sparse_matrix(n: int, density: float = 0.02, seed: int = 42) -> SparseMatrixCOO:
    """Generate a random symmetric sparse matrix in COO format."""
    rng = random.Random(seed)
    target_nnz = max(1, int(n * n * density / 2))
    row_idx, col_idx, values = [], [], []

    entries = set()
    while len(entries) < target_nnz:
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i != j and (i, j) not in entries:
            v = rng.gauss(0, 1.0)
            entries.add((i, j))
            entries.add((j, i))
            row_idx.extend([i, j])
            col_idx.extend([j, i])
            values.extend([v, v])

    # Add diagonal dominance
    diag_vals = [0.0] * n
    for idx, r in enumerate(row_idx):
        diag_vals[r] += abs(values[idx])
    for i in range(n):
        row_idx.append(i)
        col_idx.append(i)
        values.append(diag_vals[i] + rng.uniform(0.1, 1.0))

    return SparseMatrixCOO(n, n, row_idx, col_idx, values)


# ---------------------------------------------------------------------------
# Simulated spectral operations
# ---------------------------------------------------------------------------

def simulate_laplacian_construction(matrix: SparseMatrixCOO) -> float:
    """Simulate building the normalized Laplacian. Returns elapsed seconds."""
    start = time.perf_counter()
    # Simulate degree computation
    degrees = [0.0] * matrix.rows
    for idx, r in enumerate(matrix.row_idx):
        degrees[r] += abs(matrix.values[idx])
    # Simulate D^{-1/2} scaling
    inv_sqrt_deg = [1.0 / math.sqrt(d) if d > 0 else 0.0 for d in degrees]
    # Simulate element-wise scaling (L_norm = D^{-1/2} A D^{-1/2})
    _ = [
        matrix.values[k] * inv_sqrt_deg[matrix.row_idx[k]] * inv_sqrt_deg[matrix.col_idx[k]]
        for k in range(matrix.nnz)
    ]
    return time.perf_counter() - start


def simulate_eigensolve(n: int, k: int = 8) -> Tuple[float, List[float]]:
    """Simulate eigenvalue computation via Lanczos. Returns (elapsed, eigenvalues)."""
    start = time.perf_counter()
    rng = random.Random(n)
    # Simulate Lanczos iterations: O(k * nnz) work
    iterations = min(k * 20, n)
    basis = []
    for _ in range(iterations):
        vec = [rng.gauss(0, 1) for _ in range(min(n, 500))]
        norm = math.sqrt(sum(v * v for v in vec))
        basis.append([v / norm for v in vec])

    # Produce synthetic eigenvalues with realistic spectral gap
    eigenvalues = sorted([rng.uniform(0, 0.01)] + [rng.uniform(0.05, 2.0) for _ in range(k - 1)])
    elapsed = time.perf_counter() - start
    return elapsed, eigenvalues


def simulate_feature_extraction(eigenvalues: List[float]) -> Tuple[float, Dict[str, float]]:
    """Extract 8 spectral features from eigenvalues. Returns (elapsed, features)."""
    start = time.perf_counter()
    spectral_gap = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
    spectral_radius = max(eigenvalues)
    spectral_width = spectral_radius - min(eigenvalues)

    # Eigenvalue entropy
    total = sum(eigenvalues) or 1.0
    probs = [ev / total for ev in eigenvalues if ev > 0]
    entropy = -sum(p * math.log(p + 1e-15) for p in probs)

    # Decay rate: ratio of consecutive eigenvalues
    decay_rate = eigenvalues[1] / eigenvalues[-1] if len(eigenvalues) > 1 and eigenvalues[-1] > 0 else 0.0

    features = {
        "spectral_gap": spectral_gap,
        "algebraic_connectivity": spectral_gap,
        "fiedler_entropy": entropy,
        "spectral_radius": spectral_radius,
        "spectral_width": spectral_width,
        "normalized_cut_est": spectral_gap / 2.0,
        "cheeger_est": math.sqrt(2.0 * spectral_gap),
        "eigenvalue_decay_rate": decay_rate,
    }
    elapsed = time.perf_counter() - start
    return elapsed, features


def simulate_baseline_syntactic(matrix: SparseMatrixCOO) -> Tuple[float, Dict[str, float]]:
    """Baseline: syntactic features only (no spectral analysis)."""
    start = time.perf_counter()
    features = {
        "num_rows": float(matrix.rows),
        "num_cols": float(matrix.cols),
        "nnz": float(matrix.nnz),
        "density": matrix.density,
        "avg_nnz_per_row": matrix.nnz / matrix.rows if matrix.rows > 0 else 0.0,
    }
    elapsed = time.perf_counter() - start
    return elapsed, features


# ---------------------------------------------------------------------------
# Benchmark result types
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    matrix_size: int
    nnz: int
    density: float
    laplacian_time_ms: float
    eigensolve_time_ms: float
    feature_extraction_time_ms: float
    total_spectral_time_ms: float
    baseline_time_ms: float
    spectral_overhead_factor: float
    eigenvalues: List[float]
    features: Dict[str, float]
    trial: int = 0


@dataclass
class BenchmarkSummary:
    matrix_size: int
    num_trials: int
    mean_total_ms: float
    std_total_ms: float
    median_total_ms: float
    min_total_ms: float
    max_total_ms: float
    mean_overhead: float


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_single_benchmark(n: int, trial: int = 0, density: float = 0.02) -> BenchmarkResult:
    """Run a single benchmark for matrix size n."""
    matrix = generate_sparse_matrix(n, density=density, seed=42 + trial)

    lap_time = simulate_laplacian_construction(matrix)
    eig_time, eigenvalues = simulate_eigensolve(n)
    feat_time, features = simulate_feature_extraction(eigenvalues)
    base_time, _ = simulate_baseline_syntactic(matrix)

    total_spectral = lap_time + eig_time + feat_time
    overhead = total_spectral / base_time if base_time > 0 else float("inf")

    return BenchmarkResult(
        matrix_size=n,
        nnz=matrix.nnz,
        density=matrix.density,
        laplacian_time_ms=lap_time * 1000,
        eigensolve_time_ms=eig_time * 1000,
        feature_extraction_time_ms=feat_time * 1000,
        total_spectral_time_ms=total_spectral * 1000,
        baseline_time_ms=base_time * 1000,
        spectral_overhead_factor=overhead,
        eigenvalues=eigenvalues,
        features=features,
        trial=trial,
    )


def run_benchmark_suite(
    sizes: List[int], num_trials: int = 5, density: float = 0.02
) -> Tuple[List[BenchmarkResult], List[BenchmarkSummary]]:
    """Run benchmarks across all sizes and trials."""
    all_results: List[BenchmarkResult] = []
    summaries: List[BenchmarkSummary] = []

    for n in sizes:
        size_results = []
        print(f"  Benchmarking n={n:>6d} ... ", end="", flush=True)

        for trial in range(num_trials):
            result = run_single_benchmark(n, trial=trial, density=density)
            size_results.append(result)
            all_results.append(result)

        totals = [r.total_spectral_time_ms for r in size_results]
        overheads = [r.spectral_overhead_factor for r in size_results]

        summary = BenchmarkSummary(
            matrix_size=n,
            num_trials=num_trials,
            mean_total_ms=statistics.mean(totals),
            std_total_ms=statistics.stdev(totals) if len(totals) > 1 else 0.0,
            median_total_ms=statistics.median(totals),
            min_total_ms=min(totals),
            max_total_ms=max(totals),
            mean_overhead=statistics.mean(overheads),
        )
        summaries.append(summary)
        print(f"mean={summary.mean_total_ms:.2f}ms  overhead={summary.mean_overhead:.1f}x")

    return all_results, summaries


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_json(results: List[BenchmarkResult], summaries: List[BenchmarkSummary], path: str):
    """Write benchmark results to JSON."""
    data = {
        "metadata": {
            "tool": "spectral-decomposition-oracle",
            "benchmark_type": "simulated_python",
            "description": "Synthetic benchmark of spectral feature extraction pipeline",
        },
        "summaries": [asdict(s) for s in summaries],
        "detailed_results": [asdict(r) for r in results],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  JSON results written to {path}")


def write_csv(summaries: List[BenchmarkSummary], path: str):
    """Write summary results to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(summaries[0]).keys()))
        writer.writeheader()
        for s in summaries:
            writer.writerow(asdict(s))
    print(f"  CSV summary written to {path}")


def print_summary_table(summaries: List[BenchmarkSummary]):
    """Print a formatted summary table to stdout."""
    print()
    print("=" * 80)
    print(f"{'Size':>8} {'Trials':>7} {'Mean(ms)':>10} {'Std(ms)':>10} "
          f"{'Med(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10} {'Overhead':>10}")
    print("-" * 80)
    for s in summaries:
        print(f"{s.matrix_size:>8d} {s.num_trials:>7d} {s.mean_total_ms:>10.2f} "
              f"{s.std_total_ms:>10.2f} {s.median_total_ms:>10.2f} "
              f"{s.min_total_ms:>10.2f} {s.max_total_ms:>10.2f} "
              f"{s.mean_overhead:>9.1f}x")
    print("=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run spectral decomposition oracle benchmarks (simulated)"
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="100,500,1000,5000,10000",
        help="Comma-separated matrix sizes (default: 100,500,1000,5000,10000)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of trials per size (default: 5)",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.02,
        help="Sparse matrix density (default: 0.02)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results",
        help="Output directory for results (default: benchmarks/results)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sizes = [int(s.strip()) for s in args.sizes.split(",")]

    print("Spectral Decomposition Oracle — Benchmark Suite")
    print(f"  Sizes:   {sizes}")
    print(f"  Trials:  {args.trials}")
    print(f"  Density: {args.density}")
    print()

    results, summaries = run_benchmark_suite(sizes, args.trials, args.density)

    print_summary_table(summaries)

    os.makedirs(args.output, exist_ok=True)
    json_path = os.path.join(args.output, "benchmark_results.json")
    csv_path = os.path.join(args.output, "benchmark_summary.csv")

    write_json(results, summaries, json_path)
    write_csv(summaries, csv_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
