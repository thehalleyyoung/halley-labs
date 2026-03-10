"""Stress tests: large graphs, high-dimensional data, and edge cases.

Use this module to push CausalCert to its limits and find performance
regressions or numerical issues.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray


# ====================================================================
# Data containers
# ====================================================================

@dataclass
class StressResult:
    """Outcome of a single stress test."""
    scenario: str
    n_nodes: int
    n_edges: int
    n_samples: int
    succeeded: bool
    runtime_s: float
    radius_lower: int | None = None
    radius_upper: int | None = None
    error_message: str = ""
    peak_memory_mb: float = 0.0


@dataclass
class StressSuite:
    """Collected results from running the full stress suite."""
    results: list[StressResult] = field(default_factory=list)

    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.succeeded)

    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.succeeded)

    def summary(self) -> str:
        lines = [
            f"Stress suite: {self.n_passed()}/{len(self.results)} passed",
        ]
        for r in self.results:
            status = "✓" if r.succeeded else "✗"
            lines.append(
                f"  {status} {r.scenario:<40s}  "
                f"|V|={r.n_nodes:>4d}  |E|={r.n_edges:>5d}  "
                f"n={r.n_samples:>6d}  {r.runtime_s:.2f}s"
            )
            if r.error_message:
                lines.append(f"      Error: {r.error_message}")
        return "\n".join(lines)


# ====================================================================
# 1. Large Graph Generation
# ====================================================================

def generate_large_random_dag(
    n_nodes: int,
    edge_prob: float = 0.05,
    seed: int = 42,
) -> tuple[NDArray[np.int8], list[str]]:
    """Generate a large sparse DAG (lower-triangular Erdős–Rényi)."""
    rng = np.random.default_rng(seed)
    names = [f"V{i}" for i in range(n_nodes)]
    adj = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_prob:
                adj[i, j] = 1
    return adj, names


def generate_wide_layered_dag(
    n_layers: int,
    width: int,
    inter_prob: float = 0.3,
    seed: int = 42,
) -> tuple[NDArray[np.int8], list[str]]:
    """Generate a wide layered DAG for bandwidth stress testing."""
    rng = np.random.default_rng(seed)
    n = n_layers * width
    names = [f"L{l}_N{i}" for l in range(n_layers) for i in range(width)]
    adj = np.zeros((n, n), dtype=np.int8)

    for layer in range(n_layers - 1):
        for i in range(width):
            src = layer * width + i
            for j in range(width):
                dst = (layer + 1) * width + j
                if rng.random() < inter_prob:
                    adj[src, dst] = 1

    return adj, names


def generate_dense_dag(
    n_nodes: int,
    seed: int = 42,
) -> tuple[NDArray[np.int8], list[str]]:
    """Maximally dense DAG: every i→j for i < j."""
    names = [f"V{i}" for i in range(n_nodes)]
    adj = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            adj[i, j] = 1
    return adj, names


def generate_tree_dag(
    depth: int,
    branching: int = 2,
    seed: int = 42,
) -> tuple[NDArray[np.int8], list[str]]:
    """Generate a perfect k-ary tree DAG."""
    nodes: list[str] = ["root"]
    adj_list: list[tuple[int, int]] = []
    queue = [0]

    for _ in range(depth):
        next_queue: list[int] = []
        for parent in queue:
            for b in range(branching):
                child_id = len(nodes)
                nodes.append(f"N{child_id}")
                adj_list.append((parent, child_id))
                next_queue.append(child_id)
        queue = next_queue

    n = len(nodes)
    adj = np.zeros((n, n), dtype=np.int8)
    for src, dst in adj_list:
        adj[src, dst] = 1
    return adj, nodes


def generate_bipartite_dag(
    n_left: int,
    n_right: int,
    edge_prob: float = 0.3,
    seed: int = 42,
) -> tuple[NDArray[np.int8], list[str]]:
    """Bipartite DAG: all edges go from left partition to right."""
    rng = np.random.default_rng(seed)
    n = n_left + n_right
    names = [f"L{i}" for i in range(n_left)] + [f"R{i}" for i in range(n_right)]
    adj = np.zeros((n, n), dtype=np.int8)

    for i in range(n_left):
        for j in range(n_left, n):
            if rng.random() < edge_prob:
                adj[i, j] = 1

    return adj, names


# ====================================================================
# 2. High-Dimensional Data Generation
# ====================================================================

def generate_high_dim_data(
    adj: NDArray[np.int8],
    names: list[str],
    n_samples: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate data from a linear-Gaussian SEM."""
    from causalcert.data.synthetic import generate_linear_gaussian

    df, _W = generate_linear_gaussian(
        adj, n=n_samples, noise_scale=1.0,
        edge_weight_range=(0.3, 0.9), seed=seed,
    )
    df.columns = names
    return df


def generate_sparse_data(
    n_nodes: int,
    n_samples: int,
    sparsity: float = 0.9,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate data where most values are zero (sparse regime)."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_samples, n_nodes))
    mask = rng.random((n_samples, n_nodes)) < sparsity
    data[mask] = 0.0
    return pd.DataFrame(data, columns=[f"V{i}" for i in range(n_nodes)])


# ====================================================================
# 3. Edge Case Scenarios
# ====================================================================

def disconnected_dag(n_components: int = 3, nodes_per: int = 5) -> tuple[NDArray[np.int8], list[str]]:
    """DAG with multiple disconnected components."""
    n = n_components * nodes_per
    names = [f"C{c}_V{i}" for c in range(n_components) for i in range(nodes_per)]
    adj = np.zeros((n, n), dtype=np.int8)
    for c in range(n_components):
        offset = c * nodes_per
        for i in range(nodes_per - 1):
            adj[offset + i, offset + i + 1] = 1
    return adj, names


def single_edge_dag() -> tuple[NDArray[np.int8], list[str]]:
    """Minimal DAG: just X → Y."""
    names = ["X", "Y"]
    adj = np.zeros((2, 2), dtype=np.int8)
    adj[0, 1] = 1
    return adj, names


def empty_dag(n: int = 5) -> tuple[NDArray[np.int8], list[str]]:
    """DAG with no edges."""
    names = [f"V{i}" for i in range(n)]
    return np.zeros((n, n), dtype=np.int8), names


def star_dag(n_leaves: int = 10) -> tuple[NDArray[np.int8], list[str]]:
    """Star graph: central hub → all leaves."""
    n = 1 + n_leaves
    names = ["Hub"] + [f"Leaf{i}" for i in range(n_leaves)]
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(1, n):
        adj[0, i] = 1
    return adj, names


def long_chain_dag(length: int = 100) -> tuple[NDArray[np.int8], list[str]]:
    """Very long chain for testing linear-time algorithms."""
    names = [f"V{i}" for i in range(length)]
    adj = np.zeros((length, length), dtype=np.int8)
    for i in range(length - 1):
        adj[i, i + 1] = 1
    return adj, names


# ====================================================================
# 4. Stress Test Runner
# ====================================================================

def _run_one_stress(
    scenario: str,
    adj: NDArray[np.int8],
    names: list[str],
    n_samples: int,
    seed: int,
    timeout_s: float = 120.0,
) -> StressResult:
    """Run CausalCert on a single stress scenario."""
    from causalcert.pipeline.config import PipelineRunConfig
    from causalcert.pipeline.orchestrator import CausalCertPipeline
    from causalcert.types import SolverStrategy

    treatment = 0
    outcome = adj.shape[0] - 1

    try:
        df = generate_high_dim_data(adj, names, n_samples, seed)
        config = PipelineRunConfig(
            treatment=treatment, outcome=outcome,
            alpha=0.05, solver_strategy=SolverStrategy.AUTO,
        )
        pipeline = CausalCertPipeline(config)
        t0 = time.perf_counter()
        report = pipeline.run(adj_matrix=adj, data=df)
        elapsed = time.perf_counter() - t0

        return StressResult(
            scenario=scenario,
            n_nodes=len(names),
            n_edges=int(adj.sum()),
            n_samples=n_samples,
            succeeded=True,
            runtime_s=elapsed,
            radius_lower=report.radius.lower_bound,
            radius_upper=report.radius.upper_bound,
        )
    except Exception as exc:
        return StressResult(
            scenario=scenario,
            n_nodes=len(names),
            n_edges=int(adj.sum()),
            n_samples=n_samples,
            succeeded=False,
            runtime_s=0.0,
            error_message=str(exc),
        )


def run_stress_suite(
    seed: int = 42,
    n_samples: int = 500,
) -> StressSuite:
    """Run the full stress test suite and return collected results."""
    suite = StressSuite()

    scenarios: list[tuple[str, NDArray[np.int8], list[str], int]] = [
        ("single-edge", *single_edge_dag(), n_samples),
        ("empty-5", *empty_dag(5), n_samples),
        ("star-10", *star_dag(10), n_samples),
        ("chain-50", *long_chain_dag(50), n_samples),
        ("disconnected-3x5", *disconnected_dag(3, 5), n_samples),
        ("dense-8", *generate_dense_dag(8, seed), n_samples),
        ("sparse-50", *generate_large_random_dag(50, 0.03, seed), n_samples),
        ("layered-5x10", *generate_wide_layered_dag(5, 10, 0.2, seed), n_samples),
        ("tree-d4-b2", *generate_tree_dag(4, 2, seed), n_samples),
        ("bipartite-10x10", *generate_bipartite_dag(10, 10, 0.2, seed), n_samples),
        ("large-100-sparse", *generate_large_random_dag(100, 0.02, seed), n_samples),
    ]

    for name, adj, names, ns in scenarios:
        result = _run_one_stress(name, adj, names, ns, seed)
        suite.results.append(result)

    return suite
