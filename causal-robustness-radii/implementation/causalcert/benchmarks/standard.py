"""Standard benchmark DAGs with full specifications, datasets, and baselines.

This module provides a reproducible set of benchmark scenarios that can be
used for regression testing, method validation, and paper replication.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray


# ====================================================================
# Data containers
# ====================================================================

@dataclass(frozen=True)
class BenchmarkDAG:
    """Complete specification of a benchmark DAG scenario."""
    name: str
    description: str
    adj_matrix: NDArray[np.int8]
    node_names: list[str]
    treatment: int
    outcome: int
    expected_radius_range: tuple[int, int]  # (min_expected, max_expected)
    expected_n_load_bearing: int
    true_ate: float | None = None
    source: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class BenchmarkDataset:
    """Observational data paired with a ground-truth DAG."""
    dag: BenchmarkDAG
    data: pd.DataFrame
    seed: int = 42


@dataclass
class BenchmarkBaseline:
    """Expected outputs for a benchmark — used for regression testing."""
    dag_name: str
    solver: str
    radius_lower: int
    radius_upper: int
    n_fragile_edges: int
    runtime_s_upper_bound: float
    top_fragile_edge: tuple[int, int] | None = None


# ====================================================================
# Standard DAGs
# ====================================================================

def _chain(n: int) -> tuple[NDArray[np.int8], list[str]]:
    names = [f"V{i}" for i in range(n)]
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n - 1):
        adj[i, i + 1] = 1
    return adj, names


def _fork() -> tuple[NDArray[np.int8], list[str]]:
    """X ← Z → Y (common cause)."""
    names = ["Z", "X", "Y"]
    adj = np.zeros((3, 3), dtype=np.int8)
    adj[0, 1] = 1  # Z → X
    adj[0, 2] = 1  # Z → Y
    return adj, names


def _collider() -> tuple[NDArray[np.int8], list[str]]:
    """X → Z ← Y (collider)."""
    names = ["X", "Y", "Z"]
    adj = np.zeros((3, 3), dtype=np.int8)
    adj[0, 2] = 1  # X → Z
    adj[1, 2] = 1  # Y → Z
    return adj, names


def _mediator() -> tuple[NDArray[np.int8], list[str]]:
    """X → M → Y with X → Y direct."""
    names = ["X", "M", "Y"]
    adj = np.zeros((3, 3), dtype=np.int8)
    adj[0, 1] = 1
    adj[1, 2] = 1
    adj[0, 2] = 1
    return adj, names


def _iv_graph() -> tuple[NDArray[np.int8], list[str]]:
    """Instrumental variable: Z → X → Y, U → X, U → Y (U observed for simplicity)."""
    names = ["Z", "X", "Y", "U"]
    adj = np.zeros((4, 4), dtype=np.int8)
    adj[0, 1] = 1  # Z → X
    adj[1, 2] = 1  # X → Y
    adj[3, 1] = 1  # U → X
    adj[3, 2] = 1  # U → Y
    return adj, names


def _backdoor_graph() -> tuple[NDArray[np.int8], list[str]]:
    """Classic backdoor: C → X, C → Y, X → Y."""
    names = ["C", "X", "Y"]
    adj = np.zeros((3, 3), dtype=np.int8)
    adj[0, 1] = 1
    adj[0, 2] = 1
    adj[1, 2] = 1
    return adj, names


def _frontdoor_graph() -> tuple[NDArray[np.int8], list[str]]:
    """Front-door criterion: U → X, U → Y, X → M → Y."""
    names = ["U", "X", "M", "Y"]
    adj = np.zeros((4, 4), dtype=np.int8)
    adj[0, 1] = 1  # U → X
    adj[0, 3] = 1  # U → Y
    adj[1, 2] = 1  # X → M
    adj[2, 3] = 1  # M → Y
    return adj, names


def _diamond_graph() -> tuple[NDArray[np.int8], list[str]]:
    """Diamond: X → M1, X → M2, M1 → Y, M2 → Y."""
    names = ["X", "M1", "M2", "Y"]
    adj = np.zeros((4, 4), dtype=np.int8)
    adj[0, 1] = 1
    adj[0, 2] = 1
    adj[1, 3] = 1
    adj[2, 3] = 1
    return adj, names


def _napkin_graph() -> tuple[NDArray[np.int8], list[str]]:
    """The Napkin graph (Pearl): W1 → Z, Z → X, X → Y, W1 → W2, W2 → Y, W1 → X."""
    names = ["W1", "Z", "X", "Y", "W2"]
    adj = np.zeros((5, 5), dtype=np.int8)
    adj[0, 1] = 1  # W1 → Z
    adj[1, 2] = 1  # Z → X
    adj[2, 3] = 1  # X → Y
    adj[0, 4] = 1  # W1 → W2
    adj[4, 3] = 1  # W2 → Y
    adj[0, 2] = 1  # W1 → X
    return adj, names


def _bow_graph() -> tuple[NDArray[np.int8], list[str]]:
    """Bow graph: X → Y, U → X, U → Y (unconfounded IV not available)."""
    names = ["U", "X", "Y"]
    adj = np.zeros((3, 3), dtype=np.int8)
    adj[0, 1] = 1
    adj[0, 2] = 1
    adj[1, 2] = 1
    return adj, names


# ====================================================================
# Registry
# ====================================================================

_STANDARD_BENCHMARKS: dict[str, BenchmarkDAG] = {}


def _register(b: BenchmarkDAG) -> BenchmarkDAG:
    _STANDARD_BENCHMARKS[b.name] = b
    return b


# --- Registration ---

_chain5_adj, _chain5_names = _chain(5)
_register(BenchmarkDAG(
    name="chain-5",
    description="Simple 5-node chain V0→V1→…→V4",
    adj_matrix=_chain5_adj, node_names=_chain5_names,
    treatment=0, outcome=4,
    expected_radius_range=(1, 2), expected_n_load_bearing=4,
    true_ate=None, source="synthetic", tags=["chain", "small"],
))

_chain10_adj, _chain10_names = _chain(10)
_register(BenchmarkDAG(
    name="chain-10",
    description="10-node chain",
    adj_matrix=_chain10_adj, node_names=_chain10_names,
    treatment=0, outcome=9,
    expected_radius_range=(1, 2), expected_n_load_bearing=9,
    source="synthetic", tags=["chain", "medium"],
))

_fork_adj, _fork_names = _fork()
_register(BenchmarkDAG(
    name="fork",
    description="Common-cause fork Z→X, Z→Y",
    adj_matrix=_fork_adj, node_names=_fork_names,
    treatment=1, outcome=2,
    expected_radius_range=(1, 3), expected_n_load_bearing=1,
    source="synthetic", tags=["elemental", "small"],
))

_coll_adj, _coll_names = _collider()
_register(BenchmarkDAG(
    name="collider",
    description="Collider X→Z←Y",
    adj_matrix=_coll_adj, node_names=_coll_names,
    treatment=0, outcome=1,
    expected_radius_range=(1, 3), expected_n_load_bearing=0,
    source="synthetic", tags=["elemental", "small"],
))

_med_adj, _med_names = _mediator()
_register(BenchmarkDAG(
    name="mediator",
    description="Mediator X→M→Y with direct X→Y",
    adj_matrix=_med_adj, node_names=_med_names,
    treatment=0, outcome=2,
    expected_radius_range=(2, 4), expected_n_load_bearing=1,
    true_ate=1.0, source="synthetic", tags=["elemental", "small"],
))

_iv_adj, _iv_names = _iv_graph()
_register(BenchmarkDAG(
    name="iv",
    description="Instrumental variable graph",
    adj_matrix=_iv_adj, node_names=_iv_names,
    treatment=1, outcome=2,
    expected_radius_range=(1, 3), expected_n_load_bearing=2,
    source="synthetic", tags=["identification", "small"],
))

_bd_adj, _bd_names = _backdoor_graph()
_register(BenchmarkDAG(
    name="backdoor",
    description="Classic back-door C→X, C→Y, X→Y",
    adj_matrix=_bd_adj, node_names=_bd_names,
    treatment=1, outcome=2,
    expected_radius_range=(1, 3), expected_n_load_bearing=1,
    source="synthetic", tags=["identification", "small"],
))

_fd_adj, _fd_names = _frontdoor_graph()
_register(BenchmarkDAG(
    name="frontdoor",
    description="Front-door criterion graph",
    adj_matrix=_fd_adj, node_names=_fd_names,
    treatment=1, outcome=3,
    expected_radius_range=(1, 3), expected_n_load_bearing=2,
    source="synthetic", tags=["identification", "small"],
))

_dia_adj, _dia_names = _diamond_graph()
_register(BenchmarkDAG(
    name="diamond",
    description="Diamond X→M1,M2→Y",
    adj_matrix=_dia_adj, node_names=_dia_names,
    treatment=0, outcome=3,
    expected_radius_range=(2, 4), expected_n_load_bearing=2,
    source="synthetic", tags=["mediation", "small"],
))

_nap_adj, _nap_names = _napkin_graph()
_register(BenchmarkDAG(
    name="napkin",
    description="Pearl's Napkin problem",
    adj_matrix=_nap_adj, node_names=_nap_names,
    treatment=2, outcome=3,
    expected_radius_range=(1, 3), expected_n_load_bearing=2,
    source="Pearl (2009)", tags=["identification", "medium"],
))

_bow_adj, _bow_names = _bow_graph()
_register(BenchmarkDAG(
    name="bow",
    description="Bow graph (unmeasured confounding)",
    adj_matrix=_bow_adj, node_names=_bow_names,
    treatment=1, outcome=2,
    expected_radius_range=(1, 2), expected_n_load_bearing=1,
    source="synthetic", tags=["confounding", "small"],
))


# ====================================================================
# Public API
# ====================================================================

def list_benchmarks(tags: list[str] | None = None) -> list[str]:
    """List available benchmark names, optionally filtered by tags."""
    if tags is None:
        return list(_STANDARD_BENCHMARKS.keys())
    return [
        name for name, b in _STANDARD_BENCHMARKS.items()
        if any(t in b.tags for t in tags)
    ]


def get_benchmark(name: str) -> BenchmarkDAG:
    """Retrieve a benchmark DAG by name."""
    if name not in _STANDARD_BENCHMARKS:
        available = ", ".join(sorted(_STANDARD_BENCHMARKS.keys()))
        raise KeyError(f"Unknown benchmark '{name}'. Available: {available}")
    return _STANDARD_BENCHMARKS[name]


def generate_benchmark_data(
    name: str,
    n_samples: int = 1000,
    seed: int = 42,
) -> BenchmarkDataset:
    """Generate synthetic data for a named benchmark."""
    from causalcert.data.synthetic import generate_linear_gaussian

    b = get_benchmark(name)
    rng = np.random.default_rng(seed)
    weights = b.adj_matrix.astype(np.float64) * rng.uniform(
        0.3, 0.9, size=b.adj_matrix.shape,
    )
    data = generate_linear_gaussian(
        adj_matrix=b.adj_matrix,
        weights=weights,
        n_samples=n_samples,
        noise_scale=1.0,
        seed=seed,
    )
    df = pd.DataFrame(data, columns=b.node_names)
    return BenchmarkDataset(dag=b, data=df, seed=seed)


# ====================================================================
# Baselines (expected results for regression testing)
# ====================================================================

_BASELINES: list[BenchmarkBaseline] = [
    BenchmarkBaseline("chain-5", "ilp", 1, 1, 4, 5.0, (0, 1)),
    BenchmarkBaseline("chain-5", "lp_relaxation", 1, 2, 4, 2.0, (0, 1)),
    BenchmarkBaseline("fork", "ilp", 1, 1, 1, 3.0, (0, 1)),
    BenchmarkBaseline("mediator", "ilp", 2, 2, 1, 5.0, (0, 2)),
    BenchmarkBaseline("diamond", "ilp", 2, 2, 2, 5.0, (0, 1)),
    BenchmarkBaseline("iv", "ilp", 1, 1, 2, 5.0, (1, 2)),
    BenchmarkBaseline("backdoor", "ilp", 1, 1, 1, 3.0, (0, 1)),
]


def get_baselines(dag_name: str | None = None) -> list[BenchmarkBaseline]:
    """Return regression baselines, optionally filtered by DAG name."""
    if dag_name is None:
        return list(_BASELINES)
    return [b for b in _BASELINES if b.dag_name == dag_name]


def validate_against_baseline(
    dag_name: str,
    solver: str,
    radius_lower: int,
    radius_upper: int,
    n_fragile: int,
    runtime_s: float,
) -> list[str]:
    """Check results against baselines.  Returns list of violation messages."""
    violations: list[str] = []
    baselines = [b for b in _BASELINES if b.dag_name == dag_name and b.solver == solver]

    for bl in baselines:
        if radius_lower < bl.radius_lower:
            violations.append(
                f"radius_lower {radius_lower} < baseline {bl.radius_lower}"
            )
        if radius_upper > bl.radius_upper:
            violations.append(
                f"radius_upper {radius_upper} > baseline {bl.radius_upper}"
            )
        if runtime_s > bl.runtime_s_upper_bound:
            violations.append(
                f"runtime {runtime_s:.1f}s > bound {bl.runtime_s_upper_bound}s"
            )

    return violations
