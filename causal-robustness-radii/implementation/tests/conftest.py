"""
Shared fixtures, helper factories, and configuration presets for the
CausalCert test suite.

Every fixture uses deterministic seeds so that the full suite is reproducible.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from causalcert.types import (
    AdjacencyMatrix,
    AuditReport,
    CITestMethod,
    CITestResult,
    EditType,
    EstimationResult,
    FragilityChannel,
    FragilityScore,
    NodeId,
    NodeSet,
    PipelineConfig,
    RobustnessRadius,
    SolverStrategy,
    StructuralEdit,
)

# ---------------------------------------------------------------------------
# Canonical small DAGs
# ---------------------------------------------------------------------------


def _adj(n: int, edges: list[tuple[int, int]]) -> AdjacencyMatrix:
    """Build an ``int8`` adjacency matrix from an edge list."""
    a = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        a[u, v] = 1
    return a


# --- Chain: 0 -> 1 -> 2 -> 3 ---
@pytest.fixture
def chain4_adj() -> AdjacencyMatrix:
    return _adj(4, [(0, 1), (1, 2), (2, 3)])


@pytest.fixture
def chain4_names() -> list[str]:
    return ["X0", "X1", "X2", "X3"]


# --- Fork: 1 <- 0 -> 2 ---
@pytest.fixture
def fork3_adj() -> AdjacencyMatrix:
    return _adj(3, [(0, 1), (0, 2)])


# --- Collider: 0 -> 2 <- 1 ---
@pytest.fixture
def collider3_adj() -> AdjacencyMatrix:
    return _adj(3, [(0, 2), (1, 2)])


# --- Diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3 ---
@pytest.fixture
def diamond4_adj() -> AdjacencyMatrix:
    return _adj(4, [(0, 1), (0, 2), (1, 3), (2, 3)])


# --- M-bias / bow-tie: U1->X, U1->M, U2->M, U2->Y, X->Y ---
#     0=U1, 1=X, 2=M, 3=U2, 4=Y
@pytest.fixture
def mbias5_adj() -> AdjacencyMatrix:
    return _adj(5, [(0, 1), (0, 2), (3, 2), (3, 4), (1, 4)])


# --- Single node (no edges) ---
@pytest.fixture
def single_node_adj() -> AdjacencyMatrix:
    return _adj(1, [])


# --- Empty graph on 4 nodes ---
@pytest.fixture
def empty4_adj() -> AdjacencyMatrix:
    return _adj(4, [])


# --- Disconnected: two chains 0->1 and 2->3 ---
@pytest.fixture
def disconnected4_adj() -> AdjacencyMatrix:
    return _adj(4, [(0, 1), (2, 3)])


# --- Complete DAG on 4 nodes (topological: 0<1<2<3) ---
@pytest.fixture
def complete4_adj() -> AdjacencyMatrix:
    edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    return _adj(4, edges)


# --- Longer chain: 0 -> 1 -> ... -> 7 ---
@pytest.fixture
def chain8_adj() -> AdjacencyMatrix:
    return _adj(8, [(i, i + 1) for i in range(7)])


# --- Instrument DAG: Z->X->Y, no Z->Y ---
@pytest.fixture
def instrument3_adj() -> AdjacencyMatrix:
    return _adj(3, [(0, 1), (1, 2)])


# --- Confounded DAG: X<-C->Y, X->Y ---
@pytest.fixture
def confounded3_adj() -> AdjacencyMatrix:
    """DAG 0=C, 1=X, 2=Y: C->X, C->Y, X->Y."""
    return _adj(3, [(0, 1), (0, 2), (1, 2)])


# --- Mediator DAG: X -> M -> Y ---
@pytest.fixture
def mediator3_adj() -> AdjacencyMatrix:
    return _adj(3, [(0, 1), (1, 2)])


# --- Butterfly / 5-node complex ---
@pytest.fixture
def butterfly5_adj() -> AdjacencyMatrix:
    return _adj(5, [(0, 2), (1, 2), (2, 3), (2, 4)])


# --- Tree on 7 nodes (rooted at 0) ---
@pytest.fixture
def tree7_adj() -> AdjacencyMatrix:
    return _adj(7, [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])


# ---------------------------------------------------------------------------
# Synthetic datasets
# ---------------------------------------------------------------------------


def _linear_gaussian_data(
    adj: AdjacencyMatrix,
    n: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate data from a linear-Gaussian SCM with unit-normal noise."""
    rng = np.random.default_rng(seed)
    p = adj.shape[0]
    topo = _topo_sort(adj)
    data = np.zeros((n, p))
    weights = rng.uniform(0.5, 1.5, size=(p, p)) * adj
    for v in topo:
        pa = np.where(adj[:, v] == 1)[0]
        mean = data[:, pa] @ weights[pa, v] if len(pa) else 0.0
        data[:, v] = mean + rng.standard_normal(n)
    cols = [f"X{i}" for i in range(p)]
    return pd.DataFrame(data, columns=cols)


def _topo_sort(adj: np.ndarray) -> list[int]:
    p = adj.shape[0]
    in_deg = adj.sum(axis=0).copy()
    queue = [v for v in range(p) if in_deg[v] == 0]
    order: list[int] = []
    while queue:
        v = queue.pop(0)
        order.append(v)
        for w in range(p):
            if adj[v, w]:
                in_deg[w] -= 1
                if in_deg[w] == 0:
                    queue.append(w)
    return order


@pytest.fixture
def chain4_data(chain4_adj: AdjacencyMatrix) -> pd.DataFrame:
    return _linear_gaussian_data(chain4_adj, n=500, seed=42)


@pytest.fixture
def diamond4_data(diamond4_adj: AdjacencyMatrix) -> pd.DataFrame:
    return _linear_gaussian_data(diamond4_adj, n=500, seed=42)


@pytest.fixture
def confounded3_data(confounded3_adj: AdjacencyMatrix) -> pd.DataFrame:
    return _linear_gaussian_data(confounded3_adj, n=500, seed=42)


@pytest.fixture
def fork3_data(fork3_adj: AdjacencyMatrix) -> pd.DataFrame:
    return _linear_gaussian_data(fork3_adj, n=500, seed=42)


@pytest.fixture
def collider3_data(collider3_adj: AdjacencyMatrix) -> pd.DataFrame:
    return _linear_gaussian_data(collider3_adj, n=500, seed=42)


@pytest.fixture
def mbias5_data(mbias5_adj: AdjacencyMatrix) -> pd.DataFrame:
    return _linear_gaussian_data(mbias5_adj, n=1000, seed=42)


@pytest.fixture
def complete4_data(complete4_adj: AdjacencyMatrix) -> pd.DataFrame:
    return _linear_gaussian_data(complete4_adj, n=500, seed=42)


@pytest.fixture
def butterfly5_data(butterfly5_adj: AdjacencyMatrix) -> pd.DataFrame:
    return _linear_gaussian_data(butterfly5_adj, n=500, seed=42)


# Gaussian independent data (no relationships)
@pytest.fixture
def independent_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame(
        {"X0": rng.standard_normal(n), "X1": rng.standard_normal(n), "X2": rng.standard_normal(n)}
    )


# Data with binary treatment
@pytest.fixture
def binary_treatment_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 1000
    C = rng.standard_normal(n)
    T = (rng.standard_normal(n) + 0.5 * C > 0).astype(float)
    Y = 2.0 * T + 0.8 * C + rng.standard_normal(n) * 0.5
    return pd.DataFrame({"C": C, "T": T, "Y": Y})


# ---------------------------------------------------------------------------
# Config presets
# ---------------------------------------------------------------------------


@pytest.fixture
def quick_config() -> PipelineConfig:
    return PipelineConfig(
        treatment=0,
        outcome=1,
        alpha=0.05,
        ci_method=CITestMethod.PARTIAL_CORRELATION,
        solver_strategy=SolverStrategy.AUTO,
        max_k=3,
        n_folds=2,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Structural edit helpers
# ---------------------------------------------------------------------------


def make_add(u: int, v: int) -> StructuralEdit:
    return StructuralEdit(EditType.ADD, u, v)


def make_del(u: int, v: int) -> StructuralEdit:
    return StructuralEdit(EditType.DELETE, u, v)


def make_rev(u: int, v: int) -> StructuralEdit:
    return StructuralEdit(EditType.REVERSE, u, v)


# ---------------------------------------------------------------------------
# Audit report factory
# ---------------------------------------------------------------------------


def make_audit_report(
    treatment: int = 0,
    outcome: int = 1,
    n_nodes: int = 4,
    n_edges: int = 3,
    radius_lb: int = 2,
    radius_ub: int = 2,
) -> AuditReport:
    """Build a minimal valid AuditReport for testing reporters."""
    radius = RobustnessRadius(
        lower_bound=radius_lb,
        upper_bound=radius_ub,
        witness_edits=(make_del(0, 1),),
        solver_strategy=SolverStrategy.AUTO,
        solver_time_s=0.5,
        gap=0.0,
        certified=radius_lb == radius_ub,
    )
    frag = [
        FragilityScore(
            edge=(0, 1),
            total_score=0.9,
            channel_scores={
                FragilityChannel.D_SEPARATION: 0.8,
                FragilityChannel.IDENTIFICATION: 0.7,
                FragilityChannel.ESTIMATION: 0.6,
            },
        ),
        FragilityScore(
            edge=(1, 2),
            total_score=0.3,
            channel_scores={
                FragilityChannel.D_SEPARATION: 0.2,
                FragilityChannel.IDENTIFICATION: 0.1,
                FragilityChannel.ESTIMATION: 0.05,
            },
        ),
    ]
    baseline = EstimationResult(
        ate=1.5,
        se=0.3,
        ci_lower=0.9,
        ci_upper=2.1,
        adjustment_set=frozenset(),
        method="aipw",
        n_obs=500,
    )
    return AuditReport(
        treatment=treatment,
        outcome=outcome,
        n_nodes=n_nodes,
        n_edges=n_edges,
        radius=radius,
        fragility_ranking=frag,
        baseline_estimate=baseline,
        perturbed_estimates=[],
        ci_results=[],
        metadata={"seed": 42},
    )


@pytest.fixture
def sample_audit_report() -> AuditReport:
    return make_audit_report()


# ---------------------------------------------------------------------------
# CI test result helpers
# ---------------------------------------------------------------------------


def make_ci_result(
    x: int,
    y: int,
    cond: frozenset[int],
    p: float,
    reject: bool | None = None,
    alpha: float = 0.05,
    method: CITestMethod = CITestMethod.PARTIAL_CORRELATION,
) -> CITestResult:
    if reject is None:
        reject = p < alpha
    return CITestResult(
        x=x,
        y=y,
        conditioning_set=cond,
        statistic=0.0,
        p_value=p,
        method=method,
        reject=reject,
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Temporary file helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def sample_csv(tmp_path: Path, chain4_data: pd.DataFrame) -> Path:
    p = tmp_path / "test_data.csv"
    chain4_data.to_csv(p, index=False)
    return p


@pytest.fixture
def sample_parquet(tmp_path: Path, chain4_data: pd.DataFrame) -> Path:
    p = tmp_path / "test_data.parquet"
    chain4_data.to_parquet(p, index=False)
    return p


@pytest.fixture
def sample_dot() -> str:
    return "digraph {\n  X0 -> X1;\n  X1 -> X2;\n  X2 -> X3;\n}"


@pytest.fixture
def sample_json_dag() -> str:
    return json.dumps(
        {
            "n_nodes": 4,
            "node_names": ["X0", "X1", "X2", "X3"],
            "edges": [[0, 1], [1, 2], [2, 3]],
        }
    )


# ---------------------------------------------------------------------------
# Helper utilities exposed to all test modules
# ---------------------------------------------------------------------------


def assert_is_dag(adj: np.ndarray) -> None:
    """Assert that *adj* has no cycles (via topological sort)."""
    p = adj.shape[0]
    in_deg = adj.sum(axis=0).astype(int).copy()
    queue = [v for v in range(p) if in_deg[v] == 0]
    count = 0
    while queue:
        v = queue.pop(0)
        count += 1
        for w in range(p):
            if adj[v, w]:
                in_deg[w] -= 1
                if in_deg[w] == 0:
                    queue.append(w)
    assert count == p, f"Graph contains a cycle (sorted {count}/{p} nodes)"


def random_dag(n: int, edge_prob: float = 0.3, seed: int = 42) -> AdjacencyMatrix:
    """Return a random DAG on *n* nodes by lower-triangular sampling."""
    rng = np.random.default_rng(seed)
    a = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < edge_prob:
                a[i, j] = 1
    return a
