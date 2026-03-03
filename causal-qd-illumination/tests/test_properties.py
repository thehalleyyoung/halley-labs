"""Hypothesis property-based tests for CausalQD operators and DAG invariants."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings, strategies as st

from causal_qd.core.dag import DAG
from causal_qd.operators.mutation import (
    EdgeAddMutation,
    EdgeRemoveMutation,
    EdgeReverseMutation,
    TopologicalMutation,
    _has_cycle,
    _topological_sort,
)
from causal_qd.operators.crossover import (
    MarkovBlanketCrossover,
    OrderBasedCrossover,
    OrderCrossover,
    SkeletonCrossover,
    SubgraphCrossover,
    UniformCrossover,
)


# ===================================================================
# Hypothesis strategy for random DAGs
# ===================================================================

@st.composite
def random_dag(draw, min_nodes=2, max_nodes=12):
    """Generate a random DAG adjacency matrix via random permutation."""
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    density = draw(st.floats(min_value=0.1, max_value=0.5))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    perm = rng.permutation(n)
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                adj[perm[i], perm[j]] = 1
    return adj


# ===================================================================
# Mutation property tests
# ===================================================================

MUTATION_OPERATORS = [
    TopologicalMutation(),
    EdgeAddMutation(),
    EdgeRemoveMutation(),
    EdgeReverseMutation(),
]


@given(adj=random_dag(), op_idx=st.integers(0, len(MUTATION_OPERATORS) - 1),
       seed=st.integers(0, 2**32 - 1))
@settings(max_examples=50, deadline=None)
def test_mutation_preserves_acyclicity(adj, op_idx, seed):
    """Every mutation operator preserves DAG acyclicity."""
    op = MUTATION_OPERATORS[op_idx]
    rng = np.random.default_rng(seed)
    result = op.mutate(adj, rng)
    assert not _has_cycle(result), (
        f"{op.__class__.__name__} introduced a cycle"
    )


# ===================================================================
# Crossover property tests
# ===================================================================

CROSSOVER_OPERATORS = [
    OrderCrossover(),
    OrderBasedCrossover(),
    UniformCrossover(),
    SkeletonCrossover(),
    SubgraphCrossover(),
    MarkovBlanketCrossover(),
]


@given(adj=random_dag(), op_idx=st.integers(0, len(CROSSOVER_OPERATORS) - 1),
       seed=st.integers(0, 2**32 - 1))
@settings(max_examples=50, deadline=None)
def test_crossover_preserves_acyclicity(adj, op_idx, seed):
    """Every crossover operator preserves DAG acyclicity."""
    op = CROSSOVER_OPERATORS[op_idx]
    rng = np.random.default_rng(seed)
    # Create a second parent by mutating the first
    perm = rng.permutation(adj.shape[0])
    parent2 = adj[np.ix_(perm, perm)].copy()
    # Ensure parent2 is acyclic (permutation of a DAG is still a DAG only
    # if we relabel consistently, but the permuted matrix may have cycles).
    if _has_cycle(parent2):
        # Fall back to a fresh random DAG of same size
        n = adj.shape[0]
        parent2 = np.zeros((n, n), dtype=np.int8)
        p = rng.permutation(n)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.3:
                    parent2[p[i], p[j]] = 1
    child1, child2 = op.crossover(adj, parent2, rng)
    assert not _has_cycle(child1), (
        f"{op.__class__.__name__} child1 has a cycle"
    )
    assert not _has_cycle(child2), (
        f"{op.__class__.__name__} child2 has a cycle"
    )


# ===================================================================
# DAG ancestor / descendant symmetry
# ===================================================================

@given(adj=random_dag())
@settings(max_examples=50, deadline=None)
def test_dag_ancestors_descendants_symmetry(adj):
    """y ∈ descendants(x) iff x ∈ ancestors(y)."""
    dag = DAG(adj)
    n = adj.shape[0]
    for x in range(n):
        for y in dag.descendants(x):
            assert x in dag.ancestors(y), (
                f"node {y} is a descendant of {x} but {x} not in ancestors({y})"
            )
    for y in range(n):
        for x in dag.ancestors(y):
            assert y in dag.descendants(x), (
                f"node {x} is an ancestor of {y} but {y} not in descendants({x})"
            )


# ===================================================================
# Topological order respects edges
# ===================================================================

@given(adj=random_dag())
@settings(max_examples=50, deadline=None)
def test_dag_topological_order_respects_edges(adj):
    """For all edges i→j, i appears before j in topological order."""
    dag = DAG(adj)
    order = dag.topological_order
    pos = {node: idx for idx, node in enumerate(order)}
    n = adj.shape[0]
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                assert pos[i] < pos[j], (
                    f"Edge {i}→{j} but topo pos {pos[i]} >= {pos[j]}"
                )


# ===================================================================
# d-separation symmetry
# ===================================================================

@given(adj=random_dag(min_nodes=3, max_nodes=8),
       seed=st.integers(0, 2**32 - 1))
@settings(max_examples=50, deadline=None)
def test_dseparation_symmetry(adj, seed):
    """d_sep(X, Y | Z) iff d_sep(Y, X | Z)."""
    dag = DAG(adj)
    n = adj.shape[0]
    rng = np.random.default_rng(seed)
    nodes = list(range(n))
    rng.shuffle(nodes)
    # Pick disjoint non-empty X, Y, and possibly-empty Z
    split1 = max(1, n // 3)
    split2 = max(split1 + 1, 2 * n // 3)
    x = set(nodes[:split1])
    y = set(nodes[split1:split2])
    z = set(nodes[split2:])
    if not x or not y:
        return  # degenerate partition, skip
    assert dag.d_separated(x, y, z) == dag.d_separated(y, x, z), (
        f"d-separation not symmetric: X={x}, Y={y}, Z={z}"
    )
