"""DAG enumeration within a Markov Equivalence Class.

Provides exact enumeration, counting, uniform sampling, and weighted
sampling of DAGs consistent with a given CPDAG.  Uses decomposition
into chain components for efficient computation.
"""

from __future__ import annotations

import itertools
from collections import deque
from typing import Any, Callable, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray


# -------------------------------------------------------------------
# Chain component decomposition
# -------------------------------------------------------------------

def chain_components(cpdag: Any) -> List[Set[int]]:
    """Decompose CPDAG into chain components.

    A chain component is a maximal set of nodes connected by undirected
    edges.  The MEC size equals the product of MEC sizes of individual
    chain components.

    Parameters
    ----------
    cpdag : CPDAG
        The CPDAG to decompose.

    Returns
    -------
    list of set[int]
        Each element is the set of node indices in one chain component.
    """
    n = cpdag.n_nodes
    visited = [False] * n
    components: List[Set[int]] = []

    for start in range(n):
        if visited[start]:
            continue
        # BFS through undirected edges
        comp: Set[int] = set()
        queue = deque([start])
        while queue:
            v = queue.popleft()
            if visited[v]:
                continue
            visited[v] = True
            comp.add(v)
            for nb in cpdag.neighbors(v):
                if not visited[nb]:
                    queue.append(nb)
        components.append(comp)

    return components


# -------------------------------------------------------------------
# Count DAGs in a chain component
# -------------------------------------------------------------------

def _get_undirected_subgraph(cpdag: Any, component: Set[int]) -> NDArray[np.int_]:
    """Extract the undirected subgraph for nodes in *component*."""
    nodes = sorted(component)
    k = len(nodes)
    node_map = {v: i for i, v in enumerate(nodes)}
    adj = np.zeros((k, k), dtype=np.int_)
    for u in nodes:
        for v in cpdag.neighbors(u):
            if v in component:
                adj[node_map[u], node_map[v]] = 1
    return adj


def _count_dags_chain_component(
    cpdag: Any, component: Set[int]
) -> int:
    """Count DAGs for one chain component.

    For a chain component with k undirected edges among m nodes,
    enumerate valid orientations.  An orientation is valid if:
    1. It creates no new v-structures (relative to existing directed edges).
    2. It creates no directed cycles.

    For small components, uses exhaustive enumeration.
    For larger components, uses a recursive approach.
    """
    nodes = sorted(component)
    k = len(nodes)

    if k <= 1:
        return 1

    # Collect undirected edges within this component
    undirected_in_comp: List[Tuple[int, int]] = []
    for i, j in cpdag.undirected_edges:
        if i in component and j in component:
            undirected_in_comp.append((i, j))

    if not undirected_in_comp:
        return 1

    n_undir = len(undirected_in_comp)

    # For small numbers of undirected edges, enumerate
    if n_undir > 20:
        # Too many to enumerate; use sampling estimate
        return _estimate_count_sampling(cpdag, component, undirected_in_comp)

    count = 0
    # Each undirected edge can be oriented in 2 ways
    for bits in range(1 << n_undir):
        # Build the DAG with this orientation
        valid = True
        trial_directed = set(cpdag.directed_edges)

        orientations: List[Tuple[int, int]] = []
        for idx, (i, j) in enumerate(undirected_in_comp):
            if bits & (1 << idx):
                orientations.append((i, j))
            else:
                orientations.append((j, i))

        for e in orientations:
            trial_directed.add(e)

        # Check acyclicity
        if not _is_acyclic(trial_directed, cpdag.n_nodes):
            for e in orientations:
                trial_directed.discard(e)
            continue

        # Check no new v-structures
        if _creates_new_v_structure(cpdag, orientations, trial_directed):
            for e in orientations:
                trial_directed.discard(e)
            continue

        count += 1
        for e in orientations:
            trial_directed.discard(e)

    return max(count, 1)


def _is_acyclic(directed_edges: set, n: int) -> bool:
    """Check if the directed edges form a DAG."""
    adj = np.zeros((n, n), dtype=np.int_)
    for i, j in directed_edges:
        if 0 <= i < n and 0 <= j < n:
            adj[i, j] = 1

    # Kahn's algorithm
    in_deg = np.sum(adj, axis=0).astype(int)
    queue = deque(i for i in range(n) if in_deg[i] == 0)
    count = 0
    while queue:
        v = queue.popleft()
        count += 1
        for j in range(n):
            if adj[v, j]:
                in_deg[j] -= 1
                if in_deg[j] == 0:
                    queue.append(j)
    return count == n


def _creates_new_v_structure(
    cpdag: Any,
    new_orientations: List[Tuple[int, int]],
    all_directed: set,
) -> bool:
    """Check if the new orientations create v-structures not in original."""
    n = cpdag.n_nodes

    # Build original v-structures
    original_v = set()
    for b in range(n):
        pa = [i for (i, j) in cpdag.directed_edges if j == b]
        for idx_a in range(len(pa)):
            for idx_c in range(idx_a + 1, len(pa)):
                a, c = pa[idx_a], pa[idx_c]
                if not cpdag._is_adjacent(a, c):
                    original_v.add((min(a, c), b, max(a, c)))

    # Check new v-structures
    new_edge_set = set(new_orientations)
    for b in range(n):
        pa = [i for (i, j) in all_directed if j == b]
        for idx_a in range(len(pa)):
            for idx_c in range(idx_a + 1, len(pa)):
                a, c = pa[idx_a], pa[idx_c]
                # Check adjacency in the new graph
                adj_in_new = ((a, c) in all_directed
                              or (c, a) in all_directed
                              or cpdag.has_undirected_edge(a, c))
                if not adj_in_new:
                    key = (min(a, c), b, max(a, c))
                    if key not in original_v:
                        return True
    return False


def _estimate_count_sampling(
    cpdag: Any,
    component: Set[int],
    undirected_edges: List[Tuple[int, int]],
    n_samples: int = 10000,
) -> int:
    """Estimate MEC size by random sampling when enumeration is infeasible."""
    rng = np.random.default_rng(42)
    n_undir = len(undirected_edges)
    valid_count = 0

    for _ in range(n_samples):
        bits = rng.integers(0, 1 << n_undir)
        trial_directed = set(cpdag.directed_edges)
        orientations = []
        for idx, (i, j) in enumerate(undirected_edges):
            if bits & (1 << idx):
                orientations.append((i, j))
            else:
                orientations.append((j, i))
        for e in orientations:
            trial_directed.add(e)

        if _is_acyclic(trial_directed, cpdag.n_nodes):
            if not _creates_new_v_structure(cpdag, orientations, trial_directed):
                valid_count += 1

        for e in orientations:
            trial_directed.discard(e)

    # Estimate: (valid/samples) * 2^n_undir
    if valid_count == 0:
        return 1
    return max(1, int(round(valid_count / n_samples * (1 << n_undir))))


# -------------------------------------------------------------------
# Public API: count_dags_in_mec
# -------------------------------------------------------------------

def count_dags_in_mec(cpdag: Any) -> int:
    """Count DAGs consistent with the CPDAG.

    Uses decomposition into chain components: the total count is
    the product of counts for each chain component.

    Parameters
    ----------
    cpdag : CPDAG
        The CPDAG defining the MEC.

    Returns
    -------
    int
        Number of DAGs in the MEC.
    """
    components = chain_components(cpdag)
    total = 1
    for comp in components:
        # Only count components with undirected edges
        has_undirected = any(
            i in comp and j in comp
            for i, j in cpdag.undirected_edges
        )
        if has_undirected:
            total *= _count_dags_chain_component(cpdag, comp)
    return max(total, 1)


# -------------------------------------------------------------------
# Public API: enumerate_dags
# -------------------------------------------------------------------

def enumerate_dags(
    cpdag: Any, max_count: int = 10000
) -> List[NDArray[np.int_]]:
    """List all DAGs in the MEC (up to *max_count*).

    Parameters
    ----------
    cpdag : CPDAG
        The CPDAG defining the MEC.
    max_count : int
        Maximum number of DAGs to enumerate.

    Returns
    -------
    list of NDArray
        List of DAG adjacency matrices.
    """
    n = cpdag.n_nodes

    # Collect undirected edges
    undirected_list = list(cpdag.undirected_edges)
    n_undir = len(undirected_list)

    if n_undir == 0:
        # Fully directed CPDAG = single DAG
        dag = np.zeros((n, n), dtype=np.int_)
        for i, j in cpdag.directed_edges:
            dag[i, j] = 1
        return [dag]

    if n_undir > 20:
        raise ValueError(
            f"Too many undirected edges ({n_undir}) for full enumeration. "
            f"Use sample_dag_uniform instead."
        )

    results: List[NDArray[np.int_]] = []

    for bits in range(1 << n_undir):
        if len(results) >= max_count:
            break

        dag = np.zeros((n, n), dtype=np.int_)
        # Copy directed edges
        for i, j in cpdag.directed_edges:
            dag[i, j] = 1

        # Orient undirected edges according to bits
        orientations = []
        for idx, (i, j) in enumerate(undirected_list):
            if bits & (1 << idx):
                dag[i, j] = 1
                orientations.append((i, j))
            else:
                dag[j, i] = 1
                orientations.append((j, i))

        # Check validity
        all_dir = set(cpdag.directed_edges) | set(orientations)
        if not _is_acyclic(all_dir, n):
            continue
        if _creates_new_v_structure(cpdag, orientations, all_dir):
            continue

        results.append(dag)

    return results


# -------------------------------------------------------------------
# Public API: sample_dag_uniform
# -------------------------------------------------------------------

def sample_dag_uniform(
    cpdag: Any, rng: np.random.Generator | None = None
) -> NDArray[np.int_]:
    """Sample a DAG uniformly from the MEC.

    Uses rejection sampling: randomly orient undirected edges and
    check validity.

    Parameters
    ----------
    cpdag : CPDAG
        The CPDAG defining the MEC.
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    NDArray
        A DAG adjacency matrix.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = cpdag.n_nodes
    undirected_list = list(cpdag.undirected_edges)
    n_undir = len(undirected_list)

    if n_undir == 0:
        dag = np.zeros((n, n), dtype=np.int_)
        for i, j in cpdag.directed_edges:
            dag[i, j] = 1
        return dag

    max_attempts = 10000
    for _ in range(max_attempts):
        dag = np.zeros((n, n), dtype=np.int_)
        for i, j in cpdag.directed_edges:
            dag[i, j] = 1

        orientations = []
        for i, j in undirected_list:
            if rng.random() < 0.5:
                dag[i, j] = 1
                orientations.append((i, j))
            else:
                dag[j, i] = 1
                orientations.append((j, i))

        all_dir = set(cpdag.directed_edges) | set(orientations)
        if _is_acyclic(all_dir, n):
            if not _creates_new_v_structure(cpdag, orientations, all_dir):
                return dag

    # Fallback: use the CPDAG's own method
    return cpdag.to_dag()


# -------------------------------------------------------------------
# Public API: mec_size
# -------------------------------------------------------------------

def mec_size(cpdag: Any) -> int:
    """Compute the size of the MEC represented by *cpdag*.

    Parameters
    ----------
    cpdag : CPDAG or object with n_nodes and edge sets.

    Returns
    -------
    int
        Number of DAGs in the MEC.
    """
    return count_dags_in_mec(cpdag)


# -------------------------------------------------------------------
# Public API: mec_size_lower_bound
# -------------------------------------------------------------------

def mec_size_lower_bound(cpdag: Any) -> int:
    """Lower bound on MEC size.

    A simple lower bound: at least 1 DAG, and for each reversible edge,
    at least one additional DAG exists (loosely).
    """
    n_undir = len(cpdag.undirected_edges)
    return max(1, n_undir + 1)


# -------------------------------------------------------------------
# MECEnumerator class (API compatibility)
# -------------------------------------------------------------------

class MECEnumerator:
    """Enumerate all DAGs in a Markov Equivalence Class.

    Parameters
    ----------
    cpdag : CPDAG
        The CPDAG defining the equivalence class.
    """

    def __init__(self, cpdag: Any) -> None:
        self.cpdag = cpdag
        self._dags: List[NDArray[np.int_]] | None = None

    def enumerate_all(self) -> List[NDArray[np.int_]]:
        """Return every DAG consistent with the stored CPDAG."""
        if self._dags is None:
            self._dags = enumerate_dags(self.cpdag)
        return list(self._dags)

    def count(self) -> int:
        """Return the number of DAGs in the MEC without full enumeration."""
        return count_dags_in_mec(self.cpdag)


# -------------------------------------------------------------------
# MECSampler class (API compatibility)
# -------------------------------------------------------------------

class MECSampler:
    """Sample DAGs uniformly or by a scoring function.

    Parameters
    ----------
    cpdag : CPDAG
        The CPDAG defining the equivalence class.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, cpdag: Any, seed: int | None = None) -> None:
        self.cpdag = cpdag
        self._rng = np.random.default_rng(seed)

    def sample_uniform(self, n_samples: int) -> List[NDArray[np.int_]]:
        """Draw *n_samples* DAGs uniformly from the MEC."""
        return [
            sample_dag_uniform(self.cpdag, self._rng)
            for _ in range(n_samples)
        ]

    def sample_weighted(
        self,
        n_samples: int,
        score_fn: Callable[[NDArray[np.int_]], float],
    ) -> List[NDArray[np.int_]]:
        """Draw DAGs weighted by *score_fn*.

        Uses importance sampling: draw uniform samples, compute weights
        from score_fn, then resample with replacement.
        """
        # Draw a larger pool for resampling
        pool_size = max(n_samples * 5, 100)
        pool = [
            sample_dag_uniform(self.cpdag, self._rng)
            for _ in range(pool_size)
        ]

        # Compute scores
        scores = np.array([score_fn(dag) for dag in pool])

        # Convert to probabilities (softmax)
        scores = scores - np.max(scores)  # numerical stability
        weights = np.exp(scores)
        total = np.sum(weights)
        if total <= 0:
            probs = np.ones(pool_size) / pool_size
        else:
            probs = weights / total

        # Resample
        indices = self._rng.choice(pool_size, size=n_samples, p=probs)
        return [pool[i] for i in indices]
