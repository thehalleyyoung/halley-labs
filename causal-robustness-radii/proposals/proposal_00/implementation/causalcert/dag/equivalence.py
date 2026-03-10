"""Advanced equivalence-class operations for causal DAGs.

Provides Markov equivalence class (MEC) operations, interventional equivalence,
I-MEC conversion, MEC enumeration/sampling, and distinguishing-experiment
characterisation.
"""

from __future__ import annotations

import itertools
import random
from collections import defaultdict, deque
from typing import (
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)

import numpy as np

from causalcert.dag.graph import CausalDAG

NodeId = int


# ===================================================================
# 1.  Skeleton and v-structure extraction
# ===================================================================

def skeleton(dag: CausalDAG) -> np.ndarray:
    """Return the undirected skeleton as a symmetric 0/1 adjacency matrix."""
    a = dag.adj.astype(np.int8)
    return np.clip(a + a.T, 0, 1)


def v_structures(dag: CausalDAG) -> List[Tuple[NodeId, NodeId, NodeId]]:
    """Return all v-structures (i→k←j) where i and j are non-adjacent.

    Returns
    -------
    list of (i, k, j) with i < j for canonical ordering.
    """
    adj = dag.adj
    n = dag.n_nodes
    skel = skeleton(dag)
    vs: List[Tuple[NodeId, NodeId, NodeId]] = []
    for k in range(n):
        parents_k = [p for p in range(n) if adj[p, k]]
        for idx_a, i in enumerate(parents_k):
            for j in parents_k[idx_a + 1 :]:
                if not skel[i, j]:
                    a, b = (i, j) if i < j else (j, i)
                    vs.append((a, k, b))
    return vs


# ===================================================================
# 2.  Markov equivalence checking
# ===================================================================

def same_skeleton(dag1: CausalDAG, dag2: CausalDAG) -> bool:
    """Check if two DAGs have the same skeleton."""
    return bool(np.array_equal(skeleton(dag1), skeleton(dag2)))


def same_v_structures(dag1: CausalDAG, dag2: CausalDAG) -> bool:
    """Check if two DAGs have the same set of v-structures."""
    return set(v_structures(dag1)) == set(v_structures(dag2))


def is_markov_equivalent(dag1: CausalDAG, dag2: CausalDAG) -> bool:
    """Two DAGs are Markov equivalent iff they share skeleton and v-structures.

    Vermeij & Chickering (2002).
    """
    return same_skeleton(dag1, dag2) and same_v_structures(dag1, dag2)


# ===================================================================
# 3.  CPDAG construction (Chickering's algorithm)
# ===================================================================

def _order_edges(dag: CausalDAG) -> List[Tuple[NodeId, NodeId]]:
    """Return edges in a topological total order consistent ordering."""
    order = dag.topological_sort()
    rank = {node: idx for idx, node in enumerate(order)}
    edges = dag.edge_list()
    return sorted(edges, key=lambda e: (rank[e[1]], rank[e[0]]))


def _label_edges(dag: CausalDAG) -> Dict[Tuple[NodeId, NodeId], str]:
    """Label each edge as 'compelled' or 'reversible' (Chickering 1995)."""
    adj = dag.adj
    n = dag.n_nodes
    ordered = _order_edges(dag)
    label: Dict[Tuple[NodeId, NodeId], str] = {}

    for x, y in ordered:
        parents_y = {p for p in range(n) if adj[p, y] and p != x}
        compelled_by_parent = False
        for w in parents_y:
            if (w, y) in label and label[(w, y)] == "compelled":
                if not adj[w, x] and not adj[x, w]:
                    label[(x, y)] = "compelled"
                    compelled_by_parent = True
                    for z, y2 in ordered:
                        if y2 == y and z != x and (z, y) not in label:
                            label[(z, y)] = "compelled"
                    break
        if compelled_by_parent:
            continue

        exists_w = False
        for w in parents_y:
            if not adj[w, x] and not adj[x, w]:
                exists_w = True
                break
        if exists_w:
            label[(x, y)] = "compelled"
            for z in parents_y:
                if (z, y) not in label:
                    label[(z, y)] = "compelled"
        else:
            label[(x, y)] = "reversible"
    return label


def compelled_edges(dag: CausalDAG) -> List[Tuple[NodeId, NodeId]]:
    """Edges that have the same orientation in every MEC member."""
    labels = _label_edges(dag)
    return [e for e, lbl in labels.items() if lbl == "compelled"]


def reversible_edges(dag: CausalDAG) -> List[Tuple[NodeId, NodeId]]:
    """Edges that may be reversed within the MEC."""
    labels = _label_edges(dag)
    return [e for e, lbl in labels.items() if lbl == "reversible"]


def to_cpdag(dag: CausalDAG) -> np.ndarray:
    """Convert DAG to its CPDAG (completed partially directed acyclic graph).

    Compelled edges keep direction; reversible edges become undirected
    (represented by 1 in both directions).

    Returns
    -------
    np.ndarray  shape (n, n), int8
    """
    n = dag.n_nodes
    cpdag = np.zeros((n, n), dtype=np.int8)
    labels = _label_edges(dag)
    for (u, v), lbl in labels.items():
        cpdag[u, v] = 1
        if lbl == "reversible":
            cpdag[v, u] = 1
    return cpdag


# ===================================================================
# 4.  Meek rules for CPDAG orientation propagation
# ===================================================================

def _apply_meek_r1(cpdag: np.ndarray) -> bool:
    """R1: If a→b — c and a ⊥ c, orient b→c."""
    n = cpdag.shape[0]
    changed = False
    for b in range(n):
        for a in range(n):
            if a == b:
                continue
            if cpdag[a, b] == 1 and cpdag[b, a] == 0:
                for c in range(n):
                    if c == a or c == b:
                        continue
                    if cpdag[b, c] == 1 and cpdag[c, b] == 1:
                        if cpdag[a, c] == 0 and cpdag[c, a] == 0:
                            cpdag[c, b] = 0
                            changed = True
    return changed


def _apply_meek_r2(cpdag: np.ndarray) -> bool:
    """R2: If a→c→b and a — b, orient a→b."""
    n = cpdag.shape[0]
    changed = False
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            if cpdag[a, b] == 1 and cpdag[b, a] == 1:
                for c in range(n):
                    if c == a or c == b:
                        continue
                    if (cpdag[a, c] == 1 and cpdag[c, a] == 0 and
                            cpdag[c, b] == 1 and cpdag[b, c] == 0):
                        cpdag[b, a] = 0
                        changed = True
    return changed


def _apply_meek_r3(cpdag: np.ndarray) -> bool:
    """R3: If a — c → b and a — d → b and c ⊥ d and a — b, orient a→b."""
    n = cpdag.shape[0]
    changed = False
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            if not (cpdag[a, b] == 1 and cpdag[b, a] == 1):
                continue
            neighbors_a = [
                c for c in range(n) if c != a and c != b
                and cpdag[a, c] == 1 and cpdag[c, a] == 1
                and cpdag[c, b] == 1 and cpdag[b, c] == 0
            ]
            for idx_c, c in enumerate(neighbors_a):
                for d in neighbors_a[idx_c + 1:]:
                    if cpdag[c, d] == 0 and cpdag[d, c] == 0:
                        cpdag[b, a] = 0
                        changed = True
    return changed


def _apply_meek_r4(cpdag: np.ndarray) -> bool:
    """R4: If a — c → d → b and a — b, orient a→b."""
    n = cpdag.shape[0]
    changed = False
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            if not (cpdag[a, b] == 1 and cpdag[b, a] == 1):
                continue
            for c in range(n):
                if c == a or c == b:
                    continue
                if not (cpdag[a, c] == 1 and cpdag[c, a] == 1):
                    continue
                for d in range(n):
                    if d == a or d == b or d == c:
                        continue
                    if (cpdag[c, d] == 1 and cpdag[d, c] == 0 and
                            cpdag[d, b] == 1 and cpdag[b, d] == 0):
                        cpdag[b, a] = 0
                        changed = True
    return changed


def apply_meek_rules(cpdag: np.ndarray, *, max_iter: int = 100) -> np.ndarray:
    """Repeatedly apply Meek rules R1–R4 until convergence.

    Parameters
    ----------
    cpdag : np.ndarray  (n, n) int8
        Partially directed graph with directed (1,0) and undirected (1,1) edges.
    max_iter : int
        Safety bound on iterations.

    Returns
    -------
    np.ndarray  oriented CPDAG
    """
    out = cpdag.copy()
    for _ in range(max_iter):
        c1 = _apply_meek_r1(out)
        c2 = _apply_meek_r2(out)
        c3 = _apply_meek_r3(out)
        c4 = _apply_meek_r4(out)
        if not (c1 or c2 or c3 or c4):
            break
    return out


# ===================================================================
# 5.  CPDAG → DAG extension
# ===================================================================

def cpdag_to_dag(cpdag: np.ndarray) -> Optional[CausalDAG]:
    """Extend a CPDAG to a consistent DAG (if possible).

    Uses the algorithm of Dor & Tarsi (1992): repeatedly find a node
    with no undirected neighbors that have higher topological rank.
    """
    n = cpdag.shape[0]
    adj = cpdag.copy()
    order: List[NodeId] = []
    remaining = set(range(n))

    for _ in range(n):
        found = False
        for v in sorted(remaining):
            undirected_neighbors = [
                u for u in remaining if u != v and adj[u, v] == 1 and adj[v, u] == 1
            ]
            sinks_ok = True
            for u in undirected_neighbors:
                neighbors_of_v_directed = [
                    w for w in remaining if w != v and w != u
                    and adj[w, v] == 1 and adj[v, w] == 0
                ]
                for w in neighbors_of_v_directed:
                    if adj[w, u] == 0 and adj[u, w] == 0:
                        sinks_ok = False
                        break
                if not sinks_ok:
                    break
            if sinks_ok:
                for u in undirected_neighbors:
                    adj[v, u] = 0
                remaining.remove(v)
                order.append(v)
                found = True
                break
        if not found:
            return None

    result = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            if adj[i, j] == 1:
                result[i, j] = 1
    try:
        return CausalDAG.from_adjacency_matrix(result)
    except Exception:
        return None


# ===================================================================
# 6.  MEC enumeration
# ===================================================================

def enumerate_mec(dag: CausalDAG, *, max_dags: int = 1000) -> List[CausalDAG]:
    """Enumerate all DAGs in the Markov equivalence class of *dag*.

    Uses the CPDAG and systematically orients undirected edges in all
    consistent ways.
    """
    cp = to_cpdag(dag)
    n = cp.shape[0]

    undirected: List[Tuple[NodeId, NodeId]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if cp[i, j] == 1 and cp[j, i] == 1:
                undirected.append((i, j))

    if not undirected:
        return [dag.copy()]

    results: List[CausalDAG] = []
    for orientations in itertools.product([0, 1], repeat=len(undirected)):
        if len(results) >= max_dags:
            break
        candidate = cp.copy()
        for idx, (i, j) in enumerate(undirected):
            if orientations[idx] == 0:
                candidate[j, i] = 0
            else:
                candidate[i, j] = 0

        if _is_dag(candidate):
            try:
                d = CausalDAG.from_adjacency_matrix(candidate)
                if is_markov_equivalent(dag, d):
                    results.append(d)
            except Exception:
                continue
    return results


def _is_dag(adj: np.ndarray) -> bool:
    """Quick acyclicity check via topological sort attempt."""
    n = adj.shape[0]
    in_deg = adj.sum(axis=0).astype(int)
    queue = deque(i for i in range(n) if in_deg[i] == 0)
    count = 0
    while queue:
        v = queue.popleft()
        count += 1
        for w in range(n):
            if adj[v, w]:
                in_deg[w] -= 1
                if in_deg[w] == 0:
                    queue.append(w)
    return count == n


# ===================================================================
# 7.  MEC size computation / bounds
# ===================================================================

def mec_size(dag: CausalDAG, *, max_count: int = 10000) -> int:
    """Count the number of DAGs in the MEC (exact for small, bounded for large)."""
    return len(enumerate_mec(dag, max_dags=max_count))


def mec_size_upper_bound(dag: CausalDAG) -> int:
    """Fast upper bound: 2^(number of reversible edges)."""
    return 2 ** len(reversible_edges(dag))


def mec_log_size_bound(dag: CausalDAG) -> float:
    """log2 of the upper bound on MEC size."""
    return float(len(reversible_edges(dag)))


# ===================================================================
# 8.  Representative DAG selection from MEC
# ===================================================================

def representative_dag(dag: CausalDAG, *, strategy: str = "min_edges") -> CausalDAG:
    """Select a representative DAG from the MEC.

    Parameters
    ----------
    strategy : str
        ``"min_edges"`` — fewest forward edges (most reversals).
        ``"max_edges"`` — most forward edges.
        ``"lexicographic"`` — lexicographically smallest adjacency matrix.
    """
    members = enumerate_mec(dag, max_dags=500)
    if not members:
        return dag.copy()

    if strategy == "min_edges":
        return min(members, key=lambda d: d.n_edges)
    elif strategy == "max_edges":
        return max(members, key=lambda d: d.n_edges)
    elif strategy == "lexicographic":
        return min(members, key=lambda d: d.adj.tobytes())
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ===================================================================
# 9.  MEC sampling (uniform random DAG from MEC)
# ===================================================================

def sample_from_mec(
    dag: CausalDAG,
    *,
    n_samples: int = 1,
    rng: Optional[random.Random] = None,
    max_enumeration: int = 5000,
) -> List[CausalDAG]:
    """Sample uniformly at random from the MEC.

    For small MECs we enumerate then sample.  For large MECs we use
    random CPDAG extension with rejection.
    """
    if rng is None:
        rng = random.Random()

    members = enumerate_mec(dag, max_dags=max_enumeration)
    if members:
        return [rng.choice(members) for _ in range(n_samples)]

    cp = to_cpdag(dag)
    samples: List[CausalDAG] = []
    attempts = 0
    while len(samples) < n_samples and attempts < n_samples * 50:
        attempts += 1
        d = _random_extension(cp, rng)
        if d is not None:
            samples.append(d)
    return samples


def _random_extension(cpdag: np.ndarray, rng: random.Random) -> Optional[CausalDAG]:
    """Randomly orient undirected edges and check consistency."""
    n = cpdag.shape[0]
    candidate = cpdag.copy()
    undirected = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if cpdag[i, j] == 1 and cpdag[j, i] == 1
    ]
    for i, j in undirected:
        if rng.random() < 0.5:
            candidate[j, i] = 0
        else:
            candidate[i, j] = 0

    candidate = apply_meek_rules(candidate)
    if not _is_dag(candidate):
        return None
    try:
        return CausalDAG.from_adjacency_matrix(candidate)
    except Exception:
        return None


# ===================================================================
# 10. Interventional equivalence
# ===================================================================

def interventional_equivalence(
    dag1: CausalDAG,
    dag2: CausalDAG,
    intervention_targets: List[Set[NodeId]],
) -> bool:
    """Check if two DAGs are I-equivalent for given intervention targets.

    Two DAGs are I-equivalent w.r.t. targets {I_1, ..., I_k} if they
    induce the same observational AND interventional distributions for
    each target set.

    For a set of targets I, the I-essential graph removes edges into
    intervened-upon nodes before computing the CPDAG.  Two DAGs are
    I-equivalent iff they yield the same I-essential graph for every
    target.
    """
    for targets in intervention_targets:
        ie1 = _interventional_essential_graph(dag1, targets)
        ie2 = _interventional_essential_graph(dag2, targets)
        if not np.array_equal(ie1, ie2):
            return False
    obs_eq = is_markov_equivalent(dag1, dag2)
    return obs_eq


def _interventional_essential_graph(
    dag: CausalDAG,
    targets: Set[NodeId],
) -> np.ndarray:
    """Compute the I-essential graph for a single intervention target set.

    Edges into intervened nodes become compelled.
    """
    cp = to_cpdag(dag)
    n = cp.shape[0]
    for t in targets:
        for p in range(n):
            if cp[p, t] == 1 and cp[t, p] == 1:
                cp[t, p] = 0
    return apply_meek_rules(cp)


def i_mec(
    dag: CausalDAG,
    intervention_targets: List[Set[NodeId]],
    *,
    max_dags: int = 500,
) -> List[CausalDAG]:
    """Enumerate the I-MEC: DAGs interventionally equivalent to *dag*.

    Filters the observational MEC by interventional equivalence.
    """
    obs_mec = enumerate_mec(dag, max_dags=max_dags)
    return [
        d for d in obs_mec
        if interventional_equivalence(dag, d, intervention_targets)
    ]


# ===================================================================
# 11. Distinguishing experiments
# ===================================================================

def distinguishing_targets(
    dag1: CausalDAG,
    dag2: CausalDAG,
    *,
    max_target_size: int = 2,
) -> Optional[Set[NodeId]]:
    """Find a minimal single-target intervention that distinguishes two MECmates.

    Returns None if they are the same DAG.
    """
    if np.array_equal(dag1.adj, dag2.adj):
        return None

    n = dag1.n_nodes
    for size in range(1, max_target_size + 1):
        for combo in itertools.combinations(range(n), size):
            targets = set(combo)
            ie1 = _interventional_essential_graph(dag1, targets)
            ie2 = _interventional_essential_graph(dag2, targets)
            if not np.array_equal(ie1, ie2):
                return targets
    return None


def all_distinguishing_experiments(
    dag1: CausalDAG,
    dag2: CausalDAG,
    *,
    max_target_size: int = 3,
) -> List[Set[NodeId]]:
    """All intervention target sets (up to *max_target_size*) that distinguish."""
    if np.array_equal(dag1.adj, dag2.adj):
        return []

    n = dag1.n_nodes
    results: List[Set[NodeId]] = []
    for size in range(1, max_target_size + 1):
        for combo in itertools.combinations(range(n), size):
            targets = set(combo)
            ie1 = _interventional_essential_graph(dag1, targets)
            ie2 = _interventional_essential_graph(dag2, targets)
            if not np.array_equal(ie1, ie2):
                results.append(targets)
    return results


def minimum_distinguishing_set_size(
    dag1: CausalDAG,
    dag2: CausalDAG,
    *,
    max_size: int = 5,
) -> int:
    """Minimum number of single-node interventions to distinguish two DAGs.

    Returns -1 if not distinguishable within *max_size*.
    """
    n = dag1.n_nodes
    for size in range(1, max_size + 1):
        for combo in itertools.combinations(range(n), size):
            targets = [{c} for c in combo]
            if not interventional_equivalence(dag1, dag2, targets):
                return size
    return -1


# ===================================================================
# 12. Utility helpers
# ===================================================================

def cpdag_edge_counts(dag: CausalDAG) -> Dict[str, int]:
    """Count directed and undirected edges in the CPDAG."""
    cp = to_cpdag(dag)
    n = cp.shape[0]
    directed = 0
    undirected = 0
    for i in range(n):
        for j in range(i + 1, n):
            if cp[i, j] == 1 and cp[j, i] == 1:
                undirected += 1
            elif cp[i, j] == 1 or cp[j, i] == 1:
                directed += 1
    return {"directed": directed, "undirected": undirected}


def fraction_orientable(dag: CausalDAG) -> float:
    """Fraction of edges that are compelled (oriented in every MEC member)."""
    counts = cpdag_edge_counts(dag)
    total = counts["directed"] + counts["undirected"]
    if total == 0:
        return 1.0
    return counts["directed"] / total
