"""Symmetry-breaking constraints for ILP-based DAG solvers.

Implements DAG automorphism detection, orbit computation, and
symmetry-breaking constraints that prune the ILP search space
without eliminating any non-isomorphic solutions.
"""

from __future__ import annotations

import itertools
from collections import defaultdict, deque
from typing import (
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np

from causalcert.dag.graph import CausalDAG

NodeId = int


# ===================================================================
# 1.  Permutation helpers
# ===================================================================

def _apply_permutation(adj: np.ndarray, perm: Sequence[int]) -> np.ndarray:
    """Return the adjacency matrix after relabelling nodes by *perm*."""
    n = adj.shape[0]
    out = np.zeros_like(adj)
    for i in range(n):
        for j in range(n):
            out[perm[i], perm[j]] = adj[i, j]
    return out


def _compose(p: Sequence[int], q: Sequence[int]) -> List[int]:
    """Compose two permutations: (p ∘ q)(i) = p[q[i]]."""
    return [p[q[i]] for i in range(len(p))]


def _inverse(perm: Sequence[int]) -> List[int]:
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def _identity(n: int) -> List[int]:
    return list(range(n))


def _is_identity(perm: Sequence[int]) -> bool:
    return all(perm[i] == i for i in range(len(perm)))


# ===================================================================
# 2.  Automorphism detection (brute-force for small graphs)
# ===================================================================

def automorphism_group(dag: CausalDAG, *, max_group_size: int = 10000) -> List[List[int]]:
    """Compute the automorphism group of the DAG.

    An automorphism is a permutation π such that the adjacency matrix is
    unchanged under π.  This brute-force implementation is only suitable
    for small graphs (n ≤ 10); for larger graphs use the partition-refinement
    approach in :func:`automorphism_group_refined`.

    Returns
    -------
    list of permutation lists  (including the identity)
    """
    adj = dag.adj
    n = dag.n_nodes

    if n > 10:
        return automorphism_group_refined(dag, max_group_size=max_group_size)

    in_deg = adj.sum(axis=0)
    out_deg = adj.sum(axis=1)
    deg_class: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for v in range(n):
        deg_class[(int(in_deg[v]), int(out_deg[v]))].append(v)

    partitions = list(deg_class.values())
    partition_map = {}
    for idx, part in enumerate(partitions):
        for v in part:
            partition_map[v] = idx

    generators: List[List[int]] = [_identity(n)]

    candidate_images: List[List[int]] = []
    for v in range(n):
        candidate_images.append(sorted(partitions[partition_map[v]]))

    for perm_tuple in itertools.product(*candidate_images):
        perm = list(perm_tuple)
        if len(set(perm)) != n:
            continue
        if np.array_equal(_apply_permutation(adj, perm), adj):
            generators.append(perm)
            if len(generators) >= max_group_size:
                break
    return generators


def automorphism_group_refined(
    dag: CausalDAG,
    *,
    max_group_size: int = 10000,
) -> List[List[int]]:
    """Partition-refinement automorphism detection for larger DAGs.

    Uses iterative refinement of the initial degree-based partition to
    prune the search tree.
    """
    adj = dag.adj
    n = dag.n_nodes
    in_deg = adj.sum(axis=0).astype(int)
    out_deg = adj.sum(axis=1).astype(int)

    color = np.zeros(n, dtype=int)
    key_map: Dict[Tuple[int, int], int] = {}
    cid = 0
    for v in range(n):
        k = (int(in_deg[v]), int(out_deg[v]))
        if k not in key_map:
            key_map[k] = cid
            cid += 1
        color[v] = key_map[k]

    for _iteration in range(n):
        new_key_map: Dict[tuple, int] = {}
        new_cid = 0
        new_color = np.zeros(n, dtype=int)
        for v in range(n):
            neighbor_colors_out = tuple(sorted(color[w] for w in range(n) if adj[v, w]))
            neighbor_colors_in = tuple(sorted(color[w] for w in range(n) if adj[w, v]))
            k = (color[v], neighbor_colors_out, neighbor_colors_in)
            if k not in new_key_map:
                new_key_map[k] = new_cid
                new_cid += 1
            new_color[v] = new_key_map[k]
        if np.array_equal(color, new_color):
            break
        color = new_color

    cells: Dict[int, List[int]] = defaultdict(list)
    for v in range(n):
        cells[color[v]].append(v)
    partitions = [sorted(cell) for cell in cells.values()]

    generators: List[List[int]] = [_identity(n)]

    def _try_extend(partial: Dict[int, int], remaining: List[int]) -> None:
        if len(generators) >= max_group_size:
            return
        if not remaining:
            perm = [partial[i] for i in range(n)]
            if np.array_equal(_apply_permutation(adj, perm), adj):
                generators.append(perm)
            return
        v = remaining[0]
        rest = remaining[1:]
        cell = partitions[_partition_index(partitions, v)]
        used = set(partial.values())
        for candidate in cell:
            if candidate in used:
                continue
            ok = True
            for u, img_u in partial.items():
                if adj[u, v] != adj[img_u, candidate]:
                    ok = False
                    break
                if adj[v, u] != adj[candidate, img_u]:
                    ok = False
                    break
            if ok:
                partial[v] = candidate
                _try_extend(partial, rest)
                del partial[v]

    order = list(range(n))
    _try_extend({}, order)
    return generators


def _partition_index(partitions: List[List[int]], v: int) -> int:
    for idx, cell in enumerate(partitions):
        if v in cell:
            return idx
    return 0


# ===================================================================
# 3.  Orbit computation
# ===================================================================

def orbits(group: List[List[int]], n: int) -> List[FrozenSet[int]]:
    """Compute the orbits of the automorphism group on nodes.

    Two nodes are in the same orbit if some automorphism maps one to the other.
    """
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x: int, y: int) -> None:
        rx, ry = _find(x), _find(y)
        if rx != ry:
            parent[rx] = ry

    for perm in group:
        for i in range(n):
            _union(i, perm[i])

    orbit_map: Dict[int, Set[int]] = defaultdict(set)
    for i in range(n):
        orbit_map[_find(i)].add(i)
    return [frozenset(s) for s in orbit_map.values()]


def edge_orbits(
    group: List[List[int]],
    n: int,
) -> List[FrozenSet[Tuple[int, int]]]:
    """Compute orbits on directed edges under the automorphism group."""
    edge_parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
    all_edges = [(i, j) for i in range(n) for j in range(n) if i != j]
    for e in all_edges:
        edge_parent[e] = e

    def _find_e(e: Tuple[int, int]) -> Tuple[int, int]:
        while edge_parent[e] != e:
            edge_parent[e] = edge_parent[edge_parent[e]]
            e = edge_parent[e]
        return e

    def _union_e(e1: Tuple[int, int], e2: Tuple[int, int]) -> None:
        r1, r2 = _find_e(e1), _find_e(e2)
        if r1 != r2:
            edge_parent[r1] = r2

    for perm in group:
        for i, j in all_edges:
            _union_e((i, j), (perm[i], perm[j]))

    orbit_map: Dict[Tuple[int, int], Set[Tuple[int, int]]] = defaultdict(set)
    for e in all_edges:
        orbit_map[_find_e(e)].add(e)
    return [frozenset(s) for s in orbit_map.values()]


def orbit_representatives(group: List[List[int]], n: int) -> List[int]:
    """One representative node from each orbit (smallest id)."""
    return [min(o) for o in orbits(group, n)]


# ===================================================================
# 4.  Lexicographic ordering constraints
# ===================================================================

def lex_leader_constraints(
    group: List[List[int]],
    n: int,
) -> List[List[Tuple[int, int, int, int]]]:
    """Generate lexicographic-leader symmetry-breaking constraints.

    For each non-identity automorphism π, the lex-leader constraint
    requires that the adjacency matrix entries (in row-major order)
    be lexicographically ≤ the permuted version.

    Returns a list of constraints, one per generator.  Each constraint
    is a sequence of (i, j, π(i), π(j)) tuples for sequential comparison.
    """
    constraints: List[List[Tuple[int, int, int, int]]] = []
    for perm in group:
        if _is_identity(perm):
            continue
        pairs: List[Tuple[int, int, int, int]] = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                pi, pj = perm[i], perm[j]
                if (i, j) != (pi, pj):
                    pairs.append((i, j, pi, pj))
        if pairs:
            constraints.append(pairs)
    return constraints


def orbit_fixing_constraints(
    group: List[List[int]],
    n: int,
) -> List[Tuple[int, int]]:
    """For each orbit, fix the representative to be the smallest node.

    Returns constraints of the form (representative, other_member) meaning
    the solution should map representative before other_member.
    """
    constraints: List[Tuple[int, int]] = []
    for orbit in orbits(group, n):
        rep = min(orbit)
        for v in sorted(orbit):
            if v != rep:
                constraints.append((rep, v))
    return constraints


# ===================================================================
# 5.  Symmetry-reduced search space
# ===================================================================

def symmetry_reduced_edges(
    dag: CausalDAG,
    edit_candidates: Optional[List[Tuple[int, int]]] = None,
) -> List[Tuple[int, int]]:
    """Return a reduced set of candidate edge edits using symmetry.

    If two edges are in the same orbit, we only need to consider one
    of them in the ILP.  This can dramatically reduce the number of
    binary variables.
    """
    group = automorphism_group(dag)
    n = dag.n_nodes

    if edit_candidates is None:
        edit_candidates = [(i, j) for i in range(n) for j in range(n) if i != j]

    e_orbits = edge_orbits(group, n)
    orbit_of: Dict[Tuple[int, int], int] = {}
    for idx, orbit in enumerate(e_orbits):
        for e in orbit:
            orbit_of[e] = idx

    seen_orbits: Set[int] = set()
    reduced: List[Tuple[int, int]] = []
    for e in edit_candidates:
        oid = orbit_of.get(e)
        if oid is not None and oid not in seen_orbits:
            reduced.append(e)
            seen_orbits.add(oid)
        elif oid is None:
            reduced.append(e)
    return reduced


def symmetry_reduction_ratio(dag: CausalDAG) -> float:
    """Fraction of the search space eliminated by symmetry.

    Returns a value in [0, 1] where 0 means no reduction and 1 means
    all edges are equivalent.
    """
    n = dag.n_nodes
    total_edges = n * (n - 1)
    if total_edges == 0:
        return 0.0
    reduced = len(symmetry_reduced_edges(dag))
    return 1.0 - reduced / total_edges


# ===================================================================
# 6.  Symmetry-aware branching
# ===================================================================

class SymmetryAwareBrancher:
    """Branching strategy that uses orbit information to break ties.

    When the ILP solver must choose which variable to branch on next,
    prefer variables in smaller orbits (more constrained) and avoid
    branching on symmetric copies of variables that have already been
    decided.
    """

    def __init__(self, dag: CausalDAG) -> None:
        self._dag = dag
        group = automorphism_group(dag)
        self._node_orbits = orbits(group, dag.n_nodes)
        self._edge_orbs = edge_orbits(group, dag.n_nodes)
        self._orbit_size: Dict[Tuple[int, int], int] = {}
        for orb in self._edge_orbs:
            sz = len(orb)
            for e in orb:
                self._orbit_size[e] = sz

    def priority(self, edge: Tuple[int, int]) -> float:
        """Return branching priority (lower = branch first).

        Variables in smaller orbits are branched on first, since they
        have fewer symmetric copies and thus branch choices propagate
        more information.
        """
        return float(self._orbit_size.get(edge, 1))

    def rank_candidates(
        self,
        candidates: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """Sort candidates by branching priority (best first)."""
        return sorted(candidates, key=self.priority)

    def is_symmetry_broken(
        self,
        fixed_edges: Dict[Tuple[int, int], int],
        candidate: Tuple[int, int],
    ) -> bool:
        """Check if branching on *candidate* is redundant due to a symmetric
        edge already being fixed.
        """
        for orb in self._edge_orbs:
            if candidate in orb:
                for e in orb:
                    if e != candidate and e in fixed_edges:
                        return True
        return False

    def prune_symmetric(
        self,
        candidates: List[Tuple[int, int]],
        fixed_edges: Dict[Tuple[int, int], int],
    ) -> List[Tuple[int, int]]:
        """Remove candidates that are symmetric copies of already-fixed edges."""
        return [
            c for c in candidates
            if not self.is_symmetry_broken(fixed_edges, c)
        ]


# ===================================================================
# 7.  Isomorphism checking
# ===================================================================

def is_isomorphic(dag1: CausalDAG, dag2: CausalDAG) -> bool:
    """Check if two DAGs are isomorphic.

    Uses degree-sequence pruning + partition refinement + backtracking.
    """
    if dag1.n_nodes != dag2.n_nodes or dag1.n_edges != dag2.n_edges:
        return False

    n = dag1.n_nodes
    adj1, adj2 = dag1.adj, dag2.adj
    in1 = adj1.sum(axis=0).astype(int)
    out1 = adj1.sum(axis=1).astype(int)
    in2 = adj2.sum(axis=0).astype(int)
    out2 = adj2.sum(axis=1).astype(int)

    sig1 = sorted(zip(in1.tolist(), out1.tolist()))
    sig2 = sorted(zip(in2.tolist(), out2.tolist()))
    if sig1 != sig2:
        return False

    deg_class1: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    deg_class2: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for v in range(n):
        deg_class1[(int(in1[v]), int(out1[v]))].append(v)
        deg_class2[(int(in2[v]), int(out2[v]))].append(v)

    node_order = list(range(n))

    def _backtrack(mapping: Dict[int, int], idx: int) -> bool:
        if idx == n:
            return True
        v = node_order[idx]
        k = (int(in1[v]), int(out1[v]))
        used = set(mapping.values())
        for candidate in deg_class2[k]:
            if candidate in used:
                continue
            ok = True
            for u, img_u in mapping.items():
                if adj1[u, v] != adj2[img_u, candidate]:
                    ok = False
                    break
                if adj1[v, u] != adj2[candidate, img_u]:
                    ok = False
                    break
            if ok:
                mapping[v] = candidate
                if _backtrack(mapping, idx + 1):
                    return True
                del mapping[v]
        return False

    return _backtrack({}, 0)


def find_isomorphism(
    dag1: CausalDAG,
    dag2: CausalDAG,
) -> Optional[List[int]]:
    """Find an isomorphism from dag1 to dag2, or None.

    Returns permutation such that dag2.adj == _apply_permutation(dag1.adj, perm).
    """
    if dag1.n_nodes != dag2.n_nodes or dag1.n_edges != dag2.n_edges:
        return None

    n = dag1.n_nodes
    adj1, adj2 = dag1.adj, dag2.adj
    in1, out1 = adj1.sum(axis=0).astype(int), adj1.sum(axis=1).astype(int)
    in2, out2 = adj2.sum(axis=0).astype(int), adj2.sum(axis=1).astype(int)

    deg_class2: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for v in range(n):
        deg_class2[(int(in2[v]), int(out2[v]))].append(v)

    def _backtrack(mapping: Dict[int, int], idx: int) -> Optional[List[int]]:
        if idx == n:
            return [mapping[i] for i in range(n)]
        v = idx
        k = (int(in1[v]), int(out1[v]))
        used = set(mapping.values())
        for candidate in deg_class2.get(k, []):
            if candidate in used:
                continue
            ok = True
            for u, img_u in mapping.items():
                if adj1[u, v] != adj2[img_u, candidate]:
                    ok = False
                    break
                if adj1[v, u] != adj2[candidate, img_u]:
                    ok = False
                    break
            if ok:
                mapping[v] = candidate
                result = _backtrack(mapping, idx + 1)
                if result is not None:
                    return result
                del mapping[v]
        return None

    return _backtrack({}, 0)


# ===================================================================
# 8.  Summary statistics
# ===================================================================

def symmetry_summary(dag: CausalDAG) -> Dict[str, object]:
    """Compute a summary of symmetry properties."""
    group = automorphism_group(dag)
    orbs = orbits(group, dag.n_nodes)
    return {
        "automorphism_group_size": len(group),
        "n_orbits": len(orbs),
        "orbit_sizes": sorted(len(o) for o in orbs),
        "reduction_ratio": symmetry_reduction_ratio(dag),
        "is_asymmetric": len(group) == 1,
    }
