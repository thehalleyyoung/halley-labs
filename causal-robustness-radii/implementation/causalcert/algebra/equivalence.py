"""
Edit equivalence — MEC-aware equivalence classes and quotient lattice.

Provides algorithms for grouping DAG edits that lead to the same Markov
equivalence class, enumerating equivalence classes, selecting canonical
representatives, constructing the quotient lattice, and testing DAG
isomorphism under edit transformations.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from causalcert.types import (
    AdjacencyMatrix,
    EditType,
    EdgeTuple,
    NodeId,
    StructuralEdit,
)
from causalcert.algebra.types import EditSequence
from causalcert.dag.edit import apply_edits, diff_edits, edit_distance
from causalcert.dag.mec import is_mec_equivalent, to_cpdag
from causalcert.dag.validation import is_dag


# ---------------------------------------------------------------------------
# CPDAG fingerprint
# ---------------------------------------------------------------------------


def _cpdag_key(adj: np.ndarray) -> bytes:
    """Return a hashable key for the CPDAG of *adj*."""
    return to_cpdag(adj).tobytes()


def _adj_key(adj: np.ndarray) -> bytes:
    """Return a hashable key for a DAG adjacency matrix."""
    return np.asarray(adj, dtype=np.int8).tobytes()


# ---------------------------------------------------------------------------
# EquivalenceClass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EquivalenceClass:
    """A group of edit sequences that produce DAGs in the same MEC.

    Attributes
    ----------
    cpdag_key : bytes
        Hashable fingerprint of the shared CPDAG.
    members : list[EditSequence]
        Edit sequences belonging to this class.
    representative : EditSequence | None
        Canonical representative (shortest, then lex-first).
    member_adjs : list[AdjacencyMatrix]
        Resulting adjacency matrices for each member.
    """

    cpdag_key: bytes
    members: list[EditSequence] = field(default_factory=list)
    representative: EditSequence | None = None
    member_adjs: list[AdjacencyMatrix] = field(default_factory=list)


# ---------------------------------------------------------------------------
# QuotientNode / QuotientLattice
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class QuotientNode:
    """Node in the quotient lattice (one per equivalence class).

    Attributes
    ----------
    class_id : int
        Index of the equivalence class.
    cpdag_key : bytes
        CPDAG fingerprint.
    representative : EditSequence
        Canonical representative edit sequence.
    size : int
        Number of DAGs in the class reachable within the edit radius.
    """

    class_id: int
    cpdag_key: bytes
    representative: EditSequence
    size: int = 1


@dataclass(slots=True)
class QuotientLattice:
    """Quotient lattice: equivalence classes ordered by edit distance.

    Attributes
    ----------
    nodes : list[QuotientNode]
        Nodes of the quotient lattice.
    edges : list[tuple[int, int, int]]
        ``(src_class, dst_class, min_edit_dist)`` — transitions between
        equivalence classes with the minimum edit distance.
    class_of_origin : int
        Index of the class containing the original DAG.
    """

    nodes: list[QuotientNode] = field(default_factory=list)
    edges: list[tuple[int, int, int]] = field(default_factory=list)
    class_of_origin: int = 0


# ---------------------------------------------------------------------------
# EditEquivalence
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EditEquivalence:
    """MEC-aware edit equivalence operations.

    Groups edit sequences by the Markov equivalence class of their
    resulting DAGs, enumerates equivalence classes, selects canonical
    representatives, constructs quotient lattices, and tests DAG
    isomorphism under edit transformations.

    Attributes
    ----------
    adj : AdjacencyMatrix
        The original (centre) DAG.
    """

    adj: AdjacencyMatrix

    def __post_init__(self) -> None:
        self.adj = np.asarray(self.adj, dtype=np.int8)

    # ------------------------------------------------------------------
    # MEC-aware equivalence
    # ------------------------------------------------------------------

    def are_mec_equivalent(
        self,
        seq_a: EditSequence,
        seq_b: EditSequence,
    ) -> bool:
        """Check whether two edit sequences lead to the same MEC.

        Parameters
        ----------
        seq_a, seq_b : EditSequence

        Returns
        -------
        bool
        """
        adj_a = apply_edits(self.adj, seq_a.edits)
        adj_b = apply_edits(self.adj, seq_b.edits)
        if not is_dag(adj_a) or not is_dag(adj_b):
            return _adj_key(adj_a) == _adj_key(adj_b)
        return is_mec_equivalent(adj_a, adj_b)

    def mec_equivalence_classes(
        self,
        sequences: Sequence[EditSequence],
    ) -> list[EquivalenceClass]:
        """Partition edit sequences into MEC-equivalence classes.

        Parameters
        ----------
        sequences : Sequence[EditSequence]

        Returns
        -------
        list[EquivalenceClass]
            Each class groups sequences producing MEC-equivalent DAGs.
        """
        classes: dict[bytes, EquivalenceClass] = {}
        for seq in sequences:
            result_adj = apply_edits(self.adj, seq.edits)
            if is_dag(result_adj):
                key = _cpdag_key(result_adj)
            else:
                key = _adj_key(result_adj)
            if key not in classes:
                classes[key] = EquivalenceClass(cpdag_key=key)
            classes[key].members.append(seq)
            classes[key].member_adjs.append(result_adj)

        result: list[EquivalenceClass] = []
        for ec in classes.values():
            ec.representative = self._select_representative(ec.members)
            result.append(ec)
        return result

    # ------------------------------------------------------------------
    # Canonical representative
    # ------------------------------------------------------------------

    def _select_representative(
        self,
        members: list[EditSequence],
    ) -> EditSequence:
        """Select the canonical representative: shortest, then lex-first.

        Parameters
        ----------
        members : list[EditSequence]

        Returns
        -------
        EditSequence
        """
        def _key(seq: EditSequence) -> tuple[int, tuple[tuple[str, int, int], ...]]:
            edit_keys = tuple(
                (e.edit_type.value, e.source, e.target) for e in seq.edits
            )
            return (seq.length, edit_keys)

        return min(members, key=_key)

    def canonical_representative(
        self,
        seq: EditSequence,
        others: Sequence[EditSequence],
    ) -> EditSequence:
        """Find the canonical representative of *seq*'s equivalence class.

        Searches among *others* (and *seq* itself) for the representative.

        Parameters
        ----------
        seq : EditSequence
        others : Sequence[EditSequence]

        Returns
        -------
        EditSequence
        """
        adj_seq = apply_edits(self.adj, seq.edits)
        candidates = [seq]
        for other in others:
            adj_other = apply_edits(self.adj, other.edits)
            if is_dag(adj_seq) and is_dag(adj_other):
                if is_mec_equivalent(adj_seq, adj_other):
                    candidates.append(other)
            elif _adj_key(adj_seq) == _adj_key(adj_other):
                candidates.append(other)
        return self._select_representative(candidates)

    # ------------------------------------------------------------------
    # Equivalence class enumeration
    # ------------------------------------------------------------------

    def enumerate_classes(
        self,
        max_distance: int,
        *,
        acyclic_only: bool = True,
    ) -> list[EquivalenceClass]:
        """Enumerate MEC-equivalence classes within an edit radius.

        BFS from the centre DAG up to *max_distance*, grouping every
        visited DAG by its CPDAG fingerprint.

        Parameters
        ----------
        max_distance : int
        acyclic_only : bool

        Returns
        -------
        list[EquivalenceClass]
        """
        from causalcert.algebra.lattice import _all_possible_edits
        from causalcert.dag.edit import apply_edit

        classes: dict[bytes, EquivalenceClass] = {}
        seen_adj: set[bytes] = {_adj_key(self.adj)}

        # Register the origin
        origin_key = _cpdag_key(self.adj)
        origin_seq = EditSequence(edits=())
        classes[origin_key] = EquivalenceClass(cpdag_key=origin_key)
        classes[origin_key].members.append(origin_seq)
        classes[origin_key].member_adjs.append(self.adj.copy())

        from collections import deque

        queue: deque[tuple[np.ndarray, tuple[StructuralEdit, ...], int]] = deque()
        queue.append((self.adj.copy(), (), 0))

        while queue:
            cur_adj, cur_edits, depth = queue.popleft()
            if depth >= max_distance:
                continue
            for edit in _all_possible_edits(cur_adj):
                new_adj = apply_edit(cur_adj, edit)
                if acyclic_only and not is_dag(new_adj):
                    continue
                adj_bytes = _adj_key(new_adj)
                if adj_bytes in seen_adj:
                    continue
                seen_adj.add(adj_bytes)

                new_edits = cur_edits + (edit,)
                seq = EditSequence(edits=new_edits)
                cp_key = _cpdag_key(new_adj)
                if cp_key not in classes:
                    classes[cp_key] = EquivalenceClass(cpdag_key=cp_key)
                classes[cp_key].members.append(seq)
                classes[cp_key].member_adjs.append(new_adj)

                if depth + 1 < max_distance:
                    queue.append((new_adj, new_edits, depth + 1))

        for ec in classes.values():
            ec.representative = self._select_representative(ec.members)
        return list(classes.values())

    # ------------------------------------------------------------------
    # Quotient lattice
    # ------------------------------------------------------------------

    def quotient_lattice(
        self,
        classes: Sequence[EquivalenceClass],
    ) -> QuotientLattice:
        """Build the quotient lattice from equivalence classes.

        Nodes are equivalence classes; edges connect classes whose
        representative DAGs are at edit distance 1.

        Parameters
        ----------
        classes : Sequence[EquivalenceClass]

        Returns
        -------
        QuotientLattice
        """
        lattice = QuotientLattice()
        origin_cpdag = _cpdag_key(self.adj)

        for i, ec in enumerate(classes):
            rep = ec.representative or EditSequence(edits=())
            node = QuotientNode(
                class_id=i,
                cpdag_key=ec.cpdag_key,
                representative=rep,
                size=len(ec.members),
            )
            lattice.nodes.append(node)
            if ec.cpdag_key == origin_cpdag:
                lattice.class_of_origin = i

        # Build edges: connect classes whose representative DAGs differ
        # by a small edit distance.
        rep_adjs: list[np.ndarray | None] = []
        for ec in classes:
            if ec.member_adjs:
                rep_adjs.append(np.asarray(ec.member_adjs[0], dtype=np.int8))
            else:
                rep_adjs.append(None)

        n_classes = len(classes)
        for i in range(n_classes):
            if rep_adjs[i] is None:
                continue
            for j in range(i + 1, n_classes):
                if rep_adjs[j] is None:
                    continue
                dist = edit_distance(rep_adjs[i], rep_adjs[j])
                if dist <= 2:
                    lattice.edges.append((i, j, dist))
                    lattice.edges.append((j, i, dist))

        return lattice

    def build_quotient(self, max_distance: int) -> QuotientLattice:
        """Convenience: enumerate classes then build the quotient lattice.

        Parameters
        ----------
        max_distance : int

        Returns
        -------
        QuotientLattice
        """
        classes = self.enumerate_classes(max_distance)
        return self.quotient_lattice(classes)

    # ------------------------------------------------------------------
    # Isomorphism testing
    # ------------------------------------------------------------------

    def is_isomorphic(
        self,
        adj1: AdjacencyMatrix,
        adj2: AdjacencyMatrix,
    ) -> bool:
        """Test whether two DAGs are isomorphic under node relabelling.

        Uses a canonical-form approach: sort rows/columns by
        (in-degree, out-degree) then check all permutations within
        each degree class.  Exact for small graphs; falls back to
        heuristic for large graphs.

        Parameters
        ----------
        adj1, adj2 : AdjacencyMatrix

        Returns
        -------
        bool
        """
        a1 = np.asarray(adj1, dtype=np.int8)
        a2 = np.asarray(adj2, dtype=np.int8)
        if a1.shape != a2.shape:
            return False
        n = a1.shape[0]
        if n == 0:
            return True

        # Quick checks: edge count, degree sequences
        if int(a1.sum()) != int(a2.sum()):
            return False

        in1 = sorted(a1.sum(axis=0).tolist())
        in2 = sorted(a2.sum(axis=0).tolist())
        if in1 != in2:
            return False

        out1 = sorted(a1.sum(axis=1).tolist())
        out2 = sorted(a2.sum(axis=1).tolist())
        if out1 != out2:
            return False

        if n <= 8:
            return self._isomorphic_exact(a1, a2)
        return self._isomorphic_hash(a1, a2)

    def _isomorphic_exact(
        self, a1: np.ndarray, a2: np.ndarray
    ) -> bool:
        """Exact isomorphism via degree-class permutation enumeration."""
        n = a1.shape[0]
        # Group nodes by (in_degree, out_degree)
        deg1: dict[tuple[int, int], list[int]] = defaultdict(list)
        deg2: dict[tuple[int, int], list[int]] = defaultdict(list)
        for v in range(n):
            d1 = (int(a1[:, v].sum()), int(a1[v, :].sum()))
            d2 = (int(a2[:, v].sum()), int(a2[v, :].sum()))
            deg1[d1].append(v)
            deg2[d2].append(v)

        if sorted(deg1.keys()) != sorted(deg2.keys()):
            return False
        for key in deg1:
            if len(deg1[key]) != len(deg2[key]):
                return False

        # Build all valid permutations by matching degree classes
        from itertools import permutations as iterperms

        classes = sorted(deg1.keys())
        groups1 = [deg1[k] for k in classes]
        groups2 = [deg2[k] for k in classes]

        def _check_perm(perm: list[int]) -> bool:
            for i in range(n):
                for j in range(n):
                    if a1[i, j] != a2[perm[i], perm[j]]:
                        return False
            return True

        def _enumerate(idx: int, partial: list[int]) -> bool:
            if idx == len(groups1):
                return _check_perm(partial)
            for p in iterperms(groups2[idx]):
                candidate = partial.copy()
                for orig, mapped in zip(groups1[idx], p):
                    candidate[orig] = mapped
                if _enumerate(idx + 1, candidate):
                    return True
            return False

        return _enumerate(0, [0] * n)

    def _isomorphic_hash(
        self, a1: np.ndarray, a2: np.ndarray
    ) -> bool:
        """Heuristic isomorphism via iterated neighbourhood hashing.

        Assigns each node a hash based on its neighbourhood structure,
        then refines iteratively.  If the sorted hash sequences match,
        the graphs are likely isomorphic.  This is a necessary condition
        (not sufficient), but efficient for large graphs.
        """
        n = a1.shape[0]
        rounds = min(n, 10)

        def _hash_sequence(adj: np.ndarray) -> tuple[int, ...]:
            h = np.array([hash((int(adj[:, v].sum()), int(adj[v, :].sum())))
                          for v in range(n)], dtype=np.int64)
            for _ in range(rounds):
                new_h = np.zeros(n, dtype=np.int64)
                for v in range(n):
                    parents = tuple(sorted(h[np.nonzero(adj[:, v])[0]].tolist()))
                    children = tuple(sorted(h[np.nonzero(adj[v, :])[0]].tolist()))
                    new_h[v] = hash((h[v], parents, children))
                h = new_h
            return tuple(sorted(h.tolist()))

        return _hash_sequence(a1) == _hash_sequence(a2)

    def isomorphism_classes(
        self,
        adjs: Sequence[AdjacencyMatrix],
    ) -> list[list[int]]:
        """Partition adjacency matrices into isomorphism classes.

        Parameters
        ----------
        adjs : Sequence[AdjacencyMatrix]

        Returns
        -------
        list[list[int]]
            Groups of indices that are pairwise isomorphic.
        """
        n = len(adjs)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for i in range(n):
            for j in range(i + 1, n):
                if find(i) != find(j) and self.is_isomorphic(adjs[i], adjs[j]):
                    union(i, j)

        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)
        return list(groups.values())

    # ------------------------------------------------------------------
    # Equivalence-aware edit distance
    # ------------------------------------------------------------------

    def mec_edit_distance(
        self,
        adj1: AdjacencyMatrix,
        adj2: AdjacencyMatrix,
    ) -> int:
        """Minimum edit distance considering MEC equivalence.

        If *adj1* and *adj2* are MEC-equivalent, returns 0.  Otherwise,
        returns the standard edit distance.

        Parameters
        ----------
        adj1, adj2 : AdjacencyMatrix

        Returns
        -------
        int
        """
        a1 = np.asarray(adj1, dtype=np.int8)
        a2 = np.asarray(adj2, dtype=np.int8)
        if is_dag(a1) and is_dag(a2) and is_mec_equivalent(a1, a2):
            return 0
        return edit_distance(a1, a2)

    def equivalence_class_distances(
        self,
        classes: Sequence[EquivalenceClass],
    ) -> np.ndarray:
        """Compute pairwise distances between equivalence classes.

        Distance between two classes is the minimum edit distance between
        any pair of representative DAGs.

        Parameters
        ----------
        classes : Sequence[EquivalenceClass]

        Returns
        -------
        np.ndarray
            Distance matrix of shape ``(n_classes, n_classes)``.
        """
        n = len(classes)
        dist = np.zeros((n, n), dtype=int)
        rep_adjs = []
        for ec in classes:
            if ec.member_adjs:
                rep_adjs.append(np.asarray(ec.member_adjs[0], dtype=np.int8))
            else:
                rep_adjs.append(None)

        for i in range(n):
            for j in range(i + 1, n):
                if rep_adjs[i] is not None and rep_adjs[j] is not None:
                    d = edit_distance(rep_adjs[i], rep_adjs[j])
                else:
                    d = -1
                dist[i, j] = d
                dist[j, i] = d
        return dist
