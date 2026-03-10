"""
Edit algebra implementation — composition, inversion, and canonicalisation.

Implements the :class:`EditAlgebra` protocol with concrete algorithms for
composing, inverting, and canonicalising edit sequences, testing commutativity,
computing edit distance, and extracting group-theoretic properties.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np

from causalcert.types import (
    AdjacencyMatrix,
    EditType,
    EdgeTuple,
    NodeId,
    StructuralEdit,
)
from causalcert.algebra.types import EditComposition, EditLattice, EditSequence
from causalcert.dag.edit import apply_edit, apply_edits, diff_edits, edit_distance
from causalcert.dag.validation import is_dag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INVERSE_MAP = {
    EditType.ADD: EditType.DELETE,
    EditType.DELETE: EditType.ADD,
    EditType.REVERSE: EditType.REVERSE,
}

_SORT_KEY_MAP = {EditType.ADD: 0, EditType.DELETE: 1, EditType.REVERSE: 2}


def _edit_sort_key(e: StructuralEdit) -> tuple[int, int, int]:
    """Canonical sort key: (edit_type_ord, source, target)."""
    return (_SORT_KEY_MAP[e.edit_type], e.source, e.target)


def _invert_single(edit: StructuralEdit) -> StructuralEdit:
    """Invert a single edit.

    ADD ↔ DELETE; REVERSE is self-inverse (but swap source/target).
    """
    if edit.edit_type == EditType.REVERSE:
        return StructuralEdit(EditType.REVERSE, edit.target, edit.source)
    return StructuralEdit(_INVERSE_MAP[edit.edit_type], edit.source, edit.target)


def _cancels(a: StructuralEdit, b: StructuralEdit) -> bool:
    """Return True if *b* undoes *a* on the same edge."""
    if a.edge != b.edge:
        return False
    if a.edit_type == EditType.ADD and b.edit_type == EditType.DELETE:
        return True
    if a.edit_type == EditType.DELETE and b.edit_type == EditType.ADD:
        return True
    if a.edit_type == EditType.REVERSE and b.edit_type == EditType.REVERSE:
        return a.source == b.target and a.target == b.source
    return False


def _simplify(edits: Sequence[StructuralEdit]) -> tuple[StructuralEdit, ...]:
    """Cancel inverse pairs and collapse duplicates.

    Scans through *edits* and removes opposing pairs (ADD then DELETE of the
    same edge, or two REVERSE operations on the same undirected edge).

    Returns
    -------
    tuple[StructuralEdit, ...]
        Simplified edit sequence with redundant pairs removed.
    """
    remaining: list[StructuralEdit] = []
    for edit in edits:
        cancelled = False
        for idx in range(len(remaining) - 1, -1, -1):
            if _cancels(remaining[idx], edit):
                remaining.pop(idx)
                cancelled = True
                break
        if not cancelled:
            remaining.append(edit)
    return tuple(remaining)


def _canonical_order(edits: Sequence[StructuralEdit]) -> tuple[StructuralEdit, ...]:
    """Sort edits into canonical lexicographic order."""
    return tuple(sorted(edits, key=_edit_sort_key))


def _edges_independent(a: StructuralEdit, b: StructuralEdit) -> bool:
    """Two edits are independent when they touch disjoint node pairs.

    Independence means the edits trivially commute because neither
    modifies a cell read by the other.
    """
    a_nodes = {a.source, a.target}
    b_nodes = {b.source, b.target}
    return a_nodes.isdisjoint(b_nodes)


# ---------------------------------------------------------------------------
# EditAlgebraImpl
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EditAlgebraImpl:
    """Concrete implementation of the :class:`EditAlgebra` protocol.

    Provides composition, canonicalisation, commutativity testing,
    subsumption checking, lattice construction, and inversion of
    edit sequences over DAG adjacency matrices.
    """

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    def compose(
        self,
        first: EditSequence,
        second: EditSequence,
        adj: AdjacencyMatrix,
    ) -> EditComposition:
        """Compose two edit sequences, cancelling redundant edits.

        Parameters
        ----------
        first : EditSequence
            Edit sequence applied first.
        second : EditSequence
            Edit sequence applied after *first*.
        adj : AdjacencyMatrix
            DAG to which the edits are applied.

        Returns
        -------
        EditComposition
            Composition with canonical form and commutativity flag.
        """
        combined = list(first.edits) + list(second.edits)
        simplified = _simplify(combined)
        canonical_edits = _canonical_order(simplified)
        canonical_seq = EditSequence(edits=canonical_edits)
        commutes_flag = self.commutes(first, second, adj)
        return EditComposition(
            first=first,
            second=second,
            canonical=canonical_seq,
            commutes=commutes_flag,
        )

    def canonicalise(
        self,
        seq: EditSequence,
        adj: AdjacencyMatrix,
    ) -> EditSequence:
        """Reduce an edit sequence to canonical form.

        The canonical form is the shortest equivalent sequence with edits
        in lexicographic order by ``(edit_type, source, target)``.

        Parameters
        ----------
        seq : EditSequence
            An edit sequence to canonicalise.
        adj : AdjacencyMatrix
            The DAG providing context for redundancy detection.

        Returns
        -------
        EditSequence
            Canonical representative.
        """
        simplified = _simplify(seq.edits)
        ordered = _canonical_order(simplified)
        return EditSequence(edits=ordered, source_hash=seq.source_hash)

    def commutes(
        self,
        a: EditSequence,
        b: EditSequence,
        adj: AdjacencyMatrix,
    ) -> bool:
        """Test whether two edit sequences commute on *adj*.

        Two sequences commute if applying them in either order produces
        the same adjacency matrix.

        Parameters
        ----------
        a, b : EditSequence
            Edit sequences to test.
        adj : AdjacencyMatrix
            The DAG on which commutativity is tested.

        Returns
        -------
        bool
        """
        adj = np.asarray(adj, dtype=np.int8)
        ab = apply_edits(apply_edits(adj, a.edits), b.edits)
        ba = apply_edits(apply_edits(adj, b.edits), a.edits)
        return bool(np.array_equal(ab, ba))

    def subsumes(
        self,
        sub: EditSequence,
        sup: EditSequence,
    ) -> bool:
        """Test whether *sub*'s affected edges are a subset of *sup*'s.

        Parameters
        ----------
        sub, sup : EditSequence
            Candidate subset and superset sequences.

        Returns
        -------
        bool
        """
        return sub.affected_edges <= sup.affected_edges

    def build_lattice(
        self,
        candidates: Sequence[EditSequence],
        adj: AdjacencyMatrix,
    ) -> EditLattice:
        """Build an edit lattice from candidate sequences.

        Inserts every candidate into the lattice, marking each as
        overturning if applying it to *adj* produces a different CPDAG
        (conservative heuristic: any change is considered overturning).

        Parameters
        ----------
        candidates : Sequence[EditSequence]
            Edit sequences to insert.
        adj : AdjacencyMatrix
            The original DAG.

        Returns
        -------
        EditLattice
        """
        adj = np.asarray(adj, dtype=np.int8)
        lattice = EditLattice()
        for seq in candidates:
            perturbed = apply_edits(adj, seq.edits)
            changed = not np.array_equal(adj, perturbed)
            lattice.insert(seq, overturns=changed)
        return lattice

    def inverse(self, seq: EditSequence) -> EditSequence:
        """Compute the inverse of an edit sequence.

        Each edit is inverted and the order is reversed so that composing
        the sequence with its inverse yields the identity.

        Parameters
        ----------
        seq : EditSequence
            Edit sequence to invert.

        Returns
        -------
        EditSequence
        """
        inv_edits = tuple(_invert_single(e) for e in reversed(seq.edits))
        return EditSequence(edits=inv_edits, source_hash=seq.source_hash)

    # ------------------------------------------------------------------
    # Edit distance
    # ------------------------------------------------------------------

    def edit_distance(
        self,
        adj1: AdjacencyMatrix,
        adj2: AdjacencyMatrix,
    ) -> int:
        """Structural Hamming distance (SHD) between two DAGs.

        Each addition, deletion, or reversal counts as one edit.

        Parameters
        ----------
        adj1, adj2 : AdjacencyMatrix
            Adjacency matrices (same shape).

        Returns
        -------
        int
        """
        return edit_distance(adj1, adj2)

    def minimal_edit_sequence(
        self,
        adj1: AdjacencyMatrix,
        adj2: AdjacencyMatrix,
    ) -> EditSequence:
        """Return a minimal (shortest) edit sequence from *adj1* to *adj2*.

        Uses :func:`diff_edits` which already produces one minimal
        sequence.  When multiple equivalent minimal sequences exist,
        the canonical (lexicographically first) representative is returned.

        Parameters
        ----------
        adj1, adj2 : AdjacencyMatrix

        Returns
        -------
        EditSequence
        """
        edits = diff_edits(adj1, adj2)
        canonical = _canonical_order(edits)
        return EditSequence(edits=canonical)

    # ------------------------------------------------------------------
    # Commutativity analysis
    # ------------------------------------------------------------------

    def pairwise_commutativity(
        self,
        edits: Sequence[StructuralEdit],
        adj: AdjacencyMatrix,
    ) -> np.ndarray:
        """Build a boolean pairwise commutativity matrix.

        ``result[i, j]`` is ``True`` iff applying edit *i* then *j*
        produces the same DAG as applying *j* then *i*.

        Parameters
        ----------
        edits : Sequence[StructuralEdit]
            Individual edits to compare.
        adj : AdjacencyMatrix
            The DAG on which commutativity is tested.

        Returns
        -------
        np.ndarray
            Boolean matrix of shape ``(len(edits), len(edits))``.
        """
        adj = np.asarray(adj, dtype=np.int8)
        n = len(edits)
        mat = np.ones((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                seq_i = EditSequence(edits=(edits[i],))
                seq_j = EditSequence(edits=(edits[j],))
                c = self.commutes(seq_i, seq_j, adj)
                mat[i, j] = c
                mat[j, i] = c
        return mat

    def commuting_groups(
        self,
        edits: Sequence[StructuralEdit],
        adj: AdjacencyMatrix,
    ) -> list[list[int]]:
        """Partition edit indices into maximal sets of mutually commuting edits.

        Uses a greedy algorithm: iterate over edits and place each into the
        first group where it commutes with every existing member.

        Parameters
        ----------
        edits : Sequence[StructuralEdit]
            Individual edits.
        adj : AdjacencyMatrix

        Returns
        -------
        list[list[int]]
            Groups of edit indices.
        """
        comm = self.pairwise_commutativity(edits, adj)
        groups: list[list[int]] = []
        for idx in range(len(edits)):
            placed = False
            for g in groups:
                if all(comm[idx, m] for m in g):
                    g.append(idx)
                    placed = True
                    break
            if not placed:
                groups.append([idx])
        return groups

    # ------------------------------------------------------------------
    # Normal form
    # ------------------------------------------------------------------

    def normal_form(
        self,
        seq: EditSequence,
        adj: AdjacencyMatrix,
    ) -> EditSequence:
        """Compute the normal form of an edit sequence.

        The normal form first simplifies (cancels inverse pairs), then
        sorts independent edits into canonical order.  Dependent edits
        preserve their relative order to maintain semantic equivalence.

        Parameters
        ----------
        seq : EditSequence
            Input sequence.
        adj : AdjacencyMatrix

        Returns
        -------
        EditSequence
            Normal-form representative.
        """
        simplified = list(_simplify(seq.edits))
        if len(simplified) <= 1:
            return EditSequence(edits=tuple(simplified), source_hash=seq.source_hash)

        # Build dependency graph: edge (i, j) when edits i and j do NOT
        # commute and thus their relative order matters.
        n = len(simplified)
        dep = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if not _edges_independent(simplified[i], simplified[j]):
                    si = EditSequence(edits=(simplified[i],))
                    sj = EditSequence(edits=(simplified[j],))
                    if not self.commutes(si, sj, adj):
                        dep[i][j] = True

        # Topological sort with canonical tie-breaking (Kahn's algorithm)
        in_deg = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if dep[i][j]:
                    in_deg[j] += 1

        import heapq
        ready: list[tuple[int, int, int, int]] = []
        for i in range(n):
            if in_deg[i] == 0:
                heapq.heappush(ready, _edit_sort_key(simplified[i]) + (i,))

        ordered: list[StructuralEdit] = []
        while ready:
            *_, idx = heapq.heappop(ready)
            ordered.append(simplified[idx])
            for j in range(n):
                if dep[idx][j]:
                    in_deg[j] -= 1
                    if in_deg[j] == 0:
                        heapq.heappush(ready, _edit_sort_key(simplified[j]) + (j,))

        return EditSequence(edits=tuple(ordered), source_hash=seq.source_hash)

    # ------------------------------------------------------------------
    # Group-theoretic properties
    # ------------------------------------------------------------------

    def is_identity(self, seq: EditSequence, adj: AdjacencyMatrix) -> bool:
        """Check whether applying *seq* to *adj* yields the same graph.

        Parameters
        ----------
        seq : EditSequence
        adj : AdjacencyMatrix

        Returns
        -------
        bool
        """
        result = apply_edits(np.asarray(adj, dtype=np.int8), seq.edits)
        return bool(np.array_equal(adj, result))

    def is_involution(self, seq: EditSequence, adj: AdjacencyMatrix) -> bool:
        """Check whether applying *seq* twice is the identity.

        An involution satisfies ``seq ∘ seq = id``.

        Parameters
        ----------
        seq : EditSequence
        adj : AdjacencyMatrix

        Returns
        -------
        bool
        """
        adj = np.asarray(adj, dtype=np.int8)
        once = apply_edits(adj, seq.edits)
        twice = apply_edits(once, seq.edits)
        return bool(np.array_equal(adj, twice))

    def order(self, seq: EditSequence, adj: AdjacencyMatrix, max_order: int = 64) -> int:
        """Compute the order of an edit sequence (smallest k with seq^k = id).

        Parameters
        ----------
        seq : EditSequence
        adj : AdjacencyMatrix
        max_order : int
            Maximum order to search before returning -1.

        Returns
        -------
        int
            Order of the element, or -1 if it exceeds *max_order*.
        """
        adj = np.asarray(adj, dtype=np.int8)
        current = adj.copy()
        for k in range(1, max_order + 1):
            current = apply_edits(current, seq.edits)
            if np.array_equal(adj, current):
                return k
        return -1

    def orbit(
        self,
        seq: EditSequence,
        adj: AdjacencyMatrix,
        max_size: int = 128,
    ) -> list[AdjacencyMatrix]:
        """Compute the orbit of *adj* under repeated application of *seq*.

        Parameters
        ----------
        seq : EditSequence
        adj : AdjacencyMatrix
        max_size : int
            Maximum orbit size before truncation.

        Returns
        -------
        list[AdjacencyMatrix]
            Distinct adjacency matrices visited.
        """
        adj = np.asarray(adj, dtype=np.int8)
        orbit: list[AdjacencyMatrix] = [adj.copy()]
        seen: set[bytes] = {adj.tobytes()}
        current = adj.copy()
        for _ in range(max_size):
            current = apply_edits(current, seq.edits)
            key = current.tobytes()
            if key in seen:
                break
            seen.add(key)
            orbit.append(current.copy())
        return orbit

    def is_dag_preserving(self, seq: EditSequence, adj: AdjacencyMatrix) -> bool:
        """Check whether applying *seq* to *adj* yields a valid DAG.

        Parameters
        ----------
        seq : EditSequence
        adj : AdjacencyMatrix

        Returns
        -------
        bool
        """
        result = apply_edits(np.asarray(adj, dtype=np.int8), seq.edits)
        return is_dag(result)

    def closure(
        self,
        sequences: Sequence[EditSequence],
        adj: AdjacencyMatrix,
        max_elements: int = 256,
    ) -> list[EditSequence]:
        """Compute the closure of a set of edit sequences under composition.

        Repeatedly composes elements until no new canonical sequences are
        produced or *max_elements* is reached.

        Parameters
        ----------
        sequences : Sequence[EditSequence]
            Generator set.
        adj : AdjacencyMatrix
        max_elements : int

        Returns
        -------
        list[EditSequence]
            All distinct (canonical) sequences in the generated group.
        """
        adj = np.asarray(adj, dtype=np.int8)
        canonical_set: dict[tuple[StructuralEdit, ...], EditSequence] = {}
        for seq in sequences:
            canon = self.canonicalise(seq, adj)
            canonical_set[canon.edits] = canon

        changed = True
        while changed and len(canonical_set) < max_elements:
            changed = False
            current = list(canonical_set.values())
            for a in current:
                for b in current:
                    comp = self.compose(a, b, adj)
                    key = comp.canonical.edits
                    if key not in canonical_set:
                        canonical_set[key] = comp.canonical
                        changed = True
                        if len(canonical_set) >= max_elements:
                            break
                if len(canonical_set) >= max_elements:
                    break

        return list(canonical_set.values())
