"""
Edit lattice operations — Hasse diagrams, meet/join, and traversal.

Implements lattice-theoretic operations over the space of DAG edit sets,
including Hasse diagram construction, meet/join computation, level-set
enumeration, BFS/DFS traversal, connected-component analysis, and full
enumeration of all DAGs within a given edit radius.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Iterator, Sequence

import numpy as np

from causalcert.types import (
    AdjacencyMatrix,
    EditType,
    EdgeTuple,
    NodeId,
    StructuralEdit,
)
from causalcert.algebra.types import EditLattice, EditSequence
from causalcert.dag.edit import (
    apply_edit,
    apply_edits,
    all_single_edits,
    edit_distance,
    diff_edits,
)
from causalcert.dag.validation import is_dag


# ---------------------------------------------------------------------------
# Hasse diagram
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HasseDiagram:
    """Hasse diagram of an edit lattice.

    The Hasse diagram records *covering relations*: an edge ``(a, b)``
    means that ``a`` is covered by ``b`` (i.e. ``a ⊂ b`` with no
    intermediate element between them in the partial order).

    Attributes
    ----------
    nodes : list[frozenset[EdgeTuple]]
        Each node is the set of affected edges of an edit sequence.
    covers : list[tuple[int, int]]
        Covering relations ``(i, j)`` meaning node *i* is covered by *j*.
    node_to_idx : dict[frozenset[EdgeTuple], int]
        Map from edge set to its index in *nodes*.
    """

    nodes: list[frozenset[EdgeTuple]] = field(default_factory=list)
    covers: list[tuple[int, int]] = field(default_factory=list)
    node_to_idx: dict[frozenset[EdgeTuple], int] = field(default_factory=dict)


def build_hasse(sequences: Sequence[EditSequence]) -> HasseDiagram:
    """Build the Hasse diagram of edit sequences ordered by edge-set inclusion.

    Parameters
    ----------
    sequences : Sequence[EditSequence]
        Edit sequences whose affected edge sets define lattice elements.

    Returns
    -------
    HasseDiagram
    """
    edge_sets: list[frozenset[EdgeTuple]] = []
    seen: set[frozenset[EdgeTuple]] = set()
    for seq in sequences:
        es = seq.affected_edges
        if es not in seen:
            seen.add(es)
            edge_sets.append(es)

    # Sort by size for efficient subset checking
    edge_sets.sort(key=len)

    diagram = HasseDiagram()
    for es in edge_sets:
        idx = len(diagram.nodes)
        diagram.nodes.append(es)
        diagram.node_to_idx[es] = idx

    # Build covering relations: (a, b) if a ⊂ b with no c s.t. a ⊂ c ⊂ b
    n = len(diagram.nodes)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = diagram.nodes[i], diagram.nodes[j]
            if not a < b:
                continue
            # Check that no intermediate element exists
            intermediate = False
            for k in range(n):
                if k == i or k == j:
                    continue
                c = diagram.nodes[k]
                if a < c < b:
                    intermediate = True
                    break
            if not intermediate:
                diagram.covers.append((i, j))

    return diagram


# ---------------------------------------------------------------------------
# EditLatticeImpl
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EditLatticeImpl:
    """Lattice operations over DAG edit sets.

    Wraps an adjacency matrix and provides methods for Hasse diagram
    construction, meet/join computation, level-set enumeration,
    lattice traversal, connected-component analysis, and full
    neighbourhood enumeration.

    Attributes
    ----------
    adj : AdjacencyMatrix
        The centre DAG of the lattice.
    """

    adj: AdjacencyMatrix

    def __post_init__(self) -> None:
        self.adj = np.asarray(self.adj, dtype=np.int8)

    # ------------------------------------------------------------------
    # Hasse diagram
    # ------------------------------------------------------------------

    def hasse_diagram(
        self, sequences: Sequence[EditSequence]
    ) -> HasseDiagram:
        """Build the Hasse diagram for a collection of edit sequences.

        Parameters
        ----------
        sequences : Sequence[EditSequence]

        Returns
        -------
        HasseDiagram
        """
        return build_hasse(sequences)

    # ------------------------------------------------------------------
    # Meet / join
    # ------------------------------------------------------------------

    def meet(
        self,
        a: EditSequence,
        b: EditSequence,
    ) -> frozenset[EdgeTuple]:
        """Greatest lower bound: intersection of affected edge sets.

        Parameters
        ----------
        a, b : EditSequence

        Returns
        -------
        frozenset[EdgeTuple]
        """
        return a.affected_edges & b.affected_edges

    def join(
        self,
        a: EditSequence,
        b: EditSequence,
    ) -> frozenset[EdgeTuple]:
        """Least upper bound: union of affected edge sets.

        Parameters
        ----------
        a, b : EditSequence

        Returns
        -------
        frozenset[EdgeTuple]
        """
        return a.affected_edges | b.affected_edges

    def meet_sequence(
        self,
        a: EditSequence,
        b: EditSequence,
    ) -> EditSequence:
        """Return the sub-sequence of *a* restricted to edges shared with *b*.

        Parameters
        ----------
        a, b : EditSequence

        Returns
        -------
        EditSequence
        """
        shared = a.affected_edges & b.affected_edges
        edits = tuple(e for e in a.edits if e.edge in shared)
        return EditSequence(edits=edits)

    def join_sequence(
        self,
        a: EditSequence,
        b: EditSequence,
    ) -> EditSequence:
        """Return the merged sequence containing edits from both *a* and *b*.

        If both sequences edit the same edge, the edit from *a* is kept.

        Parameters
        ----------
        a, b : EditSequence

        Returns
        -------
        EditSequence
        """
        edge_map: dict[EdgeTuple, StructuralEdit] = {}
        for e in a.edits:
            edge_map[e.edge] = e
        for e in b.edits:
            if e.edge not in edge_map:
                edge_map[e.edge] = e
        sorted_edits = sorted(edge_map.values(), key=lambda e: (e.source, e.target))
        return EditSequence(edits=tuple(sorted_edits))

    # ------------------------------------------------------------------
    # Level sets
    # ------------------------------------------------------------------

    def level_set(
        self,
        k: int,
        *,
        acyclic_only: bool = True,
    ) -> list[tuple[AdjacencyMatrix, tuple[StructuralEdit, ...]]]:
        """Return all DAGs at edit distance exactly *k* from the centre DAG.

        Parameters
        ----------
        k : int
            Exact edit distance.
        acyclic_only : bool
            If ``True``, only return valid DAGs.

        Returns
        -------
        list[tuple[AdjacencyMatrix, tuple[StructuralEdit, ...]]]
            ``(adj, edits)`` pairs at distance exactly *k*.
        """
        if k < 0:
            return []
        if k == 0:
            return [(self.adj.copy(), ())]

        results: list[tuple[AdjacencyMatrix, tuple[StructuralEdit, ...]]] = []
        seen: set[bytes] = {self.adj.tobytes()}

        # BFS up to distance k, collect only those at exactly distance k
        current_level: list[tuple[np.ndarray, tuple[StructuralEdit, ...]]] = [
            (self.adj.copy(), ())
        ]
        for depth in range(1, k + 1):
            next_level: list[tuple[np.ndarray, tuple[StructuralEdit, ...]]] = []
            for cur_adj, cur_edits in current_level:
                for edit in _all_possible_edits(cur_adj):
                    new_adj = apply_edit(cur_adj, edit)
                    if acyclic_only and not is_dag(new_adj):
                        continue
                    key = new_adj.tobytes()
                    if key in seen:
                        continue
                    seen.add(key)
                    new_edits = cur_edits + (edit,)
                    next_level.append((new_adj, new_edits))
            current_level = next_level

        for adj_k, edits_k in current_level:
            results.append((adj_k, edits_k))
        return results

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def bfs(
        self,
        max_distance: int,
        *,
        acyclic_only: bool = True,
        visitor: Callable[
            [AdjacencyMatrix, tuple[StructuralEdit, ...], int], bool
        ]
        | None = None,
    ) -> list[tuple[AdjacencyMatrix, tuple[StructuralEdit, ...], int]]:
        """BFS traversal of the edit neighbourhood ordered by distance.

        Parameters
        ----------
        max_distance : int
            Maximum edit distance to explore.
        acyclic_only : bool
            Skip graphs with cycles.
        visitor : callable, optional
            Called as ``visitor(adj, edits, distance)``.  Return ``False``
            to prune the subtree rooted at this node.

        Returns
        -------
        list[tuple[AdjacencyMatrix, tuple[StructuralEdit, ...], int]]
            ``(adj, edits, distance)`` triples in BFS order.
        """
        result: list[tuple[AdjacencyMatrix, tuple[StructuralEdit, ...], int]] = []
        seen: set[bytes] = {self.adj.tobytes()}
        queue: deque[tuple[np.ndarray, tuple[StructuralEdit, ...], int]] = deque()
        queue.append((self.adj.copy(), (), 0))
        result.append((self.adj.copy(), (), 0))

        while queue:
            cur_adj, cur_edits, depth = queue.popleft()
            if depth >= max_distance:
                continue
            for edit in _all_possible_edits(cur_adj):
                new_adj = apply_edit(cur_adj, edit)
                if acyclic_only and not is_dag(new_adj):
                    continue
                key = new_adj.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                new_edits = cur_edits + (edit,)
                new_depth = depth + 1
                if visitor is not None and not visitor(new_adj, new_edits, new_depth):
                    continue
                result.append((new_adj, new_edits, new_depth))
                if new_depth < max_distance:
                    queue.append((new_adj, new_edits, new_depth))
        return result

    def dfs(
        self,
        max_distance: int,
        *,
        acyclic_only: bool = True,
        visitor: Callable[
            [AdjacencyMatrix, tuple[StructuralEdit, ...], int], bool
        ]
        | None = None,
    ) -> list[tuple[AdjacencyMatrix, tuple[StructuralEdit, ...], int]]:
        """DFS traversal of the edit neighbourhood.

        Parameters
        ----------
        max_distance : int
            Maximum edit distance.
        acyclic_only : bool
            Skip graphs with cycles.
        visitor : callable, optional
            Return ``False`` to prune.

        Returns
        -------
        list[tuple[AdjacencyMatrix, tuple[StructuralEdit, ...], int]]
        """
        result: list[tuple[AdjacencyMatrix, tuple[StructuralEdit, ...], int]] = []
        seen: set[bytes] = {self.adj.tobytes()}
        result.append((self.adj.copy(), (), 0))

        stack: list[tuple[np.ndarray, tuple[StructuralEdit, ...], int]] = [
            (self.adj.copy(), (), 0)
        ]
        while stack:
            cur_adj, cur_edits, depth = stack.pop()
            if depth >= max_distance:
                continue
            for edit in _all_possible_edits(cur_adj):
                new_adj = apply_edit(cur_adj, edit)
                if acyclic_only and not is_dag(new_adj):
                    continue
                key = new_adj.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                new_edits = cur_edits + (edit,)
                new_depth = depth + 1
                if visitor is not None and not visitor(new_adj, new_edits, new_depth):
                    continue
                result.append((new_adj, new_edits, new_depth))
                if new_depth < max_distance:
                    stack.append((new_adj, new_edits, new_depth))
        return result

    # ------------------------------------------------------------------
    # Connected components
    # ------------------------------------------------------------------

    def connected_components(
        self,
        sequences: Sequence[EditSequence],
    ) -> list[list[int]]:
        """Find connected components among edit sequences.

        Two sequences are *adjacent* if their affected edge sets differ by
        exactly one edge (i.e. edit distance 1 in the Hasse diagram).

        Parameters
        ----------
        sequences : Sequence[EditSequence]

        Returns
        -------
        list[list[int]]
            Groups of indices forming connected components.
        """
        n = len(sequences)
        if n == 0:
            return []

        adj_list: dict[int, list[int]] = defaultdict(list)
        edge_sets = [seq.affected_edges for seq in sequences]
        for i in range(n):
            for j in range(i + 1, n):
                # Adjacent if symmetric difference has size 1
                sym_diff = edge_sets[i] ^ edge_sets[j]
                if len(sym_diff) == 1:
                    adj_list[i].append(j)
                    adj_list[j].append(i)

        visited: set[int] = set()
        components: list[list[int]] = []
        for start in range(n):
            if start in visited:
                continue
            comp: list[int] = []
            q: deque[int] = deque([start])
            visited.add(start)
            while q:
                node = q.popleft()
                comp.append(node)
                for nb in adj_list[node]:
                    if nb not in visited:
                        visited.add(nb)
                        q.append(nb)
            components.append(comp)
        return components

    def connected_components_by_overlap(
        self,
        sequences: Sequence[EditSequence],
    ) -> list[list[int]]:
        """Find connected components by overlapping edge sets.

        Two sequences are connected if their affected edge sets share at
        least one edge.

        Parameters
        ----------
        sequences : Sequence[EditSequence]

        Returns
        -------
        list[list[int]]
        """
        n = len(sequences)
        if n == 0:
            return []

        # Union-Find
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

        edge_sets = [seq.affected_edges for seq in sequences]
        for i in range(n):
            for j in range(i + 1, n):
                if edge_sets[i] & edge_sets[j]:
                    union(i, j)

        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)
        return list(groups.values())

    # ------------------------------------------------------------------
    # Enumeration
    # ------------------------------------------------------------------

    def enumerate_neighbourhood(
        self,
        k: int,
        *,
        acyclic_only: bool = True,
    ) -> list[tuple[AdjacencyMatrix, tuple[StructuralEdit, ...]]]:
        """Enumerate all DAGs within edit distance ≤ *k*.

        Parameters
        ----------
        k : int
            Maximum edit radius.
        acyclic_only : bool
            Restrict to valid DAGs.

        Returns
        -------
        list[tuple[AdjacencyMatrix, tuple[StructuralEdit, ...]]]
        """
        results: list[tuple[AdjacencyMatrix, tuple[StructuralEdit, ...]]] = []
        seen: set[bytes] = {self.adj.tobytes()}
        results.append((self.adj.copy(), ()))

        queue: deque[tuple[np.ndarray, tuple[StructuralEdit, ...], int]] = deque()
        queue.append((self.adj.copy(), (), 0))

        while queue:
            cur_adj, cur_edits, depth = queue.popleft()
            if depth >= k:
                continue
            for edit in _all_possible_edits(cur_adj):
                new_adj = apply_edit(cur_adj, edit)
                if acyclic_only and not is_dag(new_adj):
                    continue
                key = new_adj.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                new_edits = cur_edits + (edit,)
                results.append((new_adj, new_edits))
                if depth + 1 < k:
                    queue.append((new_adj, new_edits, depth + 1))
        return results

    def count_by_distance(
        self,
        max_k: int,
        *,
        acyclic_only: bool = True,
    ) -> dict[int, int]:
        """Count DAGs at each edit distance from 0 to *max_k*.

        Parameters
        ----------
        max_k : int
        acyclic_only : bool

        Returns
        -------
        dict[int, int]
            Mapping from distance to count.
        """
        counts: dict[int, int] = {0: 1}
        seen: set[bytes] = {self.adj.tobytes()}

        current: list[np.ndarray] = [self.adj.copy()]
        for d in range(1, max_k + 1):
            next_level: list[np.ndarray] = []
            for cur_adj in current:
                for edit in _all_possible_edits(cur_adj):
                    new_adj = apply_edit(cur_adj, edit)
                    if acyclic_only and not is_dag(new_adj):
                        continue
                    key = new_adj.tobytes()
                    if key in seen:
                        continue
                    seen.add(key)
                    next_level.append(new_adj)
            counts[d] = len(next_level)
            current = next_level
        return counts

    def lattice_statistics(
        self,
        sequences: Sequence[EditSequence],
    ) -> dict[str, int | float]:
        """Compute summary statistics for the edit lattice.

        Parameters
        ----------
        sequences : Sequence[EditSequence]

        Returns
        -------
        dict[str, int | float]
            Keys: ``n_elements``, ``n_covers``, ``n_components``,
            ``max_chain_length``, ``width``.
        """
        if not sequences:
            return {
                "n_elements": 0,
                "n_covers": 0,
                "n_components": 0,
                "max_chain_length": 0,
                "width": 0,
            }

        diagram = build_hasse(sequences)
        components = self.connected_components(list(sequences))

        # Compute max chain length via longest path in the Hasse diagram
        n = len(diagram.nodes)
        children: dict[int, list[int]] = defaultdict(list)
        for a, b in diagram.covers:
            children[a].append(b)

        # Longest path (DAG, so topological order + DP)
        longest = [1] * n
        # Process in reverse topological order (by size, which is valid)
        for i in range(n - 1, -1, -1):
            for j in children[i]:
                longest[i] = max(longest[i], 1 + longest[j])

        # Width: maximum antichain size (by Dilworth, equals min # chains)
        # Approximate with the largest level set
        size_groups: dict[int, int] = defaultdict(int)
        for es in diagram.nodes:
            size_groups[len(es)] += 1
        width = max(size_groups.values()) if size_groups else 0

        return {
            "n_elements": n,
            "n_covers": len(diagram.covers),
            "n_components": len(components),
            "max_chain_length": max(longest) if longest else 0,
            "width": width,
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _all_possible_edits(adj: np.ndarray) -> list[StructuralEdit]:
    """Generate all structurally possible single-edge edits."""
    n = adj.shape[0]
    edits: list[StructuralEdit] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if adj[i, j]:
                edits.append(StructuralEdit(EditType.DELETE, i, j))
                edits.append(StructuralEdit(EditType.REVERSE, i, j))
            else:
                edits.append(StructuralEdit(EditType.ADD, i, j))
    return edits
