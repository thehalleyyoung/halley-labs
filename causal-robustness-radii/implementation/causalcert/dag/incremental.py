"""
Incremental d-separation under single-edge edits (ALG 2 / Theorem 6).

After a single edge addition, deletion, or reversal, only a bounded subset
of d-separation relations can change.  This module computes exactly which
``(x, y, S)`` triples are *affected* without re-running Bayes-Ball from
scratch, yielding amortised speed-ups during the fragility-scoring loop.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from causalcert.types import (
    AdjacencyMatrix,
    EditType,
    NodeId,
    NodeSet,
    StructuralEdit,
)
from causalcert.dag.dsep import DSeparationOracle


def _ancestors_set(adj: np.ndarray, nodes: set[int]) -> set[int]:
    """Return *nodes* ∪ ancestors(*nodes*) by reverse BFS."""
    result = set(nodes)
    queue = deque(nodes)
    while queue:
        v = queue.popleft()
        for p in np.nonzero(adj[:, v])[0]:
            p = int(p)
            if p not in result:
                result.add(p)
                queue.append(p)
    return result


def _descendants_set(adj: np.ndarray, nodes: set[int]) -> set[int]:
    """Return *nodes* ∪ descendants(*nodes*) by forward BFS."""
    result = set(nodes)
    queue = deque(nodes)
    while queue:
        v = queue.popleft()
        for c in np.nonzero(adj[v, :])[0]:
            c = int(c)
            if c not in result:
                result.add(c)
                queue.append(c)
    return result


@dataclass(frozen=True, slots=True)
class AffectedSeparation:
    """A d-separation relation that may have changed after an edit.

    Attributes
    ----------
    x : NodeId
        First endpoint.
    y : NodeId
        Second endpoint.
    conditioning : NodeSet
        Conditioning set.
    was_separated : bool
        Status before the edit.
    is_separated : bool
        Status after the edit.
    """

    x: NodeId
    y: NodeId
    conditioning: NodeSet
    was_separated: bool
    is_separated: bool

    @property
    def changed(self) -> bool:
        """Whether the d-separation status actually flipped."""
        return self.was_separated != self.is_separated


def _default_watched_triples(n: int) -> list[tuple[NodeId, NodeId, NodeSet]]:
    """Generate all pairwise triples with empty conditioning set."""
    triples: list[tuple[NodeId, NodeId, NodeSet]] = []
    for i in range(n):
        for j in range(i + 1, n):
            triples.append((i, j, frozenset()))
    return triples


def _compute_affected_nodes(
    adj: np.ndarray,
    source: NodeId,
    target: NodeId,
) -> set[int]:
    """Compute the set of nodes whose d-separation relations could change.

    Per Theorem 6: when edge (u, v) is modified, only CI relations involving
    nodes in An(u) ∪ An(v) ∪ Desc(u) ∪ Desc(v) can change.
    """
    n = adj.shape[0]
    an_u = _ancestors_set(adj, {source})
    an_v = _ancestors_set(adj, {target})
    desc_u = _descendants_set(adj, {source})
    desc_v = _descendants_set(adj, {target})
    return an_u | an_v | desc_u | desc_v


class IncrementalDSep:
    """Incremental d-separation maintainer for single-edge edits.

    Maintains the d-separation status of a configurable set of queries and
    updates them efficiently after each edit, in accordance with Theorem 6:
    only triples whose active-path structure intersects the edited edge
    need re-evaluation.

    Parameters
    ----------
    adj : AdjacencyMatrix
        Initial adjacency matrix.
    watched_triples : Sequence[tuple[NodeId, NodeId, NodeSet]] | None
        Specific triples to track.  ``None`` means track all pairwise
        with the empty conditioning set (useful for small DAGs).
    """

    def __init__(
        self,
        adj: AdjacencyMatrix,
        watched_triples: Sequence[tuple[NodeId, NodeId, NodeSet]] | None = None,
    ) -> None:
        self._adj = np.asarray(adj, dtype=np.int8).copy()
        self._n = self._adj.shape[0]
        if watched_triples is not None:
            self._watched: list[tuple[NodeId, NodeId, NodeSet]] = list(watched_triples)
        else:
            self._watched = _default_watched_triples(self._n)
        self._cache: dict[tuple[NodeId, NodeId, NodeSet], bool] = {}
        # Initialize cache with current d-separation status
        self._rebuild_cache()

    def _rebuild_cache(self) -> None:
        """Recompute d-separation status for all watched triples."""
        oracle = DSeparationOracle(self._adj)
        self._cache.clear()
        for x, y, cond in self._watched:
            self._cache[(x, y, cond)] = oracle.is_d_separated(x, y, cond)

    def _apply_edit_to_adj(self, edit: StructuralEdit) -> None:
        """Apply an edit directly to the internal adjacency matrix."""
        if edit.edit_type == EditType.ADD:
            self._adj[edit.source, edit.target] = 1
        elif edit.edit_type == EditType.DELETE:
            self._adj[edit.source, edit.target] = 0
        elif edit.edit_type == EditType.REVERSE:
            self._adj[edit.source, edit.target] = 0
            self._adj[edit.target, edit.source] = 1

    def _identify_affected_triples(
        self,
        source: NodeId,
        target: NodeId,
    ) -> list[tuple[NodeId, NodeId, NodeSet]]:
        """Identify watched triples that might be affected by editing (source, target).

        Per Theorem 6: a triple (x, y, S) can only change if {x, y} ∪ S intersects
        the set of ancestors/descendants of the edit endpoints.
        """
        affected_nodes = _compute_affected_nodes(self._adj, source, target)
        affected: list[tuple[NodeId, NodeId, NodeSet]] = []
        for x, y, cond in self._watched:
            # A triple is potentially affected if either endpoint or any
            # conditioning variable is in the affected region
            triple_nodes = {x, y} | set(cond)
            if triple_nodes & affected_nodes:
                affected.append((x, y, cond))
        return affected

    def update(self, edit: StructuralEdit) -> list[AffectedSeparation]:
        """Apply *edit* and return all affected d-separation relations.

        Parameters
        ----------
        edit : StructuralEdit
            The edge edit to apply.

        Returns
        -------
        list[AffectedSeparation]
            Triples whose d-separation status may have changed.  Only triples
            for which ``changed`` is ``True`` are guaranteed to have flipped.
        """
        # Step 1: identify potentially affected triples BEFORE the edit
        affected_triples = self._identify_affected_triples(
            edit.source, edit.target
        )

        # Store old statuses
        old_status: dict[tuple[NodeId, NodeId, NodeSet], bool] = {}
        for triple in affected_triples:
            old_status[triple] = self._cache.get(triple, False)

        # Step 2: apply the edit
        self._apply_edit_to_adj(edit)

        # Step 3: recompute only affected triples
        oracle = DSeparationOracle(self._adj)
        results: list[AffectedSeparation] = []
        for triple in affected_triples:
            x, y, cond = triple
            was_sep = old_status[triple]
            is_sep = oracle.is_d_separated(x, y, cond)
            self._cache[triple] = is_sep
            results.append(AffectedSeparation(
                x=x,
                y=y,
                conditioning=cond,
                was_separated=was_sep,
                is_separated=is_sep,
            ))

        return results

    def batch_update(
        self, edits: Sequence[StructuralEdit]
    ) -> list[AffectedSeparation]:
        """Apply a sequence of edits and return the cumulative affected relations.

        Parameters
        ----------
        edits : Sequence[StructuralEdit]
            Edits to apply in order.

        Returns
        -------
        list[AffectedSeparation]
            Union of all affected separations. Status reflects original
            (before first edit) vs. final (after all edits).
        """
        if not edits:
            return []

        # Snapshot original status for all watched triples
        original_status = dict(self._cache)

        # Apply all edits sequentially
        all_affected_keys: set[tuple[NodeId, NodeId, NodeSet]] = set()
        for edit in edits:
            affected = self._identify_affected_triples(edit.source, edit.target)
            all_affected_keys.update(
                (x, y, cond) for x, y, cond in affected
            )
            self._apply_edit_to_adj(edit)

        # Recompute all affected triples under the final graph
        oracle = DSeparationOracle(self._adj)
        results: list[AffectedSeparation] = []
        for triple in all_affected_keys:
            x, y, cond = triple
            was_sep = original_status.get(triple, False)
            is_sep = oracle.is_d_separated(x, y, cond)
            self._cache[triple] = is_sep
            results.append(AffectedSeparation(
                x=x,
                y=y,
                conditioning=cond,
                was_separated=was_sep,
                is_separated=is_sep,
            ))

        return results

    def query(self, x: NodeId, y: NodeId, conditioning: NodeSet) -> bool:
        """Return current d-separation status (cache-backed).

        Parameters
        ----------
        x, y : NodeId
            Query endpoints.
        conditioning : NodeSet
            Conditioning set.

        Returns
        -------
        bool
            ``True`` if x ⊥_d y | conditioning in the current graph.
        """
        key = (x, y, frozenset(conditioning))
        if key in self._cache:
            return self._cache[key]
        # Not in cache; compute and store
        oracle = DSeparationOracle(self._adj)
        result = oracle.is_d_separated(x, y, frozenset(conditioning))
        self._cache[key] = result
        return result

    def invalidate_cache(self) -> None:
        """Force full cache invalidation. Next query re-computes from scratch."""
        self._cache.clear()
        self._rebuild_cache()

    def add_watched_triple(
        self, x: NodeId, y: NodeId, conditioning: NodeSet
    ) -> None:
        """Add a new triple to the watched set."""
        triple = (x, y, frozenset(conditioning))
        if triple not in [(wx, wy, wc) for wx, wy, wc in self._watched]:
            self._watched.append(triple)
            oracle = DSeparationOracle(self._adj)
            self._cache[triple] = oracle.is_d_separated(x, y, frozenset(conditioning))

    def remove_watched_triple(
        self, x: NodeId, y: NodeId, conditioning: NodeSet
    ) -> None:
        """Remove a triple from the watched set."""
        triple = (x, y, frozenset(conditioning))
        self._watched = [
            t for t in self._watched if t != triple
        ]
        self._cache.pop(triple, None)

    def changed_triples(
        self, edit: StructuralEdit
    ) -> list[tuple[NodeId, NodeId, NodeSet]]:
        """Return triples that *actually* changed status under the edit.

        This is a peek operation — does NOT modify internal state.
        """
        # Compute on a copy
        adj_copy = self._adj.copy()
        if edit.edit_type == EditType.ADD:
            adj_copy[edit.source, edit.target] = 1
        elif edit.edit_type == EditType.DELETE:
            adj_copy[edit.source, edit.target] = 0
        elif edit.edit_type == EditType.REVERSE:
            adj_copy[edit.source, edit.target] = 0
            adj_copy[edit.target, edit.source] = 1

        affected_nodes = _compute_affected_nodes(self._adj, edit.source, edit.target)
        new_oracle = DSeparationOracle(adj_copy)
        changed: list[tuple[NodeId, NodeId, NodeSet]] = []

        for x, y, cond in self._watched:
            triple_nodes = {x, y} | set(cond)
            if not (triple_nodes & affected_nodes):
                continue
            old_val = self._cache.get((x, y, cond))
            if old_val is None:
                continue
            new_val = new_oracle.is_d_separated(x, y, cond)
            if old_val != new_val:
                changed.append((x, y, cond))

        return changed

    @property
    def adj(self) -> AdjacencyMatrix:
        """Current adjacency matrix (read-only view)."""
        view = self._adj.view()
        view.flags.writeable = False
        return view

    @property
    def n_watched(self) -> int:
        """Number of currently watched triples."""
        return len(self._watched)

    @property
    def cache_size(self) -> int:
        """Number of entries in the d-separation cache."""
        return len(self._cache)
