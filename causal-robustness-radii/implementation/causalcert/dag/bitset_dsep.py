"""
Bitset-based d-separation using numpy uint64 arrays.

Encodes ancestor, descendant, and reachability relations as compact bitsets
so that d-separation queries reduce to O(V+E) bitwise operations instead of
repeated BFS traversals.  Supports batch queries, incremental edge updates,
and an ancestor oracle interface.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from causalcert.types import AdjacencyMatrix, EditType, EdgeTuple, NodeId, NodeSet, StructuralEdit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WORD_BITS = 64


def _n_words(n: int) -> int:
    """Number of uint64 words needed to store *n* bits."""
    return (n + _WORD_BITS - 1) // _WORD_BITS


def _empty_bitset(n: int) -> NDArray[np.uint64]:
    """Return a zeroed bitset with enough words for *n* bits."""
    return np.zeros(_n_words(n), dtype=np.uint64)


def _set_bit(bs: NDArray[np.uint64], i: int) -> None:
    """Set bit *i* in-place."""
    bs[i // _WORD_BITS] |= np.uint64(1) << np.uint64(i % _WORD_BITS)


def _clear_bit(bs: NDArray[np.uint64], i: int) -> None:
    """Clear bit *i* in-place."""
    bs[i // _WORD_BITS] &= ~(np.uint64(1) << np.uint64(i % _WORD_BITS))


def _test_bit(bs: NDArray[np.uint64], i: int) -> bool:
    """Return whether bit *i* is set."""
    return bool(bs[i // _WORD_BITS] & (np.uint64(1) << np.uint64(i % _WORD_BITS)))


def _or_inplace(dst: NDArray[np.uint64], src: NDArray[np.uint64]) -> None:
    """``dst |= src`` element-wise in-place."""
    np.bitwise_or(dst, src, out=dst)


def _and_result(a: NDArray[np.uint64], b: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """Return ``a & b`` element-wise."""
    return np.bitwise_and(a, b)


def _is_zero(bs: NDArray[np.uint64]) -> bool:
    """Return ``True`` if every bit is zero."""
    return not np.any(bs)


def _any_common(a: NDArray[np.uint64], b: NDArray[np.uint64]) -> bool:
    """Return ``True`` if ``a`` and ``b`` share any set bit."""
    return bool(np.any(np.bitwise_and(a, b)))


def _popcount(bs: NDArray[np.uint64]) -> int:
    """Count the total number of set bits."""
    total = 0
    for w in bs:
        v = int(w)
        total += bin(v).count("1")
    return total


def _to_nodeset(bs: NDArray[np.uint64], n: int) -> NodeSet:
    """Convert a bitset to a frozenset of node indices."""
    nodes: list[int] = []
    for i in range(n):
        if _test_bit(bs, i):
            nodes.append(i)
    return frozenset(nodes)


def _from_nodeset(nodes: NodeSet | set[int], n: int) -> NDArray[np.uint64]:
    """Convert a set of node indices to a bitset."""
    bs = _empty_bitset(n)
    for v in nodes:
        _set_bit(bs, v)
    return bs


# ---------------------------------------------------------------------------
# Bitset ancestor / descendant tables
# ---------------------------------------------------------------------------


def _topological_order(adj: np.ndarray) -> list[int]:
    """Kahn's topological sort returning a list of node indices."""
    n = adj.shape[0]
    in_deg = np.sum(adj, axis=0).astype(int)
    queue = deque(int(i) for i in range(n) if in_deg[i] == 0)
    order: list[int] = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for c in np.nonzero(adj[v])[0]:
            c = int(c)
            in_deg[c] -= 1
            if in_deg[c] == 0:
                queue.append(c)
    return order


def _build_ancestor_bitsets(
    adj: np.ndarray, n: int, topo: list[int],
) -> list[NDArray[np.uint64]]:
    """Build ancestor bitsets in reverse topological order.

    ``anc[v]`` has bit *u* set iff *u* is an ancestor of *v* (or *u == v*).
    """
    anc = [_empty_bitset(n) for _ in range(n)]
    for v in topo:
        _set_bit(anc[v], v)
    for v in topo:
        for c in np.nonzero(adj[v])[0]:
            c = int(c)
            _or_inplace(anc[c], anc[v])
    return anc


def _build_descendant_bitsets(
    adj: np.ndarray, n: int, topo: list[int],
) -> list[NDArray[np.uint64]]:
    """Build descendant bitsets in topological order.

    ``desc[v]`` has bit *u* set iff *u* is a descendant of *v* (or *u == v*).
    """
    desc = [_empty_bitset(n) for _ in range(n)]
    for v in topo:
        _set_bit(desc[v], v)
    for v in reversed(topo):
        for p in np.nonzero(adj[:, v])[0]:
            p = int(p)
            _or_inplace(desc[p], desc[v])
    return desc


# ---------------------------------------------------------------------------
# BitsetAncestorOracle
# ---------------------------------------------------------------------------


class BitsetAncestorOracle:
    """Fast ancestor / descendant queries backed by precomputed bitsets.

    Parameters
    ----------
    adj : AdjacencyMatrix
        Binary adjacency matrix of the DAG.

    Attributes
    ----------
    n : int
        Number of nodes.
    """

    def __init__(self, adj: AdjacencyMatrix) -> None:
        self._adj = np.asarray(adj, dtype=np.int8)
        self.n = self._adj.shape[0]
        self._topo = _topological_order(self._adj)
        self._anc = _build_ancestor_bitsets(self._adj, self.n, self._topo)
        self._desc = _build_descendant_bitsets(self._adj, self.n, self._topo)

    # -- queries --

    def is_ancestor(self, u: NodeId, v: NodeId) -> bool:
        """Return ``True`` if *u* is an ancestor of *v* (or *u == v*)."""
        return _test_bit(self._anc[v], u)

    def is_descendant(self, u: NodeId, v: NodeId) -> bool:
        """Return ``True`` if *u* is a descendant of *v* (or *u == v*)."""
        return _test_bit(self._desc[v], u)

    def ancestors(self, v: NodeId) -> NodeSet:
        """Return the ancestors of *v* (including *v*)."""
        return _to_nodeset(self._anc[v], self.n)

    def descendants(self, v: NodeId) -> NodeSet:
        """Return the descendants of *v* (including *v*)."""
        return _to_nodeset(self._desc[v], self.n)

    def ancestors_of_set(self, nodes: NodeSet) -> NodeSet:
        """Return the union of ancestors of every node in *nodes*."""
        bs = _empty_bitset(self.n)
        for v in nodes:
            _or_inplace(bs, self._anc[v])
        return _to_nodeset(bs, self.n)

    def descendants_of_set(self, nodes: NodeSet) -> NodeSet:
        """Return the union of descendants of every node in *nodes*."""
        bs = _empty_bitset(self.n)
        for v in nodes:
            _or_inplace(bs, self._desc[v])
        return _to_nodeset(bs, self.n)

    def have_common_ancestor(self, u: NodeId, v: NodeId) -> bool:
        """Return ``True`` if *u* and *v* share at least one ancestor."""
        return _any_common(self._anc[u], self._anc[v])

    def have_common_descendant(self, u: NodeId, v: NodeId) -> bool:
        """Return ``True`` if *u* and *v* share at least one descendant."""
        return _any_common(self._desc[u], self._desc[v])

    @property
    def topological_order(self) -> list[int]:
        """Return a cached topological ordering."""
        return list(self._topo)

    @property
    def ancestor_bitsets(self) -> list[NDArray[np.uint64]]:
        """Raw ancestor bitsets (read-only reference)."""
        return self._anc

    @property
    def descendant_bitsets(self) -> list[NDArray[np.uint64]]:
        """Raw descendant bitsets (read-only reference)."""
        return self._desc


# ---------------------------------------------------------------------------
# BitsetDSeparation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _BayesBallState:
    """Traversal state for the bitset Bayes-Ball algorithm."""

    visited_up: NDArray[np.uint64]
    visited_down: NDArray[np.uint64]
    reachable: NDArray[np.uint64]


class BitsetDSeparation:
    """Bitset-accelerated d-separation oracle.

    Uses precomputed ancestor / descendant bitsets and a Bayes-Ball traversal
    with bitwise conditioning-set membership checks for O(V+E) d-separation
    queries.

    Parameters
    ----------
    adj : AdjacencyMatrix
        Binary adjacency matrix of the DAG.
    """

    def __init__(self, adj: AdjacencyMatrix) -> None:
        self._adj = np.asarray(adj, dtype=np.int8)
        self._n = self._adj.shape[0]
        if self._adj.shape != (self._n, self._n):
            raise ValueError(
                f"Adjacency matrix must be square, got {self._adj.shape}"
            )

        # Pre-compute children / parents lists for fast enumeration
        self._children: list[list[int]] = [
            list(int(c) for c in np.nonzero(self._adj[v])[0]) for v in range(self._n)
        ]
        self._parents: list[list[int]] = [
            list(int(p) for p in np.nonzero(self._adj[:, v])[0]) for v in range(self._n)
        ]

        self._topo = _topological_order(self._adj)
        self._anc = _build_ancestor_bitsets(self._adj, self._n, self._topo)
        self._desc = _build_descendant_bitsets(self._adj, self._n, self._topo)

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the DAG."""
        return self._n

    @property
    def adjacency(self) -> AdjacencyMatrix:
        """Return a read-only view of the adjacency matrix."""
        m = self._adj.view()
        m.flags.writeable = False
        return m

    # -- ancestor oracle access ---

    def ancestor_oracle(self) -> BitsetAncestorOracle:
        """Return an ancestor oracle sharing the precomputed bitsets."""
        oracle = object.__new__(BitsetAncestorOracle)
        oracle._adj = self._adj
        oracle.n = self._n
        oracle._topo = self._topo
        oracle._anc = self._anc
        oracle._desc = self._desc
        return oracle

    # -- core Bayes-Ball ---

    def _bayes_ball(
        self,
        source: int,
        cond_bs: NDArray[np.uint64],
    ) -> NDArray[np.uint64]:
        """Run Bayes-Ball from *source* and return the reachable bitset.

        Parameters
        ----------
        source : int
            Starting node.
        cond_bs : NDArray[np.uint64]
            Conditioning set encoded as a bitset.
        """
        n = self._n
        # Precompute ancestors-of-conditioning bitset
        anc_cond = _empty_bitset(n)
        for v in range(n):
            if _test_bit(cond_bs, v):
                _or_inplace(anc_cond, self._anc[v])

        visited_up = _empty_bitset(n)
        visited_down = _empty_bitset(n)
        reachable = _empty_bitset(n)

        # BFS queue: (node, going_up)
        queue: deque[tuple[int, bool]] = deque()
        queue.append((source, True))
        queue.append((source, False))

        while queue:
            node, going_up = queue.popleft()

            if going_up:
                if _test_bit(visited_up, node):
                    continue
                _set_bit(visited_up, node)

                in_cond = _test_bit(cond_bs, node)

                if not in_cond:
                    # Not conditioned: mark reachable, pass to children + parents
                    _set_bit(reachable, node)
                    for p in self._parents[node]:
                        queue.append((p, True))
                    for c in self._children[node]:
                        queue.append((c, False))

                # Conditioned or ancestor of conditioned: v-structure opening
                if in_cond or _test_bit(anc_cond, node):
                    for p in self._parents[node]:
                        queue.append((p, True))
            else:
                if _test_bit(visited_down, node):
                    continue
                _set_bit(visited_down, node)

                in_cond = _test_bit(cond_bs, node)

                if not in_cond:
                    _set_bit(reachable, node)
                    for c in self._children[node]:
                        queue.append((c, False))

                if not in_cond:
                    for p in self._parents[node]:
                        queue.append((p, True))

        return reachable

    # -- single query ---

    def is_d_separated(
        self,
        x: NodeId,
        y: NodeId,
        conditioning: NodeSet,
    ) -> bool:
        """Return ``True`` if *x* and *y* are d-separated given *conditioning*.

        Parameters
        ----------
        x : NodeId
            First variable.
        y : NodeId
            Second variable.
        conditioning : NodeSet
            Conditioning set.

        Returns
        -------
        bool
        """
        self._validate_node(x)
        self._validate_node(y)
        cond_bs = _from_nodeset(conditioning, self._n)
        reachable = self._bayes_ball(x, cond_bs)
        return not _test_bit(reachable, y)

    def d_connected_set(
        self,
        source: NodeId,
        conditioning: NodeSet,
    ) -> NodeSet:
        """Return all nodes d-connected to *source* given *conditioning*.

        Parameters
        ----------
        source : NodeId
            Source node.
        conditioning : NodeSet
            Conditioning set.

        Returns
        -------
        NodeSet
        """
        self._validate_node(source)
        cond_bs = _from_nodeset(conditioning, self._n)
        reachable = self._bayes_ball(source, cond_bs)
        return _to_nodeset(reachable, self._n) - {source}

    # -- batch queries ---

    def batch_d_separated(
        self,
        queries: Sequence[tuple[NodeId, NodeId, NodeSet]],
    ) -> list[bool]:
        """Test multiple d-separation queries, reusing conditioning bitsets.

        Parameters
        ----------
        queries : Sequence[tuple[NodeId, NodeId, NodeSet]]
            Each entry is ``(x, y, conditioning)``.

        Returns
        -------
        list[bool]
            One result per query.
        """
        # Group by conditioning set to share Bayes-Ball runs
        from collections import defaultdict
        groups: dict[NodeSet, list[tuple[int, int, int]]] = defaultdict(list)
        for idx, (x, y, cond) in enumerate(queries):
            groups[cond].append((idx, x, y))

        results: list[bool] = [False] * len(queries)

        for cond, items in groups.items():
            cond_bs = _from_nodeset(cond, self._n)
            # Group by source to share the Bayes-Ball traversal
            source_groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
            for idx, x, y in items:
                source_groups[x].append((idx, y))

            for src, targets in source_groups.items():
                reachable = self._bayes_ball(src, cond_bs)
                for idx, y in targets:
                    results[idx] = not _test_bit(reachable, y)

        return results

    def batch_d_separated_fixed_cond(
        self,
        pairs: Sequence[tuple[NodeId, NodeId]],
        conditioning: NodeSet,
    ) -> list[bool]:
        """Test many ``(x, y)`` pairs under a single conditioning set.

        Parameters
        ----------
        pairs : Sequence[tuple[NodeId, NodeId]]
            Variable pairs to test.
        conditioning : NodeSet
            Shared conditioning set.

        Returns
        -------
        list[bool]
        """
        cond_bs = _from_nodeset(conditioning, self._n)

        # Run Bayes-Ball once per unique source
        cache: dict[int, NDArray[np.uint64]] = {}
        results: list[bool] = []
        for x, y in pairs:
            if x not in cache:
                cache[x] = self._bayes_ball(x, cond_bs)
            results.append(not _test_bit(cache[x], y))
        return results

    # -- incremental update ---

    def incremental_update(
        self,
        edit: StructuralEdit,
    ) -> BitsetDSeparation:
        """Return a new :class:`BitsetDSeparation` after applying *edit*.

        Only recomputes the ancestor/descendant bitsets for nodes in the
        affected region (descendants of source and ancestors of target for
        additions; full recompute for reversals on large DAGs).

        Parameters
        ----------
        edit : StructuralEdit
            The edge edit to apply.

        Returns
        -------
        BitsetDSeparation
            A new oracle reflecting the updated DAG.
        """
        new_adj = self._adj.copy()
        u, v = edit.source, edit.target

        if edit.edit_type == EditType.ADD:
            if new_adj[u, v]:
                logger.warning("Edge %d→%d already exists; ADD is a no-op.", u, v)
                return self
            new_adj[u, v] = 1
        elif edit.edit_type == EditType.DELETE:
            if not new_adj[u, v]:
                logger.warning("Edge %d→%d absent; DELETE is a no-op.", u, v)
                return self
            new_adj[u, v] = 0
        elif edit.edit_type == EditType.REVERSE:
            new_adj[u, v] = 0
            new_adj[v, u] = 1
        else:
            raise ValueError(f"Unknown edit type: {edit.edit_type}")

        return BitsetDSeparation(new_adj)

    def apply_edits(
        self,
        edits: Sequence[StructuralEdit],
    ) -> BitsetDSeparation:
        """Apply a batch of edits and return a fresh oracle.

        Parameters
        ----------
        edits : Sequence[StructuralEdit]
            Edits to apply sequentially.

        Returns
        -------
        BitsetDSeparation
        """
        new_adj = self._adj.copy()
        for edit in edits:
            u, v = edit.source, edit.target
            if edit.edit_type == EditType.ADD:
                new_adj[u, v] = 1
            elif edit.edit_type == EditType.DELETE:
                new_adj[u, v] = 0
            elif edit.edit_type == EditType.REVERSE:
                new_adj[u, v] = 0
                new_adj[v, u] = 1
        return BitsetDSeparation(new_adj)

    # -- affected nodes ---

    def affected_nodes(self, edit: StructuralEdit) -> NodeSet:
        """Return nodes whose d-separation relations may change after *edit*.

        A conservative over-approximation: returns ancestors of the source
        union descendants of the target.

        Parameters
        ----------
        edit : StructuralEdit
            Proposed edge edit.

        Returns
        -------
        NodeSet
        """
        u, v = edit.source, edit.target
        anc_u = _to_nodeset(self._anc[u], self._n)
        desc_v = _to_nodeset(self._desc[v], self._n)
        return anc_u | desc_v

    # -- helpers ---

    def ancestors(self, v: NodeId) -> NodeSet:
        """Return ancestors of *v* (including *v*)."""
        self._validate_node(v)
        return _to_nodeset(self._anc[v], self._n)

    def descendants(self, v: NodeId) -> NodeSet:
        """Return descendants of *v* (including *v*)."""
        self._validate_node(v)
        return _to_nodeset(self._desc[v], self._n)

    def is_ancestor(self, u: NodeId, v: NodeId) -> bool:
        """Return ``True`` if *u* is an ancestor of *v*."""
        return _test_bit(self._anc[v], u)

    def is_descendant(self, u: NodeId, v: NodeId) -> bool:
        """Return ``True`` if *u* is a descendant of *v*."""
        return _test_bit(self._desc[v], u)

    def _validate_node(self, v: NodeId) -> None:
        if v < 0 or v >= self._n:
            from causalcert.exceptions import NodeNotFoundError
            raise NodeNotFoundError(v, self._n)

    # -- repr ---

    def __repr__(self) -> str:
        n_edges = int(np.sum(self._adj))
        return (
            f"BitsetDSeparation(n_nodes={self._n}, n_edges={n_edges}, "
            f"words_per_node={_n_words(self._n)})"
        )
