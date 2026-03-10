"""
d-Separation oracle for causal DAGs.

Implements the Bayes-Ball algorithm for answering d-separation queries and
an all-paths enumeration used by the fragility scorer.
"""

from __future__ import annotations

from collections import deque
from itertools import combinations
from typing import Sequence

import numpy as np

from causalcert.types import AdjacencyMatrix, NodeId, NodeSet


def _ancestors_of(adj: np.ndarray, nodes: set[int]) -> set[int]:
    """Return *nodes* plus all their ancestors via reverse BFS."""
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


class DSeparationOracle:
    """Stateless d-separation oracle operating on an adjacency matrix.

    Parameters
    ----------
    adj : AdjacencyMatrix
        Binary adjacency matrix of the DAG.
    """

    def __init__(self, adj: AdjacencyMatrix) -> None:
        self._adj = np.asarray(adj, dtype=np.int8)
        if self._adj.shape[0] != self._adj.shape[1]:
            raise ValueError(f"Adjacency matrix must be square, got shape {self._adj.shape}")
        self._n = self._adj.shape[0]

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the DAG."""
        return self._n

    # ------------------------------------------------------------------
    # Core: Bayes-Ball algorithm
    # ------------------------------------------------------------------

    def _bayes_ball_reachable(
        self,
        source: NodeId,
        conditioning: frozenset[int],
    ) -> set[int]:
        """Return all nodes reachable from *source* via active trails.

        Uses the Bayes-Ball algorithm (Shachter, 1998).  Each state in the
        traversal is ``(node, going_up)`` where *going_up* indicates
        whether the ball is traveling upward (child→parent direction).

        Returns the set of nodes that are d-connected to *source* given
        *conditioning*.
        """
        cond = set(conditioning)
        adj = self._adj

        # visited_up[v]   = ball arrived going UP at v
        # visited_down[v] = ball arrived going DOWN at v
        visited_up: set[int] = set()
        visited_down: set[int] = set()
        reachable: set[int] = set()

        # Queue entries: (node, going_up)
        queue: deque[tuple[int, bool]] = deque()
        # Seed: from source go UP to parents and DOWN to children
        for p in np.nonzero(adj[:, source])[0]:
            queue.append((int(p), True))
        for c in np.nonzero(adj[source, :])[0]:
            queue.append((int(c), False))

        while queue:
            node, going_up = queue.popleft()

            if going_up:
                if node in visited_up:
                    continue
                visited_up.add(node)
                reachable.add(node)
                if node not in cond:
                    # Ball going UP at unconditioned node:
                    # continue UP to parents, bounce DOWN to children
                    for p in np.nonzero(adj[:, node])[0]:
                        p = int(p)
                        if p not in visited_up:
                            queue.append((p, True))
                    for c in np.nonzero(adj[node, :])[0]:
                        c = int(c)
                        if c not in visited_down:
                            queue.append((c, False))
                # If conditioned: ball going UP is blocked (absorbed)

            else:  # going_down
                if node in visited_down:
                    continue
                visited_down.add(node)
                reachable.add(node)
                if node not in cond:
                    # Ball going DOWN at unconditioned node:
                    # continue DOWN to children only
                    for c in np.nonzero(adj[node, :])[0]:
                        c = int(c)
                        if c not in visited_down:
                            queue.append((c, False))
                if node in cond:
                    # Ball going DOWN at conditioned node:
                    # bounce UP to parents (activated collider)
                    for p in np.nonzero(adj[:, node])[0]:
                        p = int(p)
                        if p not in visited_up:
                            queue.append((p, True))

        reachable.discard(source)
        return reachable

    def is_d_separated(
        self,
        x: NodeId,
        y: NodeId,
        conditioning: NodeSet,
    ) -> bool:
        """Test whether *x* ⊥_d *y* | *conditioning* via the Bayes-Ball algorithm.

        Parameters
        ----------
        x : NodeId
            First query node.
        y : NodeId
            Second query node.
        conditioning : NodeSet
            Conditioning set.

        Returns
        -------
        bool
            ``True`` if x and y are d-separated given the conditioning set.
        """
        if x == y:
            return False
        cond = frozenset(conditioning)
        reachable = self._bayes_ball_reachable(x, cond)
        return y not in reachable

    def is_d_connected(
        self,
        x: NodeId,
        y: NodeId,
        conditioning: NodeSet,
    ) -> bool:
        """Test whether *x* and *y* are d-connected given *conditioning*."""
        return not self.is_d_separated(x, y, conditioning)

    def d_connected_set(
        self,
        x: NodeId,
        conditioning: NodeSet,
    ) -> NodeSet:
        """Return all nodes d-connected to *x* given *conditioning*.

        Parameters
        ----------
        x : NodeId
            Source node.
        conditioning : NodeSet
            Conditioning set.

        Returns
        -------
        NodeSet
            Set of nodes reachable via active paths from *x*.
        """
        cond = frozenset(conditioning)
        return frozenset(self._bayes_ball_reachable(x, cond))

    # ------------------------------------------------------------------
    # All CI implications (Markov property)
    # ------------------------------------------------------------------

    def all_ci_implications(
        self,
        max_cond_size: int | None = None,
    ) -> list[tuple[NodeId, NodeId, NodeSet]]:
        """Enumerate all CI relations implied by the DAG (local Markov property).

        For each node X and each non-descendant Y (Y not in desc(X)),
        X ⊥ Y | Pa(X) holds by the local Markov property.

        Parameters
        ----------
        max_cond_size : int | None
            If given, also enumerate d-separations with conditioning sets
            up to this size. If None, returns only the local Markov blanket
            implications.

        Returns
        -------
        list[tuple[NodeId, NodeId, NodeSet]]
            List of ``(x, y, S)`` triples where X ⊥ Y | S.
        """
        results: list[tuple[NodeId, NodeId, NodeSet]] = []
        adj = self._adj

        # Local Markov property: for each node, conditioned on its parents,
        # it is independent of all non-descendants
        for v in range(self._n):
            parents_v = frozenset(int(p) for p in np.nonzero(adj[:, v])[0])
            # Compute descendants of v
            desc_v: set[int] = set()
            queue = deque(int(c) for c in np.nonzero(adj[v, :])[0])
            while queue:
                node = queue.popleft()
                if node not in desc_v:
                    desc_v.add(node)
                    for c in np.nonzero(adj[node, :])[0]:
                        c = int(c)
                        if c not in desc_v:
                            queue.append(c)
            # Non-descendants (excluding v itself and parents which are trivially separated)
            non_desc = set(range(self._n)) - desc_v - {v} - set(parents_v)
            for nd in sorted(non_desc):
                results.append((v, nd, parents_v))

        if max_cond_size is not None:
            # Brute-force additional d-separations for small conditioning sets
            candidate_nodes = list(range(self._n))
            seen: set[tuple[int, int, frozenset[int]]] = set(
                (min(x, y), max(x, y), s) for x, y, s in results
            )
            for i in range(self._n):
                for j in range(i + 1, self._n):
                    others = [k for k in range(self._n) if k != i and k != j]
                    for size in range(min(max_cond_size + 1, len(others) + 1)):
                        for cond in combinations(others, size):
                            cond_set = frozenset(cond)
                            key = (i, j, cond_set)
                            if key in seen:
                                continue
                            if self.is_d_separated(i, j, cond_set):
                                seen.add(key)
                                results.append((i, j, cond_set))

        return results

    # ------------------------------------------------------------------
    # Minimal separating sets
    # ------------------------------------------------------------------

    def all_d_separations(
        self,
        x: NodeId,
        y: NodeId,
        max_size: int | None = None,
    ) -> list[NodeSet]:
        """Enumerate all minimal d-separating sets between *x* and *y*.

        Returns
        -------
        list[NodeSet]
            Each element is a minimal conditioning set that d-separates *x*
            and *y*.  Returns an empty list if *x* and *y* are adjacent.
        """
        if x == y:
            return []
        # If x and y are adjacent, no separating set exists
        if self._adj[x, y] or self._adj[y, x]:
            return []

        n = self._n
        candidates = [k for k in range(n) if k != x and k != y]
        if max_size is None:
            max_size = len(candidates)
        else:
            max_size = min(max_size, len(candidates))

        # Find all separating sets, then filter to minimal
        all_sep: list[frozenset[int]] = []
        for size in range(max_size + 1):
            for comb in combinations(candidates, size):
                cond = frozenset(comb)
                if self.is_d_separated(x, y, cond):
                    all_sep.append(cond)

        # Filter to minimal: a set S is minimal if no proper subset is separating
        minimal: list[NodeSet] = []
        # Sort by size so we check smaller sets first
        all_sep.sort(key=len)
        for s in all_sep:
            is_minimal = True
            for m in minimal:
                if m.issubset(s) and m != s:
                    is_minimal = False
                    break
            if is_minimal:
                minimal.append(s)
        return minimal

    def find_separating_set(
        self,
        x: NodeId,
        y: NodeId,
        max_size: int | None = None,
    ) -> NodeSet | None:
        """Find one d-separating set between *x* and *y*, or None if none exists.

        Tries conditioning sets in order of increasing size.
        """
        if x == y:
            return None
        if self._adj[x, y] or self._adj[y, x]:
            return None
        candidates = [k for k in range(self._n) if k != x and k != y]
        if max_size is None:
            max_size = len(candidates)
        else:
            max_size = min(max_size, len(candidates))
        for size in range(max_size + 1):
            for comb in combinations(candidates, size):
                cond = frozenset(comb)
                if self.is_d_separated(x, y, cond):
                    return cond
        return None

    # ------------------------------------------------------------------
    # Active path enumeration
    # ------------------------------------------------------------------

    def active_paths(
        self,
        x: NodeId,
        y: NodeId,
        conditioning: NodeSet,
    ) -> list[list[NodeId]]:
        """Return all active (d-connected) paths from *x* to *y* given *conditioning*.

        Uses a DFS that tracks the direction of the ball at each node to
        correctly handle colliders and non-colliders. Paths are limited
        to non-repeating nodes to avoid infinite loops.

        Parameters
        ----------
        x, y : NodeId
            Endpoints.
        conditioning : NodeSet
            Conditioning set.

        Returns
        -------
        list[list[NodeId]]
            Each path is a list of node ids from *x* to *y*.
        """
        if x == y:
            return [[x]]

        cond = set(conditioning)
        adj = self._adj
        paths: list[list[NodeId]] = []

        # DFS: state = (current_node, going_up, path_so_far)
        # going_up=True: ball traveling from child to parent
        # going_up=False: ball traveling from parent to child (going down)
        stack: list[tuple[int, bool, list[int]]] = []

        # From x, send ball UP to parents and DOWN to children
        for p in np.nonzero(adj[:, x])[0]:
            stack.append((int(p), True, [x]))
        for c in np.nonzero(adj[x, :])[0]:
            stack.append((int(c), False, [x]))

        while stack:
            node, going_up, path = stack.pop()

            if node in path:
                continue  # avoid cycles in path

            new_path = path + [node]

            if node == y:
                paths.append(new_path)
                continue

            if going_up:
                # Ball going UP at this node
                if node not in cond:
                    # Unconditioned: continue UP to parents, bounce DOWN to children
                    for p in np.nonzero(adj[:, node])[0]:
                        p = int(p)
                        if p not in path:
                            stack.append((p, True, new_path))
                    for c in np.nonzero(adj[node, :])[0]:
                        c = int(c)
                        if c not in path:
                            stack.append((c, False, new_path))
                # If conditioned: ball going UP is blocked
            else:
                # Ball going DOWN at this node
                if node not in cond:
                    # Unconditioned: continue DOWN to children
                    for c in np.nonzero(adj[node, :])[0]:
                        c = int(c)
                        if c not in path:
                            stack.append((c, False, new_path))
                if node in cond:
                    # Conditioned: bounce UP to parents (collider activation)
                    for p in np.nonzero(adj[:, node])[0]:
                        p = int(p)
                        if p not in path:
                            stack.append((p, True, new_path))

        return paths

    # ------------------------------------------------------------------
    # Pairwise d-separation matrix
    # ------------------------------------------------------------------

    def pairwise_dsep_matrix(
        self,
        conditioning: NodeSet,
    ) -> np.ndarray:
        """Compute the full pairwise d-separation matrix for given conditioning.

        Parameters
        ----------
        conditioning : NodeSet
            Conditioning set.

        Returns
        -------
        np.ndarray
            Boolean matrix M where M[i,j] is True if i ⊥_d j | conditioning.
        """
        n = self._n
        result = np.zeros((n, n), dtype=bool)
        cond = frozenset(conditioning)
        for i in range(n):
            reachable = self._bayes_ball_reachable(i, cond)
            for j in range(i + 1, n):
                if j not in reachable:
                    result[i, j] = True
                    result[j, i] = True
        return result

    # ------------------------------------------------------------------
    # Markov blanket
    # ------------------------------------------------------------------

    def markov_blanket(self, v: NodeId) -> NodeSet:
        """Return the Markov blanket of node *v*.

        The Markov blanket consists of: parents, children, and parents
        of children (co-parents / spouses).
        """
        adj = self._adj
        parents = set(int(p) for p in np.nonzero(adj[:, v])[0])
        children = set(int(c) for c in np.nonzero(adj[v, :])[0])
        co_parents: set[int] = set()
        for c in children:
            for p in np.nonzero(adj[:, c])[0]:
                p = int(p)
                if p != v:
                    co_parents.add(p)
        return frozenset(parents | children | co_parents)

    # ------------------------------------------------------------------
    # Convenience: check a list of triples
    # ------------------------------------------------------------------

    def check_triples(
        self,
        triples: Sequence[tuple[NodeId, NodeId, NodeSet]],
    ) -> list[bool]:
        """Check d-separation for a list of ``(x, y, S)`` triples.

        Returns a list of booleans, one per triple.
        """
        return [
            self.is_d_separated(x, y, cond)
            for x, y, cond in triples
        ]
