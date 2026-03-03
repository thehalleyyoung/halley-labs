"""Enumeration and sampling of DAGs within a Markov Equivalence Class.

This module provides :class:`MECEnumerator`, a toolkit for enumerating,
counting, and sampling DAGs from a Markov Equivalence Class (MEC) given
its CPDAG representation.  It also supports enumerating all distinct MECs
on a small number of nodes.

An undirected edge in a CPDAG represents a *reversible* edge whose
orientation may differ across Markov-equivalent DAGs.  The enumerator
explores valid orientations of these reversible edges while maintaining
acyclicity, using DFS with constraint propagation (Meek rules) for
efficient pruning.

Key features
------------
- **Exact enumeration** via recursive DFS with early cycle pruning.
- **Constraint propagation** using Meek's orientation rules R1–R3 to
  reduce the effective branching factor.
- **Uniform sampling** for small MECs (enumerate-then-sample) and
  rejection sampling for larger ones.
- **MEC enumeration** over all DAGs on *n* nodes (feasible for *n* ≤ 5).
"""

from __future__ import annotations

from collections import deque
from itertools import product
from typing import Iterator, List

import numpy as np

from causal_qd.core.dag import DAG
from causal_qd.types import AdjacencyMatrix

# Known MEC counts for small numbers of nodes (OEIS A003024-derived).
_KNOWN_MEC_COUNTS = {1: 1, 2: 2, 3: 11, 4: 185, 5: 8782}


class MECEnumerator:
    """Enumerate, count, or sample DAGs from a MEC given its CPDAG.

    An undirected edge in the CPDAG represents a reversible edge whose
    orientation may vary across equivalent DAGs.  This class explores all
    valid orientations of the reversible edges that maintain acyclicity.

    The implementation uses a recursive DFS strategy with two key
    optimisations:

    1. **Early pruning** – after each edge orientation, partial acyclicity
       is checked using only the directed portion of the current matrix,
       avoiding the cost of checking the full ``2^k`` orientations.
    2. **Meek-rule propagation** – after orienting an edge, the three
       Meek orientation rules (R1, R2, R3) are applied to determine
       whether any remaining undirected edges are forced.  This can
       dramatically reduce the search space.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enumerate(self, cpdag: AdjacencyMatrix) -> Iterator[DAG]:
        """Yield every DAG in the MEC represented by *cpdag*.

        The CPDAG is first decomposed into compelled directed edges and
        a list of undirected (reversible) edge pairs.  A recursive DFS
        explores every valid orientation of the undirected edges, pruning
        branches that would introduce directed cycles.

        After each edge orientation, Meek's rules R1–R3 are applied to
        propagate forced orientations, reducing the search space.

        Parameters
        ----------
        cpdag : AdjacencyMatrix
            CPDAG adjacency matrix where mutual edges (``cpdag[i,j]==1``
            and ``cpdag[j,i]==1``) denote undirected / reversible edges.

        Yields
        ------
        DAG
            Each Markov-equivalent DAG exactly once.
        """
        cpdag = np.asarray(cpdag, dtype=np.int8)
        directed, undirected_pairs = self._decompose(cpdag)

        if not undirected_pairs:
            yield DAG(directed)
            return

        yield from self._enumerate_dags_recursive(
            directed, undirected_pairs, 0, directed.copy()
        )

    def count(self, cpdag: AdjacencyMatrix) -> int:
        """Count the number of DAGs in the MEC represented by *cpdag*.

        For CPDAGs with a small number of undirected edges the method
        performs exact enumeration.  For larger CPDAGs a chain-component
        decomposition is used: the undirected edges form connected
        components (chain components) and the MEC size is the product of
        per-component sizes, enabling independent enumeration.

        Parameters
        ----------
        cpdag : AdjacencyMatrix
            CPDAG adjacency matrix.

        Returns
        -------
        int
            Number of DAGs in the MEC.
        """
        cpdag = np.asarray(cpdag, dtype=np.int8)
        directed, undirected_pairs = self._decompose(cpdag)

        if not undirected_pairs:
            return 1

        # Chain-component decomposition: partition undirected edges into
        # connected components based on shared endpoints, then multiply
        # per-component counts.
        components = self._chain_components(undirected_pairs)

        if len(components) <= 1:
            # Single component – enumerate directly.
            return sum(1 for _ in self.enumerate(cpdag))

        total = 1
        n = cpdag.shape[0]
        for comp_pairs in components:
            # Build a sub-CPDAG restricted to the nodes in this component
            # plus the directed edges among those nodes.
            comp_nodes = set()
            for i, j in comp_pairs:
                comp_nodes.add(i)
                comp_nodes.add(j)

            sub_cpdag = np.zeros((n, n), dtype=np.int8)
            # Copy directed edges that connect component nodes.
            for node in comp_nodes:
                for other in range(n):
                    if directed[node, other]:
                        sub_cpdag[node, other] = 1
                    if directed[other, node]:
                        sub_cpdag[other, node] = 1
            # Add undirected edges for this component.
            for i, j in comp_pairs:
                sub_cpdag[i, j] = 1
                sub_cpdag[j, i] = 1

            sub_directed, sub_undirected = self._decompose(sub_cpdag)
            comp_count = sum(
                1
                for _ in self._enumerate_dags_recursive(
                    sub_directed, sub_undirected, 0, sub_directed.copy()
                )
            )
            total *= comp_count

        return total

    def sample(
        self,
        cpdag: AdjacencyMatrix,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> List[DAG]:
        """Uniformly sample *n* DAGs from the MEC.

        For small MECs (≤ 15 undirected edges) the method enumerates all
        members and samples with replacement.  For larger MECs it uses
        rejection sampling: random orientations of the undirected edges
        are generated and accepted only if the resulting graph is acyclic.

        Parameters
        ----------
        cpdag : AdjacencyMatrix
            CPDAG adjacency matrix.
        n : int
            Number of DAG samples to draw.
        rng : numpy.random.Generator or None
            Random number generator.  If ``None``, a fresh default
            generator is created.

        Returns
        -------
        List[DAG]
            Up to *n* sampled DAGs.  Fewer may be returned if rejection
            sampling exhausts its attempt budget.
        """
        if rng is None:
            rng = np.random.default_rng()

        cpdag = np.asarray(cpdag, dtype=np.int8)
        directed, undirected_pairs = self._decompose(cpdag)
        k = len(undirected_pairs)

        # If few reversible edges, enumerate and sample.
        if k <= 15:
            all_dags = list(self.enumerate(cpdag))
            if not all_dags:
                return []
            indices = rng.integers(0, len(all_dags), size=n)
            return [all_dags[i] for i in indices]

        # Rejection sampling for larger MECs.
        samples: list[DAG] = []
        max_attempts = n * 100
        attempts = 0
        while len(samples) < n and attempts < max_attempts:
            bits = rng.integers(0, 2, size=k)
            candidate = directed.copy()
            for bit, (i, j) in zip(bits, undirected_pairs):
                if bit == 0:
                    candidate[i, j] = 1
                else:
                    candidate[j, i] = 1
            if DAG.is_acyclic(candidate):
                samples.append(DAG(candidate))
            attempts += 1
        return samples

    def enumerate_all_mecs(self, n: int) -> Iterator[AdjacencyMatrix]:
        """Enumerate all distinct MECs (CPDAGs) on *n* nodes.

        For small *n* (≤ 5), generates every possible DAG on *n* nodes,
        converts each to its CPDAG, and yields unique CPDAGs.  Larger
        values of *n* are infeasible and raise ``ValueError``.

        Parameters
        ----------
        n : int
            Number of nodes.  Must be ≤ 5 for tractability.

        Yields
        ------
        AdjacencyMatrix
            Each unique CPDAG (representing one MEC) exactly once.

        Raises
        ------
        ValueError
            If *n* > 5 (enumeration is computationally infeasible).
        """
        if n > 5:
            raise ValueError(
                f"Enumerating all MECs on {n} nodes is infeasible (n must be ≤ 5)."
            )
        if n <= 0:
            return

        seen: set[bytes] = set()

        # Generate all possible DAGs on n nodes by trying every subset
        # of the n*(n-1)/2 possible forward edges under each permutation
        # ordering.  Equivalently, iterate over all binary upper-
        # triangular matrices for every node permutation.
        for perm in _permutations(n):
            max_forward = n * (n - 1) // 2
            # Collect all forward pairs under this permutation.
            forward_pairs: list[tuple[int, int]] = []
            for a in range(n):
                for b in range(a + 1, n):
                    forward_pairs.append((perm[a], perm[b]))

            for bits in product([0, 1], repeat=len(forward_pairs)):
                adj: AdjacencyMatrix = np.zeros((n, n), dtype=np.int8)
                for bit, (src, tgt) in zip(bits, forward_pairs):
                    if bit:
                        adj[src, tgt] = 1

                # Verify acyclicity (guaranteed by construction for a
                # single permutation, but duplicates exist).
                if not DAG.is_acyclic(adj):
                    continue

                dag = DAG(adj)
                cpdag = dag.to_cpdag()
                key = cpdag.tobytes()
                if key not in seen:
                    seen.add(key)
                    yield cpdag

    def count_mecs(self, n: int) -> int:
        """Count the total number of distinct MECs on *n* nodes.

        For *n* ≤ 5, known exact values are returned directly.  For
        other small *n*, a full enumeration is performed.

        Known values (OEIS A003024-derived):
            n=1 → 1, n=2 → 2, n=3 → 11, n=4 → 185, n=5 → 8782

        Parameters
        ----------
        n : int
            Number of nodes.

        Returns
        -------
        int
            Total number of distinct MECs.

        Raises
        ------
        ValueError
            If *n* > 5 and no known value exists.
        """
        if n in _KNOWN_MEC_COUNTS:
            return _KNOWN_MEC_COUNTS[n]
        if n <= 0:
            return 0
        # Fall back to enumeration for unknown small n.
        return sum(1 for _ in self.enumerate_all_mecs(n))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _decompose(
        cpdag: AdjacencyMatrix,
    ) -> tuple[AdjacencyMatrix, list[tuple[int, int]]]:
        """Split a CPDAG into directed edges and undirected edge pairs.

        An edge pair ``(i, j)`` with ``i < j`` is classified as:
        - **undirected** if both ``cpdag[i,j]`` and ``cpdag[j,i]`` are 1.
        - **directed i → j** if only ``cpdag[i,j]`` is 1.
        - **directed j → i** if only ``cpdag[j,i]`` is 1.

        Parameters
        ----------
        cpdag : AdjacencyMatrix
            CPDAG adjacency matrix.

        Returns
        -------
        directed : AdjacencyMatrix
            Matrix containing only the compelled directed edges.
        undirected_pairs : list of (i, j)
            Each undirected edge listed once with ``i < j``.
        """
        n = cpdag.shape[0]
        directed = np.zeros((n, n), dtype=np.int8)
        undirected_pairs: list[tuple[int, int]] = []

        for i in range(n):
            for j in range(i + 1, n):
                ij = cpdag[i, j]
                ji = cpdag[j, i]
                if ij and ji:
                    undirected_pairs.append((i, j))
                elif ij:
                    directed[i, j] = 1
                elif ji:
                    directed[j, i] = 1

        return directed, undirected_pairs

    def _enumerate_dags_recursive(
        self,
        directed: AdjacencyMatrix,
        undirected_pairs: list[tuple[int, int]],
        idx: int,
        current: AdjacencyMatrix,
    ) -> Iterator[DAG]:
        """Recursively enumerate valid orientations via DFS with pruning.

        At each recursion level the undirected edge at position *idx* is
        oriented in both possible directions.  After orienting, Meek
        rules are applied to propagate forced orientations for remaining
        edges.  The branch is pruned if the partial orientation contains
        a directed cycle.

        Parameters
        ----------
        directed : AdjacencyMatrix
            The compelled (fixed) directed edges.
        undirected_pairs : list of (i, j)
            All undirected edge pairs (``i < j``).
        idx : int
            Index into *undirected_pairs* indicating the next edge to
            orient.
        current : AdjacencyMatrix
            The current (partially oriented) adjacency matrix.

        Yields
        ------
        DAG
            Each valid fully-oriented DAG.
        """
        n = current.shape[0]

        # Base case: all undirected edges have been oriented.
        if idx >= len(undirected_pairs):
            if self._is_partial_acyclic(current, n):
                yield DAG(current.copy())
            return

        i, j = undirected_pairs[idx]

        # Try orientation i → j
        candidate_fwd = current.copy()
        candidate_fwd[i, j] = 1
        candidate_fwd[j, i] = 0

        # Apply Meek-rule propagation on remaining undirected edges.
        remaining = undirected_pairs[idx + 1 :]
        fwd_remaining = self._apply_meek_rules(candidate_fwd, remaining, n)

        if self._is_partial_acyclic(candidate_fwd, n):
            yield from self._enumerate_dags_recursive(
                directed, undirected_pairs[:idx + 1] + fwd_remaining,
                idx + 1, candidate_fwd,
            )

        # Try orientation j → i
        candidate_rev = current.copy()
        candidate_rev[j, i] = 1
        candidate_rev[i, j] = 0

        rev_remaining = self._apply_meek_rules(candidate_rev, remaining, n)

        if self._is_partial_acyclic(candidate_rev, n):
            yield from self._enumerate_dags_recursive(
                directed, undirected_pairs[:idx + 1] + rev_remaining,
                idx + 1, candidate_rev,
            )

    @staticmethod
    def _is_partial_acyclic(adj: AdjacencyMatrix, n: int) -> bool:
        """Check whether the directed edges in *adj* contain a cycle.

        Only considers entries where ``adj[i,j]==1`` and ``adj[j,i]==0``
        (i.e. truly directed edges).  Undirected pairs (mutual 1s) are
        ignored so that partially oriented graphs can be checked before
        all edges are resolved.

        Uses Kahn's algorithm (iterative topological sort) restricted to
        directed edges.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Possibly partially oriented adjacency matrix.
        n : int
            Number of nodes.

        Returns
        -------
        bool
            ``True`` if the directed portion is acyclic.
        """
        # Build a directed-only view: edge i→j exists iff adj[i,j]==1
        # and adj[j,i]==0.
        dir_adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(n):
                if adj[i, j] and not adj[j, i]:
                    dir_adj[i, j] = 1

        in_degree = dir_adj.sum(axis=0).copy()
        queue = deque(int(i) for i in range(n) if in_degree[i] == 0)
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for child in range(n):
                if dir_adj[node, child]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
        return visited == n

    @staticmethod
    def _apply_meek_rules(
        adj: AdjacencyMatrix,
        remaining: list[tuple[int, int]],
        n: int,
    ) -> list[tuple[int, int]]:
        """Apply Meek orientation rules R1–R3 to propagate forced edges.

        After orienting one edge, some remaining undirected edges may be
        forced to a particular direction.  This method iteratively
        applies three of Meek's four rules until no more edges can be
        oriented:

        **R1 (chain rule):** If *a → b — c* and *a* is not adjacent to
        *c*, orient *b → c* (to avoid a new v-structure).

        **R2 (acyclicity):** If *a → b → c* and *a — c*, orient
        *a → c* (to avoid a directed cycle *a → b → c → a*).

        **R3 (double non-adjacent parent):** If *a — c*, *b — c*,
        *a → d*, *b → d*, and *a*, *b* are not adjacent, orient
        *c → d* is not directly applicable in standard form; instead
        we use: if *a — c*, *d → a*, *d → c* doesn't exist, and
        *d — c* exists, check for the R3 diamond pattern.

        Parameters
        ----------
        adj : AdjacencyMatrix
            The current adjacency matrix (modified in place).
        remaining : list of (i, j)
            Undirected edges still to be resolved.
        n : int
            Number of nodes.

        Returns
        -------
        list of (i, j)
            Updated list of still-undirected edge pairs.
        """
        changed = True
        while changed:
            changed = False
            still_undirected: list[tuple[int, int]] = []
            for i, j in remaining:
                # Check if this edge was already oriented by a prior rule.
                if not (adj[i, j] and adj[j, i]):
                    still_undirected.append((i, j))
                    continue

                oriented = False

                # --- R1: chain rule ---
                # If ∃ k such that k → i (directed) and k not adj to j,
                # then orient i → j.
                for k in range(n):
                    if k == i or k == j:
                        continue
                    # k → i directed (k→i and not i→k)
                    if adj[k, i] and not adj[i, k]:
                        if not adj[k, j] and not adj[j, k]:
                            adj[i, j] = 1
                            adj[j, i] = 0
                            oriented = True
                            changed = True
                            break
                    # k → j directed and k not adj to i ⇒ orient j → i
                    if adj[k, j] and not adj[j, k]:
                        if not adj[k, i] and not adj[i, k]:
                            adj[j, i] = 1
                            adj[i, j] = 0
                            oriented = True
                            changed = True
                            break

                if oriented:
                    continue

                # --- R2: acyclicity rule ---
                # If ∃ k such that i → k → j (both directed), orient i → j.
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if (adj[i, k] and not adj[k, i]) and (adj[k, j] and not adj[j, k]):
                        adj[i, j] = 1
                        adj[j, i] = 0
                        oriented = True
                        changed = True
                        break
                    if (adj[j, k] and not adj[k, j]) and (adj[k, i] and not adj[i, k]):
                        adj[j, i] = 1
                        adj[i, j] = 0
                        oriented = True
                        changed = True
                        break

                if oriented:
                    continue

                # --- R3: two non-adjacent parents rule ---
                # If i — j, and ∃ k1, k2 both adjacent to i undirected,
                # k1 → j directed, k2 → j directed, and k1 not adj k2,
                # then orient i → j.
                for k1 in range(n):
                    if k1 == i or k1 == j:
                        continue
                    if not (adj[k1, i] and adj[i, k1]):
                        continue  # k1 must be undirected neighbour of i
                    if not (adj[k1, j] and not adj[j, k1]):
                        continue  # k1 → j must be directed
                    for k2 in range(k1 + 1, n):
                        if k2 == i or k2 == j:
                            continue
                        if not (adj[k2, i] and adj[i, k2]):
                            continue
                        if not (adj[k2, j] and not adj[j, k2]):
                            continue
                        if not adj[k1, k2] and not adj[k2, k1]:
                            adj[i, j] = 1
                            adj[j, i] = 0
                            oriented = True
                            changed = True
                            break
                    if oriented:
                        break

                if not oriented:
                    still_undirected.append((i, j))

            remaining = still_undirected

        return remaining

    @staticmethod
    def _chain_components(
        undirected_pairs: list[tuple[int, int]],
    ) -> list[list[tuple[int, int]]]:
        """Partition undirected edge pairs into connected components.

        Two undirected edges belong to the same component if they share
        at least one endpoint (transitively).  This corresponds to the
        *chain components* of the CPDAG.

        Parameters
        ----------
        undirected_pairs : list of (i, j)
            Undirected edge pairs.

        Returns
        -------
        list of list of (i, j)
            Each inner list is one connected component of undirected
            edges.
        """
        if not undirected_pairs:
            return []

        # Union-Find for endpoint grouping.
        parent: dict[int, int] = {}

        def find(x: int) -> int:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i, j in undirected_pairs:
            parent.setdefault(i, i)
            parent.setdefault(j, j)
            union(i, j)

        # Group edges by root.
        groups: dict[int, list[tuple[int, int]]] = {}
        for i, j in undirected_pairs:
            root = find(i)
            groups.setdefault(root, []).append((i, j))

        return list(groups.values())


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _permutations(n: int) -> Iterator[list[int]]:
    """Yield all permutations of ``[0, 1, …, n-1]``.

    A simple recursive implementation sufficient for small *n* (≤ 5).
    """
    if n == 0:
        yield []
        return
    if n == 1:
        yield [0]
        return
    for perm in _permutations(n - 1):
        for pos in range(n):
            yield perm[:pos] + [n - 1] + perm[pos:]
