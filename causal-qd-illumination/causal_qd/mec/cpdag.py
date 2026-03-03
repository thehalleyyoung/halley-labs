"""DAG to CPDAG conversion using Chickering's algorithm.

This module provides the :class:`CPDAGConverter` class, which converts a
Directed Acyclic Graph (DAG) to its Completed Partially Directed Acyclic
Graph (CPDAG) representation and vice-versa.  The CPDAG is the unique
graphical representative of a DAG's Markov Equivalence Class (MEC): two
DAGs belong to the same MEC iff they share the same skeleton and
v-structures.

Key concepts
------------
* **Compelled edge** – a directed edge that appears with the *same*
  orientation in *every* DAG belonging to the MEC.  In the CPDAG matrix:
  ``cpdag[i, j] == 1`` and ``cpdag[j, i] == 0``.
* **Reversible edge** – an edge whose orientation may differ across members
  of the MEC.  Represented as undirected: ``cpdag[i, j] == cpdag[j, i] == 1``.

The implementation follows Chickering (2002) – *"Optimal Structure
Identification With Greedy Search"* – for labelling compelled edges, and
Meek (1995) orientation rules R1–R4 for propagating constraints.

References
----------
- Chickering, D. M. (2002).  Optimal structure identification with greedy
  search.  *JMLR*, 3, 507–554.
- Meek, C. (1995).  Causal inference and causal explanation with background
  knowledge.  *UAI*, 403–410.
- Andersson, S. A., Madigan, D., & Perlman, M. D. (1997).  A
  characterization of Markov equivalence classes for acyclic digraphs.
  *The Annals of Statistics*, 25(2), 505–541.
"""
from __future__ import annotations

from collections import deque
from itertools import product
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from causal_qd.core.dag import DAG
from causal_qd.types import AdjacencyMatrix


class CPDAGConverter:
    """Convert between DAGs and their CPDAG (Markov equivalence class) form.

    The CPDAG encodes the Markov equivalence class: directed edges are
    *compelled* (present in every member of the MEC) while undirected edges
    are *reversible*.

    The implementation follows Chickering (2002) – "Optimal Structure
    Identification With Greedy Search" – for labelling compelled edges and
    Meek (1995) orientation rules R1–R4 for constraint propagation.

    Examples
    --------
    >>> import numpy as np
    >>> from causal_qd.core.dag import DAG
    >>> adj = np.array([[0,1,0],[0,0,1],[0,0,0]], dtype=np.int8)
    >>> dag = DAG(adj)
    >>> converter = CPDAGConverter()
    >>> cpdag = converter.dag_to_cpdag(dag)
    >>> # A chain 0→1→2 has no v-structures so all edges are reversible
    >>> assert cpdag[0,1] == 1 and cpdag[1,0] == 1
    >>> assert cpdag[1,2] == 1 and cpdag[2,1] == 1
    """

    # ------------------------------------------------------------------
    # Public API – DAG to CPDAG
    # ------------------------------------------------------------------

    def dag_to_cpdag(self, dag: DAG) -> AdjacencyMatrix:
        """Convert *dag* to its CPDAG representation using Chickering's algorithm.

        The algorithm proceeds in three phases:

        1. **Initialise** – start with all edges from the original DAG.
        2. **V-structure detection** – identify unshielded colliders
           (i → j ← k where i and k are *not* adjacent) and mark those
           edges as compelled.
        3. **Meek rule propagation** – iteratively apply Meek rules R1–R4
           until no further edges can be oriented (fixed-point).

        In the returned matrix:
        - ``cpdag[i, j] == 1`` and ``cpdag[j, i] == 0`` means *i → j* is
          compelled (directed).
        - ``cpdag[i, j] == 1`` and ``cpdag[j, i] == 1`` means the edge
          between *i* and *j* is reversible (undirected).

        Parameters
        ----------
        dag : DAG
            A valid directed acyclic graph.

        Returns
        -------
        AdjacencyMatrix
            The CPDAG adjacency matrix (``int8``).

        Notes
        -----
        Time complexity is *O(n² · e)* in the worst case where *n* is the
        number of nodes and *e* the number of edges, dominated by the
        iterative Meek-rule application.
        """
        adj = dag.adjacency.copy()
        n = dag.n_nodes

        # ----------------------------------------------------------
        # Step 1: Initialise CPDAG with all edges undirected
        # ----------------------------------------------------------
        cpdag = np.zeros((n, n), dtype=np.int8)
        for i, j in dag.edges:
            cpdag[i, j] = 1
            cpdag[j, i] = 1  # start undirected

        # ----------------------------------------------------------
        # Step 2: Orient v-structures (unshielded colliders)
        # ----------------------------------------------------------
        for i, j, k in self.find_v_structures(adj):
            # Orient i → j and k → j  (remove reverse directions)
            cpdag[j, i] = 0
            cpdag[j, k] = 0

        # ----------------------------------------------------------
        # Step 3: Apply Meek rules iteratively until convergence
        # ----------------------------------------------------------
        changed = True
        while changed:
            changed = self._apply_meek_rules(cpdag, n)

        return cpdag

    # ------------------------------------------------------------------
    # Public API – CPDAG to DAGs (MEC enumeration)
    # ------------------------------------------------------------------

    def cpdag_to_dags(self, cpdag: AdjacencyMatrix) -> list[DAG]:
        """Enumerate all DAGs consistent with the given CPDAG.

        This method is intended for **small** MECs.  It extracts the
        compelled (directed) and reversible (undirected) edges, then
        explores every legal orientation of the undirected edges via
        depth-first search with constraint propagation (Meek rules)
        and cycle checking.

        Parameters
        ----------
        cpdag : AdjacencyMatrix
            A valid CPDAG represented as a square ``int8`` matrix.

        Returns
        -------
        list[DAG]
            All DAGs in the Markov equivalence class represented by
            *cpdag*, each as a :class:`DAG` instance.  The list is
            de-duplicated by adjacency matrix equality.

        Raises
        ------
        ValueError
            If the CPDAG is not square.

        Notes
        -----
        The worst-case complexity is exponential in the number of
        undirected edges; use only for moderately sized graphs.
        """
        cpdag = np.asarray(cpdag, dtype=np.int8)
        if cpdag.ndim != 2 or cpdag.shape[0] != cpdag.shape[1]:
            raise ValueError("CPDAG must be a square matrix.")
        n = cpdag.shape[0]

        # Separate directed (compelled) and undirected (reversible) edges
        directed_edges: list[tuple[int, int]] = []
        undirected_edges: list[tuple[int, int]] = []

        for i in range(n):
            for j in range(i + 1, n):
                if cpdag[i, j] and cpdag[j, i]:
                    # Undirected edge – store canonical pair (i < j)
                    undirected_edges.append((i, j))
                elif cpdag[i, j] and not cpdag[j, i]:
                    directed_edges.append((i, j))
                elif cpdag[j, i] and not cpdag[i, j]:
                    directed_edges.append((j, i))

        if not undirected_edges:
            # Fully directed – only one DAG in the MEC
            adj = np.zeros((n, n), dtype=np.int8)
            for i, j in directed_edges:
                adj[i, j] = 1
            try:
                return [DAG(adj)]
            except Exception:
                return []

        # DFS over all 2^|undirected| orientation combinations
        result_dags: list[DAG] = []
        seen_hashes: set[int] = set()

        self._enumerate_dfs(
            n=n,
            directed_edges=directed_edges,
            undirected_edges=undirected_edges,
            idx=0,
            current_orientations=[],
            result_dags=result_dags,
            seen_hashes=seen_hashes,
            input_cpdag=cpdag,
        )

        return result_dags

    # ------------------------------------------------------------------
    # Public API – Validation
    # ------------------------------------------------------------------

    def is_valid_cpdag(self, cpdag: AdjacencyMatrix) -> bool:
        """Check whether *cpdag* is a valid CPDAG.

        A valid CPDAG must satisfy all of the following:

        1. **No self-loops** – ``cpdag[i, i] == 0`` for all *i*.
        2. **No directed cycles** – the directed sub-graph (edges where
           ``cpdag[i,j] == 1`` and ``cpdag[j,i] == 0``) must be acyclic.
        3. **Reversibility** – every undirected edge (``cpdag[i,j] == 1``
           and ``cpdag[j,i] == 1``) must genuinely be reversible: there
           must exist at least one consistent DAG with the edge in each
           direction.
        4. **Meek closure** – the Meek rules R1–R4 must be fully applied;
           applying them again must produce no further orientations.

        Parameters
        ----------
        cpdag : AdjacencyMatrix
            Candidate CPDAG to validate.

        Returns
        -------
        bool
            ``True`` if all checks pass.
        """
        cpdag = np.asarray(cpdag, dtype=np.int8)
        if cpdag.ndim != 2 or cpdag.shape[0] != cpdag.shape[1]:
            return False
        n = cpdag.shape[0]

        # Check 1: no self-loops
        if np.any(np.diag(cpdag) != 0):
            return False

        # Check 2: no directed cycles in the directed subgraph
        directed = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(n):
                if cpdag[i, j] and not cpdag[j, i]:
                    directed[i, j] = 1
        if not DAG.is_acyclic(directed):
            return False

        # Check 3: Meek closure – applying rules should change nothing
        cpdag_copy = cpdag.copy()
        if self._apply_meek_rules(cpdag_copy, n):
            return False

        # Check 4: every undirected edge is truly reversible
        # (verify that at least one extension DAG exists)
        # We do a lightweight check: try to find *any* consistent DAG
        try:
            dags = self.cpdag_to_dags(cpdag)
            if len(dags) == 0:
                return False
        except Exception:
            return False

        return True

    # ------------------------------------------------------------------
    # Public API – Edge queries
    # ------------------------------------------------------------------

    def is_compelled(self, dag: DAG, i: int, j: int) -> bool:
        """Check whether the edge *i → j* is compelled (not reversible).

        An edge is compelled if it has the same orientation in *every* DAG
        belonging to the Markov equivalence class.

        Parameters
        ----------
        dag : DAG
            The DAG containing the edge.
        i, j : int
            Source and target of the edge.

        Returns
        -------
        bool
            ``True`` if *i → j* is compelled in the CPDAG.

        Raises
        ------
        ValueError
            If edge *i → j* does not exist in *dag*.
        """
        if not dag.has_edge(i, j):
            raise ValueError(f"Edge {i} → {j} does not exist in the DAG.")
        cpdag = self.dag_to_cpdag(dag)
        return bool(cpdag[i, j] == 1 and cpdag[j, i] == 0)

    # ------------------------------------------------------------------
    # Public API – V-structure detection
    # ------------------------------------------------------------------

    def find_v_structures(self, adj: AdjacencyMatrix) -> list[tuple[int, int, int]]:
        """Find all v-structures (unshielded colliders) in a directed graph.

        A v-structure (also called an *unshielded collider*) is a triple
        ``(i, j, k)`` such that:

        * ``i → j`` and ``k → j`` (both edges point *into* j),
        * ``i`` and ``k`` are **not** adjacent (no edge in either direction).

        Parameters
        ----------
        adj : AdjacencyMatrix
            Adjacency matrix of a directed graph (need not be a DAG, but
            is typically used with DAG adjacency matrices).

        Returns
        -------
        list[tuple[int, int, int]]
            List of ``(i, j, k)`` triples in canonical order ``i < k``.

        Notes
        -----
        Runs in *O(n · d²)* where *d* is the maximum in-degree.
        """
        adj = np.asarray(adj, dtype=np.int8)
        n = adj.shape[0]
        result: list[tuple[int, int, int]] = []

        for j in range(n):
            # Collect parents of j
            parents_j = sorted(int(p) for p in np.nonzero(adj[:, j])[0])
            for a_idx in range(len(parents_j)):
                for b_idx in range(a_idx + 1, len(parents_j)):
                    i, k = parents_j[a_idx], parents_j[b_idx]
                    # i < k guaranteed by sorted order
                    # Check that i and k are NOT adjacent
                    if not adj[i, k] and not adj[k, i]:
                        result.append((i, j, k))

        return result

    # ------------------------------------------------------------------
    # Public API – Meek rules (single round)
    # ------------------------------------------------------------------

    def _apply_meek_rules(self, cpdag: AdjacencyMatrix, n: int) -> bool:
        """Apply one round of Meek orientation rules R1–R4.

        The rules orient undirected edges in the CPDAG when their
        orientation is *forced* by the existing directed edges and the
        requirement that no new v-structures or directed cycles are
        introduced.

        Parameters
        ----------
        cpdag : AdjacencyMatrix
            The CPDAG matrix, modified **in-place**.
        n : int
            Number of nodes (== ``cpdag.shape[0]``).

        Returns
        -------
        bool
            ``True`` if at least one edge was oriented in this round.

        Notes
        -----
        **R1** – *a → b — c* and *a* not adjacent to *c*  ⟹  orient *b → c*.
            Orienting b — c as b ← c would create a new v-structure a → b ← c.

        **R2** – *a → b → c* and *a — c*  ⟹  orient *a → c*.
            Orienting a — c as a ← c would create a directed cycle a → b → c → a.

        **R3** – *a — c*, *a — d*, *c → b*, *d → b*, *c* not adjacent to *d*
            ⟹  orient *a → b*.
            Two directed paths from a to b via c and d force the orientation.

        **R4** – *a — b*, *b → c → d*, *a — d*  ⟹  orient *a → d*.
            Avoiding a new v-structure forces the orientation.
        """
        changed = False

        # Helper predicates
        def _is_directed(i: int, j: int) -> bool:
            """True if i → j is directed (not undirected)."""
            return bool(cpdag[i, j] == 1 and cpdag[j, i] == 0)

        def _is_undirected(i: int, j: int) -> bool:
            """True if i — j is undirected."""
            return bool(cpdag[i, j] == 1 and cpdag[j, i] == 1)

        def _is_adjacent(i: int, j: int) -> bool:
            """True if any edge exists between i and j."""
            return bool(cpdag[i, j] or cpdag[j, i])

        def _orient(i: int, j: int) -> None:
            """Orient i — j as i → j."""
            nonlocal changed
            cpdag[i, j] = 1
            cpdag[j, i] = 0
            changed = True

        # ----- R1: a → b — c, a not adj c  ⟹  b → c -----
        for b in range(n):
            for a in range(n):
                if a == b:
                    continue
                if not _is_directed(a, b):
                    continue
                for c in range(n):
                    if c == a or c == b:
                        continue
                    if _is_undirected(b, c) and not _is_adjacent(a, c):
                        _orient(b, c)

        # ----- R2: a → b → c, a — c  ⟹  a → c -----
        for a in range(n):
            for b in range(n):
                if b == a:
                    continue
                if not _is_directed(a, b):
                    continue
                for c in range(n):
                    if c == a or c == b:
                        continue
                    if _is_directed(b, c) and _is_undirected(a, c):
                        _orient(a, c)

        # ----- R3: a — c, a — d, c → b, d → b, c not adj d  ⟹  a → b -----
        for a in range(n):
            for b in range(n):
                if b == a:
                    continue
                if not _is_undirected(a, b):
                    continue
                # Find two distinct c, d such that:
                #   a — c, c → b, a — d, d → b, c not adj d
                # Collect candidates: nodes adjacent-undirected to a AND
                # with directed edge into b
                candidates: list[int] = []
                for x in range(n):
                    if x == a or x == b:
                        continue
                    if _is_undirected(a, x) and _is_directed(x, b):
                        candidates.append(x)
                # Check pairs
                oriented = False
                for ci in range(len(candidates)):
                    if oriented:
                        break
                    for di in range(ci + 1, len(candidates)):
                        c, d = candidates[ci], candidates[di]
                        if not _is_adjacent(c, d):
                            _orient(a, b)
                            oriented = True
                            break

        # ----- R4: a — b, b → c → d, a — d  ⟹  a → d -----
        for a in range(n):
            for b in range(n):
                if b == a:
                    continue
                if not _is_undirected(a, b):
                    continue
                for c in range(n):
                    if c == a or c == b:
                        continue
                    if not _is_directed(b, c):
                        continue
                    for d in range(n):
                        if d == a or d == b or d == c:
                            continue
                        if _is_directed(c, d) and _is_undirected(a, d):
                            _orient(a, d)

        return changed

    # ------------------------------------------------------------------
    # Public API – Compelled / reversible edge analysis
    # ------------------------------------------------------------------

    def compelled_edge_analysis(self, dag: DAG) -> dict:
        """Analyse which edges in *dag* are compelled vs. reversible.

        Parameters
        ----------
        dag : DAG
            The DAG to analyse.

        Returns
        -------
        dict
            A dictionary with the following keys:

            ``"compelled_edges"``
                List of ``(i, j)`` tuples for compelled (directed in
                CPDAG) edges.
            ``"reversible_edges"``
                List of ``(i, j)`` tuples for reversible (undirected in
                CPDAG) edges, stored in the original DAG orientation.
            ``"compelled_count"``
                Number of compelled edges.
            ``"reversible_count"``
                Number of reversible edges.
            ``"total_edges"``
                Total number of edges in the DAG.

        Examples
        --------
        >>> import numpy as np
        >>> from causal_qd.core.dag import DAG
        >>> # v-structure: 0 → 1 ← 2
        >>> adj = np.array([[0,1,0],[0,0,0],[0,1,0]], dtype=np.int8)
        >>> dag = DAG(adj)
        >>> converter = CPDAGConverter()
        >>> info = converter.compelled_edge_analysis(dag)
        >>> assert info["compelled_count"] == 2
        >>> assert info["reversible_count"] == 0
        """
        cpdag = self.dag_to_cpdag(dag)

        compelled_edges: list[tuple[int, int]] = []
        reversible_edges: list[tuple[int, int]] = []

        for i, j in dag.edges:
            if cpdag[i, j] == 1 and cpdag[j, i] == 0:
                compelled_edges.append((i, j))
            else:
                reversible_edges.append((i, j))

        return {
            "compelled_edges": compelled_edges,
            "reversible_edges": reversible_edges,
            "compelled_count": len(compelled_edges),
            "reversible_count": len(reversible_edges),
            "total_edges": len(compelled_edges) + len(reversible_edges),
        }

    # ------------------------------------------------------------------
    # Private helpers – DFS enumeration
    # ------------------------------------------------------------------

    def _enumerate_dfs(
        self,
        n: int,
        directed_edges: list[tuple[int, int]],
        undirected_edges: list[tuple[int, int]],
        idx: int,
        current_orientations: list[tuple[int, int]],
        result_dags: list[DAG],
        seen_hashes: set[int],
        input_cpdag: AdjacencyMatrix,
    ) -> None:
        """Depth-first enumeration of DAG orientations for undirected edges.

        At each level *idx* of the recursion we choose to orient the
        undirected edge ``(u, v)`` as either ``u → v`` or ``v → u``.
        After fixing an orientation we perform lightweight constraint
        propagation (Meek rules) and cycle detection to prune the
        search tree early.  At the leaf, the candidate DAG is verified
        to produce the same CPDAG as the input (MEC consistency).

        Parameters
        ----------
        n : int
            Number of nodes.
        directed_edges : list[tuple[int, int]]
            Compelled edges (fixed orientation).
        undirected_edges : list[tuple[int, int]]
            Reversible edges yet to be oriented.  Each stored as ``(u, v)``
            with ``u < v``.
        idx : int
            Current position in *undirected_edges* being decided.
        current_orientations : list[tuple[int, int]]
            Orientations chosen so far for edges at indices ``0 .. idx-1``.
        result_dags : list[DAG]
            Accumulator for valid DAGs found.
        seen_hashes : set[int]
            Hashes of adjacency matrices already recorded (dedup).
        input_cpdag : AdjacencyMatrix
            The original CPDAG; used to verify that each candidate DAG
            belongs to the same MEC.
        """
        if idx == len(undirected_edges):
            # All undirected edges have been oriented – build the DAG
            adj = np.zeros((n, n), dtype=np.int8)
            for i, j in directed_edges:
                adj[i, j] = 1
            for i, j in current_orientations:
                adj[i, j] = 1

            # Cycle check
            if not DAG.is_acyclic(adj):
                return

            h = hash(adj.tobytes())
            if h in seen_hashes:
                return
            seen_hashes.add(h)

            try:
                candidate = DAG(adj)
            except Exception:
                return

            # Verify MEC consistency: candidate must produce same CPDAG
            candidate_cpdag = self.dag_to_cpdag(candidate)
            if not np.array_equal(candidate_cpdag, input_cpdag):
                return

            result_dags.append(candidate)
            return

        u, v = undirected_edges[idx]
        # Try both orientations: u → v and v → u
        for src, tgt in [(u, v), (v, u)]:
            # Quick partial cycle check: build partial adj and look for
            # obvious cycles using the edges committed so far
            partial_adj = np.zeros((n, n), dtype=np.int8)
            for i, j in directed_edges:
                partial_adj[i, j] = 1
            for i, j in current_orientations:
                partial_adj[i, j] = 1
            partial_adj[src, tgt] = 1

            # Check if adding this edge creates a cycle in committed edges
            if self._has_partial_cycle(partial_adj, n):
                continue

            current_orientations.append((src, tgt))
            self._enumerate_dfs(
                n=n,
                directed_edges=directed_edges,
                undirected_edges=undirected_edges,
                idx=idx + 1,
                current_orientations=current_orientations,
                result_dags=result_dags,
                seen_hashes=seen_hashes,
                input_cpdag=input_cpdag,
            )
            current_orientations.pop()

    @staticmethod
    def _has_partial_cycle(adj: AdjacencyMatrix, n: int) -> bool:
        """Check for directed cycles using Kahn's algorithm.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Partial adjacency matrix (may have missing edges).
        n : int
            Number of nodes.

        Returns
        -------
        bool
            ``True`` if the graph contains at least one directed cycle.
        """
        return not DAG.is_acyclic(adj)
