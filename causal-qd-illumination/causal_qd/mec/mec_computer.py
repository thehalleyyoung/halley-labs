"""Markov Equivalence Class computation and comparison.

This module provides the :class:`MECComputer` class for computing properties
of Markov Equivalence Classes (MECs) of directed acyclic graphs (DAGs).

Two DAGs belong to the same MEC if and only if they share the same skeleton
(underlying undirected graph) and the same set of v-structures (unshielded
colliders).  The CPDAG (Completed Partially Directed Acyclic Graph) is the
unique graphical representative of an MEC: directed edges are *compelled*
(present in every member of the class) while undirected edges are *reversible*
(their orientation varies across class members).

Key capabilities:

* **MEC size estimation** – count (exactly for small components, heuristically
  for large ones) how many DAGs share the same MEC.
* **Equivalence testing** – decide whether two DAGs are Markov equivalent by
  comparing skeletons and v-structures.
* **V-structure enumeration** – list all unshielded colliders in a DAG.
* **MEC distance** – Structural Hamming Distance (SHD) between two CPDAGs,
  normalised to [0, 1].
* **Skeleton distance** – number of edge differences between undirected
  skeletons.
* **V-structure distance** – size of the symmetric difference of v-structure
  sets.
* **MEC sampling** – draw DAGs uniformly (or approximately so) from the MEC
  of a given DAG via enumeration or rejection sampling.
"""
from __future__ import annotations

from collections import deque
from itertools import product as iterproduct
from typing import List, Tuple

import numpy as np

from causal_qd.core.dag import DAG
from causal_qd.mec.cpdag import CPDAGConverter
from causal_qd.types import AdjacencyMatrix

# Maximum number of undirected edges for which we perform exact enumeration
# instead of falling back to a heuristic or rejection sampling.
_EXACT_ENUMERATION_THRESHOLD = 15


class MECComputer:
    """Compute properties of Markov Equivalence Classes.

    Two DAGs belong to the same MEC iff they share the same skeleton and
    the same set of v-structures (unshielded colliders).  This class
    provides methods for MEC size estimation, equivalence checking,
    v-structure enumeration, distance metrics between MECs, and sampling
    DAGs from an MEC.

    Examples
    --------
    >>> from causal_qd.core.dag import DAG
    >>> import numpy as np
    >>> adj = np.array([[0, 1, 0],
    ...                 [0, 0, 1],
    ...                 [0, 0, 0]], dtype=np.int8)
    >>> dag = DAG(adj)
    >>> mec = MECComputer()
    >>> mec.compute_mec_size(dag)
    1
    """

    def __init__(self) -> None:
        """Initialise the MEC computer.

        Creates an internal :class:`CPDAGConverter` instance used by all
        methods that require conversion between DAG and CPDAG
        representations.
        """
        self._cpdag_converter = CPDAGConverter()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _undirected_mask(self, cpdag: AdjacencyMatrix) -> np.ndarray:
        """Return a boolean matrix marking undirected (reversible) edges.

        An edge (i, j) is undirected in the CPDAG when both
        ``cpdag[i, j] == 1`` and ``cpdag[j, i] == 1``.

        Parameters
        ----------
        cpdag : AdjacencyMatrix
            CPDAG adjacency matrix.

        Returns
        -------
        np.ndarray
            Boolean matrix of shape ``(n, n)``.
        """
        mask = (cpdag == 1) & (cpdag.T == 1)
        np.fill_diagonal(mask, False)
        return mask

    def _find_components(
        self, undirected: np.ndarray
    ) -> list[list[int]]:
        """Find connected components of undirected (reversible) edges.

        Uses breadth-first search over the symmetric ``undirected``
        adjacency to partition nodes that participate in at least one
        undirected edge into connected components.

        Parameters
        ----------
        undirected : np.ndarray
            Boolean symmetric matrix where ``True`` indicates an
            undirected edge.

        Returns
        -------
        list[list[int]]
            Each inner list contains the node indices of one connected
            component, sorted in ascending order.
        """
        n = undirected.shape[0]
        visited = np.zeros(n, dtype=bool)
        components: list[list[int]] = []

        for start in range(n):
            if visited[start] or not undirected[start].any():
                continue
            component: list[int] = []
            queue: deque[int] = deque([start])
            visited[start] = True
            while queue:
                node = queue.popleft()
                component.append(node)
                for nbr in np.nonzero(undirected[node])[0]:
                    if not visited[nbr]:
                        visited[nbr] = True
                        queue.append(int(nbr))
            components.append(sorted(component))

        return components

    def _count_acyclic_orientations(
        self,
        component_nodes: list[int],
        undirected: np.ndarray,
        directed_adj: AdjacencyMatrix,
    ) -> int:
        """Count valid acyclic orientations for a single component.

        For small components (≤ ``_EXACT_ENUMERATION_THRESHOLD`` undirected
        edges) an exact count is obtained by enumerating all 2^m possible
        orientations and checking acyclicity via Kahn's algorithm on the
        sub-graph.  For larger components a heuristic estimate is returned:
        each chain of *k* nodes contributes roughly *k* valid orientations.

        Parameters
        ----------
        component_nodes : list[int]
            Node indices belonging to this connected component.
        undirected : np.ndarray
            Full boolean matrix of undirected edges.
        directed_adj : AdjacencyMatrix
            Full CPDAG adjacency matrix (directed edges only, i.e.
            ``cpdag[i,j]==1`` and ``cpdag[j,i]==0``).

        Returns
        -------
        int
            Number (exact or estimated) of valid acyclic orientations.
        """
        # Collect undirected edges (upper triangle only to avoid double
        # counting).
        edges: list[tuple[int, int]] = []
        for i in component_nodes:
            for j in component_nodes:
                if j > i and undirected[i, j]:
                    edges.append((i, j))

        n_edges = len(edges)

        if n_edges == 0:
            return 1

        # --- Exact enumeration for small components --------------------
        if n_edges <= _EXACT_ENUMERATION_THRESHOLD:
            n = undirected.shape[0]
            count = 0
            for orientation in iterproduct([0, 1], repeat=n_edges):
                # Build a trial adjacency matrix with directed edges fixed.
                trial = directed_adj.copy()
                for idx, (u, v) in enumerate(edges):
                    if orientation[idx] == 0:
                        trial[u, v] = 1
                        trial[v, u] = 0
                    else:
                        trial[v, u] = 1
                        trial[u, v] = 0

                if DAG.is_acyclic(trial):
                    count += 1
            return max(count, 1)

        # --- Heuristic for large components ----------------------------
        # Approximate: each chain of k nodes has ~k acyclic orientations.
        k = len(component_nodes)
        return max(k, 1)

    def _extract_edge_lists(
        self, cpdag: AdjacencyMatrix
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Split CPDAG edges into directed and undirected lists.

        Parameters
        ----------
        cpdag : AdjacencyMatrix
            CPDAG adjacency matrix.

        Returns
        -------
        directed : list[tuple[int, int]]
            Directed (compelled) edges ``(i, j)`` where ``cpdag[i,j]==1``
            and ``cpdag[j,i]==0``.
        undirected : list[tuple[int, int]]
            Undirected (reversible) edges ``(i, j)`` with ``i < j`` where
            both ``cpdag[i,j]==1`` and ``cpdag[j,i]==1``.
        """
        n = cpdag.shape[0]
        directed: list[tuple[int, int]] = []
        undirected: list[tuple[int, int]] = []
        seen_undirected: set[tuple[int, int]] = set()

        for i in range(n):
            for j in range(n):
                if cpdag[i, j] != 1:
                    continue
                if cpdag[j, i] == 1:
                    pair = (min(i, j), max(i, j))
                    if pair not in seen_undirected:
                        seen_undirected.add(pair)
                        undirected.append(pair)
                else:
                    directed.append((i, j))

        return directed, undirected

    @staticmethod
    def _orientation_is_acyclic(
        n: int,
        directed: list[tuple[int, int]],
        undirected: list[tuple[int, int]],
        orientation: tuple[int, ...] | list[int],
    ) -> np.ndarray | None:
        """Build adjacency from orientation and return it if acyclic.

        Parameters
        ----------
        n : int
            Number of nodes.
        directed : list[tuple[int, int]]
            Compelled directed edges.
        undirected : list[tuple[int, int]]
            Reversible edges ``(i, j)`` with ``i < j``.
        orientation : sequence of int
            For each undirected edge, ``0`` means orient ``i → j``,
            ``1`` means ``j → i``.

        Returns
        -------
        np.ndarray or None
            The resulting adjacency matrix if the orientation is acyclic,
            otherwise ``None``.
        """
        adj = np.zeros((n, n), dtype=np.int8)
        for i, j in directed:
            adj[i, j] = 1

        for idx, (u, v) in enumerate(undirected):
            if orientation[idx] == 0:
                adj[u, v] = 1
            else:
                adj[v, u] = 1

        if DAG.is_acyclic(adj):
            return adj
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_mec_size(self, dag: DAG) -> int:
        """Estimate the size of the MEC containing *dag*.

        Uses the CPDAG representation to enumerate orientations of
        undirected (reversible) edges.  For each connected component of
        reversible edges the number of valid acyclic orientations is
        computed and multiplied together.

        For components with at most ``_EXACT_ENUMERATION_THRESHOLD``
        undirected edges the count is exact (brute-force enumeration of
        all 2^m orientations with acyclicity checks).  For larger
        components a chain-based heuristic is used: a connected component
        of *k* nodes is assumed to have roughly *k* valid orientations.

        Parameters
        ----------
        dag : DAG
            A directed acyclic graph.

        Returns
        -------
        int
            Estimated (or exact, for small MECs) number of DAGs in the
            Markov Equivalence Class.
        """
        cpdag = self._cpdag_converter.dag_to_cpdag(dag)
        undirected = self._undirected_mask(cpdag)

        # Build a "directed-only" copy of the CPDAG for sub-graph checks.
        directed_adj = cpdag.copy()
        # Zero out entries that are part of undirected edges.
        directed_adj[undirected] = 0

        components = self._find_components(undirected)

        if not components:
            return 1

        total = 1
        for comp in components:
            total *= self._count_acyclic_orientations(
                comp, undirected, directed_adj
            )
        return total

    def are_equivalent(self, dag1: DAG, dag2: DAG) -> bool:
        """Check whether *dag1* and *dag2* are Markov equivalent.

        Two DAGs are Markov equivalent iff they share the same skeleton
        and the same set of v-structures (unshielded colliders).  This is
        a necessary and sufficient condition (Verma & Pearl, 1990).

        Parameters
        ----------
        dag1 : DAG
            First directed acyclic graph.
        dag2 : DAG
            Second directed acyclic graph.  Must have the same number of
            nodes as *dag1*.

        Returns
        -------
        bool
            ``True`` if both DAGs represent the same conditional
            independence relations, ``False`` otherwise.
        """
        if dag1.n_nodes != dag2.n_nodes:
            return False

        # Same skeleton?
        skel1 = dag1.skeleton()
        skel2 = dag2.skeleton()
        if not np.array_equal(skel1, skel2):
            return False

        # Same v-structures?
        vs1 = set(self.find_v_structures(dag1))
        vs2 = set(self.find_v_structures(dag2))
        return vs1 == vs2

    def find_v_structures(self, dag: DAG) -> List[Tuple[int, int, int]]:
        """Find all v-structures (unshielded colliders) in *dag*.

        A v-structure is a triple ``(i, j, k)`` where ``i → j ← k`` and
        ``i`` and ``k`` are *not* adjacent (no edge in either direction).
        The canonical form stores each triple with ``i < k`` to avoid
        counting the same collider twice.

        Parameters
        ----------
        dag : DAG
            A directed acyclic graph.

        Returns
        -------
        List[Tuple[int, int, int]]
            Sorted list of ``(i, j, k)`` triples with ``i < k``, where
            ``j`` is the collider node.
        """
        adj = dag.adjacency
        n = dag.n_nodes
        v_structures: list[Tuple[int, int, int]] = []

        for j in range(n):
            parents_j = sorted(dag.parents(j))
            for idx_a, i in enumerate(parents_j):
                for k in parents_j[idx_a + 1:]:
                    # i -> j <- k; check i and k are NOT adjacent.
                    if not adj[i, k] and not adj[k, i]:
                        v_structures.append((i, j, k))

        return sorted(v_structures)

    def mec_distance(self, dag1: DAG, dag2: DAG) -> float:
        """Compute the normalised Structural Hamming Distance between MECs.

        Both DAGs are converted to their CPDAG representations and the
        Structural Hamming Distance (SHD) is computed.  The SHD counts:

        * Edges present in one CPDAG but absent in the other.
        * Edges present in both CPDAGs but with different types (directed
          vs. undirected, or opposite directed orientations).

        The raw count is normalised by the maximum possible number of
        directed entries ``n * (n - 1)`` so the result lies in [0, 1].

        Parameters
        ----------
        dag1 : DAG
            First directed acyclic graph.
        dag2 : DAG
            Second directed acyclic graph.  Must have the same number of
            nodes as *dag1*.

        Returns
        -------
        float
            Normalised SHD in the range [0.0, 1.0].  A value of 0.0
            indicates that the two DAGs belong to the same MEC.

        Raises
        ------
        ValueError
            If the two DAGs have different numbers of nodes.
        """
        if dag1.n_nodes != dag2.n_nodes:
            raise ValueError(
                f"Node count mismatch: {dag1.n_nodes} vs {dag2.n_nodes}"
            )

        n = dag1.n_nodes
        if n <= 1:
            return 0.0

        cpdag1 = self._cpdag_converter.dag_to_cpdag(dag1)
        cpdag2 = self._cpdag_converter.dag_to_cpdag(dag2)

        # Count entry-level differences in the CPDAG adjacency matrices.
        # Each ordered pair (i, j) with i ≠ j is inspected independently;
        # this naturally accounts for directed vs. undirected mismatches.
        diff = 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                c1 = int(cpdag1[i, j])
                c2 = int(cpdag2[i, j])
                if c1 != c2:
                    diff += 1

        max_distance = n * (n - 1)
        return diff / max_distance

    def mec_sample(
        self,
        dag: DAG,
        n_samples: int,
        rng: np.random.Generator | None = None,
    ) -> list[DAG]:
        """Sample DAGs uniformly from the MEC of *dag*.

        The algorithm converts *dag* to its CPDAG, separates directed
        (compelled) and undirected (reversible) edges, and then generates
        DAGs by choosing orientations for the undirected edges such that
        the resulting graph remains acyclic.

        Two strategies are used depending on the number of undirected
        edges:

        * **Exact enumeration** (≤ ``_EXACT_ENUMERATION_THRESHOLD``
          undirected edges): all valid orientations are enumerated and
          samples are drawn with replacement from the valid set.
        * **Rejection sampling** (many undirected edges): random
          orientations are proposed and accepted only if the result is
          acyclic.

        Parameters
        ----------
        dag : DAG
            A directed acyclic graph whose MEC is to be sampled.
        n_samples : int
            Number of DAG samples to draw.  Must be ≥ 1.
        rng : np.random.Generator or None, optional
            NumPy random generator for reproducibility.  If ``None`` a
            new default generator is created.

        Returns
        -------
        list[DAG]
            A list of ``n_samples`` DAGs, each belonging to the same MEC
            as *dag*.

        Raises
        ------
        ValueError
            If ``n_samples < 1``.
        RuntimeError
            If rejection sampling fails to find enough valid orientations
            within a reasonable number of attempts.
        """
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")

        if rng is None:
            rng = np.random.default_rng()

        cpdag = self._cpdag_converter.dag_to_cpdag(dag)
        n = cpdag.shape[0]
        directed, undirected = self._extract_edge_lists(cpdag)
        n_undirected = len(undirected)

        # Trivial case: no reversible edges → the MEC contains only this
        # DAG.
        if n_undirected == 0:
            return [dag] * n_samples

        # ------ Exact enumeration for small MECs ----------------------
        if n_undirected <= _EXACT_ENUMERATION_THRESHOLD:
            valid_adjs: list[np.ndarray] = []
            for orientation in iterproduct([0, 1], repeat=n_undirected):
                result = self._orientation_is_acyclic(
                    n, directed, undirected, orientation
                )
                if result is not None:
                    valid_adjs.append(result)

            if not valid_adjs:
                # Fallback: at minimum the original DAG is valid.
                valid_adjs.append(dag.adjacency)

            indices = rng.integers(0, len(valid_adjs), size=n_samples)
            return [DAG(valid_adjs[idx]) for idx in indices]

        # ------ Rejection sampling for large MECs ---------------------
        max_attempts = n_samples * 200
        samples: list[DAG] = []
        attempts = 0

        while len(samples) < n_samples and attempts < max_attempts:
            orientation = rng.integers(0, 2, size=n_undirected).tolist()
            result = self._orientation_is_acyclic(
                n, directed, undirected, tuple(orientation)
            )
            if result is not None:
                samples.append(DAG(result))
            attempts += 1

        if len(samples) < n_samples:
            raise RuntimeError(
                f"Rejection sampling could only find {len(samples)} valid "
                f"orientations in {max_attempts} attempts (requested "
                f"{n_samples})."
            )

        return samples

    def skeleton_distance(self, dag1: DAG, dag2: DAG) -> int:
        """Count edge differences between undirected skeletons.

        The skeleton of a DAG is the undirected graph obtained by
        ignoring edge directions.  This method returns the number of
        edges that appear in one skeleton but not the other (symmetric
        difference).

        Only the upper triangle of each symmetric skeleton matrix is
        compared so that each undirected edge is counted once.

        Parameters
        ----------
        dag1 : DAG
            First directed acyclic graph.
        dag2 : DAG
            Second directed acyclic graph.  Must have the same number of
            nodes as *dag1*.

        Returns
        -------
        int
            Number of edges in the symmetric difference of the two
            skeletons.  Zero means the skeletons are identical.

        Raises
        ------
        ValueError
            If the two DAGs have different numbers of nodes.
        """
        if dag1.n_nodes != dag2.n_nodes:
            raise ValueError(
                f"Node count mismatch: {dag1.n_nodes} vs {dag2.n_nodes}"
            )

        skel1 = dag1.skeleton()
        skel2 = dag2.skeleton()
        n = dag1.n_nodes

        distance = 0
        for i in range(n):
            for j in range(i + 1, n):
                if skel1[i, j] != skel2[i, j]:
                    distance += 1

        return distance

    def v_structure_distance(self, dag1: DAG, dag2: DAG) -> int:
        """Size of the symmetric difference of v-structure sets.

        A v-structure (unshielded collider) is a triple ``(i, j, k)``
        with ``i → j ← k`` and ``i``, ``k`` not adjacent.  This method
        computes the number of v-structures that appear in one DAG but
        not the other.

        Parameters
        ----------
        dag1 : DAG
            First directed acyclic graph.
        dag2 : DAG
            Second directed acyclic graph.  Must have the same number of
            nodes as *dag1*.

        Returns
        -------
        int
            ``|V1 △ V2|`` where ``V1``, ``V2`` are the v-structure sets
            of *dag1* and *dag2* respectively.  Zero means both DAGs
            share exactly the same v-structures.

        Raises
        ------
        ValueError
            If the two DAGs have different numbers of nodes.
        """
        if dag1.n_nodes != dag2.n_nodes:
            raise ValueError(
                f"Node count mismatch: {dag1.n_nodes} vs {dag2.n_nodes}"
            )

        vs1 = set(self.find_v_structures(dag1))
        vs2 = set(self.find_v_structures(dag2))
        return len(vs1.symmetric_difference(vs2))
