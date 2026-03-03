"""Markov equivalence class (MEC) behavioral descriptors.

This module provides a rich set of descriptors that capture structural
properties of the Markov equivalence class to which a given DAG belongs.
The descriptors quantify how "constrained" or "flexible" a DAG's
equivalence class is by examining its CPDAG (completed partially directed
acyclic graph), v-structures, compelled/reversible edges, and chain
components.

Features
--------
The six default features are:

1. **mec_size** – Log-normalized estimate of the number of DAGs in the
   Markov equivalence class.
2. **compelled_fraction** – Fraction of directed edges that are compelled
   (i.e., oriented the same way in every member of the MEC).
3. **v_structure_density** – Number of unshielded colliders (v-structures)
   normalized by the theoretical maximum ``C(n, 2) * (n - 2)``.
4. **reversible_fraction** – Fraction of directed edges that are reversible
   (i.e., appear as undirected edges in the CPDAG).
5. **cpdag_density** – Edge density of the CPDAG, counting each undirected
   edge once, normalized by ``n * (n - 1) / 2``.
6. **avg_chain_component_size** – Average size of the connected components
   formed by reversible (undirected) edges in the CPDAG, normalized by *n*.

All descriptor values are normalized to the interval ``[0, 1]``.

DAG-to-CPDAG Conversion
------------------------
The conversion uses the standard algorithm:

1. Identify all v-structures (unshielded colliders) and mark their edges
   as compelled.
2. Iteratively apply Meek's orientation rules R1–R4 until convergence to
   orient additional edges that must be compelled in every member of the
   MEC.
3. All remaining edges that could not be oriented are marked as reversible
   (undirected) in the CPDAG.

References
----------
- Meek, C. (1995). "Causal Inference and Causal Explanation with
  Background Knowledge." *UAI*.
- Chickering, D. M. (2002). "Optimal Structure Identification With
  Greedy Search." *JMLR*, 3, 507–554.
- Andersson, S. A., Madigan, D., & Perlman, M. D. (1997).
  "A characterization of Markov equivalence classes for acyclic
  digraphs." *Annals of Statistics*, 25(2), 505–541.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.types import AdjacencyMatrix, BehavioralDescriptor, DataMatrix
from causal_qd.descriptors.descriptor_base import DescriptorComputer

# Default feature set for the equivalence descriptor.
_DEFAULT_FEATURES: list[str] = [
    "mec_size",
    "compelled_fraction",
    "v_structure_density",
    "reversible_fraction",
    "cpdag_density",
    "avg_chain_component_size",
]


class EquivalenceDescriptor(DescriptorComputer):
    """Descriptor based on Markov equivalence class properties.

    Computes a configurable set of features that characterize the CPDAG
    (essential graph) of a DAG and the properties of its Markov
    equivalence class.

    Parameters
    ----------
    features : list[str] | None
        Ordered list of feature names to include in the descriptor
        vector.  Each name must be one of the six supported features
        (see module docstring).  When ``None`` (the default), all six
        features are used in their canonical order.

    Examples
    --------
    >>> import numpy as np
    >>> from causal_qd.descriptors.equivalence_desc import EquivalenceDescriptor
    >>> desc = EquivalenceDescriptor()
    >>> dag = np.array([[0, 1, 0],
    ...                 [0, 0, 1],
    ...                 [0, 0, 0]], dtype=np.int8)
    >>> bd = desc.compute(dag)
    >>> bd.shape
    (6,)
    >>> all(0.0 <= v <= 1.0 for v in bd)
    True
    """

    # Supported feature names mapped to internal computation keys.
    _SUPPORTED_FEATURES: frozenset[str] = frozenset(_DEFAULT_FEATURES)

    def __init__(self, features: list[str] | None = None) -> None:
        """Initialize the equivalence descriptor.

        Parameters
        ----------
        features : list[str] | None
            Ordered list of feature names to include.  Must be a subset
            of ``{"mec_size", "compelled_fraction", "v_structure_density",
            "reversible_fraction", "cpdag_density",
            "avg_chain_component_size"}``.  Defaults to all six features.

        Raises
        ------
        ValueError
            If *features* is empty or contains unsupported names.
        """
        if features is None:
            self._features: list[str] = list(_DEFAULT_FEATURES)
        else:
            if len(features) == 0:
                raise ValueError("features must be a non-empty list")
            unknown = set(features) - self._SUPPORTED_FEATURES
            if unknown:
                raise ValueError(
                    f"Unsupported feature(s): {sorted(unknown)}. "
                    f"Supported: {sorted(self._SUPPORTED_FEATURES)}"
                )
            self._features = list(features)

    # ------------------------------------------------------------------
    # DescriptorComputer interface
    # ------------------------------------------------------------------

    @property
    def descriptor_dim(self) -> int:  # noqa: D401
        """Dimensionality of the descriptor vector.

        Returns the number of selected features (default 6).
        """
        return len(self._features)

    @property
    def descriptor_bounds(
        self,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Per-dimension ``[0, 1]`` bounds.

        All features are normalized to ``[0, 1]``, so bounds are
        simply zero and one vectors of length :pyattr:`descriptor_dim`.
        """
        low = np.zeros(self.descriptor_dim, dtype=np.float64)
        high = np.ones(self.descriptor_dim, dtype=np.float64)
        return low, high

    def compute(
        self, dag: AdjacencyMatrix, data: Optional[DataMatrix] = None
    ) -> BehavioralDescriptor:
        """Compute the equivalence-class descriptor for *dag*.

        Parameters
        ----------
        dag : AdjacencyMatrix
            ``n × n`` binary adjacency matrix of the DAG where
            ``dag[i, j] = 1`` indicates edge ``i → j``.
        data : DataMatrix | None
            Unused.  Accepted for interface compatibility.

        Returns
        -------
        BehavioralDescriptor
            1-D float64 array of length :pyattr:`descriptor_dim` with
            all values in ``[0, 1]``.
        """
        n: int = dag.shape[0]

        # Pre-compute shared intermediate results.
        cpdag = self._dag_to_cpdag(dag, n)

        # Build a mapping from feature name to computed value.
        feature_values: dict[str, float] = {}

        # Lazily compute only what is needed.
        needs_v = "v_structure_density" in self._features
        needs_compelled = "compelled_fraction" in self._features
        needs_reversible = "reversible_fraction" in self._features
        needs_mec = "mec_size" in self._features
        needs_cpdag_dens = "cpdag_density" in self._features
        needs_chain = "avg_chain_component_size" in self._features

        if needs_v:
            v_count = self._count_v_structures(dag, n)
            max_v = (n * (n - 1) // 2) * max(n - 2, 1) if n >= 3 else 1
            feature_values["v_structure_density"] = float(
                np.clip(v_count / max_v, 0.0, 1.0)
            )

        if needs_compelled:
            feature_values["compelled_fraction"] = self._fraction_compelled(
                dag, cpdag, n
            )

        if needs_reversible:
            feature_values["reversible_fraction"] = self._fraction_reversible(
                dag, cpdag, n
            )

        if needs_mec:
            feature_values["mec_size"] = self._estimate_mec_size(cpdag, n)

        if needs_cpdag_dens:
            feature_values["cpdag_density"] = self._cpdag_density(cpdag, n)

        if needs_chain:
            feature_values["avg_chain_component_size"] = (
                self._avg_chain_component_size(cpdag, n)
            )

        result = np.array(
            [feature_values[f] for f in self._features], dtype=np.float64
        )
        return result

    # ------------------------------------------------------------------
    # DAG → CPDAG conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _dag_to_cpdag(dag: AdjacencyMatrix, n: int) -> AdjacencyMatrix:
        """Convert a DAG to its CPDAG (essential graph).

        The algorithm proceeds in two phases:

        1. **V-structure identification** – For every node *j*, each pair
           of parents ``(i, k)`` that are *not* adjacent in the skeleton
           forms a v-structure ``i → j ← k``.  Both edges are marked as
           compelled (directed) in the CPDAG.

        2. **Meek rule propagation** – The four Meek orientation rules
           (R1–R4) are applied iteratively until no further edges can be
           oriented.  Any remaining edge that has not been compelled is
           marked as undirected (reversible) in the CPDAG.

        Parameters
        ----------
        dag : AdjacencyMatrix
            ``n × n`` binary adjacency matrix of the input DAG.
        n : int
            Number of nodes.

        Returns
        -------
        AdjacencyMatrix
            The CPDAG where a directed edge ``i → j`` is represented by
            ``cpdag[i, j] = 1, cpdag[j, i] = 0`` and an undirected edge
            ``i — j`` is represented by ``cpdag[i, j] = 1, cpdag[j, i] = 1``.
        """
        if n == 0:
            return dag.copy()

        # Start with every edge undirected, then compel as needed.
        cpdag = dag.copy().astype(dag.dtype)

        # Skeleton adjacency (symmetric).
        skeleton = np.zeros((n, n), dtype=np.bool_)
        for i in range(n):
            for j in range(n):
                if dag[i, j] or dag[j, i]:
                    skeleton[i, j] = True

        # Track which directed edges in the original DAG are compelled.
        # An edge i→j is compelled iff cpdag[i,j]=1 and cpdag[j,i]=0.
        # We start by making every DAG edge undirected in the CPDAG,
        # then compel edges as we discover constraints.
        for i in range(n):
            for j in range(n):
                if dag[i, j]:
                    cpdag[j, i] = 1  # make undirected initially

        # Phase 1: Mark v-structure edges as compelled.
        for j in range(n):
            parents_j = np.where(dag[:, j])[0]
            num_parents = len(parents_j)
            for idx_a in range(num_parents):
                for idx_b in range(idx_a + 1, num_parents):
                    i = parents_j[idx_a]
                    k = parents_j[idx_b]
                    # Unshielded collider: i and k are NOT adjacent.
                    if not skeleton[i, k]:
                        # Compel i → j: remove j → i from CPDAG.
                        cpdag[j, i] = 0 if dag[i, j] else cpdag[j, i]
                        # Compel k → j: remove j → k from CPDAG.
                        cpdag[j, k] = 0 if dag[k, j] else cpdag[j, k]
                        # Ensure directed entries exist.
                        cpdag[i, j] = 1
                        cpdag[k, j] = 1

        # Phase 2: Meek rules R1–R4 applied iteratively.
        changed = True
        while changed:
            changed = False
            changed |= EquivalenceDescriptor._meek_r1(cpdag, skeleton, n)
            changed |= EquivalenceDescriptor._meek_r2(cpdag, n)
            changed |= EquivalenceDescriptor._meek_r3(cpdag, skeleton, n)
            changed |= EquivalenceDescriptor._meek_r4(cpdag, skeleton, n)

        return cpdag

    # ------------------------------------------------------------------
    # Meek orientation rules
    # ------------------------------------------------------------------

    @staticmethod
    def _is_directed(cpdag: AdjacencyMatrix, i: int, j: int) -> bool:
        """Return True if ``i → j`` is a directed (compelled) edge."""
        return bool(cpdag[i, j]) and not bool(cpdag[j, i])

    @staticmethod
    def _is_undirected(cpdag: AdjacencyMatrix, i: int, j: int) -> bool:
        """Return True if ``i — j`` is an undirected (reversible) edge."""
        return bool(cpdag[i, j]) and bool(cpdag[j, i])

    @staticmethod
    def _orient(cpdag: AdjacencyMatrix, i: int, j: int) -> bool:
        """Orient an undirected edge ``i — j`` as ``i → j``.

        Returns True if the edge was actually oriented (was undirected).
        """
        if cpdag[i, j] and cpdag[j, i]:
            cpdag[j, i] = 0
            return True
        return False

    @staticmethod
    def _meek_r1(
        cpdag: AdjacencyMatrix, skeleton: npt.NDArray[np.bool_], n: int
    ) -> bool:
        """Meek Rule R1: Orient ``j — k`` as ``j → k`` when ``i → j — k``
        and ``i`` is not adjacent to ``k``.

        This prevents the creation of new v-structures.

        Parameters
        ----------
        cpdag : AdjacencyMatrix
            Current CPDAG (modified in place).
        skeleton : ndarray
            Symmetric boolean skeleton adjacency.
        n : int
            Number of nodes.

        Returns
        -------
        bool
            True if any edge was oriented.
        """
        changed = False
        for j in range(n):
            for k in range(n):
                if j == k:
                    continue
                if not EquivalenceDescriptor._is_undirected(cpdag, j, k):
                    continue
                # Look for i → j where i is not adjacent to k.
                for i in range(n):
                    if i == j or i == k:
                        continue
                    if (
                        EquivalenceDescriptor._is_directed(cpdag, i, j)
                        and not skeleton[i, k]
                    ):
                        changed |= EquivalenceDescriptor._orient(cpdag, j, k)
                        break  # edge already oriented
        return changed

    @staticmethod
    def _meek_r2(cpdag: AdjacencyMatrix, n: int) -> bool:
        """Meek Rule R2: Orient ``i — k`` as ``i → k`` when there exists
        a directed path ``i → j → k`` and ``i — k``.

        This prevents the creation of directed cycles.

        Parameters
        ----------
        cpdag : AdjacencyMatrix
            Current CPDAG (modified in place).
        n : int
            Number of nodes.

        Returns
        -------
        bool
            True if any edge was oriented.
        """
        changed = False
        for i in range(n):
            for k in range(n):
                if i == k:
                    continue
                if not EquivalenceDescriptor._is_undirected(cpdag, i, k):
                    continue
                # Look for j such that i → j → k.
                for j in range(n):
                    if j == i or j == k:
                        continue
                    if (
                        EquivalenceDescriptor._is_directed(cpdag, i, j)
                        and EquivalenceDescriptor._is_directed(cpdag, j, k)
                    ):
                        changed |= EquivalenceDescriptor._orient(cpdag, i, k)
                        break
        return changed

    @staticmethod
    def _meek_r3(
        cpdag: AdjacencyMatrix, skeleton: npt.NDArray[np.bool_], n: int
    ) -> bool:
        """Meek Rule R3: Orient ``i — l`` as ``i → l`` when ``i — j``,
        ``i — k``, ``j → l ← k``, and ``i`` is not adjacent to ``l``.

        Parameters
        ----------
        cpdag : AdjacencyMatrix
            Current CPDAG (modified in place).
        skeleton : ndarray
            Symmetric boolean skeleton adjacency.
        n : int
            Number of nodes.

        Returns
        -------
        bool
            True if any edge was oriented.
        """
        changed = False
        for i in range(n):
            for l in range(n):  # noqa: E741
                if i == l:
                    continue
                if not EquivalenceDescriptor._is_undirected(cpdag, i, l):
                    continue
                # Find pairs (j, k) both undirected neighbors of i,
                # with j → l and k → l, and j not adjacent to k.
                undirected_neighbors_i = [
                    x
                    for x in range(n)
                    if x != i
                    and x != l
                    and EquivalenceDescriptor._is_undirected(cpdag, i, x)
                ]
                oriented = False
                for idx_j in range(len(undirected_neighbors_i)):
                    if oriented:
                        break
                    j = undirected_neighbors_i[idx_j]
                    if not EquivalenceDescriptor._is_directed(cpdag, j, l):
                        continue
                    for idx_k in range(idx_j + 1, len(undirected_neighbors_i)):
                        k = undirected_neighbors_i[idx_k]
                        if not EquivalenceDescriptor._is_directed(cpdag, k, l):
                            continue
                        if not skeleton[j, k]:
                            changed |= EquivalenceDescriptor._orient(
                                cpdag, i, l
                            )
                            oriented = True
                            break
        return changed

    @staticmethod
    def _meek_r4(
        cpdag: AdjacencyMatrix, skeleton: npt.NDArray[np.bool_], n: int
    ) -> bool:
        """Meek Rule R4: Orient ``i — j`` as ``i → j`` when ``i — l``,
        ``l → k``, ``k → j``, and ``i — j``.

        Although R4 is not strictly necessary for CPDAG construction
        (R1–R3 suffice for DAG-to-CPDAG), it is included for
        completeness and for use with general PDAG-to-CPDAG extensions.

        Parameters
        ----------
        cpdag : AdjacencyMatrix
            Current CPDAG (modified in place).
        skeleton : ndarray
            Symmetric boolean skeleton adjacency.
        n : int
            Number of nodes.

        Returns
        -------
        bool
            True if any edge was oriented.
        """
        changed = False
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if not EquivalenceDescriptor._is_undirected(cpdag, i, j):
                    continue
                # Find l such that i — l, and k such that l → k → j.
                for l in range(n):  # noqa: E741
                    if l == i or l == j:
                        continue
                    if not EquivalenceDescriptor._is_undirected(cpdag, i, l):
                        continue
                    if not skeleton[i, l]:
                        continue
                    for k in range(n):
                        if k == i or k == j or k == l:
                            continue
                        if (
                            EquivalenceDescriptor._is_directed(cpdag, l, k)
                            and EquivalenceDescriptor._is_directed(cpdag, k, j)
                        ):
                            changed |= EquivalenceDescriptor._orient(
                                cpdag, i, j
                            )
                            break
                    # If we already oriented i→j, stop searching.
                    if EquivalenceDescriptor._is_directed(cpdag, i, j):
                        break
        return changed

    # ------------------------------------------------------------------
    # Feature computation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_v_structures(dag: AdjacencyMatrix, n: int) -> int:
        """Count unshielded colliders (v-structures / immoralities).

        A v-structure (unshielded collider) ``i → j ← k`` exists when
        both ``i`` and ``k`` are parents of ``j`` and there is **no**
        edge between ``i`` and ``k`` in either direction.

        Parameters
        ----------
        dag : AdjacencyMatrix
            ``n × n`` binary adjacency matrix of the DAG.
        n : int
            Number of nodes.

        Returns
        -------
        int
            Number of unshielded colliders.  Each ordered pair ``(i, k)``
            with ``i < k`` is counted once.
        """
        count = 0
        for j in range(n):
            parents = np.where(dag[:, j])[0]
            num_parents = len(parents)
            for idx_i in range(num_parents):
                for idx_k in range(idx_i + 1, num_parents):
                    pi, pk = parents[idx_i], parents[idx_k]
                    if not dag[pi, pk] and not dag[pk, pi]:
                        count += 1
        return count

    @staticmethod
    def _fraction_compelled(
        dag: AdjacencyMatrix, cpdag: AdjacencyMatrix, n: int
    ) -> float:
        """Fraction of DAG edges that are compelled (directed in CPDAG).

        A compelled edge ``i → j`` appears as a directed edge in the
        CPDAG, meaning it has the same orientation in every DAG within
        the Markov equivalence class.

        Parameters
        ----------
        dag : AdjacencyMatrix
            ``n × n`` binary adjacency matrix of the original DAG.
        cpdag : AdjacencyMatrix
            ``n × n`` CPDAG adjacency matrix.
        n : int
            Number of nodes.

        Returns
        -------
        float
            Value in ``[0, 1]``.  Returns ``0.0`` for the empty graph.
        """
        total = int(dag.sum())
        if total == 0:
            return 0.0
        compelled = 0
        for i in range(n):
            for j in range(n):
                if dag[i, j] and not cpdag[j, i]:
                    compelled += 1
        return float(np.clip(compelled / total, 0.0, 1.0))

    @staticmethod
    def _fraction_reversible(
        dag: AdjacencyMatrix, cpdag: AdjacencyMatrix, n: int
    ) -> float:
        """Fraction of DAG edges that are reversible (undirected in CPDAG).

        A reversible edge ``i → j`` appears as an undirected edge
        ``i — j`` in the CPDAG, meaning there exists another DAG in the
        same MEC where the edge has the opposite orientation ``j → i``.

        Parameters
        ----------
        dag : AdjacencyMatrix
            ``n × n`` binary adjacency matrix of the original DAG.
        cpdag : AdjacencyMatrix
            ``n × n`` CPDAG adjacency matrix.
        n : int
            Number of nodes.

        Returns
        -------
        float
            Value in ``[0, 1]``.  Returns ``0.0`` for the empty graph.
            Note that ``compelled_fraction + reversible_fraction == 1.0``.
        """
        total = int(dag.sum())
        if total == 0:
            return 0.0
        reversible = 0
        for i in range(n):
            for j in range(n):
                if dag[i, j] and cpdag[j, i]:
                    reversible += 1
        return float(np.clip(reversible / total, 0.0, 1.0))

    @staticmethod
    def _cpdag_density(cpdag: AdjacencyMatrix, n: int) -> float:
        """Edge density of the CPDAG, counting undirected edges once.

        The density is defined as the number of distinct edges (directed
        edges counted once, undirected edges counted once) divided by the
        maximum possible number of edges ``n * (n - 1) / 2``.

        Parameters
        ----------
        cpdag : AdjacencyMatrix
            ``n × n`` CPDAG adjacency matrix.
        n : int
            Number of nodes.

        Returns
        -------
        float
            Value in ``[0, 1]``.  Returns ``0.0`` for graphs with fewer
            than 2 nodes.
        """
        if n <= 1:
            return 0.0
        max_edges = n * (n - 1) / 2.0

        edge_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if cpdag[i, j] or cpdag[j, i]:
                    edge_count += 1

        return float(np.clip(edge_count / max_edges, 0.0, 1.0))

    @staticmethod
    def _estimate_mec_size(cpdag: AdjacencyMatrix, n: int) -> float:
        """Estimate the size of the Markov equivalence class.

        The estimate is based on the chain components of the CPDAG.  For
        each connected component of reversible (undirected) edges, the
        number of valid DAG orientations is at least ``1`` and at most
        ``2^m`` where ``m`` is the number of undirected edges in that
        component.  We take the product across components to get a rough
        upper-bound estimate of the MEC size.

        The result is log-normalized:

        .. math::

            \\text{descriptor} = \\frac{\\log_2(\\text{estimate})}{n(n-1)/2}

        so that the value lies in ``[0, 1]``.

        Parameters
        ----------
        cpdag : AdjacencyMatrix
            ``n × n`` CPDAG adjacency matrix.
        n : int
            Number of nodes.

        Returns
        -------
        float
            Value in ``[0, 1]``.  Returns ``0.0`` when the MEC contains
            exactly one DAG (all edges compelled).
        """
        if n <= 1:
            return 0.0

        max_log = n * (n - 1) / 2.0  # log2(2^(n*(n-1)/2))
        if max_log == 0.0:
            return 0.0

        # Find chain components (connected components of undirected edges).
        components = EquivalenceDescriptor._find_chain_components(cpdag, n)

        total_log2 = 0.0
        for component_nodes in components:
            if len(component_nodes) <= 1:
                continue
            # Count undirected edges within this component.
            undirected_edges = 0
            comp_list = sorted(component_nodes)
            for idx_i in range(len(comp_list)):
                for idx_j in range(idx_i + 1, len(comp_list)):
                    ni, nj = comp_list[idx_i], comp_list[idx_j]
                    if cpdag[ni, nj] and cpdag[nj, ni]:
                        undirected_edges += 1
            # Upper bound on orientations for this component.
            # A tree on k nodes has exactly k-1 edges and 2^(k-1)
            # orientations; a chordal graph may have fewer valid
            # orientations than 2^m but we use 2^m as an upper bound.
            total_log2 += undirected_edges

        return float(np.clip(total_log2 / max_log, 0.0, 1.0))

    @staticmethod
    def _avg_chain_component_size(cpdag: AdjacencyMatrix, n: int) -> float:
        """Average size of chain components of reversible edges.

        A *chain component* is a maximal connected subgraph induced by
        the undirected (reversible) edges of the CPDAG.  Isolated nodes
        (nodes with no undirected edges) each form a trivial chain
        component of size 1.

        The average component size is normalized by ``n`` so that the
        result lies in ``[0, 1]``.  A value of ``1/n`` indicates all
        components are singletons (fully compelled graph); a value of
        ``1.0`` indicates a single component spanning all nodes.

        Parameters
        ----------
        cpdag : AdjacencyMatrix
            ``n × n`` CPDAG adjacency matrix.
        n : int
            Number of nodes.

        Returns
        -------
        float
            Value in ``[0, 1]``.  Returns ``0.0`` for empty graphs.
        """
        if n == 0:
            return 0.0

        components = EquivalenceDescriptor._find_chain_components(cpdag, n)

        if len(components) == 0:
            return 0.0

        avg_size = sum(len(c) for c in components) / len(components)
        return float(np.clip(avg_size / n, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Graph utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_chain_components(
        cpdag: AdjacencyMatrix, n: int
    ) -> list[set[int]]:
        """Find connected components of undirected (reversible) edges.

        Each node belongs to exactly one chain component.  Nodes with no
        undirected edges form singleton components.

        Parameters
        ----------
        cpdag : AdjacencyMatrix
            ``n × n`` CPDAG adjacency matrix.
        n : int
            Number of nodes.

        Returns
        -------
        list[set[int]]
            List of sets, each containing the node indices of one
            connected component.
        """
        visited = [False] * n
        components: list[set[int]] = []

        for start in range(n):
            if visited[start]:
                continue
            # BFS over undirected edges only.
            component: set[int] = set()
            queue: deque[int] = deque([start])
            visited[start] = True
            while queue:
                node = queue.popleft()
                component.add(node)
                for neighbor in range(n):
                    if not visited[neighbor] and neighbor != node:
                        # Check for undirected edge: both directions present.
                        if cpdag[node, neighbor] and cpdag[neighbor, node]:
                            visited[neighbor] = True
                            queue.append(neighbor)
            components.append(component)

        return components
