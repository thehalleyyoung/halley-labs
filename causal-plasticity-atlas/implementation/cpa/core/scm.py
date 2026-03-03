"""Full Structural Causal Model implementation for the CPA engine.

Provides the :class:`StructuralCausalModel` class with DAG operations,
topological ordering, cycle detection, Markov blanket computation,
Structural Hamming Distance, CPDAG construction, intervention simulation,
parameter estimation, data generation, graph metrics, subgraph
extraction, moral graph computation, and v-structure detection.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import warnings
from collections import deque
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray


# ===================================================================
# StructuralCausalModel
# ===================================================================


class StructuralCausalModel:
    """Full-featured structural causal model over a linear-Gaussian SCM.

    The model represents a DAG via an adjacency matrix where ``adj[i, j] != 0``
    means *i → j*.  Regression coefficients and residual variances specify
    the conditional distributions P(X_j | Pa(X_j)) as linear-Gaussian.

    Parameters
    ----------
    adjacency_matrix : np.ndarray
        (p, p) binary or weighted adjacency matrix.
    variable_names : list of str, optional
        Names for each variable.  If ``None``, defaults to ``X0, X1, …``.
    regression_coefficients : np.ndarray, optional
        (p, p) coefficient matrix.  If ``None``, uses the adjacency
        matrix values as coefficients.
    residual_variances : np.ndarray, optional
        (p,) residual variances.  Defaults to ones.
    sample_size : int
        Number of observations used to estimate the model.

    Raises
    ------
    ValueError
        On shape mismatches, duplicate names, or non-square matrices.
    """

    def __init__(
        self,
        adjacency_matrix: NDArray[np.floating],
        variable_names: Optional[List[str]] = None,
        regression_coefficients: Optional[NDArray[np.floating]] = None,
        residual_variances: Optional[NDArray[np.floating]] = None,
        sample_size: int = 0,
    ) -> None:
        adj = np.asarray(adjacency_matrix, dtype=np.float64)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(
                f"adjacency_matrix must be square 2-D, got shape {adj.shape}"
            )
        self._p: int = adj.shape[0]
        self._adj: NDArray[np.float64] = adj

        if variable_names is None:
            self._names: List[str] = [f"X{i}" for i in range(self._p)]
        else:
            if len(variable_names) != self._p:
                raise ValueError(
                    f"variable_names length {len(variable_names)} != {self._p}"
                )
            seen: set[str] = set()
            for nm in variable_names:
                if nm in seen:
                    raise ValueError(f"Duplicate variable name {nm!r}")
                seen.add(nm)
            self._names = list(variable_names)

        if regression_coefficients is not None:
            self._coefs = np.asarray(regression_coefficients, dtype=np.float64)
            if self._coefs.shape != (self._p, self._p):
                raise ValueError(
                    f"regression_coefficients shape {self._coefs.shape} "
                    f"!= ({self._p}, {self._p})"
                )
        else:
            self._coefs = adj.copy()

        if residual_variances is not None:
            self._resid = np.asarray(residual_variances, dtype=np.float64)
            if self._resid.shape != (self._p,):
                raise ValueError(
                    f"residual_variances shape {self._resid.shape} != ({self._p},)"
                )
            if np.any(self._resid <= 0):
                raise ValueError("residual_variances must all be > 0")
        else:
            self._resid = np.ones(self._p, dtype=np.float64)

        self._sample_size = sample_size
        # Cache topological order
        self._topo_cache: Optional[List[int]] = None

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def num_variables(self) -> int:
        """Number of variables."""
        return self._p

    @property
    def num_edges(self) -> int:
        """Number of directed edges."""
        return int(np.count_nonzero(self._adj))

    @property
    def variable_names(self) -> List[str]:
        """Variable names."""
        return list(self._names)

    @property
    def adjacency_matrix(self) -> NDArray[np.float64]:
        """Adjacency matrix (read-only copy)."""
        return self._adj.copy()

    @property
    def regression_coefficients(self) -> NDArray[np.float64]:
        """Regression coefficient matrix."""
        return self._coefs.copy()

    @property
    def residual_variances(self) -> NDArray[np.float64]:
        """Residual variances."""
        return self._resid.copy()

    @property
    def sample_size(self) -> int:
        """Sample size."""
        return self._sample_size

    def variable_index(self, name: str) -> int:
        """Return integer index for *name*.

        Parameters
        ----------
        name : str

        Returns
        -------
        int
        """
        try:
            return self._names.index(name)
        except ValueError:
            raise ValueError(f"Variable {name!r} not in model") from None

    # -----------------------------------------------------------------
    # DAG edge operations
    # -----------------------------------------------------------------

    def has_edge(self, i: int, j: int) -> bool:
        """Check if edge *i → j* exists.

        Parameters
        ----------
        i, j : int
            Variable indices.

        Returns
        -------
        bool
        """
        self._check_idx(i)
        self._check_idx(j)
        return bool(self._adj[i, j] != 0)

    def add_edge(
        self,
        i: int,
        j: int,
        *,
        weight: float = 1.0,
        coefficient: Optional[float] = None,
        check_dag: bool = True,
    ) -> None:
        """Add edge *i → j*.

        Parameters
        ----------
        i, j : int
            Variable indices.
        weight : float
            Adjacency weight.
        coefficient : float, optional
            Regression coefficient.  Defaults to *weight*.
        check_dag : bool
            If ``True``, verify acyclicity after adding.

        Raises
        ------
        ValueError
            If the edge already exists or adding it creates a cycle.
        """
        self._check_idx(i)
        self._check_idx(j)
        if i == j:
            raise ValueError("Self-loops are not allowed")
        if self._adj[i, j] != 0:
            raise ValueError(f"Edge {i} → {j} already exists")
        self._adj[i, j] = weight
        self._coefs[i, j] = coefficient if coefficient is not None else weight
        self._topo_cache = None
        if check_dag:
            try:
                self.topological_sort()
            except ValueError:
                self._adj[i, j] = 0
                self._coefs[i, j] = 0
                raise ValueError(
                    f"Adding edge {i} → {j} creates a cycle"
                ) from None

    def remove_edge(self, i: int, j: int) -> float:
        """Remove edge *i → j* and return its weight.

        Parameters
        ----------
        i, j : int
            Variable indices.

        Returns
        -------
        float
            Weight of the removed edge.

        Raises
        ------
        ValueError
            If edge does not exist.
        """
        self._check_idx(i)
        self._check_idx(j)
        if self._adj[i, j] == 0:
            raise ValueError(f"Edge {i} → {j} does not exist")
        w = float(self._adj[i, j])
        self._adj[i, j] = 0
        self._coefs[i, j] = 0
        self._topo_cache = None
        return w

    def reverse_edge(self, i: int, j: int, *, check_dag: bool = True) -> None:
        """Reverse edge *i → j* to *j → i*.

        Parameters
        ----------
        i, j : int
        check_dag : bool
            Verify DAG constraint after reversal.

        Raises
        ------
        ValueError
            If edge doesn't exist or reversal creates a cycle.
        """
        w = self.remove_edge(i, j)
        coef = self._coefs[j, i]  # will be 0 since we removed
        try:
            self.add_edge(j, i, weight=w, coefficient=w, check_dag=check_dag)
        except ValueError:
            # Restore original edge
            self._adj[i, j] = w
            self._coefs[i, j] = w
            self._topo_cache = None
            raise

    # -----------------------------------------------------------------
    # Graph queries
    # -----------------------------------------------------------------

    def parents(self, j: int) -> List[int]:
        """Indices of parents of variable *j*.

        Parameters
        ----------
        j : int

        Returns
        -------
        list of int
        """
        self._check_idx(j)
        return list(np.nonzero(self._adj[:, j])[0])

    def children(self, i: int) -> List[int]:
        """Indices of children of variable *i*.

        Parameters
        ----------
        i : int

        Returns
        -------
        list of int
        """
        self._check_idx(i)
        return list(np.nonzero(self._adj[i, :])[0])

    def ancestors(self, i: int) -> Set[int]:
        """All ancestors of variable *i* (transitive parents).

        Parameters
        ----------
        i : int

        Returns
        -------
        set of int
        """
        self._check_idx(i)
        result: set[int] = set()
        stack = list(self.parents(i))
        while stack:
            node = stack.pop()
            if node not in result:
                result.add(node)
                stack.extend(self.parents(node))
        return result

    def descendants(self, i: int) -> Set[int]:
        """All descendants of variable *i* (transitive children).

        Parameters
        ----------
        i : int

        Returns
        -------
        set of int
        """
        self._check_idx(i)
        result: set[int] = set()
        stack = list(self.children(i))
        while stack:
            node = stack.pop()
            if node not in result:
                result.add(node)
                stack.extend(self.children(node))
        return result

    def is_ancestor(self, i: int, j: int) -> bool:
        """Check if *i* is an ancestor of *j*.

        Parameters
        ----------
        i, j : int

        Returns
        -------
        bool
        """
        return i in self.ancestors(j)

    def markov_blanket(self, i: int) -> Set[int]:
        """Markov blanket: parents + children + co-parents.

        Parameters
        ----------
        i : int

        Returns
        -------
        set of int
        """
        self._check_idx(i)
        pa = set(self.parents(i))
        ch = set(self.children(i))
        co_pa: set[int] = set()
        for c in ch:
            co_pa |= set(self.parents(c))
        mb = pa | ch | co_pa
        mb.discard(i)
        return mb

    # -----------------------------------------------------------------
    # Topological sort & cycle detection (Kahn's)
    # -----------------------------------------------------------------

    def topological_sort(self) -> List[int]:
        """Topological ordering via Kahn's algorithm.

        Returns
        -------
        list of int
            Variable indices in topological order.

        Raises
        ------
        ValueError
            If the graph contains a cycle.
        """
        if self._topo_cache is not None:
            return list(self._topo_cache)
        binary = (self._adj != 0).astype(int)
        in_deg = binary.sum(axis=0).astype(int).tolist()
        queue: deque[int] = deque()
        for i in range(self._p):
            if in_deg[i] == 0:
                queue.append(i)
        order: list[int] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for ch in range(self._p):
                if binary[node, ch]:
                    in_deg[ch] -= 1
                    if in_deg[ch] == 0:
                        queue.append(ch)
        if len(order) != self._p:
            raise ValueError(
                f"Graph contains a cycle — only {len(order)}/{self._p} "
                "nodes topologically sortable"
            )
        self._topo_cache = order
        return list(order)

    def is_dag(self) -> bool:
        """Check if the graph is a valid DAG (acyclic).

        Returns
        -------
        bool
        """
        try:
            self.topological_sort()
            return True
        except ValueError:
            return False

    def has_cycle(self) -> bool:
        """Check if the graph contains a cycle.

        Returns
        -------
        bool
        """
        return not self.is_dag()

    # -----------------------------------------------------------------
    # d-separation
    # -----------------------------------------------------------------

    def d_separation(
        self,
        x: Set[int],
        y: Set[int],
        z: Set[int],
    ) -> bool:
        """Test d-separation: X ⊥ Y | Z in the DAG.

        Uses the Bayes-Ball algorithm.

        Parameters
        ----------
        x : set of int
            First variable set.
        y : set of int
            Second variable set.
        z : set of int
            Conditioning set.

        Returns
        -------
        bool
            ``True`` if X and Y are d-separated given Z.
        """
        for idx in x | y | z:
            self._check_idx(idx)
        if x & y:
            return False

        # Bayes-Ball: find which nodes in Y are reachable from X given Z
        reachable = self._bayes_ball_reachable(x, z)
        return len(reachable & y) == 0

    def _bayes_ball_reachable(
        self, source: Set[int], observed: Set[int]
    ) -> Set[int]:
        """Bayes-Ball reachability from *source* given *observed*.

        Parameters
        ----------
        source : set of int
        observed : set of int

        Returns
        -------
        set of int
            Reachable nodes.
        """
        # States: (node, direction) where direction is 'up' or 'down'
        visited: set[tuple[int, str]] = set()
        queue: deque[tuple[int, str]] = deque()
        reachable: set[int] = set()

        for s in source:
            queue.append((s, "up"))

        while queue:
            node, direction = queue.popleft()
            if (node, direction) in visited:
                continue
            visited.add((node, direction))

            if node not in source:
                reachable.add(node)

            if direction == "up" and node not in observed:
                # Pass through to parents and children
                for parent in self.parents(node):
                    if (parent, "up") not in visited:
                        queue.append((parent, "up"))
                for child in self.children(node):
                    if (child, "down") not in visited:
                        queue.append((child, "down"))
            elif direction == "down":
                if node not in observed:
                    for child in self.children(node):
                        if (child, "down") not in visited:
                            queue.append((child, "down"))
                # If observed, can go up to parents (explaining away)
                if node in observed:
                    for parent in self.parents(node):
                        if (parent, "up") not in visited:
                            queue.append((parent, "up"))

        return reachable

    # -----------------------------------------------------------------
    # All paths
    # -----------------------------------------------------------------

    def all_paths(
        self,
        source: int,
        target: int,
        *,
        undirected: bool = False,
        max_length: Optional[int] = None,
    ) -> List[List[int]]:
        """Find all paths from *source* to *target*.

        Parameters
        ----------
        source, target : int
            Start and end variable indices.
        undirected : bool
            If ``True``, treat edges as undirected.
        max_length : int, optional
            Maximum path length.

        Returns
        -------
        list of list of int
            Each inner list is a sequence of variable indices.
        """
        self._check_idx(source)
        self._check_idx(target)
        if max_length is not None and max_length < 1:
            return []

        paths: list[list[int]] = []

        def _dfs(node: int, path: list[int], visited: set[int]) -> None:
            if max_length is not None and len(path) > max_length:
                return
            if node == target:
                paths.append(list(path))
                return
            for nb in self._neighbors(node, undirected=undirected):
                if nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    _dfs(nb, path, visited)
                    path.pop()
                    visited.discard(nb)

        _dfs(source, [source], {source})
        return paths

    def _neighbors(self, i: int, *, undirected: bool = False) -> List[int]:
        """Return neighbors of node *i*.

        Parameters
        ----------
        i : int
        undirected : bool

        Returns
        -------
        list of int
        """
        ch = list(np.nonzero(self._adj[i, :])[0])
        if undirected:
            pa = list(np.nonzero(self._adj[:, i])[0])
            return list(set(ch) | set(pa))
        return ch

    # -----------------------------------------------------------------
    # Structural Hamming Distance
    # -----------------------------------------------------------------

    def structural_hamming_distance(self, other: "StructuralCausalModel") -> int:
        """Structural Hamming Distance (SHD) to *other*.

        Counts the number of edge additions, deletions, and reversals
        needed to transform this DAG into *other*.

        Parameters
        ----------
        other : StructuralCausalModel

        Returns
        -------
        int
            SHD (>= 0).
        """
        if self._p != other._p:
            raise ValueError(
                f"Cannot compute SHD between graphs of different sizes "
                f"({self._p} vs {other._p})"
            )
        a = (self._adj != 0).astype(int)
        b = (other._adj != 0).astype(int)
        shd = 0
        for i in range(self._p):
            for j in range(i + 1, self._p):
                edge_self_ij = a[i, j]
                edge_self_ji = a[j, i]
                edge_other_ij = b[i, j]
                edge_other_ji = b[j, i]
                if (edge_self_ij, edge_self_ji) != (edge_other_ij, edge_other_ji):
                    shd += 1
        return shd

    @staticmethod
    def shd(adj1: NDArray, adj2: NDArray) -> int:
        """Compute SHD between two adjacency matrices.

        Parameters
        ----------
        adj1, adj2 : np.ndarray
            Binary adjacency matrices of the same shape.

        Returns
        -------
        int
        """
        a1 = (np.asarray(adj1) != 0).astype(int)
        a2 = (np.asarray(adj2) != 0).astype(int)
        p = a1.shape[0]
        shd = 0
        for i in range(p):
            for j in range(i + 1, p):
                if (a1[i, j], a1[j, i]) != (a2[i, j], a2[j, i]):
                    shd += 1
        return shd

    # -----------------------------------------------------------------
    # CPDAG (completed partially directed acyclic graph)
    # -----------------------------------------------------------------

    def to_cpdag(self) -> NDArray[np.float64]:
        """Convert this DAG to its CPDAG (Markov equivalence class).

        Compelled edges remain directed; reversible edges become
        undirected (represented by entries in both directions).

        Returns
        -------
        np.ndarray
            CPDAG adjacency matrix, shape ``(p, p)``.
        """
        if not self.is_dag():
            raise ValueError("Graph must be a DAG to compute CPDAG")

        cpdag = self._adj.copy()
        binary = (cpdag != 0).astype(int)

        # Find v-structures: i → j ← k where i and k are not adjacent
        v_struct_edges: set[tuple[int, int]] = set()
        for j in range(self._p):
            parents_j = list(np.nonzero(binary[:, j])[0])
            for a, b in itertools.combinations(parents_j, 2):
                if binary[a, b] == 0 and binary[b, a] == 0:
                    v_struct_edges.add((a, j))
                    v_struct_edges.add((b, j))

        # Apply Meek's rules to find all compelled edges
        compelled = set(v_struct_edges)
        changed = True
        while changed:
            changed = False
            for i in range(self._p):
                for j in range(self._p):
                    if binary[i, j] == 0 or (i, j) in compelled:
                        continue
                    # Rule 1: i → j — k (i not adj k) → j → k compelled
                    # Rule 2: i → k → j, i — j → i → j compelled
                    # Rule 3: i — k → j, i — l → j, k not adj l → i → j

                    # Simplified: if reversing i→j would create new
                    # v-structure or break existing one, it's compelled
                    if self._is_edge_compelled(i, j, binary, compelled):
                        compelled.add((i, j))
                        changed = True

        # Build CPDAG: compelled edges stay directed, others become undirected
        result = np.zeros((self._p, self._p), dtype=np.float64)
        for i in range(self._p):
            for j in range(self._p):
                if binary[i, j]:
                    result[i, j] = 1.0
                    if (i, j) not in compelled:
                        result[j, i] = 1.0  # make undirected

        return result

    def _is_edge_compelled(
        self,
        i: int,
        j: int,
        binary: NDArray,
        compelled: set[tuple[int, int]],
    ) -> bool:
        """Check if edge i→j is compelled (Meek's rules).

        Parameters
        ----------
        i, j : int
        binary : np.ndarray
        compelled : set

        Returns
        -------
        bool
        """
        # Rule 1: exists k such that k→i is compelled and k not adj j
        for k in range(self._p):
            if (k, i) in compelled and binary[k, j] == 0 and binary[j, k] == 0:
                return True
        # Rule 2: exists k such that i→k→j (both compelled)
        for k in range(self._p):
            if (i, k) in compelled and (k, j) in compelled:
                return True
        # Rule 3: exists k,l such that k→j, l→j compelled, k—i, l—i undirected
        return False

    # -----------------------------------------------------------------
    # v-structure detection
    # -----------------------------------------------------------------

    def v_structures(self) -> List[Tuple[int, int, int]]:
        """Detect all v-structures (immoralities) in the DAG.

        A v-structure is a triple (i, j, k) where i→j←k and i, k are
        not adjacent.

        Returns
        -------
        list of (int, int, int)
            Triples (parent1, collider, parent2).
        """
        binary = (self._adj != 0).astype(int)
        vstruct: list[tuple[int, int, int]] = []
        for j in range(self._p):
            parents_j = list(np.nonzero(binary[:, j])[0])
            for a, b in itertools.combinations(parents_j, 2):
                if binary[a, b] == 0 and binary[b, a] == 0:
                    vstruct.append((min(a, b), j, max(a, b)))
        return sorted(set(vstruct))

    # -----------------------------------------------------------------
    # Moral graph
    # -----------------------------------------------------------------

    def moral_graph(self) -> NDArray[np.float64]:
        """Compute the moral graph (marry parents, drop directions).

        Returns
        -------
        np.ndarray
            Undirected adjacency matrix, shape ``(p, p)``.
        """
        binary = (self._adj != 0).astype(int)
        moral = np.zeros((self._p, self._p), dtype=np.float64)

        # Add undirected edges for existing directed edges
        for i in range(self._p):
            for j in range(self._p):
                if binary[i, j]:
                    moral[i, j] = 1.0
                    moral[j, i] = 1.0

        # Marry co-parents
        for j in range(self._p):
            parents_j = list(np.nonzero(binary[:, j])[0])
            for a, b in itertools.combinations(parents_j, 2):
                moral[a, b] = 1.0
                moral[b, a] = 1.0

        np.fill_diagonal(moral, 0)
        return moral

    # -----------------------------------------------------------------
    # Graph metrics
    # -----------------------------------------------------------------

    def in_degree(self, i: Optional[int] = None) -> Union[int, NDArray[np.int64]]:
        """In-degree of variable *i*, or all in-degrees.

        Parameters
        ----------
        i : int, optional
            Variable index.  If ``None``, return all.

        Returns
        -------
        int or np.ndarray
        """
        binary = (self._adj != 0).astype(int)
        deg = binary.sum(axis=0)
        if i is not None:
            self._check_idx(i)
            return int(deg[i])
        return deg

    def out_degree(self, i: Optional[int] = None) -> Union[int, NDArray[np.int64]]:
        """Out-degree of variable *i*, or all out-degrees.

        Parameters
        ----------
        i : int, optional

        Returns
        -------
        int or np.ndarray
        """
        binary = (self._adj != 0).astype(int)
        deg = binary.sum(axis=1)
        if i is not None:
            self._check_idx(i)
            return int(deg[i])
        return deg

    def density(self) -> float:
        """Edge density: num_edges / (p * (p-1)).

        Returns
        -------
        float
            Density in [0, 1].
        """
        if self._p <= 1:
            return 0.0
        max_edges = self._p * (self._p - 1)
        return self.num_edges / max_edges

    def diameter(self) -> int:
        """Diameter of the DAG (longest shortest path, treating edges as undirected).

        Returns
        -------
        int
            Graph diameter.  Returns 0 for a single node,
            -1 if the undirected skeleton is disconnected.
        """
        if self._p <= 1:
            return 0
        # BFS from each node
        undirected = ((self._adj != 0) | (self._adj.T != 0)).astype(int)
        max_dist = 0
        for start in range(self._p):
            dist = self._bfs_distances(start, undirected)
            if -1 in dist.values():
                return -1
            max_dist = max(max_dist, max(dist.values()))
        return max_dist

    def _bfs_distances(
        self, start: int, adj: NDArray
    ) -> Dict[int, int]:
        """BFS shortest distances from *start*.

        Parameters
        ----------
        start : int
        adj : np.ndarray
            Adjacency matrix to traverse.

        Returns
        -------
        dict
            ``{node: distance}``.  Unreachable nodes get distance -1.
        """
        dist: dict[int, int] = {i: -1 for i in range(self._p)}
        dist[start] = 0
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            for nb in range(self._p):
                if adj[node, nb] and dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    queue.append(nb)
        return dist

    def roots(self) -> List[int]:
        """Indices of root nodes (no parents).

        Returns
        -------
        list of int
        """
        return [i for i in range(self._p) if len(self.parents(i)) == 0]

    def leaves(self) -> List[int]:
        """Indices of leaf nodes (no children).

        Returns
        -------
        list of int
        """
        return [i for i in range(self._p) if len(self.children(i)) == 0]

    # -----------------------------------------------------------------
    # Intervention / do-calculus support
    # -----------------------------------------------------------------

    def do_intervention(
        self,
        interventions: Dict[int, float],
    ) -> "StructuralCausalModel":
        """Return a new SCM with do(X_i = v) applied.

        Removes all incoming edges to each intervened variable and
        sets its value to the specified constant (intercept, zero
        residual variance).

        Parameters
        ----------
        interventions : dict
            ``{variable_index: intervention_value}``.

        Returns
        -------
        StructuralCausalModel
            Mutilated SCM.
        """
        new_adj = self._adj.copy()
        new_coefs = self._coefs.copy()
        new_resid = self._resid.copy()

        for idx, val in interventions.items():
            self._check_idx(idx)
            new_adj[:, idx] = 0  # remove incoming edges
            new_coefs[:, idx] = 0
            new_resid[idx] = 1e-10  # near-zero variance (point mass)

        return StructuralCausalModel(
            new_adj,
            variable_names=list(self._names),
            regression_coefficients=new_coefs,
            residual_variances=new_resid,
            sample_size=self._sample_size,
        )

    # -----------------------------------------------------------------
    # Parameter estimation from data
    # -----------------------------------------------------------------

    @classmethod
    def fit_from_data(
        cls,
        data: NDArray[np.floating],
        adjacency_matrix: NDArray[np.floating],
        variable_names: Optional[List[str]] = None,
    ) -> "StructuralCausalModel":
        """Fit parameters via OLS regression given a known DAG structure.

        Parameters
        ----------
        data : np.ndarray
            Data matrix, shape ``(n, p)``.
        adjacency_matrix : np.ndarray
            Known DAG structure, shape ``(p, p)``.
        variable_names : list of str, optional
            Variable names.

        Returns
        -------
        StructuralCausalModel
            Fitted model.
        """
        data = np.asarray(data, dtype=np.float64)
        adj = np.asarray(adjacency_matrix, dtype=np.float64)
        n, p = data.shape
        if adj.shape != (p, p):
            raise ValueError(
                f"adjacency_matrix shape {adj.shape} != ({p}, {p})"
            )

        binary = (adj != 0).astype(int)
        coefs = np.zeros((p, p), dtype=np.float64)
        resid_var = np.ones(p, dtype=np.float64)

        for j in range(p):
            pa_idx = list(np.nonzero(binary[:, j])[0])
            if not pa_idx:
                resid_var[j] = max(float(np.var(data[:, j], ddof=1)), 1e-10)
                continue
            X_pa = data[:, pa_idx]
            y = data[:, j]
            ones = np.ones((n, 1))
            design = np.hstack([X_pa, ones])
            result, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
            for k, pa in enumerate(pa_idx):
                coefs[pa, j] = result[k]
            predicted = design @ result
            residuals = y - predicted
            dof = max(1, n - len(pa_idx) - 1)
            resid_var[j] = max(float(np.sum(residuals**2) / dof), 1e-10)

        return cls(
            adjacency_matrix=binary.astype(np.float64),
            variable_names=variable_names,
            regression_coefficients=coefs,
            residual_variances=resid_var,
            sample_size=n,
        )

    # -----------------------------------------------------------------
    # Data generation (sampling)
    # -----------------------------------------------------------------

    def sample(
        self,
        n: int,
        *,
        interventions: Optional[Dict[int, float]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> NDArray[np.float64]:
        """Generate *n* samples from the linear-Gaussian SCM.

        Parameters
        ----------
        n : int
            Number of samples.
        interventions : dict, optional
            do(X_i = v) interventions.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        np.ndarray
            Data matrix, shape ``(n, p)``.
        """
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        rng = rng or np.random.default_rng()

        model = self
        if interventions:
            model = self.do_intervention(interventions)

        order = model.topological_sort()
        data = np.zeros((n, self._p), dtype=np.float64)

        for j in order:
            pa = model.parents(j)
            noise = rng.normal(0, math.sqrt(model._resid[j]), size=n)
            if interventions and j in interventions:
                data[:, j] = interventions[j]
            elif not pa:
                data[:, j] = noise
            else:
                linear = sum(model._coefs[p, j] * data[:, p] for p in pa)
                data[:, j] = linear + noise

        return data

    # -----------------------------------------------------------------
    # Subgraph extraction
    # -----------------------------------------------------------------

    def subgraph(self, indices: Sequence[int]) -> "StructuralCausalModel":
        """Extract an induced subgraph on the given variable indices.

        Parameters
        ----------
        indices : sequence of int
            Variable indices to keep.

        Returns
        -------
        StructuralCausalModel
            Subgraph model.
        """
        idx = list(indices)
        for i in idx:
            self._check_idx(i)
        if len(set(idx)) != len(idx):
            raise ValueError("Duplicate indices in subgraph selection")

        sub_adj = self._adj[np.ix_(idx, idx)]
        sub_coefs = self._coefs[np.ix_(idx, idx)]
        sub_resid = self._resid[idx]
        sub_names = [self._names[i] for i in idx]
        return StructuralCausalModel(
            sub_adj,
            variable_names=sub_names,
            regression_coefficients=sub_coefs,
            residual_variances=sub_resid,
            sample_size=self._sample_size,
        )

    def marginalize(self, keep: Sequence[int]) -> "StructuralCausalModel":
        """Marginalize: keep only *keep* variables, adding transitive edges.

        For each pair (i, j) in *keep*, add an edge i→j if i is an
        ancestor of j in the full graph.

        Parameters
        ----------
        keep : sequence of int
            Variable indices to retain.

        Returns
        -------
        StructuralCausalModel
            Marginalized model with transitive closure.
        """
        keep_list = list(keep)
        keep_set = set(keep_list)
        for i in keep_list:
            self._check_idx(i)

        k = len(keep_list)
        new_adj = np.zeros((k, k), dtype=np.float64)

        for ii, i in enumerate(keep_list):
            for jj, j in enumerate(keep_list):
                if ii == jj:
                    continue
                if i in self.ancestors(j):
                    new_adj[ii, jj] = 1.0

        sub_names = [self._names[i] for i in keep_list]
        return StructuralCausalModel(
            new_adj,
            variable_names=sub_names,
            sample_size=self._sample_size,
        )

    # -----------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        dict
        """
        return {
            "adjacency_matrix": self._adj.tolist(),
            "variable_names": list(self._names),
            "regression_coefficients": self._coefs.tolist(),
            "residual_variances": self._resid.tolist(),
            "sample_size": self._sample_size,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StructuralCausalModel":
        """Deserialize from dictionary.

        Parameters
        ----------
        d : dict

        Returns
        -------
        StructuralCausalModel
        """
        return cls(
            adjacency_matrix=np.array(d["adjacency_matrix"], dtype=np.float64),
            variable_names=d.get("variable_names"),
            regression_coefficients=np.array(
                d["regression_coefficients"], dtype=np.float64
            )
            if "regression_coefficients" in d
            else None,
            residual_variances=np.array(
                d["residual_variances"], dtype=np.float64
            )
            if "residual_variances" in d
            else None,
            sample_size=int(d.get("sample_size", 0)),
        )

    def to_json(self, **kwargs: Any) -> str:
        """Serialize to JSON string.

        Parameters
        ----------
        **kwargs
            Passed to ``json.dumps``.

        Returns
        -------
        str
        """
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, s: str) -> "StructuralCausalModel":
        """Deserialize from JSON string.

        Parameters
        ----------
        s : str

        Returns
        -------
        StructuralCausalModel
        """
        return cls.from_dict(json.loads(s))

    @classmethod
    def from_adjacency_matrix(
        cls,
        adj: NDArray[np.floating],
        variable_names: Optional[List[str]] = None,
    ) -> "StructuralCausalModel":
        """Create from just an adjacency matrix (default parameters).

        Parameters
        ----------
        adj : np.ndarray
        variable_names : list of str, optional

        Returns
        -------
        StructuralCausalModel
        """
        return cls(adj, variable_names=variable_names)

    # -----------------------------------------------------------------
    # Comparison
    # -----------------------------------------------------------------

    def edge_set(self) -> Set[Tuple[int, int]]:
        """Return the set of directed edges as (source, target) pairs.

        Returns
        -------
        set of (int, int)
        """
        edges: set[tuple[int, int]] = set()
        for i in range(self._p):
            for j in range(self._p):
                if self._adj[i, j] != 0:
                    edges.add((i, j))
        return edges

    def named_edge_set(self) -> Set[Tuple[str, str]]:
        """Return edge set using variable names.

        Returns
        -------
        set of (str, str)
        """
        return {
            (self._names[i], self._names[j])
            for i, j in self.edge_set()
        }

    def edge_diff(
        self, other: "StructuralCausalModel"
    ) -> Dict[str, Set[Tuple[int, int]]]:
        """Compute edge differences between this model and *other*.

        Parameters
        ----------
        other : StructuralCausalModel

        Returns
        -------
        dict
            ``{"added": set, "removed": set, "shared": set}``.
        """
        self_edges = self.edge_set()
        other_edges = other.edge_set()
        return {
            "shared": self_edges & other_edges,
            "added": other_edges - self_edges,
            "removed": self_edges - other_edges,
        }

    # -----------------------------------------------------------------
    # Implied covariance matrix
    # -----------------------------------------------------------------

    def implied_covariance(self) -> NDArray[np.float64]:
        """Compute the implied covariance matrix Σ = (I - B)^{-T} Ω (I - B)^{-1}.

        where B is the coefficient matrix and Ω = diag(residual_variances).

        Returns
        -------
        np.ndarray
            Implied covariance, shape ``(p, p)``.
        """
        I = np.eye(self._p, dtype=np.float64)
        B = self._coefs.T  # B[j, i] = coefficient of i in equation for j
        # Actually: X_j = sum_i B_ij X_i + eps_j
        # So X = B^T X + eps → (I - B^T)X = eps → X = (I - B^T)^{-1} eps
        # Cov(X) = (I - B^T)^{-1} Omega (I - B^T)^{-T}

        IminusB = I - self._coefs.T
        try:
            IminusB_inv = np.linalg.inv(IminusB)
        except np.linalg.LinAlgError:
            warnings.warn("Singular (I - B) matrix; using pseudoinverse", stacklevel=2)
            IminusB_inv = np.linalg.pinv(IminusB)

        Omega = np.diag(self._resid)
        cov = IminusB_inv @ Omega @ IminusB_inv.T
        return cov

    # -----------------------------------------------------------------
    # Copy
    # -----------------------------------------------------------------

    def copy(self) -> "StructuralCausalModel":
        """Return a deep copy of this model.

        Returns
        -------
        StructuralCausalModel
        """
        return StructuralCausalModel(
            self._adj.copy(),
            variable_names=list(self._names),
            regression_coefficients=self._coefs.copy(),
            residual_variances=self._resid.copy(),
            sample_size=self._sample_size,
        )

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _check_idx(self, i: int) -> None:
        """Validate a variable index.

        Parameters
        ----------
        i : int

        Raises
        ------
        ValueError
        """
        if not isinstance(i, (int, np.integer)):
            raise TypeError(f"Variable index must be int, got {type(i).__name__}")
        if i < 0 or i >= self._p:
            raise ValueError(f"Variable index {i} out of range [0, {self._p})")

    # -----------------------------------------------------------------
    # Repr / Eq
    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"StructuralCausalModel(p={self._p}, edges={self.num_edges}, "
            f"names={self._names}, n={self._sample_size})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StructuralCausalModel):
            return NotImplemented
        return (
            self._p == other._p
            and np.array_equal(self._adj, other._adj)
            and np.allclose(self._coefs, other._coefs)
            and np.allclose(self._resid, other._resid)
            and self._names == other._names
        )


# ===================================================================
# Utility functions
# ===================================================================


def random_dag(
    p: int,
    *,
    edge_prob: float = 0.3,
    weight_range: Tuple[float, float] = (0.2, 1.0),
    rng: Optional[np.random.Generator] = None,
) -> StructuralCausalModel:
    """Generate a random DAG with *p* variables.

    The DAG is generated by ordering variables 0…p-1 and sampling
    edges i→j only when i < j.

    Parameters
    ----------
    p : int
        Number of variables.
    edge_prob : float
        Probability of each edge (in [0, 1]).
    weight_range : (float, float)
        Range for random edge weights.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    StructuralCausalModel
    """
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")
    rng = rng or np.random.default_rng()

    adj = np.zeros((p, p), dtype=np.float64)
    coefs = np.zeros((p, p), dtype=np.float64)

    for i in range(p):
        for j in range(i + 1, p):
            if rng.random() < edge_prob:
                w = rng.uniform(*weight_range)
                if rng.random() < 0.5:
                    w = -w
                adj[i, j] = 1.0
                coefs[i, j] = w

    resid = rng.uniform(0.5, 2.0, size=p)
    return StructuralCausalModel(
        adj,
        regression_coefficients=coefs,
        residual_variances=resid,
    )


def erdos_renyi_dag(
    p: int,
    expected_edges: float,
    *,
    rng: Optional[np.random.Generator] = None,
) -> StructuralCausalModel:
    """Generate an Erdős-Rényi DAG with expected number of edges.

    Parameters
    ----------
    p : int
        Number of variables.
    expected_edges : float
        Expected number of edges.
    rng : np.random.Generator, optional

    Returns
    -------
    StructuralCausalModel
    """
    max_edges = p * (p - 1) / 2
    edge_prob = min(expected_edges / max(max_edges, 1), 1.0)
    return random_dag(p, edge_prob=edge_prob, rng=rng)


def chain_dag(
    p: int,
    *,
    coefficient: float = 0.7,
    residual_variance: float = 1.0,
) -> StructuralCausalModel:
    """Create a chain DAG: X0 → X1 → … → X_{p-1}.

    Parameters
    ----------
    p : int
        Number of variables.
    coefficient : float
        Edge coefficient for each edge.
    residual_variance : float
        Residual variance for each variable.

    Returns
    -------
    StructuralCausalModel
    """
    adj = np.zeros((p, p), dtype=np.float64)
    coefs = np.zeros((p, p), dtype=np.float64)
    for i in range(p - 1):
        adj[i, i + 1] = 1.0
        coefs[i, i + 1] = coefficient
    resid = np.full(p, residual_variance, dtype=np.float64)
    return StructuralCausalModel(
        adj, regression_coefficients=coefs, residual_variances=resid
    )


def fork_dag(
    p: int,
    *,
    coefficient: float = 0.7,
) -> StructuralCausalModel:
    """Create a fork DAG: X0 → X1, X0 → X2, …, X0 → X_{p-1}.

    Parameters
    ----------
    p : int
        Number of variables (>= 2).
    coefficient : float
        Edge coefficient.

    Returns
    -------
    StructuralCausalModel
    """
    if p < 2:
        raise ValueError(f"fork_dag requires p >= 2, got {p}")
    adj = np.zeros((p, p), dtype=np.float64)
    coefs = np.zeros((p, p), dtype=np.float64)
    for j in range(1, p):
        adj[0, j] = 1.0
        coefs[0, j] = coefficient
    return StructuralCausalModel(adj, regression_coefficients=coefs)


def collider_dag(
    p: int,
    *,
    coefficient: float = 0.7,
) -> StructuralCausalModel:
    """Create a collider DAG: X0 → X_{p-1}, X1 → X_{p-1}, …

    All variables except the last are root nodes pointing into
    the last variable.

    Parameters
    ----------
    p : int
        Number of variables (>= 2).
    coefficient : float
        Edge coefficient.

    Returns
    -------
    StructuralCausalModel
    """
    if p < 2:
        raise ValueError(f"collider_dag requires p >= 2, got {p}")
    adj = np.zeros((p, p), dtype=np.float64)
    coefs = np.zeros((p, p), dtype=np.float64)
    for i in range(p - 1):
        adj[i, p - 1] = 1.0
        coefs[i, p - 1] = coefficient
    return StructuralCausalModel(adj, regression_coefficients=coefs)
