"""Optimized descriptor computation using vectorized numpy operations.

This module provides high-performance alternatives to the standard
descriptor classes, with:

* Vectorized degree and structural property computation
* Matrix-based v-structure detection
* BFS-layer batch path length computation
* Vectorized MI computation using correlation/precision matrices
* Pre-fitted PCA transformation with SIMD-friendly memory layout

Classes
-------
* :class:`FastStructuralDescriptor` – vectorized structural features
* :class:`FastInfoTheoreticDescriptor` – vectorized info-theoretic features
* :class:`FastCompositeDescriptor` – combined descriptors with fast PCA
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

try:
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def _wrapper(fn):  # type: ignore[no-untyped-def]
            return fn
        if args and callable(args[0]):
            return args[0]
        return _wrapper

    prange = range  # type: ignore[assignment,misc]

from causal_qd.types import AdjacencyMatrix, BehavioralDescriptor, DataMatrix

__all__ = [
    "FastStructuralDescriptor",
    "FastInfoTheoreticDescriptor",
    "FastCompositeDescriptor",
    "batch_structural_descriptors",
    "batch_info_theoretic_descriptors",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VARIANCE_FLOOR: float = 1e-12
_REGULARIZATION_EPS: float = 1e-8

STRUCTURAL_FEATURES: List[str] = [
    "edge_density",
    "max_in_degree",
    "v_structure_count",
    "longest_path",
    "avg_markov_blanket",
    "dag_depth",
    "parent_set_entropy",
    "clustering_coefficient",
    "avg_path_length",
    "connected_components",
]

INFO_THEORETIC_FEATURES: List[str] = [
    "mean_mi",
    "std_mi",
    "min_mi",
    "max_mi",
    "mean_partial_corr",
    "std_partial_corr",
    "mean_conditional_entropy",
    "std_conditional_entropy",
]


# ---------------------------------------------------------------------------
# Numba kernels for structural descriptors
# ---------------------------------------------------------------------------


@njit(cache=True)  # type: ignore[misc]
def _jit_longest_path(adj: np.ndarray) -> int:
    """JIT-compiled longest path via topological sort + DP."""
    n = adj.shape[0]
    if n == 0:
        return 0

    # Kahn's topological sort
    in_deg = np.zeros(n, dtype=np.int64)
    for j in range(n):
        for i in range(n):
            if adj[i, j] != 0:
                in_deg[j] += 1

    queue = np.empty(n, dtype=np.int64)
    head = 0
    tail = 0
    for i in range(n):
        if in_deg[i] == 0:
            queue[tail] = i
            tail += 1

    order = np.empty(n, dtype=np.int64)
    count = 0
    while head < tail:
        u = queue[head]
        head += 1
        order[count] = u
        count += 1
        for v in range(n):
            if adj[u, v] != 0:
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    queue[tail] = v
                    tail += 1

    if count != n:
        return 0

    dist = np.zeros(n, dtype=np.int64)
    for idx in range(count):
        u = order[idx]
        for v in range(n):
            if adj[u, v] != 0:
                if dist[v] < dist[u] + 1:
                    dist[v] = dist[u] + 1

    max_d = 0
    for i in range(n):
        if dist[i] > max_d:
            max_d = dist[i]
    return max_d


@njit(cache=True)  # type: ignore[misc]
def _jit_v_structure_count(adj: np.ndarray) -> int:
    """JIT-compiled v-structure counting."""
    n = adj.shape[0]
    count = 0
    for j in range(n):
        parents = []
        for i in range(n):
            if adj[i, j] != 0:
                parents.append(i)
        np_pa = len(parents)
        for a in range(np_pa):
            for b in range(a + 1, np_pa):
                i = parents[a]
                k = parents[b]
                if adj[i, k] == 0 and adj[k, i] == 0:
                    count += 1
    return count


@njit(cache=True)  # type: ignore[misc]
def _jit_avg_path_length(adj: np.ndarray) -> float:
    """JIT-compiled average shortest directed path length via BFS."""
    n = adj.shape[0]
    if n <= 1:
        return 0.0

    total_length = 0.0
    total_pairs = 0

    for src in range(n):
        dist = np.full(n, -1, dtype=np.int64)
        dist[src] = 0
        queue = np.empty(n, dtype=np.int64)
        head = 0
        tail = 0
        queue[tail] = src
        tail += 1

        while head < tail:
            u = queue[head]
            head += 1
            for v in range(n):
                if adj[u, v] != 0 and dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue[tail] = v
                    tail += 1

        for v in range(n):
            if v != src and dist[v] > 0:
                total_length += dist[v]
                total_pairs += 1

    if total_pairs == 0:
        return 0.0
    return total_length / total_pairs


# ---------------------------------------------------------------------------
# FastStructuralDescriptor
# ---------------------------------------------------------------------------


class FastStructuralDescriptor:
    """Vectorized structural descriptor computation.

    Uses numpy vectorization and Numba JIT for all structural feature
    computations on DAG adjacency matrices.

    Parameters
    ----------
    features : list of str, optional
        Which features to compute.  Defaults to all available features.

    Examples
    --------
    >>> desc = FastStructuralDescriptor(["edge_density", "max_in_degree"])
    >>> desc.compute(adj).shape
    (2,)
    """

    def __init__(self, features: Optional[List[str]] = None) -> None:
        self._features = features or STRUCTURAL_FEATURES[:6]
        for f in self._features:
            if f not in STRUCTURAL_FEATURES:
                raise ValueError(f"Unknown structural feature: {f}")

    @property
    def descriptor_dim(self) -> int:
        """Number of descriptor dimensions."""
        return len(self._features)

    @property
    def descriptor_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """All features are normalized to [0, 1]."""
        return (
            np.zeros(self.descriptor_dim),
            np.ones(self.descriptor_dim),
        )

    def compute(
        self,
        dag: AdjacencyMatrix,
        data: Optional[DataMatrix] = None,
    ) -> BehavioralDescriptor:
        """Compute structural descriptors for a DAG.

        Parameters
        ----------
        dag : np.ndarray
            ``(n, n)`` adjacency matrix.
        data : np.ndarray, optional
            Not used for structural descriptors.

        Returns
        -------
        np.ndarray
            Descriptor vector of length ``descriptor_dim``.
        """
        adj = np.asarray(dag, dtype=np.int8)
        n = adj.shape[0]
        result = np.zeros(self.descriptor_dim, dtype=np.float64)

        # Pre-compute skeleton once for features that need it
        needs_skeleton = {"clustering_coefficient", "connected_components"}
        skeleton: Optional[np.ndarray] = None
        if needs_skeleton & set(self._features):
            skeleton = adj | adj.T

        for idx, feat in enumerate(self._features):
            result[idx] = self._compute_feature(feat, adj, n, skeleton)

        return result

    def _compute_feature(
        self,
        feature: str,
        adj: np.ndarray,
        n: int,
        skeleton: Optional[np.ndarray] = None,
    ) -> float:
        """Dispatch to individual feature computation."""
        if n <= 1:
            return 0.0

        if feature == "edge_density":
            return self._edge_density(adj, n)
        elif feature == "max_in_degree":
            return self._max_in_degree(adj, n)
        elif feature == "v_structure_count":
            return self._v_structure_count_feat(adj, n)
        elif feature == "longest_path":
            return self._longest_path_feat(adj, n)
        elif feature == "avg_markov_blanket":
            return self._avg_markov_blanket(adj, n)
        elif feature == "dag_depth":
            return self._dag_depth(adj, n)
        elif feature == "parent_set_entropy":
            return self._parent_set_entropy(adj, n)
        elif feature == "clustering_coefficient":
            return self._clustering_coefficient(adj, n, skeleton)
        elif feature == "avg_path_length":
            return self._avg_path_length_feat(adj, n)
        elif feature == "connected_components":
            return self._connected_components(adj, n, skeleton)
        return 0.0

    # -- Individual features (all return values in [0, 1]) ------------------

    @staticmethod
    def _edge_density(adj: np.ndarray, n: int) -> float:
        """Edge density = num_edges / (n*(n-1))."""
        max_edges = n * (n - 1)
        return float(np.sum(adj)) / max_edges if max_edges > 0 else 0.0

    @staticmethod
    def _max_in_degree(adj: np.ndarray, n: int) -> float:
        """Max in-degree normalized by (n-1)."""
        in_deg = adj.sum(axis=0)
        return float(np.max(in_deg)) / (n - 1) if n > 1 else 0.0

    @staticmethod
    def _v_structure_count_feat(adj: np.ndarray, n: int) -> float:
        """V-structure count normalized by C(n,2)."""
        count = _jit_v_structure_count(adj)
        max_possible = n * (n - 1) * (n - 2) // 6  # rough upper bound
        if max_possible <= 0:
            return 0.0
        return min(float(count) / max(max_possible, 1), 1.0)

    @staticmethod
    def _longest_path_feat(adj: np.ndarray, n: int) -> float:
        """Longest path normalized by (n-1)."""
        lp = _jit_longest_path(adj)
        return float(lp) / (n - 1) if n > 1 else 0.0

    @staticmethod
    def _avg_markov_blanket(adj: np.ndarray, n: int) -> float:
        """Average Markov blanket size normalized by (n-1).

        MB(X) = Pa(X) ∪ Ch(X) ∪ Pa(Ch(X)) \\ {X}

        Vectorized: parents = adj.T, children = adj,
        co-parents of children = (adj @ adj.T) > 0.
        """
        adj_bool = adj.astype(np.bool_)
        # adj.T[i,j] = adj[j,i] = 1 means j is parent of i
        # adj[i,j] = 1 means j is child of i
        # (adj @ adj.T)[i,j] > 0 means i and j share a child (co-parents)
        coparent = adj.astype(np.int32) @ adj.astype(np.int32).T
        mb_matrix = adj_bool.T | adj_bool | (coparent > 0)
        np.fill_diagonal(mb_matrix, False)
        avg = float(np.mean(mb_matrix.sum(axis=1)))
        return min(avg / (n - 1), 1.0) if n > 1 else 0.0

    @staticmethod
    def _dag_depth(adj: np.ndarray, n: int) -> float:
        """Number of topological layers normalized by n."""
        in_deg = adj.sum(axis=0).astype(np.int64)
        remaining = np.ones(n, dtype=np.bool_)
        n_layers = 0
        processed = 0

        while processed < n:
            layer_mask = (in_deg == 0) & remaining
            layer = np.nonzero(layer_mask)[0]
            if len(layer) == 0:
                break
            n_layers += 1
            processed += len(layer)
            for u in layer:
                remaining[u] = False
                children_idx = np.nonzero(adj[u])[0]
                in_deg[children_idx] -= 1

        return float(n_layers) / n if n > 0 else 0.0

    @staticmethod
    def _parent_set_entropy(adj: np.ndarray, n: int) -> float:
        """Entropy of parent set size distribution, normalized."""
        in_deg = adj.sum(axis=0).astype(np.int64)
        max_in = int(np.max(in_deg)) + 1
        counts = np.bincount(in_deg, minlength=max_in).astype(np.float64)
        probs = counts / n
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs + 1e-30))
        max_entropy = np.log(n) if n > 1 else 1.0
        return float(entropy / max_entropy)

    @staticmethod
    def _clustering_coefficient(
        adj: np.ndarray, n: int, skeleton: Optional[np.ndarray] = None
    ) -> float:
        """Average local clustering coefficient on the skeleton."""
        if skeleton is None:
            skeleton = adj | adj.T
        total_cc = 0.0
        counted = 0

        for v in range(n):
            neighbors = np.nonzero(skeleton[v])[0]
            k = len(neighbors)
            if k < 2:
                continue
            # Count edges among neighbors
            sub = skeleton[np.ix_(neighbors, neighbors)]
            links = (np.sum(sub) - np.trace(sub)) / 2
            total_cc += 2.0 * links / (k * (k - 1))
            counted += 1

        return total_cc / counted if counted > 0 else 0.0

    @staticmethod
    def _avg_path_length_feat(adj: np.ndarray, n: int) -> float:
        """Average directed path length, normalized by (n-1)."""
        avg = _jit_avg_path_length(adj)
        return min(avg / (n - 1), 1.0) if n > 1 else 0.0

    @staticmethod
    def _connected_components(
        adj: np.ndarray, n: int, skeleton: Optional[np.ndarray] = None
    ) -> float:
        """Number of weakly connected components, normalized by n."""
        if skeleton is None:
            skeleton = adj | adj.T
        visited = np.zeros(n, dtype=np.bool_)
        n_comp = 0
        for start in range(n):
            if visited[start]:
                continue
            n_comp += 1
            stack = [start]
            visited[start] = True
            while stack:
                u = stack.pop()
                for v in np.nonzero(skeleton[u])[0]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(int(v))
        return float(n_comp) / n if n > 0 else 0.0

    # -- Batch computation --------------------------------------------------

    def batch_compute(
        self,
        dags: Sequence[AdjacencyMatrix],
        data: Optional[DataMatrix] = None,
    ) -> np.ndarray:
        """Compute descriptors for a batch of DAGs.

        Parameters
        ----------
        dags : sequence of np.ndarray
            List of adjacency matrices.
        data : np.ndarray, optional
            Not used.

        Returns
        -------
        np.ndarray
            ``(batch_size, descriptor_dim)`` array.
        """
        return batch_structural_descriptors(list(dags), features=self._features)


# ---------------------------------------------------------------------------
# FastInfoTheoreticDescriptor
# ---------------------------------------------------------------------------


class FastInfoTheoreticDescriptor:
    """Vectorized information-theoretic descriptor computation.

    Uses correlation and precision matrices for fast MI and partial
    correlation computation, avoiding repeated pairwise calculations.

    Parameters
    ----------
    features : list of str, optional
        Which features to compute.  Defaults to MI profile features.

    Notes
    -----
    For Gaussian data, MI between X_i and X_j is:
        MI(X_i; X_j) = -0.5 * log(1 - r_{ij}^2)
    where r_{ij} is the Pearson correlation.

    Partial correlation from the precision matrix P = Sigma^{-1}:
        rho_{ij|rest} = -P_{ij} / sqrt(P_{ii} * P_{jj})
    """

    def __init__(self, features: Optional[List[str]] = None) -> None:
        self._features = features or INFO_THEORETIC_FEATURES[:4]
        for f in self._features:
            if f not in INFO_THEORETIC_FEATURES:
                raise ValueError(f"Unknown info-theoretic feature: {f}")
        self._cached_corr: Optional[np.ndarray] = None
        self._cached_precision: Optional[np.ndarray] = None
        self._cached_data_id: Optional[int] = None
        self._cached_data_shape: Optional[Tuple[int, ...]] = None

    @property
    def descriptor_dim(self) -> int:
        return len(self._features)

    @property
    def descriptor_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.zeros(self.descriptor_dim),
            np.ones(self.descriptor_dim),
        )

    def _precompute(self, data: DataMatrix) -> None:
        """Pre-compute correlation and precision matrices."""
        data_id = id(data)
        data_shape = data.shape
        if self._cached_data_id == data_id and self._cached_data_shape == data_shape:
            return

        N, p = data.shape
        # Correlation matrix
        stds = data.std(axis=0)
        stds = np.where(stds < _VARIANCE_FLOOR, 1.0, stds)
        centered = (data - data.mean(axis=0)) / stds
        self._cached_corr = (centered.T @ centered) / N
        np.fill_diagonal(self._cached_corr, 1.0)

        # Precision matrix (regularized inverse of correlation)
        reg = self._cached_corr + _REGULARIZATION_EPS * np.eye(p)
        try:
            self._cached_precision = np.linalg.inv(reg)
        except np.linalg.LinAlgError:
            self._cached_precision = np.linalg.pinv(reg)

        self._cached_data_id = data_id
        self._cached_data_shape = data_shape

    def _mi_matrix(self) -> np.ndarray:
        """Mutual information matrix from correlations.

        MI(X_i; X_j) = -0.5 * log(1 - r_{ij}^2)
        """
        corr = self._cached_corr
        r_sq = np.clip(corr ** 2, 0, 1 - 1e-15)  # type: ignore[union-attr]
        mi = -0.5 * np.log(1 - r_sq)
        np.fill_diagonal(mi, 0.0)
        return mi

    def _partial_corr_matrix(self) -> np.ndarray:
        """Partial correlation matrix from precision matrix.

        rho_{ij|rest} = -P_{ij} / sqrt(P_{ii} * P_{jj})
        """
        P = self._cached_precision
        diag = np.sqrt(np.abs(np.diag(P)) + _VARIANCE_FLOOR)  # type: ignore[union-attr]
        pcorr = -P / np.outer(diag, diag)  # type: ignore[operator]
        np.fill_diagonal(pcorr, 1.0)
        return np.clip(pcorr, -1, 1)

    def compute(
        self,
        dag: AdjacencyMatrix,
        data: Optional[DataMatrix] = None,
    ) -> BehavioralDescriptor:
        """Compute info-theoretic descriptors.

        Parameters
        ----------
        dag : np.ndarray
            ``(n, n)`` adjacency matrix.
        data : np.ndarray
            ``(N, p)`` data matrix (required).

        Returns
        -------
        np.ndarray
            Descriptor vector.
        """
        if data is None:
            return np.zeros(self.descriptor_dim, dtype=np.float64)

        self._precompute(data)
        adj = np.asarray(dag, dtype=np.int8)
        n = adj.shape[0]

        result = np.zeros(self.descriptor_dim, dtype=np.float64)

        # Compute MI for edges in the DAG
        mi_mat = self._mi_matrix()
        edge_mask = adj.astype(np.bool_)
        edge_mis = mi_mat[edge_mask]

        # Compute partial correlations for edges
        pcorr_mat = self._partial_corr_matrix()
        edge_pcorrs = np.abs(pcorr_mat[edge_mask])

        # Conditional entropies: H(X_j | Pa(X_j))
        cond_entropies = self._conditional_entropies(adj, data)

        for idx, feat in enumerate(self._features):
            if feat == "mean_mi":
                result[idx] = np.mean(edge_mis) if len(edge_mis) > 0 else 0.0
            elif feat == "std_mi":
                result[idx] = np.std(edge_mis) if len(edge_mis) > 1 else 0.0
            elif feat == "min_mi":
                result[idx] = np.min(edge_mis) if len(edge_mis) > 0 else 0.0
            elif feat == "max_mi":
                result[idx] = np.max(edge_mis) if len(edge_mis) > 0 else 0.0
            elif feat == "mean_partial_corr":
                result[idx] = (
                    np.mean(edge_pcorrs) if len(edge_pcorrs) > 0 else 0.0
                )
            elif feat == "std_partial_corr":
                result[idx] = (
                    np.std(edge_pcorrs) if len(edge_pcorrs) > 1 else 0.0
                )
            elif feat == "mean_conditional_entropy":
                result[idx] = np.mean(cond_entropies)
            elif feat == "std_conditional_entropy":
                result[idx] = np.std(cond_entropies)

        # Normalize to [0, 1] using sigmoid-like transformation
        result = np.clip(result, 0, None)
        max_val = np.max(result) if np.any(result > 0) else 1.0
        if max_val > 0:
            result = result / (result + max_val)

        return result

    def _conditional_entropies(
        self, adj: np.ndarray, data: DataMatrix
    ) -> np.ndarray:
        """Compute conditional entropy H(X_j | Pa(X_j)) for each node.

        For Gaussian data: H(X|Pa) = 0.5 * log(2*pi*e * var(X|Pa))
        """
        N, p = data.shape
        entropies = np.zeros(p, dtype=np.float64)

        for j in range(p):
            parents = np.nonzero(adj[:, j])[0]
            if len(parents) == 0:
                var_j = np.var(data[:, j])
            else:
                X_pa = data[:, parents]
                y_j = data[:, j]
                # Regress j on parents
                try:
                    beta, residuals, _, _ = np.linalg.lstsq(
                        X_pa, y_j, rcond=None
                    )
                    pred = X_pa @ beta
                    var_j = np.var(y_j - pred)
                except np.linalg.LinAlgError:
                    var_j = np.var(y_j)

            var_j = max(var_j, _VARIANCE_FLOOR)
            entropies[j] = 0.5 * np.log(2 * np.pi * np.e * var_j)

        return entropies

    def batch_compute(
        self,
        dags: Sequence[AdjacencyMatrix],
        data: Optional[DataMatrix] = None,
    ) -> np.ndarray:
        """Compute descriptors for a batch of DAGs."""
        batch_size = len(dags)
        result = np.empty(
            (batch_size, self.descriptor_dim), dtype=np.float64
        )
        for i, dag in enumerate(dags):
            result[i] = self.compute(dag, data)
        return result


# ---------------------------------------------------------------------------
# FastCompositeDescriptor
# ---------------------------------------------------------------------------


class FastCompositeDescriptor:
    """Composite descriptor combining multiple sub-descriptors with fast PCA.

    Concatenates outputs from multiple descriptor computers, applies optional
    normalization and PCA dimensionality reduction, all with SIMD-friendly
    contiguous memory layout.

    Parameters
    ----------
    descriptors : list
        Sub-descriptor instances (must have ``compute()`` and ``descriptor_dim``).
    weights : np.ndarray, optional
        Per-descriptor weights for weighting before PCA.
    normalization : str
        Normalization method: ``"none"``, ``"zscore"``, ``"minmax"``.
    pca_dim : int, optional
        If set, reduce to this many dimensions via PCA.

    Examples
    --------
    >>> struct = FastStructuralDescriptor(["edge_density", "max_in_degree"])
    >>> info = FastInfoTheoreticDescriptor(["mean_mi", "std_mi"])
    >>> composite = FastCompositeDescriptor([struct, info], pca_dim=2)
    >>> composite.fit([dag1, dag2, ...], data)
    >>> desc = composite.compute(dag, data)
    """

    def __init__(
        self,
        descriptors: Sequence[Any],
        weights: Optional[np.ndarray] = None,
        normalization: str = "none",
        pca_dim: Optional[int] = None,
    ) -> None:
        self._descriptors = list(descriptors)
        self._normalization = normalization
        self._pca_dim = pca_dim

        total_dim = sum(d.descriptor_dim for d in self._descriptors)
        self._raw_dim = total_dim

        if weights is not None:
            self._weights = np.asarray(weights, dtype=np.float64)
        else:
            self._weights = np.ones(total_dim, dtype=np.float64)

        # Normalization stats (fitted)
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._min: Optional[np.ndarray] = None
        self._max: Optional[np.ndarray] = None

        # PCA components (fitted)
        self._pca_components: Optional[np.ndarray] = None
        self._pca_mean: Optional[np.ndarray] = None
        self._fitted = False

    @property
    def descriptor_dim(self) -> int:
        """Dimension of the output descriptor."""
        if self._pca_dim is not None and self._fitted:
            return self._pca_dim
        return self._raw_dim

    @property
    def descriptor_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.zeros(self.descriptor_dim),
            np.ones(self.descriptor_dim),
        )

    def fit(
        self,
        dags: Sequence[AdjacencyMatrix],
        data: Optional[DataMatrix] = None,
    ) -> None:
        """Fit normalization and PCA from a sample of DAGs.

        Parameters
        ----------
        dags : sequence of np.ndarray
            Sample of DAGs to fit statistics from.
        data : np.ndarray, optional
            Data matrix passed to sub-descriptors.
        """
        if len(dags) < 2:
            warnings.warn("Need at least 2 DAGs to fit; skipping fit.")
            return

        # Compute raw descriptors for all DAGs
        raw = np.empty((len(dags), self._raw_dim), dtype=np.float64)
        for i, dag in enumerate(dags):
            raw[i] = self._compute_raw(dag, data)

        # Fit normalization
        self._mean = np.nanmean(raw, axis=0)
        self._std = np.nanstd(raw, axis=0)
        self._std[self._std < _VARIANCE_FLOOR] = 1.0
        self._min = np.nanmin(raw, axis=0)
        self._max = np.nanmax(raw, axis=0)
        range_vals = self._max - self._min
        range_vals[range_vals < _VARIANCE_FLOOR] = 1.0
        self._max = self._min + range_vals

        # Normalize
        normalized = self._normalize(raw)

        # Fit PCA
        if self._pca_dim is not None:
            self._fit_pca(normalized)

        self._fitted = True

    def _compute_raw(
        self, dag: AdjacencyMatrix, data: Optional[DataMatrix]
    ) -> np.ndarray:
        """Concatenate all sub-descriptor outputs."""
        parts: List[np.ndarray] = []
        for desc in self._descriptors:
            d = desc.compute(dag, data)
            parts.append(np.asarray(d, dtype=np.float64))
        raw = np.concatenate(parts)
        return raw * self._weights[: len(raw)]

    def _normalize(self, raw: np.ndarray) -> np.ndarray:
        """Apply normalization to raw descriptor matrix."""
        if self._normalization == "none":
            return raw
        elif self._normalization == "zscore":
            if self._mean is None:
                return raw
            return (raw - self._mean) / self._std  # type: ignore[operator]
        elif self._normalization == "minmax":
            if self._min is None:
                return raw
            return (raw - self._min) / (self._max - self._min)  # type: ignore[operator]
        return raw

    def _normalize_single(self, raw: np.ndarray) -> np.ndarray:
        """Apply normalization to a single descriptor vector."""
        if self._normalization == "none":
            return raw
        elif self._normalization == "zscore":
            if self._mean is None:
                return raw
            return (raw - self._mean) / self._std  # type: ignore[operator]
        elif self._normalization == "minmax":
            if self._min is None:
                return raw
            return (raw - self._min) / (self._max - self._min)  # type: ignore[operator]
        return raw

    def _fit_pca(self, data: np.ndarray) -> None:
        """Fit PCA components from normalized data."""
        self._pca_mean = np.nanmean(data, axis=0)
        centered = data - self._pca_mean
        # Handle NaNs
        centered = np.nan_to_num(centered, nan=0.0)

        # SVD for PCA
        U, s, Vt = np.linalg.svd(centered, full_matrices=False)
        k = min(self._pca_dim or data.shape[1], Vt.shape[0])  # type: ignore[arg-type]
        self._pca_components = np.ascontiguousarray(Vt[:k])

    def _apply_pca(self, x: np.ndarray) -> np.ndarray:
        """Project onto PCA components."""
        if self._pca_components is None or self._pca_mean is None:
            return x
        centered = x - self._pca_mean
        centered = np.nan_to_num(centered, nan=0.0)
        return centered @ self._pca_components.T

    def compute(
        self,
        dag: AdjacencyMatrix,
        data: Optional[DataMatrix] = None,
    ) -> BehavioralDescriptor:
        """Compute composite descriptor.

        Parameters
        ----------
        dag : np.ndarray
            Adjacency matrix.
        data : np.ndarray, optional
            Data matrix.

        Returns
        -------
        np.ndarray
            Descriptor vector.
        """
        raw = self._compute_raw(dag, data)

        if self._fitted:
            normalized = self._normalize_single(raw)
            if self._pca_dim is not None and self._pca_components is not None:
                result = self._apply_pca(normalized)
            else:
                result = normalized
        else:
            result = raw

        return np.clip(result, 0, 1)

    def batch_compute(
        self,
        dags: Sequence[AdjacencyMatrix],
        data: Optional[DataMatrix] = None,
    ) -> np.ndarray:
        """Compute descriptors for a batch of DAGs.

        Parameters
        ----------
        dags : sequence of np.ndarray
            List of adjacency matrices.
        data : np.ndarray, optional
            Data matrix.

        Returns
        -------
        np.ndarray
            ``(batch_size, descriptor_dim)`` array.
        """
        batch_size = len(dags)
        raw = np.empty((batch_size, self._raw_dim), dtype=np.float64)
        for i, dag in enumerate(dags):
            raw[i] = self._compute_raw(dag, data)

        if self._fitted:
            normalized = self._normalize(raw)
            if self._pca_dim is not None and self._pca_components is not None:
                centered = normalized - self._pca_mean  # type: ignore[operator]
                centered = np.nan_to_num(centered, nan=0.0)
                result = centered @ self._pca_components.T
            else:
                result = normalized
        else:
            result = raw

        return np.clip(result, 0, 1)


# ---------------------------------------------------------------------------
# Batch descriptor functions
# ---------------------------------------------------------------------------


def batch_structural_descriptors(
    dags: list[np.ndarray],
    features: Optional[List[str]] = None,
) -> np.ndarray:
    """Compute structural descriptors for multiple DAGs using vectorized ops.

    Stacks adjacency matrices into a 3D array and computes degree stats,
    density, and other structural features using numpy broadcasting.

    Parameters
    ----------
    dags : list[np.ndarray]
        List of ``(n, n)`` adjacency matrices (all same size).
    features : list[str], optional
        Subset of features.  Defaults to first 6 structural features.

    Returns
    -------
    np.ndarray
        ``(len(dags), descriptor_dim)`` array of descriptors.
    """
    feat_list = features or STRUCTURAL_FEATURES[:6]
    if len(dags) == 0:
        return np.empty((0, len(feat_list)), dtype=np.float64)

    n = dags[0].shape[0]
    batch = len(dags)

    # Stack into (batch, n, n) 3D array
    stacked = np.stack([np.asarray(d, dtype=np.int8) for d in dags], axis=0)

    result = np.zeros((batch, len(feat_list)), dtype=np.float64)

    # Precompute vectorized quantities shared across features
    edge_counts = stacked.reshape(batch, -1).sum(axis=1).astype(np.float64)
    in_degrees = stacked.sum(axis=1)  # (batch, n)
    max_in = in_degrees.max(axis=1).astype(np.float64)  # (batch,)

    # Skeletons for features that need them
    needs_skeleton = {"clustering_coefficient", "connected_components"}
    skeleton = None
    if needs_skeleton & set(feat_list):
        stacked_bool = stacked.astype(np.bool_)
        skeleton = stacked_bool | stacked_bool.transpose(0, 2, 1)

    for fidx, feat in enumerate(feat_list):
        if n <= 1:
            result[:, fidx] = 0.0
            continue

        if feat == "edge_density":
            max_edges = n * (n - 1)
            result[:, fidx] = edge_counts / max_edges if max_edges > 0 else 0.0

        elif feat == "max_in_degree":
            result[:, fidx] = max_in / (n - 1)

        elif feat == "parent_set_entropy":
            for b in range(batch):
                in_deg_b = in_degrees[b].astype(np.int64)
                counts = np.bincount(in_deg_b).astype(np.float64)
                probs = counts / n
                probs = probs[probs > 0]
                entropy = -np.sum(probs * np.log(probs + 1e-30))
                max_ent = np.log(n) if n > 1 else 1.0
                result[b, fidx] = entropy / max_ent

        elif feat == "avg_markov_blanket":
            stacked_bool = stacked.astype(np.bool_)
            stacked_i32 = stacked.astype(np.int32)
            coparent = np.matmul(stacked_i32, stacked_i32.transpose(0, 2, 1))
            mb = stacked_bool.transpose(0, 2, 1) | stacked_bool | (coparent > 0)
            for b in range(batch):
                np.fill_diagonal(mb[b], False)
            avg_mb = mb.sum(axis=2).mean(axis=1).astype(np.float64)
            result[:, fidx] = np.minimum(avg_mb / (n - 1), 1.0)

        else:
            # Fall back to per-DAG computation for complex features
            desc = FastStructuralDescriptor(features=[feat])
            for b in range(batch):
                result[b, fidx] = desc.compute(dags[b])[0]

    return result


def batch_info_theoretic_descriptors(
    dags: list[np.ndarray],
    data: np.ndarray,
    features: Optional[List[str]] = None,
) -> np.ndarray:
    """Compute info-theoretic descriptors for multiple DAGs.

    Precomputes the mutual information matrix once from data and indexes
    it for each DAG, avoiding redundant correlation/MI computation.

    Parameters
    ----------
    dags : list[np.ndarray]
        List of ``(n, n)`` adjacency matrices (all same size).
    data : np.ndarray
        ``(N, p)`` data matrix.
    features : list[str], optional
        Subset of info-theoretic features.

    Returns
    -------
    np.ndarray
        ``(len(dags), descriptor_dim)`` array of descriptors.
    """
    feat_list = features or INFO_THEORETIC_FEATURES[:4]
    if len(dags) == 0:
        return np.empty((0, len(feat_list)), dtype=np.float64)

    batch = len(dags)
    N, p = data.shape

    # Precompute correlation and MI matrices once
    stds = data.std(axis=0)
    stds = np.where(stds < _VARIANCE_FLOOR, 1.0, stds)
    centered = (data - data.mean(axis=0)) / stds
    corr = (centered.T @ centered) / N
    np.fill_diagonal(corr, 1.0)

    r_sq = np.clip(corr ** 2, 0, 1 - 1e-15)
    mi_mat = -0.5 * np.log(1 - r_sq)
    np.fill_diagonal(mi_mat, 0.0)

    # Precompute precision matrix for partial correlations
    reg = corr + _REGULARIZATION_EPS * np.eye(p)
    try:
        precision = np.linalg.inv(reg)
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(reg)
    diag_p = np.sqrt(np.abs(np.diag(precision)) + _VARIANCE_FLOOR)
    pcorr_mat = -precision / np.outer(diag_p, diag_p)
    np.fill_diagonal(pcorr_mat, 1.0)
    pcorr_mat = np.clip(pcorr_mat, -1, 1)

    needs_cond_ent = {"mean_conditional_entropy", "std_conditional_entropy"}
    compute_cond_ent = bool(needs_cond_ent & set(feat_list))

    result = np.zeros((batch, len(feat_list)), dtype=np.float64)

    for b in range(batch):
        adj = np.asarray(dags[b], dtype=np.int8)
        edge_mask = adj.astype(np.bool_)
        edge_mis = mi_mat[edge_mask]
        edge_pcorrs = np.abs(pcorr_mat[edge_mask])

        cond_entropies = None
        if compute_cond_ent:
            cond_entropies = np.zeros(p, dtype=np.float64)
            for j in range(p):
                parents = np.nonzero(adj[:, j])[0]
                if len(parents) == 0:
                    var_j = np.var(data[:, j])
                else:
                    X_pa = data[:, parents]
                    y_j = data[:, j]
                    try:
                        beta = np.linalg.lstsq(X_pa, y_j, rcond=None)[0]
                        var_j = np.var(y_j - X_pa @ beta)
                    except np.linalg.LinAlgError:
                        var_j = np.var(y_j)
                var_j = max(var_j, _VARIANCE_FLOOR)
                cond_entropies[j] = 0.5 * np.log(2 * np.pi * np.e * var_j)

        for fidx, feat in enumerate(feat_list):
            if feat == "mean_mi":
                result[b, fidx] = np.mean(edge_mis) if len(edge_mis) > 0 else 0.0
            elif feat == "std_mi":
                result[b, fidx] = np.std(edge_mis) if len(edge_mis) > 1 else 0.0
            elif feat == "min_mi":
                result[b, fidx] = np.min(edge_mis) if len(edge_mis) > 0 else 0.0
            elif feat == "max_mi":
                result[b, fidx] = np.max(edge_mis) if len(edge_mis) > 0 else 0.0
            elif feat == "mean_partial_corr":
                result[b, fidx] = np.mean(edge_pcorrs) if len(edge_pcorrs) > 0 else 0.0
            elif feat == "std_partial_corr":
                result[b, fidx] = np.std(edge_pcorrs) if len(edge_pcorrs) > 1 else 0.0
            elif feat == "mean_conditional_entropy":
                result[b, fidx] = np.mean(cond_entropies) if cond_entropies is not None else 0.0
            elif feat == "std_conditional_entropy":
                result[b, fidx] = np.std(cond_entropies) if cond_entropies is not None else 0.0

    # Normalize using same sigmoid-like transform as FastInfoTheoreticDescriptor
    result = np.clip(result, 0, None)
    for b in range(batch):
        max_val = np.max(result[b]) if np.any(result[b] > 0) else 1.0
        if max_val > 0:
            result[b] = result[b] / (result[b] + max_val)

    return result
