"""Advanced behavioral descriptors for causal DAGs.

Provides descriptors based on interventional distributions, causal
effects, spectral properties, and path statistics.  These go beyond
simple structural features to capture the causal semantics of the
DAG.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.descriptors.descriptor_base import DescriptorComputer
from causal_qd.types import AdjacencyMatrix, BehavioralDescriptor, DataMatrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _topological_sort(adj: AdjacencyMatrix) -> List[int]:
    """Kahn's algorithm for topological ordering."""
    n = adj.shape[0]
    in_deg = adj.sum(axis=0).copy()
    queue: deque[int] = deque(i for i in range(n) if in_deg[i] == 0)
    order: List[int] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in range(n):
            if adj[node, child]:
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    queue.append(child)
    return order


def _ancestors(adj: AdjacencyMatrix, node: int) -> set:
    """Return all ancestors of node via BFS."""
    n = adj.shape[0]
    visited: set = set()
    queue = deque(int(i) for i in np.where(adj[:, node])[0])
    while queue:
        cur = queue.popleft()
        if cur not in visited:
            visited.add(cur)
            queue.extend(
                int(i) for i in np.where(adj[:, cur])[0] if i not in visited
            )
    return visited


def _descendants(adj: AdjacencyMatrix, node: int) -> set:
    """Return all descendants of node via BFS."""
    n = adj.shape[0]
    visited: set = set()
    queue = deque(int(i) for i in np.where(adj[node])[0])
    while queue:
        cur = queue.popleft()
        if cur not in visited:
            visited.add(cur)
            queue.extend(
                int(i) for i in np.where(adj[cur])[0] if i not in visited
            )
    return visited


# ---------------------------------------------------------------------------
# InterventionalDescriptor
# ---------------------------------------------------------------------------


class InterventionalDescriptor(DescriptorComputer):
    """Describe interventional distributions implied by a DAG.

    Computes do-calculus-based effects for selected variable pairs.
    Uses truncated factorization to compute the interventional
    distribution under ``do(X=x)`` for each intervention target.

    For a DAG with variables V and data D, the interventional
    distribution P(Y | do(X=x)) is computed using the truncated
    factorization:

        P(Y | do(X=x)) = Σ_{V\\X} Π_{i≠X} P(V_i | Pa(V_i))

    We summarize this as the change in mean/variance of Y under
    a unit intervention on X.

    Parameters
    ----------
    n_pairs : int
        Number of variable pairs to use as descriptor dimensions.
        If the number of nodes is less than this, all pairs are used.
        Default ``5``.
    intervention_value : float
        The value to set for intervention.  Default ``1.0``.
    """

    def __init__(
        self, n_pairs: int = 5, intervention_value: float = 1.0
    ) -> None:
        self._n_pairs = n_pairs
        self._int_val = intervention_value

    @property
    def descriptor_dim(self) -> int:
        """Number of descriptor dimensions."""
        return self._n_pairs

    @property
    def descriptor_bounds(
        self,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Bounds for each dimension."""
        low = np.zeros(self._n_pairs, dtype=np.float64)
        high = np.ones(self._n_pairs, dtype=np.float64)
        return low, high

    def compute(
        self,
        dag: AdjacencyMatrix,
        data: Optional[DataMatrix] = None,
    ) -> BehavioralDescriptor:
        """Compute interventional descriptor.

        For each selected (source, target) pair, computes the causal
        effect of intervening on source on target using truncated
        factorization with the provided data.

        Parameters
        ----------
        dag : AdjacencyMatrix
            DAG adjacency matrix.
        data : DataMatrix | None
            N × p data matrix.  If ``None``, structural effects are
            used (path-based approximation).

        Returns
        -------
        BehavioralDescriptor
        """
        n = dag.shape[0]
        adj = np.asarray(dag, dtype=np.int8)

        # Select variable pairs
        pairs = self._select_pairs(n)

        effects = np.zeros(self._n_pairs, dtype=np.float64)

        for idx, (src, tgt) in enumerate(pairs):
            if data is not None:
                effect = self._compute_interventional_effect(
                    adj, src, tgt, data
                )
            else:
                effect = self._structural_effect(adj, src, tgt)
            effects[idx] = effect

        # Normalize to [0, 1] using sigmoid
        normalized = 1.0 / (1.0 + np.exp(-effects))
        return normalized

    def _select_pairs(self, n: int) -> List[Tuple[int, int]]:
        """Select deterministic variable pairs for descriptor."""
        pairs: List[Tuple[int, int]] = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    pairs.append((i, j))
                if len(pairs) >= self._n_pairs:
                    return pairs
        # Pad if needed
        while len(pairs) < self._n_pairs:
            pairs.append((0, min(1, n - 1)))
        return pairs[:self._n_pairs]

    def _compute_interventional_effect(
        self,
        adj: AdjacencyMatrix,
        source: int,
        target: int,
        data: DataMatrix,
    ) -> float:
        """Compute E[Y | do(X=1)] - E[Y | do(X=0)] using truncated factorization.

        Uses linear regression on the causal DAG to estimate the effect.

        Parameters
        ----------
        adj, source, target, data

        Returns
        -------
        float
            Estimated causal effect.
        """
        n = adj.shape[0]

        # Check if target is a descendant of source
        desc = _descendants(adj, source)
        if target not in desc:
            return 0.0

        # Estimate regression coefficients along the causal path
        # Use total effect via regression with parent adjustment
        try:
            # Simple identification: adjust for parents of source
            parents_src = list(np.where(adj[:, source])[0])

            if parents_src:
                # Regress target on source + parents of source
                X = np.column_stack([data[:, source], data[:, parents_src]])
            else:
                X = data[:, source:source + 1]

            X_with_intercept = np.column_stack([np.ones(data.shape[0]), X])
            y = data[:, target]
            coeffs, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
            return float(coeffs[1])  # Coefficient of source variable
        except (np.linalg.LinAlgError, IndexError):
            return 0.0

    def _structural_effect(
        self, adj: AdjacencyMatrix, source: int, target: int
    ) -> float:
        """Approximate causal effect using path structure only.

        Counts directed paths from source to target weighted by
        path length.

        Parameters
        ----------
        adj, source, target

        Returns
        -------
        float
        """
        desc = _descendants(adj, source)
        if target not in desc:
            return 0.0

        # Count paths weighted by 1/length
        n = adj.shape[0]
        total = 0.0
        # BFS counting paths
        queue = deque([(source, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth > n:
                continue
            for child in range(n):
                if adj[node, child]:
                    if child == target:
                        total += 1.0 / (depth + 1)
                    else:
                        queue.append((child, depth + 1))
        return total


# ---------------------------------------------------------------------------
# CausalEffectDescriptor
# ---------------------------------------------------------------------------


class CausalEffectDescriptor(DescriptorComputer):
    """Average causal effects as behavioral descriptors.

    Computes total, direct, and indirect causal effects between
    selected variable pairs using linear regression on the DAG
    structure.

    Parameters
    ----------
    effect_type : str
        Type of effect: ``"total"``, ``"direct"``, ``"indirect"``,
        or ``"all"`` (concatenates all three).  Default ``"total"``.
    n_pairs : int
        Number of variable pairs.  Default ``4``.
    """

    def __init__(
        self,
        effect_type: str = "total",
        n_pairs: int = 4,
    ) -> None:
        self._effect_type = effect_type
        self._n_pairs = n_pairs
        if effect_type == "all":
            self._dim = n_pairs * 3
        else:
            self._dim = n_pairs

    @property
    def descriptor_dim(self) -> int:
        return self._dim

    @property
    def descriptor_bounds(
        self,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        low = np.zeros(self._dim, dtype=np.float64)
        high = np.ones(self._dim, dtype=np.float64)
        return low, high

    def compute(
        self,
        dag: AdjacencyMatrix,
        data: Optional[DataMatrix] = None,
    ) -> BehavioralDescriptor:
        """Compute causal effect descriptor.

        Parameters
        ----------
        dag, data

        Returns
        -------
        BehavioralDescriptor
        """
        n = dag.shape[0]
        adj = np.asarray(dag, dtype=np.int8)
        pairs = self._select_pairs(n)

        if self._effect_type == "all":
            values = []
            for src, tgt in pairs:
                total = self._total_effect(adj, src, tgt, data)
                direct = self._direct_effect(adj, src, tgt, data)
                indirect = total - direct
                values.extend([total, direct, indirect])
            result = np.array(values[:self._dim], dtype=np.float64)
        else:
            values = []
            for src, tgt in pairs:
                if self._effect_type == "total":
                    val = self._total_effect(adj, src, tgt, data)
                elif self._effect_type == "direct":
                    val = self._direct_effect(adj, src, tgt, data)
                else:  # indirect
                    total = self._total_effect(adj, src, tgt, data)
                    direct = self._direct_effect(adj, src, tgt, data)
                    val = total - direct
                values.append(val)
            result = np.array(values[:self._dim], dtype=np.float64)

        # Normalize to [0, 1]
        return 1.0 / (1.0 + np.exp(-result))

    def _select_pairs(self, n: int) -> List[Tuple[int, int]]:
        """Select variable pairs deterministically."""
        pairs: List[Tuple[int, int]] = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    pairs.append((i, j))
                if len(pairs) >= self._n_pairs:
                    return pairs
        while len(pairs) < self._n_pairs:
            pairs.append((0, min(1, n - 1)))
        return pairs[:self._n_pairs]

    def _total_effect(
        self,
        adj: AdjacencyMatrix,
        source: int,
        target: int,
        data: Optional[DataMatrix],
    ) -> float:
        """Total causal effect: regression coefficient in causal model.

        Parameters
        ----------
        adj, source, target, data

        Returns
        -------
        float
        """
        if data is None:
            return 1.0 if target in _descendants(adj, source) else 0.0

        try:
            X = np.column_stack([np.ones(data.shape[0]), data[:, source]])
            y = data[:, target]
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            return float(coeffs[1])
        except (np.linalg.LinAlgError, IndexError):
            return 0.0

    def _direct_effect(
        self,
        adj: AdjacencyMatrix,
        source: int,
        target: int,
        data: Optional[DataMatrix],
    ) -> float:
        """Direct effect: coefficient holding other parents fixed.

        Parameters
        ----------
        adj, source, target, data

        Returns
        -------
        float
        """
        if not adj[source, target]:
            return 0.0

        if data is None:
            return 1.0

        # Control for other parents of target
        parents_target = list(np.where(adj[:, target])[0])

        try:
            cols = [source] + [p for p in parents_target if p != source]
            X = np.column_stack([np.ones(data.shape[0]), data[:, cols]])
            y = data[:, target]
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            return float(coeffs[1])
        except (np.linalg.LinAlgError, IndexError):
            return 0.0


# ---------------------------------------------------------------------------
# SpectralDescriptor
# ---------------------------------------------------------------------------


class SpectralDescriptor(DescriptorComputer):
    """Eigenvalue-based graph features as descriptors.

    Computes features from the Laplacian and normalized Laplacian
    eigenvalues of the moralized (undirected) graph.

    Features:
    - Top k eigenvalues of the Laplacian
    - Spectral gap (λ₂ - λ₁)
    - Algebraic connectivity (second-smallest eigenvalue)
    - Spectral radius

    Parameters
    ----------
    n_eigenvalues : int
        Number of top eigenvalues to include.  Default ``3``.
    include_gap : bool
        Whether to include the spectral gap.  Default ``True``.
    include_connectivity : bool
        Whether to include algebraic connectivity.  Default ``True``.
    """

    def __init__(
        self,
        n_eigenvalues: int = 3,
        include_gap: bool = True,
        include_connectivity: bool = True,
    ) -> None:
        self._n_eig = n_eigenvalues
        self._gap = include_gap
        self._conn = include_connectivity
        self._dim = n_eigenvalues + int(include_gap) + int(include_connectivity)

    @property
    def descriptor_dim(self) -> int:
        return self._dim

    @property
    def descriptor_bounds(
        self,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        low = np.zeros(self._dim, dtype=np.float64)
        high = np.ones(self._dim, dtype=np.float64)
        return low, high

    def compute(
        self,
        dag: AdjacencyMatrix,
        data: Optional[DataMatrix] = None,
    ) -> BehavioralDescriptor:
        """Compute spectral descriptor.

        Parameters
        ----------
        dag, data

        Returns
        -------
        BehavioralDescriptor
        """
        n = dag.shape[0]
        adj = np.asarray(dag, dtype=np.int8)

        # Moralize the graph
        moral = self._moralize(adj, n)

        # Compute Laplacian
        degree = moral.sum(axis=1).astype(np.float64)
        laplacian = np.diag(degree) - moral.astype(np.float64)

        # Compute eigenvalues
        try:
            eigenvalues = np.sort(np.linalg.eigvalsh(laplacian))
        except np.linalg.LinAlgError:
            eigenvalues = np.zeros(n)

        features: List[float] = []

        # Top k eigenvalues (normalized)
        max_eig = eigenvalues[-1] if len(eigenvalues) > 0 and eigenvalues[-1] > 0 else 1.0
        for i in range(self._n_eig):
            if i < len(eigenvalues):
                features.append(float(eigenvalues[-(i + 1)]) / max(max_eig, 1e-10))
            else:
                features.append(0.0)

        # Spectral gap
        if self._gap:
            if len(eigenvalues) >= 2:
                gap = float(eigenvalues[1] - eigenvalues[0])
                features.append(gap / max(max_eig, 1e-10))
            else:
                features.append(0.0)

        # Algebraic connectivity
        if self._conn:
            if len(eigenvalues) >= 2:
                features.append(float(eigenvalues[1]) / max(max_eig, 1e-10))
            else:
                features.append(0.0)

        result = np.array(features[:self._dim], dtype=np.float64)
        return np.clip(result, 0.0, 1.0)

    @staticmethod
    def _moralize(adj: AdjacencyMatrix, n: int) -> AdjacencyMatrix:
        """Moralize the DAG: marry parents, drop orientations."""
        skeleton = (adj | adj.T).astype(np.int8)
        for node in range(n):
            parents = list(np.where(adj[:, node])[0])
            for a in range(len(parents)):
                for b in range(a + 1, len(parents)):
                    skeleton[parents[a], parents[b]] = 1
                    skeleton[parents[b], parents[a]] = 1
        np.fill_diagonal(skeleton, 0)
        return skeleton


# ---------------------------------------------------------------------------
# PathDescriptor
# ---------------------------------------------------------------------------


class PathDescriptor(DescriptorComputer):
    """Directed path statistics as behavioral descriptors.

    Computes features related to the directed path structure:
    - Number of directed paths (normalized)
    - Average path length
    - Maximum path length
    - Path count distribution entropy
    - Backdoor path indicator

    Parameters
    ----------
    n_features : int
        Number of features to compute.  Default ``5``.
    """

    def __init__(self, n_features: int = 5) -> None:
        self._dim = n_features

    @property
    def descriptor_dim(self) -> int:
        return self._dim

    @property
    def descriptor_bounds(
        self,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        low = np.zeros(self._dim, dtype=np.float64)
        high = np.ones(self._dim, dtype=np.float64)
        return low, high

    def compute(
        self,
        dag: AdjacencyMatrix,
        data: Optional[DataMatrix] = None,
    ) -> BehavioralDescriptor:
        """Compute path-based descriptor.

        Parameters
        ----------
        dag, data

        Returns
        -------
        BehavioralDescriptor
        """
        n = dag.shape[0]
        adj = np.asarray(dag, dtype=np.int8)

        features: List[float] = []

        # 1. Normalized path count
        path_count = self._count_all_paths(adj, n)
        max_paths = n * (n - 1) * (2 ** (n - 2)) if n > 2 else max(n - 1, 1)
        features.append(min(path_count / max(max_paths, 1), 1.0))

        # 2. Average directed path length
        avg_len = self._avg_path_length(adj, n)
        features.append(min(avg_len / max(n - 1, 1), 1.0))

        # 3. Maximum directed path length (longest path)
        max_len = self._longest_path(adj, n)
        features.append(max_len / max(n - 1, 1))

        # 4. Path count entropy per node
        entropy = self._path_count_entropy(adj, n)
        features.append(entropy)

        # 5. Fraction of node pairs with backdoor paths
        backdoor_frac = self._backdoor_fraction(adj, n)
        features.append(backdoor_frac)

        result = np.array(features[:self._dim], dtype=np.float64)
        return np.clip(result, 0.0, 1.0)

    @staticmethod
    def _count_all_paths(adj: AdjacencyMatrix, n: int) -> int:
        """Count the total number of directed paths (all lengths ≥ 1).

        Uses matrix power series: total paths = sum of A^k for k=1..n-1.

        Parameters
        ----------
        adj, n

        Returns
        -------
        int
        """
        if n <= 1:
            return 0
        a = adj.astype(np.float64)
        total = 0
        power = a.copy()
        for _ in range(1, n):
            total += int(power.sum())
            power = power @ a
            if power.sum() == 0:
                break
        return total

    @staticmethod
    def _avg_path_length(adj: AdjacencyMatrix, n: int) -> float:
        """Average shortest directed path length over reachable pairs.

        Parameters
        ----------
        adj, n

        Returns
        -------
        float
        """
        if n <= 1:
            return 0.0

        total = 0.0
        count = 0
        for source in range(n):
            dist = np.full(n, -1, dtype=int)
            dist[source] = 0
            queue: deque[int] = deque([source])
            while queue:
                node = queue.popleft()
                for child in range(n):
                    if adj[node, child] and dist[child] == -1:
                        dist[child] = dist[node] + 1
                        queue.append(child)
            reachable = dist[dist > 0]
            total += float(reachable.sum())
            count += len(reachable)

        return total / max(count, 1)

    @staticmethod
    def _longest_path(adj: AdjacencyMatrix, n: int) -> float:
        """Longest directed path length via DP on topological order.

        Parameters
        ----------
        adj, n

        Returns
        -------
        float
        """
        if n <= 1:
            return 0.0

        order = _topological_sort(adj)
        dist = np.zeros(n, dtype=int)
        for v in order:
            parents = np.where(adj[:, v])[0]
            if len(parents) > 0:
                dist[v] = int(dist[parents].max()) + 1

        return float(dist.max())

    @staticmethod
    def _path_count_entropy(adj: AdjacencyMatrix, n: int) -> float:
        """Entropy of the out-path-count distribution.

        For each node, count the total number of descendant paths.
        Compute entropy of this distribution normalized by log(n).

        Parameters
        ----------
        adj, n

        Returns
        -------
        float
            Normalized entropy in [0, 1].
        """
        if n <= 1:
            return 0.0

        desc_counts = np.zeros(n, dtype=np.float64)
        for node in range(n):
            desc = _descendants(adj, node)
            desc_counts[node] = len(desc)

        total = desc_counts.sum()
        if total == 0:
            return 0.0

        probs = desc_counts / total
        probs = probs[probs > 0]
        entropy = -float(np.sum(probs * np.log(probs)))
        return min(entropy / max(math.log(n), 1e-10), 1.0)

    @staticmethod
    def _backdoor_fraction(adj: AdjacencyMatrix, n: int) -> float:
        """Fraction of directed-edge pairs with a backdoor path.

        A backdoor path from X to Y is a path that starts with an
        edge into X: X ← ... → Y.  We check for each edge X → Y
        whether X's parents share a common ancestor with Y.

        Parameters
        ----------
        adj, n

        Returns
        -------
        float
            Fraction of edges with backdoor paths.
        """
        if n <= 2:
            return 0.0

        edges = list(zip(*np.nonzero(adj)))
        if not edges:
            return 0.0

        backdoor_count = 0
        for x, y in edges:
            # X → Y exists.  Check if there's a path from some parent of X to Y
            # that doesn't go through X.
            parents_x = set(np.where(adj[:, x])[0])
            if not parents_x:
                continue

            # For each parent of X, check if it can reach Y without going through X
            adj_no_x = adj.copy()
            adj_no_x[x, :] = 0
            adj_no_x[:, x] = 0
            skel = adj_no_x | adj_no_x.T

            for p in parents_x:
                # BFS on skeleton from p to Y
                visited = {p}
                queue: deque[int] = deque([p])
                found = False
                while queue and not found:
                    node = queue.popleft()
                    for nb in range(n):
                        if skel[node, nb] and nb not in visited:
                            if nb == y:
                                found = True
                                break
                            visited.add(nb)
                            queue.append(nb)
                if found:
                    backdoor_count += 1
                    break

        return backdoor_count / len(edges)


# ---------------------------------------------------------------------------
# CompositeAdvancedDescriptor
# ---------------------------------------------------------------------------


class CompositeAdvancedDescriptor(DescriptorComputer):
    """Combine multiple advanced descriptors into one.

    Parameters
    ----------
    descriptors : List[DescriptorComputer]
        Component descriptors to concatenate.
    """

    def __init__(self, descriptors: List[DescriptorComputer]) -> None:
        self._descriptors = descriptors

    @property
    def descriptor_dim(self) -> int:
        return sum(d.descriptor_dim for d in self._descriptors)

    @property
    def descriptor_bounds(
        self,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        lows = []
        highs = []
        for d in self._descriptors:
            lo, hi = d.descriptor_bounds
            lows.append(lo)
            highs.append(hi)
        return np.concatenate(lows), np.concatenate(highs)

    def compute(
        self,
        dag: AdjacencyMatrix,
        data: Optional[DataMatrix] = None,
    ) -> BehavioralDescriptor:
        """Concatenate descriptors from all components.

        Parameters
        ----------
        dag, data

        Returns
        -------
        BehavioralDescriptor
        """
        parts = [d.compute(dag, data) for d in self._descriptors]
        return np.concatenate(parts)
