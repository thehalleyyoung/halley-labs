"""Initialization strategies for populating the MAP-Elites archive.

Provides several strategies for generating seed DAGs that are inserted
into the archive before the evolutionary loop begins.

Strategies
----------
- :class:`RandomInitialization`: Erdős–Rényi random DAGs.
- :class:`HeuristicInitialization`: Correlation-skeleton + random orientations.
- :class:`DiverseInitialization`: Maximize descriptor-space coverage.
- :class:`WarmStartInitialization`: Load from a previous archive.
- :class:`MixedInitialization`: Combine multiple strategies.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.types import AdjacencyMatrix, BehavioralDescriptor, DataMatrix


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class InitializationStrategy(ABC):
    """Abstract base class for archive initialization strategies.

    An initialization strategy produces a collection of seed DAGs that
    populate the MAP-Elites archive before the evolutionary loop begins.
    """

    @abstractmethod
    def initialize(
        self,
        n: int,
        n_nodes: int,
        data: DataMatrix,
        rng: np.random.Generator,
    ) -> List[AdjacencyMatrix]:
        """Generate *n* seed DAGs.

        Parameters
        ----------
        n : int
            Number of DAGs to generate.
        n_nodes : int
            Number of nodes (variables) in each DAG.
        data : DataMatrix
            Observed data matrix (N × p), available for data-driven
            initialization strategies.
        rng : np.random.Generator
            NumPy random generator for reproducibility.

        Returns
        -------
        List[AdjacencyMatrix]
            A list of *n* adjacency matrices, each guaranteed to be a DAG.
        """


# ---------------------------------------------------------------------------
# Random initialisation
# ---------------------------------------------------------------------------

class RandomInitialization(InitializationStrategy):
    """Generate random DAGs via Erdős–Rényi sampling over a random ordering.

    Each DAG is constructed by first drawing a uniformly random
    permutation (the topological order) and then independently including
    each forward edge with probability *edge_prob*.

    Parameters
    ----------
    edge_prob : float or None
        Probability of including each candidate edge.  Defaults to
        ``2 / n_nodes`` (sparse graphs) if ``None``.
    max_parents : int
        Maximum in-degree for any node (``-1`` = no limit).
    """

    def __init__(
        self,
        edge_prob: Optional[float] = None,
        max_parents: int = -1,
    ) -> None:
        self._edge_prob = edge_prob
        self._max_parents = max_parents

    def initialize(
        self,
        n: int,
        n_nodes: int,
        data: DataMatrix,
        rng: np.random.Generator,
    ) -> List[AdjacencyMatrix]:
        """Generate *n* random Erdős–Rényi DAGs."""
        p = self._edge_prob if self._edge_prob is not None else 2.0 / max(n_nodes, 1)

        dags: List[AdjacencyMatrix] = []
        for _ in range(n):
            perm = rng.permutation(n_nodes)
            adj = np.zeros((n_nodes, n_nodes), dtype=np.int8)
            for idx_i in range(n_nodes):
                for idx_j in range(idx_i + 1, n_nodes):
                    if rng.random() < p:
                        src, tgt = perm[idx_i], perm[idx_j]
                        # Enforce max_parents
                        if self._max_parents > 0:
                            if adj[:, tgt].sum() >= self._max_parents:
                                continue
                        adj[src, tgt] = 1
            dags.append(adj)
        return dags


# ---------------------------------------------------------------------------
# Heuristic (correlation-based) initialisation
# ---------------------------------------------------------------------------

class HeuristicInitialization(InitializationStrategy):
    """Generate seed DAGs using constraint-based heuristic skeletons.

    Builds a skeleton by thresholding pairwise correlations (a
    lightweight proxy for the PC algorithm), then orients edges
    according to multiple random topological orderings.

    Parameters
    ----------
    correlation_threshold : float
        Absolute-correlation threshold for including a skeleton edge.
    n_orientations : int
        Number of random orientations per skeleton.
    partial_correlation : bool
        If ``True``, use partial correlations (controlling for all other
        variables) instead of marginal correlations.
    """

    def __init__(
        self,
        correlation_threshold: float = 0.1,
        n_orientations: int = 5,
        partial_correlation: bool = False,
    ) -> None:
        self._corr_threshold = correlation_threshold
        self._n_orientations = n_orientations
        self._partial_correlation = partial_correlation

    def initialize(
        self,
        n: int,
        n_nodes: int,
        data: DataMatrix,
        rng: np.random.Generator,
    ) -> List[AdjacencyMatrix]:
        """Generate DAGs from a correlation-based skeleton.

        1. Compute the absolute sample correlation (or partial correlation).
        2. Build a skeleton by keeping pairs with |r| > threshold.
        3. For each requested DAG, draw a random permutation and orient
           skeleton edges in the forward direction of that permutation.
        """
        if data.shape[0] < 2 or data.shape[1] < 2:
            return RandomInitialization().initialize(n, n_nodes, data, rng)

        if self._partial_correlation:
            corr = self._compute_partial_correlations(data)
        else:
            corr = np.corrcoef(data, rowvar=False)
        np.fill_diagonal(corr, 0.0)
        skeleton = np.abs(corr) > self._corr_threshold

        dags: List[AdjacencyMatrix] = []
        for _ in range(n):
            perm = rng.permutation(n_nodes)
            order = np.argsort(perm)

            adj = np.zeros((n_nodes, n_nodes), dtype=np.int8)
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if skeleton[i, j] or skeleton[j, i]:
                        if order[i] < order[j]:
                            adj[i, j] = 1
                        else:
                            adj[j, i] = 1
            dags.append(adj)
        return dags

    @staticmethod
    def _compute_partial_correlations(data: DataMatrix) -> np.ndarray:
        """Compute the partial correlation matrix from data.

        Uses the inverse of the covariance matrix (precision matrix).

        Parameters
        ----------
        data : DataMatrix

        Returns
        -------
        np.ndarray
            Partial correlation matrix of shape (p, p).
        """
        cov = np.cov(data, rowvar=False)
        try:
            precision = np.linalg.inv(cov + 1e-8 * np.eye(cov.shape[0]))
        except np.linalg.LinAlgError:
            return np.corrcoef(data, rowvar=False)

        diag = np.sqrt(np.diag(precision))
        diag[diag == 0] = 1.0
        outer = np.outer(diag, diag)
        pcorr = -precision / outer
        np.fill_diagonal(pcorr, 1.0)
        return pcorr


# ---------------------------------------------------------------------------
# Diverse initialisation (maximize descriptor coverage)
# ---------------------------------------------------------------------------

class DiverseInitialization(InitializationStrategy):
    """Generate seed DAGs that maximise descriptor-space coverage.

    1. Generate a large pool of random candidate DAGs.
    2. Compute descriptors for each.
    3. Greedily select *n* DAGs that maximise the sum of nearest-neighbour
       distances in descriptor space.

    Parameters
    ----------
    pool_size_factor : int
        Multiplier for the candidate pool size (``n * pool_size_factor``).
    descriptor_fn : callable
        Function ``(adjacency, data) -> BehavioralDescriptor``.
    edge_prob : float or None
        Edge probability for random DAG generation.
    """

    def __init__(
        self,
        descriptor_fn: Callable[[AdjacencyMatrix, DataMatrix], BehavioralDescriptor],
        pool_size_factor: int = 10,
        edge_prob: Optional[float] = None,
    ) -> None:
        self._descriptor_fn = descriptor_fn
        self._pool_factor = pool_size_factor
        self._edge_prob = edge_prob

    def initialize(
        self,
        n: int,
        n_nodes: int,
        data: DataMatrix,
        rng: np.random.Generator,
    ) -> List[AdjacencyMatrix]:
        """Generate *n* diverse DAGs.

        A pool of ``n * pool_size_factor`` random DAGs is generated,
        descriptors are computed, and a greedy furthest-point selection
        picks the *n* most spread-out candidates.
        """
        pool_size = max(n * self._pool_factor, n + 1)
        rand_init = RandomInitialization(edge_prob=self._edge_prob)
        pool = rand_init.initialize(pool_size, n_nodes, data, rng)

        # Compute descriptors
        descriptors = np.array(
            [self._descriptor_fn(dag, data) for dag in pool]
        )

        # Greedy farthest-point sampling
        selected_idx: List[int] = []
        remaining = set(range(len(pool)))

        # Start with a random point
        first = int(rng.integers(0, len(pool)))
        selected_idx.append(first)
        remaining.discard(first)

        while len(selected_idx) < n and remaining:
            sel_descs = descriptors[selected_idx]
            best_idx = -1
            best_dist = -1.0
            for idx in remaining:
                d = descriptors[idx]
                min_dist = np.min(np.linalg.norm(sel_descs - d, axis=1))
                if min_dist > best_dist:
                    best_dist = min_dist
                    best_idx = idx
            selected_idx.append(best_idx)
            remaining.discard(best_idx)

        return [pool[i] for i in selected_idx]


# ---------------------------------------------------------------------------
# Warm-start from existing archive
# ---------------------------------------------------------------------------

class WarmStartInitialization(InitializationStrategy):
    """Load seed DAGs from a previously saved archive.

    Parameters
    ----------
    archive_path : str
        Path to a pickled archive file.
    max_seeds : int or None
        Maximum number of seeds to extract (``None`` = all).
    """

    def __init__(
        self,
        archive_path: str,
        max_seeds: Optional[int] = None,
    ) -> None:
        self._archive_path = archive_path
        self._max_seeds = max_seeds

    def initialize(
        self,
        n: int,
        n_nodes: int,
        data: DataMatrix,
        rng: np.random.Generator,
    ) -> List[AdjacencyMatrix]:
        """Load seed DAGs from the archive file.

        The file should be a pickle containing a dict with a "grid"
        key mapping cell indices to archive entries with a "solution"
        attribute (adjacency matrix).
        """
        path = Path(self._archive_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Archive file not found: {self._archive_path}"
            )

        with open(path, "rb") as f:
            state = pickle.load(f)

        # Extract adjacency matrices from the archive
        dags: List[AdjacencyMatrix] = []
        if isinstance(state, dict):
            grid = state.get("grid", state.get("cells", {}))
            for entry in grid.values():
                if hasattr(entry, "solution"):
                    dags.append(entry.solution)
                elif hasattr(entry, "adjacency"):
                    dags.append(entry.adjacency)
                elif isinstance(entry, np.ndarray):
                    dags.append(entry)

        if self._max_seeds is not None:
            dags = dags[: self._max_seeds]

        # If we need more DAGs than we loaded, pad with random ones
        if len(dags) < n:
            rand = RandomInitialization()
            extra = rand.initialize(n - len(dags), n_nodes, data, rng)
            dags.extend(extra)

        # Limit to n
        if len(dags) > n:
            indices = rng.choice(len(dags), size=n, replace=False)
            dags = [dags[int(i)] for i in indices]

        return dags


# ---------------------------------------------------------------------------
# Empty initialisation
# ---------------------------------------------------------------------------

class EmptyInitialization(InitializationStrategy):
    """Generate *n* empty DAGs (no edges).

    Useful as a baseline or starting point for purely exploratory runs.
    """

    def initialize(
        self,
        n: int,
        n_nodes: int,
        data: DataMatrix,
        rng: np.random.Generator,
    ) -> List[AdjacencyMatrix]:
        """Return *n* empty adjacency matrices."""
        return [np.zeros((n_nodes, n_nodes), dtype=np.int8) for _ in range(n)]


# ---------------------------------------------------------------------------
# Mixed initialisation
# ---------------------------------------------------------------------------

class MixedInitialization(InitializationStrategy):
    """Combine multiple initialization strategies.

    Each sub-strategy is assigned a proportion of the total DAGs to
    generate.

    Parameters
    ----------
    strategies : List[Tuple[InitializationStrategy, float]]
        List of ``(strategy, proportion)`` pairs.  Proportions are
        normalised to sum to 1.

    Example
    -------
    >>> mixed = MixedInitialization([
    ...     (RandomInitialization(), 0.5),
    ...     (HeuristicInitialization(), 0.3),
    ...     (EmptyInitialization(), 0.2),
    ... ])
    >>> dags = mixed.initialize(100, 5, data, rng)
    """

    def __init__(
        self,
        strategies: List[Tuple[InitializationStrategy, float]],
    ) -> None:
        if not strategies:
            raise ValueError("At least one strategy must be provided.")
        self._strategies = strategies

    def initialize(
        self,
        n: int,
        n_nodes: int,
        data: DataMatrix,
        rng: np.random.Generator,
    ) -> List[AdjacencyMatrix]:
        """Generate DAGs by distributing *n* across sub-strategies.

        Proportions are normalised.  Rounding errors are assigned to the
        first strategy.
        """
        total_weight = sum(w for _, w in self._strategies)
        if total_weight <= 0:
            total_weight = len(self._strategies)
            normalised = [(s, 1.0 / total_weight) for s, _ in self._strategies]
        else:
            normalised = [(s, w / total_weight) for s, w in self._strategies]

        # Compute per-strategy counts
        counts: List[int] = []
        remaining = n
        for i, (_, prop) in enumerate(normalised):
            if i == len(normalised) - 1:
                counts.append(remaining)
            else:
                c = max(0, int(round(prop * n)))
                c = min(c, remaining)
                counts.append(c)
                remaining -= c

        # Generate from each strategy
        all_dags: List[AdjacencyMatrix] = []
        for (strategy, _), count in zip(normalised, counts):
            if count > 0:
                dags = strategy.initialize(count, n_nodes, data, rng)
                all_dags.extend(dags)

        # Shuffle to mix strategies
        perm = rng.permutation(len(all_dags))
        return [all_dags[int(i)] for i in perm]
