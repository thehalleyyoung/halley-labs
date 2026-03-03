"""Distributed computation support for CausalQD.

This module provides the island-model parallelization layer for MAP-Elites,
where multiple independent MAP-Elites instances (islands) run concurrently
and periodically exchange their best solutions (migration).

Key classes
-----------
* :class:`DistributedEvaluator` – manage islands and migration
* :class:`IslandModelMAPElites` – island-model MAP-Elites runner
* :class:`MigrationTopology` – migration graph topologies
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.types import (
    AdjacencyMatrix,
    BehavioralDescriptor,
    DataMatrix,
    QualityScore,
)

__all__ = [
    "DistributedEvaluator",
    "IslandModelMAPElites",
    "MigrationTopology",
    "IslandConfig",
    "MigrationPolicy",
    "IslandState",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class MigrationTopology(Enum):
    """Topology for migration between islands."""

    RING = "ring"
    FULLY_CONNECTED = "fully_connected"
    RANDOM = "random"
    STAR = "star"


@dataclass
class MigrationPolicy:
    """Policy controlling migration between islands.

    Attributes
    ----------
    topology : MigrationTopology
        Migration graph topology.
    interval : int
        Generations between migrations.
    n_migrants : int
        Number of elites to migrate each time.
    selection : str
        How to select migrants: ``"best"`` or ``"random"``.
    replacement : str
        How to integrate migrants: ``"quality"`` (replace if better)
        or ``"always"``.
    synchronous : bool
        If True, all islands synchronize at migration points.
    """

    topology: MigrationTopology = MigrationTopology.RING
    interval: int = 50
    n_migrants: int = 5
    selection: str = "best"
    replacement: str = "quality"
    synchronous: bool = True


@dataclass
class IslandConfig:
    """Configuration for a single island.

    Attributes
    ----------
    island_id : int
        Unique identifier.
    mutation_rate : float
        Mutation probability for this island.
    crossover_rate : float
        Crossover probability.
    archive_dims : tuple of int
        Grid archive dimensions.
    seed : int
        Random seed.
    """

    island_id: int = 0
    mutation_rate: float = 0.7
    crossover_rate: float = 0.3
    archive_dims: Tuple[int, ...] = (20, 20)
    seed: int = 42


@dataclass
class IslandState:
    """State of a single island for serialization.

    Attributes
    ----------
    island_id : int
        Island identifier.
    generation : int
        Current generation count.
    archive_size : int
        Number of occupied cells.
    best_quality : float
        Best quality score.
    qd_score : float
        Total QD score.
    coverage : float
        Archive coverage.
    solutions : list
        List of (adjacency, quality, descriptor) tuples.
    """

    island_id: int = 0
    generation: int = 0
    archive_size: int = 0
    best_quality: float = float("-inf")
    qd_score: float = 0.0
    coverage: float = 0.0
    solutions: List[Tuple[np.ndarray, float, np.ndarray]] = field(
        default_factory=list
    )


# ---------------------------------------------------------------------------
# Migration graph construction
# ---------------------------------------------------------------------------


def _build_migration_graph(
    n_islands: int,
    topology: MigrationTopology,
    rng: Optional[np.random.Generator] = None,
) -> Dict[int, List[int]]:
    """Build adjacency list for migration topology.

    Parameters
    ----------
    n_islands : int
        Number of islands.
    topology : MigrationTopology
        Topology type.
    rng : np.random.Generator, optional
        RNG for random topology.

    Returns
    -------
    dict
        Mapping from island ID to list of target island IDs.
    """
    graph: Dict[int, List[int]] = {i: [] for i in range(n_islands)}

    if topology == MigrationTopology.RING:
        for i in range(n_islands):
            graph[i].append((i + 1) % n_islands)

    elif topology == MigrationTopology.FULLY_CONNECTED:
        for i in range(n_islands):
            for j in range(n_islands):
                if i != j:
                    graph[i].append(j)

    elif topology == MigrationTopology.STAR:
        # Island 0 is the hub
        for i in range(1, n_islands):
            graph[0].append(i)
            graph[i].append(0)

    elif topology == MigrationTopology.RANDOM:
        if rng is None:
            rng = np.random.default_rng()
        for i in range(n_islands):
            others = [j for j in range(n_islands) if j != i]
            n_neighbors = max(1, rng.integers(1, len(others) + 1))
            targets = rng.choice(others, size=n_neighbors, replace=False)
            graph[i] = targets.tolist()

    return graph


# ---------------------------------------------------------------------------
# Island archive (lightweight)
# ---------------------------------------------------------------------------


class _IslandArchive:
    """Lightweight grid archive for an island.

    Implements the minimum interface needed for island-model MAP-Elites.
    """

    def __init__(
        self,
        dims: Tuple[int, ...],
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> None:
        self._dims = np.array(dims, dtype=np.int64)
        self._lower = np.asarray(lower_bounds, dtype=np.float64)
        self._upper = np.asarray(upper_bounds, dtype=np.float64)
        self._total_cells = int(np.prod(self._dims))
        self._widths = (self._upper - self._lower) / self._dims.astype(np.float64)
        self._d = len(dims)

        self._qualities = np.full(self._total_cells, float("-inf"), dtype=np.float64)
        self._solutions: Dict[int, np.ndarray] = {}
        self._descriptors: Dict[int, np.ndarray] = {}

    def _to_flat_index(self, descriptor: np.ndarray) -> int:
        clipped = np.clip(descriptor, self._lower, self._upper - 1e-12)
        coords = ((clipped - self._lower) / self._widths).astype(np.int64)
        coords = np.clip(coords, 0, self._dims - 1)
        multipliers = np.ones(self._d, dtype=np.int64)
        for i in range(self._d - 2, -1, -1):
            multipliers[i] = multipliers[i + 1] * self._dims[i + 1]
        return int(np.sum(coords * multipliers))

    def add(
        self,
        solution: np.ndarray,
        quality: float,
        descriptor: np.ndarray,
    ) -> bool:
        ci = self._to_flat_index(descriptor)
        if quality > self._qualities[ci]:
            self._qualities[ci] = quality
            self._solutions[ci] = solution.copy()
            self._descriptors[ci] = descriptor.copy()
            return True
        return False

    def sample(self, n: int, rng: np.random.Generator) -> List[Tuple[np.ndarray, float, np.ndarray]]:
        occupied = list(self._solutions.keys())
        if not occupied:
            return []
        indices = rng.choice(occupied, size=min(n, len(occupied)), replace=True)
        return [
            (self._solutions[i], float(self._qualities[i]), self._descriptors[i])
            for i in indices
        ]

    def best_elites(self, n: int) -> List[Tuple[np.ndarray, float, np.ndarray]]:
        occupied = list(self._solutions.keys())
        if not occupied:
            return []
        qualities = np.array([self._qualities[i] for i in occupied])
        top_k = min(n, len(occupied))
        top_indices = np.argsort(qualities)[-top_k:][::-1]
        return [
            (
                self._solutions[occupied[i]],
                float(self._qualities[occupied[i]]),
                self._descriptors[occupied[i]],
            )
            for i in top_indices
        ]

    def random_elites(self, n: int, rng: np.random.Generator) -> List[Tuple[np.ndarray, float, np.ndarray]]:
        return self.sample(n, rng)

    @property
    def size(self) -> int:
        return len(self._solutions)

    @property
    def qd_score(self) -> float:
        if not self._solutions:
            return 0.0
        return float(sum(self._qualities[i] for i in self._solutions))

    @property
    def coverage(self) -> float:
        return len(self._solutions) / self._total_cells

    @property
    def best_quality(self) -> float:
        if not self._solutions:
            return float("-inf")
        return float(max(self._qualities[i] for i in self._solutions))


# ---------------------------------------------------------------------------
# DistributedEvaluator
# ---------------------------------------------------------------------------


class DistributedEvaluator:
    """Manage multiple island MAP-Elites instances with migration.

    Runs ``n_islands`` independent MAP-Elites searches and periodically
    exchanges elite solutions between islands according to a migration
    policy.

    Parameters
    ----------
    n_islands : int
        Number of islands.
    score_fn : callable
        ``(dag, data) -> float`` scoring function.
    descriptor_fn : callable
        ``(dag, data) -> np.ndarray`` descriptor function.
    mutation_fns : list of callable
        Mutation operators.
    crossover_fns : list of callable, optional
        Crossover operators.
    migration_policy : MigrationPolicy, optional
        Migration configuration.
    island_configs : list of IslandConfig, optional
        Per-island configuration. If None, default configs are used.
    archive_bounds : tuple, optional
        (lower_bounds, upper_bounds) for the archive.
    """

    def __init__(
        self,
        n_islands: int,
        score_fn: Callable,
        descriptor_fn: Callable,
        mutation_fns: List[Callable],
        crossover_fns: Optional[List[Callable]] = None,
        migration_policy: Optional[MigrationPolicy] = None,
        island_configs: Optional[List[IslandConfig]] = None,
        archive_bounds: Optional[
            Tuple[np.ndarray, np.ndarray]
        ] = None,
    ) -> None:
        self._n_islands = n_islands
        self._score_fn = score_fn
        self._descriptor_fn = descriptor_fn
        self._mutation_fns = mutation_fns
        self._crossover_fns = crossover_fns or []
        self._migration_policy = migration_policy or MigrationPolicy()

        # Default archive bounds
        if archive_bounds is None:
            archive_bounds = (np.zeros(2), np.ones(2))

        lower, upper = archive_bounds

        # Island configs
        if island_configs is None:
            island_configs = [
                IslandConfig(
                    island_id=i,
                    seed=42 + i * 1000,
                    mutation_rate=0.5 + 0.3 * (i / max(n_islands - 1, 1)),
                )
                for i in range(n_islands)
            ]
        self._island_configs = island_configs

        # Create archives
        dims = island_configs[0].archive_dims
        self._archives = [
            _IslandArchive(dims, lower, upper) for _ in range(n_islands)
        ]

        # RNGs
        self._rngs = [
            np.random.default_rng(cfg.seed) for cfg in island_configs
        ]

        # Migration graph
        self._migration_graph = _build_migration_graph(
            n_islands,
            self._migration_policy.topology,
            self._rngs[0],
        )

        # History
        self._generation = 0
        self._migration_history: List[Dict[str, Any]] = []

    def run(
        self,
        data: DataMatrix,
        n_generations: int,
        batch_size: int = 16,
        initial_dags: Optional[List[AdjacencyMatrix]] = None,
    ) -> List[IslandState]:
        """Run island-model MAP-Elites.

        Parameters
        ----------
        data : np.ndarray
            ``(N, p)`` data matrix.
        n_generations : int
            Number of generations to run.
        batch_size : int
            Solutions per island per generation.
        initial_dags : list of np.ndarray, optional
            Initial DAGs to seed all islands.

        Returns
        -------
        list of IslandState
            Final state of each island.
        """
        n_nodes = data.shape[1]

        # Seed islands
        if initial_dags is not None:
            for archive in self._archives:
                for dag in initial_dags:
                    try:
                        q = self._score_fn(dag, data)
                        d = self._descriptor_fn(dag, data)
                        archive.add(dag, float(q), np.asarray(d))
                    except Exception:
                        pass

        for gen in range(n_generations):
            self._generation = gen

            # Evolve each island
            for island_id in range(self._n_islands):
                self._evolve_island(island_id, data, batch_size, n_nodes)

            # Migration
            if (
                self._migration_policy.interval > 0
                and gen > 0
                and gen % self._migration_policy.interval == 0
            ):
                self._migrate()

        return self._get_island_states()

    def _evolve_island(
        self,
        island_id: int,
        data: DataMatrix,
        batch_size: int,
        n_nodes: int,
    ) -> None:
        """Run one generation on a single island."""
        archive = self._archives[island_id]
        rng = self._rngs[island_id]
        config = self._island_configs[island_id]

        # Select parents
        if archive.size == 0:
            # Generate random DAGs
            candidates = []
            for _ in range(batch_size):
                adj = np.zeros((n_nodes, n_nodes), dtype=np.int8)
                perm = rng.permutation(n_nodes)
                for a in range(n_nodes):
                    for b in range(a + 1, n_nodes):
                        if rng.random() < 0.3:
                            adj[perm[a], perm[b]] = 1
                candidates.append(adj)
        else:
            parents = archive.sample(batch_size, rng)
            candidates = []
            for sol, _, _ in parents:
                if rng.random() < config.mutation_rate and self._mutation_fns:
                    op = rng.choice(len(self._mutation_fns))
                    try:
                        child = self._mutation_fns[op](sol, rng)
                        candidates.append(child)
                    except Exception:
                        candidates.append(sol.copy())
                elif self._crossover_fns and len(parents) >= 2:
                    p2_idx = rng.integers(0, len(parents))
                    op = rng.choice(len(self._crossover_fns))
                    try:
                        c1, c2 = self._crossover_fns[op](
                            sol, parents[p2_idx][0], rng
                        )
                        candidates.append(c1)
                    except Exception:
                        candidates.append(sol.copy())
                else:
                    candidates.append(sol.copy())

        # Evaluate and add
        for dag in candidates:
            try:
                q = self._score_fn(dag, data)
                d = self._descriptor_fn(dag, data)
                archive.add(dag, float(q), np.asarray(d, dtype=np.float64))
            except Exception:
                pass

    def _migrate(self) -> None:
        """Perform migration between islands according to policy."""
        policy = self._migration_policy
        migration_record: Dict[str, Any] = {
            "generation": self._generation,
            "transfers": [],
        }

        for src_id, targets in self._migration_graph.items():
            src_archive = self._archives[src_id]
            rng = self._rngs[src_id]

            # Select migrants
            if policy.selection == "best":
                migrants = src_archive.best_elites(policy.n_migrants)
            else:
                migrants = src_archive.random_elites(policy.n_migrants, rng)

            if not migrants:
                continue

            # Send to targets
            for tgt_id in targets:
                tgt_archive = self._archives[tgt_id]
                n_accepted = 0

                for sol, quality, desc in migrants:
                    if policy.replacement == "quality":
                        accepted = tgt_archive.add(sol, quality, desc)
                    else:
                        tgt_archive.add(sol, quality, desc)
                        accepted = True
                    if accepted:
                        n_accepted += 1

                migration_record["transfers"].append(
                    {
                        "from": src_id,
                        "to": tgt_id,
                        "sent": len(migrants),
                        "accepted": n_accepted,
                    }
                )

        self._migration_history.append(migration_record)

    def _get_island_states(self) -> List[IslandState]:
        """Snapshot current state of all islands."""
        states = []
        for i in range(self._n_islands):
            archive = self._archives[i]
            elites = archive.best_elites(archive.size)
            states.append(
                IslandState(
                    island_id=i,
                    generation=self._generation,
                    archive_size=archive.size,
                    best_quality=archive.best_quality,
                    qd_score=archive.qd_score,
                    coverage=archive.coverage,
                    solutions=elites,
                )
            )
        return states

    def merge_archives(self) -> _IslandArchive:
        """Merge all island archives into a single archive.

        The merged archive takes the best solution for each cell across
        all islands.

        Returns
        -------
        _IslandArchive
            Merged archive.
        """
        dims = self._island_configs[0].archive_dims
        lower = self._archives[0]._lower
        upper = self._archives[0]._upper
        merged = _IslandArchive(dims, lower, upper)

        for archive in self._archives:
            for ci in archive._solutions:
                merged.add(
                    archive._solutions[ci],
                    float(archive._qualities[ci]),
                    archive._descriptors[ci],
                )

        return merged

    @property
    def migration_history(self) -> List[Dict[str, Any]]:
        return self._migration_history

    def stats(self) -> Dict[str, Any]:
        """Return summary statistics across all islands."""
        return {
            "n_islands": self._n_islands,
            "generation": self._generation,
            "island_sizes": [a.size for a in self._archives],
            "island_qd_scores": [a.qd_score for a in self._archives],
            "island_coverages": [a.coverage for a in self._archives],
            "island_best_qualities": [a.best_quality for a in self._archives],
            "total_migrations": len(self._migration_history),
        }


# ---------------------------------------------------------------------------
# IslandModelMAPElites (high-level runner)
# ---------------------------------------------------------------------------


class IslandModelMAPElites:
    """High-level island-model MAP-Elites with diverse operator configurations.

    Each island may use different mutation/crossover rates, operator
    selections, and archive configurations to promote diversity in the
    search.

    Parameters
    ----------
    n_islands : int
        Number of islands.
    score_fn : callable
        Scoring function.
    descriptor_fn : callable
        Descriptor function.
    mutation_fns : list of callable
        Available mutation operators.
    crossover_fns : list of callable, optional
        Available crossover operators.
    migration_policy : MigrationPolicy, optional
        Migration configuration.
    archive_dims : tuple of int
        Default archive dimensions.
    archive_bounds : tuple of np.ndarray
        ``(lower, upper)`` bounds for archive.
    seed : int
        Base random seed.

    Examples
    --------
    >>> islands = IslandModelMAPElites(
    ...     n_islands=4,
    ...     score_fn=bic_score,
    ...     descriptor_fn=structural_desc,
    ...     mutation_fns=[add_edge, remove_edge, reverse_edge],
    ...     archive_dims=(20, 20),
    ...     archive_bounds=(np.zeros(2), np.ones(2)),
    ... )
    >>> states = islands.run(data, n_generations=500, batch_size=32)
    >>> merged = islands.merged_archive
    """

    def __init__(
        self,
        n_islands: int,
        score_fn: Callable,
        descriptor_fn: Callable,
        mutation_fns: List[Callable],
        crossover_fns: Optional[List[Callable]] = None,
        migration_policy: Optional[MigrationPolicy] = None,
        archive_dims: Tuple[int, ...] = (20, 20),
        archive_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        seed: int = 42,
    ) -> None:
        if archive_bounds is None:
            archive_bounds = (np.zeros(2), np.ones(2))

        # Create diverse island configs
        configs = []
        for i in range(n_islands):
            configs.append(
                IslandConfig(
                    island_id=i,
                    mutation_rate=0.5 + 0.4 * np.sin(np.pi * i / n_islands),
                    crossover_rate=0.1 + 0.3 * np.cos(np.pi * i / n_islands),
                    archive_dims=archive_dims,
                    seed=seed + i * 7919,
                )
            )

        self._evaluator = DistributedEvaluator(
            n_islands=n_islands,
            score_fn=score_fn,
            descriptor_fn=descriptor_fn,
            mutation_fns=mutation_fns,
            crossover_fns=crossover_fns or [],
            migration_policy=migration_policy or MigrationPolicy(),
            island_configs=configs,
            archive_bounds=archive_bounds,
        )
        self._last_states: Optional[List[IslandState]] = None
        self._merged: Optional[_IslandArchive] = None

    def run(
        self,
        data: DataMatrix,
        n_generations: int,
        batch_size: int = 16,
        initial_dags: Optional[List[AdjacencyMatrix]] = None,
    ) -> List[IslandState]:
        """Run all islands.

        Returns the final state of each island.
        """
        self._last_states = self._evaluator.run(
            data, n_generations, batch_size, initial_dags
        )
        self._merged = None  # invalidate
        return self._last_states

    @property
    def merged_archive(self) -> _IslandArchive:
        """Merged archive from all islands (computed lazily)."""
        if self._merged is None:
            self._merged = self._evaluator.merge_archives()
        return self._merged

    @property
    def island_states(self) -> Optional[List[IslandState]]:
        return self._last_states

    def stats(self) -> Dict[str, Any]:
        return self._evaluator.stats()
