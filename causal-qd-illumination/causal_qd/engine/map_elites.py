"""Core MAP-Elites loop for causal structure discovery.

Provides the :class:`CausalMAPElites` engine which evolves a diverse
archive of DAGs using mutation and crossover operators, scoring each
candidate with a configurable quality function and projecting it into
a behavioural descriptor space.

Features
--------
- Multiple initialisation strategies (random, heuristic, warm-start).
- Adaptive operator selection via a multi-armed bandit (UCB1).
- Configurable selection strategies (uniform, curiosity, quality-proportional).
- Batch evaluation with optional parallelisation.
- Early stopping via convergence detection.
- Checkpoint save / resume.
- Callback hooks after each generation.
- Progress logging.
"""

from __future__ import annotations

import logging
import math
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from causal_qd.archive.archive_base import Archive, ArchiveEntry
from causal_qd.archive.grid_archive import GridArchive
from causal_qd.archive.stats import ArchiveStats, ArchiveStatsTracker
from causal_qd.types import (
    AdjacencyMatrix,
    BehavioralDescriptor,
    DataMatrix,
    ParamDict,
    QualityScore,
)

if TYPE_CHECKING:
    from causal_qd.engine.evaluator import BatchEvaluator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lightweight protocol-style type aliases for pluggable components
# ---------------------------------------------------------------------------

ScoreFn = Callable[[AdjacencyMatrix, DataMatrix], float]
DescriptorFn = Callable[[AdjacencyMatrix, DataMatrix], BehavioralDescriptor]
MutationOp = Callable[[AdjacencyMatrix, np.random.Generator], AdjacencyMatrix]
CrossoverOp = Callable[
    [AdjacencyMatrix, AdjacencyMatrix, np.random.Generator], AdjacencyMatrix
]
CallbackFn = Callable[["CausalMAPElites", int, Archive, ArchiveStatsTracker], None]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MAPElitesConfig:
    """Configuration for :class:`CausalMAPElites`.

    Parameters
    ----------
    mutation_prob : float
        Probability of applying a mutation (vs. crossover).
    crossover_rate : float
        Probability of crossover when mutation is not selected.
    archive_dims : Tuple[int, ...]
        Number of bins along each descriptor dimension (for grid archive).
    archive_ranges : Tuple[Tuple[float, float], ...]
        ``(low, high)`` bounds for each descriptor dimension.
    seed : int
        Random seed for reproducibility.
    selection_strategy : str
        One of ``"uniform"``, ``"curiosity"``, ``"quality_proportional"``.
    adaptive_operators : bool
        If ``True``, use UCB1-based adaptive operator selection.
    early_stopping_window : int
        Window size for convergence detection (0 = disabled).
    early_stopping_threshold : float
        Relative QD-score improvement threshold for early stopping.
    checkpoint_interval : int
        Save a checkpoint every N generations (0 = disabled).
    checkpoint_dir : str
        Directory for checkpoint files.
    log_interval : int
        Log progress every N generations.
    """

    mutation_prob: float = 0.7
    crossover_rate: float = 0.3
    archive_dims: Tuple[int, ...] = (20, 20)
    archive_ranges: Tuple[Tuple[float, float], ...] = ((0.0, 1.0), (0.0, 1.0))
    seed: int = 42
    selection_strategy: str = "uniform"
    adaptive_operators: bool = False
    early_stopping_window: int = 0
    early_stopping_threshold: float = 1e-4
    checkpoint_interval: int = 0
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10


# ---------------------------------------------------------------------------
# Iteration statistics
# ---------------------------------------------------------------------------

@dataclass
class IterationStats:
    """Statistics collected after each MAP-Elites iteration."""

    iteration: int
    archive_size: int
    best_quality: float
    mean_quality: float
    improvements: int
    fills: int = 0
    replacements: int = 0
    elapsed_time: float = 0.0


# ---------------------------------------------------------------------------
# Multi-armed bandit for adaptive operator selection
# ---------------------------------------------------------------------------

class _OperatorBandit:
    """UCB1-based bandit for selecting mutation / crossover operators.

    Each operator is an arm.  Reward is 1 if the offspring was inserted
    into the archive, 0 otherwise.

    Parameters
    ----------
    n_arms : int
        Number of operators.
    exploration : float
        UCB1 exploration constant (higher → more exploration).
    """

    def __init__(self, n_arms: int, exploration: float = 1.41) -> None:
        self._n = n_arms
        self._c = exploration
        self._counts = np.zeros(n_arms, dtype=np.float64)
        self._rewards = np.zeros(n_arms, dtype=np.float64)
        self._total = 0

    def select(self, rng: np.random.Generator) -> int:
        """Select an arm using UCB1.

        Parameters
        ----------
        rng : np.random.Generator

        Returns
        -------
        int
            Selected arm index.
        """
        # Ensure each arm is tried at least once
        untried = np.where(self._counts == 0)[0]
        if len(untried) > 0:
            return int(rng.choice(untried))

        means = self._rewards / np.maximum(self._counts, 1)
        exploration = self._c * np.sqrt(
            np.log(self._total) / np.maximum(self._counts, 1)
        )
        ucb = means + exploration
        return int(np.argmax(ucb))

    def update(self, arm: int, reward: float) -> None:
        """Update the statistics for *arm*.

        Parameters
        ----------
        arm : int
        reward : float
        """
        self._counts[arm] += 1
        self._rewards[arm] += reward
        self._total += 1

    @property
    def arm_stats(self) -> List[Tuple[int, float, float]]:
        """Return ``(count, total_reward, mean_reward)`` per arm."""
        stats = []
        for i in range(self._n):
            c = int(self._counts[i])
            r = float(self._rewards[i])
            m = r / max(c, 1)
            stats.append((c, r, m))
        return stats


# ---------------------------------------------------------------------------
# Grid archive (lightweight built-in used by CausalMAPElites)
# ---------------------------------------------------------------------------

class _GridArchive:
    """Simple grid archive mapping descriptor cells to best-quality entries.

    Used internally by :class:`CausalMAPElites` when the external archive
    interface is not provided.
    """

    def __init__(
        self,
        dims: Tuple[int, ...],
        ranges: Tuple[Tuple[float, float], ...],
    ) -> None:
        self._dims = dims
        self._ranges = ranges
        self._grid: Dict[Tuple[int, ...], ArchiveEntry] = {}
        self._fill_count: int = 0
        self._replace_count: int = 0

    def _to_index(self, descriptor: BehavioralDescriptor) -> Tuple[int, ...]:
        idx: List[int] = []
        for i, (lo, hi) in enumerate(self._ranges):
            span = hi - lo if hi > lo else 1.0
            bin_idx = int((float(descriptor[i]) - lo) / span * self._dims[i])
            bin_idx = max(0, min(self._dims[i] - 1, bin_idx))
            idx.append(bin_idx)
        return tuple(idx)

    def add(self, entry: ArchiveEntry) -> bool:
        """Add *entry* if it improves the cell or the cell is empty."""
        idx = self._to_index(entry.descriptor)
        existing = self._grid.get(idx)
        if existing is None:
            self._grid[idx] = entry
            self._fill_count += 1
            return True
        if entry.quality > existing.quality:
            self._grid[idx] = entry
            self._replace_count += 1
            return True
        return False

    def sample(self, n: int, rng: np.random.Generator) -> List[ArchiveEntry]:
        """Uniformly sample *n* entries (with replacement)."""
        entries = list(self._grid.values())
        if not entries:
            return []
        indices = rng.integers(0, len(entries), size=n)
        return [entries[i] for i in indices]

    def sample_curiosity(self, n: int, rng: np.random.Generator) -> List[ArchiveEntry]:
        """Uniform sampling (no curiosity info in simple archive)."""
        return self.sample(n, rng)

    def sample_quality_proportional(
        self, n: int, rng: np.random.Generator
    ) -> List[ArchiveEntry]:
        """Sample entries proportional to quality."""
        entries = list(self._grid.values())
        if not entries:
            return []
        qualities = np.array([e.quality for e in entries], dtype=np.float64)
        shifted = qualities - qualities.min()
        total = shifted.sum()
        if total < 1e-15:
            probs = np.ones(len(entries)) / len(entries)
        else:
            probs = shifted / total
        chosen = rng.choice(len(entries), size=n, replace=True, p=probs)
        return [entries[i] for i in chosen]

    @property
    def size(self) -> int:
        return len(self._grid)

    @property
    def entries(self) -> List[ArchiveEntry]:
        return list(self._grid.values())

    @property
    def fill_count(self) -> int:
        return self._fill_count

    @property
    def replace_count(self) -> int:
        return self._replace_count

    def best(self) -> Optional[ArchiveEntry]:
        if not self._grid:
            return None
        return max(self._grid.values(), key=lambda e: e.quality)

    def qd_score(self) -> float:
        return sum(e.quality for e in self._grid.values())

    def coverage(self) -> float:
        total = 1
        for d in self._dims:
            total *= d
        return len(self._grid) / max(total, 1)

    def clear(self) -> None:
        self._grid.clear()
        self._fill_count = 0
        self._replace_count = 0

    def save(self, path: str) -> None:
        """Pickle the grid to disk."""
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "grid": self._grid,
                    "dims": self._dims,
                    "ranges": self._ranges,
                    "fill_count": self._fill_count,
                    "replace_count": self._replace_count,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "_GridArchive":
        """Restore a pickled grid archive."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        archive = cls(dims=state["dims"], ranges=state["ranges"])
        archive._grid = state["grid"]
        archive._fill_count = state.get("fill_count", 0)
        archive._replace_count = state.get("replace_count", 0)
        return archive


# ---------------------------------------------------------------------------
# CausalMAPElites
# ---------------------------------------------------------------------------

class CausalMAPElites:
    """MAP-Elites engine for causal discovery.

    Maintains an archive of diverse DAGs and evolves them using mutation
    and crossover operators, guided by a scoring function and projected
    into a behavioural descriptor space.

    Parameters
    ----------
    mutations : List[MutationOp]
        Mutation operators ``(adj, rng) -> adj``.
    crossovers : List[CrossoverOp]
        Crossover operators ``(adj1, adj2, rng) -> adj``.
    descriptor_fn : DescriptorFn
        ``(adjacency, data) -> BehavioralDescriptor``.
    score_fn : ScoreFn
        ``(adjacency, data) -> float``.
    config : MAPElitesConfig or None
    callbacks : List[CallbackFn]
        Functions called after each generation with
        ``(engine, gen, archive, stats_tracker)``.
    """

    def __init__(
        self,
        mutations: List[MutationOp],
        crossovers: List[CrossoverOp],
        descriptor_fn: DescriptorFn,
        score_fn: ScoreFn,
        config: MAPElitesConfig | None = None,
        callbacks: Optional[List[CallbackFn]] = None,
        evaluator: Optional["BatchEvaluator"] = None,
    ) -> None:
        self._mutations = list(mutations)
        self._crossovers = list(crossovers)
        self._descriptor_fn = descriptor_fn
        self._score_fn = score_fn
        self._config = config or MAPElitesConfig()
        self._rng = np.random.default_rng(self._config.seed)
        self._archive = _GridArchive(
            dims=self._config.archive_dims,
            ranges=self._config.archive_ranges,
        )
        self._history: List[IterationStats] = []
        self._stats_tracker = ArchiveStatsTracker(
            window_size=max(self._config.early_stopping_window, 20),
            convergence_threshold=self._config.early_stopping_threshold,
        )
        self._iteration = 0
        self._callbacks: List[CallbackFn] = callbacks or []
        self._evaluator = evaluator
        self._start_time: float = 0.0
        self._stopped_early: bool = False

        # Adaptive operator selection
        n_ops = len(self._mutations) + len(self._crossovers)
        self._bandit: Optional[_OperatorBandit] = None
        if self._config.adaptive_operators and n_ops > 0:
            self._bandit = _OperatorBandit(n_ops)

    # -- properties ----------------------------------------------------------

    @property
    def archive(self) -> _GridArchive:
        """The current MAP-Elites archive."""
        return self._archive

    @property
    def history(self) -> List[IterationStats]:
        """Per-iteration convergence statistics."""
        return list(self._history)

    @property
    def stats_tracker(self) -> ArchiveStatsTracker:
        """Longitudinal statistics tracker."""
        return self._stats_tracker

    @property
    def iteration(self) -> int:
        """Current iteration counter."""
        return self._iteration

    @property
    def stopped_early(self) -> bool:
        """Whether the last ``run()`` was terminated by early stopping."""
        return self._stopped_early

    # -- main loop -----------------------------------------------------------

    def run(
        self,
        data: DataMatrix,
        n_iterations: int,
        batch_size: int = 16,
        initial_dags: Optional[List[AdjacencyMatrix]] = None,
    ) -> _GridArchive:
        """Execute the MAP-Elites loop.

        Parameters
        ----------
        data : DataMatrix
            Observed data matrix (N × p).
        n_iterations : int
            Number of evolution iterations.
        batch_size : int
            Number of offspring generated per iteration.
        initial_dags : List[AdjacencyMatrix], optional
            Seed DAGs to insert before the loop begins.

        Returns
        -------
        _GridArchive
            The filled archive after all iterations.
        """
        self._start_time = time.monotonic()
        self._stopped_early = False

        logger.info(
            "Starting MAP-Elites: %d iterations, batch_size=%d",
            n_iterations,
            batch_size,
        )

        # Warm-start with provided initial DAGs
        if initial_dags:
            self._seed_archive(initial_dags, data)

        for _ in range(n_iterations):
            stats = self.step(data, batch_size)

            # Early stopping
            if self._config.early_stopping_window > 0:
                if self._stats_tracker.is_converged():
                    logger.info(
                        "Early stopping at iteration %d: QD-score converged.",
                        self._iteration,
                    )
                    self._stopped_early = True
                    break

            # Checkpoint
            if (
                self._config.checkpoint_interval > 0
                and self._iteration % self._config.checkpoint_interval == 0
            ):
                self._save_checkpoint()

        logger.info(
            "MAP-Elites finished: archive size=%d, qd_score=%.4f",
            self._archive.size,
            self._archive.qd_score(),
        )
        return self._archive

    def step(self, data: DataMatrix, batch_size: int = 16) -> IterationStats:
        """Execute a single MAP-Elites iteration.

        Parameters
        ----------
        data : DataMatrix
            Observed data matrix.
        batch_size : int
            Number of offspring to generate.

        Returns
        -------
        IterationStats
            Statistics for this iteration.
        """
        self._iteration += 1
        iter_start = time.monotonic()

        # If the archive is empty, seed with random DAGs
        if self._archive.size == 0:
            n_nodes = data.shape[1]
            candidates = self._random_dags(batch_size, n_nodes)
            op_indices = [-1] * batch_size
        else:
            parents = self._select_parents(batch_size, self._rng)
            candidates, op_indices = self._generate_offspring(parents, self._rng)

        scored = self._evaluate_batch(candidates, data)

        improvements = 0
        fills = 0
        replacements = 0
        prev_size = self._archive.size
        for i, (dag, (quality, descriptor)) in enumerate(zip(candidates, scored)):
            entry = ArchiveEntry(
                solution=dag,
                quality=quality,
                descriptor=descriptor,
                metadata={"iteration": self._iteration},
            )
            if self._archive.add(entry):
                improvements += 1
                if self._archive.size > prev_size:
                    fills += 1
                    prev_size = self._archive.size
                else:
                    replacements += 1
                # Update bandit with success
                if self._bandit is not None and op_indices[i] >= 0:
                    self._bandit.update(op_indices[i], 1.0)
            else:
                if self._bandit is not None and op_indices[i] >= 0:
                    self._bandit.update(op_indices[i], 0.0)

        elapsed = time.monotonic() - iter_start
        qualities = [q for q, _ in scored]
        stats = IterationStats(
            iteration=self._iteration,
            archive_size=self._archive.size,
            best_quality=max(qualities) if qualities else 0.0,
            mean_quality=float(np.mean(qualities)) if qualities else 0.0,
            improvements=improvements,
            fills=fills,
            replacements=replacements,
            elapsed_time=elapsed,
        )
        self._history.append(stats)

        # Record in longitudinal tracker (create a minimal archive wrapper)
        self._stats_tracker.record(
            self._iteration,
            _ArchiveAdapter(self._archive),
            improvements=improvements,
        )

        # Callbacks
        for cb in self._callbacks:
            try:
                cb(self, self._iteration, self._archive, self._stats_tracker)
            except Exception:
                logger.exception("Callback error at iteration %d", self._iteration)

        # Logging
        if self._iteration % self._config.log_interval == 0:
            logger.info(
                "Iter %d: archive=%d, best=%.4f, improved=%d, "
                "fills=%d, replacements=%d, time=%.3fs",
                stats.iteration,
                stats.archive_size,
                stats.best_quality,
                stats.improvements,
                stats.fills,
                stats.replacements,
                stats.elapsed_time,
            )
        return stats

    # -- initialisation helpers -----------------------------------------------

    def _seed_archive(
        self,
        dags: List[AdjacencyMatrix],
        data: DataMatrix,
    ) -> int:
        """Insert seed DAGs into the archive.

        Returns the number of successfully inserted DAGs.
        """
        inserted = 0
        for dag in dags:
            quality = self._score_fn(dag, data)
            descriptor = self._descriptor_fn(dag, data)
            entry = ArchiveEntry(
                solution=dag,
                quality=quality,
                descriptor=descriptor,
                metadata={"iteration": 0, "seed": True},
            )
            if self._archive.add(entry):
                inserted += 1
        logger.info(
            "Seeded archive with %d / %d DAGs", inserted, len(dags)
        )
        return inserted

    # -- selection -----------------------------------------------------------

    def _select_parents(
        self, n: int, rng: np.random.Generator
    ) -> List[ArchiveEntry]:
        """Select *n* parents from the archive using the configured strategy."""
        strategy = self._config.selection_strategy
        if strategy == "curiosity":
            return self._archive.sample_curiosity(n, rng)
        elif strategy == "quality_proportional":
            return self._archive.sample_quality_proportional(n, rng)
        else:
            return self._archive.sample(n, rng)

    # -- offspring generation ------------------------------------------------

    def _generate_offspring(
        self,
        parents: List[ArchiveEntry],
        rng: np.random.Generator,
    ) -> Tuple[List[AdjacencyMatrix], List[int]]:
        """Apply mutation or crossover to each parent.

        Returns
        -------
        Tuple[List[AdjacencyMatrix], List[int]]
            Offspring and the operator index used for each.
        """
        offspring: List[AdjacencyMatrix] = []
        op_indices: List[int] = []
        n_mut = len(self._mutations)

        for i, parent in enumerate(parents):
            if self._bandit is not None:
                # Adaptive operator selection
                arm = self._bandit.select(rng)
                if arm < n_mut:
                    child = self._mutations[arm](parent.solution.copy(), rng)
                else:
                    cx_idx = arm - n_mut
                    other = parents[rng.integers(0, len(parents))]
                    child = self._crossovers[cx_idx](
                        parent.solution.copy(), other.solution.copy(), rng
                    )
                    # Handle crossover operators that return two children
                    if isinstance(child, tuple):
                        child = child[0]  # Take first child; second available for future use
                op_indices.append(arm)
            elif (
                rng.random() < self._config.mutation_prob
                or not self._crossovers
            ):
                mut_idx = int(rng.integers(0, len(self._mutations)))
                child = self._mutations[mut_idx](parent.solution.copy(), rng)
                op_indices.append(mut_idx)
            else:
                other = parents[rng.integers(0, len(parents))]
                cx_idx = int(rng.integers(0, len(self._crossovers)))
                child = self._crossovers[cx_idx](
                    parent.solution.copy(), other.solution.copy(), rng
                )
                # Handle crossover operators that return two children
                if isinstance(child, tuple):
                    child = child[0]  # Take first child; second available for future use
                op_indices.append(n_mut + cx_idx)
            offspring.append(child)
        return offspring, op_indices

    # -- evaluation ----------------------------------------------------------

    def _evaluate_batch(
        self,
        candidates: List[AdjacencyMatrix],
        data: DataMatrix,
    ) -> List[Tuple[QualityScore, BehavioralDescriptor]]:
        """Score and compute descriptors for a batch of candidates."""
        if self._evaluator is not None:
            return self._evaluator.evaluate(candidates, data)
        results: List[Tuple[QualityScore, BehavioralDescriptor]] = []
        for dag in candidates:
            try:
                quality = self._score_fn(dag, data)
                descriptor = self._descriptor_fn(dag, data)
                if not np.isfinite(quality):
                    logger.warning("Non-finite quality score; skipping candidate.")
                    quality = float("-inf")
                results.append((quality, descriptor))
            except Exception:
                logger.warning("Evaluation failed for a candidate; skipping.")
                results.append((float("-inf"), np.zeros(len(self._config.archive_dims), dtype=np.float64)))
        return results

    # -- random DAG generation -----------------------------------------------

    def _random_dags(self, n: int, n_nodes: int) -> List[AdjacencyMatrix]:
        """Generate *n* random DAGs with *n_nodes* nodes.

        Uses random permutation ordering to guarantee acyclicity.
        """
        dags: List[AdjacencyMatrix] = []
        for _ in range(n):
            perm = self._rng.permutation(n_nodes)
            adj = np.zeros((n_nodes, n_nodes), dtype=np.int8)
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if self._rng.random() < 0.3:
                        adj[perm[i], perm[j]] = 1
            dags.append(adj)
        return dags

    # -- checkpointing -------------------------------------------------------

    def _save_checkpoint(self) -> None:
        """Save a checkpoint to disk."""
        ckpt_dir = Path(self._config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"checkpoint_iter{self._iteration}.pkl"
        state = {
            "iteration": self._iteration,
            "history": self._history,
            "rng_state": self._rng.bit_generator.state,
        }
        self._archive.save(str(ckpt_dir / f"archive_iter{self._iteration}.pkl"))
        with open(ckpt_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Checkpoint saved at iteration %d", self._iteration)

    def load_checkpoint(self, path: str) -> None:
        """Resume from a checkpoint.

        Parameters
        ----------
        path : str
            Path to the checkpoint ``.pkl`` file.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._iteration = state["iteration"]
        self._history = state["history"]
        self._rng.bit_generator.state = state["rng_state"]

        # Try to load the matching archive
        archive_path = str(Path(path).parent / f"archive_iter{self._iteration}.pkl")
        if Path(archive_path).exists():
            self._archive = _GridArchive.load(archive_path)
        logger.info("Resumed from checkpoint at iteration %d", self._iteration)

    # -- accessors -----------------------------------------------------------

    def operator_stats(self) -> Optional[List[Tuple[int, float, float]]]:
        """Return per-operator statistics if adaptive selection is enabled.

        Returns
        -------
        List[Tuple[int, float, float]] or None
            ``(count, total_reward, mean_reward)`` per operator arm.
        """
        if self._bandit is None:
            return None
        return self._bandit.arm_stats

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the current engine state.

        Returns
        -------
        Dict[str, Any]
        """
        return {
            "iteration": self._iteration,
            "archive_size": self._archive.size,
            "qd_score": self._archive.qd_score(),
            "best_quality": (
                self._archive.best().quality if self._archive.size > 0 else None
            ),
            "stopped_early": self._stopped_early,
            "operator_stats": self.operator_stats(),
        }


# ---------------------------------------------------------------------------
# Adapter so _GridArchive looks like an Archive for ArchiveStatsTracker
# ---------------------------------------------------------------------------

class _ArchiveAdapter:
    """Thin adapter wrapping _GridArchive with the Archive protocol."""

    def __init__(self, grid: _GridArchive) -> None:
        self._grid = grid

    def elites(self) -> List[ArchiveEntry]:
        return self._grid.entries

    def coverage(self) -> float:
        return self._grid.coverage()

    def qd_score(self) -> float:
        return self._grid.qd_score()
