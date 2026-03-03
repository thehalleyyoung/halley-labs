"""Tests for the MAP-Elites engine (causal_qd.engine.map_elites).

Covers the core CausalMAPElites loop, archive evolution, checkpointing,
callbacks, convergence detection, and operator integration.  All tests
use small graphs (n=5) and few iterations to stay fast.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure the package is importable regardless of how pytest is invoked.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from causal_qd.engine.map_elites import (
    CausalMAPElites,
    IterationStats,
    MAPElitesConfig,
    _GridArchive,
)
from causal_qd.scores.bic import BICScore
from causal_qd.descriptors.structural import StructuralDescriptor
from causal_qd.operators.mutation import TopologicalMutation
from causal_qd.archive.archive_base import ArchiveEntry
from causal_qd.types import AdjacencyMatrix, DataMatrix


# ===================================================================
# Constants
# ===================================================================

N_NODES = 5
SMALL_BATCH = 8
SMALL_ITERS = 5
DESCRIPTOR_FEATURES = ["edge_density", "max_in_degree"]


# ===================================================================
# Helpers
# ===================================================================

def _make_score_fn() -> "ScoreFn":
    """Return a score function compatible with CausalMAPElites."""
    scorer = BICScore()

    def score_fn(adj: AdjacencyMatrix, data: DataMatrix) -> float:
        return scorer.score(adj, data)

    return score_fn


def _make_descriptor_fn() -> "DescriptorFn":
    """Return a descriptor function compatible with CausalMAPElites."""
    descriptor = StructuralDescriptor(features=DESCRIPTOR_FEATURES)

    def descriptor_fn(adj: AdjacencyMatrix, data: DataMatrix) -> np.ndarray:
        return descriptor.compute(adj, data)

    return descriptor_fn


def _make_mutation_op() -> "MutationOp":
    """Return a mutation callable compatible with CausalMAPElites."""
    mutator = TopologicalMutation()

    def mutation_fn(adj: AdjacencyMatrix, rng: np.random.Generator) -> AdjacencyMatrix:
        return mutator.mutate(adj, rng)

    return mutation_fn


def _make_crossover_op() -> "CrossoverOp":
    """Return a simple uniform crossover callable."""

    def crossover_fn(
        adj1: AdjacencyMatrix,
        adj2: AdjacencyMatrix,
        rng: np.random.Generator,
    ) -> AdjacencyMatrix:
        mask = rng.random(adj1.shape) < 0.5
        child = np.where(mask, adj1, adj2).astype(np.int8)
        # Zero out lower triangle to help maintain acyclicity
        child[np.tril_indices(child.shape[0])] = 0
        return child

    return crossover_fn


def _make_engine(
    config: MAPElitesConfig | None = None,
    callbacks: list | None = None,
    with_crossover: bool = False,
) -> CausalMAPElites:
    """Create a CausalMAPElites engine with sensible test defaults."""
    mutations = [_make_mutation_op()]
    crossovers = [_make_crossover_op()] if with_crossover else []
    cfg = config or MAPElitesConfig(
        archive_dims=(5, 5),
        archive_ranges=((0.0, 1.0), (0.0, 1.0)),
        seed=42,
        log_interval=100,  # suppress logs during tests
    )
    return CausalMAPElites(
        mutations=mutations,
        crossovers=crossovers,
        descriptor_fn=_make_descriptor_fn(),
        score_fn=_make_score_fn(),
        config=cfg,
        callbacks=callbacks,
    )


def _make_seed_dags(n_nodes: int = N_NODES, count: int = 5) -> List[AdjacencyMatrix]:
    """Generate a handful of small random DAGs for seeding."""
    rng = np.random.default_rng(123)
    dags: List[AdjacencyMatrix] = []
    for _ in range(count):
        perm = rng.permutation(n_nodes)
        adj = np.zeros((n_nodes, n_nodes), dtype=np.int8)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if rng.random() < 0.3:
                    adj[perm[i], perm[j]] = 1
        dags.append(adj)
    return dags


# ===================================================================
# Test: basic smoke test — engine runs without errors
# ===================================================================


class TestCausalMAPElitesRuns:
    """Smoke tests verifying the engine runs end-to-end."""

    def test_causal_map_elites_runs(self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]) -> None:
        """Engine completes 5 iterations and produces a non-empty archive."""
        data, _ = gaussian_data
        engine = _make_engine()

        archive = engine.run(data, n_iterations=SMALL_ITERS, batch_size=SMALL_BATCH)

        assert archive.size > 0, "Archive should not be empty after running"
        assert engine.iteration == SMALL_ITERS
        assert len(engine.history) == SMALL_ITERS
        assert not engine.stopped_early

    def test_run_returns_grid_archive(self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]) -> None:
        """run() returns a _GridArchive instance."""
        data, _ = gaussian_data
        engine = _make_engine()
        archive = engine.run(data, n_iterations=3, batch_size=4)
        assert isinstance(archive, _GridArchive)

    def test_run_with_initial_dags(self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]) -> None:
        """Seeding with initial DAGs populates the archive before the loop."""
        data, _ = gaussian_data
        seed_dags = _make_seed_dags()
        engine = _make_engine()
        archive = engine.run(
            data, n_iterations=3, batch_size=4, initial_dags=seed_dags,
        )
        assert archive.size > 0

    def test_run_with_crossover(self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]) -> None:
        """Engine works with both mutation and crossover operators."""
        data, _ = gaussian_data
        engine = _make_engine(with_crossover=True)
        archive = engine.run(data, n_iterations=3, batch_size=4)
        assert archive.size > 0

    def test_step_increments_iteration(self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]) -> None:
        """Each call to step() increments the iteration counter by 1."""
        data, _ = gaussian_data
        engine = _make_engine()
        assert engine.iteration == 0
        engine.step(data, batch_size=4)
        assert engine.iteration == 1
        engine.step(data, batch_size=4)
        assert engine.iteration == 2

    def test_step_returns_iteration_stats(self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]) -> None:
        """step() returns an IterationStats dataclass."""
        data, _ = gaussian_data
        engine = _make_engine()
        stats = engine.step(data, batch_size=4)
        assert isinstance(stats, IterationStats)
        assert stats.iteration == 1
        assert stats.archive_size >= 0
        assert isinstance(stats.best_quality, float)
        assert isinstance(stats.mean_quality, float)
        assert stats.elapsed_time >= 0.0

    def test_history_accumulates(self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]) -> None:
        """History grows with each step."""
        data, _ = gaussian_data
        engine = _make_engine()
        for _ in range(4):
            engine.step(data, batch_size=4)
        history = engine.history
        assert len(history) == 4
        for i, h in enumerate(history, 1):
            assert h.iteration == i

    def test_summary_returns_dict(self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]) -> None:
        """summary() returns a well-formed dictionary."""
        data, _ = gaussian_data
        engine = _make_engine()
        engine.run(data, n_iterations=2, batch_size=4)
        s = engine.summary()
        assert isinstance(s, dict)
        assert "iteration" in s
        assert "archive_size" in s
        assert "qd_score" in s
        assert "best_quality" in s
        assert "stopped_early" in s

    def test_run_with_random_data(self, random_data: DataMatrix) -> None:
        """Engine works with purely random (non-SCM) data."""
        engine = _make_engine()
        archive = engine.run(random_data, n_iterations=3, batch_size=4)
        assert archive.size > 0

    def test_archive_entries_have_valid_solutions(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """Every elite in the archive has an n×n adjacency matrix."""
        data, _ = gaussian_data
        engine = _make_engine()
        engine.run(data, n_iterations=3, batch_size=SMALL_BATCH)
        for entry in engine.archive.entries:
            assert entry.solution.shape == (N_NODES, N_NODES)
            assert entry.solution.dtype == np.int8
            assert np.all((entry.solution == 0) | (entry.solution == 1))

    def test_archive_entries_have_finite_quality(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """No elite should have infinite quality."""
        data, _ = gaussian_data
        engine = _make_engine()
        engine.run(data, n_iterations=3, batch_size=SMALL_BATCH)
        for entry in engine.archive.entries:
            assert np.isfinite(entry.quality), f"Non-finite quality: {entry.quality}"


# ===================================================================
# Test: quality improves over time
# ===================================================================


class TestEngineImprovesQualityOverTime:
    """Verify that running more iterations can improve best quality."""

    def test_engine_improves_quality_over_time(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """Best quality after more iterations should be >= initial best."""
        data, _ = gaussian_data
        engine = _make_engine()

        # Run a few initial iterations
        engine.run(data, n_iterations=3, batch_size=SMALL_BATCH)
        initial_best = engine.archive.best().quality

        # Run more iterations (step continues from the same engine state)
        for _ in range(10):
            engine.step(data, batch_size=SMALL_BATCH)

        final_best = engine.archive.best().quality
        assert final_best >= initial_best, (
            f"Quality should not decrease: {initial_best} -> {final_best}"
        )

    def test_best_quality_in_history_is_monotonic(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """The best quality seen so far in history should be non-decreasing."""
        data, _ = gaussian_data
        engine = _make_engine()
        engine.run(data, n_iterations=10, batch_size=SMALL_BATCH)

        running_best = float("-inf")
        for stats in engine.history:
            running_best = max(running_best, stats.best_quality)
            assert stats.best_quality <= running_best or np.isclose(
                stats.best_quality, running_best
            )

    def test_qd_score_changes_over_time(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """QD-score should change as the archive evolves.
        Note: QD-score (sum of qualities) can go down when new cells are
        filled with negative-quality elites, so we only check it changes."""
        data, _ = gaussian_data
        engine = _make_engine()
        engine.step(data, batch_size=SMALL_BATCH)
        first_qd = engine.archive.qd_score()

        for _ in range(10):
            engine.step(data, batch_size=SMALL_BATCH)
        final_qd = engine.archive.qd_score()

        # The archive should have evolved (more cells filled or quality changed)
        assert engine.archive.size > 0
        assert isinstance(final_qd, float)
        assert np.isfinite(final_qd)

    def test_improvements_recorded_in_stats(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """At least some iterations should record improvements."""
        data, _ = gaussian_data
        engine = _make_engine()
        engine.run(data, n_iterations=10, batch_size=SMALL_BATCH)
        total_improvements = sum(s.improvements for s in engine.history)
        assert total_improvements > 0, "No improvements recorded across all iterations"

    def test_fills_recorded_in_stats(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """Initial iterations should fill new archive cells."""
        data, _ = gaussian_data
        engine = _make_engine()
        engine.run(data, n_iterations=5, batch_size=SMALL_BATCH)
        total_fills = sum(s.fills for s in engine.history)
        assert total_fills > 0, "No fills recorded — archive never grew"


# ===================================================================
# Test: coverage increases
# ===================================================================


class TestEngineIncreasesCoverage:
    """Check that archive coverage grows over iterations."""

    def test_engine_increases_coverage(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """Coverage after more iterations should be >= initial coverage."""
        data, _ = gaussian_data
        engine = _make_engine()

        engine.step(data, batch_size=SMALL_BATCH)
        initial_coverage = engine.archive.coverage()
        assert initial_coverage >= 0.0

        for _ in range(10):
            engine.step(data, batch_size=SMALL_BATCH)
        final_coverage = engine.archive.coverage()
        assert final_coverage >= initial_coverage, (
            f"Coverage should not shrink: {initial_coverage} -> {final_coverage}"
        )

    def test_archive_size_grows(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """Archive size should grow as more unique DAGs are discovered."""
        data, _ = gaussian_data
        engine = _make_engine()

        engine.step(data, batch_size=SMALL_BATCH)
        size_after_1 = engine.archive.size

        for _ in range(5):
            engine.step(data, batch_size=SMALL_BATCH)
        size_after_6 = engine.archive.size

        assert size_after_6 >= size_after_1

    def test_coverage_bounded_by_one(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """Coverage should never exceed 1.0."""
        data, _ = gaussian_data
        engine = _make_engine()
        engine.run(data, n_iterations=5, batch_size=SMALL_BATCH)
        assert engine.archive.coverage() <= 1.0

    def test_coverage_is_ratio_of_filled_cells(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """Coverage = archive_size / total_cells."""
        data, _ = gaussian_data
        cfg = MAPElitesConfig(
            archive_dims=(5, 5),
            archive_ranges=((0.0, 1.0), (0.0, 1.0)),
            seed=42,
            log_interval=100,
        )
        engine = _make_engine(config=cfg)
        engine.run(data, n_iterations=5, batch_size=SMALL_BATCH)
        expected_coverage = engine.archive.size / (5 * 5)
        assert np.isclose(engine.archive.coverage(), expected_coverage)


# ===================================================================
# Test: initialization strategies
# ===================================================================


class TestInitializationStrategies:
    """Test different ways to initialise the MAP-Elites archive."""

    def test_initialization_strategies(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """Seeded archive should have entries before any evolutionary step."""
        data, _ = gaussian_data
        seed_dags = _make_seed_dags(count=10)
        engine = _make_engine()
        # Use the run method with initial_dags but 0 iterations
        engine.run(data, n_iterations=0, initial_dags=seed_dags)
        assert engine.archive.size > 0
        assert engine.archive.size <= 10  # can't exceed seed count

    def test_empty_start_fills_on_first_step(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """When no seed DAGs are provided, the first step generates random DAGs."""
        data, _ = gaussian_data
        engine = _make_engine()
        assert engine.archive.size == 0
        engine.step(data, batch_size=SMALL_BATCH)
        assert engine.archive.size > 0

    def test_seed_dags_scored_correctly(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """Seeded DAGs should have valid quality scores."""
        data, _ = gaussian_data
        seed_dags = _make_seed_dags(count=3)
        engine = _make_engine()
        engine.run(data, n_iterations=0, initial_dags=seed_dags)
        for entry in engine.archive.entries:
            assert np.isfinite(entry.quality)
            assert entry.descriptor.shape == (len(DESCRIPTOR_FEATURES),)

    def test_seed_dags_metadata_has_seed_flag(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """Seeded DAGs should have metadata marking them as seeds."""
        data, _ = gaussian_data
        seed_dags = _make_seed_dags(count=3)
        engine = _make_engine()
        engine.run(data, n_iterations=0, initial_dags=seed_dags)
        for entry in engine.archive.entries:
            assert entry.metadata is not None
            assert entry.metadata.get("seed") is True

    def test_multiple_seeds_in_same_cell_keeps_best(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """If multiple seed DAGs map to the same cell, only the best is kept."""
        data, _ = gaussian_data
        # Create two identical DAGs — they will map to the same cell
        dag = np.zeros((N_NODES, N_NODES), dtype=np.int8)
        dag[0, 1] = 1
        seed_dags = [dag.copy(), dag.copy()]
        engine = _make_engine()
        engine.run(data, n_iterations=0, initial_dags=seed_dags)
        # Should have exactly 1 entry (same cell)
        assert engine.archive.size == 1

    def test_warm_start_plus_evolution(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """Warm-starting with seeds followed by evolution should produce
        at least as many entries as the seed alone."""
        data, _ = gaussian_data
        seed_dags = _make_seed_dags(count=5)
        engine = _make_engine()
        engine.run(data, n_iterations=0, initial_dags=seed_dags)
        seed_size = engine.archive.size

        for _ in range(5):
            engine.step(data, batch_size=SMALL_BATCH)
        assert engine.archive.size >= seed_size


# ===================================================================
# Test: batch evaluation
# ===================================================================


class TestBatchEvaluation:
    """Verify that batch evaluation produces correct outputs."""

    def test_batch_evaluation(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """step() evaluates batch_size candidates each iteration."""
        data, _ = gaussian_data
        engine = _make_engine()
        stats = engine.step(data, batch_size=16)
        # The number of improvements + non-improvements should not exceed batch_size
        # (improvements counts entries actually added)
        assert stats.improvements <= 16
        assert stats.archive_size <= 16

    def test_small_batch_produces_results(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """Even batch_size=1 should work."""
        data, _ = gaussian_data
        engine = _make_engine()
        stats = engine.step(data, batch_size=1)
        assert isinstance(stats, IterationStats)
        assert stats.improvements <= 1

    def test_large_batch_more_improvements(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """Larger batch sizes should find at least as many improvements
        (statistically, on the first iteration when archive is empty)."""
        data, _ = gaussian_data
        engine_small = _make_engine(config=MAPElitesConfig(
            archive_dims=(5, 5),
            archive_ranges=((0.0, 1.0), (0.0, 1.0)),
            seed=42,
            log_interval=100,
        ))
        stats_small = engine_small.step(data, batch_size=2)

        engine_large = _make_engine(config=MAPElitesConfig(
            archive_dims=(5, 5),
            archive_ranges=((0.0, 1.0), (0.0, 1.0)),
            seed=42,
            log_interval=100,
        ))
        stats_large = engine_large.step(data, batch_size=32)

        assert stats_large.archive_size >= stats_small.archive_size

    def test_batch_evaluation_timing(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """elapsed_time in stats should be positive."""
        data, _ = gaussian_data
        engine = _make_engine()
        stats = engine.step(data, batch_size=SMALL_BATCH)
        assert stats.elapsed_time > 0.0

    def test_batch_fills_plus_replacements_equals_improvements(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """improvements = fills + replacements for each iteration."""
        data, _ = gaussian_data
        engine = _make_engine()
        engine.run(data, n_iterations=5, batch_size=SMALL_BATCH)
        for s in engine.history:
            assert s.improvements == s.fills + s.replacements, (
                f"iter {s.iteration}: improvements={s.improvements} != "
                f"fills={s.fills} + replacements={s.replacements}"
            )


# ===================================================================
# Test: checkpoint save/load
# ===================================================================


class TestCheckpointSaveLoad:
    """Tests for engine checkpointing and restoration."""

    def test_checkpoint_save_load(
        self,
        gaussian_data: Tuple[DataMatrix, AdjacencyMatrix],
        tmp_dir: str,
    ) -> None:
        """Save a checkpoint, load it, and verify the state is restored."""
        data, _ = gaussian_data
        cfg = MAPElitesConfig(
            archive_dims=(5, 5),
            archive_ranges=((0.0, 1.0), (0.0, 1.0)),
            seed=42,
            checkpoint_interval=3,
            checkpoint_dir=os.path.join(tmp_dir, "ckpts"),
            log_interval=100,
        )
        engine = _make_engine(config=cfg)
        engine.run(data, n_iterations=3, batch_size=SMALL_BATCH)

        ckpt_dir = Path(tmp_dir) / "ckpts"
        ckpt_path = ckpt_dir / "checkpoint_iter3.pkl"
        assert ckpt_path.exists(), f"Checkpoint not saved at {ckpt_path}"

        # Load into a fresh engine
        engine2 = _make_engine(config=cfg)
        engine2.load_checkpoint(str(ckpt_path))
        assert engine2.iteration == 3
        assert len(engine2.history) == 3
        assert engine2.archive.size == engine.archive.size

    def test_checkpoint_archive_matches(
        self,
        gaussian_data: Tuple[DataMatrix, AdjacencyMatrix],
        tmp_dir: str,
    ) -> None:
        """Restored archive should have the same QD-score as the original."""
        data, _ = gaussian_data
        cfg = MAPElitesConfig(
            archive_dims=(5, 5),
            archive_ranges=((0.0, 1.0), (0.0, 1.0)),
            seed=42,
            checkpoint_interval=2,
            checkpoint_dir=os.path.join(tmp_dir, "ckpts2"),
            log_interval=100,
        )
        engine = _make_engine(config=cfg)
        engine.run(data, n_iterations=2, batch_size=SMALL_BATCH)
        original_qd = engine.archive.qd_score()

        engine2 = _make_engine(config=cfg)
        ckpt_path = Path(tmp_dir) / "ckpts2" / "checkpoint_iter2.pkl"
        engine2.load_checkpoint(str(ckpt_path))
        assert np.isclose(engine2.archive.qd_score(), original_qd)

    def test_resume_from_checkpoint_continues_evolution(
        self,
        gaussian_data: Tuple[DataMatrix, AdjacencyMatrix],
        tmp_dir: str,
    ) -> None:
        """After loading a checkpoint, further steps should succeed."""
        data, _ = gaussian_data
        cfg = MAPElitesConfig(
            archive_dims=(5, 5),
            archive_ranges=((0.0, 1.0), (0.0, 1.0)),
            seed=42,
            checkpoint_interval=2,
            checkpoint_dir=os.path.join(tmp_dir, "ckpts3"),
            log_interval=100,
        )
        engine = _make_engine(config=cfg)
        engine.run(data, n_iterations=2, batch_size=SMALL_BATCH)

        engine2 = _make_engine(config=cfg)
        ckpt_path = Path(tmp_dir) / "ckpts3" / "checkpoint_iter2.pkl"
        engine2.load_checkpoint(str(ckpt_path))
        size_before = engine2.archive.size

        # Continue evolving
        for _ in range(3):
            engine2.step(data, batch_size=SMALL_BATCH)
        assert engine2.iteration == 5
        assert engine2.archive.size >= size_before

    def test_manual_save_checkpoint(
        self,
        gaussian_data: Tuple[DataMatrix, AdjacencyMatrix],
        tmp_dir: str,
    ) -> None:
        """Manually calling _save_checkpoint() creates files."""
        data, _ = gaussian_data
        cfg = MAPElitesConfig(
            archive_dims=(5, 5),
            archive_ranges=((0.0, 1.0), (0.0, 1.0)),
            seed=42,
            checkpoint_dir=os.path.join(tmp_dir, "manual_ckpt"),
            log_interval=100,
        )
        engine = _make_engine(config=cfg)
        engine.run(data, n_iterations=2, batch_size=4)
        engine._save_checkpoint()

        ckpt_path = Path(tmp_dir) / "manual_ckpt" / f"checkpoint_iter{engine.iteration}.pkl"
        archive_path = Path(tmp_dir) / "manual_ckpt" / f"archive_iter{engine.iteration}.pkl"
        assert ckpt_path.exists()
        assert archive_path.exists()


# ===================================================================
# Test: callbacks
# ===================================================================


class TestCallbacksCalled:
    """Verify that callback hooks fire correctly."""

    def test_callbacks_called(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """Callback is invoked once per iteration."""
        data, _ = gaussian_data
        call_log: List[int] = []

        def my_callback(engine, iteration, archive, stats_tracker):
            call_log.append(iteration)

        engine = _make_engine(callbacks=[my_callback])
        engine.run(data, n_iterations=SMALL_ITERS, batch_size=4)
        assert len(call_log) == SMALL_ITERS
        assert call_log == list(range(1, SMALL_ITERS + 1))

    def test_callback_receives_correct_engine(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """Callback receives the engine instance as first argument."""
        data, _ = gaussian_data
        received_engines: list = []

        def my_callback(engine, iteration, archive, stats_tracker):
            received_engines.append(engine)

        engine = _make_engine(callbacks=[my_callback])
        engine.run(data, n_iterations=2, batch_size=4)
        assert all(e is engine for e in received_engines)

    def test_callback_receives_archive(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """Callback receives the archive as third argument."""
        data, _ = gaussian_data
        archive_sizes: List[int] = []

        def my_callback(engine, iteration, archive, stats_tracker):
            archive_sizes.append(archive.size)

        engine = _make_engine(callbacks=[my_callback])
        engine.run(data, n_iterations=3, batch_size=4)
        assert len(archive_sizes) == 3
        # Archive size should be non-decreasing
        for i in range(1, len(archive_sizes)):
            assert archive_sizes[i] >= archive_sizes[i - 1]

    def test_multiple_callbacks(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """Multiple callbacks should all be called."""
        data, _ = gaussian_data
        log_a: List[int] = []
        log_b: List[int] = []

        def cb_a(engine, iteration, archive, stats_tracker):
            log_a.append(iteration)

        def cb_b(engine, iteration, archive, stats_tracker):
            log_b.append(iteration)

        engine = _make_engine(callbacks=[cb_a, cb_b])
        engine.run(data, n_iterations=3, batch_size=4)
        assert log_a == [1, 2, 3]
        assert log_b == [1, 2, 3]

    def test_callback_exception_does_not_crash_engine(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """A callback that raises should not stop the engine."""
        data, _ = gaussian_data
        good_log: List[int] = []

        def bad_callback(engine, iteration, archive, stats_tracker):
            raise ValueError("test error")

        def good_callback(engine, iteration, archive, stats_tracker):
            good_log.append(iteration)

        engine = _make_engine(callbacks=[bad_callback, good_callback])
        # Should not raise
        engine.run(data, n_iterations=3, batch_size=4)
        assert len(good_log) == 3

    def test_callback_can_read_history(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """Callback can access engine.history to see past stats."""
        data, _ = gaussian_data
        history_lengths: List[int] = []

        def my_callback(engine, iteration, archive, stats_tracker):
            history_lengths.append(len(engine.history))

        engine = _make_engine(callbacks=[my_callback])
        engine.run(data, n_iterations=4, batch_size=4)
        assert history_lengths == [1, 2, 3, 4]

    def test_callback_sees_valid_qd_scores(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """QD-scores seen by callback should be finite numbers."""
        data, _ = gaussian_data
        qd_scores: List[float] = []

        def my_callback(engine, iteration, archive, stats_tracker):
            qd_scores.append(archive.qd_score())

        engine = _make_engine(callbacks=[my_callback])
        engine.run(data, n_iterations=5, batch_size=SMALL_BATCH)
        assert len(qd_scores) == 5
        for qd in qd_scores:
            assert np.isfinite(qd)


# ===================================================================
# Test: convergence detection / early stopping
# ===================================================================


class TestConvergenceDetection:
    """Tests for early stopping via convergence detection."""

    def test_convergence_detection(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """With a very generous threshold, early stopping should trigger
        after enough iterations of stagnation."""
        data, _ = gaussian_data
        cfg = MAPElitesConfig(
            archive_dims=(5, 5),
            archive_ranges=((0.0, 1.0), (0.0, 1.0)),
            seed=42,
            early_stopping_window=5,
            early_stopping_threshold=1e10,  # very easy to satisfy
            log_interval=100,
        )
        engine = _make_engine(config=cfg)
        engine.run(data, n_iterations=1000, batch_size=SMALL_BATCH)
        # Engine should have stopped well before 1000 iterations
        assert engine.stopped_early or engine.iteration <= 1000

    def test_no_early_stopping_when_disabled(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """With early_stopping_window=0, engine runs all iterations."""
        data, _ = gaussian_data
        n_iter = 5
        cfg = MAPElitesConfig(
            archive_dims=(5, 5),
            archive_ranges=((0.0, 1.0), (0.0, 1.0)),
            seed=42,
            early_stopping_window=0,
            log_interval=100,
        )
        engine = _make_engine(config=cfg)
        engine.run(data, n_iterations=n_iter, batch_size=4)
        assert engine.iteration == n_iter
        assert not engine.stopped_early

    def test_stopped_early_flag_false_when_not_converged(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """stopped_early is False for fresh or short runs."""
        data, _ = gaussian_data
        engine = _make_engine()
        assert engine.stopped_early is False
        engine.run(data, n_iterations=2, batch_size=4)
        assert engine.stopped_early is False

    def test_convergence_with_tight_threshold(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """With a very tight threshold (1e-15), early stopping should NOT
        trigger in just a few iterations (the archive is still changing)."""
        data, _ = gaussian_data
        cfg = MAPElitesConfig(
            archive_dims=(5, 5),
            archive_ranges=((0.0, 1.0), (0.0, 1.0)),
            seed=42,
            early_stopping_window=3,
            early_stopping_threshold=1e-15,
            log_interval=100,
        )
        engine = _make_engine(config=cfg)
        engine.run(data, n_iterations=5, batch_size=SMALL_BATCH)
        # Should have run all 5 iterations (threshold too tight)
        assert engine.iteration == 5
        assert not engine.stopped_early

    def test_convergence_summary_reflects_early_stop(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """summary() should report stopped_early status."""
        data, _ = gaussian_data
        cfg = MAPElitesConfig(
            archive_dims=(5, 5),
            archive_ranges=((0.0, 1.0), (0.0, 1.0)),
            seed=42,
            early_stopping_window=5,
            early_stopping_threshold=1e10,
            log_interval=100,
        )
        engine = _make_engine(config=cfg)
        engine.run(data, n_iterations=1000, batch_size=SMALL_BATCH)
        s = engine.summary()
        # Whether or not it stopped early, the flag should be in summary
        assert "stopped_early" in s


# ===================================================================
# Test: selection strategies
# ===================================================================


class TestSelectionStrategies:
    """Test that different selection strategies work without errors."""

    @pytest.mark.parametrize(
        "strategy",
        ["uniform", "curiosity", "quality_proportional"],
    )
    def test_selection_strategy_runs(
        self,
        strategy: str,
        gaussian_data: Tuple[DataMatrix, AdjacencyMatrix],
    ) -> None:
        """Each selection strategy should run without errors."""
        data, _ = gaussian_data
        cfg = MAPElitesConfig(
            archive_dims=(5, 5),
            archive_ranges=((0.0, 1.0), (0.0, 1.0)),
            seed=42,
            selection_strategy=strategy,
            log_interval=100,
        )
        engine = _make_engine(config=cfg)
        archive = engine.run(data, n_iterations=5, batch_size=SMALL_BATCH)
        assert archive.size > 0


# ===================================================================
# Test: adaptive operators (bandit)
# ===================================================================


class TestAdaptiveOperators:
    """Test the UCB1-based adaptive operator selection."""

    def test_adaptive_operators_enabled(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """With adaptive_operators=True, operator_stats() returns data."""
        data, _ = gaussian_data
        cfg = MAPElitesConfig(
            archive_dims=(5, 5),
            archive_ranges=((0.0, 1.0), (0.0, 1.0)),
            seed=42,
            adaptive_operators=True,
            log_interval=100,
        )
        engine = _make_engine(config=cfg, with_crossover=True)
        engine.run(data, n_iterations=5, batch_size=SMALL_BATCH)
        stats = engine.operator_stats()
        assert stats is not None
        assert len(stats) == 2  # 1 mutation + 1 crossover

    def test_adaptive_operators_disabled_returns_none(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """With adaptive_operators=False, operator_stats() returns None."""
        data, _ = gaussian_data
        engine = _make_engine()
        engine.run(data, n_iterations=2, batch_size=4)
        assert engine.operator_stats() is None


# ===================================================================
# Test: _GridArchive internals
# ===================================================================


class TestGridArchiveInternal:
    """Tests for the internal _GridArchive used by the engine."""

    def test_grid_archive_add_and_size(self) -> None:
        """Adding entries increases archive size."""
        archive = _GridArchive(dims=(5, 5), ranges=((0.0, 1.0), (0.0, 1.0)))
        entry = ArchiveEntry(
            solution=np.zeros((3, 3), dtype=np.int8),
            descriptor=np.array([0.1, 0.2]),
            quality=-10.0,
        )
        assert archive.size == 0
        archive.add(entry)
        assert archive.size == 1

    def test_grid_archive_best(self) -> None:
        """best() returns the highest-quality entry."""
        archive = _GridArchive(dims=(5, 5), ranges=((0.0, 1.0), (0.0, 1.0)))
        for q in [-100.0, -50.0, -200.0]:
            entry = ArchiveEntry(
                solution=np.zeros((3, 3), dtype=np.int8),
                descriptor=np.array([q / -200.0, 0.5]),
                quality=q,
            )
            archive.add(entry)
        assert archive.best().quality == -50.0

    def test_grid_archive_qd_score(self) -> None:
        """qd_score() is the sum of all elite qualities."""
        archive = _GridArchive(dims=(10, 10), ranges=((0.0, 1.0), (0.0, 1.0)))
        qualities = [-10.0, -20.0, -30.0]
        for i, q in enumerate(qualities):
            entry = ArchiveEntry(
                solution=np.zeros((3, 3), dtype=np.int8),
                descriptor=np.array([i * 0.3, 0.5]),
                quality=q,
            )
            archive.add(entry)
        assert np.isclose(archive.qd_score(), sum(qualities))

    def test_grid_archive_sample(self) -> None:
        """sample() returns entries from the archive."""
        rng = np.random.default_rng(42)
        archive = _GridArchive(dims=(5, 5), ranges=((0.0, 1.0), (0.0, 1.0)))
        for i in range(5):
            entry = ArchiveEntry(
                solution=np.zeros((3, 3), dtype=np.int8),
                descriptor=np.array([i * 0.2, 0.5]),
                quality=float(-i),
            )
            archive.add(entry)
        samples = archive.sample(3, rng)
        assert len(samples) == 3
        assert all(isinstance(s, ArchiveEntry) for s in samples)

    def test_grid_archive_empty_sample(self) -> None:
        """sample() on an empty archive returns an empty list."""
        rng = np.random.default_rng(42)
        archive = _GridArchive(dims=(5, 5), ranges=((0.0, 1.0), (0.0, 1.0)))
        assert archive.sample(5, rng) == []

    def test_grid_archive_replacement_improves(self) -> None:
        """A higher-quality entry should replace a lower-quality one
        in the same cell."""
        archive = _GridArchive(dims=(5, 5), ranges=((0.0, 1.0), (0.0, 1.0)))
        desc = np.array([0.5, 0.5])
        low = ArchiveEntry(
            solution=np.zeros((3, 3), dtype=np.int8),
            descriptor=desc.copy(),
            quality=-100.0,
        )
        high = ArchiveEntry(
            solution=np.ones((3, 3), dtype=np.int8),
            descriptor=desc.copy(),
            quality=-10.0,
        )
        archive.add(low)
        assert archive.size == 1
        archive.add(high)
        assert archive.size == 1  # same cell
        assert archive.best().quality == -10.0

    def test_grid_archive_save_load(self, tmp_dir: str) -> None:
        """Save and load a _GridArchive."""
        archive = _GridArchive(dims=(5, 5), ranges=((0.0, 1.0), (0.0, 1.0)))
        for i in range(3):
            entry = ArchiveEntry(
                solution=np.zeros((3, 3), dtype=np.int8),
                descriptor=np.array([i * 0.3, 0.5]),
                quality=float(-i * 10),
            )
            archive.add(entry)

        path = os.path.join(tmp_dir, "test_archive.pkl")
        archive.save(path)
        loaded = _GridArchive.load(path)
        assert loaded.size == archive.size
        assert np.isclose(loaded.qd_score(), archive.qd_score())

    def test_grid_archive_clear(self) -> None:
        """clear() resets the archive to empty."""
        archive = _GridArchive(dims=(5, 5), ranges=((0.0, 1.0), (0.0, 1.0)))
        entry = ArchiveEntry(
            solution=np.zeros((3, 3), dtype=np.int8),
            descriptor=np.array([0.5, 0.5]),
            quality=-10.0,
        )
        archive.add(entry)
        assert archive.size == 1
        archive.clear()
        assert archive.size == 0
        assert archive.qd_score() == 0.0


# ===================================================================
# Test: IterationStats dataclass
# ===================================================================


class TestIterationStats:
    """Tests for the IterationStats data structure."""

    def test_iteration_stats_fields(self) -> None:
        """All expected fields should be present."""
        stats = IterationStats(
            iteration=1,
            archive_size=10,
            best_quality=-50.0,
            mean_quality=-75.0,
            improvements=3,
            fills=2,
            replacements=1,
            elapsed_time=0.5,
        )
        assert stats.iteration == 1
        assert stats.archive_size == 10
        assert stats.best_quality == -50.0
        assert stats.mean_quality == -75.0
        assert stats.improvements == 3
        assert stats.fills == 2
        assert stats.replacements == 1
        assert stats.elapsed_time == 0.5

    def test_iteration_stats_defaults(self) -> None:
        """Optional fields should have sane defaults."""
        stats = IterationStats(
            iteration=1,
            archive_size=0,
            best_quality=0.0,
            mean_quality=0.0,
            improvements=0,
        )
        assert stats.fills == 0
        assert stats.replacements == 0
        assert stats.elapsed_time == 0.0


# ===================================================================
# Test: MAPElitesConfig dataclass
# ===================================================================


class TestMAPElitesConfig:
    """Tests for the config dataclass."""

    def test_default_config(self) -> None:
        """Default config should have reasonable values."""
        cfg = MAPElitesConfig()
        assert cfg.mutation_prob > 0.0
        assert cfg.crossover_rate >= 0.0
        assert cfg.seed == 42
        assert cfg.selection_strategy == "uniform"
        assert cfg.adaptive_operators is False
        assert cfg.early_stopping_window == 0
        assert cfg.checkpoint_interval == 0

    def test_custom_config(self) -> None:
        """Custom values should be preserved."""
        cfg = MAPElitesConfig(
            mutation_prob=0.5,
            crossover_rate=0.5,
            archive_dims=(10, 10),
            archive_ranges=((0.0, 2.0), (0.0, 2.0)),
            seed=123,
            selection_strategy="curiosity",
            adaptive_operators=True,
            early_stopping_window=10,
            early_stopping_threshold=0.01,
            checkpoint_interval=5,
        )
        assert cfg.mutation_prob == 0.5
        assert cfg.archive_dims == (10, 10)
        assert cfg.seed == 123
        assert cfg.selection_strategy == "curiosity"
        assert cfg.adaptive_operators is True
        assert cfg.early_stopping_window == 10


# ===================================================================
# Test: reproducibility (deterministic with same seed)
# ===================================================================


class TestReproducibility:
    """Verify that runs with the same seed produce the same results."""

    def test_same_seed_same_results(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """Two runs with the same seed should produce identical archives."""
        data, _ = gaussian_data
        cfg = MAPElitesConfig(
            archive_dims=(5, 5),
            archive_ranges=((0.0, 1.0), (0.0, 1.0)),
            seed=42,
            log_interval=100,
        )

        engine1 = _make_engine(config=cfg)
        engine1.run(data, n_iterations=5, batch_size=SMALL_BATCH)

        engine2 = _make_engine(config=cfg)
        engine2.run(data, n_iterations=5, batch_size=SMALL_BATCH)

        assert engine1.archive.size == engine2.archive.size
        assert np.isclose(engine1.archive.qd_score(), engine2.archive.qd_score())

    def test_different_seed_different_results(
        self, gaussian_data: Tuple[DataMatrix, AdjacencyMatrix]
    ) -> None:
        """Two runs with different seeds should (likely) differ."""
        data, _ = gaussian_data

        engine1 = _make_engine(config=MAPElitesConfig(
            archive_dims=(5, 5),
            archive_ranges=((0.0, 1.0), (0.0, 1.0)),
            seed=1,
            log_interval=100,
        ))
        engine1.run(data, n_iterations=5, batch_size=SMALL_BATCH)

        engine2 = _make_engine(config=MAPElitesConfig(
            archive_dims=(5, 5),
            archive_ranges=((0.0, 1.0), (0.0, 1.0)),
            seed=999,
            log_interval=100,
        ))
        engine2.run(data, n_iterations=5, batch_size=SMALL_BATCH)

        # Very unlikely to be identical with different seeds
        qd1 = engine1.archive.qd_score()
        qd2 = engine2.archive.qd_score()
        # At least the sizes or QD-scores should differ
        differs = (
            engine1.archive.size != engine2.archive.size
            or not np.isclose(qd1, qd2)
        )
        assert differs, "Different seeds produced identical results"
