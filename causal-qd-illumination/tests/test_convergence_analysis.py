"""Tests for causal_qd.analysis.convergence_analysis module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from causal_qd.analysis.convergence_analysis import (
    ConvergenceAnalyzer,
    ConvergenceSnapshot,
    _archive_entropy,
    _mann_kendall_trend,
)
from causal_qd.archive.archive_base import ArchiveEntry
from causal_qd.archive.grid_archive import GridArchive


# ===================================================================
# Helpers
# ===================================================================


def _make_archive(
    n_elites: int = 10,
    seed: int = 42,
    quality_range: tuple = (-100.0, -10.0),
) -> GridArchive:
    """Create a GridArchive populated with *n_elites* entries."""
    archive = GridArchive(
        dims=(10, 10),
        lower_bounds=np.array([0.0, 0.0]),
        upper_bounds=np.array([1.0, 1.0]),
    )
    rng = np.random.default_rng(seed)
    for _ in range(n_elites):
        adj = np.zeros((5, 5), dtype=np.int8)
        desc = rng.uniform(0, 1, size=2)
        entry = ArchiveEntry(
            solution=adj,
            descriptor=desc,
            quality=float(rng.uniform(*quality_range)),
        )
        archive.add(entry)
    return archive


def _make_empty_archive() -> GridArchive:
    return GridArchive(
        dims=(5, 5),
        lower_bounds=np.array([0.0, 0.0]),
        upper_bounds=np.array([1.0, 1.0]),
    )


def _make_growing_archives(n: int, seed: int = 0):
    """Return a list of *n* archives with increasing coverage/quality."""
    rng = np.random.default_rng(seed)
    archives = []
    for i in range(n):
        arch = GridArchive(
            dims=(10, 10),
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
        )
        n_elites = min(i + 1, 100)
        for _ in range(n_elites):
            adj = np.zeros((5, 5), dtype=np.int8)
            desc = rng.uniform(0, 1, size=2)
            entry = ArchiveEntry(
                solution=adj,
                descriptor=desc,
                quality=float(rng.uniform(0, 1) + i * 0.1),
            )
            arch.add(entry)
        archives.append(arch)
    return archives


# ===================================================================
# Test: snapshot recording
# ===================================================================


class TestSnapshotRecording:
    def test_record_returns_snapshot(self):
        analyzer = ConvergenceAnalyzer()
        archive = _make_archive(5)
        snap = analyzer.record(archive, generation=1)
        assert isinstance(snap, ConvergenceSnapshot)
        assert snap.generation == 1

    def test_snapshot_fields_populated(self):
        analyzer = ConvergenceAnalyzer()
        archive = _make_archive(10)
        snap = analyzer.record(archive, generation=0)
        assert snap.qd_score != 0.0
        assert snap.coverage > 0.0
        assert snap.max_fitness != 0.0
        assert snap.mean_fitness != 0.0

    def test_multiple_records(self):
        analyzer = ConvergenceAnalyzer()
        archive = _make_archive(5)
        for gen in range(5):
            analyzer.record(archive, generation=gen)
        assert len(analyzer.qd_score_history()) == 5


# ===================================================================
# Test: QD-score tracking
# ===================================================================


class TestQDScoreTracking:
    def test_qd_score_history_length(self):
        analyzer = ConvergenceAnalyzer()
        for i in range(3):
            analyzer.record(_make_archive(5, seed=i), generation=i)
        assert len(analyzer.qd_score_history()) == 3

    def test_qd_score_values_match_snapshots(self):
        analyzer = ConvergenceAnalyzer()
        archive = _make_archive(8)
        snap = analyzer.record(archive, generation=0)
        assert analyzer.qd_score_history()[0] == snap.qd_score


# ===================================================================
# Test: coverage tracking
# ===================================================================


class TestCoverageTracking:
    def test_coverage_history_length(self):
        analyzer = ConvergenceAnalyzer()
        for i in range(4):
            analyzer.record(_make_archive(3, seed=i), generation=i)
        assert len(analyzer.coverage_history()) == 4

    def test_coverage_bounded(self):
        analyzer = ConvergenceAnalyzer()
        archive = _make_archive(10)
        snap = analyzer.record(archive, generation=0)
        assert 0.0 <= snap.coverage <= 1.0


# ===================================================================
# Test: plateau convergence detection
# ===================================================================


class TestPlateauConvergence:
    def test_not_converged_with_few_snapshots(self):
        analyzer = ConvergenceAnalyzer(window_size=10)
        for i in range(5):
            analyzer.record(_make_archive(5), generation=i)
        assert analyzer.has_converged("plateau") is False

    def test_converged_with_constant_scores(self):
        analyzer = ConvergenceAnalyzer(window_size=10, significance_level=0.05)
        archive = _make_archive(10, seed=42)
        for i in range(20):
            analyzer.record(archive, generation=i)
        assert analyzer.has_converged("plateau") is True

    def test_not_converged_with_increasing_scores(self):
        analyzer = ConvergenceAnalyzer(window_size=10, significance_level=0.01)
        archives = _make_growing_archives(20)
        for i, arch in enumerate(archives):
            analyzer.record(arch, generation=i)
        assert analyzer.has_converged("plateau") is False


# ===================================================================
# Test: Mann-Kendall convergence
# ===================================================================


class TestMannKendallConvergence:
    def test_flat_series_converged(self):
        analyzer = ConvergenceAnalyzer(window_size=10, significance_level=0.1)
        archive = _make_archive(10, seed=7)
        for i in range(20):
            analyzer.record(archive, generation=i)
        assert analyzer.has_converged("mann_kendall") is True

    def test_strong_trend_not_converged(self):
        analyzer = ConvergenceAnalyzer(window_size=10, significance_level=0.01)
        archives = _make_growing_archives(20)
        for i, arch in enumerate(archives):
            analyzer.record(arch, generation=i)
        assert analyzer.has_converged("mann_kendall") is False


# ===================================================================
# Test: archive entropy
# ===================================================================


class TestArchiveEntropy:
    def test_empty_archive_zero_entropy(self):
        archive = _make_empty_archive()
        assert _archive_entropy(archive) == 0.0

    def test_single_elite_zero_entropy(self):
        archive = _make_archive(1)
        assert _archive_entropy(archive) == 0.0

    def test_multiple_elites_positive_entropy(self):
        archive = _make_archive(30, quality_range=(0.0, 10.0))
        ent = _archive_entropy(archive)
        assert ent >= 0.0

    def test_entropy_bounded(self):
        archive = _make_archive(50, quality_range=(0.0, 100.0))
        ent = _archive_entropy(archive)
        assert 0.0 <= ent <= 1.0 + 1e-6  # normalised


# ===================================================================
# Test: convergence rate
# ===================================================================


class TestConvergenceRate:
    def test_rate_zero_no_data(self):
        analyzer = ConvergenceAnalyzer()
        assert analyzer.convergence_rate() == 0.0

    def test_rate_positive_for_growing(self):
        analyzer = ConvergenceAnalyzer()
        archives = _make_growing_archives(10)
        for i, arch in enumerate(archives):
            analyzer.record(arch, generation=i)
        assert analyzer.convergence_rate() > 0.0

    def test_rate_zero_for_constant(self):
        analyzer = ConvergenceAnalyzer()
        archive = _make_archive(10, seed=0)
        for i in range(10):
            analyzer.record(archive, generation=i)
        assert abs(analyzer.convergence_rate()) < 1e-6


# ===================================================================
# Test: callback integration
# ===================================================================


class TestCallbackIntegration:
    def test_callback_callable(self):
        analyzer = ConvergenceAnalyzer()
        cb = analyzer.as_callback()
        assert callable(cb)

    def test_callback_records_snapshot(self):
        analyzer = ConvergenceAnalyzer()
        cb = analyzer.as_callback()
        archive = _make_archive(5)
        cb(None, 1, archive, None)  # engine=None, stats_tracker=None
        assert len(analyzer.qd_score_history()) == 1

    def test_callback_signature_compatible(self):
        """Callback should accept (engine, generation, archive, tracker)."""
        analyzer = ConvergenceAnalyzer()
        cb = analyzer.as_callback()
        archive = _make_archive(5)
        # Should not raise
        cb("dummy_engine", 42, archive, "dummy_tracker")
        assert analyzer.qd_score_history()[-1] is not None


# ===================================================================
# Test: edge cases
# ===================================================================


class TestEdgeCases:
    def test_empty_archive_snapshot(self):
        analyzer = ConvergenceAnalyzer()
        empty = _make_empty_archive()
        snap = analyzer.record(empty, generation=0)
        assert snap.qd_score == 0.0
        assert snap.coverage == 0.0
        assert snap.max_fitness == 0.0
        assert snap.mean_fitness == 0.0
        assert snap.archive_entropy == 0.0
        assert snap.num_improvements == 0

    def test_single_generation(self):
        analyzer = ConvergenceAnalyzer(window_size=10)
        archive = _make_archive(5)
        analyzer.record(archive, generation=0)
        assert analyzer.has_converged("plateau") is False
        assert analyzer.convergence_rate() == 0.0

    def test_constant_scores_all_methods(self):
        analyzer = ConvergenceAnalyzer(window_size=10, significance_level=0.05)
        archive = _make_archive(10, seed=99)
        for i in range(20):
            analyzer.record(archive, generation=i)
        assert analyzer.has_converged("plateau") is True
        assert analyzer.has_converged("relative") is True
        assert analyzer.has_converged("mann_kendall") is True
        assert analyzer.has_converged("geweke") is True

    def test_invalid_method_raises(self):
        analyzer = ConvergenceAnalyzer(window_size=5)
        archive = _make_archive(5)
        for i in range(10):
            analyzer.record(archive, generation=i)
        with pytest.raises(ValueError, match="Unknown convergence method"):
            analyzer.has_converged("invalid_method")


# ===================================================================
# Test: expected remaining generations
# ===================================================================


class TestExpectedRemaining:
    def test_already_reached(self):
        analyzer = ConvergenceAnalyzer()
        archive = _make_archive(100, quality_range=(0.0, 1.0))
        analyzer.record(archive, generation=0)
        # coverage is n_elites/100; with 100 unique descriptors ~= 1.0
        # but we only need coverage >= target
        remaining = analyzer.expected_remaining_generations(
            target_coverage=analyzer.coverage_history()[-1]
        )
        assert remaining == 0

    def test_no_data_returns_negative(self):
        analyzer = ConvergenceAnalyzer()
        assert analyzer.expected_remaining_generations(0.5) == -1


# ===================================================================
# Test: summary
# ===================================================================


class TestSummary:
    def test_summary_empty(self):
        analyzer = ConvergenceAnalyzer()
        s = analyzer.summary()
        assert s["n_snapshots"] == 0

    def test_summary_populated(self):
        analyzer = ConvergenceAnalyzer(window_size=5)
        archive = _make_archive(10)
        for i in range(10):
            analyzer.record(archive, generation=i)
        s = analyzer.summary()
        assert s["n_snapshots"] == 10
        assert "qd_score" in s
        assert "coverage" in s
        assert "converged_plateau" in s


# ===================================================================
# Test: Mann-Kendall helper
# ===================================================================


class TestMannKendall:
    def test_increasing(self):
        tau = _mann_kendall_trend([1, 2, 3, 4, 5])
        assert tau > 0.9

    def test_decreasing(self):
        tau = _mann_kendall_trend([5, 4, 3, 2, 1])
        assert tau < -0.9

    def test_constant(self):
        tau = _mann_kendall_trend([3, 3, 3, 3, 3])
        assert tau == 0.0

    def test_short_series(self):
        tau = _mann_kendall_trend([1, 2])
        assert tau == 0.0
