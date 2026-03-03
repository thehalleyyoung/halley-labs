"""Tests for causal_qd.metrics module.

Covers SHD, F1, QDScore, Coverage, and Diversity.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from causal_qd.archive.archive_base import ArchiveEntry
from causal_qd.archive.grid_archive import GridArchive
from causal_qd.metrics.structural import SHD, F1, skeleton_f1, edge_precision, edge_recall
from causal_qd.metrics.qd_metrics import QDScore, Coverage, Diversity, qd_score, coverage, diversity
from causal_qd.types import AdjacencyMatrix


# ===================================================================
# Helpers
# ===================================================================

def _chain_adj(n: int) -> AdjacencyMatrix:
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n - 1):
        adj[i, i + 1] = 1
    return adj


def _empty_adj(n: int) -> AdjacencyMatrix:
    return np.zeros((n, n), dtype=np.int8)


def _reversed_chain_adj(n: int) -> AdjacencyMatrix:
    """Reversed chain: (n-1)→(n-2)→…→0."""
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n - 1, 0, -1):
        adj[i, i - 1] = 1
    return adj


def _fork_adj() -> AdjacencyMatrix:
    adj = np.zeros((4, 4), dtype=np.int8)
    adj[0, 1] = 1
    adj[0, 2] = 1
    adj[0, 3] = 1
    return adj


def _make_archive(n_elites: int = 10, seed: int = 42) -> GridArchive:
    """Create a GridArchive populated with n_elites entries."""
    archive = GridArchive(
        dims=(10, 10),
        lower_bounds=np.array([0.0, 0.0]),
        upper_bounds=np.array([1.0, 1.0]),
    )
    rng = np.random.default_rng(seed)
    for i in range(n_elites):
        adj = np.zeros((5, 5), dtype=np.int8)
        if i > 0:
            adj[0, i % 5] = 1
        desc = rng.uniform(0, 1, size=2)
        entry = ArchiveEntry(
            solution=adj,
            descriptor=desc,
            quality=float(rng.uniform(-100, -10)),
        )
        archive.add(entry)
    return archive


# ===================================================================
# SHD Tests
# ===================================================================

class TestSHD:
    """Tests for causal_qd.metrics.structural.SHD."""

    def test_shd_identical_graphs(self):
        adj = _chain_adj(5)
        assert SHD.compute(adj, adj) == 0

    def test_shd_identical_empty(self):
        adj = _empty_adj(5)
        assert SHD.compute(adj, adj) == 0

    def test_shd_different_graphs(self):
        pred = _chain_adj(5)
        true = _fork_adj()
        # These have different sizes, test same-size case
        pred5 = _chain_adj(5)
        true5 = _empty_adj(5)
        shd = SHD.compute(pred5, true5)
        assert shd > 0

    def test_shd_single_edge_difference(self):
        true = _chain_adj(4)
        pred = true.copy()
        pred[2, 3] = 0  # remove one edge
        shd = SHD.compute(pred, true)
        assert shd >= 1

    def test_shd_reversed_edge(self):
        true = _chain_adj(4)
        pred = true.copy()
        pred[0, 1] = 0
        pred[1, 0] = 1  # reverse edge 0→1 to 1→0
        shd = SHD.compute(pred, true)
        assert shd >= 1

    def test_shd_symmetry(self):
        """SHD should be symmetric: SHD(A, B) == SHD(B, A)."""
        rng = np.random.default_rng(0)
        n = 5
        adj1 = np.zeros((n, n), dtype=np.int8)
        adj2 = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                adj1[i, j] = rng.choice([0, 1])
                adj2[i, j] = rng.choice([0, 1])

        assert SHD.compute(adj1, adj2) == SHD.compute(adj2, adj1)

    def test_shd_compute_simple(self):
        true = _chain_adj(4)
        pred = _empty_adj(4)
        shd_full = SHD.compute(pred, true)
        shd_simple = SHD.compute_simple(pred, true)
        assert isinstance(shd_simple, int)
        assert shd_simple >= 0

    def test_shd_added_extra_edge(self):
        true = _chain_adj(4)
        pred = true.copy()
        pred[0, 3] = 1  # extra edge
        shd = SHD.compute(pred, true)
        assert shd >= 1

    def test_shd_complete_vs_empty(self):
        n = 4
        complete = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                complete[i, j] = 1
        empty = _empty_adj(n)
        shd = SHD.compute(complete, empty)
        n_edges = n * (n - 1) // 2
        assert shd == n_edges

    def test_shd_is_nonnegative(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            n = 5
            a = np.zeros((n, n), dtype=np.int8)
            b = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i + 1, n):
                    a[i, j] = rng.choice([0, 1])
                    b[i, j] = rng.choice([0, 1])
            assert SHD.compute(a, b) >= 0


# ===================================================================
# F1 Tests
# ===================================================================

class TestF1:
    """Tests for causal_qd.metrics.structural.F1."""

    def test_f1_perfect_recovery(self):
        adj = _chain_adj(5)
        f1 = F1()
        score = f1.compute(adj, adj)
        assert score == pytest.approx(1.0)
        assert f1.precision() == pytest.approx(1.0)
        assert f1.recall() == pytest.approx(1.0)

    def test_f1_no_recovery(self):
        pred = _empty_adj(5)
        true = _chain_adj(5)
        f1 = F1()
        score = f1.compute(pred, true)
        assert score == pytest.approx(0.0)

    def test_f1_partial_recovery(self):
        true = _chain_adj(5)
        pred = true.copy()
        pred[3, 4] = 0  # miss one edge
        f1 = F1()
        score = f1.compute(pred, true)
        assert 0.0 < score < 1.0
        assert f1.recall() < 1.0
        assert f1.precision() == pytest.approx(1.0)

    def test_f1_extra_edges(self):
        true = _chain_adj(4)
        pred = true.copy()
        pred[0, 3] = 1  # extra edge
        f1 = F1()
        score = f1.compute(pred, true)
        assert 0.0 < score < 1.0
        assert f1.precision() < 1.0
        assert f1.recall() == pytest.approx(1.0)

    def test_f1_symmetry_of_score(self):
        """F1 score of pred vs. true is NOT generally symmetric."""
        true = _chain_adj(4)
        pred = _fork_adj()
        f1a = F1()
        f1b = F1()
        sa = f1a.compute(pred, true)
        sb = f1b.compute(true, pred)
        # Not necessarily equal, just both valid
        assert 0.0 <= sa <= 1.0
        assert 0.0 <= sb <= 1.0

    def test_f1_values_in_range(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            n = 5
            pred = np.zeros((n, n), dtype=np.int8)
            true = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i + 1, n):
                    pred[i, j] = rng.choice([0, 1])
                    true[i, j] = rng.choice([0, 1])
            f1 = F1()
            score = f1.compute(pred, true)
            assert 0.0 <= score <= 1.0

    def test_edge_precision_and_recall(self):
        true = _chain_adj(5)
        pred = true.copy()
        pred[3, 4] = 0  # miss one
        pred[0, 4] = 1  # add one

        prec = edge_precision(pred, true)
        rec = edge_recall(pred, true)
        assert 0.0 <= prec <= 1.0
        assert 0.0 <= rec <= 1.0

    def test_skeleton_f1(self):
        true = _chain_adj(5)
        pred = _reversed_chain_adj(5)
        score = skeleton_f1(pred, true)
        # Same skeleton, different orientation
        assert score >= 0.5


# ===================================================================
# QDScore Tests
# ===================================================================

class TestQDScore:
    """Tests for causal_qd.metrics.qd_metrics.QDScore."""

    def test_qd_score_computation(self):
        archive = _make_archive(10)
        score = QDScore.compute(archive)
        assert isinstance(score, float)
        # QD score is sum of qualities; qualities are negative here
        assert score != 0.0

    def test_qd_score_empty_archive(self):
        archive = GridArchive(
            dims=(5, 5),
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
        )
        score = QDScore.compute(archive)
        assert score == pytest.approx(0.0)

    def test_qd_score_increases_with_more_elites(self):
        """Adding higher-quality elites should increase QD score."""
        archive1 = GridArchive(
            dims=(10, 10),
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
        )
        archive2 = GridArchive(
            dims=(10, 10),
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
        )

        rng = np.random.default_rng(42)
        adj = np.zeros((5, 5), dtype=np.int8)

        # Archive 1: 5 elites
        for i in range(5):
            desc = rng.uniform(0, 1, size=2)
            archive1.add(ArchiveEntry(solution=adj, descriptor=desc, quality=10.0))

        # Archive 2: 10 elites with same quality
        rng2 = np.random.default_rng(42)
        for i in range(10):
            desc = rng2.uniform(0, 1, size=2)
            archive2.add(ArchiveEntry(solution=adj, descriptor=desc, quality=10.0))

        s1 = QDScore.compute(archive1)
        s2 = QDScore.compute(archive2)
        assert s2 >= s1

    def test_qd_score_matches_function(self):
        archive = _make_archive(10)
        assert QDScore.compute(archive) == pytest.approx(qd_score(archive))


# ===================================================================
# Coverage Tests
# ===================================================================

class TestCoverage:
    """Tests for causal_qd.metrics.qd_metrics.Coverage."""

    def test_coverage_computation(self):
        archive = _make_archive(10)
        cov = Coverage.compute(archive)
        assert isinstance(cov, float)
        assert 0.0 <= cov <= 1.0

    def test_coverage_empty_archive(self):
        archive = GridArchive(
            dims=(5, 5),
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
        )
        cov = Coverage.compute(archive)
        assert cov == pytest.approx(0.0)

    def test_coverage_full_archive(self):
        """Fill every cell → coverage = 1.0."""
        dims = (3, 3)
        archive = GridArchive(
            dims=dims,
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
        )
        adj = np.zeros((5, 5), dtype=np.int8)

        for i in range(dims[0]):
            for j in range(dims[1]):
                d0 = (i + 0.5) / dims[0]
                d1 = (j + 0.5) / dims[1]
                archive.add(ArchiveEntry(
                    solution=adj,
                    descriptor=np.array([d0, d1]),
                    quality=1.0,
                ))

        cov = Coverage.compute(archive)
        assert cov == pytest.approx(1.0)

    def test_coverage_matches_function(self):
        archive = _make_archive(10)
        assert Coverage.compute(archive) == pytest.approx(coverage(archive))

    def test_coverage_increases_with_elites(self):
        archive = GridArchive(
            dims=(10, 10),
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
        )
        adj = np.zeros((5, 5), dtype=np.int8)
        rng = np.random.default_rng(0)

        coverages = []
        for i in range(20):
            desc = rng.uniform(0, 1, size=2)
            archive.add(ArchiveEntry(solution=adj, descriptor=desc, quality=1.0))
            coverages.append(Coverage.compute(archive))

        # Coverage should be non-decreasing
        for i in range(1, len(coverages)):
            assert coverages[i] >= coverages[i - 1]


# ===================================================================
# Diversity Tests
# ===================================================================

class TestDiversity:
    """Tests for causal_qd.metrics.qd_metrics.Diversity."""

    def test_diversity_metric(self):
        archive = _make_archive(10)
        div = Diversity.compute(archive)
        assert isinstance(div, float)
        assert div >= 0.0

    def test_diversity_empty_archive(self):
        archive = GridArchive(
            dims=(5, 5),
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
        )
        div = Diversity.compute(archive)
        assert div >= 0.0  # should handle empty archive gracefully

    def test_diversity_single_elite(self):
        archive = GridArchive(
            dims=(5, 5),
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
        )
        adj = np.zeros((5, 5), dtype=np.int8)
        archive.add(ArchiveEntry(
            solution=adj, descriptor=np.array([0.5, 0.5]), quality=1.0,
        ))
        div = Diversity.compute(archive)
        assert div >= 0.0

    def test_diversity_increases_with_spread(self):
        """Widely spread descriptors should yield higher diversity than clustered."""
        adj = np.zeros((5, 5), dtype=np.int8)

        # Clustered archive
        archive_c = GridArchive(
            dims=(10, 10),
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
        )
        rng = np.random.default_rng(0)
        for _ in range(20):
            desc = rng.uniform(0.4, 0.6, size=2)
            archive_c.add(ArchiveEntry(solution=adj, descriptor=desc, quality=1.0))

        # Spread archive
        archive_s = GridArchive(
            dims=(10, 10),
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
        )
        rng2 = np.random.default_rng(1)
        for _ in range(20):
            desc = rng2.uniform(0.0, 1.0, size=2)
            archive_s.add(ArchiveEntry(solution=adj, descriptor=desc, quality=1.0))

        div_c = Diversity.compute(archive_c)
        div_s = Diversity.compute(archive_s)
        assert div_s >= div_c

    def test_diversity_matches_function(self):
        archive = _make_archive(10)
        assert Diversity.compute(archive) == pytest.approx(diversity(archive))
