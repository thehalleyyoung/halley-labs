"""Tests for archive data structures: GridArchive, CVTArchive, and ArchiveEntry.

Exercises insertion, replacement, coverage, QD-score, serialisation,
stats tracking, sampling, and Voronoi cell assignment.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pytest

from causal_qd.archive.archive_base import ArchiveEntry
from causal_qd.archive.grid_archive import GridArchive
from causal_qd.archive.cvt_archive import CVTArchive
from causal_qd.archive.stats import ArchiveStats, ArchiveStatsTracker


# ===================================================================
# Helpers
# ===================================================================

def _make_entry(
    n_nodes: int = 5,
    descriptor: np.ndarray | None = None,
    quality: float = 0.0,
    edge_list: list | None = None,
    metadata: dict | None = None,
) -> ArchiveEntry:
    """Create an ArchiveEntry with a valid adjacency matrix."""
    adj = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    if edge_list is not None:
        for i, j in edge_list:
            adj[i, j] = 1
    if descriptor is None:
        descriptor = np.array([0.5, 0.5])
    return ArchiveEntry(
        solution=adj,
        descriptor=descriptor,
        quality=quality,
        metadata=metadata or {},
    )


def _make_dag_entry(
    rng: np.random.Generator,
    n_nodes: int = 5,
    edge_prob: float = 0.3,
    quality: float | None = None,
    descriptor: np.ndarray | None = None,
) -> ArchiveEntry:
    """Create an entry whose solution is a random DAG (upper-triangular)."""
    adj = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_prob:
                adj[i, j] = 1
    if descriptor is None:
        descriptor = rng.uniform(0.0, 1.0, size=2)
    if quality is None:
        quality = float(rng.uniform(-100, 0))
    return ArchiveEntry(
        solution=adj,
        descriptor=descriptor,
        quality=quality,
    )


def _is_dag(adj: np.ndarray) -> bool:
    """Return True if adj represents a DAG (no cycles)."""
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    in_stack = np.zeros(n, dtype=bool)

    def _dfs(node: int) -> bool:
        visited[node] = True
        in_stack[node] = True
        for child in range(n):
            if adj[node, child]:
                if not visited[child]:
                    if not _dfs(child):
                        return False
                elif in_stack[child]:
                    return False
        in_stack[node] = False
        return True

    for node in range(n):
        if not visited[node]:
            if not _dfs(node):
                return False
    return True


# ===================================================================
# GridArchive — add and retrieve
# ===================================================================


class TestGridArchiveAddAndRetrieve:
    """Test basic insertion and retrieval for GridArchive."""

    def test_grid_archive_add_and_retrieve(self, empty_archive: GridArchive, rng):
        """Adding an entry should make it retrievable at the correct cell."""
        entry = _make_entry(
            descriptor=np.array([0.1, 0.1]),
            quality=-50.0,
            edge_list=[(0, 1)],
        )
        added = empty_archive.add(entry)
        assert added is True
        assert len(empty_archive) == 1

        # Retrieve by computing the expected cell index
        idx = empty_archive._descriptor_to_index(entry.descriptor)
        retrieved = empty_archive.get(idx)
        assert retrieved is not None
        assert retrieved.quality == -50.0
        np.testing.assert_array_equal(retrieved.solution, entry.solution)
        np.testing.assert_array_almost_equal(retrieved.descriptor, entry.descriptor)

    def test_add_returns_true_for_new_cell(self, empty_archive):
        entry = _make_entry(descriptor=np.array([0.5, 0.5]), quality=-20.0)
        assert empty_archive.add(entry) is True

    def test_add_to_multiple_cells(self, empty_archive):
        """Entries with different descriptors land in different cells."""
        e1 = _make_entry(descriptor=np.array([0.1, 0.1]), quality=-10.0)
        e2 = _make_entry(descriptor=np.array([0.9, 0.9]), quality=-20.0)
        empty_archive.add(e1)
        empty_archive.add(e2)
        assert len(empty_archive) == 2

        idx1 = empty_archive._descriptor_to_index(e1.descriptor)
        idx2 = empty_archive._descriptor_to_index(e2.descriptor)
        assert idx1 != idx2
        assert empty_archive.get(idx1).quality == -10.0
        assert empty_archive.get(idx2).quality == -20.0

    def test_get_nonexistent_cell_returns_none(self, empty_archive):
        assert empty_archive.get((99, 99)) is None

    def test_contains(self, empty_archive):
        entry = _make_entry(descriptor=np.array([0.3, 0.3]), quality=-5.0)
        empty_archive.add(entry)
        idx = empty_archive._descriptor_to_index(entry.descriptor)
        assert idx in empty_archive
        assert (99, 99) not in empty_archive

    def test_iter_yields_all_elites(self, sample_archive):
        elites_list = list(sample_archive)
        assert len(elites_list) == len(sample_archive)
        for e in elites_list:
            assert isinstance(e, ArchiveEntry)

    def test_len_matches_occupied(self, sample_archive):
        assert len(sample_archive) == len(sample_archive.elites())

    def test_timestamp_monotonically_increases(self, empty_archive):
        """Each inserted entry should get a strictly increasing timestamp."""
        entries = []
        for i in range(5):
            e = _make_entry(
                descriptor=np.array([0.1 * (i + 1), 0.1 * (i + 1)]),
                quality=float(-50 + i),
            )
            empty_archive.add(e)
            entries.append(e)

        for i in range(1, len(entries)):
            assert entries[i].timestamp > entries[i - 1].timestamp


# ===================================================================
# GridArchive — replacement semantics
# ===================================================================


class TestGridArchiveReplacement:
    """Test that higher-quality solutions replace lower-quality ones."""

    def test_grid_archive_replaces_lower_quality(self, empty_archive):
        """A higher-quality entry at the same cell should replace the old one."""
        desc = np.array([0.5, 0.5])
        low = _make_entry(descriptor=desc, quality=-80.0, edge_list=[(0, 1)])
        high = _make_entry(descriptor=desc, quality=-20.0, edge_list=[(0, 2)])

        assert empty_archive.add(low) is True
        assert empty_archive.add(high) is True
        assert len(empty_archive) == 1

        idx = empty_archive._descriptor_to_index(desc)
        stored = empty_archive.get(idx)
        assert stored.quality == -20.0
        assert stored.solution[0, 2] == 1

    def test_grid_archive_keeps_better_quality(self, empty_archive):
        """A lower-quality entry must NOT replace a higher-quality one."""
        desc = np.array([0.5, 0.5])
        high = _make_entry(descriptor=desc, quality=-10.0, edge_list=[(0, 1)])
        low = _make_entry(descriptor=desc, quality=-90.0, edge_list=[(0, 2)])

        assert empty_archive.add(high) is True
        assert empty_archive.add(low) is False
        assert len(empty_archive) == 1

        idx = empty_archive._descriptor_to_index(desc)
        stored = empty_archive.get(idx)
        assert stored.quality == -10.0
        assert stored.solution[0, 1] == 1
        assert stored.solution[0, 2] == 0

    def test_equal_quality_does_not_replace(self, empty_archive):
        """Same-quality entry should NOT replace the incumbent."""
        desc = np.array([0.5, 0.5])
        e1 = _make_entry(descriptor=desc, quality=-50.0, edge_list=[(0, 1)])
        e2 = _make_entry(descriptor=desc, quality=-50.0, edge_list=[(1, 2)])

        empty_archive.add(e1)
        assert empty_archive.add(e2) is False

        idx = empty_archive._descriptor_to_index(desc)
        stored = empty_archive.get(idx)
        assert stored.solution[0, 1] == 1

    def test_replacement_records_improvement(self, empty_archive):
        """Replacement should be reflected in the improvement_history."""
        desc = np.array([0.5, 0.5])
        low = _make_entry(descriptor=desc, quality=-80.0)
        high = _make_entry(descriptor=desc, quality=-30.0)

        empty_archive.add(low)
        empty_archive.add(high)

        hist = empty_archive.improvement_history
        assert len(hist) == 1
        ts, delta = hist[0]
        assert delta == pytest.approx(50.0)

    def test_multiple_replacements_tracked(self, empty_archive):
        """Multiple quality improvements in the same cell should all be tracked."""
        desc = np.array([0.5, 0.5])
        qualities = [-100.0, -70.0, -40.0, -10.0]
        for q in qualities:
            empty_archive.add(_make_entry(descriptor=desc, quality=q))

        assert empty_archive.total_replacements == 3
        assert empty_archive.total_fills == 1
        assert len(empty_archive.improvement_history) == 3


# ===================================================================
# GridArchive — coverage
# ===================================================================


class TestGridArchiveCoverage:
    """Test coverage computation."""

    def test_grid_archive_coverage(self, empty_archive):
        """Coverage should be occupied / total_cells."""
        assert empty_archive.coverage() == 0.0

        # Fill 5 distinct cells in a 5×5 grid
        for i in range(5):
            desc = np.array([0.1 + i * 0.2, 0.5])
            empty_archive.add(_make_entry(descriptor=desc, quality=-10.0))

        n_occupied = len(empty_archive)
        expected_coverage = n_occupied / empty_archive.total_cells
        assert empty_archive.coverage() == pytest.approx(expected_coverage)

    def test_coverage_never_exceeds_one(self, empty_archive, rng):
        """Even with many insertions, coverage ≤ 1.0."""
        for _ in range(200):
            desc = rng.uniform(0.0, 1.0, size=2)
            empty_archive.add(_make_entry(descriptor=desc, quality=rng.uniform(-100, 0)))
        assert empty_archive.coverage() <= 1.0

    def test_coverage_with_sample_archive(self, sample_archive):
        """Pre-populated archive should have positive coverage."""
        cov = sample_archive.coverage()
        assert cov > 0.0
        assert cov == len(sample_archive) / sample_archive.total_cells

    def test_coverage_after_clear(self, sample_archive):
        sample_archive.clear()
        assert sample_archive.coverage() == 0.0
        assert len(sample_archive) == 0

    def test_total_cells_correct(self, empty_archive):
        """A 5×5 grid should have 25 total cells."""
        assert empty_archive.total_cells == 25
        assert empty_archive.dims == (5, 5)
        assert empty_archive.descriptor_dim == 2


# ===================================================================
# GridArchive — QD score
# ===================================================================


class TestGridArchiveQDScore:
    """Test QD-score computation."""

    def test_grid_archive_qd_score(self, empty_archive):
        """QD score is the sum of qualities of all stored elites."""
        assert empty_archive.qd_score() == 0.0

        qualities = [-10.0, -20.0, -30.0]
        for i, q in enumerate(qualities):
            desc = np.array([0.1 + i * 0.3, 0.5])
            empty_archive.add(_make_entry(descriptor=desc, quality=q))

        assert empty_archive.qd_score() == pytest.approx(sum(qualities))

    def test_qd_score_after_replacement(self, empty_archive):
        """QD score should reflect the current (replaced) elites, not history."""
        desc = np.array([0.5, 0.5])
        empty_archive.add(_make_entry(descriptor=desc, quality=-80.0))
        assert empty_archive.qd_score() == pytest.approx(-80.0)

        empty_archive.add(_make_entry(descriptor=desc, quality=-20.0))
        assert empty_archive.qd_score() == pytest.approx(-20.0)

    def test_qd_score_sample_archive(self, sample_archive):
        """QD score of pre-populated archive equals sum of elite qualities."""
        expected = sum(e.quality for e in sample_archive.elites())
        assert sample_archive.qd_score() == pytest.approx(expected)

    def test_mean_quality(self, empty_archive):
        """Mean quality should equal qd_score / num_elites."""
        assert empty_archive.mean_quality() == 0.0

        for i in range(4):
            desc = np.array([0.1 + i * 0.25, 0.5])
            empty_archive.add(_make_entry(descriptor=desc, quality=float(-(i + 1) * 10)))

        expected_mean = empty_archive.qd_score() / len(empty_archive)
        assert empty_archive.mean_quality() == pytest.approx(expected_mean)

    def test_best_raises_on_empty(self, empty_archive):
        with pytest.raises(ValueError, match="empty"):
            empty_archive.best()

    def test_best_returns_highest_quality(self, sample_archive):
        best = sample_archive.best()
        all_qualities = [e.quality for e in sample_archive.elites()]
        assert best.quality == max(all_qualities)


# ===================================================================
# CVTArchive — add and retrieve
# ===================================================================


class TestCVTArchiveAddAndRetrieve:
    """Test CVTArchive insertion and retrieval."""

    def test_cvt_archive_add_and_retrieve(self, rng):
        """An entry added to CVTArchive should be retrievable."""
        archive = CVTArchive(
            n_cells=10,
            descriptor_dim=2,
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
            rng=rng,
            n_samples=500,
        )
        entry = _make_entry(
            descriptor=np.array([0.5, 0.5]),
            quality=-25.0,
            edge_list=[(0, 1), (1, 2)],
        )
        assert archive.add(entry) is True
        assert len(archive) == 1

        idx = archive._descriptor_to_index(entry.descriptor)
        retrieved = archive.get(idx)
        assert retrieved is not None
        assert retrieved.quality == -25.0
        np.testing.assert_array_equal(retrieved.solution, entry.solution)

    def test_cvt_multiple_cells(self, rng):
        """Entries with very different descriptors should land in different cells."""
        archive = CVTArchive(
            n_cells=20,
            descriptor_dim=2,
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
            rng=rng,
            n_samples=500,
        )
        e1 = _make_entry(descriptor=np.array([0.05, 0.05]), quality=-10.0)
        e2 = _make_entry(descriptor=np.array([0.95, 0.95]), quality=-20.0)

        archive.add(e1)
        archive.add(e2)

        idx1 = archive._descriptor_to_index(e1.descriptor)
        idx2 = archive._descriptor_to_index(e2.descriptor)
        assert idx1 != idx2
        assert len(archive) == 2

    def test_cvt_replacement(self, rng):
        """Higher quality should replace lower quality in the same Voronoi cell."""
        archive = CVTArchive(
            n_cells=5,
            descriptor_dim=2,
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
            rng=rng,
            n_samples=200,
        )
        desc = np.array([0.5, 0.5])
        low = _make_entry(descriptor=desc, quality=-80.0)
        high = _make_entry(descriptor=desc, quality=-10.0)

        archive.add(low)
        archive.add(high)
        assert len(archive) == 1

        idx = archive._descriptor_to_index(desc)
        assert archive.get(idx).quality == -10.0

    def test_cvt_coverage(self, rng):
        """CVTArchive coverage should match n_occupied / n_cells."""
        archive = CVTArchive(
            n_cells=10,
            descriptor_dim=2,
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
            rng=rng,
            n_samples=500,
        )
        for _ in range(50):
            desc = rng.uniform(0.0, 1.0, size=2)
            archive.add(_make_entry(descriptor=desc, quality=rng.uniform(-100, 0)))

        cov = archive.coverage()
        assert cov == len(archive) / archive.n_cells
        assert 0.0 < cov <= 1.0

    def test_cvt_qd_score(self, rng):
        archive = CVTArchive(
            n_cells=10,
            descriptor_dim=2,
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
            rng=rng,
            n_samples=500,
        )
        qualities = []
        for i in range(8):
            q = float(-(i + 1) * 5)
            qualities.append(q)
            desc = rng.uniform(0.0, 1.0, size=2)
            archive.add(_make_entry(descriptor=desc, quality=q))

        expected_qd = sum(e.quality for e in archive.elites())
        assert archive.qd_score() == pytest.approx(expected_qd)

    def test_cvt_best(self, rng):
        archive = CVTArchive(
            n_cells=10,
            descriptor_dim=2,
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
            rng=rng,
            n_samples=500,
        )
        for i in range(10):
            desc = rng.uniform(0.0, 1.0, size=2)
            archive.add(_make_entry(descriptor=desc, quality=float(-(i + 1) * 10)))

        best = archive.best()
        all_qualities = [e.quality for e in archive.elites()]
        assert best.quality == max(all_qualities)

    def test_cvt_best_empty_raises(self, rng):
        archive = CVTArchive(
            n_cells=5,
            descriptor_dim=2,
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
            rng=rng,
            n_samples=100,
        )
        with pytest.raises(ValueError):
            archive.best()

    def test_cvt_clear(self, rng):
        archive = CVTArchive(
            n_cells=5,
            descriptor_dim=2,
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
            rng=rng,
            n_samples=100,
        )
        archive.add(_make_entry(descriptor=np.array([0.5, 0.5]), quality=-10.0))
        assert len(archive) == 1
        archive.clear()
        assert len(archive) == 0
        assert archive.coverage() == 0.0


# ===================================================================
# CVTArchive — Voronoi assignment
# ===================================================================


class TestCVTArchiveVoronoiAssignment:
    """Test that descriptors map to the nearest centroid."""

    def test_cvt_archive_voronoi_assignment(self, rng):
        """A descriptor should map to its nearest centroid."""
        archive = CVTArchive(
            n_cells=20,
            descriptor_dim=2,
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
            rng=rng,
            n_samples=1000,
        )
        centroids = archive.centroids

        # Pick a descriptor very close to one specific centroid
        target_idx = 7
        target_centroid = centroids[target_idx]
        near_desc = target_centroid + rng.uniform(-1e-6, 1e-6, size=2)
        near_desc = np.clip(near_desc, 0.0, 1.0)

        cell_idx = archive._descriptor_to_index(near_desc)
        assert cell_idx == (target_idx,)

    def test_all_centroids_reachable(self, rng):
        """Each centroid should be the nearest to itself."""
        archive = CVTArchive(
            n_cells=15,
            descriptor_dim=2,
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
            rng=rng,
            n_samples=1000,
        )
        centroids = archive.centroids
        for i, c in enumerate(centroids):
            idx = archive._descriptor_to_index(c)
            assert idx == (i,), f"Centroid {i} maps to cell {idx} instead of ({i},)"

    def test_voronoi_consistency_with_brute_force(self, rng):
        """KD-tree assignment should match brute-force nearest-centroid."""
        archive = CVTArchive(
            n_cells=10,
            descriptor_dim=2,
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
            rng=rng,
            n_samples=500,
            use_kd_tree=True,
        )
        centroids = archive.centroids

        for _ in range(50):
            desc = rng.uniform(0.0, 1.0, size=2)
            kd_idx = archive._descriptor_to_index(desc)
            # Brute force
            dists = np.linalg.norm(centroids - desc, axis=1)
            brute_idx = (int(np.argmin(dists)),)
            assert kd_idx == brute_idx

    def test_centroids_within_bounds(self, rng):
        """All centroids should lie within [lower_bounds, upper_bounds]."""
        lb = np.array([-1.0, -1.0])
        ub = np.array([2.0, 3.0])
        archive = CVTArchive(
            n_cells=10,
            descriptor_dim=2,
            lower_bounds=lb,
            upper_bounds=ub,
            rng=rng,
            n_samples=1000,
        )
        centroids = archive.centroids
        assert np.all(centroids >= lb - 1e-10)
        assert np.all(centroids <= ub + 1e-10)

    def test_centroids_property_returns_copy(self, rng):
        """Modifying the returned centroids should not affect the archive."""
        archive = CVTArchive(
            n_cells=5,
            descriptor_dim=2,
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
            rng=rng,
            n_samples=200,
        )
        c1 = archive.centroids
        c1[:] = 999.0
        c2 = archive.centroids
        assert not np.allclose(c2, 999.0)


# ===================================================================
# Serialisation round-trip
# ===================================================================


class TestArchiveSerializationRoundtrip:
    """Test save/load for both GridArchive and CVTArchive."""

    def test_archive_serialization_roundtrip(self, sample_archive, tmp_dir):
        """GridArchive should survive a save→load cycle unchanged."""
        path = os.path.join(tmp_dir, "grid_archive")
        sample_archive.save(path)

        # Verify files exist
        assert Path(path).with_suffix(".pkl").exists()
        assert Path(f"{path}.meta.json").exists()

        loaded = GridArchive.load(path)
        assert len(loaded) == len(sample_archive)
        assert loaded.dims == sample_archive.dims
        assert loaded.total_cells == sample_archive.total_cells
        assert loaded.qd_score() == pytest.approx(sample_archive.qd_score())
        assert loaded.coverage() == pytest.approx(sample_archive.coverage())

        # Check all elites
        orig_elites = sorted(sample_archive.elites(), key=lambda e: e.quality)
        load_elites = sorted(loaded.elites(), key=lambda e: e.quality)
        for orig, load in zip(orig_elites, load_elites):
            assert orig.quality == pytest.approx(load.quality)
            np.testing.assert_array_equal(orig.solution, load.solution)
            np.testing.assert_array_almost_equal(orig.descriptor, load.descriptor)

    def test_grid_archive_metadata_json(self, sample_archive, tmp_dir):
        """The .meta.json file should contain correct summary info."""
        path = os.path.join(tmp_dir, "grid_meta")
        sample_archive.save(path)

        with open(f"{path}.meta.json", "r") as f:
            meta = json.load(f)

        assert meta["dims"] == list(sample_archive.dims)
        assert meta["total_cells"] == sample_archive.total_cells
        assert meta["occupied_cells"] == len(sample_archive)
        assert meta["coverage"] == pytest.approx(sample_archive.coverage())
        assert meta["qd_score"] == pytest.approx(sample_archive.qd_score())

    def test_cvt_archive_serialization_roundtrip(self, rng, tmp_dir):
        """CVTArchive save should persist data and metadata correctly."""
        archive = CVTArchive(
            n_cells=8,
            descriptor_dim=2,
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
            rng=rng,
            n_samples=500,
        )
        for _ in range(15):
            desc = rng.uniform(0.0, 1.0, size=2)
            archive.add(_make_entry(descriptor=desc, quality=rng.uniform(-100, 0)))

        path = os.path.join(tmp_dir, "cvt_archive")
        archive.save(path)

        # Verify files exist
        pkl_path = Path(path).with_suffix(".pkl")
        meta_path = Path(f"{path}.meta.json")
        assert pkl_path.exists()
        assert meta_path.exists()

        # Verify metadata JSON is correct
        with open(meta_path, "r") as f:
            meta = json.load(f)
        assert meta["n_cells"] == 8
        assert meta["descriptor_dim"] == 2
        assert meta["occupied_cells"] == len(archive)
        assert meta["qd_score"] == pytest.approx(archive.qd_score())
        assert meta["coverage"] == pytest.approx(archive.coverage())

        # Verify pickle contains centroids and cells
        with open(pkl_path, "rb") as f:
            state = pickle.load(f)
        np.testing.assert_array_almost_equal(state["centroids"], archive.centroids)
        assert len(state["cells"]) == len(archive)
        saved_qualities = sorted(e.quality for e in state["cells"].values())
        orig_qualities = sorted(e.quality for e in archive.elites())
        for sq, oq in zip(saved_qualities, orig_qualities):
            assert sq == pytest.approx(oq)

    def test_empty_archive_serialization(self, empty_archive, tmp_dir):
        """An empty archive should serialise and deserialise correctly."""
        path = os.path.join(tmp_dir, "empty_archive")
        empty_archive.save(path)

        loaded = GridArchive.load(path)
        assert len(loaded) == 0
        assert loaded.coverage() == 0.0
        assert loaded.qd_score() == 0.0

    def test_serialization_preserves_stats(self, empty_archive, tmp_dir):
        """Internal counters should survive a save→load cycle."""
        desc = np.array([0.5, 0.5])
        empty_archive.add(_make_entry(descriptor=desc, quality=-80.0))
        empty_archive.add(_make_entry(descriptor=desc, quality=-30.0))

        path = os.path.join(tmp_dir, "stats_archive")
        empty_archive.save(path)
        loaded = GridArchive.load(path)

        assert loaded.total_replacements == empty_archive.total_replacements
        assert loaded.total_fills == empty_archive.total_fills
        assert len(loaded.improvement_history) == len(empty_archive.improvement_history)


# ===================================================================
# Archive stats tracking
# ===================================================================


class TestArchiveStatsTracking:
    """Test ArchiveStats snapshot and ArchiveStatsTracker longitudinal tracking."""

    def test_archive_stats_tracking(self, sample_archive):
        """ArchiveStats.from_archive should capture current state."""
        stats = ArchiveStats.from_archive(sample_archive)

        assert stats.num_elites == len(sample_archive)
        assert stats.coverage == pytest.approx(sample_archive.coverage())
        assert stats.qd_score == pytest.approx(sample_archive.qd_score())
        assert stats.best_quality == sample_archive.best().quality
        assert stats.mean_quality == pytest.approx(sample_archive.mean_quality())
        assert stats.diversity >= 0.0

    def test_stats_empty_archive(self, empty_archive):
        stats = ArchiveStats.from_archive(empty_archive)
        assert stats.num_elites == 0
        assert stats.coverage == 0.0
        assert stats.qd_score == 0.0
        assert stats.best_quality == float("-inf")
        assert stats.diversity == 0.0

    def test_tracker_records_generations(self, empty_archive, rng):
        """ArchiveStatsTracker should record per-generation metrics."""
        tracker = ArchiveStatsTracker(window_size=5)

        for gen in range(10):
            desc = rng.uniform(0.0, 1.0, size=2)
            entry = _make_entry(descriptor=desc, quality=rng.uniform(-100, 0))
            added = empty_archive.add(entry)
            improvements = 1 if added else 0
            tracker.record(generation=gen, archive=empty_archive, improvements=improvements)

        assert len(tracker.records) == 10
        assert len(tracker.coverage_history) == 10
        assert len(tracker.qd_score_history) == 10

    def test_tracker_improvement_rate(self, empty_archive, rng):
        tracker = ArchiveStatsTracker(window_size=3)
        for gen in range(5):
            desc = rng.uniform(0.0, 1.0, size=2)
            empty_archive.add(_make_entry(descriptor=desc, quality=rng.uniform(-100, 0)))
            tracker.record(generation=gen, archive=empty_archive, improvements=gen + 1)

        rate = tracker.improvement_rate(window=3)
        recent_improvements = [r.improvements for r in tracker.records[-3:]]
        assert rate == pytest.approx(np.mean(recent_improvements))

    def test_tracker_plateau_length(self):
        tracker = ArchiveStatsTracker()
        archive = GridArchive(
            dims=(5, 5),
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
        )
        archive.add(_make_entry(descriptor=np.array([0.5, 0.5]), quality=-50.0))

        # 3 generations with improvements=0
        for gen in range(3):
            tracker.record(generation=gen, archive=archive, improvements=0)

        assert tracker.plateau_length() == 3

        # Add one with improvements
        tracker.record(generation=3, archive=archive, improvements=2)
        assert tracker.plateau_length() == 0

    def test_tracker_best_generation(self, rng):
        tracker = ArchiveStatsTracker()
        archive = GridArchive(
            dims=(10, 10),
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
        )

        best_gen = None
        best_q = float("-inf")
        for gen in range(20):
            q = float(rng.uniform(-100, 0))
            desc = rng.uniform(0.0, 1.0, size=2)
            archive.add(_make_entry(descriptor=desc, quality=q))
            tracker.record(generation=gen, archive=archive, improvements=1)

            current_best = archive.best().quality
            if current_best > best_q:
                best_q = current_best
                best_gen = gen

        assert tracker.best_generation() is not None

    def test_tracker_summary(self, sample_archive):
        tracker = ArchiveStatsTracker(window_size=3)
        for gen in range(5):
            tracker.record(generation=gen, archive=sample_archive, improvements=gen)

        summary = tracker.summary()
        assert "coverage" in summary
        assert "qd_score" in summary
        assert "best_quality" in summary
        assert "improvement_rate" in summary
        assert "plateau_length" in summary

    def test_tracker_clear(self, sample_archive):
        tracker = ArchiveStatsTracker()
        tracker.record(generation=0, archive=sample_archive, improvements=1)
        assert len(tracker.records) == 1
        tracker.clear()
        assert len(tracker.records) == 0

    def test_grid_archive_summary_dict(self, sample_archive):
        """GridArchive.summary() should return a dict with expected keys."""
        summary = sample_archive.summary()
        expected_keys = {
            "coverage", "qd_score", "best_quality", "mean_quality",
            "num_elites", "diversity", "total_insertions",
            "total_replacements", "total_fills",
        }
        assert set(summary.keys()) == expected_keys
        assert summary["num_elites"] == len(sample_archive)

    def test_diversity_increases_with_spread(self, empty_archive):
        """Diversity should increase when elites are spread across descriptor space."""
        # Insert two entries very close together
        e1 = _make_entry(descriptor=np.array([0.50, 0.50]), quality=-10.0)
        e2 = _make_entry(descriptor=np.array([0.52, 0.52]), quality=-20.0)
        empty_archive.add(e1)
        empty_archive.add(e2)
        div_close = empty_archive.diversity()

        # Create archive with widely separated entries
        wide = GridArchive(
            dims=(5, 5),
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
        )
        w1 = _make_entry(descriptor=np.array([0.05, 0.05]), quality=-10.0)
        w2 = _make_entry(descriptor=np.array([0.95, 0.95]), quality=-20.0)
        wide.add(w1)
        wide.add(w2)
        div_wide = wide.diversity()

        assert div_wide > div_close


# ===================================================================
# Sampling — valid DAGs
# ===================================================================


class TestSampleElitesReturnsValidDAGs:
    """Test that sampled elites contain valid DAG adjacency matrices."""

    def test_sample_elites_returns_valid_dags(self, rng):
        """Sampled elites should have solutions that are valid DAGs."""
        archive = GridArchive(
            dims=(10, 10),
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
        )

        # Populate with entries whose solutions are valid DAGs
        for _ in range(30):
            entry = _make_dag_entry(rng, n_nodes=5, edge_prob=0.3)
            archive.add(entry)

        # Sample and verify
        sampled = archive.sample(n=20, rng=rng)
        assert len(sampled) == 20

        for s in sampled:
            assert isinstance(s, ArchiveEntry)
            adj = s.solution
            assert adj.shape == (5, 5)
            assert _is_dag(adj), f"Sampled solution is not a DAG:\n{adj}"

    def test_sample_from_empty_returns_empty(self, empty_archive, rng):
        result = empty_archive.sample(n=5, rng=rng)
        assert result == []

    def test_sample_with_replacement(self, sample_archive, rng):
        """Sampling more than num_elites should work (with replacement)."""
        n_elites = len(sample_archive)
        sampled = sample_archive.sample(n=n_elites * 3, rng=rng)
        assert len(sampled) == n_elites * 3

    def test_sample_quality_proportional(self, rng):
        """Quality-proportional sampling should favour higher-quality elites."""
        archive = GridArchive(
            dims=(10, 10),
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
        )
        # One very high quality, rest low
        archive.add(_make_entry(descriptor=np.array([0.15, 0.15]), quality=-1.0))
        for i in range(9):
            desc = np.array([0.15 + (i + 1) * 0.08, 0.5])
            archive.add(_make_entry(descriptor=desc, quality=-100.0))

        sampled = archive.sample_quality_proportional(n=100, rng=rng)
        high_count = sum(1 for s in sampled if s.quality > -50.0)
        # The high-quality elite should be sampled frequently
        assert high_count > 20

    def test_sample_curiosity(self, rng):
        """Curiosity sampling should exist and return the right count."""
        archive = GridArchive(
            dims=(5, 5),
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
        )
        for i in range(5):
            desc = np.array([0.1 + i * 0.2, 0.5])
            archive.add(_make_entry(descriptor=desc, quality=float(-10 * (i + 1))))

        sampled = archive.sample_curiosity(n=10, rng=rng)
        assert len(sampled) == 10
        for s in sampled:
            assert isinstance(s, ArchiveEntry)

    def test_cvt_sample_returns_valid_entries(self, rng):
        """CVTArchive.sample should return valid ArchiveEntry objects."""
        archive = CVTArchive(
            n_cells=10,
            descriptor_dim=2,
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
            rng=rng,
            n_samples=500,
        )
        for _ in range(15):
            entry = _make_dag_entry(rng, n_nodes=5)
            archive.add(entry)

        sampled = archive.sample(n=10, rng=rng)
        assert len(sampled) == 10
        for s in sampled:
            assert isinstance(s, ArchiveEntry)
            assert s.solution.shape == (5, 5)
            assert _is_dag(s.solution)


# ===================================================================
# GridArchive — additional edge cases
# ===================================================================


class TestGridArchiveEdgeCases:
    """Miscellaneous edge-case tests for GridArchive."""

    def test_descriptor_clipping(self, empty_archive):
        """Descriptors outside bounds should be clipped to valid cells."""
        out_of_bounds = _make_entry(
            descriptor=np.array([-5.0, 100.0]),
            quality=-30.0,
        )
        assert empty_archive.add(out_of_bounds) is True
        idx = empty_archive._descriptor_to_index(out_of_bounds.descriptor)
        # Should be clipped to corner cell
        assert idx[0] == 0  # lower bound dimension 0
        assert idx[1] == empty_archive.dims[1] - 1  # upper bound dimension 1

    def test_descriptor_on_boundary(self, empty_archive):
        """Descriptor exactly at upper bound should map to the last bin."""
        boundary = _make_entry(
            descriptor=np.array([1.0, 1.0]),
            quality=-10.0,
        )
        empty_archive.add(boundary)
        idx = empty_archive._descriptor_to_index(boundary.descriptor)
        assert idx == (4, 4)

    def test_add_batch(self, empty_archive, rng):
        """add_batch should process multiple entries at once."""
        entries = [_make_dag_entry(rng) for _ in range(10)]
        results = empty_archive.add_batch(entries)
        assert len(results) == 10
        assert any(results)

    def test_quality_grid(self, sample_archive):
        """as_quality_grid should return array with correct shape."""
        grid = sample_archive.as_quality_grid()
        assert grid.shape == sample_archive.dims
        # Occupied cells should have finite values
        occupied_count = np.isfinite(grid).sum()
        assert occupied_count == len(sample_archive)

    def test_occupancy_grid(self, sample_archive):
        grid = sample_archive.as_occupancy_grid()
        assert grid.shape == sample_archive.dims
        assert grid.sum() == len(sample_archive)

    def test_occupied_indices(self, sample_archive):
        indices = sample_archive.occupied_indices()
        assert len(indices) == len(sample_archive)
        for idx in indices:
            assert idx in sample_archive

    def test_index_to_descriptor_center(self, empty_archive):
        """Cell center should lie within that cell's boundaries."""
        center = empty_archive.index_to_descriptor_center((2, 3))
        # For a 5×5 grid in [0,1]×[0,1], cell (2,3) center should be
        # (2.5/5, 3.5/5) = (0.5, 0.7)
        np.testing.assert_array_almost_equal(center, [0.5, 0.7])

    def test_empty_neighbor_count(self, empty_archive):
        """Corner cells should have 2 neighbors; all empty initially."""
        count = empty_archive.empty_neighbor_count((0, 0))
        assert count == 2  # right and down

    def test_descriptor_variance(self, sample_archive):
        var = sample_archive.descriptor_variance()
        assert var.shape == (2,)
        assert np.all(var >= 0.0)

    def test_descriptor_variance_empty(self, empty_archive):
        var = empty_archive.descriptor_variance()
        np.testing.assert_array_equal(var, [0.0, 0.0])

    def test_repr(self, sample_archive):
        r = repr(sample_archive)
        assert "GridArchive" in r
        assert "dims=" in r

    def test_3d_grid_archive(self, rng):
        """GridArchive should work with 3-dimensional descriptor spaces."""
        archive = GridArchive(
            dims=(3, 3, 3),
            lower_bounds=np.array([0.0, 0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0, 1.0]),
        )
        assert archive.total_cells == 27
        assert archive.descriptor_dim == 3

        entry = ArchiveEntry(
            solution=np.zeros((5, 5), dtype=np.int8),
            descriptor=np.array([0.5, 0.5, 0.5]),
            quality=-10.0,
        )
        assert archive.add(entry) is True
        assert len(archive) == 1

    def test_single_dim_grid_archive(self):
        """GridArchive should work with 1-dimensional descriptor space."""
        archive = GridArchive(
            dims=(10,),
            lower_bounds=np.array([0.0]),
            upper_bounds=np.array([1.0]),
        )
        assert archive.total_cells == 10
        assert archive.descriptor_dim == 1

        entry = ArchiveEntry(
            solution=np.eye(3, dtype=np.int8),
            descriptor=np.array([0.5]),
            quality=-5.0,
        )
        assert archive.add(entry) is True


# ===================================================================
# CVTArchive — additional tests
# ===================================================================


class TestCVTArchiveAdditional:
    """Additional CVTArchive tests covering edge cases and properties."""

    def test_cvt_diversity(self, rng):
        """Diversity should be non-negative for populated archive."""
        archive = CVTArchive(
            n_cells=10,
            descriptor_dim=2,
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
            rng=rng,
            n_samples=500,
        )
        for _ in range(5):
            desc = rng.uniform(0.0, 1.0, size=2)
            archive.add(_make_entry(descriptor=desc, quality=rng.uniform(-100, 0)))

        assert archive.diversity() >= 0.0

    def test_cvt_repr(self, rng):
        archive = CVTArchive(
            n_cells=5,
            descriptor_dim=2,
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
            rng=rng,
            n_samples=100,
        )
        r = repr(archive)
        assert "CVTArchive" in r

    def test_cvt_iter_and_contains(self, rng):
        archive = CVTArchive(
            n_cells=10,
            descriptor_dim=2,
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
            rng=rng,
            n_samples=500,
        )
        desc = np.array([0.5, 0.5])
        archive.add(_make_entry(descriptor=desc, quality=-10.0))

        idx = archive._descriptor_to_index(desc)
        assert idx in archive

        elites = list(archive)
        assert len(elites) == 1
        assert elites[0].quality == -10.0

    def test_cvt_add_batch(self, rng):
        archive = CVTArchive(
            n_cells=10,
            descriptor_dim=2,
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
            rng=rng,
            n_samples=500,
        )
        entries = [_make_dag_entry(rng) for _ in range(8)]
        results = archive.add_batch(entries)
        assert len(results) == 8
        assert any(results)

    def test_cvt_summary(self, rng):
        archive = CVTArchive(
            n_cells=10,
            descriptor_dim=2,
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
            rng=rng,
            n_samples=500,
        )
        for _ in range(5):
            archive.add(_make_dag_entry(rng))

        summary = archive.summary()
        assert "coverage" in summary
        assert "qd_score" in summary
        assert "num_elites" in summary
        assert summary["num_elites"] == len(archive)

    def test_cvt_higher_dim(self, rng):
        """CVTArchive should work with higher-dimensional descriptors."""
        archive = CVTArchive(
            n_cells=10,
            descriptor_dim=5,
            lower_bounds=np.zeros(5),
            upper_bounds=np.ones(5),
            rng=rng,
            n_samples=500,
        )
        assert archive.descriptor_dim == 5
        assert archive.centroids.shape == (10, 5)

        entry = ArchiveEntry(
            solution=np.zeros((3, 3), dtype=np.int8),
            descriptor=rng.uniform(0.0, 1.0, size=5),
            quality=-10.0,
        )
        assert archive.add(entry) is True
        assert len(archive) == 1
