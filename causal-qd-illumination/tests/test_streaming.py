"""Tests for causal_qd.streaming module.

Covers OnlineArchive, IncrementalDescriptor, and StreamingStats.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from causal_qd.core.dag import DAG
from causal_qd.data.scm import LinearGaussianSCM
from causal_qd.descriptors.structural import StructuralDescriptor
from causal_qd.streaming.online_archive import OnlineArchive
from causal_qd.streaming.incremental_descriptor import IncrementalDescriptor
from causal_qd.streaming.stats import StreamingStats
from causal_qd.types import AdjacencyMatrix, BehavioralDescriptor, DataMatrix


# ===================================================================
# Helpers
# ===================================================================

def _chain_dag(n: int) -> DAG:
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n - 1):
        adj[i, i + 1] = 1
    return DAG(adj)


class _DAGAwareStructuralDescriptor:
    """Wrapper that extracts adjacency from DAG objects before delegating."""

    def __init__(self) -> None:
        self._inner = StructuralDescriptor()

    def compute(self, dag_or_adj, data):
        adj = dag_or_adj.adjacency if hasattr(dag_or_adj, "adjacency") else dag_or_adj
        return self._inner.compute(adj, data)


def _make_data(dag: DAG, n_samples: int = 500, seed: int = 42) -> DataMatrix:
    scm = LinearGaussianSCM.from_dag(dag, rng=np.random.default_rng(seed))
    return scm.sample(n_samples, rng=np.random.default_rng(seed + 1))


# ===================================================================
# OnlineArchive Tests
# ===================================================================

class TestOnlineArchive:
    """Tests for causal_qd.streaming.online_archive.OnlineArchive."""

    def test_online_archive_updates(self):
        archive = OnlineArchive(
            dims=(5, 5),
            bounds=(np.array([0.0, 0.0]), np.array([1.0, 1.0])),
        )
        adj = np.zeros((5, 5), dtype=np.int8)
        desc = np.array([0.3, 0.7])

        added = archive.add(adj, quality=-50.0, descriptor=desc)
        assert added is True
        assert len(archive) == 1

    def test_add_better_quality_replaces(self):
        archive = OnlineArchive(
            dims=(5, 5),
            bounds=(np.array([0.0, 0.0]), np.array([1.0, 1.0])),
        )
        adj = np.zeros((5, 5), dtype=np.int8)
        desc = np.array([0.3, 0.7])

        archive.add(adj, quality=-100.0, descriptor=desc)
        replaced = archive.add(adj, quality=-10.0, descriptor=desc)
        assert replaced is True
        assert len(archive) == 1

    def test_add_worse_quality_not_replaced(self):
        archive = OnlineArchive(
            dims=(5, 5),
            bounds=(np.array([0.0, 0.0]), np.array([1.0, 1.0])),
        )
        adj = np.zeros((5, 5), dtype=np.int8)
        desc = np.array([0.3, 0.7])

        archive.add(adj, quality=-10.0, descriptor=desc)
        replaced = archive.add(adj, quality=-100.0, descriptor=desc)
        assert replaced is False
        assert len(archive) == 1

    def test_coverage_increases(self):
        archive = OnlineArchive(
            dims=(5, 5),
            bounds=(np.array([0.0, 0.0]), np.array([1.0, 1.0])),
        )
        adj = np.zeros((5, 5), dtype=np.int8)
        rng = np.random.default_rng(0)

        coverages = []
        for _ in range(15):
            desc = rng.uniform(0, 1, size=2)
            archive.add(adj, quality=float(rng.uniform(-100, -10)), descriptor=desc)
            coverages.append(archive.coverage())

        for i in range(1, len(coverages)):
            assert coverages[i] >= coverages[i - 1] - 1e-12

    def test_qd_score(self):
        archive = OnlineArchive(
            dims=(5, 5),
            bounds=(np.array([0.0, 0.0]), np.array([1.0, 1.0])),
        )
        adj = np.zeros((5, 5), dtype=np.int8)
        archive.add(adj, quality=10.0, descriptor=np.array([0.5, 0.5]))
        assert archive.qd_score() == pytest.approx(10.0)

    def test_sample(self):
        archive = OnlineArchive(
            dims=(5, 5),
            bounds=(np.array([0.0, 0.0]), np.array([1.0, 1.0])),
        )
        adj = np.zeros((5, 5), dtype=np.int8)
        rng = np.random.default_rng(0)
        for _ in range(5):
            desc = rng.uniform(0, 1, size=2)
            archive.add(adj, quality=-50.0, descriptor=desc)

        samples = archive.sample(3, rng=np.random.default_rng(1))
        assert len(samples) == 3

    def test_step(self):
        archive = OnlineArchive(
            dims=(5, 5),
            bounds=(np.array([0.0, 0.0]), np.array([1.0, 1.0])),
            decay_factor=0.99,
        )
        adj = np.zeros((5, 5), dtype=np.int8)
        archive.add(adj, quality=100.0, descriptor=np.array([0.5, 0.5]))

        evicted = archive.step()
        assert isinstance(evicted, int)

    def test_stats(self):
        archive = OnlineArchive(
            dims=(5, 5),
            bounds=(np.array([0.0, 0.0]), np.array([1.0, 1.0])),
        )
        adj = np.zeros((5, 5), dtype=np.int8)
        archive.add(adj, quality=-50.0, descriptor=np.array([0.5, 0.5]))
        s = archive.stats()
        assert isinstance(s, dict)

    def test_empty_archive_coverage_zero(self):
        archive = OnlineArchive(
            dims=(5, 5),
            bounds=(np.array([0.0, 0.0]), np.array([1.0, 1.0])),
        )
        assert archive.coverage() == pytest.approx(0.0)

    def test_many_additions(self):
        archive = OnlineArchive(
            dims=(10, 10),
            bounds=(np.array([0.0, 0.0]), np.array([1.0, 1.0])),
        )
        adj = np.zeros((5, 5), dtype=np.int8)
        rng = np.random.default_rng(0)
        for _ in range(100):
            desc = rng.uniform(0, 1, size=2)
            archive.add(adj, quality=float(rng.uniform(-100, 0)), descriptor=desc)
        assert len(archive) > 0
        assert archive.coverage() > 0.0


# ===================================================================
# IncrementalDescriptor Tests
# ===================================================================

class TestIncrementalDescriptor:
    """Tests for causal_qd.streaming.incremental_descriptor.IncrementalDescriptor."""

    def test_compute_matches_base(self):
        """IncrementalDescriptor.compute should match the base descriptor computer."""
        dag = _chain_dag(5)
        data = _make_data(dag, 200)

        base = _DAGAwareStructuralDescriptor()
        inc = IncrementalDescriptor(base)

        d_base = base.compute(dag.adjacency, data)
        d_inc = inc.compute(dag, data)

        npt.assert_allclose(d_inc, d_base, atol=1e-10)

    def test_incremental_descriptor_matches_batch(self):
        """After an edge update, incremental should match a fresh batch compute."""
        dag = _chain_dag(5)
        data = _make_data(dag, 200)

        base = _DAGAwareStructuralDescriptor()
        inc = IncrementalDescriptor(base)

        old_desc = inc.compute(dag, data)

        # Add an edge 0→3
        new_adj = dag.adjacency.copy()
        new_adj[0, 3] = 1
        new_dag = DAG(new_adj)

        updated = inc.update(old_desc, new_dag, data, edge=(0, 3), added=True)
        batch = base.compute(new_adj, data)

        npt.assert_allclose(updated, batch, atol=1e-10)

    def test_incremental_edge_removal(self):
        """Removing an edge incrementally should match batch recompute."""
        dag = _chain_dag(5)
        data = _make_data(dag, 200)

        base = _DAGAwareStructuralDescriptor()
        inc = IncrementalDescriptor(base)

        old_desc = inc.compute(dag, data)

        new_adj = dag.adjacency.copy()
        new_adj[1, 2] = 0  # remove edge 1→2
        new_dag = DAG(new_adj)

        updated = inc.update(old_desc, new_dag, data, edge=(1, 2), added=False)
        batch = base.compute(new_adj, data)

        npt.assert_allclose(updated, batch, atol=1e-10)

    def test_descriptor_stats(self):
        dag = _chain_dag(5)
        data = _make_data(dag, 200)

        base = _DAGAwareStructuralDescriptor()
        inc = IncrementalDescriptor(base, cache_statistics=True)

        inc.compute(dag, data)
        # After at least one compute, stats should be available
        # The descriptor_stats property returns _DescriptorStats
        stats = inc.descriptor_stats
        assert stats is not None


# ===================================================================
# StreamingStats Tests
# ===================================================================

class TestStreamingStats:
    """Tests for causal_qd.streaming.stats.StreamingStats."""

    def test_streaming_stats_accuracy(self):
        """StreamingStats mean/variance should match numpy on the same data."""
        rng = np.random.default_rng(0)
        values = rng.standard_normal(1000)

        ss = StreamingStats()
        for v in values:
            ss.update(float(v))

        npt.assert_allclose(ss.mean, np.mean(values), atol=1e-10)
        npt.assert_allclose(ss.variance, np.var(values, ddof=1), atol=1e-6)
        npt.assert_allclose(ss.std, np.std(values, ddof=1), atol=1e-6)
        assert ss.count == 1000

    def test_min_max(self):
        ss = StreamingStats()
        for v in [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]:
            ss.update(v)
        assert ss.min == pytest.approx(1.0)
        assert ss.max == pytest.approx(9.0)

    def test_single_value(self):
        ss = StreamingStats()
        ss.update(5.0)
        assert ss.mean == pytest.approx(5.0)
        assert ss.count == 1
        assert ss.min == pytest.approx(5.0)
        assert ss.max == pytest.approx(5.0)

    def test_update_batch(self):
        rng = np.random.default_rng(0)
        values = list(rng.standard_normal(100))

        ss1 = StreamingStats()
        for v in values:
            ss1.update(float(v))

        ss2 = StreamingStats()
        ss2.update_batch(values)

        npt.assert_allclose(ss1.mean, ss2.mean, atol=1e-10)
        npt.assert_allclose(ss1.variance, ss2.variance, atol=1e-10)

    def test_summary(self):
        ss = StreamingStats()
        for v in [1.0, 2.0, 3.0]:
            ss.update(v)
        s = ss.summary()
        assert isinstance(s, dict)
        assert "mean" in s
        assert "variance" in s or "var" in s

    def test_merge(self):
        rng = np.random.default_rng(0)
        all_values = rng.standard_normal(200)

        ss1 = StreamingStats()
        for v in all_values[:100]:
            ss1.update(float(v))

        ss2 = StreamingStats()
        for v in all_values[100:]:
            ss2.update(float(v))

        merged = ss1.merge(ss2)
        assert merged.count == 200
        npt.assert_allclose(merged.mean, np.mean(all_values), atol=1e-10)

    def test_reset(self):
        ss = StreamingStats()
        ss.update(1.0)
        ss.update(2.0)
        ss.reset()
        assert ss.count == 0

    def test_large_values(self):
        """Welford's algorithm should handle large values without overflow."""
        ss = StreamingStats()
        vals = [1e10, 1e10 + 1, 1e10 + 2]
        for v in vals:
            ss.update(v)
        assert ss.mean == pytest.approx(np.mean(vals), rel=1e-10)
        assert ss.variance == pytest.approx(np.var(vals, ddof=1), rel=1e-6)
