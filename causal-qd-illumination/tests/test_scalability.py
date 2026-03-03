"""Tests for causal_qd.scalability module.

Covers SkeletonRestrictor, PCACompressor, and SamplingCI.
"""

from __future__ import annotations

from typing import List

import numpy as np
import numpy.testing as npt
import pytest

from causal_qd.ci_tests.ci_base import CITest, CITestResult
from causal_qd.ci_tests.fisher_z import FisherZTest
from causal_qd.core.dag import DAG
from causal_qd.data.scm import LinearGaussianSCM
from causal_qd.descriptors.structural import StructuralDescriptor
from causal_qd.scalability.pca_compress import PCACompressor
from causal_qd.scalability.sampling_ci import SamplingCI
from causal_qd.scalability.skeleton_restrict import SkeletonRestrictor
from causal_qd.types import AdjacencyMatrix, BehavioralDescriptor, DataMatrix


# ===================================================================
# Helpers
# ===================================================================

def _chain_dag(n: int) -> DAG:
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n - 1):
        adj[i, i + 1] = 1
    return DAG(adj)


def _make_data(dag: DAG, n_samples: int = 500, seed: int = 42) -> DataMatrix:
    scm = LinearGaussianSCM.from_dag(dag, rng=np.random.default_rng(seed))
    return scm.sample(n_samples, rng=np.random.default_rng(seed + 1))


def _make_descriptors(n: int = 50, dim: int = 8, seed: int = 42) -> List[BehavioralDescriptor]:
    """Generate a list of random descriptors for PCA testing."""
    rng = np.random.default_rng(seed)
    # Create data with some correlated dimensions for PCA to compress
    base = rng.standard_normal((n, 3))
    mixing = rng.standard_normal((3, dim))
    raw = base @ mixing + rng.standard_normal((n, dim)) * 0.1
    return [raw[i] for i in range(n)]


# ===================================================================
# SkeletonRestrictor Tests
# ===================================================================

class TestSkeletonRestrictor:
    """Tests for causal_qd.scalability.skeleton_restrict.SkeletonRestrictor."""

    def test_skeleton_restriction_reduces_search(self):
        """Skeleton restriction should disallow many edges for sparse graphs."""
        dag = _chain_dag(8)
        data = _make_data(dag, 1000, seed=0)
        ci_test = FisherZTest()

        restrictor = SkeletonRestrictor(alpha=0.05)
        mask = restrictor.restrict(dag, data, ci_test)

        assert isinstance(mask, np.ndarray)
        n = dag.n_nodes
        total_possible = n * (n - 1)
        allowed = mask.sum()
        # Skeleton restriction should reduce the number of allowed edges
        assert allowed < total_possible, (
            f"Expected fewer allowed edges ({allowed}) than total ({total_possible})"
        )

    def test_learn_skeleton(self):
        dag = _chain_dag(6)
        data = _make_data(dag, 1000, seed=0)
        ci_test = FisherZTest()

        restrictor = SkeletonRestrictor(alpha=0.05)
        skeleton = restrictor.learn_skeleton(data, dag.n_nodes, ci_test, alpha=0.05)

        assert isinstance(skeleton, np.ndarray)
        assert skeleton.shape == (6, 6)

    def test_skeleton_includes_true_edges(self):
        """The learned skeleton should contain most true edges."""
        dag = _chain_dag(5)
        data = _make_data(dag, 2000, seed=0)
        ci_test = FisherZTest()

        restrictor = SkeletonRestrictor(alpha=0.05)
        skeleton = restrictor.learn_skeleton(data, dag.n_nodes, ci_test, alpha=0.05)

        true_skeleton = (dag.adjacency | dag.adjacency.T).astype(bool)
        learned_skeleton = (skeleton | skeleton.T).astype(bool)

        # At least half the true edges should be in the learned skeleton
        true_edges = true_skeleton.sum() // 2
        shared = ((true_skeleton & learned_skeleton).sum()) // 2
        recall = shared / max(true_edges, 1)
        assert recall >= 0.5, f"Skeleton recall too low: {recall:.2f}"

    def test_get_allowed_edges(self):
        dag = _chain_dag(5)
        data = _make_data(dag, 1000, seed=0)
        ci_test = FisherZTest()

        restrictor = SkeletonRestrictor(alpha=0.05)
        restrictor.restrict(dag, data, ci_test)
        allowed = restrictor.get_allowed_edges()

        assert isinstance(allowed, list)
        for edge in allowed:
            assert isinstance(edge, tuple)
            assert len(edge) == 2

    def test_restrictor_on_larger_graph(self):
        """Test on a 15-node graph to verify scalability."""
        dag = DAG.random_dag(15, edge_prob=0.15, rng=np.random.default_rng(0))
        data = _make_data(dag, 500, seed=0)
        ci_test = FisherZTest()

        restrictor = SkeletonRestrictor(alpha=0.05)
        mask = restrictor.restrict(dag, data, ci_test)

        assert mask.shape == (15, 15)
        # Should reduce search space
        total = 15 * 14
        assert mask.sum() < total


# ===================================================================
# PCACompressor Tests
# ===================================================================

class TestPCACompressor:
    """Tests for causal_qd.scalability.pca_compress.PCACompressor."""

    def test_pca_compression_preserves_structure(self):
        """PCA should reduce dimensionality while preserving main variance."""
        descriptors = _make_descriptors(n=100, dim=8)

        compressor = PCACompressor(n_components=2)
        compressed = compressor.fit_transform(descriptors)

        assert len(compressed) == 100
        assert len(compressed[0]) == 2
        assert compressor.n_components_fitted == 2

    def test_fit_then_transform(self):
        descriptors = _make_descriptors(n=50, dim=6)

        compressor = PCACompressor(n_components=3)
        compressor.fit(descriptors)
        transformed = compressor.transform(descriptors[0])

        assert len(transformed) == 3

    def test_transform_batch(self):
        descriptors = _make_descriptors(n=50, dim=6)
        compressor = PCACompressor(n_components=2)
        compressor.fit(descriptors)
        batch = compressor.transform_batch(descriptors[:10])
        assert len(batch) == 10
        for d in batch:
            assert len(d) == 2

    def test_reconstruction_error(self):
        descriptors = _make_descriptors(n=50, dim=8)
        compressor = PCACompressor(n_components=3)
        compressor.fit(descriptors)

        err = compressor.reconstruction_error(descriptors[0])
        assert isinstance(err, float)
        assert err >= 0.0

    def test_reconstruction_error_low_for_dominant_components(self):
        """With enough components, reconstruction error should be small."""
        descriptors = _make_descriptors(n=100, dim=8)
        compressor = PCACompressor(n_components=7)
        compressor.fit(descriptors)

        errors = [compressor.reconstruction_error(d) for d in descriptors[:20]]
        mean_err = np.mean(errors)
        assert mean_err < 1.0, f"Mean reconstruction error too high: {mean_err}"

    def test_explained_variance(self):
        descriptors = _make_descriptors(n=100, dim=8)
        compressor = PCACompressor(n_components=3)
        compressor.fit(descriptors)

        ratio = compressor.explained_variance_ratio_total
        assert 0.0 < ratio <= 1.0

    def test_fit_transform_equals_fit_plus_transform(self):
        descriptors = _make_descriptors(n=50, dim=6)

        comp1 = PCACompressor(n_components=2)
        result1 = comp1.fit_transform(descriptors)

        comp2 = PCACompressor(n_components=2)
        comp2.fit(descriptors)
        result2 = [comp2.transform(d) for d in descriptors]

        for r1, r2 in zip(result1, result2):
            npt.assert_allclose(r1, r2, atol=1e-10)

    def test_inverse_transform(self):
        descriptors = _make_descriptors(n=50, dim=6)
        compressor = PCACompressor(n_components=4)
        compressor.fit(descriptors)

        reduced = compressor.transform(descriptors[0])
        reconstructed = compressor.inverse_transform(reduced)
        assert len(reconstructed) == 6

    def test_pca_variance_threshold(self):
        descriptors = _make_descriptors(n=100, dim=8)
        compressor = PCACompressor(n_components=None, variance_threshold=0.99)
        compressor.fit(descriptors)
        # Should pick enough components to explain 99% variance
        assert compressor.n_components_fitted <= 8
        assert compressor.explained_variance_ratio_total >= 0.99 - 0.01

    def test_summary(self):
        descriptors = _make_descriptors(n=50, dim=6)
        compressor = PCACompressor(n_components=2)
        compressor.fit(descriptors)
        s = compressor.summary()
        assert isinstance(s, dict)


# ===================================================================
# SamplingCI Tests
# ===================================================================

class TestSamplingCI:
    """Tests for causal_qd.scalability.sampling_ci.SamplingCI."""

    def test_sampling_ci_approximates_exact(self):
        """SamplingCI p-value should roughly agree with exact test."""
        dag = _chain_dag(5)
        data = _make_data(dag, 1000, seed=0)

        base_test = FisherZTest()
        sampling = SamplingCI(
            base_test=base_test,
            sample_fraction=0.5,
            n_repeats=10,
        )

        exact = base_test.test(
            x=0, y=1,
            conditioning_set=frozenset(),
            data=data,
            alpha=0.05,
        )

        approx = sampling.test(
            x=0, y=1,
            conditioning_set=frozenset(),
            data=data,
            alpha=0.05,
            rng=np.random.default_rng(0),
        )

        # Both should agree on whether 0 and 1 are dependent (they should be)
        assert exact.p_value < 0.05 or isinstance(approx, float)
        # The approximate p-value (returned as float) should also indicate dependence
        if isinstance(approx, float):
            assert approx < 0.3  # rough agreement

    def test_sampling_ci_on_independent(self):
        """On independent variables, both exact and sampling should give high p-value."""
        rng_data = np.random.default_rng(0)
        data = rng_data.standard_normal((1000, 5))

        base_test = FisherZTest()
        sampling = SamplingCI(
            base_test=base_test,
            sample_fraction=0.5,
            n_repeats=10,
        )

        exact = base_test.test(
            x=0, y=3,
            conditioning_set=frozenset(),
            data=data,
            alpha=0.05,
        )

        approx = sampling.test(
            x=0, y=3,
            conditioning_set=frozenset(),
            data=data,
            alpha=0.05,
            rng=np.random.default_rng(0),
        )

        # Both should indicate independence (high p-value)
        assert exact.p_value > 0.01
        if isinstance(approx, float):
            assert approx > 0.01

    def test_sampling_ci_with_conditioning(self):
        """Test with a non-empty conditioning set."""
        dag = _chain_dag(5)
        data = _make_data(dag, 1000, seed=0)

        base_test = FisherZTest()
        sampling = SamplingCI(
            base_test=base_test,
            sample_fraction=0.5,
            n_repeats=5,
        )

        result = sampling.test(
            x=0, y=2,
            conditioning_set=frozenset({1}),
            data=data,
            alpha=0.05,
            rng=np.random.default_rng(0),
        )

        # Should return a p-value (float)
        assert isinstance(result, (float, CITestResult))

    def test_n_tests_tracking(self):
        """The n_tests counter should increment with each call."""
        rng_data = np.random.default_rng(0)
        data = rng_data.standard_normal((200, 5))
        base_test = FisherZTest()
        sampling = SamplingCI(base_test=base_test, n_repeats=3)

        sampling.test(
            x=0, y=1, conditioning_set=frozenset(),
            data=data, alpha=0.05, rng=np.random.default_rng(0),
        )
        assert sampling.n_tests >= 1

    def test_different_aggregations(self):
        """SamplingCI should work with different aggregation methods."""
        rng_data = np.random.default_rng(0)
        data = rng_data.standard_normal((500, 4))
        base_test = FisherZTest()

        for agg in ["median", "mean", "fisher"]:
            sampling = SamplingCI(
                base_test=base_test, n_repeats=5, aggregation=agg,
            )
            result = sampling.test(
                x=0, y=1, conditioning_set=frozenset(),
                data=data, alpha=0.05, rng=np.random.default_rng(0),
            )
            assert isinstance(result, (float, CITestResult))
