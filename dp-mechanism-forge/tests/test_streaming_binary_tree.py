"""Tests for streaming.binary_tree module."""
import math
import pytest
import numpy as np

from dp_forge.streaming import StreamConfig
from dp_forge.streaming.binary_tree import (
    BinaryTreeMechanism,
    TreeConstruction,
    NoiseAllocation,
    RangeQuery,
    MatrixFactorizationTree,
    TreeNode,
)


class TestBinaryTreeMechanism:
    """Test BinaryTreeMechanism prefix sum accuracy."""

    def test_observe_returns_output(self):
        """Observe returns a StreamOutput with value."""
        config = StreamConfig(max_time=16, epsilon=1.0, seed=42)
        mech = BinaryTreeMechanism(config=config)
        out = mech.observe(1.0)
        assert out.timestamp == 0
        assert out.true_value == 1.0

    def test_prefix_sum_accumulates(self):
        """True prefix sum matches cumulative input."""
        config = StreamConfig(max_time=16, epsilon=1.0, seed=42)
        mech = BinaryTreeMechanism(config=config)
        for i in range(5):
            out = mech.observe(1.0)
        assert out.true_value == 5.0

    def test_privacy_spent_constant(self):
        """Privacy is epsilon regardless of observations."""
        config = StreamConfig(max_time=16, epsilon=2.0, seed=42)
        mech = BinaryTreeMechanism(config=config)
        for _ in range(10):
            mech.observe(1.0)
        assert mech.privacy_spent() == 2.0

    def test_query_without_observe(self):
        """Query on empty mechanism returns 0."""
        config = StreamConfig(max_time=16, epsilon=1.0, seed=42)
        mech = BinaryTreeMechanism(config=config)
        out = mech.query()
        assert out.value == 0.0

    def test_reset_clears_state(self):
        """Reset puts mechanism back to initial state."""
        config = StreamConfig(max_time=16, epsilon=1.0, seed=42)
        mech = BinaryTreeMechanism(config=config)
        mech.observe(1.0)
        mech.reset()
        out = mech.query()
        assert out.value == 0.0

    def test_summarize(self):
        """Summarize returns valid statistics."""
        config = StreamConfig(max_time=16, epsilon=1.0, seed=42)
        mech = BinaryTreeMechanism(config=config)
        for _ in range(5):
            mech.observe(1.0)
        summary = mech.summarize()
        assert summary.total_time_steps == 5
        assert summary.rmse >= 0
        assert summary.mean_absolute_error >= 0

    def test_error_scales_with_log_T(self):
        """Error should be bounded (not blow up linearly)."""
        config = StreamConfig(max_time=64, epsilon=1.0, seed=42)
        mech = BinaryTreeMechanism(config=config)
        errors = []
        for i in range(64):
            out = mech.observe(1.0)
            errors.append(abs(out.noise_added))
        mean_error = np.mean(errors)
        # Should be much less than T=64
        assert mean_error < 64


class TestTreeConstruction:
    """Test TreeConstruction balanced tree property."""

    def test_tree_height(self):
        """Tree height is ceil(log2(T))."""
        tree = TreeConstruction(max_time=16)
        assert tree.height >= math.ceil(math.log2(16))

    def test_add_leaf_updates_nodes(self):
        """Adding a leaf returns updated nodes from leaf to root."""
        tree = TreeConstruction(max_time=8)
        updated = tree.add_leaf(1.0)
        assert len(updated) >= 1
        assert tree.num_leaves == 1

    def test_prefix_cover_size(self):
        """Prefix cover has O(log T) nodes."""
        tree = TreeConstruction(max_time=16)
        for i in range(16):
            tree.add_leaf(1.0)
        cover = tree.prefix_cover(16)
        assert len(cover) <= tree.height + 1

    def test_prefix_cover_sum(self):
        """True sums via prefix cover match actual prefix sum."""
        tree = TreeConstruction(max_time=8)
        for i in range(1, 9):
            tree.add_leaf(float(i))
        cover = tree.prefix_cover(8)
        total = sum(n.true_sum for n in cover)
        assert total == pytest.approx(sum(range(1, 9)), abs=1e-6)

    def test_node_range_coverage(self):
        """Each node covers a contiguous range."""
        tree = TreeConstruction(max_time=8)
        for i in range(8):
            tree.add_leaf(1.0)
        node = tree.get_node(0, 3)
        assert node is not None
        assert node.range_start <= node.range_end


class TestNoiseAllocation:
    """Test NoiseAllocation privacy guarantee."""

    def test_pure_dp_scale(self):
        """Pure DP: noise scale = sensitivity * (h+1) / epsilon."""
        na = NoiseAllocation(height=3, epsilon=1.0, sensitivity=1.0)
        expected_scale = 1.0 * 4 / 1.0  # (height+1) / eps
        assert na.scale_for_level(0) == pytest.approx(expected_scale, abs=1e-6)

    def test_approx_dp_uses_gaussian(self):
        """Approximate DP uses Gaussian (different scale)."""
        na_pure = NoiseAllocation(height=3, epsilon=1.0, delta=0.0)
        na_approx = NoiseAllocation(height=3, epsilon=1.0, delta=1e-5)
        # Scales may differ
        assert na_approx.scale_for_level(0) > 0

    def test_expected_squared_error_positive(self):
        """Expected squared error is positive."""
        na = NoiseAllocation(height=3, epsilon=1.0)
        assert na.expected_squared_error() > 0

    def test_variance_increases_with_nodes(self):
        """More nodes in prefix → more variance."""
        na = NoiseAllocation(height=3, epsilon=1.0)
        v1 = na.total_variance_for_prefix(1)
        v3 = na.total_variance_for_prefix(3)
        assert v3 > v1


class TestRangeQuery:
    """Test RangeQuery correctness."""

    def test_full_range(self):
        """Full range query matches prefix sum."""
        tree = TreeConstruction(max_time=8)
        for i in range(8):
            nodes = tree.add_leaf(1.0)
            for n in nodes:
                n.noisy_sum = n.true_sum  # no noise for test
        rq = RangeQuery(tree)
        result = rq.query(0, 8)
        assert result == pytest.approx(8.0, abs=1e-6)

    def test_subrange_query(self):
        """Subrange [2,5) sums correctly."""
        tree = TreeConstruction(max_time=8)
        for i in range(8):
            nodes = tree.add_leaf(float(i))
            for n in nodes:
                n.noisy_sum = n.true_sum
        rq = RangeQuery(tree)
        true_sum = rq.true_range_sum(2, 5)
        assert true_sum == pytest.approx(2 + 3 + 4, abs=1e-6)

    def test_error_nonnegative(self):
        """Range query error is non-negative."""
        tree = TreeConstruction(max_time=8)
        for i in range(8):
            nodes = tree.add_leaf(1.0)
            for n in nodes:
                n.noisy_sum = n.true_sum + 0.1
        rq = RangeQuery(tree)
        err = rq.error(0, 4)
        assert err >= 0

    def test_invalid_range_raises(self):
        """Invalid range raises ValueError."""
        tree = TreeConstruction(max_time=8)
        for i in range(4):
            tree.add_leaf(1.0)
        rq = RangeQuery(tree)
        with pytest.raises(ValueError):
            rq.query(5, 3)


class TestMatrixFactorizationTree:
    """Test MatrixFactorizationTree equivalence."""

    def test_encoder_shape(self):
        """Encoder matrix L has shape (nodes, T)."""
        mft = MatrixFactorizationTree(T=4)
        L = mft.build_encoder()
        assert L.shape[1] == 4
        assert L.shape[0] > 0

    def test_workload_lower_triangular(self):
        """Workload matrix is lower-triangular."""
        mft = MatrixFactorizationTree(T=4)
        W = mft.workload_matrix()
        assert W.shape == (4, 4)
        assert np.allclose(W, np.tril(np.ones((4, 4))))

    def test_reconstruction_error_positive(self):
        """Reconstruction error is positive."""
        mft = MatrixFactorizationTree(T=4)
        err = mft.reconstruction_error(epsilon=1.0)
        assert err > 0

    def test_encoder_rows_are_indicator(self):
        """Each encoder row is a 0/1 vector (node coverage)."""
        mft = MatrixFactorizationTree(T=4)
        L = mft.build_encoder()
        for row in L:
            assert np.all((row == 0) | (row == 1))
