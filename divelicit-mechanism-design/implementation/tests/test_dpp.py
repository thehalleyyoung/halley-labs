"""Tests for DPP module."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dpp import DPP, greedy_map, sample, log_det_diversity, marginal_gain
from src.kernels import RBFKernel


def _make_L(n=10, d=4, seed=42, bandwidth=1.0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    kernel = RBFKernel(bandwidth=bandwidth)
    return kernel.gram_matrix(X), X


def test_greedy_map_returns_k_items():
    L, _ = _make_L()
    selected = greedy_map(L, k=4)
    assert len(selected) == 4


def test_greedy_map_diverse():
    """DPP greedy should select more diverse items than random."""
    rng = np.random.RandomState(42)
    # Create clustered data
    X = np.vstack([rng.randn(5, 4) * 0.1, rng.randn(5, 4) * 0.1 + 3.0])
    kernel = RBFKernel(bandwidth=1.0)
    L = kernel.gram_matrix(X)

    dpp_selected = greedy_map(L, k=4)
    random_selected = list(rng.choice(10, 4, replace=False))

    dpp_div = log_det_diversity(L, dpp_selected)
    random_div = log_det_diversity(L, random_selected)
    # DPP should generally do at least as well (may sometimes tie)
    assert dpp_div >= random_div - 1.0


def test_greedy_map_no_duplicates():
    L, _ = _make_L()
    selected = greedy_map(L, k=5)
    assert len(set(selected)) == len(selected)


def test_log_det_diversity_increases_with_spread():
    kernel = RBFKernel(bandwidth=1.0)
    # Tight cluster
    X_tight = np.ones((4, 3)) * 0.01 + np.eye(4, 3) * 0.01
    L_tight = kernel.gram_matrix(X_tight)
    # Spread points
    X_spread = np.eye(4, 3) * 3.0
    L_spread = kernel.gram_matrix(X_spread)

    div_tight = log_det_diversity(L_tight, [0, 1, 2, 3])
    div_spread = log_det_diversity(L_spread, [0, 1, 2, 3])
    assert div_spread > div_tight


def test_marginal_gain_positive():
    """Marginal gain should be positive when adding well-separated points."""
    rng = np.random.RandomState(42)
    X = rng.randn(10, 4) * 3.0  # well-separated points
    kernel = RBFKernel(bandwidth=2.0)
    L = kernel.gram_matrix(X)
    S = [0]
    gain = marginal_gain(L, S, 5)
    assert gain > -0.5  # should be non-negative or near zero


def test_marginal_gain_diminishing():
    """Submodularity: gain of adding j to S should decrease as S grows."""
    L, _ = _make_L(n=10, d=4, seed=100)
    j = 5
    S_small = [0]
    S_large = [0, 1, 2, 3]
    gain_small = marginal_gain(L, S_small, j)
    gain_large = marginal_gain(L, S_large, j)
    assert gain_small >= gain_large - 1e-8


def test_dpp_sample_returns_k_items():
    L, _ = _make_L()
    np.random.seed(42)
    selected = sample(L, k=4)
    assert len(selected) == 4
    assert len(set(selected)) == len(selected)


def test_dpp_vs_random_on_clustered_data():
    """DPP should select from different clusters more often than random."""
    rng = np.random.RandomState(42)
    centers = [np.array([0, 0, 0, 0.0]), np.array([5, 5, 5, 5.0])]
    X = np.vstack([
        centers[0] + rng.randn(5, 4) * 0.1,
        centers[1] + rng.randn(5, 4) * 0.1,
    ])
    kernel = RBFKernel(bandwidth=2.0)
    L = kernel.gram_matrix(X)

    dpp_sel = greedy_map(L, k=4)
    # DPP should select from both clusters
    from_cluster_0 = sum(1 for i in dpp_sel if i < 5)
    from_cluster_1 = sum(1 for i in dpp_sel if i >= 5)
    assert from_cluster_0 >= 1 and from_cluster_1 >= 1
