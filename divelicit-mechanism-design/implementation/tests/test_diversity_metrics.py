"""Tests for diversity metrics."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.diversity_metrics import (
    cosine_diversity,
    mmd,
    sinkhorn_diversity_metric,
    log_det_diversity,
    vendi_score,
    dispersion_metric,
    diversity_profile,
)
from src.kernels import RBFKernel


def test_cosine_diversity_identical_zero():
    """Identical vectors should have zero cosine diversity."""
    X = np.tile(np.array([1.0, 2.0, 3.0]), (5, 1))
    assert cosine_diversity(X) < 1e-6


def test_cosine_diversity_orthogonal_one():
    """Orthogonal vectors should have cosine diversity close to 1."""
    X = np.eye(5, 5)
    div = cosine_diversity(X)
    assert div > 0.9


def test_mmd_identical_zero():
    """MMD between identical sets should be ~0."""
    X = np.random.RandomState(42).randn(10, 3)
    m = mmd(X, X)
    assert m < 1e-6


def test_sinkhorn_diversity_uniform_high():
    """Well-spread points should have high Sinkhorn diversity score."""
    rng = np.random.RandomState(42)
    X = rng.uniform(-5, 5, size=(20, 3))
    score = sinkhorn_diversity_metric(X)
    assert score > 0.0  # Should be positive


def test_log_det_diversity_collinear_low():
    """Collinear points should have very low log-det diversity."""
    t = np.linspace(0, 1, 5)
    X = np.column_stack([t, t, t])
    kernel = RBFKernel(bandwidth=1.0)
    ld = log_det_diversity(X, kernel)
    # Compare with well-spread points
    rng = np.random.RandomState(42)
    X_spread = rng.randn(5, 3) * 3.0
    ld_spread = log_det_diversity(X_spread, kernel)
    assert ld < ld_spread


def test_vendi_score_identical_one():
    """Identical items should give Vendi score of 1."""
    X = np.tile(np.array([1.0, 0.0, 0.0]), (5, 1))
    vs = vendi_score(X)
    assert vs < 1.5  # Should be close to 1


def test_dispersion_properties():
    """Dispersion should be positive for non-identical points."""
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    d = dispersion_metric(X)
    assert d > 0.9  # minimum distance is 1.0


def test_diversity_profile_all_keys():
    """Profile should contain all expected metrics."""
    X = np.random.RandomState(42).randn(5, 3)
    profile = diversity_profile(X)
    expected_keys = {"cosine_diversity", "log_det_diversity", "vendi_score",
                     "dispersion", "sinkhorn_diversity"}
    assert set(profile.keys()) == expected_keys
    for v in profile.values():
        assert np.isfinite(v)
