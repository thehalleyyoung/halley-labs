"""Tests for coverage certificates."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.coverage import (
    CoverageCertificate,
    estimate_coverage,
    coverage_lower_bound,
    required_samples,
    coverage_test,
    dispersion,
    fill_distance,
)


def test_full_coverage_high():
    """Dense grid covering domain should have high coverage."""
    # 2D grid covering [0,1]^2
    x = np.linspace(0, 1, 10)
    points = np.array([[xi, xj] for xi in x for xj in x])
    cert = estimate_coverage(points, epsilon=0.2)
    assert cert.coverage_fraction > 0.5


def test_no_coverage_low():
    """Single point in large domain should have low coverage."""
    points = np.array([[0.0, 0.0]])
    cert = estimate_coverage(points, epsilon=0.01, domain_volume=1000.0)
    assert cert.coverage_fraction < 0.01


def test_coverage_increases_with_points():
    """More points should increase coverage."""
    rng = np.random.RandomState(42)
    coverages = []
    for n in [5, 20, 50]:
        points = rng.randn(n, 2)
        cert = estimate_coverage(points, epsilon=0.5)
        coverages.append(cert.coverage_fraction)
    # Coverage should generally increase (or at least not decrease much)
    assert coverages[-1] >= coverages[0] - 0.1


def test_coverage_certificate_confidence():
    cert = estimate_coverage(np.random.randn(10, 2), epsilon=0.5)
    assert 0.0 < cert.confidence <= 1.0
    assert cert.n_samples == 10
    assert cert.epsilon_radius == 0.5


def test_required_samples_increases_with_dim():
    """Higher dimensions need more samples for same coverage."""
    n_2d = required_samples(0.5, epsilon=0.3, dim=2)
    n_8d = required_samples(0.5, epsilon=0.3, dim=8)
    assert n_8d >= n_2d


def test_dispersion_uniform_high():
    """Well-spread points should have high dispersion."""
    # Points on vertices of a cube
    points = np.array([[0, 0], [0, 10], [10, 0], [10, 10]], dtype=float)
    d = dispersion(points)
    assert d >= 9.9  # minimum distance is 10


def test_fill_distance_decreases_with_points():
    """More well-placed points should decrease fill distance."""
    rng = np.random.RandomState(42)
    reference = rng.uniform(0, 1, size=(100, 2))

    fd_small = fill_distance(rng.uniform(0, 1, size=(5, 2)), reference)
    fd_large = fill_distance(rng.uniform(0, 1, size=(50, 2)), reference)
    assert fd_large <= fd_small + 0.1  # more points -> lower fill distance


def test_coverage_test_empirical():
    """Empirical coverage test against reference points."""
    rng = np.random.RandomState(42)
    points = rng.randn(20, 2)
    reference = rng.randn(50, 2)
    cert = coverage_test(points, reference, epsilon=2.0)
    assert 0.0 <= cert.coverage_fraction <= 1.0
    assert cert.coverage_fraction > 0.0  # should cover at least some points
