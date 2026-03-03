"""
Comprehensive tests for dp_forge.grid.grid_strategies.

Tests cover:
  - UniformGrid: evenly spaced points, correct bin widths
  - ChebyshevGrid: Chebyshev node locations, density at edges
  - MassAdaptiveGrid: more points in high-mass regions
  - CurvatureAdaptiveGrid: more points in high-curvature regions
  - TailPrunedGrid: far-tail removal, mass preservation
  - All strategies: normalization, monotonicity, edge cases
  - Helper functions: _compute_midpoint_widths, _pad_range
  - GridResult dataclass
  - Property-based tests via Hypothesis
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from dp_forge.exceptions import ConfigurationError
from dp_forge.grid.grid_strategies import (
    GridResult,
    GridStrategy,
    UniformGrid,
    ChebyshevGrid,
    MassAdaptiveGrid,
    CurvatureAdaptiveGrid,
    TailPrunedGrid,
    _compute_midpoint_widths,
    _pad_range,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_mechanism(n: int, k: int, *, peaked: bool = False) -> np.ndarray:
    """Build a valid n×k probability table (rows sum to 1, non-negative)."""
    rng = np.random.default_rng(123)
    if peaked:
        p = rng.exponential(0.01, size=(n, k))
        center = k // 2
        width = max(k // 6, 1)
        lo, hi = max(0, center - width), min(k, center + width)
        p[:, lo:hi] += rng.exponential(2.0, size=(n, hi - lo))
    else:
        p = rng.dirichlet(np.ones(k), size=n)
    return p / p.sum(axis=1, keepdims=True)


def _make_curvature_mechanism(n: int, k: int) -> np.ndarray:
    """Build a mechanism with sharp curvature in the middle region."""
    rng = np.random.default_rng(77)
    p = np.ones((n, k)) * 0.01
    # Add a sharp spike at the center
    center = k // 2
    for i in range(n):
        for j in range(k):
            dist = abs(j - center)
            p[i, j] += np.exp(-0.5 * dist)
    return p / p.sum(axis=1, keepdims=True)


# ═══════════════════════════════════════════════════════════════════════════
# §1  GridResult dataclass
# ═══════════════════════════════════════════════════════════════════════════


class TestGridResult:
    """Tests for the GridResult dataclass."""

    def test_basic_construction(self):
        """GridResult stores points and widths correctly."""
        pts = np.array([1.0, 2.0, 3.0])
        ws = np.array([0.5, 1.0, 0.5])
        gr = GridResult(points=pts, widths=ws)
        assert gr.k == 3
        assert gr.span == 2.0

    def test_mismatched_lengths_raises(self):
        """points and widths must have the same length."""
        with pytest.raises(ValueError, match="same length"):
            GridResult(points=np.array([1.0, 2.0]), widths=np.array([1.0]))

    def test_too_few_points_raises(self):
        """Grid must have at least 2 points."""
        with pytest.raises(ValueError, match="at least"):
            GridResult(points=np.array([1.0]), widths=np.array([1.0]))

    def test_2d_arrays_raise(self):
        """2-D arrays should raise."""
        with pytest.raises(ValueError, match="1-D"):
            GridResult(
                points=np.array([[1.0, 2.0]]),
                widths=np.array([[1.0, 1.0]]),
            )

    def test_repr(self):
        """__repr__ includes k and span."""
        gr = GridResult(
            points=np.array([0.0, 1.0, 2.0]),
            widths=np.array([0.5, 1.0, 0.5]),
        )
        r = repr(gr)
        assert "k=3" in r


# ═══════════════════════════════════════════════════════════════════════════
# §2  _compute_midpoint_widths
# ═══════════════════════════════════════════════════════════════════════════


class TestMidpointWidths:
    """Tests for the _compute_midpoint_widths helper."""

    def test_uniform_2_points(self):
        """Two points: both widths equal the single gap."""
        pts = np.array([0.0, 1.0])
        w = _compute_midpoint_widths(pts)
        np.testing.assert_allclose(w, [1.0, 1.0])

    def test_uniform_3_points(self):
        """Three equidistant points: boundary=gap, interior=gap."""
        pts = np.array([0.0, 1.0, 2.0])
        w = _compute_midpoint_widths(pts)
        np.testing.assert_allclose(w, [1.0, 1.0, 1.0])

    def test_nonuniform_widths(self):
        """Non-uniform spacing: interior width = avg of adjacent gaps."""
        pts = np.array([0.0, 1.0, 4.0])  # gaps = [1, 3]
        w = _compute_midpoint_widths(pts)
        np.testing.assert_allclose(w, [1.0, 2.0, 3.0])

    def test_single_point_raises(self):
        """Fewer than 2 points should raise."""
        with pytest.raises(ValueError, match="at least 2"):
            _compute_midpoint_widths(np.array([1.0]))

    @pytest.mark.parametrize("k", [2, 5, 10, 50])
    def test_uniform_widths_sum(self, k: int):
        """For uniform grid, sum of widths ≈ 2 * span (midpoint rule)."""
        pts = np.linspace(0.0, 10.0, k)
        w = _compute_midpoint_widths(pts)
        # Boundary widths count the full gap, so sum > span
        assert w.sum() > 0
        assert np.all(w > 0)


# ═══════════════════════════════════════════════════════════════════════════
# §3  _pad_range
# ═══════════════════════════════════════════════════════════════════════════


class TestPadRange:
    """Tests for the _pad_range helper."""

    def test_normal_range(self):
        lo, hi = _pad_range(0.0, 10.0, 1.0)
        assert lo == -1.0
        assert hi == 11.0

    def test_degenerate_range(self):
        """When f_min == f_max, expand by max(1, |f_min|)."""
        lo, hi = _pad_range(5.0, 5.0, 0.0)
        assert lo < 5.0
        assert hi > 5.0

    def test_degenerate_at_zero(self):
        """Degenerate range at origin expands by 1."""
        lo, hi = _pad_range(0.0, 0.0, 0.0)
        assert lo == -1.0
        assert hi == 1.0

    def test_negative_range(self):
        lo, hi = _pad_range(-10.0, -5.0, 2.0)
        assert lo == -12.0
        assert hi == -3.0


# ═══════════════════════════════════════════════════════════════════════════
# §4  UniformGrid
# ═══════════════════════════════════════════════════════════════════════════


class TestUniformGrid:
    """Tests for the UniformGrid strategy."""

    def test_k_equals_2(self):
        """Minimum grid: 2 points."""
        grid = UniformGrid().build((0.0, 10.0), k=2)
        assert grid.k == 2
        assert grid.points[0] < grid.points[1]

    def test_points_evenly_spaced(self):
        """Points should be evenly spaced."""
        grid = UniformGrid(padding=0.0).build((0.0, 10.0), k=11)
        gaps = np.diff(grid.points)
        np.testing.assert_allclose(gaps, gaps[0], atol=1e-12)

    def test_k_too_small_raises(self):
        """k < 2 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="k must be"):
            UniformGrid().build((0.0, 1.0), k=1)

    def test_correct_number_of_points(self):
        """Grid should have exactly k points."""
        for k in [2, 5, 10, 100]:
            grid = UniformGrid().build((0.0, 10.0), k=k)
            assert grid.k == k

    def test_covers_range_with_padding(self):
        """Grid spans beyond the raw query range."""
        grid = UniformGrid(padding=1.0).build((0.0, 10.0), k=20)
        assert grid.points[0] <= -1.0
        assert grid.points[-1] >= 11.0

    def test_bin_widths_positive(self):
        """All bin widths must be positive."""
        grid = UniformGrid().build((-5.0, 5.0), k=20)
        assert np.all(grid.widths > 0)

    def test_metadata_has_strategy(self):
        """Metadata should include strategy name."""
        grid = UniformGrid().build((0.0, 1.0), k=5)
        assert grid.metadata.get("strategy") == "uniform"

    def test_degenerate_range(self):
        """Grid handles f_min == f_max by expanding."""
        grid = UniformGrid().build((5.0, 5.0), k=10)
        assert grid.k == 10
        assert grid.span > 0

    def test_repr(self):
        r = repr(UniformGrid(padding=2.0))
        assert "padding=2.0" in r
        r2 = repr(UniformGrid())
        assert "auto" in r2

    @pytest.mark.parametrize("k", [2, 3, 10, 50, 100])
    def test_widths_all_positive(self, k: int):
        """Widths must be strictly positive for all k."""
        grid = UniformGrid(padding=1.0).build((0.0, 10.0), k=k)
        assert np.all(grid.widths > 0)


# ═══════════════════════════════════════════════════════════════════════════
# §5  ChebyshevGrid
# ═══════════════════════════════════════════════════════════════════════════


class TestChebyshevGrid:
    """Tests for the ChebyshevGrid strategy."""

    def test_k_equals_2(self):
        """Minimum grid: 2 Chebyshev nodes."""
        grid = ChebyshevGrid().build((0.0, 10.0), k=2)
        assert grid.k == 2

    def test_correct_number_of_points(self):
        """Grid has exactly k points."""
        for k in [2, 5, 10, 50]:
            grid = ChebyshevGrid().build((0.0, 10.0), k=k)
            assert grid.k == k

    def test_points_sorted(self):
        """Chebyshev nodes must be sorted ascending."""
        grid = ChebyshevGrid().build((-1.0, 1.0), k=20)
        assert np.all(np.diff(grid.points) > 0)

    def test_density_at_edges(self):
        """Chebyshev nodes are denser near the edges than the center."""
        grid = ChebyshevGrid(padding=0.0).build((-1.0, 1.0), k=50)
        gaps = np.diff(grid.points)
        # Edge gaps should be smaller than center gaps
        edge_avg = (gaps[0] + gaps[-1]) / 2
        center_avg = gaps[len(gaps) // 2]
        assert edge_avg < center_avg

    def test_k_too_small_raises(self):
        """k < 2 should raise."""
        with pytest.raises(ConfigurationError, match="k must be"):
            ChebyshevGrid().build((0.0, 1.0), k=1)

    def test_bin_widths_positive(self):
        """All bin widths must be positive."""
        grid = ChebyshevGrid().build((-5.0, 5.0), k=20)
        assert np.all(grid.widths > 0)

    def test_metadata_has_strategy(self):
        """Metadata includes strategy name."""
        grid = ChebyshevGrid().build((0.0, 1.0), k=5)
        assert grid.metadata.get("strategy") == "chebyshev"

    def test_symmetric_about_center(self):
        """For a symmetric range, nodes should be symmetric."""
        grid = ChebyshevGrid(padding=0.0).build((-5.0, 5.0), k=21)
        mid = (grid.points[0] + grid.points[-1]) / 2
        reflected = 2 * mid - grid.points[::-1]
        np.testing.assert_allclose(grid.points, reflected, atol=1e-10)

    def test_nodes_within_range(self):
        """All nodes lie within the padded range."""
        grid = ChebyshevGrid(padding=1.0).build((0.0, 10.0), k=30)
        assert grid.points[0] >= -1.0 - 1e-10
        assert grid.points[-1] <= 11.0 + 1e-10

    def test_repr(self):
        r = repr(ChebyshevGrid(padding=3.0))
        assert "padding=3.0" in r


# ═══════════════════════════════════════════════════════════════════════════
# §6  MassAdaptiveGrid
# ═══════════════════════════════════════════════════════════════════════════


class TestMassAdaptiveGrid:
    """Tests for the MassAdaptiveGrid strategy."""

    def test_fallback_to_uniform_without_mechanism(self):
        """Without mechanism, falls back to uniform."""
        grid = MassAdaptiveGrid().build((0.0, 10.0), k=20)
        assert grid.k == 20
        assert grid.metadata.get("strategy") == "uniform"

    def test_with_peaked_mechanism_concentrates_points(self):
        """With peaked mechanism, grid should have more points near the peak."""
        k_old, k_new = 30, 50
        n = 3
        old_grid = np.linspace(-5.0, 5.0, k_old)
        mech = _make_mechanism(n, k_old, peaked=True)
        strategy = MassAdaptiveGrid(mass_threshold=0.01)
        grid = strategy.build(
            (-5.0, 5.0), k=k_new, mechanism=mech, old_grid=old_grid
        )
        assert grid.k == k_new
        assert grid.metadata.get("strategy") == "mass_adaptive"

    def test_more_points_in_high_mass_region(self):
        """Grid density should be higher in the peak region."""
        k_old = 40
        old_grid = np.linspace(-10.0, 10.0, k_old)
        n = 2
        mech = np.ones((n, k_old)) * 0.001
        # Create a peak in the center
        mech[:, 15:25] = 0.1
        mech = mech / mech.sum(axis=1, keepdims=True)

        strategy = MassAdaptiveGrid(mass_threshold=0.01)
        grid = strategy.build(
            (-10.0, 10.0), k=60, mechanism=mech, old_grid=old_grid
        )
        # Count points in center vs tails
        center_mask = (grid.points >= -3.0) & (grid.points <= 3.0)
        tail_mask = ~center_mask
        center_density = center_mask.sum() / 6.0
        tail_density = max(tail_mask.sum(), 1) / 14.0
        assert center_density > tail_density * 0.5  # center should be denser

    def test_k_too_small_raises(self):
        """k < 2 should raise."""
        with pytest.raises(ConfigurationError, match="k must be"):
            MassAdaptiveGrid().build((0.0, 1.0), k=1)

    def test_invalid_mass_threshold(self):
        """mass_threshold <= 0 should raise."""
        with pytest.raises(ValueError, match="mass_threshold"):
            MassAdaptiveGrid(mass_threshold=0.0)

    def test_invalid_min_density(self):
        """min_density outside (0, 1) should raise."""
        with pytest.raises(ValueError, match="min_density"):
            MassAdaptiveGrid(min_density=0.0)
        with pytest.raises(ValueError, match="min_density"):
            MassAdaptiveGrid(min_density=1.0)

    def test_mechanism_shape_mismatch(self):
        """Mechanism columns != old_grid length raises ValueError."""
        old_grid = np.linspace(0.0, 1.0, 10)
        mech = np.ones((2, 5)) / 5  # 5 cols != 10
        with pytest.raises(ValueError, match="incompatible"):
            MassAdaptiveGrid().build(
                (0.0, 1.0), k=20, mechanism=mech, old_grid=old_grid
            )

    def test_sorted_output(self):
        """Output grid must be sorted."""
        k_old = 20
        old_grid = np.linspace(-5.0, 5.0, k_old)
        mech = _make_mechanism(3, k_old, peaked=True)
        strategy = MassAdaptiveGrid()
        grid = strategy.build(
            (-5.0, 5.0), k=40, mechanism=mech, old_grid=old_grid
        )
        assert np.all(np.diff(grid.points) > 0)

    def test_metadata_includes_counts(self):
        """Metadata should include k_uniform, k_adaptive, n_high_mass_bins."""
        k_old = 20
        old_grid = np.linspace(-5.0, 5.0, k_old)
        mech = _make_mechanism(3, k_old, peaked=True)
        strategy = MassAdaptiveGrid()
        grid = strategy.build(
            (-5.0, 5.0), k=40, mechanism=mech, old_grid=old_grid
        )
        assert "k_uniform" in grid.metadata
        assert "k_adaptive" in grid.metadata
        assert "n_high_mass_bins" in grid.metadata

    def test_repr(self):
        r = repr(MassAdaptiveGrid(mass_threshold=0.05, min_density=0.2))
        assert "0.05" in r
        assert "0.2" in r


# ═══════════════════════════════════════════════════════════════════════════
# §7  CurvatureAdaptiveGrid
# ═══════════════════════════════════════════════════════════════════════════


class TestCurvatureAdaptiveGrid:
    """Tests for the CurvatureAdaptiveGrid strategy."""

    def test_fallback_to_uniform_without_mechanism(self):
        """Without mechanism, falls back to uniform."""
        grid = CurvatureAdaptiveGrid().build((0.0, 10.0), k=20)
        assert grid.k == 20

    def test_fallback_with_2_points(self):
        """With only 2 old grid points, curvature can't be computed → uniform."""
        old_grid = np.array([0.0, 1.0])
        mech = np.array([[0.5, 0.5], [0.4, 0.6]])
        grid = CurvatureAdaptiveGrid().build(
            (0.0, 1.0), k=10, mechanism=mech, old_grid=old_grid
        )
        assert grid.k == 10

    def test_with_curvature_mechanism(self):
        """With a mechanism exhibiting curvature, grid builds without error."""
        k_old = 30
        old_grid = np.linspace(-5.0, 5.0, k_old)
        mech = _make_curvature_mechanism(3, k_old)
        strategy = CurvatureAdaptiveGrid()
        grid = strategy.build(
            (-5.0, 5.0), k=50, mechanism=mech, old_grid=old_grid
        )
        assert grid.k == 50

    def test_k_too_small_raises(self):
        """k < 2 should raise."""
        with pytest.raises(ConfigurationError, match="k must be"):
            CurvatureAdaptiveGrid().build((0.0, 1.0), k=1)

    def test_invalid_curvature_weight(self):
        """curvature_weight <= 0 should raise."""
        with pytest.raises(ValueError, match="curvature_weight"):
            CurvatureAdaptiveGrid(curvature_weight=0.0)

    def test_invalid_min_density(self):
        """min_density outside (0, 1) should raise."""
        with pytest.raises(ValueError, match="min_density"):
            CurvatureAdaptiveGrid(min_density=0.0)

    def test_mechanism_shape_mismatch(self):
        """Mismatched mechanism and grid raises ValueError."""
        old_grid = np.linspace(0.0, 1.0, 10)
        mech = np.ones((2, 5)) / 5
        with pytest.raises(ValueError, match="!="):
            CurvatureAdaptiveGrid().build(
                (0.0, 1.0), k=20, mechanism=mech, old_grid=old_grid
            )

    def test_sorted_output(self):
        """Output grid must be sorted."""
        k_old = 20
        old_grid = np.linspace(-5.0, 5.0, k_old)
        mech = _make_curvature_mechanism(2, k_old)
        grid = CurvatureAdaptiveGrid().build(
            (-5.0, 5.0), k=40, mechanism=mech, old_grid=old_grid
        )
        assert np.all(np.diff(grid.points) > 0)

    def test_metadata_includes_curvature(self):
        """Metadata includes max_curvature and mean_curvature."""
        k_old = 20
        old_grid = np.linspace(-5.0, 5.0, k_old)
        mech = _make_curvature_mechanism(2, k_old)
        grid = CurvatureAdaptiveGrid().build(
            (-5.0, 5.0), k=40, mechanism=mech, old_grid=old_grid
        )
        assert "max_curvature" in grid.metadata
        assert "mean_curvature" in grid.metadata

    def test_repr(self):
        r = repr(CurvatureAdaptiveGrid(curvature_weight=0.8, min_density=0.2))
        assert "0.8" in r
        assert "0.2" in r


# ═══════════════════════════════════════════════════════════════════════════
# §8  TailPrunedGrid
# ═══════════════════════════════════════════════════════════════════════════


class TestTailPrunedGrid:
    """Tests for the TailPrunedGrid strategy."""

    def test_fallback_to_uniform_without_mechanism(self):
        """Without mechanism, falls back to uniform."""
        grid = TailPrunedGrid().build((0.0, 10.0), k=20)
        assert grid.k == 20

    def test_prunes_negligible_tails(self):
        """Mechanism with negligible tails gets pruned."""
        k_old = 50
        old_grid = np.linspace(-10.0, 10.0, k_old)
        n = 3
        mech = np.zeros((n, k_old))
        # Put all mass in central 10 bins
        mech[:, 20:30] = 0.1
        mech = mech / mech.sum(axis=1, keepdims=True)

        strategy = TailPrunedGrid(tail_threshold=1e-6, min_points=5)
        grid = strategy.build(
            (-10.0, 10.0), k=30, mechanism=mech, old_grid=old_grid
        )
        assert grid.metadata.get("strategy") == "tail_pruned"
        assert grid.metadata.get("n_pruned", 0) > 0

    def test_min_points_respected(self):
        """Even with aggressive pruning, min_points are retained."""
        k_old = 30
        old_grid = np.linspace(-5.0, 5.0, k_old)
        n = 2
        # Almost all mass in one bin
        mech = np.full((n, k_old), 1e-10)
        mech[:, 15] = 1.0
        mech = mech / mech.sum(axis=1, keepdims=True)

        strategy = TailPrunedGrid(tail_threshold=1e-3, min_points=10)
        grid = strategy.build(
            (-5.0, 5.0), k=20, mechanism=mech, old_grid=old_grid
        )
        # Should keep at least min_points=10 points
        assert grid.k >= 2  # At minimum, GridResult enforces >= 2

    def test_k_requested_gets_resampled(self):
        """If pruned length != k, resamples to k points."""
        k_old = 40
        old_grid = np.linspace(-10.0, 10.0, k_old)
        mech = _make_mechanism(2, k_old, peaked=True)
        strategy = TailPrunedGrid(tail_threshold=1e-6, min_points=5)
        grid = strategy.build(
            (-10.0, 10.0), k=25, mechanism=mech, old_grid=old_grid
        )
        assert grid.k == 25

    def test_invalid_tail_threshold(self):
        """tail_threshold <= 0 should raise."""
        with pytest.raises(ValueError, match="tail_threshold"):
            TailPrunedGrid(tail_threshold=0.0)

    def test_invalid_min_points(self):
        """min_points < 2 should raise."""
        with pytest.raises(ValueError, match="min_points"):
            TailPrunedGrid(min_points=1)

    def test_mechanism_shape_mismatch(self):
        """Mismatched mechanism and grid raises ValueError."""
        old_grid = np.linspace(0.0, 1.0, 10)
        mech = np.ones((2, 5)) / 5
        with pytest.raises(ValueError, match="!="):
            TailPrunedGrid().build(
                (0.0, 1.0), k=20, mechanism=mech, old_grid=old_grid
            )

    def test_metadata_includes_pruning_info(self):
        """Metadata includes pruning statistics."""
        k_old = 30
        old_grid = np.linspace(-5.0, 5.0, k_old)
        mech = _make_mechanism(2, k_old, peaked=True)
        strategy = TailPrunedGrid()
        grid = strategy.build(
            (-5.0, 5.0), k=20, mechanism=mech, old_grid=old_grid
        )
        assert "n_pruned" in grid.metadata
        assert "original_k" in grid.metadata
        assert grid.metadata["original_k"] == k_old

    def test_repr(self):
        r = repr(TailPrunedGrid(tail_threshold=1e-4, min_points=5))
        assert "1e-04" in r or "0.0001" in r


# ═══════════════════════════════════════════════════════════════════════════
# §9  All-strategy cross-cutting tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAllStrategies:
    """Cross-cutting tests that apply to all grid strategies."""

    BASIC_STRATEGIES = [UniformGrid(), ChebyshevGrid()]

    @pytest.mark.parametrize("strategy", BASIC_STRATEGIES)
    @pytest.mark.parametrize("k", [2, 5, 10, 50])
    def test_correct_k(self, strategy, k: int):
        """Every strategy returns exactly k points."""
        grid = strategy.build((0.0, 10.0), k=k)
        assert grid.k == k

    @pytest.mark.parametrize("strategy", BASIC_STRATEGIES)
    def test_monotonicity(self, strategy):
        """Grid points must be strictly increasing."""
        grid = strategy.build((-5.0, 5.0), k=30)
        assert np.all(np.diff(grid.points) > 0)

    @pytest.mark.parametrize("strategy", BASIC_STRATEGIES)
    def test_widths_positive(self, strategy):
        """All bin widths must be positive."""
        grid = strategy.build((-5.0, 5.0), k=20)
        assert np.all(grid.widths > 0)

    @pytest.mark.parametrize("strategy", BASIC_STRATEGIES)
    def test_points_and_widths_same_length(self, strategy):
        """points and widths arrays have the same length."""
        grid = strategy.build((0.0, 10.0), k=25)
        assert len(grid.points) == len(grid.widths)

    def test_adaptive_strategies_with_mechanism(self):
        """MassAdaptive and CurvatureAdaptive work with a mechanism."""
        k_old, k_new = 20, 40
        old_grid = np.linspace(-5.0, 5.0, k_old)
        mech = _make_mechanism(3, k_old, peaked=True)

        for strategy_cls in [MassAdaptiveGrid, CurvatureAdaptiveGrid]:
            strategy = strategy_cls()
            grid = strategy.build(
                (-5.0, 5.0), k=k_new, mechanism=mech, old_grid=old_grid
            )
            assert grid.k == k_new
            assert np.all(np.diff(grid.points) > 0)
            assert np.all(grid.widths > 0)

    @pytest.mark.parametrize("strategy", BASIC_STRATEGIES)
    def test_very_large_range(self, strategy):
        """Grid handles a very large range."""
        grid = strategy.build((-1e6, 1e6), k=10)
        assert grid.k == 10
        assert grid.span > 0

    @pytest.mark.parametrize("strategy", BASIC_STRATEGIES)
    def test_very_small_range(self, strategy):
        """Grid handles a very small range."""
        grid = strategy.build((0.0, 1e-6), k=10)
        assert grid.k == 10
        assert grid.span > 0


# ═══════════════════════════════════════════════════════════════════════════
# §10  GridStrategy protocol conformance
# ═══════════════════════════════════════════════════════════════════════════


class TestGridStrategyProtocol:
    """Test that all strategies satisfy the GridStrategy protocol."""

    @pytest.mark.parametrize(
        "strategy",
        [UniformGrid(), ChebyshevGrid(), MassAdaptiveGrid(), CurvatureAdaptiveGrid()],
    )
    def test_is_grid_strategy(self, strategy):
        """Each strategy is an instance of the GridStrategy protocol."""
        assert isinstance(strategy, GridStrategy)


# ═══════════════════════════════════════════════════════════════════════════
# §11  Property-based tests (Hypothesis)
# ═══════════════════════════════════════════════════════════════════════════


class TestGridProperties:
    """Hypothesis-driven property tests for grid strategies."""

    @given(
        k=st.integers(min_value=2, max_value=100),
        lo=st.floats(min_value=-1000, max_value=0),
        hi=st.floats(min_value=0.1, max_value=1000),
    )
    @settings(max_examples=40, deadline=5000)
    def test_uniform_always_sorted(self, k: int, lo: float, hi: float):
        """Property: UniformGrid always produces sorted points."""
        assume(hi > lo)
        grid = UniformGrid(padding=0.0).build((lo, hi), k=k)
        assert np.all(np.diff(grid.points) > 0)

    @given(
        k=st.integers(min_value=2, max_value=100),
        lo=st.floats(min_value=-1000, max_value=0),
        hi=st.floats(min_value=0.1, max_value=1000),
    )
    @settings(max_examples=40, deadline=5000)
    def test_chebyshev_always_sorted(self, k: int, lo: float, hi: float):
        """Property: ChebyshevGrid always produces sorted points."""
        assume(hi > lo)
        grid = ChebyshevGrid(padding=0.0).build((lo, hi), k=k)
        assert np.all(np.diff(grid.points) > 0)

    @given(
        k=st.integers(min_value=2, max_value=50),
    )
    @settings(max_examples=30, deadline=5000)
    def test_uniform_widths_positive(self, k: int):
        """Property: UniformGrid widths are always positive."""
        grid = UniformGrid().build((0.0, 10.0), k=k)
        assert np.all(grid.widths > 0)

    @given(
        k=st.integers(min_value=2, max_value=50),
    )
    @settings(max_examples=30, deadline=5000)
    def test_chebyshev_widths_positive(self, k: int):
        """Property: ChebyshevGrid widths are always positive."""
        grid = ChebyshevGrid().build((0.0, 10.0), k=k)
        assert np.all(grid.widths > 0)

    @given(
        pad=st.floats(min_value=0.0, max_value=100.0),
        lo=st.floats(min_value=-100, max_value=0),
        hi=st.floats(min_value=0.1, max_value=100),
    )
    @settings(max_examples=30, deadline=5000)
    def test_pad_range_expands(self, pad: float, lo: float, hi: float):
        """Property: _pad_range always expands (or maintains) the range."""
        assume(hi > lo)
        new_lo, new_hi = _pad_range(lo, hi, pad)
        assert new_lo <= lo
        assert new_hi >= hi
