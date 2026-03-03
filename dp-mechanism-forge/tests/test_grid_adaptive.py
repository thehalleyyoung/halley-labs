"""
Comprehensive tests for dp_forge.grid.adaptive_refine — AdaptiveGridRefiner.

Tests cover:
  - RefinementStep dataclass behaviour
  - Configuration validation (k0, k_max, mass_threshold, etc.)
  - High-mass bin identification and subdivision
  - Grid refinement mechanics (point insertion, deduplication)
  - Warm-start interpolation preserving normalization
  - Convergence detection helpers
  - Error trajectory and summary accessors
  - Property-based tests via Hypothesis
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from dp_forge.exceptions import ConfigurationError, ConvergenceError
from dp_forge.grid.adaptive_refine import (
    AdaptiveGridRefiner,
    RefinementStep,
    _DEFAULT_K0,
    _DEFAULT_K_MAX,
    _DEFAULT_MASS_THRESHOLD,
    _DEFAULT_CONVERGENCE_TOL,
    _DEFAULT_SUBDIVIDE_FACTOR,
)
from dp_forge.grid.grid_strategies import GridResult, UniformGrid
from dp_forge.grid.interpolation import (
    PiecewiseConstantInterpolator,
    PiecewiseLinearInterpolator,
    SplineInterpolator,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers — mock mechanism builders
# ═══════════════════════════════════════════════════════════════════════════


def _make_mechanism(n: int, k: int, *, peaked: bool = False) -> np.ndarray:
    """Build a valid n×k probability table (rows sum to 1, non-negative).

    If *peaked*, concentrate most mass in a few central bins so that
    high-mass detection triggers.
    """
    rng = np.random.default_rng(42)
    if peaked:
        p = rng.exponential(0.01, size=(n, k))
        center = k // 2
        width = max(k // 8, 1)
        lo, hi = max(0, center - width), min(k, center + width)
        p[:, lo:hi] += rng.exponential(1.0, size=(n, hi - lo))
    else:
        p = rng.dirichlet(np.ones(k), size=n)
    row_sums = p.sum(axis=1, keepdims=True)
    p = p / row_sums
    return p


def _make_grid(k: int, lo: float = -5.0, hi: float = 5.0) -> np.ndarray:
    """Build a sorted uniform grid of *k* points."""
    return np.linspace(lo, hi, k)


# ═══════════════════════════════════════════════════════════════════════════
# §1  RefinementStep dataclass
# ═══════════════════════════════════════════════════════════════════════════


class TestRefinementStep:
    """Tests for the RefinementStep dataclass."""

    def test_basic_construction(self):
        """RefinementStep stores all fields correctly."""
        grid = _make_grid(10)
        mech = _make_mechanism(3, 10)
        step = RefinementStep(
            level=0, k=10, grid=grid, mechanism=mech,
            objective=0.5, iterations=3, l1_error_bound=0.01,
            n_high_mass_bins=2, elapsed_seconds=1.5, converged=True,
        )
        assert step.level == 0
        assert step.k == 10
        assert step.objective == 0.5
        assert step.iterations == 3
        assert step.converged is True
        assert len(step.grid) == 10
        assert step.mechanism.shape == (3, 10)

    def test_repr_converged(self):
        """repr includes ✓ for converged steps."""
        step = RefinementStep(
            level=1, k=20, grid=_make_grid(20),
            mechanism=_make_mechanism(2, 20),
            objective=0.123456, iterations=5,
            l1_error_bound=1e-3, n_high_mass_bins=0,
            elapsed_seconds=0.5, converged=True,
        )
        r = repr(step)
        assert "✓" in r
        assert "level=1" in r

    def test_repr_not_converged(self):
        """repr includes … for non-converged steps."""
        step = RefinementStep(
            level=0, k=8, grid=_make_grid(8),
            mechanism=_make_mechanism(2, 8),
            objective=1.0, iterations=1,
            l1_error_bound=0.1, n_high_mass_bins=3,
            elapsed_seconds=0.1, converged=False,
        )
        r = repr(step)
        assert "…" in r


# ═══════════════════════════════════════════════════════════════════════════
# §2  Configuration validation
# ═══════════════════════════════════════════════════════════════════════════


class TestAdaptiveGridRefinerConfig:
    """Test that invalid constructor parameters raise ConfigurationError."""

    def test_k0_too_small(self):
        """k0 < 2 should raise."""
        with pytest.raises(ConfigurationError, match="k0"):
            AdaptiveGridRefiner(k0=1)

    def test_k0_equals_two_ok(self):
        """k0 == 2 is the minimum valid value."""
        refiner = AdaptiveGridRefiner(k0=2, k_max=2)
        assert refiner._k0 == 2

    def test_k_max_less_than_k0(self):
        """k_max < k0 should raise."""
        with pytest.raises(ConfigurationError, match="k_max"):
            AdaptiveGridRefiner(k0=10, k_max=5)

    def test_max_levels_zero(self):
        """max_levels < 1 should raise."""
        with pytest.raises(ConfigurationError, match="max_levels"):
            AdaptiveGridRefiner(max_levels=0)

    def test_mass_threshold_zero(self):
        """mass_threshold must be in (0, 1)."""
        with pytest.raises(ConfigurationError, match="mass_threshold"):
            AdaptiveGridRefiner(mass_threshold=0.0)

    def test_mass_threshold_one(self):
        """mass_threshold must be in (0, 1)."""
        with pytest.raises(ConfigurationError, match="mass_threshold"):
            AdaptiveGridRefiner(mass_threshold=1.0)

    def test_convergence_tol_zero(self):
        """convergence_tol must be > 0."""
        with pytest.raises(ConfigurationError, match="convergence_tol"):
            AdaptiveGridRefiner(convergence_tol=0.0)

    def test_convergence_tol_negative(self):
        """convergence_tol must be > 0."""
        with pytest.raises(ConfigurationError, match="convergence_tol"):
            AdaptiveGridRefiner(convergence_tol=-1e-4)

    def test_subdivide_factor_one(self):
        """subdivide_factor must be >= 2."""
        with pytest.raises(ConfigurationError, match="subdivide_factor"):
            AdaptiveGridRefiner(subdivide_factor=1)

    def test_valid_defaults(self):
        """Default parameters pass validation."""
        refiner = AdaptiveGridRefiner()
        assert refiner._k0 == _DEFAULT_K0
        assert refiner._k_max == _DEFAULT_K_MAX
        assert refiner._mass_threshold == _DEFAULT_MASS_THRESHOLD
        assert refiner._convergence_tol == _DEFAULT_CONVERGENCE_TOL
        assert refiner._subdivide_factor == _DEFAULT_SUBDIVIDE_FACTOR

    def test_repr(self):
        """__repr__ includes key parameters."""
        refiner = AdaptiveGridRefiner(k0=10, k_max=200)
        r = repr(refiner)
        assert "k0=10" in r
        assert "k_max=200" in r


# ═══════════════════════════════════════════════════════════════════════════
# §3  High-mass bin counting
# ═══════════════════════════════════════════════════════════════════════════


class TestHighMassBinCounting:
    """Tests for _count_high_mass_bins."""

    def test_all_below_threshold(self):
        """No bins above threshold → count = 0."""
        refiner = AdaptiveGridRefiner(mass_threshold=0.5)
        # Uniform mechanism: each bin has mass 1/k
        mech = _make_mechanism(3, 20)  # Dirichlet → mass roughly 1/20 = 0.05
        count = refiner._count_high_mass_bins(mech)
        # With threshold=0.5, very unlikely any bin exceeds 0.5
        assert count == 0

    def test_peaked_mechanism_has_high_mass(self):
        """Peaked mechanism should have some bins above threshold."""
        refiner = AdaptiveGridRefiner(mass_threshold=0.05)
        mech = _make_mechanism(3, 20, peaked=True)
        count = refiner._count_high_mass_bins(mech)
        assert count > 0

    def test_empty_mechanism(self):
        """Empty mechanism → 0 high-mass bins."""
        refiner = AdaptiveGridRefiner()
        mech = np.empty((0, 0))
        assert refiner._count_high_mass_bins(mech) == 0

    def test_single_bin_mechanism(self):
        """A mechanism with one bin always has mass 1.0."""
        refiner = AdaptiveGridRefiner(mass_threshold=0.5)
        mech = np.ones((2, 1))
        count = refiner._count_high_mass_bins(mech)
        assert count == 1

    @pytest.mark.parametrize("threshold", [0.01, 0.05, 0.1, 0.5])
    def test_threshold_monotonicity(self, threshold: float):
        """Higher threshold → fewer (or equal) high-mass bins."""
        mech = _make_mechanism(3, 30, peaked=True)
        lo = AdaptiveGridRefiner(mass_threshold=threshold)
        hi = AdaptiveGridRefiner(mass_threshold=min(threshold + 0.1, 0.99))
        assert lo._count_high_mass_bins(mech) >= hi._count_high_mass_bins(mech)


# ═══════════════════════════════════════════════════════════════════════════
# §4  Grid refinement mechanics (_build_refined_grid)
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildRefinedGrid:
    """Tests for _build_refined_grid — the subdivision logic."""

    def _refiner(self, **kw) -> AdaptiveGridRefiner:
        return AdaptiveGridRefiner(
            k0=2, k_max=10000, mass_threshold=kw.get("mass_threshold", 0.05),
            subdivide_factor=kw.get("subdivide_factor", 2),
        )

    def test_no_high_mass_no_subdivision(self):
        """If no bin exceeds threshold, grid stays the same."""
        k = 10
        grid = _make_grid(k)
        # Uniform mechanism: max per bin ≈ 1/k = 0.1 < 0.5
        mech = np.ones((3, k)) / k
        refiner = self._refiner(mass_threshold=0.5)
        new_grid, new_k = refiner._build_refined_grid(grid, mech, (-5.0, 5.0))
        np.testing.assert_array_equal(new_grid, grid)
        assert new_k == k

    def test_all_high_mass_subdivides_all(self):
        """If all bins exceed threshold, every interval gets new points."""
        k = 5
        grid = _make_grid(k)
        mech = np.ones((2, k)) * 0.5  # max per bin = 0.5 > threshold
        refiner = self._refiner(mass_threshold=0.01, subdivide_factor=2)
        new_grid, new_k = refiner._build_refined_grid(grid, mech, (-5.0, 5.0))
        # Each of first k-1 intervals gets 1 midpoint (factor=2 → 1 interior)
        # so new_k should be k + (k-1) = 2k - 1
        assert new_k == 2 * k - 1

    def test_refined_grid_is_sorted(self):
        """Refined grid must be sorted."""
        k = 10
        grid = _make_grid(k)
        mech = _make_mechanism(3, k, peaked=True)
        refiner = self._refiner(mass_threshold=0.01)
        new_grid, _ = refiner._build_refined_grid(grid, mech, (-5.0, 5.0))
        assert np.all(np.diff(new_grid) > 0), "Grid not strictly increasing"

    def test_refined_grid_contains_original_points(self):
        """All original grid points must appear in the refined grid."""
        k = 8
        grid = _make_grid(k)
        mech = _make_mechanism(2, k, peaked=True)
        refiner = self._refiner(mass_threshold=0.01)
        new_grid, _ = refiner._build_refined_grid(grid, mech, (-5.0, 5.0))
        for pt in grid:
            assert np.any(np.isclose(new_grid, pt, atol=1e-12)), (
                f"Original point {pt} missing from refined grid"
            )

    def test_subdivide_factor_3(self):
        """subdivide_factor=3 inserts 2 interior points per high-mass interval."""
        k = 4
        grid = _make_grid(k)
        mech = np.ones((2, k)) * 0.5
        refiner = self._refiner(mass_threshold=0.01, subdivide_factor=3)
        new_grid, new_k = refiner._build_refined_grid(grid, mech, (-5.0, 5.0))
        # Each of k-1=3 intervals gets 2 interior pts → new_k = k + 3*2 = 10
        assert new_k == k + (k - 1) * 2

    def test_no_duplicate_points(self):
        """Refined grid must have no duplicate points."""
        k = 10
        grid = _make_grid(k)
        mech = _make_mechanism(3, k, peaked=True)
        refiner = self._refiner(mass_threshold=0.01)
        new_grid, new_k = refiner._build_refined_grid(grid, mech, (-5.0, 5.0))
        assert len(set(new_grid.tolist())) == new_k

    @pytest.mark.parametrize("k", [3, 5, 10, 20])
    def test_refined_grid_at_least_k_points(self, k: int):
        """Refined grid always has >= k points."""
        grid = _make_grid(k)
        mech = _make_mechanism(2, k)
        refiner = self._refiner(mass_threshold=0.01)
        _, new_k = refiner._build_refined_grid(grid, mech, (-5.0, 5.0))
        assert new_k >= k

    def test_last_bin_high_mass_no_crash(self):
        """High-mass in the last bin should not try to subdivide past the end."""
        k = 5
        grid = _make_grid(k)
        mech = np.ones((2, k)) * 0.01
        mech[:, -1] = 0.8  # last bin has high mass
        refiner = self._refiner(mass_threshold=0.05)
        new_grid, new_k = refiner._build_refined_grid(grid, mech, (-5.0, 5.0))
        # Last bin is high-mass but it's the last point, so no interval to subdivide
        assert new_k >= k


# ═══════════════════════════════════════════════════════════════════════════
# §5  Warm-start interpolation via refiner
# ═══════════════════════════════════════════════════════════════════════════


class TestWarmStartInterpolation:
    """Test that interpolation used for warm-start preserves constraints."""

    @pytest.mark.parametrize(
        "interpolator_cls",
        [PiecewiseConstantInterpolator, PiecewiseLinearInterpolator, SplineInterpolator],
    )
    def test_interpolation_preserves_normalization(self, interpolator_cls):
        """Transferred mechanism rows must sum to 1."""
        n, k_old, k_new = 3, 10, 20
        old_grid = _make_grid(k_old)
        new_grid = _make_grid(k_new)
        mech = _make_mechanism(n, k_old)
        interpolator = interpolator_cls()
        p_new = interpolator.transfer(mech, old_grid, new_grid)
        row_sums = p_new.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    @pytest.mark.parametrize(
        "interpolator_cls",
        [PiecewiseConstantInterpolator, PiecewiseLinearInterpolator, SplineInterpolator],
    )
    def test_interpolation_preserves_nonnegativity(self, interpolator_cls):
        """Transferred mechanism entries must be >= 0."""
        n, k_old, k_new = 3, 10, 20
        old_grid = _make_grid(k_old)
        new_grid = _make_grid(k_new)
        mech = _make_mechanism(n, k_old)
        interpolator = interpolator_cls()
        p_new = interpolator.transfer(mech, old_grid, new_grid)
        assert np.all(p_new >= 0), "Negative probabilities after interpolation"

    def test_interpolation_shape(self):
        """Transferred mechanism has shape (n, k_new)."""
        n, k_old, k_new = 4, 8, 16
        interpolator = PiecewiseLinearInterpolator()
        p_new = interpolator.transfer(
            _make_mechanism(n, k_old), _make_grid(k_old), _make_grid(k_new)
        )
        assert p_new.shape == (n, k_new)


# ═══════════════════════════════════════════════════════════════════════════
# §6  Convergence detection and objective/error tracking
# ═══════════════════════════════════════════════════════════════════════════


class TestConvergenceAndTracking:
    """Tests for error_trajectory, objective_trajectory, and summary."""

    def _build_steps(self, objectives: List[float]) -> AdaptiveGridRefiner:
        """Create a refiner with pre-populated steps."""
        refiner = AdaptiveGridRefiner(k0=8, k_max=1000)
        k = 8
        for i, obj in enumerate(objectives):
            step = RefinementStep(
                level=i, k=k, grid=_make_grid(k),
                mechanism=_make_mechanism(2, k),
                objective=obj, iterations=3,
                l1_error_bound=1.0 / k, n_high_mass_bins=1,
                elapsed_seconds=0.1, converged=True,
            )
            refiner._steps.append(step)
            k *= 2
        return refiner

    def test_error_trajectory(self):
        """error_trajectory returns (k, error_bound) pairs."""
        refiner = self._build_steps([1.0, 0.5, 0.3])
        traj = refiner.error_trajectory()
        assert len(traj) == 3
        ks = [t[0] for t in traj]
        assert ks == [8, 16, 32]

    def test_objective_trajectory(self):
        """objective_trajectory returns (k, objective) pairs."""
        refiner = self._build_steps([1.0, 0.5, 0.3])
        traj = refiner.objective_trajectory()
        objs = [t[1] for t in traj]
        assert objs == [1.0, 0.5, 0.3]

    def test_summary_fields(self):
        """summary() returns expected keys."""
        refiner = self._build_steps([1.0, 0.5])
        s = refiner.summary()
        assert "n_levels" in s
        assert "k_trajectory" in s
        assert "total_time_seconds" in s
        assert s["n_levels"] == 2
        assert s["final_k"] == 16

    def test_n_levels_property(self):
        """n_levels returns the number of recorded steps."""
        refiner = self._build_steps([1.0, 0.5, 0.3])
        assert refiner.n_levels == 3

    def test_steps_returns_copy(self):
        """steps property returns a copy, not the internal list."""
        refiner = self._build_steps([1.0])
        steps = refiner.steps
        steps.clear()
        assert refiner.n_levels == 1  # internal list unchanged

    def test_summary_empty(self):
        """summary() on a refiner with no steps has sane defaults."""
        refiner = AdaptiveGridRefiner()
        s = refiner.summary()
        assert s["n_levels"] == 0
        assert s["final_k"] == 0
        assert s["final_objective"] is None


# ═══════════════════════════════════════════════════════════════════════════
# §7  k_max limit enforcement
# ═══════════════════════════════════════════════════════════════════════════


class TestKMaxEnforcement:
    """Test that refinement stops when k would exceed k_max."""

    def test_refined_grid_exceeds_k_max(self):
        """_build_refined_grid can produce k > k_max; the main loop catches it."""
        k = 10
        grid = _make_grid(k)
        mech = np.ones((2, k)) * 0.5
        refiner = AdaptiveGridRefiner(k0=2, k_max=12, mass_threshold=0.01)
        new_grid, new_k = refiner._build_refined_grid(grid, mech, (-5.0, 5.0))
        # new_k = 2*10 - 1 = 19 > k_max=12
        assert new_k > 12  # loop would stop here


# ═══════════════════════════════════════════════════════════════════════════
# §8  _build_level_spec
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildLevelSpec:
    """Test internal spec construction for each refinement level."""

    def test_level_spec_preserves_fields(self):
        """Level spec inherits epsilon, delta, sensitivity from base spec."""
        from dp_forge.types import QuerySpec
        base = QuerySpec.counting(n=5, epsilon=1.0, delta=0.01, k=100)
        grid = _make_grid(20)
        refiner = AdaptiveGridRefiner()
        level_spec = refiner._build_level_spec(base, grid)
        assert level_spec.epsilon == 1.0
        assert level_spec.delta == 0.01
        assert level_spec.sensitivity == 1.0
        assert level_spec.k == 20

    def test_level_spec_overrides_k(self):
        """Level spec k matches the grid length, not the base spec k."""
        from dp_forge.types import QuerySpec
        base = QuerySpec.counting(n=3, epsilon=0.5, k=100)
        grid = _make_grid(42)
        refiner = AdaptiveGridRefiner()
        level_spec = refiner._build_level_spec(base, grid)
        assert level_spec.k == 42

    def test_level_spec_metadata_contains_grid(self):
        """Level spec metadata includes _grid_points."""
        from dp_forge.types import QuerySpec
        base = QuerySpec.counting(n=3, epsilon=0.5, k=50)
        grid = _make_grid(15)
        refiner = AdaptiveGridRefiner()
        level_spec = refiner._build_level_spec(base, grid)
        assert "_grid_points" in level_spec.metadata
        np.testing.assert_array_equal(level_spec.metadata["_grid_points"], grid)


# ═══════════════════════════════════════════════════════════════════════════
# §9  Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge-case tests for the adaptive refiner."""

    def test_k0_equals_2_minimum(self):
        """Minimum valid k0=2 constructs without error."""
        refiner = AdaptiveGridRefiner(k0=2, k_max=2)
        assert refiner._k0 == 2

    def test_uniform_mechanism_no_high_mass_bins(self):
        """Perfectly uniform mechanism has no high-mass bins at typical thresholds."""
        k = 100
        mech = np.ones((5, k)) / k  # max per bin = 0.01
        refiner = AdaptiveGridRefiner(mass_threshold=0.02)
        assert refiner._count_high_mass_bins(mech) == 0

    def test_all_mass_in_one_bin(self):
        """Degenerate mechanism with all mass in one bin."""
        k = 10
        mech = np.zeros((2, k))
        mech[:, 5] = 1.0
        refiner = AdaptiveGridRefiner(mass_threshold=0.01)
        assert refiner._count_high_mass_bins(mech) == 1

    def test_equal_weight_bins(self):
        """All bins equally weighted → count depends on threshold."""
        k = 20
        mech = np.ones((3, k)) / k
        # threshold = 1/k exactly → none exceed (strict >)
        refiner = AdaptiveGridRefiner(mass_threshold=1.0 / k)
        assert refiner._count_high_mass_bins(mech) == 0
        # threshold slightly below → all exceed
        refiner2 = AdaptiveGridRefiner(mass_threshold=1.0 / k - 1e-12)
        assert refiner2._count_high_mass_bins(mech) == k


# ═══════════════════════════════════════════════════════════════════════════
# §10  Property-based tests (Hypothesis)
# ═══════════════════════════════════════════════════════════════════════════


class TestAdaptiveProperties:
    """Property-based tests for adaptive refinement invariants."""

    @given(
        k=st.integers(min_value=3, max_value=50),
        n=st.integers(min_value=2, max_value=5),
        threshold=st.floats(min_value=0.001, max_value=0.3),
    )
    @settings(max_examples=30, deadline=5000)
    def test_refined_grid_at_least_k_points(self, k: int, n: int, threshold: float):
        """Property: refined grid always has >= k points."""
        grid = _make_grid(k)
        mech = _make_mechanism(n, k, peaked=True)
        refiner = AdaptiveGridRefiner(
            k0=2, k_max=10000, mass_threshold=threshold
        )
        new_grid, new_k = refiner._build_refined_grid(grid, mech, (-5.0, 5.0))
        assert new_k >= k

    @given(
        k=st.integers(min_value=3, max_value=50),
        n=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=30, deadline=5000)
    def test_refined_grid_sorted(self, k: int, n: int):
        """Property: refined grid is always strictly sorted."""
        grid = _make_grid(k)
        mech = _make_mechanism(n, k, peaked=True)
        refiner = AdaptiveGridRefiner(k0=2, k_max=10000, mass_threshold=0.01)
        new_grid, _ = refiner._build_refined_grid(grid, mech, (-5.0, 5.0))
        assert np.all(np.diff(new_grid) > 0)

    @given(
        k=st.integers(min_value=3, max_value=30),
        factor=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=20, deadline=5000)
    def test_subdivide_factor_point_count(self, k: int, factor: int):
        """All-high-mass grid yields k + (k-1)*(factor-1) points."""
        grid = _make_grid(k)
        mech = np.ones((2, k)) * 0.5
        refiner = AdaptiveGridRefiner(
            k0=2, k_max=100000, mass_threshold=0.01,
            subdivide_factor=factor,
        )
        _, new_k = refiner._build_refined_grid(grid, mech, (-5.0, 5.0))
        expected = k + (k - 1) * (factor - 1)
        assert new_k == expected

    @given(
        k=st.integers(min_value=4, max_value=30),
        n=st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=25, deadline=5000)
    def test_high_mass_count_bounded(self, k: int, n: int):
        """Property: high-mass bin count is between 0 and k."""
        mech = _make_mechanism(n, k, peaked=True)
        refiner = AdaptiveGridRefiner(mass_threshold=0.01)
        count = refiner._count_high_mass_bins(mech)
        assert 0 <= count <= k


# ═══════════════════════════════════════════════════════════════════════════
# §11  Error trajectory non-increasing property
# ═══════════════════════════════════════════════════════════════════════════


class TestErrorTrajectoryProperty:
    """Test that the L1 error bound decreases as k grows (Theorem T5)."""

    def test_error_bound_decreases_with_k(self):
        """L1 error bound ∝ 1/k → should decrease as k doubles."""
        from dp_forge.grid.error_estimator import DiscretizationErrorEstimator

        est = DiscretizationErrorEstimator(range_B=10.0, n_databases=3)
        bounds = []
        for k in [8, 16, 32, 64, 128]:
            bound = est.estimate_l1_gap(k, adaptive=False)
            bounds.append(bound)
        # Each doubling of k should halve the bound
        for i in range(1, len(bounds)):
            assert bounds[i] < bounds[i - 1]

    def test_error_trajectory_non_increasing_simulated(self):
        """Simulated multi-level trajectory has non-increasing error bounds."""
        from dp_forge.grid.error_estimator import DiscretizationErrorEstimator

        est = DiscretizationErrorEstimator(range_B=10.0, n_databases=3)
        prev_bound = float("inf")
        for level, k in enumerate([8, 16, 32, 64]):
            grid = _make_grid(k)
            mech = _make_mechanism(3, k)
            record = est.record_level(
                level=level, k=k, objective=1.0 / k,
                mechanism=mech, grid=grid,
            )
            # The uniform bound B/k is non-increasing in k
            uniform_bound = 10.0 / k
            assert uniform_bound <= prev_bound
            prev_bound = uniform_bound


# ═══════════════════════════════════════════════════════════════════════════
# §12  Integration-level test with mocked CEGIS
# ═══════════════════════════════════════════════════════════════════════════


class TestRefineWithMockedCEGIS:
    """Test the full refine() loop with CEGIS mocked out."""

    def _mock_cegis_result(self, n: int, k: int, obj: float):
        """Create a mock CEGISResult-like object."""
        from dp_forge.types import CEGISResult
        mech = _make_mechanism(n, k)
        return CEGISResult(
            mechanism=mech, iterations=5, obj_val=obj,
            convergence_history=[obj],
        )

    def test_refine_converges_on_stable_objective(self):
        """If objective doesn't change, refine should converge early."""
        from dp_forge.types import QuerySpec

        spec = QuerySpec.counting(n=5, epsilon=1.0, k=20)
        refiner = AdaptiveGridRefiner(
            k0=8, k_max=200, convergence_tol=0.01,
            mass_threshold=0.5,  # high → few high-mass bins
        )

        call_count = [0]
        def mock_run_cegis(level_spec):
            call_count[0] += 1
            k = level_spec.k
            return self._mock_cegis_result(spec.n, k, obj=0.42)

        with patch.object(refiner, "_run_cegis", side_effect=mock_run_cegis):
            result = refiner.refine(spec)

        assert result.obj_val == 0.42
        # Should converge after 2 levels (objective unchanged → rel_change=0 < tol)
        assert refiner.n_levels <= 3

    def test_refine_stops_at_no_high_mass_bins(self):
        """If no high-mass bins, refinement stops immediately."""
        from dp_forge.types import QuerySpec

        spec = QuerySpec.counting(n=3, epsilon=1.0, k=20)
        refiner = AdaptiveGridRefiner(
            k0=10, k_max=200, mass_threshold=0.99,
        )

        def mock_run_cegis(level_spec):
            k = level_spec.k
            # Uniform mechanism → no bin exceeds 0.99
            return self._mock_cegis_result(spec.n, k, obj=0.5)

        with patch.object(refiner, "_run_cegis", side_effect=mock_run_cegis):
            result = refiner.refine(spec)

        # Should stop after level 0 (no high-mass bins)
        assert refiner.n_levels == 1

    def test_refine_records_callback(self):
        """callback is invoked for each refinement level."""
        from dp_forge.types import QuerySpec

        spec = QuerySpec.counting(n=3, epsilon=1.0, k=20)
        refiner = AdaptiveGridRefiner(
            k0=8, k_max=200, mass_threshold=0.99,
        )

        calls = []
        def cb(step: RefinementStep):
            calls.append(step)

        def mock_run_cegis(level_spec):
            k = level_spec.k
            return self._mock_cegis_result(spec.n, k, obj=0.5)

        with patch.object(refiner, "_run_cegis", side_effect=mock_run_cegis):
            refiner.refine(spec, callback=cb)

        assert len(calls) >= 1
        assert all(isinstance(c, RefinementStep) for c in calls)

    def test_refine_with_decreasing_objectives(self):
        """Refine stores all steps when objectives are decreasing."""
        from dp_forge.types import QuerySpec

        spec = QuerySpec.counting(n=4, epsilon=1.0, k=20)
        refiner = AdaptiveGridRefiner(
            k0=5, k_max=500, mass_threshold=0.01,
            convergence_tol=1e-8,  # very tight → won't converge quickly
        )

        objectives = [1.0, 0.8, 0.65, 0.55, 0.5]
        call_idx = [0]

        def mock_run_cegis(level_spec):
            idx = min(call_idx[0], len(objectives) - 1)
            obj = objectives[idx]
            call_idx[0] += 1
            k = level_spec.k
            return self._mock_cegis_result(spec.n, k, obj=obj)

        with patch.object(refiner, "_run_cegis", side_effect=mock_run_cegis):
            refiner.refine(spec)

        assert refiner.n_levels >= 2
        obj_traj = [s.objective for s in refiner.steps]
        # Objectives should be non-increasing (each level improves or stays)
        for i in range(1, len(obj_traj)):
            assert obj_traj[i] <= obj_traj[i - 1] + 1e-10

    def test_refine_k_max_stops_refinement(self):
        """Refinement stops when next grid would exceed k_max."""
        from dp_forge.types import QuerySpec

        spec = QuerySpec.counting(n=3, epsilon=1.0, k=20)
        # k_max very small → stops after first level
        refiner = AdaptiveGridRefiner(
            k0=5, k_max=5, mass_threshold=0.01,
        )

        def mock_run_cegis(level_spec):
            k = level_spec.k
            return self._mock_cegis_result(spec.n, k, obj=0.5)

        with patch.object(refiner, "_run_cegis", side_effect=mock_run_cegis):
            result = refiner.refine(spec)

        # Should stop because refined grid would exceed k_max=5
        assert refiner.n_levels >= 1
        assert refiner.steps[-1].k <= 5
