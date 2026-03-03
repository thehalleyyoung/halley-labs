"""
Comprehensive tests for dp_forge.grid.interpolation.

Tests cover:
  - PiecewiseConstantInterpolator: nearest-neighbour, normalization
  - PiecewiseLinearInterpolator: linear transfer, non-negativity
  - SplineInterpolator: PCHIP smoothness, positivity, fallback
  - MechanismInterpolator protocol conformance
  - Grid transfer invariants: p sums to 1, shape correctness
  - Identity transfer: same grid → same (normalized) mechanism
  - Refinement transfer: coarse→fine shape and normalization
  - Edge cases: boundary grids, very different sizes
  - Input validation: mismatched shapes, wrong dimensions
  - Property-based tests via Hypothesis
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from dp_forge.exceptions import ConfigurationError, InvalidMechanismError
from dp_forge.grid.interpolation import (
    MechanismInterpolator,
    PiecewiseConstantInterpolator,
    PiecewiseLinearInterpolator,
    SplineInterpolator,
    _normalize_rows,
    _validate_inputs,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_mechanism(n: int, k: int, *, seed: int = 42) -> np.ndarray:
    """Build a valid n×k probability table (rows sum to 1, non-negative)."""
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(k), size=n)
    return p


def _make_peaked_mechanism(n: int, k: int, *, seed: int = 42) -> np.ndarray:
    """Mechanism with a pronounced peak at the center."""
    rng = np.random.default_rng(seed)
    p = rng.exponential(0.01, size=(n, k))
    center = k // 2
    for i in range(n):
        p[i, center] += 1.0
    return p / p.sum(axis=1, keepdims=True)


def _make_grid(k: int, lo: float = -5.0, hi: float = 5.0) -> np.ndarray:
    """Sorted uniform grid of k points."""
    return np.linspace(lo, hi, k)


ALL_INTERPOLATORS = [
    PiecewiseConstantInterpolator(),
    PiecewiseLinearInterpolator(),
    SplineInterpolator(),
]


# ═══════════════════════════════════════════════════════════════════════════
# §1  _normalize_rows helper
# ═══════════════════════════════════════════════════════════════════════════


class TestNormalizeRows:
    """Tests for the _normalize_rows helper."""

    def test_already_normalized(self):
        """Already-normalized rows remain unchanged."""
        p = np.array([[0.3, 0.7], [0.5, 0.5]])
        result = _normalize_rows(p)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-14)
        np.testing.assert_allclose(result, p, atol=1e-14)

    def test_clamps_negatives(self):
        """Negative entries are clamped to zero before normalization."""
        p = np.array([[0.5, -0.2, 0.7]])
        result = _normalize_rows(p)
        assert np.all(result >= 0)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-14)

    def test_all_zero_row(self):
        """All-zero row doesn't produce NaN."""
        p = np.array([[0.0, 0.0, 0.0]])
        result = _normalize_rows(p)
        assert not np.any(np.isnan(result))

    def test_large_values(self):
        """Large values are normalized correctly."""
        p = np.array([[1000.0, 2000.0, 3000.0]])
        result = _normalize_rows(p)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-14)


# ═══════════════════════════════════════════════════════════════════════════
# §2  _validate_inputs helper
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateInputs:
    """Tests for the _validate_inputs helper."""

    def test_valid_inputs(self):
        """Correct inputs pass validation."""
        mech = _make_mechanism(3, 10)
        old_grid = _make_grid(10)
        new_grid = _make_grid(20)
        m, og, ng = _validate_inputs(mech, old_grid, new_grid)
        assert m.shape == (3, 10)
        assert len(og) == 10
        assert len(ng) == 20

    def test_mechanism_not_2d(self):
        """1-D mechanism raises InvalidMechanismError."""
        with pytest.raises(InvalidMechanismError, match="2-D"):
            _validate_inputs(np.ones(10), _make_grid(10), _make_grid(5))

    def test_grid_not_1d(self):
        """2-D grid raises ConfigurationError."""
        mech = _make_mechanism(2, 5)
        with pytest.raises(ConfigurationError, match="1-D"):
            _validate_inputs(mech, np.ones((5, 1)), _make_grid(5))

    def test_mechanism_columns_mismatch(self):
        """Mechanism columns != old_grid length raises InvalidMechanismError."""
        mech = _make_mechanism(2, 5)
        with pytest.raises(InvalidMechanismError, match="old_grid length"):
            _validate_inputs(mech, _make_grid(10), _make_grid(5))

    def test_old_grid_too_short(self):
        """old_grid < 2 points raises ConfigurationError."""
        mech = np.array([[1.0]])
        with pytest.raises(ConfigurationError, match="old_grid"):
            _validate_inputs(mech, np.array([0.0]), _make_grid(5))

    def test_new_grid_too_short(self):
        """new_grid < 2 points raises ConfigurationError."""
        mech = _make_mechanism(2, 5)
        with pytest.raises(ConfigurationError, match="new_grid"):
            _validate_inputs(mech, _make_grid(5), np.array([0.0]))


# ═══════════════════════════════════════════════════════════════════════════
# §3  PiecewiseConstantInterpolator
# ═══════════════════════════════════════════════════════════════════════════


class TestPiecewiseConstantInterpolator:
    """Tests for the PiecewiseConstantInterpolator."""

    def test_transfer_shape(self):
        """Transferred mechanism has shape (n, k_new)."""
        interp = PiecewiseConstantInterpolator()
        mech = _make_mechanism(3, 10)
        p_new = interp.transfer(mech, _make_grid(10), _make_grid(20))
        assert p_new.shape == (3, 20)

    def test_normalization(self):
        """Each row sums to 1 after transfer."""
        interp = PiecewiseConstantInterpolator()
        mech = _make_mechanism(4, 15)
        p_new = interp.transfer(mech, _make_grid(15), _make_grid(30))
        np.testing.assert_allclose(p_new.sum(axis=1), 1.0, atol=1e-10)

    def test_nonnegativity(self):
        """All entries are >= 0 after transfer."""
        interp = PiecewiseConstantInterpolator()
        mech = _make_mechanism(3, 10)
        p_new = interp.transfer(mech, _make_grid(10), _make_grid(25))
        assert np.all(p_new >= 0)

    def test_same_grid_preserves_shape(self):
        """Transferring to the same grid preserves probabilities."""
        interp = PiecewiseConstantInterpolator()
        grid = _make_grid(10)
        mech = _make_mechanism(2, 10)
        p_new = interp.transfer(mech, grid, grid.copy())
        # Should be identical (nearest neighbour on same grid)
        np.testing.assert_allclose(p_new, mech, atol=1e-10)

    def test_coarse_to_fine(self):
        """Coarse → fine transfer produces valid mechanism."""
        interp = PiecewiseConstantInterpolator()
        mech = _make_mechanism(3, 5)
        p_new = interp.transfer(mech, _make_grid(5), _make_grid(50))
        assert p_new.shape == (3, 50)
        np.testing.assert_allclose(p_new.sum(axis=1), 1.0, atol=1e-10)

    def test_fine_to_coarse(self):
        """Fine → coarse transfer produces valid mechanism."""
        interp = PiecewiseConstantInterpolator()
        mech = _make_mechanism(3, 50)
        p_new = interp.transfer(mech, _make_grid(50), _make_grid(5))
        assert p_new.shape == (3, 5)
        np.testing.assert_allclose(p_new.sum(axis=1), 1.0, atol=1e-10)

    def test_repr(self):
        r = repr(PiecewiseConstantInterpolator())
        assert "PiecewiseConstant" in r


# ═══════════════════════════════════════════════════════════════════════════
# §4  PiecewiseLinearInterpolator
# ═══════════════════════════════════════════════════════════════════════════


class TestPiecewiseLinearInterpolator:
    """Tests for the PiecewiseLinearInterpolator."""

    def test_transfer_shape(self):
        """Transferred mechanism has correct shape."""
        interp = PiecewiseLinearInterpolator()
        mech = _make_mechanism(3, 10)
        p_new = interp.transfer(mech, _make_grid(10), _make_grid(20))
        assert p_new.shape == (3, 20)

    def test_normalization(self):
        """Each row sums to 1."""
        interp = PiecewiseLinearInterpolator()
        mech = _make_mechanism(4, 15)
        p_new = interp.transfer(mech, _make_grid(15), _make_grid(30))
        np.testing.assert_allclose(p_new.sum(axis=1), 1.0, atol=1e-10)

    def test_nonnegativity(self):
        """All entries are >= 0."""
        interp = PiecewiseLinearInterpolator()
        mech = _make_mechanism(3, 10)
        p_new = interp.transfer(mech, _make_grid(10), _make_grid(25))
        assert np.all(p_new >= 0)

    def test_same_grid_identity(self):
        """Transferring to the same grid returns the same (normalized) mechanism."""
        interp = PiecewiseLinearInterpolator()
        grid = _make_grid(10)
        mech = _make_mechanism(2, 10)
        p_new = interp.transfer(mech, grid, grid.copy())
        np.testing.assert_allclose(p_new, mech, atol=1e-10)

    def test_coarse_to_fine_preserves_mass(self):
        """Coarse → fine: rows sum to 1 (probability conservation)."""
        interp = PiecewiseLinearInterpolator()
        mech = _make_mechanism(3, 8)
        p_new = interp.transfer(mech, _make_grid(8), _make_grid(32))
        np.testing.assert_allclose(p_new.sum(axis=1), 1.0, atol=1e-10)

    def test_interpolation_at_grid_points(self):
        """At old grid points, interpolated values match originals."""
        interp = PiecewiseLinearInterpolator()
        old_grid = _make_grid(10)
        mech = _make_mechanism(2, 10)
        # Use old_grid as the new_grid
        p_new = interp.transfer(mech, old_grid, old_grid)
        np.testing.assert_allclose(p_new, mech, atol=1e-10)

    def test_peaked_mechanism_transfer(self):
        """Peaked mechanism transfers without losing peak structure."""
        interp = PiecewiseLinearInterpolator()
        k_old, k_new = 20, 40
        old_grid = _make_grid(k_old)
        mech = _make_peaked_mechanism(2, k_old)
        p_new = interp.transfer(mech, old_grid, _make_grid(k_new))
        # The peak should still be the maximum
        max_col = np.argmax(p_new[0])
        # Peak should be near the center of the new grid
        assert abs(max_col - k_new // 2) < k_new // 4

    def test_repr(self):
        r = repr(PiecewiseLinearInterpolator())
        assert "PiecewiseLinear" in r


# ═══════════════════════════════════════════════════════════════════════════
# §5  SplineInterpolator
# ═══════════════════════════════════════════════════════════════════════════


class TestSplineInterpolator:
    """Tests for the SplineInterpolator (PCHIP-based)."""

    def test_transfer_shape(self):
        """Transferred mechanism has correct shape."""
        interp = SplineInterpolator()
        mech = _make_mechanism(3, 10)
        p_new = interp.transfer(mech, _make_grid(10), _make_grid(20))
        assert p_new.shape == (3, 20)

    def test_normalization(self):
        """Each row sums to 1."""
        interp = SplineInterpolator()
        mech = _make_mechanism(4, 15)
        p_new = interp.transfer(mech, _make_grid(15), _make_grid(30))
        np.testing.assert_allclose(p_new.sum(axis=1), 1.0, atol=1e-10)

    def test_nonnegativity(self):
        """All entries are >= 0 after clamping."""
        interp = SplineInterpolator()
        mech = _make_mechanism(3, 10)
        p_new = interp.transfer(mech, _make_grid(10), _make_grid(25))
        assert np.all(p_new >= 0)

    def test_same_grid_identity(self):
        """Transferring to the same grid returns the same mechanism."""
        interp = SplineInterpolator()
        grid = _make_grid(10)
        mech = _make_mechanism(2, 10)
        p_new = interp.transfer(mech, grid, grid.copy())
        np.testing.assert_allclose(p_new, mech, atol=1e-8)

    def test_smoothness(self):
        """Spline transfer should produce a smooth output (no step artefacts)."""
        k_old, k_new = 10, 100
        old_grid = _make_grid(k_old)
        new_grid = _make_grid(k_new)
        mech = _make_peaked_mechanism(1, k_old)

        spline = SplineInterpolator()
        p_spline = spline.transfer(mech, old_grid, new_grid)

        # Second-order differences should be small (smooth curve)
        d2 = np.diff(p_spline, n=2, axis=1)
        max_d2 = float(np.max(np.abs(d2)))
        # For a smooth interpolation, max second difference should be bounded
        assert max_d2 < 0.1, f"Max second difference too large: {max_d2}"

    def test_extrapolate_false_boundary_values(self):
        """With extrapolate=False, points outside old grid get boundary values."""
        interp = SplineInterpolator(extrapolate=False)
        old_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        new_grid = np.array([-1.0, 0.0, 2.0, 4.0, 5.0])
        mech = _make_mechanism(2, 5)
        p_new = interp.transfer(mech, old_grid, new_grid)
        # Points at -1.0 and 5.0 are outside → should get boundary values
        assert p_new.shape == (2, 5)
        np.testing.assert_allclose(p_new.sum(axis=1), 1.0, atol=1e-10)

    def test_coarse_to_fine(self):
        """Coarse → fine transfer with spline."""
        interp = SplineInterpolator()
        mech = _make_mechanism(3, 5)
        p_new = interp.transfer(mech, _make_grid(5), _make_grid(50))
        assert p_new.shape == (3, 50)
        np.testing.assert_allclose(p_new.sum(axis=1), 1.0, atol=1e-10)

    def test_repr(self):
        r = repr(SplineInterpolator(extrapolate=True))
        assert "extrapolate=True" in r


# ═══════════════════════════════════════════════════════════════════════════
# §6  MechanismInterpolator protocol
# ═══════════════════════════════════════════════════════════════════════════


class TestInterpolatorProtocol:
    """Test that all interpolators conform to the MechanismInterpolator protocol."""

    @pytest.mark.parametrize("interp", ALL_INTERPOLATORS)
    def test_is_mechanism_interpolator(self, interp):
        """Each interpolator satisfies the protocol."""
        assert isinstance(interp, MechanismInterpolator)

    @pytest.mark.parametrize("interp", ALL_INTERPOLATORS)
    def test_transfer_signature(self, interp):
        """transfer() returns an ndarray of correct shape."""
        mech = _make_mechanism(2, 10)
        result = interp.transfer(mech, _make_grid(10), _make_grid(15))
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 15)


# ═══════════════════════════════════════════════════════════════════════════
# §7  Grid transfer invariants (all interpolators)
# ═══════════════════════════════════════════════════════════════════════════


class TestGridTransferInvariants:
    """Cross-cutting invariants that must hold for all interpolators."""

    @pytest.mark.parametrize("interp", ALL_INTERPOLATORS)
    def test_rows_sum_to_one(self, interp):
        """After transfer, every row sums to 1."""
        mech = _make_mechanism(3, 10)
        p_new = interp.transfer(mech, _make_grid(10), _make_grid(20))
        np.testing.assert_allclose(p_new.sum(axis=1), 1.0, atol=1e-10)

    @pytest.mark.parametrize("interp", ALL_INTERPOLATORS)
    def test_all_nonnegative(self, interp):
        """After transfer, all entries are >= 0."""
        mech = _make_mechanism(3, 10)
        p_new = interp.transfer(mech, _make_grid(10), _make_grid(20))
        assert np.all(p_new >= 0)

    @pytest.mark.parametrize("interp", ALL_INTERPOLATORS)
    def test_correct_shape(self, interp):
        """Transferred mechanism has shape (n, k_new)."""
        n, k_old, k_new = 4, 12, 25
        mech = _make_mechanism(n, k_old)
        p_new = interp.transfer(mech, _make_grid(k_old), _make_grid(k_new))
        assert p_new.shape == (n, k_new)

    @pytest.mark.parametrize("interp", ALL_INTERPOLATORS)
    @pytest.mark.parametrize(
        "k_old,k_new",
        [(5, 50), (10, 100), (20, 200), (50, 5), (100, 10)],
    )
    def test_various_sizes(self, interp, k_old: int, k_new: int):
        """Transfer works for various grid size combinations."""
        n = 3
        mech = _make_mechanism(n, k_old)
        p_new = interp.transfer(mech, _make_grid(k_old), _make_grid(k_new))
        assert p_new.shape == (n, k_new)
        np.testing.assert_allclose(p_new.sum(axis=1), 1.0, atol=1e-10)
        assert np.all(p_new >= 0)


# ═══════════════════════════════════════════════════════════════════════════
# §8  Identity transfer
# ═══════════════════════════════════════════════════════════════════════════


class TestIdentityTransfer:
    """Same grid in and out should give (approximately) the same mechanism."""

    @pytest.mark.parametrize("interp", ALL_INTERPOLATORS)
    @pytest.mark.parametrize("k", [5, 10, 20])
    def test_identity_transfer(self, interp, k: int):
        """Transferring to the same grid preserves the mechanism."""
        grid = _make_grid(k)
        mech = _make_mechanism(2, k)
        p_new = interp.transfer(mech, grid, grid.copy())
        np.testing.assert_allclose(p_new, mech, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# §9  Refinement transfer (coarse → fine)
# ═══════════════════════════════════════════════════════════════════════════


class TestRefinementTransfer:
    """Tests for coarse → fine grid transfer."""

    @pytest.mark.parametrize("interp", ALL_INTERPOLATORS)
    def test_coarse_to_fine_normalization(self, interp):
        """Coarse → fine preserves normalization."""
        mech = _make_mechanism(3, 8)
        p_new = interp.transfer(mech, _make_grid(8), _make_grid(64))
        np.testing.assert_allclose(p_new.sum(axis=1), 1.0, atol=1e-10)

    @pytest.mark.parametrize("interp", ALL_INTERPOLATORS)
    def test_fine_to_coarse_normalization(self, interp):
        """Fine → coarse preserves normalization."""
        mech = _make_mechanism(3, 64)
        p_new = interp.transfer(mech, _make_grid(64), _make_grid(8))
        np.testing.assert_allclose(p_new.sum(axis=1), 1.0, atol=1e-10)

    @pytest.mark.parametrize("interp", ALL_INTERPOLATORS)
    def test_coarse_to_fine_shape_preserved(self, interp):
        """Peak shape should survive coarse → fine transfer."""
        k_old, k_new = 10, 100
        old_grid = _make_grid(k_old)
        new_grid = _make_grid(k_new)
        mech = _make_peaked_mechanism(1, k_old)
        p_new = interp.transfer(mech, old_grid, new_grid)
        # The maximum should be near the center
        peak_idx = np.argmax(p_new[0])
        assert abs(peak_idx - k_new // 2) < k_new // 3


# ═══════════════════════════════════════════════════════════════════════════
# §10  Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestInterpolationEdgeCases:
    """Edge-case tests for interpolation."""

    @pytest.mark.parametrize("interp", ALL_INTERPOLATORS)
    def test_single_row_mechanism(self, interp):
        """Mechanism with a single row (n=1) works."""
        mech = _make_mechanism(1, 10)
        p_new = interp.transfer(mech, _make_grid(10), _make_grid(20))
        assert p_new.shape == (1, 20)
        np.testing.assert_allclose(p_new.sum(axis=1), 1.0, atol=1e-10)

    @pytest.mark.parametrize("interp", ALL_INTERPOLATORS)
    def test_k_old_equals_2(self, interp):
        """Minimum old grid size (2 points)."""
        mech = _make_mechanism(2, 2)
        p_new = interp.transfer(mech, _make_grid(2), _make_grid(10))
        assert p_new.shape == (2, 10)
        np.testing.assert_allclose(p_new.sum(axis=1), 1.0, atol=1e-10)

    @pytest.mark.parametrize("interp", ALL_INTERPOLATORS)
    def test_k_new_equals_2(self, interp):
        """Minimum new grid size (2 points)."""
        mech = _make_mechanism(2, 10)
        p_new = interp.transfer(mech, _make_grid(10), _make_grid(2))
        assert p_new.shape == (2, 2)
        np.testing.assert_allclose(p_new.sum(axis=1), 1.0, atol=1e-10)

    @pytest.mark.parametrize("interp", ALL_INTERPOLATORS)
    def test_overlapping_but_shifted_grids(self, interp):
        """Transfer between grids that overlap but have different boundaries."""
        old_grid = np.linspace(-5.0, 5.0, 20)
        new_grid = np.linspace(-3.0, 8.0, 25)
        mech = _make_mechanism(2, 20)
        p_new = interp.transfer(mech, old_grid, new_grid)
        assert p_new.shape == (2, 25)
        np.testing.assert_allclose(p_new.sum(axis=1), 1.0, atol=1e-10)
        assert np.all(p_new >= 0)

    @pytest.mark.parametrize("interp", ALL_INTERPOLATORS)
    def test_new_grid_subset_of_old(self, interp):
        """New grid is a subset of the old grid."""
        old_grid = np.linspace(-5.0, 5.0, 20)
        new_grid = old_grid[5:15]  # inner 10 points
        mech = _make_mechanism(2, 20)
        p_new = interp.transfer(mech, old_grid, new_grid)
        assert p_new.shape == (2, 10)
        np.testing.assert_allclose(p_new.sum(axis=1), 1.0, atol=1e-10)

    @pytest.mark.parametrize("interp", ALL_INTERPOLATORS)
    def test_uniform_mechanism_stays_uniform_like(self, interp):
        """A uniform mechanism should stay roughly uniform after transfer."""
        k_old, k_new = 10, 20
        mech = np.ones((2, k_old)) / k_old
        p_new = interp.transfer(mech, _make_grid(k_old), _make_grid(k_new))
        np.testing.assert_allclose(p_new.sum(axis=1), 1.0, atol=1e-10)
        # All entries should be similar (not exactly 1/k_new due to boundary effects)
        assert np.std(p_new[0]) < 0.05

    @pytest.mark.parametrize("interp", ALL_INTERPOLATORS)
    def test_very_different_grid_sizes(self, interp):
        """Transfer between very different grid sizes (2 → 200)."""
        mech = _make_mechanism(2, 2)
        p_new = interp.transfer(mech, _make_grid(2), _make_grid(200))
        assert p_new.shape == (2, 200)
        np.testing.assert_allclose(p_new.sum(axis=1), 1.0, atol=1e-10)
        assert np.all(p_new >= 0)


# ═══════════════════════════════════════════════════════════════════════════
# §11  Input validation errors
# ═══════════════════════════════════════════════════════════════════════════


class TestInterpolationInputValidation:
    """Test that invalid inputs produce clear error messages."""

    @pytest.mark.parametrize("interp", ALL_INTERPOLATORS)
    def test_1d_mechanism_raises(self, interp):
        """1-D mechanism raises InvalidMechanismError."""
        with pytest.raises(InvalidMechanismError):
            interp.transfer(np.ones(10), _make_grid(10), _make_grid(5))

    @pytest.mark.parametrize("interp", ALL_INTERPOLATORS)
    def test_3d_mechanism_raises(self, interp):
        """3-D mechanism raises InvalidMechanismError."""
        with pytest.raises(InvalidMechanismError):
            interp.transfer(np.ones((2, 3, 4)), _make_grid(3), _make_grid(5))

    @pytest.mark.parametrize("interp", ALL_INTERPOLATORS)
    def test_mechanism_grid_mismatch(self, interp):
        """Mechanism columns ≠ old_grid length raises."""
        mech = _make_mechanism(2, 5)
        with pytest.raises(InvalidMechanismError):
            interp.transfer(mech, _make_grid(10), _make_grid(5))

    @pytest.mark.parametrize("interp", ALL_INTERPOLATORS)
    def test_old_grid_single_point(self, interp):
        """old_grid with 1 point raises ConfigurationError."""
        mech = np.array([[1.0]])
        with pytest.raises(ConfigurationError):
            interp.transfer(mech, np.array([0.0]), _make_grid(5))

    @pytest.mark.parametrize("interp", ALL_INTERPOLATORS)
    def test_new_grid_single_point(self, interp):
        """new_grid with 1 point raises ConfigurationError."""
        mech = _make_mechanism(2, 5)
        with pytest.raises(ConfigurationError):
            interp.transfer(mech, _make_grid(5), np.array([0.0]))


# ═══════════════════════════════════════════════════════════════════════════
# §12  Property-based tests (Hypothesis)
# ═══════════════════════════════════════════════════════════════════════════


class TestInterpolationProperties:
    """Hypothesis-driven property tests for interpolation."""

    @given(
        n=st.integers(min_value=1, max_value=5),
        k_old=st.integers(min_value=2, max_value=30),
        k_new=st.integers(min_value=2, max_value=30),
    )
    @settings(max_examples=30, deadline=5000)
    def test_piecewise_linear_normalization(self, n: int, k_old: int, k_new: int):
        """Property: PiecewiseLinear always produces normalized rows."""
        interp = PiecewiseLinearInterpolator()
        mech = _make_mechanism(n, k_old)
        old_grid = _make_grid(k_old)
        new_grid = _make_grid(k_new)
        p_new = interp.transfer(mech, old_grid, new_grid)
        np.testing.assert_allclose(p_new.sum(axis=1), 1.0, atol=1e-10)

    @given(
        n=st.integers(min_value=1, max_value=5),
        k_old=st.integers(min_value=2, max_value=30),
        k_new=st.integers(min_value=2, max_value=30),
    )
    @settings(max_examples=30, deadline=5000)
    def test_piecewise_linear_nonnegativity(self, n: int, k_old: int, k_new: int):
        """Property: PiecewiseLinear always produces non-negative entries."""
        interp = PiecewiseLinearInterpolator()
        mech = _make_mechanism(n, k_old)
        p_new = interp.transfer(mech, _make_grid(k_old), _make_grid(k_new))
        assert np.all(p_new >= 0)

    @given(
        n=st.integers(min_value=1, max_value=5),
        k_old=st.integers(min_value=2, max_value=20),
        k_new=st.integers(min_value=2, max_value=20),
    )
    @settings(max_examples=20, deadline=5000)
    def test_spline_normalization(self, n: int, k_old: int, k_new: int):
        """Property: Spline always produces normalized rows."""
        interp = SplineInterpolator()
        mech = _make_mechanism(n, k_old)
        p_new = interp.transfer(mech, _make_grid(k_old), _make_grid(k_new))
        np.testing.assert_allclose(p_new.sum(axis=1), 1.0, atol=1e-10)

    @given(
        n=st.integers(min_value=1, max_value=5),
        k_old=st.integers(min_value=2, max_value=20),
        k_new=st.integers(min_value=2, max_value=20),
    )
    @settings(max_examples=20, deadline=5000)
    def test_spline_nonnegativity(self, n: int, k_old: int, k_new: int):
        """Property: Spline always produces non-negative entries."""
        interp = SplineInterpolator()
        mech = _make_mechanism(n, k_old)
        p_new = interp.transfer(mech, _make_grid(k_old), _make_grid(k_new))
        assert np.all(p_new >= 0)

    @given(
        n=st.integers(min_value=1, max_value=5),
        k=st.integers(min_value=2, max_value=30),
    )
    @settings(max_examples=30, deadline=5000)
    def test_identity_transfer_all_interpolators(self, n: int, k: int):
        """Property: identity transfer preserves mechanism for all interpolators."""
        grid = _make_grid(k)
        mech = _make_mechanism(n, k)
        for interp in ALL_INTERPOLATORS:
            p_new = interp.transfer(mech, grid, grid.copy())
            np.testing.assert_allclose(p_new, mech, atol=1e-4)
