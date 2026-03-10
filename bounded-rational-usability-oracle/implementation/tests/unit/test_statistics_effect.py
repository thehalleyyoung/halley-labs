"""Unit tests for usability_oracle.statistics.effect_size — Effect-size estimation.

Tests Cohen's d, Hedges' g, Cliff's delta, and common-language effect size.
"""

from __future__ import annotations

import numpy as np
import pytest

from usability_oracle.statistics.effect_size import (
    EffectSizeCalculator,
    cliff_delta,
    cohens_d,
    common_language_effect_size,
    hedges_g,
)
from usability_oracle.statistics.types import EffectSize, EffectSizeType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


# ===================================================================
# Cohen's d
# ===================================================================


class TestCohensD:

    def test_identical_groups_d_zero(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = cohens_d(data, data)
        np.testing.assert_allclose(result.value, 0.0, atol=1e-10)

    def test_sign_depends_on_direction(self):
        rng = _rng()
        a = rng.normal(0, 1, 100)
        b = rng.normal(2, 1, 100)
        d_ab = cohens_d(a, b)
        d_ba = cohens_d(b, a)
        # Signs should be opposite
        assert d_ab.value * d_ba.value < 0 or (d_ab.value == 0 and d_ba.value == 0)

    def test_large_effect_detected(self):
        rng = _rng()
        a = rng.normal(0, 1, 100)
        b = rng.normal(3, 1, 100)
        result = cohens_d(a, b)
        assert abs(result.value) > 0.8  # "large" effect

    def test_returns_effect_size_type(self):
        rng = _rng()
        a = rng.normal(0, 1, 30)
        b = rng.normal(1, 1, 30)
        result = cohens_d(a, b)
        assert isinstance(result, EffectSize)
        assert result.measure == EffectSizeType.COHENS_D

    def test_ci_present(self):
        rng = _rng()
        a = rng.normal(0, 1, 30)
        b = rng.normal(1, 1, 30)
        result = cohens_d(a, b)
        assert result.ci.lower <= result.value <= result.ci.upper


# ===================================================================
# Hedges' g
# ===================================================================


class TestHedgesG:

    def test_approx_cohens_d_for_large_n(self):
        rng = _rng()
        a = rng.normal(0, 1, 500)
        b = rng.normal(1, 1, 500)
        d = cohens_d(a, b)
        g = hedges_g(a, b)
        np.testing.assert_allclose(g.value, d.value, atol=0.05)

    def test_hedges_g_smaller_than_d(self):
        """Hedges' g applies a correction factor J < 1, so |g| ≤ |d|."""
        rng = _rng()
        a = rng.normal(0, 1, 20)
        b = rng.normal(1, 1, 20)
        d = cohens_d(a, b)
        g = hedges_g(a, b)
        assert abs(g.value) <= abs(d.value) + 1e-10

    def test_returns_hedges_g_type(self):
        rng = _rng()
        a = rng.normal(0, 1, 30)
        b = rng.normal(1, 1, 30)
        result = hedges_g(a, b)
        assert result.measure == EffectSizeType.HEDGES_G


# ===================================================================
# Cliff's delta
# ===================================================================


class TestCliffDelta:

    def test_in_range(self):
        rng = _rng()
        a = rng.normal(0, 1, 50)
        b = rng.normal(1, 1, 50)
        result = cliff_delta(a, b)
        assert -1.0 <= result.value <= 1.0

    def test_identical_data_is_zero(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = cliff_delta(data, data)
        np.testing.assert_allclose(result.value, 0.0, atol=1e-10)

    def test_perfect_dominance(self):
        a = np.array([10.0, 11.0, 12.0])
        b = np.array([1.0, 2.0, 3.0])
        result = cliff_delta(a, b)
        # Cliff's delta sign depends on implementation (y - x or x - y)
        assert abs(result.value) > 0.9

    def test_returns_cliffs_delta_type(self):
        rng = _rng()
        a = rng.normal(0, 1, 30)
        b = rng.normal(1, 1, 30)
        result = cliff_delta(a, b)
        assert result.measure == EffectSizeType.CLIFFS_DELTA


# ===================================================================
# Common-language effect size (CLES)
# ===================================================================


class TestCLES:

    def test_in_range(self):
        rng = _rng()
        a = rng.normal(0, 1, 50)
        b = rng.normal(1, 1, 50)
        result = common_language_effect_size(a, b)
        assert 0.0 <= result.value <= 1.0

    def test_identical_data_near_half(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = common_language_effect_size(data, data)
        np.testing.assert_allclose(result.value, 0.5, atol=0.01)

    def test_returns_common_language_type(self):
        rng = _rng()
        a = rng.normal(0, 1, 30)
        b = rng.normal(1, 1, 30)
        result = common_language_effect_size(a, b)
        assert result.measure == EffectSizeType.COMMON_LANGUAGE


# ===================================================================
# EffectSizeCalculator
# ===================================================================


class TestEffectSizeCalculator:

    @pytest.mark.parametrize("measure", [
        EffectSizeType.COHENS_D,
        EffectSizeType.HEDGES_G,
        EffectSizeType.CLIFFS_DELTA,
        EffectSizeType.COMMON_LANGUAGE,
    ])
    def test_estimate_returns_effect_size(self, measure):
        rng = _rng()
        a = rng.normal(0, 1, 40)
        b = rng.normal(0.5, 1, 40)
        calc = EffectSizeCalculator()
        result = calc.estimate(a, b, measure=measure)
        assert isinstance(result, EffectSize)
        assert result.measure == measure
