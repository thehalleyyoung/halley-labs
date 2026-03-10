"""Unit tests for usability_oracle.statistics.multiple_comparison — FDR/FWER control.

Tests Bonferroni, Holm–Bonferroni, and Benjamini–Hochberg corrections.
"""

from __future__ import annotations

import numpy as np
import pytest

from usability_oracle.statistics.multiple_comparison import (
    BenjaminiHochberg,
    BonferroniCorrection,
    HolmBonferroni,
    correct,
)
from usability_oracle.statistics.types import CorrectionMethod, FDRResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


# ===================================================================
# Bonferroni
# ===================================================================


class TestBonferroni:

    def test_single_test_is_identity(self):
        result = BonferroniCorrection().correct([0.03], alpha=0.05)
        assert isinstance(result, FDRResult)
        np.testing.assert_allclose(result.adjusted_p_values[0], 0.03, atol=1e-10)

    def test_adjusted_p_values_le_1(self):
        p_values = [0.01, 0.04, 0.5, 0.99]
        result = BonferroniCorrection().correct(p_values, alpha=0.05)
        for p in result.adjusted_p_values:
            assert p <= 1.0 + 1e-10

    def test_all_reject_p_zero(self):
        p_values = [0.0, 0.0, 0.0]
        result = BonferroniCorrection().correct(p_values, alpha=0.05)
        for rej in result.rejected:
            assert rej is True

    def test_none_reject_p_one(self):
        p_values = [1.0, 1.0, 1.0]
        result = BonferroniCorrection().correct(p_values, alpha=0.05)
        for rej in result.rejected:
            assert rej is False

    def test_scales_by_number_of_tests(self):
        result = BonferroniCorrection().correct([0.02, 0.03], alpha=0.05)
        np.testing.assert_allclose(result.adjusted_p_values[0], 0.04, atol=1e-10)
        np.testing.assert_allclose(result.adjusted_p_values[1], 0.06, atol=1e-10)

    @pytest.mark.parametrize("n", [1, 5, 10, 50])
    def test_adjusted_ge_original(self, n):
        rng = _rng()
        p_values = rng.uniform(0, 1, n).tolist()
        result = BonferroniCorrection().correct(p_values, alpha=0.05)
        for orig, adj in zip(p_values, result.adjusted_p_values):
            assert adj >= orig - 1e-10


# ===================================================================
# Holm–Bonferroni
# ===================================================================


class TestHolmBonferroni:

    def test_single_test_is_identity(self):
        result = HolmBonferroni().correct([0.03], alpha=0.05)
        np.testing.assert_allclose(result.adjusted_p_values[0], 0.03, atol=1e-10)

    def test_more_powerful_than_bonferroni(self):
        """Holm should reject at least as many tests as Bonferroni."""
        p_values = [0.001, 0.01, 0.04, 0.05, 0.10]
        bonf = BonferroniCorrection().correct(p_values, alpha=0.05)
        holm = HolmBonferroni().correct(p_values, alpha=0.05)
        n_bonf = sum(bonf.rejected)
        n_holm = sum(holm.rejected)
        assert n_holm >= n_bonf

    def test_all_reject_p_zero(self):
        result = HolmBonferroni().correct([0.0, 0.0], alpha=0.05)
        assert all(result.rejected)

    def test_none_reject_p_one(self):
        result = HolmBonferroni().correct([1.0, 1.0], alpha=0.05)
        assert not any(result.rejected)

    def test_adjusted_p_values_le_1(self):
        p_values = [0.01, 0.04, 0.5]
        result = HolmBonferroni().correct(p_values, alpha=0.05)
        for p in result.adjusted_p_values:
            assert p <= 1.0 + 1e-10


# ===================================================================
# Benjamini–Hochberg
# ===================================================================


class TestBenjaminiHochberg:

    def test_controls_fdr_simulation(self):
        """Under global null, FDR should be ≤ α (approximately)."""
        alpha = 0.10
        n_tests = 20
        n_sims = 200
        rng = _rng(0)
        false_discovery_counts = []
        for _ in range(n_sims):
            # All null p-values
            p_values = rng.uniform(0, 1, n_tests).tolist()
            result = BenjaminiHochberg().correct(p_values, alpha=alpha)
            false_discovery_counts.append(sum(result.rejected))
        # Expected FDR = false discoveries / total discoveries ≤ α
        total_discoveries = sum(false_discovery_counts)
        if total_discoveries > 0:
            fdr = sum(false_discovery_counts) / (n_sims * n_tests)
            # Allow generous margin for simulation noise
            assert fdr <= alpha + 0.05

    def test_all_reject_p_zero(self):
        result = BenjaminiHochberg().correct([0.0, 0.0, 0.0], alpha=0.05)
        assert all(result.rejected)

    def test_none_reject_p_one(self):
        result = BenjaminiHochberg().correct([1.0, 1.0, 1.0], alpha=0.05)
        assert not any(result.rejected)

    def test_adjusted_p_values_le_1(self):
        p_values = [0.01, 0.04, 0.5]
        result = BenjaminiHochberg().correct(p_values, alpha=0.05)
        for p in result.adjusted_p_values:
            assert p <= 1.0 + 1e-10

    def test_more_powerful_than_bonferroni(self):
        """BH should reject at least as many tests as Bonferroni."""
        p_values = [0.001, 0.01, 0.04, 0.05, 0.10]
        bonf = BonferroniCorrection().correct(p_values, alpha=0.05)
        bh = BenjaminiHochberg().correct(p_values, alpha=0.05)
        assert sum(bh.rejected) >= sum(bonf.rejected)


# ===================================================================
# Dispatcher
# ===================================================================


class TestCorrectDispatcher:

    @pytest.mark.parametrize("method", [
        CorrectionMethod.BONFERRONI,
        CorrectionMethod.HOLM,
        CorrectionMethod.BENJAMINI_HOCHBERG,
    ])
    def test_dispatch_returns_fdr_result(self, method):
        result = correct([0.01, 0.03, 0.10], method=method, alpha=0.05)
        assert isinstance(result, FDRResult)

    def test_none_correction_passes_through(self):
        p_values = [0.01, 0.05, 0.10]
        result = correct(p_values, method=CorrectionMethod.NONE, alpha=0.05)
        np.testing.assert_allclose(result.adjusted_p_values, p_values, atol=1e-10)
