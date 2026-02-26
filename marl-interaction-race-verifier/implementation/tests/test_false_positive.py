"""Tests for marace.race.false_positive_analysis — FP rate modelling."""

import math
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marace.race.false_positive_analysis import (
    FalsePositiveModel,
    ArchitecturalSensitivity,
)


# ======================================================================
# FalsePositiveModel
# ======================================================================

class TestFalsePositiveModel:
    """Test false-positive rate modelling."""

    def test_inflation_increases_with_K(self):
        """Larger K → larger inflation factor."""
        n, T = 4, 5
        phi_1 = FalsePositiveModel.compute_inflation(1.0, n, T)
        phi_2 = FalsePositiveModel.compute_inflation(2.0, n, T)
        phi_3 = FalsePositiveModel.compute_inflation(3.0, n, T)
        assert phi_1 < phi_2 < phi_3

    def test_inflation_K_1_is_1(self):
        """K=1 → no inflation."""
        phi = FalsePositiveModel.compute_inflation(1.0, 4, 10)
        assert phi == pytest.approx(1.0, abs=1e-10)

    def test_inflation_increases_with_dimension(self):
        """Higher dimension → larger inflation for same K."""
        K, T = 2.0, 3
        phi_2d = FalsePositiveModel.compute_inflation(K, 2, T)
        phi_4d = FalsePositiveModel.compute_inflation(K, 4, T)
        phi_8d = FalsePositiveModel.compute_inflation(K, 8, T)
        assert phi_2d < phi_4d < phi_8d

    def test_fp_bound_range(self):
        """FP bound should be in [0, 1]."""
        fp = FalsePositiveModel.false_positive_bound(2.0, 4, 5, 1.0)
        assert 0.0 <= fp <= 1.0

    def test_fp_bound_K_1_is_zero(self):
        """K=1, s=1 → FP bound = 0 (tight bound, fully safe)."""
        fp = FalsePositiveModel.false_positive_bound(1.0, 4, 5, 1.0)
        assert fp == pytest.approx(0.0, abs=1e-10)

    def test_fp_bound_increases_with_K(self):
        K_vals = [1.0, 1.5, 2.0, 3.0]
        fp_vals = [
            FalsePositiveModel.false_positive_bound(K, 4, 3, 1.0)
            for K in K_vals
        ]
        for i in range(len(fp_vals) - 1):
            assert fp_vals[i] <= fp_vals[i + 1]

    def test_instance_fp_bound(self):
        model = FalsePositiveModel(state_dim=4, horizon=5,
                                    true_lipschitz=1.0, bound_lipschitz=2.0)
        assert model.looseness == pytest.approx(2.0)
        fp = model.fp_bound(safe_volume_fraction=1.0)
        assert 0.0 <= fp <= 1.0

    def test_sweep_K(self):
        model = FalsePositiveModel(state_dim=4, horizon=3,
                                    true_lipschitz=1.0, bound_lipschitz=1.0)
        fps = model.sweep_K([1.0, 1.5, 2.0])
        assert len(fps) == 3

    def test_invalid_state_dim(self):
        with pytest.raises(ValueError):
            FalsePositiveModel(state_dim=0, horizon=5)

    def test_invalid_lipschitz(self):
        with pytest.raises(ValueError):
            FalsePositiveModel(state_dim=4, horizon=5,
                                true_lipschitz=2.0, bound_lipschitz=1.0)


# ======================================================================
# ArchitecturalSensitivity
# ======================================================================

class TestArchitecturalSensitivity:
    """Test architecture sensitivity analysis."""

    def test_deeper_network_higher_sensitivity(self):
        """Deeper plain networks should have higher estimated K."""
        sens = ArchitecturalSensitivity(
            state_dim=4, horizon=5,
            base_spectral_norm=1.5, effective_spectral_norm=1.2,
        )
        r_shallow = sens.analyse_architecture(depth=2, width=64, has_skip=False)
        r_deep = sens.analyse_architecture(depth=8, width=64, has_skip=False)
        assert r_deep.estimated_K > r_shallow.estimated_K

    def test_skip_reduces_sensitivity(self):
        """Skip connections should reduce estimated K."""
        sens = ArchitecturalSensitivity(
            state_dim=4, horizon=5,
            base_spectral_norm=1.5, effective_spectral_norm=1.2,
        )
        r_plain = sens.analyse_architecture(depth=6, width=64, has_skip=False)
        r_skip = sens.analyse_architecture(depth=6, width=64, has_skip=True)
        assert r_skip.estimated_K < r_plain.estimated_K

    def test_sweep_depth(self):
        sens = ArchitecturalSensitivity(state_dim=4, horizon=3)
        results = sens.sweep_depth([2, 4, 6])
        assert len(results) == 6  # plain + skip for each depth

    def test_fp_bound_in_result(self):
        sens = ArchitecturalSensitivity(state_dim=4, horizon=3)
        r = sens.analyse_architecture(depth=4, width=64, has_skip=False)
        assert 0.0 <= r.fp_bound <= 1.0
