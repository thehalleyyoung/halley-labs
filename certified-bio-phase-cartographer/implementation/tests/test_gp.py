"""
Tests for GP surrogate with ARD kernel, calibration, and LOO cross-validation.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase_cartographer.gp.surrogate import (
    GPSurrogate, GPPrediction,
    matern52_kernel, matern52_kernel_ard,
    matern52_kernel_matrix, matern52_kernel_matrix_ard,
)
from phase_cartographer.gp.acquisition import (
    expected_improvement, upper_confidence_bound, boundary_uncertainty,
    phase_boundary_score, AcquisitionOptimizer,
)


class TestMatern52Kernel:
    def test_self_kernel(self):
        x = np.array([1.0, 2.0])
        k = matern52_kernel(x, x, 1.0, 1.0)
        assert abs(k - 1.0) < 1e-10

    def test_symmetry(self):
        x1 = np.array([1.0, 2.0])
        x2 = np.array([3.0, 4.0])
        assert abs(matern52_kernel(x1, x2, 1.0, 1.0) -
                   matern52_kernel(x2, x1, 1.0, 1.0)) < 1e-10

    def test_ard_self_kernel(self):
        x = np.array([1.0, 2.0])
        ls = np.array([1.0, 1.0])
        k = matern52_kernel_ard(x, x, ls, 1.0)
        assert abs(k - 1.0) < 1e-10

    def test_ard_vs_isotropic(self):
        """ARD with equal length scales should match isotropic."""
        x1 = np.array([1.0, 2.0])
        x2 = np.array([3.0, 4.0])
        k_iso = matern52_kernel(x1, x2, 1.0, 1.0)
        k_ard = matern52_kernel_ard(x1, x2, np.array([1.0, 1.0]), 1.0)
        assert abs(k_iso - k_ard) < 1e-10

    def test_ard_different_scales(self):
        """ARD with different scales should weight dimensions differently."""
        x1 = np.array([0.0, 0.0])
        x2 = np.array([1.0, 0.0])
        x3 = np.array([0.0, 1.0])
        ls = np.array([0.5, 5.0])  # dim 0 is much more sensitive
        k_dim0 = matern52_kernel_ard(x1, x2, ls, 1.0)
        k_dim1 = matern52_kernel_ard(x1, x3, ls, 1.0)
        # x2 is 1.0 away in dim 0 (short length scale) -> lower kernel
        # x3 is 1.0 away in dim 1 (long length scale) -> higher kernel
        assert k_dim1 > k_dim0


class TestGPSurrogate:
    def _make_data(self):
        """Simple 1D regression dataset."""
        np.random.seed(42)
        X = np.linspace(0, 5, 20).reshape(-1, 1)
        y = np.sin(X.ravel())
        return X, y

    def test_fit_predict(self):
        X, y = self._make_data()
        gp = GPSurrogate(length_scale=1.0, signal_var=1.0)
        gp.fit(X, y)
        pred = gp.predict(np.array([2.5]))
        assert abs(pred.mean - np.sin(2.5)) < 0.5

    def test_ard_fit(self):
        X = np.random.randn(30, 2)
        y = X[:, 0]  # only depends on dim 0
        gp = GPSurrogate(use_ard=True)
        gp.fit(X, y)
        assert gp.use_ard
        assert gp.length_scales is not None

    def test_optimize_length_scale(self):
        X, y = self._make_data()
        gp = GPSurrogate()
        ls = gp.optimize_length_scale(X, y)
        assert ls > 0

    def test_optimize_ard(self):
        X = np.random.randn(30, 3)
        y = X[:, 0] + 0.1 * X[:, 1]
        gp = GPSurrogate()
        scales = gp.optimize_ard_length_scales(X, y)
        assert len(scales) == 3

    def test_calibration_error(self):
        X, y = self._make_data()
        gp = GPSurrogate(length_scale=1.0)
        gp.fit(X[:15], y[:15])
        ece = gp.calibration_error(X[15:], y[15:])
        assert 0.0 <= ece <= 1.0

    def test_loo_cv(self):
        X, y = self._make_data()
        gp = GPSurrogate(length_scale=1.0)
        loo = gp.loo_cross_validation(X, y)
        assert loo >= 0
        assert loo < 10  # should be reasonable

    def test_predict_batch(self):
        X, y = self._make_data()
        gp = GPSurrogate(length_scale=1.0)
        gp.fit(X, y)
        preds = gp.predict_batch(X[:5])
        assert len(preds) == 5


class TestAcquisitionFunctions:
    def test_ei_zero_variance(self):
        pred = GPPrediction(mean=1.0, variance=0.0, std=0.0)
        assert expected_improvement(pred, 0.5) == 0.0

    def test_ucb(self):
        pred = GPPrediction(mean=1.0, variance=1.0, std=1.0)
        ucb = upper_confidence_bound(pred, beta=2.0)
        assert abs(ucb - 3.0) < 1e-10

    def test_boundary_uncertainty(self):
        # Near integer boundary with high uncertainty
        pred = GPPrediction(mean=0.5, variance=1.0, std=1.0)
        bu = boundary_uncertainty(pred)
        assert bu > 0

    def test_phase_boundary_score_no_sensitivity(self):
        pred = GPPrediction(mean=0.5, variance=1.0, std=1.0)
        score = phase_boundary_score(pred, eigenvalue_sensitivity=0.0)
        # Without eigenvalue info, should equal boundary_uncertainty
        expected = boundary_uncertainty(pred)
        assert abs(score - 0.7 * expected) < 1e-10

    def test_phase_boundary_score_with_sensitivity(self):
        pred = GPPrediction(mean=0.5, variance=1.0, std=1.0)
        low = phase_boundary_score(pred, eigenvalue_sensitivity=0.0)
        high = phase_boundary_score(pred, eigenvalue_sensitivity=10.0)
        assert high > low

    def test_phase_boundary_score_inf_sensitivity(self):
        pred = GPPrediction(mean=0.5, variance=1.0, std=1.0)
        score = phase_boundary_score(pred, eigenvalue_sensitivity=float('inf'))
        assert score > 0

class TestAcquisitionOptimizer:
    def test_optimizer_ranking(self):
        X = np.random.randn(20, 2)
        y = (X[:, 0] > 0).astype(float)
        gp = GPSurrogate(length_scale=1.0)
        gp.fit(X, y)
        
        opt = AcquisitionOptimizer(gp, acquisition="boundary_uncertainty")
        query = np.random.randn(5, 2)
        ranked = opt.rank_boxes(query)
        assert len(ranked) == 5
        # Scores should be sorted descending
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)


class TestGPSurrogateAtlas:
    """Tests for atlas-aware GP helpers."""

    def _make_mock_atlas(self):
        """Build a minimal PhaseAtlas with a few cells."""
        from phase_cartographer.atlas.builder import PhaseAtlas
        from phase_cartographer.tiered.certificate import (
            CertifiedCell, EquilibriumCertificate, VerificationTier,
            RegimeType, StabilityType,
        )

        atlas = PhaseAtlas("test", [(0.0, 1.0), (0.0, 1.0)])
        for i in range(5):
            lo = i * 0.2
            hi = lo + 0.2
            eq = EquilibriumCertificate(
                state_enclosure=[(0.5, 1.5)],
                stability=StabilityType.STABLE_NODE,
                eigenvalue_real_parts=[(-2.0, -0.5)],
                krawczyk_contraction=0.3,
                krawczyk_iterations=5,
            )
            cell = CertifiedCell(
                parameter_box=[(lo, hi), (0.0, 1.0)],
                model_name="test",
                n_states=1, n_params=2,
                equilibria=[eq],
                regime=RegimeType.MONOSTABLE,
                tier=VerificationTier.TIER1_IA,
            )
            atlas.add_cell(cell)
        return atlas

    def test_train_from_atlas(self):
        atlas = self._make_mock_atlas()
        gp = GPSurrogate.train_from_atlas(atlas)
        assert gp._fitted
        assert gp.X_train.shape[0] == 5

    def test_train_from_empty_atlas(self):
        from phase_cartographer.atlas.builder import PhaseAtlas
        atlas = PhaseAtlas("empty", [(0.0, 1.0)])
        gp = GPSurrogate.train_from_atlas(atlas)
        assert not gp._fitted

    def test_predict_regime_boundary(self):
        atlas = self._make_mock_atlas()
        gp = GPSurrogate.train_from_atlas(atlas)
        score = gp.predict_regime_boundary(np.array([0.5, 0.5]))
        assert isinstance(score, float)
        assert score >= 0.0


class TestOctreeNewFunctions:
    """Tests for anisotropic_split_box and GP-guided refinement config."""

    def test_anisotropic_split_no_eigenvalues(self):
        from phase_cartographer.refinement.octree import anisotropic_split_box
        box = [(0.0, 2.0), (0.0, 1.0)]
        b1, b2 = anisotropic_split_box(box, None)
        # Should fall back to widest-dimension split (dim 0)
        assert b1[0][1] == 1.0
        assert b2[0][0] == 1.0

    def test_anisotropic_split_with_eigenvalues(self):
        from phase_cartographer.refinement.octree import anisotropic_split_box
        box = [(0.0, 1.0), (0.0, 1.0)]
        # Make dim 1 more sensitive
        eig = [(-2.0, -1.5), (-0.1, 0.05)]
        b1, b2 = anisotropic_split_box(box, eig)
        # dim 1 eigenvalue crosses zero → inf sensitivity → split dim 1
        assert b1[1] != box[1] or b2[1] != box[1]

    def test_gp_guided_config_inherits(self):
        from phase_cartographer.refinement.octree import GPGuidedRefinementConfig
        cfg = GPGuidedRefinementConfig(max_depth=5, gp_warmup_cells=20)
        assert cfg.max_depth == 5
        assert cfg.gp_warmup_cells == 20
        assert cfg.target_coverage == 0.95  # inherited default

    def test_convergence_record(self):
        from phase_cartographer.refinement.octree import ConvergenceRecord
        cr = ConvergenceRecord()
        cr.iteration.append(1)
        cr.coverage.append(0.5)
        assert len(cr.iteration) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
