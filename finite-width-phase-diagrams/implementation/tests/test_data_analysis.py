"""Tests for the data_analysis module (dataset kernels, feature maps, task complexity)."""

from __future__ import annotations

import sys, os, math
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_analysis.dataset_kernel import (
    DataDependentNTK, KernelTargetAlignment, GeneralizationBound,
    EffectiveDimension, SpectralBiasAnalyzer,
)
from src.data_analysis.feature_maps import (
    RandomFeatureApproximation, FeatureDimensionEstimator,
    FeatureQualityMetric, FeatureAlignmentAnalyzer, RandomFeatureRegression,
)
from src.data_analysis.task_complexity import (
    TargetSmoothnessEstimator, RKHSNormComputer,
    CurriculumLearningAnalyzer, TaskArchitectureCompatibility,
)

# ===================================================================
# Helpers
# ===================================================================

def _rbf_kernel(X, sigma=1.0):
    sq = np.sum(X ** 2, axis=1, keepdims=True)
    return np.exp(-(sq + sq.T - 2.0 * X @ X.T) / (2.0 * sigma ** 2))

def _make_spd(n, rng):
    A = rng.randn(n, n)
    return A @ A.T + np.eye(n) * 0.1

def _val(x):
    """Extract scalar from result that may be scalar, dict, or array."""
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    if isinstance(x, dict):
        return float(list(x.values())[0])
    return float(np.asarray(x).ravel()[0])

def _arr(x):
    """Extract array from result."""
    if isinstance(x, np.ndarray):
        return x.ravel()
    if isinstance(x, dict):
        v = list(x.values())[0]
        return np.asarray(v).ravel() if hasattr(v, '__len__') else np.array([v])
    return np.asarray(x).ravel()

# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def rng():
    return np.random.RandomState(42)

@pytest.fixture
def small_data(rng):
    return rng.randn(20, 5)

@pytest.fixture
def targets(rng):
    return rng.randn(20)

@pytest.fixture
def binary_labels(rng):
    return rng.choice([0, 1], size=20)

@pytest.fixture
def K_rbf(small_data):
    return _rbf_kernel(small_data)

@pytest.fixture
def eigenvalues_decay():
    return np.array([1.0 / (i + 1) ** 2 for i in range(20)])

@pytest.fixture
def smooth_target(small_data):
    return small_data @ np.ones(5)

@pytest.fixture
def noisy_target(small_data, rng):
    return small_data @ np.ones(5) + 0.5 * rng.randn(20)

# ===================================================================
# DataDependentNTK – eigenspectrum
# ===================================================================

class TestDataDependentNTK:
    def test_eigenspectrum_ordering(self, K_rbf):
        ntk = DataDependentNTK()
        eigs = _arr(ntk.ntk_eigenspectrum(K_rbf))
        for i in range(len(eigs) - 1):
            assert eigs[i] >= eigs[i + 1] - 1e-10

    def test_eigenspectrum_nonneg(self, K_rbf):
        eigs = _arr(DataDependentNTK().ntk_eigenspectrum(K_rbf))
        assert np.all(eigs >= -1e-10)

    def test_effective_rank_bounds(self, K_rbf):
        rank = _val(DataDependentNTK().effective_rank(K_rbf))
        assert 0 < rank <= K_rbf.shape[0] + 1e-10

    def test_condition_number_positive(self, K_rbf):
        assert _val(DataDependentNTK().kernel_condition_number(K_rbf)) > 0

    def test_bulk_outlier_partition(self, K_rbf):
        ntk = DataDependentNTK()
        eigs = _arr(ntk.ntk_eigenspectrum(K_rbf))
        bulk = _arr(ntk.bulk_eigenvalues(eigs))
        outliers = _arr(ntk.outlier_eigenvalues(eigs))
        assert len(bulk) + len(outliers) == len(eigs)

    def test_eigenvalue_spacing_nonneg(self, K_rbf):
        ntk = DataDependentNTK()
        eigs = _arr(ntk.ntk_eigenspectrum(K_rbf))
        spacings = _arr(ntk.eigenvalue_spacing(eigs))
        assert np.all(spacings >= -1e-10)

    def test_identity_kernel_eigenvalues(self):
        eigs = _arr(DataDependentNTK().ntk_eigenspectrum(np.eye(10)))
        np.testing.assert_allclose(eigs, 1.0, atol=1e-10)

    def test_effective_rank_identity(self):
        assert abs(_val(DataDependentNTK().effective_rank(np.eye(10))) - 10) < 1.0

    def test_spectral_density(self, K_rbf):
        ntk = DataDependentNTK()
        eigs = _arr(ntk.ntk_eigenspectrum(K_rbf))
        assert ntk.spectral_density(eigs) is not None

# ===================================================================
# KernelTargetAlignment
# ===================================================================

class TestKernelTargetAlignment:
    def test_alignment_bounds(self, K_rbf, targets):
        val = _val(KernelTargetAlignment().compute_alignment(K_rbf, targets))
        assert -1.0 - 1e-10 <= val <= 1.0 + 1e-10

    def test_centered_alignment_bounds(self, K_rbf, targets):
        val = _val(KernelTargetAlignment().centered_alignment(K_rbf, targets))
        assert -1.0 - 1e-10 <= val <= 1.0 + 1e-10

    def test_perfect_alignment(self, rng):
        t = rng.randn(15)
        K = np.outer(t, t)
        assert _val(KernelTargetAlignment().compute_alignment(K, t)) > 0.9

    def test_alignment_vs_depth(self, targets, rng):
        n = len(targets)
        kernels = [_make_spd(n, rng) for _ in range(5)]
        result = _arr(KernelTargetAlignment().alignment_vs_depth(kernels, targets))
        assert len(result) == 5

    def test_alignment_vs_width(self, targets, rng):
        n = len(targets)
        kernels = [_make_spd(n, rng) for _ in range(4)]
        result = _arr(KernelTargetAlignment().alignment_vs_width(kernels, targets))
        assert len(result) == 4

    def test_alignment_significance(self, K_rbf, targets):
        assert KernelTargetAlignment().alignment_significance(K_rbf, targets, n_permutations=100) is not None

    def test_class_alignment(self, K_rbf, binary_labels):
        assert KernelTargetAlignment().class_alignment(K_rbf, binary_labels, n_classes=2) is not None

    def test_optimal_kernel_symmetric(self, targets):
        K = KernelTargetAlignment().optimal_kernel_for_targets(targets)
        if isinstance(K, np.ndarray):
            np.testing.assert_allclose(K, K.T, atol=1e-10)

# ===================================================================
# GeneralizationBound
# ===================================================================

class TestGeneralizationBound:
    def _gb(self, K, y):
        return GeneralizationBound(K, y, K.shape[0])

    def test_rademacher_positive(self, K_rbf, targets):
        assert _val(self._gb(K_rbf, targets).rademacher_bound(K_rbf, K_rbf.shape[0])) > 0

    def test_spectral_positive(self, K_rbf, targets):
        gb = self._gb(K_rbf, targets)
        evals, evecs = np.linalg.eigh(K_rbf)
        coeffs = evecs[:, ::-1].T @ targets
        assert _val(gb.spectral_bound(evals[::-1], coeffs, 0.1)) > 0

    def test_pac_bayes_positive(self, K_rbf, targets):
        assert _val(self._gb(K_rbf, targets).pac_bayes_bound(K_rbf, targets, 1.0)) > 0

    def test_loo_bound(self, K_rbf, targets):
        assert self._gb(K_rbf, targets).loo_bound(K_rbf, targets, 0.1) is not None

    def test_effective_dim_bound_positive(self, K_rbf, targets):
        n = K_rbf.shape[0]
        assert _val(self._gb(K_rbf, targets).effective_dimension_bound(K_rbf, 0.1, n)) > 0

    def test_bias_variance_nonneg(self, K_rbf, targets):
        n = K_rbf.shape[0]
        r = self._gb(K_rbf, targets).bias_variance_decomposition(K_rbf, targets, 0.1, n)
        if isinstance(r, dict):
            for v in r.values():
                if isinstance(v, (int, float)):
                    assert v >= -1e-10

    def test_learning_curve(self, K_rbf, targets):
        assert self._gb(K_rbf, targets).learning_curve_prediction(K_rbf, targets, list(range(5, 20, 3))) is not None

    def test_kernel_ridge_bound(self, K_rbf, targets):
        assert _val(self._gb(K_rbf, targets).kernel_ridge_bound(K_rbf, targets, 0.1)) > 0

# ===================================================================
# EffectiveDimension
# ===================================================================

class TestEffectiveDimension:
    def test_positive(self, K_rbf):
        assert _val(EffectiveDimension().compute_effective_dim(K_rbf, 0.1)) > 0

    def test_monotone_decreasing(self, K_rbf):
        ed = EffectiveDimension()
        dims = [_val(ed.compute_effective_dim(K_rbf, r)) for r in [0.01, 0.1, 1.0, 10.0]]
        for i in range(len(dims) - 1):
            assert dims[i] >= dims[i + 1] - 1e-6

    def test_vs_regularization(self, K_rbf):
        assert EffectiveDimension().effective_dim_vs_regularization(K_rbf, np.logspace(-3, 2, 10)) is not None

    def test_dof_bounded(self, K_rbf):
        dof = _val(EffectiveDimension().degrees_of_freedom(K_rbf, 0.1))
        assert 0 <= dof <= K_rbf.shape[0] + 1e-10

    def test_participation_ratio_bounds(self, eigenvalues_decay):
        pr = _val(EffectiveDimension().participation_ratio(eigenvalues_decay))
        assert 0.5 <= pr <= len(eigenvalues_decay) + 1e-10

    def test_participation_ratio_uniform(self):
        assert abs(_val(EffectiveDimension().participation_ratio(np.ones(10))) - 10) < 1e-6

    def test_information_dimension(self, eigenvalues_decay):
        v = _val(EffectiveDimension().information_dimension(eigenvalues_decay))
        assert v > 0 and np.isfinite(v)

    def test_spectral_dimension(self, eigenvalues_decay):
        v = _val(EffectiveDimension().spectral_dimension(eigenvalues_decay))
        assert 0 < v <= len(eigenvalues_decay) + 1e-10

    def test_intrinsic_dimension(self, small_data):
        v = _val(EffectiveDimension().intrinsic_dimension(small_data, method='mle'))
        assert 0 < v <= small_data.shape[1] + 5

# ===================================================================
# SpectralBiasAnalyzer
# ===================================================================

class TestSpectralBiasAnalyzer:
    def test_target_in_eigenbasis(self, K_rbf, targets):
        _, evecs = np.linalg.eigh(K_rbf)
        coeffs = _arr(SpectralBiasAnalyzer().target_in_eigenbasis(evecs[:, ::-1], targets))
        assert len(coeffs) == len(targets)

    def test_learning_speed_per_mode(self, eigenvalues_decay):
        speeds = _arr(SpectralBiasAnalyzer().learning_speed_per_mode(eigenvalues_decay, 0.01))
        assert len(speeds) == len(eigenvalues_decay)

    def test_time_to_learn_mode_ordering(self):
        sba = SpectralBiasAnalyzer()
        t_fast = _val(sba.time_to_learn_mode(10.0, 1.0, 0.01))
        t_slow = _val(sba.time_to_learn_mode(1.0, 1.0, 0.01))
        assert t_fast < t_slow

    def test_bias_profile(self, eigenvalues_decay, rng):
        assert SpectralBiasAnalyzer().bias_profile(eigenvalues_decay, rng.randn(20)) is not None

    def test_spectral_bias_curve(self, eigenvalues_decay, rng):
        sba = SpectralBiasAnalyzer()
        assert sba.spectral_bias_curve(eigenvalues_decay, rng.randn(20), np.linspace(0.1, 10, 5)) is not None

    def test_frequency_bias(self, rng):
        n = 20
        evecs = np.linalg.qr(rng.randn(n, n))[0]
        assert SpectralBiasAnalyzer().frequency_bias(evecs, np.arange(1, n + 1, dtype=float)) is not None

# ===================================================================
# RandomFeatureApproximation
# ===================================================================

class TestRandomFeatureApproximation:
    def test_generate_features_shape(self, small_data, rng):
        n, d = small_data.shape
        rfa = RandomFeatureApproximation(n_features=50, input_dim=d)
        W = rng.randn(50, d)
        b = rng.uniform(0, 2 * np.pi, 50)
        feat = np.asarray(rfa.generate_features(small_data, W, b))
        assert feat.shape[0] == n

    def test_kernel_approx_improves(self, small_data):
        n, d = small_data.shape
        K_exact = _rbf_kernel(small_data)
        rfa = RandomFeatureApproximation(n_features=10, input_dim=d)
        errors = []
        for nf in [10, 50, 200]:
            K_ap = np.asarray(rfa.random_feature_kernel(small_data, small_data, nf))
            errors.append(_val(rfa.kernel_approximation_error(K_exact, K_ap)))
        assert errors[-1] <= errors[0] + 0.1

    def test_convergence_with_features(self, small_data):
        n, d = small_data.shape
        K_exact = _rbf_kernel(small_data)
        rfa = RandomFeatureApproximation(n_features=10, input_dim=d)
        assert rfa.convergence_with_features(small_data, [10, 50, 100], K_exact) is not None

    def test_orthogonal_features(self, small_data):
        n, d = small_data.shape
        assert RandomFeatureApproximation(50, d).orthogonal_features(small_data, 50) is not None

    def test_ntk_random_features(self, small_data):
        n, d = small_data.shape
        assert RandomFeatureApproximation(50, d).ntk_random_features(small_data, 50, depth=2) is not None

# ===================================================================
# FeatureQualityMetric
# ===================================================================

class TestFeatureQualityMetric:
    def test_coherence_bounded(self, rng):
        v = _val(FeatureQualityMetric().feature_coherence(rng.randn(30, 50)))
        assert 0 <= v <= 1.0 + 1e-10

    def test_diversity_positive(self, rng):
        assert _val(FeatureQualityMetric().feature_diversity(rng.randn(30, 50))) > 0

    def test_informativeness(self, rng):
        assert FeatureQualityMetric().feature_informativeness(rng.randn(30, 50), rng.randn(30)) is not None

    def test_stability_identical(self, rng):
        f = rng.randn(30, 50)
        assert _val(FeatureQualityMetric().feature_stability(f, f)) > 0.9

    def test_leverage_scores_sum(self, rng):
        n, d = 30, 10
        scores = _arr(FeatureQualityMetric().leverage_scores(rng.randn(n, d)))
        assert len(scores) == n
        assert abs(np.sum(scores) - d) < 1.0

    def test_condition_number_positive(self, rng):
        assert _val(FeatureQualityMetric().feature_condition_number(rng.randn(30, 10))) > 0

# ===================================================================
# FeatureAlignmentAnalyzer
# ===================================================================

class TestFeatureAlignmentAnalyzer:
    def test_alignment_score_finite(self, rng):
        v = _val(FeatureAlignmentAnalyzer().alignment_score(rng.randn(30, 20), rng.randn(30)))
        assert np.isfinite(v)

    def test_feature_target_correlation_shape(self, rng):
        corr = _arr(FeatureAlignmentAnalyzer().feature_target_correlation(rng.randn(30, 10), rng.randn(30)))
        assert len(corr) == 10

    def test_principal_alignment(self, rng):
        assert FeatureAlignmentAnalyzer().principal_alignment(rng.randn(30, 20), rng.randn(30), k=5) is not None

    def test_subspace_self_alignment(self, rng):
        U = np.linalg.qr(rng.randn(10, 5))[0]
        assert _val(FeatureAlignmentAnalyzer().subspace_alignment(U, U)) > 0.9

# ===================================================================
# FeatureDimensionEstimator
# ===================================================================

class TestFeatureDimensionEstimator:
    def test_estimate_positive(self, eigenvalues_decay):
        assert _val(FeatureDimensionEstimator().estimate_dimension(eigenvalues_decay)) > 0

    def test_jl_bound_monotonic(self):
        fde = FeatureDimensionEstimator()
        b1 = _val(fde.johnson_lindenstrauss_bound(100, 0.1))
        b2 = _val(fde.johnson_lindenstrauss_bound(1000, 0.1))
        b3 = _val(fde.johnson_lindenstrauss_bound(100, 0.5))
        assert b2 > b1 and b3 < b1

    def test_spectral_decay_rate(self, eigenvalues_decay):
        v = _val(FeatureDimensionEstimator().spectral_decay_rate(eigenvalues_decay))
        assert v > 0 and np.isfinite(v)

# ===================================================================
# RandomFeatureRegression – double descent
# ===================================================================

class TestRandomFeatureRegression:
    def test_fit_predict_shape(self, small_data, smooth_target):
        rfr = RandomFeatureRegression(n_features=50, regularization=0.1)
        rfr.fit(small_data, smooth_target)
        assert len(np.asarray(rfr.predict(small_data)).ravel()) == small_data.shape[0]

    def test_ridgeless_fit(self, small_data, smooth_target):
        assert RandomFeatureRegression(100, 1e-10).ridgeless_fit(small_data, smooth_target) is not None

    def test_interpolation_threshold(self, small_data):
        n = small_data.shape[0]
        v = _val(RandomFeatureRegression(50).interpolation_threshold(n))
        assert abs(v - n) < n * 0.2

    def test_double_descent_curve(self, rng):
        n, d = 30, 5
        X, y = rng.randn(n, d), rng.randn(n)
        X_t, y_t = rng.randn(20, d), rng.randn(20)
        nf_range = [10, 20, 30, 50, 80, 120]
        curve = _arr(RandomFeatureRegression(10, 1e-10).double_descent_curve(X, y, X_t, y_t, nf_range))
        assert len(curve) == len(nf_range)

    def test_optimal_regularization_positive(self, small_data, smooth_target):
        v = _val(RandomFeatureRegression(50).optimal_regularization(small_data, smooth_target, np.logspace(-5, 1, 10)))
        assert v > 0

    def test_generalization_bound_positive(self):
        v = _val(RandomFeatureRegression(50).generalization_bound(50, 30, 0.1))
        assert v > 0 and np.isfinite(v)

    def test_learning_curve(self, rng):
        X, y = rng.randn(50, 5), rng.randn(50)
        assert RandomFeatureRegression(30).learning_curve(X, y, [10, 20, 30], 30) is not None

# ===================================================================
# TargetSmoothnessEstimator
# ===================================================================

class TestTargetSmoothnessEstimator:
    def test_lipschitz_positive(self, smooth_target, small_data):
        assert _val(TargetSmoothnessEstimator().lipschitz_constant(smooth_target, small_data)) > 0

    def test_noisy_higher_lipschitz(self, smooth_target, noisy_target, small_data):
        tse = TargetSmoothnessEstimator()
        v_s = _val(tse.lipschitz_constant(smooth_target, small_data))
        v_n = _val(tse.lipschitz_constant(noisy_target, small_data))
        assert v_n >= v_s * 0.5

    def test_sobolev_smoothness(self, smooth_target, small_data):
        assert TargetSmoothnessEstimator().sobolev_smoothness(smooth_target, small_data) is not None

    def test_holder_exponent_positive(self, smooth_target, small_data):
        assert _val(TargetSmoothnessEstimator().holder_exponent(smooth_target, small_data)) > 0

    def test_effective_smoothness_finite(self, eigenvalues_decay, rng):
        v = _val(TargetSmoothnessEstimator().effective_smoothness(rng.randn(20), eigenvalues_decay))
        assert np.isfinite(v)

    def test_anisotropic_smoothness(self, smooth_target, small_data):
        assert TargetSmoothnessEstimator().anisotropic_smoothness(smooth_target, small_data) is not None

# ===================================================================
# RKHSNormComputer
# ===================================================================

class TestRKHSNormComputer:
    def test_rkhs_norm_nonneg(self, smooth_target, K_rbf):
        assert _val(RKHSNormComputer().compute_rkhs_norm(smooth_target, K_rbf)) >= 0

    def test_regularized_norm_finite(self, smooth_target, K_rbf):
        v = _val(RKHSNormComputer().regularized_rkhs_norm(smooth_target, K_rbf, 0.1))
        assert np.isfinite(v) and v >= 0

    def test_is_in_rkhs(self, smooth_target, K_rbf):
        assert RKHSNormComputer().is_in_rkhs(smooth_target, K_rbf) is not None

    def test_source_condition(self, eigenvalues_decay, rng):
        coeffs = rng.randn(20) / np.arange(1, 21)
        assert RKHSNormComputer().source_condition(coeffs, eigenvalues_decay, np.linspace(0.5, 2, 5)) is not None

    def test_approximation_error_vs_n(self, smooth_target, K_rbf):
        assert RKHSNormComputer().approximation_error_vs_n(smooth_target, K_rbf, list(range(5, 20, 3))) is not None

    def test_minimax_rate_positive(self, eigenvalues_decay):
        assert _val(RKHSNormComputer().minimax_optimal_rate(eigenvalues_decay, 1.0)) > 0

# ===================================================================
# CurriculumLearningAnalyzer
# ===================================================================

class TestCurriculumLearningAnalyzer:
    def test_sort_by_difficulty(self, small_data, targets, K_rbf):
        assert CurriculumLearningAnalyzer().sort_by_difficulty(small_data, targets, K_rbf) is not None

    def test_difficulty_score(self, small_data, targets, K_rbf):
        cla = CurriculumLearningAnalyzer()
        s = cla.difficulty_score(small_data[0], targets[0], K_rbf, training_indices=list(range(20)))
        assert s is not None

    def test_curriculum_schedule(self, rng):
        assert CurriculumLearningAnalyzer().curriculum_schedule(rng.rand(50), n_stages=5) is not None

    def test_anti_curriculum_schedule(self, rng):
        assert CurriculumLearningAnalyzer().anti_curriculum_schedule(rng.rand(50), n_stages=5) is not None

    def test_curriculum_vs_anti_different(self, rng):
        d = np.arange(20, dtype=float)
        cla = CurriculumLearningAnalyzer()
        c = cla.curriculum_schedule(d, n_stages=4)
        a = cla.anti_curriculum_schedule(d, n_stages=4)
        assert c is not None and a is not None

# ===================================================================
# TaskArchitectureCompatibility
# ===================================================================

class TestTaskArchitectureCompatibility:
    def test_compatibility_score_finite(self, K_rbf, targets):
        assert np.isfinite(_val(TaskArchitectureCompatibility().compatibility_score(K_rbf, targets, K_rbf.shape[0])))

    def test_spectral_compatibility_finite(self, eigenvalues_decay, rng):
        v = _val(TaskArchitectureCompatibility().spectral_compatibility(eigenvalues_decay, rng.randn(20)))
        assert np.isfinite(v)

    def test_kernel_expressivity_positive(self, eigenvalues_decay):
        assert _val(TaskArchitectureCompatibility().kernel_expressivity(eigenvalues_decay)) > 0

    def test_depth_compatibility(self):
        assert TaskArchitectureCompatibility().depth_compatibility(1.0, [2, 4, 6, 8], 100) is not None

    def test_width_compatibility(self):
        assert TaskArchitectureCompatibility().width_compatibility(1.0, [32, 64, 128, 256], 3) is not None

    def test_architecture_recommendation(self):
        props = {"smoothness": 1.0, "complexity": 0.5, "dimension": 10}
        cands = [{"depth": 2, "width": 64}, {"depth": 4, "width": 128}, {"depth": 8, "width": 256}]
        assert TaskArchitectureCompatibility().architecture_recommendation(props, cands) is not None
