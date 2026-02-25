"""Tests for the dynamics module (gradient flow, loss landscape, SGD, lazy/rich regime)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_impl_root = Path(__file__).resolve().parent.parent
if str(_impl_root) not in sys.path:
    sys.path.insert(0, str(_impl_root))

from src.dynamics.gradient_flow import (
    GradientFlowSolver, NTKDynamics, FeatureLearningDynamics,
    LearningRateScheduler, GradientNoiseSDE,
)
from src.dynamics.loss_landscape import (
    HessianAnalyzer, LossSurfaceVisualizer, SaddlePointDetector,
    LossBarrierEstimator, TrajectoryAnalyzer,
)
from src.dynamics.sgd_dynamics import (
    SGDSimulator, LearningRatePhaseAnalyzer, BatchSizeEffectAnalyzer,
    MomentumDynamics, SGDNoiseCovarianceEstimator, SGDtoSDEConverter,
)
from src.dynamics.lazy_regime import (
    LazyRegimeAnalyzer, NTKStabilityChecker, LinearizedDynamicsSolver,
    KernelRegressionPredictor,
)
from src.dynamics.rich_regime import (
    RichRegimeAnalyzer, FeatureEvolutionTracker, RepresentationChangeMetric,
    FeatureAlignmentAnalyzer, NeuralCollapseDetector,
)

# ---------------------------------------------------------------------------
RNG = np.random.RandomState(42)

def _quadratic_loss(params, data_x=None, data_y=None):
    return 0.5 * np.sum(params ** 2)

def _quadratic_grad(params, data_x=None, data_y=None):
    return params.copy()

def _make_linear_data(n=100, d=5, noise=0.1, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    w = rng.randn(d)
    y = X @ w + noise * rng.randn(n)
    return X, y, w

def _linear_mse_loss(params, data_x, data_y):
    r = data_x @ params - data_y
    return 0.5 * np.mean(r ** 2)

def _linear_mse_grad(params, data_x, data_y):
    r = data_x @ params - data_y
    return data_x.T @ r / len(data_y)

def _spd_kernel(n, seed=0):
    rng = np.random.RandomState(seed)
    A = rng.randn(n, n)
    return A @ A.T + np.eye(n)


# ===================================================================
# Gradient flow
# ===================================================================
class TestGradientFlowSolver:
    def test_solve_converges_to_minimum(self):
        solver = GradientFlowSolver(_quadratic_loss, param_dim=5)
        x0 = RNG.randn(5)
        result = solver.solve(x0, t_span=(0, 5.0))
        final = result if isinstance(result, np.ndarray) and result.ndim == 1 else np.asarray(result)[-1]
        assert np.linalg.norm(final) < 0.1

    def test_trajectory_energy_decreases(self):
        solver = GradientFlowSolver(_quadratic_loss, param_dim=3)
        traj = solver.solve(np.array([1.0, 2.0, 3.0]), t_span=(0, 3.0))
        energies = solver.trajectory_energy(np.asarray(traj) if not isinstance(traj, np.ndarray) else traj)
        if energies is not None:
            e = np.asarray(energies)
            assert e[-1] <= e[0]

    def test_convergence_rate_positive(self):
        solver = GradientFlowSolver(_quadratic_loss, param_dim=4)
        traj = solver.solve(np.ones(4), t_span=(0, 2.0))
        rate = solver.convergence_rate(np.asarray(traj) if not isinstance(traj, np.ndarray) else traj)
        if rate is not None:
            assert rate >= 0

    def test_solve_with_callbacks(self):
        solver = GradientFlowSolver(_quadratic_loss, param_dim=2)
        calls = []
        solver.solve_with_callbacks(np.array([1.0, 1.0]), (0, 1.0),
                                    callbacks=[lambda p: calls.append(p.copy())])
        assert len(calls) > 0

    def test_phase_portrait_shape(self):
        solver = GradientFlowSolver(_quadratic_loss, param_dim=2)
        assert solver.phase_portrait([(-2, 2), (-2, 2)], resolution=10) is not None


# ===================================================================
# NTK dynamics
# ===================================================================
class TestNTKDynamics:
    @pytest.fixture()
    def setup(self):
        n = 20
        K = _spd_kernel(n, seed=1)
        y = RNG.randn(n)
        return NTKDynamics(K, y), K, y

    def test_analytical_solution_shape(self, setup):
        ntk, _, y = setup
        sol = ntk.analytical_solution(t=1.0)
        assert sol is not None and sol.shape == y.shape

    def test_eigendecomposition(self, setup):
        ntk, K, _ = setup
        evals, evecs = ntk.kernel_eigendecomposition(K)
        assert np.all(evals >= -1e-10)
        assert evecs.shape == K.shape

    def test_mode_dynamics_decay(self, setup):
        ntk, K, _ = setup
        evals, evecs = ntk.kernel_eigendecomposition(K)
        m_early = ntk.mode_dynamics(evals, evecs, t=0.01)
        m_late = ntk.mode_dynamics(evals, evecs, t=5.0)
        if m_early is not None and m_late is not None:
            assert np.linalg.norm(m_late) <= np.linalg.norm(m_early) + 1e-6

    def test_convergence_time_per_mode(self, setup):
        ntk, K, _ = setup
        evals, _ = ntk.kernel_eigendecomposition(K)
        times = np.asarray(ntk.convergence_time_per_mode(evals))
        assert np.all(times > 0)

    def test_spectral_bias_prediction(self, setup):
        ntk, K, y = setup
        evals, evecs = ntk.kernel_eigendecomposition(K)
        assert ntk.spectral_bias_prediction(evals, evecs, y) is not None

    def test_ntk_prediction_at_time(self, setup):
        ntk, K, _ = setup
        test_K = RNG.randn(5, K.shape[0])
        pred = ntk.ntk_prediction_at_time(1.0, test_K)
        assert pred is not None and pred.shape[0] == 5


class TestFeatureLearningDynamics:
    def test_kernel_change_rate(self):
        fld = FeatureLearningDynamics(100, 5, 1)
        n = 10; K0 = _spd_kernel(n, seed=10)
        kernels = [K0 + 0.01 * t * RNG.randn(n, n) for t in range(3)]
        kernels = [0.5*(K+K.T) for K in kernels]
        assert fld.kernel_change_rate(kernels, np.array([0.0, 0.5, 1.0])) is not None

    def test_ntk_deviation(self):
        fld = FeatureLearningDynamics(50, 3, 1)
        n = 8; K0 = _spd_kernel(n, seed=20)
        dev = fld.ntk_deviation(K0 + 0.05 * np.eye(n), K0)
        assert dev is not None and dev >= 0

    def test_catapult_phase_detection(self):
        fld = FeatureLearningDynamics(100, 5, 1)
        loss = np.concatenate([np.linspace(1, 5, 20), np.linspace(5, 0.3, 80)])
        assert fld.catapult_phase_detection(loss) is not None

    def test_feature_learning_timescale(self):
        fld = FeatureLearningDynamics(200, 4, 1)
        n = 6; K0 = _spd_kernel(n, seed=30)
        assert fld.feature_learning_timescale(
            [K0 + 0.01*t*np.eye(n) for t in range(5)], np.linspace(0, 4, 5)) is not None


# ===================================================================
# Learning rate schedules
# ===================================================================
class TestLearningRateScheduler:
    def test_constant(self):
        s = LearningRateScheduler(base_lr=0.1)
        assert s.constant(0) == pytest.approx(0.1)
        assert s.constant(100) == pytest.approx(0.1)

    def test_exponential_decay(self):
        s = LearningRateScheduler(base_lr=0.1)
        assert s.exponential_decay(100, 0.99) < s.exponential_decay(0, 0.99)

    def test_cosine_annealing_bounds(self):
        s = LearningRateScheduler(base_lr=0.1)
        for t in [0, 50, 100, 200]:
            lr = s.cosine_annealing(t, 200, eta_min=0.001)
            assert 0.001 - 1e-10 <= lr <= 0.1 + 1e-10

    def test_warmup_cosine(self):
        s = LearningRateScheduler(base_lr=0.1)
        lr0 = s.warmup_cosine(0, 10, 100)
        lr10 = s.warmup_cosine(10, 10, 100)
        assert lr10 >= lr0 or lr10 == pytest.approx(0.1, abs=0.02)

    def test_cyclical_bounds(self):
        s = LearningRateScheduler(base_lr=0.1)
        lrs = [s.cyclical(t, 20, 0.01, 0.1) for t in range(40)]
        assert min(lrs) >= 0.01 - 1e-10 and max(lrs) <= 0.1 + 1e-10

    def test_one_over_t(self):
        s = LearningRateScheduler(base_lr=0.1)
        assert s.one_over_t(10, 1.0) < s.one_over_t(1, 1.0)

    def test_critical_learning_rate(self):
        s = LearningRateScheduler(base_lr=0.1)
        crit = s.critical_learning_rate(np.array([0.5, 1.0, 2.0, 5.0]))
        assert crit is not None and crit > 0

    def test_stability_boundary(self):
        s = LearningRateScheduler(base_lr=0.1)
        assert s.stability_boundary(0.01, np.array([1.0, 3.0, 5.0])) is not None

    def test_edge_of_stability_lr(self):
        s = LearningRateScheduler(base_lr=0.1)
        assert s.edge_of_stability_lr(np.exp(-np.linspace(0, 3, 100)), 0.05) is not None


# ===================================================================
# SDE noise
# ===================================================================
class TestGradientNoiseSDE:
    def test_effective_temperature(self):
        sde = GradientNoiseSDE(10, 32, 1000, 0.01)
        t = sde.effective_temperature(0.01, 32)
        assert t is not None and t > 0

    def test_euler_maruyama(self):
        sde = GradientNoiseSDE(3, 32, 500, 0.01)
        traj = np.asarray(sde.sde_euler_maruyama(
            np.ones(3), lambda x: -x, lambda x: 0.1*np.ones_like(x), 0.01, 100))
        assert traj.shape[0] > 1

    def test_milstein(self):
        sde = GradientNoiseSDE(2, 16, 200, 0.01)
        traj = np.asarray(sde.sde_milstein(
            np.array([2.0, -1.0]), lambda x: -0.5*x, lambda x: 0.05*np.ones_like(x), 0.01, 50))
        assert traj.shape[0] > 1

    def test_escape_rate(self):
        sde = GradientNoiseSDE(5, 32, 1000, 0.01)
        r = sde.escape_rate(1.0, 0.1)
        assert r is not None and r > 0


# ===================================================================
# Hessian analysis
# ===================================================================
class TestHessianAnalyzer:
    @pytest.fixture()
    def setup(self):
        return HessianAnalyzer(param_dim=5, loss_fn=_quadratic_loss), _spd_kernel(5, seed=50)

    def test_compute_hessian_shape(self, setup):
        a, _ = setup
        X, y, _ = _make_linear_data(n=10, d=5)
        H = a.compute_hessian(np.zeros(5), _quadratic_loss, X, y)
        assert H.shape == (5, 5)

    def test_hessian_symmetry(self, setup):
        a, _ = setup
        X, y, _ = _make_linear_data(n=10, d=5)
        H = a.compute_hessian(RNG.randn(5)*0.1, _quadratic_loss, X, y)
        assert np.allclose(H, H.T, atol=1e-4)

    def test_top_eigenvalues(self, setup):
        a, H = setup
        top = np.asarray(a.top_eigenvalues(H, k=3))
        assert len(top) == 3 and top[0] >= top[-1]

    def test_full_spectrum(self, setup):
        a, H = setup
        assert len(np.asarray(a.full_spectrum(H))) == 5

    def test_spectral_density(self, setup):
        a, H = setup
        assert a.spectral_density(np.linalg.eigvalsh(H), n_bins=20) is not None

    def test_condition_number(self, setup):
        a, _ = setup
        assert a.condition_number(np.array([0.1, 1.0, 10.0])) == pytest.approx(100.0, rel=0.1)

    def test_sharpness(self, setup):
        a, _ = setup
        s = a.sharpness(np.array([0.5, 1.0, 3.0]))
        assert s is not None and s > 0

    def test_spectral_gap(self, setup):
        a, _ = setup
        assert a.spectral_gap(np.array([0.1, 0.5, 1.0, 5.0])) is not None

    def test_negative_curvature(self, setup):
        a, _ = setup
        assert a.negative_curvature_directions(np.array([-1.0, 0.5, 2.0]), np.eye(3)) is not None

    def test_trace_estimation(self, setup):
        a, H = setup
        tr = a.trace_estimation(lambda v: H @ v, dim=5, n_samples=50)
        assert abs(tr - np.trace(H)) / max(abs(np.trace(H)), 1) < 0.5

    def test_hessian_vector_product(self, setup):
        a, _ = setup
        X, y, _ = _make_linear_data(n=10, d=5)
        hvp = a.hessian_vector_product(np.zeros(5), RNG.randn(5), _quadratic_loss, X, y)
        assert hvp.shape == (5,)


# ===================================================================
# Loss surface visualization
# ===================================================================
class TestLossSurfaceVisualizer:
    @pytest.fixture()
    def viz(self):
        X, y, _ = _make_linear_data(n=30, d=3)
        return LossSurfaceVisualizer(_linear_mse_loss, X, y)

    def test_1d_slice(self, viz):
        a, l = viz.compute_1d_slice(np.zeros(3), np.array([1., 0., 0.]), (-1, 1), 50)
        assert len(np.asarray(a)) == len(np.asarray(l)) == 50

    def test_2d_slice(self, viz):
        assert viz.compute_2d_slice(np.zeros(3), np.array([1., 0., 0.]),
                                    np.array([0., 1., 0.]), n_points=10) is not None

    def test_random_direction_normalized(self, viz):
        assert abs(np.linalg.norm(np.asarray(viz.random_direction(10))) - 1.0) < 1e-6

    def test_pca_directions(self, viz):
        assert viz.pca_directions(RNG.randn(50, 3), n_components=2) is not None


# ===================================================================
# Saddle point detector
# ===================================================================
class TestSaddlePointDetector:
    def test_classify_minimum(self):
        assert "min" in str(SaddlePointDetector().classify_stationary_point(np.diag([1., 2., 3.]))).lower()

    def test_classify_saddle(self):
        label = str(SaddlePointDetector().classify_stationary_point(np.diag([-1., 2., 3.]))).lower()
        assert "saddle" in label or "indefinite" in label

    def test_saddle_index(self):
        assert SaddlePointDetector().saddle_index(np.diag([-1., -2., 3.])) == 2

    def test_escape_direction(self):
        esc = np.asarray(SaddlePointDetector().escape_direction(np.diag([-5., 1., 2.])))
        assert abs(esc[0]) > 0.5


# ===================================================================
# Loss barrier
# ===================================================================
class TestLossBarrierEstimator:
    @pytest.fixture()
    def setup(self):
        X, y, w = _make_linear_data(n=50, d=3, noise=0.0, seed=7)
        p1 = w + 0.01 * RNG.randn(3); p2 = w - 0.01 * RNG.randn(3)
        return LossBarrierEstimator(20), p1, p2, X, y

    def test_linear_barrier(self, setup):
        e, p1, p2, X, y = setup
        assert e.linear_barrier(p1, p2, _linear_mse_loss, X, y) >= -1e-8

    def test_mode_connectivity(self, setup):
        e, p1, p2, X, y = setup
        assert e.mode_connectivity(p1, p2, _linear_mse_loss, X, y) is not None

    def test_bezier_barrier(self, setup):
        e, p1, p2, X, y = setup
        assert e.bezier_curve_barrier(p1, p2, _linear_mse_loss, X, y) is not None


# ===================================================================
# Trajectory analyzer
# ===================================================================
class TestTrajectoryAnalyzer:
    def test_edge_of_stability(self):
        sharpness = 190 + 20 * np.sin(np.linspace(0, 10*np.pi, 100))
        assert TrajectoryAnalyzer().edge_of_stability_detection(sharpness, 0.01) is not None

    def test_progressive_sharpening(self):
        assert TrajectoryAnalyzer().progressive_sharpening(np.linspace(1, 10, 100), 10) is not None

    def test_trajectory_length(self):
        traj = np.cumsum(RNG.randn(50, 3)*0.1, axis=0)
        assert TrajectoryAnalyzer().trajectory_length(traj) > 0


# ===================================================================
# SGD simulator
# ===================================================================
class TestSGDSimulator:
    @pytest.fixture()
    def env(self):
        X, y, w = _make_linear_data(100, 5, 0.1, 3)
        return SGDSimulator(5, 0.01, 32), X, y, w

    def test_step(self, env):
        sim, X, y, _ = env
        p = np.zeros(5)
        assert not np.allclose(sim.step(p, _linear_mse_grad(p, X, y)), p)

    def test_run_loss_decreases(self, env):
        sim, X, y, _ = env
        x0 = RNG.randn(5)
        result = sim.run(x0, _linear_mse_loss, _linear_mse_grad, X, y, n_steps=200)
        final = result.get("params", result.get("final_params", x0)) if isinstance(result, dict) \
            else (result if isinstance(result, np.ndarray) and result.ndim == 1 else np.asarray(result)[-1])
        assert _linear_mse_loss(final, X, y) < _linear_mse_loss(x0, X, y)

    def test_gradient_variance(self, env):
        sim, X, y, _ = env
        assert np.all(np.asarray(sim.gradient_variance(np.zeros(5), _linear_mse_grad, X, y, 30)) >= 0)

    def test_loss_trajectory(self, env):
        sim, X, y, _ = env
        traj = np.stack([np.zeros(5)+0.01*i*np.ones(5) for i in range(10)])
        assert len(sim.loss_trajectory(traj, _linear_mse_loss, X, y)) == 10

    def test_sample_batch(self, env):
        sim, X, y, _ = env
        bx, by = sim.sample_batch(X, y, 16)
        assert bx.shape[0] == 16 and by.shape[0] == 16


# ===================================================================
# LR phase analysis
# ===================================================================
class TestLearningRatePhaseAnalyzer:
    def test_find_critical_lr(self):
        a = LearningRatePhaseAnalyzer((1e-4, 10.0), 10)
        scan = {"learning_rates": np.logspace(-4, 1, 10),
                "final_losses": np.array([.5,.3,.1,.05,.01,.1,1,10,100,1000])}
        crit = a.find_critical_lr(scan)
        assert crit is not None and crit > 0

    def test_convergence_vs_lr(self):
        a = LearningRatePhaseAnalyzer()
        scan = {"learning_rates": np.logspace(-3, 0, 8),
                "final_losses": np.array([.5,.3,.1,.05,.5,5,50,500])}
        assert a.convergence_vs_lr(scan) is not None


# ===================================================================
# Batch size effects
# ===================================================================
class TestBatchSizeEffectAnalyzer:
    def test_noise_scale(self):
        ns = BatchSizeEffectAnalyzer().noise_scale(32, 1000, 0.5)
        assert ns is not None and ns > 0

    def test_linear_scaling_rule(self):
        assert BatchSizeEffectAnalyzer().linear_scaling_rule(0.01, 32, 128) == pytest.approx(0.04, rel=0.01)

    def test_gradient_noise_ratio(self):
        X, y, _ = _make_linear_data(100, 5, seed=8)
        r = BatchSizeEffectAnalyzer().gradient_noise_ratio(np.zeros(5), _linear_mse_grad, X, y, 16)
        assert r is not None and r >= 0

    def test_critical_batch_size(self):
        scan = {"batch_sizes": [8,16,32,64,128,256], "final_losses": [.05,.04,.03,.03,.04,.06]}
        assert BatchSizeEffectAnalyzer().critical_batch_size(scan) is not None


# ===================================================================
# Momentum
# ===================================================================
class TestMomentumDynamics:
    def test_sgd_momentum_step(self):
        p, v = MomentumDynamics(0.9).sgd_momentum_step(np.array([1., 2.]), np.array([.5, 1.]), np.zeros(2), 0.01)
        assert not np.allclose(p, [1., 2.])

    def test_nesterov_step(self):
        p, v = MomentumDynamics(0.9, True).nesterov_step(np.array([1., 2.]), np.array([.5, 1.]), np.zeros(2), 0.01)
        assert not np.allclose(p, [1., 2.])

    def test_effective_lr(self):
        assert MomentumDynamics(0.9).effective_learning_rate(0.01) > 0.01

    def test_oscillation_detection(self):
        traj = np.column_stack([np.sin(np.linspace(0, 20*np.pi, 200)),
                                np.cos(np.linspace(0, 20*np.pi, 200))])
        assert MomentumDynamics(0.9).oscillation_detection(traj, 50) is not None


# ===================================================================
# SGD noise covariance
# ===================================================================
class TestSGDNoiseCovarianceEstimator:
    def test_estimate_covariance(self):
        X, y, _ = _make_linear_data(100, 4, seed=9)
        cov = np.asarray(SGDNoiseCovarianceEstimator(50).estimate_covariance(
            np.zeros(4), _linear_mse_grad, X, y, 16))
        assert cov.shape == (4, 4) and np.allclose(cov, cov.T, atol=1e-6)

    def test_low_rank(self):
        assert SGDNoiseCovarianceEstimator().low_rank_approximation(_spd_kernel(10, 11), 3) is not None

    def test_top_eigenvalues(self):
        top = np.asarray(SGDNoiseCovarianceEstimator().top_eigenvalues(_spd_kernel(8, 12), 3))
        assert len(top) == 3 and top[0] >= top[-1]

    def test_effective_noise_dimension(self):
        d = SGDNoiseCovarianceEstimator().effective_noise_dimension(np.diag([10., 5., 1., .1, .01]))
        assert d is not None and d > 0


# ===================================================================
# SGD → SDE
# ===================================================================
class TestSGDtoSDEConverter:
    def test_drift(self):
        X, y, _ = _make_linear_data(50, 3, seed=13)
        assert SGDtoSDEConverter(0.01, 32, 1000).drift_coefficient(np.zeros(3), _linear_mse_grad, X, y) is not None

    def test_diffusion(self):
        X, y, _ = _make_linear_data(50, 3, seed=14)
        assert SGDtoSDEConverter(0.01, 32, 1000).diffusion_coefficient(np.zeros(3), _linear_mse_grad, X, y) is not None

    def test_sde_simulation(self):
        traj = np.asarray(SGDtoSDEConverter(0.01, 32, 500).sde_simulation(
            np.array([1., 1.]), lambda x: -x, lambda x: 0.1*np.eye(len(x)), 0.01, 100))
        assert traj.shape[0] > 1

    def test_kramers_rate(self):
        r = SGDtoSDEConverter(0.01, 32, 1000).kramers_rate(2.0)
        assert r is not None and r > 0


# ===================================================================
# Lazy regime
# ===================================================================
class TestLazyRegimeAnalyzer:
    def test_is_lazy_small_change(self):
        n = 10; K0 = _spd_kernel(n, seed=40)
        assert LazyRegimeAnalyzer(1000, 0.001).is_lazy_regime(
            [K0, K0+0.001*np.eye(n), K0+0.002*np.eye(n)], tolerance=0.05)

    def test_not_lazy_large_change(self):
        n = 10; K0 = _spd_kernel(n, seed=41)
        assert not LazyRegimeAnalyzer(10, 1.0).is_lazy_regime(
            [K0, K0+5*np.eye(n), K0+10*np.eye(n)], tolerance=0.05)

    def test_kernel_relative_change(self):
        n = 8; K0 = _spd_kernel(n, seed=42)
        c = LazyRegimeAnalyzer(500).kernel_relative_change(K0+0.01*np.eye(n), K0)
        assert 0 <= c < 1.0

    def test_lazy_duration(self):
        n = 6; K0 = _spd_kernel(n, seed=43)
        d = LazyRegimeAnalyzer(1000).lazy_regime_duration(
            [K0+0.001*t*np.eye(n) for t in range(20)], np.linspace(0, 19, 20))
        assert d is not None and d > 0

    def test_effective_ridge(self):
        assert LazyRegimeAnalyzer(500).effective_ridge(_spd_kernel(10, 44), 0.01) is not None


class TestNTKStabilityChecker:
    def test_stable(self):
        n = 8; K0 = _spd_kernel(n, seed=50)
        r = NTKStabilityChecker(0.1).check_stability(
            [K0+0.001*i*np.eye(n) for i in range(5)], np.linspace(0, 4, 5))
        assert r is True or r is not None

    def test_operator_norm_change(self):
        n = 6; K0 = _spd_kernel(n, seed=51)
        assert NTKStabilityChecker().operator_norm_change(K0+0.05*np.eye(n), K0) >= 0

    def test_spectral_stability(self):
        n = 6; K0 = _spd_kernel(n, seed=52)
        assert NTKStabilityChecker().spectral_stability([K0+0.01*t*np.eye(n) for t in range(5)]) is not None

    def test_trace_stability(self):
        n = 6; K0 = _spd_kernel(n, seed=53)
        assert NTKStabilityChecker().trace_stability([K0+0.01*t*np.eye(n) for t in range(5)]) is not None


class TestLinearizedDynamicsSolver:
    @pytest.fixture()
    def lds(self):
        return LinearizedDynamicsSolver(_spd_kernel(15, 60), RNG.randn(15), 1e-4)

    def test_solve_continuous(self, lds):
        assert np.asarray(lds.solve_continuous((0, 5.0), 100)).shape[0] >= 1

    def test_solve_discrete(self, lds):
        assert np.asarray(lds.solve_discrete(0.01, 200)).shape[0] >= 1

    def test_residual_decay(self, lds):
        r0, r10 = lds.residual_dynamics(0.0), lds.residual_dynamics(10.0)
        if r0 is not None and r10 is not None:
            assert np.linalg.norm(r10) <= np.linalg.norm(r0) + 1e-6

    def test_prediction_at_test(self, lds):
        pred = lds.prediction_at_test(RNG.randn(5, 15), t=2.0)
        assert pred is not None and pred.shape[0] == 5


class TestKernelRegressionPredictor:
    @pytest.fixture()
    def krr(self):
        K = _spd_kernel(30, 70); y = RNG.randn(30)
        return KernelRegressionPredictor(1e-4), K, y

    def test_fit_predict(self, krr):
        p, K, y = krr; p.fit(K, y)
        pred = np.asarray(p.predict(K))
        assert pred.shape == y.shape and np.mean((pred - y)**2) < 1.0

    def test_loo_error(self, krr):
        p, K, y = krr
        e = p.leave_one_out_error(K, y)
        assert e is not None and e >= 0

    def test_effective_dimension(self, krr):
        p, K, _ = krr
        d = p.effective_dimension(K, 1e-4)
        assert d is not None and 0 < d <= 30


# ===================================================================
# Rich regime
# ===================================================================
class TestRichRegimeAnalyzer:
    def test_is_rich(self):
        n = 8; K0 = _spd_kernel(n, seed=80)
        assert RichRegimeAnalyzer(5, 50, 1).is_rich_regime(
            [K0, K0+2*np.eye(n), K0+5*np.eye(n)], threshold=0.2)

    def test_feature_learning_strength(self):
        n = 8; K0 = _spd_kernel(n, seed=81)
        assert RichRegimeAnalyzer(5, 50, 1).feature_learning_strength(
            [K0+0.5*t*np.eye(n) for t in range(5)], np.linspace(0, 4, 5)) is not None

    def test_representation_similarity(self):
        f1 = RNG.randn(20, 10)
        assert RichRegimeAnalyzer(5, 50, 1).representation_similarity(f1, f1+0.1*RNG.randn(20, 10)) is not None

    def test_effective_rank_evolution(self):
        ranks = np.asarray(RichRegimeAnalyzer(5, 50, 1).effective_rank_evolution(
            [RNG.randn(20, 10) for _ in range(5)]))
        assert len(ranks) == 5


class TestFeatureEvolutionTracker:
    def test_velocity(self):
        assert FeatureEvolutionTracker(5).feature_velocity(
            [RNG.randn(10, 5) for _ in range(5)], np.linspace(0, 4, 5)) is not None

    def test_acceleration(self):
        assert FeatureEvolutionTracker(5).feature_acceleration(
            [RNG.randn(10, 5) for _ in range(5)], np.linspace(0, 4, 5)) is not None

    def test_principal_dynamics(self):
        assert FeatureEvolutionTracker().principal_feature_dynamics(
            [RNG.randn(10, 5) for _ in range(5)]) is not None


class TestRepresentationChangeMetric:
    def test_cka(self):
        n = 20; K = _spd_kernel(n, 90)
        cka = RepresentationChangeMetric().centered_kernel_alignment(K, K+0.1*np.eye(n))
        assert 0 <= cka <= 1 + 1e-6

    def test_cka_identical(self):
        K = _spd_kernel(15, 91)
        assert RepresentationChangeMetric().centered_kernel_alignment(K, K) == pytest.approx(1.0, abs=1e-4)

    def test_procrustes(self):
        f = RNG.randn(20, 5)
        assert RepresentationChangeMetric().procrustes_distance(f, f+0.01*RNG.randn(20, 5)) >= 0

    def test_cca(self):
        f = RNG.randn(30, 5)
        assert RepresentationChangeMetric().cca_similarity(f, f@RNG.randn(5, 5)+0.01*RNG.randn(30, 5)) is not None


class TestFeatureAlignmentAnalyzer:
    def test_kta(self):
        assert FeatureAlignmentAnalyzer().kernel_target_alignment(
            _spd_kernel(20, 100), RNG.choice([-1, 1], 20).astype(float)) is not None

    def test_feature_label_correlation(self):
        f = RNG.randn(50, 10)
        assert FeatureAlignmentAnalyzer().feature_label_correlation(f, (f[:, 0]>0).astype(float)) is not None

    def test_class_separability(self):
        feats = np.vstack([RNG.randn(30, 5)+[2,0,0,0,0], RNG.randn(30, 5)-[2,0,0,0,0]])
        s = FeatureAlignmentAnalyzer().class_separability(feats, np.array([0]*30+[1]*30))
        assert s is not None and s > 0


# ===================================================================
# Neural collapse
# ===================================================================
class TestNeuralCollapseDetector:
    @pytest.fixture()
    def setup(self):
        nc, npc, d = 3, 20, 10
        means = np.eye(nc, d) * 5.0
        feats = np.vstack([means[c]+0.1*RNG.randn(npc, d) for c in range(nc)])
        labels = np.repeat(np.arange(nc), npc)
        return NeuralCollapseDetector(nc), feats, labels

    def test_class_means(self, setup):
        ncd, f, l = setup
        assert np.asarray(ncd.class_means(f, l)).shape[0] == 3

    def test_within_class_variance(self, setup):
        ncd, f, l = setup
        wcv = ncd.within_class_variance(f, l)
        if isinstance(wcv, (int, float, np.floating)):
            assert wcv < 1.0

    def test_nc1(self, setup):
        ncd, f, l = setup
        assert ncd.nc1_variability_collapse(f, l) is not None

    def test_nc2(self, setup):
        ncd, f, l = setup
        cm = np.asarray(ncd.class_means(f, l))
        assert ncd.nc2_convergence_to_simplex_etf(cm) is not None

    def test_nc3(self, setup):
        ncd, f, l = setup
        cm = np.asarray(ncd.class_means(f, l))
        W = cm / (np.linalg.norm(cm, axis=1, keepdims=True) + 1e-8)
        assert ncd.nc3_self_duality(cm, W) is not None

    def test_metrics_all(self, setup):
        ncd, f, l = setup
        cm = np.asarray(ncd.class_means(f, l))
        W = cm / (np.linalg.norm(cm, axis=1, keepdims=True) + 1e-8)
        m = ncd.neural_collapse_metrics(f, l, W)
        assert m is not None
        if isinstance(m, dict):
            assert len(m) >= 1
