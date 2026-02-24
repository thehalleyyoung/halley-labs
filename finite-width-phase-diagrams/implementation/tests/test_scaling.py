"""Tests for the scaling module (scaling_laws, mu_p, width_scaling, depth_scaling)."""

from __future__ import annotations

import sys
from pathlib import Path

_impl_root = Path(__file__).resolve().parent.parent
if str(_impl_root) not in sys.path:
    sys.path.insert(0, str(_impl_root))

import numpy as np
import pytest

from src.scaling import (
    MuPScalingComputer, MuPLearningRateTransfer, MuPInitialization,
    MuPViolationDetector, CoordinateCheck,
    NTKWidthScaling, FiniteWidthCorrectionScaling, CriticalExponentExtractor,
    ScalingCollapseAnalyzer, PowerLawFitter,
    KernelDepthPropagation, SignalPropagationAnalyzer, DepthPhaseBoundary,
    OptimalDepthPredictor, DepthWidthInteraction,
    ScalingExponentComputer, ScalingLawFitter, ScalingLawPredictor,
    ChinchillaAllocator, ArchitectureScalingComparator,
)
from src.scaling.scaling_laws import FitResult


@pytest.fixture
def rng():
    return np.random.RandomState(42)


# ===================================================================
# µP Scaling Computer
# ===================================================================

class TestMuPScalingComputer:

    def test_init_default_base_width(self):
        assert MuPScalingComputer().base_width == 64

    def test_init_custom_base_width(self):
        assert MuPScalingComputer(128).base_width == 128

    def test_init_invalid_base_width(self):
        with pytest.raises(ValueError):
            MuPScalingComputer(0)

    def test_compute_scaling_exponents_all_types(self):
        comp = MuPScalingComputer(64)
        types = ["input", "hidden", "output", "attention", "embedding"]
        results = comp.compute_scaling_exponents(types, [128] * 5)
        assert len(results) == 5
        for r in results:
            assert all(k in r for k in ("a", "b", "c", "sigma_mult", "lr_mult", "output_mult"))

    def test_hidden_exponents(self):
        comp = MuPScalingComputer(64)
        results = comp.compute_scaling_exponents(["input", "hidden", "output"], [128] * 3)
        r = results[1]  # hidden is the middle layer
        assert r["a"] == pytest.approx(-0.5, abs=0.01)
        assert r["b"] == pytest.approx(-1.0, abs=0.01)

    def test_init_scale_hidden(self):
        comp = MuPScalingComputer(64)
        scale = comp.init_scale("hidden", fan_in=128, fan_out=128)
        assert scale > 0 and np.isfinite(scale)

    def test_lr_scale_decreases_with_width(self):
        comp = MuPScalingComputer(64)
        lr_narrow = comp.lr_scale("hidden", 64, 64, 0.01)
        lr_wide = comp.lr_scale("hidden", 512, 512, 0.01)
        assert lr_wide < lr_narrow

    def test_output_multiplier(self):
        assert MuPScalingComputer(64).output_multiplier(fan_in=256) > 0

    def test_layer_scaling_methods(self):
        comp = MuPScalingComputer(64)
        assert isinstance(comp.input_layer_scaling(128, 128), dict)
        assert isinstance(comp.hidden_layer_scaling(128, 128), dict)
        assert isinstance(comp.output_layer_scaling(128, 128), dict)
        assert isinstance(comp.attention_scaling(128, 8), dict)

    def test_validate_scaling_table(self):
        table = [{"layer_type": "hidden", "fan_in": 128, "fan_out": 128}]
        result = MuPScalingComputer(64).validate_scaling_table(table)
        assert isinstance(result, dict)


# ===================================================================
# µP Learning Rate Transfer
# ===================================================================

class TestMuPLearningRateTransfer:

    def test_transfer_lr_invariant(self):
        t = MuPLearningRateTransfer(64, 0.01)
        for w in [64, 128, 512, 1024]:
            assert t.transfer_lr(w) == pytest.approx(0.01)

    def test_per_layer_lr_hidden_decreases(self):
        t = MuPLearningRateTransfer(64, 0.01)
        lr_s = t.per_layer_lr("hidden", 64, 0.01)
        lr_l = t.per_layer_lr("hidden", 512, 0.01)
        assert lr_l < lr_s

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            MuPLearningRateTransfer(0, 0.01)
        with pytest.raises(ValueError):
            MuPLearningRateTransfer(64, -0.01)

    def test_optimal_lr_at_width(self):
        t = MuPLearningRateTransfer(32, 0.01)

        def loss_fn(w, x, y):
            return float(np.mean((x @ w - y) ** 2))

        rng = np.random.RandomState(42)
        x = rng.randn(50, 4)
        y = x @ rng.randn(4, 1)
        best_lr, best_loss = t.optimal_lr_at_width(32, loss_fn, (x, y), n_trials=10)
        assert best_lr > 0 and np.isfinite(best_loss)

    def test_lr_transfer_error(self):
        t = MuPLearningRateTransfer(64, 0.01)
        result = t.lr_transfer_error(
            [32, 64, 128],
            [0.02, 0.01, 0.005],
            [0.019, 0.01, 0.006],
        )
        assert isinstance(result, dict)
        assert "mean_relative_error" in result


# ===================================================================
# µP Initialization
# ===================================================================

class TestMuPInitialization:

    def test_compute_init_std_hidden(self):
        init = MuPInitialization(64)
        assert init.compute_init_std("hidden", 256, 256, 256) > 0

    def test_compute_init_std_embedding(self):
        assert MuPInitialization(64).compute_init_std("embedding", 100, 256, 256) == 1.0

    def test_initialize_layer_shape_and_stats(self):
        init = MuPInitialization(64)
        expected = init.compute_init_std("hidden", 512, 512, 512)
        W = init.initialize_layer("hidden", (512, 512), 512, np.random.default_rng(0))
        assert W.shape == (512, 512)
        assert np.std(W) == pytest.approx(expected, rel=0.15)

    def test_he_mup(self):
        assert MuPInitialization(64).he_mup(256, 256, 256) > 0

    def test_lecun_mup(self):
        assert MuPInitialization(64).lecun_mup(256, 256) > 0

    def test_xavier_mup(self):
        assert MuPInitialization(64).xavier_mup(256, 256, 256) > 0

    def test_verify_activation_scale(self):
        init = MuPInitialization(64)
        rng = np.random.default_rng(42)
        W = init.initialize_layer("hidden", (32, 128), 128, rng)
        x = rng.standard_normal((20, 32))
        assert isinstance(init.verify_activation_scale([W], x), dict)

    def test_initialize_network(self):
        init = MuPInitialization(64)
        specs = [{"layer_type": "input", "shape": (10, 128)},
                 {"layer_type": "hidden", "shape": (128, 128)},
                 {"layer_type": "output", "shape": (128, 1)}]
        weights = init.initialize_network(specs, 128, seed=0)
        assert len(weights) == 3
        assert weights[1].shape == (128, 128)


# ===================================================================
# µP Violation Detector
# ===================================================================

class TestMuPViolationDetector:

    def test_init_scale_no_violation(self):
        d = MuPViolationDetector(0.1)
        W = np.random.RandomState(0).randn(500, 500) * 0.05
        r = d.check_init_scale(W, 0.05)
        assert not r["violation"] and r["severity"] == "none"

    def test_init_scale_major_violation(self):
        d = MuPViolationDetector(0.1)
        W = np.random.RandomState(0).randn(500, 500) * 1.0
        assert d.check_init_scale(W, 0.05)["severity"] == "major"

    def test_detect_blowup(self):
        d = MuPViolationDetector(0.1)
        assert d.detect_blowup({64: 1, 128: 2, 256: 4, 512: 8})["blowup"]

    def test_no_blowup(self):
        d = MuPViolationDetector(0.1)
        assert not d.detect_blowup({64: 1.01, 128: 0.99, 256: 1.02})["blowup"]

    def test_detect_vanishing(self):
        d = MuPViolationDetector(0.1)
        assert d.detect_vanishing({64: 1, 128: 0.5, 256: 0.25})["vanishing"]

    def test_no_vanishing(self):
        d = MuPViolationDetector(0.1)
        assert not d.detect_vanishing({64: 1.0, 128: 1.01, 256: 0.99})["vanishing"]

    def test_check_output_scale(self):
        d = MuPViolationDetector(0.1)
        r = d.check_output_scale(np.random.RandomState(0).randn(100) * 0.5, 256)
        assert not r["violation"]

    def test_check_lr_scale(self):
        d = MuPViolationDetector(0.1)
        r = d.check_lr_scale(np.random.RandomState(0).randn(1000) * 0.01, 0.01)
        assert "ratio" in r


# ===================================================================
# NTK Width Scaling
# ===================================================================

class TestNTKWidthScaling:

    def test_ntk_shape_and_symmetry(self):
        ntk = NTKWidthScaling(64)
        x = np.random.RandomState(0).randn(5, 3)
        K = ntk.ntk_at_width(128, x, depth=2)
        assert K.shape == (5, 5)
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_ntk_trace_vs_width(self):
        ntk = NTKWidthScaling(32)
        x = np.random.RandomState(2).randn(3, 2)
        widths_out, traces = ntk.ntk_trace_vs_width([32, 64, 128], x)
        assert len(traces) == 3 and all(np.isfinite(traces))

    def test_eigenvalue_scaling(self):
        ntk = NTKWidthScaling(32)
        x = np.random.RandomState(3).randn(5, 3)
        assert isinstance(ntk.eigenvalue_scaling([32, 64], x, k=3), dict)

    def test_finite_width_kernel_correction(self):
        ntk = NTKWidthScaling(64)
        K_inf = np.eye(4) + 0.3 * np.ones((4, 4))
        assert ntk.finite_width_kernel_correction(128, K_inf).shape == (4, 4)


# ===================================================================
# Finite-Width Correction Scaling
# ===================================================================

class TestFiniteWidthCorrectionScaling:

    def _1_over_n(self, a=2.0, b=5.0):
        w = np.array([32, 64, 128, 256, 512, 1024])
        return w, a + b / w

    def test_first_order_correction(self):
        fwc = FiniteWidthCorrectionScaling()
        w, v = self._1_over_n(2.0, 5.0)
        r = fwc.first_order_correction(w, v)
        assert r["infinite_width_value"] == pytest.approx(2.0, abs=0.01)
        assert r["first_order_coeff"] == pytest.approx(5.0, abs=0.1)
        assert r["r_squared"] > 0.99

    def test_second_order_correction(self):
        fwc = FiniteWidthCorrectionScaling()
        w = np.array([32, 64, 128, 256, 512, 1024])
        v = 3.0 + 2.0 / w + 10.0 / w**2
        r = fwc.second_order_correction(w, v)
        assert r["infinite_width_value"] == pytest.approx(3.0, abs=0.05)
        assert r["r_squared"] > 0.99

    def test_correction_exponent(self):
        fwc = FiniteWidthCorrectionScaling()
        w = np.array([32, 64, 128, 256, 512, 1024])
        v = 1.0 + 50.0 / w**1.5
        r = fwc.correction_exponent(w, v)
        assert r["exponent"] == pytest.approx(1.5, abs=0.2)

    def test_extrapolate_to_infinite(self):
        fwc = FiniteWidthCorrectionScaling()
        w, v = self._1_over_n(2.0, 5.0)
        r = fwc.extrapolate_to_infinite(w, v)
        assert r["estimate"] == pytest.approx(2.0, abs=0.1)
        assert len(r["all_orders"]) > 1

    def test_extrapolate_convergence(self):
        fwc = FiniteWidthCorrectionScaling()
        w = np.array([64, 128, 256, 512, 1024, 2048])
        v = 1.5 + 3.0 / w
        orders = fwc.extrapolate_to_infinite(w, v)["all_orders"]
        assert abs(orders[-1] - 1.5) <= abs(orders[0] - 1.5) + 0.01

    def test_finite_size_scaling(self):
        fwc = FiniteWidthCorrectionScaling()
        widths = np.array([32, 64, 128, 256])
        ctrl = np.linspace(-1, 1, 20)
        vals = {w: np.exp(-ctrl**2) + 1.0 / w for w in widths}
        assert isinstance(fwc.finite_size_scaling(widths, vals, ctrl), dict)


# ===================================================================
# Critical Exponent Extraction
# ===================================================================

class TestCriticalExponentExtractor:

    def test_log_log_fit(self):
        ext = CriticalExponentExtractor()
        x = np.logspace(0, 3, 50)
        r = ext.log_log_fit(x, 2.5 * x ** (-0.75))
        assert r["exponent"] == pytest.approx(-0.75, abs=0.01)
        assert r["r_squared"] > 0.999

    def test_extract_exponent_with_critical_point(self):
        ext = CriticalExponentExtractor()
        x = np.linspace(5.5, 15.0, 40)
        r = ext.extract_exponent(x, 3.0 * np.abs(x - 5.0) ** 0.5, critical_point=5.0)
        assert r["exponent"] == pytest.approx(0.5, abs=0.05)

    def test_confidence_interval(self):
        ext = CriticalExponentExtractor()
        x = np.linspace(0.1, 5.0, 50)
        y = 2.0 * x ** 0.8 + 0.05 * np.random.RandomState(0).randn(50)
        r = ext.confidence_interval(x, y, 0.0, n_bootstrap=200)
        lo, hi = r["ci_95"]
        assert lo < hi

    def test_crossover_detection(self):
        ext = CriticalExponentExtractor(min_points=5)
        x = np.logspace(0, 4, 100)
        y = np.where(x < 100, 10 * x ** (-0.5), 100 * x ** (-1.5))
        r = ext.crossover_detection(x, y)
        assert np.isfinite(r["crossover_point"])
        assert "exponent_below" in r and "exponent_above" in r

    def test_weighted_log_log_fit(self):
        ext = CriticalExponentExtractor()
        x = np.logspace(0, 2, 30)
        r = ext.weighted_log_log_fit(x, 3.0 * x ** (-1.0))
        assert r["exponent"] == pytest.approx(-1.0, abs=0.01)


# ===================================================================
# Scaling Collapse Analysis
# ===================================================================

class TestScalingCollapseAnalyzer:

    def _synth(self, nu=1.5, eta=0.5):
        sizes = [16, 32, 64, 128]
        cp = np.linspace(-2, 2, 30)
        data = []
        for L in sizes:
            xs = L ** (1.0 / nu) * cp
            data.append(np.exp(-xs**2) / L ** (eta / nu))
        return data, sizes, cp

    def test_collapse_and_quality(self):
        a = ScalingCollapseAnalyzer()
        ds, sz, cp = self._synth()
        collapsed = a.scaling_collapse(ds, sz, cp, 1.5, 0.5)
        assert len(collapsed) == 4
        assert a.collapse_quality(collapsed)["quality"] > 0.5

    def test_optimize_exponents(self):
        a = ScalingCollapseAnalyzer()
        ds, sz, cp = self._synth()
        r = a.optimize_exponents(ds, sz, cp, (0.5, 3.0), (-1.0, 2.0))
        assert r["nu"] == pytest.approx(1.5, abs=0.5)

    def test_residual_analysis(self):
        a = ScalingCollapseAnalyzer()
        ds, sz, cp = self._synth()
        collapsed = a.scaling_collapse(ds, sz, cp, 1.5, 0.5)
        uf = a.universal_function(collapsed)
        assert isinstance(a.residual_analysis(collapsed, uf["predict"]), dict)


# ===================================================================
# Power Law Fitter (width_scaling)
# ===================================================================

class TestPowerLawFitter:

    def test_fit_power_law_mle(self):
        f = PowerLawFitter("mle")
        x = np.logspace(0, 3, 50)
        r = f.fit_power_law(x, 3.0 * x ** (-0.7))
        assert "b" in r
        assert r["r_squared"] > 0.99

    def test_fit_power_law_ols(self):
        f = PowerLawFitter("ols")
        x = np.logspace(0, 3, 50)
        r = f.fit_power_law(x, 3.0 * x ** (-0.7))
        assert r["b"] == pytest.approx(-0.7, abs=0.02)
        assert r["r_squared"] > 0.99

    def test_fit_with_constant(self):
        f = PowerLawFitter()
        x = np.logspace(0, 3, 50)
        r = f.fit_with_constant(x, 2.0 * x ** (-1.0) + 0.5)
        assert r["c"] == pytest.approx(0.5, abs=0.15)
        assert r["r_squared"] > 0.95

    def test_power_law_vs_exponential(self):
        f = PowerLawFitter("ols")
        x = np.logspace(0, 2, 30)
        assert isinstance(f.power_law_vs_exponential(x, 5.0 * x ** (-0.5)), dict)

    def test_bayesian_fit(self):
        f = PowerLawFitter()
        x = np.logspace(0, 2, 30)
        y = 2.0 * x ** (-1.0) + 0.01 * np.random.RandomState(0).randn(30)
        assert isinstance(f.bayesian_fit(x, y), dict)


# ===================================================================
# Kernel Depth Propagation
# ===================================================================

class TestKernelDepthPropagation:

    def test_single_layer_scalar(self):
        p = KernelDepthPropagation("relu", 2.0)
        assert p.single_layer_kernel_map(1.0) > 0

    def test_single_layer_matrix(self):
        p = KernelDepthPropagation("relu", 2.0)
        K = p.single_layer_kernel_map(np.array([[1.0, 0.5], [0.5, 1.0]]))
        assert K.shape == (2, 2)
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_propagate_kernel(self):
        p = KernelDepthPropagation("relu", 2.0)
        assert np.isfinite(p.propagate_kernel(1.0, 10))

    def test_kernel_fixed_point_relu(self):
        p = KernelDepthPropagation("relu", 2.0)
        qstar = p.kernel_fixed_point(1.0)
        assert qstar > 0
        assert float(p.single_layer_kernel_map(qstar)) == pytest.approx(float(qstar), abs=1e-6)

    def test_kernel_fixed_point_tanh(self):
        assert np.isfinite(KernelDepthPropagation("tanh", 1.5).kernel_fixed_point(1.0))

    def test_correlation_propagation(self):
        rhos = KernelDepthPropagation("relu", 2.0).correlation_propagation(0.5, 20)
        assert len(rhos) == 21
        assert all(-1.01 <= r <= 1.01 for r in rhos)

    def test_kernel_trajectory(self):
        traj = KernelDepthPropagation("relu", 2.0).kernel_trajectory(1.5, 50)
        assert len(traj) <= 51 and all(np.isfinite(traj))


# ===================================================================
# Signal Propagation Analyzer
# ===================================================================

class TestSignalPropagationAnalyzer:

    def test_chi_1_relu_at_criticality(self):
        a = SignalPropagationAnalyzer("relu")
        assert a.chi_1(2.0, 1.0) == pytest.approx(1.0, abs=0.01)

    def test_chi_1_ordered_and_chaotic(self):
        a = SignalPropagationAnalyzer("relu")
        assert a.chi_1(1.5, 1.0) < 1.0
        assert a.chi_1(3.0, 1.0) > 1.0

    def test_ordered_phase(self):
        a = SignalPropagationAnalyzer("relu")
        o = a.ordered_phase(0.8)
        assert o["phase"] == "ordered" and o["gradient_vanishing"]
        assert a.ordered_phase(1.5)["phase"] == "not_ordered"

    def test_chaotic_phase(self):
        a = SignalPropagationAnalyzer("relu")
        c = a.chaotic_phase(1.5)
        assert c["phase"] == "chaotic" and c["gradient_exploding"]
        assert a.chaotic_phase(0.5)["phase"] == "not_chaotic"

    def test_edge_of_chaos_relu(self):
        a = SignalPropagationAnalyzer("relu")
        assert a.edge_of_chaos(np.linspace(0.5, 4.0, 50)) == pytest.approx(2.0, abs=0.1)

    def test_depth_scale(self):
        a = SignalPropagationAnalyzer()
        assert a.depth_scale(0.9) == pytest.approx(1 / abs(np.log(0.9)), abs=0.1)
        assert a.depth_scale(1.0) == np.inf

    def test_gradient_norm_propagation_decays(self):
        g = SignalPropagationAnalyzer().gradient_norm_propagation(10, 0.9)
        assert len(g) == 11 and g[-1] < g[0]

    def test_correlation_length(self):
        r = SignalPropagationAnalyzer().correlation_length(0.8, 20)
        assert isinstance(r, np.ndarray)
        assert len(r) == 21


# ===================================================================
# Depth Phase Boundary
# ===================================================================

class TestDepthPhaseBoundary:

    def test_critical_depth_ordered(self):
        dpb = DepthPhaseBoundary()
        assert dpb.critical_depth(0.9) == pytest.approx(np.log(0.01) / np.log(0.9), rel=0.01)

    def test_critical_depth_at_criticality(self):
        dpb = DepthPhaseBoundary()
        assert dpb.critical_depth(1.0) == np.inf

    def test_critical_depth_chaotic(self):
        dpb = DepthPhaseBoundary()
        assert dpb.critical_depth(1.5) == np.inf

    def test_critical_depth_zero(self):
        dpb = DepthPhaseBoundary()
        assert dpb.critical_depth(0.0) == 0.0

    def test_phase_boundary_vs_depth(self):
        dpb = DepthPhaseBoundary()
        b = dpb.phase_boundary_vs_depth(np.linspace(0.5, 4.0, 50), np.array([5, 10, 20, 50.0]))
        assert len(b) == 4

    def test_depth_width_phase_diagram(self):
        dpb = DepthPhaseBoundary()
        r = dpb.depth_width_phase_diagram(
            np.array([2, 4, 8]), np.array([32, 64]),
            lambda d, w: 1 / (1 + d / w),
        )
        assert r["diagram"].shape == (3, 2) and np.all(np.isfinite(r["diagram"]))

    def test_maximal_useful_depth(self):
        assert DepthPhaseBoundary().maximal_useful_depth(0.95) > 0

    def test_depth_scaling_of_order_parameter(self):
        r = DepthPhaseBoundary().depth_scaling_of_order_parameter(
            np.array([2, 4, 8, 16, 32, 64.0]), lambda L: 0.9 ** L,
        )
        assert isinstance(r, dict)


# ===================================================================
# Depth-Width Interaction
# ===================================================================

class TestDepthWidthInteraction:

    def test_compute_depth_width_grid(self):
        dwi = DepthWidthInteraction()
        r = dwi.compute_depth_width_grid([2, 4, 8], [32, 64], lambda d, w: d * w)
        assert r["grid"].shape == (3, 2) and r["grid"][0, 0] == 64

    def test_iso_performance_curves(self):
        dwi = DepthWidthInteraction()
        r = dwi.iso_performance_curves(
            np.linspace(1, 20, 20), np.linspace(16, 256, 20),
            lambda d, w: np.log(d * w), [5.0, 6.0],
        )
        assert "contours" in r

    def test_optimal_aspect_ratio(self):
        r = DepthWidthInteraction().optimal_aspect_ratio(
            np.array([1000, 5000, 10000]),
            lambda d, w: -((d * w**2 - 5000) ** 2),
        )
        assert isinstance(r, dict)

    def test_phase_diagram_depth_width(self):
        r = DepthWidthInteraction().phase_diagram_depth_width(
            [2, 4, 8], [16, 32, 64],
            lambda d, w: 0.95 ** d * (1 + 1 / w),
        )
        assert isinstance(r, dict)

    def test_interaction_strength(self):
        dwi = DepthWidthInteraction()
        metric = np.zeros((3, 2))
        for i, d in enumerate([2, 4, 8]):
            for j, w in enumerate([32, 64]):
                metric[i, j] = d + w
        assert isinstance(dwi.interaction_strength([2, 4, 8], [32, 64], metric), dict)

    def test_separability_test(self):
        dwi = DepthWidthInteraction()
        metric = np.outer([2, 4, 8], [32, 64])
        assert isinstance(dwi.separability_test([2, 4, 8], [32, 64], metric), dict)


# ===================================================================
# Scaling Exponent Computer
# ===================================================================

class TestScalingExponentComputer:

    def test_loss_vs_compute(self):
        c = ScalingExponentComputer()
        C = np.logspace(15, 22, 20)
        assert c.loss_vs_compute(C, 10 * C ** (-0.05))["exponent"] == pytest.approx(-0.05, abs=0.005)

    def test_loss_vs_params(self):
        c = ScalingExponentComputer()
        N = np.logspace(6, 10, 15)
        assert c.loss_vs_params(N, 5 * N ** (-0.076))["exponent"] == pytest.approx(-0.076, abs=0.01)

    def test_loss_vs_data(self):
        c = ScalingExponentComputer()
        D = np.logspace(8, 12, 15)
        assert c.loss_vs_data(D, 8 * D ** (-0.095))["exponent"] == pytest.approx(-0.095, abs=0.01)

    def test_loss_vs_width(self):
        c = ScalingExponentComputer()
        w = np.array([64, 128, 256, 512, 1024, 2048], dtype=float)
        assert c.loss_vs_width(w, 3 * w ** (-0.5))["exponent"] == pytest.approx(-0.5, abs=0.02)

    def test_compute_all_exponents(self):
        c = ScalingExponentComputer()
        C, N = np.logspace(15, 22, 10), np.logspace(6, 10, 10)
        r = c.compute_all_exponents({
            "compute": {"x": C, "y": 10 * C ** (-0.05)},
            "params": {"x": N, "y": 5 * N ** (-0.076)},
        })
        assert "compute" in r and "params" in r

    def test_effective_exponent(self):
        c = ScalingExponentComputer()
        x = np.logspace(1, 5, 50)
        assert isinstance(c.effective_exponent(x, 2 * x ** (-0.5), window_size=10), dict)


# ===================================================================
# Scaling Law Fitter
# ===================================================================

class TestScalingLawFitter:

    def test_fit_power_law(self):
        f = ScalingLawFitter()
        x = np.logspace(1, 5, 30)
        r = f.fit_power_law(x, 5 * x ** (-0.3) + 1)
        assert isinstance(r, FitResult) and r.r_squared > 0.9

    def test_fit_broken_power_law(self):
        f = ScalingLawFitter()
        x = np.logspace(0, 4, 60)
        y = np.where(x < 100, 10 * x ** (-0.3), 50 * x ** (-0.8))
        assert isinstance(f.fit_broken_power_law(x, y), FitResult)

    def test_fit_chinchilla(self):
        f = ScalingLawFitter()
        rng = np.random.RandomState(42)
        N = np.logspace(6, 10, 20)
        D = np.logspace(8, 12, 20)
        L = 1.5 + 400 * N ** (-0.34) + 400 * D ** (-0.28) + 0.01 * rng.randn(20)
        r = f.fit_chinchilla(N, D, L)
        assert isinstance(r, FitResult) and r.r_squared > 0.9

    def test_model_selection(self):
        f = ScalingLawFitter()
        x = np.logspace(1, 4, 40)
        r = f.model_selection(x, 3 * x ** (-0.5) + 0.5, ["power"])
        assert isinstance(r, dict)
        assert "best" in r

    def test_cross_validate(self):
        f = ScalingLawFitter()
        x = np.logspace(1, 4, 40)
        y = 3 * x ** (-0.5) + 0.5

        def model_fn(x_train, y_train):
            fr = ScalingLawFitter().fit_power_law(x_train, y_train)
            return fr.model_fn

        r = f.cross_validate(x, y, model_fn, k_folds=5)
        assert "mean_mse" in r

    def test_residual_analysis(self):
        f = ScalingLawFitter()
        x = np.logspace(1, 3, 30)
        y = 2 * x ** (-0.4) + 1
        assert isinstance(f.residual_analysis(x, y, lambda xv: 2 * xv ** (-0.4) + 1), dict)

    def test_prediction_intervals(self):
        f = ScalingLawFitter()
        x = np.logspace(1, 3, 30)
        y = 2 * x ** (-0.4) + 1 + 0.01 * np.random.RandomState(0).randn(30)
        fit = f.fit_power_law(x, y)
        r = f.prediction_intervals(x, fit)
        assert "lower" in r and "upper" in r


# ===================================================================
# Scaling Law Predictor
# ===================================================================

class TestScalingLawPredictor:

    def _fit(self):
        f = ScalingLawFitter()
        x = np.logspace(1, 5, 50)
        return f.fit_power_law(x, 5 * x ** (-0.3) + 1)

    def test_predict_loss(self):
        p = ScalingLawPredictor(self._fit())
        preds = p.predict_loss(np.array([100, 1000, 10000]))
        assert len(preds) == 3 and all(np.isfinite(preds))

    def test_predict_and_invert(self):
        fit = self._fit()
        p = ScalingLawPredictor(fit)
        target = p.predict_at_scale(1e4)
        assert np.isfinite(target)
        assert np.isfinite(p.compute_for_target_loss(target))

    def test_extrapolation_reliability(self):
        fit = self._fit()
        r = ScalingLawPredictor(fit).extrapolation_reliability(1e8, (10, 1e5))
        assert isinstance(r, dict)
        assert "reliability_score" in r


# ===================================================================
# Chinchilla Allocator
# ===================================================================

class TestChinchillaAllocator:

    def test_optimal_allocation(self):
        a = ChinchillaAllocator()
        r = a.optimal_allocation(1e18)
        assert r["N_star"] > 0 and r["D_star"] > 0
        assert r["N_star"] * r["D_star"] * 6 == pytest.approx(1e18, rel=0.1)

    def test_loss_decreases_with_compute(self):
        a = ChinchillaAllocator()
        assert a.loss_at_optimal(1e21) < a.loss_at_optimal(1e15)
        assert a.loss_at_optimal(1e18) > a.E

    def test_iso_loss_curves(self):
        a = ChinchillaAllocator()
        assert len(a.iso_loss_curves([2.5, 3.0], (1e6, 1e10), (1e8, 1e12), 50)) == 2

    def test_pareto_frontier(self):
        ca = ChinchillaAllocator()
        N = np.logspace(6, 10, 10)
        D = np.logspace(8, 12, 10)
        L = ca.E + ca.A * N**(-ca.alpha_N) + ca.B * D**(-ca.alpha_D)
        assert isinstance(ca.pareto_frontier(N, D, L), dict)

    def test_custom_exponents(self):
        r = ChinchillaAllocator(0.5, 0.5, 100, 100, 1.0).optimal_allocation(1e18)
        assert r["N_star"] > 0


# ===================================================================
# Architecture Scaling Comparator
# ===================================================================

class TestArchitectureScalingComparator:

    def _arch_results(self, n=20):
        x = np.logspace(6, 10, n)
        return {
            "arch1": {"x": x, "loss": 10 * x ** (-0.05)},
            "arch2": {"x": x, "loss": 10 * x ** (-0.07)},
        }

    def test_compare_exponents(self):
        c = ArchitectureScalingComparator()
        r = c.compare_exponents(self._arch_results())
        assert "exponents" in r

    def test_normalized_comparison(self):
        c = ArchitectureScalingComparator()
        r = c.normalized_comparison(self._arch_results(), (1e6, 1e10))
        assert "arch1" in r and "arch2" in r

    def test_crossover_point(self):
        c = ArchitectureScalingComparator()
        f = ScalingLawFitter()
        x = np.logspace(6, 12, 30)
        fr1 = f.fit_power_law(x, 20 * x ** (-0.03))
        fr2 = f.fit_power_law(x, 5 * x ** (-0.06))
        r = c.crossover_point(fr1, fr2)
        assert isinstance(r, dict)

    def test_scaling_efficiency(self):
        c = ArchitectureScalingComparator()
        r = c.scaling_efficiency(self._arch_results())
        assert "arch1" in r and "arch2" in r


# ===================================================================
# Optimal Depth Predictor
# ===================================================================

class TestOptimalDepthPredictor:

    def test_compute_optimal_depth(self):
        p = OptimalDepthPredictor(256, 10000)
        assert isinstance(p.compute_optimal_depth(256, 10000, 1.0), (int, float, dict))

    def test_depth_vs_width_tradeoff(self):
        assert isinstance(
            OptimalDepthPredictor(128, 5000).depth_vs_width_tradeoff(100000, np.arange(2, 20)),
            dict,
        )

    def test_excess_risk_vs_depth(self):
        assert isinstance(
            OptimalDepthPredictor(128, 5000).excess_risk_vs_depth(np.array([2, 4, 8, 16]), 128, 5000),
            dict,
        )


# ===================================================================
# Coordinate Check
# ===================================================================

class TestCoordinateCheck:

    def test_expected_scaling(self):
        cc = CoordinateCheck(base_width=64, widths=[64, 128, 256])
        val = cc.expected_scaling("hidden", "activation")
        assert isinstance(val, float)

    def test_deviation_from_expected(self):
        cc = CoordinateCheck(base_width=64, widths=[64, 128, 256])
        obs = {64: 1.0, 128: 0.98, 256: 1.02}
        exp = {64: 1.0, 128: 1.0, 256: 1.0}
        r = cc.deviation_from_expected(obs, exp)
        assert isinstance(r, dict)
        assert "passes" in r
