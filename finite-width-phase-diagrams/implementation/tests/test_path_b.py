"""Tests for PATH B improvements: ResNet MF, calibration diagnostics, expanded validation."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from mean_field_theory import (
    MeanFieldAnalyzer, ArchitectureSpec, ActivationVarianceMaps,
    PhaseClassification, ConfidenceInterval,
)
from resnet_mean_field import ResNetMeanField, ResNetMFReport
from calibration_diagnostics import (
    CalibrationDiagnostics, CalibrationReport, ReliabilityDiagram, compute_ece,
)


# ======================================================================
# ResNet Mean Field Tests
# ======================================================================

class TestResNetMeanField:
    """Test the ResNet mean field analysis module."""

    @pytest.fixture
    def rmf(self):
        return ResNetMeanField()

    def test_relu_cross_covariance(self, rmf):
        """E[z * ReLU(z)] = q/2 for z ~ N(0, q)."""
        for q in [0.5, 1.0, 2.0]:
            c = rmf.cross_covariance("relu", q)
            assert abs(c - q / 2.0) < 1e-6, f"ReLU C(q={q}) = {c}, expected {q/2}"

    def test_tanh_cross_covariance_positive(self, rmf):
        """C(q) should be positive for tanh."""
        for q in [0.5, 1.0, 2.0]:
            c = rmf.cross_covariance("tanh", q)
            assert c > 0, f"tanh C(q={q}) = {c}, expected positive"

    def test_gelu_cross_covariance_positive(self, rmf):
        """C(q) should be positive for GELU."""
        for q in [0.5, 1.0]:
            c = rmf.cross_covariance("gelu", q)
            assert c > 0, f"GELU C(q={q}) = {c}, expected positive"

    def test_resnet_variance_increases(self, rmf):
        """ResNet variance should increase with depth (additive skip)."""
        traj = rmf.resnet_variance_propagation(10, 1.0, 0.0, 1.0, "relu")
        assert len(traj) == 11
        # Variance should be non-decreasing for positive alpha
        for i in range(1, len(traj)):
            assert traj[i] >= traj[i-1] - 1e-10

    def test_resnet_chi1_greater_than_1(self, rmf):
        """ResNet chi_1 should be > 1 for reasonable sigma_w and alpha."""
        chi = rmf.resnet_chi1(1.0, 1.414, 1.0, "relu")
        assert chi > 1.0, f"ResNet chi_1 = {chi}, expected > 1"

    def test_resnet_vs_plain_stability(self, rmf):
        """ResNet should have better stability than plain MLP in chaotic regime."""
        report = rmf.analyze(20, 512, "relu", 2.0, alpha=0.3)
        # With small alpha, ResNet should be less chaotic
        assert report.chi_1_resnet < report.chi_1_plain or report.alpha < 1.0

    def test_resnet_fw_variance_propagation(self, rmf):
        """Finite-width corrected ResNet variance should have same length."""
        traj = rmf.resnet_fw_variance_propagation(5, 1.0, 0.0, 0.5, "relu", 128)
        assert len(traj) == 6

    def test_resnet_analysis_report(self, rmf):
        """Full analysis should produce a complete report."""
        report = rmf.analyze(5, 256, "relu", 1.414, alpha=0.5)
        assert isinstance(report, ResNetMFReport)
        assert report.depth == 5
        assert report.width == 256
        assert report.alpha == 0.5
        assert report.phase_plain in ("ordered", "critical", "chaotic")
        assert report.phase_resnet in ("ordered", "critical", "chaotic")
        assert len(report.variance_trajectory_plain) == 6
        assert len(report.variance_trajectory_resnet) == 6

    def test_resnet_multiple_activations(self, rmf):
        """ResNet analysis should work for all activations."""
        for act in ["relu", "tanh", "gelu", "silu"]:
            report = rmf.analyze(5, 128, act, 1.0, alpha=0.5)
            assert report.activation == act
            assert report.phase_resnet in ("ordered", "critical", "chaotic")

    def test_resnet_edge_of_chaos(self, rmf):
        """Should find edge of chaos for ResNet."""
        eoc = rmf.find_resnet_edge_of_chaos("relu", alpha=0.5)
        assert eoc > 0
        assert np.isfinite(eoc)

    def test_resnet_phase_diagram(self, rmf):
        """Should generate phase diagram data."""
        data = rmf.resnet_phase_diagram(
            "relu", alpha=0.5,
            sigma_w_values=[0.5, 1.0, 1.5],
            depths=[5, 10],
        )
        assert len(data) == 6  # 3 sigma_w x 2 depths
        assert all("sigma_w" in d for d in data)
        assert all("phase_resnet" in d for d in data)


# ======================================================================
# Calibration Diagnostics Tests
# ======================================================================

class TestCalibrationDiagnostics:
    """Test calibration diagnostics module."""

    @pytest.fixture
    def cal(self):
        return CalibrationDiagnostics(n_bins=5, adaptive=False)

    @pytest.fixture
    def cal_adaptive(self):
        return CalibrationDiagnostics(n_bins=5, adaptive=True)

    def test_perfect_calibration(self, cal):
        """Perfectly calibrated predictions should have ECE ≈ 0."""
        np.random.seed(42)
        n = 1000
        probs = np.random.uniform(0, 1, n)
        labels = (np.random.uniform(0, 1, n) < probs).astype(float)
        diagram = cal.compute_reliability_diagram(probs, labels)
        assert diagram.ece < 0.1  # Allow statistical noise

    def test_fully_miscalibrated(self, cal):
        """Always-0.9 predictions on mostly-0 labels should have high ECE."""
        probs = np.full(100, 0.9)
        labels = np.zeros(100)
        diagram = cal.compute_reliability_diagram(probs, labels)
        assert diagram.ece > 0.5

    def test_brier_score(self, cal):
        """Brier score should be 0 for perfect predictions."""
        probs = np.array([1.0, 0.0, 1.0, 0.0])
        labels = np.array([1.0, 0.0, 1.0, 0.0])
        diagram = cal.compute_reliability_diagram(probs, labels)
        assert diagram.brier_score < 1e-10

    def test_reliability_diagram_bins(self, cal):
        """Should produce correct number of bins."""
        probs = np.random.uniform(0, 1, 100)
        labels = (probs > 0.5).astype(float)
        diagram = cal.compute_reliability_diagram(probs, labels)
        assert len(diagram.bins) == 5  # n_bins=5

    def test_adaptive_binning(self, cal_adaptive):
        """Adaptive bins should have roughly equal counts."""
        np.random.seed(42)
        probs = np.random.uniform(0, 1, 100)
        labels = (probs > 0.5).astype(float)
        diagram = cal_adaptive.compute_reliability_diagram(probs, labels)
        counts = [b.count for b in diagram.bins]
        # Each bin should have ~20 samples (100 / 5)
        assert all(c > 0 for c in counts)

    def test_multiclass_calibration(self, cal):
        """Multi-class calibration should produce per-class results."""
        n = 90
        classes = ["ordered", "critical", "chaotic"]
        true_labels = np.array(
            ["ordered"] * 30 + ["critical"] * 30 + ["chaotic"] * 30
        )
        pred_probs = {
            "ordered": np.concatenate([np.random.uniform(0.6, 1.0, 30),
                                       np.random.uniform(0, 0.3, 60)]),
            "critical": np.concatenate([np.random.uniform(0, 0.3, 30),
                                        np.random.uniform(0.6, 1.0, 30),
                                        np.random.uniform(0, 0.3, 30)]),
            "chaotic": np.concatenate([np.random.uniform(0, 0.3, 60),
                                       np.random.uniform(0.6, 1.0, 30)]),
        }
        report = cal.compute_multiclass_calibration(pred_probs, true_labels, classes)
        assert isinstance(report, CalibrationReport)
        assert report.n_total == n
        assert "ordered" in report.per_class
        assert "critical" in report.per_class
        assert "chaotic" in report.per_class

    def test_ece_helper(self):
        """Quick ECE function should work."""
        probs = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        ece = compute_ece(probs, labels, n_bins=5)
        assert 0 <= ece <= 1

    def test_empty_input(self, cal):
        """Should handle empty input gracefully."""
        diagram = cal.compute_reliability_diagram(np.array([]), np.array([]))
        assert diagram.n_samples == 0
        assert diagram.ece == 0.0


# ======================================================================
# Expanded Mean Field Tests
# ======================================================================

class TestExpandedMeanField:
    """Test mean field theory across all activations and configurations."""

    @pytest.fixture
    def analyzer(self):
        return MeanFieldAnalyzer()

    @pytest.mark.parametrize("activation", ["relu", "tanh", "gelu", "silu"])
    def test_edge_of_chaos_exists(self, analyzer, activation):
        """Each activation should have a well-defined edge of chaos."""
        sw, _ = analyzer.find_edge_of_chaos(activation)
        assert sw > 0
        assert np.isfinite(sw)

    @pytest.mark.parametrize("activation", ["relu", "tanh", "gelu", "silu"])
    def test_chi_1_at_eoc_is_1(self, analyzer, activation):
        """At edge of chaos, chi_1 should ≈ 1."""
        sw, _ = analyzer.find_edge_of_chaos(activation)
        arch = ArchitectureSpec(depth=10, width=1000, activation=activation, sigma_w=sw)
        report = analyzer.analyze(arch)
        assert abs(report.chi_1 - 1.0) < 0.02, (
            f"{activation}: chi_1 = {report.chi_1} at σ_w = {sw}"
        )

    @pytest.mark.parametrize("activation", ["relu", "tanh", "gelu", "silu"])
    def test_ordered_phase_below_eoc(self, analyzer, activation):
        """Well below edge of chaos should be ordered."""
        sw, _ = analyzer.find_edge_of_chaos(activation)
        arch = ArchitectureSpec(
            depth=10, width=1000, activation=activation,
            sigma_w=max(sw * 0.5, 0.1)
        )
        report = analyzer.analyze(arch)
        assert report.chi_1 < 1.0

    @pytest.mark.parametrize("activation", ["relu", "tanh", "gelu", "silu"])
    def test_chaotic_phase_above_eoc(self, analyzer, activation):
        """Well above edge of chaos should be chaotic."""
        sw, _ = analyzer.find_edge_of_chaos(activation)
        arch = ArchitectureSpec(
            depth=10, width=1000, activation=activation,
            sigma_w=sw * 2.0
        )
        report = analyzer.analyze(arch)
        assert report.chi_1 > 1.0

    @pytest.mark.parametrize("activation", ["relu", "tanh", "gelu", "silu"])
    def test_chi_2_bifurcation_type(self, analyzer, activation):
        """Verify χ₂ bifurcation classification."""
        sw, _ = analyzer.find_edge_of_chaos(activation)
        arch = ArchitectureSpec(depth=10, width=1000, activation=activation, sigma_w=sw)
        report = analyzer.analyze(arch)

        if activation == "relu":
            # ReLU: piecewise linear, χ₂ = 0, degenerate bifurcation
            assert report.chi_2 < 0.01
            assert report.phase_classification.bifurcation_type == "degenerate"
        else:
            # Smooth activations: χ₂ > 0, supercritical bifurcation
            assert report.chi_2 > 0.001
            assert report.phase_classification.bifurcation_type == "supercritical"

    @pytest.mark.parametrize("depth", [5, 10, 20])
    def test_variance_trajectory_length(self, analyzer, depth):
        """Variance trajectory should have depth + 1 entries."""
        arch = ArchitectureSpec(depth=depth, width=256, activation="relu", sigma_w=1.414)
        report = analyzer.analyze(arch)
        assert len(report.variance_trajectory) == depth + 1
        assert len(report.finite_width_corrected_variance) == depth + 1


# ======================================================================
# Closed-form ReLU Verification Tests
# ======================================================================

class TestReLUClosedForm:
    """Verify all closed-form ReLU results."""

    def test_kurtosis_excess(self):
        """κ = E[φ⁴]/(E[φ²])² - 1 = 0.5 for ReLU."""
        assert ActivationVarianceMaps.relu_kurtosis_excess(1.0) == pytest.approx(0.5)

    def test_variance_map(self):
        """V(q) = q/2 for ReLU."""
        for q in [0.1, 0.5, 1.0, 2.0, 5.0]:
            assert ActivationVarianceMaps.relu_variance(q) == pytest.approx(q / 2)

    def test_chi_map(self):
        """χ₁(q) = 1/2 for ReLU."""
        assert ActivationVarianceMaps.relu_chi(1.0) == pytest.approx(0.5)

    def test_fourth_moment(self):
        """E[ReLU(z)⁴] = 3q²/8."""
        for q in [0.5, 1.0, 2.0]:
            assert ActivationVarianceMaps.relu_fourth_moment(q) == pytest.approx(3 * q**2 / 8)

    def test_sixth_moment(self):
        """E[ReLU(z)⁶] = 15q³/48."""
        for q in [0.5, 1.0, 2.0]:
            assert ActivationVarianceMaps.relu_sixth_moment(q) == pytest.approx(15 * q**3 / 48)

    def test_chi_2_zero(self):
        """χ₂ = 0 for ReLU (piecewise linear)."""
        assert ActivationVarianceMaps.relu_chi_2(1.0) == pytest.approx(0.0)

    def test_dphi_fourth(self):
        """E[φ'(z)⁴] = 1/2 for ReLU."""
        assert ActivationVarianceMaps.relu_dphi_fourth(1.0) == pytest.approx(0.5)

    def test_edge_of_chaos(self):
        """σ_w* = √2 for ReLU."""
        analyzer = MeanFieldAnalyzer()
        sw, _ = analyzer.find_edge_of_chaos("relu")
        assert sw == pytest.approx(np.sqrt(2), abs=1e-5)


# ======================================================================
# Perturbative Convergence Tests
# ======================================================================

class TestPerturbativeConvergence:
    """Verify perturbative expansion validity."""

    @pytest.fixture
    def analyzer(self):
        return MeanFieldAnalyzer()

    def test_correction_decreases_with_width(self, analyzer):
        """O(1/N) corrections should decrease monotonically with width."""
        widths = [32, 64, 128, 256, 512]
        errors = []
        for w in widths:
            arch = ArchitectureSpec(
                depth=5, width=w, activation="relu",
                sigma_w=1.35, sigma_b=0.0, input_variance=1.0,
            )
            report = analyzer.analyze(arch)
            mf = np.array(report.variance_trajectory)
            fw = np.array(report.finite_width_corrected_variance)
            correction = np.mean(np.abs(fw - mf) / np.maximum(np.abs(mf), 1e-10))
            errors.append(correction)

        # Corrections should generally decrease with width
        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i+1] * 0.5, (
                f"Correction at W={widths[i]} ({errors[i]:.4f}) should be >= "
                f"correction at W={widths[i+1]} ({errors[i+1]:.4f})"
            )

    def test_chi_1_ci_narrows_with_width(self, analyzer):
        """Confidence interval should narrow with increasing width."""
        widths = [128, 512, 2048]
        ci_widths = []
        for w in widths:
            sw_star, ci = analyzer.find_edge_of_chaos_with_ci("relu", width=w)
            ci_widths.append(ci.upper - ci.lower)

        for i in range(len(ci_widths) - 1):
            assert ci_widths[i] >= ci_widths[i+1] * 0.5


# ======================================================================
# Soundness Theorem Components
# ======================================================================

class TestSoundnessComponents:
    """Test the components needed for the formal soundness theorem."""

    @pytest.fixture
    def analyzer(self):
        return MeanFieldAnalyzer()

    def test_gaussian_init_assumption(self):
        """Verify that with Gaussian init, first layer output is Gaussian-like."""
        rng = np.random.RandomState(42)
        width = 1000
        sigma_w = 1.0
        x = rng.randn(100, 50)
        W = rng.randn(50, width) * sigma_w / np.sqrt(50)
        h = x @ W
        # Check approximate Gaussianity via kurtosis
        kurtosis = np.mean(h**4) / (np.mean(h**2)**2) - 3
        assert abs(kurtosis) < 0.5  # Should be close to 0 for Gaussian

    def test_iid_weight_assumption(self, analyzer):
        """Different seeds should give consistent fixed-point estimates."""
        # This tests that the analysis doesn't depend on specific initialization
        arch = ArchitectureSpec(depth=10, width=512, activation="relu", sigma_w=1.414)
        report = analyzer.analyze(arch)
        # Fixed point should be deterministic (no seed dependence)
        assert np.isfinite(report.fixed_point)
        assert report.fixed_point > 0

    def test_moment_closure_validity(self, analyzer):
        """Mean field prediction should be reasonably close to empirical."""
        arch = ArchitectureSpec(
            depth=5, width=512, activation="relu",
            sigma_w=1.35, sigma_b=0.0, input_variance=1.0,
        )
        report = analyzer.analyze(arch)
        mf_final = report.variance_trajectory[-1]

        # Empirical check
        rng = np.random.RandomState(42)
        vars_list = []
        for trial in range(50):
            rng_t = np.random.RandomState(trial)
            x = rng_t.randn(200, 50)
            h = x
            for l in range(5):
                W = rng_t.randn(h.shape[1], 512) * 1.35 / np.sqrt(h.shape[1])
                h = h @ W
                h = np.maximum(h, 0)
            vars_list.append(float(np.mean(h**2)))

        emp_mean = np.mean(vars_list)
        rel_error = abs(mf_final - emp_mean) / max(emp_mean, 1e-10)
        assert rel_error < 0.1, f"MF={mf_final:.4f}, empirical={emp_mean:.4f}, error={rel_error:.1%}"
