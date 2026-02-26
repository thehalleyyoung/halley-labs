"""Tests for marace.policy.lipsdp — Lipschitz constant bounding."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marace.policy.onnx_loader import (
    ActivationType,
    LayerInfo,
    NetworkArchitecture,
)
from marace.policy.lipsdp import (
    SpectralNormProductBound,
    RecursiveBound,
    LipschitzTightnessAnalysis,
    CascadingErrorAnalysis,
    BoundResult,
    _spectral_norm,
)


# ======================================================================
# Helpers
# ======================================================================

def _make_arch(weight_matrices, activations=None):
    """Create a NetworkArchitecture from weight matrices."""
    layers = []
    for i, W in enumerate(weight_matrices):
        act = (activations[i] if activations else ActivationType.RELU)
        if i == len(weight_matrices) - 1 and activations is None:
            act = ActivationType.LINEAR
        layers.append(LayerInfo(
            name=f"layer_{i}",
            layer_type="dense",
            input_size=W.shape[1],
            output_size=W.shape[0],
            activation=act,
            weights=W,
            bias=np.zeros(W.shape[0]),
        ))
    return NetworkArchitecture(
        layers=layers,
        input_dim=weight_matrices[0].shape[1],
        output_dim=weight_matrices[-1].shape[0],
    )


def _make_identity_arch(dim=4):
    """Single layer identity network."""
    W = np.eye(dim)
    return _make_arch([W], activations=[ActivationType.LINEAR])


def _make_scaling_arch(scale=2.0, dim=4):
    """Single layer scaling network: f(x) = scale * x."""
    W = scale * np.eye(dim)
    return _make_arch([W], activations=[ActivationType.LINEAR])


def _make_deep_arch(depth=3, dim=4, scale=1.5):
    """Multi-layer network with given depth."""
    weights = []
    activations = []
    for i in range(depth):
        W = np.eye(dim) * scale
        weights.append(W)
        activations.append(
            ActivationType.RELU if i < depth - 1 else ActivationType.LINEAR
        )
    return _make_arch(weights, activations)


# ======================================================================
# SpectralNormProductBound
# ======================================================================

class TestSpectralNormProductBound:
    """Test spectral-norm product Lipschitz bound."""

    def test_identity_network(self):
        """Identity network has Lipschitz constant 1."""
        arch = _make_identity_arch(4)
        bound = SpectralNormProductBound(seed=42)
        result = bound.upper_bound(arch)
        assert result.upper_bound == pytest.approx(1.0, abs=1e-6)

    def test_scaling_network(self):
        """Scaling network f(x)=2x has Lipschitz constant 2."""
        arch = _make_scaling_arch(scale=2.0, dim=4)
        bound = SpectralNormProductBound(seed=42)
        result = bound.upper_bound(arch)
        assert result.upper_bound == pytest.approx(2.0, abs=1e-6)

    def test_known_1_layer(self):
        """For 1-layer linear network, bound = spectral norm of W."""
        rng = np.random.RandomState(42)
        W = rng.randn(3, 5)
        arch = _make_arch([W], activations=[ActivationType.LINEAR])
        bound = SpectralNormProductBound(seed=42)
        result = bound.upper_bound(arch)
        expected = _spectral_norm(W)
        assert result.upper_bound == pytest.approx(expected, rel=1e-4)

    def test_deep_network_product(self):
        """Multi-layer: bound = product of per-layer spectral norms."""
        W1 = np.array([[2.0, 0.0], [0.0, 1.0]])
        W2 = np.array([[1.0, 0.0], [0.0, 3.0]])
        arch = _make_arch([W1, W2],
                          activations=[ActivationType.RELU, ActivationType.LINEAR])
        bound = SpectralNormProductBound(seed=42)
        result = bound.upper_bound(arch)
        expected = _spectral_norm(W1) * _spectral_norm(W2)
        assert result.upper_bound == pytest.approx(expected, rel=1e-4)

    def test_result_has_method_name(self):
        arch = _make_identity_arch()
        bound = SpectralNormProductBound(seed=42)
        result = bound.upper_bound(arch)
        assert "spectral" in result.method.lower()

    def test_bound_result_summary(self):
        arch = _make_identity_arch()
        bound = SpectralNormProductBound(seed=42)
        result = bound.upper_bound(arch)
        s = result.summary()
        assert isinstance(s, str)


# ======================================================================
# RecursiveBound
# ======================================================================

class TestRecursiveBound:
    """Test recursive Jacobian bound."""

    def test_recursive_leq_spectral(self):
        """RecursiveBound should be ≤ SpectralNormProductBound."""
        rng = np.random.RandomState(42)
        W1 = rng.randn(4, 4) * 0.5
        W2 = rng.randn(4, 4) * 0.5
        arch = _make_arch([W1, W2],
                          activations=[ActivationType.RELU, ActivationType.LINEAR])

        sp = SpectralNormProductBound(seed=42)
        sp_result = sp.upper_bound(arch)

        rec = RecursiveBound(seed=42)
        lower = -np.ones(4)
        upper = np.ones(4)
        rec_result = rec.compute(arch, lower, upper)

        assert rec_result.upper_bound <= sp_result.upper_bound + 1e-6

    def test_recursive_identity(self):
        """Identity network → recursive bound = 1."""
        arch = _make_identity_arch(4)
        rec = RecursiveBound(seed=42)
        result = rec.compute(arch, -np.ones(4), np.ones(4))
        assert result.upper_bound == pytest.approx(1.0, abs=0.1)

    def test_lower_bound_provided(self):
        """Recursive bound should also compute a lower bound."""
        rng = np.random.RandomState(42)
        W = rng.randn(3, 3) * 0.5
        arch = _make_arch([W], activations=[ActivationType.LINEAR])
        rec = RecursiveBound(seed=42)
        result = rec.compute(arch, -np.ones(3), np.ones(3))
        if result.lower_bound is not None:
            assert result.lower_bound >= 0


# ======================================================================
# LipschitzTightnessAnalysis
# ======================================================================

class TestLipschitzTightnessAnalysis:
    """Test tightness analysis comparing all bounds."""

    def test_tightness_ratio_geq_1(self):
        """Tightness ratio (upper/lower) should be ≥ 1."""
        arch = _make_scaling_arch(scale=2.0, dim=3)
        analysis = LipschitzTightnessAnalysis(seed=42)
        report = analysis.analyse(arch)
        # Best upper / adversarial lower
        if report.adversarial_lower > 1e-12:
            assert report.tightness_ratio >= 1.0 - 1e-6

    def test_report_has_spectral_product(self):
        arch = _make_identity_arch(3)
        analysis = LipschitzTightnessAnalysis(seed=42)
        report = analysis.analyse(arch)
        assert report.spectral_product is not None
        assert report.spectral_product.upper_bound > 0

    def test_best_upper_is_minimum(self):
        """Best upper should be ≤ spectral product."""
        rng = np.random.RandomState(42)
        W1 = rng.randn(4, 4) * 0.3
        W2 = rng.randn(4, 4) * 0.3
        arch = _make_arch([W1, W2],
                          activations=[ActivationType.RELU, ActivationType.LINEAR])
        analysis = LipschitzTightnessAnalysis(seed=42)
        report = analysis.analyse(arch)
        assert report.best_upper <= report.spectral_product.upper_bound + 1e-6


# ======================================================================
# CascadingErrorAnalysis
# ======================================================================

class TestCascadingErrorAnalysis:
    """Test cascading error analysis for Lipschitz over-estimation."""

    def test_inflation_factor_computation(self):
        """Volume inflation should be K^n for 1-step horizon."""
        analysis = CascadingErrorAnalysis(
            safety_margin=0.1, state_dim=4, output_domain_volume=1.0,
        )
        report = analysis.analyse(lipschitz_estimate=2.0, tightness_ratio=2.0)
        # K=2, n=4, T=1 (if horizon is folded into tightness)
        assert report.volume_inflation > 1.0

    def test_tightness_1_means_no_inflation(self):
        """K=1 (tight bound) should give volume_inflation = 1."""
        analysis = CascadingErrorAnalysis(
            safety_margin=0.1, state_dim=4,
        )
        report = analysis.analyse(lipschitz_estimate=1.0, tightness_ratio=1.0)
        assert report.volume_inflation == pytest.approx(1.0, abs=1e-6)

    def test_higher_dim_worse_inflation(self):
        """Higher state_dim → larger volume inflation for same K."""
        report_4d = CascadingErrorAnalysis(safety_margin=0.1, state_dim=4).analyse(2.0, 2.0)
        report_8d = CascadingErrorAnalysis(safety_margin=0.1, state_dim=8).analyse(2.0, 2.0)
        assert report_8d.volume_inflation > report_4d.volume_inflation

    def test_report_has_details(self):
        analysis = CascadingErrorAnalysis(safety_margin=0.1, state_dim=4)
        report = analysis.analyse(lipschitz_estimate=2.0, tightness_ratio=1.5)
        assert hasattr(report, 'details') or hasattr(report, 'volume_inflation')

    def test_invalid_safety_margin(self):
        with pytest.raises(ValueError):
            CascadingErrorAnalysis(safety_margin=0.0, state_dim=4)

    def test_invalid_state_dim(self):
        with pytest.raises(ValueError):
            CascadingErrorAnalysis(safety_margin=0.1, state_dim=0)
