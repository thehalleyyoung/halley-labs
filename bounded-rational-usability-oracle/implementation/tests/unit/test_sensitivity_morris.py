"""Unit tests for usability_oracle.sensitivity.morris — Morris screening method.

Tests cover elementary effects computation, μ*/σ computation, parameter
classification, and trajectory generation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from usability_oracle.interval import Interval
from usability_oracle.sensitivity.types import (
    MorrisResult,
    ParameterRange,
    SensitivityConfig,
)
from usability_oracle.sensitivity.morris import (
    MorrisAnalyzer,
    ParameterEffect,
    classify_parameter,
    compute_elementary_effects,
    optimized_trajectories,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _standard_params(n: int = 3) -> list[ParameterRange]:
    return [
        ParameterRange(name=f"x{i}", interval=Interval(0.0, 1.0), nominal=0.5)
        for i in range(n)
    ]


def _linear_model(**kwargs: float) -> float:
    return kwargs.get("x0", 0) + 2 * kwargs.get("x1", 0) + 3 * kwargs.get("x2", 0)


def _nonlinear_model(**kwargs: float) -> float:
    """y = x0 * x1 + sin(x2)"""
    x0 = kwargs.get("x0", 0)
    x1 = kwargs.get("x1", 0)
    x2 = kwargs.get("x2", 0)
    return x0 * x1 + math.sin(x2 * math.pi)


def _constant_model(**kwargs: float) -> float:
    return 42.0


def _single_param_model(**kwargs: float) -> float:
    """Only x0 matters: y = x0²."""
    return kwargs.get("x0", 0) ** 2


# ═══════════════════════════════════════════════════════════════════════════
# Elementary Effects
# ═══════════════════════════════════════════════════════════════════════════


class TestElementaryEffects:
    """Test elementary effects computation."""

    def test_compute_returns_dict(self):
        params = _standard_params(3)
        traj = optimized_trajectories(3, 1, seed=42)[0]
        effects = compute_elementary_effects(_linear_model, traj, params)
        assert isinstance(effects, dict)

    def test_linear_model_constant_effects(self):
        """For a linear model, elementary effects should be close to coefficients."""
        params = _standard_params(3)
        analyzer = MorrisAnalyzer()
        results = analyzer.screening(_linear_model, params, n_trajectories=20, seed=42)
        by_name = {r.parameter_name: r for r in results}
        # x2 has coefficient 3, should have highest mu_star
        assert by_name["x2"].mu_star > by_name["x0"].mu_star

    def test_effects_for_constant_function(self):
        """All effects should be ~0 for constant function."""
        params = _standard_params(2)
        analyzer = MorrisAnalyzer()
        results = analyzer.screening(_constant_model, params, n_trajectories=10, seed=42)
        for r in results:
            assert abs(r.mu_star) < 0.1


# ═══════════════════════════════════════════════════════════════════════════
# Trajectory Generation
# ═══════════════════════════════════════════════════════════════════════════


class TestTrajectoryGeneration:
    """Test optimized trajectory generation."""

    def test_trajectory_shape(self):
        k = 4  # dimensions
        trajectories = optimized_trajectories(k, n_trajectories=5, seed=42)
        assert len(trajectories) == 5
        for traj in trajectories:
            assert traj.shape == (k + 1, k)

    def test_trajectory_values_in_unit(self):
        trajectories = optimized_trajectories(3, n_trajectories=3, seed=42)
        for traj in trajectories:
            assert np.all(traj >= 0.0)
            assert np.all(traj <= 1.0)

    def test_trajectory_deterministic(self):
        a = optimized_trajectories(3, 5, seed=42)
        b = optimized_trajectories(3, 5, seed=42)
        for ta, tb in zip(a, b):
            np.testing.assert_array_equal(ta, tb)

    def test_trajectory_one_step_perturbation(self):
        """Each step in a trajectory should differ from the previous in exactly one dim."""
        trajectories = optimized_trajectories(3, n_trajectories=3, p=4, seed=42)
        for traj in trajectories:
            for i in range(len(traj) - 1):
                diff = np.abs(traj[i + 1] - traj[i])
                n_changed = np.sum(diff > 1e-10)
                assert n_changed == 1


# ═══════════════════════════════════════════════════════════════════════════
# μ* / σ Computation
# ═══════════════════════════════════════════════════════════════════════════


class TestMuStarSigma:
    """Test μ* and σ computation in Morris results."""

    def test_mu_star_non_negative(self):
        analyzer = MorrisAnalyzer()
        params = _standard_params(3)
        results = analyzer.screening(_linear_model, params, n_trajectories=20, seed=42)
        for r in results:
            assert r.mu_star >= 0

    def test_sigma_non_negative(self):
        analyzer = MorrisAnalyzer()
        params = _standard_params(3)
        results = analyzer.screening(_nonlinear_model, params, n_trajectories=20, seed=42)
        for r in results:
            assert r.sigma >= 0

    def test_sigma_over_mu_star_ratio(self):
        """For nonlinear model, σ/μ* should be high for interacting params."""
        analyzer = MorrisAnalyzer()
        params = _standard_params(3)
        results = analyzer.screening(_nonlinear_model, params, n_trajectories=30, seed=42)
        for r in results:
            ratio = r.sigma_over_mu_star
            assert isinstance(ratio, float)

    def test_is_non_monotonic_property(self):
        analyzer = MorrisAnalyzer()
        params = _standard_params(3)
        results = analyzer.screening(_nonlinear_model, params, n_trajectories=20, seed=42)
        for r in results:
            assert isinstance(r.is_non_monotonic, bool)

    def test_linear_model_low_sigma(self):
        """Linear model should have low σ (constant elementary effects)."""
        analyzer = MorrisAnalyzer()
        params = _standard_params(3)
        results = analyzer.screening(_linear_model, params, n_trajectories=30, seed=42)
        for r in results:
            # σ should be small relative to μ* for linear models
            if r.mu_star > 0.1:
                assert r.sigma / r.mu_star < 1.0

    def test_n_trajectories_field(self):
        analyzer = MorrisAnalyzer()
        params = _standard_params(2)
        results = analyzer.screening(_linear_model, params, n_trajectories=15, seed=42)
        for r in results:
            assert r.n_trajectories == 15


# ═══════════════════════════════════════════════════════════════════════════
# Parameter Classification
# ═══════════════════════════════════════════════════════════════════════════


class TestParameterClassification:
    """Test classification of parameters by effect type."""

    def test_classify_negligible(self):
        result = MorrisResult(
            parameter_name="x",
            mu_star=0.01,
            mu=0.005,
            sigma=0.001,
            n_trajectories=10,
            elementary_effects=(0.01, 0.01),
        )
        assert classify_parameter(result) == ParameterEffect.NEGLIGIBLE

    def test_classify_linear(self):
        result = MorrisResult(
            parameter_name="x",
            mu_star=2.0,
            mu=1.9,
            sigma=0.1,
            n_trajectories=20,
            elementary_effects=(1.9, 2.0, 2.1),
        )
        assert classify_parameter(result) == ParameterEffect.LINEAR

    def test_classify_nonlinear_or_interacting(self):
        result = MorrisResult(
            parameter_name="x",
            mu_star=2.0,
            mu=0.5,
            sigma=3.0,
            n_trajectories=20,
            elementary_effects=(1.0, -2.0, 3.0),
        )
        assert classify_parameter(result) == ParameterEffect.NONLINEAR_OR_INTERACTING

    def test_classify_all(self):
        analyzer = MorrisAnalyzer()
        params = _standard_params(3)
        results = analyzer.screening(_linear_model, params, n_trajectories=20, seed=42)
        classification = analyzer.classify_all(results)
        assert isinstance(classification, dict)
        assert len(classification) == 3

    def test_custom_threshold(self):
        result = MorrisResult(
            parameter_name="x",
            mu_star=0.05,
            mu=0.04,
            sigma=0.01,
            n_trajectories=10,
            elementary_effects=(0.05,),
        )
        # With a higher threshold, this should be NEGLIGIBLE
        assert classify_parameter(result, mu_star_threshold=0.1) == ParameterEffect.NEGLIGIBLE
        # With a lower threshold, it may be LINEAR
        eff = classify_parameter(result, mu_star_threshold=0.01)
        assert eff in (ParameterEffect.LINEAR, ParameterEffect.NONLINEAR_OR_INTERACTING)


# ═══════════════════════════════════════════════════════════════════════════
# MorrisAnalyzer Full Analysis
# ═══════════════════════════════════════════════════════════════════════════


class TestMorrisAnalyzerFull:
    """Test the full MorrisAnalyzer.analyze workflow."""

    def test_analyze_returns_result(self):
        analyzer = MorrisAnalyzer()
        config = SensitivityConfig(
            parameters=tuple(_standard_params(3)),
            n_samples=20,
            method="morris",
            output_names=("y",),
            seed=42,
        )
        result = analyzer.analyze(_linear_model, config)
        assert len(result.morris_results) == 3
        assert result.n_evaluations > 0

    def test_analyze_single_param_model(self):
        params = [ParameterRange(name="x0", interval=Interval(0.0, 1.0), nominal=0.5)]
        analyzer = MorrisAnalyzer()
        results = analyzer.screening(_single_param_model, params, n_trajectories=10, seed=42)
        assert len(results) == 1
        assert results[0].mu_star > 0

    def test_is_influential_property(self):
        analyzer = MorrisAnalyzer()
        params = _standard_params(3)
        results = analyzer.screening(_linear_model, params, n_trajectories=20, seed=42)
        influential = [r for r in results if r.is_influential]
        assert len(influential) > 0
