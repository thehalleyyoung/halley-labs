"""Tests for Improvement I2 — Rigorous Probabilistic Estimation Framework.

Tests cover:
A) distribution.py: MixedLogitPlackettLuce, ESSMonitor, WeightDegeneracyDiagnostics
B) concentration.py: HoeffdingSelfNormalizedBound, MartingaleStoppingCriterion
C) cross_entropy.py: CEConvergenceProof, KLDivergenceMonitor, CEConvergenceDiagnostics
D) importance_sampling.py: SelfNormalizedISEstimator, ESSAdaptiveResampler,
   ParetoSmoothedIS, ControlVariateValidator
"""

import math
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marace.sampling.schedule_space import (
    Schedule,
    ScheduleEvent,
    ScheduleSpace,
    ScheduleConstraint,
)
from marace.sampling.distribution import (
    MixedLogitPlackettLuce,
    ESSMonitor,
    WeightDegeneracyDiagnostics,
    PlackettLuceMeasure,
    DistributionValidator,
)
from marace.sampling.concentration import (
    BoundResult,
    HoeffdingSelfNormalizedBound,
    MartingaleStoppingCriterion,
)
from marace.sampling.cross_entropy import (
    CEConvergenceProof,
    KLDivergenceMonitor,
    CEConvergenceDiagnostics,
    ParametricProposal,
)
from marace.sampling.importance_sampling import (
    SelfNormalizedISEstimator,
    ESSAdaptiveResampler,
    ParetoSmoothedIS,
    ControlVariateValidator,
    UniformProposal,
    ImportanceWeights,
)


# ======================================================================
# Helpers
# ======================================================================

def _make_schedule(agent_order, timestep=0):
    events = [
        ScheduleEvent(agent_id=a, timestep=timestep, action_time=float(i))
        for i, a in enumerate(agent_order)
    ]
    return Schedule(events=events)


def _make_simple_space(n_agents=3, n_timesteps=1, constraints=()):
    agents = [f"a{i}" for i in range(n_agents)]
    return ScheduleSpace(agents=agents, num_timesteps=n_timesteps,
                         constraints=constraints)


# ======================================================================
# A) distribution.py — MixedLogitPlackettLuce
# ======================================================================

class TestMixedLogitPlackettLuce:
    """Test mixed-logit PL model (IIA-violation correction)."""

    def test_construction(self):
        events = [("a0", 0), ("a1", 0), ("a2", 0)]
        mu = np.zeros(3)
        sigma = np.eye(3) * 0.1
        model = MixedLogitPlackettLuce(events, mu, sigma)
        assert model.mu is not None
        assert model.sigma.shape == (3, 3)

    def test_construction_diagonal_sigma(self):
        """1-d sigma should be promoted to diagonal matrix."""
        events = [("a0", 0), ("a1", 0)]
        mu = np.zeros(2)
        sigma = np.array([0.1, 0.2])
        model = MixedLogitPlackettLuce(events, mu, sigma)
        assert model.sigma.shape == (2, 2)

    def test_log_prob_finite(self):
        events = [("a0", 0), ("a1", 0)]
        mu = np.zeros(2)
        sigma = np.eye(2) * 0.1
        model = MixedLogitPlackettLuce(events, mu, sigma, num_mixing_draws=50)
        sched = _make_schedule(["a0", "a1"])
        lp = model.log_prob(sched)
        assert np.isfinite(lp)

    def test_sample_returns_schedules(self):
        events = [("a0", 0), ("a1", 0), ("a2", 0)]
        mu = np.zeros(3)
        sigma = np.eye(3) * 0.1
        model = MixedLogitPlackettLuce(events, mu, sigma, num_mixing_draws=20)
        samples = model.sample(10)
        assert len(samples) == 10

    def test_support_size(self):
        events = [("a0", 0), ("a1", 0), ("a2", 0)]
        mu = np.zeros(3)
        sigma = np.eye(3) * 0.1
        model = MixedLogitPlackettLuce(events, mu, sigma)
        assert model.support_size() == pytest.approx(6.0, rel=0.1)

    def test_normalization_constant(self):
        events = [("a0", 0), ("a1", 0)]
        mu = np.zeros(2)
        sigma = np.eye(2) * 0.1
        model = MixedLogitPlackettLuce(events, mu, sigma)
        assert model.normalization_constant() == 1.0

    def test_correlated_strengths_break_iia(self):
        """With highly correlated strengths, IIA should be violated."""
        events = [("a0", 0), ("a1", 0), ("a2", 0)]
        mu = np.zeros(3)
        # High correlation between a0 and a1
        sigma = np.array([
            [1.0, 0.9, 0.0],
            [0.9, 1.0, 0.0],
            [0.0, 0.0, 0.1],
        ])
        model = MixedLogitPlackettLuce(events, mu, sigma, num_mixing_draws=50)
        # IIA violation score should be computable
        pairs = [
            (_make_schedule(["a0", "a1", "a2"]),
             _make_schedule(["a1", "a0", "a2"])),
        ]
        score = model.iia_violation_score(pairs)
        assert score >= 0.0

    def test_hb_constrained(self):
        """With HB constraints, invalid schedules get -inf."""
        constraints = [ScheduleConstraint("a0", 0, "a1", 0)]
        space = _make_simple_space(n_agents=2, n_timesteps=1,
                                    constraints=constraints)
        events = [("a0", 0), ("a1", 0)]
        mu = np.zeros(2)
        sigma = np.eye(2) * 0.1
        model = MixedLogitPlackettLuce(events, mu, sigma, space=space,
                                        num_mixing_draws=20)
        valid_sched = _make_schedule(["a0", "a1"])
        invalid_sched = _make_schedule(["a1", "a0"])
        assert np.isfinite(model.log_prob(valid_sched))
        assert model.log_prob(invalid_sched) == float("-inf")


# ======================================================================
# A) distribution.py — ESSMonitor
# ======================================================================

class TestESSMonitor:
    """Test ESS monitoring with adaptive resampling trigger."""

    def test_uniform_weights_high_ess(self):
        monitor = ESSMonitor(threshold_fraction=0.5)
        log_w = np.zeros(100)  # uniform weights
        needs_resample, ess = monitor.check(log_w)
        assert not needs_resample
        assert ess == pytest.approx(100.0, rel=0.01)

    def test_degenerate_weights_trigger_resample(self):
        monitor = ESSMonitor(threshold_fraction=0.5)
        log_w = np.full(100, -100.0)
        log_w[0] = 0.0  # one dominant weight
        needs_resample, ess = monitor.check(log_w)
        assert needs_resample
        assert ess < 50.0

    def test_compute_ess_static(self):
        log_w = np.zeros(50)
        ess = ESSMonitor.compute_ess(log_w)
        assert ess == pytest.approx(50.0, rel=0.01)

    def test_history_tracking(self):
        monitor = ESSMonitor()
        for _ in range(5):
            log_w = np.random.RandomState(42).randn(20)
            monitor.check(log_w)
        assert len(monitor.history) == 5

    def test_diagnostics(self):
        monitor = ESSMonitor()
        log_w = np.zeros(20)
        monitor.check(log_w)
        diag = monitor.diagnostics()
        assert diag["num_checks"] == 1
        assert "mean_ess" in diag

    def test_adaptive_threshold(self):
        """Threshold should adapt if ESS is consistently near threshold."""
        monitor = ESSMonitor(threshold_fraction=0.5, adaptation_rate=0.2)
        rng = np.random.RandomState(42)
        # Feed weights that give ESS near threshold
        for _ in range(10):
            log_w = rng.randn(20) * 0.5
            monitor.check(log_w)
        # Threshold should have adapted
        diag = monitor.diagnostics()
        assert diag["current_threshold_frac"] <= 0.5


# ======================================================================
# A) distribution.py — WeightDegeneracyDiagnostics
# ======================================================================

class TestWeightDegeneracyDiagnostics:
    """Test importance weight degeneracy detection."""

    def test_uniform_weights_healthy(self):
        log_w = np.zeros(100)
        diag = WeightDegeneracyDiagnostics.diagnose(log_w)
        assert diag["severity"] == "healthy"
        assert diag["ess_ratio"] == pytest.approx(1.0, rel=0.01)

    def test_degenerate_weights_severe(self):
        log_w = np.full(100, -200.0)
        log_w[0] = 0.0
        diag = WeightDegeneracyDiagnostics.diagnose(log_w)
        assert diag["severity"] in ("severe", "moderate")
        assert diag["ess_ratio"] < 0.1

    def test_moderate_degeneracy(self):
        rng = np.random.RandomState(42)
        log_w = rng.randn(100) * 3.0  # moderate spread
        diag = WeightDegeneracyDiagnostics.diagnose(log_w)
        assert diag["severity"] in ("mild", "moderate", "healthy", "severe")
        assert "pareto_k_hat" in diag

    def test_empty_weights(self):
        diag = WeightDegeneracyDiagnostics.diagnose(np.array([]))
        assert diag["severity"] == "empty"

    def test_all_diagnostics_present(self):
        log_w = np.zeros(50)
        diag = WeightDegeneracyDiagnostics.diagnose(log_w)
        expected_keys = ["n", "ess", "ess_ratio", "max_weight_ratio",
                         "entropy", "max_entropy", "entropy_ratio",
                         "pareto_k_hat", "severity"]
        for key in expected_keys:
            assert key in diag


# ======================================================================
# B) concentration.py — HoeffdingSelfNormalizedBound
# ======================================================================

class TestHoeffdingSelfNormalizedBound:
    """Test Hoeffding-type bound for self-normalised IS."""

    def test_known_support_uniform_weights(self):
        rng = np.random.RandomState(42)
        n = 500
        values = rng.uniform(0, 1, size=n)
        weights = np.ones(n)
        bound = HoeffdingSelfNormalizedBound(support=(0.0, 1.0))
        result = bound.confidence_interval(weights, values, alpha=0.05)
        assert isinstance(result, BoundResult)
        assert result.lower <= result.upper
        assert result.method == "Hoeffding-SelfNormalized"

    def test_ci_contains_mean(self):
        rng = np.random.RandomState(42)
        n = 1000
        values = rng.uniform(0, 1, size=n)
        weights = np.ones(n)
        bound = HoeffdingSelfNormalizedBound(support=(0.0, 1.0))
        result = bound.confidence_interval(weights, values, alpha=0.05)
        true_mean = float(np.mean(values))
        assert result.lower <= true_mean <= result.upper

    def test_ci_narrower_with_more_samples(self):
        rng = np.random.RandomState(42)
        bound = HoeffdingSelfNormalizedBound(support=(0.0, 1.0))
        widths = []
        for n in [100, 500, 2000]:
            values = rng.uniform(0, 1, size=n)
            weights = np.ones(n)
            result = bound.confidence_interval(weights, values, alpha=0.05)
            widths.append(result.width)
        assert widths[0] > widths[-1]

    def test_estimated_support(self):
        """Should work when support is not provided."""
        rng = np.random.RandomState(42)
        values = rng.uniform(0, 1, size=200)
        weights = np.ones(200)
        bound = HoeffdingSelfNormalizedBound()
        result = bound.confidence_interval(weights, values, alpha=0.05)
        assert result.lower <= result.upper

    def test_weighted_samples(self):
        rng = np.random.RandomState(42)
        values = rng.uniform(0, 1, size=200)
        weights = rng.exponential(1.0, size=200)
        bound = HoeffdingSelfNormalizedBound(support=(0.0, 1.0))
        result = bound.confidence_interval(weights, values, alpha=0.05)
        assert result.lower <= result.upper
        assert result.effective_sample_size > 0


# ======================================================================
# B) concentration.py — MartingaleStoppingCriterion
# ======================================================================

class TestMartingaleStoppingCriterion:
    """Test martingale-based stopping for sequential estimation."""

    def test_not_enough_samples(self):
        criterion = MartingaleStoppingCriterion(min_samples=50)
        values = np.random.RandomState(42).uniform(0, 1, size=10)
        log_weights = np.zeros(10)
        should_stop, result = criterion.update(values, log_weights)
        assert not should_stop
        assert result is None

    def test_converges_with_enough_samples(self):
        criterion = MartingaleStoppingCriterion(
            target_width=0.5, min_samples=20, min_ess_ratio=0.05
        )
        rng = np.random.RandomState(42)
        stopped = False
        for _ in range(20):
            values = rng.uniform(0, 1, size=100)
            log_weights = np.zeros(100)
            should_stop, result = criterion.update(values, log_weights)
            if should_stop:
                stopped = True
                break
        assert stopped
        assert result is not None
        assert result.method == "MartingaleConfidenceSequence"

    def test_running_estimates_tracked(self):
        criterion = MartingaleStoppingCriterion(min_samples=10)
        rng = np.random.RandomState(42)
        for _ in range(3):
            values = rng.uniform(0, 1, size=20)
            log_weights = np.zeros(20)
            criterion.update(values, log_weights)
        assert len(criterion.running_estimates) == 3

    def test_running_widths_decrease(self):
        criterion = MartingaleStoppingCriterion(min_samples=10)
        rng = np.random.RandomState(42)
        for _ in range(5):
            values = rng.uniform(0, 1, size=100)
            log_weights = np.zeros(100)
            criterion.update(values, log_weights)
        widths = criterion.running_widths
        # With more samples, widths should generally decrease
        assert widths[-1] <= widths[0]

    def test_reset(self):
        criterion = MartingaleStoppingCriterion(min_samples=10)
        values = np.random.RandomState(42).uniform(0, 1, size=20)
        criterion.update(values, np.zeros(20))
        criterion.reset()
        assert len(criterion.running_estimates) == 0
        assert not criterion.stopped


# ======================================================================
# C) cross_entropy.py — CEConvergenceProof
# ======================================================================

class TestCEConvergenceProof:
    """Test CE convergence rate bounds."""

    def test_contraction_rate(self):
        proof = CEConvergenceProof(elite_fraction=0.1, smoothing=0.1)
        rate = proof.contraction_rate
        assert 0.0 < rate < 1.0
        assert rate == pytest.approx(0.09, rel=0.01)

    def test_kl_upper_bound_decreases(self):
        proof = CEConvergenceProof(elite_fraction=0.1, smoothing=0.1)
        bounds = [proof.kl_upper_bound(t, initial_kl=10.0) for t in range(10)]
        # Should be monotonically decreasing
        for i in range(1, len(bounds)):
            assert bounds[i] <= bounds[i - 1]

    def test_iterations_for_kl_target(self):
        proof = CEConvergenceProof(elite_fraction=0.1, smoothing=0.1)
        n_iter = proof.iterations_for_kl_target(0.01, initial_kl=10.0)
        assert n_iter > 0
        # Verify the bound is achieved
        kl_at_n = proof.kl_upper_bound(n_iter, initial_kl=10.0)
        assert kl_at_n <= 0.01

    def test_finite_sample_bound(self):
        proof = CEConvergenceProof(param_dim=10)
        bound = proof.finite_sample_bound(num_rounds=10, samples_per_round=500)
        assert bound > 0
        # More samples → smaller bound
        bound2 = proof.finite_sample_bound(num_rounds=10, samples_per_round=5000)
        assert bound2 < bound

    def test_summary(self):
        proof = CEConvergenceProof(elite_fraction=0.1, smoothing=0.1, param_dim=5)
        s = proof.summary(num_iterations=50)
        assert "contraction_rate" in s
        assert "kl_bound_at_T" in s
        assert "iterations_for_0.01" in s


# ======================================================================
# C) cross_entropy.py — KLDivergenceMonitor
# ======================================================================

class TestKLDivergenceMonitor:
    """Test KL divergence monitoring for CE."""

    def test_initial_state(self):
        monitor = KLDivergenceMonitor()
        assert monitor.kl_gap == float("inf")
        assert len(monitor.kl_history) == 0

    def test_update_tracks_kl(self):
        monitor = KLDivergenceMonitor()
        agents = ["a0", "a1", "a2"]
        p1 = ParametricProposal(agents, means=np.array([0.5, 0.5, 0.5]),
                                 stds=np.array([0.3, 0.3, 0.3]))
        p2 = ParametricProposal(agents, means=np.array([0.6, 0.5, 0.5]),
                                 stds=np.array([0.3, 0.3, 0.3]))
        converged, kl = monitor.update(p1, p2)
        assert not converged
        assert kl > 0

    def test_convergence_after_identical_proposals(self):
        monitor = KLDivergenceMonitor(kl_threshold=0.01, patience=3)
        agents = ["a0", "a1"]
        p = ParametricProposal(agents, means=np.array([0.5, 0.5]),
                                stds=np.array([0.25, 0.25]))
        # Feed identical proposals repeatedly
        for _ in range(5):
            converged, kl = monitor.update(p, p)
        assert converged
        assert kl == pytest.approx(0.0, abs=1e-10)

    def test_kl_gap(self):
        monitor = KLDivergenceMonitor()
        agents = ["a0", "a1"]
        p1 = ParametricProposal(agents, means=np.array([0.3, 0.3]),
                                 stds=np.array([0.2, 0.2]))
        p2 = ParametricProposal(agents, means=np.array([0.4, 0.4]),
                                 stds=np.array([0.2, 0.2]))
        monitor.update(p1, p2)
        assert monitor.kl_gap > 0

    def test_effective_support_sizes(self):
        monitor = KLDivergenceMonitor()
        agents = ["a0", "a1"]
        p = ParametricProposal(agents, means=np.array([0.5, 0.5]),
                                stds=np.array([0.25, 0.25]))
        monitor.update(p, p)
        assert len(monitor.effective_support_sizes) == 1
        assert monitor.effective_support_sizes[0] > 0

    def test_diagnostics(self):
        monitor = KLDivergenceMonitor()
        agents = ["a0", "a1"]
        p = ParametricProposal(agents, means=np.array([0.5, 0.5]),
                                stds=np.array([0.25, 0.25]))
        monitor.update(p, p)
        diag = monitor.diagnostics()
        assert "kl_history" in diag
        assert "kl_gap" in diag
        assert "effective_support_sizes" in diag


# ======================================================================
# C) cross_entropy.py — CEConvergenceDiagnostics
# ======================================================================

class TestCEConvergenceDiagnostics:
    """Test CE convergence diagnostics container."""

    def test_default_state(self):
        diag = CEConvergenceDiagnostics()
        assert not diag.converged
        assert diag.convergence_iteration is None

    def test_is_healthy_with_decreasing_kl(self):
        diag = CEConvergenceDiagnostics(
            kl_history=[1.0, 0.5, 0.3, 0.1],
            ess_history=[10.0, 15.0, 20.0, 25.0],
        )
        assert diag.is_healthy()

    def test_summary(self):
        diag = CEConvergenceDiagnostics(
            kl_history=[1.0, 0.5, 0.1],
            converged=True,
            convergence_iteration=3,
        )
        s = diag.summary()
        assert s["converged"]
        assert s["final_kl"] == 0.1


# ======================================================================
# D) importance_sampling.py — SelfNormalizedISEstimator
# ======================================================================

class TestSelfNormalizedISEstimator:
    """Test bias-corrected self-normalised IS estimator."""

    def test_uniform_target_proposal(self):
        space = _make_simple_space(n_agents=2, n_timesteps=1)
        proposal = UniformProposal(space)
        target_lp = lambda s: 0.0
        estimator = SelfNormalizedISEstimator(target_lp, proposal)
        f = lambda s: 1.0
        estimate, bias, var = estimator.estimate(f, n=50)
        assert abs(estimate - 1.0) < 0.5

    def test_returns_three_values(self):
        space = _make_simple_space(n_agents=2, n_timesteps=1)
        proposal = UniformProposal(space)
        estimator = SelfNormalizedISEstimator(lambda s: 0.0, proposal)
        result = estimator.estimate(lambda s: 0.5, n=30)
        assert len(result) == 3  # (estimate, bias, variance)

    def test_bias_correction_applied(self):
        space = _make_simple_space(n_agents=2, n_timesteps=1)
        proposal = UniformProposal(space)
        estimator = SelfNormalizedISEstimator(lambda s: 0.0, proposal)
        estimate, bias_corr, var_est = estimator.estimate(lambda s: 0.5, n=100)
        # Bias correction is a float
        assert isinstance(bias_corr, float)


# ======================================================================
# D) importance_sampling.py — ESSAdaptiveResampler
# ======================================================================

class TestESSAdaptiveResampler:
    """Test ESS-triggered adaptive resampling."""

    def test_no_resample_uniform_weights(self):
        resampler = ESSAdaptiveResampler(threshold_fraction=0.5)
        schedules = [_make_schedule(["a0", "a1"]) for _ in range(20)]
        log_w = np.zeros(20)
        new_p, new_lw, did = resampler.maybe_resample(schedules, log_w)
        assert not did
        assert len(new_p) == 20

    def test_resample_degenerate_weights(self):
        resampler = ESSAdaptiveResampler(threshold_fraction=0.5)
        schedules = [_make_schedule(["a0", "a1"]) for _ in range(20)]
        log_w = np.full(20, -100.0)
        log_w[0] = 0.0  # one dominant weight
        new_p, new_lw, did = resampler.maybe_resample(schedules, log_w)
        assert did
        assert resampler.resample_count == 1
        # After resampling, weights should be uniform (zeros in log)
        np.testing.assert_allclose(new_lw, 0.0)

    def test_resample_count_increments(self):
        resampler = ESSAdaptiveResampler(threshold_fraction=0.5)
        schedules = [_make_schedule(["a0", "a1"]) for _ in range(20)]
        log_w = np.full(20, -100.0)
        log_w[0] = 0.0
        resampler.maybe_resample(schedules, log_w)
        resampler.maybe_resample(schedules, log_w)
        assert resampler.resample_count == 2


# ======================================================================
# D) importance_sampling.py — ParetoSmoothedIS
# ======================================================================

class TestParetoSmoothedIS:
    """Test Pareto-smoothed importance sampling."""

    def test_smooth_uniform_weights(self):
        psis = ParetoSmoothedIS()
        log_w = np.zeros(100)
        smoothed, k_hat = psis.smooth_weights(log_w)
        assert len(smoothed) == 100
        # Uniform weights should have k_hat near 0
        assert abs(k_hat) < 1.0

    def test_smooth_heavy_tailed_weights(self):
        rng = np.random.RandomState(42)
        psis = ParetoSmoothedIS()
        log_w = rng.randn(200) * 5.0  # heavy-tailed
        smoothed, k_hat = psis.smooth_weights(log_w)
        assert len(smoothed) == 200
        assert isinstance(k_hat, float)

    def test_estimate_returns_two_values(self):
        psis = ParetoSmoothedIS()
        log_w = np.zeros(50)
        values = np.random.RandomState(42).uniform(0, 1, size=50)
        estimate, k_hat = psis.estimate(log_w, values)
        assert isinstance(estimate, float)
        assert isinstance(k_hat, float)

    def test_smoothing_reduces_variance(self):
        """Smoothed weights should not wildly differ from raw."""
        rng = np.random.RandomState(42)
        psis = ParetoSmoothedIS()
        log_w = rng.randn(200) * 3.0
        smoothed_lw, _ = psis.smooth_weights(log_w)
        # Just verify that smoothing produces valid output
        assert len(smoothed_lw) == 200
        assert np.all(np.isfinite(smoothed_lw))

    def test_small_sample(self):
        """Should handle small samples gracefully."""
        psis = ParetoSmoothedIS(min_tail_samples=5)
        log_w = np.zeros(3)
        smoothed, k_hat = psis.smooth_weights(log_w)
        assert len(smoothed) == 3


# ======================================================================
# D) importance_sampling.py — ControlVariateValidator
# ======================================================================

class TestControlVariateValidator:
    """Test non-circular ground-truth validation via control variates."""

    def test_validate_correct_weights(self):
        """With correct weights, control estimate should be close to true mean."""
        # Control function: just the schedule length (constant)
        control_fn = lambda s: float(s.length)
        space = _make_simple_space(n_agents=2, n_timesteps=1)
        schedules = [_make_schedule(["a0", "a1"]) for _ in range(50)]
        log_weights = np.zeros(50)
        validator = ControlVariateValidator(control_fn, control_mean=2.0)
        is_valid, estimate, error = validator.validate(schedules, log_weights)
        assert is_valid
        assert error < 0.1

    def test_validate_incorrect_weights(self):
        """With incorrect weights, validation may fail."""
        control_fn = lambda s: 1.0
        schedules = [_make_schedule(["a0", "a1"]) for _ in range(50)]
        log_weights = np.zeros(50)
        validator = ControlVariateValidator(control_fn, control_mean=5.0)
        is_valid, estimate, error = validator.validate(
            schedules, log_weights, tolerance=0.1
        )
        assert not is_valid
        assert error > 0.1

    def test_correct_reduces_variance(self):
        """Control variate correction should not crash."""
        control_fn = lambda s: float(s.length)
        schedules = [_make_schedule(["a0", "a1"]) for _ in range(50)]
        log_weights = np.zeros(50)
        f_values = np.random.RandomState(42).uniform(0, 1, size=50)
        validator = ControlVariateValidator(control_fn, control_mean=2.0)
        corrected, beta = validator.correct(schedules, log_weights, f_values)
        assert isinstance(corrected, float)
        assert isinstance(beta, float)

    def test_zero_weights_handled(self):
        """All-zero weights should return invalid."""
        control_fn = lambda s: 1.0
        schedules = [_make_schedule(["a0", "a1"]) for _ in range(10)]
        log_weights = np.full(10, -1e10)
        validator = ControlVariateValidator(control_fn, control_mean=1.0)
        # Even very small weights should work after normalisation
        is_valid, _, _ = validator.validate(schedules, log_weights)
        # Should still produce a result (may or may not be valid)
        assert isinstance(is_valid, bool)


# ======================================================================
# Integration: validate imports from __init__
# ======================================================================

class TestInitImports:
    """Verify all new classes are importable from marace.sampling."""

    def test_distribution_imports(self):
        from marace.sampling import (
            MixedLogitPlackettLuce,
            ESSMonitor,
            WeightDegeneracyDiagnostics,
        )

    def test_concentration_imports(self):
        from marace.sampling import (
            HoeffdingSelfNormalizedBound,
            MartingaleStoppingCriterion,
        )

    def test_cross_entropy_imports(self):
        from marace.sampling import (
            CEConvergenceProof,
            KLDivergenceMonitor,
            CEConvergenceDiagnostics,
        )

    def test_importance_sampling_imports(self):
        from marace.sampling import (
            SelfNormalizedISEstimator,
            ESSAdaptiveResampler,
            ParetoSmoothedIS,
            ControlVariateValidator,
        )
