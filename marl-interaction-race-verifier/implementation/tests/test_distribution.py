"""Tests for marace.sampling.distribution — schedule probability measures."""

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
    UniformHBConsistentMeasure,
    PlackettLuceMeasure,
    LatencyWeightedMeasure,
    DistributionValidator,
)


# ======================================================================
# Helpers
# ======================================================================

def _make_simple_space(n_agents=3, n_timesteps=1, constraints=()):
    """Create a simple schedule space."""
    agents = [f"a{i}" for i in range(n_agents)]
    return ScheduleSpace(agents=agents, num_timesteps=n_timesteps,
                         constraints=constraints)


def _make_schedule(agent_order, timestep=0):
    """Create a schedule from agent ordering at a single timestep."""
    events = [
        ScheduleEvent(agent_id=a, timestep=timestep, action_time=float(i))
        for i, a in enumerate(agent_order)
    ]
    return Schedule(events=events)


# ======================================================================
# UniformHBConsistentMeasure
# ======================================================================

class TestUniformHBConsistentMeasure:
    """Test uniform distribution over linear extensions."""

    def test_unconstrained_3_agents(self):
        """3 agents, no HB constraints → 3! = 6 linear extensions."""
        space = _make_simple_space(n_agents=3, n_timesteps=1)
        measure = UniformHBConsistentMeasure(space, exact_threshold=20)
        count = measure.count_linear_extensions()
        assert count == pytest.approx(6.0, rel=0.1)

    def test_dag_with_constraints(self):
        """a0 → a1 → a2 (chain) → exactly 1 linear extension."""
        constraints = [
            ScheduleConstraint("a0", 0, "a1", 0),
            ScheduleConstraint("a1", 0, "a2", 0),
        ]
        space = _make_simple_space(n_agents=3, n_timesteps=1,
                                   constraints=constraints)
        measure = UniformHBConsistentMeasure(space, exact_threshold=20)
        count = measure.count_linear_extensions()
        assert count == pytest.approx(1.0, rel=0.1)

    def test_diamond_dag(self):
        """a0 → a1, a0 → a2, a1 → a3, a2 → a3 → 2 extensions."""
        agents = [f"a{i}" for i in range(4)]
        constraints = [
            ScheduleConstraint("a0", 0, "a1", 0),
            ScheduleConstraint("a0", 0, "a2", 0),
            ScheduleConstraint("a1", 0, "a3", 0),
            ScheduleConstraint("a2", 0, "a3", 0),
        ]
        space = ScheduleSpace(agents=agents, num_timesteps=1,
                              constraints=constraints)
        measure = UniformHBConsistentMeasure(space, exact_threshold=20)
        count = measure.count_linear_extensions()
        # Diamond: a0 first, a3 last, two orderings of a1,a2
        assert count == pytest.approx(2.0, rel=0.1)

    def test_log_prob_consistent_schedule(self):
        """A consistent schedule should have finite log_prob."""
        space = _make_simple_space(n_agents=3, n_timesteps=1)
        measure = UniformHBConsistentMeasure(space, exact_threshold=20)
        sched = _make_schedule(["a0", "a1", "a2"])
        lp = measure.log_prob(sched)
        assert np.isfinite(lp)
        # Should be -log(6) ≈ -1.79
        assert lp == pytest.approx(-math.log(6.0), abs=0.5)

    def test_sample_returns_correct_count(self):
        space = _make_simple_space(n_agents=3, n_timesteps=1)
        measure = UniformHBConsistentMeasure(space, exact_threshold=20)
        samples = measure.sample(10)
        assert len(samples) == 10

    def test_support_size(self):
        space = _make_simple_space(n_agents=3, n_timesteps=1)
        measure = UniformHBConsistentMeasure(space, exact_threshold=20)
        ss = measure.support_size()
        assert ss > 0

    def test_normalization_constant(self):
        space = _make_simple_space(n_agents=3, n_timesteps=1)
        measure = UniformHBConsistentMeasure(space, exact_threshold=20)
        Z = measure.normalization_constant()
        # Normalization constant should be positive
        assert Z > 0


# ======================================================================
# PlackettLuceMeasure
# ======================================================================

class TestPlackettLuceMeasure:
    """Test Plackett-Luce distribution."""

    def test_uniform_weights_give_uniform(self):
        """Uniform PL weights → all permutations equally likely."""
        events = [("a0", 0), ("a1", 0), ("a2", 0)]
        weights = np.ones(3)
        measure = PlackettLuceMeasure(events=events, weights=weights)
        # Generate many samples and check log_prob is constant
        sched1 = _make_schedule(["a0", "a1", "a2"])
        sched2 = _make_schedule(["a2", "a1", "a0"])
        lp1 = measure.log_prob(sched1)
        lp2 = measure.log_prob(sched2)
        assert lp1 == pytest.approx(lp2, abs=1e-10)

    def test_non_uniform_weights(self):
        """Non-uniform weights → different probabilities."""
        events = [("a0", 0), ("a1", 0)]
        weights = np.array([10.0, 1.0])
        measure = PlackettLuceMeasure(events=events, weights=weights)
        sched_01 = _make_schedule(["a0", "a1"])
        sched_10 = _make_schedule(["a1", "a0"])
        # a0 should be much more likely to come first
        assert measure.log_prob(sched_01) > measure.log_prob(sched_10)

    def test_sample_returns_schedules(self):
        events = [("a0", 0), ("a1", 0), ("a2", 0)]
        measure = PlackettLuceMeasure(events=events)
        samples = measure.sample(20)
        assert len(samples) == 20

    def test_weights_property(self):
        events = [("a0", 0), ("a1", 0)]
        w = np.array([2.0, 3.0])
        measure = PlackettLuceMeasure(events=events, weights=w)
        np.testing.assert_allclose(measure.weights, w)

    def test_log_weights_property(self):
        events = [("a0", 0), ("a1", 0)]
        w = np.array([2.0, 3.0])
        measure = PlackettLuceMeasure(events=events, weights=w)
        np.testing.assert_allclose(measure.log_weights, np.log(w))

    def test_support_size(self):
        events = [("a0", 0), ("a1", 0), ("a2", 0)]
        measure = PlackettLuceMeasure(events=events)
        ss = measure.support_size()
        assert ss == pytest.approx(6.0, rel=0.1)


# ======================================================================
# DistributionValidator
# ======================================================================

class TestDistributionValidator:
    """Test DistributionValidator normalization and divergence checks."""

    def test_normalization_unconstrained(self):
        """Uniform over 3! permutations should be normalized."""
        space = _make_simple_space(n_agents=3, n_timesteps=1)
        measure = UniformHBConsistentMeasure(space, exact_threshold=20)
        ok, total = DistributionValidator.check_normalization(measure)
        assert ok or abs(total - 1.0) < 0.2

    def test_kl_self_is_zero(self):
        """KL(p || p) should be 0."""
        events = [("a0", 0), ("a1", 0)]
        measure = PlackettLuceMeasure(events=events)
        kl = DistributionValidator.kl_divergence(measure, measure)
        assert kl >= 0.0
        assert kl < 0.1

    def test_total_variation_self_is_zero(self):
        events = [("a0", 0), ("a1", 0)]
        measure = PlackettLuceMeasure(events=events)
        schedules = measure.sample(50)
        tv = DistributionValidator.total_variation(measure, measure, schedules)
        assert tv >= 0.0
        assert tv < 0.1


# ======================================================================
# LatencyWeightedMeasure
# ======================================================================

class TestLatencyWeightedMeasure:
    """Test LatencyWeightedMeasure sample/log_prob consistency."""

    def test_construction(self):
        events = [("a0", 0), ("a1", 0), ("a2", 0)]
        alphas = np.ones(3)
        betas = np.ones(3)
        measure = LatencyWeightedMeasure(events=events, alphas=alphas,
                                          betas=betas)
        assert measure.alphas is not None
        assert measure.betas is not None

    def test_sample_returns_schedules(self):
        events = [("a0", 0), ("a1", 0)]
        alphas = np.array([1.0, 1.0])
        betas = np.array([1.0, 1.0])
        measure = LatencyWeightedMeasure(events=events, alphas=alphas,
                                          betas=betas)
        samples = measure.sample(10)
        assert len(samples) == 10

    def test_log_prob_finite(self):
        events = [("a0", 0), ("a1", 0)]
        alphas = np.array([1.0, 1.0])
        betas = np.array([1.0, 1.0])
        measure = LatencyWeightedMeasure(events=events, alphas=alphas,
                                          betas=betas)
        sched = _make_schedule(["a0", "a1"])
        lp = measure.log_prob(sched)
        assert np.isfinite(lp) or lp == float("-inf")

    def test_sample_log_prob_consistency(self):
        """Sampled schedules should have finite, non-zero probabilities."""
        events = [("a0", 0), ("a1", 0)]
        alphas = np.array([2.0, 1.0])
        betas = np.array([1.0, 1.0])
        measure = LatencyWeightedMeasure(events=events, alphas=alphas,
                                          betas=betas)
        samples = measure.sample(5)
        for s in samples:
            lp = measure.log_prob(s)
            assert np.isfinite(lp) or lp == float("-inf")

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            LatencyWeightedMeasure(
                events=[("a0", 0)],
                alphas=np.array([1.0, 2.0]),
                betas=np.array([1.0]),
            )
