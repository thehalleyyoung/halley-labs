"""Tests for sampling module."""

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
    ScheduleGenerator,
    ScheduleDistance,
    ContinuousSchedule,
)
from marace.sampling.importance_sampling import (
    ImportanceSampler,
    SequentialImportanceSampler,
    UniformProposal,
    ImportanceWeights,
    EffectiveSampleSize,
    ConfidenceInterval,
    RaceProbabilityEstimator,
)
from marace.sampling.cross_entropy import (
    CrossEntropyOptimizer,
    ParametricProposal,
    EliteSampleSelector,
)
from marace.sampling.statistics import (
    SampleStatistics,
    ConfidenceBound,
    StratifiedEstimator,
)


def _make_schedule(agent_time_pairs):
    """Helper: build a Schedule from (agent_id, action_time) tuples."""
    events = [
        ScheduleEvent(agent_id=a, timestep=0, action_time=t)
        for a, t in agent_time_pairs
    ]
    return Schedule(events=events)


class TestSchedule:
    """Test schedule representation."""

    def test_schedule_creation(self):
        """Test creating a schedule."""
        s = _make_schedule([("agent_0", 0.0), ("agent_1", 0.1), ("agent_0", 0.2)])
        assert s.length == 3

    def test_schedule_agents(self):
        """Test extracting agents from schedule."""
        s = _make_schedule([("a", 0.0), ("b", 0.1), ("a", 0.2), ("c", 0.3)])
        assert set(s.agents) == {"a", "b", "c"}

    def test_schedule_ordering(self):
        """Test schedule respects time ordering."""
        s = _make_schedule([("a", 0.0), ("b", 0.1), ("c", 0.2)])
        times = list(s.action_times())
        assert times == sorted(times)


class TestScheduleSpace:
    """Test schedule space."""

    def test_space_creation(self):
        """Test creating schedule space."""
        space = ScheduleSpace(
            agents=["a", "b", "c"],
            num_timesteps=5,
        )
        assert space.num_agents == 3

    def test_schedule_constraint(self):
        """Test schedule constraint."""
        constraint = ScheduleConstraint(
            before_agent="a",
            before_timestep=0,
            after_agent="b",
            after_timestep=0,
        )
        s = Schedule(events=[
            ScheduleEvent(agent_id="a", timestep=0, action_time=0.0),
            ScheduleEvent(agent_id="b", timestep=0, action_time=0.1),
        ])
        assert constraint.is_satisfied(s)


class TestScheduleGenerator:
    """Test schedule generation."""

    def test_uniform_generation(self):
        """Test uniform schedule generation."""
        space = ScheduleSpace(agents=["a", "b"], num_timesteps=2)
        gen = ScheduleGenerator(space, rng=np.random.RandomState(42))
        schedules = gen.sample_uniform(n_samples=10)
        assert len(schedules) == 10
        for s in schedules:
            assert s.length == 4  # 2 agents * 2 timesteps

    def test_constrained_generation(self):
        """Test generating schedules with constraints."""
        constraints = [ScheduleConstraint("a", 0, "b", 0)]
        space = ScheduleSpace(agents=["a", "b", "c"], num_timesteps=2, constraints=constraints)
        gen = ScheduleGenerator(space, rng=np.random.RandomState(42))
        schedules = gen.sample_uniform(n_samples=10)
        assert len(schedules) == 10
        for s in schedules:
            assert s.validate(constraints)


class TestScheduleDistance:
    """Test schedule distance metrics."""

    def test_kendall_tau(self):
        """Test Kendall tau distance."""
        s1 = _make_schedule([("a", 0.0), ("b", 0.1), ("c", 0.2)])
        s2 = _make_schedule([("a", 0.0), ("c", 0.1), ("b", 0.2)])
        d = ScheduleDistance.kendall_tau(s1, s2)
        assert d >= 0

    def test_same_schedule_distance_zero(self):
        """Test distance to self is zero."""
        s = _make_schedule([("a", 0.0), ("b", 0.1)])
        d = ScheduleDistance.kendall_tau(s, s)
        assert d == 0


class TestImportanceSampling:
    """Test importance sampling."""

    def test_uniform_proposal(self):
        """Test uniform proposal distribution."""
        space = ScheduleSpace(agents=["a", "b"], num_timesteps=2)
        proposal = UniformProposal(space)
        rng = np.random.RandomState(42)
        samples = proposal.sample(10, rng)
        assert len(samples) == 10

    def test_importance_weights(self):
        """Test importance weight computation."""
        log_weights = np.log(np.array([0.1, 0.2, 0.3, 0.4]) / np.array([0.25, 0.25, 0.25, 0.25]))
        iw = ImportanceWeights(log_weights=log_weights)
        w = iw.normalised_weights()
        assert len(w) == 4
        assert np.all(w > 0)
        assert np.isclose(w.sum(), 1.0)

    def test_effective_sample_size(self):
        """Test ESS computation."""
        # Equal weights -> ESS = N
        log_weights = np.zeros(100)
        iw = ImportanceWeights(log_weights=log_weights)
        n_eff = EffectiveSampleSize.compute(iw)
        assert np.isclose(n_eff, 100.0)
        # All weight on one sample -> ESS = 1
        log_weights = np.full(100, -1e10)
        log_weights[0] = 0.0
        iw = ImportanceWeights(log_weights=log_weights)
        n_eff = EffectiveSampleSize.compute(iw)
        assert np.isclose(n_eff, 1.0)

    def test_confidence_interval(self):
        """Test confidence interval computation."""
        np.random.seed(42)
        race_indicators = (np.random.rand(1000) < 0.3).astype(float)
        log_weights = np.zeros(1000)
        iw = ImportanceWeights(log_weights=log_weights)
        ci = ConfidenceInterval.from_importance_samples(
            race_indicators, iw, confidence_level=0.95
        )
        assert ci.lower < ci.upper
        assert ci.lower <= ci.estimate <= ci.upper


class TestCrossEntropy:
    """Test cross-entropy optimization."""

    def test_elite_selection(self):
        """Test elite sample selection."""
        selector = EliteSampleSelector(elite_fraction=0.1)
        space = ScheduleSpace(agents=["a", "b"], num_timesteps=1)
        gen = ScheduleGenerator(space, rng=np.random.RandomState(42))
        schedules = gen.sample_uniform(n_samples=100)
        scores = np.random.RandomState(42).randn(100)
        elite_schedules, elite_scores, threshold = selector.select(schedules, scores)
        assert len(elite_schedules) == 10

    def test_parametric_proposal(self):
        """Test parametric proposal."""
        proposal = ParametricProposal(
            agent_order=["a", "b", "c", "d"],
            means=np.zeros(4),
            stds=np.ones(4),
        )
        rng = np.random.RandomState(42)
        samples = proposal.sample(50, rng)
        assert len(samples) == 50

    def test_proposal_copy(self):
        """Test proposal copy preserves parameters."""
        proposal = ParametricProposal(
            agent_order=["a", "b"],
            means=np.array([1.0, 2.0]),
            stds=np.array([0.5, 0.5]),
        )
        copy = proposal.copy()
        assert np.allclose(copy.means, proposal.means)
        assert np.allclose(copy.stds, proposal.stds)


class TestStatistics:
    """Test statistical analysis."""

    def test_sample_statistics(self):
        """Test basic sample statistics."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        log_weights = np.zeros(5)
        iw = ImportanceWeights(log_weights=log_weights)
        mean = SampleStatistics.weighted_mean(values, iw)
        var = SampleStatistics.weighted_variance(values, iw)
        assert np.isclose(mean, 3.0)
        assert var > 0

    def test_hoeffding_bound(self):
        """Test Hoeffding confidence bound."""
        n_eff = 1000
        delta = 0.05
        bound = ConfidenceBound.hoeffding(n_eff=n_eff, delta=delta, value_range=1.0)
        assert bound > 0
        assert bound < 1.0

    def test_bernstein_bound(self):
        """Test Bernstein confidence bound."""
        n_eff = 1000
        delta = 0.05
        bound = ConfidenceBound.bernstein(n_eff=n_eff, sample_variance=0.01, delta=delta, value_range=1.0)
        assert bound > 0
        # With small variance, Bernstein should be tighter than Hoeffding
        assert bound < ConfidenceBound.hoeffding(n_eff=n_eff, delta=delta, value_range=1.0)

    def test_stratified_estimator(self):
        """Test stratified estimator."""
        stratum_probs = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        estimator = StratifiedEstimator(stratum_probs=stratum_probs)
        assert estimator.num_strata == 3
        allocation = estimator.allocate_samples(300)
        assert allocation.sum() == 300
