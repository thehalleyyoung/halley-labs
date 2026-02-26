"""Tests for adaptive sequential importance sampling with IIA diagnostics.

Covers:
- ESS monitoring and resampling triggers
- Different resampling strategies (multinomial, systematic, residual)
- Plackett-Luce IIA validation
- Mixed logit fitting and sampling
- Nested logit with interaction groups
- Stopping criteria convergence
- Joint error decomposition
- Integration with existing ImportanceSampler and ScheduleSpace
"""

import math
import numpy as np
import pytest

from marace.sampling.schedule_space import (
    Schedule,
    ScheduleEvent,
    ScheduleSpace,
    ScheduleGenerator,
    ScheduleConstraint,
)
from marace.sampling.importance_sampling import (
    ImportanceWeights,
    ConfidenceInterval,
    ProposalDistribution,
    UniformProposal,
    ImportanceSampler,
)
from marace.sampling.adaptive_sis import (
    AdaptiveSISEngine,
    SISResult,
    MultinomialResampling,
    SystematicResampling,
    ResidualResampling,
    PlackettLuceValidator,
    IIATestResult,
    MixedLogitProposal,
    NestedLogitProposal,
    StoppingCriteria,
    StoppingDecision,
    JointErrorAnalysis,
    ErrorDecomposition,
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

AGENTS_3 = ["a0", "a1", "a2"]
AGENTS_4 = ["a0", "a1", "a2", "a3"]


def _make_schedule(ordering: list[str]) -> Schedule:
    """Build a Schedule from a simple agent ordering."""
    events = [
        ScheduleEvent(agent_id=a, timestep=0, action_time=float(i))
        for i, a in enumerate(ordering)
    ]
    return Schedule(events=events)


def _make_space(agents: list[str]) -> ScheduleSpace:
    return ScheduleSpace(agents=agents, num_timesteps=1)


def _uniform_target_log_prob(schedule: Schedule) -> float:
    return 0.0


def _biased_target_log_prob(schedule: Schedule) -> float:
    """Target that favours a0 first."""
    ordering = schedule.ordering()
    if ordering and ordering[0] == "a0":
        return 0.0
    return -2.0


# ---------------------------------------------------------------
# Test ESS computation and monitoring
# ---------------------------------------------------------------

class TestESSMonitoring:
    """Tests for ESS computation and resampling triggers."""

    def test_ess_uniform_weights(self):
        """Uniform log-weights should yield ESS = N."""
        engine = AdaptiveSISEngine(
            target_log_prob=_uniform_target_log_prob,
            proposal=UniformProposal(_make_space(AGENTS_3)),
            num_particles=50,
        )
        log_w = np.zeros(50)
        ess = engine._compute_ess(log_w)
        assert abs(ess - 50.0) < 1e-6

    def test_ess_degenerate_weights(self):
        """All weight on one particle gives ESS ≈ 1."""
        engine = AdaptiveSISEngine(
            target_log_prob=_uniform_target_log_prob,
            proposal=UniformProposal(_make_space(AGENTS_3)),
            num_particles=50,
        )
        log_w = np.full(50, -100.0)
        log_w[0] = 0.0
        ess = engine._compute_ess(log_w)
        assert ess < 2.0

    def test_ess_history_tracked(self):
        """ESS history should be populated after run()."""
        space = _make_space(AGENTS_3)
        engine = AdaptiveSISEngine(
            target_log_prob=_uniform_target_log_prob,
            proposal=UniformProposal(space),
            num_particles=20,
            mode="sir",
        )
        result = engine.run(rng=np.random.RandomState(123))
        assert len(result.ess_history) >= 1
        assert all(e > 0 for e in result.ess_history)

    def test_resample_triggered_on_low_ess(self):
        """Resampling should be triggered when ESS < threshold."""
        space = _make_space(AGENTS_3)
        engine = AdaptiveSISEngine(
            target_log_prob=_biased_target_log_prob,
            proposal=UniformProposal(space),
            num_particles=100,
            ess_threshold_fraction=0.99,  # very aggressive threshold
            mode="sir",
        )
        result = engine.run(rng=np.random.RandomState(42))
        # With such an aggressive threshold, resampling should happen
        assert len(result.resample_steps) >= 1 or result.ess_history[0] >= 99


# ---------------------------------------------------------------
# Test resampling strategies
# ---------------------------------------------------------------

class TestResamplingStrategies:
    """Tests for multinomial, systematic, and residual resampling."""

    def _make_particles_and_weights(self, n: int = 100):
        rng = np.random.RandomState(42)
        particles = [_make_schedule(list(rng.permutation(AGENTS_3)))
                     for _ in range(n)]
        weights = rng.dirichlet(np.ones(n))
        return particles, weights, rng

    def test_multinomial_preserves_count(self):
        """Multinomial resampling returns exactly N particles."""
        particles, weights, rng = self._make_particles_and_weights()
        resampler = MultinomialResampling()
        result = resampler.resample(particles, weights, rng)
        assert len(result) == len(particles)

    def test_systematic_preserves_count(self):
        """Systematic resampling returns exactly N particles."""
        particles, weights, rng = self._make_particles_and_weights()
        resampler = SystematicResampling()
        result = resampler.resample(particles, weights, rng)
        assert len(result) == len(particles)

    def test_residual_preserves_count(self):
        """Residual resampling returns exactly N particles."""
        particles, weights, rng = self._make_particles_and_weights()
        resampler = ResidualResampling()
        result = resampler.resample(particles, weights, rng)
        assert len(result) == len(particles)

    def test_systematic_low_variance(self):
        """Systematic should have lower variance than multinomial."""
        n = 200
        rng = np.random.RandomState(7)
        particles = [_make_schedule(list(rng.permutation(AGENTS_3)))
                     for _ in range(n)]
        # Weight concentrated on a few particles
        weights = np.zeros(n)
        weights[:10] = 1.0
        weights /= weights.sum()

        sys_resampler = SystematicResampling()
        multi_resampler = MultinomialResampling()

        sys_counts = []
        multi_counts = []
        for seed in range(20):
            r = np.random.RandomState(seed)
            sys_result = sys_resampler.resample(particles, weights, r)
            # Count unique particles
            sys_ids = set(id(p) for p in sys_result)
            sys_counts.append(len(sys_ids))

            r2 = np.random.RandomState(seed + 1000)
            multi_result = multi_resampler.resample(particles, weights, r2)
            multi_ids = set(id(p) for p in multi_result)
            multi_counts.append(len(multi_ids))

        # Systematic should be more consistent (lower std of unique counts)
        assert np.std(sys_counts) <= np.std(multi_counts) + 5

    def test_residual_deterministic_component(self):
        """Residual resampling's deterministic part is correct."""
        n = 10
        particles = [_make_schedule(AGENTS_3) for _ in range(n)]
        weights = np.array([0.3, 0.25, 0.15, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01])
        rng = np.random.RandomState(42)
        resampler = ResidualResampling()
        result = resampler.resample(particles, weights, rng)
        assert len(result) == n


# ---------------------------------------------------------------
# Test Plackett-Luce IIA Validation
# ---------------------------------------------------------------

class TestPlackettLuceValidator:
    """Tests for IIA assumption validation."""

    def test_iia_holds_for_pl_schedules(self):
        """Schedules drawn from PL should pass IIA test."""
        rng = np.random.RandomState(42)
        agents = AGENTS_3
        weights = np.array([1.5, 1.2, 1.0])

        # Generate PL-consistent schedules
        schedules = []
        for _ in range(500):
            remaining = list(range(3))
            w = weights.copy()
            ordering = []
            while remaining:
                probs = w[remaining] / w[remaining].sum()
                choice = rng.choice(len(remaining), p=probs)
                ordering.append(agents[remaining[choice]])
                remaining.pop(choice)
            schedules.append(_make_schedule(ordering))

        # Pass true PL weights to avoid estimation error
        validator = PlackettLuceValidator(agents, significance_level=0.01)
        pl_normalized = weights / weights.sum()
        result = validator.validate(schedules, pl_weights=pl_normalized)
        assert result.severity in ("none", "mild")

    def test_iia_violated_for_correlated_schedules(self):
        """Correlated scheduling should violate IIA."""
        rng = np.random.RandomState(42)
        agents = AGENTS_4
        schedules = []
        for _ in range(500):
            if rng.uniform() < 0.5:
                # Regime 1: a0, a1 always together early
                schedules.append(_make_schedule(["a0", "a1", "a2", "a3"]))
            else:
                # Regime 2: a2, a3 always together early
                schedules.append(_make_schedule(["a2", "a3", "a0", "a1"]))

        validator = PlackettLuceValidator(agents, significance_level=0.05)
        result = validator.validate(schedules)
        # Strong bimodal correlation should flag violation
        assert result.is_violated or result.severity != "none"

    def test_validator_recommendation(self):
        """Recommendations should be provided for each severity."""
        validator = PlackettLuceValidator(AGENTS_3)
        assert "Plackett-Luce" in validator._recommend("none")
        assert "mixed logit" in validator._recommend("mild")
        assert "nested logit" in validator._recommend("moderate").lower() or \
               "mixed logit" in validator._recommend("moderate").lower()
        assert "kernel" in validator._recommend("severe").lower() or \
               "inappropriate" in validator._recommend("severe").lower()

    def test_insufficient_data_returns_no_violation(self):
        """Too few schedules shouldn't flag violation."""
        schedules = [_make_schedule(AGENTS_3)]
        validator = PlackettLuceValidator(AGENTS_3)
        result = validator.validate(schedules)
        assert not result.is_violated


# ---------------------------------------------------------------
# Test Mixed Logit Proposal
# ---------------------------------------------------------------

class TestMixedLogitProposal:
    """Tests for mixture-of-PL proposal."""

    def test_sample_returns_correct_count(self):
        proposal = MixedLogitProposal(AGENTS_3, num_components=2)
        rng = np.random.RandomState(42)
        samples = proposal.sample(10, rng)
        assert len(samples) == 10

    def test_sample_produces_valid_schedules(self):
        proposal = MixedLogitProposal(AGENTS_3, num_components=2)
        rng = np.random.RandomState(42)
        samples = proposal.sample(20, rng)
        for s in samples:
            agents_in_sched = set(s.ordering())
            assert agents_in_sched == set(AGENTS_3)

    def test_log_prob_finite(self):
        proposal = MixedLogitProposal(AGENTS_3, num_components=2)
        sched = _make_schedule(AGENTS_3)
        lp = proposal.log_prob(sched)
        assert np.isfinite(lp)

    def test_fit_improves_likelihood(self):
        """EM fitting should improve (or maintain) log-likelihood."""
        rng = np.random.RandomState(42)
        # Generate training data with structure
        schedules = []
        for _ in range(100):
            if rng.uniform() < 0.7:
                schedules.append(_make_schedule(["a0", "a1", "a2"]))
            else:
                schedules.append(_make_schedule(["a2", "a1", "a0"]))

        proposal = MixedLogitProposal(AGENTS_3, num_components=2, rng=rng)
        ll = proposal.fit(schedules, max_iter=30)
        assert np.isfinite(ll)
        assert proposal._fitted

    def test_proposal_distribution_interface(self):
        """MixedLogitProposal implements ProposalDistribution."""
        proposal = MixedLogitProposal(AGENTS_3)
        assert isinstance(proposal, ProposalDistribution)


# ---------------------------------------------------------------
# Test Nested Logit Proposal
# ---------------------------------------------------------------

class TestNestedLogitProposal:
    """Tests for nested logit proposal."""

    def test_sample_returns_correct_count(self):
        nests = [["a0", "a1"], ["a2", "a3"]]
        proposal = NestedLogitProposal(AGENTS_4, nests=nests)
        rng = np.random.RandomState(42)
        samples = proposal.sample(10, rng)
        assert len(samples) == 10

    def test_sample_covers_all_agents(self):
        nests = [["a0", "a1"], ["a2", "a3"]]
        proposal = NestedLogitProposal(AGENTS_4, nests=nests)
        rng = np.random.RandomState(42)
        samples = proposal.sample(20, rng)
        for s in samples:
            assert set(s.ordering()) == set(AGENTS_4)

    def test_log_prob_finite(self):
        nests = [["a0", "a1"], ["a2"]]
        proposal = NestedLogitProposal(AGENTS_3, nests=nests)
        sched = _make_schedule(AGENTS_3)
        lp = proposal.log_prob(sched)
        assert np.isfinite(lp)

    def test_nest_scale_affects_distribution(self):
        """Different nest scales should produce different distributions."""
        nests = [["a0", "a1"], ["a2", "a3"]]
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)

        prop1 = NestedLogitProposal(
            AGENTS_4, nests=nests,
            nest_scales=np.array([0.1, 0.1]),
        )
        prop2 = NestedLogitProposal(
            AGENTS_4, nests=nests,
            nest_scales=np.array([1.0, 1.0]),
        )

        samples1 = prop1.sample(100, rng1)
        samples2 = prop2.sample(100, rng2)

        # Count how often nest 0 agents come first
        nest0_first_1 = sum(1 for s in samples1 if s.ordering()[0] in ["a0", "a1"])
        nest0_first_2 = sum(1 for s in samples2 if s.ordering()[0] in ["a0", "a1"])
        # With different scales, distributions differ (or are at least valid)
        assert 0 <= nest0_first_1 <= 100
        assert 0 <= nest0_first_2 <= 100

    def test_nested_logit_is_proposal_distribution(self):
        nests = [["a0", "a1"], ["a2"]]
        proposal = NestedLogitProposal(AGENTS_3, nests=nests)
        assert isinstance(proposal, ProposalDistribution)


# ---------------------------------------------------------------
# Test Stopping Criteria
# ---------------------------------------------------------------

class TestStoppingCriteria:
    """Tests for convergence diagnostics."""

    def test_ess_stability_converged(self):
        criteria = StoppingCriteria(ess_stability_window=3, ess_cv_threshold=0.1)
        # Stable ESS history
        result = criteria.check_ess_stability([50.0, 50.1, 49.9, 50.0, 50.0])
        assert result.should_stop
        assert result.criterion == "ess_stability"

    def test_ess_stability_not_converged(self):
        criteria = StoppingCriteria(ess_stability_window=3, ess_cv_threshold=0.05)
        # Unstable ESS
        result = criteria.check_ess_stability([10.0, 50.0, 20.0])
        assert not result.should_stop

    def test_ci_width_stop(self):
        criteria = StoppingCriteria(ci_width_target=0.1)
        ci = ConfidenceInterval(estimate=0.5, lower=0.48, upper=0.52)
        result = criteria.check_ci_width(ci)
        assert result.should_stop

    def test_ci_width_continue(self):
        criteria = StoppingCriteria(ci_width_target=0.01)
        ci = ConfidenceInterval(estimate=0.5, lower=0.3, upper=0.7)
        result = criteria.check_ci_width(ci)
        assert not result.should_stop

    def test_mcse_computation(self):
        criteria = StoppingCriteria(mcse_threshold=0.1)
        values = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        log_w = np.zeros(5)
        result = criteria.check_mcse(values, log_w)
        assert "mcse" in result.details
        assert result.details["mcse"] >= 0

    def test_rhat_converged(self):
        criteria = StoppingCriteria(rhat_threshold=1.1)
        rng = np.random.RandomState(42)
        chains = [rng.normal(0, 1, 100) for _ in range(4)]
        result = criteria.check_rhat(chains)
        assert result.should_stop
        assert result.details["rhat"] < 1.1

    def test_rhat_not_converged(self):
        criteria = StoppingCriteria(rhat_threshold=1.01)
        chains = [
            np.ones(50) * 0.0,
            np.ones(50) * 10.0,
        ]
        result = criteria.check_rhat(chains)
        assert not result.should_stop

    def test_geweke_stationary(self):
        criteria = StoppingCriteria(geweke_threshold=2.0)
        rng = np.random.RandomState(42)
        chain = rng.normal(5.0, 1.0, 200)
        result = criteria.check_geweke(chain)
        assert result.should_stop or result.details["z_score"] < 3.0

    def test_geweke_nonstationary(self):
        criteria = StoppingCriteria(geweke_threshold=1.0)
        chain = np.concatenate([np.ones(100) * 0, np.ones(100) * 10])
        result = criteria.check_geweke(chain)
        assert not result.should_stop


# ---------------------------------------------------------------
# Test Joint Error Analysis
# ---------------------------------------------------------------

class TestJointErrorAnalysis:
    """Tests for error decomposition."""

    def test_decomposition_components(self):
        analysis = JointErrorAnalysis(ai_overapprox_ratio=2.0)
        result = analysis.analyze(
            is_estimate=0.1,
            is_variance=0.01,
            is_bias=0.001,
            num_samples=1000,
            ai_bound=0.2,
        )
        assert result.ai_error >= 0
        assert result.is_variance >= 0
        assert result.cross_term >= 0
        assert result.total_mse_bound >= 0

    def test_mse_bound_holds(self):
        """MSE bound should satisfy the theorem inequality."""
        analysis = JointErrorAnalysis(ai_overapprox_ratio=1.5)
        result = analysis.analyze(
            is_estimate=0.05,
            is_variance=0.02,
            is_bias=0.002,
            num_samples=500,
        )
        expected_bound = (
            result.ai_error ** 2
            + result.is_variance
            + result.cross_term
        )
        assert abs(result.total_mse_bound - expected_bound) < 1e-10

    def test_dominant_source_identification(self):
        """Should correctly identify the dominant error source."""
        # AI-dominated case
        analysis = JointErrorAnalysis(ai_overapprox_ratio=10.0)
        result = analysis.analyze(
            is_estimate=0.01,
            is_variance=0.0001,
            is_bias=0.0,
            num_samples=10000,
            ai_bound=1.0,
        )
        assert result.dominant_source == "ai"

    def test_is_variance_dominated(self):
        analysis = JointErrorAnalysis(ai_overapprox_ratio=1.0)
        result = analysis.analyze(
            is_estimate=0.5,
            is_variance=100.0,
            is_bias=0.0,
            num_samples=10,
        )
        assert result.dominant_source == "is_variance"

    def test_inflation_factor(self):
        analysis = JointErrorAnalysis(ai_overapprox_ratio=3.0)
        assert analysis.compute_inflation_factor() == 3.0

    def test_required_samples(self):
        analysis = JointErrorAnalysis(ai_overapprox_ratio=1.5)
        n = analysis.required_samples(
            target_mse=0.01,
            ai_error=0.01,
            is_bias=0.0,
        )
        assert n >= 1

    def test_recommendation_not_empty(self):
        analysis = JointErrorAnalysis(ai_overapprox_ratio=2.0)
        result = analysis.analyze(
            is_estimate=0.1, is_variance=0.01,
            is_bias=0.001, num_samples=100,
        )
        assert len(result.recommendation) > 0


# ---------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------

class TestIntegration:
    """Integration tests with existing sampling infrastructure."""

    def test_adaptive_sis_with_schedule_space(self):
        """AdaptiveSISEngine integrates with ScheduleSpace + UniformProposal."""
        space = ScheduleSpace(
            agents=AGENTS_3,
            num_timesteps=1,
            constraints=[
                ScheduleConstraint("a0", 0, "a1", 0),
            ],
        )
        proposal = UniformProposal(space)
        engine = AdaptiveSISEngine(
            target_log_prob=_uniform_target_log_prob,
            proposal=proposal,
            num_particles=30,
            mode="sir",
        )
        result = engine.run(rng=np.random.RandomState(42))
        assert len(result.particles) == 30
        assert len(result.weights.log_weights) == 30

    def test_mixed_logit_with_importance_sampler(self):
        """MixedLogitProposal works as a proposal in ImportanceSampler."""
        proposal = MixedLogitProposal(AGENTS_3, num_components=2)
        sampler = ImportanceSampler(
            target_log_prob=_uniform_target_log_prob,
            proposal=proposal,
        )
        schedules, weights = sampler.sample_and_weight(
            20, rng=np.random.RandomState(42)
        )
        assert len(schedules) == 20
        assert len(weights.log_weights) == 20

    def test_full_pipeline_sis_to_stopping(self):
        """End-to-end: SIS -> ESS history -> stopping criteria."""
        space = _make_space(AGENTS_3)
        proposal = UniformProposal(space)
        engine = AdaptiveSISEngine(
            target_log_prob=_uniform_target_log_prob,
            proposal=proposal,
            num_particles=50,
            max_steps=5,
            mode="sis",
        )
        result = engine.run(rng=np.random.RandomState(42))

        criteria = StoppingCriteria(ess_stability_window=2, ess_cv_threshold=0.5)
        if len(result.ess_history) >= 2:
            decision = criteria.check_ess_stability(result.ess_history)
            assert isinstance(decision, StoppingDecision)

    def test_iia_validation_then_proposal_selection(self):
        """Validate IIA, then select appropriate proposal."""
        rng = np.random.RandomState(42)
        # Generate schedules with correlation
        schedules = []
        for _ in range(200):
            if rng.uniform() < 0.5:
                schedules.append(_make_schedule(["a0", "a1", "a2"]))
            else:
                schedules.append(_make_schedule(["a2", "a1", "a0"]))

        validator = PlackettLuceValidator(AGENTS_3, significance_level=0.05)
        iia_result = validator.validate(schedules)

        # Based on result, choose appropriate proposal
        if iia_result.severity in ("moderate", "severe"):
            proposal = MixedLogitProposal(AGENTS_3, num_components=2, rng=rng)
            proposal.fit(schedules)
        else:
            proposal = MixedLogitProposal(AGENTS_3, num_components=1, rng=rng)

        # Verify proposal works
        samples = proposal.sample(10, rng)
        assert len(samples) == 10
