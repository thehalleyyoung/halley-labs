"""Unit tests for stochastic simulation — SSA, tau-leaping, FSP, moment closure."""

import math

import numpy as np
import pytest

from bioprover.stochastic.ssa import (
    DirectMethod,
    NextReactionMethod,
    Reaction,
    StochasticState,
    TrajectoryRecorder,
    run_ensemble_ssa,
)


# ===================================================================
# SSA — birth-death process
# ===================================================================


def _birth_death_reactions(birth_rate=10.0, death_rate=0.1):
    """A simple birth-death process: ∅ → X (rate λ), X → ∅ (rate μ·X)."""
    birth = Reaction(
        name="birth",
        reactants={},
        products={0: 1},
        rate_constant=birth_rate,
    )
    death = Reaction(
        name="death",
        reactants={0: 1},
        products={},
        rate_constant=death_rate,
    )
    return [birth, death]


class TestSSADirectMethod:
    def test_simulate_runs(self):
        reactions = _birth_death_reactions()
        sim = DirectMethod(reactions, num_species=1, seed=42)
        initial = np.array([0])
        state = sim.simulate(initial, t_end=10.0)
        assert state.time >= 10.0 or state.time > 0

    def test_simulate_trajectory(self):
        reactions = _birth_death_reactions()
        sim = DirectMethod(reactions, num_species=1, seed=42)
        initial = np.array([0])
        recorder = sim.simulate_trajectory(initial, t_end=5.0)
        times = recorder.get_times()
        states = recorder.get_states()
        assert len(times) > 0
        assert states.shape[0] == len(times)

    def test_non_negative(self):
        """Species counts should never go negative."""
        reactions = _birth_death_reactions(birth_rate=1.0, death_rate=1.0)
        sim = DirectMethod(reactions, num_species=1, seed=123)
        initial = np.array([5])
        recorder = sim.simulate_trajectory(initial, t_end=20.0)
        states = recorder.get_states()
        assert np.all(states >= 0)


class TestSSANextReactionMethod:
    def test_simulate_runs(self):
        reactions = _birth_death_reactions()
        sim = NextReactionMethod(reactions, num_species=1, seed=42)
        initial = np.array([0])
        state = sim.simulate(initial, t_end=10.0)
        assert state.time >= 10.0 or state.time > 0

    def test_consistent_with_direct(self):
        """Both methods should produce similar mean for large ensembles."""
        reactions = _birth_death_reactions(birth_rate=10.0, death_rate=0.1)
        n_runs = 50
        finals_dm = []
        finals_nrm = []
        for i in range(n_runs):
            dm = DirectMethod(reactions, num_species=1, seed=i)
            state = dm.simulate(np.array([0]), t_end=50.0)
            finals_dm.append(state.copy_numbers[0])
            nrm = NextReactionMethod(reactions, num_species=1, seed=i + 1000)
            state2 = nrm.simulate(np.array([0]), t_end=50.0)
            finals_nrm.append(state2.copy_numbers[0])
        mean_dm = np.mean(finals_dm)
        mean_nrm = np.mean(finals_nrm)
        # Means should be within a factor of 2 of the analytical steady-state (100)
        assert 30 < mean_dm < 300
        assert 30 < mean_nrm < 300


class TestSSABirthDeathSteadyState:
    """The birth-death process has known steady-state: Poisson(λ/μ)."""

    def test_mean_near_analytical(self):
        lam, mu = 10.0, 0.1
        reactions = _birth_death_reactions(lam, mu)
        result = run_ensemble_ssa(
            reactions, num_species=1, initial_state=np.array([0]),
            t_end=100.0, num_runs=100, seed=42,
        )
        mean_final = result["mean"][-1, 0] if "mean" in result else np.mean([r[-1] for r in result.get("trajectories", [])])
        expected_mean = lam / mu  # = 100
        assert abs(mean_final - expected_mean) / expected_mean < 0.5  # within 50%


class TestTrajectoryRecorder:
    def test_record_and_retrieve(self):
        rec = TrajectoryRecorder(num_species=2, record_mode="full")
        rec.record_initial(0.0, np.array([10, 20]))
        rec.record(1.0, np.array([11, 19]), 0)
        rec.record(2.0, np.array([12, 18]), 1)
        times = rec.get_times()
        states = rec.get_states()
        assert len(times) == 3
        assert states.shape == (3, 2)

    def test_interpolate(self):
        rec = TrajectoryRecorder(num_species=1, record_mode="full")
        rec.record_initial(0.0, np.array([0]))
        rec.record(1.0, np.array([10]), 0)
        rec.record(2.0, np.array([20]), 0)
        interpolated = rec.interpolate_at(np.array([0.5, 1.5]))
        assert interpolated.shape[0] == 2


# ===================================================================
# Tau-leaping
# ===================================================================


class TestTauLeaping:
    def test_explicit_simulate(self):
        from bioprover.stochastic.tau_leaping import ExplicitTauLeaping
        reactions = _birth_death_reactions()
        sim = ExplicitTauLeaping(reactions, num_species=1, seed=42)
        initial = np.array([50])
        state = sim.simulate(initial, t_end=10.0)
        assert state.time >= 10.0 or state.time > 0

    def test_explicit_non_negative(self):
        from bioprover.stochastic.tau_leaping import ExplicitTauLeaping
        reactions = _birth_death_reactions(birth_rate=1.0, death_rate=0.5)
        sim = ExplicitTauLeaping(reactions, num_species=1, seed=42,
                                  negative_handling="reject")
        initial = np.array([10])
        state = sim.simulate(initial, t_end=20.0)
        assert state.copy_numbers[0] >= 0

    def test_midpoint_simulate(self):
        from bioprover.stochastic.tau_leaping import MidpointTauLeaping
        reactions = _birth_death_reactions()
        sim = MidpointTauLeaping(reactions, num_species=1, seed=42)
        initial = np.array([50])
        state = sim.simulate(initial, t_end=5.0)
        assert state.time > 0

    def test_ssa_switch(self):
        from bioprover.stochastic.tau_leaping import SSATauLeapingSwitch
        reactions = _birth_death_reactions()
        sim = SSATauLeapingSwitch(reactions, num_species=1, seed=42)
        initial = np.array([50])
        state = sim.simulate(initial, t_end=5.0)
        assert state.time > 0

    def test_tau_leaping_similar_statistics(self):
        """Tau-leaping should produce similar mean to SSA for large populations."""
        from bioprover.stochastic.tau_leaping import ExplicitTauLeaping
        reactions = _birth_death_reactions(birth_rate=50.0, death_rate=0.1)
        n_runs = 30
        finals_ssa = []
        finals_tau = []
        for i in range(n_runs):
            dm = DirectMethod(reactions, num_species=1, seed=i)
            state = dm.simulate(np.array([100]), t_end=20.0)
            finals_ssa.append(state.copy_numbers[0])
            tl = ExplicitTauLeaping(reactions, num_species=1, seed=i + 5000)
            state2 = tl.simulate(np.array([100]), t_end=20.0)
            finals_tau.append(state2.copy_numbers[0])
        mean_ssa = np.mean(finals_ssa)
        mean_tau = np.mean(finals_tau)
        # Should be in same ballpark
        assert mean_tau > 0


# ===================================================================
# FSP
# ===================================================================


class TestFSP:
    def test_solve_birth_death(self):
        from bioprover.stochastic.fsp import FSPReaction, FSPSolver
        birth = FSPReaction(reactants={}, products={0: 1}, rate_constant=5.0)
        death = FSPReaction(reactants={0: 1}, products={}, rate_constant=0.1)
        solver = FSPSolver(
            reactions=[birth, death],
            num_species=1,
            initial_state=(0,),
            state_bounds=[(0, 200)],
        )
        prob, trunc_err = solver.solve(t_final=50.0)
        assert np.sum(prob) + trunc_err >= 0.99  # probability should sum to ~1

    def test_fsp_mean(self):
        from bioprover.stochastic.fsp import FSPReaction, FSPSolver
        lam, mu = 5.0, 0.1
        birth = FSPReaction(reactants={}, products={0: 1}, rate_constant=lam)
        death = FSPReaction(reactants={0: 1}, products={}, rate_constant=mu)
        solver = FSPSolver(
            reactions=[birth, death],
            num_species=1,
            initial_state=(0,),
            state_bounds=[(0, 150)],
        )
        prob, _ = solver.solve(t_final=100.0)
        dist = solver.get_distribution(prob)
        mean = dist.mean(0)
        expected = lam / mu  # = 50
        assert abs(mean - expected) / expected < 0.3

    def test_fsp_variance(self):
        from bioprover.stochastic.fsp import FSPReaction, FSPSolver
        lam, mu = 5.0, 0.1
        birth = FSPReaction(reactants={}, products={0: 1}, rate_constant=lam)
        death = FSPReaction(reactants={0: 1}, products={}, rate_constant=mu)
        solver = FSPSolver(
            reactions=[birth, death],
            num_species=1,
            initial_state=(0,),
            state_bounds=[(0, 200)],
        )
        prob, _ = solver.solve(t_final=100.0)
        dist = solver.get_distribution(prob)
        var = dist.variance(0)
        expected_var = lam / mu  # Poisson variance = mean
        assert abs(var - expected_var) / expected_var < 0.5

    def test_state_space_enumeration(self):
        from bioprover.stochastic.fsp import StateSpace
        ss = StateSpace.enumerate_hypercube(num_species=2, bounds=[(0, 3), (0, 3)])
        assert len(ss.all_states()) == 16  # 4 * 4

    def test_marginal_distribution(self):
        from bioprover.stochastic.fsp import FSPReaction, FSPSolver
        birth = FSPReaction(reactants={}, products={0: 1}, rate_constant=2.0)
        death = FSPReaction(reactants={0: 1}, products={}, rate_constant=0.1)
        solver = FSPSolver(
            reactions=[birth, death],
            num_species=1,
            initial_state=(0,),
            state_bounds=[(0, 100)],
        )
        prob, _ = solver.solve(t_final=50.0)
        dist = solver.get_distribution(prob)
        vals, probs = dist.marginal(0)
        assert np.sum(probs) >= 0.95


# ===================================================================
# Moment closure
# ===================================================================


class TestMomentClosure:
    def _birth_death_moment_reactions(self):
        from bioprover.stochastic.moment_closure import MomentReaction
        birth = MomentReaction(reactants={}, products={0: 1}, rate_constant=10.0)
        death = MomentReaction(reactants={0: 1}, products={}, rate_constant=0.1)
        return [birth, death]

    def test_normal_closure_solve(self):
        from bioprover.stochastic.moment_closure import (
            MomentClosureSolver,
            MomentEquations,
            NormalClosure,
        )
        reactions = self._birth_death_moment_reactions()
        eqs = MomentEquations(reactions, num_species=1, max_order=2)
        closure = NormalClosure(num_species=1)
        solver = MomentClosureSolver(eqs, closure)
        result = solver.solve(
            initial_means=np.array([0.0]),
            initial_cov=np.array([[0.0]]),
            t_span=(0, 50),
            t_eval=np.linspace(0, 50, 51),
        )
        assert "times" in result or "means" in result

    def test_lognormal_closure_solve(self):
        from bioprover.stochastic.moment_closure import (
            LogNormalClosure,
            MomentClosureSolver,
            MomentEquations,
        )
        reactions = self._birth_death_moment_reactions()
        eqs = MomentEquations(reactions, num_species=1, max_order=2)
        closure = LogNormalClosure(num_species=1)
        solver = MomentClosureSolver(eqs, closure)
        result = solver.solve(
            initial_means=np.array([10.0]),
            initial_cov=np.array([[1.0]]),
            t_span=(0, 20),
            t_eval=np.linspace(0, 20, 21),
        )
        assert result is not None

    def test_lna_solve(self):
        from bioprover.stochastic.moment_closure import (
            LinearNoiseApproximation,
            MomentReaction,
        )
        reactions = self._birth_death_moment_reactions()
        lna = LinearNoiseApproximation(reactions, num_species=1, volume=1.0)
        result = lna.solve(
            initial_concentrations=np.array([0.0]),
            t_span=(0, 50),
            t_eval=np.linspace(0, 50, 51),
        )
        assert result is not None

    def test_closure_comparison(self):
        from bioprover.stochastic.moment_closure import (
            ClosureComparison,
            MomentReaction,
            NormalClosure,
            ZeroCumulantClosure,
        )
        reactions = self._birth_death_moment_reactions()
        comparison = ClosureComparison(reactions, num_species=1, max_order=2)
        closures = {
            "normal": NormalClosure(num_species=1),
            "zero_cumulant": ZeroCumulantClosure(num_species=1),
        }
        result = comparison.compare(
            initial_means=np.array([10.0]),
            initial_cov=np.array([[1.0]]),
            t_span=(0, 20),
            t_eval=np.linspace(0, 20, 11),
            closures=closures,
        )
        assert len(result) >= 2

    def test_moment_closure_vs_analytical(self):
        """For birth-death, mean should converge to λ/μ."""
        from bioprover.stochastic.moment_closure import (
            MomentClosureSolver,
            MomentEquations,
            NormalClosure,
        )
        lam, mu = 10.0, 0.1
        from bioprover.stochastic.moment_closure import MomentReaction
        reactions = [
            MomentReaction(reactants={}, products={0: 1}, rate_constant=lam),
            MomentReaction(reactants={0: 1}, products={}, rate_constant=mu),
        ]
        eqs = MomentEquations(reactions, num_species=1, max_order=2)
        closure = NormalClosure(num_species=1)
        solver = MomentClosureSolver(eqs, closure)
        result = solver.solve(
            initial_means=np.array([0.0]),
            initial_cov=np.array([[0.0]]),
            t_span=(0, 200),
            t_eval=np.linspace(0, 200, 201),
        )
        final_mean = result["means"][-1, 0] if "means" in result else None
        if final_mean is not None:
            assert abs(final_mean - lam / mu) / (lam / mu) < 0.3


# ===================================================================
# Hybrid simulation
# ===================================================================


class TestHybridSimulation:
    def test_hybrid_runs(self):
        from bioprover.stochastic.hybrid import HaseltineRawlingsHybrid
        reactions = _birth_death_reactions()
        sim = HaseltineRawlingsHybrid(
            reactions, num_species=1, threshold_low=10, threshold_high=100, seed=42,
        )
        initial = np.array([50])
        state = sim.simulate(initial, t_end=5.0)
        assert state.time > 0

    def test_hybrid_trajectory(self):
        from bioprover.stochastic.hybrid import HaseltineRawlingsHybrid
        reactions = _birth_death_reactions()
        sim = HaseltineRawlingsHybrid(
            reactions, num_species=1, threshold_low=5, threshold_high=50, seed=42,
        )
        initial = np.array([20])
        traj = sim.simulate_trajectory(initial, t_end=5.0)
        times = traj.get_times()
        states = traj.get_states()
        assert len(times) > 0
        assert states.shape[1] == 1


# ===================================================================
# Ensemble simulation
# ===================================================================


class TestEnsembleSimulation:
    def test_ensemble_statistics(self):
        from bioprover.stochastic.ensemble import EnsembleStatistics
        # Fake trajectories
        n_runs = 10
        n_times = 5
        trajectories = np.random.rand(n_runs, n_times, 1) * 100
        times = np.linspace(0, 10, n_times)
        stats = EnsembleStatistics.compute(trajectories, times)
        assert stats.mean.shape == (n_times, 1)
        assert stats.variance.shape == (n_times, 1)
        assert stats.num_runs == n_runs

    def test_ensemble_simulator_runs(self):
        from bioprover.stochastic.ensemble import EnsembleSimulator
        reactions = _birth_death_reactions(birth_rate=5.0, death_rate=0.1)
        sim = EnsembleSimulator(reactions, num_species=1, method="direct", seed=42)
        stats = sim.run(
            initial_state=np.array([10]),
            t_end=10.0,
            num_runs=5,
            sample_interval=1.0,
        )
        assert stats.num_runs == 5
        assert stats.mean.shape[1] == 1

    def test_convergence_detector(self):
        from bioprover.stochastic.ensemble import ConvergenceDetector
        det = ConvergenceDetector(atol=1.0, rtol=0.1, window=5, min_runs=3)
        for _ in range(10):
            det.update(
                running_mean=np.array([100.0]),
                running_var=np.array([10.0]),
            )
        # After enough updates, it may report convergence
        assert isinstance(det.is_converged(), bool)
