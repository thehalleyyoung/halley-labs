"""Unit tests for usability_oracle.mdp.pomdp — POMDP, belief, and solvers.

Tests cover POMDP construction, belief update, PBVI solver, QMDP heuristic,
and information gathering.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from usability_oracle.mdp.models import State, Action, Transition, MDP
from usability_oracle.mdp.pomdp import (
    POMDP,
    BeliefState,
    Observation,
    ObservationModel,
)
from usability_oracle.mdp.belief import (
    BeliefUpdater,
    BeliefCompressor,
    ParticleFilter,
    ParticleBeliefState,
    belief_entropy,
    belief_kl_divergence,
    belief_uncertainty_level,
)
from usability_oracle.mdp.pomdp_solver import (
    AlphaVector,
    BoundedRationalPOMDPPolicy,
    PBVISolver,
    POMDPPolicy,
    QMDPSolver,
    FIBSolver,
)
from usability_oracle.mdp.information_gathering import (
    InformationGain,
    ValueOfInformation,
    OptimalStopping,
    EntropyReductionReward,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_tiger_pomdp() -> POMDP:
    """Classic Tiger POMDP: 2 states, 3 actions, 2 observations."""
    states = {
        "tiger_left": State(state_id="tiger_left", features={}, label="tiger_left"),
        "tiger_right": State(state_id="tiger_right", features={}, label="tiger_right"),
    }
    actions = {
        "listen": Action(action_id="listen", action_type="observe",
                         target_node_id="", description="Listen"),
        "open_left": Action(action_id="open_left", action_type="act",
                            target_node_id="", description="Open left"),
        "open_right": Action(action_id="open_right", action_type="act",
                             target_node_id="", description="Open right"),
    }
    transitions = [
        # Listen doesn't change state
        Transition("tiger_left", "listen", "tiger_left", 1.0, cost=1.0),
        Transition("tiger_right", "listen", "tiger_right", 1.0, cost=1.0),
        # Opening resets (uniform)
        Transition("tiger_left", "open_left", "tiger_left", 0.5, cost=100.0),
        Transition("tiger_left", "open_left", "tiger_right", 0.5, cost=100.0),
        Transition("tiger_left", "open_right", "tiger_left", 0.5, cost=-10.0),
        Transition("tiger_left", "open_right", "tiger_right", 0.5, cost=-10.0),
        Transition("tiger_right", "open_left", "tiger_left", 0.5, cost=-10.0),
        Transition("tiger_right", "open_left", "tiger_right", 0.5, cost=-10.0),
        Transition("tiger_right", "open_right", "tiger_left", 0.5, cost=100.0),
        Transition("tiger_right", "open_right", "tiger_right", 0.5, cost=100.0),
    ]
    mdp = MDP(
        states=states, actions=actions, transitions=transitions,
        initial_state="tiger_left", goal_states=set(), discount=0.95,
    )
    observations = {
        "hear_left": Observation(obs_id="hear_left", features={}),
        "hear_right": Observation(obs_id="hear_right", features={}),
    }
    obs_model = ObservationModel()
    # Listen: 85% accurate
    obs_model.add("tiger_left", "listen", "hear_left", 0.85)
    obs_model.add("tiger_left", "listen", "hear_right", 0.15)
    obs_model.add("tiger_right", "listen", "hear_left", 0.15)
    obs_model.add("tiger_right", "listen", "hear_right", 0.85)
    # Open actions: uniform observation
    for s in ["tiger_left", "tiger_right"]:
        for a in ["open_left", "open_right"]:
            obs_model.add(s, a, "hear_left", 0.5)
            obs_model.add(s, a, "hear_right", 0.5)

    initial_belief = BeliefState.uniform(["tiger_left", "tiger_right"])
    return POMDP(
        mdp=mdp,
        observations=observations,
        observation_model=obs_model,
        initial_belief=initial_belief,
    )


def _make_simple_pomdp() -> POMDP:
    """A simple 2-state, 2-action, 2-obs POMDP."""
    states = {
        "s0": State(state_id="s0", features={}, label="s0"),
        "s1": State(state_id="s1", features={}, label="s1", is_terminal=True, is_goal=True),
    }
    actions = {
        "go": Action(action_id="go", action_type="act", target_node_id="", description="go"),
        "stay": Action(action_id="stay", action_type="observe", target_node_id="", description="stay"),
    }
    transitions = [
        Transition("s0", "go", "s1", 0.8, cost=1.0),
        Transition("s0", "go", "s0", 0.2, cost=1.0),
        Transition("s0", "stay", "s0", 1.0, cost=0.5),
        Transition("s1", "go", "s1", 1.0, cost=0.0),
        Transition("s1", "stay", "s1", 1.0, cost=0.0),
    ]
    mdp = MDP(states=states, actions=actions, transitions=transitions,
              initial_state="s0", goal_states={"s1"}, discount=0.95)
    observations = {
        "o0": Observation(obs_id="o0", features={}),
        "o1": Observation(obs_id="o1", features={}),
    }
    obs_model = ObservationModel()
    obs_model.add("s0", "go", "o0", 0.7)
    obs_model.add("s0", "go", "o1", 0.3)
    obs_model.add("s1", "go", "o0", 0.2)
    obs_model.add("s1", "go", "o1", 0.8)
    obs_model.add("s0", "stay", "o0", 0.9)
    obs_model.add("s0", "stay", "o1", 0.1)
    obs_model.add("s1", "stay", "o0", 0.1)
    obs_model.add("s1", "stay", "o1", 0.9)
    return POMDP(mdp=mdp, observations=observations, observation_model=obs_model,
                 initial_belief=BeliefState.uniform(["s0", "s1"]))


# ═══════════════════════════════════════════════════════════════════════════
# POMDP Construction
# ═══════════════════════════════════════════════════════════════════════════


class TestPOMDPConstruction:
    """Test POMDP creation and validation."""

    def test_tiger_pomdp_construction(self):
        pomdp = _make_tiger_pomdp()
        assert len(pomdp.mdp.states) == 2
        assert len(pomdp.observations) == 2
        assert len(pomdp.mdp.actions) == 3

    def test_observation_model_probabilities(self):
        pomdp = _make_tiger_pomdp()
        prob = pomdp.observation_model.prob("hear_left", "tiger_left", "listen")
        assert prob == pytest.approx(0.85)

    def test_observation_model_sample(self):
        pomdp = _make_tiger_pomdp()
        rng = np.random.default_rng(42)
        obs = pomdp.observation_model.sample("tiger_left", "listen", rng=rng)
        assert obs in ("hear_left", "hear_right")

    def test_validate(self):
        pomdp = _make_tiger_pomdp()
        errors = pomdp.validate()
        assert isinstance(errors, list)

    def test_expected_reward(self):
        pomdp = _make_tiger_pomdp()
        belief = BeliefState.uniform(["tiger_left", "tiger_right"])
        r = pomdp.expected_reward(belief, "listen")
        assert isinstance(r, float)


# ═══════════════════════════════════════════════════════════════════════════
# Belief State
# ═══════════════════════════════════════════════════════════════════════════


class TestBeliefState:
    """Test belief state construction and operations."""

    def test_uniform_belief(self):
        b = BeliefState.uniform(["s0", "s1", "s2"])
        for p in b.distribution.values():
            assert p == pytest.approx(1.0 / 3)

    def test_point_belief(self):
        b = BeliefState.point("s0")
        assert b.distribution["s0"] == pytest.approx(1.0)

    def test_to_vector(self):
        b = BeliefState.uniform(["a", "b"])
        vec = b.to_vector(["a", "b"])
        assert len(vec) == 2
        assert abs(vec.sum() - 1.0) < 1e-10

    def test_from_vector(self):
        vec = np.array([0.3, 0.7])
        b = BeliefState.from_vector(vec, ["s0", "s1"])
        assert b.distribution["s0"] == pytest.approx(0.3)
        assert b.distribution["s1"] == pytest.approx(0.7)


# ═══════════════════════════════════════════════════════════════════════════
# Belief Update
# ═══════════════════════════════════════════════════════════════════════════


class TestBeliefUpdate:
    """Test Bayesian belief update."""

    def test_belief_update_on_tiger(self):
        pomdp = _make_tiger_pomdp()
        b0 = BeliefState.uniform(["tiger_left", "tiger_right"])
        # After hearing left, should believe tiger is more likely on left
        b1 = pomdp.belief_update(b0, "listen", "hear_left")
        assert b1.distribution["tiger_left"] > 0.5

    def test_repeated_observation_concentrates(self):
        pomdp = _make_tiger_pomdp()
        b = BeliefState.uniform(["tiger_left", "tiger_right"])
        for _ in range(5):
            b = pomdp.belief_update(b, "listen", "hear_left")
        assert b.distribution["tiger_left"] > 0.9

    def test_belief_sums_to_one(self):
        pomdp = _make_tiger_pomdp()
        b = BeliefState.uniform(["tiger_left", "tiger_right"])
        b1 = pomdp.belief_update(b, "listen", "hear_right")
        total = sum(b1.distribution.values())
        assert total == pytest.approx(1.0)

    def test_updater_class(self):
        pomdp = _make_tiger_pomdp()
        updater = BeliefUpdater(pomdp)
        b0 = BeliefState.uniform(["tiger_left", "tiger_right"])
        b1 = updater.update(b0, "listen", "hear_left")
        assert b1.distribution["tiger_left"] > 0.5

    def test_observation_likelihood(self):
        pomdp = _make_tiger_pomdp()
        updater = BeliefUpdater(pomdp)
        b0 = BeliefState.uniform(["tiger_left", "tiger_right"])
        lik = updater.observation_likelihood(b0, "listen", "hear_left")
        assert 0.0 < lik < 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Belief Entropy
# ═══════════════════════════════════════════════════════════════════════════


class TestBeliefEntropy:
    """Test belief entropy and information measures."""

    def test_uniform_max_entropy(self):
        b = BeliefState.uniform(["s0", "s1"])
        e = belief_entropy(b)
        # Uses log2, so H(uniform over 2) = 1.0 bit
        assert e == pytest.approx(1.0, abs=0.01)

    def test_point_zero_entropy(self):
        b = BeliefState.point("s0")
        e = belief_entropy(b)
        assert e == pytest.approx(0.0, abs=0.01)

    def test_kl_divergence_non_negative(self):
        b1 = BeliefState(distribution={"s0": 0.3, "s1": 0.7})
        b2 = BeliefState(distribution={"s0": 0.5, "s1": 0.5})
        kl = belief_kl_divergence(b1, b2)
        assert kl >= -1e-10

    def test_uncertainty_level(self):
        b_certain = BeliefState(distribution={"s0": 0.99, "s1": 0.01})
        b_uncertain = BeliefState.uniform(["s0", "s1"])
        assert isinstance(belief_uncertainty_level(b_certain), str)
        assert isinstance(belief_uncertainty_level(b_uncertain), str)


# ═══════════════════════════════════════════════════════════════════════════
# QMDP Solver
# ═══════════════════════════════════════════════════════════════════════════


class TestQMDPSolver:
    """Test QMDP heuristic for POMDPs."""

    def test_qmdp_solves_simple(self):
        pomdp = _make_simple_pomdp()
        solver = QMDPSolver()
        policy = solver.solve(pomdp)
        assert isinstance(policy, POMDPPolicy)

    def test_qmdp_policy_value(self):
        pomdp = _make_simple_pomdp()
        solver = QMDPSolver()
        policy = solver.solve(pomdp)
        b = BeliefState.uniform(["s0", "s1"])
        v = policy.value(b)
        assert isinstance(v, float)

    def test_qmdp_policy_action(self):
        pomdp = _make_simple_pomdp()
        solver = QMDPSolver()
        policy = solver.solve(pomdp)
        b = BeliefState.uniform(["s0", "s1"])
        a = policy.action(b)
        assert a in pomdp.mdp.actions

    def test_fib_solver(self):
        pomdp = _make_simple_pomdp()
        solver = FIBSolver()
        policy = solver.solve(pomdp)
        assert isinstance(policy, POMDPPolicy)


# ═══════════════════════════════════════════════════════════════════════════
# PBVI Solver
# ═══════════════════════════════════════════════════════════════════════════


class TestPBVISolver:
    """Test Point-Based Value Iteration."""

    def test_pbvi_solves(self):
        pomdp = _make_simple_pomdp()
        solver = PBVISolver(n_belief_points=20, n_expand_steps=5)
        policy, info = solver.solve(pomdp, max_iter=50)
        assert isinstance(policy, POMDPPolicy)

    def test_pbvi_alpha_vectors(self):
        pomdp = _make_simple_pomdp()
        solver = PBVISolver(n_belief_points=20, n_expand_steps=5)
        policy, _ = solver.solve(pomdp, max_iter=50)
        assert len(policy.alpha_vectors) > 0

    def test_pbvi_convergence_info(self):
        pomdp = _make_simple_pomdp()
        solver = PBVISolver(n_belief_points=10, n_expand_steps=3)
        _, info = solver.solve(pomdp, max_iter=30)
        assert hasattr(info, "converged") or hasattr(info, "iterations")


# ═══════════════════════════════════════════════════════════════════════════
# Bounded-Rational POMDP Policy
# ═══════════════════════════════════════════════════════════════════════════


class TestBoundedRationalPOMDP:
    """Test bounded-rational POMDP policy wrapper."""

    def test_action_probabilities(self):
        pomdp = _make_simple_pomdp()
        solver = QMDPSolver()
        base = solver.solve(pomdp)
        br = BoundedRationalPOMDPPolicy(base, rationality=1.0)
        b = BeliefState.uniform(["s0", "s1"])
        probs = br.action_probabilities(b)
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_high_rationality_deterministic(self):
        pomdp = _make_simple_pomdp()
        solver = QMDPSolver()
        base = solver.solve(pomdp)
        br = BoundedRationalPOMDPPolicy(base, rationality=100.0)
        b = BeliefState.uniform(["s0", "s1"])
        probs = br.action_probabilities(b)
        assert max(probs.values()) > 0.9

    def test_sample_action(self):
        pomdp = _make_simple_pomdp()
        solver = QMDPSolver()
        base = solver.solve(pomdp)
        br = BoundedRationalPOMDPPolicy(base, rationality=1.0)
        b = BeliefState.uniform(["s0", "s1"])
        rng = np.random.default_rng(42)
        a = br.sample_action(b, rng=rng)
        assert a in pomdp.mdp.actions


# ═══════════════════════════════════════════════════════════════════════════
# Information Gathering
# ═══════════════════════════════════════════════════════════════════════════


class TestInformationGathering:
    """Test information gathering strategies."""

    def test_information_gain(self):
        pomdp = _make_tiger_pomdp()
        updater = BeliefUpdater(pomdp)
        ig = InformationGain(pomdp, updater)
        b = BeliefState.uniform(["tiger_left", "tiger_right"])
        gain = ig.expected_gain(b, "listen")
        assert gain >= 0

    def test_most_informative_action(self):
        pomdp = _make_tiger_pomdp()
        updater = BeliefUpdater(pomdp)
        ig = InformationGain(pomdp, updater)
        b = BeliefState.uniform(["tiger_left", "tiger_right"])
        action, gain = ig.most_informative_action(b)
        assert action in pomdp.mdp.actions
        assert gain >= 0

    def test_entropy_reduction_reward(self):
        pomdp = _make_tiger_pomdp()
        err = EntropyReductionReward(pomdp, weight=1.0)
        b = BeliefState.uniform(["tiger_left", "tiger_right"])
        intrinsic = err.intrinsic_reward(b, "listen")
        assert isinstance(intrinsic, float)

    def test_value_of_information(self):
        pomdp = _make_tiger_pomdp()
        solver = QMDPSolver()
        policy = solver.solve(pomdp)
        voi = ValueOfInformation(pomdp, policy)
        b = BeliefState.uniform(["tiger_left", "tiger_right"])
        v = voi.voi_action(b, "listen")
        assert isinstance(v, float)

    def test_particle_filter(self):
        pomdp = _make_tiger_pomdp()
        pf = ParticleFilter(pomdp, n_particles=500, rng=np.random.default_rng(42))
        particles = pf.initialize()
        assert isinstance(particles, ParticleBeliefState)
        updated = pf.update(particles, "listen", "hear_left")
        belief = updated.to_belief_state()
        assert belief.distribution["tiger_left"] > 0.5
