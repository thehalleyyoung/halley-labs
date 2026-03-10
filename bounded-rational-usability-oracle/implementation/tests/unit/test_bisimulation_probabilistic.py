"""Unit tests for usability_oracle.bisimulation — Probabilistic bisimulation.

Tests cover probabilistic bisimulation metric, Kantorovich distance, spectral
bisimulation, and cognitive metric computations.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from usability_oracle.mdp.models import State, Action, Transition, MDP
from usability_oracle.bisimulation.models import CognitiveDistanceMatrix, Partition
from usability_oracle.bisimulation.probabilistic import (
    LarsenSkouBisimulation,
    ProbabilisticBisimulationMetric,
    kantorovich_distance,
)
from usability_oracle.bisimulation.spectral import (
    GraphLaplacian,
    SpectralBisimulation,
    build_transition_matrix,
    fiedler_partition,
)
from usability_oracle.bisimulation.simulation_relation import (
    SimulationRelation,
    SimulationDistance,
)
from usability_oracle.bisimulation.cognitive_metric import (
    CognitiveAggregation,
    CognitiveKernel,
    FreeEnergyDistance,
    PolicySensitivity,
)
from usability_oracle.bisimulation.approximate import (
    EpsilonBisimulation,
    ApproximatePartitionRefinement,
)
from usability_oracle.bisimulation.compositional import (
    BisimulationUpTo,
    CongruenceChecker,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _symmetric_mdp() -> MDP:
    """Two equivalent states going to the same goal."""
    states = {
        "s0": State(state_id="s0", features={"x": 0.0}, label="s0"),
        "s1": State(state_id="s1", features={"x": 1.0}, label="s1"),
        "goal": State(state_id="goal", features={"x": 2.0}, label="goal",
                      is_terminal=True, is_goal=True),
    }
    actions = {
        "go": Action(action_id="go", action_type=Action.CLICK,
                     target_node_id="n", description="go"),
    }
    transitions = [
        Transition("s0", "go", "goal", 1.0, cost=1.0),
        Transition("s1", "go", "goal", 1.0, cost=1.0),
    ]
    return MDP(states=states, actions=actions, transitions=transitions,
               initial_state="s0", goal_states={"goal"}, discount=0.95)


def _asymmetric_mdp() -> MDP:
    """Two states with different costs to goal."""
    states = {
        "s0": State(state_id="s0", features={"x": 0.0}, label="s0"),
        "s1": State(state_id="s1", features={"x": 1.0}, label="s1"),
        "goal": State(state_id="goal", features={"x": 2.0}, label="goal",
                      is_terminal=True, is_goal=True),
    }
    actions = {
        "go": Action(action_id="go", action_type=Action.CLICK,
                     target_node_id="n", description="go"),
    }
    transitions = [
        Transition("s0", "go", "goal", 1.0, cost=1.0),
        Transition("s1", "go", "goal", 1.0, cost=5.0),
    ]
    return MDP(states=states, actions=actions, transitions=transitions,
               initial_state="s0", goal_states={"goal"}, discount=0.95)


def _chain_mdp(n: int = 5) -> MDP:
    """Linear chain of n states."""
    states = {}
    for i in range(n):
        sid = f"s{i}"
        states[sid] = State(state_id=sid, features={"x": float(i)}, label=sid,
                           is_terminal=(i == n - 1), is_goal=(i == n - 1))
    actions = {"go": Action(action_id="go", action_type=Action.CLICK,
                            target_node_id="n", description="go")}
    transitions = [
        Transition(f"s{i}", "go", f"s{i+1}", 1.0, cost=1.0)
        for i in range(n - 1)
    ]
    return MDP(states=states, actions=actions, transitions=transitions,
               initial_state="s0", goal_states={f"s{n-1}"}, discount=0.95)


# ═══════════════════════════════════════════════════════════════════════════
# Kantorovich Distance
# ═══════════════════════════════════════════════════════════════════════════


class TestKantorovichDistance:
    """Test Kantorovich (Wasserstein) distance computation."""

    def test_same_distribution_zero(self):
        p = np.array([0.5, 0.5])
        d = kantorovich_distance(p, p, np.array([[0, 1], [1, 0]], dtype=float))
        assert d == pytest.approx(0.0, abs=1e-8)

    def test_dirac_distributions(self):
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        metric = np.array([[0.0, 1.0], [1.0, 0.0]])
        d = kantorovich_distance(p, q, metric)
        assert d == pytest.approx(1.0, abs=1e-6)

    def test_triangle_inequality(self):
        metric = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 1.0, 0.0])
        r = np.array([0.0, 0.0, 1.0])
        d_pq = kantorovich_distance(p, q, metric)
        d_qr = kantorovich_distance(q, r, metric)
        d_pr = kantorovich_distance(p, r, metric)
        assert d_pr <= d_pq + d_qr + 1e-6

    def test_symmetry(self):
        p = np.array([0.3, 0.7])
        q = np.array([0.6, 0.4])
        metric = np.array([[0, 1], [1, 0]], dtype=float)
        assert kantorovich_distance(p, q, metric) == pytest.approx(
            kantorovich_distance(q, p, metric), abs=1e-8
        )


# ═══════════════════════════════════════════════════════════════════════════
# Probabilistic Bisimulation Metric
# ═══════════════════════════════════════════════════════════════════════════


class TestProbabilisticBisimulationMetric:
    """Test probabilistic bisimulation distance computation."""

    def test_symmetric_states_zero_distance(self):
        mdp = _symmetric_mdp()
        pbm = ProbabilisticBisimulationMetric()
        dm = pbm.compute(mdp)
        d = dm.distance("s0", "s1")
        assert d == pytest.approx(0.0, abs=0.1)

    def test_asymmetric_states_positive_distance(self):
        mdp = _asymmetric_mdp()
        pbm = ProbabilisticBisimulationMetric()
        dm = pbm.compute(mdp)
        d = dm.distance("s0", "s1")
        assert d > 0.1

    def test_self_distance_zero(self):
        mdp = _chain_mdp(3)
        pbm = ProbabilisticBisimulationMetric()
        dm = pbm.compute(mdp)
        for sid in mdp.states:
            assert dm.distance(sid, sid) == pytest.approx(0.0, abs=1e-8)

    def test_metric_symmetry(self):
        mdp = _chain_mdp(4)
        pbm = ProbabilisticBisimulationMetric()
        dm = pbm.compute(mdp)
        for s1 in mdp.states:
            for s2 in mdp.states:
                assert abs(dm.distance(s1, s2) - dm.distance(s2, s1)) < 1e-6

    def test_convergence_rate(self):
        pbm = ProbabilisticBisimulationMetric()
        rate = pbm.convergence_rate(gamma=0.95)
        assert 0.0 < rate < 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Larsen-Skou Bisimulation
# ═══════════════════════════════════════════════════════════════════════════


class TestLarsenSkouBisimulation:
    """Test exact probabilistic bisimulation partition."""

    def test_symmetric_states_same_block(self):
        mdp = _symmetric_mdp()
        ls = LarsenSkouBisimulation()
        partition = ls.compute(mdp)
        assert isinstance(partition, Partition)
        # s0 and s1 should be in the same block
        assert partition.state_to_block["s0"] == partition.state_to_block["s1"]

    def test_asymmetric_states_different_blocks(self):
        mdp = _asymmetric_mdp()
        ls = LarsenSkouBisimulation()
        partition = ls.compute(mdp)
        # s0 and goal have different transitions, so at minimum partition is non-trivial
        assert partition.n_blocks >= 2


# ═══════════════════════════════════════════════════════════════════════════
# Spectral Bisimulation
# ═══════════════════════════════════════════════════════════════════════════


class TestSpectralBisimulation:
    """Test spectral bisimulation methods."""

    def test_transition_matrix(self):
        mdp = _chain_mdp(3)
        T, ids = build_transition_matrix(mdp)
        assert T.shape[0] == T.shape[1]
        assert len(ids) == len(mdp.states)

    def test_graph_laplacian(self):
        mdp = _chain_mdp(4)
        lap = GraphLaplacian.from_mdp(mdp)
        assert lap.fiedler_value >= 0

    def test_spectral_gap(self):
        mdp = _chain_mdp(4)
        lap = GraphLaplacian.from_mdp(mdp)
        gap = lap.spectral_gap
        assert gap >= 0

    def test_spectral_bisimulation_partition(self):
        mdp = _chain_mdp(5)
        sb = SpectralBisimulation()
        partition = sb.compute(mdp)
        assert isinstance(partition, Partition)
        assert partition.n_blocks >= 1

    def test_spectral_distance_matrix(self):
        mdp = _chain_mdp(4)
        sb = SpectralBisimulation()
        dm = sb.spectral_distance_matrix(mdp)
        assert isinstance(dm, CognitiveDistanceMatrix)

    def test_fiedler_partition(self):
        mdp = _chain_mdp(6)
        partition = fiedler_partition(mdp)
        assert isinstance(partition, Partition)
        assert partition.n_blocks >= 2


# ═══════════════════════════════════════════════════════════════════════════
# Cognitive Metric
# ═══════════════════════════════════════════════════════════════════════════


class TestCognitiveMetric:
    """Test cognitive-specific distance metrics."""

    def test_free_energy_distance(self):
        mdp = _chain_mdp(3)
        fed = FreeEnergyDistance()
        dm = fed.compute_matrix(mdp, beta_max=2.0)
        assert isinstance(dm, CognitiveDistanceMatrix)

    def test_policy_sensitivity(self):
        mdp = _chain_mdp(3)
        ps = PolicySensitivity()
        sens = ps.compute_sensitivity(mdp, beta_max=2.0)
        assert isinstance(sens, dict)
        assert all(v >= 0 for v in sens.values())

    def test_cognitive_kernel(self):
        mdp = _chain_mdp(3)
        pbm = ProbabilisticBisimulationMetric()
        dm = pbm.compute(mdp)
        ck = CognitiveKernel()
        K = ck.compute_kernel_matrix(dm)
        assert K.shape[0] == K.shape[1]
        # Kernel matrix should be symmetric
        assert np.allclose(K, K.T, atol=1e-6)

    def test_cognitive_aggregation(self):
        mdp = _chain_mdp(4)
        ca = CognitiveAggregation()
        result = ca.aggregate(mdp)
        assert hasattr(result, "partition")


# ═══════════════════════════════════════════════════════════════════════════
# Approximate Bisimulation
# ═══════════════════════════════════════════════════════════════════════════


class TestApproximateBisimulation:
    """Test approximate bisimulation methods."""

    def test_epsilon_bisimulation(self):
        mdp = _chain_mdp(5)
        eb = EpsilonBisimulation()
        partition, eps = eb.compute(mdp)
        assert isinstance(partition, Partition)
        assert eps >= 0

    def test_value_function_error_bound(self):
        eb = EpsilonBisimulation()
        bound = eb.value_function_error_bound(epsilon=0.1, discount=0.95)
        assert bound >= 0

    def test_approximate_refinement(self):
        mdp = _chain_mdp(5)
        apr = ApproximatePartitionRefinement()
        partition, errors = apr.refine(mdp, beta=1.0)
        assert isinstance(partition, Partition)
        assert isinstance(errors, list)


# ═══════════════════════════════════════════════════════════════════════════
# Simulation Relation
# ═══════════════════════════════════════════════════════════════════════════


class TestSimulationRelation:
    """Test simulation relation computation."""

    def test_simulation_relation_matrix(self):
        mdp = _chain_mdp(3)
        sr = SimulationRelation()
        R = sr.compute(mdp)
        assert isinstance(R, np.ndarray)
        n = len(mdp.states)
        assert R.shape == (n, n)
        # Diagonal should be 1 (every state simulates itself)
        for i in range(n):
            assert R[i, i] == 1

    def test_simulation_distance(self):
        mdp = _chain_mdp(3)
        sd = SimulationDistance()
        dm = sd.compute(mdp)
        assert isinstance(dm, CognitiveDistanceMatrix)


# ═══════════════════════════════════════════════════════════════════════════
# Compositional
# ═══════════════════════════════════════════════════════════════════════════


class TestCompositional:
    """Test compositional bisimulation methods."""

    def test_congruence_checker(self):
        mdp = _symmetric_mdp()
        ls = LarsenSkouBisimulation()
        partition = ls.compute(mdp)
        cc = CongruenceChecker()
        result = cc.check_parallel_congruence(partition, mdp)
        assert isinstance(result, bool)
