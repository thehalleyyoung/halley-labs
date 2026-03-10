"""Unit tests for QuotientMDPBuilder.

Tests cover quotient MDP construction, state/transition/action aggregation,
probability normalisation, verification, and special cases (trivial and
discrete partitions).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from usability_oracle.bisimulation.models import Partition
from usability_oracle.bisimulation.partition import PartitionRefinement
from usability_oracle.bisimulation.quotient import QuotientMDPBuilder
from usability_oracle.mdp.models import State, Action, Transition, MDP
from tests.fixtures.sample_mdps import (
    make_two_state_mdp,
    make_cyclic_mdp,
    make_choice_mdp,
    make_large_chain_mdp,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_four_state_pair_mdp() -> MDP:
    """Four active states forming two equivalence pairs for non-trivial partitioning."""
    states = {
        "s0": State(state_id="s0", features={"x": 0.0}, label="s0"),
        "s1": State(state_id="s1", features={"x": 0.1}, label="s1"),
        "s2": State(state_id="s2", features={"x": 0.8}, label="s2"),
        "s3": State(state_id="s3", features={"x": 0.9}, label="s3"),
        "goal": State(state_id="goal", features={"x": 1.0}, label="goal",
                      is_terminal=True, is_goal=True),
    }
    actions = {
        "a": Action(action_id="a", action_type="click", target_node_id="g",
                     description="go"),
    }
    transitions = [
        Transition(source="s0", action="a", target="goal", probability=1.0, cost=0.1),
        Transition(source="s1", action="a", target="goal", probability=1.0, cost=0.1),
        Transition(source="s2", action="a", target="goal", probability=1.0, cost=0.9),
        Transition(source="s3", action="a", target="goal", probability=1.0, cost=0.9),
    ]
    return MDP(states=states, actions=actions, transitions=transitions,
               initial_state="s0", goal_states={"goal"}, discount=0.99)


def _partition_for_four_state() -> Partition:
    """Partition grouping {s0,s1}, {s2,s3}, {goal} for _make_four_state_pair_mdp."""
    return Partition.from_blocks([
        frozenset(["s0", "s1"]),
        frozenset(["s2", "s3"]),
        frozenset(["goal"]),
    ])


def _make_stochastic_mdp() -> MDP:
    """MDP with stochastic transitions for probability normalisation tests."""
    states = {
        "s0": State(state_id="s0", features={"x": 0.0}, label="s0"),
        "s1": State(state_id="s1", features={"x": 0.5}, label="s1"),
        "s2": State(state_id="s2", features={"x": 1.0}, label="s2"),
        "goal": State(state_id="goal", features={"x": 1.0}, label="goal",
                      is_terminal=True, is_goal=True),
    }
    actions = {
        "a": Action(action_id="a", action_type="click", target_node_id="g",
                     description="choose"),
        "b": Action(action_id="b", action_type="click", target_node_id="g",
                     description="finish"),
    }
    transitions = [
        Transition(source="s0", action="a", target="s1", probability=0.6, cost=0.2),
        Transition(source="s0", action="a", target="s2", probability=0.4, cost=0.3),
        Transition(source="s1", action="b", target="goal", probability=1.0, cost=0.1),
        Transition(source="s2", action="b", target="goal", probability=1.0, cost=0.5),
    ]
    return MDP(states=states, actions=actions, transitions=transitions,
               initial_state="s0", goal_states={"goal"}, discount=0.99)


def _get_transition_prob_sums(mdp: MDP) -> dict[tuple[str, str], float]:
    """Compute the sum of transition probabilities per (source, action)."""
    sums: dict[tuple[str, str], float] = defaultdict(float)
    for t in mdp.transitions:
        sums[(t.source, t.action)] += t.probability
    return dict(sums)


# ---------------------------------------------------------------------------
# Tests: build basics
# ---------------------------------------------------------------------------


class TestQuotientBuildBasics:
    """Basic properties of QuotientMDPBuilder.build."""

    def test_build_returns_mdp(self):
        """build() must return an MDP instance."""
        mdp = make_two_state_mdp()
        partition = Partition.discrete(sorted(mdp.states.keys()))
        builder = QuotientMDPBuilder(verify=False)
        quotient = builder.build(mdp, partition)
        assert isinstance(quotient, MDP)

    def test_build_quotient_state_count_matches_blocks(self):
        """Quotient MDP should have exactly as many states as partition blocks."""
        mdp = _make_four_state_pair_mdp()
        partition = _partition_for_four_state()
        builder = QuotientMDPBuilder(verify=False)
        quotient = builder.build(mdp, partition)
        assert quotient.n_states == partition.n_blocks

    def test_build_with_invalid_partition_raises(self):
        """build() should raise ValueError for an invalid partition."""
        mdp = make_two_state_mdp()
        bad_partition = Partition(
            blocks=[frozenset(["start", "goal"]), frozenset(["goal"])],
            state_to_block={"start": 0, "goal": 0},
        )
        builder = QuotientMDPBuilder(verify=False)
        with pytest.raises(ValueError, match="not valid"):
            builder.build(mdp, bad_partition)

    def test_quotient_has_fewer_states_non_discrete(self):
        """Quotient from a non-discrete partition has fewer states than original."""
        mdp = _make_four_state_pair_mdp()
        partition = _partition_for_four_state()  # 3 blocks from 5 states
        builder = QuotientMDPBuilder(verify=False)
        quotient = builder.build(mdp, partition)
        assert quotient.n_states < mdp.n_states

    def test_quotient_state_ids_use_prefix(self):
        """Quotient state IDs should use the configured prefix."""
        mdp = make_two_state_mdp()
        partition = Partition.discrete(sorted(mdp.states.keys()))
        builder = QuotientMDPBuilder(verify=False, state_prefix="Block")
        quotient = builder.build(mdp, partition)
        for sid in quotient.states:
            assert sid.startswith("Block")


# ---------------------------------------------------------------------------
# Tests: discrete partition → quotient ≈ original
# ---------------------------------------------------------------------------


class TestDiscretePartition:
    """Discrete partition should produce a quotient isomorphic to the original."""

    def test_discrete_partition_same_state_count(self):
        """Discrete partition → quotient has same number of states as original."""
        mdp = make_cyclic_mdp()
        partition = Partition.discrete(sorted(mdp.states.keys()))
        builder = QuotientMDPBuilder(verify=False)
        quotient = builder.build(mdp, partition)
        assert quotient.n_states == mdp.n_states

    def test_discrete_partition_same_transition_count(self):
        """Discrete partition → quotient preserves the number of transitions."""
        mdp = make_two_state_mdp()
        partition = Partition.discrete(sorted(mdp.states.keys()))
        builder = QuotientMDPBuilder(verify=False)
        quotient = builder.build(mdp, partition)
        assert quotient.n_transitions == mdp.n_transitions

    def test_discrete_partition_preserves_goal(self):
        """Discrete partition quotient should have goal states."""
        mdp = make_two_state_mdp()
        partition = Partition.discrete(sorted(mdp.states.keys()))
        builder = QuotientMDPBuilder(verify=False)
        quotient = builder.build(mdp, partition)
        assert len(quotient.goal_states) >= 1

    def test_discrete_partition_verify_low_error(self):
        """Verification on discrete-partition quotient should yield ~0 error."""
        mdp = make_two_state_mdp()
        partition = Partition.discrete(sorted(mdp.states.keys()))
        builder = QuotientMDPBuilder(verify=True)
        quotient = builder.build(mdp, partition)
        error = builder.verify_quotient(mdp, quotient, partition)
        assert error == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Tests: trivial partition → single-block quotient
# ---------------------------------------------------------------------------


class TestTrivialPartition:
    """Trivial partition should produce a single-state quotient MDP."""

    def test_trivial_partition_one_state(self):
        """Trivial partition → quotient has exactly 1 state."""
        mdp = make_two_state_mdp()
        partition = Partition.trivial(sorted(mdp.states.keys()))
        builder = QuotientMDPBuilder(verify=False)
        quotient = builder.build(mdp, partition)
        assert quotient.n_states == 1

    def test_trivial_partition_goal_preserved(self):
        """The single quotient state should be marked as goal (since goal ∈ block)."""
        mdp = make_two_state_mdp()
        partition = Partition.trivial(sorted(mdp.states.keys()))
        builder = QuotientMDPBuilder(verify=False)
        quotient = builder.build(mdp, partition)
        assert len(quotient.goal_states) == 1

    def test_trivial_partition_self_transitions(self):
        """All transitions in a trivial quotient go from the single state to itself."""
        mdp = make_cyclic_mdp()
        partition = Partition.trivial(sorted(mdp.states.keys()))
        builder = QuotientMDPBuilder(verify=False)
        quotient = builder.build(mdp, partition)
        for t in quotient.transitions:
            assert t.source == t.target


# ---------------------------------------------------------------------------
# Tests: transition probability normalisation
# ---------------------------------------------------------------------------


class TestTransitionProbabilities:
    """Transition probabilities in the quotient MDP must sum to 1."""

    def test_probabilities_sum_to_one_two_state(self):
        """Probabilities per (source, action) should sum to 1 in the quotient."""
        mdp = make_two_state_mdp()
        partition = Partition.discrete(sorted(mdp.states.keys()))
        builder = QuotientMDPBuilder(verify=False)
        quotient = builder.build(mdp, partition)
        sums = _get_transition_prob_sums(quotient)
        for key, total in sums.items():
            assert total == pytest.approx(1.0, abs=1e-6), (
                f"Probabilities for {key} sum to {total}"
            )

    def test_probabilities_sum_to_one_cyclic(self):
        """Probability normalisation holds for cyclic MDP quotient."""
        mdp = make_cyclic_mdp()
        refiner = PartitionRefinement(max_iterations=100)
        partition = refiner.refine(mdp, beta=2.0, epsilon=0.05)
        builder = QuotientMDPBuilder(verify=False)
        quotient = builder.build(mdp, partition)
        sums = _get_transition_prob_sums(quotient)
        for key, total in sums.items():
            assert total == pytest.approx(1.0, abs=1e-6), (
                f"Probabilities for {key} sum to {total}"
            )

    def test_probabilities_sum_to_one_stochastic(self):
        """Probability normalisation holds for a stochastic MDP with merged blocks."""
        mdp = _make_stochastic_mdp()
        # Merge s1 and s2 into one block
        partition = Partition.from_blocks([
            frozenset(["s0"]),
            frozenset(["s1", "s2"]),
            frozenset(["goal"]),
        ])
        builder = QuotientMDPBuilder(verify=False)
        quotient = builder.build(mdp, partition)
        sums = _get_transition_prob_sums(quotient)
        for key, total in sums.items():
            assert total == pytest.approx(1.0, abs=1e-6), (
                f"Probabilities for {key} sum to {total}"
            )

    def test_all_probabilities_non_negative(self):
        """All transition probabilities in the quotient must be >= 0."""
        mdp = make_cyclic_mdp()
        partition = Partition.trivial(sorted(mdp.states.keys()))
        builder = QuotientMDPBuilder(verify=False)
        quotient = builder.build(mdp, partition)
        for t in quotient.transitions:
            assert t.probability >= 0.0


# ---------------------------------------------------------------------------
# Tests: goal reachability preservation
# ---------------------------------------------------------------------------


class TestGoalReachability:
    """Quotient MDP should preserve goal reachability from the initial state."""

    def test_goal_reachable_two_state(self):
        """Goal must be reachable in the two-state quotient."""
        mdp = make_two_state_mdp()
        partition = Partition.discrete(sorted(mdp.states.keys()))
        builder = QuotientMDPBuilder(verify=False)
        quotient = builder.build(mdp, partition)
        reachable = quotient.reachable_states()
        assert quotient.goal_states & reachable, "Goal not reachable in quotient"

    def test_goal_reachable_cyclic(self):
        """Goal must be reachable in the cyclic MDP quotient."""
        mdp = make_cyclic_mdp()
        refiner = PartitionRefinement(max_iterations=100)
        partition = refiner.refine(mdp, beta=2.0, epsilon=0.05)
        builder = QuotientMDPBuilder(verify=False)
        quotient = builder.build(mdp, partition)
        reachable = quotient.reachable_states()
        assert quotient.goal_states & reachable, "Goal not reachable in quotient"

    def test_goal_reachable_chain(self):
        """Goal must be reachable in the chain MDP quotient."""
        mdp = make_large_chain_mdp(n=8)
        refiner = PartitionRefinement(max_iterations=200)
        partition = refiner.refine(mdp, beta=2.0, epsilon=0.1)
        builder = QuotientMDPBuilder(verify=False)
        quotient = builder.build(mdp, partition)
        reachable = quotient.reachable_states()
        assert quotient.goal_states & reachable, "Goal not reachable in quotient"

    def test_goal_reachable_choice(self):
        """Goal must be reachable in the choice MDP quotient."""
        mdp = make_choice_mdp(n_choices=4)
        partition = Partition.discrete(sorted(mdp.states.keys()))
        builder = QuotientMDPBuilder(verify=False)
        quotient = builder.build(mdp, partition)
        reachable = quotient.reachable_states()
        assert quotient.goal_states & reachable, "Goal not reachable in quotient"


# ---------------------------------------------------------------------------
# Tests: verify_quotient
# ---------------------------------------------------------------------------


class TestVerifyQuotient:
    """Tests for QuotientMDPBuilder.verify_quotient."""

    def test_verify_returns_float(self):
        """verify_quotient must return a float."""
        mdp = make_two_state_mdp()
        partition = Partition.discrete(sorted(mdp.states.keys()))
        builder = QuotientMDPBuilder()
        quotient = builder.build(mdp, partition)
        error = builder.verify_quotient(mdp, quotient, partition)
        assert isinstance(error, float)

    def test_verify_non_negative(self):
        """Abstraction error must be non-negative."""
        mdp = make_cyclic_mdp()
        refiner = PartitionRefinement(max_iterations=100)
        partition = refiner.refine(mdp, beta=2.0, epsilon=0.05)
        builder = QuotientMDPBuilder()
        quotient = builder.build(mdp, partition)
        error = builder.verify_quotient(mdp, quotient, partition)
        assert error >= 0.0

    def test_verify_discrete_zero_error(self):
        """Discrete partition quotient should have zero abstraction error."""
        mdp = make_cyclic_mdp()
        partition = Partition.discrete(sorted(mdp.states.keys()))
        builder = QuotientMDPBuilder()
        quotient = builder.build(mdp, partition)
        error = builder.verify_quotient(mdp, quotient, partition)
        assert error == pytest.approx(0.0, abs=1e-9)

    def test_verify_trivial_bounded_error(self):
        """Trivial partition quotient should have bounded abstraction error."""
        mdp = make_cyclic_mdp()
        partition = Partition.trivial(sorted(mdp.states.keys()))
        builder = QuotientMDPBuilder()
        quotient = builder.build(mdp, partition)
        error = builder.verify_quotient(mdp, quotient, partition)
        # Error should be finite and non-negative
        assert 0.0 <= error < float("inf")


# ---------------------------------------------------------------------------
# Tests: state feature aggregation
# ---------------------------------------------------------------------------


class TestFeatureAggregation:
    """Tests for how state features are aggregated in the quotient."""

    def test_block_features_contain_mean(self):
        """Quotient states should have mean features for numeric attributes."""
        mdp = _make_four_state_pair_mdp()
        partition = _partition_for_four_state()
        builder = QuotientMDPBuilder(verify=False)
        quotient = builder.build(mdp, partition)
        # Block 0 = {s0(x=0.0), s1(x=0.1)} → x_mean = 0.05
        block0_id = f"{builder.state_prefix}0"
        features = quotient.states[block0_id].features
        assert "x_mean" in features
        assert features["x_mean"] == pytest.approx(0.05, abs=1e-9)

    def test_block_features_contain_std(self):
        """Quotient states should have std features for numeric attributes."""
        mdp = _make_four_state_pair_mdp()
        partition = _partition_for_four_state()
        builder = QuotientMDPBuilder(verify=False)
        quotient = builder.build(mdp, partition)
        block0_id = f"{builder.state_prefix}0"
        features = quotient.states[block0_id].features
        assert "x_std" in features
        assert features["x_std"] >= 0.0

    def test_block_features_contain_min_max(self):
        """Quotient states should include min and max feature statistics."""
        mdp = _make_four_state_pair_mdp()
        partition = _partition_for_four_state()
        builder = QuotientMDPBuilder(verify=False)
        quotient = builder.build(mdp, partition)
        block1_id = f"{builder.state_prefix}1"
        features = quotient.states[block1_id].features
        assert "x_min" in features
        assert "x_max" in features
        assert features["x_min"] <= features["x_max"]

    def test_block_size_in_features(self):
        """Quotient state features should include block_size."""
        mdp = _make_four_state_pair_mdp()
        partition = _partition_for_four_state()
        builder = QuotientMDPBuilder(verify=False)
        quotient = builder.build(mdp, partition)
        block0_id = f"{builder.state_prefix}0"
        assert quotient.states[block0_id].features["block_size"] == 2.0

    def test_singleton_block_features(self):
        """Singleton block features should match the original state features."""
        mdp = _make_four_state_pair_mdp()
        partition = _partition_for_four_state()
        builder = QuotientMDPBuilder(verify=False)
        quotient = builder.build(mdp, partition)
        # Block 2 = {goal} with x=1.0
        block2_id = f"{builder.state_prefix}2"
        features = quotient.states[block2_id].features
        assert features["x_mean"] == pytest.approx(1.0, abs=1e-9)
        assert features["x_std"] == pytest.approx(0.0, abs=1e-9)

    def test_metadata_contains_members(self):
        """Quotient state metadata should list block members."""
        mdp = _make_four_state_pair_mdp()
        partition = _partition_for_four_state()
        builder = QuotientMDPBuilder(verify=False)
        quotient = builder.build(mdp, partition)
        block0_id = f"{builder.state_prefix}0"
        members = quotient.states[block0_id].metadata["members"]
        assert set(members) == {"s0", "s1"}


# ---------------------------------------------------------------------------
# Tests: end-to-end with PartitionRefinement
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Integration tests combining PartitionRefinement + QuotientMDPBuilder."""

    def test_refine_then_build_cyclic(self):
        """Full pipeline: refine → build quotient for cyclic MDP."""
        mdp = make_cyclic_mdp()
        refiner = PartitionRefinement(max_iterations=100)
        partition = refiner.refine(mdp, beta=2.0, epsilon=0.05)
        builder = QuotientMDPBuilder(verify=True)
        quotient = builder.build(mdp, partition)
        assert quotient.n_states <= mdp.n_states
        assert quotient.n_states == partition.n_blocks
        sums = _get_transition_prob_sums(quotient)
        for key, total in sums.items():
            assert total == pytest.approx(1.0, abs=1e-6)

    def test_refine_then_build_chain(self):
        """Full pipeline: refine → build quotient for chain MDP."""
        mdp = make_large_chain_mdp(n=10)
        refiner = PartitionRefinement(max_iterations=200)
        partition = refiner.refine(mdp, beta=3.0, epsilon=0.1)
        builder = QuotientMDPBuilder(verify=False)
        quotient = builder.build(mdp, partition)
        assert 1 <= quotient.n_states <= 10
        assert quotient.goal_states & quotient.reachable_states()

    def test_quotient_discount_and_initial_state(self):
        """Quotient MDP should preserve discount factor and have valid initial state."""
        mdp = make_cyclic_mdp()
        partition = Partition.discrete(sorted(mdp.states.keys()))
        builder = QuotientMDPBuilder(verify=False)
        quotient = builder.build(mdp, partition)
        assert quotient.discount == mdp.discount
        assert quotient.initial_state in quotient.states
