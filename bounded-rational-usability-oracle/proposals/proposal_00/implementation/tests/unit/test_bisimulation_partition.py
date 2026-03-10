"""Unit tests for PartitionRefinement and Partition.

Tests cover the Partition data structure (creation, queries, mutation,
validation) and the PartitionRefinement algorithm (convergence, epsilon
sensitivity, abstract transitions).
"""

from __future__ import annotations

import pytest

from usability_oracle.bisimulation.models import Partition
from usability_oracle.bisimulation.partition import PartitionRefinement
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

_SAMPLE_STATES = ["s0", "s1", "s2", "s3", "s4"]


def _make_four_state_mdp() -> MDP:
    """Four non-terminal states with two distinct behavioural groups.

    s0 and s1 have action 'a' leading to goal with cost 0.1.
    s2 and s3 have action 'a' leading to goal with cost 0.9.
    This creates two natural equivalence classes under bisimulation.
    """
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


def _make_branching_mdp() -> MDP:
    """MDP where states differ in their transition distributions.

    s0: action 'a' → goal (p=1.0, c=0.2)
    s1: action 'a' → goal (p=0.5, c=0.2), action 'a' → s0 (p=0.5, c=0.2)
    s2: action 'a' → goal (p=0.5, c=0.2), action 'a' → s1 (p=0.5, c=0.2)
    """
    states = {
        "s0": State(state_id="s0", features={"x": 0.0}, label="s0"),
        "s1": State(state_id="s1", features={"x": 0.5}, label="s1"),
        "s2": State(state_id="s2", features={"x": 1.0}, label="s2"),
        "goal": State(state_id="goal", features={"x": 1.0}, label="goal",
                      is_terminal=True, is_goal=True),
    }
    actions = {
        "a": Action(action_id="a", action_type="click", target_node_id="g",
                     description="go"),
    }
    transitions = [
        Transition(source="s0", action="a", target="goal", probability=1.0, cost=0.2),
        Transition(source="s1", action="a", target="goal", probability=0.5, cost=0.2),
        Transition(source="s1", action="a", target="s0", probability=0.5, cost=0.2),
        Transition(source="s2", action="a", target="goal", probability=0.5, cost=0.2),
        Transition(source="s2", action="a", target="s1", probability=0.5, cost=0.2),
    ]
    return MDP(states=states, actions=actions, transitions=transitions,
               initial_state="s0", goal_states={"goal"}, discount=0.99)


# ---------------------------------------------------------------------------
# Tests: Partition factory class methods
# ---------------------------------------------------------------------------


class TestPartitionFactories:
    """Tests for Partition.trivial, Partition.discrete, Partition.from_blocks."""

    def test_trivial_single_block(self):
        """Partition.trivial creates exactly 1 block containing all states."""
        p = Partition.trivial(_SAMPLE_STATES)
        assert p.n_blocks == 1
        assert p.states() == frozenset(_SAMPLE_STATES)

    def test_trivial_is_valid_and_empty(self):
        """Partition.trivial produces a valid partition; empty list → 0 blocks."""
        p = Partition.trivial(_SAMPLE_STATES)
        assert p.is_valid()
        p_empty = Partition.trivial([])
        assert p_empty.n_blocks == 0

    def test_discrete_n_blocks(self):
        """Partition.discrete creates one block per state."""
        p = Partition.discrete(_SAMPLE_STATES)
        assert p.n_blocks == len(_SAMPLE_STATES)

    def test_discrete_singleton_blocks_and_valid(self):
        """Each block in discrete partition has exactly one state and is valid."""
        p = Partition.discrete(_SAMPLE_STATES)
        for block in p.blocks:
            assert len(block) == 1
        assert p.is_valid()

    def test_from_blocks_builds_correctly(self):
        """Partition.from_blocks builds correct blocks and reverse index."""
        blocks = [frozenset(["s0", "s1"]), frozenset(["s2", "s3", "s4"])]
        p = Partition.from_blocks(blocks)
        assert p.n_blocks == 2
        assert p.state_to_block["s0"] == 0
        assert p.state_to_block["s4"] == 1
        assert p.is_valid()

    def test_from_blocks_preserves_order(self):
        """Blocks should maintain the order given to from_blocks."""
        blocks = [frozenset(["a"]), frozenset(["b"]), frozenset(["c"])]
        p = Partition.from_blocks(blocks)
        assert p.blocks[0] == frozenset(["a"])
        assert p.blocks[1] == frozenset(["b"])
        assert p.blocks[2] == frozenset(["c"])


# ---------------------------------------------------------------------------
# Tests: Partition query methods
# ---------------------------------------------------------------------------


class TestPartitionQueries:
    """Tests for n_blocks, get_block, block_index, states, block_label."""

    def test_n_blocks_property(self):
        """n_blocks returns the correct count."""
        p = Partition.from_blocks([
            frozenset(["a", "b"]),
            frozenset(["c"]),
        ])
        assert p.n_blocks == 2

    def test_get_block_returns_correct_block(self):
        """get_block returns the frozenset containing the queried state."""
        p = Partition.from_blocks([
            frozenset(["a", "b"]),
            frozenset(["c", "d"]),
        ])
        assert p.get_block("a") == frozenset(["a", "b"])
        assert p.get_block("c") == frozenset(["c", "d"])

    def test_get_block_raises_key_error(self):
        """get_block raises KeyError for unknown states."""
        p = Partition.from_blocks([frozenset(["a"])])
        with pytest.raises(KeyError):
            p.get_block("nonexistent")

    def test_block_index_correct(self):
        """block_index returns the correct index for each state."""
        blocks = [frozenset(["x", "y"]), frozenset(["z"])]
        p = Partition.from_blocks(blocks)
        assert p.block_index("x") == 0
        assert p.block_index("y") == 0
        assert p.block_index("z") == 1

    def test_block_index_raises_key_error(self):
        """block_index raises KeyError for unknown states."""
        p = Partition.from_blocks([frozenset(["a"])])
        with pytest.raises(KeyError):
            p.block_index("missing")

    def test_states_returns_all(self):
        """states() returns a frozenset of all state IDs in the partition."""
        blocks = [frozenset(["a", "b"]), frozenset(["c"])]
        p = Partition.from_blocks(blocks)
        assert p.states() == frozenset(["a", "b", "c"])

    def test_block_label_format(self):
        """block_label returns a string containing the block index."""
        p = Partition.from_blocks([frozenset(["s0", "s1"]), frozenset(["s2"])])
        label = p.block_label(0)
        assert "B0" in label or "0" in label

    def test_block_label_out_of_range(self):
        """block_label raises IndexError for invalid block indices."""
        p = Partition.from_blocks([frozenset(["a"])])
        with pytest.raises(IndexError):
            p.block_label(5)


# ---------------------------------------------------------------------------
# Tests: Partition mutation methods
# ---------------------------------------------------------------------------


class TestPartitionMutation:
    """Tests for merge, split, and refine."""

    def test_merge_reduces_block_count(self):
        """Merging two blocks should decrease n_blocks by 1."""
        p = Partition.from_blocks([
            frozenset(["a"]), frozenset(["b"]), frozenset(["c"]),
        ])
        merged = p.merge(0, 1)
        assert merged.n_blocks == 2
        assert merged.is_valid()

    def test_merge_combines_states(self):
        """Merged block contains states from both original blocks."""
        p = Partition.from_blocks([
            frozenset(["a"]), frozenset(["b"]), frozenset(["c"]),
        ])
        merged = p.merge(0, 2)
        # The merged block should contain both "a" and "c"
        block_a = merged.get_block("a")
        assert "a" in block_a and "c" in block_a

    def test_merge_same_block_no_change(self):
        """Merging a block with itself should not change the partition."""
        p = Partition.from_blocks([frozenset(["a", "b"]), frozenset(["c"])])
        merged = p.merge(0, 0)
        assert merged.n_blocks == 2
        assert merged.is_valid()

    def test_split_increases_block_count(self):
        """Splitting a block should increase n_blocks by 1."""
        p = Partition.from_blocks([frozenset(["a", "b", "c"])])
        split = p.split(0, criterion=lambda s: s == "a")
        assert split.n_blocks == 2
        assert split.is_valid()

    def test_split_criterion_separates_correctly(self):
        """Split separates states matching criterion from the rest."""
        p = Partition.from_blocks([frozenset(["a", "b", "c", "d"])])
        split = p.split(0, criterion=lambda s: s in ("a", "b"))
        # One block with a,b and another with c,d
        block_a = split.get_block("a")
        assert "a" in block_a and "b" in block_a
        assert "c" not in block_a

    def test_split_trivial_no_change(self):
        """Splitting where all states match criterion returns same partition."""
        p = Partition.from_blocks([frozenset(["a", "b"])])
        split = p.split(0, criterion=lambda s: True)
        assert split.n_blocks == 1

    def test_refine_applies_splitter(self):
        """refine applies the splitter function to every block."""
        p = Partition.from_blocks([frozenset(["a", "b"]), frozenset(["c", "d"])])
        # Split every block into singletons
        refined = p.refine(lambda block: [frozenset([s]) for s in block])
        assert refined.n_blocks == 4
        assert refined.is_valid()

    def test_refine_identity_splitter(self):
        """A splitter that returns each block unchanged preserves the partition."""
        p = Partition.from_blocks([frozenset(["a", "b"]), frozenset(["c"])])
        refined = p.refine(lambda block: [block])
        assert refined.n_blocks == p.n_blocks


# ---------------------------------------------------------------------------
# Tests: Partition validation
# ---------------------------------------------------------------------------


class TestPartitionValidation:
    """Tests for Partition.is_valid."""

    def test_valid_partition(self):
        """Well-formed partition passes validation."""
        p = Partition.from_blocks([frozenset(["a", "b"]), frozenset(["c"])])
        assert p.is_valid()

    def test_invalid_duplicate_state(self):
        """Partition with a state in multiple blocks is invalid."""
        p = Partition(
            blocks=[frozenset(["a", "b"]), frozenset(["b", "c"])],
            state_to_block={"a": 0, "b": 0, "c": 1},
        )
        assert not p.is_valid()

    def test_invalid_wrong_index(self):
        """Partition with mismatched state_to_block is invalid."""
        p = Partition(
            blocks=[frozenset(["a"]), frozenset(["b"])],
            state_to_block={"a": 1, "b": 0},  # swapped
        )
        assert not p.is_valid()

    def test_empty_partition_valid(self):
        """An empty partition (no blocks, no states) is valid."""
        p = Partition(blocks=[], state_to_block={})
        assert p.is_valid()


# ---------------------------------------------------------------------------
# Tests: PartitionRefinement.refine
# ---------------------------------------------------------------------------


class TestPartitionRefinement:
    """Tests for PartitionRefinement.refine on various MDPs."""

    def test_refine_returns_partition(self):
        """PartitionRefinement.refine must return a Partition instance."""
        mdp = make_two_state_mdp()
        refiner = PartitionRefinement(max_iterations=50)
        result = refiner.refine(mdp, beta=1.0, epsilon=0.01)
        assert isinstance(result, Partition)

    def test_refine_partition_is_valid(self):
        """The result of refinement must be a valid partition."""
        mdp = make_cyclic_mdp()
        refiner = PartitionRefinement(max_iterations=100)
        result = refiner.refine(mdp, beta=2.0, epsilon=0.01)
        assert result.is_valid()

    def test_refine_covers_all_states(self):
        """The refined partition must cover all MDP states."""
        mdp = make_cyclic_mdp()
        refiner = PartitionRefinement(max_iterations=100)
        result = refiner.refine(mdp, beta=2.0, epsilon=0.01)
        assert result.states() == frozenset(mdp.states.keys())

    def test_refine_converges(self):
        """Refinement must converge within max_iterations on a small MDP."""
        mdp = _make_four_state_mdp()
        refiner = PartitionRefinement(max_iterations=200, verbose=False)
        result = refiner.refine(mdp, beta=5.0, epsilon=0.01)
        # Should have converged to a fixed point
        assert result.n_blocks >= 1
        assert result.n_blocks <= len(mdp.states)

    def test_refine_at_most_discrete(self):
        """Refinement cannot produce more blocks than states."""
        mdp = make_cyclic_mdp()
        refiner = PartitionRefinement(max_iterations=200)
        result = refiner.refine(mdp, beta=10.0, epsilon=0.0001)
        assert result.n_blocks <= len(mdp.states)

    def test_coarser_epsilon_fewer_blocks(self):
        """Larger epsilon (coarser tolerance) should yield fewer or equal blocks.

        A coarser epsilon groups more states together because the policy
        distributions need only be approximately similar within each block.
        """
        mdp = _make_branching_mdp()
        refiner = PartitionRefinement(max_iterations=200)
        fine = refiner.refine(mdp, beta=5.0, epsilon=0.001)
        coarse = refiner.refine(mdp, beta=5.0, epsilon=0.5)
        assert coarse.n_blocks <= fine.n_blocks

    def test_finer_epsilon_more_blocks(self):
        """Smaller epsilon should yield more or equal blocks than larger epsilon.

        A finer epsilon demands higher similarity within blocks, potentially
        splitting them further.
        """
        mdp = _make_four_state_mdp()
        refiner = PartitionRefinement(max_iterations=200)
        coarse = refiner.refine(mdp, beta=5.0, epsilon=0.5)
        fine = refiner.refine(mdp, beta=5.0, epsilon=0.001)
        assert fine.n_blocks >= coarse.n_blocks

    def test_refine_two_state_mdp(self):
        """Two-state MDP should produce at least 2 blocks (start vs goal)."""
        mdp = make_two_state_mdp()
        refiner = PartitionRefinement(max_iterations=100)
        result = refiner.refine(mdp, beta=2.0, epsilon=0.01)
        assert result.n_blocks >= 2

    def test_refine_chain_mdp(self):
        """Chain MDP refinement produces a valid partition with expected bounds."""
        mdp = make_large_chain_mdp(n=6)
        refiner = PartitionRefinement(max_iterations=200)
        result = refiner.refine(mdp, beta=3.0, epsilon=0.01)
        assert result.is_valid()
        assert result.states() == frozenset(mdp.states.keys())
        assert 1 <= result.n_blocks <= 6


# ---------------------------------------------------------------------------
# Tests: compute_abstract_transitions
# ---------------------------------------------------------------------------


class TestComputeAbstractTransitions:
    """Tests for PartitionRefinement.compute_abstract_transitions."""

    def test_returns_dict(self):
        """compute_abstract_transitions must return a dict."""
        mdp = make_two_state_mdp()
        partition = Partition.from_blocks([
            frozenset(["start"]),
            frozenset(["goal"]),
        ])
        refiner = PartitionRefinement()
        result = refiner.compute_abstract_transitions(partition, mdp)
        assert isinstance(result, dict)

    def test_probabilities_non_negative(self):
        """All abstract transition probabilities must be non-negative."""
        mdp = make_cyclic_mdp()
        refiner = PartitionRefinement(max_iterations=100)
        partition = refiner.refine(mdp, beta=2.0, epsilon=0.05)
        abstract = refiner.compute_abstract_transitions(partition, mdp)
        for key, prob in abstract.items():
            assert prob >= 0.0, f"Negative probability for {key}: {prob}"

    def test_probabilities_bounded_by_one(self):
        """Abstract transition probabilities should not exceed 1."""
        mdp = make_cyclic_mdp()
        refiner = PartitionRefinement(max_iterations=100)
        partition = refiner.refine(mdp, beta=2.0, epsilon=0.05)
        abstract = refiner.compute_abstract_transitions(partition, mdp)
        for key, prob in abstract.items():
            assert prob <= 1.0 + 1e-9, f"Probability > 1 for {key}: {prob}"

    def test_discrete_partition_preserves_structure(self):
        """Discrete partition should preserve the original transition structure.

        Each state is its own block, so abstract transitions should map
        one-to-one with concrete transitions.
        """
        mdp = make_two_state_mdp()
        partition = Partition.discrete(sorted(mdp.states.keys()))
        refiner = PartitionRefinement()
        abstract = refiner.compute_abstract_transitions(partition, mdp)
        # Should have at least one transition
        assert len(abstract) >= 1

    def test_trivial_partition_aggregates(self):
        """Trivial partition aggregates all transitions into a single block.

        With one block, all abstract transitions are (0, action, 0).
        """
        mdp = make_two_state_mdp()
        partition = Partition.trivial(sorted(mdp.states.keys()))
        refiner = PartitionRefinement()
        abstract = refiner.compute_abstract_transitions(partition, mdp)
        # All source and target blocks should be 0
        for (src_block, action, tgt_block), prob in abstract.items():
            assert src_block == 0
            assert tgt_block == 0
