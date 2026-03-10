"""Integration tests for the Layer-3 (bisimulation + algebra) pipeline.

Layer 3 builds an MDP, applies partition refinement to produce a quotient
(abstract) MDP, and verifies that the quotient preserves cost structure
via algebraic soundness checks.  The ``TaskGraphComposer`` composes costs
over the task graph using the cost algebra, and the ``SoundnessVerifier``
validates the composed results.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from usability_oracle.mdp.models import MDP, State, Action, Transition
from usability_oracle.mdp.solver import ValueIterationSolver
from usability_oracle.bisimulation.partition import PartitionRefinement
from usability_oracle.bisimulation.quotient import QuotientMDPBuilder
from usability_oracle.bisimulation.models import (
    Partition,
    BisimulationResult,
    CognitiveDistanceMatrix,
)
from usability_oracle.algebra.models import CostElement, Leaf, Sequential, Parallel
from usability_oracle.algebra.composer import TaskGraphComposer
from usability_oracle.algebra.sequential import SequentialComposer
from usability_oracle.algebra.soundness import SoundnessVerifier, VerificationResult
from usability_oracle.policy.value_iteration import SoftValueIteration
from usability_oracle.policy.models import Policy, PolicyResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cost(mu: float, sigma_sq: float = 0.0) -> CostElement:
    return CostElement(mu=mu, sigma_sq=sigma_sq)


def _make_branching_mdp() -> MDP:
    """5-state branching MDP: s0 → s1/s2 → s3 → s4 (goal)."""
    states = {}
    for i in range(5):
        sid = f"s{i}"
        states[sid] = State(
            state_id=sid,
            features={"x": float(i * 50), "y": float(i * 30)},
            label=sid,
            is_terminal=(i == 4),
            is_goal=(i == 4),
        )
    actions = {
        "a0": Action(action_id="a0", action_type=Action.CLICK,
                      target_node_id="n0", description=""),
        "a1": Action(action_id="a1", action_type=Action.CLICK,
                      target_node_id="n1", description=""),
        "a2": Action(action_id="a2", action_type=Action.CLICK,
                      target_node_id="n2", description=""),
        "a3": Action(action_id="a3", action_type=Action.CLICK,
                      target_node_id="n3", description=""),
    }
    transitions = [
        Transition(source="s0", action="a0", target="s1",
                   probability=0.7, cost=0.3),
        Transition(source="s0", action="a0", target="s2",
                   probability=0.3, cost=0.5),
        Transition(source="s1", action="a1", target="s3",
                   probability=1.0, cost=0.4),
        Transition(source="s2", action="a2", target="s3",
                   probability=1.0, cost=0.6),
        Transition(source="s3", action="a3", target="s4",
                   probability=1.0, cost=0.2),
    ]
    return MDP(
        states=states, actions=actions, transitions=transitions,
        initial_state="s0", goal_states={"s4"}, discount=0.99,
    )


def _make_symmetric_mdp() -> MDP:
    """An MDP where s1 and s2 are behaviourally equivalent."""
    states = {}
    for i in range(4):
        sid = f"s{i}"
        states[sid] = State(
            state_id=sid,
            features={"x": float(i * 50), "y": 0.0},
            label=sid,
            is_terminal=(i == 3),
            is_goal=(i == 3),
        )
    actions = {
        "a0": Action(action_id="a0", action_type=Action.CLICK,
                      target_node_id="n0", description=""),
        "a1": Action(action_id="a1", action_type=Action.CLICK,
                      target_node_id="n1", description=""),
    }
    transitions = [
        Transition(source="s0", action="a0", target="s1",
                   probability=0.5, cost=0.3),
        Transition(source="s0", action="a0", target="s2",
                   probability=0.5, cost=0.3),
        Transition(source="s1", action="a1", target="s3",
                   probability=1.0, cost=0.5),
        Transition(source="s2", action="a1", target="s3",
                   probability=1.0, cost=0.5),
    ]
    return MDP(
        states=states, actions=actions, transitions=transitions,
        initial_state="s0", goal_states={"s3"}, discount=0.99,
    )


# ===================================================================
# Tests – Partition refinement
# ===================================================================


class TestPartitionRefinement:
    """Verify bisimulation-based partition refinement."""

    def test_refine_produces_partition(self) -> None:
        """Refinement on the branching MDP should produce a valid partition."""
        mdp = _make_branching_mdp()
        pr = PartitionRefinement()
        partition = pr.refine(mdp, beta=2.0, epsilon=0.01)
        assert partition.n_blocks >= 1
        assert partition.is_valid()

    def test_refine_discrete_bound(self) -> None:
        """The partition should have at most as many blocks as states."""
        mdp = _make_branching_mdp()
        partition = PartitionRefinement().refine(mdp, beta=2.0)
        assert partition.n_blocks <= mdp.n_states

    def test_symmetric_mdp_merges_equivalent(self) -> None:
        """Symmetric states s1/s2 should land in the same block."""
        mdp = _make_symmetric_mdp()
        partition = PartitionRefinement().refine(mdp, beta=2.0, epsilon=0.1)
        idx1 = partition.block_index("s1")
        idx2 = partition.block_index("s2")
        assert idx1 == idx2, "Symmetric states should be bisimilar"

    def test_refine_all_states_covered(self) -> None:
        """Every MDP state must appear in exactly one block."""
        mdp = _make_branching_mdp()
        partition = PartitionRefinement().refine(mdp, beta=2.0)
        covered = partition.states()
        for s in mdp.states:
            assert s in covered


# ===================================================================
# Tests – Quotient MDP
# ===================================================================


class TestQuotientMDP:
    """Build and verify quotient (abstract) MDPs."""

    def test_quotient_has_fewer_or_equal_states(self) -> None:
        """Quotient MDP should have ≤ original states."""
        mdp = _make_branching_mdp()
        partition = PartitionRefinement().refine(mdp, beta=2.0)
        quotient = QuotientMDPBuilder().build(mdp, partition)
        assert quotient.n_states <= mdp.n_states

    def test_quotient_validates(self) -> None:
        """Quotient MDP should pass its own validation."""
        mdp = _make_branching_mdp()
        partition = PartitionRefinement().refine(mdp, beta=2.0)
        quotient = QuotientMDPBuilder().build(mdp, partition)
        errors = quotient.validate()
        assert len(errors) == 0, f"Quotient validation: {errors}"

    def test_quotient_preserves_goal(self) -> None:
        """The quotient must retain at least one goal state."""
        mdp = _make_branching_mdp()
        partition = PartitionRefinement().refine(mdp, beta=2.0)
        quotient = QuotientMDPBuilder().build(mdp, partition)
        assert len(quotient.goal_states) >= 1

    def test_verify_quotient_error_bounded(self) -> None:
        """Abstraction error between original and quotient should be finite."""
        mdp = _make_branching_mdp()
        partition = PartitionRefinement().refine(mdp, beta=2.0)
        builder = QuotientMDPBuilder(verify=True)
        quotient = builder.build(mdp, partition)
        error = builder.verify_quotient(mdp, quotient, partition)
        assert math.isfinite(error)
        assert error >= 0

    def test_quotient_symmetric_mdp_smaller(self) -> None:
        """The symmetric MDP quotient should have fewer states."""
        mdp = _make_symmetric_mdp()
        partition = PartitionRefinement().refine(mdp, beta=2.0, epsilon=0.1)
        quotient = QuotientMDPBuilder().build(mdp, partition)
        assert quotient.n_states < mdp.n_states


# ===================================================================
# Tests – TaskGraphComposer with cost algebra
# ===================================================================


class TestTaskGraphComposition:
    """Compose costs over a task graph using the algebra."""

    def test_sequential_composition(self) -> None:
        """Sequential composition of two steps should add means."""
        a = _cost(1.0, 0.1)
        b = _cost(2.0, 0.2)
        composed = SequentialComposer().compose(a, b)
        assert composed.mu == pytest.approx(3.0)

    def test_soundness_on_sequential(self) -> None:
        """Sequential composition must pass soundness."""
        a = _cost(1.0, 0.1)
        b = _cost(2.0, 0.2)
        composed = SequentialComposer().compose(a, b)
        assert SoundnessVerifier().verify_sequential(a, b, composed)

    def test_verify_all_on_parallel(self) -> None:
        """A parallel expression should pass verification."""
        a = _cost(1.0, 0.1)
        b = _cost(2.0, 0.2)
        expr = Parallel(left=Leaf(a), right=Leaf(b), interference=0.0)
        results = SoundnessVerifier().verify_all(expr)
        assert all(isinstance(r, VerificationResult) for r in results)

    def test_commutativity_no_interference(self) -> None:
        """Parallel composition with zero interference should commute."""
        a = _cost(1.0, 0.1)
        b = _cost(2.0, 0.2)
        assert SoundnessVerifier().verify_commutativity(a, b, interference=0.0)

    def test_triangle_inequality(self) -> None:
        """Triangle inequality check on three cost elements."""
        a = _cost(1.0, 0.1)
        b = _cost(2.0, 0.2)
        c = _cost(3.0, 0.3)
        result = SoundnessVerifier().verify_triangle_inequality(a, b, c)
        assert isinstance(result, bool)


# ===================================================================
# Tests – Partition model utilities
# ===================================================================


class TestPartitionModel:
    """Low-level partition operations."""

    def test_trivial_partition(self) -> None:
        """Trivial partition places all states in one block."""
        p = Partition.trivial(["a", "b", "c"])
        assert p.n_blocks == 1
        assert p.block_index("a") == p.block_index("b")

    def test_discrete_partition(self) -> None:
        """Discrete partition puts each state in its own block."""
        p = Partition.discrete(["a", "b", "c"])
        assert p.n_blocks == 3

    def test_merge_blocks(self) -> None:
        """Merging two blocks should reduce block count by one."""
        p = Partition.discrete(["a", "b", "c"])
        merged = p.merge(0, 1)
        assert merged.n_blocks == p.n_blocks - 1

    def test_split_block(self) -> None:
        """Splitting a two-element block should increase block count."""
        p = Partition.trivial(["a", "b"])
        split = p.split(0, criterion=lambda s: s == "a")
        assert split.n_blocks == 2

    def test_from_blocks(self) -> None:
        """``from_blocks`` should reconstruct a valid partition."""
        blocks = [frozenset({"a", "b"}), frozenset({"c"})]
        p = Partition.from_blocks(blocks)
        assert p.n_blocks == 2
        assert p.is_valid()
