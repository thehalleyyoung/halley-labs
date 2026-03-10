"""Tests for usability_oracle.montecarlo.sampler.

Verifies trajectory sampling: single trajectories, batch sampling,
reproducibility, early termination, max_steps, and importance weighting.
"""

from __future__ import annotations

from typing import Dict, FrozenSet

import numpy as np
import pytest

from usability_oracle.montecarlo.sampler import TrajectorySamplerImpl
from usability_oracle.montecarlo.types import (
    MCConfig,
    SamplingStrategy,
    TerminationReason,
)


# =====================================================================
# Fixtures: simple MDP models as dicts
# =====================================================================

@pytest.fixture
def simple_transition() -> Dict[str, Dict[str, Dict[str, float]]]:
    """3-state deterministic transition model: start -> mid -> goal."""
    return {
        "start": {
            "go": {"mid": 1.0},
        },
        "mid": {
            "finish": {"goal": 1.0},
        },
    }


@pytest.fixture
def simple_cost() -> Dict[str, Dict[str, float]]:
    return {
        "start": {"go": 0.5},
        "mid": {"finish": 1.0},
    }


@pytest.fixture
def simple_policy() -> Dict[str, Dict[str, float]]:
    return {
        "start": {"go": 1.0},
        "mid": {"finish": 1.0},
    }


@pytest.fixture
def goal_states() -> FrozenSet[str]:
    return frozenset({"goal"})


@pytest.fixture
def initial_dist() -> Dict[str, float]:
    return {"start": 1.0}


@pytest.fixture
def branching_transition() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Branching MDP with stochastic transitions."""
    return {
        "s0": {
            "a0": {"s1": 0.6, "s2": 0.4},
            "a1": {"s0": 1.0},
        },
        "s1": {
            "a0": {"goal": 1.0},
        },
        "s2": {
            "a0": {"goal": 1.0},
        },
    }


@pytest.fixture
def branching_cost() -> Dict[str, Dict[str, float]]:
    return {
        "s0": {"a0": 0.3, "a1": 0.1},
        "s1": {"a0": 0.5},
        "s2": {"a0": 0.7},
    }


@pytest.fixture
def branching_policy() -> Dict[str, Dict[str, float]]:
    return {
        "s0": {"a0": 0.8, "a1": 0.2},
        "s1": {"a0": 1.0},
        "s2": {"a0": 1.0},
    }


@pytest.fixture
def branching_initial_dist() -> Dict[str, float]:
    return {"s0": 1.0}


# =====================================================================
# Single trajectory sampling
# =====================================================================

class TestSingleTrajectory:
    """Test sample_single method."""

    def test_reaches_goal(
        self, simple_transition, simple_cost, simple_policy, goal_states
    ) -> None:
        """Single trajectory should reach the goal state."""
        sampler = TrajectorySamplerImpl(seed=42)
        traj, reason = sampler.sample_single(
            simple_transition, simple_cost, simple_policy,
            initial_state="start", goal_states=goal_states, max_steps=10,
        )
        assert reason == TerminationReason.GOAL_REACHED

    def test_trajectory_records_all_steps(
        self, simple_transition, simple_cost, simple_policy, goal_states
    ) -> None:
        """Trajectory should have a step for each action taken."""
        sampler = TrajectorySamplerImpl(seed=42)
        traj, _ = sampler.sample_single(
            simple_transition, simple_cost, simple_policy,
            initial_state="start", goal_states=goal_states, max_steps=10,
        )
        # start → mid → goal  = 2 actions
        assert traj.length == 2

    def test_total_cost_correct(
        self, simple_transition, simple_cost, simple_policy, goal_states
    ) -> None:
        """Total cost should equal sum of step costs."""
        sampler = TrajectorySamplerImpl(seed=42)
        traj, _ = sampler.sample_single(
            simple_transition, simple_cost, simple_policy,
            initial_state="start", goal_states=goal_states, max_steps=10,
        )
        expected_cost = 0.5 + 1.0  # go + finish
        assert traj.total_cost.mu == pytest.approx(expected_cost, abs=1e-10)


# =====================================================================
# Batch sampling
# =====================================================================

class TestBatchSampling:
    """Test batch trajectory sampling."""

    def test_correct_count(
        self, simple_transition, simple_cost, simple_policy,
        initial_dist, goal_states
    ) -> None:
        """Batch should produce exactly num_samples trajectories."""
        sampler = TrajectorySamplerImpl(seed=42)
        config = MCConfig(num_samples=50, max_trajectory_length=20, seed=42)
        bundle = sampler.sample(
            simple_transition, simple_cost, simple_policy,
            initial_dist, goal_states, config,
        )
        assert bundle.num_trajectories == 50
        assert len(bundle.costs) == 50
        assert len(bundle.lengths) == 50

    def test_all_reach_goal(
        self, simple_transition, simple_cost, simple_policy,
        initial_dist, goal_states
    ) -> None:
        """In a deterministic MDP, all trajectories should reach the goal."""
        sampler = TrajectorySamplerImpl(seed=42)
        config = MCConfig(num_samples=10, max_trajectory_length=20, seed=42)
        bundle = sampler.sample(
            simple_transition, simple_cost, simple_policy,
            initial_dist, goal_states, config,
        )
        for reason in bundle.termination_reasons:
            assert reason == TerminationReason.GOAL_REACHED


# =====================================================================
# Reproducibility
# =====================================================================

class TestReproducibility:
    """Test that same seed produces identical results."""

    def test_same_seed_same_results(
        self, branching_transition, branching_cost, branching_policy,
        branching_initial_dist, goal_states
    ) -> None:
        """Two runs with the same seed should produce identical costs."""
        config = MCConfig(num_samples=20, max_trajectory_length=10, seed=123)

        sampler1 = TrajectorySamplerImpl(seed=123)
        bundle1 = sampler1.sample(
            branching_transition, branching_cost, branching_policy,
            branching_initial_dist, goal_states, config,
        )

        sampler2 = TrajectorySamplerImpl(seed=123)
        bundle2 = sampler2.sample(
            branching_transition, branching_cost, branching_policy,
            branching_initial_dist, goal_states, config,
        )

        assert bundle1.costs == bundle2.costs

    def test_different_seeds_different_results(
        self, branching_transition, branching_cost, branching_policy,
        branching_initial_dist, goal_states
    ) -> None:
        """Different seeds should (with high probability) produce different trajectories."""
        config1 = MCConfig(num_samples=50, max_trajectory_length=10, seed=111)
        config2 = MCConfig(num_samples=50, max_trajectory_length=10, seed=999)

        sampler = TrajectorySamplerImpl(seed=111)
        bundle1 = sampler.sample(
            branching_transition, branching_cost, branching_policy,
            branching_initial_dist, goal_states, config1,
        )

        sampler2 = TrajectorySamplerImpl(seed=999)
        bundle2 = sampler2.sample(
            branching_transition, branching_cost, branching_policy,
            branching_initial_dist, goal_states, config2,
        )

        # With stochastic transitions and different seeds, costs should differ
        assert bundle1.costs != bundle2.costs


# =====================================================================
# Early termination
# =====================================================================

class TestEarlyTermination:
    """Test termination conditions."""

    def test_early_termination_on_goal(
        self, simple_transition, simple_cost, simple_policy, goal_states
    ) -> None:
        """Trajectory should stop at goal even if max_steps is large."""
        sampler = TrajectorySamplerImpl(seed=42)
        traj, reason = sampler.sample_single(
            simple_transition, simple_cost, simple_policy,
            initial_state="start", goal_states=goal_states, max_steps=1000,
        )
        assert reason == TerminationReason.GOAL_REACHED
        assert traj.length <= 2  # start->mid->goal

    def test_max_steps_respected(self) -> None:
        """Trajectory should not exceed max_steps."""
        # Create a cycle: s0 -> s0 with cycle detection off
        transition = {"s0": {"a0": {"s0": 1.0}}}
        cost = {"s0": {"a0": 0.1}}
        policy = {"s0": {"a0": 1.0}}
        sampler = TrajectorySamplerImpl(seed=42)
        traj, reason = sampler.sample_single(
            transition, cost, policy,
            initial_state="s0", goal_states=frozenset({"goal"}),
            max_steps=5, detect_cycles=False,
        )
        assert traj.length <= 5

    def test_dead_end_termination(self) -> None:
        """Trajectory should terminate at a dead end (no actions)."""
        transition = {"s0": {"a0": {"s1": 1.0}}}
        cost = {"s0": {"a0": 0.5}}
        policy = {"s0": {"a0": 1.0}}
        # s1 has no actions → dead end
        sampler = TrajectorySamplerImpl(seed=42)
        traj, reason = sampler.sample_single(
            transition, cost, policy,
            initial_state="s0", goal_states=frozenset({"goal"}),
            max_steps=10,
        )
        assert reason == TerminationReason.DEAD_END

    def test_cycle_detection(self) -> None:
        """Cycle detection should terminate when revisiting a state."""
        transition = {
            "s0": {"a0": {"s1": 1.0}},
            "s1": {"a0": {"s0": 1.0}},
        }
        cost = {"s0": {"a0": 0.1}, "s1": {"a0": 0.1}}
        policy = {"s0": {"a0": 1.0}, "s1": {"a0": 1.0}}
        sampler = TrajectorySamplerImpl(seed=42)
        traj, reason = sampler.sample_single(
            transition, cost, policy,
            initial_state="s0", goal_states=frozenset({"goal"}),
            max_steps=100, detect_cycles=True,
        )
        assert reason == TerminationReason.CYCLE_DETECTED


# =====================================================================
# Importance weighting
# =====================================================================

class TestImportanceWeighting:
    """Test importance-weighted sampling."""

    def test_importance_sampling_produces_weights(
        self, branching_transition, branching_cost, branching_policy,
        branching_initial_dist, goal_states
    ) -> None:
        """Importance sampling should produce importance weights."""
        config = MCConfig(
            num_samples=20, max_trajectory_length=10, seed=42,
            strategy=SamplingStrategy.IMPORTANCE,
        )
        sampler = TrajectorySamplerImpl(seed=42)
        bundle = sampler.sample(
            branching_transition, branching_cost, branching_policy,
            branching_initial_dist, goal_states, config,
        )
        assert bundle.importance_weights is not None
        assert len(bundle.importance_weights) == 20

    def test_normalised_weights_sum_to_one(
        self, branching_transition, branching_cost, branching_policy,
        branching_initial_dist, goal_states
    ) -> None:
        """Self-normalised importance weights should sum to ~1."""
        config = MCConfig(
            num_samples=50, max_trajectory_length=10, seed=42,
            strategy=SamplingStrategy.IMPORTANCE,
        )
        sampler = TrajectorySamplerImpl(seed=42)
        bundle = sampler.sample(
            branching_transition, branching_cost, branching_policy,
            branching_initial_dist, goal_states, config,
        )
        assert bundle.importance_weights is not None
        total = sum(iw.normalised_weight for iw in bundle.importance_weights)
        assert total == pytest.approx(1.0, abs=1e-6)
