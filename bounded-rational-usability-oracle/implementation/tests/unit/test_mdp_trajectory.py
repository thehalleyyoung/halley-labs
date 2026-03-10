"""Unit tests for usability_oracle.mdp.trajectory — TrajectorySampler, Trajectory, TrajectoryStats.

Validates trajectory sampling from MDPs under deterministic and stochastic
policies, trajectory data-structure properties, and the trajectory_statistics
summary computation.
"""

from __future__ import annotations

import numpy as np
import pytest

from usability_oracle.mdp.trajectory import (
    TrajectorySampler,
    TrajectoryStep,
    Trajectory,
    TrajectoryStats,
)
from usability_oracle.mdp.models import MDP, State, Action, Transition
from tests.fixtures.sample_mdps import (
    make_two_state_mdp,
    make_cyclic_mdp,
    make_choice_mdp,
    make_large_chain_mdp,
)


# ═══════════════════════════════════════════════════════════════════════════
# TrajectoryStep tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTrajectoryStep:
    """Tests for TrajectoryStep — a single (s, a, s', cost, p, idx) record."""

    def test_fields(self):
        """TrajectoryStep stores all required fields."""
        step = TrajectoryStep(
            state_id="s0", action_id="go", next_state_id="s1",
            cost=1.5, probability=0.8, step_index=0,
        )
        assert step.state_id == "s0"
        assert step.action_id == "go"
        assert step.next_state_id == "s1"
        assert step.cost == 1.5
        assert step.probability == 0.8
        assert step.step_index == 0

    def test_step_index_sequential(self):
        """step_index should record position within a trajectory."""
        steps = [
            TrajectoryStep("s0", "a", "s1", 1.0, 1.0, i)
            for i in range(5)
        ]
        assert [s.step_index for s in steps] == [0, 1, 2, 3, 4]


# ═══════════════════════════════════════════════════════════════════════════
# Trajectory tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTrajectory:
    """Tests for Trajectory — a complete path from initial to terminal state."""

    def _two_step_trajectory(self) -> Trajectory:
        """Helper: s0 → s1 → s2 with costs 1.0 and 2.0."""
        steps = [
            TrajectoryStep("s0", "a", "s1", 1.0, 1.0, 0),
            TrajectoryStep("s1", "b", "s2", 2.0, 1.0, 1),
        ]
        return Trajectory(steps=steps, total_cost=3.0,
                          reached_goal=True, terminated=True)

    def test_length(self):
        """length should equal the number of steps."""
        traj = self._two_step_trajectory()
        assert traj.length == 2

    def test_empty_trajectory_length(self):
        """An empty trajectory has length 0."""
        traj = Trajectory()
        assert traj.length == 0

    def test_states_visited(self):
        """states_visited includes every state plus the final next_state."""
        traj = self._two_step_trajectory()
        assert traj.states_visited == ["s0", "s1", "s2"]

    def test_states_visited_empty(self):
        """Empty trajectory has no states_visited."""
        traj = Trajectory()
        assert traj.states_visited == []

    def test_actions_taken(self):
        """actions_taken returns the sequence of action IDs."""
        traj = self._two_step_trajectory()
        assert traj.actions_taken == ["a", "b"]

    def test_state_visit_counts(self):
        """state_visit_counts maps each state to its visit frequency."""
        traj = self._two_step_trajectory()
        counts = traj.state_visit_counts
        assert counts["s0"] == 1
        assert counts["s1"] == 1
        assert counts["s2"] == 1

    def test_state_visit_counts_with_cycle(self):
        """A trajectory with a revisited state should count > 1."""
        steps = [
            TrajectoryStep("s0", "a", "s1", 1.0, 1.0, 0),
            TrajectoryStep("s1", "b", "s0", 1.0, 1.0, 1),
            TrajectoryStep("s0", "a", "s1", 1.0, 1.0, 2),
        ]
        traj = Trajectory(steps=steps, total_cost=3.0)
        counts = traj.state_visit_counts
        assert counts["s0"] == 2
        assert counts["s1"] == 2

    def test_total_cost(self):
        """total_cost should be the sum of step costs."""
        traj = self._two_step_trajectory()
        assert traj.total_cost == 3.0

    def test_discounted_cost(self):
        """discounted_cost applies γ^t weighting (γ=0.99 default)."""
        traj = self._two_step_trajectory()
        expected = 1.0 * (0.99 ** 0) + 2.0 * (0.99 ** 1)
        assert traj.discounted_cost == pytest.approx(expected, rel=1e-6)

    def test_reached_goal_flag(self):
        """reached_goal should be True when the trajectory ends at a goal."""
        traj = self._two_step_trajectory()
        assert traj.reached_goal is True

    def test_terminated_flag(self):
        """terminated should be True when the trajectory stopped."""
        traj = self._two_step_trajectory()
        assert traj.terminated is True


# ═══════════════════════════════════════════════════════════════════════════
# TrajectorySampler tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTrajectorySampler:
    """Tests for TrajectorySampler — sampling paths through an MDP under a policy."""

    def test_sample_returns_list(self):
        """sample() returns a list of Trajectory objects."""
        sampler = TrajectorySampler(rng=np.random.default_rng(42))
        mdp = make_two_state_mdp()
        policy = {"start": "go"}
        trajs = sampler.sample(mdp, policy, n_trajectories=5)
        assert isinstance(trajs, list)
        assert len(trajs) == 5
        assert all(isinstance(t, Trajectory) for t in trajs)

    def test_deterministic_policy_identical_trajectories(self):
        """Under a deterministic policy on a deterministic MDP, all trajectories
        should be identical."""
        sampler = TrajectorySampler(rng=np.random.default_rng(0))
        mdp = make_two_state_mdp()
        policy = {"start": "go"}
        trajs = sampler.sample(mdp, policy, n_trajectories=10)
        costs = [t.total_cost for t in trajs]
        assert all(c == costs[0] for c in costs)

    def test_all_reach_goal(self):
        """In a simple reachable MDP, all trajectories should reach the goal."""
        sampler = TrajectorySampler(rng=np.random.default_rng(42))
        mdp = make_two_state_mdp()
        policy = {"start": "go"}
        trajs = sampler.sample(mdp, policy, n_trajectories=20)
        assert all(t.reached_goal for t in trajs)

    def test_stochastic_policy_sample(self):
        """TrajectorySampler handles stochastic (dict-valued) policies."""
        sampler = TrajectorySampler(rng=np.random.default_rng(42))
        mdp = make_choice_mdp(n_choices=3)
        policy = {"start": {"choice_0": 0.5, "choice_1": 0.3, "choice_2": 0.2}}
        trajs = sampler.sample(mdp, policy, n_trajectories=50)
        assert len(trajs) == 50
        assert all(t.reached_goal for t in trajs)

    def test_max_steps_limit(self):
        """Trajectories should not exceed max_steps."""
        sampler = TrajectorySampler(rng=np.random.default_rng(42))
        mdp = make_cyclic_mdp()
        # Policy that always cycles (never finishes)
        policy = {"s0": "next", "s1": "next", "s2": "next"}
        trajs = sampler.sample(mdp, policy, n_trajectories=5, max_steps=10)
        for t in trajs:
            assert t.length <= 10

    def test_terminal_state_stops(self):
        """Sampler should stop when reaching a terminal / goal state."""
        sampler = TrajectorySampler(rng=np.random.default_rng(42))
        mdp = make_two_state_mdp()
        policy = {"start": "go"}
        trajs = sampler.sample(mdp, policy, n_trajectories=1, max_steps=100)
        assert trajs[0].length == 1  # single step: start → goal

    def test_custom_rng_reproducibility(self):
        """Same seed produces identical trajectory sets."""
        mdp = make_cyclic_mdp()
        policy = {"s0": "next", "s1": "next", "s2": {"next": 0.5, "finish": 0.5}}
        s1 = TrajectorySampler(rng=np.random.default_rng(99))
        s2 = TrajectorySampler(rng=np.random.default_rng(99))
        t1 = s1.sample(mdp, policy, n_trajectories=20)
        t2 = s2.sample(mdp, policy, n_trajectories=20)
        for a, b in zip(t1, t2):
            assert a.total_cost == b.total_cost

    def test_trajectory_cost_nonnegative(self):
        """All trajectory total costs should be >= 0."""
        sampler = TrajectorySampler(rng=np.random.default_rng(42))
        mdp = make_cyclic_mdp()
        policy = {"s0": "next", "s1": "next", "s2": "finish"}
        trajs = sampler.sample(mdp, policy, n_trajectories=30)
        for t in trajs:
            assert t.total_cost >= 0


# ═══════════════════════════════════════════════════════════════════════════
# TrajectoryStats tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTrajectoryStats:
    """Tests for trajectory_statistics() — summary statistics over trajectory sets."""

    def _sample_trajs(self, mdp, policy, n=50):
        """Helper: sample trajectories with a fixed seed."""
        sampler = TrajectorySampler(rng=np.random.default_rng(42))
        return sampler.sample(mdp, policy, n_trajectories=n)

    def test_returns_trajectory_stats(self):
        """trajectory_statistics() returns a TrajectoryStats instance."""
        trajs = self._sample_trajs(make_two_state_mdp(), {"start": "go"})
        stats = TrajectorySampler.trajectory_statistics(trajs)
        assert isinstance(stats, TrajectoryStats)

    def test_mean_cost(self):
        """mean_cost should equal the average total_cost across trajectories."""
        trajs = self._sample_trajs(make_two_state_mdp(), {"start": "go"})
        stats = TrajectorySampler.trajectory_statistics(trajs)
        expected = np.mean([t.total_cost for t in trajs])
        assert stats.mean_cost == pytest.approx(float(expected), rel=1e-6)

    def test_std_cost(self):
        """std_cost should be 0 for identical deterministic trajectories."""
        trajs = self._sample_trajs(make_two_state_mdp(), {"start": "go"})
        stats = TrajectorySampler.trajectory_statistics(trajs)
        assert stats.std_cost == pytest.approx(0.0, abs=1e-10)

    def test_completion_rate_all_complete(self):
        """completion_rate should be 1.0 when all trajectories reach the goal."""
        trajs = self._sample_trajs(make_two_state_mdp(), {"start": "go"})
        stats = TrajectorySampler.trajectory_statistics(trajs)
        assert stats.completion_rate == 1.0

    def test_mean_steps(self):
        """mean_steps should be 1 for the two-state MDP (single step each)."""
        trajs = self._sample_trajs(make_two_state_mdp(), {"start": "go"})
        stats = TrajectorySampler.trajectory_statistics(trajs)
        assert stats.mean_steps == pytest.approx(1.0, abs=1e-6)

    def test_bottleneck_states(self):
        """bottleneck_states should list the most-visited states."""
        trajs = self._sample_trajs(make_two_state_mdp(), {"start": "go"})
        stats = TrajectorySampler.trajectory_statistics(trajs)
        assert isinstance(stats.bottleneck_states, list)
        assert len(stats.bottleneck_states) > 0

    def test_percentile_95(self):
        """percentile_95 should be >= median_cost."""
        mdp = make_cyclic_mdp()
        policy = {"s0": "next", "s1": "next", "s2": "finish"}
        trajs = self._sample_trajs(mdp, policy, n=100)
        stats = TrajectorySampler.trajectory_statistics(trajs)
        assert stats.percentile_95 >= stats.median_cost

    def test_empty_trajectories(self):
        """trajectory_statistics on an empty list returns zero-valued stats."""
        stats = TrajectorySampler.trajectory_statistics([])
        assert stats.mean_cost == 0.0
        assert stats.completion_rate == 0.0
        assert stats.mean_steps == 0.0
        assert stats.bottleneck_states == []

    def test_median_cost_nonnegative(self):
        """median_cost should be non-negative."""
        trajs = self._sample_trajs(make_two_state_mdp(), {"start": "go"})
        stats = TrajectorySampler.trajectory_statistics(trajs)
        assert stats.median_cost >= 0.0
