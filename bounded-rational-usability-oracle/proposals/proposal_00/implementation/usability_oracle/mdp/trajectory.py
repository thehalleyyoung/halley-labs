"""
usability_oracle.mdp.trajectory — Trajectory sampling and statistics.

Provides :class:`TrajectorySampler` to generate sample paths through an MDP
under a given (stochastic or deterministic) policy, and
:class:`TrajectoryStats` to compute summary statistics over trajectory sets.

A trajectory is a sequence of (state, action, next_state, cost) tuples
terminated when either a goal / terminal state is reached or the step
budget is exhausted.

References
----------
- Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning*, 2nd ed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from usability_oracle.mdp.models import MDP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trajectory data structure
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryStep:
    """A single step in a sampled trajectory."""

    state_id: str
    action_id: str
    next_state_id: str
    cost: float
    probability: float
    step_index: int


@dataclass
class Trajectory:
    """A complete sampled trajectory from initial to terminal/goal state."""

    steps: list[TrajectoryStep] = field(default_factory=list)
    total_cost: float = 0.0
    reached_goal: bool = False
    terminated: bool = False

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def states_visited(self) -> list[str]:
        """Ordered list of state IDs visited (including final state)."""
        if not self.steps:
            return []
        result = [step.state_id for step in self.steps]
        result.append(self.steps[-1].next_state_id)
        return result

    @property
    def actions_taken(self) -> list[str]:
        return [step.action_id for step in self.steps]

    @property
    def state_visit_counts(self) -> dict[str, int]:
        """Number of times each state was visited."""
        counts: dict[str, int] = {}
        for sid in self.states_visited:
            counts[sid] = counts.get(sid, 0) + 1
        return counts

    @property
    def discounted_cost(self) -> float:
        """Discounted total cost (γ = 0.99 default)."""
        gamma = 0.99
        total = 0.0
        for step in self.steps:
            total += (gamma ** step.step_index) * step.cost
        return total


# ---------------------------------------------------------------------------
# Trajectory statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrajectoryStats:
    """Summary statistics over a set of trajectories.

    Attributes
    ----------
    mean_cost : float
        Mean total cost across trajectories.
    std_cost : float
        Standard deviation of total cost.
    median_cost : float
    percentile_95 : float
        95th percentile of total cost.
    completion_rate : float
        Fraction of trajectories that reached a goal state.
    mean_steps : float
        Mean trajectory length (number of steps).
    bottleneck_states : list[str]
        States with highest average visit count (most revisited).
    """

    mean_cost: float
    std_cost: float
    median_cost: float
    percentile_95: float
    completion_rate: float
    mean_steps: float
    bottleneck_states: list[str]


# ---------------------------------------------------------------------------
# TrajectorySampler
# ---------------------------------------------------------------------------


class TrajectorySampler:
    """Sample trajectories from an MDP under a given policy.

    The policy can be either:
    - A deterministic mapping ``dict[str, str]``  (state → action)
    - A stochastic mapping ``dict[str, dict[str, float]]``
      (state → {action: probability})

    Parameters
    ----------
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self.rng = rng or np.random.default_rng()

    def sample(
        self,
        mdp: MDP,
        policy: dict[str, Any],
        n_trajectories: int = 100,
        max_steps: int = 500,
    ) -> list[Trajectory]:
        """Sample *n_trajectories* from *mdp* under *policy*.

        Parameters
        ----------
        mdp : MDP
        policy : dict[str, str] or dict[str, dict[str, float]]
            Deterministic or stochastic policy.
        n_trajectories : int
        max_steps : int
            Maximum number of steps per trajectory.

        Returns
        -------
        list[Trajectory]
        """
        trajectories: list[Trajectory] = []
        for _ in range(n_trajectories):
            traj = self._sample_single(mdp, policy, max_steps)
            trajectories.append(traj)
        return trajectories

    def _sample_single(
        self,
        mdp: MDP,
        policy: dict[str, Any],
        max_steps: int,
    ) -> Trajectory:
        """Sample a single trajectory."""
        traj = Trajectory()
        current = mdp.initial_state

        for step_idx in range(max_steps):
            state = mdp.states.get(current)
            if state is None:
                traj.terminated = True
                break

            if state.is_terminal or state.is_goal:
                if state.is_goal:
                    traj.reached_goal = True
                traj.terminated = True
                break

            # Select action
            action_id = self._sample_action(policy, current, mdp)
            if action_id is None:
                traj.terminated = True
                break

            # Sample transition
            next_state, prob, cost = self._sample_transition(mdp, current, action_id)
            if next_state is None:
                traj.terminated = True
                break

            step = TrajectoryStep(
                state_id=current,
                action_id=action_id,
                next_state_id=next_state,
                cost=cost,
                probability=prob,
                step_index=step_idx,
            )
            traj.steps.append(step)
            traj.total_cost += cost
            current = next_state

        return traj

    def _sample_action(
        self,
        policy: dict[str, Any],
        state_id: str,
        mdp: MDP,
    ) -> Optional[str]:
        """Sample an action from the policy for *state_id*.

        Handles both deterministic (str) and stochastic (dict) policies.
        """
        pol_entry = policy.get(state_id)
        if pol_entry is None:
            # Fallback: pick uniformly from available actions
            available = mdp.get_actions(state_id)
            if not available:
                return None
            return str(self.rng.choice(available))

        if isinstance(pol_entry, str):
            return pol_entry

        # Stochastic policy: dict[str, float]
        if isinstance(pol_entry, dict):
            actions = list(pol_entry.keys())
            probs = np.array([pol_entry[a] for a in actions], dtype=np.float64)

            # Renormalise for numerical safety
            total = probs.sum()
            if total <= 0:
                return str(self.rng.choice(actions)) if actions else None
            probs /= total

            idx = self.rng.choice(len(actions), p=probs)
            return actions[idx]

        return None

    def _sample_transition(
        self,
        mdp: MDP,
        state_id: str,
        action_id: str,
    ) -> tuple[Optional[str], float, float]:
        """Sample a successor state from the transition distribution.

        Returns
        -------
        (next_state_id, probability, cost) or (None, 0, 0) if no transitions.
        """
        outcomes = mdp.get_transitions(state_id, action_id)
        if not outcomes:
            return None, 0.0, 0.0

        targets = [t for t, _p, _c in outcomes]
        probs = np.array([p for _t, p, _c in outcomes], dtype=np.float64)
        costs = [c for _t, _p, c in outcomes]

        # Renormalise
        total = probs.sum()
        if total <= 0:
            idx = int(self.rng.integers(len(targets)))
        else:
            probs /= total
            idx = int(self.rng.choice(len(targets), p=probs))

        return targets[idx], float(probs[idx]) if total > 0 else 1.0, costs[idx]

    # ── Statistics --------------------------------------------------------

    @staticmethod
    def trajectory_statistics(trajectories: list[Trajectory]) -> TrajectoryStats:
        """Compute summary statistics over a collection of trajectories.

        Parameters
        ----------
        trajectories : list[Trajectory]

        Returns
        -------
        TrajectoryStats
        """
        if not trajectories:
            return TrajectoryStats(
                mean_cost=0.0,
                std_cost=0.0,
                median_cost=0.0,
                percentile_95=0.0,
                completion_rate=0.0,
                mean_steps=0.0,
                bottleneck_states=[],
            )

        costs = np.array([t.total_cost for t in trajectories])
        lengths = np.array([t.length for t in trajectories], dtype=np.float64)
        n_completed = sum(1 for t in trajectories if t.reached_goal)

        # Bottleneck states: find states with highest average visit count
        visit_totals: dict[str, int] = {}
        for traj in trajectories:
            for sid, count in traj.state_visit_counts.items():
                visit_totals[sid] = visit_totals.get(sid, 0) + count

        # Average visit count per trajectory
        n_traj = len(trajectories)
        avg_visits = {
            sid: total / n_traj for sid, total in visit_totals.items()
        }
        # Top-5 most visited states (excluding start/goal if trivially visited)
        sorted_states = sorted(avg_visits.items(), key=lambda x: -x[1])
        bottleneck_states = [sid for sid, _ in sorted_states[:5]]

        return TrajectoryStats(
            mean_cost=float(np.mean(costs)),
            std_cost=float(np.std(costs)),
            median_cost=float(np.median(costs)),
            percentile_95=float(np.percentile(costs, 95)),
            completion_rate=n_completed / n_traj,
            mean_steps=float(np.mean(lengths)),
            bottleneck_states=bottleneck_states,
        )
