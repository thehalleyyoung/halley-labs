"""
usability_oracle.mdp.reward — Reward / cost functions and reward shaping.

Provides:
- :class:`RewardFunction` — component reward / cost signals for UI navigation
- :class:`TaskRewardShaper` — potential-based reward shaping (Ng et al., 1999)

The MDP uses a *cost* formulation (lower is better); rewards are negated costs.
The combined reward for a transition (s, a, s') is:

    R(s, a, s') = w_g · R_goal(s') + w_s · R_step(a) + w_i · R_info(s')

where *w_g*, *w_s*, *w_i* are configurable weights.

References
----------
- Ng, A. Y., Harada, D. & Russell, S. (1999). Policy invariance under
  reward transformations. *ICML*.
- Levine, S. (2018). Reinforcement learning and control as probabilistic
  inference. *arXiv:1805.00909*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from usability_oracle.mdp.models import Action, MDP, State


# ---------------------------------------------------------------------------
# Reward Function
# ---------------------------------------------------------------------------


class RewardFunction:
    """Component-wise reward / cost computation for UI navigation.

    All methods return *negative cost* (reward) so that higher values are
    better, consistent with the free-energy formulation
    F(π) = E_π[R] − (1/β) D_KL(π ‖ p₀).
    """

    def __init__(
        self,
        goal_reward_value: float = 10.0,
        step_cost_value: float = -1.0,
        cognitive_cost_weight: float = 0.5,
        info_reward_weight: float = 0.2,
    ) -> None:
        self.goal_reward_value = goal_reward_value
        self.step_cost_value = step_cost_value
        self.cognitive_cost_weight = cognitive_cost_weight
        self.info_reward_weight = info_reward_weight

    def goal_reward(self, state: State, goal_states: set[str]) -> float:
        """Return a large positive reward if *state* is a goal state.

        R_goal(s) = +G  if s ∈ S_goal,  0 otherwise.

        Parameters
        ----------
        state : State
        goal_states : set[str]

        Returns
        -------
        float
        """
        if state.state_id in goal_states or state.is_goal:
            return self.goal_reward_value
        return 0.0

    def step_cost(self, action: Action, cognitive_cost: float = 0.0) -> float:
        """Return the (negative) cost of performing an action.

        R_step(a) = −(c_base + w_cog · c_cog(a))

        where c_cog(a) is the cognitive cost of the action (e.g., from
        Fitts' law, Hick-Hyman) and c_base is a constant per-step penalty.

        Parameters
        ----------
        action : Action
        cognitive_cost : float
            External cognitive cost estimate for this action.

        Returns
        -------
        float
            A non-positive value.
        """
        base = self.step_cost_value
        cog = self.cognitive_cost_weight * cognitive_cost
        return base - cog

    def information_reward(
        self, state: State, task_progress: float
    ) -> float:
        """Reward for making progress toward the task goal.

        R_info(s) = w_info · progress(s)

        This encourages the agent to explore states that bring it
        closer to the goal even before the goal is reached.

        Parameters
        ----------
        state : State
        task_progress : float
            Fraction of sub-goals completed in this state, in [0, 1].

        Returns
        -------
        float
        """
        return self.info_reward_weight * task_progress

    def combined_reward(
        self,
        state: State,
        action: Action,
        next_state: State,
        goal_states: set[str],
        weights: Optional[dict[str, float]] = None,
    ) -> float:
        """Combine all reward components.

        R(s, a, s') = w_g · R_goal(s') + w_s · R_step(a) + w_i · R_info(s')

        Parameters
        ----------
        state : State
            Source state.
        action : Action
        next_state : State
            Target state.
        goal_states : set[str]
        weights : dict, optional
            Override default weights: ``{"goal": w_g, "step": w_s, "info": w_i}``.

        Returns
        -------
        float
        """
        w = weights or {"goal": 1.0, "step": 1.0, "info": 1.0}

        r_goal = self.goal_reward(next_state, goal_states)
        r_step = self.step_cost(action)
        progress = next_state.features.get("task_progress", 0.0)
        r_info = self.information_reward(next_state, progress)

        return (
            w.get("goal", 1.0) * r_goal
            + w.get("step", 1.0) * r_step
            + w.get("info", 1.0) * r_info
        )


# ---------------------------------------------------------------------------
# Task Reward Shaper
# ---------------------------------------------------------------------------


class TaskRewardShaper:
    """Potential-based reward shaping (Ng et al., 1999).

    Adds a shaping reward F(s, a, s') = γ Φ(s') − Φ(s) to the base reward
    so that the optimal policy is preserved but learning / convergence is
    accelerated.

    The potential Φ(s) is derived from the task specification's sub-goal
    structure, with higher potential for states closer to completion.

    References
    ----------
    - Ng, A. Y., Harada, D. & Russell, S. (1999). Policy invariance under
      reward transformations: Theory and application to reward shaping. *ICML*.
    """

    def __init__(
        self,
        progress_weight: float = 5.0,
        distance_weight: float = 1.0,
        discount: float = 0.99,
    ) -> None:
        self.progress_weight = progress_weight
        self.distance_weight = distance_weight
        self.discount = discount

    def shape(self, mdp: MDP, task_spec: Any) -> dict[str, float]:
        """Compute potential Φ(s) for every state in the MDP.

        Φ(s) = w_prog · progress(s) + w_dist · (1 − normalised_distance(s))

        Parameters
        ----------
        mdp : MDP
        task_spec : TaskSpec

        Returns
        -------
        dict[str, float]
            Mapping state_id → Φ(s).
        """
        potentials: dict[str, float] = {}

        # Compute max distance for normalisation (BFS from initial state)
        distances = self._bfs_distances(mdp)
        max_dist = max(distances.values()) if distances else 1.0
        max_dist = max(max_dist, 1.0)

        for sid, state in mdp.states.items():
            prog = self._progress_potential(state, task_spec)
            dist = distances.get(sid, max_dist)
            normalised_dist = 1.0 - (dist / max_dist)

            phi = (
                self.progress_weight * prog
                + self.distance_weight * normalised_dist
            )
            potentials[sid] = phi

        return potentials

    def _progress_potential(self, state: State, task_spec: Any) -> float:
        """Compute progress-based potential for a state.

        Φ_prog(s) = |completed sub-goals| / |total sub-goals|

        Scales from 0 (no progress) to 1 (all sub-goals done).
        """
        n_subgoals = len(getattr(task_spec, "sub_goals", []))
        if n_subgoals == 0:
            return 1.0 if state.is_goal else 0.0
        progress = state.features.get("task_progress", 0.0)
        return progress

    def shaping_reward(
        self,
        potentials: dict[str, float],
        source_id: str,
        target_id: str,
    ) -> float:
        """Compute the shaping reward F(s, a, s') = γ Φ(s') − Φ(s).

        This additive term preserves the set of optimal policies (Ng, 1999).
        """
        phi_s = potentials.get(source_id, 0.0)
        phi_sp = potentials.get(target_id, 0.0)
        return self.discount * phi_sp - phi_s

    def _bfs_distances(self, mdp: MDP) -> dict[str, int]:
        """BFS from goal states backwards to compute hop-distance to goal."""
        from collections import deque

        distances: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque()

        for gid in mdp.goal_states:
            distances[gid] = 0
            queue.append((gid, 0))

        while queue:
            sid, d = queue.popleft()
            for pred in mdp.get_predecessors(sid):
                if pred not in distances:
                    distances[pred] = d + 1
                    queue.append((pred, d + 1))

        return distances
