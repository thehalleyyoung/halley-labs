"""
usability_oracle.visualization.mdp_viz — MDP state-space visualization.

Renders MDP state spaces as text-based graph diagrams, transition tables,
policy maps, value function displays, and trajectory visualizations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class MDPVizConfig:
    """Configuration for MDP visualization."""
    max_states: int = 50
    max_actions: int = 10
    show_probabilities: bool = True
    show_rewards: bool = True
    prob_threshold: float = 0.01
    value_precision: int = 3
    compact: bool = False


class MDPVisualizer:
    """Render MDP structures in text format.

    Parameters:
        config: Visualization configuration.
    """

    def __init__(self, config: MDPVizConfig | None = None) -> None:
        self._config = config or MDPVizConfig()

    # ------------------------------------------------------------------
    # Transition table
    # ------------------------------------------------------------------

    def render_transition_table(
        self,
        transition_matrix: np.ndarray,
        state_names: list[str] | None = None,
    ) -> str:
        """Render the transition matrix as a formatted table."""
        P = np.asarray(transition_matrix, dtype=float)
        n = P.shape[0]
        names = state_names or [f"s{i}" for i in range(n)]

        if n > self._config.max_states:
            names = names[:self._config.max_states]
            P = P[:self._config.max_states, :self._config.max_states]
            n = self._config.max_states

        # Determine column width
        col_w = max(6, max(len(s) for s in names) + 1)
        header = " " * col_w + " ".join(f"{s:>{col_w}}" for s in names)
        lines = ["Transition Matrix:", header]

        for i in range(n):
            row_vals = []
            for j in range(n):
                v = P[i, j]
                if v < self._config.prob_threshold:
                    row_vals.append(f"{'·':>{col_w}}")
                else:
                    row_vals.append(f"{v:>{col_w}.{self._config.value_precision}f}")
            lines.append(f"{names[i]:>{col_w}} " + " ".join(row_vals))

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Value function
    # ------------------------------------------------------------------

    def render_value_function(
        self,
        values: np.ndarray,
        state_names: list[str] | None = None,
        sort: bool = True,
    ) -> str:
        """Render the value function as a bar chart."""
        V = np.asarray(values, dtype=float).ravel()
        n = len(V)
        names = state_names or [f"s{i}" for i in range(n)]

        if n > self._config.max_states:
            # Show top and bottom states
            top_k = self._config.max_states // 2
            top_idx = np.argsort(-V)[:top_k]
            bot_idx = np.argsort(V)[:top_k]
            indices = sorted(set(top_idx.tolist() + bot_idx.tolist()))
        elif sort:
            indices = np.argsort(-V).tolist()
        else:
            indices = list(range(n))

        max_name = max(len(names[i]) for i in indices) if indices else 0
        v_min, v_max = float(V.min()), float(V.max())
        bar_width = 40

        lines = ["Value Function:"]
        for i in indices:
            name = names[i]
            val = V[i]
            # Normalise for bar length
            if v_max > v_min:
                norm = (val - v_min) / (v_max - v_min)
            else:
                norm = 0.5
            bar_len = max(0, int(norm * bar_width))
            bar = "█" * bar_len + "░" * (bar_width - bar_len)
            lines.append(f"  {name:>{max_name}} │{bar}│ {val:.{self._config.value_precision}f}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Policy map
    # ------------------------------------------------------------------

    def render_policy(
        self,
        policy: np.ndarray | dict[Any, Any],
        state_names: list[str] | None = None,
        action_names: list[str] | None = None,
    ) -> str:
        """Render a policy as state -> action mapping."""
        lines = ["Policy:"]

        if isinstance(policy, dict):
            for state, action in policy.items():
                lines.append(f"  {state} → {action}")
            return "\n".join(lines)

        P = np.asarray(policy, dtype=float)
        if P.ndim == 1:
            # Deterministic policy
            names = state_names or [f"s{i}" for i in range(len(P))]
            a_names = action_names or [f"a{int(a)}" for a in P]
            for i, name in enumerate(names[:self._config.max_states]):
                action_idx = int(P[i])
                a_name = action_names[action_idx] if action_names and action_idx < len(action_names) else f"a{action_idx}"
                lines.append(f"  {name} → {a_name}")
        elif P.ndim == 2:
            # Stochastic policy
            n_states, n_actions = P.shape
            s_names = state_names or [f"s{i}" for i in range(n_states)]
            a_names = action_names or [f"a{j}" for j in range(n_actions)]
            for i in range(min(n_states, self._config.max_states)):
                probs = []
                for j in range(n_actions):
                    if P[i, j] > self._config.prob_threshold:
                        probs.append(f"{a_names[j]}:{P[i, j]:.2f}")
                lines.append(f"  {s_names[i]} → {', '.join(probs)}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Trajectory visualization
    # ------------------------------------------------------------------

    def render_trajectory(
        self,
        states: list[Any],
        actions: list[Any] | None = None,
        rewards: list[float] | None = None,
        costs: list[float] | None = None,
    ) -> str:
        """Render a state-action trajectory."""
        lines = ["Trajectory:"]
        for i, state in enumerate(states):
            parts = [f"  t={i}: {state}"]
            if actions and i < len(actions):
                parts.append(f"──[{actions[i]}]──→")
            if rewards and i < len(rewards):
                parts.append(f"  r={rewards[i]:.3f}")
            if costs and i < len(costs):
                parts.append(f"  c={costs[i]:.3f}")
            lines.append(" ".join(parts))

        if rewards:
            total_r = sum(rewards)
            lines.append(f"\n  Total reward: {total_r:.3f}")
        if costs:
            total_c = sum(costs)
            lines.append(f"  Total cost:   {total_c:.3f}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # State graph (adjacency list)
    # ------------------------------------------------------------------

    def render_graph(
        self,
        transition_matrix: np.ndarray,
        state_names: list[str] | None = None,
    ) -> str:
        """Render the MDP as a text adjacency list."""
        P = np.asarray(transition_matrix, dtype=float)
        n = P.shape[0]
        names = state_names or [f"s{i}" for i in range(n)]

        lines = ["State Graph:"]
        for i in range(min(n, self._config.max_states)):
            successors = []
            for j in range(n):
                if P[i, j] > self._config.prob_threshold:
                    if self._config.show_probabilities:
                        successors.append(f"{names[j]}({P[i, j]:.2f})")
                    else:
                        successors.append(names[j])
            lines.append(f"  {names[i]} → {', '.join(successors) if successors else '(terminal)'}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def render_stats(
        self,
        transition_matrix: np.ndarray,
        reward: np.ndarray | None = None,
    ) -> str:
        """Render MDP statistics."""
        P = np.asarray(transition_matrix, dtype=float)
        n = P.shape[0]
        n_actions = P.shape[2] if P.ndim == 3 else 1

        # Sparsity
        n_nonzero = np.count_nonzero(P > self._config.prob_threshold)
        total = P.size
        sparsity = 1.0 - n_nonzero / total if total > 0 else 0.0

        # Connectivity
        if P.ndim == 2:
            reachable = np.zeros(n, dtype=bool)
            reachable[0] = True
            for _ in range(n):
                new_reach = P[reachable, :].sum(axis=0) > self._config.prob_threshold
                reachable = reachable | new_reach
            n_reachable = int(reachable.sum())
        else:
            n_reachable = n

        lines = [
            "MDP Statistics:",
            f"  States:      {n}",
            f"  Actions:     {n_actions}",
            f"  Sparsity:    {sparsity:.2%}",
            f"  Reachable:   {n_reachable}/{n}",
        ]

        if reward is not None:
            r = np.asarray(reward, dtype=float)
            lines.extend([
                f"  Reward range: [{r.min():.3f}, {r.max():.3f}]",
                f"  Reward mean:  {r.mean():.3f}",
            ])

        # Eigenvalue analysis
        if P.ndim == 2 and n <= 500:
            eigenvalues = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
            lines.append(f"  Spectral gap: {1.0 - eigenvalues[1] if len(eigenvalues) > 1 else 1.0:.6f}")

        return "\n".join(lines)
