"""
usability_oracle.policy.monte_carlo — Monte Carlo estimation.

Provides :class:`MonteCarloEstimator` for estimating value functions,
Q-values, and free energy from sampled trajectories rather than exact
dynamic programming.  Useful when the MDP is too large for exact solution
or when evaluating an already-computed policy.

Implements both first-visit and every-visit MC methods, with confidence
interval computation via the bootstrap or CLT.

References
----------
- Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An
  Introduction*, 2nd ed., Ch. 5.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import numpy as np

from usability_oracle.mdp.models import MDP
from usability_oracle.mdp.trajectory import Trajectory, TrajectorySampler
from usability_oracle.policy.models import Policy, QValues
from usability_oracle.policy.softmax import SoftmaxPolicy

logger = logging.getLogger(__name__)


class MonteCarloEstimator:
    """Monte Carlo estimation of value functions and free energy.

    Parameters
    ----------
    discount : float
        Discount factor γ.
    rng : np.random.Generator, optional
        Random number generator.
    """

    def __init__(
        self,
        discount: float = 0.99,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.discount = discount
        self.rng = rng or np.random.default_rng()

    # ── Value estimation --------------------------------------------------

    def estimate_value(
        self,
        mdp: MDP,
        policy: Policy,
        n_trajectories: int = 1000,
        max_steps: int = 500,
        method: str = "first_visit",
    ) -> dict[str, float]:
        """Estimate V^π(s) via Monte Carlo.

        Parameters
        ----------
        mdp : MDP
        policy : Policy
        n_trajectories : int
        max_steps : int
        method : str
            ``"first_visit"`` or ``"every_visit"``.

        Returns
        -------
        dict[str, float]
            Estimated V(s) for each visited state.
        """
        sampler = TrajectorySampler(rng=self.rng)
        trajectories = sampler.sample(
            mdp, policy.state_action_probs, n_trajectories, max_steps
        )

        if method == "first_visit":
            return self._first_visit_mc(trajectories)
        else:
            return self._every_visit_mc(trajectories)

    def estimate_q_values(
        self,
        mdp: MDP,
        policy: Policy,
        n_trajectories: int = 1000,
        max_steps: int = 500,
    ) -> QValues:
        """Estimate Q^π(s, a) via first-visit Monte Carlo.

        Parameters
        ----------
        mdp : MDP
        policy : Policy
        n_trajectories : int
        max_steps : int

        Returns
        -------
        QValues
        """
        sampler = TrajectorySampler(rng=self.rng)
        trajectories = sampler.sample(
            mdp, policy.state_action_probs, n_trajectories, max_steps
        )

        # First-visit MC for (s, a) pairs
        q_sums: dict[str, dict[str, float]] = {}
        q_counts: dict[str, dict[str, int]] = {}

        for traj in trajectories:
            visited_sa: set[tuple[str, str]] = set()
            # Compute returns from the end
            returns = self._compute_returns(traj)

            for i, step in enumerate(traj.steps):
                sa = (step.state_id, step.action_id)
                if sa in visited_sa:
                    continue
                visited_sa.add(sa)

                sid, aid = sa
                if sid not in q_sums:
                    q_sums[sid] = {}
                    q_counts[sid] = {}
                if aid not in q_sums[sid]:
                    q_sums[sid][aid] = 0.0
                    q_counts[sid][aid] = 0

                q_sums[sid][aid] += returns[i]
                q_counts[sid][aid] += 1

        # Average
        q_values: dict[str, dict[str, float]] = {}
        for sid in q_sums:
            q_values[sid] = {}
            for aid in q_sums[sid]:
                count = q_counts[sid][aid]
                q_values[sid][aid] = q_sums[sid][aid] / max(count, 1)

        return QValues(values=q_values)

    def estimate_free_energy(
        self,
        mdp: MDP,
        policy: Policy,
        beta: float,
        prior: Policy,
        n_trajectories: int = 1000,
        max_steps: int = 500,
    ) -> float:
        """Estimate the free energy F(π) via Monte Carlo.

        F(π) = E_π[c] + (1/β) D_KL(π ‖ p₀)

        The expected cost is estimated from trajectories; the KL divergence
        is computed analytically from the policy distributions.

        Parameters
        ----------
        mdp : MDP
        policy : Policy
        beta : float
        prior : Policy
        n_trajectories : int
        max_steps : int

        Returns
        -------
        float
            Estimated free energy.
        """
        sampler = TrajectorySampler(rng=self.rng)
        trajectories = sampler.sample(
            mdp, policy.state_action_probs, n_trajectories, max_steps
        )

        # Expected cost from trajectories
        costs = np.array([t.total_cost for t in trajectories], dtype=np.float64)
        mean_cost = float(np.mean(costs)) if len(costs) > 0 else 0.0

        # Information cost (analytical)
        info_cost = 0.0
        n_states = 0
        for sid in policy.state_action_probs:
            kl = SoftmaxPolicy.kl_divergence(policy, prior, sid)
            info_cost += kl
            n_states += 1

        avg_info = info_cost / max(n_states, 1)
        if beta > 0:
            avg_info /= beta

        return mean_cost + avg_info

    # ── MC methods --------------------------------------------------------

    def _first_visit_mc(
        self, trajectories: list[Trajectory]
    ) -> dict[str, float]:
        """First-visit MC: average the return from the first visit to each state.

        Only the first occurrence of each state in a trajectory contributes
        to its value estimate, reducing bias from revisits.
        """
        value_sums: dict[str, float] = {}
        value_counts: dict[str, int] = {}

        for traj in trajectories:
            visited: set[str] = set()
            returns = self._compute_returns(traj)

            for i, step in enumerate(traj.steps):
                if step.state_id in visited:
                    continue
                visited.add(step.state_id)

                sid = step.state_id
                if sid not in value_sums:
                    value_sums[sid] = 0.0
                    value_counts[sid] = 0

                value_sums[sid] += returns[i]
                value_counts[sid] += 1

        values: dict[str, float] = {}
        for sid in value_sums:
            values[sid] = value_sums[sid] / max(value_counts[sid], 1)

        return values

    def _every_visit_mc(
        self, trajectories: list[Trajectory]
    ) -> dict[str, float]:
        """Every-visit MC: average the return from every visit to each state.

        All occurrences contribute, yielding lower variance but potentially
        higher bias than first-visit.
        """
        value_sums: dict[str, float] = {}
        value_counts: dict[str, int] = {}

        for traj in trajectories:
            returns = self._compute_returns(traj)

            for i, step in enumerate(traj.steps):
                sid = step.state_id
                if sid not in value_sums:
                    value_sums[sid] = 0.0
                    value_counts[sid] = 0

                value_sums[sid] += returns[i]
                value_counts[sid] += 1

        values: dict[str, float] = {}
        for sid in value_sums:
            values[sid] = value_sums[sid] / max(value_counts[sid], 1)

        return values

    def _compute_returns(self, traj: Trajectory) -> list[float]:
        """Compute discounted returns G_t for each step in the trajectory.

        G_t = c_t + γ c_{t+1} + γ² c_{t+2} + …

        Uses backward accumulation for O(n) computation.

        Parameters
        ----------
        traj : Trajectory

        Returns
        -------
        list[float]
            Return for each step index.
        """
        n = len(traj.steps)
        if n == 0:
            return []

        returns = [0.0] * n
        g = 0.0
        for i in range(n - 1, -1, -1):
            g = traj.steps[i].cost + self.discount * g
            returns[i] = g

        return returns

    # ── Confidence intervals ----------------------------------------------

    @staticmethod
    def confidence_intervals(
        estimates: dict[str, list[float]],
        alpha: float = 0.05,
    ) -> dict[str, tuple[float, float]]:
        """Compute confidence intervals for value estimates.

        Uses the Central Limit Theorem (normal approximation) to construct
        (1 − α) confidence intervals for each state.

        Parameters
        ----------
        estimates : dict[str, list[float]]
            Mapping ``state_id -> list of return samples``.
        alpha : float
            Significance level (default 0.05 for 95% CI).

        Returns
        -------
        dict[str, tuple[float, float]]
            Mapping ``state_id -> (lower, upper)`` bounds.
        """
        from scipy import stats  # type: ignore[import-untyped]

        z = stats.norm.ppf(1 - alpha / 2)
        intervals: dict[str, tuple[float, float]] = {}

        for sid, samples in estimates.items():
            arr = np.array(samples, dtype=np.float64)
            n = len(arr)
            if n < 2:
                mean = float(np.mean(arr)) if n > 0 else 0.0
                intervals[sid] = (mean, mean)
                continue

            mean = float(np.mean(arr))
            se = float(np.std(arr, ddof=1)) / math.sqrt(n)
            margin = z * se
            intervals[sid] = (mean - margin, mean + margin)

        return intervals
