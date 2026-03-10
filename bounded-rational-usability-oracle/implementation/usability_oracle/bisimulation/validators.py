"""
usability_oracle.bisimulation.validators — Validation of bisimulation partitions.

Provides structural and semantic validation of a bisimulation partition:
  - **Policy consistency**: states within a block should have similar π_β.
  - **Transition consistency**: abstract transition probabilities should be
    well-defined (no large variance within blocks).
  - **Reward consistency**: states in the same block should have similar costs.
  - **Abstraction error**: bound on the value-function loss.

References
----------
- Ferns, Panangaden & Precup (2004). Metrics for finite Markov decision
  processes. *UAI*.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from usability_oracle.bisimulation.cognitive_distance import (
    CognitiveDistanceComputer,
    _soft_value_iteration,
)
from usability_oracle.bisimulation.models import Partition
from usability_oracle.mdp.models import MDP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of bisimulation partition validation.

    Attributes
    ----------
    is_valid : bool
        True if no errors were found.
    policy_issues : list[str]
        Descriptions of policy-consistency violations.
    transition_issues : list[str]
        Descriptions of transition-consistency violations.
    reward_issues : list[str]
        Descriptions of reward-consistency violations.
    abstraction_error : float
        Estimated upper bound on value-function error.
    max_policy_divergence : float
        Maximum TV distance between policies within any block.
    max_reward_variance : float
        Maximum reward variance within any block.
    metadata : dict[str, Any]
        Additional diagnostics.
    """

    is_valid: bool
    policy_issues: list[str] = field(default_factory=list)
    transition_issues: list[str] = field(default_factory=list)
    reward_issues: list[str] = field(default_factory=list)
    abstraction_error: float = 0.0
    max_policy_divergence: float = 0.0
    max_reward_variance: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_issues(self) -> int:
        return (
            len(self.policy_issues)
            + len(self.transition_issues)
            + len(self.reward_issues)
        )

    def summary(self) -> str:
        parts = [
            f"Valid: {self.is_valid}",
            f"Policy issues: {len(self.policy_issues)}",
            f"Transition issues: {len(self.transition_issues)}",
            f"Reward issues: {len(self.reward_issues)}",
            f"Abstraction error: {self.abstraction_error:.6f}",
        ]
        return " | ".join(parts)

    def __repr__(self) -> str:
        return (
            f"ValidationResult(valid={self.is_valid}, "
            f"issues={self.n_issues}, error={self.abstraction_error:.4f})"
        )


# ---------------------------------------------------------------------------
# BisimulationValidator
# ---------------------------------------------------------------------------

@dataclass
class BisimulationValidator:
    """Validate a bisimulation partition against an MDP.

    Checks policy consistency, transition consistency, reward consistency,
    and computes the abstraction error bound.

    Parameters
    ----------
    policy_tolerance : float
        Maximum allowed TV distance between policies within a block.
    reward_tolerance : float
        Maximum allowed reward standard deviation within a block.
    transition_tolerance : float
        Maximum allowed variance in abstract transition probabilities.
    """

    policy_tolerance: float = 0.05
    reward_tolerance: float = 0.1
    transition_tolerance: float = 0.05

    # ── Public API --------------------------------------------------------

    def validate(
        self,
        mdp: MDP,
        partition: Partition,
        beta: float,
        epsilon: float = 0.01,
    ) -> ValidationResult:
        """Run all validation checks.

        Parameters
        ----------
        mdp : MDP
        partition : Partition
        beta : float
            Rationality parameter.
        epsilon : float
            Tolerance for the bisimulation relation.

        Returns
        -------
        ValidationResult
        """
        # Structural check first
        if not partition.is_valid():
            return ValidationResult(
                is_valid=False,
                policy_issues=["Partition structural check failed"],
            )

        # Check partition covers MDP states
        partition_states = partition.states()
        mdp_states = set(mdp.states.keys())
        if partition_states != frozenset(mdp_states):
            missing = mdp_states - partition_states
            extra = partition_states - mdp_states
            issues = []
            if missing:
                issues.append(f"States missing from partition: {sorted(missing)[:5]}")
            if extra:
                issues.append(f"Extra states in partition: {sorted(extra)[:5]}")
            return ValidationResult(
                is_valid=False,
                policy_issues=issues,
            )

        policy_issues = self._check_policy_consistency(mdp, partition, beta)
        transition_issues = self._check_transition_consistency(mdp, partition)
        reward_issues = self._check_reward_consistency(mdp, partition)
        abstraction_error = self._compute_abstraction_error(mdp, partition, beta)

        # Compute max policy divergence across all blocks
        max_policy_div = self._max_policy_divergence(mdp, partition, beta)

        # Compute max reward variance
        max_reward_var = self._max_reward_variance(mdp, partition)

        is_valid = (
            len(policy_issues) == 0
            and len(transition_issues) == 0
            and len(reward_issues) == 0
        )

        return ValidationResult(
            is_valid=is_valid,
            policy_issues=policy_issues,
            transition_issues=transition_issues,
            reward_issues=reward_issues,
            abstraction_error=abstraction_error,
            max_policy_divergence=max_policy_div,
            max_reward_variance=max_reward_var,
            metadata={
                "beta": beta,
                "epsilon": epsilon,
                "n_blocks": partition.n_blocks,
                "n_states": len(partition.state_to_block),
            },
        )

    # ── Policy consistency ------------------------------------------------

    def _check_policy_consistency(
        self,
        mdp: MDP,
        partition: Partition,
        beta: float,
    ) -> list[str]:
        """Check that states within each block have similar π_β distributions.

        For each block, computes the TV distance between the policy at each
        state and the block-average policy.  Reports blocks where the max
        divergence exceeds ``policy_tolerance``.

        Returns
        -------
        list[str]
            Issue descriptions.
        """
        issues: list[str] = []
        values = _soft_value_iteration(mdp, beta)

        for block_idx, block in enumerate(partition.blocks):
            if len(block) <= 1:
                continue

            states = sorted(block)
            actions = set()
            for s in states:
                actions.update(mdp.get_actions(s))
            action_order = sorted(actions)

            if not action_order:
                continue

            # Compute policy distribution for each state
            policies: list[np.ndarray] = []
            for s in states:
                pi = self._compute_ordered_policy(
                    mdp, s, beta, values, action_order
                )
                policies.append(pi)

            # Average policy as reference
            avg_policy = np.mean(policies, axis=0)

            # Check each state against average
            for i, s in enumerate(states):
                tv = 0.5 * np.sum(np.abs(policies[i] - avg_policy))
                if tv > self.policy_tolerance:
                    issues.append(
                        f"Block {block_idx}: state {s!r} diverges from "
                        f"block average by d_TV={tv:.4f} "
                        f"(tolerance={self.policy_tolerance})"
                    )

        return issues

    # ── Transition consistency --------------------------------------------

    def _check_transition_consistency(
        self,
        mdp: MDP,
        partition: Partition,
    ) -> list[str]:
        """Check that abstract transition probabilities are consistent.

        For each block and action, compute the variance of per-state transition
        probabilities to each target block.  High variance means the
        abstraction poorly approximates the concrete transitions.

        Returns
        -------
        list[str]
        """
        issues: list[str] = []

        for block_idx, block in enumerate(partition.blocks):
            if len(block) <= 1:
                continue

            states = sorted(block)

            # Collect all actions available from this block
            block_actions: set[str] = set()
            for s in states:
                block_actions.update(mdp.get_actions(s))

            for aid in sorted(block_actions):
                # For each state, compute distribution over target blocks
                per_state_dists: list[dict[int, float]] = []
                for s in states:
                    dist: dict[int, float] = defaultdict(float)
                    for target, prob, _ in mdp.get_transitions(s, aid):
                        target_block = partition.state_to_block.get(target, -1)
                        dist[target_block] += prob
                    per_state_dists.append(dict(dist))

                # Compute variance of transition probabilities to each target block
                all_targets = set()
                for d in per_state_dists:
                    all_targets.update(d.keys())

                for target_block in all_targets:
                    probs = [d.get(target_block, 0.0) for d in per_state_dists]
                    variance = float(np.var(probs))
                    if variance > self.transition_tolerance ** 2:
                        issues.append(
                            f"Block {block_idx}, action {aid!r} → "
                            f"block {target_block}: transition prob "
                            f"variance={variance:.4f} "
                            f"(tolerance={self.transition_tolerance**2:.4f})"
                        )

        return issues

    # ── Reward consistency ------------------------------------------------

    def _check_reward_consistency(
        self,
        mdp: MDP,
        partition: Partition,
    ) -> list[str]:
        """Check that states in the same block have similar expected costs.

        For each block, computes the standard deviation of per-state expected
        costs and reports violations.

        Returns
        -------
        list[str]
        """
        issues: list[str] = []

        for block_idx, block in enumerate(partition.blocks):
            if len(block) <= 1:
                continue

            states = sorted(block)

            # Compute expected cost per state (averaged over actions)
            state_costs: list[float] = []
            for s in states:
                actions = mdp.get_actions(s)
                if not actions:
                    state_costs.append(0.0)
                    continue
                total_cost = 0.0
                for aid in actions:
                    for _, prob, cost in mdp.get_transitions(s, aid):
                        total_cost += prob * cost
                state_costs.append(total_cost / len(actions))

            std_cost = float(np.std(state_costs))
            if std_cost > self.reward_tolerance:
                issues.append(
                    f"Block {block_idx}: cost std={std_cost:.4f} "
                    f"(tolerance={self.reward_tolerance})"
                )

        return issues

    # ── Abstraction error -------------------------------------------------

    def _compute_abstraction_error(
        self,
        mdp: MDP,
        partition: Partition,
        beta: float,
    ) -> float:
        """Compute an upper bound on the value-function abstraction error.

        error ≈ max_block max_{s1, s2 ∈ block} |V(s1) − V(s2)|

        where V is the soft value function at rationality β.

        Parameters
        ----------
        mdp : MDP
        partition : Partition
        beta : float

        Returns
        -------
        float
        """
        values = _soft_value_iteration(mdp, beta)
        max_error = 0.0

        for block in partition.blocks:
            if len(block) <= 1:
                continue
            block_values = [values.get(s, 0.0) for s in block]
            block_range = max(block_values) - min(block_values)
            max_error = max(max_error, block_range)

        return max_error

    # ── Helper methods ----------------------------------------------------

    def _max_policy_divergence(
        self,
        mdp: MDP,
        partition: Partition,
        beta: float,
    ) -> float:
        """Compute maximum policy TV distance within any block."""
        values = _soft_value_iteration(mdp, beta)
        max_div = 0.0

        for block in partition.blocks:
            if len(block) <= 1:
                continue

            states = sorted(block)
            actions = set()
            for s in states:
                actions.update(mdp.get_actions(s))
            action_order = sorted(actions)
            if not action_order:
                continue

            policies = [
                self._compute_ordered_policy(mdp, s, beta, values, action_order)
                for s in states
            ]

            for i in range(len(policies)):
                for j in range(i + 1, len(policies)):
                    tv = 0.5 * np.sum(np.abs(policies[i] - policies[j]))
                    max_div = max(max_div, tv)

        return max_div

    def _max_reward_variance(
        self,
        mdp: MDP,
        partition: Partition,
    ) -> float:
        """Compute maximum cost variance within any block."""
        max_var = 0.0

        for block in partition.blocks:
            if len(block) <= 1:
                continue
            costs: list[float] = []
            for s in block:
                for aid in mdp.get_actions(s):
                    for _, prob, cost in mdp.get_transitions(s, aid):
                        costs.append(cost)
            if costs:
                max_var = max(max_var, float(np.var(costs)))

        return max_var

    def _compute_ordered_policy(
        self,
        mdp: MDP,
        state: str,
        beta: float,
        values: dict[str, float],
        action_order: list[str],
    ) -> np.ndarray:
        """Compute π_β(·|s) in a canonical action ordering."""
        gamma = mdp.discount
        q = np.zeros(len(action_order), dtype=np.float64)

        for idx, aid in enumerate(action_order):
            transitions = mdp.get_transitions(state, aid)
            expected_cost = 0.0
            expected_future = 0.0
            for target, prob, cost in transitions:
                expected_cost += prob * cost
                expected_future += prob * values.get(target, 0.0)
            q[idx] = -expected_cost + gamma * expected_future

        if beta <= 1e-10:
            return np.ones(len(action_order)) / len(action_order)

        scaled = beta * q
        scaled -= np.max(scaled)
        exp_q = np.exp(scaled)
        total = np.sum(exp_q)
        if total <= 0:
            return np.ones(len(action_order)) / len(action_order)
        return exp_q / total
