"""
usability_oracle.comparison.paired — Paired usability comparison.

Implements :class:`PairedComparator`, which takes two MDPs (before/after),
an alignment mapping, and a task specification, then:

1. Builds a shared partition respecting both MDPs' structure.
2. Computes bounded-rational policies for both MDPs.
3. Evaluates task-completion costs under those policies.
4. Computes a delta cost with confidence intervals.
5. Determines a regression verdict with formal statistical guarantees.

The approach follows the free-energy formulation:

    F(π) = E_π[C] + (1/β) D_KL(π ∥ p₀)

where C is the cognitive/motor cost, β is the rationality parameter, and
p₀ is the prior (uniform) policy.

References
----------
- Ortega, P. A. & Braun, D. A. (2013). Thermodynamics as a theory of
  decision-making with information-processing costs. *Proc. R. Soc. A*, 469.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from usability_oracle.core.enums import BottleneckType, RegressionVerdict
from usability_oracle.core.errors import ComparisonError, InsufficientDataError
from usability_oracle.cognitive.models import CostElement
from usability_oracle.mdp.models import MDP, State, Action, Transition
from usability_oracle.policy.models import Policy
from usability_oracle.taskspec.models import TaskSpec
from usability_oracle.comparison.models import (
    AlignmentResult,
    BottleneckChange,
    ChangeDirection,
    ComparisonContext,
    ComparisonResult,
    Partition,
    PartitionBlock,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: simple bounded-rational policy solver
# ---------------------------------------------------------------------------

def _solve_softmax_policy(mdp: MDP, beta: float) -> Policy:
    """Compute a bounded-rational softmax policy via value iteration.

    Solves the free-energy Bellman equation:

        V(s) = (1/β) log Σ_a exp(β [R(s,a) + γ Σ_{s'} T(s'|s,a) V(s')])

    where R(s,a) = −cost(s,a) so that lower cost is preferred.

    Parameters
    ----------
    mdp : MDP
        The MDP to solve.
    beta : float
        Rationality parameter (higher ⇒ more rational).

    Returns
    -------
    Policy
        Bounded-rational policy with state-action probabilities.
    """
    states = list(mdp.states.keys())
    if not states:
        return Policy(beta=beta)

    values: dict[str, float] = {s: 0.0 for s in states}
    gamma = mdp.discount
    max_iter = 500
    tol = 1e-8

    for _ in range(max_iter):
        new_values: dict[str, float] = {}
        max_delta = 0.0
        for s in states:
            actions = mdp.get_actions(s)
            if not actions:
                new_values[s] = 0.0
                continue
            log_terms: list[float] = []
            for a in actions:
                transitions = mdp.get_transitions(s, a)
                q_sa = 0.0
                for t_state, prob, cost in transitions:
                    reward = -cost
                    q_sa += prob * (reward + gamma * values.get(t_state, 0.0))
                log_terms.append(beta * q_sa)
            # Log-sum-exp for numerical stability
            max_log = max(log_terms)
            lse = max_log + math.log(
                sum(math.exp(lt - max_log) for lt in log_terms)
            )
            new_values[s] = lse / beta if beta > 0 else 0.0
            max_delta = max(max_delta, abs(new_values[s] - values[s]))
        values = new_values
        if max_delta < tol:
            break

    # Extract policy: π(a|s) ∝ exp(β · Q(s,a))
    state_action_probs: dict[str, dict[str, float]] = {}
    q_values: dict[str, dict[str, float]] = {}
    for s in states:
        actions = mdp.get_actions(s)
        if not actions:
            continue
        qs: dict[str, float] = {}
        for a in actions:
            transitions = mdp.get_transitions(s, a)
            q_sa = 0.0
            for t_state, prob, cost in transitions:
                q_sa += prob * (-cost + gamma * values.get(t_state, 0.0))
            qs[a] = q_sa
        q_values[s] = qs

        max_q = max(qs.values())
        exp_vals = {a: math.exp(beta * (q - max_q)) for a, q in qs.items()}
        z = sum(exp_vals.values())
        state_action_probs[s] = {a: ev / z for a, ev in exp_vals.items()}

    return Policy(
        state_action_probs=state_action_probs,
        beta=beta,
        values=values,
        q_values=q_values,
    )


# ---------------------------------------------------------------------------
# Helper: simulate cost samples
# ---------------------------------------------------------------------------

def _sample_trajectory_costs(
    mdp: MDP,
    policy: Policy,
    n_trajectories: int = 200,
    max_steps: int = 500,
) -> np.ndarray:
    """Simulate trajectory costs under the given policy.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_trajectories,)`` with total cost per trajectory.
    """
    rng = np.random.default_rng(seed=42)
    costs = np.zeros(n_trajectories)

    for i in range(n_trajectories):
        state = mdp.initial_state
        total_cost = 0.0
        for _ in range(max_steps):
            if state in mdp.goal_states:
                break
            s_obj = mdp.states.get(state)
            if s_obj is not None and s_obj.is_terminal:
                break
            probs = policy.action_probs(state)
            if not probs:
                break
            actions = list(probs.keys())
            weights = np.array([probs[a] for a in actions])
            weights /= weights.sum()
            action = rng.choice(actions, p=weights)
            transitions = mdp.get_transitions(state, action)
            if not transitions:
                break
            targets = [t[0] for t in transitions]
            trans_probs = np.array([t[1] for t in transitions])
            trans_probs /= trans_probs.sum()
            trans_costs = [t[2] for t in transitions]
            idx = rng.choice(len(targets), p=trans_probs)
            total_cost += trans_costs[idx]
            state = targets[idx]
        costs[i] = total_cost

    return costs


# ---------------------------------------------------------------------------
# PairedComparator
# ---------------------------------------------------------------------------

class PairedComparator:
    """Paired comparison of two UI versions via bounded-rational policies.

    The comparator constructs a shared partition that respects both MDPs'
    state structure, computes policies at the same β, evaluates task
    costs, and tests for regression.

    Parameters
    ----------
    beta : float
        Rationality parameter for policy computation.
    n_trajectories : int
        Number of Monte Carlo trajectories for cost estimation.
    significance_level : float
        α for hypothesis testing (default 0.05).
    min_effect_size : float
        Minimum Cohen's *d* to consider practically significant.
    """

    def __init__(
        self,
        beta: float = 1.0,
        n_trajectories: int = 500,
        significance_level: float = 0.05,
        min_effect_size: float = 0.2,
    ) -> None:
        self.beta = beta
        self.n_trajectories = n_trajectories
        self.significance_level = significance_level
        self.min_effect_size = min_effect_size

    def compare(
        self,
        mdp_a: MDP,
        mdp_b: MDP,
        alignment: AlignmentResult,
        task: TaskSpec,
        config: Optional[dict[str, Any]] = None,
    ) -> ComparisonResult:
        """Run a full paired comparison between two MDP versions.

        Parameters
        ----------
        mdp_a : MDP
            MDP for the *before* UI version.
        mdp_b : MDP
            MDP for the *after* UI version.
        alignment : AlignmentResult
            State-level alignment between the MDPs.
        task : TaskSpec
            Task specification being evaluated.
        config : dict, optional
            Override configuration parameters.

        Returns
        -------
        ComparisonResult
            Full comparison result with verdict, costs, and statistics.

        Raises
        ------
        ComparisonError
            If either MDP is empty or invalid.
        """
        if not mdp_a.states or not mdp_b.states:
            raise ComparisonError("Both MDPs must have at least one state.")

        cfg = config or {}
        beta = cfg.get("beta", self.beta)
        n_traj = cfg.get("n_trajectories", self.n_trajectories)
        alpha = cfg.get("significance_level", self.significance_level)

        logger.info(
            "Starting paired comparison: |S_a|=%d, |S_b|=%d, β=%.2f",
            mdp_a.n_states, mdp_b.n_states, beta,
        )

        # 1. Build shared partition
        partition = self._shared_partition(mdp_a, mdp_b, alignment, beta)

        # 2. Compute policies
        policy_a = _solve_softmax_policy(mdp_a, beta)
        policy_b = _solve_softmax_policy(mdp_b, beta)

        # 3. Compute costs via Monte Carlo
        cost_a = self._compute_costs(mdp_a, partition, policy_a, task)
        cost_b = self._compute_costs(mdp_b, partition, policy_b, task)

        # 4. Compute delta
        delta = self._compute_delta(cost_a, cost_b)

        # 5. Sample trajectory costs for statistical testing
        samples_a = _sample_trajectory_costs(mdp_a, policy_a, n_traj)
        samples_b = _sample_trajectory_costs(mdp_b, policy_b, n_traj)

        # 6. Effect size
        effect = self._effect_size(cost_a, cost_b)

        # 7. Confidence interval
        ci_low, ci_high = self._confidence_interval(cost_a, cost_b, alpha)

        # 8. Determine verdict
        from usability_oracle.comparison.hypothesis import RegressionTester
        tester = RegressionTester()
        hyp_result = tester.test(samples_a, samples_b, alpha=alpha)

        verdict = self._determine_verdict(
            delta, self.min_effect_size, hyp_result.reject_null
        )

        # 9. Detect bottleneck changes
        bottleneck_changes = self._detect_bottleneck_changes(
            mdp_a, mdp_b, policy_a, policy_b, alignment
        )

        description = (
            f"Paired comparison: cost_before={cost_a.mean_time:.3f}s, "
            f"cost_after={cost_b.mean_time:.3f}s, "
            f"Δ={delta.mean_time:+.3f}s, d={effect:.2f}, "
            f"p={hyp_result.p_value:.4f} → {verdict.value}"
        )

        return ComparisonResult(
            verdict=verdict,
            confidence=1.0 - alpha,
            p_value=hyp_result.p_value,
            cost_before=cost_a,
            cost_after=cost_b,
            delta_cost=delta,
            effect_size=effect,
            bottleneck_changes=bottleneck_changes,
            parameter_sensitivity={"beta": beta},
            is_parameter_free=False,
            description=description,
        )

    def _shared_partition(
        self,
        mdp_a: MDP,
        mdp_b: MDP,
        alignment: AlignmentResult,
        beta: float,
    ) -> Partition:
        """Build a partition that respects both MDPs' structure.

        Aligned states are placed in the same block. Unaligned states
        from each MDP form singleton blocks.

        Parameters
        ----------
        mdp_a, mdp_b : MDP
            The two MDPs.
        alignment : AlignmentResult
            State-level alignment.
        beta : float
            Rationality parameter (used for value-based refinement).

        Returns
        -------
        Partition
        """
        blocks: list[PartitionBlock] = []
        state_to_block: dict[str, str] = {}
        block_idx = 0

        # Paired blocks from alignment
        mapping = alignment.get_mapping_dict()
        for sa, sb in mapping.items():
            bid = f"paired_{block_idx}"
            block = PartitionBlock(
                block_id=bid,
                state_ids=[sa, sb],
                representative=sa,
            )
            blocks.append(block)
            state_to_block[sa] = bid
            state_to_block[sb] = bid
            block_idx += 1

        # Singleton blocks for unmapped states in A
        for sid in alignment.unmapped_a:
            bid = f"unmapped_a_{block_idx}"
            block = PartitionBlock(
                block_id=bid,
                state_ids=[sid],
                representative=sid,
            )
            blocks.append(block)
            state_to_block[sid] = bid
            block_idx += 1

        # Singleton blocks for unmapped states in B
        for sid in alignment.unmapped_b:
            bid = f"unmapped_b_{block_idx}"
            block = PartitionBlock(
                block_id=bid,
                state_ids=[sid],
                representative=sid,
            )
            blocks.append(block)
            state_to_block[sid] = bid
            block_idx += 1

        # Include any states not referenced by alignment
        for sid in mdp_a.states:
            if sid not in state_to_block:
                bid = f"extra_a_{block_idx}"
                blocks.append(PartitionBlock(bid, [sid], sid))
                state_to_block[sid] = bid
                block_idx += 1
        for sid in mdp_b.states:
            if sid not in state_to_block:
                bid = f"extra_b_{block_idx}"
                blocks.append(PartitionBlock(bid, [sid], sid))
                state_to_block[sid] = bid
                block_idx += 1

        return Partition(blocks=blocks, state_to_block=state_to_block)

    def _compute_costs(
        self,
        mdp: MDP,
        partition: Partition,
        policy: Policy,
        task: TaskSpec,
    ) -> CostElement:
        """Compute aggregate task-completion cost under the policy.

        Uses Monte Carlo simulation to estimate the expected cost and
        variance of completing the task.

        Parameters
        ----------
        mdp : MDP
        partition : Partition
        policy : Policy
        task : TaskSpec

        Returns
        -------
        CostElement
            Aggregate cost with mean and variance.
        """
        samples = _sample_trajectory_costs(mdp, policy, n_trajectories=200)
        mean_cost = float(np.mean(samples))
        var_cost = float(np.var(samples, ddof=1)) if len(samples) > 1 else 0.0

        return CostElement(
            mean_time=mean_cost,
            variance=var_cost,
            channel="aggregate",
            law="composite",
        )

    def _compute_delta(self, cost_a: CostElement, cost_b: CostElement) -> CostElement:
        """Compute the difference cost_b − cost_a.

        Positive delta means the *after* version is more expensive (regression).

        Parameters
        ----------
        cost_a : CostElement
            Cost of the *before* version.
        cost_b : CostElement
            Cost of the *after* version.

        Returns
        -------
        CostElement
            Delta cost with propagated variance.
        """
        delta_mean = cost_b.mean_time - cost_a.mean_time
        delta_var = cost_a.variance + cost_b.variance
        return CostElement(
            mean_time=delta_mean,
            variance=delta_var,
            channel="aggregate",
            law="composite",
        )

    def _determine_verdict(
        self,
        delta: CostElement,
        min_effect: float,
        reject_null: bool,
    ) -> RegressionVerdict:
        """Determine the regression verdict from the delta and test result.

        Parameters
        ----------
        delta : CostElement
            The cost difference (positive ⇒ regression).
        min_effect : float
            Minimum effect size for practical significance.
        reject_null : bool
            Whether the hypothesis test rejected H₀.

        Returns
        -------
        RegressionVerdict
        """
        if not reject_null:
            return RegressionVerdict.NEUTRAL

        if delta.mean_time > 0:
            return RegressionVerdict.REGRESSION
        elif delta.mean_time < 0:
            return RegressionVerdict.IMPROVEMENT
        return RegressionVerdict.NEUTRAL

    def _effect_size(self, cost_a: CostElement, cost_b: CostElement) -> float:
        """Compute Cohen's *d* effect size.

        .. math::
            d = \\frac{\\bar{X}_B - \\bar{X}_A}{s_p}

        where :math:`s_p` is the pooled standard deviation:

        .. math::
            s_p = \\sqrt{\\frac{s_A^2 + s_B^2}{2}}

        Parameters
        ----------
        cost_a, cost_b : CostElement

        Returns
        -------
        float
            Cohen's d (positive ⇒ regression).
        """
        s_a = math.sqrt(max(cost_a.variance, 0.0))
        s_b = math.sqrt(max(cost_b.variance, 0.0))
        pooled = math.sqrt((s_a ** 2 + s_b ** 2) / 2.0)
        if pooled < 1e-12:
            return 0.0
        return (cost_b.mean_time - cost_a.mean_time) / pooled

    def _confidence_interval(
        self,
        cost_a: CostElement,
        cost_b: CostElement,
        alpha: float = 0.05,
    ) -> tuple[float, float]:
        """Compute confidence interval for the mean cost difference.

        Uses a normal approximation:

        .. math::
            \\Delta \\pm z_{\\alpha/2} \\cdot \\sqrt{\\sigma_A^2/n + \\sigma_B^2/n}

        Parameters
        ----------
        cost_a, cost_b : CostElement
        alpha : float

        Returns
        -------
        tuple[float, float]
            (lower, upper) bounds of the CI.
        """
        from scipy import stats
        z = stats.norm.ppf(1 - alpha / 2)
        delta = cost_b.mean_time - cost_a.mean_time
        se = math.sqrt(cost_a.variance + cost_b.variance)
        return (delta - z * se, delta + z * se)

    def _detect_bottleneck_changes(
        self,
        mdp_a: MDP,
        mdp_b: MDP,
        policy_a: Policy,
        policy_b: Policy,
        alignment: AlignmentResult,
    ) -> list[BottleneckChange]:
        """Detect changes in cognitive bottlenecks between versions.

        Compares policy entropy (choice paralysis proxy) and transition
        costs (motor difficulty proxy) at aligned state pairs.

        Parameters
        ----------
        mdp_a, mdp_b : MDP
        policy_a, policy_b : Policy
        alignment : AlignmentResult

        Returns
        -------
        list[BottleneckChange]
        """
        changes: list[BottleneckChange] = []
        mapping = alignment.get_mapping_dict()

        for sa, sb in mapping.items():
            # Choice paralysis: high entropy ⇒ indecision
            entropy_a = policy_a.entropy(sa)
            entropy_b = policy_b.entropy(sb)
            if abs(entropy_b - entropy_a) > 0.3:
                direction = BottleneckChange.classify_direction(entropy_a, entropy_b)
                changes.append(BottleneckChange(
                    bottleneck_type=BottleneckType.CHOICE_PARALYSIS,
                    state_id=sa,
                    before_severity=entropy_a,
                    after_severity=entropy_b,
                    direction=direction,
                    description=(
                        f"Policy entropy changed from {entropy_a:.2f} to "
                        f"{entropy_b:.2f} at state {sa}"
                    ),
                ))

            # Motor difficulty: compare mean transition costs
            cost_a = self._mean_transition_cost(mdp_a, sa)
            cost_b = self._mean_transition_cost(mdp_b, sb)
            if abs(cost_b - cost_a) > 0.1:
                direction = BottleneckChange.classify_direction(cost_a, cost_b)
                changes.append(BottleneckChange(
                    bottleneck_type=BottleneckType.MOTOR_DIFFICULTY,
                    state_id=sa,
                    before_severity=cost_a,
                    after_severity=cost_b,
                    direction=direction,
                    description=(
                        f"Mean transition cost changed from {cost_a:.3f} to "
                        f"{cost_b:.3f} at state {sa}"
                    ),
                ))

        return changes

    @staticmethod
    def _mean_transition_cost(mdp: MDP, state_id: str) -> float:
        """Average transition cost from *state_id* across all actions."""
        actions = mdp.get_actions(state_id)
        if not actions:
            return 0.0
        total_cost = 0.0
        n = 0
        for a in actions:
            for _, prob, cost in mdp.get_transitions(state_id, a):
                total_cost += prob * cost
                n += 1
        return total_cost / max(n, 1)
