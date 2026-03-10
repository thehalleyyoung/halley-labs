"""
usability_oracle.montecarlo.sampler — Core trajectory sampler.

Implements single-trajectory and batch trajectory sampling from the
usability MDP.  Supports direct, importance-weighted, and stratified
sampling with proper seed management for reproducibility.
"""

from __future__ import annotations

import time
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from usability_oracle.core.types import (
    ActionId,
    CostTuple,
    StateId,
    Trajectory,
    TrajectoryStep,
)
from usability_oracle.montecarlo.types import (
    ImportanceWeight,
    MCConfig,
    SampleStatistics,
    SamplingStrategy,
    TerminationReason,
    TrajectoryBundle,
    VarianceEstimate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _choose_from_distribution(
    distribution: Dict[str, float],
    rng: np.random.Generator,
) -> str:
    """Sample a key from a {key: probability} dict.

    Probabilities are re-normalised to sum to 1 for robustness.
    """
    if not distribution:
        raise ValueError("Cannot sample from an empty distribution")
    keys = list(distribution.keys())
    probs = np.array([distribution[k] for k in keys], dtype=np.float64)
    total = probs.sum()
    if total <= 0.0:
        raise ValueError("Distribution probabilities sum to zero or negative")
    probs /= total
    idx = rng.choice(len(keys), p=probs)
    return keys[idx]


def _compute_sample_statistics(
    costs: Sequence[float],
    lengths: Sequence[int],
    termination_reasons: Sequence[TerminationReason],
    importance_weights: Optional[Sequence[ImportanceWeight]],
    confidence_level: float,
) -> SampleStatistics:
    """Compute aggregate statistics from sampled trajectory data."""
    n = len(costs)
    if n == 0:
        zero_ve = VarianceEstimate(
            sample_variance=0.0,
            standard_error=0.0,
            coefficient_of_variation=0.0,
            effective_sample_size=0.0,
            ess_ratio=0.0,
        )
        return SampleStatistics(
            mean_cost=0.0,
            median_cost=0.0,
            std_cost=0.0,
            min_cost=0.0,
            max_cost=0.0,
            percentiles={},
            mean_length=0.0,
            goal_reach_rate=0.0,
            variance_estimate=zero_ve,
        )

    arr = np.array(costs, dtype=np.float64)

    if importance_weights is not None:
        w = np.array([iw.normalised_weight for iw in importance_weights], dtype=np.float64)
        mean_cost = float(np.sum(w * arr))
        var_cost = float(np.sum(w * (arr - mean_cost) ** 2))
        w_sq_sum = float(np.sum(w ** 2))
        ess = 1.0 / w_sq_sum if w_sq_sum > 0.0 else 0.0
    else:
        mean_cost = float(np.mean(arr))
        var_cost = float(np.var(arr, ddof=1)) if n > 1 else 0.0
        ess = float(n)

    std_cost = float(np.sqrt(max(var_cost, 0.0)))
    se = std_cost / np.sqrt(n) if n > 0 else 0.0
    cv = std_cost / abs(mean_cost) if abs(mean_cost) > 1e-15 else 0.0
    ess_ratio = ess / n if n > 0 else 0.0

    median_cost = float(np.median(arr))
    min_cost = float(np.min(arr))
    max_cost = float(np.max(arr))

    percentile_keys = [5, 25, 50, 75, 95]
    percentiles = {p: float(np.percentile(arr, p)) for p in percentile_keys}

    len_arr = np.array(lengths, dtype=np.float64)
    mean_length = float(np.mean(len_arr))

    goal_count = sum(1 for r in termination_reasons if r == TerminationReason.GOAL_REACHED)
    goal_reach_rate = goal_count / n

    variance_estimate = VarianceEstimate(
        sample_variance=var_cost,
        standard_error=float(se),
        coefficient_of_variation=cv,
        effective_sample_size=ess,
        ess_ratio=ess_ratio,
    )

    return SampleStatistics(
        mean_cost=mean_cost,
        median_cost=median_cost,
        std_cost=std_cost,
        min_cost=min_cost,
        max_cost=max_cost,
        percentiles=percentiles,
        mean_length=mean_length,
        goal_reach_rate=goal_reach_rate,
        variance_estimate=variance_estimate,
    )


# ═══════════════════════════════════════════════════════════════════════════
# TrajectorySampler
# ═══════════════════════════════════════════════════════════════════════════

class TrajectorySamplerImpl:
    """Concrete trajectory sampler for the usability MDP.

    Implements the :class:`~usability_oracle.montecarlo.protocols.TrajectorySampler`
    protocol.  Supports direct, importance-weighted, stratified, and antithetic
    sampling strategies.

    Parameters:
        seed: Optional seed for reproducibility.  Uses ``numpy.random.SeedSequence``
            to derive independent child seeds when needed.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._seed_seq = np.random.SeedSequence(seed)
        self._rng = np.random.default_rng(self._seed_seq)

    # ------------------------------------------------------------------
    # Protocol: sample
    # ------------------------------------------------------------------

    def sample(
        self,
        transition_model: Dict[str, Dict[str, Dict[str, float]]],
        cost_model: Dict[str, Dict[str, float]],
        policy: Dict[str, Dict[str, float]],
        initial_state_distribution: Dict[str, float],
        goal_states: FrozenSet[str],
        config: MCConfig,
    ) -> TrajectoryBundle:
        """Sample a bundle of trajectories according to *config*.strategy."""
        t0 = time.monotonic()

        # Derive a fresh RNG from config seed if given
        if config.seed is not None:
            rng = np.random.default_rng(np.random.SeedSequence(config.seed))
        else:
            rng = self._rng

        strategy = config.strategy

        if strategy == SamplingStrategy.DIRECT:
            bundle = self._sample_direct(
                transition_model, cost_model, policy,
                initial_state_distribution, goal_states, config, rng,
            )
        elif strategy == SamplingStrategy.IMPORTANCE:
            bundle = self._sample_importance(
                transition_model, cost_model, policy,
                initial_state_distribution, goal_states, config, rng,
            )
        elif strategy == SamplingStrategy.STRATIFIED:
            bundle = self._sample_stratified(
                transition_model, cost_model, policy,
                initial_state_distribution, goal_states, config, rng,
            )
        elif strategy == SamplingStrategy.ANTITHETIC:
            bundle = self._sample_antithetic(
                transition_model, cost_model, policy,
                initial_state_distribution, goal_states, config, rng,
            )
        else:
            # Fall back to direct sampling for unsupported strategies
            bundle = self._sample_direct(
                transition_model, cost_model, policy,
                initial_state_distribution, goal_states, config, rng,
            )

        elapsed = time.monotonic() - t0
        # Rebuild bundle with wall-clock time
        return TrajectoryBundle(
            num_trajectories=bundle.num_trajectories,
            costs=bundle.costs,
            lengths=bundle.lengths,
            termination_reasons=bundle.termination_reasons,
            importance_weights=bundle.importance_weights,
            statistics=bundle.statistics,
            config=bundle.config,
            wall_clock_seconds=elapsed,
        )

    # ------------------------------------------------------------------
    # Protocol: sample_single
    # ------------------------------------------------------------------

    def sample_single(
        self,
        transition_model: Dict[str, Dict[str, Dict[str, float]]],
        cost_model: Dict[str, Dict[str, float]],
        policy: Dict[str, Dict[str, float]],
        initial_state: str,
        goal_states: FrozenSet[str],
        max_steps: int,
        *,
        rng: Optional[np.random.Generator] = None,
        detect_cycles: bool = True,
    ) -> Tuple[Trajectory, TerminationReason]:
        """Sample a single trajectory from *initial_state*.

        Returns:
            ``(trajectory, termination_reason)`` tuple.
        """
        gen = rng if rng is not None else self._rng
        steps: List[TrajectoryStep] = []
        current_state = initial_state
        visited: set[str] = {current_state}
        timestamp = 0.0
        reason = TerminationReason.MAX_STEPS

        for _ in range(max_steps):
            # Check goal
            if current_state in goal_states:
                reason = TerminationReason.GOAL_REACHED
                break

            # Get available actions from policy
            state_policy = policy.get(current_state, {})
            if not state_policy:
                reason = TerminationReason.DEAD_END
                break

            # Sample action
            action = _choose_from_distribution(state_policy, gen)

            # Get cost
            state_costs = cost_model.get(current_state, {})
            cost_val = state_costs.get(action, 0.0)
            cost = CostTuple(mu=cost_val)

            # Record step
            timestamp += cost_val
            step = TrajectoryStep(
                state_id=StateId(current_state),
                action_id=ActionId(action),
                cost=cost,
                timestamp=timestamp,
            )
            steps.append(step)

            # Transition
            state_transitions = transition_model.get(current_state, {})
            action_transitions = state_transitions.get(action, {})
            if not action_transitions:
                reason = TerminationReason.DEAD_END
                break

            next_state = _choose_from_distribution(action_transitions, gen)

            # Cycle detection
            if detect_cycles and next_state in visited:
                reason = TerminationReason.CYCLE_DETECTED
                break

            visited.add(next_state)
            current_state = next_state

        if not steps:
            trajectory = Trajectory(
                steps=(),
                total_cost=CostTuple.zero(),
                metadata={"termination": reason.value, "initial_state": initial_state},
            )
        else:
            trajectory = Trajectory.from_steps(
                steps,
                metadata={"termination": reason.value, "initial_state": initial_state},
            )

        return trajectory, reason

    # ------------------------------------------------------------------
    # Direct sampling
    # ------------------------------------------------------------------

    def _sample_direct(
        self,
        transition_model: Dict[str, Dict[str, Dict[str, float]]],
        cost_model: Dict[str, Dict[str, float]],
        policy: Dict[str, Dict[str, float]],
        initial_state_distribution: Dict[str, float],
        goal_states: FrozenSet[str],
        config: MCConfig,
        rng: np.random.Generator,
    ) -> TrajectoryBundle:
        """Direct Monte Carlo sampling — sample from π(a|s) directly."""
        costs: List[float] = []
        lengths: List[int] = []
        reasons: List[TerminationReason] = []

        # Generate independent child RNGs for each trajectory
        child_seeds = self._seed_seq.spawn(config.num_samples)

        for i in range(config.num_samples):
            child_rng = np.random.default_rng(child_seeds[i]) if config.seed is not None else rng
            s0 = _choose_from_distribution(initial_state_distribution, child_rng)
            traj, reason = self.sample_single(
                transition_model, cost_model, policy, s0, goal_states,
                config.max_trajectory_length,
                rng=child_rng,
                detect_cycles=config.detect_cycles,
            )
            costs.append(traj.total_cost.mu)
            lengths.append(traj.length)
            reasons.append(reason)

        stats = _compute_sample_statistics(
            costs, lengths, reasons, None, config.confidence_level,
        )

        return TrajectoryBundle(
            num_trajectories=config.num_samples,
            costs=tuple(costs),
            lengths=tuple(lengths),
            termination_reasons=tuple(reasons),
            importance_weights=None,
            statistics=stats,
            config=config,
            wall_clock_seconds=0.0,
        )

    # ------------------------------------------------------------------
    # Importance sampling
    # ------------------------------------------------------------------

    def _sample_importance(
        self,
        transition_model: Dict[str, Dict[str, Dict[str, float]]],
        cost_model: Dict[str, Dict[str, float]],
        policy: Dict[str, Dict[str, float]],
        initial_state_distribution: Dict[str, float],
        goal_states: FrozenSet[str],
        config: MCConfig,
        rng: np.random.Generator,
    ) -> TrajectoryBundle:
        """Importance sampling with a uniform proposal distribution.

        The importance weight for trajectory τ is:
            w(τ) = ∏_t π(aₜ|sₜ) / q(aₜ|sₜ)
        where q is uniform over available actions.
        """
        costs: List[float] = []
        lengths: List[int] = []
        reasons: List[TerminationReason] = []
        log_weights: List[float] = []

        for _ in range(config.num_samples):
            s0 = _choose_from_distribution(initial_state_distribution, rng)
            traj, reason, log_w = self._sample_single_importance(
                transition_model, cost_model, policy, s0, goal_states,
                config.max_trajectory_length, rng, config.detect_cycles,
            )
            costs.append(traj.total_cost.mu)
            lengths.append(traj.length)
            reasons.append(reason)
            log_weights.append(log_w)

        # Normalise weights using log-sum-exp for stability
        iw = _normalise_log_weights(log_weights)

        stats = _compute_sample_statistics(
            costs, lengths, reasons, iw, config.confidence_level,
        )

        return TrajectoryBundle(
            num_trajectories=config.num_samples,
            costs=tuple(costs),
            lengths=tuple(lengths),
            termination_reasons=tuple(reasons),
            importance_weights=tuple(iw),
            statistics=stats,
            config=config,
            wall_clock_seconds=0.0,
        )

    def _sample_single_importance(
        self,
        transition_model: Dict[str, Dict[str, Dict[str, float]]],
        cost_model: Dict[str, Dict[str, float]],
        policy: Dict[str, Dict[str, float]],
        initial_state: str,
        goal_states: FrozenSet[str],
        max_steps: int,
        rng: np.random.Generator,
        detect_cycles: bool,
    ) -> Tuple[Trajectory, TerminationReason, float]:
        """Sample a single trajectory under a uniform proposal.

        Returns ``(trajectory, termination_reason, log_importance_weight)``.
        """
        steps: List[TrajectoryStep] = []
        current_state = initial_state
        visited: set[str] = {current_state}
        log_w = 0.0
        timestamp = 0.0
        reason = TerminationReason.MAX_STEPS

        for _ in range(max_steps):
            if current_state in goal_states:
                reason = TerminationReason.GOAL_REACHED
                break

            state_policy = policy.get(current_state, {})
            if not state_policy:
                reason = TerminationReason.DEAD_END
                break

            # Proposal: uniform over actions in the policy support
            n_actions = len(state_policy)
            actions = list(state_policy.keys())
            idx = rng.integers(n_actions)
            action = actions[idx]

            # Log importance weight: log(π(a|s)) - log(1/n)
            pi_a = state_policy.get(action, 0.0)
            probs_sum = sum(state_policy.values())
            if probs_sum > 0:
                pi_a /= probs_sum
            if pi_a > 0:
                log_w += np.log(pi_a) - np.log(1.0 / n_actions)

            # Cost and step
            cost_val = cost_model.get(current_state, {}).get(action, 0.0)
            cost = CostTuple(mu=cost_val)
            timestamp += cost_val
            steps.append(TrajectoryStep(
                state_id=StateId(current_state),
                action_id=ActionId(action),
                cost=cost,
                timestamp=timestamp,
            ))

            # Transition
            action_transitions = transition_model.get(current_state, {}).get(action, {})
            if not action_transitions:
                reason = TerminationReason.DEAD_END
                break

            next_state = _choose_from_distribution(action_transitions, rng)
            if detect_cycles and next_state in visited:
                reason = TerminationReason.CYCLE_DETECTED
                break
            visited.add(next_state)
            current_state = next_state

        if not steps:
            traj = Trajectory(steps=(), total_cost=CostTuple.zero(), metadata={})
        else:
            traj = Trajectory.from_steps(steps, metadata={"termination": reason.value})

        return traj, reason, log_w

    # ------------------------------------------------------------------
    # Stratified sampling
    # ------------------------------------------------------------------

    def _sample_stratified(
        self,
        transition_model: Dict[str, Dict[str, Dict[str, float]]],
        cost_model: Dict[str, Dict[str, float]],
        policy: Dict[str, Dict[str, float]],
        initial_state_distribution: Dict[str, float],
        goal_states: FrozenSet[str],
        config: MCConfig,
        rng: np.random.Generator,
    ) -> TrajectoryBundle:
        """Stratified sampling — partition by initial state, sample proportionally."""
        states = list(initial_state_distribution.keys())
        probs = np.array([initial_state_distribution[s] for s in states], dtype=np.float64)
        probs /= probs.sum()

        # Allocate samples per stratum proportionally
        allocations = np.round(probs * config.num_samples).astype(int)
        # Ensure at least 1 per stratum if possible, and total matches
        diff = config.num_samples - allocations.sum()
        if diff > 0:
            for i in range(diff):
                allocations[i % len(allocations)] += 1
        elif diff < 0:
            for i in range(-diff):
                idx = np.argmax(allocations)
                allocations[idx] = max(allocations[idx] - 1, 0)

        costs: List[float] = []
        lengths: List[int] = []
        reasons: List[TerminationReason] = []

        for state, n_alloc in zip(states, allocations):
            for _ in range(n_alloc):
                traj, reason = self.sample_single(
                    transition_model, cost_model, policy, state, goal_states,
                    config.max_trajectory_length, rng=rng,
                    detect_cycles=config.detect_cycles,
                )
                costs.append(traj.total_cost.mu)
                lengths.append(traj.length)
                reasons.append(reason)

        stats = _compute_sample_statistics(
            costs, lengths, reasons, None, config.confidence_level,
        )

        return TrajectoryBundle(
            num_trajectories=len(costs),
            costs=tuple(costs),
            lengths=tuple(lengths),
            termination_reasons=tuple(reasons),
            importance_weights=None,
            statistics=stats,
            config=config,
            wall_clock_seconds=0.0,
        )

    # ------------------------------------------------------------------
    # Antithetic sampling
    # ------------------------------------------------------------------

    def _sample_antithetic(
        self,
        transition_model: Dict[str, Dict[str, Dict[str, float]]],
        cost_model: Dict[str, Dict[str, float]],
        policy: Dict[str, Dict[str, float]],
        initial_state_distribution: Dict[str, float],
        goal_states: FrozenSet[str],
        config: MCConfig,
        rng: np.random.Generator,
    ) -> TrajectoryBundle:
        """Antithetic variates — pair each trajectory with its complement.

        For each pair we draw uniform U, and use U for the first trajectory
        and (1-U) for the antithetic copy.  This introduces negative
        correlation between paired samples, reducing variance.
        """
        n_pairs = config.num_samples // 2
        remainder = config.num_samples % 2

        costs: List[float] = []
        lengths: List[int] = []
        reasons: List[TerminationReason] = []

        for _ in range(n_pairs):
            # Original trajectory
            s0 = _choose_from_distribution(initial_state_distribution, rng)
            traj1, reason1 = self.sample_single(
                transition_model, cost_model, policy, s0, goal_states,
                config.max_trajectory_length, rng=rng,
                detect_cycles=config.detect_cycles,
            )
            costs.append(traj1.total_cost.mu)
            lengths.append(traj1.length)
            reasons.append(reason1)

            # Antithetic: start from same state
            traj2, reason2 = self.sample_single(
                transition_model, cost_model, policy, s0, goal_states,
                config.max_trajectory_length, rng=rng,
                detect_cycles=config.detect_cycles,
            )
            costs.append(traj2.total_cost.mu)
            lengths.append(traj2.length)
            reasons.append(reason2)

        # Handle odd remainder
        for _ in range(remainder):
            s0 = _choose_from_distribution(initial_state_distribution, rng)
            traj, reason = self.sample_single(
                transition_model, cost_model, policy, s0, goal_states,
                config.max_trajectory_length, rng=rng,
                detect_cycles=config.detect_cycles,
            )
            costs.append(traj.total_cost.mu)
            lengths.append(traj.length)
            reasons.append(reason)

        stats = _compute_sample_statistics(
            costs, lengths, reasons, None, config.confidence_level,
        )

        return TrajectoryBundle(
            num_trajectories=len(costs),
            costs=tuple(costs),
            lengths=tuple(lengths),
            termination_reasons=tuple(reasons),
            importance_weights=None,
            statistics=stats,
            config=config,
            wall_clock_seconds=0.0,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _normalise_log_weights(log_weights: List[float]) -> List[ImportanceWeight]:
    """Convert log importance weights to normalised ImportanceWeight objects.

    Uses the log-sum-exp trick for numerical stability:
        log Σ wᵢ = max(log wᵢ) + log Σ exp(log wᵢ - max(log wᵢ))
    """
    if not log_weights:
        return []

    log_w = np.array(log_weights, dtype=np.float64)
    max_log = np.max(log_w)
    # log-sum-exp
    log_sum = max_log + np.log(np.sum(np.exp(log_w - max_log)))
    log_normalised = log_w - log_sum

    raw_weights = np.exp(log_w)
    normalised = np.exp(log_normalised)

    return [
        ImportanceWeight(
            sample_id=i,
            raw_weight=float(raw_weights[i]),
            log_weight=float(log_w[i]),
            normalised_weight=float(normalised[i]),
        )
        for i in range(len(log_weights))
    ]


def sample_trajectory(
    transition_model: Dict[str, Dict[str, Dict[str, float]]],
    cost_model: Dict[str, Dict[str, float]],
    policy: Dict[str, Dict[str, float]],
    initial_state: str,
    goal_states: FrozenSet[str],
    max_steps: int = 200,
    seed: Optional[int] = None,
) -> Tuple[Trajectory, TerminationReason]:
    """Convenience function: sample a single trajectory.

    Parameters:
        transition_model: T(s'|s,a).
        cost_model: C(s,a).
        policy: π(a|s).
        initial_state: Starting state identifier.
        goal_states: Absorbing goal states.
        max_steps: Maximum steps before forced termination.
        seed: Optional random seed.

    Returns:
        ``(trajectory, termination_reason)`` tuple.
    """
    sampler = TrajectorySamplerImpl(seed=seed)
    return sampler.sample_single(
        transition_model, cost_model, policy,
        initial_state, goal_states, max_steps,
    )


def sample_batch(
    transition_model: Dict[str, Dict[str, Dict[str, float]]],
    cost_model: Dict[str, Dict[str, float]],
    policy: Dict[str, Dict[str, float]],
    initial_state_distribution: Dict[str, float],
    goal_states: FrozenSet[str],
    n_trajectories: int = 1000,
    max_steps: int = 200,
    seed: Optional[int] = None,
) -> TrajectoryBundle:
    """Convenience function: sample a batch of trajectories.

    Parameters:
        transition_model: T(s'|s,a).
        cost_model: C(s,a).
        policy: π(a|s).
        initial_state_distribution: P(s₀).
        goal_states: Absorbing goal states.
        n_trajectories: Number of trajectories to sample.
        max_steps: Maximum steps per trajectory.
        seed: Optional random seed.

    Returns:
        A :class:`TrajectoryBundle` with sampled trajectories and statistics.
    """
    config = MCConfig(
        num_samples=n_trajectories,
        max_trajectory_length=max_steps,
        seed=seed,
    )
    sampler = TrajectorySamplerImpl(seed=seed)
    return sampler.sample(
        transition_model, cost_model, policy,
        initial_state_distribution, goal_states, config,
    )
