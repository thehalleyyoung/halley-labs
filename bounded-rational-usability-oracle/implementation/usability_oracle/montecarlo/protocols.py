"""
usability_oracle.montecarlo.protocols — Structural interfaces for the
Monte Carlo trajectory engine.

Defines protocols for trajectory sampling, variance reduction, and
parallel execution of Monte Carlo simulations over the usability MDP.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from usability_oracle.core.types import (
        PolicyDistribution,
        StateId,
        Trajectory,
    )
    from usability_oracle.montecarlo.types import (
        MCConfig,
        SampleStatistics,
        TrajectoryBundle,
        VarianceEstimate,
    )


# ═══════════════════════════════════════════════════════════════════════════
# TrajectorySampler
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class TrajectorySampler(Protocol):
    """Sample trajectory bundles from a usability MDP under a given policy.

    Implementations handle the full sampling loop: state initialisation,
    action selection according to the policy, transition, cost accumulation,
    and termination detection.
    """

    def sample(
        self,
        transition_model: Dict[str, Dict[str, Dict[str, float]]],
        cost_model: Dict[str, Dict[str, float]],
        policy: Dict[str, Dict[str, float]],
        initial_state_distribution: Dict[str, float],
        goal_states: frozenset[str],
        config: MCConfig,
    ) -> TrajectoryBundle:
        """Sample a bundle of trajectories.

        Parameters:
            transition_model: T(s'|s,a) — mapping
                state → {action → {next_state → probability}}.
            cost_model: C(s,a) — mapping state → {action → cost}.
            policy: π(a|s) — mapping state → {action → probability}.
            initial_state_distribution: P(s₀) — mapping state → probability.
            goal_states: Set of absorbing goal state identifiers.
            config: Sampling configuration.

        Returns:
            A :class:`TrajectoryBundle` with all sampled trajectories
            and aggregate statistics.
        """
        ...

    def sample_single(
        self,
        transition_model: Dict[str, Dict[str, Dict[str, float]]],
        cost_model: Dict[str, Dict[str, float]],
        policy: Dict[str, Dict[str, float]],
        initial_state: str,
        goal_states: frozenset[str],
        max_steps: int,
    ) -> Trajectory:
        """Sample a single trajectory from a fixed initial state.

        Parameters:
            transition_model: T(s'|s,a).
            cost_model: C(s,a).
            policy: π(a|s).
            initial_state: Starting state identifier.
            goal_states: Set of absorbing goal states.
            max_steps: Maximum allowed steps.

        Returns:
            A single :class:`Trajectory`.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# VarianceReducer
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class VarianceReducer(Protocol):
    """Apply variance-reduction techniques to Monte Carlo estimates.

    Techniques include control variates, importance sampling weight
    adjustment, Rao–Blackwellisation, and common random numbers.
    """

    def compute_proposal(
        self,
        target_policy: Dict[str, Dict[str, float]],
        cost_model: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        """Construct an importance-sampling proposal distribution.

        The proposal q(a|s) should minimise the variance of the
        importance-weighted cost estimator  E_q[w(τ) C(τ)].

        Parameters:
            target_policy: Target policy π(a|s).
            cost_model: Cost model C(s,a).

        Returns:
            Proposal policy  q(a|s)  in the same format as the input.
        """
        ...

    def estimate_variance(
        self,
        bundle: TrajectoryBundle,
    ) -> VarianceEstimate:
        """Compute variance diagnostics for a trajectory bundle.

        Parameters:
            bundle: A sampled :class:`TrajectoryBundle`.

        Returns:
            Detailed :class:`VarianceEstimate` with ESS, CV, etc.
        """
        ...

    def apply_control_variate(
        self,
        raw_estimates: Sequence[float],
        control_values: Sequence[float],
        control_expectation: float,
    ) -> Sequence[float]:
        """Apply a control-variate adjustment.

        The adjusted estimator is  X̂_cv = X̂ − c (Z − E[Z])
        where c is chosen to minimise variance.

        Parameters:
            raw_estimates: Original per-sample estimates.
            control_values: Per-sample values of the control variate Z.
            control_expectation: Known expectation E[Z].

        Returns:
            Adjusted estimates with reduced variance.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# ParallelExecutor
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class ParallelExecutor(Protocol):
    """Execute trajectory sampling across multiple workers.

    Handles work partitioning, seed management (ensuring reproducibility
    and independence across workers), and result aggregation.
    """

    def execute(
        self,
        sampler: TrajectorySampler,
        transition_model: Dict[str, Dict[str, Dict[str, float]]],
        cost_model: Dict[str, Dict[str, float]],
        policy: Dict[str, Dict[str, float]],
        initial_state_distribution: Dict[str, float],
        goal_states: frozenset[str],
        config: MCConfig,
    ) -> TrajectoryBundle:
        """Execute parallel Monte Carlo sampling.

        Distributes the requested *num_samples* across
        *config.parallel_workers* workers and aggregates results.

        Parameters:
            sampler: The trajectory sampler to use on each worker.
            transition_model: T(s'|s,a).
            cost_model: C(s,a).
            policy: π(a|s).
            initial_state_distribution: P(s₀).
            goal_states: Set of absorbing goal states.
            config: Sampling configuration (parallel_workers > 1).

        Returns:
            Merged :class:`TrajectoryBundle` combining all workers' output.
        """
        ...

    @property
    def num_workers(self) -> int:
        """Number of active parallel workers."""
        ...
