"""
usability_oracle.bisimulation.approximate — Approximate bisimulation.

Implements ε-bisimulation with explicit error bounds, approximate partition
refinement, and adaptive refinement strategies that trade off abstraction
quality against state count.

Key capabilities:
  - ε-bisimulation with error bounds
  - Approximate partition refinement
  - Error propagation through abstraction
  - Trade-off analysis: abstraction quality vs. state count
  - Iterative refinement with error budget
  - Value-function error bounds under approximation
  - Adaptive refinement based on policy sensitivity

References
----------
- Ferns, N., Panangaden, P. & Precup, D. (2004). Metrics for finite
  Markov decision processes. *UAI*.
- Givan, R., Dean, T. & Greig, M. (2003). Equivalence notions and
  model minimization in MDPs. *Artificial Intelligence*.
- Abel, D. et al. (2016). Near optimal behavior via approximate state
  abstraction. *ICML*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from usability_oracle.bisimulation.cognitive_distance import (
    CognitiveDistanceComputer,
    _soft_value_iteration,
)
from usability_oracle.bisimulation.models import (
    BisimulationResult,
    CognitiveDistanceMatrix,
    Partition,
)
from usability_oracle.bisimulation.probabilistic import (
    ProbabilisticBisimulationMetric,
    kantorovich_distance,
)
from usability_oracle.mdp.models import MDP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ε-Bisimulation
# ---------------------------------------------------------------------------

@dataclass
class EpsilonBisimulation:
    """ε-bisimulation with explicit error bounds.

    Two states are ε-bisimilar if for every action a:
      1. |c(s₁, a) − c(s₂, a)| ≤ ε_r   (reward closeness)
      2. W_d(T(·|s₁,a), T(·|s₂,a)) ≤ ε_t   (transition closeness)

    where W_d is the Kantorovich distance under the bisimulation metric
    and ε = ε_r + γ · ε_t.

    Parameters
    ----------
    epsilon : float
        Total error tolerance.
    discount : float
        Discount factor.
    max_metric_iterations : int
        Maximum iterations for metric computation.
    """

    epsilon: float = 0.05
    discount: float = 0.99
    max_metric_iterations: int = 200

    def compute(self, mdp: MDP) -> tuple[Partition, float]:
        """Compute ε-bisimulation partition with error bound.

        Parameters
        ----------
        mdp : MDP

        Returns
        -------
        tuple[Partition, float]
            (partition, actual_error) where actual_error ≤ epsilon.
        """
        metric_computer = ProbabilisticBisimulationMetric(
            discount=self.discount,
            max_iterations=self.max_metric_iterations,
        )
        dm = metric_computer.compute(mdp)
        partition = dm.threshold_partition(self.epsilon)

        # Compute actual max within-block distance
        actual_error = self._max_within_block_distance(partition, dm)

        logger.info(
            "ε-bisimulation: ε=%.4f, actual_error=%.4f, %d blocks",
            self.epsilon, actual_error, partition.n_blocks,
        )

        return partition, actual_error

    def value_function_error_bound(
        self,
        epsilon: float,
        discount: float,
    ) -> float:
        """Compute the bound on value function error due to ε-bisimulation.

        For an ε-bisimulation, the value function error is bounded by:

            |V(s) - V_abs(block(s))| ≤ ε / (1 − γ)

        Parameters
        ----------
        epsilon : float
            Bisimulation approximation error.
        discount : float
            Discount factor γ.

        Returns
        -------
        float
            Upper bound on value function error.
        """
        if discount >= 1.0:
            return float("inf")
        return epsilon / (1.0 - discount)

    def _max_within_block_distance(
        self,
        partition: Partition,
        dm: CognitiveDistanceMatrix,
    ) -> float:
        """Compute the maximum pairwise distance within any block."""
        max_d = 0.0
        for block in partition.blocks:
            states = sorted(block)
            for i in range(len(states)):
                for j in range(i + 1, len(states)):
                    d = dm.distance(states[i], states[j])
                    max_d = max(max_d, d)
        return max_d


# ---------------------------------------------------------------------------
# Approximate partition refinement
# ---------------------------------------------------------------------------

@dataclass
class ApproximatePartitionRefinement:
    """Partition refinement with controlled approximation error.

    Unlike exact refinement, allows blocks to remain merged if splitting
    would only reduce the within-block error by a small amount.  This
    produces coarser (smaller) partitions at the cost of bounded error.

    Parameters
    ----------
    epsilon : float
        Maximum allowed within-block policy divergence.
    max_iterations : int
        Maximum refinement rounds.
    min_improvement : float
        Minimum error reduction required to justify a split.
    """

    epsilon: float = 0.05
    max_iterations: int = 500
    min_improvement: float = 0.01

    def refine(
        self,
        mdp: MDP,
        beta: float,
        initial_partition: Optional[Partition] = None,
    ) -> tuple[Partition, list[float]]:
        """Run approximate partition refinement.

        Parameters
        ----------
        mdp : MDP
        beta : float
            Rationality parameter.
        initial_partition : Partition or None
            Starting partition. If None, uses the trivial partition.

        Returns
        -------
        tuple[Partition, list[float]]
            (partition, error_history) where error_history tracks the
            max within-block error at each iteration.
        """
        state_ids = sorted(mdp.states.keys())
        if initial_partition is None:
            partition = Partition.trivial(state_ids)
        else:
            partition = initial_partition

        values = _soft_value_iteration(mdp, beta)
        cdc = CognitiveDistanceComputer(n_grid=1, refine=False)
        error_history: list[float] = []

        for iteration in range(self.max_iterations):
            max_error = 0.0
            new_blocks: list[frozenset[str]] = []

            for block in partition.blocks:
                if len(block) <= 1:
                    new_blocks.append(block)
                    continue

                block_error = self._block_error(
                    block, mdp, beta, values, cdc, partition,
                )
                max_error = max(max_error, block_error)

                if block_error > self.epsilon:
                    # Split this block
                    sub_blocks = self._split_block(
                        block, mdp, beta, values, cdc, partition,
                    )
                    # Only accept split if it meaningfully reduces error
                    sub_error = max(
                        self._block_error(sb, mdp, beta, values, cdc, partition)
                        for sb in sub_blocks
                    )
                    if block_error - sub_error >= self.min_improvement:
                        new_blocks.extend(sub_blocks)
                    else:
                        new_blocks.append(block)
                else:
                    new_blocks.append(block)

            error_history.append(max_error)
            new_partition = Partition.from_blocks(new_blocks)

            if set(new_partition.blocks) == set(partition.blocks):
                logger.info(
                    "Approximate refinement converged: %d iters, %d blocks, "
                    "max_error=%.4f",
                    iteration + 1, new_partition.n_blocks, max_error,
                )
                return new_partition, error_history

            partition = new_partition

        logger.warning(
            "Approximate refinement did not converge after %d iterations",
            self.max_iterations,
        )
        return partition, error_history

    def _block_error(
        self,
        block: frozenset[str],
        mdp: MDP,
        beta: float,
        values: dict[str, float],
        cdc: CognitiveDistanceComputer,
        partition: Partition,
    ) -> float:
        """Compute the maximum policy divergence within a block."""
        if len(block) <= 1:
            return 0.0

        states = sorted(block)
        all_actions: set[str] = set()
        for s in states:
            all_actions.update(mdp.get_actions(s))
        action_order = sorted(all_actions)

        if not action_order:
            return 0.0

        policies = []
        for s in states:
            pi = cdc._policy_at_state_ordered(
                mdp, s, beta, values, action_order,
            )
            policies.append(pi)

        max_tv = 0.0
        for i in range(len(policies)):
            for j in range(i + 1, len(policies)):
                tv = 0.5 * float(np.sum(np.abs(policies[i] - policies[j])))
                max_tv = max(max_tv, tv)

        return max_tv

    def _split_block(
        self,
        block: frozenset[str],
        mdp: MDP,
        beta: float,
        values: dict[str, float],
        cdc: CognitiveDistanceComputer,
        partition: Partition,
    ) -> list[frozenset[str]]:
        """Split a block into two sub-blocks."""
        states = sorted(block)
        n = len(states)

        if n <= 1:
            return [block]

        all_actions: set[str] = set()
        for s in states:
            all_actions.update(mdp.get_actions(s))
        action_order = sorted(all_actions)

        policies = []
        for s in states:
            pi = cdc._policy_at_state_ordered(
                mdp, s, beta, values, action_order,
            )
            policies.append(pi)

        # Distance matrix within block
        dist = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d = 0.5 * float(np.sum(np.abs(policies[i] - policies[j])))
                dist[i, j] = d
                dist[j, i] = d

        # Split by most distant pair
        if n == 2:
            return [frozenset([states[0]]), frozenset([states[1]])]

        flat_idx = int(np.argmax(dist))
        si, sj = divmod(flat_idx, n)

        group_a: set[str] = set()
        group_b: set[str] = set()

        for k, state in enumerate(states):
            if dist[k, si] <= dist[k, sj]:
                group_a.add(state)
            else:
                group_b.add(state)

        result = []
        if group_a:
            result.append(frozenset(group_a))
        if group_b:
            result.append(frozenset(group_b))

        if len(result) < 2:
            mid = n // 2
            return [frozenset(states[:mid]), frozenset(states[mid:])]

        return result


# ---------------------------------------------------------------------------
# Error propagation analysis
# ---------------------------------------------------------------------------

@dataclass
class ErrorPropagation:
    """Analyse how abstraction error propagates through the MDP.

    Computes bounds on how local abstraction errors (within-block
    distances) compound across multiple time steps.

    Parameters
    ----------
    horizon : int
        Planning horizon for error propagation.
    """

    horizon: int = 50

    def propagation_bound(
        self,
        local_error: float,
        discount: float,
        horizon: Optional[int] = None,
    ) -> float:
        """Compute the cumulative error bound over a horizon.

        The total value-function error is bounded by:

            Σ_{t=0}^{H} γ^t · ε_local = ε_local · (1 − γ^{H+1}) / (1 − γ)

        Parameters
        ----------
        local_error : float
            Maximum single-step abstraction error.
        discount : float
            Discount factor.
        horizon : int or None
            Planning horizon. If None, uses self.horizon.

        Returns
        -------
        float
            Cumulative error bound.
        """
        H = horizon if horizon is not None else self.horizon
        if discount >= 1.0 - 1e-12:
            return local_error * (H + 1)
        return local_error * (1.0 - discount ** (H + 1)) / (1.0 - discount)

    def per_block_error(
        self,
        partition: Partition,
        mdp: MDP,
        beta: float,
    ) -> dict[int, float]:
        """Compute abstraction error per block.

        Parameters
        ----------
        partition : Partition
        mdp : MDP
        beta : float

        Returns
        -------
        dict[int, float]
            Maps block_index → within-block value range.
        """
        values = _soft_value_iteration(mdp, beta)
        errors: dict[int, float] = {}

        for idx, block in enumerate(partition.blocks):
            if len(block) <= 1:
                errors[idx] = 0.0
                continue
            block_vals = [values.get(s, 0.0) for s in block]
            errors[idx] = max(block_vals) - min(block_vals)

        return errors


# ---------------------------------------------------------------------------
# Adaptive refinement
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveRefinement:
    """Adaptive refinement based on policy sensitivity.

    Preferentially refines blocks where the policy is most sensitive to
    state differences, allocating the error budget to blocks where it
    matters most.

    Parameters
    ----------
    total_error_budget : float
        Total allowed abstraction error.
    max_blocks : int
        Maximum number of blocks in the partition.
    max_iterations : int
        Maximum refinement iterations.
    """

    total_error_budget: float = 0.1
    max_blocks: int = 200
    max_iterations: int = 100

    def refine(
        self,
        mdp: MDP,
        beta: float,
        initial_partition: Optional[Partition] = None,
    ) -> BisimulationResult:
        """Run adaptive refinement.

        Iteratively splits the block with the highest error, stopping
        when the error budget is satisfied or the maximum block count
        is reached.

        Parameters
        ----------
        mdp : MDP
        beta : float
        initial_partition : Partition or None

        Returns
        -------
        BisimulationResult
        """
        state_ids = sorted(mdp.states.keys())
        if initial_partition is None:
            partition = Partition.trivial(state_ids)
        else:
            partition = initial_partition

        values = _soft_value_iteration(mdp, beta)
        cdc = CognitiveDistanceComputer(n_grid=1, refine=False)
        refinement_history: list[int] = [partition.n_blocks]

        approx = ApproximatePartitionRefinement(epsilon=0.0)

        for iteration in range(self.max_iterations):
            if partition.n_blocks >= self.max_blocks:
                logger.info("Reached max blocks (%d)", self.max_blocks)
                break

            # Find block with highest error
            block_errors: list[tuple[int, float]] = []
            for idx, block in enumerate(partition.blocks):
                err = approx._block_error(
                    block, mdp, beta, values, cdc, partition,
                )
                block_errors.append((idx, err))

            block_errors.sort(key=lambda x: -x[1])
            worst_idx, worst_error = block_errors[0]

            if worst_error <= self.total_error_budget:
                logger.info(
                    "Error budget satisfied: max_error=%.4f ≤ %.4f",
                    worst_error, self.total_error_budget,
                )
                break

            # Split the worst block
            worst_block = partition.blocks[worst_idx]
            if len(worst_block) <= 1:
                break

            sub_blocks = approx._split_block(
                worst_block, mdp, beta, values, cdc, partition,
            )

            new_blocks = list(partition.blocks)
            new_blocks[worst_idx] = sub_blocks[0]
            if len(sub_blocks) > 1:
                new_blocks.extend(sub_blocks[1:])
            partition = Partition.from_blocks(new_blocks)
            refinement_history.append(partition.n_blocks)

        # Compute final error
        max_error = max(
            approx._block_error(b, mdp, beta, values, cdc, partition)
            for b in partition.blocks
        ) if partition.blocks else 0.0

        # Build quotient MDP
        from usability_oracle.bisimulation.quotient import QuotientMDPBuilder
        builder = QuotientMDPBuilder(verify=False)
        quotient = builder.build(mdp, partition)

        return BisimulationResult(
            partition=partition,
            quotient_mdp=quotient,
            abstraction_error=max_error,
            beta_used=beta,
            iterations=len(refinement_history),
            refinement_history=refinement_history,
            metadata={
                "method": "adaptive",
                "total_error_budget": self.total_error_budget,
                "max_blocks": self.max_blocks,
            },
        )


# ---------------------------------------------------------------------------
# Quality-vs-size trade-off analysis
# ---------------------------------------------------------------------------

@dataclass
class AbstractionTradeoff:
    """Analyse the trade-off between abstraction quality and state count.

    Sweeps over a range of ε values and reports the Pareto frontier of
    (n_blocks, error) pairs.

    Parameters
    ----------
    epsilon_range : tuple[float, float]
        Range of ε values to sweep.
    n_points : int
        Number of points in the sweep.
    """

    epsilon_range: tuple[float, float] = (0.01, 0.5)
    n_points: int = 20

    def analyse(
        self,
        mdp: MDP,
        beta: float,
    ) -> list[tuple[int, float, float]]:
        """Compute the Pareto frontier.

        Parameters
        ----------
        mdp : MDP
        beta : float

        Returns
        -------
        list[tuple[int, float, float]]
            List of (n_blocks, abstraction_error, epsilon) sorted by n_blocks.
        """
        epsilons = np.linspace(
            self.epsilon_range[0], self.epsilon_range[1], self.n_points,
        )

        results: list[tuple[int, float, float]] = []

        metric_computer = ProbabilisticBisimulationMetric(
            discount=mdp.discount,
            max_iterations=100,
        )
        dm = metric_computer.compute(mdp)

        for eps in epsilons:
            partition = dm.threshold_partition(float(eps))

            # Compute actual error
            max_d = 0.0
            for block in partition.blocks:
                states = sorted(block)
                for i in range(len(states)):
                    for j in range(i + 1, len(states)):
                        d = dm.distance(states[i], states[j])
                        max_d = max(max_d, d)

            results.append((partition.n_blocks, max_d, float(eps)))

        # Sort by n_blocks ascending
        results.sort(key=lambda x: x[0])

        # Filter to Pareto frontier
        pareto: list[tuple[int, float, float]] = []
        min_error = float("inf")
        for n_blocks, error, eps in results:
            if error < min_error:
                pareto.append((n_blocks, error, eps))
                min_error = error

        logger.info(
            "Abstraction trade-off: %d Pareto-optimal points from %d sweeps",
            len(pareto), len(results),
        )

        return pareto
