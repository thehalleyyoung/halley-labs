"""
usability_oracle.bisimulation.probabilistic — Probabilistic bisimulation.

Implements the probabilistic bisimulation equivalence of Larsen & Skou and
the quantitative bisimulation metric of Desharnais et al.  The metric is
computed via fixed-point iteration over the Kantorovich (Wasserstein-1)
distance between transition distributions.

Key algorithms:
  - Larsen-Skou probabilistic bisimulation equivalence
  - Desharnais et al. bisimulation metric via Bellman-style fixed point
  - Kantorovich distance via linear programming and coupling
  - ε-approximate bisimulation
  - Bounded-rational extension: capacity-weighted metric

References
----------
- Larsen, K. G. & Skou, A. (1991). Bisimulation through probabilistic
  testing. *Information and Computation* 94, 1–28.
- Desharnais, J., Gupta, V., Jagadeesan, R. & Panangaden, P. (2004).
  Metrics for labelled Markov processes. *TCS* 318, 323–354.
- Ferns, N., Panangaden, P. & Precup, D. (2004). Metrics for finite
  Markov decision processes. *UAI*.
- Villani, C. (2009). *Optimal Transport*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import linprog  # type: ignore[import-untyped]

from usability_oracle.bisimulation.cognitive_distance import (
    CognitiveDistanceComputer,
    _soft_value_iteration,
)
from usability_oracle.bisimulation.models import (
    CognitiveDistanceMatrix,
    Partition,
)
from usability_oracle.mdp.models import MDP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kantorovich (Wasserstein-1) distance
# ---------------------------------------------------------------------------

def kantorovich_distance(
    p: np.ndarray,
    q: np.ndarray,
    ground_metric: np.ndarray,
) -> float:
    """Compute the Kantorovich (Wasserstein-1) distance between distributions.

    Solves the optimal transport problem:

        W_1(p, q) = min_{γ ∈ Γ(p,q)} Σ_{i,j} γ_{ij} d(i, j)

    using a linear programming formulation.

    Parameters
    ----------
    p : np.ndarray
        Source distribution (n,).
    q : np.ndarray
        Target distribution (n,).
    ground_metric : np.ndarray
        Pairwise distance matrix d(i, j), shape (n, n).

    Returns
    -------
    float
        Kantorovich distance ≥ 0.
    """
    n = len(p)
    if n == 0:
        return 0.0

    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Normalise to avoid numerical issues
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum > 0:
        p = p / p_sum
    if q_sum > 0:
        q = q / q_sum

    # LP variables: γ_{ij} for i,j in [n]×[n], flattened row-major
    n_vars = n * n
    c = ground_metric.ravel().astype(np.float64)

    # Equality constraints: row sums = p, column sums = q
    # Row sum constraints: Σ_j γ_{ij} = p_i for each i
    A_eq = np.zeros((2 * n, n_vars), dtype=np.float64)
    b_eq = np.zeros(2 * n, dtype=np.float64)

    for i in range(n):
        A_eq[i, i * n : (i + 1) * n] = 1.0
        b_eq[i] = p[i]

    # Column sum constraints: Σ_i γ_{ij} = q_j for each j
    for j in range(n):
        for i in range(n):
            A_eq[n + j, i * n + j] = 1.0
        b_eq[n + j] = q[j]

    bounds = [(0.0, None)] * n_vars

    try:
        result = linprog(
            c, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
            method="highs", options={"presolve": True},
        )
        if result.success:
            return max(0.0, float(result.fun))
    except Exception:
        pass

    # Fallback: use the closed-form for 1-D Wasserstein (sort-based)
    return float(np.sum(np.abs(np.cumsum(p) - np.cumsum(q))))


def kantorovich_coupling(
    p: np.ndarray,
    q: np.ndarray,
    ground_metric: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Compute the Kantorovich distance and optimal coupling.

    Parameters
    ----------
    p, q : np.ndarray
        Probability distributions (n,).
    ground_metric : np.ndarray
        Pairwise distance matrix (n, n).

    Returns
    -------
    tuple[float, np.ndarray]
        (distance, coupling_matrix) where coupling_matrix is (n, n).
    """
    n = len(p)
    if n == 0:
        return 0.0, np.empty((0, 0))

    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum > 0:
        p = p / p_sum
    if q_sum > 0:
        q = q / q_sum

    n_vars = n * n
    c = ground_metric.ravel().astype(np.float64)

    A_eq = np.zeros((2 * n, n_vars), dtype=np.float64)
    b_eq = np.zeros(2 * n, dtype=np.float64)

    for i in range(n):
        A_eq[i, i * n : (i + 1) * n] = 1.0
        b_eq[i] = p[i]
    for j in range(n):
        for i in range(n):
            A_eq[n + j, i * n + j] = 1.0
        b_eq[n + j] = q[j]

    bounds = [(0.0, None)] * n_vars

    try:
        result = linprog(
            c, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
            method="highs", options={"presolve": True},
        )
        if result.success:
            coupling = result.x.reshape(n, n)
            return max(0.0, float(result.fun)), coupling
    except Exception:
        pass

    # Fallback: independent coupling
    coupling = np.outer(p, q)
    dist = float(np.sum(coupling * ground_metric))
    return dist, coupling


# ---------------------------------------------------------------------------
# Probabilistic bisimulation equivalence (Larsen-Skou)
# ---------------------------------------------------------------------------

@dataclass
class LarsenSkouBisimulation:
    """Probabilistic bisimulation equivalence via Larsen-Skou.

    Two states s₁, s₂ are probabilistically bisimilar iff for every
    equivalence class C of the bisimulation, and every action a:

        Σ_{s' ∈ C} T(s'|s₁, a) = Σ_{s' ∈ C} T(s'|s₂, a)

    The algorithm iteratively refines a partition until stable.

    Parameters
    ----------
    max_iterations : int
        Maximum refinement rounds.
    tolerance : float
        Numerical tolerance for probability comparison.
    """

    max_iterations: int = 500
    tolerance: float = 1e-10

    def compute(self, mdp: MDP) -> Partition:
        """Compute the coarsest probabilistic bisimulation partition.

        Parameters
        ----------
        mdp : MDP

        Returns
        -------
        Partition
            The coarsest partition such that all states within a block are
            probabilistically bisimilar.
        """
        state_ids = sorted(mdp.states.keys())
        if not state_ids:
            return Partition()

        # Initial partition: group by (action_set, is_goal, is_terminal)
        partition = self._initial_partition(mdp, state_ids)

        for iteration in range(self.max_iterations):
            new_partition = self._refine(partition, mdp, state_ids)
            if set(new_partition.blocks) == set(partition.blocks):
                logger.info(
                    "Larsen-Skou converged after %d iterations: %d blocks",
                    iteration + 1, new_partition.n_blocks,
                )
                return new_partition
            partition = new_partition

        logger.warning(
            "Larsen-Skou did not converge after %d iterations (%d blocks)",
            self.max_iterations, partition.n_blocks,
        )
        return partition

    def _initial_partition(
        self, mdp: MDP, state_ids: list[str]
    ) -> Partition:
        """Create initial partition grouping by action set and status."""
        from collections import defaultdict

        groups: dict[tuple, set[str]] = defaultdict(set)
        for sid in state_ids:
            state = mdp.states[sid]
            status = ("goal" if state.is_goal else
                      "terminal" if state.is_terminal else "active")
            actions = tuple(sorted(mdp.get_actions(sid)))
            groups[(status, actions)].add(sid)
        blocks = [frozenset(g) for g in groups.values()]
        return Partition.from_blocks(blocks)

    def _refine(
        self,
        partition: Partition,
        mdp: MDP,
        state_ids: list[str],
    ) -> Partition:
        """One refinement pass: split blocks with different transition signatures."""
        new_blocks: list[frozenset[str]] = []

        for block in partition.blocks:
            if len(block) <= 1:
                new_blocks.append(block)
                continue

            sub = self._split_block(block, partition, mdp)
            new_blocks.extend(sub)

        return Partition.from_blocks(new_blocks)

    def _split_block(
        self,
        block: frozenset[str],
        partition: Partition,
        mdp: MDP,
    ) -> list[frozenset[str]]:
        """Split a block by probabilistic transition signatures."""
        signatures: dict[str, tuple] = {}

        for sid in block:
            sig = self._transition_signature(sid, partition, mdp)
            signatures[sid] = sig

        groups: dict[tuple, set[str]] = {}
        for sid, sig in signatures.items():
            # Discretise floats for grouping
            rounded = tuple(
                (k, round(v, 10)) for k, v in sorted(sig)
            )
            groups.setdefault(rounded, set()).add(sid)

        return [frozenset(g) for g in groups.values()]

    def _transition_signature(
        self,
        state: str,
        partition: Partition,
        mdp: MDP,
    ) -> tuple[tuple[tuple[str, int], float], ...]:
        """Compute the signature: (action, block_idx) → total probability."""
        sig: dict[tuple[str, int], float] = {}
        for aid in mdp.get_actions(state):
            for target, prob, _ in mdp.get_transitions(state, aid):
                bi = partition.state_to_block.get(target, -1)
                key = (aid, bi)
                sig[key] = sig.get(key, 0.0) + prob
        return tuple(sorted(sig.items()))


# ---------------------------------------------------------------------------
# Probabilistic bisimulation metric (Desharnais et al.)
# ---------------------------------------------------------------------------

@dataclass
class ProbabilisticBisimulationMetric:
    """Quantitative bisimulation metric via fixed-point iteration.

    Computes the metric d* that is the least fixed point of the operator:

        (Fd)(s₁, s₂) = max_a [ c_diff(s₁,s₂,a)
                                 + γ · W_d(T(·|s₁,a), T(·|s₂,a)) ]

    where c_diff is the reward/cost difference and W_d is the Kantorovich
    distance with ground metric d.

    Parameters
    ----------
    discount : float
        Discount factor γ for the contraction mapping.
    max_iterations : int
        Maximum number of fixed-point iterations.
    tolerance : float
        Convergence tolerance on max |d_new - d_old|.
    use_lp : bool
        If True, use LP for Kantorovich; otherwise use Sinkhorn approximation.
    """

    discount: float = 0.99
    max_iterations: int = 200
    tolerance: float = 1e-6
    use_lp: bool = True

    def compute(self, mdp: MDP) -> CognitiveDistanceMatrix:
        """Compute the bisimulation metric for all state pairs.

        Parameters
        ----------
        mdp : MDP

        Returns
        -------
        CognitiveDistanceMatrix
            Symmetric matrix of pairwise bisimulation distances.
        """
        state_ids = sorted(mdp.states.keys())
        n = len(state_ids)
        state_idx = {sid: i for i, sid in enumerate(state_ids)}

        # Initialise metric: d(s, s) = 0, d(s, t) = 1 for s != t
        d = np.ones((n, n), dtype=np.float64)
        np.fill_diagonal(d, 0.0)

        gamma = self.discount
        convergence_history: list[float] = []

        for iteration in range(self.max_iterations):
            d_new = np.zeros_like(d)

            for i in range(n):
                for j in range(i + 1, n):
                    s1, s2 = state_ids[i], state_ids[j]
                    val = self._bellman_update(
                        s1, s2, mdp, d, state_idx, gamma,
                    )
                    d_new[i, j] = val
                    d_new[j, i] = val

            delta = float(np.max(np.abs(d_new - d)))
            convergence_history.append(delta)
            d = d_new

            if delta < self.tolerance:
                logger.info(
                    "Bisimulation metric converged after %d iterations "
                    "(Δ=%.2e)", iteration + 1, delta,
                )
                break
        else:
            logger.warning(
                "Bisimulation metric did not converge after %d iterations "
                "(Δ=%.2e)", self.max_iterations, convergence_history[-1],
            )

        return CognitiveDistanceMatrix(distances=d, state_ids=state_ids)

    def _bellman_update(
        self,
        s1: str,
        s2: str,
        mdp: MDP,
        d: np.ndarray,
        state_idx: dict[str, int],
        gamma: float,
    ) -> float:
        """Compute one step of the Bellman operator for d(s1, s2)."""
        actions_1 = set(mdp.get_actions(s1))
        actions_2 = set(mdp.get_actions(s2))
        common_actions = actions_1 & actions_2

        if not common_actions:
            # Different action availability
            return 1.0

        max_dist = 0.0
        for aid in common_actions:
            # Cost difference
            cost_diff = self._cost_difference(s1, s2, aid, mdp)

            # Build transition distributions over state index space
            n = d.shape[0]
            p = np.zeros(n, dtype=np.float64)
            q = np.zeros(n, dtype=np.float64)

            for target, prob, _ in mdp.get_transitions(s1, aid):
                idx = state_idx.get(target)
                if idx is not None:
                    p[idx] += prob
            for target, prob, _ in mdp.get_transitions(s2, aid):
                idx = state_idx.get(target)
                if idx is not None:
                    q[idx] += prob

            # Kantorovich distance with ground metric d
            if self.use_lp:
                w = kantorovich_distance(p, q, d)
            else:
                w = self._sinkhorn_distance(p, q, d)

            dist = cost_diff + gamma * w
            max_dist = max(max_dist, dist)

        return min(max_dist, 1.0)

    def _cost_difference(
        self, s1: str, s2: str, action: str, mdp: MDP,
    ) -> float:
        """Compute |E[c(s1,a)] - E[c(s2,a)]|."""
        c1 = sum(p * c for _, p, c in mdp.get_transitions(s1, action))
        c2 = sum(p * c for _, p, c in mdp.get_transitions(s2, action))
        return abs(c1 - c2)

    def _sinkhorn_distance(
        self,
        p: np.ndarray,
        q: np.ndarray,
        ground_metric: np.ndarray,
        reg: float = 0.1,
        max_iter: int = 100,
    ) -> float:
        """Sinkhorn approximation of the Kantorovich distance.

        Uses entropic regularisation for faster computation on large
        state spaces.

        Parameters
        ----------
        p, q : np.ndarray
            Distributions.
        ground_metric : np.ndarray
            Cost matrix.
        reg : float
            Regularisation parameter.
        max_iter : int
            Maximum Sinkhorn iterations.

        Returns
        -------
        float
        """
        n = len(p)
        if n == 0:
            return 0.0

        p = p.copy()
        q = q.copy()
        p_sum = p.sum()
        q_sum = q.sum()
        if p_sum > 0:
            p /= p_sum
        else:
            p = np.ones(n) / n
        if q_sum > 0:
            q /= q_sum
        else:
            q = np.ones(n) / n

        K = np.exp(-ground_metric / max(reg, 1e-10))
        K = np.maximum(K, 1e-300)

        u = np.ones(n, dtype=np.float64)
        for _ in range(max_iter):
            v = q / (K.T @ u + 1e-300)
            u = p / (K @ v + 1e-300)

        coupling = np.diag(u) @ K @ np.diag(v)
        return float(np.sum(coupling * ground_metric))

    def convergence_rate(self, gamma: float) -> float:
        """Return the theoretical convergence rate of the fixed-point iteration.

        The operator is a γ-contraction in the sup-norm, so after k iterations
        the error is bounded by γ^k · d_max where d_max is the initial metric
        diameter.

        Parameters
        ----------
        gamma : float
            Discount factor.

        Returns
        -------
        float
            Contraction factor per iteration.
        """
        return gamma


# ---------------------------------------------------------------------------
# ε-Approximate probabilistic bisimulation
# ---------------------------------------------------------------------------

@dataclass
class ApproximateProbabilisticBisimulation:
    """ε-approximate probabilistic bisimulation.

    Two states are ε-bisimilar if their transition distributions differ by
    at most ε in Kantorovich distance (under the bisimulation metric) for
    every action.

    Parameters
    ----------
    epsilon : float
        Approximation tolerance.
    max_iterations : int
        Maximum fixed-point iterations for metric computation.
    """

    epsilon: float = 0.05
    max_iterations: int = 200

    def compute(self, mdp: MDP) -> Partition:
        """Compute the ε-approximate bisimulation partition.

        States with bisimulation distance ≤ ε are grouped together.

        Parameters
        ----------
        mdp : MDP

        Returns
        -------
        Partition
        """
        metric_computer = ProbabilisticBisimulationMetric(
            discount=mdp.discount,
            max_iterations=self.max_iterations,
        )
        distance_matrix = metric_computer.compute(mdp)
        return distance_matrix.threshold_partition(self.epsilon)


# ---------------------------------------------------------------------------
# Capacity-weighted probabilistic bisimulation metric
# ---------------------------------------------------------------------------

@dataclass
class CapacityWeightedMetric:
    """Bounded-rational extension of the probabilistic bisimulation metric.

    Weights the bisimulation metric by the bounded-rational policy at
    rationality level β.  Actions that the agent is unlikely to take under
    bounded rationality contribute less to the distance:

        d_cw(s₁, s₂) = Σ_a  ½(π_β(a|s₁) + π_β(a|s₂))
                         · [|c(s₁,a) - c(s₂,a)| + γ W_d(T(·|s₁,a), T(·|s₂,a))]

    Parameters
    ----------
    beta : float
        Rationality parameter.
    discount : float
        Discount factor.
    max_iterations : int
        Maximum fixed-point iterations.
    tolerance : float
        Convergence tolerance.
    """

    beta: float = 1.0
    discount: float = 0.99
    max_iterations: int = 200
    tolerance: float = 1e-6

    def compute(self, mdp: MDP) -> CognitiveDistanceMatrix:
        """Compute the capacity-weighted bisimulation metric.

        Parameters
        ----------
        mdp : MDP

        Returns
        -------
        CognitiveDistanceMatrix
        """
        state_ids = sorted(mdp.states.keys())
        n = len(state_ids)
        state_idx = {sid: i for i, sid in enumerate(state_ids)}

        # Pre-compute soft values and policies
        values = _soft_value_iteration(mdp, self.beta)
        cdc = CognitiveDistanceComputer(n_grid=1, refine=False)

        d = np.ones((n, n), dtype=np.float64)
        np.fill_diagonal(d, 0.0)

        gamma = self.discount

        for iteration in range(self.max_iterations):
            d_new = np.zeros_like(d)

            for i in range(n):
                for j in range(i + 1, n):
                    s1, s2 = state_ids[i], state_ids[j]
                    val = self._weighted_update(
                        s1, s2, mdp, d, state_idx, gamma, values, cdc,
                    )
                    d_new[i, j] = val
                    d_new[j, i] = val

            delta = float(np.max(np.abs(d_new - d)))
            d = d_new

            if delta < self.tolerance:
                logger.info(
                    "Capacity-weighted metric converged after %d iters "
                    "(Δ=%.2e)", iteration + 1, delta,
                )
                break

        return CognitiveDistanceMatrix(distances=d, state_ids=state_ids)

    def _weighted_update(
        self,
        s1: str,
        s2: str,
        mdp: MDP,
        d: np.ndarray,
        state_idx: dict[str, int],
        gamma: float,
        values: dict[str, float],
        cdc: CognitiveDistanceComputer,
    ) -> float:
        """Compute one capacity-weighted Bellman update."""
        actions_1 = mdp.get_actions(s1)
        actions_2 = mdp.get_actions(s2)
        all_actions = sorted(set(actions_1) | set(actions_2))

        if not all_actions:
            return 0.0

        # Get policies
        pi1 = cdc._policy_at_state_ordered(mdp, s1, self.beta, values, all_actions)
        pi2 = cdc._policy_at_state_ordered(mdp, s2, self.beta, values, all_actions)

        n = d.shape[0]
        total = 0.0

        for k, aid in enumerate(all_actions):
            weight = 0.5 * (pi1[k] + pi2[k])
            if weight < 1e-12:
                continue

            # Cost difference
            c1 = sum(p * c for _, p, c in mdp.get_transitions(s1, aid))
            c2 = sum(p * c for _, p, c in mdp.get_transitions(s2, aid))
            cost_diff = abs(c1 - c2)

            # Transition distributions
            p_dist = np.zeros(n, dtype=np.float64)
            q_dist = np.zeros(n, dtype=np.float64)
            for target, prob, _ in mdp.get_transitions(s1, aid):
                idx = state_idx.get(target)
                if idx is not None:
                    p_dist[idx] += prob
            for target, prob, _ in mdp.get_transitions(s2, aid):
                idx = state_idx.get(target)
                if idx is not None:
                    q_dist[idx] += prob

            w = kantorovich_distance(p_dist, q_dist, d)
            total += weight * (cost_diff + gamma * w)

        return min(total, 1.0)
