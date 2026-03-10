"""
usability_oracle.policy.multi_objective — Multi-objective policy computation.

Computes policies that trade off multiple objectives simultaneously, such as
task completion time, error rate, and cognitive load.  Provides:

- **Pareto-optimal policies** via enumeration over the weight simplex.
- **Scalarisation methods** — weighted sum and Chebyshev (augmented
  Tchebycheff).
- **Constrained MDP** via Lagrangian relaxation.
- **Lexicographic optimisation** — strict priority ordering of objectives.
- **Multi-objective value iteration** — vector-valued Bellman backup.
- **Pareto frontier visualisation** data.

The bounded-rational free-energy formulation naturally extends to multiple
objectives:

    F(π) = Σ_k w_k E_π[c_k] + (1/β) D_KL(π ‖ p₀)

where c_k are the per-objective cost functions and w_k are scalarisation
weights.

References
----------
- Roijers, D. M. et al. (2013). A survey of multi-objective sequential
  decision-making. *JAIR*, 48, 67–113.
- Altman, E. (1999). *Constrained Markov Decision Processes*. Chapman & Hall.
- Wierzbicki, A. P. (1982). A mathematical basis for satisficing decision
  making. *Mathematical Modelling*, 3(5), 391–405.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from usability_oracle.policy.models import Policy, QValues

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MultiObjectiveQValues:
    """Vector-valued Q-function Q(s, a) ∈ ℝ^K for K objectives.

    Attributes
    ----------
    values : dict[str, dict[str, np.ndarray]]
        ``values[state][action]`` = K-dimensional cost vector.
    n_objectives : int
    objective_names : list[str]
    """

    values: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    n_objectives: int = 1
    objective_names: list[str] = field(default_factory=list)

    def scalarise(self, weights: np.ndarray) -> QValues:
        """Convert to scalar Q-values via weighted sum.

        Q_scalar(s, a) = w^T · Q_vec(s, a)

        Parameters
        ----------
        weights : np.ndarray
            Weight vector ∈ ℝ^K.

        Returns
        -------
        QValues
        """
        scalar: dict[str, dict[str, float]] = {}
        for s, actions in self.values.items():
            scalar[s] = {}
            for a, q_vec in actions.items():
                scalar[s][a] = float(np.dot(weights, q_vec))
        return QValues(values=scalar)


@dataclass
class ParetoPoint:
    """A point on the Pareto frontier.

    Attributes
    ----------
    objectives : np.ndarray
        Objective values (costs).
    weights : np.ndarray
        Scalarisation weights that produced this point.
    policy : Policy
    """

    objectives: np.ndarray = field(default_factory=lambda: np.array([]))
    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    policy: Policy = field(default_factory=Policy)


# ---------------------------------------------------------------------------
# Scalarisation methods
# ---------------------------------------------------------------------------

def weighted_sum_scalarisation(
    q_multi: MultiObjectiveQValues,
    weights: np.ndarray,
    beta: float = 1.0,
) -> Policy:
    """Weighted-sum scalarisation to a softmax policy.

    Q_scalar(s,a) = w^T · Q_vec(s,a)
    π(a|s) ∝ exp(−β · Q_scalar(s,a))

    Parameters
    ----------
    q_multi : MultiObjectiveQValues
    weights : np.ndarray
    beta : float

    Returns
    -------
    Policy
    """
    scalar_q = q_multi.scalarise(weights)
    return scalar_q.to_policy(beta)


def chebyshev_scalarisation(
    q_multi: MultiObjectiveQValues,
    weights: np.ndarray,
    reference_point: np.ndarray,
    beta: float = 1.0,
    augmentation: float = 0.01,
) -> Policy:
    """Augmented Chebyshev (Tchebycheff) scalarisation.

    Minimises the worst weighted deviation from a reference point:

        f(Q) = max_k w_k · (Q_k − z_k^*) + ρ · Σ_k w_k · (Q_k − z_k^*)

    This can reach Pareto points in non-convex regions of the frontier.

    Parameters
    ----------
    q_multi : MultiObjectiveQValues
    weights : np.ndarray
    reference_point : np.ndarray
        Utopia / ideal point z* (best achievable per objective).
    beta : float
    augmentation : float
        Augmentation factor ρ for the weighted-sum term.

    Returns
    -------
    Policy
    """
    scalar: dict[str, dict[str, float]] = {}
    for s, actions in q_multi.values.items():
        scalar[s] = {}
        for a, q_vec in actions.items():
            diff = q_vec - reference_point
            weighted_diff = weights * diff
            cheb = float(np.max(weighted_diff))
            aug = augmentation * float(np.sum(weighted_diff))
            scalar[s][a] = cheb + aug
    return QValues(values=scalar).to_policy(beta)


# ---------------------------------------------------------------------------
# Pareto frontier computation
# ---------------------------------------------------------------------------

class ParetoFrontierComputer:
    """Compute the Pareto frontier by sweeping over weight vectors.

    Parameters
    ----------
    n_objectives : int
    n_weight_samples : int
        Number of weight vectors to sample from the simplex.
    beta : float
    rng : np.random.Generator, optional
    """

    def __init__(
        self,
        n_objectives: int = 2,
        n_weight_samples: int = 50,
        beta: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.n_objectives = n_objectives
        self.n_samples = n_weight_samples
        self.beta = beta
        self.rng = rng or np.random.default_rng()

    def compute(
        self,
        q_multi: MultiObjectiveQValues,
        evaluate_fn: Optional[Any] = None,
    ) -> list[ParetoPoint]:
        """Compute approximate Pareto frontier.

        Parameters
        ----------
        q_multi : MultiObjectiveQValues
        evaluate_fn : callable, optional
            ``(Policy) -> np.ndarray`` returning the K objective values.
            If None, objectives are estimated from the Q-values.

        Returns
        -------
        list[ParetoPoint]
        """
        weight_vectors = self._sample_weight_vectors()
        points: list[ParetoPoint] = []

        for w in weight_vectors:
            policy = weighted_sum_scalarisation(q_multi, w, self.beta)

            if evaluate_fn is not None:
                obj = evaluate_fn(policy)
            else:
                obj = self._estimate_objectives(policy, q_multi)

            points.append(ParetoPoint(
                objectives=np.array(obj, dtype=np.float64),
                weights=w,
                policy=policy,
            ))

        return self._filter_dominated(points)

    def _sample_weight_vectors(self) -> list[np.ndarray]:
        """Sample weight vectors uniformly from the simplex."""
        vectors: list[np.ndarray] = []
        for _ in range(self.n_samples):
            w = self.rng.dirichlet(np.ones(self.n_objectives))
            vectors.append(w)
        return vectors

    @staticmethod
    def _estimate_objectives(
        policy: Policy,
        q_multi: MultiObjectiveQValues,
    ) -> np.ndarray:
        """Estimate objective values from Q-values and policy."""
        n_obj = q_multi.n_objectives
        totals = np.zeros(n_obj, dtype=np.float64)
        n_states = 0

        for s, dist in policy.state_action_probs.items():
            if s not in q_multi.values:
                continue
            for a, pi_a in dist.items():
                if a in q_multi.values[s]:
                    totals += pi_a * q_multi.values[s][a]
            n_states += 1

        if n_states > 0:
            totals /= n_states
        return totals

    @staticmethod
    def _filter_dominated(points: list[ParetoPoint]) -> list[ParetoPoint]:
        """Remove Pareto-dominated points (for minimisation)."""
        if not points:
            return []

        non_dominated: list[ParetoPoint] = []
        for i, p in enumerate(points):
            dominated = False
            for j, q in enumerate(points):
                if i == j:
                    continue
                # q dominates p if q is ≤ in all objectives and < in at least one
                if (np.all(q.objectives <= p.objectives) and
                        np.any(q.objectives < p.objectives)):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(p)
        return non_dominated


# ---------------------------------------------------------------------------
# Constrained MDP via Lagrangian relaxation
# ---------------------------------------------------------------------------

class ConstrainedMDPSolver:
    """Constrained MDP solver via Lagrangian relaxation.

    Solves:
        min_π  E_π[c₀]                    (primary objective)
        s.t.   E_π[c_k] ≤ d_k,  k=1,…,K  (constraints)

    by converting to the unconstrained Lagrangian:

        L(π, λ) = E_π[c₀] + Σ_k λ_k (E_π[c_k] − d_k) + (1/β) D_KL(π ‖ p₀)

    and performing dual ascent on λ.

    Parameters
    ----------
    beta : float
    dual_lr : float
        Learning rate for dual variable updates.
    max_dual_iter : int
    """

    def __init__(
        self,
        beta: float = 1.0,
        dual_lr: float = 0.1,
        max_dual_iter: int = 100,
    ) -> None:
        self.beta = beta
        self.dual_lr = dual_lr
        self.max_dual_iter = max_dual_iter

    def solve(
        self,
        q_multi: MultiObjectiveQValues,
        constraints: dict[int, float],
    ) -> tuple[Policy, np.ndarray]:
        """Solve the constrained MDP.

        Parameters
        ----------
        q_multi : MultiObjectiveQValues
            Objective 0 is the primary; objectives 1..K are constraints.
        constraints : dict[int, float]
            Mapping ``objective_index → upper_bound``.

        Returns
        -------
        tuple[Policy, np.ndarray]
            Optimal policy and final Lagrange multipliers.
        """
        n_constraints = len(constraints)
        lambdas = np.zeros(n_constraints, dtype=np.float64)
        constraint_keys = sorted(constraints.keys())

        best_policy: Optional[Policy] = None
        best_violation = float("inf")

        for iteration in range(self.max_dual_iter):
            # Build Lagrangian weights
            weights = np.zeros(q_multi.n_objectives, dtype=np.float64)
            weights[0] = 1.0
            for i, k in enumerate(constraint_keys):
                weights[k] = lambdas[i]

            # Solve inner problem
            policy = weighted_sum_scalarisation(q_multi, weights, self.beta)

            # Evaluate constraints
            obj_vals = ParetoFrontierComputer._estimate_objectives(policy, q_multi)
            violations = np.zeros(n_constraints, dtype=np.float64)
            for i, k in enumerate(constraint_keys):
                violations[i] = obj_vals[k] - constraints[k]

            # Update dual variables (projected gradient ascent)
            lambdas = np.maximum(lambdas + self.dual_lr * violations, 0.0)

            # Track best feasible solution
            max_viol = float(np.max(violations))
            if max_viol < best_violation:
                best_violation = max_viol
                best_policy = policy

            if max_viol <= 0:
                logger.debug("Lagrangian converged at iteration %d", iteration + 1)
                break

        return best_policy or policy, lambdas


# ---------------------------------------------------------------------------
# Lexicographic optimisation
# ---------------------------------------------------------------------------

def lexicographic_optimise(
    q_multi: MultiObjectiveQValues,
    priority_order: list[int],
    beta: float = 1.0,
    tolerance: float = 0.05,
) -> Policy:
    """Lexicographic optimisation over ordered objectives.

    Optimises objectives in strict priority order: first minimise
    objective priority_order[0], then among near-optimal solutions
    minimise priority_order[1], etc.

    Parameters
    ----------
    q_multi : MultiObjectiveQValues
    priority_order : list[int]
        Objective indices in decreasing priority.
    beta : float
    tolerance : float
        Relative tolerance for "near-optimal" in each objective.

    Returns
    -------
    Policy
    """
    n_obj = q_multi.n_objectives
    # Start with uniform weights
    weights = np.ones(n_obj, dtype=np.float64) * 1e-6

    for obj_idx in priority_order:
        weights[obj_idx] = 1.0
        policy = weighted_sum_scalarisation(q_multi, weights, beta)

        obj_vals = ParetoFrontierComputer._estimate_objectives(policy, q_multi)
        threshold = obj_vals[obj_idx] * (1.0 + tolerance)

        # Add constraint: lock this objective within tolerance
        # by increasing its weight so subsequent objectives don't degrade it
        weights[obj_idx] = 10.0 / max(abs(threshold), 1e-6)

    return weighted_sum_scalarisation(q_multi, weights, beta)


# ---------------------------------------------------------------------------
# Multi-objective value iteration
# ---------------------------------------------------------------------------

def multi_objective_value_iteration(
    states: list[str],
    actions_per_state: dict[str, list[str]],
    transition_fn: dict[str, dict[str, list[tuple[str, float]]]],
    cost_fn: dict[str, dict[str, np.ndarray]],
    n_objectives: int,
    discount: float = 0.99,
    n_iterations: int = 200,
) -> MultiObjectiveQValues:
    """Vector-valued value iteration.

    Computes Q-vectors Q(s, a) ∈ ℝ^K where each component is the
    expected discounted cost for one objective.

    Parameters
    ----------
    states : list[str]
    actions_per_state : dict[str, list[str]]
    transition_fn : dict
        ``T[s][a]`` = list of ``(next_state, probability)``.
    cost_fn : dict
        ``cost_fn[s][a]`` = K-dimensional cost vector.
    n_objectives : int
    discount : float
    n_iterations : int

    Returns
    -------
    MultiObjectiveQValues
    """
    # Initialise V(s) as zero vectors
    V: dict[str, np.ndarray] = {
        s: np.zeros(n_objectives, dtype=np.float64) for s in states
    }

    for _ in range(n_iterations):
        new_V: dict[str, np.ndarray] = {}
        for s in states:
            actions = actions_per_state.get(s, [])
            if not actions:
                new_V[s] = V.get(s, np.zeros(n_objectives, dtype=np.float64))
                continue

            # Compute Q(s,a) for each action
            best_q: Optional[np.ndarray] = None
            best_norm = float("inf")
            for a in actions:
                c = cost_fn.get(s, {}).get(a, np.zeros(n_objectives, dtype=np.float64))
                transitions = transition_fn.get(s, {}).get(a, [])
                future = np.zeros(n_objectives, dtype=np.float64)
                for ns, prob in transitions:
                    future += prob * V.get(ns, np.zeros(n_objectives, dtype=np.float64))
                q = c + discount * future

                # Use L2 norm as tie-breaker for "best" V
                norm = float(np.linalg.norm(q))
                if best_q is None or norm < best_norm:
                    best_q = q
                    best_norm = norm

            new_V[s] = best_q if best_q is not None else V.get(s, np.zeros(n_objectives))
        V = new_V

    # Extract Q-values
    q_values: dict[str, dict[str, np.ndarray]] = {}
    for s in states:
        actions = actions_per_state.get(s, [])
        if not actions:
            continue
        q_values[s] = {}
        for a in actions:
            c = cost_fn.get(s, {}).get(a, np.zeros(n_objectives, dtype=np.float64))
            transitions = transition_fn.get(s, {}).get(a, [])
            future = np.zeros(n_objectives, dtype=np.float64)
            for ns, prob in transitions:
                future += prob * V.get(ns, np.zeros(n_objectives, dtype=np.float64))
            q_values[s][a] = c + discount * future

    return MultiObjectiveQValues(
        values=q_values,
        n_objectives=n_objectives,
    )


# ---------------------------------------------------------------------------
# Pareto frontier data for visualisation
# ---------------------------------------------------------------------------

def pareto_frontier_data(
    points: list[ParetoPoint],
) -> dict[str, Any]:
    """Extract Pareto frontier data suitable for plotting.

    Parameters
    ----------
    points : list[ParetoPoint]
        Non-dominated Pareto points.

    Returns
    -------
    dict[str, Any]
        Keys: ``objectives`` (N×K array), ``weights`` (N×K array),
        ``n_points``, ``n_objectives``.
    """
    if not points:
        return {"objectives": [], "weights": [], "n_points": 0, "n_objectives": 0}

    obj_arr = np.array([p.objectives for p in points])
    w_arr = np.array([p.weights for p in points])

    return {
        "objectives": obj_arr.tolist(),
        "weights": w_arr.tolist(),
        "n_points": len(points),
        "n_objectives": obj_arr.shape[1] if obj_arr.ndim == 2 else 0,
    }
