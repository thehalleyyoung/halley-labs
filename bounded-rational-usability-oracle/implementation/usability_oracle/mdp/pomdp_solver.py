"""
usability_oracle.mdp.pomdp_solver — POMDP solvers.

Implements several algorithms for computing (approximate) policies in
partially observable MDPs:

1. **Exact value iteration** — alpha-vector Bellman backup (small POMDPs)
2. **PBVI** — Point-Based Value Iteration (Pineau et al., 2003)
3. **Perseus** — randomised point-based solver (Spaan & Vlassis, 2005)
4. **SARSOP** — basic Successive Approximations (Kurniawati et al., 2008)
5. **QMDP** — MDP-based upper-bound heuristic (Littman et al., 1995)
6. **FIB** — Fast Informed Bound (Hauskrecht, 2000)
7. **Bounded-rational policy** — softmax over belief-space Q-values

All solvers operate on the :class:`POMDP` data structure and return
alpha vectors or belief-space value functions.

References
----------
- Pineau, J., Gordon, G. & Thrun, S. (2003). Point-based value iteration:
  An anytime algorithm for POMDPs. *IJCAI*.
- Spaan, M. T. J. & Vlassis, N. (2005). Perseus: Randomized point-based
  value iteration for POMDPs. *JAIR*.
- Kurniawati, H., Hsu, D. & Lee, W. S. (2008). SARSOP: Efficient
  point-based POMDP planning. *RSS*.
- Littman, M. L., Cassandra, A. R. & Kaelbling, L. P. (1995). Learning
  policies for partially observable environments. *ICML*.
- Hauskrecht, M. (2000). Value-function approximations for partially
  observable Markov decision processes. *JAIR*.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from usability_oracle.mdp.models import MDP
from usability_oracle.mdp.pomdp import BeliefState, POMDP, point_based_beliefs
from usability_oracle.mdp.belief import BeliefUpdater, belief_entropy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alpha vector representation
# ---------------------------------------------------------------------------


@dataclass
class AlphaVector:
    """A single alpha vector in the value function representation.

    The POMDP value function is piece-wise linear and convex (PWLC):

        V(b) = max_α  α · b

    where each alpha vector α is associated with an action.

    Parameters
    ----------
    vector : np.ndarray
        Value vector of length |S|.
    action_id : str
        Action associated with this alpha vector.
    """

    vector: np.ndarray
    action_id: str

    def value(self, belief: BeliefState, state_ids: list[str]) -> float:
        """Compute α · b for a given belief state."""
        b = belief.to_vector(state_ids)
        return float(np.dot(self.vector, b))

    def __repr__(self) -> str:
        return f"AlphaVector(action={self.action_id!r}, |α|={np.linalg.norm(self.vector):.3f})"


@dataclass
class POMDPPolicy:
    """A POMDP policy represented as a set of alpha vectors.

    The policy selects the action associated with the alpha vector
    that maximises V(b) = max_α α · b.

    Parameters
    ----------
    alpha_vectors : list[AlphaVector]
    state_ids : list[str]
        Ordered state IDs for vector alignment.
    """

    alpha_vectors: list[AlphaVector] = field(default_factory=list)
    state_ids: list[str] = field(default_factory=list)

    def value(self, belief: BeliefState) -> float:
        """Evaluate V(b) = max_α α · b."""
        if not self.alpha_vectors:
            return 0.0
        b = belief.to_vector(self.state_ids)
        values = [float(np.dot(av.vector, b)) for av in self.alpha_vectors]
        return max(values)

    def action(self, belief: BeliefState) -> str:
        """Select the action for belief state b: argmax_α α · b."""
        if not self.alpha_vectors:
            return ""
        b = belief.to_vector(self.state_ids)
        best_idx = 0
        best_val = -math.inf
        for i, av in enumerate(self.alpha_vectors):
            val = float(np.dot(av.vector, b))
            if val > best_val:
                best_val = val
                best_idx = i
        return self.alpha_vectors[best_idx].action_id

    @property
    def n_vectors(self) -> int:
        return len(self.alpha_vectors)

    def __repr__(self) -> str:
        return f"POMDPPolicy(|α|={self.n_vectors}, |S|={len(self.state_ids)})"


# ---------------------------------------------------------------------------
# Convergence monitor
# ---------------------------------------------------------------------------


@dataclass
class ConvergenceInfo:
    """Convergence diagnostics for a POMDP solver."""

    iterations: int = 0
    value_changes: list[float] = field(default_factory=list)
    n_alpha_vectors: list[int] = field(default_factory=list)
    converged: bool = False
    final_epsilon: float = float("inf")

    @property
    def converged_at(self) -> Optional[int]:
        if self.converged and self.value_changes:
            for i, vc in enumerate(self.value_changes):
                if vc < self.final_epsilon:
                    return i
        return None


def _check_convergence(
    old_policy: POMDPPolicy,
    new_policy: POMDPPolicy,
    belief_points: list[BeliefState],
    epsilon: float,
) -> tuple[bool, float]:
    """Check if the value function has converged at the belief points."""
    max_diff = 0.0
    for bp in belief_points:
        old_val = old_policy.value(bp)
        new_val = new_policy.value(bp)
        max_diff = max(max_diff, abs(new_val - old_val))
    return max_diff < epsilon, max_diff


# ---------------------------------------------------------------------------
# Helpers: transition and reward matrices
# ---------------------------------------------------------------------------


def _build_reward_matrix(pomdp: POMDP) -> dict[str, np.ndarray]:
    """Build R_a[s] = Σ_{s'} T(s'|s,a) · (−cost(s,a,s')) for each action."""
    sids = pomdp.state_ids
    n = len(sids)
    sid_to_idx = {s: i for i, s in enumerate(sids)}
    R: dict[str, np.ndarray] = {}

    for aid in pomdp.action_ids:
        r = np.zeros(n, dtype=np.float64)
        for i, sid in enumerate(sids):
            for target, prob, cost in pomdp.mdp.get_transitions(sid, aid):
                r[i] += prob * (-cost)
        R[aid] = r
    return R


def _build_transition_matrices(pomdp: POMDP) -> dict[str, np.ndarray]:
    """Build T_a[s, s'] = T(s'|s, a) for each action."""
    sids = pomdp.state_ids
    n = len(sids)
    sid_to_idx = {s: i for i, s in enumerate(sids)}
    T: dict[str, np.ndarray] = {}

    for aid in pomdp.action_ids:
        t = np.zeros((n, n), dtype=np.float64)
        for i, sid in enumerate(sids):
            for target, prob, _cost in pomdp.mdp.get_transitions(sid, aid):
                j = sid_to_idx.get(target)
                if j is not None:
                    t[i, j] += prob
        T[aid] = t
    return T


def _build_observation_matrices(pomdp: POMDP) -> dict[str, dict[str, np.ndarray]]:
    """Build O_a,o[s'] = O(o | s', a) for each (action, observation)."""
    sids = pomdp.state_ids
    n = len(sids)
    O: dict[str, dict[str, np.ndarray]] = {}

    for aid in pomdp.action_ids:
        O[aid] = {}
        for oid in pomdp.observation_ids:
            o_vec = np.zeros(n, dtype=np.float64)
            for j, sid in enumerate(sids):
                o_vec[j] = pomdp.observation_model.prob(oid, sid, aid)
            O[aid][oid] = o_vec
    return O


# ---------------------------------------------------------------------------
# QMDP heuristic
# ---------------------------------------------------------------------------


class QMDPSolver:
    """QMDP heuristic: solve the underlying MDP, then use Q-values.

    The QMDP approximation ignores future observations and uses the
    MDP Q-function:

        V_QMDP(b) = max_a  Σ_s b(s) Q_MDP(s, a)

    This provides a fast upper bound on the true POMDP value function.

    References
    ----------
    - Littman, M. L., Cassandra, A. R. & Kaelbling, L. P. (1995).
    """

    def solve(
        self,
        pomdp: POMDP,
        discount: Optional[float] = None,
        epsilon: float = 1e-6,
        max_iter: int = 10_000,
    ) -> POMDPPolicy:
        """Solve using QMDP heuristic.

        Returns
        -------
        POMDPPolicy
            Policy with one alpha vector per action (the Q-function row).
        """
        gamma = discount if discount is not None else pomdp.mdp.discount
        sids = pomdp.state_ids
        n = len(sids)
        sid_to_idx = {s: i for i, s in enumerate(sids)}

        R = _build_reward_matrix(pomdp)
        T = _build_transition_matrices(pomdp)

        # Value iteration on underlying MDP (reward formulation)
        V = np.zeros(n, dtype=np.float64)
        for iteration in range(max_iter):
            V_new = np.full(n, -np.inf, dtype=np.float64)
            for aid in pomdp.action_ids:
                Q_a = R[aid] + gamma * (T[aid] @ V)
                V_new = np.maximum(V_new, Q_a)

            diff = np.max(np.abs(V_new - V))
            V = V_new
            if diff < epsilon:
                logger.info("QMDP converged in %d iterations", iteration + 1)
                break

        # Build alpha vectors: one per action = Q(·, a)
        alphas: list[AlphaVector] = []
        for aid in pomdp.action_ids:
            Q_a = R[aid] + gamma * (T[aid] @ V)
            alphas.append(AlphaVector(vector=Q_a, action_id=aid))

        return POMDPPolicy(alpha_vectors=alphas, state_ids=sids)


# ---------------------------------------------------------------------------
# FIB (Fast Informed Bound)
# ---------------------------------------------------------------------------


class FIBSolver:
    """Fast Informed Bound (Hauskrecht, 2000).

    Tighter than QMDP by incorporating observation structure into the
    upper bound.  For each action and observation, computes a separate
    bound and combines them.

    V_FIB(b) = max_a [R(b,a) + γ Σ_o max_α Σ_{s'} O(o|s',a) α(s') T(s'|s,a) b(s)]
    """

    def solve(
        self,
        pomdp: POMDP,
        discount: Optional[float] = None,
        epsilon: float = 1e-6,
        max_iter: int = 1_000,
    ) -> POMDPPolicy:
        """Solve using the FIB algorithm.

        Returns
        -------
        POMDPPolicy
        """
        gamma = discount if discount is not None else pomdp.mdp.discount
        sids = pomdp.state_ids
        n = len(sids)

        R = _build_reward_matrix(pomdp)
        T = _build_transition_matrices(pomdp)
        O = _build_observation_matrices(pomdp)

        # Initialise with one alpha vector per action (QMDP-like)
        alphas: list[AlphaVector] = []
        for aid in pomdp.action_ids:
            alphas.append(AlphaVector(vector=R[aid].copy(), action_id=aid))

        for iteration in range(max_iter):
            new_alphas: list[AlphaVector] = []

            for aid in pomdp.action_ids:
                alpha_a = R[aid].copy()

                for oid in pomdp.observation_ids:
                    o_diag = np.diag(O[aid][oid])

                    # For each existing alpha vector, compute contribution
                    best_contrib = np.full(n, -np.inf, dtype=np.float64)
                    for av in alphas:
                        # g_{a,o,α}(s) = Σ_{s'} O(o|s',a) T(s'|s,a) α(s')
                        contrib = T[aid] @ (O[aid][oid] * av.vector)
                        best_contrib = np.maximum(best_contrib, contrib)

                    alpha_a += gamma * best_contrib

                new_alphas.append(AlphaVector(vector=alpha_a, action_id=aid))

            # Check convergence
            max_diff = 0.0
            for old_a, new_a in zip(alphas, new_alphas):
                max_diff = max(max_diff, float(np.max(np.abs(old_a.vector - new_a.vector))))

            alphas = new_alphas

            if max_diff < epsilon:
                logger.info("FIB converged in %d iterations", iteration + 1)
                break

        return POMDPPolicy(alpha_vectors=alphas, state_ids=sids)


# ---------------------------------------------------------------------------
# Exact POMDP value iteration (small POMDPs only)
# ---------------------------------------------------------------------------


class ExactPOMDPSolver:
    """Exact value iteration using alpha-vector Bellman backups.

    Only feasible for very small POMDPs (|S| ≤ ~10, |A| ≤ ~5, |Ω| ≤ ~5).
    The number of alpha vectors can grow exponentially.

    The Bellman backup produces new alpha vectors:

        α'_a,o(s) = R(s,a) + γ Σ_{s'} T(s'|s,a) O(o|s',a) α(s')

    Parameters
    ----------
    max_vectors : int
        Prune the alpha vector set if it exceeds this size.
    """

    def solve(
        self,
        pomdp: POMDP,
        discount: Optional[float] = None,
        epsilon: float = 1e-6,
        max_iter: int = 200,
        max_vectors: int = 500,
    ) -> tuple[POMDPPolicy, ConvergenceInfo]:
        """Run exact value iteration.

        Returns
        -------
        POMDPPolicy
        ConvergenceInfo
        """
        gamma = discount if discount is not None else pomdp.mdp.discount
        sids = pomdp.state_ids
        n = len(sids)
        conv = ConvergenceInfo()

        R = _build_reward_matrix(pomdp)
        T = _build_transition_matrices(pomdp)
        O = _build_observation_matrices(pomdp)

        # Initialise: one alpha vector per action
        alphas: list[AlphaVector] = []
        for aid in pomdp.action_ids:
            alphas.append(AlphaVector(vector=R[aid].copy(), action_id=aid))

        belief_points = point_based_beliefs(pomdp, n_points=max(50, n * 5))
        policy = POMDPPolicy(alpha_vectors=alphas, state_ids=sids)

        for iteration in range(max_iter):
            new_alphas = self._bellman_backup(
                pomdp, alphas, R, T, O, gamma, sids
            )

            # Prune dominated vectors
            if len(new_alphas) > max_vectors:
                new_alphas = self._prune(new_alphas, belief_points, sids, max_vectors)

            new_policy = POMDPPolicy(alpha_vectors=new_alphas, state_ids=sids)
            converged, diff = _check_convergence(
                policy, new_policy, belief_points, epsilon
            )

            conv.iterations = iteration + 1
            conv.value_changes.append(diff)
            conv.n_alpha_vectors.append(len(new_alphas))

            alphas = new_alphas
            policy = new_policy

            if converged:
                conv.converged = True
                conv.final_epsilon = diff
                logger.info(
                    "Exact POMDP VI converged in %d iterations (%d α-vectors)",
                    iteration + 1, len(alphas),
                )
                break

        return policy, conv

    def _bellman_backup(
        self,
        pomdp: POMDP,
        alphas: list[AlphaVector],
        R: dict[str, np.ndarray],
        T: dict[str, np.ndarray],
        O: dict[str, dict[str, np.ndarray]],
        gamma: float,
        sids: list[str],
    ) -> list[AlphaVector]:
        """One Bellman backup step: generate new alpha vectors.

        For each action a and observation o, and for each existing α:
            g_{a,o,α}(s) = Σ_{s'} T(s'|s,a) O(o|s',a) α(s')

        Then for each action a:
            α'_a(s) = R(s,a) + γ Σ_o max_{α} g_{a,o,α}(s)
        """
        n = len(sids)
        new_alphas: list[AlphaVector] = []

        # Pre-compute g vectors: g[a][o] = list of vectors (one per old α)
        g: dict[str, dict[str, list[np.ndarray]]] = {}
        for aid in pomdp.action_ids:
            g[aid] = {}
            for oid in pomdp.observation_ids:
                g[aid][oid] = []
                for av in alphas:
                    # g_{a,o,α}(s) = Σ_{s'} T(s'|s,a) O(o|s',a) α(s')
                    vec = T[aid] @ (O[aid][oid] * av.vector)
                    g[aid][oid].append(vec)

        # For each action, combine best g-vectors across observations
        for aid in pomdp.action_ids:
            # For each observation, pick the best g-vector
            alpha_a = R[aid].copy()
            for oid in pomdp.observation_ids:
                g_vecs = g[aid][oid]
                if not g_vecs:
                    continue
                # Take element-wise max across all g-vectors
                best_g = g_vecs[0].copy()
                for gv in g_vecs[1:]:
                    best_g = np.maximum(best_g, gv)
                alpha_a += gamma * best_g

            new_alphas.append(AlphaVector(vector=alpha_a, action_id=aid))

        return new_alphas

    @staticmethod
    def _prune(
        alphas: list[AlphaVector],
        belief_points: list[BeliefState],
        state_ids: list[str],
        max_vectors: int,
    ) -> list[AlphaVector]:
        """Prune dominated alpha vectors, keeping at most *max_vectors*."""
        if len(alphas) <= max_vectors:
            return alphas

        # Keep vectors that are best for at least one belief point
        useful: set[int] = set()
        for bp in belief_points:
            b = bp.to_vector(state_ids)
            best_idx = 0
            best_val = -math.inf
            for i, av in enumerate(alphas):
                val = float(np.dot(av.vector, b))
                if val > best_val:
                    best_val = val
                    best_idx = i
            useful.add(best_idx)

        result = [alphas[i] for i in sorted(useful)]
        # If still too many, keep the top by average value
        if len(result) > max_vectors:
            result.sort(key=lambda av: -float(np.mean(av.vector)))
            result = result[:max_vectors]
        return result


# ---------------------------------------------------------------------------
# PBVI (Point-Based Value Iteration)
# ---------------------------------------------------------------------------


class PBVISolver:
    """Point-Based Value Iteration (Pineau, Gordon & Thrun, 2003).

    Performs Bellman backups only at a finite set of reachable belief
    points, avoiding the exponential blowup of exact methods.

    Parameters
    ----------
    n_belief_points : int
        Number of belief points to sample.
    n_expand_steps : int
        Expansion steps for belief point generation.
    """

    def __init__(
        self, n_belief_points: int = 100, n_expand_steps: int = 20
    ) -> None:
        self.n_belief_points = n_belief_points
        self.n_expand_steps = n_expand_steps

    def solve(
        self,
        pomdp: POMDP,
        discount: Optional[float] = None,
        epsilon: float = 1e-4,
        max_iter: int = 200,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[POMDPPolicy, ConvergenceInfo]:
        """Run PBVI.

        Returns
        -------
        POMDPPolicy
        ConvergenceInfo
        """
        gamma = discount if discount is not None else pomdp.mdp.discount
        sids = pomdp.state_ids
        n = len(sids)
        conv = ConvergenceInfo()

        # Generate belief points
        belief_points = point_based_beliefs(
            pomdp, self.n_belief_points, self.n_expand_steps, rng
        )
        logger.info("PBVI: generated %d belief points", len(belief_points))

        R = _build_reward_matrix(pomdp)
        T = _build_transition_matrices(pomdp)
        O = _build_observation_matrices(pomdp)

        # Initialise alpha vectors
        alphas: list[AlphaVector] = []
        for aid in pomdp.action_ids:
            alphas.append(AlphaVector(vector=R[aid].copy(), action_id=aid))

        policy = POMDPPolicy(alpha_vectors=alphas, state_ids=sids)

        for iteration in range(max_iter):
            new_alphas = self._pbvi_backup(
                pomdp, alphas, belief_points, R, T, O, gamma, sids
            )

            new_policy = POMDPPolicy(alpha_vectors=new_alphas, state_ids=sids)
            converged, diff = _check_convergence(
                policy, new_policy, belief_points, epsilon
            )

            conv.iterations = iteration + 1
            conv.value_changes.append(diff)
            conv.n_alpha_vectors.append(len(new_alphas))

            alphas = new_alphas
            policy = new_policy

            if converged:
                conv.converged = True
                conv.final_epsilon = diff
                logger.info("PBVI converged in %d iterations", iteration + 1)
                break

        return policy, conv

    def _pbvi_backup(
        self,
        pomdp: POMDP,
        alphas: list[AlphaVector],
        belief_points: list[BeliefState],
        R: dict[str, np.ndarray],
        T: dict[str, np.ndarray],
        O: dict[str, dict[str, np.ndarray]],
        gamma: float,
        sids: list[str],
    ) -> list[AlphaVector]:
        """PBVI backup: for each belief point, find best alpha vector.

        For each b in B, compute:
            α_b = argmax_a [R_a + γ Σ_o argmax_α g_{a,o,α}] · b
        """
        n = len(sids)
        new_alphas: list[AlphaVector] = []

        # Pre-compute g vectors
        g: dict[str, dict[str, list[np.ndarray]]] = {}
        for aid in pomdp.action_ids:
            g[aid] = {}
            for oid in pomdp.observation_ids:
                g[aid][oid] = []
                for av in alphas:
                    vec = T[aid] @ (O[aid][oid] * av.vector)
                    g[aid][oid].append(vec)

        seen_actions: set[str] = set()

        for bp in belief_points:
            b = bp.to_vector(sids)

            best_action = ""
            best_alpha = np.zeros(n, dtype=np.float64)
            best_val = -math.inf

            for aid in pomdp.action_ids:
                alpha_a = R[aid].copy()

                for oid in pomdp.observation_ids:
                    g_vecs = g[aid][oid]
                    if not g_vecs:
                        continue
                    # Pick the g-vector best for this belief
                    g_vals = [float(np.dot(gv, b)) for gv in g_vecs]
                    best_g_idx = int(np.argmax(g_vals))
                    alpha_a += gamma * g_vecs[best_g_idx]

                val = float(np.dot(alpha_a, b))
                if val > best_val:
                    best_val = val
                    best_alpha = alpha_a
                    best_action = aid

            new_alphas.append(AlphaVector(vector=best_alpha, action_id=best_action))
            seen_actions.add(best_action)

        return new_alphas


# ---------------------------------------------------------------------------
# Perseus algorithm
# ---------------------------------------------------------------------------


class PerseusSolver:
    """Perseus: randomised point-based value iteration.

    Like PBVI but only backs up belief points whose value has *not*
    improved, leading to faster convergence with fewer backups.

    References
    ----------
    - Spaan, M. T. J. & Vlassis, N. (2005). Perseus. *JAIR 24*.
    """

    def __init__(
        self, n_belief_points: int = 200, n_expand_steps: int = 20
    ) -> None:
        self.n_belief_points = n_belief_points
        self.n_expand_steps = n_expand_steps

    def solve(
        self,
        pomdp: POMDP,
        discount: Optional[float] = None,
        epsilon: float = 1e-4,
        max_iter: int = 200,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[POMDPPolicy, ConvergenceInfo]:
        """Run Perseus.

        Returns
        -------
        POMDPPolicy
        ConvergenceInfo
        """
        rng = rng or np.random.default_rng()
        gamma = discount if discount is not None else pomdp.mdp.discount
        sids = pomdp.state_ids
        n = len(sids)
        conv = ConvergenceInfo()

        belief_points = point_based_beliefs(
            pomdp, self.n_belief_points, self.n_expand_steps, rng
        )
        logger.info("Perseus: generated %d belief points", len(belief_points))

        R = _build_reward_matrix(pomdp)
        T = _build_transition_matrices(pomdp)
        O = _build_observation_matrices(pomdp)

        # Initialise
        alphas: list[AlphaVector] = []
        for aid in pomdp.action_ids:
            alphas.append(AlphaVector(vector=R[aid].copy(), action_id=aid))

        policy = POMDPPolicy(alpha_vectors=alphas, state_ids=sids)

        for iteration in range(max_iter):
            # Compute current values for all belief points
            current_values = np.array(
                [policy.value(bp) for bp in belief_points], dtype=np.float64
            )

            # Find not-improved belief points
            new_alphas = list(alphas)
            not_improved = list(range(len(belief_points)))
            rng.shuffle(not_improved)  # type: ignore[arg-type]

            while not_improved:
                # Pick a random not-improved point
                idx = not_improved[0]
                bp = belief_points[idx]
                b = bp.to_vector(sids)

                # Backup at this point
                best_alpha, best_action = self._backup_point(
                    pomdp, alphas, bp, R, T, O, gamma, sids
                )

                # Check which not-improved points are now improved
                new_val = float(np.dot(best_alpha, b))
                if new_val >= current_values[idx] - 1e-10:
                    new_alphas.append(AlphaVector(vector=best_alpha, action_id=best_action))

                # Remove improved points
                still_not_improved = []
                temp_policy = POMDPPolicy(alpha_vectors=new_alphas, state_ids=sids)
                for i in not_improved:
                    if temp_policy.value(belief_points[i]) < current_values[i] - 1e-10:
                        still_not_improved.append(i)
                not_improved = still_not_improved

            new_policy = POMDPPolicy(alpha_vectors=new_alphas, state_ids=sids)
            converged, diff = _check_convergence(
                policy, new_policy, belief_points, epsilon
            )

            conv.iterations = iteration + 1
            conv.value_changes.append(diff)
            conv.n_alpha_vectors.append(len(new_alphas))

            alphas = new_alphas
            policy = new_policy

            if converged:
                conv.converged = True
                conv.final_epsilon = diff
                logger.info("Perseus converged in %d iterations", iteration + 1)
                break

        return policy, conv

    @staticmethod
    def _backup_point(
        pomdp: POMDP,
        alphas: list[AlphaVector],
        belief: BeliefState,
        R: dict[str, np.ndarray],
        T: dict[str, np.ndarray],
        O: dict[str, dict[str, np.ndarray]],
        gamma: float,
        sids: list[str],
    ) -> tuple[np.ndarray, str]:
        """Perform a single-point backup."""
        n = len(sids)
        b = belief.to_vector(sids)

        best_action = ""
        best_alpha = np.zeros(n, dtype=np.float64)
        best_val = -math.inf

        for aid in pomdp.action_ids:
            alpha_a = R[aid].copy()

            for oid in pomdp.observation_ids:
                g_vecs = []
                for av in alphas:
                    vec = T[aid] @ (O[aid][oid] * av.vector)
                    g_vecs.append(vec)

                if g_vecs:
                    g_vals = [float(np.dot(gv, b)) for gv in g_vecs]
                    best_g_idx = int(np.argmax(g_vals))
                    alpha_a += gamma * g_vecs[best_g_idx]

            val = float(np.dot(alpha_a, b))
            if val > best_val:
                best_val = val
                best_alpha = alpha_a
                best_action = aid

        return best_alpha, best_action


# ---------------------------------------------------------------------------
# SARSOP (basic version)
# ---------------------------------------------------------------------------


class SARSOPSolver:
    """Basic SARSOP: Successive Approximations of Reachable Space.

    Focuses belief-point exploration on the reachable space under
    optimal and exploratory policies, pruning unreachable belief points.

    This is a simplified version of the full SARSOP algorithm that
    combines PBVI-style backups with targeted belief-space exploration.

    References
    ----------
    - Kurniawati, H., Hsu, D. & Lee, W. S. (2008). SARSOP. *RSS*.
    """

    def __init__(
        self,
        n_belief_points: int = 200,
        n_expand_per_iter: int = 10,
    ) -> None:
        self.n_belief_points = n_belief_points
        self.n_expand_per_iter = n_expand_per_iter

    def solve(
        self,
        pomdp: POMDP,
        discount: Optional[float] = None,
        epsilon: float = 1e-4,
        max_iter: int = 200,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[POMDPPolicy, ConvergenceInfo]:
        """Run basic SARSOP.

        Returns
        -------
        POMDPPolicy
        ConvergenceInfo
        """
        rng = rng or np.random.default_rng()
        gamma = discount if discount is not None else pomdp.mdp.discount
        sids = pomdp.state_ids
        n = len(sids)
        conv = ConvergenceInfo()

        R = _build_reward_matrix(pomdp)
        T = _build_transition_matrices(pomdp)
        O = _build_observation_matrices(pomdp)

        # Start with initial belief and its neighbours
        belief_points = [pomdp.initial_belief]
        updater = BeliefUpdater(pomdp)

        # Initialise alpha vectors
        alphas: list[AlphaVector] = []
        for aid in pomdp.action_ids:
            alphas.append(AlphaVector(vector=R[aid].copy(), action_id=aid))

        policy = POMDPPolicy(alpha_vectors=alphas, state_ids=sids)

        for iteration in range(max_iter):
            # Expand belief points along optimal and exploratory paths
            belief_points = self._expand_reachable(
                pomdp, belief_points, policy, updater, rng
            )

            # PBVI-style backup at all belief points
            new_alphas = self._backup_all(
                pomdp, alphas, belief_points, R, T, O, gamma, sids
            )

            new_policy = POMDPPolicy(alpha_vectors=new_alphas, state_ids=sids)
            converged, diff = _check_convergence(
                policy, new_policy, belief_points, epsilon
            )

            conv.iterations = iteration + 1
            conv.value_changes.append(diff)
            conv.n_alpha_vectors.append(len(new_alphas))

            alphas = new_alphas
            policy = new_policy

            if converged:
                conv.converged = True
                conv.final_epsilon = diff
                logger.info("SARSOP converged in %d iterations", iteration + 1)
                break

        return policy, conv

    def _expand_reachable(
        self,
        pomdp: POMDP,
        belief_points: list[BeliefState],
        policy: POMDPPolicy,
        updater: BeliefUpdater,
        rng: np.random.Generator,
    ) -> list[BeliefState]:
        """Expand belief points along the reachable space."""
        if len(belief_points) >= self.n_belief_points:
            return belief_points

        new_points = list(belief_points)

        for _ in range(self.n_expand_per_iter):
            if len(new_points) >= self.n_belief_points:
                break

            # Pick a random existing belief point
            bp = new_points[int(rng.integers(len(new_points)))]

            # Follow the current policy action
            aid = policy.action(bp)
            if not aid:
                aid = pomdp.action_ids[int(rng.integers(len(pomdp.action_ids)))]

            # Try each observation
            for oid in pomdp.observation_ids:
                p_obs = updater.observation_likelihood(bp, aid, oid)
                if p_obs < 1e-10:
                    continue

                b_prime = updater.update(bp, aid, oid)
                new_points.append(b_prime)

                if len(new_points) >= self.n_belief_points:
                    break

        return new_points

    @staticmethod
    def _backup_all(
        pomdp: POMDP,
        alphas: list[AlphaVector],
        belief_points: list[BeliefState],
        R: dict[str, np.ndarray],
        T: dict[str, np.ndarray],
        O: dict[str, dict[str, np.ndarray]],
        gamma: float,
        sids: list[str],
    ) -> list[AlphaVector]:
        """Backup at all belief points (PBVI-style)."""
        n = len(sids)
        new_alphas: list[AlphaVector] = []

        # Pre-compute g vectors
        g: dict[str, dict[str, list[np.ndarray]]] = {}
        for aid in pomdp.action_ids:
            g[aid] = {}
            for oid in pomdp.observation_ids:
                g[aid][oid] = []
                for av in alphas:
                    vec = T[aid] @ (O[aid][oid] * av.vector)
                    g[aid][oid].append(vec)

        for bp in belief_points:
            b = bp.to_vector(sids)

            best_action = ""
            best_alpha = np.zeros(n, dtype=np.float64)
            best_val = -math.inf

            for aid in pomdp.action_ids:
                alpha_a = R[aid].copy()

                for oid in pomdp.observation_ids:
                    g_vecs = g[aid][oid]
                    if not g_vecs:
                        continue
                    g_vals = [float(np.dot(gv, b)) for gv in g_vecs]
                    best_g_idx = int(np.argmax(g_vals))
                    alpha_a += gamma * g_vecs[best_g_idx]

                val = float(np.dot(alpha_a, b))
                if val > best_val:
                    best_val = val
                    best_alpha = alpha_a
                    best_action = aid

            new_alphas.append(AlphaVector(vector=best_alpha, action_id=best_action))

        return new_alphas


# ---------------------------------------------------------------------------
# Bounded-rational POMDP policy
# ---------------------------------------------------------------------------


class BoundedRationalPOMDPPolicy:
    """Softmax policy over belief-space Q-values.

    Instead of hard maximisation, uses a Boltzmann (softmax) distribution:

        π(a | b) ∝ exp(β · Q(b, a))

    where β is the rationality parameter.  β→∞ recovers the optimal
    policy; β→0 gives uniform random behaviour.

    Parameters
    ----------
    base_policy : POMDPPolicy
        Alpha-vector policy providing Q(b, a).
    rationality : float
        Softmax temperature β.  Higher → more rational.
    """

    def __init__(
        self, base_policy: POMDPPolicy, rationality: float = 1.0
    ) -> None:
        self.base_policy = base_policy
        self.rationality = rationality

    def action_probabilities(
        self, belief: BeliefState
    ) -> dict[str, float]:
        """Compute softmax action probabilities π(a | b).

        Parameters
        ----------
        belief : BeliefState

        Returns
        -------
        dict[str, float]
            Action → probability.
        """
        if not self.base_policy.alpha_vectors:
            return {}

        b = belief.to_vector(self.base_policy.state_ids)

        # Group alpha vectors by action
        action_q: dict[str, float] = {}
        for av in self.base_policy.alpha_vectors:
            val = float(np.dot(av.vector, b))
            if av.action_id not in action_q or val > action_q[av.action_id]:
                action_q[av.action_id] = val

        if not action_q:
            return {}

        # Softmax
        actions = list(action_q.keys())
        q_values = np.array([action_q[a] for a in actions], dtype=np.float64)
        q_values -= q_values.max()  # numerical stability
        exp_q = np.exp(self.rationality * q_values)
        probs = exp_q / exp_q.sum()

        return {a: float(p) for a, p in zip(actions, probs)}

    def sample_action(
        self, belief: BeliefState, rng: Optional[np.random.Generator] = None
    ) -> str:
        """Sample an action from the softmax distribution.

        Parameters
        ----------
        belief : BeliefState
        rng : np.random.Generator, optional

        Returns
        -------
        str
            Sampled action ID.
        """
        rng = rng or np.random.default_rng()
        probs = self.action_probabilities(belief)
        if not probs:
            return ""
        actions = list(probs.keys())
        p = np.array([probs[a] for a in actions], dtype=np.float64)
        idx = int(rng.choice(len(actions), p=p))
        return actions[idx]

    def expected_cost(self, belief: BeliefState) -> float:
        """Expected cost under the softmax policy: E_π[−Q(b, a)]."""
        probs = self.action_probabilities(belief)
        b = belief.to_vector(self.base_policy.state_ids)

        total = 0.0
        for av in self.base_policy.alpha_vectors:
            p = probs.get(av.action_id, 0.0)
            total -= p * float(np.dot(av.vector, b))
        return total


# ---------------------------------------------------------------------------
# Policy tree representation
# ---------------------------------------------------------------------------


@dataclass
class PolicyTreeNode:
    """A node in a finite-horizon POMDP policy tree.

    Parameters
    ----------
    action_id : str
        Action to take at this node.
    children : dict[str, PolicyTreeNode]
        Mapping observation_id → child node.
    value : float
        Expected value at this node.
    depth : int
        Depth in the tree (0 = root).
    """

    action_id: str
    children: dict[str, PolicyTreeNode] = field(default_factory=dict)
    value: float = 0.0
    depth: int = 0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def tree_size(self) -> int:
        """Total number of nodes in this subtree."""
        return 1 + sum(c.tree_size for c in self.children.values())

    def action_at(self, observation_history: Sequence[str]) -> str:
        """Follow observations to find the action at a given history."""
        node = self
        for obs in observation_history:
            if obs not in node.children:
                return node.action_id
            node = node.children[obs]
        return node.action_id

    def __repr__(self) -> str:
        return (
            f"PolicyTreeNode(a={self.action_id!r}, "
            f"|children|={len(self.children)}, d={self.depth})"
        )


def build_policy_tree(
    pomdp: POMDP,
    policy: POMDPPolicy,
    initial_belief: Optional[BeliefState] = None,
    max_depth: int = 5,
) -> PolicyTreeNode:
    """Build a finite-horizon policy tree from an alpha-vector policy.

    Parameters
    ----------
    pomdp : POMDP
    policy : POMDPPolicy
    initial_belief : BeliefState, optional
    max_depth : int

    Returns
    -------
    PolicyTreeNode
    """
    belief = initial_belief or pomdp.initial_belief
    updater = BeliefUpdater(pomdp)

    def _build(b: BeliefState, depth: int) -> PolicyTreeNode:
        action = policy.action(b)
        value = policy.value(b)
        node = PolicyTreeNode(action_id=action, value=value, depth=depth)

        if depth >= max_depth:
            return node

        for oid in pomdp.observation_ids:
            p_obs = updater.observation_likelihood(b, action, oid)
            if p_obs < 1e-10:
                continue
            b_prime = updater.update(b, action, oid)
            child = _build(b_prime, depth + 1)
            node.children[oid] = child

        return node

    return _build(belief, 0)
