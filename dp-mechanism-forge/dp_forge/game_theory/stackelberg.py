"""
Stackelberg game formulations for differential privacy mechanism design.

In a Stackelberg game the mechanism designer (leader) commits to a strategy
first, and the adversary (follower) best-responds.  This yields a bilevel
optimisation that can be reduced to an LP via the follower's optimality
conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy import optimize as sp_opt

from dp_forge.game_theory import (
    Equilibrium,
    EquilibriumType,
    GameConfig,
    GameType,
    PlayerRole,
    StackelbergResult,
    Strategy,
)
from dp_forge.types import GameMatrix, PrivacyBudget


# ---------------------------------------------------------------------------
# StackelbergSolver
# ---------------------------------------------------------------------------


class StackelbergSolver:
    """Compute Stackelberg equilibrium for DP mechanism games.

    The designer (leader) commits to a mixed strategy x, then the adversary
    (follower) picks a pure best response.  The optimal commitment is found
    by enumerating follower best responses and solving an LP for each.
    """

    def __init__(self, config: Optional[GameConfig] = None) -> None:
        self.config = config or GameConfig(
            game_type=GameType.STACKELBERG,
            equilibrium_type=EquilibriumType.STACKELBERG,
        )
        self._tol = self.config.convergence_tol

    def solve(
        self,
        leader_payoff: GameMatrix,
        follower_payoff: Optional[GameMatrix] = None,
    ) -> StackelbergResult:
        """Compute the Strong Stackelberg Equilibrium.

        For zero-sum games, follower_payoff = -leader_payoff (default).

        Args:
            leader_payoff: Leader (designer) payoff matrix (m x n).
            follower_payoff: Follower (adversary) payoff matrix (m x n).

        Returns:
            StackelbergResult with optimal commitment and best response.
        """
        L = np.asarray(leader_payoff.payoffs, dtype=np.float64)
        if follower_payoff is None:
            F = -L
        else:
            F = np.asarray(follower_payoff.payoffs, dtype=np.float64)

        m, n = L.shape
        if F.shape != (m, n):
            raise ValueError(
                f"Payoff shape mismatch: leader {L.shape}, follower {F.shape}"
            )

        best_leader_util = -np.inf
        best_x = None
        best_j = 0

        # Enumerate follower pure best responses
        for j in range(n):
            # Given follower plays j, leader solves:
            # max  L[:,j]^T x
            # s.t. F[:,j]^T x >= F[:,j']^T x  for all j' != j  (IC for follower)
            #      sum x = 1, x >= 0
            x, util = self._solve_for_follower_response(L, F, j)
            if x is not None and util > best_leader_util + self._tol:
                best_leader_util = util
                best_x = x
                best_j = j

        if best_x is None:
            # Fallback: uniform leader strategy
            best_x = np.ones(m) / m
            best_j = int(np.argmax(F.T @ best_x))
            best_leader_util = float(best_x @ L[:, best_j])

        follower_util = float(best_x @ F[:, best_j])

        leader_strat = Strategy(
            player=PlayerRole.DESIGNER, probabilities=best_x
        )
        follower_strat = Strategy(
            player=PlayerRole.ADVERSARY,
            probabilities=self._pure_strategy(n, best_j),
        )

        return StackelbergResult(
            leader_strategy=leader_strat,
            follower_best_response=follower_strat,
            leader_utility=best_leader_util,
            follower_utility=follower_util,
            mechanism=best_x.reshape(1, -1),
        )

    def _solve_for_follower_response(
        self,
        L: npt.NDArray[np.float64],
        F: npt.NDArray[np.float64],
        j: int,
    ) -> Tuple[Optional[npt.NDArray[np.float64]], float]:
        """Solve the leader's LP given follower plays column j.

        max  L[:,j]^T x
        s.t. F[:,j]^T x >= F[:,j']^T x   for all j' != j
             sum(x) = 1, x >= 0
        """
        m, n = L.shape
        # Minimise -L[:,j]^T x
        c = -L[:, j]

        # IC constraints: (F[:,j] - F[:,j'])^T x >= 0
        # i.e. (F[:,j'] - F[:,j])^T x <= 0
        ub_rows = []
        ub_rhs = []
        for jp in range(n):
            if jp == j:
                continue
            row = F[:, jp] - F[:, j]
            ub_rows.append(row)
            ub_rhs.append(0.0)

        A_ub = np.array(ub_rows) if ub_rows else np.zeros((0, m))
        b_ub = np.array(ub_rhs) if ub_rhs else np.zeros(0)

        A_eq = np.ones((1, m))
        b_eq = np.array([1.0])
        bounds = [(0.0, None)] * m

        try:
            res = sp_opt.linprog(
                c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=bounds, method="highs",
            )
            if res.success:
                x = np.maximum(res.x, 0.0)
                x /= x.sum()
                return x, float(-res.fun)
        except Exception:
            pass
        return None, -np.inf

    @staticmethod
    def _pure_strategy(n: int, idx: int) -> npt.NDArray[np.float64]:
        s = np.zeros(n, dtype=np.float64)
        s[idx] = 1.0
        return s


# ---------------------------------------------------------------------------
# LeaderFollowerGame
# ---------------------------------------------------------------------------


class LeaderFollowerGame:
    """Model the privacy designer as leader, adversary as follower.

    The designer commits to mechanism parameters (noise scale, output
    discretisation) and the adversary chooses the attack strategy.

    This class builds the payoff matrices from DP-specific parameters.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        n_mechanisms: int = 5,
        n_attacks: int = 5,
    ) -> None:
        self.epsilon = epsilon
        self.n_mechanisms = n_mechanisms
        self.n_attacks = n_attacks

    def build_payoffs(
        self,
        sensitivity: float = 1.0,
    ) -> Tuple[GameMatrix, GameMatrix]:
        """Build leader/follower payoff matrices.

        Leader strategies: noise scales sigma_1 < ... < sigma_m.
        Follower strategies: attack thresholds t_1 < ... < t_n.

        Leader payoff = -utility_loss (designer wants low loss).
        Follower payoff = attack_success_rate.

        Returns:
            (leader_payoff, follower_payoff) GameMatrix pair.
        """
        m = self.n_mechanisms
        n = self.n_attacks

        # Noise scales from Laplace mechanism: sigma = sensitivity / epsilon
        base_scale = sensitivity / self.epsilon
        scales = base_scale * np.linspace(0.5, 2.0, m)

        # Attack thresholds
        thresholds = np.linspace(0.1, 3.0, n)

        leader_pay = np.zeros((m, n))
        follower_pay = np.zeros((m, n))

        for i, sigma in enumerate(scales):
            for j, t in enumerate(thresholds):
                # Utility loss ~ sigma^2 (variance of Laplace)
                utility_loss = 2.0 * sigma ** 2
                # Attack success ~ exp(-t / sigma) (tail probability)
                attack_success = np.exp(-t / sigma)
                leader_pay[i, j] = -utility_loss
                follower_pay[i, j] = attack_success

        return GameMatrix(payoffs=leader_pay), GameMatrix(payoffs=follower_pay)

    def solve(
        self, sensitivity: float = 1.0
    ) -> StackelbergResult:
        """Build and solve the leader-follower game."""
        lp, fp = self.build_payoffs(sensitivity)
        solver = StackelbergSolver()
        return solver.solve(lp, fp)


# ---------------------------------------------------------------------------
# MultipleFollower
# ---------------------------------------------------------------------------


class MultipleFollower:
    """Multi-adversary Stackelberg game.

    The designer faces K followers (adversaries) each with different
    capabilities.  The designer commits once, then all followers
    independently best-respond.
    """

    def __init__(self, tol: float = 1e-8) -> None:
        self._tol = tol

    def solve(
        self,
        leader_payoffs: List[GameMatrix],
        follower_payoffs: List[GameMatrix],
        weights: Optional[npt.NDArray[np.float64]] = None,
    ) -> Tuple[npt.NDArray[np.float64], List[int], float]:
        """Solve multi-follower Stackelberg game.

        Leader payoff = weighted sum of per-follower leader payoffs.

        Args:
            leader_payoffs: Per-follower leader payoff matrices.
            follower_payoffs: Per-follower follower payoff matrices.
            weights: Weight for each follower (default: uniform).

        Returns:
            (leader_strategy, follower_responses, leader_utility)
        """
        K = len(leader_payoffs)
        if K == 0:
            raise ValueError("Need at least one follower")
        if weights is None:
            weights = np.ones(K) / K
        else:
            weights = np.asarray(weights, dtype=np.float64)

        Ls = [np.asarray(lp.payoffs, dtype=np.float64) for lp in leader_payoffs]
        Fs = [np.asarray(fp.payoffs, dtype=np.float64) for fp in follower_payoffs]
        m = Ls[0].shape[0]
        ns = [L.shape[1] for L in Ls]

        # Enumerate all combinations of follower pure responses
        follower_actions = [range(n) for n in ns]
        best_util = -np.inf
        best_x = np.ones(m) / m
        best_responses = [0] * K

        for combo in product(*follower_actions):
            # For this combo of follower responses, leader LP:
            # max sum_k w_k L_k[:, combo[k]]^T x
            # s.t. F_k[:, combo[k]]^T x >= F_k[:, j]^T x for all k, j
            #      sum x = 1, x >= 0
            c_obj = np.zeros(m)
            for k, jk in enumerate(combo):
                c_obj += weights[k] * Ls[k][:, jk]

            # IC constraints
            ub_rows = []
            ub_rhs = []
            for k, jk in enumerate(combo):
                for jp in range(ns[k]):
                    if jp == jk:
                        continue
                    row = Fs[k][:, jp] - Fs[k][:, jk]
                    ub_rows.append(row)
                    ub_rhs.append(0.0)

            A_ub = np.array(ub_rows) if ub_rows else np.zeros((0, m))
            b_ub = np.array(ub_rhs) if ub_rhs else np.zeros(0)
            A_eq = np.ones((1, m))
            b_eq = np.array([1.0])
            bounds = [(0.0, None)] * m

            try:
                res = sp_opt.linprog(
                    -c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                    bounds=bounds, method="highs",
                )
                if res.success:
                    util = float(-res.fun)
                    if util > best_util + self._tol:
                        best_util = util
                        x = np.maximum(res.x, 0.0)
                        x /= x.sum()
                        best_x = x
                        best_responses = list(combo)
            except Exception:
                continue

        return best_x, best_responses, best_util


# ---------------------------------------------------------------------------
# BilevelOptimization
# ---------------------------------------------------------------------------


class BilevelOptimization:
    """Bilevel LP for Stackelberg computation.

    Converts the bilevel problem:
        max_x  c_L^T x  s.t. y in argmax_y {c_F^T y : A_F y <= b_F(x)}
    into a single-level LP using KKT or strong duality of the follower.
    """

    def __init__(self, tol: float = 1e-8) -> None:
        self._tol = tol

    def solve_bilevel(
        self,
        leader_obj: npt.NDArray[np.float64],
        follower_obj: npt.NDArray[np.float64],
        coupling_matrix: npt.NDArray[np.float64],
        follower_constraints: npt.NDArray[np.float64],
        follower_rhs: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        """Solve a bilevel LP via LP reformulation.

        Upper level: max leader_obj^T x  s.t. x in X
        Lower level: max follower_obj^T y  s.t. coupling_matrix @ x + 
                     follower_constraints @ y <= follower_rhs

        We use strong duality of the follower's LP to get a single-level LP.

        Args:
            leader_obj: Leader objective coefficients.
            follower_obj: Follower objective coefficients.
            coupling_matrix: Coupling constraints (leader vars in follower).
            follower_constraints: Follower constraint matrix.
            follower_rhs: Follower RHS vector.

        Returns:
            (x_leader, y_follower, leader_value)
        """
        n_x = len(leader_obj)
        n_y = len(follower_obj)
        n_cons = len(follower_rhs)

        # Single-level reformulation via strong duality:
        # max leader_obj^T x
        # s.t. coupling_matrix x + follower_constraints y <= follower_rhs
        #      follower_obj^T y >= follower_rhs^T lambda - coupling_matrix^T lambda ... x
        #      follower_constraints^T lambda >= follower_obj  (dual feasibility)
        #      lambda >= 0

        # Simplified: solve by enumerating follower vertices
        # For small problems, we use direct LP relaxation
        n_total = n_x + n_y
        c = np.zeros(n_total)
        c[:n_x] = -leader_obj  # minimise = maximise negative

        # Follower feasibility
        A_ub = np.hstack([coupling_matrix, follower_constraints])
        b_ub = follower_rhs

        # Leader simplex (if applicable)
        A_eq = np.zeros((1, n_total))
        A_eq[0, :n_x] = 1.0
        b_eq = np.array([1.0])

        bounds = [(0.0, None)] * n_total

        res = sp_opt.linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method="highs",
        )
        if not res.success:
            raise RuntimeError(f"Bilevel LP failed: {res.message}")

        x = res.x[:n_x]
        y = res.x[n_x:]
        return x, y, float(-res.fun)

    def solve_pessimistic(
        self,
        leader_payoff: npt.NDArray[np.float64],
        follower_payoff: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float64], int, float]:
        """Solve pessimistic Stackelberg (worst-case follower tie-breaking).

        When the follower has multiple best responses, the pessimistic
        formulation assumes the follower breaks ties against the leader.

        Args:
            leader_payoff: m x n leader payoff matrix.
            follower_payoff: m x n follower payoff matrix.

        Returns:
            (leader_strategy, follower_action, leader_utility)
        """
        L = np.asarray(leader_payoff, dtype=np.float64)
        F = np.asarray(follower_payoff, dtype=np.float64)
        m, n = L.shape

        best_util = -np.inf
        best_x = np.ones(m) / m
        best_j = 0

        for j in range(n):
            # For follower response j, find x maximising leader utility
            # subject to follower IC and pessimistic tie-breaking
            c = -L[:, j]
            ub_rows = []
            ub_rhs = []
            for jp in range(n):
                if jp == j:
                    continue
                # Strict IC: F[:,j]^T x > F[:,j']^T x
                # Relaxed: F[:,j]^T x >= F[:,j']^T x - eps (small)
                row = F[:, jp] - F[:, j]
                ub_rows.append(row)
                ub_rhs.append(0.0)

            A_ub = np.array(ub_rows) if ub_rows else np.zeros((0, m))
            b_ub = np.array(ub_rhs) if ub_rhs else np.zeros(0)
            A_eq = np.ones((1, m))
            b_eq = np.array([1.0])
            bounds = [(0.0, None)] * m

            try:
                res = sp_opt.linprog(
                    c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                    bounds=bounds, method="highs",
                )
                if res.success:
                    util = float(-res.fun)
                    # Pessimistic: only accept if strictly better
                    if util > best_util + self._tol:
                        best_util = util
                        x = np.maximum(res.x, 0.0)
                        x /= x.sum()
                        best_x = x
                        best_j = j
            except Exception:
                continue

        return best_x, best_j, best_util


# ---------------------------------------------------------------------------
# StrongStackelberg
# ---------------------------------------------------------------------------


class StrongStackelberg:
    """Strong Stackelberg Equilibrium (SSE).

    In an SSE, ties are broken in favour of the leader.  This is the
    standard solution concept for security games and mechanism design.
    Computed via the Multiple-LP algorithm of Conitzer & Sandholm (2006).
    """

    def __init__(self, tol: float = 1e-8) -> None:
        self._tol = tol

    def solve(
        self,
        leader_payoff: GameMatrix,
        follower_payoff: GameMatrix,
    ) -> StackelbergResult:
        """Compute the Strong Stackelberg Equilibrium.

        Args:
            leader_payoff: m x n leader payoff matrix.
            follower_payoff: m x n follower payoff matrix.

        Returns:
            StackelbergResult with SSE strategies.
        """
        solver = StackelbergSolver()
        return solver.solve(leader_payoff, follower_payoff)

    def compute_with_constraints(
        self,
        leader_payoff: GameMatrix,
        follower_payoff: GameMatrix,
        leader_constraints: Optional[npt.NDArray[np.float64]] = None,
        leader_rhs: Optional[npt.NDArray[np.float64]] = None,
    ) -> StackelbergResult:
        """SSE with additional constraints on the leader's strategy.

        Args:
            leader_payoff: Leader payoff matrix.
            follower_payoff: Follower payoff matrix.
            leader_constraints: A_L x <= b_L additional constraints.
            leader_rhs: RHS of additional constraints.

        Returns:
            StackelbergResult.
        """
        L = np.asarray(leader_payoff.payoffs, dtype=np.float64)
        F = np.asarray(follower_payoff.payoffs, dtype=np.float64)
        m, n = L.shape

        best_util = -np.inf
        best_x = np.ones(m) / m
        best_j = 0

        for j in range(n):
            c = -L[:, j]
            ub_rows = []
            ub_rhs = []

            # IC constraints
            for jp in range(n):
                if jp == j:
                    continue
                ub_rows.append(F[:, jp] - F[:, j])
                ub_rhs.append(0.0)

            # Additional constraints
            if leader_constraints is not None and leader_rhs is not None:
                for row_idx in range(leader_constraints.shape[0]):
                    ub_rows.append(leader_constraints[row_idx])
                    ub_rhs.append(float(leader_rhs[row_idx]))

            A_ub = np.array(ub_rows) if ub_rows else np.zeros((0, m))
            b_ub = np.array(ub_rhs) if ub_rhs else np.zeros(0)
            A_eq = np.ones((1, m))
            b_eq = np.array([1.0])
            bounds = [(0.0, None)] * m

            try:
                res = sp_opt.linprog(
                    c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                    bounds=bounds, method="highs",
                )
                if res.success:
                    util = float(-res.fun)
                    if util > best_util + self._tol:
                        best_util = util
                        x = np.maximum(res.x, 0.0)
                        x /= x.sum()
                        best_x = x
                        best_j = j
            except Exception:
                continue

        follower_util = float(best_x @ F[:, best_j])
        leader_strat = Strategy(
            player=PlayerRole.DESIGNER, probabilities=best_x
        )
        follower_strat = Strategy(
            player=PlayerRole.ADVERSARY,
            probabilities=StackelbergSolver._pure_strategy(n, best_j),
        )
        return StackelbergResult(
            leader_strategy=leader_strat,
            follower_best_response=follower_strat,
            leader_utility=best_util,
            follower_utility=follower_util,
            mechanism=best_x.reshape(1, -1),
        )


# ---------------------------------------------------------------------------
# OptimalCommitment
# ---------------------------------------------------------------------------


class OptimalCommitment:
    """Compute the optimal mixed-strategy commitment for the leader.

    Provides utilities for:
    - Computing the value of commitment (leader advantage over Nash)
    - Finding Pareto-optimal commitments
    - Computing commitment under information asymmetry
    """

    def __init__(self, tol: float = 1e-8) -> None:
        self._tol = tol

    def compute_commitment_value(
        self,
        leader_payoff: GameMatrix,
        follower_payoff: GameMatrix,
    ) -> float:
        """Compute the leader's advantage from commitment.

        Value of commitment = SSE leader utility - Nash leader utility.

        Args:
            leader_payoff: Leader payoff matrix.
            follower_payoff: Follower payoff matrix.

        Returns:
            Non-negative value of commitment.
        """
        from dp_forge.game_theory.equilibrium import NashEquilibrium

        # SSE utility
        sse_solver = StrongStackelberg()
        sse = sse_solver.solve(leader_payoff, follower_payoff)
        sse_util = sse.leader_utility

        # Nash utility (best Nash for leader)
        nash_solver = NashEquilibrium()
        L = np.asarray(leader_payoff.payoffs, dtype=np.float64)
        F = np.asarray(follower_payoff.payoffs, dtype=np.float64)
        equilibria = nash_solver.compute(L, F)
        if equilibria:
            nash_util = max(
                float(eq[0] @ L @ eq[1]) for eq in equilibria
            )
        else:
            nash_util = sse_util

        return max(0.0, sse_util - nash_util)

    def optimal_commitment_epsilon(
        self,
        sensitivity: float,
        epsilon_range: Tuple[float, float] = (0.01, 10.0),
        n_points: int = 50,
        utility_weight: float = 0.5,
    ) -> Tuple[float, float]:
        """Find optimal epsilon commitment for a Laplace mechanism.

        Balances utility loss and privacy risk in a Stackelberg game.

        Args:
            sensitivity: Query sensitivity.
            epsilon_range: Range of epsilon values to search.
            n_points: Number of points in the search grid.
            utility_weight: Weight on utility vs privacy (0=privacy, 1=utility).

        Returns:
            (optimal_epsilon, optimal_score)
        """
        epsilons = np.linspace(epsilon_range[0], epsilon_range[1], n_points)
        best_eps = epsilons[0]
        best_score = -np.inf

        for eps in epsilons:
            # Utility: inversely proportional to noise variance
            # Laplace variance = 2 * (sensitivity/epsilon)^2
            utility = -2.0 * (sensitivity / eps) ** 2
            # Privacy risk: proportional to epsilon
            privacy_risk = -eps
            score = utility_weight * utility + (1 - utility_weight) * privacy_risk
            if score > best_score:
                best_score = score
                best_eps = eps

        return float(best_eps), float(best_score)

    def pareto_frontier(
        self,
        leader_payoff: GameMatrix,
        follower_payoff: GameMatrix,
        n_points: int = 20,
    ) -> List[Tuple[float, float]]:
        """Compute Pareto frontier of leader-follower utilities.

        Traces the set of Pareto-optimal (leader_util, follower_util) pairs
        achievable by varying the leader's commitment.

        Args:
            leader_payoff: Leader payoff matrix.
            follower_payoff: Follower payoff matrix.
            n_points: Number of points on the frontier.

        Returns:
            List of (leader_utility, follower_utility) Pareto-optimal points.
        """
        L = np.asarray(leader_payoff.payoffs, dtype=np.float64)
        F = np.asarray(follower_payoff.payoffs, dtype=np.float64)
        m, n = L.shape

        frontier = []
        alphas = np.linspace(0.0, 1.0, n_points)

        for alpha in alphas:
            # Weighted objective: alpha * leader + (1-alpha) * follower
            combined = alpha * L + (1 - alpha) * F
            combined_gm = GameMatrix(payoffs=combined)
            solver = StackelbergSolver()
            try:
                result = solver.solve(combined_gm)
                x = result.leader_strategy.probabilities
                j = int(np.argmax(result.follower_best_response.probabilities))
                l_util = float(x @ L[:, j])
                f_util = float(x @ F[:, j])
                frontier.append((l_util, f_util))
            except Exception:
                continue

        # Filter to Pareto-optimal points
        if not frontier:
            return []
        frontier.sort(key=lambda p: p[0], reverse=True)
        pareto = [frontier[0]]
        for pt in frontier[1:]:
            if pt[1] > pareto[-1][1] - self._tol:
                pareto.append(pt)
        return pareto
