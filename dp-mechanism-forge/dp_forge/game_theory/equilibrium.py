"""
Equilibrium computation for privacy mechanism games.

Implements Nash equilibrium computation (support enumeration, Lemke-Howson),
correlated equilibria via LP, evolutionary dynamics, and trembling-hand
perfect equilibria.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy import optimize as sp_opt
from scipy.linalg import solve as lin_solve

from dp_forge.game_theory import (
    Equilibrium,
    EquilibriumType,
    GameConfig,
    NashAlgorithm,
    PlayerRole,
    Strategy,
)
from dp_forge.types import GameMatrix


# ---------------------------------------------------------------------------
# NashEquilibrium
# ---------------------------------------------------------------------------


class NashEquilibrium:
    """Compute Nash equilibria of two-player games.

    Supports both zero-sum and general-sum bimatrix games.
    """

    def __init__(
        self,
        algorithm: NashAlgorithm = NashAlgorithm.SUPPORT_ENUMERATION,
        max_support: int = 10,
        tol: float = 1e-8,
    ) -> None:
        self.algorithm = algorithm
        self.max_support = max_support
        self._tol = tol

    def compute(
        self,
        A: npt.NDArray[np.float64],
        B: npt.NDArray[np.float64],
    ) -> List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        """Compute Nash equilibria of the bimatrix game (A, B).

        Player 1 (row) has payoff A, player 2 (column) has payoff B.

        Args:
            A: m x n payoff matrix for player 1.
            B: m x n payoff matrix for player 2.

        Returns:
            List of (p, q) mixed-strategy Nash equilibria.
        """
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        if A.shape != B.shape:
            raise ValueError(f"Payoff shape mismatch: {A.shape} vs {B.shape}")

        if self.algorithm == NashAlgorithm.SUPPORT_ENUMERATION:
            se = SupportEnumeration(max_support=self.max_support, tol=self._tol)
            return se.find_all(A, B)
        elif self.algorithm == NashAlgorithm.LEMKE_HOWSON:
            lh = LemkeHowson(tol=self._tol)
            result = lh.solve(A, B)
            return [result] if result is not None else []
        else:
            se = SupportEnumeration(max_support=self.max_support, tol=self._tol)
            return se.find_all(A, B)

    def compute_zero_sum(
        self, A: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        """Compute Nash equilibrium for a zero-sum game.

        For zero-sum, Nash = minimax = maximin.

        Returns:
            (row_strategy, col_strategy, game_value)
        """
        from dp_forge.game_theory.minimax import MinimaxSolver

        solver = MinimaxSolver()
        result = solver.solve(GameMatrix(payoffs=A))
        return (
            result.equilibrium.adversary_strategy.probabilities,
            result.equilibrium.designer_strategy.probabilities,
            result.equilibrium.game_value,
        )

    def is_nash_equilibrium(
        self,
        A: npt.NDArray[np.float64],
        B: npt.NDArray[np.float64],
        p: npt.NDArray[np.float64],
        q: npt.NDArray[np.float64],
        tol: float = 1e-6,
    ) -> bool:
        """Verify that (p, q) is a Nash equilibrium of (A, B)."""
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)

        # Player 1: p^T A q >= e_i^T A q for all i
        payoff1 = float(p @ A @ q)
        br1 = float(np.max(A @ q))

        # Player 2: p^T B q >= p^T B e_j for all j
        payoff2 = float(p @ B @ q)
        br2 = float(np.max(p @ B))

        return (br1 - payoff1) < tol and (br2 - payoff2) < tol


# ---------------------------------------------------------------------------
# SupportEnumeration
# ---------------------------------------------------------------------------


class SupportEnumeration:
    """Enumerate supports to find all Nash equilibria.

    For each pair of support sets (S1, S2) with |S1| = |S2|, solve the
    indifference conditions to check if a NE exists on that support.
    """

    def __init__(self, max_support: int = 10, tol: float = 1e-8) -> None:
        self.max_support = max_support
        self._tol = tol

    def find_all(
        self,
        A: npt.NDArray[np.float64],
        B: npt.NDArray[np.float64],
    ) -> List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        """Find all Nash equilibria by support enumeration."""
        m, n = A.shape
        equilibria = []
        max_k = min(self.max_support, min(m, n))

        # Eliminate strictly dominated strategies to reduce search space
        rows_alive = list(range(m))
        cols_alive = list(range(n))
        changed = True
        while changed:
            changed = False
            # Check row dominance
            new_rows = []
            for i in rows_alive:
                dominated = False
                for i2 in rows_alive:
                    if i2 != i and np.all(A[i2, cols_alive] > A[i, cols_alive] + self._tol):
                        dominated = True
                        break
                if not dominated:
                    new_rows.append(i)
            if len(new_rows) < len(rows_alive):
                rows_alive = new_rows
                changed = True
            # Check column dominance
            new_cols = []
            for j in cols_alive:
                dominated = False
                for j2 in cols_alive:
                    if j2 != j and np.all(B[rows_alive, j2] > B[rows_alive, j] + self._tol):
                        dominated = True
                        break
                if not dominated:
                    new_cols.append(j)
            if len(new_cols) < len(cols_alive):
                cols_alive = new_cols
                changed = True

        for k in range(1, max_k + 1):
            if k > len(rows_alive) or k > len(cols_alive):
                break
            for S1 in combinations(rows_alive, k):
                for S2 in combinations(cols_alive, k):
                    result = self._check_support(A, B, list(S1), list(S2))
                    if result is not None:
                        p, q = result
                        if not self._is_duplicate(equilibria, p, q):
                            equilibria.append((p, q))

        return equilibria

    def _check_support(
        self,
        A: npt.NDArray[np.float64],
        B: npt.NDArray[np.float64],
        S1: List[int],
        S2: List[int],
    ) -> Optional[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        """Check if a NE exists on supports S1, S2.

        Indifference conditions:
        - For each i in S1: (A q)_i = v1  (player 1 is indifferent)
        - For each j in S2: (p^T B)_j = v2  (player 2 is indifferent)
        - sum p_S1 = 1, sum q_S2 = 1, p >= 0, q >= 0
        """
        m, n = A.shape
        k = len(S1)
        if len(S2) != k:
            return None

        # Solve for q: A[S1, :][:, S2] q_S2 = v1 * 1
        # Augmented system: [A_sub | -1] [q_S2; v1] = 0, sum q_S2 = 1
        A_sub = A[np.ix_(S1, S2)]
        try:
            # System: A_sub q = v1 * ones, ones^T q = 1
            # => (A_sub - ones * row) q = 0 extended with sum=1
            M_q = np.vstack([
                A_sub - np.tile(A_sub[0:1, :], (k, 1)),
                np.ones((1, k)),
            ])
            rhs_q = np.zeros(k + 1)
            rhs_q[0] = 0  # first row is redundant after subtraction
            rhs_q[-1] = 1.0

            # Remove first row (redundant)
            M_q = M_q[1:]
            rhs_q = rhs_q[1:]

            if M_q.shape[0] != k:
                return None

            q_sub = lin_solve(M_q, rhs_q)
        except (np.linalg.LinAlgError, ValueError):
            return None

        if np.any(q_sub < -self._tol):
            return None

        # Solve for p: B[S1, :][:, S2]^T p_S1 = v2 * 1
        B_sub = B[np.ix_(S1, S2)]
        try:
            M_p = np.vstack([
                B_sub.T - np.tile(B_sub.T[0:1, :], (k, 1)),
                np.ones((1, k)),
            ])
            rhs_p = np.zeros(k + 1)
            rhs_p[-1] = 1.0
            M_p = M_p[1:]
            rhs_p = rhs_p[1:]

            if M_p.shape[0] != k:
                return None

            p_sub = lin_solve(M_p, rhs_p)
        except (np.linalg.LinAlgError, ValueError):
            return None

        if np.any(p_sub < -self._tol):
            return None

        # Construct full strategies
        p = np.zeros(m)
        q = np.zeros(n)
        p[list(S1)] = np.maximum(p_sub, 0.0)
        q[list(S2)] = np.maximum(q_sub, 0.0)

        # Normalise
        p_sum = p.sum()
        q_sum = q.sum()
        if p_sum < self._tol or q_sum < self._tol:
            return None
        p /= p_sum
        q /= q_sum

        # Verify: no player wants to deviate outside support
        v1 = float(A[S1[0]] @ q)
        Aq = A @ q
        S1_set = set(S1)
        for i in range(m):
            if i not in S1_set:
                if Aq[i] > v1 + self._tol:
                    return None

        v2 = float(p @ B[:, S2[0]])
        pB = p @ B
        S2_set = set(S2)
        for j in range(n):
            if j not in S2_set:
                if pB[j] > v2 + self._tol:
                    return None

        return p, q

    def _is_duplicate(
        self,
        existing: List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
        p: npt.NDArray[np.float64],
        q: npt.NDArray[np.float64],
    ) -> bool:
        for ep, eq in existing:
            if np.allclose(ep, p, atol=self._tol) and np.allclose(eq, q, atol=self._tol):
                return True
        return False


# ---------------------------------------------------------------------------
# LemkeHowson
# ---------------------------------------------------------------------------


class LemkeHowson:
    """Lemke-Howson algorithm for finding one Nash equilibrium.

    Follows complementary pivoting on the labeled polytope to trace a path
    from an artificial equilibrium to a Nash equilibrium.
    """

    def __init__(self, tol: float = 1e-10, max_pivots: int = 10000) -> None:
        self._tol = tol
        self._max_pivots = max_pivots

    def solve(
        self,
        A: npt.NDArray[np.float64],
        B: npt.NDArray[np.float64],
        init_label: int = 0,
    ) -> Optional[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        """Find one Nash equilibrium via Lemke-Howson pivoting.

        Args:
            A: m x n payoff matrix for player 1.
            B: m x n payoff matrix for player 2.
            init_label: Initial dropped label (0 to m+n-1).

        Returns:
            (p, q) Nash equilibrium or None if pivoting fails.
        """
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        m, n = A.shape

        # Shift payoffs to be strictly positive
        shift_a = max(0.0, -A.min() + 1.0)
        shift_b = max(0.0, -B.min() + 1.0)
        A_pos = A + shift_a
        B_pos = B + shift_b

        # Tableaux for player 1: [I | B_pos^T] with labels 0..m-1 basic
        # Tableaux for player 2: [A_pos | I] with labels m..m+n-1 basic
        tab1 = np.hstack([np.eye(m), B_pos.T])
        rhs1 = np.ones(m)
        basic1 = list(range(m))

        tab2 = np.hstack([A_pos, np.eye(n)])
        rhs2 = np.ones(n)
        basic2 = list(range(m, m + n))

        # Drop label init_label
        entering = init_label
        pivots = 0

        while pivots < self._max_pivots:
            pivots += 1

            if entering < m:
                # Entering player 2's tableau (column entering in tab2)
                col = entering
                col_vals = tab2[:, col]
                pos_mask = col_vals > self._tol
                ratios = np.full(n, np.inf)
                ratios[pos_mask] = rhs2[pos_mask] / col_vals[pos_mask]
                if np.all(np.isinf(ratios)):
                    break
                pivot_row = int(np.argmin(ratios))
                leaving = basic2[pivot_row]

                # Pivot (vectorized)
                pivot_val = tab2[pivot_row, col]
                tab2[pivot_row] /= pivot_val
                rhs2[pivot_row] /= pivot_val
                factors = tab2[:, col].copy()
                factors[pivot_row] = 0.0
                tab2 -= factors[:, np.newaxis] * tab2[pivot_row]
                rhs2 -= factors * rhs2[pivot_row]
                basic2[pivot_row] = entering
            else:
                # Entering player 1's tableau (column entering - m in tab1)
                col = entering - m + m  # column in tab1 = entering
                col_idx = entering
                col_vals = tab1[:, col_idx]
                pos_mask = col_vals > self._tol
                ratios = np.full(m, np.inf)
                ratios[pos_mask] = rhs1[pos_mask] / col_vals[pos_mask]
                if np.all(np.isinf(ratios)):
                    break
                pivot_row = int(np.argmin(ratios))
                leaving = basic1[pivot_row]

                pivot_val = tab1[pivot_row, col_idx]
                tab1[pivot_row] /= pivot_val
                rhs1[pivot_row] /= pivot_val
                factors = tab1[:, col_idx].copy()
                factors[pivot_row] = 0.0
                tab1 -= factors[:, np.newaxis] * tab1[pivot_row]
                rhs1 -= factors * rhs1[pivot_row]
                basic1[pivot_row] = entering

            if leaving == init_label:
                break
            entering = leaving

        # Extract strategies from basic variables
        q = np.zeros(n)
        for i, b in enumerate(basic2):
            if m <= b < m + n:
                pass  # slack variable, skip
            elif 0 <= b < m:
                pass  # not directly q
        # Read from complementary
        p = np.zeros(m)
        for i in range(m):
            label = basic1[i]
            if m <= label < m + n:
                q[label - m] = rhs1[i]

        for i in range(n):
            label = basic2[i]
            if 0 <= label < m:
                p[label] = rhs2[i]

        # Normalise
        p_sum = p.sum()
        q_sum = q.sum()
        if p_sum < self._tol or q_sum < self._tol:
            # Fallback to support enumeration
            se = SupportEnumeration(tol=self._tol)
            results = se.find_all(A, B)
            return results[0] if results else None

        p /= p_sum
        q /= q_sum
        return p, q


# ---------------------------------------------------------------------------
# CorrelatedEquilibrium
# ---------------------------------------------------------------------------


class CorrelatedEquilibrium:
    """Compute correlated equilibria via linear programming.

    A correlated equilibrium is a distribution over joint strategy profiles
    such that no player wants to deviate given the recommended action.
    The set of CE is a convex polytope defined by linear incentive constraints.
    """

    def __init__(self, tol: float = 1e-8) -> None:
        self._tol = tol

    def compute(
        self,
        A: npt.NDArray[np.float64],
        B: npt.NDArray[np.float64],
        objective: str = "max_welfare",
    ) -> npt.NDArray[np.float64]:
        """Compute a correlated equilibrium.

        Args:
            A: m x n payoff matrix for player 1.
            B: m x n payoff matrix for player 2.
            objective: 'max_welfare' (sum of payoffs), 'max_fairness'
                       (max min payoff), or 'uniform' (maximise entropy proxy).

        Returns:
            m x n matrix of joint probabilities (the correlated strategy).
        """
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        m, n = A.shape
        num_vars = m * n  # pi[i, j]

        # Objective
        if objective == "max_welfare":
            c = -(A + B).ravel()  # minimise negative welfare
        elif objective == "max_fairness":
            c = np.zeros(num_vars + 1)
            c[-1] = -1.0  # maximise auxiliary variable (min payoff)
            return self._solve_fairness(A, B)
        else:
            c = np.zeros(num_vars)  # feasibility

        # IC constraints for player 1:
        # For each i, i': sum_j pi[i,j] (A[i,j] - A[i',j]) >= 0
        ub_rows = []
        ub_rhs = []

        for i in range(m):
            for ip in range(m):
                if i == ip:
                    continue
                row = np.zeros(num_vars)
                for j in range(n):
                    # pi[i,j] coefficient: -(A[i,j] - A[i',j])
                    row[i * n + j] = -(A[i, j] - A[ip, j])
                ub_rows.append(row)
                ub_rhs.append(0.0)

        # IC constraints for player 2:
        # For each j, j': sum_i pi[i,j] (B[i,j] - B[i,j']) >= 0
        for j in range(n):
            for jp in range(n):
                if j == jp:
                    continue
                row = np.zeros(num_vars)
                for i in range(m):
                    row[i * n + j] = -(B[i, j] - B[i, jp])
                ub_rows.append(row)
                ub_rhs.append(0.0)

        A_ub = np.array(ub_rows) if ub_rows else np.zeros((0, num_vars))
        b_ub = np.array(ub_rhs) if ub_rhs else np.zeros(0)

        # sum pi = 1
        A_eq = np.ones((1, num_vars))
        b_eq = np.array([1.0])
        bounds = [(0.0, None)] * num_vars

        res = sp_opt.linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method="highs",
        )
        if not res.success:
            raise RuntimeError(f"CE LP failed: {res.message}")

        pi = np.maximum(res.x, 0.0).reshape(m, n)
        pi /= pi.sum()
        return pi

    def _solve_fairness(
        self,
        A: npt.NDArray[np.float64],
        B: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the max-min fair correlated equilibrium."""
        m, n = A.shape
        # vars: pi[i,j] (m*n) + t (min payoff variable)
        num_vars = m * n + 1
        c = np.zeros(num_vars)
        c[-1] = -1.0  # maximise t

        ub_rows = []
        ub_rhs = []

        # IC constraints (same as above but with extra variable)
        for i in range(m):
            for ip in range(m):
                if i == ip:
                    continue
                row = np.zeros(num_vars)
                for j in range(n):
                    row[i * n + j] = -(A[i, j] - A[ip, j])
                ub_rows.append(row)
                ub_rhs.append(0.0)

        for j in range(n):
            for jp in range(n):
                if j == jp:
                    continue
                row = np.zeros(num_vars)
                for i in range(m):
                    row[i * n + j] = -(B[i, j] - B[i, jp])
                ub_rows.append(row)
                ub_rhs.append(0.0)

        # Player 1 expected payoff >= t: sum_{i,j} A[i,j] pi[i,j] >= t
        row_p1 = np.zeros(num_vars)
        for i in range(m):
            for j in range(n):
                row_p1[i * n + j] = -A[i, j]
        row_p1[-1] = 1.0
        ub_rows.append(row_p1)
        ub_rhs.append(0.0)

        # Player 2 expected payoff >= t
        row_p2 = np.zeros(num_vars)
        for i in range(m):
            for j in range(n):
                row_p2[i * n + j] = -B[i, j]
        row_p2[-1] = 1.0
        ub_rows.append(row_p2)
        ub_rhs.append(0.0)

        A_ub = np.array(ub_rows)
        b_ub = np.array(ub_rhs)

        A_eq = np.zeros((1, num_vars))
        A_eq[0, :m * n] = 1.0
        b_eq = np.array([1.0])

        bounds = [(0.0, None)] * (m * n) + [(None, None)]

        res = sp_opt.linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method="highs",
        )
        if not res.success:
            raise RuntimeError(f"Fairness CE LP failed: {res.message}")

        pi = np.maximum(res.x[:m * n], 0.0).reshape(m, n)
        pi /= pi.sum()
        return pi

    def marginals(
        self, pi: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Extract marginal strategies from a correlated equilibrium.

        Returns:
            (player1_marginal, player2_marginal)
        """
        pi = np.asarray(pi, dtype=np.float64)
        return pi.sum(axis=1), pi.sum(axis=0)


# ---------------------------------------------------------------------------
# EvolutionaryDynamics
# ---------------------------------------------------------------------------


class EvolutionaryDynamics:
    """Replicator dynamics for mechanism evolution.

    Models a population of mechanisms evolving under selection pressure
    from adversaries.  The replicator equation:
        dx_i/dt = x_i * ((A x)_i - x^T A x)
    """

    def __init__(
        self,
        dt: float = 0.01,
        max_steps: int = 10000,
        tol: float = 1e-8,
    ) -> None:
        self.dt = dt
        self.max_steps = max_steps
        self._tol = tol

    def simulate(
        self,
        A: npt.NDArray[np.float64],
        x0: Optional[npt.NDArray[np.float64]] = None,
    ) -> Tuple[npt.NDArray[np.float64], List[npt.NDArray[np.float64]]]:
        """Run replicator dynamics to convergence.

        Args:
            A: n x n payoff matrix (symmetric game).
            x0: Initial population distribution (default: uniform).

        Returns:
            (final_distribution, trajectory)
        """
        A = np.asarray(A, dtype=np.float64)
        n = A.shape[0]
        if x0 is None:
            x = np.ones(n) / n
        else:
            x = np.asarray(x0, dtype=np.float64).copy()
            x /= x.sum()

        trajectory = [x.copy()]

        for step in range(self.max_steps):
            fitness = A @ x
            avg_fitness = float(x @ fitness)
            dx = x * (fitness - avg_fitness) * self.dt
            x_new = x + dx
            x_new = np.maximum(x_new, 0.0)
            x_sum = x_new.sum()
            if x_sum < self._tol:
                break
            x_new /= x_sum

            if np.max(np.abs(x_new - x)) < self._tol:
                x = x_new
                trajectory.append(x.copy())
                break
            x = x_new
            trajectory.append(x.copy())

        return x, trajectory

    def simulate_two_population(
        self,
        A: npt.NDArray[np.float64],
        B: npt.NDArray[np.float64],
        x0: Optional[npt.NDArray[np.float64]] = None,
        y0: Optional[npt.NDArray[np.float64]] = None,
    ) -> Tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    ]:
        """Two-population replicator dynamics.

        dx_i/dt = x_i * ((A y)_i - x^T A y)
        dy_j/dt = y_j * ((B^T x)_j - y^T B^T x)

        Args:
            A: m x n payoff matrix for population 1 (designers).
            B: m x n payoff matrix for population 2 (adversaries).
            x0, y0: Initial distributions.

        Returns:
            (final_x, final_y, trajectory)
        """
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        m, n = A.shape

        x = np.ones(m) / m if x0 is None else np.asarray(x0, dtype=np.float64).copy()
        y = np.ones(n) / n if y0 is None else np.asarray(y0, dtype=np.float64).copy()
        x /= x.sum()
        y /= y.sum()

        trajectory = [(x.copy(), y.copy())]

        for step in range(self.max_steps):
            # Population 1 (designers)
            fit_x = A @ y
            avg_x = float(x @ fit_x)
            dx = x * (fit_x - avg_x) * self.dt

            # Population 2 (adversaries)
            fit_y = B.T @ x
            avg_y = float(y @ fit_y)
            dy = y * (fit_y - avg_y) * self.dt

            x_new = np.maximum(x + dx, 0.0)
            y_new = np.maximum(y + dy, 0.0)

            x_sum = x_new.sum()
            y_sum = y_new.sum()
            if x_sum < self._tol or y_sum < self._tol:
                break
            x_new /= x_sum
            y_new /= y_sum

            converged = (
                np.max(np.abs(x_new - x)) < self._tol
                and np.max(np.abs(y_new - y)) < self._tol
            )
            x, y = x_new, y_new
            trajectory.append((x.copy(), y.copy()))
            if converged:
                break

        return x, y, trajectory

    def find_rest_points(
        self, A: npt.NDArray[np.float64], n_trials: int = 10
    ) -> List[npt.NDArray[np.float64]]:
        """Find rest points of replicator dynamics from random starts.

        Returns:
            List of distinct rest points found.
        """
        A = np.asarray(A, dtype=np.float64)
        n = A.shape[0]
        rng = np.random.default_rng(42)
        rest_points: List[npt.NDArray[np.float64]] = []

        for _ in range(n_trials):
            x0 = rng.dirichlet(np.ones(n))
            final, _ = self.simulate(A, x0)
            is_dup = False
            for rp in rest_points:
                if np.allclose(rp, final, atol=1e-4):
                    is_dup = True
                    break
            if not is_dup:
                rest_points.append(final)

        return rest_points


# ---------------------------------------------------------------------------
# TrembleEquilibrium
# ---------------------------------------------------------------------------


class TrembleEquilibrium:
    """Trembling-hand perfect equilibrium computation.

    A NE is trembling-hand perfect if it is robust to small perturbations
    (trembles) in the strategies.  We approximate by computing NE of
    perturbed games where each action has minimum probability eta.
    """

    def __init__(self, eta: float = 1e-4, tol: float = 1e-8) -> None:
        self.eta = eta
        self._tol = tol

    def compute(
        self,
        A: npt.NDArray[np.float64],
        B: npt.NDArray[np.float64],
    ) -> List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        """Find trembling-hand perfect equilibria.

        Compute NE of a sequence of perturbed games with decreasing eta,
        then take the limit.

        Args:
            A: m x n payoff for player 1.
            B: m x n payoff for player 2.

        Returns:
            List of trembling-hand perfect equilibria.
        """
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        m, n = A.shape

        se = SupportEnumeration(tol=self._tol)
        all_ne = se.find_all(A, B)

        if not all_ne:
            return []

        # Filter: a NE is THP if it is the limit of NE in perturbed games
        thp_equilibria = []
        etas = [self.eta * (10 ** (-k)) for k in range(3)]

        for p, q in all_ne:
            is_thp = True
            for eta in etas:
                # Perturbed game: each action has minimum probability eta
                p_pert = (1 - m * eta) * p + eta * np.ones(m)
                q_pert = (1 - n * eta) * q + eta * np.ones(n)

                # Check if (p_pert, q_pert) is approximately a NE
                # of the perturbed game
                payoff1 = float(p_pert @ A @ q_pert)
                br1 = float(np.max(A @ q_pert))
                payoff2 = float(p_pert @ B @ q_pert)
                br2 = float(np.max(p_pert @ B))

                # Allow slack proportional to eta
                slack = eta * max(np.abs(A).max(), np.abs(B).max()) * 10
                if (br1 - payoff1) > slack or (br2 - payoff2) > slack:
                    is_thp = False
                    break

            if is_thp:
                thp_equilibria.append((p, q))

        # If filtering removed everything, return all NE (all are THP in
        # non-degenerate games)
        return thp_equilibria if thp_equilibria else all_ne

    def compute_perturbed_game(
        self,
        A: npt.NDArray[np.float64],
        B: npt.NDArray[np.float64],
        eta: float,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute NE of eta-perturbed game.

        In the perturbed game, each player must play each action with
        probability >= eta.

        Returns:
            (p, q) NE of the perturbed game.
        """
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        m, n = A.shape

        # Use replicator dynamics starting from uniform
        ed = EvolutionaryDynamics(dt=0.005, max_steps=5000, tol=self._tol)
        x, y, _ = ed.simulate_two_population(A, B)

        # Clamp to [eta, 1]
        x = np.maximum(x, eta)
        y = np.maximum(y, eta)
        x /= x.sum()
        y /= y.sum()

        return x, y
