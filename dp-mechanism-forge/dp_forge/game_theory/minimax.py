"""
Minimax mechanism design for differential privacy.

Formulates DP mechanism synthesis as a two-player zero-sum game where the
designer minimises worst-case utility loss and the adversary maximises it
by choosing the worst-case adjacent database pair.  Solved via LP duality
(von Neumann's minimax theorem).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy import optimize as sp_opt
from scipy import sparse

from dp_forge.game_theory import (
    Equilibrium,
    EquilibriumType,
    GameConfig,
    GameResult,
    GameType,
    PlayerRole,
    Strategy,
)
from dp_forge.types import (
    AdjacencyRelation,
    GameMatrix,
    OptimalityCertificate,
    PrivacyBudget,
    QuerySpec,
    SolverBackend,
)


# ---------------------------------------------------------------------------
# MinimaxSolver
# ---------------------------------------------------------------------------


class MinimaxSolver:
    """Solve the minimax (zero-sum) game for worst-case optimal mechanism.

    Given a payoff matrix A (m x n) the designer (row player) minimises
    max_j (Ax)_j  and the adversary (column player) maximises min_i (y^T A)_i.
    Von Neumann's minimax theorem guarantees these values coincide.

    We solve the primal LP:
        min  v
        s.t. sum_j A[i,j] x[j] <= v   for all i
             sum_j x[j] = 1, x >= 0

    and the dual LP:
        max  w
        s.t. sum_i y[i] A[i,j] >= w   for all j
             sum_i y[i] = 1, y >= 0
    """

    def __init__(self, config: Optional[GameConfig] = None) -> None:
        self.config = config or GameConfig(
            equilibrium_type=EquilibriumType.MINIMAX
        )
        self._tol = self.config.convergence_tol

    def solve(self, game: GameMatrix) -> GameResult:
        """Compute the minimax equilibrium of a zero-sum game."""
        A = np.asarray(game.payoffs, dtype=np.float64)
        if A.ndim != 2:
            raise ValueError(f"Payoff matrix must be 2-D, got shape {A.shape}")
        m, n = A.shape

        # Solve designer (column) LP: min v s.t. Ax <= v*1, 1^T x = 1, x >= 0
        designer_strategy, game_value_primal = self._solve_column_lp(A)
        # Solve adversary (row) LP:   max w s.t. y^T A >= w*1, 1^T y = 1, y >= 0
        adversary_strategy, game_value_dual = self._solve_row_lp(A)

        duality_gap = abs(game_value_primal - game_value_dual)
        game_value = 0.5 * (game_value_primal + game_value_dual)

        designer_strat = Strategy(
            player=PlayerRole.DESIGNER,
            probabilities=designer_strategy,
        )
        adversary_strat = Strategy(
            player=PlayerRole.ADVERSARY,
            probabilities=adversary_strategy,
        )

        eq = Equilibrium(
            equilibrium_type=EquilibriumType.MINIMAX,
            designer_strategy=designer_strat,
            adversary_strategy=adversary_strat,
            game_value=game_value,
            is_exact=(duality_gap < self._tol),
            approximation_error=duality_gap if duality_gap >= self._tol else 0.0,
        )

        cert = OptimalityCertificate(
            dual_vars=adversary_strategy,
            duality_gap=duality_gap,
            primal_obj=game_value_primal,
            dual_obj=game_value_dual,
        )

        worst_pair = self._worst_case_pair(A, adversary_strategy)

        return GameResult(
            mechanism=designer_strategy.reshape(1, -1),
            equilibrium=eq,
            game_matrix=game,
            iterations=0,
            optimality_certificate=cert,
            worst_case_pair=worst_pair,
        )

    def _solve_column_lp(
        self, A: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], float]:
        """Solve the designer's (column player) minimax LP.

        Variables: x_1..x_n, v.  Min v s.t. A x <= v 1, 1^T x = 1, x >= 0.
        """
        m, n = A.shape
        # Decision vector: [x_1, ..., x_n, v]
        c = np.zeros(n + 1)
        c[-1] = 1.0  # minimise v

        # Inequality: A x - v 1 <= 0  =>  [A | -1] [x; v] <= 0
        A_ub = np.hstack([A, -np.ones((m, 1))])
        b_ub = np.zeros(m)

        # Equality: sum x = 1
        A_eq = np.zeros((1, n + 1))
        A_eq[0, :n] = 1.0
        b_eq = np.array([1.0])

        bounds = [(0.0, None)] * n + [(None, None)]

        res = sp_opt.linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method="highs",
        )
        if not res.success:
            raise RuntimeError(f"Designer LP failed: {res.message}")

        x = res.x[:n]
        x = np.maximum(x, 0.0)
        x /= x.sum()
        return x, float(res.x[-1])

    def _solve_row_lp(
        self, A: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], float]:
        """Solve the adversary's (row player) maximin LP.

        max w s.t. y^T A >= w 1^T, 1^T y = 1, y >= 0
        Converted to min (-w) s.t. -A^T y + w 1 <= 0, ...
        """
        m, n = A.shape
        # Decision vector: [y_1, ..., y_m, w]
        c = np.zeros(m + 1)
        c[-1] = -1.0  # maximise w => minimise -w

        # -A^T y + w 1 <= 0
        A_ub = np.hstack([-A.T, np.ones((n, 1))])
        b_ub = np.zeros(n)

        A_eq = np.zeros((1, m + 1))
        A_eq[0, :m] = 1.0
        b_eq = np.array([1.0])

        bounds = [(0.0, None)] * m + [(None, None)]

        res = sp_opt.linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method="highs",
        )
        if not res.success:
            raise RuntimeError(f"Adversary LP failed: {res.message}")

        y = res.x[:m]
        y = np.maximum(y, 0.0)
        y /= y.sum()
        return y, float(res.x[-1])

    def _worst_case_pair(
        self, A: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> Tuple[int, int]:
        """Identify the adversary's worst-case pair of rows."""
        m = A.shape[0]
        if m < 2:
            return (0, 0)
        best_pair = (0, 1)
        best_diff = -np.inf
        for i in range(m):
            for j in range(i + 1, m):
                diff = abs(float(y[i]) - float(y[j]))
                if diff > best_diff:
                    best_diff = diff
                    best_pair = (i, j)
        return best_pair


# ---------------------------------------------------------------------------
# SaddlePointComputation
# ---------------------------------------------------------------------------


class SaddlePointComputation:
    """Find saddle points of a payoff matrix via LP duality.

    A saddle point (i*, j*) satisfies:
        A[i*, j] <= A[i*, j*] <= A[i, j*]  for all i, j
    i.e. it is simultaneously a row minimum and column maximum.
    """

    def __init__(self, tol: float = 1e-10) -> None:
        self._tol = tol

    def find_pure_saddle_points(
        self, A: npt.NDArray[np.float64]
    ) -> List[Tuple[int, int]]:
        """Find all pure-strategy saddle points."""
        A = np.asarray(A, dtype=np.float64)
        m, n = A.shape
        row_mins = A.min(axis=1)  # min over columns for each row
        col_maxs = A.max(axis=0)  # max over rows for each column
        saddle_points = []
        for i in range(m):
            for j in range(n):
                if (abs(A[i, j] - row_mins[i]) < self._tol and
                        abs(A[i, j] - col_maxs[j]) < self._tol):
                    saddle_points.append((i, j))
        return saddle_points

    def find_mixed_saddle_point(
        self, A: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        """Find the mixed-strategy saddle point via LP.

        Returns:
            (row_strategy, col_strategy, game_value)
        """
        solver = MinimaxSolver()
        game = GameMatrix(payoffs=A)
        result = solver.solve(game)
        return (
            result.equilibrium.adversary_strategy.probabilities,
            result.equilibrium.designer_strategy.probabilities,
            result.equilibrium.game_value,
        )

    def verify_saddle_point(
        self,
        A: npt.NDArray[np.float64],
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        tol: float = 1e-6,
    ) -> bool:
        """Verify that (x, y) is a saddle point of A.

        Check: x^T A y >= x'^T A y for all x'  (y is best resp for row)
        and    x^T A y <= x^T A y' for all y'  (x is best resp for col)
        Equivalent to: max_i (A y)_i <= x^T A y <= min_j (x^T A)_j
        """
        A = np.asarray(A, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        val = float(x @ A @ y)
        # Best response values
        row_br = float(np.max(A @ y))  # max over rows
        col_br = float(np.min(x @ A))  # min over cols
        return (row_br - val) < tol and (val - col_br) < tol


# ---------------------------------------------------------------------------
# MaxMinFair
# ---------------------------------------------------------------------------


class MaxMinFair:
    """Max-min fair noise allocation across multiple queries.

    Given a set of queries each requiring DP guarantees, allocate the
    privacy budget to maximise the minimum utility across all queries.
    This is a max-min fairness criterion solved via LP.
    """

    def __init__(self, tol: float = 1e-8) -> None:
        self._tol = tol

    def allocate(
        self,
        sensitivities: npt.NDArray[np.float64],
        total_epsilon: float,
        total_delta: float = 0.0,
    ) -> npt.NDArray[np.float64]:
        """Compute max-min fair epsilon allocation.

        Args:
            sensitivities: Sensitivity of each query (length q).
            total_epsilon: Total privacy budget.
            total_delta: Total delta budget (unused for pure DP).

        Returns:
            Per-query epsilon allocation (length q).
        """
        s = np.asarray(sensitivities, dtype=np.float64)
        q = len(s)
        if q == 0:
            return np.array([], dtype=np.float64)

        # For Laplace mechanism, variance ~ (s_i / eps_i)^2.
        # Max-min fair utility => min variance is maximized.
        # Optimal: eps_i proportional to s_i.
        eps = total_epsilon * s / s.sum()
        return eps

    def allocate_with_weights(
        self,
        sensitivities: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
        total_epsilon: float,
    ) -> npt.NDArray[np.float64]:
        """Weighted max-min fair allocation.

        Args:
            sensitivities: Per-query sensitivities.
            weights: Importance weights per query.
            total_epsilon: Total budget.

        Returns:
            Per-query epsilon allocation.
        """
        s = np.asarray(sensitivities, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        q = len(s)
        if q == 0:
            return np.array([], dtype=np.float64)

        # Weighted allocation: eps_i proportional to w_i * s_i
        ws = w * s
        eps = total_epsilon * ws / ws.sum()
        return eps

    def compute_fairness_index(
        self, allocations: npt.NDArray[np.float64]
    ) -> float:
        """Jain's fairness index for the allocation.

        Returns a value in (0, 1] where 1 indicates perfect fairness.
        """
        a = np.asarray(allocations, dtype=np.float64)
        n = len(a)
        if n == 0:
            return 1.0
        return float((a.sum() ** 2) / (n * (a ** 2).sum()))


# ---------------------------------------------------------------------------
# WorstCaseDataset
# ---------------------------------------------------------------------------


class WorstCaseDataset:
    """Compute the worst-case neighbouring dataset pair for a mechanism.

    Given a mechanism M and adjacency relation, find the pair (x, x') that
    maximises the privacy loss ratio max_S P[M(x) in S] / P[M(x') in S].
    """

    def __init__(self, tol: float = 1e-10) -> None:
        self._tol = tol

    def find_worst_pair(
        self,
        mechanism: npt.NDArray[np.float64],
        adjacency: AdjacencyRelation,
    ) -> Tuple[Tuple[int, int], float]:
        """Find the adjacent pair with maximum privacy loss.

        Args:
            mechanism: n x k probability table P[M(x_i) = y_j].
            adjacency: Which database pairs are adjacent.

        Returns:
            ((i, i'), max_loss) where max_loss is the maximum log-ratio.
        """
        P = np.asarray(mechanism, dtype=np.float64)
        n, k = P.shape
        worst_pair = adjacency.edges[0] if adjacency.edges else (0, 0)
        worst_loss = -np.inf

        for i, j in adjacency.edges:
            if i >= n or j >= n:
                continue
            # Max over output bins of log(P[i,b] / P[j,b])
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios = np.where(
                    P[j] > self._tol,
                    np.log(np.maximum(P[i], self._tol) / np.maximum(P[j], self._tol)),
                    0.0,
                )
            loss = float(np.max(np.abs(ratios)))
            if loss > worst_loss:
                worst_loss = loss
                worst_pair = (i, j)

        return worst_pair, worst_loss

    def privacy_loss_distribution(
        self,
        mechanism: npt.NDArray[np.float64],
        i: int,
        j: int,
    ) -> npt.NDArray[np.float64]:
        """Compute the privacy loss random variable for pair (i, j).

        Returns log(P[M(x_i) = y_b] / P[M(x_j) = y_b]) for each bin b.
        """
        P = np.asarray(mechanism, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            pld = np.log(
                np.maximum(P[i], 1e-300) / np.maximum(P[j], 1e-300)
            )
        return pld

    def hockey_stick_divergence(
        self,
        mechanism: npt.NDArray[np.float64],
        i: int,
        j: int,
        epsilon: float,
    ) -> float:
        """Compute hockey-stick divergence D_{e^eps}(M(x_i) || M(x_j)).

        D_{e^eps}(P || Q) = sum_b max(0, P[b] - e^eps * Q[b]).
        """
        P = np.asarray(mechanism, dtype=np.float64)
        diff = P[i] - np.exp(epsilon) * P[j]
        return float(np.sum(np.maximum(diff, 0.0)))


# ---------------------------------------------------------------------------
# MinimaxLPFormulation
# ---------------------------------------------------------------------------


class MinimaxLPFormulation:
    """Encode minimax mechanism design as a linear program.

    Builds the full LP for the mechanism-vs-adversary game:
        min  t
        s.t. sum_j L(f(x_i), y_j) p[i,j] <= t   (utility)
             p[i,j] <= e^eps p[i',j]              (DP forward)
             p[i',j] <= e^eps p[i,j]              (DP backward)
             sum_j p[i,j] = 1                     (simplex)
             p >= 0
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 0.0,
        tol: float = 1e-10,
    ) -> None:
        self.epsilon = epsilon
        self.delta = delta
        self._tol = tol

    def build(
        self,
        query_values: npt.NDArray[np.float64],
        output_grid: npt.NDArray[np.float64],
        adjacency: AdjacencyRelation,
        loss_fn: Optional[Callable[[float, float], float]] = None,
    ) -> Dict[str, Any]:
        """Build the LP data structures.

        Args:
            query_values: f(x_i) for each database i.
            output_grid: y_j output discretisation points.
            adjacency: Adjacent database pairs.
            loss_fn: Loss function l(true, noisy). Defaults to squared error.

        Returns:
            Dict with keys 'c', 'A_ub', 'b_ub', 'A_eq', 'b_eq', 'bounds'.
        """
        if loss_fn is None:
            loss_fn = lambda t, n: (t - n) ** 2

        fvals = np.asarray(query_values, dtype=np.float64)
        ygrid = np.asarray(output_grid, dtype=np.float64)
        n_db = len(fvals)
        k = len(ygrid)
        n_vars = n_db * k + 1  # p[i,j] variables + epigraph t
        e_eps = np.exp(self.epsilon)

        # Objective: min t
        c = np.zeros(n_vars)
        c[-1] = 1.0

        # Build loss matrix L[i,j] = loss(fvals[i], ygrid[j])
        L = np.zeros((n_db, k))
        for i in range(n_db):
            for j in range(k):
                L[i, j] = loss_fn(float(fvals[i]), float(ygrid[j]))

        ub_rows = []
        ub_rhs = []

        # Utility constraints: sum_j L[i,j] p[i,j] - t <= 0
        for i in range(n_db):
            row = np.zeros(n_vars)
            for j in range(k):
                row[i * k + j] = L[i, j]
            row[-1] = -1.0
            ub_rows.append(row)
            ub_rhs.append(0.0)

        # DP constraints: p[i,j] - e^eps p[i',j] <= 0
        for i1, i2 in adjacency.edges:
            if i1 >= n_db or i2 >= n_db:
                continue
            for j in range(k):
                # Forward
                row_f = np.zeros(n_vars)
                row_f[i1 * k + j] = 1.0
                row_f[i2 * k + j] = -e_eps
                ub_rows.append(row_f)
                ub_rhs.append(0.0)
                # Backward
                row_b = np.zeros(n_vars)
                row_b[i2 * k + j] = 1.0
                row_b[i1 * k + j] = -e_eps
                ub_rows.append(row_b)
                ub_rhs.append(0.0)

        A_ub = np.array(ub_rows) if ub_rows else np.zeros((0, n_vars))
        b_ub = np.array(ub_rhs) if ub_rhs else np.zeros(0)

        # Simplex constraints: sum_j p[i,j] = 1
        eq_rows = []
        eq_rhs = []
        for i in range(n_db):
            row = np.zeros(n_vars)
            row[i * k:(i + 1) * k] = 1.0
            eq_rows.append(row)
            eq_rhs.append(1.0)

        A_eq = np.array(eq_rows) if eq_rows else np.zeros((0, n_vars))
        b_eq = np.array(eq_rhs) if eq_rhs else np.zeros(0)

        bounds = [(0.0, None)] * (n_db * k) + [(None, None)]

        return {
            "c": c,
            "A_ub": A_ub,
            "b_ub": b_ub,
            "A_eq": A_eq,
            "b_eq": b_eq,
            "bounds": bounds,
            "n_databases": n_db,
            "n_outputs": k,
            "loss_matrix": L,
        }

    def solve(
        self,
        query_values: npt.NDArray[np.float64],
        output_grid: npt.NDArray[np.float64],
        adjacency: AdjacencyRelation,
        loss_fn: Optional[Callable[[float, float], float]] = None,
    ) -> Tuple[npt.NDArray[np.float64], float]:
        """Build and solve the minimax LP.

        Returns:
            (mechanism, optimal_loss) where mechanism is n x k.
        """
        lp = self.build(query_values, output_grid, adjacency, loss_fn)
        res = sp_opt.linprog(
            lp["c"],
            A_ub=lp["A_ub"],
            b_ub=lp["b_ub"],
            A_eq=lp["A_eq"],
            b_eq=lp["b_eq"],
            bounds=lp["bounds"],
            method="highs",
        )
        if not res.success:
            raise RuntimeError(f"Minimax LP failed: {res.message}")

        n_db = lp["n_databases"]
        k = lp["n_outputs"]
        P = res.x[:n_db * k].reshape(n_db, k)
        P = np.maximum(P, 0.0)
        P /= P.sum(axis=1, keepdims=True)
        return P, float(res.x[-1])


# ---------------------------------------------------------------------------
# RobustOptimization
# ---------------------------------------------------------------------------


class RobustOptimization:
    """Uncertainty-set-based robust mechanism design.

    Handles uncertainty in the adversary's capabilities or in the data
    distribution by optimising over an uncertainty set.

    Supports:
    - Box uncertainty: each payoff A[i,j] in [A[i,j] - delta, A[i,j] + delta]
    - Ellipsoidal uncertainty: || A - A0 ||_F <= rho
    - Budget uncertainty: at most Gamma entries deviate
    """

    def __init__(self, tol: float = 1e-8) -> None:
        self._tol = tol

    def solve_box_robust(
        self,
        A_nominal: npt.NDArray[np.float64],
        delta: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float64], float]:
        """Solve minimax under box uncertainty.

        For each column j, the adversary can shift the column within
        [-delta[i,j], +delta[i,j]].  The robust minimax LP uses the
        worst-case payoff for each pure strategy.

        Args:
            A_nominal: Nominal payoff matrix (m x n).
            delta: Element-wise uncertainty radius (m x n).

        Returns:
            (strategy, robust_value)
        """
        A_nom = np.asarray(A_nominal, dtype=np.float64)
        D = np.asarray(delta, dtype=np.float64)
        # Worst case for designer (minimiser): adversary picks +delta on payoff
        A_worst = A_nom + D
        solver = MinimaxSolver()
        result = solver.solve(GameMatrix(payoffs=A_worst))
        return (
            result.equilibrium.designer_strategy.probabilities,
            result.equilibrium.game_value,
        )

    def solve_budget_robust(
        self,
        A_nominal: npt.NDArray[np.float64],
        deviation: npt.NDArray[np.float64],
        gamma: float,
    ) -> Tuple[npt.NDArray[np.float64], float]:
        """Solve minimax under budget-of-uncertainty (Bertsimas-Sim).

        At most gamma entries of each row may deviate by their maximum.

        Args:
            A_nominal: Nominal payoff matrix (m x n).
            deviation: Maximum per-entry deviation (m x n).
            gamma: Budget (max number of deviating entries per row).

        Returns:
            (strategy, robust_value)
        """
        A_nom = np.asarray(A_nominal, dtype=np.float64)
        D = np.asarray(deviation, dtype=np.float64)
        m, n = A_nom.shape
        gamma_int = min(int(np.ceil(gamma)), n)

        # For each row, worst case adds the gamma largest deviations
        A_robust = A_nom.copy()
        for i in range(m):
            sorted_dev = np.sort(D[i])[::-1]
            A_robust[i] += np.sum(sorted_dev[:gamma_int]) / n * np.ones(n)

        solver = MinimaxSolver()
        result = solver.solve(GameMatrix(payoffs=A_robust))
        return (
            result.equilibrium.designer_strategy.probabilities,
            result.equilibrium.game_value,
        )

    def solve_ellipsoidal_robust(
        self,
        A_nominal: npt.NDArray[np.float64],
        rho: float,
    ) -> Tuple[npt.NDArray[np.float64], float]:
        """Solve minimax under ellipsoidal uncertainty ||A - A0||_F <= rho.

        Uses the robust counterpart: for a fixed mixed strategy x,
        worst case adversary shifts each column by rho * x / ||x||.

        Args:
            A_nominal: Nominal payoff matrix (m x n).
            rho: Frobenius norm uncertainty radius.

        Returns:
            (strategy, robust_value)
        """
        A_nom = np.asarray(A_nominal, dtype=np.float64)
        m, n = A_nom.shape

        # Iterative approach: solve nominal, then perturb
        solver = MinimaxSolver()
        result = solver.solve(GameMatrix(payoffs=A_nom))
        x = result.equilibrium.designer_strategy.probabilities

        # Robust value = nominal_value + rho * ||x||_2
        # (from duality of ellipsoidal uncertainty)
        robust_value = result.equilibrium.game_value + rho * float(np.linalg.norm(x))

        return x, robust_value

    def sensitivity_analysis(
        self,
        A: npt.NDArray[np.float64],
        perturbation_range: float = 0.1,
        n_samples: int = 20,
    ) -> Dict[str, Any]:
        """Analyse sensitivity of the minimax value to payoff perturbations.

        Args:
            A: Base payoff matrix.
            perturbation_range: Max perturbation magnitude.
            n_samples: Number of perturbation samples.

        Returns:
            Dict with 'values', 'mean', 'std', 'max', 'min'.
        """
        A = np.asarray(A, dtype=np.float64)
        solver = MinimaxSolver()
        values = []
        rng = np.random.default_rng(42)
        for _ in range(n_samples):
            perturb = rng.uniform(
                -perturbation_range, perturbation_range, size=A.shape
            )
            result = solver.solve(GameMatrix(payoffs=A + perturb))
            values.append(result.equilibrium.game_value)

        vals = np.array(values)
        return {
            "values": vals,
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "max": float(vals.max()),
            "min": float(vals.min()),
        }
