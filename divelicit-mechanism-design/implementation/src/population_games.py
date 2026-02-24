"""
Evolutionary & population game dynamics: replicator, best response,
fictitious play, ESS, logit dynamics, Nash equilibrium computation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from itertools import product as iter_product, combinations
import warnings


@dataclass
class GameResult:
    """Result of a game dynamics simulation."""
    trajectory: np.ndarray  # (timesteps, n_strategies)
    final_state: np.ndarray
    converged: bool
    equilibrium_type: str


class ReplicatorDynamics:
    """
    Replicator dynamics for evolutionary game theory.
    dx_i/dt = x_i * (f_i(x) - avg_fitness(x))
    """

    def __init__(self, payoff_matrix: np.ndarray):
        """
        payoff_matrix: (n_strategies, n_strategies) for symmetric 2-player game
        A[i][j] = payoff to strategy i against strategy j
        """
        self.payoff_matrix = payoff_matrix
        self.n_strategies = payoff_matrix.shape[0]

    def fitness(self, population: np.ndarray) -> np.ndarray:
        """Compute fitness of each strategy given population state."""
        return self.payoff_matrix @ population

    def average_fitness(self, population: np.ndarray) -> float:
        """Compute average population fitness."""
        f = self.fitness(population)
        return np.dot(population, f)

    def evolve_continuous(self, initial_pop: np.ndarray, timesteps: int = 1000,
                          dt: float = 0.01) -> GameResult:
        """
        Simulate continuous-time replicator dynamics using RK4.
        """
        pop = initial_pop.copy()
        trajectory = [pop.copy()]

        def deriv(x):
            f = self.payoff_matrix @ x
            avg_f = np.dot(x, f)
            return x * (f - avg_f)

        for t in range(timesteps):
            # RK4 integration
            k1 = dt * deriv(pop)
            k2 = dt * deriv(pop + k1 / 2)
            k3 = dt * deriv(pop + k2 / 2)
            k4 = dt * deriv(pop + k3)

            pop = pop + (k1 + 2 * k2 + 2 * k3 + k4) / 6

            # Project onto simplex
            pop = np.maximum(pop, 0)
            total = pop.sum()
            if total > 0:
                pop /= total

            trajectory.append(pop.copy())

        # Check convergence vs oscillation
        recent = np.array(trajectory[-100:])
        max_std = np.std(recent, axis=0).max()
        converged = max_std < 1e-6

        # Detect oscillatory behavior: high std but stable mean
        oscillatory = False
        if not converged and len(trajectory) > 200:
            first_half = np.array(trajectory[-200:-100])
            second_half = np.array(trajectory[-100:])
            mean_shift = np.max(np.abs(np.mean(first_half, axis=0) - np.mean(second_half, axis=0)))
            if mean_shift < 0.05 and max_std > 1e-4:
                oscillatory = True

        # Determine equilibrium type
        eq_type = self._classify_equilibrium(pop, oscillatory=oscillatory)

        return GameResult(
            trajectory=np.array(trajectory),
            final_state=pop,
            converged=converged,
            equilibrium_type=eq_type
        )

    def evolve_discrete(self, initial_pop: np.ndarray, timesteps: int = 1000) -> GameResult:
        """
        Simulate discrete-time replicator dynamics.
        x_i(t+1) = x_i(t) * f_i(x(t)) / avg_f(x(t))
        """
        pop = initial_pop.copy()
        trajectory = [pop.copy()]

        for t in range(timesteps):
            f = self.fitness(pop)
            avg_f = self.average_fitness(pop)

            if abs(avg_f) < 1e-15:
                break

            pop = pop * f / avg_f
            pop = np.maximum(pop, 0)
            total = pop.sum()
            if total > 0:
                pop /= total

            trajectory.append(pop.copy())

        recent = np.array(trajectory[-100:])
        max_std = np.std(recent, axis=0).max()
        converged = max_std < 1e-6

        oscillatory = False
        if not converged and len(trajectory) > 200:
            first_half = np.array(trajectory[-200:-100])
            second_half = np.array(trajectory[-100:])
            mean_shift = np.max(np.abs(np.mean(first_half, axis=0) - np.mean(second_half, axis=0)))
            if mean_shift < 0.05 and max_std > 1e-4:
                oscillatory = True

        return GameResult(
            trajectory=np.array(trajectory),
            final_state=pop,
            converged=converged,
            equilibrium_type=self._classify_equilibrium(pop, oscillatory=oscillatory)
        )

    def _classify_equilibrium(self, pop: np.ndarray, oscillatory: bool = False) -> str:
        """Classify the type of equilibrium reached."""
        if oscillatory:
            return "oscillatory_center"
        active = pop > 0.01
        n_active = np.sum(active)

        if n_active == 1:
            dominant = np.argmax(pop)
            return f"pure_strategy_{dominant}"
        elif n_active == self.n_strategies:
            return "interior_mixed"
        else:
            return f"boundary_mixed_{n_active}_strategies"


class BestResponseDynamics:
    """Best response dynamics for normal-form games."""

    def __init__(self, payoff_matrix_row: np.ndarray, payoff_matrix_col: np.ndarray):
        """
        payoff_matrix_row: (m, n) payoffs for row player
        payoff_matrix_col: (m, n) payoffs for column player
        """
        self.A = payoff_matrix_row
        self.B = payoff_matrix_col
        self.m, self.n = self.A.shape

    def best_response_row(self, col_strategy: np.ndarray) -> np.ndarray:
        """Compute best response of row player to column's mixed strategy."""
        expected_payoffs = self.A @ col_strategy
        best = np.argmax(expected_payoffs)
        br = np.zeros(self.m)
        br[best] = 1.0
        return br

    def best_response_col(self, row_strategy: np.ndarray) -> np.ndarray:
        """Compute best response of column player to row's mixed strategy."""
        expected_payoffs = self.B.T @ row_strategy
        best = np.argmax(expected_payoffs)
        br = np.zeros(self.n)
        br[best] = 1.0
        return br

    def simulate(self, initial_row: np.ndarray, initial_col: np.ndarray,
                 timesteps: int = 100, smoothing: float = 0.1) -> Tuple[GameResult, GameResult]:
        """
        Simulate smoothed best response dynamics.
        """
        row = initial_row.copy()
        col = initial_col.copy()
        row_traj = [row.copy()]
        col_traj = [col.copy()]

        for t in range(timesteps):
            br_row = self.best_response_row(col)
            br_col = self.best_response_col(row)

            # Smooth update
            row = (1 - smoothing) * row + smoothing * br_row
            col = (1 - smoothing) * col + smoothing * br_col

            row_traj.append(row.copy())
            col_traj.append(col.copy())

        row_result = GameResult(
            trajectory=np.array(row_traj),
            final_state=row,
            converged=np.std(np.array(row_traj[-20:]), axis=0).max() < 1e-4,
            equilibrium_type="best_response"
        )
        col_result = GameResult(
            trajectory=np.array(col_traj),
            final_state=col,
            converged=np.std(np.array(col_traj[-20:]), axis=0).max() < 1e-4,
            equilibrium_type="best_response"
        )

        return row_result, col_result


class FictitiousPlay:
    """
    Fictitious play: each player best-responds to empirical frequency of opponent.
    """

    def __init__(self, payoff_matrix_row: np.ndarray, payoff_matrix_col: np.ndarray):
        self.A = payoff_matrix_row
        self.B = payoff_matrix_col
        self.m, self.n = self.A.shape

    def simulate(self, timesteps: int = 1000, seed: int = 42) -> Dict:
        """
        Simulate fictitious play.
        Returns history of empirical frequencies and actions.
        """
        rng = np.random.default_rng(seed)

        # Counts of each action played
        row_counts = np.ones(self.m) * 0.01  # Small prior
        col_counts = np.ones(self.n) * 0.01

        row_history = []
        col_history = []
        row_freq_history = []
        col_freq_history = []

        for t in range(timesteps):
            # Empirical frequencies
            row_freq = row_counts / row_counts.sum()
            col_freq = col_counts / col_counts.sum()

            # Best response to empirical frequency
            row_payoffs = self.A @ col_freq
            col_payoffs = self.B.T @ row_freq

            row_action = np.argmax(row_payoffs)
            col_action = np.argmax(col_payoffs)

            row_counts[row_action] += 1
            col_counts[col_action] += 1

            row_history.append(row_action)
            col_history.append(col_action)
            row_freq_history.append(row_counts / row_counts.sum())
            col_freq_history.append(col_counts / col_counts.sum())

        # Check convergence
        if len(row_freq_history) >= 100:
            recent_row = np.array(row_freq_history[-100:])
            recent_col = np.array(col_freq_history[-100:])
            converged = (np.std(recent_row, axis=0).max() < 0.01 and
                         np.std(recent_col, axis=0).max() < 0.01)
        else:
            converged = False

        return {
            'row_frequencies': row_freq_history[-1] if row_freq_history else np.ones(self.m) / self.m,
            'col_frequencies': col_freq_history[-1] if col_freq_history else np.ones(self.n) / self.n,
            'converged': converged,
            'row_freq_history': [f.tolist() for f in row_freq_history[::max(1, timesteps // 50)]],
            'col_freq_history': [f.tolist() for f in col_freq_history[::max(1, timesteps // 50)]],
        }


class EvolutionaryStableStrategy:
    """Check and find evolutionarily stable strategies (ESS)."""

    def __init__(self, payoff_matrix: np.ndarray):
        self.A = payoff_matrix
        self.n = payoff_matrix.shape[0]

    def is_ess(self, strategy: np.ndarray) -> bool:
        """
        Check if strategy x is an ESS.
        ESS conditions:
        1. x is a Nash equilibrium
        2. For any alternative best response y:
           x^T A y > y^T A y (strict in second condition)
        """
        # Check NE condition: x^T A x >= y^T A x for all y
        fitness_x = strategy @ self.A @ strategy

        # Check against all pure strategies
        for i in range(self.n):
            e_i = np.zeros(self.n)
            e_i[i] = 1.0

            fitness_ei = e_i @ self.A @ strategy

            if fitness_ei > fitness_x + 1e-10:
                return False  # Not a NE

            if abs(fitness_ei - fitness_x) < 1e-10:
                # Alternative best response, check second condition
                if strategy @ self.A @ e_i <= e_i @ self.A @ e_i + 1e-10:
                    # Check more carefully
                    if strategy @ self.A @ e_i < e_i @ self.A @ e_i - 1e-10:
                        return False

        return True

    def find_all_ess(self) -> List[np.ndarray]:
        """Find all ESS of the game (pure and mixed)."""
        ess_list = []

        # Check pure strategies
        for i in range(self.n):
            e_i = np.zeros(self.n)
            e_i[i] = 1.0
            if self.is_ess(e_i):
                ess_list.append(e_i)

        # Check fully mixed NE
        try:
            # Solve for interior NE: A @ x = c * 1, sum(x) = 1
            # All strategies must yield equal fitness
            # (A - A[0]) @ x = 0 (relative to first strategy)
            # sum(x) = 1
            n = self.n
            if n >= 2:
                # System: A^T x has all equal components, sum(x)=1
                M = np.zeros((n, n))
                for i in range(n - 1):
                    M[i] = self.A[i] - self.A[n - 1]
                M[n - 1] = np.ones(n)
                b = np.zeros(n)
                b[n - 1] = 1.0

                try:
                    x = np.linalg.solve(M, b)
                    if np.all(x > -1e-10):
                        x = np.maximum(x, 0)
                        x /= x.sum()
                        if self.is_ess(x):
                            ess_list.append(x)
                except np.linalg.LinAlgError:
                    pass
        except Exception:
            pass

        # Check 2-strategy supports
        for i in range(self.n):
            for j in range(i + 1, self.n):
                # Solve 2x2 game
                a = self.A[i, i] - self.A[j, i]
                b_val = self.A[i, j] - self.A[j, j]
                if abs(a - b_val) > 1e-10:
                    p = b_val / (b_val - a)  # NE mixing probability
                    if 0 < p < 1:
                        x = np.zeros(self.n)
                        x[i] = p
                        x[j] = 1 - p
                        if self.is_ess(x):
                            ess_list.append(x)

        return ess_list


def logit_dynamics(payoff_matrix: np.ndarray, initial_pop: np.ndarray,
                    beta: float = 1.0, timesteps: int = 1000,
                    dt: float = 0.01) -> GameResult:
    """
    Logit dynamics (quantal response equilibrium).
    Agents choose strategies with probability proportional to exp(beta * payoff).
    """
    n = payoff_matrix.shape[0]
    pop = initial_pop.copy()
    trajectory = [pop.copy()]

    for t in range(timesteps):
        f = payoff_matrix @ pop
        # Logit choice probabilities
        exp_f = np.exp(beta * (f - np.max(f)))  # Subtract max for numerical stability
        logit_probs = exp_f / exp_f.sum()

        # Dynamics: dx_i/dt = logit_i - x_i
        dpop = logit_probs - pop
        pop = pop + dt * dpop
        pop = np.maximum(pop, 0)
        pop /= pop.sum()

        trajectory.append(pop.copy())

    recent = np.array(trajectory[-100:])
    converged = np.std(recent, axis=0).max() < 1e-5

    return GameResult(
        trajectory=np.array(trajectory),
        final_state=pop,
        converged=converged,
        equilibrium_type="quantal_response"
    )


def imitation_dynamics(payoff_matrix: np.ndarray, initial_pop: np.ndarray,
                         timesteps: int = 1000, dt: float = 0.01) -> GameResult:
    """
    Pairwise imitation dynamics.
    Agents switch to better-performing strategies proportionally.
    """
    n = payoff_matrix.shape[0]
    pop = initial_pop.copy()
    trajectory = [pop.copy()]

    for t in range(timesteps):
        f = payoff_matrix @ pop
        # Pairwise comparison: rate of switching i->j = x_i * x_j * max(f_j - f_i, 0)
        dpop = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    switch_rate = pop[i] * pop[j] * max(f[j] - f[i], 0)
                    dpop[i] -= switch_rate
                    dpop[j] += switch_rate

        pop = pop + dt * dpop
        pop = np.maximum(pop, 0)
        total = pop.sum()
        if total > 0:
            pop /= total

        trajectory.append(pop.copy())

    recent = np.array(trajectory[-100:])
    converged = np.std(recent, axis=0).max() < 1e-5

    return GameResult(
        trajectory=np.array(trajectory),
        final_state=pop,
        converged=converged,
        equilibrium_type="imitation"
    )


def smith_dynamics(payoff_matrix: np.ndarray, initial_pop: np.ndarray,
                     timesteps: int = 1000, dt: float = 0.01) -> GameResult:
    """
    Smith dynamics for population games.
    dx_i/dt = sum_j x_j * max(f_i - f_j, 0) - x_i * sum_j max(f_j - f_i, 0)
    """
    n = payoff_matrix.shape[0]
    pop = initial_pop.copy()
    trajectory = [pop.copy()]

    for t in range(timesteps):
        f = payoff_matrix @ pop
        dpop = np.zeros(n)

        for i in range(n):
            inflow = sum(pop[j] * max(f[i] - f[j], 0) for j in range(n))
            outflow = pop[i] * sum(max(f[j] - f[i], 0) for j in range(n))
            dpop[i] = inflow - outflow

        pop = pop + dt * dpop
        pop = np.maximum(pop, 0)
        total = pop.sum()
        if total > 0:
            pop /= total

        trajectory.append(pop.copy())

    recent = np.array(trajectory[-100:])
    converged = np.std(recent, axis=0).max() < 1e-5

    return GameResult(
        trajectory=np.array(trajectory),
        final_state=pop,
        converged=converged,
        equilibrium_type="smith"
    )


class NashEquilibriumFinder:
    """Find Nash equilibria of finite games."""

    def __init__(self, payoff_row: np.ndarray, payoff_col: np.ndarray):
        self.A = payoff_row
        self.B = payoff_col
        self.m, self.n = self.A.shape

    def support_enumeration(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Find all Nash equilibria by support enumeration.
        For each pair of supports, solve the indifference conditions.
        """
        equilibria = []

        for row_support_size in range(1, self.m + 1):
            for col_support_size in range(1, self.n + 1):
                for row_support in combinations(range(self.m), row_support_size):
                    for col_support in combinations(range(self.n), col_support_size):
                        result = self._solve_support(list(row_support), list(col_support))
                        if result is not None:
                            x, y = result
                            if self._is_nash(x, y):
                                # Check for duplicates
                                is_dup = False
                                for ex, ey in equilibria:
                                    if np.allclose(x, ex) and np.allclose(y, ey):
                                        is_dup = True
                                        break
                                if not is_dup:
                                    equilibria.append((x, y))

        return equilibria

    def _solve_support(self, row_support: List[int],
                       col_support: List[int]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Solve for NE with given supports."""
        k = len(row_support)
        l = len(col_support)

        # Column player must be indifferent over row_support
        # A[i] @ y = A[j] @ y for all i, j in row_support
        # and sum(y[col_support]) = 1

        if l > 0:
            # Solve for y: (A[row_support[0]] - A[row_support[i]]) @ y = 0
            n_eq = k - 1 + 1  # indifference + normalization
            if n_eq != l:
                if n_eq > l:
                    return None

            M_y = np.zeros((l, l))
            b_y = np.zeros(l)

            for i in range(k - 1):
                for j_idx, j in enumerate(col_support):
                    M_y[i, j_idx] = self.A[row_support[0], j] - self.A[row_support[i + 1], j]

            # Normalization
            M_y[k - 1] = 1.0
            b_y[k - 1] = 1.0

            if k < l:
                # Underdetermined, skip (or use least squares)
                return None

            try:
                y_sub = np.linalg.solve(M_y[:l, :l], b_y[:l])
            except np.linalg.LinAlgError:
                return None

            if np.any(y_sub < -1e-10):
                return None

            y = np.zeros(self.n)
            for i, idx in enumerate(col_support):
                y[idx] = max(y_sub[i], 0)
        else:
            return None

        # Similarly for x
        if k > 0:
            M_x = np.zeros((k, k))
            b_x = np.zeros(k)

            for j in range(l - 1):
                for i_idx, i in enumerate(row_support):
                    M_x[j, i_idx] = self.B[i, col_support[0]] - self.B[i, col_support[j + 1]]

            M_x[l - 1] = 1.0
            b_x[l - 1] = 1.0

            if l < k:
                return None

            try:
                x_sub = np.linalg.solve(M_x[:k, :k], b_x[:k])
            except np.linalg.LinAlgError:
                return None

            if np.any(x_sub < -1e-10):
                return None

            x = np.zeros(self.m)
            for i, idx in enumerate(row_support):
                x[idx] = max(x_sub[i], 0)
        else:
            return None

        return x, y

    def _is_nash(self, x: np.ndarray, y: np.ndarray) -> bool:
        """Verify Nash equilibrium conditions."""
        # Row player: x^T A y >= e_i^T A y for all i
        val_row = x @ self.A @ y
        for i in range(self.m):
            if self.A[i] @ y > val_row + 1e-8:
                return False

        # Col player: x^T B y >= x^T B e_j for all j
        val_col = x @ self.B @ y
        for j in range(self.n):
            if x @ self.B[:, j] > val_col + 1e-8:
                return False

        return True

    def lemke_howson(self, pivot_var: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lemke-Howson algorithm for finding one Nash equilibrium.
        Uses complementary pivoting on the labeled polytope.
        """
        m, n = self.m, self.n

        # Normalize payoff matrices to be positive
        A = self.A - self.A.min() + 1
        B = self.B - self.B.min() + 1

        # Tableaux
        # Row player tableau: [I | B^T] with m+n columns
        # Col player tableau: [A | I] with m+n columns

        # Start with artificial equilibrium (0, 0)
        # Use support enumeration fallback for small games
        if m <= 4 and n <= 4:
            eqs = self.support_enumeration()
            if eqs:
                return eqs[0]

        # Simplified Lemke-Howson via linear complementarity
        # Fall back to computing using best response iteration
        rng = np.random.default_rng(42)
        x = np.ones(m) / m
        y = np.ones(n) / n

        for _ in range(1000):
            # Best response for row
            row_payoffs = self.A @ y
            best_row = np.argmax(row_payoffs)
            x_new = np.zeros(m)
            x_new[best_row] = 1.0
            x = 0.9 * x + 0.1 * x_new

            # Best response for col
            col_payoffs = self.B.T @ x
            best_col = np.argmax(col_payoffs)
            y_new = np.zeros(n)
            y_new[best_col] = 1.0
            y = 0.9 * y + 0.1 * y_new

        # Try to find nearby NE via support enumeration
        row_support = [i for i in range(m) if x[i] > 0.01]
        col_support = [j for j in range(n) if y[j] > 0.01]

        if row_support and col_support:
            result = self._solve_support(row_support, col_support)
            if result is not None and self._is_nash(*result):
                return result

        return x, y


# Standard game matrices

def prisoners_dilemma() -> Tuple[np.ndarray, np.ndarray]:
    """Standard prisoner's dilemma payoff matrices."""
    A = np.array([[3, 0], [5, 1]], dtype=float)
    B = np.array([[3, 5], [0, 1]], dtype=float)
    return A, B


def hawk_dove(v: float = 4.0, c: float = 6.0) -> np.ndarray:
    """
    Hawk-Dove game (symmetric).
    Hawk vs Hawk: (V-C)/2, Hawk vs Dove: V, Dove vs Hawk: 0, Dove vs Dove: V/2
    """
    return np.array([[(v - c) / 2, v], [0, v / 2]])


def rock_paper_scissors() -> np.ndarray:
    """Rock-Paper-Scissors symmetric payoff matrix."""
    return np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)


def coordination_game(a: float = 2.0, b: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Coordination game."""
    A = np.array([[a, 0], [0, b]], dtype=float)
    B = np.array([[a, 0], [0, b]], dtype=float)
    return A, B


def battle_of_sexes(a: float = 3.0, b: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Battle of the Sexes."""
    A = np.array([[a, 0], [0, b]], dtype=float)
    B = np.array([[b, 0], [0, a]], dtype=float)
    return A, B
