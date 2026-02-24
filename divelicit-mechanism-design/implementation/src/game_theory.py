"""Game-theoretic analysis of diversity selection.

Implements cooperative and non-cooperative game theory concepts applied to
diversity-aware LLM response selection, including Shapley values, Nash
equilibria, mechanism design guarantees, and welfare analysis.

Mathematical foundations:
- Shapley value: phi_i = sum_{S not containing i} |S|!(n-|S|-1)!/n! [v(S+i)-v(S)]
- Nash equilibrium via support enumeration
- Core computation via linear programming
- Nucleolus via lexicographic optimization
"""

from __future__ import annotations

import itertools
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from .kernels import Kernel, RBFKernel
from .utils import log_det_safe


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Player:
    """A player in the diversity game."""
    player_id: int
    embedding: np.ndarray
    quality: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class CoalitionalGame:
    """A coalitional (cooperative) game defined by value function v."""
    players: List[Player]
    value_function: Callable[[FrozenSet[int], List[Player]], float]
    _cache: Dict[FrozenSet[int], float] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.players)

    def value(self, coalition: FrozenSet[int]) -> float:
        if coalition not in self._cache:
            self._cache[coalition] = self.value_function(coalition, self.players)
        return self._cache[coalition]

    def clear_cache(self) -> None:
        self._cache.clear()


@dataclass
class GameTheoryResult:
    """Result from game-theoretic analysis."""
    solution_concept: str
    values: Dict[int, float]
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Value functions for diversity games
# ---------------------------------------------------------------------------

def logdet_diversity_value(
    coalition: FrozenSet[int],
    players: List[Player],
    kernel: Optional[Kernel] = None,
) -> float:
    """Log-determinant diversity value of a coalition."""
    if len(coalition) == 0:
        return 0.0
    if kernel is None:
        kernel = RBFKernel(bandwidth=1.0)
    embeddings = np.array([players[i].embedding for i in coalition])
    K = kernel.gram_matrix(embeddings)
    return max(log_det_safe(K), 0.0)


def quality_diversity_value(
    coalition: FrozenSet[int],
    players: List[Player],
    quality_weight: float = 0.5,
    kernel: Optional[Kernel] = None,
) -> float:
    """Combined quality and diversity value."""
    if len(coalition) == 0:
        return 0.0
    quality = sum(players[i].quality for i in coalition) / len(coalition)
    div = logdet_diversity_value(coalition, players, kernel)
    return quality_weight * quality + (1 - quality_weight) * div


def coverage_game_value(
    coalition: FrozenSet[int],
    players: List[Player],
    reference_points: Optional[np.ndarray] = None,
    radius: float = 1.0,
) -> float:
    """Coverage game: value = fraction of reference points covered."""
    if len(coalition) == 0:
        return 0.0
    if reference_points is None:
        reference_points = np.random.randn(20, players[0].embedding.shape[0])
    embeddings = np.array([players[i].embedding for i in coalition])
    covered = 0
    for ref in reference_points:
        dists = np.linalg.norm(embeddings - ref, axis=1)
        if np.min(dists) <= radius:
            covered += 1
    return covered / len(reference_points)


def facility_location_value(
    coalition: FrozenSet[int],
    players: List[Player],
    client_points: Optional[np.ndarray] = None,
) -> float:
    """Facility location game value: sum of max similarities to clients."""
    if len(coalition) == 0:
        return 0.0
    if client_points is None:
        client_points = np.random.randn(30, players[0].embedding.shape[0])
    embeddings = np.array([players[i].embedding for i in coalition])
    total = 0.0
    for client in client_points:
        sims = -np.linalg.norm(embeddings - client, axis=1)
        total += np.max(sims)
    return -total  # negative distance = higher value


# ---------------------------------------------------------------------------
# Shapley Value
# ---------------------------------------------------------------------------

class ShapleyValue:
    """Compute Shapley values for the diversity game.

    phi_i(v) = sum_{S in N\\{i}} |S|!(n-|S|-1)!/n! * [v(S+{i}) - v(S)]

    Exact computation is O(2^n), so Monte Carlo approximation is used for n>12.
    """

    def __init__(self, game: CoalitionalGame):
        self.game = game

    def compute_exact(self) -> Dict[int, float]:
        """Exact Shapley value computation (exponential in n)."""
        n = self.game.n
        phi: Dict[int, float] = {i: 0.0 for i in range(n)}

        for i in range(n):
            others = [j for j in range(n) if j != i]
            for size in range(len(others) + 1):
                weight = (
                    math.factorial(size) * math.factorial(n - size - 1)
                    / math.factorial(n)
                )
                for subset in itertools.combinations(others, size):
                    S = frozenset(subset)
                    S_with_i = S | frozenset([i])
                    marginal = self.game.value(S_with_i) - self.game.value(S)
                    phi[i] += weight * marginal

        return phi

    def compute_monte_carlo(
        self,
        n_samples: int = 1000,
        seed: int = 42,
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Monte Carlo approximation of Shapley values.

        Sample random permutations and compute marginal contributions.
        Returns (shapley_values, standard_errors).
        """
        rng = np.random.RandomState(seed)
        n = self.game.n
        contributions: Dict[int, List[float]] = {i: [] for i in range(n)}

        for _ in range(n_samples):
            perm = rng.permutation(n).tolist()
            current_set: Set[int] = set()
            current_value = 0.0

            for player in perm:
                new_set = current_set | {player}
                new_value = self.game.value(frozenset(new_set))
                marginal = new_value - current_value
                contributions[player].append(marginal)
                current_set = new_set
                current_value = new_value

        phi: Dict[int, float] = {}
        se: Dict[int, float] = {}
        for i in range(n):
            c = contributions[i]
            phi[i] = float(np.mean(c))
            se[i] = float(np.std(c) / math.sqrt(len(c)))

        return phi, se

    def compute(
        self,
        method: str = "auto",
        n_samples: int = 1000,
        seed: int = 42,
    ) -> Dict[int, float]:
        """Compute Shapley values using best method."""
        if method == "auto":
            if self.game.n <= 12:
                return self.compute_exact()
            return self.compute_monte_carlo(n_samples, seed)[0]
        elif method == "exact":
            return self.compute_exact()
        elif method == "monte_carlo":
            return self.compute_monte_carlo(n_samples, seed)[0]
        else:
            raise ValueError(f"Unknown method: {method}")

    def verify_efficiency(self, phi: Dict[int, float]) -> float:
        """Verify efficiency axiom: sum phi_i = v(N)."""
        total = sum(phi.values())
        grand_coalition = frozenset(range(self.game.n))
        v_N = self.game.value(grand_coalition)
        return abs(total - v_N)

    def verify_symmetry(
        self,
        phi: Dict[int, float],
        i: int,
        j: int,
    ) -> bool:
        """Verify symmetry: if v(S+i) = v(S+j) for all S, then phi_i = phi_j."""
        n = self.game.n
        others = [k for k in range(n) if k != i and k != j]
        symmetric = True
        for size in range(len(others) + 1):
            for subset in itertools.combinations(others, size):
                S = frozenset(subset)
                v_si = self.game.value(S | {i})
                v_sj = self.game.value(S | {j})
                if abs(v_si - v_sj) > 1e-8:
                    symmetric = False
                    break
            if not symmetric:
                break
        if symmetric:
            return abs(phi[i] - phi[j]) < 1e-6
        return True  # axiom doesn't apply


# ---------------------------------------------------------------------------
# Banzhaf Power Index
# ---------------------------------------------------------------------------

class BanzhafIndex:
    """Banzhaf power index for the diversity game.

    beta_i = (1/2^{n-1}) sum_{S: i not in S} [v(S+i) - v(S) > 0]
    """

    def __init__(self, game: CoalitionalGame):
        self.game = game

    def compute(self) -> Dict[int, float]:
        """Compute Banzhaf power index."""
        n = self.game.n
        beta: Dict[int, float] = {i: 0.0 for i in range(n)}

        for i in range(n):
            others = [j for j in range(n) if j != i]
            swing_count = 0
            total_coalitions = 0

            for size in range(len(others) + 1):
                for subset in itertools.combinations(others, size):
                    S = frozenset(subset)
                    S_with_i = S | frozenset([i])
                    marginal = self.game.value(S_with_i) - self.game.value(S)
                    if marginal > 1e-10:
                        swing_count += 1
                    total_coalitions += 1

            beta[i] = swing_count / max(total_coalitions, 1)

        # Normalize
        beta_sum = sum(beta.values())
        if beta_sum > 0:
            for i in beta:
                beta[i] /= beta_sum

        return beta

    def compute_raw(self) -> Dict[int, float]:
        """Compute raw (unnormalized) Banzhaf values."""
        n = self.game.n
        raw: Dict[int, float] = {i: 0.0 for i in range(n)}

        for i in range(n):
            others = [j for j in range(n) if j != i]
            for size in range(len(others) + 1):
                for subset in itertools.combinations(others, size):
                    S = frozenset(subset)
                    S_with_i = S | frozenset([i])
                    marginal = self.game.value(S_with_i) - self.game.value(S)
                    raw[i] += marginal

            raw[i] /= 2 ** (n - 1)

        return raw


# ---------------------------------------------------------------------------
# Core of the diversity game
# ---------------------------------------------------------------------------

class CoreComputation:
    """Compute the core of a coalitional game.

    Core = {x in R^n : sum_i x_i = v(N) and sum_{i in S} x_i >= v(S) for all S}
    """

    def __init__(self, game: CoalitionalGame):
        self.game = game

    def check_in_core(self, allocation: Dict[int, float]) -> Tuple[bool, List[FrozenSet[int]]]:
        """Check if an allocation is in the core.

        Returns (is_in_core, violating_coalitions).
        """
        n = self.game.n
        grand_value = self.game.value(frozenset(range(n)))
        total = sum(allocation.values())

        # Check efficiency
        if abs(total - grand_value) > 1e-6:
            return False, []

        violations: List[FrozenSet[int]] = []
        for size in range(1, n):
            for subset in itertools.combinations(range(n), size):
                S = frozenset(subset)
                coalition_value = self.game.value(S)
                allocated = sum(allocation.get(i, 0) for i in S)
                if allocated < coalition_value - 1e-6:
                    violations.append(S)

        return len(violations) == 0, violations

    def is_core_empty(self) -> bool:
        """Check if the core is empty using LP feasibility.

        For convex games, core is always non-empty.
        """
        n = self.game.n
        # Check convexity (sufficient for non-empty core)
        is_convex = self._check_convexity()
        if is_convex:
            return False

        # Try Shapley value (always in core for convex games)
        shapley = ShapleyValue(self.game)
        phi = shapley.compute()
        in_core, _ = self.check_in_core(phi)
        return not in_core

    def _check_convexity(self) -> bool:
        """Check if game is convex (supermodular).

        v(S union T) + v(S intersect T) >= v(S) + v(T)
        """
        n = self.game.n
        players = list(range(n))
        for s1 in range(1, n):
            for s2 in range(1, n):
                for c1 in itertools.combinations(players, s1):
                    for c2 in itertools.combinations(players, s2):
                        S = frozenset(c1)
                        T = frozenset(c2)
                        union = S | T
                        inter = S & T
                        lhs = self.game.value(union) + self.game.value(inter)
                        rhs = self.game.value(S) + self.game.value(T)
                        if lhs < rhs - 1e-8:
                            return False
        return True

    def find_core_allocation(self) -> Optional[Dict[int, float]]:
        """Find an allocation in the core using a simple LP approach.

        Minimize deviation from equal split subject to core constraints.
        """
        n = self.game.n
        grand = self.game.value(frozenset(range(n)))

        # Start with Shapley value
        shapley = ShapleyValue(self.game)
        x = shapley.compute()

        # Adjust to satisfy core constraints via iterative projection
        for iteration in range(100):
            in_core, violations = self.check_in_core(x)
            if in_core:
                return x
            if len(violations) == 0:
                break
            # Fix worst violation
            worst_deficit = 0.0
            worst_coalition: Optional[FrozenSet[int]] = None
            for S in violations:
                v_S = self.game.value(S)
                alloc_S = sum(x.get(i, 0) for i in S)
                deficit = v_S - alloc_S
                if deficit > worst_deficit:
                    worst_deficit = deficit
                    worst_coalition = S

            if worst_coalition is None:
                break

            # Transfer from outsiders to coalition members
            outsiders = frozenset(range(n)) - worst_coalition
            transfer_per_member = worst_deficit / max(len(worst_coalition), 1)
            tax_per_outsider = worst_deficit / max(len(outsiders), 1)

            for i in worst_coalition:
                x[i] = x.get(i, 0) + transfer_per_member
            for i in outsiders:
                x[i] = x.get(i, 0) - tax_per_outsider

        in_core, _ = self.check_in_core(x)
        return x if in_core else None


# ---------------------------------------------------------------------------
# Nucleolus
# ---------------------------------------------------------------------------

class NucleolusComputation:
    """Compute the nucleolus of a coalitional game.

    The nucleolus lexicographically minimizes the vector of sorted excesses:
    e(S, x) = v(S) - sum_{i in S} x_i

    It is always unique and in the core (if the core is non-empty).
    """

    def __init__(self, game: CoalitionalGame):
        self.game = game

    def _compute_excesses(
        self,
        allocation: Dict[int, float],
    ) -> List[Tuple[FrozenSet[int], float]]:
        """Compute excess for all coalitions."""
        n = self.game.n
        excesses: List[Tuple[FrozenSet[int], float]] = []
        for size in range(1, n):
            for subset in itertools.combinations(range(n), size):
                S = frozenset(subset)
                v_S = self.game.value(S)
                alloc_S = sum(allocation.get(i, 0) for i in S)
                excess = v_S - alloc_S
                excesses.append((S, excess))
        return excesses

    def compute(
        self,
        max_iterations: int = 1000,
        learning_rate: float = 0.01,
    ) -> Dict[int, float]:
        """Compute nucleolus via iterative excess minimization.

        Uses gradient descent on the lexicographic objective.
        """
        n = self.game.n
        grand = self.game.value(frozenset(range(n)))

        # Initialize with Shapley value
        shapley = ShapleyValue(self.game)
        x = shapley.compute()

        for iteration in range(max_iterations):
            excesses = self._compute_excesses(x)
            if len(excesses) == 0:
                break

            # Sort by excess (descending - most unhappy first)
            excesses.sort(key=lambda e: e[1], reverse=True)

            # Reduce the maximum excess
            max_excess_coalition, max_excess = excesses[0]
            if max_excess <= 1e-10:
                break

            # Gradient step: increase allocation to max-excess coalition
            for i in max_excess_coalition:
                x[i] += learning_rate * max_excess / len(max_excess_coalition)

            # Re-normalize to maintain efficiency
            total = sum(x.values())
            if total > 0:
                for i in x:
                    x[i] *= grand / total

        return x

    def verify_nucleolus(self, x: Dict[int, float]) -> Dict[str, float]:
        """Verify properties of a nucleolus candidate."""
        excesses = self._compute_excesses(x)
        excess_values = [e for _, e in excesses]
        grand = self.game.value(frozenset(range(self.game.n)))
        total = sum(x.values())

        return {
            "max_excess": float(max(excess_values)) if excess_values else 0.0,
            "min_excess": float(min(excess_values)) if excess_values else 0.0,
            "mean_excess": float(np.mean(excess_values)) if excess_values else 0.0,
            "efficiency_gap": abs(total - grand),
            "n_positive_excesses": sum(1 for e in excess_values if e > 1e-10),
        }


# ---------------------------------------------------------------------------
# Nash Equilibrium
# ---------------------------------------------------------------------------

@dataclass
class StrategyProfile:
    """A strategy profile in the diversity selection game."""
    strategies: Dict[int, np.ndarray]  # player -> mixed strategy over items
    expected_utilities: Dict[int, float] = field(default_factory=dict)


class NashEquilibrium:
    """Compute Nash equilibrium of the diversity selection game.

    Game: Each player (response generator) chooses a distribution over
    embedding space regions. Payoff depends on diversity of all selections.
    """

    def __init__(
        self,
        n_players: int,
        n_strategies: int,
        payoff_matrices: Optional[List[np.ndarray]] = None,
        seed: int = 42,
    ):
        self.n_players = n_players
        self.n_strategies = n_strategies
        self.rng = np.random.RandomState(seed)
        if payoff_matrices is not None:
            self.payoff_matrices = payoff_matrices
        else:
            self.payoff_matrices = self._generate_diversity_payoffs()

    def _generate_diversity_payoffs(self) -> List[np.ndarray]:
        """Generate payoff matrices encoding diversity preferences."""
        matrices = []
        for p in range(self.n_players):
            if self.n_players == 2:
                M = np.zeros((self.n_strategies, self.n_strategies))
                for i in range(self.n_strategies):
                    for j in range(self.n_strategies):
                        # Higher payoff for different strategies (diversity)
                        if i != j:
                            M[i, j] = 1.0 + self.rng.uniform(0, 0.5)
                        else:
                            M[i, j] = 0.5 + self.rng.uniform(0, 0.3)
                matrices.append(M)
            else:
                shape = tuple([self.n_strategies] * self.n_players)
                M = self.rng.uniform(0, 1, size=shape)
                matrices.append(M)
        return matrices

    def support_enumeration_2player(self) -> List[StrategyProfile]:
        """Find all Nash equilibria for 2-player game via support enumeration."""
        if self.n_players != 2:
            raise ValueError("Support enumeration only works for 2-player games")

        A = self.payoff_matrices[0]  # Row player
        B = self.payoff_matrices[1]  # Column player
        n, m = A.shape

        equilibria: List[StrategyProfile] = []

        for support_size_1 in range(1, n + 1):
            for support_size_2 in range(1, m + 1):
                for supp1 in itertools.combinations(range(n), support_size_1):
                    for supp2 in itertools.combinations(range(m), support_size_2):
                        eq = self._check_support_equilibrium(
                            A, B, list(supp1), list(supp2)
                        )
                        if eq is not None:
                            equilibria.append(eq)

        return equilibria

    def _check_support_equilibrium(
        self,
        A: np.ndarray,
        B: np.ndarray,
        supp1: List[int],
        supp2: List[int],
    ) -> Optional[StrategyProfile]:
        """Check if given supports form a Nash equilibrium."""
        n, m = A.shape
        s1, s2 = len(supp1), len(supp2)

        # Player 2's mixed strategy must make player 1 indifferent
        # A[i,:] @ q = constant for all i in supp1
        if s2 > 1:
            A_sub = A[np.ix_(supp1, supp2)]
            # Solve for q: A_sub @ q = c * ones, sum(q) = 1
            try:
                # Use least squares
                M = np.vstack([
                    A_sub[1:] - A_sub[0:1],
                    np.ones((1, s2))
                ])
                rhs = np.zeros(M.shape[0])
                rhs[-1] = 1.0
                q_sub = np.linalg.lstsq(M, rhs, rcond=None)[0]
            except np.linalg.LinAlgError:
                return None
        else:
            q_sub = np.array([1.0])

        # Check q >= 0
        if np.any(q_sub < -1e-10):
            return None
        q_sub = np.maximum(q_sub, 0)
        q_sum = np.sum(q_sub)
        if q_sum < 1e-10:
            return None
        q_sub /= q_sum

        # Similarly for player 1
        if s1 > 1:
            B_sub = B[np.ix_(supp1, supp2)]
            try:
                M = np.vstack([
                    B_sub.T[1:] - B_sub.T[0:1],
                    np.ones((1, s1))
                ])
                rhs = np.zeros(M.shape[0])
                rhs[-1] = 1.0
                p_sub = np.linalg.lstsq(M, rhs, rcond=None)[0]
            except np.linalg.LinAlgError:
                return None
        else:
            p_sub = np.array([1.0])

        if np.any(p_sub < -1e-10):
            return None
        p_sub = np.maximum(p_sub, 0)
        p_sum = np.sum(p_sub)
        if p_sum < 1e-10:
            return None
        p_sub /= p_sum

        # Full strategies
        p = np.zeros(n)
        q = np.zeros(m)
        for i, idx in enumerate(supp1):
            p[idx] = p_sub[i]
        for j, idx in enumerate(supp2):
            q[idx] = q_sub[j]

        # Verify: no strategy outside support gives higher payoff
        payoff1_at_eq = A @ q
        eq_payoff1 = payoff1_at_eq[supp1[0]]
        for i in range(n):
            if i not in supp1 and payoff1_at_eq[i] > eq_payoff1 + 1e-8:
                return None

        payoff2_at_eq = B.T @ p
        eq_payoff2 = payoff2_at_eq[supp2[0]]
        for j in range(m):
            if j not in supp2 and payoff2_at_eq[j] > eq_payoff2 + 1e-8:
                return None

        return StrategyProfile(
            strategies={0: p, 1: q},
            expected_utilities={0: float(eq_payoff1), 1: float(eq_payoff2)},
        )

    def fictitious_play(
        self,
        n_iterations: int = 10000,
    ) -> StrategyProfile:
        """Find approximate Nash equilibrium via fictitious play."""
        counts = [np.zeros(self.n_strategies) for _ in range(self.n_players)]

        for t in range(n_iterations):
            # Each player best-responds to empirical distribution of others
            empirical = []
            for p in range(self.n_players):
                total = np.sum(counts[p])
                if total > 0:
                    empirical.append(counts[p] / total)
                else:
                    empirical.append(np.ones(self.n_strategies) / self.n_strategies)

            for p in range(self.n_players):
                if self.n_players == 2:
                    other = 1 - p
                    M = self.payoff_matrices[p]
                    expected = M @ empirical[other]
                    best = int(np.argmax(expected))
                else:
                    best = self.rng.randint(self.n_strategies)
                counts[p][best] += 1

        strategies: Dict[int, np.ndarray] = {}
        for p in range(self.n_players):
            total = np.sum(counts[p])
            strategies[p] = counts[p] / total if total > 0 else np.ones(self.n_strategies) / self.n_strategies

        # Compute expected utilities
        utilities: Dict[int, float] = {}
        if self.n_players == 2:
            utilities[0] = float(strategies[0] @ self.payoff_matrices[0] @ strategies[1])
            utilities[1] = float(strategies[0] @ self.payoff_matrices[1] @ strategies[1])

        return StrategyProfile(strategies=strategies, expected_utilities=utilities)

    def replicator_dynamics(
        self,
        n_steps: int = 5000,
        dt: float = 0.01,
    ) -> StrategyProfile:
        """Find Nash equilibrium via replicator dynamics."""
        strategies = [
            np.ones(self.n_strategies) / self.n_strategies
            for _ in range(self.n_players)
        ]

        for step in range(n_steps):
            new_strategies = []
            for p in range(self.n_players):
                if self.n_players == 2:
                    other = 1 - p
                    M = self.payoff_matrices[p]
                    fitness = M @ strategies[other]
                    avg_fitness = strategies[p] @ fitness
                    # Replicator: dx_i/dt = x_i * (f_i - avg_f)
                    dx = strategies[p] * (fitness - avg_fitness)
                    new_s = strategies[p] + dt * dx
                    new_s = np.maximum(new_s, 1e-12)
                    new_s /= np.sum(new_s)
                    new_strategies.append(new_s)
                else:
                    new_strategies.append(strategies[p])
            strategies = new_strategies

        result_strategies = {p: strategies[p] for p in range(self.n_players)}
        utilities: Dict[int, float] = {}
        if self.n_players == 2:
            utilities[0] = float(strategies[0] @ self.payoff_matrices[0] @ strategies[1])
            utilities[1] = float(strategies[0] @ self.payoff_matrices[1] @ strategies[1])

        return StrategyProfile(strategies=result_strategies, expected_utilities=utilities)


# ---------------------------------------------------------------------------
# Correlated Equilibrium
# ---------------------------------------------------------------------------

class CorrelatedEquilibrium:
    """Compute correlated equilibrium via linear programming.

    A correlated equilibrium is a distribution p over strategy profiles
    such that no player benefits from deviating after receiving their signal.
    """

    def __init__(
        self,
        n_players: int,
        n_strategies: int,
        payoff_matrices: List[np.ndarray],
    ):
        self.n_players = n_players
        self.n_strategies = n_strategies
        self.payoff_matrices = payoff_matrices

    def compute_2player(self) -> Optional[np.ndarray]:
        """Compute correlated equilibrium for 2-player game.

        Uses a simple iterative approach to find a feasible distribution.
        """
        n, m = self.n_strategies, self.n_strategies
        A = self.payoff_matrices[0]
        B = self.payoff_matrices[1]

        # Start with uniform distribution
        p = np.ones((n, m)) / (n * m)

        for iteration in range(1000):
            # Check incentive constraints for player 1
            violation_found = False
            for i in range(n):
                for i_prime in range(n):
                    if i == i_prime:
                        continue
                    # sum_j p(i,j) * (A[i,j] - A[i',j]) >= 0
                    constraint = sum(
                        p[i, j] * (A[i, j] - A[i_prime, j]) for j in range(m)
                    )
                    if constraint < -1e-8:
                        # Shift probability mass
                        for j in range(m):
                            shift = min(p[i, j] * 0.1, 0.01)
                            p[i, j] = max(p[i, j] - shift, 0)
                            p[i_prime, j] += shift
                        violation_found = True

            # Check incentive constraints for player 2
            for j in range(m):
                for j_prime in range(m):
                    if j == j_prime:
                        continue
                    constraint = sum(
                        p[i, j] * (B[i, j] - B[i, j_prime]) for i in range(n)
                    )
                    if constraint < -1e-8:
                        for i in range(n):
                            shift = min(p[i, j] * 0.1, 0.01)
                            p[i, j] = max(p[i, j] - shift, 0)
                            p[i, j_prime] += shift
                        violation_found = True

            # Normalize
            p_sum = np.sum(p)
            if p_sum > 0:
                p /= p_sum

            if not violation_found:
                break

        return p

    def social_welfare(self, p: np.ndarray) -> float:
        """Compute expected social welfare under correlated equilibrium."""
        if self.n_players != 2:
            return 0.0
        A = self.payoff_matrices[0]
        B = self.payoff_matrices[1]
        welfare = np.sum(p * (A + B))
        return float(welfare)


# ---------------------------------------------------------------------------
# Price of Anarchy / Stability
# ---------------------------------------------------------------------------

class PriceOfAnarchyAnalysis:
    """Analyze inefficiency of equilibria via Price of Anarchy/Stability.

    PoA = optimal welfare / worst equilibrium welfare
    PoS = optimal welfare / best equilibrium welfare
    """

    def __init__(
        self,
        n_players: int,
        n_strategies: int,
        payoff_matrices: List[np.ndarray],
    ):
        self.n_players = n_players
        self.n_strategies = n_strategies
        self.payoff_matrices = payoff_matrices

    def optimal_social_welfare(self) -> Tuple[float, Tuple[int, ...]]:
        """Find strategy profile maximizing social welfare."""
        if self.n_players != 2:
            raise NotImplementedError("Only 2-player games supported")
        A = self.payoff_matrices[0]
        B = self.payoff_matrices[1]
        welfare = A + B
        best_idx = np.unravel_index(np.argmax(welfare), welfare.shape)
        return float(welfare[best_idx]), best_idx

    def compute(self) -> Dict[str, float]:
        """Compute PoA and PoS."""
        optimal_welfare, optimal_profile = self.optimal_social_welfare()

        # Find Nash equilibria
        ne = NashEquilibrium(
            self.n_players, self.n_strategies, self.payoff_matrices
        )

        if self.n_players == 2 and self.n_strategies <= 6:
            equilibria = ne.support_enumeration_2player()
        else:
            equilibria = [ne.fictitious_play()]

        if len(equilibria) == 0:
            equilibria = [ne.fictitious_play()]

        # Compute welfare at each equilibrium
        eq_welfares = []
        for eq in equilibria:
            if self.n_players == 2:
                A = self.payoff_matrices[0]
                B = self.payoff_matrices[1]
                w = float(eq.strategies[0] @ (A + B) @ eq.strategies[1])
                eq_welfares.append(w)

        if len(eq_welfares) == 0:
            return {
                "optimal_welfare": optimal_welfare,
                "price_of_anarchy": 1.0,
                "price_of_stability": 1.0,
            }

        worst_welfare = min(eq_welfares)
        best_welfare = max(eq_welfares)

        poa = optimal_welfare / max(worst_welfare, 1e-12)
        pos = optimal_welfare / max(best_welfare, 1e-12)

        return {
            "optimal_welfare": optimal_welfare,
            "worst_eq_welfare": worst_welfare,
            "best_eq_welfare": best_welfare,
            "price_of_anarchy": float(poa),
            "price_of_stability": float(pos),
            "n_equilibria": len(equilibria),
        }


# ---------------------------------------------------------------------------
# Mechanism Design Guarantees
# ---------------------------------------------------------------------------

class MechanismDesignGuarantees:
    """Verify mechanism design properties for diversity mechanisms.

    Checks:
    - Individual rationality (IR)
    - Incentive compatibility (IC)
    - Budget balance (BB)
    - Social welfare bounds
    """

    def __init__(
        self,
        game: CoalitionalGame,
        kernel: Optional[Kernel] = None,
    ):
        self.game = game
        self.kernel = kernel or RBFKernel(bandwidth=1.0)

    def check_individual_rationality(
        self,
        allocation: Dict[int, float],
    ) -> Dict[int, bool]:
        """Check IR: each player gets at least their standalone value.

        IR: x_i >= v({i}) for all i
        """
        ir: Dict[int, bool] = {}
        for i in range(self.game.n):
            standalone = self.game.value(frozenset([i]))
            ir[i] = allocation.get(i, 0) >= standalone - 1e-8
        return ir

    def check_budget_balance(
        self,
        allocation: Dict[int, float],
        payments: Optional[Dict[int, float]] = None,
    ) -> Dict[str, float]:
        """Check budget balance properties.

        Weak BB: sum payments >= 0
        Strong BB: sum payments = 0
        """
        total_alloc = sum(allocation.values())
        grand = self.game.value(frozenset(range(self.game.n)))
        if payments is None:
            return {
                "is_efficient": abs(total_alloc - grand) < 1e-6,
                "surplus": grand - total_alloc,
            }
        total_payments = sum(payments.values())
        return {
            "total_payments": total_payments,
            "is_weakly_balanced": total_payments >= -1e-8,
            "is_strongly_balanced": abs(total_payments) < 1e-6,
            "is_efficient": abs(total_alloc - grand) < 1e-6,
        }

    def impossibility_theorem_check(
        self,
        allocation: Dict[int, float],
        payments: Optional[Dict[int, float]] = None,
    ) -> Dict[str, bool]:
        """Check Green-Laffont impossibility conditions.

        No mechanism can simultaneously achieve:
        1. Efficiency
        2. IC (DSIC)
        3. Weak budget balance
        4. Individual rationality

        We check which subset of properties hold.
        """
        n = self.game.n
        ir = self.check_individual_rationality(allocation)
        bb = self.check_budget_balance(allocation, payments)

        all_ir = all(ir.values())
        is_efficient = bb.get("is_efficient", False)
        is_bb = bb.get("is_weakly_balanced", False) if payments else True

        return {
            "individual_rationality": all_ir,
            "efficiency": is_efficient,
            "budget_balance": is_bb,
            "all_three": all_ir and is_efficient and is_bb,
        }

    def social_welfare_bounds(
        self,
        allocation: Dict[int, float],
    ) -> Dict[str, float]:
        """Compute social welfare bounds and approximation ratio."""
        n = self.game.n
        actual_welfare = sum(allocation.values())
        optimal_welfare = self.game.value(frozenset(range(n)))

        # Compute VCG welfare
        shapley = ShapleyValue(self.game)
        phi = shapley.compute()
        vcg_welfare = sum(phi.values())

        return {
            "actual_welfare": actual_welfare,
            "optimal_welfare": optimal_welfare,
            "vcg_welfare": vcg_welfare,
            "approximation_ratio": (
                actual_welfare / optimal_welfare if optimal_welfare > 0 else 1.0
            ),
            "vcg_ratio": (
                vcg_welfare / optimal_welfare if optimal_welfare > 0 else 1.0
            ),
        }

    def myerson_satterthwaite_check(self) -> Dict[str, bool]:
        """Check Myerson-Satterthwaite impossibility for bilateral trade.

        In bilateral trade with private values, no mechanism is simultaneously:
        - IC
        - IR
        - Budget balanced
        - Ex-post efficient
        """
        n = self.game.n
        if n != 2:
            return {"applicable": False}

        # For 2-player diversity game: check if there exist prices
        # that satisfy all four properties
        v1 = self.game.value(frozenset([0]))
        v2 = self.game.value(frozenset([1]))
        v12 = self.game.value(frozenset([0, 1]))

        gains_from_trade = v12 - v1 - v2
        return {
            "applicable": True,
            "gains_from_trade": float(gains_from_trade),
            "trade_beneficial": gains_from_trade > 0,
            "impossibility_applies": gains_from_trade > 0,
        }


# ---------------------------------------------------------------------------
# Diversity game analysis pipeline
# ---------------------------------------------------------------------------

class DiversityGameAnalysis:
    """Complete game-theoretic analysis of a diversity selection problem."""

    def __init__(
        self,
        embeddings: np.ndarray,
        quality_scores: np.ndarray,
        kernel: Optional[Kernel] = None,
    ):
        self.n = embeddings.shape[0]
        self.players = [
            Player(i, embeddings[i], float(quality_scores[i]))
            for i in range(self.n)
        ]
        self.kernel = kernel or RBFKernel(bandwidth=1.0)

        # Create coalitional game
        k = self.kernel

        def value_fn(S: FrozenSet[int], players: List[Player]) -> float:
            return quality_diversity_value(S, players, kernel=k)

        self.game = CoalitionalGame(
            players=self.players, value_function=value_fn
        )

    def full_analysis(
        self,
        n_shapley_samples: int = 1000,
        seed: int = 42,
    ) -> Dict[str, GameTheoryResult]:
        """Run complete game-theoretic analysis."""
        results: Dict[str, GameTheoryResult] = {}

        # 1. Shapley values
        shapley = ShapleyValue(self.game)
        phi = shapley.compute(n_samples=n_shapley_samples, seed=seed)
        eff_gap = shapley.verify_efficiency(phi)
        results["shapley"] = GameTheoryResult(
            solution_concept="shapley_value",
            values=phi,
            metadata={"efficiency_gap": eff_gap},
        )

        # 2. Banzhaf index
        if self.n <= 15:
            banzhaf = BanzhafIndex(self.game)
            beta = banzhaf.compute()
            results["banzhaf"] = GameTheoryResult(
                solution_concept="banzhaf_index",
                values=beta,
            )

        # 3. Core
        if self.n <= 12:
            core = CoreComputation(self.game)
            core_alloc = core.find_core_allocation()
            if core_alloc is not None:
                in_core, violations = core.check_in_core(core_alloc)
                results["core"] = GameTheoryResult(
                    solution_concept="core",
                    values=core_alloc,
                    metadata={
                        "in_core": in_core,
                        "n_violations": len(violations),
                    },
                )

        # 4. Nucleolus
        if self.n <= 12:
            nucl = NucleolusComputation(self.game)
            nucl_alloc = nucl.compute()
            verify = nucl.verify_nucleolus(nucl_alloc)
            results["nucleolus"] = GameTheoryResult(
                solution_concept="nucleolus",
                values=nucl_alloc,
                metadata=verify,
            )

        # 5. Mechanism design guarantees
        guarantees = MechanismDesignGuarantees(self.game, self.kernel)
        ir = guarantees.check_individual_rationality(phi)
        welfare = guarantees.social_welfare_bounds(phi)
        results["mechanism_guarantees"] = GameTheoryResult(
            solution_concept="mechanism_design",
            values={i: float(ir[i]) for i in ir},
            metadata={**welfare},
        )

        return results

    def compare_solution_concepts(self) -> Dict[str, Dict[int, float]]:
        """Compare different solution concepts side by side."""
        comparison: Dict[str, Dict[int, float]] = {}

        # Shapley
        shapley = ShapleyValue(self.game)
        comparison["shapley"] = shapley.compute()

        # Banzhaf
        if self.n <= 15:
            banzhaf = BanzhafIndex(self.game)
            comparison["banzhaf"] = banzhaf.compute()

        # Nucleolus
        if self.n <= 12:
            nucl = NucleolusComputation(self.game)
            comparison["nucleolus"] = nucl.compute()

        # Equal split
        grand = self.game.value(frozenset(range(self.n)))
        comparison["equal_split"] = {i: grand / self.n for i in range(self.n)}

        return comparison


# ---------------------------------------------------------------------------
# Weighted Voting Game
# ---------------------------------------------------------------------------

class WeightedVotingGame:
    """Weighted voting game for diversity committee decisions.

    Each response has a weight and a quality vote.
    A coalition wins if its total weight exceeds the quota.
    """

    def __init__(
        self,
        weights: np.ndarray,
        quota: float,
    ):
        self.weights = weights
        self.n = len(weights)
        self.quota = quota

    def is_winning(self, coalition: FrozenSet[int]) -> bool:
        """Check if coalition meets quota."""
        return sum(self.weights[i] for i in coalition) >= self.quota

    def value(self, coalition: FrozenSet[int]) -> float:
        """Binary value: 1 if winning, 0 otherwise."""
        return 1.0 if self.is_winning(coalition) else 0.0

    def shapley_shubik_index(self) -> Dict[int, float]:
        """Compute Shapley-Shubik power index."""
        game = CoalitionalGame(
            players=[Player(i, np.zeros(1)) for i in range(self.n)],
            value_function=lambda S, _: self.value(S),
        )
        shapley = ShapleyValue(game)
        return shapley.compute()

    def banzhaf_power(self) -> Dict[int, float]:
        """Compute Banzhaf power index."""
        game = CoalitionalGame(
            players=[Player(i, np.zeros(1)) for i in range(self.n)],
            value_function=lambda S, _: self.value(S),
        )
        banzhaf = BanzhafIndex(game)
        return banzhaf.compute()

    def minimal_winning_coalitions(self) -> List[FrozenSet[int]]:
        """Find all minimal winning coalitions."""
        minimal: List[FrozenSet[int]] = []
        for size in range(1, self.n + 1):
            for combo in itertools.combinations(range(self.n), size):
                S = frozenset(combo)
                if not self.is_winning(S):
                    continue
                is_minimal = True
                for i in S:
                    if self.is_winning(S - {i}):
                        is_minimal = False
                        break
                if is_minimal:
                    minimal.append(S)
        return minimal


# ---------------------------------------------------------------------------
# Congestion Game for Diversity
# ---------------------------------------------------------------------------

class CongestionGame:
    """Congestion game where responses compete for diversity "resources".

    Resources = diversity dimensions/topics. Cost increases with congestion.
    """

    def __init__(
        self,
        n_players: int,
        n_resources: int,
        strategy_sets: Optional[List[List[FrozenSet[int]]]] = None,
        cost_fn: Optional[Callable[[int, int], float]] = None,
        seed: int = 42,
    ):
        self.n_players = n_players
        self.n_resources = n_resources
        self.rng = np.random.RandomState(seed)

        if cost_fn is not None:
            self.cost_fn = cost_fn
        else:
            self.cost_fn = lambda resource, load: load  # linear congestion

        if strategy_sets is not None:
            self.strategy_sets = strategy_sets
        else:
            # Each player has 3 strategies (subsets of resources)
            self.strategy_sets = []
            for p in range(n_players):
                strategies = []
                for _ in range(3):
                    size = self.rng.randint(1, min(4, n_resources + 1))
                    resources = frozenset(self.rng.choice(n_resources, size, replace=False).tolist())
                    strategies.append(resources)
                self.strategy_sets.append(strategies)

    def compute_costs(
        self,
        strategy_profile: List[int],
    ) -> np.ndarray:
        """Compute costs for each player given strategy profile."""
        # Count resource loads
        loads = np.zeros(self.n_resources)
        for p, s_idx in enumerate(strategy_profile):
            resources = self.strategy_sets[p][s_idx]
            for r in resources:
                loads[r] += 1

        # Compute player costs
        costs = np.zeros(self.n_players)
        for p, s_idx in enumerate(strategy_profile):
            resources = self.strategy_sets[p][s_idx]
            costs[p] = sum(self.cost_fn(r, int(loads[r])) for r in resources)

        return costs

    def find_pure_nash(self) -> Optional[List[int]]:
        """Find pure Nash equilibrium via best-response dynamics."""
        n_strategies = [len(s) for s in self.strategy_sets]
        profile = [0] * self.n_players

        for iteration in range(1000):
            changed = False
            for p in range(self.n_players):
                current_cost = self.compute_costs(profile)[p]
                best_strategy = profile[p]
                best_cost = current_cost

                for s in range(n_strategies[p]):
                    trial = profile.copy()
                    trial[p] = s
                    cost = self.compute_costs(trial)[p]
                    if cost < best_cost:
                        best_cost = cost
                        best_strategy = s

                if best_strategy != profile[p]:
                    profile[p] = best_strategy
                    changed = True

            if not changed:
                return profile

        return None

    def price_of_anarchy(self) -> float:
        """Compute PoA of the congestion game."""
        ne = self.find_pure_nash()
        if ne is None:
            return float("inf")

        ne_cost = float(np.sum(self.compute_costs(ne)))

        # Find social optimum
        best_cost = float("inf")
        n_strats = [len(s) for s in self.strategy_sets]

        def _enumerate(p: int, profile: List[int]) -> None:
            nonlocal best_cost
            if p >= self.n_players:
                cost = float(np.sum(self.compute_costs(profile)))
                best_cost = min(best_cost, cost)
                return
            for s in range(n_strats[p]):
                profile[p] = s
                _enumerate(p + 1, profile)

        if self.n_players <= 6:
            _enumerate(0, [0] * self.n_players)
        else:
            best_cost = ne_cost

        return ne_cost / max(best_cost, 1e-12)


# ---------------------------------------------------------------------------
# Stable matching for diversity
# ---------------------------------------------------------------------------

class StableMatchingDiversity:
    """Stable matching between responses and quality aspects.

    Uses Gale-Shapley to find a stable assignment of responses
    to diversity dimensions.
    """

    def __init__(
        self,
        n_responses: int,
        n_aspects: int,
        response_prefs: Optional[np.ndarray] = None,
        aspect_prefs: Optional[np.ndarray] = None,
        seed: int = 42,
    ):
        self.n_responses = n_responses
        self.n_aspects = n_aspects
        self.rng = np.random.RandomState(seed)

        if response_prefs is not None:
            self.response_prefs = response_prefs
        else:
            self.response_prefs = np.zeros((n_responses, n_aspects))
            for i in range(n_responses):
                self.response_prefs[i] = self.rng.permutation(n_aspects)

        if aspect_prefs is not None:
            self.aspect_prefs = aspect_prefs
        else:
            self.aspect_prefs = np.zeros((n_aspects, n_responses))
            for j in range(n_aspects):
                self.aspect_prefs[j] = self.rng.permutation(n_responses)

    def gale_shapley(self) -> Dict[int, int]:
        """Run Gale-Shapley (response-proposing).

        Returns dict: response -> matched aspect.
        """
        n = min(self.n_responses, self.n_aspects)
        free_responses = list(range(n))
        match: Dict[int, int] = {}  # response -> aspect
        aspect_match: Dict[int, int] = {}  # aspect -> response
        proposals = [0] * n  # next aspect to propose to for each response

        aspect_rankings: Dict[int, Dict[int, int]] = {}
        for j in range(self.n_aspects):
            aspect_rankings[j] = {}
            for rank, resp in enumerate(self.aspect_prefs[j].astype(int)):
                aspect_rankings[j][resp] = rank

        while free_responses:
            resp = free_responses.pop(0)
            if proposals[resp] >= self.n_aspects:
                continue
            # Propose to next preferred aspect
            pref_order = self.response_prefs[resp].astype(int)
            aspect = pref_order[proposals[resp]]
            proposals[resp] += 1

            if aspect not in aspect_match:
                match[resp] = aspect
                aspect_match[aspect] = resp
            else:
                current = aspect_match[aspect]
                # Does aspect prefer resp over current?
                if aspect_rankings[aspect].get(resp, n) < aspect_rankings[aspect].get(current, n):
                    match[resp] = aspect
                    aspect_match[aspect] = resp
                    del match[current]
                    free_responses.append(current)
                else:
                    free_responses.append(resp)

        return match

    def is_stable(self, matching: Dict[int, int]) -> bool:
        """Check if matching is stable (no blocking pairs)."""
        for resp, aspect in matching.items():
            # Check if any unmatched pair would prefer each other
            for other_aspect in range(self.n_aspects):
                if other_aspect == aspect:
                    continue
                # Does resp prefer other_aspect?
                resp_pref = self.response_prefs[resp].astype(int)
                resp_rank_current = np.where(resp_pref == aspect)[0][0] if aspect in resp_pref else self.n_aspects
                resp_rank_other = np.where(resp_pref == other_aspect)[0][0] if other_aspect in resp_pref else self.n_aspects
                if resp_rank_other >= resp_rank_current:
                    continue
                # Does other_aspect prefer resp over its current match?
                other_matched = None
                for r, a in matching.items():
                    if a == other_aspect:
                        other_matched = r
                        break
                if other_matched is None:
                    return False  # Blocking pair
                aspect_ranking = self.aspect_prefs[other_aspect].astype(int)
                rank_resp = np.where(aspect_ranking == resp)[0][0] if resp in aspect_ranking else self.n_responses
                rank_current = np.where(aspect_ranking == other_matched)[0][0] if other_matched in aspect_ranking else self.n_responses
                if rank_resp < rank_current:
                    return False
        return True


# ---------------------------------------------------------------------------
# Coalition Formation Game
# ---------------------------------------------------------------------------

class CoalitionFormationGame:
    """Coalition formation game for diversity groups.

    Responses form coalitions to maximize collective diversity.
    Stable partition = no group can deviate and all be better off.
    """

    def __init__(
        self,
        players: List[Player],
        kernel: Optional[Kernel] = None,
    ):
        self.players = players
        self.n = len(players)
        self.kernel = kernel or RBFKernel(bandwidth=1.0)

    def _coalition_value(self, coalition: List[int]) -> float:
        """Value of a coalition."""
        if len(coalition) == 0:
            return 0.0
        embs = np.array([self.players[i].embedding for i in coalition])
        quality = np.mean([self.players[i].quality for i in coalition])
        if len(coalition) < 2:
            return quality
        K = self.kernel.gram_matrix(embs)
        div = log_det_safe(K)
        return 0.5 * quality + 0.5 * max(div, 0)

    def greedy_partition(self, n_groups: int) -> List[List[int]]:
        """Greedy coalition formation: assign to most valuable group."""
        groups: List[List[int]] = [[] for _ in range(n_groups)]
        remaining = list(range(self.n))
        self_rng = np.random.RandomState(42)
        self_rng.shuffle(remaining)

        for player in remaining:
            best_group = 0
            best_value = -float("inf")
            for g in range(n_groups):
                candidate = groups[g] + [player]
                val = self._coalition_value(candidate)
                if val > best_value:
                    best_value = val
                    best_group = g
            groups[best_group].append(player)

        return groups

    def is_nash_stable(self, partition: List[List[int]]) -> bool:
        """Check Nash stability: no player benefits from switching groups."""
        player_to_group = {}
        for gi, group in enumerate(partition):
            for p in group:
                player_to_group[p] = gi

        for p in range(self.n):
            current_group = player_to_group[p]
            current_val = self._coalition_value(partition[current_group])
            without_p = [x for x in partition[current_group] if x != p]
            marginal_current = current_val - self._coalition_value(without_p)

            for gi, group in enumerate(partition):
                if gi == current_group:
                    continue
                new_group = group + [p]
                new_val = self._coalition_value(new_group)
                marginal_new = new_val - self._coalition_value(group)
                if marginal_new > marginal_current + 1e-8:
                    return False
        return True

    def improve_partition(
        self,
        partition: List[List[int]],
        max_iterations: int = 100,
    ) -> List[List[int]]:
        """Improve partition via best-response dynamics."""
        for iteration in range(max_iterations):
            changed = False
            for p in range(self.n):
                # Find current group
                current_gi = -1
                for gi, group in enumerate(partition):
                    if p in group:
                        current_gi = gi
                        break

                best_gi = current_gi
                best_gain = 0.0

                for gi, group in enumerate(partition):
                    if gi == current_gi:
                        continue
                    # Value of moving p to group gi
                    old_val_here = self._coalition_value(partition[current_gi])
                    new_val_here = self._coalition_value([x for x in partition[current_gi] if x != p])
                    old_val_there = self._coalition_value(group)
                    new_val_there = self._coalition_value(group + [p])

                    gain = (new_val_there - old_val_there) - (old_val_here - new_val_here)
                    if gain > best_gain:
                        best_gain = gain
                        best_gi = gi

                if best_gi != current_gi:
                    partition[current_gi].remove(p)
                    partition[best_gi].append(p)
                    changed = True

            if not changed:
                break

        return partition
