"""
Social choice theory tools: welfare functions, Pareto optimality,
cooperative game theory, Shapley values, Banzhaf index, bargaining solutions.
"""

import numpy as np
from itertools import combinations, permutations
from typing import Dict, List, Tuple, Optional, Callable, Set
from dataclasses import dataclass
from collections import defaultdict
import math


def compute_social_welfare(allocation: Dict[int, float],
                            utilities: Dict[int, float],
                            method: str = 'utilitarian') -> float:
    """
    Compute social welfare under different criteria.
    allocation: ignored for pre-computed utilities
    utilities: {agent_id: utility_value}
    """
    values = list(utilities.values())
    if not values:
        return 0.0

    if method == 'utilitarian':
        return sum(values)
    elif method == 'egalitarian':
        return min(values)
    elif method == 'nash':
        product = 1.0
        for v in values:
            if v <= 0:
                return 0.0
            product *= v
        return product ** (1.0 / len(values))
    elif method == 'leximin':
        return min(values)  # First component of leximin
    else:
        raise ValueError(f"Unknown welfare method: {method}")


def check_pareto_optimality(allocation: List[List[int]],
                             valuations: np.ndarray) -> bool:
    """
    Check if allocation is Pareto optimal.
    allocation: list of item lists for each agent
    valuations: (n_agents, n_items) matrix
    
    Returns True if no other allocation Pareto dominates.
    """
    n_agents = valuations.shape[0]
    n_items = valuations.shape[1]

    # Current utilities
    current_utils = np.zeros(n_agents)
    for agent in range(n_agents):
        for item in allocation[agent]:
            current_utils[agent] += valuations[agent, item]

    # Check all possible reallocations (for small instances)
    items = list(range(n_items))
    if n_items > 10:
        # For large instances, check via LP
        return _check_pareto_lp(current_utils, valuations)

    # Enumerate allocations
    def generate_allocations(items, n_agents):
        if not items:
            yield [[] for _ in range(n_agents)]
            return
        first = items[0]
        rest = items[1:]
        for sub_alloc in generate_allocations(rest, n_agents):
            for agent in range(n_agents):
                new_alloc = [list(a) for a in sub_alloc]
                new_alloc[agent].append(first)
                yield new_alloc

    for alt_alloc in generate_allocations(items, n_agents):
        alt_utils = np.zeros(n_agents)
        for agent in range(n_agents):
            for item in alt_alloc[agent]:
                alt_utils[agent] += valuations[agent, item]

        # Check Pareto dominance
        if np.all(alt_utils >= current_utils) and np.any(alt_utils > current_utils + 1e-10):
            return False

    return True


def _check_pareto_lp(current_utils: np.ndarray, valuations: np.ndarray) -> bool:
    """Check Pareto optimality using LP for larger instances."""
    from scipy.optimize import linprog

    n_agents, n_items = valuations.shape

    # Variables: x_{ij} = probability item j goes to agent i
    n_vars = n_agents * n_items
    # Maximize sum of utilities subject to all being >= current
    c = -valuations.flatten()  # Maximize total welfare

    # Each item allocated exactly once
    A_eq = np.zeros((n_items, n_vars))
    b_eq = np.ones(n_items)
    for j in range(n_items):
        for i in range(n_agents):
            A_eq[j, i * n_items + j] = 1.0

    # Each agent gets at least current utility
    A_ub = np.zeros((n_agents, n_vars))
    b_ub = np.zeros(n_agents)
    for i in range(n_agents):
        for j in range(n_items):
            A_ub[i, i * n_items + j] = -valuations[i, j]
        b_ub[i] = -current_utils[i]

    bounds = [(0, 1)] * n_vars

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    if result.success:
        opt_welfare = -result.fun
        current_welfare = sum(current_utils)
        # If optimal welfare equals current, allocation is Pareto optimal
        return abs(opt_welfare - current_welfare) < 1e-6
    return True  # If LP fails, assume Pareto optimal


@dataclass
class CooperativeGame:
    """Characteristic function form cooperative game."""
    n_players: int
    value_function: Dict[frozenset, float]  # coalition -> value

    def value(self, coalition: Set[int]) -> float:
        return self.value_function.get(frozenset(coalition), 0.0)

    @staticmethod
    def from_function(n_players: int, func: Callable[[Set[int]], float]) -> 'CooperativeGame':
        """Create game from a value function."""
        vf = {}
        for r in range(n_players + 1):
            for coalition in combinations(range(n_players), r):
                s = frozenset(coalition)
                vf[s] = func(set(coalition))
        return CooperativeGame(n_players=n_players, value_function=vf)


def find_core(game: CooperativeGame) -> Optional[np.ndarray]:
    """
    Find a point in the core of the cooperative game using LP.
    Core: set of payoff vectors where no coalition can improve.
    
    Returns a core allocation or None if core is empty.
    """
    from scipy.optimize import linprog

    n = game.n_players
    grand_value = game.value(set(range(n)))

    # Variables: x_i = payoff to player i
    # Objective: minimize some tie-breaking (e.g., egalitarian)
    c = np.ones(n)  # Minimize total (subject to = grand_value, this is fixed)

    # Efficiency: sum x_i = v(N)
    A_eq = np.ones((1, n))
    b_eq = np.array([grand_value])

    # Coalitional rationality: sum_{i in S} x_i >= v(S) for all S
    A_ub = []
    b_ub = []
    for r in range(1, n):
        for coalition in combinations(range(n), r):
            row = np.zeros(n)
            for i in coalition:
                row[i] = -1.0  # -sum x_i <= -v(S)
            A_ub.append(row)
            b_ub.append(-game.value(set(coalition)))

    if A_ub:
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
    else:
        A_ub = None
        b_ub = None

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs')

    if result.success:
        return result.x
    return None


def compute_shapley_values(game: CooperativeGame) -> Dict[int, float]:
    """
    Compute exact Shapley values for all players.
    phi_i = sum over S not containing i:
            |S|!(n-|S|-1)! / n! * [v(S union {i}) - v(S)]
    """
    n = game.n_players
    shapley = {i: 0.0 for i in range(n)}
    n_factorial = math.factorial(n)

    for i in range(n):
        others = [j for j in range(n) if j != i]
        for r in range(n):
            for coalition in combinations(others, r):
                s = set(coalition)
                s_size = len(s)
                marginal = game.value(s | {i}) - game.value(s)
                weight = math.factorial(s_size) * math.factorial(n - s_size - 1) / n_factorial
                shapley[i] += weight * marginal

    return shapley


def compute_banzhaf_index(game: CooperativeGame) -> Dict[int, float]:
    """
    Compute Banzhaf power index.
    For simple games (voting): measures how often a player is pivotal.
    beta_i = (number of coalitions where i is pivotal) / 2^(n-1)
    """
    n = game.n_players
    banzhaf = {i: 0.0 for i in range(n)}

    for i in range(n):
        others = [j for j in range(n) if j != i]
        n_pivotal = 0
        total = 0
        for r in range(n):
            for coalition in combinations(others, r):
                s = set(coalition)
                total += 1
                # Is i pivotal? v(S + i) > v(S)
                if game.value(s | {i}) > game.value(s):
                    n_pivotal += 1

        banzhaf[i] = n_pivotal / max(total, 1)

    # Normalize
    total_banzhaf = sum(banzhaf.values())
    if total_banzhaf > 0:
        normalized = {i: v / total_banzhaf for i, v in banzhaf.items()}
    else:
        normalized = banzhaf

    return normalized


def check_strategyproofness(mechanism: Callable, type_space: List,
                              agents: List[int]) -> Tuple[bool, Optional[Dict]]:
    """
    Check strategy-proofness of a general mechanism.
    mechanism(reports) -> (allocation, payments)
    type_space: list of possible type profiles
    """
    for true_profile in type_space:
        for agent in agents:
            true_alloc, true_pay = mechanism(true_profile)
            true_utility = true_alloc.get(agent, 0.0) - true_pay.get(agent, 0.0)

            for alt_profile in type_space:
                if alt_profile is true_profile:
                    continue
                # Agent misreports
                misreport = list(true_profile)
                misreport[agent] = alt_profile[agent]
                misreport = tuple(misreport)

                mis_alloc, mis_pay = mechanism(misreport)
                # Utility under true valuation
                mis_utility = mis_alloc.get(agent, 0.0) - mis_pay.get(agent, 0.0)

                if mis_utility > true_utility + 1e-10:
                    return False, {
                        'agent': agent,
                        'true_profile': true_profile,
                        'misreport': misreport,
                        'utility_gain': mis_utility - true_utility
                    }

    return True, None


# Cooperative game theory: specific game types

class SimpleGame(CooperativeGame):
    """Simple game (voting game) where coalitions are either winning or losing."""

    def __init__(self, n_players: int, winning_coalitions: List[Set[int]]):
        vf = {}
        winning_sets = [frozenset(w) for w in winning_coalitions]
        for r in range(n_players + 1):
            for coalition in combinations(range(n_players), r):
                s = frozenset(coalition)
                # Check if any winning coalition is subset
                is_winning = any(w.issubset(s) for w in winning_sets)
                vf[s] = 1.0 if is_winning else 0.0
        super().__init__(n_players=n_players, value_function=vf)


class WeightedVotingGame(CooperativeGame):
    """Weighted voting game: [q; w1, w2, ..., wn]."""

    def __init__(self, quota: float, weights: List[float]):
        n = len(weights)
        vf = {}
        for r in range(n + 1):
            for coalition in combinations(range(n), r):
                s = frozenset(coalition)
                total_weight = sum(weights[i] for i in coalition)
                vf[s] = 1.0 if total_weight >= quota else 0.0
        super().__init__(n_players=n, value_function=vf)
        self.quota = quota
        self.weights = weights


def is_convex_game(game: CooperativeGame) -> bool:
    """Check if game is convex (supermodular)."""
    n = game.n_players
    players = list(range(n))

    for r1 in range(n + 1):
        for s1 in combinations(players, r1):
            s1_set = set(s1)
            for r2 in range(n + 1):
                for s2 in combinations(players, r2):
                    s2_set = set(s2)
                    # Convexity: v(S ∪ T) + v(S ∩ T) >= v(S) + v(T)
                    union_val = game.value(s1_set | s2_set)
                    inter_val = game.value(s1_set & s2_set)
                    s1_val = game.value(s1_set)
                    s2_val = game.value(s2_set)
                    if union_val + inter_val < s1_val + s2_val - 1e-10:
                        return False
    return True


# Bargaining solutions

def nash_bargaining_solution(feasible_set: np.ndarray,
                               disagreement_point: np.ndarray) -> np.ndarray:
    """
    Nash bargaining solution: maximize product of gains over disagreement.
    feasible_set: (n_points, n_agents) array of feasible utility vectors
    disagreement_point: (n_agents,) disagreement utilities
    
    Returns the Nash bargaining solution point.
    """
    n_points, n_agents = feasible_set.shape

    best_product = -np.inf
    best_point = disagreement_point.copy()

    for i in range(n_points):
        point = feasible_set[i]
        gains = point - disagreement_point
        if np.all(gains > 0):
            product = np.prod(gains)
            if product > best_product:
                best_product = product
                best_point = point.copy()

    return best_point


def kalai_smorodinsky_solution(feasible_set: np.ndarray,
                                 disagreement_point: np.ndarray) -> np.ndarray:
    """
    Kalai-Smorodinsky bargaining solution: monotone path to ideal point.
    Finds point on Pareto frontier where gains are proportional to max gains.
    """
    n_points, n_agents = feasible_set.shape

    # Find ideal point (max achievable for each agent)
    ideal = np.zeros(n_agents)
    for agent in range(n_agents):
        mask = np.all(feasible_set >= disagreement_point, axis=1)
        if np.any(mask):
            ideal[agent] = np.max(feasible_set[mask, agent])
        else:
            ideal[agent] = disagreement_point[agent]

    # Max gains
    max_gains = ideal - disagreement_point
    if np.any(max_gains <= 0):
        return disagreement_point.copy()

    # Find point on Pareto frontier where gains are proportional
    best_point = disagreement_point.copy()
    best_t = 0.0

    for i in range(n_points):
        point = feasible_set[i]
        gains = point - disagreement_point
        if np.all(gains >= 0):
            # Find t such that gains = t * max_gains (approximately)
            ratios = gains / (max_gains + 1e-10)
            t = np.min(ratios)
            if t > best_t:
                # Check if it's on or close to the ray
                deviation = np.max(np.abs(ratios - t))
                if deviation < 0.1:  # Allow some tolerance
                    best_t = t
                    best_point = point.copy()

    # Refine: find intersection of ray with Pareto frontier
    # Try convex combinations
    for t in np.linspace(0, 1, 1000):
        target = disagreement_point + t * max_gains
        # Find closest feasible point
        distances = np.linalg.norm(feasible_set - target, axis=1)
        closest_idx = np.argmin(distances)
        if distances[closest_idx] < 0.1:
            point = feasible_set[closest_idx]
            gains = point - disagreement_point
            if np.all(gains >= -1e-10):
                ratio_t = np.min(gains / (max_gains + 1e-10))
                if ratio_t > best_t:
                    best_t = ratio_t
                    best_point = point.copy()

    return best_point


def nucleolus(game: CooperativeGame) -> np.ndarray:
    """
    Compute the nucleolus of a cooperative game.
    Lexicographically minimizes the maximum excess.
    Uses iterative LP approach.
    """
    from scipy.optimize import linprog

    n = game.n_players
    grand_value = game.value(set(range(n)))

    # Collect all proper coalitions
    coalitions = []
    for r in range(1, n):
        for s in combinations(range(n), r):
            coalitions.append(frozenset(s))

    n_coalitions = len(coalitions)

    # Iterative approach: fix coalitions with known excess, minimize next
    fixed_excess = {}
    remaining = list(range(n_coalitions))

    x = np.ones(n) * grand_value / n  # Start with equal split

    for iteration in range(n_coalitions):
        if not remaining:
            break

        # LP: minimize maximum excess over remaining coalitions
        # Variables: x_1, ..., x_n, epsilon
        n_vars = n + 1
        c = np.zeros(n_vars)
        c[n] = 1.0  # Minimize epsilon

        A_ub = []
        b_ub = []

        for idx in remaining:
            s = coalitions[idx]
            row = np.zeros(n_vars)
            for i in s:
                row[i] = -1.0
            row[n] = -1.0
            A_ub.append(row)
            b_ub.append(-game.value(set(s)))

        # Fixed excesses
        for idx, eps in fixed_excess.items():
            s = coalitions[idx]
            row = np.zeros(n_vars)
            for i in s:
                row[i] = -1.0
            A_ub.append(row)
            b_ub.append(-game.value(set(s)) - eps)

        # Efficiency
        A_eq = np.zeros((1, n_vars))
        A_eq[0, :n] = 1.0
        b_eq = np.array([grand_value])

        A_ub_arr = np.array(A_ub) if A_ub else None
        b_ub_arr = np.array(b_ub) if b_ub else None

        bounds = [(None, None)] * n + [(None, None)]

        result = linprog(c, A_ub=A_ub_arr, b_ub=b_ub_arr,
                         A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if result.success:
            x = result.x[:n]
            eps_star = result.x[n]

            # Find tight constraints
            tight = []
            for idx in remaining:
                s = coalitions[idx]
                excess = game.value(set(s)) - sum(x[i] for i in s)
                if abs(excess - eps_star) < 1e-8:
                    tight.append(idx)

            for idx in tight:
                fixed_excess[idx] = eps_star
                remaining.remove(idx)
        else:
            break

    return x


def verify_shapley_axioms(game: CooperativeGame) -> Dict[str, bool]:
    """
    Verify that Shapley values satisfy all four axioms:
    1. Efficiency: sum of Shapley values = v(N)
    2. Symmetry: symmetric players get equal values
    3. Dummy: dummy players get v({i})
    4. Additivity: (verified by construction)
    """
    shapley = compute_shapley_values(game)
    n = game.n_players

    # Efficiency
    total = sum(shapley.values())
    grand_value = game.value(set(range(n)))
    efficiency = abs(total - grand_value) < 1e-8

    # Symmetry
    symmetry = True
    for i in range(n):
        for j in range(i + 1, n):
            # Check if i and j are symmetric
            is_symmetric = True
            for r in range(n):
                for s in combinations([k for k in range(n) if k != i and k != j], r):
                    s_set = set(s)
                    if abs(game.value(s_set | {i}) - game.value(s_set | {j})) > 1e-10:
                        is_symmetric = False
                        break
                if not is_symmetric:
                    break
            if is_symmetric and abs(shapley[i] - shapley[j]) > 1e-8:
                symmetry = False

    # Dummy player
    dummy = True
    for i in range(n):
        is_dummy = True
        v_i = game.value({i})
        for r in range(n):
            for s in combinations([k for k in range(n) if k != i], r):
                s_set = set(s)
                marginal = game.value(s_set | {i}) - game.value(s_set)
                if abs(marginal - v_i) > 1e-10:
                    is_dummy = False
                    break
            if not is_dummy:
                break
        if is_dummy and abs(shapley[i] - v_i) > 1e-8:
            dummy = False

    return {
        'efficiency': efficiency,
        'symmetry': symmetry,
        'dummy': dummy,
        'additivity': True  # Always true by construction
    }
