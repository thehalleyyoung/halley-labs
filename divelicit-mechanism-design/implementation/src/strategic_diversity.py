"""Game-theoretic diversity mechanisms.

Implements Nash equilibrium computation, correlated equilibrium via LP,
mechanism design with Groves transfers, and incentive compatibility
verification for diversity-maximizing games.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Strategy:
    """A pure strategy available to an agent."""

    name: str
    index: int
    features: NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    def __hash__(self) -> int:
        return hash((self.name, self.index))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Strategy):
            return NotImplemented
        return self.name == other.name and self.index == other.index


@dataclass
class Agent:
    """An agent in the diversity game."""

    id: int
    type_vector: NDArray[np.float64]
    available_strategies: List[Strategy] = field(default_factory=list)
    quality: float = 1.0

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Agent):
            return NotImplemented
        return self.id == other.id


@dataclass
class GamePayoff:
    """Payoff structure for an n-player game.

    ``matrices`` is a list of n arrays, each of shape
    (s_1, s_2, ..., s_n) giving the payoff to the corresponding player
    for every pure-strategy profile.
    """

    n_players: int
    n_strategies: List[int]
    matrices: List[NDArray[np.float64]]


@dataclass
class NashResult:
    """Result of Nash equilibrium computation."""

    equilibrium_strategies: List[NDArray[np.float64]]
    expected_payoffs: NDArray[np.float64]
    diversity_score: float
    is_approximate: bool
    support_sizes: List[int]


@dataclass
class CorrelatedResult:
    """Result of correlated equilibrium computation."""

    joint_distribution: NDArray[np.float64]
    expected_welfare: float
    deviation_incentives: NDArray[np.float64]
    is_feasible: bool
    max_violation: float


@dataclass
class Allocation:
    """Outcome of a mechanism: assignments and monetary transfers."""

    assignments: Dict[int, int]
    transfers: Dict[int, float]
    total_welfare: float
    is_budget_balanced: bool


@dataclass
class Mechanism:
    """Specification of a direct-revelation mechanism."""

    allocation_rule: Callable[[List[NDArray[np.float64]]], Dict[int, int]]
    payment_rule: Callable[
        [List[NDArray[np.float64]], Dict[int, int]], Dict[int, float]
    ]
    n_agents: int
    n_outcomes: int
    description: str = ""
    valuations: Optional[NDArray[np.float64]] = None
    diversity_fn: Optional[Callable[[Dict[int, int], List[Agent]], float]] = None


# ---------------------------------------------------------------------------
# Diversity helpers
# ---------------------------------------------------------------------------

def _pairwise_diversity(profile: Sequence[int], feature_matrix: NDArray[np.float64]) -> float:
    """Average pairwise cosine distance among chosen strategies.

    Parameters
    ----------
    profile : sequence of strategy indices, one per agent.
    feature_matrix : shape (n_strategies, d) feature vectors.

    Returns
    -------
    Diversity score in [0, 1].
    """
    if len(profile) < 2 or feature_matrix.size == 0:
        return 0.0

    vecs = feature_matrix[list(profile)]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = vecs / norms
    sim_matrix = normed @ normed.T
    n = len(profile)
    total_sim = (np.sum(sim_matrix) - np.trace(sim_matrix)) / 2.0
    n_pairs = n * (n - 1) / 2.0
    return float(1.0 - total_sim / n_pairs) if n_pairs > 0 else 0.0


def _build_diversity_payoff(
    agents: List[Agent],
    strategies: List[Strategy],
    quality_weight: float = 0.5,
    diversity_weight: float = 0.5,
) -> GamePayoff:
    """Build a payoff tensor for the diversity game.

    Each agent's payoff for a strategy profile is a convex combination
    of its own quality and the pairwise diversity of the profile.
    """
    n_players = len(agents)
    n_strats = len(strategies)
    shape = tuple([n_strats] * n_players)

    feature_matrix = np.stack([s.features for s in strategies]) if strategies[0].features.size > 0 else np.eye(n_strats)

    matrices: List[NDArray[np.float64]] = []
    for agent_idx in range(n_players):
        payoff = np.zeros(shape, dtype=np.float64)
        for profile in itertools.product(range(n_strats), repeat=n_players):
            div = _pairwise_diversity(profile, feature_matrix)
            q = agents[agent_idx].quality * (1.0 + 0.1 * profile[agent_idx])
            payoff[profile] = quality_weight * q + diversity_weight * div
        matrices.append(payoff)

    return GamePayoff(n_players=n_players, n_strategies=[n_strats] * n_players, matrices=matrices)


# ---------------------------------------------------------------------------
# Support enumeration (2-player)
# ---------------------------------------------------------------------------

def _support_enumeration(
    payoff_matrices: List[NDArray[np.float64]],
    max_support: int = 5,
) -> List[Tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Enumerate Nash equilibria of a 2-player game via support enumeration.

    For each pair of support sizes (k1, k2) up to *max_support*, iterate
    over all combinations of k1 strategies for player 1 and k2 strategies
    for player 2.  For each support pair, solve the indifference equations
    and verify best-response conditions.

    Parameters
    ----------
    payoff_matrices : two matrices A, B each of shape (m, n).
        A[i, j] is player 1's payoff when 1 plays i and 2 plays j.
    max_support : largest support size to enumerate.

    Returns
    -------
    List of (sigma1, sigma2) mixed-strategy pairs that form Nash equilibria.
    """
    A = payoff_matrices[0]
    B = payoff_matrices[1]
    m, n = A.shape
    equilibria: List[Tuple[NDArray[np.float64], NDArray[np.float64]]] = []

    for k1 in range(1, min(max_support, m) + 1):
        for k2 in range(1, min(max_support, n) + 1):
            for sup1 in itertools.combinations(range(m), k1):
                for sup2 in itertools.combinations(range(n), k2):
                    result = _solve_support_pair(A, B, list(sup1), list(sup2))
                    if result is not None:
                        equilibria.append(result)
    return equilibria


def _solve_support_pair(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
    sup1: List[int],
    sup2: List[int],
) -> Optional[Tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Solve the indifference conditions for a fixed support pair.

    Player 2's mixed strategy must make player 1 indifferent over sup1,
    and player 1's mixed strategy must make player 2 indifferent over sup2.
    Additionally, the resulting strategies must be valid probability
    distributions and satisfy best-response conditions.
    """
    m, n = A.shape
    k1, k2 = len(sup1), len(sup2)

    # --- Solve for player 2's strategy that makes player 1 indifferent ---
    # A[sup1, :][:, sup2] @ q = v * 1  =>  augmented system
    A_sub = A[np.ix_(sup1, sup2)]
    lhs2 = np.zeros((k1 + 1, k2 + 1), dtype=np.float64)
    lhs2[:k1, :k2] = A_sub
    lhs2[:k1, k2] = -1.0
    lhs2[k1, :k2] = 1.0
    rhs2 = np.zeros(k1 + 1, dtype=np.float64)
    rhs2[k1] = 1.0

    try:
        sol2 = np.linalg.lstsq(lhs2, rhs2, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    q = sol2[:k2]
    if np.any(q < -1e-10) or abs(np.sum(q) - 1.0) > 1e-8:
        return None
    q = np.maximum(q, 0.0)
    q /= q.sum()

    # --- Solve for player 1's strategy that makes player 2 indifferent ---
    B_sub = B[np.ix_(sup1, sup2)].T
    lhs1 = np.zeros((k2 + 1, k1 + 1), dtype=np.float64)
    lhs1[:k2, :k1] = B_sub
    lhs1[:k2, k1] = -1.0
    lhs1[k2, :k1] = 1.0
    rhs1 = np.zeros(k2 + 1, dtype=np.float64)
    rhs1[k2] = 1.0

    try:
        sol1 = np.linalg.lstsq(lhs1, rhs1, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    p = sol1[:k1]
    if np.any(p < -1e-10) or abs(np.sum(p) - 1.0) > 1e-8:
        return None
    p = np.maximum(p, 0.0)
    p /= p.sum()

    # Expand to full strategy vectors
    sigma1 = np.zeros(m, dtype=np.float64)
    sigma2 = np.zeros(n, dtype=np.float64)
    for i, s in enumerate(sup1):
        sigma1[s] = p[i]
    for j, s in enumerate(sup2):
        sigma2[s] = q[j]

    # --- Best-response verification ---
    payoff1 = A @ sigma2
    val1 = payoff1[sup1[0]]
    if np.any(payoff1 > val1 + 1e-8):
        return None

    payoff2 = sigma1 @ B
    val2 = payoff2[sup2[0]]
    if np.any(payoff2 > val2 + 1e-8):
        return None

    return (sigma1, sigma2)


# ---------------------------------------------------------------------------
# Fictitious play (n-player)
# ---------------------------------------------------------------------------

def _fictitious_play(
    payoff_matrices: List[NDArray[np.float64]],
    n_iterations: int = 10000,
) -> List[NDArray[np.float64]]:
    """Approximate Nash equilibrium via fictitious play.

    Each player maintains empirical counts of opponents' past actions and
    best-responds to the empirical frequency.  The time-averaged strategy
    profile converges to Nash for certain game classes and provides a
    reasonable approximation otherwise.

    Parameters
    ----------
    payoff_matrices : list of n payoff tensors, each of shape
        (s_1, s_2, ..., s_n).
    n_iterations : number of rounds.

    Returns
    -------
    List of mixed-strategy vectors (empirical frequencies), one per player.
    """
    n_players = len(payoff_matrices)
    n_strats = [payoff_matrices[0].shape[i] for i in range(n_players)]
    counts = [np.ones(s, dtype=np.float64) for s in n_strats]
    actions = [0] * n_players

    rng = np.random.default_rng(42)

    for _ in range(n_iterations):
        for player in range(n_players):
            freqs = [counts[j] / counts[j].sum() for j in range(n_players)]
            expected_payoff = _expected_payoff_against(
                payoff_matrices[player], player, freqs
            )
            best_actions = np.flatnonzero(expected_payoff == expected_payoff.max())
            actions[player] = int(rng.choice(best_actions))
        for player in range(n_players):
            counts[player][actions[player]] += 1.0

    return [c / c.sum() for c in counts]


def _expected_payoff_against(
    payoff_tensor: NDArray[np.float64],
    player: int,
    opponent_freqs: List[NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Expected payoff for each pure strategy of *player* given opponent freqs.

    Marginalizes the payoff tensor over all opponents' mixed strategies.
    """
    n_players = payoff_tensor.ndim
    n_strats_player = payoff_tensor.shape[player]
    result = np.zeros(n_strats_player, dtype=np.float64)

    axes_to_contract: List[int] = []
    operands: List[NDArray[np.float64]] = [payoff_tensor]
    subscripts_in = list(range(n_players))
    out_subscript = [player]

    for j in range(n_players):
        if j != player:
            operands.append(opponent_freqs[j])
            axes_to_contract.append(j)

    # Build einsum string
    idx_chars = "abcdefghijklmnop"
    tensor_sub = "".join(idx_chars[i] for i in range(n_players))
    opponent_subs = [idx_chars[j] for j in range(n_players) if j != player]
    out_sub = idx_chars[player]
    subscript_str = tensor_sub + "," + ",".join(opponent_subs) + "->" + out_sub
    result = np.einsum(subscript_str, *operands)
    return result


# ---------------------------------------------------------------------------
# Correlated equilibrium LP
# ---------------------------------------------------------------------------

def _solve_correlated_lp(
    payoff_matrices: List[NDArray[np.float64]],
    diversity_objective: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], bool]:
    """Solve the correlated equilibrium LP via a simple iterative projection.

    Maximizes diversity_objective @ p subject to:
      - incentive-compatibility: for each player i, each strategy s_i in
        support, each alternative s_i', the expected gain from deviating
        when recommended s_i must be <= 0.
      - p >= 0, sum(p) = 1.

    Uses an augmented Lagrangian / projected gradient approach so we avoid
    requiring scipy.optimize.linprog (keeping dependencies minimal).

    Parameters
    ----------
    payoff_matrices : 2-player payoff matrices A, B each of shape (m, n).
    diversity_objective : vector of length m*n giving the diversity value
        of each pure-strategy profile (row-major).

    Returns
    -------
    (p, feasible) where p is the joint distribution over profiles and
    feasible indicates whether the IC constraints are satisfied.
    """
    A = payoff_matrices[0]
    B = payoff_matrices[1]
    m, n = A.shape
    total = m * n

    # Build IC constraint matrix  G @ p <= 0
    constraints: List[NDArray[np.float64]] = []

    # Player 1 IC: for each (s1, s1'), sum over s2 of
    #   p(s1, s2) * [A[s1', s2] - A[s1, s2]] <= 0
    for s1 in range(m):
        for s1_prime in range(m):
            if s1 == s1_prime:
                continue
            row = np.zeros(total, dtype=np.float64)
            for s2 in range(n):
                idx = s1 * n + s2
                row[idx] = A[s1_prime, s2] - A[s1, s2]
            constraints.append(row)

    # Player 2 IC: for each (s2, s2'), sum over s1 of
    #   p(s1, s2) * [B[s1, s2'] - B[s1, s2]] <= 0
    for s2 in range(n):
        for s2_prime in range(n):
            if s2 == s2_prime:
                continue
            row = np.zeros(total, dtype=np.float64)
            for s1 in range(m):
                idx = s1 * n + s2
                row[idx] = B[s1, s2_prime] - B[s1, s2]
            constraints.append(row)

    G = np.array(constraints, dtype=np.float64) if constraints else np.zeros((0, total), dtype=np.float64)

    # Projected gradient ascent on the simplex
    p = np.ones(total, dtype=np.float64) / total
    lr = 0.01
    penalty = 10.0

    for iteration in range(5000):
        grad = diversity_objective.copy()
        if G.shape[0] > 0:
            violations = G @ p
            positive_violations = np.maximum(violations, 0.0)
            grad -= penalty * (G.T @ positive_violations)

        p_new = p + lr * grad
        # Project onto simplex
        p_new = _project_simplex(p_new)
        p = p_new

        # Adaptive penalty
        if iteration % 500 == 499:
            penalty *= 1.5
            lr *= 0.9

    feasible = True
    if G.shape[0] > 0:
        max_violation = float(np.max(G @ p))
        feasible = max_violation < 1e-6
    return p, feasible


def _project_simplex(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Project vector v onto the probability simplex.

    Uses the algorithm of Duchi et al. (2008): sort, then find the
    threshold via cumulative sums.
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = u - (cssv - 1.0) / (np.arange(1, n + 1))
    rho = int(np.max(np.where(rho_candidates > 0)[0])) if np.any(rho_candidates > 0) else 0
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


# ---------------------------------------------------------------------------
# Groves transfers
# ---------------------------------------------------------------------------

def _groves_transfers(
    valuations: NDArray[np.float64],
    allocation: Dict[int, int],
    diversity_fn: Callable[[Dict[int, int], int], float],
) -> Dict[int, float]:
    """Compute Groves (VCG) transfers for a diversity mechanism.

    For each agent i the transfer is:
        t_i = sum_{j != i} v_j(a) + h_i(v_{-i})
    where a is the chosen allocation and h_i is the Clarke pivot:
        h_i = -max_{a'} sum_{j != i} v_j(a')

    This makes truth-telling a dominant strategy.

    Parameters
    ----------
    valuations : shape (n_agents, n_outcomes) – agent i's value for outcome k.
    allocation : mapping from agent id to assigned outcome.
    diversity_fn : diversity_fn(allocation, exclude_agent) -> diversity bonus
        when excluding one agent (used for Clarke pivot).

    Returns
    -------
    transfers : dict mapping agent id to monetary transfer (positive = payment
        to agent, negative = payment from agent).
    """
    n_agents, n_outcomes = valuations.shape
    transfers: Dict[int, float] = {}

    for i in range(n_agents):
        # Sum of others' valuations under chosen allocation
        others_value = sum(
            valuations[j, allocation[j]] for j in range(n_agents) if j != i
        )
        others_value += diversity_fn(allocation, i)

        # Clarke pivot: best welfare for others without agent i
        best_others = -np.inf
        for outcome_i in range(n_outcomes):
            alt_alloc = dict(allocation)
            alt_alloc[i] = outcome_i
            alt_val = sum(
                valuations[j, alt_alloc[j]] for j in range(n_agents) if j != i
            )
            alt_val += diversity_fn(alt_alloc, i)
            if alt_val > best_others:
                best_others = alt_val

        transfers[i] = others_value - best_others

    return transfers


# ---------------------------------------------------------------------------
# IC checking
# ---------------------------------------------------------------------------

def _check_ic_constraint(
    mechanism: Mechanism,
    agent_idx: int,
    true_type: NDArray[np.float64],
    reported_type: NDArray[np.float64],
    all_types: List[NDArray[np.float64]],
) -> bool:
    """Check a single incentive-compatibility constraint.

    Returns True if agent *agent_idx* weakly prefers reporting *true_type*
    over *reported_type* when its true type is *true_type*.
    """
    truthful_reports = list(all_types)
    truthful_reports[agent_idx] = true_type
    truthful_alloc = mechanism.allocation_rule(truthful_reports)
    truthful_transfers = mechanism.payment_rule(truthful_reports, truthful_alloc)

    lying_reports = list(all_types)
    lying_reports[agent_idx] = reported_type
    lying_alloc = mechanism.allocation_rule(lying_reports)
    lying_transfers = mechanism.payment_rule(lying_reports, lying_alloc)

    # Utility = value of allocation + transfer
    agent_outcome_truth = truthful_alloc.get(agent_idx, 0)
    agent_outcome_lie = lying_alloc.get(agent_idx, 0)

    n_outcomes = len(true_type)
    val_truth = true_type[agent_outcome_truth % n_outcomes] if n_outcomes > 0 else 0.0
    val_lie = true_type[agent_outcome_lie % n_outcomes] if n_outcomes > 0 else 0.0

    utility_truth = val_truth + truthful_transfers.get(agent_idx, 0.0)
    utility_lie = val_lie + lying_transfers.get(agent_idx, 0.0)

    return bool(utility_truth >= utility_lie - 1e-10)


# ---------------------------------------------------------------------------
# Diversity social welfare
# ---------------------------------------------------------------------------

def _diversity_social_welfare(allocation: Dict[int, int], agents: List[Agent]) -> float:
    """Compute social welfare including a diversity bonus.

    Welfare = sum of agent qualities * (1 + diversity premium).
    The diversity premium is based on how many distinct outcomes appear.
    """
    if not agents:
        return 0.0

    outcomes = list(allocation.values())
    n_distinct = len(set(outcomes))
    n_agents = len(agents)
    diversity_ratio = n_distinct / max(n_agents, 1)

    base_welfare = sum(a.quality for a in agents)
    diversity_bonus = diversity_ratio * base_welfare * 0.5
    return base_welfare + diversity_bonus


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def nash_diverse_equilibrium(
    agents: List[Agent],
    strategies: List[Strategy],
) -> NashResult:
    """Find Nash equilibria of the diversity game.

    Each agent simultaneously chooses a strategy (response type).  Payoffs
    reward both individual quality and the pairwise diversity of the
    strategy profile.

    For 2-player games with at most 5 strategies per player, exact Nash
    equilibria are found via support enumeration.  For larger games,
    fictitious play is used to approximate an equilibrium.

    Parameters
    ----------
    agents : list of agents participating in the game.
    strategies : list of strategies available to each agent.

    Returns
    -------
    NashResult containing equilibrium mixed strategies, expected payoffs,
    and a diversity score measuring how spread the equilibrium profile is.
    """
    game = _build_diversity_payoff(agents, strategies)
    n_players = game.n_players
    n_strats = len(strategies)

    use_exact = n_players == 2 and n_strats <= 5

    if use_exact:
        A = game.matrices[0].reshape(n_strats, n_strats)
        B = game.matrices[1].reshape(n_strats, n_strats)
        equilibria = _support_enumeration([A, B], max_support=5)

        if equilibria:
            # Pick the equilibrium with highest diversity
            best_eq = None
            best_div = -1.0
            for sigma1, sigma2 in equilibria:
                div = _mixed_diversity(sigma1, sigma2, strategies)
                if div > best_div:
                    best_div = div
                    best_eq = (sigma1, sigma2)

            eq_strategies = [best_eq[0], best_eq[1]]
            exp_payoffs = np.array([
                float(best_eq[0] @ A @ best_eq[1]),
                float(best_eq[0] @ B.T @ best_eq[1]),
            ])
            supports = [int(np.sum(s > 1e-10)) for s in eq_strategies]
            return NashResult(
                equilibrium_strategies=eq_strategies,
                expected_payoffs=exp_payoffs,
                diversity_score=best_div,
                is_approximate=False,
                support_sizes=supports,
            )

    # Fallback / large game: fictitious play
    mixed = _fictitious_play(game.matrices, n_iterations=10000)

    # Compute expected payoffs via the mixed strategy profile
    exp_payoffs = np.zeros(n_players, dtype=np.float64)
    for i in range(n_players):
        exp_payoffs[i] = _expected_payoff_against(
            game.matrices[i], i, mixed
        ) @ mixed[i]

    # Diversity of the mixed profile
    if n_players == 2 and len(mixed) == 2:
        div_score = _mixed_diversity(mixed[0], mixed[1], strategies)
    else:
        div_score = _entropy_diversity(mixed)

    supports = [int(np.sum(m > 1e-10)) for m in mixed]

    return NashResult(
        equilibrium_strategies=mixed,
        expected_payoffs=exp_payoffs,
        diversity_score=div_score,
        is_approximate=True,
        support_sizes=supports,
    )


def _mixed_diversity(
    sigma1: NDArray[np.float64],
    sigma2: NDArray[np.float64],
    strategies: List[Strategy],
) -> float:
    """Expected pairwise diversity under two independent mixed strategies."""
    feature_matrix = np.stack([s.features for s in strategies]) if strategies[0].features.size > 0 else np.eye(len(strategies))
    n = len(strategies)
    total_div = 0.0
    total_prob = 0.0
    for i in range(n):
        for j in range(n):
            prob = sigma1[i] * sigma2[j]
            if prob < 1e-15:
                continue
            div = _pairwise_diversity([i, j], feature_matrix)
            total_div += prob * div
            total_prob += prob
    return total_div / total_prob if total_prob > 0 else 0.0


def _entropy_diversity(mixed: List[NDArray[np.float64]]) -> float:
    """Diversity measured as average entropy of mixed strategies."""
    entropies = []
    for sigma in mixed:
        positive = sigma[sigma > 1e-15]
        entropies.append(float(-np.sum(positive * np.log(positive))))
    max_entropy = np.log(max(len(s) for s in mixed)) if mixed else 1.0
    return float(np.mean(entropies) / max_entropy) if max_entropy > 0 else 0.0


def correlated_equilibrium_diverse(
    game: GamePayoff,
    correlation_device: Optional[NDArray[np.float64]] = None,
) -> CorrelatedResult:
    """Compute a correlated equilibrium that maximizes diversity.

    Uses linear programming to find a joint distribution over strategy
    profiles such that:
      1. No agent wants to deviate from the recommended strategy
         (incentive compatibility).
      2. The expected diversity of the profile is maximized.

    Parameters
    ----------
    game : payoff structure (currently supports 2 players).
    correlation_device : optional initial joint distribution to warm-start.
        If None, starts from uniform.

    Returns
    -------
    CorrelatedResult with the optimal joint distribution, expected welfare,
    and deviation incentives for each player.
    """
    if game.n_players != 2:
        raise NotImplementedError("Correlated equilibrium currently supports 2 players.")

    A = game.matrices[0]
    B = game.matrices[1]
    m, n = A.shape

    # Build diversity objective: prefer profiles where strategies differ
    feature_matrix = np.eye(max(m, n))[:max(m, n), :max(m, n)]
    div_obj = np.zeros(m * n, dtype=np.float64)
    for i in range(m):
        for j in range(n):
            if m == n:
                div_obj[i * n + j] = 1.0 if i != j else 0.0
            else:
                div_obj[i * n + j] = abs(i - j) / max(m, n)

    # Add welfare component
    for i in range(m):
        for j in range(n):
            div_obj[i * n + j] += 0.3 * (A[i, j] + B[i, j])

    p, feasible = _solve_correlated_lp([A, B], div_obj)

    # Compute expected welfare
    welfare = 0.0
    for i in range(m):
        for j in range(n):
            prob = p[i * n + j]
            welfare += prob * (A[i, j] + B[i, j])

    # Compute deviation incentives per player
    dev_incentives = np.zeros(game.n_players, dtype=np.float64)

    # Player 1 max deviation gain
    max_dev1 = -np.inf
    for s1 in range(m):
        conditional_prob = p[s1 * n: (s1 + 1) * n]
        cond_sum = conditional_prob.sum()
        if cond_sum < 1e-12:
            continue
        for s1_prime in range(m):
            if s1_prime == s1:
                continue
            gain = sum(
                conditional_prob[s2] * (A[s1_prime, s2] - A[s1, s2])
                for s2 in range(n)
            )
            max_dev1 = max(max_dev1, gain)
    dev_incentives[0] = max(max_dev1, 0.0)

    # Player 2 max deviation gain
    max_dev2 = -np.inf
    for s2 in range(n):
        conditional_prob = p[s2::n]
        cond_sum = conditional_prob.sum()
        if cond_sum < 1e-12:
            continue
        for s2_prime in range(n):
            if s2_prime == s2:
                continue
            gain = sum(
                conditional_prob[s1_idx] * (B[s1_idx, s2_prime] - B[s1_idx, s2])
                for s1_idx in range(m)
            )
            max_dev2 = max(max_dev2, gain)
    dev_incentives[1] = max(max_dev2, 0.0)

    max_violation = float(np.max(dev_incentives)) if dev_incentives.size > 0 else 0.0

    return CorrelatedResult(
        joint_distribution=p.reshape(m, n),
        expected_welfare=welfare,
        deviation_incentives=dev_incentives,
        is_feasible=feasible,
        max_violation=max_violation,
    )


def mechanism_design_for_diversity(
    agent_types: List[NDArray[np.float64]],
    social_welfare_fn: Callable[[Dict[int, int], List[NDArray[np.float64]]], float],
) -> Mechanism:
    """Design a direct-revelation mechanism for diversity maximization.

    Uses the revelation principle: agents report their types, and the
    mechanism computes an allocation and Groves (VCG) transfers that make
    truthful reporting a dominant strategy.

    The allocation rule maximizes social welfare (including a diversity
    bonus) over all possible outcome assignments.  The payment rule uses
    Groves transfers (Clarke pivot) to ensure incentive compatibility.

    Parameters
    ----------
    agent_types : list of type vectors, one per agent.  agent_types[i] is
        a vector of length n_outcomes giving agent i's value for each outcome.
    social_welfare_fn : function (allocation, types) -> welfare score.

    Returns
    -------
    Mechanism with allocation_rule and payment_rule fully specified.
    """
    n_agents = len(agent_types)
    n_outcomes = agent_types[0].shape[0] if len(agent_types) > 0 else 0
    valuations = np.array(agent_types, dtype=np.float64)

    def allocation_rule(reports: List[NDArray[np.float64]]) -> Dict[int, int]:
        """Find the welfare-maximizing allocation given reported types."""
        n = len(reports)
        n_out = reports[0].shape[0] if n > 0 else 0
        best_alloc: Dict[int, int] = {}
        best_welfare = -np.inf

        # For tractability, use greedy assignment for large instances
        if n_out ** n > 100000:
            return _greedy_diverse_allocation(reports)

        for profile in itertools.product(range(n_out), repeat=n):
            alloc = {i: profile[i] for i in range(n)}
            w = social_welfare_fn(alloc, reports)
            if w > best_welfare:
                best_welfare = w
                best_alloc = dict(alloc)

        return best_alloc

    def payment_rule(
        reports: List[NDArray[np.float64]],
        alloc: Dict[int, int],
    ) -> Dict[int, float]:
        """Compute Groves transfers given reports and allocation."""
        report_matrix = np.array(reports, dtype=np.float64)
        n = len(reports)
        n_out = reports[0].shape[0] if n > 0 else 0

        def diversity_excl(a: Dict[int, int], exclude: int) -> float:
            outcomes = [a[j] for j in a if j != exclude]
            n_dist = len(set(outcomes))
            return 0.5 * n_dist / max(len(outcomes), 1) if outcomes else 0.0

        return _groves_transfers(report_matrix, alloc, diversity_excl)

    return Mechanism(
        allocation_rule=allocation_rule,
        payment_rule=payment_rule,
        n_agents=n_agents,
        n_outcomes=n_outcomes,
        description="Diversity-maximizing VCG mechanism with Groves transfers",
        valuations=valuations,
    )


def _greedy_diverse_allocation(reports: List[NDArray[np.float64]]) -> Dict[int, int]:
    """Greedy allocation that balances value and diversity.

    Assigns agents sequentially, each time picking the outcome that
    maximizes value plus a diversity bonus for choosing an outcome not
    yet taken.
    """
    n = len(reports)
    n_out = reports[0].shape[0]
    alloc: Dict[int, int] = {}
    used_outcomes: set = set()

    order = sorted(range(n), key=lambda i: -np.max(reports[i]))

    for i in order:
        best_outcome = 0
        best_score = -np.inf
        for k in range(n_out):
            score = float(reports[i][k])
            if k not in used_outcomes:
                score += 0.5
            if score > best_score:
                best_score = score
                best_outcome = k
        alloc[i] = best_outcome
        used_outcomes.add(best_outcome)

    return alloc


def revelation_principle_check(mechanism: Mechanism) -> bool:
    """Verify that a mechanism satisfies the revelation principle.

    A mechanism satisfies the revelation principle if, for every possible
    type profile, no agent can gain by misreporting.  This is equivalent
    to verifying dominant-strategy incentive compatibility (DSIC) for the
    direct mechanism.

    We check this by sampling type profiles and verifying IC constraints.
    For each sampled profile, we check all possible single-agent deviations.

    Parameters
    ----------
    mechanism : the direct mechanism to verify.

    Returns
    -------
    True if no profitable deviation is found in the sampled profiles.
    """
    n_agents = mechanism.n_agents
    n_outcomes = mechanism.n_outcomes

    if n_outcomes == 0 or n_agents == 0:
        return True

    rng = np.random.default_rng(123)
    n_samples = min(50, max(10, 3 ** n_outcomes))

    for _ in range(n_samples):
        types = [rng.uniform(0, 10, size=n_outcomes) for _ in range(n_agents)]

        for agent_idx in range(n_agents):
            true_type = types[agent_idx]
            # Check several possible deviations
            for _ in range(min(20, n_outcomes * 2)):
                fake_type = rng.uniform(0, 10, size=n_outcomes)
                if not _check_ic_constraint(
                    mechanism, agent_idx, true_type, fake_type, types
                ):
                    return False

    return True


def incentive_compatibility_test(
    mechanism: Mechanism,
    agent_strategies: List[List[NDArray[np.float64]]],
) -> bool:
    """Test if a mechanism is dominant-strategy incentive compatible (DSIC).

    For each agent and each pair of (true type, alternative report),
    checks that truthful reporting yields at least as high utility as
    any misreport, regardless of other agents' reports.

    Parameters
    ----------
    mechanism : the mechanism to test.
    agent_strategies : agent_strategies[i] is a list of possible type
        vectors that agent i might have or might report.

    Returns
    -------
    True if the mechanism is DSIC across all tested type combinations.
    """
    n_agents = mechanism.n_agents

    if n_agents == 0:
        return True

    # For each agent, test IC against all combinations of others' types
    for agent_idx in range(n_agents):
        own_types = agent_strategies[agent_idx]

        others_type_lists = [agent_strategies[j] for j in range(n_agents) if j != agent_idx]
        if not others_type_lists:
            others_combos: List[List[NDArray[np.float64]]] = [[]]
        else:
            others_combos = [list(combo) for combo in itertools.product(*others_type_lists)]

        # Limit to avoid combinatorial explosion
        max_combos = 200
        if len(others_combos) > max_combos:
            rng = np.random.default_rng(99)
            indices = rng.choice(len(others_combos), size=max_combos, replace=False)
            others_combos = [others_combos[i] for i in indices]

        for true_type in own_types:
            for others in others_combos:
                # Reconstruct the full type profile with truth-telling
                full_types: List[NDArray[np.float64]] = []
                other_iter = iter(others)
                for j in range(n_agents):
                    if j == agent_idx:
                        full_types.append(true_type)
                    else:
                        full_types.append(next(other_iter))

                # Check deviation to every alternative report
                for alt_type in own_types:
                    if np.allclose(alt_type, true_type):
                        continue
                    if not _check_ic_constraint(
                        mechanism, agent_idx, true_type, alt_type, full_types
                    ):
                        return False

    return True


def implement_diverse_allocation(
    mechanism: Mechanism,
    reports: List[NDArray[np.float64]],
) -> Allocation:
    """Execute a mechanism given agent reports.

    Applies the mechanism's allocation rule to determine assignments,
    then computes transfers via the payment rule.  Checks whether the
    mechanism is budget-balanced (sum of transfers <= 0, i.e., the
    mechanism does not inject money).

    Parameters
    ----------
    mechanism : the mechanism to execute.
    reports : list of reported type vectors, one per agent.

    Returns
    -------
    Allocation with per-agent assignments, transfers, total welfare, and
    budget-balance indicator.
    """
    if len(reports) != mechanism.n_agents:
        raise ValueError(
            f"Expected {mechanism.n_agents} reports, got {len(reports)}."
        )

    assignments = mechanism.allocation_rule(reports)
    transfers = mechanism.payment_rule(reports, assignments)

    # Compute total welfare (sum of reported values at assigned outcomes + diversity)
    total_welfare = 0.0
    for i, report in enumerate(reports):
        outcome = assignments.get(i, 0)
        if outcome < len(report):
            total_welfare += float(report[outcome])

    # Diversity bonus
    outcomes = list(assignments.values())
    if outcomes:
        n_distinct = len(set(outcomes))
        total_welfare += 0.5 * n_distinct / len(outcomes) * total_welfare

    # Budget balance: sum of transfers should be <= 0 (no money creation)
    total_transfers = sum(transfers.values())
    is_budget_balanced = total_transfers <= 1e-10

    return Allocation(
        assignments=assignments,
        transfers=transfers,
        total_welfare=total_welfare,
        is_budget_balanced=is_budget_balanced,
    )
