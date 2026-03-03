"""
Privacy mechanism games: game-theoretic models for DP mechanism design.

Provides domain-specific game formulations that embed differential privacy
semantics—adversary models, utility functions, information design, and
auction-based budget allocation—into standard game-theoretic frameworks.
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
    GameResult,
    GameType,
    PlayerRole,
    StackelbergResult,
    Strategy,
)
from dp_forge.types import (
    AdjacencyRelation,
    GameMatrix,
    OptimalityCertificate,
    PrivacyBudget,
)


# ---------------------------------------------------------------------------
# PrivacyGame
# ---------------------------------------------------------------------------


class PrivacyGame:
    """Formalise DP mechanism design as a two-player game.

    The designer chooses a mechanism (noise distribution), the adversary
    chooses an attack (neighbouring database pair + distinguishing test).
    The payoff is the privacy loss of the mechanism under that attack.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 0.0,
        n_mechanisms: int = 5,
        n_attacks: int = 5,
    ) -> None:
        self.epsilon = epsilon
        self.delta = delta
        self.n_mechanisms = n_mechanisms
        self.n_attacks = n_attacks

    def build_game(
        self,
        sensitivity: float = 1.0,
    ) -> GameMatrix:
        """Build the zero-sum privacy game payoff matrix.

        Row = designer strategies (noise scales).
        Column = adversary strategies (attack thresholds).
        Payoff = adversary's advantage (to be minimised by designer).

        Args:
            sensitivity: Query sensitivity.

        Returns:
            GameMatrix where payoffs[i, j] is adversary's advantage.
        """
        m = self.n_mechanisms
        n = self.n_attacks

        # Designer strategies: Laplace noise with varying scale
        base_scale = sensitivity / self.epsilon
        scales = base_scale * np.linspace(0.5, 2.0, m)

        # Adversary strategies: hypothesis testing thresholds
        thresholds = np.linspace(0.1 * sensitivity, 3.0 * sensitivity, n)

        payoff = np.zeros((m, n))
        for i, sigma in enumerate(scales):
            for j, t in enumerate(thresholds):
                # Adversary advantage: difference in tail probabilities
                # P[Lap(0, sigma) > t] vs P[Lap(sensitivity, sigma) > t]
                p_null = 0.5 * np.exp(-t / sigma)
                p_alt = 0.5 * np.exp(-abs(t - sensitivity) / sigma)
                payoff[i, j] = abs(p_alt - p_null)

        return GameMatrix(payoffs=payoff)

    def solve_minimax(self, sensitivity: float = 1.0) -> GameResult:
        """Solve the privacy game for the minimax mechanism."""
        from dp_forge.game_theory.minimax import MinimaxSolver

        game = self.build_game(sensitivity)
        solver = MinimaxSolver()
        return solver.solve(game)

    def solve_stackelberg(self, sensitivity: float = 1.0) -> StackelbergResult:
        """Solve the privacy game as a Stackelberg game."""
        from dp_forge.game_theory.stackelberg import StackelbergSolver

        game = self.build_game(sensitivity)
        leader_payoff = GameMatrix(payoffs=-game.payoffs)
        follower_payoff = game
        solver = StackelbergSolver()
        return solver.solve(leader_payoff, follower_payoff)


# ---------------------------------------------------------------------------
# AdversaryModel
# ---------------------------------------------------------------------------


@dataclass
class AdversaryCapabilities:
    """Description of adversary's capabilities."""
    has_auxiliary_info: bool = False
    n_auxiliary_records: int = 0
    knows_mechanism: bool = True
    has_side_channel: bool = False
    computational_bound: Optional[float] = None


class AdversaryModel:
    """Model adversary capabilities for privacy games.

    Captures what the adversary knows (auxiliary information, mechanism
    knowledge) and can do (computational bounds, side channels).
    """

    def __init__(
        self,
        capabilities: Optional[AdversaryCapabilities] = None,
    ) -> None:
        self.capabilities = capabilities or AdversaryCapabilities()

    def attack_payoff(
        self,
        mechanism_probs: npt.NDArray[np.float64],
        true_db: int,
        guess_db: int,
    ) -> float:
        """Compute the adversary's payoff for guessing a database.

        Args:
            mechanism_probs: n x k mechanism probability table.
            true_db: True database index.
            guess_db: Adversary's guessed database index.

        Returns:
            Adversary's payoff (higher = better attack).
        """
        P = np.asarray(mechanism_probs, dtype=np.float64)
        if true_db >= P.shape[0] or guess_db >= P.shape[0]:
            return 0.0

        # Statistical distance between rows
        dist = 0.5 * float(np.sum(np.abs(P[true_db] - P[guess_db])))

        # Auxiliary info bonus
        aux_bonus = 0.0
        if self.capabilities.has_auxiliary_info:
            aux_bonus = min(0.2, 0.05 * self.capabilities.n_auxiliary_records)

        return dist + aux_bonus

    def build_attack_matrix(
        self,
        mechanism_probs: npt.NDArray[np.float64],
        adjacency: AdjacencyRelation,
    ) -> npt.NDArray[np.float64]:
        """Build the adversary's payoff matrix.

        Rows = adjacent pairs, Columns = distinguishing tests.

        Args:
            mechanism_probs: n x k mechanism probability table.
            adjacency: Adjacent database pairs.

        Returns:
            Attack payoff matrix.
        """
        P = np.asarray(mechanism_probs, dtype=np.float64)
        n, k = P.shape
        n_pairs = len(adjacency.edges)

        # Each distinguishing test = a subset of outputs (here: single bins)
        attack_matrix = np.zeros((n_pairs, k))
        for idx, (i, j) in enumerate(adjacency.edges):
            if i >= n or j >= n:
                continue
            for b in range(k):
                # Test T_b: output == b
                # Advantage = |P[i,b] - P[j,b]|
                attack_matrix[idx, b] = abs(P[i, b] - P[j, b])

        return attack_matrix

    def optimal_attack(
        self,
        mechanism_probs: npt.NDArray[np.float64],
        adjacency: AdjacencyRelation,
    ) -> Tuple[Tuple[int, int], int, float]:
        """Find the optimal attack for the adversary.

        Returns:
            (worst_pair, best_test, advantage)
        """
        attack = self.build_attack_matrix(mechanism_probs, adjacency)
        if attack.size == 0:
            return ((0, 0), 0, 0.0)

        idx = np.unravel_index(np.argmax(attack), attack.shape)
        pair_idx, test_idx = int(idx[0]), int(idx[1])
        pair = adjacency.edges[pair_idx] if pair_idx < len(adjacency.edges) else (0, 0)
        return pair, test_idx, float(attack[pair_idx, test_idx])


# ---------------------------------------------------------------------------
# UtilityFunction
# ---------------------------------------------------------------------------


class UtilityFunction:
    """Utility models for data analyst and adversary.

    Provides parametric utility functions and methods to compute expected
    utility under a mechanism.
    """

    def __init__(self, loss_type: str = "squared_error") -> None:
        self.loss_type = loss_type
        self._loss_fn = self._get_loss_fn(loss_type)

    @staticmethod
    def _get_loss_fn(loss_type: str) -> Callable[[float, float], float]:
        if loss_type == "squared_error":
            return lambda t, n: (t - n) ** 2
        elif loss_type == "absolute_error":
            return lambda t, n: abs(t - n)
        elif loss_type == "zero_one":
            return lambda t, n: 0.0 if abs(t - n) < 0.5 else 1.0
        elif loss_type == "relative_error":
            return lambda t, n: abs(t - n) / max(abs(t), 1e-10)
        else:
            return lambda t, n: (t - n) ** 2

    def analyst_utility(
        self,
        true_values: npt.NDArray[np.float64],
        output_grid: npt.NDArray[np.float64],
        mechanism: npt.NDArray[np.float64],
    ) -> float:
        """Compute expected analyst utility (negative loss) under mechanism.

        Args:
            true_values: f(x_i) for each database.
            output_grid: y_j output values.
            mechanism: n x k probability table.

        Returns:
            Expected negative loss (higher = better for analyst).
        """
        P = np.asarray(mechanism, dtype=np.float64)
        fv = np.asarray(true_values, dtype=np.float64)
        yg = np.asarray(output_grid, dtype=np.float64)
        n, k = P.shape

        total_loss = 0.0
        for i in range(n):
            for j in range(k):
                total_loss += P[i, j] * self._loss_fn(float(fv[i]), float(yg[j]))

        return -total_loss / n

    def adversary_utility(
        self,
        mechanism: npt.NDArray[np.float64],
        pair: Tuple[int, int],
    ) -> float:
        """Compute adversary's utility for distinguishing a pair.

        Uses total variation distance as the adversary's payoff.

        Args:
            mechanism: n x k probability table.
            pair: Adjacent database pair (i, j).

        Returns:
            Total variation distance (adversary's advantage).
        """
        P = np.asarray(mechanism, dtype=np.float64)
        i, j = pair
        if i >= P.shape[0] or j >= P.shape[0]:
            return 0.0
        return 0.5 * float(np.sum(np.abs(P[i] - P[j])))

    def privacy_utility_tradeoff(
        self,
        sensitivity: float,
        epsilon_values: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute the privacy-utility tradeoff curve.

        For Laplace mechanism: utility_loss = 2*(sensitivity/epsilon)^2,
        privacy_loss = epsilon.

        Returns:
            (utility_losses, privacy_losses)
        """
        eps = np.asarray(epsilon_values, dtype=np.float64)
        if self.loss_type == "squared_error":
            utility_losses = 2.0 * (sensitivity / eps) ** 2
        elif self.loss_type == "absolute_error":
            utility_losses = sensitivity / eps
        else:
            utility_losses = 2.0 * (sensitivity / eps) ** 2
        return utility_losses, eps.copy()


# ---------------------------------------------------------------------------
# InformationDesign
# ---------------------------------------------------------------------------


class InformationDesign:
    """Bayesian persuasion for privacy mechanism design.

    The designer (sender) commits to a signalling scheme that maps the true
    database to a signal.  The adversary (receiver) updates beliefs and
    takes an action.  The designer chooses the scheme to minimise privacy
    leakage while maintaining utility.
    """

    def __init__(self, tol: float = 1e-8) -> None:
        self._tol = tol

    def optimal_signalling(
        self,
        prior: npt.NDArray[np.float64],
        sender_payoff: npt.NDArray[np.float64],
        receiver_payoff: npt.NDArray[np.float64],
        n_signals: int = 0,
    ) -> npt.NDArray[np.float64]:
        """Compute the optimal signalling scheme via LP.

        The sender commits to pi[s | theta] for each state theta and
        signal s.  The receiver best-responds to the posterior.

        Args:
            prior: Prior distribution over states (n_states,).
            sender_payoff: Sender payoff matrix (n_states x n_actions).
            receiver_payoff: Receiver payoff matrix (n_states x n_actions).
            n_signals: Number of signals (default: n_actions).

        Returns:
            Signalling scheme pi (n_signals x n_states) where
            pi[s, theta] = P[signal=s | state=theta].
        """
        prior = np.asarray(prior, dtype=np.float64)
        S = np.asarray(sender_payoff, dtype=np.float64)
        R = np.asarray(receiver_payoff, dtype=np.float64)
        n_states, n_actions = S.shape

        if n_signals <= 0:
            n_signals = n_actions

        # Variables: tau[s, theta] = prior[theta] * pi[s | theta]
        # (joint distribution over signals and states)
        n_vars = n_signals * n_states

        # Sender wants to maximise expected payoff when receiver best-responds
        # For each signal s, receiver picks action a(s) = argmax_a sum_theta tau[s,theta] R[theta,a]
        # This is a bilinear problem; we linearise by enumerating receiver responses.

        best_obj = -np.inf
        best_tau = np.ones((n_signals, n_states)) * prior / n_signals

        # Enumerate receiver response mappings: a: signals -> actions
        # For small n_signals * n_actions, this is tractable
        max_enum = min(n_actions ** n_signals, 1000)

        if n_actions ** n_signals <= max_enum:
            for combo in product(range(n_actions), repeat=n_signals):
                # For this receiver response mapping, solve LP for tau
                tau, obj = self._solve_for_response(
                    prior, S, R, n_signals, list(combo)
                )
                if tau is not None and obj > best_obj + self._tol:
                    best_obj = obj
                    best_tau = tau
        else:
            # Heuristic: try each signal mapped to a different action
            for perm_start in range(n_actions):
                combo = [(perm_start + s) % n_actions for s in range(n_signals)]
                tau, obj = self._solve_for_response(
                    prior, S, R, n_signals, combo
                )
                if tau is not None and obj > best_obj + self._tol:
                    best_obj = obj
                    best_tau = tau

        # Convert tau to conditional: pi[s | theta] = tau[s, theta] / prior[theta]
        pi = np.zeros((n_signals, n_states))
        for theta in range(n_states):
            if prior[theta] > self._tol:
                pi[:, theta] = best_tau[:, theta] / prior[theta]
            else:
                pi[:, theta] = 1.0 / n_signals

        # Normalise columns
        col_sums = pi.sum(axis=0)
        for theta in range(n_states):
            if col_sums[theta] > self._tol:
                pi[:, theta] /= col_sums[theta]
            else:
                pi[:, theta] = 1.0 / n_signals

        return pi

    def _solve_for_response(
        self,
        prior: npt.NDArray[np.float64],
        S: npt.NDArray[np.float64],
        R: npt.NDArray[np.float64],
        n_signals: int,
        response: List[int],
    ) -> Tuple[Optional[npt.NDArray[np.float64]], float]:
        """Solve the sender's LP for a fixed receiver response mapping."""
        n_states = len(prior)

        # Variables: tau[s, theta] for s in range(n_signals), theta in range(n_states)
        n_vars = n_signals * n_states

        # Objective: max sum_{s,theta} tau[s,theta] S[theta, response[s]]
        c = np.zeros(n_vars)
        for s in range(n_signals):
            a = response[s]
            for theta in range(n_states):
                c[s * n_states + theta] = -S[theta, a]  # minimise negative

        # Marginal constraint: sum_s tau[s,theta] = prior[theta]
        eq_rows = []
        eq_rhs = []
        for theta in range(n_states):
            row = np.zeros(n_vars)
            for s in range(n_signals):
                row[s * n_states + theta] = 1.0
            eq_rows.append(row)
            eq_rhs.append(float(prior[theta]))

        # IC constraints: receiver prefers response[s] to any other a'
        # sum_theta tau[s,theta] R[theta, response[s]] >= sum_theta tau[s,theta] R[theta, a']
        ub_rows = []
        ub_rhs = []
        n_actions = S.shape[1]
        for s in range(n_signals):
            a_s = response[s]
            for ap in range(n_actions):
                if ap == a_s:
                    continue
                row = np.zeros(n_vars)
                for theta in range(n_states):
                    # tau[s,theta] (R[theta, ap] - R[theta, a_s]) <= 0
                    row[s * n_states + theta] = R[theta, ap] - R[theta, a_s]
                ub_rows.append(row)
                ub_rhs.append(0.0)

        A_eq = np.array(eq_rows)
        b_eq = np.array(eq_rhs)
        A_ub = np.array(ub_rows) if ub_rows else np.zeros((0, n_vars))
        b_ub = np.array(ub_rhs) if ub_rhs else np.zeros(0)
        bounds = [(0.0, None)] * n_vars

        try:
            res = sp_opt.linprog(
                c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=bounds, method="highs",
            )
            if res.success:
                tau = np.maximum(res.x, 0.0).reshape(n_signals, n_states)
                return tau, float(-res.fun)
        except Exception:
            pass
        return None, -np.inf

    def compute_value_of_information(
        self,
        prior: npt.NDArray[np.float64],
        sender_payoff: npt.NDArray[np.float64],
        receiver_payoff: npt.NDArray[np.float64],
    ) -> float:
        """Value of the sender's ability to design signals.

        = Sender's optimal utility - Sender's utility with no information.

        Returns:
            Non-negative value of information design.
        """
        prior = np.asarray(prior, dtype=np.float64)
        S = np.asarray(sender_payoff, dtype=np.float64)
        R = np.asarray(receiver_payoff, dtype=np.float64)

        # No information: receiver picks action maximising E[R]
        expected_R = prior @ R
        a_no_info = int(np.argmax(expected_R))
        sender_no_info = float(prior @ S[:, a_no_info])

        # Optimal information design
        pi = self.optimal_signalling(prior, S, R)
        n_signals = pi.shape[0]

        # Compute sender utility under optimal scheme
        sender_opt = 0.0
        for s in range(n_signals):
            tau_s = prior * pi[s]
            tau_s_sum = tau_s.sum()
            if tau_s_sum < self._tol:
                continue
            posterior = tau_s / tau_s_sum
            a_s = int(np.argmax(posterior @ R))
            sender_opt += tau_s_sum * float(posterior @ S[:, a_s])

        return max(0.0, sender_opt - sender_no_info)


# ---------------------------------------------------------------------------
# AuctionMechanism
# ---------------------------------------------------------------------------


class AuctionMechanism:
    """Auction-based privacy budget allocation.

    Multiple data analysts bid for shares of a finite privacy budget.
    The mechanism allocates epsilon to each analyst and charges payments
    to ensure truthful bidding (incentive compatibility).
    """

    def __init__(self, total_epsilon: float = 1.0, tol: float = 1e-8) -> None:
        self.total_epsilon = total_epsilon
        self._tol = tol

    def vcg_allocation(
        self,
        valuations: npt.NDArray[np.float64],
        demands: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """VCG (Vickrey-Clarke-Groves) auction for epsilon allocation.

        Each analyst i has valuation v_i(eps_i) = valuations[i] * eps_i
        and demands at most demands[i] epsilon.

        Args:
            valuations: Per-unit valuation for each analyst.
            demands: Maximum epsilon demand for each analyst.

        Returns:
            (allocations, payments) for each analyst.
        """
        v = np.asarray(valuations, dtype=np.float64)
        d = np.asarray(demands, dtype=np.float64)
        n = len(v)

        # Efficient allocation: maximise sum v_i * eps_i s.t. sum eps_i <= total, eps_i <= d_i
        c = -v  # maximise valuations
        bounds = [(0.0, float(d[i])) for i in range(n)]
        A_ub = np.ones((1, n))
        b_ub = np.array([self.total_epsilon])

        res = sp_opt.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not res.success:
            # Fallback: proportional allocation
            total_demand = d.sum()
            if total_demand > self.total_epsilon:
                alloc = d * self.total_epsilon / total_demand
            else:
                alloc = d.copy()
            return alloc, np.zeros(n)

        alloc = np.maximum(res.x, 0.0)
        social_welfare = float(v @ alloc)

        # VCG payments: p_i = SW_{-i} - (SW - v_i * alloc_i)
        payments = np.zeros(n)
        for i in range(n):
            # Solve without agent i
            v_minus_i = v.copy()
            v_minus_i[i] = 0.0
            c_mi = -v_minus_i
            res_mi = sp_opt.linprog(
                c_mi, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs"
            )
            if res_mi.success:
                sw_minus_i = float(v_minus_i @ np.maximum(res_mi.x, 0.0))
            else:
                sw_minus_i = social_welfare - v[i] * alloc[i]
            # Payment = externality
            payments[i] = sw_minus_i - (social_welfare - v[i] * alloc[i])

        return alloc, np.maximum(payments, 0.0)

    def proportional_allocation(
        self,
        bids: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Simple proportional allocation of privacy budget.

        Args:
            bids: Bids from each analyst (proportional weights).

        Returns:
            Epsilon allocation proportional to bids.
        """
        b = np.asarray(bids, dtype=np.float64)
        b = np.maximum(b, 0.0)
        total = b.sum()
        if total < self._tol:
            return np.ones(len(b)) * self.total_epsilon / len(b)
        return self.total_epsilon * b / total

    def second_price_auction(
        self,
        bids: npt.NDArray[np.float64],
        n_winners: int = 1,
    ) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """Second-price auction for privacy budget slots.

        Args:
            bids: Bid from each analyst.
            n_winners: Number of budget slots to allocate.

        Returns:
            (winner_indices, payments)
        """
        b = np.asarray(bids, dtype=np.float64)
        n = len(b)
        n_winners = min(n_winners, n)

        sorted_idx = np.argsort(-b)  # descending
        winners = sorted_idx[:n_winners]
        payments = np.zeros(n_winners)

        for k in range(n_winners):
            # Payment = next highest bid
            if k + 1 < n:
                payments[k] = b[sorted_idx[n_winners]]
            else:
                payments[k] = 0.0

        return winners.astype(np.int64), payments


# ---------------------------------------------------------------------------
# BayesianGame
# ---------------------------------------------------------------------------


class BayesianGame:
    """Bayesian game with uncertain adversary type.

    The adversary's type (capabilities, objective) is drawn from a known
    prior.  The designer must commit to a mechanism without knowing the
    exact adversary type, optimising expected utility.
    """

    def __init__(self, tol: float = 1e-8) -> None:
        self._tol = tol

    def compute_bayesian_equilibrium(
        self,
        payoff_matrices: List[GameMatrix],
        type_prior: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float64], List[npt.NDArray[np.float64]]]:
        """Compute Bayes-Nash equilibrium.

        The designer plays a single mixed strategy; each adversary type
        best-responds independently.

        Args:
            payoff_matrices: Per-type payoff matrices (adversary payoff).
            type_prior: Prior probability of each type.

        Returns:
            (designer_strategy, list_of_type_best_responses)
        """
        prior = np.asarray(type_prior, dtype=np.float64)
        K = len(payoff_matrices)
        if K == 0:
            raise ValueError("Need at least one adversary type")

        As = [np.asarray(gm.payoffs, dtype=np.float64) for gm in payoff_matrices]
        m = As[0].shape[0]
        ns = [A.shape[1] for A in As]

        # Designer minimises expected adversary payoff:
        # min_x max over types of (weighted adversary best response)
        # Formulate as LP: min v s.t. for each type k, for each action j:
        #   prior[k] * A_k[:,j]^T x <= v
        # sum x = 1, x >= 0

        total_constraints = sum(ns)
        c_obj = np.zeros(m + 1)
        c_obj[-1] = 1.0  # minimise v

        ub_rows = []
        ub_rhs = []
        for k in range(K):
            for j in range(ns[k]):
                row = np.zeros(m + 1)
                row[:m] = prior[k] * As[k][:, j]
                row[-1] = -1.0
                ub_rows.append(row)
                ub_rhs.append(0.0)

        A_ub = np.array(ub_rows)
        b_ub = np.array(ub_rhs)
        A_eq = np.zeros((1, m + 1))
        A_eq[0, :m] = 1.0
        b_eq = np.array([1.0])
        bounds = [(0.0, None)] * m + [(None, None)]

        res = sp_opt.linprog(
            c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method="highs",
        )
        if not res.success:
            x = np.ones(m) / m
        else:
            x = np.maximum(res.x[:m], 0.0)
            x /= x.sum()

        # Each type best-responds
        best_responses = []
        for k in range(K):
            payoffs_k = As[k].T @ x  # shape (n_k,)
            br_j = int(np.argmax(payoffs_k))
            br = np.zeros(ns[k])
            br[br_j] = 1.0
            best_responses.append(br)

        return x, best_responses

    def expected_privacy_loss(
        self,
        designer_strategy: npt.NDArray[np.float64],
        payoff_matrices: List[GameMatrix],
        type_prior: npt.NDArray[np.float64],
    ) -> float:
        """Compute expected privacy loss under Bayesian adversary.

        Args:
            designer_strategy: Designer's mixed strategy.
            payoff_matrices: Per-type adversary payoff matrices.
            type_prior: Prior over adversary types.

        Returns:
            Expected worst-case privacy loss.
        """
        prior = np.asarray(type_prior, dtype=np.float64)
        x = np.asarray(designer_strategy, dtype=np.float64)
        total = 0.0

        for k, gm in enumerate(payoff_matrices):
            A = np.asarray(gm.payoffs, dtype=np.float64)
            # Type-k adversary best responds
            payoffs_k = A.T @ x
            best_val = float(np.max(payoffs_k))
            total += prior[k] * best_val

        return total

    def robust_bayesian(
        self,
        payoff_matrices: List[GameMatrix],
        type_prior: npt.NDArray[np.float64],
        prior_uncertainty: float = 0.1,
    ) -> Tuple[npt.NDArray[np.float64], float]:
        """Distributionally robust Bayesian game.

        Optimise against the worst-case prior within an L1-ball
        of radius prior_uncertainty around the nominal prior.

        Args:
            payoff_matrices: Per-type payoff matrices.
            type_prior: Nominal prior.
            prior_uncertainty: L1-ball radius.

        Returns:
            (designer_strategy, robust_value)
        """
        prior = np.asarray(type_prior, dtype=np.float64)
        K = len(payoff_matrices)
        As = [np.asarray(gm.payoffs, dtype=np.float64) for gm in payoff_matrices]
        m = As[0].shape[0]
        ns = [A.shape[1] for A in As]

        # Variables: x (m), v, q (K prior weights)
        # min v s.t.
        #   for each k, j: q[k] * A_k[:,j]^T x <= v
        #   sum x = 1, x >= 0
        #   sum q = 1, q >= 0
        #   ||q - prior||_1 <= prior_uncertainty
        #     => -prior_uncertainty <= sum |q_k - prior_k| <= prior_uncertainty
        #     Linearised: q_k - prior_k <= s_k, prior_k - q_k <= s_k, sum s_k <= prior_uncertainty

        # For simplicity, solve for the worst-case prior first, then optimise
        # Worst-case prior concentrates on the hardest type

        # Compute difficulty of each type
        type_values = np.zeros(K)
        for k in range(K):
            # Minimax value for type k
            from dp_forge.game_theory.minimax import MinimaxSolver
            solver = MinimaxSolver()
            result = solver.solve(GameMatrix(payoffs=As[k]))
            type_values[k] = result.equilibrium.game_value

        # Worst-case prior shifts mass toward hardest type
        worst_prior = prior.copy()
        hardest = int(np.argmax(type_values))
        budget = min(prior_uncertainty / 2, 1.0 - prior[hardest])
        worst_prior[hardest] += budget
        # Reduce others proportionally
        others = [k for k in range(K) if k != hardest]
        if others:
            reduce = budget / len(others)
            for k in others:
                worst_prior[k] = max(0.0, worst_prior[k] - reduce)
        worst_prior = np.maximum(worst_prior, 0.0)
        worst_prior /= worst_prior.sum()

        # Solve with worst-case prior
        x, _ = self.compute_bayesian_equilibrium(payoff_matrices, worst_prior)
        robust_val = self.expected_privacy_loss(x, payoff_matrices, worst_prior)

        return x, robust_val
