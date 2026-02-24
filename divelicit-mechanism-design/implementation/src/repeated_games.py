"""
Repeated games: strategies, equilibria, evolutionary dynamics, and tournaments.

Implements core repeated game theory including folk theorem verification,
subgame perfect equilibrium via backward induction, automaton strategies,
replicator dynamics on strategy spaces, Axelrod-style tournaments, and
stochastic games with state-dependent transitions.
"""

import numpy as np
from scipy.optimize import linprog
from typing import List, Tuple, Optional, Dict, Callable, Any


class History:
    """Record of actions and payoffs across rounds of a repeated game.

    Attributes:
        actions: List of (action_p1, action_p2) tuples per round.
        payoffs: List of (payoff_p1, payoff_p2) tuples per round.
    """

    def __init__(self):
        """Initialize an empty history."""
        self.actions: List[Tuple[int, int]] = []
        self.payoffs: List[Tuple[float, float]] = []

    def add_round(self, action_pair: Tuple[int, int], payoff_pair: Tuple[float, float]):
        """Append one round of play.

        Args:
            action_pair: Actions chosen by player 1 and player 2.
            payoff_pair: Resulting payoffs for player 1 and player 2.
        """
        self.actions.append(action_pair)
        self.payoffs.append(payoff_pair)

    def get_actions(self, player: int) -> List[int]:
        """Return the sequence of actions for the given player.

        Args:
            player: 0 for player 1, 1 for player 2.

        Returns:
            List of actions taken by the specified player.
        """
        return [a[player] for a in self.actions]

    def average_payoffs(self) -> Tuple[float, float]:
        """Compute time-averaged payoffs for both players.

        Returns:
            Tuple of average payoff for player 1 and player 2.
        """
        if not self.payoffs:
            return (0.0, 0.0)
        p = np.array(self.payoffs)
        return (float(np.mean(p[:, 0])), float(np.mean(p[:, 1])))

    def discounted_payoffs(self, delta: float) -> Tuple[float, float]:
        """Compute discounted sum of payoffs.

        Args:
            delta: Discount factor in [0, 1).

        Returns:
            Tuple of discounted payoff for player 1 and player 2.
        """
        if not self.payoffs:
            return (0.0, 0.0)
        p = np.array(self.payoffs)
        T = len(p)
        weights = np.array([delta ** t for t in range(T)])
        norm = (1.0 - delta) if delta < 1.0 else 1.0 / T
        s1 = float(norm * np.dot(weights, p[:, 0]))
        s2 = float(norm * np.dot(weights, p[:, 1]))
        return (s1, s2)

    def __len__(self) -> int:
        return len(self.actions)


# ---------------------------------------------------------------------------
# Strategy functions
# ---------------------------------------------------------------------------

def tit_for_tat(history: History, player: int) -> int:
    """Tit-for-tat: cooperate on first move, then copy opponent's last action.

    Args:
        history: The game history so far.
        player: 0 or 1 indicating which player we are.

    Returns:
        Action to play (0 = cooperate, 1 = defect).
    """
    if len(history) == 0:
        return 0
    opponent = 1 - player
    return history.actions[-1][opponent]


def grim_trigger(history: History, player: int) -> int:
    """Grim trigger: cooperate until opponent defects, then defect forever.

    Args:
        history: The game history so far.
        player: 0 or 1 indicating which player we are.

    Returns:
        Action to play (0 = cooperate, 1 = defect).
    """
    opponent = 1 - player
    for a in history.actions:
        if a[opponent] == 1:
            return 1
    return 0


def pavlov(history: History, player: int) -> int:
    """Pavlov (win-stay, lose-shift): cooperate if last round was mutual
    cooperation or mutual defection; otherwise switch.

    Args:
        history: The game history so far.
        player: 0 or 1 indicating which player we are.

    Returns:
        Action to play (0 = cooperate, 1 = defect).
    """
    if len(history) == 0:
        return 0
    last = history.actions[-1]
    if last[0] == last[1]:
        return 0
    return 1


def win_stay_lose_shift(history: History, player: int) -> int:
    """Win-stay lose-shift: repeat last action if payoff was above the
    mutual-defection baseline, otherwise switch.

    Uses the payoff from last round: if it was >= the payoff from mutual
    cooperation equivalent (top-half of payoff range), stay; else shift.

    Args:
        history: The game history so far.
        player: 0 or 1 indicating which player we are.

    Returns:
        Action to play (0 = cooperate, 1 = defect).
    """
    if len(history) == 0:
        return 0
    last_payoff = history.payoffs[-1][player]
    all_payoffs = [p[player] for p in history.payoffs]
    median_payoff = np.median(all_payoffs) if len(all_payoffs) > 1 else last_payoff
    last_action = history.actions[-1][player]
    if last_payoff >= median_payoff:
        return last_action
    return 1 - last_action


def always_cooperate(history: History, player: int) -> int:
    """Always cooperate regardless of history.

    Args:
        history: The game history so far.
        player: 0 or 1 indicating which player we are.

    Returns:
        0 (cooperate).
    """
    return 0


def always_defect(history: History, player: int) -> int:
    """Always defect regardless of history.

    Args:
        history: The game history so far.
        player: 0 or 1 indicating which player we are.

    Returns:
        1 (defect).
    """
    return 1


def random_strategy(history: History, player: int) -> int:
    """Play uniformly at random.

    Args:
        history: The game history so far.
        player: 0 or 1 indicating which player we are.

    Returns:
        0 or 1 chosen uniformly at random.
    """
    return int(np.random.randint(2))


def suspicious_tit_for_tat(history: History, player: int) -> int:
    """Suspicious tit-for-tat: defect first, then copy opponent's last move.

    Args:
        history: The game history so far.
        player: 0 or 1 indicating which player we are.

    Returns:
        Action to play.
    """
    if len(history) == 0:
        return 1
    opponent = 1 - player
    return history.actions[-1][opponent]


# ---------------------------------------------------------------------------
# RepeatedGame class
# ---------------------------------------------------------------------------

class RepeatedGame:
    """A two-player repeated game built on a stage game.

    Attributes:
        payoff_matrix_p1: Payoff matrix for player 1 (rows x cols).
        payoff_matrix_p2: Payoff matrix for player 2 (rows x cols).
        discount_factor: Discount factor delta in [0, 1].
        n_actions_p1: Number of actions for player 1.
        n_actions_p2: Number of actions for player 2.
    """

    def __init__(self, payoff_matrix_p1: np.ndarray, payoff_matrix_p2: np.ndarray,
                 discount_factor: float = 0.95):
        """Initialize a repeated game.

        Args:
            payoff_matrix_p1: (m x n) payoff matrix for player 1.
            payoff_matrix_p2: (m x n) payoff matrix for player 2.
            discount_factor: Discount factor for future payoffs.
        """
        self.payoff_matrix_p1 = np.array(payoff_matrix_p1, dtype=float)
        self.payoff_matrix_p2 = np.array(payoff_matrix_p2, dtype=float)
        assert self.payoff_matrix_p1.shape == self.payoff_matrix_p2.shape
        self.n_actions_p1, self.n_actions_p2 = self.payoff_matrix_p1.shape
        self.discount_factor = discount_factor

    @classmethod
    def stage_game(cls, payoff_matrix: np.ndarray,
                   discount_factor: float = 0.95) -> "RepeatedGame":
        """Create a repeated game from a single bimatrix payoff array.

        Args:
            payoff_matrix: Shape (m, n, 2) where last axis is (p1, p2) payoffs.
            discount_factor: Discount factor for the repeated game.

        Returns:
            A RepeatedGame instance.
        """
        pm = np.array(payoff_matrix, dtype=float)
        return cls(pm[:, :, 0], pm[:, :, 1], discount_factor)

    def stage_payoffs(self, a1: int, a2: int) -> Tuple[float, float]:
        """Return stage-game payoffs for an action profile.

        Args:
            a1: Action of player 1.
            a2: Action of player 2.

        Returns:
            (payoff_p1, payoff_p2).
        """
        return (float(self.payoff_matrix_p1[a1, a2]),
                float(self.payoff_matrix_p2[a1, a2]))

    def play(self, strategy1: Callable, strategy2: Callable,
             T: int = 100) -> History:
        """Simulate T rounds of the repeated game with two strategy functions.

        Args:
            strategy1: Callable(history, player=0) -> action.
            strategy2: Callable(history, player=1) -> action.
            T: Number of rounds to play.

        Returns:
            History object with recorded actions and payoffs.
        """
        history = History()
        for _ in range(T):
            a1 = strategy1(history, 0)
            a2 = strategy2(history, 1)
            a1 = int(np.clip(a1, 0, self.n_actions_p1 - 1))
            a2 = int(np.clip(a2, 0, self.n_actions_p2 - 1))
            p = self.stage_payoffs(a1, a2)
            history.add_round((a1, a2), p)
        return history

    def minimax_values(self) -> Tuple[float, float]:
        """Compute the minimax value for each player.

        Player 1's minimax value: min over p2's mixed strategy of
        max over p1's pure action of expected payoff to p1.
        Computed via linear programming.

        Returns:
            (minimax_p1, minimax_p2).
        """
        mm1 = self._minimax_value_player(self.payoff_matrix_p1)
        mm2 = self._minimax_value_player(self.payoff_matrix_p2.T)
        return (mm1, mm2)

    def _minimax_value_player(self, payoff_matrix: np.ndarray) -> float:
        """Compute minimax value for a player given their payoff matrix.

        The opponent (column player) minimises max row payoff by choosing a
        mixed strategy. Formulated as an LP.

        Args:
            payoff_matrix: (m x n) payoff matrix for the player being minimaxed.

        Returns:
            The minimax value.
        """
        m, n = payoff_matrix.shape
        # Opponent picks mixed strategy q over n actions to minimise v
        # such that for every row i: sum_j A[i,j]*q[j] >= v  (player can
        # guarantee at least v).  We minimise v.
        # Variables: [q_1 ... q_n, v]
        c = np.zeros(n + 1)
        c[-1] = -1.0  # maximise v => minimise -v

        A_ub = np.zeros((m, n + 1))
        b_ub = np.zeros(m)
        for i in range(m):
            A_ub[i, :n] = -payoff_matrix[i, :]
            A_ub[i, n] = 1.0

        A_eq = np.zeros((1, n + 1))
        A_eq[0, :n] = 1.0
        b_eq = np.array([1.0])

        bounds = [(0, 1) for _ in range(n)] + [(None, None)]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method='highs')
        if res.success:
            return float(res.x[-1])
        return float(np.min(np.max(payoff_matrix, axis=0)))

    def feasible_payoff_set(self, n_samples: int = 5000) -> np.ndarray:
        """Sample the feasible (convex hull) payoff set via random mixed
        action profiles.

        Args:
            n_samples: Number of random correlated strategy profiles to sample.

        Returns:
            (n_samples, 2) array of feasible payoff vectors.
        """
        m, n = self.n_actions_p1, self.n_actions_p2
        payoffs = np.zeros((n_samples, 2))
        for k in range(n_samples):
            prob = np.random.dirichlet(np.ones(m * n))
            prob_matrix = prob.reshape(m, n)
            payoffs[k, 0] = np.sum(prob_matrix * self.payoff_matrix_p1)
            payoffs[k, 1] = np.sum(prob_matrix * self.payoff_matrix_p2)
        return payoffs

    def folk_theorem_check(self, target_payoff: Tuple[float, float],
                           tol: float = 1e-8) -> Dict[str, Any]:
        """Check whether a payoff vector is sustainable as a Nash equilibrium
        of the infinitely repeated game via the folk theorem.

        A payoff vector v = (v1, v2) is sustainable if:
          1) v is feasible (in the convex hull of stage-game payoffs).
          2) v_i >= minimax_i for each player i (individually rational).
          3) The discount factor is sufficiently high.

        For grim-trigger sustainability the critical discount factor for each
        player is delta* = (d_i - v_i) / (d_i - p_i) where d_i is the
        one-shot deviation payoff and p_i is the punishment payoff.

        Args:
            target_payoff: The payoff vector (v1, v2) to check.
            tol: Numerical tolerance.

        Returns:
            Dict with keys 'feasible', 'individually_rational',
            'sustainable', 'critical_delta', and 'details'.
        """
        v1, v2 = target_payoff
        mm1, mm2 = self.minimax_values()

        ir = (v1 >= mm1 - tol) and (v2 >= mm2 - tol)

        feasible = self._is_feasible(target_payoff, tol)

        # Compute critical discount factor via one-shot deviation principle
        # For each player: best deviation payoff given the cooperative profile
        # that yields target_payoff
        best_dev1 = float(np.max(self.payoff_matrix_p1))
        best_dev2 = float(np.max(self.payoff_matrix_p2))

        crit_deltas = []
        for v_i, d_i, mm_i in [(v1, best_dev1, mm1), (v2, best_dev2, mm2)]:
            if d_i - mm_i > tol:
                crit = (d_i - v_i) / (d_i - mm_i)
                crit_deltas.append(max(0.0, min(crit, 1.0)))
            else:
                crit_deltas.append(0.0)

        critical_delta = max(crit_deltas)
        sustainable = feasible and ir and (self.discount_factor >= critical_delta - tol)

        return {
            'feasible': feasible,
            'individually_rational': ir,
            'sustainable': sustainable,
            'critical_delta': critical_delta,
            'minimax_values': (mm1, mm2),
            'details': {
                'target': target_payoff,
                'discount_factor': self.discount_factor,
                'crit_delta_per_player': crit_deltas,
            }
        }

    def _is_feasible(self, payoff: Tuple[float, float], tol: float) -> bool:
        """Check if a payoff vector lies in the convex hull of stage-game
        payoff profiles.

        Uses LP: find weights lambda_{ij} >= 0 summing to 1 such that
        sum lambda_{ij} * u_k(i,j) = v_k for k=1,2.

        Args:
            payoff: Target payoff (v1, v2).
            tol: Tolerance for feasibility.

        Returns:
            True if feasible.
        """
        m, n = self.n_actions_p1, self.n_actions_p2
        mn = m * n
        v1, v2 = payoff

        u1_flat = self.payoff_matrix_p1.flatten()
        u2_flat = self.payoff_matrix_p2.flatten()

        A_eq = np.vstack([u1_flat, u2_flat, np.ones(mn)])
        b_eq = np.array([v1, v2, 1.0])
        c = np.zeros(mn)
        bounds = [(0, 1) for _ in range(mn)]

        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        return res.success

    def nash_equilibria_stage(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Find Nash equilibria of the stage game via support enumeration.

        Enumerates over all possible support pairs and solves the
        indifference conditions.

        Returns:
            List of (mixed_strategy_p1, mixed_strategy_p2) tuples.
        """
        m, n = self.n_actions_p1, self.n_actions_p2
        equilibria = []

        for s1_mask in range(1, 2 ** m):
            for s2_mask in range(1, 2 ** n):
                s1 = [i for i in range(m) if s1_mask & (1 << i)]
                s2 = [j for j in range(n) if s2_mask & (1 << j)]
                result = self._solve_support(s1, s2)
                if result is not None:
                    equilibria.append(result)

        return equilibria

    def _solve_support(self, support1: List[int],
                       support2: List[int]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Attempt to solve indifference conditions for a given support pair.

        Args:
            support1: Support of player 1's mixed strategy.
            support2: Support of player 2's mixed strategy.

        Returns:
            (p, q) mixed strategy pair or None if no valid solution.
        """
        k1, k2 = len(support1), len(support2)
        m, n = self.n_actions_p1, self.n_actions_p2

        # Player 2's strategy q must make player 1 indifferent on support1
        # A1[s1, :] @ q = v1 for all s1 in support1
        A2_sub = self.payoff_matrix_p1[np.ix_(support1, support2)]
        A1_sub = self.payoff_matrix_p2[np.ix_(support1, support2)]

        # Solve for q: indifference of p1 over support1
        # A2_sub @ q = v1 * ones, sum(q) = 1, q >= 0
        if k2 == 1:
            q_sub = np.array([1.0])
        else:
            lhs = np.zeros((k1 + 1, k2))
            rhs = np.zeros(k1 + 1)
            for idx in range(k1 - 1):
                lhs[idx, :] = A2_sub[idx, :] - A2_sub[idx + 1, :]
            lhs[k1 - 1, :] = 1.0  # sum = 1 (use second-to-last row)
            rhs[k1 - 1] = 1.0
            lhs[k1, :] = 1.0
            rhs[k1] = 1.0
            try:
                q_sub, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
            except np.linalg.LinAlgError:
                return None
            if np.any(q_sub < -1e-10) or abs(np.sum(q_sub) - 1.0) > 1e-6:
                return None
            q_sub = np.maximum(q_sub, 0)
            q_sub /= np.sum(q_sub)

        # Solve for p: indifference of p2 over support2
        B_sub = self.payoff_matrix_p2[np.ix_(support1, support2)]
        if k1 == 1:
            p_sub = np.array([1.0])
        else:
            lhs2 = np.zeros((k2 + 1, k1))
            rhs2 = np.zeros(k2 + 1)
            for idx in range(k2 - 1):
                lhs2[idx, :] = B_sub[:, idx] - B_sub[:, idx + 1]
            lhs2[k2 - 1, :] = 1.0
            rhs2[k2 - 1] = 1.0
            lhs2[k2, :] = 1.0
            rhs2[k2] = 1.0
            try:
                p_sub, _, _, _ = np.linalg.lstsq(lhs2, rhs2, rcond=None)
            except np.linalg.LinAlgError:
                return None
            if np.any(p_sub < -1e-10) or abs(np.sum(p_sub) - 1.0) > 1e-6:
                return None
            p_sub = np.maximum(p_sub, 0)
            p_sub /= np.sum(p_sub)

        # Verify best response condition
        p_full = np.zeros(m)
        p_full[support1] = p_sub
        q_full = np.zeros(n)
        q_full[support2] = q_sub

        payoff1 = self.payoff_matrix_p1 @ q_full
        payoff2 = self.payoff_matrix_p2.T @ p_full

        v1 = p_full @ payoff1
        v2 = p_full @ self.payoff_matrix_p2 @ q_full

        if np.max(payoff1) > v1 + 1e-6:
            return None
        if np.max(payoff2) > v2 + 1e-6:
            return None

        return (p_full, q_full)


# ---------------------------------------------------------------------------
# Subgame Perfect Equilibrium (Backward Induction)
# ---------------------------------------------------------------------------

class FiniteRepeatedGame:
    """Finite-horizon repeated game with backward induction solver.

    Attributes:
        payoff_matrix_p1: Stage-game payoff matrix for player 1.
        payoff_matrix_p2: Stage-game payoff matrix for player 2.
        T: Number of rounds.
    """

    def __init__(self, payoff_matrix_p1: np.ndarray, payoff_matrix_p2: np.ndarray,
                 T: int):
        """Initialize a finite repeated game.

        Args:
            payoff_matrix_p1: (m x n) payoff matrix for player 1.
            payoff_matrix_p2: (m x n) payoff matrix for player 2.
            T: Number of rounds to play.
        """
        self.payoff_matrix_p1 = np.array(payoff_matrix_p1, dtype=float)
        self.payoff_matrix_p2 = np.array(payoff_matrix_p2, dtype=float)
        self.T = T
        self.m, self.n = self.payoff_matrix_p1.shape

    def stage_nash(self) -> Tuple[int, int]:
        """Find a pure-strategy Nash equilibrium of the stage game.

        Uses iterated best-response starting from action (0, 0).

        Returns:
            (action_p1, action_p2) forming a Nash equilibrium.
        """
        a1, a2 = 0, 0
        for _ in range(self.m + self.n):
            new_a1 = int(np.argmax(self.payoff_matrix_p1[:, a2]))
            new_a2 = int(np.argmax(self.payoff_matrix_p2[a1, :]))
            if new_a1 == a1 and new_a2 == a2:
                break
            a1, a2 = new_a1, new_a2
        return (a1, a2)

    def backward_induction(self) -> List[Tuple[int, int]]:
        """Compute the subgame perfect equilibrium action profile via
        backward induction.

        For a finitely repeated game with a unique stage-game Nash
        equilibrium, the SPE plays the stage NE in every period.
        When there are multiple stage NE, cooperation may be sustained
        in early rounds; we compute this by backward induction over
        continuation values.

        Returns:
            List of (action_p1, action_p2) for each round t=0..T-1.
        """
        ne = self.stage_nash()
        ne_payoff1 = self.payoff_matrix_p1[ne[0], ne[1]]
        ne_payoff2 = self.payoff_matrix_p2[ne[0], ne[1]]

        # With a unique stage NE, backward induction yields NE in each period
        # We still compute it properly: continuation values from the end
        actions_seq = []
        continuation1 = 0.0
        continuation2 = 0.0

        round_actions = [ne] * self.T

        # Work backwards: at each stage, the continuation value is fixed
        for t in range(self.T - 1, -1, -1):
            best_val = -np.inf
            best_pair = ne
            for a1 in range(self.m):
                for a2 in range(self.n):
                    # Check if (a1, a2) can be sustained: each player must
                    # prefer (a1, a2) + continuation over deviating + NE cont.
                    v1 = self.payoff_matrix_p1[a1, a2] + continuation1
                    v2 = self.payoff_matrix_p2[a1, a2] + continuation2

                    dev1 = float(np.max(self.payoff_matrix_p1[:, a2]))
                    dev2 = float(np.max(self.payoff_matrix_p2[a1, :]))

                    dev1_total = dev1 + (self.T - t - 1) * ne_payoff1
                    dev2_total = dev2 + (self.T - t - 1) * ne_payoff2

                    if v1 >= dev1_total - 1e-10 and v2 >= dev2_total - 1e-10:
                        social = v1 + v2
                        if social > best_val:
                            best_val = social
                            best_pair = (a1, a2)

            round_actions[t] = best_pair
            continuation1 += self.payoff_matrix_p1[best_pair[0], best_pair[1]]
            continuation2 += self.payoff_matrix_p2[best_pair[0], best_pair[1]]

        return round_actions

    def simulate(self) -> History:
        """Simulate the subgame perfect equilibrium path.

        Returns:
            History with the SPE action and payoff sequence.
        """
        actions_seq = self.backward_induction()
        history = History()
        for a1, a2 in actions_seq:
            p1 = float(self.payoff_matrix_p1[a1, a2])
            p2 = float(self.payoff_matrix_p2[a1, a2])
            history.add_round((a1, a2), (p1, p2))
        return history


# ---------------------------------------------------------------------------
# Automaton Strategies
# ---------------------------------------------------------------------------

class AutomatonStrategy:
    """Finite-state automaton strategy for repeated games.

    Each state maps to an action. Transitions depend on the opponent's action.

    Attributes:
        n_states: Number of states in the automaton.
        actions: Array of length n_states mapping state -> action.
        transitions: (n_states, max_opponent_actions) -> next state.
        initial_state: Starting state index.
        current_state: Current state during play.
    """

    def __init__(self, n_states: int, actions: np.ndarray,
                 transitions: np.ndarray, initial_state: int = 0):
        """Initialize the automaton.

        Args:
            n_states: Number of states.
            actions: 1D array mapping state index to action.
            transitions: 2D array (n_states x n_opponent_actions) -> next state.
            initial_state: Index of the starting state.
        """
        self.n_states = n_states
        self.actions = np.array(actions, dtype=int)
        self.transitions = np.array(transitions, dtype=int)
        self.initial_state = initial_state
        self.current_state = initial_state

    def reset(self):
        """Reset the automaton to its initial state."""
        self.current_state = self.initial_state

    def get_action(self) -> int:
        """Return the action for the current state.

        Returns:
            The action to play.
        """
        return int(self.actions[self.current_state])

    def update(self, opponent_action: int):
        """Transition to the next state based on opponent's action.

        Args:
            opponent_action: The action taken by the opponent.
        """
        opp = min(opponent_action, self.transitions.shape[1] - 1)
        self.current_state = int(self.transitions[self.current_state, opp])

    def __call__(self, history: History, player: int) -> int:
        """Play the automaton strategy within a repeated game.

        Args:
            history: Game history.
            player: Which player (0 or 1).

        Returns:
            Action to play.
        """
        if len(history) == 0:
            self.reset()
            return self.get_action()
        opponent = 1 - player
        opp_last = history.actions[-1][opponent]
        self.update(opp_last)
        return self.get_action()

    @classmethod
    def tit_for_tat_automaton(cls) -> "AutomatonStrategy":
        """Create a tit-for-tat automaton.

        State 0: cooperate. State 1: defect.
        Transitions: go to state matching opponent's last action.

        Returns:
            AutomatonStrategy implementing tit-for-tat.
        """
        return cls(
            n_states=2,
            actions=np.array([0, 1]),
            transitions=np.array([[0, 1], [0, 1]]),
            initial_state=0
        )

    @classmethod
    def grim_trigger_automaton(cls) -> "AutomatonStrategy":
        """Create a grim trigger automaton.

        State 0: cooperate (transition to 1 if opponent defects).
        State 1: defect forever (absorbing).

        Returns:
            AutomatonStrategy implementing grim trigger.
        """
        return cls(
            n_states=2,
            actions=np.array([0, 1]),
            transitions=np.array([[0, 1], [1, 1]]),
            initial_state=0
        )

    @classmethod
    def pavlov_automaton(cls) -> "AutomatonStrategy":
        """Create a Pavlov automaton.

        State 0: cooperate. State 1: defect.
        From cooperate: stay if opp cooperates, switch if opp defects.
        From defect: switch if opp cooperates, stay if opp defects.

        Returns:
            AutomatonStrategy implementing Pavlov.
        """
        return cls(
            n_states=2,
            actions=np.array([0, 1]),
            transitions=np.array([[0, 1], [0, 1]]),
            initial_state=0
        )

    def stationary_distribution(self, opponent_automaton: "AutomatonStrategy") -> np.ndarray:
        """Compute the stationary distribution over joint states when
        playing against another automaton.

        Constructs the joint Markov chain and finds its stationary distribution.

        Args:
            opponent_automaton: The opponent's automaton strategy.

        Returns:
            Stationary distribution over (my_state, opp_state) pairs,
            flattened to shape (n_states * opp.n_states,).
        """
        n1 = self.n_states
        n2 = opponent_automaton.n_states
        joint_n = n1 * n2

        T_matrix = np.zeros((joint_n, joint_n))

        for s1 in range(n1):
            for s2 in range(n2):
                a1 = self.actions[s1]
                a2 = opponent_automaton.actions[s2]
                ns1 = self.transitions[s1, min(a2, self.transitions.shape[1] - 1)]
                ns2 = opponent_automaton.transitions[
                    s2, min(a1, opponent_automaton.transitions.shape[1] - 1)]
                src = s1 * n2 + s2
                dst = ns1 * n2 + ns2
                T_matrix[src, dst] = 1.0

        # Find stationary distribution: pi @ T = pi, sum(pi) = 1
        # Equivalent to finding left eigenvector with eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(T_matrix.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.real(eigenvectors[:, idx])
        pi = np.abs(pi)
        if np.sum(pi) > 0:
            pi /= np.sum(pi)
        return pi


# ---------------------------------------------------------------------------
# Evolutionary Dynamics
# ---------------------------------------------------------------------------

class ReplicatorDynamics:
    """Replicator dynamics over a population of strategies in a repeated game.

    Strategies compete in pairwise interactions; population shares evolve
    according to relative fitness.

    Attributes:
        game: The underlying RepeatedGame.
        strategies: List of strategy callables.
        strategy_names: Names for each strategy.
        T_per_match: Rounds per pairwise match.
    """

    def __init__(self, game: RepeatedGame,
                 strategies: List[Callable],
                 strategy_names: Optional[List[str]] = None,
                 T_per_match: int = 50):
        """Initialize replicator dynamics.

        Args:
            game: The repeated game for pairwise matches.
            strategies: List of strategy functions.
            strategy_names: Optional names for strategies.
            T_per_match: Number of rounds per pairwise interaction.
        """
        self.game = game
        self.strategies = strategies
        self.n_strategies = len(strategies)
        self.strategy_names = strategy_names or [f"S{i}" for i in range(self.n_strategies)]
        self.T_per_match = T_per_match
        self._payoff_matrix = None

    def compute_payoff_matrix(self) -> np.ndarray:
        """Compute the pairwise payoff matrix by simulating all matchups.

        Entry (i, j) is the average payoff to strategy i when playing
        against strategy j.

        Returns:
            (n_strategies x n_strategies) payoff matrix.
        """
        n = self.n_strategies
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                history = self.game.play(self.strategies[i], self.strategies[j],
                                         self.T_per_match)
                avg = history.average_payoffs()
                M[i, j] = avg[0]
        self._payoff_matrix = M
        return M

    def fitness(self, population: np.ndarray) -> np.ndarray:
        """Compute fitness of each strategy given population shares.

        Fitness of strategy i = sum_j population[j] * M[i, j].

        Args:
            population: Array of population shares summing to 1.

        Returns:
            Array of fitness values for each strategy.
        """
        if self._payoff_matrix is None:
            self.compute_payoff_matrix()
        return self._payoff_matrix @ population

    def replicator_step(self, population: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """One step of discrete-time replicator dynamics.

        dx_i/dt = x_i * (f_i - f_bar)

        Args:
            population: Current population shares.
            dt: Time step size.

        Returns:
            Updated population shares.
        """
        f = self.fitness(population)
        f_bar = float(np.dot(population, f))
        dp = population * (f - f_bar) * dt
        new_pop = population + dp
        new_pop = np.maximum(new_pop, 0.0)
        s = np.sum(new_pop)
        if s > 0:
            new_pop /= s
        return new_pop

    def simulate(self, initial_population: np.ndarray,
                 n_steps: int = 1000, dt: float = 0.01) -> np.ndarray:
        """Simulate replicator dynamics over many steps.

        Args:
            initial_population: Starting population shares.
            n_steps: Number of time steps.
            dt: Time step size.

        Returns:
            (n_steps + 1, n_strategies) trajectory of population shares.
        """
        trajectory = np.zeros((n_steps + 1, self.n_strategies))
        trajectory[0] = initial_population.copy()
        for t in range(n_steps):
            trajectory[t + 1] = self.replicator_step(trajectory[t], dt)
        return trajectory

    def find_equilibrium(self, initial_population: Optional[np.ndarray] = None,
                         n_steps: int = 5000, dt: float = 0.01,
                         tol: float = 1e-8) -> np.ndarray:
        """Run replicator dynamics until convergence or max steps.

        Args:
            initial_population: Starting shares (uniform if None).
            n_steps: Maximum iterations.
            dt: Time step.
            tol: Convergence tolerance on population change.

        Returns:
            Equilibrium population shares.
        """
        if initial_population is None:
            pop = np.ones(self.n_strategies) / self.n_strategies
        else:
            pop = initial_population.copy()

        for _ in range(n_steps):
            new_pop = self.replicator_step(pop, dt)
            if np.max(np.abs(new_pop - pop)) < tol:
                return new_pop
            pop = new_pop
        return pop


# ---------------------------------------------------------------------------
# Tournament
# ---------------------------------------------------------------------------

class Tournament:
    """Axelrod-style round-robin tournament of repeated-game strategies.

    Attributes:
        game: The repeated game.
        strategies: List of strategy callables.
        strategy_names: Names for display.
        T_per_match: Rounds per match.
        n_repetitions: Number of repeated matchups to average over.
    """

    def __init__(self, game: RepeatedGame,
                 strategies: List[Callable],
                 strategy_names: Optional[List[str]] = None,
                 T_per_match: int = 200,
                 n_repetitions: int = 1):
        """Initialize the tournament.

        Args:
            game: Repeated game instance.
            strategies: Strategy functions.
            strategy_names: Optional names.
            T_per_match: Rounds per match.
            n_repetitions: How many times each pair plays.
        """
        self.game = game
        self.strategies = strategies
        self.n = len(strategies)
        self.strategy_names = strategy_names or [f"S{i}" for i in range(self.n)]
        self.T_per_match = T_per_match
        self.n_repetitions = n_repetitions

    def run(self) -> Dict[str, Any]:
        """Execute the round-robin tournament.

        Every pair of strategies plays T_per_match rounds, repeated
        n_repetitions times. Scores are accumulated.

        Returns:
            Dict with 'scores' (total per strategy), 'matrix' (pairwise scores),
            'ranking' (sorted strategy indices), and 'details'.
        """
        scores = np.zeros(self.n)
        matrix = np.zeros((self.n, self.n))

        for rep in range(self.n_repetitions):
            for i in range(self.n):
                for j in range(self.n):
                    history = self.game.play(
                        self.strategies[i], self.strategies[j], self.T_per_match)
                    avg = history.average_payoffs()
                    matrix[i, j] += avg[0] / self.n_repetitions
                    scores[i] += avg[0] * self.T_per_match / self.n_repetitions

        ranking = list(np.argsort(-scores))
        results = {
            'scores': scores,
            'matrix': matrix,
            'ranking': ranking,
            'ranked_names': [self.strategy_names[i] for i in ranking],
            'details': {
                'T_per_match': self.T_per_match,
                'n_repetitions': self.n_repetitions,
                'n_strategies': self.n,
            }
        }
        return results

    def ecological_simulation(self, n_generations: int = 100) -> np.ndarray:
        """Simulate ecological dynamics: population shares evolve based on
        tournament fitness.

        After running the round-robin, the payoff matrix drives proportional
        reproduction where successful strategies grow.

        Args:
            n_generations: Number of ecological generations.

        Returns:
            (n_generations + 1, n_strategies) population trajectory.
        """
        result = self.run()
        M = result['matrix']

        pop = np.ones(self.n) / self.n
        trajectory = np.zeros((n_generations + 1, self.n))
        trajectory[0] = pop.copy()

        for gen in range(n_generations):
            fitness = M @ pop
            avg_fitness = np.dot(pop, fitness)
            if avg_fitness > 0:
                pop = pop * fitness / avg_fitness
            pop = np.maximum(pop, 0)
            s = np.sum(pop)
            if s > 0:
                pop /= s
            trajectory[gen + 1] = pop.copy()

        return trajectory


# ---------------------------------------------------------------------------
# Stochastic Games
# ---------------------------------------------------------------------------

class StochasticGame:
    """Stochastic game: state-dependent stage games with Markov transitions.

    In each state s, players play a stage game with payoff matrices
    A_s, B_s and then transition to a new state according to a
    probability distribution that depends on the state and actions.

    Attributes:
        n_states: Number of states.
        payoff_matrices_p1: List of payoff matrices for player 1, one per state.
        payoff_matrices_p2: List of payoff matrices for player 2, one per state.
        transition_probs: transition_probs[s][a1][a2] is a probability
            distribution over next states.
        discount_factor: Discount factor.
    """

    def __init__(self, payoff_matrices_p1: List[np.ndarray],
                 payoff_matrices_p2: List[np.ndarray],
                 transition_probs: List[List[List[np.ndarray]]],
                 discount_factor: float = 0.95):
        """Initialize the stochastic game.

        Args:
            payoff_matrices_p1: List of (m_s x n_s) payoff matrices for p1.
            payoff_matrices_p2: List of (m_s x n_s) payoff matrices for p2.
            transition_probs: Nested list [state][a1][a2] -> array of length
                n_states giving transition probabilities.
            discount_factor: Discount factor.
        """
        self.n_states = len(payoff_matrices_p1)
        self.payoff_matrices_p1 = [np.array(m, dtype=float) for m in payoff_matrices_p1]
        self.payoff_matrices_p2 = [np.array(m, dtype=float) for m in payoff_matrices_p2]
        self.transition_probs = transition_probs
        self.discount_factor = discount_factor

    def simulate(self, strategy1: Callable, strategy2: Callable,
                 initial_state: int = 0, T: int = 100) -> Tuple[History, List[int]]:
        """Simulate the stochastic game.

        Strategy functions receive (history, player, current_state) and
        return an action.

        Args:
            strategy1: Strategy for player 1 taking (history, player, state).
            strategy2: Strategy for player 2 taking (history, player, state).
            initial_state: Starting state.
            T: Number of rounds.

        Returns:
            (history, state_sequence) tuple.
        """
        history = History()
        state = initial_state
        states = [state]

        for _ in range(T):
            m_s = self.payoff_matrices_p1[state].shape[0]
            n_s = self.payoff_matrices_p1[state].shape[1]
            a1 = int(np.clip(strategy1(history, 0, state), 0, m_s - 1))
            a2 = int(np.clip(strategy2(history, 1, state), 0, n_s - 1))

            p1 = float(self.payoff_matrices_p1[state][a1, a2])
            p2 = float(self.payoff_matrices_p2[state][a1, a2])
            history.add_round((a1, a2), (p1, p2))

            trans = self.transition_probs[state][a1][a2]
            trans = np.array(trans, dtype=float)
            trans = np.maximum(trans, 0)
            s = np.sum(trans)
            if s > 0:
                trans /= s
            state = int(np.random.choice(self.n_states, p=trans))
            states.append(state)

        return history, states

    def value_iteration(self, tol: float = 1e-8,
                        max_iter: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Compute approximate equilibrium values via value iteration
        for a zero-sum stochastic game.

        Uses the Shapley (1953) operator for zero-sum stochastic games:
        V(s) = val(A_s + delta * sum_{s'} T(s'|s,.) V(s'))

        where val is the value of the matrix game.

        Args:
            tol: Convergence tolerance.
            max_iter: Maximum number of iterations.

        Returns:
            (values, strategies_p1) where values is the value in each state
            and strategies_p1 is the optimal mixed strategy for p1 in each state.
        """
        V = np.zeros(self.n_states)
        strategies = [None] * self.n_states

        for iteration in range(max_iter):
            V_new = np.zeros(self.n_states)
            for s in range(self.n_states):
                m, n = self.payoff_matrices_p1[s].shape
                # Build effective matrix: A_s + delta * E[V(s')]
                M = self.payoff_matrices_p1[s].copy()
                for a1 in range(m):
                    for a2 in range(n):
                        trans = np.array(self.transition_probs[s][a1][a2], dtype=float)
                        M[a1, a2] += self.discount_factor * np.dot(trans, V)

                # Solve the matrix game via LP for player 1
                val, strat = self._solve_matrix_game(M)
                V_new[s] = val
                strategies[s] = strat

            if np.max(np.abs(V_new - V)) < tol:
                return V_new, strategies
            V = V_new

        return V, strategies

    def _solve_matrix_game(self, M: np.ndarray) -> Tuple[float, np.ndarray]:
        """Solve a zero-sum matrix game for the row player via LP.

        max v s.t. M^T p >= v*1, sum(p)=1, p >= 0.

        Args:
            M: Payoff matrix for the row player.

        Returns:
            (value, optimal_mixed_strategy) for the row player.
        """
        m, n = M.shape
        # Variables: [p_1, ..., p_m, v]
        c = np.zeros(m + 1)
        c[-1] = -1.0  # maximise v

        A_ub = np.zeros((n, m + 1))
        for j in range(n):
            A_ub[j, :m] = -M[:, j]
            A_ub[j, m] = 1.0
        b_ub = np.zeros(n)

        A_eq = np.zeros((1, m + 1))
        A_eq[0, :m] = 1.0
        b_eq = np.array([1.0])

        bounds = [(0, 1) for _ in range(m)] + [(None, None)]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method='highs')
        if res.success:
            return float(res.x[-1]), res.x[:m]
        # Fallback: uniform
        return float(np.max(np.min(M, axis=1))), np.ones(m) / m

    def stationary_state_distribution(self, policy1: List[int],
                                      policy2: List[int]) -> np.ndarray:
        """Compute the stationary distribution over states for deterministic
        policies.

        Args:
            policy1: List of actions for player 1 in each state.
            policy2: List of actions for player 2 in each state.

        Returns:
            Stationary distribution over states.
        """
        T_mat = np.zeros((self.n_states, self.n_states))
        for s in range(self.n_states):
            a1 = policy1[s]
            a2 = policy2[s]
            trans = np.array(self.transition_probs[s][a1][a2], dtype=float)
            trans = np.maximum(trans, 0)
            t_sum = np.sum(trans)
            if t_sum > 0:
                trans /= t_sum
            T_mat[s, :] = trans

        # Stationary: pi @ T = pi => (T^T - I) pi = 0, sum pi = 1
        A = T_mat.T - np.eye(self.n_states)
        A[-1, :] = 1.0
        b = np.zeros(self.n_states)
        b[-1] = 1.0
        try:
            pi = np.linalg.solve(A, b)
            pi = np.maximum(pi, 0)
            pi /= np.sum(pi)
        except np.linalg.LinAlgError:
            pi = np.ones(self.n_states) / self.n_states
        return pi


# ---------------------------------------------------------------------------
# Utility: Prisoners' Dilemma helpers
# ---------------------------------------------------------------------------

def prisoners_dilemma(R: float = 3.0, S: float = 0.0,
                      T: float = 5.0, P: float = 1.0,
                      discount_factor: float = 0.95) -> RepeatedGame:
    """Create a standard Prisoner's Dilemma repeated game.

    Actions: 0 = cooperate, 1 = defect.
    Payoffs: (R, R) for mutual cooperation, (P, P) for mutual defection,
    (S, T) if row cooperates and col defects, (T, S) vice versa.

    Args:
        R: Reward for mutual cooperation.
        S: Sucker's payoff.
        T: Temptation to defect.
        P: Punishment for mutual defection.
        discount_factor: Discount factor.

    Returns:
        A RepeatedGame instance for the Prisoner's Dilemma.
    """
    A = np.array([[R, S], [T, P]])
    B = np.array([[R, T], [S, P]])
    return RepeatedGame(A, B, discount_factor)


def build_stochastic_prisoners_dilemma(n_states: int = 2,
                                        discount_factor: float = 0.95) -> StochasticGame:
    """Build a stochastic game with n_states, each being a variant of the
    Prisoner's Dilemma with different payoff scales and random transitions.

    State 0 has standard PD payoffs, subsequent states have scaled payoffs.
    Transitions: mutual cooperation tends toward high-payoff states;
    mutual defection tends toward low-payoff states.

    Args:
        n_states: Number of states.
        discount_factor: Discount factor.

    Returns:
        A StochasticGame instance.
    """
    A_list = []
    B_list = []
    trans_list = []

    for s in range(n_states):
        scale = 1.0 + 0.5 * s
        R, S, T, P = 3 * scale, 0.0, 5 * scale, 1 * scale
        A_list.append(np.array([[R, S], [T, P]]))
        B_list.append(np.array([[R, T], [S, P]]))

        state_trans = []
        for a1 in range(2):
            row_trans = []
            for a2 in range(2):
                p = np.ones(n_states) / n_states
                if a1 == 0 and a2 == 0:
                    # Mutual cooperation biases toward higher states
                    p = np.linspace(0.1, 1.0, n_states)
                elif a1 == 1 and a2 == 1:
                    # Mutual defection biases toward lower states
                    p = np.linspace(1.0, 0.1, n_states)
                p = p / np.sum(p)
                row_trans.append(p)
            state_trans.append(row_trans)
        trans_list.append(state_trans)

    return StochasticGame(A_list, B_list, trans_list, discount_factor)


def cooperation_rate(history: History, player: int = 0) -> float:
    """Compute the fraction of rounds in which a player cooperated.

    Args:
        history: Game history.
        player: 0 or 1.

    Returns:
        Cooperation rate in [0, 1].
    """
    if len(history) == 0:
        return 0.0
    actions = history.get_actions(player)
    return float(np.mean([1.0 - a for a in actions]))


def compare_strategies(game: RepeatedGame,
                       strategies: List[Callable],
                       names: Optional[List[str]] = None,
                       T: int = 200) -> Dict[str, Any]:
    """Run a full comparison of strategies: tournament + replicator dynamics.

    Args:
        game: The repeated game.
        strategies: List of strategy functions.
        names: Optional strategy names.
        T: Rounds per match.

    Returns:
        Dict with 'tournament' results, 'replicator_equilibrium',
        and 'cooperation_rates'.
    """
    if names is None:
        names = [f"S{i}" for i in range(len(strategies))]

    tourn = Tournament(game, strategies, names, T_per_match=T)
    tourn_results = tourn.run()

    repl = ReplicatorDynamics(game, strategies, names, T_per_match=T)
    eq = repl.find_equilibrium()

    coop_rates = {}
    for i, s in enumerate(strategies):
        h = game.play(s, s, T)
        coop_rates[names[i]] = cooperation_rate(h, 0)

    return {
        'tournament': tourn_results,
        'replicator_equilibrium': dict(zip(names, eq)),
        'cooperation_rates': coop_rates,
    }
