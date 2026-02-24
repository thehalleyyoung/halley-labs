"""Information design: Bayesian persuasion, cheap talk, and optimal signal structures."""

import numpy as np
from scipy.optimize import linprog, minimize
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class OptimalSignal:
    """Optimal signal structure with posteriors, probabilities, and sender value."""
    signal_distribution: np.ndarray   # P(signal | state), shape (n_signals, n_states)
    posterior_beliefs: np.ndarray      # P(state | signal), shape (n_signals, n_states)
    signal_probabilities: np.ndarray   # P(signal), shape (n_signals,)
    sender_value: float
    receiver_actions: np.ndarray       # best-response action per signal


@dataclass
class CheapTalkEquilibrium:
    """Crawford-Sobel cheap talk equilibrium."""
    partition: List[float]
    n_messages: int
    actions: np.ndarray
    sender_expected_utility: float
    receiver_expected_utility: float


@dataclass
class InformationValue:
    """Value of a particular information structure."""
    total_value: float
    marginal_values: np.ndarray
    conditional_values: np.ndarray


@dataclass
class RatingDesign:
    """Optimal rating system design result."""
    boundaries: np.ndarray
    n_categories: int
    category_labels: np.ndarray
    expected_welfare: float
    information_loss: float


class BayesianPersuasion:
    """Bayesian persuasion via concavification (Kamenica-Gentzkow 2011)."""

    def __init__(self, n_states: int = 2, n_actions: int = 2):
        self.n_states = n_states
        self.n_actions = n_actions

    def solve(self, sender_utility: np.ndarray, receiver_utility: np.ndarray,
              prior: np.ndarray) -> OptimalSignal:
        """Solve for the optimal signal structure."""
        sender_utility = np.asarray(sender_utility, dtype=float)
        receiver_utility = np.asarray(receiver_utility, dtype=float)
        prior = np.asarray(prior, dtype=float)
        prior = prior / prior.sum()
        if self.n_states == 2 and self.n_actions == 2:
            return self._solve_binary(sender_utility, receiver_utility, prior)
        return self._solve_general(sender_utility, receiver_utility, prior)

    def _receiver_best_response(self, receiver_utility: np.ndarray,
                                belief: np.ndarray) -> int:
        """Receiver's optimal action given a belief."""
        return int(np.argmax(receiver_utility @ belief))

    def _sender_value_function(self, sender_utility: np.ndarray,
                               receiver_utility: np.ndarray,
                               belief: np.ndarray) -> float:
        """Sender's indirect utility at a given posterior."""
        action = self._receiver_best_response(receiver_utility, belief)
        return float(sender_utility[action] @ belief)

    def _solve_binary(self, sender_utility, receiver_utility, prior) -> OptimalSignal:
        """Binary-state binary-action case via concavification."""
        mu0 = prior[1]

        # Receiver indifference point: mu* where action 0 ~ action 1
        diff_0 = receiver_utility[0, 0] - receiver_utility[1, 0]
        diff_1 = receiver_utility[1, 1] - receiver_utility[0, 1]
        denom = diff_0 + diff_1

        if abs(denom) < 1e-12:
            action = self._receiver_best_response(receiver_utility, prior)
            return OptimalSignal(
                np.array([[1.0, 1.0]]), prior.reshape(1, -1),
                np.array([1.0]), float(sender_utility[action] @ prior),
                np.array([action]))

        mu_star = np.clip(diff_0 / denom, 0.0, 1.0)

        v_at_prior = self._sender_value_function(sender_utility, receiver_utility, prior)
        action_low = self._receiver_best_response(
            receiver_utility, np.array([1.0 - mu_star + 1e-10, mu_star - 1e-10]))
        action_high = self._receiver_best_response(
            receiver_utility, np.array([1.0 - mu_star - 1e-10, mu_star + 1e-10]))

        v_sender_low = sender_utility[action_low]
        v_sender_high = sender_utility[action_high]

        if 0 < mu_star < 1 and action_low != action_high:
            best_val = v_at_prior
            best_split = None

            if mu0 <= mu_star:
                w_high = mu0 / mu_star if mu_star > 1e-12 else 1.0
                w_low = 1.0 - w_high
                val = w_low * float(v_sender_low @ np.array([1.0, 0.0])) + \
                      w_high * float(v_sender_high @ np.array([1.0 - mu_star, mu_star]))
                if val > best_val + 1e-12:
                    best_val = val
                    best_split = (w_low, w_high, np.array([1.0, 0.0]),
                                  np.array([1.0 - mu_star, mu_star]),
                                  action_low, action_high)
            else:
                w_low = (1.0 - mu0) / (1.0 - mu_star) if (1.0 - mu_star) > 1e-12 else 0.0
                w_high = 1.0 - w_low
                val = w_low * float(v_sender_low @ np.array([1.0 - mu_star, mu_star])) + \
                      w_high * float(v_sender_high @ np.array([0.0, 1.0]))
                if val > best_val + 1e-12:
                    best_val = val
                    best_split = (w_low, w_high,
                                  np.array([1.0 - mu_star, mu_star]),
                                  np.array([0.0, 1.0]),
                                  action_low, action_high)

            # Try full split (0, 1)
            a0 = self._receiver_best_response(receiver_utility, np.array([1.0, 0.0]))
            a1 = self._receiver_best_response(receiver_utility, np.array([0.0, 1.0]))
            val_full = (1.0 - mu0) * float(sender_utility[a0] @ np.array([1.0, 0.0])) + \
                       mu0 * float(sender_utility[a1] @ np.array([0.0, 1.0]))
            if val_full > best_val + 1e-12:
                best_val = val_full
                best_split = (1.0 - mu0, mu0, np.array([1.0, 0.0]),
                              np.array([0.0, 1.0]), a0, a1)

            if best_split is not None:
                w_l, w_h, mu_l, mu_h, a_l, a_h = best_split
                sig_dist = np.zeros((2, 2))
                for s in range(2):
                    if prior[s] > 1e-15:
                        sig_dist[0, s] = w_l * mu_l[s] / prior[s]
                        sig_dist[1, s] = w_h * mu_h[s] / prior[s]
                sig_dist = np.clip(sig_dist, 0.0, 1.0)
                return OptimalSignal(sig_dist, np.array([mu_l, mu_h]),
                                     np.array([w_l, w_h]), best_val,
                                     np.array([a_l, a_h]))

        action = self._receiver_best_response(receiver_utility, prior)
        return OptimalSignal(
            np.array([[1.0, 1.0]]), prior.reshape(1, -1),
            np.array([1.0]), float(sender_utility[action] @ prior),
            np.array([action]))

    def _solve_general(self, sender_utility, receiver_utility, prior) -> OptimalSignal:
        """General case via linear programming over joint distributions."""
        n_a, n_s = self.n_actions, self.n_states
        n_vars = n_a * n_s

        # Objective: max sum_{a,s} sender_utility[a,s] * tau[a,s]
        c = np.zeros(n_vars)
        for a in range(n_a):
            for s in range(n_s):
                c[a * n_s + s] = -sender_utility[a, s]

        # Bayes plausibility: sum_a tau[a,s] = prior[s]
        A_eq = np.zeros((n_s, n_vars))
        for s in range(n_s):
            for a in range(n_a):
                A_eq[s, a * n_s + s] = 1.0
        b_eq = prior.copy()

        # Obedience (IC) constraints
        A_ub_rows, b_ub_rows = [], []
        for a in range(n_a):
            for ap in range(n_a):
                if ap == a:
                    continue
                row = np.zeros(n_vars)
                for s in range(n_s):
                    row[a * n_s + s] = receiver_utility[ap, s] - receiver_utility[a, s]
                A_ub_rows.append(row)
                b_ub_rows.append(0.0)

        A_ub = np.array(A_ub_rows) if A_ub_rows else None
        b_ub = np.array(b_ub_rows) if b_ub_rows else None
        bounds = [(0.0, None)] * n_vars

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method="highs")
        tau = np.maximum(result.x.reshape(n_a, n_s), 0.0)

        sig_probs = tau.sum(axis=1)
        active = sig_probs > 1e-12
        n_signals = int(active.sum())

        if n_signals == 0:
            action = self._receiver_best_response(receiver_utility, prior)
            return OptimalSignal(
                np.array([[1.0] * n_s]), prior.reshape(1, -1),
                np.array([1.0]), float(sender_utility[action] @ prior),
                np.array([action]))

        posteriors = np.zeros((n_signals, n_s))
        actions = np.zeros(n_signals, dtype=int)
        probs = np.zeros(n_signals)
        sig_dist = np.zeros((n_signals, n_s))

        idx = 0
        for a in range(n_a):
            if not active[a]:
                continue
            probs[idx] = sig_probs[a]
            posteriors[idx] = tau[a] / sig_probs[a]
            actions[idx] = a
            for s in range(n_s):
                if prior[s] > 1e-15:
                    sig_dist[idx, s] = tau[a, s] / prior[s]
            idx += 1

        return OptimalSignal(sig_dist, posteriors, probs,
                             float(np.sum(tau * sender_utility)), actions)


class MultiReceiverPersuasion:
    """Bayesian persuasion with multiple receivers (public and private signals)."""

    def __init__(self, n_states: int, n_receivers: int,
                 n_actions_per_receiver: List[int]):
        self.n_states = n_states
        self.n_receivers = n_receivers
        self.n_actions = n_actions_per_receiver

    def solve_public(self, sender_utility: np.ndarray,
                     receiver_utilities: List[np.ndarray],
                     prior: np.ndarray) -> OptimalSignal:
        """Optimal public signal observed by all receivers."""
        prior = np.asarray(prior, dtype=float)
        prior = prior / prior.sum()
        n_s = self.n_states

        grid_beliefs = self._discretize_simplex(n_s, resolution=50)
        n_grid = len(grid_beliefs)

        grid_values = np.zeros(n_grid)
        grid_actions = np.zeros(n_grid, dtype=int)
        for i, belief in enumerate(grid_beliefs):
            profile = []
            for r in range(self.n_receivers):
                eu = receiver_utilities[r] @ belief
                profile.append(int(np.argmax(eu)))
            profile_idx = self._profile_to_index(profile)
            grid_actions[i] = profile_idx
            grid_values[i] = float(sender_utility[profile_idx] @ belief)

        # LP: Bayes-plausible weights on grid beliefs maximising sender value
        c = -grid_values
        A_eq = np.vstack([
            np.array([b for b in grid_beliefs]).T,
            np.ones((1, n_grid))
        ])
        b_eq = np.append(prior, 1.0)
        bounds = [(0.0, 1.0)] * n_grid

        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        weights = np.maximum(result.x, 0.0)
        active = weights > 1e-10

        posteriors = grid_beliefs[active]
        probs = weights[active]
        probs = probs / probs.sum()
        acts = grid_actions[active]
        n_sig = int(active.sum())

        sig_dist = np.zeros((n_sig, n_s))
        for i in range(n_sig):
            for s in range(n_s):
                if prior[s] > 1e-15:
                    sig_dist[i, s] = probs[i] * posteriors[i, s] / prior[s]

        return OptimalSignal(sig_dist, posteriors, probs,
                             float(probs @ grid_values[active]), acts)

    def solve_private(self, sender_utility: np.ndarray,
                      receiver_utilities: List[np.ndarray],
                      prior: np.ndarray,
                      n_signals_per_receiver: Optional[List[int]] = None
                      ) -> List[OptimalSignal]:
        """Optimal private signals via alternating best-response."""
        prior = np.asarray(prior, dtype=float)
        prior = prior / prior.sum()
        n_s = self.n_states

        if n_signals_per_receiver is None:
            n_signals_per_receiver = [a + 1 for a in self.n_actions]

        signals = []
        for r in range(self.n_receivers):
            n_sig = min(n_signals_per_receiver[r], n_s)
            sig = np.eye(n_s)[:n_sig]
            sig = sig / sig.sum(axis=0, keepdims=True)
            signals.append(np.clip(sig, 0, 1))

        for _ in range(20):
            for r in range(self.n_receivers):
                signals[r] = self._optimize_one_signal(
                    r, signals, sender_utility, receiver_utilities, prior,
                    n_signals_per_receiver[r])

        results = []
        for r in range(self.n_receivers):
            sig = signals[r]
            n_sig = sig.shape[0]
            probs = np.maximum(sig @ prior, 1e-15)
            posteriors = np.zeros((n_sig, n_s))
            for i in range(n_sig):
                posteriors[i] = sig[i] * prior / probs[i]
            actions = np.array([int(np.argmax(receiver_utilities[r] @ posteriors[i]))
                                for i in range(n_sig)])
            s_val = sum(probs[i] * float(sender_utility[actions[i]] @ posteriors[i])
                        for i in range(n_sig))
            results.append(OptimalSignal(sig, posteriors, probs, s_val, actions))
        return results

    def _optimize_one_signal(self, r, signals, sender_utility,
                             receiver_utilities, prior, n_sig):
        """Optimize receiver r's signal holding others fixed."""
        n_s = self.n_states
        n_a = self.n_actions[r]
        n_vars = n_sig * n_s

        c = np.zeros(n_vars)
        for sig_idx in range(n_sig):
            for s in range(n_s):
                best_a = int(np.argmax(receiver_utilities[r][:, s]))
                if best_a < sender_utility.shape[0]:
                    c[sig_idx * n_s + s] = -sender_utility[best_a, s]

        # sum over signals for each state = 1
        A_eq = np.zeros((n_s, n_vars))
        for s in range(n_s):
            for sig_idx in range(n_sig):
                A_eq[s, sig_idx * n_s + s] = 1.0

        result = linprog(c, A_eq=A_eq, b_eq=np.ones(n_s),
                         bounds=[(0.0, 1.0)] * n_vars, method="highs")
        return np.clip(result.x.reshape(n_sig, n_s), 0.0, 1.0)

    def _profile_to_index(self, profile: List[int]) -> int:
        idx, multiplier = 0, 1
        for r in range(self.n_receivers - 1, -1, -1):
            idx += profile[r] * multiplier
            multiplier *= self.n_actions[r]
        return idx

    def _discretize_simplex(self, dim: int, resolution: int = 50) -> np.ndarray:
        """Evenly spaced points on the probability simplex."""
        if dim == 2:
            mus = np.linspace(0.0, 1.0, resolution + 1)
            return np.column_stack([1.0 - mus, mus])
        points = []
        self._simplex_grid_recursive(dim, resolution, [], points)
        return np.array(points)

    def _simplex_grid_recursive(self, dim, resolution, partial, points):
        if dim == 1:
            points.append(partial + [max(1.0 - sum(partial), 0.0)])
            return
        for k in range(resolution + 1):
            val = k / resolution
            if sum(partial) + val > 1.0 + 1e-10:
                break
            self._simplex_grid_recursive(dim - 1, resolution, partial + [val], points)


class CheapTalk:
    """Crawford-Sobel (1982) cheap talk: uniform-quadratic model.

    Sender type t ~ U[0,1], sender utility -(a-t-b)^2, receiver utility -(a-t)^2.
    """

    def __init__(self, bias: float):
        if bias < 0:
            raise ValueError("Bias must be non-negative.")
        self.bias = bias

    def max_partitions(self) -> int:
        """Maximum N such that an N-element partition equilibrium exists."""
        if self.bias == 0:
            return 10000
        return max(int(np.floor(0.5 * (1.0 + np.sqrt(1.0 + 2.0 / self.bias)))), 1)

    def compute_equilibrium(self, n_messages: Optional[int] = None
                            ) -> CheapTalkEquilibrium:
        """Compute Crawford-Sobel equilibrium with n_messages messages.

        Partition boundaries satisfy a_{i+1} - a_i = a_i - a_{i-1} + 4b,
        with a_0=0, a_N=1. Closed-form: a_i = i/N + 2b*i*(i-N).
        """
        n_max = self.max_partitions()
        if n_messages is None:
            n_messages = n_max
        if n_messages > n_max:
            raise ValueError(f"Max {n_max} messages for bias {self.bias}")
        if n_messages < 1:
            raise ValueError("Need at least 1 message.")

        n, b = n_messages, self.bias
        partition = [0.0] + [i / n + 2.0 * b * i * (i - n) for i in range(1, n + 1)]
        partition = np.clip(np.array(partition), 0.0, 1.0)

        if np.any(np.diff(partition) <= -1e-10):
            raise ValueError(f"Partition not feasible for n={n_messages}, bias={b}")

        actions = np.array([(partition[i] + partition[i + 1]) / 2.0 for i in range(n)])

        sender_eu, receiver_eu = 0.0, 0.0
        for i in range(n):
            lo, hi = partition[i], partition[i + 1]
            length = hi - lo
            if length < 1e-15:
                continue
            a = actions[i]
            e_t = (lo + hi) / 2.0
            e_t2 = (lo ** 2 + lo * hi + hi ** 2) / 3.0
            sender_eu += length * (-(a - b) ** 2 + 2.0 * (a - b) * e_t - e_t2)
            receiver_eu += length * (-a ** 2 + 2.0 * a * e_t - e_t2)

        return CheapTalkEquilibrium(
            partition.tolist(), n, actions, sender_eu, receiver_eu)

    def all_equilibria(self) -> List[CheapTalkEquilibrium]:
        """Compute all equilibria from 1 to max_partitions messages."""
        results = []
        for n in range(1, min(self.max_partitions(), 50) + 1):
            try:
                results.append(self.compute_equilibrium(n))
            except ValueError:
                break
        return results


class MultipleSenderPersuasion:
    """Bayesian persuasion with competing senders (Nash equilibrium via
    best-response iteration)."""

    def __init__(self, n_states: int, n_actions: int, n_senders: int):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_senders = n_senders

    def compute_equilibrium(self, sender_utilities: List[np.ndarray],
                            receiver_utility: np.ndarray,
                            prior: np.ndarray,
                            n_signals_per_sender: Optional[List[int]] = None,
                            max_iterations: int = 100,
                            tol: float = 1e-8) -> List[OptimalSignal]:
        """Nash equilibrium via iterated best response."""
        prior = np.asarray(prior, dtype=float)
        prior = prior / prior.sum()
        n_s = self.n_states

        if n_signals_per_sender is None:
            n_signals_per_sender = [self.n_actions] * self.n_senders

        # Initialize: uninformative signals
        current_signals = [np.ones((n_signals_per_sender[k], n_s)) / n_signals_per_sender[k]
                           for k in range(self.n_senders)]
        prev_values = np.full(self.n_senders, -np.inf)
        bp = BayesianPersuasion(n_states=n_s, n_actions=self.n_actions)

        for _ in range(max_iterations):
            new_values = np.zeros(self.n_senders)
            for k in range(self.n_senders):
                result = bp.solve(sender_utilities[k], receiver_utility, prior)
                current_signals[k] = result.signal_distribution
                new_values[k] = result.sender_value
            if np.max(np.abs(new_values - prev_values)) < tol:
                break
            prev_values = new_values.copy()

        results = []
        for k in range(self.n_senders):
            sig = current_signals[k]
            n_sig = sig.shape[0]
            probs = np.maximum(np.array([sig[i] @ prior for i in range(n_sig)]), 1e-15)
            posteriors = np.zeros((n_sig, n_s))
            for i in range(n_sig):
                posteriors[i] = sig[i] * prior / probs[i]
            actions = np.array([int(np.argmax(receiver_utility @ posteriors[i]))
                                for i in range(n_sig)])
            results.append(OptimalSignal(sig, posteriors, probs,
                                         prev_values[k], actions))
        return results


class InformationValueComputer:
    """Compute value of information, Blackwell dominance, mutual information."""

    def __init__(self, n_states: int, n_actions: int):
        self.n_states = n_states
        self.n_actions = n_actions

    def compute(self, utility: np.ndarray, prior: np.ndarray,
                signal_distribution: np.ndarray) -> InformationValue:
        """Value of information = E[max_a u(a,s)|signal] - max_a E[u(a,s)]."""
        utility = np.asarray(utility, dtype=float)
        prior = np.asarray(prior, dtype=float)
        prior = prior / prior.sum()
        signal_distribution = np.asarray(signal_distribution, dtype=float)
        n_signals = signal_distribution.shape[0]

        eu_prior = np.max(utility @ prior)
        sig_probs = np.maximum(signal_distribution @ prior, 1e-15)
        posteriors = signal_distribution * prior[np.newaxis, :] / sig_probs[:, np.newaxis]

        conditional_values = np.array([np.max(utility @ posteriors[i])
                                       for i in range(n_signals)])
        eu_signal = np.sum(sig_probs * conditional_values)
        total_value = eu_signal - eu_prior
        marginal_values = sig_probs * (conditional_values - eu_prior)

        return InformationValue(total_value, marginal_values, conditional_values)

    def blackwell_dominance(self, signal_a: np.ndarray,
                            signal_b: np.ndarray) -> bool:
        """Check if signal_a Blackwell-dominates signal_b (A more informative).

        Checks feasibility of a garbling matrix M: signal_b = M @ signal_a.
        """
        n_a, n_b = signal_a.shape[0], signal_b.shape[0]
        for j in range(n_b):
            A_eq = np.vstack([signal_a.T, np.ones((1, n_a))])
            b_eq = np.append(signal_b[j], 1.0)
            result = linprog(np.zeros(n_a), A_eq=A_eq, b_eq=b_eq,
                             bounds=[(0.0, 1.0)] * n_a, method="highs")
            if not result.success:
                return False
        return True

    def mutual_information(self, prior: np.ndarray,
                           signal_distribution: np.ndarray) -> float:
        """Mutual information I(state; signal) in nats."""
        prior = np.asarray(prior, dtype=float)
        prior = prior / prior.sum()
        joint = signal_distribution * prior[np.newaxis, :]
        p_signal = joint.sum(axis=1)

        mi = 0.0
        for i in range(signal_distribution.shape[0]):
            for s in range(len(prior)):
                if joint[i, s] > 1e-15 and p_signal[i] > 1e-15 and prior[s] > 1e-15:
                    mi += joint[i, s] * np.log(joint[i, s] / (p_signal[i] * prior[s]))
        return mi


class ExperimentDesign:
    """Optimal information acquisition order for sequential experiments."""

    def __init__(self, n_states: int, n_actions: int):
        self.n_states = n_states
        self.n_actions = n_actions

    def greedy_order(self, utility: np.ndarray, prior: np.ndarray,
                     experiments: List[np.ndarray], costs: np.ndarray,
                     budget: float) -> Tuple[List[int], float]:
        """Greedy experiment selection by bang-per-buck (value/cost ratio)."""
        utility = np.asarray(utility, dtype=float)
        prior = np.asarray(prior, dtype=float)
        prior = prior / prior.sum()
        costs = np.asarray(costs, dtype=float)

        iv = InformationValueComputer(self.n_states, self.n_actions)
        remaining = set(range(len(experiments)))
        order, total_value, spent = [], 0.0, 0.0
        current_prior = prior.copy()

        while remaining and spent < budget - 1e-12:
            best_ratio, best_exp, best_info_val = -np.inf, -1, None

            for e in remaining:
                if costs[e] + spent > budget + 1e-12:
                    continue
                info_val = iv.compute(utility, current_prior, experiments[e])
                ratio = info_val.total_value / max(costs[e], 1e-15)
                if ratio > best_ratio:
                    best_ratio, best_exp, best_info_val = ratio, e, info_val

            if best_exp < 0 or best_info_val is None or best_info_val.total_value < 1e-15:
                break

            order.append(best_exp)
            total_value += best_info_val.total_value
            spent += costs[best_exp]
            remaining.discard(best_exp)

            # Update belief using most likely signal realization
            exp_sig = experiments[best_exp]
            sig_probs = np.maximum(exp_sig @ current_prior, 1e-15)
            best_sig = int(np.argmax(sig_probs))
            new_prior = exp_sig[best_sig] * current_prior / sig_probs[best_sig]
            current_prior = new_prior / new_prior.sum()

        return order, total_value

    def optimal_subset(self, utility: np.ndarray, prior: np.ndarray,
                       experiments: List[np.ndarray], costs: np.ndarray,
                       budget: float) -> Tuple[List[int], float]:
        """Optimal subset via exhaustive search (falls back to greedy if >20)."""
        costs = np.asarray(costs, dtype=float)
        prior = np.asarray(prior, dtype=float)
        prior = prior / prior.sum()
        n_exp = len(experiments)

        if n_exp > 20:
            return self.greedy_order(utility, prior, experiments, costs, budget)

        iv = InformationValueComputer(self.n_states, self.n_actions)
        best_value, best_subset = -np.inf, []

        for mask in range(1, 1 << n_exp):
            indices = [i for i in range(n_exp) if mask & (1 << i)]
            if sum(costs[i] for i in indices) > budget + 1e-12:
                continue
            combined = self._compose_experiments(
                [experiments[i] for i in indices], prior)
            info_val = iv.compute(utility, prior, combined)
            if info_val.total_value > best_value:
                best_value, best_subset = info_val.total_value, indices

        return best_subset, max(best_value, 0.0)

    def _compose_experiments(self, experiment_list: List[np.ndarray],
                             prior: np.ndarray) -> np.ndarray:
        """Compose independent experiments via Kronecker product over signals."""
        n_s = len(prior)
        combined = experiment_list[0].copy()
        for i in range(1, len(experiment_list)):
            nxt = experiment_list[i]
            n1, n2 = combined.shape[0], nxt.shape[0]
            new_combined = np.zeros((n1 * n2, n_s))
            for a in range(n1):
                for b in range(n2):
                    new_combined[a * n2 + b] = combined[a] * nxt[b]
            combined = new_combined
        return combined


class RatingSystemDesign:
    """Optimal coarsening of continuous quality signals via Lloyd-Max quantization."""

    def __init__(self, n_grid: int = 200):
        self.n_grid = n_grid

    def optimal_rating(self, n_categories: int,
                       quality_density: Optional[np.ndarray] = None,
                       utility_func: Optional[callable] = None) -> RatingDesign:
        """Find optimal rating boundaries minimising E[(q - E[q|category])^2].

        Uses Lloyd-Max algorithm (iterative centroid/boundary update).
        """
        grid = np.linspace(0.0, 1.0, self.n_grid)
        dq = grid[1] - grid[0]

        if quality_density is None:
            density = np.ones(self.n_grid)
        else:
            density = np.asarray(quality_density, dtype=float)
        density = density / (density.sum() * dq)

        if utility_func is None:
            def utility_func(a, q):
                return -(a - q) ** 2

        boundaries = np.linspace(0.0, 1.0, n_categories + 1)
        labels = np.zeros(n_categories)

        for _ in range(200):
            old_boundaries = boundaries.copy()

            # Centroids: conditional expectations within each category
            for k in range(n_categories):
                mask = (grid >= boundaries[k]) & (grid < boundaries[k + 1] + 1e-12)
                if mask.sum() == 0:
                    labels[k] = (boundaries[k] + boundaries[k + 1]) / 2.0
                    continue
                weights = density[mask] * dq
                total_w = weights.sum()
                labels[k] = np.sum(grid[mask] * weights) / total_w if total_w > 1e-15 \
                    else (boundaries[k] + boundaries[k + 1]) / 2.0

            # Boundaries: midpoints between adjacent centroids
            for k in range(1, n_categories):
                boundaries[k] = (labels[k - 1] + labels[k]) / 2.0
            boundaries[0], boundaries[-1] = 0.0, 1.0

            if np.max(np.abs(boundaries - old_boundaries)) < 1e-10:
                break

        total_welfare, full_info_welfare = 0.0, 0.0
        for k in range(n_categories):
            mask = (grid >= boundaries[k]) & (grid < boundaries[k + 1] + 1e-12)
            if mask.sum() == 0:
                continue
            weights = density[mask] * dq
            total_welfare += np.sum(weights * np.array(
                [utility_func(labels[k], q) for q in grid[mask]]))
            full_info_welfare += np.sum(weights * np.array(
                [utility_func(q, q) for q in grid[mask]]))

        info_loss = 0.0
        if abs(full_info_welfare) > 1e-15:
            info_loss = 1.0 - total_welfare / full_info_welfare
        info_loss = max(0.0, min(1.0, info_loss))

        return RatingDesign(boundaries, n_categories, labels,
                            total_welfare, info_loss)

    def compare_systems(self, max_categories: int,
                        quality_density: Optional[np.ndarray] = None
                        ) -> List[RatingDesign]:
        """Compare rating systems with 1 to max_categories categories."""
        return [self.optimal_rating(k, quality_density)
                for k in range(1, max_categories + 1)]

    def optimal_number_of_categories(self, max_categories: int,
                                     info_cost_per_category: float,
                                     quality_density: Optional[np.ndarray] = None
                                     ) -> Tuple[int, RatingDesign]:
        """Find optimal n_categories trading off info value vs complexity cost."""
        systems = self.compare_systems(max_categories, quality_density)
        net_values = np.array([s.expected_welfare - info_cost_per_category * s.n_categories
                               for s in systems])
        best_idx = int(np.argmax(net_values))
        return systems[best_idx].n_categories, systems[best_idx]
