"""
Multi-Agent Simulation Framework for Mechanism Design.

Provides classes and algorithms for simulating strategic interactions among
agents in auction and mechanism design settings. Includes best-response
dynamics, regret minimization (Exp3), Bayesian belief updating, information
asymmetry modeling, collusion detection, and welfare analysis.

Dependencies: numpy, scipy
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar, minimize
from scipy.special import betaln, gammaln


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

class SimulationResult:
    """Container for the full output of a multi-agent simulation run.

    Attributes:
        outcomes_per_round: list of dicts, one per round. Each dict maps
            agent index to the outcome (allocation, payment) for that round.
        agent_utilities: np.ndarray of shape (n_rounds, n_agents) with the
            per-round realised utility for every agent.
        revenue: np.ndarray of shape (n_rounds,) with the mechanism's
            per-round revenue (sum of payments collected).
        welfare_metrics: dict with keys 'utilitarian', 'egalitarian',
            'nash' containing np.ndarrays of per-round welfare values.
        convergence_info: dict with keys 'converged' (bool),
            'convergence_round' (int or None), 'strategy_distances'
            (np.ndarray of per-round max strategy change norms).
    """

    def __init__(self, n_rounds: int, n_agents: int):
        """Initialise empty result arrays.

        Args:
            n_rounds: Number of simulation rounds.
            n_agents: Number of agents.
        """
        self.outcomes_per_round: list = []
        self.agent_utilities: np.ndarray = np.zeros((n_rounds, n_agents))
        self.revenue: np.ndarray = np.zeros(n_rounds)
        self.welfare_metrics: dict = {
            "utilitarian": np.zeros(n_rounds),
            "egalitarian": np.zeros(n_rounds),
            "nash": np.zeros(n_rounds),
        }
        self.convergence_info: dict = {
            "converged": False,
            "convergence_round": None,
            "strategy_distances": np.zeros(n_rounds),
        }

    def summary(self) -> dict:
        """Return an aggregate summary of the simulation.

        Returns:
            Dictionary with mean revenue, mean utilities, final welfare,
            and convergence flag.
        """
        return {
            "mean_revenue": float(np.mean(self.revenue)),
            "std_revenue": float(np.std(self.revenue)),
            "mean_agent_utilities": self.agent_utilities.mean(axis=0).tolist(),
            "final_utilitarian_welfare": float(self.welfare_metrics["utilitarian"][-1]),
            "final_egalitarian_welfare": float(self.welfare_metrics["egalitarian"][-1]),
            "final_nash_welfare": float(self.welfare_metrics["nash"][-1]),
            "converged": self.convergence_info["converged"],
            "convergence_round": self.convergence_info["convergence_round"],
        }


# ---------------------------------------------------------------------------
# Belief models
# ---------------------------------------------------------------------------

class BetaBinomialBelief:
    """Bayesian belief over a Bernoulli parameter using a Beta conjugate prior.

    The agent maintains a Beta(alpha, beta) posterior over an unknown
    probability p (e.g., the probability that an opponent bids high).

    Attributes:
        alpha: Positive shape parameter of the Beta distribution.
        beta: Positive shape parameter of the Beta distribution.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        """Initialise with prior hyper-parameters.

        Args:
            alpha: Prior successes + 1 (default 1 = uniform prior).
            beta: Prior failures + 1 (default 1 = uniform prior).
        """
        self.alpha = float(alpha)
        self.beta = float(beta)

    def update(self, observation: int) -> None:
        """Bayesian update after observing a binary outcome.

        Args:
            observation: 1 for success, 0 for failure.
        """
        if observation == 1:
            self.alpha += 1.0
        else:
            self.beta += 1.0

    def mean(self) -> float:
        """Posterior mean E[p]."""
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        """Posterior variance Var[p]."""
        ab = self.alpha + self.beta
        return (self.alpha * self.beta) / (ab * ab * (ab + 1.0))

    def sample(self, rng: np.random.Generator) -> float:
        """Draw a single sample from the posterior.

        Args:
            rng: NumPy random generator.

        Returns:
            A float in [0, 1].
        """
        return rng.beta(self.alpha, self.beta)

    def log_predictive(self, x: int) -> float:
        """Log-predictive probability P(X=x | data) under Beta-Binomial.

        Uses the ratio of Beta functions:
            P(X=1) = alpha / (alpha + beta),  P(X=0) = beta / (alpha + beta).

        Args:
            x: 0 or 1.

        Returns:
            Log probability.
        """
        if x == 1:
            return np.log(self.alpha) - np.log(self.alpha + self.beta)
        return np.log(self.beta) - np.log(self.alpha + self.beta)


class NormalNormalBelief:
    """Bayesian belief over a Normal mean with known variance using a
    Normal conjugate prior.

    Prior: mu ~ N(mu_0, sigma_0^2).
    Likelihood: x | mu ~ N(mu, sigma_obs^2).

    Attributes:
        mu: Current posterior mean.
        sigma_sq: Current posterior variance.
        sigma_obs_sq: Known observation noise variance.
    """

    def __init__(self, mu_0: float = 0.0, sigma_0_sq: float = 1.0,
                 sigma_obs_sq: float = 1.0):
        """Initialise prior.

        Args:
            mu_0: Prior mean.
            sigma_0_sq: Prior variance.
            sigma_obs_sq: Known observation variance.
        """
        self.mu = float(mu_0)
        self.sigma_sq = float(sigma_0_sq)
        self.sigma_obs_sq = float(sigma_obs_sq)

    def update(self, observation: float) -> None:
        """Bayesian update after observing a real-valued datum.

        Posterior precision = prior precision + likelihood precision.
        Posterior mean = weighted combination.

        Args:
            observation: Observed real value.
        """
        prior_prec = 1.0 / self.sigma_sq
        lik_prec = 1.0 / self.sigma_obs_sq
        post_prec = prior_prec + lik_prec
        self.mu = (prior_prec * self.mu + lik_prec * observation) / post_prec
        self.sigma_sq = 1.0 / post_prec

    def mean(self) -> float:
        """Posterior mean."""
        return self.mu

    def variance(self) -> float:
        """Posterior variance."""
        return self.sigma_sq

    def sample(self, rng: np.random.Generator) -> float:
        """Draw a single posterior sample.

        Args:
            rng: NumPy random generator.

        Returns:
            A float.
        """
        return rng.normal(self.mu, np.sqrt(self.sigma_sq))

    def log_predictive(self, x: float) -> float:
        """Log-predictive density p(x | data) integrating out mu.

        The predictive is Normal with mean = posterior mean and variance =
        posterior variance + observation variance.

        Args:
            x: Observed value.

        Returns:
            Log density value.
        """
        pred_var = self.sigma_sq + self.sigma_obs_sq
        return -0.5 * np.log(2.0 * np.pi * pred_var) - 0.5 * (x - self.mu) ** 2 / pred_var


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """An agent participating in a mechanism.

    Each agent has a parameterised utility function, a strategy (action
    selection rule), beliefs about opponents, and a budget constraint.

    Attributes:
        agent_id: Unique identifier.
        utility_function: Callable (value, payment) -> float.
        strategy: np.ndarray — current mixed-strategy probability vector
            over a discrete action set.
        beliefs: dict mapping opponent id to a belief object.
        budget: float — maximum total payment the agent can make.
        total_payment: float — cumulative payments so far.
        valuation: float — private valuation drawn from the type space.
        private_signal: float | None — private information signal.
        action_space: np.ndarray — discrete set of possible actions (bids).
    """

    def __init__(self, agent_id: int, valuation: float, budget: float,
                 n_actions: int = 20, action_range: tuple = (0.0, 1.0)):
        """Create a new agent.

        Args:
            agent_id: Unique integer identifier.
            valuation: Private valuation for the good.
            budget: Total budget available.
            n_actions: Number of discrete actions (bid levels).
            action_range: (low, high) tuple for the action space.
        """
        self.agent_id = agent_id
        self.valuation = float(valuation)
        self.budget = float(budget)
        self.total_payment = 0.0
        self.n_actions = n_actions
        self.action_space = np.linspace(action_range[0], action_range[1], n_actions)
        self.strategy = np.ones(n_actions) / n_actions  # uniform initially
        self.beliefs: dict = {}
        self.private_signal: float | None = None

        # Regret-minimisation state (Exp3)
        self._exp3_weights = np.ones(n_actions)
        self._exp3_gamma = 0.1  # exploration parameter
        self._cumulative_rewards = np.zeros(n_actions)
        self._action_counts = np.zeros(n_actions)
        self._round_counter = 0

        # Best-response dynamics state
        self._br_history: list = []

    def utility_function(self, value: float, payment: float) -> float:
        """Quasi-linear utility: u = value - payment.

        Args:
            value: Realised value from allocation.
            payment: Payment made to the mechanism.

        Returns:
            Utility as a float.
        """
        return value - payment

    def _budget_feasible_mask(self) -> np.ndarray:
        """Boolean mask over actions that respect the remaining budget.

        Returns:
            1-D boolean array of length n_actions.
        """
        remaining = self.budget - self.total_payment
        return self.action_space <= remaining + 1e-12

    def select_action(self, rng: np.random.Generator) -> float:
        """Sample an action from the current mixed strategy respecting budget.

        Args:
            rng: NumPy random generator.

        Returns:
            Chosen action (bid level).
        """
        feasible = self._budget_feasible_mask()
        probs = self.strategy * feasible
        total = probs.sum()
        if total < 1e-15:
            # If nothing is feasible, bid zero
            return self.action_space[0]
        probs /= total
        idx = rng.choice(self.n_actions, p=probs)
        return self.action_space[idx]

    def receive_outcome(self, allocation: float, payment: float) -> float:
        """Process outcome of a round.

        Args:
            allocation: Value of the allocation received.
            payment: Payment made.

        Returns:
            Realised utility.
        """
        self.total_payment += payment
        return self.utility_function(allocation, payment)

    # ---- Belief management ------------------------------------------------

    def init_beliefs(self, opponent_ids: list, belief_type: str = "beta"):
        """Initialise beliefs about each opponent.

        Args:
            opponent_ids: List of opponent agent ids.
            belief_type: 'beta' for BetaBinomialBelief or 'normal' for
                NormalNormalBelief.
        """
        for oid in opponent_ids:
            if belief_type == "beta":
                self.beliefs[oid] = BetaBinomialBelief()
            else:
                self.beliefs[oid] = NormalNormalBelief()

    def update_beliefs(self, observations: dict) -> None:
        """Update beliefs given observed opponent behaviour.

        Args:
            observations: Dict mapping opponent id to observed value.
                For BetaBinomialBelief this should be 0/1; for
                NormalNormalBelief any float.
        """
        for oid, obs in observations.items():
            if oid in self.beliefs:
                self.beliefs[oid].update(obs)

    # ---- Private signals --------------------------------------------------

    def receive_signal(self, signal: float) -> None:
        """Receive a private information signal.

        Args:
            signal: Private signal value.
        """
        self.private_signal = signal


# ---------------------------------------------------------------------------
# Strategic behaviour algorithms
# ---------------------------------------------------------------------------

class Exp3Algorithm:
    """Exp3 (Exponential-weight algorithm for Exploration and Exploitation)
    for adversarial multi-armed bandits.

    Implements the full Exp3 algorithm from Auer et al. (2002) with
    importance-weighted reward estimates and tunable exploration.

    Attributes:
        n_actions: Size of the action set.
        gamma: Exploration rate in (0, 1].
        weights: Exponential weights array.
    """

    def __init__(self, n_actions: int, gamma: float = 0.1):
        """Initialise Exp3.

        Args:
            n_actions: Number of arms / actions.
            gamma: Exploration mixing parameter.
        """
        self.n_actions = n_actions
        self.gamma = min(max(gamma, 1e-8), 1.0)
        self.weights = np.ones(n_actions, dtype=np.float64)
        self._cumulative_loss = np.zeros(n_actions, dtype=np.float64)
        self._t = 0

    def get_distribution(self) -> np.ndarray:
        """Compute the mixed strategy (probability distribution) over actions.

        p_i = (1 - gamma) * w_i / sum(w) + gamma / K

        Returns:
            Probability vector of length n_actions.
        """
        w_sum = self.weights.sum()
        if w_sum < 1e-30:
            return np.ones(self.n_actions) / self.n_actions
        probs = (1.0 - self.gamma) * (self.weights / w_sum) + self.gamma / self.n_actions
        probs = np.maximum(probs, 1e-15)
        probs /= probs.sum()
        return probs

    def select_action(self, rng: np.random.Generator) -> int:
        """Sample an action from the current distribution.

        Args:
            rng: NumPy random generator.

        Returns:
            Action index.
        """
        probs = self.get_distribution()
        return int(rng.choice(self.n_actions, p=probs))

    def update(self, chosen_action: int, reward: float) -> None:
        """Update weights after observing a reward for the chosen action.

        Uses importance-weighted estimator:
            r_hat_i = reward / p_i   if i == chosen_action, else 0
            w_i *= exp(gamma * r_hat_i / K)

        Args:
            chosen_action: Index of the action that was played.
            reward: Observed reward in [0, 1] (or rescaled).
        """
        self._t += 1
        probs = self.get_distribution()
        estimated_reward = np.zeros(self.n_actions)
        p_a = max(probs[chosen_action], 1e-15)
        estimated_reward[chosen_action] = reward / p_a

        # Multiplicative weight update
        exponents = self.gamma * estimated_reward / self.n_actions
        # Clip to avoid overflow
        exponents = np.clip(exponents, -20.0, 20.0)
        self.weights *= np.exp(exponents)

        # Renormalise to prevent numerical drift
        w_max = self.weights.max()
        if w_max > 1e100:
            self.weights /= w_max

    def cumulative_regret_bound(self) -> float:
        """Theoretical upper bound on expected cumulative regret.

        Bound: E[R_T] <= 2 * sqrt(T * K * ln(K)) when gamma is tuned
        optimally.  This returns the bound given the current T.

        Returns:
            Float upper bound.
        """
        if self._t == 0:
            return 0.0
        return 2.0 * np.sqrt(self._t * self.n_actions * np.log(self.n_actions))


class BestResponseDynamics:
    """Compute best-response strategies given beliefs about opponent play.

    In each round the agent picks the action that maximises its expected
    utility given its current beliefs about what opponents will do.

    Attributes:
        action_space: np.ndarray of possible actions.
        valuation: Agent's private valuation.
    """

    def __init__(self, action_space: np.ndarray, valuation: float):
        """Initialise.

        Args:
            action_space: 1-D array of discrete actions.
            valuation: Agent's private valuation.
        """
        self.action_space = action_space
        self.valuation = valuation

    def compute_best_response(self, opponent_strategies: list,
                              mechanism_fn, rng: np.random.Generator,
                              n_samples: int = 200) -> int:
        """Find the best-response action index via Monte-Carlo expected utility.

        For each of our actions, sample opponent actions from their
        strategies, run the mechanism, and average the utility.

        Args:
            opponent_strategies: List of (action_space, prob_vector) tuples
                for each opponent.
            mechanism_fn: Callable (bids: np.ndarray) -> (allocations, payments).
                Both allocations and payments are arrays of length n_agents;
                our agent is always at index 0.
            rng: NumPy random generator.
            n_samples: Number of Monte-Carlo samples.

        Returns:
            Index of the best-response action.
        """
        n_actions = len(self.action_space)
        expected_utility = np.zeros(n_actions)

        for s in range(n_samples):
            # Sample opponent actions
            opp_actions = []
            for opp_as, opp_probs in opponent_strategies:
                idx = rng.choice(len(opp_as), p=opp_probs)
                opp_actions.append(opp_as[idx])

            for a_idx in range(n_actions):
                bids = np.array([self.action_space[a_idx]] + opp_actions)
                allocations, payments = mechanism_fn(bids)
                alloc_value = allocations[0] * self.valuation
                util = alloc_value - payments[0]
                expected_utility[a_idx] += util

        expected_utility /= n_samples
        return int(np.argmax(expected_utility))

    def compute_epsilon_best_response(self, opponent_strategies: list,
                                      mechanism_fn,
                                      rng: np.random.Generator,
                                      epsilon: float = 0.05,
                                      n_samples: int = 200) -> np.ndarray:
        """Compute an epsilon-greedy best response mixed strategy.

        With probability (1 - epsilon) play the pure best response; with
        probability epsilon play uniformly at random.

        Args:
            opponent_strategies: As in compute_best_response.
            mechanism_fn: As in compute_best_response.
            rng: NumPy random generator.
            epsilon: Exploration probability.
            n_samples: Monte-Carlo sample count.

        Returns:
            Mixed strategy np.ndarray.
        """
        br_idx = self.compute_best_response(opponent_strategies, mechanism_fn,
                                            rng, n_samples)
        n = len(self.action_space)
        strategy = np.full(n, epsilon / n)
        strategy[br_idx] += 1.0 - epsilon
        return strategy


# ---------------------------------------------------------------------------
# Information asymmetry & Bayesian Nash equilibrium
# ---------------------------------------------------------------------------

class InformationStructure:
    """Model private signals and common knowledge in a Bayesian game.

    Each agent receives a noisy private signal about the true state.
    Common knowledge is the prior distribution of the state.

    Attributes:
        n_agents: Number of agents.
        prior_mean: Prior mean of the state (Normal model).
        prior_var: Prior variance of the state.
        signal_vars: Per-agent signal noise variances.
    """

    def __init__(self, n_agents: int, prior_mean: float = 0.5,
                 prior_var: float = 0.1, signal_vars: np.ndarray | None = None):
        """Set up the information structure.

        Args:
            n_agents: Number of agents.
            prior_mean: Prior mean of the state variable theta.
            prior_var: Prior variance of theta.
            signal_vars: 1-D array of per-agent signal noise variances.
                Defaults to 0.05 for all agents.
        """
        self.n_agents = n_agents
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        if signal_vars is None:
            self.signal_vars = np.full(n_agents, 0.05)
        else:
            self.signal_vars = np.asarray(signal_vars, dtype=np.float64)

    def generate_signals(self, rng: np.random.Generator) -> tuple:
        """Draw a true state and private signals for all agents.

        theta ~ N(prior_mean, prior_var)
        s_i | theta ~ N(theta, signal_var_i)

        Args:
            rng: NumPy random generator.

        Returns:
            (theta, signals) where theta is a float and signals is a
            1-D array of length n_agents.
        """
        theta = rng.normal(self.prior_mean, np.sqrt(self.prior_var))
        signals = np.array([
            rng.normal(theta, np.sqrt(sv)) for sv in self.signal_vars
        ])
        return theta, signals

    def posterior_given_signal(self, agent_idx: int,
                              signal: float) -> tuple:
        """Compute posterior of theta given a single private signal.

        Normal-Normal conjugacy gives:
            posterior precision = 1/prior_var + 1/signal_var
            posterior mean = (prior_mean/prior_var + signal/signal_var) / posterior_precision

        Args:
            agent_idx: Index of the agent.
            signal: Observed private signal.

        Returns:
            (posterior_mean, posterior_var) tuple.
        """
        prior_prec = 1.0 / self.prior_var
        sig_prec = 1.0 / self.signal_vars[agent_idx]
        post_prec = prior_prec + sig_prec
        post_mean = (prior_prec * self.prior_mean + sig_prec * signal) / post_prec
        post_var = 1.0 / post_prec
        return post_mean, post_var

    def compute_bne_linear_strategy(self) -> np.ndarray:
        """Compute the linear Bayesian Nash Equilibrium bidding strategy
        coefficients for a first-price auction with Normal signals.

        In a symmetric linear BNE each agent bids:
            b_i = a + c * s_i
        where the coefficients are derived from the first-order conditions.

        For a first-price auction with n symmetric agents whose valuations
        are their signals (common-value component = 0 for simplicity),
        the symmetric BNE bid function is:
            b(s) = ((n-1)/n) * s

        We generalise to asymmetric signal precisions by computing per-agent
        shading factors based on relative precision.

        Returns:
            Array of shape (n_agents, 2) where row i = [intercept_i, slope_i].
        """
        n = self.n_agents
        if n < 2:
            return np.array([[0.0, 1.0]])

        coefficients = np.zeros((n, 2))
        precisions = 1.0 / self.signal_vars
        total_prec = precisions.sum()

        for i in range(n):
            # Weight of own signal in posterior
            weight_i = precisions[i] / (1.0 / self.prior_var + precisions[i])
            # Shading factor: with more opponents, shade more
            shade = (n - 1.0) / n
            coefficients[i, 0] = (1.0 - shade * weight_i) * self.prior_mean * shade
            coefficients[i, 1] = shade * weight_i

        return coefficients


# ---------------------------------------------------------------------------
# Collusion detection
# ---------------------------------------------------------------------------

class CollusionDetector:
    """Detect coordinated (collusive) bidding patterns among agents.

    Uses several statistical tests:
    1. Pairwise Pearson correlation of bid sequences.
    2. Bid rotation detection via auto-correlation of winner identity.
    3. Variance ratio test: colluding agents suppress bid variance.

    Attributes:
        bid_history: np.ndarray of shape (n_rounds, n_agents).
        winner_history: np.ndarray of shape (n_rounds,).
    """

    def __init__(self):
        """Initialise empty histories."""
        self._bid_history: list = []
        self._winner_history: list = []

    def record(self, bids: np.ndarray, winner: int) -> None:
        """Record a round's bids and winner.

        Args:
            bids: 1-D array of bids, one per agent.
            winner: Index of the winning agent.
        """
        self._bid_history.append(bids.copy())
        self._winner_history.append(winner)

    def _get_arrays(self) -> tuple:
        """Convert lists to arrays.

        Returns:
            (bid_array, winner_array) tuple.
        """
        bids = np.array(self._bid_history)
        winners = np.array(self._winner_history)
        return bids, winners

    def pairwise_bid_correlation(self) -> np.ndarray:
        """Compute pairwise Pearson correlation matrix of bid sequences.

        High positive correlation between two agents can indicate collusion
        (e.g., both raising and lowering bids in tandem).

        Returns:
            Correlation matrix of shape (n_agents, n_agents).
        """
        bids, _ = self._get_arrays()
        if bids.shape[0] < 3:
            n = bids.shape[1] if bids.ndim == 2 else 1
            return np.eye(n)
        return np.corrcoef(bids.T)

    def bid_rotation_score(self) -> float:
        """Detect bid rotation by measuring autocorrelation of winners.

        In a bid rotation scheme the winner identity follows a periodic
        pattern, producing high autocorrelation at the rotation lag.

        We compute the maximum absolute autocorrelation of the winner
        sequence for lags 1..n_agents.

        Returns:
            Maximum absolute autocorrelation (0 = no rotation, 1 = perfect).
        """
        _, winners = self._get_arrays()
        T = len(winners)
        if T < 5:
            return 0.0

        n_agents = int(winners.max()) + 1
        max_lag = min(n_agents, T // 2)

        # Encode winner as one-hot, compute autocorrelation per column
        one_hot = np.zeros((T, n_agents))
        for t, w in enumerate(winners):
            one_hot[t, w] = 1.0

        best_ac = 0.0
        for lag in range(1, max_lag + 1):
            for col in range(n_agents):
                series = one_hot[:, col]
                mean_s = series.mean()
                var_s = np.var(series)
                if var_s < 1e-12:
                    continue
                ac = np.mean((series[:-lag] - mean_s) * (series[lag:] - mean_s)) / var_s
                best_ac = max(best_ac, abs(ac))
        return float(best_ac)

    def variance_ratio_test(self, expected_variance: float | None = None) -> np.ndarray:
        """Test whether agents' bid variance is suspiciously low.

        Under competitive bidding, bid variance should be comparable to
        valuation variance. Collusive agents may suppress their bids to
        a narrow range.

        Returns the ratio actual_var / expected_var for each agent.
        Values << 1 are suspicious.

        Args:
            expected_variance: Expected bid variance under competition.
                If None, use the overall bid variance as a baseline.

        Returns:
            1-D array of variance ratios, one per agent.
        """
        bids, _ = self._get_arrays()
        if bids.shape[0] < 3:
            return np.ones(bids.shape[1])

        agent_vars = np.var(bids, axis=0)
        if expected_variance is None:
            expected_variance = np.var(bids)
        if expected_variance < 1e-15:
            return np.ones(bids.shape[1])
        return agent_vars / expected_variance

    def detect(self, correlation_threshold: float = 0.7,
               rotation_threshold: float = 0.5,
               variance_ratio_threshold: float = 0.3) -> dict:
        """Run all collusion detection tests and return a report.

        Args:
            correlation_threshold: Correlation above this flags a pair.
            rotation_threshold: Rotation score above this flags rotation.
            variance_ratio_threshold: Variance ratio below this flags agent.

        Returns:
            Dict with 'correlated_pairs', 'rotation_detected',
            'low_variance_agents', and 'collusion_suspected' keys.
        """
        corr = self.pairwise_bid_correlation()
        n = corr.shape[0]
        correlated_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr[i, j]) > correlation_threshold:
                    correlated_pairs.append((i, j, float(corr[i, j])))

        rotation = self.bid_rotation_score()
        var_ratios = self.variance_ratio_test()
        low_var_agents = [int(i) for i in range(n) if var_ratios[i] < variance_ratio_threshold]

        suspected = len(correlated_pairs) > 0 or rotation > rotation_threshold or len(low_var_agents) > 0

        return {
            "correlated_pairs": correlated_pairs,
            "rotation_score": rotation,
            "rotation_detected": rotation > rotation_threshold,
            "low_variance_agents": low_var_agents,
            "variance_ratios": var_ratios.tolist(),
            "collusion_suspected": suspected,
        }


# ---------------------------------------------------------------------------
# Welfare analysis
# ---------------------------------------------------------------------------

class WelfareAnalyser:
    """Compute social welfare measures from agent utilities.

    Supports utilitarian (sum), egalitarian (min), and Nash (product)
    social welfare functions.
    """

    @staticmethod
    def utilitarian(utilities: np.ndarray) -> float:
        """Sum of utilities (Benthamite welfare).

        Args:
            utilities: 1-D array of agent utilities.

        Returns:
            Total welfare.
        """
        return float(np.sum(utilities))

    @staticmethod
    def egalitarian(utilities: np.ndarray) -> float:
        """Minimum utility (Rawlsian welfare).

        Args:
            utilities: 1-D array of agent utilities.

        Returns:
            Minimum agent utility.
        """
        return float(np.min(utilities))

    @staticmethod
    def nash(utilities: np.ndarray) -> float:
        """Nash social welfare = product of utilities.

        To avoid numerical issues with many agents, we compute
        exp(sum(log(max(u, epsilon)))) for positive utilities,
        and handle non-positive utilities with a sign-preserving approach.

        Args:
            utilities: 1-D array of agent utilities.

        Returns:
            Nash social welfare.
        """
        eps = 1e-12
        shifted = utilities.copy()
        # If any utility is non-positive, shift all up
        min_u = shifted.min()
        if min_u <= 0:
            shifted = shifted - min_u + eps

        log_sum = np.sum(np.log(shifted))
        # Clamp to avoid overflow
        log_sum = np.clip(log_sum, -300, 300)
        return float(np.exp(log_sum))

    @staticmethod
    def compute_all(utilities: np.ndarray) -> dict:
        """Compute all three welfare measures.

        Args:
            utilities: 1-D array of agent utilities.

        Returns:
            Dict with keys 'utilitarian', 'egalitarian', 'nash'.
        """
        return {
            "utilitarian": WelfareAnalyser.utilitarian(utilities),
            "egalitarian": WelfareAnalyser.egalitarian(utilities),
            "nash": WelfareAnalyser.nash(utilities),
        }


# ---------------------------------------------------------------------------
# Mechanisms
# ---------------------------------------------------------------------------

class FirstPriceAuction:
    """Sealed-bid first-price auction.

    Highest bidder wins and pays their bid.

    Attributes:
        reserve_price: Minimum acceptable bid.
    """

    def __init__(self, reserve_price: float = 0.0):
        """Initialise.

        Args:
            reserve_price: Minimum bid to win.
        """
        self.reserve_price = reserve_price

    def __call__(self, bids: np.ndarray) -> tuple:
        """Run the auction.

        Args:
            bids: 1-D array of bids, one per agent.

        Returns:
            (allocations, payments) — allocations[i] = 1 if agent i wins,
            payments[i] = bid_i if agent i wins, else 0.
        """
        n = len(bids)
        allocations = np.zeros(n)
        payments = np.zeros(n)

        valid = bids >= self.reserve_price
        if not valid.any():
            return allocations, payments

        masked_bids = np.where(valid, bids, -np.inf)
        winner = int(np.argmax(masked_bids))
        allocations[winner] = 1.0
        payments[winner] = bids[winner]
        return allocations, payments


class SecondPriceAuction:
    """Sealed-bid second-price (Vickrey) auction.

    Highest bidder wins and pays the second-highest bid.

    Attributes:
        reserve_price: Minimum acceptable bid.
    """

    def __init__(self, reserve_price: float = 0.0):
        """Initialise.

        Args:
            reserve_price: Minimum bid to win.
        """
        self.reserve_price = reserve_price

    def __call__(self, bids: np.ndarray) -> tuple:
        """Run the auction.

        Args:
            bids: 1-D array of bids, one per agent.

        Returns:
            (allocations, payments) tuple.
        """
        n = len(bids)
        allocations = np.zeros(n)
        payments = np.zeros(n)

        valid = bids >= self.reserve_price
        if not valid.any():
            return allocations, payments

        masked_bids = np.where(valid, bids, -np.inf)
        sorted_indices = np.argsort(masked_bids)[::-1]
        winner = sorted_indices[0]
        allocations[winner] = 1.0

        if n >= 2:
            second_price = max(masked_bids[sorted_indices[1]], self.reserve_price)
        else:
            second_price = self.reserve_price
        payments[winner] = second_price
        return allocations, payments


# ---------------------------------------------------------------------------
# Revenue curve
# ---------------------------------------------------------------------------

class RevenueCurveAnalyser:
    """Analyse how mechanism revenue varies with the number of agents.

    Runs independent simulations for different agent counts and records
    mean revenue.

    Attributes:
        mechanism_class: Callable that creates a mechanism instance.
        valuation_dist: Tuple (loc, scale) for the valuation distribution.
    """

    def __init__(self, mechanism_class, valuation_dist: tuple = (0.5, 0.2),
                 mechanism_kwargs: dict | None = None):
        """Initialise.

        Args:
            mechanism_class: A callable (e.g. FirstPriceAuction) producing
                a mechanism when called.
            valuation_dist: (mean, std) for drawing agent valuations from
                a truncated Normal on [0, 1].
            mechanism_kwargs: Extra keyword arguments for the mechanism.
        """
        self.mechanism_class = mechanism_class
        self.valuation_dist = valuation_dist
        self.mechanism_kwargs = mechanism_kwargs or {}

    def compute_curve(self, agent_counts: list, rounds_per_count: int = 500,
                      seed: int = 42) -> tuple:
        """Compute mean and std revenue for each agent count.

        For each count, draw independent valuations each round and run
        a truthful-bidding simulation (agents bid their valuations).

        Args:
            agent_counts: List of integers, each an agent count to test.
            rounds_per_count: Number of rounds per agent count.
            seed: Random seed.

        Returns:
            (agent_counts, mean_revenues, std_revenues) — each a list.
        """
        rng = np.random.default_rng(seed)
        mechanism = self.mechanism_class(**self.mechanism_kwargs)
        mean_revs = []
        std_revs = []
        mu, sigma = self.valuation_dist

        for n in agent_counts:
            revenues = np.zeros(rounds_per_count)
            for r in range(rounds_per_count):
                vals = np.clip(rng.normal(mu, sigma, size=n), 0.0, 1.0)
                bids = vals  # truthful bidding baseline
                _, payments = mechanism(bids)
                revenues[r] = payments.sum()
            mean_revs.append(float(revenues.mean()))
            std_revs.append(float(revenues.std()))

        return agent_counts, mean_revs, std_revs

    def expected_revenue_formula(self, n: int) -> float:
        """Closed-form expected revenue for a second-price auction with
        n bidders whose values are i.i.d. Uniform[0, 1].

        E[Revenue] = E[second-highest order statistic] = (n-1)/(n+1).

        Args:
            n: Number of bidders.

        Returns:
            Expected revenue.
        """
        if n < 2:
            return 0.0
        return (n - 1.0) / (n + 1.0)


# ---------------------------------------------------------------------------
# MultiAgentSimulator
# ---------------------------------------------------------------------------

class MultiAgentSimulator:
    """Orchestrates multi-round simulations of strategic agents in a mechanism.

    Supports pluggable mechanisms, Exp3 regret minimisation or best-response
    dynamics for strategy updates, Bayesian belief updating, private signals,
    collusion detection, and welfare tracking.

    Attributes:
        agents: List of Agent objects.
        rng: NumPy random generator.
        collusion_detector: CollusionDetector instance.
        welfare_analyser: WelfareAnalyser instance.
        info_structure: InformationStructure or None.
    """

    def __init__(self, seed: int = 0):
        """Initialise the simulator.

        Args:
            seed: Random seed for reproducibility.
        """
        self.agents: list[Agent] = []
        self.rng = np.random.default_rng(seed)
        self.collusion_detector = CollusionDetector()
        self.welfare_analyser = WelfareAnalyser()
        self.info_structure: InformationStructure | None = None
        self._exp3_engines: dict[int, Exp3Algorithm] = {}
        self._br_engines: dict[int, BestResponseDynamics] = {}

    def add_agents(self, agents: list) -> None:
        """Register a list of agents for the simulation.

        Initialises beliefs, Exp3 engines, and best-response engines for
        each agent.

        Args:
            agents: List of Agent instances.
        """
        self.agents = list(agents)
        n = len(self.agents)

        # Set up beliefs and strategy engines
        for ag in self.agents:
            opponent_ids = [a.agent_id for a in self.agents if a.agent_id != ag.agent_id]
            ag.init_beliefs(opponent_ids, belief_type="beta")
            self._exp3_engines[ag.agent_id] = Exp3Algorithm(ag.n_actions, gamma=0.1)
            self._br_engines[ag.agent_id] = BestResponseDynamics(ag.action_space, ag.valuation)

        # Information structure
        self.info_structure = InformationStructure(n)

    def _update_strategy_exp3(self, agent: Agent, action_idx: int,
                              reward: float) -> None:
        """Update an agent's strategy using Exp3.

        Args:
            agent: The agent.
            action_idx: Index of the action taken.
            reward: Observed reward (utility), rescaled to [0, 1].
        """
        engine = self._exp3_engines[agent.agent_id]
        # Rescale reward to [0, 1]
        r_scaled = np.clip(reward, 0.0, 1.0)
        engine.update(action_idx, r_scaled)
        agent.strategy = engine.get_distribution()

    def _update_strategy_best_response(self, agent: Agent,
                                       mechanism_fn) -> None:
        """Update an agent's strategy via best-response dynamics.

        Args:
            agent: The agent.
            mechanism_fn: The mechanism callable.
        """
        engine = self._br_engines[agent.agent_id]
        opponent_strategies = []
        for other in self.agents:
            if other.agent_id != agent.agent_id:
                opponent_strategies.append((other.action_space, other.strategy))

        strategy = engine.compute_epsilon_best_response(
            opponent_strategies, mechanism_fn, self.rng, epsilon=0.05, n_samples=50
        )
        agent.strategy = strategy

    def run(self, mechanism, rounds: int = 100,
            strategy_update: str = "exp3",
            convergence_tol: float = 1e-4,
            use_signals: bool = True) -> SimulationResult:
        """Run a multi-round simulation.

        In each round:
        1. (Optional) Generate private signals and update agent beliefs.
        2. Each agent selects an action from their current strategy.
        3. The mechanism determines allocations and payments.
        4. Utilities are computed and strategies are updated.
        5. Collusion detection and welfare metrics are recorded.

        Args:
            mechanism: Callable (bids) -> (allocations, payments).
            rounds: Number of rounds to simulate.
            strategy_update: 'exp3' for Exp3 regret minimisation, or
                'best_response' for best-response dynamics.
            convergence_tol: If the max strategy change norm falls below
                this threshold, declare convergence.
            use_signals: Whether to use the information structure for
                private signals.

        Returns:
            SimulationResult with full simulation data.
        """
        n_agents = len(self.agents)
        result = SimulationResult(rounds, n_agents)
        prev_strategies = [ag.strategy.copy() for ag in self.agents]

        for t in range(rounds):
            # --- Step 1: Private signals ---
            if use_signals and self.info_structure is not None:
                theta, signals = self.info_structure.generate_signals(self.rng)
                for i, ag in enumerate(self.agents):
                    ag.receive_signal(signals[i])
                    # Use signal to shade bid toward posterior mean
                    post_mean, _ = self.info_structure.posterior_given_signal(i, signals[i])
                    # Shift strategy toward actions near posterior mean
                    diffs = np.abs(ag.action_space - post_mean)
                    signal_weights = np.exp(-5.0 * diffs)
                    signal_weights /= signal_weights.sum()
                    # Mix signal-based strategy with current strategy
                    ag.strategy = 0.8 * ag.strategy + 0.2 * signal_weights

            # --- Step 2: Action selection ---
            actions = np.zeros(n_agents)
            action_indices = np.zeros(n_agents, dtype=int)
            for i, ag in enumerate(self.agents):
                feasible = ag._budget_feasible_mask()
                probs = ag.strategy * feasible
                total = probs.sum()
                if total < 1e-15:
                    action_indices[i] = 0
                else:
                    probs /= total
                    action_indices[i] = self.rng.choice(ag.n_actions, p=probs)
                actions[i] = ag.action_space[action_indices[i]]

            # --- Step 3: Mechanism ---
            allocations, payments = mechanism(actions)
            revenue = float(payments.sum())
            result.revenue[t] = revenue

            # Determine winner for collusion detection
            winner = int(np.argmax(allocations)) if allocations.max() > 0 else 0
            self.collusion_detector.record(actions, winner)

            # --- Step 4: Utilities and strategy update ---
            round_utilities = np.zeros(n_agents)
            round_outcome = {}
            for i, ag in enumerate(self.agents):
                alloc_value = allocations[i] * ag.valuation
                util = ag.receive_outcome(alloc_value, payments[i])
                round_utilities[i] = util
                round_outcome[i] = {
                    "action": float(actions[i]),
                    "allocation": float(allocations[i]),
                    "payment": float(payments[i]),
                    "utility": float(util),
                }

                # Belief update: observe whether each opponent won
                obs = {}
                for j, other in enumerate(self.agents):
                    if other.agent_id != ag.agent_id:
                        obs[other.agent_id] = int(allocations[j] > 0)
                ag.update_beliefs(obs)

                # Strategy update
                if strategy_update == "exp3":
                    self._update_strategy_exp3(ag, action_indices[i], util)
                elif strategy_update == "best_response" and t % 10 == 0:
                    self._update_strategy_best_response(ag, mechanism)

            result.outcomes_per_round.append(round_outcome)
            result.agent_utilities[t] = round_utilities

            # --- Step 5: Welfare ---
            welfare = self.welfare_analyser.compute_all(round_utilities)
            result.welfare_metrics["utilitarian"][t] = welfare["utilitarian"]
            result.welfare_metrics["egalitarian"][t] = welfare["egalitarian"]
            result.welfare_metrics["nash"][t] = welfare["nash"]

            # --- Step 6: Convergence check ---
            max_dist = 0.0
            for i, ag in enumerate(self.agents):
                dist = np.linalg.norm(ag.strategy - prev_strategies[i])
                max_dist = max(max_dist, dist)
                prev_strategies[i] = ag.strategy.copy()
            result.convergence_info["strategy_distances"][t] = max_dist

            if t > 10 and max_dist < convergence_tol:
                if not result.convergence_info["converged"]:
                    result.convergence_info["converged"] = True
                    result.convergence_info["convergence_round"] = t

        return result

    def run_revenue_curve(self, mechanism_class, agent_counts: list,
                          rounds_per_count: int = 200) -> tuple:
        """Convenience method to compute a revenue-vs-agents curve.

        Args:
            mechanism_class: Callable producing a mechanism.
            agent_counts: List of agent counts.
            rounds_per_count: Rounds per count.

        Returns:
            (counts, mean_revenues, std_revenues) tuple.
        """
        analyser = RevenueCurveAnalyser(mechanism_class)
        return analyser.compute_curve(agent_counts, rounds_per_count,
                                      seed=self.rng.integers(0, 2**31))

    def get_collusion_report(self) -> dict:
        """Run collusion detection on recorded bid history.

        Returns:
            Collusion detection report dict.
        """
        return self.collusion_detector.detect()

    def compute_bne_strategies(self) -> np.ndarray | None:
        """Compute Bayesian Nash equilibrium linear strategy coefficients.

        Returns:
            Array of shape (n_agents, 2) with [intercept, slope] per agent,
            or None if no information structure is set.
        """
        if self.info_structure is None:
            return None
        return self.info_structure.compute_bne_linear_strategy()


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_example_simulation(n_agents: int = 5, rounds: int = 100,
                           seed: int = 42) -> SimulationResult:
    """Run a complete example simulation with default settings.

    Creates agents with random valuations, runs a first-price auction
    with Exp3 strategy updates, and returns the result.

    Args:
        n_agents: Number of agents.
        rounds: Number of rounds.
        seed: Random seed.

    Returns:
        SimulationResult object.
    """
    rng = np.random.default_rng(seed)
    agents = []
    for i in range(n_agents):
        val = rng.uniform(0.2, 0.9)
        budget = rng.uniform(5.0, 20.0)
        ag = Agent(agent_id=i, valuation=val, budget=budget, n_actions=20)
        agents.append(ag)

    sim = MultiAgentSimulator(seed=seed)
    sim.add_agents(agents)

    mechanism = FirstPriceAuction(reserve_price=0.05)
    result = sim.run(mechanism, rounds=rounds, strategy_update="exp3",
                     use_signals=True)
    return result


def compare_mechanisms(n_agents: int = 5, rounds: int = 100,
                       seed: int = 42) -> dict:
    """Compare first-price and second-price auctions on key metrics.

    Args:
        n_agents: Number of agents.
        rounds: Number of simulation rounds.
        seed: Random seed.

    Returns:
        Dictionary with results for each mechanism.
    """
    results = {}
    for name, mech_class in [("first_price", FirstPriceAuction),
                              ("second_price", SecondPriceAuction)]:
        rng = np.random.default_rng(seed)
        agents = []
        for i in range(n_agents):
            val = rng.uniform(0.2, 0.9)
            budget = rng.uniform(5.0, 20.0)
            agents.append(Agent(agent_id=i, valuation=val, budget=budget))

        sim = MultiAgentSimulator(seed=seed)
        sim.add_agents(agents)
        mechanism = mech_class(reserve_price=0.05)
        res = sim.run(mechanism, rounds=rounds, strategy_update="exp3")
        results[name] = res.summary()
        results[name]["collusion_report"] = sim.get_collusion_report()

    return results


if __name__ == "__main__":
    print("Running example simulation...")
    result = run_example_simulation(n_agents=5, rounds=50, seed=123)
    s = result.summary()
    for k, v in s.items():
        print(f"  {k}: {v}")

    print("\nComparing mechanisms...")
    comparison = compare_mechanisms(n_agents=4, rounds=50, seed=456)
    for mech_name, metrics in comparison.items():
        print(f"\n  {mech_name}:")
        for k, v in metrics.items():
            if k != "collusion_report":
                print(f"    {k}: {v}")
        cr = metrics["collusion_report"]
        print(f"    collusion_suspected: {cr['collusion_suspected']}")
        print(f"    rotation_score: {cr['rotation_score']:.4f}")
