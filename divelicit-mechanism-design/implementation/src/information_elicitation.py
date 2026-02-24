"""
Truthful information gathering: scoring rules, peer prediction,
prediction markets, Delphi method, and forecast evaluation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict


class ProperScoringRule:
    """
    Proper scoring rules for eliciting truthful probability forecasts.
    A scoring rule is proper if the expected score is maximized when the
    forecaster reports their true belief.
    """

    @staticmethod
    def brier_score(forecast: np.ndarray, outcome: int) -> float:
        """
        Brier score (quadratic scoring rule).
        S(p, i) = 2*p_i - ||p||^2
        Proper: maximized at true belief.
        """
        n = len(forecast)
        indicator = np.zeros(n)
        indicator[outcome] = 1.0
        return 2.0 * forecast[outcome] - np.sum(forecast ** 2)

    @staticmethod
    def brier_score_loss(forecast: np.ndarray, outcome: int) -> float:
        """Brier score as loss (lower is better): sum (p_i - indicator_i)^2."""
        n = len(forecast)
        indicator = np.zeros(n)
        indicator[outcome] = 1.0
        return np.sum((forecast - indicator) ** 2)

    @staticmethod
    def logarithmic_score(forecast: np.ndarray, outcome: int) -> float:
        """
        Logarithmic scoring rule.
        S(p, i) = log(p_i)
        Strictly proper. Maximized at true distribution.
        """
        p = max(forecast[outcome], 1e-15)
        return np.log(p)

    @staticmethod
    def spherical_score(forecast: np.ndarray, outcome: int) -> float:
        """
        Spherical scoring rule.
        S(p, i) = p_i / ||p||
        Proper scoring rule.
        """
        norm = np.sqrt(np.sum(forecast ** 2))
        if norm < 1e-15:
            return 0.0
        return forecast[outcome] / norm

    @staticmethod
    def verify_properness(scoring_func: Callable, n_outcomes: int = 3,
                          n_test_beliefs: int = 100, n_reports: int = 50) -> Tuple[bool, float]:
        """
        Verify properness: for random beliefs q, check that reporting q
        maximizes expected score under q.
        """
        rng = np.random.default_rng(42)
        max_violation = 0.0

        for _ in range(n_test_beliefs):
            # Random true belief
            q = rng.dirichlet(np.ones(n_outcomes))

            # Expected score under truthful reporting
            true_expected = sum(q[i] * scoring_func(q, i) for i in range(n_outcomes))

            # Try random misreports
            for _ in range(n_reports):
                p = rng.dirichlet(np.ones(n_outcomes))
                misreport_expected = sum(q[i] * scoring_func(p, i) for i in range(n_outcomes))

                violation = misreport_expected - true_expected
                if violation > max_violation:
                    max_violation = violation

        is_proper = max_violation < 1e-8
        return is_proper, max_violation

    @staticmethod
    def expected_score(scoring_func: Callable, belief: np.ndarray,
                       report: np.ndarray) -> float:
        """Compute expected score when true belief is `belief` and report is `report`."""
        n = len(belief)
        return sum(belief[i] * scoring_func(report, i) for i in range(n))


class PeerPrediction:
    """
    Peer prediction mechanisms for eliciting information without verification.
    """

    @staticmethod
    def bayesian_truth_serum(reports: List[int], n_outcomes: int) -> np.ndarray:
        """
        Bayesian Truth Serum (Prelec 2004).
        Score = log(x_k) - log(y_k) for each respondent reporting k.
        x_k = empirical frequency of answer k
        y_k = geometric mean of predicted frequencies for k
        
        Returns scores for each respondent.
        """
        n = len(reports)
        # Empirical frequencies
        freq = np.zeros(n_outcomes)
        for r in reports:
            freq[r] += 1
        freq /= n

        # Information score: log(freq[report]) + 1 for being "surprisingly popular"
        scores = np.zeros(n)
        for i, r in enumerate(reports):
            # Information score
            if freq[r] > 0:
                scores[i] = np.log(max(freq[r], 1e-15))
            # Bonus for answers that are more common than expected
            # Simple version: bonus proportional to frequency
            scores[i] += freq[r]

        return scores

    @staticmethod
    def surprisingly_popular(reports: List[int], predictions: List[np.ndarray],
                              n_outcomes: int) -> int:
        """
        Surprisingly popular algorithm.
        Each agent reports their answer AND their prediction of others' answers.
        The "surprisingly popular" answer is the one whose actual frequency
        most exceeds its predicted frequency.

        reports: list of answers (integers)
        predictions: list of predicted frequency distributions
        Returns: the surprisingly popular answer
        """
        n = len(reports)

        # Actual frequencies
        actual_freq = np.zeros(n_outcomes)
        for r in reports:
            actual_freq[r] += 1
        actual_freq /= n

        # Average predicted frequencies
        avg_predicted = np.zeros(n_outcomes)
        for pred in predictions:
            avg_predicted += pred
        avg_predicted /= n

        # Surprisingly popular: largest positive difference
        surprise = actual_freq - avg_predicted
        return int(np.argmax(surprise))

    @staticmethod
    def peer_prediction_no_common_prior(reports: List[int], n_outcomes: int,
                                         payment_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Peer prediction without common prior (Witkowski & Parkes 2012).
        Pairs agents and rewards agreement.
        Uses shadow posterior mechanism.

        Returns: payment for each agent
        """
        n = len(reports)
        if payment_matrix is None:
            # Default: identity-based (reward exact match)
            payment_matrix = np.eye(n_outcomes)

        payments = np.zeros(n)
        for i in range(n):
            # Pair with random other agent
            j = (i + 1) % n
            payments[i] = payment_matrix[reports[i], reports[j]]

        return payments


@dataclass
class MarketState:
    """State of a prediction market."""
    probabilities: np.ndarray
    cost: float
    shares_outstanding: np.ndarray


class PredictionMarket:
    """
    Logarithmic Market Scoring Rule (LMSR) prediction market.
    Hanson's automated market maker.
    """

    def __init__(self, n_outcomes: int, liquidity: float = 100.0):
        self.n_outcomes = n_outcomes
        self.liquidity = liquidity  # b parameter
        self.shares = np.zeros(n_outcomes)
        self.portfolios: Dict[int, np.ndarray] = {}
        self.trade_history: List[Dict] = []

    def cost_function(self, shares: np.ndarray) -> float:
        """LMSR cost function: C(q) = b * log(sum(exp(q_i/b)))."""
        scaled = shares / self.liquidity
        max_scaled = np.max(scaled)
        return self.liquidity * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))

    def prices(self) -> np.ndarray:
        """
        Current prices (instantaneous cost of buying one share).
        p_i = exp(q_i/b) / sum(exp(q_j/b))
        """
        scaled = self.shares / self.liquidity
        max_scaled = np.max(scaled)
        exp_scaled = np.exp(scaled - max_scaled)
        return exp_scaled / np.sum(exp_scaled)

    def buy(self, trader_id: int, outcome: int, quantity: float) -> float:
        """
        Buy `quantity` shares of outcome.
        Returns the cost paid.
        """
        old_cost = self.cost_function(self.shares)
        new_shares = self.shares.copy()
        new_shares[outcome] += quantity
        new_cost = self.cost_function(new_shares)
        cost = new_cost - old_cost

        self.shares = new_shares

        if trader_id not in self.portfolios:
            self.portfolios[trader_id] = np.zeros(self.n_outcomes)
        self.portfolios[trader_id][outcome] += quantity

        self.trade_history.append({
            'trader': trader_id,
            'outcome': outcome,
            'quantity': quantity,
            'cost': cost,
            'prices_after': self.prices().tolist()
        })

        return cost

    def sell(self, trader_id: int, outcome: int, quantity: float) -> float:
        """Sell shares (negative buy). Returns revenue received."""
        return -self.buy(trader_id, outcome, -quantity)

    def get_state(self) -> MarketState:
        return MarketState(
            probabilities=self.prices(),
            cost=self.cost_function(self.shares),
            shares_outstanding=self.shares.copy()
        )

    def trader_pnl(self, trader_id: int, true_outcome: int) -> float:
        """Compute profit/loss for a trader given the true outcome."""
        if trader_id not in self.portfolios:
            return 0.0
        # Shares pay 1 if correct, 0 otherwise
        portfolio = self.portfolios[trader_id]
        payout = portfolio[true_outcome]

        # Total cost paid
        total_cost = sum(
            t['cost'] for t in self.trade_history if t['trader'] == trader_id
        )

        return payout - total_cost


class DelphinMethod:
    """
    Iterative Delphi method with Bayesian updating and convergence detection.
    """

    def __init__(self, n_experts: int, n_rounds: int = 10):
        self.n_experts = n_experts
        self.n_rounds = n_rounds
        self.rounds: List[np.ndarray] = []

    def run(self, initial_estimates: np.ndarray,
            confidence_weights: Optional[np.ndarray] = None,
            convergence_threshold: float = 0.01) -> Dict:
        """
        Run Delphi process with Bayesian updating.
        initial_estimates: (n_experts,) initial point estimates
        Returns dict with final estimates, convergence info.
        """
        if confidence_weights is None:
            confidence_weights = np.ones(self.n_experts) / self.n_experts

        current_estimates = initial_estimates.copy()
        self.rounds = [current_estimates.copy()]

        for round_num in range(self.n_rounds):
            # Compute weighted aggregate
            aggregate = np.average(current_estimates, weights=confidence_weights)
            spread = np.std(current_estimates)

            # Bayesian update: each expert updates toward aggregate
            # Weight of update depends on confidence
            prior_precision = confidence_weights * 10  # Higher confidence = less update
            data_precision = 1.0 / max(spread ** 2, 1e-10)

            new_estimates = np.zeros(self.n_experts)
            for i in range(self.n_experts):
                # Posterior mean: weighted average of prior (own estimate) and data (aggregate)
                total_precision = prior_precision[i] + data_precision
                new_estimates[i] = (prior_precision[i] * current_estimates[i] +
                                    data_precision * aggregate) / total_precision

            # Update confidence based on proximity to aggregate
            distances = np.abs(new_estimates - aggregate)
            max_dist = np.max(distances) if np.max(distances) > 0 else 1.0
            confidence_weights = 1.0 - distances / (max_dist + 1e-10)
            confidence_weights /= confidence_weights.sum()

            # Check convergence
            change = np.max(np.abs(new_estimates - current_estimates))
            current_estimates = new_estimates
            self.rounds.append(current_estimates.copy())

            if change < convergence_threshold:
                return {
                    'final_estimates': current_estimates,
                    'aggregate': np.average(current_estimates, weights=confidence_weights),
                    'spread': np.std(current_estimates),
                    'converged': True,
                    'rounds': round_num + 1,
                    'confidence_weights': confidence_weights,
                    'history': [r.tolist() for r in self.rounds]
                }

        return {
            'final_estimates': current_estimates,
            'aggregate': np.average(current_estimates, weights=confidence_weights),
            'spread': np.std(current_estimates),
            'converged': False,
            'rounds': self.n_rounds,
            'confidence_weights': confidence_weights,
            'history': [r.tolist() for r in self.rounds]
        }


# Information aggregation methods

def linear_opinion_pool(forecasts: List[np.ndarray],
                         weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Linear opinion pool: weighted average of probability forecasts.
    p(x) = sum w_i * p_i(x)
    """
    n = len(forecasts)
    if weights is None:
        weights = np.ones(n) / n

    result = np.zeros_like(forecasts[0])
    for i, f in enumerate(forecasts):
        result += weights[i] * f

    # Normalize
    total = result.sum()
    if total > 0:
        result /= total
    return result


def logarithmic_opinion_pool(forecasts: List[np.ndarray],
                               weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Logarithmic opinion pool: geometric weighted average.
    p(x) proportional to prod p_i(x)^{w_i}
    """
    n = len(forecasts)
    if weights is None:
        weights = np.ones(n) / n

    log_result = np.zeros_like(forecasts[0])
    for i, f in enumerate(forecasts):
        log_result += weights[i] * np.log(np.maximum(f, 1e-15))

    result = np.exp(log_result)
    total = result.sum()
    if total > 0:
        result /= total
    return result


def extremized_aggregation(forecasts: List[np.ndarray],
                            alpha: float = 2.5,
                            weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Extremized aggregation: push aggregate toward extreme probabilities.
    Used in intelligence forecasting (e.g., IARPA ACE program).
    p(x) proportional to (linear_pool(x))^alpha
    """
    linear = linear_opinion_pool(forecasts, weights)
    extremized = linear ** alpha
    total = extremized.sum()
    if total > 0:
        extremized /= total
    return extremized


# Forecast evaluation

def calibration_score(forecasts: List[np.ndarray], outcomes: List[int],
                       n_bins: int = 10) -> Tuple[float, List[Dict]]:
    """
    Compute calibration score and reliability diagram data.
    Groups forecasts into bins by predicted probability,
    compares average prediction to observed frequency.
    """
    n = len(forecasts)
    n_outcomes = forecasts[0].shape[0]

    # For binary or multiclass, analyze the predicted probability for each outcome
    bin_data = []
    total_cal_error = 0.0
    count = 0

    for outcome_idx in range(n_outcomes):
        bins = [{'sum_pred': 0.0, 'sum_actual': 0.0, 'count': 0}
                for _ in range(n_bins)]

        for i in range(n):
            pred = forecasts[i][outcome_idx]
            actual = 1.0 if outcomes[i] == outcome_idx else 0.0
            bin_idx = min(int(pred * n_bins), n_bins - 1)
            bins[bin_idx]['sum_pred'] += pred
            bins[bin_idx]['sum_actual'] += actual
            bins[bin_idx]['count'] += 1

        for b in bins:
            if b['count'] > 0:
                avg_pred = b['sum_pred'] / b['count']
                avg_actual = b['sum_actual'] / b['count']
                cal_error = (avg_pred - avg_actual) ** 2 * b['count']
                total_cal_error += cal_error
                count += b['count']
                bin_data.append({
                    'outcome': outcome_idx,
                    'avg_predicted': avg_pred,
                    'avg_observed': avg_actual,
                    'count': b['count']
                })

    cal_score = total_cal_error / max(count, 1)
    return cal_score, bin_data


def brier_skill_score(forecasts: List[np.ndarray], outcomes: List[int],
                       baseline: Optional[np.ndarray] = None) -> float:
    """
    Brier Skill Score: improvement over baseline (climatological) forecast.
    BSS = 1 - BS / BS_baseline
    """
    n = len(forecasts)
    n_outcomes = forecasts[0].shape[0]

    if baseline is None:
        # Climatological baseline: empirical frequency
        baseline = np.zeros(n_outcomes)
        for o in outcomes:
            baseline[o] += 1
        baseline /= n

    # Brier score of forecasts
    bs = 0.0
    for i in range(n):
        indicator = np.zeros(n_outcomes)
        indicator[outcomes[i]] = 1.0
        bs += np.sum((forecasts[i] - indicator) ** 2)
    bs /= n

    # Brier score of baseline
    bs_base = 0.0
    for i in range(n):
        indicator = np.zeros(n_outcomes)
        indicator[outcomes[i]] = 1.0
        bs_base += np.sum((baseline - indicator) ** 2)
    bs_base /= n

    if bs_base < 1e-15:
        return 0.0
    return 1.0 - bs / bs_base


def brier_decomposition(forecasts: List[np.ndarray], outcomes: List[int],
                          n_bins: int = 10) -> Dict[str, float]:
    """
    Decompose Brier score into reliability, resolution, and uncertainty.
    BS = REL - RES + UNC
    """
    n = len(forecasts)
    n_outcomes = forecasts[0].shape[0]

    # Overall Brier score
    bs = 0.0
    for i in range(n):
        indicator = np.zeros(n_outcomes)
        indicator[outcomes[i]] = 1.0
        bs += np.sum((forecasts[i] - indicator) ** 2)
    bs /= n

    # Climatological frequencies
    clim = np.zeros(n_outcomes)
    for o in outcomes:
        clim[o] += 1
    clim /= n

    # Uncertainty
    unc = np.sum(clim * (1 - clim))

    # Reliability and resolution (for binary, generalized for multiclass)
    # Group by predicted probability
    rel = 0.0
    res = 0.0

    for outcome_idx in range(n_outcomes):
        bins = defaultdict(lambda: {'sum_pred': 0.0, 'sum_actual': 0.0, 'count': 0})
        for i in range(n):
            pred = forecasts[i][outcome_idx]
            actual = 1.0 if outcomes[i] == outcome_idx else 0.0
            bin_idx = min(int(pred * n_bins), n_bins - 1)
            bins[bin_idx]['sum_pred'] += pred
            bins[bin_idx]['sum_actual'] += actual
            bins[bin_idx]['count'] += 1

        for b in bins.values():
            if b['count'] > 0:
                avg_pred = b['sum_pred'] / b['count']
                avg_actual = b['sum_actual'] / b['count']
                rel += b['count'] * (avg_pred - avg_actual) ** 2
                res += b['count'] * (avg_actual - clim[outcome_idx]) ** 2

    rel /= (n * n_outcomes)
    res /= (n * n_outcomes)

    return {
        'brier_score': bs,
        'reliability': rel,
        'resolution': res,
        'uncertainty': unc
    }


def simulate_prediction_market(n_traders: int, n_outcomes: int,
                                 true_probs: np.ndarray,
                                 n_rounds: int = 100,
                                 liquidity: float = 50.0,
                                 seed: int = 42) -> Dict:
    """
    Simulate a prediction market with informed and noise traders.
    Returns convergence metrics.
    """
    rng = np.random.default_rng(seed)
    market = PredictionMarket(n_outcomes, liquidity)

    price_history = [market.prices().tolist()]

    for round_num in range(n_rounds):
        trader_id = rng.integers(0, n_traders)

        # Mix of informed and noise trading
        is_informed = rng.random() < 0.6

        if is_informed:
            # Trade toward true probability
            current_prices = market.prices()
            # Find most mispriced outcome
            mispricings = true_probs - current_prices
            outcome = int(np.argmax(np.abs(mispricings)))
            quantity = mispricings[outcome] * liquidity * 0.1
        else:
            # Noise trade
            outcome = rng.integers(0, n_outcomes)
            quantity = rng.normal(0, 1)

        if abs(quantity) > 0.01:
            market.buy(trader_id, outcome, quantity)

        price_history.append(market.prices().tolist())

    final_prices = market.prices()
    price_error = np.sqrt(np.mean((final_prices - true_probs) ** 2))

    return {
        'final_prices': final_prices.tolist(),
        'true_probs': true_probs.tolist(),
        'price_error': price_error,
        'price_history': price_history,
        'n_trades': len(market.trade_history),
        'converged': price_error < 0.05
    }
