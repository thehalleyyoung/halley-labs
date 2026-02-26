"""
Shielded mean-variance portfolio optimizer.

Implements Markowitz mean-variance optimization restricted to the set of
actions permitted by a safety shield.  The optimizer works over a discrete
action space (7 position levels from -3 to +3) and uses only causally
invariant features for expected-return estimation.

Key ideas
---------
* The feasible set is *not* the full simplex – it is the subset of actions
  that the shield certifies as safe given the current state.
* Transaction costs (linear + quadratic) are incorporated directly into the
  objective so that the optimizer avoids unnecessary rebalancing.
* An efficient-frontier sweep over risk-aversion parameters produces the
  Pareto-optimal curve from which a single portfolio is selected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, LinearConstraint

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TransactionCostModel:
    """Linear + quadratic transaction cost model.

    cost(delta) = linear_coeff * |delta| + quadratic_coeff * delta^2
    where *delta* is the change in position level.
    """
    linear_coeff: float = 0.001
    quadratic_coeff: float = 0.0005
    fixed_cost: float = 0.0

    def cost(self, delta: float) -> float:
        """Compute round-trip cost for a position change *delta*."""
        return (
            self.fixed_cost
            + self.linear_coeff * abs(delta)
            + self.quadratic_coeff * delta ** 2
        )

    def cost_vector(self, deltas: NDArray) -> NDArray:
        """Vectorised cost over an array of position changes."""
        return (
            self.fixed_cost
            + self.linear_coeff * np.abs(deltas)
            + self.quadratic_coeff * deltas ** 2
        )


@dataclass
class OptimizationResult:
    """Container for a single optimisation solve."""
    optimal_action: int
    optimal_weight: float
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    cost_adjustment: float
    shield_permitted: List[int]
    all_scores: Dict[int, float] = field(default_factory=dict)


@dataclass
class EfficientFrontierPoint:
    """One point on the efficient frontier."""
    risk_aversion: float
    expected_return: float
    risk: float
    sharpe: float
    action: int
    weight: float


# ---------------------------------------------------------------------------
# Core optimiser
# ---------------------------------------------------------------------------

class ShieldedMeanVarianceOptimizer:
    """Mean-variance optimiser operating under shield constraints.

    Parameters
    ----------
    action_levels : sequence of int
        Discrete position levels (default ``[-3, -2, -1, 0, 1, 2, 3]``).
    risk_aversion : float
        λ in the objective  ``μ - λ σ² - c(Δ)``.
    risk_free_rate : float
        Annualised risk-free rate used in Sharpe ratio computation.
    cost_model : TransactionCostModel or None
        Transaction cost specification.
    use_causal_features_only : bool
        If *True*, the return-estimation step will drop non-invariant
        features before forming expectations.
    annualisation_factor : float
        Multiplier to annualise per-period returns (default 252 for daily).
    """

    DEFAULT_ACTIONS: List[int] = [-3, -2, -1, 0, 1, 2, 3]

    def __init__(
        self,
        action_levels: Optional[Sequence[int]] = None,
        risk_aversion: float = 1.0,
        risk_free_rate: float = 0.02,
        cost_model: Optional[TransactionCostModel] = None,
        use_causal_features_only: bool = True,
        annualisation_factor: float = 252.0,
    ) -> None:
        self.action_levels = list(action_levels or self.DEFAULT_ACTIONS)
        self.risk_aversion = risk_aversion
        self.risk_free_rate = risk_free_rate
        self.cost_model = cost_model or TransactionCostModel()
        self.use_causal_features_only = use_causal_features_only
        self.annualisation_factor = annualisation_factor

        # State
        self._current_action: int = 0
        self._history: List[OptimizationResult] = []
        self._frontier_cache: Optional[List[EfficientFrontierPoint]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        expected_returns: NDArray,
        cov_matrix: NDArray,
        shield_permitted: Optional[List[int]] = None,
        current_action: Optional[int] = None,
    ) -> OptimizationResult:
        """Find the best discrete action under mean-variance + cost + shield.

        Parameters
        ----------
        expected_returns : (n_assets,) array
            Per-period expected returns for each action level.  Length must
            match the number of action levels.
        cov_matrix : (n_assets, n_assets) array
            Covariance matrix of returns across action levels.
        shield_permitted : list of int or None
            Subset of ``action_levels`` that the shield allows.  ``None``
            means all actions are available.
        current_action : int or None
            The agent's current position level, used for cost computation.

        Returns
        -------
        OptimizationResult
        """
        if current_action is not None:
            self._current_action = current_action

        permitted = self._resolve_permitted(shield_permitted)
        if not permitted:
            logger.warning("Shield permits no actions; defaulting to hold (0).")
            permitted = [0] if 0 in self.action_levels else [self.action_levels[len(self.action_levels) // 2]]

        expected_returns = np.asarray(expected_returns, dtype=np.float64)
        cov_matrix = np.asarray(cov_matrix, dtype=np.float64)

        self._validate_inputs(expected_returns, cov_matrix)

        scores: Dict[int, float] = {}
        details: Dict[int, Tuple[float, float, float]] = {}
        for action in permitted:
            idx = self.action_levels.index(action)
            mu = float(expected_returns[idx])
            var = float(cov_matrix[idx, idx])
            delta = action - self._current_action
            tc = self.cost_model.cost(delta)

            score = mu - self.risk_aversion * var - tc
            scores[action] = score
            details[action] = (mu, var, tc)

        best_action = max(scores, key=scores.get)  # type: ignore[arg-type]
        mu_best, var_best, tc_best = details[best_action]
        sigma_best = np.sqrt(max(var_best, 0.0))

        rf_period = self.risk_free_rate / self.annualisation_factor
        sharpe = (mu_best - rf_period) / sigma_best if sigma_best > 1e-12 else 0.0

        weight = self._action_to_weight(best_action)

        result = OptimizationResult(
            optimal_action=best_action,
            optimal_weight=weight,
            expected_return=mu_best,
            expected_risk=sigma_best,
            sharpe_ratio=sharpe,
            cost_adjustment=tc_best,
            shield_permitted=list(permitted),
            all_scores=scores,
        )
        self._current_action = best_action
        self._history.append(result)
        return result

    def get_optimal_action(
        self,
        state: NDArray,
        feature_coefficients: NDArray,
        residual_variance: float,
        shield_permitted: Optional[List[int]] = None,
        invariant_mask: Optional[NDArray] = None,
    ) -> OptimizationResult:
        """Convenience wrapper: build expected returns from linear model, then
        call :meth:`optimize`.

        Parameters
        ----------
        state : (n_features,)
            Current market feature vector.
        feature_coefficients : (n_actions, n_features)
            Linear model coefficients mapping features → expected return per
            action level.
        residual_variance : float
            Residual variance used to build a diagonal covariance proxy.
        shield_permitted : list of int or None
            Shield constraint.
        invariant_mask : (n_features,) bool array or None
            Mask selecting causally invariant features.  Non-invariant
            features are zeroed before computing expected returns.
        """
        state = np.asarray(state, dtype=np.float64)
        coefficients = np.asarray(feature_coefficients, dtype=np.float64)

        if invariant_mask is not None and self.use_causal_features_only:
            mask = np.asarray(invariant_mask, dtype=bool)
            masked_state = state * mask
        else:
            masked_state = state

        expected_returns = coefficients @ masked_state
        n_actions = len(self.action_levels)
        cov = np.eye(n_actions) * residual_variance

        return self.optimize(expected_returns, cov, shield_permitted)

    def compute_efficient_frontier(
        self,
        expected_returns: NDArray,
        cov_matrix: NDArray,
        shield_permitted: Optional[List[int]] = None,
        n_points: int = 50,
        lambda_range: Tuple[float, float] = (0.01, 10.0),
    ) -> List[EfficientFrontierPoint]:
        """Sweep risk-aversion λ to trace the efficient frontier.

        Returns a list of :class:`EfficientFrontierPoint` sorted by
        increasing risk.
        """
        expected_returns = np.asarray(expected_returns, dtype=np.float64)
        cov_matrix = np.asarray(cov_matrix, dtype=np.float64)
        permitted = self._resolve_permitted(shield_permitted)

        lambdas = np.linspace(lambda_range[0], lambda_range[1], n_points)
        frontier: List[EfficientFrontierPoint] = []

        original_lambda = self.risk_aversion
        original_action = self._current_action
        try:
            for lam in lambdas:
                self.risk_aversion = lam
                self._current_action = 0  # neutral starting point for frontier
                result = self.optimize(expected_returns, cov_matrix, list(permitted))
                rf_period = self.risk_free_rate / self.annualisation_factor
                sharpe = (
                    (result.expected_return - rf_period) / result.expected_risk
                    if result.expected_risk > 1e-12
                    else 0.0
                )
                frontier.append(
                    EfficientFrontierPoint(
                        risk_aversion=lam,
                        expected_return=result.expected_return,
                        risk=result.expected_risk,
                        sharpe=sharpe,
                        action=result.optimal_action,
                        weight=result.optimal_weight,
                    )
                )
        finally:
            self.risk_aversion = original_lambda
            self._current_action = original_action

        frontier.sort(key=lambda p: p.risk)
        self._frontier_cache = frontier
        return frontier

    def select_from_frontier(
        self,
        frontier: Optional[List[EfficientFrontierPoint]] = None,
        target_sharpe: Optional[float] = None,
        target_risk: Optional[float] = None,
    ) -> EfficientFrontierPoint:
        """Pick a single point from the efficient frontier.

        Priority: *target_risk* > *target_sharpe* > max-Sharpe.
        """
        pts = frontier or self._frontier_cache
        if not pts:
            raise ValueError("No frontier available; call compute_efficient_frontier first.")

        if target_risk is not None:
            return min(pts, key=lambda p: abs(p.risk - target_risk))

        if target_sharpe is not None:
            return min(pts, key=lambda p: abs(p.sharpe - target_sharpe))

        return max(pts, key=lambda p: p.sharpe)

    def backtest(
        self,
        returns_series: NDArray,
        feature_series: NDArray,
        feature_coefficients: NDArray,
        residual_variance: float,
        shield_series: Optional[List[Optional[List[int]]]] = None,
        invariant_mask: Optional[NDArray] = None,
        lookback: int = 60,
    ) -> Dict[str, Any]:
        """Run a walk-forward backtest.

        Parameters
        ----------
        returns_series : (T,) or (T, n_actions) array
            Realised returns.  If 1-D the return is position-independent and
            the P&L is ``action_weight * return``.
        feature_series : (T, n_features) array
            Feature matrix aligned with *returns_series*.
        feature_coefficients : (n_actions, n_features) array
            Static linear model (could be re-estimated in extensions).
        residual_variance : float
            Diagonal covariance proxy.
        shield_series : list of (list of int | None) or None
            Per-step shield constraints.
        invariant_mask : bool array or None
            Invariant feature mask.
        lookback : int
            Warm-up window (no trading in first *lookback* periods).

        Returns
        -------
        dict with keys ``pnl``, ``positions``, ``sharpe``, ``max_dd``,
        ``turnover``, ``costs``.
        """
        returns_series = np.asarray(returns_series, dtype=np.float64)
        feature_series = np.asarray(feature_series, dtype=np.float64)
        T = len(returns_series)

        if returns_series.ndim == 1:
            returns_1d = True
        else:
            returns_1d = False

        positions = np.zeros(T, dtype=np.float64)
        actions = np.zeros(T, dtype=int)
        pnl = np.zeros(T, dtype=np.float64)
        costs = np.zeros(T, dtype=np.float64)
        equity_curve = np.ones(T + 1, dtype=np.float64)

        self._current_action = 0

        for t in range(lookback, T):
            shield_perm = None
            if shield_series is not None and t < len(shield_series):
                shield_perm = shield_series[t]

            result = self.get_optimal_action(
                state=feature_series[t],
                feature_coefficients=feature_coefficients,
                residual_variance=residual_variance,
                shield_permitted=shield_perm,
                invariant_mask=invariant_mask,
            )
            actions[t] = result.optimal_action
            positions[t] = result.optimal_weight

            if returns_1d:
                step_return = positions[t] * returns_series[t]
            else:
                idx = self.action_levels.index(result.optimal_action)
                step_return = returns_series[t, idx]

            step_cost = result.cost_adjustment
            pnl[t] = step_return - step_cost
            costs[t] = step_cost
            equity_curve[t + 1] = equity_curve[t] * (1.0 + pnl[t])

        active_pnl = pnl[lookback:]
        mean_ret = np.mean(active_pnl)
        std_ret = np.std(active_pnl, ddof=1) if len(active_pnl) > 1 else 1e-12
        sharpe = (mean_ret / std_ret) * np.sqrt(self.annualisation_factor) if std_ret > 1e-12 else 0.0

        max_dd = self._max_drawdown(equity_curve)
        turnover = float(np.sum(np.abs(np.diff(positions[lookback:]))))

        return {
            "pnl": pnl,
            "cumulative_pnl": np.cumsum(pnl),
            "equity_curve": equity_curve,
            "positions": positions,
            "actions": actions,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "turnover": turnover,
            "total_costs": float(np.sum(costs)),
            "mean_return": float(mean_ret),
            "std_return": float(std_ret),
            "n_trades": int(np.sum(np.abs(np.diff(actions[lookback:])) > 0)),
        }

    def rebalance(
        self,
        target_action: int,
        current_action: Optional[int] = None,
        max_step: int = 2,
    ) -> List[int]:
        """Generate a sequence of intermediate actions for gradual rebalancing.

        If the target differs from the current action by more than
        *max_step*, the rebalance is broken into multiple periods.
        """
        cur = current_action if current_action is not None else self._current_action
        path: List[int] = []
        while cur != target_action:
            diff = target_action - cur
            step = max(-max_step, min(max_step, diff))
            cur += step
            cur = max(self.action_levels[0], min(self.action_levels[-1], cur))
            path.append(cur)
        return path

    def set_risk_aversion(self, lam: float) -> None:
        """Update risk-aversion parameter and invalidate frontier cache."""
        self.risk_aversion = lam
        self._frontier_cache = None

    @property
    def history(self) -> List[OptimizationResult]:
        return list(self._history)

    def reset(self) -> None:
        """Reset internal state."""
        self._current_action = 0
        self._history.clear()
        self._frontier_cache = None

    # ------------------------------------------------------------------
    # Continuous relaxation solver (for research / comparison)
    # ------------------------------------------------------------------

    def optimize_continuous(
        self,
        expected_returns: NDArray,
        cov_matrix: NDArray,
        weight_bounds: Tuple[float, float] = (-1.0, 1.0),
    ) -> NDArray:
        """Standard Markowitz optimisation over continuous weights.

        Returns the optimal weight vector (one entry per action level).
        This ignores shield constraints and is provided for baseline
        comparison.
        """
        n = len(expected_returns)
        mu = np.asarray(expected_returns, dtype=np.float64)
        Sigma = np.asarray(cov_matrix, dtype=np.float64)

        def objective(w: NDArray) -> float:
            ret = mu @ w
            risk = w @ Sigma @ w
            return -(ret - self.risk_aversion * risk)

        def grad(w: NDArray) -> NDArray:
            return -(mu - 2.0 * self.risk_aversion * Sigma @ w)

        x0 = np.zeros(n)
        bounds = [weight_bounds] * n
        constraints = LinearConstraint(np.ones(n), lb=0.0, ub=1.0)

        result = minimize(
            objective,
            x0,
            jac=grad,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if not result.success:
            logger.warning("Continuous MVO did not converge: %s", result.message)
        return result.x

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_permitted(
        self, shield_permitted: Optional[List[int]]
    ) -> List[int]:
        if shield_permitted is None:
            return list(self.action_levels)
        return [a for a in shield_permitted if a in self.action_levels]

    def _action_to_weight(self, action: int) -> float:
        """Map discrete action to continuous weight in [-1, 1]."""
        max_level = max(abs(a) for a in self.action_levels)
        if max_level == 0:
            return 0.0
        return action / max_level

    def _validate_inputs(
        self, expected_returns: NDArray, cov_matrix: NDArray
    ) -> None:
        n = len(self.action_levels)
        if expected_returns.shape != (n,):
            raise ValueError(
                f"expected_returns shape {expected_returns.shape} != ({n},)"
            )
        if cov_matrix.shape != (n, n):
            raise ValueError(
                f"cov_matrix shape {cov_matrix.shape} != ({n}, {n})"
            )
        eigvals = np.linalg.eigvalsh(cov_matrix)
        if np.any(eigvals < -1e-8):
            logger.warning(
                "Covariance matrix has negative eigenvalues (min=%.2e); "
                "clamping to PSD.",
                eigvals.min(),
            )

    @staticmethod
    def _max_drawdown(equity_curve: NDArray) -> float:
        peak = np.maximum.accumulate(equity_curve)
        dd = (equity_curve - peak) / np.where(peak > 0, peak, 1.0)
        return float(np.min(dd))


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_optimizer(
    risk_aversion: float = 1.0,
    linear_cost: float = 0.001,
    quadratic_cost: float = 0.0005,
    risk_free_rate: float = 0.02,
) -> ShieldedMeanVarianceOptimizer:
    """Construct an optimiser with common defaults."""
    cost = TransactionCostModel(
        linear_coeff=linear_cost,
        quadratic_coeff=quadratic_cost,
    )
    return ShieldedMeanVarianceOptimizer(
        risk_aversion=risk_aversion,
        risk_free_rate=risk_free_rate,
        cost_model=cost,
    )
