"""
Baseline trading strategies for comparative evaluation.

Implements buy-and-hold, risk-parity, momentum, mean-reversion,
unshielded mean-variance, and oracle strategies to benchmark against
the full CSAT pipeline.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract strategy interface
# ---------------------------------------------------------------------------

class BaseStrategy(ABC):
    """Base class for trading strategies."""

    name: str = "base"

    @abstractmethod
    def compute_signal(
        self,
        features: NDArray,
        returns_history: NDArray,
        t: int,
    ) -> int:
        """Compute discrete action in {-3, -2, -1, 0, 1, 2, 3}."""
        ...

    def reset(self) -> None:
        """Reset internal state for a new fold."""
        pass


# ---------------------------------------------------------------------------
# Buy-and-hold
# ---------------------------------------------------------------------------

class BuyAndHoldStrategy(BaseStrategy):
    """Always hold a fixed long position."""

    name = "buy_hold"

    def __init__(self, action: int = 1) -> None:
        self.action = action

    def compute_signal(self, features, returns_history, t):
        return self.action


# ---------------------------------------------------------------------------
# Momentum (cross-sectional and time-series)
# ---------------------------------------------------------------------------

class MomentumStrategy(BaseStrategy):
    """Time-series momentum: go long if trailing return > 0, short otherwise."""

    name = "momentum"

    def __init__(self, lookback: int = 20, threshold: float = 0.0) -> None:
        self.lookback = lookback
        self.threshold = threshold

    def compute_signal(self, features, returns_history, t):
        if t < self.lookback:
            return 0
        trailing = np.sum(returns_history[t - self.lookback : t])
        if trailing > self.threshold:
            return min(int(np.clip(trailing / 0.02, 1, 3)), 3)
        elif trailing < -self.threshold:
            return max(int(np.clip(trailing / 0.02, -3, -1)), -3)
        return 0


# ---------------------------------------------------------------------------
# Mean-reversion
# ---------------------------------------------------------------------------

class MeanReversionStrategy(BaseStrategy):
    """Mean-reversion: go long when price is below MA, short above."""

    name = "mean_reversion"

    def __init__(self, lookback: int = 50, n_std: float = 1.5) -> None:
        self.lookback = lookback
        self.n_std = n_std

    def compute_signal(self, features, returns_history, t):
        if t < self.lookback:
            return 0
        window = returns_history[t - self.lookback : t]
        mu = np.mean(window)
        sigma = np.std(window) + 1e-10
        z = (returns_history[t - 1] - mu) / sigma
        if z < -self.n_std:
            return min(int(-z), 3)
        elif z > self.n_std:
            return max(int(-z), -3)
        return 0


# ---------------------------------------------------------------------------
# Risk-parity (equal vol contribution)
# ---------------------------------------------------------------------------

class RiskParityStrategy(BaseStrategy):
    """Risk-parity: size position inversely proportional to volatility."""

    name = "risk_parity"

    def __init__(self, lookback: int = 60, target_vol: float = 0.10) -> None:
        self.lookback = lookback
        self.target_vol = target_vol

    def compute_signal(self, features, returns_history, t):
        if t < self.lookback:
            return 1
        window = returns_history[t - self.lookback : t]
        vol = np.std(window) * np.sqrt(252) + 1e-10
        weight = self.target_vol / vol
        action = int(np.clip(np.round(weight), -3, 3))
        return action if action != 0 else 1


# ---------------------------------------------------------------------------
# Unshielded mean-variance (no safety shield)
# ---------------------------------------------------------------------------

class UnshieldedMeanVarianceStrategy(BaseStrategy):
    """Mean-variance optimiser without the safety shield."""

    name = "unshielded_mv"

    def __init__(
        self,
        lookback: int = 60,
        risk_aversion: float = 1.0,
    ) -> None:
        self.lookback = lookback
        self.risk_aversion = risk_aversion
        self._current_action = 0

    def compute_signal(self, features, returns_history, t):
        if t < self.lookback:
            return 0
        window = returns_history[t - self.lookback : t]
        mu = np.mean(window)
        sigma = np.std(window) + 1e-10

        actions = np.array([-3, -2, -1, 0, 1, 2, 3])
        scores = []
        for a in actions:
            exp_ret = a * mu
            risk = abs(a) * sigma
            cost = 0.001 * abs(a - self._current_action)
            scores.append(exp_ret - self.risk_aversion * risk ** 2 - cost)

        best_idx = np.argmax(scores)
        self._current_action = int(actions[best_idx])
        return self._current_action

    def reset(self):
        self._current_action = 0


# ---------------------------------------------------------------------------
# Oracle (knows true regime, for upper bound)
# ---------------------------------------------------------------------------

class OracleStrategy(BaseStrategy):
    """Oracle: uses ground-truth regime labels for perfect regime-aware trading."""

    name = "oracle"

    def __init__(self, regime_labels: NDArray, regime_actions: Dict[int, int]) -> None:
        self.regime_labels = regime_labels
        self.regime_actions = regime_actions

    def compute_signal(self, features, returns_history, t):
        if t >= len(self.regime_labels):
            return 0
        regime = int(self.regime_labels[t])
        return self.regime_actions.get(regime, 0)


# ---------------------------------------------------------------------------
# CSAT strategy (the full pipeline)
# ---------------------------------------------------------------------------

class CSATStrategy(BaseStrategy):
    """Full CSAT pipeline: regime detection + causal discovery + shield."""

    name = "CSAT"

    def __init__(
        self,
        n_regimes: int = 3,
        shield_delta: float = 0.05,
        n_features: int = 5,
        lookback: int = 100,
        risk_aversion: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.n_regimes = n_regimes
        self.shield_delta = shield_delta
        self.n_features = n_features
        self.lookback = lookback
        self.risk_aversion = risk_aversion
        self.seed = seed
        self._fitted = False
        self._current_action = 0
        self._hmm = None
        self._shield = None
        self._regime_returns: Dict[int, List[float]] = {}

    def _fit(self, features: NDArray, returns_history: NDArray) -> None:
        """Fit the CSAT pipeline on training data."""
        from causal_trading.regime.sticky_hdp_hmm import StickyHDPHMM
        from causal_trading.shield.shield_synthesis import PosteriorPredictiveShield
        from causal_trading.shield.bounded_liveness_specs import (
            DrawdownRecoverySpec,
        )

        self._hmm = StickyHDPHMM(
            K_max=self.n_regimes + 2,
            n_iter=30,
            random_state=self.seed,
        )
        self._hmm.fit(features)

        n_states = 10
        n_actions = 7
        self._shield = PosteriorPredictiveShield(
            n_states=n_states,
            n_actions=n_actions,
            delta=self.shield_delta,
        )
        self._shield.add_spec(
            DrawdownRecoverySpec(threshold=0.10, horizon=20),
            name="drawdown",
        )
        self._shield.synthesize()

        # Learn regime-return mapping
        states = self._hmm.states_
        self._regime_returns = {}
        for r in np.unique(states):
            mask = states == r
            if np.sum(mask) > 0 and len(returns_history) >= len(mask):
                self._regime_returns[int(r)] = float(
                    np.mean(returns_history[: len(states)][mask])
                )

        self._fitted = True

    def compute_signal(self, features, returns_history, t):
        if t < self.lookback:
            return 0

        if not self._fitted:
            train_features = features[: t]
            train_returns = returns_history[: t]
            self._fit(train_features, train_returns)

        # Predict regime
        if self._hmm is not None and t < len(features):
            try:
                curr_feat = features[t : t + 1]
                if hasattr(self._hmm, 'predict_single'):
                    regime = self._hmm.predict_single(curr_feat)
                else:
                    regime = int(self._hmm.states_[-1]) if len(self._hmm.states_) > 0 else 0
            except Exception:
                regime = 0
        else:
            regime = 0

        # Estimate expected return from regime
        exp_ret = self._regime_returns.get(regime, 0.0)
        window = returns_history[max(0, t - 60) : t]
        vol = np.std(window) + 1e-10 if len(window) > 5 else 0.01

        # Score actions
        actions = np.array([-3, -2, -1, 0, 1, 2, 3])
        scores = []
        for a in actions:
            ret = a * exp_ret
            risk = abs(a) * vol
            cost = 0.001 * abs(a - self._current_action)
            scores.append(ret - self.risk_aversion * risk ** 2 - cost)

        # Apply shield
        if self._shield is not None:
            state_idx = min(int(abs(self._current_action)), 9)
            permitted = self._shield.get_permitted_actions(state=state_idx)
            for i, a in enumerate(actions):
                if i < len(permitted) and not permitted[i]:
                    scores[i] = -1e10

        best_idx = int(np.argmax(scores))
        self._current_action = int(actions[best_idx])
        return self._current_action

    def reset(self):
        self._fitted = False
        self._current_action = 0
        self._hmm = None
        self._shield = None
        self._regime_returns = {}


# ---------------------------------------------------------------------------
# Walk-forward backtester
# ---------------------------------------------------------------------------

@dataclass
class StrategyResult:
    """Results for one strategy across the backtest."""
    name: str
    equity_curve: NDArray
    returns: NDArray
    actions: NDArray
    total_return: float
    annualised_return: float
    annualised_vol: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    n_trades: int
    avg_action_magnitude: float
    action_changes: int


def run_walk_forward_backtest(
    features: NDArray,
    returns: NDArray,
    strategy: BaseStrategy,
    warmup: int = 100,
    cost_bps: float = 5.0,
) -> StrategyResult:
    """Run a walk-forward backtest for one strategy.

    Parameters
    ----------
    features : (T, D) array of feature observations
    returns : (T,) array of per-period returns
    strategy : strategy to evaluate
    warmup : number of initial periods to skip
    cost_bps : transaction cost in basis points

    Returns
    -------
    StrategyResult
    """
    T = len(returns)
    equity = np.ones(T) * 1_000_000.0
    strat_returns = np.zeros(T)
    actions = np.zeros(T, dtype=int)
    prev_action = 0

    strategy.reset()

    for t in range(warmup, T):
        action = strategy.compute_signal(features, returns, t)
        actions[t] = action

        # Position return = action_level * market_return
        # Scaled to fraction of capital (action/3 = fraction)
        position_frac = action / 3.0
        period_ret = position_frac * returns[t]

        # Transaction cost
        trade_delta = abs(action - prev_action) / 3.0
        tc = trade_delta * cost_bps / 1e4
        period_ret -= tc

        strat_returns[t] = period_ret
        equity[t] = equity[t - 1] * (1 + period_ret)
        prev_action = action

    # Trim to active period
    active = strat_returns[warmup:]
    eq = equity[warmup:]

    ann_ret = float(np.mean(active) * 252)
    ann_vol = float(np.std(active) * np.sqrt(252)) + 1e-10
    sharpe = ann_ret / ann_vol
    downside = np.std(active[active < 0]) * np.sqrt(252) + 1e-10
    sortino = ann_ret / downside

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    max_dd = float(np.max(dd))
    calmar = ann_ret / max_dd if max_dd > 1e-10 else 0.0

    # Trade counts
    action_changes = int(np.sum(np.diff(actions[warmup:]) != 0))

    total_ret = float(eq[-1] / eq[0] - 1)

    return StrategyResult(
        name=strategy.name,
        equity_curve=eq,
        returns=active,
        actions=actions[warmup:],
        total_return=total_ret,
        annualised_return=ann_ret,
        annualised_vol=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        calmar_ratio=calmar,
        n_trades=action_changes,
        avg_action_magnitude=float(np.mean(np.abs(actions[warmup:]))),
        action_changes=action_changes,
    )


def compare_strategies(
    features: NDArray,
    returns: NDArray,
    strategies: List[BaseStrategy],
    warmup: int = 100,
    cost_bps: float = 5.0,
) -> Dict[str, StrategyResult]:
    """Run multiple strategies on the same data and compare."""
    results = {}
    for strategy in strategies:
        result = run_walk_forward_backtest(
            features, returns, strategy, warmup=warmup, cost_bps=cost_bps
        )
        results[result.name] = result
        logger.info(
            "%s: Sharpe=%.3f MaxDD=%.3f TotalRet=%.3f",
            result.name,
            result.sharpe_ratio,
            result.max_drawdown,
            result.total_return,
        )
    return results
