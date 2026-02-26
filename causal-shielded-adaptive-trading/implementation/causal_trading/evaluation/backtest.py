"""
Backtest infrastructure for Causal-Shielded Adaptive Trading.

Provides walk-forward backtesting with expanding/rolling windows, embargo
periods, transaction cost modeling, P&L computation, drawdown tracking,
and multi-instrument support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class Side(Enum):
    """Trade side."""
    BUY = 1
    SELL = -1
    FLAT = 0


@dataclass
class TradeRecord:
    """Record of a single executed trade."""
    timestamp_idx: int
    instrument: str
    side: Side
    quantity: float
    price: float
    commission: float
    slippage: float
    spread_cost: float
    total_cost: float
    pnl_realized: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionState:
    """Current position for one instrument."""
    instrument: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0

    def mark_to_market(self, current_price: float) -> float:
        """Compute unrealised P&L at *current_price*."""
        self.unrealized_pnl = self.quantity * (current_price - self.avg_entry_price)
        return self.unrealized_pnl

    def apply_fill(
        self,
        side: Side,
        qty: float,
        fill_price: float,
        commission: float,
    ) -> float:
        """Update position with a fill and return realised P&L from this fill."""
        realized = 0.0
        signed_qty = qty * side.value

        if self.quantity == 0.0:
            self.avg_entry_price = fill_price
            self.quantity = signed_qty
        elif np.sign(signed_qty) == np.sign(self.quantity):
            # Adding to position – update average entry
            total_cost = self.avg_entry_price * abs(self.quantity) + fill_price * qty
            self.quantity += signed_qty
            if abs(self.quantity) > 1e-12:
                self.avg_entry_price = total_cost / abs(self.quantity)
        else:
            # Reducing or flipping position
            close_qty = min(abs(signed_qty), abs(self.quantity))
            realized = close_qty * (fill_price - self.avg_entry_price) * np.sign(self.quantity)
            remaining = signed_qty + self.quantity
            if abs(remaining) < 1e-12:
                self.quantity = 0.0
                self.avg_entry_price = 0.0
            elif np.sign(remaining) == np.sign(self.quantity):
                self.quantity = remaining
            else:
                # Flipped
                self.quantity = remaining
                self.avg_entry_price = fill_price

        self.realized_pnl += realized
        self.total_commission += commission
        return realized


@dataclass
class TransactionCostModel:
    """Models spread, commission, and slippage costs."""
    commission_per_unit: float = 0.0
    commission_min: float = 0.0
    spread_bps: float = 0.0
    slippage_bps: float = 0.0
    market_impact_coeff: float = 0.0  # sqrt-impact coefficient

    def compute_costs(
        self,
        price: float,
        quantity: float,
        avg_daily_volume: float = 1e9,
    ) -> Tuple[float, float, float, float]:
        """Return (commission, spread_cost, slippage, total)."""
        commission = max(self.commission_per_unit * quantity, self.commission_min)
        half_spread = price * self.spread_bps * 0.5 / 1e4
        spread_cost = half_spread * quantity
        linear_slip = price * self.slippage_bps / 1e4 * quantity
        impact = 0.0
        if self.market_impact_coeff > 0.0 and avg_daily_volume > 0.0:
            participation = quantity / avg_daily_volume
            impact = self.market_impact_coeff * price * np.sqrt(participation) * quantity
        slippage = linear_slip + impact
        total = commission + spread_cost + slippage
        return commission, spread_cost, slippage, total


@dataclass
class BacktestConfig:
    """Configuration for the backtest engine."""
    initial_capital: float = 1_000_000.0
    cost_model: TransactionCostModel = field(default_factory=TransactionCostModel)
    warmup_periods: int = 0
    embargo_periods: int = 0
    window_type: str = "expanding"  # "expanding" or "rolling"
    rolling_window_size: int = 252
    rebalance_frequency: int = 1
    max_position_size: float = 1e8
    instruments: List[str] = field(default_factory=lambda: ["default"])
    risk_free_rate: float = 0.0
    annualisation_factor: int = 252


class Strategy(Protocol):
    """Protocol that strategies must satisfy."""

    def on_bar(
        self,
        timestamp_idx: int,
        prices: Dict[str, float],
        positions: Dict[str, PositionState],
        portfolio_value: float,
    ) -> Dict[str, Tuple[Side, float]]:
        """Return desired trades as {instrument: (side, quantity)}."""
        ...


# ---------------------------------------------------------------------------
# Drawdown tracker
# ---------------------------------------------------------------------------

class DrawdownTracker:
    """Tracks drawdown statistics from an equity curve."""

    def __init__(self) -> None:
        self._peak: float = -np.inf
        self._drawdowns: List[float] = []
        self._dd_durations: List[int] = []
        self._current_dd_start: Optional[int] = None
        self._max_dd: float = 0.0
        self._max_dd_duration: int = 0

    def update(self, idx: int, equity: float) -> float:
        """Update with a new equity value; return current drawdown fraction."""
        if equity > self._peak:
            self._peak = equity
            if self._current_dd_start is not None:
                dur = idx - self._current_dd_start
                self._dd_durations.append(dur)
                self._max_dd_duration = max(self._max_dd_duration, dur)
                self._current_dd_start = None

        dd = 0.0
        if self._peak > 0:
            dd = (self._peak - equity) / self._peak

        if dd > 0 and self._current_dd_start is None:
            self._current_dd_start = idx

        self._drawdowns.append(dd)
        self._max_dd = max(self._max_dd, dd)
        return dd

    @property
    def max_drawdown(self) -> float:
        return self._max_dd

    @property
    def max_drawdown_duration(self) -> int:
        return self._max_dd_duration

    @property
    def drawdown_series(self) -> NDArray[np.float64]:
        return np.array(self._drawdowns, dtype=np.float64)

    def average_drawdown(self) -> float:
        if len(self._drawdowns) == 0:
            return 0.0
        arr = np.array(self._drawdowns)
        mask = arr > 0
        if not np.any(mask):
            return 0.0
        return float(np.mean(arr[mask]))

    def calmar_ratio(self, annualised_return: float) -> float:
        if self._max_dd < 1e-12:
            return np.inf if annualised_return > 0 else 0.0
        return annualised_return / self._max_dd


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResults:
    """Aggregated results of a backtest run."""
    equity_curve: NDArray[np.float64]
    returns: NDArray[np.float64]
    positions_history: Dict[str, NDArray[np.float64]]
    trades: List[TradeRecord]
    drawdown_series: NDArray[np.float64]
    max_drawdown: float
    max_drawdown_duration: int
    total_return: float
    annualised_return: float
    annualised_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    total_commission: float
    total_slippage: float
    initial_capital: float
    final_capital: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            "=== Backtest Summary ===",
            f"Total Return:        {self.total_return:>12.4%}",
            f"Annualised Return:   {self.annualised_return:>12.4%}",
            f"Annualised Vol:      {self.annualised_volatility:>12.4%}",
            f"Sharpe Ratio:        {self.sharpe_ratio:>12.4f}",
            f"Sortino Ratio:       {self.sortino_ratio:>12.4f}",
            f"Calmar Ratio:        {self.calmar_ratio:>12.4f}",
            f"Max Drawdown:        {self.max_drawdown:>12.4%}",
            f"Max DD Duration:     {self.max_drawdown_duration:>12d} bars",
            f"Total Trades:        {self.total_trades:>12d}",
            f"Win Rate:            {self.win_rate:>12.4%}",
            f"Profit Factor:       {self.profit_factor:>12.4f}",
            f"Avg Trade P&L:       {self.avg_trade_pnl:>12.2f}",
            f"Total Commission:    {self.total_commission:>12.2f}",
            f"Total Slippage:      {self.total_slippage:>12.2f}",
            f"Initial Capital:     {self.initial_capital:>12.2f}",
            f"Final Capital:       {self.final_capital:>12.2f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Core backtest engine with walk-forward support.

    Parameters
    ----------
    config : BacktestConfig
        Backtest configuration.
    """

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        self.config = config or BacktestConfig()
        self._positions: Dict[str, PositionState] = {}
        self._trades: List[TradeRecord] = []
        self._equity_curve: List[float] = []
        self._positions_history: Dict[str, List[float]] = {}
        self._dd_tracker = DrawdownTracker()
        self._portfolio_value: float = self.config.initial_capital
        self._cash: float = self.config.initial_capital
        self._bar_count: int = 0

        for inst in self.config.instruments:
            self._positions[inst] = PositionState(instrument=inst)
            self._positions_history[inst] = []

    # ----- internal helpers ------------------------------------------------

    def _apply_transaction_costs(
        self,
        price: float,
        quantity: float,
        avg_daily_volume: float = 1e9,
    ) -> Tuple[float, float, float, float]:
        return self.config.cost_model.compute_costs(price, quantity, avg_daily_volume)

    def _execute_trade(
        self,
        timestamp_idx: int,
        instrument: str,
        side: Side,
        quantity: float,
        price: float,
        avg_daily_volume: float = 1e9,
    ) -> TradeRecord:
        """Execute a trade, updating position and cash."""
        if quantity <= 0:
            raise ValueError("Trade quantity must be positive.")

        commission, spread_cost, slippage, total_cost = self._apply_transaction_costs(
            price, quantity, avg_daily_volume,
        )

        # Effective fill price includes slippage
        if side == Side.BUY:
            fill_price = price + slippage / max(quantity, 1e-12)
        else:
            fill_price = price - slippage / max(quantity, 1e-12)

        pos = self._positions[instrument]
        realized_pnl = pos.apply_fill(side, quantity, fill_price, commission)

        # Cash impact
        if side == Side.BUY:
            self._cash -= fill_price * quantity + commission + spread_cost
        else:
            self._cash += fill_price * quantity - commission - spread_cost

        trade = TradeRecord(
            timestamp_idx=timestamp_idx,
            instrument=instrument,
            side=side,
            quantity=quantity,
            price=price,
            commission=commission,
            slippage=slippage,
            spread_cost=spread_cost,
            total_cost=total_cost,
            pnl_realized=realized_pnl,
        )
        self._trades.append(trade)
        return trade

    def _mark_all_positions(self, prices: Dict[str, float]) -> float:
        """Mark all positions to market and return total portfolio value."""
        total_unrealised = 0.0
        for inst, pos in self._positions.items():
            if inst in prices and abs(pos.quantity) > 1e-12:
                pos.mark_to_market(prices[inst])
                total_unrealised += pos.unrealized_pnl
        return self._cash + total_unrealised + sum(
            abs(p.quantity) * prices.get(p.instrument, 0.0)
            for p in self._positions.values()
            if p.instrument in prices
        )

    def _recompute_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Full portfolio value = cash + sum(position_value)."""
        pv = self._cash
        for inst, pos in self._positions.items():
            if inst in prices:
                pv += pos.quantity * prices[inst]
        return pv

    def _clamp_quantity(self, instrument: str, side: Side, qty: float) -> float:
        """Clamp quantity to max position size."""
        pos = self._positions[instrument]
        projected = pos.quantity + qty * side.value
        if abs(projected) > self.config.max_position_size:
            allowed = self.config.max_position_size - abs(pos.quantity)
            return max(allowed, 0.0)
        return qty

    # ----- public API -------------------------------------------------------

    def reset(self) -> None:
        """Reset engine to initial state."""
        self._positions = {}
        self._trades = []
        self._equity_curve = []
        self._positions_history = {inst: [] for inst in self.config.instruments}
        self._dd_tracker = DrawdownTracker()
        self._portfolio_value = self.config.initial_capital
        self._cash = self.config.initial_capital
        self._bar_count = 0
        for inst in self.config.instruments:
            self._positions[inst] = PositionState(instrument=inst)

    def run(
        self,
        prices: Dict[str, NDArray[np.float64]],
        strategy: Strategy,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BacktestResults:
        """Run the backtest over *prices* using *strategy*.

        Parameters
        ----------
        prices : dict mapping instrument name → 1-D array of prices
        strategy : object satisfying the Strategy protocol
        metadata : optional extra info attached to results

        Returns
        -------
        BacktestResults
        """
        self.reset()
        n_bars = min(len(v) for v in prices.values())
        if n_bars == 0:
            raise ValueError("Price arrays must be non-empty.")

        warmup = self.config.warmup_periods
        embargo = self.config.embargo_periods
        start_idx = warmup + embargo

        for t in range(n_bars):
            current_prices = {inst: float(prices[inst][t]) for inst in prices}

            # Record position sizes
            for inst in self.config.instruments:
                self._positions_history[inst].append(self._positions[inst].quantity)

            # Skip warmup + embargo period
            if t < start_idx:
                pv = self._recompute_portfolio_value(current_prices)
                self._portfolio_value = pv
                self._equity_curve.append(pv)
                self._dd_tracker.update(t, pv)
                self._bar_count += 1
                continue

            # Only trade at rebalance frequency
            if (t - start_idx) % self.config.rebalance_frequency != 0:
                pv = self._recompute_portfolio_value(current_prices)
                self._portfolio_value = pv
                self._equity_curve.append(pv)
                self._dd_tracker.update(t, pv)
                self._bar_count += 1
                continue

            # Strategy signal
            desired = strategy.on_bar(
                t, current_prices, dict(self._positions), self._portfolio_value,
            )

            # Execute desired trades
            for inst, (side, qty) in desired.items():
                if side == Side.FLAT or qty <= 0:
                    continue
                if inst not in self._positions:
                    logger.warning("Unknown instrument %s – skipping.", inst)
                    continue
                qty = self._clamp_quantity(inst, side, qty)
                if qty > 0:
                    self._execute_trade(t, inst, side, qty, current_prices[inst])

            pv = self._recompute_portfolio_value(current_prices)
            self._portfolio_value = pv
            self._equity_curve.append(pv)
            self._dd_tracker.update(t, pv)
            self._bar_count += 1

        return self._compile_results(metadata)

    def _compile_results(self, metadata: Optional[Dict[str, Any]]) -> BacktestResults:
        """Build the BacktestResults object."""
        equity = np.array(self._equity_curve, dtype=np.float64)
        returns = np.diff(equity) / np.maximum(equity[:-1], 1e-12) if len(equity) > 1 else np.array([])

        ann = self.config.annualisation_factor
        rf_per_bar = self.config.risk_free_rate / ann

        total_ret = (equity[-1] / equity[0]) - 1.0 if len(equity) > 1 else 0.0
        n_years = len(equity) / ann if ann > 0 else 1.0
        ann_ret = (1.0 + total_ret) ** (1.0 / max(n_years, 1e-6)) - 1.0

        vol = float(np.std(returns, ddof=1)) * np.sqrt(ann) if len(returns) > 1 else 0.0
        excess = returns - rf_per_bar if len(returns) > 0 else np.array([])
        sharpe = float(np.mean(excess)) / max(float(np.std(excess, ddof=1)), 1e-12) * np.sqrt(ann) if len(excess) > 1 else 0.0

        downside = excess[excess < 0]
        down_vol = float(np.std(downside, ddof=1)) * np.sqrt(ann) if len(downside) > 1 else 1e-12
        sortino = float(np.mean(excess)) * ann / max(down_vol, 1e-12) if len(excess) > 0 else 0.0

        calmar = self._dd_tracker.calmar_ratio(ann_ret)

        # Trade statistics
        realised_pnls = np.array([t.pnl_realized for t in self._trades], dtype=np.float64)
        n_trades = len(self._trades)
        wins = realised_pnls[realised_pnls > 0]
        losses = realised_pnls[realised_pnls < 0]
        win_rate = len(wins) / max(n_trades, 1)
        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = abs(float(np.sum(losses))) if len(losses) > 0 else 1e-12
        profit_factor = gross_profit / max(gross_loss, 1e-12)
        avg_pnl = float(np.mean(realised_pnls)) if n_trades > 0 else 0.0

        total_comm = sum(t.commission for t in self._trades)
        total_slip = sum(t.slippage for t in self._trades)

        pos_hist = {
            inst: np.array(vals, dtype=np.float64)
            for inst, vals in self._positions_history.items()
        }

        return BacktestResults(
            equity_curve=equity,
            returns=returns,
            positions_history=pos_hist,
            trades=list(self._trades),
            drawdown_series=self._dd_tracker.drawdown_series,
            max_drawdown=self._dd_tracker.max_drawdown,
            max_drawdown_duration=self._dd_tracker.max_drawdown_duration,
            total_return=total_ret,
            annualised_return=ann_ret,
            annualised_volatility=vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            total_trades=n_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_pnl,
            total_commission=total_comm,
            total_slippage=total_slip,
            initial_capital=self.config.initial_capital,
            final_capital=float(equity[-1]) if len(equity) > 0 else self.config.initial_capital,
            metadata=metadata or {},
        )

    def get_results(self) -> BacktestResults:
        """Return results of the last run (re-compiles from current state)."""
        return self._compile_results(None)

    def get_trades(self) -> List[TradeRecord]:
        """Return list of all executed trades."""
        return list(self._trades)

    def get_equity_curve(self) -> NDArray[np.float64]:
        """Return the equity curve array."""
        return np.array(self._equity_curve, dtype=np.float64)


# ---------------------------------------------------------------------------
# Multi-instrument convenience
# ---------------------------------------------------------------------------

def run_multi_instrument_backtest(
    prices: Dict[str, NDArray[np.float64]],
    strategy: Strategy,
    config: Optional[BacktestConfig] = None,
) -> BacktestResults:
    """Convenience wrapper to run a multi-instrument backtest.

    Parameters
    ----------
    prices : dict mapping instrument name → price array
    strategy : Strategy implementation
    config : optional backtest configuration

    Returns
    -------
    BacktestResults
    """
    if config is None:
        config = BacktestConfig(instruments=list(prices.keys()))
    else:
        config.instruments = list(prices.keys())
    engine = BacktestEngine(config)
    return engine.run(prices, strategy)


def compute_rolling_sharpe(
    returns: NDArray[np.float64],
    window: int = 63,
    annualisation: int = 252,
    risk_free_rate: float = 0.0,
) -> NDArray[np.float64]:
    """Compute rolling Sharpe ratio.

    Parameters
    ----------
    returns : 1-D array of per-bar returns
    window : rolling window length
    annualisation : annualisation factor
    risk_free_rate : annualised risk-free rate

    Returns
    -------
    1-D array of rolling Sharpe values (NaN-padded at the front)
    """
    n = len(returns)
    result = np.full(n, np.nan, dtype=np.float64)
    rf = risk_free_rate / annualisation
    for i in range(window - 1, n):
        chunk = returns[i - window + 1 : i + 1]
        excess = chunk - rf
        mu = np.mean(excess)
        sigma = np.std(excess, ddof=1)
        if sigma > 1e-12:
            result[i] = mu / sigma * np.sqrt(annualisation)
    return result


def compute_rolling_sortino(
    returns: NDArray[np.float64],
    window: int = 63,
    annualisation: int = 252,
    risk_free_rate: float = 0.0,
) -> NDArray[np.float64]:
    """Compute rolling Sortino ratio."""
    n = len(returns)
    result = np.full(n, np.nan, dtype=np.float64)
    rf = risk_free_rate / annualisation
    for i in range(window - 1, n):
        chunk = returns[i - window + 1 : i + 1]
        excess = chunk - rf
        mu = np.mean(excess)
        down = excess[excess < 0]
        dsigma = np.std(down, ddof=1) if len(down) > 1 else 1e-12
        result[i] = mu / max(dsigma, 1e-12) * np.sqrt(annualisation)
    return result


def compute_trade_statistics(trades: List[TradeRecord]) -> Dict[str, float]:
    """Compute summary statistics from a list of trades.

    Returns a dict with keys: n_trades, win_rate, avg_win, avg_loss,
    profit_factor, expectancy, max_win, max_loss, avg_holding_bars.
    """
    if not trades:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "max_win": 0.0,
            "max_loss": 0.0,
        }

    pnls = np.array([t.pnl_realized for t in trades], dtype=np.float64)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    n = len(pnls)
    wr = len(wins) / n if n > 0 else 0.0
    aw = float(np.mean(wins)) if len(wins) > 0 else 0.0
    al = float(np.mean(losses)) if len(losses) > 0 else 0.0
    gp = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gl = abs(float(np.sum(losses))) if len(losses) > 0 else 1e-12
    pf = gp / max(gl, 1e-12)
    expectancy = wr * aw + (1 - wr) * al

    return {
        "n_trades": n,
        "win_rate": wr,
        "avg_win": aw,
        "avg_loss": al,
        "profit_factor": pf,
        "expectancy": expectancy,
        "max_win": float(np.max(pnls)) if n > 0 else 0.0,
        "max_loss": float(np.min(pnls)) if n > 0 else 0.0,
    }
