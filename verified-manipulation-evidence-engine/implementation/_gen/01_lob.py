# Generate LOB matching, features, simulator, manipulation modules

total += w("vmee/lob/matching.py", '''\
"""
Order matching engine with price-time priority.
Handles order routing, execution, and trade generation across multiple instruments.
"""
from __future__ import annotations
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from vmee.lob.orderbook import (
    LimitOrderBook, Order, OrderSide, OrderStatus, OrderType, Trade, BookSnapshot,
)

logger = logging.getLogger(__name__)


@dataclass
class ExecutionReport:
    """Report of order execution status."""
    order_id: str
    status: OrderStatus
    filled_quantity: int
    remaining_quantity: int
    average_price: float
    trades: list[Trade]
    timestamp: float
    latency_us: float = 0.0
    reject_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "status": self.status.name,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "average_price": self.average_price,
            "num_trades": len(self.trades),
            "timestamp": self.timestamp,
            "latency_us": self.latency_us,
            "reject_reason": self.reject_reason,
        }


@dataclass
class OrderFlowStats:
    """Aggregate statistics about order flow."""
    total_orders: int = 0
    total_trades: int = 0
    total_cancels: int = 0
    total_modifies: int = 0
    buy_orders: int = 0
    sell_orders: int = 0
    limit_orders: int = 0
    market_orders: int = 0
    total_volume_traded: int = 0
    total_notional: float = 0.0
    cancel_rate: float = 0.0
    trade_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_orders": self.total_orders,
            "total_trades": self.total_trades,
            "total_cancels": self.total_cancels,
            "total_modifies": self.total_modifies,
            "buy_orders": self.buy_orders,
            "sell_orders": self.sell_orders,
            "limit_orders": self.limit_orders,
            "market_orders": self.market_orders,
            "total_volume_traded": self.total_volume_traded,
            "total_notional": self.total_notional,
            "cancel_rate": self.cancel_rate,
            "trade_rate": self.trade_rate,
        }


class LatencyModel:
    """Model network latency for realistic simulation."""
    def __init__(self, mean_us: float = 50.0, std_us: float = 10.0, seed: int = 42):
        self.mean_us = mean_us
        self.std_us = std_us
        self.rng = np.random.RandomState(seed)

    def sample(self) -> float:
        return max(1.0, self.rng.normal(self.mean_us, self.std_us))

    def sample_batch(self, n: int) -> np.ndarray:
        return np.maximum(1.0, self.rng.normal(self.mean_us, self.std_us, n))


class MatchingEngine:
    """
    Order matching engine managing multiple order books.
    Provides order routing, pre-trade risk checks, execution,
    and post-trade processing across multiple instruments.
    """
    def __init__(
        self,
        tick_size: float = 0.01,
        lot_size: int = 1,
        max_order_size: int = 100000,
        max_price_deviation: float = 0.10,
        latency_model: Optional[LatencyModel] = None,
    ):
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.max_order_size = max_order_size
        self.max_price_deviation = max_price_deviation
        self._books: dict[str, LimitOrderBook] = {}
        self._order_sequence: int = 0
        self._global_trades: list[Trade] = []
        self._execution_reports: list[ExecutionReport] = []
        self._stats: dict[str, OrderFlowStats] = defaultdict(OrderFlowStats)
        self._latency_model = latency_model
        self._positions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._pnl: dict[str, float] = defaultdict(float)
        self._circuit_breaker_active: dict[str, bool] = defaultdict(bool)
        self._reference_prices: dict[str, float] = {}

    def create_book(self, instrument_id: str, **kwargs) -> LimitOrderBook:
        book = LimitOrderBook(
            instrument_id=instrument_id,
            tick_size=self.tick_size,
            lot_size=self.lot_size,
            **kwargs,
        )
        self._books[instrument_id] = book
        self._stats[instrument_id] = OrderFlowStats()
        return book

    def get_book(self, instrument_id: str) -> Optional[LimitOrderBook]:
        return self._books.get(instrument_id)

    def submit_order(self, order: Order) -> ExecutionReport:
        """Submit an order to the matching engine."""
        self._order_sequence += 1
        start_time = time.monotonic()
        order.sequence_number = self._order_sequence

        reject_reason = self._validate_order(order)
        if reject_reason:
            order.status = OrderStatus.REJECTED
            return ExecutionReport(
                order_id=order.order_id, status=OrderStatus.REJECTED,
                filled_quantity=0, remaining_quantity=order.quantity,
                average_price=0.0, trades=[], timestamp=order.timestamp,
                reject_reason=reject_reason,
            )

        book = self._books.get(order.instrument_id)
        if book is None:
            book = self.create_book(order.instrument_id)

        trades = book.add_order(order)

        stats = self._stats[order.instrument_id]
        stats.total_orders += 1
        if order.is_buy:
            stats.buy_orders += 1
        else:
            stats.sell_orders += 1
        if order.order_type == OrderType.MARKET:
            stats.market_orders += 1
        else:
            stats.limit_orders += 1

        for trade in trades:
            stats.total_trades += 1
            stats.total_volume_traded += trade.quantity
            stats.total_notional += trade.quantity * trade.price
            self._positions[trade.buyer_trader_id][trade.instrument_id] += trade.quantity
            self._pnl[trade.buyer_trader_id] -= trade.quantity * trade.price
            self._positions[trade.seller_trader_id][trade.instrument_id] -= trade.quantity
            self._pnl[trade.seller_trader_id] += trade.quantity * trade.price

        self._global_trades.extend(trades)

        if trades:
            total_notional = sum(t.price * t.quantity for t in trades)
            total_qty = sum(t.quantity for t in trades)
            avg_price = total_notional / total_qty if total_qty > 0 else 0.0
        else:
            avg_price = 0.0

        elapsed = (time.monotonic() - start_time) * 1e6

        return ExecutionReport(
            order_id=order.order_id, status=order.status,
            filled_quantity=order.filled_quantity,
            remaining_quantity=order.remaining_quantity,
            average_price=avg_price, trades=trades,
            timestamp=order.timestamp, latency_us=elapsed,
        )

    def cancel_order(self, instrument_id: str, order_id: str, timestamp: float = 0.0) -> Optional[Order]:
        book = self._books.get(instrument_id)
        if book is None:
            return None
        cancelled = book.cancel_order(order_id, timestamp)
        if cancelled:
            self._stats[instrument_id].total_cancels += 1
        return cancelled

    def modify_order(self, instrument_id: str, order_id: str,
                     new_price: Optional[float] = None,
                     new_quantity: Optional[int] = None,
                     timestamp: float = 0.0) -> tuple[Optional[Order], list[Trade]]:
        book = self._books.get(instrument_id)
        if book is None:
            return None, []
        result = book.modify_order(order_id, new_price, new_quantity, timestamp)
        if result[0] is not None:
            self._stats[instrument_id].total_modifies += 1
        return result

    def _validate_order(self, order: Order) -> str:
        if order.quantity <= 0:
            return "Order quantity must be positive"
        if order.quantity > self.max_order_size:
            return f"Order quantity {order.quantity} exceeds max {self.max_order_size}"
        if order.quantity % self.lot_size != 0:
            return f"Order quantity must be multiple of lot size {self.lot_size}"
        if order.order_type != OrderType.MARKET:
            if order.price <= 0:
                return "Limit order price must be positive"
            ref = self._reference_prices.get(order.instrument_id)
            if ref is not None and ref > 0:
                deviation = abs(order.price - ref) / ref
                if deviation > self.max_price_deviation:
                    return f"Price deviation {deviation:.2%} exceeds max {self.max_price_deviation:.2%}"
        if self._circuit_breaker_active.get(order.instrument_id, False):
            return "Circuit breaker is active for this instrument"
        return ""

    def check_circuit_breaker(self, instrument_id: str, threshold: float = 0.05) -> bool:
        ref = self._reference_prices.get(instrument_id)
        book = self._books.get(instrument_id)
        if ref is None or book is None or book.mid_price is None:
            return False
        deviation = abs(book.mid_price - ref) / ref
        if deviation > threshold:
            self._circuit_breaker_active[instrument_id] = True
            logger.warning(f"Circuit breaker triggered for {instrument_id}: deviation={deviation:.2%}")
            return True
        return False

    def set_reference_price(self, instrument_id: str, price: float) -> None:
        self._reference_prices[instrument_id] = price

    def get_position(self, trader_id: str, instrument_id: str) -> int:
        return self._positions[trader_id][instrument_id]

    def get_pnl(self, trader_id: str) -> float:
        return self._pnl[trader_id]

    def get_stats(self, instrument_id: str) -> OrderFlowStats:
        stats = self._stats[instrument_id]
        if stats.total_orders > 0:
            stats.cancel_rate = stats.total_cancels / stats.total_orders
            stats.trade_rate = stats.total_trades / stats.total_orders
        return stats

    def all_trades(self) -> list[Trade]:
        return list(self._global_trades)

    def snapshot_all(self, timestamp: float = 0.0) -> dict[str, BookSnapshot]:
        return {inst: book.snapshot(timestamp=timestamp) for inst, book in self._books.items()}
''')

total += w("vmee/lob/features.py", '''\
"""
Market microstructure feature extraction from LOB state and order flow.
"""
from __future__ import annotations
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from vmee.lob.orderbook import BookSnapshot, LimitOrderBook, Order, OrderSide, OrderType, Trade

logger = logging.getLogger(__name__)


@dataclass
class MicrostructureFeatures:
    """Container for a single timestamp's microstructure features."""
    timestamp: float = 0.0
    mid_price: float = 0.0
    bid_ask_spread: float = 0.0
    spread_ticks: int = 0
    weighted_mid_price: float = 0.0
    micro_price: float = 0.0
    bid_depth_1: int = 0
    ask_depth_1: int = 0
    bid_depth_5: int = 0
    ask_depth_5: int = 0
    bid_depth_10: int = 0
    ask_depth_10: int = 0
    queue_imbalance: float = 0.0
    depth_imbalance_5: float = 0.0
    depth_imbalance_10: float = 0.0
    order_flow_imbalance: float = 0.0
    order_arrival_rate: float = 0.0
    buy_arrival_rate: float = 0.0
    sell_arrival_rate: float = 0.0
    market_order_rate: float = 0.0
    limit_order_rate: float = 0.0
    cancellation_rate: float = 0.0
    buy_cancel_rate: float = 0.0
    sell_cancel_rate: float = 0.0
    cancel_to_trade_ratio: float = 0.0
    lifetime_before_cancel: float = 0.0
    trade_rate: float = 0.0
    trade_volume: int = 0
    vwap: float = 0.0
    trade_imbalance: float = 0.0
    return_1s: float = 0.0
    return_5s: float = 0.0
    realized_volatility_1m: float = 0.0
    realized_volatility_5m: float = 0.0
    queue_replenish_rate: float = 0.0
    queue_deplete_rate: float = 0.0
    avg_order_size: float = 0.0
    median_order_size: float = 0.0
    large_order_ratio: float = 0.0
    sweep_count: int = 0
    sweep_volume: int = 0
    inter_arrival_mean: float = 0.0
    inter_arrival_std: float = 0.0
    trade_through_rate: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.mid_price, self.bid_ask_spread, self.spread_ticks,
            self.weighted_mid_price, self.micro_price,
            self.bid_depth_1, self.ask_depth_1, self.bid_depth_5, self.ask_depth_5,
            self.bid_depth_10, self.ask_depth_10,
            self.queue_imbalance, self.depth_imbalance_5, self.depth_imbalance_10,
            self.order_flow_imbalance,
            self.order_arrival_rate, self.buy_arrival_rate, self.sell_arrival_rate,
            self.market_order_rate, self.limit_order_rate,
            self.cancellation_rate, self.buy_cancel_rate, self.sell_cancel_rate,
            self.cancel_to_trade_ratio, self.lifetime_before_cancel,
            self.trade_rate, self.trade_volume, self.vwap, self.trade_imbalance,
            self.return_1s, self.return_5s,
            self.realized_volatility_1m, self.realized_volatility_5m,
            self.queue_replenish_rate, self.queue_deplete_rate,
            self.avg_order_size, self.median_order_size, self.large_order_ratio,
            self.sweep_count, self.sweep_volume,
            self.inter_arrival_mean, self.inter_arrival_std,
            self.trade_through_rate,
        ], dtype=np.float64)

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "mid_price", "bid_ask_spread", "spread_ticks",
            "weighted_mid_price", "micro_price",
            "bid_depth_1", "ask_depth_1", "bid_depth_5", "ask_depth_5",
            "bid_depth_10", "ask_depth_10",
            "queue_imbalance", "depth_imbalance_5", "depth_imbalance_10",
            "order_flow_imbalance",
            "order_arrival_rate", "buy_arrival_rate", "sell_arrival_rate",
            "market_order_rate", "limit_order_rate",
            "cancellation_rate", "buy_cancel_rate", "sell_cancel_rate",
            "cancel_to_trade_ratio", "lifetime_before_cancel",
            "trade_rate", "trade_volume", "vwap", "trade_imbalance",
            "return_1s", "return_5s",
            "realized_volatility_1m", "realized_volatility_5m",
            "queue_replenish_rate", "queue_deplete_rate",
            "avg_order_size", "median_order_size", "large_order_ratio",
            "sweep_count", "sweep_volume",
            "inter_arrival_mean", "inter_arrival_std",
            "trade_through_rate",
        ]

    @staticmethod
    def num_features() -> int:
        return len(MicrostructureFeatures.feature_names())

    def to_dict(self) -> dict:
        return {name: float(val) for name, val in zip(self.feature_names(), self.to_array())}


class MicrostructureFeatureExtractor:
    """Extract microstructure features from LOB state and order flow."""
    def __init__(self, window_seconds: float = 1.0, long_window_seconds: float = 60.0,
                 large_order_threshold: int = 1000, max_history: int = 100000):
        self.window_seconds = window_seconds
        self.long_window_seconds = long_window_seconds
        self.large_order_threshold = large_order_threshold
        self.max_history = max_history
        self._order_timestamps: deque = deque(maxlen=max_history)
        self._buy_order_timestamps: deque = deque(maxlen=max_history)
        self._sell_order_timestamps: deque = deque(maxlen=max_history)
        self._market_order_timestamps: deque = deque(maxlen=max_history)
        self._limit_order_timestamps: deque = deque(maxlen=max_history)
        self._cancel_timestamps: deque = deque(maxlen=max_history)
        self._buy_cancel_timestamps: deque = deque(maxlen=max_history)
        self._sell_cancel_timestamps: deque = deque(maxlen=max_history)
        self._trade_timestamps: deque = deque(maxlen=max_history)
        self._trade_prices: deque = deque(maxlen=max_history)
        self._trade_volumes: deque = deque(maxlen=max_history)
        self._trade_sides: deque = deque(maxlen=max_history)
        self._mid_prices: deque = deque(maxlen=max_history)
        self._mid_price_timestamps: deque = deque(maxlen=max_history)
        self._order_sizes: deque = deque(maxlen=max_history)
        self._cancel_lifetimes: deque = deque(maxlen=max_history)
        self._inter_arrival_times: deque = deque(maxlen=max_history)
        self._last_order_time: float = 0.0
        self._sweep_events: deque = deque(maxlen=max_history)

    def record_order(self, order: Order) -> None:
        ts = order.timestamp
        self._order_timestamps.append(ts)
        self._order_sizes.append(order.quantity)
        if order.is_buy:
            self._buy_order_timestamps.append(ts)
        else:
            self._sell_order_timestamps.append(ts)
        if order.order_type in (OrderType.MARKET, OrderType.IOC):
            self._market_order_timestamps.append(ts)
        else:
            self._limit_order_timestamps.append(ts)
        if self._last_order_time > 0:
            iat = ts - self._last_order_time
            if iat > 0:
                self._inter_arrival_times.append(iat)
        self._last_order_time = ts

    def record_cancel(self, order: Order, cancel_time: float) -> None:
        self._cancel_timestamps.append(cancel_time)
        if order.is_buy:
            self._buy_cancel_timestamps.append(cancel_time)
        else:
            self._sell_cancel_timestamps.append(cancel_time)
        lifetime = cancel_time - order.timestamp
        if lifetime > 0:
            self._cancel_lifetimes.append(lifetime)

    def record_trade(self, trade: Trade) -> None:
        self._trade_timestamps.append(trade.timestamp)
        self._trade_prices.append(trade.price)
        self._trade_volumes.append(trade.quantity)
        self._trade_sides.append(1 if trade.aggressor_side == OrderSide.BUY else -1)

    def record_mid_price(self, mid_price: float, timestamp: float) -> None:
        self._mid_prices.append(mid_price)
        self._mid_price_timestamps.append(timestamp)

    def _count_since(self, timestamps: deque, since: float) -> int:
        count = 0
        for ts in reversed(timestamps):
            if ts < since:
                break
            count += 1
        return count

    def _compute_return(self, timestamp: float, horizon: float) -> float:
        if len(self._mid_prices) < 2:
            return 0.0
        target_time = timestamp - horizon
        current_price = self._mid_prices[-1]
        past_price = current_price
        for t, p in zip(reversed(list(self._mid_price_timestamps)), reversed(list(self._mid_prices))):
            if t <= target_time:
                past_price = p
                break
        if past_price > 0 and current_price > 0:
            return float(np.log(current_price / past_price))
        return 0.0

    def _compute_realized_vol(self, timestamp: float, horizon: float) -> float:
        window_start = timestamp - horizon
        prices = list(self._mid_prices)
        times = list(self._mid_price_timestamps)
        window_prices = [p for t, p in zip(times, prices) if t >= window_start]
        if len(window_prices) < 2:
            return 0.0
        log_returns = np.diff(np.log(np.array(window_prices, dtype=np.float64)))
        log_returns = log_returns[np.isfinite(log_returns)]
        if len(log_returns) < 1:
            return 0.0
        return float(np.std(log_returns) * np.sqrt(len(log_returns)))

    def extract(self, book: LimitOrderBook, timestamp: float) -> MicrostructureFeatures:
        """Extract all features from current book state and recent history."""
        features = MicrostructureFeatures(timestamp=timestamp)
        if book.best_bid is not None and book.best_ask is not None:
            features.mid_price = book.mid_price or 0.0
            features.bid_ask_spread = book.spread or 0.0
            features.spread_ticks = book.spread_ticks or 0
            bid_size = book.get_depth(OrderSide.BUY, 1)
            ask_size = book.get_depth(OrderSide.SELL, 1)
            b1 = bid_size[0][1] if bid_size else 0
            a1 = ask_size[0][1] if ask_size else 0
            if b1 + a1 > 0:
                features.weighted_mid_price = (book.best_bid * a1 + book.best_ask * b1) / (b1 + a1)
                features.micro_price = features.weighted_mid_price
            else:
                features.weighted_mid_price = features.mid_price
                features.micro_price = features.mid_price
        bid_depths = book.get_depth(OrderSide.BUY, 10)
        ask_depths = book.get_depth(OrderSide.SELL, 10)
        features.bid_depth_1 = sum(q for _, q in bid_depths[:1])
        features.ask_depth_1 = sum(q for _, q in ask_depths[:1])
        features.bid_depth_5 = sum(q for _, q in bid_depths[:5])
        features.ask_depth_5 = sum(q for _, q in ask_depths[:5])
        features.bid_depth_10 = sum(q for _, q in bid_depths[:10])
        features.ask_depth_10 = sum(q for _, q in ask_depths[:10])
        b1, a1 = features.bid_depth_1, features.ask_depth_1
        features.queue_imbalance = (b1 - a1) / max(b1 + a1, 1)
        features.depth_imbalance_5 = (features.bid_depth_5 - features.ask_depth_5) / max(features.bid_depth_5 + features.ask_depth_5, 1)
        features.depth_imbalance_10 = (features.bid_depth_10 - features.ask_depth_10) / max(features.bid_depth_10 + features.ask_depth_10, 1)
        window_start = timestamp - self.window_seconds
        n_orders = self._count_since(self._order_timestamps, window_start)
        n_buys = self._count_since(self._buy_order_timestamps, window_start)
        n_sells = self._count_since(self._sell_order_timestamps, window_start)
        n_cancels = self._count_since(self._cancel_timestamps, window_start)
        features.order_arrival_rate = n_orders / max(self.window_seconds, 1e-9)
        features.buy_arrival_rate = n_buys / max(self.window_seconds, 1e-9)
        features.sell_arrival_rate = n_sells / max(self.window_seconds, 1e-9)
        features.cancellation_rate = n_cancels / max(n_orders, 1)
        features.order_flow_imbalance = (n_buys - n_sells) / max(n_buys + n_sells, 1)
        features.return_1s = self._compute_return(timestamp, 1.0)
        features.return_5s = self._compute_return(timestamp, 5.0)
        features.realized_volatility_1m = self._compute_realized_vol(timestamp, 60.0)
        features.realized_volatility_5m = self._compute_realized_vol(timestamp, 300.0)
        if self._order_sizes:
            recent_sizes = list(self._order_sizes)[-100:]
            features.avg_order_size = float(np.mean(recent_sizes))
            features.median_order_size = float(np.median(recent_sizes))
            features.large_order_ratio = sum(1 for s in recent_sizes if s >= self.large_order_threshold) / len(recent_sizes)
        if self._inter_arrival_times:
            recent_iats = list(self._inter_arrival_times)[-100:]
            features.inter_arrival_mean = float(np.mean(recent_iats))
            features.inter_arrival_std = float(np.std(recent_iats))
        if features.mid_price > 0:
            self.record_mid_price(features.mid_price, timestamp)
        return features

    def reset(self) -> None:
        self._order_timestamps.clear()
        self._buy_order_timestamps.clear()
        self._sell_order_timestamps.clear()
        self._market_order_timestamps.clear()
        self._limit_order_timestamps.clear()
        self._cancel_timestamps.clear()
        self._buy_cancel_timestamps.clear()
        self._sell_cancel_timestamps.clear()
        self._trade_timestamps.clear()
        self._trade_prices.clear()
        self._trade_volumes.clear()
        self._trade_sides.clear()
        self._mid_prices.clear()
        self._mid_price_timestamps.clear()
        self._order_sizes.clear()
        self._cancel_lifetimes.clear()
        self._inter_arrival_times.clear()
        self._last_order_time = 0.0
        self._sweep_events.clear()
''')
