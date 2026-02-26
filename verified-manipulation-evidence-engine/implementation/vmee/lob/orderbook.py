"""
Limit order book implementation with price-time priority.

Implements a full-featured limit order book supporting:
- Limit orders with price-time priority
- Market orders with immediate execution
- Iceberg (hidden) orders
- Order cancellation and modification
- Level-2 market data snapshots
- Order queue position tracking
"""

from __future__ import annotations

import bisect
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Deque, Optional


class OrderSide(Enum):
    """Order side: buy or sell."""
    BUY = auto()
    SELL = auto()

    @property
    def opposite(self) -> OrderSide:
        return OrderSide.SELL if self == OrderSide.BUY else OrderSide.BUY


class OrderType(Enum):
    """Order type."""
    LIMIT = auto()
    MARKET = auto()
    ICEBERG = auto()
    IOC = auto()
    FOK = auto()
    GTC = auto()
    STOP_LIMIT = auto()


class OrderStatus(Enum):
    """Order lifecycle status."""
    NEW = auto()
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()
    EXPIRED = auto()


@dataclass
class Order:
    """Represents a single order in the book."""
    order_id: str
    trader_id: str
    side: OrderSide
    price: float
    quantity: int
    order_type: OrderType = OrderType.LIMIT
    timestamp: float = 0.0
    filled_quantity: int = 0
    status: OrderStatus = OrderStatus.NEW
    iceberg_visible: int = 0
    iceberg_total: int = 0
    stop_price: float = 0.0
    time_in_force: str = "GTC"
    instrument_id: str = ""
    sequence_number: int = 0
    cancel_timestamp: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED)

    @property
    def is_buy(self) -> bool:
        return self.side == OrderSide.BUY

    @property
    def is_sell(self) -> bool:
        return self.side == OrderSide.SELL

    def fill(self, qty: int) -> int:
        """Fill the order by qty shares. Returns actual filled amount."""
        actual = min(qty, self.remaining_quantity)
        self.filled_quantity += actual
        if self.remaining_quantity == 0:
            self.status = OrderStatus.FILLED
        elif actual > 0:
            self.status = OrderStatus.PARTIALLY_FILLED
        return actual

    def cancel(self, timestamp: float = 0.0) -> None:
        """Cancel this order."""
        self.status = OrderStatus.CANCELLED
        self.cancel_timestamp = timestamp


@dataclass
class Trade:
    """Represents an executed trade."""
    trade_id: str
    instrument_id: str
    price: float
    quantity: int
    aggressor_side: OrderSide
    buyer_order_id: str
    seller_order_id: str
    buyer_trader_id: str
    seller_trader_id: str
    timestamp: float
    sequence_number: int
    is_wash: bool = False

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "instrument_id": self.instrument_id,
            "price": self.price,
            "quantity": self.quantity,
            "aggressor_side": self.aggressor_side.name,
            "buyer_order_id": self.buyer_order_id,
            "seller_order_id": self.seller_order_id,
            "timestamp": self.timestamp,
            "is_wash": self.is_wash,
        }


@dataclass
class PriceLevel:
    """A single price level in the order book containing a FIFO queue of orders."""
    price: float
    orders: Deque[Order] = field(default_factory=deque)
    _total_quantity: int = 0
    _order_count: int = 0

    @property
    def total_quantity(self) -> int:
        return self._total_quantity

    @property
    def order_count(self) -> int:
        return self._order_count

    @property
    def is_empty(self) -> bool:
        return self._order_count == 0

    def add_order(self, order: Order) -> None:
        """Add an order to the back of the queue (price-time priority)."""
        self.orders.append(order)
        self._total_quantity += order.remaining_quantity
        self._order_count += 1

    def remove_order(self, order_id: str) -> Optional[Order]:
        """Remove a specific order by ID."""
        for i, order in enumerate(self.orders):
            if order.order_id == order_id:
                removed = self.orders[i]
                del self.orders[i]
                self._total_quantity -= removed.remaining_quantity
                self._order_count -= 1
                return removed
        return None

    def peek_front(self) -> Optional[Order]:
        """Look at the front order without removing it."""
        if self.orders:
            return self.orders[0]
        return None

    def pop_front(self) -> Optional[Order]:
        """Remove and return the front order."""
        if self.orders:
            order = self.orders.popleft()
            self._total_quantity -= order.remaining_quantity
            self._order_count -= 1
            return order
        return None

    def update_quantity(self, delta: int) -> None:
        """Update total quantity after a partial fill."""
        self._total_quantity += delta


@dataclass
class BookSnapshot:
    """A point-in-time snapshot of the order book state."""
    timestamp: float
    instrument_id: str
    bids: list[tuple[float, int, int]]
    asks: list[tuple[float, int, int]]
    last_trade_price: Optional[float] = None
    last_trade_quantity: Optional[int] = None
    sequence_number: int = 0

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2.0
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def best_bid_size(self) -> int:
        return self.bids[0][1] if self.bids else 0

    @property
    def best_ask_size(self) -> int:
        return self.asks[0][1] if self.asks else 0

    def depth(self, levels: int = 5) -> dict:
        """Return depth information for the top N levels."""
        bid_depth = sum(q for _, q, _ in self.bids[:levels])
        ask_depth = sum(q for _, q, _ in self.asks[:levels])
        return {
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "total_depth": bid_depth + ask_depth,
            "imbalance": (bid_depth - ask_depth) / max(bid_depth + ask_depth, 1),
        }

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "instrument_id": self.instrument_id,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "mid_price": self.mid_price,
            "spread": self.spread,
            "bids": self.bids[:5],
            "asks": self.asks[:5],
        }


class LimitOrderBook:
    """
    Full limit order book implementation with price-time priority.

    Supports limit orders, market orders, iceberg orders, IOC/FOK orders,
    order cancellation, and modification. Maintains both bid and ask sides
    with efficient price level management.
    """

    def __init__(
        self,
        instrument_id: str = "TEST",
        tick_size: float = 0.01,
        lot_size: int = 1,
        max_price_levels: int = 1000,
    ):
        self.instrument_id = instrument_id
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.max_price_levels = max_price_levels

        self._bid_levels: dict[float, PriceLevel] = {}
        self._ask_levels: dict[float, PriceLevel] = {}
        self._bid_prices: list[float] = []
        self._ask_prices: list[float] = []
        self._orders: dict[str, Order] = {}
        self._order_to_level: dict[str, float] = {}
        self._trades: list[Trade] = []
        self._trade_count: int = 0
        self._sequence_number: int = 0
        self._message_count: int = 0
        self._last_trade_price: Optional[float] = None
        self._last_trade_quantity: Optional[int] = None
        self._trade_callbacks: list = []
        self._book_update_callbacks: list = []
        self._snapshots: list[BookSnapshot] = []

    @property
    def best_bid(self) -> Optional[float]:
        return self._bid_prices[0] if self._bid_prices else None

    @property
    def best_ask(self) -> Optional[float]:
        return self._ask_prices[0] if self._ask_prices else None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2.0
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def spread_ticks(self) -> Optional[int]:
        if self.spread is not None:
            return round(self.spread / self.tick_size)
        return None

    def _round_price(self, price: float) -> float:
        return round(round(price / self.tick_size) * self.tick_size, 10)

    def _add_bid_price(self, price: float) -> None:
        idx = bisect.bisect_left([-p for p in self._bid_prices], -price)
        self._bid_prices.insert(idx, price)

    def _remove_bid_price(self, price: float) -> None:
        idx = bisect.bisect_left([-p for p in self._bid_prices], -price)
        if idx < len(self._bid_prices) and abs(self._bid_prices[idx] - price) < 1e-12:
            self._bid_prices.pop(idx)

    def _add_ask_price(self, price: float) -> None:
        idx = bisect.bisect_left(self._ask_prices, price)
        self._ask_prices.insert(idx, price)

    def _remove_ask_price(self, price: float) -> None:
        idx = bisect.bisect_left(self._ask_prices, price)
        if idx < len(self._ask_prices) and abs(self._ask_prices[idx] - price) < 1e-12:
            self._ask_prices.pop(idx)

    def add_order(self, order: Order) -> list[Trade]:
        """Add an order to the book. Returns list of trades generated."""
        self._sequence_number += 1
        self._message_count += 1
        order.sequence_number = self._sequence_number
        order.instrument_id = self.instrument_id

        if order.order_type == OrderType.MARKET:
            return self._process_market_order(order)
        elif order.order_type == OrderType.IOC:
            return self._process_ioc_order(order)
        elif order.order_type == OrderType.FOK:
            return self._process_fok_order(order)
        else:
            return self._process_limit_order(order)

    def _process_limit_order(self, order: Order) -> list[Trade]:
        trades = self._match_order(order)
        if order.remaining_quantity > 0:
            self._rest_order(order)
        return trades

    def _process_market_order(self, order: Order) -> list[Trade]:
        return self._match_order(order)

    def _process_ioc_order(self, order: Order) -> list[Trade]:
        trades = self._match_order(order)
        if order.remaining_quantity > 0:
            order.cancel()
        return trades

    def _process_fok_order(self, order: Order) -> list[Trade]:
        available = self._available_liquidity(order.side.opposite, order.price, order.quantity)
        if available >= order.quantity:
            return self._match_order(order)
        else:
            order.cancel()
            return []

    def _available_liquidity(self, side: OrderSide, price: float, max_qty: int) -> int:
        total = 0
        if side == OrderSide.BUY:
            for p in self._bid_prices:
                if p < price:
                    break
                level = self._bid_levels[p]
                total += level.total_quantity
                if total >= max_qty:
                    return total
        else:
            for p in self._ask_prices:
                if p > price:
                    break
                level = self._ask_levels[p]
                total += level.total_quantity
                if total >= max_qty:
                    return total
        return total

    def _match_order(self, order: Order) -> list[Trade]:
        """Match an incoming order against the opposite side of the book."""
        trades = []

        if order.is_buy:
            prices = self._ask_prices
            levels = self._ask_levels
            can_match = lambda p: (order.order_type == OrderType.MARKET or p <= order.price)
        else:
            prices = self._bid_prices
            levels = self._bid_levels
            can_match = lambda p: (order.order_type == OrderType.MARKET or p >= order.price)

        prices_to_remove = []

        for price in list(prices):
            if order.remaining_quantity <= 0:
                break
            if not can_match(price):
                break

            level = levels[price]

            while level.orders and order.remaining_quantity > 0:
                resting = level.orders[0]
                trade_qty = min(order.remaining_quantity, resting.remaining_quantity)

                order.fill(trade_qty)
                resting.fill(trade_qty)
                level.update_quantity(-trade_qty)

                self._trade_count += 1
                trade = Trade(
                    trade_id=f"T{self._trade_count:010d}",
                    instrument_id=self.instrument_id,
                    price=price,
                    quantity=trade_qty,
                    aggressor_side=order.side,
                    buyer_order_id=order.order_id if order.is_buy else resting.order_id,
                    seller_order_id=order.order_id if order.is_sell else resting.order_id,
                    buyer_trader_id=order.trader_id if order.is_buy else resting.trader_id,
                    seller_trader_id=order.trader_id if order.is_sell else resting.trader_id,
                    timestamp=order.timestamp,
                    sequence_number=self._sequence_number,
                )
                trades.append(trade)
                self._trades.append(trade)
                self._last_trade_price = price
                self._last_trade_quantity = trade_qty

                if resting.remaining_quantity == 0:
                    level.pop_front()
                    self._orders.pop(resting.order_id, None)
                    self._order_to_level.pop(resting.order_id, None)

            if level.is_empty:
                prices_to_remove.append(price)

        for price in prices_to_remove:
            if order.is_buy:
                del self._ask_levels[price]
                self._remove_ask_price(price)
            else:
                del self._bid_levels[price]
                self._remove_bid_price(price)

        for trade in trades:
            for callback in self._trade_callbacks:
                callback(trade)

        return trades

    def _rest_order(self, order: Order) -> None:
        price = self._round_price(order.price)
        if order.is_buy:
            if price not in self._bid_levels:
                self._bid_levels[price] = PriceLevel(price=price)
                self._add_bid_price(price)
            self._bid_levels[price].add_order(order)
        else:
            if price not in self._ask_levels:
                self._ask_levels[price] = PriceLevel(price=price)
                self._add_ask_price(price)
            self._ask_levels[price].add_order(order)

        self._orders[order.order_id] = order
        self._order_to_level[order.order_id] = price

        for callback in self._book_update_callbacks:
            callback("add", order)

    def cancel_order(self, order_id: str, timestamp: float = 0.0) -> Optional[Order]:
        """Cancel an order by ID."""
        if order_id not in self._orders:
            return None
        order = self._orders[order_id]
        if not order.is_active:
            return None
        price = self._order_to_level.get(order_id)
        if price is None:
            return None

        order.cancel(timestamp)
        self._message_count += 1
        self._sequence_number += 1

        if order.is_buy and price in self._bid_levels:
            level = self._bid_levels[price]
            level.remove_order(order_id)
            if level.is_empty:
                del self._bid_levels[price]
                self._remove_bid_price(price)
        elif order.is_sell and price in self._ask_levels:
            level = self._ask_levels[price]
            level.remove_order(order_id)
            if level.is_empty:
                del self._ask_levels[price]
                self._remove_ask_price(price)

        del self._orders[order_id]
        del self._order_to_level[order_id]

        for callback in self._book_update_callbacks:
            callback("cancel", order)

        return order

    def modify_order(
        self,
        order_id: str,
        new_price: Optional[float] = None,
        new_quantity: Optional[int] = None,
        timestamp: float = 0.0,
    ) -> tuple[Optional[Order], list[Trade]]:
        """Modify an existing order. Price changes lose time priority."""
        if order_id not in self._orders:
            return None, []
        old_order = self._orders[order_id]
        if not old_order.is_active:
            return None, []

        price_changed = new_price is not None and abs(new_price - old_order.price) > 1e-12
        qty_increased = new_quantity is not None and new_quantity > old_order.remaining_quantity

        if price_changed or qty_increased:
            self.cancel_order(order_id, timestamp)
            new_order = Order(
                order_id=f"{order_id}_mod",
                trader_id=old_order.trader_id,
                side=old_order.side,
                price=new_price if new_price is not None else old_order.price,
                quantity=new_quantity if new_quantity is not None else old_order.remaining_quantity,
                order_type=old_order.order_type,
                timestamp=timestamp,
                instrument_id=old_order.instrument_id,
            )
            trades = self.add_order(new_order)
            return new_order, trades
        elif new_quantity is not None:
            decrease = old_order.remaining_quantity - new_quantity
            if decrease > 0:
                old_order.quantity -= decrease
                price = self._order_to_level.get(order_id)
                if price is not None:
                    if old_order.is_buy and price in self._bid_levels:
                        self._bid_levels[price].update_quantity(-decrease)
                    elif old_order.is_sell and price in self._ask_levels:
                        self._ask_levels[price].update_quantity(-decrease)
            return old_order, []
        return old_order, []

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def snapshot(self, levels: int = 10, timestamp: float = 0.0) -> BookSnapshot:
        """Take a snapshot of the current book state."""
        bids = []
        for price in self._bid_prices[:levels]:
            level = self._bid_levels[price]
            bids.append((price, level.total_quantity, level.order_count))

        asks = []
        for price in self._ask_prices[:levels]:
            level = self._ask_levels[price]
            asks.append((price, level.total_quantity, level.order_count))

        snap = BookSnapshot(
            timestamp=timestamp,
            instrument_id=self.instrument_id,
            bids=bids,
            asks=asks,
            last_trade_price=self._last_trade_price,
            last_trade_quantity=self._last_trade_quantity,
            sequence_number=self._sequence_number,
        )
        self._snapshots.append(snap)
        return snap

    def get_depth(self, side: OrderSide, levels: int = 10) -> list[tuple[float, int]]:
        if side == OrderSide.BUY:
            return [(p, self._bid_levels[p].total_quantity) for p in self._bid_prices[:levels]]
        else:
            return [(p, self._ask_levels[p].total_quantity) for p in self._ask_prices[:levels]]

    def total_bid_volume(self) -> int:
        return sum(level.total_quantity for level in self._bid_levels.values())

    def total_ask_volume(self) -> int:
        return sum(level.total_quantity for level in self._ask_levels.values())

    def num_bid_levels(self) -> int:
        return len(self._bid_prices)

    def num_ask_levels(self) -> int:
        return len(self._ask_prices)

    def num_orders(self) -> int:
        return len(self._orders)

    def trade_history(self, last_n: Optional[int] = None) -> list[Trade]:
        if last_n is not None:
            return self._trades[-last_n:]
        return list(self._trades)

    def register_trade_callback(self, callback) -> None:
        self._trade_callbacks.append(callback)

    def register_book_update_callback(self, callback) -> None:
        self._book_update_callbacks.append(callback)

    def clear(self) -> None:
        self._bid_levels.clear()
        self._ask_levels.clear()
        self._bid_prices.clear()
        self._ask_prices.clear()
        self._orders.clear()
        self._order_to_level.clear()

    def vwap(self, side: OrderSide, quantity: int) -> Optional[float]:
        """Calculate VWAP for sweeping quantity shares from one side."""
        total_cost = 0.0
        remaining = quantity
        if side == OrderSide.BUY:
            prices = self._ask_prices
            levels = self._ask_levels
        else:
            prices = self._bid_prices
            levels = self._bid_levels

        for price in prices:
            level = levels[price]
            fill = min(remaining, level.total_quantity)
            total_cost += fill * price
            remaining -= fill
            if remaining <= 0:
                break

        if remaining > 0:
            return None
        return total_cost / quantity

    def queue_position(self, order_id: str) -> Optional[int]:
        if order_id not in self._orders:
            return None
        order = self._orders[order_id]
        price = self._order_to_level.get(order_id)
        if price is None:
            return None

        if order.is_buy and price in self._bid_levels:
            level = self._bid_levels[price]
        elif order.is_sell and price in self._ask_levels:
            level = self._ask_levels[price]
        else:
            return None

        for i, o in enumerate(level.orders):
            if o.order_id == order_id:
                return i
        return None

    def __repr__(self) -> str:
        return (
            f"LimitOrderBook(instrument={self.instrument_id}, "
            f"bid={self.best_bid}, ask={self.best_ask}, "
            f"orders={self.num_orders()}, trades={self._trade_count})"
        )
