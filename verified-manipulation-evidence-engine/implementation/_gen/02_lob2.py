# Generate LOB simulator and manipulation modules

total += w("vmee/lob/simulator.py", '''\
"""
LOB Simulator for generating synthetic market data.
"""
from __future__ import annotations
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
from vmee.config import LOBConfig
from vmee.lob.orderbook import LimitOrderBook, Order, OrderSide, OrderType, Trade, BookSnapshot
from vmee.lob.matching import MatchingEngine
from vmee.lob.features import MicrostructureFeatureExtractor, MicrostructureFeatures

logger = logging.getLogger(__name__)


@dataclass
class TraderProfile:
    """Profile defining a trader's behavior parameters."""
    trader_id: str
    trader_type: str
    order_rate: float = 1.0
    cancel_rate: float = 0.3
    market_order_fraction: float = 0.1
    spread_preference: float = 1.0
    size_mean: int = 100
    size_std: int = 50
    min_size: int = 1
    max_size: int = 10000
    momentum_sensitivity: float = 0.0
    inventory_limit: int = 10000
    is_manipulator: bool = False
    manipulation_type: str = ""


@dataclass
class MarketData:
    """Container for generated market data."""
    instrument_id: str
    snapshots: list[BookSnapshot] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    orders: list[Order] = field(default_factory=list)
    cancels: list[tuple[str, float]] = field(default_factory=list)
    features: list[MicrostructureFeatures] = field(default_factory=list)
    trader_profiles: list[TraderProfile] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    ground_truth_labels: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "instrument_id": self.instrument_id,
            "num_snapshots": len(self.snapshots),
            "num_trades": len(self.trades),
            "num_orders": len(self.orders),
            "num_features": len(self.features),
            "metadata": self.metadata,
            "ground_truth_labels": self.ground_truth_labels,
        }

    def feature_matrix(self) -> np.ndarray:
        if not self.features:
            return np.empty((0, 0))
        return np.vstack([f.to_array() for f in self.features])

    @property
    def feature_names(self) -> list[str]:
        return MicrostructureFeatures.feature_names()

    @property
    def duration(self) -> float:
        if not self.snapshots:
            return 0.0
        return self.snapshots[-1].timestamp - self.snapshots[0].timestamp


class OrderFlowGenerator:
    """Generates realistic order flow for the LOB simulator."""
    def __init__(self, config: LOBConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.RandomState(seed)
        self._order_counter = 0

    def generate_trader_population(self, num_traders: int = 50) -> list[TraderProfile]:
        traders = []
        n_mm = max(1, int(num_traders * 0.1))
        for i in range(n_mm):
            traders.append(TraderProfile(
                trader_id=f"MM_{i:03d}", trader_type="market_maker",
                order_rate=5.0 + self.rng.exponential(2.0),
                cancel_rate=0.7 + self.rng.uniform(0, 0.2),
                market_order_fraction=0.02,
                spread_preference=1.0 + self.rng.exponential(0.5),
                size_mean=200 + self.rng.randint(0, 300), size_std=50,
                min_size=100, max_size=5000, inventory_limit=50000,
            ))
        n_informed = max(1, int(num_traders * 0.1))
        for i in range(n_informed):
            traders.append(TraderProfile(
                trader_id=f"INF_{i:03d}", trader_type="informed",
                order_rate=0.5 + self.rng.exponential(0.3),
                cancel_rate=0.1, market_order_fraction=0.5 + self.rng.uniform(0, 0.3),
                spread_preference=0.0,
                size_mean=500 + self.rng.randint(0, 500), size_std=200,
                min_size=100, max_size=10000,
                momentum_sensitivity=0.5 + self.rng.uniform(0, 0.5),
            ))
        n_noise = max(1, int(num_traders * 0.6))
        for i in range(n_noise):
            traders.append(TraderProfile(
                trader_id=f"NOISE_{i:03d}", trader_type="noise",
                order_rate=0.2 + self.rng.exponential(0.5),
                cancel_rate=0.2 + self.rng.uniform(0, 0.3),
                market_order_fraction=0.2 + self.rng.uniform(0, 0.3),
                spread_preference=2.0 + self.rng.exponential(3.0),
                size_mean=100 + self.rng.randint(0, 200), size_std=50,
                min_size=1, max_size=5000,
            ))
        n_hft = num_traders - n_mm - n_informed - n_noise
        for i in range(max(0, n_hft)):
            traders.append(TraderProfile(
                trader_id=f"HFT_{i:03d}", trader_type="hft",
                order_rate=10.0 + self.rng.exponential(5.0),
                cancel_rate=0.8 + self.rng.uniform(0, 0.15),
                market_order_fraction=0.05,
                spread_preference=0.5 + self.rng.exponential(0.3),
                size_mean=50 + self.rng.randint(0, 100), size_std=20,
                min_size=1, max_size=1000, inventory_limit=5000,
            ))
        return traders

    def generate_order(self, trader: TraderProfile, book: LimitOrderBook,
                       timestamp: float) -> Optional[Order]:
        mid = book.mid_price
        if mid is None:
            mid = self.config.initial_mid_price
        self._order_counter += 1
        is_market = self.rng.random() < trader.market_order_fraction
        order_type = OrderType.MARKET if is_market else OrderType.LIMIT
        if trader.trader_type == "market_maker":
            side = OrderSide.BUY if self.rng.random() < 0.5 else OrderSide.SELL
        elif trader.trader_type == "informed":
            bias = 0.5 + trader.momentum_sensitivity * 0.1
            side = OrderSide.BUY if self.rng.random() < bias else OrderSide.SELL
        else:
            side = OrderSide.BUY if self.rng.random() < 0.5 else OrderSide.SELL
        if is_market:
            price = mid
        else:
            offset = self.rng.exponential(trader.spread_preference) * self.config.tick_size
            if side == OrderSide.BUY:
                price = mid - offset
            else:
                price = mid + offset
            price = round(price / self.config.tick_size) * self.config.tick_size
        size = max(trader.min_size, min(trader.max_size,
            int(self.rng.lognormal(np.log(max(trader.size_mean, 1)),
                np.log(1 + trader.size_std / max(trader.size_mean, 1))))))
        size = max(1, (size // max(self.config.lot_size, 1)) * max(self.config.lot_size, 1))
        return Order(
            order_id=f"O{self._order_counter:010d}", trader_id=trader.trader_id,
            side=side, price=price, quantity=size, order_type=order_type, timestamp=timestamp,
        )


class LOBSimulator:
    """Full LOB simulator generating synthetic market data."""
    def __init__(self, config: LOBConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self._flow_gen = OrderFlowGenerator(config, seed=config.seed)

    def generate_trading_day(self, instrument_id: str = "SYM001",
                             num_traders: int = 50) -> MarketData:
        logger.info(f"Generating trading day for {instrument_id}")
        book = LimitOrderBook(instrument_id=instrument_id, tick_size=self.config.tick_size,
                              lot_size=self.config.lot_size)
        extractor = MicrostructureFeatureExtractor()
        traders = self._flow_gen.generate_trader_population(num_traders)
        self._initialize_book(book, traders)
        market_data = MarketData(
            instrument_id=instrument_id, trader_profiles=traders,
            metadata={"tick_size": self.config.tick_size, "lot_size": self.config.lot_size,
                      "initial_mid_price": self.config.initial_mid_price, "num_traders": num_traders}
        )
        total_events = min(self.config.total_events(), 100000)
        duration = self.config.trading_hours * 3600
        dt = duration / max(total_events, 1)
        active_orders: dict[str, tuple[Order, TraderProfile]] = {}
        current_time = 0.0
        snapshot_interval = max(1.0, duration / 10000)
        next_snapshot = snapshot_interval
        for event_idx in range(total_events):
            current_time += self.rng.exponential(dt)
            rates = np.array([t.order_rate for t in traders])
            probs = rates / rates.sum()
            trader_idx = self.rng.choice(len(traders), p=probs)
            trader = traders[trader_idx]
            action_roll = self.rng.random()
            if action_roll < 0.6:
                order = self._flow_gen.generate_order(trader, book, current_time)
                if order is not None:
                    trades = book.add_order(order)
                    extractor.record_order(order)
                    market_data.orders.append(order)
                    for trade in trades:
                        extractor.record_trade(trade)
                        market_data.trades.append(trade)
                    if order.is_active:
                        active_orders[order.order_id] = (order, trader)
            elif action_roll < 0.85 and active_orders:
                order_ids = list(active_orders.keys())
                cancel_id = self.rng.choice(order_ids)
                order, _ = active_orders[cancel_id]
                cancelled = book.cancel_order(cancel_id, current_time)
                if cancelled:
                    extractor.record_cancel(cancelled, current_time)
                    market_data.cancels.append((cancel_id, current_time))
                    del active_orders[cancel_id]
            elif active_orders:
                order_ids = list(active_orders.keys())
                mod_id = self.rng.choice(order_ids)
                order, _ = active_orders[mod_id]
                new_qty = max(1, int(order.remaining_quantity * self.rng.uniform(0.5, 1.5)))
                book.modify_order(mod_id, new_quantity=new_qty, timestamp=current_time)
            if current_time >= next_snapshot:
                snap = book.snapshot(timestamp=current_time)
                market_data.snapshots.append(snap)
                features = extractor.extract(book, current_time)
                market_data.features.append(features)
                next_snapshot = current_time + snapshot_interval
            filled_ids = [oid for oid, (o, _) in active_orders.items() if not o.is_active]
            for oid in filled_ids:
                del active_orders[oid]
        snap = book.snapshot(timestamp=current_time)
        market_data.snapshots.append(snap)
        features = extractor.extract(book, current_time)
        market_data.features.append(features)
        logger.info(f"Generated {len(market_data.orders)} orders, {len(market_data.trades)} trades")
        return market_data

    def _initialize_book(self, book: LimitOrderBook, traders: list[TraderProfile]) -> None:
        mid = self.config.initial_mid_price
        tick = self.config.tick_size
        depth = self.config.initial_depth_per_level
        mm_traders = [t for t in traders if t.trader_type == "market_maker"]
        if not mm_traders:
            mm_traders = traders[:1]
        for level in range(1, min(self.config.initial_spread + 10, 20)):
            for side in [OrderSide.BUY, OrderSide.SELL]:
                price = mid - level * tick if side == OrderSide.BUY else mid + level * tick
                remaining = depth
                while remaining > 0:
                    trader = self.rng.choice(mm_traders)
                    size = min(remaining, max(1, int(self.rng.exponential(100))))
                    size = max(1, size)
                    order = Order(
                        order_id=f"INIT_{side.name}_{level}_{remaining}",
                        trader_id=trader.trader_id, side=side,
                        price=round(price, 10), quantity=size,
                        order_type=OrderType.LIMIT, timestamp=0.0,
                    )
                    book.add_order(order)
                    remaining -= size

    def generate_multi_instrument(self, num_instruments: int = 10,
                                  num_traders: int = 50) -> dict[str, MarketData]:
        results = {}
        for i in range(num_instruments):
            inst_id = f"SYM{i:03d}"
            results[inst_id] = self.generate_trading_day(instrument_id=inst_id, num_traders=num_traders)
        return results
''')

total += w("vmee/lob/manipulation.py", '''\
"""
Manipulation scenario planting for synthetic market data.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from vmee.lob.orderbook import LimitOrderBook, Order, OrderSide, OrderType, Trade, BookSnapshot
from vmee.lob.simulator import MarketData, TraderProfile

logger = logging.getLogger(__name__)


@dataclass
class ManipulationEvent:
    """Ground truth label for a planted manipulation event."""
    event_id: str
    manipulation_type: str
    subtype: str
    trader_id: str
    start_time: float
    end_time: float
    orders: list[str] = field(default_factory=list)
    trades: list[str] = field(default_factory=list)
    intended_direction: str = ""
    profit_target: float = 0.0
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id, "manipulation_type": self.manipulation_type,
            "subtype": self.subtype, "trader_id": self.trader_id,
            "start_time": self.start_time, "end_time": self.end_time,
            "num_orders": len(self.orders), "num_trades": len(self.trades),
            "intended_direction": self.intended_direction, "description": self.description,
        }


class ManipulationPlanter:
    """Plants manipulation scenarios into synthetic market data."""
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self._event_counter = 0

    def _new_event_id(self) -> str:
        self._event_counter += 1
        return f"MANIP_{self._event_counter:06d}"

    def _find_nearest_snapshot(self, data: MarketData, target_time: float) -> Optional[BookSnapshot]:
        if not data.snapshots:
            return None
        best, best_dist = None, float("inf")
        for snap in data.snapshots:
            dist = abs(snap.timestamp - target_time)
            if dist < best_dist:
                best_dist, best = dist, snap
        return best

    def plant_spoofing(self, market_data: MarketData, num_events: int = 5,
                       subtype: str = "basic") -> MarketData:
        if not market_data.snapshots:
            return market_data
        duration = market_data.duration
        manipulator_id = f"SPOOFER_{self.rng.randint(1000)}"
        for _ in range(num_events):
            start_time = self.rng.uniform(duration * 0.1, duration * 0.9)
            event = self._plant_spoofing_event(market_data, manipulator_id, start_time, subtype)
            if event:
                market_data.ground_truth_labels.append(event.to_dict())
        return market_data

    def _plant_spoofing_event(self, data: MarketData, trader_id: str,
                               start_time: float, subtype: str) -> ManipulationEvent:
        event = ManipulationEvent(
            event_id=self._new_event_id(), manipulation_type="spoofing",
            subtype=subtype, trader_id=trader_id, start_time=start_time,
            end_time=start_time + 5.0, intended_direction="up",
            description=f"Spoofing event ({subtype})",
        )
        snap = self._find_nearest_snapshot(data, start_time)
        if snap is None or snap.mid_price is None:
            return event
        mid = snap.mid_price
        num_layers = 1 if subtype == "basic" else self.rng.randint(3, 7)
        for i in range(num_layers):
            spoof_size = self.rng.randint(5000, 20000)
            spoof_price = mid + (i + 1) * 0.01
            spoof_order = Order(
                order_id=f"SPOOF_{event.event_id}_{i}", trader_id=trader_id,
                side=OrderSide.SELL, price=round(spoof_price, 2), quantity=spoof_size,
                order_type=OrderType.LIMIT, timestamp=start_time + i * 0.1,
            )
            data.orders.append(spoof_order)
            event.orders.append(spoof_order.order_id)
        cancel_time = start_time + self.rng.uniform(0.5, 3.0)
        for i in range(num_layers):
            data.cancels.append((f"SPOOF_{event.event_id}_{i}", cancel_time + i * 0.05))
        trade_time = cancel_time + self.rng.uniform(0.1, 0.5)
        real_order = Order(
            order_id=f"SPOOF_{event.event_id}_real", trader_id=trader_id,
            side=OrderSide.BUY, price=mid, quantity=self.rng.randint(100, 1000),
            order_type=OrderType.MARKET, timestamp=trade_time,
        )
        data.orders.append(real_order)
        event.orders.append(real_order.order_id)
        event.end_time = trade_time
        return event

    def plant_layering(self, market_data: MarketData, num_events: int = 5,
                       subtype: str = "basic") -> MarketData:
        if not market_data.snapshots:
            return market_data
        duration = market_data.duration
        manipulator_id = f"LAYERER_{self.rng.randint(1000)}"
        for _ in range(num_events):
            start_time = self.rng.uniform(duration * 0.1, duration * 0.9)
            event = self._plant_layering_event(market_data, manipulator_id, start_time, subtype)
            if event:
                market_data.ground_truth_labels.append(event.to_dict())
        return market_data

    def _plant_layering_event(self, data: MarketData, trader_id: str,
                               start_time: float, subtype: str) -> ManipulationEvent:
        event = ManipulationEvent(
            event_id=self._new_event_id(), manipulation_type="layering",
            subtype=subtype, trader_id=trader_id, start_time=start_time,
            end_time=start_time + 8.0, intended_direction="down",
        )
        snap = self._find_nearest_snapshot(data, start_time)
        if snap is None or snap.mid_price is None:
            return event
        mid = snap.mid_price
        num_layers = self.rng.randint(5, 12)
        sizes = sorted([self.rng.randint(1000, 8000) for _ in range(num_layers)], reverse=True)
        for i in range(num_layers):
            price = mid + (i + 1) * 0.01
            order = Order(
                order_id=f"LAYERING_{event.event_id}_{i}", trader_id=trader_id,
                side=OrderSide.SELL, price=round(price, 2), quantity=sizes[i],
                order_type=OrderType.LIMIT, timestamp=start_time + i * 0.2,
            )
            data.orders.append(order)
            event.orders.append(order.order_id)
        cancel_start = start_time + self.rng.uniform(3.0, 6.0)
        for i in range(num_layers - 1, -1, -1):
            data.cancels.append((f"LAYERING_{event.event_id}_{i}",
                                 cancel_start + (num_layers - 1 - i) * 0.1))
        real_order = Order(
            order_id=f"LAYERING_{event.event_id}_real", trader_id=trader_id,
            side=OrderSide.BUY, price=mid - 0.03,
            quantity=self.rng.randint(500, 3000), order_type=OrderType.LIMIT,
            timestamp=cancel_start + 0.3,
        )
        data.orders.append(real_order)
        event.orders.append(real_order.order_id)
        event.end_time = cancel_start + 1.0
        return event

    def plant_wash_trading(self, market_data: MarketData, num_events: int = 5,
                           subtype: str = "basic") -> MarketData:
        if not market_data.snapshots:
            return market_data
        duration = market_data.duration
        trader_a = f"WASH_A_{self.rng.randint(1000)}"
        trader_b = f"WASH_B_{self.rng.randint(1000)}"
        for _ in range(num_events):
            start_time = self.rng.uniform(duration * 0.1, duration * 0.9)
            event = self._plant_wash_event(market_data, trader_a, trader_b, start_time, subtype)
            if event:
                market_data.ground_truth_labels.append(event.to_dict())
        return market_data

    def _plant_wash_event(self, data: MarketData, trader_a: str, trader_b: str,
                           start_time: float, subtype: str) -> ManipulationEvent:
        event = ManipulationEvent(
            event_id=self._new_event_id(), manipulation_type="wash_trading",
            subtype=subtype, trader_id=f"{trader_a},{trader_b}",
            start_time=start_time, end_time=start_time + 3.0,
            description=f"Wash trading event ({subtype})",
        )
        snap = self._find_nearest_snapshot(data, start_time)
        if snap is None or snap.mid_price is None:
            return event
        mid = snap.mid_price
        num_washes = self.rng.randint(3, 10)
        for i in range(num_washes):
            wash_time = start_time + i * self.rng.uniform(0.1, 0.5)
            size = self.rng.randint(100, 1000)
            price = round(mid + self.rng.uniform(-0.02, 0.02), 2)
            buy_order = Order(
                order_id=f"WASH_{event.event_id}_buy_{i}", trader_id=trader_a,
                side=OrderSide.BUY, price=price, quantity=size,
                order_type=OrderType.LIMIT, timestamp=wash_time,
            )
            sell_order = Order(
                order_id=f"WASH_{event.event_id}_sell_{i}", trader_id=trader_b,
                side=OrderSide.SELL, price=price, quantity=size,
                order_type=OrderType.MARKET, timestamp=wash_time + 0.001,
            )
            data.orders.extend([buy_order, sell_order])
            event.orders.extend([buy_order.order_id, sell_order.order_id])
            wash_trade = Trade(
                trade_id=f"WT_{event.event_id}_{i}", instrument_id=data.instrument_id,
                price=price, quantity=size, aggressor_side=OrderSide.SELL,
                buyer_order_id=buy_order.order_id, seller_order_id=sell_order.order_id,
                buyer_trader_id=trader_a, seller_trader_id=trader_b,
                timestamp=wash_time + 0.001, sequence_number=0, is_wash=True,
            )
            data.trades.append(wash_trade)
            event.trades.append(wash_trade.trade_id)
        event.end_time = start_time + num_washes * 0.5
        return event

    def plant_all_types(self, market_data: MarketData, events_per_type: int = 3) -> MarketData:
        for subtype in ["basic", "layered_spoof", "momentum_ignition"]:
            self.plant_spoofing(market_data, events_per_type, subtype)
        for subtype in ["basic", "ascending", "descending"]:
            self.plant_layering(market_data, events_per_type, subtype)
        for subtype in ["basic", "pre_arranged", "circular"]:
            self.plant_wash_trading(market_data, events_per_type, subtype)
        return market_data
''')
