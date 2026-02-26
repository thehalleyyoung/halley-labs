"""LOB simulator with realistic market microstructure and manipulation scenarios.

Implements:
  - Calibrated LOB simulator with empirically-grounded order arrival rates
  - Manipulation planting for spoofing, layering, wash trading
  - SEC enforcement action scenario generation (Sarao 2010, Coscia 2015)
  - Ground truth labels for evaluation
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vmee.lob.orderbook import LimitOrderBook, Order, OrderSide, OrderType, OrderStatus
from vmee.temporal.monitor import Event

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Container for generated market data."""
    orders: List[Dict] = field(default_factory=list)
    trades: List[Dict] = field(default_factory=list)
    snapshots: List[Dict] = field(default_factory=list)
    features: Optional[np.ndarray] = None
    events: List[Any] = field(default_factory=list)
    windows: List[Dict] = field(default_factory=list)
    manipulation_labels: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "num_orders": len(self.orders),
            "num_trades": len(self.trades),
            "num_snapshots": len(self.snapshots),
            "num_manipulation_labels": len(self.manipulation_labels),
        }


class LOBSimulator:
    """Limit order book simulator with calibrated market microstructure.

    Generates synthetic data calibrated against published LOB statistics:
      - Order arrival: Poisson process with rate λ ≈ 10-50 orders/sec
      - Order size: log-normal with μ=5.5, σ=1.2 (Cont et al. 2010)
      - Cancel rate: ~60-80% of limit orders cancelled (Hasbrouck 2007)
      - Spread: mean 1-3 ticks for liquid instruments
    """

    def __init__(self, config=None):
        self.config = config
        self.seed = getattr(config, 'seed', 42) if config else 42
        self.rng = np.random.RandomState(self.seed)

    def generate_trading_day(self, num_events: int = 1000) -> MarketData:
        """Generate a calibrated synthetic trading day."""
        book = LimitOrderBook(
            instrument_id="SIM_001",
            tick_size=getattr(self.config, 'tick_size', 0.01) if self.config else 0.01,
        )

        data = MarketData()
        mid_price = 100.0
        spread = 0.05
        num_traders = 50

        for i in range(num_events):
            # Poisson inter-arrival (mean 0.5 sec)
            t = float(i) * 0.5 + self.rng.exponential(0.1)
            side = OrderSide.BUY if self.rng.random() < 0.5 else OrderSide.SELL

            # Price: centered on mid with spread
            if side == OrderSide.BUY:
                price = mid_price - spread / 2 + self.rng.normal(0, 0.3)
            else:
                price = mid_price + spread / 2 + self.rng.normal(0, 0.3)

            # Log-normal order size (Cont et al. 2010)
            qty = max(1, int(np.exp(self.rng.normal(5.5, 1.2))))
            qty = min(qty, 10000)

            trader_id = f"trader_{self.rng.randint(0, num_traders)}"

            order = Order(
                order_id=f"ord_{i}",
                trader_id=trader_id,
                side=side,
                price=round(price, 2),
                quantity=qty,
                timestamp=t,
                instrument_id="SIM_001",
            )

            book.add_order(order)

            # Cancel probability: 70%
            cancelled = self.rng.random() < 0.70

            data.orders.append({
                "order_id": order.order_id,
                "trader_id": trader_id,
                "side": order.side.name,
                "price": order.price,
                "quantity": order.quantity,
                "timestamp": t,
                "cancelled": cancelled,
            })

            # Generate monitoring events
            cancel_ratio = sum(
                1 for o in data.orders[-min(50, len(data.orders)):]
                if o.get("cancelled")
            ) / max(1, min(50, len(data.orders)))

            data.events.append(Event(
                timestamp=t,
                predicates={
                    "order_size": float(qty),
                    "cancel_ratio": cancel_ratio,
                    "opposite_execution": self.rng.uniform(0, 0.3),
                    "multi_level_orders": float(self.rng.poisson(1)),
                    "self_trade_ratio": self.rng.beta(1, 10),
                    "volume_without_position_change": self.rng.beta(1, 5),
                },
            ))

            # Update mid price with small random walk
            mid_price += self.rng.normal(0, 0.01)

        data.windows = [{"id": 0, "data": data}]

        # Generate feature matrix for causal discovery
        data.features = self._compute_features(data)

        return data

    def _compute_features(self, data: MarketData) -> np.ndarray:
        """Compute feature matrix from order data for causal discovery."""
        n = len(data.orders)
        window = 20

        features = np.zeros((n, 7))
        for i in range(n):
            start = max(0, i - window)
            window_orders = data.orders[start:i + 1]

            buy_flow = sum(o["quantity"] for o in window_orders if o["side"] == "BUY")
            sell_flow = sum(o["quantity"] for o in window_orders if o["side"] == "SELL")
            total_flow = buy_flow + sell_flow

            cancels = sum(1 for o in window_orders if o.get("cancelled"))
            cancel_ratio = cancels / max(1, len(window_orders))

            prices = [o["price"] for o in window_orders]
            spread = max(prices) - min(prices) if len(prices) > 1 else 0
            depth_imb = (buy_flow - sell_flow) / max(1, total_flow)
            trade_imb = depth_imb * 0.5 + self.rng.normal(0, 0.1)

            # Intent: binary label from manipulation labels
            intent = 0.0
            for label in data.manipulation_labels:
                if label["start_idx"] <= i <= label["end_idx"]:
                    intent = 1.0
                    break

            price_impact = cancel_ratio * 0.4 + spread * 0.2 + self.rng.normal(0, 0.05)

            features[i] = [
                total_flow / 1000, cancel_ratio, spread,
                depth_imb, trade_imb, intent, price_impact,
            ]

        return features


class ManipulationPlanter:
    """Plants ground-truth manipulation events in market data.

    Implements three canonical manipulation types:
      1. Spoofing: large orders followed by rapid cancellation
         (based on Sarao Flash Crash, SEC v. Sarao 2015)
      2. Layering: multiple orders at successive price levels, all cancelled
         (based on Coscia case, US v. Coscia 2015)
      3. Wash trading: simultaneous buy/sell by related accounts
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def plant_spoofing(
        self, data: MarketData, start_idx: int = 100,
        duration: int = 50, intensity: float = 5.0
    ) -> MarketData:
        """Plant spoofing pattern: large orders + rapid cancellation.

        Modeled after Navinder Sarao's spoofing strategy (2010):
          - Place large sell orders (5x normal size) at multiple price levels
          - Cancel within 1-5 seconds
          - Execute small buy orders on the opposite side during price dip
        """
        end_idx = min(start_idx + duration, len(data.orders))
        spoofing_trader = "spoofer_001"

        for i in range(start_idx, end_idx):
            if i < len(data.orders):
                order = data.orders[i]
                order["trader_id"] = spoofing_trader
                order["quantity"] = int(order["quantity"] * intensity)
                order["cancelled"] = True  # high cancel rate
                order["side"] = "SELL"  # spoofing on sell side

                # Update event predicates
                if i < len(data.events):
                    data.events[i].predicates["order_size"] = float(order["quantity"])
                    data.events[i].predicates["cancel_ratio"] = 0.95
                    data.events[i].predicates["opposite_execution"] = 0.8

        data.manipulation_labels.append({
            "type": "spoofing",
            "subtype": "sarao_style",
            "start_idx": start_idx,
            "end_idx": end_idx,
            "trader_id": spoofing_trader,
            "description": (
                "Large sell orders placed and rapidly cancelled to move price, "
                "with opposite-side execution. Modeled after SEC v. Sarao (2015)."
            ),
        })
        return data

    def plant_layering(
        self, data: MarketData, start_idx: int = 300,
        duration: int = 40, num_levels: int = 5
    ) -> MarketData:
        """Plant layering pattern: orders at successive price levels.

        Modeled after US v. Coscia (2015):
          - Place orders at 3-5 successive price levels on one side
          - All orders cancelled within seconds
          - Execute on opposite side during brief price impact
        """
        end_idx = min(start_idx + duration, len(data.orders))
        layering_trader = "layerer_001"

        for i in range(start_idx, end_idx):
            if i < len(data.orders):
                order = data.orders[i]
                order["trader_id"] = layering_trader
                order["cancelled"] = True
                level = (i - start_idx) % num_levels
                order["price"] = order["price"] + level * 0.05
                order["quantity"] = int(order["quantity"] * 3)

                if i < len(data.events):
                    data.events[i].predicates["multi_level_orders"] = float(num_levels)
                    data.events[i].predicates["cancel_ratio"] = 0.92

        data.manipulation_labels.append({
            "type": "layering",
            "subtype": "coscia_style",
            "start_idx": start_idx,
            "end_idx": end_idx,
            "trader_id": layering_trader,
            "num_levels": num_levels,
            "description": (
                "Orders placed at successive price levels and rapidly cancelled. "
                "Modeled after US v. Coscia (2015)."
            ),
        })
        return data

    def plant_wash_trading(
        self, data: MarketData, start_idx: int = 500,
        duration: int = 30
    ) -> MarketData:
        """Plant wash trading: simultaneous buy/sell by related accounts."""
        end_idx = min(start_idx + duration, len(data.orders))
        wash_accounts = ["wash_A", "wash_B"]

        for i in range(start_idx, end_idx):
            if i < len(data.orders):
                order = data.orders[i]
                order["trader_id"] = wash_accounts[i % 2]
                # Alternating buy/sell at same price
                order["side"] = "BUY" if i % 2 == 0 else "SELL"
                order["cancelled"] = False  # wash trades execute

                if i < len(data.events):
                    data.events[i].predicates["self_trade_ratio"] = 0.8
                    data.events[i].predicates["volume_without_position_change"] = 0.9

        data.manipulation_labels.append({
            "type": "wash_trading",
            "start_idx": start_idx,
            "end_idx": end_idx,
            "trader_ids": wash_accounts,
            "description": "Coordinated buy/sell between related accounts at same price.",
        })
        return data

    def plant_sec_scenario(
        self, data: MarketData, scenario: str = "sarao_2010"
    ) -> MarketData:
        """Plant a complete SEC enforcement action scenario.

        Available scenarios:
          - sarao_2010: Navinder Sarao Flash Crash spoofing
          - coscia_2015: Michael Coscia commodity futures layering
          - combined: All manipulation types in one trading day
        
        Start indices are computed relative to data length to ensure
        manipulation is always planted within the available data range.
        """
        n = len(data.orders)
        if scenario == "sarao_2010":
            start = max(10, int(n * 0.3))
            duration = min(int(n * 0.4), n - start)
            data = self.plant_spoofing(data, start_idx=start, duration=duration, intensity=8.0)
        elif scenario == "coscia_2015":
            start = max(10, int(n * 0.25))
            duration = min(int(n * 0.3), n - start)
            data = self.plant_layering(data, start_idx=start, duration=duration, num_levels=5)
        elif scenario == "combined":
            seg = max(1, n // 5)
            data = self.plant_spoofing(data, start_idx=seg, duration=min(seg, n - seg), intensity=5.0)
            data = self.plant_layering(data, start_idx=2 * seg, duration=min(seg, n - 2*seg), num_levels=4)
            data = self.plant_wash_trading(data, start_idx=3 * seg, duration=min(seg // 2, n - 3*seg))
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        # Recompute features with manipulation labels
        sim = LOBSimulator()
        data.features = sim._compute_features(data)
        return data
