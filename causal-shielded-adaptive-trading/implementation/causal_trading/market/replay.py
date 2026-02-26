"""
Market replay engine for walk-forward backtesting.

Delivers data sequentially to simulate a live trading environment,
preventing look-ahead bias by construction.  Supports event-driven
callbacks so that strategy logic can be decoupled from the data loop.

Usage
-----
>>> replay = MarketReplay(prices, features, volumes)
>>> replay.register_callback("on_bar", my_strategy.on_bar)
>>> replay.start()
"""

from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class EventType(Enum):
    """Types of market events emitted during replay."""
    ON_BAR = auto()
    ON_REGIME_CHANGE = auto()
    ON_HIGH_VOL = auto()
    ON_END = auto()


@dataclass
class MarketBar:
    """Single time-step market observation."""
    timestamp: int  # bar index
    price: float
    volume: float
    returns: float
    features: NDArray  # (n_features,)
    regime: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "price": self.price,
            "volume": self.volume,
            "returns": self.returns,
            "features": self.features.tolist(),
            "regime": self.regime,
        }


@dataclass
class ReplayStats:
    """Aggregate statistics from a completed replay."""
    n_bars: int = 0
    n_callbacks_fired: int = 0
    n_regime_changes: int = 0
    elapsed_seconds: float = 0.0
    mean_return: float = 0.0
    std_return: float = 0.0
    max_price: float = 0.0
    min_price: float = 0.0


# ---------------------------------------------------------------------------
# Core replay engine
# ---------------------------------------------------------------------------

CallbackFn = Callable[[MarketBar], None]


class MarketReplay:
    """Sequential market-data delivery engine.

    Parameters
    ----------
    prices : (T,) array
        Price series.
    features : (T, p) array
        Feature matrix.
    volumes : (T,) array or None
        Volume series.
    regime_labels : (T,) int array or None
        Regime assignments (for ``ON_REGIME_CHANGE`` events).
    warmup : int
        Number of initial bars to skip (deliver but don't fire callbacks)
        so that look-back indicators are populated.
    high_vol_threshold : float
        Z-score of realised vol above which ``ON_HIGH_VOL`` fires.
    """

    def __init__(
        self,
        prices: NDArray,
        features: NDArray,
        volumes: Optional[NDArray] = None,
        regime_labels: Optional[NDArray] = None,
        warmup: int = 63,
        high_vol_threshold: float = 2.0,
    ) -> None:
        self._prices = np.asarray(prices, dtype=np.float64)
        self._features = np.asarray(features, dtype=np.float64)
        self._T = len(self._prices)

        if volumes is not None:
            self._volumes = np.asarray(volumes, dtype=np.float64)
        else:
            self._volumes = np.ones(self._T, dtype=np.float64)

        if regime_labels is not None:
            self._regimes = np.asarray(regime_labels, dtype=int)
        else:
            self._regimes = None

        self.warmup = warmup
        self.high_vol_threshold = high_vol_threshold

        # Compute returns
        self._returns = np.zeros(self._T, dtype=np.float64)
        self._returns[1:] = np.diff(np.log(np.maximum(self._prices, 1e-10)))

        # Callback registry
        self._callbacks: Dict[EventType, List[CallbackFn]] = {
            et: [] for et in EventType
        }

        # State
        self._cursor: int = 0
        self._running: bool = False
        self._stats = ReplayStats()
        self._bar_history: List[MarketBar] = []

        # Expanding vol tracker for high-vol detection
        self._vol_sum: float = 0.0
        self._vol_sq_sum: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def total_bars(self) -> int:
        return self._T

    @property
    def current_bar(self) -> int:
        return self._cursor

    @property
    def is_running(self) -> bool:
        return self._running

    def register_callback(
        self,
        event: str | EventType,
        fn: CallbackFn,
    ) -> None:
        """Register a callback for a given event type.

        Parameters
        ----------
        event : str or EventType
            One of ``"on_bar"``, ``"on_regime_change"``, ``"on_high_vol"``,
            ``"on_end"``, or the corresponding ``EventType`` enum.
        fn : callable(MarketBar) → None
            Callback function.
        """
        if isinstance(event, str):
            event = EventType[event.upper()]
        self._callbacks[event].append(fn)

    def start(self) -> ReplayStats:
        """Run the full replay from the current cursor to the end.

        Returns aggregate statistics.
        """
        self._running = True
        t0 = _time.monotonic()
        logger.info("Replay started at bar %d / %d", self._cursor, self._T)

        while self._cursor < self._T:
            bar = self._build_bar(self._cursor)
            self._bar_history.append(bar)

            if self._cursor >= self.warmup:
                self._fire(EventType.ON_BAR, bar)
                self._check_regime_change(bar)
                self._check_high_vol(bar)

            self._cursor += 1

        self._running = False
        self._fire_end()

        elapsed = _time.monotonic() - t0
        self._stats.n_bars = self._T
        self._stats.elapsed_seconds = elapsed
        active = self._returns[self.warmup :]
        self._stats.mean_return = float(np.mean(active))
        self._stats.std_return = float(np.std(active, ddof=1)) if len(active) > 1 else 0.0
        self._stats.max_price = float(np.max(self._prices))
        self._stats.min_price = float(np.min(self._prices))

        logger.info(
            "Replay finished: %d bars in %.2fs, %d callbacks fired",
            self._stats.n_bars,
            elapsed,
            self._stats.n_callbacks_fired,
        )
        return self._stats

    def next(self) -> Optional[MarketBar]:
        """Advance one bar and return it (manual stepping mode).

        Returns ``None`` when the series is exhausted.
        """
        if self._cursor >= self._T:
            return None

        bar = self._build_bar(self._cursor)
        self._bar_history.append(bar)

        if self._cursor >= self.warmup:
            self._fire(EventType.ON_BAR, bar)
            self._check_regime_change(bar)
            self._check_high_vol(bar)

        self._cursor += 1
        return bar

    def reset(self) -> None:
        """Reset the replay to the beginning."""
        self._cursor = 0
        self._running = False
        self._stats = ReplayStats()
        self._bar_history.clear()
        self._vol_sum = 0.0
        self._vol_sq_sum = 0.0
        logger.debug("Replay reset")

    def get_history(
        self, last_n: Optional[int] = None
    ) -> List[MarketBar]:
        """Return delivered bars.  Optionally limit to last *n*."""
        if last_n is not None:
            return self._bar_history[-last_n:]
        return list(self._bar_history)

    def get_feature_window(
        self, window: int, up_to: Optional[int] = None
    ) -> NDArray:
        """Return (window, n_features) look-back slice ending at *up_to*.

        Raises ``ValueError`` if insufficient history.
        """
        end = up_to if up_to is not None else self._cursor
        start = end - window
        if start < 0:
            raise ValueError(
                f"Requested window={window} but only {end} bars available"
            )
        return self._features[start:end].copy()

    def get_price_window(
        self, window: int, up_to: Optional[int] = None
    ) -> NDArray:
        """Return (window,) price look-back slice."""
        end = up_to if up_to is not None else self._cursor
        start = end - window
        if start < 0:
            raise ValueError(
                f"Requested window={window} but only {end} bars available"
            )
        return self._prices[start:end].copy()

    def get_stats(self) -> ReplayStats:
        return self._stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_bar(self, t: int) -> MarketBar:
        regime = int(self._regimes[t]) if self._regimes is not None else None
        return MarketBar(
            timestamp=t,
            price=float(self._prices[t]),
            volume=float(self._volumes[t]),
            returns=float(self._returns[t]),
            features=self._features[t].copy(),
            regime=regime,
        )

    def _fire(self, event: EventType, bar: MarketBar) -> None:
        for fn in self._callbacks[event]:
            try:
                fn(bar)
                self._stats.n_callbacks_fired += 1
            except Exception:
                logger.exception(
                    "Callback error on %s at bar %d", event.name, bar.timestamp
                )

    def _fire_end(self) -> None:
        dummy = MarketBar(
            timestamp=self._T,
            price=float(self._prices[-1]),
            volume=0.0,
            returns=0.0,
            features=np.zeros(self._features.shape[1]),
            regime=None,
        )
        self._fire(EventType.ON_END, dummy)

    def _check_regime_change(self, bar: MarketBar) -> None:
        if self._regimes is None:
            return
        t = bar.timestamp
        if t > 0 and self._regimes[t] != self._regimes[t - 1]:
            self._stats.n_regime_changes += 1
            self._fire(EventType.ON_REGIME_CHANGE, bar)

    def _check_high_vol(self, bar: MarketBar) -> None:
        abs_ret = abs(bar.returns)
        self._vol_sum += abs_ret
        self._vol_sq_sum += abs_ret ** 2
        n = bar.timestamp + 1
        if n < 10:
            return
        mean = self._vol_sum / n
        var = self._vol_sq_sum / n - mean ** 2
        std = np.sqrt(max(var, 0.0))
        if std > 1e-10:
            z = (abs_ret - mean) / std
            if z > self.high_vol_threshold:
                self._fire(EventType.ON_HIGH_VOL, bar)
