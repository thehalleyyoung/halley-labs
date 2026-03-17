"""General utilities for CollusionProof."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import time
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
)

import numpy as np

logger = logging.getLogger("collusion_proof")

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure the ``collusion_proof`` logger."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger("collusion_proof")
    root.setLevel(numeric_level)
    root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler()
    sh.setLevel(numeric_level)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setLevel(numeric_level)
        fh.setFormatter(fmt)
        root.addHandler(fh)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
@contextmanager
def timer(label: str = "") -> Iterator[Dict[str, float]]:
    """Context manager for timing code blocks.

    Usage::

        with timer("my_block") as t:
            ...
        print(t["elapsed"])
    """
    result: Dict[str, float] = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - start
        if label:
            logger.info("%s completed in %.4f s", label, result["elapsed"])


# ---------------------------------------------------------------------------
# Random seed
# ---------------------------------------------------------------------------
def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility (NumPy + stdlib)."""
    import random

    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Moving-average helpers
# ---------------------------------------------------------------------------
def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Compute a simple moving average with a centred window."""
    if window < 1:
        raise ValueError("window must be >= 1")
    if len(data) < window:
        return np.full(len(data), np.nan)
    kernel = np.ones(window) / window
    # "valid" convolution yields len(data) - window + 1 elements
    return np.convolve(data, kernel, mode="valid")


def exponential_moving_average(data: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Compute an exponential moving average (EMA)."""
    if alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be in (0, 1]")
    ema = np.empty_like(data, dtype=float)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema


# ---------------------------------------------------------------------------
# Convergence detection
# ---------------------------------------------------------------------------
def detect_convergence(
    data: np.ndarray,
    window: int = 1000,
    threshold: float = 0.001,
) -> Optional[int]:
    """Detect the convergence point in a time series.

    Slides a window of size *window* and checks whether the standard
    deviation within the window is below *threshold*.  Returns the index
    of the first window start that qualifies, or ``None``.
    """
    if len(data) < window:
        return None
    for start in range(len(data) - window + 1):
        segment = data[start : start + window]
        if np.std(segment) < threshold:
            return start
    return None


# ---------------------------------------------------------------------------
# Economic helpers
# ---------------------------------------------------------------------------
def compute_nash_price(
    marginal_cost: float,
    demand_intercept: float,
    demand_slope: float,
    num_players: int,
) -> float:
    """Compute Nash equilibrium price for symmetric Bertrand competition."""
    a = demand_intercept / demand_slope
    return (a + num_players * marginal_cost) / (num_players + 1)


def compute_monopoly_price(
    marginal_cost: float,
    demand_intercept: float,
    demand_slope: float,
) -> float:
    """Compute the monopoly price."""
    a = demand_intercept / demand_slope
    return (a + marginal_cost) / 2.0


def compute_demand(
    price: float, demand_intercept: float, demand_slope: float
) -> float:
    """Compute demand at a given price (linear demand model)."""
    return max(demand_intercept - demand_slope * price, 0.0)


def compute_profit(price: float, quantity: float, marginal_cost: float) -> float:
    """Compute profit: (p - c) * q."""
    return (price - marginal_cost) * quantity


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------
def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Divide *a* by *b*, returning *default* when *b* is zero."""
    if b == 0:
        return default
    return a / b


def clip(value: float, low: float, high: float) -> float:
    """Clip *value* to [low, high]."""
    return max(low, min(value, high))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_price_matrix(prices: np.ndarray) -> None:
    """Validate a price matrix (num_rounds × num_players).

    Raises ``ValueError`` for any structural or value problem.
    """
    if prices.ndim != 2:
        raise ValueError(
            f"prices must be a 2-D array, got {prices.ndim}-D"
        )
    if prices.shape[0] == 0 or prices.shape[1] == 0:
        raise ValueError("prices must be non-empty along both axes")
    if np.any(np.isnan(prices)):
        raise ValueError("prices contain NaN values")
    if np.any(prices < 0):
        raise ValueError("prices contain negative values")


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def hash_config(config: Dict[str, Any]) -> str:
    """Create a deterministic SHA-256 hex digest of *config*."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------
def chunk_list(lst: List[T], chunk_size: int) -> List[List[T]]:
    """Split *lst* into chunks of at most *chunk_size*."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten(nested: List[List[T]]) -> List[T]:
    """Flatten one level of nesting."""
    return [item for sub in nested for item in sub]


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------
def retry(max_attempts: int = 3, delay: float = 1.0) -> Callable:
    """Retry decorator – retries up to *max_attempts* times on exception."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Optional[Exception] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    logger.warning(
                        "Attempt %d/%d for %s failed: %s",
                        attempt,
                        max_attempts,
                        func.__name__,
                        exc,
                    )
                    if attempt < max_attempts:
                        time.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator


def memoize(func: Callable) -> Callable:
    """Simple memoisation decorator (unhashable args fall through)."""
    cache: Dict[Any, Any] = {}

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        key = (args, tuple(sorted(kwargs.items())))
        try:
            hash(key)
        except TypeError:
            return func(*args, **kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    wrapper.cache = cache  # type: ignore[attr-defined]
    return wrapper


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------
class ProgressTracker:
    """Track progress of long-running operations."""

    def __init__(self, total: int, label: str = "") -> None:
        self.total = total
        self.label = label
        self.current = 0
        self._start = time.perf_counter()
        self._last_log = self._start

    def update(self, n: int = 1) -> None:
        """Advance the tracker by *n* steps, logging every 10 %."""
        self.current += n
        now = time.perf_counter()
        pct = self.current / self.total if self.total > 0 else 1.0
        # Log at most once per second or at 10 % increments
        if now - self._last_log >= 1.0 or pct >= 1.0:
            elapsed = now - self._start
            eta = (elapsed / pct - elapsed) if pct > 0 else 0.0
            logger.info(
                "%s %.1f%% (%d/%d) elapsed=%.1fs ETA=%.1fs",
                self.label,
                pct * 100,
                self.current,
                self.total,
                elapsed,
                eta,
            )
            self._last_log = now

    def finish(self) -> None:
        """Mark the tracker as complete."""
        self.current = self.total
        elapsed = time.perf_counter() - self._start
        logger.info(
            "%s finished in %.2f s (%d items)",
            self.label,
            elapsed,
            self.total,
        )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def format_p_value(p: float) -> str:
    """Format a p-value for human-readable display."""
    if p < 1e-16:
        return "< 1e-16"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def format_ci(lower: float, upper: float, level: float = 0.95) -> str:
    """Format a confidence interval for display."""
    pct = int(level * 100)
    return f"{pct}% CI [{lower:.4f}, {upper:.4f}]"


def format_verdict(verdict: str, confidence: float) -> str:
    """Format a collusion verdict string."""
    return f"{verdict} (confidence={confidence:.2%})"


# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------
def ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure *arr* is 2-D (promotes 1-D to column vector)."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Cannot promote {arr.ndim}-D array to 2-D")


def windowed_statistic(
    data: np.ndarray,
    window: int,
    func: Callable,
) -> np.ndarray:
    """Compute a rolling statistic over non-overlapping windows.

    Returns one value per window.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    n_windows = len(data) // window
    result = np.empty(n_windows)
    for i in range(n_windows):
        segment = data[i * window : (i + 1) * window]
        result[i] = func(segment)
    return result
