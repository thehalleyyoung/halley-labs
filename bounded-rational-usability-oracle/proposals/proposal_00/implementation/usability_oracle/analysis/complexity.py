"""
usability_oracle.analysis.complexity — Computational complexity analysis.

Estimates the empirical computational complexity of oracle operations
through timing-based regression, predicts scaling behaviour for larger
inputs, and identifies computational bottlenecks.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ComplexityEstimate:
    """Empirical complexity estimate for a computation.

    Attributes:
        complexity_class: Estimated asymptotic class (e.g. "O(n log n)").
        exponent: Fitted power-law exponent (time ~ n^exponent).
        r_squared: Goodness of fit for the power-law model.
        predicted_times: Predicted times for extrapolated sizes.
        measurements: Raw (size, time) measurements.
        model_params: Fitted model parameters.
    """
    complexity_class: str = "unknown"
    exponent: float = 0.0
    r_squared: float = 0.0
    predicted_times: dict[int, float] = field(default_factory=dict)
    measurements: list[tuple[int, float]] = field(default_factory=list)
    model_params: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Complexity: {self.complexity_class}",
            f"  Exponent: {self.exponent:.3f}",
            f"  R²:       {self.r_squared:.4f}",
        ]
        if self.predicted_times:
            lines.append("  Predictions:")
            for size, t in sorted(self.predicted_times.items()):
                lines.append(f"    n={size}: {t:.4f}s")
        return "\n".join(lines)


@dataclass
class BottleneckProfile:
    """Profile of computational bottlenecks."""
    stage_times: dict[str, float] = field(default_factory=dict)
    bottleneck_stage: str = ""
    bottleneck_fraction: float = 0.0
    total_time: float = 0.0
    scaling_estimates: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

_COMPLEXITY_MODELS = {
    "O(1)": lambda n, a, b: a * np.ones_like(n) + b,
    "O(log n)": lambda n, a, b: a * np.log2(np.maximum(n, 1)) + b,
    "O(n)": lambda n, a, b: a * n + b,
    "O(n log n)": lambda n, a, b: a * n * np.log2(np.maximum(n, 1)) + b,
    "O(n²)": lambda n, a, b: a * n ** 2 + b,
    "O(n³)": lambda n, a, b: a * n ** 3 + b,
    "O(2^n)": lambda n, a, b: a * np.power(2.0, np.minimum(n, 30)) + b,
}


def _fit_power_law(sizes: np.ndarray, times: np.ndarray) -> tuple[float, float, float]:
    """Fit a power law t = a * n^k + b using log-linear regression.

    Returns (exponent, a_coefficient, r_squared).
    """
    valid = (sizes > 0) & (times > 0)
    if np.sum(valid) < 3:
        return 1.0, 0.0, 0.0

    log_n = np.log(sizes[valid])
    log_t = np.log(times[valid])

    # Linear regression in log-log space
    n_pts = len(log_n)
    mean_ln = np.mean(log_n)
    mean_lt = np.mean(log_t)
    ss_nn = np.sum((log_n - mean_ln) ** 2)
    if ss_nn < 1e-12:
        return 1.0, 0.0, 0.0

    ss_nt = np.sum((log_n - mean_ln) * (log_t - mean_lt))
    exponent = float(ss_nt / ss_nn)
    intercept = mean_lt - exponent * mean_ln
    a_coeff = math.exp(intercept)

    # R² in log space
    ss_res = np.sum((log_t - (exponent * log_n + intercept)) ** 2)
    ss_tot = np.sum((log_t - mean_lt) ** 2)
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return exponent, a_coeff, r_sq


def _fit_all_models(
    sizes: np.ndarray,
    times: np.ndarray,
) -> list[tuple[str, float, dict[str, float]]]:
    """Fit all complexity models and return sorted by R².

    Returns list of (model_name, r_squared, params) sorted by best fit.
    """
    results = []
    n = sizes.astype(float)
    t = times.astype(float)

    for name, model_fn in _COMPLEXITY_MODELS.items():
        try:
            # Simple least squares for a and b
            if name == "O(2^n)" and np.max(n) > 30:
                continue
            basis = model_fn(n, 1.0, 0.0)
            if np.any(np.isinf(basis)) or np.any(np.isnan(basis)):
                continue
            # Solve [basis, 1] @ [a, b] = t in least squares
            A_mat = np.column_stack([basis - np.zeros_like(basis), np.ones_like(basis)])
            # Adjust: model_fn(n, a, b) = a * f(n) + b
            f_n = model_fn(n, 1.0, 0.0) - model_fn(n, 0.0, 1.0)
            A_mat = np.column_stack([f_n, np.ones(len(n))])
            result, residuals, _, _ = np.linalg.lstsq(A_mat, t, rcond=None)
            a_fit, b_fit = float(result[0]), float(result[1])

            predicted = model_fn(n, a_fit, b_fit)
            ss_res = float(np.sum((t - predicted) ** 2))
            ss_tot = float(np.sum((t - np.mean(t)) ** 2))
            r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

            results.append((name, r_sq, {"a": a_fit, "b": b_fit}))
        except Exception:
            continue

    results.sort(key=lambda x: -x[1])
    return results


def _classify_exponent(exp: float) -> str:
    """Classify a power-law exponent into a complexity class."""
    if exp < 0.15:
        return "O(1)"
    if exp < 0.6:
        return "O(√n)"
    if exp < 0.85:
        return "O(n^{:.2f})".format(exp)
    if exp < 1.15:
        return "O(n)"
    if exp < 1.35:
        return "O(n log n)"
    if exp < 1.7:
        return "O(n^{:.2f})".format(exp)
    if exp < 2.15:
        return "O(n²)"
    if exp < 2.7:
        return "O(n^{:.2f})".format(exp)
    if exp < 3.15:
        return "O(n³)"
    return f"O(n^{exp:.2f})"


# ---------------------------------------------------------------------------
# Timer utility
# ---------------------------------------------------------------------------

class _Timer:
    """Context manager for precise timing."""

    def __init__(self) -> None:
        self.elapsed = 0.0
        self._start = 0.0

    def __enter__(self) -> "_Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.perf_counter() - self._start


# ---------------------------------------------------------------------------
# ComplexityAnalyzer
# ---------------------------------------------------------------------------

class ComplexityAnalyzer:
    """Empirical computational complexity analysis.

    Parameters:
        min_size: Minimum input size for benchmarking.
        max_size: Maximum input size for benchmarking.
        n_sizes: Number of distinct sizes to test.
        n_repeats: Number of repeats per size for stable timing.
        warmup: Number of warmup iterations before timing.
    """

    def __init__(
        self,
        min_size: int = 10,
        max_size: int = 10000,
        n_sizes: int = 10,
        n_repeats: int = 3,
        warmup: int = 1,
    ) -> None:
        self._min_size = max(1, min_size)
        self._max_size = max(self._min_size + 1, max_size)
        self._n_sizes = max(3, n_sizes)
        self._n_repeats = max(1, n_repeats)
        self._warmup = max(0, warmup)

    # ------------------------------------------------------------------
    # Estimate complexity from a generator + function
    # ------------------------------------------------------------------

    def estimate(
        self,
        generator_fn: Callable[[int], Any],
        target_fn: Callable[[Any], Any],
        sizes: Optional[list[int]] = None,
        extrapolate_to: Optional[list[int]] = None,
    ) -> ComplexityEstimate:
        """Empirically estimate the time complexity of target_fn.

        Parameters:
            generator_fn: Creates an input of given size n.
            target_fn: The function to benchmark.
            sizes: Custom list of sizes (overrides min/max/n_sizes).
            extrapolate_to: Sizes to predict execution time for.
        """
        if sizes is None:
            sizes = np.logspace(
                math.log10(self._min_size),
                math.log10(self._max_size),
                self._n_sizes,
            ).astype(int).tolist()
            sizes = sorted(set(sizes))

        measurements: list[tuple[int, float]] = []

        for size in sizes:
            try:
                inp = generator_fn(size)
            except Exception:
                continue

            # Warmup
            for _ in range(self._warmup):
                try:
                    target_fn(inp)
                except Exception:
                    pass

            # Timed runs
            times = []
            for _ in range(self._n_repeats):
                with _Timer() as timer:
                    try:
                        target_fn(inp)
                    except Exception:
                        pass
                times.append(timer.elapsed)

            median_time = float(np.median(times))
            measurements.append((size, median_time))

        if len(measurements) < 3:
            return ComplexityEstimate(measurements=measurements)

        sizes_arr = np.array([m[0] for m in measurements], dtype=float)
        times_arr = np.array([m[1] for m in measurements], dtype=float)

        # Power law fit
        exponent, a_coeff, r_sq = _fit_power_law(sizes_arr, times_arr)
        complexity_class = _classify_exponent(exponent)

        # Fit all models
        model_fits = _fit_all_models(sizes_arr, times_arr)
        if model_fits and model_fits[0][1] > r_sq:
            complexity_class = model_fits[0][0]
            r_sq = model_fits[0][1]

        # Extrapolation
        predicted: dict[int, float] = {}
        if extrapolate_to:
            for sz in extrapolate_to:
                predicted[sz] = float(a_coeff * sz ** exponent)

        return ComplexityEstimate(
            complexity_class=complexity_class,
            exponent=exponent,
            r_squared=r_sq,
            predicted_times=predicted,
            measurements=measurements,
            model_params={"a": a_coeff, "exponent": exponent},
        )

    # ------------------------------------------------------------------
    # Profile pipeline stages
    # ------------------------------------------------------------------

    def profile_stages(
        self,
        stage_fns: dict[str, Callable[[], Any]],
        n_repeats: int = 5,
    ) -> BottleneckProfile:
        """Profile multiple pipeline stages and identify the bottleneck.

        Parameters:
            stage_fns: Mapping of stage_name -> callable to profile.
            n_repeats: Number of timing repeats per stage.
        """
        stage_times: dict[str, float] = {}

        for name, fn in stage_fns.items():
            # Warmup
            try:
                fn()
            except Exception:
                pass

            times = []
            for _ in range(n_repeats):
                with _Timer() as timer:
                    try:
                        fn()
                    except Exception:
                        pass
                times.append(timer.elapsed)

            stage_times[name] = float(np.median(times))

        total = sum(stage_times.values())
        bottleneck = max(stage_times, key=stage_times.get) if stage_times else ""
        fraction = stage_times.get(bottleneck, 0.0) / total if total > 0 else 0.0

        return BottleneckProfile(
            stage_times=stage_times,
            bottleneck_stage=bottleneck,
            bottleneck_fraction=fraction,
            total_time=total,
        )

    # ------------------------------------------------------------------
    # Amortized complexity from recorded data
    # ------------------------------------------------------------------

    @staticmethod
    def from_measurements(
        measurements: list[tuple[int, float]],
        extrapolate_to: Optional[list[int]] = None,
    ) -> ComplexityEstimate:
        """Estimate complexity from pre-recorded (size, time) measurements."""
        if len(measurements) < 3:
            return ComplexityEstimate(measurements=measurements)

        sizes_arr = np.array([m[0] for m in measurements], dtype=float)
        times_arr = np.array([m[1] for m in measurements], dtype=float)

        exponent, a_coeff, r_sq = _fit_power_law(sizes_arr, times_arr)
        complexity_class = _classify_exponent(exponent)

        predicted: dict[int, float] = {}
        if extrapolate_to:
            for sz in extrapolate_to:
                predicted[sz] = float(a_coeff * sz ** exponent)

        return ComplexityEstimate(
            complexity_class=complexity_class,
            exponent=exponent,
            r_squared=r_sq,
            predicted_times=predicted,
            measurements=measurements,
            model_params={"a": a_coeff, "exponent": exponent},
        )

    # ------------------------------------------------------------------
    # Memory complexity estimation
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_memory(
        generator_fn: Callable[[int], Any],
        sizes: list[int],
    ) -> list[tuple[int, int]]:
        """Estimate memory usage as a function of input size.

        Uses sys.getsizeof recursively for a rough estimate.
        Returns list of (size, estimated_bytes).
        """
        import sys

        def _deep_sizeof(obj: Any, seen: Optional[set] = None) -> int:
            """Recursively estimate object size."""
            if seen is None:
                seen = set()
            obj_id = id(obj)
            if obj_id in seen:
                return 0
            seen.add(obj_id)
            size = sys.getsizeof(obj)
            if isinstance(obj, dict):
                size += sum(_deep_sizeof(k, seen) + _deep_sizeof(v, seen) for k, v in obj.items())
            elif isinstance(obj, (list, tuple, set, frozenset)):
                size += sum(_deep_sizeof(item, seen) for item in obj)
            elif isinstance(obj, np.ndarray):
                size = obj.nbytes
            elif hasattr(obj, '__dict__'):
                size += _deep_sizeof(vars(obj), seen)
            return size

        results = []
        for sz in sizes:
            try:
                obj = generator_fn(sz)
                mem = _deep_sizeof(obj)
                results.append((sz, mem))
            except Exception:
                continue
        return results

    # ------------------------------------------------------------------
    # Amdahl's law speedup prediction
    # ------------------------------------------------------------------

    @staticmethod
    def amdahl_speedup(
        parallel_fraction: float,
        n_processors: int,
    ) -> float:
        """Compute theoretical speedup using Amdahl's law.

        S(p) = 1 / ((1 - f) + f/p)
        where f is the parallelisable fraction and p is the number of processors.
        """
        if n_processors < 1:
            return 1.0
        f = np.clip(parallel_fraction, 0.0, 1.0)
        return 1.0 / ((1.0 - f) + f / n_processors)

    @staticmethod
    def gustafson_speedup(
        parallel_fraction: float,
        n_processors: int,
    ) -> float:
        """Compute scaled speedup using Gustafson's law.

        S(p) = p - (1 - f) * (p - 1)
        """
        if n_processors < 1:
            return 1.0
        f = np.clip(parallel_fraction, 0.0, 1.0)
        return float(n_processors - (1.0 - f) * (n_processors - 1))
