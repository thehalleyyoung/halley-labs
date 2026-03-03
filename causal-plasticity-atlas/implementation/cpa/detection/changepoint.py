"""
Generic changepoint detection algorithms.

Provides reusable, well-tested changepoint detection primitives:
  - PELT (Pruned Exact Linear Time)
  - Binary Segmentation
  - CUSUM detector
  - Configurable cost functions and penalty selection

These are general-purpose routines consumed by the higher-level
tipping-point detection pipeline (ALG4).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Cost function registry
# ---------------------------------------------------------------------------

class CostType(str, Enum):
    """Supported segment cost functions."""

    L2 = "l2"
    GAUSSIAN_LIKELIHOOD = "gaussian_likelihood"
    POISSON = "poisson"
    RBFKERNEL = "rbf_kernel"
    CUSTOM = "custom"


class PenaltyType(str, Enum):
    """Penalty selection strategies."""

    BIC = "bic"
    AIC = "aic"
    HANNAN_QUINN = "hannan_quinn"
    MANUAL = "manual"
    ELBOW = "elbow"


# ---------------------------------------------------------------------------
# Segment description dataclass
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """A contiguous segment between two changepoints."""

    start: int
    end: int  # exclusive
    cost: float
    mean: Optional[NDArray] = None
    variance: Optional[float] = None
    count: int = 0

    @property
    def length(self) -> int:
        return self.end - self.start

    def __repr__(self) -> str:
        return f"Segment([{self.start}, {self.end}), len={self.length}, cost={self.cost:.4f})"


@dataclass
class ChangepointResult:
    """Result of a changepoint detection run."""

    changepoints: list[int]
    segments: list[Segment]
    cost: float
    penalty: float
    n_changepoints: int
    method: str
    metadata: dict = field(default_factory=dict)

    def segment_boundaries(self) -> list[tuple[int, int]]:
        """Return list of (start, end) tuples for each segment."""
        return [(s.start, s.end) for s in self.segments]


# ---------------------------------------------------------------------------
# Cost function implementations
# ---------------------------------------------------------------------------

def _validate_signal(signal: NDArray) -> NDArray:
    """Ensure signal is 2-D float array (n_samples, n_dims)."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]
    if signal.ndim != 2:
        raise ValueError(f"Signal must be 1-D or 2-D, got {signal.ndim}-D.")
    if signal.shape[0] < 2:
        raise ValueError("Signal must have at least 2 samples.")
    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal contains non-finite values (NaN/Inf).")
    return signal


class CostFunction:
    """Base class for segment cost functions.

    A cost function takes a signal slice ``signal[start:end]`` and returns
    the cost of representing that slice as a single segment.
    """

    def __init__(self, signal: NDArray):
        self._signal = _validate_signal(signal)
        self._n = self._signal.shape[0]
        self._d = self._signal.shape[1]
        self._precompute()

    def _precompute(self) -> None:
        """Pre-compute cumulative statistics for O(1) segment costs."""
        pass

    def cost(self, start: int, end: int) -> float:
        """Return cost of segment [start, end)."""
        raise NotImplementedError

    def segment_stats(self, start: int, end: int) -> dict:
        """Return descriptive statistics for a segment."""
        seg = self._signal[start:end]
        return {
            "mean": np.mean(seg, axis=0),
            "var": np.var(seg, axis=0),
            "n": end - start,
        }


class L2Cost(CostFunction):
    """Sum of squared deviations from segment mean (L2 cost).

    Cost([start, end)) = sum_{i=start}^{end-1} ||x_i - mean||^2
    """

    def _precompute(self) -> None:
        # Cumulative sums for O(1) evaluation
        self._cumsum = np.zeros((self._n + 1, self._d), dtype=np.float64)
        self._cumsum_sq = np.zeros((self._n + 1, self._d), dtype=np.float64)
        np.cumsum(self._signal, axis=0, out=self._cumsum[1:])
        np.cumsum(self._signal ** 2, axis=0, out=self._cumsum_sq[1:])

    def cost(self, start: int, end: int) -> float:
        if end <= start:
            return 0.0
        n = end - start
        seg_sum = self._cumsum[end] - self._cumsum[start]
        seg_sum_sq = self._cumsum_sq[end] - self._cumsum_sq[start]
        # sum(x_i^2) - (sum x_i)^2 / n
        c = np.sum(seg_sum_sq - seg_sum ** 2 / n)
        return max(c, 0.0)


class GaussianLikelihoodCost(CostFunction):
    """Negative log-likelihood cost for univariate/multivariate Gaussian.

    Cost = (n/2) * log(2 * pi * var) + n/2  (per dimension, summed)
    Equivalent to minimizing within-segment variance.
    """

    def _precompute(self) -> None:
        self._cumsum = np.zeros((self._n + 1, self._d), dtype=np.float64)
        self._cumsum_sq = np.zeros((self._n + 1, self._d), dtype=np.float64)
        np.cumsum(self._signal, axis=0, out=self._cumsum[1:])
        np.cumsum(self._signal ** 2, axis=0, out=self._cumsum_sq[1:])

    def cost(self, start: int, end: int) -> float:
        if end <= start:
            return 0.0
        n = end - start
        if n < 2:
            return 0.0
        seg_sum = self._cumsum[end] - self._cumsum[start]
        seg_sum_sq = self._cumsum_sq[end] - self._cumsum_sq[start]
        var = (seg_sum_sq - seg_sum ** 2 / n) / n
        # Clamp variance to avoid log(0)
        var = np.maximum(var, 1e-15)
        # -2 * log-likelihood (up to constant)
        c = n * np.sum(np.log(var)) + n * self._d
        return float(c)


class PoissonCost(CostFunction):
    """Poisson cost: sum_{i} x_i * log(x_i / lambda_hat) for non-negative data.

    Used when the signal represents counts.
    """

    def _precompute(self) -> None:
        if np.any(self._signal < 0):
            raise ValueError("Poisson cost requires non-negative data.")
        self._cumsum = np.zeros((self._n + 1, self._d), dtype=np.float64)
        np.cumsum(self._signal, axis=0, out=self._cumsum[1:])
        # Precompute cumulative x*log(x) for O(1) cost
        safe = np.where(self._signal > 0, self._signal, 1.0)
        xlogx = self._signal * np.log(safe)
        self._cumsum_xlogx = np.zeros((self._n + 1, self._d), dtype=np.float64)
        np.cumsum(xlogx, axis=0, out=self._cumsum_xlogx[1:])

    def cost(self, start: int, end: int) -> float:
        if end <= start:
            return 0.0
        n = end - start
        seg_sum = self._cumsum[end] - self._cumsum[start]
        seg_xlogx = self._cumsum_xlogx[end] - self._cumsum_xlogx[start]
        lam = seg_sum / n
        lam_safe = np.maximum(lam, 1e-15)
        # 2 * (sum x_i log x_i - sum x_i log lambda)
        c = 2.0 * np.sum(seg_xlogx - seg_sum * np.log(lam_safe))
        return max(float(c), 0.0)


class RBFKernelCost(CostFunction):
    """RBF kernel cost for non-parametric changepoint detection.

    Cost = -2 * sum_{i<j in seg} K(x_i, x_j) / n^2
    Uses an approximation via cumulative statistics.
    """

    def __init__(self, signal: NDArray, gamma: float = 1.0):
        self._gamma = gamma
        super().__init__(signal)

    def _precompute(self) -> None:
        # Full kernel matrix (only feasible for moderate n)
        if self._n > 5000:
            warnings.warn(
                f"RBF kernel cost with n={self._n} will be slow. "
                "Consider using L2 cost instead.",
                stacklevel=2,
            )
        # Precompute kernel matrix
        diff = self._signal[:, np.newaxis, :] - self._signal[np.newaxis, :, :]
        sq_dist = np.sum(diff ** 2, axis=-1)
        self._kernel = np.exp(-self._gamma * sq_dist)
        # Cumulative kernel sums for O(n) segment cost
        self._cum_kernel = np.zeros((self._n + 1, self._n + 1), dtype=np.float64)
        for i in range(self._n):
            for j in range(self._n):
                self._cum_kernel[i + 1, j + 1] = (
                    self._kernel[i, j]
                    + self._cum_kernel[i, j + 1]
                    + self._cum_kernel[i + 1, j]
                    - self._cum_kernel[i, j]
                )

    def cost(self, start: int, end: int) -> float:
        if end <= start:
            return 0.0
        n = end - start
        if n < 2:
            return 0.0
        # sum K(x_i, x_j) for i,j in [start, end)
        k_sum = (
            self._cum_kernel[end, end]
            - self._cum_kernel[start, end]
            - self._cum_kernel[end, start]
            + self._cum_kernel[start, start]
        )
        return float(-k_sum / n)


class CustomCost(CostFunction):
    """Wrapper for user-supplied cost functions."""

    def __init__(self, signal: NDArray, cost_fn: Callable[[NDArray], float]):
        self._custom_fn = cost_fn
        super().__init__(signal)

    def cost(self, start: int, end: int) -> float:
        if end <= start:
            return 0.0
        return self._custom_fn(self._signal[start:end])


def make_cost_function(
    signal: NDArray,
    cost_type: CostType | str = CostType.L2,
    **kwargs,
) -> CostFunction:
    """Factory for cost functions.

    Parameters
    ----------
    signal : array-like, shape (n_samples,) or (n_samples, n_dims)
    cost_type : CostType or str
    **kwargs : passed to cost function constructor

    Returns
    -------
    CostFunction instance
    """
    ct = CostType(cost_type) if isinstance(cost_type, str) else cost_type
    if ct == CostType.L2:
        return L2Cost(signal, **kwargs)
    elif ct == CostType.GAUSSIAN_LIKELIHOOD:
        return GaussianLikelihoodCost(signal, **kwargs)
    elif ct == CostType.POISSON:
        return PoissonCost(signal, **kwargs)
    elif ct == CostType.RBFKERNEL:
        return RBFKernelCost(signal, **kwargs)
    elif ct == CostType.CUSTOM:
        if "cost_fn" not in kwargs:
            raise ValueError("Custom cost requires 'cost_fn' argument.")
        return CustomCost(signal, cost_fn=kwargs.pop("cost_fn"), **kwargs)
    else:
        raise ValueError(f"Unknown cost type: {ct}")


# ---------------------------------------------------------------------------
# Penalty selection
# ---------------------------------------------------------------------------

def compute_penalty(
    n: int,
    n_dims: int = 1,
    penalty_type: PenaltyType | str = PenaltyType.BIC,
    manual_value: Optional[float] = None,
    signal: Optional[NDArray] = None,
) -> float:
    """Compute the changepoint penalty.

    Parameters
    ----------
    n : Number of data points
    n_dims : Dimensionality of the signal
    penalty_type : Penalty selection strategy
    manual_value : Used if penalty_type is MANUAL
    signal : Required for ELBOW method

    Returns
    -------
    float : Penalty value C in cost + C * n_changepoints formulation
    """
    pt = PenaltyType(penalty_type) if isinstance(penalty_type, str) else penalty_type

    if pt == PenaltyType.BIC:
        # BIC: C * log(n) per changepoint
        return n_dims * math.log(n)
    elif pt == PenaltyType.AIC:
        return 2.0 * n_dims
    elif pt == PenaltyType.HANNAN_QUINN:
        return 2.0 * n_dims * math.log(math.log(max(n, 3)))
    elif pt == PenaltyType.MANUAL:
        if manual_value is None:
            raise ValueError("manual_value required for MANUAL penalty.")
        return float(manual_value)
    elif pt == PenaltyType.ELBOW:
        return _elbow_penalty(n, n_dims, signal)
    else:
        raise ValueError(f"Unknown penalty type: {pt}")


def _elbow_penalty(n: int, n_dims: int, signal: Optional[NDArray]) -> float:
    """Select penalty via elbow method.

    Run PELT with a range of penalties and pick the elbow point where
    adding more changepoints yields diminishing cost reduction.
    """
    if signal is None:
        raise ValueError("Signal required for elbow penalty selection.")
    signal = _validate_signal(signal)
    cost_fn = L2Cost(signal)

    penalties = np.logspace(-1, np.log10(10 * n), num=30)
    n_cps = []
    total_costs = []

    for pen in penalties:
        result = _pelt_core(cost_fn, n, pen, min_size=2)
        n_cps.append(len(result))
        # Compute total segment cost
        bps = [0] + sorted(result) + [n]
        tc = sum(cost_fn.cost(bps[i], bps[i + 1]) for i in range(len(bps) - 1))
        total_costs.append(tc + pen * len(result))

    n_cps = np.array(n_cps)
    total_costs = np.array(total_costs)

    # Find elbow: maximum curvature point
    if len(set(n_cps)) < 3:
        return float(penalties[len(penalties) // 2])

    # Normalize
    x = (n_cps - n_cps.min()) / max(n_cps.max() - n_cps.min(), 1e-10)
    y = (total_costs - total_costs.min()) / max(total_costs.max() - total_costs.min(), 1e-10)

    # Distance from line connecting first and last points
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    line = p2 - p1
    line_len = np.linalg.norm(line)
    if line_len < 1e-10:
        return float(penalties[len(penalties) // 2])

    dists = np.abs(np.cross(line, p1 - np.column_stack([x, y]))) / line_len
    elbow_idx = int(np.argmax(dists))
    return float(penalties[elbow_idx])


# ---------------------------------------------------------------------------
# Segment statistics
# ---------------------------------------------------------------------------

def compute_segment_statistics(
    signal: NDArray,
    segments: list[Segment],
) -> list[dict]:
    """Compute descriptive statistics for each segment.

    Parameters
    ----------
    signal : (n_samples, n_dims) signal array
    segments : list of Segment objects

    Returns
    -------
    list of dicts with keys: mean, std, var, min, max, median, iqr, n
    """
    signal = _validate_signal(signal)
    stats = []
    for seg in segments:
        s = signal[seg.start:seg.end]
        n = s.shape[0]
        if n == 0:
            stats.append({
                "mean": np.zeros(signal.shape[1]),
                "std": np.zeros(signal.shape[1]),
                "var": np.zeros(signal.shape[1]),
                "min": np.zeros(signal.shape[1]),
                "max": np.zeros(signal.shape[1]),
                "median": np.zeros(signal.shape[1]),
                "iqr": np.zeros(signal.shape[1]),
                "n": 0,
            })
            continue

        q25 = np.percentile(s, 25, axis=0)
        q75 = np.percentile(s, 75, axis=0)
        stats.append({
            "mean": np.mean(s, axis=0),
            "std": np.std(s, axis=0, ddof=1) if n > 1 else np.zeros(signal.shape[1]),
            "var": np.var(s, axis=0, ddof=1) if n > 1 else np.zeros(signal.shape[1]),
            "min": np.min(s, axis=0),
            "max": np.max(s, axis=0),
            "median": np.median(s, axis=0),
            "iqr": q75 - q25,
            "n": n,
        })
    return stats


# ---------------------------------------------------------------------------
# PELT — Pruned Exact Linear Time
# ---------------------------------------------------------------------------

def _pelt_core(
    cost_fn: CostFunction,
    n: int,
    penalty: float,
    min_size: int = 2,
) -> list[int]:
    """Core PELT algorithm returning changepoint indices.

    Implements the pruning strategy of Killick et al. (2012) for
    O(n) expected-time changepoint detection.

    Parameters
    ----------
    cost_fn : CostFunction with O(1) segment cost evaluation
    n : Number of data points
    penalty : Per-changepoint penalty C
    min_size : Minimum segment length

    Returns
    -------
    list[int] : Sorted changepoint indices (positions where change occurs)
    """
    if n < 2 * min_size:
        return []

    # F[t] = optimal cost of signal[0:t]
    F = np.full(n + 1, np.inf)
    F[0] = -penalty  # So that F[0] + cost(0,t) + penalty = cost(0,t)
    cp = [[] for _ in range(n + 1)]  # changepoint lists
    # R = set of candidate last-changepoints (admissible set)
    R = [0]

    for t in range(min_size, n + 1):
        # Find best partition point from admissible set
        best_F = np.inf
        best_tau = 0

        candidates_to_keep = []
        for tau in R:
            if t - tau < min_size:
                candidates_to_keep.append(tau)
                continue

            seg_cost = cost_fn.cost(tau, t)
            candidate_cost = F[tau] + seg_cost + penalty

            if candidate_cost < best_F:
                best_F = candidate_cost
                best_tau = tau

            # PELT pruning: keep tau only if F[tau] + cost(tau, t) <= F[t]
            # (will check after F[t] is set)
            candidates_to_keep.append(tau)

        F[t] = best_F
        if best_tau == 0:
            cp[t] = []
        else:
            cp[t] = cp[best_tau] + [best_tau]

        # Pruning step: remove candidates that can never be optimal
        R_new = []
        for tau in candidates_to_keep:
            if t - tau < min_size:
                R_new.append(tau)
                continue
            seg_cost = cost_fn.cost(tau, t)
            if F[tau] + seg_cost <= F[t]:
                R_new.append(tau)

        R_new.append(t)
        R = R_new

    return sorted(cp[n])


class PELTSolver:
    """PELT (Pruned Exact Linear Time) changepoint detector.

    Finds the optimal set of changepoints that minimise
        sum_{segments} cost(segment) + C * n_changepoints
    using the pruning strategy of Killick et al. (JASA, 2012).

    Parameters
    ----------
    cost_type : Cost function to use (default: L2)
    penalty_type : Penalty selection strategy (default: BIC)
    penalty_value : Manual penalty value (only for MANUAL penalty_type)
    min_size : Minimum segment length (default: 2)

    Examples
    --------
    >>> solver = PELTSolver(cost_type="l2", penalty_type="bic")
    >>> result = solver.fit(signal)
    >>> print(result.changepoints)
    """

    def __init__(
        self,
        cost_type: CostType | str = CostType.L2,
        penalty_type: PenaltyType | str = PenaltyType.BIC,
        penalty_value: Optional[float] = None,
        min_size: int = 2,
        **cost_kwargs,
    ):
        self.cost_type = CostType(cost_type) if isinstance(cost_type, str) else cost_type
        self.penalty_type = PenaltyType(penalty_type) if isinstance(penalty_type, str) else penalty_type
        self.penalty_value = penalty_value
        self.min_size = max(min_size, 2)
        self.cost_kwargs = cost_kwargs

    def fit(self, signal: NDArray) -> ChangepointResult:
        """Detect changepoints in the given signal.

        Parameters
        ----------
        signal : array-like, shape (n,) or (n, d)

        Returns
        -------
        ChangepointResult with detected changepoints and segment info
        """
        signal = _validate_signal(signal)
        n, d = signal.shape

        # Build cost function
        cost_fn = make_cost_function(signal, self.cost_type, **self.cost_kwargs)

        # Compute penalty
        penalty = compute_penalty(
            n,
            n_dims=d,
            penalty_type=self.penalty_type,
            manual_value=self.penalty_value,
            signal=signal if self.penalty_type == PenaltyType.ELBOW else None,
        )

        # Run PELT
        cps = _pelt_core(cost_fn, n, penalty, self.min_size)

        # Build segments
        boundaries = [0] + cps + [n]
        segments = []
        total_cost = 0.0
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i + 1]
            sc = cost_fn.cost(s, e)
            total_cost += sc
            seg_data = signal[s:e]
            segments.append(Segment(
                start=s,
                end=e,
                cost=sc,
                mean=np.mean(seg_data, axis=0) if e > s else None,
                variance=float(np.mean(np.var(seg_data, axis=0))) if e - s > 1 else 0.0,
                count=e - s,
            ))

        return ChangepointResult(
            changepoints=cps,
            segments=segments,
            cost=total_cost,
            penalty=penalty,
            n_changepoints=len(cps),
            method="PELT",
            metadata={
                "cost_type": self.cost_type.value,
                "penalty_type": self.penalty_type.value,
                "min_size": self.min_size,
                "n_samples": n,
                "n_dims": d,
            },
        )

    def fit_range(
        self,
        signal: NDArray,
        penalties: Sequence[float],
    ) -> list[ChangepointResult]:
        """Run PELT with multiple penalty values for model selection.

        Parameters
        ----------
        signal : input signal
        penalties : sequence of penalty values to try

        Returns
        -------
        list of ChangepointResult, one per penalty value
        """
        signal = _validate_signal(signal)
        cost_fn = make_cost_function(signal, self.cost_type, **self.cost_kwargs)
        n = signal.shape[0]

        results = []
        for pen in penalties:
            cps = _pelt_core(cost_fn, n, pen, self.min_size)
            boundaries = [0] + cps + [n]
            segments = []
            total_cost = 0.0
            for i in range(len(boundaries) - 1):
                s, e = boundaries[i], boundaries[i + 1]
                sc = cost_fn.cost(s, e)
                total_cost += sc
                seg_data = signal[s:e]
                segments.append(Segment(
                    start=s,
                    end=e,
                    cost=sc,
                    mean=np.mean(seg_data, axis=0) if e > s else None,
                    variance=float(np.mean(np.var(seg_data, axis=0))) if e - s > 1 else 0.0,
                    count=e - s,
                ))
            results.append(ChangepointResult(
                changepoints=cps,
                segments=segments,
                cost=total_cost,
                penalty=pen,
                n_changepoints=len(cps),
                method="PELT",
            ))
        return results


# ---------------------------------------------------------------------------
# Binary Segmentation
# ---------------------------------------------------------------------------

class BinarySegmentation:
    """Binary segmentation changepoint detector.

    A greedy top-down algorithm that recursively splits the signal
    at the point yielding the largest cost reduction, until no split
    exceeds the penalty.

    Parameters
    ----------
    cost_type : Cost function type
    penalty_type : Penalty selection strategy
    penalty_value : Manual penalty (if MANUAL type)
    min_size : Minimum segment length
    max_changepoints : Maximum number of changepoints to detect

    Notes
    -----
    Binary segmentation is O(n log n) but not guaranteed optimal.
    Prefer PELT for exact solutions.
    """

    def __init__(
        self,
        cost_type: CostType | str = CostType.L2,
        penalty_type: PenaltyType | str = PenaltyType.BIC,
        penalty_value: Optional[float] = None,
        min_size: int = 2,
        max_changepoints: int = 100,
        **cost_kwargs,
    ):
        self.cost_type = CostType(cost_type) if isinstance(cost_type, str) else cost_type
        self.penalty_type = PenaltyType(penalty_type) if isinstance(penalty_type, str) else penalty_type
        self.penalty_value = penalty_value
        self.min_size = max(min_size, 2)
        self.max_changepoints = max_changepoints
        self.cost_kwargs = cost_kwargs

    def fit(self, signal: NDArray) -> ChangepointResult:
        """Detect changepoints via binary segmentation.

        Parameters
        ----------
        signal : array-like, shape (n,) or (n, d)

        Returns
        -------
        ChangepointResult
        """
        signal = _validate_signal(signal)
        n, d = signal.shape

        cost_fn = make_cost_function(signal, self.cost_type, **self.cost_kwargs)
        penalty = compute_penalty(
            n,
            n_dims=d,
            penalty_type=self.penalty_type,
            manual_value=self.penalty_value,
        )

        changepoints: list[int] = []
        self._binary_split(cost_fn, 0, n, penalty, changepoints)
        changepoints.sort()

        # Truncate to max_changepoints
        if len(changepoints) > self.max_changepoints:
            # Keep the ones with largest cost reduction (re-evaluate)
            cp_gains = []
            for cp in changepoints:
                # Find enclosing segment
                left = 0
                right = n
                for c in sorted(changepoints):
                    if c < cp:
                        left = max(left, c)
                    elif c > cp:
                        right = min(right, c)
                        break
                gain = cost_fn.cost(left, right) - cost_fn.cost(left, cp) - cost_fn.cost(cp, right)
                cp_gains.append((gain, cp))
            cp_gains.sort(reverse=True)
            changepoints = sorted([cp for _, cp in cp_gains[:self.max_changepoints]])

        # Build segments
        boundaries = [0] + changepoints + [n]
        segments = []
        total_cost = 0.0
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i + 1]
            sc = cost_fn.cost(s, e)
            total_cost += sc
            seg_data = signal[s:e]
            segments.append(Segment(
                start=s,
                end=e,
                cost=sc,
                mean=np.mean(seg_data, axis=0) if e > s else None,
                variance=float(np.mean(np.var(seg_data, axis=0))) if e - s > 1 else 0.0,
                count=e - s,
            ))

        return ChangepointResult(
            changepoints=changepoints,
            segments=segments,
            cost=total_cost,
            penalty=penalty,
            n_changepoints=len(changepoints),
            method="BinarySegmentation",
            metadata={
                "cost_type": self.cost_type.value,
                "penalty_type": self.penalty_type.value,
                "min_size": self.min_size,
                "max_changepoints": self.max_changepoints,
            },
        )

    def _binary_split(
        self,
        cost_fn: CostFunction,
        start: int,
        end: int,
        penalty: float,
        changepoints: list[int],
    ) -> None:
        """Recursively find the best split in [start, end)."""
        if end - start < 2 * self.min_size:
            return
        if len(changepoints) >= self.max_changepoints:
            return

        full_cost = cost_fn.cost(start, end)
        best_gain = -np.inf
        best_cp = -1

        for t in range(start + self.min_size, end - self.min_size + 1):
            left_cost = cost_fn.cost(start, t)
            right_cost = cost_fn.cost(t, end)
            gain = full_cost - left_cost - right_cost - penalty
            if gain > best_gain:
                best_gain = gain
                best_cp = t

        if best_gain > 0 and best_cp > 0:
            changepoints.append(best_cp)
            self._binary_split(cost_fn, start, best_cp, penalty, changepoints)
            self._binary_split(cost_fn, best_cp, end, penalty, changepoints)


# ---------------------------------------------------------------------------
# CUSUM Detector
# ---------------------------------------------------------------------------

class CUSUMDetector:
    """Cumulative Sum (CUSUM) changepoint detector.

    Detects a single changepoint by finding the point where the
    cumulative deviation from the overall mean is maximised.
    Can be applied iteratively for multiple changepoints.

    Parameters
    ----------
    threshold : Detection threshold (if None, uses permutation test)
    n_permutations : Number of permutations for threshold estimation
    min_size : Minimum segment length
    max_changepoints : Maximum changepoints to detect in iterative mode
    significance_level : Significance level for permutation test
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        n_permutations: int = 999,
        min_size: int = 2,
        max_changepoints: int = 10,
        significance_level: float = 0.05,
    ):
        self.threshold = threshold
        self.n_permutations = n_permutations
        self.min_size = max(min_size, 2)
        self.max_changepoints = max_changepoints
        self.significance_level = significance_level

    def fit(self, signal: NDArray) -> ChangepointResult:
        """Detect changepoints using iterative CUSUM.

        Parameters
        ----------
        signal : array-like, shape (n,) or (n, d)

        Returns
        -------
        ChangepointResult
        """
        signal = _validate_signal(signal)
        n, d = signal.shape

        changepoints: list[int] = []
        self._iterative_cusum(signal, 0, n, changepoints)
        changepoints.sort()

        # Build segments
        boundaries = [0] + changepoints + [n]
        cost_fn = L2Cost(signal)
        segments = []
        total_cost = 0.0
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i + 1]
            sc = cost_fn.cost(s, e)
            total_cost += sc
            seg_data = signal[s:e]
            segments.append(Segment(
                start=s,
                end=e,
                cost=sc,
                mean=np.mean(seg_data, axis=0) if e > s else None,
                variance=float(np.mean(np.var(seg_data, axis=0))) if e - s > 1 else 0.0,
                count=e - s,
            ))

        return ChangepointResult(
            changepoints=changepoints,
            segments=segments,
            cost=total_cost,
            penalty=0.0,
            n_changepoints=len(changepoints),
            method="CUSUM",
            metadata={
                "threshold": self.threshold,
                "n_permutations": self.n_permutations,
                "significance_level": self.significance_level,
            },
        )

    def _cusum_statistic(self, signal: NDArray, start: int, end: int) -> tuple[float, int]:
        """Compute CUSUM statistic and location for signal[start:end].

        Returns (max_cusum, changepoint_location)
        """
        seg = signal[start:end]
        n = seg.shape[0]
        if n < 2 * self.min_size:
            return 0.0, start

        mean = np.mean(seg, axis=0)
        cusum = np.cumsum(seg - mean, axis=0)
        # Use L2 norm for multivariate
        cusum_norm = np.sqrt(np.sum(cusum ** 2, axis=1))

        # Only consider valid positions
        valid_start = self.min_size
        valid_end = n - self.min_size + 1
        if valid_start >= valid_end:
            return 0.0, start

        cusum_valid = cusum_norm[valid_start:valid_end]
        if len(cusum_valid) == 0:
            return 0.0, start

        best_idx = int(np.argmax(cusum_valid))
        return float(cusum_valid[best_idx]), start + valid_start + best_idx

    def _permutation_threshold(self, signal: NDArray, start: int, end: int) -> float:
        """Compute CUSUM threshold via permutation test."""
        seg = signal[start:end]
        n = seg.shape[0]
        rng = np.random.default_rng(42)

        null_stats = np.zeros(self.n_permutations)
        for b in range(self.n_permutations):
            perm = rng.permutation(n)
            perm_seg = seg[perm]
            null_stat, _ = self._cusum_statistic(
                perm_seg.reshape(1, n, -1).squeeze(0) if perm_seg.ndim == 1 else perm_seg,
                0,
                n,
            )
            null_stats[b] = null_stat

        return float(np.percentile(null_stats, 100 * (1 - self.significance_level)))

    def _iterative_cusum(
        self,
        signal: NDArray,
        start: int,
        end: int,
        changepoints: list[int],
    ) -> None:
        """Iteratively apply CUSUM to detect multiple changepoints."""
        if end - start < 2 * self.min_size:
            return
        if len(changepoints) >= self.max_changepoints:
            return

        stat, cp = self._cusum_statistic(signal, start, end)

        # Determine threshold
        if self.threshold is not None:
            thresh = self.threshold
        else:
            thresh = self._permutation_threshold(signal, start, end)

        if stat > thresh:
            changepoints.append(cp)
            self._iterative_cusum(signal, start, cp, changepoints)
            self._iterative_cusum(signal, cp, end, changepoints)

    def single_changepoint(self, signal: NDArray) -> tuple[Optional[int], float, float]:
        """Detect a single changepoint with p-value.

        Returns
        -------
        (changepoint, statistic, p_value) or (None, 0, 1.0) if no changepoint
        """
        signal = _validate_signal(signal)
        n = signal.shape[0]

        stat, cp = self._cusum_statistic(signal, 0, n)

        if stat <= 0:
            return None, 0.0, 1.0

        # Permutation p-value
        rng = np.random.default_rng(42)
        count_greater = 0
        for _ in range(self.n_permutations):
            perm = rng.permutation(n)
            null_stat, _ = self._cusum_statistic(signal[perm], 0, n)
            if null_stat >= stat:
                count_greater += 1

        p_value = (count_greater + 1) / (self.n_permutations + 1)
        if p_value > self.significance_level:
            return None, stat, p_value

        return cp, stat, p_value


# ---------------------------------------------------------------------------
# Utility: run multiple detectors and combine
# ---------------------------------------------------------------------------

def ensemble_changepoints(
    signal: NDArray,
    detectors: Optional[list] = None,
    vote_threshold: float = 0.5,
) -> ChangepointResult:
    """Combine changepoints from multiple detectors via voting.

    A changepoint is included if detected by at least ``vote_threshold``
    fraction of detectors.

    Parameters
    ----------
    signal : input signal
    detectors : list of detector instances (default: PELT + BinSeg + CUSUM)
    vote_threshold : fraction of detectors that must agree

    Returns
    -------
    ChangepointResult with consensus changepoints
    """
    signal = _validate_signal(signal)
    n = signal.shape[0]

    if detectors is None:
        detectors = [
            PELTSolver(cost_type="l2"),
            BinarySegmentation(cost_type="l2"),
            CUSUMDetector(n_permutations=199),
        ]

    all_cps: list[list[int]] = []
    for det in detectors:
        result = det.fit(signal)
        all_cps.append(result.changepoints)

    # Vote: for each time point, count how many detectors placed a CP nearby
    votes = np.zeros(n, dtype=np.float64)
    tolerance = max(2, n // 50)  # Allow +/- tolerance

    for cps in all_cps:
        for cp in cps:
            lo = max(0, cp - tolerance)
            hi = min(n, cp + tolerance + 1)
            votes[lo:hi] += 1.0

    # Normalize
    n_detectors = len(detectors)
    votes /= n_detectors

    # Find peaks above threshold
    consensus_cps = []
    above = votes >= vote_threshold
    if np.any(above):
        # Find contiguous regions above threshold and pick peak of each
        in_region = False
        region_start = 0
        for i in range(n):
            if above[i] and not in_region:
                in_region = True
                region_start = i
            elif not above[i] and in_region:
                in_region = False
                peak = region_start + int(np.argmax(votes[region_start:i]))
                consensus_cps.append(peak)
        if in_region:
            peak = region_start + int(np.argmax(votes[region_start:n]))
            consensus_cps.append(peak)

    # Build segments
    cost_fn = L2Cost(signal)
    boundaries = [0] + consensus_cps + [n]
    segments = []
    total_cost = 0.0
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        sc = cost_fn.cost(s, e)
        total_cost += sc
        seg_data = signal[s:e]
        segments.append(Segment(
            start=s,
            end=e,
            cost=sc,
            mean=np.mean(seg_data, axis=0) if e > s else None,
            variance=float(np.mean(np.var(seg_data, axis=0))) if e - s > 1 else 0.0,
            count=e - s,
        ))

    return ChangepointResult(
        changepoints=consensus_cps,
        segments=segments,
        cost=total_cost,
        penalty=0.0,
        n_changepoints=len(consensus_cps),
        method="Ensemble",
        metadata={
            "detectors": [type(d).__name__ for d in detectors],
            "vote_threshold": vote_threshold,
            "tolerance": tolerance,
        },
    )


# ---------------------------------------------------------------------------
# Convenience: segment signal
# ---------------------------------------------------------------------------

def segment_signal(
    signal: NDArray,
    method: str = "pelt",
    **kwargs,
) -> ChangepointResult:
    """One-line convenience for changepoint detection.

    Parameters
    ----------
    signal : input signal
    method : "pelt", "binseg", "cusum", or "ensemble"
    **kwargs : passed to the detector

    Returns
    -------
    ChangepointResult
    """
    signal = _validate_signal(signal)
    if method == "pelt":
        return PELTSolver(**kwargs).fit(signal)
    elif method == "binseg":
        return BinarySegmentation(**kwargs).fit(signal)
    elif method == "cusum":
        return CUSUMDetector(**kwargs).fit(signal)
    elif method == "ensemble":
        return ensemble_changepoints(signal, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use pelt, binseg, cusum, or ensemble.")
