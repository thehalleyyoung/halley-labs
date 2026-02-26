"""
Adaptive discretization of continuous variables for junction-tree inference.

Provides several binning strategies—uniform, quantile, tail-preserving,
entropy-optimal—and supports dynamic re-discretization driven by
inference results and error-bound estimation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats
from scipy.optimize import minimize_scalar


# ------------------------------------------------------------------ #
#  Strategy enum
# ------------------------------------------------------------------ #

class BinningStrategy(Enum):
    UNIFORM = "uniform"
    QUANTILE = "quantile"
    TAIL_PRESERVING = "tail_preserving"
    ENTROPY_OPTIMAL = "entropy_optimal"
    INSTRUMENT_SPECIFIC = "instrument_specific"


# ------------------------------------------------------------------ #
#  Binning result
# ------------------------------------------------------------------ #

@dataclass
class DiscretizationResult:
    """Stores the output of a discretization pass."""

    bin_edges: NDArray
    bin_centers: NDArray
    bin_widths: NDArray
    n_bins: int
    strategy: BinningStrategy
    error_bound: float = 0.0
    kl_divergence: float = 0.0
    variable_name: str = ""

    @property
    def cardinality(self) -> int:
        return self.n_bins

    def bin_index(self, value: float) -> int:
        """Map a continuous value to its bin index (clipped)."""
        idx = int(np.searchsorted(self.bin_edges[1:-1], value))
        return min(max(idx, 0), self.n_bins - 1)

    def bin_probabilities(self, values: NDArray) -> NDArray:
        """Compute a histogram (probability vector) from raw values."""
        counts, _ = np.histogram(values, bins=self.bin_edges)
        total = counts.sum()
        if total == 0:
            return np.ones(self.n_bins) / self.n_bins
        return counts.astype(np.float64) / total


# ------------------------------------------------------------------ #
#  Main discretizer
# ------------------------------------------------------------------ #

class AdaptiveDiscretizer:
    """Adaptive discretization engine for continuous random variables.

    Parameters
    ----------
    default_bins : int
        Default number of bins when none is specified.
    default_strategy : BinningStrategy
        Default binning strategy.
    tail_fraction : float
        Fraction of probability mass to allocate to each tail when
        using tail-preserving discretization.
    min_bins : int
        Minimum number of bins during refinement.
    max_bins : int
        Maximum number of bins during refinement.
    """

    def __init__(
        self,
        default_bins: int = 20,
        default_strategy: BinningStrategy = BinningStrategy.QUANTILE,
        tail_fraction: float = 0.05,
        min_bins: int = 5,
        max_bins: int = 200,
    ) -> None:
        self.default_bins = default_bins
        self.default_strategy = default_strategy
        self.tail_fraction = tail_fraction
        self.min_bins = min_bins
        self.max_bins = max_bins

        # Cache of per-variable results
        self._cache: Dict[str, DiscretizationResult] = {}

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def discretize(
        self,
        values: NDArray,
        n_bins: Optional[int] = None,
        strategy: Optional[BinningStrategy] = None,
        variable_name: str = "",
        domain: Optional[Tuple[float, float]] = None,
    ) -> DiscretizationResult:
        """Discretize an array of continuous samples.

        Parameters
        ----------
        values : 1-D array of observed values.
        n_bins : number of bins (defaults to ``self.default_bins``).
        strategy : binning strategy (defaults to ``self.default_strategy``).
        variable_name : optional label for caching / error reporting.
        domain : optional (lo, hi) clipping domain.

        Returns
        -------
        DiscretizationResult
        """
        n_bins = n_bins or self.default_bins
        strategy = strategy or self.default_strategy
        values = np.asarray(values, dtype=np.float64).ravel()

        if domain is not None:
            values = np.clip(values, domain[0], domain[1])

        if len(values) == 0:
            lo, hi = (0.0, 1.0) if domain is None else domain
            edges = np.linspace(lo, hi, n_bins + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            return DiscretizationResult(
                bin_edges=edges,
                bin_centers=centers,
                bin_widths=np.diff(edges),
                n_bins=n_bins,
                strategy=strategy,
                variable_name=variable_name,
            )

        if strategy == BinningStrategy.UNIFORM:
            result = self._uniform(values, n_bins)
        elif strategy == BinningStrategy.QUANTILE:
            result = self._quantile(values, n_bins)
        elif strategy == BinningStrategy.TAIL_PRESERVING:
            result = self._tail_preserving(values, n_bins)
        elif strategy == BinningStrategy.ENTROPY_OPTIMAL:
            result = self._entropy_optimal(values, n_bins)
        elif strategy == BinningStrategy.INSTRUMENT_SPECIFIC:
            result = self._instrument_specific(values, n_bins)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        result.variable_name = variable_name
        result.strategy = strategy
        result.error_bound = self.compute_discretization_error(values, result)

        if variable_name:
            self._cache[variable_name] = result

        return result

    def get_bin_edges(self, variable_name: str) -> NDArray:
        """Return cached bin edges for a previously discretized variable."""
        if variable_name not in self._cache:
            raise KeyError(f"No discretization cached for '{variable_name}'")
        return self._cache[variable_name].bin_edges.copy()

    def get_result(self, variable_name: str) -> DiscretizationResult:
        """Retrieve cached discretization result."""
        if variable_name not in self._cache:
            raise KeyError(f"No discretization cached for '{variable_name}'")
        return self._cache[variable_name]

    def compute_discretization_error(
        self, values: NDArray, result: DiscretizationResult
    ) -> float:
        """Estimate discretization error as KL divergence between
        a KDE of the true density and the piecewise-constant approximation.

        Returns the estimated KL(true || discretized).
        """
        values = np.asarray(values, dtype=np.float64).ravel()
        if len(values) < 5 or result.n_bins < 2:
            return 0.0

        # Fit KDE
        try:
            kde = sp_stats.gaussian_kde(values)
        except (np.linalg.LinAlgError, ValueError):
            return 0.0

        # Evaluate KDE at bin centers
        p_true = kde(result.bin_centers)
        p_true = np.maximum(p_true, 1e-300)
        p_true /= p_true.sum()

        # Histogram density
        p_disc = result.bin_probabilities(values)
        p_disc = np.maximum(p_disc, 1e-300)
        p_disc /= p_disc.sum()

        kl = float(np.sum(p_true * (np.log(p_true) - np.log(p_disc))))
        result.kl_divergence = max(kl, 0.0)
        return max(kl, 0.0)

    def refine(
        self,
        values: NDArray,
        target_error: float,
        variable_name: str = "",
        strategy: Optional[BinningStrategy] = None,
    ) -> DiscretizationResult:
        """Iteratively increase bin count until discretization error
        drops below ``target_error`` or ``max_bins`` is reached.

        Uses a doubling search followed by binary search on bin count.
        """
        strategy = strategy or self.default_strategy
        values = np.asarray(values, dtype=np.float64).ravel()

        # Phase 1: doubling to find upper bound on n_bins
        lo_bins = self.min_bins
        hi_bins = self.min_bins

        result = self.discretize(values, lo_bins, strategy, variable_name)
        if result.error_bound <= target_error:
            return result

        while hi_bins < self.max_bins:
            hi_bins = min(hi_bins * 2, self.max_bins)
            result = self.discretize(values, hi_bins, strategy, variable_name)
            if result.error_bound <= target_error:
                break

        if result.error_bound > target_error:
            return result  # best we can do

        # Phase 2: binary search between lo_bins and hi_bins
        while hi_bins - lo_bins > 1:
            mid = (lo_bins + hi_bins) // 2
            result = self.discretize(values, mid, strategy, variable_name)
            if result.error_bound <= target_error:
                hi_bins = mid
            else:
                lo_bins = mid

        return self.discretize(values, hi_bins, strategy, variable_name)

    def rediscretize(
        self,
        variable_name: str,
        posterior_probs: NDArray,
        values: NDArray,
        concentration_factor: float = 2.0,
    ) -> DiscretizationResult:
        """Re-discretize a variable based on posterior probabilities.

        Places more bins in regions of high posterior mass, improving
        accuracy where it matters most for downstream inference.
        """
        if variable_name not in self._cache:
            return self.discretize(
                values, variable_name=variable_name,
                strategy=BinningStrategy.QUANTILE,
            )

        old_result = self._cache[variable_name]
        n_bins = old_result.n_bins
        posterior_probs = np.asarray(posterior_probs, dtype=np.float64).ravel()

        if len(posterior_probs) != n_bins:
            return self.discretize(
                values, n_bins, variable_name=variable_name,
                strategy=old_result.strategy,
            )

        # Compute density to allocate bins: raise posterior to a power
        density = np.power(np.maximum(posterior_probs, 1e-10), concentration_factor)
        density /= density.sum()

        # Cumulative density → quantile edges
        cum_density = np.concatenate([[0.0], np.cumsum(density)])
        cum_density /= cum_density[-1]

        lo = old_result.bin_edges[0]
        hi = old_result.bin_edges[-1]
        uniform_cdf = np.linspace(0, 1, n_bins + 1)

        # Interpolate to get new bin edges
        new_edges = np.interp(uniform_cdf, cum_density, old_result.bin_edges)
        new_edges[0] = lo
        new_edges[-1] = hi

        # Ensure monotonically increasing
        for i in range(1, len(new_edges)):
            if new_edges[i] <= new_edges[i - 1]:
                new_edges[i] = new_edges[i - 1] + 1e-10

        centers = 0.5 * (new_edges[:-1] + new_edges[1:])
        widths = np.diff(new_edges)

        result = DiscretizationResult(
            bin_edges=new_edges,
            bin_centers=centers,
            bin_widths=widths,
            n_bins=n_bins,
            strategy=BinningStrategy.QUANTILE,
            variable_name=variable_name,
        )
        result.error_bound = self.compute_discretization_error(values, result)
        self._cache[variable_name] = result
        return result

    # ------------------------------------------------------------------ #
    #  Binning strategies (private)
    # ------------------------------------------------------------------ #

    def _uniform(self, values: NDArray, n_bins: int) -> DiscretizationResult:
        """Equal-width binning."""
        lo, hi = float(values.min()), float(values.max())
        if lo == hi:
            lo -= 0.5
            hi += 0.5
        edges = np.linspace(lo, hi, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return DiscretizationResult(
            bin_edges=edges,
            bin_centers=centers,
            bin_widths=np.diff(edges),
            n_bins=n_bins,
            strategy=BinningStrategy.UNIFORM,
        )

    def _quantile(self, values: NDArray, n_bins: int) -> DiscretizationResult:
        """Equal-count (quantile) binning."""
        quantiles = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(values, quantiles)

        # Ensure strictly increasing
        edges = np.unique(edges)
        if len(edges) < 2:
            return self._uniform(values, n_bins)

        actual_bins = len(edges) - 1
        centers = 0.5 * (edges[:-1] + edges[1:])
        return DiscretizationResult(
            bin_edges=edges,
            bin_centers=centers,
            bin_widths=np.diff(edges),
            n_bins=actual_bins,
            strategy=BinningStrategy.QUANTILE,
        )

    def _tail_preserving(
        self, values: NDArray, n_bins: int
    ) -> DiscretizationResult:
        """Tail-preserving quantization.

        Allocates extra bins in the tails (below ``tail_fraction`` and
        above ``1 - tail_fraction``) to accurately capture extreme
        quantiles relevant to systemic-risk measurement.
        """
        tf = self.tail_fraction
        n_tail = max(int(n_bins * 0.3), 2)  # 30% of bins per tail
        n_body = max(n_bins - 2 * n_tail, 2)

        lo_q = np.percentile(values, tf * 100)
        hi_q = np.percentile(values, (1 - tf) * 100)
        lo_val = float(values.min())
        hi_val = float(values.max())

        # Tail bins
        left_edges = np.linspace(lo_val, lo_q, n_tail + 1)
        right_edges = np.linspace(hi_q, hi_val, n_tail + 1)

        # Body bins (quantile-based within the central region)
        body_mask = (values >= lo_q) & (values <= hi_q)
        body_values = values[body_mask] if body_mask.any() else values
        body_quantiles = np.linspace(0, 100, n_body + 1)
        body_edges = np.percentile(body_values, body_quantiles)

        # Merge edges
        edges = np.concatenate([
            left_edges[:-1],
            body_edges,
            right_edges[1:],
        ])
        edges = np.unique(edges)

        actual_bins = len(edges) - 1
        centers = 0.5 * (edges[:-1] + edges[1:])
        return DiscretizationResult(
            bin_edges=edges,
            bin_centers=centers,
            bin_widths=np.diff(edges),
            n_bins=actual_bins,
            strategy=BinningStrategy.TAIL_PRESERVING,
        )

    def _entropy_optimal(
        self, values: NDArray, n_bins: int
    ) -> DiscretizationResult:
        """Entropy-optimal binning.

        Finds bin edges that maximise the entropy of the resulting
        histogram, which minimises information loss under a fixed
        bin-count constraint.

        Uses an iterative Lloyd-Max style algorithm:
        1. Start with quantile-based edges.
        2. Compute histogram → entropy.
        3. Perturb edges to increase entropy.
        4. Iterate until convergence.
        """
        # Initialise with quantile edges
        init = self._quantile(values, n_bins)
        edges = init.bin_edges.copy()
        actual_bins = len(edges) - 1

        best_entropy = self._histogram_entropy(values, edges)
        best_edges = edges.copy()

        rng = np.random.RandomState(42)
        temperature = 0.1 * (edges[-1] - edges[0]) / actual_bins

        for iteration in range(200):
            # Perturb interior edges
            candidate = best_edges.copy()
            idx = rng.randint(1, len(candidate) - 1)
            lo_bound = candidate[idx - 1] + 1e-10
            hi_bound = candidate[idx + 1] - 1e-10 if idx + 1 < len(candidate) else candidate[-1]
            if lo_bound >= hi_bound:
                continue
            perturbation = rng.normal(0, temperature)
            candidate[idx] = np.clip(
                candidate[idx] + perturbation, lo_bound, hi_bound
            )

            ent = self._histogram_entropy(values, candidate)
            if ent > best_entropy:
                best_entropy = ent
                best_edges = candidate
            else:
                # Simulated-annealing acceptance
                delta = ent - best_entropy
                if rng.random() < math.exp(delta / max(temperature, 1e-10)):
                    best_entropy = ent
                    best_edges = candidate

            temperature *= 0.995  # cooling

        edges = best_edges
        centers = 0.5 * (edges[:-1] + edges[1:])
        return DiscretizationResult(
            bin_edges=edges,
            bin_centers=centers,
            bin_widths=np.diff(edges),
            n_bins=len(edges) - 1,
            strategy=BinningStrategy.ENTROPY_OPTIMAL,
        )

    def _instrument_specific(
        self, values: NDArray, n_bins: int
    ) -> DiscretizationResult:
        """Instrument-specific discretization for financial variables.

        Uses log-transform for positive-valued instruments (e.g. prices,
        spreads) and tail-preserving for others.
        """
        if np.all(values > 0):
            log_vals = np.log(values)
            inner = self._quantile(log_vals, n_bins)
            edges = np.exp(inner.bin_edges)
            centers = np.exp(inner.bin_centers)
            return DiscretizationResult(
                bin_edges=edges,
                bin_centers=centers,
                bin_widths=np.diff(edges),
                n_bins=inner.n_bins,
                strategy=BinningStrategy.INSTRUMENT_SPECIFIC,
            )
        else:
            return self._tail_preserving(values, n_bins)

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _histogram_entropy(self, values: NDArray, edges: NDArray) -> float:
        """Compute Shannon entropy of the histogram defined by ``edges``."""
        counts, _ = np.histogram(values, bins=edges)
        total = counts.sum()
        if total == 0:
            return 0.0
        p = counts / total
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    def clear_cache(self) -> None:
        """Clear all cached discretizations."""
        self._cache.clear()

    def cached_variables(self) -> List[str]:
        """Return list of variable names with cached discretizations."""
        return list(self._cache.keys())

    def summary(self) -> Dict[str, Dict]:
        """Return a summary dict of all cached discretizations."""
        out: Dict[str, Dict] = {}
        for name, res in self._cache.items():
            out[name] = {
                "n_bins": res.n_bins,
                "strategy": res.strategy.value,
                "error_bound": res.error_bound,
                "domain": (float(res.bin_edges[0]), float(res.bin_edges[-1])),
            }
        return out


# ------------------------------------------------------------------ #
#  Convenience functions
# ------------------------------------------------------------------ #

def discretize_all(
    data: Dict[str, NDArray],
    discretizer: Optional[AdaptiveDiscretizer] = None,
    n_bins: int = 20,
    strategy: BinningStrategy = BinningStrategy.QUANTILE,
) -> Dict[str, DiscretizationResult]:
    """Discretize every variable in ``data`` (a name→values dict)."""
    if discretizer is None:
        discretizer = AdaptiveDiscretizer(default_bins=n_bins, default_strategy=strategy)
    results: Dict[str, DiscretizationResult] = {}
    for name, values in data.items():
        results[name] = discretizer.discretize(
            values, n_bins=n_bins, strategy=strategy, variable_name=name
        )
    return results


def refine_all(
    data: Dict[str, NDArray],
    target_error: float,
    discretizer: Optional[AdaptiveDiscretizer] = None,
    strategy: BinningStrategy = BinningStrategy.QUANTILE,
) -> Dict[str, DiscretizationResult]:
    """Refine discretization for all variables to meet ``target_error``."""
    if discretizer is None:
        discretizer = AdaptiveDiscretizer(default_strategy=strategy)
    results: Dict[str, DiscretizationResult] = {}
    for name, values in data.items():
        results[name] = discretizer.refine(
            values, target_error, variable_name=name, strategy=strategy
        )
    return results
