"""Adaptive discretization of continuous payoff functions for CPD encoding.

Provides tail-preserving quantization, quantile-based binning,
piecewise-linear approximation, and error-bounded discretization
for mapping financial instrument payoffs into junction-tree CPDs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, interpolate, stats, integrate
from typing import Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class DiscretizationStrategy(Enum):
    """Strategy for discretizing a continuous payoff."""

    UNIFORM = "uniform"
    QUANTILE = "quantile"
    TAIL_PRESERVING = "tail_preserving"
    ADAPTIVE = "adaptive"
    CDS_FOCUSED = "cds_focused"
    IRS_FOCUSED = "irs_focused"
    OPTION_FOCUSED = "option_focused"
    ENTROPY_OPTIMAL = "entropy_optimal"


@dataclass
class DiscretizationResult:
    """Result of a discretization operation."""

    bin_edges: NDArray[np.float64]
    bin_centers: NDArray[np.float64]
    bin_weights: NDArray[np.float64]
    approximation_error: float
    n_bins: int
    strategy: str

    def lookup(self, value: float) -> int:
        """Find the bin index for a given value.

        Parameters
        ----------
        value : float
            Value to look up.

        Returns
        -------
        int
            Bin index.
        """
        idx = np.searchsorted(self.bin_edges[1:], value)
        return min(idx, self.n_bins - 1)

    def represent(self, value: float) -> float:
        """Map a value to its bin center.

        Parameters
        ----------
        value : float
            Value to discretize.

        Returns
        -------
        float
            Bin center representing this value.
        """
        return self.bin_centers[self.lookup(value)]


class InstrumentDiscretizer:
    """Adaptive discretization engine for financial instrument payoffs.

    Converts continuous payoff functions into discrete representations
    suitable for conditional probability tables in junction-tree inference.

    Parameters
    ----------
    default_n_bins : int
        Default number of bins.
    default_strategy : DiscretizationStrategy
        Default discretization strategy.
    tail_quantile : float
        Quantile threshold for tail identification.
    """

    def __init__(
        self,
        default_n_bins: int = 50,
        default_strategy: DiscretizationStrategy = DiscretizationStrategy.ADAPTIVE,
        tail_quantile: float = 0.01,
    ) -> None:
        self.default_n_bins = default_n_bins
        self.default_strategy = default_strategy
        self.tail_quantile = tail_quantile

    def discretize_payoff(
        self,
        payoff_fn: Callable[[NDArray], NDArray],
        domain: Tuple[float, float],
        n_bins: Optional[int] = None,
        strategy: Optional[Union[DiscretizationStrategy, str]] = None,
        n_samples: int = 10000,
        rng: Optional[np.random.Generator] = None,
    ) -> DiscretizationResult:
        """Discretize a continuous payoff function.

        Parameters
        ----------
        payoff_fn : callable
            Function mapping NDArray -> NDArray.
        domain : tuple
            (lower, upper) bounds for the input domain.
        n_bins : int, optional
            Number of bins.
        strategy : DiscretizationStrategy or str, optional
            Discretization strategy.
        n_samples : int
            Samples for computing distribution.
        rng : Generator, optional

        Returns
        -------
        DiscretizationResult
        """
        if rng is None:
            rng = np.random.default_rng(42)

        nb = n_bins if n_bins is not None else self.default_n_bins

        if strategy is None:
            strat = self.default_strategy
        elif isinstance(strategy, str):
            strat = DiscretizationStrategy(strategy)
        else:
            strat = strategy

        # Sample payoff values
        x_samples = rng.uniform(domain[0], domain[1], size=n_samples)
        y_samples = payoff_fn(x_samples)
        y_samples = y_samples[np.isfinite(y_samples)]

        if len(y_samples) == 0:
            edges = np.linspace(domain[0], domain[1], nb + 1)
            return DiscretizationResult(
                bin_edges=edges,
                bin_centers=0.5 * (edges[:-1] + edges[1:]),
                bin_weights=np.ones(nb) / nb,
                approximation_error=float("inf"),
                n_bins=nb,
                strategy=strat.value,
            )

        if strat == DiscretizationStrategy.UNIFORM:
            return self._uniform_bins(y_samples, nb, strat)
        elif strat == DiscretizationStrategy.QUANTILE:
            return self._quantile_bins_impl(y_samples, nb, strat)
        elif strat == DiscretizationStrategy.TAIL_PRESERVING:
            return self._tail_preserving_impl(y_samples, nb, strat)
        elif strat == DiscretizationStrategy.ADAPTIVE:
            return self._adaptive_bins(payoff_fn, domain, y_samples, nb, strat)
        elif strat == DiscretizationStrategy.CDS_FOCUSED:
            return self._cds_focused_bins(y_samples, nb, strat)
        elif strat == DiscretizationStrategy.IRS_FOCUSED:
            return self._irs_focused_bins(y_samples, nb, strat)
        elif strat == DiscretizationStrategy.OPTION_FOCUSED:
            return self._option_focused_bins(y_samples, nb, strat)
        elif strat == DiscretizationStrategy.ENTROPY_OPTIMAL:
            return self._entropy_optimal_bins(y_samples, nb, strat)
        else:
            return self._uniform_bins(y_samples, nb, strat)

    def _uniform_bins(
        self, data: NDArray, n_bins: int, strat: DiscretizationStrategy
    ) -> DiscretizationResult:
        """Uniform-width bins across the data range."""
        edges = np.linspace(np.min(data), np.max(data), n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        weights = self._compute_weights(data, edges)
        error = self._mse_error(data, edges, centers)
        return DiscretizationResult(
            bin_edges=edges, bin_centers=centers, bin_weights=weights,
            approximation_error=error, n_bins=n_bins, strategy=strat.value,
        )

    def _quantile_bins_impl(
        self, data: NDArray, n_bins: int, strat: DiscretizationStrategy
    ) -> DiscretizationResult:
        """Quantile-based bins ensuring equal probability mass per bin."""
        quantiles = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(data, quantiles)
        # Ensure unique edges
        edges = np.unique(edges)
        if len(edges) < 3:
            edges = np.linspace(np.min(data), np.max(data), n_bins + 1)
        actual_bins = len(edges) - 1
        centers = np.array([
            np.mean(data[(data >= edges[i]) & (data < edges[i + 1])])
            if np.any((data >= edges[i]) & (data < edges[i + 1]))
            else 0.5 * (edges[i] + edges[i + 1])
            for i in range(actual_bins)
        ])
        weights = self._compute_weights(data, edges)
        error = self._mse_error(data, edges, centers)
        return DiscretizationResult(
            bin_edges=edges, bin_centers=centers, bin_weights=weights,
            approximation_error=error, n_bins=actual_bins, strategy=strat.value,
        )

    def _tail_preserving_impl(
        self, data: NDArray, n_bins: int, strat: DiscretizationStrategy
    ) -> DiscretizationResult:
        """Tail-preserving discretization with finer bins in tails."""
        tail_bins = max(int(n_bins * 0.3), 2)
        body_bins = n_bins - 2 * tail_bins

        lower_tail = np.percentile(data, self.tail_quantile * 100)
        upper_tail = np.percentile(data, (1 - self.tail_quantile) * 100)

        lower_edges = np.linspace(np.min(data), lower_tail, tail_bins + 1)
        body_edges = np.linspace(lower_tail, upper_tail, body_bins + 1)
        upper_edges = np.linspace(upper_tail, np.max(data), tail_bins + 1)

        edges = np.unique(np.concatenate([lower_edges, body_edges[1:], upper_edges[1:]]))
        actual_bins = len(edges) - 1
        centers = 0.5 * (edges[:-1] + edges[1:])
        weights = self._compute_weights(data, edges)
        error = self._mse_error(data, edges, centers)
        return DiscretizationResult(
            bin_edges=edges, bin_centers=centers, bin_weights=weights,
            approximation_error=error, n_bins=actual_bins, strategy=strat.value,
        )

    def _adaptive_bins(
        self,
        payoff_fn: Callable,
        domain: Tuple[float, float],
        data: NDArray,
        n_bins: int,
        strat: DiscretizationStrategy,
    ) -> DiscretizationResult:
        """Adaptive bins with more resolution where payoff curvature is high.

        Uses second-derivative estimation to allocate bins proportionally
        to local curvature.
        """
        # Estimate curvature on a fine grid
        fine_grid = np.linspace(domain[0], domain[1], 500)
        fine_vals = payoff_fn(fine_grid)

        # Second derivative via central differences
        h = fine_grid[1] - fine_grid[0]
        curvature = np.zeros(len(fine_grid))
        curvature[1:-1] = np.abs(
            fine_vals[2:] - 2 * fine_vals[1:-1] + fine_vals[:-2]
        ) / (h ** 2)
        curvature[0] = curvature[1]
        curvature[-1] = curvature[-2]

        # Regularize to avoid zero curvature regions getting no bins
        curvature += np.max(curvature) * 0.01

        # Cumulative curvature distribution
        cum_curv = np.cumsum(curvature)
        cum_curv /= cum_curv[-1]

        # Place bin edges at equal curvature intervals
        target_levels = np.linspace(0, 1, n_bins + 1)
        input_edges = np.interp(target_levels, cum_curv, fine_grid)
        input_edges[0] = domain[0]
        input_edges[-1] = domain[1]

        # Map input edges to output (payoff) space
        output_edges = payoff_fn(input_edges)
        output_edges = np.sort(np.unique(output_edges))

        if len(output_edges) < 3:
            return self._uniform_bins(data, n_bins, strat)

        actual_bins = len(output_edges) - 1
        centers = 0.5 * (output_edges[:-1] + output_edges[1:])
        weights = self._compute_weights(data, output_edges)
        error = self._mse_error(data, output_edges, centers)

        return DiscretizationResult(
            bin_edges=output_edges, bin_centers=centers, bin_weights=weights,
            approximation_error=error, n_bins=actual_bins, strategy=strat.value,
        )

    def _cds_focused_bins(
        self, data: NDArray, n_bins: int, strat: DiscretizationStrategy
    ) -> DiscretizationResult:
        """CDS-specific: concentrate bins around the default boundary.

        For CDS payoffs, the key transition is at the default/no-default
        boundary, so we allocate more bins around zero payoff.
        """
        # Allocate 40% of bins near zero, 30% each for positive/negative
        zero_bins = max(int(n_bins * 0.4), 3)
        pos_bins = max(int(n_bins * 0.3), 2)
        neg_bins = n_bins - zero_bins - pos_bins

        data_min = np.min(data)
        data_max = np.max(data)

        # Region around zero (default boundary)
        zero_width = max(abs(data_max), abs(data_min)) * 0.1
        zero_edges = np.linspace(-zero_width, zero_width, zero_bins + 1)

        neg_edges = np.linspace(data_min, -zero_width, neg_bins + 1)
        pos_edges = np.linspace(zero_width, data_max, pos_bins + 1)

        edges = np.unique(np.concatenate([neg_edges, zero_edges, pos_edges]))
        actual_bins = len(edges) - 1
        centers = 0.5 * (edges[:-1] + edges[1:])
        weights = self._compute_weights(data, edges)
        error = self._mse_error(data, edges, centers)

        return DiscretizationResult(
            bin_edges=edges, bin_centers=centers, bin_weights=weights,
            approximation_error=error, n_bins=actual_bins, strategy=strat.value,
        )

    def _irs_focused_bins(
        self, data: NDArray, n_bins: int, strat: DiscretizationStrategy
    ) -> DiscretizationResult:
        """IRS-specific: focus on rate move magnitudes.

        For IRS, concentrate bins around the current MTM region with
        gradual expansion for large rate moves.
        """
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        if iqr < 1e-10:
            iqr = np.std(data) * 2

        # Sinh-based spacing: fine near median, coarser at extremes
        center_bins = max(int(n_bins * 0.5), 3)
        tail_bins_each = (n_bins - center_bins) // 2

        center_edges = np.linspace(median - iqr, median + iqr, center_bins + 1)
        lower_edges = median - iqr - np.geomspace(
            0.1, abs(np.min(data) - median + iqr) + 0.1, tail_bins_each + 1
        )[::-1]
        upper_edges = median + iqr + np.geomspace(
            0.1, abs(np.max(data) - median - iqr) + 0.1, tail_bins_each + 1
        )

        edges = np.unique(np.concatenate([lower_edges, center_edges, upper_edges]))
        edges = np.sort(edges)
        actual_bins = len(edges) - 1
        centers = 0.5 * (edges[:-1] + edges[1:])
        weights = self._compute_weights(data, edges)
        error = self._mse_error(data, edges, centers)

        return DiscretizationResult(
            bin_edges=edges, bin_centers=centers, bin_weights=weights,
            approximation_error=error, n_bins=actual_bins, strategy=strat.value,
        )

    def _option_focused_bins(
        self, data: NDArray, n_bins: int, strat: DiscretizationStrategy
    ) -> DiscretizationResult:
        """Option-specific: focus on the strike region.

        For options, the kink at the strike creates a discontinuity
        in the payoff derivative that needs fine resolution.
        """
        # Most option payoffs cluster at zero (OTM) with a tail (ITM)
        zero_mass = np.mean(data < np.percentile(data, 10) + 1e-10)

        if zero_mass > 0.3:
            # Many zeros: give a single bin for zero, rest for positive
            zero_edge = np.percentile(data, max(zero_mass * 100, 5))
            pos_data = data[data > zero_edge]
            if len(pos_data) > 10:
                pos_edges = np.percentile(
                    pos_data,
                    np.linspace(0, 100, n_bins)
                )
                edges = np.unique(np.concatenate([[np.min(data)], [zero_edge], pos_edges]))
            else:
                edges = np.linspace(np.min(data), np.max(data), n_bins + 1)
        else:
            edges = np.linspace(np.min(data), np.max(data), n_bins + 1)

        edges = np.sort(np.unique(edges))
        actual_bins = len(edges) - 1
        centers = 0.5 * (edges[:-1] + edges[1:])
        weights = self._compute_weights(data, edges)
        error = self._mse_error(data, edges, centers)

        return DiscretizationResult(
            bin_edges=edges, bin_centers=centers, bin_weights=weights,
            approximation_error=error, n_bins=actual_bins, strategy=strat.value,
        )

    def _entropy_optimal_bins(
        self, data: NDArray, n_bins: int, strat: DiscretizationStrategy
    ) -> DiscretizationResult:
        """Entropy-optimal discretization (maximum entropy binning).

        Finds bin edges that maximize the entropy of the discretized
        distribution, equivalent to equal-probability bins for continuous data.
        """
        sorted_data = np.sort(data)
        n = len(sorted_data)
        target_count = n / n_bins

        edges = [sorted_data[0]]
        for b in range(1, n_bins):
            idx = min(int(b * target_count), n - 1)
            edges.append(sorted_data[idx])
        edges.append(sorted_data[-1])
        edges = np.unique(np.array(edges))

        actual_bins = len(edges) - 1
        centers = np.array([
            np.mean(data[(data >= edges[i]) & (data < edges[i + 1])])
            if np.any((data >= edges[i]) & (data < edges[i + 1]))
            else 0.5 * (edges[i] + edges[i + 1])
            for i in range(actual_bins)
        ])
        weights = self._compute_weights(data, edges)

        # Compute entropy
        probs = weights[weights > 0]
        entropy = -np.sum(probs * np.log(probs))

        error = self._mse_error(data, edges, centers)
        return DiscretizationResult(
            bin_edges=edges, bin_centers=centers, bin_weights=weights,
            approximation_error=error, n_bins=actual_bins, strategy=strat.value,
        )

    def tail_preserving_bins(
        self,
        distribution: Union[NDArray, stats.rv_continuous],
        n_bins: int,
        tail_weight: float = 0.3,
    ) -> DiscretizationResult:
        """Create tail-preserving bins from a distribution.

        Parameters
        ----------
        distribution : NDArray or rv_continuous
            Sample data or scipy distribution.
        n_bins : int
            Number of bins.
        tail_weight : float
            Fraction of bins allocated to each tail.

        Returns
        -------
        DiscretizationResult
        """
        if isinstance(distribution, np.ndarray):
            data = distribution
        else:
            data = distribution.rvs(size=10000)

        tail_bins = max(int(n_bins * tail_weight / 2), 1)
        body_bins = n_bins - 2 * tail_bins

        lower_q = self.tail_quantile
        upper_q = 1.0 - self.tail_quantile

        lower_edge = np.percentile(data, lower_q * 100)
        upper_edge = np.percentile(data, upper_q * 100)

        # Logarithmic spacing in tails for better resolution
        lower_data = data[data <= lower_edge]
        upper_data = data[data >= upper_edge]

        if len(lower_data) > tail_bins:
            lower_edges = np.percentile(
                lower_data, np.linspace(0, 100, tail_bins + 1)
            )
        else:
            lower_edges = np.linspace(np.min(data), lower_edge, tail_bins + 1)

        if len(upper_data) > tail_bins:
            upper_edges = np.percentile(
                upper_data, np.linspace(0, 100, tail_bins + 1)
            )
        else:
            upper_edges = np.linspace(upper_edge, np.max(data), tail_bins + 1)

        body_edges = np.linspace(lower_edge, upper_edge, body_bins + 1)

        edges = np.unique(np.concatenate([lower_edges, body_edges, upper_edges]))
        actual_bins = len(edges) - 1
        centers = 0.5 * (edges[:-1] + edges[1:])
        weights = self._compute_weights(data, edges)
        error = self._mse_error(data, edges, centers)

        return DiscretizationResult(
            bin_edges=edges, bin_centers=centers, bin_weights=weights,
            approximation_error=error, n_bins=actual_bins,
            strategy="tail_preserving",
        )

    def quantile_bins(
        self,
        data: NDArray,
        n_bins: int,
    ) -> DiscretizationResult:
        """Create quantile-based bins from data.

        Parameters
        ----------
        data : NDArray
            Sample data.
        n_bins : int
            Number of bins.

        Returns
        -------
        DiscretizationResult
        """
        return self._quantile_bins_impl(data, n_bins, DiscretizationStrategy.QUANTILE)

    def compute_approximation_error(
        self,
        original: NDArray,
        discretized: NDArray,
        metric: str = "mse",
    ) -> float:
        """Compute approximation error between original and discretized data.

        Parameters
        ----------
        original : NDArray
            Original continuous values.
        discretized : NDArray
            Discretized (binned) values.
        metric : str
            Error metric: 'mse', 'mae', 'max', 'kl_divergence'.

        Returns
        -------
        float
            Approximation error.
        """
        if metric == "mse":
            return float(np.mean((original - discretized) ** 2))
        elif metric == "mae":
            return float(np.mean(np.abs(original - discretized)))
        elif metric == "max":
            return float(np.max(np.abs(original - discretized)))
        elif metric == "kl_divergence":
            return self._kl_divergence(original, discretized)
        else:
            return float(np.mean((original - discretized) ** 2))

    def _kl_divergence(
        self, original: NDArray, discretized: NDArray, n_bins: int = 100
    ) -> float:
        """Compute KL divergence between empirical distributions.

        Parameters
        ----------
        original : NDArray
        discretized : NDArray
        n_bins : int

        Returns
        -------
        float
            KL(original || discretized).
        """
        all_data = np.concatenate([original, discretized])
        edges = np.linspace(np.min(all_data), np.max(all_data), n_bins + 1)

        p, _ = np.histogram(original, bins=edges, density=True)
        q, _ = np.histogram(discretized, bins=edges, density=True)

        p = p + 1e-10
        q = q + 1e-10
        p = p / p.sum()
        q = q / q.sum()

        return float(np.sum(p * np.log(p / q)))

    def optimal_bins(
        self,
        payoff_fn: Callable[[NDArray], NDArray],
        n_bins: int,
        criterion: str = "mse",
        domain: Tuple[float, float] = (-1.0, 1.0),
        n_samples: int = 10000,
        rng: Optional[np.random.Generator] = None,
    ) -> DiscretizationResult:
        """Find optimal bin edges minimizing a criterion.

        Evaluates all strategies and returns the best one.

        Parameters
        ----------
        payoff_fn : callable
        n_bins : int
        criterion : str
            'mse', 'mae', 'max', 'kl_divergence'.
        domain : tuple
        n_samples : int
        rng : Generator, optional

        Returns
        -------
        DiscretizationResult
            Best discretization found.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        strategies = [
            DiscretizationStrategy.UNIFORM,
            DiscretizationStrategy.QUANTILE,
            DiscretizationStrategy.TAIL_PRESERVING,
            DiscretizationStrategy.ADAPTIVE,
            DiscretizationStrategy.ENTROPY_OPTIMAL,
        ]

        best_result = None
        best_error = float("inf")

        x_test = rng.uniform(domain[0], domain[1], size=n_samples)
        y_test = payoff_fn(x_test)
        y_test = y_test[np.isfinite(y_test)]

        for strat in strategies:
            result = self.discretize_payoff(
                payoff_fn, domain, n_bins, strat, n_samples, rng
            )
            # Compute discretized version
            discretized = np.array([result.represent(y) for y in y_test])
            error = self.compute_approximation_error(y_test, discretized, criterion)

            if error < best_error:
                best_error = error
                best_result = result

        if best_result is not None:
            best_result.approximation_error = best_error
        return best_result

    def multidimensional_discretize(
        self,
        payoff_fn: Callable[..., NDArray],
        domains: List[Tuple[float, float]],
        n_bins_per_dim: Union[int, List[int]],
        strategy: DiscretizationStrategy = DiscretizationStrategy.ADAPTIVE,
        n_samples: int = 10000,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Union[List[DiscretizationResult], NDArray]]:
        """Multi-dimensional discretization for joint payoff distributions.

        Parameters
        ----------
        payoff_fn : callable
            Function of multiple variables.
        domains : list of tuples
            (lower, upper) for each dimension.
        n_bins_per_dim : int or list
            Bins per dimension.
        strategy : DiscretizationStrategy
        n_samples : int
        rng : Generator, optional

        Returns
        -------
        dict
            'marginal_discretizations': per-dimension results,
            'joint_cpd': joint probability table.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        n_dims = len(domains)
        if isinstance(n_bins_per_dim, int):
            bins_list = [n_bins_per_dim] * n_dims
        else:
            bins_list = list(n_bins_per_dim)

        # Sample in each dimension
        samples = np.column_stack([
            rng.uniform(domains[d][0], domains[d][1], size=n_samples)
            for d in range(n_dims)
        ])

        # Evaluate payoff
        if n_dims == 1:
            payoff_vals = payoff_fn(samples[:, 0])
        elif n_dims == 2:
            payoff_vals = payoff_fn(samples[:, 0], samples[:, 1])
        else:
            payoff_vals = payoff_fn(*[samples[:, d] for d in range(n_dims)])

        # Marginal discretizations
        marginals = []
        for d in range(n_dims):
            def marginal_fn(x, _d=d):
                fixed = np.full((len(x), n_dims), 0.0)
                fixed[:, _d] = x
                for dd in range(n_dims):
                    if dd != _d:
                        fixed[:, dd] = np.mean(samples[:, dd])
                if n_dims == 1:
                    return payoff_fn(fixed[:, 0])
                elif n_dims == 2:
                    return payoff_fn(fixed[:, 0], fixed[:, 1])
                return payoff_fn(*[fixed[:, dd] for dd in range(n_dims)])

            result = self.discretize_payoff(
                marginal_fn, domains[d], bins_list[d], strategy, n_samples, rng
            )
            marginals.append(result)

        # Build joint CPD
        shape = tuple(bins_list)
        joint = np.zeros(shape)
        for i in range(n_samples):
            indices = []
            for d in range(n_dims):
                idx = marginals[d].lookup(samples[i, d])
                indices.append(idx)
            try:
                joint[tuple(indices)] += 1
            except IndexError:
                pass

        total = joint.sum()
        if total > 0:
            joint /= total

        return {
            "marginal_discretizations": marginals,
            "joint_cpd": joint,
        }

    def piecewise_linear_approximation(
        self,
        payoff_fn: Callable[[NDArray], NDArray],
        domain: Tuple[float, float],
        n_segments: int = 20,
        error_tol: float = 0.01,
    ) -> Dict[str, NDArray]:
        """Piecewise-linear approximation of a payoff function.

        Uses adaptive refinement to achieve the target error tolerance.

        Parameters
        ----------
        payoff_fn : callable
        domain : tuple
        n_segments : int
            Initial number of segments.
        error_tol : float
            Target maximum pointwise error.

        Returns
        -------
        dict
            'knots': breakpoints,
            'values': payoff at breakpoints,
            'max_error': achieved maximum error.
        """
        knots = np.linspace(domain[0], domain[1], n_segments + 1)
        values = payoff_fn(knots)

        max_iterations = 5
        for _ in range(max_iterations):
            # Evaluate midpoint errors
            midpoints = 0.5 * (knots[:-1] + knots[1:])
            mid_exact = payoff_fn(midpoints)
            mid_approx = 0.5 * (values[:-1] + values[1:])
            errors = np.abs(mid_exact - mid_approx)

            max_err = np.max(errors)
            if max_err <= error_tol:
                break

            # Refine segments with largest errors
            refine_mask = errors > error_tol * 0.5
            new_knots = [knots[0]]
            new_values = [values[0]]
            for i in range(len(midpoints)):
                if refine_mask[i]:
                    new_knots.append(midpoints[i])
                    new_values.append(mid_exact[i])
                new_knots.append(knots[i + 1])
                new_values.append(values[i + 1])
            knots = np.array(new_knots)
            values = np.array(new_values)

        # Final error computation
        test_points = np.linspace(domain[0], domain[1], 1000)
        exact = payoff_fn(test_points)
        approx = np.interp(test_points, knots, values)
        max_error = float(np.max(np.abs(exact - approx)))

        return {
            "knots": knots,
            "values": values,
            "max_error": max_error,
        }

    @staticmethod
    def _compute_weights(data: NDArray, edges: NDArray) -> NDArray:
        """Compute probability weights for each bin."""
        counts, _ = np.histogram(data, bins=edges)
        total = counts.sum()
        if total > 0:
            return counts / total
        return np.ones(len(edges) - 1) / (len(edges) - 1)

    @staticmethod
    def _mse_error(
        data: NDArray, edges: NDArray, centers: NDArray
    ) -> float:
        """Compute MSE of discretizing data into given bins."""
        indices = np.searchsorted(edges[1:-1], data)
        indices = np.clip(indices, 0, len(centers) - 1)
        discretized = centers[indices]
        return float(np.mean((data - discretized) ** 2))
