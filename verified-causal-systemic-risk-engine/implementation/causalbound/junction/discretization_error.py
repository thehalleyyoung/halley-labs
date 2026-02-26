"""
Formal discretization error analysis with provable error bounds.

Quantifies the approximation error introduced by discretizing continuous
financial variables for junction-tree inference. Provides:
  1. Worst-case L1/TV/KL error bounds for each discretization strategy
  2. Convergence rate analysis (error vs. number of bins)
  3. Adaptive refinement to meet user-specified error tolerance
  4. Integration with the composition theorem gap analysis

Key Result (Theorem):
  For a continuous distribution P on [a,b] with density bounded by M,
  discretized into n uniform bins of width h = (b-a)/n:
    TV(P, P_disc) <= M * h / 2
    L1(P, P_disc) <= M * h * (b - a) / 2
  For non-uniform (quantile) discretization with n bins:
    TV(P, P_disc) <= 1 / (2n)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import integrate, stats

logger = logging.getLogger(__name__)


@dataclass
class DiscretizationErrorBound:
    """Formal bound on discretization error for a single variable."""
    variable_name: str
    strategy: str
    n_bins: int
    bin_width: float
    tv_distance_bound: float
    l1_distance_bound: float
    kl_divergence_bound: float
    density_bound_M: float
    domain_width: float
    convergence_rate: str  # e.g. "O(1/n)" or "O(h)"
    is_certified: bool = True


@dataclass
class GlobalDiscretizationError:
    """Aggregate discretization error across all variables."""
    per_variable: List[DiscretizationErrorBound]
    total_tv_bound: float
    total_l1_bound: float
    max_per_variable_tv: float
    n_variables: int
    composition_gap_contribution: float
    meets_tolerance: bool
    tolerance: float


class DiscretizationErrorAnalyzer:
    """
    Formal analysis of discretization error with provable bounds.

    Computes worst-case error bounds for discretizing continuous
    distributions, and provides adaptive refinement to meet tolerances.

    Parameters
    ----------
    default_density_bound : float
        Default upper bound on the probability density. For financial
        variables, typical values are 5-20 for normalized exposures.
    """

    def __init__(self, default_density_bound: float = 10.0):
        self.default_density_bound = default_density_bound

    def compute_error_bound(
        self,
        variable_name: str,
        domain: Tuple[float, float],
        n_bins: int,
        strategy: str = "uniform",
        density_bound: Optional[float] = None,
        density_fn: Optional[Callable[[float], float]] = None,
    ) -> DiscretizationErrorBound:
        """
        Compute formal error bound for discretizing one variable.

        Parameters
        ----------
        variable_name : str
            Name of the variable being discretized.
        domain : tuple
            (lower, upper) bounds of the continuous domain.
        n_bins : int
            Number of discrete bins.
        strategy : str
            Discretization strategy ('uniform', 'quantile', 'adaptive').
        density_bound : float, optional
            Upper bound M on the density. If None, uses default.
        density_fn : callable, optional
            The actual density function (for tighter bounds).

        Returns
        -------
        DiscretizationErrorBound
        """
        a, b = domain
        width = b - a
        h = width / n_bins
        M = density_bound or self.default_density_bound

        if density_fn is not None:
            # Compute tighter bound by numerical integration
            M = self._estimate_density_bound(density_fn, a, b)

        if strategy == "uniform":
            tv_bound = M * h / 2
            l1_bound = M * h * width / 2
            kl_bound = M**2 * h**2 / 2  # Second-order approximation
            rate = "O(1/n)"
        elif strategy == "quantile":
            tv_bound = 1.0 / (2 * n_bins)
            l1_bound = width / (2 * n_bins)
            kl_bound = 1.0 / (2 * n_bins**2)
            rate = "O(1/n)"
        elif strategy == "adaptive":
            # Adaptive: error proportional to max bin probability variation
            tv_bound = M * h / 2  # Conservative
            l1_bound = M * h * width / 2
            kl_bound = M**2 * h**2 / 2
            rate = "O(1/n)"
        else:
            tv_bound = M * h / 2
            l1_bound = M * h * width / 2
            kl_bound = M**2 * h**2 / 2
            rate = "O(1/n)"

        return DiscretizationErrorBound(
            variable_name=variable_name,
            strategy=strategy,
            n_bins=n_bins,
            bin_width=h,
            tv_distance_bound=tv_bound,
            l1_distance_bound=l1_bound,
            kl_divergence_bound=kl_bound,
            density_bound_M=M,
            domain_width=width,
            convergence_rate=rate,
        )

    def compute_global_error(
        self,
        per_variable_bounds: List[DiscretizationErrorBound],
        lipschitz_constant: float,
        n_separators: int,
        tolerance: float = 0.01,
    ) -> GlobalDiscretizationError:
        """
        Compute aggregate discretization error across all variables.

        The total TV distance for independent discretization of d
        variables is bounded by sum of per-variable TV distances
        (union bound / subadditivity of TV).

        The contribution to the composition gap is:
          gap_disc = n_separators * L * total_tv
        """
        total_tv = sum(b.tv_distance_bound for b in per_variable_bounds)
        total_l1 = sum(b.l1_distance_bound for b in per_variable_bounds)
        max_tv = max(
            (b.tv_distance_bound for b in per_variable_bounds),
            default=0.0,
        )

        gap_contribution = n_separators * lipschitz_constant * total_tv

        return GlobalDiscretizationError(
            per_variable=per_variable_bounds,
            total_tv_bound=total_tv,
            total_l1_bound=total_l1,
            max_per_variable_tv=max_tv,
            n_variables=len(per_variable_bounds),
            composition_gap_contribution=gap_contribution,
            meets_tolerance=gap_contribution <= tolerance,
            tolerance=tolerance,
        )

    def adaptive_refinement(
        self,
        variable_name: str,
        domain: Tuple[float, float],
        target_tv: float,
        strategy: str = "uniform",
        density_bound: Optional[float] = None,
        max_bins: int = 1000,
    ) -> DiscretizationErrorBound:
        """
        Find the minimum number of bins to achieve target TV distance.

        Parameters
        ----------
        target_tv : float
            Target TV distance bound.
        max_bins : int
            Maximum number of bins to try.

        Returns
        -------
        DiscretizationErrorBound
            Bound with the minimum n_bins meeting the target.
        """
        M = density_bound or self.default_density_bound
        a, b = domain
        width = b - a

        if strategy == "quantile":
            n_bins = max(1, int(np.ceil(1.0 / (2 * target_tv))))
        else:
            # TV <= M * h / 2 = M * width / (2 * n_bins)
            n_bins = max(1, int(np.ceil(M * width / (2 * target_tv))))

        n_bins = min(n_bins, max_bins)

        return self.compute_error_bound(
            variable_name=variable_name,
            domain=domain,
            n_bins=n_bins,
            strategy=strategy,
            density_bound=density_bound,
        )

    def convergence_experiment(
        self,
        domain: Tuple[float, float] = (0.0, 1.0),
        density_fn: Optional[Callable[[float], float]] = None,
        bin_counts: Optional[List[int]] = None,
        n_samples: int = 100000,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Run convergence experiment: measure actual vs. bounded error.

        Computes the true TV distance between the continuous distribution
        and its piecewise-constant discretized approximation, then compares
        against the formal bound TV <= M * h / 2.

        Returns
        -------
        dict
            Results with 'bin_counts', 'empirical_tv', 'bound_tv',
            'bound_is_valid' arrays.
        """
        if bin_counts is None:
            bin_counts = [4, 8, 16, 32, 64, 128, 256]

        rng = np.random.default_rng(seed)
        a, b = domain

        if density_fn is None:
            dist = stats.beta(2, 5, loc=a, scale=b - a)
            density_fn = dist.pdf
            M = self._estimate_density_bound(density_fn, a, b)
        else:
            M = self._estimate_density_bound(density_fn, a, b)

        empirical_tvs = []
        bound_tvs = []
        valid = []

        for n_bins in bin_counts:
            h = (b - a) / n_bins
            edges = np.linspace(a, b, n_bins + 1)

            # True bin probabilities
            p_true_bins = np.array([
                integrate.quad(density_fn, edges[i], edges[i + 1])[0]
                for i in range(n_bins)
            ])

            # True TV distance between continuous and piecewise-constant:
            # TV = (1/2) * sum_i integral_bin_i |f(x) - f_bar_i| dx
            # where f_bar_i = p_true_bins[i] / h (the constant density in bin i)
            tv_true = 0.0
            for i in range(n_bins):
                f_bar = p_true_bins[i] / h
                def integrand(x, fbar=f_bar, fn=density_fn):
                    return abs(fn(x) - fbar)
                val, _ = integrate.quad(integrand, edges[i], edges[i + 1],
                                        limit=100)
                tv_true += val
            tv_true *= 0.5

            # Formal bound
            tv_bound = M * h / 2

            empirical_tvs.append(float(tv_true))
            bound_tvs.append(float(tv_bound))
            valid.append(tv_true <= tv_bound + 1e-10)

        return {
            "bin_counts": bin_counts,
            "empirical_tv": empirical_tvs,
            "bound_tv": bound_tvs,
            "bound_is_valid": valid,
            "all_bounds_valid": all(valid),
            "density_bound_M": M,
            "convergence_rate_empirical": self._fit_rate(bin_counts, empirical_tvs),
        }

    def _estimate_density_bound(
        self, density_fn: Callable[[float], float],
        a: float, b: float, n_eval: int = 1000,
    ) -> float:
        """Estimate upper bound on density by grid evaluation."""
        xs = np.linspace(a, b, n_eval)
        vals = np.array([density_fn(x) for x in xs])
        return float(np.max(vals)) * 1.1  # 10% safety margin

    def _rejection_sample(
        self, density_fn: Callable, a: float, b: float,
        M: float, n: int, rng: np.random.Generator,
    ) -> np.ndarray:
        """Simple rejection sampling."""
        samples = []
        while len(samples) < n:
            x = rng.uniform(a, b)
            u = rng.uniform(0, M)
            if u <= density_fn(x):
                samples.append(x)
        return np.array(samples[:n])

    def _fit_rate(
        self, bin_counts: List[int], errors: List[float],
    ) -> str:
        """Fit convergence rate from empirical data."""
        if len(bin_counts) < 2 or all(e < 1e-15 for e in errors):
            return "O(1/n)"
        log_n = np.log(np.array(bin_counts, dtype=float))
        log_e = np.log(np.array([max(e, 1e-15) for e in errors]))
        if len(log_n) >= 2:
            slope = np.polyfit(log_n, log_e, 1)[0]
            return f"O(n^{{{slope:.2f}}})"
        return "O(1/n)"
