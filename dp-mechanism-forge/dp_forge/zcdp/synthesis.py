"""
zCDP mechanism synthesis: design and optimize mechanisms under zCDP.

Provides tools for synthesizing optimal mechanisms with zCDP guarantees,
including Gaussian noise optimization, discrete/truncated Gaussian design,
multi-query optimization, and budget allocation.

References:
    - Bun & Steinke (2016): "Concentrated Differential Privacy"
    - Canonne et al. (2020): "Discrete Gaussian for DP"
    - Hardt & Talwar (2010): "On the Geometry of DP"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy import optimize, stats

from dp_forge.types import PrivacyBudget, ZCDPBudget, QuerySpec


# ---------------------------------------------------------------------------
# Gaussian Optimizer
# ---------------------------------------------------------------------------


class GaussianOptimizer:
    """Optimize Gaussian noise scale under zCDP constraints.

    For the Gaussian mechanism with L2 sensitivity Δ, the noise σ
    determines the zCDP cost ρ = Δ²/(2σ²). This class provides methods
    to find optimal σ for various objectives.
    """

    @staticmethod
    def sigma_for_rho(rho: float, sensitivity: float = 1.0) -> float:
        """Compute σ achieving exactly ρ-zCDP.

        σ = Δ/√(2ρ).

        Args:
            rho: Target zCDP cost.
            sensitivity: L2 sensitivity Δ.

        Returns:
            Noise standard deviation σ.
        """
        if rho <= 0:
            raise ValueError(f"rho must be > 0, got {rho}")
        if sensitivity <= 0:
            raise ValueError(f"sensitivity must be > 0, got {sensitivity}")
        return sensitivity / math.sqrt(2.0 * rho)

    @staticmethod
    def rho_for_sigma(sigma: float, sensitivity: float = 1.0) -> float:
        """Compute ρ for a given σ.

        ρ = Δ²/(2σ²).

        Args:
            sigma: Noise std dev.
            sensitivity: L2 sensitivity.

        Returns:
            zCDP cost ρ.
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        return sensitivity**2 / (2.0 * sigma**2)

    @staticmethod
    def sigma_for_dp(
        epsilon: float,
        delta: float,
        sensitivity: float = 1.0,
    ) -> float:
        """Compute σ for (ε,δ)-DP via zCDP.

        First finds ρ from (ε,δ) using ε = ρ + 2√(ρ·ln(1/δ)),
        then σ = Δ/√(2ρ).

        Args:
            epsilon: Target ε.
            delta: Target δ.
            sensitivity: L2 sensitivity.

        Returns:
            Noise std dev σ.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        if delta <= 0 or delta >= 1:
            raise ValueError(f"delta must be in (0, 1), got {delta}")

        # Invert ε = ρ + 2√(ρ·L): u² + 2u√L - ε = 0
        L = math.log(1.0 / delta)
        u = -math.sqrt(L) + math.sqrt(L + epsilon)
        rho = u * u
        return sensitivity / math.sqrt(2.0 * rho)

    @staticmethod
    def mse(sigma: float) -> float:
        """Mean squared error of Gaussian noise with std dev σ.

        MSE = σ².

        Args:
            sigma: Noise std dev.

        Returns:
            MSE = σ².
        """
        return sigma**2

    @staticmethod
    def mae(sigma: float) -> float:
        """Mean absolute error of Gaussian noise with std dev σ.

        MAE = σ√(2/π).

        Args:
            sigma: Noise std dev.

        Returns:
            MAE.
        """
        return sigma * math.sqrt(2.0 / math.pi)

    @staticmethod
    def confidence_interval(
        sigma: float, confidence: float = 0.95
    ) -> float:
        """Width of confidence interval for Gaussian noise.

        Args:
            sigma: Noise std dev.
            confidence: Confidence level (e.g., 0.95).

        Returns:
            Half-width of the confidence interval.
        """
        if not 0 < confidence < 1:
            raise ValueError(f"confidence must be in (0, 1), got {confidence}")
        z = stats.norm.ppf((1 + confidence) / 2)
        return z * sigma

    @staticmethod
    def optimize_for_utility(
        rho: float,
        sensitivity: float,
        utility_fn: Callable[[float], float],
    ) -> Tuple[float, float]:
        """Find σ that maximizes utility subject to ρ-zCDP.

        The only feasible σ is σ = Δ/√(2ρ), so this returns
        the utility at the unique feasible point.

        Args:
            rho: zCDP budget.
            sensitivity: L2 sensitivity.
            utility_fn: Function mapping σ to utility (higher is better).

        Returns:
            Tuple of (optimal_sigma, utility_value).
        """
        sigma = sensitivity / math.sqrt(2.0 * rho)
        return sigma, utility_fn(sigma)


# ---------------------------------------------------------------------------
# Discrete Gaussian Synthesis
# ---------------------------------------------------------------------------


class DiscreteGaussianSynthesis:
    """Discrete Gaussian mechanism design for zCDP.

    The discrete Gaussian N_Z(0, σ²) adds integer noise sampled from
    the discrete Gaussian distribution. It satisfies ρ-zCDP with
    ρ ≤ Δ²/(2σ²) for integer sensitivity Δ.

    Reference: Canonne, Kamath, Steinke (2020).
    """

    def __init__(self, sigma_sq: float, sensitivity: int = 1) -> None:
        """
        Args:
            sigma_sq: Variance parameter σ² > 0.
            sensitivity: Integer L2 sensitivity.
        """
        if sigma_sq <= 0:
            raise ValueError(f"sigma_sq must be > 0, got {sigma_sq}")
        if sensitivity < 1:
            raise ValueError(f"sensitivity must be >= 1, got {sensitivity}")
        self.sigma_sq = sigma_sq
        self.sensitivity = sensitivity

    @property
    def name(self) -> str:
        return "discrete_gaussian"

    def zcdp_cost(self) -> float:
        """Compute ρ-zCDP cost.

        For σ² ≥ 1, the discrete Gaussian closely matches the continuous
        Gaussian: ρ ≈ Δ²/(2σ²).
        For small σ², uses a tighter analysis.

        Returns:
            zCDP cost ρ.
        """
        rho_continuous = self.sensitivity**2 / (2.0 * self.sigma_sq)
        if self.sigma_sq >= 1.0:
            return rho_continuous
        # For small σ², the discrete Gaussian has slightly higher cost
        # Use the log-ratio bound
        return rho_continuous * (1.0 + 1.0 / (12.0 * self.sigma_sq))

    @staticmethod
    def from_rho(rho: float, sensitivity: int = 1) -> "DiscreteGaussianSynthesis":
        """Create discrete Gaussian achieving target ρ-zCDP.

        Args:
            rho: Target zCDP cost.
            sensitivity: Integer sensitivity.

        Returns:
            DiscreteGaussianSynthesis.
        """
        if rho <= 0:
            raise ValueError(f"rho must be > 0, got {rho}")
        sigma_sq = sensitivity**2 / (2.0 * rho)
        return DiscreteGaussianSynthesis(sigma_sq=sigma_sq, sensitivity=sensitivity)

    def sample(
        self, size: int = 1, rng: Optional[np.random.Generator] = None
    ) -> npt.NDArray[np.int64]:
        """Sample from the discrete Gaussian distribution.

        Uses rejection sampling from the continuous Gaussian.

        Args:
            size: Number of samples.
            rng: Random number generator.

        Returns:
            Array of integer noise samples.
        """
        if rng is None:
            rng = np.random.default_rng()
        sigma = math.sqrt(self.sigma_sq)

        # For large σ, rounding the continuous Gaussian is efficient
        if sigma >= 4.0:
            continuous = rng.normal(0, sigma, size=size)
            return np.round(continuous).astype(np.int64)

        # For small σ, use exact sampling via enumeration
        support = int(max(10 * sigma, 10))
        values = np.arange(-support, support + 1)
        unnorm = np.exp(-values**2 / (2.0 * self.sigma_sq))
        probs = unnorm / unnorm.sum()
        return rng.choice(values, size=size, p=probs).astype(np.int64)

    def pmf(self, k: int) -> float:
        """Probability mass function P(X = k).

        Args:
            k: Integer value.

        Returns:
            P(X = k) ∝ exp(-k²/(2σ²)).
        """
        # Compute using log for numerical stability
        log_p = -k**2 / (2.0 * self.sigma_sq)
        # Normalization via theta function approximation
        support = int(max(10 * math.sqrt(self.sigma_sq), 10))
        values = np.arange(-support, support + 1)
        log_norm = float(
            np.logaddexp.reduce(-values**2 / (2.0 * self.sigma_sq))
        )
        return math.exp(log_p - log_norm)

    def to_approx_dp(self, delta: float) -> PrivacyBudget:
        """Convert to (ε,δ)-DP."""
        rho = self.zcdp_cost()
        budget = ZCDPBudget(rho=rho)
        return budget.to_approx_dp(delta)

    def __repr__(self) -> str:
        return (
            f"DiscreteGaussianSynthesis(σ²={self.sigma_sq:.4f}, "
            f"Δ={self.sensitivity}, ρ={self.zcdp_cost():.6f})"
        )


# ---------------------------------------------------------------------------
# Truncated Gaussian Synthesis
# ---------------------------------------------------------------------------


class TruncatedGaussianSynthesis:
    """Truncated Gaussian mechanism with zCDP guarantee.

    Adds Gaussian noise truncated to [-B, B] for some bound B.
    The truncation slightly increases the privacy cost but reduces
    the output range.
    """

    def __init__(
        self,
        sigma: float,
        bound: float,
        sensitivity: float = 1.0,
    ) -> None:
        """
        Args:
            sigma: Gaussian std dev before truncation.
            bound: Truncation bound B > 0.
            sensitivity: L2 sensitivity.
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if bound <= 0:
            raise ValueError(f"bound must be > 0, got {bound}")
        if sensitivity <= 0:
            raise ValueError(f"sensitivity must be > 0, got {sensitivity}")
        self.sigma = sigma
        self.bound = bound
        self.sensitivity = sensitivity

    @property
    def name(self) -> str:
        return "truncated_gaussian"

    def zcdp_cost(self) -> float:
        """Compute ρ-zCDP cost with truncation penalty.

        The truncated Gaussian has slightly higher ρ than the untruncated
        version due to the renormalization. For B ≥ 3σ, the penalty
        is negligible.

        Returns:
            zCDP cost ρ.
        """
        rho_base = self.sensitivity**2 / (2.0 * self.sigma**2)
        # Truncation penalty from the ratio of normalizing constants
        # P(|Z| ≤ B) for Z ~ N(0, σ²) vs Z ~ N(Δ, σ²)
        z_ratio = self.bound / self.sigma
        if z_ratio >= 6.0:
            # Negligible truncation effect
            return rho_base

        p0 = 2.0 * stats.norm.cdf(z_ratio) - 1.0  # P(|Z| ≤ B | Z ~ N(0,σ²))
        z_shifted = (self.bound - self.sensitivity) / self.sigma
        z_shifted_neg = (-self.bound - self.sensitivity) / self.sigma
        p1 = stats.norm.cdf(z_shifted) - stats.norm.cdf(z_shifted_neg)

        if p0 <= 0 or p1 <= 0:
            return rho_base

        # Additional cost from truncation: log(p0/p1)
        trunc_penalty = max(0.0, math.log(p0 / p1))
        return rho_base + trunc_penalty

    @staticmethod
    def from_rho(
        rho: float,
        sensitivity: float = 1.0,
        bound_factor: float = 6.0,
    ) -> "TruncatedGaussianSynthesis":
        """Create truncated Gaussian achieving approximately ρ-zCDP.

        Args:
            rho: Target zCDP cost.
            sensitivity: L2 sensitivity.
            bound_factor: Truncation at B = bound_factor * σ.

        Returns:
            TruncatedGaussianSynthesis.
        """
        if rho <= 0:
            raise ValueError(f"rho must be > 0, got {rho}")
        sigma = sensitivity / math.sqrt(2.0 * rho)
        bound = bound_factor * sigma
        return TruncatedGaussianSynthesis(
            sigma=sigma, bound=bound, sensitivity=sensitivity
        )

    def sample(
        self, size: int = 1, rng: Optional[np.random.Generator] = None
    ) -> npt.NDArray[np.float64]:
        """Sample from the truncated Gaussian.

        Uses scipy.stats.truncnorm.

        Args:
            size: Number of samples.
            rng: Random number generator.

        Returns:
            Array of noise samples in [-B, B].
        """
        a, b = -self.bound / self.sigma, self.bound / self.sigma
        if rng is None:
            rng = np.random.default_rng()
        samples = stats.truncnorm.rvs(
            a, b, loc=0, scale=self.sigma, size=size, random_state=rng
        )
        return np.asarray(samples, dtype=np.float64)

    def max_error(self) -> float:
        """Maximum possible noise magnitude (= B)."""
        return self.bound

    def expected_error(self) -> float:
        """Expected absolute error E[|Z|] for truncated Gaussian."""
        a, b = -self.bound / self.sigma, self.bound / self.sigma
        # E[|Z|] for truncated normal
        dist = stats.truncnorm(a, b, loc=0, scale=self.sigma)
        # Numerical integration
        grid = np.linspace(-self.bound, self.bound, 10000)
        pdf = dist.pdf(grid)
        dx = grid[1] - grid[0]
        return float(np.sum(np.abs(grid) * pdf) * dx)

    def to_approx_dp(self, delta: float) -> PrivacyBudget:
        """Convert to (ε,δ)-DP."""
        rho = self.zcdp_cost()
        budget = ZCDPBudget(rho=rho)
        return budget.to_approx_dp(delta)

    def __repr__(self) -> str:
        return (
            f"TruncatedGaussianSynthesis(σ={self.sigma:.4f}, "
            f"B={self.bound:.4f}, ρ={self.zcdp_cost():.6f})"
        )


# ---------------------------------------------------------------------------
# Multi-Query Synthesis
# ---------------------------------------------------------------------------


class MultiQuerySynthesis:
    """Optimize mechanisms across multiple queries simultaneously.

    Given k queries with sensitivities Δ_1, ..., Δ_k and a total zCDP
    budget ρ_total, find the optimal noise allocation.
    """

    def __init__(
        self,
        sensitivities: npt.NDArray[np.float64],
        total_rho: float,
    ) -> None:
        """
        Args:
            sensitivities: L2 sensitivities for each query.
            total_rho: Total zCDP budget.
        """
        self.sensitivities = np.asarray(sensitivities, dtype=np.float64)
        if np.any(self.sensitivities <= 0):
            raise ValueError("All sensitivities must be > 0")
        if total_rho <= 0:
            raise ValueError(f"total_rho must be > 0, got {total_rho}")
        self.total_rho = total_rho
        self.k = len(self.sensitivities)

    def equal_allocation(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Equal budget allocation across queries.

        Returns:
            Tuple of (rho_per_query, sigma_per_query).
        """
        rhos = np.full(self.k, self.total_rho / self.k)
        sigmas = self.sensitivities / np.sqrt(2.0 * rhos)
        return rhos, sigmas

    def minimize_total_mse(
        self,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Allocate budget to minimize total MSE.

        MSE_i = σ_i² = Δ_i²/(2ρ_i). Total MSE = Σ Δ_i²/(2ρ_i).

        Optimal allocation by Lagrange multipliers:
            ρ_i ∝ Δ_i, giving ρ_i = Δ_i / Σ_j Δ_j · ρ_total.

        Returns:
            Tuple of (rho_per_query, sigma_per_query).
        """
        rhos = self.total_rho * self.sensitivities / self.sensitivities.sum()
        sigmas = self.sensitivities / np.sqrt(2.0 * rhos)
        return rhos, sigmas

    def minimize_max_mse(
        self,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Allocate budget to minimize the maximum MSE across queries.

        Equivalent to equalizing MSE: Δ_i²/(2ρ_i) = c for all i.
        This gives ρ_i ∝ Δ_i².

        Returns:
            Tuple of (rho_per_query, sigma_per_query).
        """
        sq = self.sensitivities**2
        rhos = self.total_rho * sq / sq.sum()
        sigmas = self.sensitivities / np.sqrt(2.0 * rhos)
        return rhos, sigmas

    def minimize_weighted_mse(
        self,
        weights: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Allocate budget to minimize weighted total MSE.

        Minimize Σ w_i · Δ_i²/(2ρ_i) subject to Σρ_i = ρ_total.

        Optimal: ρ_i ∝ Δ_i · √w_i.

        Args:
            weights: Importance weights w_i > 0.

        Returns:
            Tuple of (rho_per_query, sigma_per_query).
        """
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != self.sensitivities.shape:
            raise ValueError("weights must match sensitivities shape")
        if np.any(w <= 0):
            raise ValueError("All weights must be > 0")

        alloc = self.sensitivities * np.sqrt(w)
        rhos = self.total_rho * alloc / alloc.sum()
        sigmas = self.sensitivities / np.sqrt(2.0 * rhos)
        return rhos, sigmas

    def custom_objective(
        self,
        objective: Callable[[npt.NDArray[np.float64]], float],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Optimize budget for a custom objective.

        Args:
            objective: Function mapping σ values to objective (to minimize).

        Returns:
            Tuple of (rho_per_query, sigma_per_query).
        """
        def obj_in_rho(rhos: npt.NDArray[np.float64]) -> float:
            sigmas = self.sensitivities / np.sqrt(2.0 * np.maximum(rhos, 1e-15))
            return objective(sigmas)

        x0 = np.full(self.k, self.total_rho / self.k)
        bounds = [(1e-12, self.total_rho)] * self.k
        constraint = {"type": "eq", "fun": lambda x: np.sum(x) - self.total_rho}

        result = optimize.minimize(
            obj_in_rho, x0, method="SLSQP",
            bounds=bounds, constraints=[constraint],
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        rhos = np.maximum(result.x, 1e-15) if result.success else x0
        rhos = rhos * self.total_rho / rhos.sum()
        sigmas = self.sensitivities / np.sqrt(2.0 * rhos)
        return rhos, sigmas

    def summary(
        self,
        rhos: npt.NDArray[np.float64],
        sigmas: npt.NDArray[np.float64],
    ) -> Dict[str, Any]:
        """Summarize an allocation.

        Args:
            rhos: Per-query ρ allocations.
            sigmas: Per-query σ values.

        Returns:
            Summary dict.
        """
        mses = sigmas**2
        return {
            "num_queries": self.k,
            "total_rho": float(rhos.sum()),
            "rhos": rhos.tolist(),
            "sigmas": sigmas.tolist(),
            "mses": mses.tolist(),
            "total_mse": float(mses.sum()),
            "max_mse": float(mses.max()),
        }

    def __repr__(self) -> str:
        return (
            f"MultiQuerySynthesis(k={self.k}, ρ_total={self.total_rho:.6f})"
        )


# ---------------------------------------------------------------------------
# Budget Allocation
# ---------------------------------------------------------------------------


class BudgetAllocation:
    """Optimal budget splitting across queries under zCDP.

    Provides different allocation strategies and their theoretical
    guarantees.
    """

    @staticmethod
    def equal(total_rho: float, k: int) -> npt.NDArray[np.float64]:
        """Equal allocation: ρ_i = ρ_total / k.

        Args:
            total_rho: Total budget.
            k: Number of queries.

        Returns:
            Array of k equal allocations.
        """
        if total_rho <= 0:
            raise ValueError(f"total_rho must be > 0, got {total_rho}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        return np.full(k, total_rho / k)

    @staticmethod
    def proportional(
        total_rho: float,
        weights: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Proportional allocation: ρ_i ∝ w_i.

        Args:
            total_rho: Total budget.
            weights: Non-negative weights.

        Returns:
            Proportional allocations.
        """
        if total_rho <= 0:
            raise ValueError(f"total_rho must be > 0, got {total_rho}")
        w = np.asarray(weights, dtype=np.float64)
        if np.any(w < 0):
            raise ValueError("All weights must be non-negative")
        w_sum = w.sum()
        if w_sum <= 0:
            raise ValueError("Sum of weights must be positive")
        return total_rho * w / w_sum

    @staticmethod
    def geometric(
        total_rho: float,
        k: int,
        ratio: float = 0.5,
    ) -> npt.NDArray[np.float64]:
        """Geometric allocation: ρ_i = ρ_1 · r^(i-1).

        Later queries get geometrically less budget. Useful when
        early queries are more important.

        Args:
            total_rho: Total budget.
            k: Number of queries.
            ratio: Geometric ratio r ∈ (0, 1).

        Returns:
            Geometric allocations.
        """
        if total_rho <= 0:
            raise ValueError(f"total_rho must be > 0, got {total_rho}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not 0 < ratio < 1:
            raise ValueError(f"ratio must be in (0, 1), got {ratio}")

        powers = np.array([ratio**i for i in range(k)])
        return total_rho * powers / powers.sum()

    @staticmethod
    def minimize_mse_gaussian(
        total_rho: float,
        sensitivities: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Optimal allocation minimizing total MSE for Gaussian mechanisms.

        ρ_i ∝ Δ_i (by Cauchy-Schwarz / Lagrange multipliers).

        Args:
            total_rho: Total budget.
            sensitivities: Per-query L2 sensitivities.

        Returns:
            Optimal allocations.
        """
        sens = np.asarray(sensitivities, dtype=np.float64)
        if np.any(sens <= 0):
            raise ValueError("All sensitivities must be > 0")
        return total_rho * sens / sens.sum()

    @staticmethod
    def svt_allocation(
        total_rho: float,
        num_threshold: int,
        num_above: int,
    ) -> Tuple[float, float]:
        """Budget allocation for sparse vector technique.

        Splits budget between threshold noise and above-threshold queries.

        Args:
            total_rho: Total zCDP budget.
            num_threshold: Number of threshold comparisons.
            num_above: Expected number of above-threshold answers.

        Returns:
            Tuple of (rho_threshold, rho_per_above).
        """
        if total_rho <= 0:
            raise ValueError(f"total_rho must be > 0, got {total_rho}")
        if num_threshold < 1:
            raise ValueError(
                f"num_threshold must be >= 1, got {num_threshold}"
            )
        if num_above < 1:
            raise ValueError(f"num_above must be >= 1, got {num_above}")

        # Optimal split: half for threshold, half for above-threshold
        rho_threshold = total_rho / 2.0
        rho_per_above = total_rho / (2.0 * num_above)
        return rho_threshold, rho_per_above

    @staticmethod
    def validate_allocation(
        allocation: npt.NDArray[np.float64],
        total_rho: float,
        tol: float = 1e-10,
    ) -> bool:
        """Validate that an allocation sums to the total budget.

        Args:
            allocation: Per-query ρ values.
            total_rho: Expected total.
            tol: Tolerance.

        Returns:
            True if valid.
        """
        alloc = np.asarray(allocation, dtype=np.float64)
        if np.any(alloc <= 0):
            return False
        return abs(alloc.sum() - total_rho) <= tol


# ---------------------------------------------------------------------------
# ZCDPSynthesizer (top-level)
# ---------------------------------------------------------------------------


class ZCDPSynthesizer:
    """Synthesize optimal mechanism under zCDP constraints.

    Given a query specification and a zCDP budget ρ, synthesize a mechanism
    (probability table) that maximizes utility while satisfying ρ-zCDP.
    """

    def __init__(self, verbose: int = 0) -> None:
        self.verbose = verbose

    def synthesize_gaussian(
        self,
        query_values: npt.NDArray[np.float64],
        rho: float,
        sensitivity: float = 1.0,
        k: int = 100,
    ) -> Dict[str, Any]:
        """Synthesize a Gaussian mechanism table satisfying ρ-zCDP.

        Discretizes the Gaussian noise distribution into a k-bin probability
        table suitable for downstream use.

        Args:
            query_values: Array of n query output values.
            rho: Target zCDP cost ρ.
            sensitivity: L2 sensitivity.
            k: Number of output bins.

        Returns:
            Dict with 'mechanism' (n×k array), 'rho', 'sigma', 'edges'.
        """
        if rho <= 0:
            raise ValueError(f"rho must be > 0, got {rho}")
        query_values = np.asarray(query_values, dtype=np.float64)
        n = len(query_values)

        sigma = sensitivity / math.sqrt(2.0 * rho)

        # Build output grid
        q_min = query_values.min() - 4 * sigma
        q_max = query_values.max() + 4 * sigma
        edges = np.linspace(q_min, q_max, k + 1)
        centers = (edges[:-1] + edges[1:]) / 2.0

        # Build n×k mechanism table
        mechanism = np.zeros((n, k), dtype=np.float64)
        for i in range(n):
            probs = stats.norm.cdf(edges[1:], loc=query_values[i], scale=sigma) - \
                    stats.norm.cdf(edges[:-1], loc=query_values[i], scale=sigma)
            probs = np.maximum(probs, 0.0)
            total = probs.sum()
            if total > 0:
                probs /= total
            mechanism[i] = probs

        return {
            "mechanism": mechanism,
            "rho": rho,
            "sigma": sigma,
            "edges": edges,
            "centers": centers,
            "mse": sigma**2,
        }

    def synthesize_discrete_gaussian(
        self,
        query_values: npt.NDArray[np.int64],
        rho: float,
        sensitivity: int = 1,
    ) -> Dict[str, Any]:
        """Synthesize a discrete Gaussian mechanism.

        Args:
            query_values: Integer query output values.
            rho: Target zCDP cost.
            sensitivity: Integer L2 sensitivity.

        Returns:
            Dict with 'mechanism', 'rho', 'sigma_sq', 'support'.
        """
        if rho <= 0:
            raise ValueError(f"rho must be > 0, got {rho}")
        query_values = np.asarray(query_values, dtype=np.int64)
        n = len(query_values)

        dg = DiscreteGaussianSynthesis.from_rho(rho, sensitivity)
        sigma_sq = dg.sigma_sq

        # Determine support
        support_half = int(max(6 * math.sqrt(sigma_sq), 10))
        support = np.arange(-support_half, support_half + 1)
        k = len(support)

        # Build mechanism table
        mechanism = np.zeros((n, k), dtype=np.float64)
        for i in range(n):
            outputs = support + int(query_values[i])
            log_probs = -(support.astype(np.float64))**2 / (2.0 * sigma_sq)
            probs = np.exp(log_probs - np.max(log_probs))
            probs /= probs.sum()
            mechanism[i] = probs

        return {
            "mechanism": mechanism,
            "rho": dg.zcdp_cost(),
            "sigma_sq": sigma_sq,
            "support": support,
        }

    def synthesize_for_composition(
        self,
        query_values_list: List[npt.NDArray[np.float64]],
        sensitivities: npt.NDArray[np.float64],
        total_rho: float,
        allocation: str = "mse_optimal",
    ) -> List[Dict[str, Any]]:
        """Synthesize multiple mechanisms sharing a total budget.

        Args:
            query_values_list: List of query value arrays.
            sensitivities: Per-query sensitivities.
            total_rho: Total zCDP budget.
            allocation: Allocation strategy ('equal', 'mse_optimal', 'minimax').

        Returns:
            List of mechanism dicts.
        """
        sens = np.asarray(sensitivities, dtype=np.float64)
        k_queries = len(query_values_list)
        if len(sens) != k_queries:
            raise ValueError("sensitivities must match number of queries")

        mq = MultiQuerySynthesis(sens, total_rho)
        if allocation == "equal":
            rhos, _ = mq.equal_allocation()
        elif allocation == "mse_optimal":
            rhos, _ = mq.minimize_total_mse()
        elif allocation == "minimax":
            rhos, _ = mq.minimize_max_mse()
        else:
            raise ValueError(f"Unknown allocation: {allocation}")

        results = []
        for i in range(k_queries):
            result = self.synthesize_gaussian(
                query_values_list[i],
                rho=float(rhos[i]),
                sensitivity=float(sens[i]),
            )
            results.append(result)
        return results

    def __repr__(self) -> str:
        return "ZCDPSynthesizer()"
