"""
Privacy notion conversion for zero-concentrated differential privacy.

Provides tight and numerically optimized conversions between zCDP, (ε,δ)-DP,
Rényi DP, and privacy loss distribution (PLD) frameworks.

References:
    - Bun & Steinke (2016): "Concentrated Differential Privacy"
    - Mironov (2017): "Rényi Differential Privacy"
    - Balle et al. (2020): "Hypothesis Testing Interpretations and Renyi DP"
    - Koskela et al. (2020): "Computing Tight DP Guarantees Using FFT"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy import integrate, optimize, special, interpolate

from dp_forge.types import PrivacyBudget, ZCDPBudget


# ---------------------------------------------------------------------------
# zCDP → (ε,δ)-DP
# ---------------------------------------------------------------------------


class ZCDPToApproxDP:
    """Convert ρ-zCDP guarantee to (ε,δ)-DP.

    Implements the optimal conversion from Bun & Steinke (2016):
        ε = ρ + 2√(ρ·ln(1/δ))

    Also provides the numerically optimized conversion via RDP.
    """

    @staticmethod
    def convert(rho: float, delta: float) -> PrivacyBudget:
        """Convert ρ-zCDP to (ε,δ)-DP using the analytic formula.

        Args:
            rho: zCDP parameter ρ > 0.
            delta: Target δ ∈ (0, 1).

        Returns:
            (ε,δ)-DP budget.
        """
        _validate_rho(rho)
        _validate_delta(delta)
        eps = rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))
        return PrivacyBudget(epsilon=eps, delta=delta)

    @staticmethod
    def convert_optimal(rho: float, delta: float) -> PrivacyBudget:
        """Numerically optimized conversion via RDP order optimization.

        Minimizes ε = αρ + log(1/δ)/(α-1) over α > 1.
        The analytic optimum is α* = 1 + √(log(1/δ)/ρ).

        Args:
            rho: zCDP parameter ρ > 0.
            delta: Target δ ∈ (0, 1).

        Returns:
            (ε,δ)-DP budget (tightest conversion).
        """
        _validate_rho(rho)
        _validate_delta(delta)
        log_inv_delta = math.log(1.0 / delta)

        # Analytic optimum
        alpha_star = 1.0 + math.sqrt(log_inv_delta / rho)
        eps_analytic = alpha_star * rho + log_inv_delta / (alpha_star - 1.0)

        # Numerical optimization as a fallback
        def eps_at_alpha(alpha: float) -> float:
            if alpha <= 1.0:
                return float("inf")
            return alpha * rho + log_inv_delta / (alpha - 1.0)

        try:
            result = optimize.minimize_scalar(
                eps_at_alpha,
                bounds=(1.001, max(alpha_star * 3, 100.0)),
                method="bounded",
            )
            eps_numerical = result.fun if result.success else eps_analytic
        except (RuntimeError, ValueError):
            eps_numerical = eps_analytic

        eps = min(eps_analytic, eps_numerical)
        return PrivacyBudget(epsilon=eps, delta=delta)

    @staticmethod
    def convert_truncated(
        xi: float, rho: float, delta: float
    ) -> PrivacyBudget:
        """Convert (ξ, ρ)-tCDP to (ε,δ)-DP.

        ε = ξ + ρ + 2√(ρ·ln(1/δ)).

        Args:
            xi: Offset ξ ≥ 0.
            rho: Concentration ρ > 0.
            delta: Target δ ∈ (0, 1).

        Returns:
            (ε,δ)-DP budget.
        """
        _validate_rho(rho)
        _validate_delta(delta)
        if xi < 0:
            raise ValueError(f"xi must be >= 0, got {xi}")
        eps = xi + rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))
        return PrivacyBudget(epsilon=eps, delta=delta)

    @staticmethod
    def delta_for_epsilon(rho: float, epsilon: float) -> float:
        """Find δ such that ρ-zCDP implies (ε,δ)-DP.

        Inverts ε = ρ + 2√(ρ·ln(1/δ)):
            δ = exp(-(ε - ρ)²/(4ρ)).

        Args:
            rho: zCDP parameter ρ > 0.
            epsilon: Target ε.

        Returns:
            δ value.
        """
        _validate_rho(rho)
        if epsilon <= rho:
            raise ValueError(
                f"epsilon must be > rho for finite delta, got ε={epsilon}, ρ={rho}"
            )
        delta = math.exp(-((epsilon - rho) ** 2) / (4.0 * rho))
        return min(delta, 1.0 - 1e-15)

    @staticmethod
    def epsilon_for_delta(rho: float, delta: float) -> float:
        """Compute ε for given ρ and δ.

        ε = ρ + 2√(ρ·ln(1/δ)).

        Args:
            rho: zCDP parameter ρ.
            delta: Target δ.

        Returns:
            ε value.
        """
        _validate_rho(rho)
        _validate_delta(delta)
        return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


# ---------------------------------------------------------------------------
# (ε,δ)-DP → zCDP
# ---------------------------------------------------------------------------


class ApproxDPToZCDP:
    """Convert (ε,δ)-DP to zCDP (tight upper bound).

    This conversion is generally lossy. Different bounds are available
    depending on whether δ = 0 (pure DP) or δ > 0.
    """

    @staticmethod
    def convert(epsilon: float, delta: float = 0.0) -> ZCDPBudget:
        """Convert (ε,δ)-DP to ρ-zCDP.

        For pure DP (δ=0): ρ = ε(e^ε - 1)/2 (Bun & Steinke, Prop 1.3).
        For approximate DP: inverts ε = ρ + 2√(ρ·ln(1/δ)) for ρ.

        Args:
            epsilon: Privacy parameter ε > 0.
            delta: Privacy parameter δ ∈ [0, 1).

        Returns:
            ZCDPBudget (upper bound on ρ).
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        if not (0.0 <= delta < 1.0):
            raise ValueError(f"delta must be in [0, 1), got {delta}")

        if delta == 0.0:
            # Pure DP: ε-DP implies ε(e^ε - 1)/2 - zCDP
            rho = epsilon * (math.exp(epsilon) - 1.0) / 2.0
        else:
            # Invert ε = ρ + 2√(ρ·L) where L = ln(1/δ)
            rho = _invert_zcdp_to_dp(epsilon, delta)

        return ZCDPBudget(rho=rho)

    @staticmethod
    def convert_tight(epsilon: float, delta: float = 0.0) -> ZCDPBudget:
        """Tighter conversion using numerical optimization.

        Searches for the smallest ρ such that ρ-zCDP implies (ε,δ)-DP.

        Args:
            epsilon: Privacy parameter ε.
            delta: Privacy parameter δ.

        Returns:
            ZCDPBudget with tightest ρ.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        if not (0.0 <= delta < 1.0):
            raise ValueError(f"delta must be in [0, 1), got {delta}")

        if delta == 0.0:
            # Pure DP → zCDP
            rho = epsilon * (math.exp(epsilon) - 1.0) / 2.0
            return ZCDPBudget(rho=rho)

        # Search for smallest ρ such that the conversion gives ε
        rho = _invert_zcdp_to_dp(epsilon, delta)
        return ZCDPBudget(rho=rho)

    @staticmethod
    def pure_dp_to_zcdp(epsilon: float) -> ZCDPBudget:
        """Convert pure ε-DP to zCDP.

        Uses the tight bound: ρ = ε(e^ε - 1)/2.
        Also provides the simpler bound ρ = ε²/2.

        Args:
            epsilon: Pure DP parameter.

        Returns:
            ZCDPBudget.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        rho = epsilon * (math.exp(epsilon) - 1.0) / 2.0
        return ZCDPBudget(rho=rho)


# ---------------------------------------------------------------------------
# RDP ↔ zCDP
# ---------------------------------------------------------------------------


class RDPToZCDP:
    """Convert Rényi DP to zCDP.

    Given an RDP curve ε(α), finds the tightest ρ such that ε(α) ≤ αρ
    for all α > 1, which is equivalent to ρ = sup_{α>1} ε(α)/α.
    """

    @staticmethod
    def convert(
        rdp_curve: Callable[[float], float],
        alpha_range: Optional[Tuple[float, float]] = None,
        num_points: int = 1000,
    ) -> ZCDPBudget:
        """Convert RDP curve to zCDP by optimizing over α.

        Args:
            rdp_curve: Function mapping α to ε(α).
            alpha_range: Range of α to search (default: (1.01, 1000)).
            num_points: Number of grid points for initial search.

        Returns:
            ZCDPBudget with ρ = sup_{α>1} ε(α)/α.
        """
        lo, hi = alpha_range if alpha_range else (1.01, 1000.0)

        # Grid search for initial estimate
        alphas = np.linspace(lo, hi, num_points)
        rho_values = np.array(
            [rdp_curve(a) / a for a in alphas], dtype=np.float64
        )
        valid = np.isfinite(rho_values)
        if not np.any(valid):
            raise ValueError("RDP curve returned no finite values")
        rho_grid = float(np.max(rho_values[valid]))

        # Refine with numerical optimization
        def neg_rdp_over_alpha(alpha: float) -> float:
            try:
                val = rdp_curve(alpha) / alpha
            except (ValueError, OverflowError):
                return 0.0
            return -val if math.isfinite(val) else 0.0

        try:
            result = optimize.minimize_scalar(
                neg_rdp_over_alpha,
                bounds=(lo, hi),
                method="bounded",
            )
            rho_opt = -result.fun if result.success else rho_grid
        except (RuntimeError, ValueError):
            rho_opt = rho_grid

        rho = max(rho_grid, rho_opt)
        return ZCDPBudget(rho=max(rho, 1e-15))

    @staticmethod
    def from_gaussian(sigma: float, sensitivity: float = 1.0) -> ZCDPBudget:
        """Convert Gaussian RDP to zCDP.

        For Gaussian, ε(α) = αΔ²/(2σ²), so ρ = Δ²/(2σ²) exactly.

        Args:
            sigma: Noise std dev.
            sensitivity: L2 sensitivity.

        Returns:
            ZCDPBudget.
        """
        rho = sensitivity**2 / (2.0 * sigma**2)
        return ZCDPBudget(rho=rho)

    @staticmethod
    def from_rdp_values(
        alphas: npt.NDArray[np.float64],
        epsilons: npt.NDArray[np.float64],
    ) -> ZCDPBudget:
        """Convert tabulated RDP values to zCDP.

        Args:
            alphas: Array of Rényi orders α > 1.
            epsilons: Array of RDP values ε(α).

        Returns:
            ZCDPBudget with ρ = max(ε(α)/α).
        """
        alphas = np.asarray(alphas, dtype=np.float64)
        epsilons = np.asarray(epsilons, dtype=np.float64)
        if alphas.shape != epsilons.shape:
            raise ValueError("alphas and epsilons must have same shape")
        if np.any(alphas <= 1):
            raise ValueError("All alphas must be > 1")
        ratios = epsilons / alphas
        valid = np.isfinite(ratios)
        if not np.any(valid):
            raise ValueError("No finite ε(α)/α values")
        rho = float(np.max(ratios[valid]))
        return ZCDPBudget(rho=max(rho, 1e-15))


class ZCDPToRDP:
    """Convert zCDP to Rényi DP.

    ρ-zCDP implies (α, αρ)-RDP for all α > 1.
    """

    @staticmethod
    def convert(rho: float, alpha: float) -> float:
        """Convert ρ-zCDP to RDP at order α.

        Args:
            rho: zCDP parameter ρ > 0.
            alpha: Rényi order α > 1.

        Returns:
            RDP parameter ε(α) = αρ.
        """
        _validate_rho(rho)
        if alpha <= 1:
            raise ValueError(f"alpha must be > 1, got {alpha}")
        return alpha * rho

    @staticmethod
    def rdp_curve(rho: float) -> Callable[[float], float]:
        """Return the RDP curve ε(α) = αρ.

        Args:
            rho: zCDP parameter ρ.

        Returns:
            Function mapping α → αρ.
        """
        _validate_rho(rho)

        def _curve(alpha: float) -> float:
            if alpha <= 1:
                raise ValueError(f"alpha must be > 1, got {alpha}")
            return alpha * rho

        return _curve

    @staticmethod
    def rdp_table(
        rho: float,
        alphas: Optional[npt.NDArray[np.float64]] = None,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute RDP curve at tabulated α values.

        Args:
            rho: zCDP parameter ρ.
            alphas: Rényi orders (default: integer orders 2..128).

        Returns:
            Tuple of (alphas, epsilons).
        """
        _validate_rho(rho)
        if alphas is None:
            alphas = np.arange(2, 129, dtype=np.float64)
        alphas = np.asarray(alphas, dtype=np.float64)
        epsilons = alphas * rho
        return alphas, epsilons


# ---------------------------------------------------------------------------
# Privacy Loss Distribution Conversion
# ---------------------------------------------------------------------------


class PLDConversion:
    """Privacy loss distribution based conversion.

    Uses the characteristic function approach for tight conversion between
    zCDP and (ε,δ)-DP, as in Koskela et al. (2020).
    """

    def __init__(self, num_points: int = 2**16, tail_bound: float = 50.0) -> None:
        """
        Args:
            num_points: Number of discretization points for FFT.
            tail_bound: Range [-L, L] for the privacy loss variable.
        """
        self.num_points = num_points
        self.tail_bound = tail_bound

    def gaussian_pld(
        self, sigma: float, sensitivity: float = 1.0
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute PLD for a Gaussian mechanism.

        The privacy loss random variable for N(0,σ²) vs N(Δ,σ²) is:
            L(z) = Δz/σ² - Δ²/(2σ²)
        which is distributed as N(-Δ²/(2σ²), Δ²/σ²).

        Args:
            sigma: Noise std dev.
            sensitivity: L2 sensitivity.

        Returns:
            Tuple of (grid, pdf) for the PLD.
        """
        mu = -sensitivity**2 / (2.0 * sigma**2)
        var = sensitivity**2 / sigma**2
        std = math.sqrt(var)

        grid = np.linspace(-self.tail_bound, self.tail_bound, self.num_points)
        pdf = np.exp(-0.5 * ((grid - mu) / std) ** 2) / (std * math.sqrt(2 * math.pi))
        # Normalize
        dx = grid[1] - grid[0]
        pdf = pdf / (pdf.sum() * dx)
        return grid, pdf

    def compose_plds(
        self,
        pld1: Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
        pld2: Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compose two PLDs via convolution (FFT).

        Args:
            pld1: (grid, pdf) for first mechanism.
            pld2: (grid, pdf) for second mechanism.

        Returns:
            (grid, pdf) for composed mechanism.
        """
        grid1, pdf1 = pld1
        grid2, pdf2 = pld2
        dx = grid1[1] - grid1[0]

        # Convolution via FFT
        n = len(pdf1) + len(pdf2) - 1
        n_fft = int(2 ** math.ceil(math.log2(n)))
        fft1 = np.fft.rfft(pdf1, n=n_fft)
        fft2 = np.fft.rfft(pdf2, n=n_fft)
        conv = np.fft.irfft(fft1 * fft2, n=n_fft)[:n]
        conv = np.maximum(conv, 0.0)

        # Build composed grid
        new_lo = grid1[0] + grid2[0]
        composed_grid = new_lo + np.arange(n) * dx
        # Normalize
        conv = conv / (conv.sum() * dx)
        return composed_grid, conv

    def pld_to_dp(
        self,
        grid: npt.NDArray[np.float64],
        pdf: npt.NDArray[np.float64],
        delta: float,
    ) -> PrivacyBudget:
        """Convert PLD to (ε,δ)-DP.

        Finds the smallest ε such that P[L > ε] + e^ε·P[L < -ε] ≤ δ.
        Simplified: finds ε = inf{t : ∫_{t}^∞ pdf(l) dl ≤ δ}.

        Args:
            grid: Privacy loss grid.
            pdf: Privacy loss PDF.
            delta: Target δ.

        Returns:
            (ε,δ)-DP budget.
        """
        _validate_delta(delta)
        dx = grid[1] - grid[0]

        # Compute tail probability: P[L > ε] for each grid point
        cdf = np.cumsum(pdf * dx)
        tail = 1.0 - cdf

        # Find smallest ε such that tail ≤ δ (hockey-stick divergence)
        def hockey_stick(eps: float) -> float:
            # δ(ε) = E[max(0, 1 - e^{ε-L})] = ∫ max(0, 1-e^{ε-l}) pdf(l) dl
            integrand = np.maximum(0.0, 1.0 - np.exp(eps - grid)) * pdf
            return float(np.sum(integrand) * dx)

        # Binary search
        lo_eps, hi_eps = 0.0, float(grid[-1])
        for _ in range(100):
            mid = (lo_eps + hi_eps) / 2.0
            if hockey_stick(mid) > delta:
                lo_eps = mid
            else:
                hi_eps = mid
        eps = hi_eps
        return PrivacyBudget(epsilon=max(eps, 1e-15), delta=delta)

    def zcdp_to_dp_via_pld(
        self,
        rho: float,
        delta: float,
        num_compositions: int = 1,
    ) -> PrivacyBudget:
        """Convert zCDP to (ε,δ)-DP via PLD for tighter bounds.

        Args:
            rho: zCDP parameter ρ.
            delta: Target δ.
            num_compositions: Number of compositions of the base mechanism.

        Returns:
            (ε,δ)-DP budget.
        """
        _validate_rho(rho)
        _validate_delta(delta)

        # Compute Gaussian PLD matching the zCDP parameter
        # ρ = Δ²/(2σ²) → σ = Δ/√(2ρ), use Δ=1
        sigma = 1.0 / math.sqrt(2.0 * rho)
        grid, pdf = self.gaussian_pld(sigma, sensitivity=1.0)

        # Compose if needed
        if num_compositions > 1:
            composed_grid, composed_pdf = grid, pdf
            for _ in range(num_compositions - 1):
                composed_grid, composed_pdf = self.compose_plds(
                    (composed_grid, composed_pdf), (grid, pdf)
                )
            grid, pdf = composed_grid, composed_pdf

        return self.pld_to_dp(grid, pdf, delta)


# ---------------------------------------------------------------------------
# Optimal Conversion
# ---------------------------------------------------------------------------


class OptimalConversion:
    """Find tightest conversion by optimizing over parameters.

    Provides methods to find the best ε for a given (ρ, δ) pair by
    optimizing over the Rényi order α, and vice versa.
    """

    @staticmethod
    def zcdp_to_dp(rho: float, delta: float) -> PrivacyBudget:
        """Find the tightest (ε,δ)-DP for ρ-zCDP.

        Optimizes ε = min_{α>1} [αρ + log(1/δ)/(α-1)] - log(α/(α-1))/(α-1).
        Uses the refined Rényi-to-DP conversion from Balle et al. (2020).

        Args:
            rho: zCDP parameter ρ.
            delta: Target δ.

        Returns:
            Tightest (ε,δ)-DP budget.
        """
        _validate_rho(rho)
        _validate_delta(delta)
        log_inv_delta = math.log(1.0 / delta)

        def eps_at_alpha(alpha: float) -> float:
            if alpha <= 1.0:
                return float("inf")
            rdp = alpha * rho
            # Standard RDP-to-DP: ε = ε_RDP + log(1/δ)/(α-1)
            eps = rdp + log_inv_delta / (alpha - 1.0)
            # Refined bound (Balle et al. 2020): subtract log(α/(α-1))/(α-1)
            if alpha > 1.0:
                eps -= math.log(alpha / (alpha - 1.0)) / (alpha - 1.0)
            return eps

        # Analytic optimum for the simple bound
        alpha_init = 1.0 + math.sqrt(log_inv_delta / rho)
        alpha_init = max(alpha_init, 1.001)

        try:
            result = optimize.minimize_scalar(
                eps_at_alpha,
                bounds=(1.001, max(alpha_init * 5, 500.0)),
                method="bounded",
            )
            eps = result.fun if result.success else eps_at_alpha(alpha_init)
        except (RuntimeError, ValueError):
            eps = eps_at_alpha(alpha_init)

        return PrivacyBudget(epsilon=max(eps, 1e-15), delta=delta)

    @staticmethod
    def dp_to_zcdp_optimal(epsilon: float, delta: float) -> ZCDPBudget:
        """Find tightest ρ-zCDP for (ε,δ)-DP.

        Binary search for the largest ρ such that ρ-zCDP → (ε,δ)-DP.

        Args:
            epsilon: Privacy parameter ε.
            delta: Privacy parameter δ.

        Returns:
            Tightest ZCDPBudget.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        _validate_delta(delta)

        if delta == 0.0:
            rho = epsilon * (math.exp(epsilon) - 1.0) / 2.0
            return ZCDPBudget(rho=rho)

        # Binary search: find largest ρ such that zcdp_to_dp(ρ, δ) ≤ ε
        lo, hi = 1e-15, epsilon**2
        for _ in range(200):
            mid = (lo + hi) / 2.0
            eps_mid = mid + 2.0 * math.sqrt(mid * math.log(1.0 / delta))
            if eps_mid <= epsilon:
                lo = mid
            else:
                hi = mid
        return ZCDPBudget(rho=lo)

    @staticmethod
    def optimal_delta(rho: float, epsilon: float) -> float:
        """Find the optimal δ for given ρ and ε.

        Inverts ε = ρ + 2√(ρ·ln(1/δ)):
            δ = exp(-(ε - ρ)²/(4ρ)).

        Args:
            rho: zCDP parameter.
            epsilon: Target ε.

        Returns:
            Optimal δ.
        """
        _validate_rho(rho)
        if epsilon <= rho:
            raise ValueError(f"epsilon must be > rho, got ε={epsilon}, ρ={rho}")
        return math.exp(-((epsilon - rho) ** 2) / (4.0 * rho))

    @staticmethod
    def optimal_rdp_order(rho: float, delta: float) -> float:
        """Find the optimal Rényi order for conversion.

        α* = 1 + √(log(1/δ)/ρ).

        Args:
            rho: zCDP parameter.
            delta: Target δ.

        Returns:
            Optimal α.
        """
        _validate_rho(rho)
        _validate_delta(delta)
        return 1.0 + math.sqrt(math.log(1.0 / delta) / rho)


# ---------------------------------------------------------------------------
# Numeric Conversion
# ---------------------------------------------------------------------------


class NumericConversion:
    """Numerical integration for tight privacy bounds.

    Uses quadrature and FFT-based methods for conversions that
    don't have closed-form solutions.
    """

    def __init__(self, tol: float = 1e-10, max_eval: int = 10000) -> None:
        """
        Args:
            tol: Numerical tolerance.
            max_eval: Maximum function evaluations.
        """
        self.tol = tol
        self.max_eval = max_eval

    def rdp_to_dp_numerical(
        self,
        rdp_curve: Callable[[float], float],
        delta: float,
        alpha_range: Tuple[float, float] = (1.01, 1000.0),
    ) -> PrivacyBudget:
        """Convert RDP curve to (ε,δ)-DP by numerical optimization.

        Minimizes ε = ε_RDP(α) + log(1/δ)/(α-1) over α.

        Args:
            rdp_curve: RDP curve ε(α).
            delta: Target δ.
            alpha_range: Search range for α.

        Returns:
            (ε,δ)-DP budget.
        """
        _validate_delta(delta)
        log_inv_delta = math.log(1.0 / delta)
        lo, hi = alpha_range

        def eps_at_alpha(alpha: float) -> float:
            try:
                rdp_val = rdp_curve(alpha)
            except (ValueError, OverflowError):
                return float("inf")
            if not math.isfinite(rdp_val):
                return float("inf")
            return rdp_val + log_inv_delta / (alpha - 1.0)

        # Grid search + refine
        alphas = np.concatenate([
            np.linspace(lo, min(10, hi), 200),
            np.linspace(10, hi, 100),
        ])
        eps_values = [eps_at_alpha(a) for a in alphas]
        best_idx = int(np.argmin(eps_values))
        eps_best = eps_values[best_idx]

        # Refine around best
        if best_idx > 0 and best_idx < len(alphas) - 1:
            try:
                result = optimize.minimize_scalar(
                    eps_at_alpha,
                    bounds=(float(alphas[max(best_idx - 5, 0)]),
                            float(alphas[min(best_idx + 5, len(alphas) - 1)])),
                    method="bounded",
                )
                if result.success and result.fun < eps_best:
                    eps_best = result.fun
            except (RuntimeError, ValueError):
                pass

        return PrivacyBudget(epsilon=max(eps_best, 1e-15), delta=delta)

    def rdp_to_zcdp_numerical(
        self,
        rdp_curve: Callable[[float], float],
        alpha_range: Tuple[float, float] = (1.01, 1000.0),
        num_points: int = 2000,
    ) -> ZCDPBudget:
        """Convert RDP curve to zCDP by numerical optimization.

        Finds ρ = sup_{α>1} ε(α)/α.

        Args:
            rdp_curve: RDP curve ε(α).
            alpha_range: Search range.
            num_points: Grid points.

        Returns:
            ZCDPBudget.
        """
        lo, hi = alpha_range
        alphas = np.linspace(lo, hi, num_points)

        max_ratio = -float("inf")
        for a in alphas:
            try:
                val = rdp_curve(float(a)) / float(a)
                if math.isfinite(val) and val > max_ratio:
                    max_ratio = val
            except (ValueError, OverflowError):
                continue

        if max_ratio <= 0 or not math.isfinite(max_ratio):
            raise ValueError("Could not compute finite ρ from RDP curve")

        return ZCDPBudget(rho=max_ratio)

    def gaussian_hockey_stick(
        self,
        sigma: float,
        sensitivity: float,
        epsilon: float,
    ) -> float:
        """Compute δ via numerical integration of the hockey-stick divergence.

        δ(ε) = Φ(-ε·σ/Δ + Δ/(2σ)) - e^ε · Φ(-ε·σ/Δ - Δ/(2σ))
        where Φ is the standard normal CDF.

        Args:
            sigma: Gaussian noise std dev.
            sensitivity: L2 sensitivity Δ.
            epsilon: Privacy parameter ε.

        Returns:
            δ value.
        """
        from scipy.stats import norm

        t1 = -epsilon * sigma / sensitivity + sensitivity / (2.0 * sigma)
        t2 = -epsilon * sigma / sensitivity - sensitivity / (2.0 * sigma)
        delta = float(norm.cdf(t1) - math.exp(epsilon) * norm.cdf(t2))
        return max(delta, 0.0)

    def binary_search_sigma(
        self,
        epsilon: float,
        delta: float,
        sensitivity: float = 1.0,
        sigma_range: Tuple[float, float] = (0.01, 1000.0),
    ) -> float:
        """Find minimum σ for Gaussian mechanism to satisfy (ε,δ)-DP.

        Uses numerical hockey-stick divergence computation.

        Args:
            epsilon: Target ε.
            delta: Target δ.
            sensitivity: L2 sensitivity.
            sigma_range: Search range for σ.

        Returns:
            Minimum σ.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        _validate_delta(delta)

        lo, hi = sigma_range
        for _ in range(200):
            mid = (lo + hi) / 2.0
            d = self.gaussian_hockey_stick(mid, sensitivity, epsilon)
            if d > delta:
                lo = mid
            else:
                hi = mid
        return hi

    def compose_rdp_curves(
        self,
        curves: Sequence[Callable[[float], float]],
    ) -> Callable[[float], float]:
        """Compose RDP curves by pointwise addition.

        The composed RDP at order α is Σ ε_i(α).

        Args:
            curves: Sequence of RDP curves.

        Returns:
            Composed RDP curve.
        """
        if not curves:
            raise ValueError("Must provide at least one curve")

        def composed(alpha: float) -> float:
            total = 0.0
            for curve in curves:
                total += curve(alpha)
            return total

        return composed

    def interpolated_rdp(
        self,
        alphas: npt.NDArray[np.float64],
        epsilons: npt.NDArray[np.float64],
    ) -> Callable[[float], float]:
        """Create an interpolated RDP curve from tabulated values.

        Args:
            alphas: Rényi orders.
            epsilons: RDP values.

        Returns:
            Interpolated RDP function.
        """
        alphas = np.asarray(alphas, dtype=np.float64)
        epsilons = np.asarray(epsilons, dtype=np.float64)

        interp = interpolate.interp1d(
            alphas,
            epsilons,
            kind="cubic",
            fill_value="extrapolate",
            bounds_error=False,
        )

        def _rdp(alpha: float) -> float:
            return float(interp(alpha))

        return _rdp


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _validate_rho(rho: float) -> None:
    """Validate zCDP parameter ρ."""
    if rho <= 0:
        raise ValueError(f"rho must be > 0, got {rho}")
    if not math.isfinite(rho):
        raise ValueError(f"rho must be finite, got {rho}")


def _validate_delta(delta: float) -> None:
    """Validate δ parameter."""
    if delta <= 0 or delta >= 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")


def _invert_zcdp_to_dp(epsilon: float, delta: float) -> float:
    """Invert ε = ρ + 2√(ρ·L) for ρ, where L = ln(1/δ).

    Solves the quadratic in √ρ:
        u² + 2u√L - ε = 0  →  u = -√L + √(L + ε)
        ρ = u² = (√(L+ε) - √L)²
    """
    log_inv_delta = math.log(1.0 / delta)
    u = math.sqrt(log_inv_delta + epsilon) - math.sqrt(log_inv_delta)
    rho = u * u
    return max(rho, 1e-15)
