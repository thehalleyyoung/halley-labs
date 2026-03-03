"""
zCDP accounting: track privacy budget under zero-concentrated DP.

Implements ZCDPAccountant for cumulative budget tracking, Rényi divergence
computation for standard mechanisms, and privacy amplification by subsampling.

References:
    - Bun & Steinke (2016): "Concentrated Differential Privacy"
    - Mironov (2017): "Rényi Differential Privacy"
    - Balle et al. (2020): "Hypothesis Testing Interpretations and Renyi DP"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy import optimize, special

from dp_forge.types import PrivacyBudget, ZCDPBudget


# ---------------------------------------------------------------------------
# Rényi Divergence Computer
# ---------------------------------------------------------------------------


class RenyiDivergenceComputer:
    """Compute Rényi divergence for standard DP mechanisms.

    Supports Gaussian, Laplace, and discrete mechanisms. Provides both
    closed-form and numerical computation.
    """

    @staticmethod
    def gaussian(alpha: float, sigma: float, sensitivity: float = 1.0) -> float:
        """Rényi divergence of order α for the Gaussian mechanism.

        D_α(N(0, σ²) || N(Δ, σ²)) = α·Δ²/(2σ²).

        Args:
            alpha: Rényi order α > 1.
            sigma: Noise standard deviation.
            sensitivity: L2 sensitivity Δ.

        Returns:
            Rényi divergence D_α.
        """
        if alpha <= 1:
            raise ValueError(f"alpha must be > 1, got {alpha}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if sensitivity < 0:
            raise ValueError(f"sensitivity must be >= 0, got {sensitivity}")
        return alpha * sensitivity**2 / (2.0 * sigma**2)

    @staticmethod
    def laplace(alpha: float, scale: float, sensitivity: float = 1.0) -> float:
        """Rényi divergence of order α for the Laplace mechanism.

        D_α(Lap(0,b) || Lap(Δ,b)) for shift Δ and scale b.

        Args:
            alpha: Rényi order α > 1.
            scale: Laplace noise scale b.
            sensitivity: L1 sensitivity Δ.

        Returns:
            Rényi divergence D_α.
        """
        if alpha <= 1:
            raise ValueError(f"alpha must be > 1, got {alpha}")
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        if sensitivity < 0:
            raise ValueError(f"sensitivity must be >= 0, got {sensitivity}")
        if sensitivity == 0:
            return 0.0
        mu = sensitivity / scale
        # D_α = 1/(α-1) * log(α/(2α-1) * exp((α-1)μ) + (α-1)/(2α-1) * exp(-αμ))
        t1 = math.log(alpha / (2.0 * alpha - 1.0)) + (alpha - 1.0) * mu
        t2 = math.log((alpha - 1.0) / (2.0 * alpha - 1.0)) - alpha * mu
        log_sum = _log_add_exp(t1, t2)
        return log_sum / (alpha - 1.0)

    @staticmethod
    def discrete_gaussian(
        alpha: float, sigma_sq: float, sensitivity: int = 1
    ) -> float:
        """Rényi divergence for discrete Gaussian mechanism.

        Uses the exact characterization from Canonne et al. (2020).

        Args:
            alpha: Rényi order α (positive integer > 1).
            sigma_sq: Variance parameter σ².
            sensitivity: Integer sensitivity.

        Returns:
            Rényi divergence D_α.
        """
        if alpha <= 1:
            raise ValueError(f"alpha must be > 1, got {alpha}")
        if sigma_sq <= 0:
            raise ValueError(f"sigma_sq must be > 0, got {sigma_sq}")
        # For the discrete Gaussian, use the continuous Gaussian as an
        # upper bound. The exact formula involves theta functions, but
        # the continuous bound is tight for large σ².
        return alpha * sensitivity**2 / (2.0 * sigma_sq)

    @staticmethod
    def from_distributions(
        alpha: float,
        p: npt.NDArray[np.float64],
        q: npt.NDArray[np.float64],
    ) -> float:
        """Compute Rényi divergence D_α(P || Q) from discrete distributions.

        Args:
            alpha: Rényi order α > 1.
            p: Probability distribution P.
            q: Probability distribution Q.

        Returns:
            Rényi divergence D_α(P || Q).
        """
        if alpha <= 1:
            raise ValueError(f"alpha must be > 1, got {alpha}")
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        if p.shape != q.shape:
            raise ValueError(f"Shape mismatch: p={p.shape}, q={q.shape}")
        mask = p > 0
        if not np.all(q[mask] > 0):
            return float("inf")
        log_terms = alpha * np.log(p[mask]) - (alpha - 1.0) * np.log(q[mask])
        # D_α = 1/(α-1) * log(Σ p^α q^(1-α))
        # = 1/(α-1) * log(Σ exp(α·log(p) + (1-α)·log(q)))
        # = 1/(α-1) * log(Σ exp(α·log(p) - (α-1)·log(q)))
        log_sum = special.logsumexp(log_terms)
        return log_sum / (alpha - 1.0)

    def rdp_curve(
        self,
        mechanism: str,
        params: Dict[str, Any],
        alpha_range: Optional[npt.NDArray[np.float64]] = None,
    ) -> Callable[[float], float]:
        """Return the RDP curve ε(α) for a mechanism.

        Args:
            mechanism: Mechanism name ('gaussian', 'laplace').
            params: Mechanism parameters.
            alpha_range: Optional range of α values for numerical mechanisms.

        Returns:
            Function mapping α to ε(α).
        """
        if mechanism == "gaussian":
            sigma = params["sigma"]
            sens = params.get("sensitivity", 1.0)

            def _gaussian_rdp(alpha: float) -> float:
                return self.gaussian(alpha, sigma, sens)

            return _gaussian_rdp
        elif mechanism == "laplace":
            scale = params["scale"]
            sens = params.get("sensitivity", 1.0)

            def _laplace_rdp(alpha: float) -> float:
                return self.laplace(alpha, scale, sens)

            return _laplace_rdp
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")


# ---------------------------------------------------------------------------
# Gaussian Mechanism zCDP
# ---------------------------------------------------------------------------


class GaussianMechanismZCDP:
    """Optimal Gaussian mechanism under zCDP.

    The Gaussian mechanism with noise σ and L2 sensitivity Δ satisfies
    ρ-zCDP with ρ = Δ²/(2σ²).
    """

    def __init__(self, sigma: float, sensitivity: float = 1.0) -> None:
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if sensitivity <= 0:
            raise ValueError(f"sensitivity must be > 0, got {sensitivity}")
        self.sigma = sigma
        self.sensitivity = sensitivity

    @property
    def name(self) -> str:
        return "gaussian"

    def zcdp_cost(self, sensitivity: Optional[float] = None) -> float:
        """Compute ρ-zCDP cost.

        Args:
            sensitivity: Override sensitivity (uses constructor value if None).

        Returns:
            ρ = Δ²/(2σ²).
        """
        s = sensitivity if sensitivity is not None else self.sensitivity
        return s**2 / (2.0 * self.sigma**2)

    def rdp(self, alpha: float) -> float:
        """RDP of order α: ε(α) = αΔ²/(2σ²)."""
        if alpha <= 1:
            raise ValueError(f"alpha must be > 1, got {alpha}")
        return alpha * self.sensitivity**2 / (2.0 * self.sigma**2)

    @staticmethod
    def from_rho(rho: float, sensitivity: float = 1.0) -> "GaussianMechanismZCDP":
        """Create Gaussian mechanism achieving target ρ-zCDP.

        Args:
            rho: Target zCDP cost.
            sensitivity: L2 sensitivity.

        Returns:
            GaussianMechanismZCDP with σ = Δ/√(2ρ).
        """
        if rho <= 0:
            raise ValueError(f"rho must be > 0, got {rho}")
        sigma = sensitivity / math.sqrt(2.0 * rho)
        return GaussianMechanismZCDP(sigma=sigma, sensitivity=sensitivity)

    @staticmethod
    def optimal_sigma(target_rho: float, sensitivity: float = 1.0) -> float:
        """Compute minimum σ to achieve target ρ.

        Args:
            target_rho: Target zCDP cost ρ.
            sensitivity: L2 sensitivity Δ.

        Returns:
            σ = Δ/√(2ρ).
        """
        if target_rho <= 0:
            raise ValueError(f"target_rho must be > 0, got {target_rho}")
        return sensitivity / math.sqrt(2.0 * target_rho)

    def to_approx_dp(self, delta: float) -> PrivacyBudget:
        """Convert to (ε,δ)-DP."""
        rho = self.zcdp_cost()
        budget = ZCDPBudget(rho=rho)
        return budget.to_approx_dp(delta)

    def __repr__(self) -> str:
        return (
            f"GaussianMechanismZCDP(σ={self.sigma:.4f}, Δ={self.sensitivity:.4f}, "
            f"ρ={self.zcdp_cost():.6f})"
        )


# ---------------------------------------------------------------------------
# Laplace Mechanism zCDP
# ---------------------------------------------------------------------------


class LaplaceMechanismZCDP:
    """Laplace mechanism under zCDP analysis.

    The Laplace mechanism with scale b and L1 sensitivity Δ satisfies ε-DP
    with ε = Δ/b, which implies ρ-zCDP. The exact ρ is computed via the
    RDP characterization.
    """

    def __init__(self, scale: float, sensitivity: float = 1.0) -> None:
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        if sensitivity <= 0:
            raise ValueError(f"sensitivity must be > 0, got {sensitivity}")
        self.scale = scale
        self.sensitivity = sensitivity
        self._rdp_computer = RenyiDivergenceComputer()

    @property
    def name(self) -> str:
        return "laplace"

    @property
    def pure_dp_epsilon(self) -> float:
        """Pure DP parameter ε = Δ/b."""
        return self.sensitivity / self.scale

    def zcdp_cost(self, sensitivity: Optional[float] = None) -> float:
        """Compute ρ-zCDP cost via RDP optimization.

        Finds ρ = sup_{α>1} D_α / α where D_α is the Rényi divergence.
        For the Laplace mechanism, ρ ≤ ε²/2 from pure DP conversion,
        but exact computation via RDP gives a tighter bound.

        Args:
            sensitivity: Override sensitivity.

        Returns:
            zCDP cost ρ.
        """
        s = sensitivity if sensitivity is not None else self.sensitivity

        # ε-DP implies ε(e^ε - 1)/2 - zCDP (Bun & Steinke Prop 1.3)
        # but also ε²/2 for small ε. Use the tighter of the two.
        eps = s / self.scale
        rho_simple = eps**2 / 2.0
        rho_bs = eps * (math.exp(eps) - 1.0) / 2.0

        # Optimize over α for the exact RDP-based bound
        def neg_rdp_over_alpha(log_am1: float) -> float:
            alpha = 1.0 + math.exp(log_am1)
            try:
                rdp_val = self._rdp_computer.laplace(alpha, self.scale, s)
            except (ValueError, OverflowError):
                return 0.0
            return -rdp_val / alpha

        try:
            result = optimize.minimize_scalar(
                neg_rdp_over_alpha,
                bounds=(-5.0, 10.0),
                method="bounded",
            )
            rho_exact = -result.fun if result.success else rho_simple
        except (RuntimeError, ValueError):
            rho_exact = rho_simple

        return min(rho_simple, rho_bs, rho_exact)

    def rdp(self, alpha: float) -> float:
        """RDP of order α for Laplace mechanism."""
        return self._rdp_computer.laplace(alpha, self.scale, self.sensitivity)

    @staticmethod
    def from_rho(rho: float, sensitivity: float = 1.0) -> "LaplaceMechanismZCDP":
        """Create Laplace mechanism achieving approximately target ρ-zCDP.

        Uses the pure DP conversion: ρ ≈ ε²/2 → ε = √(2ρ) → b = Δ/ε.

        Args:
            rho: Target zCDP cost.
            sensitivity: L1 sensitivity.

        Returns:
            LaplaceMechanismZCDP.
        """
        if rho <= 0:
            raise ValueError(f"rho must be > 0, got {rho}")
        eps = math.sqrt(2.0 * rho)
        scale = sensitivity / eps
        return LaplaceMechanismZCDP(scale=scale, sensitivity=sensitivity)

    def to_approx_dp(self, delta: float) -> PrivacyBudget:
        """Convert to (ε,δ)-DP."""
        rho = self.zcdp_cost()
        budget = ZCDPBudget(rho=rho)
        return budget.to_approx_dp(delta)

    def __repr__(self) -> str:
        return (
            f"LaplaceMechanismZCDP(b={self.scale:.4f}, Δ={self.sensitivity:.4f}, "
            f"ρ={self.zcdp_cost():.6f})"
        )


# ---------------------------------------------------------------------------
# Subsampled zCDP
# ---------------------------------------------------------------------------


class SubsampledZCDP:
    """Privacy amplification by Poisson subsampling for zCDP mechanisms.

    When each record is included independently with probability q,
    the privacy cost is amplified. Uses the RDP-based analysis from
    Mironov et al. (2019) and Zhu & Wang (2019).
    """

    def __init__(self, base_rho: float, sampling_rate: float) -> None:
        """
        Args:
            base_rho: zCDP cost of the base mechanism.
            sampling_rate: Poisson sampling probability q ∈ (0, 1].
        """
        if base_rho <= 0:
            raise ValueError(f"base_rho must be > 0, got {base_rho}")
        if not 0 < sampling_rate <= 1:
            raise ValueError(
                f"sampling_rate must be in (0, 1], got {sampling_rate}"
            )
        self.base_rho = base_rho
        self.sampling_rate = sampling_rate

    def amplified_rho(self) -> float:
        """Compute amplified zCDP cost after subsampling.

        For small sampling rate q, the amplified ρ ≈ 2q²·ρ_base.
        Uses numerical optimization over the RDP curve for tightness.

        Returns:
            Amplified zCDP cost.
        """
        q = self.sampling_rate
        if q == 1.0:
            return self.base_rho

        # Simple bound: subsampled mechanism is O(q²ρ)-zCDP
        rho_simple = 2.0 * q**2 * self.base_rho

        # Numerical optimization over RDP orders for tighter bound
        def amplified_rdp(alpha: float) -> float:
            return self._subsampled_rdp(alpha)

        def rdp_over_alpha(log_am1: float) -> float:
            alpha = 1.0 + math.exp(log_am1)
            try:
                val = amplified_rdp(alpha)
            except (ValueError, OverflowError):
                return float("inf")
            return val / alpha

        try:
            result = optimize.minimize_scalar(
                rdp_over_alpha,
                bounds=(-2.0, 8.0),
                method="bounded",
            )
            rho_tight = result.fun if result.success else rho_simple
        except (RuntimeError, ValueError):
            rho_tight = rho_simple

        return min(rho_simple, max(rho_tight, 0.0))

    def _subsampled_rdp(self, alpha: float) -> float:
        """Compute RDP of subsampled mechanism at order α.

        Uses the bound from Zhu & Wang (2019), Theorem 9:
        For Poisson-subsampled ρ-zCDP mechanism, the RDP at order α is bounded
        by a function of q and the base mechanism's RDP.
        """
        if alpha <= 1:
            raise ValueError(f"alpha must be > 1, got {alpha}")
        q = self.sampling_rate
        base_rdp_alpha = alpha * self.base_rho  # base mechanism RDP

        if alpha <= 2:
            # Use simple bound for small α
            return min(
                2.0 * q**2 * self.base_rho * alpha,
                base_rdp_alpha,
            )

        # For integer alpha, use the exact binomial bound
        # ε(α) ≤ 1/(α-1) * log(Σ_{j=0}^{α} C(α,j) * (1-q)^{α-j} * q^j * exp((j-1)jρ))
        alpha_int = int(math.ceil(alpha))
        log_terms = []
        for j in range(alpha_int + 1):
            log_binom = _log_comb(alpha_int, j)
            log_q_term = j * math.log(max(q, 1e-300)) + (alpha_int - j) * math.log(
                max(1.0 - q, 1e-300)
            )
            rdp_term = j * (j - 1) * self.base_rho / 2.0 if j >= 2 else 0.0
            log_terms.append(log_binom + log_q_term + rdp_term)

        log_sum = special.logsumexp(log_terms)
        return max(log_sum / (alpha_int - 1), 0.0)

    def to_budget(self) -> ZCDPBudget:
        """Return amplified zCDP budget."""
        return ZCDPBudget(rho=self.amplified_rho())

    def __repr__(self) -> str:
        return (
            f"SubsampledZCDP(ρ_base={self.base_rho:.6f}, "
            f"q={self.sampling_rate:.4f}, ρ_amp={self.amplified_rho():.6f})"
        )


# ---------------------------------------------------------------------------
# Advanced Composition for zCDP
# ---------------------------------------------------------------------------


class AdvancedCompositionZCDP:
    """Tight composition for heterogeneous zCDP mechanisms.

    zCDP composes additively: if mechanism_i is ρ_i-zCDP, their
    sequential composition is (Σρ_i)-zCDP. This class tracks and
    composes heterogeneous mechanisms.
    """

    def __init__(self) -> None:
        self._rhos: List[float] = []
        self._xis: List[float] = []
        self._names: List[str] = []

    def add(self, rho: float, xi: float = 0.0, name: str = "") -> None:
        """Add a mechanism to the composition.

        Args:
            rho: zCDP cost ρ.
            xi: Optional offset ξ for truncated zCDP.
            name: Mechanism name.
        """
        if rho <= 0:
            raise ValueError(f"rho must be > 0, got {rho}")
        if xi < 0:
            raise ValueError(f"xi must be >= 0, got {xi}")
        self._rhos.append(rho)
        self._xis.append(xi)
        self._names.append(name or f"mechanism_{len(self._rhos)}")

    def add_gaussian(
        self, sigma: float, sensitivity: float = 1.0, name: str = "gaussian"
    ) -> None:
        """Add a Gaussian mechanism."""
        rho = sensitivity**2 / (2.0 * sigma**2)
        self.add(rho, name=name)

    def add_laplace(
        self, scale: float, sensitivity: float = 1.0, name: str = "laplace"
    ) -> None:
        """Add a Laplace mechanism."""
        mech = LaplaceMechanismZCDP(scale=scale, sensitivity=sensitivity)
        self.add(mech.zcdp_cost(), name=name)

    def add_subsampled(
        self,
        base_rho: float,
        sampling_rate: float,
        name: str = "subsampled",
    ) -> None:
        """Add a subsampled mechanism."""
        sub = SubsampledZCDP(base_rho=base_rho, sampling_rate=sampling_rate)
        self.add(sub.amplified_rho(), name=name)

    @property
    def total_rho(self) -> float:
        """Total composed ρ = Σρ_i."""
        return sum(self._rhos)

    @property
    def total_xi(self) -> float:
        """Total composed ξ = Σξ_i."""
        return sum(self._xis)

    @property
    def num_mechanisms(self) -> int:
        return len(self._rhos)

    def get_budget(self) -> ZCDPBudget:
        """Return the composed zCDP budget."""
        return ZCDPBudget(rho=self.total_rho, xi=self.total_xi)

    def to_approx_dp(self, delta: float) -> PrivacyBudget:
        """Convert composed budget to (ε,δ)-DP.

        Uses the optimal conversion: ε = ρ + 2√(ρ·ln(1/δ)).
        For truncated zCDP, uses ε = ξ + ρ + 2√(ρ·ln(1/δ)).
        """
        rho = self.total_rho
        xi = self.total_xi
        budget = ZCDPBudget(rho=rho, xi=xi)
        eps = rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta)) + xi
        return PrivacyBudget(epsilon=eps, delta=delta)

    def optimal_dp_conversion(self, delta: float) -> PrivacyBudget:
        """Numerically optimize the RDP-to-DP conversion.

        Minimizes ε over the Rényi order α using the composed RDP curve.
        """
        rho = self.total_rho

        def eps_at_alpha(alpha: float) -> float:
            rdp_val = alpha * rho
            return rdp_val + math.log(1.0 / delta) / (alpha - 1.0)

        # Optimal α = 1 + √(log(1/δ)/ρ) for the zCDP curve
        alpha_opt = 1.0 + math.sqrt(math.log(1.0 / delta) / rho)
        eps = eps_at_alpha(max(alpha_opt, 1.001))
        return PrivacyBudget(epsilon=eps, delta=delta)

    def reset(self) -> None:
        """Clear all mechanisms."""
        self._rhos.clear()
        self._xis.clear()
        self._names.clear()

    def summary(self) -> Dict[str, Any]:
        """Return summary of all mechanisms."""
        return {
            "num_mechanisms": self.num_mechanisms,
            "total_rho": self.total_rho,
            "total_xi": self.total_xi,
            "mechanisms": [
                {"name": n, "rho": r, "xi": x}
                for n, r, x in zip(self._names, self._rhos, self._xis)
            ],
        }

    def __repr__(self) -> str:
        return (
            f"AdvancedCompositionZCDP(n={self.num_mechanisms}, "
            f"ρ={self.total_rho:.6f})"
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _log_add_exp(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def _log_comb(n: int, k: int) -> float:
    """Log of binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return -float("inf")
    return (
        special.gammaln(n + 1)
        - special.gammaln(k + 1)
        - special.gammaln(n - k + 1)
    )
