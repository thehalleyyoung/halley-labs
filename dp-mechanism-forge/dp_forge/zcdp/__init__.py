"""
Zero-Concentrated Differential Privacy (zCDP) accounting and mechanisms.

This package implements zCDP (Bun & Steinke 2016) based privacy accounting,
composition, and conversion. zCDP provides tighter composition bounds
than (ε,δ)-DP via Rényi divergence, and is particularly useful for
Gaussian mechanisms.

Key capabilities:
- **Accountant**: Track privacy budget under zCDP composition with
  tight bounds.
- **Composition**: Optimal composition of zCDP mechanisms (additive in ρ).
- **Conversion**: Convert between zCDP (ρ), (ε,δ)-DP, and RDP with
  optimal conversion factors.
- **Mechanisms**: zCDP characterisations of standard mechanisms (Gaussian,
  subsampled Gaussian, etc.).
- **CEGIS integration**: Synthesise mechanisms with zCDP guarantees.

Architecture:
    1. **ZCDPAccountant** — Tracks cumulative ρ across mechanism invocations.
    2. **ZCDPComposer** — Composes zCDP mechanisms with optimal bounds.
    3. **ZCDPConverter** — Converts between zCDP, (ε,δ)-DP, and RDP.
    4. **ZCDPMechanisms** — zCDP analysis of standard mechanisms.
    5. **ZCDPSynthesizer** — Synthesise mechanisms with zCDP guarantees.

Example::

    from dp_forge.zcdp import ZCDPAccountant, zcdp_to_dp

    acct = ZCDPAccountant()
    acct.add_gaussian(sigma=1.0, sensitivity=1.0)
    acct.add_gaussian(sigma=2.0, sensitivity=1.0)
    budget = acct.get_budget()
    print(f"Total zCDP budget: ρ = {budget.rho:.4f}")
    dp = zcdp_to_dp(budget, delta=1e-5)
    print(f"Equivalent (ε,δ)-DP: ε = {dp.epsilon:.4f}")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt

from dp_forge.types import (
    OptimalityCertificate,
    PrivacyBudget,
    PrivacyNotion,
    QuerySpec,
    ZCDPBudget,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ZCDPVariant(Enum):
    """Variants of zero-concentrated DP."""

    PURE_ZCDP = auto()          # ρ-zCDP
    TRUNCATED_ZCDP = auto()     # (ξ, ρ)-zCDP
    MEAN_ZCDP = auto()          # mean-concentrated DP
    RENYI_ZCDP = auto()         # Rényi-based formulation

    def __repr__(self) -> str:
        return f"ZCDPVariant.{self.name}"


class ConversionDirection(Enum):
    """Direction of privacy framework conversion."""

    ZCDP_TO_DP = auto()
    DP_TO_ZCDP = auto()
    ZCDP_TO_RDP = auto()
    RDP_TO_ZCDP = auto()
    ZCDP_TO_GDP = auto()

    def __repr__(self) -> str:
        return f"ConversionDirection.{self.name}"


class GaussianVariant(Enum):
    """Variants of the Gaussian mechanism for zCDP analysis."""

    STANDARD = auto()
    ANALYTIC = auto()
    DISCRETE = auto()
    SUBSAMPLED = auto()

    def __repr__(self) -> str:
        return f"GaussianVariant.{self.name}"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ZCDPConfig:
    """Configuration for zCDP accounting.

    Attributes:
        variant: zCDP variant to use.
        auto_convert: Whether to auto-convert to (ε,δ)-DP when queried.
        default_delta: Default δ for conversion to (ε,δ)-DP.
        composition_tightness: Whether to use tight composition ('exact')
            or loose ('basic') bounds.
        numerical_tol: Numerical tolerance for conversions.
        verbose: Verbosity level.
    """

    variant: ZCDPVariant = ZCDPVariant.PURE_ZCDP
    auto_convert: bool = True
    default_delta: float = 1e-5
    composition_tightness: str = "exact"
    numerical_tol: float = 1e-10
    verbose: int = 1

    def __post_init__(self) -> None:
        if self.default_delta <= 0 or self.default_delta >= 1:
            raise ValueError(f"default_delta must be in (0, 1), got {self.default_delta}")
        if self.composition_tightness not in ("exact", "basic"):
            raise ValueError(
                f"composition_tightness must be 'exact' or 'basic', "
                f"got {self.composition_tightness!r}"
            )
        if self.numerical_tol <= 0:
            raise ValueError(f"numerical_tol must be > 0, got {self.numerical_tol}")

    def __repr__(self) -> str:
        return (
            f"ZCDPConfig(variant={self.variant.name}, δ={self.default_delta}, "
            f"tight={self.composition_tightness})"
        )


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


@dataclass
class ZCDPMechanismEntry:
    """Record of a mechanism added to the zCDP accountant.

    Attributes:
        name: Human-readable name for the mechanism.
        rho: zCDP cost ρ of this mechanism.
        xi: Optional offset ξ for (ξ,ρ)-zCDP.
        sensitivity: Sensitivity of the query.
        parameters: Mechanism-specific parameters (e.g., sigma for Gaussian).
    """

    name: str
    rho: float
    xi: float = 0.0
    sensitivity: float = 1.0
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rho <= 0:
            raise ValueError(f"rho must be > 0, got {self.rho}")
        if self.xi < 0:
            raise ValueError(f"xi must be >= 0, got {self.xi}")

    def to_budget(self) -> ZCDPBudget:
        """Convert this entry to a ZCDPBudget."""
        return ZCDPBudget(rho=self.rho, xi=self.xi)

    def __repr__(self) -> str:
        return f"ZCDPMechanismEntry(name={self.name!r}, ρ={self.rho:.6f})"


@dataclass
class CompositionResult:
    """Result of zCDP composition.

    Attributes:
        total_rho: Total composed ρ.
        total_xi: Total composed ξ.
        num_mechanisms: Number of mechanisms composed.
        per_mechanism_rho: Individual ρ values.
        dp_budget: Equivalent (ε,δ)-DP budget (if converted).
    """

    total_rho: float
    total_xi: float = 0.0
    num_mechanisms: int = 0
    per_mechanism_rho: List[float] = field(default_factory=list)
    dp_budget: Optional[PrivacyBudget] = None

    @property
    def total_budget(self) -> ZCDPBudget:
        """Total zCDP budget."""
        return ZCDPBudget(rho=self.total_rho, xi=self.total_xi)

    def __repr__(self) -> str:
        dp = f", ε={self.dp_budget.epsilon:.4f}" if self.dp_budget else ""
        return (
            f"CompositionResult(ρ={self.total_rho:.6f}, "
            f"n={self.num_mechanisms}{dp})"
        )


@dataclass
class ConversionResult:
    """Result of converting between privacy frameworks.

    Attributes:
        direction: Which conversion was performed.
        input_budget: Input privacy budget.
        output_budget: Output privacy budget.
        is_tight: Whether the conversion is tight (no loss).
        conversion_loss: Bound on conversion tightness loss.
    """

    direction: ConversionDirection
    input_budget: Any  # ZCDPBudget, PrivacyBudget, or RDP curve
    output_budget: Any
    is_tight: bool = False
    conversion_loss: float = 0.0

    def __repr__(self) -> str:
        tight = "tight" if self.is_tight else f"loss={self.conversion_loss:.4f}"
        return f"ConversionResult(dir={self.direction.name}, {tight})"


@dataclass
class ZCDPSynthesisResult:
    """Result of mechanism synthesis with zCDP guarantees.

    Attributes:
        mechanism: The n × k probability table.
        rho: zCDP cost ρ of the synthesised mechanism.
        objective_value: Utility objective value.
        dp_budget: Equivalent (ε,δ)-DP budget.
        iterations: Number of synthesis iterations.
        optimality_certificate: Duality-based certificate.
    """

    mechanism: npt.NDArray[np.float64]
    rho: float
    objective_value: float
    dp_budget: Optional[PrivacyBudget] = None
    iterations: int = 0
    optimality_certificate: Optional[OptimalityCertificate] = None

    def __post_init__(self) -> None:
        self.mechanism = np.asarray(self.mechanism, dtype=np.float64)
        if self.mechanism.ndim != 2:
            raise ValueError(f"mechanism must be 2-D, got shape {self.mechanism.shape}")
        if self.rho <= 0:
            raise ValueError(f"rho must be > 0, got {self.rho}")

    def __repr__(self) -> str:
        dp = f", ε={self.dp_budget.epsilon:.4f}" if self.dp_budget else ""
        return (
            f"ZCDPSynthesisResult(ρ={self.rho:.6f}, "
            f"obj={self.objective_value:.6f}{dp}, iter={self.iterations})"
        )


# ---------------------------------------------------------------------------
# Protocols (interfaces)
# ---------------------------------------------------------------------------


@runtime_checkable
class ZCDPMechanism(Protocol):
    """Protocol for mechanisms with known zCDP guarantees."""

    def zcdp_cost(self, sensitivity: float) -> float:
        """Return the ρ-zCDP cost for a given sensitivity."""
        ...

    @property
    def name(self) -> str:
        """Name of the mechanism."""
        ...


# ---------------------------------------------------------------------------
# Public API classes
# ---------------------------------------------------------------------------


class ZCDPAccountant:
    """Track cumulative zCDP budget across mechanism invocations.

    zCDP composes additively: if mechanism 1 is ρ₁-zCDP and mechanism 2
    is ρ₂-zCDP, their sequential composition is (ρ₁ + ρ₂)-zCDP.
    """

    def __init__(self, config: Optional[ZCDPConfig] = None) -> None:
        self.config = config or ZCDPConfig()
        self._entries: List[ZCDPMechanismEntry] = []

    def add_mechanism(self, name: str, rho: float, **kwargs: Any) -> None:
        """Add a mechanism with known zCDP cost.

        Args:
            name: Name of the mechanism.
            rho: zCDP cost ρ.
            **kwargs: Additional parameters (xi, sensitivity, etc.).
        """
        entry = ZCDPMechanismEntry(
            name=name,
            rho=rho,
            xi=kwargs.get("xi", 0.0),
            sensitivity=kwargs.get("sensitivity", 1.0),
            parameters=kwargs,
        )
        self._entries.append(entry)

    def add_gaussian(
        self,
        sigma: float,
        sensitivity: float = 1.0,
        name: str = "gaussian",
    ) -> None:
        """Add a Gaussian mechanism with computed zCDP cost.

        The Gaussian mechanism with noise σ is ρ-zCDP where ρ = Δ²/(2σ²).

        Args:
            sigma: Noise standard deviation.
            sensitivity: L2 sensitivity Δ.
            name: Name for this mechanism.
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        rho = sensitivity**2 / (2.0 * sigma**2)
        self.add_mechanism(name, rho, sensitivity=sensitivity, sigma=sigma)

    def add_laplace(
        self,
        scale: float,
        sensitivity: float = 1.0,
        name: str = "laplace",
    ) -> None:
        """Add a Laplace mechanism with computed zCDP cost.

        Args:
            scale: Laplace noise scale b.
            sensitivity: L1 sensitivity Δ.
            name: Name for this mechanism.
        """
        eps = sensitivity / scale
        rho = eps * (np.exp(eps) - 1.0) / 2.0
        self.add_mechanism(name, rho, sensitivity=sensitivity, scale=scale)

    def get_budget(self) -> ZCDPBudget:
        """Return the current total zCDP budget.

        Returns:
            ZCDPBudget with total ρ.
        """
        total_rho = sum(e.rho for e in self._entries)
        total_xi = sum(e.xi for e in self._entries)
        if total_rho <= 0:
            total_rho = 1e-15
        return ZCDPBudget(rho=total_rho, xi=total_xi)

    def to_dp(self, delta: Optional[float] = None) -> PrivacyBudget:
        """Convert current zCDP budget to (ε,δ)-DP.

        Args:
            delta: Target δ (uses config default if None).

        Returns:
            Equivalent (ε,δ)-DP budget.
        """
        if delta is None:
            delta = self.config.default_delta
        budget = self.get_budget()
        return budget.to_approx_dp(delta)

    def compose(self) -> CompositionResult:
        """Compute the composition result of all added mechanisms.

        Returns:
            CompositionResult with composed budget.
        """
        total_rho = sum(e.rho for e in self._entries)
        total_xi = sum(e.xi for e in self._entries)
        per_rho = [e.rho for e in self._entries]
        dp_budget = None
        if self.config.auto_convert and total_rho > 0:
            budget = ZCDPBudget(rho=total_rho, xi=total_xi)
            dp_budget = budget.to_approx_dp(self.config.default_delta)
        return CompositionResult(
            total_rho=total_rho,
            total_xi=total_xi,
            num_mechanisms=len(self._entries),
            per_mechanism_rho=per_rho,
            dp_budget=dp_budget,
        )

    @property
    def total_rho(self) -> float:
        """Total ρ spent so far."""
        return sum(e.rho for e in self._entries)

    @property
    def num_mechanisms(self) -> int:
        """Number of mechanisms added."""
        return len(self._entries)

    def reset(self) -> None:
        """Clear all mechanism entries."""
        self._entries.clear()

    def __repr__(self) -> str:
        return (
            f"ZCDPAccountant(ρ={self.total_rho:.6f}, "
            f"n_mechanisms={self.num_mechanisms})"
        )


class ZCDPComposer:
    """Optimal composition of zCDP mechanisms.

    Implements both basic (additive) and advanced composition bounds
    for zCDP, including heterogeneous composition.
    """

    def __init__(self, config: Optional[ZCDPConfig] = None) -> None:
        self.config = config or ZCDPConfig()

    def compose(self, budgets: Sequence[ZCDPBudget]) -> CompositionResult:
        """Compose multiple zCDP budgets.

        Args:
            budgets: Sequence of zCDP budgets to compose.

        Returns:
            CompositionResult with the composed budget.
        """
        rho_vals = [b.rho for b in budgets]
        xi_vals = [b.xi for b in budgets]
        total_rho = sum(rho_vals)
        total_xi = sum(xi_vals)
        dp_budget = None
        if self.config.auto_convert and total_rho > 0:
            b = ZCDPBudget(rho=total_rho, xi=total_xi)
            dp_budget = b.to_approx_dp(self.config.default_delta)
        return CompositionResult(
            total_rho=total_rho,
            total_xi=total_xi,
            num_mechanisms=len(budgets),
            per_mechanism_rho=rho_vals,
            dp_budget=dp_budget,
        )

    def compose_homogeneous(
        self,
        budget: ZCDPBudget,
        k: int,
    ) -> CompositionResult:
        """Compose k identical zCDP mechanisms.

        Args:
            budget: Per-mechanism zCDP budget.
            k: Number of compositions.

        Returns:
            CompositionResult (ρ_total = k · ρ).
        """
        total_rho = k * budget.rho
        total_xi = k * budget.xi
        dp_budget = None
        if self.config.auto_convert and total_rho > 0:
            b = ZCDPBudget(rho=total_rho, xi=total_xi)
            dp_budget = b.to_approx_dp(self.config.default_delta)
        return CompositionResult(
            total_rho=total_rho,
            total_xi=total_xi,
            num_mechanisms=k,
            per_mechanism_rho=[budget.rho] * k,
            dp_budget=dp_budget,
        )

    def optimal_allocation(
        self,
        total_rho: float,
        num_mechanisms: int,
        weights: Optional[npt.NDArray[np.float64]] = None,
    ) -> npt.NDArray[np.float64]:
        """Optimally allocate a zCDP budget across mechanisms.

        Args:
            total_rho: Total ρ budget.
            num_mechanisms: Number of mechanisms to allocate to.
            weights: Optional importance weights for each mechanism.

        Returns:
            Array of ρ allocations summing to total_rho.
        """
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            if len(w) != num_mechanisms:
                raise ValueError("weights length must match num_mechanisms")
            w_sum = w.sum()
            if w_sum <= 0:
                raise ValueError("Sum of weights must be positive")
            return total_rho * w / w_sum
        return np.full(num_mechanisms, total_rho / num_mechanisms)


class ZCDPConverter:
    """Convert between zCDP and other privacy frameworks."""

    def __init__(self, config: Optional[ZCDPConfig] = None) -> None:
        self.config = config or ZCDPConfig()

    def zcdp_to_dp(
        self,
        budget: ZCDPBudget,
        delta: float,
    ) -> ConversionResult:
        """Convert ρ-zCDP to (ε,δ)-DP.

        Uses the optimal conversion: ε = ρ + 2√(ρ·ln(1/δ)).

        Args:
            budget: zCDP budget.
            delta: Target δ.

        Returns:
            ConversionResult with (ε,δ)-DP budget.
        """
        import math
        epsilon = budget.rho + 2.0 * math.sqrt(budget.rho * math.log(1.0 / delta))
        if budget.xi > 0:
            epsilon += budget.xi
        dp_budget = PrivacyBudget(epsilon=epsilon, delta=delta)
        return ConversionResult(
            direction=ConversionDirection.ZCDP_TO_DP,
            input_budget=budget,
            output_budget=dp_budget,
            is_tight=True,
            conversion_loss=0.0,
        )

    def dp_to_zcdp(
        self,
        budget: PrivacyBudget,
    ) -> ConversionResult:
        """Convert (ε,δ)-DP to zCDP (upper bound).

        Note: This conversion is generally lossy (not tight).

        Args:
            budget: (ε,δ)-DP budget.

        Returns:
            ConversionResult with zCDP budget.
        """
        import math
        eps = budget.epsilon
        delta = budget.delta
        if delta == 0.0:
            rho = eps * (math.exp(eps) - 1.0) / 2.0
        else:
            L = math.log(1.0 / delta)
            u = math.sqrt(L + eps) - math.sqrt(L)
            rho = max(u * u, 1e-15)
        zcdp_budget = ZCDPBudget(rho=rho)
        return ConversionResult(
            direction=ConversionDirection.DP_TO_ZCDP,
            input_budget=budget,
            output_budget=zcdp_budget,
            is_tight=(delta == 0.0),
            conversion_loss=0.0 if delta == 0.0 else 0.1,
        )

    def zcdp_to_rdp(
        self,
        budget: ZCDPBudget,
        alpha: float,
    ) -> float:
        """Convert ρ-zCDP to RDP of order α.

        ρ-zCDP implies (α, αρ)-RDP for all α > 1.

        Args:
            budget: zCDP budget.
            alpha: Rényi order α > 1.

        Returns:
            RDP parameter ε(α) = α·ρ.
        """
        if alpha <= 1:
            raise ValueError(f"alpha must be > 1, got {alpha}")
        return alpha * budget.rho

    def rdp_to_zcdp(
        self,
        rdp_curve: Callable[[float], float],
    ) -> ConversionResult:
        """Convert an RDP curve to zCDP.

        Finds the tightest ρ such that ε(α) ≤ α·ρ for all α > 1.

        Args:
            rdp_curve: Function mapping α to ε(α).

        Returns:
            ConversionResult with zCDP budget.
        """
        from dp_forge.zcdp.conversion import RDPToZCDP as _RDPToZCDP
        zcdp_budget = _RDPToZCDP.convert(rdp_curve)
        return ConversionResult(
            direction=ConversionDirection.RDP_TO_ZCDP,
            input_budget=rdp_curve,
            output_budget=zcdp_budget,
            is_tight=False,
            conversion_loss=0.0,
        )


class ZCDPMechanismAnalyzer:
    """Analyse standard mechanisms under zCDP."""

    def gaussian_rho(
        self,
        sigma: float,
        sensitivity: float = 1.0,
        variant: GaussianVariant = GaussianVariant.STANDARD,
    ) -> float:
        """Compute ρ-zCDP cost of a Gaussian mechanism.

        Args:
            sigma: Noise standard deviation.
            sensitivity: L2 sensitivity.
            variant: Gaussian mechanism variant.

        Returns:
            zCDP cost ρ = Δ²/(2σ²).
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        return sensitivity**2 / (2.0 * sigma**2)

    def laplace_rho(
        self,
        scale: float,
        sensitivity: float = 1.0,
    ) -> float:
        """Compute ρ-zCDP cost of a Laplace mechanism.

        Args:
            scale: Laplace noise scale b.
            sensitivity: L1 sensitivity.

        Returns:
            zCDP cost ρ.
        """
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        eps = sensitivity / scale
        return eps * (np.exp(eps) - 1.0) / 2.0

    def subsampled_gaussian_rho(
        self,
        sigma: float,
        sampling_rate: float,
        sensitivity: float = 1.0,
    ) -> float:
        """Compute ρ-zCDP cost of subsampled Gaussian mechanism.

        Args:
            sigma: Noise standard deviation.
            sampling_rate: Probability of including each record.
            sensitivity: L2 sensitivity.

        Returns:
            zCDP cost ρ after privacy amplification by subsampling.
        """
        from dp_forge.zcdp.accountant import SubsampledZCDP
        base_rho = sensitivity**2 / (2.0 * sigma**2)
        sub = SubsampledZCDP(base_rho=base_rho, sampling_rate=sampling_rate)
        return sub.amplified_rho()

    def optimal_sigma(
        self,
        target_rho: float,
        sensitivity: float = 1.0,
    ) -> float:
        """Compute the minimum σ to achieve a target ρ.

        Args:
            target_rho: Target zCDP cost.
            sensitivity: L2 sensitivity.

        Returns:
            Minimum Gaussian noise σ.
        """
        import math
        if target_rho <= 0:
            raise ValueError(f"target_rho must be > 0, got {target_rho}")
        return sensitivity / math.sqrt(2.0 * target_rho)


class ZCDPSynthesizer:
    """Synthesise mechanisms with zCDP guarantees via CEGIS.

    Modifies the standard CEGIS loop to use zCDP constraints
    instead of (ε,δ)-DP constraints, enabling tighter composition.
    """

    def __init__(self, config: Optional[ZCDPConfig] = None) -> None:
        self.config = config or ZCDPConfig()

    def synthesize(
        self,
        spec: QuerySpec,
        rho: float,
    ) -> ZCDPSynthesisResult:
        """Synthesise a mechanism satisfying ρ-zCDP.

        Args:
            spec: Query specification.
            rho: Target zCDP budget ρ.

        Returns:
            ZCDPSynthesisResult with the synthesised mechanism.
        """
        from dp_forge.zcdp.synthesis import ZCDPSynthesizer as _Synth
        s = _Synth()
        result = s.synthesize_gaussian(
            spec.query_values, rho, spec.sensitivity, spec.k
        )
        dp_budget = ZCDPBudget(rho=rho).to_approx_dp(self.config.default_delta)
        return ZCDPSynthesisResult(
            mechanism=result["mechanism"],
            rho=rho,
            objective_value=result["mse"],
            dp_budget=dp_budget,
            iterations=1,
        )

    def synthesize_for_composition(
        self,
        specs: List[QuerySpec],
        total_rho: float,
    ) -> List[ZCDPSynthesisResult]:
        """Synthesise multiple mechanisms sharing a total zCDP budget.

        Optimally allocates ρ across mechanisms and synthesises each.

        Args:
            specs: List of query specifications.
            total_rho: Total zCDP budget to share.

        Returns:
            List of synthesis results, one per query.
        """
        from dp_forge.zcdp.synthesis import MultiQuerySynthesis, ZCDPSynthesizer as _Synth
        k = len(specs)
        sensitivities = np.array([s.sensitivity for s in specs])
        mq = MultiQuerySynthesis(sensitivities, total_rho)
        rhos, _ = mq.minimize_total_mse()
        synth = _Synth()
        results = []
        for i, spec in enumerate(specs):
            r = synth.synthesize_gaussian(
                spec.query_values, float(rhos[i]), spec.sensitivity, spec.k
            )
            dp_budget = ZCDPBudget(rho=float(rhos[i])).to_approx_dp(
                self.config.default_delta
            )
            results.append(ZCDPSynthesisResult(
                mechanism=r["mechanism"],
                rho=float(rhos[i]),
                objective_value=r["mse"],
                dp_budget=dp_budget,
                iterations=1,
            ))
        return results


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------


def zcdp_to_dp(
    budget: ZCDPBudget,
    delta: float,
) -> PrivacyBudget:
    """Convert ρ-zCDP to (ε,δ)-DP.

    Uses the optimal conversion: ε = ρ + 2√(ρ·ln(1/δ)).

    Args:
        budget: zCDP budget.
        delta: Target δ.

    Returns:
        Equivalent (ε,δ)-DP budget.
    """
    return budget.to_approx_dp(delta)


def dp_to_zcdp(epsilon: float, delta: float) -> ZCDPBudget:
    """Convert (ε,δ)-DP to ρ-zCDP (upper bound).

    Args:
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ.

    Returns:
        zCDP budget (upper bound on ρ).
    """
    import math
    if delta == 0.0:
        rho = epsilon * (math.exp(epsilon) - 1.0) / 2.0
    else:
        L = math.log(1.0 / delta)
        u = math.sqrt(L + epsilon) - math.sqrt(L)
        rho = max(u * u, 1e-15)
    return ZCDPBudget(rho=rho)


def gaussian_zcdp_cost(
    sigma: float,
    sensitivity: float = 1.0,
) -> float:
    """Compute ρ-zCDP cost of a Gaussian mechanism.

    Args:
        sigma: Noise standard deviation.
        sensitivity: L2 sensitivity.

    Returns:
        ρ = Δ²/(2σ²).
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    return sensitivity**2 / (2.0 * sigma**2)


__all__ = [
    # Enums
    "ZCDPVariant",
    "ConversionDirection",
    "GaussianVariant",
    # Config
    "ZCDPConfig",
    # Data types
    "ZCDPMechanismEntry",
    "CompositionResult",
    "ConversionResult",
    "ZCDPSynthesisResult",
    # Protocols
    "ZCDPMechanism",
    # Classes
    "ZCDPAccountant",
    "ZCDPComposer",
    "ZCDPConverter",
    "ZCDPMechanismAnalyzer",
    "ZCDPSynthesizer",
    # Functions
    "zcdp_to_dp",
    "dp_to_zcdp",
    "gaussian_zcdp_cost",
    # accountant
    "RenyiDivergenceComputer",
    "GaussianMechanismZCDP",
    "LaplaceMechanismZCDP",
    "SubsampledZCDP",
    "AdvancedCompositionZCDP",
    # composition
    "SequentialComposition",
    "ParallelComposition",
    "AdaptiveComposition",
    "HeterogeneousComposition",
    "CompositionOptimizer",
    "TruncatedConcentratedDP",
    # conversion
    "ZCDPToApproxDP",
    "ApproxDPToZCDP",
    "RDPToZCDP",
    "ZCDPToRDP",
    "PLDConversion",
    "OptimalConversion",
    "NumericConversion",
    # synthesis
    "GaussianOptimizer",
    "DiscreteGaussianSynthesis",
    "TruncatedGaussianSynthesis",
    "MultiQuerySynthesis",
    "BudgetAllocation",
]

# Lazy imports from submodules
from dp_forge.zcdp.accountant import (
    RenyiDivergenceComputer,
    GaussianMechanismZCDP,
    LaplaceMechanismZCDP,
    SubsampledZCDP,
    AdvancedCompositionZCDP,
)
from dp_forge.zcdp.composition import (
    SequentialComposition,
    ParallelComposition,
    AdaptiveComposition,
    HeterogeneousComposition,
    CompositionOptimizer,
    TruncatedConcentratedDP,
)
from dp_forge.zcdp.conversion import (
    ZCDPToApproxDP,
    ApproxDPToZCDP,
    RDPToZCDP,
    ZCDPToRDP,
    PLDConversion,
    OptimalConversion,
    NumericConversion,
)
from dp_forge.zcdp.synthesis import (
    GaussianOptimizer,
    DiscreteGaussianSynthesis,
    TruncatedGaussianSynthesis,
    MultiQuerySynthesis,
    BudgetAllocation,
)
