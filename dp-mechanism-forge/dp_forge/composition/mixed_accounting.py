"""
Mixed privacy definition accounting for DP-Forge.

Provides unified privacy accounting across multiple privacy definitions:
(ε,δ)-DP, RDP, zCDP, and f-DP. Converts between representations to enable
composition of mechanisms specified in different frameworks.

Key Features:
    - MixedAccountant: unified accounting across privacy definitions
    - Conversions: RDP ↔ PLD, zCDP ↔ PLD, f-DP ↔ PLD
    - Tightest conversion selection
    - Unified budget tracking
    - Integration with composition engine

Classes:
    MixedAccountant — Unified accountant for mixed privacy definitions

Functions:
    convert_rdp_to_pld    — Convert RDP curve to PLD
    convert_zcdp_to_pld   — Convert zCDP to PLD
    convert_fdp_to_pld    — Convert f-DP tradeoff to PLD
    convert_pld_to_rdp    — Convert PLD to RDP (approximate)
    select_tightest       — Select tightest conversion

References:
    - Mironov, I. (2017). Rényi differential privacy.
    - Bun, M., & Steinke, T. (2016). Concentrated differential privacy.
    - Dong, J., Roth, A., & Su, W. J. (2019). Gaussian differential privacy.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from dp_forge.exceptions import ConfigurationError
from dp_forge.composition.pld import PrivacyLossDistribution, compose, discretize

FloatArray = npt.NDArray[np.float64]


@dataclass
class PrivacyBudget:
    """
    Unified privacy budget representation.
    
    Supports multiple privacy definitions in a union-style: at least one
    of (ε,δ), rdp_curve, rho (zCDP), or pld must be set.
    
    Attributes:
        epsilon: (ε,δ)-DP epsilon (optional)
        delta: (ε,δ)-DP delta (optional)
        rdp_curve: RDP curve as callable alpha -> eps_alpha (optional)
        rho: zCDP rho parameter (optional)
        pld: Privacy loss distribution (optional)
        metadata: Optional metadata dict
    """
    epsilon: Optional[float] = None
    delta: Optional[float] = None
    rdp_curve: Optional[Callable[[float], float]] = None
    rho: Optional[float] = None
    pld: Optional[PrivacyLossDistribution] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate that at least one privacy definition is set."""
        if all(v is None for v in [self.epsilon, self.rdp_curve, self.rho, self.pld]):
            raise ConfigurationError(
                "At least one of epsilon, rdp_curve, rho, or pld must be set",
                parameter="PrivacyBudget"
            )
        
        if self.epsilon is not None:
            if self.epsilon < 0 or not math.isfinite(self.epsilon):
                raise ValueError(f"epsilon must be non-negative and finite, got {self.epsilon}")
            if self.delta is not None and not (0 <= self.delta < 1):
                raise ValueError(f"delta must be in [0, 1), got {self.delta}")
        
        if self.rho is not None and (self.rho < 0 or not math.isfinite(self.rho)):
            raise ValueError(f"rho must be non-negative and finite, got {self.rho}")


class MixedAccountant:
    """
    Unified privacy accountant for mixed privacy definitions.
    
    Tracks privacy across mechanisms specified in different frameworks
    ((ε,δ)-DP, RDP, zCDP, f-DP) by converting to a common representation
    (PLD) and composing.
    
    Attributes:
        conversion_method: Method for converting to common representation ('pld', 'rdp')
        budgets: List of added privacy budgets
        composed_pld: Composed privacy loss distribution (if using PLD method)
        
    Example::
    
        accountant = MixedAccountant(conversion_method='pld')
        
        # Add (ε,δ)-DP mechanism
        accountant.add_epsilon_delta(epsilon=1.0, delta=1e-5)
        
        # Add RDP mechanism
        accountant.add_rdp(rdp_curve=lambda alpha: alpha * 0.5)
        
        # Add zCDP mechanism
        accountant.add_zcdp(rho=0.25)
        
        # Get composed guarantee
        eps, delta = accountant.get_epsilon_delta(target_delta=1e-5)
    """
    
    def __init__(
        self,
        conversion_method: str = "pld",
        grid_size: int = 10000,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize mixed accountant.
        
        Args:
            conversion_method: Conversion method ('pld' or 'rdp')
            grid_size: Grid size for PLD discretization
            metadata: Optional metadata
        """
        if conversion_method not in ["pld", "rdp"]:
            raise ValueError(f"conversion_method must be 'pld' or 'rdp', got '{conversion_method}'")
        
        self.conversion_method = conversion_method
        self.grid_size = grid_size
        self.metadata = metadata or {}
        
        self.budgets: List[PrivacyBudget] = []
        self.composed_pld: Optional[PrivacyLossDistribution] = None
        
        if conversion_method == "rdp":
            from dp_forge.rdp import RDPAccountant
            self.rdp_accountant = RDPAccountant()
        else:
            self.rdp_accountant = None
    
    def add_epsilon_delta(self, epsilon: float, delta: float = 0.0, name: Optional[str] = None) -> None:
        """
        Add (ε,δ)-DP mechanism.
        
        Args:
            epsilon: Mechanism epsilon
            delta: Mechanism delta
            name: Optional name
        """
        budget = PrivacyBudget(epsilon=epsilon, delta=delta, metadata={"name": name})
        self.budgets.append(budget)
        
        if self.conversion_method == "pld":
            self._add_epsilon_delta_to_pld(epsilon, delta)
        else:
            self._add_epsilon_delta_to_rdp(epsilon, delta)
    
    def add_rdp(
        self,
        rdp_curve: Union[Callable[[float], float], Any],
        name: Optional[str] = None
    ) -> None:
        """
        Add RDP mechanism.
        
        Args:
            rdp_curve: RDP curve (callable or RDPCurve)
            name: Optional name
        """
        budget = PrivacyBudget(rdp_curve=rdp_curve, metadata={"name": name})
        self.budgets.append(budget)
        
        if self.conversion_method == "pld":
            pld = convert_rdp_to_pld(rdp_curve, grid_size=self.grid_size)
            self._compose_pld(pld)
        else:
            if self.rdp_accountant is not None:
                self.rdp_accountant.add_rdp_curve(rdp_curve)
    
    def add_zcdp(self, rho: float, name: Optional[str] = None) -> None:
        """
        Add zCDP mechanism.
        
        Args:
            rho: zCDP rho parameter
            name: Optional name
        """
        budget = PrivacyBudget(rho=rho, metadata={"name": name})
        self.budgets.append(budget)
        
        if self.conversion_method == "pld":
            pld = convert_zcdp_to_pld(rho, grid_size=self.grid_size)
            self._compose_pld(pld)
        else:
            from dp_forge.rdp.conversion import zcdp_to_rdp
            rdp_curve = zcdp_to_rdp(rho)
            if self.rdp_accountant is not None:
                self.rdp_accountant.add_rdp_curve(rdp_curve)
    
    def add_fdp(self, tradeoff_fn: Callable[[float], float], name: Optional[str] = None) -> None:
        """
        Add f-DP mechanism.
        
        Args:
            tradeoff_fn: f-DP tradeoff function
            name: Optional name
        """
        if self.conversion_method != "pld":
            raise ValueError("f-DP only supported with conversion_method='pld'")
        
        pld = convert_fdp_to_pld(tradeoff_fn, grid_size=self.grid_size)
        self._compose_pld(pld)
        
        budget = PrivacyBudget(pld=pld, metadata={"name": name})
        self.budgets.append(budget)
    
    def add_pld(self, pld: PrivacyLossDistribution, name: Optional[str] = None) -> None:
        """
        Add PLD directly.
        
        Args:
            pld: Privacy loss distribution
            name: Optional name
        """
        budget = PrivacyBudget(pld=pld, metadata={"name": name})
        self.budgets.append(budget)
        
        if self.conversion_method == "pld":
            self._compose_pld(pld)
        else:
            raise ValueError("Cannot add PLD to RDP accountant")
    
    def get_epsilon_delta(self, target_delta: float) -> Tuple[float, float]:
        """
        Get composed (ε,δ) guarantee.
        
        Args:
            target_delta: Target delta
            
        Returns:
            Tuple (epsilon, delta)
        """
        if self.conversion_method == "pld":
            if self.composed_pld is None:
                return 0.0, 0.0
            epsilon = self.composed_pld.to_epsilon_delta(target_delta)
            return epsilon, target_delta
        else:
            if self.rdp_accountant is None:
                return 0.0, 0.0
            budget = self.rdp_accountant.to_dp(target_delta)
            return budget.epsilon, target_delta
    
    def _add_epsilon_delta_to_pld(self, epsilon: float, delta: float) -> None:
        """Add (ε,δ) to PLD path (approximate via Gaussian)."""
        if delta == 0.0:
            pld = _pure_dp_to_pld(epsilon, self.grid_size)
        else:
            pld = _approx_dp_to_pld(epsilon, delta, self.grid_size)
        
        self._compose_pld(pld)
    
    def _add_epsilon_delta_to_rdp(self, epsilon: float, delta: float) -> None:
        """Add (ε,δ) to RDP path."""
        from dp_forge.rdp.conversion import dp_to_rdp_bound
        
        rdp_curve = dp_to_rdp_bound(epsilon, delta)
        if self.rdp_accountant is not None:
            self.rdp_accountant.add_rdp_curve(rdp_curve)
    
    def _compose_pld(self, pld: PrivacyLossDistribution) -> None:
        """Compose PLD into accumulated PLD."""
        if self.composed_pld is None:
            self.composed_pld = pld
        else:
            self.composed_pld = compose(self.composed_pld, pld)
    
    def reset(self) -> None:
        """Reset accountant state."""
        self.budgets.clear()
        self.composed_pld = None
        if self.rdp_accountant is not None:
            self.rdp_accountant = None
            from dp_forge.rdp import RDPAccountant
            self.rdp_accountant = RDPAccountant()


def convert_rdp_to_pld(
    rdp_curve: Union[Callable[[float], float], Any],
    grid_size: int = 10000,
    alpha_min: float = 1.01,
    alpha_max: float = 100.0,
    num_alphas: int = 100
) -> PrivacyLossDistribution:
    """
    Convert RDP curve to PLD approximation.
    
    Constructs a PLD that approximates the privacy loss distribution
    implied by an RDP curve. Uses the relationship between RDP and
    privacy loss moments.
    
    Args:
        rdp_curve: RDP curve (callable alpha -> eps_alpha)
        grid_size: PLD grid size
        alpha_min: Minimum alpha for approximation
        alpha_max: Maximum alpha for approximation
        num_alphas: Number of alpha values to sample
        
    Returns:
        Approximating PLD
        
    Notes:
        - Approximation via moment matching
        - May be loose for highly non-Gaussian privacy losses
    """
    alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), num_alphas)
    
    if callable(rdp_curve):
        eps_vals = np.array([rdp_curve(alpha) for alpha in alphas])
    else:
        from dp_forge.rdp import RDPCurve
        if isinstance(rdp_curve, RDPCurve):
            eps_vals = np.interp(alphas, rdp_curve.alphas, rdp_curve.epsilons)
        else:
            raise TypeError(f"rdp_curve must be callable or RDPCurve, got {type(rdp_curve)}")
    
    mean_loss = np.mean(eps_vals)
    var_loss = np.var(eps_vals)
    
    grid_min = mean_loss - 4 * np.sqrt(var_loss + 1e-10)
    grid_max = mean_loss + 4 * np.sqrt(var_loss + 1e-10)
    
    def gaussian_pdf(x: float) -> float:
        return (1.0 / np.sqrt(2 * np.pi * var_loss)) * np.exp(-0.5 * (x - mean_loss)**2 / var_loss)
    
    pld = discretize(
        continuous_pld=gaussian_pdf,
        grid_size=grid_size,
        grid_min=grid_min,
        grid_max=grid_max,
        pessimistic=True,
        metadata={"source": "rdp_curve"}
    )
    
    return pld


def convert_zcdp_to_pld(
    rho: float,
    grid_size: int = 10000
) -> PrivacyLossDistribution:
    """
    Convert zCDP to PLD.
    
    Zero-concentrated DP with parameter rho corresponds to a Gaussian
    privacy loss distribution with variance 2*rho.
    
    Args:
        rho: zCDP parameter
        grid_size: PLD grid size
        
    Returns:
        Gaussian PLD with variance 2*rho
        
    Notes:
        - Exact conversion (zCDP defines Gaussian privacy loss)
        - Privacy loss ~ N(0, 2*rho)
    """
    if rho < 0:
        raise ValueError(f"rho must be non-negative, got {rho}")
    
    var = 2.0 * rho
    std = np.sqrt(var)
    
    grid_min = -4 * std
    grid_max = 4 * std
    
    def gaussian_pdf(x: float) -> float:
        if var == 0:
            return 1.0 if abs(x) < 1e-10 else 0.0
        return (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * x**2 / var)
    
    pld = discretize(
        continuous_pld=gaussian_pdf,
        grid_size=grid_size,
        grid_min=grid_min,
        grid_max=grid_max,
        pessimistic=True,
        metadata={"source": "zcdp", "rho": rho}
    )
    
    return pld


def convert_fdp_to_pld(
    tradeoff_fn: Callable[[float], float],
    grid_size: int = 10000,
    eps_max: float = 10.0
) -> PrivacyLossDistribution:
    """
    Convert f-DP tradeoff function to PLD approximation.
    
    Constructs a PLD from an f-DP tradeoff function via numerical
    inversion. The tradeoff function maps epsilon -> delta.
    
    Args:
        tradeoff_fn: f-DP tradeoff function eps -> delta
        grid_size: PLD grid size
        eps_max: Maximum epsilon to consider
        
    Returns:
        Approximating PLD
        
    Notes:
        - Approximation via finite difference derivative
        - May be inaccurate for non-smooth tradeoff functions
    """
    eps_vals = np.linspace(0, eps_max, 1000)
    delta_vals = np.array([tradeoff_fn(eps) for eps in eps_vals])
    
    pmf = -np.diff(delta_vals, prepend=0.0)
    pmf = np.maximum(pmf, 0.0)
    
    if np.sum(pmf) > 0:
        pmf /= np.sum(pmf)
    
    epsilon = 1e-100
    log_masses = np.log(pmf + epsilon)
    log_masses[pmf < epsilon] = -np.inf
    
    valid_indices = np.where(pmf >= epsilon)[0]
    if len(valid_indices) == 0:
        return PrivacyLossDistribution(
            log_masses=np.array([0.0]),
            grid_min=0.0,
            grid_max=0.0,
            grid_size=1,
            metadata={"source": "fdp"}
        )
    
    start_idx = valid_indices[0]
    end_idx = valid_indices[-1]
    
    log_masses_trimmed = log_masses[start_idx:end_idx + 1]
    grid_min = eps_vals[start_idx]
    grid_max = eps_vals[end_idx]
    
    if len(log_masses_trimmed) < grid_size:
        log_masses_interp = np.interp(
            np.linspace(0, len(log_masses_trimmed) - 1, grid_size),
            np.arange(len(log_masses_trimmed)),
            log_masses_trimmed
        )
        pld = PrivacyLossDistribution(
            log_masses=log_masses_interp,
            grid_min=grid_min,
            grid_max=grid_max,
            grid_size=grid_size,
            metadata={"source": "fdp"}
        )
    else:
        pld = PrivacyLossDistribution(
            log_masses=log_masses_trimmed,
            grid_min=grid_min,
            grid_max=grid_max,
            grid_size=len(log_masses_trimmed),
            metadata={"source": "fdp"}
        )
    
    return pld


def convert_pld_to_rdp(pld: PrivacyLossDistribution, alphas: Optional[FloatArray] = None) -> Callable[[float], float]:
    """
    Convert PLD to RDP curve (approximate).
    
    Computes RDP guarantees from a PLD by evaluating Rényi divergence
    of order alpha via moment generating function.
    
    Args:
        pld: Privacy loss distribution
        alphas: Alpha values to evaluate at (optional)
        
    Returns:
        RDP curve as callable alpha -> eps_alpha
        
    Notes:
        - Approximation via numerical integration
        - May underestimate true RDP for non-discretized losses
    """
    if alphas is None:
        alphas = np.logspace(np.log10(1.01), np.log10(100.0), 100)
    
    masses = np.exp(pld.log_masses)
    losses = pld.grid_values
    
    def rdp_epsilon(alpha: float) -> float:
        if alpha <= 1.0:
            return 0.0
        
        moment = np.sum(masses * np.exp((alpha - 1) * losses))
        eps = (1.0 / (alpha - 1)) * np.log(moment + 1e-100)
        
        return max(0.0, eps)
    
    return rdp_epsilon


def _pure_dp_to_pld(epsilon: float, grid_size: int) -> PrivacyLossDistribution:
    """Convert pure (ε,0)-DP to PLD (spike at epsilon)."""
    log_masses = np.full(grid_size, -np.inf)
    log_masses[grid_size // 2] = 0.0
    
    return PrivacyLossDistribution(
        log_masses=log_masses,
        grid_min=epsilon - 0.1,
        grid_max=epsilon + 0.1,
        grid_size=grid_size,
        metadata={"source": "pure_dp", "epsilon": epsilon}
    )


def _approx_dp_to_pld(epsilon: float, delta: float, grid_size: int) -> PrivacyLossDistribution:
    """Convert approximate (ε,δ)-DP to PLD via Gaussian approximation."""
    if delta <= 0:
        return _pure_dp_to_pld(epsilon, grid_size)
    
    sigma_sq = 2.0 * np.log(1.0 / delta) / epsilon**2 if epsilon > 0 else 1.0
    std = np.sqrt(sigma_sq)
    
    grid_min = epsilon - 4 * std
    grid_max = epsilon + 4 * std
    
    def gaussian_pdf(x: float) -> float:
        return (1.0 / np.sqrt(2 * np.pi * sigma_sq)) * np.exp(-0.5 * (x - epsilon)**2 / sigma_sq)
    
    return discretize(
        continuous_pld=gaussian_pdf,
        grid_size=grid_size,
        grid_min=grid_min,
        grid_max=grid_max,
        pessimistic=True,
        metadata={"source": "approx_dp", "epsilon": epsilon, "delta": delta}
    )


def select_tightest_conversion(
    rdp_curve: Optional[Callable[[float], float]] = None,
    zcdp_rho: Optional[float] = None,
    pld: Optional[PrivacyLossDistribution] = None,
    target_delta: float = 1e-5
) -> Tuple[float, str]:
    """
    Select tightest epsilon among multiple privacy definitions.
    
    Converts all provided privacy definitions to (ε,δ) and returns
    the tightest (smallest) epsilon.
    
    Args:
        rdp_curve: Optional RDP curve
        zcdp_rho: Optional zCDP parameter
        pld: Optional PLD
        target_delta: Target delta for conversion
        
    Returns:
        Tuple (tightest_epsilon, source)
        - tightest_epsilon: Smallest epsilon across conversions
        - source: Which definition gave tightest bound ('rdp', 'zcdp', 'pld')
    """
    candidates = []
    
    if rdp_curve is not None:
        from dp_forge.rdp.conversion import rdp_to_dp
        from dp_forge.rdp import RDPCurve
        
        if callable(rdp_curve) and not isinstance(rdp_curve, RDPCurve):
            alphas = np.logspace(np.log10(1.01), np.log10(100.0), 100)
            eps_vals = [rdp_curve(a) for a in alphas]
            rdp_curve = RDPCurve(alphas=alphas, epsilons=np.array(eps_vals))
        
        budget = rdp_to_dp(rdp_curve, target_delta)
        candidates.append((budget.epsilon, 'rdp'))
    
    if zcdp_rho is not None:
        eps = zcdp_rho + 2 * np.sqrt(zcdp_rho * np.log(1.0 / target_delta))
        candidates.append((eps, 'zcdp'))
    
    if pld is not None:
        eps = pld.to_epsilon_delta(target_delta)
        candidates.append((eps, 'pld'))
    
    if len(candidates) == 0:
        return 0.0, 'none'
    
    return min(candidates, key=lambda x: x[0])
