"""
usability_oracle.sensitivity.types — Parametric sensitivity analysis types.

Value types for Sobol' indices, Morris screening, and local sensitivity
analysis of cognitive model parameters (β, Fitts a/b, Hick a/b, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from usability_oracle.core.types import Interval


# ═══════════════════════════════════════════════════════════════════════════
# ParameterRange — specification of a parameter's analysis domain
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ParameterRange:
    """Domain specification for a single model parameter.

    Attributes
    ----------
    name : str
        Parameter name (e.g. ``"fitts_b"``, ``"beta"``).
    interval : Interval
        Closed range [low, high] for the parameter.
    nominal : float
        Nominal (baseline) value for local sensitivity.
    distribution : str
        Distribution assumption: ``"uniform"``, ``"normal"``, ``"lognormal"``.
    description : str
        Human-readable description of the parameter.
    """

    name: str
    interval: Interval
    nominal: float = 0.0
    distribution: str = "uniform"
    description: str = ""

    @property
    def range_width(self) -> float:
        return self.interval.width

    @property
    def relative_range(self) -> float:
        """Range width relative to nominal value (inf if nominal = 0)."""
        if self.nominal == 0.0:
            return float("inf")
        return self.interval.width / abs(self.nominal)


# ═══════════════════════════════════════════════════════════════════════════
# SensitivityConfig — configuration for a sensitivity analysis run
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SensitivityConfig:
    """Configuration for a parametric sensitivity analysis.

    Attributes
    ----------
    parameters : tuple[ParameterRange, ...]
        Parameters to vary.
    n_samples : int
        Number of samples (Sobol') or trajectories (Morris).
    method : str
        Analysis method: ``"sobol"``, ``"morris"``, ``"local"``, ``"oat"``.
    output_names : tuple[str, ...]
        Names of the output quantities of interest.
    confidence_level : float
        Confidence level for bootstrap CIs (e.g. 0.95).
    seed : int
        RNG seed for reproducibility.
    """

    parameters: Tuple[ParameterRange, ...] = ()
    n_samples: int = 1024
    method: str = "sobol"
    output_names: Tuple[str, ...] = ("task_time_s",)
    confidence_level: float = 0.95
    seed: int = 42

    @property
    def n_parameters(self) -> int:
        return len(self.parameters)

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        return tuple(p.name for p in self.parameters)


# ═══════════════════════════════════════════════════════════════════════════
# SobolIndices — first-order and total Sobol' sensitivity indices
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SobolIndices:
    """Sobol' variance-based sensitivity indices.

    Attributes
    ----------
    parameter_name : str
        Name of the parameter.
    first_order : float
        First-order index Sᵢ = Var[E[Y|Xᵢ]] / Var[Y].
    total_order : float
        Total-order index ST_i = 1 - Var[E[Y|X~i]] / Var[Y].
    first_order_ci : Interval
        Confidence interval for Sᵢ.
    total_order_ci : Interval
        Confidence interval for ST_i.
    second_order : Mapping[str, float]
        Second-order indices Sᵢⱼ with other parameters (parameter name → value).
    """

    parameter_name: str
    first_order: float
    total_order: float
    first_order_ci: Interval = field(default_factory=lambda: Interval(0.0, 1.0))
    total_order_ci: Interval = field(default_factory=lambda: Interval(0.0, 1.0))
    second_order: Mapping[str, float] = field(default_factory=dict)

    @property
    def interaction_index(self) -> float:
        """Interaction index = ST_i - S_i (contribution from interactions)."""
        return self.total_order - self.first_order

    @property
    def is_influential(self) -> bool:
        """Heuristic: parameter is influential if total-order > 0.05."""
        return self.total_order > 0.05


# ═══════════════════════════════════════════════════════════════════════════
# MorrisResult — Morris elementary effects screening result
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class MorrisResult:
    """Morris method (elementary effects) screening result for one parameter.

    Attributes
    ----------
    parameter_name : str
        Name of the parameter.
    mu_star : float
        μ* — mean of absolute elementary effects (overall influence).
    mu : float
        μ — mean of elementary effects (direction of influence).
    sigma : float
        σ — standard deviation of elementary effects (non-linearity / interaction).
    n_trajectories : int
        Number of Morris trajectories evaluated.
    elementary_effects : tuple[float, ...]
        Individual elementary effects for each trajectory.
    """

    parameter_name: str
    mu_star: float
    mu: float
    sigma: float
    n_trajectories: int = 0
    elementary_effects: Tuple[float, ...] = ()

    @property
    def sigma_over_mu_star(self) -> float:
        """σ/μ* ratio — high values indicate non-linear or interaction effects."""
        if self.mu_star == 0:
            return float("inf")
        return self.sigma / self.mu_star

    @property
    def is_non_monotonic(self) -> bool:
        """Heuristic: parameter has non-monotonic effect if σ/μ* > 1."""
        return self.sigma_over_mu_star > 1.0

    @property
    def is_influential(self) -> bool:
        """Heuristic: parameter is influential if μ* > 0.1 * max(μ* across params)."""
        return self.mu_star > 0.0


# ═══════════════════════════════════════════════════════════════════════════
# SensitivityResult — aggregate sensitivity analysis output
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SensitivityResult:
    """Aggregate result of a sensitivity analysis run.

    Attributes
    ----------
    config : SensitivityConfig
        Configuration used for this analysis.
    output_name : str
        Name of the output quantity analysed.
    sobol_indices : tuple[SobolIndices, ...]
        Sobol' indices for each parameter (empty if method ≠ "sobol").
    morris_results : tuple[MorrisResult, ...]
        Morris results for each parameter (empty if method ≠ "morris").
    total_variance : float
        Total variance of the output quantity.
    mean_output : float
        Mean value of the output quantity.
    n_evaluations : int
        Total number of model evaluations performed.
    metadata : Mapping[str, Any]
        Additional analysis metadata.
    """

    config: SensitivityConfig
    output_name: str = ""
    sobol_indices: Tuple[SobolIndices, ...] = ()
    morris_results: Tuple[MorrisResult, ...] = ()
    total_variance: float = 0.0
    mean_output: float = 0.0
    n_evaluations: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def most_influential_parameter(self) -> Optional[str]:
        """Name of the most influential parameter (by total-order or μ*)."""
        if self.sobol_indices:
            best = max(self.sobol_indices, key=lambda s: s.total_order)
            return best.parameter_name
        if self.morris_results:
            best_m = max(self.morris_results, key=lambda m: m.mu_star)
            return best_m.parameter_name
        return None

    @property
    def influential_parameters(self) -> Tuple[str, ...]:
        """Parameters classified as influential."""
        if self.sobol_indices:
            return tuple(s.parameter_name for s in self.sobol_indices if s.is_influential)
        if self.morris_results:
            return tuple(m.parameter_name for m in self.morris_results if m.is_influential)
        return ()

    @property
    def coefficient_of_variation(self) -> float:
        """CV = sqrt(Var) / mean."""
        import math
        if self.mean_output == 0:
            return float("inf")
        return math.sqrt(self.total_variance) / abs(self.mean_output)


__all__ = [
    "MorrisResult",
    "ParameterRange",
    "SensitivityConfig",
    "SensitivityResult",
    "SobolIndices",
]
