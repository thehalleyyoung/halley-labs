"""
usability_oracle.differential.types — Differential privacy for usability data.

Value types for (ε, δ)-differential privacy mechanisms, privacy accounting
(composition theorems), and noise calibration for usability metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# Enumerations
# ═══════════════════════════════════════════════════════════════════════════

@unique
class MechanismType(Enum):
    """Standard differential-privacy noise mechanisms."""

    LAPLACE = "laplace"
    """Laplace mechanism for (ε, 0)-DP."""

    GAUSSIAN = "gaussian"
    """Gaussian mechanism for (ε, δ)-DP."""

    EXPONENTIAL = "exponential"
    """Exponential mechanism for discrete outputs."""

    RANDOMIZED_RESPONSE = "randomized_response"
    """Randomised response for binary/categorical data."""

    DISCRETE_LAPLACE = "discrete_laplace"
    """Discrete Laplace for integer-valued queries."""

    def __str__(self) -> str:
        return self.value


@unique
class CompositionTheorem(Enum):
    """Composition theorems for privacy accounting."""

    BASIC = "basic"
    """Basic (linear) composition: εₜₒₜ = Σ εᵢ."""

    ADVANCED = "advanced"
    """Advanced composition: εₜₒₜ = √(2k ln(1/δ')) · ε + k·ε·(e^ε - 1)."""

    RENYI = "renyi"
    """Rényi DP composition (optimal for Gaussian mechanisms)."""

    ZERO_CONCENTRATED = "zcdp"
    """Zero-concentrated DP composition."""

    MOMENTS_ACCOUNTANT = "moments_accountant"
    """Moments accountant (Abadi et al. 2016)."""

    def __str__(self) -> str:
        return self.value


# ═══════════════════════════════════════════════════════════════════════════
# PrivacyBudget — (ε, δ) privacy parameters
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class PrivacyBudget:
    """(ε, δ)-differential privacy budget.

    Attributes
    ----------
    epsilon : float
        Privacy loss parameter ε ≥ 0.  Smaller ⟹ more private.
    delta : float
        Probability of pure-DP failure δ ∈ [0, 1).  δ = 0 for pure DP.
    description : str
        Human-readable description of this budget allocation.
    """

    epsilon: float
    delta: float = 0.0
    description: str = ""

    def __post_init__(self) -> None:
        if self.epsilon < 0:
            raise ValueError(f"epsilon must be >= 0, got {self.epsilon}")
        if not (0 <= self.delta < 1):
            raise ValueError(f"delta must be in [0, 1), got {self.delta}")

    @property
    def is_pure_dp(self) -> bool:
        """True if δ = 0 (pure ε-differential privacy)."""
        return self.delta == 0.0

    def compose_basic(self, other: PrivacyBudget) -> PrivacyBudget:
        """Basic (sequential) composition."""
        return PrivacyBudget(
            epsilon=self.epsilon + other.epsilon,
            delta=self.delta + other.delta,
            description=f"basic({self.description}, {other.description})",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"epsilon": self.epsilon, "delta": self.delta, "description": self.description}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PrivacyBudget:
        return cls(
            epsilon=float(d["epsilon"]),
            delta=float(d.get("delta", 0.0)),
            description=d.get("description", ""),
        )


# ═══════════════════════════════════════════════════════════════════════════
# NoiseConfig — noise calibration parameters
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class NoiseConfig:
    """Calibrated noise parameters for a differential-privacy mechanism.

    Attributes
    ----------
    mechanism : MechanismType
        Which noise mechanism to use.
    scale : float
        Noise scale parameter (b for Laplace, σ for Gaussian).
    sensitivity : float
        Global sensitivity Δf of the query function.
    budget : PrivacyBudget
        Privacy budget this noise calibration satisfies.
    clipping_bound : Optional[float]
        Clipping bound applied to per-record contributions.
    """

    mechanism: MechanismType
    scale: float
    sensitivity: float
    budget: PrivacyBudget
    clipping_bound: Optional[float] = None

    @property
    def signal_to_noise(self) -> float:
        """Expected signal-to-noise ratio (sensitivity / scale)."""
        if self.scale == 0:
            return float("inf")
        return self.sensitivity / self.scale


# ═══════════════════════════════════════════════════════════════════════════
# PrivacyMechanism (dataclass) — description of a noise mechanism instance
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class PrivacyMechanismSpec:
    """Specification of a concrete privacy mechanism instance.

    Attributes
    ----------
    name : str
        Human-readable mechanism name.
    mechanism_type : MechanismType
        The underlying noise mechanism.
    noise_config : NoiseConfig
        Calibrated noise parameters.
    input_domain : str
        Description of the input domain (e.g. "task_completion_time_s").
    output_domain : str
        Description of the output domain.
    """

    name: str
    mechanism_type: MechanismType
    noise_config: NoiseConfig
    input_domain: str = ""
    output_domain: str = ""

    @property
    def budget(self) -> PrivacyBudget:
        return self.noise_config.budget


# ═══════════════════════════════════════════════════════════════════════════
# CompositionResult — result of composing multiple mechanisms
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class CompositionResult:
    """Result of composing multiple privacy mechanisms.

    Attributes
    ----------
    total_budget : PrivacyBudget
        Composed (ε, δ) budget for the entire pipeline.
    theorem_used : CompositionTheorem
        Which composition theorem was applied.
    n_mechanisms : int
        Number of mechanisms composed.
    per_mechanism_budgets : tuple[PrivacyBudget, ...]
        Individual budgets that were composed.
    tightness_gap : float
        Estimated gap between the bound and the true privacy loss.
        Lower is tighter.
    metadata : Mapping[str, Any]
        Additional composition metadata (e.g. Rényi order α).
    """

    total_budget: PrivacyBudget
    theorem_used: CompositionTheorem = CompositionTheorem.BASIC
    n_mechanisms: int = 0
    per_mechanism_budgets: Tuple[PrivacyBudget, ...] = ()
    tightness_gap: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def total_epsilon(self) -> float:
        return self.total_budget.epsilon

    @property
    def total_delta(self) -> float:
        return self.total_budget.delta

    @property
    def average_epsilon_per_mechanism(self) -> float:
        """Average per-mechanism epsilon."""
        if self.n_mechanisms == 0:
            return 0.0
        return self.total_budget.epsilon / self.n_mechanisms


# ═══════════════════════════════════════════════════════════════════════════
# PrivacyGuarantee — attestation of privacy properties
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class PrivacyGuarantee:
    """Attestation of the privacy guarantee for a data release.

    Attributes
    ----------
    budget : PrivacyBudget
        The (ε, δ) guarantee.
    composition : Optional[CompositionResult]
        Composition details if multiple mechanisms contributed.
    query_description : str
        What was queried (e.g. "mean_task_time across 50 users").
    n_records : int
        Number of individual records protected.
    mechanism_names : tuple[str, ...]
        Names of the mechanisms applied.
    timestamp : str
        ISO 8601 timestamp of the guarantee.
    """

    budget: PrivacyBudget
    composition: Optional[CompositionResult] = None
    query_description: str = ""
    n_records: int = 0
    mechanism_names: Tuple[str, ...] = ()
    timestamp: str = ""

    @property
    def is_composed(self) -> bool:
        return self.composition is not None

    @property
    def epsilon(self) -> float:
        return self.budget.epsilon

    @property
    def delta(self) -> float:
        return self.budget.delta


__all__ = [
    "CompositionResult",
    "CompositionTheorem",
    "MechanismType",
    "NoiseConfig",
    "PrivacyBudget",
    "PrivacyGuarantee",
    "PrivacyMechanismSpec",
]
