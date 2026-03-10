"""
usability_oracle.variational.types — Data types for free-energy variational inference.

Provides immutable value types used by the variational solver that computes
bounded-rational policies by minimising the *variational free energy*:

    F[π] = E_π[C(τ)] − (1/β) H[π]

where C(τ) is trajectory cost, H[π] is policy entropy, and β is the
rationality parameter (inverse temperature).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, List, NewType, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# NewType aliases
# ---------------------------------------------------------------------------

ParameterIndex = NewType("ParameterIndex", int)
"""Index into a variational parameter vector."""

IterationCount = NewType("IterationCount", int)
"""Number of optimisation iterations completed."""


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

@unique
class ConvergenceStatus(Enum):
    """Outcome of a variational optimisation run."""

    CONVERGED = "converged"
    """Objective change fell below the tolerance threshold."""

    MAX_ITERATIONS = "max_iterations"
    """Iteration budget exhausted before convergence."""

    DIVERGED = "diverged"
    """Objective increased or produced NaN/Inf values."""

    SADDLE_POINT = "saddle_point"
    """Optimiser stalled at a saddle point (Hessian indefinite)."""


@unique
class ObjectiveType(Enum):
    """Which free-energy decomposition to optimise."""

    VARIATIONAL_FREE_ENERGY = "variational_free_energy"
    """F = E[C] − (1/β) H  (standard formulation)."""

    ELBO = "elbo"
    """Evidence lower bound (negated free energy)."""

    KL_COST_TRADEOFF = "kl_cost_tradeoff"
    """Explicit KL(π ‖ π₀) + β E_π[C] formulation."""

    RATE_DISTORTION = "rate_distortion"
    """Rate–distortion: min I(S;A) s.t. E[C] ≤ D."""


# ═══════════════════════════════════════════════════════════════════════════
# VariationalConfig
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class VariationalConfig:
    """Configuration for the free-energy variational solver.

    Attributes:
        beta: Rationality parameter (inverse temperature).  β → ∞ recovers
            the fully rational (optimal) policy; β → 0 yields the maximum-
            entropy (uniform) policy.
        max_iterations: Upper bound on solver iterations.
        tolerance: Convergence tolerance on the relative change in the
            objective:  |F_{k} − F_{k−1}| / |F_{k−1}| < tolerance.
        learning_rate: Step size for gradient-based updates.
        objective: Which free-energy decomposition to optimise.
        use_natural_gradient: If ``True``, precondition gradients with the
            Fisher information matrix for faster convergence on the
            probability simplex.
        line_search: Enable Armijo backtracking line search.
        regularisation_lambda: L2 regularisation strength on the variational
            parameters (prevents over-fitting to small trajectory samples).
        num_inner_iterations: Number of inner fixed-point iterations for
            the Blahut–Arimoto-style alternating projection.
        seed: Random seed for reproducibility.
    """

    beta: float = 1.0
    max_iterations: int = 500
    tolerance: float = 1e-8
    learning_rate: float = 0.01
    objective: ObjectiveType = ObjectiveType.VARIATIONAL_FREE_ENERGY
    use_natural_gradient: bool = False
    line_search: bool = True
    regularisation_lambda: float = 0.0
    num_inner_iterations: int = 10
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.beta < 0.0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "beta": self.beta,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "learning_rate": self.learning_rate,
            "objective": self.objective.value,
            "use_natural_gradient": self.use_natural_gradient,
            "line_search": self.line_search,
            "regularisation_lambda": self.regularisation_lambda,
            "num_inner_iterations": self.num_inner_iterations,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> VariationalConfig:
        return cls(
            beta=float(d["beta"]),
            max_iterations=int(d["max_iterations"]),
            tolerance=float(d["tolerance"]),
            learning_rate=float(d["learning_rate"]),
            objective=ObjectiveType(d.get("objective", "variational_free_energy")),
            use_natural_gradient=bool(d.get("use_natural_gradient", False)),
            line_search=bool(d.get("line_search", True)),
            regularisation_lambda=float(d.get("regularisation_lambda", 0.0)),
            num_inner_iterations=int(d.get("num_inner_iterations", 10)),
            seed=d.get("seed"),
        )


# ═══════════════════════════════════════════════════════════════════════════
# ConvergenceInfo
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ConvergenceInfo:
    """Diagnostic information about solver convergence.

    Records the full optimisation trace so that downstream components can
    inspect convergence behaviour and detect pathologies.

    Attributes:
        status: Terminal convergence status.
        iterations_used: Number of iterations actually executed.
        objective_trace: Sequence of objective values, one per iteration.
        gradient_norm_trace: Sequence of gradient L2 norms.
        relative_change: Final |F_k − F_{k−1}| / |F_{k−1}|.
        wall_clock_seconds: Elapsed wall-clock time.
    """

    status: ConvergenceStatus
    iterations_used: int
    objective_trace: Tuple[float, ...]
    gradient_norm_trace: Tuple[float, ...]
    relative_change: float
    wall_clock_seconds: float

    @property
    def converged(self) -> bool:
        """Whether the solver reached the convergence criterion."""
        return self.status == ConvergenceStatus.CONVERGED

    @property
    def final_objective(self) -> float:
        """Last recorded objective value."""
        return self.objective_trace[-1] if self.objective_trace else math.nan

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "iterations_used": self.iterations_used,
            "objective_trace": list(self.objective_trace),
            "gradient_norm_trace": list(self.gradient_norm_trace),
            "relative_change": self.relative_change,
            "wall_clock_seconds": self.wall_clock_seconds,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ConvergenceInfo:
        return cls(
            status=ConvergenceStatus(d["status"]),
            iterations_used=int(d["iterations_used"]),
            objective_trace=tuple(d["objective_trace"]),
            gradient_norm_trace=tuple(d["gradient_norm_trace"]),
            relative_change=float(d["relative_change"]),
            wall_clock_seconds=float(d["wall_clock_seconds"]),
        )


# ═══════════════════════════════════════════════════════════════════════════
# KLDivergenceResult
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class KLDivergenceResult:
    """Result of a KL-divergence computation between two policies.

    Stores  D_KL(π ‖ π₀) = Σ_s d_π(s) Σ_a π(a|s) log(π(a|s)/π₀(a|s))
    together with per-state breakdowns for diagnostic purposes.

    Attributes:
        total_kl: Aggregate KL divergence (nats).
        per_state_kl: Mapping from state identifier to its KL contribution.
        max_state_kl: Maximum per-state KL divergence.
        max_state_id: Identifier of the state with the largest KL.
        is_finite: ``True`` iff the KL is finite (no zero-support issues).
    """

    total_kl: float
    per_state_kl: Dict[str, float]
    max_state_kl: float
    max_state_id: str
    is_finite: bool

    def to_bits(self) -> float:
        """Convert total KL from nats to bits (÷ ln 2)."""
        return self.total_kl / math.log(2.0) if self.is_finite else math.inf

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_kl": self.total_kl,
            "per_state_kl": dict(self.per_state_kl),
            "max_state_kl": self.max_state_kl,
            "max_state_id": self.max_state_id,
            "is_finite": self.is_finite,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> KLDivergenceResult:
        return cls(
            total_kl=float(d["total_kl"]),
            per_state_kl={k: float(v) for k, v in d["per_state_kl"].items()},
            max_state_kl=float(d["max_state_kl"]),
            max_state_id=str(d["max_state_id"]),
            is_finite=bool(d["is_finite"]),
        )


# ═══════════════════════════════════════════════════════════════════════════
# CapacityProfile
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class CapacityProfile:
    """Channel-capacity profile across rationality levels.

    At each β value, the agent's effective capacity (bits per decision)
    is computed from the mutual information I(S;A) under the corresponding
    bounded-rational policy.  This traces out the rate–distortion curve.

    Attributes:
        beta_values: Monotonically increasing sequence of β values tested.
        capacity_bits: Capacity in bits at each β level.
        expected_costs: Expected trajectory cost E_π[C] at each β.
        kl_nats: KL(π‖π₀) in nats at each β.
        pareto_front_indices: Indices of points on the Pareto front
            of the rate–distortion trade-off.
    """

    beta_values: Tuple[float, ...]
    capacity_bits: Tuple[float, ...]
    expected_costs: Tuple[float, ...]
    kl_nats: Tuple[float, ...]
    pareto_front_indices: Tuple[int, ...]

    def __post_init__(self) -> None:
        n = len(self.beta_values)
        if not (len(self.capacity_bits) == len(self.expected_costs)
                == len(self.kl_nats) == n):
            raise ValueError("All profile sequences must have the same length")

    @property
    def num_points(self) -> int:
        """Number of β values in the sweep."""
        return len(self.beta_values)

    def capacity_at(self, beta: float) -> float:
        """Linearly interpolate capacity at an arbitrary β."""
        return float(np.interp(beta, self.beta_values, self.capacity_bits))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "beta_values": list(self.beta_values),
            "capacity_bits": list(self.capacity_bits),
            "expected_costs": list(self.expected_costs),
            "kl_nats": list(self.kl_nats),
            "pareto_front_indices": list(self.pareto_front_indices),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CapacityProfile:
        return cls(
            beta_values=tuple(d["beta_values"]),
            capacity_bits=tuple(d["capacity_bits"]),
            expected_costs=tuple(d["expected_costs"]),
            kl_nats=tuple(d["kl_nats"]),
            pareto_front_indices=tuple(d["pareto_front_indices"]),
        )


# ═══════════════════════════════════════════════════════════════════════════
# FreeEnergyResult
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class FreeEnergyResult:
    """Full result of a variational free-energy minimisation.

    Contains the optimal policy distribution, the decomposed free-energy
    terms, and convergence diagnostics.

    Attributes:
        free_energy: Optimal value of F[π*] = E[C] − (1/β) H[π*].
        expected_cost: E_{π*}[C(τ)] — expected trajectory cost.
        entropy: H[π*] — Shannon entropy of the optimal policy (nats).
        kl_divergence: KL(π* ‖ π₀) between optimal and reference policy.
        policy: Mapping  state_id → {action_id → probability}.
        convergence: Solver convergence diagnostics.
        capacity_profile: Optional rate–distortion sweep.
        config: The configuration used for this solve.
    """

    free_energy: float
    expected_cost: float
    entropy: float
    kl_divergence: KLDivergenceResult
    policy: Dict[str, Dict[str, float]]
    convergence: ConvergenceInfo
    capacity_profile: Optional[CapacityProfile] = None
    config: Optional[VariationalConfig] = None

    @property
    def rationality_cost(self) -> float:
        """Cost of bounded rationality: E[C*] − E[C_optimal].

        This is the performance loss attributable to the agent's limited
        channel capacity.  Positive values indicate sub-optimality.
        """
        return self.expected_cost  # baseline optimal must be set externally

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "free_energy": self.free_energy,
            "expected_cost": self.expected_cost,
            "entropy": self.entropy,
            "kl_divergence": self.kl_divergence.to_dict(),
            "policy": self.policy,
            "convergence": self.convergence.to_dict(),
        }
        if self.capacity_profile is not None:
            d["capacity_profile"] = self.capacity_profile.to_dict()
        if self.config is not None:
            d["config"] = self.config.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> FreeEnergyResult:
        cap = d.get("capacity_profile")
        cfg = d.get("config")
        return cls(
            free_energy=float(d["free_energy"]),
            expected_cost=float(d["expected_cost"]),
            entropy=float(d["entropy"]),
            kl_divergence=KLDivergenceResult.from_dict(d["kl_divergence"]),
            policy=d["policy"],
            convergence=ConvergenceInfo.from_dict(d["convergence"]),
            capacity_profile=CapacityProfile.from_dict(cap) if cap else None,
            config=VariationalConfig.from_dict(cfg) if cfg else None,
        )
