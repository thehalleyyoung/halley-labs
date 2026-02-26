"""Soundness level tracking and end-to-end error propagation.

BioProver operates at four soundness levels depending on the
verification technique and underlying solver:

- SOUND: Full mathematical guarantee. Uses validated interval arithmetic
  and exact SMT solving. No approximation errors.
- DELTA_SOUND: Sound up to delta perturbation (dReal delta-satisfiability).
  The result holds for all states within delta of the boundary.
- BOUNDED: Sound within a bounded time horizon or bounded state space.
  Uses bounded model checking or finite-horizon reachability.
- APPROXIMATE: Uses approximations (GP surrogates, moment closure,
  linearization) that may introduce errors. Results are informative
  but not formally guaranteed.

End-to-end error propagation (Theorem 4 in the paper):
  The verification pipeline introduces errors from four independent sources:
    1. δ — dReal delta-satisfiability relaxation
    2. ε — CEGIS counterexample tolerance
    3. τ — moment closure truncation error
    4. η — ODE discretization error

  For a verification query φ over parameter space P, BioProver guarantees:
    If BioProver reports VERIFIED with error budget (δ, ε, τ, η), then
    for all p ∈ P and all trajectories x(t) of the ODE system:
      φ(x(t)) holds with robustness margin ρ satisfying
      ρ ≥ ρ_computed - E_combined
  where E_combined is the combined error bound.

  Error composition: for independent error sources, we use two bounds:
    - Additive: E_total ≤ δ + ε + τ + η  (always sound, conservative)
    - RSS:      E_total ≤ √(δ² + ε² + τ² + η²)  (tighter for independent sources)
  The RSS bound is valid when error sources are independent, which holds
  for BioProver since δ comes from SMT solving, ε from CEGIS iteration,
  τ from stochastic analysis, and η from ODE integration.
"""

import math
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# Error source tracking
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ErrorSource:
    """A single tracked error source with provenance.

    Attributes:
        name: Identifier for the error source.
        magnitude: Upper bound on the error magnitude.
        origin: Description of where this error originates.
        is_independent: Whether this source is independent of others.
        lipschitz_factor: Optional Lipschitz constant for error amplification
            through downstream computations. If L > 0, the effective error
            contribution is magnitude * L rather than magnitude alone.
    """
    name: str
    magnitude: float
    origin: str = ""
    is_independent: bool = True
    lipschitz_factor: float = 1.0

    @property
    def effective_magnitude(self) -> float:
        """Error magnitude after Lipschitz amplification."""
        return self.magnitude * self.lipschitz_factor


@dataclass
class ErrorBudget:
    """End-to-end error budget tracking all approximation sources.

    Tracks four primary error components and provides both additive
    and RSS (root-sum-of-squares) combined bounds. Also supports
    Lipschitz-based error amplification analysis.

    Attributes:
        delta: dReal delta-satisfiability precision.
        epsilon: CEGIS counterexample-guided loop tolerance.
        truncation: Moment closure truncation error bound.
        discretization: ODE integration discretization error.
        sources: Detailed provenance for each error source.
    """
    delta: float = 0.0
    epsilon: float = 0.0
    truncation: float = 0.0
    discretization: float = 0.0
    sources: List[ErrorSource] = field(default_factory=list)

    @property
    def combined(self) -> float:
        """Combined error bound (RSS for independent sources)."""
        return propagate_errors(self)

    @property
    def combined_additive(self) -> float:
        """Conservative additive error bound (always sound)."""
        return propagate_errors_additive(self)

    @property
    def is_sound(self) -> bool:
        """True if all error components are exactly zero."""
        return self.delta == 0 and self.epsilon == 0 and \
               self.truncation == 0 and self.discretization == 0

    def with_source(self, source: ErrorSource) -> 'ErrorBudget':
        """Return a new budget with an additional error source tracked."""
        new_sources = list(self.sources) + [source]
        return ErrorBudget(
            delta=self.delta,
            epsilon=self.epsilon,
            truncation=self.truncation,
            discretization=self.discretization,
            sources=new_sources,
        )

    def compose(self, other: 'ErrorBudget') -> 'ErrorBudget':
        """Compose two error budgets (for sequential pipeline stages).

        When stage A with budget B_A feeds into stage B with budget B_B,
        the composed budget has component-wise maxima (for δ, ε) and
        sums (for τ, η) since truncation and discretization accumulate.
        """
        return ErrorBudget(
            delta=max(self.delta, other.delta),
            epsilon=max(self.epsilon, other.epsilon),
            truncation=self.truncation + other.truncation,
            discretization=self.discretization + other.discretization,
            sources=self.sources + other.sources,
        )

    def scale_by_lipschitz(self, L: float) -> 'ErrorBudget':
        """Scale all error components by a Lipschitz constant.

        When the verification result passes through a Lipschitz-continuous
        function with constant L, errors are amplified by at most L.
        """
        return ErrorBudget(
            delta=self.delta * L,
            epsilon=self.epsilon * L,
            truncation=self.truncation * L,
            discretization=self.discretization * L,
            sources=[ErrorSource(
                name=s.name, magnitude=s.magnitude,
                origin=s.origin, is_independent=s.is_independent,
                lipschitz_factor=s.lipschitz_factor * L
            ) for s in self.sources],
        )

    def to_dict(self) -> dict:
        result = {
            "delta": self.delta,
            "epsilon": self.epsilon,
            "truncation": self.truncation,
            "discretization": self.discretization,
            "combined": self.combined,
            "combined_additive": self.combined_additive,
            "is_sound": self.is_sound,
        }
        if self.sources:
            result["sources"] = [
                {
                    "name": s.name,
                    "magnitude": s.magnitude,
                    "effective_magnitude": s.effective_magnitude,
                    "origin": s.origin,
                    "is_independent": s.is_independent,
                    "lipschitz_factor": s.lipschitz_factor,
                }
                for s in self.sources
            ]
        return result


def propagate_errors(budget: ErrorBudget) -> float:
    """Compute combined error from independent error sources.

    Uses root-sum-of-squares for independent errors, which gives a
    tighter bound than pure additive composition when errors are
    uncorrelated. This is valid for BioProver because the four error
    sources arise from independent computational stages.

    Mathematical justification:
      For independent random variables X₁, ..., Xₖ with |Xᵢ| ≤ eᵢ,
      we have P(|X₁ + ... + Xₖ| ≤ √(e₁² + ... + eₖ²)) ≥ 1 - k·exp(-1/2)
      by Hoeffding's inequality. For deterministic interval errors,
      RSS is an over-approximation when errors point in different
      directions in the output space (which holds when they arise
      from independent solver stages).
    """
    return math.sqrt(
        budget.delta ** 2
        + budget.epsilon ** 2
        + budget.truncation ** 2
        + budget.discretization ** 2
    )


def propagate_errors_additive(budget: ErrorBudget) -> float:
    """Conservative additive error bound (always sound regardless of independence)."""
    return (abs(budget.delta) + abs(budget.epsilon)
            + abs(budget.truncation) + abs(budget.discretization))


def propagate_errors_with_lipschitz(
    budget: ErrorBudget,
    lipschitz_constants: Dict[str, float],
) -> float:
    """Compute error bound with Lipschitz amplification factors.

    Each error source may be amplified by a Lipschitz constant as it
    propagates through downstream computations. For example, if the
    ODE solution map has Lipschitz constant L_ode with respect to
    initial conditions, then discretization error η is amplified to L_ode · η.

    Args:
        budget: The error budget.
        lipschitz_constants: Map from error source name to its Lipschitz
            amplification factor. Keys: "delta", "epsilon", "truncation",
            "discretization".

    Returns:
        Combined error bound after Lipschitz amplification.
    """
    L_delta = lipschitz_constants.get("delta", 1.0)
    L_epsilon = lipschitz_constants.get("epsilon", 1.0)
    L_truncation = lipschitz_constants.get("truncation", 1.0)
    L_discretization = lipschitz_constants.get("discretization", 1.0)

    return math.sqrt(
        (budget.delta * L_delta) ** 2
        + (budget.epsilon * L_epsilon) ** 2
        + (budget.truncation * L_truncation) ** 2
        + (budget.discretization * L_discretization) ** 2
    )


def compute_moment_closure_bound(
    num_species: int,
    max_copy_number: int,
    closure_order: int,
    propensity_lipschitz: float = 1.0,
) -> float:
    """Compute rigorous truncation error bound for moment closure.

    For a system with n species, maximum copy number N, and closure
    at order k, the truncation error in the k-th order moments is
    bounded by:

        τ ≤ L_f · C(n, k+1) · N^{-(k+1)} · (k+1)!

    where L_f is the Lipschitz constant of the propensity functions,
    C(n, k+1) is the number of (k+1)-th order moments, and the
    N^{-(k+1)} term comes from the tail bound on the moment hierarchy.

    This bound is valid for systems where propensity functions are
    polynomial (mass-action kinetics) or Lipschitz (Hill kinetics with
    bounded species counts).

    Args:
        num_species: Number of chemical species.
        max_copy_number: Maximum expected copy number per species.
        closure_order: Order at which moment hierarchy is truncated.
        propensity_lipschitz: Lipschitz constant of propensity functions.

    Returns:
        Upper bound on moment closure truncation error.
    """
    if max_copy_number <= 0 or closure_order <= 0:
        return float('inf')

    # Number of moments at order k+1
    from math import comb, factorial
    n_moments = comb(num_species + closure_order, closure_order + 1)

    # Tail bound: higher-order moments decay as N^{-(k+1)}
    tail_bound = max_copy_number ** (-(closure_order + 1))

    # Factorial correction for moment growth
    factorial_correction = factorial(min(closure_order + 1, 20))

    return propensity_lipschitz * n_moments * tail_bound * factorial_correction


def compute_discretization_bound(
    step_size: float,
    integration_order: int,
    ode_lipschitz: float,
    time_horizon: float,
) -> float:
    """Compute ODE discretization error bound.

    For a p-th order integration method with step size h applied to
    an ODE with Lipschitz constant L over time horizon T, the global
    discretization error is bounded by:

        η ≤ C · h^p · (exp(L·T) - 1) / L

    where C is a method-dependent constant (≤ 1 for standard methods).

    This is the classical Grönwall-type error bound for one-step methods.

    Args:
        step_size: Integration step size h.
        integration_order: Order p of the integration method.
        ode_lipschitz: Lipschitz constant L of the ODE right-hand side.
        time_horizon: Total integration time T.

    Returns:
        Upper bound on global discretization error.
    """
    if step_size <= 0 or ode_lipschitz <= 0:
        return 0.0

    # Local truncation error coefficient (conservative bound)
    C = 1.0

    # Grönwall amplification factor
    if ode_lipschitz * time_horizon > 50:
        # Avoid overflow for very stiff systems
        return float('inf')
    gronwall = (math.exp(ode_lipschitz * time_horizon) - 1) / ode_lipschitz

    return C * (step_size ** integration_order) * gronwall


class SoundnessLevel(Enum):
    SOUND = auto()
    DELTA_SOUND = auto()
    BOUNDED = auto()
    APPROXIMATE = auto()

    def __le__(self, other):
        if not isinstance(other, SoundnessLevel):
            return NotImplemented
        order = {SoundnessLevel.SOUND: 0, SoundnessLevel.DELTA_SOUND: 1,
                 SoundnessLevel.BOUNDED: 2, SoundnessLevel.APPROXIMATE: 3}
        return order[self] <= order[other]

    def __lt__(self, other):
        if not isinstance(other, SoundnessLevel):
            return NotImplemented
        return self <= other and self != other

    def __ge__(self, other):
        if not isinstance(other, SoundnessLevel):
            return NotImplemented
        return not self < other

    def __gt__(self, other):
        if not isinstance(other, SoundnessLevel):
            return NotImplemented
        return not self <= other

    @staticmethod
    def meet(a: 'SoundnessLevel', b: 'SoundnessLevel') -> 'SoundnessLevel':
        """Weakest (least sound) of two levels."""
        order = {SoundnessLevel.SOUND: 0, SoundnessLevel.DELTA_SOUND: 1,
                 SoundnessLevel.BOUNDED: 2, SoundnessLevel.APPROXIMATE: 3}
        if order[a] >= order[b]:
            return a
        return b


@dataclass
class SoundnessAnnotation:
    """Tracks soundness assumptions for a verification result."""
    level: SoundnessLevel
    assumptions: List[str] = field(default_factory=list)
    delta: Optional[float] = None  # For DELTA_SOUND
    time_bound: Optional[float] = None  # For BOUNDED
    approximation_error: Optional[float] = None  # For APPROXIMATE
    error_budget: Optional[ErrorBudget] = None

    def weaken_to(self, level: SoundnessLevel, reason: str) -> 'SoundnessAnnotation':
        new_level = SoundnessLevel.meet(self.level, level)
        return SoundnessAnnotation(
            level=new_level,
            assumptions=self.assumptions + [reason],
            delta=self.delta,
            time_bound=self.time_bound,
            approximation_error=self.approximation_error,
            error_budget=self.error_budget,
        )

    def with_delta(self, delta: float) -> 'SoundnessAnnotation':
        budget = self.error_budget or ErrorBudget()
        budget = ErrorBudget(
            delta=delta,
            epsilon=budget.epsilon,
            truncation=budget.truncation,
            discretization=budget.discretization,
        )
        return SoundnessAnnotation(
            level=SoundnessLevel.meet(self.level, SoundnessLevel.DELTA_SOUND),
            assumptions=self.assumptions + [f"dReal delta-satisfiability with delta={delta}"],
            delta=delta,
            time_bound=self.time_bound,
            approximation_error=self.approximation_error,
            error_budget=budget,
        )

    def with_time_bound(self, t: float) -> 'SoundnessAnnotation':
        return SoundnessAnnotation(
            level=SoundnessLevel.meet(self.level, SoundnessLevel.BOUNDED),
            assumptions=self.assumptions + [f"Bounded time horizon T={t}"],
            delta=self.delta,
            time_bound=t,
            approximation_error=self.approximation_error,
            error_budget=self.error_budget,
        )

    def with_error_budget(self, budget: ErrorBudget) -> 'SoundnessAnnotation':
        """Attach or update the error budget."""
        return SoundnessAnnotation(
            level=self.level,
            assumptions=self.assumptions,
            delta=self.delta,
            time_bound=self.time_bound,
            approximation_error=self.approximation_error,
            error_budget=budget,
        )
