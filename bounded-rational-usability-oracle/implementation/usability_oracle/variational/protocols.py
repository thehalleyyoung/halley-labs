"""
usability_oracle.variational.protocols — Structural interfaces for the
variational free-energy solver subsystem.

All protocols use :pep:`544` structural subtyping so that implementations
need not inherit explicitly — they only need to expose the correct
method signatures.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from usability_oracle.core.types import CostTuple, PolicyDistribution, StateId
    from usability_oracle.variational.types import (
        CapacityProfile,
        FreeEnergyResult,
        KLDivergenceResult,
        VariationalConfig,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ObjectiveFunction
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class ObjectiveFunction(Protocol):
    """Evaluable variational objective and its gradient.

    An objective function encapsulates a specific free-energy
    decomposition (standard, ELBO, KL+cost, rate–distortion) so that
    the solver can be agnostic to the particular formulation.

    The parameter vector θ lives on the probability simplex;
    implementations must respect this constraint.
    """

    def evaluate(
        self,
        parameters: Sequence[float],
        beta: float,
    ) -> float:
        """Evaluate the objective F(θ; β).

        Parameters:
            parameters: Current variational parameters on the simplex.
            beta: Rationality parameter (inverse temperature).

        Returns:
            Scalar objective value.
        """
        ...

    def gradient(
        self,
        parameters: Sequence[float],
        beta: float,
    ) -> Sequence[float]:
        """Compute ∇_θ F(θ; β).

        Parameters:
            parameters: Current variational parameters.
            beta: Rationality parameter.

        Returns:
            Gradient vector of the same length as *parameters*.
        """
        ...

    def hessian_vector_product(
        self,
        parameters: Sequence[float],
        vector: Sequence[float],
        beta: float,
    ) -> Sequence[float]:
        """Compute the Hessian–vector product  H(θ) · v.

        Used by second-order solvers and saddle-point detection.

        Parameters:
            parameters: Current variational parameters.
            vector: Direction vector for the product.
            beta: Rationality parameter.

        Returns:
            Product vector of the same length as *parameters*.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# VariationalSolver
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class VariationalSolver(Protocol):
    """Minimise free energy to produce a bounded-rational policy.

    Implementations may use Blahut–Arimoto iteration, mirror descent,
    natural-gradient methods, or other algorithms appropriate for
    optimisation on the probability simplex.
    """

    def solve(
        self,
        cost_matrix: Dict[str, Dict[str, float]],
        reference_policy: Dict[str, Dict[str, float]],
        config: VariationalConfig,
    ) -> FreeEnergyResult:
        """Compute the optimal bounded-rational policy.

        Parameters:
            cost_matrix: Mapping  state → {action → immediate cost}.
            reference_policy: Prior policy π₀(a|s) from which the KL
                divergence is measured.
            config: Solver configuration.

        Returns:
            A :class:`FreeEnergyResult` containing the optimal policy,
            free-energy decomposition, and convergence diagnostics.

        Raises:
            ConvergenceError: If the solver fails to converge.
            NumericalInstabilityError: On NaN / Inf in intermediate values.
        """
        ...

    def warm_start(
        self,
        initial_policy: Dict[str, Dict[str, float]],
        cost_matrix: Dict[str, Dict[str, float]],
        reference_policy: Dict[str, Dict[str, float]],
        config: VariationalConfig,
    ) -> FreeEnergyResult:
        """Re-solve from an existing policy (warm start).

        Useful for incremental updates when the cost matrix changes
        slightly between UI versions.

        Parameters:
            initial_policy: Starting policy (e.g. solution from a
                previous version).
            cost_matrix: Updated cost matrix.
            reference_policy: Prior policy π₀.
            config: Solver configuration.

        Returns:
            Updated :class:`FreeEnergyResult`.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# CapacityEstimator
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class CapacityEstimator(Protocol):
    """Estimate the effective channel capacity of a bounded-rational agent.

    The capacity  C(β) = max_{π} I(S; A)  under the constraint that
    the policy must have free energy ≤ F_β.  Sweeping β traces out the
    rate–distortion curve.
    """

    def estimate_profile(
        self,
        cost_matrix: Dict[str, Dict[str, float]],
        reference_policy: Dict[str, Dict[str, float]],
        beta_range: Sequence[float],
        config: VariationalConfig,
    ) -> CapacityProfile:
        """Sweep β values and compute the capacity profile.

        Parameters:
            cost_matrix: State–action cost matrix.
            reference_policy: Prior policy π₀.
            beta_range: Sequence of β values to evaluate (must be
                monotonically increasing).
            config: Base solver configuration (β field is overridden
                per sweep point).

        Returns:
            A :class:`CapacityProfile` with capacity, cost, and KL at
            each β.
        """
        ...

    def compute_kl(
        self,
        policy: Dict[str, Dict[str, float]],
        reference: Dict[str, Dict[str, float]],
        state_distribution: Optional[Dict[str, float]] = None,
    ) -> KLDivergenceResult:
        """Compute KL(π ‖ π₀) with per-state breakdown.

        Parameters:
            policy: Current policy  π(a|s).
            reference: Reference policy π₀(a|s).
            state_distribution: Optional stationary distribution d(s).
                If ``None``, a uniform distribution is assumed.

        Returns:
            Per-state and aggregate KL divergence.
        """
        ...
