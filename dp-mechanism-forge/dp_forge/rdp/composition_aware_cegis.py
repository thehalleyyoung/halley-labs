"""
CEGIS with RDP composition awareness.

Extends the existing CEGIS synthesis engine with RDP-based composition
accounting.  Given T sequential queries and a total RDP budget, this
module:

1. Allocates per-query RDP budgets using the budget optimizer.
2. Runs the existing CEGISEngine for each query at its allocated budget.
3. Certifies the composed privacy guarantee via RDP accounting.

This enables optimal mechanism synthesis across multiple queries while
maintaining tight privacy guarantees through Rényi DP composition.

Classes:
    CompositionAwareCEGIS  — CEGIS synthesis with RDP composition.
    ComposedSynthesisResult — Result of a composed synthesis run.

References:
    - Mironov, I. (2017). Rényi differential privacy.
    - Balle et al. (2020). Privacy profiles and amplification.
"""

from __future__ import annotations

import logging
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt

from dp_forge.exceptions import (
    BudgetExhaustedError,
    ConfigurationError,
    ConvergenceError,
)
from dp_forge.types import (
    CEGISResult,
    PrivacyBudget,
    QuerySpec,
    SynthesisConfig,
)

from dp_forge.rdp.accountant import RDPAccountant, RDPCurve, DEFAULT_ALPHAS
from dp_forge.rdp.conversion import rdp_to_dp
from dp_forge.rdp.budget_optimizer import RDPBudgetOptimizer, RDPAllocationResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FloatArray = npt.NDArray[np.float64]


# =========================================================================
# Result types
# =========================================================================


@dataclass
class ComposedSynthesisResult:
    """Result of a composition-aware CEGIS synthesis run.

    Attributes:
        mechanisms: List of per-query CEGIS results.
        allocation: Budget allocation used.
        composed_budget: Final composed (ε, δ)-DP guarantee.
        composed_curve: Composed RDP curve.
        per_query_budgets: Per-query (ε, δ)-DP budgets used.
        total_time: Total synthesis time in seconds.
        certified: Whether the composed privacy was certified.
        metadata: Additional metadata.
    """

    mechanisms: List[CEGISResult]
    allocation: RDPAllocationResult
    composed_budget: PrivacyBudget
    composed_curve: RDPCurve
    per_query_budgets: List[PrivacyBudget]
    total_time: float = 0.0
    certified: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_queries(self) -> int:
        """Number of queries synthesised."""
        return len(self.mechanisms)

    @property
    def total_iterations(self) -> int:
        """Total CEGIS iterations across all queries."""
        return sum(m.iterations for m in self.mechanisms)

    def __repr__(self) -> str:
        cert = "certified" if self.certified else "uncertified"
        return (
            f"ComposedSynthesisResult(n_queries={self.n_queries}, "
            f"ε={self.composed_budget.epsilon:.4f}, "
            f"δ={self.composed_budget.delta:.2e}, "
            f"total_iter={self.total_iterations}, {cert})"
        )


@dataclass
class CertificationResult:
    """Result of an RDP composition certification check.

    Attributes:
        certified: Whether the composition satisfies the budget.
        composed_epsilon: Composed ε at the target δ.
        target_epsilon: Target ε from the budget.
        target_delta: Target δ.
        optimal_alpha: The α achieving the tightest bound.
        slack: How much ε slack remains (target - composed).
        per_query_rdp: Per-query RDP curves.
    """

    certified: bool
    composed_epsilon: float
    target_epsilon: float
    target_delta: float
    optimal_alpha: float = 0.0
    slack: float = 0.0
    per_query_rdp: List[RDPCurve] = field(default_factory=list)

    def __repr__(self) -> str:
        status = "✓" if self.certified else "✗"
        return (
            f"CertificationResult({status}, composed_ε={self.composed_epsilon:.4f}, "
            f"target_ε={self.target_epsilon:.4f}, slack={self.slack:.4f})"
        )


# =========================================================================
# CompositionAwareCEGIS
# =========================================================================


class CompositionAwareCEGIS:
    """CEGIS synthesis engine with RDP composition awareness.

    Orchestrates the synthesis of optimal DP mechanisms for T sequential
    queries while maintaining a total RDP privacy budget.  For each query,
    it allocates a per-query budget, runs the CEGIS engine, and tracks
    the composed privacy guarantee.

    Args:
        total_budget: Total privacy budget for all queries.
        alphas: α grid for RDP accounting.
        synthesis_config: Configuration for the CEGIS engine.
        allocation_method: Budget allocation method. One of
            ``"uniform"``, ``"proportional"``, ``"convex"``.
        mechanism_type: Mechanism type for budget allocation error
            functions (``"gaussian"`` or ``"laplace"``).

    Example::

        cegis = CompositionAwareCEGIS(
            total_budget=PrivacyBudget(2.0, 1e-5),
            allocation_method="convex",
        )
        specs = [
            QuerySpec.counting(n=5, epsilon=1.0),
            QuerySpec.counting(n=5, epsilon=1.0),
        ]
        result = cegis.synthesize_composed(specs)
    """

    def __init__(
        self,
        total_budget: PrivacyBudget,
        alphas: Optional[FloatArray] = None,
        synthesis_config: Optional[SynthesisConfig] = None,
        allocation_method: str = "uniform",
        mechanism_type: str = "gaussian",
    ) -> None:
        if total_budget.epsilon <= 0:
            raise ConfigurationError(
                f"total_budget.epsilon must be > 0, got {total_budget.epsilon}",
                parameter="total_budget",
            )

        self._total_budget = total_budget
        self._alphas = (
            np.asarray(alphas, dtype=np.float64)
            if alphas is not None
            else DEFAULT_ALPHAS.copy()
        )
        self._synthesis_config = synthesis_config or SynthesisConfig()
        self._allocation_method = allocation_method.lower()
        self._mechanism_type = mechanism_type.lower()

        # State
        self._accountant = RDPAccountant(
            alphas=self._alphas,
            total_budget=total_budget,
        )
        self._results: List[CEGISResult] = []
        self._query_curves: List[RDPCurve] = []

    @property
    def total_budget(self) -> PrivacyBudget:
        """The total privacy budget."""
        return self._total_budget

    @property
    def n_synthesized(self) -> int:
        """Number of queries synthesised so far."""
        return len(self._results)

    @property
    def accountant(self) -> RDPAccountant:
        """The underlying RDP accountant."""
        return self._accountant

    # -----------------------------------------------------------------
    # Main synthesis entry point
    # -----------------------------------------------------------------

    def synthesize_composed(
        self,
        specs: Sequence[QuerySpec],
        error_fns: Optional[Sequence[Callable[[float, float], float]]] = None,
        weights: Optional[Sequence[float]] = None,
        callback: Optional[Callable[[int, CEGISResult], None]] = None,
    ) -> ComposedSynthesisResult:
        """Synthesise mechanisms for T queries with RDP composition.

        Steps:
        1. Allocate per-query budgets via the chosen method.
        2. For each query, run CEGISEngine at the allocated budget.
        3. Track composed RDP and certify the total guarantee.

        Args:
            specs: Sequence of QuerySpec objects for each query.
            error_fns: Optional per-query error functions for budget
                allocation. Defaults to mechanism-specific MSE.
            weights: Optional per-query weights for proportional
                allocation.
            callback: Optional callback invoked after each query synthesis
                with ``(query_index, cegis_result)``.

        Returns:
            ComposedSynthesisResult with all mechanisms and privacy
            accounting.

        Raises:
            ConfigurationError: If inputs are invalid.
            BudgetExhaustedError: If the budget is exceeded.
        """
        if not specs:
            raise ConfigurationError(
                "At least one query specification is required",
                parameter="specs",
            )

        T = len(specs)
        t_start = time.monotonic()

        # Step 1: Allocate budgets
        sensitivities = [float(s.sensitivity) for s in specs]
        allocation = self._allocate_budgets(sensitivities, error_fns, weights)

        logger.info(
            "Budget allocation complete: method=%s, per-query ε range=[%.4f, %.4f]",
            self._allocation_method,
            float(np.min(allocation.epsilons)),
            float(np.max(allocation.epsilons)),
        )

        # Step 2: Synthesise each query
        mechanisms: List[CEGISResult] = []
        per_query_budgets: List[PrivacyBudget] = []
        rdp_curves: List[RDPCurve] = []
        accountant = RDPAccountant(alphas=self._alphas)

        for t in range(T):
            eps_t = float(allocation.epsilons[t])
            sens_t = float(sensitivities[t])

            # Create modified QuerySpec with allocated budget
            spec_t = self._create_allocated_spec(specs[t], eps_t)
            budget_t = PrivacyBudget(epsilon=eps_t, delta=spec_t.delta)
            per_query_budgets.append(budget_t)

            logger.info(
                "Synthesising query %d/%d: ε=%.4f, Δ=%.4f",
                t + 1, T, eps_t, sens_t,
            )

            # Run CEGIS
            result = self._synthesize_single(spec_t)
            mechanisms.append(result)

            # Track RDP
            curve = self._mechanism_to_rdp_curve(eps_t, sens_t, t)
            rdp_curves.append(curve)
            accountant.add_mechanism(curve)

            if callback is not None:
                callback(t, result)

        # Step 3: Compute composed guarantee
        delta = self._total_budget.delta if self._total_budget.delta > 0 else 1e-10
        composed_budget = accountant.to_dp(delta)
        composed_curve = accountant.get_composed_curve()

        # Certify
        cert = self.certify_composition(rdp_curves)
        total_time = time.monotonic() - t_start

        return ComposedSynthesisResult(
            mechanisms=mechanisms,
            allocation=allocation,
            composed_budget=composed_budget,
            composed_curve=composed_curve,
            per_query_budgets=per_query_budgets,
            total_time=total_time,
            certified=cert.certified,
            metadata={
                "allocation_method": self._allocation_method,
                "mechanism_type": self._mechanism_type,
                "certification": cert,
            },
        )

    # -----------------------------------------------------------------
    # Budget allocation
    # -----------------------------------------------------------------

    def _allocate_budgets(
        self,
        sensitivities: Sequence[float],
        error_fns: Optional[Sequence[Callable[[float, float], float]]],
        weights: Optional[Sequence[float]],
    ) -> RDPAllocationResult:
        """Allocate per-query budgets using the configured method."""
        optimizer = RDPBudgetOptimizer(
            target_budget=self._total_budget,
            alphas=self._alphas,
            mechanism_type=self._mechanism_type,
        )

        if self._allocation_method == "uniform":
            return optimizer.optimize_uniform(sensitivities, error_fns)
        elif self._allocation_method == "proportional":
            return optimizer.optimize_proportional(
                sensitivities, weights=weights, error_fns=error_fns,
            )
        elif self._allocation_method == "convex":
            return optimizer.optimize_convex(sensitivities, error_fns)
        else:
            raise ConfigurationError(
                f"Unknown allocation method: {self._allocation_method!r}. "
                f"Supported: 'uniform', 'proportional', 'convex'.",
                parameter="allocation_method",
                value=self._allocation_method,
            )

    # -----------------------------------------------------------------
    # Single-query synthesis
    # -----------------------------------------------------------------

    def _synthesize_single(self, spec: QuerySpec) -> CEGISResult:
        """Synthesise a single mechanism using CEGISEngine.

        Imports CEGISEngine lazily to avoid circular imports.
        """
        from dp_forge.cegis_loop import CEGISEngine

        engine = CEGISEngine(self._synthesis_config)
        return engine.synthesize(spec)

    def _create_allocated_spec(
        self,
        original: QuerySpec,
        epsilon: float,
    ) -> QuerySpec:
        """Create a QuerySpec with the allocated epsilon."""
        return QuerySpec(
            query_values=original.query_values.copy(),
            domain=original.domain,
            sensitivity=original.sensitivity,
            epsilon=epsilon,
            delta=original.delta,
            k=original.k,
            loss_fn=original.loss_fn,
            custom_loss=original.custom_loss,
            edges=original.edges,
            query_type=original.query_type,
            metadata={**original.metadata, "allocated_epsilon": epsilon},
        )

    def _mechanism_to_rdp_curve(
        self,
        epsilon: float,
        sensitivity: float,
        query_index: int,
    ) -> RDPCurve:
        """Convert a mechanism's parameters to an RDP curve."""
        if self._mechanism_type == "gaussian":
            delta = self._total_budget.delta if self._total_budget.delta > 0 else 1e-10
            sigma = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
            rdp_eps = self._alphas * sensitivity ** 2 / (2.0 * sigma ** 2)
        elif self._mechanism_type == "laplace":
            rdp_eps = self._laplace_rdp_at_alphas(epsilon)
        else:
            # Default: use Gaussian
            sigma = sensitivity / epsilon
            rdp_eps = self._alphas * sensitivity ** 2 / (2.0 * sigma ** 2)

        return RDPCurve(
            alphas=self._alphas.copy(),
            epsilons=rdp_eps,
            name=f"query_{query_index}",
        )

    def _laplace_rdp_at_alphas(self, epsilon: float) -> FloatArray:
        """Compute Laplace RDP values at the configured α grid."""
        rdp = np.empty_like(self._alphas)
        for i, alpha in enumerate(self._alphas):
            if abs(alpha - 1.0) < 1e-10:
                rdp[i] = 0.0
                continue
            a_m1 = alpha - 1.0
            denom = 2.0 * alpha - 1.0
            if denom <= 0:
                rdp[i] = 0.0
                continue
            log_t1 = math.log(alpha / denom) + a_m1 * epsilon
            log_t2 = math.log(a_m1 / denom) - a_m1 * epsilon
            log_sum = float(np.logaddexp(log_t1, log_t2))
            rdp[i] = max(log_sum / a_m1, 0.0)
        return rdp

    # -----------------------------------------------------------------
    # Certification
    # -----------------------------------------------------------------

    def certify_composition(
        self,
        rdp_curves: Sequence[RDPCurve],
        target_budget: Optional[PrivacyBudget] = None,
    ) -> CertificationResult:
        """Certify that the composed RDP curves satisfy the target budget.

        Composes all per-query RDP curves, converts to (ε, δ)-DP, and
        checks whether the result satisfies the target budget.

        Args:
            rdp_curves: Per-query RDP curves.
            target_budget: Target budget. Defaults to ``total_budget``.

        Returns:
            CertificationResult with the verification outcome.
        """
        budget = target_budget or self._total_budget
        delta = budget.delta if budget.delta > 0 else 1e-10

        if not rdp_curves:
            return CertificationResult(
                certified=True,
                composed_epsilon=0.0,
                target_epsilon=budget.epsilon,
                target_delta=delta,
            )

        # Compose RDP curves
        composed_rdp = np.zeros_like(self._alphas)
        for curve in rdp_curves:
            composed_rdp += curve.evaluate_vectorized(self._alphas)

        # Convert to (ε, δ)-DP
        eps, opt_alpha = rdp_to_dp(composed_rdp, self._alphas, delta)

        certified = eps <= budget.epsilon
        slack = budget.epsilon - eps

        return CertificationResult(
            certified=certified,
            composed_epsilon=eps,
            target_epsilon=budget.epsilon,
            target_delta=delta,
            optimal_alpha=opt_alpha,
            slack=slack,
            per_query_rdp=list(rdp_curves),
        )

    # -----------------------------------------------------------------
    # Incremental synthesis
    # -----------------------------------------------------------------

    def synthesize_next(
        self,
        spec: QuerySpec,
    ) -> Tuple[CEGISResult, PrivacyBudget]:
        """Synthesise the next query incrementally.

        Uses the remaining budget from the accountant to determine the
        per-query epsilon allocation.

        Args:
            spec: Query specification for the next query.

        Returns:
            Tuple of (CEGISResult, remaining_budget).

        Raises:
            BudgetExhaustedError: If no budget remains.
        """
        delta = self._total_budget.delta if self._total_budget.delta > 0 else 1e-10

        # Check remaining budget
        try:
            remaining = self._accountant.remaining_budget(delta)
        except BudgetExhaustedError:
            raise BudgetExhaustedError(
                "No privacy budget remaining for additional queries",
                budget_epsilon=self._total_budget.epsilon,
                budget_delta=delta,
            )

        # Use a fraction of remaining budget
        eps_t = remaining.epsilon
        spec_t = self._create_allocated_spec(spec, eps_t)

        # Synthesise
        result = self._synthesize_single(spec_t)

        # Track
        curve = self._mechanism_to_rdp_curve(eps_t, spec.sensitivity, self.n_synthesized)
        self._accountant.add_mechanism(curve)
        self._results.append(result)
        self._query_curves.append(curve)

        # Return remaining budget after this query
        try:
            remaining = self._accountant.remaining_budget(delta)
        except BudgetExhaustedError:
            remaining = PrivacyBudget(epsilon=1e-15, delta=delta)

        return result, remaining

    # -----------------------------------------------------------------
    # State
    # -----------------------------------------------------------------

    def reset(self) -> None:
        """Reset the composition state, clearing all synthesised queries."""
        self._accountant.reset()
        self._results.clear()
        self._query_curves.clear()

    def __repr__(self) -> str:
        return (
            f"CompositionAwareCEGIS(budget={self._total_budget}, "
            f"n_synthesized={self.n_synthesized}, "
            f"method={self._allocation_method!r})"
        )
