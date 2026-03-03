"""
SubsampledCEGIS engine for DP-Forge.

Extends the core CEGIS synthesis loop with Poisson subsampling to achieve
privacy amplification.  The workflow is:

    1. **Budget inversion**: Given target (ε_final, δ_final) and rate q,
       back-solve for base (ε₀, δ₀) such that amplification meets the target.
    2. **Base synthesis**: Run the existing :class:`CEGISEngine` at base
       privacy (ε₀, δ₀) to produce an optimal base mechanism.
    3. **Protocol wrapping**: Wrap the result in a :class:`SubsamplingProtocol`
       that applies Poisson inclusion before the mechanism.
    4. **Error tracking**: Track total error = base_error / q, since
       subsampling scales error inversely with the sampling rate.

The key benefit: by running CEGIS at a relaxed privacy level (ε₀ > ε_final),
we obtain a more accurate base mechanism.  The subsampling step then
tightens the privacy back to the target level, while the amplified accuracy
is better than directly synthesising at ε_final.

Classes:
    - :class:`SubsampledCEGIS` — Main synthesis engine.
    - :class:`SubsampledMechanism` — Complete result container.

Functions:
    - :func:`synthesize_subsampled` — One-line high-level API.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from dp_forge.cegis_loop import CEGISEngine, CEGISProgress
from dp_forge.exceptions import (
    ConfigurationError,
    ConvergenceError,
    InfeasibleSpecError,
)
from dp_forge.types import (
    CEGISResult,
    ExtractedMechanism,
    LossFunction,
    PrivacyBudget,
    QuerySpec,
    SynthesisConfig,
)

from dp_forge.subsampling.amplification import (
    AmplificationBound,
    AmplificationResult,
    poisson_amplify,
)
from dp_forge.subsampling.budget_inversion import (
    BudgetInverter,
    InversionResult,
)
from dp_forge.subsampling.protocol import (
    SubsamplingMode,
    SubsamplingProtocol,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SubsampledMechanism:
    """Complete result of subsampled mechanism synthesis.

    Contains the base mechanism from CEGIS, the subsampling protocol,
    the amplified privacy guarantee, and metadata about the synthesis.

    Attributes:
        protocol: The SubsamplingProtocol encapsulating the full mechanism.
        cegis_result: The CEGISResult from base-level synthesis.
        inversion: The InversionResult from budget inversion.
        amplified: The AmplificationResult with final privacy guarantee.
        target_eps: The requested target ε.
        target_delta: The requested target δ.
        q_rate: The subsampling rate used.
        base_error: Estimated error of the base mechanism.
        amplified_error: Estimated error after subsampling (≈ base_error / q).
        synthesis_time: Total wall-clock time for synthesis.
    """

    protocol: SubsamplingProtocol
    cegis_result: CEGISResult
    inversion: InversionResult
    amplified: AmplificationResult
    target_eps: float
    target_delta: float
    q_rate: float
    base_error: float
    amplified_error: float
    synthesis_time: float

    @property
    def base_eps(self) -> float:
        """Base mechanism ε₀ used for CEGIS synthesis."""
        return self.inversion.base_eps

    @property
    def base_delta(self) -> float:
        """Base mechanism δ₀ used for CEGIS synthesis."""
        return self.inversion.base_delta

    @property
    def final_budget(self) -> PrivacyBudget:
        """Final amplified privacy budget."""
        return self.amplified.budget

    @property
    def mechanism_table(self) -> npt.NDArray[np.float64]:
        """The base mechanism probability table."""
        return self.cegis_result.mechanism

    @property
    def iterations(self) -> int:
        """Number of CEGIS iterations used."""
        return self.cegis_result.iterations

    def verify_amplification(self, tol: float = 1e-6) -> bool:
        """Verify that the amplified privacy meets the target.

        Recomputes amplification from the base parameters and checks
        that the result is within tolerance of the target.

        Args:
            tol: Tolerance for verification.

        Returns:
            True if the amplified guarantee meets the target.
        """
        actual = poisson_amplify(
            self.inversion.base_eps,
            self.inversion.base_delta,
            self.q_rate,
        )
        eps_ok = actual.eps <= self.target_eps + tol
        delta_ok = actual.delta <= self.target_delta + tol
        return eps_ok and delta_ok

    def __repr__(self) -> str:
        return (
            f"SubsampledMechanism("
            f"target_ε={self.target_eps:.4f}, "
            f"base_ε={self.base_eps:.4f}, "
            f"q={self.q_rate:.4f}, "
            f"amplified_ε={self.amplified.eps:.6f}, "
            f"iter={self.iterations}, "
            f"time={self.synthesis_time:.2f}s)"
        )


# =========================================================================
# SubsampledCEGIS Engine
# =========================================================================


class SubsampledCEGIS:
    """CEGIS synthesis engine with privacy amplification by subsampling.

    Given target privacy (ε_final, δ_final) and subsampling rate q:

    1. Uses :class:`BudgetInverter` to find base (ε₀, δ₀) such that
       Poisson amplification at rate q yields (ε_final, δ_final).

    2. Constructs a modified :class:`QuerySpec` at the relaxed base
       privacy level (ε₀, δ₀).

    3. Runs :class:`CEGISEngine` at the base level to synthesise an
       optimal mechanism for the relaxed privacy.

    4. Wraps the result in a :class:`SubsamplingProtocol` and verifies
       that amplification actually meets the target.

    The benefit: ε₀ > ε_final, so the base mechanism has lower error
    than one synthesised directly at ε_final.  After subsampling, the
    total error is approximately base_error / q, but for small q the
    amplification benefit outweighs the 1/q penalty.

    Args:
        config: Synthesis configuration for the CEGIS engine.
        inverter_tol: Tolerance for the budget inverter bisection.

    Example::

        engine = SubsampledCEGIS()
        result = engine.synthesize(
            spec=QuerySpec.counting(n=5, epsilon=1.0),
            q_rate=0.01,
            target_eps=0.1,
            target_delta=1e-5,
        )
        print(result.amplified.eps)  # Should be ≤ 0.1
    """

    def __init__(
        self,
        config: Optional[SynthesisConfig] = None,
        inverter_tol: float = 1e-12,
    ) -> None:
        self._config = config or SynthesisConfig()
        self._inverter = BudgetInverter(tol=inverter_tol)
        self._last_result: Optional[SubsampledMechanism] = None

    @property
    def config(self) -> SynthesisConfig:
        """Current synthesis configuration."""
        return self._config

    @property
    def last_result(self) -> Optional[SubsampledMechanism]:
        """Result from the last synthesis call."""
        return self._last_result

    def synthesize(
        self,
        spec: QuerySpec,
        q_rate: float,
        target_eps: Optional[float] = None,
        target_delta: Optional[float] = None,
        *,
        mode: SubsamplingMode = SubsamplingMode.POISSON,
        callback: Optional[Callable[[CEGISProgress], None]] = None,
        seed: Optional[int] = None,
    ) -> SubsampledMechanism:
        """Synthesise an optimal subsampled mechanism.

        Args:
            spec: Query specification (used for query structure, sensitivity,
                discretization, and loss function; the epsilon/delta in the
                spec are ignored in favour of target_eps/target_delta).
            q_rate: Poisson subsampling rate q ∈ (0, 1].
            target_eps: Target amplified ε. Defaults to spec.epsilon.
            target_delta: Target amplified δ. Defaults to spec.delta.
            mode: Subsampling strategy.
            callback: Progress callback for the CEGIS engine.
            seed: Random seed for the protocol.

        Returns:
            SubsampledMechanism with the complete result.

        Raises:
            ConfigurationError: If parameters are invalid.
            InfeasibleSpecError: If no mechanism exists at the base level.
            ConvergenceError: If CEGIS or inversion fails to converge.
        """
        t_start = time.monotonic()

        # Use spec's privacy parameters as defaults
        if target_eps is None:
            target_eps = spec.epsilon
        if target_delta is None:
            target_delta = spec.delta

        # Validate subsampling rate
        if not (0.0 < q_rate <= 1.0):
            raise ConfigurationError(
                f"q_rate must be in (0, 1], got {q_rate}",
                parameter="q_rate",
                value=q_rate,
                constraint="(0, 1]",
            )

        logger.info(
            "SubsampledCEGIS: target (ε=%.4f, δ=%.2e), q=%.4f",
            target_eps, target_delta, q_rate,
        )

        # Step 1: Invert the amplification curve
        if q_rate < 1.0:
            inversion = self._inverter.invert(
                target_eps=target_eps,
                target_delta=target_delta,
                q_rate=q_rate,
            )
            base_eps = inversion.base_eps
            base_delta = inversion.base_delta
        else:
            # No subsampling: base = target
            inversion = InversionResult(
                base_eps=target_eps,
                base_delta=target_delta,
                target_eps=target_eps,
                target_delta=target_delta,
                q_rate=q_rate,
                n_iterations=0,
                residual=0.0,
            )
            base_eps = target_eps
            base_delta = target_delta

        logger.info(
            "Budget inversion: base (ε₀=%.4f, δ₀=%.2e) → "
            "amplified (ε=%.4f, δ=%.2e)",
            base_eps, base_delta, target_eps, target_delta,
        )

        # Step 2: Build modified QuerySpec at base privacy level
        base_spec = self._build_base_spec(spec, base_eps, base_delta)

        # Step 3: Run CEGIS at base level
        engine = CEGISEngine(self._config)
        cegis_result = engine.synthesize(base_spec, callback=callback)

        logger.info(
            "CEGIS completed: %d iterations, obj=%.6f",
            cegis_result.iterations, cegis_result.obj_val,
        )

        # Step 4: Compute amplified privacy
        amplified = poisson_amplify(base_eps, base_delta, q_rate)

        # Step 5: Verify amplification meets target
        if amplified.eps > target_eps * (1.0 + 1e-6):
            warnings.warn(
                f"Amplified ε={amplified.eps:.6f} exceeds target "
                f"ε={target_eps:.6f}. This may be due to numerical "
                f"precision in the inversion.",
                stacklevel=2,
            )

        # Step 6: Build the protocol
        # Extract y_grid from the spec
        y_grid = self._build_y_grid(spec)

        protocol = SubsamplingProtocol(
            q_rate=q_rate,
            base_mechanism=cegis_result.mechanism,
            base_eps=base_eps,
            base_delta=base_delta,
            amplified=amplified,
            y_grid=y_grid,
            mode=mode,
            seed=seed,
        )

        # Step 7: Estimate errors
        base_error = cegis_result.obj_val
        amplified_error = base_error / q_rate if q_rate > 0 else float("inf")

        synthesis_time = time.monotonic() - t_start

        result = SubsampledMechanism(
            protocol=protocol,
            cegis_result=cegis_result,
            inversion=inversion,
            amplified=amplified,
            target_eps=target_eps,
            target_delta=target_delta,
            q_rate=q_rate,
            base_error=base_error,
            amplified_error=amplified_error,
            synthesis_time=synthesis_time,
        )

        self._last_result = result

        logger.info(
            "SubsampledCEGIS complete: amplified ε=%.6f, δ=%.2e, "
            "base_error=%.6f, amplified_error=%.6f, time=%.2fs",
            amplified.eps, amplified.delta,
            base_error, amplified_error, synthesis_time,
        )

        return result

    def _build_base_spec(
        self,
        spec: QuerySpec,
        base_eps: float,
        base_delta: float,
    ) -> QuerySpec:
        """Build a QuerySpec at the base privacy level.

        Copies the query structure, sensitivity, discretization, and loss
        from the original spec but replaces epsilon and delta with the
        base-level values from budget inversion.

        Args:
            spec: Original query specification.
            base_eps: Base privacy ε₀.
            base_delta: Base privacy δ₀.

        Returns:
            Modified QuerySpec at base privacy level.
        """
        return QuerySpec(
            query_values=spec.query_values.copy(),
            domain=spec.domain,
            sensitivity=spec.sensitivity,
            epsilon=base_eps,
            delta=base_delta,
            k=spec.k,
            loss_fn=spec.loss_fn,
            custom_loss=spec.custom_loss,
            edges=spec.edges,
            query_type=spec.query_type,
            metadata={
                **spec.metadata,
                "subsampled": True,
                "original_eps": spec.epsilon,
                "original_delta": spec.delta,
            },
        )

    def _build_y_grid(self, spec: QuerySpec) -> npt.NDArray[np.float64]:
        """Build the output discretization grid for the protocol.

        Uses the same grid construction as the CEGIS engine.

        Args:
            spec: Query specification.

        Returns:
            Array of output grid values.
        """
        v_min = float(np.min(spec.query_values))
        v_max = float(np.max(spec.query_values))
        spread = max(v_max - v_min, spec.sensitivity)
        margin = spread * 0.5
        return np.linspace(v_min - margin, v_max + margin, spec.k)

    def suggest_q_rate(
        self,
        spec: QuerySpec,
        target_eps: float,
        target_delta: float,
        *,
        max_error_ratio: float = 2.0,
    ) -> float:
        """Suggest an optimal subsampling rate for given targets.

        Balances the amplification benefit against the 1/q error scaling.
        The "sweet spot" is where the amplification benefit (relaxed base ε₀)
        outweighs the 1/q error penalty.

        For small ε_target, the optimal q is approximately:
            q* ≈ ε_target / ε₀_max

        where ε₀_max is the largest base ε₀ that still gives a useful
        mechanism (empirically around 2-3).

        Args:
            spec: Query specification.
            target_eps: Target amplified ε.
            target_delta: Target amplified δ.
            max_error_ratio: Maximum acceptable ratio of subsampled error
                to direct-synthesis error.

        Returns:
            Suggested subsampling rate q ∈ (0, 1].
        """
        # For very small ε_target, subsampling helps a lot
        # For ε_target ≥ 1, subsampling rarely helps
        if target_eps >= 1.0:
            return 1.0

        # Heuristic: q ≈ target_eps / eps_sweet_spot
        # where eps_sweet_spot is the ε₀ at which the base mechanism
        # has a good accuracy-privacy trade-off (typically around 1-2)
        eps_sweet_spot = min(2.0, target_eps * 10.0)

        # From Poisson formula: target_eps ≈ q · eps_sweet_spot for small q
        q_suggested = target_eps / eps_sweet_spot

        # Clamp to valid range
        q_suggested = max(0.001, min(1.0, q_suggested))

        # Verify that 1/q error penalty is within the max_error_ratio
        if 1.0 / q_suggested > max_error_ratio:
            q_suggested = 1.0 / max_error_ratio

        return q_suggested


# =========================================================================
# High-level API
# =========================================================================


import warnings  # noqa: E402 (already imported at module level)


def synthesize_subsampled(
    spec: QuerySpec,
    q_rate: float,
    *,
    target_eps: Optional[float] = None,
    target_delta: Optional[float] = None,
    config: Optional[SynthesisConfig] = None,
    mode: SubsamplingMode = SubsamplingMode.POISSON,
    seed: Optional[int] = None,
) -> SubsampledMechanism:
    """One-line API for subsampled mechanism synthesis.

    Convenience wrapper that creates a :class:`SubsampledCEGIS` engine
    and runs synthesis in a single call.

    Args:
        spec: Query specification.
        q_rate: Poisson subsampling rate q ∈ (0, 1].
        target_eps: Target amplified ε. Defaults to spec.epsilon.
        target_delta: Target amplified δ. Defaults to spec.delta.
        config: CEGIS configuration. Defaults to SynthesisConfig().
        mode: Subsampling strategy.
        seed: Random seed.

    Returns:
        SubsampledMechanism with the complete result.

    Example::

        spec = QuerySpec.counting(n=5, epsilon=1.0)
        result = synthesize_subsampled(spec, q_rate=0.01)
        print(result)
    """
    engine = SubsampledCEGIS(config=config)
    return engine.synthesize(
        spec, q_rate,
        target_eps=target_eps,
        target_delta=target_delta,
        mode=mode,
        seed=seed,
    )
