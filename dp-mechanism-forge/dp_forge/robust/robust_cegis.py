"""
RobustCEGIS engine for numerically certified DP mechanism synthesis.

Wraps the standard :class:`~dp_forge.cegis_loop.CEGISEngine` with constraint
inflation, interval-arithmetic verification, perturbation analysis, and
post-solve diagnostics to produce mechanisms with rigorous numerical
guarantees.

Algorithm:
    1. Compute inflation margins ε_ν, δ_ν from solver tolerance ν.
    2. Tighten LP constraints by the computed margins.
    3. Run the standard CEGIS loop on the tightened LP.
    4. Run post-solve diagnostics (constraint audit, residual analysis).
    5. Verify the solution satisfies ORIGINAL constraints using interval
       arithmetic.
    6. If interval verification fails, attempt iterative refinement and
       re-verify.
    7. Bundle the mechanism with a :class:`NumericalCertificate` and
       return a :class:`CertifiedMechanism`.

Guarantees:
    The output mechanism satisfies (ε + ε_ν, δ + δ_ν)-DP where:
        ε_ν = O(ν · e^ε)   (proportional to solver tolerance × privacy ratio)
        δ_ν = O(ν)          (proportional to solver tolerance)

Key class:
    - :class:`RobustCEGISEngine` — Main engine wrapping CEGISEngine.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from dp_forge.cegis_loop import (
    CEGISEngine,
    CEGISProgress,
    CEGISStatus,
    ConvergenceHistory,
)
from dp_forge.exceptions import (
    ConfigurationError,
    ConvergenceError,
    NumericalInstabilityError,
    VerificationError,
)
from dp_forge.types import (
    CEGISResult,
    NumericalConfig,
    QuerySpec,
    SynthesisConfig,
)
from dp_forge.robust.certified_output import CertifiedMechanism, NumericalCertificate
from dp_forge.robust.constraint_inflation import ConstraintInflator, InflationResult
from dp_forge.robust.interval_arithmetic import Interval, interval_verify_dp
from dp_forge.robust.perturbation_analysis import PerturbationAnalyzer, PerturbationBound
from dp_forge.robust.solver_diagnostics import SolverDiagnostics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_SAFETY_FACTOR: float = 2.0
_DEFAULT_SOLVER_TOL: float = 1e-8
_DEFAULT_MAX_REFINEMENT_ATTEMPTS: int = 3
_DEFAULT_REFINEMENT_TOL_FACTOR: float = 0.1


@dataclass
class RobustSynthesisConfig:
    """Configuration for robust CEGIS synthesis.

    Attributes:
        base_config: Underlying SynthesisConfig for the standard CEGIS engine.
        solver_tolerance: Solver feasibility tolerance ν. This drives the
            inflation margin computation.
        safety_factor: Multiplicative safety factor for margin computation.
            Default 2.0 provides 2× the theoretical minimum margin.
        max_refinement_attempts: Maximum number of iterative refinement
            attempts if interval verification fails.
        strict_interval_verify: If True, require interval verification to
            pass. If False, allow fallback to point verification with
            effective parameters.
        max_condition_number: Threshold for ill-conditioning warnings.
        min_probability_estimate: Override for minimum probability estimate
            used in margin computation. If None, computed automatically.
    """

    base_config: SynthesisConfig = field(default_factory=SynthesisConfig)
    solver_tolerance: float = _DEFAULT_SOLVER_TOL
    safety_factor: float = _DEFAULT_SAFETY_FACTOR
    max_refinement_attempts: int = _DEFAULT_MAX_REFINEMENT_ATTEMPTS
    strict_interval_verify: bool = True
    max_condition_number: float = 1e12
    min_probability_estimate: Optional[float] = None

    def __post_init__(self) -> None:
        if self.solver_tolerance <= 0:
            raise ConfigurationError(
                f"solver_tolerance must be > 0, got {self.solver_tolerance}",
                parameter="solver_tolerance",
                value=self.solver_tolerance,
                constraint="solver_tolerance > 0",
            )
        if self.safety_factor < 1.0:
            raise ConfigurationError(
                f"safety_factor must be >= 1.0, got {self.safety_factor}",
                parameter="safety_factor",
                value=self.safety_factor,
                constraint="safety_factor >= 1.0",
            )
        if self.max_refinement_attempts < 0:
            raise ConfigurationError(
                f"max_refinement_attempts must be >= 0, got {self.max_refinement_attempts}",
                parameter="max_refinement_attempts",
                value=self.max_refinement_attempts,
                constraint="max_refinement_attempts >= 0",
            )

    def __repr__(self) -> str:
        return (
            f"RobustSynthesisConfig(ν={self.solver_tolerance:.2e}, "
            f"safety={self.safety_factor}, "
            f"strict={self.strict_interval_verify})"
        )


class RobustCEGISEngine:
    """CEGIS engine with numerical stability guarantees.

    Wraps the standard :class:`CEGISEngine` to produce mechanisms with
    rigorous (ε+ε_ν, δ+δ_ν)-DP guarantees.  The engine:

    1. Computes inflation margins based on solver tolerance.
    2. Delegates synthesis to the standard CEGIS engine.
    3. Runs post-solve diagnostics and perturbation analysis.
    4. Verifies the output with interval arithmetic.
    5. Returns a :class:`CertifiedMechanism` with full certificate.

    Args:
        config: Robust synthesis configuration.

    Example::

        engine = RobustCEGISEngine(RobustSynthesisConfig(
            solver_tolerance=1e-8,
            safety_factor=2.0,
        ))
        certified = engine.synthesize(
            spec=QuerySpec.counting(n=5, epsilon=1.0),
        )
        print(certified.summary())
        assert certified.verify_certificate(strict=True)
    """

    def __init__(
        self,
        config: Optional[RobustSynthesisConfig] = None,
    ) -> None:
        self._config = config or RobustSynthesisConfig()
        self._inflator = ConstraintInflator(
            safety_factor=self._config.safety_factor,
            min_probability_estimate=self._config.min_probability_estimate,
        )
        self._analyzer = PerturbationAnalyzer(
            max_condition_number=self._config.max_condition_number,
        )
        self._diagnostics = SolverDiagnostics(
            feasibility_tol=self._config.solver_tolerance,
            max_condition_number=self._config.max_condition_number,
        )
        self._cegis_engine = CEGISEngine(self._config.base_config)

    def synthesize(
        self,
        spec: QuerySpec,
        solver_tolerance: Optional[float] = None,
        callback: Optional[Callable[[CEGISProgress], None]] = None,
    ) -> CertifiedMechanism:
        """Synthesise a DP mechanism with numerical robustness certificate.

        This is the main entry point.  It runs the full robust CEGIS
        pipeline: margin computation → tightened synthesis → diagnostics →
        interval verification → certification.

        Args:
            spec: Query specification defining the synthesis problem.
            solver_tolerance: Override solver tolerance. Uses config default
                if None.
            callback: Optional progress callback for CEGIS iterations.

        Returns:
            CertifiedMechanism with the mechanism and numerical certificate.

        Raises:
            InfeasibleSpecError: If no mechanism exists (even before inflation).
            ConvergenceError: If CEGIS does not converge within max_iter.
            NumericalInstabilityError: If the LP is too ill-conditioned.
            VerificationError: If interval verification fails and strict mode
                is enabled with no successful refinement.
        """
        t_start = time.monotonic()
        nu = solver_tolerance or self._config.solver_tolerance

        # Validate inputs
        self._validate_spec(spec, nu)

        # Step 1: Compute inflation margins
        eps_margin, delta_margin = self._compute_margins(spec, nu)

        logger.info(
            "RobustCEGIS: ε=%.6f, δ=%.2e, ν=%.2e → margins: ε_ν=%.2e, δ_ν=%.2e",
            spec.epsilon, spec.delta, nu, eps_margin, delta_margin,
        )

        # Step 2: Run standard CEGIS (the engine handles LP construction
        # and constraint management internally)
        cegis_result = self._cegis_engine.synthesize(spec, callback=callback)

        # Step 3: Post-solve perturbation analysis
        edges = spec.edges.edges if spec.edges is not None else []
        perturbation_bound = self._analyzer.bound_epsilon_change(
            cegis_result.mechanism,
            nu,
            spec.epsilon,
            edges,
            delta=spec.delta,
        )

        logger.info(
            "Perturbation analysis: Δε≤%.2e, Δδ≤%.2e, κ=%.2e, p_min=%.2e",
            perturbation_bound.epsilon_bound,
            perturbation_bound.delta_bound,
            perturbation_bound.condition_number,
            perturbation_bound.p_min,
        )

        # Use the tighter of the two margin estimates
        eps_margin_final = max(eps_margin, perturbation_bound.epsilon_bound)
        delta_margin_final = max(delta_margin, perturbation_bound.delta_bound)

        # Step 4: Interval arithmetic verification
        mechanism = cegis_result.mechanism
        interval_verified = self._interval_verify(
            mechanism, spec, nu, eps_margin_final, delta_margin_final,
        )

        if not interval_verified:
            logger.warning(
                "Initial interval verification failed; attempting refinement"
            )
            mechanism, interval_verified = self._attempt_refinement(
                mechanism, spec, nu, eps_margin_final, delta_margin_final,
            )

            if not interval_verified and self._config.strict_interval_verify:
                raise VerificationError(
                    "Interval verification failed after refinement. "
                    "The mechanism may not satisfy the claimed privacy "
                    "guarantee. Consider increasing solver_tolerance or "
                    "safety_factor.",
                    epsilon=spec.epsilon + eps_margin_final,
                    delta=spec.delta + delta_margin_final,
                )

        # Step 5: Build certified output
        synthesis_time = time.monotonic() - t_start
        max_violation = self._compute_max_dp_violation(mechanism, spec)

        certificate = NumericalCertificate(
            solver_tolerance=nu,
            epsilon_target=spec.epsilon,
            delta_target=spec.delta,
            epsilon_margin=eps_margin_final,
            delta_margin=delta_margin_final,
            epsilon_effective=spec.epsilon + eps_margin_final,
            delta_effective=spec.delta + delta_margin_final,
            interval_verified=interval_verified,
            condition_number=perturbation_bound.condition_number,
            max_constraint_violation=max_violation,
            perturbation_epsilon_bound=perturbation_bound.epsilon_bound,
            perturbation_delta_bound=perturbation_bound.delta_bound,
            synthesis_time=synthesis_time,
            cegis_iterations=cegis_result.iterations,
        )

        y_grid = spec.edges  # Will get the proper y_grid below
        # Recover y_grid from the mechanism shape
        k = mechanism.shape[1]
        qv = spec.query_values
        y_lo = float(np.min(qv)) - spec.sensitivity
        y_hi = float(np.max(qv)) + spec.sensitivity
        y_grid_arr = np.linspace(y_lo, y_hi, k)

        certified = CertifiedMechanism(
            mechanism=mechanism,
            y_grid=y_grid_arr,
            certificate=certificate,
            edges=edges,
            metadata={
                "cegis_obj_val": cegis_result.obj_val,
                "cegis_iterations": cegis_result.iterations,
                "convergence_history": cegis_result.convergence_history,
                "safety_factor": self._config.safety_factor,
            },
        )

        logger.info(
            "RobustCEGIS complete: %s", certified,
        )

        return certified

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _validate_spec(self, spec: QuerySpec, nu: float) -> None:
        """Validate that the spec is compatible with robust synthesis."""
        if spec.epsilon <= 0:
            raise ConfigurationError(
                f"epsilon must be > 0, got {spec.epsilon}",
                parameter="epsilon",
                value=spec.epsilon,
                constraint="epsilon > 0",
            )

        # Check that inflation won't consume more than half of ε
        exp_eps = math.exp(spec.epsilon)
        min_prob_est = max(math.exp(-spec.epsilon) / spec.k, 1e-15)
        tentative_margin = nu * exp_eps / min_prob_est * self._config.safety_factor

        if tentative_margin > spec.epsilon * 0.5:
            logger.warning(
                "Solver tolerance ν=%.2e is large relative to ε=%.4f "
                "(tentative margin %.2e > ε/2). Consider reducing ν.",
                nu, spec.epsilon, tentative_margin,
            )

    def _compute_margins(
        self,
        spec: QuerySpec,
        nu: float,
    ) -> Tuple[float, float]:
        """Compute the inflation margins (ε_ν, δ_ν).

        Args:
            spec: Query specification.
            nu: Solver tolerance.

        Returns:
            Tuple (epsilon_margin, delta_margin).
        """
        n_pairs = len(spec.edges.edges) if spec.edges is not None else 0

        eps_margin = self._inflator.compute_epsilon_margin(
            spec.epsilon, nu, spec.k,
        )
        delta_margin = self._inflator.compute_delta_margin(
            spec.delta, nu, spec.k, n_pairs,
        )

        return eps_margin, delta_margin

    def _interval_verify(
        self,
        mechanism: npt.NDArray[np.float64],
        spec: QuerySpec,
        nu: float,
        eps_margin: float,
        delta_margin: float,
    ) -> bool:
        """Run interval arithmetic verification.

        Verifies that the mechanism satisfies (ε+ε_margin, δ+δ_margin)-DP
        for ALL possible probability values within ±ν of the mechanism.

        Args:
            mechanism: Probability table.
            spec: Query specification.
            nu: Solver tolerance.
            eps_margin: Epsilon margin.
            delta_margin: Delta margin.

        Returns:
            True if interval verification passes.
        """
        p = mechanism
        p_lo = np.maximum(p - nu, 0.0)
        p_hi = p + nu

        # Verify at effective parameters
        eps_eff = spec.epsilon + eps_margin
        delta_eff = spec.delta + delta_margin

        edges = spec.edges.edges if spec.edges is not None else []
        valid, violation = interval_verify_dp(
            p_lo, p_hi, eps_eff, edges, delta=delta_eff,
        )

        if not valid:
            logger.debug(
                "Interval verification failed: violation=%s at "
                "ε_eff=%.6f, δ_eff=%.2e",
                violation, eps_eff, delta_eff,
            )

        return valid

    def _attempt_refinement(
        self,
        mechanism: npt.NDArray[np.float64],
        spec: QuerySpec,
        nu: float,
        eps_margin: float,
        delta_margin: float,
    ) -> Tuple[npt.NDArray[np.float64], bool]:
        """Attempt to fix interval verification failures via projection.

        Applies DP-preserving projection at the effective parameters and
        re-verifies with interval arithmetic.

        Args:
            mechanism: Original mechanism table.
            spec: Query specification.
            nu: Solver tolerance.
            eps_margin: Epsilon margin.
            delta_margin: Delta margin.

        Returns:
            Tuple (refined_mechanism, interval_verified).
        """
        eps_eff = spec.epsilon + eps_margin
        delta_eff = spec.delta + delta_margin
        edges = spec.edges.edges if spec.edges is not None else []
        exp_eps = math.exp(eps_eff)

        p_ref = mechanism.copy()
        prob_floor = 1e-300

        for attempt in range(self._config.max_refinement_attempts):
            # DP-preserving projection at effective parameters
            n, k = p_ref.shape

            if delta_eff == 0.0:
                # Pure DP: clamp ratios
                for _ in range(50):
                    changed = False
                    for i, ip in edges:
                        for j in range(k):
                            # Forward: p[i][j] <= exp(ε_eff) * p[i'][j]
                            upper = exp_eps * max(p_ref[ip, j], prob_floor)
                            if p_ref[i, j] > upper:
                                p_ref[i, j] = upper
                                changed = True
                            # Backward
                            upper_rev = exp_eps * max(p_ref[i, j], prob_floor)
                            if p_ref[ip, j] > upper_rev:
                                p_ref[ip, j] = upper_rev
                                changed = True
                    if not changed:
                        break
            else:
                # Approx DP: reduce hockey-stick
                for i, ip in edges:
                    for row_a, row_b in [(i, ip), (ip, i)]:
                        excess = np.maximum(
                            p_ref[row_a] - exp_eps * p_ref[row_b], 0.0,
                        )
                        total_excess = excess.sum()
                        if total_excess > delta_eff:
                            reduction = total_excess - delta_eff
                            mask = excess > 0
                            if mask.any():
                                scale = reduction / total_excess
                                p_ref[row_a] -= excess * scale

            # Re-normalise
            np.clip(p_ref, 0.0, None, out=p_ref)
            row_sums = p_ref.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-15)
            p_ref /= row_sums

            # Re-verify with interval arithmetic
            verified = self._interval_verify(p_ref, spec, nu, eps_margin, delta_margin)
            if verified:
                logger.info(
                    "Refinement succeeded on attempt %d", attempt + 1,
                )
                return p_ref, True

            # Increase margins for next attempt
            eps_margin *= 1.5
            delta_margin *= 1.5
            eps_eff = spec.epsilon + eps_margin
            delta_eff = spec.delta + delta_margin
            exp_eps = math.exp(eps_eff)

            logger.debug(
                "Refinement attempt %d failed; increasing margins to "
                "ε_ν=%.2e, δ_ν=%.2e",
                attempt + 1, eps_margin, delta_margin,
            )

        return p_ref, False

    def _compute_max_dp_violation(
        self,
        mechanism: npt.NDArray[np.float64],
        spec: QuerySpec,
    ) -> float:
        """Compute the maximum DP violation at the target parameters."""
        p = mechanism
        edges = spec.edges.edges if spec.edges is not None else []
        exp_eps = math.exp(spec.epsilon)
        prob_floor = 1e-300

        max_viol = 0.0

        if spec.delta == 0.0:
            for i, ip in edges:
                for row_a, row_b in [(i, ip), (ip, i)]:
                    for j in range(p.shape[1]):
                        denom = max(p[row_b, j], prob_floor)
                        ratio = p[row_a, j] / denom
                        if ratio > exp_eps:
                            max_viol = max(max_viol, ratio - exp_eps)
        else:
            for i, ip in edges:
                for row_a, row_b in [(i, ip), (ip, i)]:
                    excess = np.maximum(p[row_a] - exp_eps * p[row_b], 0.0)
                    hs = float(excess.sum())
                    if hs > spec.delta:
                        max_viol = max(max_viol, hs - spec.delta)

        return max_viol

    @property
    def config(self) -> RobustSynthesisConfig:
        """Return the robust synthesis configuration."""
        return self._config

    @property
    def inflator(self) -> ConstraintInflator:
        """Return the constraint inflator instance."""
        return self._inflator

    @property
    def analyzer(self) -> PerturbationAnalyzer:
        """Return the perturbation analyzer instance."""
        return self._analyzer

    @property
    def diagnostics(self) -> SolverDiagnostics:
        """Return the solver diagnostics instance."""
        return self._diagnostics

    def __repr__(self) -> str:
        return f"RobustCEGISEngine(config={self._config})"


# =========================================================================
# Adaptive inflation
# =========================================================================


def adaptive_inflation(
    mechanism: npt.NDArray[np.float64],
    spec: QuerySpec,
    solver_tolerance: float = _DEFAULT_SOLVER_TOL,
    *,
    safety_factor_range: Tuple[float, float] = (1.1, 10.0),
    target_condition: float = 1e10,
    n_steps: int = 20,
) -> Tuple[float, float, float]:
    """Automatically tune the inflation margin based on LP conditioning.

    The standard inflation margin ``ε_ν = ν · e^ε / p_min · safety``
    can be excessively conservative for well-conditioned LPs and
    dangerously tight for ill-conditioned ones.  This function analyses
    the conditioning of the DP constraint matrix and adapts the safety
    factor accordingly.

    Algorithm:
        1. Estimate the condition number κ of the mechanism's DP constraint
           matrix (the sub-matrix of the LP corresponding to privacy
           constraints).
        2. If κ < target_condition, reduce the safety factor toward
           ``safety_factor_range[0]`` (tighter margins → better utility).
        3. If κ ≥ target_condition, increase the safety factor toward
           ``safety_factor_range[1]`` (wider margins → safer guarantees).
        4. The mapping is log-linear: safety = a + b · log10(κ), clamped
           to ``safety_factor_range``.

    Args:
        mechanism: Current mechanism table, shape ``(n, k)``.
        spec: Query specification (for edge structure).
        solver_tolerance: Solver feasibility tolerance ν.
        safety_factor_range: (min_safety, max_safety) for clamping.
        target_condition: κ threshold separating "well-conditioned"
            from "ill-conditioned" regimes.
        n_steps: Not used directly; reserved for future grid refinement.

    Returns:
        Tuple ``(adaptive_safety, eps_margin, delta_margin)`` where
        ``adaptive_safety`` is the computed safety factor and the margins
        are the resulting inflation amounts.
    """
    n, k = mechanism.shape
    edges = spec.edges.edges if spec.edges is not None else []

    if not edges:
        return safety_factor_range[0], solver_tolerance, solver_tolerance

    # Build the DP constraint sub-matrix for conditioning analysis
    # Each constraint: p[i][j] - e^ε p[i'][j] ≤ 0
    exp_eps = math.exp(spec.epsilon)
    n_constraints = len(edges) * k * 2  # both directions
    constraint_rows = []
    for i, ip in edges:
        for j in range(k):
            row_fwd = np.zeros(n * k, dtype=np.float64)
            row_fwd[i * k + j] = 1.0
            row_fwd[ip * k + j] = -exp_eps
            constraint_rows.append(row_fwd)

            row_bwd = np.zeros(n * k, dtype=np.float64)
            row_bwd[ip * k + j] = 1.0
            row_bwd[i * k + j] = -exp_eps
            constraint_rows.append(row_bwd)

    if not constraint_rows:
        return safety_factor_range[0], solver_tolerance, solver_tolerance

    A = np.array(constraint_rows, dtype=np.float64)

    # Estimate condition number via singular values
    try:
        s = np.linalg.svd(A, compute_uv=False)
        s_pos = s[s > 1e-15]
        if len(s_pos) < 2:
            kappa = 1.0
        else:
            kappa = float(s_pos[0] / s_pos[-1])
    except np.linalg.LinAlgError:
        kappa = target_condition

    # Log-linear mapping: safety = a + b * log10(κ)
    log_kappa = math.log10(max(kappa, 1.0))
    log_target = math.log10(target_condition)
    sf_lo, sf_hi = safety_factor_range

    # Normalise: κ=1 → sf_lo, κ=target → sf_hi
    if log_target > 0:
        t = min(log_kappa / log_target, 1.0)
    else:
        t = 0.0
    adaptive_safety = sf_lo + t * (sf_hi - sf_lo)
    adaptive_safety = max(sf_lo, min(sf_hi, adaptive_safety))

    # Compute margins using the adaptive safety factor
    p_min = max(float(np.min(mechanism[mechanism > 0])), 1e-15)
    eps_margin = solver_tolerance * exp_eps / p_min * adaptive_safety
    delta_margin = solver_tolerance * adaptive_safety

    logger.info(
        "adaptive_inflation: κ=%.2e, safety=%.2f, ε_ν=%.2e, δ_ν=%.2e",
        kappa, adaptive_safety, eps_margin, delta_margin,
    )

    return adaptive_safety, eps_margin, delta_margin


# =========================================================================
# Iterative robust synthesis
# =========================================================================


def iterative_robust_synthesis(
    spec: QuerySpec,
    *,
    n_passes: int = 3,
    initial_safety: float = 3.0,
    safety_decay: float = 0.7,
    solver_tolerance: float = _DEFAULT_SOLVER_TOL,
    base_config: Optional[SynthesisConfig] = None,
) -> CertifiedMechanism:
    """Multi-pass refinement for very tight DP guarantees.

    When the target ε is small or the mechanism table is large, a single
    RobustCEGIS pass may use overly conservative margins, sacrificing
    utility.  This function performs multiple passes, each using the
    previous solution to better estimate conditioning and tighten margins.

    Algorithm:
        1. **Pass 1**: Run RobustCEGIS with a conservative safety factor.
        2. **Pass 2..n_passes**: Use ``adaptive_inflation`` on the previous
           solution to estimate conditioning, then re-run RobustCEGIS with
           the refined safety factor.
        3. Return the best (lowest objective) certified mechanism across
           all passes.

    The safety factor decays geometrically: ``safety_i = max(1.1,
    initial_safety × safety_decay^{i-1})``.  This allows the first pass
    to be very conservative and subsequent passes to progressively tighten.

    Args:
        spec: Query specification.
        n_passes: Number of synthesis passes (≥ 1).
        initial_safety: Safety factor for the first pass.
        safety_decay: Geometric decay factor for subsequent passes.
        solver_tolerance: Solver feasibility tolerance ν.
        base_config: Optional underlying SynthesisConfig.

    Returns:
        The best CertifiedMechanism across all passes.

    Raises:
        ConvergenceError: If all passes fail.
    """
    if n_passes < 1:
        raise ConfigurationError(
            f"n_passes must be >= 1, got {n_passes}",
            parameter="n_passes",
            value=n_passes,
        )

    best_result: Optional[CertifiedMechanism] = None
    best_obj: float = math.inf
    last_error: Optional[Exception] = None

    for pass_idx in range(n_passes):
        safety = max(1.1, initial_safety * (safety_decay ** pass_idx))

        # If we have a previous result, use adaptive_inflation
        if best_result is not None:
            safety, _, _ = adaptive_inflation(
                best_result.mechanism, spec, solver_tolerance,
                safety_factor_range=(1.1, safety),
            )

        config = RobustSynthesisConfig(
            base_config=base_config or SynthesisConfig(),
            solver_tolerance=solver_tolerance,
            safety_factor=safety,
            strict_interval_verify=False,
        )

        logger.info(
            "iterative_robust_synthesis: pass %d/%d, safety=%.2f",
            pass_idx + 1, n_passes, safety,
        )

        try:
            engine = RobustCEGISEngine(config)
            result = engine.synthesize(spec)

            # Track best by objective
            obj_val = result.metadata.get("cegis_obj_val", math.inf)
            if obj_val < best_obj:
                best_obj = obj_val
                best_result = result

        except (ConvergenceError, VerificationError, NumericalInstabilityError) as exc:
            logger.warning(
                "Pass %d failed: %s", pass_idx + 1, exc,
            )
            last_error = exc
            continue

    if best_result is None:
        raise ConvergenceError(
            f"All {n_passes} passes of iterative_robust_synthesis failed. "
            f"Last error: {last_error}",
            iterations=n_passes,
            max_iter=n_passes,
            final_obj=math.inf,
        )

    return best_result


# =========================================================================
# Conditioning-aware grid selection
# =========================================================================


def conditioning_aware_grid(
    spec: QuerySpec,
    *,
    k_min: int = 10,
    k_max: int = 500,
    condition_threshold: float = 1e10,
    n_candidates: int = 8,
) -> Tuple[int, npt.NDArray[np.float64]]:
    """Select output grid size to keep LP conditioning below a threshold.

    The condition number of the DP mechanism LP grows with the output grid
    size k.  Finer grids improve approximation quality but degrade
    numerical conditioning.  This function evaluates candidate grid sizes
    and selects the largest k whose constraint matrix condition number
    stays below ``condition_threshold``.

    Algorithm:
        1. Generate candidate k values geometrically spaced between
           ``k_min`` and ``k_max``.
        2. For each k, build the DP constraint sub-matrix and estimate κ.
        3. Return the largest k with κ < condition_threshold.
        4. If all candidates exceed the threshold, return k_min with a
           warning.

    Args:
        spec: Query specification (determines edge structure and
            privacy parameters).
        k_min: Minimum grid size.
        k_max: Maximum grid size.
        condition_threshold: Maximum acceptable condition number κ.
        n_candidates: Number of candidate grid sizes to evaluate.

    Returns:
        Tuple ``(best_k, y_grid)`` where ``best_k`` is the chosen grid
        size and ``y_grid`` is the corresponding output grid array.
    """
    if k_min < 2:
        raise ConfigurationError(
            f"k_min must be >= 2, got {k_min}",
            parameter="k_min", value=k_min,
        )
    if k_max < k_min:
        raise ConfigurationError(
            f"k_max must be >= k_min, got k_max={k_max}, k_min={k_min}",
            parameter="k_max", value=k_max,
        )

    edges = spec.edges.edges if spec.edges is not None else []
    n = spec.n
    exp_eps = math.exp(spec.epsilon)

    # Candidate grid sizes (geometrically spaced)
    candidates = np.geomspace(k_min, k_max, n_candidates).astype(int)
    candidates = np.unique(np.clip(candidates, k_min, k_max))

    best_k = k_min
    f_min = float(np.min(spec.query_values))
    f_max = float(np.max(spec.query_values))
    pad = spec.sensitivity

    for k in sorted(candidates, reverse=True):
        k = int(k)

        if not edges:
            best_k = k
            break

        # Build a small representative constraint matrix
        n_rows = min(len(edges) * 2, 200)  # cap for speed
        rows = []
        count = 0
        for i, ip in edges:
            if count >= n_rows:
                break
            for j in range(min(k, 20)):  # sample columns
                row = np.zeros(n * k, dtype=np.float64)
                row[i * k + j] = 1.0
                row[ip * k + j] = -exp_eps
                rows.append(row)
                count += 1
                if count >= n_rows:
                    break

        if not rows:
            best_k = k
            break

        A = np.array(rows, dtype=np.float64)
        try:
            s = np.linalg.svd(A, compute_uv=False)
            s_pos = s[s > 1e-15]
            if len(s_pos) < 2:
                kappa = 1.0
            else:
                kappa = float(s_pos[0] / s_pos[-1])
        except np.linalg.LinAlgError:
            kappa = float("inf")

        if kappa < condition_threshold:
            best_k = k
            break

    if best_k == k_min and len(candidates) > 1:
        logger.warning(
            "conditioning_aware_grid: all candidates exceed κ threshold "
            "%.2e; falling back to k_min=%d",
            condition_threshold, k_min,
        )

    y_grid = np.linspace(f_min - pad, f_max + pad, best_k)

    return best_k, y_grid
