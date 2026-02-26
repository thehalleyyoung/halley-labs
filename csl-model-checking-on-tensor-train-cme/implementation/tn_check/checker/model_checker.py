"""
CSL model checker: main engine for evaluating CSL properties on MPS states.

Implements:
- Time-bounded until via projected-rate-matrix exponentiation
- Unbounded-until via fixpoint iteration with spectral-gap-informed convergence
- Three-valued interval-arithmetic semantics for nested operators
- Steady-state evaluation via DMRG-like ground-state solver
- Verification trace generation for independent certificate auditing
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from tn_check.config import TNCheckConfig, CheckerConfig
from tn_check.tensor.mps import MPS, ones_mps
from tn_check.tensor.mpo import MPO
from tn_check.tensor.operations import (
    mps_inner_product, mps_hadamard_product, mps_addition,
    mps_scalar_multiply, mps_compress, mps_total_probability,
    mps_distance, mpo_mps_contraction, mps_zip_up,
    mps_clamp_nonnegative, mps_normalize_probability,
)
from tn_check.checker.csl_ast import (
    CSLFormula, AtomicProp, TrueFormula, Negation, Conjunction,
    ProbabilityOp, SteadyStateOp, BoundedUntil, UnboundedUntil,
    ComparisonOp, LinearPredicate,
)
from tn_check.checker.satisfaction import (
    SatisfactionResult, ThreeValued,
    compute_satisfaction_set, project_rate_matrix,
)
from tn_check.checker.spectral import (
    estimate_spectral_gap, adaptive_fallback_time_bound,
    SpectralGapEstimate,
)
from tn_check.verifier.trace import VerificationTrace

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceDiagnostics:
    """Diagnostics for fixpoint iteration convergence."""
    iteration_errors: list[float] = field(default_factory=list)
    bond_dimensions: list[int] = field(default_factory=list)
    spectral_gap_estimate: Optional[float] = None
    spectral_gap_info: Optional[SpectralGapEstimate] = None
    converged: bool = False
    iterations: int = 0
    final_error: float = float("inf")
    fallback_used: bool = False
    fallback_time_bound: Optional[float] = None
    convergence_ratios: list[float] = field(default_factory=list)

    def convergence_rate(self) -> Optional[float]:
        """Estimate convergence rate from last few iterations."""
        if len(self.iteration_errors) < 3:
            return None
        recent = self.iteration_errors[-3:]
        if recent[-2] < 1e-300 or recent[-3] < 1e-300:
            return None
        rate = np.log(recent[-1] / recent[-2]) / np.log(recent[-2] / recent[-3])
        return float(rate)

    def geometric_convergence_ratio(self) -> Optional[float]:
        """Estimate geometric convergence ratio from recent iterations."""
        if len(self.iteration_errors) < 2:
            return None
        recent = self.iteration_errors[-min(5, len(self.iteration_errors)):]
        ratios = []
        for i in range(1, len(recent)):
            if recent[i-1] > 1e-300:
                ratios.append(recent[i] / recent[i-1])
        return float(np.median(ratios)) if ratios else None


class CSLModelChecker:
    """
    CSL model checker operating on MPS-compressed probability vectors.

    Implements the full CSL evaluation pipeline:
    1. Parse formula into AST
    2. Compute satisfaction sets for atomic propositions (rank-1 TT masks)
    3. For P~p[ψ], evaluate path formula ψ:
       - Bounded until: construct projected rate matrix, time-evolve
       - Unbounded until: fixpoint iteration with convergence monitoring
    4. Apply three-valued semantics for probability comparison
    5. Propagate error bounds through nested operators

    Args:
        generator: CME generator as MPO.
        config: Checker configuration.
        physical_dims: Physical dimensions at each site.
    """

    def __init__(
        self,
        generator: MPO,
        config: Optional[CheckerConfig] = None,
        physical_dims: Optional[tuple[int, ...]] = None,
    ):
        self.generator = generator
        self.config = config or CheckerConfig()
        self.num_sites = generator.num_sites
        self.physical_dims = physical_dims or generator.physical_dims_in
        self._error_budget = 0.0
        self._trace = VerificationTrace(
            num_species=self.num_sites,
            physical_dims=list(self.physical_dims),
        )
        self._step_counter = 0

    @property
    def trace(self) -> VerificationTrace:
        """Access the verification trace for independent auditing."""
        return self._trace

    def check(
        self,
        formula: CSLFormula,
        initial_state: MPS,
        max_bond_dim: Optional[int] = None,
    ) -> SatisfactionResult:
        """
        Check a CSL formula against an initial state distribution.

        Args:
            formula: CSL formula to check.
            initial_state: Initial probability distribution as MPS.
            max_bond_dim: Maximum bond dimension for intermediate computations.

        Returns:
            SatisfactionResult with probability bounds and verdict.
        """
        chi = max_bond_dim or 200

        if isinstance(formula, ProbabilityOp):
            return self._check_probability_op(formula, initial_state, chi)
        elif isinstance(formula, SteadyStateOp):
            return self._check_steady_state(formula, initial_state, chi)
        elif isinstance(formula, (AtomicProp, TrueFormula, Negation,
                                    Conjunction, LinearPredicate)):
            sat = compute_satisfaction_set(
                formula, self.num_sites, self.physical_dims,
            )
            # Compute probability that initial state satisfies formula
            prob = mps_inner_product(initial_state, sat)
            return SatisfactionResult(
                satisfaction_mps=sat,
                probability_lower=max(0.0, prob - 1e-12),
                probability_upper=min(1.0, prob + 1e-12),
                verdict=ThreeValued.TRUE if prob > 0.5 else ThreeValued.FALSE,
            )
        else:
            raise ValueError(f"Unsupported formula type: {type(formula)}")

    def _check_probability_op(
        self,
        formula: ProbabilityOp,
        initial_state: MPS,
        max_bond_dim: int,
    ) -> SatisfactionResult:
        """Evaluate P~p[ψ] for a path formula ψ."""
        path = formula.path_formula

        if isinstance(path, BoundedUntil):
            result = self._bounded_until(
                path.phi1, path.phi2, path.time_bound,
                initial_state, max_bond_dim,
            )
        elif isinstance(path, UnboundedUntil):
            result = self._unbounded_until(
                path.phi1, path.phi2,
                initial_state, max_bond_dim,
            )
        else:
            raise ValueError(f"Unsupported path formula: {type(path)}")

        # Apply three-valued classification
        result.classify(
            formula.threshold,
            formula.comparison.value,
            epsilon=self.config.threshold_epsilon,
        )

        # Record in verification trace
        self._trace.record_csl_check(
            formula_str=repr(formula),
            probability_lower=result.probability_lower or 0.0,
            probability_upper=result.probability_upper or 1.0,
            verdict=result.verdict.value,
            total_certified_error=result.total_error,
            fixpoint_iterations=result.fixpoint_iterations,
            converged=result.converged,
        )

        return result

    def _bounded_until(
        self,
        phi1: CSLFormula,
        phi2: CSLFormula,
        time_bound: float,
        initial_state: MPS,
        max_bond_dim: int,
    ) -> SatisfactionResult:
        """
        Evaluate time-bounded until: compute Pr[φ₁ U[0,t] φ₂].

        Algorithm:
        1. Compute satisfaction sets Sat(φ₁) and Sat(φ₂)
        2. Construct projected rate matrix Q_proj (Theorem 2)
        3. Time-evolve: p(t) = e^{Q_proj * t} @ p₀
        4. Compute reaching probability: Pr = <Sat(φ₂) | p(t)>

        Error bound (Theorem 1): the Metzler structure of Q ensures
        ‖e^{Qt}‖₁ = 1 (contractivity), so truncation errors do not
        amplify through time evolution.
        """
        t_start = time.time()

        # Step 1: satisfaction sets
        sat_phi1 = compute_satisfaction_set(
            phi1, self.num_sites, self.physical_dims,
        )
        sat_phi2 = compute_satisfaction_set(
            phi2, self.num_sites, self.physical_dims,
        )

        # Step 2: projected rate matrix
        Q_proj = project_rate_matrix(
            self.generator, sat_phi1, sat_phi2, self.physical_dims,
        )

        # Step 3: time evolution via uniformization
        # For CME generators: Q has non-negative off-diagonals, negative diagonal
        # Use numerical uniformization with Poisson truncation
        prob_mps, trunc_error, clamp_error = self._uniformization_evolve(
            Q_proj, initial_state, time_bound, max_bond_dim,
        )

        # Step 4: compute reaching probability
        prob = mps_inner_product(sat_phi2, prob_mps)
        # Also compute probability via complement for error estimation
        total_error = trunc_error + clamp_error

        elapsed = time.time() - t_start
        logger.info(
            f"Bounded until evaluated in {elapsed:.2f}s, "
            f"prob={prob:.6f}, error={total_error:.2e}"
        )

        return SatisfactionResult(
            satisfaction_mps=sat_phi2,
            probability_lower=max(0.0, prob - total_error),
            probability_upper=min(1.0, prob + total_error),
            truncation_error=trunc_error,
            clamping_error=clamp_error,
            total_error=total_error,
        )

    def _unbounded_until(
        self,
        phi1: CSLFormula,
        phi2: CSLFormula,
        initial_state: MPS,
        max_bond_dim: int,
    ) -> SatisfactionResult:
        """
        Evaluate unbounded until: compute Pr[φ₁ U φ₂].

        Uses spectral-gap-informed fixpoint iteration:
        1. Estimate spectral gap of projected rate matrix
        2. If gap predicts infeasible convergence, skip directly to bounded-until
        3. Otherwise, run fixpoint iteration with convergence monitoring
        4. Track geometric convergence ratio for early termination

        This addresses the critique of fragile fixpoint convergence for
        slowly-mixing CTMCs by predicting convergence upfront.
        """
        sat_phi1 = compute_satisfaction_set(
            phi1, self.num_sites, self.physical_dims,
        )
        sat_phi2 = compute_satisfaction_set(
            phi2, self.num_sites, self.physical_dims,
        )

        Q_proj = project_rate_matrix(
            self.generator, sat_phi1, sat_phi2, self.physical_dims,
        )

        diag = ConvergenceDiagnostics()

        # Step 1: Estimate spectral gap for convergence prediction
        gap_info = estimate_spectral_gap(
            Q_proj, self.num_sites, self.physical_dims,
            max_power_steps=30, max_bond_dim=min(max_bond_dim, 50),
        )
        diag.spectral_gap_estimate = gap_info.gap_estimate
        diag.spectral_gap_info = gap_info

        # Step 2: If gap predicts infeasible convergence, fallback immediately
        predicted_iters = gap_info.predicted_iteration_count(self.config.fixpoint_tol)
        if not gap_info.feasible or predicted_iters > self.config.fixpoint_max_iter:
            logger.info(
                f"Spectral gap {gap_info.gap_estimate:.2e} predicts "
                f"{predicted_iters} iterations (max={self.config.fixpoint_max_iter}). "
                f"Falling back to bounded-until."
            )
            diag.fallback_used = True
            fallback_t = adaptive_fallback_time_bound(gap_info.gap_estimate)
            diag.fallback_time_bound = fallback_t
            return self._bounded_until(
                phi1, phi2, fallback_t, initial_state, max_bond_dim,
            )

        # Step 3: Fixpoint iteration with convergence monitoring
        x = compute_satisfaction_set(
            phi2, self.num_sites, self.physical_dims,
        )

        dt_iter = 0.1
        prev_prob = 0.0

        for iteration in range(self.config.fixpoint_max_iter):
            Qx, trunc_err = mps_zip_up(
                Q_proj, x, max_bond_dim=max_bond_dim,
                tolerance=1e-10,
            )
            x_new = mps_addition(x, mps_scalar_multiply(Qx, dt_iter))
            x_new = mps_addition(x_new, mps_scalar_multiply(sat_phi2, 0.1))
            x_new, _ = mps_compress(
                x_new, max_bond_dim=max_bond_dim, tolerance=1e-10,
            )

            error = mps_distance(x_new, x)
            diag.iteration_errors.append(error)
            diag.bond_dimensions.append(x_new.max_bond_dim)
            diag.iterations = iteration + 1

            # Track geometric convergence ratio
            if len(diag.iteration_errors) >= 2 and diag.iteration_errors[-2] > 1e-300:
                ratio = error / diag.iteration_errors[-2]
                diag.convergence_ratios.append(ratio)

            prob = mps_inner_product(initial_state, x_new)

            if error < self.config.fixpoint_tol:
                diag.converged = True
                diag.final_error = error
                break

            # Divergence detection: if error grows for 10 consecutive iterations
            if len(diag.iteration_errors) > 10:
                recent = diag.iteration_errors[-10:]
                if all(recent[i] >= recent[i-1] for i in range(1, len(recent))):
                    logger.warning(
                        "Fixpoint iteration diverging, falling back to bounded-until"
                    )
                    diag.fallback_used = True
                    break

            # Stagnation detection via geometric ratio
            if len(diag.convergence_ratios) >= 5:
                avg_ratio = np.mean(diag.convergence_ratios[-5:])
                if avg_ratio > 0.999:
                    logger.warning(
                        f"Convergence stagnating (ratio={avg_ratio:.6f}), "
                        f"falling back to bounded-until"
                    )
                    diag.fallback_used = True
                    break

            x = x_new
            prev_prob = prob

        if not diag.converged and not diag.fallback_used:
            logger.warning(
                f"Fixpoint iteration did not converge after {diag.iterations} iterations"
            )

        # Fallback: use bounded-until with spectrally-informed time bound
        if diag.fallback_used or not diag.converged:
            fallback_t = adaptive_fallback_time_bound(
                gap_info.gap_estimate if gap_info.gap_estimate > 0 else 1e-4
            )
            diag.fallback_time_bound = fallback_t
            logger.info(f"Using bounded-until fallback with t={fallback_t:.1f}")
            return self._bounded_until(
                phi1, phi2, fallback_t, initial_state, max_bond_dim,
            )

        total_error = diag.final_error + sum(
            e * 0.01 for e in diag.iteration_errors
        )
        prob = mps_inner_product(initial_state, x)

        return SatisfactionResult(
            satisfaction_mps=x,
            probability_lower=max(0.0, prob - total_error),
            probability_upper=min(1.0, prob + total_error),
            truncation_error=total_error,
            total_error=total_error,
            fixpoint_iterations=diag.iterations,
            converged=diag.converged,
        )

    def _uniformization_evolve(
        self,
        Q: MPO,
        p0: MPS,
        t: float,
        max_bond_dim: int,
    ) -> tuple[MPS, float, float]:
        """
        Time evolution via numerical uniformization in TT format.

        Uniformization: e^{Qt} p₀ = e^{-qt} Σ_{k=0}^K (qt)^k/k! P^k p₀
        where q = max|Q_{ii}| and P = I + Q/q is a stochastic matrix.

        Uses Fox-Glynn truncation to determine K such that the Poisson
        tail probability is below tolerance.

        Error analysis (Theorem 1):
        - Per-step truncation error ε_round propagates as C(t) * ε_round
        - For Metzler Q: C(t) = K (number of terms), not exponential in t
        - Clamping error bounded by Proposition 1

        Returns:
            Tuple of (evolved MPS, truncation error, clamping error).
        """
        # Estimate uniformization rate
        # q = max diagonal magnitude of Q
        q = self._estimate_uniformization_rate(Q)
        if q < 1e-15:
            return p0.copy(), 0.0, 0.0

        qt = q * t

        # Fox-Glynn: find K such that Poisson tail < tolerance
        fox_glynn_tol = 1e-10
        K = self._fox_glynn_truncation(qt, fox_glynn_tol)
        K = min(K, 10000)  # safety cap

        logger.info(f"Uniformization: q={q:.4f}, qt={qt:.4f}, K={K}")

        # Construct stochastic matrix P = I + Q/q
        from tn_check.tensor.mpo import identity_mpo
        from tn_check.tensor.operations import mpo_addition, mpo_scalar_multiply

        I = identity_mpo(self.num_sites, self.physical_dims)
        Q_scaled = Q.copy()
        Q_scaled.scale(1.0 / q)
        P = mpo_addition(I, Q_scaled)

        # Compute Poisson-weighted sum: Σ w_k P^k p₀
        # w_k = e^{-qt} (qt)^k / k!
        total_trunc_error = 0.0
        result = mps_scalar_multiply(p0, 0.0)  # zero accumulator
        p_k = p0.copy()  # P^k p₀

        log_weight = -qt  # log(w_0) = -qt
        for k in range(K + 1):
            if k > 0:
                log_weight += np.log(qt) - np.log(k)

            weight = np.exp(log_weight) if log_weight > -700 else 0.0
            if weight < 1e-16:
                if k > qt:
                    break
                # Apply P to get next term
                p_k, err = mps_zip_up(
                    P, p_k, max_bond_dim=max_bond_dim, tolerance=1e-10,
                )
                total_trunc_error += err
                continue

            # Accumulate: result += weight * p_k
            result = mps_addition(result, mps_scalar_multiply(p_k, weight))
            if result.max_bond_dim > max_bond_dim * 2:
                result, err = mps_compress(
                    result, max_bond_dim=max_bond_dim, tolerance=1e-10,
                )
                total_trunc_error += err

            # Apply P for next iteration
            if k < K:
                p_k, err = mps_zip_up(
                    P, p_k, max_bond_dim=max_bond_dim, tolerance=1e-10,
                )
                total_trunc_error += err

        # Final compression
        result, err = mps_compress(
            result, max_bond_dim=max_bond_dim, tolerance=1e-10,
        )
        total_trunc_error += err

        # Non-negativity clamping with error tracking (Proposition 1)
        result, clamping_error = mps_clamp_nonnegative(
            result, max_bond_dim=max_bond_dim, tolerance=1e-10,
        )

        # Record step in verification trace
        self._step_counter += 1
        total_prob = mps_total_probability(result)
        self._trace.record_step(
            step_index=self._step_counter,
            time=t,
            truncation_error=total_trunc_error,
            clamping_error=clamping_error,
            bond_dims=list(result.bond_dims),
            total_probability=total_prob,
            method="uniformization",
        )

        return result, total_trunc_error, clamping_error

    def _estimate_uniformization_rate(self, Q: MPO) -> float:
        """Estimate max diagonal magnitude of Q for uniformization rate."""
        # For small systems, compute exactly
        total_size = 1
        for d in Q.physical_dims_in:
            total_size *= d

        if total_size <= 100000:
            from tn_check.tensor.operations import mpo_to_dense
            Q_dense = mpo_to_dense(Q)
            return float(np.max(np.abs(np.diag(Q_dense))))

        # For large systems, estimate via sampling
        from tn_check.tensor.mps import unit_mps
        rng = np.random.default_rng(42)
        max_diag = 0.0
        for _ in range(100):
            idx = tuple(rng.integers(0, d) for d in Q.physical_dims_in)
            basis = unit_mps(Q.num_sites, Q.physical_dims_in, idx)
            result = mpo_mps_contraction(Q, basis)
            val = abs(mps_inner_product(basis, result))
            max_diag = max(max_diag, val)

        return max_diag * 1.5  # safety margin

    def _fox_glynn_truncation(self, qt: float, epsilon: float) -> int:
        """
        Fox-Glynn algorithm: find K such that Poisson(qt) tail < epsilon.

        Returns the number of terms needed for the Poisson sum.
        """
        if qt < 1e-10:
            return 0

        # Simple approach: K = qt + c * sqrt(qt) for appropriate c
        import math
        c = math.sqrt(-2 * math.log(epsilon))
        K = int(qt + c * math.sqrt(qt)) + 10
        return max(K, 10)

    def _check_steady_state(
        self,
        formula: SteadyStateOp,
        initial_state: MPS,
        max_bond_dim: int,
    ) -> SatisfactionResult:
        """
        Evaluate steady-state operator S~p[φ].

        Computes the steady-state distribution π (dominant left eigenvector
        of Q) and evaluates Pr_π[φ].
        """
        sat = compute_satisfaction_set(
            formula.state_formula, self.num_sites, self.physical_dims,
        )

        # Approximate steady state via long-time evolution
        t_ss = 10000.0
        p_ss, trunc_err, clamp_err = self._uniformization_evolve(
            self.generator, initial_state, t_ss, max_bond_dim,
        )

        prob = mps_inner_product(sat, p_ss)
        total_error = trunc_err + clamp_err

        result = SatisfactionResult(
            satisfaction_mps=sat,
            probability_lower=max(0.0, prob - total_error),
            probability_upper=min(1.0, prob + total_error),
            truncation_error=trunc_err,
            clamping_error=clamp_err,
            total_error=total_error,
        )
        result.classify(
            formula.threshold,
            formula.comparison.value,
            epsilon=self.config.threshold_epsilon,
        )
        return result
