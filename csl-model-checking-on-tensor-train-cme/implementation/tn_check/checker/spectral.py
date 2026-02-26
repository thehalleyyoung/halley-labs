"""
Spectral gap estimation for projected rate matrices.

Provides convergence prediction for fixpoint iteration in unbounded-until
CSL model checking. For slowly-mixing CTMCs with small spectral gaps,
the fixpoint requires many iterations (K ≈ log(1/ε)/gap); this module
estimates the gap upfront to decide whether fixpoint iteration is feasible
or whether to fall back to bounded-until.

The spectral gap of a rate matrix Q (smallest nonzero eigenvalue magnitude
of -Q restricted to transient states) determines the mixing time:
    t_mix = Θ(1 / gap)

For the projected rate matrix Q_proj used in unbounded-until, the gap
determines fixpoint convergence rate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from tn_check.tensor.mps import MPS
from tn_check.tensor.mpo import MPO
from tn_check.tensor.operations import (
    mpo_mps_contraction, mps_compress, mps_inner_product,
    mps_distance, mps_normalize_probability, mps_norm,
    mps_expectation_value,
)

logger = logging.getLogger(__name__)


@dataclass
class SpectralGapEstimate:
    """Result of spectral gap estimation."""
    gap_estimate: float
    confidence: str  # "high", "medium", "low"
    estimated_mixing_time: float
    predicted_iterations: int
    feasible: bool
    method: str
    power_iteration_steps: int = 0
    convergence_ratio: float = 0.0

    def predicted_iteration_count(self, tolerance: float) -> int:
        """Predict iterations needed for fixpoint convergence."""
        if self.gap_estimate < 1e-15:
            return 999999
        return int(np.ceil(np.log(1.0 / max(tolerance, 1e-15)) / self.gap_estimate)) + 1


def rayleigh_quotient_refinement(Q: MPO, v: MPS) -> float:
    """
    Refine an eigenvalue estimate via the Rayleigh quotient.

    Computes ρ = <v|Q|v> / <v|v> for an approximate eigenvector v.

    Args:
        Q: Rate matrix as MPO.
        v: Approximate eigenvector as MPS.

    Returns:
        Rayleigh quotient (eigenvalue estimate).
    """
    vv = mps_inner_product(v, v)
    if abs(vv) < 1e-300:
        return 0.0
    vQv = mps_expectation_value(Q, v)
    return float(vQv / vv)


def estimate_spectral_gap(
    Q_proj: MPO,
    num_sites: int,
    physical_dims,
    max_power_steps: int = 50,
    max_bond_dim: int = 100,
    tolerance: float = 1e-8,
) -> SpectralGapEstimate:
    """
    Estimate the spectral gap of a projected rate matrix via power iteration.

    The spectral gap is |λ_2| where λ_1 = 0 is the largest eigenvalue
    (stationary state) and λ_2 is the second largest. For a Metzler
    matrix Q with zero column sums, all eigenvalues have non-positive
    real parts.

    Algorithm:
    1. Start with random MPS vector v₀
    2. Apply e^{Q·dt} repeatedly via MPO-MPS contraction
    3. After removing the stationary component, track the decay rate
    4. The decay rate of ‖v_k‖ estimates the spectral gap

    For CME generators, we use the inverse power method on (Q + μI)
    where μ is chosen to shift the spectrum.

    Args:
        Q_proj: Projected rate matrix as MPO.
        num_sites: Number of sites.
        physical_dims: Physical dimensions.
        max_power_steps: Maximum power iteration steps.
        max_bond_dim: Bond dimension for intermediate MPS.
        tolerance: Convergence tolerance.

    Returns:
        SpectralGapEstimate with gap and feasibility assessment.
    """
    if isinstance(physical_dims, int):
        phys = tuple([physical_dims] * num_sites)
    else:
        phys = tuple(physical_dims)

    # Initialize random MPS
    rng = np.random.default_rng(42)
    cores = []
    chi = min(5, max_bond_dim)
    for k in range(num_sites):
        d = phys[k]
        chi_l = 1 if k == 0 else chi
        chi_r = 1 if k == num_sites - 1 else chi
        core = rng.standard_normal((chi_l, d, chi_r)) * 0.01
        cores.append(core)
    v = MPS(cores, copy_cores=False)

    # Normalize
    v_norm = mps_norm(v)
    if v_norm > 1e-300:
        from tn_check.tensor.operations import mps_scalar_multiply
        v = mps_scalar_multiply(v, 1.0 / v_norm)

    # Power iteration: apply Q repeatedly and track decay
    norms = []
    ratios = []
    dt_step = 0.1  # small time step for power iteration

    prev_norm = 1.0
    for step in range(max_power_steps):
        # Apply Q @ v (one step of power iteration)
        from tn_check.tensor.operations import mps_zip_up
        Qv, trunc_err = mps_zip_up(
            Q_proj, v, max_bond_dim=max_bond_dim, tolerance=1e-10,
        )

        # v_new = v + dt * Q @ v  (explicit Euler for e^{Qt})
        from tn_check.tensor.operations import mps_addition, mps_scalar_multiply
        v_new = mps_addition(v, mps_scalar_multiply(Qv, dt_step))
        v_new, _ = mps_compress(v_new, max_bond_dim=max_bond_dim, tolerance=1e-10)

        current_norm = mps_norm(v_new)
        norms.append(current_norm)

        if current_norm > 1e-300:
            v = mps_scalar_multiply(v_new, 1.0 / current_norm)
        else:
            break

        if prev_norm > 1e-300 and current_norm > 1e-300:
            ratio = current_norm / prev_norm
            ratios.append(ratio)

        prev_norm = current_norm

        # Check convergence of ratio
        if len(ratios) >= 5:
            recent = ratios[-5:]
            if all(abs(r - recent[-1]) < tolerance for r in recent):
                break

    # Estimate spectral gap from decay ratio
    if len(ratios) >= 3:
        # The dominant eigenvalue decay: ratio ≈ e^{λ₁ · dt}
        # After removing stationary component, ratio ≈ e^{λ₂ · dt}
        avg_ratio = np.mean(ratios[-min(5, len(ratios)):])
        if avg_ratio > 0 and avg_ratio < 1.0:
            gap_estimate = -np.log(avg_ratio) / dt_step
        elif avg_ratio >= 1.0:
            # Not decaying: may be dominated by stationary component
            # Try using later ratios
            if len(ratios) > 10:
                late_ratio = np.mean(ratios[-3:])
                if 0 < late_ratio < 1.0:
                    gap_estimate = -np.log(late_ratio) / dt_step
                else:
                    gap_estimate = 0.0
            else:
                gap_estimate = 0.0
        else:
            gap_estimate = 0.0

        convergence_ratio = avg_ratio
    else:
        gap_estimate = 0.0
        convergence_ratio = 1.0

    # Refine via Rayleigh quotient
    try:
        rq = rayleigh_quotient_refinement(Q_proj, v)
        if rq < 0 and abs(rq) > 1e-15:
            rq_gap = abs(rq)
            if gap_estimate > 0:
                gap_estimate = min(gap_estimate, rq_gap)
            else:
                gap_estimate = rq_gap
    except Exception:
        pass  # fall back to power iteration estimate

    # Assess feasibility
    if gap_estimate < 1e-10:
        confidence = "low"
        mixing_time = float("inf")
        predicted_iters = 999999
        feasible = False
    elif gap_estimate < 1e-4:
        confidence = "medium"
        mixing_time = 1.0 / gap_estimate
        predicted_iters = int(np.ceil(np.log(1e8) / gap_estimate))
        feasible = predicted_iters < 10000
    else:
        confidence = "high"
        mixing_time = 1.0 / gap_estimate
        predicted_iters = int(np.ceil(np.log(1e8) / gap_estimate))
        feasible = True

    result = SpectralGapEstimate(
        gap_estimate=gap_estimate,
        confidence=confidence,
        estimated_mixing_time=mixing_time,
        predicted_iterations=predicted_iters,
        feasible=feasible,
        method="power_iteration",
        power_iteration_steps=len(norms),
        convergence_ratio=convergence_ratio,
    )

    logger.info(
        f"Spectral gap estimate: gap={gap_estimate:.2e}, "
        f"mixing_time={mixing_time:.2f}, "
        f"predicted_iters={predicted_iters}, "
        f"feasible={feasible}"
    )

    return result


def adaptive_fallback_time_bound(
    gap_estimate: float,
    safety_factor: float = 5.0,
    min_time: float = 100.0,
    max_time: float = 100000.0,
) -> float:
    """
    Compute adaptive time bound for bounded-until fallback.

    When fixpoint iteration is infeasible (small spectral gap),
    we fall back to bounded-until with T = C / gap where C is
    a safety factor ensuring the transient has decayed.

    Args:
        gap_estimate: Estimated spectral gap.
        safety_factor: Multiplier for mixing time.
        min_time: Minimum fallback time.
        max_time: Maximum fallback time.

    Returns:
        Recommended time bound for bounded-until fallback.
    """
    if gap_estimate < 1e-15:
        return max_time

    mixing_time = 1.0 / gap_estimate
    fallback_t = safety_factor * mixing_time
    return float(np.clip(fallback_t, min_time, max_time))


class ConvergencePredictor:
    """
    Predict convergence behavior from a SpectralGapEstimate.

    Provides convenience methods to determine whether fixpoint iteration
    will converge within given bounds, recommend bond dimensions, and
    generate a human-readable convergence certificate.
    """

    def __init__(self, estimate: SpectralGapEstimate):
        self.estimate = estimate

    def will_converge(self, tolerance: float, max_iter: int) -> bool:
        """
        Predict whether fixpoint iteration will converge.

        Args:
            tolerance: Target convergence tolerance.
            max_iter: Maximum allowed iterations.

        Returns:
            True if predicted iterations <= max_iter.
        """
        predicted = self.estimate.predicted_iteration_count(tolerance)
        return predicted <= max_iter

    def recommended_bond_dim(self, tolerance: float, base_dim: int = 10) -> int:
        """
        Recommend a bond dimension based on the spectral gap.

        Larger spectral gaps allow smaller bond dimensions; smaller gaps
        need higher bond dimensions to capture the slow-decaying modes.

        Args:
            tolerance: Target error tolerance.
            base_dim: Minimum bond dimension.

        Returns:
            Recommended bond dimension.
        """
        gap = self.estimate.gap_estimate
        if gap < 1e-10:
            return base_dim * 10
        # Heuristic: bond dim ~ base * log(1/tol) / gap^0.25
        factor = max(1.0, np.log(1.0 / max(tolerance, 1e-15)) / gap ** 0.25)
        return max(base_dim, int(np.ceil(base_dim * factor)))

    def convergence_certificate(self) -> dict:
        """
        Summarize the convergence prediction reasoning.

        Returns:
            Dictionary with prediction details.
        """
        est = self.estimate
        return {
            "gap_estimate": est.gap_estimate,
            "confidence": est.confidence,
            "estimated_mixing_time": est.estimated_mixing_time,
            "predicted_iterations": est.predicted_iterations,
            "feasible": est.feasible,
            "method": est.method,
            "convergence_ratio": est.convergence_ratio,
            "power_iteration_steps": est.power_iteration_steps,
            "will_converge_1e6_in_1000": self.will_converge(1e-6, 1000),
            "recommended_bond_dim_1e6": self.recommended_bond_dim(1e-6),
        }
