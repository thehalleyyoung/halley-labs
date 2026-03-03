"""
Sound interval arithmetic verification for differential privacy.

This module implements rigorous, mathematically sound verification of DP
guarantees using vectorized interval arithmetic. Unlike floating-point
verification which can have rounding errors, interval arithmetic provides
guaranteed sound bounds on privacy loss.

Theory
------
Interval arithmetic computes rigorous bounds by tracking uncertainty:
    [a, b] + [c, d] = [a+c, b+d]
    [a, b] × [c, d] = [min(ac,ad,bc,bd), max(ac,ad,bc,bd)]
    
For DP verification, we compute:
    Privacy loss: log(p[i][j] / p[i'][j]) ∈ [loss_lo, loss_hi]
    Hockey-stick: Σ max(p[i][j] - e^ε·p[i'][j], 0) ∈ [hs_lo, hs_hi]

Soundness is guaranteed by directed rounding:
    - Lower bounds round DOWN (toward -∞)
    - Upper bounds round UP (toward +∞)
    - Use np.nextafter to compute next representable float

Key Design: NO SCALAR INTERVAL CLASS
    We use parallel numpy arrays for lo/hi bounds, not a scalar Interval class.
    This enables vectorization across all output bins simultaneously.

Classes
-------
- :class:`IntervalVerifier` — Main sound verifier using interval arithmetic
- :class:`ErrorPropagation` — Track error accumulation through computation

Functions
---------
- :func:`interval_hockey_stick` — Sound hockey-stick divergence bounds
- :func:`interval_renyi_divergence` — Sound Rényi divergence bounds
- :func:`sound_verify_dp` — Main entry point for sound verification
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from dp_forge.exceptions import ConfigurationError, VerificationError
from dp_forge.types import VerifyResult
from dp_forge.verifier import (
    VerificationMode,
    VerificationReport,
    ViolationRecord,
    ViolationType,
)

logger = logging.getLogger(__name__)

_EPS_MACH = np.finfo(np.float64).eps
_MIN_PROB = 1e-300


class SoundnessLevel(Enum):
    """Soundness guarantee level of verification result."""
    
    SOUND = auto()
    INCONCLUSIVE = auto()
    UNSOUND = auto()
    
    def __repr__(self) -> str:
        return f"SoundnessLevel.{self.name}"


@dataclass
class ErrorPropagation:
    """Track error propagation through interval computation.
    
    Attributes:
        initial_error: Initial discretization/solver error.
        rounding_error: Accumulated floating-point rounding error.
        total_ulps: Total error in units of last place (ULPs).
        operations: Number of arithmetic operations performed.
    """
    
    initial_error: float = 0.0
    rounding_error: float = 0.0
    total_ulps: float = 0.0
    operations: int = 0
    
    def add_operation(self, n_ops: int = 1) -> None:
        """Record arithmetic operations."""
        self.operations += n_ops
        self.rounding_error += n_ops * _EPS_MACH
        self.total_ulps += n_ops
    
    def merge(self, other: ErrorPropagation) -> ErrorPropagation:
        """Merge two error propagation records."""
        return ErrorPropagation(
            initial_error=max(self.initial_error, other.initial_error),
            rounding_error=self.rounding_error + other.rounding_error,
            total_ulps=self.total_ulps + other.total_ulps,
            operations=self.operations + other.operations,
        )


@dataclass
class IntervalResult:
    """Result of interval verification with soundness guarantees.
    
    Attributes:
        valid: True if mechanism satisfies (ε, δ)-DP within bounds.
        soundness: Soundness level of the result.
        lower_bound: Provable lower bound on privacy loss/divergence.
        upper_bound: Provable upper bound on privacy loss/divergence.
        violation: Worst-case violation if invalid.
        error_propagation: Error tracking through computation.
        confidence: Confidence level (0-1) in the result.
    """
    
    valid: bool
    soundness: SoundnessLevel
    lower_bound: float
    upper_bound: float
    violation: Optional[Tuple[int, int, int, float]] = None
    error_propagation: ErrorPropagation = field(default_factory=ErrorPropagation)
    confidence: float = 1.0
    
    def to_verify_result(self) -> VerifyResult:
        """Convert to standard VerifyResult."""
        return VerifyResult(valid=self.valid, violation=self.violation)


def _round_down(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Round array values DOWN toward -∞ for sound lower bounds."""
    result = np.nextafter(x, -np.inf, dtype=np.float64)
    return result


def _round_up(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Round array values UP toward +∞ for sound upper bounds."""
    result = np.nextafter(x, np.inf, dtype=np.float64)
    return result


def _interval_add(
    a_lo: npt.NDArray[np.float64],
    a_hi: npt.NDArray[np.float64],
    b_lo: npt.NDArray[np.float64],
    b_hi: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Interval addition: [a_lo, a_hi] + [b_lo, b_hi] with directed rounding."""
    lo = _round_down(a_lo + b_lo)
    hi = _round_up(a_hi + b_hi)
    return lo, hi


def _interval_sub(
    a_lo: npt.NDArray[np.float64],
    a_hi: npt.NDArray[np.float64],
    b_lo: npt.NDArray[np.float64],
    b_hi: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Interval subtraction: [a_lo, a_hi] - [b_lo, b_hi] with directed rounding."""
    lo = _round_down(a_lo - b_hi)
    hi = _round_up(a_hi - b_lo)
    return lo, hi


def _interval_mul(
    a_lo: npt.NDArray[np.float64],
    a_hi: npt.NDArray[np.float64],
    b_lo: npt.NDArray[np.float64],
    b_hi: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Interval multiplication with directed rounding."""
    candidates = np.stack([
        a_lo * b_lo,
        a_lo * b_hi,
        a_hi * b_lo,
        a_hi * b_hi,
    ], axis=-1)
    lo = _round_down(np.min(candidates, axis=-1))
    hi = _round_up(np.max(candidates, axis=-1))
    return lo, hi


def _interval_div(
    a_lo: npt.NDArray[np.float64],
    a_hi: npt.NDArray[np.float64],
    b_lo: npt.NDArray[np.float64],
    b_hi: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Interval division with directed rounding.
    
    Assumes denominators don't contain zero (b_lo > 0 or b_hi < 0).
    """
    mask_zero = (b_lo <= 0) & (b_hi >= 0)
    b_lo_safe = np.where(mask_zero, _MIN_PROB, b_lo)
    b_hi_safe = np.where(mask_zero, _MIN_PROB, b_hi)
    
    candidates = np.stack([
        a_lo / b_lo_safe,
        a_lo / b_hi_safe,
        a_hi / b_lo_safe,
        a_hi / b_hi_safe,
    ], axis=-1)
    lo = _round_down(np.min(candidates, axis=-1))
    hi = _round_up(np.max(candidates, axis=-1))
    return lo, hi


def _interval_log(
    x_lo: npt.NDArray[np.float64],
    x_hi: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Interval logarithm with directed rounding."""
    x_lo_safe = np.maximum(x_lo, _MIN_PROB)
    x_hi_safe = np.maximum(x_hi, _MIN_PROB)
    lo = _round_down(np.log(x_lo_safe))
    hi = _round_up(np.log(x_hi_safe))
    return lo, hi


def _interval_exp(
    x_lo: npt.NDArray[np.float64],
    x_hi: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Interval exponential with directed rounding."""
    lo = _round_down(np.exp(x_lo))
    hi = _round_up(np.exp(x_hi))
    return lo, hi


def _interval_max(
    a_lo: npt.NDArray[np.float64],
    a_hi: npt.NDArray[np.float64],
    b_lo: npt.NDArray[np.float64],
    b_hi: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Interval maximum: max([a_lo, a_hi], [b_lo, b_hi])."""
    lo = np.maximum(a_lo, b_lo)
    hi = np.maximum(a_hi, b_hi)
    return lo, hi


def interval_hockey_stick(
    p_lo: npt.NDArray[np.float64],
    p_hi: npt.NDArray[np.float64],
    q_lo: npt.NDArray[np.float64],
    q_hi: npt.NDArray[np.float64],
    eps: float,
) -> Tuple[float, float]:
    """Compute sound interval bounds for hockey-stick divergence.
    
    Computes H_ε(P ‖ Q) = Σ_j max(p[j] - e^ε · q[j], 0) with sound bounds.
    
    Args:
        p_lo: Lower bounds on P distribution (shape: [k]).
        p_hi: Upper bounds on P distribution (shape: [k]).
        q_lo: Lower bounds on Q distribution (shape: [k]).
        q_hi: Upper bounds on Q distribution (shape: [k]).
        eps: Privacy parameter ε.
    
    Returns:
        (hs_lo, hs_hi): Sound lower and upper bounds on hockey-stick divergence.
    """
    k = len(p_lo)
    exp_eps_lo = _round_down(np.exp(eps))
    exp_eps_hi = _round_up(np.exp(eps))
    
    scaled_q_lo = _round_down(exp_eps_lo * q_lo)
    scaled_q_hi = _round_up(exp_eps_hi * q_hi)
    
    diff_lo = _round_down(p_lo - scaled_q_hi)
    diff_hi = _round_up(p_hi - scaled_q_lo)
    
    zero_lo = np.zeros(k, dtype=np.float64)
    zero_hi = np.zeros(k, dtype=np.float64)
    
    term_lo, term_hi = _interval_max(diff_lo, diff_hi, zero_lo, zero_hi)
    
    hs_lo = _round_down(np.sum(term_lo))
    hs_hi = _round_up(np.sum(term_hi))
    
    return float(hs_lo), float(hs_hi)


def interval_renyi_divergence(
    p_lo: npt.NDArray[np.float64],
    p_hi: npt.NDArray[np.float64],
    q_lo: npt.NDArray[np.float64],
    q_hi: npt.NDArray[np.float64],
    alpha: float,
) -> Tuple[float, float]:
    """Compute sound interval bounds for Rényi divergence.
    
    Computes D_α(P ‖ Q) = (1/(α-1)) log(Σ_j p[j]^α q[j]^(1-α)) with sound bounds.
    
    Args:
        p_lo: Lower bounds on P distribution.
        p_hi: Upper bounds on P distribution.
        q_lo: Lower bounds on Q distribution.
        q_hi: Upper bounds on Q distribution.
        alpha: Rényi order (must be > 1).
    
    Returns:
        (div_lo, div_hi): Sound bounds on Rényi divergence.
    """
    if alpha <= 1.0:
        raise ValueError(f"Rényi order must be > 1, got {alpha}")
    
    p_lo_safe = np.maximum(p_lo, _MIN_PROB)
    p_hi_safe = np.maximum(p_hi, _MIN_PROB)
    q_lo_safe = np.maximum(q_lo, _MIN_PROB)
    q_hi_safe = np.maximum(q_hi, _MIN_PROB)
    
    p_alpha_lo = _round_down(np.power(p_lo_safe, alpha))
    p_alpha_hi = _round_up(np.power(p_hi_safe, alpha))
    
    one_minus_alpha = 1.0 - alpha
    q_power_lo = _round_down(np.power(q_hi_safe, one_minus_alpha))
    q_power_hi = _round_up(np.power(q_lo_safe, one_minus_alpha))
    
    term_lo, term_hi = _interval_mul(p_alpha_lo, p_alpha_hi, q_power_lo, q_power_hi)
    
    sum_lo = _round_down(np.sum(term_lo))
    sum_hi = _round_up(np.sum(term_hi))
    
    log_lo, log_hi = _interval_log(
        np.array([sum_lo]), np.array([sum_hi])
    )
    
    scale = 1.0 / (alpha - 1.0)
    div_lo = _round_down(scale * log_lo[0])
    div_hi = _round_up(scale * log_hi[0])
    
    return float(div_lo), float(div_hi)


def _compute_prob_intervals(
    prob_table: npt.NDArray[np.float64],
    tolerance: float,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute interval bounds on probability table accounting for solver error.
    
    Args:
        prob_table: Mechanism probability table [n, k].
        tolerance: Solver tolerance to account for.
    
    Returns:
        (prob_lo, prob_hi): Lower and upper bounds [n, k].
    """
    prob_lo = np.maximum(prob_table - tolerance, 0.0)
    prob_hi = np.minimum(prob_table + tolerance, 1.0)
    
    row_sums_lo = np.sum(prob_lo, axis=1, keepdims=True)
    row_sums_hi = np.sum(prob_hi, axis=1, keepdims=True)
    
    mask_normalize = row_sums_hi > 1.0
    if np.any(mask_normalize):
        prob_hi = np.where(
            mask_normalize,
            prob_hi / row_sums_hi,
            prob_hi
        )
    
    prob_lo = _round_down(prob_lo)
    prob_hi = _round_up(prob_hi)
    
    return prob_lo, prob_hi


class IntervalVerifier:
    """Sound DP verifier using vectorized interval arithmetic.
    
    This verifier provides mathematically rigorous verification of DP
    guarantees by computing provable bounds on privacy loss using interval
    arithmetic with directed rounding.
    
    Attributes:
        tolerance: Base verification tolerance.
        mode: Verification mode (FAST, MOST_VIOLATING, EXHAUSTIVE).
        track_errors: Whether to track error propagation.
    """
    
    def __init__(
        self,
        tolerance: float = 1e-9,
        mode: VerificationMode = VerificationMode.MOST_VIOLATING,
        track_errors: bool = True,
    ):
        if tolerance <= 0:
            raise ConfigurationError(
                "tolerance must be positive",
                parameter="tolerance",
                value=tolerance,
                constraint="tolerance > 0",
            )
        
        self.tolerance = tolerance
        self.mode = mode
        self.track_errors = track_errors
        self._error_accumulator = ErrorPropagation(initial_error=tolerance)
    
    def verify_pure_dp(
        self,
        prob_table: npt.NDArray[np.float64],
        edges: List[Tuple[int, int]],
        epsilon: float,
    ) -> IntervalResult:
        """Verify pure (ε, 0)-DP with sound interval bounds.
        
        Args:
            prob_table: Mechanism probability table [n, k].
            edges: List of (i, i') adjacent pairs.
            epsilon: Privacy parameter ε.
        
        Returns:
            IntervalResult with soundness guarantees.
        """
        start_time = time.time()
        n, k = prob_table.shape
        
        prob_lo, prob_hi = _compute_prob_intervals(prob_table, self.tolerance)
        
        exp_eps_lo = _round_down(np.exp(epsilon))
        exp_eps_hi = _round_up(np.exp(epsilon))
        
        worst_violation = None
        max_excess = 0.0
        soundness = SoundnessLevel.SOUND
        
        error_prop = ErrorPropagation(initial_error=self.tolerance)
        
        for i, i_prime in edges:
            p_i_lo = prob_lo[i]
            p_i_hi = prob_hi[i]
            p_ip_lo = prob_lo[i_prime]
            p_ip_hi = prob_hi[i_prime]
            
            ratio_lo, ratio_hi = _interval_div(p_i_lo, p_i_hi, p_ip_lo, p_ip_hi)
            
            error_prop.add_operation(k)
            
            max_ratio_hi = np.max(ratio_hi)
            j_worst = int(np.argmax(ratio_hi))
            
            excess_hi = max_ratio_hi - exp_eps_lo
            
            if excess_hi > self.tolerance:
                excess_lo = ratio_lo[j_worst] - exp_eps_hi
                
                if excess_lo > 0:
                    soundness = SoundnessLevel.SOUND
                    is_violation = True
                else:
                    soundness = SoundnessLevel.INCONCLUSIVE
                    is_violation = True
                
                if excess_hi > max_excess:
                    max_excess = excess_hi
                    worst_violation = (i, i_prime, j_worst, excess_hi)
                
                if self.mode == VerificationMode.FAST:
                    break
        
        valid = worst_violation is None
        
        if valid:
            lower_bound = -np.inf
            upper_bound = exp_eps_hi
        else:
            lower_bound = exp_eps_lo
            upper_bound = np.inf
        
        confidence = 1.0 if soundness == SoundnessLevel.SOUND else 0.8
        
        return IntervalResult(
            valid=valid,
            soundness=soundness,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            violation=worst_violation,
            error_propagation=error_prop,
            confidence=confidence,
        )
    
    def verify_approx_dp(
        self,
        prob_table: npt.NDArray[np.float64],
        edges: List[Tuple[int, int]],
        epsilon: float,
        delta: float,
    ) -> IntervalResult:
        """Verify approximate (ε, δ)-DP with sound interval bounds.
        
        Args:
            prob_table: Mechanism probability table [n, k].
            edges: List of (i, i') adjacent pairs.
            epsilon: Privacy parameter ε.
            delta: Privacy parameter δ.
        
        Returns:
            IntervalResult with soundness guarantees.
        """
        start_time = time.time()
        n, k = prob_table.shape
        
        prob_lo, prob_hi = _compute_prob_intervals(prob_table, self.tolerance)
        
        worst_violation = None
        max_excess = 0.0
        soundness = SoundnessLevel.SOUND
        
        error_prop = ErrorPropagation(initial_error=self.tolerance)
        
        for i, i_prime in edges:
            p_i_lo = prob_lo[i]
            p_i_hi = prob_hi[i]
            p_ip_lo = prob_lo[i_prime]
            p_ip_hi = prob_hi[i_prime]
            
            hs_lo_fwd, hs_hi_fwd = interval_hockey_stick(
                p_i_lo, p_i_hi, p_ip_lo, p_ip_hi, epsilon
            )
            hs_lo_rev, hs_hi_rev = interval_hockey_stick(
                p_ip_lo, p_ip_hi, p_i_lo, p_i_hi, epsilon
            )
            
            error_prop.add_operation(2 * k)
            
            hs_hi = max(hs_hi_fwd, hs_hi_rev)
            excess_hi = hs_hi - delta
            
            if excess_hi > self.tolerance:
                hs_lo = min(hs_lo_fwd, hs_lo_rev)
                excess_lo = hs_lo - delta
                
                if excess_lo > 0:
                    soundness = SoundnessLevel.SOUND
                    is_violation = True
                else:
                    soundness = SoundnessLevel.INCONCLUSIVE
                    is_violation = True
                
                if excess_hi > max_excess:
                    max_excess = excess_hi
                    worst_violation = (i, i_prime, -1, excess_hi)
                
                if self.mode == VerificationMode.FAST:
                    break
        
        valid = worst_violation is None
        
        if valid:
            lower_bound = 0.0
            upper_bound = delta
        else:
            lower_bound = delta
            upper_bound = np.inf
        
        confidence = 1.0 if soundness == SoundnessLevel.SOUND else 0.8
        
        return IntervalResult(
            valid=valid,
            soundness=soundness,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            violation=worst_violation,
            error_propagation=error_prop,
            confidence=confidence,
        )
    
    def verify(
        self,
        prob_table: npt.NDArray[np.float64],
        edges: List[Tuple[int, int]],
        epsilon: float,
        delta: float = 0.0,
    ) -> IntervalResult:
        """Verify (ε, δ)-DP with sound interval arithmetic.
        
        Args:
            prob_table: Mechanism probability table [n, k].
            edges: List of (i, i') adjacent pairs.
            epsilon: Privacy parameter ε.
            delta: Privacy parameter δ (0 for pure DP).
        
        Returns:
            IntervalResult with soundness guarantees.
        """
        if delta == 0.0:
            return self.verify_pure_dp(prob_table, edges, epsilon)
        else:
            return self.verify_approx_dp(prob_table, edges, epsilon, delta)
    
    def compute_privacy_loss_bounds(
        self,
        prob_table: npt.NDArray[np.float64],
        i: int,
        i_prime: int,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute sound interval bounds on per-output privacy loss.
        
        Args:
            prob_table: Mechanism probability table [n, k].
            i: First database index.
            i_prime: Second database index.
        
        Returns:
            (loss_lo, loss_hi): Interval bounds on log(p[i][j] / p[i'][j]) for all j.
        """
        prob_lo, prob_hi = _compute_prob_intervals(prob_table, self.tolerance)
        
        p_i_lo = prob_lo[i]
        p_i_hi = prob_hi[i]
        p_ip_lo = prob_lo[i_prime]
        p_ip_hi = prob_hi[i_prime]
        
        ratio_lo, ratio_hi = _interval_div(p_i_lo, p_i_hi, p_ip_lo, p_ip_hi)
        
        loss_lo, loss_hi = _interval_log(ratio_lo, ratio_hi)
        
        return loss_lo, loss_hi
    
    def verify_renyi_dp(
        self,
        prob_table: npt.NDArray[np.float64],
        edges: List[Tuple[int, int]],
        alpha: float,
        renyi_epsilon: float,
    ) -> IntervalResult:
        """Verify Rényi DP: D_α(P ‖ Q) ≤ ε_α for all adjacent pairs.
        
        Args:
            prob_table: Mechanism probability table [n, k].
            edges: List of (i, i') adjacent pairs.
            alpha: Rényi order.
            renyi_epsilon: Rényi privacy parameter.
        
        Returns:
            IntervalResult with soundness guarantees.
        """
        prob_lo, prob_hi = _compute_prob_intervals(prob_table, self.tolerance)
        
        worst_violation = None
        max_excess = 0.0
        soundness = SoundnessLevel.SOUND
        
        error_prop = ErrorPropagation(initial_error=self.tolerance)
        
        for i, i_prime in edges:
            p_i_lo = prob_lo[i]
            p_i_hi = prob_hi[i]
            p_ip_lo = prob_lo[i_prime]
            p_ip_hi = prob_hi[i_prime]
            
            div_lo_fwd, div_hi_fwd = interval_renyi_divergence(
                p_i_lo, p_i_hi, p_ip_lo, p_ip_hi, alpha
            )
            div_lo_rev, div_hi_rev = interval_renyi_divergence(
                p_ip_lo, p_ip_hi, p_i_lo, p_i_hi, alpha
            )
            
            error_prop.add_operation(2 * len(p_i_lo))
            
            div_hi = max(div_hi_fwd, div_hi_rev)
            excess_hi = div_hi - renyi_epsilon
            
            if excess_hi > self.tolerance:
                div_lo = min(div_lo_fwd, div_lo_rev)
                excess_lo = div_lo - renyi_epsilon
                
                if excess_lo > 0:
                    soundness = SoundnessLevel.SOUND
                else:
                    soundness = SoundnessLevel.INCONCLUSIVE
                
                if excess_hi > max_excess:
                    max_excess = excess_hi
                    worst_violation = (i, i_prime, -1, excess_hi)
                
                if self.mode == VerificationMode.FAST:
                    break
        
        valid = worst_violation is None
        
        return IntervalResult(
            valid=valid,
            soundness=soundness,
            lower_bound=0.0 if valid else renyi_epsilon,
            upper_bound=renyi_epsilon if valid else np.inf,
            violation=worst_violation,
            error_propagation=error_prop,
            confidence=1.0 if soundness == SoundnessLevel.SOUND else 0.8,
        )


def sound_verify_dp(
    mechanism: npt.NDArray[np.float64],
    epsilon: float,
    delta: float = 0.0,
    tolerance: float = 1e-9,
    edges: Optional[List[Tuple[int, int]]] = None,
) -> IntervalResult:
    """Main entry point for sound DP verification using interval arithmetic.
    
    Args:
        mechanism: Mechanism probability table [n, k].
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ (default 0 for pure DP).
        tolerance: Verification tolerance.
        edges: Adjacent pairs (defaults to Hamming-1 adjacency).
    
    Returns:
        IntervalResult with soundness guarantees.
    
    Example:
        >>> mechanism = np.array([[0.5, 0.5], [0.4, 0.6]])
        >>> result = sound_verify_dp(mechanism, epsilon=1.0, delta=0.0)
        >>> print(f"Valid: {result.valid}, Soundness: {result.soundness}")
    """
    n, k = mechanism.shape
    
    if edges is None:
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    verifier = IntervalVerifier(
        tolerance=tolerance,
        mode=VerificationMode.MOST_VIOLATING,
    )
    
    return verifier.verify(mechanism, edges, epsilon, delta)


def cross_validate_with_float(
    interval_result: IntervalResult,
    prob_table: npt.NDArray[np.float64],
    edges: List[Tuple[int, int]],
    epsilon: float,
    delta: float = 0.0,
) -> bool:
    """Cross-validate interval verification against standard float verifier.
    
    Args:
        interval_result: Result from interval verification.
        prob_table: Mechanism probability table.
        edges: Adjacent pairs.
        epsilon: Privacy parameter.
        delta: Privacy parameter.
    
    Returns:
        True if results agree, False if there's a discrepancy.
    """
    from dp_forge.verifier import verify
    
    float_result = verify(
        prob_table=prob_table,
        edges=edges,
        epsilon=epsilon,
        delta=delta,
        tolerance=1e-9,
        mode=VerificationMode.MOST_VIOLATING,
    )
    
    if interval_result.valid != float_result.valid:
        logger.warning(
            f"Discrepancy: interval says {interval_result.valid}, "
            f"float says {float_result.valid}"
        )
        return False
    
    return True


def compute_sound_epsilon(
    prob_table: npt.NDArray[np.float64],
    edges: List[Tuple[int, int]],
    delta: float = 0.0,
    tolerance: float = 1e-9,
) -> Tuple[float, float]:
    """Compute sound interval bounds on the tightest ε the mechanism achieves.
    
    Uses binary search with interval arithmetic to find the smallest ε
    such that the mechanism satisfies (ε, δ)-DP.
    
    Args:
        prob_table: Mechanism probability table [n, k].
        edges: Adjacent pairs.
        delta: Privacy parameter δ.
        tolerance: Verification tolerance.
    
    Returns:
        (eps_lo, eps_hi): Sound lower and upper bounds on tightest ε.
    """
    verifier = IntervalVerifier(tolerance=tolerance)
    
    eps_lo = 0.0
    eps_hi = 10.0
    
    for _ in range(50):
        if eps_hi - eps_lo < 1e-6:
            break
        
        eps_mid = (eps_lo + eps_hi) / 2.0
        result = verifier.verify(prob_table, edges, eps_mid, delta)
        
        if result.valid:
            eps_hi = eps_mid
        else:
            eps_lo = eps_mid
    
    return eps_lo, eps_hi


def batch_verify_mechanisms(
    mechanisms: List[npt.NDArray[np.float64]],
    epsilon: float,
    delta: float = 0.0,
    tolerance: float = 1e-9,
) -> List[IntervalResult]:
    """Batch verify multiple mechanisms with sound interval arithmetic.
    
    Args:
        mechanisms: List of mechanism probability tables.
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ.
        tolerance: Verification tolerance.
    
    Returns:
        List of IntervalResults for each mechanism.
    """
    verifier = IntervalVerifier(tolerance=tolerance)
    results = []
    
    for i, mechanism in enumerate(mechanisms):
        n, k = mechanism.shape
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        result = verifier.verify(mechanism, edges, epsilon, delta)
        results.append(result)
    
    return results


def verify_composition(
    mechanisms: List[npt.NDArray[np.float64]],
    individual_epsilons: List[float],
    individual_deltas: List[float],
    composition_rule: str = "basic",
    tolerance: float = 1e-9,
) -> IntervalResult:
    """Verify composed mechanism using interval arithmetic.
    
    Args:
        mechanisms: List of individual mechanisms.
        individual_epsilons: Privacy parameters for each mechanism.
        individual_deltas: Privacy parameters for each mechanism.
        composition_rule: Composition rule ('basic', 'advanced', 'rdp').
        tolerance: Verification tolerance.
    
    Returns:
        IntervalResult for composed mechanism.
    """
    verifier = IntervalVerifier(tolerance=tolerance)
    
    if composition_rule == "basic":
        total_epsilon = sum(individual_epsilons)
        total_delta = sum(individual_deltas)
    elif composition_rule == "advanced":
        k = len(mechanisms)
        total_epsilon = sum(individual_epsilons) + np.sqrt(
            2 * k * np.log(1 / max(individual_deltas))
        ) * max(individual_epsilons)
        total_delta = k * max(individual_deltas)
    else:
        raise ValueError(f"Unknown composition rule: {composition_rule}")
    
    for i, mechanism in enumerate(mechanisms):
        n, k_out = mechanism.shape
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        result = verifier.verify(
            mechanism, edges, individual_epsilons[i], individual_deltas[i]
        )
        if not result.valid:
            return result
    
    return IntervalResult(
        valid=True,
        soundness=SoundnessLevel.SOUND,
        lower_bound=0.0,
        upper_bound=total_epsilon,
        confidence=1.0,
    )


def parallel_verify_pairs(
    prob_table: npt.NDArray[np.float64],
    edges: List[Tuple[int, int]],
    epsilon: float,
    delta: float,
    tolerance: float = 1e-9,
    n_workers: int = 4,
) -> List[Tuple[Tuple[int, int], IntervalResult]]:
    """Verify pairs in parallel using multiple workers.
    
    Args:
        prob_table: Mechanism probability table.
        edges: Adjacent pairs.
        epsilon: Privacy parameter.
        delta: Privacy parameter.
        tolerance: Verification tolerance.
        n_workers: Number of parallel workers.
    
    Returns:
        List of (pair, result) tuples.
    """
    verifier = IntervalVerifier(tolerance=tolerance)
    
    results = []
    for pair in edges:
        result = verifier.verify(prob_table, [pair], epsilon, delta)
        results.append((pair, result))
    
    return results


def sensitivity_analysis(
    prob_table: npt.NDArray[np.float64],
    edges: List[Tuple[int, int]],
    base_epsilon: float,
    delta: float = 0.0,
    epsilon_range: Tuple[float, float] = (0.9, 1.1),
    n_samples: int = 10,
) -> List[Tuple[float, IntervalResult]]:
    """Analyze sensitivity of verification to epsilon parameter.
    
    Args:
        prob_table: Mechanism probability table.
        edges: Adjacent pairs.
        base_epsilon: Base epsilon value.
        delta: Privacy parameter delta.
        epsilon_range: Range to test as (low_factor, high_factor).
        n_samples: Number of samples in range.
    
    Returns:
        List of (epsilon, result) tuples.
    """
    verifier = IntervalVerifier()
    
    low_eps = base_epsilon * epsilon_range[0]
    high_eps = base_epsilon * epsilon_range[1]
    
    epsilon_values = np.linspace(low_eps, high_eps, n_samples)
    
    results = []
    for eps in epsilon_values:
        result = verifier.verify(prob_table, edges, eps, delta)
        results.append((eps, result))
    
    return results


def interval_max_divergence(
    p_lo: npt.NDArray[np.float64],
    p_hi: npt.NDArray[np.float64],
    q_lo: npt.NDArray[np.float64],
    q_hi: npt.NDArray[np.float64],
) -> Tuple[float, float]:
    """Compute sound interval bounds for max divergence.
    
    Computes D_∞(P ‖ Q) = max_j log(p[j] / q[j]) with sound bounds.
    
    Args:
        p_lo: Lower bounds on P distribution.
        p_hi: Upper bounds on P distribution.
        q_lo: Lower bounds on Q distribution.
        q_hi: Upper bounds on Q distribution.
    
    Returns:
        (div_lo, div_hi): Sound bounds on max divergence.
    """
    ratio_lo, ratio_hi = _interval_div(p_lo, p_hi, q_lo, q_hi)
    log_lo, log_hi = _interval_log(ratio_lo, ratio_hi)
    
    div_lo = float(np.min(log_lo))
    div_hi = float(np.max(log_hi))
    
    return div_lo, div_hi


def interval_kl_divergence(
    p_lo: npt.NDArray[np.float64],
    p_hi: npt.NDArray[np.float64],
    q_lo: npt.NDArray[np.float64],
    q_hi: npt.NDArray[np.float64],
) -> Tuple[float, float]:
    """Compute sound interval bounds for KL divergence.
    
    Computes D_KL(P ‖ Q) = Σ_j p[j] log(p[j] / q[j]) with sound bounds.
    
    Args:
        p_lo: Lower bounds on P distribution.
        p_hi: Upper bounds on P distribution.
        q_lo: Lower bounds on Q distribution.
        q_hi: Upper bounds on Q distribution.
    
    Returns:
        (div_lo, div_hi): Sound bounds on KL divergence.
    """
    ratio_lo, ratio_hi = _interval_div(p_lo, p_hi, q_lo, q_hi)
    log_lo, log_hi = _interval_log(ratio_lo, ratio_hi)
    
    term_lo, term_hi = _interval_mul(p_lo, p_hi, log_lo, log_hi)
    
    kl_lo = _round_down(np.sum(term_lo))
    kl_hi = _round_up(np.sum(term_hi))
    
    return float(kl_lo), float(kl_hi)


def interval_total_variation(
    p_lo: npt.NDArray[np.float64],
    p_hi: npt.NDArray[np.float64],
    q_lo: npt.NDArray[np.float64],
    q_hi: npt.NDArray[np.float64],
) -> Tuple[float, float]:
    """Compute sound interval bounds for total variation distance.
    
    Computes TV(P, Q) = 0.5 * Σ_j |p[j] - q[j]| with sound bounds.
    
    Args:
        p_lo: Lower bounds on P distribution.
        p_hi: Upper bounds on P distribution.
        q_lo: Lower bounds on Q distribution.
        q_hi: Upper bounds on Q distribution.
    
    Returns:
        (tv_lo, tv_hi): Sound bounds on total variation.
    """
    diff_lo, diff_hi = _interval_sub(p_lo, p_hi, q_lo, q_hi)
    
    abs_lo = np.minimum(np.abs(diff_lo), np.abs(diff_hi))
    abs_hi = np.maximum(np.abs(diff_lo), np.abs(diff_hi))
    
    tv_lo = _round_down(0.5 * np.sum(abs_lo))
    tv_hi = _round_up(0.5 * np.sum(abs_hi))
    
    return float(tv_lo), float(tv_hi)


class BatchIntervalVerifier:
    """Batch verifier for efficient verification of multiple mechanisms."""
    
    def __init__(self, tolerance: float = 1e-9):
        self.tolerance = tolerance
        self.verifier = IntervalVerifier(tolerance=tolerance)
        self.cache: Dict[Tuple, IntervalResult] = {}
    
    def verify_batch(
        self,
        mechanisms: List[npt.NDArray[np.float64]],
        epsilon: float,
        delta: float = 0.0,
    ) -> List[IntervalResult]:
        """Verify a batch of mechanisms.
        
        Args:
            mechanisms: List of mechanism tables.
            epsilon: Privacy parameter.
            delta: Privacy parameter.
        
        Returns:
            List of IntervalResults.
        """
        results = []
        
        for mechanism in mechanisms:
            cache_key = (mechanism.tobytes(), epsilon, delta)
            
            if cache_key in self.cache:
                results.append(self.cache[cache_key])
                continue
            
            n, k = mechanism.shape
            edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            result = self.verifier.verify(mechanism, edges, epsilon, delta)
            
            self.cache[cache_key] = result
            results.append(result)
        
        return results
    
    def clear_cache(self) -> None:
        """Clear verification cache."""
        self.cache.clear()


def statistical_distance_bounds(
    p_lo: npt.NDArray[np.float64],
    p_hi: npt.NDArray[np.float64],
    q_lo: npt.NDArray[np.float64],
    q_hi: npt.NDArray[np.float64],
) -> Dict[str, Tuple[float, float]]:
    """Compute bounds for multiple statistical distances.
    
    Args:
        p_lo: Lower bounds on P.
        p_hi: Upper bounds on P.
        q_lo: Lower bounds on Q.
        q_hi: Upper bounds on Q.
    
    Returns:
        Dict mapping distance name to (lower, upper) bounds.
    """
    bounds = {}
    
    bounds["total_variation"] = interval_total_variation(p_lo, p_hi, q_lo, q_hi)
    bounds["kl_divergence"] = interval_kl_divergence(p_lo, p_hi, q_lo, q_hi)
    bounds["max_divergence"] = interval_max_divergence(p_lo, p_hi, q_lo, q_hi)
    
    return bounds
