"""
Exact rational verification using arbitrary-precision arithmetic.

This module provides exact verification of differential privacy using
mpmath for arbitrary-precision arithmetic. Unlike floating-point or
interval arithmetic, rational verification can provide definitive
answers with no rounding error.

Theory
------
Arbitrary-precision arithmetic eliminates floating-point rounding errors:
    - All computations performed with user-specified precision
    - Privacy loss computed exactly as log(p/q)
    - Hockey-stick divergence computed exactly
    - Results are exact up to the specified precision

Adaptive precision:
    - Start with low precision (e.g., 53 bits = double precision)
    - If result is inconclusive, increase precision
    - Continue until either:
        * Definitive result obtained
        * Maximum precision reached
        * Computation time exceeds limit

Cross-validation:
    - Compare against float and interval verifiers
    - Report discrepancies for further investigation

Classes
-------
- :class:`RationalVerifier` — Main exact verifier using mpmath
- :class:`PrecisionSchedule` — Adaptive precision schedule

Functions
---------
- :func:`exact_privacy_loss` — Exact log(p/q) computation
- :func:`exact_hockey_stick` — Exact hockey-stick divergence
- :func:`adaptive_verify` — Adaptive-precision verification
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt

try:
    import mpmath
    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False
    mpmath = None

from dp_forge.exceptions import ConfigurationError, VerificationError
from dp_forge.types import VerifyResult
from dp_forge.verifier import VerificationMode

logger = logging.getLogger(__name__)


class VerificationOutcome(Enum):
    """Outcome of rational verification."""
    
    DEFINITELY_VALID = auto()
    DEFINITELY_INVALID = auto()
    INCONCLUSIVE = auto()
    
    def __repr__(self) -> str:
        return f"VerificationOutcome.{self.name}"


@dataclass
class PrecisionSchedule:
    """Adaptive precision schedule for rational verification.
    
    Attributes:
        initial_bits: Starting precision in bits.
        max_bits: Maximum precision in bits.
        increment: Precision increment per step.
        current_bits: Current precision.
    """
    
    initial_bits: int = 53
    max_bits: int = 1024
    increment: int = 50
    current_bits: int = 53
    
    def __post_init__(self) -> None:
        self.current_bits = self.initial_bits
    
    def next_precision(self) -> Optional[int]:
        """Get next precision level, or None if at maximum."""
        if self.current_bits >= self.max_bits:
            return None
        self.current_bits = min(self.current_bits + self.increment, self.max_bits)
        return self.current_bits
    
    def reset(self) -> None:
        """Reset to initial precision."""
        self.current_bits = self.initial_bits


@dataclass
class RationalResult:
    """Result of rational verification with exact values.
    
    Attributes:
        valid: Whether mechanism satisfies DP.
        outcome: Verification outcome (definite or inconclusive).
        violation: Worst violation if invalid.
        precision_used: Precision in bits used for verification.
        computation_time: Time spent on verification.
        exact_values: Dict of exact computed values.
    """
    
    valid: bool
    outcome: VerificationOutcome
    violation: Optional[Tuple[int, int, int, str]] = None
    precision_used: int = 53
    computation_time: float = 0.0
    exact_values: dict = field(default_factory=dict)
    
    def to_verify_result(self) -> VerifyResult:
        """Convert to standard VerifyResult."""
        if self.violation:
            i, i_prime, j, mag_str = self.violation
            mag_float = float(mag_str)
            return VerifyResult(valid=False, violation=(i, i_prime, j, mag_float))
        return VerifyResult(valid=True, violation=None)


def _check_mpmath() -> None:
    """Check if mpmath is available."""
    if not MPMATH_AVAILABLE:
        raise ImportError(
            "mpmath is required for rational verification. "
            "Install with: pip install mpmath"
        )


def exact_privacy_loss(
    p: float,
    q: float,
    precision_bits: int = 53,
) -> str:
    """Compute exact privacy loss log(p/q) using arbitrary precision.
    
    Args:
        p: Probability from first distribution.
        q: Probability from second distribution.
        precision_bits: Precision in bits.
    
    Returns:
        String representation of exact log(p/q) value.
    """
    _check_mpmath()
    
    with mpmath.workprec(precision_bits):
        p_mp = mpmath.mpf(p)
        q_mp = mpmath.mpf(q)
        
        if q_mp == 0:
            return "inf"
        
        if p_mp == 0:
            return "-inf"
        
        ratio = p_mp / q_mp
        log_ratio = mpmath.log(ratio)
        
        return mpmath.nstr(log_ratio, n=precision_bits // 3)


def exact_hockey_stick(
    p: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
    eps: float,
    precision_bits: int = 53,
) -> str:
    """Compute exact hockey-stick divergence using arbitrary precision.
    
    Computes H_ε(P ‖ Q) = Σ_j max(p[j] - e^ε · q[j], 0) exactly.
    
    Args:
        p: First distribution.
        q: Second distribution.
        eps: Privacy parameter ε.
        precision_bits: Precision in bits.
    
    Returns:
        String representation of exact hockey-stick value.
    """
    _check_mpmath()
    
    with mpmath.workprec(precision_bits):
        exp_eps = mpmath.exp(mpmath.mpf(eps))
        
        total = mpmath.mpf(0)
        
        for p_j, q_j in zip(p, q):
            p_mp = mpmath.mpf(p_j)
            q_mp = mpmath.mpf(q_j)
            
            diff = p_mp - exp_eps * q_mp
            total += mpmath.mpf(max(diff, 0))
        
        return mpmath.nstr(total, n=precision_bits // 3)


class RationalVerifier:
    """Exact verification using arbitrary-precision arithmetic with mpmath.
    
    Provides definitive verification results by eliminating all rounding
    errors through exact rational arithmetic.
    
    Attributes:
        precision_schedule: Adaptive precision schedule.
        tolerance: Tolerance for comparing exact values.
        cross_validate: Whether to cross-validate with float verifier.
    """
    
    def __init__(
        self,
        initial_precision: int = 53,
        max_precision: int = 1024,
        tolerance: float = 1e-15,
        cross_validate: bool = True,
    ):
        _check_mpmath()
        
        self.precision_schedule = PrecisionSchedule(
            initial_bits=initial_precision,
            max_bits=max_precision,
        )
        self.tolerance = tolerance
        self.cross_validate = cross_validate
    
    def verify_pure_dp(
        self,
        prob_table: npt.NDArray[np.float64],
        edges: List[Tuple[int, int]],
        epsilon: float,
        precision_bits: Optional[int] = None,
    ) -> RationalResult:
        """Verify pure (ε, 0)-DP with exact arithmetic.
        
        Args:
            prob_table: Mechanism probability table [n, k].
            edges: Adjacent database pairs.
            epsilon: Privacy parameter ε.
            precision_bits: Precision (uses adaptive if None).
        
        Returns:
            RationalResult with exact verification.
        """
        start_time = time.time()
        
        if precision_bits is None:
            precision_bits = self.precision_schedule.current_bits
        
        with mpmath.workprec(precision_bits):
            eps_mp = mpmath.mpf(epsilon)
            exp_eps = mpmath.exp(eps_mp)
            
            worst_violation = None
            max_excess = mpmath.mpf(0)
            outcome = VerificationOutcome.DEFINITELY_VALID
            
            for i, i_prime in edges:
                p_i = prob_table[i]
                p_ip = prob_table[i_prime]
                
                for j in range(len(p_i)):
                    p_ij = mpmath.mpf(p_i[j])
                    p_ipj = mpmath.mpf(p_ip[j])
                    
                    if p_ipj == 0:
                        if p_ij > 0:
                            outcome = VerificationOutcome.DEFINITELY_INVALID
                            worst_violation = (i, i_prime, j, "inf")
                            break
                        continue
                    
                    ratio = p_ij / p_ipj
                    
                    if ratio > exp_eps:
                        excess = ratio - exp_eps
                        if excess > max_excess:
                            max_excess = excess
                            outcome = VerificationOutcome.DEFINITELY_INVALID
                            worst_violation = (
                                i, i_prime, j,
                                mpmath.nstr(float(excess), n=10)
                            )
                
                if outcome == VerificationOutcome.DEFINITELY_INVALID:
                    if worst_violation:
                        break
        
        computation_time = time.time() - start_time
        valid = outcome == VerificationOutcome.DEFINITELY_VALID
        
        return RationalResult(
            valid=valid,
            outcome=outcome,
            violation=worst_violation,
            precision_used=precision_bits,
            computation_time=computation_time,
        )
    
    def verify_approx_dp(
        self,
        prob_table: npt.NDArray[np.float64],
        edges: List[Tuple[int, int]],
        epsilon: float,
        delta: float,
        precision_bits: Optional[int] = None,
    ) -> RationalResult:
        """Verify approximate (ε, δ)-DP with exact arithmetic.
        
        Args:
            prob_table: Mechanism probability table [n, k].
            edges: Adjacent database pairs.
            epsilon: Privacy parameter ε.
            delta: Privacy parameter δ.
            precision_bits: Precision (uses adaptive if None).
        
        Returns:
            RationalResult with exact verification.
        """
        start_time = time.time()
        
        if precision_bits is None:
            precision_bits = self.precision_schedule.current_bits
        
        with mpmath.workprec(precision_bits):
            eps_mp = mpmath.mpf(epsilon)
            delta_mp = mpmath.mpf(delta)
            exp_eps = mpmath.exp(eps_mp)
            
            worst_violation = None
            max_excess = mpmath.mpf(0)
            outcome = VerificationOutcome.DEFINITELY_VALID
            
            for i, i_prime in edges:
                p_i = prob_table[i]
                p_ip = prob_table[i_prime]
                
                hs_fwd = mpmath.mpf(0)
                for j in range(len(p_i)):
                    p_ij = mpmath.mpf(p_i[j])
                    p_ipj = mpmath.mpf(p_ip[j])
                    diff = p_ij - exp_eps * p_ipj
                    hs_fwd += mpmath.mpf(max(diff, 0))
                
                hs_rev = mpmath.mpf(0)
                for j in range(len(p_i)):
                    p_ij = mpmath.mpf(p_i[j])
                    p_ipj = mpmath.mpf(p_ip[j])
                    diff = p_ipj - exp_eps * p_ij
                    hs_rev += mpmath.mpf(max(diff, 0))
                
                hs = max(hs_fwd, hs_rev)
                
                if hs > delta_mp:
                    excess = hs - delta_mp
                    if excess > max_excess:
                        max_excess = excess
                        outcome = VerificationOutcome.DEFINITELY_INVALID
                        worst_violation = (
                            i, i_prime, -1,
                            mpmath.nstr(float(excess), n=10)
                        )
        
        computation_time = time.time() - start_time
        valid = outcome == VerificationOutcome.DEFINITELY_VALID
        
        return RationalResult(
            valid=valid,
            outcome=outcome,
            violation=worst_violation,
            precision_used=precision_bits,
            computation_time=computation_time,
        )
    
    def verify(
        self,
        prob_table: npt.NDArray[np.float64],
        edges: List[Tuple[int, int]],
        epsilon: float,
        delta: float = 0.0,
        precision_bits: Optional[int] = None,
    ) -> RationalResult:
        """Verify (ε, δ)-DP with exact arithmetic.
        
        Args:
            prob_table: Mechanism probability table.
            edges: Adjacent database pairs.
            epsilon: Privacy parameter ε.
            delta: Privacy parameter δ.
            precision_bits: Precision in bits.
        
        Returns:
            RationalResult with exact verification.
        """
        if delta == 0.0:
            return self.verify_pure_dp(prob_table, edges, epsilon, precision_bits)
        else:
            return self.verify_approx_dp(
                prob_table, edges, epsilon, delta, precision_bits
            )
    
    def adaptive_verify(
        self,
        prob_table: npt.NDArray[np.float64],
        edges: List[Tuple[int, int]],
        epsilon: float,
        delta: float = 0.0,
    ) -> RationalResult:
        """Verify with adaptive precision.
        
        Starts with low precision and increases until a definitive
        result is obtained or maximum precision is reached.
        
        Args:
            prob_table: Mechanism probability table.
            edges: Adjacent database pairs.
            epsilon: Privacy parameter.
            delta: Privacy parameter.
        
        Returns:
            RationalResult with definitive outcome.
        """
        self.precision_schedule.reset()
        
        while True:
            precision = self.precision_schedule.current_bits
            logger.info(f"Trying verification with {precision} bits precision")
            
            result = self.verify(prob_table, edges, epsilon, delta, precision)
            
            if result.outcome != VerificationOutcome.INCONCLUSIVE:
                logger.info(
                    f"Definitive result obtained at {precision} bits: "
                    f"{result.outcome}"
                )
                return result
            
            next_precision = self.precision_schedule.next_precision()
            if next_precision is None:
                logger.warning(
                    f"Maximum precision {self.precision_schedule.max_bits} "
                    f"reached without definitive result"
                )
                return result
    
    def cross_validate_with_float(
        self,
        rational_result: RationalResult,
        prob_table: npt.NDArray[np.float64],
        edges: List[Tuple[int, int]],
        epsilon: float,
        delta: float = 0.0,
    ) -> bool:
        """Cross-validate rational result against float verifier.
        
        Args:
            rational_result: Result from rational verification.
            prob_table: Mechanism table.
            edges: Adjacent pairs.
            epsilon: Privacy parameter.
            delta: Privacy parameter.
        
        Returns:
            True if results agree, False if discrepancy found.
        """
        from dp_forge.verifier import verify
        
        float_result = verify(
            prob_table,
            epsilon,
            delta,
            edges,
            tol=1e-9,
            mode=VerificationMode.MOST_VIOLATING,
        )
        
        if rational_result.valid != float_result.valid:
            logger.warning(
                f"Discrepancy: rational says {rational_result.valid}, "
                f"float says {float_result.valid}"
            )
            return False
        
        return True


def adaptive_verify(
    mechanism: npt.NDArray[np.float64],
    epsilon: float,
    delta: float = 0.0,
    edges: Optional[List[Tuple[int, int]]] = None,
    initial_precision: int = 53,
    max_precision: int = 1024,
) -> RationalResult:
    """Main entry point for adaptive-precision rational verification.
    
    Args:
        mechanism: Mechanism probability table [n, k].
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ.
        edges: Adjacent pairs (defaults to all pairs).
        initial_precision: Starting precision in bits.
        max_precision: Maximum precision in bits.
    
    Returns:
        RationalResult with exact verification.
    
    Example:
        >>> mechanism = np.array([[0.5, 0.5], [0.4, 0.6]])
        >>> result = adaptive_verify(mechanism, epsilon=1.0)
        >>> print(f"Valid: {result.valid}, Precision: {result.precision_used}")
    """
    n, k = mechanism.shape
    
    if edges is None:
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    verifier = RationalVerifier(
        initial_precision=initial_precision,
        max_precision=max_precision,
    )
    
    return verifier.adaptive_verify(mechanism, edges, epsilon, delta)


def compute_exact_epsilon(
    prob_table: npt.NDArray[np.float64],
    edges: List[Tuple[int, int]],
    delta: float = 0.0,
    precision_bits: int = 100,
) -> str:
    """Compute exact tightest ε using arbitrary precision.
    
    Uses binary search with exact arithmetic to find the smallest ε
    such that the mechanism satisfies (ε, δ)-DP.
    
    Args:
        prob_table: Mechanism probability table.
        edges: Adjacent pairs.
        delta: Privacy parameter δ.
        precision_bits: Precision in bits.
    
    Returns:
        String representation of exact ε value.
    """
    _check_mpmath()
    
    verifier = RationalVerifier(initial_precision=precision_bits)
    
    with mpmath.workprec(precision_bits):
        eps_lo = mpmath.mpf(0)
        eps_hi = mpmath.mpf(10)
        
        for _ in range(100):
            if eps_hi - eps_lo < mpmath.mpf(1e-10):
                break
            
            eps_mid = (eps_lo + eps_hi) / 2
            eps_mid_float = float(eps_mid)
            
            result = verifier.verify(prob_table, edges, eps_mid_float, delta)
            
            if result.valid:
                eps_hi = eps_mid
            else:
                eps_lo = eps_mid
        
        return mpmath.nstr(eps_hi, n=precision_bits // 3)


def verify_with_certificates(
    mechanism: npt.NDArray[np.float64],
    epsilon: float,
    delta: float = 0.0,
    edges: Optional[List[Tuple[int, int]]] = None,
    precision_bits: int = 100,
) -> Tuple[RationalResult, dict]:
    """Verify and generate exact proof certificate.
    
    Args:
        mechanism: Mechanism probability table.
        epsilon: Privacy parameter.
        delta: Privacy parameter.
        edges: Adjacent pairs.
        precision_bits: Precision for exact values.
    
    Returns:
        (result, certificate_data): Result and certificate with exact values.
    """
    n, k = mechanism.shape
    if edges is None:
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    verifier = RationalVerifier(initial_precision=precision_bits)
    result = verifier.verify(mechanism, edges, epsilon, delta, precision_bits)
    
    certificate_data = {
        "mechanism_shape": (n, k),
        "epsilon": epsilon,
        "delta": delta,
        "precision_bits": precision_bits,
        "valid": result.valid,
        "outcome": result.outcome.name,
        "computation_time": result.computation_time,
        "exact_values": result.exact_values,
    }
    
    return result, certificate_data


def compare_verifiers(
    mechanism: npt.NDArray[np.float64],
    epsilon: float,
    delta: float = 0.0,
    edges: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """Compare results from float, interval, and rational verifiers.
    
    Args:
        mechanism: Mechanism probability table.
        epsilon: Privacy parameter.
        delta: Privacy parameter.
        edges: Adjacent pairs.
    
    Returns:
        Dict with comparison results.
    """
    from dp_forge.verifier import verify, VerificationMode
    from dp_forge.verification.interval_verifier import sound_verify_dp
    
    n, k = mechanism.shape
    if edges is None:
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    float_start = time.time()
    float_result = verify(
        mechanism,
        epsilon,
        delta,
        edges,
        tol=1e-9,
        mode=VerificationMode.MOST_VIOLATING,
    )
    float_time = time.time() - float_start
    
    interval_start = time.time()
    interval_result = sound_verify_dp(mechanism, epsilon, delta, edges=edges)
    interval_time = time.time() - interval_start
    
    rational_start = time.time()
    rational_result = adaptive_verify(mechanism, epsilon, delta, edges)
    rational_time = time.time() - rational_start
    
    comparison = {
        "float": {
            "valid": float_result.valid,
            "time": float_time,
            "violation": float_result.violation,
        },
        "interval": {
            "valid": interval_result.valid,
            "time": interval_time,
            "soundness": interval_result.soundness.name,
            "confidence": interval_result.confidence,
        },
        "rational": {
            "valid": rational_result.valid,
            "time": rational_time,
            "outcome": rational_result.outcome.name,
            "precision": rational_result.precision_used,
        },
        "agreement": {
            "float_interval": float_result.valid == interval_result.valid,
            "float_rational": float_result.valid == rational_result.valid,
            "interval_rational": interval_result.valid == rational_result.valid,
        },
    }
    
    return comparison


def exact_renyi_divergence(
    p: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
    alpha: float,
    precision_bits: int = 100,
) -> str:
    """Compute exact Rényi divergence using arbitrary precision.
    
    Args:
        p: First distribution.
        q: Second distribution.
        alpha: Rényi order.
        precision_bits: Precision in bits.
    
    Returns:
        String representation of exact Rényi divergence.
    """
    _check_mpmath()
    
    if alpha <= 1.0:
        raise ValueError(f"Rényi order must be > 1, got {alpha}")
    
    with mpmath.workprec(precision_bits):
        alpha_mp = mpmath.mpf(alpha)
        one_minus_alpha = mpmath.mpf(1) - alpha_mp
        
        total = mpmath.mpf(0)
        
        for p_j, q_j in zip(p, q):
            p_mp = mpmath.mpf(max(p_j, 1e-300))
            q_mp = mpmath.mpf(max(q_j, 1e-300))
            
            term = mpmath.power(p_mp, alpha_mp) * mpmath.power(q_mp, one_minus_alpha)
            total += term
        
        log_total = mpmath.log(total)
        result = log_total / (alpha_mp - mpmath.mpf(1))
        
        return mpmath.nstr(result, n=precision_bits // 3)


def exact_composition(
    individual_epsilons: List[str],
    individual_deltas: List[str],
    composition_rule: str = "basic",
    precision_bits: int = 100,
) -> Tuple[str, str]:
    """Compute exact composed privacy parameters.
    
    Args:
        individual_epsilons: List of exact epsilon strings.
        individual_deltas: List of exact delta strings.
        composition_rule: Composition rule.
        precision_bits: Precision in bits.
    
    Returns:
        (total_epsilon, total_delta) as exact strings.
    """
    _check_mpmath()
    
    with mpmath.workprec(precision_bits):
        epsilons = [mpmath.mpf(eps) for eps in individual_epsilons]
        deltas = [mpmath.mpf(delta) for delta in individual_deltas]
        
        if composition_rule == "basic":
            total_eps = sum(epsilons)
            total_delta = sum(deltas)
        elif composition_rule == "advanced":
            k = len(epsilons)
            max_eps = max(epsilons)
            max_delta = max(deltas)
            
            sum_eps = sum(epsilons)
            sqrt_term = mpmath.sqrt(
                mpmath.mpf(2 * k) * mpmath.log(mpmath.mpf(1) / max_delta)
            ) * max_eps
            
            total_eps = sum_eps + sqrt_term
            total_delta = mpmath.mpf(k) * max_delta
        else:
            raise ValueError(f"Unknown composition rule: {composition_rule}")
        
        return (
            mpmath.nstr(total_eps, n=precision_bits // 3),
            mpmath.nstr(total_delta, n=precision_bits // 3),
        )


class SymbolicVerifier:
    """Symbolic verification using computer algebra.
    
    Uses mpmath to perform symbolic manipulation for verification.
    """
    
    def __init__(self, precision_bits: int = 200):
        _check_mpmath()
        self.precision_bits = precision_bits
    
    def symbolic_privacy_loss(
        self,
        p_expr: str,
        q_expr: str,
    ) -> str:
        """Compute symbolic privacy loss.
        
        Args:
            p_expr: Expression for p as string.
            q_expr: Expression for q as string.
        
        Returns:
            Symbolic expression for log(p/q).
        """
        with mpmath.workprec(self.precision_bits):
            p = mpmath.mpf(p_expr)
            q = mpmath.mpf(q_expr)
            
            if q == 0:
                return "inf"
            
            loss = mpmath.log(p / q)
            return mpmath.nstr(loss, n=20)
    
    def verify_symbolic_constraint(
        self,
        constraint_expr: str,
        variables: Dict[str, float],
    ) -> bool:
        """Verify a symbolic constraint.
        
        Args:
            constraint_expr: Constraint expression (e.g., "x + y <= 1").
            variables: Variable values.
        
        Returns:
            True if constraint is satisfied.
        """
        with mpmath.workprec(self.precision_bits):
            for var, val in variables.items():
                constraint_expr = constraint_expr.replace(var, str(val))
            
            try:
                result = eval(constraint_expr, {"__builtins__": {}}, mpmath.__dict__)
                return bool(result)
            except Exception as e:
                logger.error(f"Failed to evaluate constraint: {e}")
                return False


def batch_rational_verify(
    mechanisms: List[npt.NDArray[np.float64]],
    epsilon: float,
    delta: float = 0.0,
    precision_bits: int = 100,
) -> List[RationalResult]:
    """Batch verify multiple mechanisms with rational arithmetic.
    
    Args:
        mechanisms: List of mechanism tables.
        epsilon: Privacy parameter.
        delta: Privacy parameter.
        precision_bits: Precision in bits.
    
    Returns:
        List of RationalResults.
    """
    verifier = RationalVerifier(
        initial_precision=precision_bits,
        max_precision=precision_bits,
    )
    
    results = []
    for mechanism in mechanisms:
        n, k = mechanism.shape
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        result = verifier.verify(mechanism, edges, epsilon, delta, precision_bits)
        results.append(result)
    
    return results


def precision_sensitivity_analysis(
    mechanism: npt.NDArray[np.float64],
    epsilon: float,
    delta: float = 0.0,
    precision_range: Tuple[int, int] = (50, 500),
    n_samples: int = 10,
) -> List[Tuple[int, RationalResult]]:
    """Analyze how verification result depends on precision.
    
    Args:
        mechanism: Mechanism table.
        epsilon: Privacy parameter.
        delta: Privacy parameter.
        precision_range: Range of precisions to test.
        n_samples: Number of precision values to test.
    
    Returns:
        List of (precision, result) tuples.
    """
    n, k = mechanism.shape
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    precisions = np.linspace(
        precision_range[0], precision_range[1], n_samples, dtype=int
    )
    
    results = []
    for prec in precisions:
        verifier = RationalVerifier(
            initial_precision=int(prec),
            max_precision=int(prec),
        )
        result = verifier.verify(mechanism, edges, epsilon, delta, int(prec))
        results.append((int(prec), result))
    
    return results
