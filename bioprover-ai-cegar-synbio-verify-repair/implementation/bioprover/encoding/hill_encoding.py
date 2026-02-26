"""Hill function and nonlinear kinetics encoding for SMT solvers.

Converts Hill functions and mass-action kinetics into SMT-amenable forms,
with exact polynomial encoding for integer coefficients and piecewise-linear
or Taylor approximations for non-integer coefficients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple

from .expression import (
    Abs,
    Add,
    And,
    Const,
    Div,
    ExprNode,
    Ge,
    HillAct,
    HillRep,
    Interval,
    Ite,
    Le,
    Lt,
    Mul,
    Neg,
    Pow,
    Var,
    ZERO,
    ONE,
    const_value,
    sum_exprs,
    prod_exprs,
)


# ---------------------------------------------------------------------------
# Fragment classification
# ---------------------------------------------------------------------------

class FragmentKind(Enum):
    """Classification of a kinetics expression for solver dispatch."""
    POLYNOMIAL = auto()      # decidable in QF_NRA
    RATIONAL = auto()        # decidable in QF_NRA
    TRANSCENDENTAL = auto()  # delta-decidable (dReal)


def classify_fragment(expr: ExprNode) -> FragmentKind:
    """Classify whether *expr* is polynomial, rational, or transcendental."""
    from .expression import Exp, Log, Sin, Cos, Sqrt, _UnaryOp

    for node in expr.iter_preorder():
        if isinstance(node, (Sin, Cos, Exp, Log)):
            return FragmentKind.TRANSCENDENTAL
        if isinstance(node, Sqrt):
            return FragmentKind.TRANSCENDENTAL
        if isinstance(node, Pow):
            ev = const_value(node.right)
            if ev is None or not float(ev).is_integer() or ev < 0:
                return FragmentKind.TRANSCENDENTAL
        if isinstance(node, Div):
            return FragmentKind.RATIONAL
        if isinstance(node, (HillAct, HillRep)):
            return FragmentKind.RATIONAL
    return FragmentKind.POLYNOMIAL


# ---------------------------------------------------------------------------
# Approximation error tracking
# ---------------------------------------------------------------------------

@dataclass
class ApproxError:
    """Track approximation error bounds."""
    method: str
    max_absolute_error: float
    domain: Optional[Interval] = None
    num_pieces: int = 0
    taylor_order: int = 0

    def __repr__(self) -> str:
        return (f"ApproxError({self.method}, "
                f"max_err={self.max_absolute_error:.2e})")


# ---------------------------------------------------------------------------
# Integer Hill coefficient: exact polynomial encoding
# ---------------------------------------------------------------------------

def _integer_power(base: ExprNode, n: int) -> ExprNode:
    """Compute base^n as a product tree (exact polynomial)."""
    if n == 0:
        return ONE
    if n == 1:
        return base
    if n == 2:
        return Mul(base, base)
    # Binary exponentiation for efficiency
    if n % 2 == 0:
        half = _integer_power(base, n // 2)
        return Mul(half, half)
    else:
        return Mul(base, _integer_power(base, n - 1))


def encode_hill_act_integer(
    x: ExprNode,
    K: ExprNode,
    n: int,
) -> ExprNode:
    """Encode HillAct(x, K, n) as x^n / (K^n + x^n) for integer n.

    This is an exact polynomial/rational encoding decidable in QF_NRA.
    """
    x_n = _integer_power(x, n)
    K_n = _integer_power(K, n)
    return Div(x_n, Add(K_n, x_n))


def encode_hill_rep_integer(
    x: ExprNode,
    K: ExprNode,
    n: int,
) -> ExprNode:
    """Encode HillRep(x, K, n) as K^n / (K^n + x^n) for integer n."""
    x_n = _integer_power(x, n)
    K_n = _integer_power(K, n)
    return Div(K_n, Add(K_n, x_n))


# ---------------------------------------------------------------------------
# Non-integer Hill: piecewise-linear approximation
# ---------------------------------------------------------------------------

def _hill_act_value(x: float, K: float, n: float) -> float:
    """Evaluate the Hill activating function numerically."""
    if x <= 0:
        return 0.0
    xn = x ** n
    return xn / (K ** n + xn)


def _hill_rep_value(x: float, K: float, n: float) -> float:
    """Evaluate the Hill repressing function numerically."""
    if x <= 0:
        return 1.0
    kn = K ** n
    return kn / (kn + x ** n)


def encode_hill_act_pwl(
    x: ExprNode,
    K_val: float,
    n_val: float,
    domain: Interval,
    num_pieces: int = 10,
) -> Tuple[ExprNode, ApproxError]:
    """Piecewise-linear approximation of HillAct over *domain*.

    Returns (expr, error_bound) where expr uses Ite nodes.
    """
    breakpoints = [domain.lo + i * domain.width() / num_pieces
                   for i in range(num_pieces + 1)]
    values = [_hill_act_value(bp, K_val, n_val) for bp in breakpoints]

    # Build piecewise-linear expression via nested Ite
    result: ExprNode = Const(values[-1])
    max_error = 0.0

    for i in range(num_pieces - 1, -1, -1):
        x0, x1 = breakpoints[i], breakpoints[i + 1]
        y0, y1 = values[i], values[i + 1]
        slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0.0
        # y = y0 + slope * (x - x0)
        segment = Add(Const(y0), Mul(Const(slope), Add(x, Neg(Const(x0)))))
        result = Ite(Lt(x, Const(x1)), segment, result)

        # Error bound: sample multiple points within each segment
        for frac in [0.25, 0.5, 0.75]:
            sample = x0 + frac * (x1 - x0)
            true_val = _hill_act_value(sample, K_val, n_val)
            approx_val = y0 + slope * (sample - x0)
            max_error = max(max_error, abs(true_val - approx_val))

    error = ApproxError(
        method="piecewise_linear",
        max_absolute_error=max_error,
        domain=domain,
        num_pieces=num_pieces,
    )
    return result, error


def encode_hill_rep_pwl(
    x: ExprNode,
    K_val: float,
    n_val: float,
    domain: Interval,
    num_pieces: int = 10,
) -> Tuple[ExprNode, ApproxError]:
    """Piecewise-linear approximation of HillRep over *domain*."""
    breakpoints = [domain.lo + i * domain.width() / num_pieces
                   for i in range(num_pieces + 1)]
    values = [_hill_rep_value(bp, K_val, n_val) for bp in breakpoints]

    result: ExprNode = Const(values[-1])
    max_error = 0.0

    for i in range(num_pieces - 1, -1, -1):
        x0, x1 = breakpoints[i], breakpoints[i + 1]
        y0, y1 = values[i], values[i + 1]
        slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0.0
        segment = Add(Const(y0), Mul(Const(slope), Add(x, Neg(Const(x0)))))
        result = Ite(Lt(x, Const(x1)), segment, result)

        # Error bound: sample multiple points within each segment
        for frac in [0.25, 0.5, 0.75]:
            sample = x0 + frac * (x1 - x0)
            true_val = _hill_rep_value(sample, K_val, n_val)
            approx_val = y0 + slope * (sample - x0)
            max_error = max(max_error, abs(true_val - approx_val))

    error = ApproxError(
        method="piecewise_linear",
        max_absolute_error=max_error,
        domain=domain,
        num_pieces=num_pieces,
    )
    return result, error


# ---------------------------------------------------------------------------
# Non-integer Hill: Taylor series approximation
# ---------------------------------------------------------------------------

def _hill_act_taylor_coeffs(
    K_val: float,
    n_val: float,
    x0: float,
    order: int,
) -> List[float]:
    """Compute Taylor coefficients of HillAct around x=x0 via finite diffs."""
    h = 1e-6 * max(1.0, abs(x0))
    coeffs = [_hill_act_value(x0, K_val, n_val)]
    # Numerical derivatives via central differences
    for k in range(1, order + 1):
        # k-th derivative via finite differences
        deriv = 0.0
        for j in range(k + 1):
            sign = (-1) ** (k - j)
            binom = math.comb(k, j)
            deriv += sign * binom * _hill_act_value(x0 + j * h, K_val, n_val)
        deriv /= h ** k
        coeffs.append(deriv / math.factorial(k))
    return coeffs


def encode_hill_act_taylor(
    x: ExprNode,
    K_val: float,
    n_val: float,
    x0: float,
    order: int = 4,
    domain: Optional[Interval] = None,
) -> Tuple[ExprNode, ApproxError]:
    """Taylor series approximation of HillAct around x=x0.

    Returns polynomial: sum_{k=0}^{order} c_k * (x - x0)^k
    """
    coeffs = _hill_act_taylor_coeffs(K_val, n_val, x0, order)
    dx = Add(x, Neg(Const(x0)))

    terms: List[ExprNode] = []
    for k, c in enumerate(coeffs):
        if abs(c) < 1e-15:
            continue
        if k == 0:
            terms.append(Const(c))
        else:
            terms.append(Mul(Const(c), _integer_power(dx, k)))

    result = sum_exprs(terms) if terms else ZERO

    # Estimate error over domain
    max_err = 0.0
    if domain is not None:
        for i in range(21):
            xv = domain.lo + i * domain.width() / 20
            true_val = _hill_act_value(xv, K_val, n_val)
            approx_val = sum(c * (xv - x0) ** k for k, c in enumerate(coeffs))
            max_err = max(max_err, abs(true_val - approx_val))

    error = ApproxError(
        method="taylor",
        max_absolute_error=max_err,
        domain=domain,
        taylor_order=order,
    )
    return result, error


# ---------------------------------------------------------------------------
# Michaelis-Menten (Hill n=1)
# ---------------------------------------------------------------------------

def encode_michaelis_menten(
    substrate: ExprNode,
    vmax: ExprNode,
    km: ExprNode,
) -> ExprNode:
    """Michaelis-Menten kinetics: Vmax * S / (Km + S)."""
    return Mul(vmax, Div(substrate, Add(km, substrate)))


# ---------------------------------------------------------------------------
# Mass-action kinetics
# ---------------------------------------------------------------------------

def encode_mass_action(
    rate_constant: ExprNode,
    reactants: Sequence[Tuple[ExprNode, int]],
) -> ExprNode:
    """Mass-action kinetics: k * prod(x_i^{n_i}).

    *reactants* is a sequence of (species_expr, stoichiometry).
    Result is a polynomial expression.
    """
    factors: List[ExprNode] = [rate_constant]
    for species, stoich in reactants:
        factors.append(_integer_power(species, stoich))
    return prod_exprs(factors)


# ---------------------------------------------------------------------------
# Dimerization encoding
# ---------------------------------------------------------------------------

def encode_dimerization(
    monomer: ExprNode,
    kf: ExprNode,
    kr: ExprNode,
) -> Tuple[ExprNode, ExprNode]:
    """Dimerization: 2M <-> D with forward rate kf and reverse rate kr.

    Returns (formation_rate, dissociation_rate):
      formation = kf * M * M
      dissociation = kr * D
    where D is determined by the dimer variable.
    """
    formation = Mul(kf, Mul(monomer, monomer))
    dimer = Var(f"{monomer}_dimer") if isinstance(monomer, Var) else Var("dimer")
    dissociation = Mul(kr, dimer)
    return formation, dissociation


# ---------------------------------------------------------------------------
# Monotonicity-aware Hill encoding
# ---------------------------------------------------------------------------

def encode_hill_monotone_bounds(
    x_lo: ExprNode,
    x_hi: ExprNode,
    K: ExprNode,
    n: int,
    activating: bool = True,
) -> Tuple[ExprNode, ExprNode]:
    """Exploit monotonicity of Hill functions for interval bounds.

    For an activating Hill function (increasing in x):
      - Lower bound: HillAct(x_lo, K, n)
      - Upper bound: HillAct(x_hi, K, n)

    For a repressing Hill function (decreasing in x):
      - Lower bound: HillRep(x_hi, K, n)
      - Upper bound: HillRep(x_lo, K, n)

    Returns (lower_bound, upper_bound).
    """
    if activating:
        lo = encode_hill_act_integer(x_lo, K, n)
        hi = encode_hill_act_integer(x_hi, K, n)
    else:
        lo = encode_hill_rep_integer(x_hi, K, n)
        hi = encode_hill_rep_integer(x_lo, K, n)
    return lo, hi


# ---------------------------------------------------------------------------
# Auto-encode Hill nodes
# ---------------------------------------------------------------------------

@dataclass
class HillEncodingConfig:
    """Configuration for Hill function encoding."""
    pwl_pieces: int = 10
    taylor_order: int = 4
    prefer_pwl: bool = True


def encode_hill_node(
    node: ExprNode,
    domain: Optional[Interval] = None,
    config: Optional[HillEncodingConfig] = None,
) -> Tuple[ExprNode, List[ApproxError]]:
    """Encode a HillAct or HillRep node, choosing the best method.

    Returns (encoded_expr, list_of_approximation_errors).
    """
    cfg = config or HillEncodingConfig()
    errors: List[ApproxError] = []

    if isinstance(node, HillAct):
        nv = const_value(node.n)
        kv = const_value(node.K)
        if nv is not None and float(nv).is_integer() and nv >= 1:
            return encode_hill_act_integer(node.x, node.K, int(nv)), errors
        if nv is not None and kv is not None and domain is not None:
            if cfg.prefer_pwl:
                expr, err = encode_hill_act_pwl(
                    node.x, kv, nv, domain, cfg.pwl_pieces)
            else:
                expr, err = encode_hill_act_taylor(
                    node.x, kv, nv, domain.midpoint(), cfg.taylor_order, domain)
            errors.append(err)
            return expr, errors
        # Fall back to keeping the node as-is
        return node, errors

    if isinstance(node, HillRep):
        nv = const_value(node.n)
        kv = const_value(node.K)
        if nv is not None and float(nv).is_integer() and nv >= 1:
            return encode_hill_rep_integer(node.x, node.K, int(nv)), errors
        if nv is not None and kv is not None and domain is not None:
            if cfg.prefer_pwl:
                expr, err = encode_hill_rep_pwl(
                    node.x, kv, nv, domain, cfg.pwl_pieces)
            else:
                # Taylor for repressor via 1 - HillAct
                expr_act, err = encode_hill_act_taylor(
                    node.x, kv, nv, domain.midpoint(), cfg.taylor_order, domain)
                expr = Add(ONE, Neg(expr_act))
            errors.append(err)
            return expr, errors
        return node, errors

    return node, errors
