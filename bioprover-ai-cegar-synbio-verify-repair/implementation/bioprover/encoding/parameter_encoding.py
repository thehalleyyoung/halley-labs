"""Parameter uncertainty encoding for robust verification and synthesis.

Provides quantifier encodings for universal (verification), existential
(synthesis), and mixed (robust synthesis) queries over parameter spaces,
plus discretization and Skolemization strategies.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from .expression import (
    Add,
    And,
    Const,
    Div,
    ExprNode,
    Exists,
    ForAll,
    Ge,
    Interval,
    Le,
    Mul,
    Neg,
    Or,
    Var,
    ZERO,
    ONE,
    const_value,
    sum_exprs,
    var,
)


# ---------------------------------------------------------------------------
# Parameter roles
# ---------------------------------------------------------------------------

class ParameterRole(Enum):
    """Role of a parameter in quantifier prefix."""
    UNIVERSAL = auto()    # forall (environment uncertainty)
    EXISTENTIAL = auto()  # exists (design choice)
    FIXED = auto()        # no quantifier (known constant)


@dataclass(frozen=True)
class QuantifiedParam:
    """A parameter with its quantification role and bounds."""
    name: str
    role: ParameterRole
    lo: float
    hi: float

    @property
    def interval(self) -> Interval:
        return Interval(self.lo, self.hi)

    @property
    def is_point(self) -> bool:
        return self.lo == self.hi

    def var(self) -> Var:
        return Var(self.name)


# ---------------------------------------------------------------------------
# Universal quantification (verification)
# ---------------------------------------------------------------------------

def encode_universal(
    params: Sequence[QuantifiedParam],
    body: ExprNode,
) -> ExprNode:
    """Wrap *body* in universal quantifiers for all UNIVERSAL params.

    forall p1 in [lo1, hi1]. forall p2 in [lo2, hi2]. body
    """
    result = body
    for p in reversed(params):
        if p.role == ParameterRole.UNIVERSAL:
            result = ForAll(p.name, p.interval, result)
    return result


# ---------------------------------------------------------------------------
# Existential quantification (synthesis)
# ---------------------------------------------------------------------------

def encode_existential(
    params: Sequence[QuantifiedParam],
    body: ExprNode,
) -> ExprNode:
    """Wrap *body* in existential quantifiers for all EXISTENTIAL params.

    exists p1 in [lo1, hi1]. exists p2 in [lo2, hi2]. body
    """
    result = body
    for p in reversed(params):
        if p.role == ParameterRole.EXISTENTIAL:
            result = Exists(p.name, p.interval, result)
    return result


# ---------------------------------------------------------------------------
# Mixed quantifier prefix (robust synthesis)
# ---------------------------------------------------------------------------

def encode_robust_synthesis(
    design_params: Sequence[QuantifiedParam],
    uncertain_params: Sequence[QuantifiedParam],
    body: ExprNode,
) -> ExprNode:
    """Encode: exists p_design. forall p_uncertain. body

    This is the standard robust synthesis query: find design parameters
    such that the specification holds for all uncertain parameters.
    """
    # Inner: forall uncertain
    inner = body
    for p in reversed(uncertain_params):
        inner = ForAll(p.name, p.interval, inner)
    # Outer: exists design
    for p in reversed(design_params):
        inner = Exists(p.name, p.interval, inner)
    return inner


# ---------------------------------------------------------------------------
# Parameter bound constraints (quantifier-free)
# ---------------------------------------------------------------------------

def parameter_bound_constraints(
    params: Sequence[QuantifiedParam],
) -> List[ExprNode]:
    """Generate lo <= p <= hi constraints for each parameter."""
    constraints: List[ExprNode] = []
    for p in params:
        if p.is_point:
            # Equality constraint
            pv = p.var()
            constraints.append(Ge(pv, Const(p.lo)))
            constraints.append(Le(pv, Const(p.hi)))
        else:
            pv = p.var()
            constraints.append(Ge(pv, Const(p.lo)))
            constraints.append(Le(pv, Const(p.hi)))
    return constraints


# ---------------------------------------------------------------------------
# Parameter space discretization
# ---------------------------------------------------------------------------

@dataclass
class DiscretizationResult:
    """Result of parameter space discretization."""
    grid_points: List[Dict[str, float]]
    constraints_per_point: List[ExprNode]
    num_points: int


def discretize_parameter_space(
    params: Sequence[QuantifiedParam],
    body: ExprNode,
    points_per_dim: int = 5,
) -> DiscretizationResult:
    """Discretize universal quantification into a finite conjunction.

    Replaces: forall p in [lo, hi]. body
    With:     AND_{p_i in grid} body[p := p_i]

    This is sound for monotone properties and an over-approximation otherwise.
    """
    universal = [p for p in params if p.role == ParameterRole.UNIVERSAL]
    if not universal:
        return DiscretizationResult(
            grid_points=[{}],
            constraints_per_point=[body],
            num_points=1,
        )

    # Build grid
    dim_values: List[List[Tuple[str, float]]] = []
    for p in universal:
        if points_per_dim == 1:
            vals = [(p.name, (p.lo + p.hi) / 2.0)]
        else:
            step = (p.hi - p.lo) / (points_per_dim - 1) if points_per_dim > 1 else 0
            vals = [(p.name, p.lo + i * step) for i in range(points_per_dim)]
        dim_values.append(vals)

    grid_points: List[Dict[str, float]] = []
    constraints: List[ExprNode] = []

    for combo in itertools.product(*dim_values):
        point = {name: val for name, val in combo}
        grid_points.append(point)
        mapping = {name: Const(val) for name, val in point.items()}
        constraints.append(body.substitute(mapping))

    return DiscretizationResult(
        grid_points=grid_points,
        constraints_per_point=constraints,
        num_points=len(grid_points),
    )


# ---------------------------------------------------------------------------
# Corner-based encoding for monotone parameters
# ---------------------------------------------------------------------------

def encode_corners(
    monotone_params: Sequence[Tuple[QuantifiedParam, bool]],
    body: ExprNode,
) -> ExprNode:
    """Corner-based encoding for parameters with known monotonicity.

    For each parameter, *monotone_increasing* indicates whether the
    property is monotone increasing in that parameter.  If so, checking
    the lower and upper bounds suffices.

    Returns a conjunction over all 2^n corners.
    """
    if not monotone_params:
        return body

    corners: List[Dict[str, float]] = [{}]
    for param, increasing in monotone_params:
        new_corners: List[Dict[str, float]] = []
        for corner in corners:
            lo_corner = {**corner, param.name: param.lo}
            hi_corner = {**corner, param.name: param.hi}
            new_corners.extend([lo_corner, hi_corner])
        corners = new_corners

    substituted: List[ExprNode] = []
    for corner in corners:
        mapping = {name: Const(val) for name, val in corner.items()}
        substituted.append(body.substitute(mapping))

    return And(*substituted)


# ---------------------------------------------------------------------------
# Interval subdivision encoding
# ---------------------------------------------------------------------------

def encode_interval_subdivision(
    param: QuantifiedParam,
    body: ExprNode,
    num_subdivisions: int = 4,
) -> Tuple[ExprNode, List[Interval]]:
    """Subdivide the parameter interval and encode each sub-interval.

    Returns (conjunction_expr, list_of_sub_intervals).
    """
    intervals = param.interval.subdivide(num_subdivisions)
    sub_constraints: List[ExprNode] = []

    for sub in intervals:
        pv = param.var()
        bound_conds = And(Ge(pv, Const(sub.lo)), Le(pv, Const(sub.hi)))
        sub_body = ForAll(param.name, sub, body)
        sub_constraints.append(sub_body)

    return And(*sub_constraints), intervals


# ---------------------------------------------------------------------------
# Skolemization
# ---------------------------------------------------------------------------

def skolemize(
    expr: ExprNode,
    universal_vars: Optional[Set[str]] = None,
) -> Tuple[ExprNode, Dict[str, ExprNode]]:
    """Skolemize existential quantifiers within universal scope.

    For exists y. forall x. P(x, y), introduce Skolem constant c_y
    and return forall x. P(x, c_y).

    For forall x. exists y. P(x, y), introduce Skolem function
    y = f_y(x) and return forall x. P(x, f_y(x)).

    Returns (skolemized_expr, skolem_mapping).
    """
    uv = universal_vars or set()
    skolem_map: Dict[str, ExprNode] = {}

    def _skolem_inner(e: ExprNode, bound_universal: List[str]) -> ExprNode:
        if isinstance(e, Exists):
            if bound_universal:
                # Skolem function: use a fresh variable representing f(x1,...,xk)
                fname = f"sk_{e.var}_{'_'.join(bound_universal)}"
                skolem_var = Var(fname)
                skolem_map[e.var] = skolem_var
            else:
                # Skolem constant
                fname = f"sk_{e.var}"
                skolem_var = Var(fname)
                skolem_map[e.var] = skolem_var
            # Replace the bound variable in the body
            new_body = e.body.substitute({e.var: skolem_var})
            return _skolem_inner(new_body, bound_universal)

        if isinstance(e, ForAll):
            new_bound = bound_universal + [e.var]
            new_body = _skolem_inner(e.body, new_bound)
            return ForAll(e.var, e.domain, new_body)

        return e

    result = _skolem_inner(expr, [])
    return result, skolem_map


# ---------------------------------------------------------------------------
# Parameter constraint generation
# ---------------------------------------------------------------------------

@dataclass
class ParameterConstraints:
    """Collected constraints for a parameter encoding."""
    bound_constraints: List[ExprNode]
    quantified_body: ExprNode
    skolem_map: Dict[str, ExprNode]
    discretization: Optional[DiscretizationResult]
    num_universal: int
    num_existential: int


def generate_parameter_constraints(
    params: Sequence[QuantifiedParam],
    body: ExprNode,
    strategy: str = "quantified",
    grid_points: int = 5,
) -> ParameterConstraints:
    """Generate parameter constraints using the specified strategy.

    Strategies:
      - "quantified": use ForAll/Exists quantifiers
      - "discretized": discretize universal params on a grid
      - "corners": use corner points (for monotone params)
      - "skolemized": Skolemize existentials
    """
    bounds = parameter_bound_constraints(params)
    n_univ = sum(1 for p in params if p.role == ParameterRole.UNIVERSAL)
    n_exist = sum(1 for p in params if p.role == ParameterRole.EXISTENTIAL)

    if strategy == "quantified":
        design = [p for p in params if p.role == ParameterRole.EXISTENTIAL]
        uncertain = [p for p in params if p.role == ParameterRole.UNIVERSAL]
        qbody = encode_robust_synthesis(design, uncertain, body) if (design and uncertain) \
            else encode_universal(params, body) if uncertain \
            else encode_existential(params, body) if design \
            else body
        return ParameterConstraints(
            bound_constraints=bounds,
            quantified_body=qbody,
            skolem_map={},
            discretization=None,
            num_universal=n_univ,
            num_existential=n_exist,
        )

    if strategy == "discretized":
        disc = discretize_parameter_space(params, body, grid_points)
        conj = And(*disc.constraints_per_point) if len(disc.constraints_per_point) > 1 \
            else disc.constraints_per_point[0]
        return ParameterConstraints(
            bound_constraints=bounds,
            quantified_body=conj,
            skolem_map={},
            discretization=disc,
            num_universal=n_univ,
            num_existential=n_exist,
        )

    if strategy == "corners":
        monotone = [(p, True) for p in params if p.role == ParameterRole.UNIVERSAL]
        corner_expr = encode_corners(monotone, body)
        return ParameterConstraints(
            bound_constraints=bounds,
            quantified_body=corner_expr,
            skolem_map={},
            discretization=None,
            num_universal=n_univ,
            num_existential=n_exist,
        )

    if strategy == "skolemized":
        full = encode_robust_synthesis(
            [p for p in params if p.role == ParameterRole.EXISTENTIAL],
            [p for p in params if p.role == ParameterRole.UNIVERSAL],
            body,
        )
        sk_expr, sk_map = skolemize(full)
        return ParameterConstraints(
            bound_constraints=bounds,
            quantified_body=sk_expr,
            skolem_map=sk_map,
            discretization=None,
            num_universal=n_univ,
            num_existential=n_exist,
        )

    raise ValueError(f"Unknown strategy: {strategy}")
