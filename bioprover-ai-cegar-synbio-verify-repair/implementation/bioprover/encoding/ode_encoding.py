"""ODE discretization encoding for bounded model checking of biological ODEs.

Encodes ODE systems as SMT constraints via Euler or trapezoidal discretization,
with support for incremental unrolling, conservation laws, and positivity.
"""

from __future__ import annotations

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
    ForAll,
    Ge,
    Interval,
    Le,
    Mul,
    Neg,
    Var,
    ZERO,
    ONE,
    TWO,
    const,
    sum_exprs,
    var,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class DiscretizationMethod(Enum):
    EULER = auto()
    TRAPEZOIDAL = auto()


@dataclass(frozen=True)
class Species:
    """A state variable (molecular species) in the ODE system."""
    name: str
    initial_lo: float = 0.0
    initial_hi: float = 0.0
    nonnegative: bool = True

    @property
    def point_initial(self) -> bool:
        return self.initial_lo == self.initial_hi


@dataclass(frozen=True)
class Parameter:
    """A model parameter with uncertainty range."""
    name: str
    lo: float
    hi: float

    @property
    def interval(self) -> Interval:
        return Interval(self.lo, self.hi)

    @property
    def is_point(self) -> bool:
        return self.lo == self.hi


@dataclass(frozen=True)
class ConservationLaw:
    """Linear conservation law: sum of coeffs[i]*species[i] = total."""
    coefficients: Dict[str, float]
    total_expr: ExprNode


@dataclass
class ODESystem:
    """Complete ODE system specification."""
    species: List[Species]
    parameters: List[Parameter]
    # RHS functions: species_name -> f(state_vars, params)
    rhs: Dict[str, Callable[..., ExprNode]]
    conservation_laws: List[ConservationLaw] = field(default_factory=list)

    @property
    def species_names(self) -> List[str]:
        return [s.name for s in self.species]

    @property
    def parameter_names(self) -> List[str]:
        return [p.name for p in self.parameters]

    def eval_rhs(self, species_name: str, state: Dict[str, ExprNode],
                 params: Dict[str, ExprNode]) -> ExprNode:
        """Evaluate the RHS for a given species at the given state/params."""
        all_vars = {**state, **params}
        return self.rhs[species_name](**{k: v for k, v in all_vars.items()
                                          if k in self.rhs[species_name].__code__.co_varnames})


# ---------------------------------------------------------------------------
# Variable naming
# ---------------------------------------------------------------------------

def state_var(species: str, step: int) -> Var:
    """Create a time-indexed state variable: e.g. x_3."""
    return Var(f"{species}_{step}")


def param_var(name: str) -> Var:
    return Var(name)


# ---------------------------------------------------------------------------
# Encoding size estimation
# ---------------------------------------------------------------------------

@dataclass
class EncodingSizeEstimate:
    """Predicted encoding size metrics."""
    num_variables: int = 0
    num_constraints: int = 0
    max_degree: int = 0
    num_nonlinear: int = 0

    def __repr__(self) -> str:
        return (f"EncodingSize(vars={self.num_variables}, "
                f"constraints={self.num_constraints}, "
                f"degree={self.max_degree}, "
                f"nonlinear={self.num_nonlinear})")


def estimate_encoding_size(
    system: ODESystem,
    num_steps: int,
    method: DiscretizationMethod = DiscretizationMethod.EULER,
) -> EncodingSizeEstimate:
    """Estimate the size of the SMT encoding before generating it."""
    n_species = len(system.species)
    n_params = len(system.parameters)
    n_cons = len(system.conservation_laws)
    effective_species = n_species - n_cons

    n_vars = effective_species * (num_steps + 1) + n_params
    # Each step produces one constraint per effective species
    n_constraints = effective_species * num_steps
    # Positivity
    n_constraints += n_species * (num_steps + 1) if any(s.nonnegative for s in system.species) else 0
    # Initial conditions
    n_constraints += n_species
    # Parameter bounds
    n_constraints += 2 * n_params

    multiplier = 2 if method == DiscretizationMethod.TRAPEZOIDAL else 1

    return EncodingSizeEstimate(
        num_variables=n_vars,
        num_constraints=n_constraints * multiplier,
        max_degree=0,  # depends on RHS
        num_nonlinear=0,
    )


# ---------------------------------------------------------------------------
# Time step selection heuristics
# ---------------------------------------------------------------------------

def select_time_step(
    time_horizon: float,
    target_steps: int = 100,
    max_step: float = 0.1,
    min_step: float = 1e-6,
) -> Tuple[float, int]:
    """Choose time step size and number of steps.

    Returns (step_size, num_steps).
    """
    h = time_horizon / target_steps
    h = max(min_step, min(max_step, h))
    n = int(math.ceil(time_horizon / h))
    return h, n


# ---------------------------------------------------------------------------
# Euler discretization encoder
# ---------------------------------------------------------------------------

@dataclass
class ODEEncoding:
    """Result of encoding an ODE system as SMT constraints."""
    variables: List[Var]
    constraints: List[ExprNode]
    step_constraints: Dict[int, List[ExprNode]]
    initial_constraints: List[ExprNode]
    parameter_constraints: List[ExprNode]
    positivity_constraints: List[ExprNode]
    conservation_constraints: List[ExprNode]
    num_steps: int
    step_size: float
    method: DiscretizationMethod

    @property
    def all_constraints(self) -> List[ExprNode]:
        return (self.initial_constraints +
                self.parameter_constraints +
                self.constraints +
                self.positivity_constraints +
                self.conservation_constraints)


def _make_state(system: ODESystem, step: int) -> Dict[str, ExprNode]:
    """Create state variable dictionary for a given time step."""
    return {s.name: state_var(s.name, step) for s in system.species}


def _make_params(system: ODESystem) -> Dict[str, ExprNode]:
    """Create parameter variable dictionary."""
    return {p.name: param_var(p.name) for p in system.parameters}


def _euler_step(
    system: ODESystem,
    step: int,
    h: ExprNode,
    eliminated: Set[str],
) -> List[ExprNode]:
    """Generate Euler discretization constraints for one time step.

    x_{k+1} = x_k + h * f(x_k, p)
    """
    state_k = _make_state(system, step)
    params = _make_params(system)
    constraints: List[ExprNode] = []

    for sp in system.species:
        if sp.name in eliminated:
            continue
        x_k = state_k[sp.name]
        f_k = system.eval_rhs(sp.name, state_k, params)
        x_next = state_var(sp.name, step + 1)
        # x_{k+1} = x_k + h * f_k
        rhs = Add(x_k, Mul(h, f_k))
        constraints.append(Le(Neg(Const(1e-15)), Add(x_next, Neg(rhs))))
        constraints.append(Le(Add(x_next, Neg(rhs)), Const(1e-15)))

    return constraints


def _trapezoidal_step(
    system: ODESystem,
    step: int,
    h: ExprNode,
    eliminated: Set[str],
) -> List[ExprNode]:
    """Generate trapezoidal discretization constraints for one time step.

    x_{k+1} = x_k + (h/2) * (f(x_k, p) + f(x_{k+1}, p))
    """
    state_k = _make_state(system, step)
    state_k1 = _make_state(system, step + 1)
    params = _make_params(system)
    half_h = Div(h, TWO)
    constraints: List[ExprNode] = []

    for sp in system.species:
        if sp.name in eliminated:
            continue
        x_k = state_k[sp.name]
        x_k1 = state_k1[sp.name]
        f_k = system.eval_rhs(sp.name, state_k, params)
        f_k1 = system.eval_rhs(sp.name, state_k1, params)
        rhs = Add(x_k, Mul(half_h, Add(f_k, f_k1)))
        constraints.append(Le(Neg(Const(1e-15)), Add(x_k1, Neg(rhs))))
        constraints.append(Le(Add(x_k1, Neg(rhs)), Const(1e-15)))

    return constraints


def _initial_conditions(system: ODESystem) -> List[ExprNode]:
    """Encode initial condition constraints."""
    constraints: List[ExprNode] = []
    for sp in system.species:
        x0 = state_var(sp.name, 0)
        if sp.point_initial:
            constraints.append(Le(Neg(Const(1e-15)), Add(x0, Neg(Const(sp.initial_lo)))))
            constraints.append(Le(Add(x0, Neg(Const(sp.initial_lo))), Const(1e-15)))
        else:
            constraints.append(Ge(x0, Const(sp.initial_lo)))
            constraints.append(Le(x0, Const(sp.initial_hi)))
    return constraints


def _parameter_bounds(system: ODESystem) -> List[ExprNode]:
    """Encode parameter range constraints."""
    constraints: List[ExprNode] = []
    for p in system.parameters:
        pv = param_var(p.name)
        constraints.append(Ge(pv, Const(p.lo)))
        constraints.append(Le(pv, Const(p.hi)))
    return constraints


def _positivity_constraints(system: ODESystem, num_steps: int) -> List[ExprNode]:
    """x_k >= 0 for all nonnegative species at all time steps."""
    constraints: List[ExprNode] = []
    for sp in system.species:
        if sp.nonnegative:
            for k in range(num_steps + 1):
                constraints.append(Ge(state_var(sp.name, k), ZERO))
    return constraints


def _conservation_constraints(
    system: ODESystem,
    num_steps: int,
) -> Tuple[List[ExprNode], Set[str]]:
    """Encode conservation laws, returning constraints and eliminated species."""
    constraints: List[ExprNode] = []
    eliminated: Set[str] = set()

    for law in system.conservation_laws:
        # Find species to eliminate (last one with nonzero coefficient)
        elim_name: Optional[str] = None
        for name, coeff in law.coefficients.items():
            if coeff != 0.0:
                elim_name = name
        if elim_name is None:
            continue
        eliminated.add(elim_name)

        elim_coeff = law.coefficients[elim_name]
        for k in range(num_steps + 1):
            # elim_var = (total - sum(coeff_i * x_i for other i)) / elim_coeff
            other_sum = sum_exprs([
                Mul(Const(c), state_var(n, k))
                for n, c in law.coefficients.items()
                if n != elim_name
            ])
            elim_val = Div(Add(law.total_expr, Neg(other_sum)), Const(elim_coeff))
            elim_var = state_var(elim_name, k)
            constraints.append(Le(Neg(Const(1e-15)), Add(elim_var, Neg(elim_val))))
            constraints.append(Le(Add(elim_var, Neg(elim_val)), Const(1e-15)))

    return constraints, eliminated


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------

def encode_ode(
    system: ODESystem,
    num_steps: int,
    step_size: float,
    method: DiscretizationMethod = DiscretizationMethod.EULER,
) -> ODEEncoding:
    """Encode the full ODE system for bounded model checking."""
    h = Const(step_size)

    # Conservation laws
    cons_constraints, eliminated = _conservation_constraints(system, num_steps)

    # Collect all variables
    variables: List[Var] = []
    for sp in system.species:
        for k in range(num_steps + 1):
            variables.append(state_var(sp.name, k))
    for p in system.parameters:
        variables.append(param_var(p.name))

    # Discretization step constraints
    step_fn = _euler_step if method == DiscretizationMethod.EULER else _trapezoidal_step
    all_step_constraints: List[ExprNode] = []
    step_map: Dict[int, List[ExprNode]] = {}
    for k in range(num_steps):
        cs = step_fn(system, k, h, eliminated)
        all_step_constraints.extend(cs)
        step_map[k] = cs

    return ODEEncoding(
        variables=variables,
        constraints=all_step_constraints,
        step_constraints=step_map,
        initial_constraints=_initial_conditions(system),
        parameter_constraints=_parameter_bounds(system),
        positivity_constraints=_positivity_constraints(system, num_steps),
        conservation_constraints=cons_constraints,
        num_steps=num_steps,
        step_size=step_size,
        method=method,
    )


# ---------------------------------------------------------------------------
# Incremental encoding
# ---------------------------------------------------------------------------

def encode_ode_incremental(
    system: ODESystem,
    current_steps: int,
    additional_steps: int,
    step_size: float,
    method: DiscretizationMethod = DiscretizationMethod.EULER,
) -> Tuple[List[Var], List[ExprNode]]:
    """Add *additional_steps* to an existing encoding.

    Returns (new_variables, new_constraints) that should be conjoined with
    the existing encoding.
    """
    h = Const(step_size)
    _, eliminated = _conservation_constraints(system, current_steps + additional_steps)
    step_fn = _euler_step if method == DiscretizationMethod.EULER else _trapezoidal_step

    new_vars: List[Var] = []
    new_constraints: List[ExprNode] = []

    for k in range(current_steps, current_steps + additional_steps):
        cs = step_fn(system, k, h, eliminated)
        new_constraints.extend(cs)
        for sp in system.species:
            new_vars.append(state_var(sp.name, k + 1))
            if sp.nonnegative:
                new_constraints.append(Ge(state_var(sp.name, k + 1), ZERO))

    return new_vars, new_constraints


# ---------------------------------------------------------------------------
# Convenience: encode with automatic step selection
# ---------------------------------------------------------------------------

def encode_ode_auto(
    system: ODESystem,
    time_horizon: float,
    method: DiscretizationMethod = DiscretizationMethod.EULER,
    target_steps: int = 100,
    max_step: float = 0.1,
) -> ODEEncoding:
    """Encode with automatically selected step size."""
    h, n = select_time_step(time_horizon, target_steps, max_step)
    return encode_ode(system, n, h, method)
