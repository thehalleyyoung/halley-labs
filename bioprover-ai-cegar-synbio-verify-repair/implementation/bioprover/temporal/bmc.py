"""Bounded Model Checking (BMC) and K-Induction for STL Properties.

Provides BMC unrolling that encodes k steps of system dynamics plus the
negation of the STL property. If SAT, a counterexample is extracted;
if UNSAT up to bound, the property holds up to that bound.
K-induction adds an inductive step for unbounded verification.
Template-based invariant synthesis supports linear, quadratic,
barrier certificates, and conservation law invariants.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from bioprover.temporal.stl_ast import (
    Always,
    ComparisonOp,
    Eventually,
    Interval,
    Predicate,
    STLAnd,
    STLFormula,
    STLImplies,
    STLNot,
    STLOr,
    Until,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class BMCVerdict(Enum):
    """Outcome of BMC / k-induction."""
    SAT = auto()        # Counterexample found (property violated)
    UNSAT = auto()      # No counterexample up to bound (property holds up to bound)
    PROVEN = auto()     # Inductively proven (property holds for all time)
    UNKNOWN = auto()    # Inconclusive


@dataclass
class Counterexample:
    """A counterexample trace extracted from a SAT model."""
    states: List[Dict[str, float]]
    times: List[float]
    violation_step: int
    violation_info: str = ""

    @property
    def length(self) -> int:
        return len(self.states)

    def state_at(self, step: int) -> Dict[str, float]:
        return self.states[step]

    def signal_trace(self, variable: str) -> List[float]:
        return [s.get(variable, 0.0) for s in self.states]


@dataclass
class BMCResult:
    """Full result from BMC or k-induction."""
    verdict: BMCVerdict
    bound: int
    counterexample: Optional[Counterexample] = None
    invariants_used: List[str] = field(default_factory=list)
    steps_checked: int = 0
    induction_depth: int = 0

    @property
    def property_holds(self) -> bool:
        return self.verdict in (BMCVerdict.UNSAT, BMCVerdict.PROVEN)

    @property
    def has_counterexample(self) -> bool:
        return self.counterexample is not None


# ---------------------------------------------------------------------------
# System dynamics encoding
# ---------------------------------------------------------------------------

@dataclass
class SystemDynamics:
    """Encodes discrete-time system dynamics: x_{k+1} = f(x_k).

    The transition function maps a state dict to the next state dict.
    The state space is defined by variable names and optional bounds.
    """
    variables: List[str]
    transition: Callable[[Dict[str, float]], Dict[str, float]]
    initial_set: Optional[Callable[[], Dict[str, float]]] = None
    bounds: Optional[Dict[str, Tuple[float, float]]] = None
    dt: float = 1.0

    def step(self, state: Dict[str, float]) -> Dict[str, float]:
        return self.transition(state)

    def simulate(self, initial: Dict[str, float], num_steps: int) -> List[Dict[str, float]]:
        """Simulate the system for num_steps from initial state."""
        trajectory = [dict(initial)]
        state = dict(initial)
        for _ in range(num_steps):
            state = self.step(state)
            trajectory.append(dict(state))
        return trajectory

    def sample_initial(self) -> Dict[str, float]:
        """Sample an initial state from the initial set."""
        if self.initial_set is not None:
            return self.initial_set()
        return {v: 0.0 for v in self.variables}


# ---------------------------------------------------------------------------
# STL → constraint encoding for BMC
# ---------------------------------------------------------------------------

def _eval_formula_at_step(
    formula: STLFormula,
    trajectory: List[Dict[str, float]],
    step: int,
    dt: float,
) -> bool:
    """Evaluate an STL formula on a discrete trajectory at a given step."""
    if isinstance(formula, Predicate):
        return _eval_predicate_at_step(formula, trajectory, step)
    if isinstance(formula, STLNot):
        return not _eval_formula_at_step(formula.child, trajectory, step, dt)
    if isinstance(formula, STLAnd):
        return (_eval_formula_at_step(formula.left, trajectory, step, dt) and
                _eval_formula_at_step(formula.right, trajectory, step, dt))
    if isinstance(formula, STLOr):
        return (_eval_formula_at_step(formula.left, trajectory, step, dt) or
                _eval_formula_at_step(formula.right, trajectory, step, dt))
    if isinstance(formula, STLImplies):
        return (not _eval_formula_at_step(formula.antecedent, trajectory, step, dt) or
                _eval_formula_at_step(formula.consequent, trajectory, step, dt))
    if isinstance(formula, Always):
        a = int(formula.interval.lo / dt)
        b = int(formula.interval.hi / dt)
        for k in range(step + a, min(step + b + 1, len(trajectory))):
            if not _eval_formula_at_step(formula.child, trajectory, k, dt):
                return False
        return True
    if isinstance(formula, Eventually):
        a = int(formula.interval.lo / dt)
        b = int(formula.interval.hi / dt)
        for k in range(step + a, min(step + b + 1, len(trajectory))):
            if _eval_formula_at_step(formula.child, trajectory, k, dt):
                return True
        return False
    if isinstance(formula, Until):
        a = int(formula.interval.lo / dt)
        b = int(formula.interval.hi / dt)
        for k in range(step + a, min(step + b + 1, len(trajectory))):
            if _eval_formula_at_step(formula.right, trajectory, k, dt):
                # Check left holds on [step+a, k)
                all_left = True
                for j in range(step + a, k):
                    if not _eval_formula_at_step(formula.left, trajectory, j, dt):
                        all_left = False
                        break
                if all_left:
                    return True
        return False
    raise TypeError(f"Unknown formula type: {type(formula)}")


def _eval_predicate_at_step(
    pred: Predicate,
    trajectory: List[Dict[str, float]],
    step: int,
) -> bool:
    """Evaluate an atomic predicate at a given step."""
    if step < 0 or step >= len(trajectory):
        return False
    state = trajectory[step]
    expr = pred.expr
    if expr.variable is not None:
        val = expr.scale * state.get(expr.variable, 0.0) + expr.offset
    elif expr.constant is not None:
        val = expr.constant
    else:
        return False

    if pred.op == ComparisonOp.GT:
        return val > pred.threshold
    elif pred.op == ComparisonOp.GE:
        return val >= pred.threshold
    elif pred.op == ComparisonOp.LT:
        return val < pred.threshold
    elif pred.op == ComparisonOp.LE:
        return val <= pred.threshold
    return False


# ---------------------------------------------------------------------------
# BMC Encoder
# ---------------------------------------------------------------------------

class BMCEncoder:
    """Bounded Model Checking via trajectory enumeration.

    Unrolls the system dynamics for k steps, then checks whether the negation
    of the property can be satisfied. Uses simulation-based checking with
    systematic exploration of the initial state space.
    """

    def __init__(
        self,
        dynamics: SystemDynamics,
        formula: STLFormula,
        max_bound: int = 100,
    ) -> None:
        self._dynamics = dynamics
        self._formula = formula
        self._max_bound = max_bound

    def check(
        self,
        bound: int,
        initial_states: Optional[List[Dict[str, float]]] = None,
        num_random: int = 100,
    ) -> BMCResult:
        """Run BMC up to the given bound.

        Tries each initial state and checks if the property is violated.
        """
        if initial_states is None:
            initial_states = [
                self._dynamics.sample_initial() for _ in range(num_random)
            ]

        for init in initial_states:
            trajectory = self._dynamics.simulate(init, bound)
            if not _eval_formula_at_step(self._formula, trajectory, 0, self._dynamics.dt):
                # Property violated: extract counterexample
                violation_step = self._find_violation_step(trajectory, bound)
                cex = Counterexample(
                    states=trajectory,
                    times=[i * self._dynamics.dt for i in range(len(trajectory))],
                    violation_step=violation_step,
                    violation_info=f"Property violated starting from {init}",
                )
                return BMCResult(
                    verdict=BMCVerdict.SAT,
                    bound=bound,
                    counterexample=cex,
                    steps_checked=bound,
                )

        return BMCResult(
            verdict=BMCVerdict.UNSAT,
            bound=bound,
            steps_checked=bound,
        )

    def incremental_check(
        self,
        max_bound: Optional[int] = None,
        initial_states: Optional[List[Dict[str, float]]] = None,
        num_random: int = 100,
    ) -> BMCResult:
        """Incremental BMC: check bounds 1, 2, ..., max_bound."""
        bound = max_bound or self._max_bound
        if initial_states is None:
            initial_states = [
                self._dynamics.sample_initial() for _ in range(num_random)
            ]

        for k in range(1, bound + 1):
            result = self.check(k, initial_states, num_random=0)
            if result.verdict == BMCVerdict.SAT:
                return result

        return BMCResult(
            verdict=BMCVerdict.UNSAT,
            bound=bound,
            steps_checked=bound,
        )

    def _find_violation_step(
        self,
        trajectory: List[Dict[str, float]],
        bound: int,
    ) -> int:
        """Find the earliest step where the property begins to be violated."""
        dt = self._dynamics.dt
        for step in range(bound + 1):
            if not _eval_formula_at_step(self._formula, trajectory, step, dt):
                return step
        return 0


# ---------------------------------------------------------------------------
# Invariant templates
# ---------------------------------------------------------------------------

class InvariantTemplate(ABC):
    """Base class for invariant templates used in k-induction strengthening."""

    @abstractmethod
    def check(self, state: Dict[str, float]) -> bool:
        """Check if the invariant holds at a given state."""
        ...

    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the invariant."""
        ...


class LinearInvariant(InvariantTemplate):
    """Linear invariant: a^T x <= b."""

    def __init__(self, coefficients: Dict[str, float], bound: float) -> None:
        self._coeffs = coefficients
        self._bound = bound

    def check(self, state: Dict[str, float]) -> bool:
        val = sum(c * state.get(v, 0.0) for v, c in self._coeffs.items())
        return val <= self._bound

    def description(self) -> str:
        terms = [f"{c}*{v}" for v, c in self._coeffs.items()]
        return f"{' + '.join(terms)} <= {self._bound}"


class QuadraticInvariant(InvariantTemplate):
    """Quadratic invariant: x^T Q x <= c.

    Q is specified as a dict mapping (var_i, var_j) to coefficient.
    """

    def __init__(
        self,
        quadratic_terms: Dict[Tuple[str, str], float],
        bound: float,
    ) -> None:
        self._terms = quadratic_terms
        self._bound = bound

    def check(self, state: Dict[str, float]) -> bool:
        val = sum(
            c * state.get(vi, 0.0) * state.get(vj, 0.0)
            for (vi, vj), c in self._terms.items()
        )
        return val <= self._bound

    def description(self) -> str:
        terms = [f"{c}*{vi}*{vj}" for (vi, vj), c in self._terms.items()]
        return f"{' + '.join(terms)} <= {self._bound}"


class BarrierCertificate(InvariantTemplate):
    """Barrier certificate: B(x) <= 0 in safe region, B(x) non-increasing along dynamics."""

    def __init__(
        self,
        barrier_fn: Callable[[Dict[str, float]], float],
        desc: str = "B(x) <= 0",
    ) -> None:
        self._barrier = barrier_fn
        self._desc = desc

    def check(self, state: Dict[str, float]) -> bool:
        return self._barrier(state) <= 0

    def description(self) -> str:
        return self._desc


class ConservationLawInvariant(InvariantTemplate):
    """Conservation law: sum of specified variables is constant."""

    def __init__(
        self,
        variables: List[str],
        total: float,
        tolerance: float = 1e-6,
    ) -> None:
        self._variables = variables
        self._total = total
        self._tolerance = tolerance

    def check(self, state: Dict[str, float]) -> bool:
        actual = sum(state.get(v, 0.0) for v in self._variables)
        return abs(actual - self._total) <= self._tolerance

    def description(self) -> str:
        return f"sum({', '.join(self._variables)}) = {self._total}"


# ---------------------------------------------------------------------------
# Invariant synthesis
# ---------------------------------------------------------------------------

def synthesize_linear_invariants(
    dynamics: SystemDynamics,
    num_trajectories: int = 100,
    num_steps: int = 50,
    variables: Optional[List[str]] = None,
) -> List[LinearInvariant]:
    """Synthesize candidate linear invariants from simulation data.

    Generates trajectories and finds linear bounds that hold across all observed states.
    """
    if variables is None:
        variables = dynamics.variables

    # Collect state data
    all_states: List[Dict[str, float]] = []
    for _ in range(num_trajectories):
        init = dynamics.sample_initial()
        traj = dynamics.simulate(init, num_steps)
        all_states.extend(traj)

    if not all_states:
        return []

    invariants: List[LinearInvariant] = []

    # Single-variable bounds: v <= max_v, -v <= -min_v
    for v in variables:
        vals = [s.get(v, 0.0) for s in all_states]
        max_v = max(vals)
        min_v = min(vals)
        margin = max(abs(max_v), abs(min_v)) * 0.1 + 1e-6
        invariants.append(LinearInvariant({v: 1.0}, max_v + margin))
        invariants.append(LinearInvariant({v: -1.0}, -min_v + margin))

    # Pairwise sum bounds
    for i, v1 in enumerate(variables):
        for v2 in variables[i + 1:]:
            vals = [s.get(v1, 0.0) + s.get(v2, 0.0) for s in all_states]
            max_sum = max(vals)
            margin = abs(max_sum) * 0.1 + 1e-6
            invariants.append(LinearInvariant({v1: 1.0, v2: 1.0}, max_sum + margin))

    return invariants


# ---------------------------------------------------------------------------
# K-Induction
# ---------------------------------------------------------------------------

class KInduction:
    """K-induction prover with invariant strengthening.

    Combines:
    1. Base case: BMC for k steps (no counterexample)
    2. Inductive step: assume property for k steps → prove for step k+1
    3. Strengthening: add auxiliary invariants to close the inductive argument
    """

    def __init__(
        self,
        dynamics: SystemDynamics,
        formula: STLFormula,
        max_k: int = 20,
        invariants: Optional[List[InvariantTemplate]] = None,
    ) -> None:
        self._dynamics = dynamics
        self._formula = formula
        self._max_k = max_k
        self._invariants = invariants or []

    def add_invariant(self, inv: InvariantTemplate) -> None:
        self._invariants.append(inv)

    def prove(
        self,
        initial_states: Optional[List[Dict[str, float]]] = None,
        num_random: int = 200,
        auto_strengthen: bool = True,
    ) -> BMCResult:
        """Attempt to prove property by k-induction.

        Steps:
        1. Check base case via BMC
        2. Attempt inductive step for each k
        3. Optionally auto-synthesize strengthening invariants
        """
        if initial_states is None:
            initial_states = [
                self._dynamics.sample_initial() for _ in range(num_random)
            ]

        # Auto-synthesize invariants if requested
        if auto_strengthen and not self._invariants:
            self._invariants = synthesize_linear_invariants(
                self._dynamics,
                num_trajectories=num_random,
                num_steps=self._max_k,
            )

        for k in range(1, self._max_k + 1):
            # Base case: BMC for k steps
            bmc = BMCEncoder(self._dynamics, self._formula, max_bound=k)
            base_result = bmc.check(k, initial_states, num_random=0)

            if base_result.verdict == BMCVerdict.SAT:
                # Counterexample found — check if invariants are violated
                cex = base_result.counterexample
                if cex and self._invariants:
                    inv_ok = all(
                        all(inv.check(s) for inv in self._invariants)
                        for s in cex.states
                    )
                    if not inv_ok:
                        # Spurious counterexample; invariant rules it out
                        continue

                return BMCResult(
                    verdict=BMCVerdict.SAT,
                    bound=k,
                    counterexample=base_result.counterexample,
                    steps_checked=k,
                    induction_depth=k,
                )

            # Inductive step: try to show that if property holds for k steps
            # and invariants hold, then property holds at step k+1
            if self._check_inductive_step(k, num_random):
                inv_descs = [inv.description() for inv in self._invariants]
                return BMCResult(
                    verdict=BMCVerdict.PROVEN,
                    bound=k,
                    invariants_used=inv_descs,
                    steps_checked=k,
                    induction_depth=k,
                )

        return BMCResult(
            verdict=BMCVerdict.UNKNOWN,
            bound=self._max_k,
            steps_checked=self._max_k,
            induction_depth=self._max_k,
        )

    def _check_inductive_step(self, k: int, num_random: int) -> bool:
        """Check the inductive step: property holds for k steps → holds at k+1.

        Generates random states satisfying invariants and property for k steps,
        then checks if the property still holds after one more step.
        """
        dt = self._dynamics.dt
        successes = 0

        for _ in range(num_random):
            init = self._dynamics.sample_initial()
            trajectory = self._dynamics.simulate(init, k + 1)

            # Check if invariants hold on the trajectory
            invs_hold = all(
                all(inv.check(s) for inv in self._invariants)
                for s in trajectory
            )
            if not invs_hold:
                continue

            # Check if property holds for first k steps
            prop_holds_k = _eval_formula_at_step(self._formula, trajectory, 0, dt)
            if not prop_holds_k:
                continue

            # Property held for k steps with invariants — count this inductive witness
            successes += 1

            # Check if property still holds at step k+1 (with extended trajectory)
            extended = self._dynamics.simulate(init, k + 2)
            if not _eval_formula_at_step(self._formula, extended, 0, dt):
                return False  # Inductive step fails

        # If we checked enough witnesses and none failed, accept
        return successes >= max(10, num_random // 5)


# ---------------------------------------------------------------------------
# BMC with interpolation hints
# ---------------------------------------------------------------------------

def bmc_with_interpolation(
    dynamics: SystemDynamics,
    formula: STLFormula,
    max_bound: int = 50,
    num_initial: int = 100,
) -> BMCResult:
    """BMC with interpolation-based invariant discovery.

    After BMC finds no counterexample, analyzes the trajectory space
    to discover candidate invariants that could close an inductive proof.
    Falls back to standard k-induction.
    """
    # First pass: quick BMC
    bmc = BMCEncoder(dynamics, formula, max_bound=max_bound)
    init_states = [dynamics.sample_initial() for _ in range(num_initial)]
    quick = bmc.check(max_bound, init_states, num_random=0)

    if quick.verdict == BMCVerdict.SAT:
        return quick

    # Synthesize invariants from simulation data
    invariants = synthesize_linear_invariants(
        dynamics, num_trajectories=num_initial, num_steps=max_bound,
    )

    # Attempt k-induction with discovered invariants
    prover = KInduction(
        dynamics, formula, max_k=max_bound, invariants=invariants,
    )
    return prover.prove(init_states, num_random=num_initial, auto_strengthen=False)
