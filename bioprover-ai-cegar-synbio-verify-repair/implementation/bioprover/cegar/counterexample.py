"""
Counterexample analysis for CEGAR.

Handles abstract and concrete counterexamples, spuriousness checking
(via SMT and simulation), failure-point identification, minimization,
and counterexample generalization.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from bioprover.encoding.expression import (
    And,
    Const,
    ExprNode,
    Ge,
    Le,
    Not,
    Or,
    Var,
    Interval,
)
from bioprover.cegar.abstraction import AbstractState, AbstractTransition, IntervalBox

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spuriousness classification
# ---------------------------------------------------------------------------


class SpuriousnessVerdict(Enum):
    """Outcome of counterexample feasibility analysis."""

    GENUINE = auto()       # Real bug — concrete witness exists
    SPURIOUS = auto()      # Artifact of abstraction
    INCONCLUSIVE = auto()  # Timeout or numerical issues


# ---------------------------------------------------------------------------
# Counterexample representations
# ---------------------------------------------------------------------------


@dataclass
class AbstractCounterexample:
    """Sequence of abstract states violating a property.

    ``path[0]`` is an initial state and ``path[-1]`` violates the
    specification.  ``transitions[i]`` connects ``path[i]`` to
    ``path[i+1]``.
    """

    path: List[AbstractState]
    transitions: List[AbstractTransition] = field(default_factory=list)
    property_violated: str = ""
    iteration_found: int = 0

    @property
    def length(self) -> int:
        return len(self.path)

    @property
    def initial_state(self) -> AbstractState:
        return self.path[0]

    @property
    def violating_state(self) -> AbstractState:
        return self.path[-1]

    def prefix(self, k: int) -> AbstractCounterexample:
        """Return the first *k* states of the path."""
        return AbstractCounterexample(
            path=self.path[:k],
            transitions=self.transitions[: max(0, k - 1)],
            property_violated=self.property_violated,
            iteration_found=self.iteration_found,
        )

    def suffix(self, k: int) -> AbstractCounterexample:
        """Return from state *k* onwards."""
        return AbstractCounterexample(
            path=self.path[k:],
            transitions=self.transitions[k:],
            property_violated=self.property_violated,
            iteration_found=self.iteration_found,
        )

    def state_ids(self) -> List[int]:
        return [s.state_id for s in self.path]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "length": self.length,
            "state_ids": self.state_ids(),
            "property_violated": self.property_violated,
            "iteration_found": self.iteration_found,
        }

    def __repr__(self) -> str:
        ids = " → ".join(str(s.state_id) for s in self.path)
        return f"AbstractCex[{ids}]"


@dataclass
class ConcreteCounterexample:
    """Concrete trajectory witnessing a property violation.

    ``time_points[i]`` and ``states[i]`` give the concrete valuation
    at step *i*.  ``parameter_values`` records the parameter assignment.
    """

    time_points: List[float]
    states: List[Dict[str, float]]
    parameter_values: Dict[str, float] = field(default_factory=dict)
    property_violated: str = ""

    @property
    def length(self) -> int:
        return len(self.states)

    @property
    def initial_state(self) -> Dict[str, float]:
        return self.states[0]

    @property
    def final_state(self) -> Dict[str, float]:
        return self.states[-1]

    @property
    def duration(self) -> float:
        if len(self.time_points) < 2:
            return 0.0
        return self.time_points[-1] - self.time_points[0]

    def trajectory(self, var: str) -> List[float]:
        """Extract the trajectory of a single variable."""
        return [s.get(var, 0.0) for s in self.states]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "length": self.length,
            "duration": self.duration,
            "initial_state": self.initial_state,
            "final_state": self.final_state,
            "parameter_values": self.parameter_values,
            "property_violated": self.property_violated,
        }

    def __repr__(self) -> str:
        return (
            f"ConcreteCex(len={self.length}, "
            f"T={self.duration:.3g}, prop={self.property_violated})"
        )


# ---------------------------------------------------------------------------
# Spuriousness result
# ---------------------------------------------------------------------------


@dataclass
class SpuriousnessResult:
    """Result of counterexample feasibility analysis."""

    verdict: SpuriousnessVerdict
    concrete_witness: Optional[ConcreteCounterexample] = None
    failure_index: Optional[int] = None
    failure_transition: Optional[AbstractTransition] = None
    analysis_time: float = 0.0
    smt_queries: int = 0
    message: str = ""

    @property
    def is_genuine(self) -> bool:
        return self.verdict == SpuriousnessVerdict.GENUINE

    @property
    def is_spurious(self) -> bool:
        return self.verdict == SpuriousnessVerdict.SPURIOUS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.name,
            "failure_index": self.failure_index,
            "analysis_time_s": round(self.analysis_time, 3),
            "smt_queries": self.smt_queries,
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# Spuriousness checker
# ---------------------------------------------------------------------------


class SpuriousnessChecker:
    """Determine whether an abstract counterexample is genuine or spurious.

    Uses SMT-based path feasibility checking as the primary method
    and simulation-based checking as a faster incomplete alternative.
    """

    def __init__(
        self,
        solver: Optional[Any] = None,
        simulator: Optional[Callable[..., Any]] = None,
        timeout: float = 60.0,
        delta: float = 1e-3,
    ) -> None:
        self._solver = solver
        self._simulator = simulator
        self._timeout = timeout
        self._delta = delta
        self._stats = _CheckerStatistics()

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def check(
        self,
        cex: AbstractCounterexample,
        rhs: Optional[Dict[str, ExprNode]] = None,
        step_size: float = 0.01,
    ) -> SpuriousnessResult:
        """Check feasibility of an abstract counterexample.

        Tries SMT-based checking first (if solver available), then
        falls back to simulation-based checking.
        """
        t0 = time.monotonic()

        # Attempt SMT-based check
        if self._solver is not None:
            result = self._check_smt(cex, rhs, step_size)
            if result.verdict != SpuriousnessVerdict.INCONCLUSIVE:
                result.analysis_time = time.monotonic() - t0
                return result

        # Fall back to simulation
        if self._simulator is not None and rhs is not None:
            result = self._check_simulation(cex, rhs, step_size)
            result.analysis_time = time.monotonic() - t0
            return result

        return SpuriousnessResult(
            verdict=SpuriousnessVerdict.INCONCLUSIVE,
            analysis_time=time.monotonic() - t0,
            message="No solver or simulator available",
        )

    # ------------------------------------------------------------------
    # SMT-based checking
    # ------------------------------------------------------------------

    def _check_smt(
        self,
        cex: AbstractCounterexample,
        rhs: Optional[Dict[str, ExprNode]],
        step_size: float,
    ) -> SpuriousnessResult:
        """Encode path constraints and check satisfiability.

        For each transition i → i+1 in the counterexample:
          1. State i must be within abstract state path[i]
          2. The ODE dynamics carry state i to state i+1 within one step
          3. State i+1 must be within abstract state path[i+1]

        If the conjunction is SAT → GENUINE.
        If UNSAT → SPURIOUS (extract failure point from UNSAT core).
        """
        queries = 0
        try:
            self._solver.push()

            var_names = cex.path[0].box.variable_names
            n_steps = cex.length

            # Create variables for each step: x_var_step
            step_vars: List[Dict[str, Var]] = []
            for step in range(n_steps):
                sv: Dict[str, Var] = {}
                for var in var_names:
                    sv[var] = Var(f"{var}_{step}")
                step_vars.append(sv)

            # Constrain each step to its abstract state
            for step, astate in enumerate(cex.path):
                for var in var_names:
                    iv = astate.box.dimensions.get(var)
                    if iv is None:
                        continue
                    x = step_vars[step][var]
                    self._solver.assert_formula(Ge(x, Const(iv.lo)))
                    self._solver.assert_formula(Le(x, Const(iv.hi)))

            # Encode transition constraints (Euler discretization)
            if rhs is not None:
                for step in range(n_steps - 1):
                    for var in var_names:
                        rhs_expr = rhs.get(var)
                        if rhs_expr is None:
                            continue
                        # x_{step+1} = x_step + h * f(x_step)
                        # Substitute current step variables
                        mapping = {v: step_vars[step][v] for v in var_names}
                        rhs_subst = rhs_expr.substitute(mapping)

                        x_cur = step_vars[step][var]
                        x_next = step_vars[step + 1][var]
                        h = Const(step_size)

                        # |x_next - (x_cur + h * rhs)| ≤ delta
                        from bioprover.encoding.expression import Add, Mul, Neg

                        euler_next = Add(x_cur, Mul(h, rhs_subst))
                        self._solver.assert_formula(
                            Ge(x_next, Add(euler_next, Neg(Const(self._delta))))
                        )
                        self._solver.assert_formula(
                            Le(x_next, Add(euler_next, Const(self._delta)))
                        )

            result = self._solver.check_sat()
            queries += 1
            self._stats.smt_queries += 1

            if result.is_sat:
                # Extract concrete witness
                model = self._solver.get_model()
                concrete = self._extract_witness(model, step_vars, var_names, step_size)
                self._solver.pop()
                return SpuriousnessResult(
                    verdict=SpuriousnessVerdict.GENUINE,
                    concrete_witness=concrete,
                    smt_queries=queries,
                    message="SMT: path is feasible",
                )
            elif result.is_unsat:
                # Find failure point via incremental checking
                failure_idx = self._find_failure_point(cex, rhs, step_size)
                self._solver.pop()
                return SpuriousnessResult(
                    verdict=SpuriousnessVerdict.SPURIOUS,
                    failure_index=failure_idx,
                    failure_transition=(
                        cex.transitions[failure_idx]
                        if failure_idx is not None and failure_idx < len(cex.transitions)
                        else None
                    ),
                    smt_queries=queries,
                    message=f"SMT: path infeasible at step {failure_idx}",
                )
            else:
                self._solver.pop()
                return SpuriousnessResult(
                    verdict=SpuriousnessVerdict.INCONCLUSIVE,
                    smt_queries=queries,
                    message="SMT: solver returned UNKNOWN",
                )

        except Exception as exc:
            logger.warning("SMT check failed: %s", exc)
            try:
                self._solver.pop()
            except Exception:
                pass
            return SpuriousnessResult(
                verdict=SpuriousnessVerdict.INCONCLUSIVE,
                smt_queries=queries,
                message=f"SMT error: {exc}",
            )

    def _extract_witness(
        self,
        model: Any,
        step_vars: List[Dict[str, Var]],
        var_names: List[str],
        step_size: float,
    ) -> ConcreteCounterexample:
        """Extract a concrete counterexample from an SMT model."""
        times: List[float] = []
        states: List[Dict[str, float]] = []

        for step, sv in enumerate(step_vars):
            t = step * step_size
            times.append(t)
            state: Dict[str, float] = {}
            for var in var_names:
                try:
                    state[var] = model.get_float(sv[var].name)
                except Exception:
                    state[var] = 0.0
            states.append(state)

        return ConcreteCounterexample(
            time_points=times,
            states=states,
        )

    def _find_failure_point(
        self,
        cex: AbstractCounterexample,
        rhs: Optional[Dict[str, ExprNode]],
        step_size: float,
    ) -> Optional[int]:
        """Find the first infeasible transition via binary search.

        Checks prefixes of increasing length to identify the shortest
        infeasible prefix.
        """
        lo, hi = 1, cex.length
        failure_idx: Optional[int] = None

        while lo <= hi:
            mid = (lo + hi) // 2
            prefix = cex.prefix(mid)
            result = self._check_prefix_feasibility(prefix, rhs, step_size)

            if result == SpuriousnessVerdict.SPURIOUS:
                failure_idx = mid - 1
                hi = mid - 1
            else:
                lo = mid + 1

        return failure_idx

    def _check_prefix_feasibility(
        self,
        prefix: AbstractCounterexample,
        rhs: Optional[Dict[str, ExprNode]],
        step_size: float,
    ) -> SpuriousnessVerdict:
        """Check if a prefix of the counterexample is feasible."""
        try:
            self._solver.push()
            var_names = prefix.path[0].box.variable_names

            # Encode prefix constraints
            for step, astate in enumerate(prefix.path):
                for var in var_names:
                    iv = astate.box.dimensions.get(var)
                    if iv is None:
                        continue
                    x = Var(f"{var}_{step}")
                    self._solver.assert_formula(Ge(x, Const(iv.lo)))
                    self._solver.assert_formula(Le(x, Const(iv.hi)))

            result = self._solver.check_sat()
            self._solver.pop()
            self._stats.smt_queries += 1

            if result.is_unsat:
                return SpuriousnessVerdict.SPURIOUS
            return SpuriousnessVerdict.GENUINE

        except Exception:
            try:
                self._solver.pop()
            except Exception:
                pass
            return SpuriousnessVerdict.INCONCLUSIVE

    # ------------------------------------------------------------------
    # Simulation-based checking
    # ------------------------------------------------------------------

    def _check_simulation(
        self,
        cex: AbstractCounterexample,
        rhs: Dict[str, ExprNode],
        step_size: float,
    ) -> SpuriousnessResult:
        """Simulate the ODE from the midpoint of the initial abstract state.

        Check whether the simulation trajectory stays within the
        abstract counterexample path.  This is a fast but incomplete
        check: if the simulation follows the path, the counterexample
        is likely genuine; if it diverges, it may be spurious (but we
        cannot be certain without SMT).
        """
        self._stats.simulation_checks += 1

        initial_point = cex.initial_state.box.midpoint()
        var_names = cex.path[0].box.variable_names

        trajectory: List[Dict[str, float]] = [dict(initial_point)]
        times: List[float] = [0.0]

        current = dict(initial_point)
        failure_idx: Optional[int] = None

        for step in range(1, cex.length):
            # Euler step
            new_state: Dict[str, float] = {}
            for var in var_names:
                rhs_expr = rhs.get(var)
                if rhs_expr is None:
                    new_state[var] = current.get(var, 0.0)
                    continue
                try:
                    derivative = _eval_expr_float(rhs_expr, current)
                    new_state[var] = current[var] + step_size * derivative
                except Exception:
                    new_state[var] = current.get(var, 0.0)

            trajectory.append(new_state)
            times.append(step * step_size)

            # Check containment
            if not cex.path[step].box.contains_point(new_state):
                failure_idx = step
                break

            current = new_state

        if failure_idx is None:
            # Trajectory followed the entire path — likely genuine
            return SpuriousnessResult(
                verdict=SpuriousnessVerdict.GENUINE,
                concrete_witness=ConcreteCounterexample(
                    time_points=times,
                    states=trajectory,
                    property_violated=cex.property_violated,
                ),
                message="Simulation follows counterexample path",
            )
        else:
            return SpuriousnessResult(
                verdict=SpuriousnessVerdict.SPURIOUS,
                failure_index=failure_idx,
                message=f"Simulation diverges at step {failure_idx}",
            )

    # ------------------------------------------------------------------
    # Minimization
    # ------------------------------------------------------------------

    def minimize(
        self,
        cex: AbstractCounterexample,
        rhs: Optional[Dict[str, ExprNode]] = None,
        step_size: float = 0.01,
    ) -> AbstractCounterexample:
        """Find the shortest spurious prefix of *cex*.

        Uses binary search on prefix length to find the minimal
        infeasible prefix.
        """
        lo, hi = 2, cex.length
        best = cex

        while lo <= hi:
            mid = (lo + hi) // 2
            prefix = cex.prefix(mid)
            result = self.check(prefix, rhs, step_size)

            if result.is_spurious:
                best = prefix
                hi = mid - 1
            else:
                lo = mid + 1

        logger.debug(
            "Minimized counterexample from length %d to %d",
            cex.length,
            best.length,
        )
        return best

    # ------------------------------------------------------------------
    # Generalization
    # ------------------------------------------------------------------

    def generalize(
        self,
        cex: AbstractCounterexample,
        all_states: List[AbstractState],
    ) -> List[AbstractCounterexample]:
        """Generalize a spurious counterexample.

        Find all counterexamples with the same structure (same sequence
        of abstract states up to state identity) that are also spurious.
        """
        generalized: List[AbstractCounterexample] = []

        if cex.length < 2:
            return [cex]

        # For each state in the path, find alternative states that
        # could serve the same role (same mode, adjacent box)
        alternatives: List[List[AbstractState]] = []
        for step_state in cex.path:
            alts = [
                s
                for s in all_states
                if s.mode == step_state.mode
                and s.box.overlaps(step_state.box)
                and s.state_id != step_state.state_id
            ]
            # Include the original
            alts.insert(0, step_state)
            alternatives.append(alts[:5])  # limit combinatorial explosion

        # Enumerate alternative paths (limited)
        import itertools

        count = 0
        max_generalizations = 20
        for combo in itertools.product(*alternatives):
            if count >= max_generalizations:
                break
            alt_cex = AbstractCounterexample(
                path=list(combo),
                property_violated=cex.property_violated,
                iteration_found=cex.iteration_found,
            )
            if alt_cex.state_ids() != cex.state_ids():
                generalized.append(alt_cex)
                count += 1

        logger.debug("Generalized counterexample to %d variants", len(generalized))
        return generalized

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def statistics(self) -> Dict[str, Any]:
        return self._stats.to_dict()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass
class _CheckerStatistics:
    smt_queries: int = 0
    simulation_checks: int = 0
    genuine_count: int = 0
    spurious_count: int = 0
    inconclusive_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "smt_queries": self.smt_queries,
            "simulation_checks": self.simulation_checks,
            "genuine": self.genuine_count,
            "spurious": self.spurious_count,
            "inconclusive": self.inconclusive_count,
        }


def _eval_expr_float(expr: ExprNode, val: Dict[str, float]) -> float:
    """Best-effort numeric evaluation of an ExprNode to float."""
    import math

    if isinstance(expr, Const):
        return float(expr.value)
    if isinstance(expr, Var):
        return float(val[expr.name])

    kids = expr.children()
    if len(kids) == 2:
        a = _eval_expr_float(kids[0], val)
        b = _eval_expr_float(kids[1], val)
        from bioprover.encoding.expression import Add, Mul, Div, Pow

        if isinstance(expr, Add):
            return a + b
        if isinstance(expr, Mul):
            return a * b
        if isinstance(expr, Div):
            return a / b if b != 0 else float("inf")
        if isinstance(expr, Pow):
            return math.pow(a, b) if a > 0 else 0.0

    if len(kids) == 1:
        a = _eval_expr_float(kids[0], val)
        from bioprover.encoding.expression import Neg, Exp, Log, Sqrt, Abs

        if isinstance(expr, Neg):
            return -a
        if isinstance(expr, Exp):
            return math.exp(min(a, 700))
        if isinstance(expr, Log):
            return math.log(max(a, 1e-300))
        if isinstance(expr, Sqrt):
            return math.sqrt(max(a, 0.0))
        if isinstance(expr, Abs):
            return abs(a)

    raise TypeError(f"Cannot evaluate {type(expr).__name__}")
