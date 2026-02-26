"""Z3 solver interface for BioProver.

Wraps the Z3 Python bindings to implement :class:`AbstractSMTSolver`,
providing expression translation, incremental solving, optimisation,
tactic configuration, parallel solving, quantifier handling, array
theory encoding, and statistics extraction.
"""

from __future__ import annotations

import logging
import time
from enum import Enum, auto
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import z3

from bioprover.smt.solver_base import (
    AbstractSMTSolver,
    CounterexampleTrace,
    Model,
    SMTResult,
    SolverStatistics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tactic presets
# ---------------------------------------------------------------------------

class TacticPreset(Enum):
    """Named tactic configurations for Z3."""

    DEFAULT = auto()
    NLSAT = auto()
    QFNRA = auto()
    QF_LRA = auto()
    CUSTOM = auto()


_TACTIC_STRINGS: Dict[TacticPreset, str] = {
    TacticPreset.DEFAULT: "smt",
    TacticPreset.NLSAT: "nlsat",
    TacticPreset.QFNRA: "(then simplify solve-eqs nlsat)",
    TacticPreset.QF_LRA: "(then simplify solve-eqs smt)",
}


# ---------------------------------------------------------------------------
# Expression translation helpers
# ---------------------------------------------------------------------------

class ExprTranslator:
    """Translate BioProver / SymPy-style expression nodes into Z3 expressions.

    The translator maintains a cache keyed on expression identity to
    avoid redundant traversal.
    """

    def __init__(self) -> None:
        self._cache: Dict[int, z3.ExprRef] = {}
        self._z3_vars: Dict[str, z3.ExprRef] = {}

    # -- variable management ------------------------------------------------

    def real_var(self, name: str) -> z3.ArithRef:
        if name not in self._z3_vars:
            self._z3_vars[name] = z3.Real(name)
        return self._z3_vars[name]  # type: ignore[return-value]

    def int_var(self, name: str) -> z3.ArithRef:
        if name not in self._z3_vars:
            self._z3_vars[name] = z3.Int(name)
        return self._z3_vars[name]  # type: ignore[return-value]

    def bool_var(self, name: str) -> z3.BoolRef:
        if name not in self._z3_vars:
            self._z3_vars[name] = z3.Bool(name)
        return self._z3_vars[name]  # type: ignore[return-value]

    def get_var(self, name: str) -> Optional[z3.ExprRef]:
        return self._z3_vars.get(name)

    @property
    def variables(self) -> Dict[str, z3.ExprRef]:
        return dict(self._z3_vars)

    # -- recursive translation ---------------------------------------------

    def translate(self, expr: Any) -> z3.ExprRef:
        """Translate an expression node into a Z3 expression.

        Supported node types:

        * ``dict`` with ``"op"`` key – binary/unary operators.
        * ``str`` – variable reference.
        * ``int`` / ``float`` – numeric literal.
        * ``z3.ExprRef`` – passed through unchanged.
        """
        eid = id(expr)
        if eid in self._cache:
            return self._cache[eid]

        result = self._translate_impl(expr)
        self._cache[eid] = result
        return result

    def _translate_impl(self, expr: Any) -> z3.ExprRef:
        # Pass-through for native Z3 expressions.
        if isinstance(expr, z3.ExprRef):
            return expr

        # Numeric literal.
        if isinstance(expr, (int, float)):
            return z3.RealVal(expr)

        # Boolean literal.
        if isinstance(expr, bool):
            return z3.BoolVal(expr)

        # Variable reference.
        if isinstance(expr, str):
            return self.real_var(expr)

        # Operator node encoded as dict.
        if isinstance(expr, dict):
            return self._translate_op(expr)

        # Sequence – conjunction.
        if isinstance(expr, (list, tuple)):
            return z3.And([self.translate(e) for e in expr])

        raise TypeError(f"Cannot translate expression of type {type(expr).__name__}")

    def _translate_op(self, node: dict) -> z3.ExprRef:
        op = node.get("op", "")
        args = node.get("args", [])
        translated = [self.translate(a) for a in args]

        # Arithmetic
        if op == "+":
            return sum(translated[1:], translated[0])  # type: ignore[arg-type]
        if op == "-":
            if len(translated) == 1:
                return -translated[0]
            return translated[0] - translated[1]
        if op == "*":
            result = translated[0]
            for t in translated[1:]:
                result = result * t
            return result
        if op == "/":
            return translated[0] / translated[1]
        if op == "^" or op == "pow":
            base, exp = translated
            # Z3 doesn't natively support real exponentiation for arbitrary
            # exponents; integer powers can be expanded.
            if z3.is_int_value(exp):
                n = exp.as_long()
                if n >= 0:
                    result = z3.RealVal(1)
                    for _ in range(n):
                        result = result * base
                    return result
            return base ** exp

        # Comparison
        if op == "==" or op == "=":
            return translated[0] == translated[1]
        if op == "!=":
            return translated[0] != translated[1]
        if op == "<":
            return translated[0] < translated[1]
        if op == "<=":
            return translated[0] <= translated[1]
        if op == ">":
            return translated[0] > translated[1]
        if op == ">=":
            return translated[0] >= translated[1]

        # Boolean
        if op == "and":
            return z3.And(translated)
        if op == "or":
            return z3.Or(translated)
        if op == "not":
            return z3.Not(translated[0])
        if op == "=>":
            return z3.Implies(translated[0], translated[1])
        if op == "ite":
            return z3.If(translated[0], translated[1], translated[2])

        # Quantifiers
        if op == "forall":
            body = translated[-1]
            bound = translated[:-1]
            return z3.ForAll(bound, body)
        if op == "exists":
            body = translated[-1]
            bound = translated[:-1]
            return z3.Exists(bound, body)

        raise ValueError(f"Unknown operator: {op!r}")

    def clear_cache(self) -> None:
        self._cache.clear()


# ---------------------------------------------------------------------------
# Z3 model → BioProver model
# ---------------------------------------------------------------------------

def _z3_model_to_model(z3_model: z3.ModelRef, solver_name: str = "z3") -> Model:
    """Convert a Z3 model to a BioProver :class:`Model`."""
    assignments: Dict[str, Any] = {}
    for decl in z3_model.decls():
        name = decl.name()
        val = z3_model[decl]
        assignments[name] = _z3_val_to_python(val)
    return Model(assignments=assignments, solver_name=solver_name)


def _z3_val_to_python(val: z3.ExprRef) -> Any:
    """Convert a Z3 value to a Python primitive."""
    if z3.is_int_value(val):
        return val.as_long()
    if z3.is_rational_value(val):
        return float(val.as_fraction())
    if z3.is_algebraic_value(val):
        return float(val.approx(20).as_fraction())
    if z3.is_true(val):
        return True
    if z3.is_false(val):
        return False
    return str(val)


# ---------------------------------------------------------------------------
# Z3Solver
# ---------------------------------------------------------------------------

class Z3Solver(AbstractSMTSolver):
    """Z3-backed SMT solver implementing :class:`AbstractSMTSolver`.

    Parameters
    ----------
    tactic:
        Tactic preset to use (default: ``TacticPreset.DEFAULT``).
    timeout:
        Default timeout in seconds.
    """

    def __init__(
        self,
        tactic: TacticPreset = TacticPreset.DEFAULT,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(name="z3")
        self._tactic_preset = tactic
        self._default_timeout = timeout
        self._solver = z3.Solver()
        self._translator = ExprTranslator()
        self._last_result: Optional[SMTResult] = None
        self._apply_tactic(tactic)
        self._configure_defaults()

    # -- internal config ----------------------------------------------------

    def _configure_defaults(self) -> None:
        self._solver.set("timeout", int(self._default_timeout * 1000))

    def _apply_tactic(self, preset: TacticPreset) -> None:
        tactic_str = _TACTIC_STRINGS.get(preset)
        if tactic_str and preset != TacticPreset.DEFAULT:
            try:
                tactic = z3.Tactic(tactic_str)
                self._solver = tactic.solver()
            except z3.Z3Exception:
                logger.warning("Tactic %s unavailable, falling back to default", tactic_str)
                self._solver = z3.Solver()
        self._configure_defaults()

    # -- expression access --------------------------------------------------

    @property
    def translator(self) -> ExprTranslator:
        return self._translator

    def real_var(self, name: str) -> z3.ArithRef:
        return self._translator.real_var(name)

    def int_var(self, name: str) -> z3.ArithRef:
        return self._translator.int_var(name)

    def bool_var(self, name: str) -> z3.BoolRef:
        return self._translator.bool_var(name)

    # -- AbstractSMTSolver implementation -----------------------------------

    def assert_formula(self, expr: Any) -> None:
        z3_expr = self._translator.translate(expr)
        self._solver.add(z3_expr)

    def check_sat(self, timeout: Optional[float] = None) -> SMTResult:
        if timeout is not None:
            self._solver.set("timeout", int(timeout * 1000))
        t0 = time.perf_counter()
        try:
            r = self._solver.check()
        except z3.Z3Exception as exc:
            logger.error("Z3 exception: %s", exc)
            r = z3.unknown
        elapsed = time.perf_counter() - t0

        result = self._translate_check_result(r)
        self._last_result = result
        self.stats.record_query(result, elapsed)

        # Restore default timeout.
        if timeout is not None:
            self._solver.set("timeout", int(self._default_timeout * 1000))

        return result

    def get_model(self) -> Model:
        if self._last_result not in (SMTResult.SAT, SMTResult.DELTA_SAT):
            raise RuntimeError("No model available (last result was not SAT)")
        return _z3_model_to_model(self._solver.model(), solver_name=self.name)

    def push(self) -> None:
        self._solver.push()
        self.stats.record_push()

    def pop(self) -> None:
        self._solver.pop()
        self.stats.record_pop()

    def check_sat_assuming(
        self,
        assumptions: Sequence[Any],
        timeout: Optional[float] = None,
    ) -> SMTResult:
        z3_assumptions = [self._translator.translate(a) for a in assumptions]
        if timeout is not None:
            self._solver.set("timeout", int(timeout * 1000))

        t0 = time.perf_counter()
        try:
            r = self._solver.check(*z3_assumptions)
        except z3.Z3Exception as exc:
            logger.error("Z3 exception during check_sat_assuming: %s", exc)
            r = z3.unknown
        elapsed = time.perf_counter() - t0

        result = self._translate_check_result(r)
        self._last_result = result
        self.stats.record_query(result, elapsed)

        if timeout is not None:
            self._solver.set("timeout", int(self._default_timeout * 1000))
        return result

    def get_unsat_core(self) -> List[Any]:
        core = self._solver.unsat_core()
        return [str(c) for c in core]

    def reset(self) -> None:
        self._solver.reset()
        self._translator = ExprTranslator()
        self._last_result = None
        self._configure_defaults()

    # -- optimisation -------------------------------------------------------

    def create_optimizer(self) -> _Z3Optimizer:
        """Return a Z3 Optimize wrapper for robustness maximisation."""
        return _Z3Optimizer(translator=self._translator, timeout=self._default_timeout)

    # -- parallel tactics ---------------------------------------------------

    def parallel_check(
        self,
        expr: Any,
        tactics: Optional[List[TacticPreset]] = None,
        timeout: float = 30.0,
    ) -> SMTResult:
        """Run multiple tactics on the same formula and return the first result.

        Uses Z3's ``par-or`` combinator internally.
        """
        if tactics is None:
            tactics = [TacticPreset.DEFAULT, TacticPreset.NLSAT, TacticPreset.QFNRA]

        tactic_strs = [_TACTIC_STRINGS.get(t, "smt") for t in tactics]
        try:
            par_tactic = z3.ParOr(*[z3.Tactic(ts) for ts in tactic_strs])
        except z3.Z3Exception:
            logger.warning("ParOr tactic unavailable, falling back to sequential")
            return self.check_sat(timeout)

        solver = par_tactic.solver()
        solver.set("timeout", int(timeout * 1000))
        z3_expr = self._translator.translate(expr)
        solver.add(z3_expr)

        t0 = time.perf_counter()
        r = solver.check()
        elapsed = time.perf_counter() - t0

        result = self._translate_check_result(r)
        self.stats.record_query(result, elapsed)
        return result

    # -- quantifiers --------------------------------------------------------

    def forall(self, bound_vars: List[z3.ExprRef], body: Any) -> z3.ExprRef:
        z3_body = self._translator.translate(body)
        return z3.ForAll(bound_vars, z3_body)

    def exists(self, bound_vars: List[z3.ExprRef], body: Any) -> z3.ExprRef:
        z3_body = self._translator.translate(body)
        return z3.Exists(bound_vars, z3_body)

    # -- array theory for trajectories --------------------------------------

    def trajectory_array(
        self,
        name: str,
        index_sort: z3.SortRef | None = None,
        value_sort: z3.SortRef | None = None,
    ) -> z3.ArrayRef:
        """Create an array variable for encoding trajectories.

        Default sorts: ``Int -> Real`` (time step → concentration).
        """
        idx = index_sort or z3.IntSort()
        val = value_sort or z3.RealSort()
        arr = z3.Array(name, idx, val)
        self._translator._z3_vars[name] = arr
        return arr

    def trajectory_constraints(
        self,
        array: z3.ArrayRef,
        values: List[float],
        start_index: int = 0,
    ) -> List[z3.BoolRef]:
        """Generate ``Select(array, i) == v`` constraints for a trajectory."""
        constraints: List[z3.BoolRef] = []
        for i, v in enumerate(values):
            constraints.append(
                z3.Select(array, z3.IntVal(start_index + i)) == z3.RealVal(v)
            )
        return constraints

    # -- theory propagator hook ---------------------------------------------

    def register_propagator(self, callback: Callable[..., Any]) -> None:
        """Register a custom theory propagator callback.

        The callback receives ``(solver, expr, value)`` triples during
        search.  This is a thin wrapper around Z3's user-propagator API
        (available in newer Z3 builds).
        """
        try:
            prop = z3.UserPropagateBase(s=self._solver)
            prop.push = lambda: None
            prop.pop = lambda n: None
            prop.fixed = callback  # type: ignore[attr-defined]
            self._propagator = prop
        except AttributeError:
            logger.warning("Z3 UserPropagateBase not available in this build")

    # -- statistics ---------------------------------------------------------

    def z3_statistics(self) -> Dict[str, Any]:
        """Return raw Z3 statistics as a dictionary."""
        stats = self._solver.statistics()
        return {str(k): stats[k] for k in stats.keys()}

    # -- counterexample extraction ------------------------------------------

    def extract_counterexample(
        self,
        var_names: List[str],
        time_steps: int,
        time_var: str = "t",
    ) -> CounterexampleTrace:
        """Build a :class:`CounterexampleTrace` from the current SAT model.

        Expects the model to contain indexed variables of the form
        ``<name>_<step>``.
        """
        model = self.get_model()
        trace = CounterexampleTrace()
        for step in range(time_steps):
            t_val = model.get_float(f"{time_var}_{step}") if f"{time_var}_{step}" in model else float(step)
            state: Dict[str, Any] = {}
            for var in var_names:
                key = f"{var}_{step}"
                if key in model:
                    state[var] = model.get_float(key)
            trace.add_state(t_val, state)
        trace.source_model = model
        return trace

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _translate_check_result(r: z3.CheckSatResult) -> SMTResult:
        if r == z3.sat:
            return SMTResult.SAT
        if r == z3.unsat:
            return SMTResult.UNSAT
        return SMTResult.UNKNOWN

    def assertions(self) -> List[z3.ExprRef]:
        return list(self._solver.assertions())

    def sexpr(self) -> str:
        return self._solver.sexpr()

    def to_smtlib2(self) -> str:
        return self._solver.to_smt2()


# ---------------------------------------------------------------------------
# Z3 Optimizer wrapper
# ---------------------------------------------------------------------------

class _Z3Optimizer:
    """Thin wrapper around ``z3.Optimize`` for robustness maximisation."""

    def __init__(
        self,
        translator: ExprTranslator,
        timeout: float = 30.0,
    ) -> None:
        self._opt = z3.Optimize()
        self._translator = translator
        self._timeout = timeout
        self._opt.set("timeout", int(timeout * 1000))

    def add(self, expr: Any) -> None:
        self._opt.add(self._translator.translate(expr))

    def maximize(self, expr: Any) -> z3.OptimizeObjective:
        return self._opt.maximize(self._translator.translate(expr))

    def minimize(self, expr: Any) -> z3.OptimizeObjective:
        return self._opt.minimize(self._translator.translate(expr))

    def check(self) -> SMTResult:
        r = self._opt.check()
        if r == z3.sat:
            return SMTResult.SAT
        if r == z3.unsat:
            return SMTResult.UNSAT
        return SMTResult.UNKNOWN

    def model(self) -> Model:
        return _z3_model_to_model(self._opt.model(), solver_name="z3-optimize")

    def objectives(self) -> List[Any]:
        return list(self._opt.objectives())
