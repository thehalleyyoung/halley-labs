"""dReal-like delta-decidability interface for BioProver.

Provides two operation modes:

1. **External process** – generates SMT-LIB2 files, invokes the ``dreal``
   binary, and parses its output.
2. **Built-in ICP solver** – a contractor-based interval constraint
   propagation engine for use when the dReal binary is not available.
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from bioprover.smt.solver_base import (
    AbstractSMTSolver,
    Model,
    SMTResult,
    SolverStatistics,
)
from bioprover.soundness import SoundnessAnnotation, SoundnessLevel, ErrorBudget

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DeltaPropagator — tracks how delta compounds through CEGAR iterations
# ---------------------------------------------------------------------------

@dataclass
class DeltaPropagator:
    """Tracks how delta-satisfiability error compounds across CEGAR iterations.

    When CEGAR uses N iterations with delta-sat results, the combined
    error is at most N*delta (additive) or (1+delta)^N - 1 (multiplicative).
    """

    base_delta: float = 1e-3
    iteration_count: int = 0
    mode: str = "additive"  # "additive" or "multiplicative"

    @property
    def cumulative_delta(self) -> float:
        if self.iteration_count == 0:
            return self.base_delta
        if self.mode == "multiplicative":
            return (1.0 + self.base_delta) ** self.iteration_count - 1.0
        return self.base_delta * self.iteration_count

    def record_iteration(self) -> float:
        """Record a CEGAR iteration and return the new cumulative delta."""
        self.iteration_count += 1
        return self.cumulative_delta

    def reset(self) -> None:
        self.iteration_count = 0

    def to_error_budget(self) -> ErrorBudget:
        """Convert current cumulative delta into an ErrorBudget."""
        return ErrorBudget(delta=self.cumulative_delta)


# ---------------------------------------------------------------------------
# DeltaSatResult — wraps SMTResult with delta info
# ---------------------------------------------------------------------------

@dataclass
class DeltaSatResult:
    """Result of a delta-satisfiability check with soundness tracking.

    Wraps SMTResult with explicit delta parameter and soundness annotation.
    When result is DELTA_SAT, the formula is satisfiable up to perturbation
    delta; when UNSAT, it is exactly unsatisfiable (no delta weakening).
    """
    result: SMTResult
    delta: float = 1e-3
    model: Optional[Dict[str, Any]] = None
    soundness: Optional[SoundnessAnnotation] = None

    @property
    def is_delta_sat(self) -> bool:
        return self.result == SMTResult.DELTA_SAT

    @property
    def is_exact_unsat(self) -> bool:
        return self.result == SMTResult.UNSAT

# ---------------------------------------------------------------------------
# Interval arithmetic
# ---------------------------------------------------------------------------

@dataclass
class Interval:
    """A closed real interval ``[lo, hi]``."""

    lo: float
    hi: float

    @staticmethod
    def entire() -> Interval:
        return Interval(-math.inf, math.inf)

    @staticmethod
    def point(v: float) -> Interval:
        return Interval(v, v)

    @property
    def width(self) -> float:
        return self.hi - self.lo

    @property
    def midpoint(self) -> float:
        if math.isinf(self.lo) or math.isinf(self.hi):
            if math.isinf(self.lo) and math.isinf(self.hi):
                return 0.0
            if math.isinf(self.lo):
                return self.hi - 1.0
            return self.lo + 1.0
        return (self.lo + self.hi) / 2.0

    @property
    def is_empty(self) -> bool:
        return self.lo > self.hi

    def contains(self, v: float) -> bool:
        return self.lo <= v <= self.hi

    def intersect(self, other: Interval) -> Interval:
        return Interval(max(self.lo, other.lo), min(self.hi, other.hi))

    def union_hull(self, other: Interval) -> Interval:
        return Interval(min(self.lo, other.lo), max(self.hi, other.hi))

    def __add__(self, other: Interval) -> Interval:
        return Interval(self.lo + other.lo, self.hi + other.hi)

    def __sub__(self, other: Interval) -> Interval:
        return Interval(self.lo - other.hi, self.hi - other.lo)

    def __mul__(self, other: Interval) -> Interval:
        products = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ]
        return Interval(min(products), max(products))

    def __truediv__(self, other: Interval) -> Interval:
        if other.contains(0.0):
            return Interval.entire()
        return self * Interval(1.0 / other.hi, 1.0 / other.lo)

    def __neg__(self) -> Interval:
        return Interval(-self.hi, -self.lo)

    def __repr__(self) -> str:
        return f"[{self.lo:.6g}, {self.hi:.6g}]"


# ---------------------------------------------------------------------------
# ICP Constraint representation
# ---------------------------------------------------------------------------

@dataclass
class ICPConstraint:
    """A constraint for interval constraint propagation.

    ``kind`` is one of ``"eq"``, ``"le"``, ``"lt"``, ``"ge"``, ``"gt"``.
    ``lhs`` and ``rhs`` are either variable names or expression dicts.
    """

    kind: str
    lhs: Any
    rhs: Any
    original: Any = None


# ---------------------------------------------------------------------------
# Built-in ICP solver (contractor-based)
# ---------------------------------------------------------------------------

class ICPSolver:
    """Interval Constraint Propagation solver.

    Implements forward–backward contraction, hull consistency, box
    consistency, and adaptive precision for use as a fallback when the
    dReal binary is not installed.
    """

    def __init__(
        self,
        delta: float = 1e-3,
        max_iterations: int = 500,
        max_bisections: int = 200,
    ) -> None:
        self.delta = delta
        self.max_iterations = max_iterations
        self.max_bisections = max_bisections
        self._boxes: List[Dict[str, Interval]] = []
        self._constraints: List[ICPConstraint] = []
        self._var_bounds: Dict[str, Interval] = {}

    # -- variable / constraint setup ----------------------------------------

    def declare_variable(
        self,
        name: str,
        lo: float = -1e6,
        hi: float = 1e6,
    ) -> None:
        self._var_bounds[name] = Interval(lo, hi)

    def add_constraint(self, constraint: ICPConstraint) -> None:
        self._constraints.append(constraint)

    # -- evaluation ---------------------------------------------------------

    def _eval_expr(self, expr: Any, box: Dict[str, Interval]) -> Interval:
        """Evaluate an expression over an interval box."""
        if isinstance(expr, (int, float)):
            return Interval.point(float(expr))
        if isinstance(expr, str):
            return box.get(expr, Interval.entire())
        if isinstance(expr, dict):
            op = expr.get("op", "")
            args = expr.get("args", [])
            if op == "+":
                result = self._eval_expr(args[0], box)
                for a in args[1:]:
                    result = result + self._eval_expr(a, box)
                return result
            if op == "-":
                if len(args) == 1:
                    return -self._eval_expr(args[0], box)
                return self._eval_expr(args[0], box) - self._eval_expr(args[1], box)
            if op == "*":
                result = self._eval_expr(args[0], box)
                for a in args[1:]:
                    result = result * self._eval_expr(a, box)
                return result
            if op == "/":
                return self._eval_expr(args[0], box) / self._eval_expr(args[1], box)
            if op in ("^", "pow"):
                base = self._eval_expr(args[0], box)
                exp_iv = self._eval_expr(args[1], box)
                if exp_iv.lo == exp_iv.hi:
                    n = int(exp_iv.lo)
                    if n == 2:
                        if base.lo >= 0:
                            return Interval(base.lo ** 2, base.hi ** 2)
                        if base.hi <= 0:
                            return Interval(base.hi ** 2, base.lo ** 2)
                        return Interval(0.0, max(base.lo ** 2, base.hi ** 2))
                vals = []
                for b in (base.lo, base.hi):
                    for e in (exp_iv.lo, exp_iv.hi):
                        try:
                            vals.append(b ** e)
                        except (ValueError, OverflowError):
                            pass
                if vals:
                    return Interval(min(vals), max(vals))
                return Interval.entire()
            if op == "sin":
                return Interval(-1.0, 1.0)
            if op == "cos":
                return Interval(-1.0, 1.0)
            if op == "exp":
                iv = self._eval_expr(args[0], box)
                lo = math.exp(max(iv.lo, -700))
                hi = math.exp(min(iv.hi, 700))
                return Interval(lo, hi)
            if op == "log":
                iv = self._eval_expr(args[0], box)
                lo = math.log(max(iv.lo, 1e-300))
                hi = math.log(max(iv.hi, 1e-300))
                return Interval(lo, hi)
        return Interval.entire()

    # -- contractors --------------------------------------------------------

    def _forward_backward_contract(
        self,
        constraint: ICPConstraint,
        box: Dict[str, Interval],
    ) -> Dict[str, Interval]:
        """Apply forward-backward contraction for a single constraint."""
        lhs_iv = self._eval_expr(constraint.lhs, box)
        rhs_iv = self._eval_expr(constraint.rhs, box)

        if constraint.kind == "eq":
            target = lhs_iv.intersect(rhs_iv)
            if target.is_empty:
                return {}  # empty box signals infeasibility
        elif constraint.kind in ("le", "lt"):
            if lhs_iv.lo > rhs_iv.hi:
                return {}
        elif constraint.kind in ("ge", "gt"):
            if lhs_iv.hi < rhs_iv.lo:
                return {}

        # Backward pass: narrow variable bounds via the constraint.
        new_box = dict(box)
        self._backward_narrow(constraint.lhs, rhs_iv, constraint.kind, new_box)
        return new_box

    def _backward_narrow(
        self,
        expr: Any,
        target: Interval,
        kind: str,
        box: Dict[str, Interval],
    ) -> None:
        """Narrow variable intervals given a target interval."""
        if isinstance(expr, str) and expr in box:
            if kind == "eq":
                box[expr] = box[expr].intersect(target)
            elif kind in ("le", "lt"):
                box[expr] = box[expr].intersect(Interval(-math.inf, target.hi))
            elif kind in ("ge", "gt"):
                box[expr] = box[expr].intersect(Interval(target.lo, math.inf))

    def _hull_consistency(self, box: Dict[str, Interval]) -> Dict[str, Interval]:
        """Apply hull consistency: iterate contraction to a fixed point."""
        for _ in range(self.max_iterations):
            changed = False
            for c in self._constraints:
                new_box = self._forward_backward_contract(c, box)
                if not new_box:
                    return {}
                for v in box:
                    old = box[v]
                    upd = new_box.get(v, old)
                    if upd.lo > old.lo + 1e-15 or upd.hi < old.hi - 1e-15:
                        changed = True
                    box[v] = upd
            if not changed:
                break
        return box

    def _box_consistency(
        self,
        var: str,
        box: Dict[str, Interval],
    ) -> Interval:
        """Refine the bound of *var* via box consistency.

        Splits the variable's interval and checks feasibility of each
        sub-interval, trimming infeasible portions.
        """
        iv = box[var]
        if iv.width < self.delta:
            return iv

        mid = iv.midpoint
        left_box = dict(box)
        left_box[var] = Interval(iv.lo, mid)
        left_ok = self._hull_consistency(dict(left_box))

        right_box = dict(box)
        right_box[var] = Interval(mid, iv.hi)
        right_ok = self._hull_consistency(dict(right_box))

        if not left_ok and not right_ok:
            return Interval(1.0, -1.0)  # empty
        if not left_ok:
            return right_ok.get(var, Interval(mid, iv.hi))
        if not right_ok:
            return left_ok.get(var, Interval(iv.lo, mid))
        left_iv = left_ok.get(var, Interval(iv.lo, mid))
        right_iv = right_ok.get(var, Interval(mid, iv.hi))
        return left_iv.union_hull(right_iv)

    # -- main solve loop ----------------------------------------------------

    def solve(self) -> Tuple[SMTResult, Optional[Dict[str, Interval]]]:
        """Run ICP and return ``(result, box_or_None)``."""
        box = dict(self._var_bounds)
        if not box:
            return (SMTResult.SAT, {})

        box = self._hull_consistency(box)
        if not box:
            return (SMTResult.UNSAT, None)

        # Check if all intervals are narrow enough.
        if self._all_narrow(box):
            return (SMTResult.DELTA_SAT, box)

        # Bisection loop.
        stack: List[Dict[str, Interval]] = [box]
        for _ in range(self.max_bisections):
            if not stack:
                return (SMTResult.UNSAT, None)
            current = stack.pop()

            current = self._hull_consistency(current)
            if not current:
                continue

            if self._all_narrow(current):
                return (SMTResult.DELTA_SAT, current)

            # Bisect largest variable.
            var = self._pick_bisection_var(current)
            if var is None:
                return (SMTResult.DELTA_SAT, current)

            iv = current[var]
            mid = iv.midpoint
            left = dict(current)
            left[var] = Interval(iv.lo, mid)
            right = dict(current)
            right[var] = Interval(mid, iv.hi)
            stack.append(right)
            stack.append(left)

        # Exhausted bisection budget – report unknown.
        if stack:
            return (SMTResult.UNKNOWN, stack[-1])
        return (SMTResult.UNSAT, None)

    def _all_narrow(self, box: Dict[str, Interval]) -> bool:
        return all(iv.width <= self.delta for iv in box.values())

    def _pick_bisection_var(self, box: Dict[str, Interval]) -> Optional[str]:
        best_var: Optional[str] = None
        best_width = self.delta
        for var, iv in box.items():
            if iv.width > best_width:
                best_width = iv.width
                best_var = var
        return best_var


# ---------------------------------------------------------------------------
# SMT-LIB generation
# ---------------------------------------------------------------------------

class SMTLIBGenerator:
    """Generate SMT-LIB2 files for the dReal external solver."""

    def __init__(self, delta: float = 1e-3, logic: str = "QF_NRA") -> None:
        self.delta = delta
        self.logic = logic
        self._declarations: List[str] = []
        self._assertions: List[str] = []

    def declare_variable(
        self,
        name: str,
        lo: float = -1e6,
        hi: float = 1e6,
    ) -> None:
        self._declarations.append(f"(declare-fun {name} () Real)")
        self._assertions.append(f"(assert (>= {name} {lo}))")
        self._assertions.append(f"(assert (<= {name} {hi}))")

    def assert_formula(self, smt_expr: str) -> None:
        self._assertions.append(f"(assert {smt_expr})")

    def generate(self) -> str:
        lines = [
            f"(set-logic {self.logic})",
            f"(set-option :precision {self.delta})",
        ]
        lines.extend(self._declarations)
        lines.extend(self._assertions)
        lines.append("(check-sat)")
        lines.append("(exit)")
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        self._declarations.clear()
        self._assertions.clear()


# ---------------------------------------------------------------------------
# dReal output parsing
# ---------------------------------------------------------------------------

_DELTA_SAT_RE = re.compile(r"delta-sat")
_UNSAT_RE = re.compile(r"^unsat", re.MULTILINE)
_MODEL_LINE_RE = re.compile(
    r"\s*(\w+)\s*:\s*\[([^,]+),\s*([^\]]+)\]"
)


def _parse_dreal_output(
    stdout: str,
    stderr: str,
) -> Tuple[SMTResult, Optional[Dict[str, Tuple[float, float]]]]:
    """Parse dReal solver output into a result and optional interval model."""
    combined = stdout + "\n" + stderr

    if _UNSAT_RE.search(combined):
        return SMTResult.UNSAT, None

    if _DELTA_SAT_RE.search(combined):
        model: Dict[str, Tuple[float, float]] = {}
        for m in _MODEL_LINE_RE.finditer(combined):
            name = m.group(1)
            lo = float(m.group(2))
            hi = float(m.group(3))
            model[name] = (lo, hi)
        return SMTResult.DELTA_SAT, model if model else None

    return SMTResult.UNKNOWN, None


# ---------------------------------------------------------------------------
# DRealSolver
# ---------------------------------------------------------------------------

class DRealSolver(AbstractSMTSolver):
    """dReal-backed SMT solver implementing :class:`AbstractSMTSolver`.

    When the ``dreal`` binary is found on ``$PATH`` (or supplied via
    *binary_path*), constraints are serialised to SMT-LIB2 and solved
    externally.  Otherwise, a built-in ICP solver is used.

    Parameters
    ----------
    delta:
        Precision parameter (δ) for delta-decidability.
    binary_path:
        Path to the ``dreal`` binary.  If ``None``, searches ``$PATH``.
    timeout:
        Default timeout in seconds.
    use_icp_fallback:
        If ``True`` (default), use the built-in ICP solver when the
        binary is not available.
    """

    def __init__(
        self,
        delta: float = 1e-3,
        binary_path: Optional[str] = None,
        timeout: float = 60.0,
        use_icp_fallback: bool = True,
    ) -> None:
        super().__init__(name="dreal")
        self.delta = delta
        self._binary = binary_path or shutil.which("dreal")
        self._default_timeout = timeout
        self._use_icp_fallback = use_icp_fallback

        self._generator = SMTLIBGenerator(delta=delta)
        self._icp = ICPSolver(delta=delta)
        self._context_stack: List[Tuple[SMTLIBGenerator, ICPSolver]] = []

        self._last_result: Optional[SMTResult] = None
        self._last_model: Optional[Model] = None
        self._last_proof: Optional[str] = None

        self._variables: Dict[str, Tuple[float, float]] = {}
        self._delta_propagator = DeltaPropagator(base_delta=delta)

        if self._binary is None and not use_icp_fallback:
            raise FileNotFoundError(
                "dReal binary not found and ICP fallback is disabled"
            )

    @property
    def has_binary(self) -> bool:
        return self._binary is not None

    # -- variable declaration -----------------------------------------------

    def declare_variable(
        self,
        name: str,
        lo: float = -1e6,
        hi: float = 1e6,
    ) -> None:
        self._variables[name] = (lo, hi)
        self._generator.declare_variable(name, lo, hi)
        self._icp.declare_variable(name, lo, hi)

    # -- AbstractSMTSolver implementation -----------------------------------

    def assert_formula(self, expr: Any) -> None:
        if isinstance(expr, str):
            self._generator.assert_formula(expr)
        elif isinstance(expr, dict):
            smt_str = _expr_to_smtlib(expr)
            self._generator.assert_formula(smt_str)
            self._icp.add_constraint(_expr_to_icp_constraint(expr))
        else:
            smt_str = str(expr)
            self._generator.assert_formula(smt_str)

    def check_sat(self, timeout: Optional[float] = None) -> SMTResult:
        effective_timeout = timeout or self._default_timeout
        t0 = time.perf_counter()

        if self._binary:
            result, model_intervals, proof = self._run_binary(effective_timeout)
        elif self._use_icp_fallback:
            result, model_intervals, proof = self._run_icp()
        else:
            result = SMTResult.UNKNOWN
            model_intervals = None
            proof = None

        elapsed = time.perf_counter() - t0
        self._last_result = result
        self._last_proof = proof
        self.stats.record_query(result, elapsed)

        if model_intervals is not None:
            assignments = {k: v for k, v in model_intervals.items()}
            self._last_model = Model(
                assignments=assignments,
                solver_name=self.name,
                delta=self.delta,
            )
        else:
            self._last_model = None

        return result

    def get_model(self) -> Model:
        if self._last_model is None:
            raise RuntimeError("No model available")
        return self._last_model

    def push(self) -> None:
        import copy
        self._context_stack.append((
            copy.deepcopy(self._generator),
            copy.deepcopy(self._icp),
        ))
        self.stats.record_push()

    def pop(self) -> None:
        if not self._context_stack:
            raise RuntimeError("Cannot pop: context stack is empty")
        self._generator, self._icp = self._context_stack.pop()
        self.stats.record_pop()

    def check_sat_assuming(
        self,
        assumptions: Sequence[Any],
        timeout: Optional[float] = None,
    ) -> SMTResult:
        self.push()
        try:
            for a in assumptions:
                self.assert_formula(a)
            return self.check_sat(timeout)
        finally:
            self.pop()

    def get_unsat_core(self) -> List[Any]:
        # dReal does not natively support UNSAT cores; return empty.
        return []

    def reset(self) -> None:
        self._generator = SMTLIBGenerator(delta=self.delta)
        self._icp = ICPSolver(delta=self.delta)
        self._context_stack.clear()
        self._variables.clear()
        self._last_result = None
        self._last_model = None
        self._last_proof = None
        self._delta_propagator.reset()

    # -- delta management ---------------------------------------------------

    def set_delta(self, delta: float) -> None:
        """Update the precision parameter."""
        if delta <= 0:
            raise ValueError("Delta must be positive")
        self.delta = delta
        self._generator.delta = delta
        self._icp.delta = delta
        self._delta_propagator.base_delta = delta

    def check_delta_sat(self, timeout: Optional[float] = None) -> DeltaSatResult:
        """Check satisfiability with explicit delta-soundness tracking.

        Returns a DeltaSatResult that wraps the SMTResult with the delta
        parameter and a SoundnessAnnotation indicating the soundness level.
        Uses DeltaPropagator to track cumulative delta across iterations.
        """
        result = self.check_sat(timeout)
        soundness = SoundnessAnnotation(level=SoundnessLevel.SOUND)
        if result == SMTResult.DELTA_SAT:
            cumulative = self._delta_propagator.record_iteration()
            error_budget = self._delta_propagator.to_error_budget()
            soundness = soundness.with_delta(cumulative)
            soundness = soundness.with_error_budget(error_budget)
        model_dict = None
        if self._last_model is not None:
            model_dict = self._last_model.assignments
        return DeltaSatResult(
            result=result,
            delta=self._delta_propagator.cumulative_delta,
            model=model_dict,
            soundness=soundness,
        )

    # -- proof access -------------------------------------------------------

    @property
    def last_proof(self) -> Optional[str]:
        return self._last_proof

    # -- binary invocation --------------------------------------------------

    def _run_binary(
        self,
        timeout: float,
    ) -> Tuple[SMTResult, Optional[Dict[str, Tuple[float, float]]], Optional[str]]:
        smt_content = self._generator.generate()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".smt2", delete=False
        ) as f:
            f.write(smt_content)
            smt_path = f.name

        proof_path = smt_path + ".proof"
        cmd = [
            self._binary,
            "--precision", str(self.delta),
            "--produce-models",
            "--proof-file", proof_path,
            smt_path,
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            result, model_intervals = _parse_dreal_output(proc.stdout, proc.stderr)
            proof = None
            if os.path.exists(proof_path):
                with open(proof_path) as pf:
                    proof = pf.read()
            return result, model_intervals, proof

        except subprocess.TimeoutExpired:
            logger.warning("dReal timed out after %.1fs", timeout)
            return SMTResult.TIMEOUT, None, None
        except FileNotFoundError:
            logger.error("dReal binary not found at %s", self._binary)
            if self._use_icp_fallback:
                return self._run_icp()
            return SMTResult.UNKNOWN, None, None
        finally:
            for p in (smt_path, proof_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def _run_icp(
        self,
    ) -> Tuple[SMTResult, Optional[Dict[str, Tuple[float, float]]], Optional[str]]:
        result, box = self._icp.solve()
        if box is not None:
            model_intervals = {k: (v.lo, v.hi) for k, v in box.items()}
        else:
            model_intervals = None
        return result, model_intervals, None


# ---------------------------------------------------------------------------
# Expression → SMT-LIB string
# ---------------------------------------------------------------------------

def _expr_to_smtlib(expr: Any) -> str:
    """Convert a BioProver expression dict to an SMT-LIB2 string."""
    if isinstance(expr, (int, float)):
        v = float(expr)
        if v < 0:
            return f"(- {-v})"
        return str(v)
    if isinstance(expr, str):
        return expr
    if isinstance(expr, dict):
        op = expr.get("op", "")
        args = expr.get("args", [])
        smt_args = " ".join(_expr_to_smtlib(a) for a in args)
        op_map = {
            "+": "+", "-": "-", "*": "*", "/": "/",
            "^": "^", "pow": "^",
            "==": "=", "=": "=", "!=": "distinct",
            "<": "<", "<=": "<=", ">": ">", ">=": ">=",
            "and": "and", "or": "or", "not": "not",
            "=>": "=>",
            "sin": "sin", "cos": "cos", "exp": "exp", "log": "log",
        }
        smt_op = op_map.get(op, op)
        return f"({smt_op} {smt_args})"
    return str(expr)


def _expr_to_icp_constraint(expr: Any) -> ICPConstraint:
    """Convert an expression dict to an :class:`ICPConstraint`."""
    if isinstance(expr, dict):
        op = expr.get("op", "")
        args = expr.get("args", [])
        kind_map = {"==": "eq", "=": "eq", "<=": "le", "<": "lt", ">=": "ge", ">": "gt"}
        if op in kind_map and len(args) == 2:
            return ICPConstraint(kind=kind_map[op], lhs=args[0], rhs=args[1], original=expr)
    return ICPConstraint(kind="eq", lhs=expr, rhs=0, original=expr)
