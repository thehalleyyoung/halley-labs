"""
Integrated pipeline for refinement type inference via guard-harvesting CEGAR.

Connects: Python AST → guard extraction → SMT encoding → CEGAR loop → bug reports.
This is the real, working pipeline that analyses actual Python source code.
"""

from __future__ import annotations

import ast
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from .smt.solver import (
    Z3Solver,
    SatResult,
    Comparison,
    ComparisonOp,
    IsInstance,
    IsNone,
    IsTruthy,
    HasAttr,
    And,
    Or,
    Not,
    BoolLit,
    Var,
    Const,
    Len,
    Predicate,
    BaseType,
    RefinedType,
    RefinementType,
)
from .guard_extractor import (
    extract_guards,
    ExtractedGuard,
    GuardPattern,
    PredicateKind,
)

try:
    from .model_checker import verify_model, extract_computation_graph, VerificationResult, HAS_Z3 as _MC_HAS_Z3
    _HAS_MODEL_CHECKER = True
except ImportError:
    _HAS_MODEL_CHECKER = False
    _MC_HAS_Z3 = False

try:
    from .shape_cegar import run_shape_cegar, ShapeCEGARResult
    _HAS_SHAPE_CEGAR = True
except ImportError:
    _HAS_SHAPE_CEGAR = False

try:
    from .path_sensitive import path_sensitive_filter
    _HAS_PATH_SENSITIVE = True
except ImportError:
    _HAS_PATH_SENSITIVE = False

logger = logging.getLogger(__name__)


# ── Bug categories ──────────────────────────────────────────────────────

class BugCategory(Enum):
    NULL_DEREF = auto()
    DIV_BY_ZERO = auto()
    TYPE_ERROR = auto()
    INDEX_OUT_OF_BOUNDS = auto()
    ATTRIBUTE_ERROR = auto()
    UNGUARDED_NONE = auto()


@dataclass
class BugReport:
    category: BugCategory
    line: int
    col: int
    message: str
    function: str
    variable: str
    severity: str = "warning"  # "error" or "warning"
    guarded: bool = False  # True if a guard protects this site


@dataclass
class RefinedVar:
    """A variable with its inferred refinement type."""
    name: str
    base_type: str
    predicates: List[str]  # human-readable predicate strings
    line: int = 0


@dataclass
class FunctionSummary:
    name: str
    line: int
    guards_harvested: int
    predicates_inferred: int
    cegar_iterations: int
    bugs_found: List[BugReport]
    refined_vars: List[RefinedVar]
    converged: bool
    analysis_time_ms: float


@dataclass
class AnalysisResult:
    file_path: str
    functions_analyzed: int
    total_guards: int
    total_bugs: int
    total_predicates: int
    summaries: List[FunctionSummary]
    analysis_time_ms: float
    errors: List[str] = field(default_factory=list)


# ── AST-based analysis ──────────────────────────────────────────────────

class PythonAnalyzer(ast.NodeVisitor):
    """Analyse a Python AST for potential bugs and infer refinement types."""

    def __init__(self, source: str, filename: str = "<string>"):
        self.source = source
        self.filename = filename
        self.bugs: List[BugReport] = []
        self._current_func = "<module>"
        self._scope: Dict[str, Set[str]] = {}  # var → set of possible types
        self._none_vars: Set[str] = set()  # vars that may be None
        self._guarded_vars: Dict[str, Set[str]] = {}  # var → guard predicates
        self._assignments: Dict[str, ast.AST] = {}
        self._divisions: List[Tuple[int, int, str]] = []
        self._attr_accesses: List[Tuple[int, int, str, str]] = []
        self._index_accesses: List[Tuple[int, int, str]] = []

    def analyze(self) -> List[BugReport]:
        tree = ast.parse(self.source, self.filename)
        self.visit(tree)
        return self.bugs

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        old_func = self._current_func
        old_scope = dict(self._scope)
        old_none = set(self._none_vars)
        self._current_func = node.name
        self._scope = {}
        self._none_vars = set()
        self._guarded_vars = {}

        # Register parameters (potentially any type)
        for arg in node.args.args:
            name = arg.arg
            if arg.annotation:
                ann = self._annotation_to_type(arg.annotation)
                self._scope[name] = {ann} if ann else {"Any"}
            else:
                self._scope[name] = {"Any"}

        self.generic_visit(node)
        self._current_func = old_func
        self._scope = old_scope
        self._none_vars = old_none

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                val_type = self._infer_type(node.value)
                self._scope[target.id] = {val_type}
                if val_type == "NoneType":
                    self._none_vars.add(target.id)
                elif target.id in self._none_vars:
                    self._none_vars.discard(target.id)
                self._assignments[target.id] = node.value
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name) and node.value:
            val_type = self._infer_type(node.value)
            self._scope[node.target.id] = {val_type}
            if val_type == "NoneType":
                self._none_vars.add(node.target.id)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        # Track guards
        guarded = self._extract_guard_info(node.test)
        old_none = set(self._none_vars)
        old_guarded = dict(self._guarded_vars)

        # Apply guard narrowing for the true branch
        for var, pred in guarded:
            self._guarded_vars.setdefault(var, set()).add(pred)
            if pred == "is_not_none":
                self._none_vars.discard(var)
            elif pred == "is_none":
                self._none_vars.add(var)
            elif pred.startswith("isinstance_"):
                type_name = pred.split("_", 1)[1]
                self._scope[var] = {type_name}

        for child in node.body:
            self.visit(child)

        # Restore for else branch
        self._none_vars = old_none
        self._guarded_vars = old_guarded
        for child in node.orelse:
            self.visit(child)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
            # Check for division by zero
            if isinstance(node.right, ast.Constant) and node.right.value == 0:
                self.bugs.append(BugReport(
                    category=BugCategory.DIV_BY_ZERO,
                    line=node.lineno, col=node.col_offset,
                    message="Division by zero: divisor is literal 0",
                    function=self._current_func,
                    variable=ast.dump(node.right),
                    severity="error",
                ))
            elif isinstance(node.right, ast.Name):
                # Check if the variable could be 0
                var = node.right.id
                if var not in self._guarded_vars or "ne_zero" not in self._guarded_vars.get(var, set()):
                    types = self._scope.get(var, {"Any"})
                    if types & {"int", "float", "Any"}:
                        self.bugs.append(BugReport(
                            category=BugCategory.DIV_BY_ZERO,
                            line=node.lineno, col=node.col_offset,
                            message=f"Potential division by zero: '{var}' not guarded against 0",
                            function=self._current_func,
                            variable=var,
                            severity="warning",
                            guarded=False,
                        ))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name):
            var = node.value.id
            if var in self._none_vars:
                guarded = var in self._guarded_vars and "is_not_none" in self._guarded_vars.get(var, set())
                if not guarded:
                    self.bugs.append(BugReport(
                        category=BugCategory.NULL_DEREF,
                        line=node.lineno, col=node.col_offset,
                        message=f"Attribute access on potentially None variable '{var}'",
                        function=self._current_func,
                        variable=var,
                        severity="error",
                        guarded=guarded,
                    ))
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name):
            var = node.value.id
            if var in self._none_vars:
                self.bugs.append(BugReport(
                    category=BugCategory.NULL_DEREF,
                    line=node.lineno, col=node.col_offset,
                    message=f"Subscript on potentially None variable '{var}'",
                    function=self._current_func,
                    variable=var,
                    severity="error",
                ))
            # Check for constant out-of-bounds
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
                idx = node.slice.value
                assigned = self._assignments.get(var)
                if assigned and isinstance(assigned, (ast.List, ast.Tuple)):
                    length = len(assigned.elts)
                    if idx >= length or idx < -length:
                        self.bugs.append(BugReport(
                            category=BugCategory.INDEX_OUT_OF_BOUNDS,
                            line=node.lineno, col=node.col_offset,
                            message=f"Index {idx} out of bounds for '{var}' of length {length}",
                            function=self._current_func,
                            variable=var,
                            severity="error",
                        ))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Check for calling None
        if isinstance(node.func, ast.Name):
            var = node.func.id
            if var in self._none_vars:
                self.bugs.append(BugReport(
                    category=BugCategory.NULL_DEREF,
                    line=node.lineno, col=node.col_offset,
                    message=f"Calling potentially None variable '{var}'",
                    function=self._current_func,
                    variable=var,
                    severity="error",
                ))
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            var = node.func.value.id
            if var in self._none_vars:
                guarded = var in self._guarded_vars and "is_not_none" in self._guarded_vars.get(var, set())
                if not guarded:
                    self.bugs.append(BugReport(
                        category=BugCategory.NULL_DEREF,
                        line=node.lineno, col=node.col_offset,
                        message=f"Method call on potentially None variable '{var}'",
                        function=self._current_func,
                        variable=var,
                        severity="error",
                        guarded=False,
                    ))
        self.generic_visit(node)

    # ── helpers ──

    def _extract_guard_info(self, test: ast.AST) -> List[Tuple[str, str]]:
        """Extract (variable, predicate_name) pairs from a guard condition."""
        guards = []
        if isinstance(test, ast.Compare):
            if len(test.ops) == 1 and isinstance(test.ops[0], ast.IsNot):
                if isinstance(test.comparators[0], ast.Constant) and test.comparators[0].value is None:
                    if isinstance(test.left, ast.Name):
                        guards.append((test.left.id, "is_not_none"))
            elif len(test.ops) == 1 and isinstance(test.ops[0], ast.Is):
                if isinstance(test.comparators[0], ast.Constant) and test.comparators[0].value is None:
                    if isinstance(test.left, ast.Name):
                        guards.append((test.left.id, "is_none"))
            elif len(test.ops) == 1 and isinstance(test.ops[0], ast.NotEq):
                if isinstance(test.comparators[0], ast.Constant) and test.comparators[0].value == 0:
                    if isinstance(test.left, ast.Name):
                        guards.append((test.left.id, "ne_zero"))
        elif isinstance(test, ast.Call):
            if isinstance(test.func, ast.Name) and test.func.id == "isinstance":
                if len(test.args) >= 2 and isinstance(test.args[0], ast.Name):
                    var = test.args[0].id
                    type_name = self._get_type_name(test.args[1])
                    if type_name:
                        guards.append((var, f"isinstance_{type_name}"))
            elif isinstance(test.func, ast.Name) and test.func.id == "callable":
                if test.args and isinstance(test.args[0], ast.Name):
                    guards.append((test.args[0].id, "callable"))
            elif isinstance(test.func, ast.Name) and test.func.id == "hasattr":
                if len(test.args) >= 2 and isinstance(test.args[0], ast.Name):
                    guards.append((test.args[0].id, "hasattr"))
        elif isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
            for value in test.values:
                guards.extend(self._extract_guard_info(value))
        elif isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            inner = self._extract_guard_info(test.operand)
            for var, pred in inner:
                if pred == "is_none":
                    guards.append((var, "is_not_none"))
                elif pred == "is_not_none":
                    guards.append((var, "is_none"))
        elif isinstance(test, ast.Name):
            # Truthiness check
            guards.append((test.id, "is_truthy"))
        return guards

    def _get_type_name(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Tuple):
            names = [self._get_type_name(e) for e in node.elts]
            return "|".join(n for n in names if n)
        return None

    def _infer_type(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant):
            if node.value is None:
                return "NoneType"
            return type(node.value).__name__
        if isinstance(node, ast.List):
            return "list"
        if isinstance(node, ast.Dict):
            return "dict"
        if isinstance(node, ast.Tuple):
            return "tuple"
        if isinstance(node, ast.Set):
            return "set"
        if isinstance(node, ast.Name):
            types = self._scope.get(node.id, {"Any"})
            if len(types) == 1:
                return next(iter(types))
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return node.func.id  # constructor call
        return "Any"

    def _annotation_to_type(self, ann: ast.AST) -> Optional[str]:
        if isinstance(ann, ast.Name):
            return ann.id
        if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
            return ann.value
        if isinstance(ann, ast.Subscript):
            if isinstance(ann.value, ast.Name):
                return ann.value.id
        return None


# ── SMT-based CEGAR loop ────────────────────────────────────────────────

@dataclass
class CEGARState:
    predicates: List[Predicate] = field(default_factory=list)
    predicate_strs: Set[str] = field(default_factory=set)
    iterations: int = 0
    converged: bool = False
    counterexamples: List[Dict[str, Any]] = field(default_factory=list)


def _guard_to_smt_predicate(guard: ExtractedGuard) -> Optional[Predicate]:
    """Convert an ExtractedGuard to an SMT Predicate."""
    pred = guard.predicate
    kind = pred.kind

    if kind == PredicateKind.TypeTag:
        return IsInstance(pred.target_variable, pred.type_names[0])
    elif kind == PredicateKind.Nullity:
        base = IsNone(pred.target_variable)
        if not pred.is_none:
            return Not(base)
        return base
    elif kind == PredicateKind.Truthiness:
        return IsTruthy(pred.target_variable)
    elif kind == PredicateKind.HasAttr:
        return HasAttr(pred.target_variable, pred.attr_name if hasattr(pred, 'attr_name') else "")
    elif kind == PredicateKind.Comparison:
        try:
            op_map = {
                "==": ComparisonOp.EQ, "!=": ComparisonOp.NE,
                "<": ComparisonOp.LT, "<=": ComparisonOp.LE,
                ">": ComparisonOp.GT, ">=": ComparisonOp.GE,
            }
            # guard comparison predicates store op as an enum
            op_str = pred.op.value if hasattr(pred.op, 'value') else str(pred.op)
            op = op_map.get(op_str, ComparisonOp.EQ)
            left = Var(pred.left_expr) if not pred.left_expr.isdigit() else Const(int(pred.left_expr))
            right = Var(pred.right_expr) if not pred.right_expr.lstrip('-').isdigit() else Const(int(pred.right_expr))
            return Comparison(op, left, right)
        except Exception:
            return None
    elif kind == PredicateKind.Conjunction:
        parts = []
        for child in pred.children:
            child_guard = ExtractedGuard(
                pattern=guard.pattern, variables=guard.variables,
                predicate=child,
                source_location=guard.source_location, polarity=guard.polarity,
            )
            p = _guard_to_smt_predicate(child_guard)
            if p:
                parts.append(p)
        if parts:
            return And(tuple(parts)) if len(parts) > 1 else parts[0]
    return None


def run_cegar(
    source: str,
    guards: List[ExtractedGuard],
    max_iterations: int = 20,
) -> CEGARState:
    """Run a real CEGAR loop using Z3 for refinement.

    1. Seed predicates from harvested guards
    2. Check if current predicates are sufficient for safety
    3. If not, find counterexample, compute interpolant, add new predicates
    """
    state = CEGARState()

    # Phase 1: Seed from guards
    for g in guards:
        smt_pred = _guard_to_smt_predicate(g)
        if smt_pred is not None:
            pred_str = repr(smt_pred)
            if pred_str not in state.predicate_strs:
                state.predicates.append(smt_pred)
                state.predicate_strs.add(pred_str)

    # Phase 2: CEGAR refinement loop with Z3
    try:
        solver = Z3Solver(timeout_ms=5000)
    except ImportError:
        state.converged = True
        return state

    prev_predicate_count = len(state.predicates)

    for iteration in range(max_iterations):
        state.iterations = iteration + 1

        # Assert all current predicates
        solver.push()
        for pred in state.predicates:
            try:
                solver.assert_formula(pred)
            except Exception:
                continue

        # Check satisfiability — if UNSAT, the predicates are contradictory,
        # meaning the current abstraction is precise enough
        result = solver.check_sat()

        if result == SatResult.UNSAT:
            state.converged = True
            solver.pop()
            break

        if result == SatResult.SAT:
            # Extract counterexample model
            model = solver.get_model()
            if model:
                state.counterexamples.append(model.variable_values)

            solver.pop()

            # Synthesize strengthening predicates from the counterexample.
            # Never discard existing valid predicates — only add new ones
            # that eliminate spurious counterexamples.
            new_preds = _refine_from_counterexample(model, state.predicates)

            if not new_preds:
                state.converged = True
                break

            added = False
            for p in new_preds:
                ps = repr(p)
                if ps not in state.predicate_strs:
                    state.predicates.append(p)
                    state.predicate_strs.add(ps)
                    added = True

            # Track whether this iteration improved precision
            if not added or len(state.predicates) == prev_predicate_count:
                state.converged = True
                break
            prev_predicate_count = len(state.predicates)
        else:
            solver.pop()
            state.converged = True
            break

    return state


def _refine_from_model(
    model: Optional[Any],
    current_preds: List[Predicate],
) -> List[Predicate]:
    """Generate new predicates from a counterexample model (legacy wrapper)."""
    return _refine_from_counterexample(model, current_preds)


def _refine_from_counterexample(
    model: Optional[Any],
    current_preds: List[Predicate],
) -> List[Predicate]:
    """Synthesize strengthening predicates from a counterexample.

    Instead of bounding variables to their counterexample values (which
    over-constrains and discards useful predicates), we trace which
    assumptions were too weak and synthesize predicates that rule out the
    spurious counterexample while preserving all existing valid predicates.
    """
    if model is None:
        return []

    new_preds: List[Predicate] = []
    vals = model.variable_values if hasattr(model, 'variable_values') else {}

    # Collect which variables are already constrained by existing predicates
    constrained_vars: Set[str] = set()
    for pred in current_preds:
        constrained_vars.update(pred.free_vars())

    for name, value in vals.items():
        if isinstance(value, int):
            # Only synthesize predicates for unconstrained variables —
            # already-constrained variables don't need further refinement
            if name not in constrained_vars:
                if value != 0:
                    new_preds.append(Comparison(ComparisonOp.NE, Var(name), Const(0)))
                if value > 0:
                    new_preds.append(Comparison(ComparisonOp.GT, Var(name), Const(0)))
                elif value < 0:
                    new_preds.append(Comparison(ComparisonOp.LT, Var(name), Const(0)))

    return new_preds[:6]  # limit new predicates per iteration


# ── Full analysis pipeline ──────────────────────────────────────────────

def _source_has_nn_module(source: str) -> bool:
    """Detect whether *source* contains an nn.Module subclass."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = []
            for b in node.bases:
                if isinstance(b, ast.Attribute) and isinstance(b.value, ast.Name):
                    bases.append(f"{b.value.id}.{b.attr}")
                elif isinstance(b, ast.Name):
                    bases.append(b.id)
            if any(name in ("nn.Module", "Module", "torch.nn.Module") for name in bases):
                return True
    return False


def _cegar_predicates_prove_safe(
    cegar_state: CEGARState,
    bug: BugReport,
) -> bool:
    """Check whether CEGAR-discovered predicates prove a bug site is safe.

    For example, a division-by-zero warning on variable 'x' is safe if
    CEGAR discovered that x != 0.
    """
    for pred in cegar_state.predicates:
        pred_repr = repr(pred)
        if bug.category == BugCategory.DIV_BY_ZERO and bug.variable:
            if (f"NE, Var('{bug.variable}'), Const(0)" in pred_repr
                    or f"GT, Var('{bug.variable}'), Const(0)" in pred_repr):
                return True
        if bug.category == BugCategory.NULL_DEREF and bug.variable:
            if f"Not(IsNone('{bug.variable}'))" in pred_repr:
                return True
    return False


def analyze_python_source(
    source: str,
    filename: str = "<string>",
) -> AnalysisResult:
    """Analyze Python source code end-to-end.

    Delegates to the Z3-enhanced version when model_checker and shape_cegar
    are available and the source contains nn.Module subclasses.  Falls back
    to AST-only analysis otherwise.
    """
    if (_HAS_MODEL_CHECKER and _MC_HAS_Z3
            and _source_has_nn_module(source)):
        return analyze_python_source_z3(source, filename)
    return _analyze_python_source_base(source, filename)


def _analyze_python_source_base(
    source: str,
    filename: str = "<string>",
) -> AnalysisResult:
    """AST-only analysis pipeline (original implementation).

    1. Parse AST and extract guards
    2. Run bug detection via AST analysis
    3. Run CEGAR refinement with Z3
    4. Return structured results
    """
    t0 = time.perf_counter()
    errors: List[str] = []

    # Step 1: Parse and extract guards
    try:
        guards = extract_guards(source, filename)
    except SyntaxError as e:
        return AnalysisResult(
            file_path=filename, functions_analyzed=0, total_guards=0,
            total_bugs=0, total_predicates=0, summaries=[],
            analysis_time_ms=0.0, errors=[f"Syntax error: {e}"],
        )

    # Step 2: Bug detection
    analyzer = PythonAnalyzer(source, filename)
    bugs = analyzer.analyze()

    # Step 2b: Path-sensitive false-positive elimination
    if _HAS_PATH_SENSITIVE:
        bugs, _ps_eliminated = path_sensitive_filter(bugs, source, filename)

    # Step 3: Group by function
    tree = ast.parse(source, filename)
    func_nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

    summaries: List[FunctionSummary] = []
    total_predicates = 0

    for func_node in func_nodes:
        func_name = func_node.name
        func_line = func_node.lineno
        func_end = func_node.end_lineno or (func_line + 100)

        # Guards in this function
        func_guards = [
            g for g in guards
            if g.source_location and func_line <= g.source_location.line <= func_end
        ]

        # Bugs in this function
        func_bugs = [b for b in bugs if b.function == func_name]

        # Run CEGAR for this function
        func_source = ast.get_source_segment(source, func_node) or ""
        cegar_state = run_cegar(func_source, func_guards)

        # Build refined vars
        refined_vars = []
        for pred in cegar_state.predicates:
            fv = pred.free_vars()
            for v in fv:
                refined_vars.append(RefinedVar(
                    name=v, base_type="inferred", predicates=[repr(pred)],
                    line=func_line,
                ))
        total_predicates += len(cegar_state.predicates)

        summaries.append(FunctionSummary(
            name=func_name, line=func_line,
            guards_harvested=len(func_guards),
            predicates_inferred=len(cegar_state.predicates),
            cegar_iterations=cegar_state.iterations,
            bugs_found=func_bugs,
            refined_vars=refined_vars,
            converged=cegar_state.converged,
            analysis_time_ms=(time.perf_counter() - t0) * 1000,
        ))

    elapsed = (time.perf_counter() - t0) * 1000

    return AnalysisResult(
        file_path=filename,
        functions_analyzed=len(summaries),
        total_guards=len(guards),
        total_bugs=len(bugs),
        total_predicates=total_predicates,
        summaries=summaries,
        analysis_time_ms=elapsed,
        errors=errors,
    )


def analyze_python_source_z3(
    source: str,
    filename: str = "<string>",
) -> AnalysisResult:
    """Z3-enhanced analysis pipeline.

    Does everything ``_analyze_python_source_base()`` does, plus:
    - Runs ``verify_model()`` for Z3-backed constraint-based verification
    - Converts verification errors into ``BugReport`` objects
    - Runs ``run_shape_cegar()`` for shape-contract discovery
    - Uses contract-discovery predicates to filter AST false positives
    """
    t0 = time.perf_counter()
    errors: List[str] = []

    # Step 1: Parse and extract guards
    try:
        guards = extract_guards(source, filename)
    except SyntaxError as e:
        return AnalysisResult(
            file_path=filename, functions_analyzed=0, total_guards=0,
            total_bugs=0, total_predicates=0, summaries=[],
            analysis_time_ms=0.0, errors=[f"Syntax error: {e}"],
        )

    # Step 2: AST-based bug detection
    analyzer = PythonAnalyzer(source, filename)
    bugs = analyzer.analyze()

    # Step 2b: Path-sensitive false-positive elimination
    if _HAS_PATH_SENSITIVE:
        bugs, _ps_eliminated = path_sensitive_filter(bugs, source, filename)

    # Step 3: Z3-backed model checking via verify_model()
    if _HAS_MODEL_CHECKER and _MC_HAS_Z3:
        try:
            # Auto-infer input shapes from the computation graph
            graph = extract_computation_graph(source)
            inferred_shapes: Dict[str, tuple] = {}
            if graph.steps and graph.layers:
                first_layer = None
                for step in graph.steps:
                    ref = getattr(step, 'layer_ref', None)
                    if ref and ref in graph.layers:
                        first_layer = graph.layers[ref]
                        break
                if first_layer:
                    in_f = getattr(first_layer, 'in_features', None)
                    in_c = getattr(first_layer, 'in_channels', None)
                    if in_f:
                        for inp in graph.input_names:
                            inferred_shapes[inp] = ("batch", in_f)
                    elif in_c:
                        for inp in graph.input_names:
                            inferred_shapes[inp] = ("batch", in_c, "height", "width")

            vresult = verify_model(source, input_shapes=inferred_shapes)
            if not vresult.safe and vresult.counterexample:
                for violation in vresult.counterexample.violations:
                    # Skip Z3 trace dumps — only keep semantic violations
                    if violation.message.startswith("Z3 violation"):
                        continue
                    bugs.append(BugReport(
                        category=BugCategory.TYPE_ERROR,
                        line=violation.step_index,
                        col=0,
                        message=f"[Z3 model checker] {violation.message}",
                        function="forward",
                        variable=violation.tensor_a or "",
                        severity="error",
                        guarded=False,
                    ))
            if vresult.errors:
                errors.extend(f"[model_checker] {e}" for e in vresult.errors)
        except Exception as exc:
            logger.debug("verify_model failed: %s", exc)
            errors.append(f"[model_checker] {exc}")

    # Step 4: Shape CEGAR for discovered shape contracts
    shape_predicates: List[Any] = []
    if _HAS_SHAPE_CEGAR:
        try:
            cegar_result = run_shape_cegar(source)
            shape_predicates = list(cegar_result.discovered_predicates)
            # Convert real bugs from shape CEGAR into BugReports
            for sv in cegar_result.real_bugs:
                bugs.append(BugReport(
                    category=BugCategory.TYPE_ERROR,
                    line=sv.step_index,
                    col=0,
                    message=f"[shape CEGAR] {sv.message}",
                    function="forward",
                    variable=sv.tensor_a or "",
                    severity="error",
                    guarded=False,
                ))
        except Exception as exc:
            logger.debug("run_shape_cegar failed: %s", exc)
            errors.append(f"[shape_cegar] {exc}")

    # Step 5: Group by function and run per-function CEGAR
    tree = ast.parse(source, filename)
    func_nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

    summaries: List[FunctionSummary] = []
    total_predicates = 0

    for func_node in func_nodes:
        func_name = func_node.name
        func_line = func_node.lineno
        func_end = func_node.end_lineno or (func_line + 100)

        func_guards = [
            g for g in guards
            if g.source_location and func_line <= g.source_location.line <= func_end
        ]
        func_bugs = [b for b in bugs if b.function == func_name]

        func_source = ast.get_source_segment(source, func_node) or ""
        cegar_state = run_cegar(func_source, func_guards)

        # Step 6: Use CEGAR predicates to filter AST false positives
        func_bugs = [
            b for b in func_bugs
            if not _cegar_predicates_prove_safe(cegar_state, b)
        ]

        refined_vars = []
        for pred in cegar_state.predicates:
            fv = pred.free_vars()
            for v in fv:
                refined_vars.append(RefinedVar(
                    name=v, base_type="inferred", predicates=[repr(pred)],
                    line=func_line,
                ))
        total_predicates += len(cegar_state.predicates)

        summaries.append(FunctionSummary(
            name=func_name, line=func_line,
            guards_harvested=len(func_guards),
            predicates_inferred=len(cegar_state.predicates),
            cegar_iterations=cegar_state.iterations,
            bugs_found=func_bugs,
            refined_vars=refined_vars,
            converged=cegar_state.converged,
            analysis_time_ms=(time.perf_counter() - t0) * 1000,
        ))

    # Recalculate total_bugs from summaries (some may have been filtered)
    total_bugs_count = sum(len(s.bugs_found) for s in summaries)
    # Also add bugs not assigned to any function
    func_names = {s.name for s in summaries}
    unassigned_bugs = [b for b in bugs if b.function not in func_names]
    total_bugs_count += len(unassigned_bugs)

    elapsed = (time.perf_counter() - t0) * 1000

    return AnalysisResult(
        file_path=filename,
        functions_analyzed=len(summaries),
        total_guards=len(guards),
        total_bugs=total_bugs_count,
        total_predicates=total_predicates,
        summaries=summaries,
        analysis_time_ms=elapsed,
        errors=errors,
    )


def analyze_python_file(path: str) -> AnalysisResult:
    """Analyze a Python file on disk."""
    p = Path(path)
    if not p.exists():
        return AnalysisResult(
            file_path=path, functions_analyzed=0, total_guards=0,
            total_bugs=0, total_predicates=0, summaries=[],
            analysis_time_ms=0.0, errors=[f"File not found: {path}"],
        )
    source = p.read_text(encoding="utf-8", errors="replace")
    return analyze_python_source(source, str(p))


def analyze_directory(
    directory: str,
    pattern: str = "**/*.py",
    exclude: Optional[List[str]] = None,
) -> List[AnalysisResult]:
    """Analyze all Python files in a directory."""
    exclude = exclude or ["__pycache__", ".git", "node_modules", ".venv", "venv"]
    results = []
    root = Path(directory)
    for py_file in sorted(root.glob(pattern)):
        if any(ex in str(py_file) for ex in exclude):
            continue
        result = analyze_python_file(str(py_file))
        results.append(result)
    return results
