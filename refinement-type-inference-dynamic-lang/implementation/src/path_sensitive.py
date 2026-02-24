"""Path-sensitive guard analysis for false-positive elimination.

The core insight: the existing PythonAnalyzer is path-*insensitive* — when
different conditional branches produce different variable states, constraints
from one branch can leak into another, producing false positives.

This module adds path-sensitive tracking so that:
  - ``if isinstance(x, Tensor): x.reshape(3,4)`` only checks reshape under
    the constraint that x IS a Tensor
  - ``if x is not None: x.foo()`` doesn't warn about null deref
  - Nested conditions create conjunctions of constraints
  - The else branch sees the *negation* of the guard
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ── Path constraint representation ──────────────────────────────────────

@dataclass(frozen=True)
class PathConstraint:
    """A single constraint active on the current execution path."""
    variable: str
    predicate: str          # e.g. "isinstance_Tensor", "is_not_none", "gt_0"
    negated: bool = False   # True when we are in the else-branch

    def negate(self) -> PathConstraint:
        return PathConstraint(
            variable=self.variable,
            predicate=self.predicate,
            negated=not self.negated,
        )

    @property
    def effective_predicate(self) -> str:
        """The predicate as it effectively holds on this path."""
        if not self.negated:
            return self.predicate
        # Negate known predicate pairs
        _negations = {
            "is_none": "is_not_none",
            "is_not_none": "is_none",
            "is_truthy": "is_falsy",
            "is_falsy": "is_truthy",
            "ne_zero": "eq_zero",
            "eq_zero": "ne_zero",
        }
        return _negations.get(self.predicate, f"not_{self.predicate}")


@dataclass
class PathContext:
    """Stack of active path constraints (conjunction of all active guards)."""
    constraints: List[PathConstraint] = field(default_factory=list)

    def push(self, constraint: PathConstraint) -> None:
        self.constraints.append(constraint)

    def pop(self) -> Optional[PathConstraint]:
        return self.constraints.pop() if self.constraints else None

    def active_predicates_for(self, variable: str) -> Set[str]:
        """Return the set of effective predicates active for *variable*."""
        return {
            c.effective_predicate
            for c in self.constraints
            if c.variable == variable
        }

    def has_type_constraint(self, variable: str) -> Optional[str]:
        """If *variable* is constrained to a specific type, return it."""
        for c in self.constraints:
            if c.variable != variable:
                continue
            ep = c.effective_predicate
            if ep.startswith("isinstance_"):
                return ep.split("_", 1)[1]
            if ep.startswith("not_isinstance_"):
                return None  # negated isinstance — we know it's NOT that type
        return None

    def variable_is_none(self, variable: str) -> Optional[bool]:
        """Return True/False if the path proves the variable is/isn't None."""
        for c in self.constraints:
            if c.variable != variable:
                continue
            ep = c.effective_predicate
            if ep == "is_none":
                return True
            if ep == "is_not_none":
                return False
        return None

    def copy(self) -> PathContext:
        return PathContext(constraints=list(self.constraints))


# ── Path-sensitive AST analyzer ─────────────────────────────────────────

@dataclass
class PathBugReport:
    """A bug report annotated with the path constraints under which it occurs."""
    category: str
    line: int
    col: int
    message: str
    function: str
    variable: str
    severity: str = "warning"
    path_constraints: List[PathConstraint] = field(default_factory=list)


class PathSensitiveAnalyzer(ast.NodeVisitor):
    """Walk a Python AST with path-sensitive constraint tracking.

    Unlike the base PythonAnalyzer, this properly:
    1. Saves/restores _scope (type narrowing) across branches
    2. Applies negated guards in else branches
    3. Tracks a full path constraint stack for nested conditions
    4. Uses path constraints to suppress false positives
    """

    def __init__(self, source: str, filename: str = "<string>"):
        self.source = source
        self.filename = filename
        self.bugs: List[PathBugReport] = []
        self._current_func = "<module>"
        self._scope: Dict[str, Set[str]] = {}
        self._none_vars: Set[str] = set()
        self._guarded_vars: Dict[str, Set[str]] = {}
        self._assignments: Dict[str, ast.AST] = {}
        self._path = PathContext()

    def analyze(self) -> List[PathBugReport]:
        tree = ast.parse(self.source, self.filename)
        self.visit(tree)
        return self.bugs

    # ── Function scope ──────────────────────────────────────────────────

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        old_func = self._current_func
        old_scope = dict(self._scope)
        old_none = set(self._none_vars)
        old_guarded = {k: set(v) for k, v in self._guarded_vars.items()}
        old_path = self._path.copy()

        self._current_func = node.name
        self._scope = {}
        self._none_vars = set()
        self._guarded_vars = {}
        self._path = PathContext()

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
        self._guarded_vars = old_guarded
        self._path = old_path

    visit_AsyncFunctionDef = visit_FunctionDef

    # ── Assignments ─────────────────────────────────────────────────────

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

    # ── Path-sensitive If ───────────────────────────────────────────────

    def visit_If(self, node: ast.If) -> None:
        guards = self._extract_guard_info(node.test)

        # Snapshot state before branching
        saved_scope = {k: set(v) for k, v in self._scope.items()}
        saved_none = set(self._none_vars)
        saved_guarded = {k: set(v) for k, v in self._guarded_vars.items()}
        saved_path_len = len(self._path.constraints)

        # ── True branch: push positive constraints ──
        path_constraints_pushed = []
        for var, pred in guards:
            constraint = PathConstraint(variable=var, predicate=pred)
            self._path.push(constraint)
            path_constraints_pushed.append(constraint)
            self._guarded_vars.setdefault(var, set()).add(pred)
            if pred == "is_not_none":
                self._none_vars.discard(var)
            elif pred == "is_none":
                self._none_vars.add(var)
            elif pred.startswith("isinstance_"):
                type_name = pred.split("_", 1)[1]
                self._scope[var] = {type_name}
                # isinstance proves the variable is not None
                self._none_vars.discard(var)

        for child in node.body:
            self.visit(child)

        # Pop true-branch constraints
        while len(self._path.constraints) > saved_path_len:
            self._path.pop()

        # ── Else branch: restore state and push negated constraints ──
        self._scope = {k: set(v) for k, v in saved_scope.items()}
        self._none_vars = set(saved_none)
        self._guarded_vars = {k: set(v) for k, v in saved_guarded.items()}

        negated_pushed = []
        for var, pred in guards:
            neg_constraint = PathConstraint(variable=var, predicate=pred, negated=True)
            self._path.push(neg_constraint)
            negated_pushed.append(neg_constraint)
            neg_pred = neg_constraint.effective_predicate
            self._guarded_vars.setdefault(var, set()).add(neg_pred)
            if neg_pred == "is_not_none":
                self._none_vars.discard(var)
            elif neg_pred == "is_none":
                self._none_vars.add(var)
            elif neg_pred.startswith("isinstance_"):
                type_name = neg_pred.split("_", 1)[1]
                self._scope[var] = {type_name}
            elif neg_pred.startswith("not_isinstance_"):
                pass  # we know it's NOT that type, but can't narrow further

        for child in node.orelse:
            self.visit(child)

        # Pop else-branch constraints and restore to pre-if state
        while len(self._path.constraints) > saved_path_len:
            self._path.pop()

        self._scope = saved_scope
        self._none_vars = saved_none
        self._guarded_vars = saved_guarded

    # ── Bug detection (path-aware) ──────────────────────────────────────

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
            if isinstance(node.right, ast.Constant) and node.right.value == 0:
                self.bugs.append(PathBugReport(
                    category="DIV_BY_ZERO",
                    line=node.lineno, col=node.col_offset,
                    message="Division by zero: divisor is literal 0",
                    function=self._current_func,
                    variable=ast.dump(node.right),
                    severity="error",
                    path_constraints=list(self._path.constraints),
                ))
            elif isinstance(node.right, ast.Name):
                var = node.right.id
                active = self._path.active_predicates_for(var)
                guarded_set = self._guarded_vars.get(var, set())
                if "ne_zero" not in active and "ne_zero" not in guarded_set:
                    types = self._scope.get(var, {"Any"})
                    if types & {"int", "float", "Any"}:
                        self.bugs.append(PathBugReport(
                            category="DIV_BY_ZERO",
                            line=node.lineno, col=node.col_offset,
                            message=f"Potential division by zero: '{var}' not guarded against 0",
                            function=self._current_func,
                            variable=var,
                            severity="warning",
                            path_constraints=list(self._path.constraints),
                        ))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name):
            var = node.value.id
            # Check path constraints for null info
            path_null = self._path.variable_is_none(var)
            if path_null is False:
                # Path proves non-null, skip
                pass
            elif var in self._none_vars:
                guarded = var in self._guarded_vars and "is_not_none" in self._guarded_vars.get(var, set())
                if not guarded:
                    self.bugs.append(PathBugReport(
                        category="NULL_DEREF",
                        line=node.lineno, col=node.col_offset,
                        message=f"Attribute access on potentially None variable '{var}'",
                        function=self._current_func,
                        variable=var,
                        severity="error",
                        path_constraints=list(self._path.constraints),
                    ))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            var = node.func.id
            if var in self._none_vars:
                path_null = self._path.variable_is_none(var)
                if path_null is not False:
                    self.bugs.append(PathBugReport(
                        category="NULL_DEREF",
                        line=node.lineno, col=node.col_offset,
                        message=f"Calling potentially None variable '{var}'",
                        function=self._current_func,
                        variable=var,
                        severity="error",
                        path_constraints=list(self._path.constraints),
                    ))
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            var = node.func.value.id
            path_null = self._path.variable_is_none(var)
            if path_null is False:
                pass  # path proves non-null
            elif var in self._none_vars:
                guarded = var in self._guarded_vars and "is_not_none" in self._guarded_vars.get(var, set())
                if not guarded:
                    self.bugs.append(PathBugReport(
                        category="NULL_DEREF",
                        line=node.lineno, col=node.col_offset,
                        message=f"Method call on potentially None variable '{var}'",
                        function=self._current_func,
                        variable=var,
                        severity="error",
                        path_constraints=list(self._path.constraints),
                    ))
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name):
            var = node.value.id
            path_null = self._path.variable_is_none(var)
            if path_null is False:
                pass
            elif var in self._none_vars:
                self.bugs.append(PathBugReport(
                    category="NULL_DEREF",
                    line=node.lineno, col=node.col_offset,
                    message=f"Subscript on potentially None variable '{var}'",
                    function=self._current_func,
                    variable=var,
                    severity="error",
                    path_constraints=list(self._path.constraints),
                ))
        self.generic_visit(node)

    # ── Guard extraction ────────────────────────────────────────────────

    def _extract_guard_info(self, test: ast.AST) -> List[Tuple[str, str]]:
        """Extract (variable, predicate_name) pairs from a guard condition."""
        guards: List[Tuple[str, str]] = []
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
            elif len(test.ops) == 1 and isinstance(test.ops[0], ast.Gt):
                if isinstance(test.left, ast.Name):
                    right_str = self._expr_to_str(test.comparators[0])
                    guards.append((test.left.id, f"gt_{right_str}"))
            elif len(test.ops) == 1 and isinstance(test.ops[0], ast.Lt):
                if isinstance(test.left, ast.Name):
                    right_str = self._expr_to_str(test.comparators[0])
                    guards.append((test.left.id, f"lt_{right_str}"))
            elif len(test.ops) == 1 and isinstance(test.ops[0], ast.GtE):
                if isinstance(test.left, ast.Name):
                    right_str = self._expr_to_str(test.comparators[0])
                    guards.append((test.left.id, f"gte_{right_str}"))
            elif len(test.ops) == 1 and isinstance(test.ops[0], ast.LtE):
                if isinstance(test.left, ast.Name):
                    right_str = self._expr_to_str(test.comparators[0])
                    guards.append((test.left.id, f"lte_{right_str}"))
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
                    attr = ""
                    if isinstance(test.args[1], ast.Constant):
                        attr = str(test.args[1].value)
                    guards.append((test.args[0].id, f"hasattr_{attr}"))
        elif isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
            for value in test.values:
                guards.extend(self._extract_guard_info(value))
        elif isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or):
            # For Or, we can't narrow types, but track for constraint info
            # Only extract if all branches constrain the same variable
            sub_guards = [self._extract_guard_info(v) for v in test.values]
            vars_seen = set()
            for sg in sub_guards:
                for var, _ in sg:
                    vars_seen.add(var)
            # If single variable, track as disjunction
            if len(vars_seen) == 1:
                var = next(iter(vars_seen))
                preds = []
                for sg in sub_guards:
                    for _, pred in sg:
                        preds.append(pred)
                if preds:
                    guards.append((var, f"or_{'_'.join(preds)}"))
        elif isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            inner = self._extract_guard_info(test.operand)
            for var, pred in inner:
                if pred == "is_none":
                    guards.append((var, "is_not_none"))
                elif pred == "is_not_none":
                    guards.append((var, "is_none"))
                elif pred == "is_truthy":
                    guards.append((var, "is_falsy"))
                elif pred == "is_falsy":
                    guards.append((var, "is_truthy"))
                else:
                    guards.append((var, f"not_{pred}"))
        elif isinstance(test, ast.Name):
            guards.append((test.id, "is_truthy"))
        elif isinstance(test, ast.Attribute):
            # Handle self.training and similar attribute guards
            if (isinstance(test.value, ast.Name)
                    and test.value.id == "self"
                    and test.attr == "training"):
                guards.append(("self", "training"))
        return guards

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _expr_to_str(node: ast.AST) -> str:
        try:
            return ast.unparse(node)
        except Exception:
            return "<expr>"

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
                return node.func.id
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


# ── Pipeline integration ────────────────────────────────────────────────

def path_sensitive_filter(
    bugs: list,
    source: str,
    filename: str = "<string>",
) -> Tuple[list, int]:
    """Filter false positives from a bug list using path-sensitive analysis.

    Runs PathSensitiveAnalyzer on *source* and removes any bugs from *bugs*
    that the path-sensitive analysis proves cannot occur.

    Returns (filtered_bugs, eliminated_count).
    """
    ps_analyzer = PathSensitiveAnalyzer(source, filename)
    ps_bugs = ps_analyzer.analyze()

    # Build a set of (line, variable, category) tuples that the path-sensitive
    # analysis still considers real bugs.
    ps_bug_keys: Set[Tuple[int, str, str]] = set()
    for b in ps_bugs:
        ps_bug_keys.add((b.line, b.variable, b.category))

    filtered = []
    eliminated = 0
    for bug in bugs:
        cat_str = bug.category.name if hasattr(bug.category, 'name') else str(bug.category)
        key = (bug.line, bug.variable, cat_str)
        if key in ps_bug_keys:
            filtered.append(bug)
        else:
            # The path-sensitive analysis doesn't report this bug — it was
            # a false positive eliminated by path constraints
            eliminated += 1

    return filtered, eliminated
