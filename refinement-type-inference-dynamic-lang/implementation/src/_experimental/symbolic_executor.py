"""Symbolic execution engine for Python programs.

Performs path-sensitive analysis by executing programs with symbolic values,
collecting path constraints, and generating concrete test inputs that exercise
different execution paths. Uses interval arithmetic for constraint solving.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import textwrap
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Symbolic value classes
# ---------------------------------------------------------------------------

class SymBase:
    _counter: int = 0
    @classmethod
    def _fresh_id(cls) -> int:
        cls._counter += 1; return cls._counter
    def concretize(self) -> Any: raise NotImplementedError
    def is_concrete(self) -> bool: raise NotImplementedError
    def substitute(self, mapping: Dict[str, Any]) -> Any: raise NotImplementedError


class SymInt(SymBase):
    """Symbolic integer with optional constraint range [lo, hi]."""
    def __init__(self, name: str, lo: Optional[int] = None,
                 hi: Optional[int] = None, concrete: Optional[int] = None):
        self.name, self.concrete, self.sym_id = name, concrete, self._fresh_id()
        self.lo = lo if lo is not None else -(2 ** 31)
        self.hi = hi if hi is not None else (2 ** 31) - 1
    def concretize(self) -> int:
        return self.concrete if self.concrete is not None else int(np.clip((self.lo + self.hi) // 2, self.lo, self.hi))
    def is_concrete(self) -> bool: return self.concrete is not None
    def substitute(self, mapping: Dict[str, Any]) -> Any:
        return mapping[self.name] if self.name in mapping else SymInt(self.name, self.lo, self.hi, self.concrete)
    def narrow(self, lo: Optional[int] = None, hi: Optional[int] = None) -> "SymInt":
        return SymInt(self.name, max(self.lo, lo) if lo is not None else self.lo,
                      min(self.hi, hi) if hi is not None else self.hi, self.concrete)
    def __repr__(self) -> str:
        return f"SymInt({self.name}={self.concrete})" if self.concrete is not None else f"SymInt({self.name}:[{self.lo},{self.hi}])"


class SymStr(SymBase):
    """Symbolic string with length bounds."""
    def __init__(self, name: str, min_len: int = 0, max_len: int = 256, concrete: Optional[str] = None):
        self.name, self.min_len, self.max_len = name, min_len, max_len
        self.concrete, self.sym_id = concrete, self._fresh_id()
    def concretize(self) -> str:
        return self.concrete if self.concrete is not None else "a" * ((self.min_len + self.max_len) // 2)
    def is_concrete(self) -> bool: return self.concrete is not None
    def substitute(self, mapping: Dict[str, Any]) -> Any:
        return mapping[self.name] if self.name in mapping else SymStr(self.name, self.min_len, self.max_len, self.concrete)
    def __repr__(self) -> str:
        return f"SymStr({self.name}={self.concrete!r})" if self.concrete is not None else f"SymStr({self.name}:len[{self.min_len},{self.max_len}])"


class SymBool(SymBase):
    """Symbolic boolean."""
    def __init__(self, name: str, concrete: Optional[bool] = None):
        self.name, self.concrete, self.sym_id = name, concrete, self._fresh_id()
    def concretize(self) -> bool: return self.concrete if self.concrete is not None else True
    def is_concrete(self) -> bool: return self.concrete is not None
    def substitute(self, mapping: Dict[str, Any]) -> Any:
        return mapping[self.name] if self.name in mapping else SymBool(self.name, self.concrete)
    def __repr__(self) -> str:
        return f"SymBool({self.name}={self.concrete})" if self.concrete is not None else f"SymBool({self.name})"


class SymList(SymBase):
    """Symbolic list with element type and length range."""
    def __init__(self, name: str, elem_type: str = "int", min_len: int = 0,
                 max_len: int = 64, elements: Optional[List[Any]] = None):
        self.name, self.elem_type = name, elem_type
        self.min_len, self.max_len, self.elements = min_len, max_len, elements
        self.sym_id = self._fresh_id()
    def concretize(self) -> list:
        if self.elements is not None:
            return [e.concretize() if isinstance(e, SymBase) else e for e in self.elements]
        n = (self.min_len + self.max_len) // 2
        return list(range(n)) if self.elem_type == "int" else [f"s{i}" for i in range(n)]
    def is_concrete(self) -> bool:
        return self.elements is not None and all(not isinstance(e, SymBase) or e.is_concrete() for e in self.elements)
    def substitute(self, mapping: Dict[str, Any]) -> Any:
        if self.name in mapping: return mapping[self.name]
        elems = [e.substitute(mapping) if isinstance(e, SymBase) else e for e in self.elements] if self.elements else None
        return SymList(self.name, self.elem_type, self.min_len, self.max_len, elems)
    def __repr__(self) -> str: return f"SymList({self.name}:len[{self.min_len},{self.max_len}])"


class SymDict(SymBase):
    """Symbolic dictionary."""
    def __init__(self, name: str, entries: Optional[Dict[Any, Any]] = None):
        self.name, self.entries = name, entries if entries is not None else {}
        self.sym_id = self._fresh_id()
    def concretize(self) -> dict:
        return {(k.concretize() if isinstance(k, SymBase) else k): (v.concretize() if isinstance(v, SymBase) else v)
                for k, v in self.entries.items()}
    def is_concrete(self) -> bool:
        return all((not isinstance(k, SymBase) or k.is_concrete()) and (not isinstance(v, SymBase) or v.is_concrete())
                   for k, v in self.entries.items())
    def substitute(self, mapping: Dict[str, Any]) -> Any:
        if self.name in mapping: return mapping[self.name]
        return SymDict(self.name, {(k.substitute(mapping) if isinstance(k, SymBase) else k):
            (v.substitute(mapping) if isinstance(v, SymBase) else v) for k, v in self.entries.items()})
    def __repr__(self) -> str: return f"SymDict({self.name}:{len(self.entries)} entries)"


class SymNone(SymBase):
    """Symbolic None value."""
    def __init__(self, name: str = "_none"):
        self.name, self.sym_id = name, self._fresh_id()
    def concretize(self) -> None: return None
    def is_concrete(self) -> bool: return True
    def substitute(self, mapping: Dict[str, Any]) -> Any: return self
    def __repr__(self) -> str: return "SymNone"


# ---------------------------------------------------------------------------
# Constraint system
# ---------------------------------------------------------------------------

@dataclass
class PathConstraint:
    condition: ast.AST; is_true: bool; source_line: int
    def negate(self) -> "PathConstraint":
        return PathConstraint(self.condition, not self.is_true, self.source_line)
    def fingerprint(self) -> str:
        return hashlib.sha256(f"{ast.dump(self.condition)}:{self.is_true}:{self.source_line}".encode()).hexdigest()[:16]

@dataclass
class Interval:
    lo: int; hi: int
    def is_empty(self) -> bool: return self.lo > self.hi
    def intersect(self, other: "Interval") -> "Interval":
        return Interval(max(self.lo, other.lo), min(self.hi, other.hi))
    def union_hull(self, other: "Interval") -> "Interval":
        return Interval(min(self.lo, other.lo), max(self.hi, other.hi))
    def contains(self, value: int) -> bool: return self.lo <= value <= self.hi
    def midpoint(self) -> int: return (self.lo + self.hi) // 2
    def size(self) -> int: return 0 if self.is_empty() else self.hi - self.lo + 1


class IntervalSolver:
    """Solves interval constraints for integer variables."""
    def __init__(self) -> None:
        self._intervals: Dict[str, Interval] = {}
        self._excluded: Dict[str, Set[int]] = defaultdict(set)

    def add_constraint(self, var: str, op: str, value: int) -> None:
        cur = self._intervals.get(var, Interval(-(2 ** 31), (2 ** 31) - 1))
        if op == "<": cur = cur.intersect(Interval(cur.lo, value - 1))
        elif op == "<=": cur = cur.intersect(Interval(cur.lo, value))
        elif op == ">": cur = cur.intersect(Interval(value + 1, cur.hi))
        elif op == ">=": cur = cur.intersect(Interval(value, cur.hi))
        elif op == "==": cur = cur.intersect(Interval(value, value))
        elif op == "!=": self._excluded[var].add(value)
        self._intervals[var] = cur

    def is_satisfiable(self) -> bool:
        for var, iv in self._intervals.items():
            if iv.is_empty(): return False
            if iv.size() - len({v for v in self._excluded.get(var, set()) if iv.contains(v)}) <= 0: return False
        return True

    def solve(self) -> Dict[str, int]:
        if not self.is_satisfiable(): return {}
        result: Dict[str, int] = {}
        for var, iv in self._intervals.items():
            excluded = self._excluded.get(var, set())
            c = iv.midpoint()
            while c in excluded and c <= iv.hi: c += 1
            if c > iv.hi:
                c = iv.lo
                while c in excluded and c <= iv.hi: c += 1
            result[var] = c
        return result

    def copy(self) -> "IntervalSolver":
        new = IntervalSolver()
        new._intervals = {k: Interval(v.lo, v.hi) for k, v in self._intervals.items()}
        new._excluded = defaultdict(set, {k: set(v) for k, v in self._excluded.items()})
        return new


class ConstraintSet:
    """Collects path constraints and checks satisfiability via interval arithmetic."""
    def __init__(self) -> None:
        self.constraints: List[PathConstraint] = []
        self._solver = IntervalSolver()

    def add(self, constraint: PathConstraint) -> None:
        self.constraints.append(constraint)
        self._extract_and_add(constraint)

    def _extract_and_add(self, pc: PathConstraint) -> None:
        node = pc.condition
        if not isinstance(node, ast.Compare) or len(node.ops) != 1: return
        left, right, op_node = node.left, node.comparators[0], node.ops[0]
        op_map = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}
        op_str = op_map.get(type(op_node))
        if op_str is None: return
        if not pc.is_true:
            op_str = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "=="}.get(op_str, op_str)
        var_name, value = None, None
        if isinstance(left, ast.Name) and isinstance(right, ast.Constant) and isinstance(right.value, (int, float)):
            var_name, value = left.id, int(right.value)
        elif isinstance(right, ast.Name) and isinstance(left, ast.Constant) and isinstance(left.value, (int, float)):
            var_name, value = right.id, int(left.value)
            op_str = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}.get(op_str, op_str)
        if var_name is not None and value is not None:
            self._solver.add_constraint(var_name, op_str, value)

    def is_satisfiable(self) -> bool: return self._solver.is_satisfiable()
    def generate_assignment(self) -> Dict[str, int]: return self._solver.solve()
    def copy(self) -> "ConstraintSet":
        new = ConstraintSet(); new.constraints = list(self.constraints); new._solver = self._solver.copy()
        return new


# ---------------------------------------------------------------------------
# Execution paths and data classes
# ---------------------------------------------------------------------------

@dataclass
class ExecutionPath:
    constraints: List[PathConstraint] = field(default_factory=list)
    symbolic_state: Dict[str, Any] = field(default_factory=dict)
    return_value: Any = None; exception: Optional[str] = None
    statements_covered: Set[int] = field(default_factory=set)
    branch_decisions: List[Tuple[int, bool]] = field(default_factory=list)
    def path_id(self) -> str:
        return hashlib.sha256("|".join(f"{l}:{t}" for l, t in self.branch_decisions).encode()).hexdigest()[:16]

@dataclass
class ExecutionTree:
    paths: List[ExecutionPath] = field(default_factory=list)
    total_statements: int = 0; total_branches: int = 0
    bugs: List["BugReport"] = field(default_factory=list)
    def statement_coverage(self) -> float:
        if self.total_statements == 0: return 0.0
        covered: Set[int] = set()
        for p in self.paths: covered.update(p.statements_covered)
        return len(covered) / self.total_statements
    def branch_coverage(self) -> float:
        if self.total_branches == 0: return 0.0
        taken: Set[Tuple[int, bool]] = set()
        for p in self.paths: taken.update(p.branch_decisions)
        return len(taken) / (self.total_branches * 2)
    def path_count(self) -> int: return len(self.paths)
    def unique_path_count(self) -> int: return len({p.path_id() for p in self.paths})
    def bug_count(self) -> int: return len(self.bugs)

@dataclass
class BugReport:
    bug_type: str; location: int; description: str
    path_constraints: List[PathConstraint]; concrete_trigger: Dict[str, Any]

@dataclass
class TestCase:
    inputs: Dict[str, Any]; expected_return: Any; expected_exception: Optional[str]
    path_id: str; covered_lines: Set[int]

@dataclass
class _ExecState:
    local_vars: Dict[str, Any] = field(default_factory=dict)
    constraints: ConstraintSet = field(default_factory=ConstraintSet)
    stmts_covered: Set[int] = field(default_factory=set)
    branch_decisions: List[Tuple[int, bool]] = field(default_factory=list)
    depth: int = 0; returned: bool = False; return_value: Any = None; exception: Optional[str] = None
    def copy(self) -> "_ExecState":
        n = _ExecState(copy.deepcopy(self.local_vars), self.constraints.copy(),
                       set(self.stmts_covered), list(self.branch_decisions),
                       self.depth, self.returned, copy.deepcopy(self.return_value), self.exception)
        return n


# ---------------------------------------------------------------------------
# Main symbolic executor
# ---------------------------------------------------------------------------

class PythonSymbolicExecutor:
    def __init__(self) -> None:
        self._tree: Optional[ast.Module] = None
        self._func_defs: Dict[str, ast.FunctionDef] = {}
        self._all_statements: Set[int] = set()
        self._all_branches: Set[int] = set()

    def execute(self, source_code: str, function_name: str,
                max_paths: int = 100, max_depth: int = 50) -> ExecutionTree:
        source_code = textwrap.dedent(source_code)
        self._tree = ast.parse(source_code)
        self._func_defs.clear(); self._all_statements.clear(); self._all_branches.clear()
        for node in ast.walk(self._tree):
            if isinstance(node, ast.FunctionDef): self._func_defs[node.name] = node
            if hasattr(node, "lineno"): self._all_statements.add(node.lineno)
            if isinstance(node, (ast.If, ast.While)): self._all_branches.add(node.lineno)
        if function_name not in self._func_defs:
            raise ValueError(f"Function {function_name!r} not found")
        func = self._func_defs[function_name]
        init = self._make_initial_state(func)
        dfs = self._explore_paths(func, init, max_paths, max_depth)
        bfs = self._explore_bfs(func, init, max_paths, max_depth)
        seen: Set[str] = set(); merged: List[ExecutionPath] = []
        for p in dfs + bfs:
            pid = p.path_id()
            if pid not in seen: seen.add(pid); merged.append(p)
        tree = ExecutionTree(merged, len(self._all_statements), len(self._all_branches))
        for path in merged: tree.bugs.extend(self._detect_bugs(path))
        self._compute_coverage(tree)
        return tree

    def _make_initial_state(self, func: ast.FunctionDef) -> _ExecState:
        state = _ExecState()
        for arg in func.args.args:
            name = arg.arg
            ann = arg.annotation.id if arg.annotation and isinstance(arg.annotation, ast.Name) else None
            type_map = {"str": SymStr(name), "bool": SymBool(name), "list": SymList(name)}
            state.local_vars[name] = type_map.get(ann, SymInt(name))
        return state

    def _explore_paths(self, func: ast.FunctionDef, init: _ExecState,
                       max_paths: int, max_depth: int) -> List[ExecutionPath]:
        results: List[ExecutionPath] = []
        stack: List[Tuple[List[ast.stmt], _ExecState]] = [(list(func.body), init.copy())]
        while stack and len(results) < max_paths:
            stmts, state = stack.pop()
            if state.depth > max_depth or state.returned or state.exception or not stmts:
                results.append(self._state_to_path(state)); continue
            for fs in self._execute_statement(stmts[0], state):
                stack.append((stmts[1:], fs))
        return results

    def _explore_bfs(self, func: ast.FunctionDef, init: _ExecState,
                     max_paths: int, max_depth: int) -> List[ExecutionPath]:
        results: List[ExecutionPath] = []
        q: deque = deque([(list(func.body), init.copy())])
        while q and len(results) < max_paths:
            stmts, state = q.popleft()
            if state.depth > max_depth or state.returned or state.exception or not stmts:
                results.append(self._state_to_path(state)); continue
            for fs in self._execute_statement(stmts[0], state):
                q.append((stmts[1:], fs))
        return results

    def _state_to_path(self, s: _ExecState) -> ExecutionPath:
        return ExecutionPath(list(s.constraints.constraints), dict(s.local_vars),
                             s.return_value, s.exception, set(s.stmts_covered), list(s.branch_decisions))

    def _execute_statement(self, stmt: ast.stmt, state: _ExecState) -> List[_ExecState]:
        state.depth += 1; state.stmts_covered.add(getattr(stmt, "lineno", 0))
        if isinstance(stmt, ast.Assign): return self._exec_assign(stmt, state)
        if isinstance(stmt, ast.AugAssign): return self._exec_aug_assign(stmt, state)
        if isinstance(stmt, ast.If): return self._exec_if(stmt, state)
        if isinstance(stmt, ast.While): return self._exec_while(stmt, state)
        if isinstance(stmt, ast.For): return self._exec_for(stmt, state)
        if isinstance(stmt, ast.Return): return self._exec_return(stmt, state)
        if isinstance(stmt, ast.Assert): return self._exec_assert(stmt, state)
        if isinstance(stmt, ast.Raise): return self._exec_raise(stmt, state)
        if isinstance(stmt, ast.Try): return self._exec_try(stmt, state)
        if isinstance(stmt, ast.FunctionDef): self._func_defs[stmt.name] = stmt
        elif isinstance(stmt, ast.Expr): self._evaluate_expr(stmt.value, state)
        return [state]

    def _exec_assign(self, stmt: ast.Assign, state: _ExecState) -> List[_ExecState]:
        value = self._evaluate_expr(stmt.value, state)
        for target in stmt.targets:
            if isinstance(target, ast.Name):
                state.local_vars[target.id] = value
            elif isinstance(target, ast.Subscript):
                coll = self._evaluate_expr(target.value, state)
                idx = self._evaluate_expr(target.slice, state)
                ci = idx.concretize() if isinstance(idx, SymBase) else idx
                if isinstance(coll, SymList) and coll.elements and isinstance(ci, int) and 0 <= ci < len(coll.elements):
                    coll.elements[ci] = value
                elif isinstance(coll, (SymDict, dict)):
                    (coll.entries if isinstance(coll, SymDict) else coll)[ci] = value
                elif isinstance(coll, list) and isinstance(ci, int) and 0 <= ci < len(coll):
                    coll[ci] = value
            elif isinstance(target, ast.Tuple) and isinstance(value, (list, tuple)):
                for i, elt in enumerate(target.elts):
                    if isinstance(elt, ast.Name) and i < len(value): state.local_vars[elt.id] = value[i]
        return [state]

    def _exec_aug_assign(self, stmt: ast.AugAssign, state: _ExecState) -> List[_ExecState]:
        if isinstance(stmt.target, ast.Name):
            state.local_vars[stmt.target.id] = self._apply_binop(
                stmt.op, state.local_vars.get(stmt.target.id, 0), self._evaluate_expr(stmt.value, state))
        return [state]

    def _exec_if(self, stmt: ast.If, state: _ExecState) -> List[_ExecState]:
        results: List[_ExecState] = []
        for forked, taken in self._fork_on_condition(stmt.test, self._evaluate_expr(stmt.test, state), state):
            forked.branch_decisions.append((stmt.lineno, taken))
            for s in (stmt.body if taken else stmt.orelse):
                sub = self._execute_statement(s, forked)
                if len(sub) != 1: results.extend(sub); forked = None; break
                forked = sub[0]
                if forked.returned or forked.exception: break
            if forked is not None: results.append(forked)
        return results or [state]

    def _exec_while(self, stmt: ast.While, state: _ExecState) -> List[_ExecState]:
        results: List[_ExecState] = []; current = state
        for _ in range(10):
            forks = self._fork_on_condition(stmt.test, self._evaluate_expr(stmt.test, current), current)
            entered = False
            for forked, taken in forks:
                forked.branch_decisions.append((stmt.lineno, taken))
                if not taken:
                    for s in stmt.orelse: self._execute_statement(s, forked)
                    results.append(forked)
                else:
                    entered = True
                    for s in stmt.body:
                        forked = self._execute_statement(s, forked)[0]
                        if forked.returned or forked.exception: break
                    current = forked
            if not entered or current.returned or current.exception:
                if current.returned or current.exception: results.append(current)
                break
        else: results.append(current)
        return results or [state]

    def _exec_for(self, stmt: ast.For, state: _ExecState) -> List[_ExecState]:
        it = self._evaluate_expr(stmt.iter, state)
        items = it.concretize() if isinstance(it, SymList) else list(it) if isinstance(it, (list, tuple, range)) else []
        tgt = stmt.target.id if isinstance(stmt.target, ast.Name) else "_"
        cur = state
        for item in items[:20]:
            cur.local_vars[tgt] = item; cur.stmts_covered.add(stmt.lineno)
            for s in stmt.body:
                cur = self._execute_statement(s, cur)[0]
                if cur.returned or cur.exception: break
            if cur.returned or cur.exception: break
        for s in stmt.orelse: self._execute_statement(s, cur)
        return [cur]

    def _exec_return(self, stmt: ast.Return, state: _ExecState) -> List[_ExecState]:
        state.return_value = self._evaluate_expr(stmt.value, state) if stmt.value else SymNone()
        state.returned = True; return [state]

    def _exec_assert(self, stmt: ast.Assert, state: _ExecState) -> List[_ExecState]:
        if self._check_assertion(stmt.test, state) is False:
            state.exception = f"AssertionError at line {stmt.lineno}"
        return [state]

    def _exec_raise(self, stmt: ast.Raise, state: _ExecState) -> List[_ExecState]:
        exc = "Exception"
        if stmt.exc:
            if isinstance(stmt.exc, ast.Call) and isinstance(stmt.exc.func, ast.Name): exc = stmt.exc.func.id
            elif isinstance(stmt.exc, ast.Name): exc = stmt.exc.id
        state.exception = f"{exc} at line {stmt.lineno}"; return [state]

    def _exec_try(self, stmt: ast.Try, state: _ExecState) -> List[_ExecState]:
        results: List[_ExecState] = []; normal = state.copy()
        for s in stmt.body:
            normal = self._execute_statement(s, normal)[0]
            if normal.exception: break
        if normal.exception:
            for handler in stmt.handlers:
                hs = state.copy(); hs.exception = None
                if handler.name: hs.local_vars[handler.name] = normal.exception
                for s in handler.body: hs = self._execute_statement(s, hs)[0]
                results.append(hs); break
            else: results.append(normal)
        else:
            for s in stmt.orelse: normal = self._execute_statement(s, normal)[0]
            results.append(normal)
        for s in getattr(stmt, "finalbody", []):
            for r in results: self._execute_statement(s, r)
        return results or [state]

    # -- expression evaluation ----------------------------------------------

    def _evaluate_expr(self, expr: ast.expr, state: _ExecState) -> Any:
        if isinstance(expr, ast.Constant): return expr.value
        if isinstance(expr, ast.Name): return state.local_vars.get(expr.id, SymNone(expr.id))
        if isinstance(expr, ast.BinOp):
            return self._apply_binop(expr.op, self._evaluate_expr(expr.left, state), self._evaluate_expr(expr.right, state))
        if isinstance(expr, ast.UnaryOp): return self._apply_unaryop(expr.op, self._evaluate_expr(expr.operand, state))
        if isinstance(expr, ast.BoolOp): return self._eval_boolop(expr, state)
        if isinstance(expr, ast.Compare): return self._eval_compare(expr, state)
        if isinstance(expr, ast.IfExp):
            return self._evaluate_expr(expr.body if self._is_truthy(self._evaluate_expr(expr.test, state)) else expr.orelse, state)
        if isinstance(expr, ast.Call): return self._eval_call(expr, state)
        if isinstance(expr, ast.Subscript): return self._eval_subscript(expr, state)
        if isinstance(expr, ast.Attribute): return self._eval_attribute(expr, state)
        if isinstance(expr, ast.List): return [self._evaluate_expr(e, state) for e in expr.elts]
        if isinstance(expr, ast.Tuple): return tuple(self._evaluate_expr(e, state) for e in expr.elts)
        if isinstance(expr, ast.Dict):
            return dict(zip([self._evaluate_expr(k, state) for k in expr.keys if k],
                            [self._evaluate_expr(v, state) for v in expr.values]))
        if isinstance(expr, ast.Set): return {self._evaluate_expr(e, state) for e in expr.elts}
        if isinstance(expr, ast.ListComp): return self._eval_listcomp(expr, state)
        if isinstance(expr, ast.JoinedStr): return SymStr("_fstr")
        return SymNone()

    def _apply_binop(self, op: ast.operator, left: Any, right: Any) -> Any:
        lc = left.concretize() if isinstance(left, SymBase) else left
        rc = right.concretize() if isinstance(right, SymBase) else right
        try:
            if isinstance(op, ast.Add): return lc + rc
            if isinstance(op, ast.Sub): return lc - rc
            if isinstance(op, ast.Mult): return lc * rc
            if isinstance(op, ast.Div): return math.inf if rc == 0 else lc / rc
            if isinstance(op, ast.FloorDiv): return math.inf if rc == 0 else lc // rc
            if isinstance(op, ast.Mod): return 0 if rc == 0 else lc % rc
            if isinstance(op, ast.Pow): return lc ** min(rc, 64)
            if isinstance(op, ast.BitAnd): return int(lc) & int(rc)
            if isinstance(op, ast.BitOr): return int(lc) | int(rc)
            if isinstance(op, ast.BitXor): return int(lc) ^ int(rc)
            if isinstance(op, ast.LShift): return int(lc) << min(int(rc), 63)
            if isinstance(op, ast.RShift): return int(lc) >> min(int(rc), 63)
        except (TypeError, OverflowError, ValueError): pass
        return 0

    def _apply_unaryop(self, op: ast.unaryop, val: Any) -> Any:
        cv = val.concretize() if isinstance(val, SymBase) else val
        if isinstance(op, ast.UAdd): return +cv
        if isinstance(op, ast.USub): return -cv
        if isinstance(op, ast.Not): return not cv
        if isinstance(op, ast.Invert): return ~int(cv)
        return cv

    def _eval_boolop(self, expr: ast.BoolOp, state: _ExecState) -> Any:
        vals = [self._evaluate_expr(v, state) for v in expr.values]
        if isinstance(expr.op, ast.And):
            r = vals[0]
            for v in vals[1:]:
                if not self._is_truthy(r): return r
                r = v
            return r
        r = vals[0]
        for v in vals[1:]:
            if self._is_truthy(r): return r
            r = v
        return r

    def _eval_compare(self, expr: ast.Compare, state: _ExecState) -> Any:
        left = self._evaluate_expr(expr.left, state)
        for op, cn in zip(expr.ops, expr.comparators):
            right = self._evaluate_expr(cn, state)
            lc = left.concretize() if isinstance(left, SymBase) else left
            rc = right.concretize() if isinstance(right, SymBase) else right
            try:
                if isinstance(op, ast.Eq): res = lc == rc
                elif isinstance(op, ast.NotEq): res = lc != rc
                elif isinstance(op, ast.Lt): res = lc < rc
                elif isinstance(op, ast.LtE): res = lc <= rc
                elif isinstance(op, ast.Gt): res = lc > rc
                elif isinstance(op, ast.GtE): res = lc >= rc
                elif isinstance(op, ast.Is): res = lc is rc
                elif isinstance(op, ast.IsNot): res = lc is not rc
                elif isinstance(op, ast.In): res = lc in rc if hasattr(rc, "__contains__") else False
                elif isinstance(op, ast.NotIn): res = lc not in rc if hasattr(rc, "__contains__") else True
                else: res = False
            except (TypeError, ValueError): res = False
            if not res: return False
            left = right
        return True

    def _eval_call(self, expr: ast.Call, state: _ExecState) -> Any:
        if isinstance(expr.func, ast.Name):
            fn = expr.func.id; args = [self._evaluate_expr(a, state) for a in expr.args]
            if fn == "len" and args:
                a = args[0]
                if isinstance(a, (SymList, SymStr)): return (a.min_len + a.max_len) // 2
                return len(a) if isinstance(a, (list, tuple, str, dict)) else 0
            if fn == "abs" and args:
                v = args[0].concretize() if isinstance(args[0], SymBase) else args[0]
                return abs(v) if isinstance(v, (int, float)) else 0
            if fn == "int" and args:
                try: return int(args[0].concretize() if isinstance(args[0], SymBase) else args[0])
                except (TypeError, ValueError): return 0
            if fn == "range":
                ca = [int(a.concretize() if isinstance(a, SymBase) else a) if isinstance(a.concretize() if isinstance(a, SymBase) else a, (int, float)) else 0 for a in args]
                return range(*ca) if ca else range(0)
            if fn in ("min", "max") and args:
                conc = [a.concretize() if isinstance(a, SymBase) else a for a in args]
                try: return min(conc) if fn == "min" else max(conc)
                except (TypeError, ValueError): return 0
            if fn == "sum" and args and isinstance(args[0], (list, tuple)): return sum(args[0])
            if fn == "sorted" and args and isinstance(args[0], (list, tuple)): return sorted(args[0])
            if fn == "print": return SymNone()
            if fn == "isinstance": return SymBool("_isinstance")
            if fn in self._func_defs: return self._call_function(fn, args, state)
            return SymInt(f"_ret_{fn}")
        if isinstance(expr.func, ast.Attribute):
            obj = self._evaluate_expr(expr.func.value, state)
            return self._eval_method_call(obj, expr.func.attr, [self._evaluate_expr(a, state) for a in expr.args])
        return SymNone()

    def _call_function(self, name: str, args: List[Any], caller: _ExecState) -> Any:
        func = self._func_defs[name]
        child = _ExecState(); child.constraints, child.depth = caller.constraints.copy(), caller.depth
        for i, p in enumerate(func.args.args):
            child.local_vars[p.arg] = args[i] if i < len(args) else SymNone(p.arg)
        for s in func.body:
            child = self._execute_statement(s, child)[0]
            if child.returned or child.exception: break
        caller.stmts_covered.update(child.stmts_covered)
        caller.branch_decisions.extend(child.branch_decisions)
        if child.exception: caller.exception = child.exception
        return child.return_value if child.returned else SymNone()

    def _eval_method_call(self, obj: Any, method: str, args: List[Any]) -> Any:
        c = obj.concretize() if isinstance(obj, SymBase) else obj
        if isinstance(c, list):
            if method == "append" and args: c.append(args[0]); return SymNone()
            if method == "pop": return c.pop() if c else SymNone()
            if method == "sort": c.sort(); return SymNone()
            if method == "reverse": c.reverse(); return SymNone()
            if method == "index" and args:
                try: return c.index(args[0])
                except ValueError: return -1
        if isinstance(c, str):
            ca = lambda i: args[i].concretize() if isinstance(args[i], SymBase) else args[i]
            if method == "split": return c.split()
            if method == "strip": return c.strip()
            if method == "lower": return c.lower()
            if method == "upper": return c.upper()
            if method == "startswith" and args: return c.startswith(ca(0))
            if method == "endswith" and args: return c.endswith(ca(0))
            if method == "find" and args: return c.find(ca(0))
            if method == "replace" and len(args) >= 2: return c.replace(ca(0), ca(1))
        if isinstance(c, dict) or isinstance(obj, SymDict):
            d = obj.entries if isinstance(obj, SymDict) else c
            ca0 = args[0].concretize() if args and isinstance(args[0], SymBase) else (args[0] if args else None)
            if method == "get" and args: return d.get(ca0, args[1] if len(args) > 1 else None)
            if method == "keys": return list(d.keys())
            if method == "values": return list(d.values())
            if method == "items": return list(d.items())
        return SymNone()

    def _eval_subscript(self, expr: ast.Subscript, state: _ExecState) -> Any:
        obj = self._evaluate_expr(expr.value, state)
        idx = self._evaluate_expr(expr.slice, state)
        ci = idx.concretize() if isinstance(idx, SymBase) else idx
        if isinstance(obj, SymList):
            e = obj.concretize()
            return e[ci] if isinstance(ci, int) and 0 <= ci < len(e) else SymNone()
        if isinstance(obj, SymDict): return obj.entries.get(ci, SymNone())
        if isinstance(obj, (list, tuple)):
            return obj[ci] if isinstance(ci, int) and 0 <= ci < len(obj) else SymNone()
        if isinstance(obj, dict): return obj.get(ci, SymNone())
        if isinstance(obj, str): return obj[ci] if isinstance(ci, int) and 0 <= ci < len(obj) else ""
        return SymNone()

    def _eval_attribute(self, expr: ast.Attribute, state: _ExecState) -> Any:
        obj = self._evaluate_expr(expr.value, state)
        if isinstance(obj, SymList) and expr.attr == "elements": return obj.elements or []
        if isinstance(obj, SymDict) and expr.attr == "entries": return obj.entries
        return SymNone()

    def _eval_listcomp(self, expr: ast.ListComp, state: _ExecState) -> list:
        if not expr.generators: return []
        gen = expr.generators[0]
        it = self._evaluate_expr(gen.iter, state)
        if isinstance(it, SymList): it = it.concretize()
        if not isinstance(it, (list, tuple, range)): return []
        tgt = gen.target.id if isinstance(gen.target, ast.Name) else "_"
        child, result = state.copy(), []
        for item in list(it)[:50]:
            child.local_vars[tgt] = item
            if all(self._is_truthy(self._evaluate_expr(c, child)) for c in gen.ifs):
                result.append(self._evaluate_expr(expr.elt, child))
        return result

    # -- forking and assertions ---------------------------------------------

    def _fork_on_condition(self, test: ast.expr, cond_val: Any,
                           state: _ExecState) -> List[Tuple[_ExecState, bool]]:
        if isinstance(cond_val, bool):
            s = state.copy(); s.constraints.add(PathConstraint(test, cond_val, getattr(test, "lineno", 0)))
            return [(s, cond_val)]
        if isinstance(cond_val, SymBool) and cond_val.is_concrete():
            s = state.copy(); s.constraints.add(PathConstraint(test, cond_val.concrete, getattr(test, "lineno", 0)))
            return [(s, cond_val.concrete)]
        line = getattr(test, "lineno", 0)
        ts, fs = state.copy(), state.copy()
        ts.constraints.add(PathConstraint(test, True, line))
        fs.constraints.add(PathConstraint(test, False, line))
        self._narrow_state(test, ts, True); self._narrow_state(test, fs, False)
        forks: List[Tuple[_ExecState, bool]] = []
        if ts.constraints.is_satisfiable(): forks.append((ts, True))
        if fs.constraints.is_satisfiable(): forks.append((fs, False))
        return forks or [(state.copy(), True)]

    def _narrow_state(self, test: ast.expr, state: _ExecState, taken: bool) -> None:
        if not isinstance(test, ast.Compare) or len(test.ops) != 1: return
        left, right, op = test.left, test.comparators[0], test.ops[0]
        if not (isinstance(left, ast.Name) and isinstance(right, ast.Constant) and isinstance(right.value, (int, float))):
            return
        var, ival = left.id, int(right.value)
        sv = state.local_vars.get(var)
        if not isinstance(sv, SymInt): return
        if taken:
            if isinstance(op, ast.Lt): state.local_vars[var] = sv.narrow(hi=ival - 1)
            elif isinstance(op, ast.LtE): state.local_vars[var] = sv.narrow(hi=ival)
            elif isinstance(op, ast.Gt): state.local_vars[var] = sv.narrow(lo=ival + 1)
            elif isinstance(op, ast.GtE): state.local_vars[var] = sv.narrow(lo=ival)
            elif isinstance(op, ast.Eq): state.local_vars[var] = SymInt(var, ival, ival, ival)
        else:
            if isinstance(op, ast.Lt): state.local_vars[var] = sv.narrow(lo=ival)
            elif isinstance(op, ast.LtE): state.local_vars[var] = sv.narrow(lo=ival + 1)
            elif isinstance(op, ast.Gt): state.local_vars[var] = sv.narrow(hi=ival)
            elif isinstance(op, ast.GtE): state.local_vars[var] = sv.narrow(hi=ival - 1)

    def _check_assertion(self, test_expr: ast.expr, state: _ExecState) -> Any:
        return self._is_truthy(self._evaluate_expr(test_expr, state))

    def _is_truthy(self, val: Any) -> bool:
        if isinstance(val, SymBool): return val.concretize()
        if isinstance(val, SymInt): return val.concretize() != 0
        if isinstance(val, SymStr): return len(val.concretize()) > 0
        if isinstance(val, SymList): return val.min_len > 0 or (val.elements is not None and len(val.elements) > 0)
        if isinstance(val, SymNone): return False
        if isinstance(val, SymDict): return len(val.entries) > 0
        return bool(val)

    def _detect_bugs(self, path: ExecutionPath) -> List[BugReport]:
        bugs: List[BugReport] = []
        cset = ConstraintSet()
        for c in path.constraints: cset.add(c)
        concrete = cset.generate_assignment()
        for var, val in path.symbolic_state.items():
            if isinstance(val, SymInt) and val.lo <= 0 <= val.hi:
                bugs.append(BugReport("potential_division_by_zero", 0,
                    f"Variable {var} range includes zero: [{val.lo},{val.hi}]", list(path.constraints), concrete))
            if isinstance(val, SymNone):
                bugs.append(BugReport("potential_none_dereference", 0,
                    f"Variable {var} may be None", list(path.constraints), concrete))
            if isinstance(val, SymList) and val.elements is not None:
                for sv in path.symbolic_state.values():
                    if isinstance(sv, SymInt) and sv.hi >= len(val.elements):
                        bugs.append(BugReport("potential_index_out_of_bounds", 0,
                            f"Index {sv.name} may exceed list length {len(val.elements)}",
                            list(path.constraints), concrete))
                        break
        if path.exception:
            bugs.append(BugReport("exception", 0, path.exception, list(path.constraints), concrete))
        return bugs

    def _compute_coverage(self, tree: ExecutionTree) -> None:
        tree.total_statements = max(tree.total_statements, len(self._all_statements))
        tree.total_branches = max(tree.total_branches, len(self._all_branches))


# ---------------------------------------------------------------------------
# Test case generator
# ---------------------------------------------------------------------------

class TestCaseGenerator:
    """Generates concrete test cases from an execution tree."""
    def generate(self, tree: ExecutionTree) -> List[TestCase]:
        cases: List[TestCase] = []; seen: Set[str] = set()
        for path in tree.paths:
            tc = self._from_path(path)
            if tc is not None and tc.path_id not in seen: seen.add(tc.path_id); cases.append(tc)
        return cases

    def _from_path(self, path: ExecutionPath) -> Optional[TestCase]:
        cset = ConstraintSet()
        for c in path.constraints: cset.add(c)
        concrete = cset.generate_assignment()
        inputs: Dict[str, Any] = {}
        for var, val in path.symbolic_state.items():
            inputs[var] = concrete[var] if var in concrete else (val.concretize() if isinstance(val, SymBase) else val)
        ret = path.return_value
        if isinstance(ret, SymBase): ret = ret.concretize()
        return TestCase(inputs, ret, path.exception, path.path_id(), set(path.statements_covered))
