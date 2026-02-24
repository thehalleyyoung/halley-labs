"""Async Python code analyzer.

Detects common async/await bugs including missing awaits, blocking calls
in async context, potential deadlocks, shared mutable state issues,
fire-and-forget tasks, and performance anti-patterns.
"""
from __future__ import annotations

import ast
import copy
import textwrap
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import numpy as np

# ── Enums ──────────────────────────────────────────────────────────────────

class BugSeverity(Enum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

# ── Bug report data-classes ────────────────────────────────────────────────

@dataclass
class AsyncBug:
    severity: BugSeverity
    description: str
    line: int
    col: int = 0
    fix_suggestion: str = ""
    category: str = "general"
    def __str__(self) -> str:
        return f"[{self.severity.name}] line {self.line}: {self.description}"
@dataclass
class EventLoopIssue(AsyncBug):
    category: str = "event_loop"
    blocking_call: str = ""

@dataclass
class MissingAwait(AsyncBug):
    category: str = "missing_await"
    target_func: str = ""

@dataclass
class DeadlockRisk(AsyncBug):
    category: str = "deadlock"
    lock_order: List[str] = field(default_factory=list)

@dataclass
class ConcurrencyBug(AsyncBug):
    category: str = "concurrency"
    shared_var: str = ""
    accessing_functions: List[str] = field(default_factory=list)

@dataclass
class TaskIssue(AsyncBug):
    category: str = "task_lifecycle"
    task_call_name: str = ""

@dataclass
class PerformanceIssue(AsyncBug):
    category: str = "performance"
    awaited_calls: List[str] = field(default_factory=list)

# ── Info containers ────────────────────────────────────────────────────────

@dataclass
class AsyncFunctionInfo:
    name: str
    lineno: int
    end_lineno: int = 0
    awaits: List[int] = field(default_factory=list)
    async_fors: List[int] = field(default_factory=list)
    async_withs: List[int] = field(default_factory=list)
    called_funcs: List[str] = field(default_factory=list)
    modified_globals: Set[str] = field(default_factory=set)
    read_globals: Set[str] = field(default_factory=set)
    lock_acquisitions: List[str] = field(default_factory=list)
    parent_class: Optional[str] = None

@dataclass
class AsyncReport:
    async_functions: List[AsyncFunctionInfo] = field(default_factory=list)
    potential_bugs: List[AsyncBug] = field(default_factory=list)
    performance_issues: List[PerformanceIssue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    severity_summary: Dict[str, int] = field(default_factory=dict)
    def total_issues(self) -> int:
        return len(self.potential_bugs) + len(self.performance_issues)
    def compute_severity_summary(self) -> None:
        counts: Dict[str, int] = {s.name: 0 for s in BugSeverity}
        for bug in self.potential_bugs:
            counts[bug.severity.name] += 1
        for perf in self.performance_issues:
            counts[perf.severity.name] += 1
        self.severity_summary = counts
    def severity_vector(self) -> np.ndarray:
        self.compute_severity_summary()
        return np.array([self.severity_summary.get("LOW", 0),
                         self.severity_summary.get("MEDIUM", 0),
                         self.severity_summary.get("HIGH", 0),
                         self.severity_summary.get("CRITICAL", 0)], dtype=np.int64)
    def weighted_score(self) -> float:
        weights = np.array([1.0, 3.0, 7.0, 15.0])
        return float(np.dot(self.severity_vector(), weights))

# ── Constants ──────────────────────────────────────────────────────────────

BLOCKING_CALLS: Set[str] = {
    "time.sleep", "open", "input", "os.read", "os.write", "os.popen",
    "os.system", "subprocess.run", "subprocess.call",
    "subprocess.check_output", "subprocess.check_call", "subprocess.Popen",
    "socket.socket.recv", "socket.socket.send", "socket.socket.connect",
    "socket.socket.accept", "socket.create_connection",
    "requests.get", "requests.post", "requests.put", "requests.delete",
    "requests.head", "requests.patch", "urllib.request.urlopen",
    "http.client.HTTPConnection", "sqlite3.connect",
    "json.load", "json.dump", "pickle.load", "pickle.dump",
    "shutil.copy", "shutil.move", "glob.glob",
    "pathlib.Path.read_text", "pathlib.Path.write_text",
    "pathlib.Path.read_bytes", "pathlib.Path.write_bytes",
}

BLOCKING_SIMPLE_NAMES: Set[str] = {
    "sleep", "open", "input", "recv", "send", "connect", "accept",
}

NESTED_LOOP_CALLS: Set[str] = {
    "asyncio.run", "asyncio.get_event_loop",
    "loop.run_until_complete", "loop.run_forever",
}

# ── AST helpers ────────────────────────────────────────────────────────────

def _resolve_attr(node: ast.expr) -> str:
    parts: List[str] = []
    cur: ast.expr = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    parts.reverse()
    return ".".join(parts)
def _resolve_call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return _resolve_attr(node.func)
    return ""
def _build_parent_map(tree: ast.AST) -> Dict[int, ast.AST]:
    parent: Dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[id(child)] = node
    return parent

# ── 1. Async function detection ───────────────────────────────────────────

class _AsyncFuncCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.functions: List[AsyncFunctionInfo] = []
        self._current: Optional[AsyncFunctionInfo] = None
        self._class_stack: List[str] = []
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        info = AsyncFunctionInfo(
            name=node.name, lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", node.lineno),
            parent_class=self._class_stack[-1] if self._class_stack else None,
        )
        prev = self._current
        self._current = info
        for child in ast.walk(node):
            if isinstance(child, ast.Await):
                info.awaits.append(child.lineno)
            elif isinstance(child, ast.AsyncFor):
                info.async_fors.append(child.lineno)
            elif isinstance(child, ast.AsyncWith):
                info.async_withs.append(child.lineno)
            elif isinstance(child, ast.Call):
                cname = _resolve_call_name(child)
                if cname:
                    info.called_funcs.append(cname)
            elif isinstance(child, ast.Global):
                for gname in child.names:
                    info.modified_globals.add(gname)
        self.functions.append(info)
        self.generic_visit(node)
        self._current = prev
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        prev = self._current
        self._current = None
        self.generic_visit(node)
        self._current = prev
def collect_async_functions(tree: ast.AST) -> List[AsyncFunctionInfo]:
    collector = _AsyncFuncCollector()
    collector.visit(tree)
    return collector.functions

# ── 2. Event loop analysis ────────────────────────────────────────────────

class _EventLoopChecker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.issues: List[EventLoopIssue] = []
        self._in_async = False
        self._async_depth = 0
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._in_async = True
        self._async_depth += 1
        self.generic_visit(node)
        self._async_depth -= 1
        if self._async_depth == 0:
            self._in_async = False
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        prev = self._in_async
        self._in_async = False
        self.generic_visit(node)
        self._in_async = prev
    def visit_Call(self, node: ast.Call) -> None:
        if not self._in_async:
            self.generic_visit(node)
            return
        call_name = _resolve_call_name(node)
        if not call_name:
            self.generic_visit(node)
            return
        if call_name in NESTED_LOOP_CALLS:
            self.issues.append(EventLoopIssue(
                severity=BugSeverity.CRITICAL,
                description=f"Nested event loop via {call_name}() inside async function",
                line=node.lineno, col=node.col_offset,
                fix_suggestion="Use 'await' instead of running a new event loop.",
                blocking_call=call_name,
            ))
        if call_name in BLOCKING_CALLS or call_name in BLOCKING_SIMPLE_NAMES:
            sev = BugSeverity.LOW if call_name == "print" else BugSeverity.HIGH
            if call_name in ("time.sleep", "sleep"):
                suggestion = "Use 'await asyncio.sleep(...)' instead."
            elif call_name == "open":
                suggestion = "Use aiofiles.open() or run_in_executor."
            elif call_name.startswith("requests."):
                suggestion = "Use aiohttp or httpx for async HTTP."
            else:
                suggestion = f"Replace {call_name} with its async equivalent."
            self.issues.append(EventLoopIssue(
                severity=sev, description=f"Blocking call {call_name}() inside async function",
                line=node.lineno, col=node.col_offset,
                fix_suggestion=suggestion, blocking_call=call_name,
            ))
        self.generic_visit(node)
def check_event_loop(tree: ast.AST) -> List[EventLoopIssue]:
    checker = _EventLoopChecker()
    checker.visit(tree)
    return checker.issues

# ── 3. Missing await detection ────────────────────────────────────────────

class _MissingAwaitChecker(ast.NodeVisitor):
    def __init__(self, async_names: Set[str]) -> None:
        self.async_names = async_names
        self.issues: List[MissingAwait] = []
        self._in_async = False
        self._await_targets: Set[int] = set()
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        prev = self._in_async
        self._in_async = True
        for child in ast.walk(node):
            if isinstance(child, ast.Await) and isinstance(child.value, ast.Call):
                self._await_targets.add(id(child.value))
        self.generic_visit(node)
        self._in_async = prev
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        prev = self._in_async
        self._in_async = False
        self.generic_visit(node)
        self._in_async = prev
    def visit_Call(self, node: ast.Call) -> None:
        if not self._in_async:
            self.generic_visit(node)
            return
        call_name = _resolve_call_name(node)
        if call_name and call_name in self.async_names:
            if id(node) not in self._await_targets:
                self.issues.append(MissingAwait(
                    severity=BugSeverity.HIGH,
                    description=f"Async function '{call_name}' called without await",
                    line=node.lineno, col=node.col_offset,
                    fix_suggestion=f"Add 'await' before {call_name}(...).",
                    target_func=call_name,
                ))
        self.generic_visit(node)
def check_missing_awaits(tree: ast.AST) -> List[MissingAwait]:
    async_funcs = collect_async_functions(tree)
    async_names = {f.name for f in async_funcs}
    checker = _MissingAwaitChecker(async_names)
    checker.visit(tree)
    return checker.issues

# ── 4. Deadlock detection ─────────────────────────────────────────────────

class _LockOrderCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.orders: Dict[str, List[str]] = defaultdict(list)
        self._current_func: Optional[str] = None
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        prev = self._current_func
        self._current_func = node.name
        self.generic_visit(node)
        self._current_func = prev
    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        if self._current_func is None:
            self.generic_visit(node)
            return
        for item in node.items:
            ctx_name = ""
            if isinstance(item.context_expr, ast.Name):
                ctx_name = item.context_expr.id
            elif isinstance(item.context_expr, ast.Attribute):
                ctx_name = _resolve_attr(item.context_expr)
            elif isinstance(item.context_expr, ast.Call):
                ctx_name = _resolve_call_name(item.context_expr)
            if ctx_name:
                self.orders[self._current_func].append(ctx_name)
        self.generic_visit(node)
    def visit_Call(self, node: ast.Call) -> None:
        if self._current_func is None:
            self.generic_visit(node)
            return
        call_name = _resolve_call_name(node)
        if call_name and "acquire" in call_name.lower():
            if isinstance(node.func, ast.Attribute):
                lock_name = ""
                if isinstance(node.func.value, ast.Name):
                    lock_name = node.func.value.id
                elif isinstance(node.func.value, ast.Attribute):
                    lock_name = _resolve_attr(node.func.value)
                if lock_name:
                    self.orders[self._current_func].append(lock_name)
        self.generic_visit(node)
def _detect_cycles_in_lock_graph(orders: Dict[str, List[str]]) -> List[List[str]]:
    graph: Dict[str, Set[str]] = defaultdict(set)
    for _func, locks in orders.items():
        for i in range(len(locks) - 1):
            graph[locks[i]].add(locks[i + 1])
    visited: Set[str] = set()
    rec_stack: Set[str] = set()
    cycles: List[List[str]] = []
    path: List[str] = []
    def dfs(node: str) -> None:
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        for neighbor in graph.get(node, set()):
            if neighbor not in visited:
                dfs(neighbor)
            elif neighbor in rec_stack:
                idx = path.index(neighbor)
                cycles.append(list(path[idx:]) + [neighbor])
        path.pop()
        rec_stack.discard(node)
    all_nodes = set(graph.keys())
    for vals in graph.values():
        all_nodes |= vals
    for node in all_nodes:
        if node not in visited:
            dfs(node)
    return cycles
def check_deadlocks(tree: ast.AST) -> List[DeadlockRisk]:
    collector = _LockOrderCollector()
    collector.visit(tree)
    cycles = _detect_cycles_in_lock_graph(collector.orders)
    issues: List[DeadlockRisk] = []
    seen_cycles: Set[FrozenSet[str]] = set()
    for cycle in cycles:
        key = frozenset(cycle)
        if key in seen_cycles:
            continue
        seen_cycles.add(key)
        issues.append(DeadlockRisk(
            severity=BugSeverity.CRITICAL,
            description=f"Potential deadlock: lock cycle {' -> '.join(cycle)}",
            line=0, fix_suggestion="Always acquire locks in a consistent global order.",
            lock_order=cycle,
        ))
    for func_name, locks in collector.orders.items():
        if len(locks) >= 2:
            unique = list(dict.fromkeys(locks))
            if len(unique) >= 2 and frozenset(unique) not in seen_cycles:
                issues.append(DeadlockRisk(
                    severity=BugSeverity.MEDIUM,
                    description=f"Function '{func_name}' acquires multiple locks: {', '.join(unique)}",
                    line=0, fix_suggestion="Ensure a consistent lock ordering.",
                    lock_order=unique,
                ))
    return issues

# ── 5. Concurrency bug detection ──────────────────────────────────────────

class _SharedStateCollector(ast.NodeVisitor):
    def __init__(self, global_names: Set[str]) -> None:
        self.global_names = global_names
        self.writes: Dict[str, Set[str]] = defaultdict(set)
        self.reads: Dict[str, Set[str]] = defaultdict(set)
        self._current_func: Optional[str] = None
        self._local_names: Set[str] = set()
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        prev_func, prev_locals = self._current_func, self._local_names
        self._current_func = node.name
        self._local_names = {arg.arg for arg in node.args.args}
        self.generic_visit(node)
        self._current_func, self._local_names = prev_func, prev_locals
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        prev = self._current_func
        self._current_func = None
        self.generic_visit(node)
        self._current_func = prev
    def visit_Global(self, node: ast.Global) -> None:
        if self._current_func:
            for name in node.names:
                self.global_names.add(name)
        self.generic_visit(node)
    def visit_Assign(self, node: ast.Assign) -> None:
        if self._current_func:
            for target in node.targets:
                self._collect_assigned(target)
        self.generic_visit(node)
    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if self._current_func:
            self._collect_assigned(node.target)
        self.generic_visit(node)
    def _collect_assigned(self, target: ast.expr) -> None:
        if isinstance(target, ast.Name) and target.id in self.global_names:
            self.writes[self._current_func or ""].add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._collect_assigned(elt)
        elif isinstance(target, ast.Subscript):
            if isinstance(target.value, ast.Name) and target.value.id in self.global_names:
                self.writes[self._current_func or ""].add(target.value.id)
    def visit_Name(self, node: ast.Name) -> None:
        if (self._current_func and node.id in self.global_names
                and node.id not in self._local_names
                and isinstance(node.ctx, ast.Load)):
            self.reads[self._current_func].add(node.id)
        self.generic_visit(node)
def _find_module_level_names(tree: ast.Module) -> Set[str]:
    names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names
def check_concurrency_bugs(tree: ast.AST) -> List[ConcurrencyBug]:
    if not isinstance(tree, ast.Module):
        return []
    global_names = _find_module_level_names(tree)
    collector = _SharedStateCollector(global_names)
    collector.visit(tree)
    issues: List[ConcurrencyBug] = []
    all_written: Dict[str, List[str]] = defaultdict(list)
    for func, var_set in collector.writes.items():
        for v in var_set:
            all_written[v].append(func)
    for var, writers in all_written.items():
        readers = [f for f, vs in collector.reads.items() if var in vs]
        accessors = sorted(set(writers + readers))
        if len(accessors) >= 2:
            issues.append(ConcurrencyBug(
                severity=BugSeverity.HIGH,
                description=f"Shared mutable variable '{var}' accessed by multiple async functions without synchronization",
                line=0, fix_suggestion=f"Protect '{var}' with an asyncio.Lock.",
                shared_var=var, accessing_functions=accessors,
            ))
    for var in global_names:
        if var in all_written:
            continue
        readers = [f for f, vs in collector.reads.items() if var in vs]
        if len(readers) >= 3:
            issues.append(ConcurrencyBug(
                severity=BugSeverity.LOW,
                description=f"Module-level variable '{var}' read by {len(readers)} async functions",
                line=0, fix_suggestion="Verify the variable is truly immutable.",
                shared_var=var, accessing_functions=readers,
            ))
    return issues

# ── 6. Task lifecycle analysis ─────────────────────────────────────────────

class _TaskLifecycleChecker(ast.NodeVisitor):
    _CREATE_TASK_NAMES: Set[str] = {
        "asyncio.create_task", "asyncio.ensure_future",
        "loop.create_task", "create_task",
    }
    def __init__(self) -> None:
        self.issues: List[TaskIssue] = []
        self._in_async = False
        self._stored_tasks: Set[int] = set()
        self._try_depth = 0
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        prev = self._in_async
        self._in_async = True
        self._scan_stored_targets(node)
        self.generic_visit(node)
        self._in_async = prev
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        prev = self._in_async
        self._in_async = False
        self.generic_visit(node)
        self._in_async = prev
    def visit_Try(self, node: ast.Try) -> None:
        self._try_depth += 1
        self.generic_visit(node)
        self._try_depth -= 1
    def _scan_stored_targets(self, func_node: ast.AST) -> None:
        for node in ast.walk(func_node):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                self._stored_tasks.add(id(node.value))
            elif isinstance(node, ast.AnnAssign) and node.value and isinstance(node.value, ast.Call):
                self._stored_tasks.add(id(node.value))
    def visit_Call(self, node: ast.Call) -> None:
        if not self._in_async:
            self.generic_visit(node)
            return
        call_name = _resolve_call_name(node)
        if call_name in self._CREATE_TASK_NAMES:
            if id(node) not in self._stored_tasks:
                self.issues.append(TaskIssue(
                    severity=BugSeverity.HIGH,
                    description=f"Fire-and-forget task via {call_name}() – result not stored",
                    line=node.lineno, col=node.col_offset,
                    fix_suggestion="Store the task reference and await or add a done callback.",
                    task_call_name=call_name,
                ))
            if self._try_depth == 0:
                self.issues.append(TaskIssue(
                    severity=BugSeverity.MEDIUM,
                    description=f"Task created via {call_name}() outside try/except",
                    line=node.lineno, col=node.col_offset,
                    fix_suggestion="Wrap task creation and awaiting in try/except.",
                    task_call_name=call_name,
                ))
        self.generic_visit(node)
    def visit_Await(self, node: ast.Await) -> None:
        if isinstance(node.value, ast.Call):
            call_name = _resolve_call_name(node.value)
            if call_name and "gather" in call_name:
                has_ret_exc = any(kw.arg == "return_exceptions" for kw in node.value.keywords)
                if not has_ret_exc and self._try_depth == 0:
                    self.issues.append(TaskIssue(
                        severity=BugSeverity.MEDIUM,
                        description="asyncio.gather without return_exceptions=True and no try/except",
                        line=node.lineno, col=node.col_offset,
                        fix_suggestion="Use return_exceptions=True or wrap in try/except.",
                        task_call_name=call_name,
                    ))
        self.generic_visit(node)
def check_task_lifecycle(tree: ast.AST) -> List[TaskIssue]:
    checker = _TaskLifecycleChecker()
    checker.visit(tree)
    return checker.issues

# ── 7. Performance analysis ───────────────────────────────────────────────

@dataclass
class _AwaitInfo:
    lineno: int
    call_name: str
    names_used: Set[str]

def _extract_names(node: ast.expr) -> Set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
class _SequentialAwaitChecker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.issues: List[PerformanceIssue] = []
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_body(node.body)
        self.generic_visit(node)
    def _check_body(self, body: List[ast.stmt]) -> None:
        await_run: List[_AwaitInfo] = []
        for stmt in body:
            info = self._extract_await(stmt)
            if info is not None:
                await_run.append(info)
            else:
                self._report_run(await_run)
                await_run = []
            if isinstance(stmt, (ast.If, ast.For, ast.While, ast.With)):
                for sub in self._sub_bodies(stmt):
                    self._check_body(sub)
        self._report_run(await_run)
    @staticmethod
    def _sub_bodies(stmt: ast.stmt) -> List[List[ast.stmt]]:
        bodies: List[List[ast.stmt]] = []
        for attr in ("body", "orelse", "finalbody"):
            val = getattr(stmt, attr, None)
            if val:
                bodies.append(val)
        if hasattr(stmt, "handlers"):
            for h in stmt.handlers:
                bodies.append(h.body)
        return bodies
    def _extract_await(self, stmt: ast.stmt) -> Optional[_AwaitInfo]:
        await_node: Optional[ast.Await] = None
        assigned_name: Optional[str] = None
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Await):
            await_node = stmt.value
        elif isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Await):
            await_node = stmt.value
            if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                assigned_name = stmt.targets[0].id
        if await_node is None:
            return None
        call_name = ""
        names_used: Set[str] = set()
        if isinstance(await_node.value, ast.Call):
            call_name = _resolve_call_name(await_node.value)
            names_used = _extract_names(await_node.value)
        if assigned_name:
            names_used.add(assigned_name)
        return _AwaitInfo(lineno=stmt.lineno, call_name=call_name, names_used=names_used)
    def _report_run(self, run: List[_AwaitInfo]) -> None:
        if len(run) < 2:
            return
        n = len(run)
        dep_matrix = np.zeros((n, n), dtype=np.bool_)
        for i in range(n):
            for j in range(i + 1, n):
                if run[i].names_used & run[j].names_used:
                    dep_matrix[i, j] = True
                    dep_matrix[j, i] = True
        groups = self._find_independent_groups(dep_matrix, n)
        for group in groups:
            if len(group) < 2:
                continue
            calls = [run[idx].call_name or f"<expr@{run[idx].lineno}>" for idx in group]
            self.issues.append(PerformanceIssue(
                severity=BugSeverity.MEDIUM,
                description=(
                    f"Sequential awaits on lines "
                    f"{', '.join(str(run[i].lineno) for i in group)} "
                    f"appear independent and could be gathered"
                ),
                line=run[group[0]].lineno,
                fix_suggestion="Use asyncio.gather() to run these concurrently.",
                awaited_calls=calls,
            ))
    @staticmethod
    def _find_independent_groups(dep_matrix: np.ndarray, n: int) -> List[List[int]]:
        groups: List[List[int]] = []
        used = np.zeros(n, dtype=np.bool_)
        for start in range(n):
            if used[start]:
                continue
            group = [start]
            used[start] = True
            for candidate in range(start + 1, n):
                if used[candidate]:
                    continue
                independent = all(not dep_matrix[candidate, m] for m in group)
                if independent:
                    group.append(candidate)
                    used[candidate] = True
            groups.append(group)
        return groups
def check_performance(tree: ast.AST) -> List[PerformanceIssue]:
    checker = _SequentialAwaitChecker()
    checker.visit(tree)
    return checker.issues

# ── 8. AsyncAnalyzer main class ────────────────────────────────────────────

class AsyncAnalyzer:
    """Main entry-point: analyse async Python source for bugs and perf issues."""
    def __init__(self, *, strict: bool = False) -> None:
        self.strict = strict
    def analyze(self, source_code: str) -> AsyncReport:
        tree = ast.parse(textwrap.dedent(source_code))
        return self.analyze_tree(tree)
    def analyze_tree(self, tree: ast.AST) -> AsyncReport:
        report = AsyncReport()
        report.async_functions = collect_async_functions(tree)
        report.potential_bugs.extend(check_event_loop(tree))
        report.potential_bugs.extend(check_missing_awaits(tree))
        report.potential_bugs.extend(check_deadlocks(tree))
        report.potential_bugs.extend(check_concurrency_bugs(tree))
        report.potential_bugs.extend(check_task_lifecycle(tree))
        report.performance_issues.extend(check_performance(tree))
        report.recommendations = self._generate_recommendations(report)
        report.compute_severity_summary()
        return report
    def _generate_recommendations(self, report: AsyncReport) -> List[str]:
        recs: List[str] = []
        categories: Dict[str, int] = defaultdict(int)
        for bug in report.potential_bugs:
            categories[bug.category] += 1
        if categories["event_loop"] > 0:
            recs.append("Avoid blocking calls inside async functions. Use async equivalents or loop.run_in_executor().")
        if categories["missing_await"] > 0:
            recs.append("Ensure all coroutine calls are awaited. Un-awaited coroutines silently do nothing.")
        if categories["deadlock"] > 0:
            recs.append("Establish a global ordering for lock acquisition to prevent deadlocks.")
        if categories["concurrency"] > 0:
            recs.append("Use asyncio.Lock to protect shared mutable state accessed by concurrent tasks.")
        if categories["task_lifecycle"] > 0:
            recs.append("Always store task references from create_task() and handle their exceptions.")
        if report.performance_issues:
            recs.append("Consider asyncio.gather() for independent sequential awaits.")
        score = report.weighted_score()
        if score > 50:
            recs.append(f"High bug-risk score ({score:.0f}). Consider a thorough review of the async code.")
        return recs
    def analyze_file(self, path: str) -> AsyncReport:
        with open(path, "r", encoding="utf-8") as fh:
            return self.analyze(fh.read())

# ── Convenience ────────────────────────────────────────────────────────────

def summarize_report(report: AsyncReport) -> str:
    lines: List[str] = [
        f"Async functions found: {len(report.async_functions)}",
        f"Total issues: {report.total_issues()}",
    ]
    sv = report.severity_vector()
    for label, count in zip(["LOW", "MEDIUM", "HIGH", "CRITICAL"], sv):
        if count > 0:
            lines.append(f"  {label}: {count}")
    lines.append(f"Weighted score: {report.weighted_score():.1f}")
    if report.recommendations:
        lines.append("Recommendations:")
        for rec in report.recommendations:
            lines.append(f"  - {rec}")
    return "\n".join(lines)
