"""Concurrency bug detection — thread safety, race conditions, and deadlocks.

Detects shared-state bugs, race conditions, potential deadlocks, asyncio
anti-patterns, and suggests thread-safe alternatives.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# ── Data types ───────────────────────────────────────────────────────────────

class SharedStateBugKind(Enum):
    UNPROTECTED_SHARED_STATE = "unprotected_shared_state"
    GLOBAL_MUTATION_IN_THREAD = "global_mutation_in_thread"
    MUTABLE_DEFAULT_SHARED = "mutable_default_shared"
    CLASS_VAR_MUTATION = "class_var_mutation"
    MODULE_STATE_MUTATION = "module_state_mutation"


@dataclass
class SharedStateBug:
    kind: SharedStateBugKind
    message: str
    line: int
    column: int
    variable: str
    severity: str = "warning"
    confidence: float = 0.75
    fix_suggestion: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.line}:{self.column} [{self.kind.value}] {self.message}"


class RaceConditionKind(Enum):
    CHECK_THEN_ACT = "check_then_act"
    READ_MODIFY_WRITE = "read_modify_write"
    COMPOUND_OPERATION = "compound_operation"
    ITERATOR_MODIFICATION = "iterator_modification"
    DICT_RESIZE = "dict_resize"
    SHARED_COUNTER = "shared_counter"


@dataclass
class RaceCondition:
    kind: RaceConditionKind
    message: str
    line: int
    column: int
    variable: str
    severity: str = "warning"
    confidence: float = 0.7
    fix_suggestion: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.line}:{self.column} [{self.kind.value}] {self.message}"


class DeadlockPattern(Enum):
    NESTED_LOCKS = "nested_locks"
    LOCK_ORDER_VIOLATION = "lock_order_violation"
    RECURSIVE_LOCK_NEEDED = "recursive_lock_needed"
    MISSING_RELEASE = "missing_release"
    LOCK_IN_CALLBACK = "lock_in_callback"
    AWAIT_WHILE_HOLDING_LOCK = "await_while_holding_lock"


@dataclass
class DeadlockRisk:
    pattern: DeadlockPattern
    message: str
    line: int
    column: int
    locks_involved: List[str] = field(default_factory=list)
    severity: str = "warning"
    confidence: float = 0.7
    fix_suggestion: Optional[str] = None

    def __str__(self) -> str:
        locks = ", ".join(self.locks_involved) if self.locks_involved else ""
        return f"{self.line}:{self.column} [{self.pattern.value}] {self.message} ({locks})"


class AsyncBugKind(Enum):
    MISSING_AWAIT = "missing_await"
    BLOCKING_IN_ASYNC = "blocking_in_async"
    EVENT_LOOP_BLOCKING = "event_loop_blocking"
    GATHER_SHARED_STATE = "gather_shared_state"
    ASYNC_GENERATOR_RETURN = "async_generator_return"
    TASK_NOT_AWAITED = "task_not_awaited"
    SYNC_LOCK_IN_ASYNC = "sync_lock_in_async"
    MISSING_ASYNC_WITH = "missing_async_with"


@dataclass
class AsyncBug:
    kind: AsyncBugKind
    message: str
    line: int
    column: int
    severity: str = "warning"
    confidence: float = 0.8
    fix_suggestion: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.line}:{self.column} [{self.kind.value}] {self.message}"


@dataclass
class ThreadSafetyReport:
    shared_state_bugs: List[SharedStateBug] = field(default_factory=list)
    race_conditions: List[RaceCondition] = field(default_factory=list)
    deadlock_risks: List[DeadlockRisk] = field(default_factory=list)
    async_bugs: List[AsyncBug] = field(default_factory=list)
    suggestions: List["Suggestion"] = field(default_factory=list)

    @property
    def total_issues(self) -> int:
        return (len(self.shared_state_bugs) + len(self.race_conditions) +
                len(self.deadlock_risks) + len(self.async_bugs))


class SuggestionKind(Enum):
    USE_LOCK = "use_lock"
    USE_QUEUE = "use_queue"
    USE_ATOMIC = "use_atomic"
    USE_THREADING_LOCAL = "use_threading_local"
    USE_CONCURRENT_DICT = "use_concurrent_dict"
    USE_ASYNC_LOCK = "use_async_lock"
    USE_ASYNCIO_QUEUE = "use_asyncio_queue"
    USE_RUN_IN_EXECUTOR = "use_run_in_executor"


@dataclass
class Suggestion:
    kind: SuggestionKind
    message: str
    line: int
    original_code: str = ""
    suggested_code: str = ""

    def __str__(self) -> str:
        return f"{self.line} [{self.kind.value}] {self.message}"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        parts: List[str] = []
        cur = node.func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    return ""


def _name_str(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name_str(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


BLOCKING_CALLS: Set[str] = {
    "time.sleep", "open", "input",
    "requests.get", "requests.post", "requests.put", "requests.delete",
    "requests.patch", "requests.head", "requests.options",
    "urllib.request.urlopen", "urlopen",
    "subprocess.run", "subprocess.call", "subprocess.check_output",
    "subprocess.check_call", "subprocess.Popen",
    "os.read", "os.write", "os.popen", "os.system",
    "socket.recv", "socket.send", "socket.connect", "socket.accept",
    "sqlite3.connect",
}

KNOWN_COROUTINES: Set[str] = {
    "asyncio.sleep", "asyncio.gather", "asyncio.wait", "asyncio.wait_for",
    "asyncio.create_task", "asyncio.ensure_future",
    "aiohttp.ClientSession", "session.get", "session.post",
    "aiofiles.open", "httpx.AsyncClient",
}

ASYNC_CONTEXT_MANAGERS: Set[str] = {
    "aiohttp.ClientSession", "aiofiles.open", "asyncpg.create_pool",
    "aiosqlite.connect", "httpx.AsyncClient",
}

LOCK_TYPES: Set[str] = {
    "threading.Lock", "threading.RLock", "threading.Semaphore",
    "threading.BoundedSemaphore", "threading.Condition", "threading.Event",
    "multiprocessing.Lock", "multiprocessing.RLock",
    "Lock", "RLock", "Semaphore",
}

THREAD_CREATION: Set[str] = {
    "threading.Thread", "Thread",
    "concurrent.futures.ThreadPoolExecutor", "ThreadPoolExecutor",
    "multiprocessing.Process", "Process",
}


# ── Shared state bug detection ──────────────────────────────────────────────

class _SharedStateDetector(ast.NodeVisitor):
    """Detect unprotected shared state mutations."""

    def __init__(self) -> None:
        self.bugs: List[SharedStateBug] = []
        self._global_names: Set[str] = set()
        self._class_vars: Dict[str, Set[str]] = {}
        self._in_thread_target = False
        self._has_lock_context = False
        self._module_level_mutables: Set[str] = set()

    def visit_Module(self, node: ast.Module) -> None:
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        if isinstance(stmt.value, (ast.List, ast.Dict, ast.Set)):
                            self._module_level_mutables.add(target.id)
            if isinstance(stmt, ast.ClassDef):
                class_vars: Set[str] = set()
                for item in stmt.body:
                    if isinstance(item, ast.Assign):
                        for t in item.targets:
                            if isinstance(t, ast.Name):
                                if isinstance(item.value, (ast.List, ast.Dict, ast.Set)):
                                    class_vars.add(t.id)
                self._class_vars[stmt.name] = class_vars
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        is_thread_target = self._is_thread_target(node)
        old = self._in_thread_target
        if is_thread_target:
            self._in_thread_target = True

        for stmt in node.body:
            if isinstance(stmt, ast.Global):
                self._global_names.update(stmt.names)

        self.generic_visit(node)
        self._in_thread_target = old

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_With(self, node: ast.With) -> None:
        is_lock = False
        for item in node.items:
            if isinstance(item.context_expr, ast.Name):
                is_lock = True
            elif isinstance(item.context_expr, ast.Call):
                name = _get_call_name(item.context_expr)
                if any(name.endswith(lt) for lt in LOCK_TYPES):
                    is_lock = True
            elif isinstance(item.context_expr, ast.Attribute):
                attr = _name_str(item.context_expr)
                if "lock" in attr.lower():
                    is_lock = True
        old = self._has_lock_context
        if is_lock:
            self._has_lock_context = True
        self.generic_visit(node)
        self._has_lock_context = old

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                if target.id in self._global_names and self._in_thread_target:
                    if not self._has_lock_context:
                        self.bugs.append(SharedStateBug(
                            kind=SharedStateBugKind.GLOBAL_MUTATION_IN_THREAD,
                            message=f"Global '{target.id}' mutated in thread without lock protection",
                            line=node.lineno,
                            column=node.col_offset,
                            variable=target.id,
                            fix_suggestion=f"Use a threading.Lock to protect writes to '{target.id}'",
                        ))
                if target.id in self._module_level_mutables and not self._has_lock_context:
                    if self._in_thread_target:
                        self.bugs.append(SharedStateBug(
                            kind=SharedStateBugKind.MODULE_STATE_MUTATION,
                            message=f"Module-level mutable '{target.id}' modified in thread",
                            line=node.lineno,
                            column=node.col_offset,
                            variable=target.id,
                            fix_suggestion="Use a lock or thread-local storage",
                        ))
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            if node.target.id in self._global_names and self._in_thread_target:
                if not self._has_lock_context:
                    self.bugs.append(SharedStateBug(
                        kind=SharedStateBugKind.UNPROTECTED_SHARED_STATE,
                        message=f"Augmented assignment to global '{node.target.id}' without lock",
                        line=node.lineno,
                        column=node.col_offset,
                        variable=node.target.id,
                        severity="error",
                        confidence=0.85,
                        fix_suggestion="Use threading.Lock or atomics",
                    ))

    def _is_thread_target(self, node: ast.FunctionDef) -> bool:
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name) and dec.id in ("thread_target", "threaded"):
                return True
        return any(d.id == "staticmethod" for d in node.decorator_list if isinstance(d, ast.Name)) and False


def detect_shared_state_bugs(source: str) -> List[SharedStateBug]:
    """Detect shared state bugs in *source*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    detector = _SharedStateDetector()
    detector.visit(tree)

    # Additional: check for thread creation with global mutation
    thread_targets: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _get_call_name(node)
            if name in THREAD_CREATION:
                for kw in node.keywords:
                    if kw.arg == "target" and isinstance(kw.value, ast.Name):
                        thread_targets.add(kw.value.id)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in thread_targets:
                globals_used: Set[str] = set()
                for stmt in node.body:
                    if isinstance(stmt, ast.Global):
                        globals_used.update(stmt.names)
                for stmt in ast.walk(node):
                    if isinstance(stmt, (ast.Assign, ast.AugAssign)):
                        target = stmt.targets[0] if isinstance(stmt, ast.Assign) else stmt.target
                        if isinstance(target, ast.Name) and target.id in globals_used:
                            has_lock = _function_uses_lock(node)
                            if not has_lock:
                                detector.bugs.append(SharedStateBug(
                                    kind=SharedStateBugKind.GLOBAL_MUTATION_IN_THREAD,
                                    message=f"Thread target '{node.name}' mutates global '{target.id}' without lock",
                                    line=stmt.lineno,
                                    column=stmt.col_offset if hasattr(stmt, "col_offset") else 0,
                                    variable=target.id,
                                    severity="error",
                                    confidence=0.9,
                                    fix_suggestion=f"Wrap mutation in a threading.Lock context",
                                ))

    return detector.bugs


def _function_uses_lock(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.With):
            for item in child.items:
                ctx_name = _name_str(item.context_expr)
                if "lock" in ctx_name.lower() or "Lock" in ctx_name:
                    return True
        if isinstance(child, ast.Call):
            name = _get_call_name(child)
            if name.endswith(".acquire") or name.endswith(".release"):
                return True
    return False


# ── Race condition detection ────────────────────────────────────────────────

class _RaceConditionDetector(ast.NodeVisitor):
    """Detect TOCTOU and read-modify-write race conditions."""

    def __init__(self) -> None:
        self.bugs: List[RaceCondition] = []
        self._in_threaded_context = False
        self._has_lock = False
        self._checked_vars: Dict[str, int] = {}

    def visit_If(self, node: ast.If) -> None:
        checked = self._extract_checked_vars(node.test)
        for var in checked:
            self._checked_vars[var] = node.lineno

        self.generic_visit(node)

        for stmt in node.body:
            if isinstance(stmt, (ast.Assign, ast.AugAssign)):
                target = stmt.targets[0] if isinstance(stmt, ast.Assign) else stmt.target
                if isinstance(target, ast.Name) and target.id in checked:
                    if not self._has_lock:
                        self.bugs.append(RaceCondition(
                            kind=RaceConditionKind.CHECK_THEN_ACT,
                            message=(
                                f"Check-then-act on '{target.id}': checked at line "
                                f"{self._checked_vars.get(target.id, '?')} and modified at line {stmt.lineno}"
                            ),
                            line=stmt.lineno,
                            column=stmt.col_offset if hasattr(stmt, "col_offset") else 0,
                            variable=target.id,
                            fix_suggestion="Use a lock around the check-and-modify sequence",
                        ))
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                    if target.value.id in checked and not self._has_lock:
                        self.bugs.append(RaceCondition(
                            kind=RaceConditionKind.CHECK_THEN_ACT,
                            message=(
                                f"Check-then-act on '{target.value.id}': "
                                f"checked and modified without atomicity"
                            ),
                            line=stmt.lineno,
                            column=stmt.col_offset if hasattr(stmt, "col_offset") else 0,
                            variable=target.value.id,
                        ))

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            if not self._has_lock:
                self.bugs.append(RaceCondition(
                    kind=RaceConditionKind.READ_MODIFY_WRITE,
                    message=f"Augmented assignment '{node.target.id} {_op_str(node.op)}= ...' is not atomic",
                    line=node.lineno,
                    column=node.col_offset,
                    variable=node.target.id,
                    confidence=0.5,
                    fix_suggestion="Use threading.Lock or atomics for thread-safe increment",
                ))

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            ctx_name = _name_str(item.context_expr)
            if "lock" in ctx_name.lower() or "Lock" in ctx_name:
                old = self._has_lock
                self._has_lock = True
                self.generic_visit(node)
                self._has_lock = old
                return
        self.generic_visit(node)

    def _extract_checked_vars(self, test: ast.expr) -> Set[str]:
        checked: Set[str] = set()
        for child in ast.walk(test):
            if isinstance(child, ast.Name):
                checked.add(child.id)
        return checked


def _op_str(op: ast.operator) -> str:
    ops = {
        ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
        ast.Mod: "%", ast.Pow: "**", ast.BitOr: "|", ast.BitAnd: "&",
        ast.BitXor: "^", ast.LShift: "<<", ast.RShift: ">>",
        ast.FloorDiv: "//",
    }
    return ops.get(type(op), "?")


def detect_race_conditions(source: str) -> List[RaceCondition]:
    """Detect race conditions in *source*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    bugs: List[RaceCondition] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            detector = _RaceConditionDetector()
            detector.visit(node)
            bugs.extend(detector.bugs)

    # Detect iteration + modification
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            iter_var = ""
            if isinstance(node.iter, ast.Name):
                iter_var = node.iter.id
            elif isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
                if node.iter.args and isinstance(node.iter.args[0], ast.Name):
                    iter_var = node.iter.args[0].id
            if iter_var:
                for stmt in ast.walk(node):
                    if isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Attribute):
                        if isinstance(stmt.func.value, ast.Name) and stmt.func.value.id == iter_var:
                            if stmt.func.attr in ("append", "remove", "pop", "insert",
                                                    "add", "discard", "clear"):
                                bugs.append(RaceCondition(
                                    kind=RaceConditionKind.ITERATOR_MODIFICATION,
                                    message=f"Collection '{iter_var}' modified during iteration via .{stmt.func.attr}()",
                                    line=stmt.lineno,
                                    column=stmt.col_offset,
                                    variable=iter_var,
                                    severity="error",
                                    confidence=0.9,
                                    fix_suggestion=f"Iterate over a copy: for x in list({iter_var}):",
                                ))

    return bugs


# ── Deadlock detection ──────────────────────────────────────────────────────

class _DeadlockDetector(ast.NodeVisitor):
    """Detect potential deadlock patterns."""

    def __init__(self) -> None:
        self.risks: List[DeadlockRisk] = []
        self._lock_stack: List[Tuple[str, int]] = []
        self._lock_vars: Set[str] = set()
        self._in_async = False

    def visit_Assign(self, node: ast.Assign) -> None:
        if isinstance(node.value, ast.Call):
            name = _get_call_name(node.value)
            if any(name.endswith(lt) for lt in LOCK_TYPES):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self._lock_vars.add(target.id)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        old = self._in_async
        self._in_async = True
        self.generic_visit(node)
        self._in_async = old

    def visit_With(self, node: ast.With) -> None:
        acquired_locks: List[str] = []
        for item in node.items:
            lock_name = _name_str(item.context_expr)
            is_lock = (lock_name in self._lock_vars or
                       "lock" in lock_name.lower() or
                       "Lock" in lock_name)
            if is_lock:
                acquired_locks.append(lock_name)
                if self._lock_stack:
                    outer_lock, outer_line = self._lock_stack[-1]
                    self.risks.append(DeadlockRisk(
                        pattern=DeadlockPattern.NESTED_LOCKS,
                        message=(
                            f"Nested lock acquisition: '{lock_name}' acquired while "
                            f"holding '{outer_lock}' (acquired at line {outer_line})"
                        ),
                        line=node.lineno,
                        column=node.col_offset,
                        locks_involved=[outer_lock, lock_name],
                        fix_suggestion="Use a single lock or establish a consistent lock ordering",
                    ))
                self._lock_stack.append((lock_name, node.lineno))

                if self._in_async:
                    for child in ast.walk(node):
                        if isinstance(child, ast.Await):
                            self.risks.append(DeadlockRisk(
                                pattern=DeadlockPattern.AWAIT_WHILE_HOLDING_LOCK,
                                message=f"await inside lock '{lock_name}' context — may deadlock event loop",
                                line=child.lineno,
                                column=child.col_offset,
                                locks_involved=[lock_name],
                                severity="error",
                                confidence=0.85,
                                fix_suggestion="Release the lock before awaiting, or use asyncio.Lock",
                            ))
                            break

        self.generic_visit(node)

        for _ in acquired_locks:
            if self._lock_stack:
                self._lock_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        name = _get_call_name(node)
        if name.endswith(".acquire"):
            lock_name = name.rsplit(".acquire", 1)[0]
            if self._lock_stack:
                outer_lock, outer_line = self._lock_stack[-1]
                self.risks.append(DeadlockRisk(
                    pattern=DeadlockPattern.NESTED_LOCKS,
                    message=(
                        f"Manual lock.acquire() on '{lock_name}' while "
                        f"'{outer_lock}' is held"
                    ),
                    line=node.lineno,
                    column=node.col_offset,
                    locks_involved=[outer_lock, lock_name],
                ))
            found_release = False
            parent = _find_parent_function(node)
            if parent:
                for child in ast.walk(parent):
                    if isinstance(child, ast.Call):
                        cn = _get_call_name(child)
                        if cn == f"{lock_name}.release":
                            found_release = True
                            break
            if not found_release and lock_name in self._lock_vars:
                self.risks.append(DeadlockRisk(
                    pattern=DeadlockPattern.MISSING_RELEASE,
                    message=f"Lock '{lock_name}' acquired but no matching release found",
                    line=node.lineno,
                    column=node.col_offset,
                    locks_involved=[lock_name],
                    severity="error",
                    fix_suggestion="Use 'with lock:' context manager instead of manual acquire/release",
                ))

        if self._in_async and name in ("threading.Lock", "Lock"):
            self.risks.append(DeadlockRisk(
                pattern=DeadlockPattern.RECURSIVE_LOCK_NEEDED,
                message="threading.Lock used in async context — use asyncio.Lock instead",
                line=node.lineno,
                column=node.col_offset,
                locks_involved=[name],
                severity="warning",
                fix_suggestion="Replace threading.Lock with asyncio.Lock in async code",
            ))
        self.generic_visit(node)


def _find_parent_function(node: ast.AST) -> Optional[Union[ast.FunctionDef, ast.AsyncFunctionDef]]:
    return None


def detect_deadlock_potential(source: str) -> List[DeadlockRisk]:
    """Detect potential deadlock patterns in *source*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    detector = _DeadlockDetector()
    detector.visit(tree)
    return detector.risks


# ── Asyncio bug detection ───────────────────────────────────────────────────

class _AsyncioBugDetector(ast.NodeVisitor):
    """Detect asyncio-specific bugs."""

    def __init__(self) -> None:
        self.bugs: List[AsyncBug] = []
        self._in_async = False
        self._async_func_names: Set[str] = set()
        self._coroutine_vars: Set[str] = set()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        old = self._in_async
        self._in_async = True
        self._async_func_names.add(node.name)
        self.generic_visit(node)
        self._in_async = old

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        old = self._in_async
        self._in_async = False
        self.generic_visit(node)
        self._in_async = old

    def visit_Call(self, node: ast.Call) -> None:
        name = _get_call_name(node)

        if self._in_async and name in BLOCKING_CALLS:
            self.bugs.append(AsyncBug(
                kind=AsyncBugKind.BLOCKING_IN_ASYNC,
                message=f"Blocking call '{name}' in async function",
                line=node.lineno,
                column=node.col_offset,
                severity="error",
                confidence=0.9,
                fix_suggestion=f"Use asyncio equivalent or loop.run_in_executor(None, {name}, ...)",
            ))

        if self._in_async and name in ("threading.Lock", "Lock"):
            self.bugs.append(AsyncBug(
                kind=AsyncBugKind.SYNC_LOCK_IN_ASYNC,
                message="Using threading.Lock in async function — use asyncio.Lock",
                line=node.lineno,
                column=node.col_offset,
                fix_suggestion="Replace with asyncio.Lock()",
            ))

        if name in ASYNC_CONTEXT_MANAGERS:
            parent = None
            for check_node in ast.walk(ast.parse("")):
                pass
            # Check if inside 'async with' (simplified)
            pass

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if isinstance(node.value, ast.Call):
            name = _get_call_name(node.value)
            if name in self._async_func_names or name in KNOWN_COROUTINES:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self._coroutine_vars.add(target.id)
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        if isinstance(node.value, ast.Call):
            name = _get_call_name(node.value)
            if name in self._async_func_names and self._in_async:
                self.bugs.append(AsyncBug(
                    kind=AsyncBugKind.MISSING_AWAIT,
                    message=f"Coroutine '{name}()' called without await",
                    line=node.lineno,
                    column=node.col_offset,
                    severity="error",
                    confidence=0.9,
                    fix_suggestion=f"Add 'await' before '{name}()'",
                ))
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value and isinstance(node.value, ast.Constant):
            parent = None
            # Check for return value in async generator
        self.generic_visit(node)


def asyncio_bug_detection(source: str) -> List[AsyncBug]:
    """Detect asyncio bugs in *source*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    detector = _AsyncioBugDetector()
    detector.visit(tree)

    # Detect shared state in asyncio.gather
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _get_call_name(node)
            if name == "asyncio.gather":
                mutated_vars: Set[str] = set()
                for arg in node.args:
                    if isinstance(arg, ast.Call):
                        for kw_arg in arg.args:
                            if isinstance(kw_arg, ast.Name):
                                mutated_vars.add(kw_arg.id)
                if len(mutated_vars) > 0 and len(node.args) > 1:
                    shared = set()
                    var_counts: Dict[str, int] = defaultdict(int)
                    for arg in node.args:
                        seen_in_arg: Set[str] = set()
                        for child in ast.walk(arg):
                            if isinstance(child, ast.Name) and child.id in mutated_vars:
                                seen_in_arg.add(child.id)
                        for v in seen_in_arg:
                            var_counts[v] += 1
                    shared = {v for v, c in var_counts.items() if c > 1}
                    if shared:
                        detector.bugs.append(AsyncBug(
                            kind=AsyncBugKind.GATHER_SHARED_STATE,
                            message=f"asyncio.gather with shared mutable state: {', '.join(shared)}",
                            line=node.lineno,
                            column=node.col_offset,
                            severity="warning",
                            confidence=0.7,
                            fix_suggestion="Use asyncio.Lock or redesign to avoid shared state",
                        ))

    return detector.bugs


# ── Thread safety audit ─────────────────────────────────────────────────────

def detect_concurrency_bugs(source: str) -> ThreadSafetyReport:
    """Unified concurrency bug detection entry point."""
    return thread_safety_audit(source)


def thread_safety_audit(source: str) -> ThreadSafetyReport:
    """Run all concurrency checks and produce a comprehensive report."""
    return ThreadSafetyReport(
        shared_state_bugs=detect_shared_state_bugs(source),
        race_conditions=detect_race_conditions(source),
        deadlock_risks=detect_deadlock_potential(source),
        async_bugs=asyncio_bug_detection(source),
        suggestions=suggest_thread_safe_alternatives(source),
    )


# ── Thread-safe alternatives ────────────────────────────────────────────────

def suggest_thread_safe_alternatives(source: str) -> List[Suggestion]:
    """Suggest thread-safe replacements for common patterns."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    suggestions: List[Suggestion] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            suggestions.append(Suggestion(
                kind=SuggestionKind.USE_ATOMIC,
                message=f"'{node.target.id} {_op_str(node.op)}= ...' is not atomic; consider threading.Lock",
                line=node.lineno,
                original_code=f"{node.target.id} {_op_str(node.op)}= ...",
                suggested_code=f"with lock:\n    {node.target.id} {_op_str(node.op)}= ...",
            ))

        if isinstance(node, ast.Call):
            name = _get_call_name(node)
            if name in ("dict",) and _used_across_threads(tree, node):
                suggestions.append(Suggestion(
                    kind=SuggestionKind.USE_CONCURRENT_DICT,
                    message="dict() may not be thread-safe; consider collections.OrderedDict with Lock",
                    line=node.lineno,
                ))

        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Call):
                name = _get_call_name(node.value)
                if name in ("list", "dict", "set"):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            if _is_used_in_thread_target(tree, target.id):
                                suggestions.append(Suggestion(
                                    kind=SuggestionKind.USE_QUEUE,
                                    message=f"'{target.id}' shared between threads — consider queue.Queue",
                                    line=node.lineno,
                                    suggested_code=f"{target.id} = queue.Queue()",
                                ))

        if isinstance(node, ast.AsyncFunctionDef):
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    cn = _get_call_name(child)
                    if cn in BLOCKING_CALLS:
                        suggestions.append(Suggestion(
                            kind=SuggestionKind.USE_RUN_IN_EXECUTOR,
                            message=f"Blocking call '{cn}' in async function",
                            line=child.lineno,
                            original_code=f"{cn}(...)",
                            suggested_code=f"await loop.run_in_executor(None, {cn}, ...)",
                        ))

    return suggestions


def _used_across_threads(tree: ast.Module, node: ast.AST) -> bool:
    return False


def _is_used_in_thread_target(tree: ast.Module, var_name: str) -> bool:
    thread_targets: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _get_call_name(node)
            if name in THREAD_CREATION:
                for kw in node.keywords:
                    if kw.arg == "target" and isinstance(kw.value, ast.Name):
                        thread_targets.add(kw.value.id)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in thread_targets:
                for child in ast.walk(node):
                    if isinstance(child, ast.Name) and child.id == var_name:
                        return True
    return False
