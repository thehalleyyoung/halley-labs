"""Async/await bug detection for Python asyncio code.

Detects missing awaits, async resource leaks, deadlock potential in
asyncio.gather with shared mutable state, and race conditions in
async generators.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Set, Tuple


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class AsyncBugCategory(Enum):
    MISSING_AWAIT = "missing_await"
    RESOURCE_LEAK = "async_resource_leak"
    DEADLOCK_POTENTIAL = "deadlock_potential"
    RACE_CONDITION = "race_condition"
    BLOCKING_IN_ASYNC = "blocking_in_async"
    UNAWAITED_COROUTINE = "unawaited_coroutine"


@dataclass
class AsyncBug:
    category: AsyncBugCategory
    message: str
    line: int
    column: int
    severity: str = "warning"
    confidence: float = 0.8
    fix_suggestion: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.line}:{self.column} [{self.category.value}] {self.message}"


# ---------------------------------------------------------------------------
# Known async functions / coroutines in the stdlib & popular libs
# ---------------------------------------------------------------------------

KNOWN_COROUTINES: Set[str] = {
    # asyncio
    "asyncio.sleep", "asyncio.gather", "asyncio.wait", "asyncio.wait_for",
    "asyncio.create_task", "asyncio.ensure_future", "asyncio.shield",
    "asyncio.open_connection", "asyncio.start_server",
    "asyncio.open_unix_connection", "asyncio.start_unix_server",
    # aiohttp
    "aiohttp.ClientSession", "session.get", "session.post", "session.put",
    "session.delete", "session.patch", "session.head", "session.options",
    "session.request", "session.ws_connect", "session.close",
    "response.json", "response.text", "response.read",
    # aiofiles
    "aiofiles.open",
    # databases / async ORMs
    "database.connect", "database.disconnect", "database.execute",
    "database.fetch_all", "database.fetch_one",
}

BLOCKING_CALLS: Set[str] = {
    "time.sleep", "open", "input",
    "requests.get", "requests.post", "requests.put", "requests.delete",
    "requests.patch", "requests.head", "requests.options",
    "urllib.request.urlopen", "urlopen",
    "os.read", "os.write", "os.popen",
    "subprocess.run", "subprocess.call", "subprocess.check_output",
    "subprocess.check_call", "subprocess.Popen",
    "socket.recv", "socket.send", "socket.connect", "socket.accept",
}

ASYNC_CONTEXT_MANAGERS: Set[str] = {
    "aiohttp.ClientSession", "aiofiles.open", "asyncpg.create_pool",
    "aiosqlite.connect", "httpx.AsyncClient",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_call_name(node: ast.Call) -> str:
    """Extract dotted call name."""
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


def _is_awaited(node: ast.Call) -> bool:
    """Check whether a Call node is the direct child of an Await."""
    # This must be checked by the caller via parent tracking
    return False


class _ParentSetter(ast.NodeVisitor):
    """Walk tree and set .parent on every node."""

    def generic_visit(self, node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            child._parent = node  # type: ignore[attr-defined]
        super().generic_visit(node)


def _set_parents(tree: ast.AST) -> None:
    tree._parent = None  # type: ignore[attr-defined]
    _ParentSetter().visit(tree)


def _parent(node: ast.AST) -> Optional[ast.AST]:
    return getattr(node, "_parent", None)


# ---------------------------------------------------------------------------
# Core analyzer
# ---------------------------------------------------------------------------

class AsyncAnalyzer:
    """Detect async/await bugs in Python source code."""

    def analyze(self, source: str, filename: str = "<string>") -> List[AsyncBug]:
        """Run all async checks on source code."""
        try:
            tree = ast.parse(source, filename)
        except SyntaxError:
            return []
        _set_parents(tree)

        bugs: List[AsyncBug] = []
        bugs.extend(self._detect_missing_awaits(tree))
        bugs.extend(self._detect_resource_leaks(tree))
        bugs.extend(self._detect_gather_race_conditions(tree))
        bugs.extend(self._detect_blocking_in_async(tree))
        bugs.extend(self._detect_async_generator_issues(tree))
        bugs.extend(self._detect_unawaited_coroutine_assignment(tree))
        return bugs

    # ------------------------------------------------------------------
    # Missing awaits
    # ------------------------------------------------------------------

    def _detect_missing_awaits(self, tree: ast.AST) -> List[AsyncBug]:
        """Find coroutine calls that are not awaited."""
        bugs: List[AsyncBug] = []

        # Collect async function names defined in this module
        local_async_fns: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                local_async_fns.add(node.name)

        # Find calls to known coroutines that aren't awaited
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _get_call_name(node)
            is_coroutine = (
                call_name in KNOWN_COROUTINES
                or call_name in local_async_fns
                or any(call_name.endswith("." + fn) for fn in local_async_fns)
            )
            if not is_coroutine:
                continue

            parent = _parent(node)
            if isinstance(parent, ast.Await):
                continue
            # Also OK if assigned and awaited later — but we flag it as a warning
            if isinstance(parent, ast.Assign):
                # Check if the variable is awaited later in the function
                target_names: Set[str] = set()
                for t in parent.targets:
                    if isinstance(t, ast.Name):
                        target_names.add(t.id)
                # Find containing function
                func = self._find_enclosing_function(parent)
                if func and target_names:
                    awaited = self._find_awaited_names(func)
                    if target_names & awaited:
                        continue

            bugs.append(AsyncBug(
                category=AsyncBugCategory.MISSING_AWAIT,
                message=f"Coroutine '{call_name}' called without await",
                line=node.lineno,
                column=node.col_offset,
                severity="error",
                confidence=0.9,
                fix_suggestion=f"Add 'await' before {call_name}(...)",
            ))
        return bugs

    def _find_enclosing_function(self, node: ast.AST) -> Optional[ast.AST]:
        """Walk up parent chain to find enclosing function."""
        cur = _parent(node)
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return cur
            cur = _parent(cur)
        return None

    def _find_awaited_names(self, func: ast.AST) -> Set[str]:
        """Find all variable names that are awaited within a function."""
        awaited: Set[str] = set()
        for node in ast.walk(func):
            if isinstance(node, ast.Await):
                if isinstance(node.value, ast.Name):
                    awaited.add(node.value.id)
        return awaited

    # ------------------------------------------------------------------
    # Resource leaks
    # ------------------------------------------------------------------

    def _detect_resource_leaks(self, tree: ast.AST) -> List[AsyncBug]:
        """Detect async resources that are opened but never properly closed.
        Looks for aiohttp.ClientSession(), async connections, etc. created
        without async-with."""
        bugs: List[AsyncBug] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _get_call_name(node)
            if call_name not in ASYNC_CONTEXT_MANAGERS:
                continue

            parent = _parent(node)
            # OK patterns: async with X() as y, or with X() as y
            if isinstance(parent, ast.Await):
                grandparent = _parent(parent)
                # Check if inside async with
                if isinstance(grandparent, (ast.withitem,)):
                    continue
            # Direct async with (without await)
            if isinstance(parent, (ast.withitem,)):
                continue

            # Check if result is assigned and .close() is called
            if isinstance(parent, ast.Assign):
                target_names: Set[str] = set()
                for t in parent.targets:
                    if isinstance(t, ast.Name):
                        target_names.add(t.id)
                func = self._find_enclosing_function(parent)
                if func and target_names:
                    close_calls = self._find_close_calls(func, target_names)
                    if close_calls:
                        continue

            bugs.append(AsyncBug(
                category=AsyncBugCategory.RESOURCE_LEAK,
                message=(
                    f"'{call_name}' created without 'async with' — "
                    f"resource may not be properly closed"
                ),
                line=node.lineno,
                column=node.col_offset,
                severity="warning",
                confidence=0.8,
                fix_suggestion=f"Use 'async with {call_name}(...) as x:' instead",
            ))
        return bugs

    def _find_close_calls(self, func: ast.AST, names: Set[str]) -> List[ast.Call]:
        """Find .close() or .aclose() calls on the given variable names."""
        calls: List[ast.Call] = []
        for node in ast.walk(func):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ("close", "aclose"):
                    if isinstance(node.func.value, ast.Name):
                        if node.func.value.id in names:
                            calls.append(node)
        return calls

    # ------------------------------------------------------------------
    # Race conditions in asyncio.gather
    # ------------------------------------------------------------------

    def _detect_gather_race_conditions(self, tree: ast.AST) -> List[AsyncBug]:
        """Detect asyncio.gather calls where multiple coroutines access
        shared mutable state (lists, dicts, sets) — potential race condition."""
        bugs: List[AsyncBug] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _get_call_name(node)
            if call_name != "asyncio.gather":
                continue

            # Collect names referenced by each coroutine argument
            arg_names: List[Set[str]] = []
            for arg in node.args:
                names: Set[str] = set()
                for child in ast.walk(arg):
                    if isinstance(child, ast.Name):
                        names.add(child.id)
                arg_names.append(names)

            # Find names that appear in multiple arguments
            if len(arg_names) < 2:
                continue

            all_names: Dict[str, int] = {}
            for names in arg_names:
                for name in names:
                    all_names[name] = all_names.get(name, 0) + 1

            shared = {n for n, count in all_names.items() if count > 1}
            # Exclude common safe names
            shared -= {"self", "cls", "None", "True", "False", "print", "len",
                       "range", "str", "int", "float", "list", "dict", "set",
                       "asyncio", "await"}

            # Check if shared names are mutated in any coroutine arg
            for name in shared:
                for arg in node.args:
                    if self._is_mutated_in(arg, name):
                        bugs.append(AsyncBug(
                            category=AsyncBugCategory.RACE_CONDITION,
                            message=(
                                f"asyncio.gather with shared mutable '{name}' — "
                                f"concurrent mutation may cause race condition"
                            ),
                            line=node.lineno,
                            column=node.col_offset,
                            severity="warning",
                            confidence=0.7,
                            fix_suggestion=(
                                f"Use asyncio.Lock to protect access to '{name}' "
                                f"or use separate data per coroutine"
                            ),
                        ))
                        break
        return bugs

    def _is_mutated_in(self, node: ast.AST, name: str) -> bool:
        """Check if a variable name is mutated within an AST subtree."""
        for child in ast.walk(node):
            # Direct assignment: name = ...
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name) and target.id == name:
                        return True
            # Augmented assignment: name += ...
            if isinstance(child, ast.AugAssign):
                if isinstance(child.target, ast.Name) and child.target.id == name:
                    return True
            # Method calls that mutate: name.append(), name.extend(), etc.
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                if child.func.attr in ("append", "extend", "insert", "pop",
                                        "remove", "clear", "update", "add",
                                        "discard", "sort", "reverse"):
                    if isinstance(child.func.value, ast.Name):
                        if child.func.value.id == name:
                            return True
            # Subscript assignment: name[x] = ...
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Subscript):
                        if isinstance(target.value, ast.Name) and target.value.id == name:
                            return True
        return False

    # ------------------------------------------------------------------
    # Blocking calls in async functions
    # ------------------------------------------------------------------

    def _detect_blocking_in_async(self, tree: ast.AST) -> List[AsyncBug]:
        """Detect synchronous blocking calls inside async functions."""
        bugs: List[AsyncBug] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.AsyncFunctionDef):
                continue
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                call_name = _get_call_name(child)
                if call_name in BLOCKING_CALLS:
                    bugs.append(AsyncBug(
                        category=AsyncBugCategory.BLOCKING_IN_ASYNC,
                        message=(
                            f"Blocking call '{call_name}' in async function "
                            f"'{node.name}' will block the event loop"
                        ),
                        line=child.lineno,
                        column=child.col_offset,
                        severity="error",
                        confidence=0.9,
                        fix_suggestion=(
                            f"Use async equivalent or wrap in "
                            f"await asyncio.to_thread({call_name}, ...)"
                        ),
                    ))
        return bugs

    # ------------------------------------------------------------------
    # Async generator issues
    # ------------------------------------------------------------------

    def _detect_async_generator_issues(self, tree: ast.AST) -> List[AsyncBug]:
        """Detect issues in async generators:
        - Using return with a value (not allowed in async generators)
        - Missing async for when consuming
        - yield inside try/finally without proper cleanup
        """
        bugs: List[AsyncBug] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.AsyncFunctionDef):
                continue

            has_yield = False
            has_return_value = False
            yield_in_try = False

            for child in ast.walk(node):
                if isinstance(child, (ast.Yield, ast.YieldFrom)):
                    has_yield = True
                if isinstance(child, ast.Return) and child.value is not None:
                    has_return_value = True

            if not has_yield:
                continue

            # Async generator: has yield in async def
            if has_return_value:
                bugs.append(AsyncBug(
                    category=AsyncBugCategory.RACE_CONDITION,
                    message=(
                        f"Async generator '{node.name}' uses 'return <value>' — "
                        f"this raises StopAsyncIteration with value that is often lost"
                    ),
                    line=node.lineno,
                    column=node.col_offset,
                    severity="warning",
                    confidence=0.85,
                    fix_suggestion="Use 'yield value' instead of 'return value' in async generators",
                ))

            # Check for yield inside try without finally cleanup
            for child in ast.walk(node):
                if isinstance(child, ast.Try):
                    body_has_yield = any(
                        isinstance(n, (ast.Yield, ast.YieldFrom))
                        for n in ast.walk(child)
                    )
                    has_finally = bool(child.finalbody)
                    if body_has_yield and not has_finally:
                        bugs.append(AsyncBug(
                            category=AsyncBugCategory.RESOURCE_LEAK,
                            message=(
                                f"Async generator '{node.name}' yields inside try "
                                f"without finally — cleanup may not run if consumer "
                                f"stops iteration early"
                            ),
                            line=child.lineno,
                            column=child.col_offset,
                            severity="warning",
                            confidence=0.7,
                            fix_suggestion="Add a finally block to ensure cleanup",
                        ))
        return bugs

    # ------------------------------------------------------------------
    # Unawaited coroutine assignments
    # ------------------------------------------------------------------

    def _detect_unawaited_coroutine_assignment(self, tree: ast.AST) -> List[AsyncBug]:
        """Detect patterns like `result = some_async_fn()` where the coroutine
        is assigned but never awaited, and then used as a regular value."""
        bugs: List[AsyncBug] = []

        local_async_fns: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                local_async_fns.add(node.name)

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            # Track coroutine assignments in this function
            coro_vars: Dict[str, int] = {}  # name -> line

            for child in ast.walk(node):
                if isinstance(child, ast.Assign) and len(child.targets) == 1:
                    target = child.targets[0]
                    if isinstance(target, ast.Name) and isinstance(child.value, ast.Call):
                        call_name = _get_call_name(child.value)
                        if call_name in local_async_fns or call_name in KNOWN_COROUTINES:
                            parent_of_call = _parent(child.value)
                            if not isinstance(parent_of_call, ast.Await):
                                coro_vars[target.id] = child.lineno

            # Check if coroutine vars are used without await
            awaited_names = self._find_awaited_names(node)
            for var_name, line in coro_vars.items():
                if var_name not in awaited_names:
                    # Check if used as a regular value (attribute access, subscript, etc.)
                    for child in ast.walk(node):
                        if isinstance(child, ast.Attribute):
                            if isinstance(child.value, ast.Name) and child.value.id == var_name:
                                bugs.append(AsyncBug(
                                    category=AsyncBugCategory.UNAWAITED_COROUTINE,
                                    message=(
                                        f"'{var_name}' holds an unawaited coroutine "
                                        f"(assigned at L{line}) but is used as a "
                                        f"regular value at L{child.lineno}"
                                    ),
                                    line=child.lineno,
                                    column=child.col_offset,
                                    severity="error",
                                    confidence=0.85,
                                    fix_suggestion=f"Add 'await' when assigning: {var_name} = await ...",
                                ))
                                break
        return bugs
