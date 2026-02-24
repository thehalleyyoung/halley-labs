"""
Refinement type inference for Python async/await patterns.

Models coroutine types, async iterators, async context managers, task groups,
and common async library patterns (asyncio, aiohttp, trio).  Integrates with
the refinement lattice so that ``await`` expressions, ``async for``, and
``async with`` all produce properly refined types.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from src.refinement_lattice import (
    ANY_TYPE,
    BOOL_TYPE,
    FLOAT_TYPE,
    INT_TYPE,
    NEVER_TYPE,
    NONE_TYPE,
    STR_TYPE,
    BaseTypeKind,
    BaseTypeR,
    Pred,
    PredOp,
    RefType,
)

# ---------------------------------------------------------------------------
# Supporting data-classes
# ---------------------------------------------------------------------------

OBJECT_TYPE = BaseTypeR(BaseTypeKind.OBJECT)


class WarningLevel(Enum):
    """Severity of an async-related diagnostic."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()


@dataclass
class AsyncWarning:
    """A diagnostic produced during async analysis."""
    level: WarningLevel
    message: str
    location: Tuple[int, int]  # (line, col)
    node: Optional[ast.AST] = None

    def pretty(self) -> str:
        lvl = self.level.name
        line, col = self.location
        return f"[{lvl}] line {line}:{col} – {self.message}"


@dataclass
class CoroutineType:
    """The three-parameter Coroutine[YieldType, SendType, ReturnType]."""
    yield_type: RefType
    send_type: RefType
    return_type: RefType

    def as_base(self) -> BaseTypeR:
        return BaseTypeR(
            BaseTypeKind.OBJECT,
            type_args=(
                self.yield_type.base,
                self.send_type.base,
                self.return_type.base,
            ),
        )

    def as_reftype(self, binder: str = "ν") -> RefType:
        pred = Pred.isinstance_(binder, "Coroutine")
        return RefType(binder, self.as_base(), pred)


@dataclass
class AsyncContextInfo:
    """Information about an async context manager binding."""
    manager_type: RefType
    resource_type: RefType
    has_aenter: bool = True
    has_aexit: bool = True


@dataclass
class TaskInfo:
    """A tracked ``asyncio.Task`` or equivalent."""
    coroutine_type: RefType
    group: Optional[str] = None
    cancel_scope: Optional[str] = None
    creation_site: Optional[Tuple[int, int]] = None


@dataclass
class AnalysisState:
    """Snapshot of the abstract state during async analysis."""
    var_types: Dict[str, RefType] = field(default_factory=dict)
    predicates: List[Pred] = field(default_factory=list)
    async_context_stack: List[AsyncContextInfo] = field(default_factory=list)
    active_tasks: Dict[str, TaskInfo] = field(default_factory=dict)
    in_async_function: bool = False
    warnings: List[AsyncWarning] = field(default_factory=list)

    # ---- helpers ----
    def copy(self) -> AnalysisState:
        return AnalysisState(
            var_types=dict(self.var_types),
            predicates=list(self.predicates),
            async_context_stack=list(self.async_context_stack),
            active_tasks=dict(self.active_tasks),
            in_async_function=self.in_async_function,
            warnings=list(self.warnings),
        )

    def bind(self, name: str, typ: RefType) -> AnalysisState:
        new = self.copy()
        new.var_types[name] = typ
        return new

    def add_pred(self, p: Pred) -> AnalysisState:
        new = self.copy()
        new.predicates.append(p)
        return new

    def push_context(self, ctx: AsyncContextInfo) -> AnalysisState:
        new = self.copy()
        new.async_context_stack.append(ctx)
        return new

    def pop_context(self) -> Tuple[Optional[AsyncContextInfo], AnalysisState]:
        new = self.copy()
        if new.async_context_stack:
            return new.async_context_stack.pop(), new
        return None, new

    def get_type(self, name: str) -> RefType:
        return self.var_types.get(name, RefType.trivial(ANY_TYPE))


@dataclass
class FunctionRefinement:
    """Refinement for a (possibly async) function."""
    name: str
    param_types: Dict[str, RefType] = field(default_factory=dict)
    return_type: RefType = field(default_factory=lambda: RefType.trivial(NONE_TYPE))
    preconditions: List[Pred] = field(default_factory=list)
    postconditions: List[Pred] = field(default_factory=list)
    is_async: bool = False
    is_generator: bool = False
    coroutine_type: Optional[CoroutineType] = None

    def pretty(self) -> str:
        params = ", ".join(f"{n}: {t.pretty()}" for n, t in self.param_types.items())
        ret = self.return_type.pretty()
        prefix = "async " if self.is_async else ""
        return f"{prefix}def {self.name}({params}) -> {ret}"


# ---------------------------------------------------------------------------
# Common async library type patterns
# ---------------------------------------------------------------------------

def _obj(*args: BaseTypeR) -> BaseTypeR:
    return BaseTypeR(BaseTypeKind.OBJECT, type_args=tuple(args))


_LIST_BYTES = BaseTypeR(BaseTypeKind.LIST, type_args=(BaseTypeR(BaseTypeKind.STR),))


ASYNC_PATTERNS: Dict[str, Dict[str, RefType]] = {
    # asyncio builtins
    "asyncio": {
        "sleep": RefType("ν", NONE_TYPE, Pred.isinstance_("ν", "Coroutine")),
        "gather": RefType("ν", BaseTypeR(BaseTypeKind.TUPLE), Pred.isinstance_("ν", "tuple")),
        "wait": RefType(
            "ν",
            BaseTypeR(BaseTypeKind.TUPLE, type_args=(
                BaseTypeR(BaseTypeKind.SET),
                BaseTypeR(BaseTypeKind.SET),
            )),
            Pred.true_(),
        ),
        "create_task": RefType("ν", OBJECT_TYPE, Pred.isinstance_("ν", "Task")),
        "wait_for": RefType("ν", ANY_TYPE, Pred.true_()),
        "shield": RefType("ν", ANY_TYPE, Pred.true_()),
        "Lock": RefType("ν", OBJECT_TYPE, Pred.isinstance_("ν", "Lock")),
        "Semaphore": RefType("ν", OBJECT_TYPE, Pred.isinstance_("ν", "Semaphore")),
        "Event": RefType("ν", OBJECT_TYPE, Pred.isinstance_("ν", "Event")),
        "Queue": RefType("ν", OBJECT_TYPE, Pred.isinstance_("ν", "Queue")),
        "TaskGroup": RefType("ν", OBJECT_TYPE, Pred.isinstance_("ν", "TaskGroup")),
    },
    # aiohttp
    "aiohttp": {
        "ClientSession": RefType("ν", OBJECT_TYPE, Pred.isinstance_("ν", "ClientSession")),
        "ClientResponse": RefType("ν", OBJECT_TYPE, Pred.isinstance_("ν", "ClientResponse")),
        "get": RefType("ν", OBJECT_TYPE, Pred.isinstance_("ν", "ClientResponse")),
        "post": RefType("ν", OBJECT_TYPE, Pred.isinstance_("ν", "ClientResponse")),
        "json": RefType("ν", ANY_TYPE, Pred.true_()),
        "text": RefType("ν", STR_TYPE, Pred.true_()),
        "read": RefType("ν", BaseTypeR(BaseTypeKind.STR), Pred.true_()),
    },
    # trio
    "trio": {
        "sleep": RefType("ν", NONE_TYPE, Pred.isinstance_("ν", "Coroutine")),
        "open_nursery": RefType("ν", OBJECT_TYPE, Pred.isinstance_("ν", "Nursery")),
        "CancelScope": RefType("ν", OBJECT_TYPE, Pred.isinstance_("ν", "CancelScope")),
        "open_tcp_stream": RefType("ν", OBJECT_TYPE, Pred.isinstance_("ν", "SocketStream")),
        "open_file": RefType("ν", OBJECT_TYPE, Pred.isinstance_("ν", "AsyncFile")),
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COROUTINE_MARKER = "Coroutine"
_ASYNC_ITER_MARKER = "AsyncIterator"
_ASYNC_CM_MARKER = "AsyncContextManager"


def _node_loc(node: ast.AST) -> Tuple[int, int]:
    return (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))


def _name_of(node: ast.expr) -> Optional[str]:
    """Extract a simple dotted name from an expression, if possible."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name_of(node.value)
        if base is not None:
            return f"{base}.{node.attr}"
    return None


def _resolve_library_type(dotted: str) -> Optional[RefType]:
    """Look up *dotted* in ``ASYNC_PATTERNS``."""
    parts = dotted.split(".", 1)
    if len(parts) == 2:
        lib, member = parts
        lib_map = ASYNC_PATTERNS.get(lib)
        if lib_map is not None:
            return lib_map.get(member)
    return None


# ---------------------------------------------------------------------------
# AsyncAnalyzer
# ---------------------------------------------------------------------------

class AsyncAnalyzer:
    """Refinement-type analysis for async/await constructs.

    Works over the Python ``ast`` and an ``AnalysisState`` that tracks
    variable bindings, predicates, and async-context information.
    """

    def __init__(self) -> None:
        self._warnings: List[AsyncWarning] = []

    # ------------------------------------------------------------------ #
    # Top-level entry points                                              #
    # ------------------------------------------------------------------ #

    def analyze_async_function(
        self, func: ast.AsyncFunctionDef, state: AnalysisState
    ) -> FunctionRefinement:
        """Produce a ``FunctionRefinement`` for an ``async def``."""
        inner = state.copy()
        inner.in_async_function = True

        param_types: Dict[str, RefType] = {}
        for arg in func.args.args:
            name = arg.arg
            ann = self._annotation_to_reftype(arg.annotation) if arg.annotation else RefType.trivial(ANY_TYPE)
            param_types[name] = ann
            inner = inner.bind(name, ann)

        # Walk the body to determine the return type
        ret_type, inner = self._analyze_body(func.body, inner)

        # Check for async-generator (contains ``yield``)
        is_gen = self._body_has_yield(func.body)

        if is_gen:
            coro = CoroutineType(
                yield_type=ret_type,
                send_type=RefType.trivial(NONE_TYPE),
                return_type=RefType.trivial(NONE_TYPE),
            )
        else:
            coro = CoroutineType(
                yield_type=RefType.trivial(NONE_TYPE),
                send_type=RefType.trivial(NONE_TYPE),
                return_type=ret_type,
            )

        return FunctionRefinement(
            name=func.name,
            param_types=param_types,
            return_type=coro.as_reftype(),
            preconditions=[p for p in inner.predicates],
            postconditions=[],
            is_async=True,
            is_generator=is_gen,
            coroutine_type=coro,
        )

    def analyze_await(
        self, expr: ast.Await, state: AnalysisState
    ) -> Tuple[RefType, AnalysisState]:
        """Type the result of ``await <expr>``."""
        awaitable_type, state = self._analyze_expr(expr.value, state)

        warnings = self.check_await_safety(expr.value, state)
        state.warnings.extend(warnings)

        if self._is_coroutine_type(awaitable_type):
            result = self._unwrap_coroutine(awaitable_type)
        elif self._has_pred_marker(awaitable_type, "Awaitable"):
            result = self._unwrap_awaitable(awaitable_type)
        else:
            # Possibly not awaitable – still produce a type but warn
            result = RefType.trivial(ANY_TYPE)

        return result, state

    def analyze_async_for(
        self, node: ast.AsyncFor, state: AnalysisState
    ) -> AnalysisState:
        """Analyze ``async for target in iter_expr``."""
        iter_type, state = self._analyze_expr(node.iter, state)

        if self._is_async_iterable(iter_type):
            elem_type = self._extract_async_iter_element(iter_type)
        else:
            elem_type = RefType.trivial(ANY_TYPE)
            state.warnings.append(AsyncWarning(
                level=WarningLevel.WARNING,
                message="Expression may not be an async iterable",
                location=_node_loc(node.iter),
                node=node.iter,
            ))

        # Bind the loop variable
        target_name = self._target_name(node.target)
        if target_name is not None:
            state = state.bind(target_name, elem_type)
            state = state.add_pred(Pred.is_not_none(target_name))

        # Walk loop body
        _, state = self._analyze_body(node.body, state)

        # Walk else clause
        if node.orelse:
            _, state = self._analyze_body(node.orelse, state)

        return state

    def analyze_async_with(
        self, node: ast.AsyncWith, state: AnalysisState
    ) -> AnalysisState:
        """Analyze ``async with expr as var``."""
        for item in node.items:
            ctx_type, state = self._analyze_expr(item.context_expr, state)

            if self._is_async_context_manager(ctx_type):
                resource_type = self._extract_aenter_type(ctx_type)
            else:
                resource_type = RefType.trivial(ANY_TYPE)
                state.warnings.append(AsyncWarning(
                    level=WarningLevel.WARNING,
                    message="Expression may not be an async context manager",
                    location=_node_loc(item.context_expr),
                    node=item.context_expr,
                ))

            ctx_info = AsyncContextInfo(
                manager_type=ctx_type,
                resource_type=resource_type,
                has_aenter=self._is_async_context_manager(ctx_type),
                has_aexit=self._is_async_context_manager(ctx_type),
            )
            state = state.push_context(ctx_info)

            if item.optional_vars is not None:
                var_name = self._target_name(item.optional_vars)
                if var_name is not None:
                    state = state.bind(var_name, resource_type)
                    state = state.add_pred(Pred.is_not_none(var_name))

        _, state = self._analyze_body(node.body, state)

        # Pop contexts in reverse
        for _ in node.items:
            _, state = state.pop_context()

        return state

    def analyze_task_group(
        self, node: ast.With, state: AnalysisState
    ) -> AnalysisState:
        """Detect and analyze ``async with asyncio.TaskGroup() as tg`` or
        ``async with trio.open_nursery() as nursery``."""
        for item in node.items:
            func_name = _name_of(item.context_expr)
            if func_name is None and isinstance(item.context_expr, ast.Call):
                func_name = _name_of(item.context_expr.func)

            is_task_group = func_name in (
                "asyncio.TaskGroup",
                "trio.open_nursery",
                "anyio.create_task_group",
            )

            if is_task_group:
                tg_type = RefType(
                    "ν", OBJECT_TYPE, Pred.isinstance_("ν", "TaskGroup"),
                )
                var_name = self._target_name(item.optional_vars) if item.optional_vars else None
                if var_name is not None:
                    state = state.bind(var_name, tg_type)

        _, state = self._analyze_body(node.body, state)
        return state

    def analyze_gather(
        self, call: ast.Call, state: AnalysisState
    ) -> RefType:
        """Infer the result type of ``asyncio.gather(*coros)``."""
        elem_types: List[BaseTypeR] = []
        for arg in call.args:
            arg_type, state = self._analyze_expr(arg, state)
            if self._is_coroutine_type(arg_type):
                unwrapped = self._unwrap_coroutine(arg_type)
                elem_types.append(unwrapped.base)
            else:
                elem_types.append(arg_type.base)

        if not elem_types:
            result_base = BaseTypeR(BaseTypeKind.TUPLE)
        else:
            result_base = BaseTypeR(BaseTypeKind.TUPLE, type_args=tuple(elem_types))

        return RefType("ν", result_base, Pred.true_())

    def check_await_safety(
        self, expr: ast.expr, state: AnalysisState
    ) -> List[AsyncWarning]:
        """Check whether *expr* is safely awaitable in the current state."""
        warnings: List[AsyncWarning] = []

        if not state.in_async_function:
            warnings.append(AsyncWarning(
                level=WarningLevel.ERROR,
                message="'await' used outside of an async function",
                location=_node_loc(expr),
                node=expr,
            ))

        expr_type, _ = self._analyze_expr(expr, state)

        if not (self._is_coroutine_type(expr_type)
                or self._has_pred_marker(expr_type, "Awaitable")):
            name = _name_of(expr) or "<expr>"
            warnings.append(AsyncWarning(
                level=WarningLevel.ERROR,
                message=f"Object '{name}' is not awaitable",
                location=_node_loc(expr),
                node=expr,
            ))

        return warnings

    def detect_unawaited_coroutine(
        self, node: ast.expr, state: AnalysisState
    ) -> Optional[AsyncWarning]:
        """Warn when a coroutine call result is discarded without ``await``."""
        if not isinstance(node, ast.Call):
            return None

        result_type, _ = self._analyze_expr(node, state)
        if not self._is_coroutine_type(result_type):
            return None

        # If the call appears as a standalone ``Expr`` statement it is likely
        # un-awaited (the caller must check context).
        func_name = _name_of(node.func) if isinstance(node, ast.Call) else None
        label = func_name or "<coroutine>"
        return AsyncWarning(
            level=WarningLevel.WARNING,
            message=f"Coroutine '{label}' was never awaited",
            location=_node_loc(node),
            node=node,
        )

    def analyze_async_generator(
        self, func: ast.AsyncFunctionDef, state: AnalysisState
    ) -> RefType:
        """Infer yield/send types for an async generator function."""
        inner = state.copy()
        inner.in_async_function = True

        yield_types: List[RefType] = []
        send_type = RefType.trivial(NONE_TYPE)

        for node in ast.walk(func):
            if isinstance(node, ast.Yield):
                if node.value is not None:
                    yt, inner = self._analyze_expr(node.value, inner)
                    yield_types.append(yt)
                else:
                    yield_types.append(RefType.trivial(NONE_TYPE))
            elif isinstance(node, ast.YieldFrom):
                if node.value is not None:
                    yt, inner = self._analyze_expr(node.value, inner)
                    yield_types.append(yt)

        if yield_types:
            combined_base = self._join_bases([yt.base for yt in yield_types])
            yield_ref = RefType.trivial(combined_base)
        else:
            yield_ref = RefType.trivial(NONE_TYPE)

        coro = CoroutineType(
            yield_type=yield_ref,
            send_type=send_type,
            return_type=RefType.trivial(NONE_TYPE),
        )
        pred = Pred.isinstance_("ν", "AsyncGenerator")
        return RefType("ν", coro.as_base(), pred)

    # ------------------------------------------------------------------ #
    # Type predicates / extractors                                        #
    # ------------------------------------------------------------------ #

    def _is_coroutine_type(self, typ: RefType) -> bool:
        if typ.pred.op == PredOp.ISINSTANCE and len(typ.pred.args) >= 2:
            return typ.pred.args[1] in (_COROUTINE_MARKER, "Task")
        if typ.base.kind == BaseTypeKind.OBJECT and len(typ.base.type_args) == 3:
            return True
        return False

    def _unwrap_coroutine(self, typ: RefType) -> RefType:
        """Extract the return type from ``Coroutine[Y, S, R]``."""
        if typ.base.type_args and len(typ.base.type_args) >= 3:
            return_base = typ.base.type_args[2]
            return RefType.trivial(return_base)
        return RefType.trivial(ANY_TYPE)

    def _is_async_iterable(self, typ: RefType) -> bool:
        if self._has_pred_marker(typ, _ASYNC_ITER_MARKER):
            return True
        if self._has_pred_marker(typ, "AsyncIterable"):
            return True
        if self._has_pred_marker(typ, "AsyncGenerator"):
            return True
        return False

    def _extract_async_iter_element(self, typ: RefType) -> RefType:
        if typ.base.type_args:
            return RefType.trivial(typ.base.type_args[0])
        return RefType.trivial(ANY_TYPE)

    def _is_async_context_manager(self, typ: RefType) -> bool:
        if self._has_pred_marker(typ, _ASYNC_CM_MARKER):
            return True
        if self._has_pred_marker(typ, "ClientSession"):
            return True
        if self._has_pred_marker(typ, "Nursery"):
            return True
        if typ.base.kind == BaseTypeKind.OBJECT:
            return True  # conservative – objects might be CMs
        return False

    def _extract_aenter_type(self, typ: RefType) -> RefType:
        """Approximate the return type of ``__aenter__``."""
        if typ.base.type_args:
            return RefType.trivial(typ.base.type_args[0])
        return typ  # many CMs return *self*

    def _analyze_cancellation_scope(
        self, node: ast.With, state: AnalysisState
    ) -> AnalysisState:
        """Track cancellation-scope information (trio/anyio)."""
        for item in node.items:
            func_name = _name_of(item.context_expr)
            if func_name is None and isinstance(item.context_expr, ast.Call):
                func_name = _name_of(item.context_expr.func)

            if func_name in ("trio.CancelScope", "anyio.CancelScope",
                             "trio.move_on_after", "trio.fail_after",
                             "anyio.move_on_after", "anyio.fail_after"):
                scope_type = RefType("ν", OBJECT_TYPE, Pred.isinstance_("ν", "CancelScope"))
                var_name = (
                    self._target_name(item.optional_vars) if item.optional_vars else None
                )
                if var_name:
                    state = state.bind(var_name, scope_type)
                    # Inside a cancel scope, code may be cancelled
                    state = state.add_pred(Pred.hasattr_(var_name, "cancelled_caught"))

        _, state = self._analyze_body(node.body, state)
        return state

    def _track_task_creation(
        self, call: ast.Call, state: AnalysisState
    ) -> AnalysisState:
        """Track ``create_task`` / ``nursery.start_soon`` calls."""
        func_name = _name_of(call.func) if isinstance(call.func, (ast.Name, ast.Attribute)) else None
        if func_name is None:
            return state

        is_create_task = func_name in (
            "asyncio.create_task",
            "loop.create_task",
            "asyncio.ensure_future",
        )
        is_nursery_start = func_name.endswith(".start_soon") or func_name.endswith(".start")

        if not (is_create_task or is_nursery_start):
            return state

        if call.args:
            coro_type, state = self._analyze_expr(call.args[0], state)
        else:
            coro_type = RefType.trivial(ANY_TYPE)

        # Determine a label for the task
        task_label = f"task_{_node_loc(call)[0]}_{_node_loc(call)[1]}"

        current_group: Optional[str] = None
        for ctx in reversed(state.async_context_stack):
            if self._has_pred_marker(ctx.manager_type, "TaskGroup"):
                current_group = "TaskGroup"
                break
            if self._has_pred_marker(ctx.manager_type, "Nursery"):
                current_group = "Nursery"
                break

        info = TaskInfo(
            coroutine_type=coro_type,
            group=current_group,
            cancel_scope=None,
            creation_site=_node_loc(call),
        )
        state = state.copy()
        state.active_tasks[task_label] = info
        return state

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _has_pred_marker(self, typ: RefType, marker: str) -> bool:
        """Check whether the predicate mentions an ``isinstance`` marker."""
        return self._pred_mentions(typ.pred, marker)

    def _pred_mentions(self, p: Pred, marker: str) -> bool:
        if p.op == PredOp.ISINSTANCE and len(p.args) >= 2 and p.args[1] == marker:
            return True
        for child in p.children:
            if self._pred_mentions(child, marker):
                return True
        return False

    def _unwrap_awaitable(self, typ: RefType) -> RefType:
        if typ.base.type_args:
            return RefType.trivial(typ.base.type_args[0])
        return RefType.trivial(ANY_TYPE)

    def _analyze_expr(
        self, node: ast.expr, state: AnalysisState
    ) -> Tuple[RefType, AnalysisState]:
        """Best-effort type inference for an expression AST node."""
        if isinstance(node, ast.Name):
            return state.get_type(node.id), state

        if isinstance(node, ast.Constant):
            return self._const_type(node.value), state

        if isinstance(node, ast.Await):
            return self.analyze_await(node, state)

        if isinstance(node, ast.Call):
            return self._analyze_call(node, state)

        if isinstance(node, ast.Attribute):
            base_type, state = self._analyze_expr(node.value, state)
            dotted = _name_of(node)
            if dotted is not None:
                lib_type = _resolve_library_type(dotted)
                if lib_type is not None:
                    return lib_type, state
            return RefType.trivial(ANY_TYPE), state

        if isinstance(node, ast.ListComp):
            return RefType.trivial(BaseTypeR(BaseTypeKind.LIST)), state

        if isinstance(node, ast.SetComp):
            return RefType.trivial(BaseTypeR(BaseTypeKind.SET)), state

        if isinstance(node, ast.DictComp):
            return RefType.trivial(BaseTypeR(BaseTypeKind.DICT)), state

        if isinstance(node, ast.Tuple):
            return RefType.trivial(BaseTypeR(BaseTypeKind.TUPLE)), state

        return RefType.trivial(ANY_TYPE), state

    def _analyze_call(
        self, call: ast.Call, state: AnalysisState
    ) -> Tuple[RefType, AnalysisState]:
        """Infer the type of a function call."""
        func_name = _name_of(call.func) if isinstance(call.func, (ast.Name, ast.Attribute)) else None

        # Try library patterns first
        if func_name is not None:
            lib_type = _resolve_library_type(func_name)
            if lib_type is not None:
                state = self._track_task_creation(call, state)
                return lib_type, state

        # asyncio.gather special handling
        if func_name in ("asyncio.gather",):
            result = self.analyze_gather(call, state)
            return result, state

        # Known asyncio functions
        if func_name == "asyncio.create_task":
            state = self._track_task_creation(call, state)
            return RefType("ν", OBJECT_TYPE, Pred.isinstance_("ν", "Task")), state

        if func_name == "asyncio.wait_for":
            if call.args:
                inner_type, state = self._analyze_expr(call.args[0], state)
                if self._is_coroutine_type(inner_type):
                    return self._unwrap_coroutine(inner_type), state
            return RefType.trivial(ANY_TYPE), state

        # If the callee is a known coroutine-typed variable, return coroutine
        if func_name is not None and func_name in state.var_types:
            callee_type = state.var_types[func_name]
            if callee_type.base.kind == BaseTypeKind.CALLABLE:
                ret = callee_type.base.return_type
                if ret is not None:
                    return RefType.trivial(ret), state

        return RefType.trivial(ANY_TYPE), state

    def _analyze_body(
        self, stmts: Sequence[ast.stmt], state: AnalysisState
    ) -> Tuple[RefType, AnalysisState]:
        """Walk a statement list, return the inferred return type."""
        return_type: RefType = RefType.trivial(NONE_TYPE)

        for stmt in stmts:
            if isinstance(stmt, ast.Return):
                if stmt.value is not None:
                    ret, state = self._analyze_expr(stmt.value, state)
                    return_type = ret
                else:
                    return_type = RefType.trivial(NONE_TYPE)

            elif isinstance(stmt, ast.Assign):
                if stmt.targets and isinstance(stmt.targets[0], ast.Name):
                    val_type, state = self._analyze_expr(stmt.value, state)
                    state = state.bind(stmt.targets[0].id, val_type)

            elif isinstance(stmt, ast.AnnAssign):
                if stmt.target and isinstance(stmt.target, ast.Name):
                    ann_type = (
                        self._annotation_to_reftype(stmt.annotation)
                        if stmt.annotation
                        else RefType.trivial(ANY_TYPE)
                    )
                    if stmt.value is not None:
                        val_type, state = self._analyze_expr(stmt.value, state)
                        state = state.bind(stmt.target.id, val_type)
                    else:
                        state = state.bind(stmt.target.id, ann_type)

            elif isinstance(stmt, ast.Expr):
                # Standalone expression – check for unawaited coroutines
                if isinstance(stmt.value, ast.Call):
                    warn = self.detect_unawaited_coroutine(stmt.value, state)
                    if warn is not None:
                        state.warnings.append(warn)
                _, state = self._analyze_expr(stmt.value, state)

            elif isinstance(stmt, (ast.AsyncFor,)):
                state = self.analyze_async_for(stmt, state)

            elif isinstance(stmt, (ast.AsyncWith,)):
                state = self.analyze_async_with(stmt, state)

            elif isinstance(stmt, ast.With):
                state = self.analyze_task_group(stmt, state)
                state = self._analyze_cancellation_scope(stmt, state)

            elif isinstance(stmt, ast.If):
                _, state = self._analyze_body(stmt.body, state)
                if stmt.orelse:
                    _, state = self._analyze_body(stmt.orelse, state)

            elif isinstance(stmt, (ast.For, ast.While)):
                _, state = self._analyze_body(stmt.body, state)

            elif isinstance(stmt, ast.Try):
                _, state = self._analyze_body(stmt.body, state)
                for handler in stmt.handlers:
                    if handler.name:
                        exc_type = RefType("ν", OBJECT_TYPE, Pred.isinstance_("ν", "BaseException"))
                        state = state.bind(handler.name, exc_type)
                    _, state = self._analyze_body(handler.body, state)
                if stmt.orelse:
                    _, state = self._analyze_body(stmt.orelse, state)
                if stmt.finalbody:
                    _, state = self._analyze_body(stmt.finalbody, state)

        return return_type, state

    def _const_type(self, value: object) -> RefType:
        if value is None:
            return RefType.trivial(NONE_TYPE)
        if isinstance(value, bool):
            return RefType.trivial(BOOL_TYPE)
        if isinstance(value, int):
            return RefType.trivial(INT_TYPE)
        if isinstance(value, float):
            return RefType.trivial(FLOAT_TYPE)
        if isinstance(value, str):
            return RefType.trivial(STR_TYPE)
        return RefType.trivial(ANY_TYPE)

    def _annotation_to_reftype(self, ann: Optional[ast.expr]) -> RefType:
        """Best-effort mapping from an annotation AST to a RefType."""
        if ann is None:
            return RefType.trivial(ANY_TYPE)
        name = _name_of(ann)
        _SIMPLE: Dict[str, BaseTypeR] = {
            "int": INT_TYPE, "float": FLOAT_TYPE, "str": STR_TYPE,
            "bool": BOOL_TYPE, "None": NONE_TYPE, "list": BaseTypeR(BaseTypeKind.LIST),
            "dict": BaseTypeR(BaseTypeKind.DICT), "set": BaseTypeR(BaseTypeKind.SET),
            "tuple": BaseTypeR(BaseTypeKind.TUPLE), "object": OBJECT_TYPE,
            "Any": ANY_TYPE,
        }
        if name is not None and name in _SIMPLE:
            return RefType.trivial(_SIMPLE[name])
        if name is not None and name.startswith("Coroutine"):
            return RefType("ν", OBJECT_TYPE, Pred.isinstance_("ν", "Coroutine"))
        return RefType.trivial(ANY_TYPE)

    def _target_name(self, target: ast.expr) -> Optional[str]:
        if isinstance(target, ast.Name):
            return target.id
        return None

    def _body_has_yield(self, body: Sequence[ast.stmt]) -> bool:
        for node in ast.walk(ast.Module(body=list(body), type_ignores=[])):
            if isinstance(node, (ast.Yield, ast.YieldFrom)):
                return True
        return False

    def _join_bases(self, bases: List[BaseTypeR]) -> BaseTypeR:
        """Compute a simple join over base types."""
        if not bases:
            return NEVER_TYPE
        kinds = {b.kind for b in bases}
        if len(kinds) == 1:
            return bases[0]
        # Numeric widening
        if kinds <= {BaseTypeKind.INT, BaseTypeKind.FLOAT, BaseTypeKind.BOOL}:
            return FLOAT_TYPE
        return ANY_TYPE
