from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Type, Union
)
from enum import Enum, auto
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Local Type Definitions
# ---------------------------------------------------------------------------

class TypeTag(Enum):
    INT = auto()
    FLOAT = auto()
    STR = auto()
    BOOL = auto()
    NONE = auto()
    LIST = auto()
    DICT = auto()
    SET = auto()
    TUPLE = auto()
    CALLABLE = auto()
    OBJECT = auto()
    ANY = auto()
    BYTES = auto()


class NullityTag(Enum):
    DEFINITELY_NULL = auto()
    DEFINITELY_NOT_NULL = auto()
    MAYBE_NULL = auto()


@dataclass
class Interval:
    lo: Optional[float] = None
    hi: Optional[float] = None

    def contains(self, value: float) -> bool:
        if self.lo is not None and value < self.lo:
            return False
        if self.hi is not None and value > self.hi:
            return False
        return True

    def overlaps(self, other: Interval) -> bool:
        if self.lo is not None and other.hi is not None and self.lo > other.hi:
            return False
        if self.hi is not None and other.lo is not None and self.hi < other.lo:
            return False
        return True

    def intersect(self, other: Interval) -> Interval:
        lo = max(self.lo, other.lo) if self.lo is not None and other.lo is not None else (self.lo or other.lo)
        hi = min(self.hi, other.hi) if self.hi is not None and other.hi is not None else (self.hi or other.hi)
        return Interval(lo=lo, hi=hi)

    def union(self, other: Interval) -> Interval:
        lo = min(self.lo, other.lo) if self.lo is not None and other.lo is not None else None
        hi = max(self.hi, other.hi) if self.hi is not None and other.hi is not None else None
        return Interval(lo=lo, hi=hi)

    def is_empty(self) -> bool:
        if self.lo is not None and self.hi is not None:
            return self.lo > self.hi
        return False

    def width(self) -> Optional[float]:
        if self.lo is not None and self.hi is not None:
            return self.hi - self.lo
        return None


@dataclass
class RefinementType:
    base_type: str
    predicate: Optional[str] = None
    interval: Optional[Interval] = None
    nullity: NullityTag = NullityTag.MAYBE_NULL

    def is_subtype_of(self, other: RefinementType) -> bool:
        if other.base_type == "Any":
            return True
        if self.base_type != other.base_type:
            return False
        if other.interval is not None and self.interval is not None:
            if not other.interval.contains(self.interval.lo or float("-inf")):
                return False
            if not other.interval.contains(self.interval.hi or float("inf")):
                return False
        if self.nullity == NullityTag.MAYBE_NULL and other.nullity == NullityTag.DEFINITELY_NOT_NULL:
            return False
        return True

    def meet(self, other: RefinementType) -> RefinementType:
        base = self.base_type if self.base_type == other.base_type else "Never"
        interval = None
        if self.interval and other.interval:
            interval = self.interval.intersect(other.interval)
        elif self.interval:
            interval = self.interval
        elif other.interval:
            interval = other.interval
        pred = None
        if self.predicate and other.predicate:
            pred = f"({self.predicate}) and ({other.predicate})"
        elif self.predicate:
            pred = self.predicate
        elif other.predicate:
            pred = other.predicate
        nullity = self.nullity
        if other.nullity == NullityTag.DEFINITELY_NOT_NULL or self.nullity == NullityTag.DEFINITELY_NOT_NULL:
            nullity = NullityTag.DEFINITELY_NOT_NULL
        return RefinementType(base_type=base, predicate=pred, interval=interval, nullity=nullity)


@dataclass
class ExceptionInfo:
    exception_type: str
    message: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    cause: Optional[ExceptionInfo] = None
    context: Optional[ExceptionInfo] = None

    def with_cause(self, cause: ExceptionInfo) -> ExceptionInfo:
        return ExceptionInfo(
            exception_type=self.exception_type,
            message=self.message,
            attributes=dict(self.attributes),
            cause=cause,
            context=self.context,
        )

    def with_context(self, context: ExceptionInfo) -> ExceptionInfo:
        return ExceptionInfo(
            exception_type=self.exception_type,
            message=self.message,
            attributes=dict(self.attributes),
            cause=self.cause,
            context=context,
        )


@dataclass
class ExceptionState:
    possible_exceptions: List[ExceptionInfo] = field(default_factory=list)
    definitely_raises: bool = False
    caught_exceptions: List[ExceptionInfo] = field(default_factory=list)

    def add_exception(self, exc: ExceptionInfo) -> None:
        self.possible_exceptions.append(exc)

    def mark_caught(self, exc: ExceptionInfo) -> None:
        self.caught_exceptions.append(exc)
        self.possible_exceptions = [
            e for e in self.possible_exceptions
            if e.exception_type != exc.exception_type
        ]

    def is_clean(self) -> bool:
        return len(self.possible_exceptions) == 0


@dataclass
class GuardCondition:
    kind: str
    variable: str
    args: List[Any] = field(default_factory=list)

    def to_source(self) -> str:
        if self.kind == "isinstance":
            types = ", ".join(str(a) for a in self.args)
            return f"isinstance({self.variable}, ({types},))"
        elif self.kind == "comparison":
            parts = []
            for op, val in self.args:
                parts.append(f"{self.variable} {op} {val}")
            return " and ".join(parts)
        elif self.kind == "contains":
            container = self.args[0] if self.args else "container"
            return f"{self.variable} in {container}"
        elif self.kind == "hasattr":
            attr = self.args[0] if self.args else "attr"
            return f"hasattr({self.variable}, {attr!r})"
        elif self.kind == "predicate":
            func = self.args[0] if self.args else "check"
            return f"{func}({self.variable})"
        elif self.kind == "not_none":
            return f"{self.variable} is not None"
        elif self.kind == "truthy":
            return f"bool({self.variable})"
        else:
            return f"{self.kind}({self.variable}, {self.args})"


@dataclass
class AbstractState:
    variables: Dict[str, RefinementType] = field(default_factory=dict)
    exception_state: ExceptionState = field(default_factory=ExceptionState)
    reachable: bool = True

    def copy(self) -> AbstractState:
        return AbstractState(
            variables={k: copy.deepcopy(v) for k, v in self.variables.items()},
            exception_state=copy.deepcopy(self.exception_state),
            reachable=self.reachable,
        )

    def set_variable(self, name: str, typ: RefinementType) -> None:
        self.variables[name] = typ

    def get_variable(self, name: str) -> Optional[RefinementType]:
        return self.variables.get(name)

    def mark_unreachable(self) -> None:
        self.reachable = False


@dataclass
class HandlerMatch:
    handler_index: int
    exception_types: List[str] = field(default_factory=list)
    binds_as: Optional[str] = None
    body_label: str = ""


@dataclass
class ResourceInfo:
    resource_type: str
    variable: str
    acquired_at: str
    released: bool = False


@dataclass
class SafetyViolation:
    kind: str
    description: str
    location: str
    severity: str  # "error", "warning", "info"


# ---------------------------------------------------------------------------
# 1. ExceptionHierarchy
# ---------------------------------------------------------------------------

class ExceptionHierarchy:
    """Models the complete Python exception hierarchy as a tree structure.

    Provides methods to query subtype relationships, method resolution order,
    common bases, and exception-specific attributes. The hierarchy faithfully
    mirrors CPython's built-in exception tree.
    """

    def __init__(self) -> None:
        # parent_map: child -> parent
        self._parent: Dict[str, Optional[str]] = {}
        # children_map: parent -> list of children
        self._children: Dict[str, List[str]] = {}
        # attributes per exception type
        self._attributes: Dict[str, Dict[str, str]] = {}
        self._build_hierarchy()
        self._build_attributes()

    # ---- construction -------------------------------------------------------

    def _add(self, child: str, parent: Optional[str]) -> None:
        self._parent[child] = parent
        self._children.setdefault(child, [])
        if parent is not None:
            self._children.setdefault(parent, [])
            if child not in self._children[parent]:
                self._children[parent].append(child)

    def _build_hierarchy(self) -> None:
        self._add("BaseException", None)
        # direct children of BaseException
        for c in ("SystemExit", "KeyboardInterrupt", "GeneratorExit", "Exception"):
            self._add(c, "BaseException")

        # --- Exception direct children (alphabetical in CPython) ---
        exception_children = [
            "ArithmeticError", "AssertionError", "AttributeError",
            "BlockingIOError", "BrokenPipeError", "BufferError",
            "ChildProcessError", "ConnectionError", "EOFError",
            "FileExistsError", "FileNotFoundError",
            "ImportError", "InterruptedError", "IsADirectoryError",
            "KeyError", "LookupError", "MemoryError",
            "NameError", "NotADirectoryError", "NotImplementedError",
            "OSError", "OverflowError", "PermissionError",
            "ProcessLookupError", "RecursionError", "ReferenceError",
            "RuntimeError", "StopAsyncIteration", "StopIteration",
            "SyntaxError", "SystemError", "TimeoutError", "TypeError",
            "UnicodeError", "ValueError", "Warning",
        ]
        for c in exception_children:
            self._add(c, "Exception")

        # ArithmeticError subtree
        for c in ("FloatingPointError", "OverflowError", "ZeroDivisionError"):
            self._add(c, "ArithmeticError")

        # ConnectionError subtree
        for c in ("BrokenPipeError", "ConnectionAbortedError",
                   "ConnectionRefusedError", "ConnectionResetError"):
            self._add(c, "ConnectionError")

        # ImportError subtree
        self._add("ModuleNotFoundError", "ImportError")

        # LookupError subtree
        for c in ("IndexError", "KeyError"):
            self._add(c, "LookupError")

        # NameError subtree
        self._add("UnboundLocalError", "NameError")

        # OSError subtree
        for c in ("BlockingIOError", "ChildProcessError", "ConnectionError",
                   "FileExistsError", "FileNotFoundError", "InterruptedError",
                   "IsADirectoryError", "NotADirectoryError", "PermissionError",
                   "ProcessLookupError", "TimeoutError"):
            self._add(c, "OSError")

        # RuntimeError subtree
        for c in ("NotImplementedError", "RecursionError"):
            self._add(c, "RuntimeError")

        # SyntaxError subtree
        self._add("IndentationError", "SyntaxError")
        self._add("TabError", "IndentationError")

        # UnicodeError subtree
        for c in ("UnicodeDecodeError", "UnicodeEncodeError", "UnicodeTranslateError"):
            self._add(c, "UnicodeError")

        # ValueError subtree  (UnicodeError is also under ValueError)
        self._add("UnicodeError", "ValueError")

        # Warning subtree
        for c in ("BytesWarning", "DeprecationWarning", "FutureWarning",
                   "ImportWarning", "PendingDeprecationWarning",
                   "ResourceWarning", "RuntimeWarning", "SyntaxWarning",
                   "UnicodeWarning", "UserWarning"):
            self._add(c, "Warning")

    def _build_attributes(self) -> None:
        base = {"args": "Tuple[Any, ...]", "__traceback__": "Optional[TracebackType]",
                "__cause__": "Optional[BaseException]", "__context__": "Optional[BaseException]",
                "__suppress_context__": "bool"}
        self._attributes["BaseException"] = dict(base)
        self._attributes["Exception"] = dict(base)

        self._attributes["SystemExit"] = {**base, "code": "Optional[int]"}
        self._attributes["KeyboardInterrupt"] = dict(base)
        self._attributes["GeneratorExit"] = dict(base)
        self._attributes["StopIteration"] = {**base, "value": "Any"}
        self._attributes["StopAsyncIteration"] = {**base, "value": "Any"}

        self._attributes["OSError"] = {**base, "errno": "Optional[int]",
                                        "strerror": "Optional[str]",
                                        "filename": "Optional[str]",
                                        "filename2": "Optional[str]"}
        for sub in ("BlockingIOError", "ChildProcessError", "ConnectionError",
                     "FileExistsError", "FileNotFoundError", "InterruptedError",
                     "IsADirectoryError", "NotADirectoryError", "PermissionError",
                     "ProcessLookupError", "TimeoutError", "BrokenPipeError",
                     "ConnectionAbortedError", "ConnectionRefusedError",
                     "ConnectionResetError"):
            self._attributes[sub] = dict(self._attributes["OSError"])

        self._attributes["BlockingIOError"]["characters_written"] = "int"

        self._attributes["ImportError"] = {**base, "name": "Optional[str]",
                                            "path": "Optional[str]"}
        self._attributes["ModuleNotFoundError"] = dict(self._attributes["ImportError"])

        self._attributes["SyntaxError"] = {**base, "filename": "Optional[str]",
                                            "lineno": "Optional[int]",
                                            "offset": "Optional[int]",
                                            "text": "Optional[str]",
                                            "end_lineno": "Optional[int]",
                                            "end_offset": "Optional[int]",
                                            "msg": "str"}
        self._attributes["IndentationError"] = dict(self._attributes["SyntaxError"])
        self._attributes["TabError"] = dict(self._attributes["SyntaxError"])

        self._attributes["UnicodeError"] = {**base, "encoding": "str",
                                             "reason": "str"}
        self._attributes["UnicodeDecodeError"] = {**self._attributes["UnicodeError"],
                                                    "object": "bytes",
                                                    "start": "int", "end": "int"}
        self._attributes["UnicodeEncodeError"] = {**self._attributes["UnicodeError"],
                                                    "object": "str",
                                                    "start": "int", "end": "int"}
        self._attributes["UnicodeTranslateError"] = {**self._attributes["UnicodeError"],
                                                      "object": "str",
                                                      "start": "int", "end": "int"}

        self._attributes["LookupError"] = dict(base)
        self._attributes["IndexError"] = dict(base)
        self._attributes["KeyError"] = dict(base)

        self._attributes["AttributeError"] = {**base, "name": "Optional[str]",
                                               "obj": "Optional[Any]"}
        self._attributes["NameError"] = {**base, "name": "Optional[str]"}
        self._attributes["UnboundLocalError"] = dict(self._attributes["NameError"])

        # Warnings
        for w in ("BytesWarning", "DeprecationWarning", "FutureWarning",
                   "ImportWarning", "PendingDeprecationWarning",
                   "ResourceWarning", "RuntimeWarning", "SyntaxWarning",
                   "UnicodeWarning", "UserWarning", "Warning"):
            self._attributes[w] = dict(base)

        # Everything else inherits base
        for exc in self._parent:
            if exc not in self._attributes:
                self._attributes[exc] = dict(base)

    # ---- public API ---------------------------------------------------------

    def is_subtype(self, child: str, parent: str) -> bool:
        """Return True if *child* is the same as or a subtype of *parent*."""
        if child == parent:
            return True
        mro = self.get_mro(child)
        return parent in mro

    def get_parent(self, exc_type: str) -> Optional[str]:
        """Return the immediate parent of *exc_type*, or None for BaseException."""
        return self._parent.get(exc_type)

    def get_all_subtypes(self, exc_type: str) -> Set[str]:
        """Return the set of all (transitive) subtypes of *exc_type*."""
        result: Set[str] = set()
        stack = list(self._children.get(exc_type, []))
        while stack:
            current = stack.pop()
            if current not in result:
                result.add(current)
                stack.extend(self._children.get(current, []))
        return result

    def get_mro(self, exc_type: str) -> List[str]:
        """Return the method resolution order (list from *exc_type* up to BaseException)."""
        mro: List[str] = []
        current: Optional[str] = exc_type
        visited: Set[str] = set()
        while current is not None and current not in visited:
            visited.add(current)
            mro.append(current)
            current = self._parent.get(current)
        return mro

    def get_common_base(self, type1: str, type2: str) -> str:
        """Return the nearest common ancestor of *type1* and *type2*."""
        mro1 = self.get_mro(type1)
        mro2_set = set(self.get_mro(type2))
        for ancestor in mro1:
            if ancestor in mro2_set:
                return ancestor
        return "BaseException"

    def get_exception_attributes(self, exc_type: str) -> Dict[str, str]:
        """Return the attribute dict for *exc_type* (inheriting from parents)."""
        if exc_type in self._attributes:
            return dict(self._attributes[exc_type])
        # walk up the MRO until we find attributes
        for ancestor in self.get_mro(exc_type):
            if ancestor in self._attributes:
                return dict(self._attributes[ancestor])
        return {"args": "Tuple[Any, ...]"}

    def is_concrete(self, exc_type: str) -> bool:
        """Return True if *exc_type* is a concrete (leaf or commonly-instantiated) exception."""
        # In Python every built-in exception is concrete; we treat types with
        # children as abstract only when they are used purely as grouping types.
        abstract_groups = {"ArithmeticError", "LookupError", "OSError",
                           "ConnectionError", "Warning", "BaseException", "Exception"}
        return exc_type not in abstract_groups

    def get_all_exceptions(self) -> List[str]:
        """Return a sorted list of every exception in the hierarchy."""
        return sorted(self._parent.keys())

    def get_depth(self, exc_type: str) -> int:
        """Return the depth of *exc_type* in the tree (BaseException = 0)."""
        depth = 0
        current: Optional[str] = exc_type
        while current is not None:
            parent = self._parent.get(current)
            if parent is None:
                break
            depth += 1
            current = parent
        return depth


# ---------------------------------------------------------------------------
# 2. ExceptionModel / ExceptionModelRegistry
# ---------------------------------------------------------------------------

@dataclass
class ExceptionModel:
    """Describes one exception type: when it fires, how to guard against it,
    what attributes it carries, and strategies for recovery."""

    exception_type: str
    parent_type: str
    trigger_conditions: List[str] = field(default_factory=list)
    prevention_guards: List[GuardCondition] = field(default_factory=list)
    attributes: Dict[str, str] = field(default_factory=dict)
    common_causes: List[str] = field(default_factory=list)
    recovery_strategies: List[str] = field(default_factory=list)

    # ---- methods -----------------------------------------------------------

    def can_be_raised_by(self, operation: str, operand_types: List[str]) -> bool:
        """Heuristically decide whether *operation* on *operand_types* can raise
        this exception."""
        op_lower = operation.lower()
        type_strs = [t.lower() for t in operand_types]

        rules: Dict[str, Callable[[], bool]] = {
            "TypeError": lambda: self._check_type_error(op_lower, type_strs),
            "ValueError": lambda: self._check_value_error(op_lower, type_strs),
            "KeyError": lambda: "dict" in type_strs and op_lower in ("getitem", "subscript", "__getitem__", "[]", "pop"),
            "IndexError": lambda: any(t in ("list", "tuple", "str", "bytes") for t in type_strs) and op_lower in ("getitem", "subscript", "__getitem__", "[]"),
            "AttributeError": lambda: op_lower in ("getattr", ".", "attribute_access"),
            "NameError": lambda: op_lower in ("name_lookup", "load_name", "load_global"),
            "ZeroDivisionError": lambda: op_lower in ("div", "truediv", "floordiv", "mod", "/", "//", "%", "__truediv__", "__floordiv__", "__mod__"),
            "FileNotFoundError": lambda: op_lower in ("open", "read_file", "load", "os.open"),
            "PermissionError": lambda: op_lower in ("open", "write_file", "os.open", "os.mkdir", "os.remove"),
            "ImportError": lambda: op_lower in ("import", "__import__", "import_module"),
            "ModuleNotFoundError": lambda: op_lower in ("import", "__import__", "import_module"),
            "StopIteration": lambda: op_lower in ("next", "__next__", "send"),
            "RuntimeError": lambda: op_lower in ("call", "invoke", "generator_throw"),
            "RecursionError": lambda: op_lower in ("call", "invoke", "__call__"),
            "MemoryError": lambda: op_lower in ("alloc", "malloc", "list_extend", "bytearray"),
            "OverflowError": lambda: op_lower in ("exp", "pow", "**", "math.exp", "math.pow"),
            "NotImplementedError": lambda: op_lower in ("call", "invoke", "__call__"),
            "OSError": lambda: op_lower in ("open", "read", "write", "os.open", "os.stat", "os.listdir", "socket"),
            "UnicodeDecodeError": lambda: op_lower in ("decode", "bytes.decode", "str"),
            "UnicodeEncodeError": lambda: op_lower in ("encode", "str.encode", "bytes"),
            "ConnectionError": lambda: op_lower in ("connect", "send", "recv", "socket.connect"),
            "ConnectionRefusedError": lambda: op_lower in ("connect", "socket.connect"),
            "ConnectionResetError": lambda: op_lower in ("send", "recv", "read", "write"),
            "BrokenPipeError": lambda: op_lower in ("write", "send", "flush"),
            "TimeoutError": lambda: op_lower in ("connect", "recv", "send", "socket", "urlopen"),
            "EOFError": lambda: op_lower in ("input", "raw_input", "read", "readline"),
            "BufferError": lambda: op_lower in ("buffer", "memoryview", "export"),
        }
        checker = rules.get(self.exception_type)
        if checker is not None:
            return checker()
        return False

    def _check_type_error(self, op: str, types: List[str]) -> bool:
        mixed_numeric_str = ("str" in types and any(t in ("int", "float") for t in types))
        if op in ("add", "+", "__add__") and mixed_numeric_str:
            return True
        if op in ("call", "__call__", "invoke") and "nonetype" in types:
            return True
        if op in ("subscript", "[]", "__getitem__") and "nonetype" in types:
            return True
        if op in ("iter", "__iter__", "for") and any(t in ("int", "float", "nonetype") for t in types):
            return True
        return False

    def _check_value_error(self, op: str, types: List[str]) -> bool:
        if op in ("int", "float", "complex") and "str" in types:
            return True
        if op in ("unpack", "starred_unpack"):
            return True
        if op in ("index", "list.index", "str.index"):
            return True
        return False

    def get_prevention_guard(self, operation: str) -> Optional[GuardCondition]:
        """Return a guard that would prevent this exception for *operation*,
        or None if no guard is known."""
        if not self.prevention_guards:
            return None
        op_lower = operation.lower()
        # try to find operation-specific guard
        for g in self.prevention_guards:
            if op_lower in g.kind or op_lower in str(g.args):
                return g
        return self.prevention_guards[0]

    def get_refined_state_after_catch(
        self, state: AbstractState, bind_var: Optional[str]
    ) -> AbstractState:
        """Return the abstract state inside an ``except`` clause that caught
        this exception type, optionally binding to *bind_var*."""
        new_state = state.copy()
        if bind_var is not None:
            attrs_type = ", ".join(f"{k}: {v}" for k, v in self.attributes.items())
            new_state.set_variable(
                bind_var,
                RefinementType(
                    base_type=self.exception_type,
                    predicate=f"isinstance({bind_var}, {self.exception_type})",
                    nullity=NullityTag.DEFINITELY_NOT_NULL,
                ),
            )
        # mark that the exception has been caught
        new_state.exception_state.caught_exceptions.append(
            ExceptionInfo(exception_type=self.exception_type, message="<caught>")
        )
        new_state.exception_state.possible_exceptions = [
            e for e in new_state.exception_state.possible_exceptions
            if e.exception_type != self.exception_type
        ]
        return new_state


class ExceptionModelRegistry:
    """Registry of ``ExceptionModel`` instances for every major built-in
    exception type, populated with real trigger conditions, prevention guards,
    common causes, and recovery strategies."""

    def __init__(self) -> None:
        self._models: Dict[str, ExceptionModel] = {}
        self._hierarchy = ExceptionHierarchy()
        self._build_all_models()

    def get(self, exc_type: str) -> Optional[ExceptionModel]:
        return self._models.get(exc_type)

    def get_or_default(self, exc_type: str) -> ExceptionModel:
        if exc_type in self._models:
            return self._models[exc_type]
        parent = self._hierarchy.get_parent(exc_type) or "Exception"
        return ExceptionModel(
            exception_type=exc_type,
            parent_type=parent,
            attributes=self._hierarchy.get_exception_attributes(exc_type),
        )

    def all_models(self) -> List[ExceptionModel]:
        return list(self._models.values())

    def _register(self, model: ExceptionModel) -> None:
        self._models[model.exception_type] = model

    # ---- model construction -------------------------------------------------

    def _build_all_models(self) -> None:
        self._build_type_error()
        self._build_value_error()
        self._build_key_error()
        self._build_index_error()
        self._build_attribute_error()
        self._build_name_error()
        self._build_unbound_local_error()
        self._build_zero_division_error()
        self._build_file_not_found_error()
        self._build_permission_error()
        self._build_import_error()
        self._build_module_not_found_error()
        self._build_stop_iteration()
        self._build_stop_async_iteration()
        self._build_runtime_error()
        self._build_recursion_error()
        self._build_memory_error()
        self._build_overflow_error()
        self._build_not_implemented_error()
        self._build_os_error()
        self._build_unicode_decode_error()
        self._build_unicode_encode_error()
        self._build_unicode_translate_error()
        self._build_connection_error()
        self._build_connection_refused_error()
        self._build_connection_reset_error()
        self._build_connection_aborted_error()
        self._build_broken_pipe_error()
        self._build_timeout_error()
        self._build_eof_error()
        self._build_buffer_error()
        self._build_assertion_error()
        self._build_floating_point_error()
        self._build_reference_error()
        self._build_system_error()
        self._build_blocking_io_error()
        self._build_child_process_error()
        self._build_interrupted_error()
        self._build_is_a_directory_error()
        self._build_not_a_directory_error()
        self._build_process_lookup_error()
        self._build_file_exists_error()
        self._build_syntax_error()
        self._build_indentation_error()
        self._build_tab_error()
        self._build_lookup_error()
        self._build_arithmetic_error()

    def _build_type_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="TypeError",
            parent_type="Exception",
            trigger_conditions=[
                "Binary operation on incompatible types (e.g., str + int)",
                "Calling a non-callable object",
                "Wrong number of arguments to a function",
                "Subscripting a non-subscriptable object",
                "Iterating over a non-iterable",
                "Unpacking a non-iterable",
                "Using unhashable type as dict key or set element",
                "Passing wrong argument type to a built-in function",
                "Descriptor __get__/__set__ receives wrong owner type",
            ],
            prevention_guards=[
                GuardCondition("isinstance", "x", ["expected_type"]),
                GuardCondition("predicate", "x", ["callable"]),
            ],
            attributes={"args": "Tuple[Any, ...]"},
            common_causes=[
                "'hello' + 42",
                "None()",
                "len(42)",
                "for x in 42: pass",
                "{[1,2]: 'val'}",
                "func(a, b, c)  # wrong arity",
            ],
            recovery_strategies=[
                "Check types with isinstance() before operation",
                "Convert operands to compatible types (str(x), int(x))",
                "Use typing or runtime type checks",
            ],
        ))

    def _build_value_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="ValueError",
            parent_type="Exception",
            trigger_conditions=[
                "int() or float() on non-numeric string",
                "Unpacking with wrong number of values",
                "list.remove(x) where x not in list",
                "math.sqrt of negative number",
                "Invalid literal for numeric conversion",
                "datetime with invalid date components",
            ],
            prevention_guards=[
                GuardCondition("predicate", "x", ["str.isdigit"]),
                GuardCondition("predicate", "x", ["validate_format"]),
            ],
            attributes={"args": "Tuple[Any, ...]"},
            common_causes=[
                "int('abc')",
                "float('not_a_number')",
                "a, b = [1, 2, 3]",
                "[1,2,3].remove(4)",
                "math.sqrt(-1)",
            ],
            recovery_strategies=[
                "Validate input before conversion",
                "Use try/except around conversion",
                "Check value ranges before operations",
            ],
        ))

    def _build_key_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="KeyError",
            parent_type="LookupError",
            trigger_conditions=[
                "Accessing a dict with a key that does not exist",
                "dict.__getitem__ with missing key",
                "dict.pop(key) with missing key and no default",
                "Accessing os.environ with missing key",
            ],
            prevention_guards=[
                GuardCondition("contains", "key", ["dict"]),
                GuardCondition("predicate", "key", ["dict.get"]),
            ],
            attributes={"args": "Tuple[Any, ...]"},
            common_causes=[
                "d = {}; d['missing']",
                "os.environ['UNDEFINED_VAR']",
                "collections.Counter()[key]  # actually returns 0, not KeyError",
            ],
            recovery_strategies=[
                "Use dict.get(key, default)",
                "Check 'key in dict' before access",
                "Use collections.defaultdict",
                "Use try/except KeyError",
            ],
        ))

    def _build_index_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="IndexError",
            parent_type="LookupError",
            trigger_conditions=[
                "List/tuple index out of range",
                "Accessing seq[i] where i >= len(seq) or i < -len(seq)",
                "pop() from empty list",
            ],
            prevention_guards=[
                GuardCondition("comparison", "i", [(">=", 0), ("<", "len(seq)")]),
                GuardCondition("predicate", "seq", ["len(seq) > 0"]),
            ],
            attributes={"args": "Tuple[Any, ...]"},
            common_causes=[
                "lst = [1,2,3]; lst[5]",
                "[].pop()",
                "for i in range(10): lst[i]  # if lst has < 10 elements",
            ],
            recovery_strategies=[
                "Check index bounds before access",
                "Use try/except IndexError",
                "Use itertools.islice for safe slicing",
                "Check len(seq) > 0 before pop()",
            ],
        ))

    def _build_attribute_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="AttributeError",
            parent_type="Exception",
            trigger_conditions=[
                "Accessing attribute that does not exist on an object",
                "Calling method on None",
                "Accessing attribute on wrong type",
                "Module does not have requested attribute",
            ],
            prevention_guards=[
                GuardCondition("hasattr", "obj", ["attr_name"]),
                GuardCondition("not_none", "obj", []),
            ],
            attributes={"args": "Tuple[Any, ...]", "name": "Optional[str]", "obj": "Optional[Any]"},
            common_causes=[
                "None.some_method()",
                "import os; os.nonexistent",
                "'hello'.nonexistent_method()",
                "obj.misspelled_attr",
            ],
            recovery_strategies=[
                "Use hasattr(obj, 'attr') before access",
                "Use getattr(obj, 'attr', default)",
                "Check for None before attribute access",
            ],
        ))

    def _build_name_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="NameError",
            parent_type="Exception",
            trigger_conditions=[
                "Referencing a variable that has not been defined",
                "Using a name before assignment in enclosing scope",
                "Referencing a deleted variable",
            ],
            prevention_guards=[
                GuardCondition("predicate", "name", ["is_defined"]),
            ],
            attributes={"args": "Tuple[Any, ...]", "name": "Optional[str]"},
            common_causes=[
                "print(undefined_var)",
                "del x; print(x)",
            ],
            recovery_strategies=[
                "Define variables before use",
                "Use try/except NameError",
                "Check locals()/globals() for variable existence",
            ],
        ))

    def _build_unbound_local_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="UnboundLocalError",
            parent_type="NameError",
            trigger_conditions=[
                "Local variable referenced before assignment",
                "Variable assigned later in function scope shadows outer name",
            ],
            prevention_guards=[
                GuardCondition("predicate", "var", ["is_initialized"]),
            ],
            attributes={"args": "Tuple[Any, ...]", "name": "Optional[str]"},
            common_causes=[
                "x = 1\ndef f():\n    print(x)\n    x = 2",
            ],
            recovery_strategies=[
                "Initialize local variables before use",
                "Use 'global' or 'nonlocal' declarations",
            ],
        ))

    def _build_zero_division_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="ZeroDivisionError",
            parent_type="ArithmeticError",
            trigger_conditions=[
                "Division by zero (a / 0)",
                "Floor division by zero (a // 0)",
                "Modulo by zero (a % 0)",
                "divmod(a, 0)",
                "Decimal division by zero",
            ],
            prevention_guards=[
                GuardCondition("comparison", "divisor", [("!=", 0)]),
            ],
            attributes={"args": "Tuple[Any, ...]"},
            common_causes=[
                "1 / 0",
                "10 // 0",
                "10 % 0",
                "divmod(10, 0)",
            ],
            recovery_strategies=[
                "Check divisor != 0 before division",
                "Use try/except ZeroDivisionError",
                "Return a default value (e.g., float('inf')) for zero divisor",
            ],
        ))

    def _build_file_not_found_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="FileNotFoundError",
            parent_type="OSError",
            trigger_conditions=[
                "open() on a path that does not exist",
                "os.stat() on a non-existent path",
                "os.remove() on a non-existent file",
                "shutil operations on missing paths",
            ],
            prevention_guards=[
                GuardCondition("predicate", "path", ["os.path.exists"]),
                GuardCondition("predicate", "path", ["pathlib.Path.exists"]),
            ],
            attributes={"args": "Tuple[Any, ...]", "errno": "Optional[int]",
                         "strerror": "Optional[str]", "filename": "Optional[str]"},
            common_causes=[
                "open('/nonexistent/file.txt')",
                "os.remove('missing.txt')",
            ],
            recovery_strategies=[
                "Check os.path.exists() before file operations",
                "Use pathlib.Path.exists()",
                "Use try/except FileNotFoundError with fallback",
                "Create parent directories with os.makedirs(exist_ok=True)",
            ],
        ))

    def _build_permission_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="PermissionError",
            parent_type="OSError",
            trigger_conditions=[
                "open() on file without read/write permissions",
                "os.remove() on protected file",
                "os.mkdir() in protected directory",
                "Modifying read-only file system",
            ],
            prevention_guards=[
                GuardCondition("predicate", "path", ["os.access"]),
            ],
            attributes={"args": "Tuple[Any, ...]", "errno": "Optional[int]",
                         "strerror": "Optional[str]", "filename": "Optional[str]"},
            common_causes=[
                "open('/etc/shadow', 'r')",
                "os.remove('/usr/bin/python')",
            ],
            recovery_strategies=[
                "Check os.access(path, mode) before operations",
                "Run with appropriate permissions",
                "Use try/except PermissionError",
            ],
        ))

    def _build_import_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="ImportError",
            parent_type="Exception",
            trigger_conditions=[
                "Importing a module that does not exist",
                "from X import Y where Y is not defined in X",
                "Circular import at import time",
            ],
            prevention_guards=[
                GuardCondition("predicate", "module", ["importlib.util.find_spec"]),
            ],
            attributes={"args": "Tuple[Any, ...]", "name": "Optional[str]",
                         "path": "Optional[str]"},
            common_causes=[
                "import nonexistent_module",
                "from os import nonexistent_func",
            ],
            recovery_strategies=[
                "Check importlib.util.find_spec() before import",
                "Use try/except ImportError with fallback module",
                "Install missing package with pip",
            ],
        ))

    def _build_module_not_found_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="ModuleNotFoundError",
            parent_type="ImportError",
            trigger_conditions=[
                "Import of a module not installed or not on sys.path",
            ],
            prevention_guards=[
                GuardCondition("predicate", "module", ["importlib.util.find_spec"]),
            ],
            attributes={"args": "Tuple[Any, ...]", "name": "Optional[str]",
                         "path": "Optional[str]"},
            common_causes=[
                "import some_uninstalled_package",
            ],
            recovery_strategies=[
                "pip install the missing package",
                "Add the package directory to sys.path",
                "Use optional import pattern: try/except ImportError",
            ],
        ))

    def _build_stop_iteration(self) -> None:
        self._register(ExceptionModel(
            exception_type="StopIteration",
            parent_type="Exception",
            trigger_conditions=[
                "Calling next() on an exhausted iterator",
                "Generator function returns (implicitly raises StopIteration)",
                "Iterator __next__ method has no more items",
            ],
            prevention_guards=[
                GuardCondition("predicate", "iter", ["has_next"]),
            ],
            attributes={"args": "Tuple[Any, ...]", "value": "Any"},
            common_causes=[
                "it = iter([]); next(it)",
                "next(x for x in [] if True)",
            ],
            recovery_strategies=[
                "Use next(iter, default) with a default value",
                "Use for-loop instead of manual next() calls",
                "Check iterator before calling next()",
            ],
        ))

    def _build_stop_async_iteration(self) -> None:
        self._register(ExceptionModel(
            exception_type="StopAsyncIteration",
            parent_type="Exception",
            trigger_conditions=[
                "Calling __anext__() on an exhausted async iterator",
            ],
            prevention_guards=[],
            attributes={"args": "Tuple[Any, ...]", "value": "Any"},
            common_causes=[
                "async for with exhausted async generator",
            ],
            recovery_strategies=[
                "Use async for loop instead of manual __anext__",
            ],
        ))

    def _build_runtime_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="RuntimeError",
            parent_type="Exception",
            trigger_conditions=[
                "Generic runtime error not covered by more specific types",
                "Generator or coroutine used incorrectly",
                "Event loop already running (asyncio)",
                "dictionary changed size during iteration",
            ],
            prevention_guards=[],
            attributes={"args": "Tuple[Any, ...]"},
            common_causes=[
                "Modifying dict while iterating over it",
                "Sending into a just-started generator without None",
                "asyncio.run() inside running event loop",
            ],
            recovery_strategies=[
                "Copy collection before modifying during iteration",
                "Use asyncio.get_event_loop() patterns correctly",
                "Ensure generator protocol is followed",
            ],
        ))

    def _build_recursion_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="RecursionError",
            parent_type="RuntimeError",
            trigger_conditions=[
                "Maximum recursion depth exceeded",
                "Infinite mutual recursion",
            ],
            prevention_guards=[
                GuardCondition("comparison", "depth", [("<", "sys.getrecursionlimit()")]),
            ],
            attributes={"args": "Tuple[Any, ...]"},
            common_causes=[
                "def f(): f()",
                "Infinite __repr__ or __str__ recursion",
                "Deeply nested data structure traversal",
            ],
            recovery_strategies=[
                "Add base case to recursive function",
                "Use iterative approach instead of recursion",
                "Increase sys.setrecursionlimit() if needed",
                "Use @functools.lru_cache to memoize",
            ],
        ))

    def _build_memory_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="MemoryError",
            parent_type="Exception",
            trigger_conditions=[
                "Allocating too much memory",
                "Creating extremely large data structures",
            ],
            prevention_guards=[
                GuardCondition("comparison", "size", [("<", "available_memory")]),
            ],
            attributes={"args": "Tuple[Any, ...]"},
            common_causes=[
                "list(range(10**12))",
                "b'x' * (10**12)",
                "Unbounded growth of caches",
            ],
            recovery_strategies=[
                "Process data in chunks/batches",
                "Use generators instead of lists",
                "Use memory-mapped files",
                "Set resource limits with resource module",
            ],
        ))

    def _build_overflow_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="OverflowError",
            parent_type="ArithmeticError",
            trigger_conditions=[
                "Result too large to represent as float",
                "math.exp() with large argument",
                "Converting large int to float",
            ],
            prevention_guards=[
                GuardCondition("comparison", "x", [("<", "sys.float_info.max")]),
            ],
            attributes={"args": "Tuple[Any, ...]"},
            common_causes=[
                "math.exp(1000)",
                "float(10**400)",
            ],
            recovery_strategies=[
                "Use arbitrary-precision integers instead of floats",
                "Use decimal.Decimal for large numbers",
                "Check range before math operations",
            ],
        ))

    def _build_not_implemented_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="NotImplementedError",
            parent_type="RuntimeError",
            trigger_conditions=[
                "Calling an abstract method that was not overridden",
                "Method stub left unimplemented",
            ],
            prevention_guards=[],
            attributes={"args": "Tuple[Any, ...]"},
            common_causes=[
                "class Base:\n    def method(self): raise NotImplementedError",
            ],
            recovery_strategies=[
                "Implement the abstract method in the subclass",
                "Use abc.ABC and @abc.abstractmethod for proper enforcement",
            ],
        ))

    def _build_os_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="OSError",
            parent_type="Exception",
            trigger_conditions=[
                "OS-level system call failure",
                "File I/O errors",
                "Network socket errors",
                "Process-related errors",
            ],
            prevention_guards=[
                GuardCondition("predicate", "path", ["os.path.exists"]),
                GuardCondition("predicate", "path", ["os.access"]),
            ],
            attributes={"args": "Tuple[Any, ...]", "errno": "Optional[int]",
                         "strerror": "Optional[str]", "filename": "Optional[str]",
                         "filename2": "Optional[str]"},
            common_causes=[
                "File system full (ENOSPC)",
                "Too many open files (EMFILE)",
                "Disk I/O error (EIO)",
            ],
            recovery_strategies=[
                "Check preconditions (file exists, permissions, disk space)",
                "Use context managers for resource cleanup",
                "Implement retry logic for transient errors",
            ],
        ))

    def _build_unicode_decode_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="UnicodeDecodeError",
            parent_type="UnicodeError",
            trigger_conditions=[
                "Decoding bytes that are not valid in the specified encoding",
                "Reading a file with wrong encoding",
            ],
            prevention_guards=[
                GuardCondition("predicate", "data", ["is_valid_encoding"]),
            ],
            attributes={"args": "Tuple[Any, ...]", "encoding": "str",
                         "reason": "str", "object": "bytes",
                         "start": "int", "end": "int"},
            common_causes=[
                "b'\\xff\\xfe'.decode('utf-8')",
                "open('binary.dat').read()  # without 'rb'",
            ],
            recovery_strategies=[
                "Specify errors='replace' or errors='ignore'",
                "Detect encoding with chardet/charset-normalizer",
                "Open files in binary mode when content is not text",
            ],
        ))

    def _build_unicode_encode_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="UnicodeEncodeError",
            parent_type="UnicodeError",
            trigger_conditions=[
                "Encoding a string with characters not in the target encoding",
            ],
            prevention_guards=[
                GuardCondition("predicate", "text", ["can_encode"]),
            ],
            attributes={"args": "Tuple[Any, ...]", "encoding": "str",
                         "reason": "str", "object": "str",
                         "start": "int", "end": "int"},
            common_causes=[
                "'café'.encode('ascii')",
            ],
            recovery_strategies=[
                "Use UTF-8 encoding",
                "Specify errors='replace' or errors='xmlcharrefreplace'",
            ],
        ))

    def _build_unicode_translate_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="UnicodeTranslateError",
            parent_type="UnicodeError",
            trigger_conditions=[
                "Character mapping translation failure",
            ],
            prevention_guards=[],
            attributes={"args": "Tuple[Any, ...]", "encoding": "str",
                         "reason": "str", "object": "str",
                         "start": "int", "end": "int"},
            common_causes=[
                "str.translate() with incomplete translation table",
            ],
            recovery_strategies=[
                "Ensure translation table covers all characters",
                "Use errors='replace' parameter",
            ],
        ))

    def _build_connection_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="ConnectionError",
            parent_type="OSError",
            trigger_conditions=[
                "Network connection failure",
                "Connection dropped unexpectedly",
            ],
            prevention_guards=[
                GuardCondition("predicate", "host", ["is_reachable"]),
            ],
            attributes={"args": "Tuple[Any, ...]", "errno": "Optional[int]",
                         "strerror": "Optional[str]"},
            common_causes=[
                "Connecting to a down server",
                "Network cable unplugged",
            ],
            recovery_strategies=[
                "Implement retry with exponential backoff",
                "Use connection pooling",
                "Set timeouts on network operations",
            ],
        ))

    def _build_connection_refused_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="ConnectionRefusedError",
            parent_type="ConnectionError",
            trigger_conditions=[
                "Target host actively refuses connection (no service on port)",
            ],
            prevention_guards=[
                GuardCondition("predicate", "host_port", ["is_port_open"]),
            ],
            attributes={"args": "Tuple[Any, ...]", "errno": "Optional[int]",
                         "strerror": "Optional[str]"},
            common_causes=[
                "socket.connect(('localhost', 9999))  # nothing listening",
                "requests.get('http://localhost:1234')",
            ],
            recovery_strategies=[
                "Verify service is running on target host/port",
                "Retry with backoff",
            ],
        ))

    def _build_connection_reset_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="ConnectionResetError",
            parent_type="ConnectionError",
            trigger_conditions=[
                "Remote end forcibly closed the connection",
            ],
            prevention_guards=[],
            attributes={"args": "Tuple[Any, ...]", "errno": "Optional[int]",
                         "strerror": "Optional[str]"},
            common_causes=[
                "Server crashes mid-request",
                "Firewall drops connection",
            ],
            recovery_strategies=[
                "Retry the operation",
                "Implement connection health checks",
            ],
        ))

    def _build_connection_aborted_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="ConnectionAbortedError",
            parent_type="ConnectionError",
            trigger_conditions=[
                "Connection aborted by local host (e.g., timeout during handshake)",
            ],
            prevention_guards=[],
            attributes={"args": "Tuple[Any, ...]", "errno": "Optional[int]",
                         "strerror": "Optional[str]"},
            common_causes=[
                "SSL handshake failure",
                "TCP connection aborted during setup",
            ],
            recovery_strategies=[
                "Check SSL certificates",
                "Retry connection",
            ],
        ))

    def _build_broken_pipe_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="BrokenPipeError",
            parent_type="ConnectionError",
            trigger_conditions=[
                "Writing to a pipe or socket whose other end has been closed",
            ],
            prevention_guards=[],
            attributes={"args": "Tuple[Any, ...]", "errno": "Optional[int]",
                         "strerror": "Optional[str]"},
            common_causes=[
                "Writing to a subprocess whose stdin was closed",
                "Sending data after remote TCP close",
                "python script.py | head  # head closes early",
            ],
            recovery_strategies=[
                "Handle SIGPIPE signal",
                "Check connection is alive before writing",
                "Use try/except BrokenPipeError",
            ],
        ))

    def _build_timeout_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="TimeoutError",
            parent_type="OSError",
            trigger_conditions=[
                "Operation timed out at the OS level",
                "socket.timeout (alias for TimeoutError in Python 3.3+)",
            ],
            prevention_guards=[
                GuardCondition("predicate", "host", ["is_reachable"]),
            ],
            attributes={"args": "Tuple[Any, ...]", "errno": "Optional[int]",
                         "strerror": "Optional[str]"},
            common_causes=[
                "socket.settimeout(1); socket.connect(slow_host)",
                "urllib.request.urlopen(url, timeout=1)",
            ],
            recovery_strategies=[
                "Increase timeout value",
                "Implement retry with exponential backoff",
                "Use async I/O for concurrent requests",
            ],
        ))

    def _build_eof_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="EOFError",
            parent_type="Exception",
            trigger_conditions=[
                "input() hits end-of-file without reading data",
                "Reading from a closed pipe or file at EOF",
            ],
            prevention_guards=[
                GuardCondition("predicate", "stream", ["not_at_eof"]),
            ],
            attributes={"args": "Tuple[Any, ...]"},
            common_causes=[
                "input() when stdin is closed",
                "Piped input exhausted",
            ],
            recovery_strategies=[
                "Check for EOF before reading",
                "Handle EOFError with try/except",
                "Use sys.stdin.readline() and check for empty string",
            ],
        ))

    def _build_buffer_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="BufferError",
            parent_type="Exception",
            trigger_conditions=[
                "Buffer-related operation failure",
                "Resizing a buffer that has exported views",
            ],
            prevention_guards=[],
            attributes={"args": "Tuple[Any, ...]"},
            common_causes=[
                "Resizing a bytearray while a memoryview references it",
            ],
            recovery_strategies=[
                "Release all memoryview references before resizing",
                "Copy data to a new buffer instead of resizing in place",
            ],
        ))

    def _build_assertion_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="AssertionError",
            parent_type="Exception",
            trigger_conditions=[
                "assert statement with a falsy expression",
            ],
            prevention_guards=[
                GuardCondition("predicate", "condition", ["is_true"]),
            ],
            attributes={"args": "Tuple[Any, ...]"},
            common_causes=[
                "assert False",
                "assert len(lst) > 0",
            ],
            recovery_strategies=[
                "Fix the condition that caused the assertion to fail",
                "Use proper validation instead of assert in production code",
            ],
        ))

    def _build_floating_point_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="FloatingPointError",
            parent_type="ArithmeticError",
            trigger_conditions=[
                "Floating-point operation failure when fpectl is active",
            ],
            prevention_guards=[],
            attributes={"args": "Tuple[Any, ...]"},
            common_causes=[
                "Enabled via fpectl module (rarely used)",
            ],
            recovery_strategies=[
                "Use math.isfinite() to check results",
                "Use decimal.Decimal for precise arithmetic",
            ],
        ))

    def _build_reference_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="ReferenceError",
            parent_type="Exception",
            trigger_conditions=[
                "Accessing a weakref proxy after the referent is garbage-collected",
            ],
            prevention_guards=[
                GuardCondition("predicate", "ref", ["is_alive"]),
            ],
            attributes={"args": "Tuple[Any, ...]"},
            common_causes=[
                "weakref.proxy(obj)() after del obj and gc.collect()",
            ],
            recovery_strategies=[
                "Check weakref is alive before access",
                "Use weakref.ref() and check for None return",
            ],
        ))

    def _build_system_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="SystemError",
            parent_type="Exception",
            trigger_conditions=[
                "Internal CPython interpreter error",
                "C extension returns an error without setting an exception",
            ],
            prevention_guards=[],
            attributes={"args": "Tuple[Any, ...]"},
            common_causes=[
                "Bug in a C extension module",
                "Interpreter internal inconsistency",
            ],
            recovery_strategies=[
                "Report as a bug to the extension/interpreter maintainer",
            ],
        ))

    def _build_blocking_io_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="BlockingIOError",
            parent_type="OSError",
            trigger_conditions=[
                "Non-blocking I/O operation would block",
            ],
            prevention_guards=[],
            attributes={"args": "Tuple[Any, ...]", "errno": "Optional[int]",
                         "strerror": "Optional[str]", "characters_written": "int"},
            common_causes=[
                "Writing to a non-blocking socket when the send buffer is full",
            ],
            recovery_strategies=[
                "Use select/poll/epoll to wait for I/O readiness",
                "Use asyncio for non-blocking I/O",
            ],
        ))

    def _build_child_process_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="ChildProcessError",
            parent_type="OSError",
            trigger_conditions=[
                "Operation on a child process that no longer exists",
            ],
            prevention_guards=[],
            attributes={"args": "Tuple[Any, ...]", "errno": "Optional[int]",
                         "strerror": "Optional[str]"},
            common_causes=[
                "os.waitpid() on a process that was already reaped",
            ],
            recovery_strategies=[
                "Track child process lifecycle correctly",
                "Use subprocess module with proper wait() calls",
            ],
        ))

    def _build_interrupted_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="InterruptedError",
            parent_type="OSError",
            trigger_conditions=[
                "System call interrupted by a signal",
            ],
            prevention_guards=[],
            attributes={"args": "Tuple[Any, ...]", "errno": "Optional[int]",
                         "strerror": "Optional[str]"},
            common_causes=[
                "EINTR during a blocking I/O call",
            ],
            recovery_strategies=[
                "Retry the interrupted system call",
                "Python 3.5+ automatically retries on EINTR",
            ],
        ))

    def _build_is_a_directory_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="IsADirectoryError",
            parent_type="OSError",
            trigger_conditions=[
                "File operation on a path that is a directory",
            ],
            prevention_guards=[
                GuardCondition("predicate", "path", ["os.path.isfile"]),
            ],
            attributes={"args": "Tuple[Any, ...]", "errno": "Optional[int]",
                         "strerror": "Optional[str]", "filename": "Optional[str]"},
            common_causes=[
                "open('/tmp/', 'w')",
            ],
            recovery_strategies=[
                "Check os.path.isfile() before file operations",
            ],
        ))

    def _build_not_a_directory_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="NotADirectoryError",
            parent_type="OSError",
            trigger_conditions=[
                "Directory operation on a path that is not a directory",
            ],
            prevention_guards=[
                GuardCondition("predicate", "path", ["os.path.isdir"]),
            ],
            attributes={"args": "Tuple[Any, ...]", "errno": "Optional[int]",
                         "strerror": "Optional[str]", "filename": "Optional[str]"},
            common_causes=[
                "os.listdir('/etc/passwd')  # a file, not a directory",
            ],
            recovery_strategies=[
                "Check os.path.isdir() before directory operations",
            ],
        ))

    def _build_process_lookup_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="ProcessLookupError",
            parent_type="OSError",
            trigger_conditions=[
                "os.kill() on a PID that does not exist",
            ],
            prevention_guards=[],
            attributes={"args": "Tuple[Any, ...]", "errno": "Optional[int]",
                         "strerror": "Optional[str]"},
            common_causes=[
                "os.kill(99999, signal.SIGTERM)",
            ],
            recovery_strategies=[
                "Check process existence before sending signals",
                "Handle ProcessLookupError gracefully",
            ],
        ))

    def _build_file_exists_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="FileExistsError",
            parent_type="OSError",
            trigger_conditions=[
                "Creating a file or directory that already exists (without exist_ok)",
            ],
            prevention_guards=[
                GuardCondition("predicate", "path", ["not os.path.exists"]),
            ],
            attributes={"args": "Tuple[Any, ...]", "errno": "Optional[int]",
                         "strerror": "Optional[str]", "filename": "Optional[str]"},
            common_causes=[
                "os.mkdir('existing_dir')",
                "open('existing.txt', 'x')",
            ],
            recovery_strategies=[
                "Use exist_ok=True with os.makedirs()",
                "Check existence before creation",
            ],
        ))

    def _build_syntax_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="SyntaxError",
            parent_type="Exception",
            trigger_conditions=[
                "Invalid Python syntax detected during parsing",
                "compile() or eval() of malformed source",
            ],
            prevention_guards=[
                GuardCondition("predicate", "source", ["ast.parse"]),
            ],
            attributes={"args": "Tuple[Any, ...]", "filename": "Optional[str]",
                         "lineno": "Optional[int]", "offset": "Optional[int]",
                         "text": "Optional[str]", "msg": "str"},
            common_causes=[
                "eval('def f( : pass')",
                "exec('if True')",
            ],
            recovery_strategies=[
                "Fix the syntax in the source code",
                "Validate source with ast.parse() before eval/exec",
            ],
        ))

    def _build_indentation_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="IndentationError",
            parent_type="SyntaxError",
            trigger_conditions=[
                "Incorrect indentation in Python source",
            ],
            prevention_guards=[],
            attributes={"args": "Tuple[Any, ...]", "filename": "Optional[str]",
                         "lineno": "Optional[int]", "offset": "Optional[int]",
                         "text": "Optional[str]", "msg": "str"},
            common_causes=[
                "Mixing tabs and spaces (use TabError)",
                "Missing indentation after colon",
            ],
            recovery_strategies=[
                "Use consistent indentation (4 spaces recommended)",
                "Configure editor to convert tabs to spaces",
            ],
        ))

    def _build_tab_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="TabError",
            parent_type="IndentationError",
            trigger_conditions=[
                "Inconsistent use of tabs and spaces in indentation",
            ],
            prevention_guards=[],
            attributes={"args": "Tuple[Any, ...]", "filename": "Optional[str]",
                         "lineno": "Optional[int]", "offset": "Optional[int]",
                         "text": "Optional[str]", "msg": "str"},
            common_causes=[
                "Mixing tabs and spaces in the same block",
            ],
            recovery_strategies=[
                "Convert all tabs to spaces (or vice versa)",
                "Use expandtabs() or retab in editor",
            ],
        ))

    def _build_lookup_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="LookupError",
            parent_type="Exception",
            trigger_conditions=[
                "Abstract base for IndexError and KeyError",
            ],
            prevention_guards=[],
            attributes={"args": "Tuple[Any, ...]"},
            common_causes=[],
            recovery_strategies=[
                "Catch specific IndexError or KeyError instead",
            ],
        ))

    def _build_arithmetic_error(self) -> None:
        self._register(ExceptionModel(
            exception_type="ArithmeticError",
            parent_type="Exception",
            trigger_conditions=[
                "Abstract base for numeric errors",
            ],
            prevention_guards=[],
            attributes={"args": "Tuple[Any, ...]"},
            common_causes=[],
            recovery_strategies=[
                "Catch specific ZeroDivisionError, OverflowError, or FloatingPointError",
            ],
        ))


# ---------------------------------------------------------------------------
# 3. ExceptionFlowAnalysis
# ---------------------------------------------------------------------------

class ExceptionFlowAnalysis:
    """Analyses exception propagation, handler matching, try/except semantics,
    and exception chaining for a refinement-type–based abstract interpreter."""

    def __init__(self) -> None:
        self._hierarchy = ExceptionHierarchy()
        self._registry = ExceptionModelRegistry()

    def propagate_exception(
        self, exc: ExceptionInfo, call_stack: List[str]
    ) -> ExceptionInfo:
        """Annotate *exc* with traceback information from *call_stack*."""
        new_attrs = dict(exc.attributes)
        new_attrs["__traceback_frames__"] = list(call_stack)
        new_attrs["__traceback_depth__"] = len(call_stack)
        if call_stack:
            new_attrs["__traceback_top__"] = call_stack[-1]
            new_attrs["__traceback_bottom__"] = call_stack[0]
        propagated_message = exc.message
        if call_stack:
            propagated_message = (
                f"{exc.message} "
                f"(propagated through {len(call_stack)} frame(s), "
                f"origin: {call_stack[-1]})"
            )
        return ExceptionInfo(
            exception_type=exc.exception_type,
            message=propagated_message,
            attributes=new_attrs,
            cause=exc.cause,
            context=exc.context,
        )

    def match_handler(
        self, exc: ExceptionInfo, handlers: List[HandlerMatch]
    ) -> Optional[int]:
        """Return the index of the first handler that catches *exc*, or None."""
        for handler in handlers:
            if not handler.exception_types:
                # bare ``except:`` catches everything
                return handler.handler_index
            for etype in handler.exception_types:
                if self._hierarchy.is_subtype(exc.exception_type, etype):
                    return handler.handler_index
        return None

    def analyze_try_except(
        self,
        try_state: AbstractState,
        handlers: List[HandlerMatch],
        finally_body: Optional[str] = None,
    ) -> AbstractState:
        """Compute the abstract state after a try/except/finally block.

        For every possible exception in *try_state*, determine which handler
        catches it, narrow the state inside that handler, and merge all
        resulting states together.  Uncaught exceptions remain in the
        exception state.
        """
        merged = try_state.copy()
        uncaught: List[ExceptionInfo] = []
        handler_states: List[AbstractState] = []

        for exc in try_state.exception_state.possible_exceptions:
            idx = self.match_handler(exc, handlers)
            if idx is not None:
                handler = handlers[idx]
                narrowed = self.narrow_in_except(
                    try_state, exc.exception_type, handler.binds_as
                )
                handler_states.append(narrowed)
                merged.exception_state.caught_exceptions.append(exc)
            else:
                uncaught.append(exc)

        # Merge handler states into merged
        for hs in handler_states:
            for var, typ in hs.variables.items():
                if var in merged.variables:
                    existing = merged.variables[var]
                    # join: widen to the weaker of the two
                    if existing.base_type != typ.base_type:
                        merged.variables[var] = RefinementType(base_type="Any")
                    else:
                        merged.variables[var] = RefinementType(
                            base_type=existing.base_type,
                            nullity=NullityTag.MAYBE_NULL
                            if existing.nullity != typ.nullity
                            else existing.nullity,
                        )
                else:
                    merged.variables[var] = typ

        merged.exception_state.possible_exceptions = uncaught
        merged.exception_state.definitely_raises = (
            len(uncaught) > 0 and try_state.exception_state.definitely_raises
        )

        if finally_body is not None:
            merged.exception_state.possible_exceptions = list(uncaught)

        return merged

    def narrow_in_except(
        self,
        state: AbstractState,
        caught_type: str,
        bind_var: Optional[str],
    ) -> AbstractState:
        """Narrow the abstract state to reflect that we are inside an except
        handler for *caught_type*."""
        narrowed = state.copy()
        narrowed.exception_state.possible_exceptions = [
            e for e in narrowed.exception_state.possible_exceptions
            if not self._hierarchy.is_subtype(e.exception_type, caught_type)
        ]
        if bind_var is not None:
            model = self._registry.get_or_default(caught_type)
            narrowed = model.get_refined_state_after_catch(narrowed, bind_var)
        return narrowed

    def merge_exception_states(
        self, states: List[ExceptionState]
    ) -> ExceptionState:
        """Merge multiple exception states into one (union of possibilities)."""
        merged_possible: List[ExceptionInfo] = []
        merged_caught: List[ExceptionInfo] = []
        seen_types: Set[str] = set()
        definitely = False

        for s in states:
            for exc in s.possible_exceptions:
                if exc.exception_type not in seen_types:
                    merged_possible.append(exc)
                    seen_types.add(exc.exception_type)
            merged_caught.extend(s.caught_exceptions)
            if s.definitely_raises:
                definitely = True

        # definitely_raises only if ALL branches raise
        all_raise = all(s.definitely_raises for s in states) if states else False
        return ExceptionState(
            possible_exceptions=merged_possible,
            definitely_raises=all_raise,
            caught_exceptions=merged_caught,
        )

    def analyze_raise_from(
        self, exc: ExceptionInfo, cause: ExceptionInfo
    ) -> ExceptionInfo:
        """Model ``raise exc from cause`` — explicit exception chaining."""
        return ExceptionInfo(
            exception_type=exc.exception_type,
            message=exc.message,
            attributes={**exc.attributes, "__suppress_context__": True},
            cause=cause,
            context=exc.context,
        )

    def analyze_bare_raise(self, state: AbstractState) -> AbstractState:
        """Model a bare ``raise`` statement that re-raises the current exception."""
        new_state = state.copy()
        if state.exception_state.caught_exceptions:
            last = state.exception_state.caught_exceptions[-1]
            new_state.exception_state.possible_exceptions.append(last)
            new_state.exception_state.definitely_raises = True
        else:
            new_state.exception_state.definitely_raises = True
            new_state.exception_state.possible_exceptions.append(
                ExceptionInfo(exception_type="RuntimeError",
                              message="No active exception to re-raise")
            )
        return new_state

    def analyze_exception_group(
        self, exceptions: List[ExceptionInfo]
    ) -> ExceptionInfo:
        """Model Python 3.11+ ``ExceptionGroup``."""
        types = [e.exception_type for e in exceptions]
        messages = [e.message for e in exceptions]
        return ExceptionInfo(
            exception_type="ExceptionGroup",
            message=f"ExceptionGroup({len(exceptions)} sub-exceptions)",
            attributes={
                "exceptions": exceptions,
                "exception_types": types,
                "messages": messages,
                "count": len(exceptions),
            },
        )

    def get_uncaught_exceptions(
        self, try_state: AbstractState, handlers: List[HandlerMatch]
    ) -> List[ExceptionInfo]:
        """Return all exceptions from *try_state* that no handler catches."""
        uncaught: List[ExceptionInfo] = []
        for exc in try_state.exception_state.possible_exceptions:
            if self.match_handler(exc, handlers) is None:
                uncaught.append(exc)
        return uncaught

    def compute_exception_summary(
        self, func_body_states: List[AbstractState]
    ) -> ExceptionState:
        """Compute the overall exception state for a function by merging
        the states of all basic blocks / exit points."""
        all_exc_states = [s.exception_state for s in func_body_states if s.reachable]
        if not all_exc_states:
            return ExceptionState()
        return self.merge_exception_states(all_exc_states)


# ---------------------------------------------------------------------------
# 4. ExceptionPreconditionInference
# ---------------------------------------------------------------------------

class ExceptionPreconditionInference:
    """Infer guard conditions (preconditions) that prevent specific exceptions
    from being raised during a given operation."""

    def __init__(self) -> None:
        self._hierarchy = ExceptionHierarchy()
        self._registry = ExceptionModelRegistry()
        self._operation_guards = self._build_operation_guards()

    def _build_operation_guards(self) -> Dict[str, List[Tuple[str, GuardCondition]]]:
        """Map (operation) → list of (exception_type, guard)."""
        guards: Dict[str, List[Tuple[str, GuardCondition]]] = {}

        def _add(op: str, exc: str, g: GuardCondition) -> None:
            guards.setdefault(op, []).append((exc, g))

        # subscript / getitem
        _add("getitem", "IndexError", GuardCondition("comparison", "i", [(">=", 0), ("<", "len(seq)")]))
        _add("getitem", "KeyError", GuardCondition("contains", "key", ["dict"]))
        _add("getitem", "TypeError", GuardCondition("isinstance", "obj", ["Mapping", "Sequence"]))
        _add("subscript", "IndexError", GuardCondition("comparison", "i", [(">=", 0), ("<", "len(seq)")]))
        _add("subscript", "KeyError", GuardCondition("contains", "key", ["dict"]))

        # division
        for op in ("div", "truediv", "floordiv", "mod", "/", "//", "%"):
            _add(op, "ZeroDivisionError", GuardCondition("comparison", "divisor", [("!=", 0)]))

        # attribute access
        for op in ("getattr", ".", "attribute_access"):
            _add(op, "AttributeError", GuardCondition("hasattr", "obj", ["attr_name"]))
            _add(op, "AttributeError", GuardCondition("not_none", "obj", []))

        # call
        for op in ("call", "__call__", "invoke"):
            _add(op, "TypeError", GuardCondition("predicate", "obj", ["callable"]))
            _add(op, "RecursionError", GuardCondition("comparison", "depth", [("<", "sys.getrecursionlimit()")]))

        # name lookup
        _add("name_lookup", "NameError", GuardCondition("predicate", "name", ["is_defined"]))
        _add("load_name", "NameError", GuardCondition("predicate", "name", ["is_defined"]))

        # conversion
        for op in ("int", "float"):
            _add(op, "ValueError", GuardCondition("predicate", "x", ["str.isnumeric"]))
            _add(op, "TypeError", GuardCondition("isinstance", "x", ["str", "int", "float"]))

        # iteration
        for op in ("iter", "__iter__", "for"):
            _add(op, "TypeError", GuardCondition("predicate", "obj", ["is_iterable"]))

        # next
        _add("next", "StopIteration", GuardCondition("predicate", "iter", ["has_next"]))
        _add("__next__", "StopIteration", GuardCondition("predicate", "iter", ["has_next"]))

        # file operations
        _add("open", "FileNotFoundError", GuardCondition("predicate", "path", ["os.path.exists"]))
        _add("open", "PermissionError", GuardCondition("predicate", "path", ["os.access"]))
        _add("open", "IsADirectoryError", GuardCondition("predicate", "path", ["os.path.isfile"]))

        # import
        _add("import", "ImportError", GuardCondition("predicate", "module", ["importlib.util.find_spec"]))
        _add("import", "ModuleNotFoundError", GuardCondition("predicate", "module", ["importlib.util.find_spec"]))

        # encoding
        _add("decode", "UnicodeDecodeError", GuardCondition("predicate", "data", ["is_valid_encoding"]))
        _add("encode", "UnicodeEncodeError", GuardCondition("predicate", "text", ["can_encode"]))

        # network
        _add("connect", "ConnectionRefusedError", GuardCondition("predicate", "host_port", ["is_port_open"]))
        _add("connect", "TimeoutError", GuardCondition("predicate", "host", ["is_reachable"]))

        # math
        _add("exp", "OverflowError", GuardCondition("comparison", "x", [("<", 709)]))
        _add("pow", "OverflowError", GuardCondition("comparison", "result", [("<", "sys.float_info.max")]))

        return guards

    def infer_precondition(
        self, operation: str, operand_types: List[RefinementType]
    ) -> List[GuardCondition]:
        """Return all guard conditions needed to make *operation* safe."""
        guards: List[GuardCondition] = []
        op_lower = operation.lower()

        # check the pre-built guard table
        if op_lower in self._operation_guards:
            for exc_type, guard in self._operation_guards[op_lower]:
                type_strs = [t.base_type for t in operand_types]
                model = self._registry.get(exc_type)
                if model is not None and model.can_be_raised_by(operation, type_strs):
                    guards.append(guard)

        # add nullity guards for all nullable operands
        for i, ot in enumerate(operand_types):
            if ot.nullity in (NullityTag.MAYBE_NULL, NullityTag.DEFINITELY_NULL):
                guards.append(GuardCondition("not_none", f"operand_{i}", []))

        return self.combine_preconditions(guards)

    def combine_preconditions(
        self, preconds: List[GuardCondition]
    ) -> List[GuardCondition]:
        """Simplify a list of preconditions by merging compatible guards."""
        if not preconds:
            return []

        # Group by (kind, variable)
        groups: Dict[Tuple[str, str], List[GuardCondition]] = {}
        for g in preconds:
            key = (g.kind, g.variable)
            groups.setdefault(key, []).append(g)

        merged: List[GuardCondition] = []
        for (kind, var), group in groups.items():
            if len(group) == 1:
                merged.append(group[0])
                continue

            if kind == "comparison":
                # merge all comparison args into one guard
                all_args: List[Any] = []
                seen: Set[str] = set()
                for g in group:
                    for arg in g.args:
                        key_str = str(arg)
                        if key_str not in seen:
                            all_args.append(arg)
                            seen.add(key_str)
                merged.append(GuardCondition(kind, var, all_args))
            elif kind == "isinstance":
                # union the type sets
                all_types: Set[str] = set()
                for g in group:
                    all_types.update(str(a) for a in g.args)
                merged.append(GuardCondition(kind, var, sorted(all_types)))
            else:
                # deduplicate
                seen_args: Set[str] = set()
                for g in group:
                    arg_key = str(g.args)
                    if arg_key not in seen_args:
                        merged.append(g)
                        seen_args.add(arg_key)
        return merged

    def negate_precondition(self, guard: GuardCondition) -> GuardCondition:
        """Return the logical negation of *guard*."""
        negation_map = {
            "==": "!=", "!=": "==",
            "<": ">=", ">=": "<",
            ">": "<=", "<=": ">",
        }
        if guard.kind == "comparison":
            negated_args = []
            for arg in guard.args:
                if isinstance(arg, (list, tuple)) and len(arg) == 2:
                    op, val = arg
                    neg_op = negation_map.get(op, f"not({op})")
                    negated_args.append((neg_op, val))
                else:
                    negated_args.append(arg)
            return GuardCondition("comparison", guard.variable, negated_args)
        elif guard.kind == "isinstance":
            return GuardCondition("not_isinstance", guard.variable, list(guard.args))
        elif guard.kind == "contains":
            return GuardCondition("not_contains", guard.variable, list(guard.args))
        elif guard.kind == "hasattr":
            return GuardCondition("not_hasattr", guard.variable, list(guard.args))
        elif guard.kind == "not_none":
            return GuardCondition("is_none", guard.variable, [])
        elif guard.kind == "is_none":
            return GuardCondition("not_none", guard.variable, [])
        elif guard.kind == "predicate":
            func = guard.args[0] if guard.args else "check"
            return GuardCondition("predicate", guard.variable, [f"not {func}"])
        elif guard.kind == "truthy":
            return GuardCondition("falsy", guard.variable, [])
        else:
            return GuardCondition(f"not_{guard.kind}", guard.variable, list(guard.args))

    def strengthen_precondition(
        self, existing: GuardCondition, new: GuardCondition
    ) -> GuardCondition:
        """Strengthen *existing* by conjoining it with *new* (intersection)."""
        if existing.kind == new.kind and existing.variable == new.variable:
            if existing.kind == "comparison":
                merged_args = list(existing.args)
                existing_strs = {str(a) for a in existing.args}
                for arg in new.args:
                    if str(arg) not in existing_strs:
                        merged_args.append(arg)
                return GuardCondition("comparison", existing.variable, merged_args)
            elif existing.kind == "isinstance":
                # intersection of allowed types
                existing_set = set(str(a) for a in existing.args)
                new_set = set(str(a) for a in new.args)
                common = existing_set & new_set
                if common:
                    return GuardCondition("isinstance", existing.variable, sorted(common))
                return GuardCondition("isinstance", existing.variable, sorted(new_set))
        # fallback: conjunction expressed as a new compound guard
        combined_source = f"({existing.to_source()}) and ({new.to_source()})"
        return GuardCondition("predicate", existing.variable, [combined_source])

    def weaken_precondition(self, guard: GuardCondition) -> GuardCondition:
        """Weaken *guard* to a less restrictive condition (over-approximation)."""
        if guard.kind == "comparison":
            # drop the most restrictive bound
            if len(guard.args) > 1:
                return GuardCondition("comparison", guard.variable, guard.args[:1])
            return GuardCondition("truthy", guard.variable, [])
        elif guard.kind == "isinstance":
            # widen type set: walk up the hierarchy for each type
            widened: Set[str] = set()
            for t in guard.args:
                parent = self._hierarchy.get_parent(str(t))
                widened.add(parent if parent else str(t))
            return GuardCondition("isinstance", guard.variable, sorted(widened))
        elif guard.kind == "contains":
            return GuardCondition("truthy", guard.variable, [])
        elif guard.kind == "not_none":
            return GuardCondition("truthy", guard.variable, [])
        else:
            return GuardCondition("truthy", guard.variable, [])


# ---------------------------------------------------------------------------
# 5. ExceptionSafetyChecker
# ---------------------------------------------------------------------------

class ExceptionSafetyChecker:
    """Check exception safety properties: resource leaks, transaction safety,
    invariant preservation, and cleanup completeness."""

    def __init__(self) -> None:
        self._hierarchy = ExceptionHierarchy()

    def check_resource_safety(
        self,
        resources: List[ResourceInfo],
        exception_points: List[str],
    ) -> List[SafetyViolation]:
        """Check whether resources acquired before exception points are
        properly released (even in the face of exceptions)."""
        violations: List[SafetyViolation] = []
        for res in resources:
            if res.released:
                continue
            # check if any exception point occurs after resource acquisition
            for ep in exception_points:
                if self._location_after(ep, res.acquired_at):
                    violations.append(SafetyViolation(
                        kind="resource_leak",
                        description=(
                            f"Resource '{res.variable}' (type: {res.resource_type}) "
                            f"acquired at {res.acquired_at} may leak if exception "
                            f"is raised at {ep}"
                        ),
                        location=ep,
                        severity="error",
                    ))
                    break  # one violation per resource is enough

        # special patterns
        for res in resources:
            if not res.released:
                if res.resource_type == "file":
                    violations.append(SafetyViolation(
                        kind="file_not_closed",
                        description=(
                            f"File '{res.variable}' opened at {res.acquired_at} "
                            f"is never closed; use 'with' statement"
                        ),
                        location=res.acquired_at,
                        severity="warning",
                    ))
                elif res.resource_type == "lock":
                    violations.append(SafetyViolation(
                        kind="lock_not_released",
                        description=(
                            f"Lock '{res.variable}' acquired at {res.acquired_at} "
                            f"is never released; use 'with' statement or finally block"
                        ),
                        location=res.acquired_at,
                        severity="error",
                    ))
                elif res.resource_type == "connection":
                    violations.append(SafetyViolation(
                        kind="connection_not_closed",
                        description=(
                            f"Connection '{res.variable}' opened at {res.acquired_at} "
                            f"is never closed"
                        ),
                        location=res.acquired_at,
                        severity="error",
                    ))
                elif res.resource_type == "temp_file":
                    violations.append(SafetyViolation(
                        kind="temp_file_not_deleted",
                        description=(
                            f"Temporary file '{res.variable}' created at "
                            f"{res.acquired_at} is never deleted"
                        ),
                        location=res.acquired_at,
                        severity="warning",
                    ))
        return violations

    def check_transaction_safety(
        self,
        state_modifications: List[str],
        exception_points: List[str],
    ) -> List[SafetyViolation]:
        """Check that state modifications are atomic with respect to
        exceptions (i.e., no partial updates visible)."""
        violations: List[SafetyViolation] = []
        if len(state_modifications) < 2:
            return violations

        for i, mod in enumerate(state_modifications[:-1]):
            for ep in exception_points:
                if self._location_between(mod, ep, state_modifications[i + 1]):
                    violations.append(SafetyViolation(
                        kind="partial_update",
                        description=(
                            f"State modification '{mod}' may be visible without "
                            f"subsequent modification '{state_modifications[i + 1]}' "
                            f"if exception at {ep}"
                        ),
                        location=ep,
                        severity="error",
                    ))
        return violations

    def check_invariant_preservation(
        self,
        invariants: List[str],
        exception_handlers: List[HandlerMatch],
    ) -> List[SafetyViolation]:
        """Check that invariants are restored by exception handlers."""
        violations: List[SafetyViolation] = []
        for inv in invariants:
            restored = False
            for handler in exception_handlers:
                if inv.lower() in handler.body_label.lower():
                    restored = True
                    break
            if not restored:
                violations.append(SafetyViolation(
                    kind="invariant_broken",
                    description=(
                        f"Invariant '{inv}' may not be restored by any "
                        f"exception handler"
                    ),
                    location="exception_handlers",
                    severity="warning",
                ))
        return violations

    def check_cleanup_completeness(
        self,
        acquired: List[ResourceInfo],
        cleanup_handlers: List[str],
    ) -> List[SafetyViolation]:
        """Check that every acquired resource has a matching cleanup handler."""
        violations: List[SafetyViolation] = []
        cleanup_text = " ".join(cleanup_handlers).lower()
        for res in acquired:
            var_lower = res.variable.lower()
            type_lower = res.resource_type.lower()
            has_cleanup = (
                var_lower in cleanup_text
                or f"close({var_lower})" in cleanup_text
                or f"{var_lower}.close()" in cleanup_text
                or f"{var_lower}.release()" in cleanup_text
                or f"del {var_lower}" in cleanup_text
                or f"os.remove({var_lower})" in cleanup_text
            )
            if not has_cleanup and not res.released:
                violations.append(SafetyViolation(
                    kind="missing_cleanup",
                    description=(
                        f"Resource '{res.variable}' ({res.resource_type}) has no "
                        f"cleanup handler"
                    ),
                    location=res.acquired_at,
                    severity="warning",
                ))
        return violations

    def analyze_context_manager(
        self,
        enter_state: AbstractState,
        exit_handler: str,
    ) -> List[SafetyViolation]:
        """Check that a context manager's __exit__ properly handles exceptions."""
        violations: List[SafetyViolation] = []
        exit_lower = exit_handler.lower()

        # check if __exit__ swallows exceptions silently
        if "return true" in exit_lower and "log" not in exit_lower:
            violations.append(SafetyViolation(
                kind="swallowed_exception",
                description=(
                    "Context manager __exit__ returns True (suppresses exception) "
                    "without logging"
                ),
                location="__exit__",
                severity="warning",
            ))

        # check if cleanup is unconditional
        if "if exc_type" in exit_lower:
            violations.append(SafetyViolation(
                kind="conditional_cleanup",
                description=(
                    "Context manager __exit__ performs cleanup conditionally; "
                    "cleanup should be unconditional (use finally pattern)"
                ),
                location="__exit__",
                severity="info",
            ))

        # check for uncaught exceptions in the exit handler itself
        risky_ops = ["open(", "connect(", "send(", "write("]
        for op in risky_ops:
            if op in exit_lower:
                violations.append(SafetyViolation(
                    kind="exit_may_raise",
                    description=(
                        f"Context manager __exit__ performs risky operation "
                        f"'{op.rstrip('(')}' that may itself raise"
                    ),
                    location="__exit__",
                    severity="warning",
                ))

        return violations

    # ---- helpers ------------------------------------------------------------

    @staticmethod
    def _location_after(loc: str, reference: str) -> bool:
        """Simple heuristic: a location is 'after' if its line number is higher."""
        try:
            loc_line = int(loc.split(":")[-1]) if ":" in loc else hash(loc)
            ref_line = int(reference.split(":")[-1]) if ":" in reference else hash(reference)
            return loc_line >= ref_line
        except (ValueError, IndexError):
            return True  # conservative assumption

    @staticmethod
    def _location_between(before: str, mid: str, after: str) -> bool:
        try:
            b = int(before.split(":")[-1]) if ":" in before else hash(before)
            m = int(mid.split(":")[-1]) if ":" in mid else hash(mid)
            a = int(after.split(":")[-1]) if ":" in after else hash(after)
            return b <= m <= a
        except (ValueError, IndexError):
            return True


# ---------------------------------------------------------------------------
# 6. ExceptionHandlerAnalysis
# ---------------------------------------------------------------------------

class ExceptionHandlerAnalysis:
    """Analyse exception handler quality: over-broad catches, swallowed
    exceptions, handler coverage, retry patterns, and masking."""

    def __init__(self) -> None:
        self._hierarchy = ExceptionHierarchy()

    def detect_catch_all(
        self, handlers: List[HandlerMatch]
    ) -> List[SafetyViolation]:
        """Detect bare ``except:`` or ``except Exception:`` handlers."""
        violations: List[SafetyViolation] = []
        for h in handlers:
            if not h.exception_types:
                violations.append(SafetyViolation(
                    kind="bare_except",
                    description=(
                        f"Handler {h.handler_index} uses bare 'except:' which "
                        f"catches all exceptions including SystemExit and "
                        f"KeyboardInterrupt"
                    ),
                    location=h.body_label,
                    severity="error",
                ))
            elif h.exception_types == ["Exception"]:
                violations.append(SafetyViolation(
                    kind="broad_except",
                    description=(
                        f"Handler {h.handler_index} catches 'Exception' which "
                        f"is very broad; consider catching specific exception types"
                    ),
                    location=h.body_label,
                    severity="warning",
                ))
            elif "BaseException" in h.exception_types:
                violations.append(SafetyViolation(
                    kind="base_exception_catch",
                    description=(
                        f"Handler {h.handler_index} catches 'BaseException' which "
                        f"includes SystemExit and KeyboardInterrupt"
                    ),
                    location=h.body_label,
                    severity="error",
                ))
        return violations

    def detect_too_broad(
        self,
        handlers: List[HandlerMatch],
        raised_types: List[str],
    ) -> List[SafetyViolation]:
        """Detect handlers that are broader than necessary for the exceptions
        actually raised."""
        violations: List[SafetyViolation] = []
        for h in handlers:
            for etype in h.exception_types:
                subtypes = self._hierarchy.get_all_subtypes(etype)
                subtypes.add(etype)
                actually_caught = [r for r in raised_types if r in subtypes]
                unnecessarily_caught = subtypes - set(raised_types) - {etype}

                if actually_caught and unnecessarily_caught and len(unnecessarily_caught) > 3:
                    violations.append(SafetyViolation(
                        kind="too_broad_handler",
                        description=(
                            f"Handler {h.handler_index} catches '{etype}' but only "
                            f"{actually_caught} are raised; also catches "
                            f"{len(unnecessarily_caught)} unrelated subtypes. "
                            f"Consider catching {', '.join(actually_caught)} directly."
                        ),
                        location=h.body_label,
                        severity="warning",
                    ))
        return violations

    def detect_swallowed(
        self,
        handlers: List[HandlerMatch],
        handler_bodies: List[str],
    ) -> List[SafetyViolation]:
        """Detect handlers that catch an exception and do nothing (swallow it)."""
        violations: List[SafetyViolation] = []
        for i, h in enumerate(handlers):
            if i >= len(handler_bodies):
                continue
            body = handler_bodies[i].strip().lower()

            is_swallowed = (
                body == "pass"
                or body == ""
                or body == "..."
                or (body.startswith("pass") and len(body.splitlines()) == 1)
            )
            if is_swallowed:
                violations.append(SafetyViolation(
                    kind="swallowed_exception",
                    description=(
                        f"Handler {h.handler_index} catches "
                        f"{h.exception_types or ['all']} but does nothing "
                        f"(exception is silently swallowed)"
                    ),
                    location=h.body_label,
                    severity="error",
                ))
            elif body == "continue" or body == "break":
                violations.append(SafetyViolation(
                    kind="swallowed_exception_in_loop",
                    description=(
                        f"Handler {h.handler_index} catches "
                        f"{h.exception_types or ['all']} and only has '{body}'; "
                        f"consider logging the exception"
                    ),
                    location=h.body_label,
                    severity="warning",
                ))
        return violations

    def detect_exception_wrapping(
        self, handlers: List[HandlerMatch]
    ) -> List[Tuple[str, str]]:
        """Detect patterns like ``except X: raise Y from e`` (wrapping)."""
        wrappings: List[Tuple[str, str]] = []
        for h in handlers:
            body_lower = h.body_label.lower()
            if "raise" in body_lower and "from" in body_lower:
                # parse: the caught type wraps into the raised type
                for caught in h.exception_types:
                    # heuristic extraction from body_label
                    parts = body_lower.split("raise")
                    if len(parts) > 1:
                        raised_part = parts[1].strip().split("(")[0].strip()
                        if raised_part:
                            wrappings.append((caught, raised_part))
                        else:
                            wrappings.append((caught, "Unknown"))
        return wrappings

    def detect_retry_pattern(
        self,
        handlers: List[HandlerMatch],
        handler_bodies: List[str],
    ) -> bool:
        """Detect if any handler implements a retry pattern."""
        retry_keywords = [
            "retry", "again", "attempt", "backoff", "sleep",
            "while", "for _ in range", "max_retries", "num_retries",
        ]
        for i, h in enumerate(handlers):
            if i >= len(handler_bodies):
                continue
            body_lower = handler_bodies[i].lower()
            if any(kw in body_lower for kw in retry_keywords):
                return True
        return False

    def suggest_handler_improvements(
        self,
        handlers: List[HandlerMatch],
        raised_types: List[str],
    ) -> List[str]:
        """Return a list of improvement suggestions for the handlers."""
        suggestions: List[str] = []

        # check for catch-all
        catch_all_violations = self.detect_catch_all(handlers)
        for v in catch_all_violations:
            if v.kind == "bare_except":
                suggestions.append(
                    "Replace bare 'except:' with 'except Exception:' at minimum, "
                    "or better yet catch specific exception types."
                )
            elif v.kind == "broad_except":
                if raised_types:
                    suggestions.append(
                        f"Replace 'except Exception' with specific types: "
                        f"except ({', '.join(raised_types)})"
                    )

        # check coverage
        coverage = self.analyze_handler_coverage(handlers, raised_types)
        uncovered = [exc for exc, covered in coverage.items() if not covered]
        if uncovered:
            suggestions.append(
                f"Add handlers for uncaught exceptions: {', '.join(uncovered)}"
            )

        # check for multiple handlers catching the same type
        caught_by: Dict[str, List[int]] = {}
        for h in handlers:
            for etype in h.exception_types:
                caught_by.setdefault(etype, []).append(h.handler_index)
        for etype, indices in caught_by.items():
            if len(indices) > 1:
                suggestions.append(
                    f"Exception '{etype}' is caught by multiple handlers "
                    f"({indices}); only the first one will execute."
                )

        # suggest logging if no handler mentions logging
        has_logging = any(
            "log" in h.body_label.lower() or "print" in h.body_label.lower()
            for h in handlers
        )
        if not has_logging and handlers:
            suggestions.append(
                "Consider adding logging in exception handlers to aid debugging."
            )

        return suggestions

    def analyze_handler_coverage(
        self,
        handlers: List[HandlerMatch],
        possible_exceptions: List[str],
    ) -> Dict[str, bool]:
        """For each possible exception type, return whether it is caught."""
        coverage: Dict[str, bool] = {}
        for exc in possible_exceptions:
            caught = False
            for h in handlers:
                if not h.exception_types:
                    caught = True
                    break
                for etype in h.exception_types:
                    if self._hierarchy.is_subtype(exc, etype):
                        caught = True
                        break
                if caught:
                    break
            coverage[exc] = caught
        return coverage

    def detect_exception_masking(
        self, handlers: List[HandlerMatch]
    ) -> List[SafetyViolation]:
        """Detect when a handler at index i catches everything that a later
        handler at index j would catch, making j unreachable."""
        violations: List[SafetyViolation] = []
        for i, h_i in enumerate(handlers):
            for j, h_j in enumerate(handlers):
                if j <= i:
                    continue
                # check if h_i masks h_j
                if not h_i.exception_types:
                    # bare except masks everything after it
                    violations.append(SafetyViolation(
                        kind="masked_handler",
                        description=(
                            f"Handler {h_j.handler_index} catching "
                            f"{h_j.exception_types} is unreachable because "
                            f"handler {h_i.handler_index} is a bare 'except:' "
                            f"that catches everything"
                        ),
                        location=h_j.body_label,
                        severity="warning",
                    ))
                    continue
                for etype_j in h_j.exception_types:
                    for etype_i in h_i.exception_types:
                        if self._hierarchy.is_subtype(etype_j, etype_i):
                            violations.append(SafetyViolation(
                                kind="masked_handler",
                                description=(
                                    f"Handler {h_j.handler_index} catching "
                                    f"'{etype_j}' is unreachable because "
                                    f"handler {h_i.handler_index} already "
                                    f"catches '{etype_i}' which is a supertype"
                                ),
                                location=h_j.body_label,
                                severity="warning",
                            ))
        return violations


# ---------------------------------------------------------------------------
# 7. TypeScriptErrorModels
# ---------------------------------------------------------------------------

@dataclass
class TSErrorModel:
    """Model of a single TypeScript/JavaScript error type."""
    error_type: str
    parent_type: Optional[str]
    attributes: Dict[str, str]
    common_causes: List[str]


class TypeScriptErrorModels:
    """Models the TypeScript / JavaScript error hierarchy, custom error classes,
    Promise rejection semantics, and async error handling."""

    def __init__(self) -> None:
        self._parent: Dict[str, Optional[str]] = {}
        self._children: Dict[str, List[str]] = {}
        self._attributes: Dict[str, Dict[str, str]] = {}
        self._models: Dict[str, TSErrorModel] = {}
        self._build_hierarchy()
        self._build_models()

    def _add(self, child: str, parent: Optional[str]) -> None:
        self._parent[child] = parent
        self._children.setdefault(child, [])
        if parent is not None:
            self._children.setdefault(parent, [])
            if child not in self._children[parent]:
                self._children[parent].append(child)

    def _build_hierarchy(self) -> None:
        self._add("Error", None)
        for c in ("TypeError", "RangeError", "ReferenceError",
                   "SyntaxError", "URIError", "EvalError",
                   "AggregateError", "InternalError"):
            self._add(c, "Error")

        # common custom errors
        for c in ("HttpError", "ValidationError", "AuthenticationError",
                   "AuthorizationError", "NotFoundError", "TimeoutError",
                   "NetworkError", "DatabaseError", "SerializationError"):
            self._add(c, "Error")

    def _build_models(self) -> None:
        base_attrs = {"message": "string", "name": "string", "stack": "string | undefined"}

        self._models["Error"] = TSErrorModel(
            error_type="Error",
            parent_type=None,
            attributes=dict(base_attrs),
            common_causes=["throw new Error('message')"],
        )
        self._models["TypeError"] = TSErrorModel(
            error_type="TypeError",
            parent_type="Error",
            attributes={**base_attrs},
            common_causes=[
                "Calling undefined as a function",
                "Accessing property on null/undefined",
                "Cannot read properties of undefined",
                "x is not a function",
                "Cannot set properties of null",
            ],
        )
        self._models["RangeError"] = TSErrorModel(
            error_type="RangeError",
            parent_type="Error",
            attributes={**base_attrs},
            common_causes=[
                "Maximum call stack size exceeded",
                "Invalid array length",
                "toFixed() digits argument must be between 0 and 100",
                "Invalid string length",
            ],
        )
        self._models["ReferenceError"] = TSErrorModel(
            error_type="ReferenceError",
            parent_type="Error",
            attributes={**base_attrs},
            common_causes=[
                "x is not defined",
                "Cannot access before initialization (let/const TDZ)",
                "assignment to undeclared variable in strict mode",
            ],
        )
        self._models["SyntaxError"] = TSErrorModel(
            error_type="SyntaxError",
            parent_type="Error",
            attributes={**base_attrs},
            common_causes=[
                "Unexpected token",
                "Unexpected end of JSON input",
                "Invalid regular expression",
                "Missing initializer in const declaration",
            ],
        )
        self._models["URIError"] = TSErrorModel(
            error_type="URIError",
            parent_type="Error",
            attributes={**base_attrs},
            common_causes=[
                "decodeURIComponent('%') – malformed URI sequence",
            ],
        )
        self._models["EvalError"] = TSErrorModel(
            error_type="EvalError",
            parent_type="Error",
            attributes={**base_attrs},
            common_causes=[
                "Rarely thrown in modern JS; legacy eval errors",
            ],
        )
        self._models["AggregateError"] = TSErrorModel(
            error_type="AggregateError",
            parent_type="Error",
            attributes={**base_attrs, "errors": "Error[]"},
            common_causes=[
                "Promise.any() when all promises reject",
            ],
        )
        self._models["InternalError"] = TSErrorModel(
            error_type="InternalError",
            parent_type="Error",
            attributes={**base_attrs},
            common_causes=[
                "Too much recursion (Firefox-specific)",
            ],
        )

        # custom application errors
        self._models["HttpError"] = TSErrorModel(
            error_type="HttpError",
            parent_type="Error",
            attributes={**base_attrs, "statusCode": "number",
                        "response": "Response | undefined"},
            common_causes=["HTTP request returns non-2xx status"],
        )
        self._models["ValidationError"] = TSErrorModel(
            error_type="ValidationError",
            parent_type="Error",
            attributes={**base_attrs, "field": "string",
                        "constraint": "string"},
            common_causes=["Schema validation failure", "Form input validation"],
        )
        self._models["AuthenticationError"] = TSErrorModel(
            error_type="AuthenticationError",
            parent_type="Error",
            attributes={**base_attrs},
            common_causes=["Invalid credentials", "Expired token"],
        )
        self._models["AuthorizationError"] = TSErrorModel(
            error_type="AuthorizationError",
            parent_type="Error",
            attributes={**base_attrs, "requiredRole": "string"},
            common_causes=["Insufficient permissions"],
        )
        self._models["NotFoundError"] = TSErrorModel(
            error_type="NotFoundError",
            parent_type="Error",
            attributes={**base_attrs, "resource": "string"},
            common_causes=["Database record not found", "API endpoint not found"],
        )
        self._models["TimeoutError"] = TSErrorModel(
            error_type="TimeoutError",
            parent_type="Error",
            attributes={**base_attrs, "timeout": "number"},
            common_causes=["Request exceeded timeout", "Operation timed out"],
        )
        self._models["NetworkError"] = TSErrorModel(
            error_type="NetworkError",
            parent_type="Error",
            attributes={**base_attrs},
            common_causes=["fetch() failed (CORS, network down, DNS failure)"],
        )
        self._models["DatabaseError"] = TSErrorModel(
            error_type="DatabaseError",
            parent_type="Error",
            attributes={**base_attrs, "code": "string",
                        "query": "string | undefined"},
            common_causes=["SQL syntax error", "Constraint violation", "Connection lost"],
        )
        self._models["SerializationError"] = TSErrorModel(
            error_type="SerializationError",
            parent_type="Error",
            attributes={**base_attrs},
            common_causes=["JSON.parse on invalid JSON", "Circular reference in JSON.stringify"],
        )

        for k, model in self._models.items():
            self._attributes[k] = model.attributes

    # ---- public API ---------------------------------------------------------

    def is_subtype(self, child: str, parent: str) -> bool:
        if child == parent:
            return True
        current: Optional[str] = child
        visited: Set[str] = set()
        while current is not None and current not in visited:
            visited.add(current)
            if current == parent:
                return True
            current = self._parent.get(current)
        return False

    def get_parent(self, error_type: str) -> Optional[str]:
        return self._parent.get(error_type)

    def get_attributes(self, error_type: str) -> Dict[str, str]:
        if error_type in self._attributes:
            return dict(self._attributes[error_type])
        # walk up
        current: Optional[str] = error_type
        while current is not None:
            if current in self._attributes:
                return dict(self._attributes[current])
            current = self._parent.get(current)
        return {"message": "string", "name": "string", "stack": "string | undefined"}

    def get_model(self, error_type: str) -> Optional[TSErrorModel]:
        return self._models.get(error_type)

    def register_custom_error(
        self,
        error_type: str,
        parent: str = "Error",
        attributes: Optional[Dict[str, str]] = None,
        common_causes: Optional[List[str]] = None,
    ) -> None:
        """Register a custom error class (e.g., ``class MyError extends Error``)."""
        self._add(error_type, parent)
        parent_attrs = self.get_attributes(parent)
        merged = {**parent_attrs, **(attributes or {})}
        self._attributes[error_type] = merged
        self._models[error_type] = TSErrorModel(
            error_type=error_type,
            parent_type=parent,
            attributes=merged,
            common_causes=common_causes or [],
        )

    def model_promise_rejection(
        self,
        rejected_type: str,
        message: str,
    ) -> ExceptionInfo:
        """Model an unhandled Promise rejection."""
        return ExceptionInfo(
            exception_type=rejected_type,
            message=message,
            attributes={
                "promise_state": "rejected",
                "unhandled": True,
            },
        )

    def model_catch_handler(
        self,
        rejection: ExceptionInfo,
        handler_catches: List[str],
    ) -> Optional[str]:
        """Determine which ``.catch`` handler type matches a rejection."""
        for catch_type in handler_catches:
            if self.is_subtype(rejection.exception_type, catch_type):
                return catch_type
        return None

    def model_async_try_catch(
        self,
        async_body_exceptions: List[ExceptionInfo],
        catch_types: List[str],
    ) -> Tuple[List[ExceptionInfo], List[ExceptionInfo]]:
        """Model try/catch in an async function.

        Returns (caught, uncaught) exceptions.
        """
        caught: List[ExceptionInfo] = []
        uncaught: List[ExceptionInfo] = []
        for exc in async_body_exceptions:
            is_caught = False
            for ct in catch_types:
                if self.is_subtype(exc.exception_type, ct):
                    is_caught = True
                    break
            if is_caught:
                caught.append(exc)
            else:
                uncaught.append(exc)
        return caught, uncaught

    def model_promise_all_settled(
        self, results: List[ExceptionInfo]
    ) -> List[Dict[str, Any]]:
        """Model Promise.allSettled: every result is either fulfilled or rejected."""
        output: List[Dict[str, Any]] = []
        for r in results:
            if r.exception_type == "__fulfilled__":
                output.append({"status": "fulfilled", "value": r.message})
            else:
                output.append({"status": "rejected", "reason": r})
        return output

    def map_to_refinement_type(self, error_type: str) -> RefinementType:
        """Map a TS error type to a RefinementType."""
        return RefinementType(
            base_type=error_type,
            predicate=f"instanceof {error_type}",
            nullity=NullityTag.DEFINITELY_NOT_NULL,
        )

    def get_all_errors(self) -> List[str]:
        return sorted(self._parent.keys())


# ---------------------------------------------------------------------------
# 8. ErrorCodeModels
# ---------------------------------------------------------------------------

@dataclass
class HttpStatus:
    """Model of an HTTP status code."""
    code: int
    name: str
    description: str
    is_error: bool
    is_retryable: bool
    suggested_handling: str


@dataclass
class ErrnoInfo:
    """Model of a POSIX errno code."""
    code: int
    name: str
    description: str
    python_exception: str


@dataclass
class ExitCodeInfo:
    """Model of a process exit code."""
    code: int
    name: str
    description: str
    is_error: bool
    signal_name: Optional[str] = None


class ErrorCodeModels:
    """Comprehensive models for HTTP status codes, POSIX errno codes, and
    process exit codes, with mappings to Python exception types."""

    def __init__(self) -> None:
        self._http: Dict[int, HttpStatus] = {}
        self._errno: Dict[int, ErrnoInfo] = {}
        self._exit: Dict[int, ExitCodeInfo] = {}
        self._build_http()
        self._build_errno()
        self._build_exit()

    # ---- HTTP status codes --------------------------------------------------

    def _build_http(self) -> None:
        statuses: List[HttpStatus] = [
            # 1xx Informational
            HttpStatus(100, "Continue", "Server received request headers; client should proceed to send body", False, False, "Continue sending the request body"),
            HttpStatus(101, "Switching Protocols", "Server is switching to the protocol requested in the Upgrade header", False, False, "Proceed with the new protocol"),
            HttpStatus(102, "Processing", "Server has received and is processing the request (WebDAV)", False, False, "Wait for final response"),
            HttpStatus(103, "Early Hints", "Returns response headers before final HTTP message", False, False, "Preload indicated resources"),
            # 2xx Success
            HttpStatus(200, "OK", "Standard successful response", False, False, "Process response body"),
            HttpStatus(201, "Created", "Request succeeded and a new resource was created", False, False, "Use Location header for new resource URL"),
            HttpStatus(202, "Accepted", "Request accepted for processing but not yet completed", False, False, "Poll for completion status"),
            HttpStatus(204, "No Content", "Request succeeded with no response body", False, False, "No response body expected"),
            HttpStatus(206, "Partial Content", "Server is delivering part of the resource (range request)", False, False, "Process partial content; continue range requests"),
            # 3xx Redirection
            HttpStatus(301, "Moved Permanently", "Resource has been permanently moved to a new URL", False, False, "Update bookmarks/links to the new URL"),
            HttpStatus(302, "Found", "Resource temporarily located at a different URL", False, False, "Follow the redirect; do not update bookmarks"),
            HttpStatus(303, "See Other", "Response to the request can be found at another URL via GET", False, False, "Issue GET request to the Location URL"),
            HttpStatus(304, "Not Modified", "Resource has not been modified since the last request", False, False, "Use cached version of the resource"),
            HttpStatus(307, "Temporary Redirect", "Request should be repeated with the same method at another URL", False, False, "Repeat request at Location URL with same method"),
            HttpStatus(308, "Permanent Redirect", "Request should be repeated at another URL permanently", False, False, "Update links and repeat with same method"),
            # 4xx Client Error
            HttpStatus(400, "Bad Request", "Server cannot process request due to client error (malformed syntax)", True, False, "Fix the request syntax or parameters"),
            HttpStatus(401, "Unauthorized", "Authentication is required and has failed or not been provided", True, False, "Provide valid authentication credentials"),
            HttpStatus(403, "Forbidden", "Server understood the request but refuses to authorize it", True, False, "Request appropriate permissions or use different credentials"),
            HttpStatus(404, "Not Found", "Requested resource could not be found on the server", True, False, "Verify URL; resource may have been moved or deleted"),
            HttpStatus(405, "Method Not Allowed", "HTTP method is not allowed for the requested resource", True, False, "Use an allowed HTTP method (check Allow header)"),
            HttpStatus(406, "Not Acceptable", "Server cannot produce a response matching the Accept headers", True, False, "Adjust Accept headers to match server capabilities"),
            HttpStatus(408, "Request Timeout", "Server timed out waiting for the request", True, True, "Retry the request"),
            HttpStatus(409, "Conflict", "Request conflicts with current state of the target resource", True, False, "Resolve the conflict and retry"),
            HttpStatus(410, "Gone", "Resource is no longer available and will not be available again", True, False, "Remove references to this resource"),
            HttpStatus(411, "Length Required", "Content-Length header is required but not provided", True, False, "Include Content-Length header in request"),
            HttpStatus(412, "Precondition Failed", "One or more precondition headers evaluated to false", True, False, "Check and update precondition headers"),
            HttpStatus(413, "Payload Too Large", "Request entity is larger than the server is willing to process", True, False, "Reduce payload size or use chunked transfer"),
            HttpStatus(414, "URI Too Long", "Request URI is longer than the server is willing to interpret", True, False, "Shorten the URI; use POST with body instead"),
            HttpStatus(415, "Unsupported Media Type", "Media format of the request is not supported", True, False, "Use a supported Content-Type"),
            HttpStatus(416, "Range Not Satisfiable", "Range specified in the Range header cannot be fulfilled", True, False, "Check Content-Range; adjust byte range"),
            HttpStatus(418, "I'm a Teapot", "Server refuses to brew coffee because it is a teapot (RFC 2324)", True, False, "Find a coffee pot instead"),
            HttpStatus(422, "Unprocessable Entity", "Request is well-formed but semantically erroneous (WebDAV)", True, False, "Fix semantic errors in the request body"),
            HttpStatus(425, "Too Early", "Server is unwilling to process a request that might be replayed", True, True, "Retry after TLS handshake completes"),
            HttpStatus(429, "Too Many Requests", "Rate limit exceeded; too many requests in a given time period", True, True, "Wait for Retry-After period; implement rate limiting"),
            HttpStatus(431, "Request Header Fields Too Large", "Server refuses request because headers are too large", True, False, "Reduce the size of request headers"),
            HttpStatus(451, "Unavailable For Legal Reasons", "Resource unavailable due to legal demand", True, False, "Content is legally restricted"),
            # 5xx Server Error
            HttpStatus(500, "Internal Server Error", "Generic server error; an unexpected condition was encountered", True, True, "Retry; report to server administrator if persistent"),
            HttpStatus(501, "Not Implemented", "Server does not support the functionality required to fulfill the request", True, False, "Use a different server or endpoint that supports the operation"),
            HttpStatus(502, "Bad Gateway", "Server acting as gateway received invalid response from upstream", True, True, "Retry after a short delay; check upstream server health"),
            HttpStatus(503, "Service Unavailable", "Server is temporarily unable to handle the request (overloaded or maintenance)", True, True, "Retry after Retry-After period; implement circuit breaker"),
            HttpStatus(504, "Gateway Timeout", "Server acting as gateway did not receive a timely response from upstream", True, True, "Retry with increased timeout; check upstream server"),
            HttpStatus(505, "HTTP Version Not Supported", "Server does not support the HTTP version used in the request", True, False, "Use a supported HTTP version"),
            HttpStatus(507, "Insufficient Storage", "Server cannot store the representation needed to complete the request (WebDAV)", True, False, "Free up storage on the server"),
            HttpStatus(508, "Loop Detected", "Server detected an infinite loop while processing the request (WebDAV)", True, False, "Fix the loop in server configuration"),
            HttpStatus(511, "Network Authentication Required", "Client needs to authenticate to gain network access", True, False, "Authenticate with the network (e.g., captive portal)"),
        ]
        for s in statuses:
            self._http[s.code] = s

    # ---- errno codes --------------------------------------------------------

    def _build_errno(self) -> None:
        errnos: List[ErrnoInfo] = [
            ErrnoInfo(1, "EPERM", "Operation not permitted", "PermissionError"),
            ErrnoInfo(2, "ENOENT", "No such file or directory", "FileNotFoundError"),
            ErrnoInfo(3, "ESRCH", "No such process", "ProcessLookupError"),
            ErrnoInfo(4, "EINTR", "Interrupted system call", "InterruptedError"),
            ErrnoInfo(5, "EIO", "Input/output error", "OSError"),
            ErrnoInfo(6, "ENXIO", "No such device or address", "OSError"),
            ErrnoInfo(7, "E2BIG", "Argument list too long", "OSError"),
            ErrnoInfo(8, "ENOEXEC", "Exec format error", "OSError"),
            ErrnoInfo(9, "EBADF", "Bad file descriptor", "OSError"),
            ErrnoInfo(10, "ECHILD", "No child processes", "ChildProcessError"),
            ErrnoInfo(11, "EAGAIN", "Resource temporarily unavailable (EWOULDBLOCK)", "BlockingIOError"),
            ErrnoInfo(12, "ENOMEM", "Cannot allocate memory", "MemoryError"),
            ErrnoInfo(13, "EACCES", "Permission denied", "PermissionError"),
            ErrnoInfo(14, "EFAULT", "Bad address", "OSError"),
            ErrnoInfo(16, "EBUSY", "Device or resource busy", "OSError"),
            ErrnoInfo(17, "EEXIST", "File exists", "FileExistsError"),
            ErrnoInfo(18, "EXDEV", "Invalid cross-device link", "OSError"),
            ErrnoInfo(19, "ENODEV", "No such device", "OSError"),
            ErrnoInfo(20, "ENOTDIR", "Not a directory", "NotADirectoryError"),
            ErrnoInfo(21, "EISDIR", "Is a directory", "IsADirectoryError"),
            ErrnoInfo(22, "EINVAL", "Invalid argument", "OSError"),
            ErrnoInfo(23, "ENFILE", "Too many open files in system", "OSError"),
            ErrnoInfo(24, "EMFILE", "Too many open files", "OSError"),
            ErrnoInfo(25, "ENOTTY", "Inappropriate ioctl for device", "OSError"),
            ErrnoInfo(26, "ETXTBSY", "Text file busy", "OSError"),
            ErrnoInfo(27, "EFBIG", "File too large", "OSError"),
            ErrnoInfo(28, "ENOSPC", "No space left on device", "OSError"),
            ErrnoInfo(29, "ESPIPE", "Illegal seek", "OSError"),
            ErrnoInfo(30, "EROFS", "Read-only file system", "OSError"),
            ErrnoInfo(32, "EPIPE", "Broken pipe", "BrokenPipeError"),
            ErrnoInfo(33, "EDOM", "Numerical argument out of domain", "ValueError"),
            ErrnoInfo(34, "ERANGE", "Numerical result out of range", "OverflowError"),
            ErrnoInfo(35, "EDEADLK", "Resource deadlock avoided", "OSError"),
            ErrnoInfo(36, "ENAMETOOLONG", "File name too long", "OSError"),
            ErrnoInfo(37, "ENOLCK", "No locks available", "OSError"),
            ErrnoInfo(38, "ENOSYS", "Function not implemented", "OSError"),
            ErrnoInfo(39, "ENOTEMPTY", "Directory not empty", "OSError"),
            ErrnoInfo(40, "ELOOP", "Too many levels of symbolic links", "OSError"),
            ErrnoInfo(42, "ENOMSG", "No message of desired type", "OSError"),
            ErrnoInfo(43, "EIDRM", "Identifier removed", "OSError"),
            ErrnoInfo(61, "ENODATA", "No data available", "OSError"),
            ErrnoInfo(62, "ETIME", "Timer expired", "OSError"),
            ErrnoInfo(75, "EOVERFLOW", "Value too large for defined data type", "OverflowError"),
            ErrnoInfo(88, "ENOTSOCK", "Socket operation on non-socket", "OSError"),
            ErrnoInfo(89, "EDESTADDRREQ", "Destination address required", "OSError"),
            ErrnoInfo(90, "EMSGSIZE", "Message too long", "OSError"),
            ErrnoInfo(91, "EPROTOTYPE", "Protocol wrong type for socket", "OSError"),
            ErrnoInfo(92, "ENOPROTOOPT", "Protocol not available", "OSError"),
            ErrnoInfo(93, "EPROTONOSUPPORT", "Protocol not supported", "OSError"),
            ErrnoInfo(95, "EOPNOTSUPP", "Operation not supported", "OSError"),
            ErrnoInfo(97, "EAFNOSUPPORT", "Address family not supported by protocol", "OSError"),
            ErrnoInfo(98, "EADDRINUSE", "Address already in use", "OSError"),
            ErrnoInfo(99, "EADDRNOTAVAIL", "Cannot assign requested address", "OSError"),
            ErrnoInfo(100, "ENETDOWN", "Network is down", "OSError"),
            ErrnoInfo(101, "ENETUNREACH", "Network is unreachable", "OSError"),
            ErrnoInfo(102, "ENETRESET", "Network dropped connection on reset", "ConnectionResetError"),
            ErrnoInfo(103, "ECONNABORTED", "Software caused connection abort", "ConnectionAbortedError"),
            ErrnoInfo(104, "ECONNRESET", "Connection reset by peer", "ConnectionResetError"),
            ErrnoInfo(105, "ENOBUFS", "No buffer space available", "OSError"),
            ErrnoInfo(106, "EISCONN", "Transport endpoint is already connected", "OSError"),
            ErrnoInfo(107, "ENOTCONN", "Transport endpoint is not connected", "OSError"),
            ErrnoInfo(110, "ETIMEDOUT", "Connection timed out", "TimeoutError"),
            ErrnoInfo(111, "ECONNREFUSED", "Connection refused", "ConnectionRefusedError"),
            ErrnoInfo(112, "EHOSTDOWN", "Host is down", "OSError"),
            ErrnoInfo(113, "EHOSTUNREACH", "No route to host", "OSError"),
            ErrnoInfo(114, "EALREADY", "Operation already in progress", "OSError"),
            ErrnoInfo(115, "EINPROGRESS", "Operation now in progress", "BlockingIOError"),
        ]
        for e in errnos:
            self._errno[e.code] = e

    # ---- exit codes ---------------------------------------------------------

    def _build_exit(self) -> None:
        exits: List[ExitCodeInfo] = [
            ExitCodeInfo(0, "SUCCESS", "Successful execution", False),
            ExitCodeInfo(1, "GENERAL_ERROR", "General error / catchall for errors", True),
            ExitCodeInfo(2, "MISUSE", "Misuse of shell command (e.g., invalid option)", True),
            ExitCodeInfo(126, "NOT_EXECUTABLE", "Command invoked cannot execute (permission or not executable)", True),
            ExitCodeInfo(127, "NOT_FOUND", "Command not found", True),
            ExitCodeInfo(128, "INVALID_EXIT", "Invalid exit argument (exit takes only integer 0-255)", True),
            ExitCodeInfo(129, "SIGHUP", "Terminated by SIGHUP (hangup)", True, "SIGHUP"),
            ExitCodeInfo(130, "SIGINT", "Terminated by SIGINT (Ctrl+C)", True, "SIGINT"),
            ExitCodeInfo(131, "SIGQUIT", "Terminated by SIGQUIT (Ctrl+\\)", True, "SIGQUIT"),
            ExitCodeInfo(132, "SIGILL", "Terminated by SIGILL (illegal instruction)", True, "SIGILL"),
            ExitCodeInfo(133, "SIGTRAP", "Terminated by SIGTRAP (trace/breakpoint trap)", True, "SIGTRAP"),
            ExitCodeInfo(134, "SIGABRT", "Terminated by SIGABRT (abort)", True, "SIGABRT"),
            ExitCodeInfo(135, "SIGBUS", "Terminated by SIGBUS (bus error)", True, "SIGBUS"),
            ExitCodeInfo(136, "SIGFPE", "Terminated by SIGFPE (floating point exception)", True, "SIGFPE"),
            ExitCodeInfo(137, "SIGKILL", "Terminated by SIGKILL (force kill, cannot be caught)", True, "SIGKILL"),
            ExitCodeInfo(139, "SIGSEGV", "Terminated by SIGSEGV (segmentation fault)", True, "SIGSEGV"),
            ExitCodeInfo(141, "SIGPIPE", "Terminated by SIGPIPE (broken pipe)", True, "SIGPIPE"),
            ExitCodeInfo(143, "SIGTERM", "Terminated by SIGTERM (graceful termination)", True, "SIGTERM"),
        ]
        for e in exits:
            self._exit[e.code] = e

    # ---- public API ---------------------------------------------------------

    def get_http_status(self, code: int) -> Optional[HttpStatus]:
        """Return the HttpStatus for *code*, or None if unknown."""
        return self._http.get(code)

    def get_errno(self, code: int) -> Optional[ErrnoInfo]:
        """Return the ErrnoInfo for *code*, or None if unknown."""
        return self._errno.get(code)

    def get_exit_code(self, code: int) -> Optional[ExitCodeInfo]:
        """Return the ExitCodeInfo for *code*, or None if unknown."""
        if code in self._exit:
            return self._exit[code]
        # compute signal-based exit codes: 128 + N
        if 128 < code < 165:
            signal_num = code - 128
            signal_names: Dict[int, str] = {
                1: "SIGHUP", 2: "SIGINT", 3: "SIGQUIT", 4: "SIGILL",
                5: "SIGTRAP", 6: "SIGABRT", 7: "SIGBUS", 8: "SIGFPE",
                9: "SIGKILL", 10: "SIGUSR1", 11: "SIGSEGV", 12: "SIGUSR2",
                13: "SIGPIPE", 14: "SIGALRM", 15: "SIGTERM",
            }
            sig = signal_names.get(signal_num, f"SIG{signal_num}")
            return ExitCodeInfo(
                code=code,
                name=sig,
                description=f"Terminated by {sig} (signal {signal_num})",
                is_error=True,
                signal_name=sig,
            )
        return None

    def is_retryable_http(self, code: int) -> bool:
        """Return True if the HTTP status code is retryable."""
        status = self._http.get(code)
        if status is not None:
            return status.is_retryable
        # default: 5xx are generally retryable, 429 is retryable
        if code == 429:
            return True
        if 500 <= code < 600:
            return True
        return False

    def errno_to_python_exception(self, errno_code: int) -> str:
        """Map an errno code to the Python exception class name."""
        info = self._errno.get(errno_code)
        if info is not None:
            return info.python_exception
        return "OSError"

    def classify_http_error(self, code: int) -> str:
        """Return a broad category for the HTTP status code."""
        if 100 <= code < 200:
            return "informational"
        elif 200 <= code < 300:
            return "success"
        elif 300 <= code < 400:
            return "redirection"
        elif 400 <= code < 500:
            return "client_error"
        elif 500 <= code < 600:
            return "server_error"
        else:
            return "unknown"

    def get_all_http_codes(self) -> List[int]:
        return sorted(self._http.keys())

    def get_all_errno_codes(self) -> List[int]:
        return sorted(self._errno.keys())

    def get_all_exit_codes(self) -> List[int]:
        return sorted(self._exit.keys())

    def get_retryable_http_codes(self) -> List[int]:
        return [code for code, s in self._http.items() if s.is_retryable]

    def get_error_http_codes(self) -> List[int]:
        return [code for code, s in self._http.items() if s.is_error]
