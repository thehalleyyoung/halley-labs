"""
Exception-based refinement inference for Python programs.

Python uses EAFP (Easier to Ask Forgiveness than Permission) extensively,
meaning try/except blocks create implicit refinements about program state.
When an operation succeeds without raising, we know something about the
operands. When we catch a specific exception, we know the negation.

This module models how Python exceptions create implicit refinements in
the refinement type system, bridging the gap between EAFP-style code and
the predicates in our refinement lattice.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
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


# ===================================================================
# Supporting data structures
# ===================================================================


class RaisingOpKind(Enum):
    """Classification of operations that may raise exceptions."""
    SUBSCRIPT = auto()        # d[key], lst[idx]
    ATTRIBUTE = auto()        # obj.attr
    CALL = auto()             # f(x)  — int(), float(), open(), next(), ...
    BINARY_OP = auto()        # x / y, x % y
    ITERATION = auto()        # for x in it / next(it)
    UNPACK = auto()           # a, b = seq
    COMPARISON = auto()       # custom __eq__ may raise
    FILE_IO = auto()          # f.read(), f.write()
    IMPORT = auto()           # import m


@dataclass
class RaisingOp:
    """An operation that may raise an exception.

    Attributes:
        node: The AST node of the operation.
        kind: Classification of the raising operation.
        target_var: Primary variable involved (e.g. the dict in d[k]).
        secondary_var: Secondary variable (e.g. the key in d[k]).
        possible_exceptions: Set of exception type names this op may raise.
        implied_predicates: Predicates that hold when the op succeeds.
    """
    node: ast.expr
    kind: RaisingOpKind
    target_var: Optional[str] = None
    secondary_var: Optional[str] = None
    possible_exceptions: Set[str] = field(default_factory=set)
    implied_predicates: List[Pred] = field(default_factory=list)

    def __repr__(self) -> str:
        loc = f"L{self.node.lineno}" if hasattr(self.node, "lineno") else "?"
        return (
            f"RaisingOp({self.kind.name}, target={self.target_var}, "
            f"exc={self.possible_exceptions}, at={loc})"
        )


@dataclass
class ExceptionMapping:
    """Maps an exception type to the conditions that cause it.

    For example, KeyError is caused by accessing a missing key in a dict.
    The negation of this condition — the key IS present — becomes a
    refinement when the operation succeeds.

    Attributes:
        exception_type: Fully qualified exception type name.
        description: Human-readable description of the condition.
        cause_predicate_factory: Given target and secondary vars, produces
            the predicate that would cause this exception.
        negated_predicate_factory: Given target and secondary vars, produces
            the predicate that holds when the exception does NOT occur.
        applicable_op_kinds: Which operation kinds this mapping applies to.
    """
    exception_type: str
    description: str
    cause_predicate_factory: Optional[Callable[..., Pred]] = None
    negated_predicate_factory: Optional[Callable[..., Pred]] = None
    applicable_op_kinds: Set[RaisingOpKind] = field(default_factory=set)


@dataclass
class AnalysisState:
    """Tracks variable refinements during analysis.

    Attributes:
        bindings: Maps variable names to their current refined types.
        path_predicates: Predicates known to hold on the current path.
        exception_context: Stack of exception handlers we are inside.
        assumptions: Additional assumptions from enclosing scopes.
    """
    bindings: Dict[str, RefType] = field(default_factory=dict)
    path_predicates: List[Pred] = field(default_factory=list)
    exception_context: List[str] = field(default_factory=list)
    assumptions: List[Pred] = field(default_factory=list)

    def copy(self) -> AnalysisState:
        return AnalysisState(
            bindings=dict(self.bindings),
            path_predicates=list(self.path_predicates),
            exception_context=list(self.exception_context),
            assumptions=list(self.assumptions),
        )

    def add_predicate(self, pred: Pred) -> None:
        """Add a predicate to the current path."""
        if pred.op != PredOp.TRUE:
            self.path_predicates.append(pred)

    def refine_variable(self, var: str, pred: Pred) -> None:
        """Strengthen the refinement on a variable."""
        if var in self.bindings:
            existing = self.bindings[var]
            combined = existing.pred.and_(pred)
            self.bindings[var] = existing.with_pred(combined)
        else:
            self.bindings[var] = RefType("ν", ANY_TYPE, pred)

    def join(self, other: AnalysisState) -> AnalysisState:
        """Join two analysis states (at control-flow merge points)."""
        result = AnalysisState()
        all_vars = set(self.bindings.keys()) | set(other.bindings.keys())
        for v in all_vars:
            self_ty = self.bindings.get(v)
            other_ty = other.bindings.get(v)
            if self_ty is not None and other_ty is not None:
                joined_pred = self_ty.pred.or_(other_ty.pred)
                base = self_ty.base if self_ty.base == other_ty.base else ANY_TYPE
                result.bindings[v] = RefType("ν", base, joined_pred)
            elif self_ty is not None:
                result.bindings[v] = self_ty
            else:
                assert other_ty is not None
                result.bindings[v] = other_ty
        # Path predicates: only keep those common to both paths
        self_preds = set(id(p) for p in self.path_predicates)
        for p in other.path_predicates:
            if id(p) in self_preds:
                result.path_predicates.append(p)
        return result

    def meet(self, other: AnalysisState) -> AnalysisState:
        """Meet two analysis states (conjunction)."""
        result = self.copy()
        for v, ty in other.bindings.items():
            result.refine_variable(v, ty.pred)
        result.path_predicates.extend(other.path_predicates)
        return result

    def get_all_predicates(self) -> Pred:
        """Conjoin all path predicates into a single predicate."""
        result = Pred.true_()
        for p in self.path_predicates:
            result = result.and_(p)
        for p in self.assumptions:
            result = result.and_(p)
        return result


@dataclass
class ComparisonResult:
    """Result of comparing LBYL (Look Before You Leap) vs EAFP patterns.

    Attributes:
        lbyl_predicates: Predicates derived from the LBYL guard.
        eafp_predicates: Predicates derived from the EAFP try/except.
        equivalent: Whether the two sets of predicates are equivalent.
        lbyl_stronger: Whether LBYL predicates imply EAFP predicates
            but not vice versa.
        eafp_stronger: Whether EAFP predicates imply LBYL predicates
            but not vice versa.
        notes: Human-readable notes about differences.
    """
    lbyl_predicates: List[Pred] = field(default_factory=list)
    eafp_predicates: List[Pred] = field(default_factory=list)
    equivalent: bool = False
    lbyl_stronger: bool = False
    eafp_stronger: bool = False
    notes: List[str] = field(default_factory=list)


# ===================================================================
# Exception refinement database
# ===================================================================

# Maps exception type names to information about what predicates they
# negate.  When we catch ExcType, we know the operation failed and the
# cause predicate holds.  When the operation succeeds (else branch or
# after the try body with no exception), the negated predicate holds.

EXCEPTION_REFINEMENT_DB: Dict[str, ExceptionMapping] = {
    "KeyError": ExceptionMapping(
        exception_type="KeyError",
        description="Raised when a dict key is not found.",
        applicable_op_kinds={RaisingOpKind.SUBSCRIPT},
    ),
    "IndexError": ExceptionMapping(
        exception_type="IndexError",
        description="Raised when a sequence index is out of range.",
        applicable_op_kinds={RaisingOpKind.SUBSCRIPT},
    ),
    "AttributeError": ExceptionMapping(
        exception_type="AttributeError",
        description="Raised when an attribute reference or assignment fails.",
        applicable_op_kinds={RaisingOpKind.ATTRIBUTE},
    ),
    "TypeError": ExceptionMapping(
        exception_type="TypeError",
        description="Raised when an operation is applied to an object "
                    "of inappropriate type.",
        applicable_op_kinds={
            RaisingOpKind.CALL, RaisingOpKind.BINARY_OP,
            RaisingOpKind.SUBSCRIPT, RaisingOpKind.ITERATION,
            RaisingOpKind.UNPACK,
        },
    ),
    "ValueError": ExceptionMapping(
        exception_type="ValueError",
        description="Raised when an operation receives an argument of the "
                    "right type but inappropriate value.",
        applicable_op_kinds={RaisingOpKind.CALL, RaisingOpKind.UNPACK},
    ),
    "StopIteration": ExceptionMapping(
        exception_type="StopIteration",
        description="Raised by next() when the iterator is exhausted.",
        applicable_op_kinds={RaisingOpKind.CALL, RaisingOpKind.ITERATION},
    ),
    "FileNotFoundError": ExceptionMapping(
        exception_type="FileNotFoundError",
        description="Raised when a file or directory is requested but "
                    "doesn't exist.",
        applicable_op_kinds={RaisingOpKind.CALL, RaisingOpKind.FILE_IO},
    ),
    "ZeroDivisionError": ExceptionMapping(
        exception_type="ZeroDivisionError",
        description="Raised when dividing or modulo by zero.",
        applicable_op_kinds={RaisingOpKind.BINARY_OP},
    ),
    "OverflowError": ExceptionMapping(
        exception_type="OverflowError",
        description="Raised when an arithmetic operation is too large.",
        applicable_op_kinds={RaisingOpKind.BINARY_OP, RaisingOpKind.CALL},
    ),
    "UnicodeDecodeError": ExceptionMapping(
        exception_type="UnicodeDecodeError",
        description="Raised when a Unicode-related encoding or decoding "
                    "error occurs.",
        applicable_op_kinds={RaisingOpKind.CALL, RaisingOpKind.FILE_IO},
    ),
}


# ===================================================================
# Helper: AST utilities
# ===================================================================

def _get_name(node: ast.expr) -> Optional[str]:
    """Extract a simple variable name from an AST node, if possible."""
    if isinstance(node, ast.Name):
        return node.id
    return None


def _get_attr_chain(node: ast.expr) -> Optional[str]:
    """Extract a dotted name like 'obj.attr' from an AST node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _get_attr_chain(node.value)
        if base is not None:
            return f"{base}.{node.attr}"
    return None


def _get_string_literal(node: ast.expr) -> Optional[str]:
    """Extract a string literal from an AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _get_int_literal(node: ast.expr) -> Optional[int]:
    """Extract an integer literal from an AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    return None


def _is_call_to(node: ast.expr, name: str) -> bool:
    """Check if node is a call to a function with the given name."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name) and func.id == name:
        return True
    if isinstance(func, ast.Attribute) and func.attr == name:
        return True
    return False


def _get_call_name(node: ast.Call) -> Optional[str]:
    """Get the name of the called function."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _extract_handler_exception_types(
    handler: ast.ExceptHandler,
) -> List[str]:
    """Extract the exception type names from an except handler."""
    if handler.type is None:
        return ["BaseException"]
    if isinstance(handler.type, ast.Name):
        return [handler.type.id]
    if isinstance(handler.type, ast.Tuple):
        result = []
        for elt in handler.type.elts:
            if isinstance(elt, ast.Name):
                result.append(elt.id)
        return result
    if isinstance(handler.type, ast.Attribute):
        chain = _get_attr_chain(handler.type)
        if chain:
            return [chain]
    return ["BaseException"]


# ===================================================================
# Main analyzer
# ===================================================================


class ExceptionRefinementAnalyzer:
    """Analyzes try/except blocks to derive refinement predicates.

    Python uses exceptions for flow control far more than other languages.
    try/except creates implicit refinements:

    - In the ``else`` branch of try/except, we know that NO exception
      was raised by the try body, so every potentially-raising operation
      succeeded.
    - In an ``except ExcType`` handler, we know that particular exception
      was raised, so the cause predicate holds.
    - After the entire try/except/else/finally, we merge the refined
      states from all paths.
    """

    def __init__(self) -> None:
        self._db = EXCEPTION_REFINEMENT_DB
        # Cache of function names known to raise specific exceptions.
        self._call_exception_map: Dict[str, Set[str]] = {
            "int": {"ValueError", "TypeError"},
            "float": {"ValueError", "TypeError"},
            "str": set(),
            "bool": set(),
            "list": {"TypeError"},
            "dict": {"TypeError", "ValueError"},
            "set": {"TypeError"},
            "tuple": {"TypeError"},
            "next": {"StopIteration"},
            "iter": {"TypeError"},
            "open": {"FileNotFoundError", "PermissionError", "OSError"},
            "len": {"TypeError"},
            "abs": {"TypeError"},
            "max": {"ValueError", "TypeError"},
            "min": {"ValueError", "TypeError"},
            "sum": {"TypeError"},
            "sorted": {"TypeError"},
            "reversed": {"TypeError"},
            "enumerate": {"TypeError"},
            "zip": {"TypeError"},
            "map": {"TypeError"},
            "filter": {"TypeError"},
            "getattr": {"AttributeError"},
            "setattr": {"AttributeError"},
            "delattr": {"AttributeError"},
            "hasattr": set(),
            "isinstance": set(),
            "issubclass": {"TypeError"},
            "divmod": {"ZeroDivisionError", "TypeError"},
            "pow": {"ZeroDivisionError", "TypeError", "ValueError"},
            "round": {"TypeError"},
            "hash": {"TypeError"},
            "json.loads": {"ValueError"},
            "json.dumps": {"TypeError", "ValueError"},
        }
        # Maps method names to possible exceptions.
        self._method_exception_map: Dict[str, Set[str]] = {
            "index": {"ValueError"},
            "remove": {"ValueError"},
            "pop": {"KeyError", "IndexError"},
            "append": set(),
            "extend": {"TypeError"},
            "insert": set(),
            "sort": {"TypeError"},
            "update": {"TypeError"},
            "read": {"IOError", "OSError", "UnicodeDecodeError"},
            "write": {"IOError", "OSError"},
            "readline": {"IOError", "OSError", "UnicodeDecodeError"},
            "readlines": {"IOError", "OSError", "UnicodeDecodeError"},
            "close": set(),
            "seek": {"IOError", "OSError"},
            "tell": {"IOError", "OSError"},
            "encode": {"UnicodeEncodeError"},
            "decode": {"UnicodeDecodeError"},
            "split": set(),
            "join": {"TypeError"},
            "format": {"KeyError", "IndexError", "ValueError"},
            "startswith": {"TypeError"},
            "endswith": {"TypeError"},
            "__getitem__": {"KeyError", "IndexError", "TypeError"},
            "__setitem__": {"KeyError", "TypeError"},
            "__delitem__": {"KeyError", "IndexError", "TypeError"},
            "__contains__": {"TypeError"},
            "__iter__": {"TypeError"},
            "__next__": {"StopIteration"},
        }

    # ---------------------------------------------------------------
    # Public interface
    # ---------------------------------------------------------------

    def analyze_try_except(
        self, node: ast.Try, state: AnalysisState
    ) -> AnalysisState:
        """Analyze a try/except/else/finally block.

        Computes refined analysis states for each branch and merges them.

        The control flow is:
            try body → (success) → else body → finally
                     ↘ (exception) → matching handler → finally

        Returns:
            The merged AnalysisState after the entire try construct.
        """
        # 1. Analyze the try body to find raising operations.
        raising_ops = self._extract_potentially_raising_ops(node.body)

        # 2. Build the success (else-branch) state: all raising ops
        #    succeeded, so their success predicates hold.
        else_state = self._compute_else_refinements(
            node.body, node.handlers, state
        )

        # 3. Analyze each except handler.
        handler_states: List[AnalysisState] = []
        for handler in node.handlers:
            handler_state = self._analyze_handler(
                handler, raising_ops, state
            )
            handler_states.append(handler_state)

        # 4. If there is an explicit else block, further refine the
        #    else state by analyzing its statements.
        if node.orelse:
            else_state = self._analyze_else_body(
                node.orelse, else_state
            )

        # 5. Merge all outgoing states.
        if handler_states:
            merged = handler_states[0]
            for hs in handler_states[1:]:
                merged = merged.join(hs)
            result = merged.join(else_state)
        else:
            result = else_state

        # 6. Apply finally-block effects (finally always runs, so it
        #    does not refine — it only adds side effects).
        if node.finalbody:
            result = self._analyze_finally(node.finalbody, result)

        return result

    def infer_exception_refinement(
        self, expr: ast.expr, exc_type: str
    ) -> Optional[Pred]:
        """Given that ``expr`` did NOT raise ``exc_type``, infer predicates.

        This is the core inference rule:
            If ``expr`` can raise ``ExcType`` only when condition C holds,
            then not raising implies ¬C.

        Examples:
            - ``d[k]`` not raising KeyError → k ∈ d
            - ``int(s)`` not raising ValueError → s is numeric
            - ``next(it)`` not raising StopIteration → iterator not exhausted
            - ``x / y`` not raising ZeroDivisionError → y ≠ 0
        """
        # Subscript access: d[k]
        if isinstance(expr, ast.Subscript):
            return self._infer_subscript_refinement(expr, exc_type)

        # Attribute access: obj.attr
        if isinstance(expr, ast.Attribute):
            return self._infer_attribute_refinement(expr, exc_type)

        # Function/method call
        if isinstance(expr, ast.Call):
            return self._infer_call_refinement(expr, exc_type)

        # Binary operation: x / y, x % y
        if isinstance(expr, ast.BinOp):
            return self._infer_binop_refinement(expr, exc_type)

        # Starred unpacking: a, b = expr
        if isinstance(expr, ast.Starred):
            return self._infer_unpack_refinement(expr, exc_type)

        return None

    def model_eafp_pattern(
        self,
        try_body: List[ast.stmt],
        handler: ast.ExceptHandler,
    ) -> List[Pred]:
        """Model EAFP: success implies all operations succeeded.

        Examines the try body for potentially-raising operations and
        the handler for the caught exception type.  Returns predicates
        that hold when the try body executes without raising.

        This is the key insight: in Python's EAFP pattern, the try body
        is a conjunction of operations.  If none raised, every individual
        operation succeeded.
        """
        exc_types = _extract_handler_exception_types(handler)
        raising_ops = self._extract_potentially_raising_ops(try_body)

        success_predicates: List[Pred] = []
        for op in raising_ops:
            relevant_exc = op.possible_exceptions & set(exc_types)
            if not relevant_exc:
                continue
            for exc in relevant_exc:
                pred = self._map_exception_to_negated_predicate(op, exc)
                if pred is not None:
                    success_predicates.append(pred)

        return success_predicates

    def model_lbyl_vs_eafp(
        self,
        guard: ast.expr,
        guarded_body: List[ast.stmt],
    ) -> ComparisonResult:
        """Compare LBYL guard with EAFP try/except, showing equivalence.

        LBYL: ``if key in d: val = d[key]``
        EAFP: ``try: val = d[key] except KeyError: ...``

        Both produce the same refinement (key ∈ d) but through different
        mechanisms.  This method produces a ComparisonResult that documents
        the equivalence.
        """
        result = ComparisonResult()

        # 1. Extract predicates from the LBYL guard.
        lbyl_preds = self._extract_guard_predicates(guard)
        result.lbyl_predicates = lbyl_preds

        # 2. Extract EAFP predicates by assuming the body is a try block
        #    whose raising ops all succeeded.
        raising_ops = self._extract_potentially_raising_ops(guarded_body)
        eafp_preds: List[Pred] = []
        for op in raising_ops:
            for exc in op.possible_exceptions:
                pred = self._map_exception_to_negated_predicate(op, exc)
                if pred is not None:
                    eafp_preds.append(pred)
        result.eafp_predicates = eafp_preds

        # 3. Compare the predicate sets.
        lbyl_vars = set()
        for p in lbyl_preds:
            lbyl_vars |= p.free_vars()
        eafp_vars = set()
        for p in eafp_preds:
            eafp_vars |= p.free_vars()

        shared_vars = lbyl_vars & eafp_vars

        if not lbyl_preds and not eafp_preds:
            result.equivalent = True
            result.notes.append("Both patterns produce no refinements.")
        elif not lbyl_preds:
            result.eafp_stronger = True
            result.notes.append(
                "EAFP produces refinements but LBYL guard has none."
            )
        elif not eafp_preds:
            result.lbyl_stronger = True
            result.notes.append(
                "LBYL produces refinements but EAFP body has no raising ops."
            )
        else:
            # Heuristic comparison based on predicate structure.
            lbyl_ops = {p.op for p in lbyl_preds}
            eafp_ops = {p.op for p in eafp_preds}
            if lbyl_ops == eafp_ops and shared_vars:
                result.equivalent = True
                result.notes.append(
                    "LBYL and EAFP produce structurally similar refinements "
                    f"over shared variables: {shared_vars}."
                )
            elif lbyl_ops.issubset(eafp_ops):
                result.eafp_stronger = True
                result.notes.append(
                    "EAFP produces a superset of LBYL refinement operators."
                )
            elif eafp_ops.issubset(lbyl_ops):
                result.lbyl_stronger = True
                result.notes.append(
                    "LBYL produces a superset of EAFP refinement operators."
                )
            else:
                result.notes.append(
                    "LBYL and EAFP produce incomparable refinements."
                )

        return result

    # ---------------------------------------------------------------
    # Extraction of raising operations
    # ---------------------------------------------------------------

    def _extract_potentially_raising_ops(
        self, stmts: List[ast.stmt]
    ) -> List[RaisingOp]:
        """Find all operations in ``stmts`` that may raise exceptions.

        Walks the AST of the given statements and identifies subscripts,
        attribute accesses, calls, binary operations, and other constructs
        that may raise.
        """
        ops: List[RaisingOp] = []
        for stmt in stmts:
            self._walk_for_raising_ops(stmt, ops)
        return ops

    def _walk_for_raising_ops(
        self, node: ast.AST, ops: List[RaisingOp]
    ) -> None:
        """Recursively walk an AST node collecting raising operations."""
        if isinstance(node, ast.Subscript):
            ops.append(self._classify_subscript(node))

        elif isinstance(node, ast.Attribute):
            target = _get_name(node.value) or _get_attr_chain(node.value)
            if target:
                op = RaisingOp(
                    node=node,
                    kind=RaisingOpKind.ATTRIBUTE,
                    target_var=target,
                    secondary_var=node.attr,
                    possible_exceptions={"AttributeError"},
                    implied_predicates=[
                        Pred.hasattr_(target, node.attr)
                    ],
                )
                ops.append(op)

        elif isinstance(node, ast.Call):
            call_op = self._classify_call(node)
            if call_op is not None:
                ops.append(call_op)

        elif isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                right_var = _get_name(node.right)
                if right_var:
                    op = RaisingOp(
                        node=node,
                        kind=RaisingOpKind.BINARY_OP,
                        target_var=right_var,
                        possible_exceptions={"ZeroDivisionError"},
                        implied_predicates=[
                            Pred.var_neq(right_var, 0)
                        ],
                    )
                    ops.append(op)
            elif isinstance(node.op, ast.Pow):
                right_var = _get_name(node.right)
                left_var = _get_name(node.left)
                exc: Set[str] = {"OverflowError", "ZeroDivisionError"}
                preds: List[Pred] = []
                if right_var:
                    preds.append(Pred.var_ge(right_var, 0))
                op = RaisingOp(
                    node=node,
                    kind=RaisingOpKind.BINARY_OP,
                    target_var=left_var,
                    secondary_var=right_var,
                    possible_exceptions=exc,
                    implied_predicates=preds,
                )
                ops.append(op)

        elif isinstance(node, ast.Assign):
            # Check for unpacking: a, b = expr
            if (len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Tuple)):
                n_targets = len(node.targets[0].elts)
                val_var = _get_name(node.value)
                op = RaisingOp(
                    node=node.value if isinstance(node.value, ast.expr) else node,
                    kind=RaisingOpKind.UNPACK,
                    target_var=val_var,
                    possible_exceptions={"ValueError", "TypeError"},
                    implied_predicates=(
                        [Pred.len_eq(val_var, n_targets)] if val_var else []
                    ),
                )
                ops.append(op)

        # Recurse into child nodes.
        for child in ast.iter_child_nodes(node):
            # Don't recurse into nested try blocks — they have their
            # own exception handling.
            if isinstance(child, ast.Try):
                continue
            self._walk_for_raising_ops(child, ops)

    def _classify_subscript(self, node: ast.Subscript) -> RaisingOp:
        """Classify a subscript operation (d[k] or lst[i])."""
        target = _get_name(node.value) or _get_attr_chain(node.value)
        key_var = _get_name(node.slice) if isinstance(node.slice, ast.expr) else None
        key_int = None
        if isinstance(node.slice, ast.expr):
            key_int = _get_int_literal(node.slice)

        # We cannot statically tell dict from list in general, so we
        # report both KeyError and IndexError as possible.
        possible_exc: Set[str] = {"KeyError", "IndexError", "TypeError"}
        preds: List[Pred] = []

        if target and key_var:
            preds.append(Pred.hasattr_(target, "__getitem__"))
        if target and key_int is not None:
            # lst[n] succeeds → len(lst) > n (for positive n)
            if key_int >= 0:
                preds.append(Pred.len_gt(target, key_int))

        return RaisingOp(
            node=node,
            kind=RaisingOpKind.SUBSCRIPT,
            target_var=target,
            secondary_var=key_var,
            possible_exceptions=possible_exc,
            implied_predicates=preds,
        )

    def _classify_call(self, node: ast.Call) -> Optional[RaisingOp]:
        """Classify a function/method call."""
        call_name = _get_call_name(node)
        if call_name is None:
            return None

        # Check the call exception map.
        possible_exc: Set[str] = set()
        if call_name in self._call_exception_map:
            possible_exc = set(self._call_exception_map[call_name])
        elif isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            if method_name in self._method_exception_map:
                possible_exc = set(self._method_exception_map[method_name])
            else:
                # Unknown method — conservatively assume it can raise.
                possible_exc = {"Exception"}

        if not possible_exc:
            return None

        target_var: Optional[str] = None
        secondary_var: Optional[str] = None
        preds: List[Pred] = []

        # Extract target and arguments for common patterns.
        if isinstance(node.func, ast.Attribute):
            target_var = _get_name(node.func.value)
        if node.args:
            secondary_var = _get_name(node.args[0])

        # Specific patterns.
        if call_name in ("int", "float"):
            if secondary_var:
                preds.append(Pred.isinstance_(secondary_var, "str"))
        elif call_name == "next":
            if node.args:
                it_var = _get_name(node.args[0])
                if it_var:
                    target_var = it_var
                    preds.append(Pred.hasattr_(it_var, "__next__"))
        elif call_name == "open":
            if node.args:
                fname_var = _get_name(node.args[0])
                if fname_var:
                    target_var = fname_var
        elif call_name == "getattr":
            if len(node.args) >= 2:
                obj_var = _get_name(node.args[0])
                attr_lit = _get_string_literal(node.args[1])
                if obj_var and attr_lit:
                    target_var = obj_var
                    secondary_var = attr_lit
                    preds.append(Pred.hasattr_(obj_var, attr_lit))
                    if len(node.args) >= 3:
                        # getattr with default never raises AttributeError
                        possible_exc.discard("AttributeError")

        kind = RaisingOpKind.CALL
        if call_name in ("read", "write", "readline", "readlines"):
            kind = RaisingOpKind.FILE_IO
        elif call_name == "next":
            kind = RaisingOpKind.ITERATION

        return RaisingOp(
            node=node,
            kind=kind,
            target_var=target_var,
            secondary_var=secondary_var,
            possible_exceptions=possible_exc,
            implied_predicates=preds,
        )

    # ---------------------------------------------------------------
    # Exception → negated predicate mapping
    # ---------------------------------------------------------------

    def _map_exception_to_negated_predicate(
        self, op: RaisingOp, exc_type: str
    ) -> Optional[Pred]:
        """Map an exception type to the negated condition (success pred).

        Given a raising operation and the exception it might raise,
        returns the predicate that holds when that exception does NOT
        occur (i.e. the operation succeeds).
        """
        if exc_type not in self._db:
            return None

        target = op.target_var
        secondary = op.secondary_var

        if exc_type == "KeyError" and target:
            # d[k] didn't raise KeyError → key is present.
            if secondary:
                return Pred.hasattr_(target, secondary)
            return Pred.is_not_none(target)

        if exc_type == "IndexError" and target:
            # lst[i] didn't raise IndexError → index is valid.
            if secondary:
                return Pred.len_gt(target, 0)
            idx = self._extract_index_from_op(op)
            if idx is not None and idx >= 0:
                return Pred.len_gt(target, idx)
            return Pred.len_ge(target, 1)

        if exc_type == "AttributeError":
            if target and secondary:
                return Pred.hasattr_(target, secondary)
            return None

        if exc_type == "TypeError":
            if target:
                return Pred.is_not_none(target)
            return None

        if exc_type == "ValueError":
            return self._map_value_error_predicate(op)

        if exc_type == "StopIteration":
            if target:
                return Pred.hasattr_(target, "__next__")
            return None

        if exc_type == "FileNotFoundError":
            if target:
                return Pred.is_not_none(target)
            return None

        if exc_type == "ZeroDivisionError":
            divisor = secondary or target
            if divisor:
                return Pred.var_neq(divisor, 0)
            return None

        if exc_type == "OverflowError":
            return Pred.true_()

        if exc_type == "UnicodeDecodeError":
            if target:
                return Pred.isinstance_(target, "str")
            return None

        return None

    def _map_value_error_predicate(self, op: RaisingOp) -> Optional[Pred]:
        """Map ValueError to a success predicate."""
        if not isinstance(op.node, ast.Call):
            if op.target_var:
                return Pred.is_not_none(op.target_var)
            return None

        call_name = _get_call_name(op.node)
        if call_name in ("int", "float"):
            # int(x) / float(x) didn't raise → x is convertible.
            arg_var = None
            if op.node.args:
                arg_var = _get_name(op.node.args[0])
            if arg_var:
                return Pred.isinstance_(arg_var, "str").and_(
                    Pred.truthy(arg_var)
                )
        if call_name == "index":
            # list.index(x) didn't raise → x is in the list.
            if op.target_var and op.secondary_var:
                return Pred.truthy(op.target_var)
        if call_name == "remove":
            if op.target_var and op.secondary_var:
                return Pred.len_ge(op.target_var, 1)

        if op.target_var:
            return Pred.is_not_none(op.target_var)
        return None

    def _extract_index_from_op(self, op: RaisingOp) -> Optional[int]:
        """Try to extract an integer index from a subscript operation."""
        if isinstance(op.node, ast.Subscript):
            if isinstance(op.node.slice, ast.expr):
                return _get_int_literal(op.node.slice)
        return None

    # ---------------------------------------------------------------
    # Else-branch refinements
    # ---------------------------------------------------------------

    def _compute_else_refinements(
        self,
        try_body: List[ast.stmt],
        handlers: List[ast.ExceptHandler],
        state: AnalysisState,
    ) -> AnalysisState:
        """Compute refinements for the else block.

        The else block runs only when NO exception was raised by the try
        body.  This means every potentially-raising operation succeeded.
        We collect the conjunction of all success predicates.
        """
        else_state = state.copy()
        raising_ops = self._extract_potentially_raising_ops(try_body)

        # Determine which exception types are handled.
        handled_types: Set[str] = set()
        for handler in handlers:
            handled_types.update(_extract_handler_exception_types(handler))

        # For each raising op whose exceptions are handled, the success
        # predicate holds in the else branch.
        for op in raising_ops:
            relevant_exc = op.possible_exceptions & handled_types
            if not relevant_exc:
                # If the exception isn't caught, the else block is only
                # reached if no exception was raised at all — which still
                # means the op succeeded.
                relevant_exc = op.possible_exceptions

            for exc in relevant_exc:
                pred = self._map_exception_to_negated_predicate(op, exc)
                if pred is not None:
                    else_state.add_predicate(pred)
                    # Also refine the specific variable.
                    if op.target_var:
                        else_state.refine_variable(op.target_var, pred)

            # Add the operation's own implied predicates.
            for pred in op.implied_predicates:
                else_state.add_predicate(pred)
                for v in pred.free_vars():
                    else_state.refine_variable(v, pred)

        return else_state

    # ---------------------------------------------------------------
    # Handler analysis
    # ---------------------------------------------------------------

    def _analyze_handler(
        self,
        handler: ast.ExceptHandler,
        raising_ops: List[RaisingOp],
        state: AnalysisState,
    ) -> AnalysisState:
        """Analyze a single except handler.

        In the handler, we know the exception WAS raised.  For specific
        exception types we can infer the cause predicate (the negation
        of the success predicate).
        """
        handler_state = state.copy()
        exc_types = _extract_handler_exception_types(handler)
        handler_state.exception_context.extend(exc_types)

        # If the handler binds the exception (except E as e:), add the
        # exception variable with its type.
        if handler.name:
            exc_base = self._exc_type_to_base(exc_types[0] if exc_types else "Exception")
            handler_state.bindings[handler.name] = RefType.trivial(exc_base)

        # For each raising op, if this handler catches its exception,
        # we know the cause predicate holds (the operation failed).
        for op in raising_ops:
            matched_exc = op.possible_exceptions & set(exc_types)
            if not matched_exc:
                continue

            for exc in matched_exc:
                cause_pred = self._get_cause_predicate(op, exc)
                if cause_pred is not None:
                    handler_state.add_predicate(cause_pred)
                    if op.target_var:
                        handler_state.refine_variable(
                            op.target_var, cause_pred
                        )

        # Analyze the handler body for additional bindings.
        handler_state = self._analyze_handler_body(handler.body, handler_state)

        return handler_state

    def _get_cause_predicate(
        self, op: RaisingOp, exc_type: str
    ) -> Optional[Pred]:
        """Get the predicate that CAUSED the exception.

        This is the negation of the success predicate.
        """
        success = self._map_exception_to_negated_predicate(op, exc_type)
        if success is not None:
            return success.not_()
        return None

    def _exc_type_to_base(self, exc_name: str) -> BaseTypeR:
        """Map an exception type name to a BaseTypeR."""
        return BaseTypeR(BaseTypeKind.OBJECT)

    def _analyze_handler_body(
        self,
        body: List[ast.stmt],
        state: AnalysisState,
    ) -> AnalysisState:
        """Analyze statements in an exception handler body.

        Look for common patterns like re-raising, logging, default
        value assignment, etc.
        """
        for stmt in body:
            if isinstance(stmt, ast.Raise):
                # Re-raise means this path terminates exceptionally.
                # The state is effectively bottom for the merge.
                pass
            elif isinstance(stmt, ast.Assign):
                # Default value assignment: x = default_val
                if (len(stmt.targets) == 1
                        and isinstance(stmt.targets[0], ast.Name)):
                    var_name = stmt.targets[0].id
                    state.refine_variable(var_name, Pred.is_not_none(var_name))
            elif isinstance(stmt, ast.Return):
                # Early return from handler.
                pass
        return state

    def _analyze_else_body(
        self,
        body: List[ast.stmt],
        state: AnalysisState,
    ) -> AnalysisState:
        """Analyze the else block of a try/except."""
        for stmt in body:
            if isinstance(stmt, ast.Assign):
                if (len(stmt.targets) == 1
                        and isinstance(stmt.targets[0], ast.Name)):
                    var_name = stmt.targets[0].id
                    state.refine_variable(
                        var_name, Pred.is_not_none(var_name)
                    )
        return state

    def _analyze_finally(
        self,
        body: List[ast.stmt],
        state: AnalysisState,
    ) -> AnalysisState:
        """Analyze the finally block.

        The finally block always executes regardless of whether an
        exception occurred.  It does not add refinements, but it may
        invalidate them (e.g. closing a file invalidates is-open).
        """
        result = state.copy()
        for stmt in body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call = stmt.value
                if (isinstance(call.func, ast.Attribute)
                        and call.func.attr == "close"):
                    target = _get_name(call.func.value)
                    if target and target in result.bindings:
                        # After close(), the file handle refinements
                        # (is-open, is-readable) are invalidated.
                        result.bindings[target] = RefType.trivial(
                            result.bindings[target].base
                        )
        return result

    # ---------------------------------------------------------------
    # Inference helpers for specific expression kinds
    # ---------------------------------------------------------------

    def _infer_subscript_refinement(
        self, node: ast.Subscript, exc_type: str
    ) -> Optional[Pred]:
        """Infer refinement from a subscript that didn't raise."""
        target = _get_name(node.value)
        if target is None:
            return None

        if exc_type == "KeyError":
            return self._analyze_dict_access(node, target)

        if exc_type == "IndexError":
            idx = None
            if isinstance(node.slice, ast.expr):
                idx = _get_int_literal(node.slice)
            if idx is not None and idx >= 0:
                return Pred.len_gt(target, idx)
            return Pred.len_ge(target, 1)

        if exc_type == "TypeError":
            return Pred.hasattr_(target, "__getitem__")

        return None

    def _infer_attribute_refinement(
        self, node: ast.Attribute, exc_type: str
    ) -> Optional[Pred]:
        """Infer refinement from an attribute access that didn't raise."""
        target = _get_name(node.value)
        if target is None:
            return None

        if exc_type == "AttributeError":
            return self._analyze_attribute_access(node, target)

        return None

    def _infer_call_refinement(
        self, node: ast.Call, exc_type: str
    ) -> Optional[Pred]:
        """Infer refinement from a call that didn't raise."""
        call_name = _get_call_name(node)
        if call_name is None:
            return None

        if call_name in ("int", "float") and exc_type == "ValueError":
            return self._analyze_int_conversion(node, call_name)

        if call_name == "next" and exc_type == "StopIteration":
            return self._analyze_iteration(node)

        if call_name == "open" and exc_type in (
            "FileNotFoundError", "OSError", "IOError"
        ):
            return self._analyze_file_operation(node, call_name)

        if call_name == "getattr" and exc_type == "AttributeError":
            if len(node.args) >= 2:
                obj_var = _get_name(node.args[0])
                attr_lit = _get_string_literal(node.args[1])
                if obj_var and attr_lit:
                    return Pred.hasattr_(obj_var, attr_lit)
            return None

        if exc_type == "TypeError":
            if node.args:
                arg_var = _get_name(node.args[0])
                if arg_var:
                    return Pred.is_not_none(arg_var)
            return None

        return None

    def _infer_binop_refinement(
        self, node: ast.BinOp, exc_type: str
    ) -> Optional[Pred]:
        """Infer refinement from a binary op that didn't raise."""
        if exc_type == "ZeroDivisionError":
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                right_var = _get_name(node.right)
                if right_var:
                    return Pred.var_neq(right_var, 0)
        if exc_type == "OverflowError":
            return Pred.true_()
        if exc_type == "TypeError":
            left_var = _get_name(node.left)
            right_var = _get_name(node.right)
            preds: List[Pred] = []
            if left_var:
                preds.append(Pred.is_not_none(left_var))
            if right_var:
                preds.append(Pred.is_not_none(right_var))
            if preds:
                result = preds[0]
                for p in preds[1:]:
                    result = result.and_(p)
                return result
        return None

    def _infer_unpack_refinement(
        self, node: ast.Starred, exc_type: str
    ) -> Optional[Pred]:
        """Infer refinement from unpacking that didn't raise."""
        target = _get_name(node.value)
        if target and exc_type == "ValueError":
            return Pred.len_ge(target, 1)
        return None

    # ---------------------------------------------------------------
    # Pattern-specific analysis helpers
    # ---------------------------------------------------------------

    def _analyze_dict_access(
        self, node: ast.Subscript, target: str
    ) -> Optional[Pred]:
        """d[key] in try → key in d in else.

        When ``d[key]`` does not raise ``KeyError``, we know:
        1. ``d`` is a mapping (has ``__getitem__``).
        2. ``key`` is present in ``d``.

        We model (2) via ``hasattr`` on the key as a proxy for
        key-membership, since our predicate language does not have
        a direct ``in`` operator for dict keys.
        """
        key_var = _get_name(node.slice) if isinstance(node.slice, ast.expr) else None
        key_str = None
        if isinstance(node.slice, ast.expr):
            key_str = _get_string_literal(node.slice)

        # If key is a variable, produce hasattr(target, key_var_name)
        # as a proxy for "key in target".
        if key_var:
            return Pred.hasattr_(target, key_var)
        if key_str:
            return Pred.hasattr_(target, key_str)

        # If key is an integer literal, this is likely a list, not dict.
        key_int = None
        if isinstance(node.slice, ast.expr):
            key_int = _get_int_literal(node.slice)
        if key_int is not None and key_int >= 0:
            return Pred.len_gt(target, key_int)

        return Pred.is_not_none(target)

    def _analyze_int_conversion(
        self, node: ast.Call, call_name: str
    ) -> Optional[Pred]:
        """int(x) / float(x) in try → x is numeric string in else.

        When ``int(x)`` does not raise ``ValueError``, we know:
        1. ``x`` is a string (or already numeric).
        2. ``x`` represents a valid integer literal.

        We model this as ``isinstance(x, str) ∧ truthy(x)`` meaning
        ``x`` is a non-empty string that is convertible.
        """
        if not node.args:
            return None
        arg = node.args[0]
        arg_var = _get_name(arg)
        if arg_var is None:
            return None

        # int(x) succeeding tells us x is numeric.
        # We approximate this as: x is a truthy string.
        is_str = Pred.isinstance_(arg_var, "str")
        is_truthy = Pred.truthy(arg_var)

        if call_name == "int":
            return is_str.and_(is_truthy)
        elif call_name == "float":
            return is_str.and_(is_truthy)

        return Pred.truthy(arg_var)

    def _analyze_iteration(self, node: ast.Call) -> Optional[Pred]:
        """next(it) in try → iterator not exhausted in else.

        When ``next(it)`` does not raise ``StopIteration``, we know:
        1. ``it`` is an iterator (has ``__next__``).
        2. The iterator was not exhausted — it had at least one more
           element.

        We model (1) via ``hasattr(it, '__next__')`` and approximate (2)
        with ``truthy(it)`` since a non-exhausted iterator is truthy.
        """
        if not node.args:
            return None

        it_var = _get_name(node.args[0])
        if it_var is None:
            return None

        # If next() has a default argument, it never raises StopIteration.
        if len(node.args) >= 2 or node.keywords:
            return Pred.true_()

        has_next = Pred.hasattr_(it_var, "__next__")
        not_exhausted = Pred.truthy(it_var)
        return has_next.and_(not_exhausted)

    def _analyze_attribute_access(
        self, node: ast.Attribute, target: str
    ) -> Optional[Pred]:
        """obj.attr in try → hasattr(obj, attr) in else.

        When ``obj.attr`` does not raise ``AttributeError``, we know:
        1. ``obj`` is not None (has attributes).
        2. ``obj`` has the attribute ``attr``.

        We model both via ``hasattr(obj, attr)`` which implies
        ``is_not_none(obj)``.
        """
        return Pred.hasattr_(target, node.attr).and_(
            Pred.is_not_none(target)
        )

    def _analyze_file_operation(
        self, node: ast.Call, call_name: str
    ) -> Optional[Pred]:
        """f.read() in try → file is open/readable in else.

        When file operations don't raise IOError/OSError, we know:
        1. The file handle is valid and open.
        2. The operation-specific precondition holds (readable for
           read, writable for write, etc.).

        For ``open(path)``, not raising means the path exists and is
        accessible.
        """
        if call_name == "open":
            if node.args:
                path_var = _get_name(node.args[0])
                if path_var:
                    return Pred.is_not_none(path_var).and_(
                        Pred.truthy(path_var)
                    )
            return None

        # Method call: f.read(), f.write(), etc.
        if isinstance(node.func, ast.Attribute):
            file_var = _get_name(node.func.value)
            if file_var:
                method = node.func.attr
                # File is open and supports the operation.
                is_open = Pred.hasattr_(file_var, "read")
                is_not_none = Pred.is_not_none(file_var)

                if method in ("read", "readline", "readlines"):
                    return is_not_none.and_(is_open)
                elif method in ("write", "writelines"):
                    is_writable = Pred.hasattr_(file_var, "write")
                    return is_not_none.and_(is_writable)
                elif method == "seek":
                    return is_not_none.and_(is_open)
                else:
                    return is_not_none

        return None

    # ---------------------------------------------------------------
    # Guard predicate extraction (for LBYL comparison)
    # ---------------------------------------------------------------

    def _extract_guard_predicates(self, guard: ast.expr) -> List[Pred]:
        """Extract refinement predicates from a guard expression.

        Handles common LBYL patterns:
        - ``key in d`` → hasattr(d, key)
        - ``hasattr(obj, 'attr')`` → hasattr(obj, attr)
        - ``isinstance(x, T)`` → isinstance(x, T)
        - ``x is not None`` → is_not_none(x)
        - ``len(lst) > n`` → len_gt(lst, n)
        - ``x`` (truthy check) → truthy(x)
        """
        preds: List[Pred] = []
        self._extract_guard_preds_recursive(guard, preds, negated=False)
        return preds

    def _extract_guard_preds_recursive(
        self,
        node: ast.expr,
        preds: List[Pred],
        negated: bool,
    ) -> None:
        """Recursively extract predicates from a guard expression."""
        # Boolean operators: and, or
        if isinstance(node, ast.BoolOp):
            for value in node.values:
                self._extract_guard_preds_recursive(value, preds, negated)
            return

        # Unary not
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            self._extract_guard_preds_recursive(
                node.operand, preds, not negated
            )
            return

        # Comparison: key in d, x is not None, len(lst) > n
        if isinstance(node, ast.Compare):
            pred = self._compare_to_pred(node, negated)
            if pred is not None:
                preds.append(pred)
            return

        # Call: hasattr(obj, 'attr'), isinstance(x, T), callable(x)
        if isinstance(node, ast.Call):
            pred = self._call_to_pred(node, negated)
            if pred is not None:
                preds.append(pred)
            return

        # Simple name as truthy check
        if isinstance(node, ast.Name):
            p = Pred.truthy(node.id)
            preds.append(p.not_() if negated else p)
            return

    def _compare_to_pred(
        self, node: ast.Compare, negated: bool
    ) -> Optional[Pred]:
        """Convert a comparison to a predicate."""
        if len(node.ops) != 1 or len(node.comparators) != 1:
            return None

        op = node.ops[0]
        left = node.left
        right = node.comparators[0]

        # key in d
        if isinstance(op, ast.In):
            key_var = _get_name(left)
            container = _get_name(right)
            if key_var and container:
                p = Pred.hasattr_(container, key_var)
                return p.not_() if negated else p

        # key not in d
        if isinstance(op, ast.NotIn):
            key_var = _get_name(left)
            container = _get_name(right)
            if key_var and container:
                p = Pred.hasattr_(container, key_var)
                return p if negated else p.not_()

        # x is None / x is not None
        if isinstance(op, ast.Is):
            if isinstance(right, ast.Constant) and right.value is None:
                var = _get_name(left)
                if var:
                    p = Pred.is_none(var)
                    return p.not_() if negated else p

        if isinstance(op, ast.IsNot):
            if isinstance(right, ast.Constant) and right.value is None:
                var = _get_name(left)
                if var:
                    p = Pred.is_not_none(var)
                    return p.not_() if negated else p

        # Numeric comparisons: x < n, len(lst) > n, etc.
        if isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)):
            return self._numeric_compare_to_pred(left, op, right, negated)

        return None

    def _numeric_compare_to_pred(
        self,
        left: ast.expr,
        op: ast.cmpop,
        right: ast.expr,
        negated: bool,
    ) -> Optional[Pred]:
        """Convert a numeric comparison to a Pred."""
        # len(x) > n / len(x) >= n / etc.
        if isinstance(left, ast.Call) and _is_call_to(left, "len"):
            if left.args:
                var = _get_name(left.args[0])
                val = _get_int_literal(right)
                if var is not None and val is not None:
                    p = self._len_compare_pred(var, op, val)
                    if p is not None:
                        return p.not_() if negated else p

        # x > n, x >= n, etc.
        left_var = _get_name(left)
        right_val = _get_int_literal(right)
        if left_var is not None and right_val is not None:
            p = self._var_compare_pred(left_var, op, right_val)
            if p is not None:
                return p.not_() if negated else p

        # n < x  →  x > n
        left_val = _get_int_literal(left)
        right_var = _get_name(right)
        if left_val is not None and right_var is not None:
            flipped = self._flip_cmpop(op)
            if flipped is not None:
                p = self._var_compare_pred(right_var, flipped, left_val)
                if p is not None:
                    return p.not_() if negated else p

        return None

    def _len_compare_pred(
        self, var: str, op: ast.cmpop, val: int
    ) -> Optional[Pred]:
        """Build a length comparison predicate."""
        if isinstance(op, ast.Gt):
            return Pred.len_gt(var, val)
        if isinstance(op, ast.GtE):
            return Pred.len_ge(var, val)
        if isinstance(op, ast.Eq):
            return Pred.len_eq(var, val)
        if isinstance(op, ast.Lt):
            # len(x) < n  ↔  ¬(len(x) >= n)
            return Pred.len_ge(var, val).not_()
        if isinstance(op, ast.LtE):
            # len(x) <= n  ↔  ¬(len(x) > n)
            return Pred.len_gt(var, val).not_()
        if isinstance(op, ast.NotEq):
            return Pred.len_eq(var, val).not_()
        return None

    def _var_compare_pred(
        self, var: str, op: ast.cmpop, val: int
    ) -> Optional[Pred]:
        """Build a variable comparison predicate."""
        if isinstance(op, ast.Gt):
            return Pred.var_gt(var, val)
        if isinstance(op, ast.GtE):
            return Pred.var_ge(var, val)
        if isinstance(op, ast.Lt):
            return Pred.var_lt(var, val)
        if isinstance(op, ast.LtE):
            return Pred.var_le(var, val)
        if isinstance(op, ast.Eq):
            return Pred.var_eq(var, val)
        if isinstance(op, ast.NotEq):
            return Pred.var_neq(var, val)
        return None

    @staticmethod
    def _flip_cmpop(op: ast.cmpop) -> Optional[ast.cmpop]:
        """Flip a comparison operator (for n < x → x > n)."""
        flips = {
            ast.Lt: ast.Gt,
            ast.LtE: ast.GtE,
            ast.Gt: ast.Lt,
            ast.GtE: ast.LtE,
            ast.Eq: ast.Eq,
            ast.NotEq: ast.NotEq,
        }
        return flips.get(type(op), lambda: None)()  # type: ignore[arg-type]

    def _call_to_pred(
        self, node: ast.Call, negated: bool
    ) -> Optional[Pred]:
        """Convert a guard call (hasattr, isinstance, callable) to a Pred."""
        call_name = _get_call_name(node)
        if call_name is None:
            return None

        if call_name == "hasattr" and len(node.args) >= 2:
            obj_var = _get_name(node.args[0])
            attr_lit = _get_string_literal(node.args[1])
            if obj_var and attr_lit:
                p = Pred.hasattr_(obj_var, attr_lit)
                return p.not_() if negated else p

        if call_name == "isinstance" and len(node.args) >= 2:
            obj_var = _get_name(node.args[0])
            type_name = _get_name(node.args[1])
            if obj_var and type_name:
                p = Pred.isinstance_(obj_var, type_name)
                return p.not_() if negated else p

        if call_name == "callable" and len(node.args) >= 1:
            obj_var = _get_name(node.args[0])
            if obj_var:
                p = Pred.hasattr_(obj_var, "__call__")
                return p.not_() if negated else p

        if call_name == "len" and len(node.args) >= 1:
            # Bare len(x) as truthy guard — means len(x) > 0.
            obj_var = _get_name(node.args[0])
            if obj_var:
                p = Pred.len_gt(obj_var, 0)
                return p.not_() if negated else p

        return None

    # ---------------------------------------------------------------
    # Bulk pattern analysis
    # ---------------------------------------------------------------

    def analyze_eafp_patterns_in_module(
        self, tree: ast.Module
    ) -> List[Tuple[ast.Try, List[Pred]]]:
        """Find all EAFP patterns in a module and their refinements.

        Walks the entire module AST, finds try/except blocks, and
        computes the success-path refinements for each.
        """
        results: List[Tuple[ast.Try, List[Pred]]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    preds = self.model_eafp_pattern(node.body, handler)
                    if preds:
                        results.append((node, preds))
        return results

    def find_lbyl_eafp_pairs(
        self, tree: ast.Module
    ) -> List[Tuple[ast.If, ast.Try, ComparisonResult]]:
        """Find pairs of LBYL and EAFP patterns in a module.

        Heuristically pairs ``if`` guards with nearby ``try/except``
        blocks that operate on the same variables.
        """
        ifs: List[ast.If] = []
        trys: List[ast.Try] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                ifs.append(node)
            elif isinstance(node, ast.Try):
                trys.append(node)

        pairs: List[Tuple[ast.If, ast.Try, ComparisonResult]] = []
        for if_node in ifs:
            guard_vars = self._expr_vars(if_node.test)
            if not guard_vars:
                continue
            for try_node in trys:
                try_vars = set()
                for op in self._extract_potentially_raising_ops(try_node.body):
                    if op.target_var:
                        try_vars.add(op.target_var)
                    if op.secondary_var:
                        try_vars.add(op.secondary_var)

                shared = guard_vars & try_vars
                if shared:
                    comparison = self.model_lbyl_vs_eafp(
                        if_node.test, if_node.body
                    )
                    pairs.append((if_node, try_node, comparison))

        return pairs

    def _expr_vars(self, node: ast.expr) -> Set[str]:
        """Collect all variable names referenced in an expression."""
        result: Set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                result.add(child.id)
        return result

    # ---------------------------------------------------------------
    # Summary and pretty-printing
    # ---------------------------------------------------------------

    def summarize_refinements(
        self, state: AnalysisState
    ) -> Dict[str, str]:
        """Produce a human-readable summary of refined variable types."""
        summary: Dict[str, str] = {}
        for var, ref_type in state.bindings.items():
            summary[var] = ref_type.pretty()
        return summary

    def explain_try_except(self, node: ast.Try) -> List[str]:
        """Produce a human-readable explanation of refinements.

        Returns a list of strings, one per raising operation, explaining
        what refinement the success of that operation implies.
        """
        lines: List[str] = []
        raising_ops = self._extract_potentially_raising_ops(node.body)

        if not raising_ops:
            lines.append("No potentially-raising operations in try body.")
            return lines

        for op in raising_ops:
            desc = f"  {op.kind.name}"
            if op.target_var:
                desc += f" on '{op.target_var}'"
            desc += f" may raise: {', '.join(sorted(op.possible_exceptions))}"
            lines.append(desc)

            for exc in sorted(op.possible_exceptions):
                neg = self._map_exception_to_negated_predicate(op, exc)
                if neg is not None:
                    lines.append(
                        f"    If no {exc}: {neg}"
                    )

        return lines
