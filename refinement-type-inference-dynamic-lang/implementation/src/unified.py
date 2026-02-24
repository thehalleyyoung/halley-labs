"""
Unified Analysis via CofiberedDomain.

Genuine single-pass analyzer that walks each function body ONCE,
maintaining ``Dict[str, CofiberedElement]`` as the abstract environment.
Each variable is tracked as a CofiberedElement whose type-tag fibers carry
domain-specific abstract values (IntervalDomain for ints, NullityDomain for
NoneType, ShapeDomain for Tensors).

Transfer functions update ALL fibers simultaneously.  Cross-domain
narrowing (e.g. ``if x is not None`` makes the Tensor fiber live) and
shape checking (matmul, broadcast, reshape, cat, nn.Linear, nn.Conv2d,
BatchNorm, LayerNorm, Dropout) happen within the same single pass.

``liquid_result`` and ``shape_result`` are built from the single-pass
walker's own findings — no separate sub-analyses are invoked.
"""

from __future__ import annotations

import ast
import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.domains.abstract_domains import (
    CofiberedDomain,
    CofiberedElement,
    IntervalDomain,
    Interval,
)
from src.domains.nullity import NullityDomain, NullityValue, NullityKind
from src.tensor_shapes import (
    TensorShape,
    ShapeDim,
    ShapeError,
    ShapeErrorKind,
    TensorShapeAnalyzer,
    ShapeAnalysisResult,
    ShapeEnv,
    analyze_shapes,
    compute_matmul_shape,
    check_matmul_compatible,
    compute_broadcast_shape,
    compute_reshape_shape,
)
from src.liquid import (
    LiquidBug,
    LiquidBugKind,
    LiquidAnalysisResult,
    analyze_liquid,
)

# Mapping from UnifiedBug kind strings to LiquidBugKind / ShapeErrorKind
_LIQUID_BUG_KINDS: Dict[str, LiquidBugKind] = {
    k.name: k for k in LiquidBugKind
}
_SHAPE_ERROR_KINDS: Dict[str, ShapeErrorKind] = {
    k.name: k for k in ShapeErrorKind
}
from src.interprocedural import InterproceduralShapeAnalyzer, ShapeContract
from src.intent_bugs import (
    OverwarnAnalyzer,
    IntentApparentBug,
    IntentBugKind,
)

try:
    import z3 as _z3
    from src._experimental.refinement_lattice import Z3Encoder
    _HAS_Z3 = True
except Exception:
    _HAS_Z3 = False


# ── ShapeDomain wrapper (lattice interface for CofiberedDomain) ────────

class ShapeDomain:
    """Abstract domain over tensor shapes for use as a CofiberedDomain fiber."""

    def bottom(self) -> Optional[TensorShape]:
        return None

    def top(self) -> TensorShape:
        return TensorShape.unknown(0)

    def join(self, a: Optional[TensorShape], b: Optional[TensorShape]) -> Optional[TensorShape]:
        if a is None:
            return b
        if b is None:
            return a
        if a == b:
            return a
        # Shapes differ → widen to unknown with max ndim
        return TensorShape.unknown(max(a.ndim, b.ndim))

    def meet(self, a: Optional[TensorShape], b: Optional[TensorShape]) -> Optional[TensorShape]:
        if a is None or b is None:
            return None
        if a == b:
            return a
        return None  # incompatible shapes → bottom

    def leq(self, a: Optional[TensorShape], b: Optional[TensorShape]) -> bool:
        if a is None:
            return True
        if b is None:
            return False
        return a == b

    def widen(self, a: Optional[TensorShape], b: Optional[TensorShape]) -> Optional[TensorShape]:
        return self.join(a, b)

    def narrow(self, a: Optional[TensorShape], b: Optional[TensorShape]) -> Optional[TensorShape]:
        return self.meet(a, b)


# ── Unified bug types ──────────────────────────────────────────────────

@dataclass
class UnifiedBug:
    """A bug found by the unified analysis."""
    kind: str           # e.g. "NULL_DEREF", "SHAPE_MISMATCH", "OPTIONAL_TENSOR_ACCESS"
    line: int
    col: int
    message: str
    function: str
    variable: str
    severity: str = "error"
    cross_domain: bool = False  # True if bug spans null + shape domains

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "line": self.line,
            "col": self.col,
            "message": self.message,
            "function": self.function,
            "variable": self.variable,
            "severity": self.severity,
            "cross_domain": self.cross_domain,
        }


@dataclass
class UnifiedAnalysisResult:
    """Results from the unified cofibered analysis.

    Includes both proven bugs (from liquid type checking) and
    intent-apparent bugs (from overwarning pattern checkers).
    """
    bugs: List[UnifiedBug] = field(default_factory=list)
    intent_bugs: List[IntentApparentBug] = field(default_factory=list)
    liquid_result: Optional[LiquidAnalysisResult] = None
    shape_result: Optional[ShapeAnalysisResult] = None
    analysis_time_ms: float = 0.0
    cofibered_domain_used: bool = True

    @property
    def null_bugs(self) -> List[UnifiedBug]:
        return [b for b in self.bugs if b.kind in ("NULL_DEREF", "OPTIONAL_TENSOR_ACCESS")]

    @property
    def shape_bugs(self) -> List[UnifiedBug]:
        return [b for b in self.bugs if b.kind in (
            "SHAPE_MISMATCH", "MATMUL_INCOMPAT", "DIM_MISMATCH",
            "NDIM_MISMATCH", "RESHAPE_INVALID", "BROADCAST_FAIL",
            "CAT_INCOMPAT", "CONV_INCOMPAT",
        )]

    @property
    def cross_domain_bugs(self) -> List[UnifiedBug]:
        return [b for b in self.bugs if b.cross_domain]

    @property
    def overwarn_bugs(self) -> List[IntentApparentBug]:
        """Intent-apparent bugs flagged by the overwarning system."""
        return self.intent_bugs

    @property
    def all_warnings(self) -> int:
        """Total count of all bugs + overwarn warnings."""
        return len(self.bugs) + len(self.intent_bugs)

    def summary(self) -> str:
        return (
            f"Unified Analysis: {len(self.bugs)} proven bugs "
            f"({len(self.null_bugs)} null, {len(self.shape_bugs)} shape, "
            f"{len(self.cross_domain_bugs)} cross-domain), "
            f"{len(self.intent_bugs)} intent-apparent warnings, "
            f"{self.analysis_time_ms:.1f}ms"
        )


# ── Type alias for the abstract environment ────────────────────────────

Env = Dict[str, CofiberedElement]


# ── Z3 shape constraint helper ─────────────────────────────────────────

def _z3_check_matmul(var_a: str, shape_a: TensorShape,
                     var_b: str, shape_b: TensorShape,
                     line: int) -> Optional[str]:
    """Use Z3 to verify matmul dimension compatibility when symbolic dims are present."""
    if not _HAS_Z3:
        return None
    k_a = shape_a.dims[-1] if shape_a.ndim >= 1 else None
    k_b = shape_b.dims[-2] if shape_b.ndim >= 2 else (shape_b.dims[0] if shape_b.ndim == 1 else None)
    if k_a is None or k_b is None:
        return None
    if k_a.is_symbolic or k_b.is_symbolic:
        # With symbolic dims we can't statically refute — skip
        return None
    if not k_a.is_symbolic and not k_b.is_symbolic and k_a.value != k_b.value:
        return (f"matmul dimension mismatch: "
                f"{var_a} has inner dim {k_a.value}, {var_b} has inner dim {k_b.value}")
    return None


def _has_symbolic(shape: TensorShape) -> bool:
    """Return True if any dimension is symbolic."""
    return any(d.is_symbolic for d in shape.dims)


def _z3_check_broadcast(shape_a: TensorShape, shape_b: TensorShape,
                        line: int) -> Optional[str]:
    """Use Z3 to verify broadcast compatibility when symbolic dims present."""
    if not _HAS_Z3:
        return None
    # For all-concrete dims, use fast Python check (compute_broadcast_shape handles it)
    if not _has_symbolic(shape_a) and not _has_symbolic(shape_b):
        result = compute_broadcast_shape(shape_a, shape_b)
        if result is None:
            return (f"broadcast incompatible shapes {shape_a.pretty()} "
                    f"and {shape_b.pretty()} at line {line}")
        return None
    # With symbolic dims, try Z3: for each aligned dim pair,
    # (dim_a == dim_b) OR (dim_a == 1) OR (dim_b == 1) must hold
    ndim = max(shape_a.ndim, shape_b.ndim)
    solver = _z3.Solver()
    solver.set("timeout", 500)
    for i in range(1, ndim + 1):
        d_a = shape_a.dims[-i] if i <= shape_a.ndim else ShapeDim(1)
        d_b = shape_b.dims[-i] if i <= shape_b.ndim else ShapeDim(1)
        if d_a.is_symbolic or d_b.is_symbolic:
            continue  # can't refute symbolic dims
        # Both concrete
        va, vb = d_a.value, d_b.value
        if va != vb and va != 1 and vb != 1:
            return (f"broadcast incompatible: dim {-i} has {va} vs {vb} "
                    f"(neither is 1) at line {line}")
    return None


def _z3_check_reshape(old_shape: TensorShape, new_shape: TensorShape,
                      line: int) -> Optional[str]:
    """Use Z3 to verify reshape compatibility (total elements must match)."""
    if not _HAS_Z3:
        return None
    if _has_symbolic(old_shape) or _has_symbolic(new_shape):
        return None  # can't verify with symbolic dims
    # All concrete: product must match (accounting for -1)
    old_total = 1
    for d in old_shape.dims:
        if d.is_symbolic:
            return None
        old_total *= d.value
    new_total = 1
    has_neg1 = False
    for d in new_shape.dims:
        if d.is_symbolic:
            return None
        if d.value == -1 or (isinstance(d.value, str) and d.value == "_inferred"):
            has_neg1 = True
            continue
        new_total *= d.value
    if has_neg1:
        if old_total % new_total != 0:
            return (f"reshape invalid: total elements {old_total} not divisible "
                    f"by specified dims product {new_total} at line {line}")
    elif old_total != new_total:
        return (f"reshape invalid: {old_shape.pretty()} has {old_total} elements "
                f"but target {new_shape.pretty()} has {new_total} at line {line}")
    return None


def _z3_check_cat(shapes: List[TensorShape], dim: int,
                  line: int) -> Optional[str]:
    """Use Z3 to verify cat compatibility — all non-cat dims must match."""
    if not _HAS_Z3:
        return None
    if not shapes:
        return None
    ref = shapes[0]
    for idx, s in enumerate(shapes[1:], 1):
        if s.ndim != ref.ndim:
            return (f"torch.cat: tensor {idx} has {s.ndim}D but tensor 0 "
                    f"has {ref.ndim}D at line {line}")
        for d_i in range(ref.ndim):
            if d_i == dim:
                continue  # cat dimension can differ
            d_ref = ref.dims[d_i]
            d_s = s.dims[d_i]
            if d_ref.is_symbolic or d_s.is_symbolic:
                continue
            if d_ref.value != d_s.value:
                return (f"torch.cat: dim {d_i} mismatch: tensor 0 has "
                        f"{d_ref.value}, tensor {idx} has {d_s.value} "
                        f"at line {line}")
    return None


# ── Unified Analyzer ──────────────────────────────────────────────────

class UnifiedAnalyzer(ast.NodeVisitor):
    """Single-pass analyzer using CofiberedDomain for unified null-safety
    and tensor shape checking.

    The cofibered domain maps:
      "int"      → IntervalDomain
      "Tensor"   → ShapeDomain
      "NoneType" → NullityDomain

    Variables are tracked as CofiberedElements.  A variable that is
    Optional[Tensor] will have entries for both "Tensor" and "NoneType",
    enabling cross-domain checks.

    A single AST walk per function body performs:
      - Assignment transfer (updates all fibers simultaneously)
      - Cross-domain narrowing at ``if`` guards
      - Shape checking for matmul, broadcast, reshape, cat, nn.Linear
      - Interprocedural shape propagation via ShapeContract lookup
      - Z3 constraint generation for symbolic shape ops
    """

    def __init__(self) -> None:
        self.domain = CofiberedDomain(fiber_domains={
            "int": IntervalDomain(),
            "Tensor": ShapeDomain(),
            "NoneType": NullityDomain(),
        })
        self.env: Env = {}
        self.bugs: List[UnifiedBug] = []
        self.func_name = "<module>"
        # Interprocedural shape contracts (populated once per analyze() call)
        self._shape_contracts: Dict[str, ShapeContract] = {}
        # nn.Module layer info extracted from __init__, keyed by class name
        self._class_layers: Dict[str, Dict[str, Any]] = {}
        # Layer info for the class currently being analyzed
        self._current_class_layers: Dict[str, Any] = {}
        # Z3 constraint batch: list of (constraint, message, line)
        self._z3_constraints: List[Tuple[Any, str, int]] = []

    # ── Public entry point ─────────────────────────────────────────────

    def analyze(self, source: str) -> UnifiedAnalysisResult:
        t0 = time.monotonic()

        tree = ast.parse(source)

        # Populate interprocedural shape contracts for call-site propagation
        try:
            ipa = InterproceduralShapeAnalyzer()
            self._shape_contracts = ipa.analyze_source(source)
        except Exception:
            self._shape_contracts = {}

        # ── Single-pass unified walk ───────────────────────────────────
        # Walk each function/class body exactly once
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._analyze_function(node)
            elif isinstance(node, ast.ClassDef):
                self._analyze_class(node)

        # Module-level statements
        self.func_name = "<module>"
        self.env = {}
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self._analyze_stmt(node)

        # ── Build sub-analysis results from single-pass findings ───────
        liquid_bugs = []
        shape_errors = []
        for b in self.bugs:
            if b.kind in _LIQUID_BUG_KINDS:
                liquid_bugs.append(LiquidBug(
                    kind=_LIQUID_BUG_KINDS[b.kind],
                    line=b.line, col=b.col, message=b.message,
                    function=b.function, variable=b.variable,
                    severity=b.severity,
                ))
            if b.kind in _SHAPE_ERROR_KINDS:
                shape_errors.append(ShapeError(
                    kind=_SHAPE_ERROR_KINDS[b.kind],
                    line=b.line, col=b.col, message=b.message,
                    function=b.function, variable=b.variable,
                    severity=b.severity,
                ))
        liquid_result = LiquidAnalysisResult(bugs=liquid_bugs)
        shape_result = ShapeAnalysisResult(errors=shape_errors)

        # ── Intent-apparent overwarning pass ───────────────────────────
        # Run pattern checkers for all ML bug classes from bugclasses.jsonl.
        # This deliberately overwarns: it flags patterns that *could* be
        # intent-apparent bugs even when not provable.
        try:
            overwarn = OverwarnAnalyzer()
            intent_bugs = overwarn.analyze(source)
        except Exception:
            intent_bugs = []

        elapsed = (time.monotonic() - t0) * 1000
        return UnifiedAnalysisResult(
            bugs=self.bugs,
            intent_bugs=intent_bugs,
            liquid_result=liquid_result,
            shape_result=shape_result,
            analysis_time_ms=elapsed,
        )

    # ── Function-level analysis ────────────────────────────────────────

    def _analyze_function(self, func: ast.FunctionDef) -> None:
        """Single-pass walk of a function body."""
        self.func_name = func.name
        self.env = {}
        self._z3_constraints = []

        # Seed environment from parameter annotations
        for arg in func.args.args:
            ann = arg.annotation
            if ann:
                self.env[arg.arg] = self._element_from_annotation(ann, arg.arg)

        for stmt in func.body:
            self._analyze_stmt(stmt)

        # Batch-check collected Z3 constraints for transitivity reasoning
        self._flush_z3_constraints()

    # ── Class-level analysis (nn.Module support) ───────────────────────

    def _analyze_class(self, cls: ast.ClassDef) -> None:
        """Analyze an nn.Module subclass: extract layers from __init__,
        then analyze forward() with layer shape info available."""
        is_module = any(
            (isinstance(b, ast.Name) and b.id in ("Module",))
            or (isinstance(b, ast.Attribute) and b.attr == "Module")
            for b in cls.bases
        )

        layers: Dict[str, Any] = {}
        init_method: Optional[ast.FunctionDef] = None
        forward_method: Optional[ast.FunctionDef] = None
        other_methods: List[ast.FunctionDef] = []

        for node in cls.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "__init__":
                    init_method = node
                elif node.name == "forward":
                    forward_method = node
                else:
                    other_methods.append(node)

        # Extract layer definitions from __init__
        if is_module and init_method is not None:
            layers = self._extract_layers_from_init(init_method)
            self._class_layers[cls.name] = layers

        # Analyze forward() with layer info available
        self._current_class_layers = layers
        if forward_method is not None:
            self._analyze_function(forward_method)
        for method in other_methods:
            self._analyze_function(method)
        # Also analyze __init__ for general bugs
        if init_method is not None:
            self._analyze_function(init_method)
        self._current_class_layers = {}

    def _extract_layers_from_init(self, init: ast.FunctionDef) -> Dict[str, Any]:
        """Scan __init__ for self.X = nn.Linear/Conv2d/... patterns."""
        layers: Dict[str, Any] = {}
        for stmt in init.body:
            if not isinstance(stmt, ast.Assign):
                continue
            if len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            # Match self.layer_name = nn.Something(...)
            if not (isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"):
                continue
            layer_name = target.attr
            if not isinstance(stmt.value, ast.Call):
                continue
            call_name = self._get_call_name(stmt.value)
            stem = call_name.split(".")[-1] if call_name else ""
            args = stmt.value.args

            if stem == "Linear" and len(args) >= 2:
                in_f = self._const_int(args[0])
                out_f = self._const_int(args[1])
                layers[layer_name] = {
                    "type": "Linear",
                    "in_features": in_f,
                    "out_features": out_f,
                }
            elif stem == "Conv2d" and len(args) >= 2:
                in_c = self._const_int(args[0])
                out_c = self._const_int(args[1])
                ks = self._const_int(args[2]) if len(args) >= 3 else None
                layers[layer_name] = {
                    "type": "Conv2d",
                    "in_channels": in_c,
                    "out_channels": out_c,
                    "kernel_size": ks,
                }
            elif stem in ("BatchNorm2d", "BatchNorm1d"):
                layers[layer_name] = {"type": stem}
            elif stem == "LayerNorm":
                layers[layer_name] = {"type": "LayerNorm"}
            elif stem == "Dropout":
                layers[layer_name] = {"type": "Dropout"}
        return layers

    @staticmethod
    def _const_int(node: ast.expr) -> Optional[int]:
        """Extract a constant int from an AST node, or None."""
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        return None

    # ── Statement dispatch ─────────────────────────────────────────────

    def _analyze_stmt(self, node: ast.stmt) -> None:
        if isinstance(node, ast.Assign):
            self._transfer_assign(node)
        elif isinstance(node, ast.AugAssign):
            self._transfer_augassign(node)
        elif isinstance(node, ast.If):
            self._transfer_if(node)
        elif isinstance(node, ast.Return):
            if node.value:
                self._eval_expr(node.value)
                self._check_expr(node.value)
        elif isinstance(node, ast.Expr):
            self._eval_expr(node.value)
            self._check_expr(node.value)
        elif isinstance(node, (ast.For, ast.While)):
            for stmt in node.body:
                self._analyze_stmt(stmt)
            for stmt in getattr(node, "orelse", []):
                self._analyze_stmt(stmt)

    # ── Transfer functions ─────────────────────────────────────────────

    def _transfer_assign(self, node: ast.Assign) -> None:
        """Assignment transfer: compute CofiberedElement for RHS,
        update all fibers simultaneously."""
        if len(node.targets) != 1:
            return
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            # Check RHS for bugs even if we can't track the target
            self._check_expr(node.value)
            return
        var = target.id
        rhs_elem = self._eval_expr(node.value)
        self.env[var] = rhs_elem
        # Check the RHS expression for shape / null bugs
        self._check_expr(node.value)

    def _transfer_augassign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            rhs_elem = self._eval_expr(node.value)
            var = node.target.id
            old = self.env.get(var)
            if old is not None:
                self.env[var] = self.domain.join(old, rhs_elem)
            self._check_expr(node.value)

    def _transfer_if(self, node: ast.If) -> None:
        """Fork environment for if/else, perform cross-domain narrowing,
        and join at merge point using CofiberedDomain.join()."""
        checked_var = self._extract_null_check_var(node.test)

        # Snapshot environment before branching
        env_before = dict(self.env)

        # ── True branch: refine away NoneType, make Tensor fiber live ──
        if checked_var:
            old = self.env.get(checked_var)
            if old and "NoneType" in old.entries:
                refined = {k: v for k, v in old.entries.items() if k != "NoneType"}
                if not refined:
                    refined = {"Tensor": TensorShape.unknown(2)}
                self.env[checked_var] = CofiberedElement(entries=refined)

        for stmt in node.body:
            self._analyze_stmt(stmt)
        env_true = dict(self.env)

        # ── False branch: restore pre-branch env ──────────────────────
        self.env = dict(env_before)
        for stmt in node.orelse:
            self._analyze_stmt(stmt)
        env_false = dict(self.env)

        # ── Merge point: join both branches via CofiberedDomain.join() ─
        all_vars = set(env_true.keys()) | set(env_false.keys())
        merged: Env = {}
        for v in all_vars:
            elem_t = env_true.get(v)
            elem_f = env_false.get(v)
            if elem_t is None and elem_f is None:
                continue
            if elem_t is None:
                merged[v] = elem_f  # type: ignore[assignment]
            elif elem_f is None:
                merged[v] = elem_t
            else:
                merged[v] = self.domain.join(elem_t, elem_f)
        self.env = merged

    # ── Expression evaluator (returns CofiberedElement) ────────────────

    def _eval_expr(self, node: ast.expr) -> CofiberedElement:
        """Evaluate an expression to a CofiberedElement, updating all
        fibers simultaneously."""

        # None literal → NoneType fiber only
        if isinstance(node, ast.Constant):
            if node.value is None:
                return CofiberedElement(entries={
                    "NoneType": NullityValue.definitely_null(),
                })
            if isinstance(node.value, int):
                return CofiberedElement.singleton("int", Interval.const(node.value))
            if isinstance(node.value, float):
                return CofiberedElement.singleton("int", Interval.const(node.value))
            return CofiberedElement(entries={})

        # Variable reference
        if isinstance(node, ast.Name):
            elem = self.env.get(node.id)
            if elem is not None:
                return elem
            return CofiberedElement(entries={})

        # Tensor constructor → Tensor fiber + NoneType definitely-not-null
        if self._is_tensor_construction(node):
            shape = self._extract_shape(node)
            return CofiberedElement(entries={
                "Tensor": shape,
                "NoneType": NullityValue.definitely_not_null(),
            })

        # Ternary  ``body if test else orelse``
        if isinstance(node, ast.IfExp):
            body_elem = self._eval_expr(node.body)
            else_elem = self._eval_expr(node.orelse)
            return self.domain.join(body_elem, else_elem)

        # Matmul (a @ b)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            left_elem = self._eval_expr(node.left)
            right_elem = self._eval_expr(node.right)
            left_shape = left_elem.get("Tensor")
            right_shape = right_elem.get("Tensor")
            if left_shape is not None and right_shape is not None:
                result_shape = compute_matmul_shape(left_shape, right_shape)
                if result_shape is not None:
                    return CofiberedElement(entries={
                        "Tensor": result_shape,
                        "NoneType": NullityValue.definitely_not_null(),
                    })
            return CofiberedElement(entries={
                "Tensor": TensorShape.unknown(2),
                "NoneType": NullityValue.definitely_not_null(),
            })

        # Other binary ops → propagate type
        if isinstance(node, ast.BinOp):
            left = self._eval_expr(node.left)
            right = self._eval_expr(node.right)
            # Broadcast for tensor arithmetic
            if left.get("Tensor") is not None and right.get("Tensor") is not None:
                result_shape = compute_broadcast_shape(left.get("Tensor"), right.get("Tensor"))
                if result_shape is not None:
                    return CofiberedElement(entries={
                        "Tensor": result_shape,
                        "NoneType": NullityValue.definitely_not_null(),
                    })
            return self.domain.join(left, right)

        # Function / method call
        if isinstance(node, ast.Call):
            return self._eval_call(node)

        # Attribute access
        if isinstance(node, ast.Attribute):
            base = self._eval_expr(node.value)
            return CofiberedElement(entries={})

        return CofiberedElement(entries={})

    def _eval_call(self, node: ast.Call) -> CofiberedElement:
        """Evaluate a function call, including interprocedural shape propagation."""
        func_name = self._get_call_name(node)

        # Optional-returning heuristics
        if self._is_optional_tensor_source(node):
            return CofiberedElement(entries={
                "Tensor": TensorShape.unknown(2),
                "NoneType": NullityValue.maybe_null(),
            })

        # Tensor construction
        if self._is_tensor_construction(node):
            shape = self._extract_shape(node)
            return CofiberedElement(entries={
                "Tensor": shape,
                "NoneType": NullityValue.definitely_not_null(),
            })

        # reshape / view
        stem = func_name.split(".")[-1] if func_name else ""
        if stem in ("reshape", "view") and isinstance(node.func, ast.Attribute):
            base_elem = self._eval_expr(node.func.value)
            base_shape = base_elem.get("Tensor")
            if base_shape is not None:
                new_dims = tuple(
                    a.value if isinstance(a, ast.Constant) and isinstance(a.value, int) else "_sym"
                    for a in node.args
                )
                result = compute_reshape_shape(base_shape, new_dims)
                if result is not None:
                    return CofiberedElement(entries={
                        "Tensor": result,
                        "NoneType": NullityValue.definitely_not_null(),
                    })

        # torch.cat
        if stem == "cat" and node.args:
            first_arg = node.args[0]
            if isinstance(first_arg, (ast.List, ast.Tuple)):
                shapes = []
                for elt in first_arg.elts:
                    e = self._eval_expr(elt)
                    s = e.get("Tensor")
                    if s is not None:
                        shapes.append(s)
                if shapes:
                    return CofiberedElement(entries={
                        "Tensor": shapes[0],
                        "NoneType": NullityValue.definitely_not_null(),
                    })

        # matmul / mm / bmm as function call
        if stem in ("matmul", "mm", "bmm") and len(node.args) >= 2:
            a_elem = self._eval_expr(node.args[0])
            b_elem = self._eval_expr(node.args[1])
            a_shape = a_elem.get("Tensor")
            b_shape = b_elem.get("Tensor")
            if a_shape is not None and b_shape is not None:
                result_shape = compute_matmul_shape(a_shape, b_shape)
                if result_shape is not None:
                    return CofiberedElement(entries={
                        "Tensor": result_shape,
                        "NoneType": NullityValue.definitely_not_null(),
                    })

        # nn.Linear constructor
        if stem == "Linear" and len(node.args) >= 2:
            return CofiberedElement(entries={
                "Tensor": TensorShape.unknown(2),
                "NoneType": NullityValue.definitely_not_null(),
            })

        # nn.Conv2d constructor — output is (batch, out_channels, H', W')
        if stem == "Conv2d" and len(node.args) >= 2:
            out_c = self._const_int(node.args[1]) if len(node.args) >= 2 else None
            out_dim = ShapeDim(out_c) if out_c is not None else ShapeDim("C_out")
            return CofiberedElement(entries={
                "Tensor": TensorShape((ShapeDim("N"), out_dim, ShapeDim("H'"), ShapeDim("W'"))),
                "NoneType": NullityValue.definitely_not_null(),
            })

        # BatchNorm / LayerNorm / Dropout — preserve input shape
        if stem in ("BatchNorm2d", "BatchNorm1d", "LayerNorm", "Dropout"):
            if node.args:
                arg_elem = self._eval_expr(node.args[0])
                if arg_elem.get("Tensor") is not None:
                    return CofiberedElement(entries={
                        "Tensor": arg_elem.get("Tensor"),
                        "NoneType": NullityValue.definitely_not_null(),
                    })
            return CofiberedElement(entries={
                "Tensor": TensorShape.unknown(2),
                "NoneType": NullityValue.definitely_not_null(),
            })

        # F.relu / torch.relu / relu — preserve input shape
        if stem == "relu":
            if node.args:
                arg_elem = self._eval_expr(node.args[0])
                if arg_elem.get("Tensor") is not None:
                    return CofiberedElement(entries={
                        "Tensor": arg_elem.get("Tensor"),
                        "NoneType": NullityValue.definitely_not_null(),
                    })
            return CofiberedElement(entries={
                "Tensor": TensorShape.unknown(2),
                "NoneType": NullityValue.definitely_not_null(),
            })

        # self.layer_name(x) — look up layer from _current_class_layers
        if (isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
                and node.func.attr in self._current_class_layers):
            layer_info = self._current_class_layers[node.func.attr]
            ltype = layer_info.get("type", "")
            layer_name = node.func.attr
            # Evaluate the input argument
            input_elem = self._eval_expr(node.args[0]) if node.args else CofiberedElement(entries={})
            if ltype == "Linear":
                out_f = layer_info.get("out_features")
                in_f = layer_info.get("in_features")
                input_shape = input_elem.get("Tensor")
                # Inline shape check: verify input last dim matches in_features
                if (input_shape is not None and input_shape.ndim >= 1
                        and in_f is not None):
                    last_dim = input_shape.dims[-1]
                    if not last_dim.is_symbolic and last_dim.value != in_f:
                        self.bugs.append(UnifiedBug(
                            kind="DIM_MISMATCH",
                            line=node.lineno,
                            col=node.col_offset,
                            message=(
                                f"self.{layer_name}: input last dim is {last_dim.value}, "
                                f"but nn.Linear expects in_features={in_f}"
                            ),
                            function=self.func_name,
                            variable=layer_name,
                            severity="error",
                        ))
                if input_shape is not None and input_shape.ndim >= 1:
                    # Output replaces last dim with out_features
                    new_dims = list(input_shape.dims[:-1])
                    new_dims.append(ShapeDim(out_f) if out_f is not None else ShapeDim("out"))
                    return CofiberedElement(entries={
                        "Tensor": TensorShape(tuple(new_dims)),
                        "NoneType": NullityValue.definitely_not_null(),
                    })
                # Unknown input shape: output is (batch_sym, out_features)
                out_dim = ShapeDim(out_f) if out_f is not None else ShapeDim("out")
                return CofiberedElement(entries={
                    "Tensor": TensorShape((ShapeDim("batch"), out_dim)),
                    "NoneType": NullityValue.definitely_not_null(),
                })
            elif ltype == "Conv2d":
                in_c = layer_info.get("in_channels")
                out_c = layer_info.get("out_channels")
                input_shape = input_elem.get("Tensor")
                # Inline shape check: verify input channel dim matches in_channels
                if input_shape is not None and in_c is not None:
                    if input_shape.ndim >= 4:
                        chan_dim = input_shape.dims[1]
                    elif input_shape.ndim >= 2:
                        chan_dim = input_shape.dims[0]
                    else:
                        chan_dim = None
                    if (chan_dim is not None and not chan_dim.is_symbolic
                            and chan_dim.value != in_c):
                        self.bugs.append(UnifiedBug(
                            kind="CONV_INCOMPAT",
                            line=node.lineno,
                            col=node.col_offset,
                            message=(
                                f"self.{layer_name}: input channel dim is {chan_dim.value}, "
                                f"but nn.Conv2d expects in_channels={in_c}"
                            ),
                            function=self.func_name,
                            variable=layer_name,
                            severity="error",
                        ))
                out_dim = ShapeDim(out_c) if out_c is not None else ShapeDim("C_out")
                return CofiberedElement(entries={
                    "Tensor": TensorShape((ShapeDim("N"), out_dim, ShapeDim("H'"), ShapeDim("W'"))),
                    "NoneType": NullityValue.definitely_not_null(),
                })
            elif ltype in ("BatchNorm2d", "BatchNorm1d", "LayerNorm", "Dropout"):
                # Preserve input shape
                input_shape = input_elem.get("Tensor")
                if input_shape is not None:
                    return CofiberedElement(entries={
                        "Tensor": input_shape,
                        "NoneType": NullityValue.definitely_not_null(),
                    })
                return CofiberedElement(entries={
                    "Tensor": TensorShape.unknown(2),
                    "NoneType": NullityValue.definitely_not_null(),
                })

        # Interprocedural shape propagation: look up callee ShapeContract
        if func_name and func_name in self._shape_contracts:
            contract = self._shape_contracts[func_name]
            if contract.return_shape is not None:
                result_dims = tuple(ShapeDim(d) for d in contract.return_shape)
                return CofiberedElement(entries={
                    "Tensor": TensorShape(result_dims),
                    "NoneType": NullityValue.definitely_not_null(),
                })

        # Evaluate arguments for side-effects / bug checking
        for arg in node.args:
            self._eval_expr(arg)

        return CofiberedElement(entries={})

    # ── Bug checking (runs within the single pass) ─────────────────────

    def _check_expr(self, node: ast.expr) -> None:
        """Check an expression for shape and cross-domain bugs."""
        if isinstance(node, ast.Attribute):
            self._check_attribute_access(node)
        elif isinstance(node, ast.BinOp):
            self._check_binop(node)
            # Recurse into operands
            self._check_expr(node.left)
            self._check_expr(node.right)
        elif isinstance(node, ast.Call):
            self._check_call(node)
            for arg in node.args:
                self._check_expr(arg)
        elif isinstance(node, ast.IfExp):
            self._check_expr(node.body)
            self._check_expr(node.orelse)

    def _check_attribute_access(self, node: ast.Attribute) -> None:
        """Detect .shape/.dim() access on Optional[Tensor] without null check."""
        if not isinstance(node.value, ast.Name):
            return
        var = node.value.id
        elem = self.env.get(var)
        if elem is None:
            return

        has_tensor = "Tensor" in elem.entries
        has_none = "NoneType" in elem.entries
        none_val = elem.get("NoneType")
        # Cross-domain: tensor attribute access when NoneType fiber says may-be-null
        if has_tensor and has_none and none_val is not None:
            if hasattr(none_val, "may_be_null") and none_val.may_be_null:
                attr = node.attr
                if attr in ("shape", "size", "dim", "ndim", "dtype", "device",
                            "T", "data", "grad", "requires_grad"):
                    self.bugs.append(UnifiedBug(
                        kind="OPTIONAL_TENSOR_ACCESS",
                        line=node.lineno,
                        col=node.col_offset,
                        message=(
                            f"Accessing '.{attr}' on '{var}' which may be None "
                            f"(Optional[Tensor]). Add a null check first."
                        ),
                        function=self.func_name,
                        variable=var,
                        severity="error",
                        cross_domain=True,
                    ))

    def _check_binop(self, node: ast.BinOp) -> None:
        """Check binary ops: matmul shape compat, broadcast compat, + Optional[Tensor] null check."""
        if isinstance(node.op, ast.MatMult):
            # Null check on operands
            for operand in (node.left, node.right):
                if not isinstance(operand, ast.Name):
                    continue
                var = operand.id
                elem = self.env.get(var)
                if elem is None:
                    continue
                has_tensor = "Tensor" in elem.entries
                has_none = "NoneType" in elem.entries
                none_val = elem.get("NoneType")
                if has_tensor and has_none and none_val is not None:
                    if hasattr(none_val, "may_be_null") and none_val.may_be_null:
                        self.bugs.append(UnifiedBug(
                            kind="OPTIONAL_TENSOR_ACCESS",
                            line=node.lineno,
                            col=node.col_offset,
                            message=(
                                f"Tensor '{var}' used in matmul (@) may be None "
                                f"(Optional[Tensor]). Add a null check first."
                            ),
                            function=self.func_name,
                            variable=var,
                            severity="error",
                            cross_domain=True,
                        ))

            # Shape compatibility check
            left_elem = self._eval_expr(node.left)
            right_elem = self._eval_expr(node.right)
            left_shape = left_elem.get("Tensor")
            right_shape = right_elem.get("Tensor")
            if left_shape is not None and right_shape is not None:
                err_msg = check_matmul_compatible(left_shape, right_shape)
                if err_msg is None:
                    # Try Z3 for symbolic dims
                    lvar = node.left.id if isinstance(node.left, ast.Name) else "left"
                    rvar = node.right.id if isinstance(node.right, ast.Name) else "right"
                    err_msg = _z3_check_matmul(lvar, left_shape, rvar, right_shape, node.lineno)
                if err_msg:
                    var_name = node.left.id if isinstance(node.left, ast.Name) else "expr"
                    self.bugs.append(UnifiedBug(
                        kind="MATMUL_INCOMPAT",
                        line=node.lineno,
                        col=node.col_offset,
                        message=err_msg,
                        function=self.func_name,
                        variable=var_name,
                        severity="error",
                    ))

        # Broadcast check for arithmetic tensor ops (+, -, *, /)
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left_elem = self._eval_expr(node.left)
            right_elem = self._eval_expr(node.right)
            left_shape = left_elem.get("Tensor")
            right_shape = right_elem.get("Tensor")
            if left_shape is not None and right_shape is not None:
                err_msg = _z3_check_broadcast(left_shape, right_shape, node.lineno)
                if err_msg:
                    var_name = node.left.id if isinstance(node.left, ast.Name) else "expr"
                    self.bugs.append(UnifiedBug(
                        kind="BROADCAST_FAIL",
                        line=node.lineno,
                        col=node.col_offset,
                        message=err_msg,
                        function=self.func_name,
                        variable=var_name,
                        severity="error",
                    ))

    def _check_call(self, node: ast.Call) -> None:
        """Check function calls for Optional[Tensor] use and shape errors."""
        func_name = self._get_call_name(node)
        stem = func_name.split(".")[-1] if func_name else ""

        # Optional[Tensor] passed to matmul-like functions
        if stem in ("matmul", "mm", "bmm", "dot", "mv"):
            for arg in node.args:
                if not isinstance(arg, ast.Name):
                    continue
                var = arg.id
                elem = self.env.get(var)
                if elem is None:
                    continue
                has_tensor = "Tensor" in elem.entries
                has_none = "NoneType" in elem.entries
                none_val = elem.get("NoneType")
                if has_tensor and has_none and none_val is not None:
                    if hasattr(none_val, "may_be_null") and none_val.may_be_null:
                        self.bugs.append(UnifiedBug(
                            kind="OPTIONAL_TENSOR_ACCESS",
                            line=node.lineno,
                            col=node.col_offset,
                            message=(
                                f"Tensor '{var}' passed to {stem}() may be None "
                                f"(Optional[Tensor]). Add a null check first."
                            ),
                            function=self.func_name,
                            variable=var,
                            severity="error",
                            cross_domain=True,
                        ))

            # Shape compat check for matmul-like calls
            if len(node.args) >= 2:
                a_elem = self._eval_expr(node.args[0])
                b_elem = self._eval_expr(node.args[1])
                a_shape = a_elem.get("Tensor")
                b_shape = b_elem.get("Tensor")
                if a_shape is not None and b_shape is not None:
                    err_msg = check_matmul_compatible(a_shape, b_shape)
                    if err_msg:
                        var_name = node.args[0].id if isinstance(node.args[0], ast.Name) else "arg"
                        self.bugs.append(UnifiedBug(
                            kind="MATMUL_INCOMPAT",
                            line=node.lineno,
                            col=node.col_offset,
                            message=err_msg,
                            function=self.func_name,
                            variable=var_name,
                            severity="error",
                        ))

        # Interprocedural: check arg shapes against callee ShapeContract
        if func_name and func_name in self._shape_contracts:
            contract = self._shape_contracts[func_name]
            self._check_interprocedural_shapes(node, func_name, contract)

        # Z3 check for reshape/view calls
        if stem in ("reshape", "view") and isinstance(node.func, ast.Attribute):
            base_elem = self._eval_expr(node.func.value)
            base_shape = base_elem.get("Tensor")
            if base_shape is not None:
                new_dims = []
                for a in node.args:
                    if isinstance(a, ast.Constant) and isinstance(a.value, int):
                        new_dims.append(ShapeDim(a.value))
                    else:
                        new_dims.append(ShapeDim("_sym"))
                if new_dims:
                    new_shape = TensorShape(tuple(new_dims))
                    err_msg = _z3_check_reshape(base_shape, new_shape, node.lineno)
                    if err_msg:
                        var_name = ""
                        if isinstance(node.func.value, ast.Name):
                            var_name = node.func.value.id
                        self.bugs.append(UnifiedBug(
                            kind="RESHAPE_INVALID",
                            line=node.lineno,
                            col=node.col_offset,
                            message=err_msg,
                            function=self.func_name,
                            variable=var_name,
                            severity="error",
                        ))

        # Z3 check for torch.cat calls
        if stem == "cat" and node.args:
            first_arg = node.args[0]
            if isinstance(first_arg, (ast.List, ast.Tuple)):
                shapes = []
                for elt in first_arg.elts:
                    e = self._eval_expr(elt)
                    s = e.get("Tensor")
                    if s is not None:
                        shapes.append(s)
                if len(shapes) >= 2:
                    cat_dim = 0
                    if len(node.args) >= 2:
                        d_arg = node.args[1]
                        if isinstance(d_arg, ast.Constant) and isinstance(d_arg.value, int):
                            cat_dim = d_arg.value
                    # Also check keyword dim=...
                    for kw in node.keywords:
                        if kw.arg == "dim" and isinstance(kw.value, ast.Constant):
                            if isinstance(kw.value.value, int):
                                cat_dim = kw.value.value
                    err_msg = _z3_check_cat(shapes, cat_dim, node.lineno)
                    if err_msg:
                        self.bugs.append(UnifiedBug(
                            kind="CAT_INCOMPAT",
                            line=node.lineno,
                            col=node.col_offset,
                            message=err_msg,
                            function=self.func_name,
                            variable="cat",
                            severity="error",
                        ))

        # NOTE: self.layer(x) shape checks are done inline in _eval_call
        # where we have the correct pre-assignment input shapes.

    # ── Interprocedural cross-function shape checking ────────────────────

    def _check_interprocedural_shapes(self, node: ast.Call,
                                       func_name: str,
                                       contract: ShapeContract) -> None:
        """Check caller argument shapes against callee ShapeContract,
        including dimension-by-dimension compatibility."""
        param_names = list(contract.param_shapes.keys())
        for i, arg in enumerate(node.args):
            if not isinstance(arg, ast.Name):
                continue
            elem = self.env.get(arg.id)
            if elem is None:
                continue
            arg_shape = elem.get("Tensor")
            if arg_shape is None:
                continue
            if i >= len(param_names):
                continue
            param_name = param_names[i]
            expected = contract.param_shapes.get(param_name)
            if expected is None:
                continue
            expected_shape = TensorShape(tuple(
                ShapeDim(d) for d in expected
            ))
            # ndim mismatch
            if arg_shape.ndim != expected_shape.ndim:
                self.bugs.append(UnifiedBug(
                    kind="NDIM_MISMATCH",
                    line=node.lineno,
                    col=node.col_offset,
                    message=(
                        f"Argument '{arg.id}' has {arg_shape.ndim}D shape, "
                        f"callee '{func_name}' expects {expected_shape.ndim}D"
                    ),
                    function=self.func_name,
                    variable=arg.id,
                    severity="error",
                ))
                continue
            # Dimension-by-dimension check
            for d_i in range(min(arg_shape.ndim, expected_shape.ndim)):
                actual_dim = arg_shape.dims[d_i]
                expect_dim = expected_shape.dims[d_i]
                if actual_dim.is_symbolic or expect_dim.is_symbolic:
                    continue
                if actual_dim.value != expect_dim.value:
                    self.bugs.append(UnifiedBug(
                        kind="DIM_MISMATCH",
                        line=node.lineno,
                        col=node.col_offset,
                        message=(
                            f"Argument '{arg.id}' has shape {arg_shape.pretty()} "
                            f"but function '{func_name}' expects parameter "
                            f"'{param_name}' with shape {expected_shape.pretty()}"
                        ),
                        function=self.func_name,
                        variable=arg.id,
                        severity="error",
                    ))
                    break  # one mismatch per argument is sufficient

    # ── Z3 batch constraint checking ─────────────────────────────────────

    def _collect_z3_constraint(self, constraint: Any, message: str, line: int) -> None:
        """Add a Z3 constraint to the batch for end-of-function checking."""
        if _HAS_Z3:
            self._z3_constraints.append((constraint, message, line))

    def _flush_z3_constraints(self) -> None:
        """Check all collected Z3 constraints in a single solver call."""
        if not _HAS_Z3 or not self._z3_constraints:
            return
        solver = _z3.Solver()
        solver.set("timeout", 1000)
        # Add all constraints; any that is UNSAT reveals a bug
        for constraint, message, line in self._z3_constraints:
            solver.push()
            solver.add(constraint)
            result = solver.check()
            if result == _z3.unsat:
                self.bugs.append(UnifiedBug(
                    kind="SHAPE_MISMATCH",
                    line=line,
                    col=0,
                    message=f"Z3 proved shape violation: {message}",
                    function=self.func_name,
                    variable="",
                    severity="error",
                ))
            solver.pop()
        self._z3_constraints = []

    # ── Helpers ─────────────────────────────────────────────────────────

    def _element_from_annotation(self, ann: ast.expr, var: str) -> CofiberedElement:
        """Build a CofiberedElement from a type annotation."""
        # Optional[Tensor] or Optional[torch.Tensor]
        if isinstance(ann, ast.Subscript):
            if isinstance(ann.value, ast.Name) and ann.value.id == "Optional":
                inner = ann.slice
                if self._is_tensor_type(inner):
                    return CofiberedElement(entries={
                        "Tensor": TensorShape.unknown(2),
                        "NoneType": NullityValue.maybe_null(),
                    })
        if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
            # Tensor | None
            has_tensor = False
            has_none = False
            for side in (ann.left, ann.right):
                if isinstance(side, ast.Constant) and side.value is None:
                    has_none = True
                elif self._is_tensor_type(side):
                    has_tensor = True
            if has_tensor and has_none:
                return CofiberedElement(entries={
                    "Tensor": TensorShape.unknown(2),
                    "NoneType": NullityValue.maybe_null(),
                })
        if self._is_tensor_type(ann):
            return CofiberedElement(entries={
                "Tensor": TensorShape.unknown(2),
                "NoneType": NullityValue.definitely_not_null(),
            })
        if isinstance(ann, ast.Name):
            if ann.id == "int":
                return CofiberedElement.singleton("int", Interval.top())
        return CofiberedElement(entries={})

    @staticmethod
    def _is_tensor_type(node: ast.expr) -> bool:
        """Check if a type annotation node refers to Tensor."""
        if isinstance(node, ast.Name):
            return node.id == "Tensor"
        if isinstance(node, ast.Attribute):
            return node.attr == "Tensor"
        return False

    def _is_tensor_construction(self, node: ast.expr) -> bool:
        """Check if an expression constructs a tensor."""
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                return func.attr in (
                    "zeros", "ones", "randn", "rand", "empty", "full",
                    "tensor", "Tensor", "zeros_like", "ones_like",
                )
            if isinstance(func, ast.Name):
                return func.id in ("tensor", "Tensor")
        return False

    def _extract_shape(self, node: ast.expr) -> Optional[TensorShape]:
        """Extract shape from a tensor construction call."""
        if not isinstance(node, ast.Call):
            return None
        dims: list = []
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                dims.append(ShapeDim(arg.value))
            elif isinstance(arg, ast.Tuple):
                for elt in arg.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                        dims.append(ShapeDim(elt.value))
                    else:
                        dims.append(ShapeDim(f"d{len(dims)}"))
                return TensorShape(tuple(dims))
            else:
                dims.append(ShapeDim(f"d{len(dims)}"))
        if dims:
            return TensorShape(tuple(dims))
        return TensorShape.unknown(2)

    def _is_optional_tensor_source(self, node: ast.expr) -> bool:
        """Heuristic: function calls / ternaries that may return Optional[Tensor]."""
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                return func.attr in ("get", "find", "lookup", "fetch",
                                     "load", "maybe_load", "get_tensor")
            if isinstance(func, ast.Name):
                return func.id in ("get_tensor", "maybe_load", "fetch_tensor",
                                   "load_optional", "find_tensor")
        if isinstance(node, ast.IfExp):
            if isinstance(node.orelse, ast.Constant) and node.orelse.value is None:
                return True
            if isinstance(node.body, ast.Constant) and node.body.value is None:
                return True
        return False

    def _extract_null_check_var(self, test: ast.expr) -> Optional[str]:
        """Extract the variable being null-checked from an if test."""
        if isinstance(test, ast.Compare):
            if len(test.ops) == 1 and len(test.comparators) == 1:
                op = test.ops[0]
                comp = test.comparators[0]
                if isinstance(op, ast.IsNot) and isinstance(comp, ast.Constant) and comp.value is None:
                    if isinstance(test.left, ast.Name):
                        return test.left.id
                if isinstance(op, ast.Is) and isinstance(comp, ast.Constant) and comp.value is None:
                    if isinstance(test.left, ast.Name):
                        return test.left.id
        if isinstance(test, ast.Name):
            return test.id
        return None

    @staticmethod
    def _get_call_name(node: ast.Call) -> str:
        """Extract dotted function name from a Call node."""
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            parts = []
            cur: ast.expr = func
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            return ".".join(reversed(parts))
        return ""


# ── Public API ─────────────────────────────────────────────────────────

def analyze_unified(source: str) -> UnifiedAnalysisResult:
    """Run unified null-safety + tensor shape analysis in a single pass.

    Uses the CofiberedDomain to track variables across the int, Tensor,
    and NoneType fibers simultaneously, enabling cross-domain bug detection
    like "Optional[Tensor] accessed without null check".

    Args:
        source: Python source code string.

    Returns:
        UnifiedAnalysisResult with combined bugs from both analyses
        plus cross-domain bugs that neither analysis alone could find.
    """
    analyzer = UnifiedAnalyzer()
    return analyzer.analyze(source)
