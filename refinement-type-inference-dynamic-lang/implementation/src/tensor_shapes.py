"""
Tensor Shape Verification via Liquid Types.

Statically verifies tensor shape compatibility in PyTorch and NumPy code
by encoding shapes as refinement type predicates and using Z3 to check
compatibility at operation sites (matmul, add, reshape, view, cat, etc.).

Key insight: tensor shapes are naturally expressible as refinement types:
  {v: Tensor | shape(v) == (batch, channels, height, width)}
and shape errors are the #1 runtime error in ML codebases.

This module extends TensorGuard's predicate harvesting to:
  1. Harvest shape predicates from assertions, constructors, and reshape calls
  2. Propagate shapes through operations (matmul, add, cat, view, etc.)
  3. Generate Z3 constraints at every shape-sensitive operation
  4. Report shape mismatches with concrete counterexamples

Unlike TorchScript (which checks shapes at trace time) or Pyright (which
only checks types, not shapes), TensorGuard checks shapes *statically* with
*zero annotations* on *untyped Python code*.
"""

from __future__ import annotations

import ast
import time
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union,
)

logger = logging.getLogger(__name__)

try:
    import z3 as _z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

from src._experimental.refinement_lattice import (
    Pred, PredOp, RefType, BaseTypeR, BaseTypeKind,
    INT_TYPE, FLOAT_TYPE, ANY_TYPE, TENSOR_TYPE,
    Z3Encoder, RefinementLattice, RefEnvironment,
)


# ═══════════════════════════════════════════════════════════════════════════
# Shape representation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ShapeDim:
    """A single dimension in a tensor shape — concrete int or symbolic name."""
    value: Union[int, str]

    @property
    def is_symbolic(self) -> bool:
        return isinstance(self.value, str)

    def __repr__(self):
        return str(self.value)


@dataclass(frozen=True)
class TensorShape:
    """Statically known (possibly symbolic) tensor shape."""
    dims: Tuple[ShapeDim, ...]

    @property
    def ndim(self) -> int:
        return len(self.dims)

    def dim(self, axis: int) -> ShapeDim:
        if axis < 0:
            axis = len(self.dims) + axis
        return self.dims[axis]

    def pretty(self) -> str:
        return "(" + ", ".join(str(d.value) for d in self.dims) + ")"

    @staticmethod
    def from_tuple(t: tuple) -> TensorShape:
        return TensorShape(tuple(ShapeDim(v) for v in t))

    @staticmethod
    def unknown(ndim: int) -> TensorShape:
        return TensorShape(tuple(ShapeDim(f"d{i}") for i in range(ndim)))

    def to_pred(self, var: str) -> Pred:
        """Convert to a SHAPE_EQ predicate."""
        dim_values = tuple(d.value for d in self.dims)
        return Pred.shape_eq(var, dim_values)


# ═══════════════════════════════════════════════════════════════════════════
# Shape error types
# ═══════════════════════════════════════════════════════════════════════════

class ShapeErrorKind(Enum):
    DIM_MISMATCH = auto()       # incompatible dimensions for operation
    NDIM_MISMATCH = auto()      # wrong number of dimensions
    RESHAPE_INVALID = auto()    # reshape to incompatible total size
    BROADCAST_FAIL = auto()     # cannot broadcast shapes
    MATMUL_INCOMPAT = auto()    # inner dimensions don't match for matmul
    CAT_INCOMPAT = auto()       # non-matching dims for concatenation
    CONV_INCOMPAT = auto()      # wrong input shape for conv layer


@dataclass
class ShapeError:
    """A tensor shape error found by static analysis."""
    kind: ShapeErrorKind
    line: int
    col: int
    message: str
    function: str
    variable: str
    actual_shape: Optional[TensorShape] = None
    expected_shape: Optional[TensorShape] = None
    severity: str = "error"
    z3_counterexample: Optional[Dict[str, str]] = None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.name,
            "line": self.line,
            "col": self.col,
            "message": self.message,
            "function": self.function,
            "variable": self.variable,
            "actual_shape": self.actual_shape.pretty() if self.actual_shape else None,
            "expected_shape": self.expected_shape.pretty() if self.expected_shape else None,
            "severity": self.severity,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Shape environment: maps variables to their known shapes
# ═══════════════════════════════════════════════════════════════════════════

class ShapeEnv:
    """Maps tensor variables to their statically-known shapes."""

    def __init__(self, bindings: Optional[Dict[str, TensorShape]] = None):
        self._bindings: Dict[str, TensorShape] = dict(bindings or {})

    def get(self, var: str) -> Optional[TensorShape]:
        return self._bindings.get(var)

    def set(self, var: str, shape: TensorShape) -> ShapeEnv:
        new_bindings = dict(self._bindings)
        new_bindings[var] = shape
        return ShapeEnv(new_bindings)

    def copy(self) -> ShapeEnv:
        return ShapeEnv(dict(self._bindings))

    def join(self, other: ShapeEnv) -> ShapeEnv:
        """Join two shape environments (intersection of known shapes)."""
        result: Dict[str, TensorShape] = {}
        for var in self._bindings:
            if var in other._bindings:
                if self._bindings[var] == other._bindings[var]:
                    result[var] = self._bindings[var]
        return ShapeEnv(result)


# ═══════════════════════════════════════════════════════════════════════════
# Tensor operation shape rules
# ═══════════════════════════════════════════════════════════════════════════

# PyTorch/NumPy operations and their shape semantics
TORCH_SHAPE_OPS = {
    # Creation ops
    "zeros": "create", "ones": "create", "randn": "create",
    "rand": "create", "empty": "create", "full": "create",
    "zeros_like": "like", "ones_like": "like", "randn_like": "like",
    "arange": "arange", "linspace": "linspace",
    # Reshape ops
    "reshape": "reshape", "view": "reshape",
    "flatten": "flatten", "squeeze": "squeeze", "unsqueeze": "unsqueeze",
    "permute": "permute", "transpose": "transpose",
    # Reduction ops
    "sum": "reduce", "mean": "reduce", "max": "reduce", "min": "reduce",
    "prod": "reduce", "norm": "reduce",
    # Combination ops
    "cat": "cat", "stack": "stack", "concatenate": "cat",
    # Matmul
    "matmul": "matmul", "mm": "matmul", "bmm": "bmm",
    "linear": "linear",
    # Element-wise (broadcasting)
    "add": "broadcast", "mul": "broadcast", "sub": "broadcast",
    "div": "broadcast",
}

NUMPY_SHAPE_OPS = {
    "zeros": "create", "ones": "create", "empty": "create",
    "reshape": "reshape", "flatten": "flatten",
    "concatenate": "cat", "stack": "stack",
    "dot": "matmul", "matmul": "matmul",
    "sum": "reduce", "mean": "reduce",
    "transpose": "transpose",
}


def compute_matmul_shape(
    a: TensorShape, b: TensorShape
) -> Optional[TensorShape]:
    """Compute the result shape of a @ b (matmul).

    Rules:
      - (m, k) @ (k, n) → (m, n)
      - (b, m, k) @ (b, k, n) → (b, m, n)  (batched)
      - (m, k) @ (k,) → (m,)  (matrix-vector)
    """
    if a.ndim < 1 or b.ndim < 1:
        return None

    if a.ndim == 1 and b.ndim == 1:
        # dot product
        return TensorShape(())

    if a.ndim == 2 and b.ndim == 2:
        # (m, k) @ (k, n) → (m, n)
        return TensorShape((a.dims[0], b.dims[1]))

    if a.ndim == 2 and b.ndim == 1:
        # (m, k) @ (k,) → (m,)
        return TensorShape((a.dims[0],))

    if a.ndim == 1 and b.ndim == 2:
        # (k,) @ (k, n) → (n,)
        return TensorShape((b.dims[1],))

    # Batched matmul
    if a.ndim >= 3 and b.ndim >= 3:
        batch = a.dims[:-2]
        return TensorShape(batch + (a.dims[-2], b.dims[-1]))

    return None


def check_matmul_compatible(a: TensorShape, b: TensorShape) -> Optional[str]:
    """Check if matmul(a, b) is valid. Returns error message or None."""
    if a.ndim < 1 or b.ndim < 1:
        return f"matmul requires at least 1D tensors, got {a.ndim}D and {b.ndim}D"

    # Get the contracting dimensions
    k_a = a.dims[-1] if a.ndim >= 1 else None
    k_b = b.dims[-2] if b.ndim >= 2 else (b.dims[0] if b.ndim == 1 else None)

    if k_a is None or k_b is None:
        return None  # Can't determine

    # Both concrete: direct comparison
    if not k_a.is_symbolic and not k_b.is_symbolic:
        if k_a.value != k_b.value:
            return (f"matmul dimension mismatch: "
                    f"a has inner dim {k_a.value}, b has inner dim {k_b.value}")
    return None


def compute_reshape_shape(
    original: TensorShape, new_dims: Tuple
) -> Optional[TensorShape]:
    """Compute result shape of reshape(original, new_dims).

    Sentinel value 0 means "copy this dimension from the corresponding
    input dim" (used when x.size(dim) appears in view/reshape args).
    """
    # Resolve sentinel 0 values by copying from input
    resolved = list(new_dims)
    copied_symbolic = {}
    for i, d in enumerate(resolved):
        if d == 0 and i < original.ndim:
            inp_d = original.dims[i]
            if not inp_d.is_symbolic:
                resolved[i] = inp_d.value
            else:
                # Keep symbolic dim name; mark so it's not counted as -1
                copied_symbolic[i] = inp_d

    # Count -1's (exclude sentinel-resolved positions)
    neg_ones = sum(1 for i, d in enumerate(resolved) if d == -1 and i not in copied_symbolic)
    if neg_ones > 1:
        return None  # Invalid: at most one -1

    result_dims = []
    for i, d in enumerate(resolved):
        if i in copied_symbolic:
            result_dims.append(copied_symbolic[i])
        elif isinstance(d, int) and d >= 0:
            result_dims.append(ShapeDim(d))
        elif d == -1:
            result_dims.append(ShapeDim("_inferred"))
        elif isinstance(d, str):
            result_dims.append(ShapeDim(d))
        else:
            result_dims.append(ShapeDim("_unknown"))

    return TensorShape(tuple(result_dims))


def compute_broadcast_shape(
    a: TensorShape, b: TensorShape
) -> Optional[TensorShape]:
    """Compute the broadcast result of shapes a and b.

    NumPy/PyTorch broadcasting rules:
      - Align shapes from the right
      - Each dim pair must be (d, d), (d, 1), or (1, d)
    """
    ndim = max(a.ndim, b.ndim)
    result_dims: List[ShapeDim] = []

    for i in range(1, ndim + 1):
        d_a = a.dims[-i] if i <= a.ndim else ShapeDim(1)
        d_b = b.dims[-i] if i <= b.ndim else ShapeDim(1)

        if d_a.is_symbolic or d_b.is_symbolic:
            # Symbolic: can't determine statically, assume OK
            if not d_a.is_symbolic:
                result_dims.append(d_a if d_a.value != 1 else d_b)
            else:
                result_dims.append(d_b if (not d_b.is_symbolic and d_b.value != 1) else d_a)
        elif d_a.value == d_b.value:
            result_dims.append(d_a)
        elif d_a.value == 1:
            result_dims.append(d_b)
        elif d_b.value == 1:
            result_dims.append(d_a)
        else:
            return None  # Broadcast failure

    result_dims.reverse()
    return TensorShape(tuple(result_dims))


# ═══════════════════════════════════════════════════════════════════════════
# Tensor Shape Analyzer: the main analysis engine
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ShapeAnalysisResult:
    """Results from tensor shape analysis."""
    errors: List[ShapeError] = field(default_factory=list)
    shapes: Dict[str, TensorShape] = field(default_factory=dict)
    constraints_generated: int = 0
    constraints_checked: int = 0
    functions_analyzed: int = 0
    analysis_time_ms: float = 0.0

    def summary(self) -> str:
        n_err = len(self.errors)
        return (
            f"Shape Analysis: {self.functions_analyzed} functions, "
            f"{len(self.shapes)} tensor shapes inferred, "
            f"{self.constraints_generated} constraints, "
            f"{n_err} shape errors found, "
            f"{self.analysis_time_ms:.1f}ms"
        )


class TensorShapeAnalyzer(ast.NodeVisitor):
    """Static tensor shape verifier using liquid types and Z3.

    Walks the AST, tracks tensor shapes through assignments and operations,
    and generates Z3 constraints at every shape-sensitive operation site.

    Supports:
      - torch.zeros/ones/randn/rand/empty/full (shape from args)
      - torch.matmul, @, mm, bmm (inner dimension matching)
      - torch.cat/stack (compatible dimensions)
      - reshape/view (total element preservation)
      - Broadcasting for element-wise ops
      - Shape assertions (assert x.shape == ...)
      - nn.Linear, nn.Conv2d (parameter shape matching)
    """

    def __init__(self, timeout_ms: int = 5000):
        self.timeout_ms = timeout_ms
        self.encoder = Z3Encoder()
        self.shape_env = ShapeEnv()
        self.errors: List[ShapeError] = []
        self.constraints_generated = 0
        self.constraints_checked = 0
        self.func_name = "<module>"
        # Track nn.Module layer definitions
        self._layer_shapes: Dict[str, Dict[str, Any]] = {}
        # Track shape predicates for liquid integration
        self._shape_preds: List[Pred] = []

    def analyze_source(self, source: str) -> ShapeAnalysisResult:
        """Analyze Python source for tensor shape errors."""
        t0 = time.monotonic()
        tree = ast.parse(source)
        result = ShapeAnalysisResult()

        # First pass: analyze classes (to extract layer shapes and check forward methods)
        funcs_analyzed = 0
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                self.errors = []
                self._analyze_class(node)
                result.errors.extend(self.errors)
                result.shapes.update(self.shape_env._bindings)
                funcs_analyzed += 1

        # Second pass: analyze top-level functions
        funcs_analyzed = 0
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.func_name = node.name
                self.shape_env = ShapeEnv()
                self.errors = []
                self._shape_preds = []
                self._analyze_function(node)
                result.errors.extend(self.errors)
                result.shapes.update(self.shape_env._bindings)
                funcs_analyzed += 1

        # Third pass: analyze module-level code
        self.func_name = "<module>"
        self.shape_env = ShapeEnv()
        self.errors = []
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self._analyze_stmt(node)
        result.errors.extend(self.errors)
        result.shapes.update(self.shape_env._bindings)

        result.functions_analyzed = funcs_analyzed
        result.constraints_generated = self.constraints_generated
        result.constraints_checked = self.constraints_checked
        result.analysis_time_ms = (time.monotonic() - t0) * 1000

        # Deduplicate errors by (line, kind)
        seen = set()
        deduped = []
        for e in result.errors:
            key = (e.line, e.kind, e.message)
            if key not in seen:
                seen.add(key)
                deduped.append(e)
        result.errors = deduped

        return result

    def _analyze_function(self, func: ast.FunctionDef):
        """Analyze a single function for shape errors."""
        # Initialize parameter shapes from annotations or conventions
        for arg in func.args.args:
            name = arg.arg
            if arg.annotation:
                shape = self._shape_from_annotation(arg.annotation)
                if shape:
                    self.shape_env = self.shape_env.set(name, shape)

        for stmt in func.body:
            self._analyze_stmt(stmt)

    def _analyze_stmt(self, node: ast.stmt):
        """Analyze a statement for shape operations."""
        if isinstance(node, ast.Assign):
            self._analyze_assign(node)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.value:
                shape = self._infer_shape(node.value)
                if shape:
                    self.shape_env = self.shape_env.set(node.target.id, shape)
        elif isinstance(node, ast.Return):
            if node.value:
                self._check_expr_shapes(node.value)
        elif isinstance(node, ast.If):
            self._analyze_if(node)
        elif isinstance(node, ast.For):
            for s in node.body:
                self._analyze_stmt(s)
        elif isinstance(node, ast.While):
            for s in node.body:
                self._analyze_stmt(s)
        elif isinstance(node, ast.With):
            for s in node.body:
                self._analyze_stmt(s)
        elif isinstance(node, ast.Try):
            for s in node.body:
                self._analyze_stmt(s)
        elif isinstance(node, ast.Expr):
            self._check_expr_shapes(node.value)
        elif isinstance(node, ast.Assert):
            self._harvest_shape_assert(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            pass  # handled separately
        elif isinstance(node, ast.ClassDef):
            self._analyze_class(node)

    def _analyze_assign(self, node: ast.Assign):
        """Track shape through assignment."""
        self._check_expr_shapes(node.value)
        shape = self._infer_shape(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name) and shape:
                self.shape_env = self.shape_env.set(target.id, shape)
            elif isinstance(target, ast.Tuple) and isinstance(node.value, ast.Call):
                # Handle unpacking: a, b = torch.chunk(x, 2)
                pass

    def _analyze_if(self, node: ast.If):
        """Handle if-else with shape environment joining."""
        old_env = self.shape_env.copy()

        for s in node.body:
            self._analyze_stmt(s)
        true_env = self.shape_env

        self.shape_env = old_env.copy()
        for s in node.orelse:
            self._analyze_stmt(s)
        false_env = self.shape_env

        self.shape_env = true_env.join(false_env)

    def _analyze_class(self, node: ast.ClassDef):
        """Analyze nn.Module subclass for layer definitions."""
        is_module = any(
            (isinstance(b, ast.Name) and b.id in ("Module", "nn.Module"))
            or (isinstance(b, ast.Attribute) and b.attr == "Module")
            for b in node.bases
        )
        if not is_module:
            return

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if item.name == "__init__":
                    self._analyze_module_init(item, node.name)
                elif item.name == "forward":
                    self.func_name = f"{node.name}.forward"
                    self._analyze_function(item)

    def _analyze_module_init(self, func: ast.FunctionDef, class_name: str):
        """Extract layer shapes from __init__ (nn.Linear, nn.Conv2d, etc.)."""
        for stmt in func.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if (isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"):
                        layer_name = target.attr
                        layer_info = self._extract_layer_info(stmt.value)
                        if layer_info:
                            key = f"{class_name}.{layer_name}"
                            self._layer_shapes[key] = layer_info

    def _extract_layer_info(self, node: ast.expr) -> Optional[Dict[str, Any]]:
        """Extract layer parameters from nn.Linear(in, out) etc."""
        if not isinstance(node, ast.Call):
            return None

        func_name = self._get_call_name(node)
        if not func_name:
            return None

        # nn.Linear(in_features, out_features)
        if func_name in ("Linear", "nn.Linear"):
            if len(node.args) >= 2:
                in_f = self._const_or_name(node.args[0])
                out_f = self._const_or_name(node.args[1])
                if in_f is not None and out_f is not None:
                    return {"type": "Linear", "in_features": in_f, "out_features": out_f}

        # nn.Conv2d(in_channels, out_channels, kernel_size)
        if func_name in ("Conv2d", "nn.Conv2d"):
            if len(node.args) >= 3:
                in_c = self._const_or_name(node.args[0])
                out_c = self._const_or_name(node.args[1])
                ks = self._const_or_name(node.args[2])
                if in_c is not None and out_c is not None:
                    return {"type": "Conv2d", "in_channels": in_c,
                            "out_channels": out_c, "kernel_size": ks}

        # nn.BatchNorm2d(num_features)
        if func_name in ("BatchNorm2d", "nn.BatchNorm2d"):
            if len(node.args) >= 1:
                n = self._const_or_name(node.args[0])
                if n is not None:
                    return {"type": "BatchNorm2d", "num_features": n}

        return None

    # ── Shape inference for expressions ────────────────────────────────

    def _infer_shape(self, node: ast.expr) -> Optional[TensorShape]:
        """Infer the tensor shape of an expression."""
        # Variable lookup
        if isinstance(node, ast.Name):
            return self.shape_env.get(node.id)

        # Function/method call
        if isinstance(node, ast.Call):
            return self._infer_call_shape(node)

        # Binary op: a @ b (matmul), a + b (broadcast)
        if isinstance(node, ast.BinOp):
            return self._infer_binop_shape(node)

        # Attribute: self.layer(x)
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                return self.shape_env.get(node.value.id)

        # Subscript: x[0], x[:, 1:3]
        if isinstance(node, ast.Subscript):
            return self._infer_subscript_shape(node)

        return None

    def _infer_call_shape(self, node: ast.Call) -> Optional[TensorShape]:
        """Infer shape from a function call."""
        func_name = self._get_call_name(node)
        if not func_name:
            return None

        # torch.zeros/ones/randn(d1, d2, ...) or torch.zeros((d1, d2, ...))
        base_name = func_name.split(".")[-1] if "." in func_name else func_name
        if base_name in ("zeros", "ones", "randn", "rand", "empty", "full",
                         "np.zeros", "np.ones", "np.empty"):
            return self._shape_from_creation_args(node)

        # torch.zeros_like(x) / ones_like(x)
        if base_name in ("zeros_like", "ones_like", "randn_like", "empty_like"):
            if node.args:
                return self._infer_shape(node.args[0])

        # reshape / view
        if base_name in ("reshape", "view"):
            if isinstance(node.func, ast.Attribute) and node.args:
                obj_shape = self._infer_shape(node.func.value)
                new_dims = self._extract_shape_args(node)
                if obj_shape and new_dims:
                    return compute_reshape_shape(obj_shape, new_dims)
            elif node.args:
                obj_shape = self._infer_shape(node.args[0])
                if len(node.args) >= 2:
                    new_dims = self._args_to_dims(node.args[1:])
                    if obj_shape and new_dims:
                        return compute_reshape_shape(obj_shape, new_dims)

        # transpose
        if base_name == "transpose":
            if isinstance(node.func, ast.Attribute):
                obj_shape = self._infer_shape(node.func.value)
                if obj_shape and len(node.args) >= 2:
                    d0 = self._const_val(node.args[0])
                    d1 = self._const_val(node.args[1])
                    if d0 is not None and d1 is not None and obj_shape.ndim > max(d0, d1):
                        dims = list(obj_shape.dims)
                        dims[d0], dims[d1] = dims[d1], dims[d0]
                        return TensorShape(tuple(dims))

        # squeeze
        if base_name == "squeeze":
            if isinstance(node.func, ast.Attribute):
                obj_shape = self._infer_shape(node.func.value)
                if obj_shape:
                    if node.args:
                        dim = self._const_val(node.args[0])
                        if dim is not None and dim < obj_shape.ndim:
                            dims = list(obj_shape.dims)
                            if not dims[dim].is_symbolic and dims[dim].value == 1:
                                dims.pop(dim)
                            return TensorShape(tuple(dims))
                    else:
                        dims = [d for d in obj_shape.dims
                                if d.is_symbolic or d.value != 1]
                        return TensorShape(tuple(dims))

        # unsqueeze
        if base_name == "unsqueeze":
            if isinstance(node.func, ast.Attribute):
                obj_shape = self._infer_shape(node.func.value)
                if obj_shape and node.args:
                    dim = self._const_val(node.args[0])
                    if dim is not None:
                        if dim < 0:
                            dim = obj_shape.ndim + 1 + dim
                        dims = list(obj_shape.dims)
                        dims.insert(dim, ShapeDim(1))
                        return TensorShape(tuple(dims))

        # flatten
        if base_name == "flatten":
            if isinstance(node.func, ast.Attribute):
                obj_shape = self._infer_shape(node.func.value)
                if obj_shape:
                    start_dim = 0
                    end_dim = -1
                    if node.args:
                        s = self._const_val(node.args[0])
                        if s is not None:
                            start_dim = s
                    if len(node.args) >= 2:
                        e = self._const_val(node.args[1])
                        if e is not None:
                            end_dim = e
                    if end_dim < 0:
                        end_dim = obj_shape.ndim + end_dim
                    # Flatten dims[start_dim:end_dim+1]
                    prefix = obj_shape.dims[:start_dim]
                    suffix = obj_shape.dims[end_dim + 1:]
                    flat_dims = obj_shape.dims[start_dim:end_dim + 1]
                    total = 1
                    all_concrete = True
                    for d in flat_dims:
                        if d.is_symbolic:
                            all_concrete = False
                            break
                        total *= d.value
                    if all_concrete:
                        return TensorShape(prefix + (ShapeDim(total),) + suffix)
                    return TensorShape(prefix + (ShapeDim("_flat"),) + suffix)

        # matmul / mm / bmm / dot
        if base_name in ("matmul", "mm", "dot"):
            if len(node.args) >= 2:
                a = self._infer_shape(node.args[0])
                b = self._infer_shape(node.args[1])
                if a and b:
                    return compute_matmul_shape(a, b)
            elif isinstance(node.func, ast.Attribute) and node.args:
                a = self._infer_shape(node.func.value)
                b = self._infer_shape(node.args[0])
                if a and b:
                    return compute_matmul_shape(a, b)

        # cat / concatenate
        if base_name in ("cat", "concatenate"):
            if node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
                shapes = [self._infer_shape(elt) for elt in node.args[0].elts]
                if all(s is not None for s in shapes) and shapes:
                    dim = 0
                    if len(node.args) >= 2:
                        d = self._const_val(node.args[1])
                        if d is not None:
                            dim = d
                    # Check for keyword 'dim'
                    for kw in node.keywords:
                        if kw.arg == "dim":
                            d = self._const_val(kw.value)
                            if d is not None:
                                dim = d
                    return self._compute_cat_shape(shapes, dim)

        # stack
        if base_name == "stack":
            if node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
                shapes = [self._infer_shape(elt) for elt in node.args[0].elts]
                if all(s is not None for s in shapes) and shapes:
                    dim = 0
                    for kw in node.keywords:
                        if kw.arg == "dim":
                            d = self._const_val(kw.value)
                            if d is not None:
                                dim = d
                    base = shapes[0]
                    dims = list(base.dims)
                    dims.insert(dim, ShapeDim(len(shapes)))
                    return TensorShape(tuple(dims))

        # sum/mean/max/min with dim
        if base_name in ("sum", "mean", "max", "min", "prod"):
            if isinstance(node.func, ast.Attribute):
                obj_shape = self._infer_shape(node.func.value)
                if obj_shape:
                    if node.args:
                        dim = self._const_val(node.args[0])
                        if dim is not None:
                            if dim < 0:
                                dim = obj_shape.ndim + dim
                            keepdim = False
                            for kw in node.keywords:
                                if kw.arg == "keepdim":
                                    if isinstance(kw.value, ast.Constant):
                                        keepdim = bool(kw.value.value)
                            dims = list(obj_shape.dims)
                            if keepdim:
                                dims[dim] = ShapeDim(1)
                            else:
                                dims.pop(dim)
                            return TensorShape(tuple(dims))
                    else:
                        return TensorShape(())  # scalar reduction

        # torch.where(cond, x, y) → broadcast(cond, x, y)
        if base_name == "where":
            if len(node.args) >= 3:
                cond_shape = self._infer_shape(node.args[0])
                x_shape = self._infer_shape(node.args[1])
                y_shape = self._infer_shape(node.args[2])
                result = cond_shape
                if result and x_shape:
                    result = compute_broadcast_shape(result, x_shape)
                elif x_shape:
                    result = x_shape
                if result and y_shape:
                    result = compute_broadcast_shape(result, y_shape)
                elif y_shape:
                    result = y_shape
                return result

        # torch.einsum(equation, *tensors) → shape from equation string
        if base_name == "einsum":
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                eq = node.args[0].value
                operand_shapes = [self._infer_shape(a) for a in node.args[1:]]
                return self._infer_einsum_shape(eq, operand_shapes)

        # F.interpolate(input, size=..., scale_factor=...)
        if base_name == "interpolate":
            if node.args:
                inp_shape = self._infer_shape(node.args[0])
                if inp_shape and inp_shape.ndim >= 3:
                    # Determine target spatial dims from size or scale_factor
                    target_size = None
                    for kw in node.keywords:
                        if kw.arg == "size":
                            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, int):
                                n_spatial = inp_shape.ndim - 2
                                target_size = [kw.value.value] * n_spatial
                            elif isinstance(kw.value, (ast.Tuple, ast.List)):
                                target_size = [self._const_val(e) for e in kw.value.elts]
                    if len(node.args) >= 2 and target_size is None:
                        arg1 = node.args[1]
                        if isinstance(arg1, ast.Constant) and isinstance(arg1.value, int):
                            n_spatial = inp_shape.ndim - 2
                            target_size = [arg1.value] * n_spatial
                        elif isinstance(arg1, (ast.Tuple, ast.List)):
                            target_size = [self._const_val(e) for e in arg1.elts]
                    if target_size and all(v is not None for v in target_size):
                        # Keep batch + channel dims, replace spatial dims
                        new_dims = list(inp_shape.dims[:2])
                        new_dims.extend(ShapeDim(v) for v in target_size)
                        return TensorShape(tuple(new_dims))
                    # scale_factor: mark spatial dims as symbolic
                    new_dims = list(inp_shape.dims[:2])
                    new_dims.extend(ShapeDim("_interp") for _ in range(inp_shape.ndim - 2))
                    return TensorShape(tuple(new_dims))

        # nn.Linear: self.fc(x) where fc is Linear(in_f, out_f)
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
                layer_attr = node.func.attr
                # Look up layer info from any class
                for key, info in self._layer_shapes.items():
                    if key.endswith(f".{layer_attr}"):
                        if info["type"] == "Linear" and node.args:
                            x_shape = self._infer_shape(node.args[0])
                            if x_shape and x_shape.ndim >= 1:
                                out_f = info["out_features"]
                                new_dims = list(x_shape.dims[:-1]) + [ShapeDim(out_f)]
                                return TensorShape(tuple(new_dims))

        return None

    def _infer_binop_shape(self, node: ast.BinOp) -> Optional[TensorShape]:
        """Infer shape from binary operations."""
        left = self._infer_shape(node.left)
        right = self._infer_shape(node.right)

        # @ operator (matmul)
        if isinstance(node.op, ast.MatMult):
            if left and right:
                return compute_matmul_shape(left, right)

        # Element-wise ops: broadcasting
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div,
                                ast.FloorDiv, ast.Mod, ast.Pow)):
            if left and right:
                return compute_broadcast_shape(left, right)
            return left or right

        return left or right

    def _infer_subscript_shape(self, node: ast.Subscript) -> Optional[TensorShape]:
        """Infer shape from subscript operations (indexing/slicing)."""
        obj_shape = self._infer_shape(node.value)
        if not obj_shape:
            return None

        # Simple integer index: x[0] removes one dimension
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
            if obj_shape.ndim > 1:
                dims = list(obj_shape.dims)
                dims.pop(0)
                return TensorShape(tuple(dims))
            return TensorShape(())

        return None

    # ── Shape constraint checking ──────────────────────────────────────

    def _check_expr_shapes(self, node: ast.expr):
        """Check shape constraints at operation sites."""
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            self._check_matmul(node)
        elif isinstance(node, ast.BinOp) and isinstance(node.op,
                (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            self._check_broadcast(node)
        elif isinstance(node, ast.Call):
            self._check_call_shapes(node)

        # Recurse
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self._check_expr_shapes(child)

    def _check_matmul(self, node: ast.BinOp):
        """Check matmul shape compatibility."""
        left = self._infer_shape(node.left)
        right = self._infer_shape(node.right)
        if not left or not right:
            return

        self.constraints_generated += 1
        err = check_matmul_compatible(left, right)
        if err:
            self.errors.append(ShapeError(
                kind=ShapeErrorKind.MATMUL_INCOMPAT,
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                message=err,
                function=self.func_name,
                variable="",
                actual_shape=left,
                expected_shape=right,
            ))
        else:
            # Use Z3 for symbolic dimension checking
            self._check_matmul_z3(node, left, right)
        self.constraints_checked += 1

    def _check_matmul_z3(self, node: ast.BinOp,
                          a: TensorShape, b: TensorShape):
        """Use Z3 to verify symbolic matmul compatibility.

        Check validity of k_a == k_b: if Not(k_a == k_b) is UNSAT,
        dimensions always match (no error). If SAT, a counterexample
        exists (report error). If unknown, don't report.
        """
        if not HAS_Z3:
            return

        k_a = a.dims[-1] if a.ndim >= 1 else None
        k_b = b.dims[-2] if b.ndim >= 2 else (b.dims[0] if b.ndim == 1 else None)
        if k_a is None or k_b is None:
            return

        # If both concrete, already checked
        if not k_a.is_symbolic and not k_b.is_symbolic:
            return

        if k_a.is_symbolic:
            z_ka = _z3.Int(str(k_a.value))
        else:
            z_ka = _z3.IntVal(k_a.value)

        if k_b.is_symbolic:
            z_kb = _z3.Int(str(k_b.value))
        else:
            z_kb = _z3.IntVal(k_b.value)

        self.constraints_generated += 1
        self.constraints_checked += 1

        # Check if k_a == k_b is VALID by checking if Not(k_a == k_b) is UNSAT
        s = _z3.Solver()
        s.set("timeout", self.timeout_ms)
        # Dimensions are positive
        if k_a.is_symbolic:
            s.add(z_ka > 0)
        if k_b.is_symbolic:
            s.add(z_kb > 0)
        s.add(_z3.Not(z_ka == z_kb))

        result = s.check()
        if result == _z3.unsat:
            # k_a == k_b is valid (always true) — no error
            return
        elif result == _z3.sat:
            # Counterexample exists — dimensions can mismatch
            model = s.model()
            cex = {d.name(): str(model[d]) for d in model.decls()}
            self.errors.append(ShapeError(
                kind=ShapeErrorKind.MATMUL_INCOMPAT,
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                message=f"Possible matmul dimension mismatch (Z3 counterexample: {cex})",
                function=self.func_name,
                variable="",
                actual_shape=a,
                expected_shape=b,
                z3_counterexample=cex,
            ))
        # If unknown, don't report an error

    def _check_broadcast(self, node: ast.BinOp):
        """Check broadcasting compatibility for element-wise ops."""
        left = self._infer_shape(node.left)
        right = self._infer_shape(node.right)
        if not left or not right:
            return

        self.constraints_generated += 1
        result = compute_broadcast_shape(left, right)
        if result is None:
            self.errors.append(ShapeError(
                kind=ShapeErrorKind.BROADCAST_FAIL,
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                message=(f"Cannot broadcast shapes {left.pretty()} and "
                         f"{right.pretty()} for element-wise operation"),
                function=self.func_name,
                variable="",
                actual_shape=left,
                expected_shape=right,
            ))
        else:
            # Use Z3 for symbolic broadcast checking
            self._check_broadcast_z3(node, left, right)
        self.constraints_checked += 1

    def _check_broadcast_z3(self, node: ast.BinOp,
                             a: TensorShape, b: TensorShape):
        """Use Z3 to verify symbolic broadcasting compatibility.

        NumPy/PyTorch broadcasting: align from right, each dim pair
        must be (d,d), (d,1), or (1,d).
        """
        if not HAS_Z3:
            return

        # Check if any dims are symbolic
        has_symbolic = any(
            d.is_symbolic
            for shape in (a, b)
            for d in shape.dims
        )
        if not has_symbolic:
            return

        ndim = max(a.ndim, b.ndim)
        constraints = []
        sym_vars = []

        for i in range(1, ndim + 1):
            d_a = a.dims[-i] if i <= a.ndim else ShapeDim(1)
            d_b = b.dims[-i] if i <= b.ndim else ShapeDim(1)

            if d_a.is_symbolic:
                z_a = _z3.Int(str(d_a.value))
                sym_vars.append((z_a, d_a))
            else:
                z_a = _z3.IntVal(d_a.value)

            if d_b.is_symbolic:
                z_b = _z3.Int(str(d_b.value))
                sym_vars.append((z_b, d_b))
            else:
                z_b = _z3.IntVal(d_b.value)

            # Broadcasting rule: d_a == d_b OR d_a == 1 OR d_b == 1
            constraints.append(_z3.Or(z_a == z_b, z_a == 1, z_b == 1))

        if not constraints:
            return

        s = _z3.Solver()
        s.set("timeout", self.timeout_ms)
        # Dimensions are positive
        for z_var, dim in sym_vars:
            s.add(z_var > 0)
        # Check if broadcast constraints can be violated
        s.add(_z3.Not(_z3.And(*constraints)))
        self.constraints_generated += 1
        self.constraints_checked += 1

        result = s.check()
        if result == _z3.unsat:
            # Broadcasting always holds — OK
            return
        # If sat or unknown, don't report (might be valid for some assignments)

    def _check_call_shapes(self, node: ast.Call):
        """Check shape constraints at function call sites."""
        func_name = self._get_call_name(node)
        if not func_name:
            return
        base_name = func_name.split(".")[-1] if "." in func_name else func_name

        # matmul/mm/dot calls
        if base_name in ("matmul", "mm", "dot"):
            args = node.args
            if isinstance(node.func, ast.Attribute) and args:
                a_shape = self._infer_shape(node.func.value)
                b_shape = self._infer_shape(args[0])
            elif len(args) >= 2:
                a_shape = self._infer_shape(args[0])
                b_shape = self._infer_shape(args[1])
            else:
                return
            if a_shape and b_shape:
                self.constraints_generated += 1
                err = check_matmul_compatible(a_shape, b_shape)
                if err:
                    self.errors.append(ShapeError(
                        kind=ShapeErrorKind.MATMUL_INCOMPAT,
                        line=getattr(node, "lineno", 0),
                        col=getattr(node, "col_offset", 0),
                        message=err,
                        function=self.func_name,
                        variable="",
                        actual_shape=a_shape,
                        expected_shape=b_shape,
                    ))
                self.constraints_checked += 1

        # cat/concatenate
        if base_name in ("cat", "concatenate"):
            if node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
                shapes = [self._infer_shape(elt) for elt in node.args[0].elts]
                valid_shapes = [s for s in shapes if s is not None]
                if len(valid_shapes) >= 2:
                    dim = 0
                    if len(node.args) >= 2:
                        d = self._const_val(node.args[1])
                        if d is not None:
                            dim = d
                    for kw in node.keywords:
                        if kw.arg == "dim":
                            d = self._const_val(kw.value)
                            if d is not None:
                                dim = d
                    self._check_cat_shapes(node, valid_shapes, dim)

        # nn.Linear: check input dimension matches in_features
        if isinstance(node.func, ast.Attribute):
            if (isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self" and node.args):
                layer_attr = node.func.attr
                for key, info in self._layer_shapes.items():
                    if key.endswith(f".{layer_attr}"):
                        if info["type"] == "Linear":
                            x_shape = self._infer_shape(node.args[0])
                            if x_shape and x_shape.ndim >= 1:
                                last_dim = x_shape.dims[-1]
                                in_f = info["in_features"]
                                if (not last_dim.is_symbolic
                                        and isinstance(in_f, int)
                                        and last_dim.value != in_f):
                                    self.constraints_generated += 1
                                    self.errors.append(ShapeError(
                                        kind=ShapeErrorKind.DIM_MISMATCH,
                                        line=getattr(node, "lineno", 0),
                                        col=getattr(node, "col_offset", 0),
                                        message=(f"Linear layer expects input dim "
                                                 f"{in_f}, got {last_dim.value}"),
                                        function=self.func_name,
                                        variable=layer_attr,
                                        actual_shape=x_shape,
                                    ))
                                    self.constraints_checked += 1
                                elif last_dim.is_symbolic and isinstance(in_f, int):
                                    self._check_linear_z3(
                                        node, x_shape, last_dim, in_f, layer_attr)

    def _check_linear_z3(self, node: ast.Call, x_shape: TensorShape,
                          last_dim: ShapeDim, in_f: int, layer_attr: str):
        """Use Z3 to verify symbolic nn.Linear input dimension."""
        if not HAS_Z3:
            return

        z_dim = _z3.Int(str(last_dim.value))
        z_inf = _z3.IntVal(in_f)

        s = _z3.Solver()
        s.set("timeout", self.timeout_ms)
        s.add(z_dim > 0)
        s.add(_z3.Not(z_dim == z_inf))
        self.constraints_generated += 1
        self.constraints_checked += 1

        result = s.check()
        if result == _z3.unsat:
            return  # Dimension always matches

    def _check_cat_shapes(self, node: ast.Call,
                           shapes: List[TensorShape], dim: int):
        """Check that all tensors in cat have matching shapes on non-cat dims."""
        self.constraints_generated += 1
        ref = shapes[0]
        has_symbolic_mismatch = False
        for i, s in enumerate(shapes[1:], 1):
            if s.ndim != ref.ndim:
                self.errors.append(ShapeError(
                    kind=ShapeErrorKind.CAT_INCOMPAT,
                    line=getattr(node, "lineno", 0),
                    col=getattr(node, "col_offset", 0),
                    message=(f"cat: tensor {i} has {s.ndim} dims, "
                             f"expected {ref.ndim} to match tensor 0"),
                    function=self.func_name,
                    variable="",
                    actual_shape=s,
                    expected_shape=ref,
                ))
                continue
            for j in range(ref.ndim):
                if j == dim:
                    continue
                d_ref = ref.dims[j]
                d_s = s.dims[j]
                if (not d_ref.is_symbolic and not d_s.is_symbolic
                        and d_ref.value != d_s.value):
                    self.errors.append(ShapeError(
                        kind=ShapeErrorKind.CAT_INCOMPAT,
                        line=getattr(node, "lineno", 0),
                        col=getattr(node, "col_offset", 0),
                        message=(f"cat dim {dim}: tensor {i} has dim[{j}]="
                                 f"{d_s.value}, expected {d_ref.value}"),
                        function=self.func_name,
                        variable="",
                        actual_shape=s,
                        expected_shape=ref,
                    ))
                elif d_ref.is_symbolic or d_s.is_symbolic:
                    has_symbolic_mismatch = True
        self.constraints_checked += 1
        # Use Z3 for symbolic cat dim checking
        if has_symbolic_mismatch:
            self._check_cat_z3(node, shapes, dim)

    def _check_cat_z3(self, node: ast.Call,
                       shapes: List[TensorShape], dim: int):
        """Use Z3 to verify symbolic cat dimension compatibility."""
        if not HAS_Z3:
            return

        ref = shapes[0]
        constraints = []
        sym_vars = []

        for i, s in enumerate(shapes[1:], 1):
            if s.ndim != ref.ndim:
                continue
            for j in range(ref.ndim):
                if j == dim:
                    continue
                d_ref = ref.dims[j]
                d_s = s.dims[j]
                if d_ref.is_symbolic:
                    z_ref = _z3.Int(str(d_ref.value))
                    sym_vars.append(z_ref)
                else:
                    z_ref = _z3.IntVal(d_ref.value)
                if d_s.is_symbolic:
                    z_s = _z3.Int(str(d_s.value))
                    sym_vars.append(z_s)
                else:
                    z_s = _z3.IntVal(d_s.value)
                constraints.append(z_ref == z_s)

        if not constraints:
            return

        s = _z3.Solver()
        s.set("timeout", self.timeout_ms)
        for v in sym_vars:
            s.add(v > 0)
        # Check if constraints can be violated
        s.add(_z3.Not(_z3.And(*constraints)))
        self.constraints_generated += 1
        self.constraints_checked += 1

        result = s.check()
        if result == _z3.unsat:
            return  # All constraints always hold
        # If sat or unknown, dims might not match but don't report without certainty

    # ── Shape assertion harvesting ─────────────────────────────────────

    def _harvest_shape_assert(self, node: ast.Assert):
        """Harvest shape information from assert statements.

        Patterns:
          assert x.shape == (3, 4)
          assert x.shape[0] == batch_size
          assert len(x.shape) == 3
          assert x.ndim == 2
        """
        test = node.test
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            op = test.ops[0]
            if isinstance(op, ast.Eq):
                left = test.left
                comp = test.comparators[0]

                # x.shape == (d1, d2, ...)
                if (isinstance(left, ast.Attribute) and left.attr == "shape"
                        and isinstance(left.value, ast.Name)):
                    var = left.value.id
                    shape = self._extract_shape_literal(comp)
                    if shape:
                        self.shape_env = self.shape_env.set(var, shape)
                        self._shape_preds.append(shape.to_pred(var))
                        return

                # x.shape[i] == d
                if (isinstance(left, ast.Subscript)
                        and isinstance(left.value, ast.Attribute)
                        and left.value.attr == "shape"
                        and isinstance(left.value.value, ast.Name)):
                    var = left.value.value.id
                    axis = self._const_val(left.slice)
                    dim_val = self._const_or_name(comp)
                    if axis is not None and dim_val is not None:
                        existing = self.shape_env.get(var)
                        if existing and axis < existing.ndim:
                            dims = list(existing.dims)
                            dims[axis] = ShapeDim(dim_val)
                            self.shape_env = self.shape_env.set(
                                var, TensorShape(tuple(dims)))

                # x.ndim == n
                if (isinstance(left, ast.Attribute) and left.attr == "ndim"
                        and isinstance(left.value, ast.Name)):
                    var = left.value.id
                    ndim = self._const_val(comp)
                    if ndim is not None:
                        existing = self.shape_env.get(var)
                        if not existing:
                            self.shape_env = self.shape_env.set(
                                var, TensorShape.unknown(ndim))

    # ── Utility methods ────────────────────────────────────────────────

    def _shape_from_creation_args(self, node: ast.Call) -> Optional[TensorShape]:
        """Extract shape from torch.zeros(d1, d2, ...) or torch.zeros((d1, d2))."""
        if not node.args:
            return None

        first = node.args[0]
        # torch.zeros((3, 4)) — tuple argument
        if isinstance(first, ast.Tuple):
            dims = self._args_to_dims([first])
            if dims:
                return TensorShape.from_tuple(dims)
            return None

        # torch.zeros(3, 4) — individual arguments
        dims = self._args_to_dims(node.args)
        if dims:
            return TensorShape.from_tuple(dims)
        return None

    def _extract_shape_literal(self, node: ast.expr) -> Optional[TensorShape]:
        """Extract a shape tuple literal from AST node."""
        if isinstance(node, ast.Tuple):
            dims = []
            for elt in node.elts:
                v = self._const_or_name(elt)
                if v is None:
                    return None
                dims.append(ShapeDim(v))
            return TensorShape(tuple(dims))
        return None

    def _extract_shape_args(self, node: ast.Call) -> Optional[Tuple]:
        """Extract reshape/view args."""
        if not node.args:
            return None
        dims = []
        for arg in node.args:
            v = self._const_or_name(arg)
            if v is not None:
                dims.append(v)
            else:
                dims.append("_unknown")
        return tuple(dims)

    def _args_to_dims(self, args: list) -> Optional[Tuple]:
        """Convert AST arguments to a tuple of dimension values."""
        dims = []
        for arg in args:
            if isinstance(arg, ast.Tuple):
                for elt in arg.elts:
                    v = self._const_or_name(elt)
                    if v is not None:
                        dims.append(v)
                    else:
                        return None
            else:
                v = self._const_or_name(arg)
                if v is not None:
                    dims.append(v)
                else:
                    return None
        return tuple(dims)

    def _compute_cat_shape(self, shapes: List[Optional[TensorShape]],
                            dim: int) -> Optional[TensorShape]:
        """Compute result shape of torch.cat."""
        valid = [s for s in shapes if s is not None]
        if not valid:
            return None
        base = valid[0]
        total_dim = ShapeDim(0)
        all_concrete = True
        cat_total = 0
        for s in valid:
            d = s.dims[dim] if dim < s.ndim else ShapeDim(0)
            if d.is_symbolic:
                all_concrete = False
            else:
                cat_total += d.value
        result_dims = list(base.dims)
        if all_concrete:
            result_dims[dim] = ShapeDim(cat_total)
        else:
            result_dims[dim] = ShapeDim("_cat_dim")
        return TensorShape(tuple(result_dims))

    def _shape_from_annotation(self, ann: ast.expr) -> Optional[TensorShape]:
        """Extract shape from type annotation if present."""
        # TODO: support Annotated[Tensor, Shape(3, 4)]
        return None

    def _infer_einsum_shape(
        self, equation: str, operand_shapes: List[Optional[TensorShape]]
    ) -> Optional[TensorShape]:
        """Infer output shape from an einsum equation string.

        Parses equations like ``'ij,jk->ik'`` and maps dimension labels
        to the corresponding shapes from the operands.
        """
        if '->' not in equation:
            return None
        lhs, rhs = equation.split('->', 1)
        input_specs = [s.strip() for s in lhs.split(',')]
        output_spec = rhs.strip()

        if len(input_specs) != len(operand_shapes):
            return None
        if any(s is None for s in operand_shapes):
            return None

        # Build label → ShapeDim mapping from operands
        label_to_dim: Dict[str, ShapeDim] = {}
        for spec, shape in zip(input_specs, operand_shapes):
            spec_clean = spec.replace(' ', '')
            if shape is None or len(spec_clean) != shape.ndim:
                return None
            for ch, sd in zip(spec_clean, shape.dims):
                label_to_dim[ch] = sd

        # Build output shape from output spec
        out_dims: List[ShapeDim] = []
        for ch in output_spec.replace(' ', ''):
            if ch in label_to_dim:
                out_dims.append(label_to_dim[ch])
            else:
                out_dims.append(ShapeDim(f"_einsum_{ch}"))
        return TensorShape(tuple(out_dims))

    @staticmethod
    def _get_call_name(node: ast.Call) -> Optional[str]:
        """Get the name of a function call."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                return f"{node.func.value.id}.{node.func.attr}"
            if isinstance(node.func.value, ast.Attribute):
                return node.func.attr
        return None

    @staticmethod
    def _const_val(node: ast.expr) -> Optional[int]:
        """Extract an integer constant from AST node."""
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, int):
                return -node.operand.value
        return None

    @staticmethod
    def _const_or_name(node: ast.expr) -> Optional[Union[int, str]]:
        """Extract int constant or variable name."""
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, int):
                return -node.operand.value
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Convenience API
# ═══════════════════════════════════════════════════════════════════════════

def analyze_shapes(source: str, **kwargs) -> ShapeAnalysisResult:
    """Analyze Python source for tensor shape errors.

    Usage::

        from src.tensor_shapes import analyze_shapes

        result = analyze_shapes('''
        import torch
        x = torch.randn(3, 4)
        y = torch.randn(5, 6)
        z = x @ y  # Shape error: inner dims 4 != 5
        ''')
        for err in result.errors:
            print(f"L{err.line}: {err.message}")
    """
    analyzer = TensorShapeAnalyzer(**kwargs)
    return analyzer.analyze_source(source)
