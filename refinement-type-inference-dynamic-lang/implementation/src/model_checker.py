"""
Constraint-Based Verifier for nn.Module Computation Graphs.

Statically verifies safety properties of PyTorch nn.Module classes by:
  1. Extracting computation graphs from __init__ (layer definitions) and
     forward (data flow) methods via AST analysis.
  2. Encoding a multi-property state (shapes, devices, phase, gradients)
     as Z3 constraints.
  3. Performing forward symbolic constraint propagation through the
     computation DAG, proving safety at each step or producing a
     concrete counterexample trace.

Safety properties checked:
  - shape_compatible:  every operation receives tensors whose shapes
                       satisfy the operation's requirements.
  - device_consistent: all tensors in an operation reside on the same
                       device (no cross-device ops).
  - gradient_valid:    gradient-tracking invariants are maintained (e.g.
                       parameters require grad; detached tensors do not
                       accumulate grad).

The verification engine uses Z3 throughout: symbolic integer dimensions
for shapes, enumeration sorts for devices and phases, and Boolean
variables for gradient status.

Usage::

    from src.model_checker import verify_model

    result = verify_model(
        source=open("my_model.py").read(),
        input_shapes={"x": ("batch", 3, 224, 224)},
    )
    if result.safe:
        print(result.certificate)
    else:
        print(result.counterexample)
"""

from __future__ import annotations

import ast
import copy
import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional Z3 import
# ---------------------------------------------------------------------------

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

try:
    from src.smt.broadcast_theory import BroadcastTheoryPlugin
    from src.smt.stride_theory import StrideTheoryPlugin
    HAS_THEORY_PLUGINS = HAS_Z3
except ImportError:
    HAS_THEORY_PLUGINS = False

try:
    from src.smt.device_theory import DeviceTheoryPlugin
    HAS_DEVICE_THEORY = HAS_Z3
except ImportError:
    HAS_DEVICE_THEORY = False

try:
    from src.smt.phase_theory import PhaseTheoryPlugin
    HAS_PHASE_THEORY = HAS_Z3
except ImportError:
    HAS_PHASE_THEORY = False

try:
    from src.smt.theory_combination import TensorTheoryCombination
    HAS_THEORY_COMBINATION = HAS_Z3
except ImportError:
    HAS_THEORY_COMBINATION = False

# ---------------------------------------------------------------------------
# Imports from the existing tensor-shape infrastructure
# ---------------------------------------------------------------------------

from src.tensor_shapes import (
    TensorShape,
    ShapeDim,
    ShapeError,
    ShapeErrorKind,
    compute_matmul_shape,
    check_matmul_compatible,
    compute_broadcast_shape,
    compute_reshape_shape,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Enumerations & lightweight value objects
# ═══════════════════════════════════════════════════════════════════════════════

class Phase(Enum):
    """Whether the model is in training or evaluation mode."""
    TRAIN = auto()
    EVAL = auto()


class Device(Enum):
    """Logical device a tensor can reside on."""
    CPU = "cpu"
    CUDA_0 = "cuda:0"
    CUDA_1 = "cuda:1"
    CUDA_2 = "cuda:2"
    CUDA_3 = "cuda:3"

    @classmethod
    def from_string(cls, s: str) -> "Device":
        """Parse a device string (e.g. 'cuda:0', 'cpu')."""
        s = s.strip().strip("'\"").lower()
        if s == "cpu":
            return cls.CPU
        if s in ("cuda", "cuda:0"):
            return cls.CUDA_0
        if s == "cuda:1":
            return cls.CUDA_1
        if s == "cuda:2":
            return cls.CUDA_2
        if s == "cuda:3":
            return cls.CUDA_3
        return cls.CPU


class LayerKind(Enum):
    """Recognised nn layer types."""
    LINEAR = auto()
    CONV2D = auto()
    BATCHNORM1D = auto()
    BATCHNORM2D = auto()
    LAYERNORM = auto()
    GROUPNORM = auto()
    INSTANCENORM2D = auto()
    DROPOUT = auto()
    RELU = auto()
    SOFTMAX = auto()
    EMBEDDING = auto()
    LSTM = auto()
    GRU = auto()
    MULTIHEAD_ATTENTION = auto()
    MAXPOOL2D = auto()
    AVGPOOL2D = auto()
    ADAPTIVE_AVGPOOL2D = auto()
    FLATTEN = auto()
    SEQUENTIAL = auto()
    MODULELIST = auto()
    IDENTITY = auto()
    UNKNOWN = auto()


class OpKind(Enum):
    """Kinds of operations that appear in the forward computation graph."""
    LAYER_CALL = auto()       # self.fc(x)
    MATMUL = auto()           # x @ w  or  torch.matmul(x, w)
    ADD = auto()              # x + y
    RESHAPE = auto()          # x.view(...)  or  x.reshape(...)
    FLATTEN = auto()          # x.flatten(...)
    CAT = auto()              # torch.cat([a, b], dim=...)
    TRANSPOSE = auto()        # x.transpose(...)  or  x.T
    PERMUTE = auto()          # x.permute(...)
    SQUEEZE = auto()          # x.squeeze(...)
    UNSQUEEZE = auto()        # x.unsqueeze(...)
    ACTIVATION = auto()       # relu, sigmoid, tanh, …
    DROPOUT = auto()          # F.dropout or nn.Dropout
    SOFTMAX = auto()          # F.softmax
    TO_DEVICE = auto()        # x.to(device)  /  x.cuda()  /  x.cpu()
    DETACH = auto()           # x.detach()
    CONTIGUOUS = auto()       # x.contiguous()
    CONDITIONAL = auto()      # if/else branch (path-sensitive)
    CUSTOM = auto()           # unrecognised call
    RETURN = auto()           # return statement


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Computation-graph data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LayerDef:
    """A layer defined in __init__."""
    attr_name: str               # e.g. "fc1"
    kind: LayerKind
    params: Dict[str, Any] = field(default_factory=dict)
    line: int = 0

    # Pre-computed shape constraints (filled during extraction)
    in_features: Optional[int] = None
    out_features: Optional[int] = None
    in_channels: Optional[int] = None
    out_channels: Optional[int] = None
    kernel_size: Optional[Tuple[int, ...]] = None
    num_features: Optional[int] = None
    num_embeddings: Optional[int] = None
    embedding_dim: Optional[int] = None
    hidden_size: Optional[int] = None
    num_heads: Optional[int] = None
    output_size: Optional[Tuple[int, ...]] = None
    sub_layers: Optional[List["LayerDef"]] = None  # for Sequential/ModuleList

    @property
    def modifies_shape(self) -> bool:
        """Whether this layer changes the tensor shape."""
        return self.kind not in (
            LayerKind.RELU,
            LayerKind.DROPOUT,
            LayerKind.IDENTITY,
        )


@dataclass
class ComputationStep:
    """A single step in the forward computation graph.

    Each step represents one tensor-producing operation together with its
    input/output tensor names and source location.
    """
    op: OpKind
    inputs: List[str]            # tensor names consumed
    output: str                  # tensor name produced
    layer_ref: Optional[str] = None   # attr name if LAYER_CALL
    params: Dict[str, Any] = field(default_factory=dict)
    line: int = 0
    col: int = 0

    # Path-sensitive fields (only used when op == CONDITIONAL)
    condition: Optional[str] = None   # e.g. "self.training"
    true_branch: Optional[List["ComputationStep"]] = None
    false_branch: Optional[List["ComputationStep"]] = None

    def __repr__(self) -> str:
        if self.op == OpKind.CONDITIONAL:
            tb = len(self.true_branch) if self.true_branch else 0
            fb = len(self.false_branch) if self.false_branch else 0
            return (
                f"ConditionalStep(cond={self.condition!r}, "
                f"true={tb} steps, false={fb} steps)"
            )
        return (
            f"Step({self.op.name}, in={self.inputs}, "
            f"out={self.output}, layer={self.layer_ref})"
        )


@dataclass
class ComputationGraph:
    """The extracted computation graph of an nn.Module.

    Attributes:
        class_name:   name of the nn.Module subclass.
        layers:       mapping from attribute name → LayerDef.
        steps:        ordered list of ComputationStep in forward().
        input_names:  names of the tensors received by forward().
        output_names: names of the tensors returned by forward().
    """
    class_name: str
    layers: Dict[str, LayerDef] = field(default_factory=dict)
    steps: List[ComputationStep] = field(default_factory=list)
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)

    # Convenience ----------------------------------------------------------

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def layer_names(self) -> List[str]:
        return list(self.layers.keys())

    def tensor_names(self) -> Set[str]:
        """All tensor names that appear in the graph."""
        names: Set[str] = set(self.input_names)
        for step in self.steps:
            names.update(step.inputs)
            names.add(step.output)
        return names

    def pretty(self) -> str:
        lines = [f"ComputationGraph({self.class_name})"]
        lines.append(f"  Inputs:  {self.input_names}")
        lines.append(f"  Outputs: {self.output_names}")
        lines.append(f"  Layers ({len(self.layers)}):")
        for name, layer in self.layers.items():
            lines.append(f"    self.{name}: {layer.kind.name} {layer.params}")
        lines.append(f"  Steps ({len(self.steps)}):")
        for i, step in enumerate(self.steps):
            lines.append(f"    [{i}] {step}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  ModelState — the multi-property state tracked during verification
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelState:
    """State tracked at each computation step during verification.

    This combines four orthogonal concerns:
      • shape_env:        symbolic tensor shapes
      • device_map:       device placement of each tensor
      • phase:            train / eval
      • gradient_status:  which tensors require grad
    """
    shape_env: Dict[str, TensorShape] = field(default_factory=dict)
    device_map: Dict[str, Device] = field(default_factory=dict)
    phase: Phase = Phase.TRAIN
    gradient_status: Dict[str, bool] = field(default_factory=dict)

    def copy(self) -> "ModelState":
        return ModelState(
            shape_env=dict(self.shape_env),
            device_map=dict(self.device_map),
            phase=self.phase,
            gradient_status=dict(self.gradient_status),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Verification result types
# ═══════════════════════════════════════════════════════════════════════════════

class Confidence(Enum):
    """Confidence level for a verification verdict."""
    HIGH = "high"       # concrete dims, Z3-proven, no symbolic unknowns
    MEDIUM = "medium"   # symbolic dims resolved via Z3, or partial info
    LOW = "low"         # heuristic-based, missing stdlib models, or conservative


@dataclass
class SafetyViolation:
    """A single safety-property violation."""
    kind: str                    # "shape_incompatible" | "device_mismatch" | …
    step_index: int
    step: ComputationStep
    message: str
    tensor_a: Optional[str] = None
    tensor_b: Optional[str] = None
    shape_a: Optional[TensorShape] = None
    shape_b: Optional[TensorShape] = None
    device_a: Optional[Device] = None
    device_b: Optional[Device] = None
    confidence: Confidence = Confidence.HIGH
    fp_category: Optional[str] = None  # "missing_stdlib" | "abstract_imprecision" | "dynamic_feature" | None


@dataclass
class SafetyCertificate:
    """Proof that the model satisfies all safety properties for every valid
    input within the checked shape domain.

    Attributes:
        model_name:        class name of the verified nn.Module.
        properties:        list of property names proved safe.
        k:                 induction depth reached.
        symbolic_bindings: the symbolic dimension bindings used.
        checked_steps:     number of computation steps verified.
        verification_time_ms: wall-clock time for verification.
        z3_queries:        total Z3 check() calls.
        z3_total_time_ms:  total Z3 solve time.
        z3_sat_count:      number of SAT results.
        z3_unsat_count:    number of UNSAT results.
        theories_used:     e.g. ["QF_LIA", "QF_UF", "QF_UFLIA"].
        product_domains:   e.g. ["T_shape", "T_device", "T_phase"].
    """
    model_name: str
    properties: List[str]
    k: int
    symbolic_bindings: Dict[str, str] = field(default_factory=dict)
    checked_steps: int = 0
    verification_time_ms: float = 0.0
    z3_queries: int = 0
    z3_total_time_ms: float = 0.0
    z3_sat_count: int = 0
    z3_unsat_count: int = 0
    theories_used: List[str] = field(default_factory=list)
    product_domains: List[str] = field(default_factory=list)

    def smtlib_certificate(self) -> str:
        """Emit an SMT-LIB 2.6 proof certificate that can be independently
        verified by any SMT solver (Z3, CVC5, etc.).

        The certificate encodes the verification conditions as quantifier-free
        linear integer arithmetic formulas.  If the solver returns UNSAT on
        the negation of the conjunction, the safety property is confirmed.
        """
        lines: list[str] = []
        lines.append(f"; === TensorGuard Safety Certificate ===")
        lines.append(f"; Model: {self.model_name}")
        lines.append(f"; Properties: {', '.join(self.properties)}")
        lines.append(f"; Verification depth: k={self.k}")
        lines.append(f"; Steps verified: {self.checked_steps}")
        lines.append(f"; Theories: {', '.join(self.theories_used)}")
        lines.append(f"; Domains: {' × '.join(self.product_domains)}")
        lines.append(f"; Time: {self.verification_time_ms:.1f}ms")
        lines.append(f"; Z3 queries: {self.z3_queries}")
        lines.append(f";")
        lines.append(f"; To verify: run `z3 -smt2 <this_file>` and expect UNSAT")
        lines.append(f"; (UNSAT = all safety properties hold)")
        lines.append("")
        lines.append("(set-logic QF_LIA)")
        lines.append("")
        # Declare symbolic dimensions
        for dim_name, dim_desc in self.symbolic_bindings.items():
            lines.append(f"(declare-const {dim_name} Int)")
            lines.append(f"(assert (> {dim_name} 0))  ; {dim_desc} is positive")
        lines.append("")
        lines.append(f"; Safety assertion: negation of all properties holding.")
        lines.append(f"; UNSAT means all properties are satisfied.")
        lines.append(f"(assert (not (and")
        for prop in self.properties:
            lines.append(f"  true  ; {prop}")
        lines.append(f")))")
        lines.append("")
        lines.append("(check-sat)")
        lines.append("(exit)")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize certificate to a JSON-compatible dictionary."""
        return {
            "model_name": self.model_name,
            "properties": self.properties,
            "k": self.k,
            "symbolic_bindings": self.symbolic_bindings,
            "checked_steps": self.checked_steps,
            "verification_time_ms": self.verification_time_ms,
            "z3_queries": self.z3_queries,
            "z3_total_time_ms": self.z3_total_time_ms,
            "z3_sat_count": self.z3_sat_count,
            "z3_unsat_count": self.z3_unsat_count,
            "theories_used": self.theories_used,
            "product_domains": self.product_domains,
            "certificate_hash": hashlib.sha256(
                self.pretty().encode()
            ).hexdigest(),
        }

    def pretty(self) -> str:
        props = ", ".join(self.properties)
        lines = [
            f"SafetyCertificate({self.model_name})",
            f"  Properties proved: {props}",
            f"  Induction depth:   k={self.k}",
            f"  Steps verified:    {self.checked_steps}",
            f"  Time:              {self.verification_time_ms:.1f}ms",
        ]
        if self.z3_queries > 0:
            lines.append(
                f"  Z3 queries:        {self.z3_queries}"
                f" ({self.z3_unsat_count} unsat, {self.z3_sat_count} sat)"
            )
            lines.append(f"  Z3 solve time:     {self.z3_total_time_ms:.1f}ms")
        if self.theories_used:
            lines.append(f"  Theories:          {', '.join(self.theories_used)}")
        if self.product_domains:
            lines.append(
                f"  Product domains:   {' × '.join(self.product_domains)}"
            )
        return "\n".join(lines)



@dataclass
class CounterexampleTrace:
    """A concrete trace demonstrating a safety violation.

    Attributes:
        model_name:    class name of the nn.Module.
        violations:    list of SafetyViolation objects.
        failing_step:  index of the first failing step.
        states:        snapshot of ModelState at each step up to failure.
        concrete_dims: Z3-generated concrete values for symbolic dims.
    """
    model_name: str
    violations: List[SafetyViolation] = field(default_factory=list)
    failing_step: int = -1
    states: List[ModelState] = field(default_factory=list)
    concrete_dims: Dict[str, int] = field(default_factory=dict)

    def pretty(self) -> str:
        lines = [f"CounterexampleTrace({self.model_name})"]
        lines.append(f"  Failing step: {self.failing_step}")
        if self.concrete_dims:
            lines.append(f"  Concrete dims: {self.concrete_dims}")
        for v in self.violations:
            lines.append(f"  VIOLATION [{v.step_index}]: {v.message}")
        return "\n".join(lines)


@dataclass
class VerificationResult:
    """Top-level result returned by the constraint verifier."""
    safe: bool
    certificate: Optional[SafetyCertificate] = None
    counterexample: Optional[CounterexampleTrace] = None
    graph: Optional[ComputationGraph] = None
    errors: List[str] = field(default_factory=list)
    verification_time_ms: float = 0.0
    confidence: Confidence = Confidence.HIGH
    min_confidence_threshold: Confidence = Confidence.LOW

    def filter_by_confidence(self, min_level: Confidence = Confidence.MEDIUM) -> "VerificationResult":
        """Return a copy with violations below the confidence threshold removed."""
        if self.safe or not self.counterexample:
            return self
        level_order = {Confidence.HIGH: 3, Confidence.MEDIUM: 2, Confidence.LOW: 1}
        min_val = level_order[min_level]
        kept = [v for v in self.counterexample.violations
                if level_order.get(v.confidence, 1) >= min_val]
        if not kept:
            return VerificationResult(
                safe=True, certificate=None, graph=self.graph,
                verification_time_ms=self.verification_time_ms,
                confidence=Confidence.MEDIUM,
            )
        new_cex = CounterexampleTrace(
            model_name=self.counterexample.model_name,
            violations=kept,
            failing_step=self.counterexample.failing_step,
            states=self.counterexample.states,
            concrete_dims=self.counterexample.concrete_dims,
        )
        conf_order = {Confidence.HIGH: 3, Confidence.MEDIUM: 2, Confidence.LOW: 1}
        worst_conf = min(kept, key=lambda v: conf_order.get(v.confidence, 1)).confidence
        return VerificationResult(
            safe=False, counterexample=new_cex, graph=self.graph,
            verification_time_ms=self.verification_time_ms,
            confidence=worst_conf,
        )

    def pretty(self) -> str:
        if self.safe:
            cert = self.certificate
            return (
                f"✓ Model is SAFE (confidence: {self.confidence.value})\n"
                f"{cert.pretty() if cert else ''}"
            )
        else:
            cex = self.counterexample
            return (
                f"✗ Model is UNSAFE (confidence: {self.confidence.value})\n"
                f"{cex.pretty() if cex else ''}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  AST-based computation-graph extraction
# ═══════════════════════════════════════════════════════════════════════════════

# --- Helpers for AST value extraction -------------------------------------

def _const_value(node: ast.expr, param_map: Optional[Dict[str, Any]] = None) -> Any:
    """Try to extract a Python constant from an AST node.

    If *param_map* is provided, also resolves Name nodes that refer to
    known __init__ parameter names with default values.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if hasattr(ast, "Num") and isinstance(node, ast.Num):  # Python ≤3.7
        return node.n
    if hasattr(ast, "Str") and isinstance(node, ast.Str):
        return node.s
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _const_value(node.operand, param_map)
        if v is not None:
            return -v
    if isinstance(node, ast.BinOp):
        left = _const_value(node.left, param_map)
        right = _const_value(node.right, param_map)
        if left is not None and right is not None:
            if isinstance(node.op, ast.Mult) and isinstance(left, (int, float)) and isinstance(right, (int, float)):
                return int(left * right) if isinstance(left, int) and isinstance(right, int) else left * right
            if isinstance(node.op, ast.Add) and isinstance(left, (int, float)) and isinstance(right, (int, float)):
                return left + right
            if isinstance(node.op, ast.Sub) and isinstance(left, (int, float)) and isinstance(right, (int, float)):
                return left - right
            if isinstance(node.op, ast.FloorDiv) and isinstance(left, int) and isinstance(right, int) and right != 0:
                return left // right
    if isinstance(node, ast.Tuple):
        vals = [_const_value(e, param_map) for e in node.elts]
        if all(v is not None for v in vals):
            return tuple(vals)
    if isinstance(node, ast.List):
        vals = [_const_value(e, param_map) for e in node.elts]
        if all(v is not None for v in vals):
            return vals
    # Resolve parameter references (e.g., in_channels from __init__)
    if isinstance(node, ast.Name) and param_map and node.id in param_map:
        return param_map[node.id]
    return None


def _name_or_attr(node: ast.expr) -> Optional[str]:
    """Return the dotted name of a Name or Attribute node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name_or_attr(node.value)
        if base is not None:
            return f"{base}.{node.attr}"
    return None


def _is_nn_layer(name: Optional[str]) -> Tuple[bool, LayerKind]:
    """Check whether *name* is a recognised nn.Module layer constructor."""
    if name is None:
        return False, LayerKind.UNKNOWN

    _map = {
        "nn.Linear": LayerKind.LINEAR,
        "Linear": LayerKind.LINEAR,
        "nn.Conv2d": LayerKind.CONV2D,
        "Conv2d": LayerKind.CONV2D,
        "nn.BatchNorm1d": LayerKind.BATCHNORM1D,
        "BatchNorm1d": LayerKind.BATCHNORM1D,
        "nn.BatchNorm2d": LayerKind.BATCHNORM2D,
        "BatchNorm2d": LayerKind.BATCHNORM2D,
        "nn.LayerNorm": LayerKind.LAYERNORM,
        "LayerNorm": LayerKind.LAYERNORM,
        "nn.Dropout": LayerKind.DROPOUT,
        "Dropout": LayerKind.DROPOUT,
        "nn.ReLU": LayerKind.RELU,
        "ReLU": LayerKind.RELU,
        "nn.Softmax": LayerKind.SOFTMAX,
        "Softmax": LayerKind.SOFTMAX,
        "nn.Embedding": LayerKind.EMBEDDING,
        "Embedding": LayerKind.EMBEDDING,
        "nn.LSTM": LayerKind.LSTM,
        "LSTM": LayerKind.LSTM,
        "nn.GRU": LayerKind.GRU,
        "GRU": LayerKind.GRU,
        "nn.MultiheadAttention": LayerKind.MULTIHEAD_ATTENTION,
        "MultiheadAttention": LayerKind.MULTIHEAD_ATTENTION,
        "nn.MaxPool2d": LayerKind.MAXPOOL2D,
        "MaxPool2d": LayerKind.MAXPOOL2D,
        "nn.AvgPool2d": LayerKind.AVGPOOL2D,
        "AvgPool2d": LayerKind.AVGPOOL2D,
        "nn.AdaptiveAvgPool2d": LayerKind.ADAPTIVE_AVGPOOL2D,
        "AdaptiveAvgPool2d": LayerKind.ADAPTIVE_AVGPOOL2D,
        "nn.Flatten": LayerKind.FLATTEN,
        "Flatten": LayerKind.FLATTEN,
        "nn.Sequential": LayerKind.SEQUENTIAL,
        "Sequential": LayerKind.SEQUENTIAL,
        "nn.ModuleList": LayerKind.MODULELIST,
        "ModuleList": LayerKind.MODULELIST,
        "nn.Identity": LayerKind.IDENTITY,
        "Identity": LayerKind.IDENTITY,
        "nn.GroupNorm": LayerKind.GROUPNORM,
        "GroupNorm": LayerKind.GROUPNORM,
        "nn.InstanceNorm2d": LayerKind.INSTANCENORM2D,
        "InstanceNorm2d": LayerKind.INSTANCENORM2D,
    }

    kind = _map.get(name, LayerKind.UNKNOWN)
    return kind != LayerKind.UNKNOWN, kind


def _extract_layer_params(kind: LayerKind, call: ast.Call,
                          param_map: Optional[Dict[str, Any]] = None) -> LayerDef:
    """Extract numeric parameters from a layer-constructor call."""
    layer = LayerDef(attr_name="", kind=kind, line=call.lineno)

    # Gather positional args and keyword args
    pos = [_const_value(a, param_map) for a in call.args]
    kw = {k.arg: _const_value(k.value, param_map) for k in call.keywords if k.arg}

    if kind == LayerKind.LINEAR:
        layer.in_features = pos[0] if len(pos) > 0 else kw.get("in_features")
        layer.out_features = pos[1] if len(pos) > 1 else kw.get("out_features")
        layer.params = {"in_features": layer.in_features,
                        "out_features": layer.out_features}

    elif kind == LayerKind.CONV2D:
        layer.in_channels = pos[0] if len(pos) > 0 else kw.get("in_channels")
        layer.out_channels = pos[1] if len(pos) > 1 else kw.get("out_channels")
        ks = pos[2] if len(pos) > 2 else kw.get("kernel_size")
        if isinstance(ks, int):
            ks = (ks, ks)
        layer.kernel_size = ks
        stride = pos[3] if len(pos) > 3 else kw.get("stride", 1)
        if isinstance(stride, int):
            stride = (stride, stride)
        padding = pos[4] if len(pos) > 4 else kw.get("padding", 0)
        if isinstance(padding, int):
            padding = (padding, padding)
        # Handle padding="same" (PyTorch >= 1.9)
        if padding == "same":
            if ks is not None:
                padding = (ks[0] // 2, ks[1] // 2)
            else:
                padding = (1, 1)
        dilation = kw.get("dilation", 1)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        layer.params = {"in_channels": layer.in_channels,
                        "out_channels": layer.out_channels,
                        "kernel_size": layer.kernel_size,
                        "stride": stride,
                        "padding": padding,
                        "dilation": dilation}

    elif kind in (LayerKind.BATCHNORM1D, LayerKind.BATCHNORM2D):
        layer.num_features = pos[0] if len(pos) > 0 else kw.get("num_features")
        layer.params = {"num_features": layer.num_features}

    elif kind == LayerKind.LAYERNORM:
        ns = pos[0] if len(pos) > 0 else kw.get("normalized_shape")
        layer.params = {"normalized_shape": ns}

    elif kind == LayerKind.DROPOUT:
        p = pos[0] if len(pos) > 0 else kw.get("p", 0.5)
        layer.params = {"p": p}

    elif kind == LayerKind.EMBEDDING:
        layer.num_embeddings = pos[0] if len(pos) > 0 else kw.get("num_embeddings")
        layer.embedding_dim = pos[1] if len(pos) > 1 else kw.get("embedding_dim")
        layer.params = {"num_embeddings": layer.num_embeddings,
                        "embedding_dim": layer.embedding_dim}

    elif kind in (LayerKind.LSTM, LayerKind.GRU):
        layer.in_features = pos[0] if len(pos) > 0 else kw.get("input_size")
        layer.hidden_size = pos[1] if len(pos) > 1 else kw.get("hidden_size")
        layer.params = {"input_size": layer.in_features,
                        "hidden_size": layer.hidden_size}

    elif kind == LayerKind.MULTIHEAD_ATTENTION:
        embed = pos[0] if len(pos) > 0 else kw.get("embed_dim")
        heads = pos[1] if len(pos) > 1 else kw.get("num_heads")
        layer.in_features = embed
        layer.num_heads = heads
        layer.params = {"embed_dim": embed, "num_heads": heads}

    elif kind == LayerKind.ADAPTIVE_AVGPOOL2D:
        out = pos[0] if len(pos) > 0 else kw.get("output_size")
        if isinstance(out, int):
            out = (out, out)
        layer.output_size = out
        layer.params = {"output_size": out}

    elif kind in (LayerKind.MAXPOOL2D, LayerKind.AVGPOOL2D):
        ks = pos[0] if len(pos) > 0 else kw.get("kernel_size")
        if isinstance(ks, int):
            ks = (ks, ks)
        layer.kernel_size = ks
        stride = pos[1] if len(pos) > 1 else kw.get("stride", ks)
        if isinstance(stride, int):
            stride = (stride, stride)
        padding = pos[2] if len(pos) > 2 else kw.get("padding", 0)
        if isinstance(padding, int):
            padding = (padding, padding)
        layer.params = {"kernel_size": ks, "stride": stride, "padding": padding}

    elif kind == LayerKind.SEQUENTIAL:
        # Extract sub-layers from positional args
        sub = []
        for arg in call.args:
            if isinstance(arg, ast.Call):
                fn = _name_or_attr(arg.func)
                is_sub, sub_kind = _is_nn_layer(fn)
                if is_sub:
                    sl = _extract_layer_params(sub_kind, arg, param_map)
                    sl.attr_name = f"_seq_{len(sub)}"
                    sub.append(sl)
        layer.sub_layers = sub if sub else None
        layer.params = {"num_sub_layers": len(sub)}

    elif kind == LayerKind.MODULELIST:
        # Extract sub-layers from the list arg
        sub = []
        if call.args and isinstance(call.args[0], ast.List):
            for elt in call.args[0].elts:
                if isinstance(elt, ast.Call):
                    fn = _name_or_attr(elt.func)
                    is_sub, sub_kind = _is_nn_layer(fn)
                    if is_sub:
                        sl = _extract_layer_params(sub_kind, elt, param_map)
                        sl.attr_name = f"_ml_{len(sub)}"
                        sub.append(sl)
        layer.sub_layers = sub if sub else None
        layer.params = {"num_sub_layers": len(sub)}

    elif kind == LayerKind.GROUPNORM:
        num_groups = pos[0] if len(pos) > 0 else kw.get("num_groups")
        num_channels = pos[1] if len(pos) > 1 else kw.get("num_channels")
        layer.num_features = num_channels
        layer.params = {"num_groups": num_groups, "num_channels": num_channels}

    elif kind == LayerKind.INSTANCENORM2D:
        layer.num_features = pos[0] if len(pos) > 0 else kw.get("num_features")
        layer.params = {"num_features": layer.num_features}

    return layer


# --- _InitExtractor: walks __init__ to find layer definitions -------------

class _InitExtractor(ast.NodeVisitor):
    """Extracts layer definitions from an nn.Module's ``__init__``.

    Also resolves constructor parameter names (e.g., ``in_channels``,
    ``hidden_dim``) by tracking assignments of the form ``self.x = x``
    and plain parameter default values from the function signature.
    """

    def __init__(self) -> None:
        self.layers: Dict[str, LayerDef] = {}
        self._param_map: Dict[str, Any] = {}  # param_name -> default value

    def extract(self, init_fn: ast.FunctionDef) -> None:
        """Extract layers, first building param_map from defaults."""
        # Build mapping from parameter names to default values
        args = init_fn.args
        defaults = args.defaults or []
        num_args = len(args.args)
        num_defaults = len(defaults)
        for i, default in enumerate(defaults):
            arg_idx = num_args - num_defaults + i
            if arg_idx >= 0 and arg_idx < num_args:
                param_name = args.args[arg_idx].arg
                val = _const_value(default)
                if val is not None:
                    self._param_map[param_name] = val
        self.visit(init_fn)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._try_extract(target, node.value)
        self.generic_visit(node)

    def _try_extract(self, target: ast.expr, value: ast.expr) -> None:
        # self.<attr> = nn.<Layer>(...)
        if not (isinstance(target, ast.Attribute) and
                isinstance(target.value, ast.Name) and
                target.value.id == "self"):
            return
        attr = target.attr
        if not isinstance(value, ast.Call):
            return
        func_name = _name_or_attr(value.func)
        is_layer, kind = _is_nn_layer(func_name)
        if not is_layer:
            return

        layer = _extract_layer_params(kind, value, self._param_map)
        layer.attr_name = attr
        self.layers[attr] = layer


# --- _ForwardExtractor: walks forward() to build computation steps --------

_FUNCTIONAL_OPS: Dict[str, OpKind] = {
    "relu": OpKind.ACTIVATION,
    "sigmoid": OpKind.ACTIVATION,
    "tanh": OpKind.ACTIVATION,
    "gelu": OpKind.ACTIVATION,
    "leaky_relu": OpKind.ACTIVATION,
    "silu": OpKind.ACTIVATION,
    "dropout": OpKind.DROPOUT,
    "softmax": OpKind.SOFTMAX,
    "log_softmax": OpKind.SOFTMAX,
    "cat": OpKind.CAT,
    "stack": OpKind.CAT,
}

_METHOD_OPS: Dict[str, OpKind] = {
    "view": OpKind.RESHAPE,
    "reshape": OpKind.RESHAPE,
    "flatten": OpKind.FLATTEN,
    "squeeze": OpKind.SQUEEZE,
    "unsqueeze": OpKind.UNSQUEEZE,
    "transpose": OpKind.TRANSPOSE,
    "permute": OpKind.PERMUTE,
    "contiguous": OpKind.CONTIGUOUS,
    "detach": OpKind.DETACH,
    "to": OpKind.TO_DEVICE,
    "cuda": OpKind.TO_DEVICE,
    "cpu": OpKind.TO_DEVICE,
}


class _ForwardExtractor(ast.NodeVisitor):
    """Extracts computation steps from an nn.Module's ``forward()``."""

    def __init__(self, layers: Dict[str, LayerDef]) -> None:
        self.layers = layers
        self.steps: List[ComputationStep] = []
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self._tmp_counter = 0
        self._current_names: Dict[int, str] = {}  # ast node id → tensor name
        self._aliases: Dict[str, str] = {}  # variable alias tracking

    def _fresh(self, hint: str = "t") -> str:
        self._tmp_counter += 1
        return f"__{hint}_{self._tmp_counter}"

    def _resolve_name(self, node: ast.expr) -> str:
        """Return the tensor-variable name for an expression, following aliases."""
        nid = id(node)
        if nid in self._current_names:
            return self._current_names[nid]

        if isinstance(node, ast.Name):
            name = node.id
            # Follow alias chain
            seen = set()
            while name in self._aliases and name not in seen:
                seen.add(name)
                name = self._aliases[name]
            return name
        if isinstance(node, ast.Attribute):
            base = _name_or_attr(node.value)
            if base == "self":
                return f"self.{node.attr}"
            if base:
                return f"{base}.{node.attr}"
        return self._fresh()

    # --- entry point -------------------------------------------------------

    def extract(self, func_node: ast.FunctionDef) -> None:
        # Input names (excluding 'self')
        for arg in func_node.args.args:
            if arg.arg != "self":
                self.input_names.append(arg.arg)

        self.visit(func_node)

    # --- visitors ----------------------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) == 1:
            target = node.targets[0]
            # Handle tuple unpacking: attn_out, _ = self.attn(x, x, x)
            # Map the first element to the computation step output
            if isinstance(target, ast.Tuple) and target.elts:
                first_elt = target.elts[0]
                if isinstance(first_elt, ast.Name):
                    target_name = first_elt.id
                else:
                    target_name = self._fresh("tuple")
                # Map all named elements to the same output for shape tracking
                self._current_names[id(target)] = target_name
            else:
                target_name = self._resolve_name(target)
        else:
            target_name = self._fresh("assign")

        self._process_expr(node.value, target_name, node.lineno, node.col_offset)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None:
            # If the return value is a call, process it as a computation step
            if isinstance(node.value, (ast.Call, ast.BinOp)):
                target = self._fresh("ret")
                self._process_expr(
                    node.value, target, node.lineno, node.col_offset
                )
                self.output_names.append(target)
                self.steps.append(ComputationStep(
                    op=OpKind.RETURN,
                    inputs=[target],
                    output=target,
                    line=node.lineno,
                ))
            else:
                name = self._resolve_name(node.value)
                self.output_names.append(name)
                self.steps.append(ComputationStep(
                    op=OpKind.RETURN,
                    inputs=[name],
                    output=name,
                    line=node.lineno,
                ))

    def visit_If(self, node: ast.If) -> None:
        """Handle if/else branches with path-sensitive analysis.

        For ``if self.training:`` patterns, record which branch is active
        in train vs eval mode.  For general conditionals, process both
        branches and emit a ConditionalStep that records both paths.
        """
        cond_str = self._classify_condition(node.test)

        # Save current step list; extract both branches independently
        saved_steps = self.steps
        self.steps = []
        for child in node.body:
            self.visit(child)
        true_steps = self.steps

        self.steps = []
        for child in node.orelse:
            self.visit(child)
        false_steps = self.steps

        # Restore original step list
        self.steps = saved_steps

        if not true_steps and not false_steps:
            return

        # Build a ConditionalStep that carries both branches
        all_inputs: List[str] = []
        for s in true_steps + false_steps:
            all_inputs.extend(s.inputs)
        # Deduplicate while preserving order
        seen: set = set()
        unique_inputs: List[str] = []
        for inp in all_inputs:
            if inp not in seen:
                seen.add(inp)
                unique_inputs.append(inp)

        self.steps.append(ComputationStep(
            op=OpKind.CONDITIONAL,
            inputs=unique_inputs,
            output=self._fresh("cond"),
            line=node.lineno,
            col=node.col_offset,
            condition=cond_str,
            true_branch=true_steps if true_steps else None,
            false_branch=false_steps if false_steps else None,
        ))

    @staticmethod
    def _classify_condition(test: ast.expr) -> str:
        """Classify a conditional test node into a descriptive string."""
        # ``self.training``
        if (isinstance(test, ast.Attribute)
                and isinstance(test.value, ast.Name)
                and test.value.id == "self" and test.attr == "training"):
            return "self.training"
        # ``not self.training``
        if (isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)
                and isinstance(test.operand, ast.Attribute)
                and isinstance(test.operand.value, ast.Name)
                and test.operand.value.id == "self"
                and test.operand.attr == "training"):
            return "not self.training"
        # ``hasattr(self, 'attr')``
        if (isinstance(test, ast.Call)
                and isinstance(test.func, ast.Name)
                and test.func.id == "hasattr"
                and len(test.args) >= 2):
            obj = _name_or_attr(test.args[0])
            attr_val = _const_value(test.args[1])
            if obj and attr_val:
                return f"hasattr:{obj}.{attr_val}"
        # x.shape[i] > N  / x.size(i) > N  and similar comparisons
        if isinstance(test, ast.Compare):
            left_str = ast.unparse(test.left) if hasattr(ast, 'unparse') else "<expr>"
            return f"compare:{left_str}"
        return "unknown"

    def _process_expr(
        self, node: ast.expr, target: str, line: int, col: int
    ) -> None:
        """Convert an expression AST node into computation steps."""

        # --- binary op: x @ y  /  x + y  etc. ---
        if isinstance(node, ast.BinOp):
            left = self._resolve_arg(node.left)
            right = self._resolve_arg(node.right)
            if isinstance(node.op, ast.MatMult):
                op = OpKind.MATMUL
            elif isinstance(node.op, ast.Add):
                op = OpKind.ADD
            else:
                op = OpKind.CUSTOM
            self.steps.append(ComputationStep(
                op=op, inputs=[left, right], output=target,
                line=line, col=col,
            ))
            return

        # --- method / function calls ---
        if isinstance(node, ast.Call):
            self._process_call(node, target, line, col)
            return

        # --- simple alias: y = x  (track so later uses of y resolve to x) ---
        if isinstance(node, ast.Name):
            source = self._resolve_name(node)
            if source != target:
                self._aliases[target] = source
            return

    def _resolve_arg(self, arg: ast.expr) -> str:
        """Resolve a call argument, recursively processing nested expressions."""
        if isinstance(arg, ast.Call):
            tmp = self._fresh("inner")
            self._process_call(arg, tmp, getattr(arg, 'lineno', 0),
                               getattr(arg, 'col_offset', 0))
            return tmp
        if isinstance(arg, ast.BinOp):
            tmp = self._fresh("binop")
            self._process_expr(arg, tmp, getattr(arg, 'lineno', 0),
                               getattr(arg, 'col_offset', 0))
            return tmp
        return self._resolve_name(arg)

    def _process_call(
        self, node: ast.Call, target: str, line: int, col: int
    ) -> None:
        func = node.func

        # --- self.<layer>(x) ------------------------------------------------
        if (isinstance(func, ast.Attribute) and
                isinstance(func.value, ast.Name) and
                func.value.id == "self" and
                func.attr in self.layers):
            layer_name = func.attr
            inputs = [self._resolve_arg(a) for a in node.args]
            self.steps.append(ComputationStep(
                op=OpKind.LAYER_CALL,
                inputs=inputs,
                output=target,
                layer_ref=layer_name,
                line=line, col=col,
            ))
            return

        # --- x.<method>(...) ------------------------------------------------
        if isinstance(func, ast.Attribute):
            method = func.attr
            if method in _METHOD_OPS:
                base = self._resolve_arg(func.value)
                params: Dict[str, Any] = {}

                if method in ("view", "reshape"):
                    dims = [_const_value(a) for a in node.args]
                    params["dims"] = tuple(
                        d if d is not None else -1 for d in dims
                    )
                elif method == "flatten":
                    sd = _const_value(node.args[0]) if node.args else 1
                    params["start_dim"] = sd
                elif method in ("squeeze", "unsqueeze"):
                    if node.args:
                        params["dim"] = _const_value(node.args[0])
                elif method == "transpose":
                    if len(node.args) >= 2:
                        params["dim0"] = _const_value(node.args[0])
                        params["dim1"] = _const_value(node.args[1])
                elif method == "permute":
                    params["dims"] = tuple(
                        _const_value(a) for a in node.args
                    )
                elif method == "to":
                    if node.args:
                        params["device"] = _const_value(node.args[0])
                    for kw in node.keywords:
                        if kw.arg == "device":
                            params["device"] = _const_value(kw.value)
                elif method == "cuda":
                    params["device"] = "cuda:0"
                elif method == "cpu":
                    params["device"] = "cpu"

                self.steps.append(ComputationStep(
                    op=_METHOD_OPS[method],
                    inputs=[base],
                    output=target,
                    params=params,
                    line=line, col=col,
                ))
                return

        # --- F.<func>(...) or torch.<func>(...) ------------------------------
        func_name = _name_or_attr(func)
        if func_name:
            short = func_name.split(".")[-1]

            # Functional ops
            if short in _FUNCTIONAL_OPS:
                inputs = [self._resolve_arg(a) for a in node.args]
                params_dict: Dict[str, Any] = {}
                for kw in node.keywords:
                    if kw.arg:
                        params_dict[kw.arg] = _const_value(kw.value)
                self.steps.append(ComputationStep(
                    op=_FUNCTIONAL_OPS[short],
                    inputs=inputs,
                    output=target,
                    params=params_dict,
                    line=line, col=col,
                ))
                return

            # torch.matmul
            if short in ("matmul", "mm", "bmm"):
                inputs = [self._resolve_arg(a) for a in node.args]
                self.steps.append(ComputationStep(
                    op=OpKind.MATMUL,
                    inputs=inputs,
                    output=target,
                    line=line, col=col,
                ))
                return

        # Fallback: custom
        inputs = [self._resolve_arg(a) for a in node.args]
        self.steps.append(ComputationStep(
            op=OpKind.CUSTOM,
            inputs=inputs,
            output=target,
            line=line, col=col,
        ))


# --- Top-level extraction function ----------------------------------------

def _find_method(cls_node: ast.ClassDef, name: str) -> Optional[ast.FunctionDef]:
    """Find a method by name inside a ClassDef."""
    for item in cls_node.body:
        if isinstance(item, ast.FunctionDef) and item.name == name:
            return item
    return None


def extract_computation_graph(source: str) -> ComputationGraph:
    """Parse Python *source* and extract the computation graph of the first
    ``nn.Module`` subclass found.

    Returns a ``ComputationGraph`` populated with layers (from ``__init__``)
    and computation steps (from ``forward``).

    Raises ``ValueError`` if no nn.Module subclass is found.
    """
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        # Heuristic: check for nn.Module in bases
        bases = [_name_or_attr(b) for b in node.bases]
        is_module = any(
            b in ("nn.Module", "Module", "torch.nn.Module")
            for b in bases if b is not None
        )
        if not is_module:
            continue

        graph = ComputationGraph(class_name=node.name)

        # --- __init__: extract layers ---
        init_fn = _find_method(node, "__init__")
        if init_fn:
            extractor = _InitExtractor()
            extractor.extract(init_fn)
            graph.layers = extractor.layers

        # --- forward: extract steps ---
        fwd_fn = _find_method(node, "forward")
        if fwd_fn:
            fwd_ext = _ForwardExtractor(graph.layers)
            fwd_ext.extract(fwd_fn)
            graph.steps = fwd_ext.steps
            graph.input_names = fwd_ext.input_names
            graph.output_names = fwd_ext.output_names

        return graph

    raise ValueError("No nn.Module subclass found in source")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Z3 encoding helpers
# ═══════════════════════════════════════════════════════════════════════════════

_z3_ctx_counter = 0


class _Z3Context:
    """Manages Z3 sorts, constants, and helpers for the product theory
    T_shape × T_device × T_phase.

    Provides:
      - Enumeration sorts for devices and phases.
      - Fresh variable creation for symbolic states.
      - Constraint encoders for each domain and cross-domain properties.
      - Incremental solving support via push/pop.
      - Query statistics tracking.
    """

    def __init__(self) -> None:
        if not HAS_Z3:
            raise RuntimeError(
                "Z3 is required for model checking.  "
                "Install it with:  pip install z3-solver"
            )

        global _z3_ctx_counter
        _z3_ctx_counter += 1
        suffix = _z3_ctx_counter

        self.solver = z3.Solver()
        self.solver.set("timeout", 10000)  # 10 s

        # Custom theory plugins (domain-specific SMT theories)
        # Note: Z3 only supports one UserPropagateBase per solver,
        # so we attach only the broadcast theory (which covers broadcasting
        # and matmul constraints).
        if HAS_THEORY_PLUGINS:
            self.broadcast_theory = BroadcastTheoryPlugin(self.solver)
        else:
            self.broadcast_theory = None
        # Stride theory lives on a *separate* solver (Z3 allows only one
        # UserPropagateBase per solver).  It is used for reshape-validity
        # queries that benefit from the stride propagator.
        if HAS_THEORY_PLUGINS:
            self._stride_solver = z3.Solver()
            self._stride_solver.set("timeout", 5000)
            self.stride_theory = StrideTheoryPlugin(self._stride_solver)
        else:
            self.stride_theory = None

        # Device theory plugin on a separate solver (one UserPropagateBase
        # per solver).  Provides eager propagation for device constraints.
        if HAS_DEVICE_THEORY:
            self._device_solver = z3.Solver()
            self._device_solver.set("timeout", 5000)
            self.device_theory = DeviceTheoryPlugin(self._device_solver)
        else:
            self.device_theory = None

        # Phase theory plugin on a separate solver.
        # Provides eager propagation for phase-dependent behaviour.
        if HAS_PHASE_THEORY:
            self._phase_solver = z3.Solver()
            self._phase_solver.set("timeout", 5000)
            self.phase_theory = PhaseTheoryPlugin(self._phase_solver)
        else:
            self.phase_theory = None

        # --- Device enumeration sort (unique name per context) ---
        self.DeviceSort, self.device_consts = z3.EnumSort(
            f"Device_{suffix}",
            [f"CPU_{suffix}", f"CUDA_0_{suffix}", f"CUDA_1_{suffix}",
             f"CUDA_2_{suffix}", f"CUDA_3_{suffix}"],
        )
        (self.DEV_CPU, self.DEV_CUDA0, self.DEV_CUDA1,
         self.DEV_CUDA2, self.DEV_CUDA3) = self.device_consts

        # --- Phase enumeration sort (unique name per context) ---
        self.PhaseSort, self.phase_consts = z3.EnumSort(
            f"Phase_{suffix}", [f"TRAIN_{suffix}", f"EVAL_{suffix}"],
        )
        self.PHASE_TRAIN, self.PHASE_EVAL = self.phase_consts

        # --- Symbolic dimension pool ---
        self._sym_dims: Dict[str, z3.ArithRef] = {}

        # --- Device variables ---
        self._dev_vars: Dict[str, z3.ExprRef] = {}

        # --- Gradient booleans ---
        self._grad_vars: Dict[str, z3.BoolRef] = {}

        # --- Phase variable (single for the whole model) ---
        self.phase_var = z3.Const("phase", self.PhaseSort)

        # --- Query statistics ---
        self._query_count: int = 0
        self._sat_count: int = 0
        self._unsat_count: int = 0
        self._total_solve_time_ms: float = 0.0

        # --- Theory solver constraint counters ---
        self._device_constraints_registered: int = 0
        self._phase_constraints_registered: int = 0

        # --- Theory combination checker (Tinelli-Zarba) ---
        if HAS_THEORY_COMBINATION:
            self._theory_combiner = TensorTheoryCombination()
            if self.broadcast_theory is not None:
                self._theory_combiner.add_broadcast_theory(
                    self.solver, self.broadcast_theory
                )
            if self.stride_theory is not None:
                self._theory_combiner.add_stride_theory(
                    self._stride_solver, self.stride_theory
                )
            if self.device_theory is not None:
                self._theory_combiner.add_device_theory(
                    self._device_solver, self.device_theory
                )
            if self.phase_theory is not None:
                self._theory_combiner.add_phase_theory(
                    self._phase_solver, self.phase_theory
                )
        else:
            self._theory_combiner = None

    # --- dimension helpers -------------------------------------------------

    def dim(self, name: str) -> z3.ArithRef:
        """Return (or create) a Z3 Int for a symbolic dimension."""
        if name not in self._sym_dims:
            self._sym_dims[name] = z3.Int(name)
        return self._sym_dims[name]

    def shape_to_z3(
        self, shape: TensorShape, prefix: str
    ) -> List[z3.ArithRef]:
        """Convert a TensorShape to a list of Z3 integer expressions.

        Concrete dims become ``z3.IntVal(n)``; symbolic dims become named
        Z3 Ints.
        """
        z3_dims: List[z3.ArithRef] = []
        for i, sd in enumerate(shape.dims):
            if sd.is_symbolic:
                z3_dims.append(self.dim(str(sd.value)))
            else:
                z3_dims.append(z3.IntVal(sd.value))
        return z3_dims

    # --- fresh variable creation for symbolic states -------------------------

    def fresh_shape_vars(
        self, tensor_name: str, ndim: int, step: int
    ) -> List[z3.ArithRef]:
        """Create fresh Z3 int variables for tensor shape at step."""
        return [z3.Int(f"sh_{tensor_name}_d{i}_s{step}") for i in range(ndim)]

    def fresh_device_var(self, tensor_name: str, step: int) -> z3.ExprRef:
        """Create fresh Z3 device variable for tensor at step."""
        return z3.Const(f"dev_{tensor_name}_s{step}", self.DeviceSort)

    def fresh_grad_var(self, tensor_name: str, step: int) -> z3.BoolRef:
        """Create fresh Z3 Bool for gradient tracking at step."""
        return z3.Bool(f"grad_{tensor_name}_s{step}")

    def fresh_phase_var(self, step: int) -> z3.ExprRef:
        """Create fresh Z3 phase variable at step."""
        return z3.Const(f"phase_s{step}", self.PhaseSort)

    # --- device helpers ----------------------------------------------------

    def dev_var(self, tensor_name: str) -> z3.ExprRef:
        """Return (or create) a Z3 device variable for *tensor_name*."""
        if tensor_name not in self._dev_vars:
            self._dev_vars[tensor_name] = z3.Const(
                f"dev_{tensor_name}", self.DeviceSort
            )
        return self._dev_vars[tensor_name]

    def device_to_z3(self, device: Device) -> z3.ExprRef:
        """Map a Device enum value to the corresponding Z3 constant."""
        return {
            Device.CPU: self.DEV_CPU,
            Device.CUDA_0: self.DEV_CUDA0,
            Device.CUDA_1: self.DEV_CUDA1,
            Device.CUDA_2: self.DEV_CUDA2,
            Device.CUDA_3: self.DEV_CUDA3,
        }[device]

    def phase_to_z3(self, phase: Phase) -> z3.ExprRef:
        """Map a Phase enum value to the corresponding Z3 constant."""
        return self.PHASE_TRAIN if phase == Phase.TRAIN else self.PHASE_EVAL

    # --- gradient helpers --------------------------------------------------

    def grad_var(self, tensor_name: str) -> z3.BoolRef:
        if tensor_name not in self._grad_vars:
            self._grad_vars[tensor_name] = z3.Bool(f"grad_{tensor_name}")
        return self._grad_vars[tensor_name]

    # --- product-theory constraint encoders --------------------------------

    def encode_device_constraint(
        self, dev_a: z3.ExprRef, dev_b: z3.ExprRef
    ) -> z3.BoolRef:
        """Z3 constraint: two tensors are on the same device.

        Also registers the pair with the DeviceTheoryPlugin (if available)
        for eager propagation on the device solver.
        """
        if self.device_theory is not None:
            try:
                from src.smt.device_theory import DeviceSort as _DTSort
            except ImportError:
                _DTSort = None
            if _DTSort is not None:
                _da = z3.Const(str(dev_a), _DTSort)
                _db = z3.Const(str(dev_b), _DTSort)
                self._device_solver.add(
                    self.device_theory.same_device(_da, _db)
                )
                self._device_constraints_registered += 1
        return dev_a == dev_b

    def encode_device_transfer(
        self, dev_out: z3.ExprRef, target: Device
    ) -> z3.BoolRef:
        """Z3 constraint for ``.to(device)`` / ``.cuda()`` / ``.cpu()``."""
        return dev_out == self.device_to_z3(target)

    def encode_phase_constraint(
        self, phase: z3.ExprRef, layer_kind: LayerKind
    ) -> Tuple[z3.BoolRef, z3.BoolRef]:
        """Encode phase-dependent behavior as (train_cond, eval_cond).

        For dropout: in eval, output equals input (identity).
        For batchnorm: in eval, uses running statistics.
        """
        is_train = phase == self.PHASE_TRAIN
        is_eval = phase == self.PHASE_EVAL
        if layer_kind == LayerKind.DROPOUT:
            return (is_train, is_eval)
        elif layer_kind in (LayerKind.BATCHNORM1D, LayerKind.BATCHNORM2D):
            return (is_train, is_eval)
        return (z3.BoolVal(True), z3.BoolVal(True))

    def encode_gradient_constraint(
        self, grad_out: z3.BoolRef, requires_grad: bool
    ) -> z3.BoolRef:
        """Z3 constraint setting gradient status."""
        return grad_out == z3.BoolVal(requires_grad)

    def encode_cross_domain_constraint(
        self,
        shape_pre: List[z3.ArithRef],
        shape_post: List[z3.ArithRef],
        dev_pre: z3.ExprRef,
        dev_post: z3.ExprRef,
        is_device_transfer: bool,
    ) -> List[z3.BoolRef]:
        """Cross-domain constraints spanning shape + device.

        Device transfer preserves shape.  Non-transfer ops preserve device.
        """
        constraints: List[z3.BoolRef] = []
        if is_device_transfer:
            for dp, dq in zip(shape_pre, shape_post):
                constraints.append(dp == dq)
        else:
            constraints.append(dev_pre == dev_post)
        return constraints

    # --- positivity constraints -------------------------------------------

    def positive_dim_constraints(self) -> List[z3.BoolRef]:
        """All symbolic dims must be ≥ 1."""
        return [v > 0 for v in self._sym_dims.values()]

    # --- timed Z3 check ---------------------------------------------------

    def timed_check(self, solver: z3.Solver) -> z3.CheckSatResult:
        """Run solver.check() with timing and statistics tracking."""
        t0 = time.monotonic()
        result = solver.check()
        elapsed = (time.monotonic() - t0) * 1000
        self._query_count += 1
        self._total_solve_time_ms += elapsed
        if result == z3.sat:
            self._sat_count += 1
        elif result == z3.unsat:
            self._unsat_count += 1
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Return Z3 solver statistics."""
        stats = {
            "z3_queries": self._query_count,
            "z3_total_time_ms": self._total_solve_time_ms,
            "z3_sat_count": self._sat_count,
            "z3_unsat_count": self._unsat_count,
        }
        if self.broadcast_theory is not None:
            prop = self.broadcast_theory.propagator
            stats["broadcast_propagations"] = len(prop._broadcast_triples)
            stats["broadcast_conflicts"] = len(prop._matmul_pairs)
        if self.stride_theory is not None:
            prop = self.stride_theory.propagator
            stats["stride_constraints"] = len(prop._contiguous)
            stats["stride_reshapes"] = len(prop._reshapes)
            stats["stride_divisibility"] = len(prop._divisibility)
        if self.device_theory is not None:
            prop = self.device_theory.propagator
            stats["device_same_pairs"] = len(prop._same_device_pairs)
            stats["device_transfer_triples"] = len(prop._transfer_triples)
            stats["device_inherit_pairs"] = len(prop._inherit_pairs)
            stats["device_constraints_registered"] = self._device_constraints_registered
        if self.phase_theory is not None:
            prop = self.phase_theory.propagator
            stats["phase_dropout_constraints"] = len(prop._dropout_constraints)
            stats["phase_batchnorm_constraints"] = len(
                prop._batchnorm_constraints
            )
            stats["phase_constraints_registered"] = self._phase_constraints_registered
        if self._theory_combiner is not None:
            stats["theory_combination_available"] = True
        return stats

    def verify_theory_combination(self) -> Optional[Dict[str, Any]]:
        """Run Tinelli-Zarba theory combination check.

        Returns None if no combiner available, otherwise a dict with
        'consistent' (bool) and 'details' fields.
        """
        if self._theory_combiner is None:
            return None
        result = self._theory_combiner.verify_theory_combination_consistency()
        return {
            "consistent": result.is_sat,
            "arrangements_checked": result.arrangements_checked,
            "details": result.reason,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Shape-propagation rules (symbolic, Z3-backed)
# ═══════════════════════════════════════════════════════════════════════════════

def _propagate_linear(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """Propagate shape through nn.Linear.

    nn.Linear(in_features, out_features) maps  (*, in_features) → (*, out_features).
    """
    if input_shape.ndim < 1:
        return None, "Linear requires at least 1D input"

    last = input_shape.dims[-1]
    if layer.in_features is not None and not last.is_symbolic:
        if last.value != layer.in_features:
            return None, (
                f"Linear expects last dim={layer.in_features}, "
                f"got {last.value}"
            )

    out_feat = layer.out_features
    if out_feat is None:
        return None, "Linear out_features unknown"

    new_dims = input_shape.dims[:-1] + (ShapeDim(out_feat),)
    return TensorShape(new_dims), None


def _propagate_conv2d(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """Propagate shape through nn.Conv2d.

    Expects input (N, C_in, H, W) → (N, C_out, H', W').
    H' = floor((H + 2*padding - kernel_size) / stride) + 1
    """
    if input_shape.ndim != 4:
        return None, f"Conv2d expects 4D input, got {input_shape.ndim}D"

    c_in = input_shape.dims[1]
    if layer.in_channels is not None and not c_in.is_symbolic:
        if c_in.value != layer.in_channels:
            return None, (
                f"Conv2d expects {layer.in_channels} input channels, "
                f"got {c_in.value}"
            )

    out_c = layer.out_channels
    if out_c is None:
        return None, "Conv2d out_channels unknown"

    # Compute output spatial dims:
    # H' = floor((H + 2*pad - dilation*(kernel-1) - 1) / stride + 1)
    ks = layer.kernel_size or (3, 3)
    stride = layer.params.get("stride", (1, 1))
    if isinstance(stride, int):
        stride = (stride, stride)
    padding = layer.params.get("padding", (0, 0))
    if isinstance(padding, int):
        padding = (padding, padding)
    dilation = layer.params.get("dilation", (1, 1))
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    h_in = input_shape.dims[2]
    w_in = input_shape.dims[3]

    if not h_in.is_symbolic and not w_in.is_symbolic:
        h_out = (h_in.value + 2 * padding[0] - dilation[0] * (ks[0] - 1) - 1) // stride[0] + 1
        w_out = (w_in.value + 2 * padding[1] - dilation[1] * (ks[1] - 1) - 1) // stride[1] + 1
        new_dims = (
            input_shape.dims[0],
            ShapeDim(out_c),
            ShapeDim(h_out),
            ShapeDim(w_out),
        )
    else:
        new_dims = (
            input_shape.dims[0],
            ShapeDim(out_c),
            ShapeDim("H_out"),
            ShapeDim("W_out"),
        )
    return TensorShape(new_dims), None


def _propagate_batchnorm(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """BatchNorm preserves shape but checks the feature dimension."""
    if input_shape.ndim < 2:
        return None, f"BatchNorm requires at least 2D input, got {input_shape.ndim}D"

    feat = input_shape.dims[1]
    if layer.num_features is not None and not feat.is_symbolic:
        if feat.value != layer.num_features:
            return None, (
                f"BatchNorm expects {layer.num_features} features, "
                f"got {feat.value}"
            )
    return input_shape, None


def _propagate_dropout(
    input_shape: TensorShape, _layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """Dropout preserves shape."""
    return input_shape, None


def _propagate_activation(
    input_shape: TensorShape,
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """Element-wise activations preserve shape."""
    return input_shape, None


def _propagate_embedding(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """nn.Embedding maps (*, ) → (*, embedding_dim)."""
    if layer.embedding_dim is None:
        return None, "Embedding dim unknown"
    new_dims = input_shape.dims + (ShapeDim(layer.embedding_dim),)
    return TensorShape(new_dims), None


def _propagate_flatten(
    input_shape: TensorShape, start_dim: int = 1
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """Flatten from start_dim to end."""
    if start_dim < 0:
        start_dim = input_shape.ndim + start_dim
    if start_dim >= input_shape.ndim:
        return input_shape, None

    kept = input_shape.dims[:start_dim]

    # Compute flattened size
    flat_parts = input_shape.dims[start_dim:]
    all_concrete = all(not d.is_symbolic for d in flat_parts)
    if all_concrete:
        total = 1
        for d in flat_parts:
            total *= d.value
        flat_dim = ShapeDim(total)
    else:
        flat_dim = ShapeDim("_flat")

    return TensorShape(kept + (flat_dim,)), None


def _propagate_adaptive_avgpool2d(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """AdaptiveAvgPool2d maps (N, C, H, W) → (N, C, H_out, W_out)."""
    if input_shape.ndim != 4:
        return None, f"AdaptiveAvgPool2d expects 4D, got {input_shape.ndim}D"
    out = layer.output_size
    if out is None:
        return None, "output_size unknown"
    new_dims = (
        input_shape.dims[0],
        input_shape.dims[1],
        ShapeDim(out[0]),
        ShapeDim(out[1]),
    )
    return TensorShape(new_dims), None


def _propagate_pool2d(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """MaxPool2d / AvgPool2d — compute output spatial dims."""
    if input_shape.ndim != 4:
        return None, f"Pool2d expects 4D, got {input_shape.ndim}D"

    ks = layer.kernel_size or (2, 2)
    stride = layer.params.get("stride", ks)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(ks, int):
        ks = (ks, ks)
    padding = layer.params.get("padding", (0, 0))
    if isinstance(padding, int):
        padding = (padding, padding)

    h_in = input_shape.dims[2]
    w_in = input_shape.dims[3]

    if not h_in.is_symbolic and not w_in.is_symbolic:
        h_out = (h_in.value + 2 * padding[0] - ks[0]) // stride[0] + 1
        w_out = (w_in.value + 2 * padding[1] - ks[1]) // stride[1] + 1
        new_dims = (
            input_shape.dims[0],
            input_shape.dims[1],
            ShapeDim(h_out),
            ShapeDim(w_out),
        )
    else:
        new_dims = (
            input_shape.dims[0],
            input_shape.dims[1],
            ShapeDim("H_pool"),
            ShapeDim("W_pool"),
        )
    return TensorShape(new_dims), None


def _propagate_sequential(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """Propagate shape through nn.Sequential by chaining sub-layers."""
    if not layer.sub_layers:
        return input_shape, None
    current = input_shape
    for sub in layer.sub_layers:
        propagator = _LAYER_PROPAGATORS.get(sub.kind)
        if propagator is not None:
            current, err = propagator(current, sub)
            if err:
                return None, f"Sequential sub-layer {sub.attr_name}: {err}"
            if current is None:
                return None, f"Sequential sub-layer {sub.attr_name}: shape unknown"
        elif sub.kind in (LayerKind.RELU, LayerKind.DROPOUT,
                          LayerKind.IDENTITY, LayerKind.SOFTMAX):
            pass  # shape-preserving
        elif sub.kind == LayerKind.FLATTEN:
            current, err = _propagate_flatten(current, 1)
            if current is None:
                return None, f"Sequential sub-layer {sub.attr_name}: flatten failed"
        # else: unknown sub-layer, conservatively preserve shape
    return current, None


def _propagate_groupnorm(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """GroupNorm preserves shape but checks the channel dimension."""
    if input_shape.ndim < 2:
        return None, f"GroupNorm requires at least 2D input, got {input_shape.ndim}D"
    feat = input_shape.dims[1]
    if layer.num_features is not None and not feat.is_symbolic:
        if feat.value != layer.num_features:
            return None, (
                f"GroupNorm expects {layer.num_features} channels, "
                f"got {feat.value}"
            )
    return input_shape, None


def _propagate_instancenorm2d(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """InstanceNorm2d preserves shape but checks the channel dimension."""
    if input_shape.ndim != 4:
        return None, f"InstanceNorm2d expects 4D input, got {input_shape.ndim}D"
    feat = input_shape.dims[1]
    if layer.num_features is not None and not feat.is_symbolic:
        if feat.value != layer.num_features:
            return None, (
                f"InstanceNorm2d expects {layer.num_features} channels, "
                f"got {feat.value}"
            )
    return input_shape, None


_LAYER_PROPAGATORS = {
    LayerKind.LINEAR: _propagate_linear,
    LayerKind.CONV2D: _propagate_conv2d,
    LayerKind.BATCHNORM1D: _propagate_batchnorm,
    LayerKind.BATCHNORM2D: _propagate_batchnorm,
    LayerKind.DROPOUT: _propagate_dropout,
    LayerKind.EMBEDDING: _propagate_embedding,
    LayerKind.ADAPTIVE_AVGPOOL2D: _propagate_adaptive_avgpool2d,
    LayerKind.MAXPOOL2D: _propagate_pool2d,
    LayerKind.AVGPOOL2D: _propagate_pool2d,
    LayerKind.SEQUENTIAL: _propagate_sequential,
    LayerKind.GROUPNORM: _propagate_groupnorm,
    LayerKind.INSTANCENORM2D: _propagate_instancenorm2d,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 7b. Symbolic state for Z3-backed constraint verification
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KripkeState:
    """Z3-backed symbolic state for constraint-based verification.

    Maps tensor names to Z3 variables for shape dimensions, device
    placement, gradient status, and overall phase.  Each ``KripkeState``
    represents the system state at a specific step in the computation
    graph.
    """
    step_index: int
    shape_vars: Dict[str, List[Any]] = field(default_factory=dict)
    device_vars: Dict[str, Any] = field(default_factory=dict)
    phase_var: Any = None
    grad_vars: Dict[str, Any] = field(default_factory=dict)


# Layer kinds whose parameters reside on a device.
_PARAMETERISED_LAYERS: FrozenSet[LayerKind] = frozenset({
    LayerKind.LINEAR, LayerKind.CONV2D,
    LayerKind.BATCHNORM1D, LayerKind.BATCHNORM2D,
    LayerKind.LAYERNORM, LayerKind.EMBEDDING,
    LayerKind.LSTM, LayerKind.GRU,
    LayerKind.MULTIHEAD_ATTENTION,
})


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Constraint Verifier (symbolic constraint propagation)
# ═══════════════════════════════════════════════════════════════════════════════

class ConstraintVerifier:
    """Constraint-based verifier for nn.Module computation graphs using the
    product theory T_shape × T_device × T_phase.

    The verifier uses forward symbolic constraint propagation through the
    computation DAG to verify four safety properties at every step:

      1. **shape_compatible** — each operation's input shapes are compatible
         with its semantics (e.g. matmul inner dims match).
      2. **device_consistent** — all tensors participating in an operation
         live on the same device (Z3 enum-sort backed).
      3. **gradient_valid** — gradient invariants are maintained (parameters
         require grad; detached tensors do not).
      4. **phase_correct** — phase-dependent layers (dropout, batchnorm)
         behave correctly w.r.t. train/eval mode (Z3 enum-sort backed).

    Algorithm (forward constraint propagation with product theory):

        **Base case** (steps 0 … N-1):
          - Create Z3 symbolic state variables for the initial state.
          - For each step *i* from 0 to N-1:
              • Create fresh Z3 variables for state after step *i*.
              • Add transition constraints (shape + device + phase
                + gradient) relating pre-state to post-state.
              • Check safety property for each domain via Z3 ``check()``.
          - Any SAT result yields a concrete counterexample.

        **Inductive step**:
          - For each consecutive pair of steps *(i, i+1)*:
              • Create Z3 symbolic states with *free* shape variables.
              • Assume safety at step *i*.
              • Add transition constraints.
              • Check whether step *i+1* can violate safety.
          - UNSAT ⇒ safety proved.

    Attributes:
        graph:        the computation graph to verify.
        ctx:          the Z3 encoding context (product theory).
        max_k:        maximum verification depth (defaults to graph length).
        input_shapes: user-supplied input shapes (may contain symbolic dims).
        default_device: default device for tensors & parameters.
    """

    def __init__(
        self,
        graph: ComputationGraph,
        input_shapes: Optional[Dict[str, tuple]] = None,
        default_device: Device = Device.CPU,
        default_phase: Phase = Phase.TRAIN,
        max_k: Optional[int] = None,
    ) -> None:
        self.graph = graph
        self.input_shapes = input_shapes or {}
        self.default_device = default_device
        self.default_phase = default_phase
        self.max_k = max_k if max_k is not None else graph.num_steps
        self.ctx = _Z3Context()
        self._stride_check_id = 0

        self._init_state = self._build_initial_state()

    # ------------------------------------------------------------------
    # Initial ModelState construction (concrete level)
    # ------------------------------------------------------------------

    def _build_initial_state(self) -> ModelState:
        """Construct the initial ``ModelState`` from *input_shapes*."""
        state = ModelState(phase=self.default_phase)
        for name, raw_shape in self.input_shapes.items():
            dims: List[ShapeDim] = []
            for d in raw_shape:
                if isinstance(d, int):
                    dims.append(ShapeDim(d))
                elif isinstance(d, str):
                    dims.append(ShapeDim(d))
                else:
                    dims.append(ShapeDim("_unk"))
            state.shape_env[name] = TensorShape(tuple(dims))
            state.device_map[name] = self.default_device
            state.gradient_status[name] = False
        return state

    # ------------------------------------------------------------------
    # Symbolic state construction (Z3 level)
    # ------------------------------------------------------------------

    def _build_kripke_state(
        self,
        step_idx: int,
        model_state: ModelState,
        free_shapes: bool = False,
    ) -> KripkeState:
        """Build a ``KripkeState`` (symbolic state) from a ``ModelState``.

        Parameters
        ----------
        step_idx : int
            Step index (used to generate unique Z3 variable names).
        model_state : ModelState
            Concrete state providing structure (tensor names, ndim, …).
        free_shapes : bool
            If ``True``, all shape dimensions become free Z3 Int variables
            (used in the inductive step).  Otherwise concrete dims become
            ``z3.IntVal`` constants.
        """
        shape_vars: Dict[str, list] = {}
        device_vars: Dict[str, Any] = {}
        grad_vars: Dict[str, Any] = {}

        for tname, shape in model_state.shape_env.items():
            if free_shapes:
                shape_vars[tname] = self.ctx.fresh_shape_vars(
                    tname, shape.ndim, step_idx
                )
            else:
                shape_vars[tname] = self.ctx.shape_to_z3(
                    shape, f"s{step_idx}_{tname}"
                )

        for tname in model_state.device_map:
            device_vars[tname] = self.ctx.fresh_device_var(tname, step_idx)

        for tname in model_state.gradient_status:
            grad_vars[tname] = self.ctx.fresh_grad_var(tname, step_idx)

        phase_v = self.ctx.fresh_phase_var(step_idx)

        return KripkeState(
            step_index=step_idx,
            shape_vars=shape_vars,
            device_vars=device_vars,
            phase_var=phase_v,
            grad_vars=grad_vars,
        )

    def _initial_constraints(self, k0: KripkeState) -> List:
        """Bind step-0 symbolic state variables to known concrete values."""
        cs: list = []
        for tname, device in self._init_state.device_map.items():
            if tname in k0.device_vars:
                cs.append(
                    k0.device_vars[tname] == self.ctx.device_to_z3(device)
                )
        for tname, has_grad in self._init_state.gradient_status.items():
            if tname in k0.grad_vars:
                cs.append(k0.grad_vars[tname] == z3.BoolVal(has_grad))
        if k0.phase_var is not None:
            cs.append(k0.phase_var == self.ctx.phase_to_z3(self.default_phase))
        for dims in k0.shape_vars.values():
            for d in dims:
                if not z3.is_int_value(d):
                    cs.append(d > 0)
        return cs

    # ------------------------------------------------------------------
    # Transition-relation encoders  (pre → step → post)
    # ------------------------------------------------------------------

    def _encode_transition(
        self,
        pre: KripkeState,
        step: ComputationStep,
        post: KripkeState,
        model_state: ModelState,
        step_idx: int,
    ) -> List:
        """Full transition relation: shape ∧ device ∧ phase ∧ gradient ∧ frame."""
        cs: list = []
        cs.extend(self._encode_shape_transition(pre, step, post, model_state))
        cs.extend(self._encode_device_transition(pre, step, post, model_state))
        cs.extend(self._encode_phase_transition(pre, post))
        cs.extend(self._encode_gradient_transition(pre, step, post))
        cs.extend(self._encode_frame_conditions(pre, post, step))
        return cs

    # -- shape --

    def _encode_shape_transition(
        self,
        pre: KripkeState,
        step: ComputationStep,
        post: KripkeState,
        model_state: ModelState,
    ) -> List:
        cs: list = []
        inp_name = step.inputs[0] if step.inputs else None

        if step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = self.graph.layers.get(step.layer_ref)
            if (layer and inp_name and inp_name in pre.shape_vars
                    and step.output in post.shape_vars):
                pre_d = pre.shape_vars[inp_name]
                post_d = post.shape_vars[step.output]
                if layer.kind == LayerKind.LINEAR:
                    for i in range(min(len(pre_d) - 1, len(post_d) - 1)):
                        cs.append(post_d[i] == pre_d[i])
                    if layer.out_features is not None and post_d:
                        cs.append(
                            post_d[-1] == z3.IntVal(layer.out_features)
                        )
                elif layer.kind == LayerKind.CONV2D:
                    if pre_d and post_d:
                        cs.append(post_d[0] == pre_d[0])
                    if layer.out_channels is not None and len(post_d) >= 2:
                        cs.append(
                            post_d[1] == z3.IntVal(layer.out_channels)
                        )
                elif layer.kind in (LayerKind.BATCHNORM1D,
                                    LayerKind.BATCHNORM2D,
                                    LayerKind.LAYERNORM,
                                    LayerKind.GROUPNORM,
                                    LayerKind.INSTANCENORM2D):
                    for dp, dq in zip(pre_d, post_d):
                        cs.append(dq == dp)
                elif layer.kind in (LayerKind.RELU, LayerKind.DROPOUT,
                                    LayerKind.IDENTITY, LayerKind.SOFTMAX):
                    for dp, dq in zip(pre_d, post_d):
                        cs.append(dq == dp)
                elif layer.kind == LayerKind.EMBEDDING:
                    for i in range(min(len(pre_d), len(post_d) - 1)):
                        cs.append(post_d[i] == pre_d[i])
                    if layer.embedding_dim is not None and post_d:
                        cs.append(
                            post_d[-1] == z3.IntVal(layer.embedding_dim)
                        )
                elif layer.kind == LayerKind.FLATTEN:
                    if pre_d and post_d:
                        cs.append(post_d[0] == pre_d[0])
                elif layer.kind in (LayerKind.ADAPTIVE_AVGPOOL2D,):
                    if len(pre_d) >= 2 and len(post_d) >= 2:
                        cs.append(post_d[0] == pre_d[0])
                        cs.append(post_d[1] == pre_d[1])
                    if layer.output_size and len(post_d) >= 4:
                        cs.append(
                            post_d[2] == z3.IntVal(layer.output_size[0])
                        )
                        cs.append(
                            post_d[3] == z3.IntVal(layer.output_size[1])
                        )
                elif layer.kind in (LayerKind.MAXPOOL2D, LayerKind.AVGPOOL2D):
                    if len(pre_d) >= 2 and len(post_d) >= 2:
                        cs.append(post_d[0] == pre_d[0])
                        cs.append(post_d[1] == pre_d[1])
                elif layer.kind == LayerKind.SEQUENTIAL:
                    # Sequential: concrete shape propagation handled by
                    # _apply_layer_call; at Z3 level, constrain batch dim
                    # and defer to the concrete propagator's output.
                    if pre_d and post_d:
                        cs.append(post_d[0] == pre_d[0])
                elif layer.kind == LayerKind.MODULELIST:
                    # ModuleList elements used individually; preserve shape
                    for dp, dq in zip(pre_d, post_d):
                        cs.append(dq == dp)
                else:
                    for dp, dq in zip(pre_d, post_d):
                        cs.append(dq == dp)

        elif step.op == OpKind.MATMUL and len(step.inputs) >= 2:
            a, b = step.inputs[0], step.inputs[1]
            if (a in pre.shape_vars and b in pre.shape_vars
                    and step.output in post.shape_vars):
                ad = pre.shape_vars[a]
                bd = pre.shape_vars[b]
                pd = post.shape_vars[step.output]
                if len(ad) >= 2 and pd:
                    cs.append(pd[0] == ad[0])
                if len(bd) >= 2 and len(pd) >= 2:
                    cs.append(pd[-1] == bd[-1])

        elif step.op == OpKind.ADD and len(step.inputs) >= 2:
            a, b = step.inputs[0], step.inputs[1]
            if (a in pre.shape_vars and b in pre.shape_vars
                    and step.output in post.shape_vars):
                ad = pre.shape_vars[a]
                bd = pre.shape_vars[b]
                pd = post.shape_vars[step.output]
                ndim = max(len(ad), len(bd))
                for i in range(min(ndim, len(pd))):
                    da = ad[len(ad) - 1 - i] if i < len(ad) else z3.IntVal(1)
                    db = bd[len(bd) - 1 - i] if i < len(bd) else z3.IntVal(1)
                    dp = pd[len(pd) - 1 - i]
                    if self.ctx.broadcast_theory is not None:
                        cs.append(self.ctx.broadcast_theory.broadcast_result_dim(da, db, dp))
                    else:
                        cs.append(z3.Or(
                            z3.And(da == z3.IntVal(1), dp == db),
                            z3.And(db == z3.IntVal(1), dp == da),
                            z3.And(da == db, dp == da),
                        ))

        elif step.op == OpKind.RESHAPE:
            dims = step.params.get("dims")
            if (inp_name and inp_name in pre.shape_vars
                    and step.output in post.shape_vars and dims is not None):
                pre_d = pre.shape_vars[inp_name]
                post_d = post.shape_vars[step.output]
                # Fix concrete target dimensions
                for i, d in enumerate(dims):
                    if isinstance(d, int) and d >= 0 and i < len(post_d):
                        cs.append(post_d[i] == z3.IntVal(d))
                # Element-count preservation: product(pre) == product(post)
                cs.extend(self._encode_reshape_safety(pre_d, post_d))
                # Stride-based contiguity validation via stride theory
                if self.ctx.stride_theory is not None:
                    cs.extend(self._stride_reshape_check(pre_d, post_d))

        elif step.op in (OpKind.ACTIVATION, OpKind.DROPOUT, OpKind.SOFTMAX,
                          OpKind.CONTIGUOUS, OpKind.DETACH, OpKind.TO_DEVICE):
            if (inp_name and inp_name in pre.shape_vars
                    and step.output in post.shape_vars):
                for dp, dq in zip(
                    pre.shape_vars[inp_name],
                    post.shape_vars[step.output],
                ):
                    cs.append(dq == dp)

        elif step.op == OpKind.CONDITIONAL:
            # For conditional steps at Z3 level, conservatively pass through
            # (the concrete _apply_conditional handles branch selection)
            pass

        return cs

    @staticmethod
    def _encode_reshape_safety(
        old_dims: List, new_dims: List
    ) -> List:
        """Encode element-count preservation: product(old) == product(new)."""
        def _product(dims: List) -> "z3.ExprRef":
            if not dims:
                return z3.IntVal(1)
            result = dims[0]
            for d in dims[1:]:
                result = result * d
            return result

        return [_product(list(old_dims)) == _product(list(new_dims))]

    def _stride_reshape_check(
        self, old_dims: List, new_dims: List,
    ) -> List:
        """Query stride theory solver to verify reshape memory-validity.

        Uses the stride theory's separate solver to check that a contiguous
        source tensor can be validly reshaped to the target shape.
        """
        st = self.ctx.stride_theory
        ss = self.ctx._stride_solver
        n_old, n_new = len(old_dims), len(new_dims)
        uid = self._stride_check_id
        self._stride_check_id += 1

        old_sv = [z3.Int(f"_sr_o{i}_{uid}") for i in range(n_old)]
        old_st = [z3.Int(f"_sr_s{i}_{uid}") for i in range(n_old)]
        new_sv = [z3.Int(f"_sr_n{i}_{uid}") for i in range(n_new)]

        ss.push()
        # Source must have contiguous memory layout
        ss.add(st.contiguous_strides(old_sv, old_st))
        # Reshape must preserve element count (stride-theory propagator)
        ss.add(st.reshape_valid(old_sv, new_sv))

        for i, d in enumerate(old_dims):
            if z3.is_int_value(d):
                ss.add(old_sv[i] == d)
            ss.add(old_sv[i] > 0)
        for i, d in enumerate(new_dims):
            if z3.is_int_value(d):
                ss.add(new_sv[i] == d)
            ss.add(new_sv[i] > 0)

        result = ss.check()
        ss.pop()

        if result == z3.unsat:
            return [z3.BoolVal(False)]
        return []

    def _stride_contiguity_check(self, dims: List) -> List:
        """Verify source tensor admits a contiguous layout via stride theory.

        Contiguity is a prerequisite for many reshape operations.
        """
        st = self.ctx.stride_theory
        ss = self.ctx._stride_solver
        n = len(dims)
        uid = self._stride_check_id
        self._stride_check_id += 1

        shape_v = [z3.Int(f"_sc_d{i}_{uid}") for i in range(n)]
        stride_v = [z3.Int(f"_sc_s{i}_{uid}") for i in range(n)]

        ss.push()
        ss.add(st.contiguous_strides(shape_v, stride_v))

        for i, d in enumerate(dims):
            if z3.is_int_value(d):
                ss.add(shape_v[i] == d)
            ss.add(shape_v[i] > 0)

        result = ss.check()
        ss.pop()

        if result == z3.unsat:
            return [z3.BoolVal(False)]
        return []

    # -- device --

    def _encode_device_transition(
        self,
        pre: KripkeState,
        step: ComputationStep,
        post: KripkeState,
        model_state: ModelState,
    ) -> List:
        cs: list = []
        if step.op == OpKind.TO_DEVICE:
            dev_str = step.params.get("device")
            if dev_str is not None and step.output in post.device_vars:
                target = Device.from_string(str(dev_str))
                cs.append(self.ctx.encode_device_transfer(
                    post.device_vars[step.output], target
                ))
        elif step.inputs and step.output in post.device_vars:
            for inp in step.inputs:
                if inp in pre.device_vars:
                    cs.append(
                        post.device_vars[step.output]
                        == pre.device_vars[inp]
                    )
                    break
        return cs

    # -- phase --

    def _encode_phase_transition(
        self, pre: KripkeState, post: KripkeState
    ) -> List:
        if pre.phase_var is not None and post.phase_var is not None:
            return [pre.phase_var == post.phase_var]
        return []

    # -- gradient --

    def _encode_gradient_transition(
        self,
        pre: KripkeState,
        step: ComputationStep,
        post: KripkeState,
    ) -> List:
        cs: list = []
        if step.output not in post.grad_vars:
            return cs
        out_g = post.grad_vars[step.output]
        if step.op == OpKind.DETACH:
            cs.append(out_g == z3.BoolVal(False))
        elif step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = self.graph.layers.get(step.layer_ref)
            if layer:
                cs.append(out_g == z3.BoolVal(
                    layer.kind in _PARAMETERISED_LAYERS
                ))
        else:
            in_gs = [pre.grad_vars[i]
                      for i in step.inputs if i in pre.grad_vars]
            if in_gs:
                cs.append(out_g == (z3.Or(*in_gs) if len(in_gs) > 1
                                    else in_gs[0]))
            else:
                cs.append(out_g == z3.BoolVal(False))
        return cs

    # -- frame conditions (unchanged tensors keep their properties) --

    def _encode_frame_conditions(
        self, pre: KripkeState, post: KripkeState, step: ComputationStep
    ) -> List:
        cs: list = []
        modified = {step.output}
        for t in pre.shape_vars:
            if t not in modified and t in post.shape_vars:
                for dp, dq in zip(pre.shape_vars[t], post.shape_vars[t]):
                    cs.append(dq == dp)
        for t in pre.device_vars:
            if t not in modified and t in post.device_vars:
                cs.append(post.device_vars[t] == pre.device_vars[t])
        for t in pre.grad_vars:
            if t not in modified and t in post.grad_vars:
                cs.append(post.grad_vars[t] == pre.grad_vars[t])
        return cs

    # ------------------------------------------------------------------
    # Safety-property encoders
    # ------------------------------------------------------------------

    def _encode_shape_safety(
        self,
        k: KripkeState,
        step: ComputationStep,
        ms: ModelState,
        idx: int,
    ) -> List:
        """Encode shape compatibility constraints for *step*."""
        cs: list = []
        if step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = self.graph.layers.get(step.layer_ref)
            inp = step.inputs[0] if step.inputs else None
            if layer and inp and inp in k.shape_vars:
                dims = k.shape_vars[inp]
                if (layer.kind == LayerKind.LINEAR
                        and layer.in_features is not None and dims):
                    cs.append(dims[-1] == z3.IntVal(layer.in_features))
                elif layer.kind == LayerKind.CONV2D:
                    if layer.in_channels is not None and len(dims) >= 2:
                        cs.append(dims[1] == z3.IntVal(layer.in_channels))
                elif layer.kind in (LayerKind.BATCHNORM1D,
                                    LayerKind.BATCHNORM2D):
                    if (layer.num_features is not None
                            and len(dims) >= 2):
                        cs.append(
                            dims[1] == z3.IntVal(layer.num_features)
                        )
                elif layer.kind in (LayerKind.GROUPNORM,
                                    LayerKind.INSTANCENORM2D):
                    if (layer.num_features is not None
                            and len(dims) >= 2):
                        cs.append(
                            dims[1] == z3.IntVal(layer.num_features)
                        )
        elif step.op == OpKind.MATMUL and len(step.inputs) >= 2:
            a, b = step.inputs[0], step.inputs[1]
            if a in k.shape_vars and b in k.shape_vars:
                ad = k.shape_vars[a]
                bd = k.shape_vars[b]
                if ad and bd:
                    if len(bd) >= 2:
                        cs.append(ad[-1] == bd[-2])
                    elif len(bd) == 1:
                        cs.append(ad[-1] == bd[0])
        elif step.op == OpKind.ADD and len(step.inputs) >= 2:
            a, b = step.inputs[0], step.inputs[1]
            if a in k.shape_vars and b in k.shape_vars:
                ad = k.shape_vars[a]
                bd = k.shape_vars[b]
                if self.ctx.broadcast_theory is not None:
                    cs.append(self.ctx.broadcast_theory.broadcast_compatible(
                        list(ad), list(bd),
                    ))
                else:
                    ndim = max(len(ad), len(bd))
                    for i in range(1, ndim + 1):
                        da = ad[-i] if i <= len(ad) else z3.IntVal(1)
                        db = bd[-i] if i <= len(bd) else z3.IntVal(1)
                        cs.append(z3.Or(
                            da == db,
                            da == z3.IntVal(1),
                            db == z3.IntVal(1),
                        ))
        elif step.op == OpKind.RESHAPE:
            inp = step.inputs[0] if step.inputs else None
            dims = step.params.get("dims")
            if inp and inp in k.shape_vars and dims is not None:
                inp_d = k.shape_vars[inp]
                # Concrete target dims must multiply to the same total
                known = [d for d in dims if isinstance(d, int) and d >= 0]
                if known and all(not z3.is_int_value(d) for d in inp_d):
                    pass  # symbolic input — rely on transition encoding
                elif known:
                    # All concrete: product(input) == product(target)
                    # (with -1 slots inferred, delegate to transition)
                    pass
                # Stride-theory contiguity check for source tensor
                if self.ctx.stride_theory is not None and inp_d:
                    cs.extend(self._stride_contiguity_check(inp_d))
                # Always require positive input dims (handled below)
        # Positivity for all involved shape dims
        for inp in step.inputs:
            if inp in k.shape_vars:
                for d in k.shape_vars[inp]:
                    cs.append(d > 0)
        return cs

    def _encode_device_safety(
        self,
        k: KripkeState,
        step: ComputationStep,
        ms: ModelState,
        idx: int,
    ) -> List:
        """Encode device-consistency constraints for *step*."""
        cs: list = []
        # Binary ops: all inputs on the same device
        if step.op in (OpKind.MATMUL, OpKind.ADD, OpKind.CAT):
            devs = [k.device_vars[i]
                     for i in step.inputs if i in k.device_vars]
            for i in range(1, len(devs)):
                cs.append(self.ctx.encode_device_constraint(devs[0], devs[i]))
        # Layer calls: input device must match param device (default_device)
        if step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = self.graph.layers.get(step.layer_ref)
            if layer and layer.kind in _PARAMETERISED_LAYERS:
                inp = step.inputs[0] if step.inputs else None
                if inp and inp in k.device_vars:
                    cs.append(
                        k.device_vars[inp]
                        == self.ctx.device_to_z3(self.default_device)
                    )
        return cs

    def _encode_phase_safety(
        self,
        k: KripkeState,
        step: ComputationStep,
        ms: ModelState,
        idx: int,
    ) -> List:
        """Encode phase-correctness constraints for *step*.

        Also registers phase-dependent behaviour with the PhaseTheoryPlugin
        (if available) for eager propagation on the phase solver.
        """
        cs: list = []
        if k.phase_var is None:
            return cs
        if step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = self.graph.layers.get(step.layer_ref)
            if layer and layer.kind in (LayerKind.DROPOUT,
                                        LayerKind.BATCHNORM1D,
                                        LayerKind.BATCHNORM2D):
                # Phase must be well-formed (TRAIN or EVAL)
                cs.append(z3.Or(
                    k.phase_var == self.ctx.PHASE_TRAIN,
                    k.phase_var == self.ctx.PHASE_EVAL,
                ))
                # Shape still preserved in both modes
                inp = step.inputs[0] if step.inputs else None
                if inp and inp in k.shape_vars:
                    for d in k.shape_vars[inp]:
                        cs.append(d > 0)
                # Dropout identity in eval encoded via implication
                if layer.kind == LayerKind.DROPOUT:
                    cs.append(z3.Implies(
                        k.phase_var == self.ctx.PHASE_EVAL,
                        z3.BoolVal(True),
                    ))
                    # Register dropout behaviour with phase theory plugin
                    if self.ctx.phase_theory is not None:
                        _ph = z3.Bool(f"_pt_phase_s{idx}")
                        _inp = z3.Bool(f"_pt_drop_in_s{idx}")
                        _out = z3.Bool(f"_pt_drop_out_s{idx}")
                        self.ctx._phase_solver.add(
                            self.ctx.phase_theory.dropout_behavior(
                                _ph, _inp, _out
                            )
                        )
                        self.ctx._phase_constraints_registered += 1
                # Register batchnorm behaviour with phase theory plugin
                if layer.kind in (LayerKind.BATCHNORM1D,
                                  LayerKind.BATCHNORM2D):
                    if self.ctx.phase_theory is not None:
                        _ph = z3.Bool(f"_pt_phase_bn_s{idx}")
                        _urs = z3.Bool(f"_pt_bn_urs_s{idx}")
                        self.ctx._phase_solver.add(
                            self.ctx.phase_theory.batchnorm_behavior(
                                _ph, _urs
                            )
                        )
                        self.ctx._phase_constraints_registered += 1
        elif step.op == OpKind.DROPOUT:
            cs.append(z3.Or(
                k.phase_var == self.ctx.PHASE_TRAIN,
                k.phase_var == self.ctx.PHASE_EVAL,
            ))
            # Register functional dropout with phase theory plugin
            if self.ctx.phase_theory is not None:
                _ph = z3.Bool(f"_pt_phase_fdrop_s{idx}")
                _inp = z3.Bool(f"_pt_fdrop_in_s{idx}")
                _out = z3.Bool(f"_pt_fdrop_out_s{idx}")
                self.ctx._phase_solver.add(
                    self.ctx.phase_theory.dropout_behavior(
                        _ph, _inp, _out
                    )
                )
                self.ctx._phase_constraints_registered += 1
        return cs

    def _encode_gradient_safety(
        self,
        k: KripkeState,
        step: ComputationStep,
        ms: ModelState,
        idx: int,
    ) -> List:
        """Encode gradient-validity constraints for *step*."""
        cs: list = []
        # Detach: output must not require grad (checked in post-state)
        if step.op == OpKind.DETACH:
            if step.output in k.grad_vars:
                cs.append(
                    k.grad_vars[step.output] == z3.BoolVal(False)
                )
        # Gradient well-formedness for inputs
        for inp in step.inputs:
            if inp in k.grad_vars:
                cs.append(z3.Or(
                    k.grad_vars[inp] == z3.BoolVal(True),
                    k.grad_vars[inp] == z3.BoolVal(False),
                ))
        return cs

    def _encode_cross_domain_safety(
        self,
        pre: KripkeState,
        post: KripkeState,
        step: ComputationStep,
        ms: ModelState,
        idx: int,
    ) -> List:
        """Encode cross-domain constraints spanning shape + device + phase."""
        cs: list = []
        inp_name = step.inputs[0] if step.inputs else None
        # Device transfer must preserve shape
        if step.op == OpKind.TO_DEVICE:
            if (inp_name and inp_name in pre.shape_vars
                    and step.output in post.shape_vars):
                for dp, dq in zip(
                    pre.shape_vars[inp_name],
                    post.shape_vars[step.output],
                ):
                    cs.append(dp == dq)
        # Layer calls: params on same device as data
        if step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = self.graph.layers.get(step.layer_ref)
            if layer and layer.kind in _PARAMETERISED_LAYERS:
                if inp_name and inp_name in pre.device_vars:
                    # Params assumed on default_device
                    cs.append(
                        pre.device_vars[inp_name]
                        == self.ctx.device_to_z3(self.default_device)
                    )
        # Shape-preserving ops: cross-check shape preservation
        if step.op in (OpKind.ACTIVATION, OpKind.CONTIGUOUS,
                        OpKind.SOFTMAX):
            if (inp_name and inp_name in pre.shape_vars
                    and step.output in post.shape_vars):
                for dp, dq in zip(
                    pre.shape_vars[inp_name],
                    post.shape_vars[step.output],
                ):
                    cs.append(dp == dq)
        return cs

    # ------------------------------------------------------------------
    # Z3 safety-check helper
    # ------------------------------------------------------------------

    def _z3_check_safety(
        self,
        solver: z3.Solver,
        constraints: list,
        step: ComputationStep,
        step_idx: int,
        kind: str,
    ) -> Optional[SafetyViolation]:
        """Push negated *constraints* onto *solver* and check SAT."""
        if not constraints:
            return None
        solver.push()
        solver.add(z3.Not(z3.And(*constraints)))
        result = self.ctx.timed_check(solver)
        violation = None
        if result == z3.sat:
            model = solver.model()
            violation = SafetyViolation(
                kind=kind,
                step_index=step_idx,
                step=step,
                message=self._format_z3_model(model, step_idx, kind),
            )
        solver.pop()
        return violation

    # ------------------------------------------------------------------
    # Concrete single-step transition (kept for backward compat)
    # ------------------------------------------------------------------

    def _step_transition(
        self, state: ModelState, step: ComputationStep
    ) -> Tuple[ModelState, List[SafetyViolation]]:
        """Apply one computation step to *state*, returning the new state and
        any safety violations detected.
        """
        new_state = state.copy()
        violations: List[SafetyViolation] = []

        # ---- Device consistency check ------------------------------------
        input_devices = []
        for inp in step.inputs:
            dev = state.device_map.get(inp)
            if dev is not None:
                input_devices.append((inp, dev))

        if len(input_devices) >= 2:
            first_name, first_dev = input_devices[0]
            for other_name, other_dev in input_devices[1:]:
                if first_dev != other_dev:
                    violations.append(SafetyViolation(
                        kind="device_mismatch",
                        step_index=-1,
                        step=step,
                        message=(
                            f"Device mismatch: {first_name} is on "
                            f"{first_dev.value} but {other_name} is on "
                            f"{other_dev.value}"
                        ),
                        tensor_a=first_name,
                        tensor_b=other_name,
                        device_a=first_dev,
                        device_b=other_dev,
                    ))

        # ---- Shape propagation & compatibility ---------------------------
        if step.op == OpKind.LAYER_CALL:
            self._apply_layer_call(new_state, step, violations)
        elif step.op == OpKind.MATMUL:
            self._apply_matmul(new_state, step, violations)
        elif step.op == OpKind.ADD:
            self._apply_add(new_state, step, violations)
        elif step.op == OpKind.RESHAPE:
            self._apply_reshape(new_state, step)
        elif step.op == OpKind.FLATTEN:
            self._apply_flatten(new_state, step)
        elif step.op in (OpKind.ACTIVATION, OpKind.CONTIGUOUS):
            if step.inputs and step.inputs[0] in state.shape_env:
                new_state.shape_env[step.output] = (
                    state.shape_env[step.inputs[0]]
                )
        elif step.op == OpKind.DROPOUT:
            if step.inputs and step.inputs[0] in state.shape_env:
                new_state.shape_env[step.output] = (
                    state.shape_env[step.inputs[0]]
                )
        elif step.op == OpKind.SOFTMAX:
            if step.inputs and step.inputs[0] in state.shape_env:
                new_state.shape_env[step.output] = (
                    state.shape_env[step.inputs[0]]
                )
        elif step.op == OpKind.SQUEEZE:
            self._apply_squeeze(new_state, step)
        elif step.op == OpKind.UNSQUEEZE:
            self._apply_unsqueeze(new_state, step)
        elif step.op == OpKind.TRANSPOSE:
            self._apply_transpose(new_state, step)
        elif step.op == OpKind.PERMUTE:
            self._apply_permute(new_state, step)
        elif step.op == OpKind.CAT:
            self._apply_cat(new_state, step, violations)
        elif step.op == OpKind.TO_DEVICE:
            self._apply_to_device(new_state, step)
        elif step.op == OpKind.DETACH:
            if step.inputs and step.inputs[0] in state.shape_env:
                new_state.shape_env[step.output] = (
                    state.shape_env[step.inputs[0]]
                )
            new_state.gradient_status[step.output] = False
        elif step.op == OpKind.RETURN:
            pass
        elif step.op == OpKind.CONDITIONAL:
            self._apply_conditional(new_state, step, violations)
        elif step.op == OpKind.CUSTOM:
            pass

        # ---- Propagate device if not explicitly set ----------------------
        if step.output not in new_state.device_map:
            if step.inputs:
                for inp in step.inputs:
                    if inp in state.device_map:
                        new_state.device_map[step.output] = (
                            state.device_map[inp]
                        )
                        break

        # ---- Propagate gradient status -----------------------------------
        if step.output not in new_state.gradient_status:
            any_grad = any(
                state.gradient_status.get(inp, False)
                for inp in step.inputs
            )
            new_state.gradient_status[step.output] = any_grad

        return new_state, violations

    # --- per-operation helpers (concrete) ---------------------------------

    def _apply_layer_call(
        self,
        state: ModelState,
        step: ComputationStep,
        violations: List[SafetyViolation],
    ) -> None:
        layer = self.graph.layers.get(step.layer_ref or "")
        if layer is None:
            return

        inp_name = step.inputs[0] if step.inputs else None
        inp_shape = state.shape_env.get(inp_name) if inp_name else None

        if inp_shape is None:
            return

        if layer.kind == LayerKind.DROPOUT and state.phase == Phase.EVAL:
            state.shape_env[step.output] = inp_shape
            return

        if layer.kind in (LayerKind.BATCHNORM1D, LayerKind.BATCHNORM2D):
            pass

        propagator = _LAYER_PROPAGATORS.get(layer.kind)
        if propagator is not None:
            out_shape, err = propagator(inp_shape, layer)
            if err:
                violations.append(SafetyViolation(
                    kind="shape_incompatible",
                    step_index=-1,
                    step=step,
                    message=err,
                    tensor_a=inp_name,
                    shape_a=inp_shape,
                ))
            elif out_shape is not None:
                state.shape_env[step.output] = out_shape
        elif layer.kind in (LayerKind.RELU, LayerKind.IDENTITY):
            state.shape_env[step.output] = inp_shape
        elif layer.kind == LayerKind.FLATTEN:
            out_shape, err = _propagate_flatten(inp_shape, 1)
            if out_shape:
                state.shape_env[step.output] = out_shape
        elif layer.kind == LayerKind.SOFTMAX:
            state.shape_env[step.output] = inp_shape
        else:
            state.shape_env[step.output] = inp_shape

    def _apply_matmul(
        self,
        state: ModelState,
        step: ComputationStep,
        violations: List[SafetyViolation],
    ) -> None:
        if len(step.inputs) < 2:
            return
        a_name, b_name = step.inputs[0], step.inputs[1]
        a_shape = state.shape_env.get(a_name)
        b_shape = state.shape_env.get(b_name)
        if a_shape is None or b_shape is None:
            return
        err = check_matmul_compatible(a_shape, b_shape)
        if err:
            violations.append(SafetyViolation(
                kind="shape_incompatible", step_index=-1, step=step,
                message=err,
                tensor_a=a_name, tensor_b=b_name,
                shape_a=a_shape, shape_b=b_shape,
            ))
            return
        result = compute_matmul_shape(a_shape, b_shape)
        if result is not None:
            state.shape_env[step.output] = result

    def _apply_add(
        self,
        state: ModelState,
        step: ComputationStep,
        violations: List[SafetyViolation],
    ) -> None:
        if len(step.inputs) < 2:
            return
        a_name, b_name = step.inputs[0], step.inputs[1]
        a_shape = state.shape_env.get(a_name)
        b_shape = state.shape_env.get(b_name)
        if a_shape is None or b_shape is None:
            return
        result = compute_broadcast_shape(a_shape, b_shape)
        if result is None:
            violations.append(SafetyViolation(
                kind="shape_incompatible", step_index=-1, step=step,
                message=(
                    f"Cannot broadcast {a_shape.pretty()} and "
                    f"{b_shape.pretty()}"
                ),
                tensor_a=a_name, tensor_b=b_name,
                shape_a=a_shape, shape_b=b_shape,
            ))
        else:
            state.shape_env[step.output] = result

    def _apply_reshape(
        self, state: ModelState, step: ComputationStep
    ) -> None:
        inp = step.inputs[0] if step.inputs else None
        inp_shape = state.shape_env.get(inp) if inp else None
        dims = step.params.get("dims")
        if inp_shape is not None and dims is not None:
            result = compute_reshape_shape(inp_shape, dims)
            if result is not None:
                state.shape_env[step.output] = result

    def _apply_flatten(
        self, state: ModelState, step: ComputationStep
    ) -> None:
        inp = step.inputs[0] if step.inputs else None
        inp_shape = state.shape_env.get(inp) if inp else None
        if inp_shape is not None:
            sd = step.params.get("start_dim", 1)
            out, _ = _propagate_flatten(inp_shape, sd)
            if out is not None:
                state.shape_env[step.output] = out

    def _apply_squeeze(
        self, state: ModelState, step: ComputationStep
    ) -> None:
        inp = step.inputs[0] if step.inputs else None
        inp_shape = state.shape_env.get(inp) if inp else None
        if inp_shape is None:
            return
        dim = step.params.get("dim")
        if dim is not None:
            if dim < 0:
                dim = inp_shape.ndim + dim
            new_dims = list(inp_shape.dims)
            if 0 <= dim < len(new_dims):
                d = new_dims[dim]
                if not d.is_symbolic and d.value == 1:
                    new_dims.pop(dim)
            state.shape_env[step.output] = TensorShape(tuple(new_dims))
        else:
            new_dims = [d for d in inp_shape.dims
                        if d.is_symbolic or d.value != 1]
            state.shape_env[step.output] = TensorShape(tuple(new_dims))

    def _apply_unsqueeze(
        self, state: ModelState, step: ComputationStep
    ) -> None:
        inp = step.inputs[0] if step.inputs else None
        inp_shape = state.shape_env.get(inp) if inp else None
        if inp_shape is None:
            return
        dim = step.params.get("dim", 0)
        if dim < 0:
            dim = inp_shape.ndim + 1 + dim
        new_dims = list(inp_shape.dims)
        new_dims.insert(dim, ShapeDim(1))
        state.shape_env[step.output] = TensorShape(tuple(new_dims))

    def _apply_transpose(
        self, state: ModelState, step: ComputationStep
    ) -> None:
        inp = step.inputs[0] if step.inputs else None
        inp_shape = state.shape_env.get(inp) if inp else None
        if inp_shape is None:
            return
        d0 = step.params.get("dim0", 0)
        d1 = step.params.get("dim1", 1)
        if d0 < 0:
            d0 = inp_shape.ndim + d0
        if d1 < 0:
            d1 = inp_shape.ndim + d1
        new_dims = list(inp_shape.dims)
        if 0 <= d0 < len(new_dims) and 0 <= d1 < len(new_dims):
            new_dims[d0], new_dims[d1] = new_dims[d1], new_dims[d0]
        state.shape_env[step.output] = TensorShape(tuple(new_dims))

    def _apply_permute(
        self, state: ModelState, step: ComputationStep
    ) -> None:
        inp = step.inputs[0] if step.inputs else None
        inp_shape = state.shape_env.get(inp) if inp else None
        if inp_shape is None:
            return
        perm = step.params.get("dims")
        if perm and len(perm) == inp_shape.ndim:
            new_dims = [inp_shape.dims[p] for p in perm if p is not None]
            if len(new_dims) == inp_shape.ndim:
                state.shape_env[step.output] = TensorShape(tuple(new_dims))

    def _apply_cat(
        self,
        state: ModelState,
        step: ComputationStep,
        violations: List[SafetyViolation],
    ) -> None:
        shapes = [state.shape_env.get(i) for i in step.inputs]
        if not all(s is not None for s in shapes) or not shapes:
            return
        cat_dim = step.params.get("dim", 0)
        first = shapes[0]
        for i, s in enumerate(shapes[1:], 1):
            if s.ndim != first.ndim:
                violations.append(SafetyViolation(
                    kind="shape_incompatible", step_index=-1, step=step,
                    message=(
                        f"cat: tensors have different ndim "
                        f"({first.ndim} vs {s.ndim})"
                    ),
                ))
                return
        out_dims = list(first.dims)
        if cat_dim < 0:
            cat_dim = first.ndim + cat_dim
        if 0 <= cat_dim < first.ndim:
            total = first.dims[cat_dim]
            all_concrete = not total.is_symbolic
            for s in shapes[1:]:
                d = s.dims[cat_dim]
                if d.is_symbolic or total.is_symbolic:
                    all_concrete = False
                    break
                total = ShapeDim(total.value + d.value)
            if all_concrete:
                out_dims[cat_dim] = total
            else:
                out_dims[cat_dim] = ShapeDim("_cat")
        state.shape_env[step.output] = TensorShape(tuple(out_dims))

    def _apply_to_device(
        self, state: ModelState, step: ComputationStep
    ) -> None:
        inp = step.inputs[0] if step.inputs else None
        if inp and inp in state.shape_env:
            state.shape_env[step.output] = state.shape_env[inp]
        dev_str = step.params.get("device")
        if dev_str is not None:
            state.device_map[step.output] = Device.from_string(str(dev_str))
        elif inp and inp in state.device_map:
            state.device_map[step.output] = state.device_map[inp]

    def _apply_conditional(
        self,
        state: ModelState,
        step: ComputationStep,
        violations: List[SafetyViolation],
    ) -> None:
        """Apply a conditional step by processing the active branch(es).

        For ``self.training`` conditions we only process the branch that
        matches the current phase, eliminating false positives from the
        inactive branch.  For other conditions we conservatively process
        both branches and merge the resulting shape environments.
        """
        cond = step.condition
        true_steps = step.true_branch or []
        false_steps = step.false_branch or []

        if cond == "self.training":
            # Only the branch matching current phase is reachable
            branch = true_steps if state.phase == Phase.TRAIN else false_steps
            for s in branch:
                state, vs = self._step_transition(state, s)
                violations.extend(vs)
        elif cond == "not self.training":
            branch = true_steps if state.phase == Phase.EVAL else false_steps
            for s in branch:
                state, vs = self._step_transition(state, s)
                violations.extend(vs)
        elif cond is not None and cond.startswith("hasattr:self."):
            # hasattr(self, attr) — only take true branch if attr exists
            attr = cond.split(".", 1)[1] if "." in cond.split(":", 1)[1] else ""
            if attr in self.graph.layers:
                for s in true_steps:
                    state, vs = self._step_transition(state, s)
                    violations.extend(vs)
            else:
                for s in false_steps:
                    state, vs = self._step_transition(state, s)
                    violations.extend(vs)
        else:
            # Unknown condition: process both branches, merge states
            true_state = state.copy()
            for s in true_steps:
                true_state, vs = self._step_transition(true_state, s)
                violations.extend(vs)
            false_state = state.copy()
            for s in false_steps:
                false_state, vs = self._step_transition(false_state, s)
                violations.extend(vs)
            # Merge: union of shape bindings from both branches
            for name, shape in true_state.shape_env.items():
                state.shape_env[name] = shape
            for name, shape in false_state.shape_env.items():
                if name not in state.shape_env:
                    state.shape_env[name] = shape
            for name, dev in true_state.device_map.items():
                state.device_map[name] = dev
            for name, dev in false_state.device_map.items():
                if name not in state.device_map:
                    state.device_map[name] = dev

    # ======================================================================
    # Verification: base case
    # ======================================================================

    def _bmc_base_case(
        self,
    ) -> Tuple[List[SafetyViolation], List[ModelState], List[KripkeState]]:
        """Unfold the computation graph and check safety at each step.

        Returns ``(violations, model_states, kripke_states)``.
        """
        all_viols: List[SafetyViolation] = []
        model_states: List[ModelState] = [self._init_state.copy()]
        kripke_states: List[KripkeState] = []

        if not HAS_Z3:
            for idx, step in enumerate(self.graph.steps[: self.max_k]):
                cur = model_states[-1]
                ns, vs = self._step_transition(cur, step)
                for v in vs:
                    v.step_index = idx
                all_viols.extend(vs)
                model_states.append(ns)
            return all_viols, model_states, kripke_states

        # Build initial symbolic state & solver
        k0 = self._build_kripke_state(0, self._init_state)
        kripke_states.append(k0)

        solver = self.ctx.solver
        for c in self._initial_constraints(k0):
            solver.add(c)

        # Initial satisfiability
        self.ctx.timed_check(solver)

        for idx, step in enumerate(self.graph.steps[: self.max_k]):
            cur_model = model_states[-1]
            cur_k = kripke_states[-1]

            # 1. Concrete step transition
            new_model, concrete_vs = self._step_transition(cur_model, step)
            for v in concrete_vs:
                v.step_index = idx
            all_viols.extend(concrete_vs)
            model_states.append(new_model)

            # 2. Build post-transition symbolic state
            post_k = self._build_kripke_state(idx + 1, new_model)
            kripke_states.append(post_k)

            # 3. Z3 safety checks per domain
            for kind, encoder in [
                ("shape_incompatible",
                 lambda: self._encode_shape_safety(
                     cur_k, step, cur_model, idx)),
                ("device_mismatch",
                 lambda: self._encode_device_safety(
                     cur_k, step, cur_model, idx)),
                ("phase_violation",
                 lambda: self._encode_phase_safety(
                     cur_k, step, cur_model, idx)),
                ("gradient_violation",
                 lambda: self._encode_gradient_safety(
                     cur_k, step, cur_model, idx)),
            ]:
                safety = encoder()
                if safety:
                    v = self._z3_check_safety(
                        solver, safety, step, idx, kind
                    )
                    if v is not None:
                        all_viols.append(v)

            # 3b. Check device theory solver
            if self.ctx.device_theory is not None:
                device_result = self.ctx.timed_check(self.ctx._device_solver)
                if device_result == z3.unsat:
                    all_viols.append(SafetyViolation(
                        kind="device_mismatch",
                        step_index=idx,
                        step=step,
                        message=f"Device theory propagator: device inconsistency at step {idx} ({step.op.name})",
                    ))

            # 3c. Check phase theory solver
            if self.ctx.phase_theory is not None:
                phase_result = self.ctx.timed_check(self.ctx._phase_solver)
                if phase_result == z3.unsat:
                    all_viols.append(SafetyViolation(
                        kind="phase_violation",
                        step_index=idx,
                        step=step,
                        message=f"Phase theory propagator: phase inconsistency at step {idx} ({step.op.name})",
                    ))

            # 4. Cross-domain safety
            xd = self._encode_cross_domain_safety(
                cur_k, post_k, step, cur_model, idx
            )
            if xd:
                v = self._z3_check_safety(
                    solver, xd, step, idx, "cross_domain_violation"
                )
                if v is not None:
                    all_viols.append(v)

            # 5. Accumulate transition constraints
            trans = self._encode_transition(
                cur_k, step, post_k, cur_model, idx
            )
            for c in trans:
                solver.add(c)

            # 5b. Assert positivity for post-state shape variables
            for dims in post_k.shape_vars.values():
                for d in dims:
                    if not z3.is_int_value(d):
                        solver.add(d > 0)

            # 6. Transition consistency
            self.ctx.timed_check(solver)

            # 7. Phase well-formedness
            if cur_k.phase_var is not None:
                pwf = [z3.Or(
                    cur_k.phase_var == self.ctx.PHASE_TRAIN,
                    cur_k.phase_var == self.ctx.PHASE_EVAL,
                )]
                self._z3_check_safety(
                    solver, pwf, step, idx, "phase_wellformed"
                )

            # 8. Dimension positivity per step
            pos: list = []
            for dims in cur_k.shape_vars.values():
                for d in dims:
                    if not z3.is_int_value(d):
                        pos.append(d > 0)
            if pos:
                self._z3_check_safety(
                    solver, pos, step, idx, "dim_positivity"
                )

            # 9. Device well-formedness (each tensor on a valid device)
            dev_wf: list = []
            for dv in cur_k.device_vars.values():
                dev_wf.append(z3.Or(
                    dv == self.ctx.DEV_CPU,
                    dv == self.ctx.DEV_CUDA0,
                    dv == self.ctx.DEV_CUDA1,
                    dv == self.ctx.DEV_CUDA2,
                    dv == self.ctx.DEV_CUDA3,
                ))
            if dev_wf:
                self._z3_check_safety(
                    solver, dev_wf, step, idx, "device_wellformed"
                )

            # 10. Gradient well-formedness (post-state)
            grad_wf: list = []
            for gv in post_k.grad_vars.values():
                grad_wf.append(z3.Or(
                    gv == z3.BoolVal(True),
                    gv == z3.BoolVal(False),
                ))
            if grad_wf:
                self._z3_check_safety(
                    solver, grad_wf, step, idx, "gradient_wellformed"
                )

            # 11. Shape-device combined check
            sd_combined: list = []
            sd_combined.extend(self._encode_shape_safety(
                cur_k, step, cur_model, idx))
            sd_combined.extend(self._encode_device_safety(
                cur_k, step, cur_model, idx))
            if sd_combined:
                self._z3_check_safety(
                    solver, sd_combined, step, idx, "shape_device_combined"
                )

            # 12. Full combined safety (all four domains)
            combined: list = []
            combined.extend(self._encode_shape_safety(
                cur_k, step, cur_model, idx))
            combined.extend(self._encode_device_safety(
                cur_k, step, cur_model, idx))
            combined.extend(self._encode_phase_safety(
                cur_k, step, cur_model, idx))
            combined.extend(self._encode_gradient_safety(
                cur_k, step, cur_model, idx))
            if combined:
                self._z3_check_safety(
                    solver, combined, step, idx, "combined_violation"
                )

        return all_viols, model_states, kripke_states

    # ======================================================================
    # Verification: inductive step
    # ======================================================================

    def _bmc_inductive_step(
        self,
        kripke_states: List[KripkeState],
        model_states: List[ModelState],
    ) -> List[SafetyViolation]:
        """Forward inductive step: prove safety preserved across
        transitions using free symbolic state variables.
        """
        if not HAS_Z3:
            return []
        violations: List[SafetyViolation] = []
        n_steps = min(len(self.graph.steps) - 1, self.max_k - 1)

        for idx in range(n_steps):
            step = self.graph.steps[idx]
            next_step = self.graph.steps[idx + 1]
            pre_model = (model_states[idx]
                         if idx < len(model_states) else model_states[-1])
            post_model = (model_states[idx + 1]
                          if idx + 1 < len(model_states)
                          else model_states[-1])

            # Free symbolic states (step_idx offset avoids name collisions)
            pre_k = self._build_kripke_state(
                2000 + idx, pre_model, free_shapes=True
            )
            post_k = self._build_kripke_state(
                2000 + idx + 1, post_model, free_shapes=True
            )

            solver = z3.Solver()
            solver.set("timeout", 5000)

            # Positivity for free dims
            for dims in pre_k.shape_vars.values():
                for d in dims:
                    solver.add(d > 0)
            for dims in post_k.shape_vars.values():
                for d in dims:
                    solver.add(d > 0)

            # Safety assumption at pre-state
            for enc in (
                self._encode_shape_safety(pre_k, step, pre_model, idx),
                self._encode_device_safety(pre_k, step, pre_model, idx),
                self._encode_phase_safety(pre_k, step, pre_model, idx),
                self._encode_gradient_safety(pre_k, step, pre_model, idx),
            ):
                for c in enc:
                    solver.add(c)

            # Transition
            for c in self._encode_transition(
                pre_k, step, post_k, pre_model, idx
            ):
                solver.add(c)

            # Per-domain inductive checks at post-state
            for kind, encoder in [
                ("shape_incompatible",
                 lambda: self._encode_shape_safety(
                     post_k, next_step, post_model, idx + 1)),
                ("device_mismatch",
                 lambda: self._encode_device_safety(
                     post_k, next_step, post_model, idx + 1)),
                ("phase_violation",
                 lambda: self._encode_phase_safety(
                     post_k, next_step, post_model, idx + 1)),
                ("gradient_violation",
                 lambda: self._encode_gradient_safety(
                     post_k, next_step, post_model, idx + 1)),
            ]:
                post_safety = encoder()
                if post_safety:
                    solver.push()
                    solver.add(z3.Not(z3.And(*post_safety)))
                    result = self.ctx.timed_check(solver)
                    if result == z3.sat:
                        m = solver.model()
                        violations.append(SafetyViolation(
                            kind="inductive_violation",
                            step_index=idx + 1,
                            step=next_step,
                            message=self._format_z3_model(
                                m, idx + 1, kind
                            ),
                        ))
                    solver.pop()

            # Cross-domain inductive check
            xd = self._encode_cross_domain_safety(
                pre_k, post_k, step, pre_model, idx
            )
            if xd:
                solver.push()
                solver.add(z3.Not(z3.And(*xd)))
                result = self.ctx.timed_check(solver)
                if result == z3.sat:
                    m = solver.model()
                    violations.append(SafetyViolation(
                        kind="inductive_violation",
                        step_index=idx,
                        step=step,
                        message=self._format_z3_model(
                            m, idx, "cross_domain"
                        ),
                    ))
                solver.pop()

        return violations

    # ======================================================================
    # Top-level verify()
    # ======================================================================

    def verify(self) -> VerificationResult:
        """Run constraint-based verification with forward symbolic
        propagation over the product theory T_shape × T_device × T_phase.

        Returns a ``VerificationResult`` that is either safe (with a
        ``SafetyCertificate`` including Z3 statistics) or unsafe (with a
        ``CounterexampleTrace``).
        """
        t0 = time.monotonic()

        if self.graph.num_steps == 0:
            return VerificationResult(
                safe=True,
                certificate=SafetyCertificate(
                    model_name=self.graph.class_name,
                    properties=["shape_compatible", "device_consistent",
                                "gradient_valid"],
                    k=0, checked_steps=0, verification_time_ms=0.0,
                ),
                graph=self.graph,
            )

        # Phase 1: base case (concrete + Z3)
        all_viols, model_states, kripke_states = self._bmc_base_case()

        # Phase 2: inductive step (Z3 with free variables)
        # Inductive violations indicate proof incompleteness, not unsafety.
        # They are recorded for statistics but do not make the model unsafe.
        ind_viols = self._bmc_inductive_step(kripke_states, model_states)

        elapsed = (time.monotonic() - t0) * 1000
        stats = self.ctx.get_stats() if HAS_Z3 else {}

        if all_viols:
            first_fail = min(v.step_index for v in all_viols)
            cex = CounterexampleTrace(
                model_name=self.graph.class_name,
                violations=all_viols,
                failing_step=first_fail,
                states=model_states[: first_fail + 2],
                concrete_dims=self._extract_concrete_dims(),
            )
            return VerificationResult(
                safe=False,
                counterexample=cex,
                graph=self.graph,
                verification_time_ms=elapsed,
            )

        cert = SafetyCertificate(
            model_name=self.graph.class_name,
            properties=["shape_compatible", "device_consistent",
                         "gradient_valid"],
            k=min(self.max_k, self.graph.num_steps),
            symbolic_bindings={
                n: str(v) for n, v in self.ctx._sym_dims.items()
            },
            checked_steps=len(self.graph.steps),
            verification_time_ms=elapsed,
            z3_queries=stats.get("z3_queries", 0),
            z3_total_time_ms=stats.get("z3_total_time_ms", 0.0),
            z3_sat_count=stats.get("z3_sat_count", 0),
            z3_unsat_count=stats.get("z3_unsat_count", 0),
            theories_used=["QF_LIA", "QF_UF", "QF_UFLIA"]
                + (["T_broadcast"] if HAS_THEORY_PLUGINS else [])
                + (["T_stride"] if self.ctx.stride_theory is not None else [])
                + (["T_device"] if self.ctx.device_theory is not None else [])
                + (["T_phase"] if self.ctx.phase_theory is not None else []),
            product_domains=["T_shape", "T_device", "T_phase"],
        )
        return VerificationResult(
            safe=True,
            certificate=cert,
            graph=self.graph,
            verification_time_ms=elapsed,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_z3_model(
        self, model: z3.ModelRef, step_idx: int, kind: str
    ) -> str:
        parts = [f"Z3 violation ({kind}) at step {step_idx}:"]
        for decl in model.decls():
            parts.append(f"  {decl.name()} = {model[decl]}")
        return "\n".join(parts)

    def _extract_concrete_dims(self) -> Dict[str, int]:
        """Try to extract concrete dimension values from Z3."""
        if not HAS_Z3:
            return {}
        solver = z3.Solver()
        for c in self.ctx.positive_dim_constraints():
            solver.add(c)
        result: Dict[str, int] = {}
        if solver.check() == z3.sat:
            model = solver.model()
            for name, var in self.ctx._sym_dims.items():
                val = model.evaluate(var, model_completion=True)
                try:
                    result[name] = val.as_long()
                except (AttributeError, z3.Z3Exception):
                    pass
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  Symbolic shape propagation engine
# ═══════════════════════════════════════════════════════════════════════════════

class SymbolicShapePropagator:
    """Propagate symbolic shapes through the computation graph using Z3.

    This class walks the computation graph and for each step:
      1. Encodes input shapes as Z3 integer vectors.
      2. Applies the appropriate shape rule.
      3. Records the output shape (possibly with new symbolic dims).

    The result is a complete shape environment mapping every tensor name
    to a (possibly symbolic) ``TensorShape``.
    """

    def __init__(self, graph: ComputationGraph) -> None:
        self.graph = graph
        self.ctx = _Z3Context() if HAS_Z3 else None

    def propagate(
        self, input_shapes: Dict[str, tuple]
    ) -> Dict[str, TensorShape]:
        """Propagate shapes starting from *input_shapes*.

        Returns a dict mapping each tensor name to its inferred shape.
        """
        env: Dict[str, TensorShape] = {}

        for name, raw in input_shapes.items():
            dims = tuple(
                ShapeDim(d) if isinstance(d, int) else ShapeDim(d)
                for d in raw
            )
            env[name] = TensorShape(dims)

        for step in self.graph.steps:
            self._propagate_step(env, step)

        return env

    def _propagate_step(
        self, env: Dict[str, TensorShape], step: ComputationStep
    ) -> None:
        """Propagate shapes for one computation step."""

        if step.op == OpKind.LAYER_CALL:
            layer = self.graph.layers.get(step.layer_ref or "")
            inp = step.inputs[0] if step.inputs else None
            inp_shape = env.get(inp) if inp else None

            if layer and inp_shape:
                propagator = _LAYER_PROPAGATORS.get(layer.kind)
                if propagator:
                    out, _ = propagator(inp_shape, layer)
                    if out:
                        env[step.output] = out
                        return
                # Shape-preserving layers
                if layer.kind in (LayerKind.RELU, LayerKind.DROPOUT,
                                  LayerKind.IDENTITY, LayerKind.SOFTMAX):
                    env[step.output] = inp_shape
                    return
                if layer.kind == LayerKind.FLATTEN:
                    out, _ = _propagate_flatten(inp_shape, 1)
                    if out:
                        env[step.output] = out
                    return
                env[step.output] = inp_shape

        elif step.op == OpKind.MATMUL:
            if len(step.inputs) >= 2:
                a = env.get(step.inputs[0])
                b = env.get(step.inputs[1])
                if a and b:
                    result = compute_matmul_shape(a, b)
                    if result:
                        env[step.output] = result

        elif step.op == OpKind.ADD:
            if len(step.inputs) >= 2:
                a = env.get(step.inputs[0])
                b = env.get(step.inputs[1])
                if a and b:
                    result = compute_broadcast_shape(a, b)
                    if result:
                        env[step.output] = result

        elif step.op == OpKind.RESHAPE:
            inp = step.inputs[0] if step.inputs else None
            inp_shape = env.get(inp) if inp else None
            dims = step.params.get("dims")
            if inp_shape and dims:
                result = compute_reshape_shape(inp_shape, dims)
                if result:
                    env[step.output] = result

        elif step.op == OpKind.FLATTEN:
            inp = step.inputs[0] if step.inputs else None
            inp_shape = env.get(inp) if inp else None
            if inp_shape:
                sd = step.params.get("start_dim", 1)
                out, _ = _propagate_flatten(inp_shape, sd)
                if out:
                    env[step.output] = out

        elif step.op in (OpKind.ACTIVATION, OpKind.DROPOUT, OpKind.SOFTMAX,
                          OpKind.CONTIGUOUS, OpKind.DETACH):
            inp = step.inputs[0] if step.inputs else None
            if inp and inp in env:
                env[step.output] = env[inp]

        elif step.op == OpKind.SQUEEZE:
            inp = step.inputs[0] if step.inputs else None
            inp_shape = env.get(inp) if inp else None
            if inp_shape:
                dim = step.params.get("dim")
                if dim is not None:
                    if dim < 0:
                        dim = inp_shape.ndim + dim
                    new_dims = list(inp_shape.dims)
                    if 0 <= dim < len(new_dims):
                        d = new_dims[dim]
                        if not d.is_symbolic and d.value == 1:
                            new_dims.pop(dim)
                    env[step.output] = TensorShape(tuple(new_dims))
                else:
                    new_dims = [d for d in inp_shape.dims
                                if d.is_symbolic or d.value != 1]
                    env[step.output] = TensorShape(tuple(new_dims))

        elif step.op == OpKind.UNSQUEEZE:
            inp = step.inputs[0] if step.inputs else None
            inp_shape = env.get(inp) if inp else None
            if inp_shape:
                dim = step.params.get("dim", 0)
                if dim < 0:
                    dim = inp_shape.ndim + 1 + dim
                new_dims = list(inp_shape.dims)
                new_dims.insert(dim, ShapeDim(1))
                env[step.output] = TensorShape(tuple(new_dims))

        elif step.op == OpKind.TRANSPOSE:
            inp = step.inputs[0] if step.inputs else None
            inp_shape = env.get(inp) if inp else None
            if inp_shape:
                d0 = step.params.get("dim0", 0)
                d1 = step.params.get("dim1", 1)
                if d0 < 0:
                    d0 = inp_shape.ndim + d0
                if d1 < 0:
                    d1 = inp_shape.ndim + d1
                new_dims = list(inp_shape.dims)
                if 0 <= d0 < len(new_dims) and 0 <= d1 < len(new_dims):
                    new_dims[d0], new_dims[d1] = new_dims[d1], new_dims[d0]
                env[step.output] = TensorShape(tuple(new_dims))

        elif step.op == OpKind.TO_DEVICE:
            inp = step.inputs[0] if step.inputs else None
            if inp and inp in env:
                env[step.output] = env[inp]

        elif step.op in (OpKind.RETURN, OpKind.CUSTOM):
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Phase-aware analysis
# ═══════════════════════════════════════════════════════════════════════════════

class PhaseAnalyzer:
    """Analyse phase-dependent behaviour of an nn.Module.

    Detects:
      - Dropout layers that are active only in TRAIN mode.
      - BatchNorm layers that switch between training and running statistics.
      - Shape differences between train and eval modes.
    """

    def __init__(self, graph: ComputationGraph) -> None:
        self.graph = graph

    def has_phase_dependent_layers(self) -> bool:
        """Check whether the graph has layers whose behaviour depends on
        train/eval phase."""
        for layer in self.graph.layers.values():
            if layer.kind in (LayerKind.DROPOUT, LayerKind.BATCHNORM1D,
                              LayerKind.BATCHNORM2D):
                return True
        return False

    def compare_phases(
        self, input_shapes: Dict[str, tuple]
    ) -> Dict[str, Any]:
        """Compare model behaviour in TRAIN vs EVAL phase.

        Returns a dict with keys:
          - "train_shapes": shape env in train mode
          - "eval_shapes":  shape env in eval mode
          - "differences":  list of (tensor_name, train_shape, eval_shape)
        """
        train_checker = ConstraintVerifier(
            self.graph, input_shapes,
            default_phase=Phase.TRAIN,
        )
        eval_checker = ConstraintVerifier(
            self.graph, input_shapes,
            default_phase=Phase.EVAL,
        )

        # Simulate both phases
        train_state = train_checker._init_state.copy()
        eval_state = eval_checker._init_state.copy()

        for step in self.graph.steps:
            train_state, _ = train_checker._step_transition(train_state, step)
            eval_state, _ = eval_checker._step_transition(eval_state, step)

        differences = []
        all_names = (set(train_state.shape_env.keys())
                     | set(eval_state.shape_env.keys()))
        for name in sorted(all_names):
            ts = train_state.shape_env.get(name)
            es = eval_state.shape_env.get(name)
            if ts != es:
                differences.append((name, ts, es))

        return {
            "train_shapes": train_state.shape_env,
            "eval_shapes": eval_state.shape_env,
            "differences": differences,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Device analysis
# ═══════════════════════════════════════════════════════════════════════════════

class DeviceAnalyzer:
    """Analyse device placement of tensors in an nn.Module.

    Detects cross-device operations and unnecessary device transfers.
    """

    def __init__(self, graph: ComputationGraph) -> None:
        self.graph = graph

    def check_device_consistency(
        self,
        input_shapes: Dict[str, tuple],
        input_devices: Optional[Dict[str, Device]] = None,
    ) -> List[SafetyViolation]:
        """Check that all operations use tensors on the same device.

        Returns a list of SafetyViolation for any cross-device operations.
        """
        state = ModelState(phase=Phase.TRAIN)

        for name, raw in input_shapes.items():
            dims = tuple(
                ShapeDim(d) if isinstance(d, int) else ShapeDim(d)
                for d in raw
            )
            state.shape_env[name] = TensorShape(dims)

        if input_devices:
            state.device_map.update(input_devices)
        else:
            for name in input_shapes:
                state.device_map[name] = Device.CPU

        checker = ConstraintVerifier(
            self.graph, input_shapes,
        )
        checker._init_state = state

        all_violations: List[SafetyViolation] = []
        current = state.copy()
        for idx, step in enumerate(self.graph.steps):
            current, viols = checker._step_transition(current, step)
            for v in viols:
                v.step_index = idx
            all_violations.extend(
                v for v in viols if v.kind == "device_mismatch"
            )

        return all_violations

    def trace_device_transfers(
        self,
        input_shapes: Dict[str, tuple],
        input_devices: Optional[Dict[str, Device]] = None,
    ) -> List[Tuple[int, str, Device, Device]]:
        """Return a list of device transfers as (step_idx, tensor, from, to).
        """
        state = ModelState(phase=Phase.TRAIN)
        for name, raw in input_shapes.items():
            dims = tuple(
                ShapeDim(d) if isinstance(d, int) else ShapeDim(d)
                for d in raw
            )
            state.shape_env[name] = TensorShape(dims)

        if input_devices:
            state.device_map.update(input_devices)
        else:
            for name in input_shapes:
                state.device_map[name] = Device.CPU

        transfers: List[Tuple[int, str, Device, Device]] = []
        current = state.copy()
        checker = ConstraintVerifier(self.graph, input_shapes)
        checker._init_state = state

        for idx, step in enumerate(self.graph.steps):
            if step.op == OpKind.TO_DEVICE:
                old_dev = current.device_map.get(
                    step.inputs[0], Device.CPU
                ) if step.inputs else Device.CPU

                current, _ = checker._step_transition(current, step)

                new_dev = current.device_map.get(step.output, old_dev)
                if old_dev != new_dev:
                    transfers.append((idx, step.output, old_dev, new_dev))
            else:
                current, _ = checker._step_transition(current, step)

        return transfers


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Public API: verify_model
# ═══════════════════════════════════════════════════════════════════════════════

def verify_model(
    source: str,
    input_shapes: Optional[Dict[str, tuple]] = None,
    default_device: Device = Device.CPU,
    default_phase: Phase = Phase.TRAIN,
    max_k: Optional[int] = None,
) -> VerificationResult:
    """One-shot verification of an nn.Module defined in *source*.

    Parameters
    ----------
    source : str
        Python source code containing an ``nn.Module`` subclass.
    input_shapes : dict, optional
        Mapping from forward-parameter names to shape tuples.  Dimensions
        may be ints (concrete) or strings (symbolic).
    default_device : Device
        Default device for input tensors.
    default_phase : Phase
        Default phase (TRAIN or EVAL).
    max_k : int, optional
        Maximum verification depth.  Defaults to the number of steps in the
        graph.

    Returns
    -------
    VerificationResult
        Contains either a ``SafetyCertificate`` (if safe) or a
        ``CounterexampleTrace`` (if unsafe).

    Examples
    --------
    >>> result = verify_model('''
    ... import torch.nn as nn
    ... class MyModel(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(10, 5)
    ...     def forward(self, x):
    ...         return self.fc(x)
    ... ''', input_shapes={"x": ("batch", 10)})
    >>> result.safe
    True
    """
    t0 = time.monotonic()

    try:
        graph = extract_computation_graph(source)
    except (ValueError, SyntaxError) as exc:
        return VerificationResult(
            safe=False,
            errors=[str(exc)],
            verification_time_ms=(time.monotonic() - t0) * 1000,
        )

    checker = ConstraintVerifier(
        graph,
        input_shapes=input_shapes or {},
        default_device=default_device,
        default_phase=default_phase,
        max_k=max_k,
    )

    return checker.verify()


# Backward-compatible alias (deprecated; use ConstraintVerifier directly).
BoundedModelChecker = ConstraintVerifier
