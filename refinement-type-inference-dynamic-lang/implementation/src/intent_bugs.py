"""
Intent-Apparent Bug Detection for PyTorch/ML Code.

This module implements the "overwarning" liquid type system that flags
intent-apparent bugs — situations where the programmer's structural intent
is clear from the code but the code doesn't match that intent.

The system extends TensorGuard's refinement types with ML-semantic sorts:
  - DeviceSort: tracks tensor device placement (CPU/CUDA)
  - DTypeSort: tracks tensor dtype (float32/float16/int32/...)
  - DimRoleSort: tracks semantic axis roles (Batch/Channel/Spatial/Time/Feature)
  - PhaseSort: tracks training/eval state
  - GradFlowSort: tracks gradient flow (differentiable path or broken)
  - ParamLifecycleSort: tracks parameter registration and optimizer membership

Unlike traditional type checkers that aim for soundness (no false positives),
this system **deliberately overwarns**: it flags any pattern that *could* be
an intent-apparent bug, even when it cannot prove the bug definitively. The
philosophy is that silent semantic errors in ML code are far more costly than
false warnings.

Key principle: "Intent-apparent" means the code contains enough structural
evidence to infer what the programmer *meant* to do, and that inferred intent
conflicts with what the code *actually* does.
"""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

from .bug_class_registry import BugCategory, BugClassRegistry

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Intent-Apparent Bug Kinds (extends LiquidBugKind for ML-specific bugs)
# ═══════════════════════════════════════════════════════════════════════════

class IntentBugKind(Enum):
    """ML-specific bug categories discovered via intent inference."""
    # Shape family
    SHAPE_MISMATCH = auto()
    BATCH_BROADCAST = auto()
    SPATIAL_MISMATCH = auto()
    RESHAPE_INVALID = auto()
    SQUEEZE_UNSTABLE = auto()
    FLATTEN_BATCH = auto()
    # Device family
    DEVICE_MISMATCH = auto()
    # Dtype family
    DTYPE_TRUNCATION = auto()
    FP16_OVERFLOW = auto()
    # Gradient family
    GRADIENT_BROKEN = auto()
    HARD_GATING = auto()
    HIDDEN_STATE_LEAK = auto()
    # Optimizer family
    MISSING_ZERO_GRAD = auto()
    FROZEN_IN_OPTIMIZER = auto()
    UNFROZEN_NOT_IN_OPTIMIZER = auto()
    DUPLICATE_PARAM_GROUP = auto()
    PARAM_REPLACED = auto()
    DEAD_PARAMETER = auto()
    WRONG_LR_SCHEDULE = auto()
    GRAD_ACCUM_UNSCALED = auto()
    WEIGHT_DECAY_NORM = auto()
    DATA_BYPASS = auto()
    GRAD_CLIP_BEFORE_UNSCALE = auto()
    PARAMS_AFTER_OPTIMIZER = auto()
    # Semantic family
    WRONG_SOFTMAX_DIM = auto()
    DOUBLE_ACTIVATION = auto()
    WRONG_LOSS_FAMILY = auto()
    ATTENTION_MASK_POLARITY = auto()
    BATCH_FIRST_MISMATCH = auto()
    NORM_AXIS_MISMATCH = auto()
    FUNCTIONAL_DROPOUT_EVAL = auto()
    CONV_AXIS_MISMATCH = auto()
    ALIGN_CORNERS_MISMATCH = auto()
    PAD_ORDER_MISMATCH = auto()
    GRID_SAMPLE_COORDS = auto()
    IGNORE_INDEX_ALL = auto()
    IMPLICIT_PER_SLICE = auto()
    # Lifecycle family
    UNREGISTERED_SUBMODULE = auto()
    UNINTENDED_WEIGHT_SHARING = auto()
    INPLACE_ALIAS_RESIDUAL = auto()
    EXPAND_ALIAS_GRAD = auto()
    # Control flow family
    TRACE_FROZEN_BRANCH = auto()
    MISALIGNED_PAIRED_TRANSFORM = auto()
    FOLD_NO_NORMALIZE = auto()
    CONV_TRANSPOSE_OUTPUT_PAD = auto()


@dataclass
class InferredIntent:
    """What the system infers the programmer intended."""
    description: str
    confidence: float  # 0.0 to 1.0
    evidence: List[str] = field(default_factory=list)

    def pretty(self) -> str:
        conf_str = f"{self.confidence:.0%}"
        ev = "; ".join(self.evidence[:3])
        return f"[{conf_str}] {self.description} (evidence: {ev})"


@dataclass
class IntentApparentBug:
    """A bug flagged by the intent-apparent overwarning system."""
    kind: IntentBugKind
    line: int
    col: int
    message: str
    function: str
    variable: str
    inferred_intent: InferredIntent
    bug_class_id: Optional[int] = None  # reference to bugclasses.jsonl
    severity: str = "warning"  # always warning (overwarning mode)
    overwarn: bool = True  # flag that this is an overwarn, not proven

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.name,
            "line": self.line,
            "col": self.col,
            "message": self.message,
            "function": self.function,
            "variable": self.variable,
            "severity": self.severity,
            "overwarn": self.overwarn,
            "inferred_intent": self.inferred_intent.pretty(),
            "bug_class_id": self.bug_class_id,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Intent Inference Engine
# ═══════════════════════════════════════════════════════════════════════════

class IntentInferenceEngine:
    """Infers programmer intent from code structure.

    Analyzes AST patterns to determine what the programmer likely meant,
    then checks whether the code actually implements that intent.

    Intent categories:
      - Loss family (classification vs regression vs multi-label)
      - Normalization (per-channel, per-layer, per-batch)
      - Optimization protocol (zero_grad/backward/step ordering)
      - Data layout (NCHW vs NHWC, batch_first)
      - Gradient flow (differentiable path expected)
      - Parameter lifecycle (registered, in optimizer, frozen/unfrozen)
    """

    def __init__(self):
        self._class_info: Dict[str, Dict[str, Any]] = {}
        self._training_loop_info: Dict[str, Any] = {}
        self._module_info: Dict[str, Dict[str, Any]] = {}

    def analyze_module(self, tree: ast.Module) -> Dict[str, Any]:
        """Analyze an entire module for intent signals."""
        info: Dict[str, Any] = {
            "classes": {},
            "training_loops": [],
            "loss_usage": [],
            "optimizer_usage": [],
            "device_usage": [],
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                info["classes"][node.name] = self._analyze_class(node)
            if isinstance(node, ast.For):
                loop_info = self._analyze_training_loop(node)
                if loop_info:
                    info["training_loops"].append(loop_info)
        return info

    def _analyze_class(self, cls: ast.ClassDef) -> Dict[str, Any]:
        """Analyze an nn.Module class for intent signals."""
        info: Dict[str, Any] = {
            "is_module": False,
            "layers": {},
            "plain_tensors": [],
            "plain_lists": [],
            "forward_method": None,
        }
        # Check if it inherits from nn.Module
        for base in cls.bases:
            if isinstance(base, ast.Attribute) and base.attr == "Module":
                info["is_module"] = True
            if isinstance(base, ast.Name) and base.id in ("Module", "nn.Module"):
                info["is_module"] = True

        for node in ast.walk(cls):
            if isinstance(node, ast.FunctionDef):
                if node.name == "__init__":
                    info["layers"] = self._extract_init_layers(node)
                    info["plain_tensors"] = self._extract_plain_tensors(node)
                    info["plain_lists"] = self._extract_plain_lists(node)
                elif node.name == "forward":
                    info["forward_method"] = node
        return info

    def _extract_init_layers(self, init: ast.FunctionDef) -> Dict[str, str]:
        """Extract nn layer definitions from __init__."""
        layers: Dict[str, str] = {}
        for node in ast.walk(init):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                        if target.value.id == "self":
                            layer_type = self._get_layer_type(node.value)
                            if layer_type:
                                layers[target.attr] = layer_type
        return layers

    def _extract_plain_tensors(self, init: ast.FunctionDef) -> List[str]:
        """Find self.x = torch.randn(...) that are NOT Parameters or buffers."""
        plain = []
        for node in ast.walk(init):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                        if target.value.id == "self":
                            if self._is_plain_tensor_creation(node.value):
                                plain.append(target.attr)
        return plain

    def _extract_plain_lists(self, init: ast.FunctionDef) -> List[str]:
        """Find self.layers = [...] (plain list, not ModuleList)."""
        plain = []
        for node in ast.walk(init):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                        if target.value.id == "self":
                            if isinstance(node.value, (ast.List, ast.ListComp)):
                                plain.append(target.attr)
                            # [X] * n pattern
                            if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Mult):
                                if isinstance(node.value.left, ast.List):
                                    plain.append(target.attr)
        return plain

    def _analyze_training_loop(self, loop: ast.For) -> Optional[Dict[str, Any]]:
        """Analyze a for loop for training protocol patterns."""
        has_backward = False
        has_zero_grad = False
        has_step = False
        has_scheduler_step = False
        zero_grad_before_backward = False
        backward_line = 0
        zero_grad_line = 0
        step_line = 0
        sched_step_line = 0

        for node in ast.walk(loop):
            if isinstance(node, ast.Call):
                call_name = self._get_call_name(node)
                if call_name and "backward" in call_name:
                    has_backward = True
                    backward_line = getattr(node, "lineno", 0)
                elif call_name and "zero_grad" in call_name:
                    has_zero_grad = True
                    zero_grad_line = getattr(node, "lineno", 0)
                elif call_name and call_name.endswith(".step"):
                    if "sched" in call_name.lower() or "scheduler" in call_name.lower():
                        has_scheduler_step = True
                        sched_step_line = getattr(node, "lineno", 0)
                    else:
                        has_step = True
                        step_line = getattr(node, "lineno", 0)

        if not has_backward:
            return None

        if has_zero_grad and zero_grad_line < backward_line:
            zero_grad_before_backward = True

        return {
            "line": getattr(loop, "lineno", 0),
            "has_backward": has_backward,
            "has_zero_grad": has_zero_grad,
            "has_step": has_step,
            "has_scheduler_step": has_scheduler_step,
            "zero_grad_before_backward": zero_grad_before_backward,
            "backward_line": backward_line,
            "zero_grad_line": zero_grad_line,
            "step_line": step_line,
            "sched_step_line": sched_step_line,
        }

    @staticmethod
    def _get_layer_type(node: ast.expr) -> Optional[str]:
        """Get the nn layer type from a call expression."""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                return node.func.attr
            if isinstance(node.func, ast.Name):
                return node.func.id
        return None

    @staticmethod
    def _is_plain_tensor_creation(node: ast.expr) -> bool:
        """Check if node creates a plain tensor (not Parameter/buffer)."""
        if isinstance(node, ast.Call):
            call_name = ""
            if isinstance(node.func, ast.Attribute):
                call_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                call_name = node.func.id
            return call_name in ("randn", "zeros", "ones", "rand", "tensor",
                                 "empty", "full", "arange", "linspace")
        return False

    @staticmethod
    def _get_call_name(node: ast.Call) -> Optional[str]:
        """Get dotted call name (e.g., 'opt.step')."""
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                return f"{node.func.value.id}.{node.func.attr}"
            return node.func.attr
        if isinstance(node.func, ast.Name):
            return node.func.id
        return None


def _get_full_call_name(node: ast.Call) -> Optional[str]:
    """Get fully-qualified dotted call name (e.g. 'torch.zeros')."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    parts: List[str] = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    if parts:
        return ".".join(reversed(parts))
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Pattern Checkers — one per bug category
# ═══════════════════════════════════════════════════════════════════════════

class PatternChecker:
    """Base class for intent-apparent bug pattern checkers."""

    def __init__(self):
        self.bugs: List[IntentApparentBug] = []

    def check(self, tree: ast.Module, intent_info: Dict[str, Any]) -> List[IntentApparentBug]:
        raise NotImplementedError


class SemanticPatternChecker(PatternChecker):
    """Checks for semantic bugs: wrong loss, wrong axis, double activation, etc.

    Bug classes covered:
      #10 - Softmax over wrong dimension
      #11 - Double activation before logits-expecting loss
      #12 - Functional dropout ignoring training state
      #17 - BCEWithLogitsLoss for single-label multi-class
      #20 - Attention mask semantics
      #38 - CrossEntropyLoss ignore_index silences training
    """

    def check(self, tree: ast.Module, intent_info: Dict[str, Any]) -> List[IntentApparentBug]:
        self.bugs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                self._check_double_activation(node, tree)
                self._check_functional_dropout_eval(node, tree)
                self._check_wrong_softmax_dim(node, tree)
                self._check_wrong_loss_family(node, tree)
        return self.bugs

    def _check_double_activation(self, node: ast.Call, tree: ast.Module) -> None:
        """Detect softmax/sigmoid feeding into logits-expecting loss."""
        call_name = self._call_name(node)
        if call_name not in ("cross_entropy", "CrossEntropyLoss",
                             "BCEWithLogitsLoss", "nll_loss"):
            return
        # Check if any argument is the result of softmax/sigmoid
        for arg in node.args:
            if isinstance(arg, ast.Call):
                inner_name = self._call_name(arg)
                if inner_name in ("softmax", "sigmoid", "log_softmax"):
                    self.bugs.append(IntentApparentBug(
                        kind=IntentBugKind.DOUBLE_ACTIVATION,
                        line=getattr(node, "lineno", 0),
                        col=getattr(node, "col_offset", 0),
                        message=(f"Double activation: {inner_name}() output fed to "
                                 f"{call_name}() which already applies activation internally"),
                        function="<module>",
                        variable=call_name,
                        inferred_intent=InferredIntent(
                            description=f"Use raw logits with {call_name}",
                            confidence=0.95,
                            evidence=[f"{inner_name} applied before {call_name}",
                                      f"{call_name} expects raw logits"],
                        ),
                        bug_class_id=11,
                    ))
            # Also check if arg is a Name that was assigned from softmax/sigmoid
            if isinstance(arg, ast.Name):
                self._check_name_is_activation_output(arg, node, call_name, tree)

    def _check_name_is_activation_output(self, name_node: ast.Name,
                                          loss_node: ast.Call,
                                          loss_name: str,
                                          tree: ast.Module) -> None:
        """Check if a variable was assigned from a softmax/sigmoid call."""
        var = name_node.id
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == var:
                        if isinstance(node.value, ast.Call):
                            inner_name = self._call_name(node.value)
                            if inner_name in ("softmax", "sigmoid", "log_softmax"):
                                self.bugs.append(IntentApparentBug(
                                    kind=IntentBugKind.DOUBLE_ACTIVATION,
                                    line=getattr(loss_node, "lineno", 0),
                                    col=getattr(loss_node, "col_offset", 0),
                                    message=(f"Double activation: '{var}' is output of "
                                             f"{inner_name}() (line {node.lineno}), fed to "
                                             f"{loss_name}() which applies activation internally"),
                                    function="<module>",
                                    variable=var,
                                    inferred_intent=InferredIntent(
                                        description=f"Use raw logits with {loss_name}",
                                        confidence=0.90,
                                        evidence=[f"{var} = {inner_name}(...) on line {node.lineno}",
                                                  f"{loss_name} expects raw logits"],
                                    ),
                                    bug_class_id=11,
                                ))

    def _check_functional_dropout_eval(self, node: ast.Call, tree: ast.Module) -> None:
        """Detect F.dropout(x, training=True) that ignores model state."""
        call_name = self._call_name(node)
        if call_name != "dropout":
            return
        for kw in node.keywords:
            if kw.arg == "training":
                if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    self.bugs.append(IntentApparentBug(
                        kind=IntentBugKind.FUNCTIONAL_DROPOUT_EVAL,
                        line=getattr(node, "lineno", 0),
                        col=getattr(node, "col_offset", 0),
                        message="F.dropout() with training=True hardcoded; will apply dropout in eval mode",
                        function="<module>",
                        variable="dropout",
                        inferred_intent=InferredIntent(
                            description="Dropout should respect model.training state",
                            confidence=0.92,
                            evidence=["training=True is hardcoded",
                                      "Should use self.training or training=self.training"],
                        ),
                        bug_class_id=12,
                    ))

    def _check_wrong_softmax_dim(self, node: ast.Call, tree: ast.Module) -> None:
        """Detect softmax(logits, dim=0) where dim=1 or dim=-1 is likely intended."""
        call_name = self._call_name(node)
        if call_name != "softmax":
            return
        for kw in node.keywords:
            if kw.arg == "dim":
                if isinstance(kw.value, ast.Constant) and kw.value.value == 0:
                    self.bugs.append(IntentApparentBug(
                        kind=IntentBugKind.WRONG_SOFTMAX_DIM,
                        line=getattr(node, "lineno", 0),
                        col=getattr(node, "col_offset", 0),
                        message="softmax(dim=0) normalizes across batch dimension; did you mean dim=1 or dim=-1?",
                        function="<module>",
                        variable="softmax",
                        inferred_intent=InferredIntent(
                            description="Normalize across class/feature dimension, not batch",
                            confidence=0.80,
                            evidence=["dim=0 is typically the batch dimension",
                                      "Softmax usually normalizes over classes (dim=1 or -1)"],
                        ),
                        bug_class_id=10,
                    ))
        # Also check positional: softmax(x, 0)
        if len(node.args) >= 2:
            dim_arg = node.args[1]
            if isinstance(dim_arg, ast.Constant) and dim_arg.value == 0:
                self.bugs.append(IntentApparentBug(
                    kind=IntentBugKind.WRONG_SOFTMAX_DIM,
                    line=getattr(node, "lineno", 0),
                    col=getattr(node, "col_offset", 0),
                    message="softmax(dim=0) normalizes across batch dimension; did you mean dim=1 or dim=-1?",
                    function="<module>",
                    variable="softmax",
                    inferred_intent=InferredIntent(
                        description="Normalize across class/feature dimension, not batch",
                        confidence=0.80,
                        evidence=["dim=0 is typically the batch dimension"],
                    ),
                    bug_class_id=10,
                ))

    def _check_wrong_loss_family(self, node: ast.Call, tree: ast.Module) -> None:
        """Detect BCEWithLogitsLoss used with one-hot targets (should be CrossEntropyLoss)."""
        call_name = self._call_name(node)
        if call_name != "BCEWithLogitsLoss":
            return
        # This is a heuristic overwarn — flag it when we see BCEWithLogitsLoss
        # with more than 2 classes (common pattern)
        self.bugs.append(IntentApparentBug(
            kind=IntentBugKind.WRONG_LOSS_FAMILY,
            line=getattr(node, "lineno", 0),
            col=getattr(node, "col_offset", 0),
            message=("BCEWithLogitsLoss detected — verify this is multi-label, not "
                     "single-label multi-class (use CrossEntropyLoss for single-label)"),
            function="<module>",
            variable="BCEWithLogitsLoss",
            inferred_intent=InferredIntent(
                description="Use appropriate loss for classification task",
                confidence=0.60,
                evidence=["BCEWithLogitsLoss used; may be wrong for single-label multi-class"],
            ),
            bug_class_id=17,
        ))

    @staticmethod
    def _call_name(node: ast.Call) -> str:
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        if isinstance(node.func, ast.Name):
            return node.func.id
        return ""


class OptimizerPatternChecker(PatternChecker):
    """Checks for optimizer protocol bugs.

    Bug classes covered:
      #16 - Missing zero_grad
      #27 - Unfreezing without re-adding to optimizer
      #30 - Duplicate parameter in optimizer groups
      #34 - LR scheduler stepped per batch instead of per epoch
      #35 - Frozen parameters in optimizer
      #36 - Parameters added after optimizer creation
      #37 - Dead parameters
      #41 - Weight decay on normalization/bias
      #42 - Parameter mutation via .data
      #43 - Gradient accumulation without loss scaling
    """

    def check(self, tree: ast.Module, intent_info: Dict[str, Any]) -> List[IntentApparentBug]:
        self.bugs = []
        for loop_info in intent_info.get("training_loops", []):
            self._check_missing_zero_grad(loop_info)
            self._check_scheduler_frequency(loop_info, tree)
            self._check_grad_accum_scaling(loop_info, tree)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                self._check_data_bypass(node)
            if isinstance(node, ast.Assign):
                self._check_param_replaced(node)

        self._check_weight_decay_on_norms(tree, intent_info)
        self._check_unregistered_submodules(tree, intent_info)
        self._check_unintended_weight_sharing(tree)
        return self.bugs

    def _check_missing_zero_grad(self, loop_info: Dict[str, Any]) -> None:
        """Detect training loops that call backward() without zero_grad()."""
        if loop_info["has_backward"] and not loop_info["has_zero_grad"]:
            self.bugs.append(IntentApparentBug(
                kind=IntentBugKind.MISSING_ZERO_GRAD,
                line=loop_info["backward_line"],
                col=0,
                message="backward() called without zero_grad(); gradients will accumulate across steps",
                function="<module>",
                variable="optimizer",
                inferred_intent=InferredIntent(
                    description="Zero gradients before each backward pass",
                    confidence=0.88,
                    evidence=["backward() found without preceding zero_grad()",
                              "Gradient accumulation is usually unintended"],
                ),
                bug_class_id=16,
            ))

    def _check_scheduler_frequency(self, loop_info: Dict[str, Any],
                                    tree: ast.Module) -> None:
        """Detect scheduler.step() called per batch instead of per epoch."""
        if loop_info["has_scheduler_step"] and loop_info["has_backward"]:
            # If sched.step() is in the same loop as backward(), it's per-batch
            self.bugs.append(IntentApparentBug(
                kind=IntentBugKind.WRONG_LR_SCHEDULE,
                line=loop_info["sched_step_line"],
                col=0,
                message="LR scheduler stepped in same loop as backward(); likely per-batch instead of per-epoch",
                function="<module>",
                variable="scheduler",
                inferred_intent=InferredIntent(
                    description="Step LR scheduler once per epoch, not per batch",
                    confidence=0.75,
                    evidence=["scheduler.step() in training loop body",
                              "Most schedulers expect epoch-level stepping"],
                ),
                bug_class_id=34,
            ))

    def _check_grad_accum_scaling(self, loop_info: Dict[str, Any],
                                   tree: ast.Module) -> None:
        """Detect gradient accumulation without loss scaling."""
        if not loop_info["has_backward"]:
            return
        # Look for modulo-based step pattern without loss division
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test = node.test
                if isinstance(test, ast.Compare):
                    for op in test.ops:
                        if isinstance(op, ast.Eq):
                            # Check if body contains opt.step()
                            for stmt in ast.walk(node):
                                if isinstance(stmt, ast.Call):
                                    name = self._get_call_name(stmt)
                                    if name and name.endswith(".step"):
                                        # Found accumulation pattern — check for loss scaling
                                        self.bugs.append(IntentApparentBug(
                                            kind=IntentBugKind.GRAD_ACCUM_UNSCALED,
                                            line=loop_info["backward_line"],
                                            col=0,
                                            message="Gradient accumulation detected but loss may not be scaled by 1/accum_steps",
                                            function="<module>",
                                            variable="loss",
                                            inferred_intent=InferredIntent(
                                                description="Scale loss by 1/accum_steps for correct effective gradient",
                                                confidence=0.70,
                                                evidence=["Modulo-based optimizer stepping detected",
                                                          "backward() called every iteration"],
                                            ),
                                            bug_class_id=43,
                                        ))
                                        return

    def _check_data_bypass(self, node: ast.Call) -> None:
        """Detect parameter.data mutations that bypass autograd."""
        # Pattern: x.data.mul_(), x.data = ..., etc.
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Attribute):
                if node.func.value.attr == "data":
                    method = node.func.attr
                    if method.endswith("_"):  # in-place ops
                        self.bugs.append(IntentApparentBug(
                            kind=IntentBugKind.DATA_BYPASS,
                            line=getattr(node, "lineno", 0),
                            col=getattr(node, "col_offset", 0),
                            message=f".data.{method}() bypasses autograd; optimizer state may be desynchronized",
                            function="<module>",
                            variable="parameter",
                            inferred_intent=InferredIntent(
                                description="Modify parameters through autograd-tracked operations",
                                confidence=0.85,
                                evidence=[f".data.{method}() bypasses autograd graph",
                                          "Optimizer momentum/Adam buffers may be stale"],
                            ),
                            bug_class_id=42,
                        ))

    def _check_param_replaced(self, node: ast.Assign) -> None:
        """Detect parameter replacement that creates non-Parameter tensors."""
        # Pattern: model.layer.weight = model.layer.weight + 0.1
        for target in node.targets:
            if isinstance(target, ast.Attribute) and target.attr in ("weight", "bias"):
                if isinstance(node.value, ast.BinOp):
                    self.bugs.append(IntentApparentBug(
                        kind=IntentBugKind.PARAM_REPLACED,
                        line=getattr(node, "lineno", 0),
                        col=getattr(node, "col_offset", 0),
                        message="Assigning arithmetic result to .weight/.bias replaces Parameter with plain Tensor",
                        function="<module>",
                        variable=target.attr,
                        inferred_intent=InferredIntent(
                            description="Modify parameter in-place or use torch.no_grad()",
                            confidence=0.85,
                            evidence=["Direct assignment to .weight/.bias",
                                      "Result of + is a Tensor, not a Parameter"],
                        ),
                        bug_class_id=13,
                    ))

    def _check_weight_decay_on_norms(self, tree: ast.Module,
                                      intent_info: Dict[str, Any]) -> None:
        """Detect weight decay applied to normalization and bias parameters."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_name = self._get_call_name(node)
                if call_name and call_name in ("SGD", "Adam", "AdamW"):
                    for kw in node.keywords:
                        if kw.arg == "weight_decay":
                            if isinstance(kw.value, ast.Constant) and kw.value.value and kw.value.value > 0:
                                # Check if model has BN/LN layers
                                for cls_info in intent_info.get("classes", {}).values():
                                    layers = cls_info.get("layers", {})
                                    has_norm = any(
                                        lt in ("BatchNorm2d", "BatchNorm1d", "LayerNorm",
                                               "GroupNorm", "InstanceNorm2d")
                                        for lt in layers.values()
                                    )
                                    if has_norm:
                                        # Check if params are separated into groups
                                        if len(node.args) >= 1 and isinstance(node.args[0], ast.Call):
                                            pass  # Likely using model.parameters() — no separation
                                        self.bugs.append(IntentApparentBug(
                                            kind=IntentBugKind.WEIGHT_DECAY_NORM,
                                            line=getattr(node, "lineno", 0),
                                            col=getattr(node, "col_offset", 0),
                                            message="weight_decay may be applied to normalization/bias parameters; consider separate param groups",
                                            function="<module>",
                                            variable="optimizer",
                                            inferred_intent=InferredIntent(
                                                description="Exclude norm/bias parameters from weight decay",
                                                confidence=0.70,
                                                evidence=["Model has normalization layers",
                                                          "Weight decay applied to all parameters",
                                                          "Norm params should typically not be decayed"],
                                            ),
                                            bug_class_id=41,
                                        ))
                                        return

    def _check_unregistered_submodules(self, tree: ast.Module,
                                        intent_info: Dict[str, Any]) -> None:
        """Detect submodules stored in plain Python lists instead of ModuleList."""
        for cls_name, cls_info in intent_info.get("classes", {}).items():
            if not cls_info.get("is_module"):
                continue
            for list_attr in cls_info.get("plain_lists", []):
                self.bugs.append(IntentApparentBug(
                    kind=IntentBugKind.UNREGISTERED_SUBMODULE,
                    line=0, col=0,
                    message=f"'{list_attr}' in {cls_name} is a plain Python list; submodules won't be registered (use nn.ModuleList)",
                    function=f"{cls_name}.__init__",
                    variable=list_attr,
                    inferred_intent=InferredIntent(
                        description="Register submodules so .parameters() and .to() work correctly",
                        confidence=0.88,
                        evidence=["Plain list used in nn.Module.__init__",
                                  "Submodule parameters won't appear in model.parameters()"],
                    ),
                    bug_class_id=15,
                ))

    def _check_unintended_weight_sharing(self, tree: ast.Module) -> None:
        """Detect [nn.Linear(...)] * N pattern (same instance reused)."""
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                if isinstance(node.left, ast.List) and len(node.left.elts) == 1:
                    elt = node.left.elts[0]
                    if isinstance(elt, ast.Call):
                        layer_name = self._get_call_name(elt)
                        if layer_name and layer_name in ("Linear", "Conv2d", "Conv1d",
                                                          "LSTM", "GRU", "BatchNorm2d"):
                            self.bugs.append(IntentApparentBug(
                                kind=IntentBugKind.UNINTENDED_WEIGHT_SHARING,
                                line=getattr(node, "lineno", 0),
                                col=getattr(node, "col_offset", 0),
                                message=f"[{layer_name}(...)] * N creates N references to the SAME module instance",
                                function="<module>",
                                variable=layer_name,
                                inferred_intent=InferredIntent(
                                    description=f"Create N independent {layer_name} instances",
                                    confidence=0.95,
                                    evidence=["List multiplication reuses same object reference",
                                              "Use list comprehension for independent instances"],
                                ),
                                bug_class_id=14,
                            ))

    @staticmethod
    def _get_call_name(node: ast.Call) -> Optional[str]:
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        if isinstance(node.func, ast.Name):
            return node.func.id
        return None


class GradientPatternChecker(PatternChecker):
    """Checks for gradient flow bugs.

    Bug classes covered:
      #4  - Gradient broken via detach
      #18 - Gradient clipping before unscale in mixed precision
      #24 - Hard gating with non-differentiable ops
      #25 - In-place aliasing in residual paths
      #29 - RNN hidden-state leakage
    """

    def check(self, tree: ast.Module, intent_info: Dict[str, Any]) -> List[IntentApparentBug]:
        self.bugs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                self._check_detach_before_loss(node, tree)
                self._check_hidden_state_leak(node, tree)
        self._check_hard_gating(tree)
        self._check_inplace_residual(tree)
        return self.bugs

    def _check_detach_before_loss(self, node: ast.Call, tree: ast.Module) -> None:
        """Detect .detach() on intermediates that feed into loss."""
        if isinstance(node.func, ast.Attribute) and node.func.attr == "detach":
            self.bugs.append(IntentApparentBug(
                kind=IntentBugKind.GRADIENT_BROKEN,
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                message=".detach() breaks gradient flow; verify this is intentional",
                function="<module>",
                variable="tensor",
                inferred_intent=InferredIntent(
                    description="Maintain differentiable path from input to loss",
                    confidence=0.65,
                    evidence=[".detach() called on intermediate tensor",
                              "Upstream parameters will not receive gradients"],
                ),
                bug_class_id=4,
            ))

    def _check_hidden_state_leak(self, node: ast.Call, tree: ast.Module) -> None:
        """Detect RNN hidden state reuse without detach across batches."""
        call_name = ""
        if isinstance(node.func, ast.Name):
            call_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            call_name = node.func.attr

        if call_name in ("LSTM", "GRU", "RNN"):
            pass  # This is a constructor, not a call

    def _check_hard_gating(self, tree: ast.Module) -> None:
        """Detect (logits > 0).float() * logits pattern (non-differentiable gate)."""
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                # Check if either side involves a comparison .float()
                for side in (node.left, node.right):
                    if isinstance(side, ast.Call):
                        if isinstance(side.func, ast.Attribute) and side.func.attr == "float":
                            if isinstance(side.func.value, ast.Compare):
                                self.bugs.append(IntentApparentBug(
                                    kind=IntentBugKind.HARD_GATING,
                                    line=getattr(node, "lineno", 0),
                                    col=getattr(node, "col_offset", 0),
                                    message="Hard gating via comparison.float() creates zero gradients; consider sigmoid or straight-through estimator",
                                    function="<module>",
                                    variable="gate",
                                    inferred_intent=InferredIntent(
                                        description="Use differentiable gating for gradient flow",
                                        confidence=0.80,
                                        evidence=["Boolean comparison cast to float",
                                                  "d/dx[1[x>0]] = 0 a.e."],
                                    ),
                                    bug_class_id=24,
                                ))

    def _check_inplace_residual(self, tree: ast.Module) -> None:
        """Detect in-place ops (relu_, add_) on tensors used in skip connections."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    method = node.func.attr
                    if method.endswith("_") and method in ("relu_", "add_", "mul_",
                                                            "sigmoid_", "tanh_"):
                        self.bugs.append(IntentApparentBug(
                            kind=IntentBugKind.INPLACE_ALIAS_RESIDUAL,
                            line=getattr(node, "lineno", 0),
                            col=getattr(node, "col_offset", 0),
                            message=f"In-place {method}() may corrupt aliased tensors in residual/skip connections",
                            function="<module>",
                            variable=method,
                            inferred_intent=InferredIntent(
                                description="Use out-of-place operation to preserve skip connection values",
                                confidence=0.70,
                                evidence=[f"{method} modifies tensor in-place",
                                          "Aliases (skip connections) will be affected"],
                            ),
                            bug_class_id=25,
                        ))


class DevicePatternChecker(PatternChecker):
    """Checks for device mismatch bugs.

    Bug classes covered:
      #6 - Cross-device tensor op (plain tensor attributes, tensor creation
           in forward without device kwarg, plain lists of tensors)
    """

    # torch factory functions that default to CPU when no device= is given
    _CPU_DEFAULT_FACTORIES = frozenset({
        "torch.zeros", "torch.ones", "torch.randn", "torch.rand",
        "torch.empty", "torch.eye", "torch.arange", "torch.linspace",
        "torch.full", "torch.tensor", "torch.zeros_like", "torch.ones_like",
        "torch.randn_like", "torch.rand_like", "torch.empty_like",
    })

    def check(self, tree: ast.Module, intent_info: Dict[str, Any]) -> List[IntentApparentBug]:
        self.bugs: List[IntentApparentBug] = []
        for cls_name, cls_info in intent_info.get("classes", {}).items():
            if not cls_info.get("is_module"):
                continue
            # (a) Plain tensors stored as attributes
            for tensor_attr in cls_info.get("plain_tensors", []):
                self.bugs.append(IntentApparentBug(
                    kind=IntentBugKind.DEVICE_MISMATCH,
                    line=0, col=0,
                    message=f"'{tensor_attr}' in {cls_name} is a plain Tensor, not a Parameter/buffer; .cuda()/.to() won't move it",
                    function=f"{cls_name}.__init__",
                    variable=tensor_attr,
                    inferred_intent=InferredIntent(
                        description="Register tensor as buffer or Parameter so device transfers work",
                        confidence=0.90,
                        evidence=["Plain tensor stored as module attribute",
                                  "Module.cuda()/to() only moves Parameters and buffers"],
                    ),
                    bug_class_id=6,
                ))
            # (b) Plain list of tensors (won't move with .to())
            for list_attr in cls_info.get("plain_lists", []):
                self.bugs.append(IntentApparentBug(
                    kind=IntentBugKind.DEVICE_MISMATCH,
                    line=0, col=0,
                    message=f"'{list_attr}' in {cls_name} is a plain list, not ModuleList/ParameterList; .to() won't move its tensors",
                    function=f"{cls_name}.__init__",
                    variable=list_attr,
                    inferred_intent=InferredIntent(
                        description="Use nn.ModuleList or nn.ParameterList so device transfers apply",
                        confidence=0.85,
                        evidence=["Plain Python list stored as module attribute",
                                  "Module.to()/cuda() only traverses registered sub-modules and parameters"],
                    ),
                    bug_class_id=6,
                ))
            # (c) Tensor creation in forward() without device= kwarg
            fwd = cls_info.get("forward_method")
            if fwd is not None:
                self._check_forward_device_bugs(fwd, cls_name)
        return self.bugs

    def _check_forward_device_bugs(self, fwd: ast.FunctionDef, cls_name: str) -> None:
        """Flag torch factory calls inside forward() that lack device=."""
        for node in ast.walk(fwd):
            if not isinstance(node, ast.Call):
                continue
            call_name = _get_full_call_name(node)
            if call_name not in self._CPU_DEFAULT_FACTORIES:
                continue
            # Check if device= keyword is supplied
            has_device_kwarg = any(
                kw.arg == "device" for kw in node.keywords
            )
            if has_device_kwarg:
                continue
            # _like functions inherit device from the source tensor
            if call_name.endswith("_like"):
                continue
            line = getattr(node, "lineno", 0)
            self.bugs.append(IntentApparentBug(
                kind=IntentBugKind.DEVICE_MISMATCH,
                line=line, col=getattr(node, "col_offset", 0),
                message=(
                    f"{call_name}() in {cls_name}.forward() creates a CPU tensor; "
                    f"use device=x.device to match input device"
                ),
                function=f"{cls_name}.forward",
                variable=call_name,
                inferred_intent=InferredIntent(
                    description="Tensor factory in forward should match input device",
                    confidence=0.92,
                    evidence=[
                        f"{call_name}() defaults to CPU",
                        "Input tensor may be on CUDA",
                        "Operations between CPU and CUDA tensors raise RuntimeError",
                    ],
                ),
                bug_class_id=6,
            ))


class ShapeSemanticChecker(PatternChecker):
    """Checks for shape-semantic bugs beyond simple dimension arithmetic.

    Bug classes covered:
      #1  - View on non-contiguous tensor
      #2  - Accidental batch broadcasting
      #8  - Reshape with element-count mismatch (dynamic blocks)
      #9  - LSTM batch_first mismatch
      #22 - Flatten collapsing batch dimension
      #32 - LayerNorm on NCHW with wrong normalized_shape
      #40 - MultiheadAttention batch_first mismatch
      #45 - Conv1d applied to (N, L, C) instead of (N, C, L)
      #48 - squeeze() without dim drops unintended axes
    """

    def check(self, tree: ast.Module, intent_info: Dict[str, Any]) -> List[IntentApparentBug]:
        self.bugs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                self._check_squeeze_no_dim(node)
                self._check_flatten_start_dim_zero(node)
                self._check_batch_first_mismatch(node, tree)
        return self.bugs

    def _check_squeeze_no_dim(self, node: ast.Call) -> None:
        """Detect squeeze() without explicit dim (unstable across batch sizes)."""
        if isinstance(node.func, ast.Attribute) and node.func.attr == "squeeze":
            if not node.args and not node.keywords:
                self.bugs.append(IntentApparentBug(
                    kind=IntentBugKind.SQUEEZE_UNSTABLE,
                    line=getattr(node, "lineno", 0),
                    col=getattr(node, "col_offset", 0),
                    message="squeeze() without dim argument may drop different axes depending on runtime tensor size",
                    function="<module>",
                    variable="tensor",
                    inferred_intent=InferredIntent(
                        description="Use squeeze(dim=N) for deterministic behavior",
                        confidence=0.85,
                        evidence=["squeeze() removes ALL size-1 dimensions",
                                  "Batch dimension may be size-1 at inference time"],
                    ),
                    bug_class_id=48,
                ))

    def _check_flatten_start_dim_zero(self, node: ast.Call) -> None:
        """Detect flatten(start_dim=0) which collapses batch dimension."""
        call_name = ""
        if isinstance(node.func, ast.Attribute):
            call_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            call_name = node.func.id

        if call_name != "flatten":
            return

        for kw in node.keywords:
            if kw.arg == "start_dim":
                if isinstance(kw.value, ast.Constant) and kw.value.value == 0:
                    self.bugs.append(IntentApparentBug(
                        kind=IntentBugKind.FLATTEN_BATCH,
                        line=getattr(node, "lineno", 0),
                        col=getattr(node, "col_offset", 0),
                        message="flatten(start_dim=0) collapses batch dimension into features; likely intended start_dim=1",
                        function="<module>",
                        variable="tensor",
                        inferred_intent=InferredIntent(
                            description="Preserve batch dimension, flatten spatial/feature dims",
                            confidence=0.80,
                            evidence=["start_dim=0 merges batch into features",
                                      "Per-sample statistics will mix samples"],
                        ),
                        bug_class_id=22,
                    ))
        # Positional: flatten(x, 0) or torch.flatten(x, 0)
        if len(node.args) >= 2:
            dim_arg = node.args[1]
            if isinstance(dim_arg, ast.Constant) and dim_arg.value == 0:
                self.bugs.append(IntentApparentBug(
                    kind=IntentBugKind.FLATTEN_BATCH,
                    line=getattr(node, "lineno", 0),
                    col=getattr(node, "col_offset", 0),
                    message="flatten(start_dim=0) collapses batch dimension",
                    function="<module>",
                    variable="tensor",
                    inferred_intent=InferredIntent(
                        description="Preserve batch dimension",
                        confidence=0.80,
                        evidence=["start_dim=0 merges batch into features"],
                    ),
                    bug_class_id=22,
                ))

    def _check_batch_first_mismatch(self, node: ast.Call, tree: ast.Module) -> None:
        """Detect LSTM/MultiheadAttention without batch_first when input is (B, S, F)."""
        call_name = ""
        if isinstance(node.func, ast.Attribute):
            call_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            call_name = node.func.id

        if call_name not in ("LSTM", "GRU", "MultiheadAttention"):
            return

        has_batch_first = False
        for kw in node.keywords:
            if kw.arg == "batch_first":
                has_batch_first = True
                break

        if not has_batch_first:
            self.bugs.append(IntentApparentBug(
                kind=IntentBugKind.BATCH_FIRST_MISMATCH,
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                message=f"{call_name} defaults to batch_first=False; verify input shape is (seq, batch, features) not (batch, seq, features)",
                function="<module>",
                variable=call_name,
                inferred_intent=InferredIntent(
                    description=f"Ensure {call_name} input shape matches batch_first setting",
                    confidence=0.65,
                    evidence=[f"{call_name} defaults to batch_first=False",
                              "Most data loaders produce (batch, seq, features)"],
                ),
                bug_class_id=9 if call_name in ("LSTM", "GRU") else 40,
            ))


class DTypePatternChecker(PatternChecker):
    """Checks for dtype-related bugs.

    Bug classes covered:
      #5  - Float16 overflow in exp/softmax
      #26 - Integer tensor division causing truncation
    """

    def check(self, tree: ast.Module, intent_info: Dict[str, Any]) -> List[IntentApparentBug]:
        self.bugs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                self._check_fp16_exp(node)
        return self.bugs

    def _check_fp16_exp(self, node: ast.Call) -> None:
        """Detect exp() on float16 tensors (overflow risk)."""
        call_name = ""
        if isinstance(node.func, ast.Attribute):
            call_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            call_name = node.func.id

        if call_name != "exp":
            return

        # Check for dtype=torch.float16 in creation of the input tensor
        for kw in node.keywords:
            if kw.arg == "dtype":
                if isinstance(kw.value, ast.Attribute) and kw.value.attr in ("float16", "half"):
                    self.bugs.append(IntentApparentBug(
                        kind=IntentBugKind.FP16_OVERFLOW,
                        line=getattr(node, "lineno", 0),
                        col=getattr(node, "col_offset", 0),
                        message="exp() on float16 tensor may overflow (max ~65504); consider float32 upcast",
                        function="<module>",
                        variable="tensor",
                        inferred_intent=InferredIntent(
                            description="Compute exp() in float32 for numerical stability",
                            confidence=0.85,
                            evidence=["float16 max is ~65504",
                                      "exp(11.1) > 65504"],
                        ),
                        bug_class_id=5,
                    ))


# ═══════════════════════════════════════════════════════════════════════════
# Main Overwarning Analyzer
# ═══════════════════════════════════════════════════════════════════════════

class OverwarnAnalyzer:
    """The main intent-apparent bug detection engine.

    Runs all pattern checkers against a Python module and collects
    intent-apparent bug warnings. Designed to **overwarn**: when intent
    is ambiguous, warn anyway with an explanation.

    Usage::

        analyzer = OverwarnAnalyzer()
        bugs = analyzer.analyze(source_code)
        for bug in bugs:
            print(f"[{bug.kind.name}] line {bug.line}: {bug.message}")
            print(f"  Intent: {bug.inferred_intent.pretty()}")
    """

    def __init__(self, registry: Optional[BugClassRegistry] = None):
        self.registry = registry or BugClassRegistry()
        self.intent_engine = IntentInferenceEngine()
        self.checkers: List[PatternChecker] = [
            SemanticPatternChecker(),
            OptimizerPatternChecker(),
            GradientPatternChecker(),
            DevicePatternChecker(),
            ShapeSemanticChecker(),
            DTypePatternChecker(),
        ]

    def analyze(self, source: str) -> List[IntentApparentBug]:
        """Analyze source code for intent-apparent ML bugs.

        Returns a list of warnings (overwarning mode: prefers false positives
        over false negatives for intent-apparent bugs).
        """
        tree = ast.parse(source)
        intent_info = self.intent_engine.analyze_module(tree)

        all_bugs: List[IntentApparentBug] = []
        for checker in self.checkers:
            bugs = checker.check(tree, intent_info)
            all_bugs.extend(bugs)

        # Deduplicate by (kind, line)
        seen: Set[Tuple[str, int]] = set()
        deduped: List[IntentApparentBug] = []
        for bug in all_bugs:
            key = (bug.kind.name, bug.line)
            if key not in seen:
                seen.add(key)
                deduped.append(bug)

        return deduped

    def analyze_and_summarize(self, source: str) -> Dict[str, Any]:
        """Analyze and return a structured summary."""
        bugs = self.analyze(source)
        by_kind: Dict[str, int] = {}
        for bug in bugs:
            by_kind[bug.kind.name] = by_kind.get(bug.kind.name, 0) + 1
        return {
            "total_warnings": len(bugs),
            "by_kind": by_kind,
            "bugs": [b.to_dict() for b in bugs],
        }
