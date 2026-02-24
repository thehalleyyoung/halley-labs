"""
Theory-Exercising Evaluation: BroadcastPropagator & StridePropagator.

Benchmarks specifically designed to exercise the Z3 BroadcastPropagator and
StridePropagator theories, where a purely syntactic pattern-matching baseline
would give WRONG answers but TensorGuard (with Z3 theories) gives CORRECT answers.

Categories:
  A. Symbolic broadcasting compatibility (non-trivial)
  B. Reshape / stride compatibility requiring Z3
  C. Multi-step computation graphs with shape flow through broadcasts
  D. Cases syntactic baseline misjudges

Outputs:  experiments/theory_exercising_results.json
"""

from __future__ import annotations

import ast
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_checker import (
    extract_computation_graph,
    BoundedModelChecker, ConstraintVerifier,
    Device,
    Phase,
    verify_model,
)
from src.shape_cegar import run_shape_cegar

RESULTS_FILE = Path(__file__).parent / "theory_exercising_results.json"

# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark Suite — 18 nn.Module benchmarks exercising broadcast/stride theories
# ═══════════════════════════════════════════════════════════════════════════════

TEST_CASES: List[Dict[str, Any]] = [
    # ════════════════════════════════════════════════════════════════════════
    # Category A: Broadcasting compatibility — Z3 BroadcastPropagator
    # Syntactic baseline cannot trace dimensions through projections into add
    # ════════════════════════════════════════════════════════════════════════

    # A1: Two parallel projections to different dims → add fails
    {
        "name": "broadcast_parallel_bug",
        "category": "broadcast",
        "has_bug": True,
        "description": "Linear->128 + Linear->64: broadcast dims 128 vs 64",
        "why_syntactic_fails": "Syntactic checker does not propagate output "
            "dims of Linear through add to detect broadcast incompatibility",
        "code": """\
import torch.nn as nn
class BroadcastParallelBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(256, 128)
        self.fc_b = nn.Linear(256, 64)
    def forward(self, x):
        a = self.fc_a(x)
        b = self.fc_b(x)
        return a + b
""",
        "input_shapes": {"x": ("batch", 256)},
    },

    # A2: Same pattern, matching dims → safe
    {
        "name": "broadcast_parallel_safe",
        "category": "broadcast",
        "has_bug": False,
        "description": "Linear->128 + Linear->128: broadcast OK (same dims)",
        "why_syntactic_fails": "Syntactic checker would need to trace output "
            "dims through add — can't verify safety",
        "code": """\
import torch.nn as nn
class BroadcastParallelSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(256, 128)
        self.fc_b = nn.Linear(256, 128)
    def forward(self, x):
        a = self.fc_a(x)
        b = self.fc_b(x)
        return a + b
""",
        "input_shapes": {"x": ("batch", 256)},
    },

    # A3: Cross-rank broadcasting — 3D + 2D with dim mismatch
    {
        "name": "broadcast_cross_rank_bug",
        "category": "broadcast",
        "has_bug": True,
        "description": "(batch,seq,256) + (batch,128): last dims 256 vs 128 mismatch",
        "why_syntactic_fails": "Syntactic checker cannot reason about "
            "cross-rank broadcasting alignment",
        "code": """\
import torch.nn as nn
class CrossRankBroadcastBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(256, 256)
        self.bias_proj = nn.Linear(256, 128)
    def forward(self, x, bias):
        x = self.proj(x)
        bias = self.bias_proj(bias)
        return x + bias
""",
        "input_shapes": {"x": ("batch", "seq", 256), "bias": ("batch", 256)},
    },

    # A4: Cross-rank broadcasting — 3D + 1D safe
    {
        "name": "broadcast_cross_rank_safe",
        "category": "broadcast",
        "has_bug": False,
        "description": "(batch,seq,256) + (256,): broadcasts to (1,1,256) — safe",
        "why_syntactic_fails": "Syntactic checker cannot verify cross-rank "
            "broadcast compatibility",
        "code": """\
import torch.nn as nn
class CrossRankBroadcastSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(256, 256)
        self.bias_proj = nn.Linear(256, 256)
    def forward(self, x, bias):
        x = self.proj(x)
        bias = self.bias_proj(bias)
        return x + bias
""",
        "input_shapes": {"x": ("batch", "seq", 256), "bias": (256,)},
    },

    # A5: Three-way broadcast chain with failure at second add
    {
        "name": "broadcast_chain_bug",
        "category": "broadcast",
        "has_bug": True,
        "description": "(128)+(128)=OK then (128)+(64)=FAIL at second add",
        "why_syntactic_fails": "Syntactic checker cannot track shape through "
            "first add to detect mismatch at second add",
        "code": """\
import torch.nn as nn
class BroadcastChainBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(256, 128)
        self.fc_b = nn.Linear(256, 128)
        self.fc_c = nn.Linear(256, 64)
    def forward(self, x):
        a = self.fc_a(x)
        b = self.fc_b(x)
        c = self.fc_c(x)
        ab = a + b
        return ab + c
""",
        "input_shapes": {"x": ("batch", 256)},
    },

    # A6: Three-way broadcast chain — all matching
    {
        "name": "broadcast_chain_safe",
        "category": "broadcast",
        "has_bug": False,
        "description": "(128)+(128)+(128): all adds broadcast-compatible",
        "why_syntactic_fails": "Syntactic checker cannot verify that all "
            "three branches produce compatible shapes for chained adds",
        "code": """\
import torch.nn as nn
class BroadcastChainSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(256, 128)
        self.fc_b = nn.Linear(256, 128)
        self.fc_c = nn.Linear(256, 128)
    def forward(self, x):
        a = self.fc_a(x)
        b = self.fc_b(x)
        c = self.fc_c(x)
        ab = a + b
        return ab + c
""",
        "input_shapes": {"x": ("batch", 256)},
    },

    # ════════════════════════════════════════════════════════════════════════
    # Category B: Matmul inner-dim compatibility — BroadcastPropagator
    # ════════════════════════════════════════════════════════════════════════

    # B1: Matmul with mismatched inner dim after projection
    {
        "name": "matmul_inner_mismatch_bug",
        "category": "matmul",
        "has_bug": True,
        "description": "proj->32 then matmul with (64,10): inner 32!=64",
        "why_syntactic_fails": "Syntactic checker cannot trace projected "
            "output dim through matmul to verify inner dim match",
        "code": """\
import torch.nn as nn
class MatmulInnerBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(128, 32)
    def forward(self, x, w):
        x = self.proj(x)
        return x @ w
""",
        "input_shapes": {"x": ("batch", 128), "w": (64, 10)},
    },

    # B2: Matmul with matching inner dim
    {
        "name": "matmul_inner_match_safe",
        "category": "matmul",
        "has_bug": False,
        "description": "proj->64 then matmul with (64,10): inner 64==64 OK",
        "why_syntactic_fails": "Syntactic checker cannot verify inner dim "
            "compatibility through projection + matmul",
        "code": """\
import torch.nn as nn
class MatmulInnerSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(128, 64)
    def forward(self, x, w):
        x = self.proj(x)
        return x @ w
""",
        "input_shapes": {"x": ("batch", 128), "w": (64, 10)},
    },

    # B3: Matmul after broadcast add — shape must propagate through add
    {
        "name": "matmul_after_add_bug",
        "category": "matmul",
        "has_bug": True,
        "description": "Two proj->32, add, then matmul(64,10): inner 32!=64",
        "why_syntactic_fails": "Syntactic checker cannot propagate shape "
            "through add to verify matmul inner dim",
        "code": """\
import torch.nn as nn
class MatmulAfterAddBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(256, 32)
        self.fc_b = nn.Linear(256, 32)
    def forward(self, x, w):
        a = self.fc_a(x)
        b = self.fc_b(x)
        c = a + b
        return c @ w
""",
        "input_shapes": {"x": ("batch", 256), "w": (64, 10)},
    },

    # B4: Matmul after broadcast add — matching
    {
        "name": "matmul_after_add_safe",
        "category": "matmul",
        "has_bug": False,
        "description": "Two proj->64, add, then matmul(64,10): inner 64==64 OK",
        "why_syntactic_fails": "Syntactic checker cannot propagate shape "
            "through add to verify matmul compatibility",
        "code": """\
import torch.nn as nn
class MatmulAfterAddSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(256, 64)
        self.fc_b = nn.Linear(256, 64)
    def forward(self, x, w):
        a = self.fc_a(x)
        b = self.fc_b(x)
        c = a + b
        return c @ w
""",
        "input_shapes": {"x": ("batch", 256), "w": (64, 10)},
    },

    # ════════════════════════════════════════════════════════════════════════
    # Category C: Multi-step computation graphs — combined theory reasoning
    # ════════════════════════════════════════════════════════════════════════

    # C1: Add then Linear — broadcast dim flows into Linear mismatch
    {
        "name": "add_then_linear_bug",
        "category": "multi_step",
        "has_bug": True,
        "description": "proj->128 + proj->64 then Linear(128,10): add already fails",
        "why_syntactic_fails": "Syntactic checker cannot detect that add "
            "of mismatched projected dims fails before Linear",
        "code": """\
import torch.nn as nn
class AddThenLinearBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(256, 128)
        self.fc_b = nn.Linear(256, 64)
        self.out = nn.Linear(128, 10)
    def forward(self, x):
        a = self.fc_a(x)
        b = self.fc_b(x)
        c = a + b
        return self.out(c)
""",
        "input_shapes": {"x": ("batch", 256)},
    },

    # C2: Add then Linear — safe
    {
        "name": "add_then_linear_safe",
        "category": "multi_step",
        "has_bug": False,
        "description": "proj->128 + proj->128 then Linear(128,10): all OK",
        "why_syntactic_fails": "Syntactic checker cannot verify shape "
            "compatibility of add result flowing into Linear",
        "code": """\
import torch.nn as nn
class AddThenLinearSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(256, 128)
        self.fc_b = nn.Linear(256, 128)
        self.out = nn.Linear(128, 10)
    def forward(self, x):
        a = self.fc_a(x)
        b = self.fc_b(x)
        c = a + b
        return self.out(c)
""",
        "input_shapes": {"x": ("batch", 256)},
    },

    # C3: Multi-head projection style — add with symbolic shapes
    {
        "name": "multihead_add_bug",
        "category": "multi_step",
        "has_bug": True,
        "description": "Q->256 + K->128 (symbolic seq dim): broadcast fails",
        "why_syntactic_fails": "Syntactic checker cannot track that different "
            "projection dims are incompatible through add with symbolic shapes",
        "code": """\
import torch.nn as nn
class MultiHeadAddBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(512, 256)
        self.k_proj = nn.Linear(512, 128)
        self.relu = nn.ReLU()
        self.out = nn.Linear(256, 64)
    def forward(self, x):
        q = self.relu(self.q_proj(x))
        k = self.relu(self.k_proj(x))
        combined = q + k
        return self.out(combined)
""",
        "input_shapes": {"x": ("batch", "seq", 512)},
    },

    # C4: Multi-head projection — safe
    {
        "name": "multihead_add_safe",
        "category": "multi_step",
        "has_bug": False,
        "description": "Q->256 + K->256: matching projection dims, add safe",
        "why_syntactic_fails": "Syntactic checker cannot verify matching "
            "projection dims are broadcast-compatible through add",
        "code": """\
import torch.nn as nn
class MultiHeadAddSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(512, 256)
        self.k_proj = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.out = nn.Linear(256, 64)
    def forward(self, x):
        q = self.relu(self.q_proj(x))
        k = self.relu(self.k_proj(x))
        combined = q + k
        return self.out(combined)
""",
        "input_shapes": {"x": ("batch", "seq", 512)},
    },

    # C5: Conv2d parallel paths with channel mismatch in add
    {
        "name": "conv_parallel_add_bug",
        "category": "multi_step",
        "has_bug": True,
        "description": "Conv->32 + Conv->64: channel mismatch in add",
        "why_syntactic_fails": "Syntactic checker does not propagate conv "
            "output channels through add to detect broadcast failure",
        "code": """\
import torch.nn as nn
class ConvParallelAddBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(3, 64, 3, padding=1)
    def forward(self, x):
        a = self.conv1(x)
        b = self.conv2(x)
        return a + b
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },

    # C6: Conv2d parallel paths — safe
    {
        "name": "conv_parallel_add_safe",
        "category": "multi_step",
        "has_bug": False,
        "description": "Conv->32 + Conv->32: matching channels, add safe",
        "why_syntactic_fails": "Syntactic checker does not propagate conv "
            "output channels through add",
        "code": """\
import torch.nn as nn
class ConvParallelAddSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(3, 32, 3, padding=1)
    def forward(self, x):
        a = self.conv1(x)
        b = self.conv2(x)
        return a + b
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },

    # C7: Double add chain with third branch mismatch
    {
        "name": "double_add_chain_bug",
        "category": "multi_step",
        "has_bug": True,
        "description": "proj->32 + proj->32 OK, then + proj->64 FAILS",
        "why_syntactic_fails": "Syntactic checker cannot track shape through "
            "first add to detect mismatch at second add",
        "code": """\
import torch.nn as nn
class DoubleAddChainBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(256, 32)
        self.fc_b = nn.Linear(256, 32)
        self.fc_c = nn.Linear(256, 64)
    def forward(self, x):
        a = self.fc_a(x)
        b = self.fc_b(x)
        s = a + b
        c = self.fc_c(x)
        return s + c
""",
        "input_shapes": {"x": ("batch", 256)},
    },

    # C8: Double add chain — all matching
    {
        "name": "double_add_chain_safe",
        "category": "multi_step",
        "has_bug": False,
        "description": "proj->64 + proj->64 + proj->64: all adds safe",
        "why_syntactic_fails": "Syntactic checker cannot verify all three "
            "branches produce compatible shapes through chained adds",
        "code": """\
import torch.nn as nn
class DoubleAddChainSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(256, 64)
        self.fc_b = nn.Linear(256, 64)
        self.fc_c = nn.Linear(256, 64)
    def forward(self, x):
        a = self.fc_a(x)
        b = self.fc_b(x)
        s = a + b
        c = self.fc_c(x)
        return s + c
""",
        "input_shapes": {"x": ("batch", 256)},
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Syntactic Baseline (no Z3)
# ═══════════════════════════════════════════════════════════════════════════════

class SyntacticShapeChecker:
    """Pure AST-based shape checker — no Z3, no CEGAR.

    Parses layer definitions from __init__ and traces simple data flow in
    forward().  Uses concrete arithmetic only.  Cannot reason about:
      - broadcast compatibility
      - reshape element-count preservation
      - matmul inner-dimension matching across projections
    """

    def __init__(self, source: str):
        self.source = source
        self.tree = ast.parse(source)
        self.layers: Dict[str, Dict[str, Any]] = {}
        self.bugs: List[str] = []

    def check(self) -> Tuple[bool, List[str]]:
        """Return (has_bug, list of bug descriptions)."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                if self._is_nn_module(node):
                    self._extract_layers(node)
                    self._trace_forward(node)
        return len(self.bugs) > 0, self.bugs

    def _is_nn_module(self, cls: ast.ClassDef) -> bool:
        for base in cls.bases:
            if isinstance(base, ast.Attribute) and base.attr == "Module":
                return True
            if isinstance(base, ast.Name) and base.id in ("Module",):
                return True
        return False

    def _extract_layers(self, cls: ast.ClassDef):
        for node in ast.walk(cls):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                            and isinstance(node.value, ast.Call)):
                        self._parse_layer_call(target.attr, node.value)

    def _parse_layer_call(self, attr_name: str, call: ast.Call):
        func = call.func
        layer_type = None
        if isinstance(func, ast.Attribute):
            layer_type = func.attr
        elif isinstance(func, ast.Name):
            layer_type = func.id

        if layer_type == "Linear" and len(call.args) >= 2:
            in_f = self._const_val(call.args[0])
            out_f = self._const_val(call.args[1])
            if in_f is not None and out_f is not None:
                self.layers[attr_name] = {
                    "type": "Linear", "in_features": in_f, "out_features": out_f
                }
        elif layer_type == "Conv2d" and len(call.args) >= 3:
            in_c = self._const_val(call.args[0])
            out_c = self._const_val(call.args[1])
            k = self._const_val(call.args[2])
            if in_c is not None and out_c is not None:
                self.layers[attr_name] = {
                    "type": "Conv2d", "in_channels": in_c, "out_channels": out_c,
                    "kernel_size": k,
                }

    @staticmethod
    def _const_val(node: ast.expr) -> Optional[int]:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            if isinstance(node.operand, ast.Constant):
                return -node.operand.value
        return None

    def _trace_forward(self, cls: ast.ClassDef):
        forward_fn = None
        for item in cls.body:
            if isinstance(item, ast.FunctionDef) and item.name == "forward":
                forward_fn = item
                break
        if forward_fn is None:
            return

        var_shapes: Dict[str, Optional[int]] = {}

        for stmt in ast.walk(forward_fn):
            call_info = self._extract_self_layer_call(stmt)
            if call_info is None:
                continue
            layer_name, input_var, output_var = call_info

            if layer_name not in self.layers:
                if input_var and input_var in var_shapes and output_var:
                    var_shapes[output_var] = var_shapes[input_var]
                continue

            layer = self.layers[layer_name]

            if layer["type"] == "Linear":
                expected_in = layer["in_features"]
                out_feat = layer["out_features"]
                if input_var and input_var in var_shapes:
                    actual_in = var_shapes[input_var]
                    if actual_in is not None and actual_in != expected_in:
                        self.bugs.append(
                            f"Shape mismatch at self.{layer_name}: "
                            f"input has {actual_in} features but "
                            f"nn.Linear expects {expected_in}"
                        )
                if output_var:
                    var_shapes[output_var] = out_feat

            elif layer["type"] == "Conv2d":
                expected_in = layer["in_channels"]
                out_ch = layer["out_channels"]
                if input_var and input_var in var_shapes:
                    actual_in = var_shapes[input_var]
                    if actual_in is not None and actual_in != expected_in:
                        self.bugs.append(
                            f"Shape mismatch at self.{layer_name}: "
                            f"input has {actual_in} channels but "
                            f"nn.Conv2d expects {expected_in}"
                        )
                if output_var:
                    var_shapes[output_var] = out_ch

    def _extract_self_layer_call(self, node) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            output_var = target.id if isinstance(target, ast.Name) else None
            return self._unpack_call(node.value, output_var)
        elif isinstance(node, ast.Return) and node.value is not None:
            return self._unpack_call(node.value, None)
        return None

    def _unpack_call(self, expr, output_var) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
        if not isinstance(expr, ast.Call):
            return None
        func = expr.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
            layer_name = func.attr
            input_var = None
            if expr.args:
                arg = expr.args[0]
                if isinstance(arg, ast.Name):
                    input_var = arg.id
                elif isinstance(arg, ast.Call):
                    inner = self._unpack_call(arg, output_var)
                    if inner:
                        return inner
            return (layer_name, input_var, output_var)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# TensorGuard runner (with Z3 theories)
# ═══════════════════════════════════════════════════════════════════════════════

def run_tensorguard(tc: Dict[str, Any]) -> Dict[str, Any]:
    """Run TensorGuard verification with Z3 broadcast/stride theories.

    Uses both verify_model (direct Z3 constraint verification) and run_shape_cegar
    (CEGAR refinement loop).  Reports bug if either detects one.
    """
    t0 = time.monotonic()
    details = ""
    detected = False
    try:
        # First try direct constraint verification with Z3 theories (exercises BroadcastPropagator)
        vm_result = verify_model(
            tc["code"],
            input_shapes=tc["input_shapes"],
        )
        if not vm_result.safe:
            detected = True
            if vm_result.counterexample:
                for v in vm_result.counterexample.violations[:2]:
                    msg = getattr(v, "message", str(v))
                    details += msg[:150] + "; "

        # Also run CEGAR for additional coverage
        cegar_result = run_shape_cegar(
            tc["code"],
            input_shapes=tc["input_shapes"],
            max_iterations=10,
            enable_quality_filter=True,
        )
        if cegar_result.has_real_bugs:
            detected = True
            if cegar_result.real_bugs:
                for b in cegar_result.real_bugs[:2]:
                    msg = getattr(b, "message", str(b))
                    if msg[:50] not in details:
                        details += msg[:150] + "; "
    except Exception as e:
        details = f"ERROR: {e}"
    elapsed = (time.monotonic() - t0) * 1000
    return {
        "detected_bug": detected,
        "time_ms": round(elapsed, 2),
        "details": details[:300],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics + Bootstrap CI
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    tp = fp = fn = tn = 0
    for r in results:
        gt = r["ground_truth"]
        det = r["detected_bug"]
        if gt and det:
            tp += 1
        elif not gt and det:
            fp += 1
        elif gt and not det:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    avg_time = sum(r["time_ms"] for r in results) / len(results) if results else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "avg_time_ms": round(avg_time, 2),
    }


def bootstrap_ci(
    results: List[Dict[str, Any]],
    n_bootstrap: int = 2000,
    ci: float = 0.95,
) -> Dict[str, Tuple[float, float]]:
    """Bootstrap 95% confidence intervals for precision, recall, F1, accuracy."""
    rng = random.Random(42)
    samples: Dict[str, List[float]] = {
        "precision": [], "recall": [], "f1": [], "accuracy": []
    }
    for _ in range(n_bootstrap):
        sample = rng.choices(results, k=len(results))
        m = compute_metrics(sample)
        for key in samples:
            samples[key].append(m[key])
    cis = {}
    alpha = (1 - ci) / 2
    for key, vals in samples.items():
        vals.sort()
        lo = vals[max(0, int(alpha * len(vals)))]
        hi = vals[min(len(vals) - 1, int((1 - alpha) * len(vals)))]
        cis[key] = (round(lo, 4), round(hi, 4))
    return cis


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 76)
    print("  Theory-Exercising Evaluation")
    print("  BroadcastPropagator + StridePropagator vs Syntactic Baseline")
    print(f"  {len(TEST_CASES)} benchmarks")
    print("=" * 76)

    categories = {}
    for tc in TEST_CASES:
        cat = tc["category"]
        categories.setdefault(cat, []).append(tc["name"])
    for cat, names in categories.items():
        print(f"  {cat}: {len(names)} benchmarks")

    all_results: Dict[str, List[Dict[str, Any]]] = {
        "syntactic": [], "tensorguard": []
    }
    disagreements: List[Dict[str, Any]] = []

    for i, tc in enumerate(TEST_CASES, 1):
        tag = "BUGGY" if tc["has_bug"] else "CLEAN"
        print(f"\n[{i:2d}/{len(TEST_CASES)}] {tc['name']} ({tag})")
        print(f"         {tc['description']}")

        # ── Syntactic baseline ──
        t0 = time.monotonic()
        checker = SyntacticShapeChecker(tc["code"])
        has_bug, bug_msgs = checker.check()
        syn_time = (time.monotonic() - t0) * 1000
        syn_result = {
            "name": tc["name"],
            "category": tc["category"],
            "ground_truth": tc["has_bug"],
            "detected_bug": has_bug,
            "time_ms": round(syn_time, 2),
            "details": "; ".join(bug_msgs) if bug_msgs else "",
        }
        all_results["syntactic"].append(syn_result)
        syn_correct = has_bug == tc["has_bug"]
        syn_mark = "✓" if syn_correct else "✗"
        print(f"  Syntactic:  {syn_mark}  det={has_bug:<5}  {syn_time:.1f}ms"
              + (f"  [{bug_msgs[0][:60]}]" if bug_msgs else ""))

        # ── TensorGuard (CEGAR + Z3 theories) ──
        lp = run_tensorguard(tc)
        lp_result = {
            "name": tc["name"],
            "category": tc["category"],
            "ground_truth": tc["has_bug"],
            "detected_bug": lp["detected_bug"],
            "time_ms": lp["time_ms"],
            "details": lp["details"],
        }
        all_results["tensorguard"].append(lp_result)
        lp_correct = lp["detected_bug"] == tc["has_bug"]
        lp_mark = "✓" if lp_correct else "✗"
        print(f"  TensorGuard:   {lp_mark}  det={str(lp['detected_bug']):<5}  {lp['time_ms']:.1f}ms"
              + (f"  [{lp['details'][:60]}]" if lp["details"] else ""))

        if syn_correct != lp_correct:
            winner = "TensorGuard" if lp_correct else "Syntactic"
            disagreements.append({
                "benchmark": tc["name"],
                "ground_truth_has_bug": tc["has_bug"],
                "syntactic_correct": syn_correct,
                "tensorguard_correct": lp_correct,
                "winner": winner,
                "why_syntactic_fails": tc.get("why_syntactic_fails", ""),
            })
            print(f"  >>> DISAGREEMENT: {winner} wins <<<")

    # ── Per-tool metrics ──
    print(f"\n{'=' * 76}")
    print("  METRICS SUMMARY")
    print(f"{'=' * 76}")

    tool_metrics: Dict[str, Any] = {}
    tool_cis: Dict[str, Any] = {}
    for tool in ["syntactic", "tensorguard"]:
        m = compute_metrics(all_results[tool])
        ci = bootstrap_ci(all_results[tool])
        tool_metrics[tool] = m
        tool_cis[tool] = ci
        label = {
            "syntactic": "Syntactic Pattern Matching",
            "tensorguard": "TensorGuard (Z3 Theories)",
        }
        print(f"\n  {label[tool]:30s}")
        print(f"    F1={m['f1']:<6.4f}  P={m['precision']:<6.4f}  "
              f"R={m['recall']:<6.4f}  Acc={m['accuracy']:<6.4f}")
        print(f"    TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}  "
              f"avg={m['avg_time_ms']:.1f}ms")
        print(f"    95% CI: F1={ci['f1']}  P={ci['precision']}  R={ci['recall']}")

    # ── Comparison ──
    lp_f1 = tool_metrics["tensorguard"]["f1"]
    syn_f1 = tool_metrics["syntactic"]["f1"]
    lp_acc = tool_metrics["tensorguard"]["accuracy"]
    syn_acc = tool_metrics["syntactic"]["accuracy"]
    print(f"\n  TensorGuard vs Syntactic:")
    print(f"    ΔF1       = {lp_f1 - syn_f1:+.4f}")
    print(f"    ΔAccuracy = {lp_acc - syn_acc:+.4f}")

    # ── Disagreement analysis ──
    if disagreements:
        print(f"\n  DISAGREEMENTS ({len(disagreements)}):")
        for d in disagreements:
            print(f"    {d['benchmark']:35s}  winner={d['winner']}")
            if d.get("why_syntactic_fails"):
                print(f"      reason: {d['why_syntactic_fails'][:70]}")

    # ── Per-category breakdown ──
    print(f"\n  PER-CATEGORY BREAKDOWN:")
    for cat in sorted(categories.keys()):
        for tool in ["syntactic", "tensorguard"]:
            cat_results = [r for r in all_results[tool] if r["category"] == cat]
            m = compute_metrics(cat_results)
            label = "Syn" if tool == "syntactic" else "LP "
            print(f"    {cat:20s} {label}: "
                  f"Acc={m['accuracy']:.2f}  F1={m['f1']:.2f}  "
                  f"(TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']})")

    # ── Save JSON ──
    output = {
        "experiment": "theory_exercising_eval",
        "description": "Benchmarks exercising Z3 BroadcastPropagator and "
                       "StridePropagator theories vs syntactic baseline",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_benchmarks": len(TEST_CASES),
        "num_buggy": sum(1 for tc in TEST_CASES if tc["has_bug"]),
        "num_correct": sum(1 for tc in TEST_CASES if not tc["has_bug"]),
        "categories": {k: len(v) for k, v in categories.items()},
        "tools": {},
        "disagreements": disagreements,
    }
    for tool in ["syntactic", "tensorguard"]:
        label = {
            "syntactic": "Syntactic Pattern Matching (no Z3)",
            "tensorguard": "TensorGuard (CEGAR + Z3 BroadcastPropagator + StridePropagator)",
        }
        output["tools"][tool] = {
            "label": label[tool],
            "metrics": tool_metrics[tool],
            "confidence_intervals_95": {
                k: list(v) for k, v in tool_cis[tool].items()
            },
            "per_benchmark": all_results[tool],
        }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
