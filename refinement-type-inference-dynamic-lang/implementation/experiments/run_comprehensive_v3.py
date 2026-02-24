"""
Comprehensive Experiment v3 — Four evaluation axes.

  Experiment 1: CEGAR Ablation with Feasibility Checking (15+ architectures × 3 modes)
  Experiment 2: Theory-Exercising (Broadcast + Stride)
  Experiment 3: Conditional Branch Handling (path-sensitive)
  Experiment 4: Production Architecture Evaluation (real architectures)

Outputs: experiments/comprehensive_v3_results.json
"""

from __future__ import annotations

import ast
import json
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
from src.shape_cegar import run_shape_cegar, PREDICATE_QUALITY_THRESHOLD

RESULTS_FILE = Path(__file__).parent / "comprehensive_v3_results.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Shared utilities
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
    avg_time = sum(r.get("time_ms", 0) for r in results) / len(results) if results else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "avg_time_ms": round(avg_time, 2),
    }


def bootstrap_ci(
    results: List[Dict[str, Any]],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> Dict[str, Tuple[float, float]]:
    rng = random.Random(42)
    samples: Dict[str, List[float]] = {"precision": [], "recall": [], "f1": []}
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


# ── Syntactic baseline (AST-based, no Z3) ──

class SyntacticShapeChecker:
    """Pure AST-based shape checker for nn.Module classes."""

    def __init__(self, source: str):
        self.source = source
        self.tree = ast.parse(source)
        self.layers: Dict[str, Dict[str, Any]] = {}
        self.bugs: List[str] = []

    def check(self) -> Tuple[bool, List[str]]:
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
                self.layers[attr_name] = {"type": "Linear", "in_features": in_f, "out_features": out_f}
        elif layer_type == "Conv2d" and len(call.args) >= 3:
            in_c = self._const_val(call.args[0])
            out_c = self._const_val(call.args[1])
            if in_c is not None and out_c is not None:
                self.layers[attr_name] = {"type": "Conv2d", "in_channels": in_c, "out_channels": out_c}
        elif layer_type == "ConvTranspose2d" and len(call.args) >= 3:
            in_c = self._const_val(call.args[0])
            out_c = self._const_val(call.args[1])
            if in_c is not None and out_c is not None:
                self.layers[attr_name] = {"type": "ConvTranspose2d", "in_channels": in_c, "out_channels": out_c}
        elif layer_type == "BatchNorm2d" and len(call.args) >= 1:
            nf = self._const_val(call.args[0])
            if nf is not None:
                self.layers[attr_name] = {"type": "BatchNorm2d", "num_features": nf}

    @staticmethod
    def _const_val(node: ast.expr) -> Optional[int]:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            if isinstance(node.operand, ast.Constant):
                return -node.operand.value
        if isinstance(node, ast.BinOp):
            left = SyntacticShapeChecker._const_val(node.left)
            right = SyntacticShapeChecker._const_val(node.right)
            if left is not None and right is not None:
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.FloorDiv):
                    return left // right if right != 0 else None
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
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
                        self.bugs.append(f"Shape mismatch at self.{layer_name}")
                if output_var:
                    var_shapes[output_var] = out_feat
            elif layer["type"] in ("Conv2d", "ConvTranspose2d"):
                expected_in = layer["in_channels"]
                out_ch = layer.get("out_channels", expected_in)
                if input_var and input_var in var_shapes:
                    actual_in = var_shapes[input_var]
                    if actual_in is not None and actual_in != expected_in:
                        self.bugs.append(f"Shape mismatch at self.{layer_name}")
                if output_var:
                    var_shapes[output_var] = out_ch
            elif layer["type"] == "BatchNorm2d":
                expected_nf = layer["num_features"]
                if input_var and input_var in var_shapes:
                    actual_in = var_shapes[input_var]
                    if actual_in is not None and actual_in != expected_nf:
                        self.bugs.append(f"Shape mismatch at self.{layer_name}")
                if output_var:
                    var_shapes[output_var] = var_shapes.get(input_var) if input_var else None

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


# ── Runners ──

def run_no_cegar(tc: Dict[str, Any]) -> Dict[str, Any]:
    """Single-pass ConstraintVerifier — no contract discovery loop."""
    t0 = time.monotonic()
    try:
        graph = extract_computation_graph(tc["code"])
        checker = ConstraintVerifier(graph, input_shapes=tc["input_shapes"])
        result = checker.verify()
        detected = not result.safe
        status = "UNSAFE" if detected else "SAFE"
    except Exception as e:
        detected = False
        status = f"ERROR: {e}"
    elapsed = (time.monotonic() - t0) * 1000
    return {
        "name": tc["name"],
        "ground_truth": tc["has_bug"],
        "detected_bug": detected,
        "status": status,
        "time_ms": round(elapsed, 2),
    }


def run_cegar(tc: Dict[str, Any], enable_quality_filter: bool = True) -> Dict[str, Any]:
    """CEGAR loop with configurable quality filter."""
    t0 = time.monotonic()
    try:
        cegar_result = run_shape_cegar(
            tc["code"],
            input_shapes=tc["input_shapes"],
            max_iterations=10,
            enable_quality_filter=enable_quality_filter,
        )
        detected = cegar_result.has_real_bugs
        if not detected and cegar_result.verification_result and not cegar_result.verification_result.safe:
            detected = True
        num_preds = len(cegar_result.discovered_predicates)
        num_iters = cegar_result.iterations
        status = cegar_result.final_status.name
        qr = cegar_result.predicate_quality_report
        n_rejected = qr.get("rejected", 0) if qr else 0
    except Exception as e:
        detected = False
        num_preds = 0
        num_iters = 0
        status = f"ERROR: {e}"
        n_rejected = 0
    elapsed = (time.monotonic() - t0) * 1000
    return {
        "name": tc["name"],
        "ground_truth": tc["has_bug"],
        "detected_bug": detected,
        "status": status,
        "iterations": num_iters,
        "predicates_discovered": num_preds,
        "predicates_rejected": n_rejected,
        "time_ms": round(elapsed, 2),
    }


def run_tensorguard(tc: Dict[str, Any]) -> Dict[str, Any]:
    """Full TensorGuard: verification + contract discovery with quality filter."""
    t0 = time.monotonic()
    detected = False
    details = ""
    try:
        vm_result = verify_model(tc["code"], input_shapes=tc["input_shapes"])
        if not vm_result.safe:
            detected = True
            if vm_result.counterexample:
                for v in vm_result.counterexample.violations[:2]:
                    details += getattr(v, "message", str(v))[:120] + "; "
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
                    if msg[:40] not in details:
                        details += msg[:120] + "; "
    except Exception as e:
        details = f"ERROR: {e}"
    elapsed = (time.monotonic() - t0) * 1000
    return {
        "name": tc["name"],
        "ground_truth": tc["has_bug"],
        "detected_bug": detected,
        "time_ms": round(elapsed, 2),
        "details": details[:300],
    }


def run_syntactic(tc: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.monotonic()
    checker = SyntacticShapeChecker(tc["code"])
    has_bug, bug_msgs = checker.check()
    elapsed = (time.monotonic() - t0) * 1000
    return {
        "name": tc["name"],
        "ground_truth": tc["has_bug"],
        "detected_bug": has_bug,
        "time_ms": round(elapsed, 2),
        "details": "; ".join(bug_msgs) if bug_msgs else "",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: CEGAR Ablation with Feasibility Checking (16 architectures)
# ═══════════════════════════════════════════════════════════════════════════════

CEGAR_ABLATION_CASES: List[Dict[str, Any]] = [
    # ── 1. MLP dim mismatch ──
    {
        "name": "mlp_dim_bug",
        "arch": "MLP",
        "has_bug": True,
        "code": """\
import torch.nn as nn
class BuggyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)
""",
        "input_shapes": {"x": ("batch", 768)},
    },
    # ── 2. MLP correct ──
    {
        "name": "mlp_correct",
        "arch": "MLP",
        "has_bug": False,
        "code": """\
import torch.nn as nn
class CorrectMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)
""",
        "input_shapes": {"x": ("batch", 768)},
    },
    # ── 3. CNN channel mismatch ──
    {
        "name": "cnn_channel_bug",
        "arch": "CNN",
        "has_bug": True,
        "code": """\
import torch.nn as nn
class BuggyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    # ── 4. CNN correct ──
    {
        "name": "cnn_correct",
        "arch": "CNN",
        "has_bug": False,
        "code": """\
import torch.nn as nn
class CorrectCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    # ── 5. Transformer projection mismatch ──
    {
        "name": "transformer_proj_bug",
        "arch": "Transformer",
        "has_bug": True,
        "code": """\
import torch.nn as nn
class BuggyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(512, 512)
        self.k_proj = nn.Linear(512, 512)
        self.v_proj = nn.Linear(512, 512)
        self.out_proj = nn.Linear(512, 512)
        self.ff = nn.Linear(768, 256)
    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        x = self.out_proj(v)
        return self.ff(x)
""",
        "input_shapes": {"x": ("batch", "seq_len", 512)},
    },
    # ── 6. ResNet block correct (symbolic dims) ──
    {
        "name": "resnet_block_correct",
        "arch": "ResNet",
        "has_bug": False,
        "code": """\
import torch.nn as nn
class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out
""",
        "input_shapes": {"x": ("batch", 64, "h", "w")},
    },
    # ── 7. U-Net encoder channel bug ──
    {
        "name": "unet_encoder_bug",
        "arch": "U-Net",
        "has_bug": True,
        "code": """\
import torch.nn as nn
class UNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 128, 3, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)
""",
        "input_shapes": {"x": ("batch", 3, 128, 128)},
    },
    # ── 8. GAN discriminator dim bug ──
    {
        "name": "gan_disc_bug",
        "arch": "GAN",
        "has_bug": True,
        "code": """\
import torch.nn as nn
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
""",
        "input_shapes": {"x": ("batch", 784)},
    },
    # ── 9. Autoencoder correct ──
    {
        "name": "autoencoder_correct",
        "arch": "Autoencoder",
        "has_bug": False,
        "code": """\
import torch.nn as nn
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = nn.Linear(784, 256)
        self.encoder2 = nn.Linear(256, 64)
        self.decoder1 = nn.Linear(64, 256)
        self.decoder2 = nn.Linear(256, 784)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.encoder1(x))
        x = self.relu(self.encoder2(x))
        x = self.relu(self.decoder1(x))
        return self.decoder2(x)
""",
        "input_shapes": {"x": ("batch", 784)},
    },
    # ── 10. LSTM-style hidden dim bug ──
    {
        "name": "lstm_style_bug",
        "arch": "RNN",
        "has_bug": True,
        "code": """\
import torch.nn as nn
class LSTMStyleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(256, 128)
        self.gate_proj = nn.Linear(128, 128)
        self.output_proj = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.input_proj(x))
        x = self.relu(self.gate_proj(x))
        return self.output_proj(x)
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    # ── 11. Broadcast add with symbolic dims — mismatch ──
    {
        "name": "broadcast_sym_bug",
        "arch": "BroadcastNet",
        "has_bug": True,
        "code": """\
import torch.nn as nn
class BroadcastSymBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_a = nn.Linear(256, 128)
        self.proj_b = nn.Linear(256, 64)
    def forward(self, x):
        a = self.proj_a(x)
        b = self.proj_b(x)
        return a + b
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    # ── 12. Reshape flow — correct ──
    {
        "name": "reshape_flow_correct",
        "arch": "ReshapeNet",
        "has_bug": False,
        "code": """\
import torch.nn as nn
class ReshapeCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
""",
        "input_shapes": {"x": ("batch", 784)},
    },
    # ── 13. Multi-branch merge bug ──
    {
        "name": "multi_branch_bug",
        "arch": "MultiBranch",
        "has_bug": True,
        "code": """\
import torch.nn as nn
class MultiBranchBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = nn.Linear(512, 256)
        self.branch2 = nn.Linear(512, 128)
        self.fc_out = nn.Linear(256, 10)
    def forward(self, x):
        a = self.branch1(x)
        b = self.branch2(x)
        merged = a + b
        return self.fc_out(merged)
""",
        "input_shapes": {"x": ("batch", 512)},
    },
    # ── 14. Deep chain correct ──
    {
        "name": "deep_chain_correct",
        "arch": "DeepChain",
        "has_bug": False,
        "code": """\
import torch.nn as nn
class DeepChain(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.fc5(x)
""",
        "input_shapes": {"x": ("batch", 512)},
    },
    # ── 15. Conv with BN mismatch ──
    {
        "name": "conv_bn_bug",
        "arch": "ConvBN",
        "has_bug": True,
        "code": """\
import torch.nn as nn
class ConvBNBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.conv2(x))
        return x
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    # ── 16. Attention-style with symbolic seq_len — correct ──
    {
        "name": "attention_correct",
        "arch": "Attention",
        "has_bug": False,
        "code": """\
import torch.nn as nn
class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(256, 256)
        self.k_proj = nn.Linear(256, 256)
        self.v_proj = nn.Linear(256, 256)
        self.out_proj = nn.Linear(256, 256)
    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.out_proj(v)
""",
        "input_shapes": {"x": ("batch", "seq_len", 256)},
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Theory-Exercising (Broadcast + Stride)
# ═══════════════════════════════════════════════════════════════════════════════

THEORY_CASES: List[Dict[str, Any]] = [
    # ── Broadcast: parallel projections with add ──
    {
        "name": "broadcast_parallel_bug",
        "category": "broadcast",
        "has_bug": True,
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
    {
        "name": "broadcast_parallel_safe",
        "category": "broadcast",
        "has_bug": False,
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
    # ── Broadcast: cross-rank (3D + 2D) ──
    {
        "name": "broadcast_cross_rank_bug",
        "category": "broadcast",
        "has_bug": True,
        "code": """\
import torch.nn as nn
class CrossRankBug(nn.Module):
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
    {
        "name": "broadcast_cross_rank_safe",
        "category": "broadcast",
        "has_bug": False,
        "code": """\
import torch.nn as nn
class CrossRankSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(256, 256)
        self.bias_proj = nn.Linear(256, 256)
    def forward(self, x, bias):
        x = self.proj(x)
        bias = self.bias_proj(bias)
        return x + bias
""",
        "input_shapes": {"x": ("batch", "seq", 256), "bias": ("batch", 256)},
    },
    # ── Broadcast: mul incompatible ──
    {
        "name": "broadcast_mul_bug",
        "category": "broadcast",
        "has_bug": True,
        "code": """\
import torch.nn as nn
class BroadcastMulBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_feat = nn.Linear(512, 256)
        self.fc_gate = nn.Linear(512, 128)
    def forward(self, x):
        feat = self.fc_feat(x)
        gate = self.fc_gate(x)
        return feat * gate
""",
        "input_shapes": {"x": ("batch", 512)},
    },
    # ── Stride: reshape element count preserved ──
    {
        "name": "stride_reshape_correct",
        "category": "stride",
        "has_bug": False,
        "code": """\
import torch.nn as nn
class ReshapeCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
""",
        "input_shapes": {"x": ("batch", 784)},
    },
    # ── Stride: conv chain with stride ──
    {
        "name": "stride_conv_bug",
        "category": "stride",
        "has_bug": True,
        "code": """\
import torch.nn as nn
class StrideConvBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 128, 3, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)
""",
        "input_shapes": {"x": ("batch", 3, "h", "w")},
    },
    {
        "name": "stride_conv_correct",
        "category": "stride",
        "has_bug": False,
        "code": """\
import torch.nn as nn
class StrideConvCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)
""",
        "input_shapes": {"x": ("batch", 3, "h", "w")},
    },
    # ── Combined: reshape after broadcast ──
    {
        "name": "combined_reshape_broadcast_bug",
        "category": "combined",
        "has_bug": True,
        "code": """\
import torch.nn as nn
class CombinedBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(512, 256)
        self.fc_b = nn.Linear(512, 128)
        self.fc_out = nn.Linear(256, 10)
    def forward(self, x):
        a = self.fc_a(x)
        b = self.fc_b(x)
        s = a + b
        return self.fc_out(s)
""",
        "input_shapes": {"x": ("batch", 512)},
    },
    {
        "name": "combined_chain_safe",
        "category": "combined",
        "has_bug": False,
        "code": """\
import torch.nn as nn
class CombinedSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(512, 256)
        self.fc_b = nn.Linear(512, 256)
        self.fc_out = nn.Linear(256, 10)
    def forward(self, x):
        a = self.fc_a(x)
        b = self.fc_b(x)
        s = a + b
        return self.fc_out(s)
""",
        "input_shapes": {"x": ("batch", 512)},
    },
    # ── Broadcast: triple add chain ──
    {
        "name": "triple_add_chain_bug",
        "category": "broadcast",
        "has_bug": True,
        "code": """\
import torch.nn as nn
class TripleAddBug(nn.Module):
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
    {
        "name": "triple_add_chain_safe",
        "category": "broadcast",
        "has_bug": False,
        "code": """\
import torch.nn as nn
class TripleAddSafe(nn.Module):
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
# EXPERIMENT 3: Conditional Branch Handling
# ═══════════════════════════════════════════════════════════════════════════════

CONDITIONAL_CASES: List[Dict[str, Any]] = [
    # ── Train/eval mode with different paths — both safe ──
    {
        "name": "training_dropout_safe",
        "category": "training_guard",
        "has_bug": False,
        "code": """\
import torch.nn as nn
class TrainingDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        if self.training:
            x = self.dropout(x)
        return self.fc2(x)
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    # ── Conditional with mismatched dims in one branch ──
    {
        "name": "cond_branch_bug",
        "category": "conditional",
        "has_bug": True,
        "code": """\
import torch.nn as nn
class CondBranchBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    # ── BN in training mode — correct ──
    {
        "name": "bn_training_safe",
        "category": "training_guard",
        "has_bug": False,
        "code": """\
import torch.nn as nn
class BNTraining(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        if self.training:
            x = self.bn1(x)
        return self.relu(x)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    # ── Conditional path with shape error only reachable on one branch ──
    {
        "name": "cond_path_bug_train_only",
        "category": "conditional",
        "has_bug": True,
        "code": """\
import torch.nn as nn
class CondPathBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc_train = nn.Linear(128, 10)
        self.fc_eval = nn.Linear(256, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        if self.training:
            return self.fc_train(x)
        return self.fc_eval(x)
""",
        "input_shapes": {"x": ("batch", 512)},
    },
    # ── Multi-path merge — correct ──
    {
        "name": "multi_path_merge_safe",
        "category": "conditional",
        "has_bug": False,
        "code": """\
import torch.nn as nn
class MultiPathSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    # ── Guard with BN channel mismatch ──
    {
        "name": "guard_bn_mismatch_bug",
        "category": "training_guard",
        "has_bug": True,
        "code": """\
import torch.nn as nn
class GuardBNBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        return self.relu(x)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Production Architecture Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

PRODUCTION_CASES: List[Dict[str, Any]] = [
    # ── ResNet BasicBlock correct ──
    {
        "name": "resnet_basicblock_correct",
        "arch": "ResNet",
        "has_bug": False,
        "code": """\
import torch.nn as nn
class BasicBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out
""",
        "input_shapes": {"x": ("batch", 64, "h", "w")},
    },
    # ── BERT attention bug: proj dim mismatch ──
    {
        "name": "bert_attention_bug",
        "arch": "BERT",
        "has_bug": True,
        "code": """\
import torch.nn as nn
class BertAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(768, 768)
        self.key = nn.Linear(768, 768)
        self.value = nn.Linear(768, 768)
        self.dense = nn.Linear(512, 768)
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        return self.dense(v)
""",
        "input_shapes": {"x": ("batch", "seq_len", 768)},
    },
    # ── Transformer encoder block correct ──
    {
        "name": "transformer_encoder_correct",
        "arch": "Transformer",
        "has_bug": False,
        "code": """\
import torch.nn as nn
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(512, 512)
        self.k_proj = nn.Linear(512, 512)
        self.v_proj = nn.Linear(512, 512)
        self.out_proj = nn.Linear(512, 512)
        self.ff1 = nn.Linear(512, 2048)
        self.ff2 = nn.Linear(2048, 512)
        self.relu = nn.ReLU()
    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        x = self.out_proj(v)
        x = self.relu(self.ff1(x))
        return self.ff2(x)
""",
        "input_shapes": {"x": ("batch", "seq_len", 512)},
    },
    # ── U-Net skip connection bug ──
    {
        "name": "unet_skip_bug",
        "arch": "U-Net",
        "has_bug": True,
        "code": """\
import torch.nn as nn
class UNetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.dec_conv1 = nn.Conv2d(64, 64, 3, padding=1)
    def forward(self, x):
        x = self.enc_conv1(x)
        x = self.enc_conv2(x)
        return self.dec_conv1(x)
""",
        "input_shapes": {"x": ("batch", 3, 64, 64)},
    },
    # ── GAN generator correct ──
    {
        "name": "gan_generator_correct",
        "arch": "GAN",
        "has_bug": False,
        "code": """\
import torch.nn as nn
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 784)
        self.relu = nn.ReLU()
    def forward(self, z):
        z = self.relu(self.fc1(z))
        z = self.relu(self.fc2(z))
        return self.fc3(z)
""",
        "input_shapes": {"z": ("batch", 100)},
    },
    # ── VAE encoder bug: latent dim mismatch ──
    {
        "name": "vae_encoder_bug",
        "arch": "VAE",
        "has_bug": True,
        "code": """\
import torch.nn as nn
class VAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc_mu = nn.Linear(400, 20)
        self.fc_logvar = nn.Linear(200, 20)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
""",
        "input_shapes": {"x": ("batch", 784)},
    },
    # ── AlexNet-style classifier correct ──
    {
        "name": "alexnet_classifier_correct",
        "arch": "AlexNet",
        "has_bug": False,
        "code": """\
import torch.nn as nn
class AlexNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
""",
        "input_shapes": {"x": ("batch", 4096)},
    },
    # ── SE-Net squeeze-excitation bug ──
    {
        "name": "senet_bug",
        "arch": "SE-Net",
        "has_bug": True,
        "code": """\
import torch.nn as nn
class SEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(256, 256, 3, padding=1)
        self.fc1 = nn.Linear(256, 16)
        self.fc2 = nn.Linear(32, 256)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        scale = self.relu(self.fc1(x))
        scale = self.fc2(scale)
        return x
""",
        "input_shapes": {"x": ("batch", 256, "h", "w")},
    },
    # ── Inception-style multi-branch correct ──
    {
        "name": "inception_branch_correct",
        "arch": "Inception",
        "has_bug": False,
        "code": """\
import torch.nn as nn
class InceptionBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = nn.Conv2d(256, 64, 1)
        self.branch2_1 = nn.Conv2d(256, 64, 1)
        self.branch2_2 = nn.Conv2d(64, 64, 3, padding=1)
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2_2(self.branch2_1(x))
        return b1
""",
        "input_shapes": {"x": ("batch", 256, "h", "w")},
    },
    # ── FPN feature pyramid bug: lateral dim mismatch ──
    {
        "name": "fpn_lateral_bug",
        "arch": "FPN",
        "has_bug": True,
        "code": """\
import torch.nn as nn
class FPNBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.lateral = nn.Conv2d(512, 256, 1)
        self.smooth = nn.Conv2d(128, 256, 3, padding=1)
    def forward(self, x):
        lat = self.lateral(x)
        return self.smooth(lat)
""",
        "input_shapes": {"x": ("batch", 512, "h", "w")},
    },
    # ── MobileNet depthwise correct ──
    {
        "name": "mobilenet_correct",
        "arch": "MobileNet",
        "has_bug": False,
        "code": """\
import torch.nn as nn
class MobileBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(32, 64, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.conv3(x)
""",
        "input_shapes": {"x": ("batch", 32, "h", "w")},
    },
    # ── GPT block: FFN dim mismatch ──
    {
        "name": "gpt_ffn_bug",
        "arch": "GPT",
        "has_bug": True,
        "code": """\
import torch.nn as nn
class GPTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_proj = nn.Linear(768, 768)
        self.ff1 = nn.Linear(768, 3072)
        self.ff2 = nn.Linear(2048, 768)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.attn_proj(x)
        x = self.relu(self.ff1(x))
        return self.ff2(x)
""",
        "input_shapes": {"x": ("batch", "seq_len", 768)},
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment runners
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment_1() -> Dict[str, Any]:
    """CEGAR Ablation: 16 architectures × 3 modes."""
    print("\n" + "=" * 76)
    print("  EXPERIMENT 1: CEGAR Ablation with Feasibility Checking")
    print(f"  {len(CEGAR_ABLATION_CASES)} architectures × 3 modes")
    print("=" * 76)

    modes = [
        ("no_cegar", "No contract discovery (single-pass verification)"),
        ("cegar_no_filter", "CEGAR — quality filter OFF"),
        ("cegar_with_filter", "CEGAR — quality filter ON"),
    ]

    all_configs: Dict[str, Any] = {}

    for mode_key, mode_label in modes:
        print(f"\n{'─' * 72}")
        print(f"  Mode: {mode_label}")
        print(f"{'─' * 72}")
        per_bench: List[Dict[str, Any]] = []
        for tc in CEGAR_ABLATION_CASES:
            try:
                if mode_key == "no_cegar":
                    r = run_no_cegar(tc)
                elif mode_key == "cegar_no_filter":
                    r = run_cegar(tc, enable_quality_filter=False)
                else:
                    r = run_cegar(tc, enable_quality_filter=True)
            except Exception as exc:
                r = {
                    "name": tc["name"],
                    "ground_truth": tc["has_bug"],
                    "detected_bug": False,
                    "status": f"ERROR: {exc}",
                    "time_ms": 0.0,
                }
            mark = "✓" if r["detected_bug"] == r["ground_truth"] else "✗"
            print(f"  {mark} {r['name']:30s}  bug={r['ground_truth']}  det={r['detected_bug']}  "
                  f"{r.get('time_ms', 0):.0f}ms")
            per_bench.append(r)

        metrics = compute_metrics(per_bench)
        cis = bootstrap_ci(per_bench)
        total_time = sum(r.get("time_ms", 0) for r in per_bench)
        total_preds = sum(r.get("predicates_discovered", 0) for r in per_bench)
        total_rej = sum(r.get("predicates_rejected", 0) for r in per_bench)

        print(f"\n  F1={metrics['f1']}  P={metrics['precision']}  R={metrics['recall']}  "
              f"TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  TN={metrics['tn']}")

        all_configs[mode_key] = {
            "label": mode_label,
            "metrics": metrics,
            "confidence_intervals_95": {k: list(v) for k, v in cis.items()},
            "total_predicates": total_preds,
            "total_rejected": total_rej,
            "total_time_ms": round(total_time, 2),
            "per_benchmark": per_bench,
        }

    # Summary
    no_f1 = all_configs["no_cegar"]["metrics"]["f1"]
    unf_f1 = all_configs["cegar_no_filter"]["metrics"]["f1"]
    filt_f1 = all_configs["cegar_with_filter"]["metrics"]["f1"]
    print(f"\n  Quality-filtered CEGAR vs no-CEGAR:  ΔF1 = {filt_f1 - no_f1:+.4f}")
    print(f"  Quality-filtered vs unfiltered CEGAR: ΔF1 = {filt_f1 - unf_f1:+.4f}")

    return {
        "experiment": "cegar_ablation",
        "num_test_cases": len(CEGAR_ABLATION_CASES),
        "configs": all_configs,
        "summary": {
            "no_cegar_f1": no_f1,
            "cegar_no_filter_f1": unf_f1,
            "cegar_with_filter_f1": filt_f1,
            "filter_vs_no_cegar_delta_f1": round(filt_f1 - no_f1, 4),
            "filter_vs_unfiltered_delta_f1": round(filt_f1 - unf_f1, 4),
        },
    }


def run_experiment_2() -> Dict[str, Any]:
    """Theory-Exercising: Broadcast + Stride benchmarks."""
    print("\n" + "=" * 76)
    print("  EXPERIMENT 2: Theory-Exercising (Broadcast + Stride)")
    print(f"  {len(THEORY_CASES)} benchmarks — TensorGuard vs Syntactic")
    print("=" * 76)

    categories: Dict[str, List[str]] = {}
    for tc in THEORY_CASES:
        categories.setdefault(tc["category"], []).append(tc["name"])
    for cat, names in categories.items():
        print(f"  {cat}: {len(names)} benchmarks")

    all_results: Dict[str, List[Dict[str, Any]]] = {"syntactic": [], "tensorguard": []}
    disagreements: List[Dict[str, Any]] = []

    for i, tc in enumerate(THEORY_CASES, 1):
        tag = "BUGGY" if tc["has_bug"] else "CLEAN"
        print(f"\n[{i:2d}/{len(THEORY_CASES)}] {tc['name']} ({tag})")

        # Syntactic
        sr = run_syntactic(tc)
        all_results["syntactic"].append(sr)
        syn_ok = sr["detected_bug"] == sr["ground_truth"]

        # TensorGuard
        lr = run_tensorguard(tc)
        all_results["tensorguard"].append(lr)
        lp_ok = lr["detected_bug"] == lr["ground_truth"]

        syn_mark = "✓" if syn_ok else "✗"
        lp_mark = "✓" if lp_ok else "✗"
        print(f"  Syntactic: {syn_mark}  det={sr['detected_bug']}   TensorGuard: {lp_mark}  det={lr['detected_bug']}  {lr['time_ms']:.0f}ms")

        if syn_ok != lp_ok:
            winner = "TensorGuard" if lp_ok else "Syntactic"
            disagreements.append({
                "benchmark": tc["name"],
                "ground_truth_has_bug": tc["has_bug"],
                "winner": winner,
            })

    tool_metrics: Dict[str, Any] = {}
    tool_cis: Dict[str, Any] = {}
    for tool in ["syntactic", "tensorguard"]:
        m = compute_metrics(all_results[tool])
        ci = bootstrap_ci(all_results[tool])
        tool_metrics[tool] = m
        tool_cis[tool] = ci

    lp_f1 = tool_metrics["tensorguard"]["f1"]
    syn_f1 = tool_metrics["syntactic"]["f1"]
    print(f"\n  Syntactic: F1={syn_f1}  TensorGuard: F1={lp_f1}  ΔF1={lp_f1 - syn_f1:+.4f}")

    # Per-category breakdown
    cat_breakdown: Dict[str, Dict[str, Any]] = {}
    for cat in sorted(categories.keys()):
        cat_breakdown[cat] = {}
        for tool in ["syntactic", "tensorguard"]:
            cat_results = [r for r in all_results[tool] if any(
                tc["name"] == r["name"] and tc["category"] == cat for tc in THEORY_CASES
            )]
            cat_breakdown[cat][tool] = compute_metrics(cat_results)

    return {
        "experiment": "theory_exercising",
        "num_benchmarks": len(THEORY_CASES),
        "categories": {k: len(v) for k, v in categories.items()},
        "tools": {
            tool: {
                "metrics": tool_metrics[tool],
                "confidence_intervals_95": {k: list(v) for k, v in tool_cis[tool].items()},
                "per_benchmark": all_results[tool],
            }
            for tool in ["syntactic", "tensorguard"]
        },
        "per_category": cat_breakdown,
        "disagreements": disagreements,
    }


def run_experiment_3() -> Dict[str, Any]:
    """Conditional Branch Handling: path-sensitive analysis."""
    print("\n" + "=" * 76)
    print("  EXPERIMENT 3: Conditional Branch Handling")
    print(f"  {len(CONDITIONAL_CASES)} benchmarks — TensorGuard vs Syntactic")
    print("=" * 76)

    all_results: Dict[str, List[Dict[str, Any]]] = {"syntactic": [], "tensorguard": []}

    for i, tc in enumerate(CONDITIONAL_CASES, 1):
        tag = "BUGGY" if tc["has_bug"] else "CLEAN"
        print(f"\n[{i:2d}/{len(CONDITIONAL_CASES)}] {tc['name']} ({tag})")

        sr = run_syntactic(tc)
        all_results["syntactic"].append(sr)

        lr = run_tensorguard(tc)
        all_results["tensorguard"].append(lr)

        syn_ok = sr["detected_bug"] == sr["ground_truth"]
        lp_ok = lr["detected_bug"] == lr["ground_truth"]
        syn_mark = "✓" if syn_ok else "✗"
        lp_mark = "✓" if lp_ok else "✗"
        print(f"  Syntactic: {syn_mark}  det={sr['detected_bug']}   TensorGuard: {lp_mark}  det={lr['detected_bug']}  {lr['time_ms']:.0f}ms")

    tool_metrics: Dict[str, Any] = {}
    tool_cis: Dict[str, Any] = {}
    for tool in ["syntactic", "tensorguard"]:
        m = compute_metrics(all_results[tool])
        ci = bootstrap_ci(all_results[tool])
        tool_metrics[tool] = m
        tool_cis[tool] = ci

    lp_f1 = tool_metrics["tensorguard"]["f1"]
    syn_f1 = tool_metrics["syntactic"]["f1"]
    print(f"\n  Syntactic: F1={syn_f1}  TensorGuard: F1={lp_f1}  ΔF1={lp_f1 - syn_f1:+.4f}")

    # FP analysis
    syn_fp = tool_metrics["syntactic"]["fp"]
    lp_fp = tool_metrics["tensorguard"]["fp"]
    print(f"  FP reduction: Syntactic={syn_fp}  TensorGuard={lp_fp}")

    return {
        "experiment": "conditional_branch_handling",
        "num_benchmarks": len(CONDITIONAL_CASES),
        "tools": {
            tool: {
                "metrics": tool_metrics[tool],
                "confidence_intervals_95": {k: list(v) for k, v in tool_cis[tool].items()},
                "per_benchmark": all_results[tool],
            }
            for tool in ["syntactic", "tensorguard"]
        },
        "fp_reduction": {
            "syntactic_fp": syn_fp,
            "tensorguard_fp": lp_fp,
            "delta": syn_fp - lp_fp,
        },
    }


def run_experiment_4() -> Dict[str, Any]:
    """Production Architecture Evaluation."""
    print("\n" + "=" * 76)
    print("  EXPERIMENT 4: Production Architecture Evaluation")
    print(f"  {len(PRODUCTION_CASES)} architectures — TensorGuard vs Syntactic")
    print("=" * 76)

    all_results: Dict[str, List[Dict[str, Any]]] = {"syntactic": [], "tensorguard": []}

    for i, tc in enumerate(PRODUCTION_CASES, 1):
        tag = "BUGGY" if tc["has_bug"] else "CLEAN"
        print(f"\n[{i:2d}/{len(PRODUCTION_CASES)}] {tc['name']} ({tag}) — {tc['arch']}")

        sr = run_syntactic(tc)
        sr["arch"] = tc["arch"]
        all_results["syntactic"].append(sr)

        lr = run_tensorguard(tc)
        lr["arch"] = tc["arch"]
        all_results["tensorguard"].append(lr)

        syn_ok = sr["detected_bug"] == sr["ground_truth"]
        lp_ok = lr["detected_bug"] == lr["ground_truth"]
        syn_mark = "✓" if syn_ok else "✗"
        lp_mark = "✓" if lp_ok else "✗"
        print(f"  Syntactic: {syn_mark}  det={sr['detected_bug']}   TensorGuard: {lp_mark}  det={lr['detected_bug']}  {lr['time_ms']:.0f}ms")

    tool_metrics: Dict[str, Any] = {}
    tool_cis: Dict[str, Any] = {}
    for tool in ["syntactic", "tensorguard"]:
        m = compute_metrics(all_results[tool])
        ci = bootstrap_ci(all_results[tool])
        tool_metrics[tool] = m
        tool_cis[tool] = ci

    lp_f1 = tool_metrics["tensorguard"]["f1"]
    syn_f1 = tool_metrics["syntactic"]["f1"]
    print(f"\n  Syntactic: F1={syn_f1}  TensorGuard: F1={lp_f1}  ΔF1={lp_f1 - syn_f1:+.4f}")

    # Per-architecture breakdown
    arch_breakdown: Dict[str, Dict[str, Any]] = {}
    for r in all_results["tensorguard"]:
        arch = r.get("arch", "Unknown")
        arch_breakdown.setdefault(arch, []).append(r)
    arch_metrics = {arch: compute_metrics(results) for arch, results in arch_breakdown.items()}

    print(f"\n  PER-ARCHITECTURE (TensorGuard):")
    for arch, m in sorted(arch_metrics.items()):
        print(f"    {arch:15s}  F1={m['f1']}  P={m['precision']}  R={m['recall']}")

    return {
        "experiment": "production_architecture_eval",
        "num_benchmarks": len(PRODUCTION_CASES),
        "architectures": sorted(set(tc["arch"] for tc in PRODUCTION_CASES)),
        "tools": {
            tool: {
                "metrics": tool_metrics[tool],
                "confidence_intervals_95": {k: list(v) for k, v in tool_cis[tool].items()},
                "per_benchmark": all_results[tool],
            }
            for tool in ["syntactic", "tensorguard"]
        },
        "per_architecture": arch_metrics,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t_start = time.monotonic()

    print("=" * 76)
    print("  COMPREHENSIVE EXPERIMENT v3")
    print("  Four evaluation axes — CEGAR ablation, theory exercising,")
    print("  conditional branches, and production architectures")
    print("=" * 76)

    exp1 = run_experiment_1()
    exp2 = run_experiment_2()
    exp3 = run_experiment_3()
    exp4 = run_experiment_4()

    total_time = (time.monotonic() - t_start) * 1000

    # ── Final summary ──
    print("\n" + "=" * 76)
    print("  FINAL SUMMARY")
    print("=" * 76)

    print(f"\n  Experiment 1 (CEGAR Ablation):")
    print(f"    No-CEGAR F1:        {exp1['summary']['no_cegar_f1']}")
    print(f"    CEGAR-unfiltered F1: {exp1['summary']['cegar_no_filter_f1']}")
    print(f"    CEGAR-filtered F1:  {exp1['summary']['cegar_with_filter_f1']}")

    print(f"\n  Experiment 2 (Theory-Exercising):")
    print(f"    Syntactic F1:  {exp2['tools']['syntactic']['metrics']['f1']}")
    print(f"    TensorGuard F1:   {exp2['tools']['tensorguard']['metrics']['f1']}")

    print(f"\n  Experiment 3 (Conditional Branches):")
    print(f"    Syntactic F1:  {exp3['tools']['syntactic']['metrics']['f1']}")
    print(f"    TensorGuard F1:   {exp3['tools']['tensorguard']['metrics']['f1']}")
    print(f"    FP reduction:  {exp3['fp_reduction']['delta']}")

    print(f"\n  Experiment 4 (Production Architectures):")
    print(f"    Syntactic F1:  {exp4['tools']['syntactic']['metrics']['f1']}")
    print(f"    TensorGuard F1:   {exp4['tools']['tensorguard']['metrics']['f1']}")

    total_benchmarks = (
        len(CEGAR_ABLATION_CASES) + len(THEORY_CASES)
        + len(CONDITIONAL_CASES) + len(PRODUCTION_CASES)
    )
    print(f"\n  Total benchmarks: {total_benchmarks}")
    print(f"  Total time: {total_time / 1000:.1f}s")

    # ── Save JSON ──
    output = {
        "experiment": "comprehensive_v3",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_benchmarks": total_benchmarks,
        "total_time_ms": round(total_time, 2),
        "predicate_quality_threshold": PREDICATE_QUALITY_THRESHOLD,
        "experiment_1_cegar_ablation": exp1,
        "experiment_2_theory_exercising": exp2,
        "experiment_3_conditional_branches": exp3,
        "experiment_4_production_architectures": exp4,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
