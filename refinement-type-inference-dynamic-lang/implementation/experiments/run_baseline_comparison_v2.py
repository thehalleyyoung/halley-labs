"""
Baseline Comparison Experiment v2.

Compares three tools on 12 nn.Module benchmarks (mix of buggy and correct):
  1. Syntactic Pattern Matching — AST-based shape checker, no Z3/CEGAR
  2. LLM (gpt-4.1-nano)        — if OPENAI_API_KEY is available
  3. TensorGuard (CEGAR)           — full run_shape_cegar with quality filter

Outputs: experiments/baseline_comparison_v2_results.json
"""

from __future__ import annotations

import ast
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_checker import extract_computation_graph, BoundedModelChecker, ConstraintVerifier, Device, Phase
from src.shape_cegar import run_shape_cegar, PREDICATE_QUALITY_THRESHOLD

RESULTS_FILE = Path(__file__).parent / "baseline_comparison_v2_results.json"

# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark models — 12 nn.Module classes (7 buggy, 5 correct)
# Same 10 as CEGAR ablation v2 + 2 extras for ≥ 12
# ═══════════════════════════════════════════════════════════════════════════════

TEST_CASES: List[Dict[str, Any]] = [
    # ── 1. MLP with bug: mismatched intermediate dimensions ──
    {
        "name": "mlp_bug",
        "arch": "MLP",
        "has_bug": True,
        "description": "MLP with mismatched intermediate dims (256 -> Linear(128,10))",
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
        "description": "MLP with consistent dimensions",
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
    # ── 3. CNN with bug: channel mismatch between conv layers ──
    {
        "name": "cnn_bug",
        "arch": "CNN",
        "has_bug": True,
        "description": "CNN with channel mismatch (conv1 outputs 32, conv2 expects 64)",
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
        "description": "CNN with correct channel dimensions",
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
    # ── 5. Transformer with bug: proj dim mismatch ──
    {
        "name": "transformer_bug",
        "arch": "Transformer",
        "has_bug": True,
        "description": "Transformer with mismatched projection (512 -> Linear(768,256))",
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
    # ── 6. ResNet block correct (skip connection) ──
    {
        "name": "resnet_correct",
        "arch": "ResNet",
        "has_bug": False,
        "description": "ResNet residual block with matching skip connection dims",
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
    # ── 7. U-Net encoder with bug: channel mismatch ──
    {
        "name": "unet_bug",
        "arch": "U-Net",
        "has_bug": True,
        "description": "U-Net encoder with channel mismatch (conv1 outputs 64, conv2 expects 32)",
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
    # ── 8. GAN discriminator with bug: intermediate dim mismatch ──
    {
        "name": "gan_bug",
        "arch": "GAN",
        "has_bug": True,
        "description": "GAN discriminator: fc1 outputs 512 but fc2 expects 256",
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
        "description": "Autoencoder with matching encoder/decoder dimensions",
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
    # ── 10. LSTM-style with bug: hidden dim mismatch ──
    {
        "name": "lstm_style_bug",
        "arch": "LSTM",
        "has_bug": True,
        "description": "LSTM-style network: input proj 256->128 but output proj expects 64",
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
    # ── 11. Deep CNN with bug: flattening dim mismatch into FC ──
    {
        "name": "deep_cnn_fc_bug",
        "arch": "CNN",
        "has_bug": True,
        "description": "Deep CNN: conv output channels 128 but fc1 expects wrong flattened dim",
        "code": """\
import torch.nn as nn
class DeepCNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.fc1(x)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    # ── 12. Correct multi-head attention style ──
    {
        "name": "multihead_correct",
        "arch": "Transformer",
        "has_bug": False,
        "description": "Multi-head attention projections with matching dims",
        "code": """\
import torch.nn as nn
class MultiHeadBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(256, 256)
        self.k_proj = nn.Linear(256, 256)
        self.v_proj = nn.Linear(256, 256)
        self.out_proj = nn.Linear(256, 256)
        self.ff1 = nn.Linear(256, 512)
        self.ff2 = nn.Linear(512, 256)
        self.relu = nn.ReLU()
    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        x = self.out_proj(v)
        x = self.relu(self.ff1(x))
        return self.ff2(x)
""",
        "input_shapes": {"x": ("batch", "seq_len", 256)},
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline 1: Syntactic Pattern Matching (AST-based shape checker)
# ═══════════════════════════════════════════════════════════════════════════════

class SyntacticShapeChecker:
    """Pure AST-based shape checker for nn.Module classes.

    Parses layer definitions from __init__ and traces data flow in forward().
    Uses concrete arithmetic only — no Z3, no CEGAR.
    """

    def __init__(self, source: str):
        self.source = source
        self.tree = ast.parse(source)
        self.layers: Dict[str, Dict[str, Any]] = {}  # attr_name -> {type, in, out, ...}
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
        """Extract nn.Linear(in, out) and nn.Conv2d(in, out, k) from __init__."""
        for node in ast.walk(cls):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                            and isinstance(node.value, ast.Call)):
                        self._parse_layer_call(target.attr, node.value)

    def _parse_layer_call(self, attr_name: str, call: ast.Call):
        """Parse nn.Linear(...), nn.Conv2d(...), etc."""
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
        """Walk forward() and check shape compatibility at each layer call."""
        forward_fn = None
        for item in cls.body:
            if isinstance(item, ast.FunctionDef) and item.name == "forward":
                forward_fn = item
                break
        if forward_fn is None:
            return

        # Track the "current output features/channels" flowing through variables
        # var_name -> last known output dim (features for Linear, channels for Conv2d)
        var_shapes: Dict[str, Optional[int]] = {}

        for stmt in ast.walk(forward_fn):
            call_info = self._extract_self_layer_call(stmt)
            if call_info is None:
                continue
            layer_name, input_var, output_var = call_info

            if layer_name not in self.layers:
                # Non-layer call (relu, bn, etc.) — pass through shape
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
        """Extract (layer_name, input_var, output_var) from statements like:
           x = self.layer(x)
           return self.layer(x)
           x = self.relu(self.layer(x))
        """
        # Pattern: var = self.layer(arg) or var = self.act(self.layer(arg))
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            output_var = target.id if isinstance(target, ast.Name) else None
            return self._unpack_call(node.value, output_var)
        elif isinstance(node, ast.Return) and node.value is not None:
            return self._unpack_call(node.value, None)
        return None

    def _unpack_call(self, expr, output_var) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
        """Recursively unpack self.layer(x) or self.act(self.layer(x))."""
        if not isinstance(expr, ast.Call):
            return None

        func = expr.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
            layer_name = func.attr
            # Get input var
            input_var = None
            if expr.args:
                arg = expr.args[0]
                if isinstance(arg, ast.Name):
                    input_var = arg.id
                elif isinstance(arg, ast.Call):
                    # Nested: self.act(self.inner(x)) — process inner first
                    inner = self._unpack_call(arg, output_var)
                    if inner:
                        return inner
            return (layer_name, input_var, output_var)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline 2: LLM-based (gpt-4.1-nano)
# ═══════════════════════════════════════════════════════════════════════════════

LLM_PROMPT = """\
Analyze this PyTorch nn.Module for shape errors. Check if each layer's \
input dimensions match the previous layer's output dimensions in the forward() method.
Return ONLY valid JSON (no markdown): \
{"has_bug": true/false, "bug_description": "...", "bug_location": "..."}"""


def run_llm_check(client, code: str) -> Tuple[bool, float, str]:
    """Query gpt-4.1-nano. Returns (found_bug, time_ms, raw_response)."""
    t0 = time.monotonic()
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": LLM_PROMPT},
                {"role": "user", "content": code},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        elapsed = (time.monotonic() - t0) * 1000
        text = resp.choices[0].message.content.strip()
        # Parse JSON response
        try:
            # Strip markdown fences if present
            clean = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
            clean = re.sub(r'\s*```$', '', clean, flags=re.MULTILINE)
            data = json.loads(clean)
            found_bug = bool(data.get("has_bug", False))
        except (json.JSONDecodeError, AttributeError):
            found_bug = "bug" in text.lower() and "no bug" not in text.lower()
        return found_bug, elapsed, text
    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        return False, elapsed, f"ERROR: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# TensorGuard (CEGAR)
# ═══════════════════════════════════════════════════════════════════════════════

def run_tensorguard(tc: Dict[str, Any]) -> Dict[str, Any]:
    """Run full CEGAR pipeline with quality filter."""
    t0 = time.monotonic()
    details = ""
    try:
        result = run_shape_cegar(
            tc["code"],
            input_shapes=tc["input_shapes"],
            max_iterations=10,
            enable_quality_filter=True,
        )
        detected = result.has_real_bugs
        if result.real_bugs:
            details = "; ".join(
                getattr(b, "message", str(b)) for b in result.real_bugs
            )
        elif result.verification_result and not result.verification_result.safe:
            ce = result.verification_result.counterexample
            if ce:
                details = getattr(ce, "pretty", lambda: str(ce))()
    except Exception as e:
        detected = False
        details = f"ERROR: {e}"
    elapsed = (time.monotonic() - t0) * 1000
    return {
        "detected_bug": detected,
        "time_ms": round(elapsed, 2),
        "details": details,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics helpers
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
    avg_time = sum(r["time_ms"] for r in results) / len(results) if results else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "avg_time_ms": round(avg_time, 2),
    }


def bootstrap_ci(
    results: List[Dict[str, Any]],
    n_bootstrap: int = 2000,
    ci: float = 0.95,
) -> Dict[str, Tuple[float, float]]:
    """Bootstrap 95% confidence intervals for precision, recall, F1."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 72)
    print("  Baseline Comparison v2  —  12 benchmarks × 3 tools")
    print("=" * 72)

    # Check for OpenAI API key
    llm_available = False
    client = None
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            llm_available = True
            print("  ✓ OPENAI_API_KEY found — LLM baseline enabled")
        except ImportError:
            print("  ✗ openai package not installed — LLM baseline skipped")
    else:
        print("  ✗ OPENAI_API_KEY not set — LLM baseline skipped")

    tools = ["syntactic", "tensorguard"]
    if llm_available:
        tools.insert(1, "llm")

    all_results: Dict[str, List[Dict[str, Any]]] = {t: [] for t in tools}

    for i, tc in enumerate(TEST_CASES, 1):
        tag = "BUGGY" if tc["has_bug"] else "CLEAN"
        print(f"\n[{i:2d}/{len(TEST_CASES)}] {tc['name']} ({tag}) — {tc['description']}")

        # ── Syntactic baseline ──
        t0 = time.monotonic()
        checker = SyntacticShapeChecker(tc["code"])
        has_bug, bug_msgs = checker.check()
        syn_time = (time.monotonic() - t0) * 1000
        syn_result = {
            "name": tc["name"],
            "ground_truth": tc["has_bug"],
            "detected_bug": has_bug,
            "time_ms": round(syn_time, 2),
            "details": "; ".join(bug_msgs) if bug_msgs else "",
        }
        all_results["syntactic"].append(syn_result)
        syn_mark = "✓" if has_bug == tc["has_bug"] else "✗"
        print(f"  Syntactic:  {syn_mark}  det={has_bug}  {syn_time:.1f}ms"
              + (f"  [{bug_msgs[0][:60]}...]" if bug_msgs else ""))

        # ── LLM baseline ──
        if llm_available:
            llm_found, llm_time, llm_raw = run_llm_check(client, tc["code"])
            llm_result = {
                "name": tc["name"],
                "ground_truth": tc["has_bug"],
                "detected_bug": llm_found,
                "time_ms": round(llm_time, 2),
                "details": llm_raw[:200],
            }
            all_results["llm"].append(llm_result)
            llm_mark = "✓" if llm_found == tc["has_bug"] else "✗"
            print(f"  LLM:        {llm_mark}  det={llm_found}  {llm_time:.1f}ms")

        # ── TensorGuard (CEGAR) ──
        lp = run_tensorguard(tc)
        lp_result = {
            "name": tc["name"],
            "ground_truth": tc["has_bug"],
            "detected_bug": lp["detected_bug"],
            "time_ms": lp["time_ms"],
            "details": lp["details"],
        }
        all_results["tensorguard"].append(lp_result)
        lp_mark = "✓" if lp["detected_bug"] == tc["has_bug"] else "✗"
        print(f"  TensorGuard:   {lp_mark}  det={lp['detected_bug']}  {lp['time_ms']:.1f}ms"
              + (f"  [{lp['details'][:60]}...]" if lp["details"] else ""))

    # ── Per-tool metrics ──
    print(f"\n{'=' * 72}")
    print("  METRICS SUMMARY")
    print(f"{'=' * 72}")

    tool_metrics: Dict[str, Any] = {}
    tool_cis: Dict[str, Any] = {}
    for tool in tools:
        m = compute_metrics(all_results[tool])
        ci = bootstrap_ci(all_results[tool])
        tool_metrics[tool] = m
        tool_cis[tool] = ci
        label = {"syntactic": "Syntactic Pattern", "llm": "GPT-4.1-nano", "tensorguard": "TensorGuard (CEGAR)"}
        print(f"\n  {label.get(tool, tool):25s}  "
              f"F1={m['f1']:<6.4f}  P={m['precision']:<6.4f}  R={m['recall']:<6.4f}  "
              f"TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}  "
              f"avg={m['avg_time_ms']:.1f}ms")
        print(f"  {'':25s}  "
              f"F1 CI={ci['f1']}  P CI={ci['precision']}  R CI={ci['recall']}")

    # ── Comparison deltas ──
    lp_f1 = tool_metrics["tensorguard"]["f1"]
    syn_f1 = tool_metrics["syntactic"]["f1"]
    print(f"\n  TensorGuard vs Syntactic:  ΔF1 = {lp_f1 - syn_f1:+.4f}")
    if llm_available:
        llm_f1 = tool_metrics["llm"]["f1"]
        print(f"  TensorGuard vs LLM:       ΔF1 = {lp_f1 - llm_f1:+.4f}")

    # ── Save JSON ──
    output = {
        "experiment": "baseline_comparison_v2",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_benchmarks": len(TEST_CASES),
        "num_buggy": sum(1 for tc in TEST_CASES if tc["has_bug"]),
        "num_correct": sum(1 for tc in TEST_CASES if not tc["has_bug"]),
        "llm_available": llm_available,
        "tools": {},
    }
    for tool in tools:
        label = {"syntactic": "Syntactic Pattern Matching",
                 "llm": "GPT-4.1-nano (LLM)",
                 "tensorguard": "TensorGuard (CEGAR + quality filter)"}
        output["tools"][tool] = {
            "label": label.get(tool, tool),
            "metrics": tool_metrics[tool],
            "confidence_intervals_95": {k: list(v) for k, v in tool_cis[tool].items()},
            "per_benchmark": all_results[tool],
        }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
