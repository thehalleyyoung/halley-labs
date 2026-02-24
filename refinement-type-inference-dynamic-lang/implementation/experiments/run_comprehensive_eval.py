#!/usr/bin/env python3
"""
Comprehensive evaluation benchmark for TensorGuard.

Runs ~60 code snippets through analyze_unified covering:
  - Matmul dimension mismatches (concrete, 2D/3D)
  - nn.Linear chain mismatches
  - nn.Conv2d chain mismatches
  - Optional[Tensor] null deref patterns
  - Correct code (true negatives)
  - Reshape/view operations
  - torch.cat operations
  - Broadcasting operations
  - Cross-function shape propagation
  - Mixed null+shape (cross-domain) patterns
  - Real-world-style benchmarks (ResNet, Attention, Autoencoder, GAN, DataLoader)

Outputs: experiments/comprehensive_eval_results.json + summary table.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.unified import analyze_unified

EXPERIMENTS_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = EXPERIMENTS_DIR / "comprehensive_eval_results.json"

# ═══════════════════════════════════════════════════════════════════════
# Benchmark definitions
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Benchmark:
    name: str
    code: str
    expect_bug: bool      # True → should find ≥1 bug; False → should find 0
    category: str


BENCHMARKS: List[Benchmark] = []


def _b(name: str, code: str, expect_bug: bool, category: str):
    BENCHMARKS.append(Benchmark(name=name, code=code,
                                expect_bug=expect_bug, category=category))


# ── 1. Matmul dimension mismatches ────────────────────────────────────

_b("matmul_2d_mismatch", """
import torch
def f():
    a = torch.randn(3, 4)
    b = torch.randn(5, 6)
    return a @ b
""", True, "shape_matmul")

_b("matmul_2d_correct", """
import torch
def f():
    a = torch.randn(3, 4)
    b = torch.randn(4, 5)
    return a @ b
""", False, "shape_matmul")

_b("matmul_3d_mismatch", """
import torch
def f():
    a = torch.randn(2, 3, 4)
    b = torch.randn(2, 5, 6)
    return a @ b
""", True, "shape_matmul")

_b("matmul_3d_correct", """
import torch
def f():
    a = torch.randn(2, 3, 4)
    b = torch.randn(2, 4, 6)
    return a @ b
""", False, "shape_matmul")

_b("matmul_vector_mismatch", """
import torch
def f():
    a = torch.randn(3, 4)
    b = torch.randn(5)
    return a @ b
""", True, "shape_matmul")

_b("matmul_vector_correct", """
import torch
def f():
    a = torch.randn(3, 4)
    b = torch.randn(4)
    return a @ b
""", False, "shape_matmul")

_b("matmul_chain_mismatch", """
import torch
def f():
    a = torch.randn(3, 4)
    b = torch.randn(4, 5)
    c = torch.randn(7, 2)
    return (a @ b) @ c
""", True, "shape_matmul")

_b("matmul_chain_correct", """
import torch
def f():
    a = torch.randn(3, 4)
    b = torch.randn(4, 5)
    c = torch.randn(5, 2)
    return (a @ b) @ c
""", False, "shape_matmul")


# ── 2. nn.Linear chain mismatches ─────────────────────────────────────

_b("linear_chain_mismatch", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(30, 5)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
""", True, "shape_linear")

_b("linear_chain_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
""", False, "shape_linear")

_b("linear_three_layer_mismatch", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
""", True, "shape_linear")

_b("linear_three_layer_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
""", False, "shape_linear")

_b("linear_single_layer_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)
""", False, "shape_linear")

_b("linear_wide_mismatch", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(512, 256)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
""", True, "shape_linear")


# ── 3. nn.Conv2d chain mismatches ─────────────────────────────────────

_b("conv_chain_mismatch", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
""", True, "shape_conv")

_b("conv_chain_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
""", False, "shape_conv")

_b("conv_three_layer_mismatch", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
""", True, "shape_conv")

_b("conv_three_layer_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
""", False, "shape_conv")

_b("conv_1x1_mismatch", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 1)
        self.conv2 = nn.Conv2d(128, 64, 1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
""", True, "shape_conv")


# ── 4. Optional[Tensor] null deref ────────────────────────────────────

_b("null_optional_tensor_shape", """
import torch
def f(cond):
    t = None
    if cond:
        t = torch.randn(3, 4)
    return t.shape
""", True, "null")

_b("null_optional_tensor_matmul", """
import torch
def f(cond):
    t = None
    if cond:
        t = torch.randn(3, 4)
    return t @ torch.randn(4, 5)
""", True, "null")

_b("null_guarded_tensor_correct", """
import torch
def f(cond):
    t = None
    if cond:
        t = torch.randn(3, 4)
    if t is not None:
        return t.shape
    return None
""", False, "null")

_b("null_tensor_always_assigned", """
import torch
def f():
    t = torch.randn(3, 4)
    return t.shape
""", False, "null")

_b("null_optional_tensor_size", """
import torch
def f(cond):
    t = None
    if cond:
        t = torch.randn(5, 5)
    s = t.shape
    return s
""", True, "null")

_b("null_double_branch", """
import torch
def f(a, b):
    t = None
    if a:
        t = torch.randn(2, 3)
    if b:
        t = torch.randn(4, 5)
    return t.shape
""", True, "null")


# ── 5. Reshape/view operations ────────────────────────────────────────

_b("reshape_element_mismatch", """
import torch
def f():
    x = torch.randn(3, 4)
    return x.view(5, 5)
""", True, "shape_reshape")

_b("reshape_correct_flatten", """
import torch
def f():
    x = torch.randn(3, 4)
    return x.view(12)
""", False, "shape_reshape")

_b("reshape_correct_2d", """
import torch
def f():
    x = torch.randn(3, 4)
    return x.view(4, 3)
""", False, "shape_reshape")

_b("reshape_element_mismatch_3d", """
import torch
def f():
    x = torch.randn(2, 3, 4)
    return x.view(5, 5)
""", True, "shape_reshape")

_b("reshape_correct_3d", """
import torch
def f():
    x = torch.randn(2, 3, 4)
    return x.view(6, 4)
""", False, "shape_reshape")

_b("reshape_mismatch_large", """
import torch
def f():
    x = torch.randn(8, 8)
    return x.view(10, 7)
""", True, "shape_reshape")


# ── 6. torch.cat operations ──────────────────────────────────────────

_b("cat_dim_mismatch", """
import torch
def f():
    a = torch.randn(3, 4)
    b = torch.randn(3, 5)
    return torch.cat([a, b], dim=0)
""", True, "shape_cat")

_b("cat_correct_dim0", """
import torch
def f():
    a = torch.randn(3, 4)
    b = torch.randn(5, 4)
    return torch.cat([a, b], dim=0)
""", False, "shape_cat")

_b("cat_correct_dim1", """
import torch
def f():
    a = torch.randn(3, 4)
    b = torch.randn(3, 5)
    return torch.cat([a, b], dim=1)
""", False, "shape_cat")

_b("cat_dim1_mismatch", """
import torch
def f():
    a = torch.randn(3, 4)
    b = torch.randn(5, 4)
    return torch.cat([a, b], dim=1)
""", True, "shape_cat")

_b("cat_three_tensors_correct", """
import torch
def f():
    a = torch.randn(2, 4)
    b = torch.randn(3, 4)
    c = torch.randn(5, 4)
    return torch.cat([a, b, c], dim=0)
""", False, "shape_cat")


# ── 7. Broadcasting operations ────────────────────────────────────────

_b("broadcast_mismatch", """
import torch
def f():
    a = torch.randn(3, 4)
    b = torch.randn(5, 4)
    return a + b
""", True, "shape_broadcast")

_b("broadcast_correct_scalar", """
import torch
def f():
    a = torch.randn(3, 4)
    b = torch.randn(1, 4)
    return a + b
""", False, "shape_broadcast")

_b("broadcast_correct_same", """
import torch
def f():
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)
    return a + b
""", False, "shape_broadcast")

_b("broadcast_mismatch_3d", """
import torch
def f():
    a = torch.randn(2, 3, 4)
    b = torch.randn(2, 5, 4)
    return a + b
""", True, "shape_broadcast")

_b("broadcast_correct_expand", """
import torch
def f():
    a = torch.randn(2, 3, 4)
    b = torch.randn(1, 3, 4)
    return a * b
""", False, "shape_broadcast")


# ── 8. Cross-function shape propagation ───────────────────────────────

_b("interproc_matmul_mismatch", """
import torch
def make_tensor():
    return torch.randn(3, 4)

def use_tensor():
    t = make_tensor()
    return t @ torch.randn(5, 6)
""", True, "interprocedural")

_b("interproc_matmul_correct", """
import torch
def make_tensor():
    return torch.randn(3, 4)

def use_tensor():
    t = make_tensor()
    return t @ torch.randn(4, 6)
""", False, "interprocedural")

_b("interproc_two_funcs_mismatch", """
import torch
def get_a():
    return torch.randn(4, 5)

def get_b():
    return torch.randn(7, 3)

def combine():
    return get_a() @ get_b()
""", True, "interprocedural")

_b("interproc_two_funcs_correct", """
import torch
def get_a():
    return torch.randn(4, 5)

def get_b():
    return torch.randn(5, 3)

def combine():
    return get_a() @ get_b()
""", False, "interprocedural")


# ── 9. Cross-domain (null + shape) ────────────────────────────────────

_b("cross_null_matmul", """
import torch
def f(cond):
    t = None
    if cond:
        t = torch.randn(3, 4)
    result = t @ torch.randn(4, 5)
    return result
""", True, "cross_domain")

_b("cross_null_matmul_guarded", """
import torch
def f(cond):
    t = None
    if cond:
        t = torch.randn(3, 4)
    if t is not None:
        result = t @ torch.randn(4, 5)
        return result
    return None
""", False, "cross_domain")

_b("cross_null_shape_access", """
import torch
def f(cond):
    t = None
    if cond:
        t = torch.randn(3, 4)
    return t.shape
""", True, "cross_domain")

_b("cross_null_shape_guarded", """
import torch
def f(cond):
    t = None
    if cond:
        t = torch.randn(3, 4)
    if t is not None:
        return t.shape
    return ()
""", False, "cross_domain")


# ══════════════════════════════════════════════════════════════════════
# Real-world-style benchmarks
# ══════════════════════════════════════════════════════════════════════

_b("resnet_skip_conv_mismatch", """
import torch
import torch.nn as nn
class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 128, 3)
        self.conv2 = nn.Conv2d(256, 128, 3)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out
""", True, "shape_conv")

_b("resnet_skip_conv_correct", """
import torch
import torch.nn as nn
class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 128, 3)
        self.conv2 = nn.Conv2d(128, 128, 3)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out
""", False, "shape_conv")

_b("attention_linear_mismatch", """
import torch
import torch.nn as nn
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(512, 64)
        self.out_proj = nn.Linear(128, 512)
    def forward(self, x):
        q = self.q_proj(x)
        return self.out_proj(q)
""", True, "shape_linear")

_b("attention_linear_correct", """
import torch
import torch.nn as nn
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(512, 64)
        self.out_proj = nn.Linear(64, 512)
    def forward(self, x):
        q = self.q_proj(x)
        return self.out_proj(q)
""", False, "shape_linear")

_b("autoencoder_mismatch", """
import torch
import torch.nn as nn
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Linear(784, 256)
        self.enc2 = nn.Linear(256, 64)
        self.dec1 = nn.Linear(64, 128)
        self.dec2 = nn.Linear(256, 784)
    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.dec1(x)
        x = self.dec2(x)
        return x
""", True, "shape_linear")

_b("autoencoder_correct", """
import torch
import torch.nn as nn
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Linear(784, 256)
        self.enc2 = nn.Linear(256, 64)
        self.dec1 = nn.Linear(64, 256)
        self.dec2 = nn.Linear(256, 784)
    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.dec1(x)
        x = self.dec2(x)
        return x
""", False, "shape_linear")

_b("gan_generator_mismatch", """
import torch
import torch.nn as nn
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 784)
    def forward(self, z):
        x = self.fc1(z)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
""", True, "shape_linear")

_b("gan_generator_correct", """
import torch
import torch.nn as nn
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, 784)
    def forward(self, z):
        x = self.fc1(z)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
""", False, "shape_linear")

_b("dataloader_linear_mismatch", """
import torch
import torch.nn as nn
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)
""", True, "shape_linear")

_b("dataloader_linear_correct", """
import torch
import torch.nn as nn
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)
""", False, "shape_linear")

_b("transformer_ffn_mismatch", """
import torch
import torch.nn as nn
class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 2048)
        self.fc2 = nn.Linear(1024, 512)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
""", True, "shape_linear")

_b("transformer_ffn_correct", """
import torch
import torch.nn as nn
class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 2048)
        self.fc2 = nn.Linear(2048, 512)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
""", False, "shape_linear")

_b("unet_conv_mismatch", """
import torch
import torch.nn as nn
class UNetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Conv2d(3, 64, 3)
        self.down2 = nn.Conv2d(128, 256, 3)
    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        return x
""", True, "shape_conv")

_b("unet_conv_correct", """
import torch
import torch.nn as nn
class UNetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Conv2d(3, 64, 3)
        self.down2 = nn.Conv2d(64, 128, 3)
    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        return x
""", False, "shape_conv")


# ═══════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    name: str
    category: str
    expect_bug: bool
    found_bugs: int
    bug_kinds: List[str]
    bug_messages: List[str]
    is_tp: bool
    is_fp: bool
    is_fn: bool
    is_tn: bool
    time_ms: float


def run_benchmark(bm: Benchmark) -> BenchmarkResult:
    t0 = time.perf_counter()
    try:
        result = analyze_unified(bm.code)
        bugs = result.bugs
    except Exception as e:
        bugs = []
    elapsed = (time.perf_counter() - t0) * 1000

    found = len(bugs)
    detected = found > 0

    tp = bm.expect_bug and detected
    fp = (not bm.expect_bug) and detected
    fn = bm.expect_bug and (not detected)
    tn = (not bm.expect_bug) and (not detected)

    return BenchmarkResult(
        name=bm.name,
        category=bm.category,
        expect_bug=bm.expect_bug,
        found_bugs=found,
        bug_kinds=[b.kind for b in bugs],
        bug_messages=[b.message for b in bugs],
        is_tp=tp, is_fp=fp, is_fn=fn, is_tn=tn,
        time_ms=elapsed,
    )


@dataclass
class CategoryMetrics:
    category: str
    total: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float


def compute_metrics(results: List[BenchmarkResult],
                    category: Optional[str] = None) -> CategoryMetrics:
    if category:
        subset = [r for r in results if r.category == category]
    else:
        subset = results
    tp = sum(r.is_tp for r in subset)
    fp = sum(r.is_fp for r in subset)
    fn = sum(r.is_fn for r in subset)
    tn = sum(r.is_tn for r in subset)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return CategoryMetrics(
        category=category or "overall",
        total=len(subset), tp=tp, fp=fp, fn=fn, tn=tn,
        precision=prec, recall=rec, f1=f1,
    )


def print_table(results: List[BenchmarkResult]):
    categories = sorted(set(r.category for r in results))

    hdr = f"{'Category':<22} {'N':>3} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3} {'Prec':>6} {'Rec':>6} {'F1':>6}"
    print("\n" + "=" * len(hdr))
    print("TensorGuard Comprehensive Evaluation")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    for cat in categories:
        m = compute_metrics(results, cat)
        print(f"{m.category:<22} {m.total:>3} {m.tp:>3} {m.fp:>3} {m.fn:>3} {m.tn:>3} "
              f"{m.precision:>6.2f} {m.recall:>6.2f} {m.f1:>6.2f}")

    print("-" * len(hdr))
    overall = compute_metrics(results)
    print(f"{'OVERALL':<22} {overall.total:>3} {overall.tp:>3} {overall.fp:>3} "
          f"{overall.fn:>3} {overall.tn:>3} {overall.precision:>6.2f} "
          f"{overall.recall:>6.2f} {overall.f1:>6.2f}")
    print("=" * len(hdr))

    total_time = sum(r.time_ms for r in results)
    print(f"\nTotal benchmarks: {len(results)}")
    print(f"Total time: {total_time:.1f} ms  ({total_time/len(results):.1f} ms/benchmark)")

    # Print failures
    fns = [r for r in results if r.is_fn]
    fps = [r for r in results if r.is_fp]
    if fns:
        print(f"\n── False Negatives ({len(fns)}) ──")
        for r in fns:
            print(f"  ✗ {r.name} [{r.category}]")
    if fps:
        print(f"\n── False Positives ({len(fps)}) ──")
        for r in fps:
            print(f"  ✗ {r.name} [{r.category}]: {r.bug_kinds}")


def save_results(results: List[BenchmarkResult]):
    categories = sorted(set(r.category for r in results))
    per_cat = {}
    for cat in categories:
        m = compute_metrics(results, cat)
        per_cat[cat] = asdict(m)
    overall = compute_metrics(results)

    data = {
        "total_benchmarks": len(results),
        "total_time_ms": sum(r.time_ms for r in results),
        "overall": asdict(overall),
        "per_category": per_cat,
        "results": [asdict(r) for r in results],
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


# ═══════════════════════════════════════════════════════════════════════

def main():
    print(f"Running {len(BENCHMARKS)} benchmarks ...\n")
    results: List[BenchmarkResult] = []
    for i, bm in enumerate(BENCHMARKS, 1):
        r = run_benchmark(bm)
        status = "✓" if (r.is_tp or r.is_tn) else "✗"
        label = "BUG" if bm.expect_bug else "OK "
        found_label = f"{r.found_bugs} bugs" if r.found_bugs else "clean"
        print(f"  [{i:>2}/{len(BENCHMARKS)}] {status} {bm.name:<38} expect={label}  got={found_label:<12} {r.time_ms:.1f}ms")
        results.append(r)

    print_table(results)
    save_results(results)


if __name__ == "__main__":
    main()
