#!/usr/bin/env python3
"""
LLM Baseline Comparison for TensorGuard.

Compares TensorGuard's analyze_unified against OpenAI gpt-4.1-nano on 20
representative benchmarks (10 buggy, 10 clean) drawn from the comprehensive
eval suite.  Outputs precision / recall / F1 for both tools plus a
side-by-side comparison table.

Usage (from implementation/):
    python experiments/run_llm_baseline.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.unified import analyze_unified

EXPERIMENTS_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = EXPERIMENTS_DIR / "llm_baseline_results.json"

# ═══════════════════════════════════════════════════════════════════════
# 20 representative benchmarks (10 buggy / 10 clean)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Benchmark:
    name: str
    code: str
    expect_bug: bool
    category: str


BENCHMARKS: List[Benchmark] = [
    # ── Buggy (expect_bug=True) ──────────────────────────────────────
    Benchmark("matmul_2d_mismatch", """
import torch
def f():
    a = torch.randn(3, 4)
    b = torch.randn(5, 6)
    return a @ b
""", True, "shape_matmul"),

    Benchmark("linear_chain_mismatch", """
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
""", True, "shape_linear"),

    Benchmark("conv_chain_mismatch", """
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
""", True, "shape_conv"),

    Benchmark("null_optional_tensor_shape", """
import torch
def f(cond):
    t = None
    if cond:
        t = torch.randn(3, 4)
    return t.shape
""", True, "null"),

    Benchmark("reshape_element_mismatch", """
import torch
def f():
    x = torch.randn(3, 4)
    return x.view(5, 5)
""", True, "shape_reshape"),

    Benchmark("cat_dim_mismatch", """
import torch
def f():
    a = torch.randn(3, 4)
    b = torch.randn(3, 5)
    return torch.cat([a, b], dim=0)
""", True, "shape_cat"),

    Benchmark("broadcast_mismatch", """
import torch
def f():
    a = torch.randn(3, 4)
    b = torch.randn(5, 4)
    return a + b
""", True, "shape_broadcast"),

    Benchmark("interproc_matmul_mismatch", """
import torch
def make_tensor():
    return torch.randn(3, 4)

def use_tensor():
    t = make_tensor()
    return t @ torch.randn(5, 6)
""", True, "interprocedural"),

    Benchmark("autoencoder_mismatch", """
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
""", True, "shape_linear"),

    Benchmark("cross_null_matmul", """
import torch
def f(cond):
    t = None
    if cond:
        t = torch.randn(3, 4)
    result = t @ torch.randn(4, 5)
    return result
""", True, "cross_domain"),

    # ── Clean (expect_bug=False) ─────────────────────────────────────
    Benchmark("matmul_2d_correct", """
import torch
def f():
    a = torch.randn(3, 4)
    b = torch.randn(4, 5)
    return a @ b
""", False, "shape_matmul"),

    Benchmark("linear_chain_correct", """
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
""", False, "shape_linear"),

    Benchmark("conv_chain_correct", """
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
""", False, "shape_conv"),

    Benchmark("null_guarded_tensor_correct", """
import torch
def f(cond):
    t = None
    if cond:
        t = torch.randn(3, 4)
    if t is not None:
        return t.shape
    return None
""", False, "null"),

    Benchmark("reshape_correct_flatten", """
import torch
def f():
    x = torch.randn(3, 4)
    return x.view(12)
""", False, "shape_reshape"),

    Benchmark("cat_correct_dim0", """
import torch
def f():
    a = torch.randn(3, 4)
    b = torch.randn(5, 4)
    return torch.cat([a, b], dim=0)
""", False, "shape_cat"),

    Benchmark("broadcast_correct_scalar", """
import torch
def f():
    a = torch.randn(3, 4)
    b = torch.randn(1, 4)
    return a + b
""", False, "shape_broadcast"),

    Benchmark("interproc_matmul_correct", """
import torch
def make_tensor():
    return torch.randn(3, 4)

def use_tensor():
    t = make_tensor()
    return t @ torch.randn(4, 6)
""", False, "interprocedural"),

    Benchmark("autoencoder_correct", """
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
""", False, "shape_linear"),

    Benchmark("cross_null_matmul_guarded", """
import torch
def f(cond):
    t = None
    if cond:
        t = torch.randn(3, 4)
    if t is not None:
        result = t @ torch.randn(4, 5)
        return result
    return None
""", False, "cross_domain"),
]

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

LLM_PROMPT = (
    "Analyze this Python code for tensor shape errors, null reference errors "
    "on Optional[Tensor], and dimension mismatches in nn.Module layers. "
    "Reply with ONLY 'BUG: <description>' if you find a bug, or 'CLEAN' "
    "if the code is correct."
)


def run_tensorguard(code: str) -> bool:
    """Return True if TensorGuard finds at least one bug."""
    try:
        result = analyze_unified(code)
        return len(result.bugs) > 0
    except Exception as e:
        print(f"  [TensorGuard error] {e}")
        return False


def run_llm(client, code: str) -> tuple[bool, str]:
    """Query gpt-4.1-nano. Returns (found_bug, raw_response)."""
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
        text = resp.choices[0].message.content.strip()
        found_bug = text.upper().startswith("BUG")
        return found_bug, text
    except Exception as e:
        return False, f"ERROR: {e}"


def metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Run `source ~/.bashrc` first.")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    print("=" * 72)
    print("TensorGuard vs GPT-4.1-nano  —  20 benchmark comparison")
    print("=" * 72)

    lp_tp = lp_fp = lp_fn = lp_tn = 0
    llm_tp = llm_fp = llm_fn = llm_tn = 0
    rows = []

    for i, bm in enumerate(BENCHMARKS, 1):
        print(f"\n[{i:2d}/{len(BENCHMARKS)}] {bm.name}  (expect_bug={bm.expect_bug})")

        # TensorGuard
        t0 = time.time()
        lp_found = run_tensorguard(bm.code)
        lp_ms = round((time.time() - t0) * 1000, 1)

        # LLM
        t0 = time.time()
        llm_found, llm_raw = run_llm(client, bm.code)
        llm_ms = round((time.time() - t0) * 1000, 1)

        # Classify
        if bm.expect_bug:
            if lp_found:  lp_tp += 1
            else:         lp_fn += 1
            if llm_found: llm_tp += 1
            else:         llm_fn += 1
        else:
            if lp_found:  lp_fp += 1
            else:         lp_tn += 1
            if llm_found: llm_fp += 1
            else:         llm_tn += 1

        lp_label = ("TP" if lp_found else "FN") if bm.expect_bug else ("FP" if lp_found else "TN")
        llm_label = ("TP" if llm_found else "FN") if bm.expect_bug else ("FP" if llm_found else "TN")

        print(f"  TensorGuard: {lp_label} ({lp_ms}ms)  |  LLM: {llm_label} ({llm_ms}ms)")
        if llm_raw.startswith("ERROR"):
            print(f"  LLM response: {llm_raw}")

        rows.append({
            "name": bm.name,
            "category": bm.category,
            "expect_bug": bm.expect_bug,
            "tensorguard_found_bug": lp_found,
            "tensorguard_label": lp_label,
            "tensorguard_ms": lp_ms,
            "llm_found_bug": llm_found,
            "llm_label": llm_label,
            "llm_ms": llm_ms,
            "llm_response": llm_raw,
        })

    lp_m = metrics(lp_tp, lp_fp, lp_fn, lp_tn)
    llm_m = metrics(llm_tp, llm_fp, llm_fn, llm_tn)

    # ── Summary table ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)

    hdr = f"{'Metric':<14} {'TensorGuard':>10} {'GPT-4.1-nano':>14}"
    print(hdr)
    print("-" * len(hdr))
    for key in ["TP", "FP", "FN", "TN", "precision", "recall", "f1", "accuracy"]:
        lv = lp_m[key]
        rv = llm_m[key]
        if isinstance(lv, float):
            print(f"{key:<14} {lv:>10.4f} {rv:>14.4f}")
        else:
            print(f"{key:<14} {lv:>10} {rv:>14}")

    # ── Per-benchmark table ───────────────────────────────────────────
    print("\n" + "-" * 72)
    print(f"{'Benchmark':<35} {'Expect':>6} {'TensorGuard':>9} {'LLM':>9}")
    print("-" * 72)
    for r in rows:
        expect = "BUG" if r["expect_bug"] else "CLEAN"
        print(f"{r['name']:<35} {expect:>6} {r['tensorguard_label']:>9} {r['llm_label']:>9}")

    # ── Save results ──────────────────────────────────────────────────
    output = {
        "model": "gpt-4.1-nano",
        "num_benchmarks": len(BENCHMARKS),
        "tensorguard_metrics": lp_m,
        "llm_metrics": llm_m,
        "benchmarks": rows,
    }
    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
