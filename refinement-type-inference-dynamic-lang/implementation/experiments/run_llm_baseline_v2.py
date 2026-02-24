#!/usr/bin/env python3
"""
LLM Baseline v2: TensorGuard verify_model vs GPT-4.1-nano on nn.Module benchmarks.

Compares TensorGuard's shape verification (via verify_model) against an LLM
baseline for detecting shape mismatch bugs in 10 nn.Module benchmarks
(5 buggy, 5 correct).

Usage (from implementation/):
    source ~/.bashrc && python experiments/run_llm_baseline_v2.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_checker import verify_model

EXPERIMENTS_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = EXPERIMENTS_DIR / "llm_baseline_v2_results.json"

# ═══════════════════════════════════════════════════════════════════════════
# 10 nn.Module benchmarks (5 buggy / 5 correct)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Benchmark:
    name: str
    code: str
    expect_bug: bool
    description: str
    input_shapes: dict  # shapes passed to verify_model


BENCHMARKS: List[Benchmark] = [
    # ── Buggy (expect_bug=True) ───────────────────────────────────────────
    Benchmark(
        "BuggyMLP",
        """\
import torch
import torch.nn as nn

class BuggyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(128, 10)  # BUG: expects 128 but fc1 outputs 256

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
""",
        True,
        "Linear chain mismatch: fc1 outputs 256 but fc2 expects 128",
        {"x": ("batch", 768)},
    ),
    Benchmark(
        "BroadcastBug",
        """\
import torch
import torch.nn as nn

class BroadcastBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_a = nn.Linear(512, 64)
        self.proj_b = nn.Linear(512, 128)  # BUG: 64 + 128 broadcast fails

    def forward(self, x):
        a = self.proj_a(x)
        b = self.proj_b(x)
        return a + b  # shape mismatch: (*, 64) + (*, 128)
""",
        True,
        "Broadcast mismatch: proj_a outputs 64, proj_b outputs 128, a+b fails",
        {"x": ("batch", 512)},
    ),
    Benchmark(
        "BuggyAutoencoder",
        """\
import torch
import torch.nn as nn

class BuggyAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Linear(784, 256)
        self.enc2 = nn.Linear(256, 64)
        self.dec1 = nn.Linear(64, 128)
        self.dec2 = nn.Linear(256, 784)  # BUG: dec1 outputs 128, not 256

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.dec1(x)
        x = self.dec2(x)
        return x
""",
        True,
        "Decoder chain mismatch: dec1 outputs 128 but dec2 expects 256",
        {"x": ("batch", 784)},
    ),
    Benchmark(
        "BuggyConvNet",
        """\
import torch
import torch.nn as nn

class BuggyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)  # BUG: conv1 outputs 16 channels, not 32

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
""",
        True,
        "Conv channel mismatch: conv1 outputs 16 channels but conv2 expects 32",
        {"x": ("batch", 3, "H", "W")},
    ),
    Benchmark(
        "BuggyResidual",
        """\
import torch
import torch.nn as nn

class BuggyResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 128)  # BUG: residual add with 512-dim input

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = x + residual
        x = self.fc3(x)
        out = x + residual  # shape mismatch: (*, 128) + (*, 512)
        return out
""",
        True,
        "Residual mismatch: fc3 outputs 128 but residual is 512",
        {"x": ("batch", 512)},
    ),

    # ── Correct (expect_bug=False) ────────────────────────────────────────
    Benchmark(
        "CorrectMLP",
        """\
import torch
import torch.nn as nn

class CorrectMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
""",
        False,
        "Correct linear chain: 768->256->10",
        {"x": ("batch", 768)},
    ),
    Benchmark(
        "CorrectTransformerFFN",
        """\
import torch
import torch.nn as nn

class CorrectTransformerFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 2048)
        self.fc2 = nn.Linear(2048, 512)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
""",
        False,
        "Correct FFN block: 512->2048->512",
        {"x": ("batch", 512)},
    ),
    Benchmark(
        "CorrectAutoencoder",
        """\
import torch
import torch.nn as nn

class CorrectAutoencoder(nn.Module):
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
""",
        False,
        "Correct autoencoder: 784->256->64->256->784",
        {"x": ("batch", 784)},
    ),
    Benchmark(
        "CorrectConvNet",
        """\
import torch
import torch.nn as nn

class CorrectConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
""",
        False,
        "Correct conv chain: 3->16->32 channels",
        {"x": ("batch", 3, "H", "W")},
    ),
    Benchmark(
        "CorrectResidual",
        """\
import torch
import torch.nn as nn

class CorrectResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 512)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = x + residual  # shapes match: both (*, 512)
        return x
""",
        False,
        "Correct residual: fc2 output matches input dim for skip connection",
        {"x": ("batch", 512)},
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

LLM_PROMPT = (
    "Does this PyTorch nn.Module have a shape mismatch bug? "
    "Answer YES or NO, then explain."
)


def run_tensorguard(code: str, input_shapes: dict) -> tuple[bool, str]:
    """Return (found_bug, detail) using verify_model."""
    try:
        result = verify_model(code, input_shapes=input_shapes)
        if not result.safe:
            detail = "; ".join(result.errors) if result.errors else "unsafe"
            if result.counterexample:
                detail = result.counterexample.pretty()
            return True, detail
        return False, "safe"
    except Exception as e:
        return False, f"ERROR: {e}"


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
            max_tokens=512,
        )
        text = resp.choices[0].message.content.strip()
        first_word = text.split()[0].upper().rstrip(".,;:!") if text else ""
        found_bug = first_word == "YES"
        return found_bug, text
    except Exception as e:
        return False, f"ERROR: {e}"


def compute_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
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


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Run `source ~/.bashrc` first.")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    print("=" * 72)
    print("TensorGuard (verify_model) vs GPT-4.1-nano  —  10 nn.Module benchmarks")
    print("=" * 72)

    lp_tp = lp_fp = lp_fn = lp_tn = 0
    llm_tp = llm_fp = llm_fn = llm_tn = 0
    rows = []

    for i, bm in enumerate(BENCHMARKS, 1):
        label = "BUG" if bm.expect_bug else "CORRECT"
        print(f"\n[{i:2d}/{len(BENCHMARKS)}] {bm.name}  ({label})")
        print(f"  Description: {bm.description}")

        # TensorGuard via verify_model
        t0 = time.time()
        lp_found, lp_detail = run_tensorguard(bm.code, bm.input_shapes)
        lp_ms = round((time.time() - t0) * 1000, 1)

        # LLM via gpt-4.1-nano
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

        rows.append({
            "name": bm.name,
            "description": bm.description,
            "expect_bug": bm.expect_bug,
            "tensorguard_found_bug": lp_found,
            "tensorguard_label": lp_label,
            "tensorguard_ms": lp_ms,
            "tensorguard_detail": lp_detail[:500],
            "llm_found_bug": llm_found,
            "llm_label": llm_label,
            "llm_ms": llm_ms,
            "llm_response": llm_raw[:500],
        })

    lp_m = compute_metrics(lp_tp, lp_fp, lp_fn, lp_tn)
    llm_m = compute_metrics(llm_tp, llm_fp, llm_fn, llm_tn)

    # ── Summary table ─────────────────────────────────────────────────────
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

    # ── Per-benchmark table ───────────────────────────────────────────────
    print("\n" + "-" * 72)
    print(f"{'Benchmark':<25} {'Expect':>8} {'TensorGuard':>10} {'LLM':>10}")
    print("-" * 72)
    for r in rows:
        expect = "BUG" if r["expect_bug"] else "CORRECT"
        print(f"{r['name']:<25} {expect:>8} {r['tensorguard_label']:>10} {r['llm_label']:>10}")

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        "experiment": "llm_baseline_v2",
        "model": "gpt-4.1-nano",
        "num_benchmarks": len(BENCHMARKS),
        "num_buggy": sum(1 for b in BENCHMARKS if b.expect_bug),
        "num_correct": sum(1 for b in BENCHMARKS if not b.expect_bug),
        "tensorguard_metrics": lp_m,
        "llm_metrics": llm_m,
        "summary": {
            "tensorguard_f1": lp_m["f1"],
            "llm_f1": llm_m["f1"],
            "tensorguard_precision": lp_m["precision"],
            "llm_precision": llm_m["precision"],
            "tensorguard_recall": lp_m["recall"],
            "llm_recall": llm_m["recall"],
        },
        "benchmarks": rows,
    }
    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
