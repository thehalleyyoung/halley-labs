"""
Contract Discovery Ablation Experiment v2 — Comprehensive Architecture Coverage.

Compares three verification modes across 10 nn.Module architectures:
  (a) No contract discovery — single-pass ConstraintVerifier.verify()
  (b) Contract discovery without quality filter — run_shape_cegar(enable_quality_filter=False)
  (c) Contract discovery with quality filter   — run_shape_cegar(enable_quality_filter=True)

Outputs: experiments/cegar_ablation_v2_results.json
"""

import json
import os
import sys
import time
import random
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_checker import extract_computation_graph, BoundedModelChecker, ConstraintVerifier, Device, Phase
from src.shape_cegar import run_shape_cegar, ShapeCEGARLoop, PREDICATE_QUALITY_THRESHOLD

RESULTS_FILE = Path(__file__).parent / "cegar_ablation_v2_results.json"

# ═══════════════════════════════════════════════════════════════════════════════
# Test cases — 10 architectures, mix of buggy and correct
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
    # ── 7. U-Net encoder with bug: channel mismatch between blocks ──
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
]


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics helpers
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute precision, recall, F1, and confusion counts."""
    tp = fp = fn = tn = 0
    for r in results:
        has_bug = r["has_bug"]
        detected = r["detected_bug"]
        if has_bug and detected:
            tp += 1
        elif not has_bug and detected:
            fp += 1
        elif has_bug and not detected:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def bootstrap_ci(
    results: List[Dict[str, Any]],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> Dict[str, Tuple[float, float]]:
    """Bootstrap 95 % confidence intervals for key metrics."""
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
        lo = vals[int(alpha * len(vals))]
        hi = vals[int((1 - alpha) * len(vals))]
        cis[key] = (round(lo, 4), round(hi, 4))
    return cis


# ═══════════════════════════════════════════════════════════════════════════════
# Mode runners
# ═══════════════════════════════════════════════════════════════════════════════

def run_no_cegar(tc: Dict[str, Any]) -> Dict[str, Any]:
    """Mode (a): single-pass ConstraintVerifier — no contract discovery loop."""
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
        "arch": tc["arch"],
        "has_bug": tc["has_bug"],
        "detected_bug": detected,
        "status": status,
        "iterations": 1,
        "predicates_discovered": 0,
        "predicates_rejected": 0,
        "time_ms": round(elapsed, 2),
    }


def run_cegar_mode(tc: Dict[str, Any], enable_quality_filter: bool) -> Dict[str, Any]:
    """Mode (b)/(c): CEGAR loop with quality filter on or off."""
    t0 = time.monotonic()
    try:
        cegar_result = run_shape_cegar(
            tc["code"],
            input_shapes=tc["input_shapes"],
            max_iterations=10,
            enable_quality_filter=enable_quality_filter,
        )
        detected = cegar_result.has_real_bugs
        num_predicates = len(cegar_result.discovered_predicates)
        num_iterations = cegar_result.iterations
        status = cegar_result.final_status.name
        qr = cegar_result.predicate_quality_report
        n_rejected = qr.get("rejected", 0) if qr else 0
    except Exception as e:
        detected = False
        num_predicates = 0
        num_iterations = 0
        status = f"ERROR: {e}"
        n_rejected = 0
    elapsed = (time.monotonic() - t0) * 1000
    return {
        "name": tc["name"],
        "arch": tc["arch"],
        "has_bug": tc["has_bug"],
        "detected_bug": detected,
        "status": status,
        "iterations": num_iterations,
        "predicates_discovered": num_predicates,
        "predicates_rejected": n_rejected,
        "time_ms": round(elapsed, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 72)
    print("  CEGAR Ablation Experiment v2  —  10 architectures × 3 modes")
    print("=" * 72)

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
        for tc in TEST_CASES:
            try:
                if mode_key == "no_cegar":
                    r = run_no_cegar(tc)
                elif mode_key == "cegar_no_filter":
                    r = run_cegar_mode(tc, enable_quality_filter=False)
                else:
                    r = run_cegar_mode(tc, enable_quality_filter=True)
            except Exception as exc:
                r = {
                    "name": tc["name"],
                    "arch": tc["arch"],
                    "has_bug": tc["has_bug"],
                    "detected_bug": False,
                    "status": f"ERROR: {exc}",
                    "iterations": 0,
                    "predicates_discovered": 0,
                    "predicates_rejected": 0,
                    "time_ms": 0.0,
                }
            mark = "✓" if r["detected_bug"] == r["has_bug"] else "✗"
            print(f"  {mark} {r['name']:30s}  bug={r['has_bug']}  det={r['detected_bug']}  "
                  f"iters={r['iterations']}  preds={r['predicates_discovered']}  "
                  f"rej={r['predicates_rejected']}  {r['time_ms']:.0f}ms")
            per_bench.append(r)

        metrics = compute_metrics(per_bench)
        cis = bootstrap_ci(per_bench)
        total_preds = sum(r["predicates_discovered"] for r in per_bench)
        total_rej = sum(r["predicates_rejected"] for r in per_bench)
        total_iters = sum(r["iterations"] for r in per_bench)
        total_time = sum(r["time_ms"] for r in per_bench)

        print(f"\n  F1={metrics['f1']}  P={metrics['precision']}  R={metrics['recall']}  "
              f"TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  TN={metrics['tn']}")
        print(f"  predicates={total_preds}  rejected={total_rej}  "
              f"iterations={total_iters}  time={total_time:.0f}ms")

        all_configs[mode_key] = {
            "label": mode_label,
            "metrics": metrics,
            "confidence_intervals_95": cis,
            "total_predicates": total_preds,
            "total_rejected": total_rej,
            "total_iterations": total_iters,
            "total_time_ms": round(total_time, 2),
            "per_benchmark": per_bench,
        }

    # ── Summary comparison ──────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  SUMMARY")
    print(f"{'=' * 72}")
    for mode_key, mode_label in modes:
        m = all_configs[mode_key]["metrics"]
        print(f"  {mode_label:40s}  F1={m['f1']:<6}  P={m['precision']:<6}  R={m['recall']}")

    no_f1 = all_configs["no_cegar"]["metrics"]["f1"]
    unf_f1 = all_configs["cegar_no_filter"]["metrics"]["f1"]
    filt_f1 = all_configs["cegar_with_filter"]["metrics"]["f1"]

    print(f"\n  Quality-filtered CEGAR vs no-CEGAR:     ΔF1 = {filt_f1 - no_f1:+.4f}")
    print(f"  Quality-filtered CEGAR vs unfiltered:    ΔF1 = {filt_f1 - unf_f1:+.4f}")
    print(f"  Predicates rejected by quality filter:   "
          f"{all_configs['cegar_with_filter']['total_rejected']}")

    # ── Write JSON ──────────────────────────────────────────────────────
    output = {
        "experiment": "cegar_ablation_v2",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_test_cases": len(TEST_CASES),
        "predicate_quality_threshold": PREDICATE_QUALITY_THRESHOLD,
        "architectures": sorted(set(tc["arch"] for tc in TEST_CASES)),
        "configs": all_configs,
        "summary": {
            "no_cegar_f1": no_f1,
            "cegar_no_filter_f1": unf_f1,
            "cegar_with_filter_f1": filt_f1,
            "filter_vs_no_cegar_delta_f1": round(filt_f1 - no_f1, 4),
            "filter_vs_unfiltered_delta_f1": round(filt_f1 - unf_f1, 4),
            "total_rejected_predicates": all_configs["cegar_with_filter"]["total_rejected"],
        },
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
