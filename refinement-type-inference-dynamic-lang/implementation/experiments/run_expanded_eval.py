"""
Runner for Expanded Benchmark Suite.

Runs all 200+ benchmarks through:
  1. TensorGuard (verify_model with Z3 theories + CEGAR)
  2. Syntactic baseline (AST-based consecutive Linear/Conv dimension check)

Computes P/R/F1 with Wilson score confidence intervals.
Saves results to experiments/expanded_eval_results.json.
"""

from __future__ import annotations

import ast
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.expanded_benchmark_suite import EXPANDED_BENCHMARKS
from src.model_checker import verify_model
from src.shape_cegar import run_shape_cegar

RESULTS_FILE = Path(__file__).parent / "expanded_eval_results.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Syntactic Baseline (no Z3)
# ═══════════════════════════════════════════════════════════════════════════════

class SyntacticBaseline:
    """Pure AST-based checker: traces consecutive Linear/Conv2d feature
    dimensions and flags mismatches.  No Z3, no symbolic reasoning."""

    def __init__(self, source: str):
        self.source = source
        self.tree = ast.parse(source)
        self.layers: Dict[str, Dict[str, Any]] = {}
        self.bugs: List[str] = []

    def check(self) -> Tuple[bool, List[str]]:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef) and self._is_nn_module(node):
                self._extract_layers(node)
                self._trace_forward(node)
        return len(self.bugs) > 0, self.bugs

    # -- helpers --

    @staticmethod
    def _is_nn_module(cls: ast.ClassDef) -> bool:
        for base in cls.bases:
            if isinstance(base, ast.Attribute) and base.attr == "Module":
                return True
            if isinstance(base, ast.Name) and base.id == "Module":
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

    def _parse_layer_call(self, name: str, call: ast.Call):
        func = call.func
        ltype = func.attr if isinstance(func, ast.Attribute) else (
            func.id if isinstance(func, ast.Name) else None)
        if ltype == "Linear" and len(call.args) >= 2:
            inf = self._const(call.args[0])
            outf = self._const(call.args[1])
            if inf is not None and outf is not None:
                self.layers[name] = {"type": "Linear",
                                     "in_features": inf, "out_features": outf}
        elif ltype == "Conv2d" and len(call.args) >= 3:
            inc = self._const(call.args[0])
            outc = self._const(call.args[1])
            if inc is not None and outc is not None:
                self.layers[name] = {"type": "Conv2d",
                                     "in_channels": inc, "out_channels": outc}
        elif ltype == "BatchNorm2d" and len(call.args) >= 1:
            nf = self._const(call.args[0])
            if nf is not None:
                self.layers[name] = {"type": "BatchNorm2d", "num_features": nf}
        elif ltype == "BatchNorm1d" and len(call.args) >= 1:
            nf = self._const(call.args[0])
            if nf is not None:
                self.layers[name] = {"type": "BatchNorm1d", "num_features": nf}
        elif ltype == "LayerNorm" and len(call.args) >= 1:
            nf = self._const(call.args[0])
            if nf is not None:
                self.layers[name] = {"type": "LayerNorm", "normalized_shape": nf}
        elif ltype == "GroupNorm" and len(call.args) >= 2:
            ng = self._const(call.args[0])
            nc = self._const(call.args[1])
            if ng is not None and nc is not None:
                self.layers[name] = {"type": "GroupNorm",
                                     "num_groups": ng, "num_channels": nc}
        elif ltype == "InstanceNorm2d" and len(call.args) >= 1:
            nf = self._const(call.args[0])
            if nf is not None:
                self.layers[name] = {"type": "InstanceNorm2d", "num_features": nf}

    @staticmethod
    def _const(node: ast.expr) -> Optional[int]:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        return None

    def _trace_forward(self, cls: ast.ClassDef):
        fwd = None
        for item in cls.body:
            if isinstance(item, ast.FunctionDef) and item.name == "forward":
                fwd = item
                break
        if fwd is None:
            return

        var_shapes: Dict[str, Optional[int]] = {}

        for stmt in ast.walk(fwd):
            info = self._extract_call(stmt)
            if info is None:
                continue
            layer_name, input_var, output_var = info
            if layer_name not in self.layers:
                if input_var and input_var in var_shapes and output_var:
                    var_shapes[output_var] = var_shapes[input_var]
                continue

            layer = self.layers[layer_name]
            lt = layer["type"]

            if lt == "Linear":
                expected_in = layer["in_features"]
                out_feat = layer["out_features"]
                if input_var and input_var in var_shapes:
                    actual = var_shapes[input_var]
                    if actual is not None and actual != expected_in:
                        self.bugs.append(
                            f"self.{layer_name}: got {actual} features, "
                            f"expected {expected_in}")
                if output_var:
                    var_shapes[output_var] = out_feat

            elif lt == "Conv2d":
                expected_in = layer["in_channels"]
                out_ch = layer["out_channels"]
                if input_var and input_var in var_shapes:
                    actual = var_shapes[input_var]
                    if actual is not None and actual != expected_in:
                        self.bugs.append(
                            f"self.{layer_name}: got {actual} channels, "
                            f"expected {expected_in}")
                if output_var:
                    var_shapes[output_var] = out_ch

            elif lt in ("BatchNorm2d", "BatchNorm1d", "InstanceNorm2d"):
                nf = layer["num_features"]
                if input_var and input_var in var_shapes:
                    actual = var_shapes[input_var]
                    if actual is not None and actual != nf:
                        self.bugs.append(
                            f"self.{layer_name}: got {actual} features, "
                            f"expected {nf}")
                if output_var:
                    var_shapes[output_var] = var_shapes.get(input_var)

            elif lt == "GroupNorm":
                nc = layer["num_channels"]
                if input_var and input_var in var_shapes:
                    actual = var_shapes[input_var]
                    if actual is not None and actual != nc:
                        self.bugs.append(
                            f"self.{layer_name}: got {actual} channels, "
                            f"expected {nc}")
                if output_var:
                    var_shapes[output_var] = var_shapes.get(input_var)

            else:
                if input_var and input_var in var_shapes and output_var:
                    var_shapes[output_var] = var_shapes[input_var]

    def _extract_call(self, node) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            out_var = target.id if isinstance(target, ast.Name) else None
            return self._unpack(node.value, out_var)
        elif isinstance(node, ast.Return) and node.value is not None:
            return self._unpack(node.value, None)
        return None

    def _unpack(self, expr, out_var) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
        if not isinstance(expr, ast.Call):
            return None
        func = expr.func
        if (isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "self"):
            in_var = None
            if expr.args:
                arg = expr.args[0]
                if isinstance(arg, ast.Name):
                    in_var = arg.id
                elif isinstance(arg, ast.Call):
                    inner = self._unpack(arg, out_var)
                    if inner:
                        return inner
            return (func.attr, in_var, out_var)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# TensorGuard runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_tensorguard(tc: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.monotonic()
    detected = False
    details = ""
    try:
        vm = verify_model(tc["code"], input_shapes=tc["input_shapes"])
        if not vm.safe:
            detected = True
            if vm.counterexample:
                for v in vm.counterexample.violations[:2]:
                    msg = getattr(v, "message", str(v))
                    details += msg[:150] + "; "
        cegar = run_shape_cegar(
            tc["code"],
            input_shapes=tc["input_shapes"],
            max_iterations=10,
            enable_quality_filter=True,
        )
        if cegar.has_real_bugs:
            detected = True
            if cegar.real_bugs:
                for b in cegar.real_bugs[:2]:
                    msg = getattr(b, "message", str(b))
                    if msg[:50] not in details:
                        details += msg[:150] + "; "
        # Intent-aware analysis (device, phase, gradient, semantic bugs)
        if not detected:
            from src.intent_bugs import OverwarnAnalyzer
            analyzer = OverwarnAnalyzer()
            intent_bugs = analyzer.analyze(tc["code"])
            if intent_bugs:
                detected = True
                for b in intent_bugs[:2]:
                    details += f"[{b.kind.name}] {b.message[:120]}; "
    except Exception as e:
        details = f"ERROR: {e}"
    elapsed = (time.monotonic() - t0) * 1000
    return {"detected_bug": detected, "time_ms": round(elapsed, 2),
            "details": details[:500]}


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    tp = fp = fn = tn = 0
    for r in results:
        gt, det = r["ground_truth"], r["detected_bug"]
        if gt and det: tp += 1
        elif not gt and det: fp += 1
        elif gt and not det: fn += 1
        else: tn += 1
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0
    avg_t = sum(x["time_ms"] for x in results) / len(results) if results else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f1, 4), "accuracy": round(acc, 4),
            "avg_time_ms": round(avg_t, 2)}


def wilson_ci(p_hat: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    centre = p_hat + z * z / (2 * n)
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n)
    lo = max(0.0, (centre - spread) / denom)
    hi = min(1.0, (centre + spread) / denom)
    return (round(lo, 4), round(hi, 4))


def compute_ci(metrics: Dict[str, Any], n: int) -> Dict[str, Tuple[float, float]]:
    return {
        "precision": wilson_ci(metrics["precision"], n),
        "recall": wilson_ci(metrics["recall"], n),
        "f1": wilson_ci(metrics["f1"], n),
        "accuracy": wilson_ci(metrics["accuracy"], n),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    benchmarks = EXPANDED_BENCHMARKS
    n = len(benchmarks)
    n_buggy = sum(1 for b in benchmarks if b["has_bug"])
    n_correct = n - n_buggy

    print("=" * 76)
    print("  Expanded Benchmark Evaluation")
    print(f"  {n} benchmarks ({n_buggy} buggy, {n_correct} correct)")
    print("=" * 76)

    categories: Dict[str, List[str]] = {}
    for tc in benchmarks:
        categories.setdefault(tc["category"], []).append(tc["name"])
    for cat, names in sorted(categories.items()):
        print(f"  {cat}: {len(names)}")

    all_results: Dict[str, List[Dict[str, Any]]] = {
        "syntactic": [], "tensorguard": [],
    }

    for i, tc in enumerate(benchmarks, 1):
        tag = "BUG" if tc["has_bug"] else "OK "
        print(f"\n[{i:3d}/{n}] {tc['name']} ({tag})  {tc['description'][:50]}")

        # Syntactic baseline
        t0 = time.monotonic()
        checker = SyntacticBaseline(tc["code"])
        has_bug, bug_msgs = checker.check()
        syn_ms = (time.monotonic() - t0) * 1000
        syn_ok = has_bug == tc["has_bug"]
        all_results["syntactic"].append({
            "name": tc["name"], "category": tc["category"],
            "ground_truth": tc["has_bug"], "detected_bug": has_bug,
            "time_ms": round(syn_ms, 2),
            "details": "; ".join(bug_msgs)[:200] if bug_msgs else "",
        })
        print(f"  Syn: {'✓' if syn_ok else '✗'}  det={has_bug:<5}  {syn_ms:.1f}ms")

        # TensorGuard
        lp = run_tensorguard(tc)
        lp_ok = lp["detected_bug"] == tc["has_bug"]
        all_results["tensorguard"].append({
            "name": tc["name"], "category": tc["category"],
            "ground_truth": tc["has_bug"], "detected_bug": lp["detected_bug"],
            "time_ms": lp["time_ms"],
            "details": lp["details"],
        })
        print(f"  LP:  {'✓' if lp_ok else '✗'}  det={str(lp['detected_bug']):<5}  "
              f"{lp['time_ms']:.1f}ms")

    # ── Summary ──
    print(f"\n{'=' * 76}")
    print("  METRICS SUMMARY")
    print(f"{'=' * 76}")

    tool_metrics: Dict[str, Any] = {}
    tool_cis: Dict[str, Any] = {}
    for tool in ["syntactic", "tensorguard"]:
        m = compute_metrics(all_results[tool])
        ci = compute_ci(m, n)
        tool_metrics[tool] = m
        tool_cis[tool] = ci
        label = ("Syntactic Baseline" if tool == "syntactic"
                 else "TensorGuard (Z3 + CEGAR)")
        print(f"\n  {label}")
        print(f"    F1={m['f1']:.4f}  P={m['precision']:.4f}  "
              f"R={m['recall']:.4f}  Acc={m['accuracy']:.4f}")
        print(f"    TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}  "
              f"avg={m['avg_time_ms']:.1f}ms")
        print(f"    95% Wilson CI:  F1={ci['f1']}  P={ci['precision']}  "
              f"R={ci['recall']}")

    # ── Per-category ──
    print(f"\n  PER-CATEGORY BREAKDOWN:")
    for cat in sorted(categories.keys()):
        for tool in ["syntactic", "tensorguard"]:
            cat_r = [r for r in all_results[tool] if r["category"] == cat]
            m = compute_metrics(cat_r)
            lbl = "Syn" if tool == "syntactic" else "LP "
            print(f"    {cat:15s} {lbl}: Acc={m['accuracy']:.2f}  "
                  f"F1={m['f1']:.2f}  "
                  f"(TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']})")

    # ── Save ──
    output = {
        "experiment": "expanded_eval",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_benchmarks": n,
        "num_buggy": n_buggy,
        "num_correct": n_correct,
        "categories": {k: len(v) for k, v in categories.items()},
        "tools": {},
    }
    for tool in ["syntactic", "tensorguard"]:
        output["tools"][tool] = {
            "metrics": tool_metrics[tool],
            "wilson_ci_95": {k: list(v) for k, v in tool_cis[tool].items()},
            "per_benchmark": all_results[tool],
        }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
