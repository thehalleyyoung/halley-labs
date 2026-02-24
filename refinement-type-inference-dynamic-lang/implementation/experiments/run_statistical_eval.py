"""
Statistical evaluation for TensorGuard v2.

Runs the 205-benchmark Suite B through TensorGuard and computes:
  - Per-category P/R/F1 with 95% Wilson confidence intervals
  - Bootstrap 95% CIs for F1
  - McNemar paired significance test vs GPT-4.1-nano baseline
  - Overall and per-category breakdown

Saves results to experiments/results/statistical_eval_results.json.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.expanded_benchmark_suite import (
    HUGGINGFACE_BENCHMARKS,
    VISION_BENCHMARKS,
    DEVICE_BENCHMARKS,
    PHASE_BENCHMARKS,
    BROADCAST_BENCHMARKS,
    RESHAPE_BENCHMARKS,
    CHAIN_BENCHMARKS,
    CORRECT_BENCHMARKS,
)
from src.model_checker import verify_model
from src.shape_cegar import run_shape_cegar
from src.intent_bugs import OverwarnAnalyzer

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "statistical_eval_results.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def run_tensorguard(tc: Dict[str, Any]) -> Dict[str, Any]:
    """Run full TensorGuard pipeline: verify_model + CEGAR + intent_bugs."""
    t0 = time.monotonic()
    detected = False
    details = ""
    try:
        vm = verify_model(tc["code"], input_shapes=tc.get("input_shapes", {}))
        if not vm.safe:
            detected = True
            if vm.counterexample:
                for v in vm.counterexample.violations[:2]:
                    msg = getattr(v, "message", str(v))
                    details += msg[:150] + "; "
        if not detected:
            cegar = run_shape_cegar(
                tc["code"],
                input_shapes=tc.get("input_shapes", {}),
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
        if not detected:
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
# Statistical tests
# ═══════════════════════════════════════════════════════════════════════════════

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


def bootstrap_f1_ci(
    labels: List[bool],
    preds: List[bool],
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap 95% CI for F1 score.

    Returns (f1_point, ci_lower, ci_upper).
    """
    rng = random.Random(seed)
    n = len(labels)

    def compute_f1(lbls, prds):
        tp = sum(1 for l, p in zip(lbls, prds) if l and p)
        fp = sum(1 for l, p in zip(lbls, prds) if not l and p)
        fn = sum(1 for l, p in zip(lbls, prds) if l and not p)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    point = compute_f1(labels, preds)
    boot_f1s = []
    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        bl = [labels[i] for i in indices]
        bp = [preds[i] for i in indices]
        boot_f1s.append(compute_f1(bl, bp))

    boot_f1s.sort()
    lo_idx = int(n_bootstrap * alpha / 2)
    hi_idx = int(n_bootstrap * (1 - alpha / 2))
    return (round(point, 4), round(boot_f1s[lo_idx], 4),
            round(boot_f1s[hi_idx], 4))


def mcnemar_test(
    labels: List[bool],
    preds_a: List[bool],
    preds_b: List[bool],
) -> Dict[str, Any]:
    """McNemar's test comparing two classifiers.

    Tests whether classifier A and B have the same error rate.
    Returns chi-squared statistic, p-value, and contingency table.
    """
    # Build contingency table of correct/incorrect
    # b01 = A correct, B incorrect
    # b10 = A incorrect, B correct
    b01 = b10 = 0
    for l, pa, pb in zip(labels, preds_a, preds_b):
        a_correct = (pa == l)
        b_correct = (pb == l)
        if a_correct and not b_correct:
            b01 += 1
        elif not a_correct and b_correct:
            b10 += 1

    # McNemar's chi-squared with continuity correction
    n = b01 + b10
    if n == 0:
        return {"chi2": 0.0, "p_value": 1.0, "b01": b01, "b10": b10,
                "significant": False}

    chi2 = (abs(b01 - b10) - 1) ** 2 / n if n > 0 else 0
    # Approximate p-value from chi2(1)
    # Using the complementary error function approximation
    p_value = math.exp(-chi2 / 2) if chi2 > 0 else 1.0
    # More accurate: use scipy-like chi2 survival function approximation
    # For chi2(1): p = erfc(sqrt(chi2/2))
    p_value = math.erfc(math.sqrt(chi2 / 2))

    return {
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "b01_a_correct_b_wrong": b01,
        "b10_a_wrong_b_correct": b10,
        "significant_at_005": p_value < 0.05,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Compute metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_full_metrics(
    labels: List[bool], preds: List[bool]
) -> Dict[str, Any]:
    """Compute P, R, F1, accuracy with Wilson CIs and bootstrap F1 CI."""
    n = len(labels)
    tp = sum(1 for l, p in zip(labels, preds) if l and p)
    fp = sum(1 for l, p in zip(labels, preds) if not l and p)
    fn = sum(1 for l, p in zip(labels, preds) if l and not p)
    tn = sum(1 for l, p in zip(labels, preds) if not l and not p)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / n if n > 0 else 0.0

    f1_point, f1_lo, f1_hi = bootstrap_f1_ci(labels, preds)

    return {
        "n": n,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(prec, 4),
        "precision_ci": wilson_ci(prec, tp + fp) if (tp + fp) > 0 else (0, 0),
        "recall": round(rec, 4),
        "recall_ci": wilson_ci(rec, tp + fn) if (tp + fn) > 0 else (0, 0),
        "f1": round(f1, 4),
        "f1_bootstrap_ci": [f1_lo, f1_hi],
        "accuracy": round(acc, 4),
        "accuracy_ci": wilson_ci(acc, n),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GPT-4.1-nano baseline results (from prior evaluation)
# ═══════════════════════════════════════════════════════════════════════════════

# These are the GPT-4.1-nano predictions from the frontier_llm_baseline run
# Indexed by benchmark name -> predicted_bug
GPT_PREDICTIONS_FILE = Path(__file__).parent / "frontier_llm_baseline_results.json"


def load_gpt_predictions() -> Dict[str, bool]:
    """Load GPT-4.1-nano predictions from prior run."""
    if not GPT_PREDICTIONS_FILE.exists():
        return {}
    with open(GPT_PREDICTIONS_FILE) as f:
        data = json.load(f)
    preds = {}
    # Try to extract per-benchmark predictions
    if "per_benchmark" in data:
        for entry in data["per_benchmark"]:
            preds[entry["name"]] = entry.get("detected_bug", False)
    elif "curated_results" in data:
        for entry in data["curated_results"]:
            preds[entry["name"]] = entry.get("llm_detected", False)
    return preds


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ALL_BENCHMARKS = (
        HUGGINGFACE_BENCHMARKS + VISION_BENCHMARKS + DEVICE_BENCHMARKS +
        PHASE_BENCHMARKS + BROADCAST_BENCHMARKS + RESHAPE_BENCHMARKS +
        CHAIN_BENCHMARKS + CORRECT_BENCHMARKS
    )

    n = len(ALL_BENCHMARKS)
    print("=" * 76)
    print("  TensorGuard Statistical Evaluation (Suite B)")
    print(f"  {n} benchmarks")
    print("=" * 76)

    # Run TensorGuard on all benchmarks
    labels = []
    lp_preds = []
    per_cat: Dict[str, Dict[str, List[bool]]] = {}

    for i, tc in enumerate(ALL_BENCHMARKS, 1):
        cat = tc.get("category", "unknown")
        gt = tc.get("has_bug", False)

        result = run_tensorguard(tc)
        pred = result["detected_bug"]

        labels.append(gt)
        lp_preds.append(pred)

        if cat not in per_cat:
            per_cat[cat] = {"labels": [], "preds": []}
        per_cat[cat]["labels"].append(gt)
        per_cat[cat]["preds"].append(pred)

        status = "✓" if (pred == gt) else "✗"
        if i % 20 == 0 or i == n:
            print(f"  [{i:3d}/{n}] processed...")

    # Overall metrics
    overall = compute_full_metrics(labels, lp_preds)
    print(f"\n{'='*76}")
    print("  OVERALL RESULTS")
    print(f"{'='*76}")
    print(f"  P={overall['precision']:.3f}  R={overall['recall']:.3f}  "
          f"F1={overall['f1']:.3f}  Acc={overall['accuracy']:.3f}")
    print(f"  F1 bootstrap 95% CI: [{overall['f1_bootstrap_ci'][0]:.3f}, "
          f"{overall['f1_bootstrap_ci'][1]:.3f}]")
    print(f"  TP={overall['tp']}  FP={overall['fp']}  "
          f"TN={overall['tn']}  FN={overall['fn']}")

    # Per-category
    cat_metrics = {}
    print(f"\n  Per-Category Breakdown:")
    print(f"  {'Category':15s} {'n':>4s} {'TP':>4s} {'FP':>4s} {'TN':>4s} "
          f"{'FN':>4s} {'P':>6s} {'R':>6s} {'F1':>6s} {'F1 95% CI':>16s}")
    print(f"  {'-'*80}")
    for cat in ['huggingface', 'vision', 'device', 'phase', 'broadcast',
                'reshape', 'chain', 'correct']:
        if cat not in per_cat:
            continue
        cm = compute_full_metrics(per_cat[cat]["labels"], per_cat[cat]["preds"])
        cat_metrics[cat] = cm
        ci = cm['f1_bootstrap_ci']
        print(f"  {cat:15s} {cm['n']:4d} {cm['tp']:4d} {cm['fp']:4d} "
              f"{cm['tn']:4d} {cm['fn']:4d} {cm['precision']:6.3f} "
              f"{cm['recall']:6.3f} {cm['f1']:6.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")

    # McNemar test vs GPT-4.1-nano
    gpt_preds_map = load_gpt_predictions()
    mcnemar_result = None
    if gpt_preds_map:
        gpt_preds = []
        matched_labels = []
        matched_lp = []
        for tc, lbl, lp in zip(ALL_BENCHMARKS, labels, lp_preds):
            name = tc["name"]
            if name in gpt_preds_map:
                gpt_preds.append(gpt_preds_map[name])
                matched_labels.append(lbl)
                matched_lp.append(lp)

        if matched_labels:
            mcnemar_result = mcnemar_test(matched_labels, matched_lp, gpt_preds)
            print(f"\n  McNemar Test (TensorGuard vs GPT-4.1-nano, n={len(matched_labels)}):")
            print(f"    χ²={mcnemar_result['chi2']:.3f}  "
                  f"p={mcnemar_result['p_value']:.6f}  "
                  f"significant={mcnemar_result['significant_at_005']}")
            print(f"    TensorGuard correct, GPT wrong: "
                  f"{mcnemar_result['b01_a_correct_b_wrong']}")
            print(f"    TensorGuard wrong, GPT correct: "
                  f"{mcnemar_result['b10_a_wrong_b_correct']}")
    else:
        print("\n  [No GPT-4.1-nano predictions available for McNemar test]")

    # Save results
    output = {
        "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "suite": "B",
        "n_benchmarks": n,
        "overall": overall,
        "per_category": cat_metrics,
        "mcnemar_vs_gpt": mcnemar_result,
        "improvements_over_prior": {
            "prior_f1": 0.876,
            "new_f1": overall["f1"],
            "prior_huggingface_recall": 0.588,
            "new_huggingface_recall": cat_metrics.get("huggingface", {}).get("recall", 0),
            "prior_broadcast_recall": 0.688,
            "new_broadcast_recall": cat_metrics.get("broadcast", {}).get("recall", 0),
        },
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
