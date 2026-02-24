#!/usr/bin/env python3
"""
LLM Expanded Baseline: GPT-4.1-nano on 205 expanded benchmarks.

Sends each nn.Module benchmark to the OpenAI API and asks whether it
contains a shape/device/phase bug.  Computes precision, recall, F1 with
Wilson score 95% confidence intervals.

Usage (from implementation/):
    source ~/.bashrc
    python experiments/run_llm_expanded_baseline.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

EXPERIMENTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

from expanded_benchmark_suite import EXPANDED_BENCHMARKS

OUTPUT_FILE = EXPERIMENTS_DIR / "llm_expanded_baseline_results.json"

# ═══════════════════════════════════════════════════════════════════════
# Wilson score interval
# ═══════════════════════════════════════════════════════════════════════

def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Return (lower, upper) of Wilson score 95% CI."""
    if total == 0:
        return (0.0, 0.0)
    p_hat = successes / total
    denom = 1 + z * z / total
    centre = (p_hat + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * total)) / total) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ═══════════════════════════════════════════════════════════════════════
# LLM helper
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a PyTorch bug detector. The user will provide an nn.Module or "
    "PyTorch function. Determine whether it contains a shape mismatch, device "
    "inconsistency, or train/eval phase bug.\n"
    "Answer YES or NO on the first line, then briefly explain."
)


def query_llm(client, code: str, model: str) -> tuple[bool, str]:
    """Query the OpenAI model. Returns (predicted_bug, raw_text)."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Does this nn.Module have a shape/device/phase bug? "
                        "Answer YES or NO, then briefly explain.\n\n"
                        f"```python\n{code}\n```"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=256,
        )
        text = resp.choices[0].message.content.strip()
        first_word = text.split()[0].strip(".:,").upper() if text else ""
        predicted_bug = first_word == "YES"
        return predicted_bug, text
    except Exception as e:
        return None, f"ERROR: {e}"


# ═══════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, Any]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0

    prec_ci = wilson_ci(tp, tp + fp)
    rec_ci = wilson_ci(tp, tp + fn)

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(precision, 4),
        "precision_ci_95": [round(prec_ci[0], 4), round(prec_ci[1], 4)],
        "recall": round(recall, 4),
        "recall_ci_95": [round(rec_ci[0], 4), round(rec_ci[1], 4)],
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

    model = "gpt-4.1-nano"
    total = len(EXPANDED_BENCHMARKS)
    print("=" * 72)
    print(f"LLM Expanded Baseline  —  {model}  —  {total} benchmarks")
    print("=" * 72)

    tp = fp = fn = tn = 0
    skipped = 0
    rows: List[Dict[str, Any]] = []

    for i, bm in enumerate(EXPANDED_BENCHMARKS, 1):
        name = bm["name"]
        has_bug = bm["has_bug"]
        code = bm["code"]
        category = bm.get("category", "unknown")

        print(f"[{i:3d}/{total}] {name:<50s} (bug={has_bug})", end="  ", flush=True)

        t0 = time.time()
        predicted, raw = query_llm(client, code, model)
        elapsed_ms = round((time.time() - t0) * 1000, 1)

        if predicted is None:
            print(f"SKIPPED  ({raw})")
            skipped += 1
            rows.append({
                "name": name, "category": category,
                "has_bug": has_bug, "predicted": None,
                "label": "SKIP", "ms": elapsed_ms,
                "llm_response": raw,
            })
            time.sleep(0.1)
            continue

        # Classify
        if has_bug:
            if predicted:
                tp += 1; label = "TP"
            else:
                fn += 1; label = "FN"
        else:
            if predicted:
                fp += 1; label = "FP"
            else:
                tn += 1; label = "TN"

        print(f"{label}  ({elapsed_ms}ms)")

        rows.append({
            "name": name, "category": category,
            "has_bug": has_bug, "predicted": predicted,
            "label": label, "ms": elapsed_ms,
            "llm_response": raw,
        })

        time.sleep(0.1)

    # ── Metrics ───────────────────────────────────────────────────────
    m = compute_metrics(tp, fp, fn, tn)

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"  Total benchmarks : {total}")
    print(f"  Skipped (errors) : {skipped}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision : {m['precision']:.4f}  95% CI {m['precision_ci_95']}")
    print(f"  Recall    : {m['recall']:.4f}  95% CI {m['recall_ci_95']}")
    print(f"  F1        : {m['f1']:.4f}")
    print(f"  Accuracy  : {m['accuracy']:.4f}")

    # ── Per-category breakdown ────────────────────────────────────────
    cats: Dict[str, Dict[str, int]] = {}
    for r in rows:
        c = r["category"]
        if c not in cats:
            cats[c] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        lbl = r["label"]
        if lbl in ("TP", "FP", "FN", "TN"):
            cats[c][lbl.lower()] += 1

    print("\n" + "-" * 72)
    print(f"{'Category':<25} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} {'F1':>7}")
    print("-" * 72)
    for cat in sorted(cats):
        d = cats[cat]
        cm = compute_metrics(d["tp"], d["fp"], d["fn"], d["tn"])
        print(f"{cat:<25} {d['tp']:>4} {d['fp']:>4} {d['fn']:>4} {d['tn']:>4} {cm['f1']:>7.4f}")

    # ── Save ──────────────────────────────────────────────────────────
    output = {
        "model": model,
        "num_benchmarks": total,
        "skipped": skipped,
        "metrics": m,
        "per_category": {
            cat: compute_metrics(d["tp"], d["fp"], d["fn"], d["tn"])
            for cat, d in sorted(cats.items())
        },
        "benchmarks": rows,
    }
    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
