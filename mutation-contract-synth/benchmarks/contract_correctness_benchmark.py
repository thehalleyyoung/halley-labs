#!/usr/bin/env python3
"""
Contract Correctness Verification Benchmark
============================================

Validates the correctness of synthesized contracts against known-correct
(ground-truth) specifications across the MutSpec benchmark suite.

Three metrics:
  1. **Soundness** — no false negatives: contracts never miss a real mutation.
  2. **Precision** — fraction of detected mutations that are real bugs.
  3. **Regression** — per-function pass/fail against ground truth.

NOTE: Like fast_benchmark.py, this script generates *simulated* results using
parametric distributions calibrated to structural complexity.  It does NOT
execute the Rust MutSpec binary.  Replace with the Rust harness when
available.
"""

import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Ground-truth specifications per function (shared with fast_benchmark.py)
# ---------------------------------------------------------------------------

GROUND_TRUTH: Dict[str, Dict[str, List[str]]] = {
    # Arithmetic
    "clamp": {
        "preconditions": ["lo <= hi"],
        "postconditions": [
            "ret >= lo", "ret <= hi",
            "ret == x || ret == lo || ret == hi",
        ],
    },
    "abs": {
        "preconditions": ["true"],
        "postconditions": [
            "ret >= 0",
            "(x >= 0) -> ret == x",
            "(x < 0) -> ret == -x",
        ],
    },
    "max_fn": {
        "preconditions": ["true"],
        "postconditions": [
            "ret >= a", "ret >= b",
            "ret == a || ret == b",
        ],
    },
    "gcd": {
        "preconditions": ["a > 0", "b > 0"],
        "postconditions": [
            "ret > 0", "ret <= a", "ret <= b",
            "a % ret == 0", "b % ret == 0",
        ],
    },
    "is_prime": {
        "preconditions": ["true"],
        "postconditions": [
            "(n < 2) -> ret == false",
            "(n == 2) -> ret == true",
        ],
    },
    "midpoint": {
        "preconditions": ["true"],
        "postconditions": [
            "ret >= min(a, b)",
            "ret <= max(a, b)",
        ],
    },
    # Data structures
    "bst_contains": {
        "preconditions": ["true"],
        "postconditions": [
            "(root == null) -> ret == false",
            "(root.val == key) -> ret == true",
        ],
    },
    "bst_min": {
        "preconditions": ["root != null"],
        "postconditions": ["ret <= root.val"],
    },
    "list_length": {
        "preconditions": ["true"],
        "postconditions": [
            "ret >= 0",
            "(head == null) -> ret == 0",
        ],
    },
    "stack_push": {
        "preconditions": ["size < capacity"],
        "postconditions": ["new_size == old_size + 1"],
    },
    "stack_pop": {
        "preconditions": ["size > 0"],
        "postconditions": ["new_size == old_size - 1"],
    },
    "stack_peek": {
        "preconditions": ["size > 0"],
        "postconditions": [
            "ret == stack[size - 1]",
            "new_size == old_size",
        ],
    },
    "stack_is_empty": {
        "preconditions": ["true"],
        "postconditions": [
            "(size == 0) -> ret == true",
            "(size > 0) -> ret == false",
        ],
    },
    # Search/sort
    "binary_search": {
        "preconditions": ["is_sorted(arr)"],
        "postconditions": ["ret >= -1", "ret < len(arr)"],
    },
    "linear_search": {
        "preconditions": ["true"],
        "postconditions": ["ret >= -1", "ret <= 3"],
    },
    "insertion_sort_step": {
        "preconditions": ["n >= 0", "n < arr_len"],
        "postconditions": ["forall i in [0..n]: arr[i] <= arr[i+1]"],
    },
    # String
    "str_len": {
        "preconditions": ["s != null"],
        "postconditions": ["ret >= 0"],
    },
    "char_at": {
        "preconditions": ["s != null"],
        "postconditions": ["ret != '\\0'"],
    },
    "index_of": {
        "preconditions": ["s != null"],
        "postconditions": ["ret >= -1", "ret < str_len(s)"],
    },
    # Array
    "safe_get": {
        "preconditions": ["true"],
        "postconditions": ["(idx >= 0 && idx < 8) -> ret == arr[idx]"],
    },
    "safe_set": {
        "preconditions": ["true"],
        "postconditions": ["ret[idx] == val"],
    },
    "bounded_increment": {
        "preconditions": ["true"],
        "postconditions": ["ret >= lo", "ret <= hi"],
    },
}

# Category assignments mirror fast_benchmark.py
CATEGORIES: Dict[str, List[str]] = {
    "arithmetic":     ["clamp", "abs", "max_fn", "gcd", "is_prime", "midpoint"],
    "data_structure": ["bst_contains", "bst_min", "list_length",
                       "stack_push", "stack_pop", "stack_peek", "stack_is_empty"],
    "search_sort":    ["binary_search", "linear_search", "insertion_sort_step"],
    "string":         ["str_len", "char_at", "index_of"],
    "array":          ["safe_get", "safe_set", "bounded_increment"],
}

FUNC_TO_CAT = {f: cat for cat, fns in CATEGORIES.items() for f in fns}


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _simulate_synthesized_contract(
    gt: Dict[str, List[str]], cat: str
) -> Tuple[List[str], List[str]]:
    """Return a simulated synthesized contract (subset of ground truth + noise).

    MutSpec-style synthesizers recover a high fraction of ground truth and
    occasionally add a spurious clause (imprecision).
    """
    if cat in ("arithmetic", "array"):
        recall_rate = np.random.beta(6, 2) * 0.5 + 0.4
    elif cat in ("search_sort", "data_structure"):
        recall_rate = np.random.beta(5, 2.5) * 0.5 + 0.3
    else:
        recall_rate = np.random.beta(4, 3) * 0.5 + 0.2

    def sample_clauses(clauses: List[str]) -> List[str]:
        n = max(1, int(len(clauses) * recall_rate))
        return random.sample(clauses, min(n, len(clauses)))

    pre = sample_clauses(gt["preconditions"])
    post = sample_clauses(gt["postconditions"])

    # With ~7% probability inject one spurious postcondition (false positive)
    if random.random() < 0.07:
        post.append("ret != INVALID_SENTINEL")

    return pre, post


def _simulate_mutations(func: str, cat: str) -> List[Dict[str, Any]]:
    """Simulate a set of mutations for a function.

    Each mutation is either a *real* bug or an *equivalent* mutant.
    """
    if cat == "arithmetic":
        n_mutants = random.randint(5, 12)
        equiv_rate = 0.10
    elif cat == "data_structure":
        n_mutants = random.randint(4, 10)
        equiv_rate = 0.15
    elif cat == "search_sort":
        n_mutants = random.randint(4, 8)
        equiv_rate = 0.12
    else:
        n_mutants = random.randint(3, 7)
        equiv_rate = 0.18

    ops = ["AOR", "ROR", "LCR", "UOI"]
    mutants = []
    for i in range(n_mutants):
        is_equiv = random.random() < equiv_rate
        mutants.append({
            "id": f"M{i+1}",
            "operator": random.choice(ops),
            "is_equivalent": is_equiv,
            "is_real_bug": not is_equiv,
        })
    return mutants


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def check_soundness(
    synth_post: List[str],
    gt_post: List[str],
    mutants: List[Dict[str, Any]],
    cat: str,
) -> Dict[str, Any]:
    """Soundness = the contract never *misses* a real (non-equivalent) mutation.

    A contract is sound for a mutation if the mutation either:
      (a) violates at least one synthesized postcondition, or
      (b) is equivalent (no behavioural difference).

    We simulate detection probability per mutant based on clause overlap with
    ground truth — the more ground-truth clauses present, the higher the
    probability that a real bug is caught.
    """
    gt_set = set(gt_post)
    synth_set = set(synth_post)
    overlap = len(synth_set & gt_set) / max(len(gt_set), 1)

    real_mutants = [m for m in mutants if m["is_real_bug"]]
    detected = 0
    missed = 0

    for m in real_mutants:
        # Detection probability scales with clause overlap and category
        if cat in ("arithmetic", "array"):
            base_detect = 0.90
        elif cat in ("search_sort", "data_structure"):
            base_detect = 0.85
        else:
            base_detect = 0.75
        p_detect = min(1.0, overlap * base_detect + random.gauss(0, 0.03))
        if random.random() < p_detect:
            detected += 1
        else:
            missed += 1

    total_real = len(real_mutants)
    soundness = detected / max(total_real, 1)
    return {
        "total_real_mutations": total_real,
        "detected": detected,
        "missed": missed,
        "soundness": round(soundness, 4),
    }


def check_precision(
    synth_post: List[str],
    gt_post: List[str],
    mutants: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Precision = fraction of *detected* mutations that are real bugs.

    A spurious clause may flag equivalent mutants as violations (false
    positives).  We simulate this: each spurious clause has a 40% chance
    of falsely flagging each equivalent mutant.
    """
    gt_set = set(gt_post)
    synth_set = set(synth_post)
    spurious = synth_set - gt_set

    equiv_mutants = [m for m in mutants if m["is_equivalent"]]
    false_positives = 0
    for _ in spurious:
        for _ in equiv_mutants:
            if random.random() < 0.40:
                false_positives += 1

    real_mutants_detected = sum(1 for m in mutants if m["is_real_bug"])
    total_flagged = real_mutants_detected + false_positives
    precision = real_mutants_detected / max(total_flagged, 1)

    return {
        "real_detected": real_mutants_detected,
        "false_positives": false_positives,
        "total_flagged": total_flagged,
        "precision": round(precision, 4),
    }


def check_regression(
    synth_pre: List[str],
    synth_post: List[str],
    gt: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Regression check: every ground-truth clause should appear in the
    synthesized contract (or a logically equivalent form).

    Returns per-clause pass/fail and an overall pass/fail.
    """
    gt_pre_set = set(gt["preconditions"])
    gt_post_set = set(gt["postconditions"])
    synth_pre_set = set(synth_pre)
    synth_post_set = set(synth_post)

    pre_results = {c: (c in synth_pre_set) for c in gt_pre_set}
    post_results = {c: (c in synth_post_set) for c in gt_post_set}

    missing_pre = [c for c, ok in pre_results.items() if not ok]
    missing_post = [c for c, ok in post_results.items() if not ok]

    all_pass = len(missing_pre) == 0 and len(missing_post) == 0
    pre_recall = sum(pre_results.values()) / max(len(pre_results), 1)
    post_recall = sum(post_results.values()) / max(len(post_results), 1)

    return {
        "pass": all_pass,
        "precondition_recall": round(pre_recall, 4),
        "postcondition_recall": round(post_recall, 4),
        "missing_preconditions": missing_pre,
        "missing_postconditions": missing_post,
    }


# ---------------------------------------------------------------------------
# Main benchmark driver
# ---------------------------------------------------------------------------

def run_contract_correctness_benchmark() -> Dict[str, Any]:
    """Run the full contract correctness verification benchmark."""
    print("Contract Correctness Verification Benchmark")
    print("=" * 50)
    print("NOTE: Simulated results from parametric distributions,")
    print("      NOT measured from the Rust MutSpec binary.\n")

    results: Dict[str, Any] = {}
    all_soundness: List[float] = []
    all_precision: List[float] = []
    all_pre_recall: List[float] = []
    all_post_recall: List[float] = []
    regression_passes = 0
    regression_total = 0

    for func_name, gt in GROUND_TRUTH.items():
        cat = FUNC_TO_CAT[func_name]
        print(f"  Verifying {func_name} ({cat})...")

        synth_pre, synth_post = _simulate_synthesized_contract(gt, cat)
        mutants = _simulate_mutations(func_name, cat)

        snd = check_soundness(synth_post, gt["postconditions"], mutants, cat)
        prec = check_precision(synth_post, gt["postconditions"], mutants)
        reg = check_regression(synth_pre, synth_post, gt)

        all_soundness.append(snd["soundness"])
        all_precision.append(prec["precision"])
        all_pre_recall.append(reg["precondition_recall"])
        all_post_recall.append(reg["postcondition_recall"])
        regression_total += 1
        if reg["pass"]:
            regression_passes += 1

        results[func_name] = {
            "category": cat,
            "ground_truth_clauses": {
                "preconditions": len(gt["preconditions"]),
                "postconditions": len(gt["postconditions"]),
            },
            "synthesized_clauses": {
                "preconditions": len(synth_pre),
                "postconditions": len(synth_post),
            },
            "soundness": snd,
            "precision": prec,
            "regression": reg,
        }

    # Aggregate summary
    summary = {
        "total_functions": len(GROUND_TRUTH),
        "soundness": {
            "mean": round(float(np.mean(all_soundness)), 4),
            "std": round(float(np.std(all_soundness)), 4),
            "min": round(float(np.min(all_soundness)), 4),
            "max": round(float(np.max(all_soundness)), 4),
        },
        "precision": {
            "mean": round(float(np.mean(all_precision)), 4),
            "std": round(float(np.std(all_precision)), 4),
            "min": round(float(np.min(all_precision)), 4),
            "max": round(float(np.max(all_precision)), 4),
        },
        "precondition_recall": {
            "mean": round(float(np.mean(all_pre_recall)), 4),
            "std": round(float(np.std(all_pre_recall)), 4),
        },
        "postcondition_recall": {
            "mean": round(float(np.mean(all_post_recall)), 4),
            "std": round(float(np.std(all_post_recall)), 4),
        },
        "regression": {
            "passed": regression_passes,
            "total": regression_total,
            "rate": round(regression_passes / max(regression_total, 1), 4),
        },
    }
    results["summary"] = summary

    # Print summary
    print("\n" + "=" * 50)
    print("CONTRACT CORRECTNESS SUMMARY")
    print("=" * 50)
    s = summary
    print(f"  Soundness:            {s['soundness']['mean']:.3f} ± {s['soundness']['std']:.3f}")
    print(f"  Precision:            {s['precision']['mean']:.3f} ± {s['precision']['std']:.3f}")
    print(f"  Precondition recall:  {s['precondition_recall']['mean']:.3f} ± {s['precondition_recall']['std']:.3f}")
    print(f"  Postcondition recall: {s['postcondition_recall']['mean']:.3f} ± {s['postcondition_recall']['std']:.3f}")
    print(f"  Regression pass rate: {s['regression']['passed']}/{s['regression']['total']}"
          f" ({s['regression']['rate']*100:.1f}%)")

    return results


def main():
    results = run_contract_correctness_benchmark()

    out_path = os.path.join(os.path.dirname(__file__), "contract_correctness_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Results saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
