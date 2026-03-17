#!/usr/bin/env python3
"""
Regression Test for Contract Correctness
=========================================

Deterministic regression test that verifies contract synthesis correctness
invariants hold across all 22 benchmark functions.  Uses a fixed random seed
so results are reproducible.

Exit code 0 = all checks pass, 1 = at least one failure.
"""

import json
import os
import sys

# Ensure reproducibility
import numpy as np
import random
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Import the benchmark module
sys.path.insert(0, os.path.dirname(__file__))
from contract_correctness_benchmark import (
    GROUND_TRUTH,
    FUNC_TO_CAT,
    _simulate_synthesized_contract,
    _simulate_mutations,
    check_soundness,
    check_precision,
    check_regression,
)

# -------------------------------------------------------------------
# Thresholds — these encode our minimum acceptable correctness bounds.
# -------------------------------------------------------------------
MIN_MEAN_SOUNDNESS = 0.55
MIN_MEAN_PRECISION = 0.80
MIN_POST_RECALL    = 0.40
MIN_REGRESSION_RATE = 0.15  # at least 15% of functions fully pass
MIN_PER_FUNC_SOUNDNESS = 0.10  # floor for any individual function


def test_contract_correctness():
    """Run deterministic correctness checks and report pass/fail."""
    print("Contract Correctness Regression Test (seed=%d)" % SEED)
    print("=" * 55)

    all_soundness = []
    all_precision = []
    all_post_recall = []
    reg_pass = 0
    reg_total = 0
    failures = []

    for func_name, gt in GROUND_TRUTH.items():
        cat = FUNC_TO_CAT[func_name]
        synth_pre, synth_post = _simulate_synthesized_contract(gt, cat)
        mutants = _simulate_mutations(func_name, cat)

        snd = check_soundness(synth_post, gt["postconditions"], mutants, cat)
        prec = check_precision(synth_post, gt["postconditions"], mutants)
        reg = check_regression(synth_pre, synth_post, gt)

        all_soundness.append(snd["soundness"])
        all_precision.append(prec["precision"])
        all_post_recall.append(reg["postcondition_recall"])
        reg_total += 1
        if reg["pass"]:
            reg_pass += 1

        # Per-function soundness floor
        if snd["soundness"] < MIN_PER_FUNC_SOUNDNESS:
            failures.append(f"  FAIL {func_name}: soundness {snd['soundness']:.3f} < {MIN_PER_FUNC_SOUNDNESS}")

    mean_snd = float(np.mean(all_soundness))
    mean_prec = float(np.mean(all_precision))
    mean_post_rec = float(np.mean(all_post_recall))
    reg_rate = reg_pass / max(reg_total, 1)

    print(f"\n  Mean soundness:       {mean_snd:.3f}  (threshold {MIN_MEAN_SOUNDNESS})")
    print(f"  Mean precision:       {mean_prec:.3f}  (threshold {MIN_MEAN_PRECISION})")
    print(f"  Mean post recall:     {mean_post_rec:.3f}  (threshold {MIN_POST_RECALL})")
    print(f"  Regression rate:      {reg_rate:.3f}  (threshold {MIN_REGRESSION_RATE})")

    if mean_snd < MIN_MEAN_SOUNDNESS:
        failures.append(f"  FAIL mean soundness {mean_snd:.3f} < {MIN_MEAN_SOUNDNESS}")
    if mean_prec < MIN_MEAN_PRECISION:
        failures.append(f"  FAIL mean precision {mean_prec:.3f} < {MIN_MEAN_PRECISION}")
    if mean_post_rec < MIN_POST_RECALL:
        failures.append(f"  FAIL mean post recall {mean_post_rec:.3f} < {MIN_POST_RECALL}")
    if reg_rate < MIN_REGRESSION_RATE:
        failures.append(f"  FAIL regression rate {reg_rate:.3f} < {MIN_REGRESSION_RATE}")

    if failures:
        print("\n" + "-" * 55)
        print("FAILURES:")
        for f in failures:
            print(f)
        print("-" * 55)
        print("RESULT: FAIL")
        return False
    else:
        print("\n" + "-" * 55)
        print("RESULT: ALL CHECKS PASSED")
        print("-" * 55)
        return True


if __name__ == "__main__":
    ok = test_contract_correctness()
    sys.exit(0 if ok else 1)
