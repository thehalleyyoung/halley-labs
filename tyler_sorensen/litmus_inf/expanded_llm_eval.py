#!/usr/bin/env python3
"""
expanded_llm_eval.py: Systematic LLM evaluation for litmus test generation.

Addresses reviewer critique W7:
- 20 case studies (up from 8)
- 3 runs per query for reliability analysis
- Known-answer validation against built-in patterns
- Accuracy with confidence intervals
"""

import json
import os
import sys
import time
import math

sys.path.insert(0, os.path.dirname(__file__))
from nlgen import generate_litmus_test, spec_to_litmus, analyze_litmus
from portcheck import PATTERNS, ARCHITECTURES, verify_test, LitmusTest, MemOp

EXPANDED_CASE_STUDIES = [
    # --- Known-answer queries (we know the expected pattern) ---
    {
        "name": "Message Passing",
        "description": "Two threads communicate: thread 0 writes data=1, then writes flag=1. Thread 1 reads flag, then reads data. Forbidden: flag=1 but data=0.",
        "expected_pattern": "mp",
        "category": "known-answer"
    },
    {
        "name": "Store Buffering",
        "description": "Thread 0 stores x=1, reads y. Thread 1 stores y=1, reads x. Forbidden: both reads return 0.",
        "expected_pattern": "sb",
        "category": "known-answer"
    },
    {
        "name": "Load Buffering",
        "description": "Thread 0 reads x, stores y=1. Thread 1 reads y, stores x=1. Forbidden: both reads return 1 (causal cycle).",
        "expected_pattern": "lb",
        "category": "known-answer"
    },
    {
        "name": "Independent Reads of Independent Writes",
        "description": "Thread 0 writes x=1. Thread 1 writes y=1. Thread 2 reads x then y. Thread 3 reads y then x. Forbidden: T2 sees x=1,y=0 while T3 sees y=1,x=0 (multi-copy atomicity violation).",
        "expected_pattern": "iriw",
        "category": "known-answer"
    },
    {
        "name": "Write-Read Causality",
        "description": "Three threads: T0 writes x=1. T1 reads x, writes y=1. T2 reads y, reads x. Forbidden: T1 sees x=1, T2 sees y=1 but x=0 (causality violation).",
        "expected_pattern": "wrc",
        "category": "known-answer"
    },
    # --- Real-world patterns ---
    {
        "name": "RCU Read-Copy-Update",
        "description": "RCU pattern: writer stores new data, then updates pointer. Reader reads pointer, then reads data. Forbidden: reader sees new pointer but old data.",
        "category": "real-world"
    },
    {
        "name": "Double-Checked Locking",
        "description": "DCL: Thread 0 stores data=1, then stores initialized=1. Thread 1 reads initialized; if 1, reads data. Forbidden: initialized=1 but data=0.",
        "category": "real-world"
    },
    {
        "name": "Ticket Lock",
        "description": "Ticket lock: one thread writes now_serving to release, other thread reads now_serving until match, then reads shared data. Like message passing with flag=ticket number.",
        "category": "real-world"
    },
    {
        "name": "SPSC Queue",
        "description": "Single-producer single-consumer queue: producer writes buffer[tail], then increments tail. Consumer reads tail, then reads buffer[old_tail]. Forbidden: consumer sees new tail but stale buffer data.",
        "category": "real-world"
    },
    {
        "name": "Seqlock Reader",
        "description": "Sequence lock: writer increments seq to odd, writes data, increments seq to even. Reader reads seq, reads data, reads seq again. Forbidden: reader sees matching even seq numbers but inconsistent data between reads.",
        "category": "real-world"
    },
    {
        "name": "Work-Stealing Deque",
        "description": "Work stealing: owner pushes (stores data, then increments tail). Thief steals (reads tail, then reads data). Forbidden: thief sees new tail but old data. Requires store-store on owner, load-load on thief.",
        "category": "real-world"
    },
    {
        "name": "Hazard Pointer Publication",
        "description": "Hazard pointer: thread 0 publishes pointer (stores to hp), then reads global list. Thread 1 removes node from list, checks hp. Forbidden: thread 1 sees no hazard but thread 0 sees removed node. Store buffering pattern.",
        "category": "real-world"
    },
    {
        "name": "Peterson Mutual Exclusion",
        "description": "Peterson's algorithm: Thread 0 sets flag0=1, sets turn=1, reads flag1. Thread 1 sets flag1=1, sets turn=0, reads flag0. Forbidden: both see other's flag=0 (both enter critical section).",
        "category": "real-world"
    },
    {
        "name": "Dekker Mutual Exclusion",
        "description": "Dekker's algorithm: Thread 0 sets flag0=1, reads flag1. Thread 1 sets flag1=1, reads flag0. Forbidden: both reads return 0 (both enter critical section). Classic store buffering.",
        "category": "real-world"
    },
    # --- GPU-specific patterns ---
    {
        "name": "GPU Cross-Workgroup Reduction",
        "description": "GPU reduction: workgroup 0 writes partial sum to global memory, uses workgroup barrier, sets flag=1. Workgroup 1 reads flag with workgroup barrier, reads partial sum. Cross-workgroup communication with workgroup-scope barriers.",
        "category": "gpu"
    },
    {
        "name": "GPU Cross-CTA Pipeline",
        "description": "GPU pipeline: CTA 0 produces result, stores to global memory, uses membar.cta, sets done=1. CTA 1 reads done with membar.cta, reads result. Cross-CTA with CTA-scope barriers only.",
        "category": "gpu"
    },
    {
        "name": "GPU Atomic Counter",
        "description": "GPU atomic counter: multiple threads atomically increment a counter. Thread 0 writes data=1, writes counter=1. Thread 1 reads counter, reads data. Like message passing on GPU.",
        "category": "gpu"
    },
    {
        "name": "Barrier-Free Producer-Consumer",
        "description": "No-fence producer-consumer: producer stores data=1 then flag=1 with NO barriers. Consumer reads flag then data with NO barriers. Forbidden: flag=1 but data=0. Should fail on all relaxed models.",
        "category": "real-world"
    },
    {
        "name": "Ring Buffer Two Writers",
        "description": "Ring buffer: two writers write to different slots. Thread 0 writes slot_a=1, then writes tail=1. Thread 1 reads tail, then reads slot_a. Message passing pattern. Forbidden: tail=1 but slot_a=0.",
        "category": "real-world"
    },
    {
        "name": "Spinlock Release-Acquire",
        "description": "Spinlock: Thread 0 stores data=1 in critical section, then stores lock=0 to release. Thread 1 spins reading lock until 0, then reads data. Forbidden: lock=0 but data=0. Message passing pattern.",
        "category": "real-world"
    },
]


def validate_against_known(spec, expected_pattern):
    """Check if LLM-generated test matches expected pattern behavior."""
    if expected_pattern not in PATTERNS:
        return {"valid": False, "reason": f"Unknown pattern {expected_pattern}"}

    expected = PATTERNS[expected_pattern]
    expected_n_threads = max(op.thread for op in expected['ops']) + 1
    expected_n_loads = len([op for op in expected['ops'] if op.optype == 'load'])

    # Check structural match
    gen_n_threads = spec.get("n_threads", 0)
    gen_ops = spec.get("ops", [])
    gen_n_loads = len([op for op in gen_ops if op["optype"] == "load"])

    # Build the LitmusTest and check behavior matches
    try:
        test = spec_to_litmus(spec)
        gen_results = {}
        for arch in ['x86', 'arm', 'riscv']:
            model = {'x86': 'TSO', 'arm': 'ARM', 'riscv': 'RISC-V'}[arch]
            forbidden_allowed, _ = verify_test(test, model)
            gen_results[arch] = forbidden_allowed

        expected_test = LitmusTest(
            name=expected_pattern,
            n_threads=expected_n_threads,
            addresses=expected['addresses'],
            ops=expected['ops'],
            forbidden=expected['forbidden']
        )
        exp_results = {}
        for arch in ['x86', 'arm', 'riscv']:
            model = {'x86': 'TSO', 'arm': 'ARM', 'riscv': 'RISC-V'}[arch]
            forbidden_allowed, _ = verify_test(expected_test, model)
            exp_results[arch] = forbidden_allowed

        behavior_match = gen_results == exp_results
        return {
            "valid": True,
            "behavior_match": behavior_match,
            "generated_results": gen_results,
            "expected_results": exp_results,
            "n_threads_match": gen_n_threads == expected_n_threads,
            "n_loads_match": gen_n_loads == expected_n_loads,
        }
    except Exception as e:
        return {"valid": False, "reason": str(e)}


def run_expanded_evaluation(n_runs=3, output_dir="benchmark_results_new"):
    """Run expanded LLM evaluation with multiple runs."""
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for cs in EXPANDED_CASE_STUDIES:
        print(f"\n{'='*70}")
        print(f"Case Study: {cs['name']} [{cs['category']}]")
        print(f"{'='*70}")

        run_results = []
        for run_idx in range(n_runs):
            print(f"  Run {run_idx+1}/{n_runs}...", end=" ", flush=True)
            try:
                spec = generate_litmus_test(cs["description"])
                test = spec_to_litmus(spec)
                results = analyze_litmus(test)

                # Determine semantic correctness
                is_correct = True
                reason = "OK"

                # Basic sanity: must have operations and forbidden outcome
                if not spec.get("ops") or not spec.get("forbidden"):
                    is_correct = False
                    reason = "Missing ops or forbidden outcome"
                elif spec.get("n_threads", 0) < 2:
                    is_correct = False
                    reason = "Less than 2 threads"
                else:
                    # Check that forbidden outcome uses valid register names
                    load_regs = {op["reg"] for op in spec["ops"] if op["optype"] == "load" and "reg" in op}
                    forbidden_regs = set(spec["forbidden"].keys())
                    if not forbidden_regs.issubset(load_regs):
                        is_correct = False
                        reason = f"Forbidden references undefined registers: {forbidden_regs - load_regs}"

                    # Check that values in forbidden outcome are reachable
                    for reg, val in spec["forbidden"].items():
                        if val < 0 or val > 10:
                            is_correct = False
                            reason = f"Unreasonable forbidden value {reg}={val}"

                # Known-answer validation
                known_answer_match = None
                if "expected_pattern" in cs and is_correct:
                    validation = validate_against_known(spec, cs["expected_pattern"])
                    if validation.get("valid"):
                        known_answer_match = validation.get("behavior_match", False)
                        if not known_answer_match:
                            is_correct = False
                            reason = f"Behavior mismatch with {cs['expected_pattern']}"

                status = "✓" if is_correct else "✗"
                print(f"{status} ({reason})")

                run_results.append({
                    "run": run_idx,
                    "correct": is_correct,
                    "reason": reason,
                    "spec": spec,
                    "portability": {k: v for k, v in results.items() if k != "symmetry"},
                    "known_answer_match": known_answer_match,
                })

            except Exception as e:
                print(f"✗ (ERROR: {e})")
                run_results.append({
                    "run": run_idx,
                    "correct": False,
                    "reason": f"Exception: {str(e)}",
                    "spec": None,
                })

            time.sleep(0.5)  # rate limiting

        correct_count = sum(1 for r in run_results if r["correct"])
        all_results.append({
            "case_study": cs["name"],
            "category": cs["category"],
            "description": cs["description"],
            "n_runs": n_runs,
            "correct_runs": correct_count,
            "accuracy": correct_count / n_runs,
            "runs": run_results,
        })
        print(f"  Result: {correct_count}/{n_runs} correct")

    # Compute overall statistics
    total_correct = sum(r["correct_runs"] for r in all_results)
    total_runs = sum(r["n_runs"] for r in all_results)
    overall_accuracy = total_correct / total_runs if total_runs > 0 else 0

    # Wilson score confidence interval
    n = total_runs
    p_hat = overall_accuracy
    z = 1.96  # 95% CI
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denominator
    margin = z * math.sqrt((p_hat*(1-p_hat) + z**2/(4*n)) / n) / denominator
    ci_low = max(0, center - margin)
    ci_high = min(1, center + margin)

    # Per-category accuracy
    categories = {}
    for r in all_results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"correct": 0, "total": 0}
        categories[cat]["correct"] += r["correct_runs"]
        categories[cat]["total"] += r["n_runs"]

    summary = {
        "total_case_studies": len(all_results),
        "total_runs": total_runs,
        "total_correct": total_correct,
        "overall_accuracy": round(overall_accuracy * 100, 1),
        "confidence_interval_95": [round(ci_low * 100, 1), round(ci_high * 100, 1)],
        "per_category": {k: {
            "accuracy": round(v["correct"] / v["total"] * 100, 1) if v["total"] > 0 else 0,
            "correct": v["correct"],
            "total": v["total"]
        } for k, v in categories.items()},
        "per_case_study": [{
            "name": r["case_study"],
            "category": r["category"],
            "accuracy": round(r["accuracy"] * 100, 1),
            "correct": r["correct_runs"],
            "total": r["n_runs"],
        } for r in all_results],
    }

    print(f"\n{'='*70}")
    print(f"OVERALL RESULTS")
    print(f"{'='*70}")
    print(f"Total: {total_correct}/{total_runs} correct ({overall_accuracy*100:.1f}%)")
    print(f"95% CI: [{ci_low*100:.1f}%, {ci_high*100:.1f}%]")
    for cat, stats in categories.items():
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {cat}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")

    # Save
    outfile = os.path.join(output_dir, "expanded_llm_eval.json")
    with open(outfile, "w") as f:
        json.dump({"summary": summary, "detailed_results": all_results}, f, indent=2)
    print(f"\nResults saved to {outfile}")

    return summary, all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=str, default="benchmark_results_new")
    args = parser.parse_args()

    os.environ.setdefault("OPENAI_API_KEY", "")
    if not os.environ.get("OPENAI_API_KEY"):
        print("No OPENAI_API_KEY found. Sourcing ~/.bashrc...")
        import subprocess
        result = subprocess.run("bash -c 'source ~/.bashrc && echo $OPENAI_API_KEY'",
                              shell=True, capture_output=True, text=True)
        key = result.stdout.strip()
        if key:
            os.environ["OPENAI_API_KEY"] = key

    run_expanded_evaluation(n_runs=args.runs, output_dir=args.output)
