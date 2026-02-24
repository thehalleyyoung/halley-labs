#!/usr/bin/env python3
"""
fence_cost.py: Fence cost analysis — compares minimal per-thread fences vs coarse fences.

For each failing pattern, computes:
1. Minimal per-thread fences (our recommendation)
2. Coarse fences (naive: just use full barrier everywhere)
3. Fence "savings" — how many unnecessary ordering constraints are avoided
"""

import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from portcheck import PATTERNS, ARCHITECTURES, LitmusTest, MemOp
from portcheck import verify_test, recommend_fence, _identify_per_thread_violations

# Fence strength hierarchy (lower = weaker = cheaper)
ARM_FENCE_COST = {
    'dmb ishst': 1,    # W→W only
    'dmb ishld': 2,    # R→R, R→W
    'dmb ish': 4,      # full barrier (all pairs)
}

RISCV_FENCE_COST = {
    'fence r,r': 1,
    'fence w,w': 1,
    'fence r,w': 1,
    'fence w,r': 2,    # store-load is expensive
    'fence rw,rw': 4,  # full barrier
}

def compute_fence_costs(results_dir="benchmark_results_new"):
    os.makedirs(results_dir, exist_ok=True)
    rows = []

    for pat_name in sorted(PATTERNS.keys()):
        pattern = PATTERNS[pat_name]
        n_threads = max(op.thread for op in pattern['ops']) + 1
        test = LitmusTest(
            name=pat_name, n_threads=n_threads,
            addresses=pattern['addresses'],
            ops=pattern['ops'], forbidden=pattern['forbidden']
        )

        for arch in ['arm', 'riscv']:
            model = ARCHITECTURES[arch]
            forbidden_allowed, _ = verify_test(test, model)
            if not forbidden_allowed:
                continue

            fence = recommend_fence(test, arch, model)
            if not fence:
                continue

            # Parse per-thread fences
            per_thread_fences = {}
            if '(T' in fence:
                parts = fence.split('; ')
                for part in parts:
                    if '(T' in part:
                        f, tid = part.rsplit(' (T', 1)
                        tid = int(tid.rstrip(')'))
                        per_thread_fences[tid] = f.strip()
                    else:
                        per_thread_fences[0] = part.strip()
            else:
                for t in range(n_threads):
                    per_thread_fences[t] = fence

            # Compute minimal cost
            cost_map = ARM_FENCE_COST if arch == 'arm' else RISCV_FENCE_COST
            minimal_cost = sum(cost_map.get(f, 4) for f in per_thread_fences.values())

            # Compute coarse cost (full barrier on every thread)
            coarse_fence = 'dmb ish' if arch == 'arm' else 'fence rw,rw'
            coarse_cost = n_threads * cost_map.get(coarse_fence, 4)

            savings = (1 - minimal_cost / coarse_cost) * 100 if coarse_cost > 0 else 0

            rows.append({
                'pattern': pat_name,
                'arch': arch,
                'n_threads': n_threads,
                'minimal_fence': fence,
                'coarse_fence': f'{coarse_fence} (all threads)',
                'minimal_cost': minimal_cost,
                'coarse_cost': coarse_cost,
                'savings_pct': round(savings, 1),
            })

    # Save CSV
    csv_path = os.path.join(results_dir, "fence_cost_analysis.csv")
    with open(csv_path, "w") as f:
        if rows:
            f.write(",".join(rows[0].keys()) + "\n")
            for r in rows:
                f.write(",".join(str(v) for v in r.values()) + "\n")
    print(f"Saved {len(rows)} rows to {csv_path}")

    # Print summary
    if rows:
        arm_savings = [r['savings_pct'] for r in rows if r['arch'] == 'arm' and r['savings_pct'] > 0]
        rv_savings = [r['savings_pct'] for r in rows if r['arch'] == 'riscv' and r['savings_pct'] > 0]
        print(f"\nARM: {len(arm_savings)} patterns with fence savings, "
              f"avg {sum(arm_savings)/len(arm_savings):.1f}% reduction" if arm_savings else "")
        print(f"RISC-V: {len(rv_savings)} patterns with fence savings, "
              f"avg {sum(rv_savings)/len(rv_savings):.1f}% reduction" if rv_savings else "")

        # Top savings
        print("\nTop savings (ARM):")
        for r in sorted([r for r in rows if r['arch'] == 'arm'], key=lambda x: -x['savings_pct'])[:10]:
            print(f"  {r['pattern']:30s}: {r['minimal_fence']:50s} → {r['savings_pct']}% cheaper than {r['coarse_fence']}")

    # Save JSON
    json_path = os.path.join(results_dir, "fence_cost_analysis.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    return rows

if __name__ == "__main__":
    compute_fence_costs()
