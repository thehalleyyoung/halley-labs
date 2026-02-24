#!/usr/bin/env python3
"""
Comprehensive experiment runner for LITMUS∞ tool paper.
Generates all benchmark data used in the paper from portcheck.py.
"""

import csv
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict

# ── Setup ────────────────────────────────────────────────────────────

RESULTS_DIR = "paper_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

from portcheck import (
    ARCHITECTURES, PATTERNS, MemOp, LitmusTest, PortabilityResult,
    check_portability,
    compute_joint_automorphisms, compute_orbits,
    HERD7_EXPECTED, HARDWARE_VALIDATION,
)

def check_all_targets(pname):
    """Check portability of a pattern across all architectures."""
    return check_portability(pname)

# ── Experiment 1: Full Portability Matrix (570 pairs) ────────────────

def run_portability_matrix():
    """Run all 57 patterns × 10 architectures. Generates Table 1 data."""
    print("\n" + "="*80)
    print("EXPERIMENT 1: Full Portability Matrix (57 × 10 = 570 pairs)")
    print("="*80)

    arch_names = list(ARCHITECTURES.keys())
    matrix = {}
    safe_count = 0
    fail_count = 0

    rows = []
    for pname in PATTERNS:
        results = check_all_targets(pname)
        row = {"pattern": pname}
        for r in results:
            status = "Safe" if r.safe else "Allowed"
            row[r.target_arch] = status
            if r.safe:
                safe_count += 1
            else:
                fail_count += 1
        rows.append(row)
        matrix[pname] = row

    # Save full matrix
    with open(f"{RESULTS_DIR}/portability_matrix.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pattern"] + arch_names)
        writer.writeheader()
        writer.writerows(rows)

    # Save JSON
    with open(f"{RESULTS_DIR}/portability_matrix.json", "w") as f:
        json.dump({"matrix": rows, "safe": safe_count, "fail": fail_count,
                    "total": safe_count + fail_count}, f, indent=2)

    print(f"  Total: {safe_count + fail_count} pairs")
    print(f"  Safe: {safe_count}, Allowed/Fail: {fail_count}")
    return matrix

# ── Experiment 2: Model Boundary Analysis ────────────────────────────

def run_boundary_analysis(matrix):
    """Find patterns that discriminate between adjacent memory models."""
    print("\n" + "="*80)
    print("EXPERIMENT 2: Model Boundary Analysis")
    print("="*80)

    # Define model pairs by increasing permissiveness
    model_pairs = [
        ("x86", "sparc", "TSO→PSO"),
        ("sparc", "arm", "PSO→ARM"),
        ("arm", "riscv", "ARM→RISC-V"),
    ]
    # GPU scope boundaries
    gpu_pairs = [
        ("opencl_dev", "opencl_wg", "GPU-Dev→GPU-WG"),
        ("vulkan_dev", "vulkan_wg", "GPU-Dev→GPU-WG (Vulkan)"),
        ("ptx_gpu", "ptx_cta", "GPU-Dev→GPU-CTA"),
    ]

    boundaries = []
    for stronger, weaker, label in model_pairs + gpu_pairs:
        discriminators = []
        for pname, row in matrix.items():
            s_safe = row.get(stronger) == "Safe"
            w_safe = row.get(weaker) == "Safe"
            if s_safe and not w_safe:
                discriminators.append(pname)
        boundaries.append({
            "stronger": stronger,
            "weaker": weaker,
            "label": label,
            "discriminating_patterns": discriminators,
            "count": len(discriminators),
        })
        print(f"  {label}: {len(discriminators)} discriminating patterns")
        if discriminators:
            for d in discriminators[:5]:
                print(f"    - {d}")
            if len(discriminators) > 5:
                print(f"    ... and {len(discriminators)-5} more")

    with open(f"{RESULTS_DIR}/boundary_analysis.json", "w") as f:
        json.dump(boundaries, f, indent=2)

    # CSV summary
    with open(f"{RESULTS_DIR}/boundary_analysis.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stronger", "weaker", "label", "count", "examples"])
        writer.writeheader()
        for b in boundaries:
            writer.writerow({
                "stronger": b["stronger"], "weaker": b["weaker"],
                "label": b["label"], "count": b["count"],
                "examples": "; ".join(b["discriminating_patterns"][:5])
            })

    return boundaries

# ── Experiment 3: GPU Scope Mismatch Detection ───────────────────────

def run_gpu_scope_analysis(matrix):
    """Detect GPU scope mismatch bugs — patterns safe on CPU but failing on GPU."""
    print("\n" + "="*80)
    print("EXPERIMENT 3: GPU Scope Mismatch Detection")
    print("="*80)

    cpu_archs = ["x86", "sparc", "arm", "riscv"]
    gpu_wg_archs = ["opencl_wg", "vulkan_wg", "ptx_cta"]
    gpu_dev_archs = ["opencl_dev", "vulkan_dev", "ptx_gpu"]

    scope_bugs = []
    for pname, row in matrix.items():
        cpu_safe = all(row.get(a) == "Safe" for a in cpu_archs)
        gpu_wg_safe = all(row.get(a) == "Safe" for a in gpu_wg_archs)
        gpu_dev_safe = all(row.get(a) == "Safe" for a in gpu_dev_archs)

        if cpu_safe and not gpu_wg_safe:
            bug_type = "scope_mismatch_all_gpu" if not gpu_dev_safe else "scope_mismatch_wg_only"
            scope_bugs.append({
                "pattern": pname,
                "bug_type": bug_type,
                "cpu_safe": True,
                "gpu_wg_safe": gpu_wg_safe,
                "gpu_dev_safe": gpu_dev_safe,
                "severity": "critical" if not gpu_dev_safe else "warning",
            })

        # Also detect: safe on dev-scope but failing on wg-scope
        if gpu_dev_safe and not gpu_wg_safe:
            found = any(b["pattern"] == pname for b in scope_bugs)
            if not found:
                scope_bugs.append({
                    "pattern": pname,
                    "bug_type": "scope_level_discrimination",
                    "cpu_safe": cpu_safe,
                    "gpu_wg_safe": False,
                    "gpu_dev_safe": True,
                    "severity": "warning",
                })

    print(f"  Total GPU scope bugs detected: {len(scope_bugs)}")
    for b in scope_bugs:
        print(f"    {b['pattern']}: {b['bug_type']} (severity={b['severity']})")

    with open(f"{RESULTS_DIR}/gpu_scope_bugs.json", "w") as f:
        json.dump(scope_bugs, f, indent=2)

    with open(f"{RESULTS_DIR}/gpu_scope_bugs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pattern", "bug_type", "cpu_safe",
                                                "gpu_wg_safe", "gpu_dev_safe", "severity"])
        writer.writeheader()
        writer.writerows(scope_bugs)

    return scope_bugs

# ── Experiment 4: Per-Thread Fence Cost Analysis ─────────────────────

def run_fence_cost_analysis():
    """Compute per-thread minimal fences and cost savings."""
    print("\n" + "="*80)
    print("EXPERIMENT 4: Per-Thread Fence Cost Analysis")
    print("="*80)

    from fence_cost import compute_fence_costs
    results = compute_fence_costs()

    # Also compute directly
    arm_savings = [r["savings_pct"] for r in results if r["arch"] == "arm" and r["savings_pct"] > 0]
    riscv_savings = [r["savings_pct"] for r in results if r["arch"] == "riscv" and r["savings_pct"] > 0]

    print(f"  ARM: {len(arm_savings)} patterns with savings, avg {statistics.mean(arm_savings):.1f}%")
    print(f"  RISC-V: {len(riscv_savings)} patterns with savings, avg {statistics.mean(riscv_savings):.1f}%")
    if arm_savings:
        print(f"    ARM range: {min(arm_savings):.1f}% - {max(arm_savings):.1f}%")
    if riscv_savings:
        print(f"    RISC-V range: {min(riscv_savings):.1f}% - {max(riscv_savings):.1f}%")

    with open(f"{RESULTS_DIR}/fence_cost_analysis.json", "w") as f:
        json.dump({
            "results": results,
            "arm_avg": statistics.mean(arm_savings) if arm_savings else 0,
            "riscv_avg": statistics.mean(riscv_savings) if riscv_savings else 0,
            "arm_count": len(arm_savings),
            "riscv_count": len(riscv_savings),
        }, f, indent=2)

    with open(f"{RESULTS_DIR}/fence_cost_analysis.csv", "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    return results

# ── Experiment 5: herd7 Validation ───────────────────────────────────

def run_herd7_validation():
    """Validate against herd7 expected results."""
    print("\n" + "="*80)
    print("EXPERIMENT 5: herd7 Validation")
    print("="*80)

    from portcheck import HERD7_EXPECTED

    agreements = 0
    disagreements = 0
    results = []

    for (pname, arch), allowed in sorted(HERD7_EXPECTED.items()):
        if pname not in PATTERNS:
            continue

        port_results = check_portability(pname, target_arch=arch)
        if not port_results:
            continue
        portability = port_results[0]
        # allowed=True means forbidden outcome CAN happen (not safe)
        # allowed=False means forbidden outcome CANNOT happen (safe)
        our_safe = portability.safe
        herd7_safe = not allowed

        agree = our_safe == herd7_safe
        if agree:
            agreements += 1
        else:
            disagreements += 1

        results.append({
            "pattern": pname, "arch": arch,
            "our_safe": our_safe, "herd7_safe": herd7_safe,
            "agree": agree,
        })

    total = agreements + disagreements
    print(f"  Agreement: {agreements}/{total} ({100*agreements/total:.1f}%)" if total else "  No comparisons")
    if disagreements:
        print(f"  DISAGREEMENTS: {disagreements}")
        for r in results:
            if not r["agree"]:
                print(f"    {r['pattern']} on {r['arch']}: ours={'Safe' if r['our_safe'] else 'Allowed'}, herd7={'Safe' if r['herd7_safe'] else 'Allowed'}")

    with open(f"{RESULTS_DIR}/herd7_validation.json", "w") as f:
        json.dump({"results": results, "agreements": agreements,
                    "total": total, "pct": 100*agreements/total if total else 0}, f, indent=2)

    with open(f"{RESULTS_DIR}/herd7_validation.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pattern", "arch",
                                                "our_safe", "herd7_safe", "agree"])
        writer.writeheader()
        writer.writerows(results)

    return results

# ── Experiment 6: Hardware Validation ────────────────────────────────

def run_hardware_validation():
    """Validate against published hardware observations."""
    print("\n" + "="*80)
    print("EXPERIMENT 6: Hardware Observation Validation")
    print("="*80)

    results = []
    agreements = 0
    for (pname, arch), obs in sorted(HARDWARE_VALIDATION.items()):
        hw_observed = obs["observed"]  # True if forbidden outcome was seen on HW
        reference = obs["source"]

        port_results = check_portability(pname, target_arch=arch)
        if not port_results:
            continue
        portability = port_results[0]
        # If hw_observed (bug was seen), our tool should say "Allowed" (not safe)
        # If not hw_observed, our tool should say "Safe"
        our_safe = portability.safe
        expected_safe = not hw_observed
        agree = our_safe == expected_safe

        if agree:
            agreements += 1

        results.append({
            "pattern": pname, "arch": arch,
            "hw_observed": hw_observed,
            "our_safe": our_safe,
            "agree": agree,
            "reference": reference,
        })

    total = len(results)
    print(f"  Consistency: {agreements}/{total} ({100*agreements/total:.1f}%)")

    with open(f"{RESULTS_DIR}/hardware_validation.json", "w") as f:
        json.dump({"results": results, "agreements": agreements,
                    "total": total}, f, indent=2)

    with open(f"{RESULTS_DIR}/hardware_validation.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pattern", "arch", "hw_observed",
                                                "our_safe", "agree", "reference"])
        writer.writeheader()
        writer.writerows(results)

    return results

# ── Experiment 7: Performance Benchmarks ─────────────────────────────

def run_performance_benchmarks():
    """Measure analysis time for various workloads."""
    print("\n" + "="*80)
    print("EXPERIMENT 7: Performance Benchmarks")
    print("="*80)

    # Single pattern timing
    single_times = {}
    for pname in list(PATTERNS.keys())[:10]:
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            check_all_targets(pname)
            times.append((time.perf_counter() - t0) * 1000)
        single_times[pname] = {
            "mean_ms": statistics.mean(times),
            "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0,
            "min_ms": min(times),
            "max_ms": max(times),
        }
        print(f"  {pname}: {statistics.mean(times):.3f}ms per 10-arch check")

    # Full 570-pair timing
    full_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        for p in PATTERNS:
            check_all_targets(p)
        full_times.append((time.perf_counter() - t0) * 1000)

    full_stats = {
        "mean_ms": statistics.mean(full_times),
        "stdev_ms": statistics.stdev(full_times),
        "min_ms": min(full_times),
        "max_ms": max(full_times),
        "n_runs": len(full_times),
    }
    print(f"  Full 570 pairs: {full_stats['mean_ms']:.1f}ms ± {full_stats['stdev_ms']:.1f}ms")

    # Automorphism computation timing
    auto_times = {}
    for pname in list(PATTERNS.keys())[:10]:
        p = PATTERNS[pname]
        n_threads = max(op.thread for op in p["ops"]) + 1
        test = LitmusTest(name=pname, n_threads=n_threads,
                          addresses=p["addresses"], ops=p["ops"],
                          forbidden=p["forbidden"])
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            autos = compute_joint_automorphisms(test)
            compute_orbits(test, autos)
            times.append((time.perf_counter() - t0) * 1000)
        auto_times[pname] = {
            "mean_ms": statistics.mean(times),
            "n_automorphisms": len(autos),
        }

    with open(f"{RESULTS_DIR}/performance_benchmarks.json", "w") as f:
        json.dump({
            "single_pattern": single_times,
            "full_analysis": full_stats,
            "automorphism": auto_times,
        }, f, indent=2)

    return full_stats

# ── Experiment 8: Symmetry and Compression ───────────────────────────

def run_symmetry_analysis():
    """Compute automorphism groups and compression ratios."""
    print("\n" + "="*80)
    print("EXPERIMENT 8: Symmetry / Compression Analysis")
    print("="*80)

    results = []
    for pname in PATTERNS:
        p = PATTERNS[pname]
        n_threads = max(op.thread for op in p["ops"]) + 1
        test = LitmusTest(name=pname, n_threads=n_threads,
                          addresses=p["addresses"], ops=p["ops"],
                          forbidden=p["forbidden"])
        autos = compute_joint_automorphisms(test)
        total, n_orbits = compute_orbits(test, autos)
        ratio = total / n_orbits if n_orbits > 0 else 1

        results.append({
            "pattern": pname,
            "n_threads": n_threads,
            "n_addresses": len(p["addresses"]),
            "n_loads": len(test.loads),
            "total_outcomes": total,
            "orbits": n_orbits,
            "compression_ratio": round(ratio, 2),
            "automorphism_group_size": len(autos),
        })

    # Sort by compression ratio
    results.sort(key=lambda x: x["compression_ratio"], reverse=True)

    print(f"  Patterns analyzed: {len(results)}")
    nontrivial = [r for r in results if r["compression_ratio"] > 1]
    print(f"  Patterns with nontrivial symmetry: {len(nontrivial)}")
    for r in nontrivial[:5]:
        print(f"    {r['pattern']}: {r['total_outcomes']}→{r['orbits']} "
              f"(ratio={r['compression_ratio']:.1f}x, |Aut|={r['automorphism_group_size']})")

    with open(f"{RESULTS_DIR}/symmetry_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(f"{RESULTS_DIR}/symmetry_analysis.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pattern", "n_threads", "n_addresses",
                                                "n_loads", "total_outcomes", "orbits",
                                                "compression_ratio", "automorphism_group_size"])
        writer.writeheader()
        writer.writerows(results)

    return results

# ── Experiment 9: RISC-V Asymmetric Fence Analysis ──────────────────

def run_riscv_fence_analysis():
    """Analyze RISC-V asymmetric fence discrimination."""
    print("\n" + "="*80)
    print("EXPERIMENT 9: RISC-V Asymmetric Fence Analysis")
    print("="*80)

    asym_patterns = []
    for pname, p in PATTERNS.items():
        has_asym = any(getattr(op, 'fence_pred', None) for op in p["ops"])
        if has_asym:
            arm_result = check_portability(pname, target_arch="arm")[0]
            riscv_result = check_portability(pname, target_arch="riscv")[0]

            # Find the asymmetric fence details
            fence_details = []
            for op in p["ops"]:
                if op.optype == 'fence' and getattr(op, 'fence_pred', None):
                    fence_details.append({
                        "thread": op.thread,
                        "pred": op.fence_pred,
                        "succ": op.fence_succ,
                    })

            asym_patterns.append({
                "pattern": pname,
                "arm_safe": arm_result.safe,
                "riscv_safe": riscv_result.safe,
                "discriminates_arm_riscv": arm_result.safe != riscv_result.safe,
                "fence_details": fence_details,
            })

    print(f"  Asymmetric fence patterns: {len(asym_patterns)}")
    disc = [p for p in asym_patterns if p["discriminates_arm_riscv"]]
    print(f"  ARM/RISC-V discriminators: {len(disc)}")
    for d in disc:
        print(f"    {d['pattern']}: ARM={'Safe' if d['arm_safe'] else 'FAIL'}, "
              f"RISC-V={'Safe' if d['riscv_safe'] else 'FAIL'}")
        for fd in d["fence_details"]:
            print(f"      fence pred={fd['pred']}, succ={fd['succ']} on T{fd['thread']}")

    with open(f"{RESULTS_DIR}/riscv_asymmetric_fence.json", "w") as f:
        json.dump(asym_patterns, f, indent=2)

    return asym_patterns

# ── Experiment 10: Dependency Analysis ───────────────────────────────

def run_dependency_analysis():
    """Analyze how data/address/control dependencies affect safety."""
    print("\n" + "="*80)
    print("EXPERIMENT 10: Dependency Analysis")
    print("="*80)

    dep_patterns = []
    for pname, p in PATTERNS.items():
        deps = [op for op in p["ops"] if getattr(op, 'dep_on', None)]
        if deps:
            results = {}
            for arch in ARCHITECTURES:
                r = check_portability(pname, target_arch=arch)[0]
                results[arch] = "Safe" if r.safe else "Allowed"

            dep_types = list(set(op.dep_on for op in deps))
            dep_patterns.append({
                "pattern": pname,
                "dependency_types": dep_types,
                "n_deps": len(deps),
                "results": results,
            })

    print(f"  Patterns with dependencies: {len(dep_patterns)}")
    for d in dep_patterns:
        cpu_safe = all(d["results"][a] == "Safe" for a in ["x86", "sparc", "arm", "riscv"])
        print(f"    {d['pattern']}: deps={d['dependency_types']}, CPU-safe={cpu_safe}")

    with open(f"{RESULTS_DIR}/dependency_analysis.json", "w") as f:
        json.dump(dep_patterns, f, indent=2)

    return dep_patterns

# ── Experiment 11: Pattern Category Breakdown ────────────────────────

def run_category_analysis():
    """Categorize patterns and analyze per-category safety rates."""
    print("\n" + "="*80)
    print("EXPERIMENT 11: Pattern Category Breakdown")
    print("="*80)

    categories = {
        "basic_ordering": ["mp", "sb", "lb", "iriw", "wrc", "rwc", "2+2w", "s", "r"],
        "fenced": [p for p in PATTERNS if "fence" in p and "scope" not in p and "barrier" not in p],
        "coherence": [p for p in PATTERNS if p.startswith("cor") or p.startswith("cow")],
        "gpu_scope": [p for p in PATTERNS if p.startswith("gpu_")],
        "dependency": [p for p in PATTERNS if any(getattr(op, 'dep_on', None) for op in PATTERNS[p]["ops"])],
        "asymmetric_fence": [p for p in PATTERNS if any(getattr(op, 'fence_pred', None) for op in PATTERNS[p]["ops"])],
        "multi_thread": [p for p in PATTERNS if (max(op.thread for op in PATTERNS[p]["ops"]) + 1) >= 3],
        "mutex": [p for p in PATTERNS if p in ("dekker", "peterson")],
    }

    cat_stats = []
    for cat, members in categories.items():
        total_pairs = len(members) * len(ARCHITECTURES)
        safe = 0
        for pname in members:
            for arch in ARCHITECTURES:
                r = check_portability(pname, target_arch=arch)[0]
                if r.safe:
                    safe += 1
        rate = 100 * safe / total_pairs if total_pairs else 0
        cat_stats.append({
            "category": cat,
            "n_patterns": len(members),
            "total_pairs": total_pairs,
            "safe_pairs": safe,
            "safety_rate_pct": round(rate, 1),
        })
        print(f"  {cat}: {len(members)} patterns, {safe}/{total_pairs} safe ({rate:.1f}%)")

    with open(f"{RESULTS_DIR}/category_analysis.json", "w") as f:
        json.dump(cat_stats, f, indent=2)

    with open(f"{RESULTS_DIR}/category_analysis.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "n_patterns", "total_pairs",
                                                "safe_pairs", "safety_rate_pct"])
        writer.writeheader()
        writer.writerows(cat_stats)

    return cat_stats

# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("LITMUS∞ — Comprehensive Experiment Suite")
    print("=" * 80)
    start = time.time()

    matrix = run_portability_matrix()
    run_boundary_analysis(matrix)
    run_gpu_scope_analysis(matrix)
    run_fence_cost_analysis()
    run_herd7_validation()
    run_hardware_validation()
    run_performance_benchmarks()
    run_symmetry_analysis()
    run_riscv_fence_analysis()
    run_dependency_analysis()
    run_category_analysis()

    elapsed = time.time() - start
    print(f"\n{'='*80}")
    print(f"All experiments complete in {elapsed:.1f}s")
    print(f"Results saved to {RESULTS_DIR}/")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
