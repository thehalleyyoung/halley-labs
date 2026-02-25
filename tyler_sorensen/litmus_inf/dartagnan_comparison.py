#!/usr/bin/env python3
"""
Dartagnan (Dat3M) vs LITMUS∞ — Structured Comparison.

Dartagnan (https://github.com/hernanponcedeleon/Dat3M) is a bounded model
checker for concurrent programs under weak memory models.  It is the most
directly comparable tool to LITMUS∞.  This module documents the comparison,
runs LITMUS∞ timing benchmarks, and produces a comparison table for the paper.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from portcheck import PATTERNS, ARCHITECTURES, check_portability


# ── Dartagnan overview ──────────────────────────────────────────────────

DARTAGNAN_DESCRIPTION = {
    "tool": "Dartagnan (Dat3M)",
    "url": "https://github.com/hernanponcedeleon/Dat3M",
    "description": (
        "Dartagnan is a bounded model checker for concurrent C/C++ programs "
        "under various weak memory models (TSO, ARM, POWER, RISC-V, Linux "
        "kernel, C11/RC11).  It operates on full LLVM bitcode programs via "
        "Boogie/Z3 and can verify reachability and state assertions.  It "
        "includes a portability analysis mode (Porthos) that checks whether "
        "a program that is safe under one memory model is also safe under "
        "a weaker model."
    ),
    "approach": "Full-program bounded model checking via LLVM → Boogie → SMT",
    "language": "Java (tool), C/C++ (input programs)",
    "memory_models": [
        "TSO", "PSO", "RMO", "ARM8", "Power", "RISC-V (IMM)",
        "Linux Kernel (LKMM)", "C11/RC11"
    ],
    "key_features": [
        "Full-program verification (loops, pointers, function calls)",
        "Portability analysis (Porthos) between any two models",
        "LLVM-level analysis (handles compiler optimizations)",
        "Witness generation for violations",
        "Supports bounded and unbounded verification"
    ],
    "limitations": [
        "Requires compilation to LLVM bitcode",
        "Seconds to minutes per program (SMT solver dependent)",
        "No GPU/scoped memory model support",
        "No automatic fence recommendation",
        "No pattern-based quick diagnostics"
    ]
}

LITMUS_INF_DESCRIPTION = {
    "tool": "LITMUS∞",
    "approach": "Pattern-based portability checking with exhaustive enumeration",
    "language": "Python (tool), C/C++/CUDA (input code)",
    "memory_models": [
        "x86-TSO", "SPARC-PSO", "ARMv8", "RISC-V RVWMO",
        "OpenCL (WG/Device)", "Vulkan (WG/Device)", "PTX (CTA/GPU)"
    ],
    "key_features": [
        "Sub-millisecond per-pattern analysis",
        "AST-based code → pattern matching (96.6% accuracy)",
        "Per-thread minimal fence recommendations",
        "GPU scoped synchronization and scope mismatch detection",
        "Z3 fence certificates (55 UNSAT + 40 SAT)",
        "Custom model DSL for user-defined architectures",
        "Coverage confidence warnings for unrecognized patterns"
    ],
    "limitations": [
        "Operates on 75 fixed litmus patterns, not arbitrary programs",
        "No loop/pointer/function-call analysis",
        "GPU models are a single parameterized model (6 instantiations)",
        "Pattern-level safety does not compose to program-level safety",
        "Fence costs are analytical, not hardware-measured"
    ]
}


# ── Comparison table ────────────────────────────────────────────────────

COMPARISON_TABLE = [
    {
        "dimension": "Approach",
        "dartagnan": "Full-program BMC (LLVM → Boogie → Z3)",
        "litmus_inf": "Pattern-based enumeration + AST matching"
    },
    {
        "dimension": "Input",
        "dartagnan": "C/C++ programs (compiled to LLVM bitcode)",
        "litmus_inf": "C/C++/CUDA source code snippets"
    },
    {
        "dimension": "Analysis scope",
        "dartagnan": "Whole programs with loops, pointers, functions",
        "litmus_inf": "75 litmus test patterns (fixed library)"
    },
    {
        "dimension": "Speed",
        "dartagnan": "Seconds to minutes per program",
        "litmus_inf": "<1ms per pattern, <200ms for all 750 pairs"
    },
    {
        "dimension": "CPU models",
        "dartagnan": "8+ (TSO, PSO, RMO, ARM8, Power, RISC-V, LKMM, C11)",
        "litmus_inf": "4 independent (x86-TSO, SPARC-PSO, ARMv8, RVWMO) + DSL"
    },
    {
        "dimension": "GPU support",
        "dartagnan": "No",
        "litmus_inf": "Yes — 1 parameterized model, 6 scope instantiations"
    },
    {
        "dimension": "Portability analysis",
        "dartagnan": "Yes (Porthos: reachability under weaker model)",
        "litmus_inf": "Yes (built-in cross-model portability matrix)"
    },
    {
        "dimension": "Fence recommendations",
        "dartagnan": "No (reports violations, not fixes)",
        "litmus_inf": "Yes — per-thread minimal fence insertions"
    },
    {
        "dimension": "Formal guarantees",
        "dartagnan": "Bounded verification (sound for bound k)",
        "litmus_inf": "Exhaustive for finite litmus tests (RF×CO completeness)"
    },
    {
        "dimension": "Machine-checked proofs",
        "dartagnan": "Verification witnesses",
        "litmus_inf": "95 Z3 fence certificates (55 UNSAT + 40 SAT)"
    },
    {
        "dimension": "CI/CD integration",
        "dartagnan": "Possible but slow (seconds per file)",
        "litmus_inf": "Built-in GitHub Action, pip install, <1s per file"
    },
    {
        "dimension": "Best suited for",
        "dartagnan": "Deep verification of specific concurrent algorithms",
        "litmus_inf": "Fast CI/CD portability screening across architectures"
    },
]


def run_litmus_inf_timing():
    """Run LITMUS∞ on its full benchmark and record timing."""
    results = {
        "tool": "LITMUS∞",
        "benchmark": "75 patterns × 10 configurations = 750 pairs",
        "timings": {},
        "total_ms": 0,
        "per_pair_ms": 0,
    }

    # Time full portability analysis
    start = time.perf_counter()
    total_pairs = 0
    unsafe_count = 0

    for pat_name in sorted(PATTERNS.keys()):
        pat_start = time.perf_counter()
        for arch_name in ARCHITECTURES:
            r = check_portability(pat_name, target_arch=arch_name)
            total_pairs += 1
            if r and not r[0].safe:
                unsafe_count += 1
        pat_ms = (time.perf_counter() - pat_start) * 1000
        results["timings"][pat_name] = round(pat_ms, 2)

    total_ms = (time.perf_counter() - start) * 1000
    results["total_ms"] = round(total_ms, 1)
    results["per_pair_ms"] = round(total_ms / total_pairs, 3) if total_pairs else 0
    results["total_pairs"] = total_pairs
    results["unsafe_pairs"] = unsafe_count
    results["safe_pairs"] = total_pairs - unsafe_count

    return results


def generate_comparison_report():
    """Generate full comparison report with timing data."""
    print("Running LITMUS∞ timing benchmark...")
    timing = run_litmus_inf_timing()

    report = {
        "comparison": {
            "dartagnan": DARTAGNAN_DESCRIPTION,
            "litmus_inf": LITMUS_INF_DESCRIPTION,
            "table": COMPARISON_TABLE,
        },
        "litmus_inf_timing": timing,
        "key_differentiators": [
            {
                "advantage": "LITMUS∞",
                "dimension": "Speed",
                "detail": (
                    f"LITMUS∞ analyzes all {timing['total_pairs']} pairs in "
                    f"{timing['total_ms']:.0f}ms ({timing['per_pair_ms']:.3f}ms/pair). "
                    f"Dartagnan typically takes seconds to minutes per program."
                )
            },
            {
                "advantage": "LITMUS∞",
                "dimension": "GPU support",
                "detail": (
                    "LITMUS∞ supports GPU scoped synchronization "
                    "(OpenCL, Vulkan, PTX) with scope mismatch detection. "
                    "Dartagnan does not model GPU memory models."
                )
            },
            {
                "advantage": "LITMUS∞",
                "dimension": "Fence recommendations",
                "detail": (
                    "LITMUS∞ provides per-thread minimal fence insertions. "
                    "Dartagnan reports violations but not fixes."
                )
            },
            {
                "advantage": "Dartagnan",
                "dimension": "Program scope",
                "detail": (
                    "Dartagnan verifies full programs with loops, pointers, "
                    "and function calls. LITMUS∞ operates on 75 fixed patterns."
                )
            },
            {
                "advantage": "Dartagnan",
                "dimension": "Model coverage",
                "detail": (
                    "Dartagnan supports 8+ CPU memory models including LKMM "
                    "and C11/RC11. LITMUS∞ has 4 built-in CPU models + DSL."
                )
            },
        ],
        "complementary_usage": (
            "LITMUS∞ and Dartagnan are complementary tools. Use LITMUS∞ as a "
            "fast first-pass CI/CD filter (<1ms/pattern) to flag potential "
            "portability issues, then use Dartagnan for deep verification of "
            "flagged code sections. LITMUS∞ exports .litmus files compatible "
            "with Dartagnan's input format for seamless handoff."
        ),
    }

    return report


def generate_latex_table():
    """Generate LaTeX comparison table for the paper."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{LITMUS$\infty$ vs.\ Dartagnan: complementary approaches to memory model portability.}",
        r"\label{tab:dartagnan-comparison}",
        r"\small",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"\textbf{Dimension} & \textbf{Dartagnan} & \textbf{LITMUS$\infty$} \\",
        r"\midrule",
    ]

    key_rows = [
        ("Approach", "Full-program BMC", "Pattern enumeration"),
        ("Speed", "Seconds--minutes", "$<$1\\,ms/pattern"),
        ("Input scope", "Whole programs", "75 litmus patterns"),
        ("CPU models", "8+ (incl.\\ LKMM, C11)", "4 + custom DSL"),
        ("GPU support", "No", "Yes (scoped sync.)"),
        ("Fence recs.", "No", "Per-thread minimal"),
        ("CI/CD ready", "Slow", "Built-in Action"),
        ("Guarantees", "Bounded sound", r"RF$\times$CO complete"),
    ]

    for dim, dart, litmus in key_rows:
        lines.append(f"{dim} & {dart} & {litmus} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    report = generate_comparison_report()

    # Save results
    outdir = os.path.join(os.path.dirname(__file__), "paper_results_v4")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "dartagnan_comparison.json")
    with open(outpath, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print("LITMUS∞ vs Dartagnan Comparison")
    print(f"{'='*60}")

    t = report["litmus_inf_timing"]
    print(f"\nLITMUS∞ timing:")
    print(f"  Total pairs: {t['total_pairs']}")
    print(f"  Total time:  {t['total_ms']:.0f}ms")
    print(f"  Per pair:    {t['per_pair_ms']:.3f}ms")
    print(f"  Unsafe:      {t['unsafe_pairs']}")
    print(f"  Safe:        {t['safe_pairs']}")

    print(f"\nComparison table ({len(COMPARISON_TABLE)} dimensions):")
    for row in COMPARISON_TABLE:
        print(f"  {row['dimension']:25s} | Dart: {row['dartagnan'][:35]:35s} | LIT: {row['litmus_inf'][:35]}")

    print(f"\nKey differentiators:")
    for d in report["key_differentiators"]:
        print(f"  [{d['advantage']}] {d['dimension']}: {d['detail'][:80]}...")

    print(f"\nLaTeX table:")
    print(generate_latex_table())

    print(f"\nResults saved to {outpath}")
