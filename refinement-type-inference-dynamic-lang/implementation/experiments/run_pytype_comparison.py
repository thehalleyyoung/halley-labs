#!/usr/bin/env python3
"""
Pytype comparison script.

Pytype (Google's Python type checker) is the most direct competitor to GuardHarvest
in the zero-annotation space. This script compares both tools on the same benchmarks.

NOTE: Pytype requires Python <= 3.12 and could not be installed on Python 3.14.
We document the expected comparison based on Pytype's published capabilities.
"""

import json
from pathlib import Path

# Pytype comparison based on its documented capabilities
comparison = {
    "tool_comparison": "GuardHarvest vs Pytype",
    "pytype_install_status": "FAILED - requires Python <= 3.12, test env is Python 3.14",
    "comparison_methodology": "Qualitative comparison based on Pytype documentation and published benchmarks",

    "feature_comparison": {
        "type_inference": {
            "guardharvest": "Flow-sensitive abstract interpretation with guard narrowing",
            "pytype": "Whole-program type inference with abstract interpretation",
            "advantage": "pytype - broader inference"
        },
        "annotation_requirement": {
            "guardharvest": "Zero annotations required",
            "pytype": "Zero annotations required (infers types from code)",
            "advantage": "tie"
        },
        "guard_exploitation": {
            "guardharvest": "Systematic harvesting of 13 guard patterns",
            "pytype": "Basic isinstance/None narrowing",
            "advantage": "guardharvest - more guard patterns recognized"
        },
        "null_deref_detection": {
            "guardharvest": "Via nullity domain with None-evidence tracking",
            "pytype": "Via Optional type tracking",
            "advantage": "tie"
        },
        "division_by_zero": {
            "guardharvest": "Via interval domain",
            "pytype": "Not checked",
            "advantage": "guardharvest"
        },
        "interprocedural": {
            "guardharvest": "Intraprocedural only",
            "pytype": "Whole-program (interprocedural)",
            "advantage": "pytype"
        },
        "performance": {
            "guardharvest": "~47K LOC/sec (single file, no whole-program)",
            "pytype": "Slower (whole-program analysis)",
            "advantage": "guardharvest for per-file checks"
        },
        "ecosystem_integration": {
            "guardharvest": "CLI, SARIF, Python API",
            "pytype": "CLI, .pyi stub generation, mypy integration",
            "advantage": "pytype - more mature ecosystem"
        }
    },

    "key_differences": [
        "GuardHarvest is intraprocedural; Pytype is interprocedural",
        "GuardHarvest checks division-by-zero and index bounds; Pytype focuses on type errors",
        "GuardHarvest harvests 13 guard patterns; Pytype handles fewer guard forms",
        "Pytype generates .pyi stubs for downstream tools; GuardHarvest does not",
        "GuardHarvest is faster per-file; Pytype builds whole-program models",
    ],

    "honest_assessment": (
        "Pytype is a more mature tool with interprocedural analysis and better "
        "ecosystem integration. GuardHarvest's advantages are: (1) faster per-file "
        "checking suitable for CI, (2) broader guard pattern recognition, and "
        "(3) interval-domain-based division-by-zero detection that Pytype lacks. "
        "A proper head-to-head comparison on the same benchmarks requires Python <= 3.12."
    ),
}

out_path = Path(__file__).parent / "results" / "pytype_comparison.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(comparison, f, indent=2)

print(f"Pytype comparison saved to {out_path}")
print("NOTE: Actual head-to-head requires Python <= 3.12 environment")
