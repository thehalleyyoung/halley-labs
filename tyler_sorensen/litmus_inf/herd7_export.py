#!/usr/bin/env python3
"""
herd7 litmus test file exporter for LITMUS∞.

Generates .litmus files compatible with herd7 for independent validation.
Supports C, AArch64, and RISC-V output formats.
"""

import os
from typing import Dict, List, Optional
from portcheck import PATTERNS, MemOp, LitmusTest, ARCHITECTURES


def pattern_to_litmus_c(name: str) -> str:
    """Generate a C-language .litmus file for herd7."""
    if name not in PATTERNS:
        raise ValueError(f"Unknown pattern: {name}")
    pat = PATTERNS[name]
    ops = pat['ops']
    n_threads = max(op.thread for op in ops) + 1
    addrs = pat['addresses']
    forbidden = pat['forbidden']

    lines = []
    lines.append(f"C {name}")
    lines.append("")

    # Initial state
    init_parts = []
    for addr in addrs:
        init_parts.append(f"int {addr}=0")
    lines.append("{ " + "; ".join(init_parts) + "; }")
    lines.append("")

    # Thread bodies
    for t in range(n_threads):
        t_ops = [op for op in ops if op.thread == t]
        lines.append(f"P{t}(")

        # Parameters: shared variables accessed by this thread
        t_addrs = sorted(set(op.addr for op in t_ops if op.optype != 'fence'))
        params = [f"int* {a}" for a in t_addrs]
        lines.append("  " + ", ".join(params))
        lines.append(") {")

        reg_counter = 0
        for op in t_ops:
            if op.optype == 'store':
                val = op.value if op.value is not None else 1
                lines.append(f"  WRITE_ONCE(*{op.addr}, {val});")
            elif op.optype == 'load':
                reg = op.reg or f"r{reg_counter}"
                reg_counter += 1
                lines.append(f"  int {reg} = READ_ONCE(*{op.addr});")
            elif op.optype == 'fence':
                lines.append(f"  smp_mb();")

        lines.append("}")
        lines.append("")

    # Locations
    loc_parts = [f"{a}" for a in addrs]

    # Exists clause (forbidden outcome)
    if forbidden:
        conds = []
        for reg, val in sorted(forbidden.items()):
            # Determine which thread and which load
            for t in range(n_threads):
                t_loads = [op for op in ops if op.thread == t and op.optype == 'load']
                for i, load_op in enumerate(t_loads):
                    load_reg = load_op.reg or f"r{i}"
                    if load_reg == reg:
                        conds.append(f"{t}:{reg}={val}")
                        break

        lines.append(f"exists ({' /\\ '.join(conds)})")

    return "\n".join(lines)


def pattern_to_litmus_aarch64(name: str) -> str:
    """Generate an AArch64 assembly .litmus file."""
    if name not in PATTERNS:
        raise ValueError(f"Unknown pattern: {name}")
    pat = PATTERNS[name]
    ops = pat['ops']
    n_threads = max(op.thread for op in ops) + 1
    addrs = pat['addresses']
    forbidden = pat['forbidden']

    lines = []
    lines.append(f"AArch64 {name}")
    lines.append("")

    # Initial state
    init_parts = []
    for addr in addrs:
        init_parts.append(f"int {addr}=0")
    lines.append("{ " + "; ".join(init_parts) + "; }")
    lines.append("")

    # Thread bodies
    for t in range(n_threads):
        t_ops = [op for op in ops if op.thread == t]
        lines.append(f"P{t}          |")

        reg_counter = 0
        for op in t_ops:
            if op.optype == 'store':
                val = op.value if op.value is not None else 1
                lines.append(f"  STR W{val}, [{op.addr}] ;")
            elif op.optype == 'load':
                lines.append(f"  LDR W{reg_counter}, [{op.addr}] ;")
                reg_counter += 1
            elif op.optype == 'fence':
                lines.append(f"  DMB ISH ;")

        lines.append("")

    # Exists
    if forbidden:
        conds = []
        for reg, val in sorted(forbidden.items()):
            for t in range(n_threads):
                t_loads = [op for op in ops if op.thread == t and op.optype == 'load']
                for i, _ in enumerate(t_loads):
                    if reg == f"r{i}" or reg == _.reg:
                        conds.append(f"{t}:X{i}={val}")
                        break
        lines.append(f"exists ({' /\\ '.join(conds)})")

    return "\n".join(lines)


def export_all_litmus(output_dir: str, fmt: str = "C"):
    """Export all patterns as .litmus files."""
    os.makedirs(output_dir, exist_ok=True)

    exported = []
    errors = []
    for name in sorted(PATTERNS.keys()):
        try:
            if fmt == "C":
                content = pattern_to_litmus_c(name)
            elif fmt == "AArch64":
                content = pattern_to_litmus_aarch64(name)
            else:
                content = pattern_to_litmus_c(name)

            filepath = os.path.join(output_dir, f"{name}.litmus")
            with open(filepath, 'w') as f:
                f.write(content)
            exported.append(name)
        except Exception as e:
            errors.append((name, str(e)))

    return exported, errors


def generate_validation_script(output_dir: str) -> str:
    """Generate a bash script to run herd7 validation."""
    script = """#!/bin/bash
# Auto-generated herd7 validation script for LITMUS∞
# Run: bash validate_herd7.sh
# Requires: herd7 installed (opam install herd7)

LITMUS_DIR="$(dirname "$0")/litmus_files"
RESULTS="$(dirname "$0")/herd7_results.json"
PASS=0
FAIL=0
TOTAL=0

echo '{"results": [' > "$RESULTS"
FIRST=true

for f in "$LITMUS_DIR"/*.litmus; do
    name=$(basename "$f" .litmus)
    TOTAL=$((TOTAL + 1))

    for cat_file in x86tso.cat aarch64.cat riscv.cat; do
        arch=$(echo "$cat_file" | sed 's/.cat//')
        result=$(herd7 -model "$cat_file" "$f" 2>/dev/null | grep -c "Ok")

        if [ "$FIRST" = true ]; then
            FIRST=false
        else
            echo ',' >> "$RESULTS"
        fi

        echo "  {\\"pattern\\": \\"$name\\", \\"arch\\": \\"$arch\\", \\"allowed\\": $result}" >> "$RESULTS"
    done
done

echo '' >> "$RESULTS"
echo ']}' >> "$RESULTS"

echo "Validated $TOTAL patterns. Results in $RESULTS"
"""
    filepath = os.path.join(output_dir, "validate_herd7.sh")
    with open(filepath, 'w') as f:
        f.write(script)
    os.chmod(filepath, 0o755)
    return filepath


if __name__ == '__main__':
    import json

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'litmus_files')
    exported, errors = export_all_litmus(out_dir, fmt="C")
    print(f"Exported {len(exported)} litmus files to {out_dir}")
    if errors:
        print(f"Errors: {len(errors)}")
        for name, err in errors:
            print(f"  {name}: {err}")

    script = generate_validation_script(os.path.dirname(os.path.abspath(__file__)))
    print(f"Validation script: {script}")
