#!/bin/bash
# Spectacles Reproducibility Script
# Reproduces all experimental results reported in the paper.
#
# Prerequisites:
#   - Rust toolchain (tested with rustc 1.91.1, cargo 1.91.1)
#   - Python 3.10+ (tested with Python 3.14.0)
#   - ~2GB disk space for build artifacts
#   - ~10 minutes total runtime on Apple M1 Pro or equivalent
#
# Hardware used for paper results:
#   - Apple M1 Pro, 16GB RAM, macOS 14.x
#   - All timing results are hardware-dependent
#
# Random seeds used: 42, 123, 456, 789, 1337, 2024, 31415, 27182, 99999, 54321

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Spectacles Reproducibility ==="
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# 1. Environment info
echo "--- Environment ---"
rustc --version
cargo --version
python3 --version
uname -a
echo ""

# 2. Build
echo "--- Building (release mode) ---"
cargo build --release 2>&1 | tail -1
echo "Build complete."
echo ""

# 3. Compilation correctness (57,518 checks)
echo "--- Compilation Correctness (57,518 checks, 10 seeds) ---"
echo "Seeds: 42, 123, 456, 789, 1337, 2024, 31415, 27182, 99999, 54321"
cargo run --release --bin compilation_correctness 2>&1 | tail -5
echo ""

# 4. STARK scaling benchmark (55 proofs)
echo "--- STARK Scaling Benchmark (55 proofs, 32-512 states, 5 trials) ---"
cargo run --release --bin stark_scaling_extended 2>&1 | tail -5
echo ""

# 5. Property-based tests (9,839 instances)
echo "--- Property-Based Tests (14 properties, 9,839 instances) ---"
cargo run --release --bin property_tests 2>&1 | tail -5
echo ""

# 6. Benchmark evaluation (2,825 checks)
echo "--- Benchmark Evaluation (565 pairs × 5 metrics) ---"
cargo run --release --bin real_benchmark 2>&1 | tail -5
echo ""

# 7. Contamination experiments
echo "--- Contamination Detection ---"
python3 contamination_expanded.py 2>&1 | tail -3
python3 contamination_adversarial.py 2>&1 | tail -3
echo ""

# 8. Ablation study
echo "--- Ablation Study ---"
python3 ablation_study.py 2>&1 | tail -2
echo ""

# 9. Corpus analysis
echo "--- Corpus Characterization ---"
python3 corpus_analysis.py 2>&1 | tail -4
echo ""

echo "=== All experiments complete ==="
echo "Results files:"
ls -la *.json 2>/dev/null | awk '{print "  "$NF": "$5" bytes"}'
