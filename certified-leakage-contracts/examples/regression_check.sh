#!/usr/bin/env bash
# =============================================================================
# Example: CI Regression Detection Between Two Binary Versions
# =============================================================================
#
# This script demonstrates how to use leakage-contracts for regression
# detection in a CI/CD pipeline.  It compares the leakage contracts of
# two versions of a cryptographic library and reports any functions whose
# leakage increased beyond a threshold.
#
# This is the primary CI use case: even when absolute bounds are conservative,
# detecting *changes* in leakage between versions catches real regressions.
#
# Prerequisites:
#   - leakage-contracts installed
#   - Two versions of a cryptographic library binary
#
# Usage:
#   ./examples/regression_check.sh <baseline.so> <candidate.so>
#
# Exit codes (CI-compatible):
#   0 - No regression detected
#   1 - Regression detected (leakage increased beyond threshold)
#   2 - Analysis error
# =============================================================================

set -euo pipefail

BASELINE="${1:?Usage: $0 <baseline.so> <candidate.so>}"
CANDIDATE="${2:?Usage: $0 <baseline.so> <candidate.so>}"
THRESHOLD="${3:-0.5}"  # Default threshold: 0.5 bits
OUTPUT_DIR="./regression_output"

echo "=== Certified Leakage Contracts: Regression Detection ==="
echo ""
echo "Baseline:  $BASELINE"
echo "Candidate: $CANDIDATE"
echo "Threshold: $THRESHOLD bits"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# --- Step 1: Analyze baseline binary ---
echo "[Step 1] Analyzing baseline binary..."
leakage-contracts analyze "$BASELINE" \
    --cache-sets 64 \
    --cache-ways 8 \
    --line-size 64 \
    --spec-window 20 \
    --output "$OUTPUT_DIR/baseline_contracts.json" \
    --json \
    --quiet

echo "[Step 1] Baseline analysis complete."

# --- Step 2: Analyze candidate binary ---
echo ""
echo "[Step 2] Analyzing candidate binary..."
leakage-contracts analyze "$CANDIDATE" \
    --cache-sets 64 \
    --cache-ways 8 \
    --line-size 64 \
    --spec-window 20 \
    --output "$OUTPUT_DIR/candidate_contracts.json" \
    --json \
    --quiet

echo "[Step 2] Candidate analysis complete."

# --- Step 3: Run regression comparison ---
echo ""
echo "[Step 3] Comparing contracts for regressions..."
echo ""

REGRESSION_EXIT=0
leakage-contracts regress \
    --baseline "$OUTPUT_DIR/baseline_contracts.json" \
    --candidate "$OUTPUT_DIR/candidate_contracts.json" \
    --threshold "$THRESHOLD" \
    --ci \
    --report "$OUTPUT_DIR/regression_report.json" \
    || REGRESSION_EXIT=$?

echo ""

# --- Step 4: Interpret results ---
case $REGRESSION_EXIT in
    0)
        echo "=== RESULT: NO REGRESSION ==="
        echo "All functions are within the $THRESHOLD-bit threshold."
        ;;
    1)
        echo "=== RESULT: REGRESSION DETECTED ==="
        echo "One or more functions exceeded the $THRESHOLD-bit threshold."
        echo "See $OUTPUT_DIR/regression_report.json for details."
        ;;
    2)
        echo "=== RESULT: ANALYSIS ERROR ==="
        echo "An error occurred during analysis. Check the logs."
        ;;
    *)
        echo "=== RESULT: UNEXPECTED EXIT CODE $REGRESSION_EXIT ==="
        ;;
esac

echo ""
echo "Full report: $OUTPUT_DIR/regression_report.json"

exit $REGRESSION_EXIT
