#!/usr/bin/env bash
# =============================================================================
# Example: Composing Contracts Across a Cryptographic Library
# =============================================================================
#
# This script demonstrates the full compositional analysis workflow:
# 1. Analyze individual functions in a crypto library
# 2. Compose per-function contracts into a whole-library bound
# 3. Validate the composed contracts for soundness
# 4. Generate a certificate chain proving the library-level bound
#
# The composition rule B_{f;g}(s) = B_f(s) + B_g(τ_f(s)) enables
# analyzing each function once and composing without re-analyzing
# the whole program.
#
# Prerequisites:
#   - leakage-contracts installed
#   - A cryptographic library binary (e.g., libsodium)
#
# Usage:
#   ./examples/compose_library.sh <path-to-library.so>
# =============================================================================

set -euo pipefail

BINARY="${1:?Usage: $0 <path-to-library.so>}"
OUTPUT_DIR="./composition_output"
CONTRACTS_DIR="$OUTPUT_DIR/contracts"

echo "=== Certified Leakage Contracts: Library Composition ==="
echo ""
echo "Library: $BINARY"
echo ""

# Create output directories
mkdir -p "$CONTRACTS_DIR"

# --- Step 1: Analyze all functions ---
echo "[Step 1] Analyzing all functions in the library..."
leakage-contracts analyze "$BINARY" \
    --cache-sets 64 \
    --cache-ways 8 \
    --line-size 64 \
    --spec-window 20 \
    --widening delayed \
    --widening-delay 3 \
    --unroll 4 \
    --output "$CONTRACTS_DIR/" \
    --json

echo "[Step 1] Per-function contracts written to $CONTRACTS_DIR/"
echo ""

# --- Step 2: Compose contracts into a library-level bound ---
echo "[Step 2] Composing contracts across function boundaries..."
leakage-contracts compose "$CONTRACTS_DIR/" \
    --strategy sequential \
    --validate \
    --output "$OUTPUT_DIR/library_contract.json" \
    --json

echo "[Step 2] Library contract written to $OUTPUT_DIR/library_contract.json"
echo ""

# --- Step 3: Generate a full certificate chain ---
echo "[Step 3] Generating certificate chain with witnesses..."
leakage-contracts certify "$OUTPUT_DIR/library_contract.json" \
    --chain \
    --witness \
    --format json \
    --output "$OUTPUT_DIR/library_certificate.json"

echo "[Step 3] Certificate written to $OUTPUT_DIR/library_certificate.json"
echo ""

# --- Step 4: Independently verify the certificate ---
echo "[Step 4] Verifying certificate independently..."
leakage-contracts certify "$OUTPUT_DIR/library_certificate.json" \
    --verify \
    --output "$OUTPUT_DIR/verification_report.json"

echo "[Step 4] Verification report written to $OUTPUT_DIR/verification_report.json"
echo ""

# --- Summary ---
echo "=== Composition Complete ==="
echo ""
echo "Output files:"
echo "  Per-function contracts:  $CONTRACTS_DIR/"
echo "  Library contract:        $OUTPUT_DIR/library_contract.json"
echo "  Certificate chain:       $OUTPUT_DIR/library_certificate.json"
echo "  Verification report:     $OUTPUT_DIR/verification_report.json"
echo ""
echo "To inspect the library-level bound:"
echo "  cat $OUTPUT_DIR/library_contract.json | python3 -m json.tool"
echo ""
echo "The certificate chain links per-function fixpoint witnesses"
echo "through composition proofs to the whole-library guarantee."
