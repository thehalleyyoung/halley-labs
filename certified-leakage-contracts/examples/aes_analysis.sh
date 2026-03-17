#!/usr/bin/env bash
# =============================================================================
# Example: Analyzing AES-128 Round Function for Cache Leakage
# =============================================================================
#
# This script demonstrates how to use leakage-contracts to analyze the
# AES T-table implementation in OpenSSL's libcrypto for cache side-channel
# leakage.  T-table AES is a classic target because each table lookup
# reveals the index (derived from the key) through cache-line access patterns.
#
# Prerequisites:
#   - leakage-contracts installed (cargo install --path crates/leak-cli)
#   - An OpenSSL build with T-table AES (not AES-NI)
#
# Usage:
#   ./examples/aes_analysis.sh <path-to-libcrypto.so>
# =============================================================================

set -euo pipefail

BINARY="${1:?Usage: $0 <path-to-libcrypto.so>}"
OUTPUT_DIR="./aes_analysis_output"

echo "=== Certified Leakage Contracts: AES Cache Leakage Analysis ==="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# --- Step 1: Analyze the AES encrypt function ---
echo "[Step 1] Analyzing AES encrypt function..."
leakage-contracts analyze "$BINARY" \
    --function aes_encrypt \
    --cache-sets 64 \
    --cache-ways 8 \
    --line-size 64 \
    --spec-window 20 \
    --widening delayed \
    --widening-delay 3 \
    --secret-regions "0x4a000:256,0x4a100:256,0x4a200:256,0x4a300:256" \
    --output "$OUTPUT_DIR/aes_encrypt_contract.json" \
    --json

echo ""
echo "[Step 1] Contract written to $OUTPUT_DIR/aes_encrypt_contract.json"

# --- Step 2: Also analyze the key expansion ---
echo ""
echo "[Step 2] Analyzing AES key expansion..."
leakage-contracts analyze "$BINARY" \
    --function aes_set_encrypt_key \
    --cache-sets 64 \
    --cache-ways 8 \
    --line-size 64 \
    --spec-window 20 \
    --output "$OUTPUT_DIR/aes_keygen_contract.json" \
    --json

echo ""
echo "[Step 2] Contract written to $OUTPUT_DIR/aes_keygen_contract.json"

# --- Step 3: Generate a certificate for the analysis ---
echo ""
echo "[Step 3] Generating certificate..."
leakage-contracts certify "$OUTPUT_DIR/aes_encrypt_contract.json" \
    --witness \
    --format json \
    --output "$OUTPUT_DIR/aes_encrypt_certificate.json"

echo ""
echo "[Step 3] Certificate written to $OUTPUT_DIR/aes_encrypt_certificate.json"

# --- Step 4: Display summary ---
echo ""
echo "=== Analysis Complete ==="
echo ""
echo "Output files:"
ls -la "$OUTPUT_DIR/"
echo ""
echo "To inspect the contract:"
echo "  cat $OUTPUT_DIR/aes_encrypt_contract.json | python3 -m json.tool"
echo ""
echo "Expected: T-table AES should show ~3-4 bits of cache leakage."
echo "AES-NI implementations should show 0 bits (constant-time)."
