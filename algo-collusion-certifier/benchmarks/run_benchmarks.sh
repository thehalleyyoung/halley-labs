#!/usr/bin/env bash
# run_benchmarks.sh — Compile and run all CollusionProof benchmarks.
#
# Usage:
#   ./benchmarks/run_benchmarks.sh          # run all
#   ./benchmarks/run_benchmarks.sh detection # run detection only
#   ./benchmarks/run_benchmarks.sh certificate # run certificate only

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$ROOT/benchmarks/results"
mkdir -p "$RESULTS_DIR"

cd "$ROOT"

run_detection() {
    echo "══════════════════════════════════════════════════════════"
    echo "  Running detection benchmark (--release)"
    echo "══════════════════════════════════════════════════════════"
    cargo run --example bench_detection --release 2>&1 | tee "$RESULTS_DIR/detection_$(date +%Y%m%d_%H%M%S).txt"
    echo ""
}

run_certificate() {
    echo "══════════════════════════════════════════════════════════"
    echo "  Running certificate benchmark (--release)"
    echo "══════════════════════════════════════════════════════════"
    cargo run --example bench_certificate_gen --release 2>&1 | tee "$RESULTS_DIR/certificate_$(date +%Y%m%d_%H%M%S).txt"
    echo ""
}

case "${1:-all}" in
    detection)
        run_detection
        ;;
    certificate)
        run_certificate
        ;;
    all)
        run_detection
        run_certificate
        ;;
    *)
        echo "Usage: $0 [detection|certificate|all]"
        exit 1
        ;;
esac

echo "✓ Results saved in $RESULTS_DIR/"
