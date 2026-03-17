#!/usr/bin/env bash
# benchmark_suite.sh — Build and run the Spectral Decomposition Oracle benchmarks
#
# Usage:
#   ./benchmarks/benchmark_suite.sh [--sizes 100,500,1000] [--skip-build]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# Defaults
SIZES="100,500,1000,5000,10000"
SKIP_BUILD=false
TRIALS=5

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sizes)    SIZES="$2"; shift 2 ;;
        --trials)   TRIALS="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--sizes N,N,...] [--trials N] [--skip-build]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
echo "============================================="
echo " Spectral Decomposition Oracle — Benchmarks"
echo "============================================="
echo ""

if ! command -v cargo &>/dev/null; then
    echo "WARNING: cargo not found. Rust benchmarks will be skipped."
    echo "         Install Rust via https://rustup.rs"
    CARGO_AVAILABLE=false
else
    CARGO_VERSION="$(cargo --version)"
    echo "cargo: $CARGO_VERSION"
    CARGO_AVAILABLE=true
fi

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 is required for the benchmark harness."
    exit 1
fi
PYTHON_VERSION="$(python3 --version)"
echo "python: $PYTHON_VERSION"
echo ""

# ---------------------------------------------------------------------------
# Build (Rust)
# ---------------------------------------------------------------------------
if [[ "$CARGO_AVAILABLE" == true && "$SKIP_BUILD" == false ]]; then
    echo "--- Building Rust project (release mode) ---"
    cd "$PROJECT_ROOT"
    if [[ -f "implementation/Cargo.toml" ]]; then
        cargo build --release --manifest-path implementation/Cargo.toml 2>&1 | tail -5
        echo "Build complete."
    else
        echo "No Cargo.toml found at implementation/Cargo.toml — skipping Rust build."
    fi
    echo ""
fi

# ---------------------------------------------------------------------------
# Run Rust benchmarks (if available)
# ---------------------------------------------------------------------------
if [[ "$CARGO_AVAILABLE" == true && -f "$PROJECT_ROOT/implementation/Cargo.toml" ]]; then
    echo "--- Running Rust benchmarks ---"
    cd "$PROJECT_ROOT"

    mkdir -p "$RESULTS_DIR"

    if cargo bench --manifest-path implementation/Cargo.toml -- --list 2>/dev/null | head -5; then
        cargo bench --manifest-path implementation/Cargo.toml 2>&1 | tee "$RESULTS_DIR/rust_bench_${TIMESTAMP}.txt"
    else
        echo "No Rust benchmarks found or bench target unavailable — skipping."
    fi
    echo ""
fi

# ---------------------------------------------------------------------------
# Run Python simulation benchmarks
# ---------------------------------------------------------------------------
echo "--- Running Python simulation benchmarks ---"
cd "$PROJECT_ROOT"

mkdir -p "$RESULTS_DIR"

python3 "$SCRIPT_DIR/run_benchmarks.py" \
    --sizes "$SIZES" \
    --trials "$TRIALS" \
    --output "$RESULTS_DIR"

echo ""

# ---------------------------------------------------------------------------
# Generate summary report
# ---------------------------------------------------------------------------
echo "--- Generating summary report ---"

REPORT_FILE="$RESULTS_DIR/report_${TIMESTAMP}.md"

cat > "$REPORT_FILE" <<EOF
# Benchmark Report — Spectral Decomposition Oracle

**Date:** $(date -u +"%Y-%m-%d %H:%M:%S UTC")
**Sizes:** $SIZES
**Trials per size:** $TRIALS

## Python Simulation Results

$(if [[ -f "$RESULTS_DIR/benchmark_summary.csv" ]]; then
    echo '```'
    column -t -s, "$RESULTS_DIR/benchmark_summary.csv" 2>/dev/null || cat "$RESULTS_DIR/benchmark_summary.csv"
    echo '```'
else
    echo "No CSV summary found."
fi)

## Notes

- Python benchmarks simulate spectral feature extraction (no actual Rust FFI).
- Timings reflect pure-Python overhead; actual Rust implementation will differ.
- Use \`cargo bench\` for native Rust benchmarks once the implementation is built.

## Files

- \`benchmark_results.json\` — Full results with per-trial data
- \`benchmark_summary.csv\` — Aggregated summary table
EOF

echo "  Report written to $REPORT_FILE"
echo ""
echo "============================================="
echo " Benchmark suite complete."
echo "============================================="
