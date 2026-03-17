#!/usr/bin/env bash
# ============================================================================
# LeakCert Benchmark Runner
# Runs quantitative leakage analysis benchmarks on cryptographic binaries
# and compares against CacheAudit, Spectector, and Binsec/Rel baselines.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LEAKCERT_BIN="${PROJECT_ROOT}/target/release/leakcert"
RESULTS_FILE="${SCRIPT_DIR}/results.json"
TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TIMEOUT_SEC=3600
QUICK_MODE=false
VERBOSE=false

BENCH_BINARIES="${PROJECT_ROOT}/bench_binaries"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ---------------------------------------------------------------------------
# Benchmark Suite
# ---------------------------------------------------------------------------
BENCHMARKS=(
    "aes-128-ecb"
    "aes-256-gcm"
    "chacha20-poly1305"
    "rsa-2048"
    "ecdsa-p256"
    "sha-256"
    "x25519"
    "hkdf-sha256"
    "boringssl-full"
)

QUICK_BENCHMARKS=(
    "aes-128-ecb"
    "sha-256"
    "x25519"
)

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
    cat <<EOF
${CYAN}LeakCert Benchmark Runner${NC}

Usage: $(basename "$0") [OPTIONS]

Options:
    --quick         Run reduced suite (AES-128, SHA-256, X25519 only)
    --full          Run complete suite including BoringSSL full-library (default)
    --timeout SEC   Per-benchmark timeout in seconds (default: 3600)
    --output FILE   Output results file (default: benchmarks/results.json)
    --verbose       Print detailed analysis output
    --baselines     Also run baseline tools (requires CacheAudit, Spectector, Binsec/Rel)
    --help          Show this help message

Examples:
    $(basename "$0") --quick                    # CI smoke test (~5 min)
    $(basename "$0") --full --verbose           # Full evaluation (~90 min)
    $(basename "$0") --full --baselines         # Full comparison run (~4 hrs)
EOF
    exit 0
}

# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------
RUN_BASELINES=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)      QUICK_MODE=true; shift ;;
        --full)       QUICK_MODE=false; shift ;;
        --timeout)    TIMEOUT_SEC="$2"; shift 2 ;;
        --output)     RESULTS_FILE="$2"; shift 2 ;;
        --verbose)    VERBOSE=true; shift ;;
        --baselines)  RUN_BASELINES=true; shift ;;
        --help)       usage ;;
        *)            echo -e "${RED}Unknown option: $1${NC}"; usage ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log_info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_err()   { echo -e "${RED}[ERROR]${NC} $*"; }
log_bench() { echo -e "${CYAN}[BENCH]${NC} $*"; }

check_prereqs() {
    if [[ ! -x "$LEAKCERT_BIN" ]]; then
        log_err "LeakCert binary not found at $LEAKCERT_BIN"
        log_info "Build with: cargo build --release"
        exit 1
    fi

    if [[ ! -d "$BENCH_BINARIES" ]]; then
        log_err "Benchmark binaries directory not found: $BENCH_BINARIES"
        log_info "Run: ./scripts/fetch_bench_binaries.sh"
        exit 1
    fi

    log_ok "LeakCert binary: $LEAKCERT_BIN"
    log_ok "Benchmark binaries: $BENCH_BINARIES"
}

get_peak_memory_kb() {
    local pid=$1
    if [[ "$(uname)" == "Darwin" ]]; then
        ps -o rss= -p "$pid" 2>/dev/null || echo "0"
    else
        cat "/proc/$pid/status" 2>/dev/null | grep VmPeak | awk '{print $2}' || echo "0"
    fi
}

# ---------------------------------------------------------------------------
# Run a single LeakCert benchmark
# ---------------------------------------------------------------------------
run_leakcert_bench() {
    local name="$1"
    local binary="${BENCH_BINARIES}/${name}.o"

    if [[ ! -f "$binary" ]]; then
        log_warn "Binary not found for $name, skipping"
        echo '{"status":"skipped","reason":"binary_not_found"}'
        return
    fi

    log_bench "Analyzing: $name"
    local start_ns
    start_ns=$(python3 -c "import time; print(int(time.time_ns()))")

    local output
    local exit_code=0
    output=$(timeout "$TIMEOUT_SEC" "$LEAKCERT_BIN" analyze \
        --binary "$binary" \
        --speculative-depth 8 \
        --composition \
        --output-format json \
        2>&1) || exit_code=$?

    local end_ns
    end_ns=$(python3 -c "import time; print(int(time.time_ns()))")

    local elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
    local elapsed_sec
    elapsed_sec=$(python3 -c "print(round($elapsed_ms / 1000.0, 2))")

    if [[ $exit_code -eq 124 ]]; then
        log_warn "$name: TIMEOUT after ${TIMEOUT_SEC}s"
        echo "{\"status\":\"timeout\",\"timeout_sec\":${TIMEOUT_SEC}}"
        return
    elif [[ $exit_code -ne 0 ]]; then
        log_err "$name: analysis failed (exit code $exit_code)"
        echo "{\"status\":\"error\",\"exit_code\":${exit_code}}"
        return
    fi

    local leakage_bits
    leakage_bits=$(echo "$output" | python3 -c "import sys,json; print(json.load(sys.stdin).get('leakage_bound_bits','N/A'))" 2>/dev/null || echo "N/A")

    local contracts
    contracts=$(echo "$output" | python3 -c "import sys,json; print(json.load(sys.stdin).get('contracts_generated',0))" 2>/dev/null || echo "0")

    log_ok "$name: ${leakage_bits} bits leakage bound in ${elapsed_sec}s"

    cat <<ENDJSON
{
    "status": "completed",
    "analysis_time_sec": ${elapsed_sec},
    "leakage_bound_bits": ${leakage_bits},
    "speculative_depth": 8,
    "contracts_generated": ${contracts}
}
ENDJSON
}

# ---------------------------------------------------------------------------
# Run baseline tool (if installed)
# ---------------------------------------------------------------------------
run_baseline() {
    local tool="$1"
    local benchmark="$2"

    case "$tool" in
        cacheaudit)
            if command -v cacheaudit &>/dev/null; then
                log_bench "Running CacheAudit on $benchmark"
                timeout 1800 cacheaudit "${BENCH_BINARIES}/${benchmark}.o" 2>&1 || true
            else
                log_warn "CacheAudit not installed, using cached results"
            fi
            ;;
        spectector)
            if command -v spectector &>/dev/null; then
                log_bench "Running Spectector on $benchmark"
                timeout 1800 spectector --check "${BENCH_BINARIES}/${benchmark}.o" 2>&1 || true
            else
                log_warn "Spectector not installed, using cached results"
            fi
            ;;
        binsecrel)
            if command -v binsec &>/dev/null; then
                log_bench "Running Binsec/Rel on $benchmark"
                timeout 1800 binsec -relse "${BENCH_BINARIES}/${benchmark}.o" 2>&1 || true
            else
                log_warn "Binsec/Rel not installed, using cached results"
            fi
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Collect system information
# ---------------------------------------------------------------------------
collect_sysinfo() {
    cat <<ENDJSON
{
    "hostname": "$(hostname)",
    "os": "$(uname -s) $(uname -r)",
    "arch": "$(uname -m)",
    "cpu": "$(sysctl -n machdep.cpu.brand_string 2>/dev/null || lscpu 2>/dev/null | grep 'Model name' | sed 's/.*: //' || echo 'unknown')",
    "cores": $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1),
    "memory_gb": $(python3 -c "import os; print(round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024.**3), 1))" 2>/dev/null || echo "\"unknown\"")
}
ENDJSON
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║          LeakCert Benchmark Suite v0.9.0            ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════╝${NC}"
    echo ""

    if $QUICK_MODE; then
        log_info "Mode: QUICK (3 benchmarks)"
        ACTIVE_BENCHMARKS=("${QUICK_BENCHMARKS[@]}")
    else
        log_info "Mode: FULL (${#BENCHMARKS[@]} benchmarks)"
        ACTIVE_BENCHMARKS=("${BENCHMARKS[@]}")
    fi

    log_info "Timeout: ${TIMEOUT_SEC}s per benchmark"
    log_info "Output:  ${RESULTS_FILE}"
    echo ""

    check_prereqs

    echo ""
    log_info "Starting benchmark run at ${TIMESTAMP}"
    echo ""

    local total=${#ACTIVE_BENCHMARKS[@]}
    local completed=0
    local failed=0
    local skipped=0

    local tmp_results
    tmp_results=$(mktemp)

    echo "[" > "$tmp_results"

    for i in "${!ACTIVE_BENCHMARKS[@]}"; do
        local bench="${ACTIVE_BENCHMARKS[$i]}"
        log_info "[$((i+1))/$total] $bench"

        local result
        result=$(run_leakcert_bench "$bench")

        local status
        status=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','error'))" 2>/dev/null || echo "error")

        case "$status" in
            completed) ((completed++)) ;;
            timeout|error) ((failed++)) ;;
            skipped) ((skipped++)) ;;
        esac

        if [[ $i -gt 0 ]]; then echo "," >> "$tmp_results"; fi
        echo "{\"name\": \"$bench\", \"leakcert\": $result}" >> "$tmp_results"

        echo ""
    done

    echo "]" >> "$tmp_results"

    cat > "$RESULTS_FILE" <<ENDJSON
{
    "metadata": {
        "tool": "LeakCert",
        "version": "0.9.0",
        "timestamp": "${TIMESTAMP}",
        "mode": "$(if $QUICK_MODE; then echo quick; else echo full; fi)",
        "timeout_sec": ${TIMEOUT_SEC},
        "system": $(collect_sysinfo)
    },
    "benchmarks": $(cat "$tmp_results"),
    "summary": {
        "total": $total,
        "completed": $completed,
        "failed": $failed,
        "skipped": $skipped
    }
}
ENDJSON

    rm -f "$tmp_results"

    echo ""
    echo -e "${CYAN}════════════════════════════════════════════════════════${NC}"
    log_info "Benchmark run complete"
    log_ok   "Completed: $completed / $total"
    if [[ $failed -gt 0 ]]; then
        log_warn "Failed:    $failed"
    fi
    if [[ $skipped -gt 0 ]]; then
        log_warn "Skipped:   $skipped"
    fi
    log_info "Results:   ${RESULTS_FILE}"
    echo -e "${CYAN}════════════════════════════════════════════════════════${NC}"
}

main "$@"
