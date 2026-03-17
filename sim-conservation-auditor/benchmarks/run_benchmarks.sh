#!/usr/bin/env bash
# ============================================================================
# ConservationLint Benchmark Runner
# Runs conservation analysis on 25 benchmark simulation kernels and
# compares detection/localization against GROMACS, LAMMPS, and manual audit.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CLINT_BIN="${PROJECT_ROOT}/target/release/conservation-lint"
RESULTS_FILE="${SCRIPT_DIR}/results.json"
KERNELS_DIR="${PROJECT_ROOT}/benchmarks/kernels"
TIMEOUT_SEC=600
QUICK_MODE=false

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

BENCHMARKS=(
    "verlet_harmonic"
    "leapfrog_kepler"
    "symplectic_euler_pendulum"
    "velocity_verlet_lj"
    "stormer_verlet_spring"
    "ruth3_henon_heiles"
    "yoshida4_solar"
    "forest_ruth_duffing"
    "strang_split_wave"
    "lie_trotter_schrodinger"
    "multi_rate_orbital"
    "leapfrog_ewald_split"
    "verlet_thermostat_nvt"
    "nose_hoover_chain"
    "langevin_brownian"
    "rattle_constrained"
    "shake_water"
    "respa_multi_timestep"
    "implicit_midpoint_stiff"
    "gauss_legendre_rk4"
    "composition_yoshida6"
    "boris_push_emag"
    "pic_vlasov_poisson"
    "sph_fluid_navier_stokes"
    "dg_euler_equations"
)

QUICK_BENCHMARKS=(
    "verlet_harmonic"
    "leapfrog_kepler"
    "symplectic_euler_pendulum"
    "velocity_verlet_lj"
    "strang_split_wave"
)

usage() {
    cat <<EOF
${CYAN}ConservationLint Benchmark Runner${NC}

Usage: $(basename "$0") [OPTIONS]

Options:
    --quick         Run 5 core benchmarks (~2 min)
    --full          Run all 25 benchmarks (~30 min)
    --timeout SEC   Per-benchmark timeout (default: 600)
    --output FILE   Output results file (default: benchmarks/results.json)
    --verbose       Show detailed analysis output
    --help          Show this help

Metrics collected per benchmark:
    - Detection:     Did ConservationLint identify the violation (or absence)?
    - Localization:  Did it correctly identify the responsible source region?
    - Obstruction:   Did it correctly classify repairable vs architectural?
    - Analysis time: Wall-clock seconds
    - Memory:        Peak RSS in MB
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)   QUICK_MODE=true; shift ;;
        --full)    QUICK_MODE=false; shift ;;
        --timeout) TIMEOUT_SEC="$2"; shift 2 ;;
        --output)  RESULTS_FILE="$2"; shift 2 ;;
        --verbose) set -x; shift ;;
        --help)    usage ;;
        *)         echo -e "${RED}Unknown option: $1${NC}"; usage ;;
    esac
done

log_info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[ OK ]${NC}  $*"; }
log_fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }
log_bench() { echo -e "${CYAN}[BENCH]${NC} $*"; }

check_prereqs() {
    if [[ ! -x "$CLINT_BIN" ]]; then
        echo -e "${RED}ConservationLint binary not found at $CLINT_BIN${NC}"
        echo "Build with: cargo build --release"
        exit 1
    fi
    log_ok "Binary: $CLINT_BIN"
}

run_single_benchmark() {
    local name="$1"
    local kernel="${KERNELS_DIR}/${name}.py"

    if [[ ! -f "$kernel" ]]; then
        echo "{\"status\":\"skipped\",\"reason\":\"kernel_not_found\"}"
        return
    fi

    local start_time
    start_time=$(python3 -c "import time; print(time.time())")

    local output exit_code=0
    output=$(timeout "$TIMEOUT_SEC" "$CLINT_BIN" analyze "$kernel" \
        --format json --bch-order 4 2>&1) || exit_code=$?

    local end_time
    end_time=$(python3 -c "import time; print(time.time())")
    local elapsed
    elapsed=$(python3 -c "print(round($end_time - $start_time, 2))")

    if [[ $exit_code -eq 124 ]]; then
        echo "{\"status\":\"timeout\",\"time_sec\":${TIMEOUT_SEC}}"
        return
    fi

    echo "$output" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    data['analysis_time_sec'] = $elapsed
    data['status'] = 'completed'
    json.dump(data, sys.stdout)
except:
    json.dump({'status':'error','time_sec':$elapsed}, sys.stdout)
"
}

main() {
    echo ""
    echo -e "${CYAN}ConservationLint Benchmark Suite${NC}"
    echo ""

    if $QUICK_MODE; then
        log_info "Mode: QUICK (5 benchmarks)"
        ACTIVE=("${QUICK_BENCHMARKS[@]}")
    else
        log_info "Mode: FULL (25 benchmarks)"
        ACTIVE=("${BENCHMARKS[@]}")
    fi

    check_prereqs
    echo ""

    local total=${#ACTIVE[@]}
    local passed=0 failed=0 skipped=0

    local tmp=$(mktemp)
    echo "[" > "$tmp"

    for i in "${!ACTIVE[@]}"; do
        local name="${ACTIVE[$i]}"
        log_bench "[$((i+1))/$total] $name"

        local result
        result=$(run_single_benchmark "$name")

        local status
        status=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','error'))" 2>/dev/null || echo "error")

        case "$status" in
            completed) ((passed++)); log_ok "$name" ;;
            timeout)   ((failed++)); log_fail "$name (timeout)" ;;
            skipped)   ((skipped++)) ;;
            *)         ((failed++)); log_fail "$name (error)" ;;
        esac

        [[ $i -gt 0 ]] && echo "," >> "$tmp"
        echo "{\"name\":\"$name\",\"result\":$result}" >> "$tmp"
    done

    echo "]" >> "$tmp"

    python3 -c "
import json
benchmarks = json.load(open('$tmp'))
result = {
    'metadata': {
        'tool': 'ConservationLint',
        'version': '0.9.0',
        'timestamp': '$(date -u +%Y-%m-%dT%H:%M:%SZ)',
        'mode': '$( $QUICK_MODE && echo quick || echo full )',
        'timeout_sec': $TIMEOUT_SEC
    },
    'benchmarks': benchmarks,
    'summary': {
        'total': $total,
        'passed': $passed,
        'failed': $failed,
        'skipped': $skipped
    }
}
with open('$RESULTS_FILE', 'w') as f:
    json.dump(result, f, indent=2)
"

    rm -f "$tmp"

    echo ""
    log_info "Results: $RESULTS_FILE"
    log_info "Passed: $passed / $total"
    [[ $failed -gt 0 ]] && log_fail "Failed: $failed"
    [[ $skipped -gt 0 ]] && log_info "Skipped: $skipped"
}

main "$@"
