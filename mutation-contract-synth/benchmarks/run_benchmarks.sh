#!/usr/bin/env bash
# run_benchmarks.sh — Run MutSpec synthesis on all examples and collect metrics.
#
# Usage:
#   ./benchmarks/run_benchmarks.sh [--trials N] [--output DIR]
#
# Options:
#   --trials N    Number of repeated trials for timing (default: 3)
#   --output DIR  Directory for results (default: benchmarks/results)

set -euo pipefail

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXAMPLES_DIR="$PROJECT_ROOT/examples"
EXPECTED_DIR="$EXAMPLES_DIR/expected_contracts"
BENCH_DATA="$SCRIPT_DIR/data/benchmark_suite.json"

TRIALS=3
OUTPUT_DIR="$SCRIPT_DIR/results"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --trials)  TRIALS="$2"; shift 2 ;;
        --output)  OUTPUT_DIR="$2"; shift 2 ;;
        *)         echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_JSON="$OUTPUT_DIR/results_${TIMESTAMP}.json"
RESULT_TXT="$OUTPUT_DIR/results_${TIMESTAMP}.txt"

mkdir -p "$OUTPUT_DIR"

# ------------------------------------------------------------------
# Prerequisites
# ------------------------------------------------------------------
check_prereqs() {
    local missing=0

    if ! command -v cargo &>/dev/null; then
        echo "ERROR: cargo not found. Install Rust: https://rustup.rs" >&2
        missing=1
    fi

    if ! command -v z3 &>/dev/null; then
        echo "ERROR: z3 not found. Install Z3: https://github.com/Z3Prover/z3" >&2
        missing=1
    fi

    if ! command -v jq &>/dev/null; then
        echo "WARNING: jq not found; JSON output will be raw." >&2
    fi

    if [[ $missing -ne 0 ]]; then
        echo "Aborting due to missing prerequisites." >&2
        exit 1
    fi

    # Ensure the binary is built
    echo "Building MutSpec (release mode)..."
    (cd "$PROJECT_ROOT" && cargo build --release --quiet 2>/dev/null) || true
}

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
now_ms() {
    if [[ "$(uname)" == "Darwin" ]]; then
        python3 -c 'import time; print(int(time.time()*1000))'
    else
        date +%s%3N
    fi
}

# Run synthesis on a single file and return JSON metrics.
run_one() {
    local src="$1"
    local name
    name="$(basename "$src" .ms)"

    local total_time=0
    local contracts=0
    local mutants_total=0
    local mutants_killed=0
    local z3_calls=0

    for trial in $(seq 1 "$TRIALS"); do
        local t_start t_end elapsed
        t_start="$(now_ms)"

        # Run synthesis; capture structured output
        local output
        output="$(cargo run --release --quiet -- synthesize "$src" 2>/dev/null)" || output="{}"

        t_end="$(now_ms)"
        elapsed=$(( t_end - t_start ))
        total_time=$(( total_time + elapsed ))

        # Parse metrics from synthesizer output (last trial wins for counts)
        contracts="$(echo "$output" | jq -r '.contracts_count // 0' 2>/dev/null || echo 0)"
        mutants_total="$(echo "$output" | jq -r '.mutants_total // 0' 2>/dev/null || echo 0)"
        mutants_killed="$(echo "$output" | jq -r '.mutants_killed // 0' 2>/dev/null || echo 0)"
        z3_calls="$(echo "$output" | jq -r '.z3_queries // 0' 2>/dev/null || echo 0)"
    done

    local avg_time=$(( total_time / TRIALS ))

    local mutation_score="0.00"
    if [[ "$mutants_total" -gt 0 ]]; then
        mutation_score="$(echo "scale=4; $mutants_killed / $mutants_total" | bc)"
    fi

    cat <<EOF
{
  "name": "$name",
  "source": "$src",
  "trials": $TRIALS,
  "avg_time_ms": $avg_time,
  "contracts_synthesized": $contracts,
  "mutants_total": $mutants_total,
  "mutants_killed": $mutants_killed,
  "mutation_score": $mutation_score,
  "z3_queries": $z3_calls
}
EOF
}

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
main() {
    check_prereqs

    echo "============================================"
    echo " MutSpec Benchmark Suite"
    echo " Trials per program: $TRIALS"
    echo " Timestamp: $TIMESTAMP"
    echo "============================================"
    echo ""

    local results=()
    local total_programs=0
    local total_contracts=0
    local total_time=0

    for src in "$EXAMPLES_DIR"/*.ms; do
        [[ -f "$src" ]] || continue
        local name
        name="$(basename "$src" .ms)"
        echo -n "Running: $name ... "

        local result
        result="$(run_one "$src")"
        results+=("$result")

        local t c
        t="$(echo "$result" | jq -r '.avg_time_ms')"
        c="$(echo "$result" | jq -r '.contracts_synthesized')"
        total_programs=$(( total_programs + 1 ))
        total_contracts=$(( total_contracts + c ))
        total_time=$(( total_time + t ))

        echo "done (${t}ms, ${c} contracts)"
    done

    # ---- Write JSON results ----
    {
        echo "{"
        echo "  \"timestamp\": \"$TIMESTAMP\","
        echo "  \"trials\": $TRIALS,"
        echo "  \"total_programs\": $total_programs,"
        echo "  \"total_contracts\": $total_contracts,"
        echo "  \"total_time_ms\": $total_time,"
        echo "  \"programs\": ["
        local first=1
        for r in "${results[@]}"; do
            [[ $first -eq 0 ]] && echo ","
            echo "    $r"
            first=0
        done
        echo "  ]"
        echo "}"
    } > "$RESULT_JSON"

    # ---- Write formatted table ----
    {
        printf "%-20s %10s %10s %10s %12s %10s\n" \
            "Program" "Time(ms)" "Contracts" "Mutants" "MutScore" "Z3Calls"
        printf "%-20s %10s %10s %10s %12s %10s\n" \
            "--------------------" "----------" "----------" "----------" "------------" "----------"
        for r in "${results[@]}"; do
            local n t c m ms z
            n="$(echo "$r" | jq -r '.name')"
            t="$(echo "$r" | jq -r '.avg_time_ms')"
            c="$(echo "$r" | jq -r '.contracts_synthesized')"
            m="$(echo "$r" | jq -r '.mutants_total')"
            ms="$(echo "$r" | jq -r '.mutation_score')"
            z="$(echo "$r" | jq -r '.z3_queries')"
            printf "%-20s %10s %10s %10s %12s %10s\n" "$n" "$t" "$c" "$m" "$ms" "$z"
        done
        echo ""
        echo "Total programs: $total_programs"
        echo "Total contracts: $total_contracts"
        echo "Total time: ${total_time}ms"
    } > "$RESULT_TXT"

    echo ""
    cat "$RESULT_TXT"
    echo ""
    echo "Results written to:"
    echo "  JSON: $RESULT_JSON"
    echo "  Table: $RESULT_TXT"
}

main "$@"
