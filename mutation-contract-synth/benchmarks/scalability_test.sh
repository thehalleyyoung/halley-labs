#!/usr/bin/env bash
# scalability_test.sh — Measure MutSpec synthesis time as program size increases.
#
# Usage:
#   ./benchmarks/scalability_test.sh [--max-sites N] [--step S] [--trials T] [--output DIR]
#
# Generates synthetic programs with increasing numbers of mutation sites,
# runs synthesis on each, and records timing data suitable for plotting.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/results"

MAX_SITES=200
STEP=10
TRIALS=3

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-sites) MAX_SITES="$2"; shift 2 ;;
        --step)      STEP="$2";      shift 2 ;;
        --trials)    TRIALS="$2";    shift 2 ;;
        --output)    OUTPUT_DIR="$2"; shift 2 ;;
        *)           echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DAT="$OUTPUT_DIR/scalability_${TIMESTAMP}.dat"
RESULT_TXT="$OUTPUT_DIR/scalability_${TIMESTAMP}.txt"
TMPDIR="$(mktemp -d)"

mkdir -p "$OUTPUT_DIR"

trap 'rm -rf "$TMPDIR"' EXIT

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

# Generate a synthetic .ms program with N mutation sites.
# Each site is an if-branch with a relational comparison (ROR target).
generate_program() {
    local n="$1"
    local outfile="$2"

    {
        echo "// Auto-generated scalability test program with $n mutation sites"
        echo "// @mutate: ROR, AOR"
        echo "function scale_test(x: int, y: int) -> int {"
        echo "    var acc: int = 0;"

        for i in $(seq 1 "$n"); do
            local op
            case $(( i % 4 )) in
                0) op="<"  ;;
                1) op=">"  ;;
                2) op="<=" ;;
                3) op=">=" ;;
            esac
            echo "    if x + $i $op y + $i {"
            echo "        acc = acc + 1;"
            echo "    }"
        done

        echo "    return acc;"
        echo "}"
    } > "$outfile"
}

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
main() {
    echo "============================================"
    echo " MutSpec Scalability Test"
    echo " Sites: $STEP .. $MAX_SITES (step $STEP)"
    echo " Trials: $TRIALS"
    echo " Timestamp: $TIMESTAMP"
    echo "============================================"
    echo ""

    # Header for .dat file (tab-separated, gnuplot-friendly)
    printf "# sites\tavg_ms\tmin_ms\tmax_ms\tcontracts\tz3_queries\n" > "$RESULT_DAT"

    # Header for human-readable table
    {
        printf "%-10s %10s %10s %10s %12s %10s\n" \
            "Sites" "Avg(ms)" "Min(ms)" "Max(ms)" "Contracts" "Z3Calls"
        printf "%-10s %10s %10s %10s %12s %10s\n" \
            "----------" "----------" "----------" "----------" "------------" "----------"
    } > "$RESULT_TXT"

    for n in $(seq "$STEP" "$STEP" "$MAX_SITES"); do
        local src="$TMPDIR/scale_${n}.ms"
        generate_program "$n" "$src"

        local total_ms=0
        local min_ms=999999999
        local max_ms=0
        local contracts=0
        local z3_calls=0

        for trial in $(seq 1 "$TRIALS"); do
            local t0 t1 elapsed
            t0="$(now_ms)"

            local output
            output="$(cargo run --release --quiet -- synthesize "$src" 2>/dev/null)" || output="{}"

            t1="$(now_ms)"
            elapsed=$(( t1 - t0 ))
            total_ms=$(( total_ms + elapsed ))

            (( elapsed < min_ms )) && min_ms=$elapsed
            (( elapsed > max_ms )) && max_ms=$elapsed

            contracts="$(echo "$output" | jq -r '.contracts_count // 0' 2>/dev/null || echo 0)"
            z3_calls="$(echo "$output" | jq -r '.z3_queries // 0' 2>/dev/null || echo 0)"
        done

        local avg_ms=$(( total_ms / TRIALS ))

        printf "%d\t%d\t%d\t%d\t%d\t%d\n" \
            "$n" "$avg_ms" "$min_ms" "$max_ms" "$contracts" "$z3_calls" >> "$RESULT_DAT"

        printf "%-10d %10d %10d %10d %12d %10d\n" \
            "$n" "$avg_ms" "$min_ms" "$max_ms" "$contracts" "$z3_calls" >> "$RESULT_TXT"

        echo "  sites=$n  avg=${avg_ms}ms  contracts=$contracts"
    done

    echo ""
    cat "$RESULT_TXT"
    echo ""
    echo "Results written to:"
    echo "  Data: $RESULT_DAT  (gnuplot / pgfplots compatible)"
    echo "  Table: $RESULT_TXT"
}

main "$@"
