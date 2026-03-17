#!/usr/bin/env bash
# compare_tools.sh — Compare MutSpec against Daikon and SpecFuzzer baselines.
#
# Usage:
#   ./benchmarks/compare_tools.sh [--output DIR]
#
# Loads pre-computed baseline data from benchmarks/data/ and runs MutSpec
# on the benchmark suite, then computes precision, recall, and F1 for each tool.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
OUTPUT_DIR="$SCRIPT_DIR/results"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        *)        echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_JSON="$OUTPUT_DIR/comparison_${TIMESTAMP}.json"
RESULT_TXT="$OUTPUT_DIR/comparison_${TIMESTAMP}.txt"

mkdir -p "$OUTPUT_DIR"

# ------------------------------------------------------------------
# Prerequisites
# ------------------------------------------------------------------
for f in "$DATA_DIR/daikon_baseline.json" "$DATA_DIR/specfuzzer_baseline.json" "$DATA_DIR/benchmark_suite.json"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing data file: $f" >&2
        exit 1
    fi
done

if ! command -v jq &>/dev/null; then
    echo "ERROR: jq is required. Install: https://stedolan.github.io/jq/" >&2
    exit 1
fi

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

# Compute precision, recall, F1 from TP, FP, FN counts.
compute_metrics() {
    local tp="$1" fp="$2" fn="$3"
    local precision recall f1

    if (( tp + fp > 0 )); then
        precision="$(echo "scale=4; $tp / ($tp + $fp)" | bc)"
    else
        precision="0.0000"
    fi

    if (( tp + fn > 0 )); then
        recall="$(echo "scale=4; $tp / ($tp + $fn)" | bc)"
    else
        recall="0.0000"
    fi

    local p_plus_r
    p_plus_r="$(echo "$precision + $recall" | bc)"
    if (( $(echo "$p_plus_r > 0" | bc -l) )); then
        f1="$(echo "scale=4; 2 * $precision * $recall / $p_plus_r" | bc)"
    else
        f1="0.0000"
    fi

    echo "$precision $recall $f1"
}

# Extract TP/FP/FN from a tool's baseline JSON for a given function.
extract_counts() {
    local json_file="$1" func_name="$2"
    local tp fp fn
    tp="$(jq -r ".functions[] | select(.name == \"$func_name\") | .true_positives // 0" "$json_file")"
    fp="$(jq -r ".functions[] | select(.name == \"$func_name\") | .false_positives // 0" "$json_file")"
    fn="$(jq -r ".functions[] | select(.name == \"$func_name\") | .false_negatives // 0" "$json_file")"
    echo "$tp $fp $fn"
}

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
main() {
    echo "============================================"
    echo " MutSpec vs. Daikon vs. SpecFuzzer"
    echo " Timestamp: $TIMESTAMP"
    echo "============================================"
    echo ""

    local functions
    functions="$(jq -r '.programs[].functions[]' "$DATA_DIR/benchmark_suite.json")"

    local header
    header="$(printf "%-15s %-10s %8s %8s %8s" "Function" "Tool" "Prec" "Recall" "F1")"

    local table_lines=()
    local json_entries=()

    for func in $functions; do
        for tool in mutspec daikon specfuzzer; do
            local tp fp fn
            case "$tool" in
                mutspec)
                    # Run MutSpec and compare against ground truth
                    local ms_out
                    ms_out="$(cargo run --release --quiet -- evaluate "$func" 2>/dev/null)" || ms_out="{}"
                    tp="$(echo "$ms_out" | jq -r '.true_positives // 0' 2>/dev/null || echo 0)"
                    fp="$(echo "$ms_out" | jq -r '.false_positives // 0' 2>/dev/null || echo 0)"
                    fn="$(echo "$ms_out" | jq -r '.false_negatives // 0' 2>/dev/null || echo 0)"
                    ;;
                daikon)
                    read -r tp fp fn <<< "$(extract_counts "$DATA_DIR/daikon_baseline.json" "$func")"
                    ;;
                specfuzzer)
                    read -r tp fp fn <<< "$(extract_counts "$DATA_DIR/specfuzzer_baseline.json" "$func")"
                    ;;
            esac

            local metrics prec rec f1
            read -r prec rec f1 <<< "$(compute_metrics "$tp" "$fp" "$fn")"

            table_lines+=("$(printf "%-15s %-10s %8s %8s %8s" "$func" "$tool" "$prec" "$rec" "$f1")")
            json_entries+=("{\"function\":\"$func\",\"tool\":\"$tool\",\"precision\":$prec,\"recall\":$rec,\"f1\":$f1,\"tp\":$tp,\"fp\":$fp,\"fn\":$fn}")
        done
    done

    # Write JSON
    {
        echo "{\"timestamp\":\"$TIMESTAMP\",\"comparisons\":["
        local first=1
        for e in "${json_entries[@]}"; do
            [[ $first -eq 0 ]] && echo ","
            echo "  $e"
            first=0
        done
        echo "]}"
    } > "$RESULT_JSON"

    # Write table
    {
        echo "$header"
        printf "%-15s %-10s %8s %8s %8s\n" "---------------" "----------" "--------" "--------" "--------"
        for line in "${table_lines[@]}"; do
            echo "$line"
        done
    } > "$RESULT_TXT"

    cat "$RESULT_TXT"
    echo ""
    echo "Results written to:"
    echo "  JSON: $RESULT_JSON"
    echo "  Table: $RESULT_TXT"
}

main "$@"
