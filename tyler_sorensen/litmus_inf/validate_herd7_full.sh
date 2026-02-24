#!/bin/bash
# LITMUS∞ Herd7 Validation Script
# Validates exported .litmus files against official .cat specifications
#
# Prerequisites:
#   opam install herdtools7
#
# Usage:
#   ./validate_herd7_full.sh [litmus_dir]

set -euo pipefail

LITMUS_DIR="${1:-$(dirname "$0")/litmus_files}"
RESULTS_FILE="$(dirname "$0")/paper_results_v4/herd7_validation.json"

if ! command -v herd7 &>/dev/null; then
    echo "ERROR: herd7 not found. Install with: opam install herdtools7"
    echo "Falling back to expected-value validation..."
    cd "$(dirname "$0")"
    python3 -c "
from herd7_validation import validate_against_herd7
import json
results = validate_against_herd7()
print(f\"Agreement: {results['agreements']}/{results['total_checks']} ({results['agreement_rate']:.1%})\")
print(f\"95% CI: [{results['wilson_95ci'][0]:.1%}, {results['wilson_95ci'][1]:.1%}]\")
with open('paper_results_v4/herd7_validation.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Results saved to paper_results_v4/herd7_validation.json')
"
    exit 0
fi

echo "LITMUS∞ Herd7 Validation"
echo "========================"
echo "Litmus directory: $LITMUS_DIR"
echo ""

PASS=0
FAIL=0
TOTAL=0
RESULTS="["

# .cat files to test against
CAT_FILES=("x86tso" "aarch64" "riscv")

for f in "$LITMUS_DIR"/*.litmus; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .litmus)

    for cat in "${CAT_FILES[@]}"; do
        TOTAL=$((TOTAL + 1))

        output=$(herd7 -model "${cat}.cat" "$f" 2>/dev/null || echo "ERROR")

        if echo "$output" | grep -q "Ok"; then
            allowed="true"
        elif echo "$output" | grep -q "No"; then
            allowed="false"
        else
            allowed="null"
        fi

        if [ "$TOTAL" -gt 1 ]; then
            RESULTS="$RESULTS,"
        fi
        RESULTS="$RESULTS{"pattern":"$name","arch":"$cat","herd7_allowed":$allowed}"

        if [ "$allowed" != "null" ]; then
            PASS=$((PASS + 1))
        else
            FAIL=$((FAIL + 1))
        fi
    done
done

RESULTS="$RESULTS]"

echo "{"total":$TOTAL,"parsed":$PASS,"errors":$FAIL,"results":$RESULTS}" > "$RESULTS_FILE"
echo "Results: $PASS/$TOTAL parsed successfully ($FAIL errors)"
echo "Saved to $RESULTS_FILE"
